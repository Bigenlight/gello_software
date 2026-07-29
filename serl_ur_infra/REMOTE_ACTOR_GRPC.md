# HIL-SERL remote actor gRPC v2

> Transport contract, re-verified against the merged tree on **2026-07-29**.
> Every flag, default, port and hash below was checked against the code it
> invokes.

This adapter keeps robot control and intervention on the laptop while policy
inference and replay routing live on the server. A server injects its policy,
reward finalizer, and replay router through `ActorSessionService` callbacks
without changing the transport API.

There are now **three** server entry points behind this same transport. Do not
confuse them — they differ in default port, in what they serve, and in whether
they are current:

| entry point | default port | serves | status |
| --- | --- | --- | --- |
| `scripts/run_actor_mock_server.py` | **50052** | zero actions, bounded in-memory sink | contract testing only |
| `scripts/run_rlpd_receive_server.py` | **50053** | real classifier reward + RAM replay, zero actions | historical milestone (`RL_RECEIVE_SERVER.md`) |
| `scripts/run_rlpd_learner_server.py` | **50053** | real classifier reward + RLPD learner + versioned policy | current (`HIL_SERL_KANU_RUNBOOK_KO.md`) |

`GrpcActorNetwork.from_config()` falls back to port **50052** when `NETWORK` does
not name one, while `run_remote_rlpd_actor.py` fills in **50053**. Always set the
port explicitly rather than relying on either default.

## Canonical observation contract

The actor and the server pin the same ordered schema hash and refuse to talk if
it differs:

```text
schema id:   hil-serl-ur-canonical-observation-v2
schema hash: 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903

state  float32 (1, 19)
cam1   uint8   (1, 128, 128, 3)
cam2   uint8   (1, 128, 128, 3)
```

The 19-D `state` is laid out in **alphabetical proprio-group order** — that is
what upstream `SERLObsWrapper` + `gym.spaces.Dict` actually emit, not the order
the keys are written in:

```text
[0]     gripper_pose   gripper_position          <- gripper is index 0, NOT -1
[1:4]   tcp_force      x, y, z
[4:10]  tcp_pose       position x,y,z + euler x,y,z
[10:13] tcp_torque     x, y, z
[13:19] tcp_vel        linear x,y,z + angular x,y,z
```

`state[0, -1]` is TCP angular velocity z. Read the gripper only through
`GRIPPER_POSITION_INDEX` / `gripper_position_from_state()`. Matching `(1, 19)`
alone does not make a v1 peer compatible with a v2 peer — the hash covers the
ordered feature names, so any reordering invalidates the handshake by design.

Never transcribe the hash by hand. Re-derive it (verified to print the value
above on 2026-07-29):

```bash
PYTHONPATH=serl_ur_infra python3 -c \
  "from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH as h; print(h)"
```

## Direction and one-observation-per-step flow

At reset:

1. **Local → Server** `BeginEpisode(O0)` sends the initial policy observation.
2. **Server → Local** returns policy action `A0`.

At environment step `t`:

1. Local executes `A(t)`. GELLO may replace it with a human action.
2. **Local → Server** `Step(O(t+1), D(t), request_action=true)` sends the next
   observation and the transition that references `O(t)` and `O(t+1)`.
3. Server finalizes reward/termination and inserts the transition into RAM
   replay routes.
4. **Server → Local** returns `ACK(D(t)) + TransitionOutcome + A(t+1)`.

For a locally terminal or truncated step, Local sends the same `Step` with
`request_action=false`. A server classifier may also finalize a provisional
non-terminal transition as terminal; in that case the server suppresses the
requested next action. Local uses the returned outcome and never resets before
the ACK is validated. Images are raw lossless numpy bytes with their nested key
path, dtype, and shape. `O(t)` is therefore not resent inside `D(t)`.

## Data envelope

Conceptually the server receives:

```python
data = {
    "meta": {
        "schema_version": 2,
        "run_id": "...",
        "actor_id": "...",
        "session_id": "...",
        "transition_id": "...",
        "env_step": 0,
        "timestamp_ns": 1721800000123456789,  # O(t) environment time
        "policy_version": 0,
        "policy_action": float32[7],
        "intervened": 0,                      # exactly 0 or 1
        "policy_actions_synthetic": False,    # True when --mock-policy-noise used
    },
    "transition": {
        "episode_id": 0,
        "step_id": 0,
        "observation_id": "...",             # O(t), server cache reference
        "actions": float32[7],                # physically executed action
        "next_observation_id": "...",        # O(t+1)
        "rewards": 0.0,
        "masks": 1.0,
        "dones": False,
        "truncated": False,
        # optional: "grasp_penalty": 0.0,
    },
}
```

Every accepted item is inserted into replay. An item with `intervened == 1` is
also inserted into the intervention buffer under the same IDs. The server
materializes `observations` and `next_observations` from its observation cache
before handing data to a replay implementation.

The mock's default sink retains at most 256 items per Python list for inspection
and rejects further Steps instead of silently dropping data or growing without
bound. A production server must supply `accept_data` to insert into its bounded
replay structures; it should return only after both required routes have
accepted the item and raise on failure so no ACK is emitted.

`meta.schema_version` must be exactly `2` (`SCHEMA_VERSION` in
`ur_env/actor_network.py`); the server rejects anything else rather than
negotiating.

`meta.policy_actions_synthetic` is stamped on **every** transition, not just in
the run summary, whenever the actor was started with `--mock-policy-noise`. Once
those transitions leave the process nothing else distinguishes mock noise from a
real policy action, so data carrying this flag must never be used as a demo or
as evidence of policy behaviour.

`timestamp_ns` is metadata, not a policy state feature, so observation shape
and checkpoint input contracts do not change. If the underlying environment
already supplies `info["timestamp_ns"]`, it is preserved. Otherwise the outer
`EnvTimestampAdapter` stamps the observation as the environment call returns.

## Network configuration

An experiment may define:

```python
NETWORK = {
    "type": "grpc",
    "host": "127.0.0.1",
    "port": 50053,
    "timeout_s": 0.6,
    "max_response_age_s": 0.8,
    "retry_count": 1,
    "observation_schema_hash": "<expected canonical SHA256>",
    # optional identity pins, all verified against GetServerInfo:
    "expected_model_id": "hil-serl-hybrid-sac-resnet10-trunk-cache-v1",
    "expected_reward_authority": "server_classifier",
    "expected_reward_model_id": "<the server's --reward-model-id>",
    "max_message_bytes": 16 * 1024 * 1024,
}
```

`create_actor_network(NETWORK, ...)` is the common factory. `agentlace` is a
reserved type and currently raises `NotImplementedError` explicitly; there is no
silent fallback.

Notes verified against `ur_env/grpc_actor_transport.py`:

- **`retry_count` must be exactly `1`.** It is not a tunable — any other value
  raises `protocol v2 requires exactly one transient retry`.
- `observation_schema_hash` and `expected_observation_schema_hash` are both
  accepted keys. Supplying both with different values raises rather than picking
  one.
- `expected_model_id` is `hil-serl-hybrid-sac-resnet10-trunk-cache-v1` for the
  production learner scope and
  `hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1` for the bounded
  synthetic acceptance scope (`ur_env/learner/config.py`). They are deliberately
  incompatible.
- `expected_reward_model_id` is **not** a code constant. It is whatever string
  the server was launched with (`--reward-model-id`), so both sides must be
  changed together — and must be changed whenever the classifier artifact
  changes.
- `expected_reward_authority` is `server_classifier` for both the receive server
  and the learner.
- `max_message_bytes` defaults to 16 MiB on both client and server.

`ur_experiments/cube_in_cup.py` ships `NETWORK` with `type/host/port/timeout_s/
max_response_age_s/retry_count` only; the schema hash and the identity pins come
from `run_remote_rlpd_actor.py` CLI flags, which override the config mapping.

## Which interpreter

The system `python3` on laptop3 carries ROS Humble's grpcio **1.30.2**, not the
`1.74.0` this contract locks. Do not run any of this with it.

| task | interpreter | why |
| --- | --- | --- |
| mock server, smoke clients, `run_fake_e2e_actor.py` | `/tmp/gello-hil-grpc-venv/bin/python` | grpcio 1.74.0 / numpy 1.26.4 / protobuf 3.20.3 — matches `requirements-grpc.lock` exactly (checked 2026-07-29) |
| `run_remote_rlpd_actor.py` against a real robot | `/home/laptop3/venvs/gello-hil-actor/bin/python` **with the ROS overlay sourced** | that script pulls in `rclpy`, `ur_gello_bringup`, `cv2`, `pyrealsense2` through `UR7eEnv`; this venv has grpcio 1.74.0 *and* system site-packages |

```bash
set +u
source /opt/ros/humble/setup.bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash
set -u
```

## Local contract test

Use an isolated environment so ROS Humble's system grpc/protobuf packages are
not replaced:

```bash
python3 -m venv /tmp/gello-hil-grpc-venv
/tmp/gello-hil-grpc-venv/bin/python -m pip install \
  -r serl_ur_infra/requirements-grpc.lock

PYTHONPATH=serl_ur_infra \
  /tmp/gello-hil-grpc-venv/bin/python \
  serl_ur_infra/scripts/run_actor_mock_server.py
```

For a robot/config-free check, run the standalone smoke client in another
terminal. It sends one normal transition followed by one terminal intervention
transition, using two raw 128x128 RGB images per observation:

```bash
/tmp/gello-hil-grpc-venv/bin/python \
  serl_ur_infra/scripts/run_actor_smoke_client.py \
  --host 127.0.0.1 --port 50052
```

Then start the actor in a second terminal with the experiment-specific config.
Without `--arm` the task config's `DRY_RUN` stays set, so neither the arm nor
the gripper moves — that is what you want against the mock:

```bash
PYTHONPATH=serl_ur_infra \
  /home/laptop3/venvs/gello-hil-actor/bin/python \
  serl_ur_infra/scripts/run_remote_rlpd_actor.py \
  --exp-name cube_in_cup \
  --ur-config-module ur_experiments.mappings \
  --server-host 127.0.0.1 --server-port 50052 \
  --fake-env
```

`--fake-env` skips the robot and cameras entirely and exercises only the
wrapper/network contract. Drop it once you want the real environment, and see
`HIL_SERL_KANU_RUNBOOK_KO.md` §8 before adding `--arm`.

> ⚠️ `run_remote_rlpd_actor.py` has **never been run on the real rig**
> (2026-07-29). Everything that has moved the UR7e so far went through
> `serl_ur_infra/tests/run_real_hil.py`, which is a different code path.

For an SSH-hosted server, keep the gRPC service on server loopback and forward
it from the laptop. The remote side is always the server's `--port`; the local
entry port only has to be free on the laptop:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 kanu

/tmp/gello-hil-grpc-venv/bin/python \
  serl_ur_infra/scripts/run_actor_smoke_client.py \
  --host 127.0.0.1 --port 50053
```

> The laptop3 operating convention is local **`50153`** → remote `50053`
> (`ros2_ur_ws/run_hil_actor.sh` defaults `SERVER_PORT=50153`, because local
> 50053 was already taken on 2026-07-27). Whichever you use, the local entry
> port and the client's `--port` / actor's `--server-port` must agree.

The server's mock stdout records only IDs, counters, timestamps, intervention
labels, tensor dtype/shape, and cumulative routing counts. It does not persist
or print image/action values.

## Failure rules

- Action identity, shape `(7,)`, finite values, strict `[-1, 1]`, freshness,
  and nondecreasing `policy_version` are checked on Local.
- A transient timeout or unavailable response retries the exact serialized
  request once. Server reply caching prevents duplicate inference and replay
  insertion.
- A failed Step keeps its transition pending and prevents reset/new actions.
- Classifier or replay insertion failures emit no ACK and make the server
  not-ready; Local fails stopped.
- Application or malformed-action failures abort the active episode; no
  fallback/random action is executed. Protocol v2 requires `random_steps == 0`.
