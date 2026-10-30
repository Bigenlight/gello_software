# HIL-SERL remote actor gRPC v2

> Transport contract, re-verified against the merged tree on **2026-07-29**.
> Every flag, default, port and hash below was checked against the code it
> invokes.

> ## 🔴 2026-07-31: the server host moved. The transport did not.
>
> The learner moved **`kanu` → `junhyeong_ai`** (166.104.146.29, hostname
> `junhyeong`, user `junhyeong`). Nothing in this contract changed shape — same
> proto, same `schema_version = 2`, same schema hash, same remote port. What
> changed is *which machine is on the far end of the tunnel*, and *which laptop3
> interpreter you use to reach it* (see [Which interpreter](#which-interpreter)).
>
> | | before | **now** |
> | --- | --- | --- |
> | ssh alias | `kanu` | **`junhyeong_ai`** |
> | tunnel | laptop3 `127.0.0.1:50153` → kanu `127.0.0.1:50053` | laptop3 `127.0.0.1:50153` → **`junhyeong_ai`** `127.0.0.1:50053` |
> | remote port | 50053 | 50053 (unchanged, and `run_hil_server.sh` now *fixes* it there) |
> | laptop3 gRPC interpreter | `/tmp/gello-hil-grpc-venv/bin/python` for mocks/smokes | **`/home/laptop3/venvs/gello-hil-actor/bin/python` for everything** — the `/tmp` venv is gone |
> | learner env vars | `HIL_KANU_REPO` / `HIL_KANU_PYTHON` | `HIL_REMOTE_REPO` / `HIL_REMOTE_PYTHON` (old names still work as aliases), plus new `HIL_REMOTE_DATA_ROOT` |
>
> Measured latencies over the new link are in
> [`SERVER_MIGRATION_E2E_JUNHYEONG_AI.md`](./SERVER_MIGRATION_E2E_JUNHYEONG_AI.md).
> Latency figures elsewhere in the repo that predate 2026-07-31 are **kanu's**,
> and are kept as kanu's — they are the only baseline the new host has to be
> compared against.

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

> 🗄️ `HIL_SERL_KANU_RUNBOOK_KO.md` still carries `KANU` in its **filename** and
> most of its body. The learner it describes now runs on `junhyeong_ai`; for
> host, GPU index, repo path and data root, `DATA_AND_MODELS_JUNHYEONG_AI_KO.md`
> wins over that runbook. In normal operation you do not launch the learner by
> hand at all — `ros2_ur_ws/run_hil_server.sh` does it with **zero env
> overrides** (and it can no longer drive kanu, on purpose: kanu's classifier
> sits outside `hil-serl-data`, so no single `HIL_REMOTE_DATA_ROOT` describes
> it). kanu is now inspected read-only.

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

## Classifier sidecar (non-canonical)

Since 2026-07-29 an observation may carry one extra **nested** key, `classifier`
(`ur_env.classifier_sidecar.CLASSIFIER_SIDECAR_KEY`), holding the reward
classifier's own view of the scene:

```text
classifier/cam1_jpeg   uint8  (N,)    one complete JPEG file
classifier/cam2_jpeg   uint8  (M,)    one complete JPEG file
```

**The schema hash does not change, and no proto change was needed.** Three
independent reasons, each worth stating because each has been misread before:

1. `Tensor{path,dtype,shape,data}` + `Observation{repeated Tensor tensors}` is a
   *generic named-tensor map*. A new key is data, not schema.
2. The handshake hash derives from `CANONICAL_OBSERVATION_SPEC` — from the
   schema **document**, never from the wire payload.
3. The server splits the sidecar off **before** canonical validation
   (`actor_network._split_classifier_sidecar`). It has to:
   `validate_canonical_observation()` rejects unknown keys outright.

So every existing pin of `3459098d…0352903` stays valid, and an old server
simply ignores the extra tensors.

**What the bytes are.** The uncropped full field of view, resized on the laptop
to 128x128 with the same deterministic `cv2.resize` the server would have run,
then JPEG-encoded at quality 95 — **13.32 KiB per pair, measured** over 100
frames per camera including protobuf framing. The policy's `IMAGE_CROP` is
deliberately *absent*: this checkpoint was trained on uncropped frames, and
feeding it the cropped policy observation cost recall@0.85 100% -> 33.3%
(`docs/testing/08_OPEN_GAPS.md` G15).

Forwarding the camera's original 720p JPEG was the first design and was
**rejected on measurement**: at the driver's `jpeg_quality=95` a live pair is
400 KiB, which exceeds a 13 Mbit/s link outright and spikes the step by +252 ms
against a 100 ms budget.

**Cadence, and why most transitions carry no sidecar.** The actor attaches one
roughly every 5 steps (~2 Hz at HZ=10) and **only while the arm is stationary**
(TCP linear speed <= 0.05 m/s); a step about to terminate always attaches. If the
last classifier probability was >= 0.05 the scheduler escalates to every step —
that bound is deliberately far *below* the reward threshold, so the step that
actually crosses it is never missed.

> 📌 The escalation bound is `escalate_probability = 0.05`
> (`classifier_sidecar.py`, verified 2026-07-31). An earlier revision of this
> paragraph called the reward threshold `0.2`; the runtime default is
> **`0.5`** (`DEFAULT_REWARD_THRESHOLD`, strict `p > 0.5`), and `run_hil_server.sh`
> pins 0.5 as well. The escalation logic is unaffected — 0.05 sits below both —
> but do not quote `0.2` as the current threshold. The 0.85 → 0.5 → 0.2 → 0.5
> history is in `REWARD_CLASSIFIER_THRESHOLD_KO.md`; the code constant is the
> authority.

`BeginEpisode` **must not** carry a sidecar and the server rejects one as a
protocol error: `O0` is no transition's `next_observations`, so a verdict on it
could not be attached to any reward.

Transitions that arrive without a sidecar are **left unclassified**: `rewards`
forced to `0.0`, `classifier_evaluated = 0`, and `classifier_probability` /
`classifier_threshold` / `reward_model_id` required to be `0.0` / `0.0` / `""`.
They are ordinary zero-reward, non-terminal samples. The one authority withheld
from them is ending an episode with a positive reward.

**`reward_model_id` now encodes the input contract**, not just the checkpoint:
`cube-in-cup-all3-ckpt150+sidecar-v1`. A pre-sidecar actor talking to a
post-sidecar server (or the reverse) would compute reward from different pixels
than its peer believes, so the pair is rejected at the handshake rather than
running a whole session on wrong rewards.

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
the ACK is validated. **Policy** images are raw lossless numpy bytes with their
nested key path, dtype, and shape. `O(t)` is therefore not resent inside `D(t)`.

> **Qualification (2026-07-29).** "Raw lossless numpy bytes" describes the
> canonical policy observation and **still does**. It does *not* describe the
> reward-classifier sidecar added on 2026-07-29, which travels in the same
> tensor map but carries **JPEG-encoded** bytes (quality 95) rather than raw
> pixels. See [Classifier sidecar](#classifier-sidecar-non-canonical) below.
> The two are deliberately different: the policy observation must round-trip
> bit-for-bit, while the classifier's copy is sized to survive a link whose
> throughput varies ~6x between sessions.
>
> 📌 That ~6x figure is **kanu-era** (13 vs 47.8 vs 83 Mbit/s across three
> laptop3→kanu sessions, `docs/testing/05_COMMS_GRPC.md` §5.3). It is kept
> because the constraint it describes is laptop3's shared 2.4 GHz WiFi, which the
> server move did not touch — ICMP RTT is the same to `junhyeong_ai` as it was to
> kanu. The sizing decision therefore still stands; the number is still kanu's.

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

> ⚠️ **The `0.6 / 0.8` above is the config-file value, not the production
> budget.** `run_hil_actor.sh` passes `--timeout-s 1.5 --max-response-age-s 2.0`
> on the command line, and CLI beats the config mapping — so a real session runs
> at **1.5 / 2.0**, not 0.6 / 0.8. The widening was forced by measurement: on the
> first real E2E a perfectly valid reply arrived at **832.3 ms** and the old
> 0.6/0.8 pair rejected it as stale. Do not "restore" 0.6/0.8 here thinking you
> are tightening something; you would only make the two sources disagree.

### What one RPC actually costs on this link

Measured on the **`junhyeong_ai`** link, 2026-07-31, over 100 transitions of the
synthetic acceptance run — full record and method in
[`SERVER_MIGRATION_E2E_JUNHYEONG_AI.md`](./SERVER_MIGRATION_E2E_JUNHYEONG_AI.md) §3:

| RPC | n | mean | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `BeginEpisode` | 100 | **57.7 ms** | 70.3 | **82.2 ms** |
| `Step` | 100 | **156.1 ms** | 179.6 | **211.0 ms** |
| `GetServerInfo` | 100 | 3.6 ms | 8.2 | 20.7 ms |
| sum per transition | 100 | **213.8 ms** | 237.9 | 268.7 ms |

Against the **kanu-era** baseline, which stays labelled as kanu's because that is
what it measures — `BeginEpisode` **84.9 ms mean / 372.8 ms max** (schema v2
sender) — the new host is modestly faster in the mean and **~4.5× shorter in the
tail**. ICMP RTT is effectively identical to both hosts (junhyeong_ai
1.078/2.811/6.979 ms vs kanu 1.272/2.518/6.314 ms, 10 packets each), so **the
gain is host compute and serialization, not the network.** Moving the server did
not change the link.

> ### 🛑 Do not compare 213.8 ms to the 512 ms production loop period
>
> Other documents in this repo record the production actor loop at **512 ms mean
> / 854 ms max (1.95 Hz)**. Subtracting one from the other, or reporting "the
> loop got 2.4× faster", is a real error — **they are not the same quantity**:
>
> - **213.8 ms** is the sum of **two gRPC calls**, measured with no robot, no
>   cameras, and no classifier sidecar attached.
> - **512 ms** is the **entire robot loop**: `env.step`'s own 100 ms self-pacing,
>   camera decode, observation assembly, *and* the blocking `Step` RPC. Most of
>   the non-RPC cost lives outside `env.step` entirely.
>
> The RPC term is a **component** of the loop period, not a competing measurement
> of it. Two further reasons the numbers cannot be lined up: the 512 ms figure is
> kanu-era, and the 156 ms `Step` above **excludes classifier inference** — the
> acceptance tool attaches no sidecar, so the ~2 Hz of real steps that do carry
> one must be slower than this.
>
> The honest conclusion is the narrow one: **on the RPC segment alone the new
> host is not slower than kanu and its tail is clearly shorter.** How much the
> real loop period improves is **unverified** and has to be re-measured on the
> rig, together with G21.

## Which interpreter

> ### 🛑 Never run gRPC code with the system `python3`. This is a hard project rule.
>
> The system `python3` on laptop3 carries ROS Humble's grpcio **1.30.2**, not the
> `1.74.0` this contract locks, and that build is **broken on this machine**:
> merely constructing a channel **spins at 100% CPU forever, with no error and no
> log**. It does not time out. It does not raise. It does not come back after 30
> minutes. There is no partial-success mode to fall back on — you just lose the
> session and your time.
>
> ```bash
> python3 -c "import grpc; print(grpc.__version__)"   # -> 1.30.2   FORBIDDEN
> /home/laptop3/venvs/gello-hil-actor/bin/python -c "import grpc; print(grpc.__version__)"
>                                                     # -> 1.74.0   the only one to use
> ```
>
> Do **not** try to fix it by upgrading the system packages: replacing ROS
> Humble's grpc/protobuf kills `rclpy` and takes the whole robot laptop with it.
> The venv exists precisely so that nothing has to be fixed.

**One laptop3 interpreter for every gRPC path, mock or real:**

| task | interpreter | why |
| --- | --- | --- |
| mock server, smoke clients, `run_fake_e2e_actor.py` | `/home/laptop3/venvs/gello-hil-actor/bin/python` | grpcio **1.74.0** and protobuf **3.20.3**, an exact match for `requirements-grpc.lock`; numpy differs, see the caveat below (re-measured 2026-07-31) |
| `run_remote_rlpd_actor.py` against a real robot | the same venv, **with the ROS overlay sourced** | that script pulls in `rclpy`, `ur_gello_bringup`, `cv2`, `pyrealsense2` through `UR7eEnv`; this venv is `--system-site-packages`, so it has grpcio 1.74.0 *and* the ROS stack |

```bash
set +u
source /opt/ros/humble/setup.bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash
set -u
```

> ### 🔧 Correction (2026-07-31): `/tmp/gello-hil-grpc-venv` **no longer exists**
>
> Every earlier revision of the table above sent you to
> `/tmp/gello-hil-grpc-venv/bin/python` for the mock server, the smoke clients
> and `run_fake_e2e_actor.py`. **laptop3 rebooted and `/tmp` was cleared with
> it.** The venv is gone. It is not misplaced, it was never in git, and there is
> nothing to recover — so do not spend an afternoon hunting for it, and do not
> recreate it under `/tmp`, where the next reboot deletes it again.
>
> **What was used instead, and what that proved.** The 2026-07-31
> `junhyeong_ai` acceptance run ran `run_fake_e2e_actor.py` on
> `/home/laptop3/venvs/gello-hil-actor/bin/python` and it worked: **200
> transitions (100 × 2 runs) serialized, sent, ACKed and validated clean**, both
> runs reporting `fake_e2e_actor_passed` with `replay_insert_delta = 100`
> ([`SERVER_MIGRATION_E2E_JUNHYEONG_AI.md`](./SERVER_MIGRATION_E2E_JUNHYEONG_AI.md)).
> That is a measurement, not an assumption.
>
> Strictly, that run exercised **one** of the three tools in the row above. The
> mock server and the smoke clients are covered by the same reasoning rather than
> by the same measurement: they drive the identical `grpc_actor_transport` stack,
> so there was never a reason to split interpreters — and the interpreter they
> would have been split onto does not exist. The actor venv is now the only
> laptop3 interpreter this document names.
>
> **The one difference from the lock, recorded rather than papered over.**
> Measured in that venv on 2026-07-31:
>
> | package | `requirements-grpc.lock` | actor venv | verdict |
> | --- | --- | --- | --- |
> | `grpcio` | 1.74.0 | **1.74.0** | exact match |
> | `protobuf` | 3.20.3 | **3.20.3** | exact match |
> | `numpy` | 1.26.4 | **2.2.6** | **differs — measured not to matter** |
>
> The two packages that actually decide wire behaviour are exact. numpy is the
> outlier, and the acceptance run is the evidence that the gap is inert on this
> path: 200 transitions of `float32` state, `uint8` images and JPEG sidecar
> buffers round-tripped through `Tensor{path,dtype,shape,data}` with every
> dtype/shape assertion and every canonical-observation validation passing. The
> transport pins dtypes explicitly rather than inheriting whatever numpy
> defaults to, which is why the major-version bump does not reach the wire.
>
> ⚠️ **This is a scoped result, not a blanket clearance.** It says the *gRPC
> transport* is indifferent to numpy 2.x. It says nothing about numpy 2.x
> elsewhere, and the learner side is a separate, exact-pinned, fail-closed
> environment (`ur_env/learner/agent.py::validate_learner_dependencies`) that
> this run did not touch. If you need a lock-exact isolated venv for some other
> reason, build it **outside `/tmp`**:
>
> ```bash
> python3 -m venv ~/venvs/gello-hil-grpc-lock          # NOT under /tmp
> ~/venvs/gello-hil-grpc-lock/bin/python -m pip install \
>   -r serl_ur_infra/requirements-grpc.lock
> ```

## Local contract test

Everything below runs on the actor venv — never the system `python3`. Export it
once so the commands stay short:

```bash
export ACTOR_PY=/home/laptop3/venvs/gello-hil-actor/bin/python
$ACTOR_PY -c "import grpc; print(grpc.__version__)"     # must print 1.74.0

PYTHONPATH=serl_ur_infra \
  $ACTOR_PY serl_ur_infra/scripts/run_actor_mock_server.py
```

For a robot/config-free check, run the standalone smoke client in another
terminal. It sends one normal transition followed by one terminal intervention
transition, using two raw 128x128 RGB images per observation:

```bash
$ACTOR_PY serl_ur_infra/scripts/run_actor_smoke_client.py \
  --host 127.0.0.1 --port 50052
```

Then start the actor in a second terminal with the experiment-specific config.
Without `--arm` the task config's `DRY_RUN` stays set, so neither the arm nor
the gripper moves — that is what you want against the mock:

```bash
PYTHONPATH=serl_ur_infra \
  $ACTOR_PY serl_ur_infra/scripts/run_remote_rlpd_actor.py \
  --exp-name cube_in_cup \
  --ur-config-module ur_experiments.mappings \
  --server-host 127.0.0.1 --server-port 50052 \
  --fake-env
```

`--fake-env` skips the robot and cameras entirely and exercises only the
wrapper/network contract. Drop it once you want the real environment. For a real
session do not hand-run this script at all — `ros2_ur_ws/run_hil_actor.sh` wraps
it with an 11-step preflight and pins the interpreter for you.

> ### 🔧 Superseded (2026-07-31)
> > **Previous text (preserved):** *"⚠️ `run_remote_rlpd_actor.py` has **never
> > been run on the real rig** (2026-07-29). Everything that has moved the UR7e
> > so far went through `serl_ur_infra/tests/run_real_hil.py`, which is a
> > different code path."*
>
> That stopped being true the same day it was written. The first real
> production-model E2E smoke ran on the actual UR7e on **2026-07-29**, and a full
> operator session against `junhyeong_ai` **passed on 2026-07-31** (replay 316 /
> intervention 210 / `last_env_step` 68). Both went through
> `run_hil_actor.sh` → `run_remote_rlpd_actor.py`. `tests/run_real_hil.py` is
> still a genuinely different code path — it needs no learner and no gRPC server
> at all — so do not confuse the two, but it is no longer the *only* thing that
> has driven the arm.

For an SSH-hosted server, keep the gRPC service on server loopback and forward
it from the laptop. The remote side is always the server's `--port`; the local
entry port only has to be free on the laptop:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50153:127.0.0.1:50053 junhyeong_ai

$ACTOR_PY serl_ur_infra/scripts/run_actor_smoke_client.py \
  --host 127.0.0.1 --port 50153
```

> The laptop3 operating convention is local **`50153`** → remote `50053`
> (`ros2_ur_ws/run_hil_actor.sh` defaults `SERVER_PORT=50153`, because local
> 50053 was already taken on 2026-07-27). Whichever you use, the local entry
> port and the client's `--port` / actor's `--server-port` must agree.
>
> 📌 In normal operation you do not open this tunnel by hand: `run_hil_server.sh`
> owns it (`HIL_LOCAL_PORT` default `50153`, `HIL_REMOTE_PORT` **fixed** at
> `50053` — it refuses any other remote port for a production learner) and tears
> it down with the session. The hand-rolled form above is for smoke tests and for
> the bounded synthetic acceptance run, which is launched manually and therefore
> used local `50053` on 2026-07-31 rather than the production `50153`. Only one
> of the two can hold the remote port at a time, so an acceptance run and a real
> session cannot overlap.

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
