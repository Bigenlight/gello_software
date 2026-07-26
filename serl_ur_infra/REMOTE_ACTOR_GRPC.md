# HIL-SERL remote actor gRPC v2

This adapter keeps robot control and intervention on the laptop while policy
inference and replay routing live on the server. The current server entry point
is a loopback-only zero-action mock for contract testing; the RL server team can
inject its policy, reward finalizer, and replay router through
`ActorSessionService` callbacks without changing the transport API. The
receive-only implementation is documented in `RL_RECEIVE_SERVER.md`.

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
}
```

`create_actor_network(NETWORK, ...)` is the common factory. `agentlace` is a
reserved type and currently raises explicitly; there is no silent fallback.

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

Then start the actor in a second terminal with the experiment-specific config:

```bash
/tmp/gello-hil-grpc-venv/bin/python \
  serl_ur_infra/scripts/run_remote_rlpd_actor.py \
  --exp-name EXPERIMENT \
  --ur-config-module YOUR_CONFIG_MODULE \
  --server-host 127.0.0.1 --server-port 50052
```

For an SSH-hosted server, keep the gRPC service on server loopback and forward
it from the laptop. Use a port confirmed free on both machines; for example:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 kanu

/tmp/gello-hil-grpc-venv/bin/python \
  serl_ur_infra/scripts/run_actor_smoke_client.py \
  --host 127.0.0.1 --port 50053
```

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
