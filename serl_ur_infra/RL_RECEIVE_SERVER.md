# HIL-SERL receive server milestone

This milestone runs reward inference and real HIL-SERL RAM replay buffers on
Kanu. It deliberately does **not** run an RL learner, update policy parameters,
or control the real robot. The action reply is a safe fake `float32[7]` zero
action with `policy_version=0`.

## One-image-transfer step

At reset, **Local → Server** sends `BeginEpisode(O0)` and **Server → Local**
returns `A0`. At step `t`, **Local → Server** sends one gRPC request containing
the provisional `data{meta, transition}` plus `O(t+1)`. The server uses that
same `O(t+1)` for all three purposes:

1. reward-classifier inference;
2. the transition's `next_observations` in replay; and
3. fake policy inference for `A(t+1)` when the finalized transition is not
   terminal.

**Server → Local** replies with the transition ACK, server-authoritative
`TransitionOutcome`, and optionally the next action. Images are not sent a
second time for reward or replay insertion.

Classifier probability strictly greater than `0.85` finalizes the transition
as `reward=1`, `done=true`, `truncated=false`, and `mask=0`. One positive frame
ends the episode immediately. If classifier success and a local time-limit
truncation coincide, classifier success wins. Otherwise valid local
done/truncation semantics are preserved.

ACK means all required **RAM** routes succeeded: every transition entered the
replay buffer, and an intervention transition also entered the intervention
buffer. There is no disk journal in this milestone. Restarting the server loses
all accepted buffer contents.

Configured capacities count sampleable logical transitions. Internally the
upstream memory-efficient store receives two physical frame slots per logical
slot for this one-frame camera stack, covering the worst case where every
transition starts a new sequence. Status and overwrite counters still report
logical transitions, not bootstrap frame slots.

## Exact observation contract

Both laptop and server verify the same deterministic schema hash:

- `state`: `float32`, shape `(1, 19)` — TCP pose 6 (quaternion converted to
  Euler), TCP velocity 6, TCP force 3, TCP torque 3, gripper 1;
- `cam1`: RGB `uint8`, shape `(1, 128, 128, 3)`;
- `cam2`: RGB `uint8`, shape `(1, 128, 128, 3)`.

The environment timestamp stays in `data.meta.timestamp_ns`; it is not appended
to policy state. `GetServerInfo` advertises the observation schema hash and
reward model identity. `GetBufferStatus` returns only capacities, insert and
overwrite counters, and the latest transition ID/env step.

## Isolated Python 3.10 container

Initialize the pinned upstream submodule in this branch's isolated worktree:

```bash
git submodule update --init --recursive third_party/hil-serl
docker build \
  -f serl_ur_infra/docker/Dockerfile.rlpd-receive \
  -t gello-hil-rlpd-receive:v2 .
```

The lock pins JAX `0.4.35`, Flax and replay dependencies, gRPC `1.74.0`,
protobuf `3.20.3`, and NumPy `1.26.4`. Agentlace is installed only because the
upstream `MemoryEfficientReplayBufferDataStore` imports its `DataStoreBase`;
no Agentlace socket or port is used.

Before choosing a GPU, re-run `nvidia-smi`; do not assume an earlier free GPU
is still free. The following example uses GPU 7 and host networking so the
process can remain bound to Kanu loopback:

```bash
docker run \
  --name gello-hil-rlpd-receive-v2 \
  --network host \
  --gpus '"device=7"' \
  -v /home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150:/checkpoint/checkpoint_150:ro \
  -v /home/junhyeong/.serl:/root/.serl:ro \
  gello-hil-rlpd-receive:v2 \
  --host 127.0.0.1 \
  --port 50053 \
  --checkpoint /checkpoint/checkpoint_150 \
  --expected-checkpoint-sha256 e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997 \
  --threshold 0.85 \
  --replay-capacity 50000 \
  --intervention-capacity 10000
```

The server loads and hashes the explicit checkpoint file, compiles one warmup
inference before reporting ready, and fails closed on classifier or buffer
errors. Its logs contain IDs, counters, tensor contracts, and timings only;
they never print image or action values.

## Laptop-to-Kanu synthetic acceptance test

Keep port 50053 private behind SSH:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 kanu
```

Then, from this branch and the isolated gRPC client environment:

```bash
PYTHONPATH=serl_ur_infra \
  python serl_ur_infra/scripts/run_rlpd_receive_smoke_client.py \
  --host 127.0.0.1 --port 50053 --steps 100
```

The client tolerates classifier-authoritative early episode termination and
checks exact deltas of 100 replay inserts and 10 intervention inserts through
`GetBufferStatus`. The final synthetic step is locally truncated so no unused
server session remains active.

## Deferred work

Disk durability/recovery, an actual JAX policy and learner updates, human
reward relabeling, real UR task configuration and safety limits, camera-pair
timestamp/skew enforcement, and reset/fault recovery remain separate follow-up
milestones.
