# HIL-SERL receive server milestone

> Historical receive-only milestone. The current production learner keeps the
> external raw observation schema, but stores explicit current/next frozen
> ResNet-10 trunk maps (`float32 (1,4,4,512)` per camera), uses no pixel
> augmentation, and serves model ID
> `hil-serl-hybrid-sac-resnet10-trunk-cache-v1`. See
> [HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md](./HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md)
> and [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md). Commands
> below describe the older receive-only server and are not the production
> learner launch command. The current bounded fake-data learning acceptance
> uses `run_rlpd_learner_server.py --synthetic-e2e` together with
> `run_fake_e2e_actor.py`; it performs real feature replay, CTA updates, policy
> publication, checkpointing, and fresh-process resume. Do not confuse that
> workflow with the receive-only synthetic smoke described below.

한국어 구현·검증·인수인계 요약은
[HIL_RLPD_RECEIVE_SERVER_KO.md](./HIL_RLPD_RECEIVE_SERVER_KO.md)를 참고한다.

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

Classifier probability strictly greater than `0.5` finalizes the transition
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

The hash covers not only dtype and shape, but the exact ordered state feature
names: TCP position XYZ, TCP Euler XYZ, TCP linear velocity XYZ, TCP angular
velocity XYZ, TCP force XYZ, TCP torque XYZ, then gripper position. Reordering
state values therefore changes the advertised contract instead of silently
sharing the old hash.

The environment timestamp stays in `data.meta.timestamp_ns`; it is not appended
to policy state. `GetServerInfo` advertises the observation schema hash and
reward model identity. `GetBufferStatus` returns only capacities, insert and
overwrite counters, and the latest transition ID/env step.

## Preferred Kanu runtime: small overlay venv

Kanu's existing `il` environment already provides Python 3.10, CUDA JAX,
Flax, NumPy, gRPC, and the classifier dependencies. Do not install into or
upgrade that shared environment. Instead, create a small venv which reads its
packages and owns only the protocol/runtime compatibility files missing from
that environment:

```bash
/home/junhyeong/miniconda3/envs/il/bin/python -m venv \
  --system-site-packages /tmp/gello-hil-rl-receive-overlay-v2
/tmp/gello-hil-rl-receive-overlay-v2/bin/python -m pip install \
  --no-deps \
  -r serl_ur_infra/requirements-rlpd-receive-overlay.txt
```

`--no-deps` is intentional: it prevents pip from replacing packages inherited
from `il`. The overlay adds Agentlace (needed as the upstream replay store's
base class), LZ4 (imported by Agentlace), and protobuf `3.20.3` (required by the
checked-in generated gRPC module). This avoids duplicating the multi-gigabyte
CUDA/Python environment or creating a Docker image.

Run the historical receive-only server from the canonical checkout:

> ⚠️ **The classifier checkpoint below (`e329986b...`) is retired.** A 2026-07-28
> measurement on Kanu found `0.0%` success recall on the 0724 domain (0 of 1,123
> success frames above threshold; mean probability `0.007`). Running a robot
> against it yields `reward=0` forever. The command is kept verbatim as the
> historical receive-only milestone record — do not reuse the path or SHA for a
> new run. The current canonical checkpoint and its orbax-directory constraints
> are in [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md) §1.3;
> the measurement is in
> [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md).
>
> Also note `DEFAULT_CHECKPOINT_SHA256` in
> `scripts/run_rlpd_receive_server.py` still defaults to that retired SHA as of
> 2026-07-29, so never omit `--expected-checkpoint-sha256`.

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=serl_ur_infra:third_party/hil-serl/serl_launcher \
/tmp/gello-hil-rl-receive-overlay-v2/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_server.py \
  --host 127.0.0.1 \
  --port 50053 \
  --checkpoint /home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150 \
  --expected-checkpoint-sha256 e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997 \
  --threshold 0.5 \
  --replay-capacity 50000 \
  --intervention-capacity 10000 \
  --require-jax-backend gpu
```

Before choosing a GPU, re-run `nvidia-smi`; do not assume an earlier free GPU
is still free.

## Optional reproducible container

Initialize the pinned upstream submodule in the canonical checkout:

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

The following optional example uses GPU 7 and host networking so the process
can remain bound to Kanu loopback:

> ⚠️ **Same retired classifier as above.** The mounted checkpoint path and
> `--expected-checkpoint-sha256 e329986b...` are the retired artifact with
> `0.0%` recall on the 0724 domain. Kept as a historical record only; substitute
> the current canonical checkpoint before any real run, and note that it is an
> orbax *directory*, which this server's `checkpoint_sha256()` cannot hash
> (`os.path.isfile()` is enforced).

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
  --threshold 0.5 \
  --replay-capacity 50000 \
  --intervention-capacity 10000 \
  --require-jax-backend gpu
```

The server loads and hashes the explicit checkpoint file, compiles one warmup
inference before reporting ready, verifies that JAX actually selected the GPU,
and fails closed on classifier or buffer errors. Its logs contain IDs,
counters, tensor contracts, and timings only; they never print image or action
values.

## Laptop-to-Kanu synthetic acceptance test

Keep port 50053 private behind SSH:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 kanu
```

Then, from the canonical checkout and the isolated gRPC client environment:

```bash
PYTHONPATH=serl_ur_infra \
  python serl_ur_infra/scripts/run_rlpd_receive_smoke_client.py \
  --host 127.0.0.1 --port 50053 --steps 100
```

The client tolerates classifier-authoritative early episode termination and
checks exact deltas of 100 replay inserts and 10 intervention inserts through
`GetBufferStatus`. The final synthetic step is locally truncated so no unused
server session remains active. Before acknowledging the 100th insert, the
server also samples and validates real packed replay and intervention batches;
the summary-only log event is `rlpd_receive_sample_probe_passed`.

## Deferred work

Disk durability/recovery, an actual JAX policy and learner updates, human
reward relabeling, real UR task configuration and safety limits, camera-pair
timestamp/skew enforcement, and reset/fault recovery remain separate follow-up
milestones.
