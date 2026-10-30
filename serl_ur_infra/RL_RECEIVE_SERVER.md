# HIL-SERL receive server milestone

> **Status: historical record, re-verified against the merged tree on
> 2026-07-29.** The commands below still match the current `argparse` surface of
> `scripts/run_rlpd_receive_server.py`, but three things moved underneath the
> original text and are corrected inline: the reward threshold (now read from
> code, not from this page), the 19-D `state` ordering (v1 ordering was wrong;
> canonical is v2), and the classifier checkpoint (the one quoted here is
> retired). Nothing on Kanu is running this server as of 2026-07-29 — port
> 50053 is unbound and the GPUs are idle.

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

Classifier probability **strictly greater than the configured threshold**
finalizes the transition as `reward=1`, `done=true`, `truncated=false`, and
`mask=0` (`ClassificationResult.success = probability > self.threshold`). One
positive frame ends the episode immediately. If classifier success and a local
time-limit truncation coincide, classifier success wins. Otherwise valid local
done/truncation semantics are preserved.

> ⚠️ **Do not copy a threshold number out of this page.** It moved twice in two
> days (0.85 → 0.5 → 0.2). The single authoritative value is the constant
> `DEFAULT_REWARD_THRESHOLD` in `ur_env/rlpd_receive_server.py`, which is also
> the `--threshold` default. Read it at run time:
>
> ```bash
> PYTHONPATH=serl_ur_infra python3 -c \
>   "from ur_env.rlpd_receive_server import DEFAULT_REWARD_THRESHOLD as t; print(t)"
> ```
>
> It was `0.2` when checked on 2026-07-29. The rationale and the conditions
> attached to it live in
> [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md).

> 🔴 **Open defect, not fixable on this side.** `_classifier_observation()`
> performs **zero** image transformation — it passes `cam1`/`cam2` through
> untouched. But the actor's `ur7e_env.get_im()` applies `IMAGE_CROP` before
> building the canonical observation, while the classifier was trained on
> uncropped full frames. So the reward numbers quoted anywhere for this
> classifier were measured on input this server never receives. Do not "fix" it
> here by transforming images in the server; the resolution is a classifier
> retrain under the crop. Analysis in
> [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md).

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

```text
schema id:   hil-serl-ur-canonical-observation-v2
schema hash: 3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903
```

- `state`: `float32`, shape `(1, 19)`;
- `cam1`: RGB `uint8`, shape `(1, 128, 128, 3)`;
- `cam2`: RGB `uint8`, shape `(1, 128, 128, 3)`.

> 🪤 **Corrected 2026-07-29.** An earlier version of this page described the
> state as "TCP pose 6, TCP velocity 6, TCP force 3, TCP torque 3, gripper 1".
> **That is the v1 ordering and it is wrong for the current contract.** The flat
> layout is produced by upstream `SERLObsWrapper`, and `gym.spaces.Dict`
> re-sorts the proprio groups **alphabetically**, so the real order is:

```text
[0]     gripper_pose   gripper_position          <- gripper is index 0, NOT -1
[1:4]   tcp_force      x, y, z
[4:10]  tcp_pose       position x,y,z + euler x,y,z
[10:13] tcp_torque     x, y, z
[13:19] tcp_vel        linear x,y,z + angular x,y,z
```

`state[0, -1]` is TCP angular velocity z, not the gripper. Read the gripper only
through `GRIPPER_POSITION_INDEX` / `gripper_position_from_state()`.

The hash covers dtype, shape **and** the exact ordered state feature names, so
reordering state values changes the advertised contract instead of silently
sharing the old hash. Matching `(1, 19)` alone does not make a v1 peer
compatible with a v2 peer.

Never hand-copy the hash. Re-derive it (verified to produce the value above on
2026-07-29):

```bash
PYTHONPATH=serl_ur_infra python3 -c \
  "from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH as h; print(h)"
```

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

> ⚠️ **This overlay is for the receive-only server only.** Do not reuse
> `/tmp/gello-hil-rl-receive-overlay-v2` for the production learner: its
> protobuf `3.20.3` pin breaks the `wandb` import that
> `run_rlpd_learner_server.py` requires.

### Where the working directory actually is

The commands below use paths relative to a `gello_software` checkout. On
**laptop3** that is `/home/laptop3/gello_software`. **That path does not exist
on Kanu.** Kanu's known directories are the classifier training tree
(`~/workspace/youngwoong/hil-serl`), the dataset/checkpoint tree
(`~/workspace/youngwoong/dataset/cube_in_cup_all3/`), the ZMQ-viewer checkout
(`~/workspace/youngwoong/gello_software_remote_classifier`), and the
receive-server worktree `/tmp/gello-hil-rl-receive-server-v2` — which lives in
`/tmp` and may be gone. `cd` to a checkout you have confirmed exists before
running anything here.

### Running the historical receive-only server

> ⚠️ **The classifier checkpoint below (`e329986b...`) is retired.** A 2026-07-28
> measurement on Kanu found `0.0%` success recall on the 0724 domain (0 of 1,123
> success frames above threshold; mean probability `0.007`). Running a robot
> against it yields `reward=0` forever. The command is kept verbatim as the
> historical receive-only milestone record — do not reuse the path or SHA for a
> new run. The current canonical checkpoint and its orbax-directory constraints
> are in [HIL_SERL_KANU_RUNBOOK_KO.md](./HIL_SERL_KANU_RUNBOOK_KO.md) §1.3; the
> measurement is in
> [REWARD_CLASSIFIER_THRESHOLD_KO.md](./REWARD_CLASSIFIER_THRESHOLD_KO.md).
>
> **`DEFAULT_CHECKPOINT_SHA256` in `scripts/run_rlpd_receive_server.py` is still
> that retired SHA** (code checked 2026-07-29), and it is the default value of
> `--expected-checkpoint-sha256`. So omitting the flag while pointing
> `--checkpoint` at the retired file passes silently and starts a run whose
> reward can never fire. Always pass the flag explicitly and confirm the value
> belongs to the artifact you actually intend to use.
>
> **The replacement checkpoint cannot be substituted into this command yet.** It
> is an orbax *directory*, and this server's `checkpoint_sha256()` enforces
> `os.path.isfile()` before the loader ever runs, so the CLI exits with
> `FileNotFoundError` at the SHA step. A directory digest contract has to land
> first; see the runbook §1.3.1.

```bash
# Pick the GPU immediately before launching. There is no standing reservation.
export HIL_GPU_INDEX=GPU_NUMBER_SELECTED_AFTER_NVIDIA_SMI

CUDA_VISIBLE_DEVICES="$HIL_GPU_INDEX" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=serl_ur_infra:third_party/hil-serl/serl_launcher \
/tmp/gello-hil-rl-receive-overlay-v2/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_server.py \
  --host 127.0.0.1 \
  --port 50053 \
  --checkpoint /home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150 \
  --expected-checkpoint-sha256 e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997 \
  --replay-capacity 50000 \
  --intervention-capacity 10000 \
  --require-jax-backend gpu
```

`--threshold` is deliberately omitted: its default *is*
`DEFAULT_REWARD_THRESHOLD`, so leaving it off always tracks the code. Pass it
only when you intend a value different from the code default, and record why.

`--replay-capacity 50000` / `--intervention-capacity 10000` are also the code
defaults (`DEFAULT_REPLAY_CAPACITY` / `DEFAULT_INTERVENTION_CAPACITY`), spelled
out here so the run log shows them.

Before choosing a GPU, re-run `nvidia-smi`; do not assume an earlier free GPU is
still free. **As of 2026-07-29 nothing HIL-related is running on Kanu** — port
50053 is unbound and the GPUs are idle — so any statement in this document that
reads as "the server is up" is a record of a past run, not current state. Check
directly:

```bash
ssh kanu 'ss -ltnp 2>/dev/null | grep ":50053\b" || echo "50053 unbound"'
ssh kanu 'pgrep -af run_rlpd_receive_server || echo "no receive server"'
ssh kanu nvidia-smi
```

## Optional reproducible container

Initialize the pinned upstream submodule in the canonical checkout:

```bash
git submodule update --init --recursive third_party/hil-serl
docker build \
  -f serl_ur_infra/docker/Dockerfile.rlpd-receive \
  -t gello-hil-rlpd-receive:v2 .
```

`docker/requirements-rlpd-receive.lock` pins JAX `0.4.35` (jaxlib `0.4.34`),
Flax `0.8.5` and the replay dependencies, gRPC `1.74.0`, protobuf `3.20.3`, and
NumPy `1.26.4` — verified against the lock file on 2026-07-29. Note that this is
**not** the learner's pin set: `requirements-learner.lock` wants JAX/jaxlib
`0.5.3` and Flax `0.10.5`. The container is for the receive-only milestone only.
Agentlace is installed only because the upstream
`MemoryEfficientReplayBufferDataStore` imports its `DataStoreBase`; no Agentlace
socket or port is used.

The following optional example uses host networking so the process can remain
bound to Kanu loopback. The GPU index is a placeholder — pick it from
`nvidia-smi` at launch time:

> ⚠️ **Same retired classifier as above.** The mounted checkpoint path and
> `--expected-checkpoint-sha256 e329986b...` are the retired artifact with
> `0.0%` recall on the 0724 domain. Kept as a historical record only. The
> current canonical checkpoint **cannot simply be substituted here**: it is an
> orbax *directory*, and `checkpoint_sha256()` enforces `os.path.isfile()`
> before the loader runs, so the container would exit with `FileNotFoundError`.

```bash
export HIL_GPU_INDEX=GPU_NUMBER_SELECTED_AFTER_NVIDIA_SMI

docker run \
  --name gello-hil-rlpd-receive-v2 \
  --network host \
  --gpus "\"device=$HIL_GPU_INDEX\"" \
  -v /home/junhyeong/workspace/youngwoong/gello_software_remote_classifier/classifier_ckpt/cube_in_cup/checkpoint_150:/checkpoint/checkpoint_150:ro \
  -v /home/junhyeong/.serl:/root/.serl:ro \
  gello-hil-rlpd-receive:v2 \
  --host 127.0.0.1 \
  --port 50053 \
  --checkpoint /checkpoint/checkpoint_150 \
  --expected-checkpoint-sha256 e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997 \
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

Keep port 50053 private behind SSH. The remote side is always `50053`; the local
entry port only has to be free on laptop3:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 kanu
```

> The laptop3 operating convention is local **`50153`** → remote `50053`
> (`ros2_ur_ws/run_hil_actor.sh` defaults `SERVER_PORT=50153`). If you use that
> tunnel, pass `--port 50153` to the client below.

Then, from the laptop3 checkout (`/home/laptop3/gello_software`) and the
isolated gRPC client environment. **Do not use the system `python3`** — its
grpcio is `1.30.2`, not the `1.74.0` this contract is locked to:

```bash
PYTHONPATH=serl_ur_infra \
  /tmp/gello-hil-grpc-venv/bin/python \
  serl_ur_infra/scripts/run_rlpd_receive_smoke_client.py \
  --host 127.0.0.1 --port 50053 --steps 100
```

That venv was verified on 2026-07-29 to hold grpcio `1.74.0`, numpy `1.26.4` and
protobuf `3.20.3`, matching `serl_ur_infra/requirements-grpc.lock` exactly.

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
