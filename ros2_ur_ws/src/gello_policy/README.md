# gello_policy

For the policy-class-independent checkpoint wrapper and remote gRPC server, see
[`GENERIC_LEROBOT_POLICY_WRAPPER.md`](GENERIC_LEROBOT_POLICY_WRAPPER.md).

자신의 LeRobot checkpoint를 준비해 Kanu GPU 서버부터 로봇 PC와 실물 UR7e까지
처음부터 배포하는 한국어 end-to-end 절차는
[`GENERIC_LEROBOT_REMOTE_GPU_E2E_GUIDE_KO.md`](GENERIC_LEROBOT_REMOTE_GPU_E2E_GUIDE_KO.md)를
참고한다. 인터페이스 내부 구조와 기존 검증 기록은
[`REMOTE_LEROBOT_POLICY_INTERFACE_KO.md`](REMOTE_LEROBOT_POLICY_INTERFACE_KO.md)에 있다.
70,000-step cube-in-cup Flow Matching checkpoint를 Kanu에서 추론하고 UR7e 실물에
연결하는 절차는
[`FLOW_MATCHING_KANU_REAL_ROBOT_GUIDE_KO.md`](FLOW_MATCHING_KANU_REAL_ROBOT_GUIDE_KO.md)를 참고한다.

Real-robot deploy package for the trained ACT "put right banana in pot" policy on the
UR7e. A `rclpy` node impersonates the physical GELLO leader arm — it publishes the
exact same `/gello/joint_states` + gripper contract `gello_publisher_node` does — so
the existing `ur_gello_bringup` stack (`gello_ur_bridge`, `gello_move_to_start`,
Robotiq Modbus) drives the arm **completely unmodified**. The joint targets come from
a separate Python 3.12 process (`policy_server/act_server.py`, torch + lerobot 0.6.1)
that the leader node queries over a localhost ZMQ REQ/REP link.

## Local (production) vs remote (WIP): which files do what

This package now holds two deploy paths that share a folder. **Local** is what runs on
the real UR7e every day (ACT, Flow-Matching, local Diffusion — all over localhost ZMQ).
**Remote** is an opt-in work-in-progress that moves Diffusion inference to a separate
GPU server over gRPC. Nothing in the layout distinguishes them, so this is the list:

**LOCAL / PRODUCTION — load-bearing, treat as safety-critical:**

- `gello_policy/policy_leader_node.py` — **SHARED**: the single executable behind ACT,
  FM *and* local Diffusion
- `gello_policy/obs_assembler.py`, `gello_policy/joint_angles.py`
- `policy_server/` — the py3.12 torch/lerobot side (`act_server.py`, `fm_server.py`,
  `diffusion_server.py`)
- `config/act_deploy.yaml`, `config/fm_deploy.yaml`, `config/diffusion_deploy.yaml`
- `launch/ur7e_act_real.launch.py`, `launch/ur7e_diffusion_real.launch.py` —
  **SHARED**: `ur7e_diffusion_real.launch.py` serves **both** `run_ur7e_fm_real.sh` and
  `run_ur7e_diffusion_real.sh`
- [`../../run_ur7e_act_real.sh`](../../run_ur7e_act_real.sh),
  [`../../run_ur7e_fm_real.sh`](../../run_ur7e_fm_real.sh),
  [`../../run_ur7e_diffusion_real.sh`](../../run_ur7e_diffusion_real.sh)

**REMOTE / WIP — opt-in only, has never driven the physical arm:**

- `gello_policy/remote_policy_client.py`, `gello_policy/remote_diffusion_client.py`,
  `gello_policy/remote_diffusion_pb2.py`,
  `gello_policy/remote_diffusion_pb2_grpc.py`
- `policy_server/lerobot_policy_wrapper.py`, `policy_server/remote_lerobot_server.py`,
  `policy_server/remote_diffusion_server.py`,
  `policy_server/requirements-remote-diffusion.lock`
- [`../../run_ur7e_diffusion_remote.sh`](../../run_ur7e_diffusion_remote.sh),
  [`../../setup_remote_client_venv.sh`](../../setup_remote_client_venv.sh)
- `deploy/remote_diffusion/`, `proto/`, `scripts/generate_remote_diffusion_stubs.sh`,
  `test/test_remote_diffusion_client.py`
  (the rest of `scripts/` — `run_*_server.sh`, `download_*.sh`, the benchmarks — is
  local/production)

**FAKE-HARDWARE HELPERS — transport-agnostic, added alongside the remote work:**

- `gello_policy/fake_diffusion_observation_node.py`
- `launch/ur_control_fake_safe.launch.py`

Both are reachable from the LOCAL FM/Diffusion launch with `use_fake_hardware:=true`
under the default `inference_transport:=zmq`; they are not remote-only.

### Invariants a future editor must preserve

- **`inference_transport` defaults to `"zmq"` in three places** — the node's
  `declare_parameter`, the launch argument default, and `config/diffusion_deploy.yaml` —
  and **none of the three production run scripts pass it**. The remote path is reached
  only by explicitly passing `inference_transport:=grpc` (which only
  `run_ur7e_diffusion_remote.sh` does).
- **`use_fake_hardware` defaults to `false`.** The real-robot `ur_control.launch.py`
  include is selected by `UnlessCondition(use_fake_hardware)` with an unchanged argument
  set, and the new `fake_diffusion_observations` node is double-gated behind **both**
  `use_fake_hardware:=true` **and** `fake_observations:=true`.
- **The two SHARED files** (`policy_leader_node.py`, `ur7e_diffusion_real.launch.py`)
  must keep defaulting to zmq + real hardware. Any change to them affects ACT, FM and
  local Diffusion **simultaneously**.
- **The remote gRPC path has never driven the physical arm.** What HAS been validated
  (see [`../../REMOTE_DIFFUSION_RUNBOOK.md`](../../REMOTE_DIFFUSION_RUNBOOK.md) §1):
  synthetic observations round-tripping to a real GPU server over the SSH tunnel, and a
  full ROS run against **fake hardware** that reached the start pose, armed, executed,
  and passed a tunnel-disconnect FAULT test. Real-robot latency has also been measured
  end-to-end (~130 ms per inference refill, ~17 ms cached).
  What remains unproven is the only part that matters: a real arm moving under remote
  inference. Keep the teach-pendant **E-STOP in hand** on the first attempt.

## Architecture

Two processes, joined by ZMQ on `localhost`, because `lerobot` requires Python
&ge;3.12 while ROS2 **Humble**'s `rclpy` is built for Python 3.10 — they cannot share
one interpreter on Humble. Rather than fork `lerobot` or migrate the whole validated
safety stack (bridge, handshake, Robotiq Modbus) to a newer ROS distro just to get one
Python, this package keeps Humble and puts the policy in its own venv:

```
 ACT server (py3.12, torch+lerobot)  <── ZMQ REQ/REP, localhost ──>  policy_leader_node (py3.10, rclpy)
        policy_server/act_server.py                                   gello_policy/policy_leader_node.py
                                                                                |
                                                                                v  publishes
                                                          /gello/joint_states + /robotiq_gripper/command_percent
                                                                                |
                                                                                v  (unmodified)
                                                gello_ur_bridge + gello_move_to_start + robotiq_gripper_modbus
                                                                                |
                                                                                v
                                                                          real UR7e + 2F-85
```

`policy_leader_node` is the **synthetic GELLO leader**: it is a drop-in replacement
for `gello_publisher_node` from the arm-bridge's point of view. This means the
handshake, slew-rate clamping, One-Euro smoothing, staleness watchdog and soft-start
re-seed all come for free and are exercised identically to human-GELLO teleop — no
new safety-critical ROS code was written for the arm itself. Position safety
(joint-limit + max-deviation clamps) lives in `policy_leader_node` because the bridge
only bounds *speed*, never *position*.

See the project-root `DEPLOY_REPO_DECISION.md` for the full design rationale
(why extend `gello_software` instead of forking `lerobot` or starting a fresh repo,
why the synthetic-leader/ZMQ split, and the HIL-SERL forward path).

For the full runbook (data flow diagram, dataset/checkpoint download, startup
handshake timeline, safety model, troubleshooting) see
[`docs/ros2/GELLO_UR7E_ACT_DEPLOY.md`](../../../docs/ros2/GELLO_UR7E_ACT_DEPLOY.md).

## Nodes

- **policy_leader_node** (`policy_leader_node:main`, py3.10) — HOLD / EXECUTE / FAULT
  state machine. Boots in **HOLD**, publishing a fixed `start_pose` perfectly still so
  `gello_move_to_start` has a stationary target to chase. On the operator's
  `~/start_execution` call it ZMQ-RESETs the server and enters **EXECUTE**: each tick
  it assembles an observation (live joint state + gripper position + both camera
  JPEGs), ZMQ-queries the ACT server, applies two safety clamps to the returned
  target, and publishes it. Any ZMQ timeout/error drives it to **FAULT**, where it
  stops publishing entirely (fail-silent).

  | Param | Default | Meaning |
  | --- | --- | --- |
  | `publish_rate_hz` | `30.0` | Single timer rate driving both publishers |
  | `start_pose` | `[3.106, -1.817, 1.653, -1.618, -1.628, -3.195]` | HELD start pose (rad, UR order) |
  | `start_gripper` | `0.0` | Held gripper command while in HOLD (0=open..1=closed) |
  | `act_host` / `act_port` | `127.0.0.1` / `5591` | ACT server ZMQ endpoint |
  | `act_timeout_s` | `0.5` | ZMQ REQ `RCVTIMEO`; a timeout drives FAULT |
  | `obs_timeout_s` | `0.5` | Max age of any observation in EXECUTE; a missing/stale obs (e.g. frozen camera) drives FAULT |
  | `joint_limits_lo` / `joint_limits_hi` | see `config/joint_safety_limits.json` | 1.2x dataset envelope; OOD targets are clipped |
  | `max_dev_rad` | `0.5` | Max per-joint deviation of a target from the live pose |
  | `auto_start_on_stream` | `false` | If true, auto-enter EXECUTE once the bridge starts streaming, instead of waiting for the operator |
  | `cam1_topic` / `cam2_topic` | `/cam1/cam1/color/image_raw/compressed` / `/cam2/cam2/color/image_raw/compressed` | Camera sources passed through as raw JPEG |

  Services (`std_srvs/srv/Trigger`):

  | Service | Effect |
  | --- | --- |
  | `~/start_execution` | HOLD/FAULT → EXECUTE. Refused unless the live arm pose is within ~0.1 rad of `start_pose`, the **full fresh observation set is present** (both cameras + gripper position must be publishing), and the ZMQ RESET succeeds. |
  | `~/hold` | → HOLD, freezing the current live pose and holding the **last commanded gripper** value (keeps a mid-grasp grip; does not re-open). Falls back to `start_pose` if no live pose seen yet. |

- **act_server** (`policy_server/act_server.py`, py3.12, run standalone via
  `scripts/run_act_server.sh` — **not** a ROS node / not installed by `setup.py`) —
  loads `ACTPolicy.from_pretrained(checkpoint)` + `make_pre_post_processors`, sets
  `policy.config.n_action_steps` (receding-horizon `k`, default 30) **before** the
  first `policy.reset()`, and serves a ZMQ REP loop. Per `act` request it decodes both
  JPEGs, resizes to 360x640 + BGR→RGB (mandatory, **not** part of the saved
  preprocessor — see `policy_server/image_preprocess.py`), runs the saved
  preprocessor → `policy.select_action` → postprocessor, and returns a 7-float action
  (`q1..q6` rad + `grip_cmd` 0..1). A warm-up inference runs at load time so the first
  real tick doesn't spuriously time out.

  CLI (`python -m gello_policy.policy_server.act_server` or `scripts/run_act_server.sh`):
  `--checkpoint` (or `$ACT_CHECKPOINT`), `--host` (default `127.0.0.1`), `--port`
  (default `5591`), `--device` (`cuda`|`cpu`), `--n-action-steps` (default `30`).

  Wire protocol (ZMQ REQ/REP, multipart, localhost only — see
  `policy_server/zmq_protocol.py`):

  | Request | Frames | Reply |
  | --- | --- | --- |
  | reset | `[{"cmd":"reset"}]` | `[{"ok":true}]` |
  | act | `[{"cmd":"act","state":[7 floats]}, <cam1 jpeg>, <cam2 jpeg>]` | `[{"ok":true,"action":[7 floats]}]` or `[{"ok":false,"err":"..."}]` |

### Topics

| Topic | Type | Dir | Notes |
| --- | --- | --- | --- |
| `/gello/joint_states` | `sensor_msgs/JointState` | pub (policy_leader_node) | Synthetic leader output: 6 arm joints, UR order, radians, position-only (velocity/effort empty) |
| `/robotiq_gripper/command_percent` | `std_msgs/Float32` | pub (policy_leader_node) | 0=open..1=closed, identity mapping (no threshold/binarize). Published directly — `gello_gripper_bridge` is **not** launched, to avoid dual writers |
| `/joint_states` | `sensor_msgs/JointState` | sub (policy_leader_node) | Live UR7e joint feedback; reordered by name to UR order for both the observation and the max-deviation clamp |
| `/robotiq_gripper/position_percent` | `std_msgs/Float32` | sub (policy_leader_node) | Live gripper position (0=open..1=closed); `observation.state[6]` |
| `/cam1/cam1/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | sub (policy_leader_node) | RealSense cam1 (D435, serial `147122072740`); raw JPEG passed through to the ACT server unmodified |
| `/cam2/cam2/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | sub (policy_leader_node) | RealSense cam2 (D435iF, serial `243222072700`); raw JPEG passed through |
| `/forward_position_controller/commands` | `std_msgs/Float64MultiArray` | sub (policy_leader_node) | Read-only — used **only** to detect the bridge has started streaming, for the optional `auto_start_on_stream` |

## Build

Requires ROS2 **Humble** for the `policy_leader_node` side (this workspace targets
`ros-humble-ur` / `realsense2_camera`; the split exists precisely so ROS stays on
Humble while ACT inference runs on a newer Python).

`ur7e_act_real.launch.py` runs `ur_gello_bringup` nodes (`gello_ur_bridge`,
`gello_move_to_start`, `robotiq_gripper_modbus`) and includes `ur_robot_driver`'s
`ur_control.launch.py`, so you **must build `ur_gello_bringup` too** — `gello_policy`
alone is not enough. `gello_policy` now declares `ur_gello_bringup` + `ur_robot_driver`
as `exec_depend`, so `rosdep install --from-paths src` pulls the driver too:

```bash
cd gello_software/ros2_ur_ws
# 1) resolve deps (ur_robot_driver, realsense2_camera, ...); skip dynamixel_sdk (pip'd)
rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk
# 2) build. Either both packages explicitly...
colcon build --packages-select gello_policy ur_gello_bringup
# ...or simplest, just build the whole workspace:
#   colcon build
source install/setup.bash
```

`ur_gello_bringup`'s own `ros2_ur_ws/build_ur7e.sh` runs that same
`rosdep install --skip-keys dynamixel_sdk` + a `colcon build`, so running it first is
an equivalent way to get the driver deps in place.

The `policy_server/` (py3.12, torch/lerobot) side is **not** installed by `colcon` —
it runs from its source directory inside its own venv. Create the venv **at
`ros2_ur_ws/act_venv`** — that is the default location all three run scripts expect
(`ACT_VENV=ros2_ur_ws/act_venv`); if you put it elsewhere, `export ACT_VENV=/your/path`:

```bash
cd gello_software/ros2_ur_ws        # create the venv HERE
python3.12 -m venv act_venv
act_venv/bin/pip install --upgrade pip
act_venv/bin/pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
act_venv/bin/pip install -r src/gello_policy/policy_server/requirements-act.lock
```

(Versions above are pinned in `policy_server/requirements-act.lock`. See that file's
header comment before bumping anything — `lerobot==0.6.1` is the exact API these
modules target.)

Download the trained checkpoint from the Hugging Face Hub:

```bash
CKPT=$(ros2_ur_ws/src/gello_policy/scripts/download_checkpoint.sh | tail -1)
```

## Run

**First, bring up both RealSense cameras by serial** (in a separate terminal). The
policy cannot run without them: EXECUTE needs a fresh `cam1`/`cam2` frame every tick,
and both `~/start_execution` and the in-EXECUTE obs-freshness watchdog will refuse /
FAULT if a camera is absent or stale (see Safety). Bind by **serial** (quoted, so the
all-digit value is not coerced to an int):

```bash
# cam1 = D435, serial 147122072740 ; cam2 = D435iF, serial 243222072700
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=cam1 camera_namespace:=cam1 \
    serial_no:="'147122072740'" rgb_camera.color_profile:="'1280x720x30'" &
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=cam2 camera_namespace:=cam2 \
    serial_no:="'243222072700'" rgb_camera.color_profile:="'1280x720x30'" &

# Confirm both compressed topics are publishing at ~30 Hz BEFORE starting the deploy:
ros2 topic hz /cam1/cam1/color/image_raw/compressed
ros2 topic hz /cam2/cam2/color/image_raw/compressed
```

**Then start the deploy:**

```bash
cd gello_software/ros2_ur_ws
ACT_CHECKPOINT="$CKPT" ./run_ur7e_act_real.sh
```

This starts the py3.12 ACT server in the background, waits for it to come up, then
`ros2 launch`es `gello_policy ur7e_act_real.launch.py` (arm driver + synthetic leader
+ bridge handshake + gripper). Both processes are torn down together on Ctrl-C.

Operator procedure:

1. Confirm the pendant's External Control program is **Play**ing (or set
   `HEADLESS=true` for Method B — requires the robot in REMOTE mode).
2. Wait for the move-to-start handshake to converge — the arm drives to the policy's
   held `start_pose` and parks. **No autonomous motion happens yet.**
3. Explicitly begin autonomous execution:
   ```bash
   ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger
   ```
4. To pause at any time (without killing the process):
   ```bash
   ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger
   ```
   `~/hold` freezes the arm at its **current live pose** and holds the **last
   commanded gripper value** — so pausing mid-grasp keeps the object gripped (it does
   **not** re-open to `start_gripper`). Only boot-time HOLD uses `start_gripper`
   (`0.0` = open).

Standalone server (for testing without a robot):

```bash
ACT_CHECKPOINT="$CKPT" ./src/gello_policy/scripts/run_act_server.sh
```

### Dry-run / mock (no real arm)

`use_fake_hardware:=true` alone does **not** make the mock arm move under ACT. Because
the launch skips the Robotiq node on fake hardware, nothing publishes
`/robotiq_gripper/position_percent`, so `grip_pos` is missing and EXECUTE FAULTs on the
obs-freshness watchdog. To dry-run the full autonomous path against mock hardware you
must **also** run both RealSense cameras **and** fake the gripper position topic:

```bash
# (with use_fake_hardware:=true launch running, plus both RealSense cameras)
ros2 topic pub /robotiq_gripper/position_percent std_msgs/Float32 "{data: 0.0}" -r 10
```

So the mock autonomous path is never a single command — it needs the fake-hardware
launch + both cameras + a faked gripper-position publisher, then `~/start_execution`.

> See [`docs/ros2/GELLO_UR7E_ACT_DEPLOY.md`](../../../docs/ros2/GELLO_UR7E_ACT_DEPLOY.md)
> for the full startup-handshake timeline, cadence-tuning guidance, and
> troubleshooting.

## Diffusion variant

A **Diffusion Policy** sibling deploy reuses this *entire* package unchanged — same
`policy_leader_node`, same bridge/handshake/clamps/gripper, same 7-D JOINT action
contract. It swaps **only the inference server**: `policy_server/diffusion_server.py`
loads `DiffusionPolicy.from_pretrained` instead of `ACTPolicy`. The one substantive code
difference is at load: it reads a `PreTrainedConfig` and **overrides the noise scheduler
to DDIM with `num_inference_steps=10`** (`cfg.noise_scheduler_type="DDIM"`), because the
model was trained as DDPM/100 and a plain `from_pretrained` would keep the slow full
schedule. DDIM-10 keeps per-replan latency low enough for the 30 Hz loop.

- **ZMQ port `5592`** (ACT is `5591`), so both servers can coexist. Env/CLI use
  `DIFFUSION_*` names (`DIFFUSION_CHECKPOINT`, `DIFFUSION_PORT`, `DIFFUSION_DEVICE`,
  `DIFFUSION_N_ACTION_STEPS` (default 32), `DIFFUSION_NUM_INFERENCE_STEPS` (default 10),
  `DIFFUSION_SCHEDULER` (default DDIM)).
- **Model:** HF `Bigenlight/diffusion_banana_in_pot_joint` (best checkpoint 80k, selected
  by open-loop rollout MAE); download via `scripts/download_diffusion_checkpoint.sh`.
- **Deps:** the py3.12 venv needs `policy_server/requirements-diffusion.lock` — the same
  pins as the ACT lock **plus `diffusers==0.35.2`** (required by lerobot's diffusion
  policy; the ACT lock omits it). If you already built the ACT venv, just
  `pip install diffusers==0.35.2` into it.
- **Timeouts** are widened in `config/diffusion_deploy.yaml` because a DDIM refill tick is
  heavier than an ACT forward: `act_timeout_s=0.6` < `staleness_timeout_s=0.8` (the leader
  stays the primary fault owner). **Run `scripts/benchmark_diffusion_latency.py` on the
  robot PC before the first arm run**; if p99 refill latency > ~0.5 s, lower
  `DIFFUSION_NUM_INFERENCE_STEPS` (e.g. 5) rather than widening those timeouts.

Run it with `ros2_ur_ws/run_ur7e_diffusion_real.sh` (mirrors `run_ur7e_act_real.sh`).
Full standalone runbook (status, model facts, setup, run, diffusion-specific safety and
inference behavior): [`docs/ros2/GELLO_UR7E_DIFFUSION_DEPLOY.md`](../../../docs/ros2/GELLO_UR7E_DIFFUSION_DEPLOY.md).

## Safety

> **Keep the teach-pendant E-STOP within reach at all times.** This package drives a
> real robot autonomously once `~/start_execution` is called.

- **Speed is bounded by the unmodified bridge**, not by this package: `gello_ur_bridge`
  never lets the arm move faster than its `max_step_rad` slew ceiling (0.625 rad/s
  sustained at the deployed 250 Hz / 0.0025 rad settings). This package does not
  change that ceiling.
- **Position is bounded by `policy_leader_node`**, in order, every EXECUTE tick:
  1. clip each joint to the 1.2x dataset envelope (`joint_limits_lo`/`joint_limits_hi`,
     see `config/joint_safety_limits.json`) — an out-of-distribution guard;
  2. clip each joint so it deviates at most `max_dev_rad` (default 0.5 rad) from the
     **live** `/joint_states` — bounds any single-step jump from the actual pose.
  A throttled `WARN` is logged whenever either clamp engages.
- **Fail-silent, never fail-frozen-forever.** On any ZMQ timeout/error or a
  `{"ok":false}` reply, the node enters FAULT and **stops publishing**
  `/gello/joint_states` entirely. This deliberately trips the bridge's 0.5 s
  staleness watchdog, which halts the arm and re-seeds with a soft-start ramp on
  recovery. The node never keeps re-publishing a stale target through an outage.
- **Observation-freshness watchdog.** In EXECUTE, if any observation
  (`joint_states`, `grip_pos`, `cam1`, `cam2`) is missing or older than
  `obs_timeout_s` (default 0.5 s), the node FAULTs — catching a frozen/hung camera
  that would otherwise drive the policy on a stale frame. Operator consequence: if a
  **camera dies mid-episode the arm FAULTs (halts) and does NOT auto-resume**;
  re-arming requires `~/start_execution` near `start_pose`, same as any FAULT.
  `~/start_execution` also refuses to arm unless the full fresh obs set is present, so
  **the cameras must be running before you start execution.**
- **The operator gates all autonomous motion.** The arm only ever chases a *held,
  stationary* pose during the handshake; ACT motion begins only after an explicit
  `~/start_execution` call (unless `auto_start_on_stream:=true` is set).
- **After a mid-episode FAULT there is no resume-from-here.** `~/start_execution`
  refuses to re-arm unless the live pose is again within ~0.1 rad of `start_pose`
  — see the runbook's FAULT-recovery section for the procedure.
- **Single gripper writer.** `ur7e_act_real.launch.py` intentionally does **not**
  launch `gello_gripper_bridge`; `policy_leader_node` is the sole publisher of
  `/robotiq_gripper/command_percent`.
