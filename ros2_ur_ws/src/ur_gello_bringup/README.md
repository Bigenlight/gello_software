# ur_gello_bringup

ROS2 (Humble/Jazzy) `ament_python` package that teleoperates a Universal
Robots arm (ur5e/ur7e) from a GELLO leader arm, with RViz visualization.

## Nodes

- **gello_publisher** (`gello_publisher_node:main`) — reads the GELLO Dynamixel
  leader arm over serial and publishes `sensor_msgs/JointState` on
  `/gello/joint_states` (6 arm joints in UR order) plus the gripper width on
  `/gripper/gripper_client/target_gripper_width_percent` (`std_msgs/Float32`, 0..1).
- **gello_ur_bridge** (`gello_ur_bridge_node:main`) — subscribes to
  `/gello/joint_states`, applies EMA smoothing / step limiting / staleness
  guarding / a `deadband_rad` noise gate (suppresses at-rest Dynamixel/hand
  tremor), seeds its first command from the robot's **actual `/joint_states`**
  (not the GELLO pose) to avoid a start-up snap (this fixed a real "External
  Control speed limit" protective stop), and publishes
  `/forward_position_controller/commands`
  (`std_msgs/Float64MultiArray`, 6 doubles, UR order).
- **robotiq_urcap** (`robotiq_urcap_node:main`) — drives the Robotiq gripper
  via the UR URCap from the gripper width topic.
- **fake_gello** (`fake_gello_node:main`) — publishes synthetic
  `/gello/joint_states` for testing the pipeline in RViz without hardware.
- **gello_move_to_start** (`gello_move_to_start_node:main`) — one-shot
  handshake used on the real robot: drives the arm to the **current live GELLO
  pose** (read from `/gello/joint_states`, reordered by name) via the
  `scaled_joint_trajectory_controller`'s `FollowJointTrajectory` action, then
  STRICT-switches control to `forward_position_controller` for streaming
  teleop. (Target is the live GELLO pose, NOT `start_joints`.) Not needed on
  the fake/mock path.

### Topics

| Topic | Type | Notes |
| --- | --- | --- |
| `/gello/joint_states` | `sensor_msgs/JointState` | 6 arm joints, UR order, radians |
| `/gripper/gripper_client/target_gripper_width_percent` | `std_msgs/Float32` | 0..1 |
| `/forward_position_controller/commands` | `std_msgs/Float64MultiArray` | 6 doubles, UR order |

UR joint order:
`[shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]`

## Build

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
colcon build --packages-select ur_gello_bringup
source install/setup.bash
```

## Run

### Fake GELLO (RViz test, no hardware)

Publishes synthetic joint states so you can verify the pipeline and see the
UR5e move in RViz without a real leader arm or robot.

```bash
ros2 launch ur_gello_bringup ur_gello_rviz.launch.py
```

### Real GELLO -> UR5e mock (RViz)

Reads the physical GELLO leader arm and drives **mock** hardware in RViz.
Config (serial port, joint offsets/signs, gripper) is in `config/ur_gello.yaml`.

```bash
export GELLO_REPO_ROOT=/home/laptop3/gello_software
ros2 launch ur_gello_bringup ur_gello_rviz.launch.py source:=gello
```

> Note: `ur_gello_rviz.launch.py` defaults to `use_fake_hardware:=true` (and a
> dummy `robot_ip:=192.168.56.101`), so the command above drives mock/RViz
> hardware from the real GELLO — it does **not** move a physical UR5e. To
> command a real UR5e, override the args, e.g.
> `... source:=gello use_fake_hardware:=false robot_ip:=<real-UR5e-ip>`, and run
> on the UR control PC (the machine connected to the robot and the GELLO serial
> adapter).

## UR7e (Humble)

UR7e support reuses the same nodes unchanged (ur5e and ur7e share identical
joint limits) via a dedicated `ur7e_gello_rviz.launch.py` (fake/mock) and
`ur7e_gello_real.launch.py` (real robot) launch pair, plus the new
`gello_move_to_start` node for a safe handshake before streaming teleop.

> Note: with `ros-humble-ur` 2.8.1, ur7e visuals in RViz currently render the
> ur5e mesh (`config/ur7e` visual_parameters point at `meshes/ur5e`, not yet
> updated upstream) and a missing `ur7e_update_rate.yaml` produces a harmless
> warning; joint-space teleop itself is unaffected. Upgrade to
> `ros-humble-ur` >= 2.13.2 for accurate ur7e visuals/kinematics.

**Safety invariant:** GELLO is always a passive, read-only input device —
its Dynamixel torque is never enabled, and it never receives commands.

1. **Fake verify** (mock hardware, no GELLO/robot attached — this is the
   path used to validate the bringup on a dev machine):

   ```bash
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake
   ```

2. **Real GELLO -> ur7e mock** (physical GELLO leader arm driving mock
   hardware/RViz, no real robot):

   ```bash
   export GELLO_REPO_ROOT=/home/laptop3/gello_software
   ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=gello
   ```

3. **Real robot** (physical ur7e; runs on the UR control PC):

   ```bash
   export GELLO_REPO_ROOT=/home/laptop3/gello_software
   ros2 launch ur_gello_bringup ur7e_gello_real.launch.py robot_ip:=<IP>
   ```

See `../../../docs/ros2/GELLO_UR7E_ROS2_BRINGUP.md` for the full pipeline
diagram, node-role/topic-contract tables, and troubleshooting.

## Pause / Resume (scene reset with both hands free)

### The problem

During teleop you are holding the GELLO leader with one or both hands. To reset
the scene (put the object back, move a fixture, re-grip) you need to let go of the
leader — but the leader is a **passive** arm, so when you release it, it droops
under gravity and its flopping joints stream straight through to the robot. On a
real UR7e that is a snap-and-crush hazard.

Pause/Resume gives you a hard OFF switch for the leader→robot signal path so you
can put the leader down, use both hands in the workspace, then bring the robot
back under control with a **smooth, bounded** glide instead of a snap.

- **Pause** unconditionally freezes streaming. The arm holds its last commanded
  pose (`forward_position_controller` keeps its setpoint); the Robotiq gripper
  holds its last position onboard. Not-streaming is the safe state, so pause is
  never gated, deferred, or refused — one click, always succeeds.
- **Resume** is **gated** and **fail-closed**: any refused or failed resume leaves
  the bridge paused and publishing nothing. It only proceeds when it can restart
  jump-free, and the first published command equals where the arm/gripper already
  is — then a rate-limited glide closes the gap.

### Operator workflow (GUI)

The **Teleop** bar lives in the recorder GUI (`ros2 run gello_recorder
gello_recorder_gui`), directly above the record controls. It shows two live state
labels (`arm:` and `gripper:`) plus a **Pause Teleop** and a **Resume Teleop**
button.

1. **Pause.** Click **Pause Teleop** (single click). Both the arm and the gripper
   bridge pause. The labels turn red (`PAUSED`). The robot holds still.
2. **Reset the scene** with both hands free. The leader can droop, be repositioned,
   whatever — nothing reaches the robot while paused.
3. **Re-pose the leader** roughly back onto the robot's frozen pose, and **hold it
   still** for about half a second. (You are matching the leader to where the robot
   is parked, not the other way around — the robot has not moved.)
4. **Resume.** Click **Resume Teleop**. Because resume moves a real robot, it is a
   **two-click confirm**: the button turns orange and reads *"Confirm Resume (robot
   will move!)"*. Click it again **within 3 seconds** to actually request resume.
   If you wait too long it reverts and nothing happens — click again to re-arm.
5. On acceptance the arm re-seeds from its actual pose (zero jump), soft-starts,
   and **glides** to the leader pose; the label shows `CHASING` during the glide
   then `FOLLOWING` once caught up. The gripper ramps from its actual position to
   the leader value over ~2 s (`RAMPING` → `FOLLOWING`). Keep clear and keep the
   leader still until both read `FOLLOWING`.

The **Resume Teleop** button is only enabled when the arm reports `PAUSED`. If a
resume is **refused**, the robot does not move, the bridge stays paused, and the
refusal reason is shown in the status bar for ~6 s — read it, fix that one thing
(see the refusal table below), and resume again.

### Services, topics, and states

All services are `std_srvs/srv/Trigger`; all state topics are `std_msgs/msg/String`
at 5 Hz. Node names: `gello_ur_bridge` and `gello_gripper_bridge` (both in this
package).

| Name | Type | Behavior |
| --- | --- | --- |
| `/gello_ur_bridge/pause` | Trigger | Unconditional. Stops publishing; arm holds last setpoint. Always succeeds. |
| `/gello_ur_bridge/resume` | Trigger | **Strict** gate (fresh leader + arm pose known + every joint within `resume_align_tol`). Used by the startup handshake; leave it alone for manual reset. |
| `/gello_ur_bridge/resume_chase` | Trigger | **The resume you want for a scene reset.** Gated glide across a larger, bounded gap under a still leader (see gates below). |
| `/gello_ur_bridge/state` | String | `PAUSED` / `WAITING` / `STALE` / `CHASING` / `FOLLOWING` (that precedence). |
| `/gello_gripper_bridge/pause` | Trigger | Unconditional. Stops republishing to `/robotiq_gripper/command_percent`; Robotiq holds onboard. Always succeeds. |
| `/gello_gripper_bridge/resume` | Trigger | Gated (fresh leader gripper sample + actual gripper position known). Seeds at actual position, then ramps. |
| `/gello_gripper_bridge/state` | String | `PAUSED` / `WAITING` / `RAMPING` / `FOLLOWING`. |

**Arm states:** `PAUSED` (frozen) · `WAITING` (no leader sample yet, or seen but not
yet seeded) · `STALE` (leader stream older than `staleness_timeout_s`) · `CHASING`
(gliding — worst joint still more than `state_chase_done_tol` from the leader) ·
`FOLLOWING` (tracking normally).

**Gripper states:** `PAUSED` · `WAITING` (no fresh leader gripper sample) ·
`RAMPING` (post-resume slew window) · `FOLLOWING`.

### Safety gates and why each exists

`resume_chase` refuses (and stays paused, publishing nothing) unless **all** hold:

| Gate | Default | Why it exists |
| --- | --- | --- |
| **Not paused → "Already following"** | — | Calling resume when already streaming is a harmless no-op success, never a double-motion. |
| **(a) Fresh leader sample** | age ≤ `staleness_timeout_s` (0.5 s) | A dead or frozen leader stream must never authorize motion — otherwise a crashed `gello_publisher` could resume against a stale target. |
| **(b) Actual robot pose known** | `/joint_states` seen | The gap can't be measured without the arm's real pose; and the zero-jump seed needs it. |
| **(c) Leader quasi-still** | worst-joint speed ≤ `resume_chase_still_speed` (0.10 rad/s) over `resume_chase_still_window_s` (0.3 s) | A moving leader means the robot would chase a moving target on resume. **Fail-closed:** too few samples / too little time coverage counts as *not still* → refuse. Hold the leader steady for ~0.5 s so the window fills. |
| **(d) Per-joint gap ≤ cap** | `resume_chase_max_gap` (1.5 rad) | Bounds the glide. 1.5 rad is a worst-case glide of ~2.4 s + 0.7 s ease-in. A larger mispose is refused so the robot never makes a long autonomous sweep — re-pose the leader closer first. |

On acceptance the **only** state change is to drop the bridge into its existing
seed branch: the next 250 Hz tick re-anchors the leader target onto the branch
nearest the arm's actual pose (`wrapped_nearest`, so the glide always takes the
short way — never a ~2π spin), seeds the command from the actual pose (first
command = where the arm already is, zero jump), restarts the `soft_start_s` (0.7 s)
ramp, and thereafter slews each joint by at most `max_step_rad` per cycle
(0.0025 rad @ 250 Hz = 0.625 rad/s sustained). No new publishing path is
introduced — this is the same jump-free, soft-started, slew-clamped machinery the
startup handshake and the staleness-recovery path already use.

**What every refusal message means:**

| Refusal (substring in the message) | Meaning | Do this |
| --- | --- | --- |
| `no fresh GELLO sample (age=… > 0.50s)` | Leader stream is stale/dead. | Check `gello_publisher` is running and publishing `/gello/joint_states`. |
| `robot actual pose unknown (no /joint_states yet)` | Robot driver / `joint_state_broadcaster` not up. | Wait for / restore `/joint_states`. |
| `leader is moving (or stillness not yet established)` | Gate (c): leader not demonstrably still. | Hold the GELLO steady for ~0.5 s, then resume again. |
| `gap too large: max … > resume_chase_max_gap 1.500` | Gate (d): leader too far from the frozen arm pose (message names the worst joint + per-joint gaps). | Move the GELLO closer to the robot's parked pose, then resume again. |
| Gripper `REFUSED: stale leader` | No fresh leader gripper sample. | Restore the leader stream. |
| Gripper `REFUSED: no actual gripper position` | No `/robotiq_gripper/position_percent` feedback and nothing published yet. | Ensure the Robotiq node is up and publishing position feedback. |

The gripper resume is seeded at the gripper's **actual** position (`invert` stays
`false` — a crush-hazard safeguard) and, for `resume_ramp_s`, slew-limits every
output toward the live leader value (deadband bypassed so the ramp advances
monotonically), so there is never a gripper jump on resume.

### New parameters (`config/ur7e_gello.yaml`)

`gello_ur_bridge`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `resume_chase_max_gap` | `1.5` | Per-joint circular-gap hard cap (rad). Above this, `resume_chase` refuses. |
| `resume_chase_still_speed` | `0.10` | Leader quasi-still speed threshold (rad/s). |
| `resume_chase_still_window_s` | `0.3` | Window (s) the quasi-still speed is measured over (fail-closed on too little coverage). |
| `state_publish_rate_hz` | `5.0` | Rate of the `~/state` status publisher. |
| `state_chase_done_tol` | `0.10` | Per-joint tolerance (rad) below which `~/state` reports `FOLLOWING` instead of `CHASING`. |

`gello_gripper_bridge`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `staleness_timeout_s` | `0.5` | A leader gripper sample older than this is stale; `~/resume` refuses on it. |
| `resume_ramp_s` | `2.0` | Duration (s) of the post-resume slew-limited ramp. |
| `resume_slew_per_s` | `0.6` | Max output change per second (fraction of stroke) during the ramp. `0.6` → full 0..1 stroke in ~1.67 s. |
| `state_publish_rate_hz` | `5.0` | Rate of the gripper `~/state` publisher. |

(The strict `resume_align_tol` used by `/gello_ur_bridge/resume` is `0.08` in
`ur7e_gello.yaml`; the node's own declared default is `0.05`. It is **not**
consulted by `resume_chase`.)

### Terminal fallback (no GUI)

Every button maps to a plain service call. Pause both bridges, reset, hold the
leader still, then resume both:

```bash
# Pause (unconditional — always succeeds)
ros2 service call /gello_ur_bridge/pause        std_srvs/srv/Trigger
ros2 service call /gello_gripper_bridge/pause   std_srvs/srv/Trigger

# ... reset the scene, re-pose the leader onto the robot, hold it still ...

# Resume (gated — read success/message; a refusal leaves the bridge PAUSED)
ros2 service call /gello_ur_bridge/resume_chase std_srvs/srv/Trigger
ros2 service call /gello_gripper_bridge/resume  std_srvs/srv/Trigger

# Watch the live states
ros2 topic echo /gello_ur_bridge/state
ros2 topic echo /gello_gripper_bridge/state
```

A `success: false` response is a **refusal, not an error** — the robot did not
move and the bridge is still paused. Read `message`, fix the one named condition,
and call resume again. (Do **not** use `/gello_ur_bridge/resume` for a manual
reset: that is the strict startup-handshake gate and will refuse any non-trivial
gap.)

### Sim rehearsal (mock-hardware caveat)

You can rehearse the whole pause/refuse/glide story against the mock stack with no
robot. In one terminal bring up the fake pipeline
(`ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake`, or
`./run_ur7e_gello_sim.sh`); in a second terminal run:

```bash
cd /home/laptop3/gello_software/ros2_ur_ws
./check_pause_resume_sim.sh
```

It drives `fake_gello` (`/fake_gello/hold`, `/fake_gello/sweep`,
`/fake_gello/collapse`, `/fake_gello/set_pose`) to exercise pause, the fail-closed
refusals (moving leader; gap > 1.5 rad), and an accepted ~0.8 rad glide, printing
PASS/FAIL per step.

> ⚠️ **SIM PASS is necessary but NOT sufficient.** Mock hardware enforces **no**
> velocity limits and **no** protective stops. Velocity-limit and smoothness
> safety MUST be re-verified by a human on the real UR7e before using this in data
> collection.
