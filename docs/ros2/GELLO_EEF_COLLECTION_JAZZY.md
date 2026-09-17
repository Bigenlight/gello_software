# GELLO EEF collection on Jazzy

For rendered videos from the MuJoCo simulation (not the ROS mock), use the
[sim_collect migration and collection guide](../../sim_collect/UBUNTU24_JAZZY_MIGRATION.md).

This checkout targets ROS 2 Jazzy on Ubuntu 24.04. The wrappers resolve paths
from their own checkout, so run them from any working directory. They also put
the ROS Jazzy and system Python 3.12 paths ahead of an auto-activated conda
base (Python 3.13). `GELLO_ROS_DISTRO` can override `jazzy` when using another
installed ROS distribution.

## Before a real run

Do not launch the real wrapper until the robot Ethernet link has carrier and
both RealSense cameras are connected. As observed on 2026-09-16, this machine
has neither a robot NIC carrier nor cameras; that is a current readiness
observation, not a permanent hardware restriction. Keep the pendant E-stop
reachable and verify the External Control or
headless/Remote setup described in the real wrapper before enabling motion.

Build or rebuild the workspace with the validated helper:

```bash
cd /path/to/gello_software_jazzy/ros2_ur_ws
./build_ur7e.sh
```

## Three-terminal EEF collection

Use the same checkout path in all three terminals.

```bash
# Terminal 1 — real UR7e + physical GELLO EEF teleop
cd /path/to/gello_software_jazzy/ros2_ur_ws
HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef gripper_mode:=continuous
```

```bash
# Terminal 2 — EEF operator GUI
cd /path/to/gello_software_jazzy/ros2_ur_ws
START_POSE_CONFIG="$PWD/src/gello_policy/config/ifql_deploy.yaml" ./run_eef_gui.sh
```

```bash
# Terminal 3 — task recorder GUI; it owns and launches both cameras
cd /path/to/gello_software_jazzy/ros2_ur_ws
./run_task_recorder.sh
```

The task recorder resolves the two connected camera serials, waits for camera
warm-up, and writes `take_<NN>_<timestamp>/` folders under the recorder output
root. Override `CAM1_SERIAL`, `CAM2_SERIAL`, `COLOR_PROFILE`,
`CAMERA_WARMUP_S`, or `RECORDER_OUTPUT_ROOT` only when needed.

## Motion and pose defaults

`control_mode:=eef` derives `start_mode:=switch_only`: bring-up switches the
controller in place and does not move the arm. EEF motion begins when the GUI
engages teleop; the recorder's **GO HOME** button and the EEF GUI's **GO TO
START POSE** button also command arm motion. The recorder **START** button only
starts a take; it is not required for those motion paths.

The EEF GUI default above reads `ifql_deploy.yaml`:

```text
start_pose   = [-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]
start_gripper = 0.0
```

The recorder's **GO HOME** uses its own fixed recorder `HOME_JOINTS` target
(`3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331`) and opens the gripper.
These are intentionally distinct targets; do not treat GO HOME as the
`START_POSE_CONFIG` target. Choose the deploy YAML for the task being collected
when changing `START_POSE_CONFIG`.

After **GO HOME**, both bridges remain `PAUSED` by design. Leave the recorder's
generic **Resume Teleop** button alone: it calls the joint-mode resume path.
With the GELLO trigger held open, use the EEF GUI in Terminal 2 and press
**ENGAGE**; it performs `eef_resume`, commits `pos_scale`, and then
`eef_engage`, re-anchoring at the current pose.
The gripper is separate: resume it with the EEF GUI's gripper resume control
when ready. Arm ENGAGE alone does not resume a paused gripper.

## Gripper contract

The real wrapper defaults to `gripper_mode:=continuous` (its `GRIPPER_MODE`
banner default and launch default). The command is preserved as `0.0 = OPEN`
and `1.0 = CLOSED` throughout the teleop, recorder, and task sidecar. Continuous
mode passes the trigger continuously in that range. `gripper_mode:=discrete` snaps
to the two endpoints with hysteresis; recorded `grip_cmd` values are then
binary. Keep the mode fixed across a dataset and leave the trigger open before
resuming after GO HOME.

## Safe wrapper validation

These checks do not contact a robot, GELLO serial device, or camera:

```bash
cd /path/to/gello_software_jazzy/ros2_ur_ws
bash -n run_eef_gui.sh run_task_recorder.sh run_recorder.sh \
  run_ur7e_gello_real.sh run_ur7e_gello_mock.sh run_ur7e_gello_sim.sh
DRY_RUN=1 ./run_ur7e_gello_mock.sh --print-args \
  control_mode:=eef launch_rviz:=false
```

The mock script pins `robot_ip:=127.0.0.1` and `use_fake_hardware:=true`; it
cannot be redirected to the physical robot. The real wrapper is intentionally
not run by this readiness check.

## Verified 2026-09-17 (no real hardware)

- Three-package build and rosdep check passed; full suite: **931 passed**.
- EEF GUI constructed and refreshed offscreen with the IFQL start-pose YAML.
- Actual ROS GenericSystem smoke passed: STRICT handoff into HOLD, singular
  anchor refusal, zero-jump engage with mismatched leader/robot poses, reclutch,
  leader delta tracking, disengaged hold, and zero-jump re-engage.
- Synthetic recorder output reopened successfully: RGB MP4s have 39 decodable
  frames per camera; separate depth run has 29 frames per camera in MP4/depth.h5.
  These are synthetic transport/storage tests, not camera or robot measurements.
- Evidence is under `ros2_ur_ws/log/jazzy_port_20260916/` (ignored by Git).

To repeat the EEF smoke in an **unused** local ROS domain:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=176 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  /usr/bin/python3 setup_jazzy/eef_mock_smoke.py
```

The smoke refuses an occupied domain and verifies GenericSystem before sending
any mock positioning commands. It seeds a non-singular mock pose after checking
that the driver's default singular pose is correctly refused; no safety gate
or production control limit is relaxed. This does not validate real UR tracking,
physical camera serial assignment, gripper motion, or a safe real HOME path.
