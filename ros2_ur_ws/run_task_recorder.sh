#!/usr/bin/env bash
# Task recorder GUI: the normal dual-camera recorder PLUS a GO HOME button.
#
# Run this in a THIRD terminal, alongside the EEF teleop and its operator GUI:
#     Terminal 1:  HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef
#     Terminal 2:  ./run_eef_gui.sh
#     Terminal 3:  ./run_task_recorder.sh
#
# Recording is IDENTICAL to gello_recorder_gui -- same take_<NN>_<stamp>/ folder,
# same vectors.h5 tables, same cam1.mp4 / cam2.mp4, same Teleop pause/resume bar.
# The only addition is one button, for tasks where every take must start from the
# same pose (the HIL RESET pose):
#
#   GO HOME (two-click confirm; the robot WILL move)
#     1. pauses BOTH teleop bridges (arm + gripper; unconditional Triggers),
#     2. strict-switches forward_position_controller -> scaled_joint_trajectory_controller,
#     3. drives the arm to the HOME joints with ONE trajectory point whose
#        duration is distance-proportional (largest joint gap / 0.8 rad/s,
#        clamped to 2-10 s), so a short hop is quick and a long one stays sane,
#     4. opens the 2F-85 (0.0 to /robotiq_gripper/command_percent, confirmed on
#        /robotiq_gripper/position_percent; a confirm timeout only WARNS),
#     5. switches back to forward_position_controller.
#
# It ends with the bridges still PAUSED -- deliberately: nothing must chase the
# leader while the arm is being repositioned by a trajectory. To resume teleop,
# use the EEF GUI's ENGAGE (it chains eef_resume -> pos_scale -> eef_engage).
# HOLD THE GELLO TRIGGER OPEN before pressing Gripper Resume. In CONTINUOUS mode
# the gripper bridge ramps toward the LIVE trigger value over ~2 s, so a closed
# trigger closes the freshly opened gripper. In DISCRETE mode the pause resets
# the latch, so a resume with the trigger inside the dead band publishes nothing
# and the gripper stays open.
#
# Every take also gets a gripper_mode.json sidecar recording which of those two
# modes was in force while it was written -- grip_cmd is a continuous 0..1 value
# in one and a binary 0.0/1.0 endpoint in the other, and mixing the two silently
# shifts the statistics of any policy trained on them. The teleop bar shows the
# live latch state next to the bridge states.
#
# The button is disabled while a take is recording and while a GO HOME is
# already running; it is force-kept enabled between the two confirm clicks.
#
# Environment variables are exactly the ones gello_recorder_gui uses:
#   CAM1_SERIAL / CAM2_SERIAL   RealSense DEVICE serials (auto-verified on the
#                               USB bus at startup -- do not hand-edit; see
#                               src/gello_recorder/README.md)
#   COLOR_PROFILE               colour stream WxHxFPS, default 1280x720x30
#   CAMERA_WARMUP_S             seconds after each camera's first frame before
#                               Start un-greys, default 4.0
#   RECORDER_OUTPUT_ROOT        where take_* folders are created
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

exec ros2 run gello_recorder task_recorder_gui
