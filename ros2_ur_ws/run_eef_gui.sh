#!/usr/bin/env bash
# Mouse-style PyQt5 operator GUI for GELLO -> UR7e EEF (Cartesian delta) teleop.
#
# Run this in a SECOND terminal while the real EEF teleop is running with
# control_mode:=eef, e.g.:
#     Terminal 1:  HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef
#     Terminal 2:  ./run_eef_gui.sh
#
# The GUI is the mouse metaphor with ONE big toggle:
#   * click in HOLD/DISENGAGED  -> ENGAGE (two-click confirm). From DISENGAGED it
#     chains eef_resume -> pos_scale commit -> eef_engage automatically.
#   * click in ENGAGED          -> DISENGAGE (single click; the arm holds).
#   Every ENGAGE re-anchors at the current pose, so "off -> move GELLO -> on" is
#   the whole workflow -- there are NO reclutch/re-arm/to-joint buttons.
#   The sensitivity slider is pos_scale (DPI); it commits only at an engage/
#   disengage, never mid-stroke (A안). Gripper PAUSE/Resume ride their own guard.
#
# The gripper row also shows a LIVE DISCRETE-LATCH READOUT from
# /gello_gripper_bridge/discrete_state, with the raw trigger value from
# .../discrete_trigger beside it ("GRIP: CLOSED  trig 0.94"). Only meaningful
# with gripper_mode:=discrete on the teleop launch; it reads "continuous
# (discrete off)" otherwise:
#     OPEN / CLOSED  the trigger crossed a threshold and the output snapped there
#     UNKNOWN        no threshold crossed YET — not a fault, but if it never
#                    leaves UNKNOWN while you squeeze, a threshold is mis-set and
#                    the gripper has SILENTLY stopped responding
#     RAMPING        the bounded post-resume slew window — not a fault
#     trig N.NN      the value the latch is thresholding; "trig --" = that topic
#                    has been quiet for 2 s
#     in band N.Ns   THE ONE TO KNOW: the trigger has sat strictly between the two
#                    thresholds for >1.5 s, so it is crossing nothing. The latch
#                    then just repeats its last word — a normal-looking CLOSED
#                    while your hand is open — and the gripper has silently
#                    stopped responding. Amber, same as UNKNOWN.
#     no signal      nothing on the state topic for 2 s (bridge down / older one)
#
# It only talks to the bridge's ~/eef_* services + /gello_gripper_bridge/{pause,
# resume} + the ~/eef/state, ~/discrete_state and ~/discrete_trigger topics +
# SetParameters on the ARM bridge and a one-time read-only GetParameters on the
# gripper bridge (to learn the thresholds it flags against — if that read fails
# the trigger value is still shown, only the automatic flag goes away). It never
# commands the robot or GELLO directly -- with ONE exception, the GO TO START
# POSE button (two-click confirm). That button pauses BOTH bridges, hands the
# joints to scaled_joint_trajectory_controller, runs ONE FollowJointTrajectory
# to the start pose, opens the gripper, hands back to forward_position_controller,
# and NEVER auto-resumes teleop: the arm holds there until the operator
# re-ENGAGEs from the GUI. Its target pose comes from START_POSE_CONFIG below.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
GELLO_ROS_DISTRO="${GELLO_ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${GELLO_ROS_DISTRO}/setup.bash"
WORKSPACE_SETUP="$SCRIPT_DIR/install/setup.bash"
[[ -r "$ROS_SETUP" ]] || { echo "### ROS setup not found: $ROS_SETUP" >&2; exit 2; }
[[ -r "$WORKSPACE_SETUP" ]] || { echo "### Workspace is not built: $WORKSPACE_SETUP" >&2; exit 2; }
source "$ROS_SETUP"
source "$WORKSPACE_SETUP"
export PATH="/opt/ros/${GELLO_ROS_DISTRO}/bin:/usr/bin:/bin:${PATH}"

# START_POSE_CONFIG -- the deploy yaml the GO TO START POSE button reads its
# target from: `policy_leader_node.ros__parameters.start_pose` (6 rad, UR joint
# order) + `start_gripper` (0=open..1=closed). It is the SAME file the inference
# launch feeds policy_leader_node, so data collection starts from exactly the
# pose the policy will later be started from (the resume-align gate requires the
# live arm within ~0.1 rad of it). There is deliberately NO pose hard-coded in
# the GUI: banana and carrot start poses look alike (pan +3.106 vs -3.164 is
# ~2*pi apart in value) but differ by up to 0.33 rad on other joints, so a wrong
# default would park the arm at a plausible-looking wrong pose with no error.
#
# Default = the carrot_in_pot deploy config. It may be ABSENT on a fresh
# checkout (it is untracked while that work is in progress) -- then the button
# is simply DISABLED with the reason shown on screen; nothing else changes.
# To collect data for another task, point this at that task's deploy yaml:
#     START_POSE_CONFIG=src/gello_policy/config/act_deploy.yaml ./run_eef_gui.sh
# (banana: act/fm/diffusion_deploy.yaml all carry the same pose). A relative
# path is resolved against the cwd you ran this script from (`exec ros2 run`
# keeps it), so the example above assumes `cd ros2_ur_ws` first; an absolute
# path is safer.
export START_POSE_CONFIG="${START_POSE_CONFIG:-$SCRIPT_DIR/src/gello_policy/config/ifql_deploy.yaml}"

exec ros2 run ur_gello_bringup gello_eef_gui
