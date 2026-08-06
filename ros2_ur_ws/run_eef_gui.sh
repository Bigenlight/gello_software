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
# commands the robot or GELLO directly.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

exec ros2 run ur_gello_bringup gello_eef_gui
