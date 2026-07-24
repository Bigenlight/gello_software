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
# It only talks to the bridge's ~/eef_* services + /gello_gripper_bridge/{pause,
# resume} + the ~/eef/state topic + SetParameters; it never commands the robot
# or GELLO directly.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

exec ros2 run ur_gello_bringup gello_eef_gui
