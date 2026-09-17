#!/usr/bin/env bash
# T1 for the RViz HIL / fake-RL test: MOCK UR7e + RViz, ZERO real-robot risk.
#
# Brings up the official ur_control stack against MOCK ros2_control hardware via
# ur_control_fake_safe.launch.py (which strips the real-robot-only
# urscript_interface). Publishes /joint_states so the HIL env
# (serl_ur_infra/tests/run_rviz_hil.py) can drive the arm in RViz.
#
# WHY THIS WRAPPER EXISTS: the mock-hardware launch argument changed between
# Humble and Jazzy.  Select the spelling with GELLO_ROS_DISTRO so the default
# is safe on Jazzy while the Humble opt-in remains usable.
#
# The four terminals of the RViz HIL test:
#   T1:  ./run_mock_rviz.sh
#   T2:  ros2 run gello_policy fake_diffusion_observations
#   T3:  GELLO_REPO_ROOT=$HOME/gello_software ros2 run ur_gello_bringup gello_publisher \
#          --ros-args --params-file \
#          "$PWD/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml"
#   T4:  cd ../serl_ur_infra && python3 tests/run_rviz_hil.py --deadman topic
#
# Nothing here touches the real robot; the arm you see is the RViz mock only.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GELLO_ROS_DISTRO="${GELLO_ROS_DISTRO:-jazzy}"
source "/opt/ros/${GELLO_ROS_DISTRO}/setup.bash"
source "$SCRIPT_DIR/install/setup.bash"

case "$GELLO_ROS_DISTRO" in
    humble) MOCK_HARDWARE_ARG=use_fake_hardware ;;
    *) MOCK_HARDWARE_ARG=use_mock_hardware ;;
esac

echo "### MOCK UR7e + RViz (${MOCK_HARDWARE_ARG}:=true) — no real robot, no 0.0.0.0 connect."
echo "### RViz view = rviz/hil_operator_view.rviz (camera on the OPERATOR's side)."
echo "### The stock ur_description view orbits from the opposite azimuth (~180 about Z),"
echo "### which makes a CORRECT base-frame arm motion LOOK X/Y-reversed while Z stays"
echo "### right. The control code is verified true base-frame — only the camera is"
echo "### rotated. NEVER negate X/Y in code to 'fix' this: it corrupts the recorded"
echo "### intervene_action (SERL buffer). See serl_ur_infra wrappers.py."

# The official ur_control launch HARDCODES ur_description's view_robot.rviz with
# no override arg, so we run the stack headless (launch_rviz:=false) and start
# RViz ourselves with the operator-oriented view, tied to this script's life.
rviz2 -d "$SCRIPT_DIR/rviz/hil_operator_view.rviz" &
RVIZ_PID=$!
cleanup() { kill "$RVIZ_PID" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

ros2 launch gello_policy ur_control_fake_safe.launch.py \
    ur_type:=ur7e robot_ip:=0.0.0.0 "${MOCK_HARDWARE_ARG}:=true" \
    initial_joint_controller:=forward_position_controller launch_rviz:=false "$@"
