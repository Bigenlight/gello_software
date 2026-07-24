#!/usr/bin/env bash
# T1 for the RViz HIL / fake-RL test: MOCK UR7e + RViz, ZERO real-robot risk.
#
# Brings up the official ur_control stack against MOCK ros2_control hardware via
# ur_control_fake_safe.launch.py (which strips the real-robot-only
# urscript_interface). Publishes /joint_states so the HIL env
# (serl_ur_infra/tests/run_rviz_hil.py) can drive the arm in RViz.
#
# WHY THIS WRAPPER EXISTS: this laptop's Humble ur_robot_driver uses
#   use_fake_hardware:=true    (NOT the Jazzy 'use_mock_hardware:=true').
# Passing use_mock_hardware here is SILENTLY IGNORED, so the real hardware
# interface loads and the driver tries to reach a real robot at robot_ip,
# failing forever with "Failed to connect to robot on IP 0.0.0.0:30001/30004".
# This wrapper bakes in the correct arg so that footgun can't recur.
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
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

echo "### MOCK UR7e + RViz (use_fake_hardware:=true) — no real robot, no 0.0.0.0 connect."
exec ros2 launch gello_policy ur_control_fake_safe.launch.py \
    ur_type:=ur7e robot_ip:=0.0.0.0 use_fake_hardware:=true \
    initial_joint_controller:=forward_position_controller launch_rviz:=true "$@"
