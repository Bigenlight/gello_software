#!/usr/bin/env bash
# run_ur7e_gripper.sh — bring up ONLY the Robotiq 2F-85 gripper on the RWH UR7e.
#
# Path B (verified): Modbus RTU straight over the UR controller's RS485
# tool-communication TCP port (ROBOT_IP:54321). Requires:
#   * the UR "RS485 / tool communication" URCap installed on PolyScope (it is), and
#   * the robot POWERED ON (tool voltage is off when POWER_OFF -> gripper is dead).
#
# No socat / /tmp/ttyUR needed — the node speaks Modbus directly over TCP.
#
# Usage:
#   ./run_ur7e_gripper.sh                 # robot_ip defaults to 192.168.10.11
#   ROBOT_IP=192.168.10.11 ./run_ur7e_gripper.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_IP="${ROBOT_IP:-192.168.10.11}"

if [[ ! -f "$SCRIPT_DIR/install/setup.bash" ]]; then
  echo "ERROR: $SCRIPT_DIR/install/setup.bash not found — build first: ./build_ur7e.sh" >&2
  exit 1
fi

# ROS Humble's setup scripts read optional variables before defining them
# (AMENT_TRACE_SETUP_FILES, AMENT_PYTHON_EXECUTABLE, ...), so they abort under
# `set -u`. Keep strict nounset checking for the rest of this script and
# disable it only while the upstream/generated environment scripts are sourced.
# Same treatment as run_ur7e_diffusion_remote.sh.
set +u
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
set -u

echo "Bringing up gripper-only for UR7e at $ROBOT_IP (Modbus over tool-comm :54321)."
echo "Reminder: the robot must be POWERED ON for the gripper to answer."
echo ""
echo "Control it from another terminal:"
echo "  ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool \"{data: true}\"   # close"
echo "  ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool \"{data: false}\"  # open"
echo ""

exec ros2 launch ur_gello_bringup ur7e_gripper_only.launch.py robot_ip:="$ROBOT_IP"
