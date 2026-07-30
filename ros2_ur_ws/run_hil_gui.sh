#!/usr/bin/env bash
# PyQt5 operator GUI for the UR7e+GELLO HIL RL intervention deadman.
#
# This GUI is the GUI ALTERNATIVE to the terminal-focus spacebar deadman. It
# publishes /hil/deadman (std_msgs/Float32MultiArray, data=[engaged, gain]) at
# 20 Hz. It runs WITHOUT the teleop bridge and WITHOUT control_mode:=eef -- it
# never calls a bridge service and never commands the robot or GELLO directly.
#
# Run it ALONGSIDE the HIL RL loop, e.g.:
#     Terminal 1 (T3):  ./run_ur7e_gello_real.sh   # gello_publisher (the leader)
#     Terminal 2 (T4):  python3 serl_ur_infra/tests/run_rviz_hil.py --deadman topic
#     Terminal 3:       ./run_hil_gui.sh
#
# The GUI has the deadman toggle plus an episode START/NEXT button:
#   * click while DISENGAGED -> ENGAGE (two-click confirm; the arm WILL follow
#     the GELLO leader once you also move it).
#   * click while ENGAGED    -> DISENGAGE (single click; the RL policy resumes).
#   The sensitivity slider is the gain (0.10 fine .. 1.00 1:1); the env latches
#   it at the engage edge. The heartbeat publishes every tick (even disengaged)
#   so the env's staleness watchdog stays fed.
#   Actor reward/classifier/episode status arrives on /hil/actor_status. At
#   WAIT_SCENE_READY, START/NEXT releases the deadman and calls
#   /hil/scene_ready; it does not command the robot directly.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

exec ros2 run ur_gello_bringup gello_hil_gui
