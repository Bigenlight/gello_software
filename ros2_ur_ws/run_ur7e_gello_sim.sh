#!/usr/bin/env bash
# Drive the MOCK UR7e in RViz2 from the PHYSICAL GELLO leader arm.
#
# This is the "simulation dry-run" of the real GELLO -> UR7e teleop: the real
# GELLO leader (serial /dev/ttyUSB0, FTBEO6QK) is read read-only and its joint
# angles command a ros2_control MOCK UR7e, visualised in RViz2. No real robot.
#
# GELLO is PASSIVE read-only: torque is never enabled on its Dynamixels.
#
# Usage:
#   ./run_ur7e_gello_sim.sh                     # with RViz2 (needs a DISPLAY)
#   ./run_ur7e_gello_sim.sh launch_rviz:=false  # headless (pipeline only)
#   DISPLAY=:1 ./run_ur7e_gello_sim.sh          # use a specific X display
set -e

export GELLO_REPO_ROOT=/home/laptop3/gello_software
source /opt/ros/humble/setup.bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash

exec ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=gello "$@"
