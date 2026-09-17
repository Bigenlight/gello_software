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

exec ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=gello "$@"
