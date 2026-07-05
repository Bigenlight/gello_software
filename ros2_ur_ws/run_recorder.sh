#!/usr/bin/env bash
# Diagnostic recorder for the GELLO -> UR7e teleop.
#
# Run this in a SECOND terminal WHILE the teleop (sim or real) is running. It
# logs GELLO joints + velocity, the bridge command, the UR7e ACTUAL joint
# position/velocity/effort, the gripper, and TCP force-torque / pose into a
# timestamped sub-folder:
#
#     ros2_ur_ws/gello_logs/session_<YYYYmmdd_HHMMSS>/
#         synchronized.csv   <- wide, time-aligned table (open in pandas/Excel)
#         gello_joint_states.csv  ur_joint_states.csv  command.csv
#         gripper.csv  wrench.csv  tcp_pose.csv  metadata.json
#
# Usage:
#     ./run_recorder.sh                 # CSV logging only
#     BAG=true ./run_recorder.sh        # ALSO capture a full `ros2 bag -a`
#     RATE=200 ./run_recorder.sh        # synchronized.csv sample rate (Hz)
#
# Stop with Ctrl-C (flushes + finalises metadata.json, and stops the bag).
# READ-ONLY: subscribes only; never commands the robot or GELLO.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

RATE="${RATE:-100}"
# ROS2 param sample_rate_hz is DOUBLE -> force a decimal so "100" isn't an INTEGER.
[[ "${RATE}" == *.* ]] || RATE="${RATE}.0"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${GELLO_REPO_ROOT}/ros2_ur_ws/gello_logs/session_${STAMP}"
mkdir -p "${SESSION}"

echo "### gello_ur_recorder -> ${SESSION}  (sample ${RATE} Hz)"

BAG_PID=""
if [ "${BAG}" = "true" ] || [ "${BAG}" = "1" ]; then
    echo "### BAG=true -> also recording ALL topics to ${SESSION}/rosbag"
    ros2 bag record -a -o "${SESSION}/rosbag" >/dev/null 2>&1 &
    BAG_PID=$!
fi

# Stop the bag cleanly when the recorder node exits (Ctrl-C).
cleanup() { [ -n "${BAG_PID}" ] && kill -INT "${BAG_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

ros2 run ur_gello_bringup gello_ur_recorder --ros-args \
    -p session_dir:="${SESSION}" \
    -p sample_rate_hz:="${RATE}"
