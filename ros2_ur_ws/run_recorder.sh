#!/usr/bin/env bash
# Diagnostic recorder for the GELLO -> UR7e teleop.
#
# Run this in a SECOND terminal WHILE the teleop (sim or real) is running. It
# logs GELLO joints + velocity, the bridge command, the UR7e ACTUAL joint
# position/velocity/effort, the gripper, and TCP force-torque / pose into a
# timestamped sub-folder:
#
#     ros2_ur_ws/gello_logs/session_<YYYYmmdd_HHMMSS>/
#         vectors.h5         <- ONE HDF5 file with all the vector-signal tables
#                               (synchronized + gello_joint_states, ur_joint_states,
#                               command, gripper, wrench, tcp_pose, and the
#                               cam1_frames/cam2_frames frame-index tables); this
#                               replaces the old per-signal CSVs.
#         cam1.mp4  cam2.mp4  <- the two RealSense color streams (only when CAMS=true)
#         metadata.json
#
# Usage:
#     ./run_recorder.sh                 # signal logging only (vectors.h5)
#     BAG=true ./run_recorder.sh        # ALSO capture a full `ros2 bag -a`
#     CAMS=true ./run_recorder.sh       # ALSO launch both RealSense cameras (-> cam1.mp4/cam2.mp4)
#     RATE=200 ./run_recorder.sh        # synchronized-table sample rate (Hz)
#
# Camera env vars (only used when CAMS=true or CAMS=1):
#     CAM1_SERIAL    RealSense #1 serial (default 147122072740, a plain D435)
#     CAM2_SERIAL    RealSense #2 serial (default 243222072700, a D435IF)
#     CAM1_NAME      camera_name/namespace for #1 (default cam1)
#     CAM2_NAME      camera_name/namespace for #2 (default cam2)
#     COLOR_PROFILE  color WxHxFPS, same for both cameras (default 1280x720x30)
# The cameras are launched via realsense2_camera rs_launch.py. The recorder node
# subscribes to the color topics realsense2_camera publishes under
# /<CAM1_NAME>/<CAM1_NAME>/color/image_raw/compressed (and likewise CAM2) and
# writes them to cam1.mp4 / cam2.mp4 in the session folder.
#
# Stop with Ctrl-C (flushes + finalises metadata.json, and stops the bag + cameras).
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

CAM1_PID=""
CAM2_PID=""
if [ "${CAMS}" = "true" ] || [ "${CAMS}" = "1" ]; then
    CAM1_SERIAL="${CAM1_SERIAL:-147122072740}"
    CAM2_SERIAL="${CAM2_SERIAL:-243222072700}"
    CAM1_NAME="${CAM1_NAME:-cam1}"
    CAM2_NAME="${CAM2_NAME:-cam2}"
    # Both cameras record at the same resolution/fps so the two MP4s line up
    # visually (D435 defaults to 640x480, D435IF to 1280x720 -- force both to
    # match). Override with COLOR_PROFILE=WxHxFPS if you ever need something else.
    COLOR_PROFILE="${COLOR_PROFILE:-1280x720x30}"
    echo "### CAMS=true -> launching RealSense ${CAM1_NAME} (${CAM1_SERIAL}) + ${CAM2_NAME} (${CAM2_SERIAL}) @ ${COLOR_PROFILE}"
    echo "###   camera launch logs -> ${SESSION}/cam1_launch.log , ${SESSION}/cam2_launch.log"
    # NOTE: serial_no MUST be wrapped in embedded single-quotes ('"'"'...'"'"') --
    # ros2 launch infers CLI arg types from content, so a bare all-digit serial
    # gets coerced to an integer and the node rejects it (it declares serial_no
    # as a string parameter and throws/dies instantly otherwise).
    ros2 launch realsense2_camera rs_launch.py \
        camera_name:="${CAM1_NAME}" camera_namespace:="${CAM1_NAME}" "serial_no:='${CAM1_SERIAL}'" \
        "rgb_camera.color_profile:='${COLOR_PROFILE}'" \
        > "${SESSION}/cam1_launch.log" 2>&1 &
    CAM1_PID=$!
    ros2 launch realsense2_camera rs_launch.py \
        camera_name:="${CAM2_NAME}" camera_namespace:="${CAM2_NAME}" "serial_no:='${CAM2_SERIAL}'" \
        "rgb_camera.color_profile:='${COLOR_PROFILE}'" \
        > "${SESSION}/cam2_launch.log" 2>&1 &
    CAM2_PID=$!
fi

# Stop the bag + cameras cleanly when the recorder node exits (Ctrl-C).
cleanup() {
    [ -n "${BAG_PID}" ] && kill -INT "${BAG_PID}" 2>/dev/null || true
    [ -n "${CAM1_PID}" ] && kill -INT "${CAM1_PID}" 2>/dev/null || true
    [ -n "${CAM2_PID}" ] && kill -INT "${CAM2_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ros2 run gello_recorder gello_ur_recorder --ros-args \
    -p session_dir:="${SESSION}" \
    -p sample_rate_hz:="${RATE}"
