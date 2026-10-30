#!/usr/bin/env bash
# Headless recorder for the GELLO -> UR7e teleop (no GUI, one session per run).
#
# Run this in a SECOND terminal WHILE the teleop (sim or real) is running. It
# records GELLO joints, the bridge command, the UR7e joint state, the gripper,
# TCP wrench / pose and (optionally) both RealSense cameras into
#
#     ros2_ur_ws/gello_logs/session_<YYYYmmdd_HHMMSS>/
#         vectors.h5           all vector tables (see src/gello_recorder/README.md)
#         cam1.mp4  cam2.mp4   color streams              (CAMS=true only)
#         depth.h5             depth frames + calibration (CAMS=true ENABLE_DEPTH=1)
#         metadata.json
#
# Usage:
#     ./run_recorder.sh                            # vectors.h5 only
#     CAMS=true ./run_recorder.sh                  # + both RealSense cameras
#     CAMS=true ENABLE_DEPTH=1 ./run_recorder.sh   # + depth.h5 (costly, opt-in)
#     BAG=true ./run_recorder.sh                   # + a full `ros2 bag record -a`
#     RATE=200 ./run_recorder.sh                   # synchronized-table rate (Hz, default 100)
#
# Camera env vars (only used with CAMS=true):
#     CAM1_SERIAL / CAM2_SERIAL  RealSense DEVICE serials. Default: camera_serials.env
#                                next to this script (scene camera = cam1, wrist = cam2).
#     CAM1_NAME / CAM2_NAME      camera_name/namespace (default cam1 / cam2)
#     COLOR_PROFILE              color WxHxFPS for both cameras (default 1280x720x30)
#     ENABLE_DEPTH               1|true -> also record depth.h5 (~6 MB/s disk, extra
#                                subscriptions and CPU; default 0 = RGB only)
#     ALIGN_DEPTH                1|true -> in-node depth->color alignment (~5x camera
#                                CPU, both streams drop to ~25 Hz; default 0 -- align
#                                offline from the recorded intrinsics/extrinsics)
#     CAMERA_ALLOW_SERIAL_FALLBACK=1  accept a different camera pair (temporary rigs)
#
# The serial pair is checked against `rs-enumerate-devices` before launch
# (_resolve_camera_serials.sh) because a camera node bound to an absent serial
# starts but publishes nothing.
#
# Stop with Ctrl-C (flushes + finalises metadata.json, stops the bag + cameras).
# READ-ONLY: subscribes only; never commands the robot or GELLO.
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

RATE="${RATE:-100}"
# ROS2 param sample_rate_hz is DOUBLE -> force a decimal so "100" isn't an INTEGER.
[[ "${RATE}" == *.* ]] || RATE="${RATE}.0"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${GELLO_REPO_ROOT}/ros2_ur_ws/gello_logs/session_${STAMP}"
mkdir -p "${SESSION}"

echo "### gello_ur_recorder -> ${SESSION}  (sample ${RATE} Hz)"

BAG_PID=""
CAM1_PID=""
CAM2_PID=""
DEPTH_NODE_ARGS=()   # filled below only when CAMS=true and depth is on

# Stop the bag + cameras cleanly when the recorder exits (including a camera
# preflight failure after BAG=true has already started the bag recorder).
cleanup() {
    [ -n "${BAG_PID}" ] && kill -INT "${BAG_PID}" 2>/dev/null || true
    [ -n "${CAM1_PID}" ] && kill -INT "${CAM1_PID}" 2>/dev/null || true
    [ -n "${CAM2_PID}" ] && kill -INT "${CAM2_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "${BAG}" = "true" ] || [ "${BAG}" = "1" ]; then
    echo "### BAG=true -> also recording ALL topics to ${SESSION}/rosbag"
    ros2 bag record -a -o "${SESSION}/rosbag" >/dev/null 2>&1 &
    BAG_PID=$!
fi

if [ "${CAMS}" = "true" ] || [ "${CAMS}" = "1" ]; then
    # Serial defaults come from camera_serials.env; exported values win.
    source "$SCRIPT_DIR/_resolve_camera_serials.sh"
    load_camera_serial_defaults
    CAM1_NAME="${CAM1_NAME:-cam1}"
    CAM2_NAME="${CAM2_NAME:-cam2}"
    # Same resolution/fps on both cameras so the two MP4s line up.
    COLOR_PROFILE="${COLOR_PROFILE:-1280x720x30}"
    # Depth / alignment are opt-in (see header). The aligned element is emitted
    # ONLY when depth is on (same rule as gello_recorder_gui._realsense_argv).
    ENABLE_DEPTH="${ENABLE_DEPTH:-0}"
    ALIGN_DEPTH="${ALIGN_DEPTH:-0}"
    case "${ENABLE_DEPTH,,}" in 1|true|yes|on) DEPTH_ON=true; DEPTH_ARG="enable_depth:=true" ;; *) DEPTH_ON=false; DEPTH_ARG="enable_depth:=false" ;; esac
    case "${ALIGN_DEPTH,,}" in 1|true|yes|on) ALIGN_ON=true ;; *) ALIGN_ON=false ;; esac
    # Extra launch args + recorder params, filled only when depth is on.
    DEPTH_LAUNCH_EXTRA=()
    DEPTH_NODE_ARGS=()
    if [ "${DEPTH_ON}" = "true" ]; then
        DEPTH_LAUNCH_EXTRA=("align_depth.enable:=${ALIGN_ON}")
        # Topic names must match gello_recorder.gello_gui_node.depth_topics_for.
        if [ "${ALIGN_ON}" = "true" ]; then
            D1_IMG="/${CAM1_NAME}/${CAM1_NAME}/aligned_depth_to_color/image_raw/compressedDepth"
            D2_IMG="/${CAM2_NAME}/${CAM2_NAME}/aligned_depth_to_color/image_raw/compressedDepth"
            D1_INFO="/${CAM1_NAME}/${CAM1_NAME}/aligned_depth_to_color/camera_info"
            D2_INFO="/${CAM2_NAME}/${CAM2_NAME}/aligned_depth_to_color/camera_info"
        else
            D1_IMG="/${CAM1_NAME}/${CAM1_NAME}/depth/image_rect_raw/compressedDepth"
            D2_IMG="/${CAM2_NAME}/${CAM2_NAME}/depth/image_rect_raw/compressedDepth"
            D1_INFO="/${CAM1_NAME}/${CAM1_NAME}/depth/camera_info"
            D2_INFO="/${CAM2_NAME}/${CAM2_NAME}/depth/camera_info"
        fi
        D1_EXT="/${CAM1_NAME}/${CAM1_NAME}/extrinsics/depth_to_color"
        D2_EXT="/${CAM2_NAME}/${CAM2_NAME}/extrinsics/depth_to_color"
        DEPTH_NODE_ARGS=(
            -p record_depth:=true
            -p "depth_aligned_to_color:=${ALIGN_ON}"
            -p "cam1_depth_topic:=${D1_IMG}"
            -p "cam2_depth_topic:=${D2_IMG}"
            -p "cam1_depth_info_topic:=${D1_INFO}"
            -p "cam2_depth_info_topic:=${D2_INFO}"
            -p "cam1_extrinsics_topic:=${D1_EXT}"
            -p "cam2_extrinsics_topic:=${D2_EXT}"
        )
    fi
    # Resolve against what is actually on the USB bus before launching anything
    # (enumeration takes no streaming lock, so this is safe here).
    resolve_serials

    if [ "${GELLO_ROS_DISTRO}" = "jazzy" ]; then
        # The RealSense image_transport publisher supplies the existing JPEG
        # endpoint directly. Check its plugin before either USB camera opens.
        if ! ros2 run image_transport list_transports 2>/dev/null | grep -q '"image_transport/compressed"'; then
            echo "FATAL: Jazzy image_transport compressed publisher plugin is unavailable." >&2
            echo "### Install ros-${GELLO_ROS_DISTRO}-compressed-image-transport, then retry." >&2
            exit 1
        fi
    fi

    echo "### CAMS=true -> launching RealSense ${CAM1_NAME} (${CAM1_SERIAL}) + ${CAM2_NAME} (${CAM2_SERIAL}) @ ${COLOR_PROFILE} | depth ${DEPTH_ARG#enable_depth:=} | aligned ${ALIGN_ON}"
    echo "###   camera launch logs -> ${SESSION}/cam1_launch.log , ${SESSION}/cam2_launch.log"
    # NOTE: serial_no MUST be wrapped in embedded single-quotes ('"'"'...'"'"') --
    # ros2 launch infers CLI arg types from content, so a bare all-digit serial
    # gets coerced to an integer and the node rejects it (it declares serial_no
    # as a string parameter and throws/dies instantly otherwise).
    # ${DEPTH_ARG} / align_depth.enable need NO such wrapping -- true/false are
    # not all-digit, so ros2 launch infers the bool the node already expects.
    ros2 launch realsense2_camera rs_launch.py \
        camera_name:="${CAM1_NAME}" camera_namespace:="${CAM1_NAME}" "serial_no:='${CAM1_SERIAL}'" \
        "rgb_camera.color_profile:='${COLOR_PROFILE}'" \
        "${DEPTH_ARG}" "${DEPTH_LAUNCH_EXTRA[@]}" \
        > "${SESSION}/cam1_launch.log" 2>&1 &
    CAM1_PID=$!
    ros2 launch realsense2_camera rs_launch.py \
        camera_name:="${CAM2_NAME}" camera_namespace:="${CAM2_NAME}" "serial_no:='${CAM2_SERIAL}'" \
        "rgb_camera.color_profile:='${COLOR_PROFILE}'" \
        "${DEPTH_ARG}" "${DEPTH_LAUNCH_EXTRA[@]}" \
        > "${SESSION}/cam2_launch.log" 2>&1 &
    CAM2_PID=$!
fi

# DEPTH_NODE_ARGS is empty unless CAMS=true and depth is on (record_depth then
# defaults to false in the node, so a signal-only run never opens depth.h5).
ros2 run gello_recorder gello_ur_recorder --ros-args \
    -p session_dir:="${SESSION}" \
    -p sample_rate_hz:="${RATE}" \
    "${DEPTH_NODE_ARGS[@]}"
