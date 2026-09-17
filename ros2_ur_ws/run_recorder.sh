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
#         depth.h5            <- both RealSense depth streams (CAMS=true and ENABLE_DEPTH
#                               not 0): one PNG 16UC1 mm frame per row + depth
#                               intrinsics + depth->color extrinsics (depth_writer.py)
#         metadata.json
#
# Usage:
#     ./run_recorder.sh                 # signal logging only (vectors.h5)
#     BAG=true ./run_recorder.sh        # ALSO capture a full `ros2 bag -a`
#     CAMS=true ./run_recorder.sh       # ALSO launch both RealSense cameras (-> cam1.mp4/cam2.mp4)
#     CAMS=true ENABLE_DEPTH=1 ./run_recorder.sh   # ...and record depth.h5 too (costly, see below)
#     RATE=200 ./run_recorder.sh        # synchronized-table sample rate (Hz)
#
# Camera env vars (only used when CAMS=true or CAMS=1):
#     CAM1_SERIAL    RealSense #1 serial (auto-resolved; default 143322071682, plain D435, SCENE)
#     CAM2_SERIAL    RealSense #2 serial (auto-resolved; default 143322072540, plain D435, WRIST)
#     CAM1_NAME      camera_name/namespace for #1 (default cam1)
#     CAM2_NAME      camera_name/namespace for #2 (default cam2)
#     COLOR_PROFILE  color WxHxFPS, same for both cameras (default 1280x720x30)
#     ENABLE_DEPTH   default 0 = RGB only. 1|true -> ALSO stream and record
#                    depth (-> depth.h5), at ~6 MB/s disk, +2 subscriptions per
#                    camera and ~+35 % recorder CPU. (Default ON for one day,
#                    2026-09-14, then reverted: that load starved the recorder's
#                    spin thread and back-dated every robot row by 0.9 s. Fixed,
#                    but still costly -- so it is opt-in per session.)
#     ALIGN_DEPTH    1|true -> align_depth.enable:=true (depth resampled onto the
#                    1280x720 color image; the node then records the
#                    aligned_depth_to_color topics). default 0: measured at ~49 %
#                    CPU per camera node with BOTH streams dropping to ~25 Hz, so
#                    it is opt-in only; align offline from the recorded
#                    intrinsics/extrinsics instead.
# The serials above are a PREFERENCE, not a requirement: resolve_serials()
# (shared with launch_cameras.sh, see _resolve_camera_serials.sh) checks them
# against what pyrealsense2 actually enumerates on the USB bus and falls back
# by model class -- or errors loudly -- if the configured pair isn't plugged
# in. See _resolve_camera_serials.sh for why this matters (binding to an
# absent serial does not fail loudly; the node just publishes nothing).
# The cameras are launched via realsense2_camera rs_launch.py. The recorder node
# subscribes to the color topics realsense2_camera publishes under
# /<CAM1_NAME>/<CAM1_NAME>/color/image_raw/compressed (and likewise CAM2) and
# writes them to cam1.mp4 / cam2.mp4 in the session folder. With depth on it
# also gets record_depth:=true plus the six depth topic params
# (/<cam>/<cam>/depth/image_rect_raw/compressedDepth, .../depth/camera_info,
# .../extrinsics/depth_to_color -- or the aligned_depth_to_color variants when
# ALIGN_DEPTH=1) and writes depth.h5.
#
# Stop with Ctrl-C (flushes + finalises metadata.json, and stops the bag + cameras).
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
if [ "${BAG}" = "true" ] || [ "${BAG}" = "1" ]; then
    echo "### BAG=true -> also recording ALL topics to ${SESSION}/rosbag"
    ros2 bag record -a -o "${SESSION}/rosbag" >/dev/null 2>&1 &
    BAG_PID=$!
fi

CAM1_PID=""
CAM2_PID=""
DEPTH_NODE_ARGS=()   # filled below only when CAMS=true and depth is on
if [ "${CAMS}" = "true" ] || [ "${CAMS}" = "1" ]; then
    # Serials are a PREFERENCE, not a requirement -- see _resolve_camera_serials.sh
    # (shared with launch_cameras.sh) for why: the two D435 "pairs" named across
    # this repo are the same two cameras under two different serial FIELDS
    # (serial_number vs asic_serial_number), and serial_no:= matches the former.
    # Binding a serial that does not resolve does NOT fail loudly -- the node
    # comes up, publishes nothing, and the recorder writes an empty cam*.mp4.
    # 2026-09-14 lab move: two PLAIN D435 bodies now (see launch_cameras.sh
    # for the measured port/serial table).  cam1=SCENE, cam2=WRIST.
    CAM1_SERIAL="${CAM1_SERIAL:-143322071682}"   # ASIC 143623022572
    CAM2_SERIAL="${CAM2_SERIAL:-143322072540}"   # ASIC 143523020769
    CAM1_NAME="${CAM1_NAME:-cam1}"
    CAM2_NAME="${CAM2_NAME:-cam2}"
    # Both cameras record at the same resolution/fps so the two MP4s line up
    # visually (D435 defaults to 640x480, D435IF to 1280x720 -- force both to
    # match). Override with COLOR_PROFILE=WxHxFPS if you ever need something else.
    COLOR_PROFILE="${COLOR_PROFILE:-1280x720x30}"
    # Depth is OPT-IN and OFF by default: RGB only unless ENABLE_DEPTH=1.
    # It was on by default for one day (2026-09-14) and that day cost a 54-take
    # corpus its timestamps -- two extra 30 Hz subscriptions per camera plus a
    # ~6 MB/s HDF5 write on the recorder's single rclpy spin thread dropped its
    # round rate to 60-69 Hz, and every faster topic was then read out of a full
    # queue (ur_joint_states 0.900 s late, tcp_pose/wrench ~0.45 s). That defect
    # is fixed (header stamps + depth-5 queues + a background frame writer +
    # a starvation watchdog, see gello_recorder/spin_health.py) but the COST is
    # not, so depth is now something a session asks for on purpose.
    # The cameras were never the problem: on the self-powered Genesys hub
    # 2x color+depth measured stable (30 Hz color / ~29 Hz depth / ~9 % CPU per
    # node). ALIGN_DEPTH=1 additionally turns on in-node depth->color alignment
    # -- opt-in only, it costs ~49 % CPU and ~25 Hz on both streams (measured);
    # the aligned element is emitted ONLY when depth is on (same rule as
    # gello_recorder_gui._realsense_argv).
    ENABLE_DEPTH="${ENABLE_DEPTH:-0}"
    ALIGN_DEPTH="${ALIGN_DEPTH:-0}"
    case "${ENABLE_DEPTH,,}" in 1|true|yes|on) DEPTH_ON=true; DEPTH_ARG="enable_depth:=true" ;; *) DEPTH_ON=false; DEPTH_ARG="enable_depth:=false" ;; esac
    case "${ALIGN_DEPTH,,}" in 1|true|yes|on) ALIGN_ON=true ;; *) ALIGN_ON=false ;; esac
    # Extra launch args + recorder params, filled only when depth is on. Bash
    # arrays so the depth-off launch line stays byte-identical to before.
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
    source "$SCRIPT_DIR/_resolve_camera_serials.sh"
    resolve_serials
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

# Stop the bag + cameras cleanly when the recorder node exits (Ctrl-C).
cleanup() {
    [ -n "${BAG_PID}" ] && kill -INT "${BAG_PID}" 2>/dev/null || true
    [ -n "${CAM1_PID}" ] && kill -INT "${CAM1_PID}" 2>/dev/null || true
    [ -n "${CAM2_PID}" ] && kill -INT "${CAM2_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# DEPTH_NODE_ARGS is empty unless CAMS=true and depth is on (record_depth then
# defaults to false in the node, so a signal-only run never opens depth.h5).
ros2 run gello_recorder gello_ur_recorder --ros-args \
    -p session_dir:="${SESSION}" \
    -p sample_rate_hz:="${RATE}" \
    "${DEPTH_NODE_ARGS[@]}"
