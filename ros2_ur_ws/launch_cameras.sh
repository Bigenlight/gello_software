#!/usr/bin/env bash
# Launch the RealSense camera PAIR used by the GELLO -> UR7e ACT-deploy workflow,
# with a clean single-Ctrl-C shutdown and (by default) a live dual-camera viewer.
#
# Run this in its OWN terminal BEFORE starting an ACT deploy. It brings up both
# RealSense cameras via realsense2_camera rs_launch.py, waits until BOTH color
# streams are actually flowing (~30 Hz), and then opens a side-by-side viewer so
# you can VISUALLY CONFIRM the camera mapping:
#
#     cam1 = SCENE   (tripod, 3rd person)  -> left pane
#     cam2 = CLOSE-UP (workspace)          -> right pane
#
# A swapped mapping silently degrades the trained ACT policy (the model was
# trained with a fixed cam1=scene / cam2=close-up convention), so the viewer is
# ON BY DEFAULT here — this is a documented safety-relevant check, see
# docs/ros2/GELLO_UR7E_ACT_DEPLOY.md §2. Set VIEW=false to skip the window.
#
# Camera launch logs land under /tmp/launch_cameras_<timestamp>/.
#
# Usage:
#     ./launch_cameras.sh                 # both cameras + live viewer (default)
#     VIEW=false ./launch_cameras.sh      # both cameras, NO viewer window
#
# ENV (same names/defaults as run_recorder.sh's camera section):
#     CAM1_SERIAL    RealSense #1 serial (default 147122072740, a plain D435)
#     CAM2_SERIAL    RealSense #2 serial (default 243222072700, a D435IF)
#     CAM1_NAME      camera_name/namespace for #1 (default cam1)
#     CAM2_NAME      camera_name/namespace for #2 (default cam2)
#     COLOR_PROFILE  color WxHxFPS, same for both cameras (default 1280x720x30)
#     VIEW           true|1 (default) -> open the dual-camera viewer;
#                    false|0          -> skip the viewer, just hold the cameras up
#
# Stop with a single Ctrl-C in THIS terminal: it stops the viewer + both cameras
# cleanly (no orphaned background processes left to kill by hand).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # = ros2_ur_ws

CAM1_SERIAL="${CAM1_SERIAL:-147122072740}"
CAM2_SERIAL="${CAM2_SERIAL:-243222072700}"
CAM1_NAME="${CAM1_NAME:-cam1}"
CAM2_NAME="${CAM2_NAME:-cam2}"
COLOR_PROFILE="${COLOR_PROFILE:-1280x720x30}"
# Viewer is ON by default (opt-OUT, unlike run_recorder.sh's opt-IN CAMS): for
# ACT deploy, eyeballing cam1=scene / cam2=close-up is a safety-relevant check.
VIEW="${VIEW:-true}"

# --- ROS2 Humble environment -------------------------------------------------
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

# --- Per-run temp dir for the launch logs -----------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
TMPDIR_RUN="/tmp/launch_cameras_${STAMP}"
mkdir -p "${TMPDIR_RUN}"

# Color topics realsense2_camera publishes under the chosen namespaces.
CAM1_TOPIC="/${CAM1_NAME}/${CAM1_NAME}/color/image_raw/compressed"
CAM2_TOPIC="/${CAM2_NAME}/${CAM2_NAME}/color/image_raw/compressed"

CAM1_PID=""
CAM2_PID=""
VIEWER_LAUNCHED=false
CLEANED=false

# Stop the viewer + both cameras cleanly when this script exits (Ctrl-C).
# IMPROVEMENT over run_recorder.sh: after signalling each PID we `wait` for it so
# the "stopped" line prints only once the process has ACTUALLY exited (not merely
# once the signal was sent). Guarded so the EXIT+INT double-fire runs it once.
cleanup() {
    [ "${CLEANED}" = "true" ] && return 0
    CLEANED=true
    echo "### Shutting down ..."
    if [ "${VIEW}" != "false" ] && [ "${VIEW}" != "0" ] && [ "${VIEWER_LAUNCHED}" = "true" ]; then
        echo "###   viewer closed"
    fi
    if [ -n "${CAM1_PID}" ]; then
        kill -INT "${CAM1_PID}" 2>/dev/null || true
        wait "${CAM1_PID}" 2>/dev/null || true
        echo "###   cam1 (pid ${CAM1_PID}) stopped"
    fi
    if [ -n "${CAM2_PID}" ]; then
        kill -INT "${CAM2_PID}" 2>/dev/null || true
        wait "${CAM2_PID}" 2>/dev/null || true
        echo "###   cam2 (pid ${CAM2_PID}) stopped"
    fi
    echo "### All camera processes cleaned up — nothing left to kill manually."
}
trap cleanup EXIT INT TERM

echo "### launch_cameras.sh — RealSense pair for ACT deploy"
echo "###   cam1 = SCENE (tripod, 3rd person)  D435    serial ${CAM1_SERIAL}"
echo "###   cam2 = CLOSE-UP (workspace)        D435if  serial ${CAM2_SERIAL}"
echo "###   profile ${COLOR_PROFILE} | logs -> ${TMPDIR_RUN}/cam1_launch.log, cam2_launch.log"

# --- Launch both cameras backgrounded ----------------------------------------
# NOTE: serial_no / color_profile MUST be wrapped in embedded single-quotes
# ('"'"'...'"'"') -- ros2 launch infers CLI arg types from content, so a bare
# all-digit serial gets coerced to an integer and the node rejects it (serial_no
# is declared a STRING param and the node dies instantly otherwise).
echo -n "### Starting cam1 ... "
ros2 launch realsense2_camera rs_launch.py \
    camera_name:="${CAM1_NAME}" camera_namespace:="${CAM1_NAME}" "serial_no:='${CAM1_SERIAL}'" \
    "rgb_camera.color_profile:='${COLOR_PROFILE}'" \
    > "${TMPDIR_RUN}/cam1_launch.log" 2>&1 &
CAM1_PID=$!
echo "started (pid ${CAM1_PID})"

echo -n "### Starting cam2 ... "
ros2 launch realsense2_camera rs_launch.py \
    camera_name:="${CAM2_NAME}" camera_namespace:="${CAM2_NAME}" "serial_no:='${CAM2_SERIAL}'" \
    "rgb_camera.color_profile:='${COLOR_PROFILE}'" \
    > "${TMPDIR_RUN}/cam2_launch.log" 2>&1 &
CAM2_PID=$!
echo "started (pid ${CAM2_PID})"

echo "### Waiting for both streams (up to 30s) ..."

# Block until <topic> is flowing at ~25+ Hz, or fail after ~30s. Uses a short
# per-sample timeout on `ros2 topic hz` (which otherwise runs forever) and polls
# ~every second within a 30s deadline. Prints the measured rate on success.
wait_for_stream() {
    local name="$1" topic="$2" pid="$3" logf="$4"
    local deadline hz
    deadline=$(( $(date +%s) + 30 ))
    while [ "$(date +%s)" -lt "${deadline}" ]; do
        # The camera process dying is a hard failure — don't keep polling a corpse.
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo ""
            echo "### FAILED — ${name} launch process (pid ${pid}) exited during startup." >&2
            echo "###   See ${logf} for the cause (bad serial? camera unplugged?)." >&2
            return 1
        fi
        # Sample the publish rate with a short timeout; grep the average rate.
        hz=$(timeout 3 ros2 topic hz "${topic}" --window 5 2>/dev/null \
             | grep -oP 'average rate:\s*\K[0-9.]+' | tail -1 || true)
        if [ -n "${hz}" ] && awk -v h="${hz}" 'BEGIN { exit !(h >= 25) }'; then
            echo "###   ${name}: ${hz} Hz  OK"
            return 0
        fi
        sleep 1
    done
    echo ""
    echo "### FAILED — ${name} never reached ~25 Hz on ${topic} within 30s." >&2
    echo "###   See ${logf} for the cause (bad serial? camera unplugged? USB bandwidth?)." >&2
    return 1
}

if ! wait_for_stream "cam1" "${CAM1_TOPIC}" "${CAM1_PID}" "${TMPDIR_RUN}/cam1_launch.log"; then
    exit 1
fi
if ! wait_for_stream "cam2" "${CAM2_TOPIC}" "${CAM2_PID}" "${TMPDIR_RUN}/cam2_launch.log"; then
    exit 1
fi

# --- Ready: hand off to the viewer (or just hold the cameras up) -------------
if [ "${VIEW}" = "false" ] || [ "${VIEW}" = "0" ]; then
    echo "### READY — both cameras streaming at ~30 Hz."
    echo "###"
    echo "### Keep this terminal open. Press Ctrl-C HERE to stop both cameras cleanly."
    # No viewer: block so the backgrounded camera launches stay alive until Ctrl-C.
    wait "${CAM1_PID}" "${CAM2_PID}"
else
    echo "### READY — both cameras streaming at ~30 Hz. Opening viewer window ..."
    echo "###"
    echo "### CHECK NOW: left pane must show the WHOLE SCENE, right pane the CLOSE-UP."
    echo "### If they look swapped, Ctrl-C and re-check camera serials before deploying."
    echo "###"
    echo "### Keep this terminal open. Press Ctrl-C HERE to stop both cameras cleanly."
    # Viewer runs in the FOREGROUND — it keeps the script alive and blocks until
    # the user closes the window or Ctrl-C's. Then the EXIT trap tears everything
    # down. (camera_viewer.py is provided separately; see its --help for the CLI.)
    VIEWER_LAUNCHED=true
    python3 "$SCRIPT_DIR/camera_viewer.py" \
        --cam1-topic "${CAM1_TOPIC}" \
        --cam2-topic "${CAM2_TOPIC}" \
        --cam1-label "cam1 - SCENE - ${CAM1_SERIAL}" \
        --cam2-label "cam2 - CLOSE-UP - ${CAM2_SERIAL}"
fi
