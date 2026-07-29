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
#     cam2 = WRIST (gripper-mounted)       -> right pane
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
#     CAM1_SERIAL    RealSense #1 serial (auto-resolved; default 147122072740, plain D435)
#     CAM2_SERIAL    RealSense #2 serial (auto-resolved; default 243222072700, D435IF)
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

# Serials are a PREFERENCE, not a requirement -- resolve_serials() below falls
# back to whatever is actually plugged in.
#
# The two "pairs" named across this repo are the SAME two cameras under two
# different serial FIELDS.  There was no hardware swap.  Measured 2026-07-29,
# same physical USB port reporting both values:
#
#   port    serial_number     asic_serial_number   device
#   4-4.1   147122072740      151623020789         D435    -> cam1
#   4-4.3   243222072700      322743060038         D435IF  -> cam2
#
# serial_no:= is matched against serial_number, NOT the ASIC serial:
#   rs.config().enable_device('151623020789')  ->  NO MATCH
#   rs.config().enable_device('147122072740')  ->  MATCHED
#
# The kernel USB descriptor (journalctl, /sys/bus/usb/devices/*/serial) exposes
# the ASIC serial.  So grepping the journal finds only 151623020789/322743060038
# and zero hits for 147122072740/243222072700 -- that means the kernel prints a
# DIFFERENT FIELD, not that those cameras were never present.  607e541 read that
# grep as a hardware swap and set these defaults to the ASIC serials, which
# realsense2_camera can never resolve.
#
# Binding a serial that does not resolve does NOT fail loudly -- the node starts,
# ros2 topic info even reports Publisher count 1, and it publishes nothing.  That
# silence is why this is auto-detected rather than hardcoded.
#
# Mount assignment is by model class: plain D435 -> cam1 (SCENE), D435IF/D435i
# -> cam2 (WRIST, gripper-mounted).  Model class is the only evidence tying each
# unit to its mount, so confirm with one arm jog: cam2 is the WRIST camera, so
# its background must sweep while the gripper fingers stay fixed in frame.
CAM1_SERIAL="${CAM1_SERIAL:-147122072740}"
CAM2_SERIAL="${CAM2_SERIAL:-243222072700}"
CAM1_NAME="${CAM1_NAME:-cam1}"
CAM2_NAME="${CAM2_NAME:-cam2}"
COLOR_PROFILE="${COLOR_PROFILE:-1280x720x30}"
# Viewer is ON by default (opt-OUT, unlike run_recorder.sh's opt-IN CAMS): for
# ACT deploy, eyeballing cam1=scene / cam2=close-up is a safety-relevant check.
VIEW="${VIEW:-true}"

# --- Resolve serials against what is actually on the USB bus ------------------
# resolve_serials() is shared with run_recorder.sh -- see
# _resolve_camera_serials.sh for the full behaviour description (model-class
# matching, ambiguous fallback, hard error below 2 devices, etc). Enumeration
# does NOT open a streaming lock, so this is safe to run even while another
# process holds the cameras (it just re-reports the same devices).
source "$SCRIPT_DIR/_resolve_camera_serials.sh"
resolve_serials

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
VIEWER_PID=""
VIEWER_LAUNCHED=false
CLEANED=false

# Stop the viewer + both cameras cleanly when this script exits (Ctrl-C).
# IMPROVEMENT over run_recorder.sh: after signalling each PID we wait (BOUNDED,
# see _kill_and_wait below) so the "stopped" line prints only once the process
# has ACTUALLY exited. Guarded so the EXIT+INT double-fire runs it once.
#
# IMPORTANT: the viewer's PID is signalled EXPLICITLY here (not left to implicit
# signal propagation to the foreground child) -- relying on the terminal/timeout
# forwarding SIGINT/SIGTERM to whatever's currently in the foreground is a race
# that sometimes left it (and the cameras it hadn't gotten to shut down yet)
# orphaned. Killing VIEWER_PID directly removes that race.
#
# IMPORTANT #2: `wait` on a plain `kill -INT` is NOT bounded -- if a process
# (observed: the rclpy+cv2 viewer, occasionally, likely a signal/GUI-event-loop
# interaction) doesn't actually exit on SIGINT, an unbounded `wait` here would
# hang cleanup() forever, which would also strand the cameras (their kill/wait
# never even runs). _kill_and_wait() gives each process a grace period, then
# escalates to SIGKILL so cleanup ALWAYS completes.
_kill_and_wait() {
    local pid="$1" name="$2" grace_s="${3:-5}"
    [ -z "${pid}" ] && return 0
    kill -INT "${pid}" 2>/dev/null || true
    local waited=0
    while kill -0 "${pid}" 2>/dev/null; do
        if [ "${waited}" -ge "${grace_s}" ]; then
            echo "###   ${name} (pid ${pid}) didn't exit in ${grace_s}s -- SIGKILL"
            kill -9 "${pid}" 2>/dev/null || true
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    wait "${pid}" 2>/dev/null || true
    echo "###   ${name} (pid ${pid}) stopped"
}

cleanup() {
    [ "${CLEANED}" = "true" ] && return 0
    CLEANED=true
    echo "### Shutting down ..."
    if [ -n "${VIEWER_PID}" ]; then
        _kill_and_wait "${VIEWER_PID}" "viewer"
    fi
    _kill_and_wait "${CAM1_PID}" "cam1"
    _kill_and_wait "${CAM2_PID}" "cam2"
    # Backstop: the `ros2 launch` wrapper PIDs above are what we track, but if
    # one of them was ever killed too abruptly to gracefully cascade to ITS
    # child (the actual realsense2_camera_node) -- observed once during
    # testing -- that grandchild is orphaned and keeps streaming/holding the
    # USB device. Sweep for it by name+namespace so a stray one never survives
    # this script even if the graceful path above didn't reach it.
    for ns in "${CAM1_NAME}" "${CAM2_NAME}"; do
        pkill -9 -f "realsense2_camera_node.*__ns:=/${ns}(\$| )" 2>/dev/null || true
    done
    echo "### All camera processes cleaned up — nothing left to kill manually."
}
trap cleanup EXIT INT TERM

echo "### launch_cameras.sh — RealSense pair for ACT deploy"
echo "###   cam1 = SCENE (tripod, 3rd person)  D435    serial ${CAM1_SERIAL}"
echo "###   cam2 = WRIST (gripper-mounted)     D435if  serial ${CAM2_SERIAL}"
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
    echo "### CHECK NOW: left pane must show the WHOLE SCENE, right pane the WRIST (gripper-mounted)."
    echo "### If they look swapped, Ctrl-C and re-check camera serials before deploying."
    echo "###"
    echo "### Keep this terminal open. Press Ctrl-C HERE to stop both cameras cleanly."
    # Backgrounded (not exec'd/plain-foreground) so cleanup() can always signal
    # VIEWER_PID directly instead of relying on the terminal/timeout forwarding
    # the signal to whatever's currently in the foreground -- see the note on
    # cleanup() above. `wait` below still blocks the script exactly like a plain
    # foreground run would, until the window closes or Ctrl-C fires the trap.
    VIEWER_LAUNCHED=true
    python3 "$SCRIPT_DIR/camera_viewer.py" \
        --cam1-topic "${CAM1_TOPIC}" \
        --cam2-topic "${CAM2_TOPIC}" \
        --cam1-label "cam1 - SCENE - ${CAM1_SERIAL}" \
        --cam2-label "cam2 - WRIST - ${CAM2_SERIAL}" &
    VIEWER_PID=$!
    wait "${VIEWER_PID}"
fi
