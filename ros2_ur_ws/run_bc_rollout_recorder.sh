#!/usr/bin/env bash
# =============================================================================
# run_bc_rollout_recorder.sh — OPTIONAL Terminal 4 during a BC real-robot eval
# =============================================================================
#
# Records the ROBOT side of a BC rollout at native rates (by reusing the proven
# headless recorder, gello_ur_recorder) PLUS the HIL session-state timeline
# (_bc_rollout_status_logger.py), into ONE rollout folder:
#
#     <ROLLOUT_DIR>/
#         robot/vectors.h5      <- native-rate + synchronized tables:
#                                  gello_joint_states, ur_joint_states, command,
#                                  gripper, wrench, tcp_pose, cam1/cam2_frames
#         robot/cam1.mp4        <- only if camera frames arrive (see CAMERAS below)
#         robot/cam2.mp4
#         robot/metadata.json   <- finalised on graceful Ctrl-C
#         status.jsonl          <- session-state timeline (status logger)
#
# READ-ONLY: everything here only subscribes. Nothing commands the robot, the
# GELLO leader, or the gripper. Safe to start/stop mid-session.
#
# On Ctrl-C the recorder node sometimes prints an rclpy shutdown traceback
# ("failed to initialize wait set" / "[ros2run]: Process exited with failure 1").
# That is cosmetic and pre-existing (its main() catches KeyboardInterrupt, but
# rclpy's own handler may shut the context down first): the finalisation runs in
# main()'s finally: block either way -- verified, metadata.json says
# "finalized": true and the h5 tables are complete.
#
# CAMERAS — this script NEVER launches cameras (unlike run_recorder.sh's
# CAMS=true). During a HIL session run_hil_session.sh already owns the two
# RealSense nodes; launching them again would fight for the USB streaming lock.
# What the recorder node actually does (verified in gello_ur_recorder_node.py
# + recording_session.py + video_writer.py):
#   * it SUBSCRIBES to /cam1/cam1/color/image_raw/compressed and
#     /cam2/cam2/color/image_raw/compressed (params cam1_topic/cam2_topic) and
#     writes every frame it receives to cam1.mp4 / cam2.mp4 -- it does not know
#     or care WHO publishes them, so the cameras launched by run_hil_session.sh
#     (launch_cameras.sh, same default names cam1/cam2) are recorded fine.
#   * MP4 capture IS gated on two things, honestly stated:
#     (1) camera_warmup_s (default 3.0 s) drops every frame until 3 s after
#         THIS recorder's FIRST received frame, per camera -- so a rollout
#         shorter than ~3 s of camera traffic yields no video at all;
#     (2) the cv2 writer opens LAZILY on the first decoded frame, so if no
#         frames ever arrive NO cam*.mp4 file is created (not even an empty
#         one -- run_recorder.sh's header comment about an "empty cam*.mp4"
#         is wrong on this point). vectors.h5 is written regardless.
#   * if the session launched the cameras under non-default names, point the
#     recorder at them with CAM1_TOPIC=/... CAM2_TOPIC=/... (env, below).
#
# Env knobs:
#     BC_ROLLOUT_DIR   output folder (default gello_logs/bc_rollouts/rollout_<stamp>)
#     RATE             synchronized-table sample rate in Hz (default 100)
#     CAM1_TOPIC/CAM2_TOPIC   override the compressed color topics (optional)
#
# Usage:
#     ./run_bc_rollout_recorder.sh              # start recording, Ctrl-C to stop
#     BC_ROLLOUT_DIR=/tmp/foo ./run_bc_rollout_recorder.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# Same two setup files as run_recorder.sh; -u is relaxed only across them
# because the ROS setup scripts reference unbound vars.
set +u
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
set -u

RATE="${RATE:-100}"
# ROS2 param sample_rate_hz is a DOUBLE -> force a decimal so "100" isn't an INTEGER.
[[ "${RATE}" == *.* ]] || RATE="${RATE}.0"

ROLLOUT_DIR="${BC_ROLLOUT_DIR:-${GELLO_REPO_ROOT}/ros2_ur_ws/gello_logs/bc_rollouts/rollout_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${ROLLOUT_DIR}"

echo "============================================================"
echo "### BC ROLLOUT RECORDER"
echo "###   ROLLOUT_DIR -> ${ROLLOUT_DIR}"
echo "###   robot streams @ ${RATE} Hz  (READ-ONLY, launches no cameras)"
echo "============================================================"

RECORDER_ARGS=(-p session_dir:="${ROLLOUT_DIR}/robot" -p sample_rate_hz:="${RATE}")
if [ -n "${CAM1_TOPIC:-}" ]; then RECORDER_ARGS+=(-p cam1_topic:="${CAM1_TOPIC}"); fi
if [ -n "${CAM2_TOPIC:-}" ]; then RECORDER_ARGS+=(-p cam2_topic:="${CAM2_TOPIC}"); fi

# setsid: each child gets its OWN process group so this script owns the ONE stop
# path (see stop_child). `ros2 run` is a python wrapper that Popen()s the node and
# does NOT forward signals -- it only works because a terminal Ctrl-C hits the whole
# group. Signalling the wrapper alone would strand the node with an unfinalised h5.
# Trade-off: if THIS script is SIGKILLed the children keep recording as orphans.
setsid ros2 run gello_recorder gello_ur_recorder --ros-args "${RECORDER_ARGS[@]}" &
REC_PID=$!

STATUS_LOGGER="${SCRIPT_DIR}/_bc_rollout_status_logger.py"
LOG_PID=""
if [ -f "${STATUS_LOGGER}" ]; then
    setsid python3 "${STATUS_LOGGER}" --output "${ROLLOUT_DIR}" &
    LOG_PID=$!
else
    echo "[bc-rollout] WARNING: ${STATUS_LOGGER} missing -> no status.jsonl; robot recording continues."
fi

# Advisory only: never abort on it. Backgrounded so recording starts immediately.
(
    sleep 10
    pubs="$(timeout 10 ros2 topic info /hil/actor_status 2>/dev/null | awk '/Publisher count:/{print $3}' || true)"
    if [ "${pubs:-0}" -eq 0 ] 2>/dev/null; then
        echo "[bc-rollout] WARNING: /hil/actor_status has no publisher -- the HIL session"
        echo "[bc-rollout]          does not appear to be running. STILL RECORDING (robot-only)."
    fi
) &

echo "### recording: vectors.h5 robot streams (gello/ur joints, command, gripper,"
echo "###            wrench, tcp_pose), cam1.mp4/cam2.mp4 if the cameras are up,"
echo "###            and status.jsonl (session-state timeline)."
echo "### Ctrl-C to stop."

# SIGINT to the child's process GROUP is what finalises vectors.h5 + metadata.json
# (run_recorder.sh relies on the same graceful path: Ctrl-C -> destroy_node ->
# RecordingSession.close()). Measured on an idle-ish graph: group SIGINT ->
# "finalized": true; wrapper-only SIGINT would hang instead.
stop_child() {  # $1=pid (== pgid, thanks to setsid)  $2=label
    local pid="$1" label="$2" i
    [ -n "${pid}" ] || return 0
    kill -0 "${pid}" 2>/dev/null || return 0
    kill -INT -"${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
    for ((i = 0; i < 30; i++)); do   # ~15 s @ 0.5 s
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "${pid}" 2>/dev/null; then
        echo "[bc-rollout] WARNING: ${label} did not exit in 15 s -> SIGTERM"
        kill -TERM -"${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
}

finish() {  # $1 = exit code (0 = operator Ctrl-C)
    trap '' INT TERM
    echo ""
    echo "[bc-rollout] stopping (graceful, so the h5 is finalised)..."
    stop_child "${REC_PID}" "recorder"
    stop_child "${LOG_PID}" "status-logger"
    echo "[bc-rollout] saved -> ${ROLLOUT_DIR}"
    ls -lh "${ROLLOUT_DIR}" "${ROLLOUT_DIR}/robot" 2>/dev/null || true
    exit "${1:-0}"
}
trap 'finish 0' INT TERM

WAIT_PIDS=("${REC_PID}")
if [ -n "${LOG_PID}" ]; then WAIT_PIDS+=("${LOG_PID}"); fi
wait "${WAIT_PIDS[@]}" || true
# Reached only if a child exited on its own (crash / node shutdown), not on Ctrl-C.
echo "[bc-rollout] WARNING: a child exited on its own -- shutting the rollout down."
finish 1
