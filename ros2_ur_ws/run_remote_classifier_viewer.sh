#!/usr/bin/env bash
# Laptop: SSH tunnel + ROS camera client + read-only classifier viewer.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"
python3 -c 'import zmq, rclpy' || {
    echo "Missing python3-zmq/rclpy; install: sudo apt install python3-zmq" >&2
    exit 2
}
# The classifier checkpoint moved to junhyeong_ai with the learner on
# 2026-07-31; kanu still has its copy, so override to look there.
SSH_HOST="${CLASSIFIER_SSH_HOST:-junhyeong_ai}"
LOCAL_PORT="${CLASSIFIER_LOCAL_PORT:-5594}"
REMOTE_PORT="${CLASSIFIER_REMOTE_PORT:-5594}"
# Keep in sync with rlpd_receive_server.py's DEFAULT_REWARD_THRESHOLD.
THRESHOLD="${CLASSIFIER_THRESHOLD:-0.5}"
TIMEOUT_S="${CLASSIFIER_TIMEOUT_S:-2.0}"
cleanup() {
    for pid in "${GUI_PID:-}" "${CLIENT_PID:-}" "${TUNNEL_PID:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    for _ in {1..20}; do
        alive=0
        for pid in "${GUI_PID:-}" "${CLIENT_PID:-}" "${TUNNEL_PID:-}"; do
            [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1
        done
        [[ "$alive" -eq 0 ]] && break
        sleep 0.1
    done
    for pid in "${GUI_PID:-}" "${CLIENT_PID:-}" "${TUNNEL_PID:-}"; do
        [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$SSH_HOST" &
TUNNEL_PID=$!
sleep 1
kill -0 "$TUNNEL_PID" 2>/dev/null || {
    echo "SSH tunnel to ${SSH_HOST} failed." >&2
    exit 2
}
ros2 run gello_recorder remote_reward_classifier --ros-args \
    -p "endpoint:=tcp://127.0.0.1:${LOCAL_PORT}" \
    -p "threshold:=${THRESHOLD}" -p "timeout_s:=${TIMEOUT_S}" &
CLIENT_PID=$!
ros2 run gello_recorder classifier_view_gui &
GUI_PID=$!
set +e
wait -n "$CLIENT_PID" "$GUI_PID" "$TUNNEL_PID"
EXIT_CODE=$?
set -e
exit "$EXIT_CODE"
