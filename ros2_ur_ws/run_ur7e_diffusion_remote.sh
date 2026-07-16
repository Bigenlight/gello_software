#!/usr/bin/env bash
# Run the ROS2/UR7e side against the Dockerized Diffusion service through SSH.
# This script never starts, stops, or restarts the GPU server container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="${SSH_HOST:-kanu}"
LOCAL_GRPC_PORT="${LOCAL_GRPC_PORT:-50051}"
REMOTE_GRPC_PORT="${REMOTE_GRPC_PORT:-50051}"
ROBOT_IP="${ROBOT_IP:-192.168.10.11}"
HEADLESS="${HEADLESS:-}"
CALIB="${CALIB:-}"
START_MODE="${START_MODE:-gello}"

for value in "$LOCAL_GRPC_PORT" "$REMOTE_GRPC_PORT"; do
    case "$value" in
        ''|*[!0-9]*) echo "ERROR: gRPC ports must be numeric." >&2; exit 1 ;;
    esac
done

case " $* " in *" headless_mode:=true "*|*"headless_mode:=true"*) HEADLESS=true ;; esac

# Binding is a reliable collision check even when `ss` is unavailable.
python3 - "$LOCAL_GRPC_PORT" <<'PY'
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError as exc:
    raise SystemExit(f"ERROR: local gRPC port {sys.argv[1]} is unavailable: {exc}")
finally:
    sock.close()
PY

TUNNEL_PID=""
cleanup() {
    if [ -n "$TUNNEL_PID" ]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "### Opening SSH tunnel: 127.0.0.1:${LOCAL_GRPC_PORT} -> ${SSH_HOST}:127.0.0.1:${REMOTE_GRPC_PORT}"
ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -L "127.0.0.1:${LOCAL_GRPC_PORT}:127.0.0.1:${REMOTE_GRPC_PORT}" \
    "$SSH_HOST" &
TUNNEL_PID=$!

# ExitOnForwardFailure covers bind/forward setup. Check both process liveness and
# that the forwarded local socket accepts connections before touching ROS/robot.
for _ in $(seq 1 20); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        wait "$TUNNEL_PID" || true
        echo "ERROR: SSH tunnel exited during startup." >&2
        exit 1
    fi
    if python3 - "$LOCAL_GRPC_PORT" <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.2):
        pass
except OSError:
    raise SystemExit(1)
PY
    then
        break
    fi
    sleep 0.25
done

if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "ERROR: SSH tunnel is not running." >&2
    exit 1
fi
if ! python3 - "$LOCAL_GRPC_PORT" <<'PY'
import socket
import sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1.0):
    pass
PY
then
    echo "ERROR: tunnel is alive but local gRPC port is not reachable." >&2
    exit 1
fi

source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

ARGS=(
    robot_ip:="$ROBOT_IP"
    start_mode:="$START_MODE"
    inference_transport:=grpc
    act_host:=127.0.0.1
    grpc_port:="$LOCAL_GRPC_PORT"
)
[ -n "$CALIB" ] && ARGS+=(kinematics_params_file:="$CALIB")
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    ARGS+=(headless_mode:=true)
else
    ARGS+=(headless_mode:=false)
fi

echo "### Tunnel ready (pid ${TUNNEL_PID}); GPU Docker lifecycle remains server-owned."
echo "### UR7e remains operator-gated; call ~/start_execution only after the handshake."
ros2 launch gello_policy ur7e_diffusion_real.launch.py "${ARGS[@]}" "$@"
