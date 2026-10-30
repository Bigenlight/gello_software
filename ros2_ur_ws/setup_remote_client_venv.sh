#!/usr/bin/env bash
# Create the small, robot-laptop-only gRPC compatibility environment.
# ROS packages remain supplied by the system Python installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/bin/python3}"
VENV_DIR="${REMOTE_CLIENT_VENV:-${SCRIPT_DIR}/.venv-remote-client}"
LOCK_FILE="${SCRIPT_DIR}/requirements-remote-client.lock"

if [ ! -x "$SYSTEM_PYTHON" ]; then
    echo "ERROR: system Python is not executable: ${SYSTEM_PYTHON}" >&2
    exit 1
fi

echo "### Creating remote-client environment: ${VENV_DIR}"
"$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
"${VENV_DIR}/bin/python" -m pip install --upgrade --requirement "$LOCK_FILE"

"${VENV_DIR}/bin/python" - <<'PY'
import grpc
if grpc.__version__ != "1.74.0":
    raise SystemExit(f"ERROR: expected grpcio 1.74.0, loaded {grpc.__version__}")
print(f"### Remote-client grpcio ready: {grpc.__version__}")
PY

echo "### Setup complete. Run ./run_ur7e_diffusion_remote.sh"
