#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 ABSOLUTE_PARAMS_FILE [GRPC_HOST] [GRPC_PORT]" >&2
  exit 2
fi

params_file=$1
grpc_host=${2:-127.0.0.1}
grpc_port=${3:-50051}
if [[ $params_file != /* || ! -f $params_file ]]; then
  echo "ERROR: params file must be an existing absolute path: $params_file" >&2
  exit 2
fi
if ! [[ $grpc_port =~ ^[0-9]+$ ]] || (( grpc_port < 1 || grpc_port > 65535 )); then
  echo "ERROR: invalid gRPC port: $grpc_port" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(cd -- "$script_dir/../../.." && pwd)
system_python=${SYSTEM_PYTHON:-/usr/bin/python3}
remote_client_venv=${REMOTE_CLIENT_VENV:-"$workspace_dir/.venv-remote-client"}
roundtrip_smoke="$script_dir/../../../remote_diffusion_roundtrip_smoke.py"
if [[ ! -f $roundtrip_smoke ]]; then
  echo "ERROR: roundtrip smoke not found in source workspace: $roundtrip_smoke" >&2
  exit 2
fi
if [[ ! -x $remote_client_venv/bin/python ]]; then
  echo "ERROR: remote-client environment is missing: $remote_client_venv" >&2
  echo "Run $workspace_dir/setup_remote_client_venv.sh first." >&2
  exit 1
fi
if [[ ! -x $system_python ]]; then
  echo "ERROR: system Python is not executable: $system_python" >&2
  exit 1
fi

remote_client_site=$("$remote_client_venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export PYTHONPATH="${remote_client_site}${PYTHONPATH:+:${PYTHONPATH}}"

"$system_python" - <<'PY'
import grpc

expected = "1.74.0"
print(f"gRPC client preflight: grpcio={grpc.__version__}")
if grpc.__version__ != expected:
    raise SystemExit(
        f"ERROR: expected grpcio {expected}, loaded {grpc.__version__}; "
        "rerun setup_remote_client_venv.sh"
    )
PY

echo "Preflight: waiting for TCP endpoint $grpc_host:$grpc_port"
if ! timeout 5 bash -c 'exec 3<>/dev/tcp/$1/$2' _ "$grpc_host" "$grpc_port"; then
  echo "ERROR: generic policy server endpoint is unreachable: $grpc_host:$grpc_port" >&2
  exit 1
fi

echo "Inference gate: running one real gRPC roundtrip"
ROUNDTRIP_TARGET="$grpc_host:$grpc_port" \
  ROUNDTRIP_TIMEOUT_S="${ROUNDTRIP_TIMEOUT_S:-15}" \
  "$system_python" "$roundtrip_smoke"

echo "ROS gate: launching mock UR7e validation (server lifecycle remains external)"
exec ros2 launch gello_policy remote_policy_fake_ur7e_validation.launch.py \
  "grpc_host:=$grpc_host" \
  "grpc_port:=$grpc_port" \
  "params_file:=$params_file" \
  "launch_rviz:=${LAUNCH_RVIZ:-false}" \
  "validator_timeout_s:=${VALIDATOR_TIMEOUT_S:-45.0}" \
  "validator_tolerance_rad:=${VALIDATOR_TOLERANCE_RAD:-0.08}"
