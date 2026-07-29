#!/usr/bin/env bash
# =============================================================================
# run_hil_server.sh -- start/reuse the Kanu HIL-SERL learner and hold its tunnel
# =============================================================================
#
# Normal use (this terminal remains the tunnel owner):
#
#   ./run_hil_server.sh
#
# Read-only status, with no learner start and no tunnel:
#
#   ./run_hil_server.sh --check
#
# Select the physical Kanu GPU and the fresh run name used only if no healthy
# learner already exists:
#
#   ./run_hil_server.sh --gpu 6 --run-id cube_in_cup_real_20260729_220000
#
# Demand a new lineage instead of reusing a healthy learner.  This never stops
# the old learner: if one exists, the command refuses and asks the operator to
# stop it explicitly first.
#
#   ./run_hil_server.sh --new-lineage --gpu 6 --run-id my_fresh_run
#
# Ownership is intentionally narrow:
#
#   * A production learner is server-owned.  It is detached on Kanu and keeps
#     running when this script exits.
#   * This script owns only the SSH process it creates for
#       laptop 127.0.0.1:50153 -> Kanu 127.0.0.1:50053.
#   * Ctrl-C, TERM, tunnel failure, and local gRPC-probe failure clean up only
#     that SSH child.  They never signal a reused or newly started learner.
#
# Configuration overrides:
#
#   HIL_SSH_HOST             default kanu
#   HIL_GPU_INDEX            default 5 (new starts only; reuse reports actual)
#   HIL_RUN_ID               default cube_in_cup_real_<UTC timestamp>
#   HIL_START_TIMEOUT_S      default 300
#   HIL_LOCAL_PORT           default 50153
#   HIL_REMOTE_PORT          fixed production default 50053
#   HIL_KANU_REPO            default /home/junhyeong/gello_software_hil
#   HIL_KANU_PYTHON          default /home/junhyeong/miniconda3/envs/il/bin/python
#   ACTOR_VENV               default /home/laptop3/venvs/gello-hil-actor
#
# The artifact pins, feature-ring sizes, RAM reserve, reward contract and
# learner options below are the production command from
# serl_ur_infra/HIL_SERL_KANU_RUNBOOK_KO.md.  Do not turn this into a generic
# remote-process runner: strict reuse is what prevents an old/synthetic learner
# from silently receiving real robot transitions.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SSH_HOST="${HIL_SSH_HOST:-kanu}"
GPU_INDEX="${HIL_GPU_INDEX:-5}"
RUN_ID="${HIL_RUN_ID:-}"
START_TIMEOUT_S="${HIL_START_TIMEOUT_S:-300}"
LOCAL_PORT="${HIL_LOCAL_PORT:-50153}"
REMOTE_PORT="${HIL_REMOTE_PORT:-50053}"
KANU_REPO="${HIL_KANU_REPO:-/home/junhyeong/gello_software_hil}"
KANU_PYTHON="${HIL_KANU_PYTHON:-/home/junhyeong/miniconda3/envs/il/bin/python}"
REMOTE_RUN_BASE="/home/junhyeong/hil-serl-data/runs"
ACTOR_VENV="${ACTOR_VENV:-/home/laptop3/venvs/gello-hil-actor}"
ACTOR_PY="$ACTOR_VENV/bin/python"

MODE="start"
NEW_LINEAGE=0
RUN_ID_EXPLICIT=0
[[ -n "$RUN_ID" ]] && RUN_ID_EXPLICIT=1

usage() {
    sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" |
        sed '$d; s/^# \{0,1\}//'
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check|--status)
            MODE="check"
            shift
            ;;
        --new-lineage)
            NEW_LINEAGE=1
            shift
            ;;
        --gpu)
            need_value "$1" "${2:-}"
            GPU_INDEX="$2"
            shift 2
            ;;
        --gpu=*)
            GPU_INDEX="${1#*=}"
            shift
            ;;
        --run-id)
            need_value "$1" "${2:-}"
            RUN_ID="$2"
            RUN_ID_EXPLICIT=1
            shift 2
            ;;
        --run-id=*)
            RUN_ID="${1#*=}"
            RUN_ID_EXPLICIT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
[[ "$START_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || \
    die "HIL_START_TIMEOUT_S must be a positive integer"
(( START_TIMEOUT_S <= 1800 )) || die "HIL_START_TIMEOUT_S must be <= 1800"
for port_name in LOCAL_PORT REMOTE_PORT; do
    port_value="${!port_name}"
    [[ "$port_value" =~ ^[0-9]+$ ]] || die "$port_name must be numeric"
    (( port_value >= 1 && port_value <= 65535 )) || \
        die "$port_name must be in [1, 65535]"
done
[[ "$REMOTE_PORT" == "50053" ]] || \
    die "production Kanu learner port is fixed at 50053 (got $REMOTE_PORT)"
[[ -n "$SSH_HOST" ]] || die "HIL_SSH_HOST cannot be empty"

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="cube_in_cup_real_$(date -u +%Y%m%d_%H%M%S)"
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    die "run ID may contain only A-Z, a-z, 0-9, dot, underscore and dash"
(( ${#RUN_ID} <= 128 )) || die "run ID must be at most 128 characters"
if [[ "$MODE" == "check" && "$NEW_LINEAGE" == "1" ]]; then
    die "--new-lineage cannot be combined with --check"
fi

command -v ssh >/dev/null 2>&1 || die "ssh is not installed"
if [[ "$MODE" == "start" && ! -x "$ACTOR_PY" ]]; then
    die "actor venv python is missing: $ACTOR_PY"
fi

local_port_is_free() {
    /usr/bin/python3 - "$LOCAL_PORT" <<'PY'
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError as exc:
    raise SystemExit(f"local port {sys.argv[1]} is unavailable: {exc}")
finally:
    sock.close()
PY
}

# Check before any remote mutation.  A second check after the learner is ready
# closes the unavoidable race where another local process binds during Kanu's
# startup warm-up; even in that race the learner remains server-owned.
if [[ "$MODE" == "start" ]] && ! local_port_is_free; then
    die "local port $LOCAL_PORT is occupied; no remote learner operation was attempted"
fi

SSH_OPTIONS=(
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=60
    -o ServerAliveCountMax=3
    -o TCPKeepAlive=yes
)

# The remote program is passed on stdin.  In check mode it performs no mkdir,
# lock-file open, process start, signal, or other external mutation.
set +e
REMOTE_OUTPUT="$(
    ssh "${SSH_OPTIONS[@]}" "$SSH_HOST" bash -s -- \
        "$MODE" "$NEW_LINEAGE" "$GPU_INDEX" "$RUN_ID" \
        "$START_TIMEOUT_S" "$KANU_REPO" "$KANU_PYTHON" \
        "$REMOTE_PORT" "$REMOTE_RUN_BASE" <<'REMOTE_SCRIPT'
set -euo pipefail

MODE="$1"
NEW_LINEAGE="$2"
GPU_INDEX="$3"
RUN_ID="$4"
START_TIMEOUT_S="$5"
KANU_REPO="$6"
KANU_PYTHON="$7"
REMOTE_PORT="$8"
REMOTE_RUN_BASE="$9"

CLASSIFIER="/home/junhyeong/workspace/youngwoong/dataset/cube_in_cup_all3/classifier_ckpt/checkpoint_150"
CLASSIFIER_SHA256="512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d"
REAL_DEMO="/home/junhyeong/hil-serl-data/demos/cube_in_cup_20260720_success_23takes.pkl"
REAL_DEMO_SHA256="f97185582401ce7570d44fddc33d1bd64b215d7e32d6384d5fe13e1b405032fa"
RESNET_SOURCE="$KANU_REPO/third_party/hil-serl/examples/experiments/resnet10_params.pkl"
RESNET_SHA256="175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b"
REWARD_THRESHOLD="0.2"
REWARD_MODEL_ID="cube-in-cup-all3-ckpt150+sidecar-v1"
POLICY_MODEL_ID="hil-serl-hybrid-sac-resnet10-trunk-cache-v1"
OBSERVATION_SCHEMA_HASH="3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903"
RUN_LOCK="/home/junhyeong/hil-serl-data/.run_hil_server.lock"

remote_die() {
    echo "REMOTE ERROR: $*" >&2
    exit 1
}

validate_static_contract() {
    [[ -d "$KANU_REPO" ]] || remote_die "Kanu repo is missing: $KANU_REPO"
    [[ -x "$KANU_PYTHON" ]] || remote_die "Kanu Python is missing: $KANU_PYTHON"
    [[ -f "$KANU_REPO/serl_ur_infra/scripts/run_rlpd_learner_server.py" ]] || \
        remote_die "learner entrypoint is missing from $KANU_REPO"
    git -C "$KANU_REPO" merge-base --is-ancestor ca19652 HEAD || \
        remote_die "Kanu checkout does not contain required learner baseline ca19652"

    submodule_state="$(git -C "$KANU_REPO" submodule status third_party/hil-serl)"
    [[ "${submodule_state:0:1}" == " " ]] || \
        remote_die "third_party/hil-serl is missing or not at the recorded gitlink: $submodule_state"

    printf '%s  %s\n' "$REAL_DEMO_SHA256" "$REAL_DEMO" |
        sha256sum --check --strict --status || \
        remote_die "production demo SHA mismatch: $REAL_DEMO"
    printf '%s  %s\n' "$RESNET_SHA256" "$RESNET_SOURCE" |
        sha256sum --check --strict --status || \
        remote_die "ResNet-10 SHA mismatch: $RESNET_SOURCE"

    actual_classifier_sha="$(
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$KANU_REPO/serl_ur_infra" \
        "$KANU_PYTHON" - "$CLASSIFIER" <<'PY'
import sys
from ur_env.classifier_sidecar import directory_sha256
print(directory_sha256(sys.argv[1]))
PY
    )"
    [[ "$actual_classifier_sha" == "$CLASSIFIER_SHA256" ]] || \
        remote_die "classifier directory SHA mismatch: $actual_classifier_sha"

    defaults="$(
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$KANU_REPO/serl_ur_infra" \
        "$KANU_PYTHON" - <<'PY'
from ur_env.classifier_sidecar import CLASSIFIER_INPUT_ID
from ur_env.rlpd_receive_server import (
    DEFAULT_CLASSIFIER_CONFIRMATIONS,
    DEFAULT_REWARD_THRESHOLD,
)
print(DEFAULT_REWARD_THRESHOLD)
print(DEFAULT_CLASSIFIER_CONFIRMATIONS)
print(CLASSIFIER_INPUT_ID)
PY
    )"
    [[ "$defaults" == $'0.2\n1\nfullframe-jpeg-passthrough-v1' ]] || \
        remote_die "remote reward defaults no longer match the pinned production contract: $defaults"
}

list_learner_pids() {
    "$KANU_PYTHON" - <<'PY'
from pathlib import Path

result = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        argv = [
            part.decode("utf-8", "surrogateescape")
            for part in (entry / "cmdline").read_bytes().split(b"\0")
            if part
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(arg.endswith("/run_rlpd_learner_server.py") for arg in argv):
        result.append(int(entry.name))
for pid in sorted(result):
    print(pid)
PY
}

validate_process_contract() {
    local pid="$1"
    "$KANU_PYTHON" - "$pid" "$KANU_REPO" "$KANU_PYTHON" \
        "$REMOTE_PORT" "$CLASSIFIER" "$CLASSIFIER_SHA256" \
        "$REWARD_THRESHOLD" "$REWARD_MODEL_ID" "$REAL_DEMO" \
        "$RESNET_SOURCE" <<'PY'
import os
from pathlib import Path
import sys

(
    pid_text,
    repo,
    expected_python,
    port,
    classifier,
    classifier_sha,
    threshold,
    reward_model_id,
    demo,
    resnet_source,
) = sys.argv[1:]
pid = int(pid_text)
proc = Path("/proc") / str(pid)

try:
    argv = [
        part.decode("utf-8", "surrogateescape")
        for part in (proc / "cmdline").read_bytes().split(b"\0")
        if part
    ]
except FileNotFoundError:
    raise SystemExit(f"learner PID {pid} disappeared")

def values(flag):
    found = []
    for index, token in enumerate(argv):
        if token == flag:
            if index + 1 >= len(argv):
                raise SystemExit(f"{flag} has no value in PID {pid}")
            found.append(argv[index + 1])
    return found

def one(flag):
    found = values(flag)
    if len(found) != 1:
        raise SystemExit(f"PID {pid}: expected exactly one {flag}, got {found}")
    return found[0]

def exact(flag, expected):
    actual = one(flag)
    if actual != expected:
        raise SystemExit(
            f"PID {pid}: {flag}={actual!r}, expected production value {expected!r}"
        )

def number(flag, expected):
    actual = one(flag)
    try:
        matches = float(actual) == float(expected)
    except ValueError:
        matches = False
    if not matches:
        raise SystemExit(
            f"PID {pid}: {flag}={actual!r}, expected numeric value {expected!r}"
        )

if not any(arg.endswith("/run_rlpd_learner_server.py") for arg in argv):
    raise SystemExit(f"PID {pid} is not the learner entrypoint")
if os.path.realpath(argv[0]) != os.path.realpath(expected_python):
    raise SystemExit(f"PID {pid} uses unexpected Python: {argv[0]}")
if os.path.realpath(proc / "cwd") != os.path.realpath(repo):
    raise SystemExit(f"PID {pid} uses unexpected cwd: {os.path.realpath(proc / 'cwd')}")

for forbidden in (
    "--dry-run",
    "--synthetic-e2e",
    "--resume-path",
    "--resume-latest",
):
    if forbidden in argv:
        raise SystemExit(f"PID {pid} is not a fresh production learner: {forbidden}")

exact("--host", "127.0.0.1")
exact("--port", port)
exact("--classifier-checkpoint", classifier)
exact("--expected-classifier-sha256", classifier_sha)
number("--reward-threshold", threshold)
exact("--reward-model-id", reward_model_id)
number("--success-confirmations", "1")
if values("--demo-path") != [demo]:
    raise SystemExit(f"PID {pid}: production demo path mismatch: {values('--demo-path')}")
number("--checkpoint-reserve-gib", "2")
exact("--wandb-mode", "offline")
exact("--wandb-project", "hil-serl")
exact("--hil-serl-root", f"{repo}/third_party/hil-serl")
exact("--resnet-source", resnet_source)
number("--replay-capacity", "50000")
number("--intervention-capacity", "10000")
number("--feature-memory-reserve-gib", "2")
number("--demo-extraction-batch-size", "64")
number("--grasp-penalty", "-0.02")
number("--utd-ratio", "1")
number("--max-workers", "4")
number("--max-message-bytes", "16777216")
exact("--require-jax-backend", "gpu")
number("--target-learner-step", "5000")
number("--poll-interval", "0.1")

checkpoint_root = Path(one("--checkpoint-root"))
run_root = checkpoint_root.parent
expected_paths = {
    "--jsonl-path": run_root / "logs" / "learner.jsonl",
    "--memory-preflight-path": run_root / "logs" / "memory-preflight.jsonl",
    "--wandb-dir": run_root / "wandb",
    "--resnet-cache": run_root / "assets" / "resnet10_params.pkl",
}
if checkpoint_root.name != "checkpoints":
    raise SystemExit(f"PID {pid}: checkpoint root is not RUN_ROOT/checkpoints")
for flag, expected in expected_paths.items():
    if Path(one(flag)) != expected:
        raise SystemExit(f"PID {pid}: {flag} is outside the learner run root")
exact("--run-name", f"{run_root.name}-kanu-5000")

environment = {}
for item in (proc / "environ").read_bytes().split(b"\0"):
    if b"=" in item:
        key, value = item.split(b"=", 1)
        environment[key.decode("utf-8", "replace")] = value.decode(
            "utf-8", "replace"
        )
gpu = environment.get("CUDA_VISIBLE_DEVICES", "")
if not gpu.isdigit():
    raise SystemExit(f"PID {pid}: CUDA_VISIBLE_DEVICES is not one physical GPU: {gpu!r}")
if environment.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
    raise SystemExit(f"PID {pid}: XLA preallocation must be disabled")
expected_pythonpath = f"{repo}/serl_ur_infra:{repo}/third_party/hil-serl/serl_launcher"
if environment.get("PYTHONPATH") != expected_pythonpath:
    raise SystemExit(
        f"PID {pid}: PYTHONPATH does not match the production learner contract"
    )
if environment.get("WANDB_SILENT") != "true":
    raise SystemExit(f"PID {pid}: WANDB_SILENT must be true")
if environment.get("WANDB_DISABLE_CODE") != "true":
    raise SystemExit(f"PID {pid}: WANDB_DISABLE_CODE must be true")

print(f"HIL_SERVER_PID={pid}")
print(f"HIL_SERVER_RUN_ROOT={run_root}")
print(f"HIL_SERVER_GPU={gpu}")
PY
}

probe_server() {
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$KANU_REPO/serl_ur_infra" \
    timeout 12 "$KANU_PYTHON" - "$REMOTE_PORT" \
        "$POLICY_MODEL_ID" "$REWARD_MODEL_ID" "$OBSERVATION_SCHEMA_HASH" <<'PY'
from dataclasses import asdict
import json
import sys

from ur_env.grpc_actor_transport import GrpcActorNetwork

port, model_id, reward_model_id, schema_hash = sys.argv[1:]
network = GrpcActorNetwork(
    f"127.0.0.1:{port}",
    actor_id="run-hil-server-readonly-probe",
    action_shape=(7,),
    timeout_s=2.0,
    max_response_age_s=3.0,
    expected_observation_schema_hash=schema_hash,
    expected_model_id=model_id,
    expected_reward_authority="server_classifier",
    expected_reward_model_id=reward_model_id,
)
try:
    alive, ready, detail = network.health()
    if not alive or not ready:
        raise SystemExit(f"learner health is not ready: {alive=} {ready=} {detail=}")
    info = network.get_server_info()
    status = network.get_buffer_status()
    print(json.dumps({
        "health": detail,
        "server": asdict(info),
        "buffer": asdict(status),
    }, sort_keys=True, separators=(",", ":")))
finally:
    network.close()
PY
}

validate_ready_evidence() {
    local run_root="$1"
    "$KANU_PYTHON" - "$run_root" <<'PY'
import json
from pathlib import Path
import sys

run_root = Path(sys.argv[1])
jsonl_path = run_root / "logs" / "learner.jsonl"
if not jsonl_path.is_file():
    raise SystemExit(f"ready evidence is missing: {jsonl_path}")

raw = jsonl_path.read_bytes()
lines = raw.splitlines()
records = []
for index, line in enumerate(lines, start=1):
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        # An active JSONL writer can be observed between write(2) calls.  Only
        # an unterminated final fragment is retryable; earlier damage is not.
        if index == len(lines) and not raw.endswith(b"\n"):
            continue
        raise SystemExit(f"invalid learner JSONL at line {index}: {exc}")
    if not isinstance(record, dict):
        raise SystemExit(f"learner JSONL line {index} is not an object")
    records.append((index, record))

warmups = [item for item in records if item[1].get("event") == "learner_update_warmup_complete"]
ready = [item for item in records if item[1].get("event") == "learner_process_ready"]
if len(warmups) != 1 or len(ready) != 1:
    raise SystemExit(
        "fresh-lineage ready evidence must contain exactly one warmup and one "
        f"process-ready event (got warmup={len(warmups)}, ready={len(ready)})"
    )

warmup_line, warmup = warmups[0]
ready_line, process_ready = ready[0]
if warmup_line >= ready_line:
    raise SystemExit("learner process-ready event precedes its update warmup")
expected_warmup = {
    "warmup_outer_steps": 3,
    "warmup_gradient_updates": 6,
    "learner_step": 0,
    "gradient_step": 0,
    "policy_version": 0,
    "production_state_advanced": False,
}
for key, expected in expected_warmup.items():
    if warmup.get(key) != expected:
        raise SystemExit(
            f"warmup evidence {key}={warmup.get(key)!r}, expected {expected!r}"
        )
expected_ready = {
    "jax_backend": "gpu",
    "demo_count": 2037,
    "synthetic_acceptance_demo_count": 0,
    "restored_checkpoint": None,
    "learner_step": 0,
    "gradient_step": 0,
    "policy_version": 0,
}
for key, expected in expected_ready.items():
    if process_ready.get(key) != expected:
        raise SystemExit(
            f"process-ready evidence {key}={process_ready.get(key)!r}, "
            f"expected {expected!r}"
        )

print("HIL_SERVER_READY_EVIDENCE=" + json.dumps({
    "demo_count": process_ready["demo_count"],
    "jax_backend": process_ready["jax_backend"],
    "warmup_gradient_updates": warmup["warmup_gradient_updates"],
    "warmup_outer_steps": warmup["warmup_outer_steps"],
}, sort_keys=True, separators=(",", ":")))
PY
}

port_is_open() {
    "$KANU_PYTHON" - "$REMOTE_PORT" <<'PY'
import socket
import sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.4):
        pass
except OSError:
    raise SystemExit(1)
PY
}

inspect_existing() {
    local pids pid contract probe run_root
    mapfile -t pids < <(list_learner_pids)
    if (( ${#pids[@]} == 0 )); then
        if port_is_open; then
            remote_die "port $REMOTE_PORT is occupied by a non-learner process"
        fi
        return 2
    fi
    if (( ${#pids[@]} != 1 )); then
        remote_die "refusing duplicate learners; found PIDs: ${pids[*]}"
    fi
    pid="${pids[0]}"
    contract="$(validate_process_contract "$pid")" || return 1
    printf '%s\n' "$contract"
    if probe="$(probe_server 2>/dev/null)"; then
        run_root="$(printf '%s\n' "$contract" | sed -n 's/^HIL_SERVER_RUN_ROOT=//p')"
        [[ -n "$run_root" ]] || remote_die "learner contract omitted its run root"
        validate_ready_evidence "$run_root" || return 1
        printf 'HIL_SERVER_HEALTH=%s\n' "$probe"
        return 0
    fi
    return 3
}

validate_static_contract

if [[ "$MODE" == "check" ]]; then
    set +e
    inspect_existing
    status=$?
    set -e
    case "$status" in
        0)
            echo "HIL_SERVER_RESULT=healthy"
            exit 0
            ;;
        2)
            echo "HIL_SERVER_RESULT=absent"
            exit 3
            ;;
        3)
            echo "HIL_SERVER_RESULT=present_but_not_ready"
            exit 4
            ;;
        *)
            exit 1
            ;;
    esac
fi

# Start/reuse is serialized before checking for a process.  The learner itself
# also owns a per-lineage writer lock, but it binds gRPC only after expensive
# JAX warm-up; this outer lock prevents two launchers from doing that warm-up
# concurrently during the pre-bind window.
command -v flock >/dev/null 2>&1 || remote_die "flock is required on Kanu"
exec 9>"$RUN_LOCK"
flock -w 10 9 || remote_die "another run_hil_server launcher holds $RUN_LOCK"

set +e
existing_output="$(inspect_existing)"
existing_status=$?
set -e
if [[ "$existing_status" -eq 0 ]]; then
    printf '%s\n' "$existing_output"
    if [[ "$NEW_LINEAGE" == "1" ]]; then
        remote_die "--new-lineage refuses to reuse the active learner; stop it explicitly first"
    fi
    echo "HIL_SERVER_RESULT=reused"
    exit 0
fi
if [[ "$existing_status" -eq 1 ]]; then
    printf '%s\n' "$existing_output"
    remote_die "the existing learner does not match the production contract"
fi
if [[ "$existing_status" -eq 3 ]]; then
    printf '%s\n' "$existing_output"
    if [[ "$NEW_LINEAGE" == "1" ]]; then
        remote_die "--new-lineage refuses while a matching learner is initializing; stop it explicitly first"
    fi
    # A matching process may simply be in its 80-150 second startup warm-up.
    # Never launch a competitor; wait for this one under the launcher lock.
    initializing_pid="$(printf '%s\n' "$existing_output" | sed -n 's/^HIL_SERVER_PID=//p')"
    initializing_run_root="$(printf '%s\n' "$existing_output" | sed -n 's/^HIL_SERVER_RUN_ROOT=//p')"
    echo "Kanu learner PID $initializing_pid exists and is initializing; waiting for ready..." >&2
    deadline=$(( $(date +%s) + START_TIMEOUT_S ))
    while kill -0 "$initializing_pid" 2>/dev/null; do
        if probe="$(probe_server 2>/dev/null)"; then
            validate_ready_evidence "$initializing_run_root" || \
                remote_die "existing learner became healthy without valid fresh-lineage ready evidence"
            printf 'HIL_SERVER_HEALTH=%s\n' "$probe"
            echo "HIL_SERVER_RESULT=reused"
            exit 0
        fi
        if (( $(date +%s) >= deadline )); then
            remote_die "existing learner PID $initializing_pid did not become ready within ${START_TIMEOUT_S}s; it was not stopped"
        fi
        sleep 2
    done
    remote_die "existing learner PID $initializing_pid exited during initialization"
fi
[[ "$existing_status" -eq 2 ]] || remote_die "unexpected learner inspection status $existing_status"

dirty_checkout="$(git -C "$KANU_REPO" status --porcelain --untracked-files=normal)"
[[ -z "$dirty_checkout" ]] || \
    remote_die "Kanu checkout is dirty; refusing a new production learner: $dirty_checkout"

gpu_name="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader 2>/dev/null)" || \
    remote_die "physical GPU $GPU_INDEX does not exist"
gpu_pids="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
if printf '%s\n' "$gpu_pids" | grep -Eq '[0-9]'; then
    remote_die "physical GPU $GPU_INDEX is already used by PID(s): $(printf '%s' "$gpu_pids" | tr '\n' ' ')"
fi

memory_values="$(
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$KANU_REPO/serl_ur_infra" \
    "$KANU_PYTHON" - <<'PY'
from ur_env.learner import (
    estimate_feature_demo_memory,
    estimate_feature_replay_memory,
    system_available_memory_bytes,
)
replay = estimate_feature_replay_memory(
    replay_capacity=50_000,
    intervention_capacity=10_000,
).fixed_tensor_bytes
demo = estimate_feature_demo_memory(2_037).total_bytes
reserve = 2 * 1024**3
print(system_available_memory_bytes(), replay + demo + reserve)
PY
)"
read -r available_bytes required_bytes <<<"$memory_values"
(( available_bytes >= required_bytes )) || \
    remote_die "RAM preflight rejected before launch: available=$available_bytes required=$required_bytes"

run_root="$REMOTE_RUN_BASE/$RUN_ID"
[[ ! -e "$run_root" ]] || \
    remote_die "fresh lineage root already exists: $run_root (choose a new --run-id)"
mkdir -p "$REMOTE_RUN_BASE"
umask 077
mkdir "$run_root"
mkdir -p "$run_root/logs" "$run_root/wandb" "$run_root/assets"

checkpoint_root="$run_root/checkpoints"
jsonl_path="$run_root/logs/learner.jsonl"
memory_path="$run_root/logs/memory-preflight.jsonl"
wandb_dir="$run_root/wandb"
resnet_cache="$run_root/assets/resnet10_params.pkl"
stdout_path="$run_root/logs/stdout.log"

echo "Starting production learner on physical GPU $GPU_INDEX ($gpu_name)" >&2
echo "Fresh lineage: $run_root" >&2
cd "$KANU_REPO"
nohup env \
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_SILENT=true \
    WANDB_DISABLE_CODE=true \
    PYTHONPATH="$KANU_REPO/serl_ur_infra:$KANU_REPO/third_party/hil-serl/serl_launcher" \
    "$KANU_PYTHON" \
    serl_ur_infra/scripts/run_rlpd_learner_server.py \
    --host 127.0.0.1 \
    --port "$REMOTE_PORT" \
    --classifier-checkpoint "$CLASSIFIER" \
    --expected-classifier-sha256 "$CLASSIFIER_SHA256" \
    --reward-threshold "$REWARD_THRESHOLD" \
    --reward-model-id "$REWARD_MODEL_ID" \
    --success-confirmations 1 \
    --demo-path "$REAL_DEMO" \
    --checkpoint-root "$checkpoint_root" \
    --checkpoint-reserve-gib 2 \
    --jsonl-path "$jsonl_path" \
    --memory-preflight-path "$memory_path" \
    --wandb-dir "$wandb_dir" \
    --wandb-mode offline \
    --wandb-project hil-serl \
    --run-name "$RUN_ID-kanu-5000" \
    --hil-serl-root "$KANU_REPO/third_party/hil-serl" \
    --resnet-source "$RESNET_SOURCE" \
    --resnet-cache "$resnet_cache" \
    --replay-capacity 50000 \
    --intervention-capacity 10000 \
    --feature-memory-reserve-gib 2 \
    --demo-extraction-batch-size 64 \
    --grasp-penalty -0.02 \
    --utd-ratio 1 \
    --max-workers 4 \
    --max-message-bytes 16777216 \
    --require-jax-backend gpu \
    --target-learner-step 5000 \
    --poll-interval 0.1 \
    >"$stdout_path" 2>&1 </dev/null 9>&- &
learner_pid=$!

kill -0 "$learner_pid" 2>/dev/null || \
    remote_die "learner exited immediately; inspect $stdout_path"
contract=""
for _ in $(seq 1 50); do
    if contract="$(validate_process_contract "$learner_pid" 2>/dev/null)"; then
        break
    fi
    kill -0 "$learner_pid" 2>/dev/null || \
        remote_die "learner exited immediately; inspect $stdout_path"
    sleep 0.1
done
[[ -n "$contract" ]] || \
    remote_die "new learner PID $learner_pid did not expose its production command contract"
printf '%s\n' "$contract"
echo "HIL_SERVER_MEMORY_AVAILABLE_BYTES=$available_bytes"
echo "HIL_SERVER_MEMORY_REQUIRED_BYTES=$required_bytes"

deadline=$(( $(date +%s) + START_TIMEOUT_S ))
while kill -0 "$learner_pid" 2>/dev/null; do
    if probe="$(probe_server 2>/dev/null)"; then
        validate_ready_evidence "$run_root" || \
            remote_die "new learner became healthy without valid fresh-lineage ready evidence; it was not stopped"
        printf 'HIL_SERVER_HEALTH=%s\n' "$probe"
        echo "HIL_SERVER_RESULT=started"
        exit 0
    fi
    if (( $(date +%s) >= deadline )); then
        echo "Learner startup log tail:" >&2
        tail -n 30 "$stdout_path" >&2 || true
        remote_die "learner PID $learner_pid did not become ready within ${START_TIMEOUT_S}s; it remains server-owned and was not stopped"
    fi
    sleep 2
done

echo "Learner startup log tail:" >&2
tail -n 50 "$stdout_path" >&2 || true
if [[ -f "$memory_path" ]]; then
    echo "Learner RAM decisions:" >&2
    tail -n 2 "$memory_path" >&2 || true
fi
remote_die "learner PID $learner_pid exited before ready; evidence is preserved at $run_root"
REMOTE_SCRIPT
)"
REMOTE_RC=$?
set -e

printf '%s\n' "$REMOTE_OUTPUT"
if [[ "$REMOTE_RC" -ne 0 ]]; then
    if [[ "$MODE" == "check" && "$REMOTE_RC" -eq 3 ]]; then
        echo "No production learner is running on $SSH_HOST."
    fi
    exit "$REMOTE_RC"
fi

if [[ "$MODE" == "check" ]]; then
    echo "CHECK PASS: one exact production learner is healthy; no tunnel was opened."
    exit 0
fi

SERVER_RESULT="$(printf '%s\n' "$REMOTE_OUTPUT" | sed -n 's/^HIL_SERVER_RESULT=//p' | tail -n1)"
SERVER_PID="$(printf '%s\n' "$REMOTE_OUTPUT" | sed -n 's/^HIL_SERVER_PID=//p' | tail -n1)"
SERVER_RUN_ROOT="$(printf '%s\n' "$REMOTE_OUTPUT" | sed -n 's/^HIL_SERVER_RUN_ROOT=//p' | tail -n1)"
SERVER_GPU="$(printf '%s\n' "$REMOTE_OUTPUT" | sed -n 's/^HIL_SERVER_GPU=//p' | tail -n1)"
[[ "$SERVER_RESULT" == "started" || "$SERVER_RESULT" == "reused" ]] || \
    die "remote launcher returned an invalid result: $SERVER_RESULT"
[[ "$SERVER_PID" =~ ^[0-9]+$ ]] || die "remote launcher did not report a learner PID"
[[ "$SERVER_GPU" =~ ^[0-9]+$ ]] || die "remote launcher did not report a physical GPU"
[[ "$SERVER_RUN_ROOT" == "$REMOTE_RUN_BASE/"* ]] || \
    die "remote launcher reported an unexpected run root: $SERVER_RUN_ROOT"
if [[ "$SERVER_RESULT" == "reused" ]]; then
    if [[ "$SERVER_GPU" != "$GPU_INDEX" ]]; then
        echo "INFO: healthy learner already uses physical GPU $SERVER_GPU; requested GPU $GPU_INDEX applies only to a new start." >&2
    fi
    if [[ "$RUN_ID_EXPLICIT" == "1" && "${SERVER_RUN_ROOT##*/}" != "$RUN_ID" ]]; then
        echo "INFO: healthy learner lineage ${SERVER_RUN_ROOT##*/} was reused; requested run ID $RUN_ID applies only to a new start." >&2
    fi
fi

# Refuse if another local process claimed the actor endpoint while the remote
# learner was being prepared.
# We never inspect-and-kill an old tunnel: ownership cannot be inferred safely
# from a port number alone.
if ! local_port_is_free; then
    die "local port $LOCAL_PORT is occupied; no process was stopped and learner PID $SERVER_PID remains running"
fi

TUNNEL_PID=""
cleanup_tunnel() {
    if [[ -n "$TUNNEL_PID" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
        TUNNEL_PID=""
    fi
}
on_signal() {
    exit 130
}
trap cleanup_tunnel EXIT
trap on_signal INT TERM HUP

echo "Opening tunnel: 127.0.0.1:$LOCAL_PORT -> $SSH_HOST:127.0.0.1:$REMOTE_PORT"
ssh -N -T "${SSH_OPTIONS[@]}" \
    -o ExitOnForwardFailure=yes \
    -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" \
    "$SSH_HOST" &
TUNNEL_PID=$!

tunnel_ready=0
for _ in $(seq 1 40); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        wait "$TUNNEL_PID" || true
        die "SSH tunnel exited during startup; learner PID $SERVER_PID was not stopped"
    fi
    if /usr/bin/python3 - "$LOCAL_PORT" <<'PY'
import socket
import sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.2):
        pass
except OSError:
    raise SystemExit(1)
PY
    then
        tunnel_ready=1
        break
    fi
    sleep 0.25
done
[[ "$tunnel_ready" -eq 1 ]] || \
    die "tunnel did not accept TCP within 10s; learner PID $SERVER_PID was not stopped"

# Verify the actual gRPC contract through the exact endpoint run_hil_actor.sh
# uses.  This is GetServerInfo/GetBufferStatus only: it does not begin an
# episode, consume policy RNG, insert replay, or command the robot.
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO_ROOT/serl_ur_infra" \
timeout 15 "$ACTOR_PY" - "$LOCAL_PORT" <<'PY'
from dataclasses import asdict
import json
import sys

from ur_env.grpc_actor_transport import GrpcActorNetwork

network = GrpcActorNetwork(
    f"127.0.0.1:{sys.argv[1]}",
    actor_id="run-hil-server-local-readonly-probe",
    action_shape=(7,),
    timeout_s=2.0,
    max_response_age_s=3.0,
    expected_observation_schema_hash=(
        "3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903"
    ),
    expected_model_id="hil-serl-hybrid-sac-resnet10-trunk-cache-v1",
    expected_reward_authority="server_classifier",
    expected_reward_model_id="cube-in-cup-all3-ckpt150+sidecar-v1",
)
try:
    alive, ready, detail = network.health()
    if not alive or not ready:
        raise SystemExit(f"remote learner is not ready: {alive=} {ready=} {detail=}")
    info = network.get_server_info()
    status = network.get_buffer_status()
    print("Tunnel gRPC check:", json.dumps({
        "server": asdict(info),
        "buffer": asdict(status),
    }, sort_keys=True, separators=(",", ":")))
finally:
    network.close()
PY

echo "============================================================"
echo "HIL server ready"
echo "  learner : $SERVER_RESULT PID $SERVER_PID (server-owned)"
echo "  GPU     : physical $SERVER_GPU"
echo "  run root: $SERVER_RUN_ROOT"
echo "  actor   : grpc://127.0.0.1:$LOCAL_PORT"
echo "  tunnel  : PID $TUNNEL_PID (owned by this launcher)"
echo "============================================================"
echo "Keep this terminal open. Ctrl-C closes only the tunnel."
echo "The Kanu learner keeps running until you stop it explicitly on Kanu."

set +e
wait "$TUNNEL_PID"
TUNNEL_RC=$?
set -e
TUNNEL_PID=""
if [[ "$TUNNEL_RC" -ne 0 ]]; then
    echo "SSH tunnel exited with status $TUNNEL_RC; learner PID $SERVER_PID is still running." >&2
fi
exit "$TUNNEL_RC"
