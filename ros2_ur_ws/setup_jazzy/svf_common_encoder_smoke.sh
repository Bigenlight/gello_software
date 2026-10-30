#!/usr/bin/env bash
# Robot/camera-free replay of the staged common-encoder SVF on recorded JPEG frames.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WS_DIR/.." && pwd)"
BASE="${SVF_ROOT:-/home/junhyeong/svf_real_carrot_commonenc_20260919}"
RUN_DIR="${SVF_RUN_DIR:-$BASE/campaign/train/exp/qflow_rc/svf_sim/new_commonenc_c3_k06_g025_c144_clip2_p1_s0_100k}"
NORM_STATS="${SVF_NORM_STATS:-$RUN_DIR/norm_stats_r18_common_carrot_newcommonc3.json}"
QFLOW_DIR="${SVF_QFLOW_DIR:-$BASE/runtime/qflow_svf_merged}"
SERVER_PY="${SVF_SERVER_PY:-$BASE/source/vision_carrot/svf_jazzy_adapter.py}"
PYTHON="${SVF_PY:-/home/junhyeong/miniconda3/envs/il/bin/python}"
FRAME_DIR="${SVF_SMOKE_FRAME_DIR:-$WS_DIR/log/jazzy_port_20260916/claude_benchmark/episode0_extracted}"
CLIENT="$SCRIPT_DIR/ifql_server_smoke_client.py"
HOST=127.0.0.1
PORT=5695
COUNT="${SVF_SMOKE_COUNT:-48}"
DEVICE="${SVF_SMOKE_DEVICE:-cuda}"
LOG_ROOT="${SVF_SMOKE_LOG_ROOT:-$WS_DIR/log/svf_common_encoder_smoke}"
STAMP="$(date -u +%Y%m%dT%H%M%S.%NZ)_pid$$"
LOG_DIR="$LOG_ROOT/$STAMP"
SERVER_LOG="$LOG_DIR/server_stdout.log"
SERVER_PID=""

listening() {
    ss -ltn | awk -v endpoint="$HOST:$PORT" '$4 == endpoint { found=1 } END { exit !found }'
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 -- "-$SERVER_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "[svf-smoke] log preserved at $SERVER_LOG" >&2
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -x "$PYTHON" ]] || { echo "ERROR: SVF_PY is not executable: $PYTHON" >&2; exit 2; }
[[ -f "$SERVER_PY" && -f "$NORM_STATS" ]] || { echo "ERROR: staged SVF server/norm missing" >&2; exit 2; }
[[ -f "$CLIENT" && -f "$FRAME_DIR/states.npy" ]] || { echo "ERROR: smoke client/frames missing" >&2; exit 2; }
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SVF_SMOKE_COUNT must be positive" >&2; exit 2; }
command -v ss >/dev/null 2>&1 || { echo "ERROR: ss is required for the smoke port gate" >&2; exit 2; }
if listening; then
    echo "ERROR: port $PORT already listening; refusing to contact a stale policy server" >&2
    exit 2
fi

"$PYTHON" "$SCRIPT_DIR/svf_real_preflight_v2.py" \
    --run-dir "$RUN_DIR" --step 100000 --norm-stats "$NORM_STATS" \
    --qflow-dir "$QFLOW_DIR" --server-py "$SERVER_PY"

mkdir -p "$LOG_DIR"
echo "[svf-smoke] starting common-encoder SVF on $HOST:$PORT device=$DEVICE"
setsid env -u FMRL_CAM1_CROP -u FMRL_CAM1_MODE \
    PYTHONPATH="$(dirname "$SERVER_PY")" QFLOW_DIR="$QFLOW_DIR" \
    TORCH_HOME="${TORCH_HOME:-/home/junhyeong/carrot_ifql/.cache/torch}" \
    HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}" \
    "$PYTHON" -B "$SERVER_PY" \
    --run-dir "$RUN_DIR" --step 100000 --sampler actor --num-samples 0 \
    --host "$HOST" --port "$PORT" --budget-s 0.6 --norm-stats "$NORM_STATS" \
    --log-dir "$LOG_DIR/episodes" --device "$DEVICE" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in {1..600}; do
    listening && break
    kill -0 "$SERVER_PID" 2>/dev/null || {
        echo "ERROR: SVF server exited during startup; see $SERVER_LOG" >&2
        tail -n 40 "$SERVER_LOG" >&2 || true
        exit 1
    }
    sleep 0.5
done
listening || { echo "ERROR: SVF server did not bind within 300 s; see $SERVER_LOG" >&2; exit 1; }

grep -qF "agent ready: sampler=actor D_a=2055" "$SERVER_LOG" || {
    echo "ERROR: server bound without the expected actor/2055-D contract" >&2
    exit 1
}

echo "[svf-smoke] server ready; replaying $COUNT recorded frame pairs"
"$PYTHON" -B "$CLIENT" --repo-root "$REPO_ROOT" --frames "$FRAME_DIR" \
    --host "$HOST" --port "$PORT" --count "$COUNT" --timeout-s 2.0
tail -n 16 "$SERVER_LOG"
