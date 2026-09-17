#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VC_DIR="/home/junhyeong/carrot_ifql/code/carrot_ifql_code_20260916/vision_carrot"
RUN_DIR="/home/junhyeong/carrot_ifql/hf/ifql_real_lead6_k0.9_s0"
QFLOW_DIR="/home/junhyeong/carrot_ifql/code/carrot_ifql_code_20260916/qflow_svf_merged"
NORM_STATS="$RUN_DIR/norm_stats_r18_ss_real_lead6.json"
FRAME_DIR="${IFQL_SMOKE_FRAME_DIR:-$ROOT_DIR/ros2_ur_ws/log/jazzy_port_20260916/claude_benchmark/episode0_extracted}"
CLIENT="$ROOT_DIR/ros2_ur_ws/setup_jazzy/ifql_server_smoke_client.py"
ADAPTER="$ROOT_DIR/ros2_ur_ws/setup_jazzy/ifql_server_compat.py"
LOG_DIR="$ROOT_DIR/ros2_ur_ws/log/ifql_smoke"
SERVER_LOG="$LOG_DIR/ifql_server_smoke_$(date +%Y%m%d_%H%M%S)_$$.log"
HOST=127.0.0.1
PORT=5695
SERVER_PID=""
STAGE_DIR=""

listening() { ss -ltn | awk -v e="$HOST:$PORT" '$4 == e { found=1 } END { exit !found }'; }

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 -- "-$SERVER_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

finish() {
    local status=$?
    trap - EXIT INT TERM
    stop_server
    if [[ -n "$STAGE_DIR" && "$STAGE_DIR" == "${TMPDIR:-/tmp}/ifql_server_smoke."* ]]; then
        rm -rf -- "$STAGE_DIR"
    fi
    echo "[smoke] log preserved at $SERVER_LOG" >&2
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$LOG_DIR"
[[ -x "$CLIENT" && -f "$ADAPTER" ]] || { echo "smoke helpers missing" >&2; exit 2; }
[[ -f "$VC_DIR/ifql_server.py" && -f "$RUN_DIR/flags.json" && -f "$NORM_STATS" ]] || { echo "IFQL inputs missing" >&2; exit 2; }
[[ -f "$FRAME_DIR/states.npy" && -f "$FRAME_DIR/frames/00000_cam1.jpg" && -f "$FRAME_DIR/frames/00000_cam2.jpg" ]] || {
    echo "frames missing: $FRAME_DIR (set IFQL_SMOKE_FRAME_DIR)" >&2
    exit 2
}
if listening; then echo "port $PORT already listening" >&2; exit 2; fi

CHECKPOINT="$RUN_DIR/params_100000.pkl"
ADAPTER_ARGS=()
if [[ ! -f "$CHECKPOINT" ]]; then
    CHECKPOINT="$RUN_DIR/params_100000.infer.pkl"
    [[ -f "$CHECKPOINT" ]] || { echo "no checkpoint found" >&2; exit 2; }
    ADAPTER_ARGS+=(--inference-checkpoint "$CHECKPOINT")
fi

STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ifql_server_smoke.XXXXXX")"
mkdir -p "$STAGE_DIR/torch/hub/checkpoints"
SOURCE_TORCH_HOME="/home/junhyeong/carrot_ifql/.cache/torch"
if [[ -d "$SOURCE_TORCH_HOME" ]]; then
    while IFS= read -r -d '' file; do
        relative="${file#"$SOURCE_TORCH_HOME/"}"
        mkdir -p "$STAGE_DIR/torch/$(dirname "$relative")"
        ln -s "$file" "$STAGE_DIR/torch/$relative"
    done < <(find "$SOURCE_TORCH_HOME" -type f -print0)
fi

echo "[smoke] starting adapter on $HOST:$PORT"
setsid conda run --no-capture-output -n il env -u FMRL_CAM1_CROP -u FMRL_CAM1_MODE \
    PYTHONPATH="$VC_DIR" QFLOW_DIR="$QFLOW_DIR" TORCH_HOME="$STAGE_DIR/torch" \
    HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    python -B "$ADAPTER" --server-script "$VC_DIR/ifql_server.py" \
    --run-dir "$RUN_DIR" --step 100000 --sampler bon --num-samples 32 \
    --norm-stats "$NORM_STATS" --host "$HOST" --port "$PORT" --budget-s 0.5 --warmup 3 \
    "${ADAPTER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in {1..600}; do
    listening && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited; see $SERVER_LOG" >&2; exit 1; }
    sleep 0.5
done
listening || { echo "server did not bind within 300 s; see $SERVER_LOG" >&2; exit 1; }

echo "[smoke] server ready; running 48 real-frame RESET + ACT requests"
conda run --no-capture-output -n il python -B "$CLIENT" --repo-root "$ROOT_DIR" \
    --frames "$FRAME_DIR" --host "$HOST" --port "$PORT" --count "${IFQL_SMOKE_COUNT:-48}" --timeout-s 2.0
tail -n 12 "$SERVER_LOG"
