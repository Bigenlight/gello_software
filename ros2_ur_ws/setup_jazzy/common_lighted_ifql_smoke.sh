#!/usr/bin/env bash
# Robot/camera-free recorded-frame checks for common Base Flow K1 and IFQL BoN32.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WS_DIR/.." && pwd)"
ROOT="${COMMON_IFQL_ROOT:-/home/junhyeong/ifql_common_lighted_20260919}"
RUN_DIR="${COMMON_IFQL_RUN_DIR:-$ROOT/carrot_seed0}"
EXPORT_MANIFEST="${COMMON_IFQL_EXPORT_MANIFEST:-$ROOT/carrot_data/ifql_export_manifest.json}"
PACKAGE="${COMMON_ENCODER_PACKAGE:-/home/junhyeong/common_encoder_lighted_20260919/carrot_seed0}"
SOURCE_DIR="${COMMON_IFQL_SOURCE_DIR:-$ROOT/source/vision_carrot}"
IFQL_SOURCE="${COMMON_IFQL_SOURCE:-$ROOT/source/qflow_svf/agents/ifql.py}"
BASE_RUNTIME="${COMMON_IFQL_BASE_RUNTIME:-/home/junhyeong/svf_real_carrot_commonenc_20260919/runtime/qflow_svf_merged}"
PYTHON="${COMMON_IFQL_PY:-/home/junhyeong/miniconda3/envs/il/bin/python}"
SERVER="$SCRIPT_DIR/common_lighted_ifql_server.py"
CLIENT="$SCRIPT_DIR/ifql_server_smoke_client.py"
FRAME_DIR="${COMMON_IFQL_SMOKE_FRAME_DIR:-}"
COUNT="${COMMON_IFQL_SMOKE_COUNT:-16}"
HOST=127.0.0.1
PORT=5695
LOG_ROOT="${COMMON_IFQL_SMOKE_LOG_ROOT:-$WS_DIR/log/common_lighted_ifql_smoke}"
SERVER_PID=""
STAGE_DIR=""

listening() { ss -ltn | awk -v endpoint="$HOST:$PORT" '$4 == endpoint { found=1 } END { exit !found }'; }
stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..30}; do kill -0 -- "-$SERVER_PID" 2>/dev/null || break; sleep 0.2; done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_server
    [[ -z "$STAGE_DIR" ]] || rm -rf -- "$STAGE_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: COMMON_IFQL_SMOKE_COUNT must be positive" >&2; exit 2; }
if listening; then echo "ERROR: port $PORT already listening" >&2; exit 2; fi
"$PYTHON" "$SCRIPT_DIR/common_lighted_ifql_preflight.py" \
    --run-dir "$RUN_DIR" --step 50000 --export-manifest "$EXPORT_MANIFEST" \
    --package "$PACKAGE" --ifql-source "$IFQL_SOURCE" --source-dir "$SOURCE_DIR" \
    --base-runtime "$BASE_RUNTIME"

if [[ -z "$FRAME_DIR" ]]; then
    STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/common-lighted-smoke.XXXXXX")"
    FRAME_DIR="$STAGE_DIR/recorded_frames"
    "$PYTHON" -B - "$ROOT/carrot_data/carrot_ifql_val.npz" "$EXPORT_MANIFEST" \
        "$PACKAGE/decoded_images" "$FRAME_DIR" "$COUNT" <<'PY'
import json, sys
from pathlib import Path
import cv2
import numpy as np

val_path, manifest_path, cache_path, out_path, count = sys.argv[1:]
count = int(count)
export = json.loads(Path(manifest_path).read_text())
norm = export["normalization"]
with np.load(val_path) as val:
    observations = np.asarray(val["observations"][:count], np.float32)
states = observations[:, -7:] * np.asarray(norm["proprio_std"], np.float32) + \
    np.asarray(norm["proprio_mean"], np.float32)
cache = Path(cache_path)
cam1 = np.load(cache / "ep_0056_cam1.npy", mmap_mode="r")
cam2 = np.load(cache / "ep_0056_cam2.npy", mmap_mode="r")
indices = np.arange(count, dtype=np.int64) * 3
if indices[-1] >= min(len(cam1), len(cam2)):
    raise SystemExit("not enough lighted validation frames for smoke")
out = Path(out_path)
(out / "frames").mkdir(parents=True)
np.save(out / "states.npy", states)
for row, index in enumerate(indices):
    for camera, source in ((1, cam1), (2, cam2)):
        ok, encoded = cv2.imencode(".jpg", np.ascontiguousarray(source[index][:, :, ::-1]),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise SystemExit(f"JPEG encode failed row={row} camera={camera}")
        (out / "frames" / f"{row:05d}_cam{camera}.jpg").write_bytes(encoded.tobytes())
print(f"[common-smoke] staged {count} lighted validation frame pairs from episode 56")
PY
fi

export COMMON_ENCODER_PACKAGE="$PACKAGE" COMMON_IFQL_SOURCE="$IFQL_SOURCE"
export COMMON_IFQL_BASE_RUNTIME="$BASE_RUNTIME" COMMON_IFQL_NATIVE_SERVER="$SOURCE_DIR/ifql_server.py"
for sampler in bc bon; do
    stamp="$(date -u +%Y%m%dT%H%M%S.%NZ)_${sampler}_pid$$"
    log_dir="$LOG_ROOT/$stamp"
    log="$log_dir/server_stdout.log"
    mkdir -p "$log_dir"
    samples=32; [[ "$sampler" == bc ]] && samples=1
    echo "[common-smoke] starting sampler=$sampler K=$samples"
    setsid env -u FMRL_CAM1_CROP -u FMRL_CAM1_MODE \
        QFLOW_DIR="$BASE_RUNTIME" HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}" \
        "$PYTHON" -B "$SERVER" --run-dir "$RUN_DIR" --step 50000 \
        --sampler "$sampler" --num-samples "$samples" --norm-stats "$EXPORT_MANIFEST" \
        --host "$HOST" --port "$PORT" --budget-s 0.6 --log-dir "$log_dir/episodes" \
        --device cuda >"$log" 2>&1 &
    SERVER_PID=$!
    for _ in {1..600}; do
        listening && break
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 60 "$log" >&2 || true; exit 1; }
        sleep 0.5
    done
    listening || { echo "ERROR: common server did not bind; see $log" >&2; exit 1; }
    grep -qF "agent ready: module=ifql px=False sampler=$sampler K=$samples D_a=2055 D_c=2055 action_dim=56" "$log" || {
        echo "ERROR: common server contract line missing" >&2; exit 1;
    }
    "$PYTHON" -B "$CLIENT" --repo-root "$REPO_ROOT" --frames "$FRAME_DIR" \
        --host "$HOST" --port "$PORT" --count "$COUNT" --timeout-s 2.0
    stop_server
    tail -n 14 "$log"
    echo "[common-smoke] sampler=$sampler OK log=$log"
done
