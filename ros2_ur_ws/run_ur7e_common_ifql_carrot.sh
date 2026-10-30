#!/usr/bin/env bash
# New lighted-carrot common encoder: frozen Base Flow (bc) or IFQL BoN (bon).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${COMMON_IFQL_ROOT:-/home/junhyeong/ifql_common_lighted_20260919}"
RUN_DIR="${COMMON_IFQL_RUN_DIR:-$ROOT/carrot_seed0}"
STEP="${COMMON_IFQL_STEP:-50000}"
SAMPLER="${COMMON_IFQL_SAMPLER:-bon}"
NUM_SAMPLES="${COMMON_IFQL_NUM_SAMPLES:-32}"
EXPORT_MANIFEST="${COMMON_IFQL_EXPORT_MANIFEST:-$ROOT/carrot_data/ifql_export_manifest.json}"
PACKAGE="${COMMON_ENCODER_PACKAGE:-/home/junhyeong/common_encoder_lighted_20260919/carrot_seed0}"
SOURCE_DIR="${COMMON_IFQL_SOURCE_DIR:-$ROOT/source/vision_carrot}"
IFQL_SOURCE="${COMMON_IFQL_SOURCE:-$ROOT/source/qflow_svf/agents/ifql.py}"
BASE_RUNTIME="${COMMON_IFQL_BASE_RUNTIME:-/home/junhyeong/svf_real_carrot_commonenc_20260919/runtime/qflow_svf_merged}"
PYTHON="${COMMON_IFQL_PY:-/home/junhyeong/miniconda3/envs/il/bin/python}"
SERVER="$SCRIPT_DIR/setup_jazzy/common_lighted_ifql_server.py"
PREFLIGHT="$SCRIPT_DIR/setup_jazzy/common_lighted_ifql_preflight.py"

case "$SAMPLER" in
    bc) NUM_SAMPLES=1 ;;
    bon) [[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: COMMON_IFQL_NUM_SAMPLES must be positive" >&2; exit 2; } ;;
    *) echo "ERROR: COMMON_IFQL_SAMPLER must be bc|bon, got '$SAMPLER'" >&2; exit 2 ;;
esac
[[ "$STEP" == 50000 ]] || { echo "ERROR: validated deployment candidate is step 50000, got $STEP" >&2; exit 2; }

"$PYTHON" "$PREFLIGHT" --run-dir "$RUN_DIR" --step "$STEP" \
    --export-manifest "$EXPORT_MANIFEST" --package "$PACKAGE" \
    --ifql-source "$IFQL_SOURCE" --source-dir "$SOURCE_DIR" --base-runtime "$BASE_RUNTIME"

# The common adapter currently implements the policy wire and episode NPZ logger,
# not the newer HDF5/MP4 recording-only arguments of run_ur7e_ifql_real.sh.
export REAL_EVAL_RECORDING=0
export COMMON_ENCODER_PACKAGE="$PACKAGE"
export COMMON_IFQL_SOURCE="$IFQL_SOURCE"
export COMMON_IFQL_BASE_RUNTIME="$BASE_RUNTIME"
export COMMON_IFQL_NATIVE_SERVER="$SOURCE_DIR/ifql_server.py"

exec env \
    IFQL_TASK=carrot IFQL_RUN_DIR="$RUN_DIR" IFQL_STEP="$STEP" \
    IFQL_SAMPLER="$SAMPLER" IFQL_NUM_SAMPLES="$NUM_SAMPLES" IFQL_Q_AGG="" \
    IFQL_NORM_STATS="$EXPORT_MANIFEST" IFQL_ALLOW_FOREIGN_NORM_STATS=1 \
    IFQL_SERVER_PY="$SERVER" IFQL_PY="$PYTHON" IFQL_COMPAT=0 \
    IFQL_LOG_TAG=common_lighted_carrot QFLOW_DIR="$BASE_RUNTIME" REAL_EVAL_POLICY_TYPE=ifql \
    "$SCRIPT_DIR/run_ur7e_ifql_real.sh" "$@"
