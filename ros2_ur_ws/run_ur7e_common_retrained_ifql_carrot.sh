#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN=/home/junhyeong/real_common_baselines_20260919
SOURCE=/home/junhyeong/ifql_common_lighted_20260919/source/vision_carrot
EXPORT=/home/junhyeong/ifql_common_lighted_20260919/carrot_data/ifql_export_manifest.json
PACKAGE=/home/junhyeong/common_encoder_lighted_20260919/carrot_seed0
RUNTIME=/home/junhyeong/svf_real_carrot_commonenc_20260919/runtime/qflow_svf_merged
SAMPLER="${COMMON_IFQL_SAMPLER:-bon}"
case "$SAMPLER" in bc) K=1;; bon) K=32;; *) echo 'Expected sampler bc|bon' >&2; exit 2;; esac
conda run -n il --no-capture-output python "$SCRIPT_DIR/setup_jazzy/common_retrained_preflight.py" \
    --method ifql --run-dir "$CAMPAIGN/ifql_seed0" --step 40000 \
    --export-manifest "$EXPORT" --package "$PACKAGE" \
    --ifql-source "$CAMPAIGN/source/qflow_svf/agents/ifql.py" \
    --source-dir "$SOURCE" --base-runtime "$RUNTIME"
export REAL_EVAL_RECORDING=0 COMMON_ENCODER_PACKAGE="$PACKAGE"
export COMMON_IFQL_SOURCE="$CAMPAIGN/source/qflow_svf/agents/ifql.py"
export COMMON_IFQL_BASE_RUNTIME="$RUNTIME" COMMON_IFQL_NATIVE_SERVER="$SOURCE/ifql_server.py"
exec env IFQL_TASK=carrot IFQL_RUN_DIR="$CAMPAIGN/ifql_seed0" IFQL_STEP=40000 \
    IFQL_SAMPLER="$SAMPLER" IFQL_NUM_SAMPLES="$K" IFQL_Q_AGG="" \
    IFQL_NORM_STATS="$EXPORT" IFQL_ALLOW_FOREIGN_NORM_STATS=1 \
    IFQL_SERVER_PY="$SCRIPT_DIR/setup_jazzy/common_retrained_ifql_server.py" \
    IFQL_PY=/home/junhyeong/miniconda3/envs/il/bin/python IFQL_COMPAT=0 \
    IFQL_LOG_TAG=common_retrained_ifql_carrot QFLOW_DIR="$RUNTIME" REAL_EVAL_POLICY_TYPE=ifql \
    "$SCRIPT_DIR/run_ur7e_ifql_real.sh" "$@"
