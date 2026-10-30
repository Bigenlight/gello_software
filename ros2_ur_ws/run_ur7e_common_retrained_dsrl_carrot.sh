#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN=/home/junhyeong/real_common_baselines_20260919
EXPORT=/home/junhyeong/ifql_common_lighted_20260919/carrot_data/ifql_export_manifest.json
PACKAGE=/home/junhyeong/common_encoder_lighted_20260919/carrot_seed0
RUNTIME=/home/junhyeong/svf_real_carrot_commonenc_20260919/runtime/qflow_svf_merged
export COMMON_ENCODER_PACKAGE="$PACKAGE"
export REAL_EVAL_RECORDING=0 REAL_EVAL_POLICY_TYPE=dsrl REAL_EVAL_EXPECTED_SAMPLER=deterministic
conda run -n il --no-capture-output python "$SCRIPT_DIR/setup_jazzy/common_retrained_preflight.py" \
    --method dsrl --run-dir "$CAMPAIGN/dsrl_seed0" --step 100000 \
    --export-manifest "$EXPORT" --package "$PACKAGE" \
    --ifql-source "$CAMPAIGN/source/qflow_svf/agents/ifql.py" \
    --source-dir /home/junhyeong/ifql_common_lighted_20260919/source/vision_carrot --base-runtime "$RUNTIME"
exec env IFQL_TASK=carrot IFQL_RUN_DIR="$CAMPAIGN/dsrl_seed0" IFQL_STEP=100000 \
    IFQL_SAMPLER=actor IFQL_NUM_SAMPLES=1 IFQL_Q_AGG="" \
    IFQL_NORM_STATS="$EXPORT" IFQL_ALLOW_FOREIGN_NORM_STATS=1 \
    IFQL_SERVER_PY="$SCRIPT_DIR/setup_jazzy/common_retrained_dsrl_server.py" \
    IFQL_PY=/home/junhyeong/miniconda3/envs/il/bin/python IFQL_COMPAT=0 \
    IFQL_LOG_TAG=common_retrained_dsrl_carrot QFLOW_DIR="$RUNTIME" \
    "$SCRIPT_DIR/run_ur7e_ifql_real.sh" "$@"
