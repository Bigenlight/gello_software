#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-}"
case "$PROFILE" in bc|ifql|dsrl) shift;; *) echo 'Usage: run_ur7e_common20_carrot.sh bc|ifql|dsrl [--dry-run]'; exit 2;; esac
CAMPAIGN=/home/junhyeong/real_common_baselines20_20260919
SOURCE=/home/junhyeong/ifql_common_lighted_20260919/source/vision_carrot
PACKAGE=/home/junhyeong/common_encoder_lighted_20260919/carrot_seed0
RUNTIME=/home/junhyeong/svf_real_carrot_commonenc_20260919/runtime/qflow_svf_merged
EXPORT="$CAMPAIGN/data/ifql_export_manifest.json"
METHOD="$PROFILE"; [[ "$PROFILE" == bc ]] && METHOD=ifql
STEP=$(conda run -n il python -c 'import json,sys; print(json.load(open(sys.argv[1]))["steps"][sys.argv[2]])' "$CAMPAIGN/deployment_manifest.json" "$METHOD")
SAMPLER=bon; K=32
[[ "$PROFILE" == bc ]] && SAMPLER=bc && K=1
[[ "$PROFILE" == dsrl ]] && SAMPLER=actor && K=1
export IFQL_DRY_RUN=0
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then export IFQL_DRY_RUN=1; else ARGS+=("$arg"); fi
done
export HEADLESS="${HEADLESS:-true}" REAL_EVAL_RECORDING=0 SVF_EPISODE_RECORDING=1
export COMMON_EVAL_PROFILE="retrained-$PROFILE"
export SVF_RECORDER_DIR="$HERE/setup_jazzy/common_eval_recording"
export IFQL_PARAMS_FILE="$HERE/src/gello_policy/config/ifql_deploy.yaml"
export IFQL_LOG_ROOT="${IFQL_LOG_ROOT:-$HERE/log/carrot20_h8}"
export COMMON_ENCODER_PACKAGE="$PACKAGE" COMMON_IFQL_SOURCE="$CAMPAIGN/source/qflow_svf/agents/ifql.py"
export COMMON_IFQL_BASE_RUNTIME="$RUNTIME" COMMON_IFQL_NATIVE_SERVER="$SOURCE/ifql_server.py"
/usr/bin/python3 "$HERE/setup_jazzy/real_eval_recording_dev/preflight.py" --root "$IFQL_LOG_ROOT"
conda run -n il --no-capture-output python "$HERE/setup_jazzy/common20_preflight.py" \
    --method "$METHOD" --run-dir "$CAMPAIGN/${METHOD}_seed0" --step "$STEP" \
    --export-manifest "$EXPORT" --package "$PACKAGE" --ifql-source "$COMMON_IFQL_SOURCE" \
    --source-dir "$SOURCE" --base-runtime "$RUNTIME"
echo "### $PROFILE: downstream train20; existing full-data encoder/base; contiguous8 queue."
[[ "$PROFILE" == bc ]] && echo '### BC weights unchanged: existing full-data pretrained actor; no 20-demo BC retrain.'
[[ "$PROFILE" == dsrl ]] && export REAL_EVAL_EXPECTED_SAMPLER=deterministic
exec env IFQL_TASK=carrot IFQL_RUN_DIR="$CAMPAIGN/${METHOD}_seed0" IFQL_STEP="$STEP" \
    IFQL_SAMPLER="$SAMPLER" IFQL_NUM_SAMPLES="$K" IFQL_Q_AGG="" \
    IFQL_NORM_STATS="$EXPORT" IFQL_ALLOW_FOREIGN_NORM_STATS=1 \
    IFQL_SERVER_PY="$HERE/setup_jazzy/common20_${METHOD}_server.py" \
    IFQL_PY=/home/junhyeong/miniconda3/envs/il/bin/python IFQL_COMPAT=0 \
    IFQL_LOG_TAG="common20_${PROFILE}_h8" QFLOW_DIR="$RUNTIME" REAL_EVAL_POLICY_TYPE="$METHOD" \
    bash "$HERE/run_ur7e_ifql_real.sh" "${ARGS[@]}"
