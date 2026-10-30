#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=/home/junhyeong/svf_real_carrot_commonenc_20260919
export SVF_ROOT=/home/junhyeong/svf_real_carrot20_20260919
export SVF_RUN_DIR="$SVF_ROOT/exp/fmrl-real/carrot20/svf_carrot20_norecover_reusedbc_s0_100k"
export SVF_STEP=100000
export SVF_NORM_STATS="$SVF_RUN_DIR/norm_stats_r18_common_carrot_ifqlparity_rows.json"
export SVF_QFLOW_DIR="$BASE/runtime/qflow_svf_merged"
export SVF_SERVER_PY="$BASE/source/vision_carrot/svf_jazzy_adapter.py"
export SVF_PY=/home/junhyeong/miniconda3/envs/il/bin/python
export IFQL_PARAMS_FILE="$HERE/src/gello_policy/config/ifql_deploy.yaml"
export REAL_EVAL_RECORDING=0
export SVF_EPISODE_RECORDING=1
export SVF_RECORDER_DIR="$HERE/setup_jazzy/real_eval_recording_dev"
export IFQL_LOG_ROOT="${IFQL_LOG_ROOT:-$HERE/log/carrot_compare}"
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then export IFQL_DRY_RUN=1; else ARGS+=("$arg"); fi
done
/usr/bin/python3 "$SVF_RECORDER_DIR/preflight.py" --root "$IFQL_LOG_ROOT"
echo 'SVF20: chronological train20 excluding recover/validation; reused encoder and SVF BC; final100k.'
echo "ROS START/HOLD recording enabled: $IFQL_LOG_ROOT"
exec bash "$HERE/run_ur7e_svf_real.sh" "${ARGS[@]}"
