#!/usr/bin/env bash
# Pipeline-parity SVF: IFQL rows/contiguous H8/next_obs=i+8/macro reward,
# while keeping the SVF actor-only deployment algorithm.
set -Eeuo pipefail
BASE=/home/junhyeong/svf_real_carrot_commonenc_20260919
export SVF_ROOT="$BASE"
export SVF_RUN_DIR="$BASE/exp/qflow_rc/svf_sim/svf_ifqlparity_rows_minmax_c3_k06_g025_c144_clip2_p0_s0_100k"
export SVF_STEP="${SVF_STEP:-100000}"
export SVF_NORM_STATS="$SVF_RUN_DIR/norm_stats_r18_common_carrot_ifqlparity_rows.json"
export SVF_QFLOW_DIR="$BASE/runtime/qflow_svf_merged"
export SVF_SERVER_PY="$BASE/source/vision_carrot/svf_jazzy_adapter.py"
export SVF_PY=/home/junhyeong/miniconda3/envs/il/bin/python
export IFQL_PARAMS_FILE="$(cd "$(dirname "$0")" && pwd)/src/gello_policy/config/ifql_deploy.yaml"
# ROS-side recorder captures real commands as well as policy input/output.
# Legacy server-side recording flags are unsupported by the staged SVF adapter.
export REAL_EVAL_RECORDING=0
export SVF_EPISODE_RECORDING="${SVF_EPISODE_RECORDING:-1}"
export SVF_RECORDER_DIR="$(cd "$(dirname "$0")" && pwd)/setup_jazzy/real_eval_recording_dev"
export IFQL_LOG_ROOT="${IFQL_LOG_ROOT:-$(cd "$(dirname "$0")" && pwd)/log/svf_ifqlparity}"
if [[ "$SVF_EPISODE_RECORDING" == 1 ]]; then
    /usr/bin/python3 "$SVF_RECORDER_DIR/preflight.py" --root "$IFQL_LOG_ROOT"
    echo "### SVF episode recording ON: $IFQL_LOG_ROOT/run_*/episodes/ep_*"
fi
exec "$(cd "$(dirname "$0")" && pwd)/run_ur7e_svf_real.sh" "$@"
