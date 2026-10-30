#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/home/junhyeong/svf_real_carrot_commonenc_20260919
export SVF_ROOT="$BASE"
export SVF_RUN_DIR="$BASE/campaign/train/exp/qflow_rc/svf_sim/new_commonenc_c3_k06_g025_c144_clip2_p1_s0_100k"
export SVF_NORM_STATS="$SVF_RUN_DIR/norm_stats_r18_common_carrot_newcommonc3.json"
export SVF_QFLOW_DIR="$BASE/runtime/qflow_svf_merged"
export SVF_SERVER_PY="$BASE/source/vision_carrot/svf_jazzy_adapter.py"
export SVF_PY=/home/junhyeong/miniconda3/envs/il/bin/python
export IFQL_PARAMS_FILE="$(cd "$(dirname "$0")" && pwd)/src/gello_policy/config/ifql_deploy.yaml"
export REAL_EVAL_RECORDING=0
exec "$(cd "$(dirname "$0")" && pwd)/run_ur7e_svf_real.sh" "$@"
