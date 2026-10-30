#!/usr/bin/env bash
# Offline replay only: verifies the completed staged SVF actor against recorded frames.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SVF_ROOT="${SVF_ROOT:-$HOME/carrot_svf/campaigns/svf_real_20260917}"
SVF_RUN_DIR="${SVF_RUN_DIR:-$SVF_ROOT/exp/qflow_rc/svf_real/svf_real_k06_c1_lead6_aug8_p1_s0_100k}"
SVF_NORM_STATS="${SVF_NORM_STATS:-$SVF_RUN_DIR/norm_stats_r18_ss_real_lead6.json}"
SVF_REAL_DATA="${SVF_REAL_DATA:-$SVF_ROOT/data/real_demo_r18_ss_lead6_aug8.npz}"
SVF_QFLOW_DIR="${SVF_QFLOW_DIR:-$SVF_ROOT/runtime/qflow_svf_merged}"
SVF_SERVER_PY="${SVF_SERVER_PY:-$SVF_ROOT/source/fmrl-vision/vision_carrot/ifql_server.py}"
SVF_PY="${SVF_PY:-/home/junhyeong/miniconda3/envs/il/bin/python}"

"$SVF_PY" "$SCRIPT_DIR/svf_real_preflight.py" --run-dir "$SVF_RUN_DIR" --step 100000 \
    --norm-stats "$SVF_NORM_STATS" --real-data "$SVF_REAL_DATA" \
    --qflow-dir "$SVF_QFLOW_DIR" --server-py "$SVF_SERVER_PY"

exec env IFQL_SERVER_DIR="$(dirname "$SVF_SERVER_PY")" IFQL_RUN_DIR="$SVF_RUN_DIR" \
    IFQL_NORM_STATS="$SVF_NORM_STATS" QFLOW_DIR="$SVF_QFLOW_DIR" \
    IFQL_SMOKE_COMPAT=0 IFQL_SMOKE_SAMPLER=actor IFQL_SMOKE_STEP=100000 IFQL_SMOKE_COUNT=253 \
    "$SCRIPT_DIR/ifql_server_smoke.sh"
