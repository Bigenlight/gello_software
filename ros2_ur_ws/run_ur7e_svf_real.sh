#!/usr/bin/env bash
# Thin SVF entrypoint.  It validates a staged real-domain SVF artifact, then delegates
# all real-robot safety, startup, server, and ROS behavior to the existing IFQL launcher.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SVF_ROOT="${SVF_ROOT:-$HOME/carrot_svf/campaigns/svf_real_20260917}"
# SVF_* is canonical.  IFQL_* aliases retain the established launcher-style invocation.
SVF_RUN_DIR="${SVF_RUN_DIR:-${IFQL_RUN_DIR:-$SVF_ROOT/exp/qflow_rc/svf_real/svf_real_k06_c1_lead6_aug8_p1_s0_100k}}"
SVF_STEP="${SVF_STEP:-${IFQL_STEP:-100000}}"
SVF_SAMPLER="${SVF_SAMPLER:-${IFQL_SAMPLER:-actor}}"
SVF_NORM_STATS="${SVF_NORM_STATS:-${IFQL_NORM_STATS:-}}"
SVF_REAL_DATA="${SVF_REAL_DATA:-}"
SVF_SERVER_PY="${SVF_SERVER_PY:-$SVF_ROOT/source/fmrl-vision/vision_carrot/ifql_server.py}"
SVF_QFLOW_DIR="${SVF_QFLOW_DIR:-$SVF_ROOT/runtime/qflow_svf_merged}"
SVF_PY="${SVF_PY:-${IFQL_PY:-/home/junhyeong/miniconda3/envs/il/bin/python}}"

usage() {
    cat >&2 <<'EOF'
Usage: SVF_RUN_DIR=/path/to/staged-real-svf HEADLESS=true ./run_ur7e_svf_real.sh

Required artifact contract: flags.json, params_<positive-step>.pkl, real NPZ, and
norm_stats inside SVF_RUN_DIR. The wrapper only accepts qflow_rc / use_critic_obs=false /
deploy_mode=actor and pinned real-data+normalization hashes. Set SVF_REAL_DATA explicitly
when flags.env_name is not a usable absolute path on this host.
For a non-actuating verification: IFQL_DRY_RUN=1 SVF_RUN_DIR=... ./run_ur7e_svf_real.sh
EOF
}

if [[ -z "$SVF_RUN_DIR" ]]; then
    usage
    exit 2
fi
if [[ "$SVF_SAMPLER" != "actor" ]]; then
    echo "ERROR: SVF_SAMPLER must be actor for the distilled SVF policy; got '$SVF_SAMPLER'." >&2
    exit 2
fi
[[ -f "$SVF_RUN_DIR/flags.json" ]] || { echo "ERROR: missing $SVF_RUN_DIR/flags.json" >&2; exit 2; }
if [[ -z "$SVF_NORM_STATS" ]]; then
    mapfile -t _norms < <(find "$SVF_RUN_DIR" -maxdepth 1 -type f -name 'norm_stats_*.json' | sort)
    [[ ${#_norms[@]} -eq 1 ]] || { echo "ERROR: set SVF_NORM_STATS; expected exactly one norm_stats_*.json in $SVF_RUN_DIR" >&2; exit 2; }
    SVF_NORM_STATS="${_norms[0]}"
fi
if [[ -z "$SVF_REAL_DATA" ]]; then
    SVF_REAL_DATA=$("$SVF_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("env_name", ""))' "$SVF_RUN_DIR/flags.json")
fi

"$SVF_PY" "$SCRIPT_DIR/setup_jazzy/svf_real_preflight.py" \
    --run-dir "$SVF_RUN_DIR" --step "$SVF_STEP" \
    --norm-stats "$SVF_NORM_STATS" --real-data "$SVF_REAL_DATA" \
    --qflow-dir "$SVF_QFLOW_DIR" --server-py "$SVF_SERVER_PY"

[[ -f "$SVF_SERVER_PY" ]] || { echo "ERROR: missing staged SVF server: $SVF_SERVER_PY" >&2; exit 2; }
[[ -d "$SVF_QFLOW_DIR/agents" ]] || { echo "ERROR: missing staged SVF runtime agents: $SVF_QFLOW_DIR" >&2; exit 2; }

exec env \
    IFQL_RUN_DIR="$SVF_RUN_DIR" IFQL_STEP="$SVF_STEP" IFQL_SAMPLER=actor \
    IFQL_NORM_STATS="$SVF_NORM_STATS" IFQL_SERVER_PY="$SVF_SERVER_PY" \
    IFQL_PY="$SVF_PY" IFQL_COMPAT=0 QFLOW_DIR="$SVF_QFLOW_DIR" \
    REAL_EVAL_POLICY_TYPE=svf REAL_EVAL_SVF_RUN_DIR="$SVF_RUN_DIR" \
    REAL_EVAL_SVF_QFLOW_DIR="$SVF_QFLOW_DIR" REAL_EVAL_SVF_REAL_DATA="$SVF_REAL_DATA" \
    "$SCRIPT_DIR/run_ur7e_ifql_real.sh" "$@"
