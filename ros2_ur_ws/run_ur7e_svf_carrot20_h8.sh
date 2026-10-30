#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=/home/junhyeong/svf_real_carrot_commonenc_20260919
export SVF_ROOT=/home/junhyeong/svf_real_carrot20_20260919
export SVF_RUN_DIR="$SVF_ROOT/deploy_h8" SVF_STEP=100000
export SVF_NORM_STATS="$SVF_RUN_DIR/norm_stats_r18_common_carrot_ifqlparity_rows.json"
export SVF_QFLOW_DIR="$BASE/runtime/qflow_svf_merged"
export SVF_SERVER_PY="$BASE/source/vision_carrot/svf_jazzy_contiguous8_adapter.py"
export SVF_PY=/home/junhyeong/miniconda3/envs/il/bin/python
export IFQL_PARAMS_FILE="$HERE/src/gello_policy/config/ifql_deploy.yaml"
export REAL_EVAL_RECORDING=0 SVF_EPISODE_RECORDING=1
export SVF_RECORDER_DIR="$HERE/setup_jazzy/real_eval_recording_dev"
export IFQL_LOG_ROOT="${IFQL_LOG_ROOT:-$HERE/log/carrot20_h8}"
export IFQL_DRY_RUN=0
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then export IFQL_DRY_RUN=1; else ARGS+=("$arg"); fi
done
/usr/bin/python3 "$SVF_RECORDER_DIR/preflight.py" --root "$IFQL_LOG_ROOT"
conda run -n il python -c 'import hashlib,json,sys; from pathlib import Path; d=json.load(open(sys.argv[1])); assert d.get("extra_sha256"); assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in d["extra_sha256"].items()); print("H8 dependency hashes OK")' "$SVF_RUN_DIR/deployment_manifest.json"
echo '### SVF20 final100k: same checkpoint, corrected contiguous8 execution, no interpolation/hold.'
exec bash "$HERE/run_ur7e_svf_real.sh" "${ARGS[@]}"
