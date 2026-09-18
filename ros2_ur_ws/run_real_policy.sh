#!/usr/bin/env bash
# One-command profiles for the currently staged real-robot policies.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-}"
[[ -n "$PROFILE" ]] && shift || true

usage() {
    cat <<'EOF'
Usage:
  ./run_real_policy.sh --list
  ./run_real_policy.sh <profile> [--dry-run] [ros2 launch args...]

Profiles:
  ifql-bc-carrot       Flow BC, K=1
  ifql-mean-carrot     IFQL BoN32, twin-critic mean
  ifql-min-carrot      IFQL BoN32, twin-critic min
  ifql-bc-orange       Orange Flow BC, K=1
  ifql-mean-orange     Orange IFQL BoN32, mean
  ifql-min-orange      Orange IFQL BoN32, min
  svf-carrot           Distilled SVF actor
  dsrl-carrot          Real-demo DSRL deterministic latent actor
  dsrl-orange          Orange real-demo DSRL deterministic latent actor

Short aliases: ifql-bc, ifql-mean, ifql-min, svf, dsrl -> carrot profiles.
Every profile records HDF5 + diagnostic MP4 by default. Start cameras first:
  ./launch_cameras.sh
EOF
}

if [[ "$PROFILE" == "--list" || "$PROFILE" == "-h" || "$PROFILE" == "--help" || -z "$PROFILE" ]]; then
    usage
    [[ -n "$PROFILE" ]] && exit 0 || exit 2
fi

DRY=0
EXTRA=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then DRY=1; else EXTRA+=("$arg"); fi
done

case "$PROFILE" in
    ifql-bc) PROFILE=ifql-bc-carrot ;;
    ifql-mean) PROFILE=ifql-mean-carrot ;;
    ifql-min) PROFILE=ifql-min-carrot ;;
    svf) PROFILE=svf-carrot ;;
    dsrl) PROFILE=dsrl-carrot ;;
esac

export HEADLESS="${HEADLESS:-true}"
export REAL_EVAL_RECORDING="${REAL_EVAL_RECORDING:-1}"

run_ifql() {
    local task="$1" sampler="$2" qagg="$3"
    local run_dir
    if [[ "$task" == carrot ]]; then
        run_dir="${IFQL_ROOT:-$HOME/carrot_ifql}/hf/ifql_real_k0.9_lead6_aug8_p1.0_s0"
    else
        run_dir="${IFQL_ROOT:-$HOME/carrot_ifql}/hf_orange/ifql_orange_k0.9_lead2_aug8_p0.5_s0"
    fi
    [[ -d "$run_dir" ]] || { echo "ERROR: staged IFQL run missing: $run_dir" >&2; exit 2; }
    env IFQL_TASK="$task" IFQL_RUN_DIR="$run_dir" IFQL_STEP=100000 \
        IFQL_SAMPLER="$sampler" IFQL_NUM_SAMPLES=32 IFQL_Q_AGG="$qagg" \
        IFQL_DRY_RUN="$DRY" "$SCRIPT_DIR/run_ur7e_ifql_real.sh" "${EXTRA[@]}"
}

case "$PROFILE" in
    ifql-bc-carrot)   run_ifql carrot bc "" ;;
    ifql-mean-carrot) run_ifql carrot bon mean ;;
    ifql-min-carrot)  run_ifql carrot bon min ;;
    ifql-bc-orange)   run_ifql orange bc "" ;;
    ifql-mean-orange) run_ifql orange bon mean ;;
    ifql-min-orange)  run_ifql orange bon min ;;
    svf-carrot)
        env IFQL_DRY_RUN="$DRY" "$SCRIPT_DIR/run_ur7e_svf_real.sh" "${EXTRA[@]}"
        ;;
    dsrl-carrot|dsrl-orange)
        task="${PROFILE#dsrl-}"
        if [[ "$task" == carrot ]]; then
            staged="$SCRIPT_DIR/setup_jazzy/staged_models/dsrl_carrot_real_aug8_ns075_s0"
        else
            staged="$SCRIPT_DIR/setup_jazzy/staged_models/dsrl_orange_off_aug8base_ns075_s0"
        fi
        code="${IFQL_ROOT:-$HOME/carrot_ifql}/code/carrot_ifql_code_20260916/vision_carrot"
        env DSRL_DRY_RUN="$DRY" DSRL_RUN_DIR="$staged" \
            DSRL_SERVER_PY="$code/dsrl/dsrl_server.py" \
            DSRL_PY="${DSRL_PY:-/home/junhyeong/miniconda3/envs/il/bin/python}" \
            "$SCRIPT_DIR/run_ur7e_dsrl_real.sh" "${EXTRA[@]}"
        ;;
    *) echo "ERROR: unknown profile: $PROFILE" >&2; usage >&2; exit 2 ;;
esac
