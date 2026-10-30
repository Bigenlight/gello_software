#!/usr/bin/env bash
# Pinned September 19 common-encoder retrained checkpoints, with ROS episode recording.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-}"
if [[ -z "$PROFILE" || "$PROFILE" == --help || "$PROFILE" == --list ]]; then
    echo 'Usage: bash ./run_real_common_policy.sh <bc|ifql|dsrl> [--dry-run] [ROS launch args...]'
    echo 'bc: frozen common Flow K1; ifql: new IFQL40k BoN32 mean; dsrl: new DSRL100k deterministic latent actor'
    echo 'Accepted START -> HOLD recording enabled; HOLD is not a failure label.'
    echo 'Viewer: SVF_EVAL_LOG_ROOT="$PWD/log/common_retrained" ./launch_cameras.sh'
    exit 0
fi
shift
case "$PROFILE" in bc|ifql|dsrl) ;; *) echo "Unknown common profile: $PROFILE" >&2; exit 2;; esac
DRY=0
EXTRA=()
for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then DRY=1; else EXTRA+=("$arg"); fi
done
export HEADLESS="${HEADLESS:-true}"
export IFQL_DRY_RUN="$DRY"
export COMMON_EVAL_PROFILE="retrained-$PROFILE"
export SVF_EPISODE_RECORDING=1
export REAL_EVAL_RECORDING=0
export SVF_RECORDER_DIR="$SCRIPT_DIR/setup_jazzy/common_eval_recording"
export IFQL_PARAMS_FILE="$SCRIPT_DIR/src/gello_policy/config/ifql_deploy.yaml"
export IFQL_LOG_ROOT="${IFQL_LOG_ROOT:-$SCRIPT_DIR/log/common_retrained}"
export COMMON_ENCODER_PACKAGE=/home/junhyeong/common_encoder_lighted_20260919/carrot_seed0
# Existing variable name is retained by the shared leader hook; metadata is generic.
if ! /usr/bin/grep -q COMMON_EVAL_PROFILE "$SVF_RECORDER_DIR/leader_recording.py"; then
    echo 'ERROR: generic IFQL/DSRL recording metadata integration is not installed.' >&2
    exit 2
fi
conda run -n il --no-capture-output /usr/bin/python3 "$SCRIPT_DIR/setup_jazzy/real_eval_recording_dev/preflight.py" --root "$IFQL_LOG_ROOT"
echo "### Common evaluation: $PROFILE; recording root=$IFQL_LOG_ROOT"
echo '### No automatic START; supervised one-episode safety gate remains required.'
if [[ "$PROFILE" == dsrl ]]; then
    exec bash "$SCRIPT_DIR/run_ur7e_common_retrained_dsrl_carrot.sh" "${EXTRA[@]}"
else
    export COMMON_IFQL_SAMPLER=bon
    [[ "$PROFILE" == bc ]] && export COMMON_IFQL_SAMPLER=bc
    exec bash "$SCRIPT_DIR/run_ur7e_common_retrained_ifql_carrot.sh" "${EXTRA[@]}"
fi
