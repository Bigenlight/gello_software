#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:---help}"
case "$PROFILE" in
    --help|--list)
        echo 'Usage: ./run_real_carrot_compare.sh <svf20|svf-parity|bc|ifql|dsrl> [--dry-run] [ROS args...]'
        echo 'Camera GUI: ./run_real_carrot_compare.sh --cameras'
        echo 'svf20/ifql/dsrl: downstream20, pretrained encoder/base reused, contiguous8 execution'
        echo 'bc: unchanged pretrained common actor from new IFQL (no BC retraining), contiguous8 execution'
        echo 'svf-parity: full56 IFQL-row parity SVF, original min-max actor, default100k'
        echo 'START -> HOLD recording is automatic; SVF adds post-HOLD critic video.'
        echo 'Logs: log/eval/<profile>/run_*/episodes. Camera automatically follows the launched policy.'
        exit 0;;
    --cameras|svf20|svf-parity|bc|ifql|dsrl) shift;;
    *) echo "Unknown profile: $PROFILE" >&2; exit 2;;
esac
export HEADLESS="${HEADLESS:-true}"
export REAL_EVAL_LOG_HUB="${REAL_EVAL_LOG_HUB:-$HERE/log}"
export REAL_EVAL_PROFILE="$PROFILE"
export IFQL_LOG_ROOT="${IFQL_LOG_ROOT:-$REAL_EVAL_LOG_HUB/eval/$PROFILE}"
if [[ "$PROFILE" == --cameras ]]; then
    export SVF_EVAL_LOG_ROOT="$REAL_EVAL_LOG_HUB"
    exec bash "$HERE/launch_cameras.sh" "$@"
elif [[ "$PROFILE" == svf20 ]]; then
    exec bash "$HERE/run_ur7e_svf_carrot20_h8.sh" "$@"
elif [[ "$PROFILE" == svf-parity ]]; then
    ARGS=()
    for arg in "$@"; do
        if [[ "$arg" == --dry-run ]]; then export IFQL_DRY_RUN=1; else ARGS+=("$arg"); fi
    done
    exec bash "$HERE/run_ur7e_svf_carrot_ifqlparity.sh" "${ARGS[@]}"
else
    if [[ ! -f /home/junhyeong/real_common_baselines20_20260919/deployment_manifest.json ]]; then
        echo '20-episode baseline training/staging not complete; no fallback to old56 models.' >&2
        exit 2
    fi
    exec bash "$HERE/run_ur7e_common20_carrot.sh" "$PROFILE" "$@"
fi
