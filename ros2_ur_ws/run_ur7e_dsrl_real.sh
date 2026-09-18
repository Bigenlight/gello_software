#!/usr/bin/env bash
# Fail-closed DSRL entrypoint. A staged artifact must bring its own
# real_serve_meta.json, which is validated before this script can reach ROS.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT="$SCRIPT_DIR/setup_jazzy/dsrl_real_preflight.py"
DSRL_RUN_DIR="${DSRL_RUN_DIR:-}"
DSRL_REAL_SERVE_META="${DSRL_REAL_SERVE_META:-}"
DSRL_SERVER_PY="${DSRL_SERVER_PY:-}"
DSRL_PY="${DSRL_PY:-python3}"
DSRL_DRY_RUN="${DSRL_DRY_RUN:-}"

usage() {
    cat >&2 <<'EOF'
Usage (only after staging an attested real DSRL artifact):
  DSRL_RUN_DIR=/path/to/run DSRL_SERVER_PY=/path/to/dsrl_server.py \
    DSRL_PY=/path/to/python ./run_ur7e_dsrl_real.sh

This launcher refuses every run without <DSRL_RUN_DIR>/real_serve_meta.json
containing domain=real, an explicit
nonprivileged actor, use_critic_obs=false, and matching checkpoint/base/norm hashes.
Set DSRL_DRY_RUN=1 to validate and print the delegated command without ROS, a
policy server bind, or any robot process.
EOF
}

if [[ -z "$DSRL_RUN_DIR" || -z "$DSRL_SERVER_PY" ]]; then
    usage
    exit 2
fi
if [[ ! -x "$DSRL_PY" || ! -f "$PREFLIGHT" || ! -f "$DSRL_SERVER_PY" ]]; then
    echo "ERROR: need an executable DSRL_PY plus existing preflight/server files." >&2
    exit 2
fi
if [[ -z "$DSRL_REAL_SERVE_META" ]]; then
    DSRL_REAL_SERVE_META="$DSRL_RUN_DIR/real_serve_meta.json"
fi

# The preflight emits only shell-quoted assignments after validating every value.
# This keeps the selected checkpoint, base checkpoint, norm stats, and YAML tied to
# the same signed-by-content manifest before a child can bind or source ROS.
if ! _preflight_env="$($DSRL_PY "$PREFLIGHT" --run-dir "$DSRL_RUN_DIR" \
    --manifest "$DSRL_REAL_SERVE_META" --emit-shell)"; then
    echo "ERROR: DSRL real launch refused before socket bind or ROS startup." >&2
    exit 2
fi
eval "$_preflight_env"
export DSRL_REAL_SERVE_META

case "$DSRL_DRY_RUN" in
    ""|0|1) ;;
    *) echo "ERROR: DSRL_DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac

# The policy-agnostic UR launcher calls its server mode "actor".  dsrl_server
# resolves that compatibility spelling to the manifest-pinned base/dsrl_det/
# dsrl_sample mode after independently validating DSRL_REAL_SERVE_META.
IFQL_SAMPLER=actor
IFQL_PORT=5596
IFQL_NUM_SAMPLES=1
case "$DSRL_TASK" in
    carrot|carrot_in_pot) IFQL_TASK=carrot ;;
    orange|orange_bowl_in_purple_bowl) IFQL_TASK=orange ;;
    *) echo "ERROR: unsupported DSRL_TASK from manifest: $DSRL_TASK" >&2; exit 2 ;;
esac

if [[ "$DSRL_DRY_RUN" == 1 ]]; then
    echo "### DSRL_DRY_RUN=1: real_serve_meta passed; no ROS, server bind, or robot process will start."
    echo "### staged only (not a released real DSRL): task=$DSRL_TASK sampler=$DSRL_SAMPLER step=$DSRL_STEP"
    echo "### checkpoint=$DSRL_CHECKPOINT sha256=$DSRL_CHECKPOINT_SHA256"
    echo "### base_checkpoint=$DSRL_BASE_CHECKPOINT sha256=$DSRL_BASE_CHECKPOINT_SHA256"
    echo "### norm_stats=$DSRL_NORM_STATS sha256=$DSRL_NORM_STATS_SHA256"
    echo "### delegated launcher: $SCRIPT_DIR/run_ur7e_ifql_real.sh (server=$DSRL_SERVER_PY, port=$IFQL_PORT)"
    exit 0
fi

# The existing launcher owns the ROS safety stack, hold/start gate, stale-port
# refusal, process teardown, and the common real_eval/v1 HDF5 + MP4 recorder.
# dsrl_server.py exposes the same recording flags as ifql_server.py.
exec env \
    IFQL_RUN_DIR="$DSRL_RUN_DIR" IFQL_STEP="$DSRL_STEP" \
    IFQL_TASK="$IFQL_TASK" IFQL_LOG_TAG="dsrl_${IFQL_TASK}_${DSRL_SAMPLER}" \
    IFQL_SAMPLER="$IFQL_SAMPLER" IFQL_NUM_SAMPLES="$IFQL_NUM_SAMPLES" \
    IFQL_NORM_STATS="$DSRL_NORM_STATS" IFQL_PARAMS_FILE="$DSRL_PARAMS_FILE" \
    IFQL_SERVER_PY="$DSRL_SERVER_PY" IFQL_PY="$DSRL_PY" IFQL_COMPAT=0 \
    IFQL_PORT="$IFQL_PORT" DSRL_REAL_SERVE_META="$DSRL_REAL_SERVE_META" \
    REAL_EVAL_POLICY_TYPE=dsrl REAL_EVAL_RECORDING="${REAL_EVAL_RECORDING:-1}" \
    "$SCRIPT_DIR/run_ur7e_ifql_real.sh" "$@"
