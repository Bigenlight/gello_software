#!/usr/bin/env bash
# Launch the py3.12 ACT inference server (torch + lerobot 0.6.1) standalone.
#
# This is the SERVER half of the ACT deploy: it loads the trained ACT checkpoint
# and answers the rclpy policy_leader_node over localhost ZMQ (REQ/REP). It runs
# in the project's py3.12 venv (lr_env) — a DIFFERENT interpreter/distro from the
# Humble ROS launch (py3.10). run_ur7e_act_real.sh calls this automatically before
# ros2 launch; run it by hand for testing / bring-up.
#
# ############################################################################
# #  USAGE                                                                    #
# #                                                                          #
# #    # Minimal (checkpoint required, via env or --checkpoint):             #
# #    ACT_CHECKPOINT=/path/to/pretrained_model ./run_act_server.sh          #
# #                                                                          #
# #    # Explicit flags (override the env defaults):                         #
# #    ./run_act_server.sh --checkpoint /path/to/pretrained_model \          #
# #        --host 127.0.0.1 --port 5591 --device cuda                        #
# #                                                                          #
# #  ENV DEFAULTS (overridden by the matching flag if given):                #
# #    ACT_VENV      py3.12 venv root      (default: ros2_ur_ws/act_venv)      #
# #    ACT_CHECKPOINT  checkpoint dir      (REQUIRED — error if unset)        #
# #    ACT_HOST      bind host             (default: 127.0.0.1)               #
# #    ACT_PORT      bind port             (default: 5591)                    #
# #    ACT_DEVICE    torch device          (default: cuda)                    #
# #    ACT_N_ACTION_STEPS  receding horizon k (default: 30)                   #
# ############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ -> gello_policy/ ; the py3.12 server lives in policy_server/act_server.py
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_PY="$PKG_DIR/policy_server/act_server.py"
# ros2_ur_ws is 2 levels up: src/gello_policy -> src -> ros2_ur_ws.
WS_DIR="$(cd "$PKG_DIR/../.." && pwd)"

# Default venv = ros2_ur_ws/act_venv (created per the deploy docs). Override with
# ACT_VENV=/path if your py3.12 lerobot+torch venv lives elsewhere.
ACT_VENV="${ACT_VENV:-$WS_DIR/act_venv}"
ACT_HOST="${ACT_HOST:-127.0.0.1}"
ACT_PORT="${ACT_PORT:-5591}"
ACT_DEVICE="${ACT_DEVICE:-cuda}"
ACT_CHECKPOINT="${ACT_CHECKPOINT:-}"
ACT_N_ACTION_STEPS="${ACT_N_ACTION_STEPS:-30}"

# --- Parse optional flags (override env) -------------------------------------
while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint) ACT_CHECKPOINT="$2"; shift 2 ;;
        --host)       ACT_HOST="$2"; shift 2 ;;
        --port)       ACT_PORT="$2"; shift 2 ;;
        --device)     ACT_DEVICE="$2"; shift 2 ;;
        --n-action-steps) ACT_N_ACTION_STEPS="$2"; shift 2 ;;
        --checkpoint=*) ACT_CHECKPOINT="${1#*=}"; shift ;;
        --host=*)     ACT_HOST="${1#*=}"; shift ;;
        --port=*)     ACT_PORT="${1#*=}"; shift ;;
        --device=*)   ACT_DEVICE="${1#*=}"; shift ;;
        --n-action-steps=*) ACT_N_ACTION_STEPS="${1#*=}"; shift ;;
        *) echo "run_act_server.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

VENV_PY="$ACT_VENV/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: py3.12 venv python not found at '$VENV_PY'." >&2
    echo "       Set ACT_VENV to the lerobot/torch venv root (see requirements-act.lock)." >&2
    exit 1
fi
if [ ! -f "$SERVER_PY" ]; then
    echo "ERROR: ACT server not found at '$SERVER_PY' (BUILDER-1 owns policy_server/act_server.py)." >&2
    exit 1
fi
if [ -z "$ACT_CHECKPOINT" ]; then
    echo "ERROR: no checkpoint. Set ACT_CHECKPOINT=/path/to/pretrained_model or pass --checkpoint." >&2
    echo "       Download one with scripts/download_checkpoint.sh (Bigenlight/act_banana_in_pot)." >&2
    exit 1
fi

echo "### ACT server | venv=$ACT_VENV | device=$ACT_DEVICE | bind=$ACT_HOST:$ACT_PORT"
echo "### checkpoint=$ACT_CHECKPOINT | n_action_steps=$ACT_N_ACTION_STEPS"

exec "$VENV_PY" "$SERVER_PY" \
    --checkpoint "$ACT_CHECKPOINT" \
    --host "$ACT_HOST" \
    --port "$ACT_PORT" \
    --device "$ACT_DEVICE" \
    --n-action-steps "$ACT_N_ACTION_STEPS"
