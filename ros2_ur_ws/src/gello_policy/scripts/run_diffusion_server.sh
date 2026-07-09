#!/usr/bin/env bash
# Launch the py3.12 Diffusion inference server (torch + lerobot 0.6.1 + diffusers)
# standalone.
#
# This is the SERVER half of the Diffusion deploy: it loads the trained diffusion
# checkpoint (JOINT model), overrides the sampler to DDIM-N, and answers the rclpy
# policy_leader_node over localhost ZMQ (REQ/REP). It runs in the project's py3.12
# venv (lr_env) — a DIFFERENT interpreter/distro from the Humble ROS launch (py3.10).
# run_ur7e_diffusion_real.sh calls this automatically before ros2 launch; run it by
# hand for testing / bring-up.
#
# ############################################################################
# #  USAGE                                                                    #
# #                                                                          #
# #    # Minimal (checkpoint required, via env or --checkpoint):             #
# #    DIFFUSION_CHECKPOINT=/path/to/pretrained_model ./run_diffusion_server.sh
# #                                                                          #
# #    # Explicit flags (override the env defaults):                         #
# #    ./run_diffusion_server.sh --checkpoint /path/to/pretrained_model \    #
# #        --host 127.0.0.1 --port 5592 --device cuda \                      #
# #        --num-inference-steps 10 --scheduler DDIM --n-action-steps 32     #
# #                                                                          #
# #  ENV DEFAULTS (overridden by the matching flag if given):                #
# #    DIFFUSION_VENV      py3.12 venv root     (default: ros2_ur_ws/act_venv)#
# #    DIFFUSION_CHECKPOINT  checkpoint dir     (REQUIRED — error if unset)   #
# #    DIFFUSION_HOST      bind host            (default: 127.0.0.1)          #
# #    DIFFUSION_PORT      bind port            (default: 5592)               #
# #    DIFFUSION_DEVICE    torch device         (default: cuda)               #
# #    DIFFUSION_N_ACTION_STEPS  receding horizon k (default: 32)             #
# #    DIFFUSION_NUM_INFERENCE_STEPS  DDIM steps (default: 10)                #
# #    DIFFUSION_SCHEDULER  sampler override    (default: DDIM)               #
# ############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ -> gello_policy/ ; the py3.12 server lives in policy_server/diffusion_server.py
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_PY="$PKG_DIR/policy_server/diffusion_server.py"
# ros2_ur_ws is 2 levels up: src/gello_policy -> src -> ros2_ur_ws.
WS_DIR="$(cd "$PKG_DIR/../.." && pwd)"

# Default venv = ros2_ur_ws/act_venv (created per the deploy docs; shared with ACT —
# add diffusers==0.35.2 per requirements-diffusion.lock). Override with DIFFUSION_VENV=/path.
DIFFUSION_VENV="${DIFFUSION_VENV:-$WS_DIR/act_venv}"
DIFFUSION_HOST="${DIFFUSION_HOST:-127.0.0.1}"
DIFFUSION_PORT="${DIFFUSION_PORT:-5592}"
DIFFUSION_DEVICE="${DIFFUSION_DEVICE:-cuda}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-}"
DIFFUSION_N_ACTION_STEPS="${DIFFUSION_N_ACTION_STEPS:-32}"
DIFFUSION_NUM_INFERENCE_STEPS="${DIFFUSION_NUM_INFERENCE_STEPS:-10}"
DIFFUSION_SCHEDULER="${DIFFUSION_SCHEDULER:-DDIM}"

# --- Parse optional flags (override env) -------------------------------------
while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint) DIFFUSION_CHECKPOINT="$2"; shift 2 ;;
        --host)       DIFFUSION_HOST="$2"; shift 2 ;;
        --port)       DIFFUSION_PORT="$2"; shift 2 ;;
        --device)     DIFFUSION_DEVICE="$2"; shift 2 ;;
        --n-action-steps) DIFFUSION_N_ACTION_STEPS="$2"; shift 2 ;;
        --num-inference-steps) DIFFUSION_NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --scheduler)  DIFFUSION_SCHEDULER="$2"; shift 2 ;;
        --checkpoint=*) DIFFUSION_CHECKPOINT="${1#*=}"; shift ;;
        --host=*)     DIFFUSION_HOST="${1#*=}"; shift ;;
        --port=*)     DIFFUSION_PORT="${1#*=}"; shift ;;
        --device=*)   DIFFUSION_DEVICE="${1#*=}"; shift ;;
        --n-action-steps=*) DIFFUSION_N_ACTION_STEPS="${1#*=}"; shift ;;
        --num-inference-steps=*) DIFFUSION_NUM_INFERENCE_STEPS="${1#*=}"; shift ;;
        --scheduler=*) DIFFUSION_SCHEDULER="${1#*=}"; shift ;;
        *) echo "run_diffusion_server.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

VENV_PY="$DIFFUSION_VENV/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: py3.12 venv python not found at '$VENV_PY'." >&2
    echo "       Set DIFFUSION_VENV to the lerobot/torch venv root (see requirements-diffusion.lock)." >&2
    exit 1
fi
if [ ! -f "$SERVER_PY" ]; then
    echo "ERROR: Diffusion server not found at '$SERVER_PY' (IMPL-1 owns policy_server/diffusion_server.py)." >&2
    exit 1
fi
if [ -z "$DIFFUSION_CHECKPOINT" ]; then
    echo "ERROR: no checkpoint. Set DIFFUSION_CHECKPOINT=/path/to/pretrained_model or pass --checkpoint." >&2
    echo "       Download one with scripts/download_diffusion_checkpoint.sh (Bigenlight/diffusion_banana_in_pot_joint)." >&2
    exit 1
fi

echo "### Diffusion server | venv=$DIFFUSION_VENV | device=$DIFFUSION_DEVICE | bind=$DIFFUSION_HOST:$DIFFUSION_PORT"
echo "### checkpoint=$DIFFUSION_CHECKPOINT | n_action_steps=$DIFFUSION_N_ACTION_STEPS | num_inference_steps=$DIFFUSION_NUM_INFERENCE_STEPS | scheduler=$DIFFUSION_SCHEDULER"

exec "$VENV_PY" "$SERVER_PY" \
    --checkpoint "$DIFFUSION_CHECKPOINT" \
    --host "$DIFFUSION_HOST" \
    --port "$DIFFUSION_PORT" \
    --device "$DIFFUSION_DEVICE" \
    --n-action-steps "$DIFFUSION_N_ACTION_STEPS" \
    --num-inference-steps "$DIFFUSION_NUM_INFERENCE_STEPS" \
    --scheduler "$DIFFUSION_SCHEDULER"
