#!/usr/bin/env bash
# Launch the py3.12 Flow-Matching inference server (torch + lerobot 0.6.1 +
# transformers/CLIP) standalone.
#
# This is the SERVER half of the FM deploy: it loads the trained flow-matching
# (multi_task_dit) JOINT checkpoint via lerobot's GENERIC policy factory, keeps the
# trained Euler-ODE integration (or a smaller --num-integration-steps for lower
# latency), and answers the rclpy policy_leader_node over localhost ZMQ (REQ/REP).
# It runs in the project's py3.12 venv (lr_env) — a DIFFERENT interpreter/distro
# from the Humble ROS launch (py3.10). run_ur7e_fm_real.sh calls this automatically
# before ros2 launch; run it by hand for testing / bring-up.
#
# ############################################################################
# #  USAGE                                                                    #
# #                                                                          #
# #    # Minimal (checkpoint required, via env or --checkpoint):             #
# #    FM_CHECKPOINT=/path/to/pretrained_model ./run_fm_server.sh            #
# #                                                                          #
# #    # Explicit flags (override the env defaults):                         #
# #    ./run_fm_server.sh --checkpoint /path/to/pretrained_model \           #
# #        --host 127.0.0.1 --port 5593 --device cuda \                      #
# #        --num-integration-steps 100 --n-action-steps 24 \                 #
# #        --task "put the right banana in the pot"                          #
# #                                                                          #
# #  ENV DEFAULTS (overridden by the matching flag if given):                #
# #    FM_VENV      py3.12 venv root       (default: ros2_ur_ws/act_venv)     #
# #    FM_CHECKPOINT  checkpoint dir       (REQUIRED — error if unset)        #
# #    FM_HOST      bind host              (default: 127.0.0.1)               #
# #    FM_PORT      bind port              (default: 5593)                    #
# #    FM_DEVICE    torch device           (default: cuda)                    #
# #    FM_N_ACTION_STEPS  receding horizon k (default: 24)                    #
# #    FM_NUM_INTEGRATION_STEPS  Euler steps (default: 10, see below)         #
# #    FM_TASK      CLIP task string       (default: per action space --       #
# #                 "put the right banana in the pot" for joint,              #
# #                 "Put carrot in pot" for eef_delta_v1)                     #
# #    FM_ACTION_SPACE  joint | eef_delta_v1 (default: joint)                  #
# ############################################################################
set -euo pipefail

# Deploy runs entirely from the checkpoint's baked weights (incl. the CLIP vision/text
# encoders). Force offline so multi_task_dit's CLIPVisionModel/CLIPTextModel.from_pretrained
# does NOT try to hit the HF hub at startup (which would crash on an offline robot PC and
# is a wasteful ~600MB re-download online — the weights are overwritten by the checkpoint).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ -> gello_policy/ ; the py3.12 server lives in policy_server/fm_server.py
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_PY="$PKG_DIR/policy_server/fm_server.py"
# ros2_ur_ws is 2 levels up: src/gello_policy -> src -> ros2_ur_ws.
WS_DIR="$(cd "$PKG_DIR/../.." && pwd)"

# Default venv = ros2_ur_ws/act_venv (created per the deploy docs; shared with ACT /
# diffusion — the FM server needs lerobot 0.6.1 + transformers/CLIP). Override with FM_VENV=/path.
FM_VENV="${FM_VENV:-$WS_DIR/act_venv}"
FM_HOST="${FM_HOST:-127.0.0.1}"
FM_PORT="${FM_PORT:-5593}"
FM_DEVICE="${FM_DEVICE:-cuda}"
FM_CHECKPOINT="${FM_CHECKPOINT:-}"
FM_N_ACTION_STEPS="${FM_N_ACTION_STEPS:-24}"
# Default 10 Euler steps: this is EXACTLY the sampler used for the offline eval that
# selected the 70k checkpoint (poseMAE 0.0735), and it keeps a refill well under the
# leader's act_timeout_s (0.6s). Raise toward the trained 100 only if you have latency
# headroom (benchmark first) — more steps ~= marginally smoother, much slower.
FM_NUM_INTEGRATION_STEPS="${FM_NUM_INTEGRATION_STEPS:-10}"
# Empty by default ON PURPOSE: the server picks the trained phrasing for the
# chosen action space (banana for joint, "Put carrot in pot" for eef_delta_v1).
# Hardcoding the banana string here would silently condition a carrot policy on
# the wrong CLIP sentence -- which nothing downstream can detect.
FM_TASK="${FM_TASK:-}"
# v2 wire protocol (CARROT_EEF_DESIGN §4). "joint" keeps the original contract
# byte-for-byte; "eef_delta_v1" serves a 16-D-state Cartesian-delta checkpoint.
# The server cross-checks this against the checkpoint and exits 4 on a mismatch.
FM_ACTION_SPACE="${FM_ACTION_SPACE:-joint}"

# --- Parse optional flags (override env) -------------------------------------
while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint) FM_CHECKPOINT="$2"; shift 2 ;;
        --host)       FM_HOST="$2"; shift 2 ;;
        --port)       FM_PORT="$2"; shift 2 ;;
        --device)     FM_DEVICE="$2"; shift 2 ;;
        --n-action-steps) FM_N_ACTION_STEPS="$2"; shift 2 ;;
        --num-integration-steps) FM_NUM_INTEGRATION_STEPS="$2"; shift 2 ;;
        --task)       FM_TASK="$2"; shift 2 ;;
        --action-space) FM_ACTION_SPACE="$2"; shift 2 ;;
        --checkpoint=*) FM_CHECKPOINT="${1#*=}"; shift ;;
        --host=*)     FM_HOST="${1#*=}"; shift ;;
        --port=*)     FM_PORT="${1#*=}"; shift ;;
        --device=*)   FM_DEVICE="${1#*=}"; shift ;;
        --n-action-steps=*) FM_N_ACTION_STEPS="${1#*=}"; shift ;;
        --num-integration-steps=*) FM_NUM_INTEGRATION_STEPS="${1#*=}"; shift ;;
        --task=*)     FM_TASK="${1#*=}"; shift ;;
        --action-space=*) FM_ACTION_SPACE="${1#*=}"; shift ;;
        *) echo "run_fm_server.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

VENV_PY="$FM_VENV/bin/python"

# zmq_protocol.py imports gello_policy.eef_space (which imports the ROS-free
# ur_gello_bringup.ur_kin) so that "eef_delta_v1"/16 are defined in exactly one
# place. The server runs as a BARE SCRIPT, so neither package is importable from
# sys.path[0]; put both source dirs on PYTHONPATH. (zmq_protocol has the same
# repair as a fallback -- this is the explicit, greppable half.)
SRC_DIR="$(cd "$PKG_DIR/.." && pwd)"
export PYTHONPATH="$SRC_DIR/gello_policy:$SRC_DIR/ur_gello_bringup${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: py3.12 venv python not found at '$VENV_PY'." >&2
    echo "       Set FM_VENV to the lerobot/torch venv root (lerobot 0.6.1 + transformers)." >&2
    exit 1
fi
if [ ! -f "$SERVER_PY" ]; then
    echo "ERROR: FM server not found at '$SERVER_PY'." >&2
    exit 1
fi
if [ -z "$FM_CHECKPOINT" ]; then
    echo "ERROR: no checkpoint. Set FM_CHECKPOINT=/path/to/pretrained_model or pass --checkpoint." >&2
    exit 1
fi

echo "### FM server | venv=$FM_VENV | device=$FM_DEVICE | bind=$FM_HOST:$FM_PORT"
echo "### checkpoint=$FM_CHECKPOINT | n_action_steps=$FM_N_ACTION_STEPS | num_integration_steps=${FM_NUM_INTEGRATION_STEPS:-<ckpt default>} | action_space=$FM_ACTION_SPACE | task='${FM_TASK:-<server default for $FM_ACTION_SPACE>}'"

# Assemble args. NOTE the guard below is vestigial for --num-integration-steps:
# FM_NUM_INTEGRATION_STEPS defaults to 10 above, so the flag is effectively ALWAYS
# passed. Set FM_NUM_INTEGRATION_STEPS= (empty) to fall back to the checkpoint's
# trained 100. --task is the one that is genuinely optional (empty -> the server's
# per-action-space default).
ARGS=(
    --checkpoint "$FM_CHECKPOINT"
    --host "$FM_HOST"
    --port "$FM_PORT"
    --device "$FM_DEVICE"
    --n-action-steps "$FM_N_ACTION_STEPS"
    --action-space "$FM_ACTION_SPACE"
)
# --task only when explicitly set: an empty --task would override the server's
# per-action-space default with the empty string (no CLIP conditioning at all).
[ -n "$FM_TASK" ] && ARGS+=(--task "$FM_TASK")
[ -n "$FM_NUM_INTEGRATION_STEPS" ] && ARGS+=(--num-integration-steps "$FM_NUM_INTEGRATION_STEPS")

exec "$VENV_PY" "$SERVER_PY" "${ARGS[@]}"
