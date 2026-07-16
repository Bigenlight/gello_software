#!/usr/bin/env bash
# Drive the REAL UR7e AUTONOMOUSLY with the trained FLOW-MATCHING policy over ROS2 Humble.
#
# ############################################################################
# #  ⚠️  REAL HARDWARE — THE UR7e WILL PHYSICALLY MOVE, DRIVEN BY A POLICY.   #
# #                                                                          #
# #  This starts TWO processes:                                             #
# #    (A) the py3.12 FM inference server (torch + lerobot 0.6.1 +           #
# #        transformers/CLIP, project lr_env), started in the BACKGROUND     #
# #        first (loads the multi_task_dit / flow_matching checkpoint via the #
# #        GENERIC lerobot factory, listens on localhost ZMQ :FM_PORT), and  #
# #    (B) the Humble (py3.10) ros2 launch (arm driver + synthetic policy     #
# #        leader + bridge handshake + Robotiq gripper).                     #
# #  They live on different interpreters/distros; localhost ZMQ joins them.  #
# #  Ctrl-C tears down BOTH (an EXIT trap kills the server).                 #
# #                                                                          #
# #  The FM model emits the IDENTICAL 7-D joint action contract as diffusion, #
# #  so this REUSES the policy-agnostic ur7e_diffusion_real.launch.py, only   #
# #  overriding params_file -> fm_deploy.yaml, act_port -> FM_PORT (5593),    #
# #  and checkpoint_path (logging).                                          #
# #                                                                          #
# #  Before running:                                                        #
# #   1) Workspace clear; keep the teach-pendant E-STOP within reach.        #
# #   2) START THE ROBOT program — pick ONE method (do NOT mix them):        #
# #      • Method A (default, HEADLESS unset/false): on the pendant, load +   #
# #        PLAY the External Control program.                                #
# #      • Method B (HEADLESS=true): the driver sends URScript directly (no   #
# #        pendant Play). REQUIRES the robot in REMOTE mode.                  #
# #   3) The arm first drives to the policy's HELD START POSE and PARKS       #
# #      there. It stays still until you explicitly run:                     #
# #        ros2 service call /policy_leader_node/start_execution \           #
# #            std_srvs/srv/Trigger                                          #
# #      Only THEN does the FM policy begin autonomous motion. On any server  #
# #      timeout the leader FAULTS and the arm halts (fail-silent).          #
# #                                                                          #
# #  GRIPPER INCLUDED: the 2F-85 is driven by the policy's gripper output    #
# #  (0=open..1=closed). ROBOT MUST BE POWERED ON. On connect the gripper    #
# #  auto-calibrates (open/close sweep) — KEEP FINGERS CLEAR.                #
# #                                                                          #
# #  Abort anytime: Ctrl-C here, and/or E-STOP on the pendant.               #
# ############################################################################
#
# ############################################################################
# #  USAGE                                                                    #
# #                                                                          #
# #    # Checkpoint is REQUIRED:                                             #
# #    FM_CHECKPOINT=/path/to/pretrained_model ./run_ur7e_fm_real.sh         #
# #                                                                          #
# #    # Method B (headless, no pendant Play — needs REMOTE mode):           #
# #    HEADLESS=true FM_CHECKPOINT=/path/... ./run_ur7e_fm_real.sh           #
# #                                                                          #
# #    # Override robot IP / port / device / Euler steps:                    #
# #    ROBOT_IP=192.168.1.50 FM_PORT=5603 FM_DEVICE=cuda \                   #
# #        FM_NUM_INTEGRATION_STEPS=10 \                                     #
# #        FM_CHECKPOINT=/path/... ./run_ur7e_fm_real.sh                     #
# #                                                                          #
# #    # Pass extra launch args after the script args:                       #
# #    FM_CHECKPOINT=/path/... ./run_ur7e_fm_real.sh launch_rviz:=false      #
# #                                                                          #
# #  ENV:                                                                    #
# #    ROBOT_IP  (default 192.168.10.11)   CALIB (optional kinematics YAML)   #
# #    HEADLESS  (true|1 -> Method B)      START_MODE (gello | init_align)    #
# #    FM_VENV  (default ros2_ur_ws/act_venv)  FM_CHECKPOINT (REQUIRED)       #
# #    FM_PORT  (default 5593)             FM_DEVICE (default cuda)           #
# #    FM_NUM_INTEGRATION_STEPS (default unset -> ckpt's 100)                 #
# #    FM_N_ACTION_STEPS (default 24)      FM_TASK (default banana-in-pot)    #
# #    FM_PARAMS_FILE (default installed config/fm_deploy.yaml)               #
# #    GELLO_REPO_ROOT (default <ros2_ur_ws>/..)                             #
# ############################################################################
set -e

ROBOT_IP="${ROBOT_IP:-192.168.10.11}"       # override: ROBOT_IP=x.x.x.x
CALIB="${CALIB:-}"                           # optional: CALIB=/path/ur7e_calibration.yaml
HEADLESS="${HEADLESS:-}"                      # HEADLESS=true|1 -> Method B (no pendant Play; needs REMOTE mode)
START_MODE="${START_MODE:-gello}"            # gello (default) | init_align

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # = ros2_ur_ws
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"    # gello_software

# --- FM (py3.12 server) settings --------------------------------------------
# Default venv = ros2_ur_ws/act_venv (created per the deploy docs; shared with ACT /
# diffusion — the FM server needs lerobot 0.6.1 + transformers/CLIP). Override with FM_VENV=/path.
FM_VENV="${FM_VENV:-$SCRIPT_DIR/act_venv}"
FM_PORT="${FM_PORT:-5593}"
FM_DEVICE="${FM_DEVICE:-cuda}"
FM_HOST="${FM_HOST:-127.0.0.1}"
FM_CHECKPOINT="${FM_CHECKPOINT:-}"
FM_N_ACTION_STEPS="${FM_N_ACTION_STEPS:-24}"
FM_NUM_INTEGRATION_STEPS="${FM_NUM_INTEGRATION_STEPS:-10}"
FM_TASK="${FM_TASK:-put the right banana in the pot}"
RUN_FM_SERVER="$SCRIPT_DIR/src/gello_policy/scripts/run_fm_server.sh"

if [ -z "${FM_CHECKPOINT}" ]; then
    echo "ERROR: FM_CHECKPOINT is required (path to the trained flow-matching pretrained_model dir)." >&2
    exit 1
fi
if [ ! -f "${RUN_FM_SERVER}" ]; then
    echo "ERROR: run_fm_server.sh not found at ${RUN_FM_SERVER}." >&2
    exit 1
fi

# Also honor headless_mode:=true passed as a launch arg so the banner can never
# disagree with the effective launch (ros2 launch is last-wins on dupes).
case " $* " in *" headless_mode:=true "*|*"headless_mode:=true"*) HEADLESS=true ;; esac

# --- ROS2 Humble environment -------------------------------------------------
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

# --- Resolve the FM params file (reuse the policy-agnostic diffusion launch, but
#     point it at fm_deploy.yaml). Prefer the installed share copy; fall back to src. ---
if [ -z "${FM_PARAMS_FILE:-}" ]; then
    PKG_PREFIX="$(ros2 pkg prefix gello_policy 2>/dev/null || true)"
    if [ -n "${PKG_PREFIX}" ] && [ -f "${PKG_PREFIX}/share/gello_policy/config/fm_deploy.yaml" ]; then
        FM_PARAMS_FILE="${PKG_PREFIX}/share/gello_policy/config/fm_deploy.yaml"
    else
        FM_PARAMS_FILE="${SCRIPT_DIR}/src/gello_policy/config/fm_deploy.yaml"
    fi
fi
if [ ! -f "${FM_PARAMS_FILE}" ]; then
    echo "ERROR: FM params file not found at ${FM_PARAMS_FILE} (did you colcon build gello_policy?)." >&2
    exit 1
fi

# --- Refuse to start if the ZMQ port is already occupied ---------------------
# A leftover server holding FM_PORT would make the NEW server die on bind, yet the
# health-check below could still see *something* listening and pass -> the leader would
# silently drive the arm from the STALE server. Fail loudly BEFORE spawning.
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":${FM_PORT}([^0-9]|$)"; then
    echo "ERROR: port ${FM_PORT} is already in use (a stale FM/diffusion/ACT server?)." >&2
    echo "       Starting now would leave the arm driven by that stale server. Kill it first:" >&2
    echo "         pkill -f fm_server.py ; ss -ltnp | grep :${FM_PORT}" >&2
    exit 1
fi

# --- Start the FM inference server in the BACKGROUND (before ros2 launch) ------
echo "### Starting FM server (py3.12): venv=${FM_VENV} device=${FM_DEVICE} bind=${FM_HOST}:${FM_PORT}"
echo "### checkpoint=${FM_CHECKPOINT} | integration_steps=${FM_NUM_INTEGRATION_STEPS:-<ckpt 100>} | n_action_steps=${FM_N_ACTION_STEPS} | task='${FM_TASK}'"
FM_VENV="${FM_VENV}" FM_HOST="${FM_HOST}" FM_PORT="${FM_PORT}" FM_DEVICE="${FM_DEVICE}" \
    FM_CHECKPOINT="${FM_CHECKPOINT}" \
    FM_N_ACTION_STEPS="${FM_N_ACTION_STEPS}" \
    FM_NUM_INTEGRATION_STEPS="${FM_NUM_INTEGRATION_STEPS}" \
    FM_TASK="${FM_TASK}" \
    bash "${RUN_FM_SERVER}" &
FM_SERVER_PID=$!
# Kill the server on ANY exit of this script (clean exit, error, or Ctrl-C).
trap 'kill "${FM_SERVER_PID}" 2>/dev/null || true' EXIT

# Health-check: catch an immediately-dying server (bad venv, missing checkpoint,
# torch/transformers import failure) NOW, instead of as a confusing ZMQ timeout.
echo "### Waiting for FM server to come up (pid ${FM_SERVER_PID})..."
for i in $(seq 1 30); do
    if ! kill -0 "${FM_SERVER_PID}" 2>/dev/null; then
        echo "ERROR: FM server (pid ${FM_SERVER_PID}) exited during startup — see its output above." >&2
        exit 1
    fi
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn 2>/dev/null | grep -qE ":${FM_PORT}([^0-9]|$)"; then
            echo "### FM server is listening on ${FM_HOST}:${FM_PORT}."
            break
        fi
    elif [ "$i" -ge 3 ]; then
        echo "### FM server process alive (port check unavailable — ss not found)."
        break
    fi
    sleep 1
done

# --- Build the ros2 launch args (REUSE the policy-agnostic diffusion launch) ---
ARGS=(
    robot_ip:="${ROBOT_IP}"
    start_mode:="${START_MODE}"
    params_file:="${FM_PARAMS_FILE}"
    act_host:="${FM_HOST}"
    act_port:="${FM_PORT}"
    checkpoint_path:="${FM_CHECKPOINT}"
)
[ -n "${CALIB}" ] && ARGS+=(kinematics_params_file:="${CALIB}")

if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    ARGS+=(headless_mode:=true)
    HEADLESS_STATE="true (Method B — REMOTE mode required, no pendant Play)"
else
    ARGS+=(headless_mode:=false)
    HEADLESS_STATE="false (Method A — PLAY External Control on the pendant)"
fi

echo "### REAL UR7e FM deploy | robot_ip=${ROBOT_IP} | calib=${CALIB:-<none>} | params=${FM_PARAMS_FILE}"
echo "### headless_mode=${HEADLESS_STATE}"
if [ "${START_MODE}" = "init_align" ]; then
    echo "### start_mode=init_align — arm parks at the init pose; then it chases the held start pose."
else
    echo "### start_mode=gello — arm drives straight to the policy's HELD start pose and parks."
fi
echo "### The FM policy is AUTONOMOUS but GATED: after the handshake the arm PARKS."
echo "### To begin motion:  ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger"
echo "### Robotiq 2F-85 gripper INCLUDED (Modbus over driver socat bridge /tmp/ttyUR)."
echo "### ROBOT MUST BE POWERED ON. Keep fingers clear (gripper auto-cal on connect)."
echo "### Ctrl-C tears down BOTH the FM server (pid ${FM_SERVER_PID}) and this launch."
if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    echo "### HEADLESS: ensure the pendant is in REMOTE mode + Real Robot (not Simulation)."
else
    echo "### Confirm External Control is PLAYING on the pendant before the handshake."
fi

# NOT exec'd on purpose: when the launch exits (including on Ctrl-C) control
# returns here and the EXIT trap kills the FM server so nothing is orphaned.
ros2 launch gello_policy ur7e_diffusion_real.launch.py "${ARGS[@]}" "$@"
