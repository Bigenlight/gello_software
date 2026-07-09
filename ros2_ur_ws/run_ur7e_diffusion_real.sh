#!/usr/bin/env bash
# Drive the REAL UR7e AUTONOMOUSLY with the trained Diffusion policy over ROS2 Humble.
#
# ############################################################################
# #  ⚠️  REAL HARDWARE — THE UR7e WILL PHYSICALLY MOVE, DRIVEN BY A POLICY.   #
# #                                                                          #
# #  This starts TWO processes:                                             #
# #    (A) the py3.12 Diffusion inference server (torch + lerobot + diffusers,#
# #        project lr_env), started in the BACKGROUND first (loads the        #
# #        checkpoint, overrides to DDIM-N, listens on localhost ZMQ          #
# #        :DIFFUSION_PORT), and                                             #
# #    (B) the Humble (py3.10) ros2 launch (arm driver + synthetic Diffusion  #
# #        leader + bridge handshake + Robotiq gripper).                     #
# #  They live on different interpreters/distros; localhost ZMQ joins them.  #
# #  Ctrl-C tears down BOTH (an EXIT trap kills the server).                 #
# #                                                                          #
# #  RUN THE LATENCY BENCHMARK FIRST (does NOT move the robot):              #
# #    src/gello_policy/scripts/benchmark_diffusion_latency.py               #
# #        --checkpoint "$DIFFUSION_CHECKPOINT"                              #
# #    Confirm p99 refill fits the bridge budget before the first arm run.   #
# #                                                                          #
# #  Before running:                                                        #
# #   1) Workspace clear; keep the teach-pendant E-STOP within reach.        #
# #   2) START THE ROBOT program — pick ONE method (do NOT mix them):        #
# #      • Method A (default, HEADLESS unset/false): on the pendant, load +   #
# #        PLAY the External Control program. The driver only moves the arm   #
# #        while External Control is running.                                #
# #      • Method B (HEADLESS=true): the driver sends URScript directly (no   #
# #        pendant Play). REQUIRES the robot in REMOTE mode.                  #
# #   3) The arm first drives to the policy's HELD START POSE (move-to-start  #
# #      handshake) and PARKS there. It stays still until you explicitly run: #
# #        ros2 service call /policy_leader_node/start_execution \           #
# #            std_srvs/srv/Trigger                                          #
# #      Only THEN does the Diffusion policy begin autonomous motion. On any  #
# #      server timeout the leader FAULTS and the arm halts (fail-silent).    #
# #                                                                          #
# #  GRIPPER INCLUDED: the 2F-85 is driven by the policy's gripper output    #
# #  (0=open..1=closed). ROBOT MUST BE POWERED ON (driver supplies 24V tool  #
# #  voltage + owns the /tmp/ttyUR socat bridge). On connect the gripper     #
# #  auto-calibrates (open/close sweep) — KEEP FINGERS CLEAR.                #
# #                                                                          #
# #  Abort anytime: Ctrl-C here, and/or E-STOP on the pendant.               #
# ############################################################################
#
# ############################################################################
# #  USAGE                                                                    #
# #                                                                          #
# #    # Checkpoint is REQUIRED (download via                                #
# #    #   src/gello_policy/scripts/download_diffusion_checkpoint.sh):        #
# #    DIFFUSION_CHECKPOINT=/path/to/pretrained_model ./run_ur7e_diffusion_real.sh
# #                                                                          #
# #    # Method B (headless, no pendant Play — needs REMOTE mode):           #
# #    HEADLESS=true DIFFUSION_CHECKPOINT=/path/... ./run_ur7e_diffusion_real.sh
# #                                                                          #
# #    # Override robot IP / calib / port / device / DDIM steps:             #
# #    ROBOT_IP=192.168.1.50 DIFFUSION_PORT=5602 DIFFUSION_DEVICE=cuda \     #
# #        DIFFUSION_NUM_INFERENCE_STEPS=5 \                                 #
# #        DIFFUSION_CHECKPOINT=/path/... ./run_ur7e_diffusion_real.sh       #
# #                                                                          #
# #    # Pass extra launch args after the script args:                       #
# #    DIFFUSION_CHECKPOINT=/path/... ./run_ur7e_diffusion_real.sh launch_rviz:=false
# #                                                                          #
# #  ENV:                                                                    #
# #    ROBOT_IP  (default 192.168.10.11)   CALIB (optional kinematics YAML)   #
# #    HEADLESS  (true|1 -> Method B)      START_MODE (gello | init_align)    #
# #    DIFFUSION_VENV  (default ros2_ur_ws/act_venv)  DIFFUSION_CHECKPOINT (REQUIRED)#
# #    DIFFUSION_PORT  (default 5592)      DIFFUSION_DEVICE (default cuda)    #
# #    DIFFUSION_NUM_INFERENCE_STEPS (default 10)  DIFFUSION_SCHEDULER (DDIM) #
# #    DIFFUSION_N_ACTION_STEPS (default 32)                                 #
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

# --- Diffusion (py3.12 server) settings -------------------------------------
# Default venv = ros2_ur_ws/act_venv (created per the deploy docs; shared with ACT —
# add diffusers==0.35.2 per requirements-diffusion.lock). Override with DIFFUSION_VENV=/path.
DIFFUSION_VENV="${DIFFUSION_VENV:-$SCRIPT_DIR/act_venv}"
DIFFUSION_PORT="${DIFFUSION_PORT:-5592}"
DIFFUSION_DEVICE="${DIFFUSION_DEVICE:-cuda}"
DIFFUSION_HOST="${DIFFUSION_HOST:-127.0.0.1}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-}"
DIFFUSION_N_ACTION_STEPS="${DIFFUSION_N_ACTION_STEPS:-32}"
DIFFUSION_NUM_INFERENCE_STEPS="${DIFFUSION_NUM_INFERENCE_STEPS:-10}"
DIFFUSION_SCHEDULER="${DIFFUSION_SCHEDULER:-DDIM}"
RUN_DIFFUSION_SERVER="$SCRIPT_DIR/src/gello_policy/scripts/run_diffusion_server.sh"

if [ -z "${DIFFUSION_CHECKPOINT}" ]; then
    echo "ERROR: DIFFUSION_CHECKPOINT is required (path to the trained diffusion pretrained_model dir)." >&2
    echo "       Download it first:" >&2
    echo "         CKPT=\$(src/gello_policy/scripts/download_diffusion_checkpoint.sh | tail -1)" >&2
    echo "         DIFFUSION_CHECKPOINT=\"\$CKPT\" ./run_ur7e_diffusion_real.sh" >&2
    exit 1
fi
if [ ! -f "${RUN_DIFFUSION_SERVER}" ]; then
    echo "ERROR: run_diffusion_server.sh not found at ${RUN_DIFFUSION_SERVER}." >&2
    exit 1
fi

# Also honor headless_mode:=true passed as a launch arg so the banner can never
# disagree with the effective launch (ros2 launch is last-wins on dupes).
case " $* " in *" headless_mode:=true "*|*"headless_mode:=true"*) HEADLESS=true ;; esac

# --- ROS2 Humble environment -------------------------------------------------
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

# --- Refuse to start if the ZMQ port is already occupied ---------------------
# A leftover server from a previous session holding DIFFUSION_PORT would make the NEW
# server die on bind ("Address already in use"), yet the health-check below would still
# see *something* listening and pass -> the leader would silently drive the arm from the
# STALE server (old checkpoint / old DDIM settings) while the banner claims the new one.
# Fail loudly BEFORE spawning: if the port is free now, only our server can occupy it after.
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":${DIFFUSION_PORT}([^0-9]|$)"; then
    echo "ERROR: port ${DIFFUSION_PORT} is already in use (a stale diffusion/ACT server?)." >&2
    echo "       Starting now would leave the arm driven by that stale server. Kill it first:" >&2
    echo "         pkill -f diffusion_server.py ; ss -ltnp | grep :${DIFFUSION_PORT}" >&2
    exit 1
fi

# --- Start the Diffusion inference server in the BACKGROUND (before ros2 launch) ---
# It loads the checkpoint and listens on DIFFUSION_HOST:DIFFUSION_PORT. The synthetic
# leader only queries it AFTER ~/start_execution, so it has ample time to warm up. We
# do NOT exec the launch below, so this script's EXIT trap fires on Ctrl-C and tears
# the server down together with the launch.
echo "### Starting Diffusion server (py3.12): venv=${DIFFUSION_VENV} device=${DIFFUSION_DEVICE} bind=${DIFFUSION_HOST}:${DIFFUSION_PORT}"
echo "### checkpoint=${DIFFUSION_CHECKPOINT} | DDIM steps=${DIFFUSION_NUM_INFERENCE_STEPS} | scheduler=${DIFFUSION_SCHEDULER} | n_action_steps=${DIFFUSION_N_ACTION_STEPS}"
DIFFUSION_VENV="${DIFFUSION_VENV}" DIFFUSION_HOST="${DIFFUSION_HOST}" DIFFUSION_PORT="${DIFFUSION_PORT}" DIFFUSION_DEVICE="${DIFFUSION_DEVICE}" \
    DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT}" \
    DIFFUSION_N_ACTION_STEPS="${DIFFUSION_N_ACTION_STEPS}" \
    DIFFUSION_NUM_INFERENCE_STEPS="${DIFFUSION_NUM_INFERENCE_STEPS}" \
    DIFFUSION_SCHEDULER="${DIFFUSION_SCHEDULER}" \
    bash "${RUN_DIFFUSION_SERVER}" &
DIFFUSION_SERVER_PID=$!
# Kill the server on ANY exit of this script (clean exit, error, or Ctrl-C).
trap 'kill "${DIFFUSION_SERVER_PID}" 2>/dev/null || true' EXIT

# Health-check: catch an immediately-dying server (bad venv, missing checkpoint,
# torch/diffusers import failure) NOW, instead of as a confusing ZMQ timeout at
# ~/start_execution. The leader is operator-gated, so we don't need full readiness —
# just confirm the process didn't crash on startup (and, if `ss` is available, that
# the port is actually listening).
echo "### Waiting for Diffusion server to come up (pid ${DIFFUSION_SERVER_PID})..."
for i in $(seq 1 30); do
    if ! kill -0 "${DIFFUSION_SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Diffusion server (pid ${DIFFUSION_SERVER_PID}) exited during startup — see its output above." >&2
        exit 1
    fi
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn 2>/dev/null | grep -qE ":${DIFFUSION_PORT}([^0-9]|$)"; then
            echo "### Diffusion server is listening on ${DIFFUSION_HOST}:${DIFFUSION_PORT}."
            break
        fi
    elif [ "$i" -ge 3 ]; then
        echo "### Diffusion server process alive (port check unavailable — ss not found)."
        break
    fi
    sleep 1
done

# --- Build the ros2 launch args ---------------------------------------------
ARGS=(
    robot_ip:="${ROBOT_IP}"
    start_mode:="${START_MODE}"
    act_host:="${DIFFUSION_HOST}"
    act_port:="${DIFFUSION_PORT}"
    checkpoint_path:="${DIFFUSION_CHECKPOINT}"
)
[ -n "${CALIB}" ] && ARGS+=(kinematics_params_file:="${CALIB}")

if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    ARGS+=(headless_mode:=true)
    HEADLESS_STATE="true (Method B — REMOTE mode required, no pendant Play)"
else
    ARGS+=(headless_mode:=false)
    HEADLESS_STATE="false (Method A — PLAY External Control on the pendant)"
fi

echo "### REAL UR7e Diffusion deploy | robot_ip=${ROBOT_IP} | calib=${CALIB:-<none>}"
echo "### headless_mode=${HEADLESS_STATE}"
if [ "${START_MODE}" = "init_align" ]; then
    echo "### start_mode=init_align — arm parks at the init pose; then it chases the held start pose."
else
    echo "### start_mode=gello — arm drives straight to the policy's HELD start pose and parks."
fi
echo "### The Diffusion policy is AUTONOMOUS but GATED: after the handshake the arm PARKS."
echo "### To begin motion:  ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger"
echo "### Robotiq 2F-85 gripper INCLUDED (Modbus over driver socat bridge /tmp/ttyUR)."
echo "### ROBOT MUST BE POWERED ON. Keep fingers clear (gripper auto-cal on connect)."
echo "### Ctrl-C tears down BOTH the Diffusion server (pid ${DIFFUSION_SERVER_PID}) and this launch."
if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    echo "### HEADLESS: ensure the pendant is in REMOTE mode + Real Robot (not Simulation)."
else
    echo "### Confirm External Control is PLAYING on the pendant before the handshake."
fi

# NOT exec'd on purpose: when the launch exits (including on Ctrl-C) control
# returns here and the EXIT trap kills the Diffusion server so nothing is orphaned.
ros2 launch gello_policy ur7e_diffusion_real.launch.py "${ARGS[@]}" "$@"
