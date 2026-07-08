#!/usr/bin/env bash
# Drive the REAL UR7e AUTONOMOUSLY with the trained ACT policy over ROS2 Humble.
#
# ############################################################################
# #  ⚠️  REAL HARDWARE — THE UR7e WILL PHYSICALLY MOVE, DRIVEN BY A POLICY.   #
# #                                                                          #
# #  This starts TWO processes:                                             #
# #    (A) the py3.12 ACT inference server (torch + lerobot, project lr_env),#
# #        started in the BACKGROUND first (loads the checkpoint, listens on #
# #        localhost ZMQ :ACT_PORT), and                                     #
# #    (B) the Humble (py3.10) ros2 launch (arm driver + synthetic ACT       #
# #        leader + bridge handshake + Robotiq gripper).                     #
# #  They live on different interpreters/distros; localhost ZMQ joins them.  #
# #  Ctrl-C tears down BOTH (an EXIT trap kills the server).                 #
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
# #      Only THEN does the ACT policy begin autonomous motion. On any server #
# #      timeout the leader FAULTS and the arm halts (fail-silent).          #
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
# #    #   src/gello_policy/scripts/download_checkpoint.sh):                  #
# #    ACT_CHECKPOINT=/path/to/pretrained_model ./run_ur7e_act_real.sh       #
# #                                                                          #
# #    # Method B (headless, no pendant Play — needs REMOTE mode):           #
# #    HEADLESS=true ACT_CHECKPOINT=/path/... ./run_ur7e_act_real.sh         #
# #                                                                          #
# #    # Override robot IP / calib / port / device:                          #
# #    ROBOT_IP=192.168.1.50 ACT_PORT=5601 ACT_DEVICE=cuda \                 #
# #        ACT_CHECKPOINT=/path/... ./run_ur7e_act_real.sh                   #
# #                                                                          #
# #    # Pass extra launch args after the script args:                       #
# #    ACT_CHECKPOINT=/path/... ./run_ur7e_act_real.sh launch_rviz:=false    #
# #                                                                          #
# #  ENV:                                                                    #
# #    ROBOT_IP  (default 192.168.10.11)   CALIB (optional kinematics YAML)   #
# #    HEADLESS  (true|1 -> Method B)      START_MODE (gello | init_align)    #
# #    ACT_VENV  (default ros2_ur_ws/act_venv)   ACT_CHECKPOINT (REQUIRED)     #
# #    ACT_PORT  (default 5591)            ACT_DEVICE (default cuda)          #
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

# --- ACT (py3.12 server) settings -------------------------------------------
# Default venv = ros2_ur_ws/act_venv (created per the deploy docs). Override with
# ACT_VENV=/path if your py3.12 lerobot+torch venv lives elsewhere.
ACT_VENV="${ACT_VENV:-$SCRIPT_DIR/act_venv}"
ACT_PORT="${ACT_PORT:-5591}"
ACT_DEVICE="${ACT_DEVICE:-cuda}"
ACT_HOST="${ACT_HOST:-127.0.0.1}"
ACT_CHECKPOINT="${ACT_CHECKPOINT:-}"
RUN_ACT_SERVER="$SCRIPT_DIR/src/gello_policy/scripts/run_act_server.sh"

if [ -z "${ACT_CHECKPOINT}" ]; then
    echo "ERROR: ACT_CHECKPOINT is required (path to the trained ACT pretrained_model dir)." >&2
    echo "       Download it first:" >&2
    echo "         CKPT=\$(src/gello_policy/scripts/download_checkpoint.sh | tail -1)" >&2
    echo "         ACT_CHECKPOINT=\"\$CKPT\" ./run_ur7e_act_real.sh" >&2
    exit 1
fi
if [ ! -f "${RUN_ACT_SERVER}" ]; then
    echo "ERROR: run_act_server.sh not found at ${RUN_ACT_SERVER}." >&2
    exit 1
fi

# Also honor headless_mode:=true passed as a launch arg so the banner can never
# disagree with the effective launch (ros2 launch is last-wins on dupes).
case " $* " in *" headless_mode:=true "*|*"headless_mode:=true"*) HEADLESS=true ;; esac

# --- ROS2 Humble environment -------------------------------------------------
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

# --- Start the ACT inference server in the BACKGROUND (before ros2 launch) ---
# It loads the checkpoint and listens on ACT_HOST:ACT_PORT. The synthetic leader
# only queries it AFTER ~/start_execution, so it has ample time to warm up. We do
# NOT exec the launch below, so this script's EXIT trap fires on Ctrl-C and tears
# the server down together with the launch.
echo "### Starting ACT server (py3.12): venv=${ACT_VENV} device=${ACT_DEVICE} bind=${ACT_HOST}:${ACT_PORT}"
echo "### checkpoint=${ACT_CHECKPOINT}"
ACT_VENV="${ACT_VENV}" ACT_HOST="${ACT_HOST}" ACT_PORT="${ACT_PORT}" ACT_DEVICE="${ACT_DEVICE}" \
    ACT_CHECKPOINT="${ACT_CHECKPOINT}" \
    bash "${RUN_ACT_SERVER}" &
ACT_SERVER_PID=$!
# Kill the server on ANY exit of this script (clean exit, error, or Ctrl-C).
trap 'kill "${ACT_SERVER_PID}" 2>/dev/null || true' EXIT

# Health-check (review M3): catch an immediately-dying server (bad venv, missing
# checkpoint, torch import failure) NOW, instead of as a confusing ZMQ timeout at
# ~/start_execution. The leader is operator-gated, so we don't need full readiness —
# just confirm the process didn't crash on startup (and, if `ss` is available, that
# the port is actually listening).
echo "### Waiting for ACT server to come up (pid ${ACT_SERVER_PID})..."
for i in $(seq 1 30); do
    if ! kill -0 "${ACT_SERVER_PID}" 2>/dev/null; then
        echo "ERROR: ACT server (pid ${ACT_SERVER_PID}) exited during startup — see its output above." >&2
        exit 1
    fi
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn 2>/dev/null | grep -qE ":${ACT_PORT}([^0-9]|$)"; then
            echo "### ACT server is listening on ${ACT_HOST}:${ACT_PORT}."
            break
        fi
    elif [ "$i" -ge 3 ]; then
        echo "### ACT server process alive (port check unavailable — ss not found)."
        break
    fi
    sleep 1
done

# --- Build the ros2 launch args ---------------------------------------------
ARGS=(
    robot_ip:="${ROBOT_IP}"
    start_mode:="${START_MODE}"
    act_host:="${ACT_HOST}"
    act_port:="${ACT_PORT}"
    checkpoint_path:="${ACT_CHECKPOINT}"
)
[ -n "${CALIB}" ] && ARGS+=(kinematics_params_file:="${CALIB}")

if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    ARGS+=(headless_mode:=true)
    HEADLESS_STATE="true (Method B — REMOTE mode required, no pendant Play)"
else
    ARGS+=(headless_mode:=false)
    HEADLESS_STATE="false (Method A — PLAY External Control on the pendant)"
fi

echo "### REAL UR7e ACT deploy | robot_ip=${ROBOT_IP} | calib=${CALIB:-<none>}"
echo "### headless_mode=${HEADLESS_STATE}"
if [ "${START_MODE}" = "init_align" ]; then
    echo "### start_mode=init_align — arm parks at the init pose; then it chases the held start pose."
else
    echo "### start_mode=gello — arm drives straight to the policy's HELD start pose and parks."
fi
echo "### The ACT policy is AUTONOMOUS but GATED: after the handshake the arm PARKS."
echo "### To begin motion:  ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger"
echo "### Robotiq 2F-85 gripper INCLUDED (Modbus over driver socat bridge /tmp/ttyUR)."
echo "### ROBOT MUST BE POWERED ON. Keep fingers clear (gripper auto-cal on connect)."
echo "### Ctrl-C tears down BOTH the ACT server (pid ${ACT_SERVER_PID}) and this launch."
if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    echo "### HEADLESS: ensure the pendant is in REMOTE mode + Real Robot (not Simulation)."
else
    echo "### Confirm External Control is PLAYING on the pendant before the handshake."
fi

# NOT exec'd on purpose: when the launch exits (including on Ctrl-C) control
# returns here and the EXIT trap kills the ACT server so nothing is orphaned.
ros2 launch gello_policy ur7e_act_real.launch.py "${ARGS[@]}" "$@"
