#!/usr/bin/env bash
# Drive the REAL UR7e AUTONOMOUSLY with the trained IFQL policy (carrot-in-pot) over ROS2 Humble.
#
# ############################################################################
# #  ⚠️  REAL HARDWARE — THE UR7e WILL PHYSICALLY MOVE, DRIVEN BY A POLICY.   #
# #                                                                          #
# #  This starts TWO processes:                                             #
# #    (A) the IFQL inference server (torch ResNet18 encoder + JAX agent,    #
# #        venv ~/carrot_ifql/.venv-svf, code ~/carrot_ifql/code — OUTSIDE   #
# #        this repo), started in the BACKGROUND first. It restores          #
# #        params_<STEP>.pkl from IFQL_RUN_DIR, warms up 3x, THEN binds a     #
# #        ZMQ REP socket on localhost :IFQL_PORT (5595), and                #
# #    (B) the Humble (py3.10) ros2 launch (arm driver + synthetic policy     #
# #        leader + bridge handshake + Robotiq gripper).                     #
# #  They live on different interpreters; localhost ZMQ joins them.         #
# #  Ctrl-C tears down BOTH (an EXIT trap kills the server).                 #
# #                                                                          #
# #  The IFQL server speaks the IDENTICAL ZMQ wire + 7-D joint action        #
# #  contract as ACT/diffusion/FM (reset / act, 7 floats back), so this      #
# #  REUSES the policy-agnostic ur7e_diffusion_real.launch.py, only          #
# #  overriding params_file -> ifql_deploy.yaml and act_port -> IFQL_PORT.   #
# #                                                                          #
# #  Before running:                                                        #
# #   1) Workspace clear; keep the teach-pendant E-STOP within reach.        #
# #   2) CAMERAS FIRST, in their own terminal:  ./launch_cameras.sh          #
# #      (its viewer carries the START / HOLD buttons). Without both cams    #
# #      ~/start_execution is refused and EXECUTE FAULTs on stale obs.      #
# #   3) START THE ROBOT program — pick ONE method (do NOT mix them):        #
# #      • Method A (default, HEADLESS unset/false): on the pendant, load +   #
# #        PLAY the External Control program.                               #
# #      • Method B (HEADLESS=true): the driver sends URScript directly (no   #
# #        pendant Play). REQUIRES the robot in REMOTE mode.                  #
# #   4) The arm first drives to the policy's HELD START POSE (the carrot    #
# #      corpus frame-0 pose, ifql_deploy.yaml) and PARKS there. It stays   #
# #      still until you explicitly START (viewer button, or):               #
# #        ros2 service call /policy_leader_node/start_execution \           #
# #            std_srvs/srv/Trigger                                          #
# #      Only THEN does the IFQL policy begin autonomous motion. On any       #
# #      server timeout the leader FAULTS and the arm halts (fail-silent).   #
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
# #    ./run_ur7e_ifql_real.sh                    # bon, K=32 (default)       #
# #    IFQL_SAMPLER=bc ./run_ur7e_ifql_real.sh    # plain BC control (K=1)    #
# #    IFQL_NUM_SAMPLES=16 ./run_ur7e_ifql_real.sh   # bon, K=16              #
# #    HEADLESS=true ./run_ur7e_ifql_real.sh      # Method B (REMOTE mode)    #
# #    ./run_ur7e_ifql_real.sh launch_rviz:=false # extra launch args pass    #
# #                                                through                    #
# #                                                                          #
# #  The paired bc-vs-bon protocol switches samplers by RESTARTING this       #
# #  script (Ctrl-C, then relaunch with the other IFQL_SAMPLER). Port 5595    #
# #  must be FREE at start — the script refuses to run next to a stale       #
# #  server (it would be the stale one driving the arm). Check/free it:      #
# #      ss -ltnp | grep :5595                                               #
# #                                                                          #
# #  The server MUST be up before the ROS side (this script orders it so):   #
# #  it loads + warms up BEFORE binding, so "port listening" already means   #
# #  warmup passed. Wait budget: IFQL_WARMUP_TIMEOUT_S (default 120 s; CPU  #
# #  startup with JAX + torch can be slow).                                  #
# #                                                                          #
# #  GPU: IFQL_DEVICE=auto (default) picks cuda when nvidia-smi sees a GPU,  #
# #  else cpu (laptop3's RTX 3060 is only usable after a reboot onto the     #
# #  non-RT kernel). On CPU the refill is slower: if the server prints       #
# #  "WARNING: warmup refill ... > 300 ms", LOWER K (IFQL_NUM_SAMPLES=16),   #
# #  do NOT widen the yaml timeouts (0.8 / 0.9 / 1.0 are already the         #
# #  widened set; the leader must stay the primary fault owner).             #
# #                                                                          #
# #  norm_stats GUARD: the server log line "norm_stats: <path>" MUST name a  #
# #  *real_lead6* file, otherwise the script kills the server and exits —   #
# #  a lead0 / sim norm_stats denormalises joints into a different box and  #
# #  the yaml envelope clamp does NOT catch the lead0 case (same action      #
# #  box). Sampler line is checked too ("sampler: kind=<IFQL_SAMPLER> K=N"). #
# #                                                                          #
# #  ENV:                                                                    #
# #    ROBOT_IP  (default 192.168.10.11)   CALIB (optional kinematics YAML)   #
# #    HEADLESS  (true|1 -> Method B)      START_MODE (gello ONLY, see below)#
# #    IFQL_ROOT (default /home/laptop3/carrot_ifql)                         #
# #    IFQL_PY   (default $IFQL_ROOT/.venv-svf/bin/python)                   #
# #    IFQL_CODE_ROOT (default $IFQL_ROOT/code; ifql_server.py is found by  #
# #              `find` under it — exactly ONE hit required)                #
# #    IFQL_SERVER_PY (explicit path to ifql_server.py; skips the find)     #
# #    IFQL_RUN_DIR (default $IFQL_ROOT/hf/ifql_real_lead6_k0.9_s0)          #
# #    IFQL_STEP (default 100000)          IFQL_SAMPLER (bon|bc|actor)       #
# #    IFQL_NUM_SAMPLES (default 32)       IFQL_PORT (default 5595)          #
# #    IFQL_BUDGET_S (default 0.6; server-side refill log budget only)       #
# #    IFQL_NORM_STATS (default $IFQL_RUN_DIR/norm_stats_r18_ss_real_lead6.json)
# #    IFQL_LOG_DIR (default $IFQL_ROOT/eval_runs/real_<YYYYMMDD>/           #
# #              ifql_lead6_<sampler tag>/<HHMMSS>; one subdir PER LAUNCH — #
# #              the server names episodes ep_0001.npz.. from 1 on every    #
# #              start and os.replace()s, so sharing a dir across restarts  #
# #              would overwrite the previous run's episodes)                #
# #    IFQL_DEVICE (auto|cuda|cpu, default auto)                             #
# #    IFQL_WARMUP_TIMEOUT_S (default 120)                                   #
# #    IFQL_PARAMS_FILE (default <this ws>/src/gello_policy/config/          #
# #              ifql_deploy.yaml — the SRC copy, never the install share)  #
# #    GELLO_REPO_ROOT (default <ros2_ur_ws>/..)                             #
# ############################################################################
set -e

ROBOT_IP="${ROBOT_IP:-192.168.10.11}"       # override: ROBOT_IP=x.x.x.x
CALIB="${CALIB:-}"                           # optional: CALIB=/path/ur7e_calibration.yaml
HEADLESS="${HEADLESS:-}"                      # HEADLESS=true|1 -> Method B (no pendant Play; needs REMOTE mode)
START_MODE="${START_MODE:-gello}"            # gello ONLY (init_align refused below)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # = ros2_ur_ws
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"    # gello_software

# --- IFQL server settings (everything lives OUTSIDE this repo, under IFQL_ROOT) ---
IFQL_ROOT="${IFQL_ROOT:-/home/laptop3/carrot_ifql}"
IFQL_PY="${IFQL_PY:-$IFQL_ROOT/.venv-svf/bin/python}"
IFQL_CODE_ROOT="${IFQL_CODE_ROOT:-$IFQL_ROOT/code}"
IFQL_SERVER_PY="${IFQL_SERVER_PY:-}"
IFQL_RUN_DIR="${IFQL_RUN_DIR:-$IFQL_ROOT/hf/ifql_real_lead6_k0.9_s0}"
IFQL_STEP="${IFQL_STEP:-100000}"
IFQL_SAMPLER="${IFQL_SAMPLER:-bon}"
IFQL_NUM_SAMPLES="${IFQL_NUM_SAMPLES:-32}"
IFQL_HOST="${IFQL_HOST:-127.0.0.1}"
IFQL_PORT="${IFQL_PORT:-5595}"
IFQL_BUDGET_S="${IFQL_BUDGET_S:-0.6}"
IFQL_NORM_STATS="${IFQL_NORM_STATS:-$IFQL_RUN_DIR/norm_stats_r18_ss_real_lead6.json}"
IFQL_DEVICE="${IFQL_DEVICE:-auto}"
IFQL_WARMUP_TIMEOUT_S="${IFQL_WARMUP_TIMEOUT_S:-120}"

# Sampler tag for the log dir: bon carries K (bon32 / bon16); bc is K=1 by definition
# (the server ignores --num-samples), so "bc32" would be a lie; actor has no K.
case "${IFQL_SAMPLER}" in
    bon)   IFQL_TAG="bon${IFQL_NUM_SAMPLES}" ;;
    bc)    IFQL_TAG="bc" ;;
    actor) IFQL_TAG="actor" ;;
    *)
        echo "ERROR: IFQL_SAMPLER must be bon|bc|actor (got '${IFQL_SAMPLER}')." >&2
        exit 1 ;;
esac
IFQL_LOG_DIR="${IFQL_LOG_DIR:-$IFQL_ROOT/eval_runs/real_$(date +%Y%m%d)/ifql_lead6_${IFQL_TAG}/$(date +%H%M%S)}"

# --- start_mode: gello ONLY -----------------------------------------------------
# In start_mode=gello gello_move_to_start chases /gello/joint_states, i.e. the pose
# policy_leader_node HOLDs = policy_leader_node.start_pose in ifql_deploy.yaml (the
# carrot frame-0 pose). init_align would first drive the arm to the yaml's generic
# init_pose and then wait for a HUMAN GELLO to align with it — a gate a synthetic
# leader holding a different pose can never pass. Refuse rather than hang the arm
# in a wrong pose.
if [ "${START_MODE}" != "gello" ]; then
    echo "ERROR: START_MODE='${START_MODE}' is not supported by the IFQL deploy; only 'gello' is" >&2
    echo "       (init_align needs a human GELLO leader and uses the generic init_pose)." >&2
    exit 1
fi

# --- Resolve the server script: exactly ONE ifql_server.py under IFQL_CODE_ROOT ---
# The code tarball's internal layout is not pinned yet (today it unpacks to
# code/<snapshot>/vision_carrot/ifql_server.py); a find keeps this script valid when
# the snapshot dir is renamed, and 0 or >1 hits are refused instead of guessed.
if [ -z "${IFQL_SERVER_PY}" ]; then
    if [ ! -d "${IFQL_CODE_ROOT}" ]; then
        echo "ERROR: IFQL_CODE_ROOT '${IFQL_CODE_ROOT}' does not exist (unpack the code tarball there or set IFQL_SERVER_PY)." >&2
        exit 1
    fi
    mapfile -t _HITS < <(find "${IFQL_CODE_ROOT}" -maxdepth 6 -type f -name ifql_server.py 2>/dev/null | sort)
    if [ "${#_HITS[@]}" -eq 0 ]; then
        echo "ERROR: no ifql_server.py found under ${IFQL_CODE_ROOT} (set IFQL_SERVER_PY explicitly)." >&2
        exit 1
    elif [ "${#_HITS[@]}" -gt 1 ]; then
        echo "ERROR: ${#_HITS[@]} copies of ifql_server.py under ${IFQL_CODE_ROOT}; set IFQL_SERVER_PY to the one you mean:" >&2
        printf '         %s\n' "${_HITS[@]}" >&2
        exit 1
    fi
    IFQL_SERVER_PY="${_HITS[0]}"
fi
IFQL_SERVER_DIR="$(cd "$(dirname "${IFQL_SERVER_PY}")" && pwd)"

# --- Preflight: interpreter / checkpoint / norm_stats ------------------------------
if [ ! -x "${IFQL_PY}" ]; then
    echo "ERROR: IFQL_PY '${IFQL_PY}' is not executable (is .venv-svf created yet?)." >&2
    exit 1
fi
if [ ! -f "${IFQL_SERVER_PY}" ]; then
    echo "ERROR: ifql_server.py not found at ${IFQL_SERVER_PY}." >&2
    exit 1
fi
if [ ! -f "${IFQL_RUN_DIR}/flags.json" ] || [ ! -f "${IFQL_RUN_DIR}/params_${IFQL_STEP}.pkl" ]; then
    echo "ERROR: IFQL_RUN_DIR '${IFQL_RUN_DIR}' needs flags.json + params_${IFQL_STEP}.pkl (HF download incomplete?)." >&2
    exit 1
fi
if [ ! -f "${IFQL_NORM_STATS}" ]; then
    echo "ERROR: IFQL_NORM_STATS '${IFQL_NORM_STATS}' not found." >&2
    exit 1
fi
# The lead6 checkpoint MUST be paired with the lead6 norm_stats. Same check again on
# the server's own log below (it is the server's resolution that counts), but a wrong
# path is cheaper to refuse here.
case "$(basename "${IFQL_NORM_STATS}")" in
    *real_lead6*) ;;
    *)
        echo "ERROR: IFQL_NORM_STATS basename '$(basename "${IFQL_NORM_STATS}")' does not contain 'real_lead6'." >&2
        echo "       A lead0 / sim norm_stats denormalises joints into a different box. Refusing." >&2
        exit 1 ;;
esac

# --- Device: auto -> cuda iff nvidia-smi can see a GPU ------------------------------
# The encoder does `.to(device)` with no fallback of its own, so 'cuda' without a
# working driver dies at load. Laptop3's RTX 3060 needs the non-RT kernel (reboot).
if [ "${IFQL_DEVICE}" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        IFQL_DEVICE=cuda
    else
        IFQL_DEVICE=cpu
        echo "### NOTE: no usable NVIDIA GPU (nvidia-smi failed) -> IFQL_DEVICE=cpu. Expect slower refills;" >&2
        echo "###       if warmup reports > 300 ms, relaunch with IFQL_NUM_SAMPLES=16 (do NOT widen timeouts)." >&2
    fi
fi
case "${IFQL_DEVICE}" in cuda|cpu) ;; *) echo "ERROR: IFQL_DEVICE must be auto|cuda|cpu." >&2; exit 1 ;; esac

# Also honor headless_mode:=true passed as a launch arg so the banner can never
# disagree with the effective launch (ros2 launch is last-wins on dupes).
case " $* " in *" headless_mode:=true "*|*"headless_mode:=true"*) HEADLESS=true ;; esac

# --- ROS2 Humble environment -------------------------------------------------
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

# --- Params file: the SRC copy, by absolute path, NEVER the install share ------------
# ifql_deploy.yaml was added without a colcon build (the workspace's install/ is in
# use by a live data-collection session and must not be rebuilt/touched). The launch
# file takes params_file as a plain path, so pointing it at src/ is fully supported
# and does not depend on `ros2 pkg prefix gello_policy` having a copy.
IFQL_PARAMS_FILE="${IFQL_PARAMS_FILE:-${SCRIPT_DIR}/src/gello_policy/config/ifql_deploy.yaml}"
if [ ! -f "${IFQL_PARAMS_FILE}" ]; then
    echo "ERROR: IFQL params file not found at ${IFQL_PARAMS_FILE}." >&2
    exit 1
fi
# The launch file itself still comes from install/ (ros2 launch gello_policy ...);
# it must be present there from an earlier build.
if ! ros2 pkg prefix gello_policy >/dev/null 2>&1; then
    echo "ERROR: package gello_policy is not in this workspace's install/ (ur7e_diffusion_real.launch.py is needed from there)." >&2
    exit 1
fi

# --- Refuse to start if the ZMQ port is already occupied ---------------------
# A leftover server holding IFQL_PORT would make the NEW server die on bind, yet the
# health-check below could still see *something* listening and pass -> the leader would
# silently drive the arm from the STALE server. Fail loudly BEFORE spawning.
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":${IFQL_PORT}([^0-9]|$)"; then
    echo "ERROR: port ${IFQL_PORT} is already in use (a stale IFQL/SVF/FM server?)." >&2
    echo "       Starting now would leave the arm driven by that stale server. Free it first:" >&2
    echo "         ss -ltnp | grep :${IFQL_PORT}     # then stop that PID and re-check" >&2
    exit 1
fi

# --- Start the IFQL inference server in the BACKGROUND (before ros2 launch) ------
mkdir -p "${IFQL_LOG_DIR}"
SERVER_LOG="${IFQL_LOG_DIR}/server_stdout.log"
SERVER_CMD=(
    "${IFQL_PY}" "${IFQL_SERVER_PY}"
    --run-dir "${IFQL_RUN_DIR}" --step "${IFQL_STEP}"
    --sampler "${IFQL_SAMPLER}" --num-samples "${IFQL_NUM_SAMPLES}"
    --host "${IFQL_HOST}" --port "${IFQL_PORT}" --budget-s "${IFQL_BUDGET_S}"
    --norm-stats "${IFQL_NORM_STATS}" --log-dir "${IFQL_LOG_DIR}"
    --device "${IFQL_DEVICE}"
)
echo "### Starting IFQL server: py=${IFQL_PY}"
echo "###   server=${IFQL_SERVER_PY}"
echo "###   run_dir=${IFQL_RUN_DIR} step=${IFQL_STEP} sampler=${IFQL_SAMPLER} K=${IFQL_NUM_SAMPLES} device=${IFQL_DEVICE}"
echo "###   norm_stats=${IFQL_NORM_STATS}"
echo "###   bind=${IFQL_HOST}:${IFQL_PORT} budget_s=${IFQL_BUDGET_S} log_dir=${IFQL_LOG_DIR}"
echo "###   stdout/stderr mirrored to ${SERVER_LOG}"
# Run from the server's own directory: ifql_server.py imports its siblings
# (episode_logger, action_queue, ...) as top-level modules.
# TORCH_HOME: offline ResNet18 (+ torch.hub) weights. XLA_PYTHON_CLIENT_PREALLOCATE:
# keep JAX from grabbing the whole GPU next to torch. HF_HUB_OFFLINE: nothing here
# should ever reach for the Hub at runtime; harmless if unused.
# Process substitution (not a pipe) keeps $! = the python PID so the EXIT trap kills
# the server itself, while tee mirrors its output to the terminal and the log file.
(
    cd "${IFQL_SERVER_DIR}"
    TORCH_HOME="${TORCH_HOME:-$IFQL_ROOT/.cache/torch}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    HF_HUB_OFFLINE=1 \
    exec "${SERVER_CMD[@]}"
) > >(tee -a "${SERVER_LOG}") 2>&1 &
IFQL_SERVER_PID=$!
# Kill the server on ANY exit of this script (clean exit, error, or Ctrl-C).
trap 'kill "${IFQL_SERVER_PID}" 2>/dev/null || true' EXIT

# --- Health-check: alive + norm_stats/sampler lines + port listening -----------------
# The server logs "norm_stats: <path>  (source: ...)" early, then "sampler: kind=.. K=..",
# warms up, and only THEN binds ("REP bound tcp://..."). So:
#   * norm_stats line present but NOT naming real_lead6 -> kill + fail immediately;
#   * port listening -> warmup already passed -> proceed;
#   * process gone -> fail; IFQL_WARMUP_TIMEOUT_S elapsed -> fail.
echo "### Waiting for IFQL server (pid ${IFQL_SERVER_PID}) to load, warm up and listen on :${IFQL_PORT} (<= ${IFQL_WARMUP_TIMEOUT_S} s)..."
NORM_OK=""
SAMPLER_OK=""
LISTENING=""
for i in $(seq 1 "${IFQL_WARMUP_TIMEOUT_S}"); do
    if ! kill -0 "${IFQL_SERVER_PID}" 2>/dev/null; then
        echo "ERROR: IFQL server (pid ${IFQL_SERVER_PID}) exited during startup — see its output above / ${SERVER_LOG}." >&2
        exit 1
    fi
    if [ -z "${NORM_OK}" ] && [ -f "${SERVER_LOG}" ]; then
        # Loose match on purpose: the exact line is "[ifql-server] norm_stats: <path>  (source: ...)"
        # today, but only the "norm_stats" word and the basename are relied on.
        NORM_LINE="$(grep -m1 -E 'norm_stats:' "${SERVER_LOG}" || true)"
        if [ -n "${NORM_LINE}" ]; then
            # Check the BASENAME of the resolved path, not the whole line: the run dir is
            # itself named ifql_real_lead6_*, so a lead0 file dropped into it would pass a
            # whole-line grep. Token after "norm_stats:" = the path.
            NORM_PATH="$(printf '%s' "${NORM_LINE}" | sed -E 's/.*norm_stats:[[:space:]]*([^[:space:]]+).*/\1/')"
            NORM_BASE="$(basename "${NORM_PATH}")"
            case "${NORM_BASE}" in
                *real_lead6*)
                    NORM_OK=1
                    echo "### norm_stats check OK: ${NORM_LINE}" ;;
                *)
                    echo "ERROR: server resolved a norm_stats whose basename ('${NORM_BASE}') is NOT a real_lead6 file:" >&2
                    echo "       ${NORM_LINE}" >&2
                    echo "       (lead0/sim stats denormalise joints into a different box). Stopping the server." >&2
                    exit 1 ;;
            esac
        fi
    fi
    if [ -z "${SAMPLER_OK}" ] && [ -f "${SERVER_LOG}" ]; then
        SAMPLER_LINE="$(grep -m1 -E 'sampler: *kind=' "${SERVER_LOG}" || true)"
        if [ -n "${SAMPLER_LINE}" ]; then
            if [ "${IFQL_SAMPLER}" = "bon" ]; then
                WANT="kind=bon K=${IFQL_NUM_SAMPLES}"
            else
                WANT="kind=${IFQL_SAMPLER}"      # bc -> K=1 fixed by the server; actor -> no K check
            fi
            if printf '%s' "${SAMPLER_LINE}" | grep -qF "${WANT}"; then
                SAMPLER_OK=1
                echo "### sampler check OK: ${SAMPLER_LINE}"
            else
                echo "ERROR: server sampler line does not match the requested '${WANT}':" >&2
                echo "       ${SAMPLER_LINE}" >&2
                exit 1
            fi
        fi
    fi
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn 2>/dev/null | grep -qE ":${IFQL_PORT}([^0-9]|$)"; then
            LISTENING=1
        fi
    elif [ "$i" -ge 5 ] && grep -q 'REP bound' "${SERVER_LOG}" 2>/dev/null; then
        LISTENING=1     # ss unavailable: trust the server's own bind line
    fi
    if [ -n "${LISTENING}" ]; then
        break
    fi
    sleep 1
done
if [ -z "${LISTENING}" ]; then
    echo "ERROR: IFQL server did not start listening on :${IFQL_PORT} within ${IFQL_WARMUP_TIMEOUT_S} s (see ${SERVER_LOG})." >&2
    exit 1
fi
# The port is up only after warmup, so the norm_stats line must have been seen by now.
# (If the log format ever drops that line, fail closed rather than run unverified.)
if [ -z "${NORM_OK}" ]; then
    echo "ERROR: server is listening but no 'norm_stats:' line was seen in ${SERVER_LOG}; cannot verify real_lead6. Stopping." >&2
    exit 1
fi
if [ -z "${SAMPLER_OK}" ]; then
    echo "ERROR: server is listening but no 'sampler: kind=' line was seen in ${SERVER_LOG}; cannot verify the sampler. Stopping." >&2
    exit 1
fi
echo "### IFQL server is listening on ${IFQL_HOST}:${IFQL_PORT} (warmup done)."
if grep -q 'WARNING: warmup refill' "${SERVER_LOG}" 2>/dev/null; then
    echo "### ⚠️  $(grep -m1 'WARNING: warmup refill' "${SERVER_LOG}")" >&2
    echo "### ⚠️  Refill is slow for this box. Consider Ctrl-C and IFQL_NUM_SAMPLES=16 — do NOT widen the yaml timeouts." >&2
fi

# --- Build the ros2 launch args (REUSE the policy-agnostic diffusion launch) ---
# start_mode is a real override (launch -> gello_move_to_start param). The launch's
# own `start_pose` arg is LOGGING ONLY (it is never handed to policy_leader_node;
# the yaml's policy_leader_node.start_pose is authoritative), so it is not passed —
# there is nothing it could override.
ARGS=(
    robot_ip:="${ROBOT_IP}"
    start_mode:="${START_MODE}"
    params_file:="${IFQL_PARAMS_FILE}"
    act_host:="${IFQL_HOST}"
    act_port:="${IFQL_PORT}"
    checkpoint_path:="${IFQL_RUN_DIR}/params_${IFQL_STEP}.pkl"
)
[ -n "${CALIB}" ] && ARGS+=(kinematics_params_file:="${CALIB}")

if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    ARGS+=(headless_mode:=true)
    HEADLESS_STATE="true (Method B — REMOTE mode required, no pendant Play)"
else
    ARGS+=(headless_mode:=false)
    HEADLESS_STATE="false (Method A — PLAY External Control on the pendant)"
fi

echo "### REAL UR7e IFQL deploy | robot_ip=${ROBOT_IP} | calib=${CALIB:-<none>} | params=${IFQL_PARAMS_FILE}"
echo "### headless_mode=${HEADLESS_STATE}"
echo "### start_mode=gello — arm drives straight to the policy's HELD carrot start pose and parks."
echo "### The IFQL policy is AUTONOMOUS but GATED: after the handshake the arm PARKS."
echo "### To begin motion:  START button in the camera viewer, or"
echo "###                   ros2 service call /policy_leader_node/start_execution std_srvs/srv/Trigger"
echo "### To pause:         HOLD button, or ros2 service call /policy_leader_node/hold std_srvs/srv/Trigger"
echo "### Robotiq 2F-85 gripper INCLUDED (Modbus over driver socat bridge /tmp/ttyUR)."
echo "### ROBOT MUST BE POWERED ON. Keep fingers clear (gripper auto-cal on connect)."
echo "### Ctrl-C tears down BOTH the IFQL server (pid ${IFQL_SERVER_PID}) and this launch."
if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    echo "### HEADLESS: ensure the pendant is in REMOTE mode + Real Robot (not Simulation)."
else
    echo "### Confirm External Control is PLAYING on the pendant before the handshake."
fi

# NOT exec'd on purpose: when the launch exits (including on Ctrl-C) control
# returns here and the EXIT trap kills the IFQL server so nothing is orphaned.
ros2 launch gello_policy ur7e_diffusion_real.launch.py "${ARGS[@]}" "$@"
