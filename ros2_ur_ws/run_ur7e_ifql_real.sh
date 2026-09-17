#!/usr/bin/env bash
# Drive the REAL UR7e AUTONOMOUSLY with a trained IFQL policy over ROS2 Jazzy.
# The task is selected by IFQL_TASK (carrot | orange); carrot is the default and its
# behaviour is unchanged from the 2026-09-16 carrot session.
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
# #    (B) the Jazzy (py3.12) ros2 launch (arm driver + synthetic policy      #
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
# #   4) The arm first drives to the policy's HELD START POSE (the task      #
# #      corpus frame-0 pose, IFQL_PARAMS_FILE) and PARKS there. It stays   #
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
# #    ./run_ur7e_ifql_real.sh                    # carrot, bon, K=32 (dflt)  #
# #    IFQL_SAMPLER=bc ./run_ur7e_ifql_real.sh    # plain BC control (K=1)    #
# #    IFQL_NUM_SAMPLES=16 ./run_ur7e_ifql_real.sh   # bon, K=16              #
# #    IFQL_TASK=orange ./run_ur7e_ifql_real.sh   # orange-bowl task profile  #
# #    IFQL_DRY_RUN=1 ./run_ur7e_ifql_real.sh     # print the two commands    #
# #                                                and exit (starts nothing)  #
# #    HEADLESS=true ./run_ur7e_ifql_real.sh      # Method B (REMOTE mode)    #
# #    ./run_ur7e_ifql_real.sh launch_rviz:=false # extra launch args pass    #
# #                                                through                    #
# #                                                                          #
# #  TASK PROFILES (IFQL_TASK, default carrot; an unknown value is REFUSED). #
# #  A profile only supplies DEFAULTS for four things — each one is still    #
# #  overridable on its own, which is how the 3 orange candidates are run    #
# #  by changing IFQL_RUN_DIR alone:                                         #
# #                                                                          #
# #             carrot                          orange                       #
# #    run dir  $IFQL_ROOT/hf/                  $IFQL_ROOT/hf_orange/        #
# #             ifql_real_lead6_k0.9_s0         ifql_orange_k0.9_lead2_      #
# #                                             aug8_p0.5_s0                 #
# #    norm     $IFQL_RUN_DIR/norm_stats_       the single norm_stats_*.json #
# #    stats    r18_ss_real_lead6.json          inside $IFQL_RUN_DIR         #
# #    yaml     src/gello_policy/config/        src/gello_policy/config/     #
# #             ifql_deploy.yaml                ifql_deploy_orange.yaml      #
# #    log tag  ifql_lead6                      basename of $IFQL_RUN_DIR    #
# #                                             (the 3 candidates differ)    #
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
# #  norm_stats GUARD: the server log line "norm_stats: <path>" MUST carry   #
# #  the SAME BASENAME as IFQL_NORM_STATS, otherwise the script kills the    #
# #  server and exits. This REPLACES the old hard-coded "*real_lead6*"       #
# #  substring test, which was carrot-only and would refuse every other      #
# #  task outright. Exact-filename equality is task/lead/px agnostic AND     #
# #  tighter (the whole name must match, not a substring) — it proves the    #
# #  server really resolved the file we asked for (--norm-stats is always    #
# #  passed, so a mismatch means the server ignored it or the log format     #
# #  changed). The reason this matters at all: a lead0 / sim / other-task    #
# #  norm_stats denormalises joints into a DIFFERENT box and the yaml        #
# #  envelope clamp does NOT catch the lead0 case (same action box).         #
# #  Paired local guard: IFQL_NORM_STATS must live INSIDE IFQL_RUN_DIR (a    #
# #  released run ships its own norm_stats next to params_<step>.pkl); a     #
# #  file from anywhere else is refused unless                               #
# #  IFQL_ALLOW_FOREIGN_NORM_STATS=1. Together those two keep the old        #
# #  carrot protection (a lead0 path is refused) without naming a task.      #
# #  Sampler line is checked too ("sampler: kind=<IFQL_SAMPLER> K=N"), and   #
# #  the server's "agent ready: ... px=<bool>" line is checked against the   #
# #  run dir's own px verdict (flags.json, is_px_run rule).                  #
# #                                                                          #
# #  ENV:                                                                    #
# #    ROBOT_IP  (default 192.168.10.11)   CALIB (optional kinematics YAML)   #
# #    HEADLESS  (true|1 -> Method B)      START_MODE (gello ONLY, see below)#
# #    IFQL_TASK (carrot|orange, default carrot — picks the 4 defaults above)#
# #    IFQL_ROOT (default $HOME/carrot_ifql)                                #
# #    IFQL_PY   (interpreter resolution, first match wins):                #
# #              1) $IFQL_PY if explicitly set                              #
# #              2) $IFQL_ROOT/.venv-svf/bin/python if it exists            #
# #              3) conda env `il` (/home/junhyeong/miniconda3/envs/il/     #
# #                 bin/python) if it exists -- verified sufficient for     #
# #                 IFQL inference on THIS PC (torch/jax/flax/pyzmq/cv2);   #
# #                 that env is never mutated by this script                #
# #              else: refuse to start with a clear message                 #
# #    IFQL_CODE_ROOT (default $IFQL_ROOT/code; ifql_server.py is found by  #
# #              `find` under it — newest name-sorted hit is selected)      #
# #    IFQL_SERVER_PY (explicit path to ifql_server.py; skips the find)     #
# #    IFQL_RUN_DIR (task default, see the table above)                      #
# #    IFQL_STEP (default 100000)          IFQL_SAMPLER (bon|bc|actor)       #
# #    IFQL_NUM_SAMPLES (default 32)       IFQL_PORT (default 5595)          #
# #    IFQL_BUDGET_S (default 0.6; server-side refill log budget only)       #
# #    IFQL_NORM_STATS (task default, see the table above)                   #
# #    IFQL_ALLOW_FOREIGN_NORM_STATS (1 -> allow a norm_stats outside the   #
# #              run dir; default refuse)                                    #
# #    IFQL_LOG_TAG (task default, see the table above)                      #
# #    IFQL_LOG_DIR (default <this ws>/log/ifql/real_<YYYYMMDD>/            #
# #              <IFQL_LOG_TAG>_<sampler tag>/<HHMMSS>; one subdir PER      #
# #              LAUNCH — the server names episodes ep_0001.npz.. from 1 on #
# #              every start and os.replace()s, so sharing a dir across      #
# #              restarts would overwrite the previous run's episodes)       #
# #    IFQL_DEVICE (auto|cuda|cpu, default auto)                             #
# #    IFQL_WARMUP_TIMEOUT_S (default 120)                                   #
# #    IFQL_PARAMS_FILE (task default under <this ws>/src/gello_policy/      #
# #              config/ — the SRC copy, never the install share)           #
# #    IFQL_DRY_RUN (1 -> run every preflight, print the resolved server +  #
# #              ros2 launch commands, exit 0 WITHOUT starting either one)   #
# #    ROS child processes use /usr/bin/python3 before an active conda base; #
# #              Jazzy's UR tool-comm script requires distro python3-pytest. #
# #    GELLO_REPO_ROOT (default <ros2_ur_ws>/..)                             #
# ############################################################################
set -e

ROBOT_IP="${ROBOT_IP:-192.168.10.11}"       # override: ROBOT_IP=x.x.x.x
CALIB="${CALIB:-}"                           # optional: CALIB=/path/ur7e_calibration.yaml
HEADLESS="${HEADLESS:-}"                      # HEADLESS=true|1 -> Method B (no pendant Play; needs REMOTE mode)
START_MODE="${START_MODE:-gello}"            # gello ONLY (init_align refused below)
# Declared explicitly rather than trusted from $ROS_DISTRO: that var is EXPORTED by
# /opt/ros/<distro>/setup.bash itself, so if some earlier step in the caller's shell
# already sourced a DIFFERENT distro's setup.bash (e.g. Humble, on a box carrying
# both), $ROS_DISTRO would already say "humble" here and this script would silently
# source the wrong one. An explicit var sidesteps that ambiguity entirely.
GELLO_ROS_DISTRO="${GELLO_ROS_DISTRO:-jazzy}"

# Jazzy's UR driver launches the canonical tool-communication script directly:
#   /opt/ros/jazzy/lib/ur_client_library/tool_communication.py
# Its shebang is /usr/bin/env python3 and the script imports pytest for bundled
# self-tests. A shell with conda base active used to resolve that shebang to conda's
# Python 3.13, where pytest is absent, so ur_tool_comm exited before creating
# /tmp/ttyUR. Keep the complete ROS launch tree on the distro interpreter instead.
# IFQL itself is unaffected because IFQL_PY is always an absolute path.
JAZZY_SYSTEM_PYTHON="/usr/bin/python3"
JAZZY_TOOL_COMM_SCRIPT="/opt/ros/${GELLO_ROS_DISTRO}/lib/ur_client_library/tool_communication.py"
JAZZY_RUNTIME_PATH="/usr/bin:/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # = ros2_ur_ws
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"    # gello_software

# --- Task profile: the ONLY place a task name appears ------------------------------
# IFQL_TASK picks DEFAULTS for exactly four things (run dir, norm_stats, deploy yaml,
# log tag). Each stays individually overridable, which is how the three orange
# candidates are compared by changing IFQL_RUN_DIR alone. An unknown IFQL_TASK is
# refused (fail-closed): a typo must never silently fall back to carrot, because the
# carrot yaml's start_pose / joint envelope would then drive the arm for another task.
IFQL_TASK="${IFQL_TASK:-carrot}"
case "${IFQL_TASK}" in
    carrot)
        _TASK_HF_SUBDIR="hf"
        _TASK_RUN="ifql_real_lead6_k0.9_s0"
        # Literal, unchanged from the carrot session: this run dir ships exactly this file.
        _TASK_NORM_BASE="norm_stats_r18_ss_real_lead6.json"
        _TASK_YAML="ifql_deploy.yaml"
        _TASK_PARAMS="${SCRIPT_DIR}/src/gello_policy/config/ifql_deploy.yaml"
        _TASK_LOG_TAG="ifql_lead6"
        ;;
    orange)
        _TASK_HF_SUBDIR="hf_orange"
        _TASK_RUN="ifql_orange_k0.9_lead2_aug8_p0.5_s0"
        # Empty -> resolved from the run dir (see below). The three orange candidates
        # each ship a DIFFERENT norm_stats (the px run's is norm_stats_orange_lead2_px96
        # .json), so a single hard-coded name would be wrong for two of the three.
        _TASK_NORM_BASE=""
        _TASK_YAML="ifql_deploy_orange.yaml"
        _TASK_PARAMS="${SCRIPT_DIR}/src/gello_policy/config/ifql_deploy_orange.yaml"
        # Empty -> basename of the run dir, so the three candidates never share a log dir.
        _TASK_LOG_TAG=""
        ;;
    *)
        echo "ERROR: IFQL_TASK must be carrot|orange (got '${IFQL_TASK}')." >&2
        exit 1 ;;
esac

# --- IFQL server settings (everything lives OUTSIDE this repo, under IFQL_ROOT) ---
IFQL_ROOT="${IFQL_ROOT:-$HOME/carrot_ifql}"
# IFQL_PY resolution (first match wins; each fallback is a LAST RESORT, not a
# preference, so this only ever picks up an existing interpreter -- it never
# creates or mutates one):
#   1) IFQL_PY, if the caller set it explicitly;
#   2) $IFQL_ROOT/.venv-svf/bin/python, the project-local venv this script was
#      originally written against;
#   3) the conda env `il` (/home/junhyeong/miniconda3/envs/il/bin/python) --
#      on THIS PC .venv-svf does not exist, but `il` was verified sufficient
#      for IFQL inference (torch 2.12.0+cu130, jax 0.5.3, flax 0.10.5, pyzmq
#      27.1.0, cv2 4.11). This script never `pip install`s into it.
# If none of the three exist, fail now with a clear message instead of letting
# the later "IFQL_PY is not executable" preflight print a path nobody chose.
if [ -z "${IFQL_PY:-}" ]; then
    _IFQL_PY_VENV="${IFQL_ROOT}/.venv-svf/bin/python"
    _IFQL_PY_CONDA="/home/junhyeong/miniconda3/envs/il/bin/python"
    if [ -x "${_IFQL_PY_VENV}" ]; then
        IFQL_PY="${_IFQL_PY_VENV}"
    elif [ -x "${_IFQL_PY_CONDA}" ]; then
        IFQL_PY="${_IFQL_PY_CONDA}"
        echo "### IFQL_PY not set and ${_IFQL_PY_VENV} does not exist -- falling back to conda env 'il': ${IFQL_PY}"
    else
        echo "ERROR: could not resolve an IFQL interpreter. Tried:" >&2
        echo "         ${_IFQL_PY_VENV}  (IFQL_ROOT/.venv-svf, does not exist)" >&2
        echo "         ${_IFQL_PY_CONDA}  (conda env 'il', does not exist)" >&2
        echo "       Set IFQL_PY explicitly to an interpreter with torch+jax+flax+pyzmq+cv2." >&2
        exit 1
    fi
fi
IFQL_CODE_ROOT="${IFQL_CODE_ROOT:-$IFQL_ROOT/code}"
IFQL_SERVER_PY="${IFQL_SERVER_PY:-}"
IFQL_RUN_DIR="${IFQL_RUN_DIR:-$IFQL_ROOT/${_TASK_HF_SUBDIR}/${_TASK_RUN}}"
IFQL_STEP="${IFQL_STEP:-100000}"
IFQL_SAMPLER="${IFQL_SAMPLER:-bon}"
IFQL_NUM_SAMPLES="${IFQL_NUM_SAMPLES:-32}"
IFQL_HOST="${IFQL_HOST:-127.0.0.1}"
IFQL_PORT="${IFQL_PORT:-5595}"
IFQL_BUDGET_S="${IFQL_BUDGET_S:-0.6}"
IFQL_DEVICE="${IFQL_DEVICE:-auto}"
IFQL_WARMUP_TIMEOUT_S="${IFQL_WARMUP_TIMEOUT_S:-120}"
IFQL_ALLOW_FOREIGN_NORM_STATS="${IFQL_ALLOW_FOREIGN_NORM_STATS:-}"
IFQL_DRY_RUN="${IFQL_DRY_RUN:-}"
REAL_EVAL_RECORDING="${REAL_EVAL_RECORDING:-1}"
REAL_EVAL_MIN_FREE_GIB="${REAL_EVAL_MIN_FREE_GIB:-10}"
REAL_EVAL_FINALIZE_TIMEOUT_S="${REAL_EVAL_FINALIZE_TIMEOUT_S:-20}"
REAL_EVAL_POLICY_TYPE="${REAL_EVAL_POLICY_TYPE:-ifql}"
case "${REAL_EVAL_RECORDING}" in 0|1) ;; *) echo "ERROR: REAL_EVAL_RECORDING must be 0 or 1." >&2; exit 1 ;; esac

# norm_stats default. With a profile basename (carrot) it is the same literal path the
# carrot session used. Without one (orange) the run dir must contain EXACTLY ONE
# norm_stats_*.json — 0 or >1 is refused rather than guessed, exactly like the
# ifql_server.py find above.
# A run dir that ships a DIFFERENT norm_stats than the profile's literal name (e.g. the
# carrot px run: norm_stats_real_px96_lead6.json) falls through to the exactly-one rule
# below instead of failing on the literal — the post-start basename guard still pins
# whatever is chosen here to what the server actually loaded.
if [ -z "${IFQL_NORM_STATS:-}" ] && [ -n "${_TASK_NORM_BASE}" ] && [ -f "${IFQL_RUN_DIR}/${_TASK_NORM_BASE}" ]; then
    IFQL_NORM_STATS="${IFQL_RUN_DIR}/${_TASK_NORM_BASE}"
fi
if [ -z "${IFQL_NORM_STATS:-}" ]; then
    mapfile -t _NS_HITS < <(find "${IFQL_RUN_DIR}" -maxdepth 1 -type f -name 'norm_stats_*.json' 2>/dev/null | sort)
    if [ "${#_NS_HITS[@]}" -ne 1 ]; then
        echo "ERROR: cannot pick a norm_stats for IFQL_TASK=${IFQL_TASK}: ${#_NS_HITS[@]} norm_stats_*.json in" >&2
        echo "       ${IFQL_RUN_DIR} (need exactly 1). Set IFQL_NORM_STATS explicitly." >&2
        printf '         %s\n' "${_NS_HITS[@]}" >&2
        exit 1
    fi
    IFQL_NORM_STATS="${_NS_HITS[0]}"
fi

# Deploy yaml default: task-specific (carrot -> ifql_deploy.yaml, orange ->
# ifql_deploy_orange.yaml), always the SRC copy under this workspace by absolute path
# (never the install share — see the "Params file" note below, near where it is used).
IFQL_PARAMS_FILE="${IFQL_PARAMS_FILE:-${_TASK_PARAMS}}"
if [ ! -f "${IFQL_PARAMS_FILE}" ]; then
    echo "ERROR: IFQL params file not found at ${IFQL_PARAMS_FILE}." >&2
    exit 1
fi
IFQL_COMPAT="${IFQL_COMPAT:-auto}"
case "$IFQL_COMPAT" in
    auto|0|1) ;;
    *) echo "ERROR: IFQL_COMPAT must be auto, 0, or 1." >&2; exit 1 ;;
esac

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
IFQL_LOG_TAG="${IFQL_LOG_TAG:-${_TASK_LOG_TAG:-$(basename "${IFQL_RUN_DIR}")}}"
# Keep IFQL_LOG_DIR as a compatibility input for an existing per-task root, but never
# give two policy launches the same leaf: episode writers restart their numbering on
# each server start. UTC nanoseconds + PID identify a run; mkdir below makes a collision
# fail closed rather than silently reusing a stale partial recording.
IFQL_LOG_ROOT="${IFQL_LOG_ROOT:-${IFQL_LOG_DIR:-$SCRIPT_DIR/log/ifql/real_$(date -u +%Y%m%d)/${IFQL_LOG_TAG}_${IFQL_TAG}}}"
REAL_EVAL_RUN_DIR="${REAL_EVAL_RUN_DIR:-${IFQL_LOG_ROOT}/run_$(date -u +%Y%m%dT%H%M%S.%NZ)_pid$$}"
IFQL_LOG_DIR="${REAL_EVAL_RUN_DIR}"

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

# --- Resolve the server script: ifql_server.py under IFQL_CODE_ROOT ---
# The code tarball's internal layout is not pinned yet (today it unpacks to
# code/<snapshot>/vision_carrot/ifql_server.py); a find keeps this script valid when
# the snapshot dir is renamed. 0 hits is refused; >1 hits (multiple snapshot dirs, e.g.
# 20260915 + 20260916) picks the name-sorted LAST one and logs every hit — set
# IFQL_SERVER_PY explicitly to pin one instead of trusting the sort.
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
        # Multiple code-tarball snapshots can legitimately coexist (e.g. 20260915 +
        # 20260916). Name-sort ascending and take the LAST = newest-looking snapshot
        # dir name by construction (YYYYMMDD-style suffixes sort correctly); this is a
        # convenience default, not a proof of recency, so it is logged loudly and
        # IFQL_SERVER_PY always overrides it.
        echo "### NOTE: ${#_HITS[@]} copies of ifql_server.py under ${IFQL_CODE_ROOT}; picking the name-sorted LAST one:" >&2
        printf '###          %s\n' "${_HITS[@]}" >&2
    fi
    IFQL_SERVER_PY="${_HITS[-1]}"
fi
IFQL_SERVER_DIR="$(cd "$(dirname "${IFQL_SERVER_PY}")" && pwd)"
REAL_EVAL_PREFLIGHT="${SCRIPT_DIR}/setup_jazzy/real_eval_recording_preflight.py"
REAL_EVAL_HDF5_LOG_DIR="${REAL_EVAL_HDF5_LOG_DIR:-${REAL_EVAL_RUN_DIR}/hdf5}"
REAL_EVAL_MP4_PATH="${REAL_EVAL_MP4_PATH:-${REAL_EVAL_RUN_DIR}/policy_render.mp4}"
# Team 1 owns this hook beside its server implementation. Keeping the hook explicit
# means a staged server can use a different renderer without changing policy safety.
REAL_EVAL_RENDERER_HOOK="${REAL_EVAL_RENDERER_HOOK:-${SCRIPT_DIR}/../sim_collect/eval/render_eval_video.py}"
# This is only a path until the unique run directory is created below.  Defining it
# here keeps the pre-spawn banner truthful without changing any server or ROS command.
SERVER_LOG="${REAL_EVAL_RUN_DIR}/server_stdout.log"

# --- Preflight: interpreter / checkpoint / norm_stats ------------------------------
if [ ! -x "${IFQL_PY}" ]; then
    echo "ERROR: IFQL_PY '${IFQL_PY}' is not executable (is .venv-svf created yet?)." >&2
    exit 1
fi
if [ ! -f "${IFQL_SERVER_PY}" ]; then
    echo "ERROR: ifql_server.py not found at ${IFQL_SERVER_PY}." >&2
    exit 1
fi
if [ "$IFQL_COMPAT" = auto ]; then
    IFQL_COMPAT=0
    if [ ! -f "${IFQL_RUN_DIR}/params_${IFQL_STEP}.pkl" ] && [ -f "${IFQL_RUN_DIR}/params_${IFQL_STEP}.infer.pkl" ]; then
        IFQL_COMPAT=1
    fi
fi
IFQL_SERVER_ENTRY=("${IFQL_SERVER_PY}")
IFQL_CHECKPOINT="${IFQL_RUN_DIR}/params_${IFQL_STEP}.pkl"
if [ "$IFQL_COMPAT" = 1 ]; then
    IFQL_ADAPTER="$SCRIPT_DIR/setup_jazzy/ifql_server_compat.py"
    [ -f "$IFQL_ADAPTER" ] || { echo "ERROR: missing IFQL compatibility adapter: $IFQL_ADAPTER" >&2; exit 1; }
    IFQL_SERVER_ENTRY=("$IFQL_ADAPTER" --server-script "$IFQL_SERVER_PY")
    [ -f "$IFQL_CHECKPOINT" ] || IFQL_CHECKPOINT="${IFQL_RUN_DIR}/params_${IFQL_STEP}.infer.pkl"
    echo "### IFQL compatibility adapter enabled (r18_ss, inference-checkpoint support): $IFQL_ADAPTER"
fi
if [ ! -f "${IFQL_RUN_DIR}/flags.json" ] || [ ! -f "$IFQL_CHECKPOINT" ]; then
    echo "ERROR: IFQL_RUN_DIR '${IFQL_RUN_DIR}' needs flags.json + $IFQL_CHECKPOINT." >&2
    exit 1
fi
if [ ! -f "${IFQL_NORM_STATS}" ]; then
    echo "ERROR: IFQL_NORM_STATS '${IFQL_NORM_STATS}' not found." >&2
    exit 1
fi
# norm_stats must live INSIDE IFQL_RUN_DIR (a released run ships its own norm_stats
# next to params_<step>.pkl) — task/lead/px agnostic on purpose, unlike the old
# carrot-only "*real_lead6*" substring test this replaces. The FILE match that actually
# proves the checkpoint got the right box is the server's own log line, checked below
# AFTER it starts (it is the server's resolution that counts); this is the cheap local
# guard against an obviously wrong path (e.g. a lead0/sim file dropped in by hand).
IFQL_NORM_STATS_REAL="$(readlink -f "${IFQL_NORM_STATS}" 2>/dev/null || echo "${IFQL_NORM_STATS}")"
IFQL_RUN_DIR_REAL="$(readlink -f "${IFQL_RUN_DIR}" 2>/dev/null || echo "${IFQL_RUN_DIR}")"
case "${IFQL_NORM_STATS_REAL}" in
    "${IFQL_RUN_DIR_REAL}"/*) ;;
    *)
        if [ "${IFQL_ALLOW_FOREIGN_NORM_STATS}" = "1" ]; then
            echo "### NOTE: IFQL_NORM_STATS '${IFQL_NORM_STATS}' is OUTSIDE IFQL_RUN_DIR '${IFQL_RUN_DIR}'; allowed by IFQL_ALLOW_FOREIGN_NORM_STATS=1." >&2
        else
            echo "ERROR: IFQL_NORM_STATS '${IFQL_NORM_STATS}' is not inside IFQL_RUN_DIR '${IFQL_RUN_DIR}'." >&2
            echo "       A released run ships its own norm_stats next to params_<step>.pkl; a file from" >&2
            echo "       elsewhere risks a lead0/sim/other-task box. Set IFQL_ALLOW_FOREIGN_NORM_STATS=1 to override." >&2
            exit 1
        fi ;;
esac

# --- px detection: mirrors ifql_server.py::is_px_run exactly (flags.json) -----------
# A run is pixel end-to-end iff its agent.encoder starts with "multicam", or its
# env_name ends in "_px<digits>(_val).npz". Computed here (not guessed) so the banner
# and the post-start "agent ready: ... px=<bool>" log check have the same expectation.
IS_PX_RUN="$(python3 - "${IFQL_RUN_DIR}/flags.json" <<'PYEOF'
import json, re, sys
flags = json.load(open(sys.argv[1]))
agent_cfg = flags.get("agent") if isinstance(flags.get("agent"), dict) else {}
enc = (agent_cfg or {}).get("encoder")
if isinstance(enc, str) and enc.startswith("multicam"):
    print("true")
else:
    env_name = flags.get("env_name")
    print("true" if (isinstance(env_name, str) and re.search(r"_px\d+(_val)?\.npz$", env_name)) else "false")
PYEOF
)"
if [ "${IS_PX_RUN}" = "true" ]; then
    echo "### NOTE: IFQL_RUN_DIR is a PX (pixel end-to-end) run (flags.json is_px_run rule). ⚠️ Refill/serving latency for px runs is UNMEASURED on this box — check server WARNING lines closely." >&2
fi

# --- Preflight: deploy yaml act_port must match IFQL_PORT --------------------------
IFQL_YAML_ACT_PORT="$(python3 - "${IFQL_PARAMS_FILE}" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    doc = yaml.safe_load(f)
try:
    print(doc["policy_leader_node"]["ros__parameters"]["act_port"])
except Exception:
    print("")
PYEOF
)"
if [ -z "${IFQL_YAML_ACT_PORT}" ]; then
    echo "ERROR: could not read policy_leader_node.ros__parameters.act_port from '${IFQL_PARAMS_FILE}'." >&2
    exit 1
fi
if [ "${IFQL_YAML_ACT_PORT}" != "${IFQL_PORT}" ]; then
    echo "ERROR: deploy yaml act_port (${IFQL_YAML_ACT_PORT}) != IFQL_PORT (${IFQL_PORT}). The leader would dial the wrong port." >&2
    exit 1
fi

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

# --- Jazzy UR tool-communication interpreter preflight ---------------------------
# ur_control.launch.py executes JAZZY_TOOL_COMM_SCRIPT by pathname, so its
# /usr/bin/env python3 shebang follows PATH inherited from this launcher. Do this
# before sourcing ROS and retain it for ros2 launch; otherwise an active conda base
# can hide the distro's python3-pytest and leave the Robotiq serial PTY absent.
if [ ! -x "${JAZZY_SYSTEM_PYTHON}" ]; then
    echo "ERROR: Jazzy runtime Python is missing or not executable: ${JAZZY_SYSTEM_PYTHON}" >&2
    exit 1
fi
if [ ! -f "${JAZZY_TOOL_COMM_SCRIPT}" ]; then
    echo "ERROR: Jazzy UR tool-communication script is missing: ${JAZZY_TOOL_COMM_SCRIPT}" >&2
    echo "       Install/repair ros-${GELLO_ROS_DISTRO}-ur-client-library before running IFQL." >&2
    exit 1
fi

# PYTHONHOME can make even an absolute /usr/bin/python3 import from conda. It is not
# needed by ROS and must not leak into the direct tool-communication subprocess.
unset PYTHONHOME
export PATH="${JAZZY_RUNTIME_PATH}"
if [ "$(command -v python3)" != "${JAZZY_SYSTEM_PYTHON}" ]; then
    echo "ERROR: could not prioritize ${JAZZY_SYSTEM_PYTHON} for Jazzy child processes." >&2
    echo "       Resolved python3: $(command -v python3)" >&2
    exit 1
fi
if ! JAZZY_PYTEST_PATH="$(/usr/bin/env python3 -c 'import pytest; print(pytest.__file__)' 2>/dev/null)"; then
    echo "ERROR: Jazzy UR tool communication cannot import pytest with ${JAZZY_SYSTEM_PYTHON}." >&2
    echo "       ${JAZZY_TOOL_COMM_SCRIPT} imports pytest before it can create /tmp/ttyUR." >&2
    echo "       Install the distro dependency python3-pytest, then rerun this command." >&2
    echo "       This launcher will not install or use packages from conda base." >&2
    exit 1
fi
echo "### Jazzy UR tool-comm preflight OK: /usr/bin/env python3 -> ${JAZZY_SYSTEM_PYTHON}; pytest=${JAZZY_PYTEST_PATH}"

# --- ROS2 Jazzy environment -------------------------------------------------
source "/opt/ros/${GELLO_ROS_DISTRO}/setup.bash"
source "$SCRIPT_DIR/install/setup.bash"

# --- Params file: the SRC copy, by absolute path, NEVER the install share ------------
# ifql_deploy.yaml (or ifql_deploy_orange.yaml) was added without a colcon build (the
# workspace's install/ is in use by a live data-collection session and must not be
# rebuilt/touched). The launch file takes params_file as a plain path, so pointing it at
# src/ is fully supported and does not depend on `ros2 pkg prefix gello_policy` having a
# copy. IFQL_PARAMS_FILE itself was already resolved (task default or override) and
# existence-checked above, before the norm_stats/px/act_port preflight.
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

# --- Build the IFQL server command (it is printed before either process starts) ---
SERVER_CMD=(
    "${IFQL_PY}" -B "${IFQL_SERVER_ENTRY[@]}"
    --run-dir "${IFQL_RUN_DIR}" --step "${IFQL_STEP}"
    --sampler "${IFQL_SAMPLER}" --num-samples "${IFQL_NUM_SAMPLES}"
    --host "${IFQL_HOST}" --port "${IFQL_PORT}" --budget-s "${IFQL_BUDGET_S}"
    --norm-stats "${IFQL_NORM_STATS}" --log-dir "${IFQL_LOG_DIR}"
    --device "${IFQL_DEVICE}"
)
if [ "${REAL_EVAL_RECORDING}" = "1" ]; then
    # Team 1's server CLI. These are recording-only outputs; the policy wire protocol,
    # start pose, joint envelope, and timing settings above remain exactly untouched.
    SERVER_CMD+=(
        --hdf5-log-dir "${REAL_EVAL_HDF5_LOG_DIR}"
        --renderer-hook "${REAL_EVAL_RENDERER_HOOK}"
        --renderer-output-path "${REAL_EVAL_MP4_PATH}"
    )
fi
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
# QFLOW_DIR: ifql_server.py imports the agent tree (agents/, utils/) from $QFLOW_DIR and
# otherwise falls back to a hard-coded path on the training PC (/home/theo_lab/...). The
# tarball ships it as <snapshot>/qflow_svf_merged next to vision_carrot/.
# FMRL_CAM1_CROP / FMRL_CAM1_MODE MUST be unset: the real norm_stats were extracted with
# cam1_mode=null / D_f_cam1=1024, so any crop/mode override would silently change the
# 2055-D observation the checkpoint expects (extract_features.py, cam1 override block).
IFQL_QFLOW_DIR="${QFLOW_DIR:-$(dirname "${IFQL_SERVER_DIR}")/qflow_svf_merged}"
if [ ! -d "${IFQL_QFLOW_DIR}/agents" ]; then
    echo "ERROR: QFLOW_DIR=${IFQL_QFLOW_DIR} has no agents/ (expected <snapshot>/qflow_svf_merged; set QFLOW_DIR)." >&2
    exit 1
fi
echo "###   QFLOW_DIR=${IFQL_QFLOW_DIR}"

# --- Build the ros2 launch args (REUSE the policy-agnostic diffusion launch) -------
# Built here (before the server even spawns) so IFQL_DRY_RUN can print both resolved
# commands without starting anything. start_mode is a real override (launch ->
# gello_move_to_start param). The launch's own `start_pose` arg is LOGGING ONLY (it is
# never handed to policy_leader_node; the yaml's policy_leader_node.start_pose is
# authoritative), so it is not passed — there is nothing it could override.
ARGS=(
    robot_ip:="${ROBOT_IP}"
    start_mode:="${START_MODE}"
    params_file:="${IFQL_PARAMS_FILE}"
    act_host:="${IFQL_HOST}"
    act_port:="${IFQL_PORT}"
    checkpoint_path:="${IFQL_CHECKPOINT}"
)
[ -n "${CALIB}" ] && ARGS+=(kinematics_params_file:="${CALIB}")
if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    ARGS+=(headless_mode:=true)
    HEADLESS_STATE="true (Method B — REMOTE mode required, no pendant Play)"
else
    ARGS+=(headless_mode:=false)
    HEADLESS_STATE="false (Method A — PLAY External Control on the pendant)"
fi
EXTRA_LAUNCH_ARGS=("$@")

write_launch_manifest() {
    local phase="$1"
    export REAL_EVAL_MANIFEST_PHASE="$phase"
    export REAL_EVAL_SERVER_COMMAND="$(printf '%q ' "${SERVER_CMD[@]}")"
    export REAL_EVAL_ROS_COMMAND="ros2 launch gello_policy ur7e_diffusion_real.launch.py $(printf '%q ' "${ARGS[@]}" "${EXTRA_LAUNCH_ARGS[@]}")"
    export REAL_EVAL_HOSTNAME="$(hostname -f 2>/dev/null || hostname)"
    "${JAZZY_SYSTEM_PYTHON}" - "${REAL_EVAL_RUN_DIR}/launch_manifest.json" <<'PYEOF'
import hashlib, json, os, socket, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

def digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def named_path(name, value):
    return {"path": value, "sha256": digest(value)}

hdf5_dir = Path(os.environ["REAL_EVAL_HDF5_LOG_DIR"])
hdf5_files = [named_path(path.name, str(path)) for path in sorted(hdf5_dir.rglob("*.h5"))] if hdf5_dir.is_dir() else []
svf = {key.lower().removeprefix("real_eval_svf_"): os.environ[key]
       for key in ("REAL_EVAL_SVF_RUN_DIR", "REAL_EVAL_SVF_QFLOW_DIR", "REAL_EVAL_SVF_REAL_DATA")
       if os.environ.get(key)}
document = {
    "schema": "real-eval-recording/v1",
    "phase": os.environ["REAL_EVAL_MANIFEST_PHASE"],
    "hostname": os.environ.get("REAL_EVAL_HOSTNAME", socket.gethostname()),
    "policy": {
        "type": os.environ["REAL_EVAL_POLICY_TYPE"],
        "family": "svf" if os.environ["REAL_EVAL_POLICY_TYPE"] == "svf" else "ifql",
        "task": os.environ["IFQL_TASK"], "sampler": os.environ["IFQL_SAMPLER"],
        "num_samples": int(os.environ["IFQL_NUM_SAMPLES"]), "step": int(os.environ["IFQL_STEP"]),
        "device": os.environ["IFQL_DEVICE"], "svf_staged_provenance": svf or None,
    },
    "commands": {"server_shell": os.environ["REAL_EVAL_SERVER_COMMAND"], "ros_shell": os.environ["REAL_EVAL_ROS_COMMAND"]},
    "artifacts": {
        "run_dir": os.environ["REAL_EVAL_RUN_DIR"],
        "inputs": {
            "checkpoint": named_path("checkpoint", os.environ["IFQL_CHECKPOINT"]),
            "norm_stats": named_path("norm_stats", os.environ["IFQL_NORM_STATS"]),
            "flags": named_path("flags", str(Path(os.environ["IFQL_RUN_DIR"]) / "flags.json")),
            "server": named_path("server", os.environ["IFQL_SERVER_PY"]),
            "renderer_hook": named_path("renderer_hook", os.environ["REAL_EVAL_RENDERER_HOOK"]),
        },
        "outputs": {"hdf5_log_dir": str(hdf5_dir), "hdf5_files": hdf5_files,
                    "mp4": named_path("mp4", os.environ["REAL_EVAL_MP4_PATH"])},
    },
    "recording": {"enabled": os.environ["REAL_EVAL_RECORDING"] == "1",
                  "ffmpeg": os.environ.get("REAL_EVAL_FFMPEG") or None,
                  "min_free_gib": float(os.environ["REAL_EVAL_MIN_FREE_GIB"])},
}
tmp = manifest_path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
tmp.replace(manifest_path)
PYEOF
}

# --- Recording preflight and manifest: no robot/camera/server process starts here --
mkdir -p "${IFQL_LOG_ROOT}"
if [ -e "${REAL_EVAL_RUN_DIR}" ]; then
    echo "ERROR: real-eval run directory already exists; refusing stale/colliding run: ${REAL_EVAL_RUN_DIR}" >&2
    exit 1
fi
if [ "${REAL_EVAL_RECORDING}" = "1" ]; then
    [ -f "${REAL_EVAL_PREFLIGHT}" ] || { echo "ERROR: missing recording preflight: ${REAL_EVAL_PREFLIGHT}" >&2; exit 1; }
    REAL_EVAL_FFMPEG="$("${IFQL_PY}" "${REAL_EVAL_PREFLIGHT}" \
        --inference-python "${IFQL_PY}" --run-dir "${REAL_EVAL_RUN_DIR}" \
        --hdf5-log-dir "${REAL_EVAL_HDF5_LOG_DIR}" --mp4-path "${REAL_EVAL_MP4_PATH}" \
        --renderer-hook "${REAL_EVAL_RENDERER_HOOK}" --min-free-gib "${REAL_EVAL_MIN_FREE_GIB}" --print-ffmpeg)"
    mkdir "${REAL_EVAL_RUN_DIR}"
    "${IFQL_PY}" "${REAL_EVAL_PREFLIGHT}" \
        --inference-python "${IFQL_PY}" --run-dir "${REAL_EVAL_RUN_DIR}" \
        --hdf5-log-dir "${REAL_EVAL_HDF5_LOG_DIR}" --mp4-path "${REAL_EVAL_MP4_PATH}" \
        --renderer-hook "${REAL_EVAL_RENDERER_HOOK}" --ffmpeg "${REAL_EVAL_FFMPEG}" \
        --min-free-gib "${REAL_EVAL_MIN_FREE_GIB}"
else
    echo "### REAL_EVAL_RECORDING=0: paired HDF5+MP4 is explicitly disabled for this launch." >&2
    mkdir "${REAL_EVAL_RUN_DIR}"
    REAL_EVAL_FFMPEG=""
fi
export REAL_EVAL_RUN_DIR REAL_EVAL_HDF5_LOG_DIR REAL_EVAL_MP4_PATH REAL_EVAL_RENDERER_HOOK
export REAL_EVAL_FFMPEG REAL_EVAL_RECORDING REAL_EVAL_MIN_FREE_GIB REAL_EVAL_POLICY_TYPE
export IFQL_TASK IFQL_SAMPLER IFQL_NUM_SAMPLES IFQL_STEP IFQL_DEVICE IFQL_CHECKPOINT
export IFQL_NORM_STATS IFQL_RUN_DIR IFQL_SERVER_PY
write_launch_manifest launching

if [ "${IFQL_DRY_RUN}" = "1" ]; then
    echo "### IFQL_DRY_RUN=1: every preflight passed; printing the two resolved commands and exiting WITHOUT starting either one."
    echo "### task=${IFQL_TASK} run_dir=${IFQL_RUN_DIR} norm_stats=${IFQL_NORM_STATS} params_file=${IFQL_PARAMS_FILE} px=${IS_PX_RUN} act_port=${IFQL_PORT}"
    echo "### recording=${REAL_EVAL_RECORDING} manifest=${REAL_EVAL_RUN_DIR}/launch_manifest.json hdf5_dir=${REAL_EVAL_HDF5_LOG_DIR} mp4=${REAL_EVAL_MP4_PATH} ffmpeg=${REAL_EVAL_FFMPEG:-<disabled>}"
    echo "### server command:"
    printf '###   env -u FMRL_CAM1_CROP -u FMRL_CAM1_MODE QFLOW_DIR=%q TORCH_HOME=%q XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1 \\\n' \
        "${IFQL_QFLOW_DIR}" "${TORCH_HOME:-$IFQL_ROOT/.cache/torch}"
    printf '###    '
    printf ' %q' "${SERVER_CMD[@]}"
    printf '\n'
    echo "### ros2 launch command:"
    printf '###   ros2 launch gello_policy ur7e_diffusion_real.launch.py'
    printf ' %q' "${ARGS[@]}" "$@"
    printf '\n'
    trap - EXIT   # nothing was ever spawned; do not attempt to kill a nonexistent PID
    exit 0
fi

(
    cd "${IFQL_SERVER_DIR}"
    exec env -u FMRL_CAM1_CROP -u FMRL_CAM1_MODE \
    QFLOW_DIR="${IFQL_QFLOW_DIR}" \
    TORCH_HOME="${TORCH_HOME:-$IFQL_ROOT/.cache/torch}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    HF_HUB_OFFLINE=1 \
    REAL_EVAL_RUN_DIR="${REAL_EVAL_RUN_DIR}" \
    REAL_EVAL_HDF5_LOG_DIR="${REAL_EVAL_HDF5_LOG_DIR}" \
    REAL_EVAL_MP4_PATH="${REAL_EVAL_MP4_PATH}" \
    REAL_EVAL_RENDERER_HOOK="${REAL_EVAL_RENDERER_HOOK}" \
    REAL_EVAL_FFMPEG="${REAL_EVAL_FFMPEG}" \
    "${SERVER_CMD[@]}"
) > >(tee -a "${SERVER_LOG}") 2>&1 &
IFQL_SERVER_PID=$!
ROS_PID=""

wait_for_owned_pid() {
    local pid="$1" timeout="$2" label="$3"
    local elapsed=0
    while kill -0 "${pid}" 2>/dev/null && [ "${elapsed}" -lt "${timeout}" ]; do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    if kill -0 "${pid}" 2>/dev/null; then
        echo "### WARNING: owned ${label} pid ${pid} did not exit within ${timeout}s after SIGINT; sending SIGTERM." >&2
        kill -TERM "${pid}" 2>/dev/null || true
        elapsed=0
        while kill -0 "${pid}" 2>/dev/null && [ "${elapsed}" -lt 5 ]; do
            sleep 1
            elapsed=$((elapsed + 1))
        done
    fi
    wait "${pid}" 2>/dev/null || true
}

cleanup_real_eval() {
    local status="$?"
    trap - EXIT INT TERM
    set +e
    # The ROS launch owns robot/camera processes; ask it to stop first so the server
    # can finish its writers only after no further requests arrive. Only PIDs created
    # by this script are signalled; unrelated processes are never selected or killed.
    if [ -n "${ROS_PID}" ] && kill -0 "${ROS_PID}" 2>/dev/null; then
        echo "### stopping owned ROS launch (pid ${ROS_PID}) before server finalization..." >&2
        kill -INT "${ROS_PID}" 2>/dev/null || true
        wait_for_owned_pid "${ROS_PID}" "${REAL_EVAL_FINALIZE_TIMEOUT_S}" "ROS launch"
    fi
    if [ -n "${IFQL_SERVER_PID:-}" ] && kill -0 "${IFQL_SERVER_PID}" 2>/dev/null; then
        echo "### stopping owned IFQL/SVF server (pid ${IFQL_SERVER_PID}) and waiting for HDF5/MP4 finalization..." >&2
        kill -INT "${IFQL_SERVER_PID}" 2>/dev/null || true
        wait_for_owned_pid "${IFQL_SERVER_PID}" "${REAL_EVAL_FINALIZE_TIMEOUT_S}" "policy server"
    fi
    if [ -d "${REAL_EVAL_RUN_DIR}" ]; then
        write_launch_manifest finalized || true
    fi
    exit "${status}"
}
trap cleanup_real_eval EXIT
trap 'exit 130' INT TERM

# --- Health-check: alive + norm_stats/sampler/px lines + port listening --------------
# The server logs "norm_stats: <path>  (source: ...)" early, then "sampler: kind=.. K=..",
# then "agent ready: ... px=<bool> ..." right before it warms up and binds
# ("REP bound tcp://..."). So:
#   * norm_stats line present but its BASENAME != IFQL_NORM_STATS's basename -> kill + fail;
#   * agent-ready line present but its px=<bool> != our IS_PX_RUN verdict -> kill + fail;
#   * port listening -> warmup already passed -> proceed;
#   * process gone -> fail; IFQL_WARMUP_TIMEOUT_S elapsed -> fail.
IFQL_NORM_STATS_BASE="$(basename "${IFQL_NORM_STATS}")"
echo "### Waiting for IFQL server (pid ${IFQL_SERVER_PID}) to load, warm up and listen on :${IFQL_PORT} (<= ${IFQL_WARMUP_TIMEOUT_S} s)..."
NORM_OK=""
SAMPLER_OK=""
PX_OK=""
LISTENING=""
for i in $(seq 1 "${IFQL_WARMUP_TIMEOUT_S}"); do
    if ! kill -0 "${IFQL_SERVER_PID}" 2>/dev/null; then
        echo "ERROR: IFQL server (pid ${IFQL_SERVER_PID}) exited during startup — see its output above / ${SERVER_LOG}." >&2
        exit 1
    fi
    if [ -z "${NORM_OK}" ] && [ -f "${SERVER_LOG}" ]; then
        # The exact line is "[ifql-server] norm_stats: <path>  (source: ...)". Task/lead/px
        # agnostic exact-basename equality against IFQL_NORM_STATS — this proves the server
        # really resolved the SAME FILE we asked for (--norm-stats is always passed, so a
        # mismatch means the server ignored it or the log format changed).
        NORM_LINE="$(grep -m1 -E 'norm_stats:' "${SERVER_LOG}" || true)"
        if [ -n "${NORM_LINE}" ]; then
            NORM_PATH="$(printf '%s' "${NORM_LINE}" | sed -E 's/.*norm_stats:[[:space:]]*([^[:space:]]+).*/\1/')"
            NORM_BASE="$(basename "${NORM_PATH}")"
            if [ "${NORM_BASE}" = "${IFQL_NORM_STATS_BASE}" ]; then
                NORM_OK=1
                echo "### norm_stats check OK: ${NORM_LINE}"
            else
                echo "ERROR: server resolved norm_stats basename '${NORM_BASE}' != requested '${IFQL_NORM_STATS_BASE}':" >&2
                echo "       ${NORM_LINE}" >&2
                echo "       A different lead/task/px norm_stats denormalises joints into a different box. Stopping the server." >&2
                kill "${IFQL_SERVER_PID}" 2>/dev/null || true
                exit 1
            fi
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
                kill "${IFQL_SERVER_PID}" 2>/dev/null || true
                exit 1
            fi
        fi
    fi
    if [ -z "${PX_OK}" ] && [ -f "${SERVER_LOG}" ]; then
        PX_LINE="$(grep -m1 -E 'agent ready:.*px=' "${SERVER_LOG}" || true)"
        if [ -n "${PX_LINE}" ]; then
            SERVER_PX="$(printf '%s' "${PX_LINE}" | sed -E 's/.*px=([A-Za-z]+).*/\1/' | tr '[:upper:]' '[:lower:]')"
            if [ "${SERVER_PX}" = "${IS_PX_RUN}" ]; then
                PX_OK=1
                echo "### px check OK: ${PX_LINE}"
            else
                echo "ERROR: server px=${SERVER_PX} does not match this run dir's flags.json is_px_run verdict (${IS_PX_RUN}):" >&2
                echo "       ${PX_LINE}" >&2
                kill "${IFQL_SERVER_PID}" 2>/dev/null || true
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
# The port is up only after warmup, so the norm_stats/sampler/px lines must have been
# seen by now. (If the log format ever drops one, fail closed rather than run unverified.)
if [ -z "${NORM_OK}" ]; then
    echo "ERROR: server is listening but no 'norm_stats:' line was seen in ${SERVER_LOG}; cannot verify it matches ${IFQL_NORM_STATS_BASE}. Stopping." >&2
    kill "${IFQL_SERVER_PID}" 2>/dev/null || true
    exit 1
fi
if [ -z "${SAMPLER_OK}" ]; then
    echo "ERROR: server is listening but no 'sampler: kind=' line was seen in ${SERVER_LOG}; cannot verify the sampler. Stopping." >&2
    kill "${IFQL_SERVER_PID}" 2>/dev/null || true
    exit 1
fi
if [ -z "${PX_OK}" ]; then
    echo "ERROR: server is listening but no 'agent ready: ... px=' line was seen in ${SERVER_LOG}; cannot verify px. Stopping." >&2
    kill "${IFQL_SERVER_PID}" 2>/dev/null || true
    exit 1
fi
echo "### IFQL server is listening on ${IFQL_HOST}:${IFQL_PORT} (warmup done)."
if grep -q 'WARNING: warmup refill' "${SERVER_LOG}" 2>/dev/null; then
    echo "### ⚠️  $(grep -m1 'WARNING: warmup refill' "${SERVER_LOG}")" >&2
    echo "### ⚠️  Refill is slow for this box. Consider Ctrl-C and IFQL_NUM_SAMPLES=16 — do NOT widen the yaml timeouts." >&2
fi

# ARGS + HEADLESS_STATE were already built above (before the server spawn, so
# IFQL_DRY_RUN could print them without starting anything) — reused as-is here.
echo "### REAL UR7e IFQL deploy | robot_ip=${ROBOT_IP} | calib=${CALIB:-<none>} | params=${IFQL_PARAMS_FILE}"
echo "### headless_mode=${HEADLESS_STATE}"
echo "### start_mode=gello — arm drives straight to the ${IFQL_TASK} policy's HELD start pose and parks."
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

# Keep the ROS launch as an owned child so cleanup can request its graceful shutdown
# before it asks the policy server to close its HDF5/MP4 writers.
ros2 launch gello_policy ur7e_diffusion_real.launch.py "${ARGS[@]}" "${EXTRA_LAUNCH_ARGS[@]}" &
ROS_PID=$!
set +e
wait "${ROS_PID}"
ROS_STATUS=$?
set -e
ROS_PID=""
exit "${ROS_STATUS}"
