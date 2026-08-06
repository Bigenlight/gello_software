#!/usr/bin/env bash
# =============================================================================
# run_bc_server_local.sh -- serve the BC policy from laptop3's OWN GPU
# =============================================================================
#
# Terminal 1 of a LOCAL BC evaluation:  ./run_bc_server_local.sh
#
# Sibling of run_bc_server.sh with the network removed.  That script starts the
# BC server on junhyeong_ai and holds an SSH tunnel so the actor's fixed dial
# address 127.0.0.1:50153 reaches it; this one runs the SAME UNMODIFIED
# serl_ur_infra/scripts/run_bc_policy_server.py here and binds 127.0.0.1:50153
# directly.  Nothing downstream can tell the difference: T2, T3 and the three
# T3 pins are byte-identical to a remote eval, because the artifact -- and
# therefore model_id -- is the same one.  What changes is that wire_ms becomes
# a loopback round trip instead of a WAN one, which is the whole point.
#
# OWNERSHIP -- read before editing:
#
#   * The only process this script may signal is its OWN direct child, by the
#     PID bash handed us when we spawned it.  No pkill, no killall, no matching
#     by name or pattern, and nothing remote is touched at all.
#   * A busy port is REPORTED, never cleared.  On laptop3 port 50153 is far
#     more likely to be a live run_hil_server.sh / run_bc_server.sh tunnel than
#     a leftover of ours, and closing someone's tunnel out from under a running
#     lineage is not ours to do.
#   * Unlike the remote sibling, a server that failed to become ready IS
#     stopped here.  Remotely the argument was "leave the evidence running";
#     locally the evidence is the log file on this disk, and what a stranded
#     child would still hold is this laptop's GPU memory and the actor's port.
#
# GPU -- laptop3 has ONE RTX 3060 Laptop with 6 GB, shared with the RealSense
# viewers, the HIL GUI and anything else on this desktop.  The server script
# already setdefault's XLA_PYTHON_CLIENT_PREALLOCATE=false (so JAX grows its
# heap instead of claiming the whole card), which is what makes that sharing
# possible -- do not "helpfully" preallocate here.  Numbers measured on an idle
# card are not the numbers a live session sees; record which it was.
#
# Overrides (defaults shown):
#   LOCAL_POLICY_PYTHON=/home/laptop3/venvs/gello-local-policy/bin/python
#   HIL_LOCAL_DATA_ROOT=/home/laptop3/hil-serl-data
#   BC_ARTIFACT_DIR=<data root>/diagnostics/<bc-init artifact>
#   LOCAL_POLICY_PORT=50153 (what the production actor dials; 50053 refused)
#   LOCAL_POLICY_MIN_FREE_G=3 (refuse to start below this on the data root)
#   BC_START_TIMEOUT_S=180
#   HIL_STEP_TIMING= (1/true/yes makes the server also write
#   <run dir>/served/timing.jsonl; inherited by the child as-is, off by default)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOCAL_POLICY_PYTHON="${LOCAL_POLICY_PYTHON:-/home/laptop3/venvs/gello-local-policy/bin/python}"
LOCAL_DATA_ROOT="${HIL_LOCAL_DATA_ROOT:-/home/laptop3/hil-serl-data}"
BC_ARTIFACT_DIR="${BC_ARTIFACT_DIR:-$LOCAL_DATA_ROOT/diagnostics/bc_cube_in_cup_raw_0731_bce_group_holdout_20epoch_20260731_213638.bc-init}"
LOCAL_POLICY_PORT="${LOCAL_POLICY_PORT:-50153}"
BC_START_TIMEOUT_S="${BC_START_TIMEOUT_S:-180}"

# The pins a BC eval's actor terminal must carry.  Printed, never exported: T3
# is a different terminal, and a pin the operator did not type is a pin nobody
# checked.  These are the gate that stops the wrong policy reaching the robot.
BC_EXPECTED_MODEL_ID="bc-cube-in-cup-raw0731-bcinit-v1"
EXPECTED_REWARD_AUTHORITY="local"
EXPECTED_REWARD_MODEL_ID="operator-manual-success-v1"

SERVER_ENTRYPOINT="$REPO_ROOT/serl_ur_infra/scripts/run_bc_policy_server.py"
READY_MARK='[bc-server] ready '
FATAL_MARK='[bc-server] FATAL'
STOPPED_MARK='[bc-server] stopped '

RUN_DIR="$LOCAL_DATA_ROOT/bc_eval/bc_eval_local_$(date -u +%Y%m%d_%H%M%S)"
LOGFILE="$RUN_DIR/server.log"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# --- argument sanity ---------------------------------------------------------
[[ "$LOCAL_POLICY_PORT" =~ ^[0-9]+$ ]] || die "LOCAL_POLICY_PORT must be numeric (got: $LOCAL_POLICY_PORT)"
# Base-10 normalization, and it must happen BEFORE the first (( )) below and
# before the value reaches argv.  Bash arithmetic reads a leading zero as OCTAL:
# `050053` evaluates to 20523 here -- sailing straight past the `!= 50053`
# refusal three lines down -- while argparse in the child reads the same string
# as decimal 50053 and binds the production learner's port.  Measured.
LOCAL_POLICY_PORT=$(( 10#$LOCAL_POLICY_PORT ))
(( LOCAL_POLICY_PORT >= 1024 && LOCAL_POLICY_PORT <= 65535 )) || die "LOCAL_POLICY_PORT must be in [1024, 65535]"
# 50053 is the production learner's number on both hosts: on laptop3 it is the
# local end of run_hil_server.sh's tunnel.  Binding it would put a frozen BC
# policy behind the address an online-RL session dials.
(( LOCAL_POLICY_PORT != 50053 )) || \
    die "port 50053 belongs to the production learner (and its laptop3 tunnel); run_bc_server_local.sh will not use it"
if (( LOCAL_POLICY_PORT != 50153 )); then
    echo "INFO: LOCAL_POLICY_PORT=$LOCAL_POLICY_PORT, but the production actor always dials 50153; it will not find this server." >&2
fi
[[ "$BC_START_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || die "BC_START_TIMEOUT_S must be a positive integer"
(( BC_START_TIMEOUT_S <= 1800 )) || die "BC_START_TIMEOUT_S must be <= 1800"
for _name in LOCAL_POLICY_PYTHON LOCAL_DATA_ROOT BC_ARTIFACT_DIR; do
    _value="${!_name}"
    [[ "$_value" == /* ]] || die "$_name must be an absolute path (got: $_value)"
    [[ "$_value" != *".."* ]] || die "$_name must not contain '..'"
    [[ "$_value" != *[[:space:]]* ]] || die "$_name must not contain whitespace"
done

# Opt-in per-step server timing.  The child inherits HIL_STEP_TIMING from this
# environment untouched -- we only READ it, so that what the server decides and
# what this banner claims can never disagree.  The vocabulary is
# ur_env/step_timing.py's: 1/true/yes, case-insensitive, everything else off.
step_timing_on=0
case "$(printf '%s' "${HIL_STEP_TIMING:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')" in
    "") ;;
    1|true|yes) step_timing_on=1 ;;
    *)
        echo "WARNING: HIL_STEP_TIMING='${HIL_STEP_TIMING}' is not 1/true/yes; step timing stays OFF." >&2
        ;;
esac

# --- [1/4] the interpreter ---------------------------------------------------
# jax must be a CUDA build for this to be a LOCAL GPU measurement at all; that
# is the new venv's job, not this script's.  What we can prove cheaply is that
# the interpreter exists, before spending a run directory on it.
if [[ ! -x "$LOCAL_POLICY_PYTHON" ]]; then
    {
        echo "ERROR: local policy interpreter is missing or not executable:"
        echo "         $LOCAL_POLICY_PYTHON"
        echo "  Build it once:   $SCRIPT_DIR/setup_local_policy_venv.sh"
        echo "  CPU fallback (no GPU numbers, but it serves):"
        # Absolute, not "$0": the hint is copy-pasted, and a relative $0 is only
        # valid from the directory this was launched from.
        echo "         LOCAL_POLICY_PYTHON=/home/laptop3/venvs/hilserl/bin/python $SCRIPT_DIR/$(basename "$0")"
        echo "  Nothing was started."
    } >&2
    exit 1
fi
[[ -f "$SERVER_ENTRYPOINT" ]] || die "BC entrypoint is missing: $SERVER_ENTRYPOINT"
# No PYTHONPATH export, unlike the remote sibling: run_bc_policy_server.py
# inserts serl_ur_infra and third_party/hil-serl/serl_launcher into sys.path
# itself, relative to its own file, so the paths are right by construction.

# Device provenance.  This launcher's entire reason to exist is the phrase
# "LOCAL GPU" in its banner, and until now it printed that phrase without ever
# asking jax what it actually got: point LOCAL_POLICY_PYTHON at the documented
# CPU fallback above and every number would still have been filed as a GPU
# number.  So we ask, once, before a run directory exists.  A cpu answer WARNS
# and CONTINUES -- the fallback is supposed to keep working -- but from here on
# the banner says what the probe said, never more.  Costs one jax import.
LOCAL_POLICY_DEVICE="$(XLA_PYTHON_CLIENT_PREALLOCATE=false "$LOCAL_POLICY_PYTHON" -c \
    'import jax; d = jax.devices(); print(d[0].platform if d else "none")' 2>/dev/null | tail -n1 || true)"
LOCAL_POLICY_DEVICE="$(printf '%s' "${LOCAL_POLICY_DEVICE:-}" | tr -dc 'A-Za-z0-9_-')"
[[ -n "$LOCAL_POLICY_DEVICE" ]] || LOCAL_POLICY_DEVICE="unknown"
case "$LOCAL_POLICY_DEVICE" in
    gpu|cuda|rocm) ;;
    cpu)
        {
            echo "WARNING: jax in $LOCAL_POLICY_PYTHON reports device platform 'cpu'."
            echo "         The server will run and serve correctly, but every latency"
            echo "         number this eval produces is a CPU number.  The ready banner"
            echo "         below will say so -- do not file these as GPU numbers."
            echo "         Build the CUDA venv with: $SCRIPT_DIR/setup_local_policy_venv.sh"
        } >&2
        ;;
    *)
        {
            echo "WARNING: could not read a jax device platform from"
            echo "         $LOCAL_POLICY_PYTHON (probe answered '$LOCAL_POLICY_DEVICE')."
            echo "         Starting anyway -- a real import failure will surface as a FATAL"
            echo "         from the server itself in a moment -- but the banner will NOT"
            echo "         claim GPU for a device nobody proved."
        } >&2
        ;;
esac

# --- [2/4] the artifact ------------------------------------------------------
# manifest.json specifically, not just the directory: a half-finished scp
# leaves a directory that exists and a loader that fails a minute later, after
# the artifact has already been half-read.
if [[ ! -d "$BC_ARTIFACT_DIR" ]]; then
    {
        echo "ERROR: bc-init artifact directory is missing:"
        echo "         $BC_ARTIFACT_DIR"
        echo "  Fetch it from the GPU server:   $SCRIPT_DIR/fetch_policy_artifacts.sh"
        echo "  Nothing was started."
    } >&2
    exit 1
fi
[[ -f "$BC_ARTIFACT_DIR/manifest.json" ]] || \
    die "no manifest.json in $BC_ARTIFACT_DIR -- the artifact looks incomplete; re-run $SCRIPT_DIR/fetch_policy_artifacts.sh"

# --- [3/4] the actor's port --------------------------------------------------
# Report and refuse.  The overwhelmingly likely owner is an SSH tunnel from a
# remote-server terminal, and that terminal's Ctrl-C is the correct fix -- it
# closes only the tunnel, leaving the learner on the GPU server untouched.
port_listener_pids() {
    # Listener PIDs we are allowed to see.  Ours always is (same user); a
    # listener owned by another user prints no users:(...) field, which is why
    # an EMPTY answer below is treated as "unproven", never as "fine".
    if command -v ss >/dev/null 2>&1; then
        ss -Htlnp "sport = :$LOCAL_POLICY_PORT" 2>/dev/null \
            | grep -o 'pid=[0-9]\+' | cut -d= -f2 | sort -u || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$LOCAL_POLICY_PORT" -sTCP:LISTEN -t 2>/dev/null | sort -u || true
    else
        printf 'NOTOOL\n'
    fi
}
port_is_bound() {
    if command -v ss >/dev/null 2>&1; then
        ss -tln 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${LOCAL_POLICY_PORT}\$"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$LOCAL_POLICY_PORT" -sTCP:LISTEN -t >/dev/null 2>&1
    else  # fallback: a successful connect proves a listener
        (exec 3<>"/dev/tcp/127.0.0.1/$LOCAL_POLICY_PORT") 2>/dev/null
    fi
}
if port_is_bound; then
    {
        echo "=================================================================="
        echo "  !!  LOCAL PORT $LOCAL_POLICY_PORT IS ALREADY BOUND  !!"
        echo "  Current listener(s):"
        ss -tlnp "sport = :$LOCAL_POLICY_PORT" 2>/dev/null || \
            lsof -nP -iTCP:"$LOCAL_POLICY_PORT" -sTCP:LISTEN 2>/dev/null || \
            echo "    (could not read the socket table)"
        # WHO it is decides what the operator must do, and the two answers are
        # opposite instructions.  The old text always said "close the remote
        # server terminal", which is exactly wrong in the one case this
        # launcher can itself create: its own orphaned child, for which no
        # terminal exists any more and whose only fix is `kill -INT`.
        _busy_pids="$(port_listener_pids)"
        _busy_classified=0
        if [[ -n "$_busy_pids" && "$_busy_pids" != "NOTOOL" ]]; then
            while read -r _busy_pid; do
                [[ -n "$_busy_pid" ]] || continue
                _busy_cmd="$(tr '\0' ' ' <"/proc/$_busy_pid/cmdline" 2>/dev/null || true)"
                _busy_classified=1
                case "$_busy_cmd" in
                    *run_bc_policy_server.py*|*run_fm_policy_server.py*)
                        echo "  PID $_busy_pid is an ORPHANED LOCAL eval server:"
                        echo "      ${_busy_cmd:-<unreadable>}"
                        echo "    No terminal owns it -- a previous local launcher died without"
                        echo "    taking it down.  Stop it with:"
                        echo "      kill -INT $_busy_pid"
                        echo "    then re-run this script."
                        ;;
                    *ssh*)
                        echo "  PID $_busy_pid is an SSH tunnel:"
                        echo "      ${_busy_cmd:-<unreadable>}"
                        echo "    Ctrl-C the REMOTE server terminal that holds it"
                        echo "    (./run_hil_server.sh, ./run_bc_server.sh or ./run_fm_server.sh)."
                        echo "    That closes only the tunnel (and, for the eval launchers, the"
                        echo "    eval server they started); the production learner on the GPU"
                        echo "    host keeps running with its buffer intact."
                        ;;
                    *)
                        echo "  PID $_busy_pid is neither a local eval server nor an ssh tunnel:"
                        echo "      ${_busy_cmd:-<unreadable>}"
                        echo "    Identify its owner before doing anything.  This script does not"
                        echo "    signal processes it did not start."
                        ;;
                esac
            done <<<"$_busy_pids"
        fi
        if (( _busy_classified == 0 )); then
            echo "  No owning PID is visible (a listener owned by another user shows"
            echo "  none, and neither ss nor lsof may be installed).  It is either an"
            echo "  SSH tunnel from ./run_hil_server.sh / ./run_bc_server.sh /"
            echo "  ./run_fm_server.sh -- Ctrl-C that terminal -- or an orphaned local"
            echo "  eval server.  Identify it before doing anything."
        fi
        echo "  Nothing was started here and nothing was stopped."
        echo "=================================================================="
    } >&2
    exit 1
fi

# --- [4/4] start the BC server ----------------------------------------------
# Disk gate first.  / on this laptop sits near 96% full, and the failure this
# prevents is the expensive kind: the server records every served step under
# $RUN_DIR/served, so running out of space happens LATE -- after the robot has
# been moved and the episodes have been flown -- and it destroys exactly the
# records the eval existed to produce.  Refusing now costs nothing.
LOCAL_POLICY_MIN_FREE_G="${LOCAL_POLICY_MIN_FREE_G:-3}"
[[ "$LOCAL_POLICY_MIN_FREE_G" =~ ^[0-9]+$ ]] || \
    die "LOCAL_POLICY_MIN_FREE_G must be a non-negative integer (got: $LOCAL_POLICY_MIN_FREE_G)"
avail_g="$(df -BG --output=avail "$LOCAL_DATA_ROOT" 2>/dev/null | tail -n1 | tr -dc '0-9' || true)"
[[ -n "$avail_g" ]] || \
    die "could not read free space on $LOCAL_DATA_ROOT -- refusing to start a run that may not be able to write its records"
if (( avail_g < LOCAL_POLICY_MIN_FREE_G )); then
    die "only ${avail_g}G free on $LOCAL_DATA_ROOT -- need >= ${LOCAL_POLICY_MIN_FREE_G}G for a BC eval's served records.
  Free space by removing old local eval run dirs (they are pure output, nothing else reads them):
      du -sh $LOCAL_DATA_ROOT/bc_eval/bc_eval_local_* | sort -h
      rm -rf $LOCAL_DATA_ROOT/bc_eval/bc_eval_local_<timestamp>
  Or raise the gate deliberately: LOCAL_POLICY_MIN_FREE_G=<n> $SCRIPT_DIR/$(basename "$0")"
fi

[[ ! -e "$RUN_DIR" ]] || die "run dir already exists: $RUN_DIR (two launches in the same second?)"
mkdir -p "$RUN_DIR"
: >"$LOGFILE"
# Ownership context, written into the log BEFORE the server's first byte.  A
# banner scrolls away; the log is what someone reads three weeks later when the
# question is "was that a GPU number?".  The server APPENDS from here on (see
# the >> below), so both writers use O_APPEND and neither can clobber the other.
{
    printf '[bc-local] launcher pid=%s python=%s device=%s\n' \
        "$$" "$LOCAL_POLICY_PYTHON" "$LOCAL_POLICY_DEVICE"
    printf '[bc-local] artifact=%s port=%s run_dir=%s\n' \
        "$BC_ARTIFACT_DIR" "$LOCAL_POLICY_PORT" "$RUN_DIR"
} >>"$LOGFILE"

echo "Starting BC policy server on laptop3 (port $LOCAL_POLICY_PORT)"
echo "  python  : $LOCAL_POLICY_PYTHON"
echo "  device  : $LOCAL_POLICY_DEVICE (jax.devices()[0].platform, probed above)"
echo "  artifact: $BC_ARTIFACT_DIR"
echo "  run dir : $RUN_DIR"
if (( step_timing_on )); then
    echo "[bc-local] step timing ON -> $RUN_DIR/served/timing.jsonl"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    # A snapshot, not a gate: 6 GB shared with viewers/GUI is exactly the
    # condition under which "it was slow" needs a before-picture.
    echo "  gpu     : $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo '<unreadable>')"
fi

# Redirected straight to the log and mirrored with `tail -F`, rather than piped
# through `tee`, for one specific reason: in `python ... | tee log &`, `$!` is
# TEE's pid.  The port-owner assertion below and the SIGINT forwarding both
# need PYTHON's.  The direct redirection also removes tee's buffering from the
# path the ready-poll greps -- the server prints with flush=True, so every line
# is on disk the moment it is written.
"$LOCAL_POLICY_PYTHON" "$SERVER_ENTRYPOINT" \
    --host 127.0.0.1 \
    --port "$LOCAL_POLICY_PORT" \
    --artifact-dir "$BC_ARTIFACT_DIR" \
    --record-root "$RUN_DIR/served" \
    >>"$LOGFILE" 2>&1 </dev/null &
SERVER_PID=$!
tail -n +1 -F "$LOGFILE" 2>/dev/null &
MIRROR_PID=$!
# Flipped by the ready poll below; stop_server reads it to decide how long the
# child is allowed to take over a graceful shutdown.
READY_SEEN=0

# The traps go up NOW, not after ready like the remote sibling does: from this
# line on there is a child holding GPU memory, and every exit path below --
# including a `set -e` abort -- must take it down with us.
SERVER_STOPPED=0
# `kill -0` alone is NOT "still running": until this shell reaps it, an exited
# child stays a zombie, and kill -0 succeeds on a zombie.  Measured with a stub
# server that died instantly -- without the state check the launcher spent its
# whole 20 s grace signalling a corpse and then escalated INT -> TERM -> KILL
# against it, and a crash before ready was reported as a startup TIMEOUT
# instead of as a crash.
server_running() {
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    ! grep -qs '^State:[[:space:]]*Z' "/proc/$SERVER_PID/status"
}
stop_mirror() {
    if [[ -n "${MIRROR_PID:-}" ]]; then
        kill "$MIRROR_PID" 2>/dev/null || true
        wait "$MIRROR_PID" 2>/dev/null || true
        MIRROR_PID=""
    fi
}
stop_server() {
    (( SERVER_STOPPED == 0 )) || return 0
    SERVER_STOPPED=1
    if ! server_running; then
        wait "$SERVER_PID" 2>/dev/null || true
        return 0
    fi
    # SIGINT, not SIGTERM: run_bc_policy_server.py installs a handler for both,
    # but SIGINT is the signal its own shutdown path is written around -- it
    # drains the gRPC server with a 5 s grace and prints its
    # "[bc-server] stopped replay_count=... " summary, which is the record of
    # what this eval actually captured.
    #
    # The grace depends on whether we ever saw ready, and that is not
    # cosmetic.  POSIX has a non-interactive shell start background children
    # with SIGINT set to IGNORE, and CPython keeps an inherited SIG_IGN instead
    # of installing its own handler -- so until _serve() calls signal.signal()
    # (after the jax import and the artifact load: tens of seconds) this child
    # is DEAF to SIGINT.  Measured: a stub that died at startup made the
    # launcher signal for the full 20 s before escalating.  After ready the
    # handler is provably installed, because ready is printed after it.
    echo "Stopping local BC server PID $SERVER_PID (INT)..."
    kill -INT "$SERVER_PID" 2>/dev/null || true
    local waited=0 grace=40
    (( READY_SEEN == 1 )) || grace=6
    while server_running && (( waited < grace )); do
        sleep 0.5
        waited=$(( waited + 1 ))
    done
    # Only now does escalation start.  A KILL here would lose the stopped
    # summary and could truncate the last recorded transition; TERM is handled
    # by the same handler as INT and is never inherited-ignored.
    if server_running; then
        echo "BC server PID $SERVER_PID did not exit on INT after $(( grace / 2 ))s; sending TERM." >&2
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        waited=0
        while server_running && (( waited < 20 )); do
            sleep 0.5
            waited=$(( waited + 1 ))
        done
    fi
    if server_running; then
        echo "BC server PID $SERVER_PID ignored TERM; sending KILL." >&2
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
}
cleanup() {
    # FIRST statement, and it is the whole fix: from here on this shell is deaf
    # to INT/TERM/HUP.  Without it a SECOND Ctrl-C -- the reflex when the first
    # one does not seem to have done anything -- lands while stop_server is
    # inside its INT -> TERM -> KILL ladder, runs the `exit 130` trap from
    # within this EXIT trap, and leaves the shell immediately.  Reproduced with
    # a stub server: the server AND the `tail -F` mirror both survived forever,
    # holding GPU memory and the actor's port.  SERVER_STOPPED was already 1 by
    # then, so nothing would ever have retried.  The children inherit the
    # ignore, so the ladder's `sleep`s survive the second Ctrl-C too.
    trap '' INT TERM HUP
    # The mirror is in this terminal's process group, so a terminal Ctrl-C
    # kills it outright and the server's own last line never reaches the
    # screen; a directed `kill -TERM <launcher pid>` leaves it running and it
    # mirrors that line itself.  Decide WHICH it is before stopping anything,
    # or the operator gets the stopped summary twice.
    local mirror_alive=0 stopped_line
    if [[ -n "${MIRROR_PID:-}" ]] && kill -0 "$MIRROR_PID" 2>/dev/null; then
        mirror_alive=1
    fi
    stop_server
    if (( mirror_alive )); then
        sleep 0.5  # let the mirror flush the server's last line
        stop_mirror
    else
        stop_mirror
        stopped_line="$(grep -F "$STOPPED_MARK" "$LOGFILE" 2>/dev/null | tail -n1 || true)"
        [[ -z "$stopped_line" ]] || echo "$stopped_line"
    fi
    echo "LOCAL_SERVER_RESULT=stopped run_dir=$RUN_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

fail_with_log() {
    # Same rationale as cleanup(): a second Ctrl-C must not abort the ladder.
    trap '' INT TERM HUP
    stop_mirror
    echo "ERROR: $*" >&2
    stop_server
    echo "--- last 40 lines of $LOGFILE ---" >&2
    # `>&2` BEFORE `2>/dev/null`, and the order is the whole point: written the
    # other way round, stdout is pointed at the already-nulled fd 2 and the log
    # tail -- the only thing this failure path exists to show -- vanishes.
    tail -n 40 "$LOGFILE" >&2 2>/dev/null || true
    echo "----------------------------------" >&2
    echo "  The log above stays on disk: $LOGFILE" >&2
    exit 1
}

# Poll the log rather than the port: a bound port only proves gRPC came up,
# while the ready line is printed after the artifact is loaded and validated.
READY_LINE=""
DEADLINE=$(( $(date +%s) + BC_START_TIMEOUT_S ))
while :; do
    HIT="$(grep -F -m1 -e "$FATAL_MARK" -e "$READY_MARK" "$LOGFILE" 2>/dev/null || true)"
    if [[ "$HIT" == *"$FATAL_MARK"* ]]; then
        fail_with_log "BC server reported $HIT"
    fi
    if [[ -n "$HIT" ]]; then
        READY_LINE="$HIT"
        READY_SEEN=1
        break
    fi
    server_running || \
        fail_with_log "BC server process exited before reporting ready"
    (( $(date +%s) < DEADLINE )) || \
        fail_with_log "BC server did not report ready within ${BC_START_TIMEOUT_S}s"
    sleep 2
done

# --- port-owner assertion ----------------------------------------------------
# Load-bearing, not paperwork.  If a remote tunnel were still holding 50153,
# the three T3 pins would all PASS -- the tunnel leads to a server serving the
# SAME artifact, hence the same model_id -- and this "local GPU" run would
# quietly measure the WAN.  A latency number is only local if the process
# answering on the port is the child we just started.
pid_is_ours() {
    local pid="$1" hops=0
    while [[ "$pid" =~ ^[1-9][0-9]*$ ]]; do
        if [[ "$pid" == "$SERVER_PID" ]]; then
            return 0
        fi
        hops=$(( hops + 1 ))
        if (( hops > 12 )); then
            return 1
        fi
        pid="$(awk '/^PPid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
        if [[ -z "$pid" || "$pid" == "0" || "$pid" == "1" ]]; then
            return 1
        fi
    done
    return 1
}
LISTENER_PIDS="$(port_listener_pids)"
if [[ "$LISTENER_PIDS" == "NOTOOL" ]]; then
    fail_with_log "neither ss nor lsof is installed, so the listener on port $LOCAL_POLICY_PORT cannot be proven to be ours (install iproute2 or lsof; refusing to report a possibly-remote server as local)"
fi
if [[ -z "$LISTENER_PIDS" ]]; then
    fail_with_log "port $LOCAL_POLICY_PORT reports ready but no owning PID is visible (a listener owned by another user shows none) -- refusing to call this a local measurement"
fi
while read -r _pid; do
    [[ -n "$_pid" ]] || continue
    pid_is_ours "$_pid" && continue
    _cmd="$(tr '\0' ' ' <"/proc/$_pid/cmdline" 2>/dev/null || true)"
    fail_with_log "port $LOCAL_POLICY_PORT is held by PID $_pid (${_cmd:-<unreadable>}), which is NOT the BC server we started (PID $SERVER_PID). If that is an ssh tunnel, this run would have measured the REMOTE server while every pin still passed. Close the remote server terminal and start again."
done <<<"$LISTENER_PIDS"

# The headline reports the device the probe in [1/4] actually found.  It used to
# say "LOCAL GPU" unconditionally, which was a claim about hardware that nothing
# in this script had checked.
case "$LOCAL_POLICY_DEVICE" in
    gpu|cuda|rocm) DEVICE_BANNER="LOCAL GPU, no tunnel (device=$LOCAL_POLICY_DEVICE)" ;;
    cpu)           DEVICE_BANNER="LOCAL CPU FALLBACK, no tunnel (device=cpu -- NOT GPU numbers)" ;;
    *)             DEVICE_BANNER="LOCAL, no tunnel (device=$LOCAL_POLICY_DEVICE -- UNVERIFIED, do not call it GPU)" ;;
esac
echo "============================================================"
echo "BC policy server ready -- $DEVICE_BANNER"
echo "  serving : grpc://127.0.0.1:$LOCAL_POLICY_PORT  (PID $SERVER_PID, port ownership verified)"
echo "  python  : $LOCAL_POLICY_PYTHON"
echo "  device  : $LOCAL_POLICY_DEVICE (jax.devices()[0].platform)"
echo "  artifact: $BC_ARTIFACT_DIR"
echo "  run dir : $RUN_DIR"
echo "  records : $RUN_DIR/served"
echo "  log     : $LOGFILE"
if (( step_timing_on )); then
    echo "  timing  : $RUN_DIR/served/timing.jsonl"
fi
echo "------------------------------------------------------------"
echo "T3 must carry these three pins, unchanged (they are the gate that"
echo "keeps the wrong policy off the robot):"
echo "  EXPECTED_MODEL_ID=$BC_EXPECTED_MODEL_ID"
echo "  EXPECTED_REWARD_AUTHORITY=$EXPECTED_REWARD_AUTHORITY"
echo "  EXPECTED_REWARD_MODEL_ID=$EXPECTED_REWARD_MODEL_ID"
echo "Dry handshake, no robot involved:"
echo "  EXPECTED_MODEL_ID=$BC_EXPECTED_MODEL_ID EXPECTED_REWARD_AUTHORITY=$EXPECTED_REWARD_AUTHORITY EXPECTED_REWARD_MODEL_ID=$EXPECTED_REWARD_MODEL_ID $SCRIPT_DIR/run_hil_actor.sh --fake-env --no-classifier-sidecar"
echo "============================================================"
echo "LOCAL_SERVER_RUN_DIR=$RUN_DIR"
echo "LOCAL_SERVER_READY_LINE=$READY_LINE"
echo "LOCAL_SERVER_RESULT=started"
echo "Keep this terminal open.  Ctrl-C stops the BC server started above --"
echo "and only that process; nothing remote is involved in this run."

# Backgrounded + `wait`, never exec'd: an exec would replace this shell and no
# trap could run, which is how a local child ends up orphaned on the GPU.  A
# terminal Ctrl-C reaches the child directly too (same process group), so the
# server usually starts its own graceful shutdown before stop_server's INT
# arrives -- the second signal is harmless, and it is the only path a directed
# `kill -TERM <launcher pid>` has.
set +e
wait "$SERVER_PID"
SERVER_RC=$?
set -e
echo "BC policy server exited with status $SERVER_RC." >&2
exit "$SERVER_RC"
