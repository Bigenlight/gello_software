#!/usr/bin/env bash
# =============================================================================
# run_fm_server.sh -- start the remote FM policy server and hold its tunnel
# =============================================================================
#
# Terminal 1 of a flow-matching (FM) real-robot evaluation:  ./run_fm_server.sh
#
# Starts serl_ur_infra/scripts/run_fm_policy_server.py on the GPU host and holds
# laptop3 127.0.0.1:50153 -> server 127.0.0.1:50055, so the UNMODIFIED production
# actor (which always dials 50153) reaches the FM server.  No production script,
# actor or hardware terminal changes.
#
# OWNERSHIP -- read before editing:
#
#   * The production HIL-SERL learner runs on the server at port 50053 and KEEPS
#     RUNNING throughout an FM eval.  It is not ours.  This script names 50053
#     only to refuse it, never runs pkill/killall, never matches a process by
#     name or pattern, and never opens the production launcher lock
#     ($FM_REMOTE_DATA_ROOT/.run_hil_server.lock).  Structurally, the only
#     process it can signal is one whose PID it read from its OWN per-run
#     pidfile and whose /proc/<pid>/cmdline still contains
#     run_fm_policy_server.py.
#   * Port 50054 belongs to run_bc_server.sh.  Sharing it would mean an actor
#     dialling "the eval server" and reaching whichever policy family bound
#     first -- a silent cross-experiment mix-up.  Refused unless
#     FM_ALLOW_PORT_50054=1 states that the BC eval is over.
#   * The FM server is detached on the server host; the cleanup trap stops it
#     because an FM eval is a bounded experiment, unlike a learner lineage.  A
#     server that never became READY is instead LEFT RUNNING with its log
#     intact (as run_hil_server.sh preserves a failed lineage) -- the
#     stale-port check plus the printed stop command dispose of it afterwards.
#
# Overrides (defaults shown):  HIL_SSH_HOST=junhyeong_ai ·
# HIL_REMOTE_REPO=/home/junhyeong/gello_software_runtime ·
# HIL_REMOTE_PYTHON=/home/junhyeong/miniconda3/envs/il/bin/python ·
# HIL_REMOTE_DATA_ROOT=/home/junhyeong/hil-serl-data · HIL_GPU_INDEX=0 ·
# FM_LOCAL_PORT=50153 (what the actor dials) · FM_REMOTE_PORT=50055 (50053 is
# refused outright, 50054 unless FM_ALLOW_PORT_50054=1) ·
# FM_ARTIFACT_DIR=<data root>/diagnostics/<fm-init artifact> · FM_WHICH=best
# (which trained parameter set to serve) · FM_START_TIMEOUT_S=300 ·
# FM_ACCEPT_HEAD_MISMATCH=0 (1 downgrades the HEAD refusal to a warning) ·
# HIL_STEP_TIMING= (1/true/yes makes the server also write
# <run dir>/served/timing.jsonl; off by default).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FM_SSH_HOST="${HIL_SSH_HOST:-junhyeong_ai}"
FM_REMOTE_REPO="${HIL_REMOTE_REPO:-/home/junhyeong/gello_software_runtime}"
FM_REMOTE_PYTHON="${HIL_REMOTE_PYTHON:-/home/junhyeong/miniconda3/envs/il/bin/python}"
FM_REMOTE_DATA_ROOT="${HIL_REMOTE_DATA_ROOT:-/home/junhyeong/hil-serl-data}"
FM_GPU_INDEX="${HIL_GPU_INDEX:-0}"
FM_LOCAL_PORT="${FM_LOCAL_PORT:-50153}"
FM_REMOTE_PORT="${FM_REMOTE_PORT:-50055}"
FM_ARTIFACT_DIR="${FM_ARTIFACT_DIR:-$FM_REMOTE_DATA_ROOT/diagnostics/jax_fm_cube_in_cup_raw_0731_h16_euler8_200epoch_20260731_215835.fm-init}"
FM_WHICH="${FM_WHICH:-best}"
FM_START_TIMEOUT_S="${FM_START_TIMEOUT_S:-300}"
FM_ACCEPT_HEAD_MISMATCH="${FM_ACCEPT_HEAD_MISMATCH:-0}"
FM_ALLOW_PORT_50054="${FM_ALLOW_PORT_50054:-0}"

# Opt-in per-step server timing, off unless asked for.  Normalised HERE rather
# than on the server so a typo is reported to the operator instead of arriving
# as a silent OFF in a run they believe is instrumented.
step_timing="${HIL_STEP_TIMING:-}"
case "${step_timing,,}" in
    "") step_timing="" ;;
    1|true|yes) step_timing="1" ;;
    *)
        echo "WARNING: HIL_STEP_TIMING='$step_timing' is not 1/true/yes; step timing stays OFF." >&2
        step_timing=""
        ;;
esac

# The ready/FATAL markers live ONLY inside the remote poll script below, and
# that is not tidiness: ssh flattens its command argv into a single string that
# the remote login shell re-splits, so an argument containing spaces (both
# markers do) would arrive as several arguments and silently never match.  The
# same rule is why every value marshalled to `bash -s --` below is validated
# whitespace-free.
#
# One directory per launch: the pidfile that authorises the only kill this
# script can perform must never be shared with an earlier FM server.
RUN_DIR="$FM_REMOTE_DATA_ROOT/fm_eval/fm_eval_$(date -u +%Y%m%d_%H%M%S)"
PIDFILE="$RUN_DIR/.fm_server.pid"
LOGFILE="$RUN_DIR/server.log"
SSH_OPTIONS=(
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=60
    -o ServerAliveCountMax=3
    -o TCPKeepAlive=yes
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

rsh() { ssh "${SSH_OPTIONS[@]}" "$FM_SSH_HOST" "$@"; }

print_stop_hint() {
    local run_dir="$1"
    echo "  To inspect and stop THAT FM server (and nothing else):" >&2
    echo "    ssh $FM_SSH_HOST \"pid=\\\$(cat $run_dir/.fm_server.pid); tr '\\\\0' ' ' < /proc/\\\$pid/cmdline; echo\"" >&2
    echo "    # confirm the line above contains run_fm_policy_server.py, then:" >&2
    echo "    ssh $FM_SSH_HOST \"kill \\\$(cat $run_dir/.fm_server.pid)\"" >&2
}

# --- argument sanity ---------------------------------------------------------
for _name in FM_LOCAL_PORT FM_REMOTE_PORT; do
    _value="${!_name}"
    [[ "$_value" =~ ^[0-9]+$ ]] || die "$_name must be numeric (got: $_value)"
    (( _value >= 1024 && _value <= 65535 )) || die "$_name must be in [1024, 65535]"
done
# The production learner owns 50053 on the server host.  Refusing the number
# here -- rather than trusting every downstream guard -- is what makes it
# impossible for this launcher to start on, probe, or stop the learner.
(( FM_REMOTE_PORT != 50053 )) || \
    die "remote port 50053 belongs to the production learner; run_fm_server.sh will not use it"
# 50054 is the BC eval's, not the learner's: legitimate once that eval is over,
# wrong while it runs -- so an opt-in rather than the flat refusal above.
(( FM_REMOTE_PORT != 50054 )) || [[ "$FM_ALLOW_PORT_50054" == "1" ]] || \
    die "remote port 50054 belongs to run_bc_server.sh; stop that BC server first, or set FM_ALLOW_PORT_50054=1 to reuse the port deliberately"
if (( FM_LOCAL_PORT != 50153 )); then
    echo "INFO: FM_LOCAL_PORT=$FM_LOCAL_PORT, but the production actor always dials 50153; it will not find this tunnel." >&2
fi
[[ "$FM_GPU_INDEX" =~ ^[0-9]+$ ]] || die "HIL_GPU_INDEX must be a non-negative integer"
[[ "$FM_START_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || die "FM_START_TIMEOUT_S must be a positive integer"
(( FM_START_TIMEOUT_S <= 1800 )) || die "FM_START_TIMEOUT_S must be <= 1800"
[[ -n "$FM_SSH_HOST" ]] || die "HIL_SSH_HOST cannot be empty"
# Marshalled through `bash -s --` like the paths, so it must be one bare token.
# The vocabulary (best/final/...) is the server's to police, not ours.
[[ "$FM_WHICH" =~ ^[A-Za-z0-9_-]+$ ]] || die "FM_WHICH must be a single bare word (got: $FM_WHICH)"
# These become remote paths.  Keep them boring: a rejected override is far
# cheaper than a remote path that expands into something plausible but wrong.
for _name in FM_REMOTE_DATA_ROOT FM_REMOTE_REPO FM_ARTIFACT_DIR; do
    _value="${!_name}"
    [[ "$_value" == /* ]] || die "$_name must be an absolute path (got: $_value)"
    [[ "$_value" != *".."* ]] || die "$_name must not contain '..'"
    [[ "$_value" != *[[:space:]]* ]] || die "$_name must not contain whitespace"
done
command -v ssh >/dev/null 2>&1 || die "ssh is not installed"

# --- [1/5] local actor endpoint ---------------------------------------------
# The production run_hil_server.sh tunnel binds the SAME local port to the
# learner, and so does run_bc_server.sh during a BC eval.  None of them can
# share it, and we must never guess which one owns a bound port, so refuse and
# tell the operator what to close.
local_port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -tln 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${FM_LOCAL_PORT}\$"
    else  # fallback: a successful connect proves a listener
        (exec 3<>"/dev/tcp/127.0.0.1/$FM_LOCAL_PORT") 2>/dev/null
    fi
}
if local_port_in_use; then
    {
        echo "=================================================================="
        echo "  !!  LOCAL PORT $FM_LOCAL_PORT IS ALREADY BOUND  !!"
        echo "  Almost certainly the production tunnel terminal running"
        echo "  ./run_hil_server.sh (or a BC eval's ./run_bc_server.sh)."
        echo "  Close THAT terminal with Ctrl-C first: Ctrl-C there closes ONLY"
        echo "  the SSH tunnel -- the learner on $FM_SSH_HOST keeps running"
        echo "  with its replay buffer intact."
        echo "  Nothing was started here and nothing remote was touched."
        echo "=================================================================="
    } >&2
    exit 1
fi

# --- [2/5] cross-host code identity -----------------------------------------
# Same defect class as run_hil_server.sh's HEAD guard: the server checkout can
# sit silently behind laptop3, and then the served policy is not the code being
# reviewed here.  Committed HEADs only -- a dirty laptop3 tree is out of scope.
LAPTOP_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
[[ "$LAPTOP_HEAD" =~ ^[0-9a-f]{40}$ ]] || die "cannot read laptop3 HEAD from $REPO_ROOT"
# The `|| true` is load-bearing: under `set -o pipefail` an ssh failure (255)
# would abort the script here with a bare ssh error, and the operator would
# never see which host/repo could not be read.
REMOTE_HEAD="$(rsh "git -C '$FM_REMOTE_REPO' rev-parse HEAD 2>/dev/null || true" | tr -d '[:space:]' || true)"
[[ "$REMOTE_HEAD" =~ ^[0-9a-f]{40}$ ]] || \
    die "cannot read HEAD of $FM_SSH_HOST:$FM_REMOTE_REPO (got: '${REMOTE_HEAD:-<empty>}')"
if [[ "$LAPTOP_HEAD" != "$REMOTE_HEAD" ]]; then
    {
        echo "=================================================================="
        echo "  !!  SERVER / LAPTOP3 CODE IDENTITY MISMATCH  !!"
        echo "  laptop3 : $LAPTOP_HEAD"
        echo "  server  : $REMOTE_HEAD  ($FM_REMOTE_REPO)"
        echo "  The FM server code runs on the server.  Align the checkouts:"
        echo "    git -C $REPO_ROOT push"
        echo "    ssh $FM_SSH_HOST 'git -C $FM_REMOTE_REPO pull --ff-only'"
        echo "  A REPORT, not a repair: nothing here fetches, resets or pulls."
        echo "=================================================================="
    } >&2
    [[ "$FM_ACCEPT_HEAD_MISMATCH" == "1" ]] || \
        die "refusing to start the FM server on a different commit (set FM_ACCEPT_HEAD_MISMATCH=1 to accept it deliberately)"
    echo "FM_ACCEPT_HEAD_MISMATCH=1: continuing despite the mismatch." >&2
fi

# --- [3/5] stale FM server --------------------------------------------------
# Never auto-kill here: an occupied port proves nothing about ownership, and
# the previous run's log is evidence someone may still need.
REMOTE_PORT_STATE="$(rsh "if ss -tln 2>/dev/null | awk '{print \$4}' | grep -Eq '[:.]${FM_REMOTE_PORT}\$'; then echo BOUND; else echo FREE; fi" || true)"
REMOTE_PORT_STATE="$(printf '%s' "$REMOTE_PORT_STATE" | tr -d '[:space:]')"
# An ssh failure must not read as "free": it returns non-zero exactly like a
# grep miss, and starting a second server on top of a live one is the failure
# this check exists to prevent.
[[ "$REMOTE_PORT_STATE" == "BOUND" || "$REMOTE_PORT_STATE" == "FREE" ]] || \
    die "could not read listening sockets on $FM_SSH_HOST (got: '${REMOTE_PORT_STATE:-<empty>}')"
if [[ "$REMOTE_PORT_STATE" == "BOUND" ]]; then
    {
        echo "=================================================================="
        echo "  !!  $FM_SSH_HOST ALREADY HAS SOMETHING ON PORT $FM_REMOTE_PORT  !!"
        echo "  A previous FM server is probably still up.  Nothing was started"
        echo "  and nothing was stopped.  Find its run directory:"
        echo "    ssh $FM_SSH_HOST 'ls -1dt $FM_REMOTE_DATA_ROOT/fm_eval/*/ | head -5'"
    } >&2
    print_stop_hint "<that run dir>"
    echo "==================================================================" >&2
    exit 1
fi

# --- [4/5] start the FM server ----------------------------------------------
# Marshalled through `bash -s --` rather than one long quoted command string
# for a specific reason: in `cd X && mkdir Y && nohup P & echo $! > pid`, the
# `&` terminates the whole AND-list, so $! would be a SHELL pid, not python's.
# The cleanup guard checks /proc/<pid>/cmdline for run_fm_policy_server.py, so
# a shell pid there would silently disable the only stop path we have.
echo "Starting FM policy server on $FM_SSH_HOST (GPU $FM_GPU_INDEX, port $FM_REMOTE_PORT)"
echo "  artifact: $FM_ARTIFACT_DIR (--which $FM_WHICH)"
echo "  run dir : $RUN_DIR"
if [[ -n "$step_timing" ]]; then
    echo "[fm-launcher] step timing ON -> $RUN_DIR/served/timing.jsonl"
fi
rsh bash -s -- "$FM_REMOTE_REPO" "$RUN_DIR" "$FM_REMOTE_PYTHON" \
    "$FM_ARTIFACT_DIR" "$FM_REMOTE_PORT" "$FM_GPU_INDEX" "$FM_WHICH" "$step_timing" <<'REMOTE_START' || \
    die "remote FM server launch failed; nothing was started and nothing else was touched"
set -euo pipefail
# $8 via ${8:-}: ssh flattens its argv into one command string, so an EMPTY
# trailing argument (timing off) disappears before the remote shell re-splits
# it, and a bare "$8" would then trip `set -u`.
repo="$1"; run_dir="$2"; py="$3"; artifact="$4"; port="$5"; gpu="$6"; which="$7"; step_timing="${8:-}"

[[ -d "$repo" ]] || { echo "REMOTE ERROR: repo is missing: $repo" >&2; exit 1; }
[[ -x "$py" ]] || { echo "REMOTE ERROR: python is missing: $py" >&2; exit 1; }
[[ -f "$repo/serl_ur_infra/scripts/run_fm_policy_server.py" ]] || \
    { echo "REMOTE ERROR: FM entrypoint is missing from $repo (pull the runtime checkout)" >&2; exit 1; }
[[ -d "$artifact" ]] || { echo "REMOTE ERROR: fm-init artifact dir is missing: $artifact" >&2; exit 1; }
[[ ! -e "$run_dir" ]] || { echo "REMOTE ERROR: run dir already exists: $run_dir" >&2; exit 1; }

mkdir -p "$run_dir"
cd "$repo"
# CUDA_VISIBLE_DEVICES + XLA_PYTHON_CLIENT_PREALLOCATE=false mirror
# run_hil_server.sh's learner launch: preallocation off is what lets this
# process coexist on the same GPU as the learner instead of fighting it for
# the whole card.  PYTHONPATH mirrors the same launch so serl_launcher
# resolves.  --gpu-index '' is explicit "the environment already chose".
nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    HIL_STEP_TIMING="$step_timing" \
    PYTHONPATH="$repo/serl_ur_infra:$repo/third_party/hil-serl/serl_launcher" \
    "$py" serl_ur_infra/scripts/run_fm_policy_server.py \
    --artifact-dir "$artifact" \
    --which "$which" \
    --port "$port" \
    --record-root "$run_dir/served" \
    --gpu-index '' \
    >"$run_dir/server.log" 2>&1 </dev/null &
echo $! >"$run_dir/.fm_server.pid"
REMOTE_START

# --- [5/5] wait for ready ---------------------------------------------------
# Poll the log rather than the port: a bound port only proves gRPC came up,
# while the ready line is printed after the artifact is loaded and validated.
poll_once() {
    rsh bash -s -- "$RUN_DIR" <<'REMOTE_POLL'
set -uo pipefail
run_dir="$1"
ready_mark='[fm-server] ready '
fatal_mark='[fm-server] FATAL'
hit="$(grep -F -m1 -e "$fatal_mark" -e "$ready_mark" "$run_dir/server.log" 2>/dev/null || true)"
if [[ "$hit" == *"$fatal_mark"* ]]; then printf 'FATAL %s\n' "$hit"; exit 0; fi
if [[ -n "$hit" ]]; then printf 'READY %s\n' "$hit"; exit 0; fi
pid="$(cat "$run_dir/.fm_server.pid" 2>/dev/null || true)"
if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then printf 'WAIT\n'; else printf 'DEAD\n'; fi
REMOTE_POLL
}

fail_with_log() {
    echo "ERROR: $*" >&2
    echo "--- last 40 lines of $LOGFILE ---" >&2
    rsh "tail -n 40 '$LOGFILE' 2>/dev/null || true" >&2 || true
    echo "----------------------------------" >&2
    echo "  The server process was NOT stopped, so the evidence is intact." >&2
    print_stop_hint "$RUN_DIR"
    exit 1
}

READY_LINE=""
DEADLINE=$(( $(date +%s) + FM_START_TIMEOUT_S ))
while :; do
    STATE="$(poll_once || true)"
    case "$STATE" in
        READY\ *) READY_LINE="${STATE#READY }"; break ;;
        FATAL\ *) fail_with_log "FM server reported ${STATE#FATAL }" ;;
        DEAD) fail_with_log "FM server process exited before reporting ready" ;;
    esac
    (( $(date +%s) < DEADLINE )) || \
        fail_with_log "FM server did not report ready within ${FM_START_TIMEOUT_S}s"
    sleep 3
done

echo "FM_SERVER_RUN_DIR=$RUN_DIR"
echo "FM_SERVER_READY_LINE=$READY_LINE"
echo "FM_SERVER_RESULT=started"

# --- cleanup contract -------------------------------------------------------
# Installed only now, on purpose: a server that never became ready is left
# running with its log, and the paths above say so.  From here on the server is
# ours for the lifetime of this terminal.
FM_SERVER_STOPPED=0
remote_cmdline() {
    rsh "tr '\\0' ' ' < /proc/$1/cmdline 2>/dev/null || true" 2>/dev/null || true
}
stop_fm_server() {
    (( FM_SERVER_STOPPED == 0 )) || return 0
    FM_SERVER_STOPPED=1
    local pid cmdline
    pid="$(rsh "cat '$PIDFILE' 2>/dev/null || true" 2>/dev/null || true)"
    pid="$(printf '%s' "$pid" | tr -d '[:space:]')"
    # Guard 1: a PID we wrote ourselves, non-empty and numeric.  No name match,
    # no pattern, no pkill -- those could reach any process on the host.
    if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
        echo "NOT STOPPED: no usable PID in $FM_SSH_HOST:$PIDFILE (read: '$pid'); stop it by hand." >&2
        return 0
    fi
    # Guard 2: the PID still IS our server.  PIDs are recycled, and the
    # recycled one could be anything, including something we must never touch.
    cmdline="$(remote_cmdline "$pid")"
    if [[ "$cmdline" != *run_fm_policy_server.py* ]]; then
        echo "NOT STOPPED: PID $pid on $FM_SSH_HOST is not the FM server (cmdline: ${cmdline:-<gone>}); pidfile: $PIDFILE" >&2
        return 0
    fi
    echo "Stopping FM server PID $pid on $FM_SSH_HOST (TERM)..."
    rsh "kill $pid" >/dev/null 2>&1 || true
    if rsh "for _ in \$(seq 1 20); do kill -0 $pid 2>/dev/null || exit 0; sleep 0.5; done; exit 1" >/dev/null 2>&1; then
        echo "FM server PID $pid exited.  Recording: $RUN_DIR"
        return 0
    fi
    # Guard 3: re-verify before escalating -- the 10 s wait is long enough for
    # the PID to have died and been reused.
    cmdline="$(remote_cmdline "$pid")"
    if [[ "$cmdline" == *run_fm_policy_server.py* ]]; then
        rsh "kill -9 $pid" >/dev/null 2>&1 || true
        echo "FM server PID $pid ignored TERM; sent KILL." >&2
    else
        echo "NOT KILLED: PID $pid is no longer the FM server; it was left alone." >&2
    fi
}
TUNNEL_PID=""
cleanup() {
    # Our own direct child, killed by the PID we spawned -- never by name.
    if [[ -n "$TUNNEL_PID" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
        TUNNEL_PID=""
    fi
    stop_fm_server
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

echo "============================================================"
echo "FM policy server ready"
echo "  server  : $FM_SSH_HOST GPU $FM_GPU_INDEX port $FM_REMOTE_PORT"
echo "  serving : $FM_ARTIFACT_DIR (--which $FM_WHICH)"
echo "  run dir : $RUN_DIR"
echo "  log     : $LOGFILE"
echo "  actor   : grpc://127.0.0.1:$FM_LOCAL_PORT"
echo "============================================================"
echo "Keep this terminal open.  Ctrl-C closes the tunnel AND stops the FM"
echo "server started above -- and only that process."
echo "The production learner on $FM_SSH_HOST is untouched and still running."

# Backgrounded + `wait`, exactly like run_hil_server.sh, and NOT exec'd -- an
# exec would replace this shell and no trap could ever run.  Terminal Ctrl-C
# reaches both the shell and this child (same process group, job control off),
# so it lands in the traps either way; what the background shape additionally
# buys is a directed `kill -TERM <launcher pid>`.  Measured in the BC sibling:
# with a FOREGROUND ssh, bash defers pending traps until the child exits, so a
# directed TERM cleans up nothing until the tunnel happens to die -- which
# would strand an FM server holding GPU memory and port $FM_REMOTE_PORT.
ssh -N -T "${SSH_OPTIONS[@]}" \
    -o ExitOnForwardFailure=yes \
    -L "127.0.0.1:$FM_LOCAL_PORT:127.0.0.1:$FM_REMOTE_PORT" \
    "$FM_SSH_HOST" &
TUNNEL_PID=$!
set +e
wait "$TUNNEL_PID"
TUNNEL_RC=$?
set -e
TUNNEL_PID=""
echo "SSH tunnel exited with status $TUNNEL_RC." >&2
exit "$TUNNEL_RC"
