#!/usr/bin/env bash
# =============================================================================
# run_bc_server.sh -- start the remote BC policy server and hold its tunnel
# =============================================================================
#
# Terminal 1 of a BC real-robot evaluation:  ./run_bc_server.sh
#
# Starts serl_ur_infra/scripts/run_bc_policy_server.py on the GPU host and holds
# laptop3 127.0.0.1:50153 -> server 127.0.0.1:50054, so the UNMODIFIED production
# actor (which always dials 50153) reaches the BC server.  No production script,
# actor or hardware terminal changes.
#
# OWNERSHIP -- read before editing:
#
#   * The production HIL-SERL learner runs on the server at port 50053 and KEEPS
#     RUNNING throughout a BC eval.  It is not ours.  This script names 50053
#     only to refuse it, never runs pkill/killall, never matches a process by
#     name or pattern, and never opens the production launcher lock
#     ($BC_REMOTE_DATA_ROOT/.run_hil_server.lock).  Structurally, the only
#     process it can signal is one whose PID it read from its OWN per-run
#     pidfile and whose /proc/<pid>/cmdline still contains
#     run_bc_policy_server.py.
#   * The BC server is detached on the server host; the cleanup trap stops it
#     because a BC eval is a bounded experiment, unlike a learner lineage.  A
#     server that never became READY is instead LEFT RUNNING with its log
#     intact (as run_hil_server.sh preserves a failed lineage) -- the
#     stale-port check plus the printed stop command dispose of it afterwards.
#
# Overrides (defaults shown):  HIL_SSH_HOST=junhyeong_ai ·
# HIL_REMOTE_REPO=/home/junhyeong/gello_software_runtime ·
# HIL_REMOTE_PYTHON=/home/junhyeong/miniconda3/envs/il/bin/python ·
# HIL_REMOTE_DATA_ROOT=/home/junhyeong/hil-serl-data · HIL_GPU_INDEX=0 ·
# BC_LOCAL_PORT=50153 (what the actor dials) · BC_REMOTE_PORT=50054 (50053 is
# refused) · BC_ARTIFACT_DIR=<data root>/diagnostics/<bc-init artifact> ·
# BC_START_TIMEOUT_S=300 · BC_ACCEPT_HEAD_MISMATCH=0 (1 downgrades the HEAD
# refusal to a warning).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BC_SSH_HOST="${HIL_SSH_HOST:-junhyeong_ai}"
BC_REMOTE_REPO="${HIL_REMOTE_REPO:-/home/junhyeong/gello_software_runtime}"
BC_REMOTE_PYTHON="${HIL_REMOTE_PYTHON:-/home/junhyeong/miniconda3/envs/il/bin/python}"
BC_REMOTE_DATA_ROOT="${HIL_REMOTE_DATA_ROOT:-/home/junhyeong/hil-serl-data}"
BC_GPU_INDEX="${HIL_GPU_INDEX:-0}"
BC_LOCAL_PORT="${BC_LOCAL_PORT:-50153}"
BC_REMOTE_PORT="${BC_REMOTE_PORT:-50054}"
BC_ARTIFACT_DIR="${BC_ARTIFACT_DIR:-$BC_REMOTE_DATA_ROOT/diagnostics/bc_cube_in_cup_raw_0731_bce_group_holdout_20epoch_20260731_213638.bc-init}"
BC_START_TIMEOUT_S="${BC_START_TIMEOUT_S:-300}"
BC_ACCEPT_HEAD_MISMATCH="${BC_ACCEPT_HEAD_MISMATCH:-0}"

# The ready/FATAL markers live ONLY inside the remote poll script below, and
# that is not tidiness: ssh flattens its command argv into a single string that
# the remote login shell re-splits, so an argument containing spaces (both
# markers do) would arrive as several arguments and silently never match.  The
# same rule is why every value marshalled to `bash -s --` below is validated
# whitespace-free.
#
# One directory per launch: the pidfile that authorises the only kill this
# script can perform must never be shared with an earlier BC server.
RUN_DIR="$BC_REMOTE_DATA_ROOT/bc_eval/bc_eval_$(date -u +%Y%m%d_%H%M%S)"
PIDFILE="$RUN_DIR/.bc_server.pid"
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

rsh() { ssh "${SSH_OPTIONS[@]}" "$BC_SSH_HOST" "$@"; }

print_stop_hint() {
    local run_dir="$1"
    echo "  To inspect and stop THAT BC server (and nothing else):" >&2
    echo "    ssh $BC_SSH_HOST \"pid=\\\$(cat $run_dir/.bc_server.pid); tr '\\\\0' ' ' < /proc/\\\$pid/cmdline; echo\"" >&2
    echo "    # confirm the line above contains run_bc_policy_server.py, then:" >&2
    echo "    ssh $BC_SSH_HOST \"kill \\\$(cat $run_dir/.bc_server.pid)\"" >&2
}

# --- argument sanity ---------------------------------------------------------
for _name in BC_LOCAL_PORT BC_REMOTE_PORT; do
    _value="${!_name}"
    [[ "$_value" =~ ^[0-9]+$ ]] || die "$_name must be numeric (got: $_value)"
    (( _value >= 1024 && _value <= 65535 )) || die "$_name must be in [1024, 65535]"
done
# The production learner owns 50053 on the server host.  Refusing the number
# here -- rather than trusting every downstream guard -- is what makes it
# impossible for this launcher to start on, probe, or stop the learner.
(( BC_REMOTE_PORT != 50053 )) || \
    die "remote port 50053 belongs to the production learner; run_bc_server.sh will not use it"
if (( BC_LOCAL_PORT != 50153 )); then
    echo "INFO: BC_LOCAL_PORT=$BC_LOCAL_PORT, but the production actor always dials 50153; it will not find this tunnel." >&2
fi
[[ "$BC_GPU_INDEX" =~ ^[0-9]+$ ]] || die "HIL_GPU_INDEX must be a non-negative integer"
[[ "$BC_START_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || die "BC_START_TIMEOUT_S must be a positive integer"
(( BC_START_TIMEOUT_S <= 1800 )) || die "BC_START_TIMEOUT_S must be <= 1800"
[[ -n "$BC_SSH_HOST" ]] || die "HIL_SSH_HOST cannot be empty"
# These become remote paths.  Keep them boring: a rejected override is far
# cheaper than a remote path that expands into something plausible but wrong.
for _name in BC_REMOTE_DATA_ROOT BC_REMOTE_REPO BC_ARTIFACT_DIR; do
    _value="${!_name}"
    [[ "$_value" == /* ]] || die "$_name must be an absolute path (got: $_value)"
    [[ "$_value" != *".."* ]] || die "$_name must not contain '..'"
    [[ "$_value" != *[[:space:]]* ]] || die "$_name must not contain whitespace"
done
command -v ssh >/dev/null 2>&1 || die "ssh is not installed"

# --- [1/5] local actor endpoint ---------------------------------------------
# The production run_hil_server.sh tunnel binds the SAME local port to the
# learner.  Two tunnels cannot share it, and we must never guess which one
# owns a bound port, so refuse and tell the operator what to close.
local_port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -tln 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${BC_LOCAL_PORT}\$"
    else  # fallback: a successful connect proves a listener
        (exec 3<>"/dev/tcp/127.0.0.1/$BC_LOCAL_PORT") 2>/dev/null
    fi
}
if local_port_in_use; then
    {
        echo "=================================================================="
        echo "  !!  LOCAL PORT $BC_LOCAL_PORT IS ALREADY BOUND  !!"
        echo "  Almost certainly the production tunnel terminal running"
        echo "  ./run_hil_server.sh.  Close THAT terminal with Ctrl-C first:"
        echo "  Ctrl-C there closes ONLY the SSH tunnel -- the learner on"
        echo "  $BC_SSH_HOST keeps running with its replay buffer intact."
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
REMOTE_HEAD="$(rsh "git -C '$BC_REMOTE_REPO' rev-parse HEAD 2>/dev/null || true" | tr -d '[:space:]' || true)"
[[ "$REMOTE_HEAD" =~ ^[0-9a-f]{40}$ ]] || \
    die "cannot read HEAD of $BC_SSH_HOST:$BC_REMOTE_REPO (got: '${REMOTE_HEAD:-<empty>}')"
if [[ "$LAPTOP_HEAD" != "$REMOTE_HEAD" ]]; then
    {
        echo "=================================================================="
        echo "  !!  SERVER / LAPTOP3 CODE IDENTITY MISMATCH  !!"
        echo "  laptop3 : $LAPTOP_HEAD"
        echo "  server  : $REMOTE_HEAD  ($BC_REMOTE_REPO)"
        echo "  The BC server code runs on the server.  Align the checkouts:"
        echo "    git -C $REPO_ROOT push"
        echo "    ssh $BC_SSH_HOST 'git -C $BC_REMOTE_REPO pull --ff-only'"
        echo "  A REPORT, not a repair: nothing here fetches, resets or pulls."
        echo "=================================================================="
    } >&2
    [[ "$BC_ACCEPT_HEAD_MISMATCH" == "1" ]] || \
        die "refusing to start the BC server on a different commit (set BC_ACCEPT_HEAD_MISMATCH=1 to accept it deliberately)"
    echo "BC_ACCEPT_HEAD_MISMATCH=1: continuing despite the mismatch." >&2
fi

# --- [3/5] stale BC server --------------------------------------------------
# Never auto-kill here: an occupied port proves nothing about ownership, and
# the previous run's log is evidence someone may still need.
REMOTE_PORT_STATE="$(rsh "if ss -tln 2>/dev/null | awk '{print \$4}' | grep -Eq '[:.]${BC_REMOTE_PORT}\$'; then echo BOUND; else echo FREE; fi" || true)"
REMOTE_PORT_STATE="$(printf '%s' "$REMOTE_PORT_STATE" | tr -d '[:space:]')"
# An ssh failure must not read as "free": it returns non-zero exactly like a
# grep miss, and starting a second server on top of a live one is the failure
# this check exists to prevent.
[[ "$REMOTE_PORT_STATE" == "BOUND" || "$REMOTE_PORT_STATE" == "FREE" ]] || \
    die "could not read listening sockets on $BC_SSH_HOST (got: '${REMOTE_PORT_STATE:-<empty>}')"
if [[ "$REMOTE_PORT_STATE" == "BOUND" ]]; then
    {
        echo "=================================================================="
        echo "  !!  $BC_SSH_HOST ALREADY HAS SOMETHING ON PORT $BC_REMOTE_PORT  !!"
        echo "  A previous BC server is probably still up.  Nothing was started"
        echo "  and nothing was stopped.  Find its run directory:"
        echo "    ssh $BC_SSH_HOST 'ls -1dt $BC_REMOTE_DATA_ROOT/bc_eval/*/ | head -5'"
    } >&2
    print_stop_hint "<that run dir>"
    echo "==================================================================" >&2
    exit 1
fi

# --- [4/5] start the BC server ----------------------------------------------
# Marshalled through `bash -s --` rather than one long quoted command string
# for a specific reason: in `cd X && mkdir Y && nohup P & echo $! > pid`, the
# `&` terminates the whole AND-list, so $! would be a SHELL pid, not python's.
# The cleanup guard checks /proc/<pid>/cmdline for run_bc_policy_server.py, so
# a shell pid there would silently disable the only stop path we have.
echo "Starting BC policy server on $BC_SSH_HOST (GPU $BC_GPU_INDEX, port $BC_REMOTE_PORT)"
echo "  artifact: $BC_ARTIFACT_DIR"
echo "  run dir : $RUN_DIR"
rsh bash -s -- "$BC_REMOTE_REPO" "$RUN_DIR" "$BC_REMOTE_PYTHON" \
    "$BC_ARTIFACT_DIR" "$BC_REMOTE_PORT" "$BC_GPU_INDEX" <<'REMOTE_START' || \
    die "remote BC server launch failed; nothing was started and nothing else was touched"
set -euo pipefail
repo="$1"; run_dir="$2"; py="$3"; artifact="$4"; port="$5"; gpu="$6"

[[ -d "$repo" ]] || { echo "REMOTE ERROR: repo is missing: $repo" >&2; exit 1; }
[[ -x "$py" ]] || { echo "REMOTE ERROR: python is missing: $py" >&2; exit 1; }
[[ -f "$repo/serl_ur_infra/scripts/run_bc_policy_server.py" ]] || \
    { echo "REMOTE ERROR: BC entrypoint is missing from $repo (pull the runtime checkout)" >&2; exit 1; }
[[ -d "$artifact" ]] || { echo "REMOTE ERROR: bc-init artifact dir is missing: $artifact" >&2; exit 1; }
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
    PYTHONPATH="$repo/serl_ur_infra:$repo/third_party/hil-serl/serl_launcher" \
    "$py" serl_ur_infra/scripts/run_bc_policy_server.py \
    --artifact-dir "$artifact" \
    --port "$port" \
    --record-root "$run_dir/served" \
    --gpu-index '' \
    >"$run_dir/server.log" 2>&1 </dev/null &
echo $! >"$run_dir/.bc_server.pid"
REMOTE_START

# --- [5/5] wait for ready ---------------------------------------------------
# Poll the log rather than the port: a bound port only proves gRPC came up,
# while the ready line is printed after the artifact is loaded and validated.
poll_once() {
    rsh bash -s -- "$RUN_DIR" <<'REMOTE_POLL'
set -uo pipefail
run_dir="$1"
ready_mark='[bc-server] ready '
fatal_mark='[bc-server] FATAL'
hit="$(grep -F -m1 -e "$fatal_mark" -e "$ready_mark" "$run_dir/server.log" 2>/dev/null || true)"
if [[ "$hit" == *"$fatal_mark"* ]]; then printf 'FATAL %s\n' "$hit"; exit 0; fi
if [[ -n "$hit" ]]; then printf 'READY %s\n' "$hit"; exit 0; fi
pid="$(cat "$run_dir/.bc_server.pid" 2>/dev/null || true)"
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
DEADLINE=$(( $(date +%s) + BC_START_TIMEOUT_S ))
while :; do
    STATE="$(poll_once || true)"
    case "$STATE" in
        READY\ *) READY_LINE="${STATE#READY }"; break ;;
        FATAL\ *) fail_with_log "BC server reported ${STATE#FATAL }" ;;
        DEAD) fail_with_log "BC server process exited before reporting ready" ;;
    esac
    (( $(date +%s) < DEADLINE )) || \
        fail_with_log "BC server did not report ready within ${BC_START_TIMEOUT_S}s"
    sleep 3
done

echo "BC_SERVER_RUN_DIR=$RUN_DIR"
echo "BC_SERVER_READY_LINE=$READY_LINE"
echo "BC_SERVER_RESULT=started"

# --- cleanup contract -------------------------------------------------------
# Installed only now, on purpose: a server that never became ready is left
# running with its log, and the paths above say so.  From here on the server is
# ours for the lifetime of this terminal.
BC_SERVER_STOPPED=0
remote_cmdline() {
    rsh "tr '\\0' ' ' < /proc/$1/cmdline 2>/dev/null || true" 2>/dev/null || true
}
stop_bc_server() {
    (( BC_SERVER_STOPPED == 0 )) || return 0
    BC_SERVER_STOPPED=1
    local pid cmdline
    pid="$(rsh "cat '$PIDFILE' 2>/dev/null || true" 2>/dev/null || true)"
    pid="$(printf '%s' "$pid" | tr -d '[:space:]')"
    # Guard 1: a PID we wrote ourselves, non-empty and numeric.  No name match,
    # no pattern, no pkill -- those could reach any process on the host.
    if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
        echo "NOT STOPPED: no usable PID in $BC_SSH_HOST:$PIDFILE (read: '$pid'); stop it by hand." >&2
        return 0
    fi
    # Guard 2: the PID still IS our server.  PIDs are recycled, and the
    # recycled one could be anything, including something we must never touch.
    cmdline="$(remote_cmdline "$pid")"
    if [[ "$cmdline" != *run_bc_policy_server.py* ]]; then
        echo "NOT STOPPED: PID $pid on $BC_SSH_HOST is not the BC server (cmdline: ${cmdline:-<gone>}); pidfile: $PIDFILE" >&2
        return 0
    fi
    echo "Stopping BC server PID $pid on $BC_SSH_HOST (TERM)..."
    rsh "kill $pid" >/dev/null 2>&1 || true
    if rsh "for _ in \$(seq 1 20); do kill -0 $pid 2>/dev/null || exit 0; sleep 0.5; done; exit 1" >/dev/null 2>&1; then
        echo "BC server PID $pid exited.  Recording: $RUN_DIR"
        return 0
    fi
    # Guard 3: re-verify before escalating -- the 10 s wait is long enough for
    # the PID to have died and been reused.
    cmdline="$(remote_cmdline "$pid")"
    if [[ "$cmdline" == *run_bc_policy_server.py* ]]; then
        rsh "kill -9 $pid" >/dev/null 2>&1 || true
        echo "BC server PID $pid ignored TERM; sent KILL." >&2
    else
        echo "NOT KILLED: PID $pid is no longer the BC server; it was left alone." >&2
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
    stop_bc_server
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

echo "============================================================"
echo "BC policy server ready"
echo "  server  : $BC_SSH_HOST GPU $BC_GPU_INDEX port $BC_REMOTE_PORT"
echo "  run dir : $RUN_DIR"
echo "  log     : $LOGFILE"
echo "  actor   : grpc://127.0.0.1:$BC_LOCAL_PORT"
echo "============================================================"
echo "Keep this terminal open.  Ctrl-C closes the tunnel AND stops the BC"
echo "server started above -- and only that process."
echo "The production learner on $BC_SSH_HOST is untouched and still running."

# Backgrounded + `wait`, exactly like run_hil_server.sh, and NOT exec'd -- an
# exec would replace this shell and no trap could ever run.  Terminal Ctrl-C
# reaches both the shell and this child (same process group, job control off),
# so it lands in the traps either way; what the background shape additionally
# buys is a directed `kill -TERM <launcher pid>`.  Measured here: with a
# FOREGROUND ssh, bash defers pending traps until the child exits, so a
# directed TERM cleans up nothing until the tunnel happens to die -- which
# would strand a BC server holding GPU memory and port $BC_REMOTE_PORT.
ssh -N -T "${SSH_OPTIONS[@]}" \
    -o ExitOnForwardFailure=yes \
    -L "127.0.0.1:$BC_LOCAL_PORT:127.0.0.1:$BC_REMOTE_PORT" \
    "$BC_SSH_HOST" &
TUNNEL_PID=$!
set +e
wait "$TUNNEL_PID"
TUNNEL_RC=$?
set -e
TUNNEL_PID=""
echo "SSH tunnel exited with status $TUNNEL_RC." >&2
exit "$TUNNEL_RC"
