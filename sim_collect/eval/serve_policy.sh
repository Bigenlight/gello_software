#!/usr/bin/env bash
# =============================================================================
# serve_policy.sh -- ONE entry point for the policy server the sim eval talks to
# =============================================================================
#
# sim_collect/eval/run_eval.py speaks the REAL deploy protocol (ZMQ REQ/REP,
# policy_server/zmq_protocol.py).  That means the thing on the other end of the
# socket is the very same act/diffusion/fm server the robot uses -- this script
# only starts it with the carrot parameters and, when it runs on kanu, holds the
# SSH tunnel that makes it look local to the eval.
#
#   LOCAL (laptop3 CPU -- slow, see the banner):
#     ./sim_collect/eval/serve_policy.sh --type act \
#         --checkpoint /path/to/.../checkpoints/last/pretrained_model
#
#   REMOTE (kanu GPU + tunnel; this terminal owns ONLY the tunnel):
#     ./sim_collect/eval/serve_policy.sh --type fm --remote kanu --gpus 7 --sync \
#         --checkpoint /home/junhyeong/workspace/youngwoong/carrot_eef/outputs/<run>/checkpoints/last/pretrained_model
#
#   DRY RUN (prints every command it would run; touches nothing, no ssh):
#     ./sim_collect/eval/serve_policy.sh --type fm --remote kanu --gpus 7 --check \
#         --checkpoint /remote/path/pretrained_model
#
#   STOP the remote server (pid file only) and any tunnel this script left:
#     ./sim_collect/eval/serve_policy.sh --type fm --remote kanu --stop
#
#   PROBE an endpoint (one RESET round-trip, exit 0 on ok):
#     ./sim_collect/eval/serve_policy.sh --port 5593 --probe
#
# OWNERSHIP -- narrow on purpose, because kanu hosts other people's jobs:
#   * The remote server is detached (nohup) and SERVER-OWNED.  Ctrl-C here, a
#     dead tunnel, or a failed probe never signal it.
#   * --stop kills exactly the pid in OUR pid file (~/logs/sim_eval_<type>_<port>.pid)
#     and only after /proc/<pid>/cmdline proves it is that server on that port.
#     Nothing is ever matched by port, name or pkill.
#   * The local half owned by this script is the one ssh -N tunnel process.
#
# FLAGS
#   --type act|diffusion|fm     which server (required except for --probe)
#   --checkpoint <dir>          lerobot pretrained_model dir (LOCAL path for a
#                               local run, path ON THE REMOTE for --remote)
#   --port N                    default 5591 act / 5592 diffusion / 5593 fm
#   --local-port N              tunnel's laptop-side port (default: --port)
#   --n-action-steps N          default 30 act / 32 diffusion / 24 fm
#   --task "Put carrot in pot"  CLIP task string; FM only (act/diffusion ignore it)
#   --device cpu|cuda           default: cpu locally, cuda on --remote
#   --remote <host>             ssh alias, e.g. kanu
#   --gpus 6,7                  CUDA_VISIBLE_DEVICES on the remote (default 7)
#   --remote-venv <path>        default $DEFAULT_REMOTE_VENV (kanu lerobot venv)
#   --remote-repo <path>        scratch dir on the remote holding policy_server/
#                               (default $DEFAULT_REMOTE_REPO)
#   --sync                      rsync policy_server/ to --remote-repo first
#                               (alone, with no --checkpoint: sync and exit)
#   --wait-s N                  how long to wait for the first RESET (default 600;
#                               a cold FM/diffusion load is minutes)
#   --check / --stop / --probe  modes described above
#   -- <extra args>             everything after -- is forwarded verbatim to the
#                               server (e.g. -- --num-integration-steps 10)
#
# ENV
#   SIM_EVAL_STATE_DIR      where the tunnel pid file goes
#                           (default ${XDG_RUNTIME_DIR:-/tmp}/sim_collect_eval)
#   SIM_EVAL_REMOTE_LOGDIR  remote log/pid dir (default ~/logs)
#   SIM_EVAL_ALLOW_BUSY_GPU 1 = start even if the requested GPU already has
#                           memory in use (default 0 -- refuse, it is a shared box)
#   SIM_EVAL_HF_OFFLINE     1 (default, FM only) sets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE
#                           like run_fm_server.sh.  Set 0 if the remote has no HF
#                           cache and CLIP's from_pretrained fails at startup.
#   SIM_EVAL_REMOTE_GPUS    default for --gpus
#
# This script starts NOTHING on the remote in --check mode and installs nothing,
# anywhere, ever: a missing python package is reported with the exact pip line.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV_PY="$REPO/.venv/bin/python"                       # probe interpreter (mujoco venv, has pyzmq)
ACT_VENV="$REPO/ros2_ur_ws/act_venv"                   # local torch/lerobot venv
SCRIPTS_DIR="$REPO/ros2_ur_ws/src/gello_policy/scripts"
POLICY_SERVER_SRC="$REPO/ros2_ur_ws/src/gello_policy/policy_server"
GELLO_POLICY_PY="$REPO/ros2_ur_ws/src/gello_policy"    # holds gello_policy/obs_assembler.py

DEFAULT_REMOTE_VENV="/home/junhyeong/workspace/youngwoong/cube_flow_matching/training/lr_env"
DEFAULT_REMOTE_REPO="~/workspace/youngwoong/sim_eval_policy_server"
DEFAULT_TASK="Put carrot in pot"

STATE_DIR="${SIM_EVAL_STATE_DIR:-${XDG_RUNTIME_DIR:-/tmp}/sim_collect_eval}"
REMOTE_LOGDIR="${SIM_EVAL_REMOTE_LOGDIR:-~/logs}"
ALLOW_BUSY_GPU="${SIM_EVAL_ALLOW_BUSY_GPU:-0}"
HF_OFFLINE="${SIM_EVAL_HF_OFFLINE:-1}"

MODE="start"
TYPE=""
CHECKPOINT=""
PORT=""
LOCAL_PORT=""
NSTEPS=""
TASK=""
TASK_EXPLICIT=0
DEVICE=""
REMOTE_HOST=""
GPUS="${SIM_EVAL_REMOTE_GPUS:-}"
REMOTE_VENV="$DEFAULT_REMOTE_VENV"
REMOTE_REPO="$DEFAULT_REMOTE_REPO"
DO_SYNC=0
WAIT_S=600
EXTRA=()

die() { echo "serve_policy.sh: $*" >&2; exit 2; }

usage() { sed -n '2,80p' "${BASH_SOURCE[0]}"; }

# POSIX-safe single-quoting: the remote LOGIN shell parses the command line ssh
# sends, so bash's printf %q (which can emit $'...') is not safe to rely on.
shquote() { local s="$1"; printf "'%s'" "${s//\'/\'\\\'\'}"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --type)           TYPE="$2"; shift 2 ;;
        --type=*)         TYPE="${1#*=}"; shift ;;
        --checkpoint)     CHECKPOINT="$2"; shift 2 ;;
        --checkpoint=*)   CHECKPOINT="${1#*=}"; shift ;;
        --port)           PORT="$2"; shift 2 ;;
        --port=*)         PORT="${1#*=}"; shift ;;
        --local-port)     LOCAL_PORT="$2"; shift 2 ;;
        --local-port=*)   LOCAL_PORT="${1#*=}"; shift ;;
        --n-action-steps) NSTEPS="$2"; shift 2 ;;
        --n-action-steps=*) NSTEPS="${1#*=}"; shift ;;
        --task)           TASK="$2"; TASK_EXPLICIT=1; shift 2 ;;
        --task=*)         TASK="${1#*=}"; TASK_EXPLICIT=1; shift ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --device=*)       DEVICE="${1#*=}"; shift ;;
        --remote)         REMOTE_HOST="$2"; shift 2 ;;
        --remote=*)       REMOTE_HOST="${1#*=}"; shift ;;
        --gpus)           GPUS="$2"; shift 2 ;;
        --gpus=*)         GPUS="${1#*=}"; shift ;;
        --remote-venv)    REMOTE_VENV="$2"; shift 2 ;;
        --remote-venv=*)  REMOTE_VENV="${1#*=}"; shift ;;
        --remote-repo)    REMOTE_REPO="$2"; shift 2 ;;
        --remote-repo=*)  REMOTE_REPO="${1#*=}"; shift ;;
        --wait-s)         WAIT_S="$2"; shift 2 ;;
        --wait-s=*)       WAIT_S="${1#*=}"; shift ;;
        --sync)           DO_SYNC=1; shift ;;
        --check|--dry-run) MODE="check"; shift ;;
        --stop)           MODE="stop"; shift ;;
        --probe)          MODE="probe"; shift ;;
        -h|--help)        usage; exit 0 ;;
        --)               shift; EXTRA=("$@"); break ;;
        *) die "unknown argument '$1' (try --help)" ;;
    esac
done

# --- resolve per-type defaults ----------------------------------------------
case "$TYPE" in
    act)       : "${PORT:=5591}"; : "${NSTEPS:=30}" ;;
    diffusion) : "${PORT:=5592}"; : "${NSTEPS:=32}" ;;
    fm)        : "${PORT:=5593}"; : "${NSTEPS:=24}" ;;
    "")        [ "$MODE" = "probe" ] || die "--type act|diffusion|fm is required" ;;
    *)         die "--type must be act, diffusion or fm (got '$TYPE')" ;;
esac
[ -n "$PORT" ] || die "--port is required when --type is omitted"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "--port must be a number (got '$PORT')"
: "${LOCAL_PORT:=$PORT}"
[[ "$LOCAL_PORT" =~ ^[0-9]+$ ]] || die "--local-port must be a number"
[[ "$WAIT_S" =~ ^[0-9]+$ ]] || die "--wait-s must be a whole number of seconds"
: "${TASK:=$DEFAULT_TASK}"
if [ -n "$REMOTE_HOST" ]; then : "${DEVICE:=cuda}"; else : "${DEVICE:=cpu}"; fi
case "$DEVICE" in cpu|cuda) ;; *) die "--device must be cpu or cuda" ;; esac
[ -n "$GPUS" ] || GPUS="7"
if [ "$DO_SYNC" = "1" ] && [ "$MODE" = "start" ] && [ -z "$CHECKPOINT" ]; then
    MODE="sync"
fi
if [ "$TASK_EXPLICIT" = "1" ] && [ -n "$TYPE" ] && [ "$TYPE" != "fm" ]; then
    echo "WARNING: --task is FM-only. ${TYPE}_server.py has no --task flag (it would" >&2
    echo "         exit 2 on it), and the ACT/Diffusion policies are not language" >&2
    echo "         conditioned anyway. The string is dropped here, not forwarded." >&2
fi

# ---------------------------------------------------------------------------
# probe: one RESET round-trip, retried until the deadline.  Used by --probe and
# by the remote start path (a cold FM load takes minutes, so the wait is long).
# A fresh REQ socket per attempt on purpose: a REQ socket that timed out is
# wedged in the send state and every later recv would fail for the wrong reason.
# ---------------------------------------------------------------------------
probe_endpoint() {   # probe_endpoint <port> <deadline_s> [per_try_timeout_s]
    local port="$1" wait_s="$2" per_try="${3:-2.0}"
    [ -x "$VENV_PY" ] || die "probe interpreter missing: $VENV_PY"
    GELLO_POLICY_PY="$GELLO_POLICY_PY" "$VENV_PY" - "$port" "$wait_s" "$per_try" <<'PY'
import json
import os
import sys
import time

sys.path.insert(0, os.environ["GELLO_POLICY_PY"])
from gello_policy.obs_assembler import build_reset_request, parse_reply  # noqa: E402

import zmq  # noqa: E402

port = int(sys.argv[1])
deadline = time.time() + float(sys.argv[2])
timeout_ms = int(float(sys.argv[3]) * 1000)

ctx = zmq.Context.instance()
attempt = 0
t0 = time.time()
last_note = t0
last_err = "no attempt made"
while True:
    attempt += 1
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        sock.connect(f"tcp://127.0.0.1:{port}")
        sock.send_multipart(build_reset_request())
        reply = parse_reply(sock.recv_multipart())
        if reply.get("ok"):
            print(f"SIM_EVAL_PROBE=ok attempts={attempt} elapsed_s={time.time() - t0:.1f}")
            extra = {k: v for k, v in reply.items() if k != "ok"}
            if extra:
                print("SIM_EVAL_PROBE_INFO=" + json.dumps(extra, sort_keys=True))
            raise SystemExit(0)
        # A reply means the server is up and answering -- ok:false is a real
        # failure (bad checkpoint, wrong dims), not something waiting fixes.
        print(f"SIM_EVAL_PROBE=refused err={reply.get('err')!r}", file=sys.stderr)
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as exc:  # zmq timeout, connection refused, malformed reply
        last_err = f"{type(exc).__name__}: {exc}"
    finally:
        sock.close(linger=0)
    now = time.time()
    if now >= deadline:
        print(
            f"SIM_EVAL_PROBE=timeout attempts={attempt} elapsed_s={now - t0:.1f} "
            f"last={last_err}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if now - last_note >= 15.0:
        print(f"  ... still waiting for RESET ({now - t0:.0f}s, last: {last_err})", file=sys.stderr)
        last_note = now
    time.sleep(1.0)
PY
}

port_is_free() {   # port_is_free <port>  (laptop side)
    "$VENV_PY" - "$1" <<'PY'
import socket
import sys

sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=3)

STATE_KEY="${REMOTE_HOST:-local}_${TYPE:-any}_${PORT}"
TUNNEL_STATE="$STATE_DIR/${STATE_KEY}.tunnel"
REMOTE_PIDFILE="$REMOTE_LOGDIR/sim_eval_${TYPE}_${PORT}.pid"

# The server argv, identical in --check and in the real launch so the preview
# never drifts from what actually runs.
server_args() {
    local args=(--checkpoint "$CHECKPOINT" --host 127.0.0.1 --port "$PORT"
                --device "$DEVICE" --n-action-steps "$NSTEPS")
    [ "$TYPE" = "fm" ] && args+=(--task "$TASK")
    args+=(${EXTRA[@]+"${EXTRA[@]}"})
    printf '%s\n' "${args[@]}"
}

quoted_server_args() {
    local out="" a
    while IFS= read -r a; do out+=" $(shquote "$a")"; done < <(server_args)
    printf '%s' "${out# }"
}

rsync_cmdline() {
    printf 'rsync -az --delete --exclude __pycache__/ %s/ %s:%s/policy_server/' \
        "$POLICY_SERVER_SRC" "$REMOTE_HOST" "$REMOTE_REPO"
}

# The remote side expands a leading ~ itself (ssh sends the arguments quoted, so
# the remote shell never sees a bare tilde).  The preview shows $HOME instead so
# the printed line is copy-pasteable on the remote as-is.
preview_path() { printf '%s' "${1/#\~/\$HOME}"; }

remote_launch_preview() {
    local repo logdir
    repo="$(preview_path "$REMOTE_REPO")"
    logdir="$(preview_path "$REMOTE_LOGDIR")"
    local envs="CUDA_VISIBLE_DEVICES=$(shquote "$GPUS") PYTHONUNBUFFERED=1 PYTHONPATH=\"$repo\""
    if [ "$TYPE" = "fm" ] && [ "$HF_OFFLINE" = "1" ]; then
        envs="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $envs"
    fi
    printf 'nohup env %s %s/bin/python %s/policy_server/%s_server.py %s > %s/sim_eval_%s_%s_<UTC>.log 2>&1 </dev/null &' \
        "$envs" "$(preview_path "$REMOTE_VENV")" "$repo" "$TYPE" "$(quoted_server_args)" \
        "$logdir" "$TYPE" "$PORT"
}

tunnel_cmdline() {
    printf 'ssh -N -T %s -o ExitOnForwardFailure=yes -L 127.0.0.1:%s:127.0.0.1:%s %s' \
        "${SSH_OPTS[*]}" "$LOCAL_PORT" "$PORT" "$REMOTE_HOST"
}

local_cmdline() {
    local out="$SCRIPTS_DIR/run_${TYPE}_server.sh"
    local a
    while IFS= read -r a; do out+=" $(shquote "$a")"; done < <(server_args)
    printf '%s' "$out"
}

# ---------------------------------------------------------------------------
# MODE: probe
# ---------------------------------------------------------------------------
if [ "$MODE" = "probe" ]; then
    echo "probing tcp://127.0.0.1:$LOCAL_PORT (RESET, up to ${WAIT_S}s)"
    probe_endpoint "$LOCAL_PORT" "$WAIT_S" 2.0
    exit 0
fi

# ---------------------------------------------------------------------------
# MODE: check -- print, touch nothing.  No ssh, no rsync, no mkdir.
# ---------------------------------------------------------------------------
if [ "$MODE" = "check" ]; then
    echo "=== serve_policy.sh --check (nothing is started) ==="
    echo "type          : $TYPE"
    echo "checkpoint    : ${CHECKPOINT:-<unset>}"
    echo "port          : $PORT  (laptop side $LOCAL_PORT)"
    echo "n_action_steps: $NSTEPS"
    echo "device        : $DEVICE"
    [ "$TYPE" = "fm" ] && echo "task          : $TASK"
    if [ -n "$REMOTE_HOST" ]; then
        echo "remote        : $REMOTE_HOST  GPU(s) $GPUS"
        echo "remote venv   : $REMOTE_VENV"
        echo "remote repo   : $REMOTE_REPO"
        echo "remote logs   : $REMOTE_LOGDIR (pid file $REMOTE_PIDFILE)"
        echo
        [ "$DO_SYNC" = "1" ] && { echo "1) sync the server code:"; echo "   $(rsync_cmdline)"; }
        echo "2) start on $REMOTE_HOST (detached, bound to 127.0.0.1, server-owned):"
        echo "   $(remote_launch_preview)"
        echo "3) tunnel (owned by this terminal):"
        echo "   $(tunnel_cmdline)"
        echo "4) wait for a RESET round-trip on 127.0.0.1:$LOCAL_PORT (up to ${WAIT_S}s)"
        echo "5) stop later:"
        echo "   $0 --type $TYPE --remote $REMOTE_HOST --port $PORT --stop"
    else
        echo "local venv    : $ACT_VENV"
        echo
        echo "1) run in the foreground (Ctrl-C stops it):"
        echo "   $(local_cmdline)"
    fi
    echo
    echo "then: $REPO/.venv/bin/python -m sim_collect.eval.run_eval \\"
    echo "        --policy zmq://127.0.0.1:$LOCAL_PORT --task $(shquote "$TASK") --seeds 0-19 \\"
    echo "        --out sim_collect/eval/runs/<name>"
    exit 0
fi

# ---------------------------------------------------------------------------
# MODE: stop -- tunnel (local pid file) + remote server (remote pid file).
# ---------------------------------------------------------------------------
if [ "$MODE" = "stop" ]; then
    stopped_any=0
    if [ -f "$TUNNEL_STATE" ]; then
        tpid="$(sed -n 's/^pid=//p' "$TUNNEL_STATE" | head -n1 || true)"
        if [ -n "$tpid" ] && kill -0 "$tpid" 2>/dev/null; then
            tcmd="$(tr '\0' ' ' < "/proc/$tpid/cmdline" 2>/dev/null || true)"
            if [[ "$tcmd" == *"127.0.0.1:$LOCAL_PORT:127.0.0.1:$PORT"* ]]; then
                kill "$tpid" 2>/dev/null || true
                echo "stopped tunnel PID $tpid"
                stopped_any=1
            else
                echo "REFUSING to kill PID $tpid: it is not our tunnel (cmdline: $tcmd)" >&2
            fi
        fi
        rm -f "$TUNNEL_STATE"
    else
        echo "no local tunnel state at $TUNNEL_STATE"
    fi

    if [ -n "$REMOTE_HOST" ]; then
        [ -n "$TYPE" ] || die "--stop on a remote needs --type (it names the pid file)"
        remote_args="bash -s -- $(shquote "$REMOTE_PIDFILE") $(shquote "$PORT") $(shquote "$TYPE")"
        ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$remote_args" <<'REMOTE_STOP'
set -euo pipefail
PIDFILE="$1"; PORT="$2"; TYPE="$3"
PIDFILE="${PIDFILE/#\~/$HOME}"
if [ ! -f "$PIDFILE" ]; then
    echo "no pid file at $PIDFILE on $(hostname); nothing was stopped."
    exit 0
fi
pid="$(sed -n 's/^pid=//p' "$PIDFILE" | head -n1 || true)"
if [ -z "$pid" ]; then
    echo "pid file $PIDFILE has no pid= line; refusing to guess. Left in place." >&2
    exit 3
fi
if ! kill -0 "$pid" 2>/dev/null; then
    echo "pid $pid from $PIDFILE is already gone; removing the stale pid file."
    rm -f "$PIDFILE"
    exit 0
fi
# Identity guard: this box runs other people's jobs. A pid number alone is never
# enough -- the cmdline must be OUR server on OUR port before anything is signalled.
cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
case "$cmdline" in
    *"${TYPE}_server.py"*"--port $PORT"*) ;;
    *)
        echo "REFUSING: pid $pid is not the ${TYPE} server on port $PORT." >&2
        echo "          cmdline: $cmdline" >&2
        echo "          Nothing was signalled; $PIDFILE left in place." >&2
        exit 3
        ;;
esac
kill -TERM "$pid" 2>/dev/null || true
for _ in $(seq 1 40); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
done
if kill -0 "$pid" 2>/dev/null; then
    echo "pid $pid ignored TERM for 20s; sending KILL." >&2
    kill -KILL "$pid" 2>/dev/null || true
fi
rm -f "$PIDFILE"
echo "SIM_EVAL_STOPPED=$pid"
REMOTE_STOP
        stopped_any=1
    fi
    [ "$stopped_any" = "1" ] || echo "nothing to stop."
    exit 0
fi

# ---------------------------------------------------------------------------
# MODE: sync only
# ---------------------------------------------------------------------------
if [ "$MODE" = "sync" ]; then
    [ -n "$REMOTE_HOST" ] || die "--sync needs --remote <host>"
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" true || die "ssh $REMOTE_HOST failed"
    # NOT shquote'd: a single-quoted ~ would create a directory literally named
    # "~" in the remote $HOME.  preview_path turns it into $HOME, which the
    # remote shell expands inside the double quotes.
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p \"$(preview_path "$REMOTE_REPO")/policy_server\""
    echo "+ $(rsync_cmdline)"
    rsync -az --delete --exclude '__pycache__/' \
        -e "ssh ${SSH_OPTS[*]}" \
        "$POLICY_SERVER_SRC/" "$REMOTE_HOST:$REMOTE_REPO/policy_server/"
    echo "synced policy_server/ -> $REMOTE_HOST:$REMOTE_REPO/policy_server/"
    exit 0
fi

# ---------------------------------------------------------------------------
# MODE: start
# ---------------------------------------------------------------------------
[ -n "$CHECKPOINT" ] || die "--checkpoint <pretrained_model dir> is required"

if [ -z "$REMOTE_HOST" ]; then
    # ---------------- local (laptop3 CPU) -----------------------------------
    [ -x "$ACT_VENV/bin/python" ] || \
        die "local torch venv missing: $ACT_VENV/bin/python (see the ACT deploy doc)"
    [ -x "$SCRIPTS_DIR/run_${TYPE}_server.sh" ] || \
        die "missing $SCRIPTS_DIR/run_${TYPE}_server.sh"
    [ -d "$CHECKPOINT" ] || die "--checkpoint is not a directory: $CHECKPOINT"
    if [ "$DEVICE" = "cuda" ]; then
        echo "WARNING: --device cuda locally. laptop3 has no usable torch GPU for this" >&2
        echo "         stack; ${TYPE}_server.py refuses to fall back and will exit 3." >&2
    fi
    cat >&2 <<BANNER
============================================================
LOCAL POLICY SERVER -- CPU. THIS IS SLOW.
  A forward pass here is hundreds of ms to seconds, not the
  ~20 ms kanu gives you. The eval is lockstep, so a slow
  server only makes the run longer -- EXCEPT that run_eval's
  ZMQ timeout (real-deploy default 0.6 s) turns a slow reply
  into a FAULT, i.e. a failed episode. Raise run_eval's
  timeout before trusting a local-CPU SR number, or serve the
  checkpoint on kanu instead (--remote kanu).
  RAM: a lerobot checkpoint + torch is GBs. Check 'free -g'.
============================================================
BANNER
    echo "+ $(local_cmdline)"
    export ACT_VENV="$ACT_VENV" FM_VENV="$ACT_VENV" DIFFUSION_VENV="$ACT_VENV"
    mapfile -t _args < <(server_args)
    exec "$SCRIPTS_DIR/run_${TYPE}_server.sh" "${_args[@]}"
fi

# ---------------- remote (kanu) ---------------------------------------------
# Checked BEFORE the remote is touched: the probe and the local-port check both
# need it, and failing after the server is up would leave a process behind.
[ -x "$VENV_PY" ] || die "probe interpreter missing: $VENV_PY (the eval venv)"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" true \
    || die "ssh $REMOTE_HOST failed (BatchMode: is the key/agent set up?)"

if [ "$DO_SYNC" = "1" ]; then
    # NOT shquote'd: a single-quoted ~ would create a directory literally named
    # "~" in the remote $HOME.  preview_path turns it into $HOME, which the
    # remote shell expands inside the double quotes.
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p \"$(preview_path "$REMOTE_REPO")/policy_server\""
    echo "+ $(rsync_cmdline)"
    rsync -az --delete --exclude '__pycache__/' \
        -e "ssh ${SSH_OPTS[*]}" \
        "$POLICY_SERVER_SRC/" "$REMOTE_HOST:$REMOTE_REPO/policy_server/"
fi

mkdir -p "$STATE_DIR"

echo "starting $TYPE server on $REMOTE_HOST (GPU $GPUS, bind 127.0.0.1:$PORT)"
remote_args="bash -s --"
for a in "$TYPE" "$REMOTE_VENV" "$REMOTE_REPO" "$CHECKPOINT" "$PORT" "$DEVICE" \
         "$GPUS" "$REMOTE_LOGDIR" "$ALLOW_BUSY_GPU" "$HF_OFFLINE"; do
    remote_args+=" $(shquote "$a")"
done
while IFS= read -r a; do remote_args+=" $(shquote "$a")"; done < <(server_args)

set +e
REMOTE_OUT="$(ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$remote_args" <<'REMOTE_START'
set -euo pipefail
TYPE="$1"; VENV="$2"; SRV_ROOT="$3"; CKPT="$4"; PORT="$5"; DEVICE="$6"
GPUS="$7"; LOGDIR="$8"; ALLOW_BUSY="$9"; HF_OFFLINE="${10}"
shift 10
SERVER_ARGS=("$@")

expand() { printf '%s' "${1/#\~/$HOME}"; }
VENV="$(expand "$VENV")"; SRV_ROOT="$(expand "$SRV_ROOT")"
CKPT="$(expand "$CKPT")"; LOGDIR="$(expand "$LOGDIR")"

PY="$VENV/bin/python"
SERVER_PY="$SRV_ROOT/policy_server/${TYPE}_server.py"

[ -x "$PY" ] || { echo "ERROR: remote venv python not found: $PY" >&2; exit 4; }
[ -f "$SERVER_PY" ] || {
    echo "ERROR: $SERVER_PY is not on $(hostname)." >&2
    echo "       Re-run with --sync to rsync policy_server/ there." >&2
    exit 4
}
[ -d "$CKPT" ] || { echo "ERROR: checkpoint dir not found on $(hostname): $CKPT" >&2; exit 4; }
[ -f "$CKPT/config.json" ] && [ -f "$CKPT/model.safetensors" ] || \
    echo "WARNING: $CKPT lacks config.json and/or model.safetensors -- is it really a pretrained_model dir?" >&2

# --- python deps: report, never install -------------------------------------
NEED=(zmq torch lerobot numpy)
[ "$TYPE" = "fm" ] && NEED+=(transformers)
MISSING="$("$PY" -c '
import importlib, sys
out = []
for m in sys.argv[1:]:
    try:
        importlib.import_module(m)
    except Exception:
        out.append(m)
print(" ".join(out))' "${NEED[@]}")"
if [ -n "$MISSING" ]; then
    PIPNAMES="${MISSING//zmq/pyzmq}"
    echo "ERROR: $(hostname):$VENV is missing: $MISSING" >&2
    echo "       This script never installs anything. Run exactly this yourself:" >&2
    echo "         $PY -m pip install $PIPNAMES" >&2
    exit 5
fi

# --- GPU courtesy check (shared box) ----------------------------------------
if [ "$DEVICE" = "cuda" ]; then
    for g in ${GPUS//,/ }; do
        used="$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "")"
        if [ -z "$used" ]; then
            echo "WARNING: could not read GPU $g with nvidia-smi; proceeding." >&2
        elif [ "$used" -gt 1000 ] 2>/dev/null; then
            if [ "$ALLOW_BUSY" = "1" ]; then
                echo "WARNING: GPU $g already has ${used} MiB in use; SIM_EVAL_ALLOW_BUSY_GPU=1 so proceeding." >&2
            else
                echo "ERROR: GPU $g on $(hostname) already has ${used} MiB in use -- it is someone else's job." >&2
                echo "       Pick a free one (--gpus N) after looking at:  ssh $(hostname) nvidia-smi" >&2
                echo "       or set SIM_EVAL_ALLOW_BUSY_GPU=1 if you know it is yours." >&2
                exit 6
            fi
        fi
    done
fi

# --- port + pid file ---------------------------------------------------------
if ! "$PY" -c '
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()' "$PORT"; then
    echo "ERROR: 127.0.0.1:$PORT is already bound on $(hostname). Pick another --port." >&2
    exit 7
fi

mkdir -p "$LOGDIR"
PIDFILE="$LOGDIR/sim_eval_${TYPE}_${PORT}.pid"
if [ -f "$PIDFILE" ]; then
    OLD="$(sed -n 's/^pid=//p' "$PIDFILE" | head -n1 || true)"
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "ERROR: a server this script started is still running (pid $OLD, $PIDFILE)." >&2
        echo "       Stop it with --stop before starting another." >&2
        exit 8
    fi
    rm -f "$PIDFILE"
fi

STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
LOG="$LOGDIR/sim_eval_${TYPE}_${PORT}_${STAMP}.log"

ENVS=(CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 PYTHONPATH="$SRV_ROOT")
if [ "$TYPE" = "fm" ] && [ "$HF_OFFLINE" = "1" ]; then
    # Same reason as run_fm_server.sh: multi_task_dit's CLIP encoders would call
    # from_pretrained against the hub at startup even though the checkpoint
    # already carries those weights.
    ENVS=(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${ENVS[@]}")
fi

cd "$SRV_ROOT"
nohup env "${ENVS[@]}" "$PY" "$SERVER_PY" "${SERVER_ARGS[@]}" \
    >"$LOG" 2>&1 </dev/null &
PID=$!
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: the server exited immediately. Log tail:" >&2
    tail -n 30 "$LOG" >&2 || true
    exit 9
fi
{
    echo "pid=$PID"
    echo "type=$TYPE"
    echo "port=$PORT"
    echo "log=$LOG"
    echo "checkpoint=$CKPT"
    echo "server_py=$SERVER_PY"
    echo "gpus=$GPUS"
    echo "device=$DEVICE"
    echo "started_utc=$STAMP"
} > "$PIDFILE"
echo "SIM_EVAL_PID=$PID"
echo "SIM_EVAL_LOG=$LOG"
echo "SIM_EVAL_PIDFILE=$PIDFILE"
REMOTE_START
)"
REMOTE_RC=$?
set -e
printf '%s\n' "$REMOTE_OUT"
[ "$REMOTE_RC" -eq 0 ] || die "remote launch failed (exit $REMOTE_RC); nothing is running on $REMOTE_HOST"

SERVER_PID="$(printf '%s\n' "$REMOTE_OUT" | sed -n 's/^SIM_EVAL_PID=//p' | tail -n1)"
SERVER_LOG="$(printf '%s\n' "$REMOTE_OUT" | sed -n 's/^SIM_EVAL_LOG=//p' | tail -n1)"
SERVER_PIDFILE="$(printf '%s\n' "$REMOTE_OUT" | sed -n 's/^SIM_EVAL_PIDFILE=//p' | tail -n1)"
[[ "$SERVER_PID" =~ ^[0-9]+$ ]] || die "remote launcher did not report a pid"

# The server is now SERVER-OWNED. Everything below may fail without touching it.
stop_hint="$0 --type $TYPE --remote $REMOTE_HOST --port $PORT --stop"

if ! port_is_free "$LOCAL_PORT"; then
    echo "local port $LOCAL_PORT is occupied; no process was stopped." >&2
    echo "The remote server (pid $SERVER_PID) is still running. Stop it with:" >&2
    echo "  $stop_hint" >&2
    exit 2
fi

TUNNEL_PID=""
cleanup_tunnel() {
    if [ -n "$TUNNEL_PID" ]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
        TUNNEL_PID=""
    fi
    rm -f "$TUNNEL_STATE"
}
trap cleanup_tunnel EXIT
trap 'exit 130' INT TERM HUP

echo "opening tunnel: 127.0.0.1:$LOCAL_PORT -> $REMOTE_HOST:127.0.0.1:$PORT"
ssh -N -T "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes \
    -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$PORT" "$REMOTE_HOST" &
TUNNEL_PID=$!
{
    echo "pid=$TUNNEL_PID"
    echo "host=$REMOTE_HOST"
    echo "local_port=$LOCAL_PORT"
    echo "remote_port=$PORT"
    echo "remote_pid=$SERVER_PID"
    echo "remote_pidfile=$SERVER_PIDFILE"
} > "$TUNNEL_STATE"

sleep 1
kill -0 "$TUNNEL_PID" 2>/dev/null || {
    echo "SSH tunnel died immediately; the remote server (pid $SERVER_PID) was NOT stopped." >&2
    echo "  $stop_hint" >&2
    exit 2
}

echo "waiting for the first RESET round-trip (model load can take minutes)..."
if ! probe_endpoint "$LOCAL_PORT" "$WAIT_S" 2.0; then
    echo "the server never answered a RESET within ${WAIT_S}s." >&2
    echo "It was NOT stopped -- look at the log, then stop it:" >&2
    echo "  ssh $REMOTE_HOST tail -n 50 $SERVER_LOG" >&2
    echo "  $stop_hint" >&2
    exit 1
fi

cat <<BANNER
============================================================
policy server ready
  type      : $TYPE   (n_action_steps $NSTEPS, device $DEVICE)
  host      : $REMOTE_HOST   CUDA_VISIBLE_DEVICES=$GPUS
  checkpoint: $CHECKPOINT
  remote pid: $SERVER_PID  (SERVER-OWNED -- Ctrl-C here does not stop it)
  remote log: $SERVER_LOG
  pid file  : $SERVER_PIDFILE
  tunnel    : PID $TUNNEL_PID   127.0.0.1:$LOCAL_PORT -> $REMOTE_HOST:127.0.0.1:$PORT
============================================================
Run the eval in another terminal:
  cd $REPO && .venv/bin/python -m sim_collect.eval.run_eval \\
      --policy zmq://127.0.0.1:$LOCAL_PORT --task $(shquote "$TASK") \\
      --seeds 0-19 --out sim_collect/eval/runs/<name>

Keep this terminal open. Ctrl-C closes ONLY the tunnel.
Stop the remote server explicitly:
  $stop_hint
============================================================
BANNER

set +e
wait "$TUNNEL_PID"
TUNNEL_RC=$?
set -e
TUNNEL_PID=""
rm -f "$TUNNEL_STATE"
if [ "$TUNNEL_RC" -ne 0 ]; then
    echo "SSH tunnel exited with status $TUNNEL_RC; remote pid $SERVER_PID is still running." >&2
fi
exit "$TUNNEL_RC"
