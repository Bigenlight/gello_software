#!/usr/bin/env bash
# sim_collect launcher — starts the three processes of sim_collect/DESIGN.md §2:
#   [A] sim_main.py  (MuJoCo physics + GELLO leader + teleop controller; owns the viewer window)
#   [B] capture.py   (offscreen cameras + take recorder, real-recorder file format)
#   [C] gui.py       (tkinter operator GUI: ENGAGE/DISENGAGE, takes, previews)
# Ctrl-C tears all three down. Logs go to $LOG_DIR (default sim_collect/logs/<timestamp>/).
#
# Usage:  ./sim_collect/run_sim_collect.sh [--config sim_collect/configs/carrot_in_pot_sim.yaml]
#                                          [--fake-leader] [--control-mode eef|joint] [--root <take root>] [--depth]
# Env:    DISPLAY (default :0), SIM_COLLECT_OUTPUT_ROOT (take root), SIM_COLLECT_IPC (ipc|tcp)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
CONFIG="$REPO/sim_collect/configs/carrot_in_pot_sim.yaml"
SIM_ARGS=()
CAPTURE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --fake-leader) SIM_ARGS+=(--fake-leader); shift ;;
    --control-mode) SIM_ARGS+=(--control-mode "$2"); shift 2 ;;
    --root) CAPTURE_ARGS+=(--root "$2"); shift 2 ;;
    --depth) CAPTURE_ARGS+=(--depth); shift ;;          # depth is off by default
    --no-depth) CAPTURE_ARGS+=(--no-depth); shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$PY" ]] || { echo "missing $PY (see docs/sim/GELLO_PANDA_SIM_TELEOP.md §2 for the uv venv)" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "missing config $CONFIG" >&2; exit 1; }

export DISPLAY="${DISPLAY:-:0}"
export MUJOCO_GL=glfw            # the only backend that renders on this machine (no EGL/OSMESA)
export PYTHONPATH="$REPO:$REPO/ros2_ur_ws/src/ur_gello_bringup:$REPO/ros2_ur_ws/src/gello_recorder${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export SIM_COLLECT_OUTPUT_ROOT="${SIM_COLLECT_OUTPUT_ROOT:-$REPO/ros2_ur_ws/gello_logs/sim}"

LOG_DIR="${LOG_DIR:-$REPO/sim_collect/logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" "$SIM_COLLECT_OUTPUT_ROOT"

# Refuse to start on top of a running stack (the Dynamixel driver would KILL the other
# sim_main to take the GELLO port) or on top of orphaned render workers (they hold GL
# contexts and a state subscription). Anchored patterns: an unanchored "sim_collect.sim_main"
# matched any shell whose argv merely contained that text (reviewer's harness, this file's
# own tests) and refused to launch.
if pgrep -f "python(3(\.[0-9]+)?)? -m sim_collect\.(sim_main|capture|gui)( |$)" >/dev/null; then
  echo "a sim_collect process is already running:" >&2
  pgrep -af "python(3(\.[0-9]+)?)? -m sim_collect\.(sim_main|capture|gui)( |$)" >&2
  echo "Stop it first (kill -TERM <pid>; orphaned render workers: pkill -f 'sim_collect.capture')." >&2
  exit 1
fi

PIDS=()
SIM_PID=""; CAPTURE_PID=""; GUI_PID=""; TAIL_PID=""
wait_gone() {  # wait_gone <seconds> <pid...>
  local secs="$1"; shift
  local n=$(( secs * 4 ))
  for _ in $(seq 1 "$n"); do
    local alive=0; for p in "$@"; do [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && alive=1; done
    [[ $alive -eq 0 ]] && return 0; sleep 0.25
  done
  return 1
}
cleanup() {
  trap '' INT TERM
  echo; echo "[run_sim_collect] stopping…"
  [[ -n "$TAIL_PID" ]] && kill -TERM "$TAIL_PID" 2>/dev/null || true
  # 1) capture FIRST: it must finalise a take that is still recording (mp4 moov atom,
  #    HDF5 close). Give it up to 20 s, then escalate. Its render workers are children
  #    in the same process group; capture terminates them itself, and the group kill
  #    below catches any orphan.
  if [[ -n "$CAPTURE_PID" ]]; then
    kill -TERM "$CAPTURE_PID" 2>/dev/null || true
    wait_gone 20 "$CAPTURE_PID" || { echo "[run_sim_collect] capture did not exit in 20 s — killing (take may be truncated)"; kill -KILL "$CAPTURE_PID" 2>/dev/null || true; }
  fi
  # 2) then the sim (closes the GELLO port) and the GUI
  for p in "$SIM_PID" "$GUI_PID"; do [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null || true; done
  wait_gone 10 "$SIM_PID" "$GUI_PID" || for p in "$SIM_PID" "$GUI_PID"; do [[ -n "$p" ]] && kill -KILL "$p" 2>/dev/null || true; done
  # 3) anything left in our process group (orphaned render workers, resource_tracker)
  # Only python processes of ours (render workers, resource_tracker) — never this shell
  # or the pgrep/grep pipeline itself (a previous version matched "grep sim_collect" and
  # killed its own pipeline).
  local leftovers
  leftovers=$(pgrep -g "$PGID" -af "python[0-9.]* (-m sim_collect|-c from multiprocessing)" 2>/dev/null || true)
  if [[ -n "$leftovers" ]]; then
    echo "[run_sim_collect] killing leftovers:"; echo "$leftovers"
    echo "$leftovers" | awk '{print $1}' | xargs -r kill -KILL 2>/dev/null || true
  fi
  echo "[run_sim_collect] logs: $LOG_DIR"
}
trap cleanup EXIT INT TERM
PGID=$(ps -o pgid= -p $$ | tr -d ' ')

echo "[run_sim_collect] config=$CONFIG takes=$SIM_COLLECT_OUTPUT_ROOT logs=$LOG_DIR"
"$PY" -m sim_collect.sim_main --config "$CONFIG" "${SIM_ARGS[@]}" > "$LOG_DIR/sim_main.log" 2>&1 &
SIM_PID=$!; PIDS+=($SIM_PID)
# wait for the sim REP to answer before starting the consumers
"$PY" - <<'PYEOF'
import sys, time
from sim_collect import ipc
c = ipc.Client("sim_rep", timeout_ms=500)
if not ipc.wait_for(c, timeout_s=60.0):
    print("[run_sim_collect] sim_main did not come up within 60 s — see sim_main.log", file=sys.stderr); sys.exit(1)
print("[run_sim_collect] sim_main ready")
PYEOF
"$PY" -m sim_collect.capture --config "$CONFIG" "${CAPTURE_ARGS[@]}" > "$LOG_DIR/capture.log" 2>&1 &
CAPTURE_PID=$!; PIDS+=($CAPTURE_PID)
"$PY" -m sim_collect.gui > "$LOG_DIR/gui.log" 2>&1 &
GUI_PID=$!; PIDS+=($GUI_PID)

echo "[run_sim_collect] running (sim $SIM_PID, capture $CAPTURE_PID, gui $GUI_PID). Ctrl-C to stop. Tailing sim_main.log:"
tail -n 5 -F "$LOG_DIR/sim_main.log" &
TAIL_PID=$!
# exit when any main process dies
while true; do
  for p in "$SIM_PID" "$CAPTURE_PID" "$GUI_PID"; do
    if ! kill -0 "$p" 2>/dev/null; then echo "[run_sim_collect] process $p exited — shutting down"; exit 1; fi
  done
  sleep 1
done
