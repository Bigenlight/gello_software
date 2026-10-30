#!/usr/bin/env bash
# sim_collect launcher — starts the processes of sim_collect/DESIGN.md §2:
#   [A] sim_main.py  (MuJoCo physics + GELLO leader + teleop controller; owns the viewer window)
#   [B] capture.py   (offscreen cameras + take recorder, real-recorder file format)
#   [C] gui.py       (tkinter operator GUI: ENGAGE/DISENGAGE, takes, previews)
# Ctrl-C tears all three down. Logs go to $LOG_DIR (default sim_collect/logs/<timestamp>/).
#
# Usage:  ./sim_collect/run_sim_collect.sh [--config <yaml>] [--fake-leader]
#                                          [--control-mode eef|joint] [--root <take root>]
#                                          [--depth|--no-depth] [--headless|--no-viewer|--no-gui]
# Env:    SIM_COLLECT_PY, MUJOCO_GL, DISPLAY, SIM_COLLECT_OUTPUT_ROOT, SIM_COLLECT_IPC, LOG_DIR
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO/sim_collect/configs/carrot_in_pot_sim.yaml"
SIM_ARGS=()
CAPTURE_ARGS=()
NO_VIEWER=0
NO_GUI=0

usage() {
  sed -n '2,13p' "$0"
}

usage_error() {
  echo "$1" >&2
  echo "Try '$0 --help'." >&2
  exit 2
}

require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || usage_error "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) require_value "$@"; CONFIG="$2"; shift 2 ;;
    --fake-leader) SIM_ARGS+=(--fake-leader); shift ;;
    --control-mode)
      require_value "$@"
      [[ "$2" == "eef" || "$2" == "joint" ]] || usage_error "--control-mode must be eef or joint"
      SIM_ARGS+=(--control-mode "$2")
      shift 2
      ;;
    --root) require_value "$@"; SIM_COLLECT_OUTPUT_ROOT="$2"; shift 2 ;;
    --depth) CAPTURE_ARGS+=(--depth); shift ;;
    --no-depth) CAPTURE_ARGS+=(--no-depth); shift ;;
    --headless) NO_VIEWER=1; NO_GUI=1; shift ;;
    --no-viewer) NO_VIEWER=1; shift ;;
    --no-gui) NO_GUI=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage_error "unknown arg: $1" ;;
  esac
done

if [[ -n "${SIM_COLLECT_PY:-}" ]]; then
  PY="$SIM_COLLECT_PY"
  PY_SOURCE="SIM_COLLECT_PY"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "gello-sim" && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
  PY="$CONDA_PREFIX/bin/python"
  PY_SOURCE="active conda env gello-sim"
elif command -v conda >/dev/null 2>&1 && CONDA_PY="$(conda run -n gello-sim python -c 'import sys; print(sys.executable)' 2>/dev/null)" && [[ -x "$CONDA_PY" ]]; then
  PY="$CONDA_PY"
  PY_SOURCE="conda env gello-sim"
elif [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
  PY_SOURCE="legacy .venv"
else
  echo "missing Python: set SIM_COLLECT_PY or create conda env gello-sim (legacy fallback: $REPO/.venv/bin/python)" >&2
  exit 1
fi

if [[ "$PY" == */* ]]; then
  [[ -x "$PY" ]] || { echo "SIM_COLLECT_PY is not executable: $PY" >&2; exit 1; }
  if [[ "$PY" != /* ]]; then
    PY="$(cd "$(dirname "$PY")" && pwd -P)/$(basename "$PY")"
  fi
else
  PY="$(command -v "$PY" 2>/dev/null || true)"
  [[ -n "$PY" && -x "$PY" ]] || { echo "SIM_COLLECT_PY command not found" >&2; exit 1; }
fi

[[ -f "$CONFIG" ]] || { echo "missing config $CONFIG" >&2; exit 1; }
CONFIG="$(realpath "$CONFIG")"
if [[ "$NO_VIEWER" -eq 1 ]]; then
  SIM_ARGS+=(--no-viewer)
fi

if [[ "$NO_VIEWER" -eq 1 && "$NO_GUI" -eq 1 ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
else
  export DISPLAY="${DISPLAY:-:0}"
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
fi
export PYTHONPATH="$REPO:$REPO/ros2_ur_ws/src/ur_gello_bringup:$REPO/ros2_ur_ws/src/gello_recorder${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export SIM_COLLECT_OUTPUT_ROOT="${SIM_COLLECT_OUTPUT_ROOT:-$REPO/ros2_ur_ws/gello_logs/sim}"

LOG_DIR="${LOG_DIR:-$REPO/sim_collect/logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" "$SIM_COLLECT_OUTPUT_ROOT"
LOG_DIR="$(cd "$LOG_DIR" && pwd -P)"
SIM_COLLECT_OUTPUT_ROOT="$(cd "$SIM_COLLECT_OUTPUT_ROOT" && pwd -P)"
export SIM_COLLECT_OUTPUT_ROOT

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

SIM_PID=""; CAPTURE_PID=""; GUI_PID=""; TAIL_PID=""
CLEANED_UP=0
command -v setsid >/dev/null 2>&1 || { echo "missing required command: setsid" >&2; exit 1; }
set +m
wait_gone() {  # wait_gone <seconds> <pid...>
  local secs="$1"; shift
  local n=$(( secs * 4 ))
  for _ in $(seq 1 "$n"); do
    local alive=0; for p in "$@"; do [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && alive=1; done
    [[ $alive -eq 0 ]] && return 0; sleep 0.25
  done
  return 1
}
kill_owned_group() {
  local leader_pid="$1"
  [[ -n "$leader_pid" ]] || return 0
  kill -KILL -- "-$leader_pid" 2>/dev/null || true
}
cleanup() {
  [[ "$CLEANED_UP" -eq 0 ]] || return 0
  CLEANED_UP=1
  trap '' INT TERM
  echo; echo "[run_sim_collect] stopping…"
  [[ -n "$TAIL_PID" ]] && kill -TERM "$TAIL_PID" 2>/dev/null || true
  [[ -n "$TAIL_PID" ]] && wait "$TAIL_PID" 2>/dev/null || true
  if [[ -n "$CAPTURE_PID" ]]; then
    kill -TERM "$CAPTURE_PID" 2>/dev/null || true
    wait_gone 20 "$CAPTURE_PID" || echo "[run_sim_collect] capture did not exit in 20 s — killing its process group (take may be truncated)"
    kill_owned_group "$CAPTURE_PID"
    wait "$CAPTURE_PID" 2>/dev/null || true
  fi
  for p in "$SIM_PID" "$GUI_PID"; do [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null || true; done
  wait_gone 10 "$SIM_PID" "$GUI_PID" || true
  for p in "$SIM_PID" "$GUI_PID"; do kill_owned_group "$p"; done
  for p in "$SIM_PID" "$GUI_PID"; do [[ -n "$p" ]] && wait "$p" 2>/dev/null || true; done
  echo "[run_sim_collect] logs: $LOG_DIR"
}
handle_signal() {
  exit "$1"
}
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

echo "[run_sim_collect] python=$PY ($PY_SOURCE) backend=$MUJOCO_GL config=$CONFIG takes=$SIM_COLLECT_OUTPUT_ROOT logs=$LOG_DIR"
setsid "$PY" -m sim_collect.sim_main --config "$CONFIG" "${SIM_ARGS[@]}" > "$LOG_DIR/sim_main.log" 2>&1 &
SIM_PID=$!
# wait for the sim REP to answer before starting the consumers
"$PY" - <<'PYEOF'
import sys, time
from sim_collect import ipc
c = ipc.Client("sim_rep", timeout_ms=500)
if not ipc.wait_for(c, timeout_s=60.0):
    print("[run_sim_collect] sim_main did not come up within 60 s — see sim_main.log", file=sys.stderr); sys.exit(1)
print("[run_sim_collect] sim_main ready")
PYEOF
setsid "$PY" -m sim_collect.capture --config "$CONFIG" "${CAPTURE_ARGS[@]}" > "$LOG_DIR/capture.log" 2>&1 &
CAPTURE_PID=$!
if [[ "$NO_GUI" -eq 0 ]]; then
  setsid "$PY" -m sim_collect.gui > "$LOG_DIR/gui.log" 2>&1 &
  GUI_PID=$!
fi

if [[ -n "$GUI_PID" ]]; then
  echo "[run_sim_collect] running (sim $SIM_PID, capture $CAPTURE_PID, gui $GUI_PID). Ctrl-C to stop. Tailing sim_main.log:"
else
  echo "[run_sim_collect] running headless (sim $SIM_PID, capture $CAPTURE_PID, no GUI). Ctrl-C to stop. Tailing sim_main.log:"
fi
tail -n 5 -F "$LOG_DIR/sim_main.log" &
TAIL_PID=$!
# exit when any main process dies
while true; do
  for p in "$SIM_PID" "$CAPTURE_PID" "$GUI_PID"; do
    [[ -n "$p" ]] || continue
    if ! kill -0 "$p" 2>/dev/null; then echo "[run_sim_collect] process $p exited — shutting down"; exit 1; fi
  done
  sleep 1
done
