#!/usr/bin/env bash
# Detach ONE carrot_sim training job from this session.
#
#   GPU_ID=0 ./launch_detached.sh act
#   GPU_ID=0 ./launch_detached.sh diffusion
#   GPU_ID=1 ./launch_detached.sh fm
#
# To run several jobs one after another on ONE GPU, use run_queue.sh instead.
#
# WHY setsid+nohup and not `./train.sh &`: a backgrounded job stays in the
# launching session's process group and dies with the SSH session -- and an agent
# harness actively reaps children at turn boundaries.
#
# WHY the PID is re-discovered instead of taken from `$!`: setsid hands back the
# PID of a WRAPPER, not of lerobot-train, so `kill`/`nvidia-smi` matching against
# it silently does nothing. We match the job_name in the real process's argv and
# take the lowest PID (the parent of the dataloader workers). The console script
# is python-wrapped, so argv is `<venv>/bin/python <venv>/bin/lerobot-train ...`
# and the trainer path can be field $2 or $3 -- see sim_job_pid() in _common.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

JOB="${1:?usage: GPU_ID=N ./launch_detached.sh act|diffusion|fm}"
sim_resolve_job "$JOB" || sim_fail "JOB must be act, diffusion or fm (got '$JOB')"
TRAIN_SCRIPT="$ROOT_DIR/$SIM_TRAIN_SCRIPT"
JOB_NAME="${JOB_NAME_OVERRIDE:-$SIM_JOB_NAME}"

GPU_ID="${GPU_ID:?set GPU_ID to an idle physical GPU index (check nvidia-smi RIGHT BEFORE launching)}"
OUTPUT_DIR="${OUTPUT_DIR:-$SIM_ROOT/outputs/$JOB_NAME}"
RUNTIME_DIR="${RUNTIME_DIR:-$ROOT_DIR/runtime/$JOB_NAME}"
LOG_FILE="${LOG_FILE:-$RUNTIME_DIR/train.log}"
PID_FILE="${PID_FILE:-$RUNTIME_DIR/train.pid}"
PID_WAIT_S="${PID_WAIT_S:-90}"

mkdir -p "$RUNTIME_DIR" "$(dirname "$OUTPUT_DIR")"

if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  sim_fail "training is already running with PID $(cat "$PID_FILE") (see $PID_FILE)"
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: output directory already exists: $OUTPUT_DIR" >&2
  echo "       Choose a new OUTPUT_DIR, or remove the failed/obsolete run first." >&2
  echo "       (lerobot resumes only via --config_path=<ckpt>/train_config.json --resume=true.)" >&2
  exit 1
fi

# The gate runs in THIS session so its failure is visible; it must pass before we
# detach anything.
GPU_ID="$GPU_ID" "$ROOT_DIR/preflight.sh" "$JOB"

setsid nohup env \
  SIM_ROOT="$SIM_ROOT" \
  DATASET_ROOT="$DATASET_ROOT" \
  REPO_ID="$REPO_ID" \
  LEROBOT_VENV="$LEROBOT_VENV" \
  SCRATCH_DIR="$SCRATCH_DIR" \
  EVAL_SPLIT="$EVAL_SPLIT" \
  GPU_ID="$GPU_ID" \
  JOB_NAME="$JOB_NAME" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  "$TRAIN_SCRIPT" >"$LOG_FILE" 2>&1 </dev/null &
wrapper_pid=$!

real_pid=""
for _ in $(seq 1 "$PID_WAIT_S"); do
  real_pid="$(sim_job_pid "$JOB_NAME")"
  [[ -n "$real_pid" ]] && break
  if ! kill -0 "$wrapper_pid" 2>/dev/null && [[ -z "$real_pid" ]]; then
    echo "ERROR: training exited during startup; inspect $LOG_FILE" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 1
done

if [[ -z "$real_pid" ]]; then
  echo "ERROR: no lerobot-train process with --job_name=$JOB_NAME appeared within ${PID_WAIT_S}s." >&2
  echo "       Wrapper PID was $wrapper_pid. Inspect $LOG_FILE and clean up by hand." >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi

printf '%s\n' "$real_pid" >"$PID_FILE"

echo "Training launched"
echo "  job          : $JOB ($JOB_NAME)"
echo "  PID          : $real_pid   (real lerobot-train; wrapper was $wrapper_pid)"
echo "  physical GPU : $GPU_ID (visible as cuda:0 inside training)"
echo "  log          : $LOG_FILE"
echo "  pid file     : $PID_FILE"
echo "  output       : $OUTPUT_DIR"
echo "  held-out eps : $(sim_heldout_episodes)"
echo
echo "Monitor : tail -f $LOG_FILE"
echo "Stop    : kill \$(cat $PID_FILE)"
