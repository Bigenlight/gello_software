#!/usr/bin/env bash
# Run several carrot_sim trainings SEQUENTIALLY on ONE GPU, detached.
#
#   ./run_queue.sh 0 act diffusion      # ACT first, then Diffusion, same GPU
#   ./run_queue.sh 1 fm
#   ./run_queue.sh --status 0           # what the chain on GPU 0 is doing
#   ./run_queue.sh --stop 0             # stop the chain and its current trainer
#
# WHY this exists: kanu's GPUs are shared and our budget is 2 concurrent
# trainings, but we have 3 policies to train. Chaining act -> diffusion on one
# GPU costs one slot instead of two, and the operator does not have to come back
# at 2 a.m. to start the second one.
#
# Contract:
#   * one log and one pid file per job, under $ROOT_DIR/runtime/<job_name>/
#     (on kanu that is ~/workspace/youngwoong/carrot_sim/training/runtime/<job>/);
#   * the chain STOPS at the first failure -- a later job never starts on top of
#     a broken earlier one, and the failure stays visible in the status file;
#   * preflight.sh runs immediately BEFORE each job, not once up front: by the
#     time job 2 starts, hours have passed and the GPU may have been taken;
#   * the whole chain is setsid+nohup'd, so it survives the SSH session.
#
# It does NOT resume, retry, or clean up after a failed job. That is deliberate:
# a failed 100k-step run needs a human to look at the log.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

usage() {
  cat >&2 <<EOS
usage: $0 GPU_ID JOB [JOB ...]      # JOB in {act, diffusion, fm}
       $0 --status GPU_ID
       $0 --stop GPU_ID
EOS
  exit 2
}

QUEUE_DIR="${QUEUE_DIR:-$ROOT_DIR/runtime}"

# ---------------------------------------------------------------------------
# --status / --stop
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--status" || "${1:-}" == "--stop" ]]; then
  mode="$1"; gpu="${2:-}"
  [[ "$gpu" =~ ^[0-9]+$ ]] || usage
  chain_pid_file="$QUEUE_DIR/queue_gpu${gpu}.pid"
  chain_log="$QUEUE_DIR/queue_gpu${gpu}.log"
  if [[ "$mode" == "--status" ]]; then
    echo "chain log : $chain_log"
    if [[ -s "$chain_pid_file" ]] && kill -0 "$(cat "$chain_pid_file")" 2>/dev/null; then
      echo "chain pid : $(cat "$chain_pid_file") (alive)"
    else
      echo "chain pid : $( [[ -s "$chain_pid_file" ]] && cat "$chain_pid_file" || echo '-') (not running)"
    fi
    for d in "$QUEUE_DIR"/*/; do
      [[ -f "$d/status" ]] || continue
      printf '  %-24s %s\n' "$(basename "$d")" "$(cat "$d/status")"
    done
    [[ -f "$chain_log" ]] && { echo "--- last 20 lines ---"; tail -n 20 "$chain_log"; }
    exit 0
  fi
  # --stop: kill the chain supervisor FIRST so it cannot start the next job,
  # then the trainer it is currently running.
  if [[ -s "$chain_pid_file" ]] && kill -0 "$(cat "$chain_pid_file")" 2>/dev/null; then
    echo "stopping chain supervisor $(cat "$chain_pid_file")"
    kill "$(cat "$chain_pid_file")" 2>/dev/null || true
  fi
  for name in "${OUR_JOB_NAMES[@]}"; do
    case "$name" in *_carrot_sim) ;; *) continue ;; esac
    pid="$(sim_job_pid "$name")"
    [[ -n "$pid" ]] || continue
    echo "stopping $name (pid $pid)"
    kill "$pid" 2>/dev/null || true
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# internal: the chain body, re-exec'd under setsid
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--_chain" ]]; then
  shift
  GPU_ID="$1"; shift
  chain_pid_file="$QUEUE_DIR/queue_gpu${GPU_ID}.pid"
  printf '%s\n' "$$" >"$chain_pid_file"
  trap 'rm -f "$chain_pid_file"' EXIT

  for job in "$@"; do
    sim_resolve_job "$job" || { echo "CHAIN ABORT: unknown job '$job'"; exit 2; }
    runtime_dir="$QUEUE_DIR/$SIM_JOB_NAME"
    mkdir -p "$runtime_dir"
    log_file="$runtime_dir/train.log"
    pid_file="$runtime_dir/train.pid"
    status_file="$runtime_dir/status"
    output_dir="$SIM_ROOT/outputs/$SIM_JOB_NAME"

    echo "[$(date -Is)] === $job ($SIM_JOB_NAME) on GPU $GPU_ID ==="
    if [[ -e "$output_dir" ]]; then
      echo "CHAIN ABORT: output dir already exists: $output_dir"
      echo "ABORTED output_dir_exists $(date -Is)" >"$status_file"
      exit 1
    fi

    echo "PREFLIGHT $(date -Is)" >"$status_file"
    if ! GPU_ID="$GPU_ID" "$ROOT_DIR/preflight.sh" "$job"; then
      echo "CHAIN ABORT: preflight failed for $job"
      echo "ABORTED preflight $(date -Is)" >"$status_file"
      exit 1
    fi

    echo "RUNNING $(date -Is)" >"$status_file"
    # The trainer runs in the FOREGROUND of this chain process: that is how the
    # chain knows when to start the next job, and `wait`ing on it is what makes
    # the sequence sequential.
    env SIM_ROOT="$SIM_ROOT" DATASET_ROOT="$DATASET_ROOT" REPO_ID="$REPO_ID" \
        LEROBOT_VENV="$LEROBOT_VENV" SCRATCH_DIR="$SCRATCH_DIR" \
        EVAL_SPLIT="$EVAL_SPLIT" GPU_ID="$GPU_ID" JOB_NAME="$SIM_JOB_NAME" \
        OUTPUT_DIR="$output_dir" \
        "$ROOT_DIR/$SIM_TRAIN_SCRIPT" >"$log_file" 2>&1 </dev/null &
    train_wrapper=$!

    # Record the REAL lerobot-train pid (not this wrapper) so an operator can
    # kill exactly the trainer. Same discovery rule as launch_detached.sh.
    real_pid=""
    for _ in $(seq 1 90); do
      real_pid="$(sim_job_pid "$SIM_JOB_NAME")"
      [[ -n "$real_pid" ]] && break
      kill -0 "$train_wrapper" 2>/dev/null || break
      sleep 1
    done
    if [[ -n "$real_pid" ]]; then
      printf '%s\n' "$real_pid" >"$pid_file"
      echo "[$(date -Is)] $SIM_JOB_NAME lerobot-train pid $real_pid (log $log_file)"
    else
      echo "[$(date -Is)] WARNING: could not discover the lerobot-train pid for $SIM_JOB_NAME"
    fi

    rc=0
    wait "$train_wrapper" || rc=$?
    rm -f "$pid_file"
    if (( rc != 0 )); then
      echo "FAILED rc=$rc $(date -Is)" >"$status_file"
      echo "[$(date -Is)] === $job FAILED with rc=$rc; STOPPING THE CHAIN ==="
      tail -n 40 "$log_file" || true
      exit "$rc"
    fi
    echo "DONE $(date -Is)" >"$status_file"
    echo "[$(date -Is)] === $job finished OK ==="
    sleep "${QUEUE_GAP_S:-10}"   # let the GPU free itself before the next preflight
  done
  echo "[$(date -Is)] === chain complete on GPU $GPU_ID ==="
  exit 0
fi

# ---------------------------------------------------------------------------
# normal entry: validate, gate on the first job, detach the chain
# ---------------------------------------------------------------------------
(( $# >= 2 )) || usage
GPU_ID="$1"; shift
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || sim_fail "GPU_ID must be a non-negative integer (got '$GPU_ID')"

declare -A seen=()
for job in "$@"; do
  sim_resolve_job "$job" || sim_fail "JOB must be act, diffusion or fm (got '$job')"
  [[ -z "${seen[$job]:-}" ]] || sim_fail "job '$job' listed twice: a chain runs each job once"
  seen[$job]=1
  [[ -e "$SIM_ROOT/outputs/$SIM_JOB_NAME" ]] \
    && sim_fail "output dir already exists: $SIM_ROOT/outputs/$SIM_JOB_NAME (remove it or pick another OUTPUT_DIR)"
done

mkdir -p "$QUEUE_DIR"
chain_log="$QUEUE_DIR/queue_gpu${GPU_ID}.log"
chain_pid_file="$QUEUE_DIR/queue_gpu${GPU_ID}.pid"
if [[ -s "$chain_pid_file" ]] && kill -0 "$(cat "$chain_pid_file")" 2>/dev/null; then
  sim_fail "a chain is already running on GPU $GPU_ID (pid $(cat "$chain_pid_file"), log $chain_log)"
fi

# Gate on the FIRST job here, in the foreground, so a broken setup fails where
# the operator can see it. Each job is gated again inside the chain.
first="$1"
echo "### foreground preflight for the first job ($first) on GPU $GPU_ID"
GPU_ID="$GPU_ID" "$ROOT_DIR/preflight.sh" "$first"

setsid nohup env \
  SIM_ROOT="$SIM_ROOT" DATASET_ROOT="$DATASET_ROOT" REPO_ID="$REPO_ID" \
  LEROBOT_VENV="$LEROBOT_VENV" SCRATCH_DIR="$SCRATCH_DIR" EVAL_SPLIT="$EVAL_SPLIT" \
  QUEUE_DIR="$QUEUE_DIR" \
  "$ROOT_DIR/run_queue.sh" --_chain "$GPU_ID" "$@" >"$chain_log" 2>&1 </dev/null &
wrapper=$!

for _ in $(seq 1 30); do
  [[ -s "$chain_pid_file" ]] && break
  kill -0 "$wrapper" 2>/dev/null || break
  sleep 1
done

echo "Queue launched"
echo "  GPU        : $GPU_ID"
echo "  jobs       : $*  (sequential; the chain stops at the first failure)"
echo "  chain pid  : $( [[ -s "$chain_pid_file" ]] && cat "$chain_pid_file" || echo "unknown (wrapper $wrapper)")"
echo "  chain log  : $chain_log"
echo "  job logs   : $QUEUE_DIR/<job_name>/train.log"
echo "  held-out   : $(sim_heldout_episodes)"
echo
echo "Status : $ROOT_DIR/run_queue.sh --status $GPU_ID"
echo "Stop   : $ROOT_DIR/run_queue.sh --stop $GPU_ID"
