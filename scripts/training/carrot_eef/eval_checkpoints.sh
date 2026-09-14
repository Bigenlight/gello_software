#!/usr/bin/env bash
# Evaluate every complete checkpoint of a carrot_eef job, open-loop, on the
# held-out episodes, and write a trend CSV.
#
#   GPU_ID=2 ./eval_checkpoints.sh act
#   GPU_ID=2 ./eval_checkpoints.sh fm
#   DEVICE=cpu ./eval_checkpoints.sh fm        # no GPU needed, slower
#
# Run this on a DIFFERENT idle GPU than the training job (or on CPU): it refuses a
# GPU that already has a compute process, which includes our own trainer.
# Results: evaluations/<job>/<step>/{metrics.json,episodes.csv} + evaluations/<job>/summary.csv
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

JOB="${1:?usage: [GPU_ID=N|DEVICE=cpu] ./eval_checkpoints.sh act|fm}"
case "$JOB" in
  act) JOB_NAME="$ACT_JOB_NAME" ;;
  fm)  JOB_NAME="$FM_JOB_NAME"  ;;
  *)   carrot_fail "JOB must be act or fm (got '$JOB')" ;;
esac

DEVICE="${DEVICE:-cuda}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-$CARROT_ROOT/outputs/$JOB_NAME}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/evaluations/$JOB_NAME}"
EPISODES="${EPISODES:-$(carrot_heldout_episodes)}"
# 10 Euler steps == the deploy default in fm_server.py. No-op for ACT.
INTEGRATION_STEPS="${INTEGRATION_STEPS:-10}"
EVAL_SEED="${EVAL_SEED:-$SEED}"
# Deploy value (act_server.py default 30; fm_server.py 24). Empty = the
# checkpoint's trained value (ACT chunk_size 100 -> pessimistic, NOT the deploy
# configuration). Use a separate EVAL_ROOT per value: existing metrics.json are skipped.
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
PYTHON="$LEROBOT_VENV/bin/python"

[[ "$DEVICE" == "cuda" || "$DEVICE" == "cpu" ]] || carrot_fail "DEVICE must be cuda or cpu"
[[ -x "$PYTHON" ]] || carrot_fail "missing $PYTHON"
[[ -f "$DATASET_ROOT/meta/info.json" ]] || carrot_fail "invalid dataset root: $DATASET_ROOT"

if [[ "$DEVICE" == "cuda" ]]; then
  [[ "${GPU_ID:-}" =~ ^[0-9]+$ ]] || carrot_fail "set GPU_ID to an idle physical GPU index (or DEVICE=cpu)"
  command -v nvidia-smi >/dev/null || carrot_fail "nvidia-smi is unavailable"
  compute_apps="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true)"
  [[ -z "${compute_apps//[[:space:]]/}" ]] \
    || carrot_fail "GPU $GPU_ID has compute processes (use a different GPU than the trainer): $compute_apps"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# torchcodec links CUDA NPP even when both frames and policy run on the CPU.
NVIDIA_LIBS="$(carrot_nvidia_libs "$LEROBOT_VENV")" \
  || carrot_fail "pip NVIDIA library directories are missing under $LEROBOT_VENV"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
carrot_export_caches
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$EVAL_ROOT"
mapfile -t checkpoints < <(
  find "$TRAIN_OUTPUT/checkpoints" -mindepth 2 -maxdepth 2 -type d -name pretrained_model -print 2>/dev/null | sort
)
if (( ${#checkpoints[@]} == 0 )); then
  echo "No checkpoints below $TRAIN_OUTPUT/checkpoints yet."
  exit 0
fi

echo "job=$JOB_NAME device=$DEVICE episodes=$EPISODES integration_steps=$INTEGRATION_STEPS n_action_steps=${N_ACTION_STEPS:-trained}"
for checkpoint in "${checkpoints[@]}"; do
  step_name="$(basename "$(dirname "$checkpoint")")"
  # 'last' is a symlink to the newest numbered checkpoint: evaluating it would
  # duplicate a row under a second name.
  if [[ "$step_name" == "last" ]]; then
    echo "SKIP $step_name: alias of the newest checkpoint"
    continue
  fi
  complete=1
  for required in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    [[ -f "$checkpoint/$required" ]] || complete=0
  done
  if (( complete == 0 )); then
    echo "SKIP $step_name: checkpoint is still being written"
    continue
  fi
  out="$EVAL_ROOT/$step_name"
  if [[ -s "$out/metrics.json" ]]; then
    echo "SKIP $step_name: $out/metrics.json already exists"
    continue
  fi
  echo "EVALUATE $step_name"
  extra=()
  [[ -n "$N_ACTION_STEPS" ]] && extra+=(--n-action-steps "$N_ACTION_STEPS")
  "$PYTHON" "$ROOT_DIR/eval_offline_eef.py" \
    --checkpoint "$checkpoint" \
    --dataset-root "$DATASET_ROOT" \
    --repo-id "$REPO_ID" \
    --episodes "$EPISODES" \
    --task "$TASK_STRING" \
    --device "$DEVICE" \
    --num-integration-steps "$INTEGRATION_STEPS" \
    --seed "$EVAL_SEED" \
    "${extra[@]}" \
    --out "$out"
done

"$PYTHON" - "$EVAL_ROOT" <<'PY'
"""Collapse every metrics.json under the eval root into one trend CSV."""
import csv, json, re, sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = ("position_mae_m", "position_err_norm_m", "rotation_mae_rad",
           "rotation_axis_mae_rad", "gripper_mae", "gripper_accuracy", "overall_l1",
           "chunk10_pos_err_m", "chunk30_pos_err_m", "chunk30_gt_norm_m")
rows = []
for path in sorted(root.glob("*/metrics.json"),
                   key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.parent.name)]):
    data = json.loads(path.read_text())
    rows.append({"step": path.parent.name, "frames": data["aggregate"]["frames"],
                 **{m: data["aggregate"].get(m, float("nan")) for m in metrics}})
    base = data.get("baseline_zero")
    if base is not None and not any(r["step"] == "zero_baseline" for r in rows):
        rows.insert(0, {"step": "zero_baseline", "frames": base["frames"],
                        **{m: base.get(m, float("nan")) for m in metrics}})
if not rows:
    print("no metrics.json found; nothing to summarize")
    raise SystemExit(0)
out = root / "summary.csv"
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["step", "frames", *metrics])
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out} ({len(rows)} checkpoints)")
for row in rows:
    print(f"  {row['step']:>13}  posMAE={row['position_mae_m']:.6f} m  "
          f"rotMAE={row['rotation_mae_rad']:.6f} rad  gripAcc={row['gripper_accuracy']:.3f}  "
          f"chunk30={row['chunk30_pos_err_m']*1e3:.1f}/{row['chunk30_gt_norm_m']*1e3:.1f} mm")
PY
