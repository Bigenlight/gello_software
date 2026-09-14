#!/usr/bin/env bash
# Flow-Matching (lerobot multi_task_dit, objective=flow_matching) training for the
# carrot_in_pot SIM joint-space dataset (state 7 / action 7).
#
# Mirrors /home/laptop3/youngwoong_ws/cube_flow_matching_training/train_flow_matching.sh:
# 100k steps, the DiT's internal 224x224 resize+crop, 100 Euler integration steps,
# batch 8, seed 1000.
#
# Differences from the cube recipe and WHY:
#   * --dataset.eval_split=0.09 -> the LAST ceil(22*0.09)=2 episodes (20, 21).
#   * repo_id / dataset root / output dir point at the SIM job tree.
#   * --policy.input_features=... -> excludes observation.sim.object_poses. See
#     _common.sh for the full rationale.
#   * HF_HUB_OFFLINE is deliberately NOT set, exactly like the cube recipe:
#     multi_task_dit pulls CLIP ViT-B/16 through transformers on first use (both
#     the vision tower and the tokenizer, `openai/clip-vit-base-patch16`). kanu's
#     HF cache already holds it, but forcing offline would turn a cache miss into
#     a hard crash instead of a download.
#   * No dataset-level image transform: the DiT resizes/crops INTERNALLY.
#
# NUM_INTEGRATION_STEPS is a TRAINING-time ODE setting. Deploy/eval may lower it
# (the real FM server uses 10) without retraining.
#
# Usage:
#   DRY_RUN=1 GPU_ID=1 ./train_fm_carrot_sim.sh
#   GPU_ID=1 ./train_fm_carrot_sim.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

sim_require_gpu_id
JOB_NAME="${JOB_NAME:-$FM_JOB_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$SIM_ROOT/outputs/$JOB_NAME}"
STEPS="${STEPS:-100000}"
IMAGE_RESIZE_SHAPE="${IMAGE_RESIZE_SHAPE:-[224,224]}"
IMAGE_CROP_SHAPE="${IMAGE_CROP_SHAPE:-[224,224]}"
NUM_INTEGRATION_STEPS="${NUM_INTEGRATION_STEPS:-100}"
INPUT_FEATURES="$(sim_input_features_json)"

TRAIN="$LEROBOT_VENV/bin/lerobot-train"
[[ -x "$TRAIN" ]] || sim_fail "missing $TRAIN (LEROBOT_VENV=$LEROBOT_VENV)"
[[ -f "$DATASET_ROOT/meta/info.json" ]] || sim_fail "invalid dataset root: $DATASET_ROOT"
sim_assert_dataset_shapes

cmd=(
  "$TRAIN"
  --dataset.repo_id="$REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.eval_split="$EVAL_SPLIT"
  --policy.type=multi_task_dit
  --policy.objective=flow_matching
  --policy.device="$POLICY_DEVICE"
  --policy.push_to_hub=false
  "--policy.input_features=$INPUT_FEATURES"
  "--policy.image_resize_shape=$IMAGE_RESIZE_SHAPE"
  "--policy.image_crop_shape=$IMAGE_CROP_SHAPE"
  --policy.num_integration_steps="$NUM_INTEGRATION_STEPS"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --eval_steps="$EVAL_STEPS"
  --log_freq="$LOG_FREQ"
  --num_workers="$NUM_WORKERS"
  --seed="$SEED"
  --wandb.enable=false
  --job_name="$JOB_NAME"
  --output_dir="$OUTPUT_DIR"
)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "# job: $JOB_NAME (Flow-Matching / multi_task_dit, joint space, SIM-ONLY)"
  echo "# held-out episodes: $(sim_heldout_episodes)"
  echo "# input_features: $INPUT_FEATURES"
  echo "# task string (CLIP tokenizer input at serve time): $TASK_STRING"
  echo "# CUDA_VISIBLE_DEVICES=${GPU_ID:-<cpu>}  SCRATCH_DIR=$SCRATCH_DIR"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

sim_export_caches
NVIDIA_LIBS="$(sim_nvidia_libs "$LEROBOT_VENV")" \
  || sim_fail "pip NVIDIA library directories are missing under $LEROBOT_VENV"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-}"

exec "${cmd[@]}"
