#!/usr/bin/env bash
# Flow-Matching (lerobot multi_task_dit, objective=flow_matching) training for the
# carrot_in_pot EEF-delta dataset (state 16 / action 7).
#
# Mirrors /home/laptop3/youngwoong_ws/cube_flow_matching_training/train_flow_matching.sh.
# Differences from the cube recipe and WHY:
#   * --dataset.eval_split=0.111 -> the LAST ceil(54*0.111)=6 episodes (48..53),
#     the same ones eval_offline_eef.py defaults to.
#   * repo_id / dataset root / output dir point at the carrot EEF job tree.
#   * No --policy.input_features override: this dataset has NO depth feature
#     (preflight.sh asserts that), so lerobot derives cam1 + cam2 + observation.state
#     straight from meta/info.json.
#   * HF_HUB_OFFLINE is deliberately NOT set here, exactly like the cube recipe:
#     multi_task_dit pulls CLIP ViT-B/16 through transformers on first use. kanu's
#     HF cache already holds it (DESIGN §1), but forcing offline would turn a cache
#     miss into a hard crash instead of a download.
#
# Usage:
#   DRY_RUN=1 GPU_ID=1 ./train_fm_carrot_eef.sh
#   GPU_ID=1 ./train_fm_carrot_eef.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

GPU_ID="${GPU_ID:?set GPU_ID to the selected physical GPU index (pick an idle one with nvidia-smi)}"
JOB_NAME="${JOB_NAME:-$FM_JOB_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$CARROT_ROOT/outputs/$JOB_NAME}"
STEPS="${STEPS:-100000}"
# The DiT resizes/crops INTERNALLY to 224x224 -- no dataset-level image transform.
IMAGE_RESIZE_SHAPE="${IMAGE_RESIZE_SHAPE:-[224,224]}"
IMAGE_CROP_SHAPE="${IMAGE_CROP_SHAPE:-[224,224]}"
# Euler steps of the training-time ODE. Deploy/eval may lower this (fm_server.py
# and eval_offline_eef.py default to 10) without retraining.
NUM_INTEGRATION_STEPS="${NUM_INTEGRATION_STEPS:-100}"

TRAIN="$LEROBOT_VENV/bin/lerobot-train"
[[ -x "$TRAIN" ]] || carrot_fail "missing $TRAIN (LEROBOT_VENV=$LEROBOT_VENV)"
[[ -f "$DATASET_ROOT/meta/info.json" ]] || carrot_fail "invalid dataset root: $DATASET_ROOT"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || carrot_fail "GPU_ID must be a non-negative integer (got '$GPU_ID')"

NVIDIA_LIBS="$(carrot_nvidia_libs "$LEROBOT_VENV")" \
  || carrot_fail "pip NVIDIA library directories are missing under $LEROBOT_VENV"

cmd=(
  "$TRAIN"
  --dataset.repo_id="$REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.eval_split="$EVAL_SPLIT"
  --policy.type=multi_task_dit
  --policy.objective=flow_matching
  --policy.device=cuda
  --policy.push_to_hub=false
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
  echo "# held-out episodes: $(carrot_heldout_episodes)"
  echo "# CUDA_VISIBLE_DEVICES=$GPU_ID  SCRATCH_DIR=$SCRATCH_DIR"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

carrot_export_caches
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "${cmd[@]}"
