#!/usr/bin/env bash
# ACT training for the carrot_in_pot EEF-delta dataset (state 16 / action 7).
#
# Mirrors /home/laptop3/youngwoong_ws/banana-in-pot-experiments/train_act.sh and
# train_act_valdiag.sh: plain `lerobot-train`, on-the-fly 360x640 resize (no
# dataset re-encode), batch 8, seed 1000. Differences from the banana recipe and
# WHY:
#   * --dataset.eval_split=0.111  -> holds out the LAST ceil(54*0.111)=6 episodes
#     (48..53). eval_offline_eef.py defaults to exactly those, so the training-time
#     held-out loss and the offline open-loop metrics see the same unseen episodes.
#   * --eval_steps=2000 (from train_act_valdiag.sh) so the held-out loss is logged.
#   * GPU_ID is REQUIRED: kanu has 8 GPUs shared with other people. There is no
#     sane default, so refuse rather than grab GPU 0.
#   * No --policy.input_features override: this dataset has NO depth feature
#     (preflight.sh asserts that), so lerobot's automatic feature discovery from
#     meta/info.json already yields exactly cam1 + cam2 + observation.state.
#
# Usage:
#   DRY_RUN=1 GPU_ID=0 ./train_act_carrot_eef.sh     # print the command, run nothing
#   GPU_ID=0 ./train_act_carrot_eef.sh               # foreground (use launch_detached.sh instead)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

GPU_ID="${GPU_ID:?set GPU_ID to the selected physical GPU index (pick an idle one with nvidia-smi)}"
JOB_NAME="${JOB_NAME:-$ACT_JOB_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$CARROT_ROOT/outputs/$JOB_NAME}"
STEPS="${STEPS:-50000}"
# Aspect-preserving half of 720p. MUST stay identical to the banana recipe and to
# eval_offline_eef.py / act_server.py, which resize to the same shape at inference.
RESIZE_HW="${RESIZE_HW:-[360,640]}"

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
  --policy.type=act
  --policy.device=cuda
  --policy.push_to_hub=false
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.max_num_transforms=1
  "--dataset.image_transforms.tfs={\"resize\":{\"weight\":1.0,\"type\":\"Resize\",\"kwargs\":{\"size\":$RESIZE_HW}}}"
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
# The ImageNet-pretrained ResNet18 backbone must come from TORCH_HOME, never the
# network (banana TROUBLESHOOTING §5).
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "${cmd[@]}"
