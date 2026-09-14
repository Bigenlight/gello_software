#!/usr/bin/env bash
# ACT training for the carrot_in_pot SIM joint-space dataset (state 7 / action 7).
#
# Mirrors /home/laptop3/youngwoong_ws/banana-in-pot-experiments/train_act.sh
# (+ train_act_valdiag.sh's --eval_steps): plain `lerobot-train`, on-the-fly
# 360x640 resize (no dataset re-encode), batch 8, 50k steps, seed 1000.
#
# Differences from the banana recipe and WHY:
#   * --dataset.eval_split=0.09 -> holds out the LAST ceil(22*0.09)=2 episodes
#     (20, 21). See _common.sh for the ceil arithmetic and the lerobot source it
#     was verified against.
#   * --eval_steps=2000 so the held-out loss is logged.
#   * --policy.input_features=... -> REQUIRED here (the banana dataset had no
#     extra observation key; this one has observation.sim.object_poses, which
#     lerobot would otherwise feed to the policy). Full rationale in _common.sh.
#   * GPU_ID is REQUIRED: kanu has 8 GPUs shared with other people.
#
# IMAGE CONTRACT AT DEPLOY TIME: the 360x640 resize lives in the DATASET
# transform, not in the policy config, so the policy's own preprocessor does NOT
# resize. The serving side must resize to [360,640] itself -- exactly what the
# real ACT deploy server does.
#
# Usage:
#   DRY_RUN=1 GPU_ID=0 ./train_act_carrot_sim.sh   # print the command, run nothing
#   GPU_ID=0 ./train_act_carrot_sim.sh             # foreground (prefer launch_detached.sh)
#   POLICY_DEVICE=cpu DRY_RUN=0 STEPS=1 BATCH_SIZE=2 NUM_WORKERS=0 ./train_act_carrot_sim.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

sim_require_gpu_id
JOB_NAME="${JOB_NAME:-$ACT_JOB_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$SIM_ROOT/outputs/$JOB_NAME}"
STEPS="${STEPS:-50000}"
# Aspect-preserving half of 720p. MUST stay identical to the banana recipe and to
# the ACT deploy/eval server, which resizes to the same shape at inference.
RESIZE_HW="${RESIZE_HW:-[360,640]}"
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
  --policy.type=act
  --policy.device="$POLICY_DEVICE"
  --policy.push_to_hub=false
  "--policy.input_features=$INPUT_FEATURES"
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
  echo "# job: $JOB_NAME (ACT, joint space, SIM-ONLY)"
  echo "# held-out episodes: $(sim_heldout_episodes)"
  echo "# input_features: $INPUT_FEATURES"
  echo "# CUDA_VISIBLE_DEVICES=${GPU_ID:-<cpu>}  SCRATCH_DIR=$SCRATCH_DIR"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

sim_export_caches
NVIDIA_LIBS="$(sim_nvidia_libs "$LEROBOT_VENV")" \
  || sim_fail "pip NVIDIA library directories are missing under $LEROBOT_VENV"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The ImageNet-pretrained ResNet18 backbone must come from TORCH_HOME, never the
# network (banana TROUBLESHOOTING §5).
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-}"

exec "${cmd[@]}"
