#!/usr/bin/env bash
# Diffusion Policy training for the carrot_in_pot SIM joint-space dataset
# (state 7 / action 7).
#
# Follows /home/laptop3/youngwoong_ws/banana-in-pot-experiments/train_diffusion_joint_valdiag.sh
# FLAG FOR FLAG. That reference is the joint-action-space diffusion recipe, which
# is exactly what this dataset is, so nothing about the architecture is invented
# here. What it carries and why each flag is kept:
#
#   --policy.drop_n_last_frames=31
#       The lerobot 0.6.1 DEFAULT is 7, which is stale: the comment next to it in
#       configuration_diffusion.py says it should be
#       horizon - n_action_steps - n_obs_steps + 1, and with the current defaults
#       (horizon 64, n_action_steps 32, n_obs_steps 2) that is 64-32-2+1 = 31.
#       The reference sets 31 explicitly; keeping the default would over-sample
#       copy-padded tails.
#   --policy.resize_shape=[360,640]
#       Diffusion's OWN image preprocessing (inside the policy), so a served
#       720x1280 frame is resized identically to training. crop_ratio stays 1.0,
#       which makes __post_init__ set crop_shape=None -> no random crop. Verified
#       in the dry run: config.json has resize_shape [360,640], crop_shape null.
#   --dataset.image_transforms ... Resize [360,640]
#       The same resize at the DATASET level, as in the reference. Redundant with
#       the line above by construction (resizing 360x640 to 360x640 is a no-op)
#       but it keeps the dataloader cost and the ACT recipe identical. Kept
#       because the instruction was flag-for-flag, not flag-minus-one.
#   --steps=100000   (the reference's number; not a guess)
#   --eval_steps=2000, --save_freq=10000, batch 8, seed 1000, num_workers 4.
#
# Differences from the banana reference and WHY:
#   * --dataset.eval_split=0.09 -> the LAST ceil(22*0.09)=2 episodes (20, 21).
#     The reference used 0.117 for its 51 episodes; the fraction is dataset size.
#   * --policy.input_features=... -> excludes observation.sim.object_poses, which
#     only MuJoCo can produce and the deploy server never sends. See _common.sh.
#   * GPU_ID is REQUIRED.
#
# NOTE on the held-out loss: diffusion's eval loss is STOCHASTIC (a fresh noise
# sample and timestep per forward). Read the smoothed trend, never single points.
# This warning is copied from the reference on purpose.
#
# Usage:
#   DRY_RUN=1 GPU_ID=0 ./train_diffusion_carrot_sim.sh
#   GPU_ID=0 ./train_diffusion_carrot_sim.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

sim_require_gpu_id
JOB_NAME="${JOB_NAME:-$DIFFUSION_JOB_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$SIM_ROOT/outputs/$JOB_NAME}"
STEPS="${STEPS:-100000}"
RESIZE_HW="${RESIZE_HW:-[360,640]}"
DROP_N_LAST_FRAMES="${DROP_N_LAST_FRAMES:-31}"
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
  --policy.type=diffusion
  --policy.device="$POLICY_DEVICE"
  --policy.push_to_hub=false
  "--policy.input_features=$INPUT_FEATURES"
  --policy.drop_n_last_frames="$DROP_N_LAST_FRAMES"
  "--policy.resize_shape=$RESIZE_HW"
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
  echo "# job: $JOB_NAME (Diffusion, joint space, SIM-ONLY)"
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
# Diffusion's vision backbone is the same ImageNet ResNet18 as ACT's: it must come
# from TORCH_HOME, never the network.
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-}"

exec "${cmd[@]}"
