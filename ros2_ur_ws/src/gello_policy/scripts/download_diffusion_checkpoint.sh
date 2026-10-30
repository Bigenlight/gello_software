#!/usr/bin/env bash
# Download the trained Diffusion "put right banana in pot" checkpoint (JOINT model,
# and optionally the LeRobot dataset) from the Hugging Face Hub into a GITIGNORED
# local dir, then print the checkpoint path to pass to the Diffusion server as
# --checkpoint.
#
#   Diffusion checkpoint : Bigenlight/diffusion_banana_in_pot_joint
#   Dataset (opt)        : Bigenlight/banana_in_pot_lerobot_v3
#
# ############################################################################
# #  USAGE                                                                    #
# #                                                                          #
# #    # Checkpoint only (default):                                          #
# #    ./download_diffusion_checkpoint.sh                                    #
# #                                                                          #
# #    # Also pull the dataset:                                             #
# #    WITH_DATASET=1 ./download_diffusion_checkpoint.sh                    #
# #                                                                          #
# #    # Use it directly:                                                   #
# #    CKPT=$(./download_diffusion_checkpoint.sh | tail -1)                 #
# #    DIFFUSION_CHECKPOINT="$CKPT" ./run_diffusion_server.sh               #
# #                                                                          #
# #  ENV:                                                                    #
# #    ACT_VENV     py3.12 venv (for huggingface-cli)  (default: ros2_ur_ws/act_venv)#
# #    DOWNLOAD_DIR local-dir root (gitignored)  (default: <pkg>/checkpoints)#
# #    CKPT_REPO    HF repo id (default: Bigenlight/diffusion_banana_in_pot_joint)#
# #    DATASET_REPO HF dataset id (default: Bigenlight/banana_in_pot_lerobot_v3)
# #    WITH_DATASET 1 => also download the dataset                          #
# ############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WS_DIR="$(cd "$PKG_DIR/../.." && pwd)"   # ros2_ur_ws

# Default venv = ros2_ur_ws/act_venv (created per the deploy docs); falls back to
# PATH for huggingface-cli/hf if it isn't there. Override with ACT_VENV=/path.
ACT_VENV="${ACT_VENV:-$WS_DIR/act_venv}"
# Downloads go under the package's checkpoints/ dir, which .gitignore excludes.
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$PKG_DIR/checkpoints}"
CKPT_REPO="${CKPT_REPO:-Bigenlight/diffusion_banana_in_pot_joint}"
DATASET_REPO="${DATASET_REPO:-Bigenlight/banana_in_pot_lerobot_v3}"
WITH_DATASET="${WITH_DATASET:-0}"

# Prefer the venv's huggingface-cli / hf; fall back to whatever is on PATH.
HF_CLI=""
for cand in "$ACT_VENV/bin/huggingface-cli" "$ACT_VENV/bin/hf" huggingface-cli hf; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
        HF_CLI="$cand"
        break
    fi
done
if [ -z "$HF_CLI" ]; then
    echo "ERROR: huggingface-cli / hf not found (looked in ACT_VENV=$ACT_VENV and PATH)." >&2
    echo "       pip install -U 'huggingface_hub[cli]' in the venv, or set ACT_VENV." >&2
    exit 1
fi

CKPT_DIR="$DOWNLOAD_DIR/diffusion_banana_in_pot_joint"
mkdir -p "$CKPT_DIR"

echo "### Downloading Diffusion checkpoint '$CKPT_REPO' -> $CKPT_DIR" >&2
"$HF_CLI" download "$CKPT_REPO" --local-dir "$CKPT_DIR" >&2

if [ "$WITH_DATASET" = "1" ]; then
    DS_DIR="$DOWNLOAD_DIR/banana_in_pot_lerobot_v3"
    mkdir -p "$DS_DIR"
    echo "### Downloading dataset '$DATASET_REPO' -> $DS_DIR" >&2
    "$HF_CLI" download "$DATASET_REPO" --repo-type dataset --local-dir "$DS_DIR" >&2
fi

# HF MODEL LAYOUT: this repo puts config.json + model.safetensors + the
# policy_pre/postprocessor jsons at the download dir ROOT (no nested
# pretrained_model/ subdir), so --checkpoint should point at $CKPT_DIR directly.
# We still probe for a nested pretrained_model/ for robustness, then print whichever
# exists (path on stdout).
if [ -d "$CKPT_DIR/pretrained_model" ]; then
    CKPT_PATH="$CKPT_DIR/pretrained_model"
else
    CKPT_PATH="$CKPT_DIR"
fi

echo "### Checkpoint ready. Pass this to --checkpoint / DIFFUSION_CHECKPOINT:" >&2
echo "$CKPT_PATH"
