#!/usr/bin/env bash
# Shared plumbing for the carrot_in_pot EEF-delta training bundle (A3).
#
# Sourced by train_*.sh / preflight.sh / launch_detached.sh / eval_checkpoints.sh.
# It only DEFINES things: no side effects other than variable assignment, so a
# DRY_RUN caller can source it without touching the GPU or the filesystem.
#
# WHY a shared file at all: the ACT and FM recipes differ only in the lerobot
# flags. Every path, guard and environment trick below is identical, and the
# cube/banana references drifted apart precisely because they were copies.

# ---------------------------------------------------------------------------
# Canonical defaults. Every one is overridable from the environment so the
# integrator can point a run at a different dataset/venv without editing files.
# ---------------------------------------------------------------------------
CARROT_ROOT_DEFAULT="$HOME/workspace/youngwoong/carrot_eef"

: "${CARROT_ROOT:=$CARROT_ROOT_DEFAULT}"
: "${DATASET_ROOT:=$CARROT_ROOT/dataset/carrot_in_pot_eef_lerobot}"
: "${REPO_ID:=Bigenlight/carrot_in_pot_eef_lerobot_v3}"
# REUSE the existing cube env. DESIGN §1: kanu is 98 % full and the env already
# has torch 2.11+cu128, lerobot 0.6.1 and the CLIP ViT-B/16 HF cache. Never
# create a second one.
: "${LEROBOT_VENV:=$HOME/workspace/youngwoong/cube_flow_matching/training/lr_env}"
: "${SCRATCH_DIR:=$CARROT_ROOT/.cache}"
: "${EVAL_SPLIT:=0.111}"        # ceil(54 * 0.111) = 6 held-out episodes -> 48..53
: "${TOTAL_EPISODES:=54}"
: "${NUM_WORKERS:=4}"
: "${SEED:=1000}"
: "${BATCH_SIZE:=8}"
: "${SAVE_FREQ:=10000}"
: "${EVAL_STEPS:=2000}"
: "${LOG_FREQ:=200}"
: "${TASK_STRING:=Put carrot in pot}"
: "${DRY_RUN:=0}"

ACT_JOB_NAME="act_carrot_eef"
FM_JOB_NAME="fm_carrot_eef"

carrot_fail() { echo "ERROR: $*" >&2; exit 1; }

# Held-out episode indices implied by EVAL_SPLIT, as lerobot computes them:
# the LAST ceil(total * eval_split) episodes.
carrot_heldout_episodes() {
  python3 - "$TOTAL_EPISODES" "$EVAL_SPLIT" <<'PY'
import math, sys
total = int(sys.argv[1]); frac = float(sys.argv[2])
n = math.ceil(total * frac)
print(",".join(str(i) for i in range(total - n, total)))
PY
}

# The pip-installed NVIDIA runtime libraries are not on the system loader path.
# torch AND torchcodec (which links CUDA NPP even for CPU decoding) need them.
# Echoes a colon-joined list; fails loudly rather than silently exporting "".
carrot_nvidia_libs() {
  local venv="$1" root libs
  root="$(echo "$venv"/lib/python*/site-packages/nvidia)"
  libs="$(find "$root" -maxdepth 2 -name lib -type d 2>/dev/null | sort | paste -sd: -)"
  [[ -n "$libs" ]] || return 1
  printf '%s' "$libs"
}

# Caches: kanu's $HOME is shared and nearly full, so keep every byte lerobot
# writes under our own job root (DESIGN: never write outside carrot_eef/).
carrot_export_caches() {
  mkdir -p "$SCRATCH_DIR/torch_home" "$SCRATCH_DIR/hf_lerobot_home"
  export TORCH_HOME="$SCRATCH_DIR/torch_home"
  export HF_LEROBOT_HOME="$SCRATCH_DIR/hf_lerobot_home"
}
