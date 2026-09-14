#!/usr/bin/env bash
# Shared plumbing for the carrot_in_pot SIM joint-space training bundle (A9).
#
# Sourced by train_*.sh / preflight.sh / launch_detached.sh / run_queue.sh.
# It only DEFINES things: no side effects other than variable assignment, so a
# DRY_RUN caller can source it without touching the GPU or the filesystem.
#
# SIM-ONLY. This bundle trains on MuJoCo sim takes and NEVER mixes them with the
# real-robot corpus: separate repo_id, separate dataset root, separate job root,
# separate job names. Nothing here points at carrot_eef/ or at any real dataset.
#
# Action/state space is JOINT (not the EEF-delta space of the carrot_eef bundle):
#   observation.state (7) = ur_q1..ur_q6 + grip_pos
#   action            (7) = cmd1..cmd6   + grip_cmd
# shoulder_pan sits on the -pi branch (J1 ~ -3.3 rad), exactly like the real
# carrot data, so a policy trained here speaks the same joint convention.

# ---------------------------------------------------------------------------
# Canonical defaults. Every one is overridable from the environment so the
# integrator can point a run at a different dataset/venv without editing files.
# ---------------------------------------------------------------------------
SIM_ROOT_DEFAULT="$HOME/workspace/youngwoong/carrot_sim"

: "${SIM_ROOT:=$SIM_ROOT_DEFAULT}"
: "${DATASET_ROOT:=$SIM_ROOT/dataset/carrot_in_pot_sim_lerobot}"
: "${REPO_ID:=Bigenlight/carrot_in_pot_sim_lerobot_v3}"
: "${DATASET_REVISION:=v3.0}"
# REUSE the existing cube env, exactly like the carrot_eef bundle. kanu is ~98 %
# full and that env already has torch 2.11+cu128, lerobot 0.6.1, transformers
# 5.13.0 and the CLIP ViT-B/16 HF cache. NEVER create a second one.
: "${LEROBOT_VENV:=$HOME/workspace/youngwoong/cube_flow_matching/training/lr_env}"
: "${SCRATCH_DIR:=$SIM_ROOT/.cache}"
# ceil(22 * 0.09) = ceil(1.98) = 2 -> lerobot holds out the LAST 2 episodes (20, 21).
# Verified against lerobot 0.6.1 datasets/factory.py::make_train_eval_datasets:
#   n_eval = math.ceil(len(eps) * eval_split); train = eps[:-n_eval]; eval = eps[-n_eval:]
# The split is computed PER TASK. This dataset has exactly 1 task string
# ("Put carrot in pot") covering all 22 episodes, so the per-task group IS the
# whole dataset and the held-out set is literally [20, 21].
# Admissible window for "exactly 2": 1/22 < eval_split <= 2/22, i.e.
# (0.045454..., 0.090909...]. 0.09 sits inside it with float slack on both sides
# (22*0.09 = 1.9800000000000002). Measured: "Train/eval split: 20 train, 2 eval".
: "${EVAL_SPLIT:=0.09}"
: "${TOTAL_EPISODES:=22}"
: "${TOTAL_FRAMES:=10725}"
: "${FPS:=30}"
: "${NUM_WORKERS:=4}"
: "${SEED:=1000}"
: "${BATCH_SIZE:=8}"
: "${SAVE_FREQ:=10000}"
: "${EVAL_STEPS:=2000}"
: "${LOG_FREQ:=200}"
: "${TASK_STRING:=Put carrot in pot}"
: "${POLICY_DEVICE:=cuda}"
: "${DRY_RUN:=0}"

ACT_JOB_NAME="act_carrot_sim"
DIFFUSION_JOB_NAME="diffusion_carrot_sim"
FM_JOB_NAME="fm_carrot_sim"

# Every lerobot-train job THIS TEAM may have on kanu, by --job_name. preflight.sh
# counts these (and only these) against MAX_OUR_GPUS. Other people's jobs are
# covered by the per-GPU compute-process check, not by this budget.
OUR_JOB_NAMES=(
  act_carrot_eef
  fm_carrot_eef
  "$ACT_JOB_NAME"
  "$DIFFUSION_JOB_NAME"
  "$FM_JOB_NAME"
)

# ---------------------------------------------------------------------------
# THE input_features override -- the single most load-bearing thing in this file.
#
# WHY it exists: meta/info.json carries a FOURTH observation feature,
# `observation.sim.object_poses` (float32[14] = carrot + pot pose), which is
# ground truth only MuJoCo can produce. lerobot 0.6.1 would feed it to the
# policy: utils/feature_utils.py::dataset_to_policy_features classifies every
# key that startswith("observation") as FeatureType.STATE, and
# policies/factory.py::make_policy does
#     if not cfg.input_features:
#         cfg.input_features = {k: ft for k, ft in features.items() if k not in output_features}
# i.e. auto-discovery is used ONLY when input_features is empty. A policy that
# learned to read object_poses would be UNDEPLOYABLE: the ZMQ deploy/eval server
# sends state + 2 images and nothing else.
#
# So we pass input_features explicitly. Note the SHAPES:
#   * state is [7] -- as in info.json.
#   * images are CHANNEL-FIRST [3,720,1280]. info.json stores [720,1280,3] with
#     names (height,width,channels); dataset_to_policy_features transposes it.
#     Passing [720,1280,3] here would silently disagree with what the dataloader
#     hands the policy.
#
# output_features is deliberately NOT passed: make_policy OVERWRITES it
# unconditionally from the dataset (`cfg.output_features = {...ACTION...}`), so a
# CLI value would be discarded. Verified in the dry runs -- config.json came out
# with output_features == {"action": ACTION [7]} in all three cases.
: "${SIM_STATE_DIM:=7}"
: "${SIM_IMAGE_H:=720}"
: "${SIM_IMAGE_W:=1280}"

sim_input_features_json() {
  printf '{"observation.state": {"type": "STATE", "shape": [%s]}, "observation.images.cam1": {"type": "VISUAL", "shape": [3, %s, %s]}, "observation.images.cam2": {"type": "VISUAL", "shape": [3, %s, %s]}}' \
    "$SIM_STATE_DIM" "$SIM_IMAGE_H" "$SIM_IMAGE_W" "$SIM_IMAGE_H" "$SIM_IMAGE_W"
}

# ---------------------------------------------------------------------------
# Disk footprint per job, in GiB. MEASURED, not guessed: on kanu an
# act_carrot_eef checkpoint (51.6 M learnable params) occupies 592 MiB and an
# fm_carrot_eef checkpoint (186 M learnable) occupies 2.4 GiB -- i.e. ~12 bytes
# per learnable parameter (fp32 weights + the AdamW moment pair), times
# steps/save_freq checkpoints, and lerobot NEVER prunes: `checkpoints/last` is a
# symlink, every periodic checkpoint is kept forever (there is no retention
# option in TrainPipelineConfig).
#
#   act        51.6 M x 12 B = 0.6 GiB x  (50000/10000) =  3 GiB  -> 6 with slack
#   diffusion 277.9 M x 12 B = 3.3 GiB x (100000/10000) = 33 GiB  -> 36 with slack
#   fm        186.0 M x 12 B = 2.4 GiB x (100000/10000) = 24 GiB  -> 27 with slack
#
# This exists because the failure it prevents is the worst kind: ENOSPC lands on
# safetensors serialization at the FIRST checkpoint save, hours in, and kills the
# run (observed on laptop3 while building this bundle).
sim_job_disk_gib() {
  case "$1" in
    act)       echo 6 ;;
    diffusion) echo 36 ;;
    fm)        echo 27 ;;
    *)         echo 20 ;;
  esac
}

sim_fail() { echo "ERROR: $*" >&2; exit 1; }

# Held-out episode indices implied by EVAL_SPLIT, as lerobot computes them:
# the LAST ceil(total * eval_split) episodes (per task; this dataset has 1 task).
sim_heldout_episodes() {
  python3 - "$TOTAL_EPISODES" "$EVAL_SPLIT" <<'PY'
import math, sys
total = int(sys.argv[1]); frac = float(sys.argv[2])
n = math.ceil(total * frac)
print(",".join(str(i) for i in range(total - n, total)))
PY
}

# Guard the hardcoded shapes above against the dataset actually on disk. Cheap
# (pure-json, ~50 ms) and it runs in DRY_RUN too, because a shape drift is
# exactly the kind of thing a dry run should catch before a 100k-step job.
sim_assert_dataset_shapes() {
  python3 - "$DATASET_ROOT" "$SIM_STATE_DIM" "$SIM_IMAGE_H" "$SIM_IMAGE_W" \
           "$TOTAL_EPISODES" "$FPS" <<'PY' || exit 1
import json, sys
from pathlib import Path
root, sdim, ih, iw, neps, fps = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
info = json.loads((root / "meta/info.json").read_text())
f = info["features"]
def bad(m): raise SystemExit(f"ERROR: dataset contract violated ({root}): {m}")
if f["observation.state"]["shape"] != [sdim]: bad(f"observation.state {f['observation.state']['shape']} != [{sdim}]")
if f["action"]["shape"] != [sdim]:            bad(f"action {f['action']['shape']} != [{sdim}]")
for cam in ("observation.images.cam1", "observation.images.cam2"):
    if cam not in f: bad(f"missing {cam}")
    if f[cam]["shape"] != [ih, iw, 3]: bad(f"{cam} shape {f[cam]['shape']} != [{ih}, {iw}, 3]")
if info.get("fps") != fps:                    bad(f"fps {info.get('fps')} != {fps}")
if info.get("total_episodes") != neps:        bad(f"total_episodes {info.get('total_episodes')} != {neps}")
depth = [k for k in f if "depth" in k]
if depth:                                     bad(f"depth features present: {depth} (this dataset must have none)")
PY
}

# The pip-installed NVIDIA runtime libraries are not on the system loader path.
# torch AND torchcodec (which links CUDA NPP even for CPU decoding) need them.
# Echoes a colon-joined list; fails loudly rather than silently exporting "".
sim_nvidia_libs() {
  local venv="$1" root libs
  root="$(echo "$venv"/lib/python*/site-packages/nvidia)"
  libs="$(find "$root" -maxdepth 2 -name lib -type d 2>/dev/null | sort | paste -sd: -)"
  [[ -n "$libs" ]] || return 1
  printf '%s' "$libs"
}

# Caches: kanu's $HOME is shared and nearly full, so keep every byte lerobot
# writes under our own job root. Never write outside carrot_sim/.
sim_export_caches() {
  mkdir -p "$SCRATCH_DIR/torch_home" "$SCRATCH_DIR/hf_lerobot_home"
  export TORCH_HOME="$SCRATCH_DIR/torch_home"
  export HF_LEROBOT_HOME="$SCRATCH_DIR/hf_lerobot_home"
}

# GPU_ID is REQUIRED for a cuda run: kanu has 8 GPUs shared with other people,
# there is no sane default, so refuse rather than grab GPU 0. A cpu run (the
# 1-step smoke test) does not need one.
sim_require_gpu_id() {
  if [[ "$POLICY_DEVICE" == "cpu" ]]; then
    GPU_ID="${GPU_ID:-}"
    return 0
  fi
  GPU_ID="${GPU_ID:?set GPU_ID to the selected physical GPU index (pick an idle one with nvidia-smi)}"
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || sim_fail "GPU_ID must be a non-negative integer (got '$GPU_ID')"
}

# One TRAINING JOB = one distinct --job_name. lerobot's DataLoader forks
# num_workers children with the identical argv, so counting processes would count
# every job (1 + num_workers) times (observed on kanu: 1 ACT job = 5 matches).
# argv[0] must itself END in /lerobot-train: pgrep -f alone also matches a shell
# that merely MENTIONS the pattern (verified on kanu -- an
# `ssh kanu '... pgrep -f bin/lerobot-train ...'` matched its own `bash -c`).
# The console script is python-wrapped, so the real argv is
# `<venv>/bin/python <venv>/bin/lerobot-train ...` -- hence field $2 OR $3.
sim_our_training_lines() {
  pgrep -u "$(id -u)" -af 'bin/lerobot-train' 2>/dev/null \
    | awk '($2 ~ /\/lerobot-train$/ || $3 ~ /\/lerobot-train$/)' \
    | awk '{job=""; for(i=1;i<=NF;i++) if ($i ~ /^--job_name=/) job=$i; if (job!="" && !(job in seen)) {seen[job]=1; print}}' \
    || true
}

# Same, but keep only OUR job names (OUR_JOB_NAMES). Anything else on the box is
# someone else's work and is handled by the per-GPU compute-process check.
sim_our_budget_lines() {
  local pattern
  pattern="--job_name=($(IFS='|'; echo "${OUR_JOB_NAMES[*]}"))([[:space:]]|$)"
  # `--`: grep must not read the leading "--job_name=" as an option.
  sim_our_training_lines | grep -E -- "$pattern" || true
}

# PID of the running lerobot-train whose --job_name is $1. Lowest PID = the
# parent, not a dataloader worker. Empty when the job is not running.
sim_job_pid() {
  local job="$1"
  pgrep -u "$(id -u)" -af "bin/lerobot-train.*--job_name=$job" 2>/dev/null \
    | awk '($2 ~ /\/lerobot-train$/ || $3 ~ /\/lerobot-train$/) {print $1}' \
    | sort -n | head -1 || true
}

# Map a short job token to (script, job name). Used by launch_detached.sh and
# run_queue.sh so the two cannot disagree about what "diffusion" means.
sim_resolve_job() {
  case "$1" in
    act)       SIM_TRAIN_SCRIPT="train_act_carrot_sim.sh";       SIM_JOB_NAME="$ACT_JOB_NAME" ;;
    diffusion) SIM_TRAIN_SCRIPT="train_diffusion_carrot_sim.sh"; SIM_JOB_NAME="$DIFFUSION_JOB_NAME" ;;
    fm)        SIM_TRAIN_SCRIPT="train_fm_carrot_sim.sh";        SIM_JOB_NAME="$FM_JOB_NAME" ;;
    *) return 1 ;;
  esac
}
