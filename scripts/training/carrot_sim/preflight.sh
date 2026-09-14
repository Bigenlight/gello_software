#!/usr/bin/env bash
# Launch gate for the carrot_in_pot SIM joint-space training jobs.
#
#   ./preflight.sh                    # all three policy classes, no GPU needed
#   ./preflight.sh --no-gpu act       # explicit: skip every GPU check
#   GPU_ID=0 ./preflight.sh act
#   GPU_ID=1 ./preflight.sh diffusion
#   GPU_ID=2 ./preflight.sh fm
#
# Refuses (exit 1) when ANY of these holds:
#   1. the chosen GPU already has a CUDA compute process (someone else's job);
#   2. starting this job would put MORE THAN MAX_OUR_GPUS (=2) of OUR OWN
#      trainings on kanu at once, counted by --job_name over OUR_JOB_NAMES
#      (act_carrot_eef, fm_carrot_eef, act/diffusion/fm_carrot_sim);
#   3. less than MIN_FREE_GIB free on the filesystem holding $SIM_ROOT;
#   4. `lerobot-train --help` does not run in the venv;
#   5. the dataset's meta/info.json is not EXACTLY the sim joint contract
#      (22 episodes, 30 fps, state[7]/action[7], cam1+cam2 720x1280, NO depth,
#      the object_poses key present but shaped [14], task string exactly
#      "Put carrot in pot" on all 22 episodes);
#   6. the DRY_RUN command of the selected job(s) does not carry an explicit
#      --policy.input_features listing exactly the three deployable keys --
#      i.e. observation.sim.object_poses would reach the policy.
#
# Check 6 is the whole reason this file differs from the carrot_eef one. The GPU
# test is process-based, never memory-based.
#
# WITHOUT a GPU: leaving GPU_ID unset (or passing --no-gpu) skips checks 1 and
# the CUDA assertion inside check 5. Everything else still runs, so the script is
# a valid gate on a busy box.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

NO_GPU=0
JOB="both"
JOB_SET=0
for arg in "$@"; do
  case "$arg" in
    --no-gpu) NO_GPU=1 ;;
    act|diffusion|fm|all|both) JOB="$arg"; JOB_SET=1 ;;
    *) sim_fail "usage: [GPU_ID=N] $0 [--no-gpu] [act|diffusion|fm|all]" ;;
  esac
done
[[ "$JOB" == "both" ]] && JOB="all"
(( JOB_SET )) || JOB="all"
if (( NO_GPU )); then GPU_ID=""; fi

case "$JOB" in
  act)       JOBS=(act) ;;
  diffusion) JOBS=(diffusion) ;;
  fm)        JOBS=(fm) ;;
  all)       JOBS=(act diffusion fm) ;;
esac

PYTHON="$LEROBOT_VENV/bin/python"
TRAIN="$LEROBOT_VENV/bin/lerobot-train"
MAX_OUR_GPUS="${MAX_OUR_GPUS:-2}"
# Default: the largest footprint among the selected jobs (see sim_job_disk_gib).
# Override with MIN_FREE_GIB=N when you intend to prune checkpoints as you go.
_need_gib=0
for _j in "${JOBS[@]}"; do
  _g="$(sim_job_disk_gib "$_j")"
  (( _g > _need_gib )) && _need_gib="$_g"
done
MIN_FREE_GIB="${MIN_FREE_GIB:-$_need_gib}"

echo "=============================================================="
echo "carrot_sim preflight   job=$JOB   (SIM-ONLY, joint space)"
echo "  dataset : $DATASET_ROOT"
echo "  repo_id : $REPO_ID ($DATASET_REVISION)"
echo "  venv    : $LEROBOT_VENV"
echo "  job root: $SIM_ROOT"
echo "=============================================================="

# ---- 0. basic existence --------------------------------------------------
[[ -f "$DATASET_ROOT/meta/info.json" ]]  || sim_fail "missing $DATASET_ROOT/meta/info.json"
[[ -f "$DATASET_ROOT/meta/stats.json" ]] || sim_fail "missing $DATASET_ROOT/meta/stats.json"
[[ -f "$DATASET_ROOT/meta/tasks.parquet" ]] || sim_fail "missing $DATASET_ROOT/meta/tasks.parquet"
[[ -x "$PYTHON" ]] || sim_fail "missing executable $PYTHON"
[[ -x "$TRAIN" ]]  || sim_fail "missing executable $TRAIN"

NVIDIA_LIBS="$(sim_nvidia_libs "$LEROBOT_VENV")" \
  || sim_fail "pip NVIDIA library directories are missing under $LEROBOT_VENV"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- 1. the chosen GPU must be compute-free ------------------------------
if [[ -n "${GPU_ID:-}" ]]; then
  command -v nvidia-smi >/dev/null || sim_fail "GPU_ID=$GPU_ID was given but nvidia-smi is unavailable"
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || sim_fail "GPU_ID must be a non-negative integer (got '$GPU_ID')"
  nvidia-smi -i "$GPU_ID" --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  compute_apps="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "${compute_apps//[[:space:]]/}" ]]; then
    echo "FAIL [1] GPU $GPU_ID already has compute processes:" >&2
    echo "$compute_apps" | sed 's/^/         /' >&2
    exit 1
  fi
  echo "PASS [1] GPU $GPU_ID has no CUDA compute process"
else
  echo "SKIP [1] no GPU_ID (--no-gpu) -- schema/venv/feature checks only, no GPU reservation"
fi

# ---- 2. at most MAX_OUR_GPUS of OUR OWN trainings ------------------------
# One lerobot-train job == one GPU (each run pins CUDA_VISIBLE_DEVICES to a
# single index), so counting OUR jobs counts OUR GPUs. Restricted to this account
# AND to OUR_JOB_NAMES: other people's jobs are covered by check 1, not by this
# budget. See sim_our_training_lines() in _common.sh for why the awk is shaped
# the way it is (python-wrapped console script; dataloader forks).
ours="$(sim_our_budget_lines)"
n_ours=0
[[ -n "${ours//[[:space:]]/}" ]] && n_ours="$(printf '%s\n' "$ours" | grep -c .)"
n_new=${#JOBS[@]}
# When asked about all three at once this is a capability question, not a launch:
# report the budget but only refuse on the per-job (n_new == 1) case.
if (( n_new == 1 )) && (( n_ours + 1 > MAX_OUR_GPUS )); then
  echo "FAIL [2] we already run $n_ours of our training job(s); one more would exceed the $MAX_OUR_GPUS-job budget:" >&2
  printf '%s\n' "$ours" | sed 's/^/         /' >&2
  echo "         (override with MAX_OUR_GPUS=N only after agreeing it with the other GPU users)" >&2
  exit 1
fi
echo "PASS [2] our running trainings: $n_ours (budget $MAX_OUR_GPUS; this launch would make $((n_ours + 1)))"
[[ -n "${ours//[[:space:]]/}" ]] && printf '%s\n' "$ours" | sed 's/^/         /'
if (( n_new > 1 )); then
  echo "         NOTE: $n_new jobs were checked at once; the budget is enforced per single-job invocation."
fi

# ---- 3. disk --------------------------------------------------------------
mkdir -p "$SIM_ROOT"
df -h "$SIM_ROOT" "$DATASET_ROOT" || true
available_kb="$(df -Pk "$SIM_ROOT" | awk 'NR==2 {print $4}')"
echo "         checkpoint footprint (measured, no pruning): $(for _j in "${JOBS[@]}"; do printf '%s=%sGiB ' "$_j" "$(sim_job_disk_gib "$_j")"; done)"
if (( available_kb < MIN_FREE_GIB * 1024 * 1024 )); then
  echo "FAIL [3] only $((available_kb / 1024 / 1024)) GiB free on the filesystem holding $SIM_ROOT (need $MIN_FREE_GIB GiB)" >&2
  echo "         lerobot keeps EVERY periodic checkpoint. Either free space, raise SAVE_FREQ" >&2
  echo "         (fewer checkpoints), or accept the risk with MIN_FREE_GIB=N and prune by hand" >&2
  echo "         DURING the run. Running out of disk kills the job at a checkpoint save." >&2
  exit 1
fi
echo "PASS [3] $((available_kb / 1024 / 1024)) GiB free under $SIM_ROOT (need $MIN_FREE_GIB)"

# ---- 4. lerobot-train --help ---------------------------------------------
HELP_ERR="$SIM_ROOT/.preflight_lerobot_train_help.err"
if ! "$TRAIN" --help >/dev/null 2>"$HELP_ERR"; then
  echo "FAIL [4] 'lerobot-train --help' failed in $LEROBOT_VENV:" >&2
  tail -n 20 "$HELP_ERR" >&2 || true
  exit 1
fi
rm -f "$HELP_ERR"
echo "PASS [4] lerobot-train --help runs in the venv"

# ---- 5. dataset schema + policy classes ----------------------------------
# Only pin CUDA_VISIBLE_DEVICES when a GPU was actually chosen: exporting an
# empty value would hide every GPU and make the CUDA assertion below meaningless.
if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export SIM_REQUIRE_CUDA=1
else
  export SIM_REQUIRE_CUDA=0
fi
"$PYTHON" - \
  "$DATASET_ROOT" "${JOBS[*]}" "$TOTAL_EPISODES" "$TOTAL_FRAMES" "$EVAL_SPLIT" \
  "$TASK_STRING" "$FPS" "$SIM_STATE_DIM" "$SIM_IMAGE_H" "$SIM_IMAGE_W" <<'PY'
"""Strict schema gate for the SIM joint-space dataset."""
import json
import math
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
jobs = sys.argv[2].split()
total_expected = int(sys.argv[3])
frames_expected = int(sys.argv[4])
eval_split = float(sys.argv[5])
task_string = sys.argv[6]
fps_expected = int(sys.argv[7])
sdim = int(sys.argv[8])
ih = int(sys.argv[9])
iw = int(sys.argv[10])

info = json.loads((root / "meta/info.json").read_text())
feats = info.get("features", {})

# --- exact feature set ----------------------------------------------------
# observation.sim.object_poses is EXPECTED to be present in the dataset. It is
# legitimate sim ground truth and useful for offline analysis; what must never
# happen is it reaching the policy. That is check 6, not this one.
EXPECTED_KEYS = {
    "action",
    "observation.state",
    "observation.images.cam1",
    "observation.images.cam2",
    "observation.sim.object_poses",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}
got = set(feats)
missing = sorted(EXPECTED_KEYS - got)
extra = sorted(got - EXPECTED_KEYS)
if missing or extra:
    msg = [f"dataset feature set != sim joint contract (root={root})"]
    if missing:
        msg.append(f"  missing: {missing}")
    if extra:
        msg.append(f"  unexpected: {extra}")
        if any("depth" in k for k in extra):
            msg.append("  NOTE: this dataset must carry NO depth feature.")
        msg.append(
            "  NOTE: any NEW observation.* key would be auto-fed to the policy unless it is "
            "also excluded from --policy.input_features in the three train scripts."
        )
    raise SystemExit("FAIL [5] " + "\n".join(msg))

depth = [k for k in got if "depth" in k] + [
    k for k, v in feats.items() if isinstance(v.get("info"), dict) and v["info"].get("is_depth_map")
]
if depth:
    raise SystemExit(f"FAIL [5] depth features present: {sorted(set(depth))}")

# --- shapes / names -------------------------------------------------------
def need(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL [5] {msg}")

need(feats["action"]["shape"] == [sdim], f"action shape {feats['action']['shape']} != [{sdim}]")
need(
    feats["observation.state"]["shape"] == [sdim],
    f"observation.state shape {feats['observation.state']['shape']} != [{sdim}]",
)
need(
    feats["observation.sim.object_poses"]["shape"] == [14],
    f"observation.sim.object_poses shape {feats['observation.sim.object_poses']['shape']} != [14]",
)
for cam in ("observation.images.cam1", "observation.images.cam2"):
    need(feats[cam]["dtype"] == "video", f"{cam} dtype {feats[cam]['dtype']} != video")
    need(feats[cam]["shape"] == [ih, iw, 3], f"{cam} shape {feats[cam]['shape']} != [{ih}, {iw}, 3]")

# JOINT space, not EEF. These names are the contract the deploy server speaks.
STATE_NAMES = [f"ur_q{i}" for i in range(1, 7)] + ["grip_pos"]
ACTION_NAMES = [f"cmd{i}" for i in range(1, 7)] + ["grip_cmd"]
state_names = list(feats["observation.state"].get("names") or [])
action_names = list(feats["action"].get("names") or [])
need(state_names == STATE_NAMES, f"observation.state names {state_names} != {STATE_NAMES}")
need(action_names == ACTION_NAMES, f"action names {action_names} != {ACTION_NAMES}")

# --- dataset-level facts --------------------------------------------------
need(info.get("codebase_version") == "v3.0", f"codebase_version {info.get('codebase_version')} != v3.0")
need(info.get("fps") == fps_expected, f"fps {info.get('fps')} != {fps_expected}")
total = info.get("total_episodes")
need(total == total_expected, f"total_episodes {total} != {total_expected}")
need(
    info.get("total_frames") == frames_expected,
    f"total_frames {info.get('total_frames')} != {frames_expected}",
)
need(info.get("total_tasks") == 1, f"total_tasks {info.get('total_tasks')} != 1")

# --- eval split -----------------------------------------------------------
# Reproduces lerobot 0.6.1 datasets/factory.py::make_train_eval_datasets exactly:
# per task, n_eval = ceil(len(eps) * eval_split) and the LAST n_eval are held out.
n_held = math.ceil(total * eval_split)
held = list(range(total - n_held, total))
need(n_held == 2, f"eval_split {eval_split} holds out {n_held} episodes, expected 2")
need(held == [total - 2, total - 1], f"held-out set {held} is not the last two episodes")

# --- task string, per episode --------------------------------------------
import pandas as pd  # noqa: E402

tasks = pd.read_parquet(root / "meta/tasks.parquet")
task_values = list(tasks.index) if tasks.index.name == "task" else list(tasks.get("task", []))
need(
    task_values == [task_string],
    f"meta/tasks.parquet tasks {task_values!r} != [{task_string!r}]",
)
ep_files = sorted((root / "meta/episodes").rglob("*.parquet"))
need(bool(ep_files), "meta/episodes/**/*.parquet is missing")
eps = pd.concat([pd.read_parquet(p) for p in ep_files])
need(len(eps) == total, f"meta/episodes rows {len(eps)} != total_episodes {total}")
bad = [
    int(r.episode_index)
    for r in eps.itertuples()
    if list(r.tasks) != [task_string]
]
need(not bad, f"episodes with a task string != {task_string!r}: {bad}")
need(
    int(eps["length"].sum()) == frames_expected,
    f"sum(episode lengths) {int(eps['length'].sum())} != total_frames {frames_expected}",
)
# The split is PER TASK; with a single task the held-out set is literally the
# last two episode indices. Assert the premise so a future multi-task dataset
# cannot silently change what "last 2" means.
need(len(set(map(tuple, eps["tasks"]))) == 1, "more than one task string: the per-task eval split would no longer be [20, 21]")

# --- torch / policy classes ----------------------------------------------
import torch  # noqa: E402
import lerobot  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402

PTYPE = {"act": "act", "diffusion": "diffusion", "fm": "multi_task_dit"}
for job in jobs:
    cls = get_policy_class(PTYPE[job])
    print(f"  policy '{job}' -> {PTYPE[job]} -> {cls.__name__}")
if "fm" in jobs:
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import (  # noqa: E402
        MultiTaskDiTConfig,
    )
    cfg = MultiTaskDiTConfig(objective="flow_matching")
    need(cfg.objective == "flow_matching", "multi_task_dit objective could not be set to flow_matching")

if os.environ.get("SIM_REQUIRE_CUDA") == "1":
    need(torch.cuda.is_available(), "PyTorch cannot access CUDA on the selected GPU")

print(f"  lerobot {lerobot.__version__}; torch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()} (devices visible: {torch.cuda.device_count()})")
print(f"  features OK: 10 exactly, state[{sdim}], action[{sdim}], cam1+cam2 {ih}x{iw}, no depth")
print(f"  state names : {state_names}")
print(f"  action names: {action_names}")
print(f"  object_poses: shape {feats['observation.sim.object_poses']['shape']} (present in the DATASET; excluded from the POLICY by check 6)")
print(f"  episodes {total}, frames {info.get('total_frames')}, fps {info.get('fps')}, held out {held}")
print(f"  task string: {task_string!r} on all {total} episodes")
PY
echo "PASS [5] dataset schema, task string and policy classes"

# ---- 6. object_poses must not reach the policy ---------------------------
# The strongest available check short of training: run each selected job's own
# DRY_RUN and inspect the command it would execute. If someone deletes the
# --policy.input_features line, lerobot's auto-discovery silently adds
# observation.sim.object_poses -- and NOTHING else in this preflight would see it.
for job in "${JOBS[@]}"; do
  sim_resolve_job "$job" || sim_fail "unknown job '$job'"
  out="$(DRY_RUN=1 GPU_ID="${GPU_ID:-0}" "$ROOT_DIR/$SIM_TRAIN_SCRIPT")" \
    || sim_fail "[6] DRY_RUN of $SIM_TRAIN_SCRIPT failed"
  # The dry-run text goes in ARGV, not on stdin: `python - <<HEREDOC` already uses
  # stdin to read the program, so a pipe into it would arrive empty.
  "$PYTHON" - "$job" "$SIM_JOB_NAME" "$SIM_STATE_DIM" "$SIM_IMAGE_H" "$SIM_IMAGE_W" "$out" <<'PY'
import json
import sys

job, job_name, sdim, ih, iw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
text = sys.argv[6]
lines = text.splitlines()

marker = "# input_features: "
found = [ln[len(marker):] for ln in lines if ln.startswith(marker)]
if len(found) != 1:
    raise SystemExit(f"FAIL [6] {job}: expected exactly one '{marker}' line, got {len(found)}")
feats = json.loads(found[0])

WANT = {
    "observation.state": ("STATE", [sdim]),
    "observation.images.cam1": ("VISUAL", [3, ih, iw]),
    "observation.images.cam2": ("VISUAL", [3, ih, iw]),
}
if set(feats) != set(WANT):
    raise SystemExit(f"FAIL [6] {job}: input_features keys {sorted(feats)} != {sorted(WANT)}")
for k, (t, s) in WANT.items():
    if feats[k]["type"] != t or list(feats[k]["shape"]) != s:
        raise SystemExit(f"FAIL [6] {job}: input_features[{k}] = {feats[k]} != type {t}, shape {s}")

# The command itself must carry the override and must not mention the sim key.
cmd_lines = [ln for ln in lines if not ln.startswith("#")]
cmd = "\n".join(cmd_lines)
if "--policy.input_features=" not in cmd:
    raise SystemExit(f"FAIL [6] {job}: the command carries no --policy.input_features override")
if "object_poses" in cmd:
    raise SystemExit(f"FAIL [6] {job}: 'object_poses' appears in the training command")
if f"--job_name={job_name}" not in cmd:
    raise SystemExit(f"FAIL [6] {job}: the command does not carry --job_name={job_name}")
print(f"  {job:9s} -> --job_name={job_name}; input_features = {sorted(feats)}; object_poses EXCLUDED")
PY
done
echo "PASS [6] every selected job passes an explicit input_features that excludes observation.sim.object_poses"

echo "=============================================================="
echo "PREFLIGHT: PASS   (held-out episodes: $(sim_heldout_episodes))"
