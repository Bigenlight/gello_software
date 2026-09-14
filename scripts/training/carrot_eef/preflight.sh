#!/usr/bin/env bash
# Launch gate for the carrot_in_pot EEF training jobs.
#
#   ./preflight.sh            # checks both jobs' policy classes
#   GPU_ID=0 ./preflight.sh act
#   GPU_ID=1 ./preflight.sh fm
#
# Refuses (exit 1) when ANY of these holds:
#   1. the chosen GPU already has a CUDA compute process (someone else's job);
#   2. starting this job would put MORE THAN 2 GPUs under our own lerobot-train
#      processes (DESIGN §1: kanu is shared, we take at most 2);
#   3. less than 20 GiB free on the filesystem holding ~/workspace/youngwoong/carrot_eef;
#   4. the dataset's meta/info.json feature set is not EXACTLY the EEF contract
#      (9 features, state 16 / action 7, cam1 + cam2, and NO depth);
#   5. `lerobot-train --help` does not run in the venv.
#
# Modelled on cube_flow_matching_training/preflight.sh; the GPU test is
# process-based, never memory-based (banana TROUBLESHOOTING §6).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$ROOT_DIR/_common.sh"

JOB="${1:-both}"
case "$JOB" in
  act|fm|both) ;;
  *) carrot_fail "JOB must be act, fm or both (got '$JOB')" ;;
esac

PYTHON="$LEROBOT_VENV/bin/python"
TRAIN="$LEROBOT_VENV/bin/lerobot-train"
MAX_OUR_GPUS="${MAX_OUR_GPUS:-2}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"

echo "=============================================================="
echo "carrot_eef preflight   job=$JOB"
echo "  dataset : $DATASET_ROOT"
echo "  venv    : $LEROBOT_VENV"
echo "  job root: $CARROT_ROOT"
echo "=============================================================="

# ---- 0. basic existence --------------------------------------------------
[[ -f "$DATASET_ROOT/meta/info.json" ]]  || carrot_fail "missing $DATASET_ROOT/meta/info.json"
[[ -f "$DATASET_ROOT/meta/stats.json" ]] || carrot_fail "missing $DATASET_ROOT/meta/stats.json"
[[ -x "$PYTHON" ]] || carrot_fail "missing executable $PYTHON"
[[ -x "$TRAIN" ]]  || carrot_fail "missing executable $TRAIN"
command -v nvidia-smi >/dev/null || carrot_fail "nvidia-smi is unavailable"

NVIDIA_LIBS="$(carrot_nvidia_libs "$LEROBOT_VENV")" \
  || carrot_fail "pip NVIDIA library directories are missing under $LEROBOT_VENV"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- 1. the chosen GPU must be compute-free ------------------------------
if [[ -n "${GPU_ID:-}" ]]; then
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || carrot_fail "GPU_ID must be a non-negative integer (got '$GPU_ID')"
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
  echo "SKIP [1] GPU_ID unset -- schema/venv checks only, no GPU reservation"
fi

# ---- 2. at most MAX_OUR_GPUS of our own trainings ------------------------
# One lerobot-train process == one GPU (each run sets CUDA_VISIBLE_DEVICES to a
# single index), so counting our processes counts our GPUs. -u restricts it to
# this account: other users' jobs are covered by check 1, not by this budget.
# The pattern is anchored on 'bin/lerobot-train' (the real argv[0]) so a shell
# that merely MENTIONS the name -- e.g. the ssh 'bash -c' that invoked this very
# script -- is not miscounted as a training job. Verified: an unanchored
# 'lerobot-train' matched an ssh bash -c on kanu.
# argv[0] must itself END in /lerobot-train. pgrep -f alone is not enough: a shell
# whose command line merely MENTIONS the pattern matches it (verified on kanu --
# an `ssh kanu '... pgrep -f bin/lerobot-train ...'` matched its own bash -c).
# One TRAINING JOB = one distinct --job_name: lerobot's DataLoader forks
# num_workers children with the identical argv, so counting processes would
# count every job (1 + num_workers) times (observed: 1 ACT job = 5 matches).
ours="$(pgrep -u "$(id -u)" -af 'bin/lerobot-train' 2>/dev/null \
  | awk '($2 ~ /\/lerobot-train$/ || $3 ~ /\/lerobot-train$/)' \
  | awk '{job=""; for(i=1;i<=NF;i++) if ($i ~ /^--job_name=/) job=$i; if (job!="" && !(job in seen)) {seen[job]=1; print}}' || true)"
n_ours=0
[[ -n "${ours//[[:space:]]/}" ]] && n_ours="$(printf '%s\n' "$ours" | grep -c . )"
if (( n_ours + 1 > MAX_OUR_GPUS )); then
  echo "FAIL [2] we already run $n_ours lerobot-train process(es); one more would exceed the $MAX_OUR_GPUS-GPU budget:" >&2
  printf '%s\n' "$ours" | sed 's/^/         /' >&2
  exit 1
fi
echo "PASS [2] our running lerobot-train processes: $n_ours (budget $MAX_OUR_GPUS, this job would make $((n_ours + 1)))"

# ---- 3. disk --------------------------------------------------------------
mkdir -p "$CARROT_ROOT"
df -h "$CARROT_ROOT" "$DATASET_ROOT" || true
available_kb="$(df -Pk "$CARROT_ROOT" | awk 'NR==2 {print $4}')"
if (( available_kb < MIN_FREE_GIB * 1024 * 1024 )); then
  echo "FAIL [3] only $((available_kb / 1024 / 1024)) GiB free on the filesystem holding $CARROT_ROOT (need $MIN_FREE_GIB GiB)" >&2
  exit 1
fi
echo "PASS [3] $((available_kb / 1024 / 1024)) GiB free under $CARROT_ROOT (need $MIN_FREE_GIB)"

# ---- 4. lerobot-train --help ---------------------------------------------
HELP_ERR="$CARROT_ROOT/.preflight_lerobot_train_help.err"
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
  export CARROT_REQUIRE_CUDA=1
else
  export CARROT_REQUIRE_CUDA=0
fi
export CARROT_ALLOW_STALE_TAIL="${ALLOW_STALE_TAIL:-0}"
"$PYTHON" - \
  "$DATASET_ROOT" "$JOB" "$TOTAL_EPISODES" "$EVAL_SPLIT" "$TASK_STRING" \
  "${EXPECTED_STATE_NAMES:-}" "${EXPECTED_ACTION_NAMES:-}" <<'PY'
"""Strict schema gate for the EEF-delta dataset (CARROT_EEF_DESIGN.md §2)."""
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
job = sys.argv[2]
total_expected = int(sys.argv[3])
eval_split = float(sys.argv[4])
task_string = sys.argv[5]
state_override = [s for s in sys.argv[6].split(",") if s]
action_override = [s for s in sys.argv[7].split(",") if s]

info = json.loads((root / "meta/info.json").read_text())
feats = info.get("features", {})

# --- exact feature set ----------------------------------------------------
EXPECTED_KEYS = {
    "action",
    "observation.state",
    "observation.images.cam1",
    "observation.images.cam2",
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
    depth = [k for k in extra if "depth" in k]
    msg = [f"dataset feature set != EEF contract (root={root})"]
    if missing:
        msg.append(f"  missing: {missing}")
    if extra:
        msg.append(f"  unexpected: {extra}")
    if depth:
        msg.append(
            "  NOTE: depth features are present. The EEF training recipes carry no "
            "--policy.input_features override, so lerobot would feed depth to the "
            "policy. Re-run A1's converter without depth."
        )
    raise SystemExit("FAIL [5] " + "\n".join(msg))

# --- shapes ---------------------------------------------------------------
assert feats["action"]["shape"] == [7], f"action shape {feats['action']['shape']} != [7]"
assert feats["observation.state"]["shape"] == [16], (
    f"observation.state shape {feats['observation.state']['shape']} != [16]"
)
for cam in ("observation.images.cam1", "observation.images.cam2"):
    assert feats[cam]["dtype"] == "video", f"{cam} dtype {feats[cam]['dtype']} != video"
    assert not feats[cam].get("info", {}).get("is_depth_map", False), f"{cam} is a depth map"

# --- names ---------------------------------------------------------------
# Single source of truth is gello_policy/eef_space.py:
#   STATE_NAMES  = ur_q1..ur_q6, tcp_x/y/z, tcp_r11,tcp_r21,tcp_r31,tcp_r12,tcp_r22,tcp_r32, grip_pos
#   ACTION_NAMES = dx, dy, dz, drx, dry, drz, grip_cmd
# CARROT_EEF_DESIGN.md §2 writes the same 16 entries in shorthand ("q1..q6",
# "r11..r32"). Both spellings denote exactly the same quantities, so accept the
# 4 combinations of {q,ur_q} x {r,tcp_r} and REPORT which one A1 emitted.
# Anything else fails: a different name means a different quantity.
ACTION_NAMES = action_override or ["dx", "dy", "dz", "drx", "dry", "drz", "grip_cmd"]
ROT_KEYS = ["r11", "r21", "r31", "r12", "r22", "r32"]
if state_override:
    STATE_CANDIDATES = [state_override]
else:
    STATE_CANDIDATES = [
        [f"{jp}{i}" for i in range(1, 7)]
        + ["tcp_x", "tcp_y", "tcp_z"]
        + [f"{rp}{k}" for k in ROT_KEYS]
        + ["grip_pos"]
        for jp in ("ur_q", "q")
        for rp in ("tcp_", "")
    ]

action_names = list(feats["action"].get("names") or [])
state_names = list(feats["observation.state"].get("names") or [])
if action_names != ACTION_NAMES:
    raise SystemExit(
        f"FAIL [5] action names {action_names} != {ACTION_NAMES} "
        "(override with EXPECTED_ACTION_NAMES=a,b,... after telling the integrator)"
    )
if state_names not in STATE_CANDIDATES:
    raise SystemExit(
        "FAIL [5] observation.state names do not match the eef_space.py contract\n"
        f"  got     : {state_names}\n"
        + "".join(f"  accepted: {c}\n" for c in STATE_CANDIDATES)
        + "  (override with EXPECTED_STATE_NAMES=a,b,... only after telling the integrator)"
    )

# --- dataset-level facts --------------------------------------------------
assert info.get("codebase_version") == "v3.0", info.get("codebase_version")
assert info.get("fps") == 30, info.get("fps")
total = info.get("total_episodes")
assert total == total_expected, f"total_episodes {total} != {total_expected}"
n_held = math.ceil(total * eval_split)
held = list(range(total - n_held, total))
assert n_held == 6, f"eval_split {eval_split} holds out {n_held} episodes, expected 6"

# --- stale-tail provenance (meta/source_takes.json) ----------------------
# The first carrot EEF conversion kept a FROZEN TAIL at the end of every take
# (the recorder's joint cache stopped updating), which the finite-difference
# action turns into a run of zero deltas the policy would learn to imitate. The
# accepted conversion drops it (--drop-stale-tail) and records how many frames
# went per take. total_frames alone cannot distinguish the two conversions
# reliably, so assert the drop explicitly: a dataset whose takes all report
# n_tail_dropped == 0 is the DISCARDED conversion.
# Escape hatch (only if the contract changes): ALLOW_STALE_TAIL=1.
import os  # noqa: E402
takes_path = root / "meta/source_takes.json"
tail_note = "meta/source_takes.json: ABSENT"
if takes_path.is_file():
    raw = json.loads(takes_path.read_text())
    entries = raw.get("episodes", raw) if isinstance(raw, dict) else raw
    if isinstance(entries, dict):
        entries = list(entries.values())
    # The converter writes n_frames_dropped_stale_tail; n_tail_dropped is the
    # name the hand-off note used. Accept either rather than silently skipping
    # the gate on a spelling difference.
    TAIL_KEYS = ("n_frames_dropped_stale_tail", "n_tail_dropped")
    def _tail(entry):
        for key in TAIL_KEYS:
            if key in entry:
                return entry[key]
        return None
    dropped = [_tail(e) for e in entries if isinstance(e, dict)]
    known = [d for d in dropped if isinstance(d, (int, float))]
    if not known:
        tail_note = (
            f"meta/source_takes.json: {len(entries)} take(s), none of {list(TAIL_KEYS)} "
            "present -- cannot tell the stale-tail conversion apart (report to the integrator)"
        )
    else:
        total_dropped = int(sum(known))
        n_zero = sum(1 for d in known if d == 0)
        if total_dropped == 0 and os.environ.get("CARROT_ALLOW_STALE_TAIL") != "1":
            raise SystemExit(
                "FAIL [5] every take reports n_tail_dropped == 0: this is the DISCARDED "
                "pre---drop-stale-tail conversion, whose frozen tails become runs of zero "
                "action deltas.\n  Re-run the converter with --drop-stale-tail and re-sync "
                "with ./sync_to_kanu.sh --delete-dataset.\n  (override: ALLOW_STALE_TAIL=1)"
            )
        tail_note = (
            f"meta/source_takes.json: {len(known)} take(s), {total_dropped} tail frames "
            f"dropped in total (min {min(known)}, max {max(known)}, "
            f"{n_zero} take(s) dropped nothing)"
        )

# --- eef_space provenance (A1 writes meta/eef_space.json) -----------------
eef_meta_path = root / "meta/eef_space.json"
eef_meta = json.loads(eef_meta_path.read_text()) if eef_meta_path.is_file() else None
if eef_meta is not None:
    space_id = eef_meta.get("ACTION_SPACE_ID") or eef_meta.get("action_space_id")
    assert space_id == "eef_delta_v1", f"meta/eef_space.json action space {space_id!r} != 'eef_delta_v1'"

# --- torch / policy classes ----------------------------------------------
import torch  # noqa: E402
import lerobot  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402

wanted = {"act": ["act"], "fm": ["multi_task_dit"], "both": ["act", "multi_task_dit"]}[job]
for ptype in wanted:
    cls = get_policy_class(ptype)
    print(f"  policy '{ptype}' -> {cls.__name__}")
if "multi_task_dit" in wanted:
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import (  # noqa: E402
        MultiTaskDiTConfig,
    )
    cfg = MultiTaskDiTConfig(objective="flow_matching")
    assert cfg.objective == "flow_matching"

if os.environ.get("CARROT_REQUIRE_CUDA") == "1":
    assert torch.cuda.is_available(), "PyTorch cannot access CUDA on the selected GPU"
print(f"  lerobot {lerobot.__version__}; torch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()} (devices visible: {torch.cuda.device_count()})")
print(f"  features OK: 9 exactly, state[16], action[7], cam1+cam2, no depth")
print(f"  state names : {state_names}")
print(f"  action names: {action_names}")
print(f"  episodes {total}, frames {info.get('total_frames')}, held out {held}")
print(f"  {tail_note}")
print(f"  task string in meta: {task_string!r} (informational)")
if eef_meta is not None:
    print(f"  meta/eef_space.json: {json.dumps(eef_meta, ensure_ascii=False)}")
else:
    print("  meta/eef_space.json: ABSENT (A1 was asked to write it -- report to the integrator)")
PY

echo "PASS [5] dataset schema and policy classes"
echo "=============================================================="
echo "PREFLIGHT: PASS   (held-out episodes: $(carrot_heldout_episodes))"
