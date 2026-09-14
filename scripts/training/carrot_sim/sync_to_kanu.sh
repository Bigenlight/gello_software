#!/usr/bin/env bash
# Push this training bundle (and, if asked, the local dataset copy) to kanu.
#
#   ./sync_to_kanu.sh --dry-run        # rsync's own --dry-run: shows every file, transfers none
#   ./sync_to_kanu.sh --bundle-only    # scripts only (the usual case: kanu downloads the
#                                      # dataset itself from the Hub)
#   ./sync_to_kanu.sh                  # bundle + dataset
#   ./sync_to_kanu.sh --delete-dataset # mirror the dataset exactly (removes remote extras)
#
# Writes ONLY under kanu:~/workspace/youngwoong/carrot_sim/. It never touches
# carrot_eef/ (two of our jobs run there) nor anyone else's tree.
#
# NOTE: this script runs on laptop3, so it does NOT source _common.sh (whose
# defaults are kanu-side paths).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="${SSH_HOST:-kanu}"
REMOTE_ROOT="${REMOTE_ROOT:-workspace/youngwoong/carrot_sim}"
LOCAL_DATASET="${LOCAL_DATASET:-/home/laptop3/youngwoong_ws/carrot_in_pot_sim_lerobot}"
REMOTE_DATASET_NAME="${REMOTE_DATASET_NAME:-carrot_in_pot_sim_lerobot}"

DRY=()
DELETE_DATASET=()
BUNDLE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=(--dry-run) ;;
    --bundle-only) BUNDLE_ONLY=1 ;;
    # Replace a DISCARDED dataset: rsync --delete, scoped to the dataset
    # directory only. NEVER applied to the bundle rsync: runtime/ and outputs/
    # live only on kanu and --delete there would destroy a running job's logs
    # and checkpoints.
    --delete-dataset) DELETE_DATASET=(--delete) ;;
    *) echo "usage: $0 [--dry-run] [--bundle-only] [--delete-dataset]" >&2; exit 2 ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

command -v rsync >/dev/null || fail "rsync is required"
# Keep the remote path inside the home directory and free of traversal: this
# script is the only thing standing between a typo and someone else's files.
[[ "$REMOTE_ROOT" =~ ^[A-Za-z0-9._/-]+$ && "$REMOTE_ROOT" != /* && "$REMOTE_ROOT" != *..* ]] \
  || fail "REMOTE_ROOT must be a safe path relative to the remote home (got '$REMOTE_ROOT')"
[[ "$REMOTE_ROOT" == workspace/youngwoong/carrot_sim* ]] \
  || fail "refusing to write outside ~/workspace/youngwoong/carrot_sim on $SSH_HOST"

if (( ! BUNDLE_ONLY )); then
  [[ -d "$LOCAL_DATASET" ]] || fail "local dataset dir is missing: $LOCAL_DATASET (use --bundle-only)"
  [[ -f "$LOCAL_DATASET/meta/info.json" ]] \
    || fail "$LOCAL_DATASET has no meta/info.json -- the download is unfinished or failed"
  episodes="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('total_episodes'))" \
    "$LOCAL_DATASET/meta/info.json" 2>/dev/null || echo '?')"
  echo "Local dataset : $LOCAL_DATASET (total_episodes=$episodes)"
else
  echo "Local dataset : (skipped: --bundle-only)"
fi
echo "Local bundle  : $ROOT_DIR"
echo "Destination   : $SSH_HOST:~/$REMOTE_ROOT/{training,dataset/$REMOTE_DATASET_NAME}"
if [[ ${#DRY[@]} -gt 0 ]]; then echo "MODE          : DRY RUN (nothing is transferred)"; fi
if [[ ${#DELETE_DATASET[@]} -gt 0 ]]; then
  echo "MODE          : --delete-dataset (remote dataset dir is mirrored exactly; extras removed)"
fi
echo

if [[ ${#DRY[@]} -eq 0 ]]; then
  ssh -o BatchMode=yes "$SSH_HOST" "mkdir -p '$REMOTE_ROOT/training' '$REMOTE_ROOT/dataset'"
fi

# Bundle: source only. runtime/ outputs/ .cache/ are produced ON kanu and must
# never be overwritten from laptop3.
rsync -av "${DRY[@]}" --checksum --partial \
  --exclude='runtime/' --exclude='outputs/' --exclude='evaluations/' \
  --exclude='.cache/' --exclude='__pycache__/' --exclude='*.pyc' \
  "$ROOT_DIR/" "$SSH_HOST:$REMOTE_ROOT/training/"

if (( ! BUNDLE_ONLY )); then
  # .cache/ is huggingface_hub's per-file download bookkeeping: local to whoever
  # downloaded, useless remotely, and it would confuse a later `hf download`.
  rsync -av "${DRY[@]}" "${DELETE_DATASET[@]}" --checksum --partial \
    --exclude='.cache/' \
    "$LOCAL_DATASET/" "$SSH_HOST:$REMOTE_ROOT/dataset/$REMOTE_DATASET_NAME/"
fi

echo
if [[ ${#DRY[@]} -gt 0 ]]; then
  echo "Dry run complete. Re-run without --dry-run to transfer."
  exit 0
fi
cat <<EOS
Transfer complete. On $SSH_HOST:

  cd ~/$REMOTE_ROOT/training
  ./preflight.sh --no-gpu               # schema + feature gate, no GPU needed
  nvidia-smi                            # pick an idle GPU, right now
  DRY_RUN=1 GPU_ID=<a> ./train_act_carrot_sim.sh
  DRY_RUN=1 GPU_ID=<a> ./train_diffusion_carrot_sim.sh
  DRY_RUN=1 GPU_ID=<b> ./train_fm_carrot_sim.sh
  # launching is the integrator's call:
  # ./run_queue.sh <a> act diffusion    # sequential on one GPU
  # GPU_ID=<b> ./launch_detached.sh fm
EOS
