#!/usr/bin/env bash
# Push this training bundle and A1's EEF LeRobot dataset to kanu.
#
#   ./sync_to_kanu.sh --dry-run     # rsync's own --dry-run: shows every file, transfers none
#   ./sync_to_kanu.sh               # real transfer
#
# Writes ONLY under kanu:~/workspace/youngwoong/carrot_eef/ (DESIGN §6 rule).
# Refuses if the local dataset has no meta/info.json -- a half-written dataset
# would otherwise be rsync'd and then silently trained on.
#
# NOTE: this script runs on laptop3, so it does NOT source _common.sh (whose
# defaults are kanu-side paths).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_HOST="${SSH_HOST:-kanu}"
REMOTE_ROOT="${REMOTE_ROOT:-workspace/youngwoong/carrot_eef}"
LOCAL_DATASET="${LOCAL_DATASET:-/home/laptop3/youngwoong_ws/carrot_in_pot_eef_lerobot}"
REMOTE_DATASET_NAME="${REMOTE_DATASET_NAME:-carrot_in_pot_eef_lerobot}"

DRY=()
DELETE_DATASET=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=(--dry-run) ;;
    # Replace a DISCARDED dataset: rsync --delete, scoped to the dataset
    # directory only. Needed when the new conversion has FEWER files/frames than
    # the one already on kanu -- without it the stale extras survive and the
    # trainer would read a mix of two conversions. NEVER applied to the bundle
    # rsync: runtime/, outputs/ and evaluations/ live only on kanu and --delete
    # there would destroy a running job's logs and checkpoints.
    --delete-dataset) DELETE_DATASET=(--delete) ;;
    *) echo "usage: $0 [--dry-run] [--delete-dataset]" >&2; exit 2 ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

command -v rsync >/dev/null || fail "rsync is required"
# Keep the remote path inside the home directory and free of traversal: this
# script is the only thing standing between a typo and someone else's files.
[[ "$REMOTE_ROOT" =~ ^[A-Za-z0-9._/-]+$ && "$REMOTE_ROOT" != /* && "$REMOTE_ROOT" != *..* ]] \
  || fail "REMOTE_ROOT must be a safe path relative to the remote home (got '$REMOTE_ROOT')"
[[ "$REMOTE_ROOT" == workspace/youngwoong/carrot_eef* ]] \
  || fail "refusing to write outside ~/workspace/youngwoong/carrot_eef on $SSH_HOST"
[[ -d "$LOCAL_DATASET" ]] || fail "local dataset dir is missing: $LOCAL_DATASET"
[[ -f "$LOCAL_DATASET/meta/info.json" ]] \
  || fail "$LOCAL_DATASET has no meta/info.json -- the conversion is unfinished or failed"

episodes="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('total_episodes'))" \
  "$LOCAL_DATASET/meta/info.json" 2>/dev/null || echo '?')"
echo "Local dataset : $LOCAL_DATASET (total_episodes=$episodes)"
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

# Bundle: source only. runtime/outputs/evaluations/.cache are produced ON kanu and
# must never be overwritten from laptop3.
rsync -av "${DRY[@]}" --checksum --partial \
  --exclude='runtime/' --exclude='outputs/' --exclude='evaluations/' \
  --exclude='.cache/' --exclude='__pycache__/' --exclude='*.pyc' \
  "$ROOT_DIR/" "$SSH_HOST:$REMOTE_ROOT/training/"

# Dataset: --delete only when explicitly asked (--delete-dataset), so an ordinary
# re-sync tops up rather than nuking a running job's input.
rsync -av "${DRY[@]}" "${DELETE_DATASET[@]}" --checksum --partial \
  "$LOCAL_DATASET/" "$SSH_HOST:$REMOTE_ROOT/dataset/$REMOTE_DATASET_NAME/"

echo
if [[ ${#DRY[@]} -gt 0 ]]; then
  echo "Dry run complete. Re-run without --dry-run to transfer."
  exit 0
fi
cat <<EOS
Transfer complete. On $SSH_HOST:

  cd ~/$REMOTE_ROOT/training
  nvidia-smi                                   # pick TWO idle GPUs, right now
  GPU_ID=<a> ./preflight.sh act
  GPU_ID=<b> ./preflight.sh fm
  DRY_RUN=1 GPU_ID=<a> ./train_act_carrot_eef.sh
  DRY_RUN=1 GPU_ID=<b> ./train_fm_carrot_eef.sh
  # launching is the integrator's call:
  # GPU_ID=<a> ./launch_detached.sh act
  # GPU_ID=<b> ./launch_detached.sh fm
EOS
