#!/usr/bin/env bash
# =============================================================================
# run_hil_server_kanu.sh -- TEST-BRANCH-ONLY: drive the learner on kanu
# =============================================================================
#
# ⚠️  THIS FILE EXISTS ONLY ON THE TEST BRANCH `test/kanu-learner-fallback`.
#     It must never be merged into feat/gello-ur7e-humble-22.04.
#
# WHY
#   junhyeong_ai's single RTX 5070 Ti is occupied indefinitely, so the HIL-SERL
#   learner temporarily runs on the OLD server `kanu` (8x RTX A4000, GPU 1 idle
#   as of 2026-08-10).  A slower learner used to be unacceptable because policy
#   inference sat inside the control loop; with HIL_POLICY_MODE=local it does
#   not, so kanu's ~460 ms learner step only makes published parameters ~25 s
#   stale instead of making the robot laggy.
#
# WHAT THIS IS
#   A thin env wrapper.  It adds FIVE environment variables and then execs the
#   real launcher with argv untouched, so `--check`, `--gpu`, `--run-id`,
#   `--new-lineage`, `--help` and everything else behave exactly as documented
#   in run_hil_server.sh.  It contains no logic of its own on purpose: every
#   guard (process contract, checkpoint SHA, GPU occupancy, cross-host HEAD
#   identity, lock file) stays in the one script that owns them.
#
#   Each variable uses ${VAR:-default}, so an explicit operator override still
#   wins:  HIL_GPU_INDEX=2 ./run_hil_server_kanu.sh   starts on GPU 2.
#
#   ⚠️  The deprecated aliases HIL_KANU_REPO / HIL_KANU_PYTHON are INERT here.
#   This wrapper always sets the canonical HIL_REMOTE_* names, and those take
#   precedence inside run_hil_server.sh, so an old alias in the operator's
#   shell is silently ignored rather than silently obeyed.  Use the canonical
#   names to override.
#
# USE
#   T1 (learner + tunnel):
#     HIL_PARAMS_EXPORT=1 HIL_EXTERNAL_POLICY_INGEST=1 HIL_LATENCY_PROFILE=1 \
#       ./run_hil_server_kanu.sh
#   Read-only status:
#     ./run_hil_server_kanu.sh --check
#
#   T3 (local inference) needs TWO variables, because a child's exports cannot
#   travel back up to its parent:
#     HIL_POLICY_MODE=local HIL_SSH_HOST=kanu \
#       HIL_SERVER_SCRIPT="$PWD/run_hil_server_kanu.sh" ./run_hil_session.sh
#   * HIL_SSH_HOST is what run_hil_local_policy.sh's ssh fallback and the
#     proxy's SshParamsFetcher (params_sync.SSH_HOST_ENV_VAR) read directly.
#   * HIL_SERVER_SCRIPT makes the authoritative "which run root is live"
#     probe -- `run_hil_server.sh --check` -- run through THIS wrapper, so it
#     asks kanu instead of junhyeong_ai.  Without it the probe would report
#     junhyeong_ai's run root and the proxy would then look for that path on
#     kanu.  This wrapper deliberately does NOT read HIL_SERVER_SCRIPT itself
#     (it always execs its own sibling), so that usage cannot recurse.
#
# REVERT
#   Delete nothing and undo nothing: `git checkout feat/gello-ur7e-humble-22.04`
#   on laptop3 (and on the kanu checkout) removes this file and restores the
#   junhyeong_ai defaults wholesale.  The classifier copy left under kanu's
#   ~/hil-serl-data/ is inert data and can stay.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ssh alias; verified reachable with BatchMode from laptop3.
export HIL_SSH_HOST="${HIL_SSH_HOST:-kanu}"
# kanu has 8 GPUs shared with other people.  GPU 1 and 2 were empty on
# 2026-08-10; 3/4/7 were saturated.  run_hil_server.sh refuses to start on a
# GPU that already has compute apps, so a stale choice fails loudly.
export HIL_GPU_INDEX="${HIL_GPU_INDEX:-1}"
# The stable symlink -> gello_software_hil_schema3_stage_20260730.  The remote
# side readlink -f's it before building the process contract.
export HIL_REMOTE_REPO="${HIL_REMOTE_REPO:-/home/junhyeong/gello_software_hil_current}"
# Same conda env name as junhyeong_ai's -- kanu's was the SOURCE of it.
export HIL_REMOTE_PYTHON="${HIL_REMOTE_PYTHON:-/home/junhyeong/miniconda3/envs/il/bin/python}"
# runs/, demos/, classifier_ckpt/ and the launcher lock, all under one root.
# Identical string to the junhyeong_ai default because both hosts run as the
# same user -- set it explicitly anyway so this file states the whole profile.
export HIL_REMOTE_DATA_ROOT="${HIL_REMOTE_DATA_ROOT:-/home/junhyeong/hil-serl-data}"

exec "$SCRIPT_DIR/run_hil_server.sh" "$@"
