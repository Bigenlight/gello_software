#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

exec "$HERE/run_sim_collect.sh" \
  --config "$HERE/configs/bowl_stack_sim.yaml" \
  --root "$REPO/ros2_ur_ws/gello_logs/sim/bowl_stack" \
  "$@"
