#!/usr/bin/env bash
# Numbered-menu operator console for the init_align safety handshake.
#
# Run this in a SECOND terminal while the real teleop is running with
# START_MODE=init_align, e.g.:
#     Terminal 1:  START_MODE=init_align ./run_ur7e_gello_real.sh
#     Terminal 2:  ./run_operator_console.sh
#
# Then just type numbers to authorize each safety gate:
#     1) 진행 (proceed)        2) 정지 (abort)        3) 강제 진행 (override)
#
# Pressing 1 when GELLO is not aligned prints which joint is off and by how
# much (so a large lone offset, e.g. the base, cannot silently drive a big move).
# READ/authorize-only: relays your authorization to gello_move_to_start; it does
# not command the robot or GELLO directly.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

exec ros2 run ur_gello_bringup gello_operator_console
