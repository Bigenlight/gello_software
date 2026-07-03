#!/usr/bin/env bash
# build_ur7e.sh — build helper for ur_gello_bringup (UR7e GELLO teleop)
#
# Usage:
#   chmod +x build_ur7e.sh   # (one-time)
#   ./build_ur7e.sh
#
# NOTE: dynamixel_sdk is NOT installed via apt/rosdep here — it is provided by
# `pip install --user dynamixel-sdk` (v4.0.5; no sudo/apt needed). It is only
# required for the real GELLO leader arm (source:=gello); the fake/mock verify
# path (source:=fake) does NOT need it. We therefore --skip-keys dynamixel_sdk
# in the rosdep call below so it does not try to apt-install ros-humble-dynamixel-sdk.
set -euo pipefail

# Resolve the ros2_ur_ws directory regardless of caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /opt/ros/humble/setup.bash

rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk

colcon build --packages-select ur_gello_bringup

echo ""
echo "Build complete. To verify (fake/mock hardware, no GELLO serial needed):"
echo ""
echo "  source \"$SCRIPT_DIR/install/setup.bash\""
echo "  ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=false"
