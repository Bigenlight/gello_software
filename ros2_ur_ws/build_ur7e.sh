#!/usr/bin/env bash
# build_ur7e.sh — build helper for the GELLO/UR7e ROS2 workspace
#
# Usage:
#   chmod +x build_ur7e.sh   # (one-time)
#   ./build_ur7e.sh
#
# Builds all three workspace packages by default: ur_gello_bringup (UR7e GELLO
# teleop), gello_policy (ACT/diffusion/FM/IFQL remote-policy deploy launch —
# needed by run_ur7e_ifql_real.sh), and gello_recorder (HDF5 + camera
# recorder). Override to build a subset, e.g.:
#   GELLO_BUILD_PACKAGES=ur_gello_bringup ./build_ur7e.sh
#   GELLO_BUILD_PACKAGES="ur_gello_bringup gello_policy" ./build_ur7e.sh
#
# NOTE: dynamixel_sdk is NOT installed via apt/rosdep here — it is provided by
# `pip install --user dynamixel-sdk` (v4.0.5; no sudo/apt needed). It is only
# required for the real GELLO leader arm (source:=gello); the fake/mock verify
# path (source:=fake) does NOT need it. We therefore --skip-keys dynamixel_sdk
# in the rosdep call below so it does not try to apt-install ros-jazzy-dynamixel-sdk
# (setup_jazzy/install_jazzy.sh installs that apt package separately anyway;
# --skip-keys here just keeps rosdep from treating it as a hard blocker if it
# hasn't been installed yet on a fresh checkout).
set -euo pipefail

# Resolve the ros2_ur_ws directory regardless of caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Declared explicitly rather than trusted from $ROS_DISTRO: that var is
# EXPORTED by /opt/ros/<distro>/setup.bash itself, so a shell that already
# sourced a DIFFERENT distro's setup.bash earlier would make this script
# source the wrong one if it just read $ROS_DISTRO. An explicit var sidesteps
# that ambiguity.
GELLO_ROS_DISTRO="${GELLO_ROS_DISTRO:-jazzy}"

# Packages to build. Space-separated; passed straight to --packages-select.
GELLO_BUILD_PACKAGES="${GELLO_BUILD_PACKAGES:-ur_gello_bringup gello_policy gello_recorder}"

# --- conda guard --------------------------------------------------------------
# This machine auto-activates miniconda `base`, whose python3 is 3.13. Jazzy's
# rclpy/colcon C extensions are built against the system 3.12 -- a `python3`
# that resolves into miniconda silently breaks `rosdep` and `colcon build`
# (wrong ABI, or import errors that only show up at ros2-launch time). Same
# guard as setup_jazzy/install_jazzy.sh: strip miniconda from PATH (do not
# just check-and-warn; that install script found a plain warning was not
# enough since colcon build would still silently pick up the wrong python3),
# then refuse outright if python3 still isn't the system one.
if [ "${CONDA_DEFAULT_ENV:-}" != "" ]; then
    echo "### conda env '${CONDA_DEFAULT_ENV}' is active -- removing miniconda from PATH for this script"
fi
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)"
export PATH
echo "### python3 -> $(command -v python3)  ($(python3 --version 2>&1))"
case "$(command -v python3)" in
    /usr/bin/python3) ;;
    *) echo "ERROR: python3 is not /usr/bin/python3 (got '$(command -v python3)'). Refusing to continue." >&2; exit 1 ;;
esac

# ROS Jazzy's setup scripts read optional variables before defining them
# (AMENT_TRACE_SETUP_FILES, AMENT_PYTHON_EXECUTABLE, ...), so they abort under
# `set -u`. Keep strict nounset checking for the rest of this script and
# disable it only while the upstream/generated environment scripts are sourced.
# Same treatment as run_ur7e_diffusion_remote.sh.
set +u
source "/opt/ros/${GELLO_ROS_DISTRO}/setup.bash"
set -u

# --from-paths src scans the WHOLE workspace regardless of which packages are
# selected below for colcon build, so all three packages' rosdep keys are
# resolved together here even when GELLO_BUILD_PACKAGES only builds a subset.
rosdep install --from-paths src --ignore-src -r -y --skip-keys dynamixel_sdk

# shellcheck disable=SC2086 # GELLO_BUILD_PACKAGES is intentionally word-split
colcon build --symlink-install --packages-select ${GELLO_BUILD_PACKAGES}

echo ""
echo "Build complete (packages: ${GELLO_BUILD_PACKAGES}). To verify (fake/mock hardware, no GELLO serial needed):"
echo ""
echo "  source \"$SCRIPT_DIR/install/setup.bash\""
echo "  ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake launch_rviz:=false"
