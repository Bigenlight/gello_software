#!/usr/bin/env bash
# Persistent integrated GUI for EEF teleop data collection.
#
# Terminal 1 (safe to stop/restart after a collision):
#   HEADLESS=true ./run_ur7e_gello_real.sh control_mode:=eef
# Terminal 2 (leave this running across those restarts):
#   ./run_eef_collection_gui.sh
#
# Global keys anywhere in the integrated window:
#   R      recording START / STOP
#   SPACE  EEF ENGAGE / DISENGAGE (engage keeps the two-press confirmation)
#   G      gripper PAUSE / RESUME (resume keeps the two-press confirmation)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
GELLO_ROS_DISTRO="${GELLO_ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${GELLO_ROS_DISTRO}/setup.bash"
WORKSPACE_SETUP="$SCRIPT_DIR/install/setup.bash"
[[ -r "$ROS_SETUP" ]] || { echo "### ROS setup not found: $ROS_SETUP" >&2; exit 2; }
[[ -r "$WORKSPACE_SETUP" ]] || { echo "### Workspace is not built: $WORKSPACE_SETUP" >&2; exit 2; }
source "$ROS_SETUP"
source "$WORKSPACE_SETUP"
export PATH="/opt/ros/${GELLO_ROS_DISTRO}/bin:/usr/bin:/bin:${PATH}"

export START_POSE_CONFIG="${START_POSE_CONFIG:-$SCRIPT_DIR/src/gello_policy/config/ifql_deploy.yaml}"

exec ros2 run ur_gello_bringup eef_recorder_gui
