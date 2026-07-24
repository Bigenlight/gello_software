#!/usr/bin/env bash
# Drive a MOCK UR7e in RViz2 through the START-ANCHORED JOINT-DELTA bridge mode.
#
# MOCK ONLY — this script can never touch a real robot: the launch it execs
# forces use_fake_hardware:=true (ros2_control mock hardware mirrors position
# commands back into the joint states, so RViz moves and nothing can be
# damaged). There is deliberately NO real-robot sibling script: the real path
# already exists and forwards extra arguments verbatim, so use
#
#   ./run_ur7e_gello_real.sh control_mode:=joint_delta jd_gain:=0.0
#
# GELLO is PASSIVE read-only: torque is never enabled on its Dynamixels.
#
# What joint_delta mode does: instead of MIRRORING the leader's absolute joints
# (q_cmd = q_leader) the arm follows how much the leader has MOVED since an
# anchor:  q_cmd = q_robot_anchor + jd_gain * (leader travel since engage).
# See docs/ros2/GELLO_UR7E_JOINT_DELTA_MODE.md.
#
# Env passthroughs:
#   SOURCE   fake|gello   -> source:=      (default: the launch default, 'fake')
#   PATTERN  <name>       -> pattern:=     (fake_gello motion; 'hold' / 'step' /
#                                           'full_rotation' are the useful ones)
#   JD_GAIN  <float>      -> jd_gain:=     (0.0 = the arm must not move at all)
#
# Usage:
#   ./run_ur7e_gello_joint_delta_mock.sh
#   PATTERN=hold JD_GAIN=0.0 ./run_ur7e_gello_joint_delta_mock.sh
#   SOURCE=gello ./run_ur7e_gello_joint_delta_mock.sh launch_rviz:=false
#   PATTERN=full_rotation ./run_ur7e_gello_joint_delta_mock.sh   # past-pi check
#
# Once it is up (bridge boots in JOINT_BOOTSTRAP = ordinary joint passthrough):
#   ros2 topic echo /gello_ur_bridge/joint_delta/state
#   ros2 service call /gello_ur_bridge/joint_delta_engage   std_srvs/srv/Trigger
#   ros2 service call /gello_ur_bridge/joint_delta_clutch   std_srvs/srv/Trigger
#   ros2 service call /gello_ur_bridge/joint_delta_reclutch std_srvs/srv/Trigger
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

ARGS=()
[ -n "${SOURCE}" ]  && ARGS+=(source:="${SOURCE}")
[ -n "${PATTERN}" ] && ARGS+=(pattern:="${PATTERN}")
[ -n "${JD_GAIN}" ] && ARGS+=(jd_gain:="${JD_GAIN}")

exec ros2 launch ur_gello_bringup ur7e_gello_joint_delta_mock.launch.py \
    "${ARGS[@]}" "$@"
