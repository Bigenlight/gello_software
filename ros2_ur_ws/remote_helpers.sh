#!/usr/bin/env bash
# remote_helpers.sh — copy-pasteable ROS2 helpers for the UR7e Remote-mode workflow.
#
#   chmod +x remote_helpers.sh    # (optional; this file is meant to be SOURCED)
#   source remote_helpers.sh      # then call:  ur_mode / ur_powerup / ur_load / ur_play ...
#
# ############################################################################
# #  METHOD A  vs  METHOD B  —  PICK ONE, DO NOT MIX IN A SINGLE RUN.        #
# #                                                                          #
# #  Both require the robot to be in REMOTE mode on the pendant             #
# #  (Settings > System > Remote Control > Enable, then toggle to Remote)   #
# #  AND set to Real Robot (bottom-right of the pendant, NOT Simulation).   #
# #  Remote->Local can only be switched back AT the pendant (safety).       #
# #                                                                          #
# #  METHOD A — External Control URCap program + Dashboard remote Play:     #
# #     Keep an External Control program (e.g. ur_caps.urp) on the pendant. #
# #     ROS2 presses Play for you via the dashboard services:               #
# #        ur_load ur_caps.urp   # load_program (ur_dashboard_msgs/srv/Load)#
# #        ur_play              # play         (std_srvs/srv/Trigger)       #
# #     On a reverse-interface drop, recover with:  ur_play                 #
# #                                                                          #
# #  METHOD B — Headless mode (no URCap program, no Play needed) ⭐:         #
# #     Launch the driver with headless_mode:=true; it streams URScript     #
# #     directly and auto-connects at startup. Our real launch already      #
# #     exposes this:                                                       #
# #        HEADLESS=true ROBOT_IP=192.168.10.11 ./run_ur7e_gello_real.sh
# #     (or the bare driver:  ur_headless_driver 192.168.10.11)             #
# #     On a reverse-interface drop, recover with:  ur_resend               #
# #                                                                          #
# #  ⚠️  Do NOT run Method A load/play AND Method B headless in the same    #
# #      session — the two ways of injecting URScript fight each other.     #
# #                                                                          #
# #  SERVICE TYPES (must match exactly, or the call is rejected):           #
# #     get_robot_mode          -> ur_dashboard_msgs/srv/GetRobotMode  {}   #
# #     load_program            -> ur_dashboard_msgs/srv/Load  {filename:}  #
# #     power_on / brake_release / play / stop /                            #
# #       unlock_protective_stop / resend_robot_program                     #
# #                              -> std_srvs/srv/Trigger  {}                 #
# ############################################################################
#
#  ⚠️  SAFETY: play / brake_release / the teleop launch make the ARM MOVE.
#      Keep the pendant E-STOP in reach. These helpers only wrap ROS2 calls;
#      none of them are auto-run on source.

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_RH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$_RH_DIR/.." && pwd)}"
export ROBOT_IP="${ROBOT_IP:-192.168.10.11}"

# Overlay names (kept as functions/vars so they are easy to re-source).
_UR_WS_SETUP="${_RH_DIR}/install/setup.bash"

# Source ROS2 + workspace overlay (idempotent; safe to call repeatedly).
ur_env() {
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  # shellcheck disable=SC1091
  [ -f "${_UR_WS_SETUP}" ] && source "${_UR_WS_SETUP}"
}
ur_env

# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------

# ur_mode — query the robot mode (POWER_OFF / IDLE / RUNNING ...).
#   Service type: ur_dashboard_msgs/srv/GetRobotMode
ur_mode() {
  ur_env
  ros2 service call /dashboard_client/get_robot_mode \
    ur_dashboard_msgs/srv/GetRobotMode "{}"
}

# ur_program — show which program is loaded on the pendant.
#   Service type: ur_dashboard_msgs/srv/GetLoadedProgram
ur_program() {
  ur_env
  ros2 service call /dashboard_client/get_loaded_program \
    ur_dashboard_msgs/srv/GetLoadedProgram "{}"
}

# ---------------------------------------------------------------------------
# POWER  (needed once after a cold controller / E-STOP; ⚠️ brake_release moves)
# ---------------------------------------------------------------------------

# ur_poweron — energise the robot (motors on, brakes still engaged).
ur_poweron() {
  ur_env
  ros2 service call /dashboard_client/power_on std_srvs/srv/Trigger "{}"
}

# ur_brake_release — release the joint brakes. ⚠️ Small settle motion possible.
ur_brake_release() {
  ur_env
  ros2 service call /dashboard_client/brake_release std_srvs/srv/Trigger "{}"
}

# ur_powerup — power_on THEN brake_release, the usual cold-start sequence.
#   ⚠️ Robot becomes energised and brakes release. Workspace clear, E-STOP ready.
ur_powerup() {
  ur_env
  echo "### power_on ..."
  ros2 service call /dashboard_client/power_on std_srvs/srv/Trigger "{}"
  echo "### brake_release (⚠️ brakes releasing) ..."
  ros2 service call /dashboard_client/brake_release std_srvs/srv/Trigger "{}"
  echo "### done — check ur_mode reports RUNNING."
}

# ---------------------------------------------------------------------------
# METHOD A — External Control program + remote Play
# ---------------------------------------------------------------------------

# ur_load PROG — load a .urp program on the pendant (must already exist there).
#   Service type: ur_dashboard_msgs/srv/Load  (filename field)
#   Example:  ur_load ur_caps.urp
ur_load() {
  local prog="${1:-}"
  if [ -z "${prog}" ]; then
    echo "usage: ur_load <program>.urp   (e.g. ur_load ur_caps.urp)" >&2
    return 2
  fi
  ur_env
  ros2 service call /dashboard_client/load_program \
    ur_dashboard_msgs/srv/Load "{filename: ${prog}}"
}

# ur_play — press Play remotely (Method A). ⚠️ Starts the program; arm may move.
#   Also the Method-A recovery on a reverse-interface drop.
#   Service type: std_srvs/srv/Trigger
ur_play() {
  ur_env
  ros2 service call /dashboard_client/play std_srvs/srv/Trigger "{}"
}

# ur_stop — stop the running program on the pendant.
#   Service type: std_srvs/srv/Trigger
ur_stop() {
  ur_env
  ros2 service call /dashboard_client/stop std_srvs/srv/Trigger "{}"
}

# ---------------------------------------------------------------------------
# METHOD B — Headless recovery
# ---------------------------------------------------------------------------

# ur_resend — re-send the headless robot program after a reverse-interface drop
#   ("Connection to reverse interface dropped."). Method-B recovery ONLY.
#   Service type: std_srvs/srv/Trigger
ur_resend() {
  ur_env
  ros2 service call /io_and_status_controller/resend_robot_program \
    std_srvs/srv/Trigger "{}"
}

# ---------------------------------------------------------------------------
# FAULT RECOVERY
# ---------------------------------------------------------------------------

# ur_unlock — clear a protective stop (fix the cause first, then unlock).
#   Service type: std_srvs/srv/Trigger
ur_unlock() {
  ur_env
  ros2 service call /dashboard_client/unlock_protective_stop \
    std_srvs/srv/Trigger "{}"
}

# ---------------------------------------------------------------------------
# BARE HEADLESS DRIVER (Method B, NO teleop nodes => NO motion)
# ---------------------------------------------------------------------------

# ur_headless_driver [IP] — bring up ONLY the UR driver in headless+Remote mode.
#   No gello nodes are started, so the arm does NOT move; use this to verify the
#   driver reaches "Robot ready to receive control commands." Requires Remote
#   mode on the pendant.  Example:  ur_headless_driver 192.168.10.11
ur_headless_driver() {
  local ip="${1:-${ROBOT_IP}}"
  ur_env
  echo "### bare headless driver | ur_type=ur7e robot_ip=${ip} (no teleop, no motion)"
  ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur7e robot_ip:="${ip}" headless_mode:=true launch_rviz:=false
}

# ---------------------------------------------------------------------------
# Quick reference
# ---------------------------------------------------------------------------
ur_help() {
  cat <<'EOF'
UR7e Remote-mode helpers (source remote_helpers.sh):

  status
    ur_mode              robot mode (GetRobotMode)
    ur_program           loaded program (GetLoadedProgram)

  power (⚠️ brake_release moves)
    ur_poweron           motors on
    ur_brake_release     release brakes
    ur_powerup           power_on + brake_release

  Method A (URCap program + remote Play)
    ur_load PROG.urp     load a program (Load, filename field)
    ur_play              press Play  (⚠️ moves) / Method-A drop recovery
    ur_stop              stop program

  Method B (headless)
    ur_headless_driver [IP]   bare driver, headless+Remote, NO teleop (no motion)
    ur_resend            re-send robot program / Method-B drop recovery

  fault recovery
    ur_unlock            clear protective stop

  full hands-free teleop (headless, ⚠️ arm moves ~t=8s):
    HEADLESS=true ROBOT_IP=192.168.10.11 ./run_ur7e_gello_real.sh

  Prereq for BOTH methods: pendant in REMOTE mode + Real Robot (not Simulation).
  ⚠️ Do NOT mix Method A (load/play) with Method B (headless) in one run.
EOF
}

echo "remote_helpers.sh sourced. Run 'ur_help' for commands. (Nothing was sent to the robot.)"
