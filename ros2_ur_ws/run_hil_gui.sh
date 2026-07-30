#!/usr/bin/env bash
# PyQt5 operator GUI for the UR7e+GELLO HIL RL intervention deadman.
#
# This GUI is the GUI ALTERNATIVE to the terminal-focus spacebar deadman. It
# publishes /hil/deadman (std_msgs/Float32MultiArray, data=[engaged, gain]) at
# 20 Hz. It runs WITHOUT the teleop bridge and WITHOUT control_mode:=eef -- it
# never calls a bridge service and never commands the robot or GELLO directly.
#
# Run it ALONGSIDE the HIL RL loop, e.g.:
#     Terminal 1 (T3):  ./run_ur7e_gello_real.sh   # gello_publisher (the leader)
#     Terminal 2 (T4):  python3 serl_ur_infra/tests/run_rviz_hil.py --deadman topic
#     Terminal 3:       ./run_hil_gui.sh
#
# The GUI has the deadman toggle plus an episode START/NEXT button:
#   * click while DISENGAGED -> ENGAGE (two-click confirm; the arm WILL follow
#     the GELLO leader once you also move it).
#   * click while ENGAGED    -> DISENGAGE (single click; the RL policy resumes).
#   The sensitivity slider is the gain (0.10 fine .. 1.00 1:1); the env latches
#   it at the engage edge. The heartbeat publishes every tick (even disengaged)
#   so the env's staleness watchdog stays fed.
#   Actor reward/classifier/episode status arrives on /hil/actor_status. At
#   WAIT_SCENE_READY, START/NEXT releases the deadman and calls
#   /hil/scene_ready; it does not command the robot directly.
#
# A HIL session now STARTS DISENGAGED (policy control). Do not press ENGAGE to
# get the session going -- run_hil_actor.sh's deadman gate wants three fresh
# DISENGAGED heartbeats, which this GUI publishes as soon as it is up. ENGAGE is
# only for taking over with GELLO mid-episode. (Old behaviour:
# HIL_STARTUP_DEADMAN=engaged.)
#
# NO Qt font-warning filter here, on purpose: this GUI is PyQt5 against the
# SYSTEM Qt5 (fontconfig) and never imports cv2, so it does not emit the
# "QFontDatabase: Cannot find font directory .../cv2/qt/fonts" spam -- verified
# 2026-07-30 by running a QApplication+QLabel with QT_QPA_PLATFORM=offscreen.
# That message comes from the Qt plugin bundled in pip's opencv-python, so the
# filter lives where cv2 windows are actually opened: launch_cameras.sh (viewer)
# and run_hil_actor.sh (the actor's DISPLAY_IMAGE window).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source /opt/ros/humble/setup.bash
source "$SCRIPT_DIR/install/setup.bash"

exec ros2 run ur_gello_bringup gello_hil_gui
