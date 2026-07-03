#!/usr/bin/env bash
# Drive the REAL UR7e from the PHYSICAL GELLO leader arm over ROS2 Humble.
#
# ############################################################################
# #  ⚠️  REAL HARDWARE — THE UR7e WILL PHYSICALLY MOVE.                       #
# #                                                                          #
# #  Before running:                                                         #
# #   1) Workspace clear; keep the teach-pendant E-STOP within reach.        #
# #   2) START THE ROBOT program — pick ONE method (do NOT mix them):        #
# #      • Method A (default, HEADLESS unset/false): on the pendant, load +   #
# #        PLAY the External Control program (PolyScope 5 classic URCap).     #
# #        The driver only moves the arm while External Control is running.   #
# #      • Method B (HEADLESS=true): the driver sends URScript directly, so   #
# #        NO External Control program and NO pendant Play is needed. This    #
# #        REQUIRES the robot to be in REMOTE mode (see one-time setup).      #
# #   3) Hold the GELLO leader at a sensible, near-neutral pose. On startup  #
# #      the robot smoothly drives TO the GELLO's current pose (move-to-     #
# #      start handshake) — so wherever GELLO is, the robot goes there.      #
# #                                                                          #
# #  Auto-sequence: driver(t=0) -> gello_publisher(t=6s, TimerAction) ->     #
# #  gello_move_to_start handshake(t=8s, smooth trajectory to GELLO pose) -> #
# #  STRICT switch to forward_position_controller -> once the handshake      #
# #  process EXITS with returncode==0, gello_ur_bridge streaming + the       #
# #  Robotiq 2F-85 gripper (Modbus over the driver's shared /tmp/ttyUR socat #
# #  bridge) + gello_gripper_bridge start together immediately (event-driven #
# #  via OnProcessExit, NOT a fixed timer). The arm STARTS MOVING ~t=8s.     #
# #                                                                          #
# #  GRIPPER IS NOW INCLUDED: the 2F-85 is driven from the GELLO gripper     #
# #  axis (closing the GELLO hand closes the robot gripper). The ROBOT MUST  #
# #  BE POWERED ON. TOOL VOLTAGE is supplied by the DRIVER (tool_voltage:=24 #
# #  in the launch), NOT the pendant Installation tab — the driver also runs #
# #  the tool_communication socat forwarder that owns :54321 and the gripper #
# #  shares as /tmp/ttyUR. A powered-off robot gives no Modbus response and  #
# #  the gripper simply won't move. On connect the gripper auto-calibrates   #
# #  (open/close sweep) — KEEP FINGERS CLEAR. Ctrl-C for a clean shutdown so #
# #  the driver releases :54321 / /tmp/ttyUR for the next launch.            #
# #                                                                          #
# #  ONE-TIME CHECK (do once, do not block on it): after Playing External    #
# #  Control, confirm /robotiq_gripper/position_percent keeps updating —     #
# #  i.e. that driver-applied tool voltage survives EC Play. If it cuts out, #
# #  see GELLO_UR7E_GRIPPER.md for the fallback.                             #
# #                                                                          #
# #  GELLO stays PASSIVE read-only throughout (no torque to its motors).     #
# #  Abort anytime: Ctrl-C here, and/or E-STOP on the pendant.               #
# ############################################################################
#
# ############################################################################
# #  ONE-TIME PENDANT SETUP for HEADLESS=true (Remote mode / Method B)        #
# #                                                                          #
# #   1) Settings > System > Remote Control > Enable.                        #
# #   2) Use the Local/Remote toggle in the top-right header to switch to    #
# #      REMOTE. (Pendant Play/Load buttons go grey — that is normal; ROS2   #
# #      drives the robot now. Switch back to Local ONLY from the pendant.)  #
# #   3) Ensure the robot is set to REAL ROBOT, not Simulation (bottom-right #
# #      of the pendant) — in Simulation the physical arm will not move.     #
# #                                                                          #
# #   If a dashboard/reverse-interface drop occurs mid-run, recover with:    #
# #      HEADLESS: /io_and_status_controller/resend_robot_program (Trigger)   #
# #      Method A: /dashboard_client/play (Trigger)                          #
# ############################################################################
#
# ############################################################################
# #  USAGE                                                                    #
# #                                                                          #
# #    # Method A (default): press Play on the pendant's External Control.    #
# #    ./run_ur7e_gello_real.sh                                              #
# #                                                                          #
# #    # Method B (headless, no pendant Play — needs REMOTE mode):           #
# #    HEADLESS=true ./run_ur7e_gello_real.sh                               #
# #                                                                          #
# #    # Override the robot IP (default 192.168.10.11):                      #
# #    ROBOT_IP=192.168.1.50 ./run_ur7e_gello_real.sh                       #
# #                                                                          #
# #    # Supply a per-robot kinematics calibration YAML:                     #
# #    CALIB=/path/to/ur7e_calibration.yaml ./run_ur7e_gello_real.sh        #
# #                                                                          #
# #    # Combine env toggles (and pass extra launch args after --):          #
# #    HEADLESS=true ROBOT_IP=192.168.10.11 ./run_ur7e_gello_real.sh \       #
# #        launch_rviz:=false                                                #
# ############################################################################
set -e

ROBOT_IP="${ROBOT_IP:-192.168.10.11}"     # override: ROBOT_IP=x.x.x.x ./run_ur7e_gello_real.sh
CALIB="${CALIB:-}"                          # optional: CALIB=/path/ur7e_calibration.yaml
HEADLESS="${HEADLESS:-}"                     # HEADLESS=true|1 -> Method B (no pendant Play; needs REMOTE mode)

# Also honor headless_mode:=true passed as a launch arg so the banner below can
# never disagree with the effective launch (ros2 launch is last-wins on dupes).
case " $* " in *" headless_mode:=true "*|*"headless_mode:=true"*) HEADLESS=true ;; esac

export GELLO_REPO_ROOT=/home/laptop3/gello_software
source /opt/ros/humble/setup.bash
source /home/laptop3/gello_software/ros2_ur_ws/install/setup.bash

ARGS=(robot_ip:="${ROBOT_IP}")
[ -n "${CALIB}" ] && ARGS+=(kinematics_params_file:="${CALIB}")

# Method B (headless): driver sends URScript directly; robot MUST be in REMOTE
# mode on the pendant. Default (HEADLESS unset) keeps headless_mode:=false so
# the External Control program must be PLAYING on the pendant (Method A).
if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    ARGS+=(headless_mode:=true)
    HEADLESS_STATE="true (Method B — REMOTE mode required, no pendant Play)"
else
    ARGS+=(headless_mode:=false)
    HEADLESS_STATE="false (Method A — PLAY External Control on the pendant)"
fi

echo "### REAL UR7e teleop | robot_ip=${ROBOT_IP} | calib=${CALIB:-<none>}"
echo "### headless_mode=${HEADLESS_STATE}"
echo "### Robotiq 2F-85 gripper INCLUDED (Modbus over driver socat bridge /tmp/ttyUR)."
echo "### ROBOT MUST BE POWERED ON. Tool voltage is supplied by the DRIVER"
echo "### (tool_voltage:=24), NOT the pendant Installation tab. Keep fingers clear —"
echo "### the gripper auto-cal sweeps on connect. Ctrl-C to release :54321 / /tmp/ttyUR."
echo "### ONE-TIME CHECK: after Play EC, confirm /robotiq_gripper/position_percent updates."
if [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "1" ]; then
    echo "### HEADLESS: ensure the pendant is in REMOTE mode + Real Robot (not Simulation); the arm will move (~t=8s)."
else
    echo "### Confirm External Control is PLAYING on the pendant, then the arm will move (~t=8s)."
fi
exec ros2 launch ur_gello_bringup ur7e_gello_real.launch.py "${ARGS[@]}" "$@"
