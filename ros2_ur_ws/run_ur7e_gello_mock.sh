#!/usr/bin/env bash
# Rehearse the REAL-ROBOT bring-up sequence against ros2_control MOCK hardware,
# driven by the PHYSICAL GELLO leader. NO real robot is ever contacted.
#
# ############################################################################
# #  MOCK — this script CANNOT reach the real UR7e.                          #
# #                                                                          #
# #  It runs the SAME launch file as ./run_ur7e_gello_real.sh                #
# #  (ur7e_gello_real.launch.py) — driver -> gello_publisher ->               #
# #  gello_move_to_start handshake -> STRICT controller switch ->            #
# #  gello_ur_bridge streaming — but with                                    #
# #                                                                          #
# #        use_fake_hardware:=true   robot_ip:=127.0.0.1                      #
# #                                                                          #
# #  BAKED IN and NON-OVERRIDABLE. Any robot_ip:=... or                      #
# #  use_fake_hardware:=false on the command line (or a ROBOT_IP env var     #
# #  that is not the loopback) makes this script REFUSE and exit before      #
# #  `ros2 launch` runs.                                                     #
# #                                                                          #
# #  WHY THIS EXISTS: ./run_ur7e_gello_real.sh defaults to                   #
# #  ROBOT_IP=192.168.10.11 — forget `use_fake_hardware:=true` once and you  #
# #  are on the live arm. ./run_ur7e_gello_sim.sh uses                       #
# #  ur7e_gello_rviz.launch.py, which has NO move-to-start handshake, so it  #
# #  cannot rehearse the real start-up order. This script closes both gaps.  #
# #                                                                          #
# #  What the launch file does under use_fake_hardware:=true (its own       #
# #  logic, not ours): headless_mode is FORCED true (no pendant to wait on), #
# #  use_tool_communication is FORCED false and the Robotiq 2F-85 gripper   #
# #  is SKIPPED after the handshake (Modbus needs a powered real robot).    #
# #  GELLO stays PASSIVE read-only throughout (no torque to its motors).    #
# ############################################################################
#
# USAGE
#   ./run_ur7e_gello_mock.sh                          # control_mode joint, RViz on
#   ./run_ur7e_gello_mock.sh control_mode:=eef        # switch_only bring-up, arm holds
#   ./run_ur7e_gello_mock.sh control_mode:=joint_delta jd_gain:=0.0
#   ./run_ur7e_gello_mock.sh launch_rviz:=false       # headless (no DISPLAY needed)
#   ./run_ur7e_gello_mock.sh --print-args             # show the final ros2 launch, exit
#   DRY_RUN=1 ./run_ur7e_gello_mock.sh control_mode:=eef   # same as --print-args
#
#   source:=gello (default) reads the PHYSICAL GELLO on /dev/ttyUSB0.
#   source:=fake is NOT supported by ur7e_gello_real.launch.py (it declares no
#   `source` argument and always spawns the real gello_publisher); the script
#   explains which synthetic-leader launch to use instead and exits.
#
#   Every other key:=value is forwarded to the launch verbatim (start_mode,
#   gripper_mode, pos_scale, v_max, w_max, jd_gain, params_file, ...). The
#   gripper_* ones are range-checked by the launch but the gripper itself never
#   starts here.
#
# RVIZ
#   The official ur_control.launch.py HARDCODES ur_description's view_robot.rviz
#   with no override argument (same reason ./run_mock_rviz.sh exists), so the
#   launch always gets launch_rviz:=false and we start rviz2 ourselves with the
#   operator-oriented rviz/hil_operator_view.rviz, tied to this script's life.
#   Pass launch_rviz:=false to skip it. NOTE the mesh you see is the UR5e one:
#   ur_description ships no meshes/ur7e and config/ur7e/visual_parameters.yaml
#   points at meshes/ur5e/visual/*.dae. Kinematics are ur7e; only the skin is
#   borrowed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# --------------------------------------------------------------------------- #
# Pinned, non-overridable launch values. These are the whole point.
# --------------------------------------------------------------------------- #
readonly MOCK_ROBOT_IP="127.0.0.1"
readonly RVIZ_CONFIG="$SCRIPT_DIR/rviz/hil_operator_view.rviz"

die() { echo "### run_ur7e_gello_mock.sh: REFUSED — $*" >&2; exit 2; }

# ROBOT_IP is the real wrapper's env knob. Here it must not exist, or must be
# the loopback — anything else means someone is holding a real-robot habit.
if [ -n "${ROBOT_IP:-}" ] && [ "${ROBOT_IP}" != "${MOCK_ROBOT_IP}" ]; then
    die "ROBOT_IP=${ROBOT_IP} is set in the environment. This script only ever" \
        "talks to ${MOCK_ROBOT_IP} (mock hardware). Unset ROBOT_IP, or use" \
        "./run_ur7e_gello_real.sh if you really mean the real arm."
fi

# --------------------------------------------------------------------------- #
# Argument scan. We consume: robot_ip, use_fake_hardware, source, control_mode,
# launch_rviz, --print-args. Everything else is forwarded verbatim.
# --------------------------------------------------------------------------- #
DRY_RUN="${DRY_RUN:-0}"
CONTROL_MODE=joint
SOURCE=gello
LAUNCH_RVIZ=true
PASS=()
for _a in "$@"; do
    case "$_a" in
        --print-args|--dry-run)
            DRY_RUN=1 ;;
        robot_ip:=*)
            die "'$_a' on the command line. robot_ip is pinned to ${MOCK_ROBOT_IP}" \
                "here and cannot be overridden — that is the real-robot escape" \
                "hatch this script exists to close. Use ./run_ur7e_gello_real.sh." ;;
        use_fake_hardware:=true)
            ;;  # redundant, already pinned; drop the duplicate (ros2 launch is last-wins)
        use_fake_hardware:=*)
            die "'$_a' on the command line. use_fake_hardware is pinned to true" \
                "here. Use ./run_ur7e_gello_real.sh for real hardware." ;;
        source:=*)
            SOURCE="${_a#source:=}" ;;
        control_mode:=*)
            CONTROL_MODE="${_a#control_mode:=}"
            PASS+=("$_a") ;;
        launch_rviz:=*)
            LAUNCH_RVIZ="${_a#launch_rviz:=}" ;;
        headless_mode:=*)
            # The launch forces headless_mode:=true under fake hardware regardless
            # of this value (PythonExpression in ur7e_gello_real.launch.py), so
            # forwarding it only invites a banner that disagrees with reality.
            echo "### NOTE: '$_a' ignored — headless_mode is forced true under mock hardware." >&2 ;;
        *)
            PASS+=("$_a") ;;
    esac
done

case "${CONTROL_MODE}" in
    joint|eef|joint_delta) ;;
    *) die "control_mode:=${CONTROL_MODE} — must be joint | eef | joint_delta." ;;
esac
case "${LAUNCH_RVIZ}" in
    true|false) ;;
    *) die "launch_rviz:=${LAUNCH_RVIZ} — must be true | false." ;;
esac

case "${SOURCE}" in
    gello) ;;
    fake)
        cat >&2 <<EOF
### run_ur7e_gello_mock.sh: source:=fake is NOT available on this path.
###
###   ur7e_gello_real.launch.py declares NO 'source' launch argument: it always
###   spawns the physical-GELLO reader (gello_publisher on /dev/ttyUSB0) because
###   its whole purpose is to rehearse the REAL start-up order — move-to-start
###   handshake chasing a LIVE leader pose, STRICT controller switch, bridge
###   resume. A synthetic leader would make that rehearsal meaningless.
###
###   For a fully synthetic leader (no GELLO plugged in) use instead:
###     joint       : ./run_ur7e_gello_sim.sh                 (ur7e_gello_rviz.launch.py, source:=fake)
###                   -> ros2 launch ur_gello_bringup ur7e_gello_rviz.launch.py source:=fake
###     eef         : ros2 launch ur_gello_bringup ur7e_gello_eef_mock.launch.py
###     joint_delta : ./run_ur7e_gello_joint_delta_mock.sh    (SOURCE=fake is its default)
###   None of those runs the move-to-start handshake; that is the trade-off.
EOF
        exit 2 ;;
    *) die "source:=${SOURCE} — must be gello (physical leader) or fake (see message for fake)." ;;
esac

# --------------------------------------------------------------------------- #
# ROS sourcing — same as the other run_*.sh wrappers.
# --------------------------------------------------------------------------- #
set +u
GELLO_ROS_DISTRO="${GELLO_ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${GELLO_ROS_DISTRO}/setup.bash"
WORKSPACE_SETUP="$SCRIPT_DIR/install/setup.bash"
[[ -r "$ROS_SETUP" ]] || die "ROS setup not found: $ROS_SETUP"
[[ -r "$WORKSPACE_SETUP" ]] || die "Workspace is not built: $WORKSPACE_SETUP"
source "$ROS_SETUP"
source "$WORKSPACE_SETUP"
export PATH="/opt/ros/${GELLO_ROS_DISTRO}/bin:/usr/bin:/bin:${PATH}"
set -u

# The launch always gets launch_rviz:=false (see RVIZ note in the header); our
# own rviz2 below honours the user's launch_rviz value.
LAUNCH_CMD=(ros2 launch ur_gello_bringup ur7e_gello_real.launch.py
    robot_ip:="${MOCK_ROBOT_IP}" use_fake_hardware:=true launch_rviz:=false
    "${PASS[@]}")

if [ "${DRY_RUN}" = "1" ]; then
    echo "### DRY RUN — nothing launched. Final command:"
    printf '  GELLO_REPO_ROOT=%q' "$GELLO_REPO_ROOT"
    printf ' %q' "${LAUNCH_CMD[@]}"
    echo
    if [ "${LAUNCH_RVIZ}" = "true" ]; then
        printf '  + rviz2 -d %q   (own process, killed with this script)\n' "$RVIZ_CONFIG"
    else
        echo "  + no rviz2 (launch_rviz:=false)"
    fi
    echo "### control_mode=${CONTROL_MODE} source=${SOURCE}"
    exit 0
fi

# --------------------------------------------------------------------------- #
# Banner. The start_mode wording mirrors what the launch auto-derives from
# control_mode (joint -> gello chase; eef/joint_delta -> switch_only).
# --------------------------------------------------------------------------- #
EFF_START_MODE=gello
for _a in "${PASS[@]}"; do case "$_a" in start_mode:=*) EFF_START_MODE="${_a#start_mode:=} (explicit)" ;; esac; done
if [ "${EFF_START_MODE}" = "gello" ] && [ "${CONTROL_MODE}" != "joint" ]; then
    EFF_START_MODE=switch_only
fi

echo "##########################################################################"
echo "###  MOCK UR7e  —  NO REAL-ROBOT CONNECTION (robot_ip=${MOCK_ROBOT_IP}, use_fake_hardware=true)"
echo "###  Physical GELLO: READ-ONLY leader (torque never enabled) on /dev/ttyUSB0"
echo "###  Gripper: NONE (Robotiq skipped by the launch under mock hardware)"
echo "###  RViz mesh: UR5e skin on ur7e kinematics (ur_description has no ur7e meshes)"
echo "###  Same launch + same start-up order as ./run_ur7e_gello_real.sh:"
echo "###    driver(t=0) -> gello_publisher + PAUSED bridge(t=6s) -> gello_move_to_start(t=8s)"
echo "###    -> STRICT switch scaled_joint_trajectory_controller -> forward_position_controller"
echo "###    -> bridge resume -> streaming on /forward_position_controller/commands"
echo "###  control_mode=${CONTROL_MODE} | start_mode=${EFF_START_MODE} | headless_mode=true (forced)"
case "${CONTROL_MODE}" in
    eef)
        echo "###  EEF: switch_only bring-up — the mock arm does NOT move at start-up; the"
        echo "###  bridge holds until eef_engage. Drive it from ANOTHER terminal with:"
        echo "###        ./run_eef_gui.sh"
        echo "###  (or: ros2 service call /gello_ur_bridge/eef_resume std_srvs/srv/Trigger)"
        ;;
    joint_delta)
        echo "###  JOINT_DELTA: switch_only bring-up, then /gello_ur_bridge/joint_delta_start"
        echo "###  anchors at the mock arm's current pose (jd_gain:=0.0 = must not move)."
        echo "###  Watch: ros2 topic echo /gello_ur_bridge/joint_delta/state"
        ;;
    *)
        echo "###  JOINT: the mock arm chases the GELLO pose from ~t=8s, then mirrors it."
        ;;
esac
if [ "${LAUNCH_RVIZ}" = "true" ]; then
    echo "###  RViz: $RVIZ_CONFIG (operator-side view). launch_rviz:=false to skip."
else
    echo "###  RViz: OFF (launch_rviz:=false)"
fi
echo "###  Success line to look for:  'Move-to-start handshake SUCCEEDED ... skipping the Robotiq gripper'"
echo "###  Ctrl-C here for a clean shutdown (takes ~15 s: under mock hardware ros2_control_node"
echo "###  ignores SIGINT and urscript_interface needs the SIGKILL escalation — measured, harmless)."
echo "##########################################################################"

# --------------------------------------------------------------------------- #
# RViz (optional) + launch.
# --------------------------------------------------------------------------- #
if [ "${LAUNCH_RVIZ}" = "true" ]; then
    if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
        echo "### WARNING: no DISPLAY — skipping rviz2 (pass launch_rviz:=false to silence)." >&2
    elif [ ! -f "$RVIZ_CONFIG" ]; then
        echo "### WARNING: $RVIZ_CONFIG missing — skipping rviz2." >&2
    else
        rviz2 -d "$RVIZ_CONFIG" &
        RVIZ_PID=$!
        cleanup() { kill "$RVIZ_PID" 2>/dev/null || true; }
        trap cleanup INT TERM EXIT
        # Foreground launch so Ctrl-C reaches ros2 launch (same process group)
        # and the EXIT trap then reaps rviz2. `exec` would drop the trap.
        "${LAUNCH_CMD[@]}"
        exit $?
    fi
fi
exec "${LAUNCH_CMD[@]}"
