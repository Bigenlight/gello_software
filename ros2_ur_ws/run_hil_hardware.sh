#!/usr/bin/env bash
# Bring up only the robot-local hardware processes required by HIL-SERL:
#
#   1. UR7e driver (headless/REMOTE, STJC active, tool communication enabled)
#   2. Robotiq 2F-85 Modbus node over the driver's /tmp/ttyUR bridge
#   3. Passive/read-only GELLO leader publisher
#
# Cameras, the HIL GUI, and the actor deliberately remain separate commands.
# Ctrl-C, a child exit, or a readiness failure stops all three process groups;
# after cleanup finishes this same command can be run again.
#
# Usage:
#   ./run_hil_hardware.sh
#   ROBOT_IP=192.168.10.11 ./run_hil_hardware.sh
#   ./run_hil_hardware.sh --dry-run   # validate/print only; touches no hardware

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WS_SETUP="$SCRIPT_DIR/install/setup.bash"
TOPIC_CHECKER="$SCRIPT_DIR/_hil_topic_rate_check.py"
ROBOT_IP="${ROBOT_IP:-192.168.10.11}"
READY_TIMEOUT_S="${HIL_HARDWARE_READY_TIMEOUT_S:-30}"
STOP_TIMEOUT_S="${HIL_HARDWARE_STOP_TIMEOUT_S:-8}"
HARDWARE_LOCK="${XDG_RUNTIME_DIR:-/tmp}/hil-hardware-${UID}.lock"
DRY_RUN=0

case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
        sed -n '2,/^$/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
        exit 0
        ;;
    *)
        echo "ERROR: unknown argument: $1 (expected --dry-run or --help)" >&2
        exit 2
        ;;
esac
if (( $# > 1 )); then
    echo "ERROR: too many arguments" >&2
    exit 2
fi

for value_name in READY_TIMEOUT_S STOP_TIMEOUT_S; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $value_name must be a positive integer (got '$value')" >&2
        exit 2
    fi
done

for required in "$ROS_SETUP" "$WS_SETUP" "$TOPIC_CHECKER"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required setup/helper is missing: $required" >&2
        exit 1
    fi
done

# Generated ROS setup scripts are not nounset-clean. Keep strict mode for this
# wrapper, but relax it only while sourcing the two verified overlays.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$WS_SETUP"
set -u

export GELLO_REPO_ROOT="${GELLO_REPO_ROOT:-$REPO_ROOT}"
GELLO_PARAMS="$SCRIPT_DIR/install/ur_gello_bringup/share/ur_gello_bringup/config/ur7e_gello.yaml"
if [[ ! -f "$GELLO_PARAMS" ]]; then
    echo "ERROR: installed GELLO config is missing: $GELLO_PARAMS" >&2
    exit 1
fi

command -v ros2 >/dev/null || {
    echo "ERROR: ros2 is not available after sourcing the overlays" >&2
    exit 1
}
for helper in awk flock ps python3 setsid timeout; do
    command -v "$helper" >/dev/null || {
        echo "ERROR: required command is unavailable: $helper" >&2
        exit 1
    }
done
ros2 pkg prefix ur_robot_driver >/dev/null
ros2 pkg prefix ur_gello_bringup >/dev/null

DRIVER_CMD=(
    ros2 launch ur_robot_driver ur_control.launch.py
    ur_type:=ur7e
    robot_ip:="$ROBOT_IP"
    headless_mode:=true
    launch_rviz:=false
    initial_joint_controller:=scaled_joint_trajectory_controller
    use_tool_communication:=true
    tool_voltage:=24
    tool_device_name:=/tmp/ttyUR
)
GRIPPER_CMD=(
    ros2 run ur_gello_bringup robotiq_gripper_modbus
    --ros-args -p serial_port:=/tmp/ttyUR
)
GELLO_CMD=(
    ros2 run ur_gello_bringup gello_publisher
    --ros-args --params-file "$GELLO_PARAMS"
)

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

echo "============================================================"
echo " HIL hardware supervisor"
echo "   robot       : UR7e at $ROBOT_IP (headless; REMOTE mode required)"
echo "   controller  : scaled_joint_trajectory_controller"
echo "   gripper     : Robotiq /tmp/ttyUR (tool voltage 24 V)"
echo "   GELLO       : passive/read-only ($GELLO_REPO_ROOT)"
echo "============================================================"
echo "Cameras, HIL GUI, and actor are NOT started by this script."
echo "Keep the pendant E-STOP reachable; the gripper may calibrate on connect."

if (( DRY_RUN )); then
    echo ""
    echo "--dry-run: setup/packages/config validated; no process is started."
    echo "[driver]"
    print_command "${DRIVER_CMD[@]}"
    echo "[gripper]"
    print_command "${GRIPPER_CMD[@]}"
    echo "[GELLO reader]"
    print_command env "GELLO_REPO_ROOT=$GELLO_REPO_ROOT" "${GELLO_CMD[@]}"
    exit 0
fi

# Serialize launchers before checking external ownership.  Children inherit
# fd 9, so even a hard-killed supervisor leaves the lock held while one of its
# ROS descendants is still alive.
exec 9>"$HARDWARE_LOCK"
if ! flock -n 9; then
    echo "ERROR: another HIL hardware bundle still owns $HARDWARE_LOCK" >&2
    echo "       stop it and wait for [cleanup] complete before retrying" >&2
    exit 1
fi

# A lock cannot identify processes started by older/manual commands.  Refuse
# the exact launch/run entrypoints instead of allowing a second GELLO process
# to take the FTDI device or a second driver to satisfy readiness with the
# first driver's topics.
existing_owners="$({ python3 - <<'PY'
from pathlib import Path

owned_tokens = {
    "ur_control.launch.py",
    "robotiq_gripper_modbus",
    "gello_publisher",
}
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = [
            item.decode("utf-8", "replace")
            for item in (proc / "cmdline").read_bytes().split(b"\0")
            if item
        ]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(Path(token).name in owned_tokens for token in argv):
        print(f"{proc.name}: {' '.join(argv)}")
PY
} 2>/dev/null)"
if [[ -n "$existing_owners" ]]; then
    echo "ERROR: an existing UR/GELLO/gripper owner is already running:" >&2
    printf '  %s\n' "$existing_owners" >&2
    echo "       stop the existing owner before starting this bundle" >&2
    exit 1
fi
if [[ -e /tmp/ttyUR || -L /tmp/ttyUR ]]; then
    echo "ERROR: /tmp/ttyUR already exists; another or stale tool bridge may own it" >&2
    echo "       resolve the existing UR driver/tool bridge before retrying" >&2
    exit 1
fi

declare -a CHILD_PIDS=()
declare -a CHILD_NAMES=()
CLEANING_UP=0

group_has_live_processes() {
    local pgid="$1"
    ps -eo pgid=,stat= | awk -v wanted="$pgid" '
        $1 == wanted && $2 !~ /^Z/ { found=1 }
        END { exit !found }
    '
}

leader_is_live() {
    local pid="$1" stat
    kill -0 "$pid" 2>/dev/null || return 1
    stat="$(ps -o stat= -p "$pid" 2>/dev/null)"
    [[ -n "$stat" && "$stat" != Z* ]]
}

cleanup() {
    local original_rc=$?
    local index deadline live pid
    if (( CLEANING_UP )); then
        return
    fi
    CLEANING_UP=1
    trap - INT TERM HUP EXIT
    set +e

    if (( ${#CHILD_PIDS[@]} > 0 )); then
        echo ""
        echo "[cleanup] stopping GELLO/gripper/driver process groups ..."
        for ((index=${#CHILD_PIDS[@]}-1; index>=0; index--)); do
            kill -TERM -- "-${CHILD_PIDS[index]}" 2>/dev/null || true
        done

        deadline=$((SECONDS + STOP_TIMEOUT_S))
        while (( SECONDS < deadline )); do
            live=0
            for pid in "${CHILD_PIDS[@]}"; do
                if group_has_live_processes "$pid"; then
                    live=1
                    break
                fi
            done
            (( live == 0 )) && break
            sleep 0.1
        done

        for ((index=${#CHILD_PIDS[@]}-1; index>=0; index--)); do
            pid="${CHILD_PIDS[index]}"
            if group_has_live_processes "$pid"; then
                echo "[cleanup] ${CHILD_NAMES[index]} did not stop; sending KILL" >&2
                kill -KILL -- "-$pid" 2>/dev/null || true
            fi
        done
        for pid in "${CHILD_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        echo "[cleanup] complete — this terminal can run ./run_hil_hardware.sh again."
    fi
    exit "$original_rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

start_child() {
    local name="$1"
    shift
    echo ""
    echo "[start] $name"
    print_command "$@"

    # Each ROS tree gets its own session/process group. The supervisor alone
    # receives terminal Ctrl-C and cleanup can terminate every descendant of a
    # ros2 launch/run command, rather than leaving nodes or :54321 ownership.
    (
        trap - EXIT INT TERM HUP
        exec setsid "$@"
    ) &
    local pid=$!
    CHILD_PIDS+=("$pid")
    CHILD_NAMES+=("$name")
    sleep 0.2
    if ! leader_is_live "$pid"; then
        set +e
        wait "$pid"
        local rc=$?
        set -e
        echo "ERROR: $name exited during startup (rc=$rc)" >&2
        return 1
    fi
}

probe_topic() {
    local label="$1" topic="$2" type="$3" min_rate="$4"
    echo "[ready] waiting for $label ($topic) ..."
    timeout --signal=TERM --kill-after=2 "$((READY_TIMEOUT_S + 4))" \
        python3 "$TOPIC_CHECKER" \
        --topic "$topic" --type "$type" --samples 5 \
        --timeout "$READY_TIMEOUT_S" --min-rate "$min_rate"
}

wait_for_tool_tty() {
    local driver_pid="$1" deadline=$((SECONDS + READY_TIMEOUT_S))
    echo "[ready] waiting for driver tool bridge /tmp/ttyUR ..."
    while (( SECONDS < deadline )); do
        [[ -e /tmp/ttyUR ]] && return 0
        if ! leader_is_live "$driver_pid"; then
            echo "ERROR: UR driver exited before /tmp/ttyUR became ready" >&2
            return 1
        fi
        sleep 0.2
    done
    echo "ERROR: /tmp/ttyUR did not appear within ${READY_TIMEOUT_S}s" >&2
    return 1
}

start_child "UR7e driver" "${DRIVER_CMD[@]}"
probe_topic "UR joint state" /joint_states joint_state 50
wait_for_tool_tty "${CHILD_PIDS[0]}"

start_child "Robotiq gripper" "${GRIPPER_CMD[@]}"
probe_topic "Robotiq position" /robotiq_gripper/position_percent float32 2

start_child "GELLO reader" "${GELLO_CMD[@]}"
probe_topic "GELLO joint state" /gello/joint_states joint_state 15

echo ""
echo "========================= READY ============================="
echo "UR7e driver + Robotiq + passive GELLO reader are live."
echo "Leave this terminal running. Ctrl-C stops the whole bundle."
echo "After a collision/disconnect: clear the robot fault if needed,"
echo "Ctrl-C here, wait for [cleanup] complete, then rerun this command."
echo "============================================================="

while true; do
    for index in "${!CHILD_PIDS[@]}"; do
        pid="${CHILD_PIDS[index]}"
        if ! leader_is_live "$pid"; then
            set +e
            wait "$pid"
            rc=$?
            set -e
            echo "ERROR: ${CHILD_NAMES[index]} exited (rc=$rc); stopping the bundle" >&2
            (( rc == 0 )) && rc=1
            exit "$rc"
        fi
    done
    sleep 1
done
