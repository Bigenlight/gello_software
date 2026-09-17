#!/usr/bin/env bash
# Read-only diagnosis for the UR ToolComm -> Robotiq 2F-85 path.
#
# This script never opens ROBOT_IP:54321, opens /tmp/ttyUR, starts ROS nodes,
# or sends a gripper/robot command.  It only inspects local processes, files,
# packages, and ROS launch logs.
set -u -o pipefail

readonly ROS_DISTRO_NAME="${GELLO_ROS_DISTRO:-jazzy}"
readonly TOOL_COMM_SCRIPT="/opt/ros/${ROS_DISTRO_NAME}/lib/ur_client_library/tool_communication.py"
readonly SYSTEM_PYTHON="/usr/bin/python3"

status=0

note_failure() {
    status=1
    printf 'FAIL: %s\n' "$*"
}

python_from_path() {
    local requested_path="$1"
    local path_entry
    local old_ifs="$IFS"
    IFS=:
    for path_entry in $requested_path; do
        [ -n "$path_entry" ] || path_entry=.
        if [ -x "$path_entry/python3" ]; then
            printf '%s\n' "$path_entry/python3"
            IFS="$old_ifs"
            return 0
        fi
    done
    IFS="$old_ifs"
    return 1
}

latest_tool_comm_launch_log() {
    local candidate_log
    while IFS= read -r candidate_log; do
        if rg -q '\[ur_tool_comm-[0-9]+\]' "$candidate_log" 2>/dev/null; then
            printf '%s\n' "$candidate_log"
            return 0
        fi
    done < <(find "$HOME/.ros/log" -mindepth 2 -maxdepth 2 -name launch.log -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | cut -d' ' -f2-)
    return 1
}

printf '%s\n' 'Robotiq ToolComm read-only diagnosis'
printf '%s\n' '  no robot TCP connection, no /tmp/ttyUR open, no ROS/gripper command'

if [ -x "$SYSTEM_PYTHON" ]; then
    printf 'system python: %s\n' "$SYSTEM_PYTHON"
else
    note_failure "missing system Python: $SYSTEM_PYTHON"
fi

if [ -f "$TOOL_COMM_SCRIPT" ]; then
    printf 'tool script: %s\n' "$TOOL_COMM_SCRIPT"
    printf 'tool shebang: %s\n' "$(head -n 1 "$TOOL_COMM_SCRIPT")"
else
    note_failure "missing ToolComm script: $TOOL_COMM_SCRIPT"
fi

if "$SYSTEM_PYTHON" -c 'import pytest; print(pytest.__file__)' >/dev/null 2>&1; then
    printf '%s\n' 'system pytest: importable'
else
    note_failure 'system pytest is not importable (ToolComm exits before creating /tmp/ttyUR)'
fi

if command -v socat >/dev/null 2>&1; then
    printf 'socat: %s\n' "$(command -v socat)"
else
    note_failure 'socat is not installed or not on PATH'
fi

if [ -e /tmp/ttyUR ] || [ -L /tmp/ttyUR ]; then
    printf 'tty bridge: present (%s)\n' "$(stat -c '%F %N' /tmp/ttyUR 2>/dev/null || printf 'unreadable')"
else
    note_failure 'tty bridge: absent (/tmp/ttyUR)'
fi

tool_comm_pids="$(pgrep -f '/ur_client_library/tool_communication\.py' 2>/dev/null || true)"
if [ -n "$tool_comm_pids" ]; then
    printf 'live ToolComm PID(s): %s\n' "$(tr '\n' ' ' <<<"$tool_comm_pids")"
else
    note_failure 'no live ToolComm process'
fi

launch_pids="$(pgrep -f '/opt/ros/.*/bin/ros2 launch gello_policy ur7e_diffusion_real\.launch\.py' 2>/dev/null || true)"
if [ -n "$launch_pids" ]; then
    while IFS= read -r launch_pid; do
        [ -r "/proc/$launch_pid/environ" ] || continue
        launch_path="$(tr '\0' '\n' < "/proc/$launch_pid/environ" | sed -n 's/^PATH=//p' | head -n 1)"
        selected_python="$(python_from_path "$launch_path" 2>/dev/null || true)"
        printf 'launch PID %s python3 via inherited PATH: %s\n' "$launch_pid" "${selected_python:-<none>}"
        if [ "$selected_python" != "$SYSTEM_PYTHON" ]; then
            note_failure "launch PID $launch_pid will run ToolComm with ${selected_python:-no python3}, not $SYSTEM_PYTHON"
        fi
    done <<<"$launch_pids"
else
    printf '%s\n' 'no live gello_policy real-robot launch found'
fi

tool_comm_log="$(latest_tool_comm_launch_log || true)"
if [ -n "$tool_comm_log" ]; then
    printf 'latest ToolComm launch log: %s\n' "$tool_comm_log"
    rg '\[ur_tool_comm-[0-9]+\].*(process started|process has died)' "$tool_comm_log" 2>/dev/null || true
fi

fin_wait_count="$(ss -tn state fin-wait-2 '( dport = :54321 )' 2>/dev/null | tail -n +2 | wc -l)"
printf 'stale local FIN-WAIT-2 connections to :54321: %s\n' "$fin_wait_count"
if [ "$fin_wait_count" -gt 0 ]; then
    printf '%s\n' 'note: wait for the controller to release these; do not start another direct-TCP client meanwhile.'
fi

if [ "$status" -eq 0 ]; then
    printf '%s\n' 'PASS: local ToolComm prerequisites and launch interpreter are healthy.'
else
    printf '%s\n' 'RESULT: local ToolComm path is not ready; this script made no robot-side change.'
fi
exit "$status"
