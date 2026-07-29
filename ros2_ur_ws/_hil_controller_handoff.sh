#!/usr/bin/env bash
# Shared, source-only safety helpers for the HIL preposition -> actor handoff.
#
# This file never runs a switch by itself.  Callers must explicitly invoke
# hil_arm_controller_handoff(), and run_hil_actor.sh does that only for a real
# (non-fake), non-probe --arm invocation after all other preflight checks pass.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source this file; do not execute it directly." >&2
    exit 2
fi

HIL_HANDOFF_MARKER_VERSION=1

hil_default_preposition_marker() {
    local uid runtime_dir
    uid="$(id -u)"
    runtime_dir="${XDG_RUNTIME_DIR:-/run/user/${uid}}"
    if [[ ! -d "$runtime_dir" || ! -w "$runtime_dir" ]]; then
        runtime_dir="/tmp/gello-hil-${uid}"
        if [[ -e "$runtime_dir" && ! -d "$runtime_dir" ]]; then
            echo "ERROR: marker runtime path is not a directory: $runtime_dir" >&2
            return 1
        fi
        mkdir -p -- "$runtime_dir" || return 1
        chmod 700 -- "$runtime_dir" || return 1
    fi
    printf '%s/hil-preposition.ready\n' "$runtime_dir"
}

hil_invalidate_preposition_marker() {
    local marker="$1"
    if [[ -z "$marker" || "$marker" == "/" ]]; then
        echo "ERROR: refusing unsafe marker path '$marker'" >&2
        return 1
    fi
    rm -f -- "$marker"
}

hil_read_controller_states() {
    # Sets HIL_SOURCE_STATE/HIL_TARGET_STATE/HIL_CONTROLLER_LIST_OUTPUT.
    local source_controller="$1" target_controller="$2" output clean
    local source_states target_states source_count target_count

    HIL_SOURCE_STATE=""
    HIL_TARGET_STATE=""
    HIL_CONTROLLER_LIST_OUTPUT=""
    if ! output="$(timeout --signal=KILL 15 ros2 control list_controllers 2>&1)"; then
        echo "ERROR: controller_manager did not answer list_controllers" >&2
        printf '%s\n' "$output" >&2
        return 1
    fi
    if [[ -z "$output" ]]; then
        echo "ERROR: controller_manager returned an empty controller list" >&2
        return 1
    fi
    clean="$(printf '%s\n' "$output" | sed 's/\x1b\[[0-9;]*m//g')"
    source_states="$(printf '%s\n' "$clean" | awk -v name="$source_controller" '$1 == name {print $NF}')"
    target_states="$(printf '%s\n' "$clean" | awk -v name="$target_controller" '$1 == name {print $NF}')"
    source_count="$(printf '%s\n' "$source_states" | awk 'NF {n++} END {print n+0}')"
    target_count="$(printf '%s\n' "$target_states" | awk 'NF {n++} END {print n+0}')"
    if [[ "$source_count" -ne 1 || "$target_count" -ne 1 ]]; then
        echo "ERROR: expected exactly one $source_controller and one $target_controller" >&2
        printf '%s\n' "$clean" >&2
        return 1
    fi
    HIL_SOURCE_STATE="$(printf '%s\n' "$source_states" | head -n1)"
    HIL_TARGET_STATE="$(printf '%s\n' "$target_states" | head -n1)"
    case "$HIL_SOURCE_STATE" in active|inactive) ;; *)
        echo "ERROR: unexpected $source_controller state: $HIL_SOURCE_STATE" >&2
        return 1
    esac
    case "$HIL_TARGET_STATE" in active|inactive) ;; *)
        echo "ERROR: unexpected $target_controller state: $HIL_TARGET_STATE" >&2
        return 1
    esac
    HIL_CONTROLLER_LIST_OUTPUT="$clean"
}

hil_read_command_publisher_count() {
    # Sets HIL_COMMAND_PUBLISHER_COUNT.
    local topic="$1" output count
    HIL_COMMAND_PUBLISHER_COUNT=""
    if ! output="$(timeout --signal=KILL 15 ros2 topic info "$topic" 2>&1)"; then
        echo "ERROR: cannot inspect publishers on $topic" >&2
        printf '%s\n' "$output" >&2
        return 1
    fi
    count="$(printf '%s\n' "$output" | awk '/Publisher count:/ {print $NF; exit}')"
    if [[ ! "$count" =~ ^[0-9]+$ ]]; then
        echo "ERROR: malformed publisher count for $topic: '$count'" >&2
        return 1
    fi
    HIL_COMMAND_PUBLISHER_COUNT="$count"
}

hil_assert_no_command_publishers() {
    local topic="$1"
    hil_read_command_publisher_count "$topic" || return 1
    if [[ "$HIL_COMMAND_PUBLISHER_COUNT" -ne 0 ]]; then
        echo "ERROR: $topic already has $HIL_COMMAND_PUBLISHER_COUNT publisher(s)" >&2
        echo "       stop gello_ur_bridge/other runners before controller handoff" >&2
        return 1
    fi
}

hil_wait_no_command_publishers() {
    # DDS graph removal is asynchronous.  After the actor has closed its ROS
    # node, give discovery a short bounded window to report publisher=0 before
    # deciding whether it is safe to hand the command interfaces back to the
    # trajectory controller.
    local topic="$1" attempts="${2:-30}" delay_s="${3:-0.1}"
    local index
    if [[ ! "$attempts" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: publisher wait attempts must be a positive integer" >&2
        return 1
    fi
    for ((index = 0; index < attempts; index++)); do
        if hil_read_command_publisher_count "$topic" && \
           [[ "$HIL_COMMAND_PUBLISHER_COUNT" -eq 0 ]]; then
            return 0
        fi
        sleep "$delay_s"
    done
    echo "ERROR: $topic still has ${HIL_COMMAND_PUBLISHER_COUNT:-unknown} publisher(s) after actor exit" >&2
    return 1
}

_hil_marker_get() {
    local marker="$1" key="$2"
    awk -F= -v key="$key" '
        $1 == key { value = substr($0, index($0, "=") + 1); count++ }
        END { if (count != 1) exit 2; print value }
    ' "$marker"
}

hil_write_preposition_marker() {
    local marker="$1" reset_joints_csv="$2" tolerance="$3" proof="$4"
    local parent tmp now domain
    case "$proof" in
        verified_existing|operator_preposition) ;;
        *) echo "ERROR: invalid preposition proof '$proof'" >&2; return 1 ;;
    esac
    parent="$(dirname -- "$marker")"
    mkdir -p -- "$parent" || return 1
    now="$(date +%s)"
    domain="${ROS_DOMAIN_ID:-0}"
    umask 077
    tmp="$(mktemp "${marker}.tmp.XXXXXX")" || return 1
    if ! printf '%s\n' \
        "version=${HIL_HANDOFF_MARKER_VERSION}" \
        "created_epoch=${now}" \
        "uid=$(id -u)" \
        "ros_domain_id=${domain}" \
        "proof=${proof}" \
        "reset_joints=${reset_joints_csv}" \
        "tolerance_rad=${tolerance}" \
        "source_controller=scaled_joint_trajectory_controller" \
        "target_controller=forward_position_controller" \
        >"$tmp"; then
        rm -f -- "$tmp"
        return 1
    fi
    chmod 600 -- "$tmp" || { rm -f -- "$tmp"; return 1; }
    mv -f -- "$tmp" "$marker"
    echo "preposition proof marker: $marker"
}

hil_validate_preposition_marker() {
    local marker="$1" expected_reset="$2" max_tolerance="$3" max_age_s="$4"
    local owner mode version created marker_uid marker_domain domain proof reset tolerance
    local now age

    if [[ ! -f "$marker" || -L "$marker" ]]; then
        echo "ERROR: valid preposition marker is missing: $marker" >&2
        return 1
    fi
    owner="$(stat -c '%u' -- "$marker" 2>/dev/null)" || return 1
    mode="$(stat -c '%a' -- "$marker" 2>/dev/null)" || return 1
    if [[ "$owner" != "$(id -u)" ]]; then
        echo "ERROR: preposition marker owner uid=$owner, expected $(id -u)" >&2
        return 1
    fi
    if (( (8#$mode & 022) != 0 )); then
        echo "ERROR: preposition marker is group/world writable (mode $mode)" >&2
        return 1
    fi

    version="$(_hil_marker_get "$marker" version)" || return 1
    created="$(_hil_marker_get "$marker" created_epoch)" || return 1
    marker_uid="$(_hil_marker_get "$marker" uid)" || return 1
    marker_domain="$(_hil_marker_get "$marker" ros_domain_id)" || return 1
    proof="$(_hil_marker_get "$marker" proof)" || return 1
    reset="$(_hil_marker_get "$marker" reset_joints)" || return 1
    tolerance="$(_hil_marker_get "$marker" tolerance_rad)" || return 1

    [[ "$version" == "$HIL_HANDOFF_MARKER_VERSION" ]] || {
        echo "ERROR: unsupported marker version '$version'" >&2; return 1;
    }
    [[ "$marker_uid" == "$(id -u)" ]] || {
        echo "ERROR: marker uid field does not match current user" >&2; return 1;
    }
    domain="${ROS_DOMAIN_ID:-0}"
    [[ "$marker_domain" == "$domain" ]] || {
        echo "ERROR: marker ROS_DOMAIN_ID=$marker_domain, current=$domain" >&2; return 1;
    }
    case "$proof" in verified_existing|operator_preposition) ;; *)
        echo "ERROR: invalid marker proof '$proof'" >&2; return 1 ;;
    esac
    [[ "$reset" == "$expected_reset" ]] || {
        echo "ERROR: marker RESET_JOINTS does not match the actor task" >&2; return 1;
    }
    if ! awk -v value="$tolerance" -v maximum="$max_tolerance" \
        'BEGIN { exit !(value > 0 && value <= maximum) }'; then
        echo "ERROR: marker tolerance $tolerance exceeds allowed $max_tolerance" >&2
        return 1
    fi
    [[ "$created" =~ ^[0-9]+$ ]] || {
        echo "ERROR: malformed marker timestamp '$created'" >&2; return 1;
    }
    now="$(date +%s)"
    age=$((now - created))
    if (( age < -30 || age > max_age_s )); then
        echo "ERROR: preposition marker age ${age}s is outside [-30, ${max_age_s}]s" >&2
        return 1
    fi
    echo "preposition marker valid: proof=$proof age=${age}s tolerance=${tolerance}rad"
}

hil_verify_live_reset_pose() {
    local checker="$1" reset_joints_csv="$2" tolerance="$3" topic="${4:-/joint_states}"
    if [[ ! -f "$checker" ]]; then
        echo "ERROR: joint pose checker missing: $checker" >&2
        return 1
    fi
    python3 "$checker" \
        --topic "$topic" \
        --target "$reset_joints_csv" \
        --tolerance "$tolerance" \
        --timeout 5.0 \
        --quiet
}

hil_assert_preposition_ready() {
    local source_controller="$1" target_controller="$2" command_topic="$3"
    hil_read_controller_states "$source_controller" "$target_controller" || return 1
    if [[ "$HIL_SOURCE_STATE" != active || "$HIL_TARGET_STATE" != inactive ]]; then
        echo "ERROR: preposition proof requires $source_controller=active and $target_controller=inactive" >&2
        echo "       observed: $source_controller=$HIL_SOURCE_STATE, $target_controller=$HIL_TARGET_STATE" >&2
        return 1
    fi
    hil_assert_no_command_publishers "$command_topic"
}

_hil_restore_preposition_controller() {
    local source_controller="$1" target_controller="$2"
    echo "attempting fail-closed rollback to $source_controller ..." >&2
    timeout --signal=KILL 15 ros2 control switch_controllers --strict \
        --activate "$source_controller" --deactivate "$target_controller" >/dev/null 2>&1 || true
}

hil_arm_controller_handoff() {
    # No reset or preposition motion happens here.  The only mutation is the
    # strict controller switch after a marker + live-pose proof.
    local source_controller="$1" target_controller="$2" command_topic="$3"
    local marker="$4" pose_checker="$5" reset_joints_csv="$6"
    local pose_tolerance="$7" marker_max_age_s="$8"
    local switch_output

    hil_read_controller_states "$source_controller" "$target_controller" || return 1
    hil_assert_no_command_publishers "$command_topic" || return 1

    if [[ "$HIL_SOURCE_STATE" == inactive && "$HIL_TARGET_STATE" == active ]]; then
        # Marker is intentionally optional for an already-completed switch, but
        # an old FPC can be left active at an arbitrary episode pose.  Starting
        # the actor there would let its automatic reset move without the narrow
        # preposition proof.  Require the same fresh live RESET pose as the
        # switching path before accepting the idempotent state.
        if ! hil_verify_live_reset_pose \
            "$pose_checker" "$reset_joints_csv" "$pose_tolerance" "/joint_states"; then
            echo "ERROR: $target_controller is active but the live arm is not at RESET pose" >&2
            return 1
        fi
        echo "controller handoff: already ready ($target_controller=active, live RESET pose verified, idempotent)"
        return 0
    fi
    if [[ "$HIL_SOURCE_STATE" != active || "$HIL_TARGET_STATE" != inactive ]]; then
        echo "ERROR: unsafe/unexpected controller combination" >&2
        echo "       $source_controller=$HIL_SOURCE_STATE, $target_controller=$HIL_TARGET_STATE" >&2
        return 1
    fi

    hil_validate_preposition_marker \
        "$marker" "$reset_joints_csv" "$pose_tolerance" "$marker_max_age_s" || return 1
    if ! hil_verify_live_reset_pose \
        "$pose_checker" "$reset_joints_csv" "$pose_tolerance" "/joint_states"; then
        echo "ERROR: live arm pose no longer matches the preposition proof" >&2
        hil_invalidate_preposition_marker "$marker" || true
        return 1
    fi
    # Close the largest practical race window immediately before switching.
    hil_assert_no_command_publishers "$command_topic" || return 1

    echo "controller handoff: strict switch $source_controller -> $target_controller"
    if ! switch_output="$(timeout --signal=KILL 15 ros2 control switch_controllers --strict \
        --activate "$target_controller" --deactivate "$source_controller" 2>&1)"; then
        echo "ERROR: controller switch command failed" >&2
        printf '%s\n' "$switch_output" >&2
        # A timed-out/service-erroring strict request should be atomic, but do
        # not assume that across controller_manager versions.  If state changed
        # at all, make one best-effort return to the proven JTC state.
        if hil_read_controller_states "$source_controller" "$target_controller" && \
           [[ "$HIL_SOURCE_STATE:$HIL_TARGET_STATE" != active:inactive ]]; then
            _hil_restore_preposition_controller "$source_controller" "$target_controller"
        fi
        hil_invalidate_preposition_marker "$marker" || true
        return 1
    fi

    if ! hil_read_controller_states "$source_controller" "$target_controller" || \
       [[ "$HIL_SOURCE_STATE" != inactive || "$HIL_TARGET_STATE" != active ]]; then
        echo "ERROR: controller switch postcondition failed" >&2
        _hil_restore_preposition_controller "$source_controller" "$target_controller"
        hil_invalidate_preposition_marker "$marker" || true
        return 1
    fi
    if ! hil_assert_no_command_publishers "$command_topic"; then
        echo "ERROR: a command publisher appeared during handoff" >&2
        _hil_restore_preposition_controller "$source_controller" "$target_controller"
        hil_invalidate_preposition_marker "$marker" || true
        return 1
    fi

    hil_invalidate_preposition_marker "$marker" || true
    echo "controller handoff PASS: $source_controller=inactive, $target_controller=active"
}

hil_restore_controller_after_actor() {
    # Restore controller ownership only after the actor publisher has vanished.
    # This does NOT move to RESET: activating the trajectory controller at the
    # current measured pose merely returns the rig to its normal holding owner.
    # Refusing the switch while any command publisher remains is deliberate --
    # a late 250 Hz actor thread and a newly activated controller must never
    # race over the same hardware interfaces.
    local source_controller="$1" target_controller="$2" command_topic="$3"
    local switch_output

    echo "controller cleanup: waiting for actor command publisher to disappear"
    hil_wait_no_command_publishers "$command_topic" || return 1
    hil_read_controller_states "$source_controller" "$target_controller" || return 1

    if [[ "$HIL_SOURCE_STATE" == active && "$HIL_TARGET_STATE" == inactive ]]; then
        echo "controller cleanup: already restored ($source_controller=active)"
        return 0
    fi
    if [[ "$HIL_SOURCE_STATE" != inactive || "$HIL_TARGET_STATE" != active ]]; then
        echo "ERROR: refusing cleanup from unexpected controller combination" >&2
        echo "       $source_controller=$HIL_SOURCE_STATE, $target_controller=$HIL_TARGET_STATE" >&2
        return 1
    fi

    echo "controller cleanup: strict switch $target_controller -> $source_controller"
    if ! switch_output="$(timeout --signal=KILL 15 ros2 control switch_controllers --strict \
        --activate "$source_controller" --deactivate "$target_controller" 2>&1)"; then
        echo "ERROR: controller cleanup switch failed" >&2
        printf '%s\n' "$switch_output" >&2
        return 1
    fi
    if ! hil_read_controller_states "$source_controller" "$target_controller" || \
       [[ "$HIL_SOURCE_STATE" != active || "$HIL_TARGET_STATE" != inactive ]]; then
        echo "ERROR: controller cleanup postcondition failed" >&2
        return 1
    fi
    hil_assert_no_command_publishers "$command_topic" || return 1
    echo "controller cleanup PASS: $source_controller=active, $target_controller=inactive"
}
