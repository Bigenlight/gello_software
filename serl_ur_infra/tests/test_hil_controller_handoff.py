"""Dependency-light regression tests for the shell controller handoff gate."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import time


_ROOT = Path(__file__).resolve().parents[2]
_LIB = _ROOT / "ros2_ur_ws" / "_hil_controller_handoff.sh"
_ACTOR = _ROOT / "ros2_ur_ws" / "run_hil_actor.sh"
_PREPOSITION = _ROOT / "ros2_ur_ws" / "run_hil_preposition.sh"
_RESET = "3.1382,-1.5276,1.7168,-1.7592,-1.5216,-3.1331"


def _state(source: str, target: str) -> str:
    return (
        "scaled_joint_trajectory_controller joint_trajectory_controller/"
        f"JointTrajectoryController {source}\n"
        "forward_position_controller forward_command_controller/"
        f"ForwardCommandController {target}\n"
    )


def _rig(tmp_path: Path, *, source="active", target="inactive", publishers=0):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state_file = tmp_path / "controllers.txt"
    state_file.write_text(_state(source, target))
    publisher_file = tmp_path / "publishers.txt"
    publisher_file.write_text(str(publishers))
    log_file = tmp_path / "ros2.log"
    ros2 = bin_dir / "ros2"
    ros2.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >>"$MOCK_ROS2_LOG"
if [[ "$1 $2" == "control list_controllers" ]]; then
    cat "$MOCK_CONTROLLER_STATE"
elif [[ "$1 $2" == "topic info" ]]; then
    echo "Type: std_msgs/msg/Float64MultiArray"
    echo "Publisher count: $(cat "$MOCK_PUBLISHER_COUNT")"
    echo "Subscription count: 1"
elif [[ "$1 $2" == "control switch_controllers" ]]; then
    if [[ "${MOCK_SWITCH_FAIL:-0}" == "1" ]]; then
        echo "mock switch failure" >&2
        exit 1
    fi
    if [[ "${MOCK_SWITCH_NO_EFFECT:-0}" != "1" ]]; then
        if [[ " $* " == *" --activate forward_position_controller "* ]]; then
            cat >"$MOCK_CONTROLLER_STATE" <<'EOF'
scaled_joint_trajectory_controller joint_trajectory_controller/JointTrajectoryController inactive
forward_position_controller forward_command_controller/ForwardCommandController active
EOF
        elif [[ " $* " == *" --activate scaled_joint_trajectory_controller "* ]]; then
            cat >"$MOCK_CONTROLLER_STATE" <<'EOF'
scaled_joint_trajectory_controller joint_trajectory_controller/JointTrajectoryController active
forward_position_controller forward_command_controller/ForwardCommandController inactive
EOF
        fi
    fi
    echo "ok"
else
    echo "unexpected mock ros2 invocation: $*" >&2
    exit 9
fi
"""
    )
    ros2.chmod(0o755)
    checker = tmp_path / "pose_check.py"
    checker.write_text(
        "import os, sys\nsys.exit(int(os.environ.get('MOCK_POSE_RC', '0')))\n"
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "MOCK_CONTROLLER_STATE": str(state_file),
            "MOCK_PUBLISHER_COUNT": str(publisher_file),
            "MOCK_ROS2_LOG": str(log_file),
            "MOCK_POSE_RC": "0",
            "ROS_DOMAIN_ID": "73",
        }
    )
    return env, state_file, publisher_file, log_file, checker


def _run(tmp_path: Path, env: dict[str, str], checker: Path, *, marker=True):
    marker_path = tmp_path / "preposition.ready"
    setup = ""
    if marker:
        setup = (
            f'hil_write_preposition_marker "{marker_path}" "{_RESET}" 0.10 '
            'verified_existing >/dev/null\n'
        )
    command = f"""
set -e
source "{_LIB}"
{setup}hil_arm_controller_handoff \
  scaled_joint_trajectory_controller forward_position_controller \
  /forward_position_controller/commands \
  "{marker_path}" "{checker}" "{_RESET}" 0.10 900
"""
    return subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_valid_marker_and_live_pose_switch_exactly_once(tmp_path: Path):
    env, state_file, _, log_file, checker = _rig(tmp_path)

    result = _run(tmp_path, env, checker)

    assert result.returncode == 0, result.stderr
    assert _state("inactive", "active") == state_file.read_text()
    switches = [line for line in log_file.read_text().splitlines() if "switch_controllers" in line]
    assert len(switches) == 1
    assert "--strict" in switches[0]
    assert not (tmp_path / "preposition.ready").exists()


def test_missing_marker_refuses_switch(tmp_path: Path):
    env, state_file, _, log_file, checker = _rig(tmp_path)

    result = _run(tmp_path, env, checker, marker=False)

    assert result.returncode != 0
    assert _state("active", "inactive") == state_file.read_text()
    assert "switch_controllers" not in log_file.read_text()


def test_pose_changed_after_marker_refuses_switch_and_consumes_proof(tmp_path: Path):
    env, state_file, _, log_file, checker = _rig(tmp_path)
    env["MOCK_POSE_RC"] = "1"

    result = _run(tmp_path, env, checker)

    assert result.returncode != 0
    assert _state("active", "inactive") == state_file.read_text()
    assert "switch_controllers" not in log_file.read_text()
    assert not (tmp_path / "preposition.ready").exists()


def test_existing_command_publisher_refuses_switch(tmp_path: Path):
    env, state_file, _, log_file, checker = _rig(tmp_path, publishers=1)

    result = _run(tmp_path, env, checker)

    assert result.returncode != 0
    assert _state("active", "inactive") == state_file.read_text()
    assert "switch_controllers" not in log_file.read_text()


def test_already_handed_off_is_idempotent_without_marker(tmp_path: Path):
    env, state_file, _, log_file, checker = _rig(
        tmp_path, source="inactive", target="active"
    )

    result = _run(tmp_path, env, checker, marker=False)

    assert result.returncode == 0, result.stderr
    assert _state("inactive", "active") == state_file.read_text()
    assert "switch_controllers" not in log_file.read_text()


def test_already_active_fpc_at_non_reset_pose_is_rejected(tmp_path: Path):
    env, state_file, _, log_file, checker = _rig(
        tmp_path, source="inactive", target="active"
    )
    env["MOCK_POSE_RC"] = "1"

    result = _run(tmp_path, env, checker, marker=False)

    assert result.returncode != 0
    assert _state("inactive", "active") == state_file.read_text()
    assert "not at RESET pose" in result.stderr
    assert "switch_controllers" not in log_file.read_text()


def test_unexpected_controller_combination_fails_closed(tmp_path: Path):
    env, state_file, _, log_file, checker = _rig(
        tmp_path, source="active", target="active"
    )

    result = _run(tmp_path, env, checker)

    assert result.returncode != 0
    assert _state("active", "active") == state_file.read_text()
    assert "switch_controllers" not in log_file.read_text()


def test_switch_success_without_postcondition_fails_and_attempts_rollback(tmp_path: Path):
    env, _, _, log_file, checker = _rig(tmp_path)
    env["MOCK_SWITCH_NO_EFFECT"] = "1"

    result = _run(tmp_path, env, checker)

    assert result.returncode != 0
    switches = [line for line in log_file.read_text().splitlines() if "switch_controllers" in line]
    assert len(switches) == 2
    assert "--activate forward_position_controller" in switches[0]
    assert "--activate scaled_joint_trajectory_controller" in switches[1]


def _run_cleanup(env: dict[str, str]):
    command = f"""
set -e
source "{_LIB}"
hil_restore_controller_after_actor \
  scaled_joint_trajectory_controller forward_position_controller \
  /forward_position_controller/commands
"""
    return subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_actor_exit_cleanup_restores_trajectory_controller(tmp_path: Path):
    env, state_file, _, log_file, _ = _rig(
        tmp_path, source="inactive", target="active", publishers=0
    )

    result = _run_cleanup(env)

    assert result.returncode == 0, result.stderr
    assert state_file.read_text() == _state("active", "inactive")
    switches = [
        line for line in log_file.read_text().splitlines()
        if "switch_controllers" in line
    ]
    assert len(switches) == 1
    assert "--activate scaled_joint_trajectory_controller" in switches[0]
    assert "--deactivate forward_position_controller" in switches[0]


def test_actor_exit_cleanup_refuses_while_publisher_remains(tmp_path: Path):
    env, state_file, _, log_file, _ = _rig(
        tmp_path, source="inactive", target="active", publishers=1
    )

    result = _run_cleanup(env)

    assert result.returncode != 0
    assert state_file.read_text() == _state("inactive", "active")
    assert "still has 1 publisher" in result.stderr
    assert "switch_controllers" not in log_file.read_text()


def test_actor_exit_cleanup_is_idempotent(tmp_path: Path):
    env, state_file, _, log_file, _ = _rig(
        tmp_path, source="active", target="inactive", publishers=0
    )

    result = _run_cleanup(env)

    assert result.returncode == 0, result.stderr
    assert state_file.read_text() == _state("active", "inactive")
    assert "switch_controllers" not in log_file.read_text()


def test_actor_exit_cleanup_rejects_unexpected_controller_pair(tmp_path: Path):
    env, state_file, _, log_file, _ = _rig(
        tmp_path, source="inactive", target="inactive", publishers=0
    )

    result = _run_cleanup(env)

    assert result.returncode != 0
    assert state_file.read_text() == _state("inactive", "inactive")
    assert "unexpected controller combination" in result.stderr
    assert "switch_controllers" not in log_file.read_text()


def test_shell_entrypoint_keeps_probe_before_mutating_handoff():
    text = _ACTOR.read_text()
    dry_exit = text.index('if [ "$DRY_PREFLIGHT" -eq 1 ]; then', text.index("# 결과 요약"))
    handoff = text.index("if ! hil_arm_controller_handoff", dry_exit)
    final_deadman = text.rindex('python3 "$DEADMAN_CHECKER"', dry_exit, handoff)
    assert dry_exit < handoff
    assert dry_exit < final_deadman < handoff
    assert 'if [ "$ARM_REQUESTED" -eq 1 ] && [ "$FAKE_ENV" -eq 0 ]; then' in text
    assert "SKIP_ROS_CHECKS=1은 금지" in text
    assert "hil_restore_controller_after_actor" in text
    assert 'trap \'forward_actor_signal INT\' INT' in text
    # An asynchronous command inherits SIGINT ignored from non-job-control
    # bash.  The child must restore default dispositions before exec, and the
    # parent must re-wait after its trap interrupts wait(1).
    assert "trap - INT TERM HUP" in text
    assert 'exec setsid "${ACTOR_CMD[@]}"' in text
    assert re.search(
        r'while true; do\s+wait "\$ACTOR_PID".*?kill -0 "\$ACTOR_PID"',
        text,
        re.DOTALL,
    )

    # The actor wrapper may tell the operator which tool to run, but must never
    # invoke preposition/reset motion itself.
    assert not re.search(
        r'^\s*(?:"?\$SCRIPT_DIR"?/|\./)run_hil_preposition\.sh(?:\s|$)',
        text,
        re.MULTILINE,
    )


def test_preposition_invalidates_old_marker_and_only_records_after_pose_pass():
    text = _PREPOSITION.read_text()
    invalidation = text.index('hil_invalidate_preposition_marker "$PREPOSITION_MARKER"')
    first_pose = text.index("pose_check 5.0", invalidation)
    first_proof = text.index("record_proof_and_optional_switch verified_existing", first_pose)
    final_pose = text.rindex("pose_check 5.0")
    moved_proof = text.index("record_proof_and_optional_switch operator_preposition", final_pose)
    assert invalidation < first_pose < first_proof
    assert final_pose < moved_proof
    assert 'if [[ "$DRY_RUN" == "1" ]]' in text


def test_marker_timestamp_is_short_lived_contract(tmp_path: Path):
    env, _, _, _, checker = _rig(tmp_path)
    marker = tmp_path / "preposition.ready"
    command = f"""
source "{_LIB}"
hil_write_preposition_marker "{marker}" "{_RESET}" 0.10 verified_existing >/dev/null
sed -i 's/^created_epoch=.*/created_epoch={int(time.time()) - 901}/' "{marker}"
hil_validate_preposition_marker "{marker}" "{_RESET}" 0.10 900
"""
    del checker
    result = subprocess.run(
        ["bash", "-c", command], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert "marker age" in result.stderr
