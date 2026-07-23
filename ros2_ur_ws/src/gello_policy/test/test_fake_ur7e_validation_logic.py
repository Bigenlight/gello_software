import math

from gello_policy.fake_ur7e_validation_logic import (
    FakeUr7eVerdict, UR_JOINTS, angular_error, reorder_joint_state,
)


def test_joint_reorder_and_wrap_aware_error():
    names = list(reversed(UR_JOINTS))
    positions = list(reversed(range(6)))
    assert reorder_joint_state(names, positions) == tuple(float(i) for i in range(6))
    assert angular_error(math.pi - 0.01, -math.pi + 0.01) < 0.021


def test_full_validation_state_machine_displacement_is_not_gate():
    verdict = FakeUr7eVerdict(tolerance=0.08, hold_stable_s=0.5)
    start = (3.1, -1.8, 1.6, -1.6, -1.6, -3.1)
    verdict.observe_joint_state(reversed(UR_JOINTS), reversed(start))
    verdict.observe_command(0.0, start)
    verdict.observe_command(0.6, start)
    assert verdict.initial_hold_ready(0.6)
    verdict.mark_started()
    verdict.observe_policy_state("EXECUTE")
    # A policy may return exactly the start pose; finite + tracking still pass.
    verdict.observe_command(0.7, start)
    assert verdict.execution_ready_for_hold()
    assert verdict.max_displacement == 0.0
    verdict.mark_hold(0.8)
    for stamp in (0.9, 1.0, 1.1, 1.2, 1.3, 1.41):
        verdict.observe_command(stamp, start)
    assert verdict.passed(1.41)


def test_nonfinite_command_and_hold_publication_gap_fail():
    verdict = FakeUr7eVerdict(tolerance=0.1, hold_stable_s=0.1)
    verdict.observe_command(0.0, [0.0] * 5 + [float("nan")])
    assert "finite" in verdict.failure

    verdict = FakeUr7eVerdict(tolerance=0.1, hold_stable_s=0.1)
    verdict.observe_joint_state(UR_JOINTS, [0.0] * 6)
    verdict.observe_command(0.0, [0.0] * 6)
    verdict.observe_command(0.2, [0.0] * 6)
    verdict.mark_started()
    verdict.observe_policy_state("EXECUTE")
    verdict.observe_command(0.21, [0.0] * 6)
    verdict.mark_hold(0.22)
    verdict.observe_command(0.6, [0.0] * 6)
    assert "stopped" in verdict.failure


def test_arming_hold_commands_are_not_mistaken_for_policy_actions():
    verdict = FakeUr7eVerdict(tolerance=0.1, hold_stable_s=0.1)
    pose = [0.0] * 6
    verdict.observe_joint_state(UR_JOINTS, pose)
    verdict.observe_command(0.0, pose)
    verdict.observe_command(0.2, pose)
    verdict.mark_started()
    verdict.observe_policy_state("ARMING")
    verdict.observe_command(0.21, pose)
    assert not verdict.execution_ready_for_hold()
    verdict.observe_policy_state("EXECUTE")
    verdict.observe_command(0.22, pose)
    assert verdict.execution_ready_for_hold()
