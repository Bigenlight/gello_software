"""gripper.py: latch truth table (mirrors gello_gripper_bridge_node), deadband, ctrl/pos maps."""
import pytest

from sim_collect import gripper as g


@pytest.mark.parametrize("value,prev,expected", [
    (0.9, None, 1.0), (0.7, None, 1.0),      # >= close_at -> CLOSED (checked first)
    (0.3, None, 0.0), (0.0, None, 0.0),      # <= open_at -> OPEN
    (0.5, None, None),                       # inside the band with no history -> unknown
    (0.5, 0.0, 0.0), (0.5, 1.0, 1.0),        # hysteresis keeps the previous latch
    (0.31, 1.0, 1.0), (0.69, 0.0, 0.0),      # strict inside
    (0.229, 1.0, 0.0),                       # the documented half-open outlier snaps OPEN
])
def test_discrete_latch_table(value, prev, expected):
    assert g.discrete_latch(value, prev, 0.3, 0.7) == expected


def test_validate_discrete_thresholds():
    assert g.validate_discrete_thresholds(0.3, 0.7) is None
    assert g.validate_discrete_thresholds(0.0, 0.7) is not None
    assert g.validate_discrete_thresholds(0.3, 1.0) is not None
    assert g.validate_discrete_thresholds(0.7, 0.3) is not None
    assert g.validate_discrete_thresholds(0.5, 0.5) is not None
    assert g.validate_discrete_thresholds(float("nan"), 0.7) is not None
    with pytest.raises(ValueError):
        g.GripperMapper("discrete", open_at=0.8, close_at=0.2)


def test_continuous_deadband():
    m = g.GripperMapper("continuous", deadband=0.02)
    assert m.update(0.0, now=0.0) == 0.0
    assert m.update(0.015, now=0.1) == 0.0        # |d| < deadband: skipped (ROS gate)
    assert m.update(0.02, now=0.2) == 0.02        # |d| == deadband: applied
    assert m.update(0.05, now=0.3) == 0.05        # moved: follows
    assert m.update(0.06, now=0.4) == 0.05
    assert m.update(1.3, now=0.5) == 1.0          # clamped
    assert m.ctrl == 255.0


def test_discrete_mode_sequence():
    m = g.GripperMapper("discrete")
    assert m.update(0.5, now=0.0) == 0.0          # unknown -> keeps the initial (open) command
    assert m.update(0.8, now=0.1) == 1.0
    assert m.update(0.5, now=0.2) == 1.0          # band: remember
    assert m.update(0.2, now=0.3) == 0.0
    assert m.update(0.69, now=0.4) == 0.0
    assert m.update(0.7, now=0.5) == 1.0


def test_pause_resume_ramp_and_reset():
    """ROS resume ramp: after resume the command slews at 0.6/s toward the leader
    for resume_ramp_s, bypassing the deadband, then plain following resumes."""
    m = g.GripperMapper("continuous", resume_ramp_s=2.0, resume_slew_per_s=0.6)
    m.update(0.4, now=0.0)
    m.pause()
    assert m.update(0.9, now=0.5) == 0.4
    m.resume(now=1.0)
    assert m.ramping
    assert m.update(1.0, now=1.0) == pytest.approx(0.4)            # zero budget on the first sample
    assert m.update(1.0, now=1.5) == pytest.approx(0.4 + 0.3)      # 0.5 s * 0.6/s
    assert m.update(1.0, now=2.0) == pytest.approx(1.0)            # reached (delta 0.3 <= budget)
    assert m.update(0.995, now=2.5) == pytest.approx(0.995)        # inside the ramp: deadband bypassed
    assert m.update(0.985, now=3.5) == pytest.approx(0.995)        # ramp over: deadband gate again
    assert not m.ramping
    # budget is capped at staleness_timeout_s per sample (a dead stream banks nothing)
    m2 = g.GripperMapper("continuous"); m2.update(0.0, now=0.0); m2.pause(); m2.resume(now=1.0)
    m2.update(1.0, now=1.0)
    assert m2.update(1.0, now=1.9) == pytest.approx(0.3)           # elapsed capped to 0.5 s
    # discrete mode honours the ramp toward the latched endpoint
    m3 = g.GripperMapper("discrete"); m3.update(0.0, now=0.0); m3.pause(); m3.resume(now=0.0)
    m3.update(0.9, now=0.0)
    assert m3.update(0.9, now=0.5) == pytest.approx(0.3)
    assert m3.update(0.5, now=1.0) == pytest.approx(0.6)           # band: latched target still 1.0
    m.reset(0.0)
    assert m.grip_cmd == 0.0 and m.ctrl == 0.0 and not m.ramping


def test_ctrl_and_pos_maps():
    assert g.grip_cmd_to_ctrl(0.0) == 0.0
    assert g.grip_cmd_to_ctrl(1.0) == 255.0
    assert g.grip_cmd_to_ctrl(0.5) == pytest.approx(127.5)
    assert g.grip_pos_from_driver(0.0, 0.0, 0.871) == 0.0
    assert g.grip_pos_from_driver(0.7822, 0.0, 0.871) == pytest.approx(0.898, abs=2e-3)  # empty-hand close
    assert g.grip_pos_from_driver(0.4) == pytest.approx(0.4 / 0.871)
    assert g.grip_pos_from_driver(-0.1) == 0.0
    assert g.grip_pos_from_driver(0.9) == 1.0
