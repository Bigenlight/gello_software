"""Unit tests for ur_gello_bringup.angle_utils.

These are the first tests in this workspace. They cover the branch-cut-safe
angle primitives (wrap_to_pi, circular_dist, wrapped_nearest) and the
SAFETY-CRITICAL leader_quasi_still gate, whose conservative-default contract
(insufficient evidence => NOT still) authorizes physical robot motion.
"""

import math

import pytest

from ur_gello_bringup.angle_utils import (
    circular_dist,
    leader_quasi_still,
    wrap_to_pi,
    wrapped_nearest,
)

PI = math.pi
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# wrap_to_pi
# --------------------------------------------------------------------------
def test_wrap_to_pi_zero():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)


def test_wrap_to_pi_pi():
    # +pi and -pi are the two representable endpoints; magnitude must be pi.
    assert abs(wrap_to_pi(PI)) == pytest.approx(PI)


def test_wrap_to_pi_neg_pi():
    assert abs(wrap_to_pi(-PI)) == pytest.approx(PI)


def test_wrap_to_pi_three_pi():
    # 3*pi == pi (mod 2*pi); wraps to +/-pi.
    assert abs(wrap_to_pi(3.0 * PI)) == pytest.approx(PI)


def test_wrap_to_pi_neg_three_pi():
    assert abs(wrap_to_pi(-3.0 * PI)) == pytest.approx(PI)


def test_wrap_to_pi_small_positive_unchanged():
    assert wrap_to_pi(0.5) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# circular_dist
# --------------------------------------------------------------------------
def test_circular_dist_symmetry():
    a, b = 1.3, -2.7
    assert circular_dist(a, b) == pytest.approx(circular_dist(b, a))


def test_circular_dist_branch_cut():
    # +3.1 and -3.1 straddle the +/-pi cut: physically ~0.083 rad apart,
    # NOT ~6.2 rad. This is the whole reason the helper exists.
    d = circular_dist(3.1, -3.1)
    assert d == pytest.approx(TWO_PI - 6.2, abs=1e-9)
    assert d == pytest.approx(0.0831853, abs=1e-6)
    assert d < 0.1


def test_circular_dist_identical_is_zero():
    assert circular_dist(2.0, 2.0) == pytest.approx(0.0)


def test_circular_dist_never_exceeds_pi():
    # The circular distance is bounded by pi for any inputs.
    for a in (-10.0, -3.0, 0.0, 3.0, 10.0):
        for b in (-7.0, -1.0, 0.5, 4.0, 9.0):
            assert circular_dist(a, b) <= PI + 1e-9


# --------------------------------------------------------------------------
# wrapped_nearest
# --------------------------------------------------------------------------
def test_wrapped_nearest_shifts_toward_reference():
    # Raw target 0.0 with reference ~2*pi should become ~2*pi (shifted by +2*pi),
    # i.e. the nearest angular equivalent to the reference.
    out = wrapped_nearest([0.0], [TWO_PI])
    assert out[0] == pytest.approx(TWO_PI, abs=1e-9)


def test_wrapped_nearest_within_pi_of_reference():
    # Result must always land within pi of the reference, per contract.
    target = [3.0, -3.0, 0.0, 6.0, -6.0, 12.0]
    reference = [0.0, 0.0, 5.0, -1.0, 7.0, 0.0]
    out = wrapped_nearest(target, reference)
    for o, r in zip(out, reference):
        assert abs(o - r) <= PI + 1e-9


def test_wrapped_nearest_preserves_physical_angle():
    # The shift is an integer multiple of 2*pi, so it never changes the
    # physical (wrapped) angle.
    target = [10.0, -8.5, 3.3]
    reference = [0.1, 0.2, 0.3]
    out = wrapped_nearest(target, reference)
    for o, t in zip(out, target):
        assert wrap_to_pi(o - t) == pytest.approx(0.0, abs=1e-9)


def test_wrapped_nearest_branch_cut_wrist():
    # A wrist neutral near +pi: actual pose 3.10, raw leader -3.10. Naively
    # the arm would spin ~2*pi; wrapped_nearest brings the target next to actual.
    out = wrapped_nearest([-3.10], [3.10])
    assert abs(out[0] - 3.10) < 0.1


# --------------------------------------------------------------------------
# leader_quasi_still — helpers
# --------------------------------------------------------------------------
def _history(dt, n, pose_fn, t0=1000.0):
    """Build an oldest-first (ts, pose) history: n samples spaced dt apart.

    pose_fn(i) -> list[float] gives the pose at sample i.
    """
    return [(t0 + i * dt, pose_fn(i)) for i in range(n)]


# --------------------------------------------------------------------------
# leader_quasi_still — conservative-default (insufficient evidence => False)
# --------------------------------------------------------------------------
def test_quasi_still_empty_history_false():
    assert leader_quasi_still([], window_s=0.3, max_speed=0.1) is False


def test_quasi_still_single_sample_false():
    hist = [(1000.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    assert leader_quasi_still(hist, window_s=0.3, max_speed=0.1) is False


def test_quasi_still_sparse_span_guard_false():
    # Two samples only 0.05 s apart with window_s=0.3: span (0.05) < half the
    # window (0.15) -> insufficient temporal coverage -> refuse (False).
    hist = _history(0.05, 2, lambda i: [0.0] * 6)
    assert leader_quasi_still(hist, window_s=0.3, max_speed=0.1) is False


# --------------------------------------------------------------------------
# leader_quasi_still — positive / negative behavior
# --------------------------------------------------------------------------
def test_quasi_still_perfectly_still_true():
    # 30 Hz over 0.5 s, perfectly stationary -> still.
    hist = _history(1.0 / 30.0, 15, lambda i: [0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    assert leader_quasi_still(hist, window_s=0.3, max_speed=0.1) is True


def test_quasi_still_one_joint_moving_false():
    # Same cadence, but joint 0 moves at 0.5 rad/s -> speed 0.5 > 0.1 -> not still.
    hist = _history(1.0 / 30.0, 15, lambda i: [0.5 * (i / 30.0), 0.0, 0.0, 0.0, 0.0, 0.0])
    assert leader_quasi_still(hist, window_s=0.3, max_speed=0.1) is False


def test_quasi_still_dither_across_pi_cut_true():
    # A wrist joint dithering across the +/-pi branch cut: physically tiny
    # motion, but a naive abs() would read ~2*pi. Circular distance keeps it
    # small, so the gate correctly reports still.
    def pose(i):
        val = 3.1415 if i % 2 == 0 else -3.1415
        return [val, 0.0, 0.0, 0.0, 0.0, 0.0]

    hist = _history(1.0 / 30.0, 15, pose)
    assert leader_quasi_still(hist, window_s=0.3, max_speed=0.1) is True


def test_quasi_still_dynamixel_noise_jitter_true():
    # Small Dynamixel-scale jitter (~0.003 rad excursions) over the window:
    # worst-case speed ~0.003/0.3 = 0.01 rad/s << 0.1 -> still.
    def pose(i):
        j = 0.003 if i % 2 == 0 else -0.003
        return [j, -j, j, -j, j, -j]

    hist = _history(1.0 / 30.0, 15, pose)
    assert leader_quasi_still(hist, window_s=0.3, max_speed=0.1) is True


def test_quasi_still_does_not_assume_six_joints():
    # The gate must iterate over len(pose), not a hard-coded 6.
    hist = _history(1.0 / 30.0, 15, lambda i: [0.0, 0.0, 0.0])
    assert leader_quasi_still(hist, window_s=0.3, max_speed=0.1) is True

    hist_move = _history(1.0 / 30.0, 15, lambda i: [0.0, 0.5 * (i / 30.0), 0.0])
    assert leader_quasi_still(hist_move, window_s=0.3, max_speed=0.1) is False
