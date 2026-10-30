import math

import pytest

from gello_policy.joint_angles import (
    angular_deviations,
    nearest_equivalent,
    positions_near_reference,
)


def test_two_pi_equivalent_pose_has_zero_deviation():
    start = [3.106, -1.817, 1.653, -1.618, -1.628, -3.195]
    live = list(start)
    live[-1] += math.tau

    assert angular_deviations(live, start) == pytest.approx([0.0] * 6)
    assert positions_near_reference(live, start) == pytest.approx(start)


def test_wrapped_mock_wrist_is_mapped_to_model_convention():
    assert nearest_equivalent(3.0881853071795864, -3.195) == pytest.approx(-3.195)


def test_genuine_deviation_is_not_hidden_by_periodicity():
    start = [0.0] * 6
    live = [0.0, 0.0, 0.0, 0.0, 0.0, math.tau + 0.25]

    assert angular_deviations(live, start)[-1] == pytest.approx(0.25)


def test_mismatched_vector_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        positions_near_reference([0.0], [0.0, 1.0])
