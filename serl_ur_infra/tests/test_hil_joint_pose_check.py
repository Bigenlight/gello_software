"""Pure math checks for the dependency-light HIL joint pose probe."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "ros2_ur_ws" / "_hil_joint_pose_check.py"
_SPEC = importlib.util.spec_from_file_location("hil_joint_pose_check", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_wrist_and_base_use_nearest_equivalent_turn():
    current = [math.pi + 0.02, 0.0, 0.0, 0.0, 0.0, math.pi + 0.03]
    target = [-math.pi + 0.01, 0.0, 0.0, 0.0, 0.0, -math.pi + 0.01]

    delta = _MODULE.branch_safe_deltas(current, target)

    assert delta[0] == pytest.approx(0.01)
    assert delta[5] == pytest.approx(0.02)


def test_elbow_keeps_literal_difference_because_full_turn_is_infeasible():
    current = [0.0, 0.0, math.pi - 0.01, 0.0, 0.0, 0.0]
    target = [0.0, 0.0, -math.pi + 0.01, 0.0, 0.0, 0.0]

    delta = _MODULE.branch_safe_deltas(current, target)

    assert delta[2] == pytest.approx(2.0 * math.pi - 0.02)


def test_target_requires_six_finite_values():
    with pytest.raises(ValueError):
        _MODULE.parse_target("1,2,3")
    with pytest.raises(ValueError):
        _MODULE.parse_target("1,2,3,4,5,nan")
