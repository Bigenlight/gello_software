"""Pure validation tests for the shell arming deadman checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "ros2_ur_ws" / "_hil_deadman_check.py"
_SPEC = importlib.util.spec_from_file_location("hil_deadman_check", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_engaged_sample_returns_gain():
    assert _MODULE.validate_sample([1.0, 0.35]) == (True, 0.35)


def test_disengaged_sample_is_valid_but_not_armed():
    assert _MODULE.validate_sample([0.0, 1.0]) == (False, 1.0)


@pytest.mark.parametrize(
    "sample",
    ([1.0], [0.5, 0.5], [1.0, 0.09], [1.0, 1.01], [float("nan"), 0.5]),
)
def test_malformed_or_out_of_contract_sample_is_rejected(sample):
    with pytest.raises(ValueError):
        _MODULE.validate_sample(sample)
