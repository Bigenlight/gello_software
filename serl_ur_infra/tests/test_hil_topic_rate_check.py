"""Dependency-light tests for the bounded ROS topic rate helper."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "ros2_ur_ws" / "_hil_topic_rate_check.py"
_ACTOR = _ROOT / "ros2_ur_ws" / "run_hil_actor.sh"
_CAMERAS = _ROOT / "ros2_ur_ws" / "launch_cameras.sh"
_SPEC = importlib.util.spec_from_file_location("hil_topic_rate_check", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Stamp:
    sec = 12
    nanosec = 34


class _Header:
    stamp = _Stamp()


class _Message:
    header = _Header()


def test_header_stamp_is_converted_to_nanoseconds():
    assert _MODULE._stamp_ns(_Message()) == 12_000_000_034
    assert _MODULE._stamp_ns(object()) is None


def test_rate_uses_inter_sample_span():
    assert _MODULE._rate_hz([10.0, 10.1, 10.2, 10.3, 10.4]) == pytest.approx(10.0)


@pytest.mark.parametrize("values", ([], [1.0], [2.0, 2.0], [2.0, 1.0]))
def test_rate_rejects_non_advancing_or_insufficient_arrivals(values):
    with pytest.raises(ValueError):
        _MODULE._rate_hz(values)


def test_actor_preflight_uses_graceful_helper_not_ros2_topic_hz():
    text = _ACTOR.read_text()
    assert 'TOPIC_CHECKER="$SCRIPT_DIR/_hil_topic_rate_check.py"' in text
    assert 'python3 "$TOPIC_CHECKER"' in text
    assert not re.search(r"^\s*(?:timeout\s+\S+\s+)?ros2 topic hz", text, re.MULTILINE)
    assert "timeout --signal=TERM --kill-after=2" in text


def test_camera_launcher_uses_graceful_helper_not_ros2_topic_hz():
    text = _CAMERAS.read_text()
    assert 'TOPIC_CHECKER="${SCRIPT_DIR}/_hil_topic_rate_check.py"' in text
    assert 'python3 "${TOPIC_CHECKER}"' in text
    assert not re.search(r"^\s*(?:timeout\s+\S+\s+)?ros2 topic hz", text, re.MULTILINE)
