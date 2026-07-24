#!/usr/bin/env python3
"""Unit test for the control_mode-derived handshake defaults in
``launch/ur7e_gello_real.launch.py``.

These lock in the START-ANCHORED no-motion bring-up wiring:

  * ``control_mode:=joint_delta`` must derive ``start_mode:='switch_only'`` and
    ``bridge_resume_service:='/gello_ur_bridge/joint_delta_start'`` (a STRICT
    switch in place + a chase-free anchor-at-current-pose re-arm), and
  * ``control_mode:=joint`` and ``control_mode:=eef`` must be UNCHANGED
    (``gello`` + ``/gello_ur_bridge/resume`` and ``switch_only`` +
    ``/gello_ur_bridge/eef_resume`` respectively).

This is the regression that proves the two-else-clause edit to the launch
derivation is additive: it evaluates the real ``PythonExpression`` substitutions
against a ``LaunchContext`` (no ROS graph, no spinning, no robot), so it stays
fast and runs in a pure venv wherever ``launch`` imports.
"""
import importlib.util
import os

import pytest

launch = pytest.importorskip("launch", reason="needs the ROS 2 launch package")
from launch import LaunchContext  # noqa: E402

_LAUNCH_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "launch",
    "ur7e_gello_real.launch.py",
)


def _load_launch_module():
    spec = importlib.util.spec_from_file_location(
        "ur7e_gello_real_launch", _LAUNCH_FILE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_launch_module()


def _perform(sub, start_mode="", bridge_resume_service=""):
    ctx = LaunchContext()
    ctx.launch_configurations["start_mode"] = start_mode
    ctx.launch_configurations["bridge_resume_service"] = bridge_resume_service
    return sub.perform(ctx)


@pytest.mark.parametrize(
    "control_mode, expected_start_mode, expected_resume_service",
    [
        ("joint", "gello", "/gello_ur_bridge/resume"),
        ("eef", "switch_only", "/gello_ur_bridge/eef_resume"),
        ("joint_delta", "switch_only", "/gello_ur_bridge/joint_delta_start"),
    ],
)
def test_auto_defaults_per_control_mode(
    control_mode, expected_start_mode, expected_resume_service
):
    """The empty-string sentinel (AUTO) resolves to the mode-specific default."""
    assert (
        _perform(_mod._auto_start_mode(control_mode)) == expected_start_mode
    )
    assert (
        _perform(_mod._auto_resume_service(control_mode))
        == expected_resume_service
    )


def test_explicit_start_mode_still_wins_for_joint_delta():
    """A non-empty start_mode launch arg overrides the AUTO derivation, so the
    old chase behaviour is one CLI argument away."""
    assert (
        _perform(_mod._auto_start_mode("joint_delta"), start_mode="gello")
        == "gello"
    )


def test_explicit_resume_service_still_wins_for_joint_delta():
    assert (
        _perform(
            _mod._auto_resume_service("joint_delta"),
            bridge_resume_service="/gello_ur_bridge/resume",
        )
        == "/gello_ur_bridge/resume"
    )


def _jd_overrides(start_mode="", control_mode="joint_delta"):
    """Evaluate _joint_delta_bridge_parameter_overrides against a bare context."""
    ctx = LaunchContext()
    ctx.launch_configurations["control_mode"] = control_mode
    ctx.launch_configurations["start_mode"] = start_mode
    # The builder iterates _JD_DOUBLE_OVERRIDES and .perform()s each name; give
    # them empty-string sentinels so they are omitted (yaml value wins).
    for name in _mod._JD_DOUBLE_OVERRIDES:
        ctx.launch_configurations[name] = ""
    return _mod._joint_delta_bridge_parameter_overrides(ctx)


def test_jd_start_allow_unstreamed_opt_in_only_for_switch_only():
    """The _has_streamed gate opt-in is scoped to the no-motion (switch_only)
    bring-up. The default AUTO start_mode resolves to switch_only, so the
    ordinary joint_delta bring-up opts in; but forcing start_mode:=gello (the old
    chase, which runs BEFORE the switch) must keep the gate LIVE so a
    mis-sequenced manual joint_delta_start cannot stream into an inactive
    controller."""
    # AUTO (empty) -> switch_only -> opt in.
    assert _jd_overrides(start_mode="")["jd_start_allow_unstreamed"] is True
    # Explicit switch_only -> opt in.
    assert (
        _jd_overrides(start_mode="switch_only")["jd_start_allow_unstreamed"] is True
    )
    # Forced chase -> gate stays live.
    assert _jd_overrides(start_mode="gello")["jd_start_allow_unstreamed"] is False
    # init_align is also a moving bring-up -> gate stays live.
    assert (
        _jd_overrides(start_mode="init_align")["jd_start_allow_unstreamed"] is False
    )
