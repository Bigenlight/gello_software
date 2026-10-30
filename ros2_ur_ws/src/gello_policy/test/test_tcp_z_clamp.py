"""TCP Z-floor clamp (gello_policy.tcp_floor): offline, no rclpy.

Run from the sourced ROS workspace:
  source /opt/ros/jazzy/setup.bash; source install/setup.bash
  /usr/bin/python3 -m pytest src/gello_policy/test/test_tcp_z_clamp.py -q
"""

import math

import numpy as np
import pytest

from gello_policy import tcp_floor

TOOL_Z = 0.174
FLOOR = 0.013
# Carrot start pose (leader branch, pan ~ -3.16): TCP ~0.22 m above the table.
START = [-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692]


def _lower_pose(z_target, seed=START):
    """Search shoulder_lift/elbow around START for a pose whose TCP z ~= z_target."""
    best, best_err = None, 1e9
    for dl in np.linspace(-0.6, 0.6, 61):
        for de in np.linspace(-0.6, 0.6, 61):
            q = list(seed)
            q[1] += dl
            q[2] += de
            err = abs(tcp_floor.tcp_z(q, TOOL_Z) - z_target)
            if err < best_err:
                best, best_err = q, err
    assert best_err < 0.01, best_err
    return best


def test_start_pose_is_far_above_floor_and_untouched():
    result = tcp_floor.clamp_to_floor(START, START, FLOOR, TOOL_Z)
    assert not result.clamped and result.alpha == 1.0
    assert result.target == pytest.approx(START)
    assert result.z > 0.15


def test_periodicity_of_tcp_height():
    q = list(START)
    q[0] += math.tau
    q[5] -= math.tau
    assert tcp_floor.tcp_z(q, TOOL_Z) == pytest.approx(tcp_floor.tcp_z(START, TOOL_Z), abs=1e-9)


def test_target_below_floor_is_shortened_onto_floor():
    below = _lower_pose(-0.05)  # 5 cm under the table plane
    assert tcp_floor.tcp_z(below, TOOL_Z) < FLOOR
    result = tcp_floor.clamp_to_floor(START, below, FLOOR, TOOL_Z)
    assert result.clamped and 0.0 < result.alpha < 1.0
    assert result.z >= FLOOR
    assert result.z == pytest.approx(FLOOR, abs=1e-3)  # bisection resolution ~2^-12 of the step
    assert result.z_raw < FLOOR
    # Returned target lies on the live_q -> target segment.
    seg = np.asarray(START) + result.alpha * (np.asarray(below) - np.asarray(START))
    assert result.target == pytest.approx(seg.tolist(), abs=1e-12)


def test_target_above_floor_is_untouched():
    above = _lower_pose(0.05)
    result = tcp_floor.clamp_to_floor(START, above, FLOOR, TOOL_Z)
    assert not result.clamped and result.alpha == 1.0
    assert result.target == pytest.approx(above)


def test_live_already_below_floor_returns_live_pose():
    live = _lower_pose(-0.02)
    deeper = _lower_pose(-0.06, seed=live)
    result = tcp_floor.clamp_to_floor(live, deeper, FLOOR, TOOL_Z)
    assert result.clamped and result.alpha == 0.0
    assert result.target == pytest.approx(live)
    # ... but an upward request that clears the floor is still honoured.
    result_up = tcp_floor.clamp_to_floor(live, START, FLOOR, TOOL_Z)
    assert not result_up.clamped and result_up.target == pytest.approx(START)


def test_recorded_dataset_floor_matches_report():
    # A pose with TCP exactly at the dataset minimum (0.0131 m) must pass the 0.013 floor.
    at_min = _lower_pose(0.0131)
    result = tcp_floor.clamp_to_floor(START, at_min, FLOOR, TOOL_Z)
    assert not result.clamped
