"""Pure tests for the ROS backend's fixed-rate joint trajectory generator.

No ROS node, executor, publisher, or wall-clock sleep is used here. Explicit
timestamps advance a deterministic 250 Hz command stream so position-step,
acceleration, braking, staleness, and reset invariants can be checked directly.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.envs.ros_backend import AccelerationLimitedJointStream  # noqa: E402


HZ = 250.0
DT = 1.0 / HZ
MAX_STEP = 0.0025
MAX_ACCEL = 8.0
STALE_S = 0.30
TOL = 2e-12


def _stream(*, soft_start_s=0.0):
    return AccelerationLimitedJointStream(
        hz=HZ,
        max_step_rad=MAX_STEP,
        max_accel_rad_s2=MAX_ACCEL,
        soft_start_s=soft_start_s,
        target_stale_s=STALE_S,
        soft_start_fraction=0.15,
    )


def _fresh_step(stream, target, tick):
    now = tick * DT
    return stream.advance(target, now=now, target_time=now)


def test_measured_seed_is_zero_jump_and_soft_start_lasts_point_seven_seconds():
    stream = _stream(soft_start_s=0.7)
    measured = np.array([0.4, -1.2, 1.5, -0.7, 0.3, -2.0])
    target = measured + 2.0

    first = stream.seed(measured, now=10.0)
    np.testing.assert_array_equal(first, measured)
    np.testing.assert_array_equal(stream.velocity, np.zeros(6))

    previous = first
    # Throughout the 0.7 s window every emitted step remains below the same
    # 15% -> 100% cap used by the live GELLO bridge.
    for tick in range(1, 176):
        now = 10.0 + tick * DT
        current = stream.advance(target, now=now, target_time=now)
        ramp = min(tick * DT / 0.7, 1.0)
        cap = MAX_STEP * (0.15 + 0.85 * ramp)
        assert np.max(np.abs(current - previous)) <= cap + TOL
        previous = current

    # At the end of the ramp a distant target is allowed to use, but never
    # exceed, the preserved 0.0025 rad command-step ceiling.
    post_ramp_step = _fresh_step(stream, target, 2676) - previous
    assert np.max(np.abs(post_ramp_step)) <= MAX_STEP + TOL


def test_maximum_per_tick_step_is_preserved():
    stream = _stream()
    target = np.array([2.0, -2.0, 1.5, -1.5, 1.0, -1.0])
    positions = [stream.seed(np.zeros(6), now=0.0)]
    for tick in range(1, 400):
        positions.append(_fresh_step(stream, target, tick))

    steps = np.diff(np.asarray(positions), axis=0)
    assert np.max(np.abs(steps)) <= MAX_STEP + TOL
    assert np.max(np.abs(steps)) == pytest.approx(MAX_STEP, abs=TOL)


def test_maximum_acceleration_is_never_exceeded():
    stream = _stream()
    positions = [stream.seed(np.zeros(3), now=0.0)]
    for tick in range(1, 80):
        target = np.ones(3) if tick < 35 else -np.ones(3)
        positions.append(_fresh_step(stream, target, tick))

    steps = np.diff(np.asarray(positions), axis=0)
    # seed() establishes zero command velocity before the first trajectory step.
    steps = np.vstack([np.zeros((1, 3)), steps])
    acceleration = np.diff(steps, axis=0) * HZ * HZ
    assert np.max(np.abs(acceleration)) <= MAX_ACCEL + TOL
    assert np.max(np.abs(acceleration)) == pytest.approx(MAX_ACCEL, abs=TOL)


def test_braking_distance_reaches_fixed_targets_without_overshoot():
    stream = _stream()
    start = np.zeros(4)
    target = np.array([0.1003, -0.0717, 0.0131, -0.0049])
    positions = [stream.seed(start, now=0.0)]
    for tick in range(1, 600):
        positions.append(_fresh_step(stream, target, tick))

    positions = np.asarray(positions)
    direction = np.sign(target - start)
    progress = (positions - start) * direction
    distance = np.abs(target - start)
    assert np.all(progress >= -TOL)
    assert np.all(progress <= distance + TOL)
    np.testing.assert_allclose(positions[-1], target, atol=TOL, rtol=0.0)
    np.testing.assert_allclose(stream.velocity, 0.0, atol=TOL, rtol=0.0)


def test_target_reversal_brakes_before_velocity_changes_sign():
    stream = _stream()
    stream.seed(np.zeros(1), now=0.0)
    velocities = [0.0]
    for tick in range(1, 31):
        _fresh_step(stream, np.ones(1), tick)
        velocities.append(float(stream.velocity[0]))

    before_reversal = velocities[-1]
    for tick in range(31, 90):
        _fresh_step(stream, -np.ones(1), tick)
        velocities.append(float(stream.velocity[0]))

    velocities = np.asarray(velocities)
    first_after_reversal = velocities[31]
    assert before_reversal == pytest.approx(MAX_STEP * HZ)
    assert 0.0 < first_after_reversal < before_reversal
    assert np.any(velocities[32:] < 0.0), "trajectory never completed reversal"
    acceleration = np.diff(velocities) * HZ
    assert np.max(np.abs(acceleration)) <= MAX_ACCEL + TOL


def test_stale_timestamp_brakes_then_holds_current_stream_position():
    stream = _stream()
    stream.seed(np.zeros(1), now=0.0)
    for tick in range(1, 31):
        _fresh_step(stream, np.ones(1), tick)
    velocity_before_stale = float(stream.velocity[0])

    positions = []
    speeds = []
    modes = []
    stale_target_time = 0.0
    for tick in range(101, 141):  # now > 0.30 s: target timestamp is stale
        positions.append(
            float(stream.advance(np.ones(1), tick * DT, stale_target_time)[0])
        )
        speeds.append(abs(float(stream.velocity[0])))
        modes.append(stream.mode)

    assert modes[0] == stream.BRAKING
    assert speeds[0] < velocity_before_stale
    assert np.all(np.diff(speeds) <= TOL)
    assert stream.HOLD in modes
    first_hold = modes.index(stream.HOLD)
    np.testing.assert_allclose(speeds[first_hold:], 0.0, atol=TOL, rtol=0.0)
    np.testing.assert_allclose(
        positions[first_hold:], positions[first_hold], atol=TOL, rtol=0.0
    )
    assert positions[-1] < 1.0, "stale target was chased instead of held"


def test_explicit_hold_target_is_the_acceleration_limited_stop_point():
    stream = _stream()
    stream.seed(np.zeros(1), now=0.0)
    positions = [0.0]
    for tick in range(1, 101):
        positions.append(float(_fresh_step(stream, np.ones(1), tick)[0]))

    old_target = 1.0
    hold_target = stream.braking_endpoint()
    assert float(hold_target[0]) < old_target
    assert float(hold_target[0]) > positions[-1]

    for tick in range(101, 181):
        positions.append(float(_fresh_step(stream, hold_target, tick)[0]))

    np.testing.assert_allclose(stream.position, hold_target, atol=TOL, rtol=0.0)
    np.testing.assert_allclose(stream.velocity, 0.0, atol=TOL, rtol=0.0)
    assert max(positions) <= float(hold_target[0]) + TOL


def test_reset_drops_position_velocity_mode_and_soft_start_state():
    stream = _stream(soft_start_s=0.7)
    stream.seed(np.zeros(2), now=0.0)
    _fresh_step(stream, np.ones(2), 1)
    assert np.any(stream.velocity != 0.0)

    stream.reset()
    assert not stream.seeded
    assert stream.position is None
    assert stream.velocity is None
    assert stream.mode == stream.UNSEEDED
    with pytest.raises(RuntimeError, match=r"seed\(\)"):
        stream.advance(np.ones(2), now=1.0, target_time=1.0)

    measured = np.array([1.25, -0.75])
    np.testing.assert_array_equal(stream.seed(measured, now=2.0), measured)
    np.testing.assert_array_equal(stream.velocity, np.zeros(2))
