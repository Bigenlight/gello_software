#!/usr/bin/env python3
"""Tests for the opt-in DISCRETE gripper mode in ``gello_gripper_bridge_node``.

Why this mode exists (measured, see ``docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md``
§5.1): across 125 takes / 52,607 samples the GELLO trigger is already effectively
binary, but once or twice per session the spring fails to fully return and the
trigger comes to REST PARTWAY OPEN — worst observed 0.229 and 0.243. The bridge
forwards that faithfully, so the Robotiq only opens ~76%. Every observed outlier
is below 0.3, so snapping <= 0.3 to exactly 0.0 fixes all of them. The test named
after those two values is the one that encodes the whole point of the feature.

Kept deliberately small: the latch itself is a pure function, so most of this
file needs no ROS at all. The few node-level tests follow the convention already
established by ``test_bridge_stages.py`` — construct the real node, call its
callbacks directly, no executor and no spinning. They additionally remap the node
name and both topics onto throwaway names so that running this suite on a machine
with a LIVE teleop stack cannot collide with the real bridge.
"""

import pytest

from ur_gello_bringup.gello_gripper_bridge_node import (
    DISCRETE_CLOSED,
    DISCRETE_OPEN,
    discrete_latch,
    validate_discrete_thresholds,
)

CLOSE_AT = 0.7
OPEN_AT = 0.3


def latch(value, previous):
    return discrete_latch(value, previous, OPEN_AT, CLOSE_AT)


# --------------------------------------------------------------------------- #
# Pure latch logic                                                             #
# --------------------------------------------------------------------------- #
def test_thresholds_map_to_endpoints():
    """<= 0.3 latches OPEN, >= 0.7 latches CLOSED — inclusive on both bounds,
    from either previous state (a crossing always wins over hysteresis)."""
    for previous in (None, DISCRETE_OPEN, DISCRETE_CLOSED):
        for value in (0.0, 0.229, 0.3):
            assert latch(value, previous) == DISCRETE_OPEN
        for value in (0.7, 0.98, 1.0):
            assert latch(value, previous) == DISCRETE_CLOSED


def test_hysteresis_holds_state_between_thresholds():
    """Inside the 0.3..0.7 band the previous latch is kept, so a trigger sweeping
    through the middle produces no chatter — that is why there are two thresholds
    and not one."""
    # Sweeping closed -> open: stays CLOSED all the way down to 0.3.
    state = DISCRETE_CLOSED
    for value in (0.69, 0.5, 0.31):
        state = latch(value, state)
        assert state == DISCRETE_CLOSED
    state = latch(0.3, state)
    assert state == DISCRETE_OPEN
    # And back up: stays OPEN until 0.7.
    for value in (0.31, 0.5, 0.69):
        state = latch(value, state)
        assert state == DISCRETE_OPEN
    assert latch(0.7, state) == DISCRETE_CLOSED
    # Unknown stays unknown inside the band — no guessing.
    assert latch(0.5, None) is None


def test_real_world_resting_outliers_0229_and_0243_resolve_to_open():
    """THE point of the feature: the two worst observed resting values snap to a
    FULL open (exactly 0.0), instead of being forwarded as a ~76%-closed gripper.

    Both arrive while the latch is CLOSED (the operator has just released a
    grasp), which is the case that matters — hysteresis must not swallow them.
    """
    for outlier in (0.229, 0.243):
        assert outlier < OPEN_AT, "documented outliers must sit below open_at"
        assert latch(outlier, DISCRETE_CLOSED) == DISCRETE_OPEN
        assert latch(outlier, DISCRETE_OPEN) == DISCRETE_OPEN


def test_invalid_thresholds_are_rejected():
    """0.0 < open_at < close_at < 1.0. Everything else is a reason string."""
    assert validate_discrete_thresholds(0.3, 0.7) is None
    assert validate_discrete_thresholds(0.7, 0.3) is not None  # inverted
    assert validate_discrete_thresholds(0.5, 0.5) is not None  # zero-width band
    assert validate_discrete_thresholds(0.0, 0.7) is not None  # open_at at bound
    assert validate_discrete_thresholds(0.3, 1.0) is not None  # close_at at bound
    assert validate_discrete_thresholds(float("nan"), 0.7) is not None


# --------------------------------------------------------------------------- #
# NODE-LEVEL tests (rclpy). Same pattern as test_bridge_stages.py: build the     #
# real node, call callbacks directly, no executor / no spinning / no robot.      #
# --------------------------------------------------------------------------- #
rclpy = pytest.importorskip("rclpy", reason="node-level tests need a ROS 2 env")

from ur_gello_bringup import gello_gripper_bridge_node as bridge_mod  # noqa: E402
from ur_gello_bringup.gello_gripper_bridge_node import (  # noqa: E402
    GelloGripperBridge,
)
from std_msgs.msg import Float32  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402


class _Clock:
    """Deterministic stand-in for the bridge module's ``time``.

    The node only ever asks for ``time.monotonic()``, so a two-method object is
    a complete substitute. Real wall clock would make the slew-budget assertions
    (which are exact arithmetic on ``elapsed``) machine-load dependent.
    """

    def __init__(self, t0=1000.0):
        self.t = float(t0)

    def monotonic(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)
        return self.t


class _Bridge:
    """Context manager: a real GelloGripperBridge with captured publishes.

    The node name and BOTH topics are remapped to throwaway names: this suite
    must be safe to run while the live teleop stack owns the real ones.
    """

    def __init__(self, clock=None, **params):
        self._params = params
        self._clock = clock
        self._real_time = None
        self.node = None
        self.published = []
        self.states = []
        self.triggers = []

    def __enter__(self):
        args = [
            "--ros-args",
            "-r",
            "__node:=gello_gripper_bridge_pytest",
            "-p",
            "input_topic:=/pytest_only/gripper_width",
            "-p",
            "output_topic:=/pytest_only/command_percent",
            "-p",
            "actual_topic:=/pytest_only/position_percent",
            "-p",
            "state_publish_rate_hz:=0.0",  # no timer; we call the method directly
        ]
        for k, v in self._params.items():
            args += ["-p", f"{k}:={v}"]
        if self._clock is not None:
            self._real_time = bridge_mod.time
            bridge_mod.time = self._clock
        rclpy.init(args=args)
        self.node = GelloGripperBridge()
        # Swap the real publishers out BEFORE any callback can run.
        self.node._pub = _CapturePub(self.published)
        self.node._state_pub = _CapturePub([])
        self.node._discrete_state_pub = _CapturePub(self.states)
        self.node._discrete_trigger_pub = _CapturePub(self.triggers)
        return self

    def __exit__(self, *exc):
        try:
            self.node.destroy_node()
        finally:
            rclpy.shutdown()
            if self._real_time is not None:
                bridge_mod.time = self._real_time
        return False

    def width(self, value):
        msg = Float32()
        msg.data = float(value)
        self.node._on_width(msg)

    def actual(self, value):
        msg = Float32()
        msg.data = float(value)
        self.node._on_actual(msg)

    def pause(self):
        return self.node._on_pause(None, Trigger.Response())

    def resume(self):
        return self.node._on_resume(None, Trigger.Response())


class _CapturePub:
    def __init__(self, sink):
        self._sink = sink

    def publish(self, msg):
        self._sink.append(msg.data)


def _go_home_and_resume(b, clk):
    """Reproduce the task recorder's GO HOME cycle, which fires once per take.

    A take ends with the object grasped (trigger 0.8 -> latch CLOSED); GO HOME
    pauses the bridge and then force-opens the Robotiq by publishing 0.0 straight
    to the command topic, BYPASSING this paused bridge — modelled here by the
    position feedback going to 0.0 with no publish of our own. The operator then
    presses Gripper Resume. Leaves ``b.published`` cleared.
    """
    b.width(0.8)
    b.pause()
    b.actual(0.0)
    clk.advance(0.1)
    b.width(0.5)  # fresh leader sample so resume's staleness gate passes
    assert b.resume().success is True
    b.published.clear()


def test_nothing_is_published_before_the_first_threshold_crossing():
    """The initial latch is UNKNOWN and we do NOT guess an endpoint: until the
    trigger actually crosses a threshold the gripper keeps whatever it has."""
    with _Bridge(discrete_mode=True) as b:
        for value in (0.5, 0.4, 0.6, 0.45):
            b.width(value)
        assert b.published == []
        assert b.node.discrete_state_token() == "UNKNOWN"
        # First crossing publishes the endpoint, and the token follows.
        b.width(0.8)
        assert b.published == [DISCRETE_CLOSED]
        assert b.node.discrete_state_token() == "CLOSED"
        # The documented outlier now snaps to a full open.
        b.width(0.229)
        assert b.published == [DISCRETE_CLOSED, DISCRETE_OPEN]
        assert b.node.discrete_state_token() == "OPEN"


def test_invalid_thresholds_fall_back_to_continuous_without_killing_the_node():
    """A mis-set threshold must never make the gripper unpredictable, and must
    never take the process down (arm teleop depends on it staying up)."""
    with _Bridge(discrete_mode=True, discrete_open_at=0.8, discrete_close_at=0.2) as b:
        assert b.node.discrete_mode is False
        assert b.node._discrete_fallback_reason is not None
        assert b.node.discrete_state_token() == "DISABLED"
        b.width(0.229)
        assert b.published == [pytest.approx(0.229)]


def test_discrete_mode_false_reproduces_the_continuous_value_untouched():
    """The default is off, and off means byte-identical to the old behaviour:
    the leader value is forwarded, mid-band values included."""
    with _Bridge() as b:
        assert b.node.discrete_mode is False
        assert b.node.discrete_state_token() == "DISABLED"
        for value in (0.229, 0.5, 0.98):
            b.width(value)
        assert b.published == [
            pytest.approx(0.229),
            pytest.approx(0.5),
            pytest.approx(0.98),
        ]


# --------------------------------------------------------------------------- #
# Pause / resume interaction with the latch                                     #
# --------------------------------------------------------------------------- #
def test_the_latch_freezes_across_a_pause_and_is_unknown_after_a_resume():
    """Intent is FROZEN for the whole pause (a flopping leader must not re-decide
    it) and then RESET, so an accepted resume can never replay a pre-pause CLOSED
    onto a gripper that was force-opened behind this bridge's back."""
    with _Bridge(discrete_mode=True) as b:
        b.width(0.8)  # take ends with the object grasped
        assert b.published == [DISCRETE_CLOSED]
        b.pause()
        b.width(0.5)  # leader collapses mid-band while paused
        assert b.published == [DISCRETE_CLOSED], "paused: nothing is republished"
        assert b.node._discrete_latch == DISCRETE_CLOSED, "intent frozen, not lost"
        assert b.node.discrete_state_token() == "CLOSED", "and still reported"
        b.actual(0.0)  # GO HOME force-opened the Robotiq
        b.width(0.5)
        assert b.resume().success is True
        assert b.node._discrete_latch is None, "resume re-earns intent from scratch"


def test_the_three_post_resume_cases():
    """First sample after a GO HOME resume, all three trigger ranges."""
    # (a) trigger >= close_at: re-latches CLOSED on the first sample, exactly as
    # before this fix — and still ramps rather than jumping.
    clk = _Clock()
    with _Bridge(clock=clk, discrete_mode=True) as b:
        _go_home_and_resume(b, clk)
        clk.advance(0.033)
        b.width(0.9)
        assert b.node._discrete_latch == DISCRETE_CLOSED
        assert b.published == [pytest.approx(0.6 * 0.033)]

    # (b) inside the band: publishes NOTHING, so the Robotiq holds what it has.
    # This is the slam-shut case — it used to ramp to the frozen CLOSED endpoint.
    clk = _Clock()
    with _Bridge(clock=clk, discrete_mode=True) as b:
        _go_home_and_resume(b, clk)
        for value in (0.45, 0.5, 0.69):
            clk.advance(0.033)
            b.width(value)
        assert b.published == []
        assert b.node._discrete_latch is None

    # (c) trigger <= open_at: latches OPEN, and since GO HOME already left the
    # gripper at 0.0 the ramp has nothing to travel — it stays open.
    clk = _Clock()
    with _Bridge(clock=clk, discrete_mode=True) as b:
        _go_home_and_resume(b, clk)
        clk.advance(0.033)
        b.width(0.2)
        assert b.node._discrete_latch == DISCRETE_OPEN
        assert b.published == [pytest.approx(DISCRETE_OPEN)]


def test_a_suppressed_sample_cannot_bank_slew_budget():
    """The ramp's slew budget is ``resume_slew_per_s * elapsed``, so ``elapsed``
    must count only time in which the output was free to move. Both leaks are
    arithmetic, so assert the arithmetic."""
    # A latch that stays UNKNOWN for 1.9 s of the 2.0 s window publishes nothing;
    # pre-fix that banked 0.6 * 1.933 = 1.16 > full stroke, so the first real
    # sample WAS the endpoint and the ramp degenerated into a single-step jump.
    clk = _Clock()
    with _Bridge(clock=clk, discrete_mode=True) as b:
        _go_home_and_resume(b, clk)
        clk.advance(1.9)
        b.width(0.45)
        assert b.published == []
        assert b.node._last_pub_time == pytest.approx(clk.t), "suppressed = re-based"
        clk.advance(0.033)
        b.width(0.9)
        assert clk.t < b.node._ramp_until, "still inside the ramp window"
        assert b.published == [pytest.approx(0.6 * 0.033)]
        assert b.published[0] < DISCRETE_CLOSED

    # And a leader that goes SILENT mid-ramp re-bases nothing, so elapsed is
    # capped at staleness_timeout_s (0.5) instead of the full 1.9 s gap.
    clk = _Clock()
    with _Bridge(clock=clk, discrete_mode=True) as b:
        _go_home_and_resume(b, clk)
        clk.advance(1.9)
        b.width(0.9)
        assert b.published == [pytest.approx(0.6 * 0.5)]


def test_discrete_state_reports_ramping_during_the_ramp_only():
    """The wire carries intermediate slew values during the ramp, so the operator
    indicator must not advertise a settled endpoint there. Continuous mode is
    untouched — it still reports DISABLED."""
    clk = _Clock()
    with _Bridge(clock=clk, discrete_mode=True) as b:
        _go_home_and_resume(b, clk)
        b.node._publish_state()
        assert b.states[-1] == "RAMPING"
        assert b.triggers[-1] == pytest.approx(0.5), "live clamped trigger value"
        clk.advance(0.033)
        b.width(0.9)  # latch commits to CLOSED, but the output is mid-travel
        b.node._publish_state()
        assert b.states[-1] == "RAMPING"
        assert b.published[-1] < DISCRETE_CLOSED
        assert b.triggers[-1] == pytest.approx(0.9)
        clk.advance(5.0)  # window over: the endpoint is real now
        b.node._publish_state()
        assert b.states[-1] == "CLOSED"

    clk = _Clock()
    with _Bridge(clock=clk) as b:  # continuous: the ramp still runs, the token does not
        _go_home_and_resume(b, clk)
        assert clk.t < b.node._ramp_until
        b.node._publish_state()
        assert b.states[-1] == "DISABLED"
