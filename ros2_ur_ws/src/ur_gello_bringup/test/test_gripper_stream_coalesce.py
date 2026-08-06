#!/usr/bin/env python3
"""Tests for the COALESCING streaming rate limiter in ``robotiq_gripper_modbus_node``.

Why this exists (reproduced on real hardware, 2026-08-06): a fast trigger release
left the gripper parked at 0.043 / 0.055 / 0.11 / 0.42 instead of full open —
**7 of 14 open cycles**. The chain was:

* the bridge publishes each setpoint exactly ONCE, only when it changes;
* the driver's rate limiter *discarded* any setpoint arriving inside its 50 ms
  window, i.e. it threw away the FRESHEST sample;
* nobody re-asserts, and ``_last_cmd_pct`` is only updated on a real write, so the
  driver's own view stays perfectly self-consistent — nothing can notice the loss.

A ~30 Hz stream is accepted at ~20 Hz, so the last sample of a release (the one
that says "fully open") landed in a closed window about half the time. Closing hid
the same bug because the fingers bottom out anyway; opening has no such stop.

``test_a_burst_faster_than_the_rate_limit_still_ends_on_the_last_value`` and
``test_a_release_ending_inside_a_closed_window_still_reaches_full_open`` are the
regression: they fail against the pre-fix "return and forget" limiter.

Convention follows ``test_gripper_discrete.py``: build the real node, call its
callbacks directly, no executor and no spinning. The node name AND namespace are
remapped onto throwaway names so this suite can never collide with a live gripper
stack, and the Modbus driver is a fake — these tests never touch hardware.
"""

import pytest

rclpy = pytest.importorskip("rclpy", reason="node-level tests need a ROS 2 env")

from std_msgs.msg import Float32  # noqa: E402
from std_srvs.srv import SetBool  # noqa: E402

from ur_gello_bringup import robotiq_gripper_modbus_node as modbus_mod  # noqa: E402
from ur_gello_bringup.robotiq_gripper_modbus_node import (  # noqa: E402
    RobotiqGripperModbusNode,
)
from ur_gello_bringup.robotiq_2f85_modbus import GripperError  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class _Clock:
    """Deterministic stand-in for the node module's ``time``.

    The rate limiter is exact arithmetic on ``monotonic()`` deltas, so a real wall
    clock would make every window assertion machine-load dependent. ``time()`` and
    ``sleep()`` are here only because the action path uses them.
    """

    def __init__(self, t0=1000.0):
        self.t = float(t0)

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def sleep(self, _dt):  # the action's poll loop; the fake status ends it at once
        return None

    def advance(self, dt):
        self.t += float(dt)
        return self.t


class _FakeGripper:
    """Stands in for ``Robotiq2F85``. Records every position written to the bus."""

    def __init__(self):
        self.moves = []       # positions successfully written, in order
        self.attempts = []    # positions attempted, including failures
        self.fail_next = 0    # how many upcoming move() calls should raise
        self.closed = False
        self._pos = 0

    def move(self, pos, speed=150, force=50):
        self.attempts.append(int(pos))
        if self.fail_next > 0:
            self.fail_next -= 1
            raise GripperError("synthetic bus failure")
        self._pos = int(pos)
        self.moves.append(int(pos))

    def read_status(self):
        return {"gACT": 1, "gGTO": 1, "gSTA": 3, "gOBJ": 3, "gFLT": 0,
                "gPR": self._pos, "gPO": self._pos, "gCU": 0}

    def get_position(self):
        return self._pos

    def stop(self):
        return None

    def close(self):
        self.closed = True


class _Goal:
    """Minimal GripperCommand goal handle; ``_execute`` only uses these members."""

    class _Cmd:
        def __init__(self, position, max_effort):
            self.position = float(position)
            self.max_effort = float(max_effort)

    class _Req:
        def __init__(self, position, max_effort):
            self.command = _Goal._Cmd(position, max_effort)

    def __init__(self, position, max_effort=0.0):
        self.request = _Goal._Req(position, max_effort)
        self.is_cancel_requested = False
        self.feedback = []
        self.outcome = None

    def publish_feedback(self, fb):
        self.feedback.append(fb)

    def succeed(self):
        self.outcome = "succeed"

    def abort(self):
        self.outcome = "abort"

    def canceled(self):
        self.outcome = "canceled"


class _Node:
    """Context manager: the real node, offline, with a fake gripper attached."""

    def __init__(self, clock=None, connected=True, **params):
        self._params = params
        self._clock = clock
        self._real_time = None
        self.node = None
        self.gripper = None
        self._connected = connected

    def __enter__(self):
        args = [
            "--ros-args",
            "-r", "__ns:=/pytest_only",
            "-r", "__node:=robotiq_gripper_pytest",
            "-p", "connect_on_start:=false",   # no reconnect thread, no sockets
            "-p", "status_rate_hz:=0.0",       # no status timer, no bus polling
        ]
        for k, v in self._params.items():
            args += ["-p", f"{k}:={v}"]
        if self._clock is not None:
            self._real_time = modbus_mod.time
            modbus_mod.time = self._clock
        rclpy.init(args=args)
        self.node = RobotiqGripperModbusNode()
        self.gripper = _FakeGripper()
        self.attach(self.gripper, connected=self._connected)
        return self

    def __exit__(self, *exc):
        try:
            self.node.destroy_node()
        finally:
            rclpy.shutdown()
            if self._real_time is not None:
                modbus_mod.time = self._real_time
        return False

    # -- helpers ---------------------------------------------------------- #
    def attach(self, gripper, connected=True):
        self.node._g = gripper
        self.node._connected = bool(connected)
        return gripper

    def command(self, pct):
        msg = Float32()
        msg.data = float(pct)
        self.node._on_command_percent(msg)

    def tick(self):
        """One flush-timer firing."""
        self.node._try_flush_pending()

    def set_closed(self, closed):
        req = SetBool.Request()
        req.data = bool(closed)
        return self.node._on_set_closed(req, SetBool.Response())


def _pos(pct):
    """The node's percent -> Robotiq position mapping, so tests state intent."""
    return int(round(pct * 255.0))


# --------------------------------------------------------------------------- #
# FIX A — the rate limiter coalesces instead of dropping                        #
# --------------------------------------------------------------------------- #
def test_a_burst_faster_than_the_rate_limit_still_ends_on_the_last_value():
    """THE regression. A 30 Hz burst against a 20 Hz limiter ends on a sample that
    lands inside a closed window; the stream then goes quiet forever. The last
    value the operator asked for must still reach the bus."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(1.0)                       # accepted: nothing written yet
        for value in (0.8, 0.6, 0.4, 0.2, 0.0):
            clk.advance(0.033)               # ~30 Hz leader
            n.command(value)

        # Half the samples are coalesced away (that part is correct and desirable
        # -- the bus is single-client), but the FINAL one is still owed.
        assert n.gripper.moves == [_pos(1.0), _pos(0.6), _pos(0.2)]
        assert n.node._pending_pct == pytest.approx(0.0)
        assert n.node._last_cmd_pct == pytest.approx(0.2), \
            "pre-fix the gripper parked here -- 20% closed instead of open"

        clk.advance(0.025)                   # flush timer period at 20 Hz
        n.tick()
        assert n.gripper.moves[-1] == _pos(0.0), "the last desire reached the bus"
        assert n.node._last_cmd_pct == pytest.approx(0.0)
        assert n.node._pending_pct is None


def test_a_release_ending_inside_a_closed_window_still_reaches_full_open():
    """The measured hardware failure, in the shape it actually occurred: a smooth
    ~30 Hz release ramp whose last sample (full open) falls in a closed window."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        ramp = [round(0.95 - 0.05 * i, 2) for i in range(20)]
        assert ramp[-1] == 0.0
        for value in ramp:
            n.command(value)
            clk.advance(0.033)

        parked = n.gripper.moves[-1]
        assert parked > _pos(0.0), "reproduces the parked-short symptom"

        clk.advance(0.025)
        n.tick()
        assert n.gripper.moves[-1] == _pos(0.0)
        assert n.node._pending_pct is None


def test_the_flush_timer_is_a_no_op_with_nothing_pending():
    """Load-bearing: the :54321 forwarder is single-client, so the timer must add
    ZERO bus traffic at rest. Not one write, not one status read."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        for _ in range(20):
            clk.advance(0.025)
            n.tick()
        assert n.gripper.attempts == []
        assert n.node._pending_pct is None


def test_nothing_is_rewritten_once_the_stream_has_settled():
    """After the last setpoint is flushed the timer goes quiet again — it must not
    keep re-asserting (that would both storm the bus and fight an object grasp)."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(0.6)
        clk.advance(0.033)
        n.command(0.4)                      # deferred
        clk.advance(0.025)
        n.tick()                            # flushed
        assert n.gripper.moves == [_pos(0.6), _pos(0.4)]
        for _ in range(40):
            clk.advance(0.025)
            n.tick()
        assert n.gripper.moves == [_pos(0.6), _pos(0.4)], "silent at rest"


def test_the_flush_timer_runs_on_the_command_callback_group():
    """The group keeps the timer from interleaving with the SUBSCRIPTION that feeds
    ``_pending_pct`` (both are real entities, so the executor's can_execute gate
    applies to them). It does NOT cover the action — see the bus-ownership section
    below — so this pins half the safety argument, not all of it."""
    with _Node() as n:
        timer = n.node._cmd_flush_timer
        assert timer is not None
        assert timer.callback_group is n.node._cmd_cb
        assert timer.timer_period_ns == pytest.approx(
            n.node._cmd_min_period / 2.0 * 1e9, rel=1e-6
        ), "period is half the rate-limit window"


def test_deadband_still_suppresses_sub_threshold_changes():
    """Coalescing must not turn the deadband into a write amplifier: a change
    smaller than command_deadband is satisfied by what is already on the wire."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(0.40)
        assert n.gripper.moves == [_pos(0.40)]
        clk.advance(1.0)                     # window wide open
        n.command(0.405)                     # 0.005 < 0.01 deadband
        assert n.gripper.moves == [_pos(0.40)]
        assert n.node._pending_pct is None, "nothing owed -- do not retry forever"
        for _ in range(10):
            clk.advance(0.025)
            n.tick()
        assert n.gripper.moves == [_pos(0.40)]
        n.command(0.42)                      # 0.02 >= deadband: through it goes
        assert n.gripper.moves == [_pos(0.40), _pos(0.42)]


def test_a_sub_threshold_sample_does_not_erase_a_bigger_pending_one():
    """The pending slot always holds the FRESHEST desire, and the deadband is
    evaluated against what was written -- not against the previous sample."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(0.40)
        clk.advance(0.010)
        n.command(0.80)                      # deferred (0.010 < 0.05)
        clk.advance(0.010)
        n.command(0.805)                     # fresher, still far from 0.40
        assert n.node._pending_pct == pytest.approx(0.805)
        clk.advance(0.040)
        n.tick()
        assert n.gripper.moves == [_pos(0.40), _pos(0.805)]


def test_an_out_of_range_setpoint_is_clamped_before_it_is_stashed():
    """The pending slot must hold a legal percent. An unclamped 1.5 would survive
    into ``_last_cmd_pct`` and poison every later deadband comparison (the driver
    clamps the wire value, so nothing else would ever notice)."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(1.5)
        assert n.node._last_cmd_pct == pytest.approx(1.0)
        assert n.gripper.moves == [255]
        clk.advance(1.0)
        n.command(-0.3)
        assert n.node._last_cmd_pct == pytest.approx(0.0)
        assert n.gripper.moves == [255, 0]


def test_a_big_jump_still_bypasses_the_rate_limit():
    """An emergency full-open/full-close must not wait for the window."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(0.0)
        clk.advance(0.005)                   # deep inside the closed window
        n.command(1.0)
        assert n.gripper.moves == [_pos(0.0), _pos(1.0)], "big jump went straight out"
        assert n.node._pending_pct is None


def test_command_rate_hz_zero_restores_plain_pass_through():
    """Setting the new machinery's rate to 0 must reproduce the un-throttled
    behaviour exactly: no timer, every sample written as it arrives."""
    clk = _Clock()
    with _Node(clock=clk, command_rate_hz=0.0) as n:
        assert n.node._cmd_min_period == 0.0
        assert n.node._cmd_flush_timer is None, "no timer => no traffic it could add"
        for value in (0.0, 0.2, 0.4, 0.6, 0.8):
            n.command(value)                 # no clock advance at all
        assert n.gripper.moves == [_pos(v) for v in (0.0, 0.2, 0.4, 0.6, 0.8)]
        assert n.node._pending_pct is None


# --------------------------------------------------------------------------- #
# Failure paths: the cache may only ever reflect a SUCCESSFUL write             #
# --------------------------------------------------------------------------- #
def test_a_failed_write_leaves_the_command_cache_untouched():
    """If the write raised, the gripper is NOT at that percent — recording it
    would let the deadband swallow the very sample that would fix things.
    ``_drop_connection`` is stubbed here so this asserts the write path alone."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(0.40)
        last_time = n.node._last_cmd_time
        dropped = []
        n.node._drop_connection = lambda: dropped.append(True)

        clk.advance(1.0)
        n.gripper.fail_next = 1
        n.command(0.80)

        assert n.gripper.attempts[-1] == _pos(0.80), "it was attempted"
        assert n.gripper.moves == [_pos(0.40)], "and it did not land"
        assert n.node._last_cmd_pct == pytest.approx(0.40), "cache unchanged"
        assert n.node._last_cmd_time == pytest.approx(last_time)
        assert dropped == [True], "an I/O error still drops the connection"
        assert n.node._pending_pct == pytest.approx(0.80), \
            "the desire is still OWED -- discharging it on a failed write is the " \
            "parked-short bug all over again"


def test_a_setpoint_lost_to_a_bus_error_is_re_asserted_after_the_reconnect():
    """The measured hardware defect, reached through the I/O-error path instead of
    the rate window: the last sample of a release (full open) is the one whose
    write fails. The bridge is silent at rest, so if the driver forgets it here
    nothing ever re-asserts and the gripper parks 20% closed."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.node._spawn_reconnect = lambda: None
        n.command(0.20)
        clk.advance(1.0)
        n.gripper.fail_next = 1
        n.command(0.0)                       # full open; the write raises
        assert n.gripper.moves == [_pos(0.20)], "parked 20% closed for now"
        assert n.node._connected is False, "the I/O error dropped the connection"

        for _ in range(40):                  # 1 s of reconnect, timer still ticking
            clk.advance(0.025)
            n.tick()
        assert n.node._pending_pct == pytest.approx(0.0), "still owed"

        fresh = n.attach(_FakeGripper())     # bus back
        clk.advance(0.025)
        n.tick()
        assert fresh.moves == [_pos(0.0)], "full open finally reached the gripper"


def test_a_disconnected_bus_issues_no_io_but_keeps_the_setpoint():
    """No connection => no bus traffic whatsoever (there is no socket to storm),
    but the desire is still owed. It cannot go stale: the subscription keeps
    overwriting it with the leader's live value throughout the outage."""
    clk = _Clock()
    with _Node(clock=clk, connected=False) as n:
        n.command(0.70)
        assert n.gripper.attempts == [], "not one byte on the bus"
        assert n.node._pending_pct == pytest.approx(0.70)
        for _ in range(10):
            clk.advance(0.025)
            n.tick()
        assert n.gripper.attempts == [], "and still none after 10 flush ticks"

        n.command(0.30)                      # the leader moved during the outage
        assert n.node._pending_pct == pytest.approx(0.30), "freshest, not stalest"

        n.node._connected = True             # robot powered on
        clk.advance(1.0)
        n.tick()
        assert n.gripper.moves == [_pos(0.30)]


# --------------------------------------------------------------------------- #
# FIX B — a reconnect invalidates the command cache                             #
# --------------------------------------------------------------------------- #
def test_dropping_the_connection_invalidates_the_cache_but_keeps_the_desire():
    """``_connect_loop`` re-activates, and activation runs the auto-calibration
    sweep that leaves the fingers wherever it ends. A surviving ``_last_cmd_pct``
    would then be a lie about where the gripper is."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.node._spawn_reconnect = lambda: None   # no background socket work
        n.command(0.42)
        assert n.node._last_cmd_pct == pytest.approx(0.42)
        n.node._pending_pct = 0.7                # a desire caught mid-window

        n.node._drop_connection()

        assert n.node._last_cmd_pct is None, "cache invalidated across a reconnect"
        assert n.node._pending_pct == pytest.approx(0.7), "the desire survives"
        assert n.node._connected is False
        assert n.node._g is None
        assert n.gripper.closed is True


def test_the_same_value_is_re_asserted_after_a_reconnect():
    """The payoff of FIX B: a steady stream of one value would otherwise never get
    past the deadband, so the gripper would sit wherever auto-cal left it."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.node._spawn_reconnect = lambda: None
        n.command(0.42)
        n.node._drop_connection()

        fresh = n.attach(_FakeGripper())          # reconnected + re-activated
        clk.advance(1.0)
        n.command(0.42)                            # unchanged leader value
        assert fresh.moves == [_pos(0.42)], "re-asserted rather than deadbanded away"


def test_a_desire_pending_across_a_reconnect_is_written_once_the_bus_returns():
    """Drives the REAL sequence, not a convenient one: the flush timer keeps firing
    at 40 Hz for the whole reconnect, which on hardware is >= 0.2 s of socket setup
    plus a 1-3 s activation sweep. A version of this test that re-attaches the
    gripper BEFORE ticking passes even when every one of those ticks throws the
    desire away, which is exactly what the first cut of this fix did."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.node._spawn_reconnect = lambda: None
        n.command(0.10)
        clk.advance(0.010)
        n.command(0.45)                            # deferred by the rate limit
        n.node._drop_connection()
        assert n.node._pending_pct == pytest.approx(0.45)

        for _ in range(120):                       # 3 s of reconnect
            clk.advance(0.025)
            n.tick()
        assert n.node._pending_pct == pytest.approx(0.45), \
            "the desire must survive the whole outage, not just the first 25 ms"

        fresh = n.attach(_FakeGripper())
        clk.advance(0.025)
        n.tick()
        assert fresh.moves == [_pos(0.45)]


def test_the_first_connect_also_invalidates_the_command_cache():
    """FIX B has TWO sites and ``_drop_connection`` is only one of them. The other
    is the commit in ``_connect_loop``, which covers every connect that did not go
    through a drop — including the one right after ``activate()``'s auto-calibration
    sweep has physically moved the fingers."""
    with _Node() as n:
        n.node._last_cmd_pct = 0.42                # stale: from before the outage
        n.node._connected = False
        n.node._g = None

        activated = []

        class _FakeDriver:
            def __init__(self, **_kw):
                self._active = False

            def connect(self):
                return None

            def activate(self):
                activated.append(True)             # the auto-cal sweep
                self._active = True

            def read_status(self):
                return {"gACT": int(self._active), "gSTA": 3 if self._active else 0}

            def close(self):
                return None

        real, modbus_mod.Robotiq2F85 = modbus_mod.Robotiq2F85, _FakeDriver
        try:
            n.node._connect_loop()
        finally:
            modbus_mod.Robotiq2F85 = real

        assert activated == [True], "a fresh gripper gets the auto-cal sweep"
        assert n.node._connected is True
        assert n.node._last_cmd_pct is None, \
            "the sweep moved the fingers -- the cached command is now a lie"


# --------------------------------------------------------------------------- #
# Action / set_closed must not be undone by a stale pending setpoint            #
# --------------------------------------------------------------------------- #
def test_set_closed_discards_a_setpoint_still_waiting_on_its_window():
    """``set_closed`` and the streaming path share ``_cmd_cb``, so a sample that was
    deferred just before the service call is still sitting in the pending slot. It
    is stale intent now: flushing it afterwards would undo the commanded move."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(0.0)
        clk.advance(0.010)
        n.command(0.20)                            # deferred
        assert n.node._pending_pct == pytest.approx(0.20)

        resp = n.set_closed(True)
        assert resp.success is True
        assert n.node._pending_pct is None
        assert n.node._last_cmd_pct == pytest.approx(1.0)

        for _ in range(10):
            clk.advance(0.025)
            n.tick()
        assert n.gripper.moves == [_pos(0.0), 255], "the close was not undone"


def test_an_action_goal_discards_a_setpoint_still_waiting_on_its_window():
    """Same for the GripperCommand action: a stale 20%-closed sample must not flush
    on top of the full open the action just performed."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(0.25)
        clk.advance(0.010)
        n.command(0.20)                            # deferred
        assert n.node._pending_pct == pytest.approx(0.20)

        goal = _Goal(position=0.085)                # metres gap: fully OPEN
        result = n.node._execute(goal)
        assert goal.outcome == "succeed"
        assert result.reached_goal is True
        assert n.gripper.moves[-1] == 0
        assert n.node._pending_pct is None
        assert n.node._last_cmd_pct == pytest.approx(0.0)

        for _ in range(10):
            clk.advance(0.025)
            n.tick()
        assert n.gripper.moves[-1] == 0, "the open was not undone"


# --------------------------------------------------------------------------- #
# Bus ownership: the callback group is NOT enough on its own                    #
# --------------------------------------------------------------------------- #
# rclpy runs an action's execute_callback as a bare executor task, NOT as a member
# of the action server's callback group: ``ActionServer.notify_execute`` calls
# ``self._node.executor.create_task(...)`` and ``Executor.create_task`` appends
# ``(task, None, None)`` -- no entity, so no can_execute()/beginning_execution()
# gate ever runs for it. Measured on Humble with a MultiThreadedExecutor: 60 group
# callbacks (timer + subscription) fired inside one 1 s execute_callback. So
# "they share a MutuallyExclusiveCallbackGroup" does NOT keep an action goal and a
# streaming flush off the single-client :54321 bus -- ``_bus_cmd_lock`` does.

def test_a_flush_never_writes_while_a_command_owns_the_bus():
    """With the bus owned (an action goal is mid-flight on another thread), the
    flush must issue nothing AND must not discharge the desire."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        n.command(0.20)
        clk.advance(1.0)
        n.command(0.90)                            # window is open: would write now
        assert n.gripper.moves == [_pos(0.20), _pos(0.90)]

        clk.advance(1.0)
        n.command(0.10)
        n.node._pending_pct = 0.10                 # re-arm: a fresh desire is owed
        assert n.node._bus_cmd_lock.acquire(blocking=False)
        try:
            before = list(n.gripper.attempts)
            for _ in range(20):
                clk.advance(0.025)
                n.tick()
            assert n.gripper.attempts == before, "not one write while the bus is owned"
            assert n.node._pending_pct == pytest.approx(0.10), "and still owed"
        finally:
            n.node._bus_cmd_lock.release()

        clk.advance(0.025)
        n.tick()
        assert n.gripper.moves[-1] == _pos(0.10), "flushed as soon as the bus is free"


def test_the_action_and_set_closed_hold_the_bus_lock_while_they_command():
    """If they did not, a streaming flush would land between an action's move() and
    its status polls -- the gripper chases the leader, reached_goal never trips, and
    _last_cmd_pct ends up describing a position the fingers are not at."""
    with _Node() as n:
        held = []
        real_move = n.gripper.move

        def spy(pos, speed=150, force=50):
            held.append(n.node._bus_cmd_lock.locked())
            return real_move(pos, speed, force)

        n.gripper.move = spy
        n.node._execute(_Goal(position=0.0))
        n.set_closed(False)
        n.command(0.5)                             # the streaming path too
        assert held == [True, True, True], \
            "every command path must own the bus while it writes"
        assert n.node._bus_cmd_lock.locked() is False, "and release it afterwards"


def test_streaming_resumes_normally_after_an_action():
    """The deliberate ``_last_cmd_pct`` poke in ``_execute`` still does its job: a
    later stream sample is judged against where the action left the gripper."""
    clk = _Clock()
    with _Node(clock=clk) as n:
        goal = _Goal(position=0.0)                 # metres gap: fully CLOSED
        n.node._execute(goal)
        assert n.node._last_cmd_pct == pytest.approx(1.0)

        clk.advance(1.0)
        n.command(0.995)                           # within the deadband of 1.0
        assert n.gripper.moves == [255], "no redundant write"
        clk.advance(1.0)
        n.command(0.0)                             # a real change: straight out
        assert n.gripper.moves == [255, 0]
