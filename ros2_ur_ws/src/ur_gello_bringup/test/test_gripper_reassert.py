#!/usr/bin/env python3
"""Tests for the gripper bridge's DELIVERY ROBUSTNESS layer (FIX C).

The defect, measured on real hardware 2026-08-06 (evidence:
``ros2_ur_ws/gello_logs/diag_gripper_halfopen_20260806_173307/``): the bridge
publishes each setpoint EXACTLY ONCE, when it changes by more than ``deadband``.
The Robotiq driver's command rate limiter used to DISCARD any sample arriving
inside its 1/command_rate_hz window, so the LAST sample of a fast trigger
release — the one that says "fully open" — was lost roughly half the time. The
gripper parked at 0.043 / 0.055 / 0.11 / 0.42 instead of 0.0118 in **7 of 14**
open cycles. Re-asserting the final setpoint for 1 s made it **0 of 6**. Nobody
noticed in code, because the deadband gate compares against ``_last_pub`` — what
we MEANT to send — so the bridge's own view stayed self-consistent forever.

Three behaviours are covered here:

1. **Settle re-assert** — republish the same value for ``settle_reassert_s``
   after the output last changed, then go silent. Silence at rest is as
   load-bearing as the re-assert: the Modbus link is single-client.
2. **Resume force-publish** — ``_on_resume`` writes ``_last_pub = seed`` without
   publishing it, so a leader already resting within ``deadband`` of the seed was
   suppressed forever and the gripper received nothing at all.
3. **Reconcile** — re-assert the last command when the MEASURED position sits
   more CLOSED than it. The direction is one-sided on purpose; the test named
   for the grasp is the one that encodes why.

Conventions follow ``test_gripper_discrete.py``: build the real node, call its
callbacks directly, no executor and no spinning, and remap the node name and
every topic onto throwaway names so this suite is safe to run beside a LIVE
teleop stack. The harness is duplicated rather than imported because there is no
conftest.py in this directory and each test file here stands alone.
"""

import pytest

rclpy = pytest.importorskip("rclpy", reason="node-level tests need a ROS 2 env")

from ur_gello_bringup import gello_gripper_bridge_node as bridge_mod  # noqa: E402
from ur_gello_bringup.gello_gripper_bridge_node import (  # noqa: E402
    DISCRETE_CLOSED,
    DISCRETE_OPEN,
    GelloGripperBridge,
)
from std_msgs.msg import Float32  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

# The measured numbers from the 2026-08-06 session, used as literals below so a
# reader can match the tests against the bug report.
FULL_OPEN_CMD = 0.0118  # 3/255 — what a released trigger commands
PARKED_SHORT = 0.11  # where the gripper actually stopped when the message was lost
GRASP_POS = 0.506  # grip_pos maxes here while holding an object (§5.1)


class _Clock:
    """Deterministic stand-in for the bridge module's ``time``.

    The node only ever asks for ``time.monotonic()``. Real wall clock would make
    every dwell/window assertion below machine-load dependent.
    """

    def __init__(self, t0=1000.0):
        self.t = float(t0)

    def monotonic(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)
        return self.t


class _CapturePub:
    def __init__(self, sink):
        self._sink = sink

    def publish(self, msg):
        self._sink.append(msg.data)


class _Bridge:
    """Context manager: a real GelloGripperBridge with captured publishes."""

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

    def tick(self):
        self.node._settle_tick()

    def pause(self):
        return self.node._on_pause(None, Trigger.Response())

    def resume(self):
        return self.node._on_resume(None, Trigger.Response())


# --------------------------------------------------------------------------- #
# 1. SETTLE RE-ASSERT                                                          #
# --------------------------------------------------------------------------- #
def test_settle_reassert_repeats_the_value_then_goes_silent():
    """The whole feature in one test: after the output changes, the same value is
    re-published for settle_reassert_s — and then the node goes QUIET.

    The leader keeps streaming the whole time (it runs at ~30 Hz and the deadband
    suppresses every one of those samples), which is exactly the at-rest
    condition. Once the window closes the bridge must publish NOTHING: the Modbus
    bus at :54321 is single-client.
    """
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=1.0, settle_reassert_hz=10.0) as b:
        b.width(0.5)
        assert b.published == [pytest.approx(0.5)], "the change itself"
        # 5 ticks inside the 1.0 s window (t + 0.1 .. t + 0.5).
        for _ in range(5):
            clk.advance(0.1)
            b.width(0.5)  # deadbanded away
            b.tick()
        assert b.published == [pytest.approx(0.5)] * 6

        # Past t + 1.0 the window has closed, and it stays closed no matter how
        # long the leader keeps streaming the same rest value.
        clk.advance(0.6)
        for _ in range(5):
            clk.advance(0.1)
            b.width(0.5)
            b.tick()
        assert len(b.published) == 6, "at rest the bridge publishes nothing"


def test_a_new_output_value_reopens_the_settle_window():
    """The window is anchored to the last CHANGE, not the last publish — so it
    tracks the operator's gesture and re-arms on every fresh intent."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=1.0, settle_reassert_hz=10.0) as b:
        b.width(0.5)
        clk.advance(1.5)  # window long closed
        b.width(0.5)
        b.tick()
        assert b.published == [pytest.approx(0.5)], "closed: nothing re-asserted"
        # A real move re-opens it.
        b.width(0.9)
        assert b.published[-1] == pytest.approx(0.9)
        clk.advance(0.1)
        b.width(0.9)
        b.tick()
        assert b.published[-1] == pytest.approx(0.9)
        assert len(b.published) == 3


def test_settle_reassert_never_publishes_while_paused():
    """PAUSED means the output is deliberately silenced (a collapsed passive
    leader is a drop hazard); the re-assert must not smuggle traffic past it."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=1.0, settle_reassert_hz=10.0) as b:
        b.width(0.5)
        b.published.clear()
        b.pause()
        for _ in range(5):
            clk.advance(0.1)
            b.width(0.5)
            b.tick()
        assert b.published == []


def test_settle_reassert_stops_when_the_leader_goes_stale():
    """A dead leader stream is not an intent to re-assert. Same staleness gate
    ~/state uses to report WAITING and ~/resume uses to refuse."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=1.0, settle_reassert_hz=10.0) as b:
        b.width(0.5)
        b.published.clear()
        clk.advance(0.2)  # still fresh (staleness_timeout_s default 0.5)
        b.tick()
        assert b.published == [pytest.approx(0.5)]
        clk.advance(0.4)  # leader now 0.6 s old, window still open (0.6 < 1.0)
        b.tick()
        assert b.published == [pytest.approx(0.5)], "stale leader: no re-assert"


def test_settle_reassert_publishes_nothing_before_the_first_output():
    """No output has been committed yet, so there is nothing to re-assert and no
    honest value to invent."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=1.0, settle_reassert_hz=10.0) as b:
        for _ in range(5):
            clk.advance(0.1)
            b.tick()
        assert b.published == []


def test_settle_tick_survives_a_seeded_last_pub_with_no_output_change_yet():
    """``_last_pub`` set but ``_output_changed_time`` still None is REACHABLE, and
    without the guard the window arithmetic is ``now - None``.

    ``_on_resume`` writes the seed straight into ``_last_pub``; with
    ``start_paused`` nothing has ever been published, so the settle timer fires
    into a state no publish ever created. Both halves of that guard are
    load-bearing — dropping either one turns the next timer tick into a
    TypeError inside a callback, i.e. a dead gripper bridge.
    """
    clk = _Clock()
    with _Bridge(
        clock=clk, start_paused=True, settle_reassert_s=1.0, settle_reassert_hz=10.0
    ) as b:
        b.width(0.5)
        b.actual(0.5)
        clk.advance(0.1)
        b.width(0.5)
        assert b.resume().success is True
        assert b.node._last_pub is not None, "seeded"
        assert b.node._output_changed_time is None, "but nothing was ever published"
        clk.advance(0.05)
        b.tick()  # must not raise
        assert b.published == []


def test_settle_reassert_runs_in_discrete_mode_too():
    """Discrete publishes even LESS often than continuous — one message per latch
    transition — so a lost message there strands the gripper for a whole grasp
    cycle. It must NOT be special-cased off."""
    clk = _Clock()
    with _Bridge(
        clock=clk, discrete_mode=True, settle_reassert_s=1.0, settle_reassert_hz=10.0
    ) as b:
        b.width(0.8)  # latches CLOSED
        assert b.published == [DISCRETE_CLOSED]
        for _ in range(3):
            clk.advance(0.1)
            b.width(0.8)
            b.tick()
        assert b.published == [DISCRETE_CLOSED] * 4
        # The release is the sample that actually got lost on hardware.
        b.width(0.229)  # documented resting outlier -> full OPEN
        assert b.published[-1] == DISCRETE_OPEN
        clk.advance(0.1)
        b.width(0.229)
        b.tick()
        assert b.published[-1] == DISCRETE_OPEN
        assert len(b.published) == 6


def test_a_settle_reassert_disturbs_neither_the_deadband_nor_the_ramp_budget():
    """A re-assert republishes ``_last_pub`` and updates NOTHING. ``_last_pub_time``
    is the resume ramp's slew-budget base and ``_output_changed_time`` is the
    settle anchor; advancing either from a duplicate message would slow the ramp
    and make the window slide forever."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=1.0, settle_reassert_hz=10.0) as b:
        b.width(0.5)
        pub_time = b.node._last_pub_time
        changed = b.node._output_changed_time
        last_pub = b.node._last_pub
        clk.advance(0.1)
        b.tick()
        assert b.published[-1] == pytest.approx(0.5)
        assert b.node._last_pub_time == pub_time
        assert b.node._output_changed_time == changed
        assert b.node._last_pub == last_pub


def test_a_settle_reassert_carries_last_pub_not_the_live_leader_value():
    """The re-assert republishes ``_last_pub`` VERBATIM and never the live
    filtered leader value.

    The two are equal whenever the leader holds perfectly still, which is why
    every other test here cannot tell them apart. Move the leader by LESS than
    ``deadband``: the pipeline deliberately suppresses that sample, so ``_f``
    and ``_last_pub`` diverge. A re-assert that carried ``_f`` would put a value
    on the wire that the deadband had just refused — a NEW setpoint, defeating
    the gate and re-introducing exactly the at-rest jitter traffic the deadband
    exists to keep off a single-client bus.
    """
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=1.0, settle_reassert_hz=10.0) as b:
        b.width(0.500)
        b.published.clear()
        clk.advance(0.05)
        b.width(0.515)  # 0.015 < deadband 0.02
        assert b.published == [], "sub-deadband sample suppressed"
        assert b.node._f == pytest.approx(0.515), "but the filter moved"
        assert b.node._last_pub == pytest.approx(0.500), "and the output did not"
        clk.advance(0.05)
        b.tick()
        assert b.published == [pytest.approx(0.500)], "re-asserted _last_pub, not _f"


def test_a_converged_ramp_tail_does_not_extend_the_settle_window():
    """``_publish_output`` re-opens the window only when the number on the wire
    actually CHANGES. The tail of a resume ramp republishes the same converged
    value on every leader sample; treating those as changes would hold the
    window open for the whole ramp and then a further settle_reassert_s.
    """
    clk = _Clock()
    with _Bridge(
        clock=clk, settle_reassert_s=1.0, settle_reassert_hz=10.0, resume_ramp_s=5.0
    ) as b:
        b.width(0.2)
        b.actual(0.0)  # gripper open, leader asking for 0.2
        b.pause()
        clk.advance(0.05)
        b.width(0.2)
        assert b.resume().success is True
        # Crawl the ramp until it converges on the leader value.
        for _ in range(12):
            clk.advance(0.05)
            b.width(0.2)
        assert b.published[-1] == pytest.approx(0.2), "ramp converged"
        changed = b.node._output_changed_time
        assert changed is not None
        # Every remaining ramp sample republishes that same converged value.
        for _ in range(30):
            clk.advance(0.05)
            b.width(0.2)
        assert b.published[-1] == pytest.approx(0.2)
        assert b.node._output_changed_time == changed, (
            "an unchanged republish must not move the settle anchor"
        )


def test_settle_reassert_s_zero_creates_no_timer_and_publishes_once():
    """0 restores exactly today's behaviour, including the wakeup profile: with
    the state timer off too, the node owns no timers at all."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        assert b.node._settle_enabled is False
        assert list(b.node.timers) == []
        b.width(0.5)
        for _ in range(10):
            clk.advance(0.1)
            b.width(0.5)
            b.tick()
        assert b.published == [pytest.approx(0.5)]


def test_settle_reassert_hz_zero_also_disables_it():
    """A zero rate has no representable period; treat it as off rather than
    dividing by zero at construction."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=1.0, settle_reassert_hz=0.0) as b:
        assert b.node._settle_enabled is False
        assert list(b.node.timers) == []
        b.width(0.5)
        clk.advance(0.1)
        b.tick()
        assert b.published == [pytest.approx(0.5)]


# --------------------------------------------------------------------------- #
# 2. RESUME FORCE-PUBLISH (the phantom seed)                                   #
# --------------------------------------------------------------------------- #
def test_resume_force_publishes_through_the_phantom_seed():
    """``_on_resume`` sets ``_last_pub = seed`` WITHOUT publishing it. That is a
    phantom: the deadband gate believes the seed reached the gripper. A leader
    already resting within ``deadband`` of the seed was therefore suppressed on
    every subsequent sample and the gripper received NOTHING, forever, while
    ~/state cheerfully reported FOLLOWING.

    Shown with ``resume_ramp_s=0.0`` because the ramp branch publishes
    unconditionally and so hid this at the default config.
    """
    clk = _Clock()
    with _Bridge(clock=clk, resume_ramp_s=0.0, settle_reassert_s=0.0) as b:
        b.width(0.42)
        b.pause()
        b.actual(0.43)  # gripper reports ~where the leader already is
        clk.advance(0.1)
        b.width(0.42)  # fresh leader sample so resume's staleness gate passes
        b.published.clear()
        assert b.resume().success is True
        assert b.node._last_pub == pytest.approx(0.43), "the phantom seed"
        assert abs(0.42 - 0.43) < b.node.deadband, "and it is inside the deadband"

        clk.advance(0.033)
        b.width(0.42)
        assert b.published == [pytest.approx(0.42)], "forced past the deadband"
        # ... and then the deadband is back in charge: no steady-state traffic.
        for _ in range(5):
            clk.advance(0.033)
            b.width(0.42)
        assert b.published == [pytest.approx(0.42)]


def test_a_refused_resume_does_not_arm_the_force_publish():
    """A refused resume must change NOTHING — it stays paused and silent."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        b.width(0.5)
        b.actual(0.5)
        b.pause()
        clk.advance(5.0)  # leader now stale -> gate (a) refuses
        b.published.clear()
        assert b.resume().success is False
        assert b.node._force_next_publish is False
        assert b.node._paused is True
        assert b.published == []


def test_an_accepted_resume_closes_the_settle_window():
    """A settle window opened BEFORE a pause must not survive the resume.

    ``_on_resume`` writes ``_last_pub = seed`` — a PHANTOM taken from
    ``_actual_pos``, never published, and explicitly allowed to be STALE (see its
    docstring). ``_output_changed_time`` used to be untouched by pause and by
    resume, so the pre-pause window was still open and ``_settle_tick`` re-asserted
    that phantom.

    THE HAZARD, reproduced offline: discrete mode; the driver's position stream
    drops so ``_actual_pos`` goes stale at 0.9; GO HOME pauses this bridge and
    force-opens the Robotiq to 0.0 behind its back; the operator resumes with the
    trigger resting inside the hysteresis band. The latch is UNKNOWN so
    ``_on_width`` publishes nothing — which is what ``_on_resume`` promises the
    operator in its own response message — and the settle timer nevertheless
    published the stale 0.9 twelve times, CLOSING the freshly-opened gripper to
    90%. That is a closing command the live trigger never asked for: the exact
    thing ``_maybe_reconcile``'s one-sided rule exists to make impossible.
    """
    clk = _Clock()
    with _Bridge(
        clock=clk,
        discrete_mode=True,
        settle_reassert_s=1.0,
        settle_reassert_hz=10.0,
    ) as b:
        b.width(0.9)  # latches CLOSED -> opens a settle window
        assert b.published == [DISCRETE_CLOSED]
        b.actual(0.9)  # last position sample before the stream dies
        clk.advance(0.1)
        b.width(0.9)
        b.pause()
        # GO HOME force-opens the Robotiq to 0.0 while we are paused and silent;
        # no position feedback arrives, so _actual_pos stays at the stale 0.9.
        clk.advance(0.2)
        b.width(0.5)  # trigger now rests INSIDE the hysteresis band
        assert b.resume().success is True
        assert b.node._last_pub == pytest.approx(0.9), "the stale phantom seed"
        b.published.clear()

        for _ in range(12):  # 0.6 s, still inside the pre-pause 1.0 s window
            clk.advance(0.05)
            b.width(0.5)  # in-band: latch UNKNOWN, publishes nothing
            b.tick()
        assert b.published == [], (
            "an accepted resume must not leave a settle window that re-asserts "
            "the phantom seed"
        )


def test_a_resume_also_clears_a_reconcile_dwell_from_before_the_pause():
    """The dwell must measure ONE continuous episode. A discrepancy that accrued
    against the pre-pause command says nothing about the freshly seeded one."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0, resume_ramp_s=0.0) as b:
        b.width(FULL_OPEN_CMD)
        clk.advance(0.2)
        b.actual(PARKED_SHORT)  # discrepancy starts accruing
        assert b.node._reconcile_since is not None
        b.pause()
        clk.advance(5.0)  # long pause; no position samples arrive to clear it
        b.width(FULL_OPEN_CMD)
        assert b.resume().success is True
        assert b.node._reconcile_since is None, "nothing is owed from before the pause"


def test_the_forced_publish_is_not_eaten_by_an_unknown_discrete_latch():
    """The force flag defeats the DEADBAND, never the latch. An accepted resume
    resets the latch to UNKNOWN, and UNKNOWN must still publish nothing (holding
    is the honest answer to "no intent yet") — so the debt stays pending until a
    threshold crossing gives it a value to carry."""
    clk = _Clock()
    with _Bridge(
        clock=clk, discrete_mode=True, resume_ramp_s=0.0, settle_reassert_s=0.0
    ) as b:
        b.width(0.8)  # take ends grasped: latch CLOSED
        b.pause()
        b.actual(0.0)  # GO HOME force-opened the Robotiq behind our back
        clk.advance(0.1)
        b.width(0.5)
        assert b.resume().success is True
        b.published.clear()

        clk.advance(0.033)
        b.width(0.5)  # in-band: latch UNKNOWN
        assert b.published == [], "UNKNOWN still publishes nothing"
        assert b.node._force_next_publish is True, "the debt survives"

        clk.advance(0.033)
        b.width(0.2)  # latches OPEN == the seed -> deadband would suppress it
        assert b.node._discrete_latch == DISCRETE_OPEN
        assert b.published == [pytest.approx(DISCRETE_OPEN)]


# --------------------------------------------------------------------------- #
# 3. RECONCILE AGAINST THE MEASURED POSITION                                   #
# --------------------------------------------------------------------------- #
def _lost_open_command(b, clk, samples, actual=PARKED_SHORT):
    """Replay the measured failure: a full-open command is published, the driver
    drops it, and the gripper sits at ``actual`` while the leader keeps streaming
    the released trigger. Returns with ``b.published`` cleared of the command
    itself, so what remains is exactly the re-asserts.
    """
    b.width(FULL_OPEN_CMD)
    b.published.clear()
    for _ in range(samples):
        clk.advance(0.2)  # the driver's status stream runs at 5 Hz
        b.width(FULL_OPEN_CMD)  # deadbanded away
        b.actual(actual)


def test_reconcile_reasserts_a_command_the_gripper_never_reached():
    """THE measured defect: commanded 0.0118, parked at 0.11, and nothing in the
    system ever said so again. After reconcile_after_s of continuous discrepancy
    the bridge re-asserts its own last command — once."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        _lost_open_command(b, clk, samples=3)
        assert b.published == [], "dwell not met yet (0.4 s < 0.5 s)"
        clk.advance(0.2)
        b.width(FULL_OPEN_CMD)
        b.actual(PARKED_SHORT)
        assert b.published == [pytest.approx(FULL_OPEN_CMD)]


def test_reconcile_is_rate_limited_to_one_reassert_per_window():
    """A discrepancy that persists must not turn into a stream. Firing RESTARTS
    the dwell clock, so re-asserts are at least reconcile_after_s apart."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        # 12 samples at 0.2 s = 2.4 s of continuous discrepancy. The dwell starts
        # on sample 1 and fires on samples 4, 7 and 10 (0.6 s apart, the first
        # 0.2 s grid point at or past each 0.5 s deadline).
        _lost_open_command(b, clk, samples=12)
        assert b.published == [pytest.approx(FULL_OPEN_CMD)] * 3
        assert len(b.published) <= int(2.4 / 0.5) + 1, "bounded by the rate limit"


def test_reconcile_never_fights_a_grasp():
    """THE safety test. A 2F-85 holding an object stops where the object is —
    grip_pos maxes out near 0.506 while gripping (UNITS_REFERENCE §5.1) — so a
    successful grasp looks EXACTLY like a lost close command from position alone.

    Reconcile therefore acts in one direction only: it fires when the fingers are
    MORE CLOSED than commanded (re-asserting can then only OPEN them) and is
    silent when they are more open. A grasp lands in the silent half, so no
    amount of dwell can make the bridge squeeze harder.
    """
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        b.width(1.0)  # operator squeezes: command fully CLOSED
        assert b.published == [pytest.approx(1.0)]
        b.published.clear()
        for _ in range(25):  # 5 s, ten times the dwell
            clk.advance(0.2)
            b.width(1.0)
            b.actual(GRASP_POS)  # fingers stopped on the object
        assert b.published == [], "a grasp is never re-commanded"
        assert b.node._reconcile_since is None, "the dwell never even starts"

    # Same for a PARTIAL close that meets the object early.
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        b.width(0.6)
        b.published.clear()
        for _ in range(25):
            clk.advance(0.2)
            b.width(0.6)
            b.actual(GRASP_POS)
        assert b.published == []


def test_reconcile_ignores_differences_within_the_tolerance():
    """reconcile_tol keeps encoder quantisation and the driver's own 0.01 command
    deadband from looking like a lost message."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        b.width(0.10)
        b.published.clear()
        for _ in range(10):
            clk.advance(0.2)
            b.width(0.10)
            b.actual(0.13)  # 0.03 more closed — under the 0.05 default
        assert b.published == []


def test_the_reconcile_dwell_must_be_continuous():
    """One in-tolerance sample clears the clock, so an intermittent mismatch can
    never accumulate its way to a re-assert."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        b.width(FULL_OPEN_CMD)
        b.published.clear()
        for value in (PARKED_SHORT, PARKED_SHORT, FULL_OPEN_CMD, PARKED_SHORT, PARKED_SHORT):
            clk.advance(0.2)
            b.width(FULL_OPEN_CMD)
            b.actual(value)
        assert b.published == [], "1.0 s elapsed but never 0.5 s continuously"


def test_reconcile_is_silent_while_paused():
    """PAUSED is the safe state: not streaming. Reconcile is streaming."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        b.width(FULL_OPEN_CMD)
        b.pause()
        b.published.clear()
        for _ in range(10):
            clk.advance(0.2)
            b.width(FULL_OPEN_CMD)
            b.actual(PARKED_SHORT)
        assert b.published == []
        assert b.node._reconcile_since is None


def test_reconcile_holds_off_during_the_resume_ramp():
    """Inside the ramp ``_last_pub`` is an intermediate slew sample the gripper is
    still chasing, so a mismatch is EXPECTED rather than evidence of loss."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0, resume_ramp_s=10.0) as b:
        b.width(0.5)
        b.pause()
        b.actual(0.5)
        clk.advance(0.1)
        b.width(0.5)
        assert b.resume().success is True
        b.published.clear()
        for _ in range(8):  # 1.6 s, far past reconcile_after_s
            clk.advance(0.2)
            b.width(0.5)  # the ramp republishes the converged value
            b.actual(1.0)  # ... while the gripper claims fully CLOSED
            assert b.node._reconcile_since is None
        assert b.published == [pytest.approx(0.5)] * 8, "only the ramp published"


def test_reconcile_requires_a_fresh_leader():
    """Re-asserting an intent nobody is currently expressing is a guess, not
    defence in depth."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        b.width(FULL_OPEN_CMD)
        b.published.clear()
        clk.advance(1.0)  # leader dies (staleness_timeout_s default 0.5)
        for _ in range(10):
            clk.advance(0.2)
            b.actual(PARKED_SHORT)
        assert b.published == []


def test_reconcile_ignores_a_stale_measurement():
    """A position that stopped arriving describes where the fingers WERE."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        _lost_open_command(b, clk, samples=4)
        assert len(b.published) == 1, "fired once while the feedback was live"
        b.published.clear()
        # The driver's status stream now dies. Time passes and the leader keeps
        # streaming, but the last measurement ages out, so nothing may be
        # inferred from it — including at t+0.6 s, where the dwell would
        # otherwise have re-fired.
        for _ in range(6):
            clk.advance(0.2)
            b.width(FULL_OPEN_CMD)
            b.node._maybe_reconcile(clk.t)
        assert b.published == []


def test_reconcile_covers_a_third_party_closing_the_gripper():
    """Other publishers share ~/command_percent (task recorder GO HOME, RL reset,
    run_hil_preposition.sh). If one leaves the gripper more CLOSED than the
    operator's live intent, re-asserting that intent OPENS it — always the safe
    direction, and never a value the operator did not themselves command."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        b.width(0.0)  # operator wants it open
        b.published.clear()
        for _ in range(4):
            clk.advance(0.2)
            b.width(0.0)
            b.actual(1.0)  # something slammed it shut
        assert b.published == [pytest.approx(0.0)], "re-assert OPEN, once"


def test_reconcile_survives_a_fresh_unpaused_bridge_that_has_published_nothing():
    """``_last_pub is None`` together with a FRESH leader and an unpaused bridge is
    REACHABLE, and without the guard the directional gate is ``actual - None``.

    Discrete mode reaches it on the very first sample: a trigger that starts
    inside the hysteresis band leaves the latch UNKNOWN, so nothing is ever
    published, while the leader stream is perfectly healthy. The driver's 5 Hz
    position feed then calls straight into reconcile. Dropping this guard turns
    the first position sample of every in-band startup into a TypeError.
    """
    clk = _Clock()
    with _Bridge(clock=clk, discrete_mode=True, settle_reassert_s=0.0) as b:
        b.width(0.5)  # in-band on the FIRST sample -> latch UNKNOWN
        assert b.published == []
        assert b.node._last_pub is None
        assert b.node._paused is False
        assert b.node._last_rx_time is not None, "and the leader is fresh"
        b.actual(1.0)  # must not raise
        assert b.published == []


def test_the_reconcile_warning_is_throttled_when_the_gripper_can_never_comply():
    """The re-assert repeats forever on purpose; the LOG must not.

    A discrepancy the gripper cannot close — jammed fingers, or a command that
    was in fact delivered so the driver's own deadband swallows every duplicate —
    is permanent. Unthrottled, this warned 99 times a minute for the rest of the
    session and buried every other line in the operator's console. The re-assert
    itself is unchanged and still bounded by ``reconcile_after_s``.
    """
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0) as b:
        warns = []
        b.node.get_logger = lambda: type(
            "_L", (), {"warn": lambda _s, m: warns.append(m),
                       "info": lambda _s, m: None,
                       "error": lambda _s, m: None}
        )()
        b.width(0.0)
        b.published.clear()
        for _ in range(300):  # 60 s of 5 Hz status, gripper stuck shut
            clk.advance(0.2)
            b.width(0.0)
            b.actual(0.9)
        assert 60 / 0.5 >= len(b.published) >= 60 / 1.0, (
            f"re-asserts stay bounded by the rate limit (got {len(b.published)})"
        )
        assert len(warns) <= 60 / bridge_mod.RECONCILE_WARN_PERIOD_S + 1, (
            f"log throttled (got {len(warns)} lines for {len(b.published)} re-asserts)"
        )
        assert warns, "but the first one of an episode is always logged"


def test_reconcile_after_s_zero_restores_todays_behaviour():
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0, reconcile_after_s=0.0) as b:
        assert b.node._reconcile_enabled is False
        _lost_open_command(b, clk, samples=20)
        assert b.published == []


def test_reconcile_tol_zero_restores_todays_behaviour():
    """A zero tolerance would fire on ANY difference, including raw 1/255
    quantisation — the opposite of off. Treat it as off."""
    clk = _Clock()
    with _Bridge(clock=clk, settle_reassert_s=0.0, reconcile_tol=0.0) as b:
        assert b.node._reconcile_enabled is False
        _lost_open_command(b, clk, samples=20)
        assert b.published == []


# --------------------------------------------------------------------------- #
# Everything off == the old node                                               #
# --------------------------------------------------------------------------- #
def test_all_new_params_at_zero_reproduce_the_old_publish_pattern():
    """The whole point of the escape hatch: publish-once-per-change, no extra
    timers, and total indifference to the measured position."""
    clk = _Clock()
    with _Bridge(
        clock=clk,
        settle_reassert_s=0.0,
        reconcile_after_s=0.0,
        reconcile_tol=0.0,
    ) as b:
        assert b.node._settle_enabled is False
        assert b.node._reconcile_enabled is False
        assert list(b.node.timers) == [], "no wakeups this node did not have before"
        for value in (0.0, 0.5, 1.0):
            b.width(value)
            for _ in range(10):
                clk.advance(0.1)
                b.width(value)
                b.actual(1.0)  # maximally discrepant; must change nothing
                b.tick()
        assert b.published == [
            pytest.approx(0.0),
            pytest.approx(0.5),
            pytest.approx(1.0),
        ]


def test_the_resume_force_publish_has_no_off_switch():
    """SCOPE OF THE ESCAPE HATCH. The test above proves the four new parameters
    at 0 reproduce publish-once-per-change — on the NON-RESUME path, which is the
    only path it exercises. The resume force-publish is the third new behaviour
    and it has NO parameter, so "zero restores exactly today's behaviour" is
    false for it and this test pins that rather than letting the claim drift.

    It is a bug fix, not a feature: without it a leader resting within
    ``deadband`` of the seed is suppressed forever and the gripper receives
    nothing at all while ~/state reports FOLLOWING. The message it adds is one
    publish of the value the operator's own live trigger is asking for, so there
    is no configuration in which restoring the hang would be correct.
    """
    clk = _Clock()
    with _Bridge(
        clock=clk,
        settle_reassert_s=0.0,
        settle_reassert_hz=0.0,
        reconcile_after_s=0.0,
        reconcile_tol=0.0,
        resume_ramp_s=0.0,
    ) as b:
        assert b.node._settle_enabled is False
        assert b.node._reconcile_enabled is False
        b.width(0.42)
        b.pause()
        b.actual(0.43)
        clk.advance(0.1)
        b.width(0.42)
        assert b.resume().success is True
        b.published.clear()
        clk.advance(0.033)
        b.width(0.42)  # |0.42 - 0.43| < deadband: the OLD node published nothing
        assert b.published == [pytest.approx(0.42)], (
            "the force-publish is unconditional -- zeroing the params does not "
            "restore the phantom-seed hang"
        )


def test_defaults_are_the_hardware_validated_configuration():
    """0/6 on hardware was measured with a ~1 s re-assert window. Pin the
    defaults so a yaml edit cannot silently ship the 7/14 configuration."""
    with _Bridge() as b:
        assert b.node.settle_reassert_s == pytest.approx(1.0)
        assert b.node.settle_reassert_hz == pytest.approx(10.0)
        assert b.node.reconcile_tol == pytest.approx(0.05)
        assert b.node.reconcile_after_s == pytest.approx(0.5)
        assert b.node._settle_enabled is True
        assert b.node._reconcile_enabled is True
