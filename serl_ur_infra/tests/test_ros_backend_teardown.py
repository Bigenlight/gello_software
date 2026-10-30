"""URRosBackend.close() teardown ordering.

Regression cover for a defect confirmed on the real UR7e stack: a READ-ONLY
probe that shut down normally still ended with

    terminate called without an active exception
    timeout: the monitored command dumped core

The old close() was two lines::

    self._shutdown = True
    self._node.destroy_node()

which joins neither worker thread and never shuts the context down, so
destroy_node() races a live 250 Hz _upsample_loop and a live spin thread. On
the live graph that aborted ~1 run in 30. A first fix that used
``executor.shutdown()`` to break ``executor.spin()`` out traded the abort for a
SIGSEGV (~1 in 50): ``Executor._wait_for_ready_callbacks`` puts ``self._guard``
into a live rcl wait set while ``Executor.shutdown()`` destroys that same guard
from the calling thread. The spin thread must therefore leave the executor by
itself, be joined, and only then may the executor and node be torn down.

Beyond the crash, the ordering is a safety property: _upsample_loop republishes
the last joint target at 250 Hz, so a worker that outlives the env keeps driving
the robot. DRY_RUN=True hides that today; DRY_RUN=False would not.

These are pure unit tests. rclpy is deliberately NOT imported (the test
PYTHONPATH excludes ROS, so ``ros_backend._ROS_AVAILABLE`` is False here): the
real ``close()`` and the real ``_upsample_loop`` are driven with fakes standing
in for the node, executor and publishers. No ROS node is created, no topic is
subscribed, and nothing is ever published to the robot.
"""

import os
import sys
import threading
import time

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.envs import ros_backend as rb  # noqa: E402
from ur_env.envs.ros_backend import URRosBackend  # noqa: E402

Q6 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])


# --------------------------------------------------------------------------- #
# doubles                                                                      #
# --------------------------------------------------------------------------- #
class _FakeMsg:
    """Stand-in for Float64MultiArray / Float32 (both are just ``.data``)."""

    data = None


class _FakePub:
    """Publisher that records the node's liveness at the END of each publish.

    ``delay`` widens the publish into a window close() can land inside. That is
    what makes the race deterministic instead of ~1%-flaky: with a real rmw the
    publish is a C call holding node-owned memory, and the question is whether
    destroy_node() can run while one is in flight.
    """

    def __init__(self, node, delay=0.0):
        self._node = node
        self.delay = delay
        self.publishes = []  # list of (payload, node_was_alive_at_completion)

    def publish(self, msg):
        if self.delay:
            time.sleep(self.delay)
        self.publishes.append((msg.data, not self._node.destroyed))

    @property
    def after_destroy(self):
        return [p for p, alive in self.publishes if not alive]


class _FakeNode:
    def __init__(self):
        self.destroyed = False
        self.destroy_calls = 0

    def destroy_node(self):
        self.destroy_calls += 1
        self.destroyed = True


class _FakeExecutor:
    """Mimics the parts of SingleThreadedExecutor that matter here.

    ``spin_once`` blocks up to ``timeout_sec`` (like the real one waiting on an
    empty graph) and tracks whether a thread is currently inside the executor,
    so a test can assert shutdown() is never called while one is.
    """

    def __init__(self):
        self.shutdown_calls = 0
        self.shutdown_while_spinning = False
        self._inside = 0
        self._lock = threading.Lock()

    def spin_once(self, timeout_sec=None):
        with self._lock:
            self._inside += 1
        try:
            time.sleep(min(timeout_sec or 0.0, 0.02))
        finally:
            with self._lock:
                self._inside -= 1

    def shutdown(self, timeout_sec=None):
        self.shutdown_calls += 1
        with self._lock:
            if self._inside:
                self.shutdown_while_spinning = True
        return True


class _FakeRclpy:
    def __init__(self, ok=True):
        self._ok = ok
        self.shutdown_calls = 0

    def ok(self):
        return self._ok

    def shutdown(self):
        self.shutdown_calls += 1
        self._ok = False


def _make_backend(
    monkeypatch,
    dry_run=True,
    owns_rclpy=True,
    start_threads=True,
    with_target=False,
    publish_delay=0.0,
):
    """Build a URRosBackend without rclpy and without running __init__.

    __init__ needs a live ROS graph, which these tests must not touch. Every
    attribute close()/_upsample_loop/_spin actually read is set explicitly, so
    the methods under test are the real ones.
    """
    b = object.__new__(URRosBackend)
    b.dry_run = dry_run
    b._lock = threading.Lock()
    b._close_lock = threading.Lock()
    b._shutdown = False
    b._closed = False
    b._node_alive = True
    b._owns_rclpy = owns_rclpy

    b._node = _FakeNode()
    b._executor = _FakeExecutor()
    b._cmd_pub = _FakePub(b._node, delay=publish_delay)
    b._gripper_pub = _FakePub(b._node)
    b._grip_pending = None
    b.grip_reassert_s = URRosBackend.grip_reassert_s
    b.grip_reassert_hz = URRosBackend.grip_reassert_hz

    b._q = (Q6.copy(), time.monotonic())
    b._dq = None
    b._q_stream = None
    b._q_target = Q6 + 0.5 if with_target else None
    b._q_target_time = time.monotonic() if with_target else None
    b._up_hz = 250.0
    b._up_step = 0.002
    b._up_accel = 8.0
    b._up_soft_start_s = 0.7
    b._up_soft_start_fraction = 0.15
    b._target_stale_s = 0.3
    b._up_streamer = rb.AccelerationLimitedJointStream(
        hz=b._up_hz,
        max_step_rad=b._up_step,
        max_accel_rad_s2=b._up_accel,
        soft_start_s=b._up_soft_start_s,
        target_stale_s=b._target_stale_s,
        soft_start_fraction=b._up_soft_start_fraction,
    )

    fake_rclpy = _FakeRclpy()
    monkeypatch.setattr(rb, "rclpy", fake_rclpy, raising=False)
    monkeypatch.setattr(rb, "Float64MultiArray", _FakeMsg, raising=False)
    monkeypatch.setattr(rb, "Float32", _FakeMsg, raising=False)
    b._fake_rclpy = fake_rclpy

    b._up_thread = threading.Thread(target=b._upsample_loop, daemon=True)
    b._spin_thread = threading.Thread(target=b._spin, daemon=True)
    if start_threads:
        b._up_thread.start()
        b._spin_thread.start()
        time.sleep(0.05)  # let both loops get going
    return b


# --------------------------------------------------------------------------- #
# the crash: close() must join before it destroys                              #
# --------------------------------------------------------------------------- #
def test_close_joins_both_worker_threads(monkeypatch):
    """The core fix: close() does not return with workers still running.

    The old close() returned immediately, leaving a 250 Hz command loop and a
    spin thread alive against a node it had just destroyed.
    """
    b = _make_backend(monkeypatch)
    assert b._up_thread.is_alive() and b._spin_thread.is_alive()

    b.close()

    assert not b._up_thread.is_alive(), "upsampler outlived close()"
    assert not b._spin_thread.is_alive(), "spin thread outlived close()"


def test_close_destroys_node_only_after_threads_stopped(monkeypatch):
    """destroy_node() must be unreachable while either worker is alive.

    Ordering is checked from inside the fake node, at the instant of the call —
    an after-the-fact assertion would pass even if the race window existed.
    """
    b = _make_backend(monkeypatch)
    observed = {}

    def _destroy():
        observed["up_alive"] = b._up_thread.is_alive()
        observed["spin_alive"] = b._spin_thread.is_alive()
        observed["executor_shutdown_first"] = b._executor.shutdown_calls > 0
        b._node.destroyed = True
        b._node.destroy_calls += 1

    b._node.destroy_node = _destroy
    b.close()

    assert observed["up_alive"] is False
    assert observed["spin_alive"] is False
    # executor teardown belongs before the node's, and after the join
    assert observed["executor_shutdown_first"] is True


def test_executor_shutdown_never_races_the_spin_thread(monkeypatch):
    """Regression for the SIGSEGV the first fix attempt introduced.

    Executor.shutdown() destroys the guard condition that a spinning thread has
    already handed to a live rcl wait set. close() must join the spin thread
    out of the executor before shutting the executor down.
    """
    b = _make_backend(monkeypatch)
    b.close()

    assert b._executor.shutdown_calls == 1
    assert b._executor.shutdown_while_spinning is False, (
        "executor.shutdown() was called while a thread was inside the executor "
        "— that is the use-after-free on Executor._guard"
    )


def test_spin_loop_polls_the_stop_flag(monkeypatch):
    """_spin must exit on _shutdown alone.

    It may not depend on the context being torn down (the old rclpy.spin(node)
    did), because close() must be able to stop our spin thread without shutting
    down a context other nodes may share.
    """
    b = _make_backend(monkeypatch, owns_rclpy=False)
    b._shutdown = True
    b._spin_thread.join(timeout=2.0)

    assert not b._spin_thread.is_alive()
    assert b._fake_rclpy.shutdown_calls == 0


# --------------------------------------------------------------------------- #
# idempotency                                                                  #
# --------------------------------------------------------------------------- #
def test_close_is_idempotent(monkeypatch):
    """env.close() and a finally: block may both call it — see run_real_hil."""
    b = _make_backend(monkeypatch)

    b.close()
    b.close()
    b.close()

    assert b._node.destroy_calls == 1
    assert b._executor.shutdown_calls == 1
    assert b._fake_rclpy.shutdown_calls == 1


def test_concurrent_close_destroys_once(monkeypatch):
    """Two threads calling close() at once must not both tear down."""
    b = _make_backend(monkeypatch)
    barrier = threading.Barrier(4)

    def _closer():
        barrier.wait()
        b.close()

    threads = [threading.Thread(target=_closer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert b._node.destroy_calls == 1
    assert b._fake_rclpy.shutdown_calls == 1


# --------------------------------------------------------------------------- #
# no publish onto a destroyed node                                             #
# --------------------------------------------------------------------------- #
def test_no_command_published_after_close(monkeypatch):
    """The safety property, with DRY_RUN off.

    _upsample_loop republishes the last target at 250 Hz. After close() the
    robot must receive nothing further, and in particular nothing may still be
    in flight against a node that has been destroyed.

    The publish is deliberately slowed so close() reliably lands *inside* one,
    which is the window destroy_node() used to be free to run in.
    """
    b = _make_backend(
        monkeypatch, dry_run=False, with_target=True, publish_delay=0.05
    )
    time.sleep(0.15)
    assert b._cmd_pub.publishes, "loop should be commanding before close()"

    b.close()
    n_at_close = len(b._cmd_pub.publishes)
    time.sleep(0.2)

    assert len(b._cmd_pub.publishes) == n_at_close, "commands continued after close()"
    assert b._cmd_pub.after_destroy == [], "published onto a destroyed node"


def test_first_command_is_exact_measured_joint_seed(monkeypatch):
    """The first target must not consume one trajectory step before publish."""
    b = _make_backend(monkeypatch, dry_run=False, with_target=True)
    try:
        assert b._cmd_pub.publishes
        first_payload, _ = b._cmd_pub.publishes[0]
        np.testing.assert_array_equal(first_payload, Q6)
    finally:
        b.close()


def test_send_gripper_percent_is_a_noop_after_close(monkeypatch):
    """A late gripper command must not reach a destroyed node's publisher.

    The re-assert is disabled here so the count is exactly the historical one;
    the re-assert's own teardown behaviour is covered separately below.
    """
    b = _make_backend(monkeypatch, dry_run=False)
    b.grip_reassert_s = 0.0
    b.send_gripper_percent(1.0)
    assert len(b._gripper_pub.publishes) == 1

    b.close()
    b.send_gripper_percent(0.0)

    assert len(b._gripper_pub.publishes) == 1
    assert b._gripper_pub.after_destroy == []


# --------------------------------------------------------------------------- #
# gripper setpoint re-assert (FIX D)                                           #
#                                                                              #
# One publish is not a command. DDS drops a datagram sent before the            #
# subscription matched, and robotiq_gripper_modbus_node's rate limiter          #
# discards a setpoint that lands inside command_min_period WITHOUT updating     #
# its cache — so the driver stays self-consistent and nothing downstream can    #
# see the loss. Measured 2026-08-06: 7 of 14 one-shot opens parked short.       #
# --------------------------------------------------------------------------- #
def test_a_gripper_setpoint_is_re_asserted_by_the_worker(monkeypatch):
    """The value is repeated on the 250 Hz worker, not sent once and forgotten."""
    b = _make_backend(monkeypatch, dry_run=False)
    b.grip_reassert_s = 0.4
    b.grip_reassert_hz = 20.0
    try:
        b.send_gripper_percent(0.0)
        assert len(b._gripper_pub.publishes) == 1  # the immediate one
        time.sleep(0.25)
        payloads = [p for p, _ in b._gripper_pub.publishes]
        assert len(payloads) >= 3, payloads
        assert set(payloads) == {0.0}, "a re-assert must repeat, not invent"
    finally:
        b.close()


def test_the_re_assert_stops_when_its_window_expires(monkeypatch):
    """No periodic traffic at rest — the :54321 bus is single-client."""
    b = _make_backend(monkeypatch, dry_run=False)
    b.grip_reassert_s = 0.1
    b.grip_reassert_hz = 20.0
    try:
        b.send_gripper_percent(1.0)
        time.sleep(0.25)
        settled = len(b._gripper_pub.publishes)
        assert b._grip_pending is None, "pending setpoint outlived its window"
        time.sleep(0.15)
        assert len(b._gripper_pub.publishes) == settled
    finally:
        b.close()


def test_a_newer_setpoint_supersedes_the_pending_one(monkeypatch):
    """LATEST WINS: a stale re-assert must never fight a newer command.

    Without this the policy channel could have an "open" re-assert still firing
    after it asked to close one GRIPPER_SLEEP later.
    """
    b = _make_backend(monkeypatch, dry_run=False)
    b.grip_reassert_s = 1.0
    b.grip_reassert_hz = 20.0
    try:
        b.send_gripper_percent(0.0)  # OPEN
        time.sleep(0.1)
        b.send_gripper_percent(1.0)  # CLOSED
        n_at_switch = len(b._gripper_pub.publishes)
        time.sleep(0.2)
        after = [p for p, _ in b._gripper_pub.publishes[n_at_switch:]]
        assert after, "the newer setpoint should still be re-asserting"
        assert set(after) == {1.0}, f"stale OPEN re-asserted after CLOSE: {after}"
    finally:
        b.close()


#: Every Event.wait()/join() below is bounded by this. A concurrency test that
#: can hang is worse than no test: it would wedge the suite instead of failing.
#: Nothing waits on it in the happy path — it is the ceiling, not the schedule.
_RACE_TIMEOUT_S = 5.0

#: The closer thread is named so the instrumented lock can tell "the thread this
#: test is watching had to wait" from "the 250 Hz worker took its own lock".
_CLOSER_THREAD = "test-gripper-closer"


class _ContendedLock:
    """A Lock that reports the moment ``_CLOSER_THREAD`` has to wait for it.

    The race below is about what a second thread does *while* one publisher
    holds the lock, so the test needs to know when that second thread has
    reached its decision point. Waiting on the lock IS that point when the
    publish is serialised, so this is the deterministic signal that replaces a
    sleep. Only the named thread is watched — the upsampler takes this lock on
    every one of its 250 Hz ticks and would otherwise flag contention that has
    nothing to do with the race.
    """

    def __init__(self, on_contended):
        self._lock = threading.Lock()
        self._on_contended = on_contended

    def acquire(self, blocking=True, timeout=-1):
        if self._lock.acquire(False):
            return True
        if threading.current_thread().name == _CLOSER_THREAD:
            self._on_contended()
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()
        return False


class _GatedGripperPub:
    """Records the wire, and freezes the worker in the middle of ONE re-assert.

    A publish is not instantaneous: with a real rmw it is a C call, and the
    worker can be descheduled anywhere inside it. This holds the first re-assert
    (the repeat of the value already on the wire) open until the test releases
    it, which turns "the worker is between its decision and its datagram" from a
    ~microsecond window into a deterministic one the closer thread runs inside.
    """

    def __init__(self, held, release, re_assert_done, closer_committed):
        self.sent = []
        self._held = held
        self._release = release
        self._re_assert_done = re_assert_done
        self._closer_committed = closer_committed
        self._gated = False
        self.release_timed_out = False

    def publish(self, msg):
        value = float(msg.data)
        gated = not self._gated and value == 0.0 and len(self.sent) == 1
        if gated:
            self._gated = True
            self._held.set()
            if not self._release.wait(_RACE_TIMEOUT_S):
                self.release_timed_out = True
        self.sent.append(value)
        # Both signals fire AFTER the append, never before. Releasing the worker
        # on a signal raised ahead of its own append is how this test silently
        # stopped testing anything: the stale repeat then lands (or does not)
        # depending on when the scheduler runs the worker, not on whether the
        # publish is serialised — and the reverted fix passed.
        if gated:
            self._re_assert_done.set()
        if value == 1.0:
            self._closer_committed.set()


def test_a_newer_setpoint_wins_the_wire_not_just_the_state(monkeypatch):
    """LATEST WINS on the WIRE — the ordering property, not the stored one.

    ``test_a_newer_setpoint_supersedes_the_pending_one`` only spaces the two
    commands apart, so it checks the state machine and never opens the window
    that costs a grasp. This one does.

    The defect: the worker decided to re-assert, released the lock, and
    published outside it. A ``send_gripper_percent(1.0)`` completing in that gap
    put wire order ``[0.0, 1.0, 0.0]`` on the bus — an OPEN landing after a
    newer CLOSE. The driver's big-jump exemption (``|dv| >= 0.5``) waves exactly
    that inversion through, so the fingers open for one re-assert period in the
    middle of a hold: a dropped object, from code whose stored state was
    correct the whole time.

    The fix publishes under the same ``_lock`` on both sides. This test drives
    the real ``_upsample_loop`` worker and a real second thread, and pins the
    only observable that distinguishes the two: what reached the wire, in what
    order. Every wait is bounded, and the assertion tolerates the trailing
    repeats of the winning setpoint — it fails only on a stale value landing
    after a newer one.
    """
    b = _make_backend(monkeypatch, dry_run=False, start_threads=False)
    b.grip_reassert_s = 1.0
    b.grip_reassert_hz = 10.0

    worker_in_re_assert = threading.Event()
    release_worker = threading.Event()
    re_assert_done = threading.Event()
    closer_committed = threading.Event()

    pub = _GatedGripperPub(
        worker_in_re_assert, release_worker, re_assert_done, closer_committed
    )
    b._gripper_pub = pub
    b._lock = _ContendedLock(closer_committed.set)

    b._up_thread.start()
    b._spin_thread.start()

    closer = threading.Thread(
        target=b.send_gripper_percent,
        args=(1.0,),
        name=_CLOSER_THREAD,
        daemon=True,
    )
    try:
        b.send_gripper_percent(0.0)  # OPEN, arms a 1 s re-assert
        assert list(pub.sent) == [0.0]

        # 1. wait until the worker is INSIDE its re-assert publish
        assert worker_in_re_assert.wait(_RACE_TIMEOUT_S), "worker never re-asserted"

        # 2. the newer setpoint, from a second thread, inside that window
        closer.start()

        # 3. wait until that thread has committed: either it published (the
        #    publish is outside the lock -> the inversion is already on the
        #    wire) or it is parked on the lock (the publish is serialised).
        #    Both outcomes set the same event, so neither version can be
        #    released early and pass by timing.
        assert closer_committed.wait(_RACE_TIMEOUT_S), "closer never reached the lock"

        # 4. let the held re-assert finish, and wait for BOTH publishes to be
        #    on the wire before reading it. Reading after the join alone is not
        #    enough: the released worker still has to be scheduled, so the
        #    stale repeat could simply not have landed yet.
        release_worker.set()
        assert re_assert_done.wait(_RACE_TIMEOUT_S), "the re-assert never finished"
        closer.join(_RACE_TIMEOUT_S)
        assert not closer.is_alive(), "the newer setpoint never completed"
        wire = list(pub.sent)

        assert pub.release_timed_out is False
        assert wire[0] == 0.0
        assert 1.0 in wire, f"the newer setpoint never reached the wire: {wire}"
        tail = wire[wire.index(1.0):]
        assert set(tail) == {1.0}, (
            f"a stale OPEN landed after a newer CLOSE: {wire} — the re-assert "
            "must publish under the same lock as send_gripper_percent()"
        )
    finally:
        release_worker.set()  # never leave the worker parked in publish()
        b.close()


def test_the_shipped_re_assert_defaults_are_on(monkeypatch):
    """The values that actually ship, pinned.

    Every other test in this section sets ``grip_reassert_s``/``_hz``
    explicitly, so the class defaults were covered by nothing: either of them
    silently becoming 0.0 restores the one-shot publish that parked the gripper
    short in 7 of 14 real release cycles, and the whole suite still passes.

    1.0 s at 10 Hz is the window measured on the real stack (0 of 6 failures);
    see GRIPPER_REASSERT_S in ros_backend for both loss mechanisms. Retuning is
    fine — updating this pin deliberately is the point; drifting past it is not.
    """
    assert URRosBackend.grip_reassert_s > 0.0, "the shipped re-assert is disabled"
    assert URRosBackend.grip_reassert_hz > 0.0, "the shipped re-assert is disabled"
    assert URRosBackend.grip_reassert_s == rb.GRIPPER_REASSERT_S == 1.0
    assert URRosBackend.grip_reassert_hz == rb.GRIPPER_REASSERT_HZ == 10.0

    # ...and the defaults are live, not just present: _make_backend copies the
    # class attributes, so this arms and repeats on the shipped numbers alone.
    b = _make_backend(monkeypatch, dry_run=False, start_threads=False)
    assert b.grip_reassert_s == URRosBackend.grip_reassert_s
    assert b.grip_reassert_hz == URRosBackend.grip_reassert_hz

    before = time.monotonic()
    b.send_gripper_percent(0.0)
    assert b._grip_pending is not None, "the shipped defaults armed no re-assert"
    value, deadline, _next_at = b._grip_pending
    assert value == 0.0
    assert deadline >= before + URRosBackend.grip_reassert_s

    b._tick_gripper_reassert(time.monotonic() + 1.0 / URRosBackend.grip_reassert_hz)
    assert [p for p, _ in b._gripper_pub.publishes] == [0.0, 0.0]


def test_the_re_assert_is_off_when_the_window_is_zero(monkeypatch):
    """0.0 must restore exactly today's behaviour: one publish, nothing armed."""
    b = _make_backend(monkeypatch, dry_run=False)
    b.grip_reassert_s = 0.0
    try:
        b.send_gripper_percent(0.0)
        time.sleep(0.15)
        assert len(b._gripper_pub.publishes) == 1
        assert b._grip_pending is None
    finally:
        b.close()


def test_disabling_mid_flight_drops_the_pending_setpoint(monkeypatch):
    """Turning the feature off must not leave an old value looping."""
    b = _make_backend(monkeypatch, dry_run=False)
    b.grip_reassert_s = 1.0
    b.grip_reassert_hz = 20.0
    try:
        b.send_gripper_percent(0.0)
        assert b._grip_pending is not None
        b.grip_reassert_s = 0.0
        b.send_gripper_percent(1.0)
        assert b._grip_pending is None
        n = len(b._gripper_pub.publishes)
        time.sleep(0.15)
        assert len(b._gripper_pub.publishes) == n
    finally:
        b.close()


def test_no_re_assert_survives_close(monkeypatch):
    """close() stops the worker before it destroys the node — the safety
    property _upsample_loop already had, extended to the gripper channel."""
    b = _make_backend(monkeypatch, dry_run=False)
    b.grip_reassert_s = 5.0
    b.grip_reassert_hz = 50.0
    b.send_gripper_percent(1.0)
    time.sleep(0.1)
    assert len(b._gripper_pub.publishes) > 1

    b.close()
    n_at_close = len(b._gripper_pub.publishes)
    time.sleep(0.2)

    assert len(b._gripper_pub.publishes) == n_at_close
    assert b._gripper_pub.after_destroy == []


def test_dry_run_publishes_nothing_and_arms_nothing(monkeypatch):
    b = _make_backend(monkeypatch, dry_run=True)
    try:
        b.send_gripper_percent(0.0)
        time.sleep(0.1)
        assert b._gripper_pub.publishes == []
        assert b._grip_pending is None
    finally:
        b.close()


def test_send_joint_command_is_a_noop_after_close(monkeypatch):
    """A late target must not linger after teardown — nothing will act on it."""
    b = _make_backend(monkeypatch)
    b.close()
    b.send_joint_command(Q6 + 0.3)

    assert b._q_target is None


def test_send_joint_command_timestamps_target_and_reset_clears_state(monkeypatch):
    """Staleness is based on target receipt, and reset drops every old state."""
    b = _make_backend(monkeypatch, start_threads=False)
    before = time.monotonic()
    b.send_joint_command(Q6 + 0.3)
    after = time.monotonic()

    assert before <= b._q_target_time <= after
    b._up_streamer.seed(Q6, before)
    b._q_stream = Q6.copy()
    b.reset_command_stream()

    assert b._q_target is None
    assert b._q_target_time is None
    assert b._q_stream is None
    assert not b._up_streamer.seeded
    assert b._up_streamer.mode == rb.AccelerationLimitedJointStream.UNSEEDED


# --------------------------------------------------------------------------- #
# context ownership                                                            #
# --------------------------------------------------------------------------- #
def test_close_shuts_down_a_context_it_created(monkeypatch):
    b = _make_backend(monkeypatch, owns_rclpy=True)
    b.close()
    assert b._fake_rclpy.shutdown_calls == 1


def test_close_leaves_a_borrowed_context_alone(monkeypatch):
    """__init__ inits only ``if not rclpy.ok()``, i.e. it supports being dropped
    into a process that already has a context with other people's nodes on it.
    Tearing that down on our way out would kill them."""
    b = _make_backend(monkeypatch, owns_rclpy=False)
    b.close()

    assert b._fake_rclpy.shutdown_calls == 0
    assert b._fake_rclpy.ok() is True
    # our own node and executor still go away
    assert b._node.destroy_calls == 1
    assert b._executor.shutdown_calls == 1


# --------------------------------------------------------------------------- #
# wedged threads                                                               #
# --------------------------------------------------------------------------- #
def test_close_does_not_hang_on_a_wedged_thread(monkeypatch):
    """close() runs from a finally: block; it must not be what wedges shutdown."""
    b = _make_backend(monkeypatch, start_threads=False)
    release = threading.Event()
    b._up_thread = threading.Thread(target=release.wait, daemon=True)
    b._up_thread.start()
    b._spin_thread = threading.Thread(target=b._spin, daemon=True)
    b._spin_thread.start()
    time.sleep(0.05)

    t0 = time.monotonic()
    b.close(join_timeout=0.2)
    elapsed = time.monotonic() - t0
    release.set()

    assert elapsed < 2.0, f"close() blocked for {elapsed:.2f}s on a wedged thread"


def test_close_skips_teardown_when_a_thread_will_not_stop(monkeypatch, capsys):
    """If a worker is still running, destroying the node it uses is the very
    use-after-free this method exists to avoid. Leaking a node in a process that
    is exiting is the cheaper failure — but it must be reported."""
    b = _make_backend(monkeypatch, start_threads=False)
    release = threading.Event()
    b._up_thread = threading.Thread(target=release.wait, daemon=True)
    b._up_thread.start()
    b._spin_thread = threading.Thread(target=b._spin, daemon=True)
    b._spin_thread.start()
    time.sleep(0.05)

    b.close(join_timeout=0.2)
    release.set()

    assert b._node.destroy_calls == 0
    assert b._fake_rclpy.shutdown_calls == 0
    out = capsys.readouterr().out
    assert "upsampler" in out and "did not stop" in out


def test_publish_gate_closes_before_the_join(monkeypatch):
    """_node_alive must be down for the whole join window, not just after it —
    that flag is what keeps a mid-iteration loop off the node."""
    b = _make_backend(monkeypatch, start_threads=False)
    seen = []
    b._up_thread = threading.Thread(
        target=lambda: seen.append(b._node_alive), daemon=True
    )
    b._up_thread.start()
    b._spin_thread = threading.Thread(target=b._spin, daemon=True)
    b._spin_thread.start()
    time.sleep(0.05)

    b.close()

    assert b._node_alive is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
