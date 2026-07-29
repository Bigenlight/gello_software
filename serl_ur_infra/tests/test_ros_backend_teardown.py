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

    b._q = (Q6.copy(), time.monotonic())
    b._dq = None
    b._q_stream = None
    b._q_target = Q6 + 0.5 if with_target else None
    b._up_hz = 250.0
    b._up_step = 0.002

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


def test_send_gripper_percent_is_a_noop_after_close(monkeypatch):
    """A late gripper command must not reach a destroyed node's publisher."""
    b = _make_backend(monkeypatch, dry_run=False)
    b.send_gripper_percent(1.0)
    assert len(b._gripper_pub.publishes) == 1

    b.close()
    b.send_gripper_percent(0.0)

    assert len(b._gripper_pub.publishes) == 1
    assert b._gripper_pub.after_destroy == []


def test_send_joint_command_is_a_noop_after_close(monkeypatch):
    """A late target must not linger after teardown — nothing will act on it."""
    b = _make_backend(monkeypatch)
    b.close()
    b.send_joint_command(Q6 + 0.3)

    assert b._q_target is None


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
