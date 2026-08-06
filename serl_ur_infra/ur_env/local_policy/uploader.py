"""Raw-bytes transition forwarder: laptop3 proxy -> real learner server.

WHAT THIS IS
------------
In local-inference mode the actor talks to a proxy on laptop3 instead of the
GPU server.  The proxy answers the ACTION half of every RPC itself, but the
server is still the owner of replay, reward and training, so every
``BeginEpisode`` and ``Step`` the actor sends must still reach it -- just not on
the control loop's clock.  This class is that second path: a FIFO queue plus one
background thread that replays the captured requests to the real server in
order, one at a time.

WHY RAW BYTES AND NOT A REBUILT REQUEST
---------------------------------------
The server dedups a retried RPC by comparing a *fingerprint* of the request it
already answered (``ActorSessionService.step`` ->
``GrpcActorServicer.Step``'s ``request.SerializeToString(deterministic=True)``)
against the new one, keyed by ``(actor_id, session_id, request_id)``.  Anything
that re-derives the request -- rebuilding it from dataclasses, re-encoding an
observation, re-stamping ``created_monotonic_ns`` -- can change those bytes, and
a changed fingerprint turns a harmless retry into a *rejected* transition.  So
the proxy hands this class the exact serialized request it received, and this
class puts those exact bytes on the wire: ``channel.unary_unary`` with
``request_serializer=None`` sends the payload unchanged, with no parse and no
re-serialize anywhere in the path.  Identical bytes in, identical fingerprint
out, retries dedup safely.

That is also why this does NOT reuse :class:`~ur_env.grpc_actor_transport.GrpcActorNetwork`:
that client's whole surface is "give me an observation and a data mapping and I
will build the request", including its own ``request_id`` sequencing and
``created_monotonic_ns`` -- exactly the fields that must survive verbatim here.
What is reused is the part that must not drift: the transient-status-code set
(imported, never copied) and the message size limits.

WHAT COMES BACK IS MOSTLY THROWN AWAY
-------------------------------------
The server still runs its own inference for every forwarded Step (wasted
milliseconds, accepted for phase 1) and returns an action.  That action is
discarded entirely -- the actor already acted on the proxy's local one, ages
ago.  What is NOT discarded is the ``TransitionOutcome``: the proxy computed a
local MANUAL verdict for the same transition, and under MANUAL the two are
supposed to be *identical* by construction.  So every reply's outcome is
compared against the local summary supplied at enqueue time and any difference
is logged loudly and counted.  A divergence here is not an expected operating
condition; it is the alarm that says the local finalizer has drifted from the
server's.

NEVER DROP, NEVER RAISE, NEVER BLOCK THE LOOP
---------------------------------------------
* :meth:`enqueue` runs inside the proxy's Step handler.  It appends bytes to a
  deque and returns; it never does I/O, never parses a proto, and never raises
  for a runtime condition (a wrong *type* is a wiring bug and does raise).
* The queue is unbounded in RAM on purpose.  A bound would mean choosing which
  transition to destroy while the arm is moving; instead the depth is a metric
  and crossing the high-water mark warns once per excursion.  At ~96 KiB per
  Step, 500 entries is ~48 MB -- visible on a laptop with ~2 GB free, which is
  why ``queued_bytes`` is reported alongside the depth.
* The item at the head is only removed once the server has answered.  A crash or
  a stop mid-flight therefore loses at most the in-flight retry, never the
  queue's order.
* The worker thread catches everything.  A transient RPC failure retries the
  same bytes forever with capped backoff; an unexpected exception is treated as
  a failure of that item, not of the thread.

THE ONE CASE WHERE AN ITEM IS ABANDONED
---------------------------------------
A *non*-transient gRPC status (``INVALID_ARGUMENT``, ``FAILED_PRECONDITION``,
``INTERNAL``) is the server refusing these exact bytes.  Retrying them cannot
change the answer, and blocking the queue behind them would strand every later
transition too.  After ``permanent_retry_count`` further attempts such an item
is dropped from the head with a loud warning and counted in
:attr:`rejected_count` -- the server rejected it, this class did not silently
lose it.  The same accounting covers a delivered request the server ACKed with
``accepted=false``.

NO RESPONSE-AGE GATE
--------------------
``GrpcActorNetwork`` refuses a reply older than ``max_response_age_s`` because a
stale action must never reach the arm.  Nothing here reaches the arm: this path
is off the control loop by design, so there is deliberately no age gate and the
per-RPC timeout is a generous 10 s.

IMPORTS
-------
grpc + stdlib, plus this repo's generated pb2 modules.  No jax, no ROS.  Runs in
the ``gello-local-policy`` venv (grpcio 1.74) with the proxy, and in the actor
venv under test.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Any, Callable, Deque, Mapping, Optional

import grpc

# Imported, not copied: if the actor client's notion of "retryable" ever
# changes, this forwarder must change with it in the same commit.
from ur_env.grpc_actor_transport import (
    DEFAULT_MAX_MESSAGE_BYTES,
    _TRANSIENT_CODES as TRANSIENT_CODES,
)
from ur_env.latency_profile import LatencyProfiler
from ur_env.proto import actor_transport_pb2 as pb

__all__ = [
    "BEGIN_EPISODE_METHOD",
    "DEFAULT_HIGH_WATER_DEPTH",
    "DEFAULT_LATENCY_PROFILE_DIR",
    "DEFAULT_TARGET",
    "DEFAULT_TIMEOUT_S",
    "KIND_BEGIN_EPISODE",
    "KIND_STEP",
    "LATENCY_ROLE",
    "STEP_METHOD",
    "TransitionUploader",
]

_LOGGER = logging.getLogger(__name__)

#: The tunnel-local end of the ssh forward to the real server's gRPC port.
#: ``run_hil_server.sh`` already publishes the learner at 127.0.0.1:50153 on
#: laptop3; the proxy's own listener is a different port (50253).
DEFAULT_TARGET = "127.0.0.1:50153"

#: Per-RPC deadline.  Ten seconds, not the actor's 0.6: a forwarded Step waits
#: behind the server's own ~156 ms of inference and replay work while the
#: control loop has already moved on, and a spuriously short deadline here would
#: manufacture retries (and duplicate server work) for no benefit.
DEFAULT_TIMEOUT_S = 10.0

#: Depth at which the backlog stops being an implementation detail.  One warning
#: per excursion above it, cleared once the depth falls back to the low-water
#: mark, so a queue oscillating around the threshold cannot spam the log.
DEFAULT_HIGH_WATER_DEPTH = 500

#: ``role`` written into every latency record from this class.
LATENCY_ROLE = "uploader"

#: Fully-qualified method paths.  Literals rather than a live
#: :class:`ActorTransportStub`, because that stub bakes in the pb2 serializers
#: this class must bypass.  ``tests/test_transition_uploader.py`` asserts they
#: still equal what the generated stub dials, so they cannot drift silently.
BEGIN_EPISODE_METHOD = "/gello.hil_serl.v1.ActorTransport/BeginEpisode"
STEP_METHOD = "/gello.hil_serl.v1.ActorTransport/Step"

KIND_BEGIN_EPISODE = "begin_episode"
KIND_STEP = "step"
_KINDS = (KIND_BEGIN_EPISODE, KIND_STEP)

#: Where opt-in latency profiling writes when no ``HIL_LATENCY_PROFILE_DIR`` is
#: set.  Same directory the actor uses, derived the same way (from this file's
#: location, so a second checkout keeps its own samples).  ``local_policy/``
#: sits at ``<repo>/serl_ur_infra/ur_env/local_policy``.
DEFAULT_LATENCY_PROFILE_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "ros2_ur_ws",
    "gello_logs",
    "hil_latency",
)

#: Outcome fields compared against the local summary.  A summary supplies as
#: many or as few as it knows; only the keys it actually contains are checked,
#: so a caller can start with the three that matter (``done``/``truncated``/
#: ``success``) and tighten later without a change here.
_BOOL_OUTCOME_KEYS = ("done", "truncated", "success", "classifier_evaluated")
_FLOAT_OUTCOME_KEYS = ("reward", "mask")
_STR_OUTCOME_KEYS = ("transition_id", "reward_model_id")

#: Tolerance for the two float outcome fields.  Both sides derive them from the
#: same python floats, so any real difference is enormous compared to this.
_FLOAT_OUTCOME_TOL = 1e-9


@dataclass
class _Item:
    """One captured request awaiting delivery."""

    seq: int
    kind: str
    payload: bytes
    local_outcome: Optional[dict]
    enqueued_monotonic: float
    attempts: int = 0
    permanent_attempts: int = 0

    @property
    def transition_id(self) -> str:
        outcome = self.local_outcome
        if not outcome:
            return ""
        return str(outcome.get("transition_id", "") or "")


@dataclass
class _Counters:
    """Everything :meth:`TransitionUploader.metrics` reports, under one lock."""

    uploaded: int = 0
    rejected: int = 0
    divergences: int = 0
    retries: int = 0
    enqueued: int = 0
    queued_bytes: int = 0
    high_water_mark: int = 0
    last_error: str = ""
    consecutive_failures: int = 0


class TransitionUploader:
    """Replay captured actor requests to the real server, in order, off-loop.

    Typical use from the proxy::

        uploader = TransitionUploader()          # 127.0.0.1:50153
        uploader.start()
        ...
        uploader.enqueue_begin_episode(raw_begin_bytes)
        uploader.enqueue_step(raw_step_bytes, local_outcome_summary={
            "transition_id": tid, "done": False, "truncated": False,
            "success": False,
        })
        ...
        uploader.stop(drain=True)                # blocks until the queue empties

    Everything on this class is safe to call from any thread.  Nothing on it
    blocks the caller except :meth:`stop` and :meth:`wait_until_idle`, which are
    the two methods that say so in their names.
    """

    def __init__(
        self,
        target: str = DEFAULT_TARGET,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        channel: Any = None,
        profiler: Optional[LatencyProfiler] = None,
        high_water_depth: int = DEFAULT_HIGH_WATER_DEPTH,
        permanent_retry_count: int = 1,
        backoff_initial_s: float = 0.25,
        backoff_max_s: float = 5.0,
        warn_interval_s: float = 10.0,
        thread_name: str = "hil-transition-uploader",
        monotonic: Callable[[], float] = time.monotonic,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not target:
            raise ValueError("uploader target is required")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        if high_water_depth <= 0:
            raise ValueError("high_water_depth must be positive")
        if permanent_retry_count < 0:
            raise ValueError("permanent_retry_count must not be negative")
        if backoff_initial_s <= 0 or backoff_max_s < backoff_initial_s:
            raise ValueError("backoff bounds are inconsistent")
        self._target = str(target)
        self._timeout_s = float(timeout_s)
        self._max_message_bytes = int(max_message_bytes)
        self._high_water = int(high_water_depth)
        self._low_water = max(1, self._high_water // 2)
        self._permanent_retry_count = int(permanent_retry_count)
        self._backoff_initial_s = float(backoff_initial_s)
        self._backoff_max_s = float(backoff_max_s)
        self._warn_interval_s = float(warn_interval_s)
        self._thread_name = str(thread_name)
        self._monotonic = monotonic

        # A profiler is built here rather than at first use so that "disabled"
        # is decided once, at construction, exactly like the actor does it.  A
        # disabled profiler touches no filesystem, so this is free when the
        # operator did not ask for profiling.
        self._owns_profiler = profiler is None
        self._profiler = (
            LatencyProfiler.from_env(
                LATENCY_ROLE, DEFAULT_LATENCY_PROFILE_DIR, env=env
            )
            if profiler is None
            else profiler
        )

        self._channel = channel
        self._owns_channel = channel is None
        self._begin_call: Any = None
        self._step_call: Any = None

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: Deque[_Item] = collections.deque()
        self._counters = _Counters()
        self._next_seq = 0
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._abort = threading.Event()
        self._high_water_warned = False
        self._not_running_warned = False
        self._last_failure_warn = 0.0

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Start the background worker.  Idempotent; never touches the network."""

        with self._cv:
            previous = self._thread
            unwinding = self._stopping
        if previous is not None and previous.is_alive() and unwinding:
            # A previous stop() is still unwinding (its last RPC has not come
            # back yet).  Wait for it rather than race a second worker into the
            # same queue, which would put two requests in flight at once.
            previous.join(timeout=max(1.0, self._timeout_s))
        with self._cv:
            if self._thread is not None and self._thread.is_alive():
                if self._stopping:
                    _LOGGER.warning(
                        "transition uploader is still stopping; not starting a "
                        "second worker (%d transition(s) queued)",
                        len(self._queue),
                    )
                return
            self._stopping = False
            self._abort.clear()
            self._not_running_warned = False
            # Daemon: a wedged uploader (server gone, backlog unsendable) must
            # never be the reason the proxy process refuses to exit.
            self._thread = threading.Thread(
                target=self._run, name=self._thread_name, daemon=True
            )
            self._thread.start()

    def stop(self, drain: bool = True, *, timeout_s: float = 30.0) -> bool:
        """Stop the worker.  With ``drain`` wait (bounded) for an empty queue.

        Returns ``True`` only when the queue really is empty and the thread has
        finished -- a ``False`` return is the caller's cue that transitions are
        still buffered and that this shutdown is NOT clean.
        """

        with self._cv:
            thread = self._thread
            self._stopping = True
            if not drain:
                self._abort.set()
            self._cv.notify_all()
        if thread is not None and thread.is_alive():
            self._join_with_progress(thread, float(timeout_s) if drain else 5.0)
            if thread.is_alive():
                # The drain budget is spent.  Cut the current retry loop short
                # so the thread can leave; the queue keeps whatever is left.
                self._abort.set()
                with self._cv:
                    self._cv.notify_all()
                thread.join(timeout=max(1.0, self._timeout_s))
        with self._cv:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            drained = not self._queue and self._thread is None
        if drained or not self._thread_alive():
            self._close_transport()
        if not drained:
            _LOGGER.warning(
                "transition uploader stopped with %d undelivered transition(s) "
                "(%.1f MiB) queued for %s",
                self.backlog_depth,
                self.queued_bytes / (1024.0 * 1024.0),
                self._target,
            )
        return drained

    def close(self) -> None:
        """Stop without draining and release the channel.  Idempotent."""

        self.stop(drain=False)
        self._close_transport()
        if self._owns_profiler:
            self._profiler.close()

    def __enter__(self) -> "TransitionUploader":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Drain on a clean exit, abandon on an exception: an exception is
        # already unwinding something, and blocking it for 30 s helps nobody.
        self.stop(drain=exc_type is None)
        self._close_transport()
        if self._owns_profiler:
            self._profiler.close()
        return False

    # -- enqueue ------------------------------------------------------------ #

    def enqueue(
        self,
        kind: str,
        payload: Any,
        *,
        local_outcome_summary: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Queue one captured request.  Returns its monotonic sequence number.

        ``payload`` is the request exactly as it arrived on the wire.  It is
        never parsed here: this runs inside the proxy's Step handler, and the
        only work it may do is an append.
        """

        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        if isinstance(payload, (bytearray, memoryview)):
            payload = bytes(payload)
        if not isinstance(payload, bytes):
            raise TypeError(
                "payload must be the serialized request bytes, got "
                f"{type(payload).__name__}"
            )
        if not payload:
            raise ValueError("payload is empty")
        summary = dict(local_outcome_summary) if local_outcome_summary else None

        now = self._monotonic()
        with self._cv:
            seq = self._next_seq
            self._next_seq += 1
            self._queue.append(
                _Item(
                    seq=seq,
                    kind=kind,
                    payload=payload,
                    local_outcome=summary,
                    enqueued_monotonic=now,
                )
            )
            self._counters.enqueued += 1
            self._counters.queued_bytes += len(payload)
            depth = len(self._queue)
            if depth > self._counters.high_water_mark:
                self._counters.high_water_mark = depth
            queued_bytes = self._counters.queued_bytes
            warn_high_water = depth > self._high_water and not self._high_water_warned
            if warn_high_water:
                self._high_water_warned = True
            warn_not_running = (
                self._thread is None or not self._thread.is_alive()
            ) and not self._not_running_warned
            if warn_not_running:
                self._not_running_warned = True
            self._cv.notify()

        # Logging happens outside the lock: the queue must never wait on a sink.
        if warn_high_water:
            _LOGGER.warning(
                "transition upload backlog crossed the high-water mark: "
                "%d transitions (%.1f MiB) waiting for %s; the server is "
                "ingesting slower than the proxy is producing",
                depth,
                queued_bytes / (1024.0 * 1024.0),
                self._target,
            )
        if warn_not_running:
            _LOGGER.warning(
                "transition uploader is not running; %d transition(s) are "
                "buffered and will only be sent after start()",
                depth,
            )
        return seq

    def enqueue_begin_episode(
        self,
        payload: Any,
        *,
        local_outcome_summary: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Queue a ``BeginEpisodeRequest``.  It carries no outcome to compare."""

        return self.enqueue(
            KIND_BEGIN_EPISODE,
            payload,
            local_outcome_summary=local_outcome_summary,
        )

    def enqueue_step(
        self,
        payload: Any,
        *,
        local_outcome_summary: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Queue a ``StepRequest`` together with the proxy's local verdict."""

        return self.enqueue(
            KIND_STEP, payload, local_outcome_summary=local_outcome_summary
        )

    # -- metrics ------------------------------------------------------------ #

    @property
    def target(self) -> str:
        return self._target

    @property
    def backlog_depth(self) -> int:
        """Transitions accepted but not yet delivered (includes the in-flight one)."""

        with self._lock:
            return len(self._queue)

    @property
    def queued_bytes(self) -> int:
        """Bytes held by the backlog -- the RAM this queue is costing."""

        with self._lock:
            return self._counters.queued_bytes

    @property
    def oldest_age_s(self) -> float:
        """Seconds the head item has been waiting; ``0.0`` when idle."""

        with self._lock:
            return self._oldest_age_locked()

    @property
    def uploaded_count(self) -> int:
        """Requests the server accepted."""

        with self._lock:
            return self._counters.uploaded

    @property
    def divergence_count(self) -> int:
        """Replies whose outcome disagreed with the local verdict."""

        with self._lock:
            return self._counters.divergences

    @property
    def rejected_count(self) -> int:
        """Requests the server refused (or abandoned after a permanent error)."""

        with self._lock:
            return self._counters.rejected

    @property
    def retry_count(self) -> int:
        """Extra attempts spent on transient failures."""

        with self._lock:
            return self._counters.retries

    @property
    def high_water_mark(self) -> int:
        """Deepest the backlog has been this session."""

        with self._lock:
            return self._counters.high_water_mark

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._counters.last_error

    @property
    def running(self) -> bool:
        return self._thread_alive()

    def metrics(self) -> dict:
        """One consistent snapshot of everything above, taken under the lock."""

        with self._lock:
            counters = self._counters
            return {
                "target": self._target,
                "running": self._thread is not None and self._thread.is_alive(),
                "backlog_depth": len(self._queue),
                "queued_bytes": counters.queued_bytes,
                "oldest_age_s": round(self._oldest_age_locked(), 3),
                "enqueued_count": counters.enqueued,
                "uploaded_count": counters.uploaded,
                "rejected_count": counters.rejected,
                "divergence_count": counters.divergences,
                "retry_count": counters.retries,
                "high_water_mark": counters.high_water_mark,
                "consecutive_failures": counters.consecutive_failures,
                "last_error": counters.last_error,
            }

    def wait_until_idle(self, timeout_s: float = 30.0) -> bool:
        """Block until the queue is empty.  ``True`` iff it emptied in time."""

        deadline = self._monotonic() + float(timeout_s)
        with self._cv:
            while self._queue:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(min(remaining, 0.25))
            return True

    # -- worker ------------------------------------------------------------- #

    def _run(self) -> None:
        """The background thread.  Total: it may log, it may not raise."""

        try:
            while True:
                with self._cv:
                    while not self._queue and not self._stopping:
                        self._cv.wait(0.25)
                    if self._abort.is_set():
                        return
                    if not self._queue:
                        if self._stopping:
                            return
                        continue
                    item = self._queue[0]
                try:
                    delivered = self._deliver(item)
                except Exception as exc:  # noqa: BLE001 - a poison item is not
                    # a reason to stop forwarding everything behind it.
                    self._abandon(item, f"uploader bug: {type(exc).__name__}: {exc}")
                    _LOGGER.exception(
                        "transition uploader dropped seq=%d after an "
                        "unexpected error",
                        item.seq,
                    )
                    delivered = True
                if not delivered:
                    # Aborted mid-flight: leave the item at the head so the
                    # order survives into the next start().
                    return
                with self._cv:
                    if self._queue and self._queue[0] is item:
                        self._queue.popleft()
                        self._counters.queued_bytes -= len(item.payload)
                        if (
                            self._high_water_warned
                            and len(self._queue) <= self._low_water
                        ):
                            self._high_water_warned = False
                    self._cv.notify_all()
        except BaseException:  # noqa: BLE001 - the thread must not die silently
            _LOGGER.exception("transition uploader worker stopped unexpectedly")

    def _deliver(self, item: _Item) -> bool:
        """Push one item until the server answers.  ``False`` iff aborted."""

        record = self._profiler.record()
        if record.enabled:
            record.set("kind", item.kind)
            # NOT "seq": that key belongs to the profiler (it stamps the line
            # number) and a caller's value would be silently overwritten.
            record.set("upload_seq", item.seq)
            if item.transition_id:
                record.set("transition_id", item.transition_id)
            record.set("backlog_depth", self.backlog_depth)
            record.set("oldest_backlog_s", round(self.oldest_age_s, 3))
            record.set("payload_bytes", len(item.payload))
        outcome_divergence = False
        delivered = False
        try:
            while not self._abort.is_set():
                item.attempts += 1
                try:
                    with record.phase("upload_rpc"):
                        reply = self._call(item)
                except grpc.RpcError as exc:
                    code = exc.code()
                    detail = _rpc_detail(exc)
                    self._note_failure(f"{code}: {detail}")
                    if code in TRANSIENT_CODES:
                        with self._lock:
                            self._counters.retries += 1
                        self._warn_failure_rate_limited(
                            item, f"transient {code}: {detail}"
                        )
                        self._backoff(item.attempts)
                        continue
                    item.permanent_attempts += 1
                    if item.permanent_attempts > self._permanent_retry_count:
                        self._abandon(item, f"{code}: {detail}")
                        delivered = True  # advance past it; it is accounted for
                        break
                    with self._lock:
                        self._counters.retries += 1
                    self._backoff(item.attempts)
                    continue
                except Exception as exc:  # noqa: BLE001 - never kill the thread
                    self._note_failure(f"{type(exc).__name__}: {exc}")
                    item.permanent_attempts += 1
                    if item.permanent_attempts > self._permanent_retry_count:
                        self._abandon(item, f"{type(exc).__name__}: {exc}")
                        delivered = True
                        break
                    self._warn_failure_rate_limited(
                        item, f"{type(exc).__name__}: {exc}"
                    )
                    self._backoff(item.attempts)
                    continue
                try:
                    outcome_divergence = self._handle_reply(item, reply, record)
                except Exception as exc:  # noqa: BLE001 - the bytes are already
                    # at the server; only the bookkeeping failed.
                    self._abandon(
                        item, f"reply not understood: {type(exc).__name__}: {exc}"
                    )
                delivered = True
                break
        finally:
            if record.enabled:
                record.set("attempts", item.attempts)
                record.set("delivered", bool(delivered))
                record.set("outcome_divergence", bool(outcome_divergence))
                record.commit()
        return delivered

    def _call(self, item: _Item) -> Any:
        """Send the captured bytes verbatim and return the parsed reply."""

        begin_call, step_call = self._ensure_transport()
        rpc = begin_call if item.kind == KIND_BEGIN_EPISODE else step_call
        return rpc(item.payload, timeout=self._timeout_s)

    def _ensure_transport(self) -> tuple[Any, Any]:
        with self._lock:
            if self._begin_call is not None and self._step_call is not None:
                return self._begin_call, self._step_call
            channel = self._channel
            if channel is None:
                channel = grpc.insecure_channel(
                    self._target,
                    options=(
                        (
                            "grpc.max_receive_message_length",
                            self._max_message_bytes,
                        ),
                        ("grpc.max_send_message_length", self._max_message_bytes),
                    ),
                )
                self._channel = channel
            # request_serializer=None is the whole point: grpc puts the payload
            # on the wire unchanged, so the server fingerprints exactly the
            # bytes the actor produced.
            self._begin_call = channel.unary_unary(
                BEGIN_EPISODE_METHOD,
                request_serializer=None,
                response_deserializer=pb.ActionReply.FromString,
            )
            self._step_call = channel.unary_unary(
                STEP_METHOD,
                request_serializer=None,
                response_deserializer=pb.StepReply.FromString,
            )
            return self._begin_call, self._step_call

    def _close_transport(self) -> None:
        with self._lock:
            channel = self._channel
            owns = self._owns_channel
            self._begin_call = None
            self._step_call = None
            if owns:
                self._channel = None
        if owns and channel is not None:
            try:
                channel.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass

    def _thread_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # -- reply handling ----------------------------------------------------- #

    def _handle_reply(self, item: _Item, reply: Any, record: Any) -> bool:
        """Account for one answered request.  Returns True on divergence."""

        if item.kind == KIND_BEGIN_EPISODE:
            # ActionReply.  The action is discarded; ok=false means the server's
            # own policy faulted on an episode the proxy already started, which
            # no retry of these bytes can change.
            if not reply.ok:
                self._abandon(item, reply.error or "server refused BeginEpisode")
                return False
            self._note_success()
            with self._lock:
                self._counters.uploaded += 1
            return False

        ack = reply.ack
        if not ack.accepted:
            self._abandon(
                item, ack.error or "server rejected the transition without a reason"
            )
            return False
        self._note_success()
        with self._lock:
            self._counters.uploaded += 1
        if record.enabled:
            record.set("deduplicated", bool(ack.deduplicated))
            if ack.transition_id:
                record.set("transition_id", ack.transition_id)
        # reply.action / reply.has_action are deliberately not read: the actor
        # acted on the proxy's local action one control period ago.
        if not reply.HasField("outcome"):
            return False
        return self._compare_outcome(item, reply.outcome)

    def _compare_outcome(self, item: _Item, outcome: Any) -> bool:
        summary = item.local_outcome
        if not summary:
            return False
        differences = []
        for key in _BOOL_OUTCOME_KEYS:
            if key not in summary:
                continue
            local = bool(summary[key])
            remote = bool(getattr(outcome, key))
            if local != remote:
                differences.append((key, local, remote))
        for key in _FLOAT_OUTCOME_KEYS:
            if key not in summary:
                continue
            try:
                local = float(summary[key])
            except (TypeError, ValueError):
                differences.append((key, summary[key], float(getattr(outcome, key))))
                continue
            remote = float(getattr(outcome, key))
            if abs(local - remote) > _FLOAT_OUTCOME_TOL:
                differences.append((key, local, remote))
        for key in _STR_OUTCOME_KEYS:
            if key not in summary:
                continue
            local = str(summary[key])
            remote = str(getattr(outcome, key))
            if local != remote:
                differences.append((key, local, remote))
        if not differences:
            return False
        with self._lock:
            self._counters.divergences += 1
            count = self._counters.divergences
        # Loud and per occurrence, never rate limited: under MANUAL the local
        # finalizer replicates the server's branch exactly, so a difference here
        # is a parity bug in one of the two -- the single most important thing
        # this class can report.
        _LOGGER.warning(
            "OUTCOME DIVERGENCE #%d: local verdict != server verdict for "
            "transition_id=%r (uploader seq=%d, kind=%s, remote "
            "transition_id=%r): %s",
            count,
            item.transition_id or "<unknown>",
            item.seq,
            item.kind,
            outcome.transition_id,
            ", ".join(
                f"{key}: local={local!r} server={remote!r}"
                for key, local, remote in differences
            ),
        )
        return True

    # -- failure accounting ------------------------------------------------- #

    def _abandon(self, item: _Item, reason: str) -> None:
        with self._lock:
            self._counters.rejected += 1
            self._counters.last_error = reason
            count = self._counters.rejected
        _LOGGER.warning(
            "TRANSITION NOT INGESTED (#%d): server refused uploader seq=%d "
            "kind=%s transition_id=%r after %d attempt(s): %s",
            count,
            item.seq,
            item.kind,
            item.transition_id or "<unknown>",
            item.attempts,
            reason,
        )

    def _note_failure(self, reason: str) -> None:
        with self._lock:
            self._counters.consecutive_failures += 1
            self._counters.last_error = reason

    def _note_success(self) -> None:
        with self._lock:
            failures = self._counters.consecutive_failures
            self._counters.consecutive_failures = 0
        if failures:
            _LOGGER.info(
                "transition uploader recovered after %d failed attempt(s); "
                "%d transition(s) still queued",
                failures,
                self.backlog_depth,
            )

    def _warn_failure_rate_limited(self, item: _Item, reason: str) -> None:
        now = self._monotonic()
        with self._lock:
            if (
                self._last_failure_warn
                and (now - self._last_failure_warn) < self._warn_interval_s
            ):
                return
            self._last_failure_warn = now
            depth = len(self._queue)
            failures = self._counters.consecutive_failures
        _LOGGER.warning(
            "transition upload to %s failing (%d consecutive, %d queued); "
            "retrying uploader seq=%d: %s",
            self._target,
            failures,
            depth,
            item.seq,
            reason,
        )

    def _backoff(self, attempts: int) -> None:
        delay = min(
            self._backoff_max_s, self._backoff_initial_s * (2 ** max(0, attempts - 1))
        )
        # Interruptible: stop(drain=False) must not wait out a 5 s sleep.
        self._abort.wait(delay)

    # -- misc --------------------------------------------------------------- #

    def _oldest_age_locked(self) -> float:
        if not self._queue:
            return 0.0
        return max(0.0, self._monotonic() - self._queue[0].enqueued_monotonic)

    def _join_with_progress(self, thread: threading.Thread, timeout_s: float) -> None:
        deadline = self._monotonic() + max(0.0, timeout_s)
        last_report = self._monotonic()
        while thread.is_alive():
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return
            thread.join(timeout=min(remaining, 0.25))
            now = self._monotonic()
            if thread.is_alive() and (now - last_report) >= 5.0:
                last_report = now
                _LOGGER.info(
                    "draining transition uploader: %d transition(s) left, "
                    "oldest queued %.1f s ago",
                    self.backlog_depth,
                    self.oldest_age_s,
                )


def _rpc_detail(exc: grpc.RpcError) -> str:
    try:
        return exc.details() or ""
    except Exception:  # noqa: BLE001 - a detail-less error is still an error
        return str(exc)
