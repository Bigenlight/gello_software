"""Server-side wiring for the opt-in per-step latency profiler.

WHAT THIS IS
------------
:mod:`ur_env.latency_profile` is the sink; this module is the plumbing that
connects it to the learner server's Step path.  It exists because the phases the
operator wants attributed are NOT all reachable from one function: the gRPC
servicer owns proto decode and reply build, ``ActorSessionService.step`` owns
validation and routing, and the actual work happens inside three collaborators
it was handed at construction time (the frozen-trunk sink, the reward finalizer,
the policy runtime).  Threading a profiler argument through every one of those
signatures would put profiling into the transport-independent service contract.

So the scope is carried on the handler's own thread instead.  gRPC serves Step
from a small pool (``--max-workers 4``) and everything the handler delegates to
runs synchronously on that same thread, so a :class:`threading.local` attributes
each phase to the request that produced it without a lock and without cross-talk
between concurrent handlers.  This is the shape ``ur_env/step_timing.py`` already
proved on the BC/FM serving entrypoints; the difference is that this one writes
through :class:`~ur_env.latency_profile.LatencyProfiler` so the actor and the
server produce the same JSONL schema and the offline analyzer can join them on
``transition_id``.

NESTING, STATED PLAINLY
-----------------------
The phases are not disjoint.  ``total`` contains everything by definition, and
``classifier`` is measured INSIDE ``reward_finalize`` (the classifier call is
part of finalizing the transition, and pretending otherwise would mean either
double bookkeeping or a subtraction the sink's accumulate-by-name contract
cannot express).  Do not sum phase columns and expect ``total``; read them as a
flamegraph, and derive the unattributed remainder as
``service_step - trunk_encode - reward_finalize - replay_insert``.

WHEN DISABLED, NOTHING HAPPENS
------------------------------
:meth:`ServerLatencyProbe.rpc` returns a shared stateless no-op scope, and
:meth:`ServerLatencyProbe.phase` a shared no-op context manager, so a server
started without ``HIL_LATENCY_PROFILE`` allocates nothing per RPC, calls no
:func:`time.perf_counter`, and -- because :func:`wrap_ingress_sink` hands back
the ingress unchanged -- does not even have the wrapper objects in its call
graph.  The one place that is not merely cheap but load-bearing is that wrapper:
``ActorSessionService._prime_replay_observation`` discovers frozen-trunk priming
with ``getattr(sink, "prime_observation", None)``, so a wrapper that forgot to
forward it would silently disable feature priming instead of failing.  That is
why there are two wrapper classes and a factory that picks between them.

NEVER BREAK THE CONTROL PATH
----------------------------
Every method here is total.  A profiling failure must cost a log line, never an
episode with the arm mid-reach, and the record for a raising handler is exactly
the record an operator investigating that failure wants -- so the scope commits
on the exception path too, with ``error`` set, and re-raises unchanged.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional

from ur_env.latency_profile import LatencyProfiler

__all__ = [
    "BEGIN_EPISODE_RPC",
    "STEP_RPC",
    "ServerLatencyProbe",
    "disabled_probe",
    "wrap_ingress_sink",
]

#: ``rpc`` field values.  The analyzer filters on these, so they are constants
#: rather than string literals repeated at the call sites.
STEP_RPC = "step"
BEGIN_EPISODE_RPC = "begin_episode"

#: A profiler with no destination is disabled by construction, and a disabled
#: profiler hands out the sink module's shared stateless no-op record.  Reaching
#: it through the public constructor keeps this module off private names.
_NULL_RECORD = LatencyProfiler("server", None, enabled=False).record()

#: The no-op context manager that same record returns for any phase.
_NULL_PHASE = _NULL_RECORD.phase("noop")


class _NullScope:
    """What :meth:`ServerLatencyProbe.rpc` returns when profiling is off."""

    __slots__ = ()

    def __enter__(self) -> Any:
        return _NULL_RECORD

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


_NULL_SCOPE = _NullScope()


class _RpcScope:
    """One profiled RPC handler: opens the record, times ``total``, commits."""

    __slots__ = ("_probe", "_rpc", "_record", "_total", "_counted")

    def __init__(self, probe: "ServerLatencyProbe", rpc: str) -> None:
        self._probe = probe
        self._rpc = rpc
        self._record: Any = _NULL_RECORD
        self._total: Any = None
        self._counted = False

    def __enter__(self) -> Any:
        try:
            record = self._probe._profiler.record()
            self._record = record
            record.set("rpc", self._rpc)
            if self._rpc == STEP_RPC:
                # Queue pressure, measured the only way one host can measure
                # it: how many Step handlers are inside the servicer right now,
                # THIS ONE INCLUDED (so the value is >= 1, and 1 means alone).
                # True queue wait would need the actor's clock, which rule 4 of
                # the design spec forbids subtracting from ours.
                self._counted = True
                record.set("concurrent_rpcs", self._probe._enter_step())
            self._probe._set_current(record)
            total = record.phase("total")
            total.__enter__()
            self._total = total
            return record
        except Exception:  # noqa: BLE001 - profiling never fails a handler
            self._record = _NULL_RECORD
            self._total = None
            return _NULL_RECORD

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if self._total is not None:
                self._total.__exit__(exc_type, exc, tb)
            if exc_type is not None:
                self._record.set("error", getattr(exc_type, "__name__", "error"))
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._counted:
                self._probe._exit_step()
            self._probe._set_current(None)
            self._record.commit()
        except Exception:  # noqa: BLE001
            pass
        # Never suppress: this object observes the handler, it does not repair
        # it.
        return False


class ServerLatencyProbe:
    """Per-RPC latency scopes for the learner server, plus the phase lookup.

    Construct one per process from :meth:`from_env` (or :func:`disabled_probe`
    in a server that has no run root), hand it to ``GrpcActorServicer`` /
    ``create_grpc_server`` and to ``build_actor_service``, and close it when the
    gRPC server has stopped.
    """

    def __init__(self, profiler: Any) -> None:
        self._profiler = profiler
        self._enabled = bool(getattr(profiler, "enabled", False))
        self._local = threading.local()
        self._lock = threading.Lock()
        self._active_steps = 0
        self._max_concurrent_steps = 0

    @classmethod
    def from_env(
        cls,
        out_path: Optional[Any] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
    ) -> "ServerLatencyProbe":
        """Enabled iff ``HIL_LATENCY_PROFILE`` is truthy in ``env``."""

        return cls(LatencyProfiler.from_env("server", out_path, env=env))

    @classmethod
    def disabled(cls) -> "ServerLatencyProbe":
        """A probe that can never write, for entrypoints that do not profile."""

        return cls(LatencyProfiler("server", None, enabled=False))

    # -- state -------------------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def profiler(self) -> Any:
        return self._profiler

    @property
    def path(self) -> Any:
        return getattr(self._profiler, "path", None)

    @property
    def active_steps(self) -> int:
        """Step handlers inside the servicer right now."""

        with self._lock:
            return self._active_steps

    @property
    def max_concurrent_steps(self) -> int:
        """High-water mark of overlapping Step handlers seen this process."""

        with self._lock:
            return self._max_concurrent_steps

    # -- use ---------------------------------------------------------------- #

    def rpc(self, rpc: str) -> Any:
        """Scope one RPC handler; the ``with`` target is the record."""

        if not self._enabled:
            return _NULL_SCOPE
        return _RpcScope(self, rpc)

    def phase(self, name: str) -> Any:
        """Time a region into whatever record this thread's RPC opened."""

        if not self._enabled:
            return _NULL_PHASE
        record = getattr(self._local, "record", None)
        if record is None:
            # Reached from the learner thread, or from a call outside any
            # handler: there is nothing to attribute it to, and inventing a
            # record here would emit rows no analyzer could key.
            return _NULL_PHASE
        return record.phase(name)

    def set(self, key: str, value: Any) -> None:
        """Attach a value to this thread's in-flight record, if there is one."""

        if not self._enabled:
            return
        record = getattr(self._local, "record", None)
        if record is None:
            return
        record.set(key, value)

    def close(self) -> None:
        """Flush and close the underlying sink.  Idempotent, never raises."""

        try:
            self._profiler.close()
        except Exception:  # noqa: BLE001
            pass

    # -- internals ---------------------------------------------------------- #

    def _set_current(self, record: Any) -> None:
        self._local.record = record

    def _enter_step(self) -> int:
        with self._lock:
            self._active_steps += 1
            if self._active_steps > self._max_concurrent_steps:
                self._max_concurrent_steps = self._active_steps
            return self._active_steps

    def _exit_step(self) -> None:
        with self._lock:
            if self._active_steps > 0:
                self._active_steps -= 1


def disabled_probe() -> ServerLatencyProbe:
    """The probe a call site uses when it was handed none."""

    return ServerLatencyProbe.disabled()


class _TimedSink:
    """Wrap an ``accept_data`` callable and time it as ``replay_insert``."""

    __slots__ = ("_inner", "_probe")

    def __init__(self, inner: Any, probe: ServerLatencyProbe) -> None:
        self._inner = inner
        self._probe = probe

    @property
    def inner(self) -> Any:
        return self._inner

    def __call__(self, data: Any, intervened: Any) -> Any:
        with self._probe.phase("replay_insert"):
            return self._inner(data, intervened)


class _TimedPrimingSink(_TimedSink):
    """The same, for a sink that also encodes the frozen trunk.

    ``prime_observation`` MUST be forwarded: ``ActorSessionService`` finds it
    with ``getattr``, so a wrapper without it would not raise -- it would
    quietly stop sharing the trunk feature, which is the whole reason the
    encoder runs once per step.
    """

    __slots__ = ()

    def prime_observation(self, **kwargs: Any) -> Any:
        with self._probe.phase("trunk_encode"):
            return self._inner.prime_observation(**kwargs)


def wrap_ingress_sink(ingress: Any, probe: Optional[ServerLatencyProbe]) -> Any:
    """Return ``ingress`` timed by ``probe``, or ``ingress`` itself when off.

    A disabled probe yields the original object, not a pass-through wrapper: a
    server that is not profiling must have the same call graph it had before
    this module existed.
    """

    if probe is None or not probe.enabled:
        return ingress
    if callable(getattr(ingress, "prime_observation", None)):
        return _TimedPrimingSink(ingress, probe)
    return _TimedSink(ingress, probe)
