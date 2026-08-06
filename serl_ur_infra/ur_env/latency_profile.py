"""Opt-in per-step, per-phase latency sink for the HIL-SERL control loop.

WHY THIS EXISTS
---------------
The production actor is measured at **1.95 Hz** against a nominal 10 Hz loop
(512 ms mean period, 854 ms max), and roughly 412 ms of that sits *outside*
``env.step`` -- in the blocking Step RPC and the camera decode.  What the stack
records today cannot say where those milliseconds actually go:
``_RoundTripStats`` (``ur_env/remote_actor.py``) keeps a streaming count/mean/max
of the Step round trip and prints it once at exit, and the learner's own
``timing/*`` fields describe the learner thread only.  Neither survives the
session as per-step samples, so no percentile, no tail, and no attribution to a
phase can be computed after the fact.

This module is that missing sample store: one JSON line per profiled region,
with every phase of that region as a separate ``<name>_ms`` field, so an offline
analyzer can report p50/p90/p99/max per phase and answer *which* phase owns the
tail.

WHY TWO LOCAL FILES INSTEAD OF A WIRE FIELD
-------------------------------------------
The obvious design -- ship the server's timings back inside the Step response --
would require a proto change, and protobuf **silently drops unknown fields**: a
half-upgraded pair (new actor, reused old learner, or the reverse) would produce
plausible-looking numbers with no error anywhere (the same failure mode as
``08_OPEN_GAPS.md`` G33).  So each host writes its own JSONL, both keyed by the
``transition_id`` that already exists in the transport dataclasses, and the join
happens offline.  No proto, no ``SCHEMA_VERSION``, no handshake impact.

THE CLOCK RULE (LOAD-BEARING)
-----------------------------
Durations come from :func:`time.perf_counter` and are therefore only ever
subtracted *within one host*.  ``t_epoch`` (:func:`time.time`) is written for
same-host ordering and for correlating a record with events in the learner's own
``logs/learner.jsonl`` -- it is **not** comparable across laptop3 and the GPU
server, whose clocks are unsynchronised.  Network+queue time is derived offline
as ``step_rpc_ms - server_total_ms``, each measured on its own host.

WHY IT NEVER RAISES AND NEVER BLOCKS STARTUP
--------------------------------------------
This object sits directly in the step path of a live UR7e session.  An exception
raised here would not merely lose a log line -- it would propagate into the
actor loop or a gRPC handler and end an episode with the arm mid-reach.  So
:meth:`LatencyRecord.commit` catches everything, warns exactly once and
self-degrades to disabled, which is the same rule commit ``c86dc54`` established
for the learner's metrics sink: a dead sink degrades the logger, it does not end
training.

For the same reason the constructor touches **no filesystem** (the precedent is
``ur_env/bc_inference_log.py``): a full disk, a read-only mount or a typo'd path
must surface as one warning at the first commit, never as a session that refuses
to start.  Directory creation and file open are lazy, on first commit.

WHEN DISABLED, NOTHING HAPPENS
------------------------------
``HIL_LATENCY_PROFILE`` unset (or ``0``) means: no path resolution, no
``mkdir``, no file, no :func:`time.perf_counter` call, and no allocation --
:meth:`LatencyProfiler.record` hands back a shared stateless no-op record whose
``phase()`` returns a shared no-op context manager.  Call sites pay one
attribute read and one branch.

IMPORTS
-------
Pure stdlib.  Both peers import this module: the actor venv
(``/home/laptop3/venvs/gello-hil-actor``) and the server's conda env, which do
not share a numpy or a jax.  Do not add third-party imports here.

Warnings go through :mod:`logging` rather than a bare ``print``.  Nothing in
``ur_env`` configures logging, but a WARNING with no handler still reaches
stderr via Python's ``lastResort`` handler -- which is what the launcher
redirects into the run's ``logs/stdout.log``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Optional, Union

__all__ = [
    "ENABLE_ENV_VAR",
    "DIR_ENV_VAR",
    "FLUSH_EVERY_N_COMMITS",
    "FLUSH_INTERVAL_S",
    "SCHEMA_VERSION",
    "LatencyProfiler",
    "LatencyRecord",
    "is_profiling_enabled",
]

_LOGGER = logging.getLogger(__name__)

#: Every emitted line carries this, so the analyzer can evolve the format
#: without having to guess what an old file means.
SCHEMA_VERSION = 1

#: Opt-in switch.  Read once, at :meth:`LatencyProfiler.from_env`.
ENABLE_ENV_VAR = "HIL_LATENCY_PROFILE"

#: Operator override for the output directory.  Takes precedence over whatever
#: default the call site passes, so a session can be redirected without editing
#: code on either host.
DIR_ENV_VAR = "HIL_LATENCY_PROFILE_DIR"

#: Flush cadence.  Buffered writes keep the control path off the disk, but an
#: unflushed tail is invisible to ``tail -f`` and lost on SIGKILL, so the buffer
#: is bounded in both records and wall-clock seconds.
FLUSH_EVERY_N_COMMITS = 50
FLUSH_INTERVAL_S = 5.0

#: Accepted truthy spellings of ``HIL_LATENCY_PROFILE``, compared lowercased
#: and stripped.  Anything else -- including ``0``, ``""`` and unset -- is off.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Keys the record owns; a caller's :meth:`LatencyRecord.set` can never shadow
#: them, because the analyzer keys off them.
_RESERVED_KEYS = ("schema", "role", "t_epoch", "seq")

#: Characters kept when a role becomes part of a filename.
_SAFE_ROLE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def is_profiling_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """True iff ``HIL_LATENCY_PROFILE`` is set to a truthy spelling.

    Exposed so a call site can skip building profiling-only values (ids,
    gauges) that cost something to compute, without reaching into the profiler.
    """

    source = os.environ if env is None else env
    return str(source.get(ENABLE_ENV_VAR, "") or "").strip().lower() in _TRUTHY


def _sanitize_role(role: str) -> str:
    text = "".join(ch for ch in str(role) if ch in _SAFE_ROLE_CHARS)
    return text or "unknown"


def _default_filename(role: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_sanitize_role(role)}_{os.getpid()}.jsonl"


class _NullPhase:
    """Context manager that does nothing, shared by every disabled call site."""

    __slots__ = ()

    def __enter__(self) -> "_NullPhase":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


_NULL_PHASE = _NullPhase()


class _NullRecord:
    """The record a disabled profiler hands out.

    Stateless on purpose: a single instance is shared by every thread and every
    call site, so a disabled profiler allocates nothing per step.
    """

    __slots__ = ()

    enabled = False

    def phase(self, name: str) -> _NullPhase:
        return _NULL_PHASE

    def set(self, key: str, value: Any) -> None:
        return None

    def mark(self, name: str) -> None:
        return None

    def commit(self) -> None:
        return None

    @property
    def fields(self) -> dict:
        return {}


_NULL_RECORD = _NullRecord()


class _Phase:
    """Times one region and folds the result into its record on exit."""

    __slots__ = ("_record", "_key", "_t0")

    def __init__(self, record: "LatencyRecord", key: str) -> None:
        self._record = record
        self._key = key
        self._t0 = 0.0

    def __enter__(self) -> "_Phase":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Record the partial duration even when the body raised -- a phase that
        # blew up is exactly the one worth seeing in the file -- then let the
        # exception through.  Swallowing here would hide control-path failures.
        self._record._add_phase(self._key, (time.perf_counter() - self._t0) * 1000.0)
        return False


class LatencyRecord:
    """One profiled region: a bag of phase durations plus caller-set values.

    Phases are flat and sequential; nesting is neither needed nor tracked.
    **The same phase name used twice in one record accumulates (the durations
    are summed into a single ``<name>_ms`` field).**  That is deliberate: a
    retry loop or a per-camera decode should report total time in that phase,
    not silently keep only the last pass.

    All operations after :meth:`commit` are ignored, so a record cannot emit two
    lines.
    """

    __slots__ = ("_profiler", "_phases", "_fields", "_t_start", "_t_epoch", "_committed")

    enabled = True

    def __init__(self, profiler: "LatencyProfiler") -> None:
        self._profiler = profiler
        self._phases: dict = {}
        self._fields: dict = {}
        # Wall clock of the *start* of the region, not of the commit: the
        # analyzer overlaps [t_epoch, t_epoch + total_ms] against same-host
        # learner events, which only works if this anchors the beginning.
        self._t_epoch = time.time()
        self._t_start = time.perf_counter()
        self._committed = False

    def phase(self, name: str):
        """Context manager timing one phase; sets ``f"{name}_ms"`` on exit."""

        if self._committed:
            return _NULL_PHASE
        return _Phase(self, f"{name}_ms")

    def set(self, key: str, value: Any) -> None:
        """Attach an id, gauge or flag.  Last write wins for a repeated key."""

        if self._committed:
            return
        self._fields[str(key)] = value

    def mark(self, name: str) -> None:
        """Stamp ``f"{name}_at_ms"``: ms from record creation to this point.

        An offset, not a duration -- use :meth:`phase` for durations.  A
        repeated mark overwrites, because a point in time has no meaningful sum.
        """

        if self._committed:
            return
        self._fields[f"{name}_at_ms"] = round(
            (time.perf_counter() - self._t_start) * 1000.0, 3
        )

    @property
    def fields(self) -> dict:
        """Copy of what this record would emit, minus the reserved keys."""

        merged = dict(self._fields)
        merged.update(
            {key: round(value, 3) for key, value in self._phases.items()}
        )
        return merged

    def commit(self) -> None:
        """Write one JSON line.  Never raises; a second call does nothing."""

        if self._committed:
            return
        self._committed = True
        self._profiler._write(self)

    # -- internals ---------------------------------------------------------- #

    def _add_phase(self, key: str, elapsed_ms: float) -> None:
        # Accumulate raw, round once at commit: rounding every addend would
        # drift a long accumulation by up to 0.5 us per call.
        self._phases[key] = self._phases.get(key, 0.0) + elapsed_ms

    def _payload(self, *, role: str, seq: int) -> dict:
        payload = dict(self._fields)
        for key, value in self._phases.items():
            payload[key] = round(value, 3)
        payload["schema"] = SCHEMA_VERSION
        payload["role"] = role
        payload["t_epoch"] = self._t_epoch
        payload["seq"] = seq
        return payload


class LatencyProfiler:
    """Opt-in per-step phase timing sink.  One JSONL line per committed record.

    Thread-safe: the gRPC servicer runs Step on up to four handler threads, so
    sequence allocation, buffering and flushing are all guarded by one lock.
    Serialisation happens under that lock as well -- it costs a few microseconds
    for a dozen small fields, and doing it outside would let lines land in an
    order that disagrees with their own ``seq``.
    """

    def __init__(
        self,
        role: str,
        path: Optional[Union[str, os.PathLike]] = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._role = str(role)
        # A profiler with nowhere to write is a disabled profiler, not an error:
        # this constructor must never be able to fail a session.
        self._enabled = bool(enabled) and path is not None
        self._path = (
            Path(os.path.expanduser(os.fspath(path)))
            if (path is not None and self._enabled)
            else None
        )
        self._lock = threading.Lock()
        self._stream = None
        self._seq = 0
        self._pending = 0
        self._last_flush = time.monotonic()
        self._closed = False
        self._degraded = False
        self._warned = False

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_env(
        cls,
        role: str,
        out_path: Optional[Union[str, os.PathLike]] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
    ) -> "LatencyProfiler":
        """Build a profiler for ``role`` (``'actor'`` / ``'server'``).

        Enabled iff ``HIL_LATENCY_PROFILE`` is truthy; a disabled profiler
        resolves no path and never touches the filesystem.

        Destination, in precedence order:

        1. ``HIL_LATENCY_PROFILE_DIR`` when set -- generated filename inside it.
           The operator override always wins.
        2. ``out_path`` from the call site.  It is read as a **file** when it
           ends in ``.jsonl`` (the server's ``<run_root>/logs/latency_server.jsonl``)
           and otherwise as a **directory** to generate a filename in (the
           actor's ``gello_logs/hil_latency/``).  A suffix check, not a
           ``stat``, so the answer does not depend on what already exists.
        3. The process working directory, as a last resort.

        Generated filenames are ``<utc YYYYmmdd_HHMMSS>_<role>_<pid>.jsonl``, so
        two processes -- or two sessions -- never collide.  Files are opened in
        append mode regardless.
        """

        if not is_profiling_enabled(env):
            return cls(role, None, enabled=False)

        source = os.environ if env is None else env
        env_dir = str(source.get(DIR_ENV_VAR, "") or "").strip()
        if env_dir:
            path = Path(os.path.expanduser(env_dir)) / _default_filename(role)
        elif out_path is not None:
            candidate = Path(os.path.expanduser(os.fspath(out_path)))
            path = (
                candidate
                if candidate.suffix == ".jsonl"
                else candidate / _default_filename(role)
            )
        else:
            path = Path.cwd() / _default_filename(role)
        return cls(role, path, enabled=True)

    # -- state -------------------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        """False when opted out, after :meth:`close`, or after degrading."""

        return self._enabled

    @property
    def degraded(self) -> bool:
        """True when an I/O failure turned a live profiler off mid-session."""

        return self._degraded

    @property
    def role(self) -> str:
        return self._role

    @property
    def path(self) -> Optional[Path]:
        """Destination file, or ``None`` when disabled.  May not exist yet."""

        return self._path

    @property
    def seq(self) -> int:
        """Number of lines written so far (also the next record's ``seq``)."""

        return self._seq

    # -- use ---------------------------------------------------------------- #

    def record(self) -> Any:
        """Start a record.  Disabled profilers return the shared no-op record."""

        if not self._enabled:
            return _NULL_RECORD
        return LatencyRecord(self)

    def close(self) -> None:
        """Flush and close.  Idempotent, and safe on a disabled profiler."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._enabled = False
            stream = self._stream
            self._stream = None
            self._pending = 0
            if stream is None:
                return
            try:
                stream.flush()
                stream.close()
            except Exception as exc:  # pragma: no cover - defensive
                self._warn_locked(f"could not close {self._path}: {exc!r}")

    # -- internals ---------------------------------------------------------- #

    def _write(self, record: LatencyRecord) -> None:
        """Serialize + buffer one record.  Never raises."""

        with self._lock:
            if not self._enabled or self._closed:
                return
            try:
                payload = record._payload(role=self._role, seq=self._seq)
                # ``default=repr`` keeps a stray non-serializable value (an
                # enum, a numpy scalar from a caller's ``set``) from killing the
                # whole sink -- it degrades that one field to a string instead.
                line = json.dumps(payload, default=repr, separators=(",", ":"))
                stream = self._stream
                if stream is None:
                    stream = self._open_locked()
                stream.write(line + "\n")
                self._seq += 1
                self._pending += 1
                now = time.monotonic()
                if (
                    self._pending >= FLUSH_EVERY_N_COMMITS
                    or (now - self._last_flush) >= FLUSH_INTERVAL_S
                ):
                    stream.flush()
                    self._pending = 0
                    self._last_flush = now
            except Exception as exc:
                self._degrade_locked(exc)

    def _open_locked(self):
        # Lazy, and append-only: two processes pointed at the same explicit path
        # interleave rather than truncate each other's session.
        path = self._path
        parent = path.parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(path, "a", encoding="utf-8")
        return self._stream

    def _degrade_locked(self, exc: BaseException) -> None:
        self._degraded = True
        self._enabled = False
        stream = self._stream
        self._stream = None
        self._pending = 0
        self._warn_locked(
            f"disabling latency profiling for role={self._role!r} "
            f"path={self._path}: {exc!r}"
        )
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def _warn_locked(self, message: str) -> None:
        # Exactly one warning per profiler, whatever fails and however often:
        # this runs at loop rate, and a repeating warning would bury the log it
        # is trying to make visible.
        if self._warned:
            return
        self._warned = True
        _LOGGER.warning("%s", message)
