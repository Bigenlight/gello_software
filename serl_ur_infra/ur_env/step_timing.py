"""Opt-in per-step latency instrumentation for the policy-serving entrypoints.

WHAT THIS IS
------------
``run_bc_policy_server.py`` and ``run_fm_policy_server.py`` already report one
number about latency: ``server_inference_ms``, the policy call measured inside
``ActorSessionService._infer``.  That number does not explain a slow Step RPC.
Between the actor's request and its reply the server also validates the
command, primes/encodes the observation, finalizes the transition and writes it
to the recording sink -- and on this stack the sink is a pickle-writing disk
sink, not an in-memory buffer.  This module measures the whole handler and the
sink separately, so ``handler - inference - sink`` names the remainder instead
of leaving it inside "the server was slow".

It writes ``timing.jsonl`` next to ``inference.jsonl`` in the run's record root:
one object per Step RPC, one per BeginEpisode.  The row count is the RPC count,
including deduplicated replays of a Step the actor retried -- those are exactly
the rows an operator investigating a stall wants, and they appear nowhere else.

DEFAULT OFF -- LOAD-BEARING
---------------------------
Nothing here is installed unless ``--step-timing`` or ``HIL_STEP_TIMING=1`` says
so.  A disabled server constructs none of these objects, creates no file and
prints the same ready line as before.  The instrumentation is a diagnostic for a
session someone is actively investigating; a robot evaluation must not change
shape because a measurement exists.

FAIL-OPEN -- ALSO LOAD-BEARING
------------------------------
These objects sit in the request path of a live UR7e session, exactly like
``bc_inference_log.InferenceLoggingPolicy``, and follow its rules: every write
is wrapped, failures are counted in ``error_count`` and reported once on stderr,
and the constructor touches no filesystem so a bad path can never turn into a
server that refuses to start.  A full disk must cost a log line, not an episode
with the arm mid-reach.

The wrapped calls are re-raised unchanged and the wrapped return values are
returned unchanged: this module observes, it never repairs.  A raising Step
still emits its record (with ``error`` set) before the exception continues.

IMPORTS
-------
Stdlib only, on purpose.  The record contains no arrays, so nothing here needs
numpy/jax/grpc, and the module stays importable from a plain venv for offline
analysis of a ``timing.jsonl`` produced elsewhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

#: Environment opt-in, honoured by both policy-serving entrypoints.
STEP_TIMING_ENV = "HIL_STEP_TIMING"

#: File written under the server's ``record_root``, beside ``inference.jsonl``.
SERVER_TIMING_FILENAME = "timing.jsonl"

#: Prefix for the single stderr warning this module is allowed to print.
_LOG_PREFIX = "[step-timing]"

_TRUE_VALUES = frozenset({"1", "true", "yes"})


def step_timing_enabled(value: Any) -> bool:
    """True for ``1``/``true``/``yes``; anything else -- including None -- is off."""

    if value is None:
        return False
    try:
        return str(value).strip().lower() in _TRUE_VALUES
    except Exception:  # noqa: BLE001 - an unprintable value is simply not "on"
        return False


class FailOpenJsonlWriter:
    """Append JSON lines to a path, swallowing every I/O failure.

    Lazy by construction: the parent directories and the file itself appear at
    the first successful write.  Append mode throughout, so a re-run against an
    existing path adds to the record rather than destroying it.
    """

    def __init__(self, path: os.PathLike[str] | str) -> None:
        self._path = Path(os.path.expanduser(os.fspath(path)))
        self._lock = threading.Lock()
        self._stream: Any = None
        self._warned = False
        #: Lines actually on disk.
        self.write_count = 0
        #: Writes lost to a swallowed failure.  Non-zero means this file is an
        #: incomplete record of the session.
        self.error_count = 0

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: Any) -> bool:
        """Serialize and append one record.  Returns False if it was lost."""

        try:
            line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        except Exception as exc:  # noqa: BLE001 - see module docstring
            self._note_failure(exc)
            return False
        try:
            with self._lock:
                if self._stream is None:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    self._stream = open(self._path, "a", encoding="utf-8")
                self._stream.write(line + "\n")
                # Flushed per line so the file is tail-able during a session
                # and survives a hard kill of the server.
                self._stream.flush()
                self.write_count += 1
            return True
        except Exception as exc:  # noqa: BLE001
            self._note_failure(exc)
            return False

    def close(self) -> None:
        """Release the stream.  Idempotent, and never raises."""

        try:
            with self._lock:
                stream, self._stream = self._stream, None
            if stream is not None:
                stream.close()
        except Exception as exc:  # noqa: BLE001
            self._note_failure(exc)

    def _note_failure(self, exc: BaseException) -> None:
        try:
            with self._lock:
                self.error_count += 1
                first = not self._warned
                self._warned = True
            if first:
                print(
                    f"{_LOG_PREFIX} WARNING step timing lost a record and will "
                    f"stay quiet about further losses: {type(exc).__name__}: "
                    f"{exc} (path={self._path}); serving continues unaffected",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception:  # noqa: BLE001
            # Last line of defence: even the bookkeeping is expendable next to
            # the guarantee that serving does not raise from this file.
            pass


class StepTimingRecorder:
    """Collect the spans of one in-flight request and emit them as one row.

    The spans are gathered on the handler's own thread: gRPC serves Step from a
    small thread pool, and the sink runs inside ``ActorSessionService.step`` on
    that same thread, so a ``threading.local`` scope attributes every span to
    the request that produced it without a lock and without cross-talk between
    concurrent handlers.
    """

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._local = threading.local()

    def begin(self) -> None:
        """Open a fresh scope for this thread, discarding any stale one."""

        self._local.scope = {}

    def note(self, key: str, ms: float) -> None:
        """Record one span.  A no-op when no scope is open; never raises."""

        try:
            scope = getattr(self._local, "scope", None)
            if scope is None:
                return
            scope[str(key)] = float(ms)
        except Exception:  # noqa: BLE001
            pass

    def notes(self) -> dict:
        """The current scope's spans, as a copy.  Empty when none is open."""

        try:
            return dict(getattr(self._local, "scope", None) or {})
        except Exception:  # noqa: BLE001
            return {}

    def emit(self, record: dict) -> bool:
        """Write ``record``, filled in from the scope, and close the scope."""

        try:
            scope = self.notes()
            self._local.scope = None
            merged = dict(record)
            # The caller wins wherever it stated a value: it resolved the same
            # spans into the frozen schema (and derived overhead_ms from them).
            for key, value in scope.items():
                if merged.get(key) is None:
                    merged[key] = value
        except Exception:  # noqa: BLE001
            merged = record
        try:
            return bool(self._writer.write(merged))
        except Exception:  # noqa: BLE001
            return False


class TimingSink:
    """Wrap an ``accept_data`` callable and time it into the current scope."""

    def __init__(self, inner: Any, recorder: Any) -> None:
        if callable(getattr(inner, "prime_observation", None)):
            raise ValueError(
                "TimingSink refuses to wrap a sink that defines "
                "prime_observation: ActorSessionService._prime_replay_"
                "observation finds it with getattr(sink, 'prime_observation', "
                "None), and this wrapper deliberately defines no such "
                "attribute, so wrapping would silently disable feature priming "
                "instead of failing"
            )
        self._inner = inner
        self._recorder = recorder

    @property
    def inner(self) -> Any:
        return self._inner

    def __call__(self, data: Any, intervened: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._inner(data, intervened)
        finally:
            # In a finally so a raising sink is still measured; note() is
            # itself total, so it cannot replace the exception in flight.
            self._recorder.note(
                "sink_ms", (time.perf_counter() - started) * 1000.0
            )


class ServiceStepTimingProxy:
    """Time the two request-path methods of an ``ActorSessionService``.

    Implements exactly the five methods ``GrpcActorServicer`` calls, and no
    ``__getattr__``: a servicer that grew a sixth call must fail loudly here
    rather than reach the service untimed through a catch-all.  ``health`` /
    ``get_server_info`` / ``get_buffer_status`` pass straight through -- they
    are cheap, unbatched and not on the control loop.
    """

    def __init__(self, service: Any, recorder: Any) -> None:
        self._service = service
        self._recorder = recorder

    @property
    def service(self) -> Any:
        return self._service

    def health(self):
        return self._service.health()

    def get_server_info(self):
        return self._service.get_server_info()

    def get_buffer_status(self):
        return self._service.get_buffer_status()

    def begin_episode(self, command):
        self._recorder.begin()
        started = time.perf_counter()
        error = None
        try:
            return self._service.begin_episode(command)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self._emit_begin_episode(
                command, (time.perf_counter() - started) * 1000.0, error
            )

    def step(self, command):
        self._recorder.begin()
        started = time.perf_counter()
        result = None
        error = None
        try:
            result = self._service.step(command)
            return result
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self._emit_step(
                command, result, (time.perf_counter() - started) * 1000.0, error
            )

    def _emit_begin_episode(
        self, command: Any, handler_ms: float, error: str | None
    ) -> None:
        try:
            record = {
                "ts": time.time(),
                "kind": "begin_episode",
                "run_id": _as_text(getattr(command, "run_id", None)),
                "episode_id": None,
                "step_id": None,
                "env_step": None,
                "transition_id": None,
                "handler_ms": handler_ms,
                "infer_ms": None,
                "sink_ms": None,
                # No sink and no separately attributed inference on this path,
                # so the whole handler is overhead by definition.
                "overhead_ms": handler_ms,
                "terminal": None,
                "deduplicated": None,
                "error": error,
            }
            self._recorder.emit(record)
        except Exception:  # noqa: BLE001 - see module docstring
            pass

    def _emit_step(
        self, command: Any, result: Any, handler_ms: float, error: str | None
    ) -> None:
        try:
            sink_ms = self._recorder.notes().get("sink_ms")
            action = getattr(result, "action", None)
            infer_ms = _as_float(getattr(action, "server_inference_ms", None))
            terminal = _as_bool(
                getattr(getattr(result, "outcome", None), "terminal", None)
            )
            deduplicated = _as_bool(
                getattr(getattr(result, "ack", None), "deduplicated", None)
            )
            meta = _sub_mapping(getattr(command, "data", None), "meta")
            transition = _sub_mapping(getattr(command, "data", None), "transition")
            record = {
                "ts": time.time(),
                "kind": "step",
                "run_id": _as_text(getattr(command, "run_id", None)),
                "episode_id": _as_int(_lookup(transition, "episode_id")),
                "step_id": _as_int(_lookup(transition, "step_id")),
                "env_step": _as_int(_lookup(meta, "env_step")),
                "transition_id": _as_text(_lookup(meta, "transition_id")),
                "handler_ms": handler_ms,
                "infer_ms": infer_ms,
                "sink_ms": sink_ms,
                "overhead_ms": handler_ms - (infer_ms or 0.0) - (sink_ms or 0.0),
                "terminal": terminal,
                "deduplicated": deduplicated,
                "error": error,
            }
            self._recorder.emit(record)
        except Exception:  # noqa: BLE001
            pass


def _sub_mapping(data: Any, key: str) -> Any:
    get = getattr(data, "get", None)
    if not callable(get):
        return None
    try:
        return get(key, None)
    except Exception:  # noqa: BLE001
        return None


def _lookup(mapping: Any, key: str) -> Any:
    get = getattr(mapping, "get", None)
    if not callable(get):
        return None
    try:
        return get(key, None)
    except Exception:  # noqa: BLE001
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(value)
    except Exception:  # noqa: BLE001
        return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return None
