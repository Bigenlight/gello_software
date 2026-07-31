"""Per-inference JSONL log for a BC rollout, as a transparent policy wrapper.

WHAT THIS IS
------------
``run_bc_policy_server.py`` hands one callable to ``ActorSessionService`` as
``sample_action``: a :class:`ur_env.learner.policy.VersionedPolicyRuntime`,
invoked as ``policy(observation, deterministic) -> (np.ndarray(7,), int)``
(``ur_env/actor_network.py:1505``).  :class:`InferenceLoggingPolicy` wraps that
runtime and writes one compact JSON object per call, so a BC evaluation can be
audited afterwards at the granularity the recording sink cannot reach.

The sink (``ur_env/bc_recording_sink.py``) records *accepted transitions* --
what the actor decided to send.  This wrapper records *inferences* -- what the
server was asked and what it answered, with the server-side latency of the
answer.  The two files do not have the same row count and are not meant to:
cached Step replies, primed/warm-up calls and any inference the actor discarded
appear here and nowhere else.  That difference is the reason this file exists.

WHY LOGGING IS FAIL-OPEN -- LOAD-BEARING
----------------------------------------
This object sits directly in the inference path of a live UR7e session.  An
exception raised here does not merely lose a log line: it propagates into
``ActorSessionService._infer``, which converts *any* exception from the policy
callback into ``PolicyInferenceError`` -- the actor then sees a failed Step and
the episode dies with a robot mid-reach.  A full disk, a read-only mount or a
typo'd path must never be able to do that.  So every part of the logging block
is caught, counted in ``log_error_count`` and reported exactly once on stderr.

The same reasoning is why the constructor touches no filesystem: a wrapper that
refused to build, or that created directories, would turn a logging concern
into a startup concern.  A bad ``log_path`` surfaces as the one-time warning at
the first inference, never as a session that cannot start.

The wrapped result is returned **unchanged and unwrapped** -- the same tuple
object, with the same array identity.  Downstream validation
(``validate_action`` / ``validate_counter``) must see exactly what the runtime
produced, so that a policy bug is never masked or mutated by an observer.

WHY NO IMAGES
-------------
The observation carries the policy's image tensors (two camera views).  They
are never logged: a single 128x128x3 uint8 frame is ~49 KB, so two per row at
BC rollout rates would produce hundreds of MB per session in a file that is
meant to stay ``tail -f``-able, and the pixels are already recoverable from the
episode pickles the recording sink writes.  Only the 19-D proprioceptive state
is logged, and only when it is exactly the canonical width.

WHY THERE IS NO ``prime_observation`` HERE
------------------------------------------
``_prime_replay_observation`` (``actor_network.py:1476``) probes the *sink* for
that attribute, and ``bc_recording_sink`` deliberately does not define it so
the BC agent keeps receiving raw pixels.  This wrapper must not introduce one
either -- ``__getattr__`` delegates to the wrapped runtime, which has no such
attribute, so the probe keeps falling through to ``None``.  Do not add it.

IMPORTS
-------
Stdlib and NumPy only.  This module is imported by the actor-side venv
(``/home/laptop3/venvs/gello-hil-actor/bin/python``), which has no jax.  The
action arriving here has already been through ``jax.device_get`` inside
``VersionedPolicyRuntime._validated_policy_action``, so ``np.asarray`` on it is
a cheap host-side view and never needs the accelerator libraries.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np


#: Prefix for the single stderr warning this module is allowed to print.
_LOG_PREFIX = "[bc-inference-log]"

#: Canonical proprioceptive width; anything else is logged as ``null`` rather
#: than as a differently shaped row, so a consumer can trust the column.
STATE_DIM = 19


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


class InferenceLoggingPolicy:
    """Wrap a policy callable and append one JSON line per inference.

    Thread-safe: gRPC serves Step from several handler threads, so the counters
    and the log stream are guarded by a single lock.  The wrapped policy is
    called *outside* that lock -- serialising inference behind a logging mutex
    would add the file's latency to the control loop, and the runtime already
    has its own lock around the snapshot it needs.

    Attribute access falls through to the wrapped policy (``model_id``,
    ``policy_version``, ``learner_step``, ``snapshot``, ``publish``, ...), so
    the wrapper is a drop-in for anything that reads the runtime.

    NOTE: this class must never grow a ``prime_observation`` attribute -- see
    the module docstring.
    """

    def __init__(
        self,
        policy: Any,
        log_path: os.PathLike[str] | str,
        *,
        log_state: bool = True,
    ) -> None:
        # Assigned first: ``__getattr__`` reads ``_policy`` and would recurse
        # forever if an attribute were looked up before this line ran.
        self._policy = policy
        self._log_path = Path(os.path.expanduser(os.fspath(log_path)))
        self._log_state = bool(log_state)
        self._lock = threading.Lock()
        self._stream: Any = None
        self._warned = False
        #: Inferences that passed through the wrapper, whether or not the log
        #: line for them was written.
        self.call_count = 0
        #: Logging failures swallowed so far.  Non-zero means this file is an
        #: incomplete record of the session.
        self.log_error_count = 0

    @property
    def log_path(self) -> Path:
        return self._log_path

    def __call__(self, observation: Any, deterministic: Any) -> tuple:
        t0 = time.perf_counter()
        result = self._policy(observation, deterministic)
        # Measured around the wrapped call only: the log write is this
        # object's own cost and must not be attributed to the policy.
        latency_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self.call_count += 1

        try:
            self._append(result, latency_ms, observation, deterministic)
        except Exception as exc:  # noqa: BLE001 - see module docstring
            self._note_log_failure(exc)

        # Same object, not a copy: downstream validation must see exactly what
        # the runtime returned.
        return result

    def _append(
        self,
        result: Any,
        latency_ms: float,
        observation: Any,
        deterministic: Any,
    ) -> None:
        """Build and write one line.  Any exception here is caught by the caller."""

        action, policy_version = result
        record = {
            "ts": _utc_now_text(),
            "latency_ms": float(latency_ms),
            "deterministic": bool(deterministic),
            "policy_version": int(policy_version),
            "action": self._action_floats(action),
            "state": self._state_floats(observation),
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self._stream is None:
                # Lazy and append-only: the file is opened at the first
                # inference, never truncated, and a re-wrapped policy adds to
                # the same log instead of destroying it.
                self._stream = open(self._log_path, "a", encoding="utf-8")
            self._stream.write(line + "\n")
            # Flushed per line so the file is tail-able during a session and
            # survives a hard kill of the server.
            self._stream.flush()

    @staticmethod
    def _action_floats(value: Any) -> list[float]:
        """Flatten the action to plain floats.

        Deliberately does not assert the 7-D contract: a malformed action is
        the actor's error to raise, and a logger that rejected it would hide
        the very row an operator needs to see it.
        """

        return [float(item) for item in np.asarray(value).ravel()]

    def _state_floats(self, observation: Any) -> list[float] | None:
        """The 19-D proprioceptive vector, or ``None``.

        ``None`` covers every "not exactly the canonical state" case -- logging
        disabled, key absent, non-mapping observation, wrong width -- so the
        column is either a 19-float row or nothing.  Image keys are never read.
        """

        if not self._log_state:
            return None
        get = getattr(observation, "get", None)
        if not callable(get):
            return None
        value = get("state", None)
        if value is None:
            return None
        # ``state`` arrives as the canonical (1, 19) float32 batch; ravel()
        # makes the row flat without caring whether it was batched.
        array = np.asarray(value).ravel()
        if array.size != STATE_DIM:
            return None
        return [float(item) for item in array]

    def _note_log_failure(self, exc: BaseException) -> None:
        """Count the failure and warn once.  Must not raise."""

        try:
            with self._lock:
                self.log_error_count += 1
                first = not self._warned
                self._warned = True
            if first:
                print(
                    f"{_LOG_PREFIX} WARNING inference logging disabled for this "
                    f"session's remaining failures: {type(exc).__name__}: {exc} "
                    f"(path={self._log_path}); inference continues unaffected",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception:  # noqa: BLE001
            # Last line of defence.  Even the bookkeeping is expendable next to
            # the guarantee that inference does not raise from this file.
            pass

    def close(self) -> None:
        """Release the log stream.  Idempotent, and never raises."""

        try:
            with self._lock:
                stream, self._stream = self._stream, None
            if stream is not None:
                stream.close()
        except Exception as exc:  # noqa: BLE001
            self._note_log_failure(exc)

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal lookup failed, so wrapper state and methods
        # always win over the wrapped object's.
        if name.startswith("__") and name.endswith("__"):
            # Protocol hooks (copy/pickle/etc.) stay unimplemented rather than
            # being answered by the policy on the wrapper's behalf.
            raise AttributeError(name)
        try:
            policy = object.__getattribute__(self, "_policy")
        except AttributeError:  # pragma: no cover - only during a failed init
            raise AttributeError(name) from None
        return getattr(policy, name)
