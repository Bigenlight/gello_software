"""Synchronous JSONL plus W&B logging for learner-only metrics and events."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

# Cadence, in consecutive failures, at which a dead sink re-announces itself.
# The learner logs once per learner step, so a streak is minutes of training
# with no metrics; one line per hundred keeps that visible in stdout.log
# without burying the events that matter.
LOGGING_WARN_STREAK = 100


class LearnerLoggingError(RuntimeError):
    """Structured learner logging could not safely persist an event."""


def _emit_logging_warning(message: str) -> None:
    """Write one operator-visible line to stderr.

    Deliberately not the ``logging`` module, for the same reason as
    ``rlpd_receive_server._emit_operator_warning``: nothing in ``ur_env``
    configures logging, so the warning about a silent failure would itself be
    silent.  stderr is what the production launcher redirects into the run's
    ``logs/stdout.log``.
    """

    print(f"[learner-logging] WARNING: {message}", file=sys.stderr, flush=True)


def _json_value(value: Any, *, path: str = "value") -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        if array.shape == ():
            return _json_value(array.item(), path=path)
        if array.size > 32:
            raise LearnerLoggingError(
                f"{path} is a large tensor; raw learner data must not enter logs"
            )
        return _json_value(array.tolist(), path=path)
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LearnerLoggingError(f"{path} must be finite")
        return value
    raise LearnerLoggingError(
        f"{path} has unsupported log value type {type(value).__name__}"
    )


def _wandb_text(run: Any, attribute: str) -> str | None:
    """Read one string attribute off a W&B run without ever raising.

    Run identity is diagnostics, not learner state: a W&B client that renames,
    removes or lazily computes one of these attributes (and raises while doing
    it) must not be able to take the learner down on the way up.
    """

    try:
        value = getattr(run, attribute, None)
    except Exception:  # pragma: no cover - defensive against client changes
        return None
    if isinstance(value, str) and value:
        return value
    return None


def _run_identity(run: Any, *, directory: Path) -> dict[str, Any]:
    """Describe a live W&B run well enough for an operator to open it."""

    identity: dict[str, Any] = {"dir": str(directory)}
    for attribute in ("id", "name", "entity", "project", "url"):
        identity[attribute] = _wandb_text(run, attribute)
    parts = (identity["entity"], identity["project"], identity["id"])
    identity["run_path"] = "/".join(parts) if all(parts) else None
    return identity


class JsonlWandbLogger:
    """Write every record to JSONL and mirror it to one W&B run.

    A DEAD SINK DEGRADES THIS LOGGER, IT DOES NOT STOP TRAINING
    ----------------------------------------------------------
    ``HILSERLLearner.train_once`` calls :meth:`log` INSIDE the try block whose
    ``except Exception`` latches the permanent learner fault, and
    ``LearnerWorker`` ends its thread on that fault.  So until this class
    stopped raising for sink failures, one ``wandb.log`` that hit a closed
    socket ended training for the life of the process -- while the gRPC service
    stayed ``ready`` and kept serving the last published policy, which is the
    worst shape of failure: the actor keeps driving the robot and collecting
    transitions that nothing will ever learn from.  It was not hypothetical:
    the 2026-07-31 15:47 learner logged ``learner log write failed:
    ConnectionResetError: Connection lost`` when the W&B service process died,
    and that was in ``offline`` mode, where ``wandb.log`` still blocks on a
    local socket to the ``wandb-core`` service (``InterfaceSock._publish`` ->
    ``AsyncioManager.run`` -> ``StreamWriter.drain``).  ``online`` mode does
    not add a network round trip to :meth:`log`, but it does give that service
    process far more ways to die or stall.

    Nothing about training correctness depends on a dashboard, so a sink
    failure is counted, warned about, and survived:

    * the JSONL sink and the W&B sink fail independently -- a full disk must
      not cost the dashboard, and a dead W&B service must not cost the run's
      own event trail;
    * the W&B mirror MUTES itself after its first failure, because
      ``ServiceClient`` latches the broken connection and re-raises it forever;
      retrying every step would only add an exception per learner step; and
    * ``jsonl_fault_count`` / ``wandb_fault_count`` / :attr:`degraded_detail`
      make the degradation readable, and the first failure of a streak plus
      every ``LOGGING_WARN_STREAK``-th one after it print by name.

    WHAT IS STILL FATAL, AND WHY
    ----------------------------
    Everything :func:`_json_value` rejects: a non-finite metric, a raw tensor,
    an unsupported type, a missing ``event``, a non-integer ``learner_step``.
    Those are not statements about the sink being unreachable, they are
    statements about the record being wrong, which means the learner produced
    something it should not have.  Degrading them would convert an upstream bug
    into a missing line in a log nobody reads.  In practice the learner already
    catches the important one earlier and harder: ``_flatten_scalars`` raises
    ``LearnerFaultError`` on a non-finite update metric before it can ever be
    handed to this class.
    """

    def __init__(
        self,
        jsonl_path: os.PathLike[str] | str,
        *,
        wandb_mode: str = "offline",
        wandb_dir: os.PathLike[str] | str | None = None,
        project: str = "hil-serl",
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        enable_wandb: bool = True,
        wandb_module: Any | None = None,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        if wandb_mode not in {"offline", "online", "disabled"}:
            raise ValueError("wandb_mode must be offline, online, or disabled")
        self.path = Path(jsonl_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(self.path, "a", encoding="utf-8")
        # Reentrant because the fault notes below run under the lock and call
        # an injected `warn`, which is free to read the counter properties --
        # a plain Lock would turn a warning handler into a deadlock.
        self._lock = threading.RLock()
        self._closed = False
        self._warn = warn or _emit_logging_warning
        self._jsonl_fault_count = 0
        self._jsonl_fault_streak = 0
        self._last_jsonl_fault = ""
        self._wandb_fault_count = 0
        self._last_wandb_fault = ""
        self._wandb_muted = False
        self._wandb_run = None
        self._wandb_metadata: dict[str, Any] = {
            "mode": wandb_mode,
            "dir": None,
            "id": None,
            "name": None,
            "entity": None,
            "project": None,
            "url": None,
            "run_path": None,
        }
        if enable_wandb and wandb_mode != "disabled":
            try:
                wandb = wandb_module
                if wandb is None:
                    import wandb as imported_wandb

                    wandb = imported_wandb
                # W&B writes its run directory under `dir` in EVERY mode.
                # "online" adds a sync worker; it does not move the files, so
                # the run root stays self-contained and `wandb sync` remains
                # available as the fallback if the network drops.
                directory = Path(wandb_dir or self.path.parent).expanduser().resolve()
                directory.mkdir(parents=True, exist_ok=True)
                self._wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    mode=wandb_mode,
                    dir=str(directory),
                    config=_json_value(config or {}, path="config"),
                    reinit=True,
                )
                self._wandb_metadata.update(
                    _run_identity(self._wandb_run, directory=directory)
                )
            except Exception as exc:
                self._stream.close()
                raise LearnerLoggingError(
                    f"W&B initialization failed: {type(exc).__name__}: {exc}"
                ) from exc

    @property
    def wandb_run(self) -> Any | None:
        return self._wandb_run

    @property
    def wandb_metadata(self) -> dict[str, Any]:
        """Return how an operator reaches this run.

        The production learner runs under ``WANDB_SILENT=true``, which
        suppresses the banner W&B normally prints with the run URL, and it is
        detached on a remote host where nobody watches stdout anyway.  Online
        logging is only useful if the URL is discoverable, so the identity is
        carried in structured output instead of a terminal banner.  Every field
        except ``mode`` is ``None`` when W&B is disabled, and ``url`` is
        additionally ``None`` for offline runs, which have no server-side page.
        """

        return dict(self._wandb_metadata)

    def log(self, event: str, *, learner_step: int, **fields: Any) -> None:
        if not isinstance(event, str) or not event:
            raise ValueError("event is required")
        if isinstance(learner_step, bool) or not isinstance(learner_step, int):
            raise ValueError("learner_step must be an integer")
        record = _json_value(
            {
                "event": event,
                "time_ns": time.time_ns(),
                "learner_step": learner_step,
                **fields,
            },
            path="record",
        )
        payload = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            if self._closed:
                # Not a sink failure: whoever holds a closed logger is calling
                # it out of order, and that is a caller bug worth raising for.
                raise LearnerLoggingError("logger is closed")
            self._write_jsonl(payload)
            self._mirror_to_wandb(record, learner_step)

    def _write_jsonl(self, payload: str) -> None:
        """Append one line, surviving a sink that cannot take it.

        Caller holds ``self._lock``.
        """

        try:
            self._stream.write(payload + "\n")
            self._stream.flush()
        except Exception as exc:
            self._note_jsonl_fault(exc)
        else:
            self._jsonl_fault_streak = 0

    def _mirror_to_wandb(self, record: Mapping[str, Any], learner_step: int) -> None:
        """Mirror one already-written record, surviving a dead W&B service.

        Caller holds ``self._lock``.
        """

        if self._wandb_run is None or self._wandb_muted:
            return
        try:
            self._wandb_run.log(record, step=learner_step)
        except Exception as exc:
            self._note_wandb_fault(exc, phase="log")

    def _note_jsonl_fault(self, exc: BaseException) -> None:
        detail = f"{type(exc).__name__}: {exc}"[:2_000]
        self._jsonl_fault_count += 1
        self._jsonl_fault_streak += 1
        self._last_jsonl_fault = detail
        if (
            self._jsonl_fault_streak == 1
            or self._jsonl_fault_streak % LOGGING_WARN_STREAK == 0
        ):
            self._warn(
                f"JSONL log write failed ({self._jsonl_fault_streak} in a row, "
                f"{self._jsonl_fault_count} total); training continues and the "
                f"events for those steps are LOST: {detail} [{self.path}]"
            )

    def _note_wandb_fault(self, exc: BaseException, *, phase: str) -> None:
        detail = f"{type(exc).__name__}: {exc}"[:2_000]
        self._wandb_fault_count += 1
        self._last_wandb_fault = detail
        # One warning, because the mirror is muted from here on.  The W&B
        # client latches its broken connection and re-raises it on every
        # subsequent call, so a streak would be one identical line per learner
        # step describing a mirror that already stopped mirroring.
        self._wandb_muted = True
        self._warn(
            f"W&B mirror failed during {phase} and is now MUTED for the rest "
            f"of this run; training continues and the JSONL log is unaffected: "
            f"{detail}"
        )

    @property
    def jsonl_fault_count(self) -> int:
        """Records lost because the JSONL sink refused them."""
        with self._lock:
            return self._jsonl_fault_count

    @property
    def last_jsonl_fault(self) -> str:
        with self._lock:
            return self._last_jsonl_fault

    @property
    def wandb_fault_count(self) -> int:
        """W&B calls that failed; at most one ``log`` fault, plus ``finish``."""
        with self._lock:
            return self._wandb_fault_count

    @property
    def last_wandb_fault(self) -> str:
        with self._lock:
            return self._last_wandb_fault

    @property
    def wandb_muted(self) -> bool:
        """True once the W&B mirror stopped being attempted."""
        with self._lock:
            return self._wandb_muted

    @property
    def degraded(self) -> bool:
        with self._lock:
            return bool(self._jsonl_fault_count or self._wandb_fault_count)

    @property
    def degraded_detail(self) -> str:
        """One operator-readable line, or "" while both sinks are healthy."""

        with self._lock:
            parts = []
            if self._jsonl_fault_count:
                parts.append(
                    f"JSONL DEGRADED: {self._jsonl_fault_count} lost record(s) "
                    f"(last: {self._last_jsonl_fault})"
                )
            if self._wandb_fault_count:
                parts.append(
                    f"W&B {'MUTED' if self._wandb_muted else 'DEGRADED'}: "
                    f"{self._wandb_fault_count} failure(s) "
                    f"(last: {self._last_wandb_fault})"
                )
            return "; ".join(parts)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._wandb_run is not None:
                    self._wandb_run.finish()
            except Exception as exc:
                # A dashboard that cannot be torn down is not a failed run.
                # This used to escape into the launcher's `finally`, which
                # raised the process exit code to 5 -- so the 2026-07-31 reboot
                # turned a learner that had just recorded exit_code 0 into a
                # failed one, purely because W&B's service had already gone.
                self._note_wandb_fault(exc, phase="finish")
            finally:
                self._stream.close()

    def __enter__(self) -> "JsonlWandbLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
