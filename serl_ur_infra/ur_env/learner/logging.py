"""Synchronous JSONL plus W&B logging for learner-only metrics and events."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import numpy as np


class LearnerLoggingError(RuntimeError):
    """Structured learner logging could not safely persist an event."""


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
    """Write every record to JSONL and mirror it to one W&B run."""

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
    ) -> None:
        if wandb_mode not in {"offline", "online", "disabled"}:
            raise ValueError("wandb_mode must be offline, online, or disabled")
        self.path = Path(jsonl_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False
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
                raise LearnerLoggingError("logger is closed")
            try:
                self._stream.write(payload + "\n")
                self._stream.flush()
                if self._wandb_run is not None:
                    self._wandb_run.log(record, step=learner_step)
            except Exception as exc:
                raise LearnerLoggingError(
                    f"learner log write failed: {type(exc).__name__}: {exc}"
                ) from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._wandb_run is not None:
                    self._wandb_run.finish()
            finally:
                self._stream.close()

    def __enter__(self) -> "JsonlWandbLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
