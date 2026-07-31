"""Rescue learner state when a run ends away from a checkpoint boundary.

Production runs end abruptly far more often than they end on a boundary.  A
UR7e protective stop kills the local actor, the operator stops the learner,
and everything the run accumulated is gone: ``run_rlpd_learner_server`` frames
its own contract as ``learner_checkpoints_only_replay_ram``, and with
``checkpoint_period`` at 5,000 outer steps the observed sessions (learner step
301 on kanu, 68 on the migrated host) never reached a single periodic save.
Eight kanu run roots held an empty ``checkpoints/``.

This module is the write half of the fix.  It runs once, on the way down, and
persists two things:

``emergency_<step>/``
    A full agent checkpoint through the normal ``CheckpointManager`` -- same
    msgpack payload, same checksum, same completion marker, and therefore the
    same Adam moments, target params, and RNG that a periodic checkpoint
    carries.  Only the directory prefix differs, which keeps it out of
    ``latest_path()`` and so out of ``--resume-latest``.  A rescue is taken at
    whatever step the run died on, which is almost never a
    checkpoint/publish boundary, and ``prepare_learner_state`` rejects such a
    step as corrupt.  Restoring one is therefore a deliberate act, not
    something that can happen to an operator by accident.

``replay_<step>.npz``
    The filled rows of both feature rings, oldest first.  The rings are
    preallocated, so this writes what the run collected rather than the 7.3
    GiB the production 50k/10k pair reserves.

Nothing here may raise into the shutdown path.  A learner that dies while
trying to save its own corpse is strictly worse than one that reports the
failure and exits: the operator still has the process output, and the caller
still gets to run its normal teardown.  ``rescue_learner_state`` therefore
converts every failure into a result record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np

from ur_env.learner.checkpoint import CheckpointManager, LearnerFingerprint


REPLAY_SNAPSHOT_FORMAT_VERSION = 1

#: Ring identities written into one snapshot.  The learner samples these two
#: independently for the RLPD 50:50 split, so a dump that merged them would be
#: unrestorable.
SNAPSHOT_RINGS = ("replay", "intervention")


def _npz_payload(name: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one ring snapshot into ``np.savez`` keys."""

    payload: dict[str, Any] = {}
    for group in ("observations", "next_observations"):
        for key, value in snapshot[group].items():
            payload[f"{name}/{group}/{key}"] = value
    for key in ("actions", "rewards", "masks", "grasp_penalty"):
        payload[f"{name}/{key}"] = snapshot[key]
    return payload


def _ring_manifest(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "size": int(snapshot["size"]),
        "capacity": int(snapshot["capacity"]),
        "insert_count": int(snapshot["insert_count"]),
        "overwrite_count": int(snapshot["overwrite_count"]),
        "expected_grasp_penalty": float(snapshot["expected_grasp_penalty"]),
    }


def snapshot_nbytes(snapshot: Mapping[str, Any]) -> int:
    """Return the uncompressed byte size one ring snapshot will occupy."""

    total = 0
    for value in _npz_payload("_", snapshot).values():
        total += int(np.asarray(value).nbytes)
    return total


def save_replay_snapshot(
    path: os.PathLike[str] | str,
    *,
    replay: Mapping[str, Any],
    intervention: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    minimum_free_bytes_after_save: int = 0,
) -> Path:
    """Write both ring snapshots to one ``.npz`` and return its path.

    The file is written to a temporary sibling and renamed, so a truncated
    write is never observable under the final name.
    """

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"replay snapshot already exists and will not be overwritten: "
            f"{destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "format_version": REPLAY_SNAPSHOT_FORMAT_VERSION,
        "created_time_ns": time.time_ns(),
        "rings": {},
        "provenance": dict(provenance or {}),
    }
    for name, snapshot in (("replay", replay), ("intervention", intervention)):
        payload.update(_npz_payload(name, snapshot))
        manifest["rings"][name] = _ring_manifest(snapshot)

    required = sum(
        int(np.asarray(value).nbytes) for value in payload.values()
    ) + minimum_free_bytes_after_save
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < required:
        raise OSError(
            "insufficient space for replay snapshot: "
            f"free={free_bytes}, required={required}, path={destination}"
        )

    payload["manifest.json"] = np.frombuffer(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8"),
        dtype=np.uint8,
    )

    staging = destination.with_name(destination.name + ".partial")
    try:
        with open(staging, "wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, destination)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def read_replay_snapshot_manifest(
    path: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Return the manifest of a snapshot without materializing its tensors."""

    with np.load(Path(path).expanduser().resolve()) as archive:
        raw = bytes(archive["manifest.json"])
    return json.loads(raw.decode("utf-8"))


@dataclass
class RescueResult:
    """What the shutdown rescue managed to persist, and what it did not."""

    attempted: bool = False
    learner_step: int | None = None
    gradient_step: int | None = None
    policy_version: int | None = None
    checkpoint_path: str | None = None
    checkpoint_error: str | None = None
    replay_snapshot_path: str | None = None
    replay_snapshot_error: str | None = None
    replay_rows: int | None = None
    intervention_rows: int | None = None
    skipped_reason: str | None = None

    @property
    def saved_anything(self) -> bool:
        return (
            self.checkpoint_path is not None
            or self.replay_snapshot_path is not None
        )

    def event_fields(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None
        }


_MAX_ERROR_DETAIL = 2_000


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:_MAX_ERROR_DETAIL]}"


def rescue_learner_state(
    *,
    learner: Any,
    ingress: Any,
    checkpoint_manager: CheckpointManager,
    fingerprint: LearnerFingerprint,
    inference_rng: Any,
    provenance: Mapping[str, Any] | None = None,
) -> RescueResult:
    """Persist agent and replay state on the way down; never raise.

    The caller must have already stopped the learner worker.  A rescue taken
    while ``train_once`` is mid-flight would serialize an agent whose optimizer
    state and step counter disagree, which is exactly the corruption the
    checkpoint format's ``gradient_step`` cross-check exists to catch -- it
    would be caught, but only at restore time, long after the state that could
    have been saved is gone.

    The checkpoint and the replay snapshot are attempted independently.  They
    fail for unrelated reasons -- the checkpoint needs JAX to still be usable,
    the snapshot only needs numpy and disk -- so one failing must not suppress
    the other.
    """

    result = RescueResult(attempted=True)

    try:
        result.learner_step = int(learner.learner_step)
        result.gradient_step = int(learner.gradient_step)
        result.policy_version = int(learner.policy_version)
    except Exception as exc:
        result.skipped_reason = f"learner counters unreadable: {_describe(exc)}"
        return result

    if result.learner_step <= 0:
        # Nothing was ever trained.  A zero-step rescue would write a
        # checkpoint identical to a fresh init and a snapshot of whatever
        # arrived before training started, both of which are noise that a
        # later operator has to tell apart from a real rescue.
        result.skipped_reason = "learner never advanced past step 0"
        return result

    periodic = checkpoint_manager.path_for_step(result.learner_step)
    if periodic.exists():
        # A clean exit on a checkpoint boundary already wrote this step
        # through the normal path.  Duplicating it as a rescue would cost a
        # full agent serialization and leave two directories that a later
        # operator has to reconcile.  The replay snapshot below still runs:
        # the periodic path never persists the rings.
        result.checkpoint_error = (
            f"skipped; periodic checkpoint already exists at {periodic}"
        )
    else:
        try:
            result.checkpoint_path = str(
                checkpoint_manager.save(
                    agent=learner.agent,
                    learner_step=result.learner_step,
                    gradient_step=result.gradient_step,
                    policy_version=result.policy_version,
                    inference_rng=inference_rng,
                    fingerprint=fingerprint,
                    emergency=True,
                )
            )
        except Exception as exc:
            result.checkpoint_error = _describe(exc)

    try:
        replay = ingress.replay_store.snapshot()
        intervention = ingress.intervention_store.snapshot()
        result.replay_rows = int(replay["size"])
        result.intervention_rows = int(intervention["size"])
        snapshot_provenance = {
            "learner_step": result.learner_step,
            "gradient_step": result.gradient_step,
            "policy_version": result.policy_version,
            "fingerprint_sha256": fingerprint.sha256,
            "observation_representation": getattr(
                ingress, "observation_representation", None
            ),
            "augmentation": getattr(ingress, "augmentation", None),
            **dict(provenance or {}),
        }
        result.replay_snapshot_path = str(
            save_replay_snapshot(
                checkpoint_manager.root
                / f"replay_{result.learner_step:012d}.npz",
                replay=replay,
                intervention=intervention,
                provenance=snapshot_provenance,
                minimum_free_bytes_after_save=(
                    checkpoint_manager.minimum_free_bytes_after_save
                ),
            )
        )
    except Exception as exc:
        result.replay_snapshot_error = _describe(exc)

    return result
