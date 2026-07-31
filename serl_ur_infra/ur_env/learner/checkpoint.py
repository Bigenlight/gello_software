"""Non-destructive full learner checkpoints with strict resume fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any, Mapping

import numpy as np

from ur_env.learner.agent import verify_resnet10_asset
from ur_env.learner.config import LearnerConfig
from ur_env.learner.policy import validate_tree_finite
from ur_env.observation_schema import (
    CANONICAL_OBSERVATION_SCHEMA_HASH,
    observation_schema_document,
)


CHECKPOINT_FORMAT_VERSION = 1
COMPLETION_MARKER_VERSION = 1
COMPLETION_MARKER_NAME = "completion.json"

#: Periodic checkpoints land on ``checkpoint_<step>``.  Shutdown rescues land
#: on ``emergency_<step>`` instead, and that split is load-bearing: an
#: emergency save happens at whatever step the run died on, which is almost
#: never a checkpoint/publish boundary, so ``prepare_learner_state`` would
#: reject it as corrupt.  ``latest_path()`` scans only the canonical prefix,
#: so a rescue can never become the silent target of ``--resume-latest``.
#: Reading one back is an explicit, operator-visible act.
CHECKPOINT_PREFIX = "checkpoint_"
EMERGENCY_PREFIX = "emergency_"


class CheckpointError(RuntimeError):
    """Base error for learner checkpoint operations."""


class CheckpointExistsError(CheckpointError):
    """Saving would overwrite an existing checkpoint."""


class CheckpointCorruptError(CheckpointError):
    """A checkpoint is incomplete, malformed, or fails its checksum."""


class CheckpointFingerprintError(CheckpointError):
    """A checkpoint belongs to a different learner/schema/asset contract."""


class CheckpointSpaceError(CheckpointError):
    """The checkpoint filesystem cannot preserve the configured reserve."""


class CheckpointLockError(CheckpointError):
    """Another learner process already owns this checkpoint root."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class LearnerFingerprint:
    document: Mapping[str, Any]
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        config: LearnerConfig,
        resnet_asset_path: os.PathLike[str] | str,
        run_contract: Mapping[str, Any] | None = None,
    ) -> "LearnerFingerprint":
        if run_contract is not None and not isinstance(run_contract, Mapping):
            raise TypeError("run_contract must be a mapping")
        resnet_sha256 = verify_resnet10_asset(resnet_asset_path)
        document = {
            "observation_schema_hash": CANONICAL_OBSERVATION_SCHEMA_HASH,
            "observation_schema": observation_schema_document(),
            "learner_config": config.fingerprint_values(),
            "resnet10_sha256": resnet_sha256,
            "run_contract": dict(run_contract or {}),
        }
        encoded = _canonical_json(document)
        # Detach the immutable fingerprint record from caller-owned nested
        # mappings/lists.  Otherwise a later mutation could make ``document``
        # disagree with the digest that was already computed.
        normalized = json.loads(encoded.decode("utf-8"))
        digest = hashlib.sha256(encoded).hexdigest()
        return cls(document=normalized, sha256=digest)


@dataclass(frozen=True)
class RestoredCheckpoint:
    agent: Any
    learner_step: int
    gradient_step: int
    policy_version: int
    inference_rng: Any
    path: Path
    emergency: bool = False


@dataclass(frozen=True)
class _CheckpointContents:
    metadata: Mapping[str, Any]
    state_bytes: bytes
    learner_step: int
    gradient_step: int
    policy_version: int
    inference_rng: np.ndarray
    emergency: bool


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointCorruptError(f"{name} must be a non-negative integer")
    return value


class CheckpointManager:
    """Write immutable ``checkpoint_<learner_step>`` directories."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        minimum_free_bytes_after_save: int = 0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if (
            isinstance(minimum_free_bytes_after_save, bool)
            or not isinstance(minimum_free_bytes_after_save, int)
            or minimum_free_bytes_after_save < 0
        ):
            raise ValueError(
                "minimum_free_bytes_after_save must be a non-negative integer"
            )
        self.minimum_free_bytes_after_save = minimum_free_bytes_after_save

    @staticmethod
    def checkpoint_name(learner_step: int, *, emergency: bool = False) -> str:
        if isinstance(learner_step, bool) or not isinstance(learner_step, int):
            raise ValueError("learner_step must be an integer")
        if learner_step < 0:
            raise ValueError("learner_step must be non-negative")
        if not isinstance(emergency, bool):
            raise TypeError("emergency must be a bool")
        prefix = EMERGENCY_PREFIX if emergency else CHECKPOINT_PREFIX
        return f"{prefix}{learner_step:012d}"

    def path_for_step(
        self, learner_step: int, *, emergency: bool = False
    ) -> Path:
        return self.root / self.checkpoint_name(
            learner_step, emergency=emergency
        )

    def save(
        self,
        *,
        agent: Any,
        learner_step: int,
        gradient_step: int,
        policy_version: int,
        inference_rng: Any,
        fingerprint: LearnerFingerprint,
        emergency: bool = False,
    ) -> Path:
        for name, value in (
            ("learner_step", learner_step),
            ("gradient_step", gradient_step),
            ("policy_version", policy_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(emergency, bool):
            raise TypeError("emergency must be a bool")
        destination = self.path_for_step(learner_step, emergency=emergency)
        if destination.exists() or destination.is_symlink():
            raise CheckpointExistsError(
                f"checkpoint already exists and will not be overwritten: {destination}"
            )

        from flax import serialization
        import jax

        validate_tree_finite(agent.state, name="agent state")
        state_bytes = serialization.to_bytes(agent.state)
        free_bytes = shutil.disk_usage(self.root).free
        required_bytes = (
            len(state_bytes) + self.minimum_free_bytes_after_save
        )
        if free_bytes < required_bytes:
            raise CheckpointSpaceError(
                "insufficient checkpoint filesystem space: "
                f"free={free_bytes}, next_checkpoint={len(state_bytes)}, "
                "required_reserve_after_save="
                f"{self.minimum_free_bytes_after_save}, root={self.root}"
            )
        state_sha256 = hashlib.sha256(state_bytes).hexdigest()
        rng_array = np.asarray(jax.device_get(inference_rng))
        if rng_array.dtype.kind not in "ui" or rng_array.size == 0:
            raise ValueError("inference_rng must be an integer JAX PRNG key")
        metadata = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "created_time_ns": time.time_ns(),
            "learner_step": learner_step,
            "gradient_step": gradient_step,
            "policy_version": policy_version,
            "inference_rng": {
                "dtype": rng_array.dtype.str,
                "shape": list(rng_array.shape),
                "values": rng_array.reshape(-1).tolist(),
            },
            "fingerprint_sha256": fingerprint.sha256,
            "fingerprint": fingerprint.document,
            "agent_state_sha256": state_sha256,
            "emergency": emergency,
        }
        metadata_bytes = _canonical_json(metadata)
        completion = {
            "completion_marker_version": COMPLETION_MARKER_VERSION,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "learner_step": learner_step,
            "agent_state_sha256": state_sha256,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        }
        try:
            destination.mkdir()
            self._write_new_file(destination / "agent_state.msgpack", state_bytes)
            self._write_new_file(destination / "metadata.json", metadata_bytes)
            # This marker is deliberately the final file.  Automatic resume
            # never considers a directory which did not reach this point.
            self._write_new_file(
                destination / COMPLETION_MARKER_NAME,
                _canonical_json(completion),
            )
            self._fsync_directory(destination)
            self._fsync_directory(self.root)
        except FileExistsError as exc:
            raise CheckpointExistsError(
                f"checkpoint already exists and will not be overwritten: {destination}"
            ) from exc
        except Exception as exc:
            raise CheckpointError(
                f"checkpoint write failed; incomplete directory retained at {destination}: {exc}"
            ) from exc
        return destination

    @staticmethod
    def _write_new_file(path: Path, payload: bytes) -> None:
        with open(path, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _read_contents(
        self,
        checkpoint_path: Path,
        *,
        require_completion_marker: bool,
    ) -> _CheckpointContents:
        metadata_path = checkpoint_path / "metadata.json"
        state_path = checkpoint_path / "agent_state.msgpack"
        marker_path = checkpoint_path / COMPLETION_MARKER_NAME
        try:
            metadata_bytes = metadata_path.read_bytes()
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            state_bytes = state_path.read_bytes()
        except Exception as exc:
            raise CheckpointCorruptError(
                f"checkpoint is incomplete or unreadable: {checkpoint_path}"
            ) from exc
        if not isinstance(metadata, dict):
            raise CheckpointCorruptError("checkpoint metadata must be an object")
        if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointCorruptError("unsupported checkpoint format version")

        learner_step = _nonnegative_int(
            metadata.get("learner_step"), name="learner_step"
        )
        gradient_step = _nonnegative_int(
            metadata.get("gradient_step"), name="gradient_step"
        )
        policy_version = _nonnegative_int(
            metadata.get("policy_version"), name="policy_version"
        )
        emergency = metadata.get("emergency", False)
        if not isinstance(emergency, bool):
            raise CheckpointCorruptError("emergency flag must be a boolean")
        # The prefix is derived from the payload, not from the directory, so a
        # rescue cannot be laundered into a periodic checkpoint by renaming it.
        if checkpoint_path.name != self.checkpoint_name(
            learner_step, emergency=emergency
        ):
            raise CheckpointCorruptError(
                "checkpoint directory name does not match learner_step"
            )

        actual_state_sha = hashlib.sha256(state_bytes).hexdigest()
        if metadata.get("agent_state_sha256") != actual_state_sha:
            raise CheckpointCorruptError("agent state checksum mismatch")
        fingerprint_sha256 = metadata.get("fingerprint_sha256")
        if (
            not isinstance(fingerprint_sha256, str)
            or len(fingerprint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in fingerprint_sha256
            )
        ):
            raise CheckpointCorruptError(
                "checkpoint fingerprint checksum is malformed"
            )
        if not isinstance(metadata.get("fingerprint"), Mapping):
            raise CheckpointCorruptError(
                "checkpoint fingerprint document is malformed"
            )

        rng_document = metadata.get("inference_rng")
        if not isinstance(rng_document, dict):
            raise CheckpointCorruptError("inference_rng metadata is missing")
        try:
            dtype = np.dtype(rng_document["dtype"])
            shape = tuple(int(dim) for dim in rng_document["shape"])
            values = np.asarray(rng_document["values"], dtype=dtype).reshape(shape)
        except Exception as exc:
            raise CheckpointCorruptError(
                "inference_rng metadata is malformed"
            ) from exc
        if dtype.kind not in "ui" or values.size == 0:
            raise CheckpointCorruptError("inference_rng metadata is invalid")

        if marker_path.exists():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise CheckpointCorruptError(
                    f"checkpoint completion marker is unreadable: {checkpoint_path}"
                ) from exc
            expected_marker = {
                "completion_marker_version": COMPLETION_MARKER_VERSION,
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "learner_step": learner_step,
                "agent_state_sha256": actual_state_sha,
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            }
            if marker != expected_marker:
                raise CheckpointCorruptError(
                    "checkpoint completion marker does not match its payload"
                )
        elif require_completion_marker:
            raise CheckpointCorruptError(
                f"checkpoint completion marker is missing: {checkpoint_path}"
            )

        return _CheckpointContents(
            metadata=metadata,
            state_bytes=state_bytes,
            learner_step=learner_step,
            gradient_step=gradient_step,
            policy_version=policy_version,
            inference_rng=values,
            emergency=emergency,
        )

    def latest_path(self) -> Path:
        candidates: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            if not path.is_dir() or not path.name.startswith(CHECKPOINT_PREFIX):
                continue
            suffix = path.name.removeprefix(CHECKPOINT_PREFIX)
            if suffix.isdigit():
                candidates.append((int(suffix), path))
        if not candidates:
            raise FileNotFoundError(f"no learner checkpoints under {self.root}")
        skipped: list[str] = []
        for _, path in sorted(candidates, key=lambda item: item[0], reverse=True):
            try:
                self._read_contents(path, require_completion_marker=True)
            except CheckpointCorruptError:
                skipped.append(path.name)
                continue
            return path
        detail = f"; skipped={','.join(skipped)}" if skipped else ""
        raise FileNotFoundError(
            "no complete, structurally valid learner checkpoints under "
            f"{self.root}{detail}"
        )

    def load(
        self,
        *,
        agent_template: Any,
        fingerprint: LearnerFingerprint,
        path: os.PathLike[str] | str | None = None,
        allow_legacy_markerless: bool = False,
    ) -> RestoredCheckpoint:
        if not isinstance(allow_legacy_markerless, bool):
            raise TypeError("allow_legacy_markerless must be a bool")
        checkpoint_path = (
            Path(path).expanduser().resolve() if path is not None else self.latest_path()
        )
        contents = self._read_contents(
            checkpoint_path,
            # A named path is not proof that the writer completed.  Production
            # resume is therefore strict for both automatic and explicit
            # selection.  Legacy markerless v1 artifacts remain readable only
            # behind an operator-visible opt-in for one-off migration.
            require_completion_marker=not allow_legacy_markerless,
        )
        metadata = contents.metadata
        state_bytes = contents.state_bytes
        if metadata.get("fingerprint_sha256") != fingerprint.sha256:
            raise CheckpointFingerprintError(
                "learner checkpoint fingerprint mismatch"
            )
        if metadata.get("fingerprint") != fingerprint.document:
            raise CheckpointFingerprintError(
                "learner checkpoint fingerprint document mismatch"
            )

        from flax import serialization
        import jax.numpy as jnp

        try:
            state = serialization.from_bytes(agent_template.state, state_bytes)
            validate_tree_finite(state, name="restored agent state")
        except Exception as exc:
            if isinstance(exc, CheckpointError):
                raise
            raise CheckpointCorruptError("agent state deserialization failed") from exc
        state_step = int(np.asarray(state.step))
        if state_step != contents.gradient_step:
            raise CheckpointCorruptError(
                "gradient_step does not match serialized agent state"
            )
        return RestoredCheckpoint(
            agent=agent_template.replace(state=state),
            learner_step=contents.learner_step,
            gradient_step=contents.gradient_step,
            policy_version=contents.policy_version,
            inference_rng=jnp.asarray(contents.inference_rng),
            path=checkpoint_path,
            emergency=contents.emergency,
        )


class CheckpointRunLock:
    """Hold one advisory single-writer lock for a checkpoint root."""

    LOCK_FILE_NAME = ".learner-writer.lock"

    def __init__(self, root: os.PathLike[str] | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.LOCK_FILE_NAME
        self._stream: Any | None = None

    @property
    def acquired(self) -> bool:
        return self._stream is not None

    def acquire(self) -> None:
        if self._stream is not None:
            raise RuntimeError("checkpoint run lock is already acquired")
        import fcntl

        stream = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            owner = stream.read(2_000).strip() or "unknown owner"
            stream.close()
            raise CheckpointLockError(
                f"checkpoint root already has an active learner: {owner}"
            ) from exc
        try:
            owner = _canonical_json(
                {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "started_time_ns": time.time_ns(),
                }
            ).decode("ascii")
            stream.seek(0)
            stream.truncate()
            stream.write(owner + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
            raise
        self._stream = stream

    def release(self) -> None:
        stream = self._stream
        if stream is None:
            return
        import fcntl

        self._stream = None
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "CheckpointRunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()
