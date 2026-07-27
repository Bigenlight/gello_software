"""Non-destructive full learner checkpoints with strict resume fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
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


class CheckpointError(RuntimeError):
    """Base error for learner checkpoint operations."""


class CheckpointExistsError(CheckpointError):
    """Saving would overwrite an existing checkpoint."""


class CheckpointCorruptError(CheckpointError):
    """A checkpoint is incomplete, malformed, or fails its checksum."""


class CheckpointFingerprintError(CheckpointError):
    """A checkpoint belongs to a different learner/schema/asset contract."""


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
    ) -> "LearnerFingerprint":
        resnet_sha256 = verify_resnet10_asset(resnet_asset_path)
        document = {
            "observation_schema_hash": CANONICAL_OBSERVATION_SCHEMA_HASH,
            "observation_schema": observation_schema_document(),
            "learner_config": config.fingerprint_values(),
            "resnet10_sha256": resnet_sha256,
        }
        digest = hashlib.sha256(_canonical_json(document)).hexdigest()
        return cls(document=document, sha256=digest)


@dataclass(frozen=True)
class RestoredCheckpoint:
    agent: Any
    learner_step: int
    gradient_step: int
    policy_version: int
    inference_rng: Any
    path: Path


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointCorruptError(f"{name} must be a non-negative integer")
    return value


class CheckpointManager:
    """Write immutable ``checkpoint_<learner_step>`` directories."""

    def __init__(self, root: os.PathLike[str] | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def checkpoint_name(learner_step: int) -> str:
        if isinstance(learner_step, bool) or not isinstance(learner_step, int):
            raise ValueError("learner_step must be an integer")
        if learner_step < 0:
            raise ValueError("learner_step must be non-negative")
        return f"checkpoint_{learner_step:012d}"

    def path_for_step(self, learner_step: int) -> Path:
        return self.root / self.checkpoint_name(learner_step)

    def save(
        self,
        *,
        agent: Any,
        learner_step: int,
        gradient_step: int,
        policy_version: int,
        inference_rng: Any,
        fingerprint: LearnerFingerprint,
    ) -> Path:
        for name, value in (
            ("learner_step", learner_step),
            ("gradient_step", gradient_step),
            ("policy_version", policy_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        destination = self.path_for_step(learner_step)
        try:
            destination.mkdir()
        except FileExistsError as exc:
            raise CheckpointExistsError(
                f"checkpoint already exists and will not be overwritten: {destination}"
            ) from exc

        from flax import serialization
        import jax

        validate_tree_finite(agent.state, name="agent state")
        state_bytes = serialization.to_bytes(agent.state)
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
        }
        try:
            self._write_new_file(destination / "agent_state.msgpack", state_bytes)
            self._write_new_file(destination / "metadata.json", _canonical_json(metadata))
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

    def latest_path(self) -> Path:
        candidates: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            if not path.is_dir() or not path.name.startswith("checkpoint_"):
                continue
            suffix = path.name.removeprefix("checkpoint_")
            if suffix.isdigit():
                candidates.append((int(suffix), path))
        if not candidates:
            raise FileNotFoundError(f"no learner checkpoints under {self.root}")
        return max(candidates, key=lambda item: item[0])[1]

    def load(
        self,
        *,
        agent_template: Any,
        fingerprint: LearnerFingerprint,
        path: os.PathLike[str] | str | None = None,
    ) -> RestoredCheckpoint:
        checkpoint_path = (
            Path(path).expanduser().resolve() if path is not None else self.latest_path()
        )
        metadata_path = checkpoint_path / "metadata.json"
        state_path = checkpoint_path / "agent_state.msgpack"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            state_bytes = state_path.read_bytes()
        except Exception as exc:
            raise CheckpointCorruptError(
                f"checkpoint is incomplete or unreadable: {checkpoint_path}"
            ) from exc
        if not isinstance(metadata, dict):
            raise CheckpointCorruptError("checkpoint metadata must be an object")
        if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointCorruptError("unsupported checkpoint format version")
        actual_state_sha = hashlib.sha256(state_bytes).hexdigest()
        if metadata.get("agent_state_sha256") != actual_state_sha:
            raise CheckpointCorruptError("agent state checksum mismatch")
        if metadata.get("fingerprint_sha256") != fingerprint.sha256:
            raise CheckpointFingerprintError(
                "learner checkpoint fingerprint mismatch"
            )
        if metadata.get("fingerprint") != fingerprint.document:
            raise CheckpointFingerprintError(
                "learner checkpoint fingerprint document mismatch"
            )

        learner_step = _nonnegative_int(
            metadata.get("learner_step"), name="learner_step"
        )
        gradient_step = _nonnegative_int(
            metadata.get("gradient_step"), name="gradient_step"
        )
        policy_version = _nonnegative_int(
            metadata.get("policy_version"), name="policy_version"
        )
        if checkpoint_path.name != self.checkpoint_name(learner_step):
            raise CheckpointCorruptError(
                "checkpoint directory name does not match learner_step"
            )

        rng_document = metadata.get("inference_rng")
        if not isinstance(rng_document, dict):
            raise CheckpointCorruptError("inference_rng metadata is missing")
        try:
            dtype = np.dtype(rng_document["dtype"])
            shape = tuple(int(dim) for dim in rng_document["shape"])
            values = np.asarray(rng_document["values"], dtype=dtype).reshape(shape)
        except Exception as exc:
            raise CheckpointCorruptError("inference_rng metadata is malformed") from exc
        if dtype.kind not in "ui" or values.size == 0:
            raise CheckpointCorruptError("inference_rng metadata is invalid")

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
        if state_step != gradient_step:
            raise CheckpointCorruptError(
                "gradient_step does not match serialized agent state"
            )
        return RestoredCheckpoint(
            agent=agent_template.replace(state=state),
            learner_step=learner_step,
            gradient_step=gradient_step,
            policy_version=policy_version,
            inference_rng=jnp.asarray(values),
            path=checkpoint_path,
        )
