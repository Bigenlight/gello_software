"""Canonical, dependency-light observation contract for remote HIL-SERL.

The contract lives outside the learner runtime so the robot laptop can verify
it without importing JAX, Flax, or the upstream HIL-SERL package.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ur_env.actor_network import ActorProtocolError


OBSERVATION_SCHEMA_ID = "hil-serl-ur-canonical-observation-v1"
CANONICAL_OBSERVATION_SPEC = MappingProxyType(
    {
        "cam1": (np.dtype(np.uint8), (1, 128, 128, 3)),
        "cam2": (np.dtype(np.uint8), (1, 128, 128, 3)),
        "state": (np.dtype(np.float32), (1, 19)),
    }
)


def _schema_document() -> dict[str, Any]:
    return {
        "schema_id": OBSERVATION_SCHEMA_ID,
        "tensors": [
            {
                "path": key,
                "dtype": dtype.name,
                "shape": list(shape),
            }
            for key, (dtype, shape) in sorted(CANONICAL_OBSERVATION_SPEC.items())
        ],
    }


CANONICAL_OBSERVATION_SCHEMA_HASH = hashlib.sha256(
    json.dumps(
        _schema_document(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()


def validate_canonical_observation(
    observation: Mapping[str, Any], *, copy: bool = True
) -> dict[str, np.ndarray]:
    """Validate the exact policy/replay observation tree.

    No coercion is performed: accepting a different dtype and casting it here
    would hide a laptop/server checkpoint-contract mismatch.
    """
    if not isinstance(observation, Mapping):
        raise ActorProtocolError("canonical observation must be a mapping")
    actual_keys = set(observation)
    expected_keys = set(CANONICAL_OBSERVATION_SPEC)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ActorProtocolError(
            "canonical observation keys mismatch: "
            f"missing={missing}, extra={extra}"
        )

    result: dict[str, np.ndarray] = {}
    for key, (expected_dtype, expected_shape) in CANONICAL_OBSERVATION_SPEC.items():
        array = np.asarray(observation[key])
        if array.dtype != expected_dtype:
            raise ActorProtocolError(
                f"observation {key!r} must have dtype {expected_dtype.name}, "
                f"got {array.dtype}"
            )
        if array.shape != expected_shape:
            raise ActorProtocolError(
                f"observation {key!r} must have shape {expected_shape}, "
                f"got {array.shape}"
            )
        if expected_dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ActorProtocolError(
                f"observation {key!r} contains a non-finite value"
            )
        contiguous = np.ascontiguousarray(array)
        result[key] = contiguous.copy() if copy else contiguous
    return result


def observation_schema_document() -> dict[str, Any]:
    """Return a mutable copy suitable for status output and tests."""
    return _schema_document()
