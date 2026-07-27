"""Server-side replay schema support for UR7e HIL-SERL metadata.

Upstream HIL-SERL replay buffers allocate a fixed dictionary of numpy arrays.
Unknown transition keys are therefore transmitted by Agentlace but discarded
when the learner inserts them.  These helpers extend a freshly constructed
buffer without modifying ``third_party/hil-serl``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def install_metadata_schema(replay_buffer: Any) -> Any:
    """Allocate metadata columns on a fresh upstream replay buffer.

    The buffer must not contain data yet: adding columns after insertion would
    leave historical rows ambiguous.
    """
    if len(replay_buffer) != 0:
        raise ValueError("metadata schema must be installed on an empty replay buffer")

    dataset = replay_buffer.dataset_dict
    actions = np.asarray(dataset["actions"])
    if actions.ndim < 2:
        raise ValueError("replay action storage must include a capacity dimension")
    capacity = actions.shape[0]

    expected = {
        "policy_actions": np.empty_like(actions),
        "intervened": np.empty((capacity,), dtype=np.uint8),
        "timestamp_ns": np.empty((capacity,), dtype=np.int64),
        "policy_version": np.empty((capacity,), dtype=np.int64),
    }
    for key, values in expected.items():
        if key in dataset:
            existing = np.asarray(dataset[key])
            if existing.shape != values.shape or existing.dtype != values.dtype:
                raise ValueError(
                    f"existing replay column {key!r} has "
                    f"shape/dtype {existing.shape}/{existing.dtype}, expected "
                    f"{values.shape}/{values.dtype}"
                )
        else:
            dataset[key] = values
    return replay_buffer


def normalize_transition_metadata(
    transition: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an insert-ready copy with validated metadata.

    Old demonstrations remain loadable: missing policy/intervention metadata
    is reconstructed conservatively, while an unknown timestamp is stored as
    ``-1``.  New online transitions always carry a positive ``timestamp_ns``.
    """
    normalized = dict(transition)

    actions = np.asarray(normalized["actions"])
    policy_actions = np.asarray(normalized.get("policy_actions", actions)).copy()
    if policy_actions.shape != actions.shape:
        raise ValueError(
            "policy_actions shape must match actions: "
            f"{policy_actions.shape} != {actions.shape}"
        )
    normalized["policy_actions"] = policy_actions

    intervened = int(normalized.get("intervened", 0))
    if intervened not in (0, 1):
        raise ValueError(f"intervened must be 0 or 1, got {intervened!r}")
    normalized["intervened"] = np.uint8(intervened)

    timestamp_ns = normalized.get("timestamp_ns", -1)
    if isinstance(timestamp_ns, (bool, np.bool_)) or not isinstance(
        timestamp_ns, (int, np.integer)
    ):
        raise TypeError("timestamp_ns must be an integer")
    timestamp_ns = int(timestamp_ns)
    if timestamp_ns == 0 or timestamp_ns < -1:
        raise ValueError("timestamp_ns must be positive, or -1 for legacy data")
    if timestamp_ns > np.iinfo(np.int64).max:
        raise ValueError("timestamp_ns exceeds signed 64-bit storage")
    normalized["timestamp_ns"] = np.int64(timestamp_ns)

    policy_version = normalized.get("policy_version", -1)
    if isinstance(policy_version, (bool, np.bool_)) or not isinstance(
        policy_version, (int, np.integer)
    ):
        raise TypeError("policy_version must be an integer")
    policy_version = int(policy_version)
    if policy_version < -1 or policy_version > np.iinfo(np.int64).max:
        raise ValueError("policy_version must be -1 or a non-negative int64")
    normalized["policy_version"] = np.int64(policy_version)
    return normalized
