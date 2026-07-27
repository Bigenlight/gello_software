"""Canonical, dependency-light observation contract for remote HIL-SERL.

The contract lives outside the learner runtime so the robot laptop can verify
it without importing JAX, Flax, or the upstream HIL-SERL package.

WHY THE 19-D ``state`` ORDER LOOKS "WRONG"
------------------------------------------
The flat ``state`` vector is not assembled by us.  It is produced by upstream
``serl_launcher.wrappers.serl_obs_wrappers.SERLObsWrapper``, which does::

    self.proprio_space = gym.spaces.Dict({k: ... for k in self.proprio_keys})
    flatten(self.proprio_space, {k: obs["state"][k] for k in self.proprio_keys})

``gym.spaces.Dict`` **re-sorts a plain mapping alphabetically** (gymnasium
``spaces/dict.py``: "for legacy reasons, we need to preserve the sorted
dictionary items ... as this could matter for projects flatten the
dictionary").  ``proprio_keys`` therefore selects *which* groups appear, never
*in what order* they are laid out.  Verified on gymnasium 1.2.0 against the real
``UR7eEnv`` observation space and the real ``SERLObsWrapper``: passing
``["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]`` still
yields ``['gripper_pose', 'tcp_force', 'tcp_pose', 'tcp_torque', 'tcp_vel']``.

``third_party/hil-serl`` is a pinned submodule we must not patch, and the
alphabetical order is what every recorded demo / replay buffer / checkpoint
actually contains.  Reality is therefore canonical: ``STATE_GROUPS`` below is
the flatten order, and ``STATE_FEATURES`` is derived from it so the two can
never drift.  ``tests/test_state_layout_contract.py`` re-derives the layout from
the live env + the live gymnasium and fails if this file ever stops describing
what the pipeline produces.

CONSEQUENCE FOR CALLERS: the gripper scalar is at index **0**, not ``-1``.
Use ``GRIPPER_POSITION_INDEX`` / ``gripper_position_from_state()`` — never
``state[..., -1]`` (that is TCP angular velocity z).
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ur_env.actor_network import ActorProtocolError


OBSERVATION_SCHEMA_ID = "hil-serl-ur-canonical-observation-v2"

# The proprio groups handed to SERLObsWrapper(env, proprio_keys=...).  This is a
# SET of selectors: the order written here has no effect on the flat layout
# (see the module docstring).  Kept as a tuple only so the schema document is
# reproducible.
PROPRIO_KEYS = ("gripper_pose", "tcp_force", "tcp_pose", "tcp_torque", "tcp_vel")

# THE flat layout, in the exact order gymnasium's Dict flatten emits it.
# Group order == alphabetical by proprio key.  Within a group the order is the
# producing env's own (UR7eEnv.observation_space, post-Quat2EulerWrapper).
STATE_GROUPS = (
    ("gripper_pose", ("gripper_position",)),
    (
        "tcp_force",
        ("tcp_force_x", "tcp_force_y", "tcp_force_z"),
    ),
    (
        "tcp_pose",
        (
            "tcp_position_x",
            "tcp_position_y",
            "tcp_position_z",
            "tcp_euler_x",
            "tcp_euler_y",
            "tcp_euler_z",
        ),
    ),
    (
        "tcp_torque",
        ("tcp_torque_x", "tcp_torque_y", "tcp_torque_z"),
    ),
    (
        "tcp_vel",
        (
            "tcp_linear_velocity_x",
            "tcp_linear_velocity_y",
            "tcp_linear_velocity_z",
            "tcp_angular_velocity_x",
            "tcp_angular_velocity_y",
            "tcp_angular_velocity_z",
        ),
    ),
)

# Derived — never hand-maintain these, or the doc can disagree with itself.
STATE_FEATURES = tuple(
    feature for _key, features in STATE_GROUPS for feature in features
)
STATE_DIM = len(STATE_FEATURES)


def _canonical_layout() -> tuple[tuple[str, int, int], ...]:
    layout: list[tuple[str, int, int]] = []
    start = 0
    for key, features in STATE_GROUPS:
        stop = start + len(features)
        layout.append((key, start, stop))
        start = stop
    return tuple(layout)


#: ``((proprio_key, start, stop), ...)`` — the half-open flat slice per group.
CANONICAL_STATE_LAYOUT = _canonical_layout()
STATE_GROUP_SLICES = MappingProxyType(
    {key: slice(start, stop) for key, start, stop in CANONICAL_STATE_LAYOUT}
)
STATE_FEATURE_INDEX = MappingProxyType(
    {feature: index for index, feature in enumerate(STATE_FEATURES)}
)

#: Index of the gripper scalar in the flat state.  It is 0, NOT -1.
GRIPPER_POSITION_INDEX = STATE_FEATURE_INDEX["gripper_position"]

CANONICAL_OBSERVATION_SPEC = MappingProxyType(
    {
        "cam1": (np.dtype(np.uint8), (1, 128, 128, 3)),
        "cam2": (np.dtype(np.uint8), (1, 128, 128, 3)),
        "state": (np.dtype(np.float32), (1, STATE_DIM)),
    }
)


def _schema_document() -> dict[str, Any]:
    return {
        "schema_id": OBSERVATION_SCHEMA_ID,
        "proprio_keys": sorted(PROPRIO_KEYS),
        # Ordered: json.dumps(sort_keys=True) sorts dict KEYS, never list
        # ELEMENTS, so permuting either list changes the hash.  That is the
        # point — a layout change must invalidate the laptop<->Kanu handshake.
        "state_features": list(STATE_FEATURES),
        "state_layout": [
            {"key": key, "start": start, "stop": stop}
            for key, start, stop in CANONICAL_STATE_LAYOUT
        ],
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


# --------------------------------------------------------------------------- #
# Runtime binding: does the LIVE pipeline actually produce the layout above?    #
#                                                                              #
# CANONICAL_OBSERVATION_SCHEMA_HASH cannot answer this.  Both peers import this #
# module, so they always agree on the hash even when the document is wrong      #
# about reality; and validate_canonical_observation() only sees dtype/shape,    #
# which a permuted 19-vector satisfies perfectly.  The only way to catch a      #
# silent reordering (a gymnasium upgrade that changes Dict sorting, a renamed   #
# or resized proprio group) is to probe the real space.                         #
# --------------------------------------------------------------------------- #
def flatten_state_layout(proprio_space: Any) -> tuple[tuple[str, int, int], ...]:
    """Empirically derive the flat ``state`` layout of a proprio Dict space.

    Feeds a sentinel observation through the *real* ``gymnasium.spaces.flatten``
    — the same call ``SERLObsWrapper`` makes — and reads back where each group
    landed.  Nothing here assumes alphabetical ordering; if a future gymnasium
    stops sorting ``spaces.Dict``, this returns the new truth and
    :func:`assert_state_layout_matches` raises.

    Args:
        proprio_space: a ``gymnasium.spaces.Dict`` whose subspaces are flattenable.

    Returns:
        ``((proprio_key, start, stop), ...)`` in flat-vector order.
    """
    try:
        from gymnasium.spaces import flatdim, flatten
    except ImportError as exc:  # pragma: no cover - gymnasium is always present
        raise ActorProtocolError(
            "gymnasium is required to verify the state layout"
        ) from exc

    subspaces = getattr(proprio_space, "spaces", None)
    if not isinstance(subspaces, Mapping) or not subspaces:
        raise ActorProtocolError(
            "proprio space must be a non-empty gymnasium Dict space, "
            f"got {type(proprio_space).__name__}"
        )

    # Every scalar gets a globally unique sentinel so we can recover both the
    # group order AND the within-group order/contiguity from the flat vector.
    # Values stay small integers, exactly representable in float32.
    sentinel_owner: dict[int, tuple[str, int]] = {}
    sample: dict[str, np.ndarray] = {}
    next_value = 1
    for key, subspace in subspaces.items():
        size = flatdim(subspace)
        values = np.arange(next_value, next_value + size)
        for local_index, value in enumerate(values):
            sentinel_owner[int(value)] = (key, local_index)
        dtype = getattr(subspace, "dtype", None) or np.float32
        sample[key] = values.reshape(getattr(subspace, "shape", (size,))).astype(dtype)
        next_value += size

    flat = np.asarray(flatten(proprio_space, sample)).reshape(-1)
    if flat.size != next_value - 1:
        raise ActorProtocolError(
            f"flattened proprio space has {flat.size} entries, expected "
            f"{next_value - 1} — a subspace is not flatten-stable"
        )

    layout: list[tuple[str, int, int]] = []
    for flat_index, raw in enumerate(flat):
        value = int(round(float(raw)))
        owner = sentinel_owner.get(value)
        if owner is None:
            raise ActorProtocolError(
                f"flatten() produced unrecognised value {raw!r} at index "
                f"{flat_index} — the proprio space is not a pure permutation"
            )
        key, local_index = owner
        if layout and layout[-1][0] == key:
            if local_index != flat_index - layout[-1][1]:
                raise ActorProtocolError(
                    f"proprio group {key!r} is scrambled inside the flat state"
                )
            layout[-1] = (key, layout[-1][1], flat_index + 1)
            continue
        if any(entry[0] == key for entry in layout):
            raise ActorProtocolError(
                f"proprio group {key!r} is not contiguous in the flat state"
            )
        if local_index != 0:
            raise ActorProtocolError(
                f"proprio group {key!r} does not start at its first element"
            )
        layout.append((key, flat_index, flat_index + 1))
    return tuple(layout)


def assert_state_layout_matches(
    proprio_space: Any, *, source: str = "proprio space"
) -> tuple[tuple[str, int, int], ...]:
    """Raise unless ``proprio_space`` flattens exactly to the canonical layout.

    Call this once at actor/learner start-up, right after the wrapper chain is
    built.  A mismatch means every transition already in flight is mislabeled,
    so failing loudly at boot is strictly cheaper than training on it.
    """
    actual = flatten_state_layout(proprio_space)
    if actual != CANONICAL_STATE_LAYOUT:
        raise ActorProtocolError(
            f"{source} flattens to {actual!r}, but the canonical contract "
            f"{OBSERVATION_SCHEMA_ID} declares {CANONICAL_STATE_LAYOUT!r}. "
            "Either the env observation space changed or gymnasium's "
            "spaces.Dict ordering changed; do NOT train until "
            "ur_env/observation_schema.py is updated to match."
        )
    return actual


def state_slice(proprio_key: str) -> slice:
    """Flat-vector slice for one proprio group (e.g. ``state[..., state_slice('tcp_pose')]``)."""
    try:
        return STATE_GROUP_SLICES[proprio_key]
    except KeyError:
        raise ActorProtocolError(
            f"unknown proprio group {proprio_key!r}; known groups: "
            f"{sorted(STATE_GROUP_SLICES)}"
        ) from None


def gripper_position_from_state(state: Any) -> float:
    """Read the gripper scalar out of a canonical ``(1, 19)`` float32 state.

    Exists so no caller has to remember the index.  ``state[0, -1]`` is
    ``tcp_angular_velocity_z``, not the gripper.
    """
    array = np.asarray(state)
    expected_dtype, expected_shape = CANONICAL_OBSERVATION_SPEC["state"]
    if array.shape != expected_shape or array.dtype != expected_dtype:
        raise ActorProtocolError(
            f"canonical state must have shape {expected_shape} and dtype "
            f"{expected_dtype.name}, got {array.shape}/{array.dtype}"
        )
    return float(array[0, GRIPPER_POSITION_INDEX])


def feature_names_by_index() -> tuple[str, ...]:
    """Alias for :data:`STATE_FEATURES`, spelled for debug/plot call sites."""
    return STATE_FEATURES
