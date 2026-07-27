"""One-time canonical demo conversion into frozen-trunk feature storage.

Raw camera arrays are accepted only during construction.  The resulting pool
owns explicit current/next frozen ResNet-10 trunk maps and never retains the
input ``LoadedDemos`` object, its transitions, the extractor, or pixel arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from ur_env.learner.config import (
    FROZEN_TRUNK_FEATURE_SHAPE,
    FROZEN_TRUNK_REPRESENTATION,
    LEARNER_AUGMENTATION,
)
from ur_env.learner.demo import (
    LEARNER_BATCH_KEYS,
    DemoSidecar,
    LoadedDemos,
)
from ur_env.observation_schema import validate_canonical_observation


_IMAGE_KEYS = ("cam1", "cam2")
_OBSERVATION_KEYS = {"state", *_IMAGE_KEYS}
_STATE_SHAPE = (1, 19)
_ACTION_SHAPE = (7,)
_FLOAT32 = np.dtype(np.float32)


class FrozenTrunkDemoExtractor(Protocol):
    """Encode a canonical raw observation batch at the frozen trunk cut."""

    def __call__(
        self, observation: Mapping[str, np.ndarray]
    ) -> Mapping[str, Any]:
        ...


class FeatureDemoContractError(ValueError):
    """A loaded demo or extractor result violates feature-demo storage."""


@dataclass(frozen=True)
class FeatureDemoMemoryEstimate:
    """Exact persistent numpy allocation; provenance sidecars are excluded."""

    transition_count: int
    camera_bytes: int
    state_bytes: int
    action_bytes: int
    scalar_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.camera_bytes
            + self.state_bytes
            + self.action_bytes
            + self.scalar_bytes
        )

    @property
    def total_gib(self) -> float:
        return self.total_bytes / 1024**3


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def estimate_feature_demo_memory(
    transition_count: int,
) -> FeatureDemoMemoryEstimate:
    """Return the exact persistent tensor bytes for a converted demo pool."""

    count = _nonnegative_int(transition_count, name="transition_count")
    float_bytes = _FLOAT32.itemsize
    feature_elements = int(
        np.prod(FROZEN_TRUNK_FEATURE_SHAPE, dtype=np.int64)
    )
    # current/next x cam1/cam2
    camera_bytes = count * 2 * len(_IMAGE_KEYS) * feature_elements * float_bytes
    state_bytes = count * 2 * int(np.prod(_STATE_SHAPE)) * float_bytes
    action_bytes = count * int(np.prod(_ACTION_SHAPE)) * float_bytes
    # rewards, masks, grasp_penalty
    scalar_bytes = count * 3 * float_bytes
    return FeatureDemoMemoryEstimate(
        transition_count=count,
        camera_bytes=camera_bytes,
        state_bytes=state_bytes,
        action_bytes=action_bytes,
        scalar_bytes=scalar_bytes,
    )


def _strict_float32_array(
    value: Any, *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise FeatureDemoContractError(f"{name} is not array-like") from exc
    if array.dtype != _FLOAT32:
        raise FeatureDemoContractError(
            f"{name} must have dtype float32, got {array.dtype}"
        )
    if tuple(array.shape) != shape:
        raise FeatureDemoContractError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FeatureDemoContractError(f"{name} contains non-finite data")
    if not shape:
        return array.copy()
    return np.ascontiguousarray(array)


def _strict_scalar(value: Any, *, name: str) -> np.float32:
    array = _strict_float32_array(value, name=name, shape=())
    return np.float32(array.item())


def _validate_loaded_demos(loaded: Any) -> LoadedDemos:
    if not isinstance(loaded, LoadedDemos):
        raise TypeError("loaded must be a LoadedDemos instance")
    if not isinstance(loaded.transitions, tuple):
        raise FeatureDemoContractError("loaded.transitions must be a tuple")
    if not isinstance(loaded.sidecars, tuple):
        raise FeatureDemoContractError("loaded.sidecars must be a tuple")
    if len(loaded.transitions) != len(loaded.sidecars):
        raise FeatureDemoContractError(
            "loaded transitions and sidecars must have equal length"
        )
    return loaded


def _copy_sidecars(sidecars: tuple[DemoSidecar, ...]) -> tuple[DemoSidecar, ...]:
    copied: list[DemoSidecar] = []
    for index, sidecar in enumerate(sidecars):
        if not isinstance(sidecar, DemoSidecar):
            raise FeatureDemoContractError(
                f"sidecars[{index}] must be a DemoSidecar"
            )
        if not isinstance(sidecar.source_path, str):
            raise FeatureDemoContractError(
                f"sidecars[{index}].source_path must be a string"
            )
        if (
            isinstance(sidecar.item_index, bool)
            or not isinstance(sidecar.item_index, int)
            or sidecar.item_index < 0
        ):
            raise FeatureDemoContractError(
                f"sidecars[{index}].item_index must be non-negative integer"
            )
        if not isinstance(sidecar.metadata, Mapping):
            raise FeatureDemoContractError(
                f"sidecars[{index}].metadata must be a mapping"
            )
        metadata: dict[str, Any] = {}
        for key, value in sidecar.metadata.items():
            if not isinstance(key, str) or not isinstance(
                value, (str, int, float, bool, type(None))
            ):
                raise FeatureDemoContractError(
                    f"sidecars[{index}].metadata must contain scalar provenance"
                )
            metadata[key] = value
        copied.append(
            DemoSidecar(
                source_path=sidecar.source_path,
                item_index=sidecar.item_index,
                metadata=MappingProxyType(metadata),
            )
        )
    return tuple(copied)


def _validate_raw_transition(
    transition: Any, *, index: int
) -> dict[str, Any]:
    if not isinstance(transition, Mapping):
        raise FeatureDemoContractError(
            f"transitions[{index}] must be a mapping"
        )
    if set(transition) != set(LEARNER_BATCH_KEYS):
        raise FeatureDemoContractError(
            f"transitions[{index}] must contain exactly the learner fields"
        )
    try:
        observations = validate_canonical_observation(
            transition["observations"], copy=False
        )
        next_observations = validate_canonical_observation(
            transition["next_observations"], copy=False
        )
    except Exception as exc:
        raise FeatureDemoContractError(
            f"transitions[{index}] contains an invalid raw observation: {exc}"
        ) from exc

    actions = _strict_float32_array(
        transition["actions"],
        name=f"transitions[{index}].actions",
        shape=_ACTION_SHAPE,
    )
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        raise FeatureDemoContractError(
            f"transitions[{index}].actions must stay within [-1, 1]"
        )
    if float(actions[-1]) not in (-1.0, 0.0, 1.0):
        raise FeatureDemoContractError(
            f"transitions[{index}] gripper action must be in {{-1, 0, 1}}"
        )
    reward = _strict_scalar(
        transition["rewards"], name=f"transitions[{index}].rewards"
    )
    mask = _strict_scalar(
        transition["masks"], name=f"transitions[{index}].masks"
    )
    penalty = _strict_scalar(
        transition["grasp_penalty"],
        name=f"transitions[{index}].grasp_penalty",
    )
    if float(reward) not in (0.0, 1.0):
        raise FeatureDemoContractError(
            f"transitions[{index}].rewards must be binary"
        )
    if float(mask) not in (0.0, 1.0):
        raise FeatureDemoContractError(
            f"transitions[{index}].masks must be binary"
        )
    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": reward,
        "masks": mask,
        "grasp_penalty": penalty,
    }


def _extract_batch(
    extractor: FrozenTrunkDemoExtractor,
    observation: Mapping[str, np.ndarray],
    *,
    name: str,
) -> dict[str, np.ndarray]:
    try:
        output = extractor(observation)
    except Exception as exc:
        raise FeatureDemoContractError(
            f"frozen-trunk extraction failed for {name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(output, Mapping) or set(output) != _OBSERVATION_KEYS:
        raise FeatureDemoContractError(
            f"{name} extractor output must contain exactly state, cam1, cam2"
        )

    batch_size = int(observation["state"].shape[0])
    state = _strict_float32_array(
        output["state"], name=f"{name}.state", shape=(batch_size, *_STATE_SHAPE)
    )
    if not np.array_equal(state, observation["state"]):
        raise FeatureDemoContractError(
            f"{name}.state must exactly equal the raw canonical state"
        )
    result = {"state": state}
    expected_feature_shape = (batch_size, *FROZEN_TRUNK_FEATURE_SHAPE)
    for image_key in _IMAGE_KEYS:
        result[image_key] = _strict_float32_array(
            output[image_key],
            name=f"{name}.{image_key}",
            shape=expected_feature_shape,
        )
    return result


class FrozenTrunkFeatureDemoPool:
    """Immutable, seeded uniform pool containing no raw camera arrays."""

    observation_representation = FROZEN_TRUNK_REPRESENTATION
    augmentation = LEARNER_AUGMENTATION
    feature_shape = FROZEN_TRUNK_FEATURE_SHAPE

    def __init__(
        self,
        loaded: LoadedDemos,
        *,
        feature_extractor: FrozenTrunkDemoExtractor,
        seed: int = 42,
        extraction_batch_size: int = 256,
    ) -> None:
        loaded = _validate_loaded_demos(loaded)
        if not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")
        seed_value = _nonnegative_int(seed, name="seed")
        chunk_size = _positive_int(
            extraction_batch_size, name="extraction_batch_size"
        )
        sidecars = _copy_sidecars(loaded.sidecars)
        transitions = tuple(
            _validate_raw_transition(transition, index=index)
            for index, transition in enumerate(loaded.transitions)
        )
        count = len(transitions)

        observations = {
            "state": np.empty((count, *_STATE_SHAPE), dtype=np.float32),
            **{
                key: np.empty(
                    (count, *FROZEN_TRUNK_FEATURE_SHAPE), dtype=np.float32
                )
                for key in _IMAGE_KEYS
            },
        }
        next_observations = {
            "state": np.empty((count, *_STATE_SHAPE), dtype=np.float32),
            **{
                key: np.empty(
                    (count, *FROZEN_TRUNK_FEATURE_SHAPE), dtype=np.float32
                )
                for key in _IMAGE_KEYS
            },
        }
        actions = np.empty((count, *_ACTION_SHAPE), dtype=np.float32)
        rewards = np.empty((count,), dtype=np.float32)
        masks = np.empty((count,), dtype=np.float32)
        grasp_penalty = np.empty((count,), dtype=np.float32)

        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            selected = transitions[start:stop]
            raw_current = {
                key: np.stack([item["observations"][key] for item in selected])
                for key in ("state", *_IMAGE_KEYS)
            }
            raw_next = {
                key: np.stack(
                    [item["next_observations"][key] for item in selected]
                )
                for key in ("state", *_IMAGE_KEYS)
            }
            encoded_current = _extract_batch(
                feature_extractor,
                raw_current,
                name=f"observations[{start}:{stop}]",
            )
            encoded_next = _extract_batch(
                feature_extractor,
                raw_next,
                name=f"next_observations[{start}:{stop}]",
            )
            for key in ("state", *_IMAGE_KEYS):
                observations[key][start:stop] = encoded_current[key]
                next_observations[key][start:stop] = encoded_next[key]
            actions[start:stop] = np.stack(
                [item["actions"] for item in selected]
            )
            rewards[start:stop] = np.asarray(
                [item["rewards"] for item in selected], dtype=np.float32
            )
            masks[start:stop] = np.asarray(
                [item["masks"] for item in selected], dtype=np.float32
            )
            grasp_penalty[start:stop] = np.asarray(
                [item["grasp_penalty"] for item in selected], dtype=np.float32
            )

        self._observations = observations
        self._next_observations = next_observations
        self._actions = actions
        self._rewards = rewards
        self._masks = masks
        self._grasp_penalty = grasp_penalty
        self._sidecars = sidecars
        self._rng = np.random.default_rng(seed_value)
        self._rng_lock = threading.Lock()
        self._memory_estimate = estimate_feature_demo_memory(count)
        if self.storage_nbytes != self._memory_estimate.total_bytes:
            raise RuntimeError("feature demo allocation differs from its estimate")

    def __len__(self) -> int:
        return int(self._actions.shape[0])

    @property
    def sidecars(self) -> tuple[DemoSidecar, ...]:
        return self._sidecars

    @property
    def memory_estimate(self) -> FeatureDemoMemoryEstimate:
        return self._memory_estimate

    @property
    def storage_nbytes(self) -> int:
        arrays = (
            *self._observations.values(),
            *self._next_observations.values(),
            self._actions,
            self._rewards,
            self._masks,
            self._grasp_penalty,
        )
        return sum(array.nbytes for array in arrays)

    def sample(self, batch_size: int, **kwargs: Any) -> dict[str, Any]:
        """Sample with replacement, always returning explicit next features."""

        if kwargs:
            raise ValueError(
                "feature demo sampling does not support packed-image options"
            )
        size = _positive_int(batch_size, name="batch_size")
        if not len(self):
            raise ValueError("cannot sample an empty feature demo pool")
        with self._rng_lock:
            indices = self._rng.integers(len(self), size=size)
        return {
            "observations": {
                key: value[indices]
                for key, value in self._observations.items()
            },
            "next_observations": {
                key: value[indices]
                for key, value in self._next_observations.items()
            },
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "masks": self._masks[indices],
            "grasp_penalty": self._grasp_penalty[indices],
        }


def convert_loaded_demos_to_feature_pool(
    loaded: LoadedDemos,
    *,
    feature_extractor: FrozenTrunkDemoExtractor,
    seed: int = 42,
    extraction_batch_size: int = 256,
) -> FrozenTrunkFeatureDemoPool:
    """Convert strict raw demos once and return feature-native storage."""

    return FrozenTrunkFeatureDemoPool(
        loaded,
        feature_extractor=feature_extractor,
        seed=seed,
        extraction_batch_size=extraction_batch_size,
    )


__all__ = [
    "FeatureDemoContractError",
    "FeatureDemoMemoryEstimate",
    "FrozenTrunkDemoExtractor",
    "FrozenTrunkFeatureDemoPool",
    "convert_loaded_demos_to_feature_pool",
    "estimate_feature_demo_memory",
]
