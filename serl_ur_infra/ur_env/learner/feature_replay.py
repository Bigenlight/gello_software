"""Feature-native, bounded RAM replay for the frozen ResNet-10 trunk.

The actor and reward classifier continue to use canonical uint8 camera
observations.  This module is the ownership boundary after reward
finalization: an injected frozen-trunk extractor converts both O(t) and
O(t+1), and only float32 feature maps plus learner tensors enter either ring.
No image stacking or later unpacking is performed.

O(t) is normally *not* re-extracted.  ``ActorSessionService`` guarantees that
one step's ``next_observation_id`` is the next step's ``observation_id`` for
the same session (``actor_network`` continuity checks plus its refusal to
accept a repeated ``observation_id``), so the trunk output for O(t) was
already computed as the previous transition's O(t+1).  A small per-session
cache returns it instead of paying a second GPU forward *and* a second
blocking ``device_get`` inside the lock this ingress shares with the learner.
The first transition of an episode still misses, because ``BeginEpisode``'s
observation never reached this ingress.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
import os
import threading
from typing import Any, Mapping, Protocol

import numpy as np

from ur_env.actor_network import ActorProtocolError, BufferStatus
from ur_env.learner.config import (
    FROZEN_TRUNK_FEATURE_SHAPE,
    FROZEN_TRUNK_REPRESENTATION,
    LEARNER_AUGMENTATION,
)
from ur_env.rlpd_receive_server import (
    DEFAULT_INTERVENTION_CAPACITY,
    DEFAULT_REPLAY_CAPACITY,
    IngressRecord,
    ReplayIngress,
)


FEATURE_ENCODING_ID = FROZEN_TRUNK_REPRESENTATION
FEATURE_AUGMENTATION = LEARNER_AUGMENTATION
FEATURE_KEYS = ("cam1", "cam2")
FEATURE_MAP_SHAPE = FROZEN_TRUNK_FEATURE_SHAPE
FEATURE_MAP_DTYPE = np.dtype(np.float32)

DEFAULT_MEMORY_RESERVE_BYTES = 2 * 1024**3

#: Encoded observations retained for the O(t+1) -> O(t) handoff.  Only the
#: immediately preceding observation can ever hit, so this exists to stay
#: correct across a retry rather than to raise the hit rate.  Each entry is
#: two (1, 4, 4, 512) float32 maps plus one state vector (~32 KiB per camera).
FEATURE_CACHE_CAPACITY = 4

_STATE_SHAPE = (1, 19)
_ACTION_SHAPE = (7,)


class FrozenTrunkFeatureExtractor(Protocol):
    """Convert one canonical raw observation to state plus camera maps."""

    def __call__(
        self, observation: Mapping[str, np.ndarray]
    ) -> Mapping[str, Any]:
        ...


class FeatureReplayError(RuntimeError):
    """The feature replay boundary cannot safely accept or sample data."""


class FeatureExtractionError(FeatureReplayError):
    """The frozen-trunk extractor failed or violated its output contract."""


class FeatureReplayMemoryError(FeatureReplayError):
    """Available RAM cannot satisfy the configured rings and reserve."""


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _expected_penalty(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("expected_grasp_penalty must be numeric")
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf":
        raise ValueError("expected_grasp_penalty must be numeric")
    result = float(array)
    if not math.isfinite(result) or result > 0.0:
        raise ValueError(
            "expected_grasp_penalty must be finite and non-positive"
        )
    return result


@dataclass(frozen=True)
class FeatureReplayMemoryEstimate:
    """Fixed numpy allocation for both rings; Python sidecars are excluded."""

    replay_capacity: int
    intervention_capacity: int
    camera_bytes: int
    state_bytes: int
    action_bytes: int
    scalar_bytes: int

    @property
    def total_capacity(self) -> int:
        return self.replay_capacity + self.intervention_capacity

    @property
    def fixed_tensor_bytes(self) -> int:
        return (
            self.camera_bytes
            + self.state_bytes
            + self.action_bytes
            + self.scalar_bytes
        )

    @property
    def camera_gib(self) -> float:
        return self.camera_bytes / 1024**3

    @property
    def fixed_tensor_gib(self) -> float:
        return self.fixed_tensor_bytes / 1024**3


def estimate_feature_replay_memory(
    *,
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
    intervention_capacity: int = DEFAULT_INTERVENTION_CAPACITY,
) -> FeatureReplayMemoryEstimate:
    """Estimate the exact fixed numpy tensors allocated by both rings.

    The camera estimate includes current and next maps for both cameras in
    every logical replay slot and every logical intervention slot.  At the
    production defaults this is 7.32421875 GiB of camera maps.
    """

    replay = _positive_int(replay_capacity, name="replay_capacity")
    intervention = _positive_int(
        intervention_capacity, name="intervention_capacity"
    )
    slots = replay + intervention
    float_bytes = FEATURE_MAP_DTYPE.itemsize
    map_elements = int(np.prod(FEATURE_MAP_SHAPE, dtype=np.int64))
    camera_bytes = slots * 2 * len(FEATURE_KEYS) * map_elements * float_bytes
    state_bytes = slots * 2 * int(np.prod(_STATE_SHAPE)) * float_bytes
    action_bytes = slots * int(np.prod(_ACTION_SHAPE)) * float_bytes
    # rewards, masks, and grasp_penalty
    scalar_bytes = slots * 3 * float_bytes
    return FeatureReplayMemoryEstimate(
        replay_capacity=replay,
        intervention_capacity=intervention,
        camera_bytes=camera_bytes,
        state_bytes=state_bytes,
        action_bytes=action_bytes,
        scalar_bytes=scalar_bytes,
    )


def system_available_memory_bytes() -> int:
    """Return Linux MemAvailable, with a portable page-count fallback."""

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    fields = line.split()
                    if len(fields) >= 2:
                        return int(fields[1]) * 1024
    except (OSError, ValueError):
        pass

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise FeatureReplayMemoryError(
            "cannot determine available system memory; pass available_bytes"
        ) from exc
    if pages <= 0 or page_size <= 0:
        raise FeatureReplayMemoryError(
            "system reported invalid available-memory page counts"
        )
    return pages * page_size


def preflight_feature_replay_memory(
    *,
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
    intervention_capacity: int = DEFAULT_INTERVENTION_CAPACITY,
    available_bytes: int | None = None,
    reserve_bytes: int = DEFAULT_MEMORY_RESERVE_BYTES,
) -> FeatureReplayMemoryEstimate:
    """Fail before allocation unless fixed tensors leave the requested RAM."""

    estimate = estimate_feature_replay_memory(
        replay_capacity=replay_capacity,
        intervention_capacity=intervention_capacity,
    )
    reserve = _nonnegative_int(reserve_bytes, name="reserve_bytes")
    available = (
        system_available_memory_bytes()
        if available_bytes is None
        else _nonnegative_int(available_bytes, name="available_bytes")
    )
    required = estimate.fixed_tensor_bytes + reserve
    if available < required:
        raise FeatureReplayMemoryError(
            "feature replay needs "
            f"{estimate.fixed_tensor_gib:.3f} GiB fixed tensors plus "
            f"{reserve / 1024**3:.3f} GiB reserve, but only "
            f"{available / 1024**3:.3f} GiB is available"
        )
    return estimate


def _strict_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] = FEATURE_MAP_DTYPE,
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise FeatureReplayError(f"{name} is not array-like") from exc
    if array.dtype != dtype:
        raise FeatureReplayError(
            f"{name} must have dtype {dtype.name}, got {array.dtype}"
        )
    if array.shape != shape:
        raise FeatureReplayError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FeatureReplayError(f"{name} contains non-finite data")
    if array.ndim == 0:
        # np.ascontiguousarray promotes a scalar to shape (1,), which both
        # weakens this strict contract and causes deprecated array-to-scalar
        # assignments in the fixed ring.
        return array.copy()
    return np.ascontiguousarray(array)


def _strict_scalar(value: Any, *, name: str) -> np.float32:
    array = _strict_array(value, name=name, shape=())
    return np.float32(array.item())


def _validated_feature_transition(
    transition: Mapping[str, Any], *, expected_grasp_penalty: float
) -> dict[str, Any]:
    expected_keys = {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "masks",
        "grasp_penalty",
    }
    if not isinstance(transition, Mapping) or set(transition) != expected_keys:
        raise FeatureReplayError(
            "feature transition must contain exactly observations, "
            "next_observations, actions, rewards, masks, and grasp_penalty"
        )

    observations: dict[str, dict[str, np.ndarray]] = {}
    expected_observation_keys = {"state", *FEATURE_KEYS}
    for tree_name in ("observations", "next_observations"):
        source = transition[tree_name]
        if not isinstance(source, Mapping) or set(source) != expected_observation_keys:
            raise FeatureReplayError(
                f"{tree_name} must contain exactly state, cam1, and cam2"
            )
        observations[tree_name] = {
            "state": _strict_array(
                source["state"],
                name=f"{tree_name}.state",
                shape=_STATE_SHAPE,
            ),
            **{
                key: _strict_array(
                    source[key],
                    name=f"{tree_name}.{key}",
                    shape=FEATURE_MAP_SHAPE,
                )
                for key in FEATURE_KEYS
            },
        }

    actions = _strict_array(
        transition["actions"], name="actions", shape=_ACTION_SHAPE
    )
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        raise FeatureReplayError("actions must be within [-1, 1]")
    if float(actions[-1]) not in (-1.0, 0.0, 1.0):
        raise FeatureReplayError("gripper action must be in {-1, 0, 1}")
    reward = _strict_scalar(transition["rewards"], name="rewards")
    mask = _strict_scalar(transition["masks"], name="masks")
    penalty = _strict_scalar(
        transition["grasp_penalty"], name="grasp_penalty"
    )
    if float(reward) not in (0.0, 1.0):
        raise FeatureReplayError("rewards must use the binary 0/1 contract")
    if float(mask) not in (0.0, 1.0):
        raise FeatureReplayError("masks must be 0 or 1")
    if not (
        math.isclose(float(penalty), 0.0, rel_tol=0.0, abs_tol=1e-7)
        or math.isclose(
            float(penalty),
            expected_grasp_penalty,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    ):
        raise FeatureReplayError(
            "grasp_penalty must be either 0 or the configured penalty "
            f"{expected_grasp_penalty}"
        )
    return {
        "observations": observations["observations"],
        "next_observations": observations["next_observations"],
        "actions": actions,
        "rewards": reward,
        "masks": mask,
        "grasp_penalty": penalty,
    }


class FeatureTransitionRing:
    """Fixed-capacity, seeded ring containing only learner feature tensors."""

    observation_representation = FROZEN_TRUNK_REPRESENTATION
    augmentation = LEARNER_AUGMENTATION

    def __init__(
        self,
        capacity: int,
        *,
        seed: int = 42,
        expected_grasp_penalty: float = -0.02,
    ) -> None:
        self.capacity = _positive_int(capacity, name="capacity")
        seed_value = _nonnegative_int(seed, name="seed")
        self.expected_grasp_penalty = _expected_penalty(
            expected_grasp_penalty
        )
        self._observations = {
            "state": np.empty((self.capacity, *_STATE_SHAPE), np.float32),
            **{
                key: np.empty(
                    (self.capacity, *FEATURE_MAP_SHAPE), np.float32
                )
                for key in FEATURE_KEYS
            },
        }
        self._next_observations = {
            "state": np.empty((self.capacity, *_STATE_SHAPE), np.float32),
            **{
                key: np.empty(
                    (self.capacity, *FEATURE_MAP_SHAPE), np.float32
                )
                for key in FEATURE_KEYS
            },
        }
        self._actions = np.empty((self.capacity, *_ACTION_SHAPE), np.float32)
        self._rewards = np.empty((self.capacity,), np.float32)
        self._masks = np.empty((self.capacity,), np.float32)
        self._grasp_penalty = np.empty((self.capacity,), np.float32)
        self._size = 0
        self._insert_index = 0
        self._insert_count = 0
        self._overwrite_count = 0
        self._rng = np.random.default_rng(seed_value)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def insert_count(self) -> int:
        with self._lock:
            return self._insert_count

    @property
    def overwrite_count(self) -> int:
        with self._lock:
            return self._overwrite_count

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

    def insert(self, transition: Mapping[str, Any]) -> bool:
        """Insert one validated feature transition and report overwrite."""

        item = _validated_feature_transition(
            transition,
            expected_grasp_penalty=self.expected_grasp_penalty,
        )
        with self._lock:
            index = self._insert_index
            overwrote = self._size == self.capacity
            for key in ("state", *FEATURE_KEYS):
                self._observations[key][index] = item["observations"][key]
                self._next_observations[key][index] = item[
                    "next_observations"
                ][key]
            self._actions[index] = item["actions"]
            self._rewards[index] = item["rewards"]
            self._masks[index] = item["masks"]
            self._grasp_penalty[index] = item["grasp_penalty"]
            self._insert_index = (index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            self._insert_count += 1
            if overwrote:
                self._overwrite_count += 1
            return overwrote

    def snapshot(self) -> dict[str, Any]:
        """Return the filled entries, oldest first, as owned copies.

        Only ``_size`` rows are returned.  The rings are preallocated at full
        capacity, so dumping the raw arrays would write 7.3 GiB for the
        production 50k/10k pair regardless of how little the run collected.

        ``_insert_index`` is where the *next* write lands, which after the ring
        wraps is also the oldest live row.  Stored order is therefore rotated
        with respect to insertion order.  Sampling would not notice -- it draws
        uniformly over ``_size`` -- but an artifact named "the replay buffer"
        should be readable as the trajectory log it looks like, so the rotation
        is undone here rather than left for every future reader to rediscover.
        """

        with self._lock:
            size = self._size
            if size < self.capacity:
                order = np.arange(size)
            else:
                order = (np.arange(size) + self._insert_index) % self.capacity
            return {
                "size": size,
                "capacity": self.capacity,
                "insert_count": self._insert_count,
                "overwrite_count": self._overwrite_count,
                "expected_grasp_penalty": self.expected_grasp_penalty,
                "observations": {
                    key: value[order].copy()
                    for key, value in self._observations.items()
                },
                "next_observations": {
                    key: value[order].copy()
                    for key, value in self._next_observations.items()
                },
                "actions": self._actions[order].copy(),
                "rewards": self._rewards[order].copy(),
                "masks": self._masks[order].copy(),
                "grasp_penalty": self._grasp_penalty[order].copy(),
            }

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample with replacement using the ring-owned deterministic RNG."""

        size = _positive_int(batch_size, name="batch_size")
        with self._lock:
            if self._size == 0:
                raise FeatureReplayError("cannot sample an empty feature ring")
            indices = self._rng.integers(self._size, size=size)
            return {
                "observations": {
                    key: value[indices].copy()
                    for key, value in self._observations.items()
                },
                "next_observations": {
                    key: value[indices].copy()
                    for key, value in self._next_observations.items()
                },
                "actions": self._actions[indices].copy(),
                "rewards": self._rewards[indices].copy(),
                "masks": self._masks[indices].copy(),
                "grasp_penalty": self._grasp_penalty[indices].copy(),
            }


@dataclass
class _RouteState:
    signature: bytes
    replay_inserted: bool = False
    intervention_inserted: bool = False
    completed: bool = False


class FeatureReplayIngress:
    """Idempotently encode finalized raw transitions into two feature rings."""

    require_grasp_penalty = True
    feature_encoding_id = FEATURE_ENCODING_ID
    observation_representation = FEATURE_ENCODING_ID
    augmentation = FEATURE_AUGMENTATION

    def __init__(
        self,
        *,
        feature_extractor: FrozenTrunkFeatureExtractor,
        replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
        intervention_capacity: int = DEFAULT_INTERVENTION_CAPACITY,
        seed: int = 42,
        ledger_capacity: int | None = None,
        expected_grasp_penalty: float = -0.02,
        preflight_memory: bool = True,
        available_memory_bytes: int | None = None,
        memory_reserve_bytes: int = DEFAULT_MEMORY_RESERVE_BYTES,
    ) -> None:
        if not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")
        if not isinstance(preflight_memory, bool):
            raise ValueError("preflight_memory must be bool")
        self.feature_extractor = feature_extractor
        self.replay_capacity = _positive_int(
            replay_capacity, name="replay_capacity"
        )
        self.intervention_capacity = _positive_int(
            intervention_capacity, name="intervention_capacity"
        )
        seed_value = _nonnegative_int(seed, name="seed")
        self.expected_grasp_penalty = _expected_penalty(
            expected_grasp_penalty
        )
        self.memory_estimate = estimate_feature_replay_memory(
            replay_capacity=self.replay_capacity,
            intervention_capacity=self.intervention_capacity,
        )
        if preflight_memory:
            self.memory_estimate = preflight_feature_replay_memory(
                replay_capacity=self.replay_capacity,
                intervention_capacity=self.intervention_capacity,
                available_bytes=available_memory_bytes,
                reserve_bytes=memory_reserve_bytes,
            )
        elif available_memory_bytes is not None:
            raise ValueError(
                "available_memory_bytes requires preflight_memory=True"
            )

        self.replay_store = FeatureTransitionRing(
            self.replay_capacity,
            seed=seed_value,
            expected_grasp_penalty=self.expected_grasp_penalty,
        )
        self.intervention_store = FeatureTransitionRing(
            self.intervention_capacity,
            seed=seed_value + 1,
            expected_grasp_penalty=self.expected_grasp_penalty,
        )
        default_ledger_capacity = max(
            self.replay_capacity, self.intervention_capacity
        )
        self._ledger_capacity = _positive_int(
            ledger_capacity or default_ledger_capacity,
            name="ledger_capacity",
        )
        self._ledger: OrderedDict[str, _RouteState] = OrderedDict()
        self._replay_sidecar: deque[IngressRecord] = deque(
            maxlen=self.replay_capacity
        )
        self._intervention_sidecar: deque[IngressRecord] = deque(
            maxlen=self.intervention_capacity
        )
        self._last_transition_id = ""
        self._last_env_step: int | None = None
        self._feature_cache: OrderedDict[
            tuple[str, str, str], dict[str, np.ndarray]
        ] = OrderedDict()
        #: Diagnostics for the O(t) reuse above.  ``trunk_extractions`` counts
        #: real frozen-trunk forwards; it is the number Stage-1 instrumentation
        #: needs to tell "one slow forward" apart from "several forwards".
        self.trunk_extractions = 0
        self.observation_cache_hits = 0
        self._lock = threading.RLock()

    def __call__(self, data: dict[str, Any], intervened: bool) -> None:
        record, raw_transition = ReplayIngress.validate_data(
            data,
            intervened=intervened,
            require_grasp_penalty=True,
            expected_grasp_penalty=self.expected_grasp_penalty,
        )
        signature = ReplayIngress.fingerprint(record, raw_transition)
        with self._lock:
            route = self._ledger.get(record.transition_id)
            if route is not None:
                if route.signature != signature:
                    raise ActorProtocolError(
                        "transition_id collision for "
                        f"{record.transition_id!r}"
                    )
                if route.completed:
                    self._ledger.move_to_end(record.transition_id)
                    return

            encoded = self._encode(raw_transition, record)
            if route is None:
                # Do not evict an idempotency record until extraction has
                # succeeded.  Extractor faults therefore leave the ledger
                # unchanged and retain no raw observation references.
                self._make_ledger_room()
                route = _RouteState(signature=signature)
                self._ledger[record.transition_id] = route

            if not route.replay_inserted:
                self.replay_store.insert(encoded)
                route.replay_inserted = True
                self._replay_sidecar.append(record)

            if intervened and not route.intervention_inserted:
                self.intervention_store.insert(encoded)
                route.intervention_inserted = True
                self._intervention_sidecar.append(record)

            route.completed = route.replay_inserted and (
                route.intervention_inserted or not intervened
            )
            if not route.completed:
                raise FeatureReplayError("feature transition routing did not complete")
            self._last_transition_id = record.transition_id
            self._last_env_step = record.env_step
            self._ledger.move_to_end(record.transition_id)

    def status(self) -> BufferStatus:
        with self._lock:
            return BufferStatus(
                replay_size=len(self.replay_store),
                replay_capacity=self.replay_capacity,
                intervention_size=len(self.intervention_store),
                intervention_capacity=self.intervention_capacity,
                replay_insert_count=self.replay_store.insert_count,
                intervention_insert_count=self.intervention_store.insert_count,
                replay_overwrite_count=self.replay_store.overwrite_count,
                intervention_overwrite_count=(
                    self.intervention_store.overwrite_count
                ),
                last_transition_id=self._last_transition_id,
                last_env_step=self._last_env_step,
            )

    def sample_replay(self, batch_size: int, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            raise ValueError(
                "feature replay sampling does not support packed-image options"
            )
        with self._lock:
            return self.replay_store.sample(batch_size)

    def sample_intervention(
        self, batch_size: int, **kwargs: Any
    ) -> dict[str, Any]:
        if kwargs:
            raise ValueError(
                "feature replay sampling does not support packed-image options"
            )
        with self._lock:
            return self.intervention_store.sample(batch_size)

    def replay_sidecar(self) -> tuple[IngressRecord, ...]:
        with self._lock:
            return tuple(self._replay_sidecar)

    def intervention_sidecar(self) -> tuple[IngressRecord, ...]:
        with self._lock:
            return tuple(self._intervention_sidecar)

    def prime_observation(
        self,
        *,
        actor_id: str,
        session_id: str,
        observation_id: str,
        observation: Mapping[str, np.ndarray],
    ) -> None:
        """Encode an episode's first observation before any transition needs it.

        ``BeginEpisode``'s observation never reaches ``__call__``, so without
        this the episode's first transition is the one step that pays for two
        trunk forwards instead of one.  Priming does not move that work off the
        GPU -- the per-episode total is unchanged -- it moves it out of a step
        inside the control loop and onto the episode boundary, where the
        operator is repositioning the scene anyway.

        Nothing is inserted into either ring, so a failure here leaves no
        partial route.  It is still raised rather than swallowed: an extractor
        that cannot encode O(0) would fail the first transition regardless, and
        failing at the boundary is both earlier and louder.
        """

        if not isinstance(observation_id, str) or not observation_id:
            raise ActorProtocolError("observation_id is required")
        key = self._cache_key(str(actor_id), str(session_id), observation_id)
        with self._lock:
            return self._cached_or_encoded(
                key, observation, name="observations"
            )

    def _encode(
        self, raw_transition: Mapping[str, Any], record: IngressRecord
    ) -> dict[str, Any]:
        # BOTH sides come through the cache.  O(t) is the previous
        # transition's O(t+1); O(t+1) is whatever the service encoded before
        # it called us.  When the service pre-encodes -- the production path
        # once encoding moved to the top of step() -- the trunk runs zero
        # times here, and the lock this ingress shares with the learner no
        # longer contains a GPU forward at all.
        observations = self._cached_or_encoded(
            self._cache_key(
                record.actor_id, record.session_id, record.observation_id
            ),
            raw_transition["observations"],
            name="observations",
        )
        next_observations = self._cached_or_encoded(
            self._cache_key(
                record.actor_id, record.session_id, record.next_observation_id
            ),
            raw_transition["next_observations"],
            name="next_observations",
        )
        return {
            "observations": observations,
            "next_observations": next_observations,
            "actions": raw_transition["actions"],
            "rewards": raw_transition["rewards"],
            "masks": raw_transition["masks"],
            "grasp_penalty": raw_transition["grasp_penalty"],
        }

    def _cached_or_encoded(
        self,
        key: tuple[str, str, str],
        raw_observation: Mapping[str, np.ndarray],
        *,
        name: str,
    ) -> dict[str, np.ndarray]:
        """Return this observation's trunk output, computing it only if new.

        The returned mapping is handed to ``insert`` (or to the policy) just
        like a fresh extraction, so it must never alias the cache entry that a
        later transition may still read.
        """

        cached = self._feature_cache.get(key)
        if cached is not None:
            # Cheap tripwire.  Identical ids are supposed to mean identical
            # pixels, so a divergent state means an observation_id was reused
            # for different content -- exactly what the skipped extractor-side
            # state check would have caught.  Comparing 19 floats keeps that
            # guarantee without re-hashing ~96 KiB of images.
            if not np.array_equal(cached["state"], raw_observation["state"]):
                raise ActorProtocolError(
                    f"observation_id {key[2]!r} was reused for a different state"
                )
            self._feature_cache.move_to_end(key)
            self.observation_cache_hits += 1
            return {name_: value.copy() for name_, value in cached.items()}
        encoded = self._encode_observation(raw_observation, name=name)
        # Store only after extraction succeeded, so a faulted transition
        # leaves no state behind -- the same rule the ledger uses.
        self._cache_store(key, encoded)
        return encoded

    def _cache_store(
        self,
        key: tuple[str, str, str],
        encoded: Mapping[str, np.ndarray],
    ) -> None:
        self._feature_cache[key] = {
            name: value.copy() for name, value in encoded.items()
        }
        self._feature_cache.move_to_end(key)
        while len(self._feature_cache) > FEATURE_CACHE_CAPACITY:
            self._feature_cache.popitem(last=False)

    @staticmethod
    def _cache_key(
        actor_id: str, session_id: str, observation_id: str
    ) -> tuple[str, str, str]:
        # observation_id is only unique within one (actor, session); the
        # service scopes its own accepted-observation set the same way.
        return (actor_id, session_id, observation_id)

    def _encode_observation(
        self, observation: Mapping[str, np.ndarray], *, name: str
    ) -> dict[str, np.ndarray]:
        self.trunk_extractions += 1
        try:
            features = self.feature_extractor(observation)
        except Exception as exc:
            raise FeatureExtractionError(
                f"frozen-trunk extraction failed for {name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        expected_keys = {"state", *FEATURE_KEYS}
        if not isinstance(features, Mapping) or set(features) != expected_keys:
            raise FeatureExtractionError(
                f"{name} extractor output must contain exactly state, cam1, "
                "and cam2"
            )
        validated: dict[str, np.ndarray] = {}
        for key in FEATURE_KEYS:
            try:
                value = _strict_array(
                    features[key],
                    name=f"{name}.{key}",
                    shape=FEATURE_MAP_SHAPE,
                )
            except FeatureReplayError as exc:
                raise FeatureExtractionError(str(exc)) from exc
            validated[key] = value.copy()
        try:
            state = _strict_array(
                features["state"],
                name=f"{name}.state",
                shape=_STATE_SHAPE,
            )
        except FeatureReplayError as exc:
            raise FeatureExtractionError(str(exc)) from exc
        if not np.array_equal(state, observation["state"]):
            raise FeatureExtractionError(
                f"{name}.state must equal the canonical raw state exactly"
            )
        validated["state"] = state.copy()
        return validated

    def _make_ledger_room(self) -> None:
        while len(self._ledger) >= self._ledger_capacity:
            removable = next(
                (
                    key
                    for key, state in self._ledger.items()
                    if state.completed
                ),
                None,
            )
            if removable is None:
                raise FeatureReplayError(
                    "idempotency ledger is full of incomplete routes"
                )
            self._ledger.pop(removable)
