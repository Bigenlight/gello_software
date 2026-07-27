"""RLPD 50:50 sampling and learner-only batch sanitation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol

import numpy as np

from ur_env.learner.demo import LEARNER_BATCH_KEYS


class LearnerBatchError(ValueError):
    """A replay source returned data outside the learner tensor contract."""


class SampleSource(Protocol):
    def __len__(self) -> int:
        ...

    def sample(self, batch_size: int) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class SamplingMetrics:
    replay_size: int
    offline_demo_size: int
    online_intervention_size: int
    replay_batch_size: int
    demo_batch_size: int
    offline_demo_batch_size: int
    online_intervention_batch_size: int

    @property
    def intervention_ratio(self) -> float:
        if self.replay_size == 0:
            return 0.0
        return self.online_intervention_size / self.replay_size


class ReplayIngressView:
    """Expose one route of ``ReplayIngress`` as a sampler source."""

    def __init__(self, ingress: Any, route: str) -> None:
        if route not in {"replay", "intervention"}:
            raise ValueError("route must be replay or intervention")
        self._ingress = ingress
        self._route = route

    def __len__(self) -> int:
        status = self._ingress.status()
        return int(
            status.replay_size
            if self._route == "replay"
            else status.intervention_size
        )

    def sample(self, batch_size: int) -> Mapping[str, Any]:
        if self._route == "replay":
            return self._ingress.sample_replay(batch_size=batch_size)
        return self._ingress.sample_intervention(batch_size=batch_size)


def _array(value: Any, *, name: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except Exception as exc:
        raise LearnerBatchError(f"{name} is not array-like") from exc


def sanitize_learner_batch(
    batch: Mapping[str, Any], *, expected_batch_size: int | None = None
) -> dict[str, Any]:
    """Drop sidecar fields and validate the packed upstream batch layout."""

    if not isinstance(batch, Mapping):
        raise LearnerBatchError("batch must be a mapping")
    missing = [key for key in LEARNER_BATCH_KEYS if key not in batch]
    if missing:
        raise LearnerBatchError(f"learner batch is missing fields: {missing}")
    if "has_grasp_penalty" in batch:
        present = _array(batch["has_grasp_penalty"], name="has_grasp_penalty")
        if not np.all(present == 1):
            raise LearnerBatchError(
                "learner batch contains a transition without grasp_penalty"
            )

    observations = batch["observations"]
    next_observations = batch["next_observations"]
    if not isinstance(observations, Mapping) or not isinstance(
        next_observations, Mapping
    ):
        raise LearnerBatchError("observations and next_observations must be mappings")
    if set(observations) != {"state", "cam1", "cam2"}:
        raise LearnerBatchError(
            "packed observations must contain exactly state, cam1, and cam2"
        )
    if set(next_observations) != {"state"}:
        raise LearnerBatchError(
            "packed next_observations must contain exactly state"
        )

    actions = _array(batch["actions"], name="actions")
    batch_size = int(actions.shape[0]) if actions.ndim else -1
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise LearnerBatchError(
            f"expected batch size {expected_batch_size}, got {batch_size}"
        )
    expected_shapes = {
        "state": (batch_size, 1, 19),
        "cam1": (batch_size, 2, 128, 128, 3),
        "cam2": (batch_size, 2, 128, 128, 3),
    }
    for key, shape in expected_shapes.items():
        array = _array(observations[key], name=f"observations.{key}")
        if array.shape != shape:
            raise LearnerBatchError(
                f"observations.{key} must have shape {shape}, got {array.shape}"
            )
        expected_dtype = np.uint8 if key.startswith("cam") else np.float32
        if array.dtype != np.dtype(expected_dtype):
            raise LearnerBatchError(
                f"observations.{key} must have dtype "
                f"{np.dtype(expected_dtype).name}, got {array.dtype}"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise LearnerBatchError(f"observations.{key} contains non-finite data")

    next_state = _array(next_observations["state"], name="next_observations.state")
    if next_state.shape != (batch_size, 1, 19) or next_state.dtype != np.float32:
        raise LearnerBatchError(
            "next_observations.state must have shape (B, 1, 19) and dtype float32"
        )
    if not np.all(np.isfinite(next_state)):
        raise LearnerBatchError("next_observations.state contains non-finite data")

    if actions.shape != (batch_size, 7) or actions.dtype != np.float32:
        raise LearnerBatchError("actions must have shape (B, 7) and dtype float32")
    if not np.all(np.isfinite(actions)) or np.any(actions < -1.0) or np.any(
        actions > 1.0
    ):
        raise LearnerBatchError("actions must be finite and within [-1, 1]")
    if not np.all(np.isin(actions[:, -1], (-1.0, 0.0, 1.0))):
        raise LearnerBatchError("gripper actions must be in {-1, 0, 1}")

    for key in ("rewards", "masks", "grasp_penalty"):
        array = _array(batch[key], name=key)
        if array.shape != (batch_size,) or array.dtype != np.float32:
            raise LearnerBatchError(
                f"{key} must have shape (B,) and dtype float32"
            )
        if not np.all(np.isfinite(array)):
            raise LearnerBatchError(f"{key} contains non-finite data")
    if not np.all(np.isin(_array(batch["rewards"], name="rewards"), (0.0, 1.0))):
        raise LearnerBatchError("rewards must use the binary 0/1 contract")
    if not np.all(np.isin(_array(batch["masks"], name="masks"), (0.0, 1.0))):
        raise LearnerBatchError("masks must be 0 or 1")

    return {
        "observations": {key: observations[key] for key in ("state", "cam1", "cam2")},
        "next_observations": {"state": next_observations["state"]},
        "actions": batch["actions"],
        "rewards": batch["rewards"],
        "masks": batch["masks"],
        "grasp_penalty": batch["grasp_penalty"],
    }


def _concat(left: Any, right: Any) -> Any:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise LearnerBatchError("cannot concatenate different batch trees")
        return {key: _concat(left[key], right[key]) for key in left}
    return np.concatenate([np.asarray(left), np.asarray(right)], axis=0)


def _permute(tree: Any, order: np.ndarray) -> Any:
    if isinstance(tree, Mapping):
        return {key: _permute(value, order) for key, value in tree.items()}
    return np.asarray(tree)[order]


def proportional_sample_counts(total: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Allocate ``total`` draws using deterministic largest remainders."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a non-negative integer")
    if not sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in sizes
    ):
        raise ValueError("sizes must be non-negative integers")
    population = sum(sizes)
    if total and population == 0:
        raise ValueError("cannot allocate samples from empty sources")
    if population == 0:
        return tuple(0 for _ in sizes)
    exact = [total * size / population for size in sizes]
    counts = [math.floor(value) for value in exact]
    remaining = total - sum(counts)
    order = sorted(
        range(len(sizes)),
        key=lambda index: (exact[index] - counts[index], sizes[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)


class RLPDBatchSampler:
    """Half online replay and half uniformly sampled demonstration union."""

    def __init__(
        self,
        *,
        online_replay: SampleSource,
        offline_demos: SampleSource,
        online_interventions: SampleSource,
        batch_size: int = 256,
        training_starts: int = 100,
        seed: int = 42,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer")
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("batch_size must be a positive even integer")
        if (
            isinstance(training_starts, bool)
            or not isinstance(training_starts, int)
            or training_starts <= 0
        ):
            raise ValueError("training_starts must be positive")
        self.online_replay = online_replay
        self.offline_demos = offline_demos
        self.online_interventions = online_interventions
        self.batch_size = batch_size
        self.training_starts = training_starts
        self._rng = np.random.default_rng(seed)
        self.last_metrics = self.metrics()

    @property
    def ready(self) -> bool:
        return (
            len(self.online_replay) >= self.training_starts
            and len(self.offline_demos) > 0
        )

    def metrics(self) -> SamplingMetrics:
        replay_size = len(self.online_replay)
        offline_size = len(self.offline_demos)
        intervention_size = len(self.online_interventions)
        half = self.batch_size // 2
        offline_count, intervention_count = proportional_sample_counts(
            half, (offline_size, intervention_size)
        ) if offline_size + intervention_size else (0, 0)
        return SamplingMetrics(
            replay_size=replay_size,
            offline_demo_size=offline_size,
            online_intervention_size=intervention_size,
            replay_batch_size=half,
            demo_batch_size=half,
            offline_demo_batch_size=offline_count,
            online_intervention_batch_size=intervention_count,
        )

    def sample(self) -> dict[str, Any]:
        if not self.ready:
            raise LearnerBatchError(
                "training requires at least "
                f"{self.training_starts} online replay transitions and one "
                "offline demonstration"
            )
        metrics = self.metrics()
        replay = sanitize_learner_batch(
            self.online_replay.sample(metrics.replay_batch_size),
            expected_batch_size=metrics.replay_batch_size,
        )
        demo_parts: list[dict[str, Any]] = []
        if metrics.offline_demo_batch_size:
            demo_parts.append(
                sanitize_learner_batch(
                    self.offline_demos.sample(metrics.offline_demo_batch_size),
                    expected_batch_size=metrics.offline_demo_batch_size,
                )
            )
        if metrics.online_intervention_batch_size:
            demo_parts.append(
                sanitize_learner_batch(
                    self.online_interventions.sample(
                        metrics.online_intervention_batch_size
                    ),
                    expected_batch_size=metrics.online_intervention_batch_size,
                )
            )
        demos = demo_parts[0]
        for part in demo_parts[1:]:
            demos = _concat(demos, part)
        demos = _permute(demos, self._rng.permutation(metrics.demo_batch_size))
        result = _concat(replay, demos)
        self.last_metrics = metrics
        return sanitize_learner_batch(result, expected_batch_size=self.batch_size)


def freeze_for_agent(batch: Mapping[str, Any]) -> Any:
    """Convert a validated numpy tree to the FrozenDict expected upstream."""

    from flax.core import freeze

    return freeze(batch)
