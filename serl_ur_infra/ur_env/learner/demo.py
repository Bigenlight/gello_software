"""Strict loaders and an in-memory pool for canonical HIL-SERL transitions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
import pickle
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ur_env.observation_schema import validate_canonical_observation


LEARNER_BATCH_KEYS = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "masks",
    "grasp_penalty",
)

# Artifact provenance only.  This value is retained in ``DemoSidecar`` and is
# deliberately excluded from the tensor batch passed to the learner.
SYNTHETIC_ACCEPTANCE_ONLY_KEY = "synthetic_acceptance_only"


class DemoContractError(ValueError):
    """A pickle item is not a canonical learner transition."""


@dataclass(frozen=True)
class DemoSidecar:
    source_path: str
    item_index: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedDemos:
    transitions: tuple[dict[str, Any], ...]
    sidecars: tuple[DemoSidecar, ...]

    def __len__(self) -> int:
        return len(self.transitions)


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if value not in (False, True, 0, 1):
        raise DemoContractError(f"{name} must be bool or 0/1")
    return bool(value)


def _finite_scalar(value: Any, *, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf":
        raise DemoContractError(f"{name} must be a numeric scalar")
    result = float(array)
    if not math.isfinite(result):
        raise DemoContractError(f"{name} must be finite")
    return result


def _looks_like_lerobot(value: Mapping[str, Any]) -> bool:
    suspicious = {
        "joint_positions",
        "joint_position",
        "joint_state",
        "qpos",
        "observation.state",
        "observation.images",
    }
    keys = {str(key).lower() for key in value}
    if keys & suspicious:
        return True
    observation = value.get("observation", value.get("observations"))
    if isinstance(observation, Mapping):
        obs_keys = {str(key).lower() for key in observation}
        return bool(obs_keys & suspicious) or "joints" in obs_keys
    return False


def _normalise_transition(
    item: Any, *, source_path: str, item_index: int
) -> tuple[dict[str, Any], DemoSidecar]:
    if not isinstance(item, Mapping):
        raise DemoContractError(
            f"{source_path}[{item_index}] must be a transition mapping"
        )

    if set(item) == {"meta", "transition"}:
        meta = item["meta"]
        source = item["transition"]
        if not isinstance(meta, Mapping) or not isinstance(source, Mapping):
            raise DemoContractError("actor backup meta/transition must be mappings")
    else:
        meta = {}
        source = item

    provenance_markers: list[tuple[str, bool]] = []
    for marker_source, container in (("meta", meta), ("transition", source)):
        if SYNTHETIC_ACCEPTANCE_ONLY_KEY in container:
            provenance_markers.append(
                (
                    marker_source,
                    _strict_bool(
                        container[SYNTHETIC_ACCEPTANCE_ONLY_KEY],
                        name=(
                            f"{marker_source}."
                            f"{SYNTHETIC_ACCEPTANCE_ONLY_KEY}"
                        ),
                    ),
                )
            )
    synthetic_acceptance_only: bool | None = None
    if provenance_markers:
        synthetic_acceptance_only = provenance_markers[0][1]
        if any(
            value != synthetic_acceptance_only
            for _, value in provenance_markers[1:]
        ):
            raise DemoContractError(
                "meta and transition synthetic_acceptance_only markers "
                "disagree"
            )

    if _looks_like_lerobot(source):
        raise DemoContractError(
            "joint-space/LeRobot demonstrations are not supported; expected "
            "canonical EEF state/cam1/cam2 transitions"
        )

    try:
        observations = validate_canonical_observation(
            source.get("observations"), copy=True
        )
        next_observations = validate_canonical_observation(
            source.get("next_observations"), copy=True
        )
    except Exception as exc:
        raise DemoContractError(
            f"{source_path}[{item_index}] has an invalid canonical observation: {exc}"
        ) from exc

    actions = np.asarray(source.get("actions"))
    if actions.dtype != np.dtype(np.float32):
        raise DemoContractError(
            f"actions must have dtype float32, got {actions.dtype}"
        )
    if actions.shape != (7,):
        raise DemoContractError(f"actions must have shape (7,), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise DemoContractError("actions contain a non-finite value")
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        raise DemoContractError("actions must stay within [-1, 1]")
    if float(actions[-1]) not in (-1.0, 0.0, 1.0):
        raise DemoContractError("gripper action must be exactly one of {-1, 0, 1}")

    reward = _finite_scalar(source.get("rewards"), name="rewards")
    mask = _finite_scalar(source.get("masks"), name="masks")
    if reward not in (0.0, 1.0):
        raise DemoContractError("rewards must use the binary 0/1 contract")
    if mask not in (0.0, 1.0):
        raise DemoContractError("masks must be 0 or 1")

    done: bool | None = None
    if "dones" in source:
        done = _strict_bool(source["dones"], name="dones")
        expected_mask = 0.0 if done else 1.0
        if mask != expected_mask:
            raise DemoContractError(
                f"masks must be {expected_mask} when dones={done}"
            )

    info = source.get("infos", {})
    if info is None:
        info = {}
    if not isinstance(info, Mapping):
        raise DemoContractError("infos must be a mapping when present")
    top_has_penalty = "grasp_penalty" in source
    info_has_penalty = "grasp_penalty" in info
    if not top_has_penalty and not info_has_penalty:
        raise DemoContractError(
            "grasp_penalty is required for learned-gripper learner data"
        )
    top_penalty = (
        _finite_scalar(source["grasp_penalty"], name="grasp_penalty")
        if top_has_penalty
        else None
    )
    info_penalty = (
        _finite_scalar(info["grasp_penalty"], name="infos.grasp_penalty")
        if info_has_penalty
        else None
    )
    if top_penalty is not None and info_penalty is not None:
        if top_penalty != info_penalty:
            raise DemoContractError(
                "top-level and infos grasp_penalty values disagree"
            )
    grasp_penalty = top_penalty if top_penalty is not None else info_penalty
    assert grasp_penalty is not None
    if "has_grasp_penalty" in source and not _strict_bool(
        source["has_grasp_penalty"], name="has_grasp_penalty"
    ):
        raise DemoContractError("has_grasp_penalty must be 1 in learner data")

    labels: list[tuple[str, bool]] = []
    for label_name in ("success", "classifier_success"):
        if label_name in source:
            labels.append(
                (label_name, _strict_bool(source[label_name], name=label_name))
            )
    if labels:
        normalised_success = labels[0][1]
        if any(value != normalised_success for _, value in labels[1:]):
            raise DemoContractError("success and classifier_success disagree")
        if normalised_success != bool(reward):
            raise DemoContractError("success label and rewards disagree")
        if normalised_success and (mask != 0.0 or done is False):
            raise DemoContractError(
                "successful transitions must be terminal with mask 0"
            )
    else:
        normalised_success = bool(reward)

    transition = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": np.ascontiguousarray(actions).copy(),
        "rewards": np.float32(reward),
        "masks": np.float32(mask),
        "grasp_penalty": np.float32(grasp_penalty),
    }

    metadata: dict[str, Any] = {}
    for key, value in meta.items():
        if key == "policy_action":
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool, type(None))):
            metadata[key] = value
    for key in (
        "episode_id",
        "step_id",
        "observation_id",
        "next_observation_id",
        "timestamp_ns",
        "intervened",
        "policy_version",
        "truncated",
        "classifier_probability",
        "classifier_threshold",
        "reward_model_id",
    ):
        if key in source:
            value = source[key]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (str, int, float, bool, type(None))):
                metadata[key] = value
    metadata["success"] = normalised_success
    if synthetic_acceptance_only is not None:
        metadata[SYNTHETIC_ACCEPTANCE_ONLY_KEY] = synthetic_acceptance_only
    if done is not None:
        metadata["done"] = done

    return transition, DemoSidecar(
        source_path=source_path,
        item_index=item_index,
        metadata=metadata,
    )


def load_demo_object(value: Any, *, source_path: str = "<memory>") -> LoadedDemos:
    """Validate a list/tuple of flat transitions or actor ``data`` items."""

    if not isinstance(value, (list, tuple)):
        raise DemoContractError(
            f"{source_path} must contain a list or tuple of transitions"
        )
    transitions: list[dict[str, Any]] = []
    sidecars: list[DemoSidecar] = []
    for index, item in enumerate(value):
        transition, sidecar = _normalise_transition(
            item, source_path=source_path, item_index=index
        )
        transitions.append(transition)
        sidecars.append(sidecar)
    return LoadedDemos(tuple(transitions), tuple(sidecars))


def load_demo_pickle(path: os.PathLike[str] | str) -> LoadedDemos:
    """Load one trusted local pickle and enforce the learner contract."""

    source_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)
    with open(source_path, "rb") as stream:
        value = pickle.load(stream)
    return load_demo_object(value, source_path=source_path)


def load_demo_pickles(paths: Iterable[os.PathLike[str] | str]) -> LoadedDemos:
    transitions: list[dict[str, Any]] = []
    sidecars: list[DemoSidecar] = []
    for path in paths:
        loaded = load_demo_pickle(path)
        transitions.extend(loaded.transitions)
        sidecars.extend(loaded.sidecars)
    return LoadedDemos(tuple(transitions), tuple(sidecars))


class CanonicalTransitionPool:
    """Uniform sampling with replacement from validated canonical data."""

    def __init__(
        self,
        transitions: Sequence[Mapping[str, Any]],
        *,
        seed: int = 42,
    ) -> None:
        self._transitions = tuple(copy.deepcopy(item) for item in transitions)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._transitions)

    def sample(self, batch_size: int, *, packed: bool = True) -> dict[str, Any]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._transitions:
            raise ValueError("cannot sample an empty transition pool")
        indices = self._rng.integers(len(self._transitions), size=batch_size)
        selected = [self._transitions[int(index)] for index in indices]
        observations: dict[str, np.ndarray] = {}
        next_observations: dict[str, np.ndarray] = {}
        for key in ("state", "cam1", "cam2"):
            current = np.stack([item["observations"][key] for item in selected])
            following = np.stack(
                [item["next_observations"][key] for item in selected]
            )
            if packed and key in ("cam1", "cam2"):
                observations[key] = np.concatenate([current, following], axis=1)
            else:
                observations[key] = current
                next_observations[key] = following
        next_observations["state"] = np.stack(
            [item["next_observations"]["state"] for item in selected]
        )
        return {
            "observations": observations,
            "next_observations": next_observations,
            "actions": np.stack([item["actions"] for item in selected]),
            "rewards": np.asarray(
                [item["rewards"] for item in selected], dtype=np.float32
            ),
            "masks": np.asarray(
                [item["masks"] for item in selected], dtype=np.float32
            ),
            "grasp_penalty": np.asarray(
                [item["grasp_penalty"] for item in selected], dtype=np.float32
            ),
        }
