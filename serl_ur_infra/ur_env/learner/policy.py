"""Validated, versioned policy publication with an atomic local runtime."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from ur_env.actor_network import PolicyInferenceError, validate_action
from ur_env.observation_schema import validate_canonical_observation


class PolicyValidationError(ValueError):
    """A candidate snapshot cannot safely replace the serving policy."""


@runtime_checkable
class PolicyPublisher(Protocol):
    def publish(self, params: Any, learner_step: int) -> int:
        """Validate and publish ``params``, returning its policy version."""


@dataclass(frozen=True)
class PolicySnapshot:
    params: Any
    policy_version: int
    learner_step: int


def canonical_policy_observation(value: int = 0) -> dict[str, np.ndarray]:
    if not 0 <= value <= 255:
        raise ValueError("canonical image value must be in [0, 255]")
    return {
        "state": np.zeros((1, 19), dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), value, dtype=np.uint8),
    }


def _leaf_signature(tree: Any) -> tuple[Any, tuple[tuple[tuple[int, ...], str], ...]]:
    import jax

    leaves, structure = jax.tree_util.tree_flatten(tree)
    signatures: list[tuple[tuple[int, ...], str]] = []
    for index, leaf in enumerate(leaves):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            raise PolicyValidationError(
                f"parameter leaf {index} is not a typed array"
            )
        signatures.append(
            (tuple(int(dim) for dim in leaf.shape), np.dtype(leaf.dtype).str)
        )
    if not signatures:
        raise PolicyValidationError("parameter tree must contain array leaves")
    return structure, tuple(signatures)


def validate_tree_finite(tree: Any, *, name: str = "tree") -> None:
    """Synchronously reject non-numeric or non-finite JAX/numpy leaves."""

    import jax
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise PolicyValidationError(f"{name} has no array leaves")
    finite_checks = []
    for index, leaf in enumerate(leaves):
        try:
            dtype = np.dtype(
                leaf.dtype
                if hasattr(leaf, "dtype")
                else np.asarray(leaf).dtype
            )
        except Exception as exc:
            raise PolicyValidationError(
                f"{name} leaf {index} is not a typed array"
            ) from exc
        if dtype.kind not in "biufc":
            raise PolicyValidationError(
                f"{name} leaf {index} has unsupported dtype {dtype}"
            )
        if dtype.kind in "fc":
            finite_checks.append(jnp.all(jnp.isfinite(leaf)))
    if finite_checks:
        all_finite = jnp.all(jnp.stack(finite_checks))
        if not bool(np.asarray(jax.device_get(all_finite))):
            raise PolicyValidationError(f"{name} contains a non-finite value")


def validate_parameter_tree(candidate: Any, reference: Any) -> None:
    expected_structure, expected_leaves = _leaf_signature(reference)
    actual_structure, actual_leaves = _leaf_signature(candidate)
    if actual_structure != expected_structure:
        raise PolicyValidationError("parameter tree structure changed")
    if actual_leaves != expected_leaves:
        raise PolicyValidationError(
            "parameter leaf shape or dtype changed from the initial policy"
        )
    validate_tree_finite(candidate, name="parameter tree")


class VersionedPolicyRuntime:
    """Serve one immutable parameter reference and atomically swap snapshots.

    Candidate validation and smoke inference happen before the lock-protected
    assignment.  Existing inference therefore continues using the last known
    good snapshot while a roughly 32 MB parameter tree is checked, and the
    tree itself is never copied or serialized during publication.
    """

    model_id = "hil-serl-hybrid-sac-resnet10"

    def __init__(
        self,
        agent: Any,
        *,
        params: Any | None = None,
        policy_version: int = 0,
        learner_step: int = 0,
        inference_rng: Any | None = None,
        sample_action: Optional[
            Callable[[Any, Mapping[str, Any], Any, bool], Any]
        ] = None,
    ) -> None:
        if isinstance(policy_version, bool) or not isinstance(policy_version, int):
            raise ValueError("policy_version must be an integer")
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if isinstance(learner_step, bool) or not isinstance(learner_step, int):
            raise ValueError("learner_step must be an integer")
        if learner_step < 0:
            raise ValueError("learner_step must be non-negative")

        import jax

        self._agent = agent
        initial_params = agent.state.params if params is None else params
        self._reference_params = agent.state.params
        self._expected_signature = _leaf_signature(self._reference_params)
        self._sample_action = sample_action or self._sample_with_agent
        self._lock = threading.Lock()
        self._inference_rng = (
            jax.random.PRNGKey(42) if inference_rng is None else inference_rng
        )
        self._validate_rng(self._inference_rng)
        validate_parameter_tree(initial_params, self._reference_params)
        self._smoke(initial_params)
        self._snapshot = PolicySnapshot(
            params=initial_params,
            policy_version=policy_version,
            learner_step=learner_step,
        )

    @staticmethod
    def _validate_rng(value: Any) -> None:
        import jax

        try:
            jax.random.split(value)
        except Exception as exc:
            raise ValueError("inference_rng must be a valid JAX PRNG key") from exc

    def _sample_with_agent(
        self,
        params: Any,
        observation: Mapping[str, Any],
        seed: Any,
        deterministic: bool,
    ) -> Any:
        candidate = self._agent.replace(
            state=self._agent.state.replace(params=params)
        )
        return candidate.sample_actions(
            observations=observation,
            seed=seed,
            argmax=deterministic,
        )

    @staticmethod
    def _validated_policy_action(value: Any, *, name: str) -> np.ndarray:
        import jax

        try:
            array = np.asarray(jax.device_get(value))
        except Exception as exc:
            raise PolicyValidationError(f"{name} could not be materialized") from exc
        if array.dtype != np.dtype(np.float32):
            raise PolicyValidationError(
                f"{name} must have dtype float32, got {array.dtype}"
            )
        try:
            action = validate_action(array, action_shape=(7,), name=name)
        except Exception as exc:
            raise PolicyValidationError(str(exc)) from exc
        if float(action[-1]) not in (-1.0, 0.0, 1.0):
            raise PolicyValidationError(
                f"{name} gripper component must be in {{-1, 0, 1}}"
            )
        return action

    def _smoke(self, params: Any) -> None:
        import jax

        observation = canonical_policy_observation()
        action = self._sample_action(
            params, observation, jax.random.PRNGKey(0), True
        )
        self._validated_policy_action(action, name="policy smoke action")

    @property
    def policy_version(self) -> int:
        with self._lock:
            return self._snapshot.policy_version

    @property
    def learner_step(self) -> int:
        with self._lock:
            return self._snapshot.learner_step

    @property
    def snapshot(self) -> PolicySnapshot:
        with self._lock:
            return self._snapshot

    @property
    def inference_rng(self) -> Any:
        with self._lock:
            return self._inference_rng

    def publish(self, params: Any, learner_step: int) -> int:
        if isinstance(learner_step, bool) or not isinstance(learner_step, int):
            raise PolicyValidationError("learner_step must be an integer")
        validate_parameter_tree(params, self._reference_params)
        self._smoke(params)
        with self._lock:
            if learner_step <= self._snapshot.learner_step:
                raise PolicyValidationError(
                    "learner_step must increase across policy publications"
                )
            version = self._snapshot.policy_version + 1
            self._snapshot = PolicySnapshot(
                params=params,
                policy_version=version,
                learner_step=learner_step,
            )
            return version

    def __call__(
        self, observation: Mapping[str, Any], deterministic: bool
    ) -> tuple[np.ndarray, int]:
        canonical = validate_canonical_observation(observation, copy=False)
        import jax

        with self._lock:
            snapshot = self._snapshot
            self._inference_rng, action_rng = jax.random.split(self._inference_rng)
        try:
            value = self._sample_action(
                snapshot.params, canonical, action_rng, bool(deterministic)
            )
            action = self._validated_policy_action(value, name="policy action")
        except Exception as exc:
            if isinstance(exc, PolicyInferenceError):
                raise
            raise PolicyInferenceError(
                f"policy inference failed: {type(exc).__name__}: {exc}"
            ) from exc
        return action.copy(), snapshot.policy_version
