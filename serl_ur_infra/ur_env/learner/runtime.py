"""Fault-isolated local CTA learner loop for upstream hybrid SAC."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Mapping, Optional

import numpy as np

from ur_env.learner.batches import RLPDBatchSampler, freeze_for_agent
from ur_env.learner.checkpoint import CheckpointManager, LearnerFingerprint
from ur_env.learner.config import LearnerConfig
from ur_env.learner.policy import PolicyPublisher, validate_tree_finite


CRITIC_NETWORKS = frozenset({"critic", "grasp_critic"})
ALL_NETWORKS = frozenset({"critic", "grasp_critic", "actor", "temperature"})


class LearnerNotReadyError(RuntimeError):
    """Replay/demo startup conditions are not yet satisfied."""


class LearnerFaultError(RuntimeError):
    """The learner stopped after an update, publish, checkpoint, or log fault."""


@dataclass(frozen=True)
class LearnerFault:
    learner_step: int
    gradient_step: int
    error_type: str
    detail: str


@dataclass(frozen=True)
class LearnerStepResult:
    learner_step: int
    gradient_step: int
    policy_version: int
    published: bool
    checkpoint_path: str | None
    metrics: Mapping[str, float]


def _flatten_scalars(value: Any, *, prefix: str = "") -> dict[str, float]:
    import jax

    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_scalars(item, prefix=child))
        return result
    array = np.asarray(jax.device_get(value))
    if array.shape != ():
        return result
    scalar = float(array)
    if not math.isfinite(scalar):
        raise LearnerFaultError(f"non-finite update metric: {prefix}")
    result[prefix] = scalar
    return result


def _block_tree(tree: Any) -> Any:
    import jax

    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


class HILSERLLearner:
    """Run one critic-only update followed by one all-network update.

    Any exception transitions only this object to a permanent fault state.
    ``PolicyPublisher`` is updated only after a validated 50-step boundary, so
    inference continues serving its previous immutable snapshot.
    """

    def __init__(
        self,
        *,
        agent: Any,
        sampler: RLPDBatchSampler,
        publisher: PolicyPublisher,
        config: LearnerConfig = LearnerConfig(),
        logger: Any | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        fingerprint: LearnerFingerprint | None = None,
        learner_step: int = 0,
        gradient_step: int | None = None,
        policy_version: int = 0,
    ) -> None:
        self.agent = agent
        self.sampler = sampler
        self.publisher = publisher
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.fingerprint = fingerprint
        self.learner_step = self._counter(learner_step, "learner_step")
        state_step = int(np.asarray(agent.state.step))
        self.gradient_step = self._counter(
            state_step if gradient_step is None else gradient_step,
            "gradient_step",
        )
        if self.gradient_step != state_step:
            raise ValueError("gradient_step must match agent.state.step")
        expected_gradient_step = self.learner_step * self.config.cta_ratio
        if self.gradient_step != expected_gradient_step:
            raise ValueError(
                "gradient_step must equal learner_step * cta_ratio"
            )
        self.policy_version = self._counter(policy_version, "policy_version")
        self._fault: LearnerFault | None = None
        self._lock = threading.Lock()
        validate_tree_finite(agent.state, name="initial agent state")

    @staticmethod
    def _counter(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    @property
    def fault(self) -> LearnerFault | None:
        with self._lock:
            return self._fault

    @property
    def ready(self) -> bool:
        return self.fault is None and self.sampler.ready

    def _set_fault(self, exc: BaseException) -> LearnerFaultError:
        detail = f"{type(exc).__name__}: {exc}"
        fault = LearnerFault(
            learner_step=self.learner_step,
            gradient_step=self.gradient_step,
            error_type=type(exc).__name__,
            detail=detail[:2_000],
        )
        with self._lock:
            if self._fault is None:
                self._fault = fault
            else:
                fault = self._fault
        if self.logger is not None:
            try:
                self.logger.log(
                    "learner_fault",
                    learner_step=fault.learner_step,
                    gradient_step=fault.gradient_step,
                    error_type=fault.error_type,
                    detail=fault.detail,
                    policy_version=self.policy_version,
                )
            except Exception:
                pass
        return LearnerFaultError(
            "learner entered fault state; last known good policy remains active: "
            f"{fault.detail}"
        )

    def enter_fault(self, exc: BaseException) -> LearnerFaultError:
        """Permanently fault this learner after a supervisor-side failure.

        ``train_once`` already converts update, publish, checkpoint, and log
        failures into the learner-only fault state.  A production worker also
        evaluates readiness outside that method, where a failed ingress can
        raise before an update starts.  This public boundary lets the worker
        record that failure with exactly the same last-known-good semantics.
        """

        if not isinstance(exc, BaseException):
            raise TypeError("exc must be an exception")
        existing = self.fault
        if existing is not None:
            return LearnerFaultError(
                "learner entered fault state; last known good policy remains "
                f"active: {existing.detail}"
            )
        return self._set_fault(exc)

    def _update(self, batch: Any, networks: frozenset[str]) -> tuple[Any, Any]:
        candidate, info = self.agent.update(
            batch,
            networks_to_update=networks,
        )
        _block_tree((candidate, info))
        validate_tree_finite(candidate.state, name="updated agent state")
        _flatten_scalars(info)
        return candidate, info

    def train_once(self) -> LearnerStepResult:
        existing_fault = self.fault
        if existing_fault is not None:
            raise LearnerFaultError(
                f"learner is faulted: {existing_fault.detail}"
            )
        if not self.sampler.ready:
            metrics = self.sampler.metrics()
            raise LearnerNotReadyError(
                "learner is waiting for replay/demo startup conditions: "
                f"replay={metrics.replay_size}/{self.config.training_starts}, "
                f"offline_demo={metrics.offline_demo_size}"
            )

        started = time.perf_counter()
        sample_ms = 0.0
        critic_ms = 0.0
        train_ms = 0.0
        critic_info: Any = {}
        update_info: Any = {}
        try:
            sample_started = time.perf_counter()
            critic_batch = freeze_for_agent(self.sampler.sample())
            sample_ms += (time.perf_counter() - sample_started) * 1000.0
            update_started = time.perf_counter()
            candidate, critic_info = self._update(
                critic_batch, CRITIC_NETWORKS
            )
            critic_ms = (time.perf_counter() - update_started) * 1000.0
            self.agent = candidate
            self.gradient_step = int(np.asarray(self.agent.state.step))

            sample_started = time.perf_counter()
            full_batch = freeze_for_agent(self.sampler.sample())
            sample_ms += (time.perf_counter() - sample_started) * 1000.0
            update_started = time.perf_counter()
            candidate, update_info = self._update(full_batch, ALL_NETWORKS)
            train_ms = (time.perf_counter() - update_started) * 1000.0
            self.agent = candidate
            self.gradient_step = int(np.asarray(self.agent.state.step))
            self.learner_step += 1

            expected_gradient_step = self.learner_step * self.config.cta_ratio
            if self.gradient_step != expected_gradient_step:
                raise LearnerFaultError(
                    "gradient_step must advance exactly once per CTA update"
                )

            published = False
            if self.learner_step % self.config.publish_period == 0:
                self.policy_version = self.publisher.publish(
                    self.agent.state.params, self.learner_step
                )
                published = True
                if self.logger is not None:
                    self.logger.log(
                        "policy_published",
                        learner_step=self.learner_step,
                        gradient_step=self.gradient_step,
                        policy_version=self.policy_version,
                    )

            checkpoint_path: str | None = None
            if self.learner_step % self.config.checkpoint_period == 0:
                if not published:
                    raise LearnerFaultError(
                        "checkpoint boundary did not publish a policy snapshot"
                    )
                if self.checkpoint_manager is None or self.fingerprint is None:
                    raise LearnerFaultError(
                        "checkpoint manager and fingerprint are required at the "
                        "checkpoint boundary"
                    )
                inference_rng = getattr(self.publisher, "inference_rng", None)
                if inference_rng is None:
                    raise LearnerFaultError(
                        "publisher does not expose checkpointable inference_rng"
                    )
                saved = self.checkpoint_manager.save(
                    agent=self.agent,
                    learner_step=self.learner_step,
                    gradient_step=self.gradient_step,
                    policy_version=self.policy_version,
                    inference_rng=inference_rng,
                    fingerprint=self.fingerprint,
                )
                checkpoint_path = str(saved)
                if self.logger is not None:
                    self.logger.log(
                        "checkpoint_saved",
                        learner_step=self.learner_step,
                        gradient_step=self.gradient_step,
                        policy_version=self.policy_version,
                        checkpoint_path=checkpoint_path,
                        fingerprint_sha256=self.fingerprint.sha256,
                    )

            sampling = self.sampler.last_metrics
            metrics = {
                **{
                    f"critic/{key}": value
                    for key, value in _flatten_scalars(critic_info).items()
                },
                **{
                    f"update/{key}": value
                    for key, value in _flatten_scalars(update_info).items()
                },
                "timing/sample_ms": sample_ms,
                "timing/critic_update_ms": critic_ms,
                "timing/full_update_ms": train_ms,
                "timing/learner_step_ms": (
                    time.perf_counter() - started
                )
                * 1000.0,
                "buffer/replay_size": float(sampling.replay_size),
                "buffer/offline_demo_size": float(sampling.offline_demo_size),
                "buffer/online_intervention_size": float(
                    sampling.online_intervention_size
                ),
                "buffer/intervention_ratio": sampling.intervention_ratio,
            }
            if self.logger is not None and self.learner_step % self.config.log_period == 0:
                self.logger.log(
                    "learner_update",
                    learner_step=self.learner_step,
                    gradient_step=self.gradient_step,
                    policy_version=self.policy_version,
                    metrics=metrics,
                )
            return LearnerStepResult(
                learner_step=self.learner_step,
                gradient_step=self.gradient_step,
                policy_version=self.policy_version,
                published=published,
                checkpoint_path=checkpoint_path,
                metrics=metrics,
            )
        except Exception as exc:
            raise self._set_fault(exc) from exc

    def run(
        self,
        *,
        target_learner_step: int,
        stop_event: Optional[threading.Event] = None,
        poll_interval: float = 0.1,
    ) -> None:
        if target_learner_step < self.learner_step:
            raise ValueError("target_learner_step is behind the current step")
        if not math.isfinite(poll_interval) or poll_interval <= 0.0:
            raise ValueError("poll_interval must be positive and finite")
        while self.learner_step < target_learner_step:
            if stop_event is not None and stop_event.is_set():
                return
            if not self.sampler.ready:
                if self.fault is not None:
                    raise LearnerFaultError(self.fault.detail)
                if stop_event is not None:
                    stop_event.wait(poll_interval)
                else:
                    time.sleep(poll_interval)
                continue
            self.train_once()
