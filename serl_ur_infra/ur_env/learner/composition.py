"""Production composition for the local receive server and CTA learner.

The replay stores are RAM-only and the checked-in protocol has no replay
sampling RPC.  The first production topology must therefore keep gRPC ingress
and exactly one learner worker in the same process.  This module owns that
boundary without deciding whether a future policy publisher is in-process or
remote.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Optional

import numpy as np

from ur_env.learner.batches import RLPDBatchSampler, ReplayIngressView
from ur_env.learner.checkpoint import (
    CheckpointCorruptError,
    CheckpointManager,
    LearnerFingerprint,
    RestoredCheckpoint,
)
from ur_env.learner.config import FROZEN_TRUNK_MODEL_REVISION, LearnerConfig
from ur_env.learner.policy import VersionedPolicyRuntime
from ur_env.learner.runtime import (
    HILSERLLearner,
    LearnerFaultError,
    LearnerNotReadyError,
)
from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH


class LearnerCompositionError(RuntimeError):
    """The production components cannot be assembled without ambiguity."""


@dataclass(frozen=True)
class LearnerAssembly:
    """Objects which must share one process and one policy reference."""

    learner: HILSERLLearner
    policy_runtime: VersionedPolicyRuntime
    sampler: RLPDBatchSampler
    ingress: Any
    restored_checkpoint: RestoredCheckpoint | None


@dataclass(frozen=True)
class PreparedLearnerState:
    """Checkpoint-validated state prepared before server resource allocation."""

    agent: Any
    learner_step: int
    gradient_step: int
    policy_version: int
    inference_rng: Any
    restored_checkpoint: RestoredCheckpoint | None
    config: LearnerConfig
    fingerprint_sha256: str


@dataclass(frozen=True)
class LearnerWorkerStatus:
    state: str
    learner_step: int
    gradient_step: int
    policy_version: int
    detail: str = ""


def _checkpoint_entries(manager: CheckpointManager) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in manager.root.iterdir()
            if path.name.startswith("checkpoint_")
        )
    )


def _checkpoint_directory_step(path: Path) -> int | None:
    suffix = path.name.removeprefix("checkpoint_")
    return int(suffix) if suffix.isdigit() else None


def preflight_checkpoint_run(
    checkpoint_manager: CheckpointManager,
    *,
    resume_path: os.PathLike[str] | str | None = None,
    resume_latest: bool = False,
) -> Path | None:
    """Resolve resume intent before allocating the agent or replay buffers."""

    if resume_path is not None and resume_latest:
        raise LearnerCompositionError(
            "resume_path and resume_latest are mutually exclusive"
        )
    entries = _checkpoint_entries(checkpoint_manager)
    if resume_path is None and not resume_latest:
        if entries:
            raise LearnerCompositionError(
                "fresh start refused because checkpoint entries already "
                f"exist under {checkpoint_manager.root}; pass an explicit "
                "resume checkpoint"
            )
        return None

    selected = (
        checkpoint_manager.latest_path()
        if resume_latest
        else Path(os.fspath(resume_path)).expanduser().resolve()
    )
    if not selected.is_dir():
        raise LearnerCompositionError(
            f"resume checkpoint is not a directory: {selected}"
        )
    selected_step = _checkpoint_directory_step(selected)
    if selected_step is None:
        raise LearnerCompositionError(
            "resume checkpoint must use the canonical checkpoint_<step> name"
        )
    if selected.parent != checkpoint_manager.root:
        if entries:
            raise LearnerCompositionError(
                "an external resume checkpoint requires a checkpoint-empty "
                "output root so lineages cannot be mixed; choose a new "
                "--checkpoint-root"
            )
        return selected

    future_entries = tuple(
        path
        for path in entries
        if (step := _checkpoint_directory_step(path)) is not None
        and step > selected_step
    )
    if future_entries:
        raise LearnerCompositionError(
            "checkpoint output root contains entries beyond the restored "
            "step; they are preserved and would collide with continued "
            "training. Use --resume-path with a new empty --checkpoint-root. "
            f"First conflict: {future_entries[0]}"
        )
    return selected


def _validate_fresh_agent(agent: Any) -> None:
    try:
        state_step = int(np.asarray(agent.state.step))
    except Exception as exc:
        raise LearnerCompositionError(
            "fresh agent must expose a scalar state.step"
        ) from exc
    if state_step != 0:
        raise LearnerCompositionError(
            f"fresh agent state.step must be 0, got {state_step}"
        )


def _validate_restored_checkpoint(
    restored: RestoredCheckpoint, config: LearnerConfig
) -> None:
    state_step = int(np.asarray(restored.agent.state.step))
    expected_gradient_step = restored.learner_step * config.cta_ratio
    expected_policy_version = restored.learner_step // config.publish_period
    if state_step != restored.gradient_step:
        raise CheckpointCorruptError(
            "restored agent state.step does not match gradient_step"
        )
    if restored.gradient_step != expected_gradient_step:
        raise CheckpointCorruptError(
            "gradient_step must equal learner_step * cta_ratio"
        )
    if restored.learner_step % config.checkpoint_period:
        raise CheckpointCorruptError(
            "production resume checkpoint is not on a checkpoint boundary"
        )
    if restored.learner_step % config.publish_period:
        raise CheckpointCorruptError(
            "production resume checkpoint is not on a publish boundary"
        )
    if restored.policy_version != expected_policy_version:
        raise CheckpointCorruptError(
            "policy_version does not match completed publish boundaries"
        )


def prepare_learner_state(
    *,
    agent_template: Any,
    checkpoint_manager: CheckpointManager,
    fingerprint: LearnerFingerprint,
    config: LearnerConfig = LearnerConfig(),
    resume_path: os.PathLike[str] | str | None = None,
    inference_rng: Any | None = None,
) -> PreparedLearnerState:
    """Load and validate all checkpoint state before allocating live ingress."""

    _validate_fresh_agent(agent_template)
    resolved_resume_path = preflight_checkpoint_run(
        checkpoint_manager,
        resume_path=resume_path,
    )
    if resolved_resume_path is None:
        if inference_rng is None:
            import jax

            inference_rng = jax.random.PRNGKey(config.seed)
        return PreparedLearnerState(
            agent=agent_template,
            learner_step=0,
            gradient_step=0,
            policy_version=0,
            inference_rng=inference_rng,
            restored_checkpoint=None,
            config=config,
            fingerprint_sha256=fingerprint.sha256,
        )

    if inference_rng is not None:
        raise LearnerCompositionError(
            "resume must use the checkpoint inference RNG"
        )
    restored = checkpoint_manager.load(
        agent_template=agent_template,
        fingerprint=fingerprint,
        path=resolved_resume_path,
    )
    _validate_restored_checkpoint(restored, config)
    return PreparedLearnerState(
        agent=restored.agent,
        learner_step=restored.learner_step,
        gradient_step=restored.gradient_step,
        policy_version=restored.policy_version,
        inference_rng=restored.inference_rng,
        restored_checkpoint=restored,
        config=config,
        fingerprint_sha256=fingerprint.sha256,
    )


def compose_learner(
    *,
    agent_template: Any,
    ingress: Any,
    offline_demos: Any,
    checkpoint_manager: CheckpointManager,
    fingerprint: LearnerFingerprint,
    config: LearnerConfig = LearnerConfig(),
    logger: Any | None = None,
    resume_path: os.PathLike[str] | str | None = None,
    inference_rng: Any | None = None,
    prepared_state: PreparedLearnerState | None = None,
    sample_action: Optional[
        Callable[[Any, Mapping[str, Any], Any, bool], Any]
    ] = None,
    parameter_validator: Callable[[Any], None] | None = None,
    candidate_postprocessor: Callable[[Any], Any] | None = None,
    policy_model_id: str = FROZEN_TRUNK_MODEL_REVISION,
) -> LearnerAssembly:
    """Create a mutually consistent policy runtime, sampler, and learner.

    ``resume_path=None`` means an intentional fresh run.  In that mode any
    checkpoint-looking directory is rejected rather than allowing a delayed
    overwrite fault at the next checkpoint boundary.  Callers that want the
    latest checkpoint must resolve it explicitly with ``latest_path()``.
    """

    if getattr(ingress, "require_grasp_penalty", None) is not True:
        raise LearnerCompositionError(
            "production learner ingress must require grasp_penalty"
        )
    if (
        getattr(ingress, "observation_representation", None)
        != config.observation_representation
        or getattr(ingress, "augmentation", None) != config.augmentation
    ):
        raise LearnerCompositionError(
            "production ingress representation/augmentation does not match "
            "the learner config"
        )
    if len(offline_demos) <= 0:
        raise LearnerCompositionError(
            "at least one canonical offline demonstration is required"
        )
    if (
        getattr(offline_demos, "observation_representation", None)
        != config.observation_representation
        or getattr(offline_demos, "augmentation", None) != config.augmentation
    ):
        raise LearnerCompositionError(
            "offline demo representation/augmentation does not match the "
            "learner config"
        )
    if prepared_state is not None:
        if resume_path is not None or inference_rng is not None:
            raise LearnerCompositionError(
                "prepared_state cannot be combined with resume_path or "
                "inference_rng"
            )
        _validate_fresh_agent(agent_template)
        if prepared_state.config != config:
            raise LearnerCompositionError(
                "prepared learner config does not match composition config"
            )
        if prepared_state.fingerprint_sha256 != fingerprint.sha256:
            raise LearnerCompositionError(
                "prepared learner fingerprint does not match composition"
            )
        prepared = prepared_state
        state_step = int(np.asarray(prepared.agent.state.step))
        if state_step != prepared.gradient_step:
            raise LearnerCompositionError(
                "prepared agent state.step does not match gradient_step"
            )
        if prepared.restored_checkpoint is None:
            if (
                prepared.agent is not agent_template
                or prepared.learner_step != 0
                or prepared.gradient_step != 0
                or prepared.policy_version != 0
            ):
                raise LearnerCompositionError(
                    "fresh prepared state must retain the supplied step-0 "
                    "agent and zero counters"
                )
        else:
            _validate_restored_checkpoint(
                prepared.restored_checkpoint,
                config,
            )
            if (
                prepared.agent is not prepared.restored_checkpoint.agent
                or prepared.learner_step
                != prepared.restored_checkpoint.learner_step
                or prepared.gradient_step
                != prepared.restored_checkpoint.gradient_step
                or prepared.policy_version
                != prepared.restored_checkpoint.policy_version
            ):
                raise LearnerCompositionError(
                    "prepared counters diverge from the restored checkpoint"
                )
    else:
        prepared = prepare_learner_state(
            agent_template=agent_template,
            checkpoint_manager=checkpoint_manager,
            fingerprint=fingerprint,
            config=config,
            resume_path=resume_path,
            inference_rng=inference_rng,
        )

    agent = prepared.agent
    learner_step = prepared.learner_step
    gradient_step = prepared.gradient_step
    policy_version = prepared.policy_version
    inference_rng = prepared.inference_rng
    restored = prepared.restored_checkpoint

    policy_runtime = VersionedPolicyRuntime(
        agent,
        policy_version=policy_version,
        learner_step=learner_step,
        inference_rng=inference_rng,
        sample_action=sample_action,
        parameter_validator=parameter_validator,
        model_id=policy_model_id,
    )
    sampler = RLPDBatchSampler(
        online_replay=ReplayIngressView(ingress, "replay"),
        offline_demos=offline_demos,
        online_interventions=ReplayIngressView(ingress, "intervention"),
        batch_size=config.batch_size,
        training_starts=config.training_starts,
        seed=config.seed,
        observation_representation=config.observation_representation,
    )
    learner = HILSERLLearner(
        agent=agent,
        sampler=sampler,
        publisher=policy_runtime,
        config=config,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        fingerprint=fingerprint,
        learner_step=learner_step,
        gradient_step=gradient_step,
        policy_version=policy_version,
        parameter_validator=parameter_validator,
        candidate_postprocessor=candidate_postprocessor,
    )
    if policy_runtime.learner_step != learner.learner_step:
        raise LearnerCompositionError("policy and learner steps diverged")
    if policy_runtime.policy_version != learner.policy_version:
        raise LearnerCompositionError("policy and learner versions diverged")
    return LearnerAssembly(
        learner=learner,
        policy_runtime=policy_runtime,
        sampler=sampler,
        ingress=ingress,
        restored_checkpoint=restored,
    )


def build_actor_service(
    *,
    assembly: LearnerAssembly,
    classifier: Any,
    allowed_actor_ids: tuple[str, ...] | None = None,
    allowed_run_ids: tuple[str, ...] | None = None,
) -> Any:
    """Bind the shared policy and ingress to the transport-neutral service."""

    if not bool(getattr(classifier, "ready", False)):
        raise LearnerCompositionError(
            f"reward classifier is not ready: "
            f"{getattr(classifier, 'fault_detail', 'unknown fault')}"
        )
    from ur_env.actor_network import ActorSessionService
    from ur_env.rlpd_receive_server import RewardTransitionFinalizer

    runtime = assembly.policy_runtime
    ingress = assembly.ingress
    return ActorSessionService(
        sample_action=runtime,
        model_id=runtime.model_id,
        reward_authority="server_classifier",
        reward_model_id=classifier.reward_model_id,
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        finalize_transition=RewardTransitionFinalizer(classifier),
        accept_data=ingress,
        buffer_status_provider=ingress.status,
        allowed_actor_ids=allowed_actor_ids,
        allowed_run_ids=allowed_run_ids,
    )


class LearnerWorker:
    """Own exactly one background caller of ``HILSERLLearner.train_once``."""

    _FINAL_STATES = frozenset({"completed", "stopped", "faulted"})

    def __init__(
        self,
        learner: HILSERLLearner,
        *,
        target_learner_step: int | None = None,
        poll_interval: float = 0.1,
        thread_name: str = "hil-serl-learner",
    ) -> None:
        if target_learner_step is not None:
            if (
                isinstance(target_learner_step, bool)
                or not isinstance(target_learner_step, int)
                or target_learner_step < learner.learner_step
            ):
                raise ValueError(
                    "target_learner_step must be an integer at or beyond the "
                    "current learner step"
                )
        if not math.isfinite(poll_interval) or poll_interval <= 0.0:
            raise ValueError("poll_interval must be positive and finite")
        self.learner = learner
        self.target_learner_step = target_learner_step
        self.poll_interval = float(poll_interval)
        self._stop_event = threading.Event()
        self._finished_event = threading.Event()
        self._lock = threading.Lock()
        self._state = "new"
        self._detail = ""
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=False,
        )

    @property
    def status(self) -> LearnerWorkerStatus:
        with self._lock:
            state = self._state
            detail = self._detail
        return LearnerWorkerStatus(
            state=state,
            learner_step=self.learner.learner_step,
            gradient_step=self.learner.gradient_step,
            policy_version=self.learner.policy_version,
            detail=detail,
        )

    @property
    def finished(self) -> bool:
        return self._finished_event.is_set()

    def start(self) -> None:
        with self._lock:
            if self._state != "new":
                raise RuntimeError("learner worker can only be started once")
            self._state = "running"
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._finished_event.wait(timeout)

    def join(self, timeout: float | None = None) -> bool:
        if self._thread.ident is None:
            return self.finished
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _finish(self, state: str, detail: str = "") -> None:
        if state not in self._FINAL_STATES:
            raise ValueError(f"invalid final worker state {state!r}")
        with self._lock:
            self._state = state
            self._detail = detail[:2_000]
        self._finished_event.set()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if (
                    self.target_learner_step is not None
                    and self.learner.learner_step
                    >= self.target_learner_step
                ):
                    self._finish("completed")
                    return
                try:
                    existing_fault = self.learner.fault
                    if existing_fault is not None:
                        self._finish("faulted", existing_fault.detail)
                        return
                    if not self.learner.ready:
                        self._stop_event.wait(self.poll_interval)
                        continue
                    self.learner.train_once()
                except LearnerNotReadyError:
                    self._stop_event.wait(self.poll_interval)
                except LearnerFaultError as exc:
                    self._finish("faulted", str(exc))
                    return
                except Exception as exc:
                    fault = self.learner.enter_fault(exc)
                    self._finish("faulted", str(fault))
                    return
            self._finish("stopped")
        except BaseException as exc:
            # Even an unexpected worker-level failure is made visible.  Do not
            # let it escape and terminate the still-valid policy service.
            fault = self.learner.enter_fault(exc)
            self._finish("faulted", str(fault))
