"""Local HIL-SERL learner boundary.

Heavy JAX/Flax modules stay lazily imported inside their public functions so
the robot-side actor and receive-server contract tests remain lightweight.
"""

from ur_env.learner.agent import (
    LearnerDependencyError,
    ResNetAssetError,
    create_hybrid_sac_agent,
    default_resnet_source,
    ensure_resnet10_cache,
    file_sha256,
    validate_learner_dependencies,
    verify_resnet10_asset,
)
from ur_env.learner.batches import (
    LearnerBatchError,
    RLPDBatchSampler,
    ReplayIngressView,
    SamplingMetrics,
    proportional_sample_counts,
    sample_proportional_counts,
    sanitize_learner_batch,
)
from ur_env.learner.checkpoint import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointExistsError,
    CheckpointFingerprintError,
    CheckpointLockError,
    CheckpointManager,
    CheckpointRunLock,
    CheckpointSpaceError,
    LearnerFingerprint,
    RestoredCheckpoint,
)
from ur_env.learner.config import LearnerConfig, RESNET10_SHA256
from ur_env.learner.composition import (
    LearnerAssembly,
    LearnerCompositionError,
    LearnerWorker,
    LearnerWorkerStatus,
    PreparedLearnerState,
    build_actor_service,
    compose_learner,
    preflight_checkpoint_run,
    prepare_learner_state,
)
from ur_env.learner.demo import (
    CanonicalTransitionPool,
    DemoContractError,
    DemoSidecar,
    LoadedDemos,
    load_demo_object,
    load_demo_pickle,
    load_demo_pickles,
)
from ur_env.learner.fake_demo import (
    FAKE_DEMO_PICKLE_PROTOCOL,
    build_fake_demo_payload,
    write_fake_demo_pickle,
)
from ur_env.learner.logging import JsonlWandbLogger, LearnerLoggingError
from ur_env.learner.ingress import (
    FaultGatedReplayIngress,
    ReplayIngressFault,
    ReplayIngressFaultError,
)
from ur_env.learner.policy import (
    PolicyPublisher,
    PolicySnapshot,
    PolicyValidationError,
    VersionedPolicyRuntime,
    canonical_policy_observation,
    validate_parameter_tree,
    validate_tree_finite,
)
from ur_env.learner.runtime import (
    HILSERLLearner,
    LearnerFault,
    LearnerFaultError,
    LearnerNotReadyError,
    LearnerStepResult,
)


__all__ = [
    "CanonicalTransitionPool",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointExistsError",
    "CheckpointFingerprintError",
    "CheckpointLockError",
    "CheckpointManager",
    "CheckpointRunLock",
    "CheckpointSpaceError",
    "DemoContractError",
    "DemoSidecar",
    "FaultGatedReplayIngress",
    "FAKE_DEMO_PICKLE_PROTOCOL",
    "HILSERLLearner",
    "JsonlWandbLogger",
    "LearnerBatchError",
    "LearnerAssembly",
    "LearnerCompositionError",
    "LearnerConfig",
    "LearnerDependencyError",
    "LearnerFault",
    "LearnerFaultError",
    "LearnerFingerprint",
    "LearnerLoggingError",
    "LearnerNotReadyError",
    "LearnerStepResult",
    "LearnerWorker",
    "LearnerWorkerStatus",
    "LoadedDemos",
    "PolicyPublisher",
    "PolicySnapshot",
    "PolicyValidationError",
    "PreparedLearnerState",
    "RESNET10_SHA256",
    "RLPDBatchSampler",
    "ReplayIngressView",
    "ReplayIngressFault",
    "ReplayIngressFaultError",
    "ResNetAssetError",
    "RestoredCheckpoint",
    "SamplingMetrics",
    "VersionedPolicyRuntime",
    "canonical_policy_observation",
    "build_actor_service",
    "build_fake_demo_payload",
    "compose_learner",
    "create_hybrid_sac_agent",
    "default_resnet_source",
    "ensure_resnet10_cache",
    "file_sha256",
    "load_demo_object",
    "load_demo_pickle",
    "load_demo_pickles",
    "proportional_sample_counts",
    "preflight_checkpoint_run",
    "prepare_learner_state",
    "sample_proportional_counts",
    "sanitize_learner_batch",
    "validate_learner_dependencies",
    "validate_parameter_tree",
    "validate_tree_finite",
    "verify_resnet10_asset",
    "write_fake_demo_pickle",
]
