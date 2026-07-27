"""Local HIL-SERL learner boundary.

Heavy JAX/Flax modules stay lazily imported inside their public functions so
the robot-side actor and receive-server contract tests remain lightweight.
"""

from ur_env.learner.agent import (
    LearnerDependencyError,
    ResNetAssetError,
    create_hybrid_sac_agent,
    ensure_resnet10_cache,
    validate_learner_dependencies,
    verify_resnet10_asset,
)
from ur_env.learner.batches import (
    LearnerBatchError,
    RLPDBatchSampler,
    ReplayIngressView,
    SamplingMetrics,
    proportional_sample_counts,
    sanitize_learner_batch,
)
from ur_env.learner.checkpoint import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointExistsError,
    CheckpointFingerprintError,
    CheckpointManager,
    LearnerFingerprint,
    RestoredCheckpoint,
)
from ur_env.learner.config import LearnerConfig, RESNET10_SHA256
from ur_env.learner.demo import (
    CanonicalTransitionPool,
    DemoContractError,
    DemoSidecar,
    LoadedDemos,
    load_demo_object,
    load_demo_pickle,
    load_demo_pickles,
)
from ur_env.learner.logging import JsonlWandbLogger, LearnerLoggingError
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
    "CheckpointManager",
    "DemoContractError",
    "DemoSidecar",
    "HILSERLLearner",
    "JsonlWandbLogger",
    "LearnerBatchError",
    "LearnerConfig",
    "LearnerDependencyError",
    "LearnerFault",
    "LearnerFaultError",
    "LearnerFingerprint",
    "LearnerLoggingError",
    "LearnerNotReadyError",
    "LearnerStepResult",
    "LoadedDemos",
    "PolicyPublisher",
    "PolicySnapshot",
    "PolicyValidationError",
    "RESNET10_SHA256",
    "RLPDBatchSampler",
    "ReplayIngressView",
    "ResNetAssetError",
    "RestoredCheckpoint",
    "SamplingMetrics",
    "VersionedPolicyRuntime",
    "canonical_policy_observation",
    "create_hybrid_sac_agent",
    "ensure_resnet10_cache",
    "load_demo_object",
    "load_demo_pickle",
    "load_demo_pickles",
    "proportional_sample_counts",
    "sanitize_learner_batch",
    "validate_learner_dependencies",
    "validate_parameter_tree",
    "validate_tree_finite",
    "verify_resnet10_asset",
]
