"""Opt-in integration test for the real upstream hybrid SAC checkpoint path.

Run this expensive CPU/JAX test explicitly with::

    RUN_HIL_SERL_ACTUAL_CHECKPOINT=1 pytest -q \
        serl_ur_infra/tests/test_actual_agent_checkpoint_integration.py

The default suite only imports this module and skips the test.  Heavy learner
and JAX imports deliberately stay inside the test body.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HIL_SERL_ACTUAL_CHECKPOINT") != "1",
    reason="set RUN_HIL_SERL_ACTUAL_CHECKPOINT=1 to run the real SAC checkpoint test",
)


_HERE = Path(__file__).resolve().parent
_INFRA = _HERE.parent
_REPO = _INFRA.parent


def _observation(value: int) -> dict[str, np.ndarray]:
    return {
        "state": np.full((1, 19), value / 255.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _transition(value: int) -> dict[str, object]:
    action = np.zeros((7,), dtype=np.float32)
    action[:6] = np.float32((value % 5) / 10.0)
    action[-1] = np.float32((-1.0, 0.0, 1.0)[value % 3])
    return {
        "observations": _observation(value),
        "next_observations": _observation(value + 1),
        "actions": action,
        "rewards": np.float32(value % 2),
        "masks": np.float32(1.0),
        "grasp_penalty": np.float32(-0.02),
    }


def _sampler(*, seed: int):
    from ur_env.learner import CanonicalTransitionPool, RLPDBatchSampler

    return RLPDBatchSampler(
        online_replay=CanonicalTransitionPool([_transition(1)], seed=seed + 1),
        offline_demos=CanonicalTransitionPool([_transition(2)], seed=seed + 2),
        online_interventions=CanonicalTransitionPool([], seed=seed + 3),
        batch_size=2,
        training_starts=1,
        seed=seed,
    )


def _assert_train_states_exact(source, restored) -> None:
    import jax

    assert type(source) is type(restored)
    source_leaves, _ = jax.tree_util.tree_flatten_with_path(source)
    restored_leaves, _ = jax.tree_util.tree_flatten_with_path(restored)
    assert len(source_leaves) == len(restored_leaves) > 0
    assert [jax.tree_util.keystr(path) for path, _ in source_leaves] == [
        jax.tree_util.keystr(path) for path, _ in restored_leaves
    ]

    for (source_path, source_leaf), (restored_path, restored_leaf) in zip(
        source_leaves, restored_leaves
    ):
        assert source_path == restored_path
        source_array = np.asarray(jax.device_get(source_leaf))
        restored_array = np.asarray(jax.device_get(restored_leaf))
        assert source_array.dtype == restored_array.dtype, jax.tree_util.keystr(
            source_path
        )
        assert source_array.shape == restored_array.shape, jax.tree_util.keystr(
            source_path
        )
        np.testing.assert_array_equal(
            source_array,
            restored_array,
            err_msg=f"train-state leaf differs at {jax.tree_util.keystr(source_path)}",
        )


def test_real_agent_checkpoint_resume_and_continue_cta(tmp_path):
    # Make the infra-owned TensorFlow annotation shim and protobuf startup
    # compatibility available before importing the real upstream launcher.
    sys.path.insert(0, str(_INFRA))

    import jax

    from ur_env.learner import (
        CheckpointManager,
        HILSERLLearner,
        LearnerConfig,
        LearnerFingerprint,
        VersionedPolicyRuntime,
        canonical_policy_observation,
        create_hybrid_sac_agent,
        validate_learner_dependencies,
        verify_resnet10_asset,
    )

    versions = validate_learner_dependencies(include_logging=False)
    assert versions == {
        "jax": "0.5.3",
        "jaxlib": "0.5.3",
        "flax": "0.10.5",
        "distrax": "0.1.5",
        "tensorflow_probability": "0.25.0",
    }

    config = LearnerConfig(
        batch_size=2,
        training_starts=1,
        publish_period=1,
        checkpoint_period=1,
        log_period=1,
    )
    resnet_source = (
        _REPO
        / "third_party"
        / "hil-serl"
        / "examples"
        / "experiments"
        / "resnet10_params.pkl"
    )
    verify_resnet10_asset(resnet_source)
    resnet_cache = tmp_path / "resnet10_params.pkl"

    def create_agent():
        return create_hybrid_sac_agent(
            config=config,
            resnet_source_path=resnet_source,
            resnet_cache_path=resnet_cache,
            validate_versions=True,
        )

    agent = create_agent()
    from serl_launcher.agents.continuous.sac_hybrid_single import (
        SACAgentHybridSingleArm,
    )

    assert isinstance(agent, SACAgentHybridSingleArm)
    assert int(np.asarray(agent.state.step)) == 0

    fingerprint = LearnerFingerprint.create(
        config=config, resnet_asset_path=resnet_source
    )
    manager = CheckpointManager(tmp_path / "checkpoints")
    publisher = VersionedPolicyRuntime(
        agent,
        inference_rng=jax.random.PRNGKey(710),
    )
    learner = HILSERLLearner(
        agent=agent,
        sampler=_sampler(seed=100),
        publisher=publisher,
        config=config,
        checkpoint_manager=manager,
        fingerprint=fingerprint,
    )

    first = learner.train_once()
    assert first.learner_step == 1
    assert first.gradient_step == 2
    assert first.policy_version == 1
    assert first.published
    assert first.checkpoint_path == str(manager.path_for_step(1))
    assert manager.path_for_step(1).is_dir()
    checkpoint_inference_rng = publisher.inference_rng

    fresh_agent = create_agent()
    assert isinstance(fresh_agent, SACAgentHybridSingleArm)
    assert int(np.asarray(fresh_agent.state.step)) == 0
    restored = manager.load(
        agent_template=fresh_agent,
        fingerprint=fingerprint,
        path=first.checkpoint_path,
    )

    assert restored.learner_step == learner.learner_step == 1
    assert restored.gradient_step == learner.gradient_step == 2
    assert restored.policy_version == learner.policy_version == 1
    assert int(np.asarray(restored.agent.state.step)) == restored.gradient_step
    _assert_train_states_exact(learner.agent.state, restored.agent.state)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(learner.agent.state.rng)),
        np.asarray(jax.device_get(restored.agent.state.rng)),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(checkpoint_inference_rng)),
        np.asarray(jax.device_get(restored.inference_rng)),
    )

    source_runtime = VersionedPolicyRuntime(
        learner.agent,
        policy_version=learner.policy_version,
        learner_step=learner.learner_step,
        inference_rng=checkpoint_inference_rng,
    )
    restored_runtime = VersionedPolicyRuntime(
        restored.agent,
        policy_version=restored.policy_version,
        learner_step=restored.learner_step,
        inference_rng=restored.inference_rng,
    )
    observation = canonical_policy_observation(value=17)
    for deterministic in (True, False):
        source_action, source_version = source_runtime(
            observation, deterministic=deterministic
        )
        restored_action, restored_version = restored_runtime(
            observation, deterministic=deterministic
        )
        assert source_version == restored_version == 1
        np.testing.assert_array_equal(source_action, restored_action)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(source_runtime.inference_rng)),
        np.asarray(jax.device_get(restored_runtime.inference_rng)),
    )

    resumed_learner = HILSERLLearner(
        agent=restored.agent,
        sampler=_sampler(seed=200),
        publisher=restored_runtime,
        config=config,
        checkpoint_manager=manager,
        fingerprint=fingerprint,
        learner_step=restored.learner_step,
        gradient_step=restored.gradient_step,
        policy_version=restored.policy_version,
    )
    second = resumed_learner.train_once()
    assert second.learner_step == 2
    assert second.gradient_step == 4
    assert second.policy_version == 2
    assert second.published
    assert second.checkpoint_path == str(manager.path_for_step(2))
    assert manager.path_for_step(2).is_dir()
    assert int(np.asarray(resumed_learner.agent.state.step)) == 4
