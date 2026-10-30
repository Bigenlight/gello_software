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
from types import SimpleNamespace

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


def _transition(
    value: int,
    *,
    reward: float = 0.0,
    penalty: float = -0.02,
) -> dict[str, object]:
    action = np.zeros((7,), dtype=np.float32)
    action[:6] = np.float32((value % 5) / 10.0)
    action[-1] = np.float32((-1.0, 0.0, 1.0)[value % 3])
    terminal = bool(reward)
    return {
        "observations": _observation(value),
        "next_observations": _observation(value + 1),
        "actions": action,
        "rewards": np.float32(reward),
        "masks": np.float32(0.0 if terminal else 1.0),
        "dones": terminal,
        "grasp_penalty": np.float32(penalty),
    }


def _feature_transition(extractor, transition):
    return {
        "observations": extractor(transition["observations"]),
        "next_observations": extractor(transition["next_observations"]),
        "actions": transition["actions"],
        "rewards": transition["rewards"],
        "masks": transition["masks"],
        "grasp_penalty": transition["grasp_penalty"],
    }


def _feature_ring(transitions, *, seed: int):
    from ur_env.learner import FeatureTransitionRing

    ring = FeatureTransitionRing(max(1, len(transitions)), seed=seed)
    for transition in transitions:
        ring.insert(transition)
    return ring


def _sampler(*, seed: int, offline_pool, extractor):
    from ur_env.learner import (
        FROZEN_TRUNK_REPRESENTATION,
        RLPDBatchSampler,
    )

    return RLPDBatchSampler(
        online_replay=_feature_ring(
            [_feature_transition(extractor, _transition(1, reward=0.0))],
            seed=seed + 1,
        ),
        offline_demos=offline_pool,
        online_interventions=_feature_ring([], seed=seed + 3),
        batch_size=2,
        training_starts=1,
        seed=seed,
        observation_representation=FROZEN_TRUNK_REPRESENTATION,
    )


class _PoolIngress:
    """Minimal strict ingress implementing the production composition contract."""

    require_grasp_penalty = True
    observation_representation = "resnet10_frozen_trunk_map_f32_v1"
    augmentation = "none"

    def __init__(self, replay, interventions) -> None:
        self._replay = replay
        self._interventions = interventions

    def status(self):
        return SimpleNamespace(
            replay_size=len(self._replay),
            intervention_size=len(self._interventions),
        )

    def sample_replay(self, batch_size: int):
        return self._replay.sample(batch_size)

    def sample_intervention(self, batch_size: int):
        return self._interventions.sample(batch_size)


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
        FROZEN_TRUNK_CONTRACT,
        FrozenResNet10TrunkExtractor,
        HILSERLLearner,
        LearnerConfig,
        LearnerFingerprint,
        VersionedPolicyRuntime,
        canonical_policy_observation,
        compose_learner,
        convert_loaded_demos_to_feature_pool,
        create_frozen_trunk_feature_agent,
        load_demo_pickle,
        prepare_learner_state,
        validate_learner_dependencies,
        verify_resnet10_asset,
        write_fake_demo_pickle,
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
    fake_demo_path = tmp_path / "canonical_fake_demo.pkl"
    write_fake_demo_pickle(fake_demo_path)
    loaded_demos = load_demo_pickle(fake_demo_path)
    assert len(loaded_demos) == 2
    assert loaded_demos.sidecars[0].metadata["success"] is True
    assert loaded_demos.sidecars[1].metadata["success"] is False
    assert loaded_demos.sidecars[1].metadata["run_id"] == "fake-acceptance-run"
    assert loaded_demos.sidecars[1].metadata["intervened"] == 1
    assert all(
        set(transition)
        == {
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "masks",
            "grasp_penalty",
        }
        for transition in loaded_demos.transitions
    )

    def create_agent():
        return create_frozen_trunk_feature_agent(
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
    assert agent.config["augmentation_function"] is None
    extractor = FrozenResNet10TrunkExtractor(
        agent, resnet_asset_path=resnet_source
    )
    feature_demos = convert_loaded_demos_to_feature_pool(
        loaded_demos,
        feature_extractor=extractor,
        seed=102,
        extraction_batch_size=2,
    )

    fingerprint = LearnerFingerprint.create(
        config=config,
        resnet_asset_path=resnet_source,
        run_contract={
            "learner_observations": FROZEN_TRUNK_CONTRACT.document(),
            "augmentation": "none",
        },
    )
    manager = CheckpointManager(tmp_path / "checkpoints")
    publisher = VersionedPolicyRuntime(
        agent,
        inference_rng=jax.random.PRNGKey(710),
        parameter_validator=extractor.validate_parameter_invariant,
    )
    learner = HILSERLLearner(
        agent=agent,
        sampler=_sampler(
            seed=100,
            offline_pool=feature_demos,
            extractor=extractor,
        ),
        publisher=publisher,
        config=config,
        checkpoint_manager=manager,
        fingerprint=fingerprint,
        parameter_validator=extractor.validate_parameter_invariant,
        candidate_postprocessor=extractor.repin_target_trunk,
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
    prepared = prepare_learner_state(
        agent_template=fresh_agent,
        checkpoint_manager=manager,
        fingerprint=fingerprint,
        config=config,
        resume_path=first.checkpoint_path,
    )
    assert prepared.restored_checkpoint is not None
    restored = prepared.restored_checkpoint
    fresh_extractor = FrozenResNet10TrunkExtractor(
        fresh_agent, resnet_asset_path=resnet_source
    )
    fresh_extractor.validate_agent_invariant(prepared.agent)
    assembly = compose_learner(
        agent_template=fresh_agent,
        ingress=_PoolIngress(
            _feature_ring(
                [
                    _feature_transition(
                        fresh_extractor, _transition(4, reward=0.0)
                    )
                ],
                seed=201,
            ),
            _feature_ring([], seed=202),
        ),
        offline_demos=feature_demos,
        checkpoint_manager=manager,
        fingerprint=fingerprint,
        config=config,
        prepared_state=prepared,
        parameter_validator=fresh_extractor.validate_parameter_invariant,
        candidate_postprocessor=fresh_extractor.repin_target_trunk,
    )
    assert assembly.restored_checkpoint is restored

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
        parameter_validator=extractor.validate_parameter_invariant,
    )
    restored_runtime = assembly.policy_runtime
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

    resumed_learner = assembly.learner
    second = resumed_learner.train_once()
    assert second.learner_step == 2
    assert second.gradient_step == 4
    assert second.policy_version == 2
    assert second.published
    assert second.checkpoint_path == str(manager.path_for_step(2))
    assert manager.path_for_step(2).is_dir()
    assert int(np.asarray(resumed_learner.agent.state.step)) == 4
