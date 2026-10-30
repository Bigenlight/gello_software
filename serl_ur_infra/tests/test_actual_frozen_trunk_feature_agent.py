"""Opt-in real JAX tests for the frozen-trunk dual-input SAC agent.

Run explicitly with::

    RUN_HIL_SERL_ACTUAL_FEATURE_AGENT=1 python -m pytest -q \
        serl_ur_infra/tests/test_actual_frozen_trunk_feature_agent.py
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HIL_SERL_ACTUAL_FEATURE_AGENT") != "1",
    reason="set RUN_HIL_SERL_ACTUAL_FEATURE_AGENT=1 to run the real SAC test",
)

_INFRA = Path(__file__).resolve().parents[1]
_REPO = _INFRA.parent
sys.path.insert(0, str(_INFRA))


def _config():
    from ur_env.learner import LearnerConfig

    return LearnerConfig(
        batch_size=2,
        training_starts=1,
        publish_period=1,
        checkpoint_period=1,
    )


def _raw_observation(value: int) -> dict[str, np.ndarray]:
    from ur_env.learner import canonical_policy_observation

    observation = canonical_policy_observation(value)
    observation["cam2"] = np.full(
        (1, 128, 128, 3), 255 - value, dtype=np.uint8
    )
    observation["state"] = np.linspace(
        -0.5, 0.5, 19, dtype=np.float32
    )[None]
    return observation


def _create_agent(tmp_path):
    from ur_env.learner import create_frozen_trunk_feature_agent

    return create_frozen_trunk_feature_agent(
        config=_config(),
        resnet_source_path=(
            _REPO
            / "third_party"
            / "hil-serl"
            / "examples"
            / "experiments"
            / "resnet10_params.pkl"
        ),
        resnet_cache_path=tmp_path / "verified-resnet10.pkl",
        validate_versions=True,
    )


def _tree_arrays(tree):
    import jax

    return [
        np.asarray(jax.device_get(leaf))
        for leaf in jax.tree_util.tree_leaves(tree)
    ]


def _assert_tree_exact(left, right):
    left_leaves = _tree_arrays(left)
    right_leaves = _tree_arrays(right)
    assert len(left_leaves) == len(right_leaves) > 0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def _assert_tree_changed(before_leaves, after):
    after_leaves = _tree_arrays(after)
    assert len(before_leaves) == len(after_leaves) > 0
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(before_leaves, after_leaves)
    )


def test_real_extractor_and_raw_cached_action_equivalence(tmp_path):
    import jax
    from ur_env.learner import (
        FROZEN_TRUNK_FEATURE_CONTRACT_REVISION,
        FROZEN_TRUNK_PARAMETER_PATH,
        FrozenResNet10TrunkExtractor,
        FrozenTrunkFeatureSchemaError,
    )
    from ur_env.learner.agent import (
        _create_legacy_raw_augmented_hybrid_sac_agent,
    )

    agent = _create_agent(tmp_path)
    legacy_agent = _create_legacy_raw_augmented_hybrid_sac_agent(
        config=_config(),
        resnet_source_path=(
            _REPO
            / "third_party"
            / "hil-serl"
            / "examples"
            / "experiments"
            / "resnet10_params.pkl"
        ),
        resnet_cache_path=tmp_path / "verified-resnet10.pkl",
        validate_versions=True,
    )
    assert jax.tree_util.tree_structure(agent.state.params) == (
        jax.tree_util.tree_structure(legacy_agent.state.params)
    )
    _assert_tree_exact(agent.state.params, legacy_agent.state.params)
    extractor = FrozenResNet10TrunkExtractor(agent)
    expected_params = agent.state.params
    for key in FROZEN_TRUNK_PARAMETER_PATH:
        expected_params = expected_params[key]
    assert extractor.parameter_path == FROZEN_TRUNK_PARAMETER_PATH
    assert extractor.parameter_reference is expected_params
    extractor.validate_parameter_invariant(agent.state.params)

    bad_params = copy.deepcopy(agent.state.params)
    target = bad_params
    for key in FROZEN_TRUNK_PARAMETER_PATH[:-1]:
        target = target[key]
    trunk_leaves, trunk_tree = jax.tree_util.tree_flatten(
        target[FROZEN_TRUNK_PARAMETER_PATH[-1]]
    )
    trunk_leaves[0] = trunk_leaves[0] + np.float32(1.0)
    target[FROZEN_TRUNK_PARAMETER_PATH[-1]] = (
        jax.tree_util.tree_unflatten(trunk_tree, trunk_leaves)
    )
    with pytest.raises(FrozenTrunkFeatureSchemaError, match="weights changed"):
        extractor.validate_parameter_invariant(bad_params)
    raw = _raw_observation(37)
    first = extractor(raw)
    second = extractor(raw)

    assert agent.config["augmentation_function"] is None
    assert agent.config["feature_contract_revision"] == (
        FROZEN_TRUNK_FEATURE_CONTRACT_REVISION
    )
    for image_key in ("cam1", "cam2"):
        assert first[image_key].shape == (1, 4, 4, 512)
        assert first[image_key].dtype == np.dtype(np.float32)
        assert np.isfinite(first[image_key]).all()
        np.testing.assert_array_equal(first[image_key], second[image_key])

    for deterministic in (True, False):
        rng = jax.random.PRNGKey(7301)
        raw_action = np.asarray(
            agent.sample_actions(raw, seed=rng, argmax=deterministic)
        )
        cached_action = np.asarray(
            agent.sample_actions(first, seed=rng, argmax=deterministic)
        )
        legacy_action = np.asarray(
            legacy_agent.sample_actions(raw, seed=rng, argmax=deterministic)
        )
        # XLA may fuse the raw trunk with the downstream head while the cached
        # path creates a materialization boundary.  They remain numerically,
        # though not necessarily bit-for-bit, equivalent.
        # A GPU may fuse the raw trunk into the head while the host-backed
        # replay path necessarily materializes the float32 trunk map.  Kanu's
        # RTX A4000 measured a 1.27e-4 maximum action delta for this smoke.
        np.testing.assert_allclose(
            raw_action, cached_action, rtol=5e-4, atol=5e-4
        )
        np.testing.assert_allclose(raw_action, legacy_action, rtol=1e-6, atol=1e-6)
        assert raw_action[-1] == cached_action[-1]


def test_real_cached_cta_updates_head_but_not_frozen_trunk(tmp_path):
    from flax.core import freeze

    from ur_env.learner import FrozenResNet10TrunkExtractor

    agent = _create_agent(tmp_path)
    extractor = FrozenResNet10TrunkExtractor(agent)
    observations = extractor(
        {
            key: np.stack([_raw_observation(11)[key], _raw_observation(29)[key]])
            for key in ("state", "cam1", "cam2")
        }
    )
    next_observations = extractor(
        {
            key: np.stack([_raw_observation(12)[key], _raw_observation(30)[key]])
            for key in ("state", "cam1", "cam2")
        }
    )
    batch = freeze(
        {
            "observations": observations,
            "next_observations": next_observations,
            "actions": np.asarray(
                [
                    [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, -1.0],
                    [-0.2, 0.1, -0.3, 0.2, -0.4, 0.3, 1.0],
                ],
                dtype=np.float32,
            ),
            "rewards": np.asarray([0.0, 1.0], dtype=np.float32),
            "masks": np.asarray([1.0, 0.0], dtype=np.float32),
            "grasp_penalty": np.asarray([-0.02, 0.0], dtype=np.float32),
        }
    )

    shared_before = agent.state.params["modules_actor"]["encoder"]
    camera1_before = shared_before["encoder_cam1"]
    trunk_before = camera1_before["pretrained_encoder"]
    camera1_head_before = _tree_arrays(
        {
            key: value
            for key, value in camera1_before.items()
            if key != "pretrained_encoder"
        }
    )
    camera2_head_before = _tree_arrays(shared_before["encoder_cam2"])
    proprio_before = _tree_arrays(
        {
            "Dense_0": shared_before["Dense_0"],
            "LayerNorm_0": shared_before["LayerNorm_0"],
        }
    )

    critic_agent, _ = agent.update(
        batch,
        networks_to_update=frozenset({"critic", "grasp_critic"}),
    )
    critic_agent = extractor.repin_target_trunk(critic_agent)
    cta_agent, _ = critic_agent.update(
        batch,
        networks_to_update=frozenset(
            {"actor", "critic", "grasp_critic", "temperature"}
        ),
    )
    cta_agent = extractor.repin_target_trunk(cta_agent)

    shared_after = cta_agent.state.params["modules_actor"]["encoder"]
    camera1_after = shared_after["encoder_cam1"]
    _assert_tree_exact(trunk_before, camera1_after["pretrained_encoder"])
    target_trunk_after = cta_agent.state.target_params["modules_actor"][
        "encoder"
    ]["encoder_cam1"]["pretrained_encoder"]
    _assert_tree_exact(trunk_before, target_trunk_after)
    extractor.validate_agent_invariant(cta_agent)
    _assert_tree_changed(
        camera1_head_before,
        {
            key: value
            for key, value in camera1_after.items()
            if key != "pretrained_encoder"
        },
    )
    _assert_tree_changed(camera2_head_before, shared_after["encoder_cam2"])
    _assert_tree_changed(
        proprio_before,
        {
            "Dense_0": shared_after["Dense_0"],
            "LayerNorm_0": shared_after["LayerNorm_0"],
        },
    )
    assert int(np.asarray(cta_agent.state.step)) == 2


def test_real_classifier_reads_the_shared_trunk_feature():
    """The reward classifier must not own a second image encoder.

    Whether it is handed pixels or the ``(1, 4, 4, 512)`` map the server
    already computed for replay, the logit has to be the same -- otherwise
    "critic, actor and classifier share one feature" is not true and the
    trunk would have to run again just for reward.
    """

    import jax
    from serl_launcher.vision.resnet_v1 import resnetv1_configs

    from ur_env.learner.frozen_trunk import create_frozen_trunk_classifier

    image_keys = ("cam1", "cam2")
    classifier = create_frozen_trunk_classifier(
        image_keys=image_keys, validate_versions=False
    )

    encoders = classifier.params["encoder_def"]
    assert set(encoders) == {f"encoder_{key}" for key in image_keys}
    # Flax stores a shared submodule under the first user only: one trunk,
    # two cameras. cam2 carrying its own copy would mean two encoders.
    assert "pretrained_encoder" in encoders["encoder_cam1"]
    assert "pretrained_encoder" not in encoders["encoder_cam2"]
    shared_trunk = encoders["encoder_cam1"]["pretrained_encoder"]

    rng = np.random.default_rng(0)
    pixels = {
        key: rng.integers(0, 256, (1, 128, 128, 3), dtype=np.uint8)
        for key in image_keys
    }
    trunk = resnetv1_configs["resnetv1-10-frozen"](
        pre_pooling=True, name="pretrained_encoder"
    )
    features = {}
    for key in image_keys:
        value = trunk.apply({"params": shared_trunk}, pixels[key][0], train=False)
        features[key] = np.asarray(jax.device_get(value), dtype=np.float32)[None]
        assert features[key].shape == (1, 4, 4, 512)

    def logit(observation):
        return float(
            np.asarray(
                classifier.apply_fn(
                    {"params": classifier.params}, observation, train=False
                )
            ).ravel()[0]
        )

    from_pixels = logit(pixels)
    from_features = logit(features)
    # Same tolerance the agent's raw/cached equivalence uses: XLA may fuse the
    # trunk into the head on one path and materialise it on the other.
    assert abs(from_pixels - from_features) < 5e-4, (
        from_pixels,
        from_features,
    )
