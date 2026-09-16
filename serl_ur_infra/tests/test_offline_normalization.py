"""Numerical normalization, serialization and model-space integration tests."""

from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ur_env.offline_rl.batch import make_chunk_batch
from ur_env.offline_rl.ifql import IFQLConfig
from ur_env.offline_rl.models import MLPConfig, canonical_action, create_feature_ifql
from ur_env.offline_rl.normalization import (
    AffineStats, Normalization, NormalizationConfig, fit_affine, fit_training_normalization,
)
from ur_env.offline_rl.vision import VisionConfig, encode_features


def observations():
    values = np.arange(4, dtype=np.float32)
    return {"state": np.broadcast_to(values[:, None, None], (4, 1, 19)).copy(),
            "cam1": np.broadcast_to(values[:, None, None, None, None], (4, 1, 4, 4, 512)).copy(),
            "cam2": np.ones((4, 1, 4, 4, 512), np.float32)}


def actions():
    values = np.array([-.2, -.1, .1, .2], np.float32)
    result = np.broadcast_to(values[:, None, None], (4, 2, 7)).copy()
    result[..., 6] = np.array([-1, -1, -1, 1])[:, None]
    return result


def batch():
    return make_chunk_batch(
        observations=observations(), actions=actions(),
        rewards=np.ones((4, 2)), valid_mask=np.ones((4, 2)),
        terminated=np.array([[0, 1], [0, 0], [0, 0], [0, 0]]),
        truncated=np.array([[0, 0], [0, 1], [0, 0], [0, 0]]),
        bootstrap_observations=observations(), discount=.5,
    )


class NormalizationTests(unittest.TestCase):
    def test_affine_modes_roundtrip_and_constant_channel(self):
        values = np.array([[0., 9.], [2., 9.], [4., 9.]])
        for mode in ("none", "mean_std", "min_max", "quantile"):
            with self.subTest(mode=mode):
                stats = fit_affine(values, mode)
                norm = stats.normalize(jnp.asarray(values))
                self.assertTrue(np.isfinite(norm).all())
                np.testing.assert_allclose(stats.unnormalize(norm), values, atol=1e-6)
                self.assertEqual(stats.scale[1], 1.)
        standardized = fit_affine(values, "mean_std").normalize(values)
        self.assertAlmostEqual(float(standardized[:, 0].std()), 1, places=6)
        np.testing.assert_allclose(fit_affine(values, "min_max").normalize(values)[:, 0], [-1, 0, 1])

    def test_quantile_outliers_are_reversible_not_clipped(self):
        values = np.arange(101, dtype=float)[:, None]
        stats = fit_affine(values, "quantile", NormalizationConfig(quantile_low=.1, quantile_high=.9))
        self.assertEqual(stats.offset, (50.,))
        self.assertEqual(stats.scale, (40.,))
        encoded = stats.normalize(values)
        self.assertGreater(float(encoded[-1, 0]), 1.)
        np.testing.assert_allclose(stats.unnormalize(encoded), values, atol=1e-5)

    def test_action_statistics_share_time_axis_and_ignore_padding(self):
        a = actions()
        mask = np.ones((4, 2), dtype=bool)
        mask[:, 1] = False
        a[:, 1] = np.nan  # Masked rows must not contaminate fitted statistics.
        norm = fit_training_normalization(observations(), a, valid_mask=mask,
                                         config=NormalizationConfig(actions="mean_std", state="mean_std"))
        self.assertEqual(len(norm.actions.scale), 7)
        self.assertAlmostEqual(norm.actions.scale[0], float(actions()[:, 0, 0].std()), places=6)
        self.assertEqual(norm.actions.scale[6], 1.)
        self.assertEqual(norm.actions.offset[6], 0.)
        self.assertEqual(norm.state.scale[0], 1.)
        self.assertEqual(norm.state.offset[0], 0.)

    def test_defaults_preserve_vectors_and_actions(self):
        identity = Normalization.identity()
        packed = identity.batch(batch())
        np.testing.assert_array_equal(packed.observations, encode_features(observations()))
        np.testing.assert_array_equal(packed.actions, actions())
        np.testing.assert_array_equal(packed.returns, batch().returns)
        with self.assertRaisesRegex(ValueError, "already normalized"):
            identity.batch(packed)

    def test_current_and_next_state_use_the_same_fixed_statistics(self):
        norm = fit_training_normalization(observations(), actions(), config=NormalizationConfig(state="mean_std"))
        nxt = observations()
        nxt["state"] += 10
        packed = norm.batch(batch().replace(bootstrap_observations=nxt))
        np.testing.assert_allclose(packed.bootstrap_observations[:, -18:] - packed.observations[:, -18:],
                                   10 / np.std(np.arange(4)), rtol=1e-6)

    def test_reward_bias_is_discounted_per_step_even_for_terminal(self):
        norm = fit_training_normalization(observations(), actions(),
                                         config=NormalizationConfig(reward_scale=2, reward_bias=-1))
        original = batch()
        # sum gamma^i*(2*r_i-1) = 1 + .5 for terminal AND truncated rows.
        np.testing.assert_allclose(norm.batch(original).returns, [1.5] * 4)
        np.testing.assert_array_equal(norm.batch(original).bootstrap_discount, original.bootstrap_discount)
        with self.assertRaisesRegex(ValueError, "reward_discount_sum"):
            norm.batch(original.replace(reward_discount_sum=None))

    def test_normalized_padding_stays_zero_and_actions_can_exceed_one(self):
        a = actions()
        a[..., :6] += .4
        norm = fit_training_normalization(observations(), a,
                                         config=NormalizationConfig(actions="mean_std"))
        original = batch().replace(actions=jnp.asarray(a).at[0, 1].set(0),
                                   valid_mask=jnp.ones((4, 2)).at[0, 1].set(0),
                                   reward_discount_sum=jnp.array([1., 1.5, 1.5, 1.5]))
        packed = norm.batch(original)
        packed.validate()
        np.testing.assert_array_equal(packed.actions[0, 1], 0.)
        self.assertGreater(float(jnp.abs(packed.actions[..., :6]).max()), 1.)

    def test_canonical_projection_occurs_after_inverse_transform(self):
        stats = AffineStats((.5,) * 6 + (.2,), (.1,) * 6 + (2.,))
        raw = jnp.full((1, 2, 7), 3.)
        # Clipping model-space to 1 first would incorrectly yield .6 instead of .8.
        projected = canonical_action(raw, stats)
        np.testing.assert_allclose(projected[..., :6], .8)
        np.testing.assert_allclose(projected[..., 6], 1.)
        np.testing.assert_allclose(stats.unnormalize(stats.normalize(projected)), projected, atol=1e-6)

    def test_statistics_roundtrip_and_pooling_mismatch(self):
        norm = fit_training_normalization(observations(), actions(),
                                         config=NormalizationConfig(state="mean_std", actions="quantile"))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "normalization.json"
            norm.save(path)
            restored = Normalization.load(path)
            self.assertEqual(restored, norm)
            np.testing.assert_array_equal(restored.observations(observations()), norm.observations(observations()))
            with self.assertRaises(FileExistsError):
                norm.save(path)
        with self.assertRaisesRegex(ValueError, "vision config mismatch"):
            create_feature_ifql(observations(), rng=jax.random.PRNGKey(1),
                                vision=VisionConfig(pooling="spatial_softmax"), normalization=norm)

    def test_training_and_sampling_with_all_gripper_statistics_enabled(self):
        vision = VisionConfig(pooling="spatial_softmax")
        norm = fit_training_normalization(observations(), actions(), vision=vision,
                                         config=NormalizationConfig(
                                             state="mean_std", features="mean_std", actions="mean_std",
                                             normalize_state_gripper=True, normalize_action_gripper=True))
        algorithm = create_feature_ifql(observations(), rng=jax.random.PRNGKey(0),
                                       vision=vision, normalization=norm,
                                       mlp=MLPConfig(hidden_dims=(8,), horizon=2, integration_steps=2),
                                       ifql=IFQLConfig(num_qs=2, num_candidates=3))
        after, metrics = algorithm.update(batch(), rng=jax.random.PRNGKey(1))
        self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in metrics.values()))
        self.assertEqual(int(after.algorithm.actor.state.step), 1)
        sampled = np.asarray(after.sample_actions(observations(), rng=jax.random.PRNGKey(2)))
        self.assertEqual(sampled.shape, (4, 2, 7))
        self.assertTrue(np.isin(sampled[..., 6], [-1, 1]).all())
        self.assertLessEqual(np.abs(sampled).max(), 1.)
        candidates = after.algorithm.actor.sample_candidates(norm.observations(observations()),
                                                             rng=jax.random.PRNGKey(3), count=3)
        canonical = np.asarray(norm.actions.unnormalize(candidates))
        np.testing.assert_allclose(np.abs(canonical[..., 6]), 1, atol=1e-6)
        self.assertLessEqual(np.abs(canonical[..., :6]).max(), 1.000001)

    def test_invalid_statistics_and_config_are_rejected(self):
        for kwargs in ({"state": "typo"}, {"reward_scale": 0}, {"epsilon": 0},
                       {"reward_bias": float("nan")}, {"quantile_low": .99, "quantile_high": .01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NormalizationConfig(**kwargs)
        with self.assertRaises(ValueError):
            AffineStats((0.,), (0.,))
        with self.assertRaises(ValueError):
            fit_affine(np.empty((0, 7)), "mean_std")


if __name__ == "__main__":
    unittest.main()
