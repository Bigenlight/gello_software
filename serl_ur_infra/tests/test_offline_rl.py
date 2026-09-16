"""CPU tests for the offline contracts and first IFQL update path.

Run with unittest so the learner environment does not need pytest installed.
"""

import unittest
from dataclasses import replace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from ur_env.offline_rl.actor import FlowActor
from ur_env.offline_rl.aggregation import aggregate_q
from ur_env.offline_rl.batch import make_chunk_batch
from ur_env.offline_rl.ifql import IFQL, IFQLConfig, IFQLCritic, expectile_loss
from ur_env.offline_rl.trainer import OfflineTrainer
from ur_env.offline_rl.vision import VisionConfig, encode_features, spatial_softmax


class TinyQ(nn.Module):
    num_qs: int = 2

    @nn.compact
    def __call__(self, obs, actions):
        bias = self.param("bias", lambda _: 2. * jnp.arange(1, self.num_qs + 1))
        return bias[:, None] + jnp.zeros((self.num_qs, actions.shape[0]))


class TinyV(nn.Module):
    @nn.compact
    def __call__(self, obs):
        bias = self.param("bias", nn.initializers.ones, ())
        return bias + jnp.zeros(obs["state"].shape[0])


class TinyFlow(nn.Module):
    @nn.compact
    def __call__(self, obs, actions, time):
        bias = self.param("bias", nn.initializers.zeros, (actions.shape[-1],))
        return jnp.broadcast_to(bias, actions.shape)


def batch_fixture(valid=None, terminated=None, truncated=None):
    valid = np.ones((2, 2)) if valid is None else np.asarray(valid)
    actions = np.full((2, 2, 7), 0.25, np.float32) * valid[..., None]
    obs = {"state": np.zeros((2, 3), np.float32)}
    return make_chunk_batch(
        observations=obs, actions=actions, rewards=np.array([[1, 2], [3, 4]]),
        valid_mask=valid,
        terminated=np.zeros((2, 2)) if terminated is None else terminated,
        truncated=np.zeros((2, 2)) if truncated is None else truncated,
        bootstrap_observations=obs, discount=0.5,
    )


def state_for(model, *args):
    params = model.init(jax.random.PRNGKey(7), *args)["params"]
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.sgd(0.1))


def make_algorithm():
    batch = batch_fixture()
    q = state_for(TinyQ(), batch.observations, batch.actions)
    v = state_for(TinyV(), batch.observations)
    flow = state_for(TinyFlow(), batch.observations, batch.actions, jnp.zeros(2))
    actor = FlowActor(flow, lambda state, obs, key: jnp.zeros((obs["state"].shape[0], 2, 7)))
    critic = IFQLCritic(q, v, q.params, IFQLConfig(num_qs=2, target_tau=0.2, num_candidates=3))
    return IFQL(actor, critic)


class OfflineRLTests(unittest.TestCase):
    def test_spatial_softmax_coordinates_temperature_and_stability(self):
        uniform = jnp.zeros((1, 4, 4, 2))
        np.testing.assert_allclose(spatial_softmax(uniform), 0, atol=1e-7)
        peaks = uniform.at[0, 0, 3, 0].set(100).at[0, 3, 0, 1].set(100)
        np.testing.assert_allclose(spatial_softmax(peaks), [[1, -1, -1, 1]], atol=1e-6)
        warm = spatial_softmax(peaks, temperature=100.)
        self.assertTrue(np.isfinite(warm).all())
        self.assertTrue((np.abs(warm) < np.abs(spatial_softmax(peaks))).all())
        np.testing.assert_allclose(spatial_softmax(jnp.zeros((1, 1, 1, 2))), 0)

    def test_shared_features_shape_and_frozen_gradient(self):
        obs = {"cam1": jnp.ones((2, 1, 4, 4, 512)),
               "cam2": jnp.ones((2, 1, 4, 4, 512)) * 2,
               "state": jnp.ones((2, 1, 19)) * 3}
        for mode, dim in (("flatten", 16403), ("spatial_softmax", 2067)):
            config = VisionConfig(pooling=mode)
            encoded = encode_features(obs, config)
            self.assertEqual(encoded.shape, (2, dim))
            np.testing.assert_allclose(encoded[:, -19:], 3)
            gradient = jax.grad(lambda feature: encode_features(dict(obs, cam1=feature), config).sum())(obs["cam1"])
            np.testing.assert_array_equal(gradient, 0)
        for kwargs in ({"pooling": "typo"}, {"spatial_softmax_temperature": 0}):
            with self.assertRaises(ValueError):
                VisionConfig(**kwargs)

    def test_aggregation_modes_and_single_critic(self):
        qs = jnp.array([[1., 7.], [5., 3.]])
        for mode, expected in (("min", [1, 3]), ("mean", [3, 5]),
                               ("max", [5, 7]), ("mean_minus_std", [2, 4])):
            with self.subTest(mode=mode):
                np.testing.assert_allclose(aggregate_q(qs, mode=mode, rho=.5), expected)
        np.testing.assert_allclose(aggregate_q(qs[:1], mode="mean_minus_std"), [1, 7])

    def test_value_and_selection_aggregation_are_independent(self):
        critic = make_algorithm().critic
        batch = batch_fixture()
        for mode, expected_loss in (("min", .9), ("mean", 3.6),
                                    ("max", 8.1), ("mean_minus_std", 2.025)):
            configured = replace(critic, config=replace(critic.config, value_q_aggregation=mode))
            _, metrics, _ = configured.update(batch)
            self.assertAlmostEqual(float(metrics["value_loss"]), expected_loss, places=5)
            np.testing.assert_allclose(configured.score(batch.observations, batch.actions), [2, 2])
        configured = replace(critic, config=replace(critic.config, selection_q_aggregation="mean"))
        _, metrics, _ = configured.update(batch)
        self.assertAlmostEqual(float(metrics["value_loss"]), 2.025, places=5)
        np.testing.assert_allclose(configured.score(batch.observations, batch.actions), [3, 3])

    def test_kappa_rho_and_num_qs_control_actual_updates(self):
        batch = batch_fixture()
        critic = make_algorithm().critic
        for count in (1, 3):
            with self.subTest(num_qs=count):
                q = state_for(TinyQ(num_qs=count), batch.observations, batch.actions)
                config = replace(critic.config, num_qs=count, kappa=.7, rho=0.)
                configured = IFQLCritic(q, critic.v_state, q.params, config)
                updated, metrics, _ = configured.update(batch)
                # Q biases 2,4,... -> mean=count+1; old V=1.
                self.assertAlmostEqual(float(metrics["value_loss"]), .7 * count**2, places=5)
                self.assertEqual(updated.q_state.params["bias"].shape, (count,))

    def test_invalid_config_and_network_count_mismatch_are_rejected(self):
        for kwargs in ({"num_qs": 0}, {"num_qs": True}, {"kappa": 1},
                       {"rho": -1}, {"rho": float("nan")},
                       {"value_q_aggregation": "typo"}, {"selection_q_aggregation": "typo"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                IFQLConfig(**kwargs)
        critic = make_algorithm().critic
        wrong = replace(critic, config=replace(critic.config, num_qs=3))
        with self.assertRaisesRegex(ValueError, "num_qs=3"):
            wrong.update(batch_fixture())
        with self.assertRaisesRegex(ValueError, "num_qs=3"):
            wrong.score(batch_fixture().observations, batch_fixture().actions)

    def test_chunk_return_and_terminal_versus_truncation(self):
        batch = batch_fixture(terminated=[[0, 1], [0, 0]], truncated=[[0, 0], [0, 1]])
        np.testing.assert_allclose(batch.returns, [2, 5])
        np.testing.assert_allclose(batch.bootstrap_discount, [0, 0.25])

    def test_short_tail_discount_and_critic_mask(self):
        batch = batch_fixture(valid=[[1, 0], [1, 1]], truncated=[[1, 0], [0, 0]])
        np.testing.assert_allclose(batch.returns, [1, 5])
        np.testing.assert_allclose(batch.bootstrap_discount, [0.5, 0.25])
        np.testing.assert_array_equal(batch.critic_mask, [0, 1])

    def test_cross_episode_and_malformed_padding_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "episode boundary"):
            batch_fixture(terminated=[[1, 0], [0, 0]])
        with self.assertRaisesRegex(ValueError, "nonempty prefix"):
            batch_fixture(valid=[[0, 1], [1, 1]])
        batch = batch_fixture(valid=[[1, 0], [1, 1]])
        with self.assertRaisesRegex(ValueError, "padded actions"):
            batch.replace(actions=jnp.ones((2, 2, 7))).validate()

    def test_expectile_asymmetry(self):
        np.testing.assert_allclose(expectile_loss(jnp.array([-2., 2.]), 0.9), [0.4, 3.6])

    def test_ifql_targets_use_old_v_and_polyak_uses_new_q(self):
        algorithm = make_algorithm()
        before = algorithm.critic
        batch = batch_fixture()
        after, metrics, _ = before.update(batch)
        # old V=1, gamma**H=.25: targets [2.25,5.25], mean 3.75.
        # old target Q=[2,4]: mean - .5*std = 2.5, expectile loss .9*1.5**2.
        self.assertAlmostEqual(float(metrics["target_mean"]), 3.75)
        self.assertAlmostEqual(float(metrics["value_loss"]), 2.025, places=5)
        self.assertEqual(int(after.q_state.step), 1)
        self.assertEqual(int(after.v_state.step), 1)
        for old, new, target in zip(jax.tree_util.tree_leaves(before.target_q_params),
                                    jax.tree_util.tree_leaves(after.q_state.params),
                                    jax.tree_util.tree_leaves(after.target_q_params)):
            np.testing.assert_allclose(target, old * .8 + new * .2, rtol=1e-6)
        self.assertEqual(int(before.q_state.step), 0)

    def test_partial_rows_do_not_train_q_or_v(self):
        critic = make_algorithm().critic
        batch = batch_fixture(valid=[[1, 0], [1, 1]])
        changed = batch.replace(returns=batch.returns.at[0].set(99999))
        a, _, _ = critic.update(batch)
        b, _, _ = critic.update(changed)
        for x, y in zip(jax.tree_util.tree_leaves(a.q_state.params),
                        jax.tree_util.tree_leaves(b.q_state.params)):
            np.testing.assert_array_equal(x, y)
        tail = batch_fixture(valid=[[1, 0], [1, 0]])
        same, metrics, _ = critic.update(tail)
        self.assertIs(same, critic)
        self.assertEqual(metrics["skipped"], 1)

    def test_offline_step_updates_actor_and_critics_without_online_replay(self):
        class Dataset:
            def sample_chunks(self, batch_size, horizon):
                return batch_fixture()

        trainer = OfflineTrainer(make_algorithm(), batch_size=2, horizon=2)
        updated, metrics = trainer.train_once(Dataset(), rng=jax.random.PRNGKey(1))
        self.assertEqual(updated.step, 1)
        self.assertEqual(trainer.step, 0)
        self.assertEqual(int(updated.algorithm.actor.state.step), 1)
        self.assertEqual(int(updated.algorithm.critic.q_state.step), 1)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in metrics.values()))
        self.assertFalse(np.array_equal(trainer.algorithm.actor.state.params["bias"],
                                       updated.algorithm.actor.state.params["bias"]))

    def test_ifql_selects_a_different_candidate_per_batch_row(self):
        class Candidates:
            def sample_candidates(self, obs, *, rng, count):
                return jnp.broadcast_to(jnp.arange(count)[None, :, None, None], (2, count, 2, 7))

        class Scores:
            config = IFQLConfig(num_candidates=3)

            def score(self, obs, actions):
                return obs["direction"][:, 0] * actions[:, 0, 0]

        algorithm = IFQL(Candidates(), Scores())
        selected = algorithm.sample_actions({"direction": jnp.array([[1.], [-1.]])}, rng=jax.random.PRNGKey(0))
        np.testing.assert_array_equal(selected[:, 0, 0], [2, 0])
        self.assertEqual(selected.shape, (2, 2, 7))

    def test_real_flow_sampler_adapter_preserves_chunk_and_gripper_contract(self):
        from ur_env.learner.flow_matching import FlowMatchingConfig, FlowMatchingPolicy, sample_action_chunks

        model = FlowMatchingPolicy(FlowMatchingConfig(
            horizon=2, hidden_dim=8, residual_blocks=1, spatial_features=1,
            camera_bottleneck_dim=4, proprio_bottleneck_dim=4,
            time_embedding_dim=8, integration_steps=2,
        ))
        obs = {"state": jnp.zeros((2, 1, 19)),
               "cam1": jnp.zeros((2, 1, 4, 4, 512)),
               "cam2": jnp.zeros((2, 1, 4, 4, 512))}
        state = state_for(model, obs, jnp.zeros((2, 2, 7)), jnp.zeros(2))
        actor = FlowActor(state, lambda state, obs, rng: sample_action_chunks(model, state.params, obs, rng))
        candidates = np.asarray(actor.sample_candidates(obs, rng=jax.random.PRNGKey(3), count=3))
        self.assertEqual(candidates.shape, (2, 3, 2, 7))
        self.assertTrue(np.isfinite(candidates).all())
        self.assertLessEqual(np.abs(candidates).max(), 1)
        self.assertTrue(np.isin(candidates[..., 6], [-1, 1]).all())


if __name__ == "__main__":
    unittest.main()
