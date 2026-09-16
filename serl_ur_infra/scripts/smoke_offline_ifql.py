#!/usr/bin/env python3
"""Exercise real MLP heads on synthetic cached vision features, without a robot.

This checks repeated optimization, not pretrained extraction or task learning.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np

from ur_env.offline_rl.batch import make_chunk_batch
from ur_env.offline_rl.ifql import IFQLConfig
from ur_env.offline_rl.models import MLPConfig, create_feature_ifql
from ur_env.offline_rl.normalization import MODES, NormalizationConfig, fit_training_normalization
from ur_env.offline_rl.trainer import OfflineTrainer
from ur_env.offline_rl.vision import VisionConfig, encode_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooling", choices=("flatten", "spatial_softmax"), default="flatten")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--state-normalization", choices=MODES, default="none")
    parser.add_argument("--feature-normalization", choices=MODES, default="none")
    parser.add_argument("--action-normalization", choices=MODES, default="none")
    parser.add_argument("--reward-scale", type=float, default=1.)
    parser.add_argument("--reward-bias", type=float, default=0.)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    rng = np.random.default_rng(42)
    b, h = 4, 4
    obs = {"state": rng.normal(size=(b, 1, 19)).astype(np.float32)}
    for name in ("cam1", "cam2"):
        obs[name] = rng.normal(size=(b, 1, 4, 4, 512)).astype(np.float32)
    actions = rng.uniform(-.5, .5, (b, h, 7)).astype(np.float32)
    actions[..., 6] = rng.choice([-1., 1.], (b, h))
    batch = make_chunk_batch(
        observations=obs, actions=actions, rewards=np.ones((b, h), np.float32),
        valid_mask=np.ones((b, h)), terminated=np.zeros((b, h)),
        truncated=np.zeros((b, h)), bootstrap_observations=obs, discount=.99,
    )

    class FixedDataset:
        def sample_chunks(self, batch_size, horizon):
            return batch

    vision = VisionConfig(pooling=args.pooling)
    normalization = fit_training_normalization(
        obs, actions, vision=vision, config=NormalizationConfig(
            state=args.state_normalization, features=args.feature_normalization,
            actions=args.action_normalization, reward_scale=args.reward_scale,
            reward_bias=args.reward_bias,
        ),
    )
    algorithm = create_feature_ifql(
        obs, rng=jax.random.PRNGKey(0), vision=vision, normalization=normalization,
        mlp=MLPConfig(hidden_dims=(32, 32), horizon=h, integration_steps=4),
        ifql=IFQLConfig(num_qs=2, num_candidates=4),
    )
    trainer = OfflineTrainer(algorithm, batch_size=b, horizon=h)

    def states(algorithm):
        inner = algorithm.algorithm
        return {"actor": inner.actor.state, "q": inner.critic.q_state, "v": inner.critic.v_state}

    initial = states(algorithm)
    key = jax.random.PRNGKey(1)
    for _ in range(args.steps):
        key, step_key = jax.random.split(key)
        trainer, metrics = trainer.train_once(FixedDataset(), rng=step_key)
        if not all(np.isfinite(np.asarray(value)).all() for value in metrics.values()):
            raise RuntimeError("nonfinite training metrics")
    for name, state in states(trainer.algorithm).items():
        before = jax.tree_util.tree_leaves(initial[name].params)
        after = jax.tree_util.tree_leaves(state.params)
        if int(state.step) != args.steps or not any(not np.array_equal(a, b) for a, b in zip(before, after)):
            raise RuntimeError(f"{name} did not update")
        if not all(np.isfinite(np.asarray(value)).all() for value in after):
            raise RuntimeError(f"{name} has nonfinite parameters")
    sampled = np.asarray(trainer.algorithm.sample_actions(obs, rng=key))
    if sampled.shape != (b, h, 7) or not np.isfinite(sampled).all() or np.abs(sampled).max() > 1:
        raise RuntimeError("invalid sampled action chunk")
    if not np.isin(sampled[..., 6], [-1, 1]).all():
        raise RuntimeError("invalid sampled gripper")
    print(f"PASS backend={jax.default_backend()} pooling={args.pooling} "
          f"input_dim={encode_features(obs, vision).shape[-1]} steps={args.steps} "
          "actor/Q/V_changed=True sampled_shape=" + str(sampled.shape))
    print({key: float(value) for key, value in metrics.items()})
    print(f"normalization: state={args.state_normalization} features={args.feature_normalization} "
          f"actions={args.action_normalization} reward={args.reward_scale}*r+{args.reward_bias}")


if __name__ == "__main__":
    main()
