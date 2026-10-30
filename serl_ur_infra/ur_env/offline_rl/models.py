"""MLP heads and IFQL assembly on a shared frozen-vision observation vector."""

from dataclasses import dataclass, replace
from functools import partial
import math

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from .actor import FlowActor
from .ifql import IFQL, IFQLConfig, IFQLCritic
from .vision import VisionConfig
from .normalization import Normalization


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    horizon: int = 16
    integration_steps: int = 8
    learning_rate: float = 3e-4
    layer_norm: bool = True

    def __post_init__(self):
        if not isinstance(self.hidden_dims, tuple) or not self.hidden_dims:
            raise ValueError("hidden_dims must be a nonempty tuple")
        for value in (*self.hidden_dims, self.horizon, self.integration_steps):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("MLP dimensions and step counts must be positive integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


class MLP(nn.Module):
    config: MLPConfig
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for width in self.config.hidden_dims:
            x = nn.Dense(width)(x)
            if self.config.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return nn.Dense(self.output_dim)(x)


class FlowMLP(nn.Module):
    config: MLPConfig

    @nn.compact
    def __call__(self, observations, actions, time):
        if actions.shape != (observations.shape[0], self.config.horizon, 7):
            raise ValueError("actions must have shape (B,horizon,7)")
        frequencies = 2. ** jnp.arange(8)
        angles = jnp.pi * time[:, None] * frequencies
        time_features = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)
        x = jnp.concatenate((observations, actions.reshape((actions.shape[0], -1)), time_features), axis=-1)
        return MLP(self.config, self.config.horizon * 7)(x).reshape(actions.shape)


class QEnsembleMLP(nn.Module):
    config: MLPConfig
    num_qs: int

    @nn.compact
    def __call__(self, observations, actions):
        if actions.shape != (observations.shape[0], self.config.horizon, 7):
            raise ValueError("actions must have shape (B,horizon,7)")
        x = jnp.concatenate((observations, actions.reshape((actions.shape[0], -1))), axis=-1)
        # Independent MLPs, not multiple scalar outputs sharing hidden layers.
        return jnp.stack([MLP(self.config, 1, name=f"q_{i}")(x)[..., 0]
                          for i in range(self.num_qs)], axis=0)


class ValueMLP(nn.Module):
    config: MLPConfig

    @nn.compact
    def __call__(self, observations):
        return MLP(self.config, 1)(observations)[..., 0]


def canonical_action(model_actions, action_stats):
    """Project after inverse normalization, in the executable action space."""
    actions = jnp.clip(action_stats.unnormalize(model_actions), -1., 1.)
    return actions.at[..., 6].set(jnp.where(actions[..., 6] >= 0, 1., -1.))


@partial(jax.jit, static_argnames=("config", "action_stats"))
def _sample_flow(state, observations, rng, *, config, action_stats):
    actions = jax.random.normal(rng, (observations.shape[0], config.horizon, 7))

    def integrate(i, actions):
        time = jnp.full((actions.shape[0],), i / config.integration_steps)
        velocity = state.apply_fn({"params": state.params}, observations, actions, time)
        return actions + velocity / config.integration_steps

    actions = jax.lax.fori_loop(0, config.integration_steps, integrate, actions)
    # Q was trained in MODEL space. Score exactly the projected executable
    # candidates after transforming them back, not pre-projection samples.
    return action_stats.normalize(canonical_action(actions, action_stats))


@dataclass(frozen=True)
class FeatureIFQL:
    algorithm: IFQL
    vision: VisionConfig
    normalization: Normalization

    def update(self, batch, *, rng):
        # Encode/normalize once; actor, Q and V share the same arrays and stats.
        encoded = self.normalization.batch(batch)
        algorithm, metrics = self.algorithm.update(encoded, rng=rng)
        return replace(self, algorithm=algorithm), metrics

    def sample_actions(self, observations, *, rng):
        actions = self.algorithm.sample_actions(self.normalization.observations(observations), rng=rng)
        return canonical_action(actions, self.normalization.actions)


def create_feature_ifql(example_observations, *, rng, mlp=MLPConfig(),
                        ifql=IFQLConfig(), vision=VisionConfig(), normalization=None):
    """Initialize independent actor/Q/V heads; no pretrained download or robot I/O.

    Inputs are cached PRE-pooling ResNet10 features. The pretrained trunk stays
    outside the train states, so none of these optimizers can change it.
    """
    normalization = normalization if normalization is not None else Normalization.identity(vision)
    if normalization.vision != vision:
        raise ValueError("normalization vision config mismatch; refit for the chosen pooling/temperature")
    observations = normalization.observations(example_observations)
    actions = jnp.zeros((observations.shape[0], mlp.horizon, 7))
    q_key, v_key, actor_key = jax.random.split(rng, 3)

    def initialize(model, key, *args):
        params = model.init(key, *args)["params"]
        return TrainState.create(apply_fn=model.apply, params=params,
                                 tx=optax.adam(mlp.learning_rate))

    q = initialize(QEnsembleMLP(mlp, ifql.num_qs), q_key, observations, actions)
    v = initialize(ValueMLP(mlp), v_key, observations)
    flow = initialize(FlowMLP(mlp), actor_key, observations, actions, jnp.zeros(observations.shape[0]))
    actor = FlowActor(flow, partial(_sample_flow, config=mlp, action_stats=normalization.actions))
    critic = IFQLCritic(q, v, q.params, ifql)
    return FeatureIFQL(IFQL(actor, critic), vision, normalization)
