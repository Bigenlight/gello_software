"""Minimal IFQL: expectile V, chunk Q, FM actor, best-of-N selection.

Algorithm reference: ~/repos/fmrl/agents/ifql.py. Network architecture and
dataset adapters are deliberately left to the caller, not copied from OGBench.
"""

from dataclasses import dataclass, replace
from functools import partial
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from .batch import ChunkBatch
from .aggregation import aggregate_q, validate_aggregation
from .interfaces import Actor


@dataclass(frozen=True)
class IFQLConfig:
    num_qs: int = 10
    kappa: float = 0.9
    rho: float = 0.5
    value_q_aggregation: str = "mean_minus_std"
    selection_q_aggregation: str = "min"
    target_tau: float = 0.005
    num_candidates: int = 32

    def __post_init__(self):
        if not 0 < self.kappa < 1:
            raise ValueError("kappa must be in (0,1)")
        if not math.isfinite(self.rho) or self.rho < 0:
            raise ValueError("rho must be finite and nonnegative")
        if isinstance(self.num_qs, bool) or not isinstance(self.num_qs, int) or self.num_qs <= 0:
            raise ValueError("num_qs must be a positive integer")
        validate_aggregation(self.value_q_aggregation)
        validate_aggregation(self.selection_q_aggregation)
        if not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0,1]")
        if isinstance(self.num_candidates, bool) or not isinstance(self.num_candidates, int) or self.num_candidates <= 0:
            raise ValueError("num_candidates must be a positive integer")


def expectile_loss(error, expectile):
    return jnp.where(error >= 0, expectile, 1 - expectile) * jnp.square(error)


def _check_q_shape(qs, batch_size, config):
    if qs.shape != (config.num_qs, batch_size):
        raise ValueError(
            f"Q model must return (num_qs={config.num_qs},B={batch_size}), got {qs.shape}; "
            "initialize the Q network with config.num_qs"
        )


@partial(jax.jit, static_argnames=("config",))
def _critic_update(q_state, v_state, target_q_params, batch, *, config):
    mask = batch.critic_mask

    def average(values):
        return (values * mask).sum() / jnp.maximum(mask.sum(), 1)

    # Both targets use the PRE-update snapshot, as in IFQL's joint loss.
    target_qs = q_state.apply_fn({"params": target_q_params}, batch.observations, batch.actions)
    _check_q_shape(target_qs, batch.actions.shape[0], config)
    value_target = jax.lax.stop_gradient(
        aggregate_q(target_qs, mode=config.value_q_aggregation, rho=config.rho)
    )
    next_v = v_state.apply_fn({"params": v_state.params}, batch.bootstrap_observations)
    if next_v.shape != batch.returns.shape:
        raise ValueError("V model must return (B,)")
    q_target = jax.lax.stop_gradient(batch.returns + batch.bootstrap_discount * next_v)

    def value_loss(params):
        value = v_state.apply_fn({"params": params}, batch.observations)
        return average(expectile_loss(value_target - value, config.kappa))

    def q_loss(params):
        qs = q_state.apply_fn({"params": params}, batch.observations, batch.actions)
        _check_q_shape(qs, batch.actions.shape[0], config)
        return average(jnp.square(qs - q_target[None, :]).mean(axis=0))

    v_loss, v_grads = jax.value_and_grad(value_loss)(v_state.params)
    q_loss_value, q_grads = jax.value_and_grad(q_loss)(q_state.params)
    new_q = q_state.apply_gradients(grads=q_grads)
    new_v = v_state.apply_gradients(grads=v_grads)
    # Explicit convention: Polyak tracks NEW online Q parameters.
    new_target = jax.tree_util.tree_map(
        lambda target, online: (1 - config.target_tau) * target + config.target_tau * online,
        target_q_params, new_q.params,
    )
    return new_q, new_v, new_target, {
        "q_loss": q_loss_value, "value_loss": v_loss,
        "target_mean": average(q_target), "valid_chunks": mask.sum(),
    }


@dataclass(frozen=True)
class IFQLCritic:
    q_state: TrainState              # apply(obs, actions[B,H,A]) -> (E,B)
    v_state: TrainState              # apply(obs) -> (B,)
    target_q_params: Any
    config: IFQLConfig = IFQLConfig()

    def update(self, batch: ChunkBatch, *, actor=None, context=None, rng=None):
        if not np.asarray(batch.critic_mask).any():
            # Avoid even advancing Adam momentum on an empty critic batch.
            return self, {"skipped": 1, "valid_chunks": 0}, context
        q, v, target, metrics = _critic_update(
            self.q_state, self.v_state, self.target_q_params, batch, config=self.config,
        )
        return replace(self, q_state=q, v_state=v, target_q_params=target), metrics, context

    def score(self, observations, actions):
        qs = self.q_state.apply_fn({"params": self.q_state.params}, observations, actions)
        _check_q_shape(qs, actions.shape[0], self.config)
        return aggregate_q(qs, mode=self.config.selection_q_aggregation, rho=self.config.rho)


@dataclass(frozen=True)
class IFQL:
    actor: Actor
    critic: IFQLCritic

    def update(self, batch: ChunkBatch, *, rng):
        batch.validate()
        critic_key, actor_key = jax.random.split(rng)
        critic, critic_info, context = self.critic.update(
            batch, actor=self.actor, context=None, rng=critic_key,
        )
        actor, actor_info = self.actor.update(batch, critic=critic, context=context, rng=actor_key)
        metrics = {f"critic/{key}": value for key, value in critic_info.items()}
        metrics.update({f"actor/{key}": value for key, value in actor_info.items()})
        return replace(self, actor=actor, critic=critic), metrics

    def sample_actions(self, observations, *, rng):
        """Return (B,H,A), independently selecting one candidate per batch row."""
        count = self.critic.config.num_candidates
        candidates = self.actor.sample_candidates(observations, rng=rng, count=count)
        b, n, h, a = candidates.shape
        repeated = jax.tree_util.tree_map(lambda x: jnp.repeat(x, n, axis=0), observations)
        scores = self.critic.score(repeated, candidates.reshape((b * n, h, a))).reshape((b, n))
        return candidates[jnp.arange(b), jnp.argmax(scores, axis=1)]
