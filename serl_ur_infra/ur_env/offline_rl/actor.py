"""IFQL's FM behavior actor; the observation encoder/model is injected."""

from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from .batch import ChunkBatch


@jax.jit
def _flow_update(state, batch, rng):
    noise_key, time_key = jax.random.split(rng)
    noise = jax.random.normal(noise_key, batch.actions.shape)
    time = jax.random.uniform(time_key, (batch.actions.shape[0],))
    noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * batch.actions
    target = batch.actions - noise

    def loss_fn(params):
        velocity = state.apply_fn({"params": params}, batch.observations, noisy, time)
        if velocity.shape != batch.actions.shape:
            raise ValueError("flow model must return (B,H,A) velocities")
        squared = jnp.square(velocity - target) * batch.valid_mask[..., None]
        return squared.sum() / (batch.valid_mask.sum() * batch.actions.shape[-1])

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), {"flow_loss": loss}


@dataclass(frozen=True)
class FlowActor:
    state: TrainState
    # sample_fn(state, batched_observations, rng) -> (B,H,A).
    # Adapter owns Euler steps, action bounds and gripper discretization.
    sample_fn: Callable

    def update(self, batch: ChunkBatch, *, critic=None, context=None, rng):
        # IFQL has no Q/advantage weighting in its actor training objective.
        state, metrics = _flow_update(self.state, batch, rng)
        return replace(self, state=state), metrics

    def sample_candidates(self, observations, *, rng, count: int):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        batch_size = jax.tree_util.tree_leaves(observations)[0].shape[0]
        repeated = jax.tree_util.tree_map(lambda x: jnp.repeat(x, count, axis=0), observations)
        actions = self.sample_fn(self.state, repeated, rng)
        if actions.ndim != 3 or actions.shape[0] != batch_size * count:
            raise ValueError("sample_fn must return (B*N,H,A)")
        return actions.reshape((batch_size, count, *actions.shape[1:]))
