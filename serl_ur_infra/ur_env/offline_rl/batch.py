"""Chunk contract at the dataset/algorithm boundary (no file loading here)."""

from typing import Any

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class ChunkBatch:
    observations: Any                 # pytree, leaves (B, ...)
    actions: Any                      # (B, H, A), normalized; padding is zero
    valid_mask: Any                   # (B, H), nonempty prefix of ones
    returns: Any                     # (B,), discounted reward over valid steps
    bootstrap_observations: Any       # observation AFTER the last valid action
    bootstrap_discount: Any           # (B,), gamma**k; zero for true termination
    reward_discount_sum: Any = None   # (B,), sum_i gamma**i over valid steps
    action_space: str = flax.struct.field(pytree_node=False, default="canonical")

    @property
    def critic_mask(self):
        # Initial IFQL contract: Q(s, action_chunk) always evaluates H actions.
        # Partial tails train FM only, not an unmodelled variable-duration Q.
        return jnp.all(self.valid_mask == 1, axis=1).astype(jnp.float32)

    def validate(self) -> None:
        actions = np.asarray(self.actions)
        if actions.ndim != 3 or min(actions.shape) <= 0:
            raise ValueError("actions must have nonempty shape (B,H,A)")
        b, h, _ = actions.shape
        valid = np.asarray(self.valid_mask)
        if valid.shape != (b, h) or not np.isin(valid, (0, 1)).all():
            raise ValueError("valid_mask must be binary with shape (B,H)")
        if not valid[:, 0].all() or (np.diff(valid.astype(int), axis=1) > 0).any():
            raise ValueError("valid_mask must be a nonempty prefix")
        if self.action_space not in ("canonical", "model"):
            raise ValueError("unknown action_space")
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        if self.action_space == "canonical" and (np.abs(actions) > 1).any():
            raise ValueError("actions must be finite and normalized to [-1,1]")
        if (actions[valid == 0] != 0).any():
            raise ValueError("padded actions must be zero")
        for name in ("returns", "bootstrap_discount"):
            value = np.asarray(getattr(self, name))
            if value.shape != (b,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape (B,)")
        discount = np.asarray(self.bootstrap_discount)
        if ((discount < 0) | (discount > 1)).any():
            raise ValueError("bootstrap_discount must be in [0,1]")
        if self.reward_discount_sum is not None:
            weights = np.asarray(self.reward_discount_sum)
            if weights.shape != (b,) or not np.isfinite(weights).all() or (weights < 1).any() or (weights > valid.sum(axis=1)).any():
                raise ValueError("reward_discount_sum must be finite and within [1,valid_steps]")
        if jax.tree_util.tree_structure(self.observations) != jax.tree_util.tree_structure(
            self.bootstrap_observations
        ):
            raise ValueError("observation trees must match")
        leaves = jax.tree_util.tree_leaves(self.observations)
        next_leaves = jax.tree_util.tree_leaves(self.bootstrap_observations)
        if not leaves:
            raise ValueError("observation tree must not be empty")
        for obs, nxt in zip(leaves, next_leaves):
            obs, nxt = np.asarray(obs), np.asarray(nxt)
            if obs.ndim < 1 or obs.shape[0] != b or obs.shape != nxt.shape:
                raise ValueError("observation leaves must have matching (B,...) shapes")
            if not np.isfinite(obs).all() or not np.isfinite(nxt).all():
                raise ValueError("observations must be finite")


def make_chunk_batch(*, observations, actions, rewards, valid_mask,
                     terminated, truncated, bootstrap_observations,
                     discount: float) -> ChunkBatch:
    """Pack already segmented sequences, never concatenate across episodes.

    rewards/terminated/truncated/valid_mask: (B,H). Rewards are per-step,
    NOT already accumulated. The caller supplies the final observation before
    reset, including at a time limit. Only terminated clears bootstrapping.
    Episode IDs and temporal continuity must be checked by the dataset adapter.
    """
    if not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("discount must be in [0,1]")
    valid = np.asarray(valid_mask)
    if valid.ndim != 2 or not np.isin(valid, (0, 1)).all():
        raise ValueError("valid_mask must be binary with shape (B,H)")
    rewards = np.asarray(rewards, dtype=np.float32)
    term, trunc = np.asarray(terminated), np.asarray(truncated)
    for name, value in (("rewards", rewards), ("terminated", term), ("truncated", trunc)):
        if value.shape != valid.shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape (B,H)")
    if not np.isin(term, (0, 1)).all() or not np.isin(trunc, (0, 1)).all():
        raise ValueError("termination flags must be binary")
    boundaries = (term.astype(bool) | trunc.astype(bool)) & valid.astype(bool)
    if (boundaries[:, :-1] & valid[:, 1:].astype(bool)).any():
        raise ValueError("chunk crosses an episode boundary")
    lengths = valid.sum(axis=1).astype(int)
    returns = (rewards * valid * discount ** np.arange(valid.shape[1])).sum(axis=1)
    bootstrap = discount ** lengths * (~(term.astype(bool) & valid.astype(bool)).any(axis=1))
    batch = ChunkBatch(
        observations=jax.tree_util.tree_map(jnp.asarray, observations),
        actions=jnp.asarray(actions, dtype=jnp.float32),
        valid_mask=jnp.asarray(valid, dtype=jnp.float32),
        returns=jnp.asarray(returns, dtype=jnp.float32),
        bootstrap_observations=jax.tree_util.tree_map(jnp.asarray, bootstrap_observations),
        bootstrap_discount=jnp.asarray(bootstrap, dtype=jnp.float32),
        reward_discount_sum=jnp.asarray((valid * discount ** np.arange(valid.shape[1])).sum(axis=1), dtype=jnp.float32),
    )
    batch.validate()
    return batch
