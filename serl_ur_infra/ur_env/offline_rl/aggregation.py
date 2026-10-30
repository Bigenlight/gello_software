"""Shared Q-ensemble reductions for offline algorithms."""

import math

import jax.numpy as jnp


Q_AGGREGATIONS = ("min", "mean", "max", "mean_minus_std")


def validate_aggregation(mode: str) -> None:
    if mode not in Q_AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {Q_AGGREGATIONS}, got {mode!r}")


def aggregate_q(qs, *, mode: str, rho: float = 0.5):
    """Reduce axis 0 of (ensemble,B); rho affects mean_minus_std only.

    mode/rho are static configuration values when called from a jitted update.
    Population std (ddof=0) also supports a single-member ensemble.
    """
    validate_aggregation(mode)
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    if qs.ndim != 2 or qs.shape[0] < 1:
        raise ValueError("Q values must have shape (ensemble,B) with ensemble >= 1")
    if mode == "min":
        return jnp.min(qs, axis=0)
    if mode == "max":
        return jnp.max(qs, axis=0)
    if mode == "mean":
        return jnp.mean(qs, axis=0)
    return jnp.mean(qs, axis=0) - rho * jnp.std(qs, axis=0)
