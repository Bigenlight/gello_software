"""Small contract tests for the standalone JAX flow-matching policy."""

from __future__ import annotations

import numpy as np
import pytest

# The actor venv has no jax; without this gate the whole-suite run there dies
# at collection instead of skipping this file.
pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from ur_env.learner.flow_matching import (
    FlowMatchingConfig,
    FlowMatchingPolicy,
    make_flow_training_batch,
    masked_velocity_losses,
    sample_action_chunks,
)


def _small_config() -> FlowMatchingConfig:
    return FlowMatchingConfig(
        horizon=3,
        spatial_features=2,
        camera_bottleneck_dim=8,
        proprio_bottleneck_dim=4,
        hidden_dim=16,
        residual_blocks=1,
        time_embedding_dim=8,
        integration_steps=2,
    )


def _observations(batch_size: int = 2):
    return {
        "cam1": jnp.zeros((batch_size, 1, 4, 4, 512), dtype=jnp.float32),
        "cam2": jnp.zeros((batch_size, 1, 4, 4, 512), dtype=jnp.float32),
        "state": jnp.zeros((batch_size, 1, 19), dtype=jnp.float32),
    }


def test_flow_path_and_model_shapes_are_exact():
    config = _small_config()
    model = FlowMatchingPolicy(config)
    actions = jnp.zeros((2, config.horizon, 7), dtype=jnp.float32)
    noisy, timestep, target = make_flow_training_batch(
        actions, jax.random.PRNGKey(0)
    )
    params = model.init(
        jax.random.PRNGKey(1), _observations(), noisy, timestep
    )["params"]
    predicted = model.apply(
        {"params": params}, _observations(), noisy, timestep
    )
    assert predicted.shape == target.shape == actions.shape
    assert np.isfinite(np.asarray(predicted)).all()


def test_masked_loss_ignores_padded_chunk_steps():
    predicted = jnp.zeros((1, 2, 7), dtype=jnp.float32)
    target = jnp.zeros_like(predicted).at[:, 1, :].set(1000.0)
    losses = masked_velocity_losses(
        predicted, target, jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    )
    assert float(losses["loss"]) == pytest.approx(0.0)
    assert float(losses["continuous_velocity_mse"]) == pytest.approx(0.0)
    assert float(losses["gripper_velocity_mse"]) == pytest.approx(0.0)


def test_sampler_keeps_gripper_inside_the_same_flow_then_discretizes_it():
    config = _small_config()
    model = FlowMatchingPolicy(config)
    actions = jnp.zeros((2, config.horizon, 7), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(2),
        _observations(),
        actions,
        jnp.zeros((2,), dtype=jnp.float32),
    )["params"]
    sampled = sample_action_chunks(
        model, params, _observations(), jax.random.PRNGKey(3)
    )
    assert sampled.shape == actions.shape
    assert set(np.unique(np.asarray(sampled[..., 6]))) <= {-1.0, 1.0}
    assert np.max(np.abs(np.asarray(sampled[..., :6]))) <= 1.0


def test_invalid_action_dimension_is_rejected():
    with pytest.raises(ValueError, match="action_dim"):
        FlowMatchingConfig(action_dim=6)
