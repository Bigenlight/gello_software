"""JAX rectified-flow policy for the canonical HIL-SERL action contract.

The model consumes the same cached ResNet-10 feature maps as the learner and
generates an action chunk with shape ``(horizon, 7)``.  All seven channels are
part of one velocity field: six normalized EEF deltas and the gripper channel.
The latter is trained as the recorded continuous ``-1/+1`` value and is only
discretized after ODE integration.

This is intentionally a separate policy family.  Its parameters are not a
drop-in replacement for the hybrid SAC actor or a production learner resume
checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp


ACTION_DIM = 7
EEF_DIM = 6
GRIPPER_INDEX = 6
FEATURE_HEIGHT = 4
FEATURE_WIDTH = 4
FEATURE_CHANNELS = 512
FEATURE_STACK = 1


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Static architecture and sampler configuration stored with each model."""

    horizon: int = 16
    action_dim: int = ACTION_DIM
    spatial_features: int = 8
    camera_bottleneck_dim: int = 128
    proprio_bottleneck_dim: int = 64
    hidden_dim: int = 256
    residual_blocks: int = 4
    time_embedding_dim: int = 64
    integration_steps: int = 8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"action_dim must remain {ACTION_DIM}")
        if self.time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even")

    def document(self) -> dict[str, int]:
        return asdict(self)


class _SpatialFeatureEncoder(nn.Module):
    spatial_features: int
    bottleneck_dim: int

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        if features.ndim != 5 or features.shape[1:] != (
            FEATURE_STACK,
            FEATURE_HEIGHT,
            FEATURE_WIDTH,
            FEATURE_CHANNELS,
        ):
            raise ValueError(
                "camera features must have shape "
                f"(B,{FEATURE_STACK},{FEATURE_HEIGHT},{FEATURE_WIDTH},"
                f"{FEATURE_CHANNELS}), got {features.shape}"
            )
        x = jax.lax.stop_gradient(features[:, 0])
        kernel = self.param(
            "spatial_kernel",
            nn.initializers.lecun_normal(),
            (
                FEATURE_HEIGHT,
                FEATURE_WIDTH,
                FEATURE_CHANNELS,
                self.spatial_features,
            ),
        )
        x = jnp.sum(x[..., None] * kernel[None, ...], axis=(1, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.bottleneck_dim)(x)
        x = nn.LayerNorm()(x)
        return jnp.tanh(x)


class _ResidualBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        residual = inputs
        x = nn.LayerNorm()(inputs)
        x = nn.Dense(self.hidden_dim * 4)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_dim)(x)
        return residual + x


def _time_embedding(timestep: jax.Array, dimension: int) -> jax.Array:
    """Deterministic Fourier features for continuous ``t in [0,1]``."""

    timestep = jnp.asarray(timestep, dtype=jnp.float32).reshape((-1, 1))
    half = dimension // 2
    frequencies = jnp.power(2.0, jnp.arange(half, dtype=jnp.float32))
    angles = jnp.pi * timestep * frequencies[None, :]
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)


class FlowMatchingPolicy(nn.Module):
    """Conditional velocity field over an EEF-delta + gripper action chunk."""

    config: FlowMatchingConfig

    @nn.compact
    def __call__(
        self,
        observations: Mapping[str, jax.Array],
        noisy_actions: jax.Array,
        timestep: jax.Array,
    ) -> jax.Array:
        expected_action_shape = (
            noisy_actions.shape[0],
            self.config.horizon,
            self.config.action_dim,
        )
        if noisy_actions.shape != expected_action_shape:
            raise ValueError(
                f"noisy_actions must have shape {expected_action_shape}, "
                f"got {noisy_actions.shape}"
            )
        if set(observations) != {"cam1", "cam2", "state"}:
            raise ValueError(
                "observations must contain exactly cam1, cam2, state; got "
                f"{sorted(observations)}"
            )

        encoded_cameras = []
        for camera in ("cam1", "cam2"):
            encoded_cameras.append(
                _SpatialFeatureEncoder(
                    spatial_features=self.config.spatial_features,
                    bottleneck_dim=self.config.camera_bottleneck_dim,
                    name=f"encoder_{camera}",
                )(observations[camera])
            )

        state = observations["state"]
        if state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0]
        if state.ndim != 2:
            raise ValueError(f"state must have shape (B,D) or (B,1,D), got {state.shape}")
        state = nn.Dense(self.config.proprio_bottleneck_dim, name="state_dense")(
            state
        )
        state = nn.LayerNorm(name="state_norm")(state)
        state = jnp.tanh(state)

        condition = jnp.concatenate((*encoded_cameras, state), axis=-1)
        action_flat = noisy_actions.reshape((noisy_actions.shape[0], -1))
        time_features = _time_embedding(
            timestep, self.config.time_embedding_dim
        )
        x = jnp.concatenate((condition, action_flat, time_features), axis=-1)
        x = nn.Dense(self.config.hidden_dim, name="input_dense")(x)
        x = nn.silu(x)
        for block_index in range(self.config.residual_blocks):
            x = _ResidualBlock(
                self.config.hidden_dim, name=f"residual_{block_index}"
            )(x)
        x = nn.LayerNorm(name="output_norm")(x)
        x = nn.silu(x)
        velocity = nn.Dense(
            self.config.horizon * self.config.action_dim,
            kernel_init=nn.initializers.zeros_init(),
            name="velocity",
        )(x)
        return velocity.reshape(expected_action_shape)


def make_flow_training_batch(
    actions: jax.Array, rng: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample the linear probability path and its constant velocity target."""

    noise_rng, time_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, actions.shape, dtype=actions.dtype)
    timestep = jax.random.uniform(
        time_rng,
        (actions.shape[0],),
        minval=0.0,
        maxval=1.0,
        dtype=actions.dtype,
    )
    interpolation = timestep[:, None, None]
    noisy_actions = (1.0 - interpolation) * noise + interpolation * actions
    target_velocity = actions - noise
    return noisy_actions, timestep, target_velocity


def masked_velocity_losses(
    predicted_velocity: jax.Array,
    target_velocity: jax.Array,
    valid_mask: jax.Array,
) -> dict[str, jax.Array]:
    """Flow loss with padded chunk positions excluded from every metric."""

    mask = valid_mask.astype(predicted_velocity.dtype)[..., None]
    valid_steps = jnp.maximum(jnp.sum(mask), 1.0)
    squared_error = jnp.square(predicted_velocity - target_velocity) * mask
    continuous = jnp.sum(squared_error[..., :EEF_DIM]) / (
        valid_steps * EEF_DIM
    )
    gripper = jnp.sum(squared_error[..., GRIPPER_INDEX]) / valid_steps
    total = jnp.sum(squared_error) / (valid_steps * ACTION_DIM)
    return {
        "loss": total,
        "continuous_velocity_mse": continuous,
        "gripper_velocity_mse": gripper,
    }


def sample_action_chunks(
    model: FlowMatchingPolicy,
    params,
    observations: Mapping[str, jax.Array],
    rng: jax.Array,
    *,
    integration_steps: int | None = None,
    discretize_gripper: bool = True,
) -> jax.Array:
    """Euler-integrate the learned velocity field from Gaussian noise to data."""

    steps = integration_steps or model.config.integration_steps
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("integration_steps must be a positive integer")
    batch_size = observations["state"].shape[0]
    shape = (batch_size, model.config.horizon, model.config.action_dim)
    initial = jax.random.normal(rng, shape, dtype=jnp.float32)
    dt = jnp.asarray(1.0 / steps, dtype=jnp.float32)

    def integrate(step_index: int, current: jax.Array) -> jax.Array:
        timestep = jnp.full(
            (batch_size,), step_index / steps, dtype=jnp.float32
        )
        velocity = model.apply(
            {"params": params}, observations, current, timestep
        )
        return current + dt * velocity

    actions = jax.lax.fori_loop(0, steps, integrate, initial)
    actions = jnp.clip(actions, -1.0, 1.0)
    if discretize_gripper:
        gripper = jnp.where(
            actions[..., GRIPPER_INDEX] >= 0.0, 1.0, -1.0
        )
        actions = actions.at[..., GRIPPER_INDEX].set(gripper)
    return actions


def load_flow_artifact(
    directory: str | Path, *, which: str = "best"
) -> tuple[FlowMatchingPolicy, object, dict[str, object]]:
    """Verify and restore a ``best`` or ``final`` FM artifact from disk."""

    from flax import serialization

    root = Path(directory).expanduser().resolve()
    completion_path = root / "completion.json"
    manifest_path = root / "manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    if completion.get("complete") is not True:
        raise ValueError(f"incomplete flow artifact: {root}")
    if hashlib.sha256(manifest_bytes).hexdigest() != completion.get(
        "manifest_sha256"
    ):
        raise ValueError("flow artifact manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "hil-serl-jax-flow-matching":
        raise ValueError(f"unsupported flow artifact format: {manifest.get('format')}")
    if which not in {"best", "final"}:
        raise ValueError("which must be 'best' or 'final'")

    file_record = manifest["parameter_files"][which]
    parameter_bytes = (root / file_record["path"]).read_bytes()
    parameter_sha256 = hashlib.sha256(parameter_bytes).hexdigest()
    if parameter_sha256 != file_record["sha256"]:
        raise ValueError(f"{which} flow parameter SHA-256 mismatch")

    config = FlowMatchingConfig(**manifest["model_config"])
    model = FlowMatchingPolicy(config)
    observations = {
        "cam1": jnp.zeros(
            (1, FEATURE_STACK, FEATURE_HEIGHT, FEATURE_WIDTH, FEATURE_CHANNELS),
            dtype=jnp.float32,
        ),
        "cam2": jnp.zeros(
            (1, FEATURE_STACK, FEATURE_HEIGHT, FEATURE_WIDTH, FEATURE_CHANNELS),
            dtype=jnp.float32,
        ),
        "state": jnp.zeros((1, 1, 19), dtype=jnp.float32),
    }
    template = model.init(
        jax.random.PRNGKey(0),
        observations,
        jnp.zeros((1, config.horizon, config.action_dim), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )["params"]
    params = serialization.from_bytes(template, parameter_bytes)
    return model, params, manifest


__all__ = [
    "ACTION_DIM",
    "EEF_DIM",
    "GRIPPER_INDEX",
    "FlowMatchingConfig",
    "FlowMatchingPolicy",
    "make_flow_training_batch",
    "masked_velocity_losses",
    "sample_action_chunks",
    "load_flow_artifact",
]
