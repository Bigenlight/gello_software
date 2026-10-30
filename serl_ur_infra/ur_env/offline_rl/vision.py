"""Parameter-free reduction of cached frozen features, shared by all heads."""

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class VisionConfig:
    pooling: str = "flatten"
    spatial_softmax_temperature: float = 1.0

    def __post_init__(self):
        if self.pooling not in ("flatten", "spatial_softmax"):
            raise ValueError("pooling must be flatten or spatial_softmax")
        if not math.isfinite(self.spatial_softmax_temperature) or self.spatial_softmax_temperature <= 0:
            raise ValueError("spatial_softmax_temperature must be finite and positive")


def spatial_softmax(features, temperature=1.0):
    """(B,H,W,C) -> (B,2*C), expected x then y per channel in [-1,1].

    Softmax normalizes over H*W independently for each channel. It returns
    coordinates, not a flattened probability map. Temperature is fixed.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if features.ndim != 4 or min(features.shape) <= 0:
        raise ValueError("features must have nonempty shape (B,H,W,C)")
    b, h, w, c = features.shape
    # A singleton spatial dimension has its coordinate at the center.
    x = jnp.linspace(-1., 1., w) if w > 1 else jnp.zeros(1)
    y = jnp.linspace(-1., 1., h) if h > 1 else jnp.zeros(1)
    xx, yy = jnp.meshgrid(x, y, indexing="xy")
    weights = jax.nn.softmax(features.reshape((b, h * w, c)) / temperature, axis=1)
    expected_x = jnp.sum(weights * xx.reshape((1, h * w, 1)), axis=1)
    expected_y = jnp.sum(weights * yy.reshape((1, h * w, 1)), axis=1)
    return jnp.concatenate((expected_x, expected_y), axis=-1)


def encode_features(observations, config=VisionConfig()):
    """Use the canonical two-camera cached feature contract, not raw pixels."""
    if set(observations) != {"cam1", "cam2", "state"}:
        raise ValueError("observations must contain cam1, cam2, state")
    state = jnp.asarray(observations["state"], dtype=jnp.float32)
    if state.ndim != 3 or state.shape[1:] != (1, 19):
        raise ValueError("state must have shape (B,1,19)")
    vectors = []
    for name in ("cam1", "cam2"):
        features = jnp.asarray(observations[name], dtype=jnp.float32)
        if features.shape != (state.shape[0], 1, 4, 4, 512):
            raise ValueError(f"{name} must have shape (B,1,4,4,512)")
        features = jax.lax.stop_gradient(features[:, 0])
        if config.pooling == "flatten":
            vectors.append(features.reshape((features.shape[0], -1)))
        else:
            vectors.append(spatial_softmax(features, config.spatial_softmax_temperature))
    return jnp.concatenate((*vectors, state[:, 0]), axis=-1)
