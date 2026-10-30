"""Fixed training-split statistics shared by policy, critic and inference.

No running updates and no implicit fitting during model initialization.
Quantile normalization is affine and reversible: it does not clip outliers.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .vision import VisionConfig, encode_features


MODES = ("none", "mean_std", "min_max", "quantile")


@dataclass(frozen=True)
class NormalizationConfig:
    state: str = "none"
    features: str = "none"
    actions: str = "none"
    normalize_state_gripper: bool = False
    normalize_action_gripper: bool = False
    quantile_low: float = 0.01
    quantile_high: float = 0.99
    epsilon: float = 1e-6
    reward_scale: float = 1.0
    reward_bias: float = 0.0

    def __post_init__(self):
        for name in ("state", "features", "actions"):
            if getattr(self, name) not in MODES:
                raise ValueError(f"{name} normalization must be one of {MODES}")
        if not 0 <= self.quantile_low < self.quantile_high <= 1:
            raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        if not math.isfinite(self.reward_bias):
            raise ValueError("reward_bias must be finite")
        for name in ("normalize_state_gripper", "normalize_action_gripper"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class AffineStats:
    offset: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self):
        offset, scale = np.asarray(self.offset), np.asarray(self.scale)
        if offset.ndim != 1 or scale.shape != offset.shape or not offset.size:
            raise ValueError("normalization statistics must be nonempty matching vectors")
        if not np.isfinite(offset).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("normalization statistics must be finite with positive scales")

    @classmethod
    def identity(cls, dimension):
        return cls((0.,) * dimension, (1.,) * dimension)

    def normalize(self, values):
        if values.shape[-1] != len(self.scale):
            raise ValueError("normalization dimension mismatch")
        return (values - jnp.asarray(self.offset)) / jnp.asarray(self.scale)

    def unnormalize(self, values):
        if values.shape[-1] != len(self.scale):
            raise ValueError("normalization dimension mismatch")
        return values * jnp.asarray(self.scale) + jnp.asarray(self.offset)


def fit_affine(values, mode, config=NormalizationConfig(), *, exclude=()):
    """Fit per final dimension on a finite (N,D) training array."""
    if mode not in MODES:
        raise ValueError(f"normalization must be one of {MODES}")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
        raise ValueError("fit data must be nonempty finite (N,D)")
    if mode == "none":
        return AffineStats.identity(values.shape[-1])
    if mode == "mean_std":
        offset, scale = values.mean(axis=0), values.std(axis=0)
    else:
        if mode == "min_max":
            low, high = values.min(axis=0), values.max(axis=0)
        else:
            low, high = np.quantile(values, [config.quantile_low, config.quantile_high], axis=0)
        offset, scale = (low + high) / 2, (high - low) / 2
    # Constant/near-constant channels must not amplify tiny measurement noise.
    scale = np.where(scale < config.epsilon, 1., scale)
    for index in exclude:
        offset[index], scale[index] = 0., 1.
    return AffineStats(tuple(offset.tolist()), tuple(scale.tolist()))


@dataclass(frozen=True)
class Normalization:
    config: NormalizationConfig
    vision: VisionConfig
    state: AffineStats
    features: AffineStats
    actions: AffineStats

    def __post_init__(self):
        feature_dim = 16384 if self.vision.pooling == "flatten" else 2048
        if len(self.state.scale) != 19 or len(self.actions.scale) != 7 or len(self.features.scale) != feature_dim:
            raise ValueError("statistics do not match the vision/state/action contract")

    @classmethod
    def identity(cls, vision=VisionConfig()):
        dim = 16384 if vision.pooling == "flatten" else 2048
        return cls(NormalizationConfig(), vision, AffineStats.identity(19),
                   AffineStats.identity(dim), AffineStats.identity(7))

    def observations(self, observations):
        vector = encode_features(observations, self.vision)
        return jnp.concatenate((self.features.normalize(vector[..., :-19]),
                                self.state.normalize(vector[..., -19:])), axis=-1)

    def batch(self, batch):
        if batch.action_space != "canonical":
            raise ValueError("normalization expects canonical actions, not an already normalized batch")
        batch.validate()
        rewards = self.config.reward_scale * batch.returns
        if self.config.reward_bias:
            if batch.reward_discount_sum is None:
                raise ValueError("reward_bias requires reward_discount_sum from make_chunk_batch")
            rewards = rewards + self.config.reward_bias * batch.reward_discount_sum
        return batch.replace(
            observations=self.observations(batch.observations),
            bootstrap_observations=self.observations(batch.bootstrap_observations),
            actions=jnp.where(batch.valid_mask[..., None] == 1,
                              self.actions.normalize(batch.actions), 0.),
            returns=rewards, action_space="model",
        )

    def document(self):
        return {"format": "offline-rl-normalization", "version": 1, **asdict(self)}

    def save(self, path):
        """Save alongside model parameters; refuse to overwrite existing stats."""
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(self.document(), stream, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("format") != "offline-rl-normalization" or document.get("version") != 1:
            raise ValueError("unsupported normalization document")
        stats = {name: AffineStats(tuple(document[name]["offset"]), tuple(document[name]["scale"]))
                 for name in ("state", "features", "actions")}
        return cls(NormalizationConfig(**document["config"]), VisionConfig(**document["vision"]), **stats)


def fit_training_normalization(observations, actions, *, config=NormalizationConfig(),
                               vision=VisionConfig(), valid_mask=None):
    """Fit ONLY on caller-selected training data, never validation/test episodes.

    observations contain distinct training observations, without padding.
    actions may be (N,7) or (B,H,7); valid_mask has shape actions.shape[:-1].
    All time positions share the same seven statistics. Prefer distinct
    transitions to overlapping chunks, which reweight repeated actions.
    """
    vector = np.asarray(encode_features(observations, vision))
    actions = np.asarray(actions)
    if actions.ndim not in (2, 3) or actions.shape[-1] != 7:
        raise ValueError("training actions must have shape (N,7) or (B,H,7)")
    if valid_mask is None:
        valid_mask = np.ones(actions.shape[:-1], dtype=bool)
    valid_mask = np.asarray(valid_mask)
    if valid_mask.shape != actions.shape[:-1] or not np.isin(valid_mask, (0, 1)).all():
        raise ValueError("invalid action statistics mask")
    valid_actions = actions[valid_mask.astype(bool)]
    if not np.isfinite(valid_actions).all() or (np.abs(valid_actions) > 1).any():
        raise ValueError("fit actions must use the canonical [-1,1] contract")
    return Normalization(
        config, vision,
        fit_affine(vector[:, -19:], config.state, config,
                   exclude=() if config.normalize_state_gripper else (0,)),
        fit_affine(vector[:, :-19], config.features, config),
        fit_affine(valid_actions, config.actions, config,
                   exclude=() if config.normalize_action_gripper else (6,)),
    )
