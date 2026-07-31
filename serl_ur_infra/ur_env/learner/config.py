"""Validated constants for the local HIL-SERL learner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


RESNET10_SHA256 = (
    "175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b"
)
FROZEN_TRUNK_REPRESENTATION = "resnet10_frozen_trunk_map_f32_v1"
FROZEN_TRUNK_MODEL_REVISION = "hil-serl-hybrid-sac-resnet10-trunk-cache-v1"
FROZEN_TRUNK_SYNTHETIC_E2E_MODEL_REVISION = (
    "hil-serl-hybrid-sac-resnet10-trunk-cache-synthetic-e2e-v1"
)
FROZEN_TRUNK_FEATURE_SHAPE = (1, 4, 4, 512)
LEARNER_AUGMENTATION = "none"


@dataclass(frozen=True)
class LearnerConfig:
    seed: int = 42
    batch_size: int = 256
    online_fraction: float = 0.5
    utd_ratio: int = 10
    cta_ratio: int = 2
    training_starts: int = 100
    publish_period: int = 50
    checkpoint_period: int = 5_000
    log_period: int = 1
    discount: float = 0.97
    encoder_type: str = "resnet-pretrained"
    image_keys: tuple[str, str] = ("cam1", "cam2")
    observation_representation: str = FROZEN_TRUNK_REPRESENTATION
    augmentation: str = LEARNER_AUGMENTATION
    wandb_mode: str = "offline"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in (
            "batch_size",
            "utd_ratio",
            "cta_ratio",
            "training_starts",
            "publish_period",
            "checkpoint_period",
            "log_period",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size % 2:
            raise ValueError("batch_size must be even for 50:50 RLPD sampling")
        if not math.isclose(self.online_fraction, 0.5):
            raise ValueError("online_fraction must remain 0.5")
        if self.cta_ratio != 2:
            raise ValueError("cta_ratio must remain 2 for this milestone")
        if self.checkpoint_period % self.publish_period:
            raise ValueError(
                "checkpoint_period must fall on a policy publish boundary"
            )
        if not math.isfinite(self.discount) or not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be finite and in [0, 1]")
        if self.encoder_type != "resnet-pretrained":
            raise ValueError("encoder_type must remain 'resnet-pretrained'")
        if tuple(self.image_keys) != ("cam1", "cam2"):
            raise ValueError("image_keys must be exactly ('cam1', 'cam2')")
        if self.observation_representation != FROZEN_TRUNK_REPRESENTATION:
            raise ValueError(
                "observation_representation must be "
                f"{FROZEN_TRUNK_REPRESENTATION!r}"
            )
        if self.augmentation != LEARNER_AUGMENTATION:
            raise ValueError("augmentation must remain 'none'")
        if self.wandb_mode not in {"offline", "online", "disabled"}:
            raise ValueError("wandb_mode must be offline, online, or disabled")

    def fingerprint_values(self) -> dict[str, Any]:
        """Return algorithm values which must match when resuming."""

        values = asdict(self)
        # Logging transport is operational state, not part of the learned
        # policy or update rule.  A Kanu run may safely switch between W&B
        # offline/disabled/online while resuming the same learner lineage.
        values.pop("wandb_mode")
        values["image_keys"] = list(self.image_keys)
        return values
