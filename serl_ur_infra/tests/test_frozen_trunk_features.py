"""Lightweight frozen-trunk replay contract tests."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


_INFRA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_INFRA))

from ur_env.learner.frozen_trunk import (  # noqa: E402
    FROZEN_TRUNK_CONTRACT,
    FROZEN_TRUNK_FEATURE_CONTRACT_REVISION,
    FROZEN_TRUNK_FEATURE_DTYPE,
    FROZEN_TRUNK_FEATURE_SHAPE,
    FrozenTrunkFeatureSchemaError,
    validate_frozen_trunk_feature,
    validate_frozen_trunk_observation,
)


def _feature(value: float = 0.0) -> np.ndarray:
    return np.full(FROZEN_TRUNK_FEATURE_SHAPE, value, dtype=np.float32)


def test_frozen_trunk_contract_names_exact_cut_and_disables_augmentation():
    document = FROZEN_TRUNK_CONTRACT.document()

    assert FROZEN_TRUNK_FEATURE_CONTRACT_REVISION == (
        "resnet10_frozen_trunk_map_f32_v1"
    )
    assert document["feature_shape"] == [1, 4, 4, 512]
    assert document["feature_dtype"] == "float32"
    assert document["cut_point"] == "pretrained_resnet10.stop_gradient"
    assert document["pixel_augmentation"] == "none"
    assert "SpatialLearnedEmbeddings" in document["downstream_head"]
    assert FROZEN_TRUNK_FEATURE_DTYPE == np.dtype(np.float32)


def test_feature_schema_accepts_single_and_batch_without_casting():
    single = _feature(0.5)
    assert validate_frozen_trunk_feature(single) is single

    observation = {
        "state": np.zeros((1, 19), dtype=np.float32),
        "cam1": single,
        "cam2": _feature(-0.5),
    }
    validated = validate_frozen_trunk_observation(observation)
    assert validated["cam1"] is single

    batched = {
        key: np.stack([value, value], axis=0)
        for key, value in observation.items()
    }
    checked_batch = validate_frozen_trunk_observation(batched, batched=True)
    assert checked_batch["cam1"].shape == (2, 1, 4, 4, 512)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.astype(np.float16),
            "dtype float32",
        ),
        (
            lambda value: value[..., :511],
            "shape",
        ),
        (
            lambda value: value.copy(),
            "non-finite",
        ),
    ],
)
def test_feature_schema_rejects_dtype_shape_and_nonfinite(mutate, message):
    value = mutate(_feature())
    if message == "non-finite":
        value.reshape(-1)[0] = np.nan

    with pytest.raises(FrozenTrunkFeatureSchemaError, match=message):
        validate_frozen_trunk_feature(value, name="cam1")


def test_feature_observation_rejects_key_and_batch_drift():
    observation = {
        "state": np.zeros((1, 19), dtype=np.float32),
        "cam1": _feature(),
        "cam2": _feature(),
    }
    with pytest.raises(FrozenTrunkFeatureSchemaError, match="keys mismatch"):
        validate_frozen_trunk_observation({**observation, "pixels": _feature()})

    batched = {
        "state": np.zeros((2, 1, 19), dtype=np.float32),
        "cam1": np.zeros((3, 1, 4, 4, 512), dtype=np.float32),
        "cam2": np.zeros((2, 1, 4, 4, 512), dtype=np.float32),
    }
    with pytest.raises(FrozenTrunkFeatureSchemaError, match="batch size"):
        validate_frozen_trunk_observation(batched, batched=True)
