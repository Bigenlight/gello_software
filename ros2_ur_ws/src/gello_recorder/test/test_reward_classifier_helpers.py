import json

import cv2
import numpy as np
import pytest

from gello_recorder.reward_classifier_runtime import (
    decode_classifier_image,
    make_observation,
    sigmoid_probability,
    status_json,
    validate_threshold,
)


def _jpeg(color_bgr):
    image = np.empty((24, 32, 3), dtype=np.uint8)
    image[:] = color_bgr
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_decode_matches_checkpoint_contract_and_rgb_order():
    decoded = decode_classifier_image(_jpeg((0, 0, 255)))
    assert decoded.shape == (1, 128, 128, 3)
    assert decoded.dtype == np.uint8
    assert decoded[0, 64, 64, 0] > 240
    assert decoded[0, 64, 64, 2] < 15


def test_observation_keys_shapes_and_dtypes():
    obs = make_observation(_jpeg((1, 2, 3)), _jpeg((4, 5, 6)))
    assert list(obs) == ["state", "cam1", "cam2"]
    assert obs["state"].shape == (1, 1)
    assert obs["state"].dtype == np.float32
    for key in ("cam1", "cam2"):
        assert obs[key].shape == (1, 128, 128, 3)
        assert obs[key].dtype == np.uint8


def test_status_json_is_parseable():
    value = json.loads(status_json(ready=True, probability=0.75))
    assert value == {"probability": 0.75, "ready": True}


@pytest.mark.parametrize(
    ("logit", "expected"),
    [
        (0.0, 0.5),
        (1000.0, 1.0),
        (-1000.0, 0.0),
    ],
)
def test_sigmoid_probability_is_stable(logit, expected):
    assert sigmoid_probability(logit) == pytest.approx(expected)


@pytest.mark.parametrize("logit", [float("nan"), float("inf"), -float("inf")])
def test_sigmoid_probability_rejects_non_finite_logits(logit):
    with pytest.raises(ValueError, match="finite"):
        sigmoid_probability(logit)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_validate_threshold_accepts_closed_unit_interval(threshold):
    assert validate_threshold(threshold) == threshold


@pytest.mark.parametrize(
    "threshold", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")]
)
def test_validate_threshold_rejects_invalid_values(threshold):
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        validate_threshold(threshold)
