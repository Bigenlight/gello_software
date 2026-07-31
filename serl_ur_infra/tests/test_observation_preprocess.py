"""The canonical pixel recipe must reproduce every recipe it will replace.

Nothing calls ``observation_preprocess`` in production yet.  These tests are
what make adopting it safe: each existing consumer's recipe is re-implemented
inline here, exactly as that consumer spells it, and asserted bit-exact
against the shared function at the matching rule.  If a later rewiring changes
a single pixel, one of these fails.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.observation_preprocess import (  # noqa: E402
    CANONICAL_IMAGE_SIZE,
    CropBox,
    PreprocessRule,
    PreprocessRuleError,
    preprocess_cameras,
    preprocess_dataset,
    preprocess_frame,
)


#: The live crops from ``ur_experiments/cube_in_cup.py``, as data.
CUBE_IN_CUP_CROPS = {
    "cam1": CropBox(20, 670, 340, 990),   # lambda img: img[20:670, 340:990]
    "cam2": CropBox(0, 720, 420, 1140),   # lambda img: img[0:720, 420:1140]
}


def _frame(seed: int, *, height: int = 720, width: int = 1280) -> np.ndarray:
    """A deterministic non-uniform BGR frame.

    Structure matters: a flat fill would pass a resize comparison even if the
    interpolation kernel differed.
    """

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_crop_boxes_match_the_shipped_cube_in_cup_windows():
    assert CUBE_IN_CUP_CROPS["cam1"].height == 650
    assert CUBE_IN_CUP_CROPS["cam1"].width == 650
    assert CUBE_IN_CUP_CROPS["cam2"].height == 720
    assert CUBE_IN_CUP_CROPS["cam2"].width == 720

    frame = _frame(1)
    for camera, crop in CUBE_IN_CUP_CROPS.items():
        y0, y1, x0, x1 = crop.as_tuple()
        np.testing.assert_array_equal(crop.apply(frame), frame[y0:y1, x0:x1])


@pytest.mark.parametrize("camera", ["cam1", "cam2"])
def test_bit_exact_against_the_policy_get_im_recipe(camera):
    """``UR7eEnv.get_im``: crop -> cv2.resize -> reverse channels."""

    frame = _frame(11)
    crop = CUBE_IN_CUP_CROPS[camera]

    y0, y1, x0, x1 = crop.as_tuple()
    cropped = frame[y0:y1, x0:x1]
    resized = cv2.resize(cropped, CANONICAL_IMAGE_SIZE)
    expected = resized[..., ::-1]  # get_im keeps the reversed view

    actual = preprocess_frame(frame, crop=crop)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
    assert actual.shape == (128, 128, 3)
    # The values match a view; the shared function additionally guarantees the
    # contiguity that get_im leaves to validate_canonical_observation.
    assert actual.flags["C_CONTIGUOUS"]


def test_bit_exact_against_the_classifier_sidecar_recipe():
    """``decode_classifier_frames``: no crop -> cv2.resize -> RGB -> contiguous."""

    frame = _frame(12)

    resized = cv2.resize(frame, CANONICAL_IMAGE_SIZE)
    expected = np.ascontiguousarray(resized[..., ::-1], dtype=np.uint8)

    actual = preprocess_frame(frame, crop=None)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (128, 128, 3)


@pytest.mark.parametrize("camera", ["cam1", "cam2"])
def test_bit_exact_against_the_recorded_demo_recipe(camera):
    """``recorded_demo``: crop -> resize -> ascontiguousarray(RGB)."""

    frame = _frame(13)
    crop = CUBE_IN_CUP_CROPS[camera]

    y0, y1, x0, x1 = crop.as_tuple()
    cropped = np.asarray(frame[y0:y1, x0:x1])
    resized = cv2.resize(cropped, (128, 128))
    expected = np.ascontiguousarray(resized[..., ::-1], dtype=np.uint8)

    np.testing.assert_array_equal(preprocess_frame(frame, crop=crop), expected)


def test_uncropped_policy_and_classifier_inputs_become_identical():
    """The whole point of the rule: one crop setting collapses G15."""

    frame = _frame(14)

    policy_like = preprocess_frame(frame, crop=None)
    classifier_like = preprocess_frame(frame, crop=None)
    np.testing.assert_array_equal(policy_like, classifier_like)

    # ...and they are NOT identical while the crop rule differs, which is
    # exactly the divergence G15 measured.
    cropped = preprocess_frame(frame, crop=CUBE_IN_CUP_CROPS["cam1"])
    assert not np.array_equal(cropped, classifier_like)


def test_dataset_conversion_is_frame_conversion_stacked():
    frames = np.stack([_frame(seed, height=200, width=320) for seed in (1, 2, 3)])
    crop = CropBox(0, 200, 0, 200)

    batch = preprocess_dataset(frames, crop=crop)

    assert batch.shape == (3, 128, 128, 3)
    assert batch.dtype == np.uint8
    for index in range(3):
        np.testing.assert_array_equal(
            batch[index], preprocess_frame(frames[index], crop=crop)
        )
    # A list of frames is the same dataset.
    np.testing.assert_array_equal(batch, preprocess_dataset(list(frames), crop=crop))


def test_preprocess_cameras_applies_one_rule_and_refuses_unlisted_cameras():
    rule = PreprocessRule(crops=CUBE_IN_CUP_CROPS)
    frames = {"cam1": _frame(21), "cam2": _frame(22)}

    out = preprocess_cameras(frames, rule)

    assert set(out) == {"cam1", "cam2"}
    for camera, image in out.items():
        np.testing.assert_array_equal(
            image, preprocess_frame(frames[camera], crop=CUBE_IN_CUP_CROPS[camera])
        )

    with pytest.raises(PreprocessRuleError, match="no entry for cameras"):
        preprocess_cameras({"cam3": _frame(23)}, rule)


def test_the_rule_is_serializable_and_its_tag_tracks_every_change():
    cropped = PreprocessRule(crops=CUBE_IN_CUP_CROPS)
    uncropped = PreprocessRule(crops={"cam1": None, "cam2": None})

    assert cropped.describe() == {
        "size": [128, 128],
        "crops": {"cam1": [20, 670, 340, 990], "cam2": [0, 720, 420, 1140]},
    }
    assert uncropped.describe()["crops"] == {"cam1": None, "cam2": None}

    tags = {
        cropped.tag(),
        uncropped.tag(),
        PreprocessRule(crops=CUBE_IN_CUP_CROPS, size=(224, 224)).tag(),
        PreprocessRule(
            crops={"cam1": CropBox(20, 670, 340, 990), "cam2": None}
        ).tag(),
    }
    assert len(tags) == 4, "distinct rules must not share a tag"
    # Same rule, rebuilt: stable across processes, so it can be stored.
    assert cropped.tag() == PreprocessRule(crops=dict(CUBE_IN_CUP_CROPS)).tag()


def test_a_non_square_crop_is_rejected_at_construction():
    """A non-square crop resized to a square output distorts aspect."""

    with pytest.raises(PreprocessRuleError, match="must be square"):
        CropBox(0, 720, 0, 1280)
    with pytest.raises(PreprocessRuleError, match="positive height and width"):
        CropBox(100, 100, 0, 0)
    with pytest.raises(PreprocessRuleError, match="must not be negative"):
        CropBox(-1, 649, 0, 650)


def test_bad_frames_and_sizes_are_refused():
    with pytest.raises(PreprocessRuleError, match="must be uint8"):
        preprocess_frame(np.zeros((10, 10, 3), dtype=np.float32))
    with pytest.raises(PreprocessRuleError, match=r"must be \(H, W, 3\)"):
        preprocess_frame(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(PreprocessRuleError, match="size width must be positive"):
        preprocess_frame(_frame(31, height=64, width=64), size=(0, 128))
    with pytest.raises(PreprocessRuleError, match="does not fit"):
        preprocess_frame(_frame(32, height=64, width=64), crop=CropBox(0, 650, 0, 650))
    with pytest.raises(PreprocessRuleError, match="dataset is empty"):
        preprocess_dataset([])
