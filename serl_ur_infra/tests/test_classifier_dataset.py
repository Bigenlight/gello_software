"""Re-exporting classifier takes must reproduce the shipped export exactly.

The synthetic tests always run.  The bit-exactness test needs the real corpus
(``~/hil-serl-data``) and is skipped without it -- it is the one that licenses
trusting this module at a DIFFERENT crop, so it is worth running wherever the
data lives.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.learner.classifier_dataset import (  # noqa: E402
    ClassifierDatasetError,
    classifier_item,
    convert_dataset,
    convert_take,
    iter_take_directories,
    load_take_timeline,
)
from ur_env.observation_preprocess import CropBox, preprocess_frame  # noqa: E402


_DATA_ROOT = Path.home() / "hil-serl-data"
_SHIPPED_TAKE = (
    _DATA_ROOT
    / "datasets/cube_in_cup_raw_0724/success_0724/take_01_20260724_213727"
)
_SHIPPED_PICKLE = (
    _DATA_ROOT
    / "datasets/cube_in_cup_all3/train/classifier_data"
    / "take_01_20260724_213727_success.pkl"
)

_CUBE_IN_CUP_CAM1 = CropBox(20, 670, 340, 990)


def _write_take(directory: Path, *, frames: int, seed: int = 0) -> Path:
    """Build a tiny take: two videos plus the camera streams of vectors.h5."""

    h5py = pytest.importorskip("h5py")

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for camera in ("cam1", "cam2"):
        writer = cv2.VideoWriter(
            str(directory / f"{camera}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (64, 64),
        )
        assert writer.isOpened()
        for _ in range(frames):
            writer.write(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        writer.release()
    with h5py.File(directory / "vectors.h5", "w") as handle:
        for camera in ("cam1", "cam2"):
            group = handle.create_group(f"{camera}_frames")
            group["frame_idx"] = np.arange(frames, dtype=np.float64)
            group["t_rel_s"] = np.arange(frames, dtype=np.float64) / 30.0
    return directory


def test_item_layout_matches_the_upstream_classifier_format():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    item = classifier_item({"cam1": image, "cam2": image}, take="t", t_rel_s=0.5)

    assert list(item) == [
        "observations",
        "actions",
        "next_observations",
        "rewards",
        "masks",
        "dones",
        "metadata",
    ]
    assert set(item["observations"]) == {"state", "cam1", "cam2"}
    assert item["observations"]["state"].shape == (1, 1)
    assert item["observations"]["cam1"].shape == (1, 128, 128, 3)
    assert item["actions"].shape == (7,) and item["actions"].dtype == np.float32
    # The label is the buffer, never the item: see the module docstring.
    assert float(item["rewards"]) == 0.0
    assert float(item["masks"]) == 1.0
    assert item["dones"] is False
    assert item["observations"] is item["next_observations"]
    assert item["metadata"] == {"take": "t", "t_rel_s": 0.5}


def test_item_rejects_a_missing_or_malformed_camera():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    with pytest.raises(ClassifierDatasetError, match="missing cam2"):
        classifier_item({"cam1": image}, take="t", t_rel_s=0.0)
    with pytest.raises(ClassifierDatasetError, match="must be"):
        classifier_item(
            {"cam1": image, "cam2": image.astype(np.float32)}, take="t", t_rel_s=0.0
        )


def test_convert_take_reads_the_camera_streams_and_honours_stride(tmp_path):
    take = _write_take(tmp_path / "take_01_synthetic", frames=9)

    timeline = load_take_timeline(take)
    assert len(timeline) == 9

    every = convert_take(take, crop=None, size=(32, 32))
    strided = convert_take(take, crop=None, size=(32, 32), stride=3)

    assert len(every) == 9
    assert len(strided) == 3
    assert every[0]["observations"]["cam1"].shape == (1, 32, 32, 3)
    assert [item["metadata"]["t_rel_s"] for item in strided] == [
        every[index]["metadata"]["t_rel_s"] for index in (0, 3, 6)
    ]
    assert all(item["metadata"]["take"] == take.name for item in every)


def test_crop_changes_the_pixels_and_nothing_else(tmp_path):
    take = _write_take(tmp_path / "take_02_synthetic", frames=4, seed=3)

    uncropped = convert_take(take, crop=None, size=(32, 32))
    cropped = convert_take(take, crop=CropBox(0, 32, 0, 32), size=(32, 32))

    assert len(uncropped) == len(cropped)
    for left, right in zip(uncropped, cropped):
        assert left["metadata"] == right["metadata"]
        assert float(left["rewards"]) == float(right["rewards"])
        assert not np.array_equal(
            left["observations"]["cam1"], right["observations"]["cam1"]
        )


def test_convert_dataset_concatenates_in_order(tmp_path):
    first = _write_take(tmp_path / "take_01_a", frames=3, seed=1)
    second = _write_take(tmp_path / "take_02_b", frames=2, seed=2)

    items = convert_dataset([first, second], crop=None, size=(32, 32))

    assert [item["metadata"]["take"] for item in items] == [first.name] * 3 + [
        second.name
    ] * 2
    assert list(iter_take_directories(tmp_path)) == [first, second]

    with pytest.raises(ClassifierDatasetError, match="no takes"):
        convert_dataset([], crop=None)
    with pytest.raises(ClassifierDatasetError, match="stride"):
        convert_take(first, crop=None, stride=0)


def test_a_take_without_vectors_is_refused(tmp_path):
    empty = tmp_path / "take_03_empty"
    empty.mkdir()
    with pytest.raises(ClassifierDatasetError, match="no vectors.h5"):
        load_take_timeline(empty)


@pytest.mark.skipif(
    not (_SHIPPED_TAKE.is_dir() and _SHIPPED_PICKLE.is_file()),
    reason="needs the real corpus at ~/hil-serl-data",
)
def test_uncropped_reexport_is_bit_exact_against_the_shipped_pickle():
    """The licence to trust this module at a different crop.

    checkpoint_150 was trained from this pickle.  If re-exporting at crop=None
    reproduces it byte for byte, then re-exporting at the task's crop differs
    only by the argument -- which is the whole claim.
    """

    shipped = pickle.load(_SHIPPED_PICKLE.open("rb"))
    timeline = load_take_timeline(_SHIPPED_TAKE)
    assert len(shipped) == len(timeline)

    # Converting ~1,800 frames twice is slow; check a spread of rows instead.
    sample = [0, 1, 2, len(shipped) // 2, len(shipped) - 1]
    wanted = {int(timeline.cam1_index[position]) for position in sample}

    capture = cv2.VideoCapture(str(_SHIPPED_TAKE / "cam1.mp4"))
    decoded: dict[int, np.ndarray] = {}
    try:
        for index in range(max(wanted) + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                decoded[index] = frame
    finally:
        capture.release()

    for position in sample:
        index = int(timeline.cam1_index[position])
        ours = preprocess_frame(decoded[index], crop=None)
        theirs = np.asarray(shipped[position]["observations"]["cam1"])[0]
        np.testing.assert_array_equal(ours, theirs)
        assert float(shipped[position]["metadata"]["t_rel_s"]) == pytest.approx(
            float(timeline.cam1_time[position])
        )


@pytest.mark.skipif(
    not _SHIPPED_TAKE.is_dir(), reason="needs the real corpus at ~/hil-serl-data"
)
def test_the_task_crop_fits_the_real_frames():
    """A 650x650 window has to exist inside the recorded 1280x720 frame."""

    capture = cv2.VideoCapture(str(_SHIPPED_TAKE / "cam1.mp4"))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok and frame.shape == (720, 1280, 3)

    cropped = preprocess_frame(frame, crop=_CUBE_IN_CUP_CAM1)
    assert cropped.shape == (128, 128, 3)
    assert not np.array_equal(cropped, preprocess_frame(frame, crop=None))
