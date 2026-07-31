"""Re-export recorder takes as reward-classifier items under a chosen rule.

The pinned classifier (``classifier_ckpt/checkpoint_150``) was trained on
UNCROPPED frames: its exporter called the pipeline's preprocessing with
``crop=None``, so a full 1280x720 frame was squashed straight to 128x128.  The
policy observation is cropped.  That gap is G15, and it is why the classifier
gets its own sidecar image today.

Sharing one frozen-trunk forward across critic, actor and classifier requires
closing the gap at the source: all three must see the same pixels.  The
shipped classifier pickles cannot be re-cropped -- they hold 128x128 already,
and the full-resolution context a crop needs is gone.  Re-exporting from the
recorder takes is the only way, and this module is that re-export.

WHAT IS FAITHFULLY REPRODUCED
-----------------------------
``tests/test_classifier_dataset`` asserts this module reproduces the shipped
``take_01_20260724_213727_success.pkl`` **bit-exactly** at ``crop=None``.  That
is the licence to trust it at a different crop: the only thing that changes is
the argument.

The item layout, the all-zero placeholder ``state``/``actions``, and the
identical ``observations``/``next_observations`` images are the upstream
classifier format, not an accident.  The classifier's ``EncodingWrapper`` sets
``use_proprio=False``, so nothing reads ``state``.

THE LABEL IS NOT IN THE ITEM
----------------------------
Every exported item carries ``rewards=0.0``, including items from a successful
take.  Upstream labels *by buffer*: the caller puts a take's items into the
success list or the failure list, and the pickle's filename is the label.  The
0724 takes were recorded so that a whole take carries one label -- the cube
starts in the cup or never enters it, and only the robot moves.

So this module deliberately returns items and takes no label argument.  A
converter that wrote ``rewards`` from a guess would produce a dataset that
looks labelled and is not.

No I/O beyond reading the take: assembling buffers and writing pickles stays
with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ur_env.observation_preprocess import CropBox, preprocess_frame


#: Camera keys of one classifier item, in the order the exporter used.
CLASSIFIER_CAMERAS = ("cam1", "cam2")

#: Placeholder proprio.  The classifier's encoder sets use_proprio=False.
_STATE_PLACEHOLDER_SHAPE = (1, 1)
_ACTION_DIM = 7


class ClassifierDatasetError(ValueError):
    """A recorder take cannot be re-exported without breaking the format."""


@dataclass(frozen=True)
class TakeTimeline:
    """Per-camera frame indices and times, as the recorder stored them."""

    cam1_index: np.ndarray
    cam1_time: np.ndarray
    cam2_index: np.ndarray
    cam2_time: np.ndarray

    def __len__(self) -> int:
        # The exporter zips the two camera streams, so the shorter one wins.
        return min(len(self.cam1_index), len(self.cam2_index))


def load_take_timeline(take_dir: Path | str) -> TakeTimeline:
    """Read ``vectors.h5`` camera streams.

    Reads ``cam1_frames`` / ``cam2_frames`` rather than ``synchronized``: the
    latter's ``cam1_frame_idx`` is empty in the 0724 takes, while the former
    has exactly one row per exported item.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ClassifierDatasetError("h5py is required to read a take") from exc

    directory = Path(take_dir)
    path = directory / "vectors.h5"
    if not path.is_file():
        raise ClassifierDatasetError(f"{directory} has no vectors.h5")
    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        for camera in CLASSIFIER_CAMERAS:
            group_name = f"{camera}_frames"
            if group_name not in handle:
                raise ClassifierDatasetError(f"{path} has no {group_name}")
            group = handle[group_name]
            for field in ("frame_idx", "t_rel_s"):
                if field not in group:
                    raise ClassifierDatasetError(
                        f"{path}:{group_name} has no {field}"
                    )
            index = np.asarray(group["frame_idx"][:], dtype=np.int64)
            time = np.asarray(group["t_rel_s"][:], dtype=np.float64)
            if index.ndim != 1 or index.shape != time.shape:
                raise ClassifierDatasetError(
                    f"{path}:{group_name} frame_idx/t_rel_s disagree"
                )
            if index.size and index.min() < 0:
                raise ClassifierDatasetError(
                    f"{path}:{group_name} has a negative frame index"
                )
            streams[camera] = (index, time)
    timeline = TakeTimeline(
        cam1_index=streams["cam1"][0],
        cam1_time=streams["cam1"][1],
        cam2_index=streams["cam2"][0],
        cam2_time=streams["cam2"][1],
    )
    if len(timeline) == 0:
        raise ClassifierDatasetError(f"{directory} has no camera frames")
    return timeline


def _preprocessed_frames(
    video_path: Path,
    wanted: Iterable[int],
    *,
    crop: CropBox | None,
    size: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Decode sequentially, preprocess on the way, keep only what is wanted.

    The take is ~1,800 frames of 1280x720; buffering them raw would cost
    ~4.8 GiB, so each frame is reduced to 128x128 before anything is retained.
    """

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ClassifierDatasetError("opencv-python is required") from exc

    needed = {int(value) for value in wanted}
    if not needed:
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ClassifierDatasetError(f"cannot open video {video_path}")
    kept: dict[int, np.ndarray] = {}
    try:
        for index in range(max(needed) + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index in needed:
                kept[index] = preprocess_frame(frame, crop=crop, size=size)
    finally:
        capture.release()
    return kept


def classifier_item(
    images: Mapping[str, np.ndarray], *, take: str, t_rel_s: float
) -> dict[str, Any]:
    """Build one upstream-format classifier item.

    ``observations`` and ``next_observations`` share the same image arrays, as
    the shipped export does; nothing downstream mutates them.
    """

    observation: dict[str, np.ndarray] = {
        "state": np.zeros(_STATE_PLACEHOLDER_SHAPE, dtype=np.float32)
    }
    for camera in CLASSIFIER_CAMERAS:
        if camera not in images:
            raise ClassifierDatasetError(f"item is missing {camera}")
        image = np.asarray(images[camera])
        if image.ndim != 3 or image.dtype != np.dtype(np.uint8):
            raise ClassifierDatasetError(
                f"{camera} must be (H, W, 3) uint8, got {image.shape}/{image.dtype}"
            )
        observation[camera] = image[None, ...]
    return {
        "observations": observation,
        "actions": np.zeros(_ACTION_DIM, dtype=np.float32),
        "next_observations": observation,
        "rewards": np.float32(0.0),
        "masks": np.float32(1.0),
        "dones": False,
        "metadata": {"take": str(take), "t_rel_s": float(t_rel_s)},
    }


def convert_take(
    take_dir: Path | str,
    *,
    crop: CropBox | None,
    size: tuple[int, int] = (128, 128),
    stride: int = 1,
) -> list[dict[str, Any]]:
    """Re-export one recorder take as classifier items under ``crop``.

    ``crop=None`` reproduces the shipped uncropped export bit-exactly; passing
    the task's window produces the same dataset as the policy sees it.

    Returns items only.  The caller decides which label buffer they belong to
    -- see the module docstring on why no label is inferred here.
    """

    directory = Path(take_dir)
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ClassifierDatasetError("stride must be an integer of at least 1")
    timeline = load_take_timeline(directory)
    positions = range(0, len(timeline), stride)
    frames = {
        camera: _preprocessed_frames(
            directory / f"{camera}.mp4",
            (indices[position] for position in positions),
            crop=crop,
            size=size,
        )
        for camera, indices in (
            ("cam1", timeline.cam1_index),
            ("cam2", timeline.cam2_index),
        )
    }
    items: list[dict[str, Any]] = []
    for position in positions:
        index1 = int(timeline.cam1_index[position])
        index2 = int(timeline.cam2_index[position])
        # The recorder can log a frame the video does not contain; the shipped
        # exporter drops those rather than failing the take.
        if index1 not in frames["cam1"] or index2 not in frames["cam2"]:
            continue
        items.append(
            classifier_item(
                {"cam1": frames["cam1"][index1], "cam2": frames["cam2"][index2]},
                take=directory.name,
                t_rel_s=float(timeline.cam1_time[position]),
            )
        )
    if not items:
        raise ClassifierDatasetError(f"{directory} produced no items")
    return items


def convert_dataset(
    take_dirs: Sequence[Path | str],
    *,
    crop: CropBox | None,
    size: tuple[int, int] = (128, 128),
    stride: int = 1,
) -> list[dict[str, Any]]:
    """Re-export several takes into one flat item list, in the given order."""

    if not take_dirs:
        raise ClassifierDatasetError("no takes to convert")
    items: list[dict[str, Any]] = []
    for take_dir in take_dirs:
        items.extend(convert_take(take_dir, crop=crop, size=size, stride=stride))
    return items


def iter_take_directories(root: Path | str) -> Iterator[Path]:
    """Yield ``take_*`` directories under ``root`` that hold a full recording."""

    directory = Path(root)
    if not directory.is_dir():
        raise ClassifierDatasetError(f"{directory} is not a directory")
    for candidate in sorted(directory.glob("take_*")):
        if not candidate.is_dir():
            continue
        if all(
            (candidate / name).is_file()
            for name in ("vectors.h5", "cam1.mp4", "cam2.mp4")
        ):
            yield candidate


__all__ = [
    "CLASSIFIER_CAMERAS",
    "ClassifierDatasetError",
    "TakeTimeline",
    "classifier_item",
    "convert_dataset",
    "convert_take",
    "iter_take_directories",
    "load_take_timeline",
]
