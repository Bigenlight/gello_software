"""The one pixel recipe every observation consumer is supposed to share.

``observation_schema`` pins the *shape and dtype* of a canonical observation.
This module pins the step before it: how a full-resolution BGR camera frame
becomes those pixels.  Three consumers currently reimplement that step --
``UR7eEnv.get_im`` (policy/critic), ``learner.recorded_demo`` (offline demos),
and ``classifier_sidecar`` (reward classifier) -- and G15 is what happens when
two of them drift apart: the classifier was fed a crop it was never trained on
and held-out recall@0.85 fell from 100% to 33.3%.

The recipe itself is small and identical in all three::

    optional square crop -> cv2.resize(..., size) -> BGR to RGB -> contiguous uint8

What differs today is only whether the crop is applied.  So the rule worth
pinning is not the code, it is the **crop**, and :class:`PreprocessRule` makes
it data rather than a lambda: serializable, comparable, and hashable into a
lineage tag.  ``cube_in_cup.IMAGE_CROP`` stores ``lambda img: img[20:670,
340:990]``, which cannot be any of those things.

NOT WIRED UP YET.  Nothing calls this module in production; the three
consumers still hold their own copies.  ``tests/test_observation_preprocess``
is what makes adopting it safe: it asserts this function is bit-exact against
each existing recipe, so a later rewiring is provably a no-op at whichever
rule the caller passes.

Why the square crop exists at all -- and what dropping it costs -- is measured
in ``ur_experiments/cube_in_cup.py``: the frame is 1280x720, so ``crop=None``
squashes the horizontal axis to 0.5625 of the vertical.  A square crop is the
only way to reach a square output without that distortion.

``cv2`` is imported inside the calls so a host without OpenCV can still import
the rule objects (the learner server does exactly this for
``classifier_sidecar``).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np


#: ``(width, height)`` as ``cv2.resize`` wants it, matching
#: ``classifier_sidecar.CLASSIFIER_IMAGE_SIZE`` and the 128x128 observation
#: contract in ``observation_schema.CANONICAL_OBSERVATION_SPEC``.
CANONICAL_IMAGE_SIZE = (128, 128)

_CHANNELS = 3


class PreprocessRuleError(ValueError):
    """A preprocessing rule is not expressible or not self-consistent."""


@dataclass(frozen=True)
class CropBox:
    """One half-open ``img[y0:y1, x0:x1]`` window, as data.

    Square by construction.  A non-square crop resized to a square output
    reintroduces exactly the aspect distortion the crop exists to avoid, so it
    is rejected here rather than left as a comment for the next reader.
    """

    y0: int
    y1: int
    x0: int
    x1: int

    def __post_init__(self) -> None:
        for name in ("y0", "y1", "x0", "x1"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise PreprocessRuleError(f"crop {name} must be an integer")
            if value < 0:
                raise PreprocessRuleError(f"crop {name} must not be negative")
        if self.y1 <= self.y0 or self.x1 <= self.x0:
            raise PreprocessRuleError("crop must have positive height and width")
        if self.height != self.width:
            raise PreprocessRuleError(
                f"crop must be square, got {self.height}x{self.width}; a "
                "non-square crop resized to a square output distorts aspect"
            )

    @property
    def height(self) -> int:
        return int(self.y1) - int(self.y0)

    @property
    def width(self) -> int:
        return int(self.x1) - int(self.x0)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return the cropped **view**; the caller owns any copy it needs."""

        if frame.shape[0] < self.y1 or frame.shape[1] < self.x1:
            raise PreprocessRuleError(
                f"crop {self.as_tuple()} does not fit a "
                f"{frame.shape[0]}x{frame.shape[1]} frame"
            )
        return frame[self.y0 : self.y1, self.x0 : self.x1]

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.y0), int(self.y1), int(self.x0), int(self.x1))


@dataclass(frozen=True)
class PreprocessRule:
    """The per-camera crop plus the output size: the whole rule, as data.

    ``crops[camera] is None`` means "no crop, squash the full frame", which is
    what the pinned reward classifier was trained on.
    """

    crops: Mapping[str, CropBox | None]
    size: tuple[int, int] = CANONICAL_IMAGE_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.crops, Mapping) or not self.crops:
            raise PreprocessRuleError("crops must be a non-empty mapping")
        for camera, crop in self.crops.items():
            if not isinstance(camera, str) or not camera:
                raise PreprocessRuleError("camera keys must be non-empty strings")
            if crop is not None and not isinstance(crop, CropBox):
                raise PreprocessRuleError(
                    f"crop for {camera!r} must be a CropBox or None"
                )
        width, height = _validated_size(self.size)
        object.__setattr__(self, "size", (width, height))

    def crop_for(self, camera: str) -> CropBox | None:
        if camera not in self.crops:
            raise PreprocessRuleError(f"rule has no entry for camera {camera!r}")
        return self.crops[camera]

    def describe(self) -> dict[str, Any]:
        """Return the JSON-serializable rule, ordered for stable hashing."""

        return {
            "size": list(self.size),
            "crops": {
                camera: (None if crop is None else list(crop.as_tuple()))
                for camera in sorted(self.crops)
                for crop in (self.crops[camera],)
            },
        }

    def tag(self) -> str:
        """Short stable id for lineage strings and fingerprints.

        Any change to a crop or the output size changes this, which is the
        point: a stored observation and the rule that produced it must not be
        able to drift apart silently (see G18 for what that costs).
        """

        payload = json.dumps(self.describe(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _validated_size(size: Any) -> tuple[int, int]:
    try:
        width, height = size
    except (TypeError, ValueError) as exc:
        raise PreprocessRuleError("size must be a (width, height) pair") from exc
    values = []
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise PreprocessRuleError(f"size {name} must be an integer")
        if value <= 0:
            raise PreprocessRuleError(f"size {name} must be positive")
        values.append(int(value))
    return values[0], values[1]


def _validated_frame(frame: Any, *, name: str) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.dtype(np.uint8):
        raise PreprocessRuleError(f"{name} must be uint8, got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != _CHANNELS:
        raise PreprocessRuleError(
            f"{name} must be (H, W, 3) BGR, got shape {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise PreprocessRuleError(f"{name} is empty")
    return array


def preprocess_frame(
    bgr: Any,
    *,
    crop: CropBox | None = None,
    size: tuple[int, int] = CANONICAL_IMAGE_SIZE,
) -> np.ndarray:
    """One full-resolution BGR frame -> canonical ``(H, W, 3)`` uint8 RGB.

    This is the primitive; everything else in this module and (once wired)
    every consumer routes through it, so bit-identity across hosts is
    structural rather than asserted.  ``cv2.resize`` is called with OpenCV's
    default INTER_LINEAR and no ``interpolation=`` argument, matching all three
    existing recipes -- a different kernel silently shifts the whole input
    distribution of both the policy and the classifier.

    The result is contiguous.  ``get_im`` currently returns the reversed view
    ``resized[..., ::-1]`` instead, which carries identical values but a
    negative stride; ``validate_canonical_observation`` normalizes it later.
    Returning contiguous here keeps that normalization from being load-bearing.
    """

    import cv2

    frame = _validated_frame(bgr, name="frame")
    width, height = _validated_size(size)
    cropped = frame if crop is None else crop.apply(frame)
    resized = cv2.resize(cropped, (width, height))
    return np.ascontiguousarray(resized[..., ::-1], dtype=np.uint8)


def preprocess_dataset(
    frames: Iterable[Any],
    *,
    crop: CropBox | None = None,
    size: tuple[int, int] = CANONICAL_IMAGE_SIZE,
) -> np.ndarray:
    """A sequence of BGR frames -> stacked ``(N, H, W, 3)`` uint8 RGB.

    Deliberately I/O-free: it takes decoded frames and returns an array.
    Reading videos and writing a demo artifact stay in
    ``learner.recorded_demo``, so this function is equally usable for a
    recorded take, a live buffer, or a handful of frames in a test.

    Every frame goes through :func:`preprocess_frame`, so a dataset conversion
    and the live path cannot diverge by construction.
    """

    stacked = [
        preprocess_frame(frame, crop=crop, size=size)
        for frame in _iterate_frames(frames)
    ]
    if not stacked:
        raise PreprocessRuleError("dataset is empty")
    return np.stack(stacked, axis=0)


def preprocess_cameras(
    bgr_by_camera: Mapping[str, Any],
    rule: PreprocessRule,
) -> dict[str, np.ndarray]:
    """``{"cam1": full-res BGR, ...}`` -> ``{"cam1": (H, W, 3) uint8 RGB, ...}``.

    Applies one rule across the cameras of a single observation.  The rule must
    name every camera supplied: silently passing an unlisted camera through
    uncropped is precisely the G15 failure mode.
    """

    if not isinstance(bgr_by_camera, Mapping):
        raise PreprocessRuleError("bgr_by_camera must be a mapping")
    missing = sorted(set(bgr_by_camera) - set(rule.crops))
    if missing:
        raise PreprocessRuleError(f"rule has no entry for cameras {missing}")
    return {
        camera: preprocess_frame(
            frame, crop=rule.crop_for(camera), size=rule.size
        )
        for camera, frame in bgr_by_camera.items()
    }


def _iterate_frames(frames: Any) -> Iterable[Any]:
    if isinstance(frames, np.ndarray):
        if frames.ndim != 4:
            raise PreprocessRuleError(
                f"a frame array must be (N, H, W, 3), got shape {frames.shape}"
            )
        return list(frames)
    if isinstance(frames, Mapping) or isinstance(frames, (str, bytes)):
        raise PreprocessRuleError("frames must be a sequence of BGR arrays")
    return frames


__all__ = [
    "CANONICAL_IMAGE_SIZE",
    "CropBox",
    "PreprocessRule",
    "PreprocessRuleError",
    "preprocess_cameras",
    "preprocess_dataset",
    "preprocess_frame",
]
