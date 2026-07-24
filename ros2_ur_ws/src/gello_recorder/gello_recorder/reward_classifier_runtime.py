"""ROS-independent input/status helpers for live reward classification."""

import json
import math

import cv2
import numpy as np


IMAGE_KEYS = ("cam1", "cam2")
IMAGE_SIZE = (128, 128)  # width, height for cv2.resize


def validate_threshold(value: float) -> float:
    """Return a finite binary-classification threshold in [0, 1]."""
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("classifier threshold must be finite and in [0, 1]")
    return threshold


def decode_classifier_image(jpeg: bytes) -> np.ndarray:
    """JPEG -> HIL-SERL RGB uint8 observation, shape (1,128,128,3)."""
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG decode failed")
    resized = cv2.resize(bgr, IMAGE_SIZE)
    rgb = np.ascontiguousarray(resized[..., ::-1], dtype=np.uint8)
    return rgb[None, ...]


def make_observation(cam1_jpeg: bytes, cam2_jpeg: bytes) -> dict:
    """Construct the exact observation tree used to initialize the checkpoint."""
    return {
        "state": np.zeros((1, 1), dtype=np.float32),
        "cam1": decode_classifier_image(cam1_jpeg),
        "cam2": decode_classifier_image(cam2_jpeg),
    }


def sigmoid_probability(logit: float) -> float:
    """Convert one binary-classifier logit without overflowing exp().

    The official HIL-SERL reward wrappers apply a sigmoid to the scalar logit
    returned by ``load_classifier_func``.  This branch-wise form is numerically
    stable for the very confident logits commonly seen on out-of-distribution
    live camera frames.
    """
    value = float(logit)
    if not math.isfinite(value):
        raise ValueError("classifier logit must be finite")
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def status_json(**values) -> str:
    """Stable compact encoding shared by the inference node and GUI."""
    return json.dumps(values, separators=(",", ":"), sort_keys=True)
