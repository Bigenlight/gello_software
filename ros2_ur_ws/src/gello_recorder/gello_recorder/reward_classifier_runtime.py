"""ROS-independent input/status helpers for live reward classification."""

import json
import math
import os

import cv2
import numpy as np


IMAGE_KEYS = ("cam1", "cam2")
IMAGE_SIZE = (128, 128)  # width, height for cv2.resize

# Deployment default, kept in sync with
# serl_ur_infra/ur_env/rlpd_receive_server.py::DEFAULT_REWARD_THRESHOLD so the
# live viewer shows the same SUCCESS/FAILURE verdict the learner would record.
# Still overridable via the ``threshold`` ROS parameter or CLASSIFIER_THRESHOLD.
DEFAULT_THRESHOLD = 0.5

# Canonical cube-in-cup classifier checkpoint.
#
# This is the orbax *directory* holding ``checkpoint_150``; flax's
# ``restore_checkpoint`` selects the newest ``checkpoint_*`` inside it.  The old
# ``classifier_ckpt/cube_in_cup`` checkpoint is retired (0% held-out recall on
# the current domain) and is deliberately not used as a fallback.
DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.environ.get("GELLO_REPO_ROOT", "/home/laptop3/gello_software"),
    "classifier_ckpt",
    "cube_in_cup_all3",
)

_MISSING_CHECKPOINT_HELP = (
    "classifier checkpoint not found: {path}\n"
    "The canonical cube-in-cup checkpoint is the orbax directory "
    "'cube_in_cup_all3' (it contains checkpoint_150).  Stage it with:\n"
    "  mkdir -p {default}\n"
    "  rsync -a kanu:~/workspace/youngwoong/dataset/cube_in_cup_all3/"
    "classifier_ckpt/ {default}/\n"
    "or point REWARD_CLASSIFIER_CHECKPOINT (or the checkpoint_path ROS "
    "parameter / --checkpoint flag) at an existing checkpoint directory."
)


def resolve_checkpoint_path(explicit: str = "") -> str:
    """Resolve param -> env -> canonical default, and fail loudly if absent.

    Returning a path that does not exist is worse than not starting: flax's
    ``restore_checkpoint`` silently returns the *randomly initialized* target
    when the path is missing, which would render a confident-looking but
    meaningless p(success) in the viewer.
    """
    path = (explicit or "").strip()
    path = path or os.environ.get("REWARD_CLASSIFIER_CHECKPOINT", "").strip()
    path = path or DEFAULT_CHECKPOINT_PATH
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise RuntimeError(
            _MISSING_CHECKPOINT_HELP.format(
                path=path, default=DEFAULT_CHECKPOINT_PATH
            )
        )
    return path


def validate_threshold(value: float) -> float:
    """Return a finite binary-classification threshold in [0, 1]."""
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("classifier threshold must be finite and in [0, 1]")
    return threshold


def default_threshold() -> float:
    """CLASSIFIER_THRESHOLD environment override, else DEFAULT_THRESHOLD."""
    raw = os.environ.get("CLASSIFIER_THRESHOLD", "").strip()
    if not raw:
        return DEFAULT_THRESHOLD
    return validate_threshold(float(raw))


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
