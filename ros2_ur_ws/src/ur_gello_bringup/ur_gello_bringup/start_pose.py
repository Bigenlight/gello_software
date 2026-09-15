"""Start-pose source for the EEF teleop GUI's "GO TO START POSE" button.

Pure module: stdlib + PyYAML only. No rclpy, no Qt. It reads ONE thing -- the
``policy_leader_node.ros__parameters.start_pose`` (+ optional ``start_gripper``)
entry of a policy deploy yaml (``src/gello_policy/config/*_deploy.yaml``) -- and
hands back validated numbers. Everything that moves the robot lives in the GUI
node; this module never touches ROS.

WHY THE POSE COMES FROM THE DEPLOY YAML (single source of truth with inference)
------------------------------------------------------------------------------
At inference time ``policy_leader_node`` HOLDS ``start_pose`` perfectly still so
``gello_move_to_start`` converges on it, and the resume-align gate
(``START_GATE_RAD = 0.1`` in ``policy_leader_node.py``) refuses
``~/start_execution`` unless the live arm is within ~0.1 rad of it on every
joint. That pose was measured FROM the dataset (e.g. ifql_deploy.yaml's value is
the mean de-staled frame-0 pose over 54 carrot takes). If data collection
started the arm anywhere else, every new demo would begin off-distribution
relative to the pose the policy will actually be started from -- and the
policy's first-frame observation would never match its training data. So the
teleop GUI must drive the arm to EXACTLY the pose inference uses, and the only
honest way to guarantee "exactly" is to read it from the same file the
inference launch reads. There is deliberately no second copy of the numbers.

WHY THERE IS NO DEFAULT POSE IN CODE (the banana vs carrot trap)
----------------------------------------------------------------
Different tasks have different start poses and the two we have on disk look
deceptively alike::

    banana (act/fm/diffusion_deploy.yaml): [ 3.106, -1.817, 1.653, -1.618, -1.628, -3.195]
    carrot (ifql_deploy.yaml):             [-3.164, -1.490, 1.726, -1.845, -1.579, -3.269]

shoulder_pan +3.106 vs -3.164 differ by ~2*pi in VALUE, i.e. they are almost the
same physical angle -- but the other joints differ by up to 0.33 rad (shoulder
lift -1.817 vs -1.490). A hard-coded default would therefore move the arm to a
plausible-looking, entirely wrong pose, with no error anywhere: the trajectory
executes fine, the arm parks somewhere sensible-looking, and every demo
recorded afterwards is silently ~0.3 rad off the pose the policy is started
from. That failure is invisible until training. Hence: the file is selected by
``START_POSE_CONFIG`` and when it is unset, missing or malformed the consumer
must FAIL CLOSED (button disabled + reason on screen). ``resolve_start_pose``
returns ``(None, reason)`` in every such case and never falls back.

BRANCH-CUT NOTE (for the consumer)
----------------------------------
This module returns the RAW numbers from the file. The UR controllers
interpolate linearly in joint space with no 2*pi awareness (see
``angle_utils``), and a wrist near +/-pi (w3 = -3.195 / -3.269 above, and pan
at +/-3.1) sits right at the branch cut. The consumer MUST apply
``angle_utils.wrapped_nearest(pose.joints, live_joints)`` against the arm's
CURRENT ``/joint_states`` before building the trajectory, or the arm may travel
~2*pi the long way. That wrap is the caller's job because it needs live data
this module does not have.

Validation contract (``load_start_pose``)
-----------------------------------------
Every rejection raises ``StartPoseError`` (a ``ValueError``) whose message names
the offending path so the GUI can show it verbatim:

  * missing file / not a regular file / unreadable / invalid YAML
  * top level not a mapping; key path ``policy_leader_node.ros__parameters
    .start_pose`` absent (each missing level is named)
  * ``start_pose`` not a list of exactly 6 real numbers (bools are rejected --
    YAML ``true`` parses to Python ``True``, which IS an ``int`` subclass and
    would otherwise sail through an ``isinstance(x, (int, float))`` check)
  * any non-finite value; any ``|value| > 2*pi + 0.5`` (implausible as radians;
    catches degrees pasted by mistake)
  * ``start_gripper`` (optional, default 0.0) not a real number in ``[0, 1]``
    (0.0 = OPEN .. 1.0 = CLOSED -- the topic/recording convention, NOT the RL
    +1/-1 action convention)
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import yaml

START_POSE_CONFIG_ENV = "START_POSE_CONFIG"
KEY_PATH: Tuple[str, ...] = ("policy_leader_node", "ros__parameters", "start_pose")
GRIPPER_KEY_PATH: Tuple[str, ...] = (
    "policy_leader_node",
    "ros__parameters",
    "start_gripper",
)

N_JOINTS = 6
# UR joint order, as the deploy yamls document it. Used only for describe()/logs.
JOINT_NAMES: Tuple[str, ...] = ("pan", "lift", "elbow", "w1", "w2", "w3")
# Anything beyond one full turn plus slack is not radians (most likely degrees,
# or a copy-paste of a different field). UR joints are limited to +/-2*pi anyway.
MAX_ABS_RAD = 2.0 * math.pi + 0.5

_UNSET_REASON = (
    f"{START_POSE_CONFIG_ENV} is not set -- export it to a deploy yaml "
    "(e.g. src/gello_policy/config/ifql_deploy.yaml)"
)


class StartPoseError(ValueError):
    """Raised by ``load_start_pose`` for every rejected file. Message names the path."""


@dataclass(frozen=True)
class StartPose:
    """A validated start pose read from a deploy yaml.

    ``joints``  6 floats, rad, UR joint order (pan, lift, elbow, w1, w2, w3).
                RAW values from the file -- apply ``wrapped_nearest`` against the
                live arm before commanding (module docstring, branch-cut note).
    ``gripper`` 0.0 = OPEN .. 1.0 = CLOSED (topic convention).
    ``source``  absolute path of the yaml the values came from.
    ``key``     dotted key path inside that file (provenance for logs/tooltips).
    """

    joints: Tuple[float, ...]
    gripper: float
    source: str
    key: str = ".".join(KEY_PATH)

    def describe(self) -> str:
        """One-line provenance string for tooltips / logs, e.g.
        ``start_pose from ifql_deploy.yaml: [-3.164, -1.490, 1.726, -1.845, -1.579, -3.269] gripper 0.00``.
        """
        joints = ", ".join(f"{v:.3f}" for v in self.joints)
        return (
            f"start_pose from {os.path.basename(self.source)}: [{joints}] "
            f"gripper {self.gripper:.2f}"
        )


def _is_real_number(value: Any) -> bool:
    """True for int/float but NOT bool (``True`` is an ``int`` subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _lookup(doc: Mapping[str, Any], key_path: Sequence[str], path: str) -> Any:
    """Walk ``key_path`` through nested mappings; name the first missing level."""
    node: Any = doc
    walked = []
    for key in key_path:
        if not isinstance(node, Mapping):
            raise StartPoseError(
                f"{path}: '{'.'.join(walked)}' is not a mapping (got "
                f"{type(node).__name__}); cannot look up '{key}'"
            )
        if key not in node:
            raise StartPoseError(
                f"{path}: key '{'.'.join(list(walked) + [key])}' is missing"
            )
        node = node[key]
        walked.append(key)
    return node


def load_start_pose(path: str) -> StartPose:
    """Read and validate ``policy_leader_node.ros__parameters.start_pose`` from ``path``.

    Raises ``StartPoseError`` (a ``ValueError``) with a path-naming message on
    every rejection; see the module docstring for the full contract. Never
    returns a partially-validated pose.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise StartPoseError(f"{path}: file does not exist")
    if not os.path.isfile(path):
        raise StartPoseError(f"{path}: not a regular file")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except OSError as exc:
        raise StartPoseError(f"{path}: cannot read file: {exc}") from exc
    except yaml.YAMLError as exc:
        # PyYAML's message is multi-line; keep the first line so the GUI reason
        # stays one line.
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
        raise StartPoseError(f"{path}: invalid YAML: {first}") from exc
    except UnicodeDecodeError as exc:
        raise StartPoseError(f"{path}: cannot decode file as UTF-8: {exc}") from exc

    if not isinstance(doc, Mapping):
        raise StartPoseError(
            f"{path}: top level is not a mapping (got {type(doc).__name__})"
        )

    key = ".".join(KEY_PATH)
    raw = _lookup(doc, KEY_PATH, path)
    if not isinstance(raw, (list, tuple)):
        raise StartPoseError(
            f"{path}: '{key}' must be a list of {N_JOINTS} numbers "
            f"(got {type(raw).__name__})"
        )
    if len(raw) != N_JOINTS:
        raise StartPoseError(
            f"{path}: '{key}' must have exactly {N_JOINTS} entries (got {len(raw)})"
        )
    joints = []
    for i, value in enumerate(raw):
        name = JOINT_NAMES[i]
        if not _is_real_number(value):
            raise StartPoseError(
                f"{path}: '{key}[{i}]' ({name}) is not a number "
                f"(got {type(value).__name__} {value!r})"
            )
        fvalue = float(value)
        if not math.isfinite(fvalue):
            raise StartPoseError(
                f"{path}: '{key}[{i}]' ({name}) is not finite (got {value!r})"
            )
        if abs(fvalue) > MAX_ABS_RAD:
            raise StartPoseError(
                f"{path}: '{key}[{i}]' ({name}) = {fvalue} is not plausible as "
                f"radians (|value| > {MAX_ABS_RAD:.3f}); degrees pasted by mistake?"
            )
        joints.append(fvalue)

    # start_gripper is optional: the leader defaults it to 0.0 (open) too, and
    # every corpus starts open. If PRESENT it must be a real number in [0, 1].
    gripper = 0.0
    params = _lookup(doc, KEY_PATH[:-1], path)
    if GRIPPER_KEY_PATH[-1] in params:
        gkey = ".".join(GRIPPER_KEY_PATH)
        gvalue = params[GRIPPER_KEY_PATH[-1]]
        if not _is_real_number(gvalue):
            raise StartPoseError(
                f"{path}: '{gkey}' is not a number (got {type(gvalue).__name__} "
                f"{gvalue!r})"
            )
        gripper = float(gvalue)
        if not math.isfinite(gripper) or not (0.0 <= gripper <= 1.0):
            raise StartPoseError(
                f"{path}: '{gkey}' = {gvalue!r} is outside [0, 1] "
                "(0.0 = open .. 1.0 = closed)"
            )

    return StartPose(
        joints=tuple(joints),
        gripper=gripper,
        source=os.path.abspath(path),
        key=key,
    )


def resolve_start_pose(
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[StartPose], str]:
    """Resolve the start pose from ``$START_POSE_CONFIG``. NEVER raises.

    Returns ``(pose, pose.describe())`` on success, else ``(None, reason)`` where
    ``reason`` is a one-line, operator-readable string: the env var is
    unset/empty, or the ``StartPoseError`` message (which names the file).
    The consumer shows ``reason`` on screen and keeps the button disabled --
    there is no fallback pose (module docstring).

    ``environ`` defaults to ``os.environ``; pass a dict in tests.
    """
    env = os.environ if environ is None else environ
    path = env.get(START_POSE_CONFIG_ENV, "")
    if path is None or not str(path).strip():
        return None, _UNSET_REASON
    try:
        pose = load_start_pose(os.path.expanduser(str(path).strip()))
    except StartPoseError as exc:
        return None, str(exc)
    except Exception as exc:  # defensive: the GUI must never crash on this path
        return None, f"{path}: unexpected error while loading start pose: {exc!r}"
    return pose, pose.describe()
