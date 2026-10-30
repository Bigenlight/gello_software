"""Shared contract for the reward-classifier image sidecar.

THE PROBLEM
-----------
The cube-in-cup reward classifier was trained on **uncropped** 1280x720 camera
frames squashed straight to 128x128.  The RL actor, however, hands the server
the *policy's* observation, and that observation is cropped by
``ur_experiments/cube_in_cup.py::IMAGE_CROP`` (650x650 / 720x720 windows) before
the 128x128 resize.  Classifier and policy therefore see two different image
distributions, and the classifier is the one that is out of distribution:
measured held-out recall at p=0.85 falls from 100% to 33.3%.  See
``docs/testing/08_OPEN_GAPS.md`` G15.

Deleting ``IMAGE_CROP`` is *not* the fix.  The crop was measured from the
dataset and the policy is its first consumer; removing it would trade a broken
reward for a broken policy.  Instead we ship the classifier its **own**
uncropped view alongside the cropped policy observation — the "sidecar".

WHY 128x128 IS ENCODED ON THE LAPTOP (AND NOT PASSED THROUGH)
-------------------------------------------------------------
The first design forwarded the camera's original JPEG untouched — zero CPU, zero
new artifacts.  **Measurement killed it.**  The camera nodes run at
``jpeg_quality = 95`` (the untuned ROS ``image_transport`` default), so a live
720p frame is **206 KiB (cam1) / 194 KiB (cam2)**, not the ~68 KiB the docs
claimed (that figure came from a q75 re-encode).  The pair is 400 KiB, which even
at 2 Hz adds 6.55 Mbit/s for a combined 14.46 Mbit/s — **over the 13 Mbit/s WiFi
link outright** — and each attachment spikes the step by **+252 ms at 13 Mbit/s**
(+72 ms at 45.6 Mbit/s) against a **100 ms** budget.  One sidecar would blow the
control loop.

So the laptop now performs the 128x128 resize itself — *the same deterministic
``cv2.resize`` call the server would have made* — and JPEG-encodes that.  The
resize is bit-identical whichever host runs it (``tests/test_classifier_sidecar.py``
asserts exactly that against the viewer), so the only thing this trades away is
one extra JPEG generation at 128x128, measured on real rig frames at
MAE 1.29-1.77 / 8-bit levels versus the viewer's own output.  The payload drops
from ~400 KiB to **~10-15 KiB per pair** — a 27x reduction that puts the
attachment back inside the step budget.

Everything downstream is unchanged: the wire keys are still ``cam1_jpeg`` /
``cam2_jpeg``, and :func:`decode_classifier_frames` still runs the viewer's
imdecode -> resize(128,128) -> RGB -> ``[None]`` recipe — the resize simply
becomes a no-op on an already-128x128 payload.  The server therefore needs no
knowledge of which host did the resizing.

Note what did *not* change: the sidecar is still the **uncropped** field of view.
That is the entire point (see THE PROBLEM above); only the transport resolution
moved.

WHY IT DOES NOT RIDE ON EVERY STEP
----------------------------------
Two independent reasons:

* **Budget.** The 10 Hz loop has a 100 ms step budget and gRPC RTT p99 is
  already 97.1 ms.  Even at ~7 KiB per frame this is wire time we do not have to
  spend 10x per second, and the measurement above shows how little slack the
  link has.
* **Correctness.** While the arm sweeps across cam1 the classifier's output
  oscillates violently (observed p swinging 0.005 -> 1.0 within a single sweep)
  because a moving gripper occludes the cup.  Sampling only when the TCP is
  effectively stationary is the primary mitigation for that pathology, not just
  a bandwidth trick.  :class:`SidecarScheduler` owns both policies.

WIRE / SCHEMA IMPACT: NONE
--------------------------
``proto/actor_transport.proto`` carries a generic named-tensor map
(``Tensor{path,dtype,shape,data}`` + ``Observation{repeated Tensor tensors}``),
so a new nested observation key needs no proto change.  The peer handshake hash
is derived from ``CANONICAL_OBSERVATION_SPEC`` in ``ur_env/observation_schema.py``
— from the *document*, never from the wire payload — so adding
:data:`CLASSIFIER_SIDECAR_KEY` does not invalidate the laptop<->Kanu handshake
and old/new peers stay compatible (an old server simply ignores the extra
tensors).

This module is deliberately dependency-light: numpy only at import time.  Both
peers import it, including processes that have no OpenCV and no ROS; ``cv2`` is
imported *inside* :func:`build_sidecar` and :func:`decode_classifier_frames`, so
a process that only validates or schedules never needs OpenCV at all.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Mapping, Optional

import numpy as np

from ur_env.observation_preprocess import preprocess_frame
from ur_env.observation_schema import (
    STATE_DIM,
    STATE_FEATURE_INDEX,
    state_slice,
)


#: Reserved top-level observation key.  The sidecar is a *nested* map so it can
#: never collide with a policy tensor name and so a consumer can drop the whole
#: thing with one ``pop``.
CLASSIFIER_SIDECAR_KEY = "classifier"

#: Tensor names inside the sidecar.  Each is a 1-D uint8 array holding one
#: complete JPEG file — since 2026-07-29 that is a 128x128 encode produced on the
#: laptop, not the camera's original full-resolution frame (see the docstring).
#: The names deliberately did NOT change with that switch: the wire contract is
#: "a JPEG of the uncropped view", and the resolution is an implementation
#: detail the server does not need to know.
SIDECAR_TENSOR_KEYS = ("cam1_jpeg", "cam2_jpeg")

#: Camera names accepted by :func:`build_sidecar` / emitted by
#: :func:`decode_classifier_frames`.  Derived so the two can never drift.
SIDECAR_CAMERA_KEYS = tuple(key[: -len("_jpeg")] for key in SIDECAR_TENSOR_KEYS)

#: Identifies *what the classifier was fed*, independent of which checkpoint
#: scored it.  Stamped into learner metadata so a replay buffer recorded before
#: this change can never be confused with one recorded after it.
#:
#: HISTORICAL NAME: "passthrough" describes the original design, which forwarded
#: the camera's own bytes.  What actually ships resizes to 128x128 on the laptop
#: and re-encodes (the 400 KiB/+252 ms measurement in the docstring killed
#: passthrough).  The *semantics* the id exists to record are unchanged and
#: still accurate — full frame, no crop — so the string is kept: it is pinned in
#: ``docs/testing/08_OPEN_GAPS.md``, ``HANDOFF_NEXT_SESSION_KO.md`` and
#: ``tests/test_rlpd_learner_server_cli.py``, and churning an id that no
#: recorded data has used yet would cost more than it explains.
CLASSIFIER_INPUT_ID = "fullframe-jpeg-passthrough-v1"

#: Per-frame size cap, sized for the 128x128 payload the laptop now produces.
#:
#: Measured on real rig frames (the 2026-07-29 ``fail_take_*`` recordings, 24
#: samples across both cameras) a q95 128x128 encode is **4.87-7.34 KiB**, so 64
#: KiB is ~8.7x headroom over the worst observed frame.  A pathological
#: incompressible frame (uniform random noise) still only reaches 19.4 KiB at
#: q95, so no realistic camera input can trip this.
#:
#: CAVEAT for whoever tunes ``build_sidecar(quality=...)``: the headroom is
#: against *q95 4:2:0*.  Pure noise at q100 with 4:4:4 chroma encodes to 66.5
#: KiB and WOULD trip the cap.  Raise this constant if the encode settings ever
#: move that far.
#:
#: The cap exists to fail loudly on a corrupt/garbage buffer rather than letting
#: it silently blow gRPC's 4 MiB message limit mid-episode.
MAX_SIDECAR_JPEG_BYTES = 64 * 1024

#: Classifier input resolution, (width, height) for ``cv2.resize`` — the exact
#: argument order and value used to train the checkpoint.
CLASSIFIER_IMAGE_SIZE = (128, 128)

#: JPEG quality for the laptop-side encode.  95 matches what the camera nodes
#: already use, and at 128x128 the size difference between q95 and q90 is ~2 KiB
#: — irrelevant next to the 400 KiB we just stopped sending — so fidelity wins
#: by default.  A concurrent measurement is pinning the best value; it is a
#: keyword argument precisely so changing it is a one-line edit.
DEFAULT_SIDECAR_JPEG_QUALITY = 95


# --------------------------------------------------------------------------- #
# Structure validation (numpy only — no cv2, no decode)                        #
# --------------------------------------------------------------------------- #
def validate_sidecar(mapping: Any) -> dict[str, np.ndarray]:
    """Validate the *structure* of a sidecar payload and return it.

    Checks shape/dtype/size only; it never decodes, so both peers can run it on
    the hot path (the actor before sending, the server before trusting bytes off
    the wire).  Nothing is coerced: a wrong dtype means the producer is broken,
    and quietly casting it here would hide that until the classifier returned
    garbage probabilities.

    Raises:
        ValueError: with a message that names the offending key and the fix.
    """
    if not isinstance(mapping, Mapping):
        raise ValueError(
            "classifier sidecar must be a mapping of "
            f"{list(SIDECAR_TENSOR_KEYS)} -> 1-D uint8 JPEG arrays, got "
            f"{type(mapping).__name__}"
        )

    actual = set(mapping)
    expected = set(SIDECAR_TENSOR_KEYS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "classifier sidecar keys mismatch: "
            f"missing={missing}, extra={extra}; expected exactly "
            f"{list(SIDECAR_TENSOR_KEYS)}"
        )

    result: dict[str, np.ndarray] = {}
    for key in SIDECAR_TENSOR_KEYS:
        value = mapping[key]
        # Raw ``bytes`` becomes a 0-d object array here rather than a uint8
        # vector, so this also catches "forgot to call build_sidecar()".
        array = np.asarray(value)
        if array.dtype != np.uint8:
            raise ValueError(
                f"classifier sidecar {key!r} must be uint8 JPEG bytes, got "
                f"dtype {array.dtype}; build it with build_sidecar()"
            )
        if array.ndim != 1:
            raise ValueError(
                f"classifier sidecar {key!r} must be a 1-D byte string, got "
                f"shape {array.shape}; the sidecar carries the *encoded* JPEG, "
                "not a decoded image"
            )
        if array.size == 0:
            raise ValueError(
                f"classifier sidecar {key!r} is empty; drop the whole "
                f"{CLASSIFIER_SIDECAR_KEY!r} key instead of sending a "
                "zero-length frame"
            )
        if array.size > MAX_SIDECAR_JPEG_BYTES:
            raise ValueError(
                f"classifier sidecar {key!r} is {array.size} bytes, over the "
                f"{MAX_SIDECAR_JPEG_BYTES}-byte per-frame cap; a 128x128 q95 "
                "encode measures 5-8 KiB, so this is either a corrupt buffer "
                "or a full-resolution frame that skipped build_sidecar()"
            )
        # C-contiguous so the transport's ``tobytes(order="C")`` is a straight
        # memcpy; already-contiguous arrays pass through untouched.
        result[key] = np.ascontiguousarray(array)
    return result


def _resize_for_classifier(bgr: np.ndarray) -> np.ndarray:
    """THE classifier resize — one definition, used by producer and consumer.

    Both :func:`build_sidecar` (laptop) and :func:`decode_classifier_frames`
    (server) call this, which is what makes "resize on whichever host" a safe
    claim: there is only one call, with OpenCV's default INTER_LINEAR, matching
    ``decode_classifier_image``.  Do not add an ``interpolation=`` argument here
    without re-running the parity test — a different kernel silently shifts the
    classifier's whole input distribution.
    """
    import cv2

    return cv2.resize(bgr, CLASSIFIER_IMAGE_SIZE)


def build_sidecar(
    bgr_by_camera: Mapping[str, np.ndarray],
    *,
    quality: int = DEFAULT_SIDECAR_JPEG_QUALITY,
) -> dict[str, np.ndarray]:
    """``{"cam1": full-res BGR, ...}`` -> ``{"cam1_jpeg": uint8[N], ...}``.

    Takes the **decoded, full-resolution, uncropped** BGR frame that
    ``UR7eEnv.get_im()`` already holds (``bgr = cv2.imdecode(...)``, before it
    applies ``IMAGE_CROP``), resizes it to 128x128 with :func:`_resize_for_classifier`
    and JPEG-encodes the result.  No crop is applied here and none may be: the
    sidecar exists precisely to give the classifier the uncropped view.

    Why the caller hands over pixels rather than the camera's JPEG bytes: see the
    module docstring.  Short version — the original 720p q95 frames are 206/194
    KiB, the pair does not fit the WiFi link at 2 Hz, and one attachment would
    add +252 ms to a 100 ms step.  The env has already paid for the decode, so
    resizing the array it is holding costs one ``cv2.resize`` on a 10 Hz loop.

    Args:
        bgr_by_camera: ``{"cam1": ndarray(H,W,3) uint8 BGR, "cam2": ...}``.
        quality: JPEG quality 1-100 for the laptop-side encode.

    ``cv2`` is imported inside the call chain so this module still imports on a
    host without OpenCV.
    """
    import cv2

    if not isinstance(bgr_by_camera, Mapping):
        raise ValueError(
            "build_sidecar expects a mapping of camera name -> full-resolution "
            f"BGR frame, got {type(bgr_by_camera).__name__}"
        )
    actual = set(bgr_by_camera)
    expected = set(SIDECAR_CAMERA_KEYS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "build_sidecar camera keys mismatch: "
            f"missing={missing}, extra={extra}; expected exactly "
            f"{list(SIDECAR_CAMERA_KEYS)}"
        )
    jpeg_quality = int(quality)
    if not 1 <= jpeg_quality <= 100:
        # cv2 clamps out-of-range quality silently, which would hide a typo
        # (e.g. 950) behind a subtly different image distribution.
        raise ValueError(f"jpeg quality must be in [1, 100], got {quality!r}")

    built: dict[str, np.ndarray] = {}
    for camera in SIDECAR_CAMERA_KEYS:
        frame = bgr_by_camera[camera]
        if isinstance(frame, (bytes, bytearray, memoryview)):
            # This signature changed on 2026-07-29 (it used to take the camera's
            # JPEG bytes), so name the migration instead of failing obscurely
            # inside cv2.
            raise ValueError(
                f"build_sidecar camera {camera!r} now takes the DECODED "
                "full-resolution BGR frame, not JPEG bytes; pass the 'bgr' "
                "that UR7eEnv.get_im() already holds"
            )
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            raise ValueError(
                f"build_sidecar camera {camera!r} must be a uint8 BGR frame, "
                f"got dtype {array.dtype}"
            )
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"build_sidecar camera {camera!r} must have shape (H, W, 3), "
                f"got {array.shape}"
            )
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError(
                f"build_sidecar camera {camera!r} has an empty frame "
                f"{array.shape}"
            )

        resized = _resize_for_classifier(np.ascontiguousarray(array))
        ok, encoded = cv2.imencode(
            ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            raise ValueError(
                f"build_sidecar camera {camera!r} JPEG encode failed"
            )
        # imencode returns (N,1); the wire contract is a flat byte string.
        built[f"{camera}_jpeg"] = encoded.reshape(-1)
    # Validate at the producing site so a bad frame is caught on the laptop,
    # with a stack that still points at the camera, instead of on Kanu.
    return validate_sidecar(built)


# --------------------------------------------------------------------------- #
# Decode (server side only)                                                    #
# --------------------------------------------------------------------------- #
def decode_classifier_frames(sidecar: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """``{"camN_jpeg": uint8[N]}`` -> ``{"camN": (1,128,128,3) uint8 RGB}``.

    This MUST stay equivalent to
    ``ros2_ur_ws/src/gello_recorder/gello_recorder/reward_classifier_runtime.py``
    ``::decode_classifier_image`` — imdecode(IMREAD_COLOR) -> **no crop** ->
    ``cv2.resize(bgr, (128,128))`` -> reverse the channel axis to RGB -> add the
    leading batch axis.  That function is the path validated on real hardware.

    The resize is kept even though :func:`build_sidecar` already delivers
    128x128 (where it degenerates to a copy).  Two reasons: a full-resolution
    payload — a legacy sender, or a future one that can afford the bandwidth —
    still decodes correctly, and keeping the call means this really is the
    viewer's recipe rather than a lookalike.

    The recipe is duplicated rather than imported on purpose: the reference
    lives in a ROS ament package under ``ros2_ur_ws/``, which is on the laptop's
    overlay ``PYTHONPATH`` but not on the Kanu learner's, and this module has to
    work on both hosts.  ``tests/test_classifier_sidecar.py`` is what keeps the
    copy honest: bit-exact against the viewer for the resize-only path, and a
    measured error bound for the full encode round trip.

    ``cv2`` is imported here, not at module scope, so peers that only need
    :func:`validate_sidecar` or :class:`SidecarScheduler` never pay for (or
    require) OpenCV.
    """
    import cv2

    validated = validate_sidecar(sidecar)
    frames: dict[str, np.ndarray] = {}
    for camera in SIDECAR_CAMERA_KEYS:
        buffer = validated[f"{camera}_jpeg"]
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(
                f"classifier sidecar {camera!r} JPEG decode failed "
                f"({buffer.size} bytes); the frame is corrupt"
            )
        # The shared recipe at this consumer's rule (crop=None): the resize is
        # the same cv2 call _resize_for_classifier makes, followed by the same
        # RGB flip.  Routing through it is what stops this copy from drifting
        # away from the policy's, which is how G15 happened.
        rgb = preprocess_frame(bgr, crop=None, size=CLASSIFIER_IMAGE_SIZE)
        frames[camera] = rgb[None, ...]
    return frames


# --------------------------------------------------------------------------- #
# Checkpoint hashing                                                           #
# --------------------------------------------------------------------------- #
def directory_sha256(path: str) -> str:
    """Deterministic sha256 over a file *or* a whole directory tree.

    ``rlpd_receive_server.checkpoint_sha256()`` used ``os.path.isfile()`` and so
    could not fingerprint an orbax checkpoint at all — orbax writes a
    *directory* (``cube_in_cup_all3/checkpoint_150/...``).  This is the
    replacement; a single regular file still hashes to exactly the same digest
    as before, so pinned single-file SHA256s stay valid.

    Directory hashing feeds the outer digest, for every file in sorted POSIX
    relpath order::

        relpath.encode() + b"\\0" + str(size).encode() + b"\\0" + contents

    Both the name and the length are mixed in so the digest is sensitive to a
    rename or a re-partitioning of identical bytes across files, which plain
    concatenation would miss.  Sorting is on the POSIX relpath (``/``
    separators), never on ``os.walk``'s arbitrary order, so laptop and Kanu
    agree.  Content is streamed in 1 MiB chunks: orbax shards can be large and
    this runs at server start-up.
    """
    target = os.path.abspath(os.path.expanduser(path))
    digest = hashlib.sha256()

    if os.path.isfile(target):
        # Backward compatible with the single-file checkpoint_sha256(): plain
        # sha256 of the contents, no framing.
        _feed_file(digest, target)
        return digest.hexdigest()

    if not os.path.isdir(target):
        raise FileNotFoundError(f"checkpoint path not found: {target}")

    relpaths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(target):
        # Sorting here is belt-and-braces; the authoritative order is the sort
        # of ``relpaths`` below, which is what makes this platform independent.
        dirnames.sort()
        for filename in sorted(filenames):
            absolute = os.path.join(dirpath, filename)
            # Skip anything that is not a regular file (sockets, fifos, broken
            # symlinks).  An orbax checkpoint contains none of these, and
            # hashing one would either hang or raise.
            if not os.path.isfile(absolute):
                continue
            relative = os.path.relpath(absolute, target)
            relpaths.append(relative.replace(os.sep, "/"))

    for relative in sorted(relpaths):
        absolute = os.path.join(target, *relative.split("/"))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(os.path.getsize(absolute)).encode("ascii"))
        digest.update(b"\0")
        _feed_file(digest, absolute)
    return digest.hexdigest()


def _feed_file(digest: "hashlib._Hash", path: str) -> None:
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)


# --------------------------------------------------------------------------- #
# Cadence policy                                                               #
# --------------------------------------------------------------------------- #
# Derive the linear-velocity indices from the canonical schema instead of
# writing 13:16.  ``state_slice("tcp_vel")`` returns the whole 6-D twist
# (linear + angular); only the linear half decides "is the arm parked".  If a
# future gymnasium changes the flatten order, these move with it and the
# assertion below turns a silent mis-read into an import-time failure.
_TCP_VEL_SLICE = state_slice("tcp_vel")
_LINEAR_VELOCITY_FEATURES = (
    "tcp_linear_velocity_x",
    "tcp_linear_velocity_y",
    "tcp_linear_velocity_z",
)
_LINEAR_VELOCITY_INDICES = tuple(
    STATE_FEATURE_INDEX[feature] for feature in _LINEAR_VELOCITY_FEATURES
)
if not all(
    _TCP_VEL_SLICE.start <= index < _TCP_VEL_SLICE.stop
    for index in _LINEAR_VELOCITY_INDICES
):  # pragma: no cover - only reachable if the canonical schema is edited wrong
    # A raise rather than an assert: `python -O` strips asserts, and this is a
    # correctness invariant, not a debug aid.
    raise RuntimeError(
        "tcp linear velocity indices "
        f"{_LINEAR_VELOCITY_INDICES} fall outside the tcp_vel group "
        f"{_TCP_VEL_SLICE}; ur_env/observation_schema.py changed shape"
    )


class SidecarScheduler:
    """Decides, per step, whether to pay for a classifier sidecar.

    Stateful and single-threaded: the actor loop owns one instance, calls
    :meth:`should_attach` exactly once per env step, feeds the server's verdict
    back with :meth:`note_outcome`, and calls :meth:`reset` at every episode
    boundary.
    """

    def __init__(
        self,
        *,
        interval_steps: int = 5,
        stationary_speed_max: float = 0.05,
        escalate_probability: float = 0.05,
        enabled: bool = True,
    ) -> None:
        interval = int(interval_steps)
        if interval < 1:
            raise ValueError("interval_steps must be >= 1")
        speed_max = float(stationary_speed_max)
        if not math.isfinite(speed_max) or speed_max < 0.0:
            raise ValueError("stationary_speed_max must be finite and >= 0")
        escalate = float(escalate_probability)
        if not math.isfinite(escalate) or not 0.0 <= escalate <= 1.0:
            raise ValueError("escalate_probability must be finite and in [0, 1]")

        #: 5 steps at HZ=10 is the ~2 Hz cadence the classifier actually needs:
        #: the cube either is or is not in the cup, and that state does not
        #: change faster than the arm can move it.
        self.interval_steps = interval
        #: m/s.  Above this the gripper is sweeping and its own occlusion makes
        #: p(success) meaningless (0.005 -> 1.0 swings observed on cam1).
        self.stationary_speed_max = speed_max
        #: Deliberately well *below* DEFAULT_REWARD_THRESHOLD (0.5): we want to
        #: switch to every-step sampling while merely approaching success, so
        #: the step that actually crosses the threshold is not missed by up to
        #: interval_steps.
        self.escalate_probability = escalate
        self.enabled = bool(enabled)

        self._last_probability = 0.0
        # "Due now": nothing has been attached yet, so the first step of the
        # process attaches and gives the learner a baseline.
        self._steps_since_attach = interval

    # -- observability ----------------------------------------------------- #
    @property
    def last_probability(self) -> float:
        """Most recent classifier probability actually evaluated by the server."""
        return self._last_probability

    @property
    def steps_since_attach(self) -> int:
        return self._steps_since_attach

    @property
    def escalated(self) -> bool:
        """True while the every-step (interval 1) regime is active."""
        return self._last_probability >= self.escalate_probability

    # -- policy ------------------------------------------------------------ #
    def should_attach(self, state: Any, provisional_terminal: bool) -> bool:
        """Return True if this step's observation should carry a sidecar."""
        if not self.enabled:
            return False

        # One call == one env step, counted before any decision so that steps
        # spent moving still age the counter (see the stationary gate below).
        self._steps_since_attach += 1

        # 1. An episode about to end is the one frame we cannot afford to miss:
        #    a success detected one step late is a success recorded as a
        #    failure.  Overrides the stationary gate on purpose.
        if bool(provisional_terminal):
            self._steps_since_attach = 0
            return True

        # 2. Stationary gate.  Note the counter kept growing while moving, so
        #    the very first settled step after a sweep attaches immediately
        #    rather than waiting out another full interval.
        if not self._is_stationary(state):
            return False

        # 3. Cadence, with escalation near success.
        interval = 1 if self.escalated else self.interval_steps
        if self._steps_since_attach >= interval:
            self._steps_since_attach = 0
            return True
        return False

    def note_outcome(self, outcome: Any) -> None:
        """Feed back the server's verdict for the step just sent.

        Accepts an ``actor_network.TransitionOutcome`` (duck-typed, so this
        module keeps no import dependency on the wire dataclasses) or a bare
        probability float.  ``None`` and outcomes the server did not actually
        classify are ignored rather than treated as p=0: a step with no sidecar
        reports ``classifier_probability=0.0`` as a placeholder, and letting
        that cancel an escalation would drop us back to 2 Hz exactly when the
        task is closest to succeeding.
        """
        if outcome is None:
            return

        if isinstance(outcome, bool):
            raise ValueError("note_outcome expects a probability, not a bool")
        if isinstance(outcome, (int, float, np.floating, np.integer)):
            probability = float(outcome)
        else:
            if not getattr(outcome, "classifier_evaluated", True):
                return
            raw = getattr(outcome, "classifier_probability", None)
            if raw is None:
                return
            probability = float(raw)

        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"classifier probability must be finite and in [0, 1], got "
                f"{probability!r}"
            )
        self._last_probability = probability

    def reset(self) -> None:
        """Episode boundary: forget the escalation, arm the next attach."""
        # A new episode starts far from success, so carrying the previous
        # episode's high probability across would pin us at every-step sampling
        # during the reset move for no benefit.
        self._last_probability = 0.0
        self._steps_since_attach = self.interval_steps

    # -- internals --------------------------------------------------------- #
    def _is_stationary(self, state: Any) -> bool:
        array = np.asarray(state)
        if array.ndim == 0 or array.shape[-1] != STATE_DIM:
            raise ValueError(
                "SidecarScheduler.should_attach needs the canonical flat state "
                f"whose last axis is {STATE_DIM}, got shape {array.shape}"
            )
        row = array.reshape(-1, STATE_DIM)[0]
        speed = float(np.linalg.norm(row[list(_LINEAR_VELOCITY_INDICES)]))
        if not math.isfinite(speed):
            # A NaN velocity compares False against every threshold, which
            # would read as "stationary" and attach.  Fail towards not
            # attaching instead.
            return False
        return speed <= self.stationary_speed_max


def sidecar_input_id(enabled: bool = True) -> Optional[str]:
    """:data:`CLASSIFIER_INPUT_ID` when the sidecar is on, else ``None``.

    Exists so metadata producers have one place to spell "which image did the
    classifier actually score", instead of each of them inlining the constant.
    """
    return CLASSIFIER_INPUT_ID if enabled else None
