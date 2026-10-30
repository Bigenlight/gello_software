"""Tests for the reward-classifier sidecar contract.

The load-bearing tests here are the two parity tests.  They are what prove the
sidecar actually closes the train/inference crop mismatch
(docs/testing/08_OPEN_GAPS.md G15) instead of introducing a third, subtly
different image pipeline:

* ``test_resize_matches_the_hardware_validated_viewer_bit_for_bit`` — the
  laptop-side resize is **bit-identical** to what the server would have
  computed.  This is the claim that lets us move the resize off the server at
  all, so it is asserted exactly, with no tolerance.
* ``test_encode_round_trip_stays_within_the_measured_error_bound`` — the one
  thing the move does cost: a JPEG encode at 128x128.  Bounded by measurement,
  not by whatever happens to pass.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_INFRA_ROOT = _HERE.parent
sys.path.insert(0, os.fspath(_INFRA_ROOT))

from ur_env.classifier_sidecar import (  # noqa: E402
    CLASSIFIER_INPUT_ID,
    CLASSIFIER_SIDECAR_KEY,
    DEFAULT_SIDECAR_JPEG_QUALITY,
    MAX_SIDECAR_JPEG_BYTES,
    SIDECAR_CAMERA_KEYS,
    SIDECAR_TENSOR_KEYS,
    SidecarScheduler,
    _resize_for_classifier,
    build_sidecar,
    decode_classifier_frames,
    directory_sha256,
    sidecar_input_id,
    validate_sidecar,
)
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SPEC,
    STATE_DIM,
    STATE_FEATURE_INDEX,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers                                                           #
# --------------------------------------------------------------------------- #
_REPO_ROOT = _INFRA_ROOT.parent
_VIEWER_RUNTIME = (
    _REPO_ROOT
    / "ros2_ur_ws"
    / "src"
    / "gello_recorder"
    / "gello_recorder"
    / "reward_classifier_runtime.py"
)


def _load_viewer_runtime():
    """Import the live viewer's decode helper straight off disk.

    ``gello_recorder`` is a ROS ament package, but this particular module
    imports only cv2/numpy/json/math/os, so loading it by file path gives us the
    real reference implementation without a ROS overlay and without depending on
    the package being installed.
    """
    spec = importlib.util.spec_from_file_location(
        "_viewer_reward_classifier_runtime", _VIEWER_RUNTIME
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _camera_frame(seed: int) -> np.ndarray:
    """A deterministic 1280x720 BGR frame with real-camera statistics.

    Calibrated against the actual rig: encoded at q95 this lands at ~130 KiB,
    inside the 71-143 KiB the 2026-07-29 ``fail_take_*`` recordings measure, and
    its 128x128 round-trip error tracks those frames too (see
    ``test_encode_round_trip_stays_within_the_measured_error_bound``).  Getting
    that right matters: pure noise or saturated flat-colour test cards compress
    and degrade nothing like a camera does, and would make the measured error
    bound below meaningless.

    Recipe: smooth lit background (upscaled coarse noise + heavy blur) + mild
    sensor speckle + the high-contrast task objects (cup rim, cube) that
    dominate the classifier's decision and the JPEG's error budget.
    """
    import cv2

    rng = np.random.default_rng(seed)
    coarse = rng.integers(40, 210, (9, 16, 3), dtype=np.uint8)
    frame = cv2.resize(coarse, (1280, 720), interpolation=cv2.INTER_CUBIC)
    frame = cv2.GaussianBlur(frame, (0, 0), 12)
    noisy = frame.astype(np.int32) + rng.integers(-4, 5, frame.shape)
    frame = np.clip(noisy, 0, 255).astype(np.uint8)
    cv2.circle(frame, (620 + 7 * seed, 380), 150, (95, 90, 88), thickness=-1)
    cv2.circle(frame, (620 + 7 * seed, 380), 130, (150, 148, 143), thickness=-1)
    cv2.rectangle(frame, (560, 320), (680, 440), (40, 40, 200), thickness=-1)
    cv2.rectangle(frame, (120, 90), (300, 250), (30, 160, 60), thickness=-1)
    return cv2.GaussianBlur(frame, (3, 3), 0)


def _published_jpeg(frame: np.ndarray) -> bytes:
    """What the camera node puts on the wire: full-resolution, quality 95."""
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return encoded.tobytes()


@pytest.fixture(scope="module")
def rig():
    """The real pipeline's starting point, per camera.

    ``jpeg`` is what the driver publishes; ``bgr`` is what ``UR7eEnv.get_im()``
    holds after ``cv2.imdecode`` and *before* ``IMAGE_CROP``, which is exactly
    what :func:`build_sidecar` now consumes.  Deriving ``bgr`` from ``jpeg``
    (rather than using the pre-encode frame) matters: it carries the camera's
    own compression artifacts, so the parity comparison isolates the *second*
    encode this change introduces.
    """
    import cv2

    frames = {}
    for index, camera in enumerate(SIDECAR_CAMERA_KEYS):
        jpeg = _published_jpeg(_camera_frame(index))
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        frames[camera] = {"jpeg": jpeg, "bgr": bgr}
    return frames


def _bgr(rig) -> dict[str, np.ndarray]:
    return {camera: rig[camera]["bgr"] for camera in SIDECAR_CAMERA_KEYS}


def _state(**velocities: float) -> np.ndarray:
    """Canonical (1, 19) float32 state with named velocity components set."""
    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    for feature, value in velocities.items():
        state[0, STATE_FEATURE_INDEX[feature]] = value
    return state


_PARKED = _state()
_SWEEPING = _state(tcp_linear_velocity_y=0.4)


class _Outcome:
    """Minimal stand-in for ``actor_network.TransitionOutcome``.

    The scheduler duck-types the outcome so the contract module never has to
    import the wire dataclasses; the test mirrors that by not importing them
    either.
    """

    def __init__(self, probability: float, *, evaluated: bool) -> None:
        self.classifier_probability = probability
        self.classifier_evaluated = evaluated


def _write_tree(root: Path, files: dict[str, bytes]) -> Path:
    for relative, payload in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return root


# --------------------------------------------------------------------------- #
# The parity guarantee                                                         #
# --------------------------------------------------------------------------- #
def test_resize_matches_the_hardware_validated_viewer_bit_for_bit(rig):
    """Moving the resize to the laptop changes NOTHING about the resize.

    This is the assumption the whole 128x128-on-the-laptop design rests on, so
    it is asserted with zero tolerance.  Any drift in the recipe — a different
    interpolation, a stray crop, a channel-order slip — trips here loudly,
    whatever the JPEG encode downstream does.
    """
    viewer = _load_viewer_runtime()

    for camera in SIDECAR_CAMERA_KEYS:
        # What build_sidecar feeds its encoder, i.e. the resize alone.
        resized = _resize_for_classifier(rig[camera]["bgr"])
        laptop = np.ascontiguousarray(resized[..., ::-1], dtype=np.uint8)[None, ...]

        expected = viewer.decode_classifier_image(rig[camera]["jpeg"])

        assert laptop.shape == (1, 128, 128, 3)
        assert laptop.dtype == np.uint8
        assert np.array_equal(laptop, expected)


def test_encode_round_trip_stays_within_the_measured_error_bound(rig):
    """The full laptop->wire->server path versus the viewer's own output.

    These are deliberately NOT bit-identical any more: shipping 128x128 means a
    second JPEG encode sits in the middle (the first being the camera's).  The
    bounds below are measurements, not whatever happened to pass.

    Measured at quality 95 on the 2026-07-29 ``fail_take_*`` rig recordings
    (24 frames, both cameras): MAE 1.29-1.77, p99 5-10, max |diff| 12-87.
    On this file's synthetic frames, which are slightly harsher: MAE 1.88-2.02,
    p99 14-18, max |diff| 62-74.  The bounds are set just above the synthetic
    worst case, so they also cover the real rig.

    MAE and p99 are the meaningful figures.  The max is dominated by JPEG 4:2:0
    chroma subsampling at saturated colour edges and lands on a handful of
    pixels — it is kept only as a coarse "did something structural break" guard.
    (Encoding 4:4:4 instead cuts the real-rig max from 87 to 19 for +22% bytes;
    not adopted here because the encode parameters are being pinned separately.)
    """
    viewer = _load_viewer_runtime()
    sidecar = build_sidecar(_bgr(rig))

    frames = decode_classifier_frames(sidecar)

    for camera in SIDECAR_CAMERA_KEYS:
        expected = viewer.decode_classifier_image(rig[camera]["jpeg"])
        assert frames[camera].shape == (1, 128, 128, 3)
        assert frames[camera].dtype == np.uint8

        diff = np.abs(
            frames[camera].astype(np.int32) - expected.astype(np.int32)
        )
        assert diff.mean() <= 3.0, f"{camera} MAE {diff.mean():.3f}"
        assert np.percentile(diff, 99) <= 24, f"{camera} p99 {np.percentile(diff, 99)}"
        assert diff.max() <= 96, f"{camera} max {diff.max()}"


def test_sidecar_payload_fits_the_step_budget(rig):
    """Why this design exists: the pair must be small enough to actually send.

    The passthrough design shipped 400 KiB per pair (206+194 KiB at the camera's
    q95), which exceeded the 13 Mbit/s link at 2 Hz and added +252 ms to a 100 ms
    step.  Measured 128x128 payloads are 4.87-7.34 KiB per frame on real rig
    frames.  20 KiB for the pair is a generous ceiling that still fails if
    someone reverts to sending full-resolution frames.
    """
    sidecar = build_sidecar(_bgr(rig))

    total = sum(int(sidecar[key].size) for key in SIDECAR_TENSOR_KEYS)
    assert total <= 20 * 1024, f"sidecar pair is {total / 1024:.1f} KiB"
    # And each frame is far below the cap, i.e. the cap is not load-bearing here.
    for key in SIDECAR_TENSOR_KEYS:
        assert sidecar[key].size < MAX_SIDECAR_JPEG_BYTES // 4


def test_round_trip_does_not_crop():
    """The sidecar's whole purpose: the classifier sees the FULL frame.

    ``ur_experiments/cube_in_cup.py::IMAGE_CROP`` keeps cam1's x in [340, 990).
    A frame that is uniform inside that window but distinctive outside it must
    still influence the decoded 128x128 image; if it did not, the sidecar would
    be reproducing the crop it exists to bypass.  Now that the laptop does the
    resizing, this also guards against someone "helpfully" cropping in
    build_sidecar.
    """
    base = np.zeros((720, 1280, 3), dtype=np.uint8)
    base[:, 340:990] = 200  # identical inside the policy crop ...
    marked = base.copy()
    marked[:, 0:100] = 255  # ... different only far outside it

    decoded_plain = decode_classifier_frames(
        build_sidecar({"cam1": base, "cam2": base})
    )
    decoded_edited = decode_classifier_frames(
        build_sidecar({"cam1": marked, "cam2": base})
    )

    assert not np.array_equal(decoded_plain["cam1"], decoded_edited["cam1"])
    assert np.array_equal(decoded_plain["cam2"], decoded_edited["cam2"])


def test_decode_rejects_corrupt_jpeg():
    garbage = np.frombuffer(b"not a jpeg at all", dtype=np.uint8)
    sidecar = {key: garbage for key in SIDECAR_TENSOR_KEYS}

    with pytest.raises(ValueError, match="decode failed"):
        decode_classifier_frames(sidecar)


# --------------------------------------------------------------------------- #
# Contract constants                                                           #
# --------------------------------------------------------------------------- #
def test_sidecar_key_is_outside_the_canonical_policy_observation():
    # The sidecar rides *inside* the observation tensor map but must never be
    # mistaken for a policy tensor, otherwise the schema hash (derived from
    # CANONICAL_OBSERVATION_SPEC) would have to change and every peer handshake
    # would break.
    assert CLASSIFIER_SIDECAR_KEY not in CANONICAL_OBSERVATION_SPEC
    assert set(SIDECAR_TENSOR_KEYS).isdisjoint(CANONICAL_OBSERVATION_SPEC)
    assert SIDECAR_CAMERA_KEYS == ("cam1", "cam2")
    # Pinned deliberately despite the 2026-07-29 switch away from passthrough:
    # the id records the *semantics* (full frame, no crop), which did not
    # change, and it is quoted in docs and in test_rlpd_learner_server_cli.py.
    assert CLASSIFIER_INPUT_ID == "fullframe-jpeg-passthrough-v1"
    assert sidecar_input_id(True) == CLASSIFIER_INPUT_ID
    assert sidecar_input_id(False) is None
    # The cap must stay sized for a 128x128 payload, not the retired full frame.
    assert MAX_SIDECAR_JPEG_BYTES == 64 * 1024
    assert DEFAULT_SIDECAR_JPEG_QUALITY == 95


# --------------------------------------------------------------------------- #
# build_sidecar / validate_sidecar                                             #
# --------------------------------------------------------------------------- #
def test_build_sidecar_emits_a_128x128_jpeg(rig):
    import cv2

    built = build_sidecar(_bgr(rig))

    assert set(built) == set(SIDECAR_TENSOR_KEYS)
    for camera in SIDECAR_CAMERA_KEYS:
        array = built[f"{camera}_jpeg"]
        assert array.dtype == np.uint8 and array.ndim == 1
        assert 0 < array.size <= MAX_SIDECAR_JPEG_BYTES
        # It really is a JPEG, and it really is already 128x128 — so the
        # server's resize is the no-op the design claims.
        assert array[:2].tobytes() == b"\xff\xd8"  # SOI marker
        decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
        assert decoded.shape == (128, 128, 3)


def test_build_sidecar_quality_is_tunable_and_validated(rig):
    frames = _bgr(rig)

    high = build_sidecar(frames, quality=95)
    low = build_sidecar(frames, quality=40)

    # Lower quality must actually reach the encoder; if the keyword were
    # ignored, the constant that is being pinned separately would do nothing.
    assert low["cam1_jpeg"].size < high["cam1_jpeg"].size

    # cv2 silently clamps out-of-range quality, so a typo like 950 would
    # otherwise pass and quietly change the classifier's input distribution.
    for bad in (0, 101, 950, -5):
        with pytest.raises(ValueError, match="quality must be in"):
            build_sidecar(frames, quality=bad)


def test_build_sidecar_names_the_signature_change(rig):
    """The old signature took JPEG bytes; say so instead of dying inside cv2."""
    with pytest.raises(ValueError, match="not JPEG bytes"):
        build_sidecar({"cam1": rig["cam1"]["jpeg"], "cam2": rig["cam2"]["jpeg"]})


@pytest.mark.parametrize(
    "frame, message",
    [
        (np.zeros((720, 1280, 3), np.float32), "must be a uint8 BGR frame"),
        (np.zeros((720, 1280), np.uint8), r"shape \(H, W, 3\)"),
        (np.zeros((720, 1280, 4), np.uint8), r"shape \(H, W, 3\)"),
        (np.zeros((0, 1280, 3), np.uint8), r"shape \(H, W, 3\)|empty frame"),
    ],
)
def test_build_sidecar_rejects_malformed_frames(rig, frame, message):
    with pytest.raises(ValueError, match=message):
        build_sidecar({"cam1": frame, "cam2": rig["cam2"]["bgr"]})


def _small() -> np.ndarray:
    """A valid but tiny BGR frame, for tests about keys rather than pixels."""
    return np.zeros((8, 8, 3), np.uint8)


@pytest.mark.parametrize(
    "cameras, message",
    [
        ({"cam1": _small()}, r"missing=\['cam2'\]"),
        (
            {"cam1": _small(), "cam2": _small(), "cam3": _small()},
            r"extra=\['cam3'\]",
        ),
        ({"cam1_jpeg": _small(), "cam2_jpeg": _small()}, "missing="),
    ],
)
def test_build_sidecar_rejects_wrong_camera_keys(cameras, message):
    with pytest.raises(ValueError, match=message):
        build_sidecar(cameras)


def test_build_sidecar_rejects_non_frame_and_non_mapping():
    with pytest.raises(ValueError, match="must be a uint8 BGR frame"):
        build_sidecar({"cam1": 42, "cam2": _small()})
    with pytest.raises(ValueError, match="mapping"):
        build_sidecar([("cam1", _small()), ("cam2", _small())])


def test_validate_sidecar_returns_contiguous_arrays(rig):
    built = build_sidecar(_bgr(rig))

    validated = validate_sidecar(built)

    assert set(validated) == set(SIDECAR_TENSOR_KEYS)
    for key in SIDECAR_TENSOR_KEYS:
        assert validated[key].flags.c_contiguous
        assert validated[key].tobytes() == built[key].tobytes()


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not a mapping", "must be a mapping"),
        ({"cam1_jpeg": np.zeros(4, np.uint8)}, "missing="),
        (
            {
                "cam1_jpeg": np.zeros(4, np.uint8),
                "cam2_jpeg": np.zeros(4, np.uint8),
                "cam3_jpeg": np.zeros(4, np.uint8),
            },
            "extra=",
        ),
        (
            {
                "cam1_jpeg": np.zeros(4, np.float32),
                "cam2_jpeg": np.zeros(4, np.uint8),
            },
            "must be uint8",
        ),
        (
            {
                "cam1_jpeg": np.zeros((2, 4), np.uint8),
                "cam2_jpeg": np.zeros(4, np.uint8),
            },
            "must be a 1-D byte string",
        ),
        (
            {
                "cam1_jpeg": np.zeros(0, np.uint8),
                "cam2_jpeg": np.zeros(4, np.uint8),
            },
            "is empty",
        ),
        (
            {
                "cam1_jpeg": np.zeros(MAX_SIDECAR_JPEG_BYTES + 1, np.uint8),
                "cam2_jpeg": np.zeros(4, np.uint8),
            },
            "per-frame cap",
        ),
        (
            # Raw bytes instead of an array: np.asarray() makes a 0-d object
            # array, so this is the "forgot build_sidecar()" case.
            {"cam1_jpeg": b"\xff\xd8", "cam2_jpeg": b"\xff\xd8"},
            "must be uint8",
        ),
    ],
)
def test_validate_sidecar_rejects(payload, message):
    with pytest.raises(ValueError, match=message):
        validate_sidecar(payload)


def test_validate_sidecar_accepts_exactly_the_cap():
    at_cap = np.zeros(MAX_SIDECAR_JPEG_BYTES, np.uint8)

    validated = validate_sidecar(
        {"cam1_jpeg": at_cap, "cam2_jpeg": np.zeros(1, np.uint8)}
    )

    assert validated["cam1_jpeg"].size == MAX_SIDECAR_JPEG_BYTES


# --------------------------------------------------------------------------- #
# directory_sha256                                                             #
# --------------------------------------------------------------------------- #
def test_directory_sha256_of_a_regular_file_is_plain_content_sha256(tmp_path):
    payload = b"orbax-shard-bytes" * 100
    single = tmp_path / "checkpoint"
    single.write_bytes(payload)

    # Backward compatibility with rlpd_receive_server.checkpoint_sha256(): any
    # SHA pinned against a single-file checkpoint must keep validating.
    assert directory_sha256(os.fspath(single)) == hashlib.sha256(payload).hexdigest()


def test_directory_sha256_is_deterministic_and_creation_order_independent(tmp_path):
    files = {
        "checkpoint_150/state": b"aaaa",
        "checkpoint_150/metadata.json": b'{"step": 150}',
        "descriptor": b"d",
    }
    first = _write_tree(tmp_path / "a", files)
    # Same content, written in the opposite order, so a hash that depended on
    # filesystem/walk order would differ here.
    second = _write_tree(tmp_path / "b", dict(reversed(list(files.items()))))

    digest = directory_sha256(os.fspath(first))

    assert digest == directory_sha256(os.fspath(first))  # stable across calls
    assert digest == directory_sha256(os.fspath(second))
    assert len(digest) == 64


def test_directory_sha256_notices_a_rename(tmp_path):
    before = _write_tree(tmp_path / "before", {"checkpoint_150/state": b"aaaa"})
    after = _write_tree(tmp_path / "after", {"checkpoint_151/state": b"aaaa"})

    assert directory_sha256(os.fspath(before)) != directory_sha256(os.fspath(after))


def test_directory_sha256_notices_content_and_repartitioning(tmp_path):
    base = _write_tree(tmp_path / "base", {"a": b"xy", "b": b""})
    edited = _write_tree(tmp_path / "edited", {"a": b"xz", "b": b""})
    # Identical concatenated bytes, different split: caught only because the
    # per-file length is mixed into the digest.
    moved = _write_tree(tmp_path / "moved", {"a": b"x", "b": b"y"})

    digests = {
        directory_sha256(os.fspath(path)) for path in (base, edited, moved)
    }
    assert len(digests) == 3


def test_directory_sha256_differs_from_the_same_bytes_as_one_file(tmp_path):
    as_file = tmp_path / "solo"
    as_file.write_bytes(b"aaaa")
    as_dir = _write_tree(tmp_path / "dir", {"solo": b"aaaa"})

    # Directory hashing frames every entry with name+size, so a directory can
    # never collide with the plain content hash of its only member.
    assert directory_sha256(os.fspath(as_file)) != directory_sha256(os.fspath(as_dir))


def test_directory_sha256_requires_an_existing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint path not found"):
        directory_sha256(os.fspath(tmp_path / "nope"))


# --------------------------------------------------------------------------- #
# SidecarScheduler                                                             #
# --------------------------------------------------------------------------- #
def test_scheduler_attaches_on_the_first_step_then_every_interval():
    scheduler = SidecarScheduler(interval_steps=5)

    attached = [scheduler.should_attach(_PARKED, False) for _ in range(11)]

    # Steps 0, 5 and 10 -> 2 Hz on a 10 Hz loop, with the first step attaching
    # so the learner gets a baseline immediately.
    assert attached == [
        True, False, False, False, False,
        True, False, False, False, False,
        True,
    ]


def test_scheduler_never_attaches_while_the_arm_sweeps():
    scheduler = SidecarScheduler(interval_steps=2)

    assert scheduler.should_attach(_PARKED, False) is True
    sweeping = [scheduler.should_attach(_SWEEPING, False) for _ in range(10)]
    assert sweeping == [False] * 10


def test_scheduler_attaches_immediately_once_the_arm_settles():
    scheduler = SidecarScheduler(interval_steps=5)
    scheduler.should_attach(_PARKED, False)  # consume the initial attach

    for _ in range(9):
        assert scheduler.should_attach(_SWEEPING, False) is False

    # The counter kept ageing during the sweep, so the first settled step fires
    # rather than waiting out another full interval.
    assert scheduler.should_attach(_PARKED, False) is True


def test_scheduler_stationary_gate_reads_linear_velocity_only():
    scheduler = SidecarScheduler(interval_steps=1, stationary_speed_max=0.05)

    spinning = _state(
        tcp_angular_velocity_x=3.0,
        tcp_angular_velocity_y=3.0,
        tcp_angular_velocity_z=3.0,
    )
    assert scheduler.should_attach(spinning, False) is True

    # Just over the threshold on a single linear axis is enough to gate.
    creeping = _state(tcp_linear_velocity_z=0.0501)
    assert scheduler.should_attach(creeping, False) is False
    assert scheduler.should_attach(_state(tcp_linear_velocity_z=0.04), False) is True

    # The gate is <=, not <.  Pinned with a threshold that is exact in float32
    # (0.05 is not: float32(0.05) rounds *up*, so a state written as "0.05"
    # would read as moving — a trap worth documenting rather than hiding).
    exact = SidecarScheduler(interval_steps=1, stationary_speed_max=0.0625)
    assert exact.should_attach(_state(tcp_linear_velocity_x=0.0625), False) is True


def test_scheduler_treats_a_non_finite_velocity_as_moving():
    scheduler = SidecarScheduler(interval_steps=1)
    broken = _state(tcp_linear_velocity_x=float("nan"))

    assert scheduler.should_attach(broken, False) is False


def test_scheduler_provisional_terminal_overrides_everything_but_disable():
    scheduler = SidecarScheduler(interval_steps=100)
    scheduler.should_attach(_PARKED, False)

    # Moving, nowhere near due — still attaches, because a success detected one
    # step late is recorded as a failure.
    assert scheduler.should_attach(_SWEEPING, True) is True
    # And it resets the cadence counter like any other attach.
    assert scheduler.should_attach(_PARKED, False) is False


def test_scheduler_escalates_to_every_step_near_success():
    scheduler = SidecarScheduler(interval_steps=5, escalate_probability=0.05)
    assert scheduler.should_attach(_PARKED, False) is True
    assert scheduler.escalated is False

    scheduler.note_outcome(_Outcome(0.05, evaluated=True))

    assert scheduler.escalated is True
    assert [scheduler.should_attach(_PARKED, False) for _ in range(4)] == [True] * 4

    # Falling back below the escalation floor restores the 2 Hz cadence.
    scheduler.note_outcome(_Outcome(0.04, evaluated=True))
    assert scheduler.should_attach(_PARKED, False) is False


def test_scheduler_escalation_still_respects_the_stationary_gate():
    scheduler = SidecarScheduler(interval_steps=5, escalate_probability=0.05)
    scheduler.note_outcome(0.9)

    assert scheduler.should_attach(_SWEEPING, False) is False
    assert scheduler.should_attach(_PARKED, False) is True


def test_scheduler_ignores_outcomes_the_server_did_not_classify():
    scheduler = SidecarScheduler(escalate_probability=0.05)
    scheduler.note_outcome(0.9)

    # A step with no sidecar reports classifier_probability=0.0 as a
    # placeholder; letting that clear the escalation would drop us back to 2 Hz
    # exactly when the task is closest to succeeding.
    scheduler.note_outcome(_Outcome(0.0, evaluated=False))
    assert scheduler.last_probability == pytest.approx(0.9)

    scheduler.note_outcome(None)
    assert scheduler.last_probability == pytest.approx(0.9)

    scheduler.note_outcome(_Outcome(0.0, evaluated=True))
    assert scheduler.last_probability == pytest.approx(0.0)


def test_scheduler_rejects_impossible_probabilities():
    scheduler = SidecarScheduler()

    for bad in (1.5, -0.1, float("nan")):
        with pytest.raises(ValueError, match="probability"):
            scheduler.note_outcome(bad)
    with pytest.raises(ValueError, match="not a bool"):
        scheduler.note_outcome(True)


def test_scheduler_reset_clears_escalation_and_rearms():
    scheduler = SidecarScheduler(interval_steps=5, escalate_probability=0.05)
    scheduler.note_outcome(0.99)
    scheduler.should_attach(_PARKED, False)

    scheduler.reset()

    assert scheduler.escalated is False
    assert scheduler.last_probability == pytest.approx(0.0)
    assert scheduler.should_attach(_PARKED, False) is True
    assert scheduler.should_attach(_PARKED, False) is False


def test_scheduler_disabled_never_attaches():
    scheduler = SidecarScheduler(interval_steps=1, enabled=False)
    scheduler.note_outcome(1.0)

    assert scheduler.should_attach(_PARKED, True) is False
    assert [scheduler.should_attach(_PARKED, False) for _ in range(5)] == [False] * 5


def test_scheduler_rejects_a_non_canonical_state():
    scheduler = SidecarScheduler()

    with pytest.raises(ValueError, match=f"last axis is {STATE_DIM}"):
        scheduler.should_attach(np.zeros((1, STATE_DIM - 1), np.float32), False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_steps": 0},
        {"interval_steps": -1},
        {"stationary_speed_max": -0.1},
        {"stationary_speed_max": float("inf")},
        {"escalate_probability": 1.5},
        {"escalate_probability": -0.01},
    ],
)
def test_scheduler_rejects_nonsensical_configuration(kwargs):
    with pytest.raises(ValueError):
        SidecarScheduler(**kwargs)
