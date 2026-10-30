"""Shutdown rescue: what it writes, what it refuses, and what it never breaks.

The rescue exists because production runs end away from checkpoint boundaries,
so the tests that matter are the ones about behaviour under failure: it must
not raise, it must not become a resume target, and it must dump only the rows
the run actually collected rather than the preallocated ring.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ur_env.learner.checkpoint import (  # noqa: E402
    CheckpointManager,
    EMERGENCY_PREFIX,
)
from ur_env.learner.config import FROZEN_TRUNK_FEATURE_SHAPE  # noqa: E402
from ur_env.learner.emergency import (  # noqa: E402
    ReplaySnapshotError,
    read_replay_snapshot_manifest,
    rescue_learner_state,
    restore_replay_snapshot,
    save_replay_snapshot,
)
from ur_env.learner.feature_replay import (  # noqa: E402
    FEATURE_KEYS,
    FeatureTransitionRing,
)


_STATE_SHAPE = (1, 19)


def _transition(index: int) -> dict[str, object]:
    """One structurally valid feature transition tagged by ``index``."""

    def _maps(offset: float) -> dict[str, np.ndarray]:
        payload = {
            key: np.full(
                FROZEN_TRUNK_FEATURE_SHAPE, index + offset, np.float32
            )
            for key in FEATURE_KEYS
        }
        payload["state"] = np.full(_STATE_SHAPE, index + offset, np.float32)
        return payload

    return {
        "observations": _maps(0.0),
        "next_observations": _maps(0.5),
        # Actions carry two contract bounds: every component in [-1, 1], and
        # the trailing gripper component in {-1, 0, 1}.  The row tag is
        # therefore scaled into the first six and the gripper left at 0.
        "actions": np.array(
            [index / 100.0] * 6 + [0.0], np.float32
        ),
        "rewards": np.float32(0.0),
        "masks": np.float32(1.0),
        "grasp_penalty": np.float32(-0.02),
    }


def _ring(count: int, *, capacity: int) -> FeatureTransitionRing:
    ring = FeatureTransitionRing(capacity, seed=7)
    for index in range(count):
        ring.insert(_transition(index))
    return ring


def test_snapshot_returns_only_filled_rows_not_the_preallocation():
    """The production rings reserve 7.3 GiB; a dump must not write that."""

    ring = _ring(3, capacity=1_000)
    snapshot = ring.snapshot()

    assert snapshot["size"] == 3
    assert snapshot["capacity"] == 1_000
    assert snapshot["actions"].shape == (3, 7)
    for key in FEATURE_KEYS:
        assert snapshot["observations"][key].shape == (
            3,
            *FROZEN_TRUNK_FEATURE_SHAPE,
        )


def test_snapshot_of_an_empty_ring_is_empty_rather_than_an_error():
    snapshot = FeatureTransitionRing(16, seed=1).snapshot()

    assert snapshot["size"] == 0
    assert snapshot["actions"].shape == (0, 7)


def test_snapshot_undoes_ring_rotation_so_rows_are_oldest_first():
    """After wrapping, stored order is rotated; the artifact must not be."""

    capacity = 4
    ring = _ring(6, capacity=capacity)
    snapshot = ring.snapshot()

    assert snapshot["size"] == capacity
    assert snapshot["overwrite_count"] == 2
    # Inserts 0 and 1 were overwritten, so the live rows are 2..5 in order.
    np.testing.assert_array_equal(
        snapshot["actions"][:, 0],
        np.array([2, 3, 4, 5], np.float32) / np.float32(100.0),
    )


def test_snapshot_rows_are_copies_the_ring_cannot_later_mutate():
    ring = _ring(1, capacity=4)
    snapshot = ring.snapshot()
    before = snapshot["actions"].copy()

    for index in range(4, 8):
        ring.insert(_transition(index))

    np.testing.assert_array_equal(snapshot["actions"], before)


def test_replay_snapshot_round_trips_both_rings_with_a_manifest(tmp_path):
    replay = _ring(5, capacity=64).snapshot()
    intervention = _ring(2, capacity=32).snapshot()

    path = save_replay_snapshot(
        tmp_path / "replay.npz",
        replay=replay,
        intervention=intervention,
        provenance={"fingerprint_sha256": "a" * 64},
    )

    manifest = read_replay_snapshot_manifest(path)
    assert manifest["rings"]["replay"]["size"] == 5
    assert manifest["rings"]["intervention"]["size"] == 2
    assert manifest["rings"]["replay"]["capacity"] == 64
    assert manifest["provenance"]["fingerprint_sha256"] == "a" * 64

    with np.load(path) as archive:
        np.testing.assert_array_equal(
            archive["replay/actions"], replay["actions"]
        )
        np.testing.assert_array_equal(
            archive["intervention/actions"], intervention["actions"]
        )
        for key in FEATURE_KEYS:
            np.testing.assert_array_equal(
                archive[f"replay/observations/{key}"],
                replay["observations"][key],
            )
            np.testing.assert_array_equal(
                archive[f"replay/next_observations/{key}"],
                replay["next_observations"][key],
            )


def test_replay_snapshot_refuses_to_overwrite(tmp_path):
    replay = _ring(1, capacity=8).snapshot()
    intervention = _ring(1, capacity=8).snapshot()
    path = save_replay_snapshot(
        tmp_path / "replay.npz", replay=replay, intervention=intervention
    )

    with pytest.raises(FileExistsError):
        save_replay_snapshot(
            path, replay=replay, intervention=intervention
        )


def test_replay_snapshot_leaves_no_partial_file_when_space_is_short(tmp_path):
    replay = _ring(1, capacity=8).snapshot()
    intervention = _ring(1, capacity=8).snapshot()

    with pytest.raises(OSError, match="insufficient space"):
        save_replay_snapshot(
            tmp_path / "replay.npz",
            replay=replay,
            intervention=intervention,
            minimum_free_bytes_after_save=1 << 62,
        )

    assert list(tmp_path.iterdir()) == []


class _StubIngress:
    observation_representation = "resnet10_frozen_trunk_map_f32_v1"
    augmentation = "none"

    def __init__(self, replay_rows: int, intervention_rows: int) -> None:
        self.replay_store = _ring(replay_rows, capacity=64)
        self.intervention_store = _ring(intervention_rows, capacity=64)


class _StubLearner:
    def __init__(self, learner_step: int) -> None:
        self.learner_step = learner_step
        self.gradient_step = learner_step * 2
        self.policy_version = learner_step // 50
        self.agent = object()


class _StubFingerprint:
    sha256 = "b" * 64


class _RecordingManager:
    """A CheckpointManager stand-in that records saves without needing JAX."""

    def __init__(self, root: Path, *, fail: bool = False) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.minimum_free_bytes_after_save = 0
        self.calls: list[dict[str, object]] = []
        self.fail = fail

    # Reuse the real naming so the prefix split under test is the shipped one.
    checkpoint_name = staticmethod(CheckpointManager.checkpoint_name)

    def path_for_step(self, learner_step: int, *, emergency: bool = False):
        return self.root / self.checkpoint_name(
            learner_step, emergency=emergency
        )

    def save(self, **kwargs):
        if self.fail:
            raise RuntimeError("device is gone")
        self.calls.append(kwargs)
        path = self.path_for_step(
            kwargs["learner_step"], emergency=kwargs.get("emergency", False)
        )
        path.mkdir()
        return path


def _rescue(manager, *, learner_step=317, replay_rows=5, intervention_rows=3):
    return rescue_learner_state(
        learner=_StubLearner(learner_step),
        ingress=_StubIngress(replay_rows, intervention_rows),
        checkpoint_manager=manager,
        fingerprint=_StubFingerprint(),
        inference_rng=np.array([0, 1], np.uint32),
    )


def test_rescue_writes_an_emergency_prefixed_checkpoint_and_a_snapshot(
    tmp_path,
):
    manager = _RecordingManager(tmp_path / "checkpoints")

    result = _rescue(manager)

    assert result.saved_anything
    assert result.checkpoint_error is None
    assert result.replay_snapshot_error is None
    assert Path(result.checkpoint_path).name == f"{EMERGENCY_PREFIX}" + "317".zfill(12)
    assert manager.calls[0]["emergency"] is True
    assert result.replay_rows == 5
    assert result.intervention_rows == 3

    manifest = read_replay_snapshot_manifest(result.replay_snapshot_path)
    assert manifest["provenance"]["learner_step"] == 317
    assert manifest["provenance"]["fingerprint_sha256"] == "b" * 64


def test_rescue_checkpoint_is_invisible_to_automatic_resume(tmp_path):
    """--resume-latest must never land on a rescue taken off-boundary."""

    real = CheckpointManager(tmp_path / "checkpoints")
    (real.root / f"{EMERGENCY_PREFIX}{317:012d}").mkdir()

    with pytest.raises(FileNotFoundError):
        real.latest_path()


def test_rescue_still_dumps_the_buffer_when_the_checkpoint_fails(tmp_path):
    """The two artifacts fail for unrelated reasons and must be independent."""

    manager = _RecordingManager(tmp_path / "checkpoints", fail=True)

    result = _rescue(manager)

    assert result.checkpoint_path is None
    assert "device is gone" in result.checkpoint_error
    assert result.replay_snapshot_path is not None
    assert result.replay_rows == 5


def test_rescue_skips_the_checkpoint_when_the_boundary_already_saved_it(
    tmp_path,
):
    manager = _RecordingManager(tmp_path / "checkpoints")
    (manager.root / f"checkpoint_{317:012d}").mkdir()

    result = _rescue(manager)

    assert result.checkpoint_path is None
    assert "periodic checkpoint already exists" in result.checkpoint_error
    assert manager.calls == []
    # The rings are never persisted by the periodic path, so they still go.
    assert result.replay_snapshot_path is not None


def test_rescue_declines_a_run_that_never_trained(tmp_path):
    manager = _RecordingManager(tmp_path / "checkpoints")

    result = _rescue(manager, learner_step=0)

    assert result.saved_anything is False
    assert result.skipped_reason == "learner never advanced past step 0"
    assert list(manager.root.iterdir()) == []


def test_rescue_reports_rather_than_raises_when_the_learner_is_unreadable(
    tmp_path,
):
    """A learner that died mid-update must not take the shutdown path with it."""

    class _BrokenLearner:
        @property
        def learner_step(self):
            raise RuntimeError("agent is gone")

    result = rescue_learner_state(
        learner=_BrokenLearner(),
        ingress=_StubIngress(1, 1),
        checkpoint_manager=_RecordingManager(tmp_path / "checkpoints"),
        fingerprint=_StubFingerprint(),
        inference_rng=np.array([0, 1], np.uint32),
    )

    assert result.attempted is True
    assert result.saved_anything is False
    assert "agent is gone" in result.skipped_reason


def test_rescue_reports_rather_than_raises_when_the_rings_are_unreadable(
    tmp_path,
):
    class _BrokenIngress:
        @property
        def replay_store(self):
            raise RuntimeError("ring is gone")

    manager = _RecordingManager(tmp_path / "checkpoints")
    result = rescue_learner_state(
        learner=_StubLearner(317),
        ingress=_BrokenIngress(),
        checkpoint_manager=manager,
        fingerprint=_StubFingerprint(),
        inference_rng=np.array([0, 1], np.uint32),
    )

    assert result.checkpoint_path is not None
    assert "ring is gone" in result.replay_snapshot_error


def test_event_fields_drop_unset_entries_so_logs_stay_readable(tmp_path):
    manager = _RecordingManager(tmp_path / "checkpoints")

    fields = _rescue(manager).event_fields()

    assert "skipped_reason" not in fields
    assert "checkpoint_error" not in fields
    assert fields["learner_step"] == 317
    assert json.dumps(fields)


# --------------------------------------------------------------------------- #
# restore                                                                      #
# --------------------------------------------------------------------------- #


class _RestoreIngress:
    """Two real rings, so restore runs through the real insert contract."""

    observation_representation = "resnet10_frozen_trunk_map_f32_v1"
    augmentation = "none"

    def __init__(self, *, capacity: int = 64) -> None:
        self.replay_store = FeatureTransitionRing(capacity, seed=11)
        self.intervention_store = FeatureTransitionRing(capacity, seed=12)


def _written_snapshot(tmp_path, *, replay_rows=5, intervention_rows=2,
                      fingerprint="b" * 64, name="replay.npz"):
    source = _StubIngress(replay_rows, intervention_rows)
    return save_replay_snapshot(
        tmp_path / name,
        replay=source.replay_store.snapshot(),
        intervention=source.intervention_store.snapshot(),
        provenance={
            "fingerprint_sha256": fingerprint,
            "observation_representation": _StubIngress.observation_representation,
            "augmentation": _StubIngress.augmentation,
        },
    )


def test_restore_refills_both_rings_with_identical_rows(tmp_path):
    path = _written_snapshot(tmp_path, replay_rows=5, intervention_rows=2)
    ingress = _RestoreIngress()

    result = restore_replay_snapshot(
        path, ingress=ingress, expected_fingerprint_sha256="b" * 64
    )

    assert result["replay_rows"] == 5
    assert result["intervention_rows"] == 2
    assert len(ingress.replay_store) == 5
    assert len(ingress.intervention_store) == 2

    # Round-tripped rows must be bit-identical, in order.
    restored = ingress.replay_store.snapshot()
    expected = _ring(5, capacity=64).snapshot()
    np.testing.assert_array_equal(restored["actions"], expected["actions"])
    for key in FEATURE_KEYS:
        np.testing.assert_array_equal(
            restored["observations"][key], expected["observations"][key]
        )


def test_restore_refuses_a_snapshot_from_another_lineage(tmp_path):
    path = _written_snapshot(tmp_path, fingerprint="c" * 64)
    ingress = _RestoreIngress()

    with pytest.raises(ReplaySnapshotError, match="different learner lineage"):
        restore_replay_snapshot(
            path, ingress=ingress, expected_fingerprint_sha256="b" * 64
        )

    assert len(ingress.replay_store) == 0


def test_restore_refuses_when_the_representation_disagrees(tmp_path):
    path = _written_snapshot(tmp_path)
    ingress = _RestoreIngress()
    ingress.observation_representation = "some_other_encoding_v9"

    with pytest.raises(ReplaySnapshotError, match="observation_representation"):
        restore_replay_snapshot(path, ingress=ingress)


def test_restore_refuses_rather_than_wrapping_a_too_small_ring(tmp_path):
    """A ring that wraps mid-restore silently loses its oldest rows."""

    path = _written_snapshot(tmp_path, replay_rows=10, intervention_rows=1)
    ingress = _RestoreIngress(capacity=4)

    with pytest.raises(ReplaySnapshotError, match="silently drop"):
        restore_replay_snapshot(path, ingress=ingress)


def test_restore_rejects_an_unsupported_format_version(tmp_path):
    path = _written_snapshot(tmp_path)
    with np.load(path) as archive:
        payload = {key: archive[key] for key in archive.files}
    manifest = json.loads(bytes(payload["manifest.json"]).decode("utf-8"))
    manifest["format_version"] = 99
    payload["manifest.json"] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8
    )
    bumped = tmp_path / "bumped.npz"
    np.savez(bumped, **payload)

    with pytest.raises(ReplaySnapshotError, match="format version"):
        restore_replay_snapshot(bumped, ingress=_RestoreIngress())


def test_restored_rows_do_not_earn_a_second_utd_budget():
    """A restart must not open with a burst of updates over old data."""

    from ur_env.learner.composition import LearnerWorker

    class _Learner:
        learner_step = 0
        gradient_step = 0
        policy_version = 0

        class config:
            training_starts = 100
            utd_ratio = 10

    inserts = {"count": 600}
    worker = LearnerWorker.__new__(LearnerWorker)
    worker.learner = _Learner()
    worker.replay_insert_count = lambda: inserts["count"]
    worker._last_replay_insert_count = None
    worker._restored_replay_rows = 500

    # 600 inserted, 500 of them restored -> only 101 fresh transitions count.
    assert worker._update_budget() == (600 - 100 + 1 - 500) * 10

    # Without the restore baseline the same buffer would authorise 5010 steps.
    worker._restored_replay_rows = 0
    worker._last_replay_insert_count = None
    assert worker._update_budget() == (600 - 100 + 1) * 10


def test_budget_is_zero_while_only_restored_rows_are_present():
    from ur_env.learner.composition import LearnerWorker

    class _Learner:
        learner_step = 0

        class config:
            training_starts = 100
            utd_ratio = 10

    worker = LearnerWorker.__new__(LearnerWorker)
    worker.learner = _Learner()
    worker.replay_insert_count = lambda: 500
    worker._last_replay_insert_count = None
    worker._restored_replay_rows = 500

    assert worker._update_budget() == 0
