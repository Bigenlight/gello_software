"""Learner demo contract and RLPD sampling tests."""

from __future__ import annotations

import copy
import os
import pickle
import sys

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.learner import (  # noqa: E402
    CanonicalTransitionPool,
    DemoContractError,
    FROZEN_TRUNK_FEATURE_SHAPE,
    FROZEN_TRUNK_REPRESENTATION,
    LearnerBatchError,
    RLPDBatchSampler,
    load_demo_object,
    load_demo_pickle,
    proportional_sample_counts,
    sample_proportional_counts,
    sanitize_learner_batch,
)


def _observation(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "state": np.full((1, 19), value / 100.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _transition(
    value: int = 0,
    *,
    reward: float = 0.0,
    penalty: float = -0.02,
) -> dict:
    return {
        "observations": _observation(value),
        "next_observations": _observation(value + 1),
        "actions": np.array([0.1, -0.1, 0.2, 0.0, 0.0, 0.0, -1.0], np.float32),
        "rewards": reward,
        "masks": 0.0 if reward else 1.0,
        "dones": bool(reward),
        "grasp_penalty": penalty,
        "episode_id": value,
        "step_id": value,
    }


def test_loads_flat_and_actor_backup_and_normalises_success(tmp_path):
    flat = _transition(1, reward=1.0)
    flat["success"] = True
    actor = {
        "meta": {
            "run_id": "run",
            "timestamp_ns": 123,
            "intervened": 1,
            "policy_action": np.zeros(7, np.float32),
        },
        "transition": _transition(2),
    }
    actor["transition"]["classifier_success"] = np.uint8(0)
    path = tmp_path / "demo.pkl"
    with open(path, "wb") as stream:
        pickle.dump([flat, actor], stream)

    loaded = load_demo_pickle(path)

    assert len(loaded) == 2
    assert set(loaded.transitions[0]) == {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "masks",
        "grasp_penalty",
    }
    assert loaded.sidecars[0].metadata["success"] is True
    assert loaded.sidecars[1].metadata["success"] is False
    assert loaded.sidecars[1].metadata["timestamp_ns"] == 123
    assert loaded.sidecars[1].metadata["intervened"] == 1


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda item: item.update(actions=item["actions"].astype(np.float64)), "dtype"),
        (
            lambda item: item["observations"].update(
                state=np.zeros((19,), np.float32)
            ),
            "shape",
        ),
        (
            lambda item: item["observations"].update(
                cam1=item["observations"]["cam1"].astype(np.float32)
            ),
            "dtype",
        ),
        (lambda item: item.pop("grasp_penalty"), "grasp_penalty is required"),
        (lambda item: item.update(actions=np.zeros(6, np.float32)), "shape"),
    ],
)
def test_demo_loader_rejects_contract_drift(mutate, match):
    item = _transition()
    mutate(item)
    with pytest.raises(DemoContractError, match=match):
        load_demo_object([item])


def test_demo_loader_rejects_joint_space_lerobot_and_label_disagreement():
    with pytest.raises(DemoContractError, match="LeRobot"):
        load_demo_object(
            [
                {
                    "observation.state": np.zeros(6),
                    "joint_positions": np.zeros(6),
                    "actions": np.zeros(6),
                }
            ]
        )

    item = _transition(reward=1.0)
    item.update(success=True, classifier_success=False)
    with pytest.raises(DemoContractError, match="disagree"):
        load_demo_object([item])


def test_canonical_pool_packs_image_pairs_and_keeps_only_learner_fields():
    pool = CanonicalTransitionPool([_transition(1), _transition(2)], seed=3)

    batch = pool.sample(4)

    assert set(batch) == {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "masks",
        "grasp_penalty",
    }
    assert batch["observations"]["cam1"].shape == (4, 2, 128, 128, 3)
    assert batch["observations"]["state"].shape == (4, 1, 19)
    assert set(batch["next_observations"]) == {"state"}


def _feature_batch(batch_size: int = 2) -> dict:
    feature_shape = (batch_size, *FROZEN_TRUNK_FEATURE_SHAPE)
    return {
        "observations": {
            "state": np.zeros((batch_size, 1, 19), dtype=np.float32),
            "cam1": np.zeros(feature_shape, dtype=np.float32),
            "cam2": np.ones(feature_shape, dtype=np.float32),
        },
        "next_observations": {
            "state": np.ones((batch_size, 1, 19), dtype=np.float32),
            "cam1": np.full(feature_shape, 2.0, dtype=np.float32),
            "cam2": np.full(feature_shape, 3.0, dtype=np.float32),
        },
        "actions": np.zeros((batch_size, 7), dtype=np.float32),
        "rewards": np.zeros((batch_size,), dtype=np.float32),
        "masks": np.ones((batch_size,), dtype=np.float32),
        "grasp_penalty": np.zeros((batch_size,), dtype=np.float32),
        "timestamp_ns": np.arange(batch_size, dtype=np.int64),
    }


def test_feature_batch_keeps_explicit_current_next_maps_and_drops_sidecars():
    batch = _feature_batch()

    clean = sanitize_learner_batch(
        batch,
        expected_batch_size=2,
        observation_representation=FROZEN_TRUNK_REPRESENTATION,
    )

    assert set(clean) == {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "masks",
        "grasp_penalty",
    }
    assert set(clean["next_observations"]) == {"state", "cam1", "cam2"}
    assert clean["observations"]["cam1"].shape == (2, 1, 4, 4, 512)


def test_feature_batch_mode_rejects_raw_packed_images_and_feature_drift():
    raw = CanonicalTransitionPool([_transition()], seed=9).sample(2)
    with pytest.raises(LearnerBatchError, match="frozen-trunk"):
        sanitize_learner_batch(
            raw,
            observation_representation=FROZEN_TRUNK_REPRESENTATION,
        )

    bad = _feature_batch()
    bad["next_observations"]["cam1"] = bad["next_observations"][
        "cam1"
    ].astype(np.float16)
    with pytest.raises(LearnerBatchError, match="dtype float32"):
        sanitize_learner_batch(
            bad,
            observation_representation=FROZEN_TRUNK_REPRESENTATION,
        )


def test_rlpd_sampler_is_half_replay_and_proportional_demo_union():
    replay = CanonicalTransitionPool(
        [_transition(value, penalty=-0.3) for value in range(4)], seed=1
    )
    offline = CanonicalTransitionPool(
        [_transition(value + 10, penalty=-0.1) for value in range(3)], seed=2
    )
    interventions = CanonicalTransitionPool(
        [_transition(20, penalty=-0.2)], seed=3
    )
    sampler = RLPDBatchSampler(
        online_replay=replay,
        offline_demos=offline,
        online_interventions=interventions,
        batch_size=8,
        training_starts=4,
        seed=4,
    )

    batch = sampler.sample()

    assert proportional_sample_counts(4, (3, 1)) == (3, 1)
    assert batch["actions"].shape == (8, 7)
    penalties = batch["grasp_penalty"]
    assert np.count_nonzero(penalties == np.float32(-0.3)) == 4
    offline_count = np.count_nonzero(penalties == np.float32(-0.1))
    intervention_count = np.count_nonzero(penalties == np.float32(-0.2))
    assert offline_count + intervention_count == 4
    assert sampler.last_metrics.replay_batch_size == 4
    assert sampler.last_metrics.offline_demo_batch_size == offline_count
    assert (
        sampler.last_metrics.online_intervention_batch_size
        == intervention_count
    )


def test_demo_union_sampling_does_not_starve_a_small_intervention_pool():
    first = sample_proportional_counts(
        200_000,
        (1_000, 1),
        rng=np.random.default_rng(123),
    )
    repeated = sample_proportional_counts(
        200_000,
        (1_000, 1),
        rng=np.random.default_rng(123),
    )

    assert first == repeated
    assert first[0] + first[1] == 200_000
    assert 100 < first[1] < 300


def test_training_requires_online_threshold_and_nonempty_offline_demo():
    replay = CanonicalTransitionPool([_transition()], seed=1)
    empty = CanonicalTransitionPool([], seed=2)
    interventions = CanonicalTransitionPool([_transition()], seed=3)
    sampler = RLPDBatchSampler(
        online_replay=replay,
        offline_demos=empty,
        online_interventions=interventions,
        batch_size=4,
        training_starts=1,
    )
    assert not sampler.ready
