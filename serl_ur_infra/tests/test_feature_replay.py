"""Focused tests for frozen-trunk feature replay ingress and rings."""

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Mapping

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import ActorProtocolError  # noqa: E402
from ur_env.learner import (  # noqa: E402
    FEATURE_AUGMENTATION,
    FEATURE_ENCODING_ID,
    FEATURE_MAP_SHAPE,
    FaultGatedReplayIngress,
    FeatureExtractionError,
    FeatureReplayIngress,
    FeatureReplayMemoryError,
    ReplayIngressFaultError,
    estimate_feature_replay_memory,
    preflight_feature_replay_memory,
)


def _observation(value: int) -> dict[str, np.ndarray]:
    return {
        "state": np.full((1, 19), value / 100.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _data(
    step: int,
    *,
    intervened: bool = False,
    penalty: float | None = -0.02,
    transition_id: str | None = None,
) -> dict[str, Any]:
    action = np.array(
        [step / 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        dtype=np.float32,
    )
    transition = {
        "episode_id": 0,
        "step_id": step,
        "observation_id": f"observation-{step}",
        "actions": action,
        "next_observation_id": f"observation-{step + 1}",
        "rewards": 0.0,
        "masks": 1.0,
        "dones": False,
        "truncated": False,
        "observations": _observation(step),
        "next_observations": _observation(step + 1),
        "classifier_evaluated": np.uint8(1),
        "classifier_probability": np.float32(0.1),
        "classifier_threshold": np.float32(0.85),
        "classifier_success": np.uint8(0),
        "success": np.uint8(0),
        "reward_model_id": "test-classifier",
    }
    if penalty is not None:
        transition["grasp_penalty"] = np.float32(penalty)
    return {
        "meta": {
            "schema_version": 3,
            "auto_success": True,
            "operator_success": False,
            "run_id": "run-0",
            "actor_id": "actor-0",
            "session_id": "session-0",
            "transition_id": transition_id or f"transition-{step}",
            "env_step": step,
            "timestamp_ns": 1_700_000_000_000_000_000 + step,
            "policy_version": 0,
            "policy_action": action.copy(),
            "intervened": int(intervened),
        },
        "transition": transition,
    }


class _FeatureExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, observation: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        self.calls += 1
        marker = float(observation["state"][0, 0] * 100.0)
        return {
            "state": observation["state"].copy(),
            "cam1": np.full(FEATURE_MAP_SHAPE, marker, np.float32),
            "cam2": np.full(FEATURE_MAP_SHAPE, marker + 1_000.0, np.float32),
        }


def _ingress(
    extractor: Any,
    *,
    replay_capacity: int = 4,
    intervention_capacity: int = 2,
    seed: int = 42,
    expected_grasp_penalty: float = -0.02,
) -> FeatureReplayIngress:
    return FeatureReplayIngress(
        feature_extractor=extractor,
        replay_capacity=replay_capacity,
        intervention_capacity=intervention_capacity,
        seed=seed,
        expected_grasp_penalty=expected_grasp_penalty,
        available_memory_bytes=16 * 1024**2,
        memory_reserve_bytes=0,
    )


def _camera_markers(batch: Mapping[str, Any], tree: str) -> set[float]:
    values = np.asarray(batch[tree]["cam1"])
    return set(values[:, 0, 0, 0, 0].tolist())


def test_default_memory_estimate_and_preflight_are_exact():
    estimate = estimate_feature_replay_memory()

    assert estimate.total_capacity == 60_000
    assert estimate.camera_bytes == 7_864_320_000
    assert estimate.camera_gib == pytest.approx(7.32421875)
    assert estimate.fixed_tensor_bytes == 7_875_840_000
    assert preflight_feature_replay_memory(
        available_bytes=estimate.fixed_tensor_bytes,
        reserve_bytes=0,
    ) == estimate
    with pytest.raises(FeatureReplayMemoryError, match="feature replay needs"):
        preflight_feature_replay_memory(
            available_bytes=estimate.fixed_tensor_bytes,
            reserve_bytes=1,
        )


def test_explicit_feature_samples_are_seeded_and_retain_no_raw_images():
    first_extractor = _FeatureExtractor()
    second_extractor = _FeatureExtractor()
    first = _ingress(first_extractor, seed=7)
    second = _ingress(second_extractor, seed=7)
    source_items = [_data(0), _data(1)]
    for item in source_items:
        first(copy.deepcopy(item), False)
        second(copy.deepcopy(item), False)

    first_batch = first.sample_replay(32)
    second_batch = second.sample_replay(32)

    assert set(first_batch) == {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "masks",
        "grasp_penalty",
    }
    assert set(first_batch["observations"]) == {"state", "cam1", "cam2"}
    assert set(first_batch["next_observations"]) == {
        "state",
        "cam1",
        "cam2",
    }
    assert first_batch["observations"]["cam1"].shape == (
        32,
        *FEATURE_MAP_SHAPE,
    )
    assert first_batch["next_observations"]["cam2"].shape == (
        32,
        *FEATURE_MAP_SHAPE,
    )
    assert first_batch["observations"]["cam1"].dtype == np.float32
    assert _camera_markers(first_batch, "next_observations") == {
        marker + 1.0
        for marker in _camera_markers(first_batch, "observations")
    }
    for key in first_batch:
        if isinstance(first_batch[key], Mapping):
            for leaf in first_batch[key].values():
                assert np.asarray(leaf).dtype != np.uint8
                assert np.asarray(leaf).shape[-3:] != (128, 128, 3)

    for tree in (first.replay_store._observations, first.replay_store._next_observations):
        for array in tree.values():
            assert array.dtype == np.float32
            assert array.shape[-3:] != (128, 128, 3)
    assert first.replay_store.storage_nbytes + first.intervention_store.storage_nbytes == (
        first.memory_estimate.fixed_tensor_bytes
    )
    for tree_name in ("observations", "next_observations"):
        for key in ("state", "cam1", "cam2"):
            np.testing.assert_array_equal(
                first_batch[tree_name][key], second_batch[tree_name][key]
            )


def test_ring_wrap_intervention_routing_and_sidecars_are_logically_bounded():
    ingress = _ingress(
        _FeatureExtractor(), replay_capacity=2, intervention_capacity=1, seed=3
    )
    ingress(_data(0, intervened=True), True)
    ingress(_data(1), False)
    ingress(_data(2, intervened=True), True)

    status = ingress.status()
    assert status.replay_size == 2
    assert status.intervention_size == 1
    assert status.replay_insert_count == 3
    assert status.intervention_insert_count == 2
    assert status.replay_overwrite_count == 1
    assert status.intervention_overwrite_count == 1
    assert status.last_transition_id == "transition-2"
    assert _camera_markers(ingress.sample_replay(256), "observations") == {
        1.0,
        2.0,
    }
    assert _camera_markers(
        ingress.sample_intervention(64), "observations"
    ) == {2.0}
    assert [record.transition_id for record in ingress.replay_sidecar()] == [
        "transition-1",
        "transition-2",
    ]
    assert all(record.auto_success for record in ingress.replay_sidecar())
    assert not any(
        record.operator_success for record in ingress.replay_sidecar()
    )
    assert not any(record.success for record in ingress.replay_sidecar())
    assert [
        record.transition_id for record in ingress.intervention_sidecar()
    ] == ["transition-2"]


def test_duplicate_is_idempotent_and_raw_byte_collision_is_rejected():
    extractor = _FeatureExtractor()
    ingress = _ingress(extractor)
    original = _data(0, intervened=True)

    ingress(original, True)
    ingress(copy.deepcopy(original), True)

    assert extractor.calls == 2
    assert ingress.status().replay_insert_count == 1
    assert ingress.status().intervention_insert_count == 1
    conflicting = copy.deepcopy(original)
    conflicting["transition"]["next_observations"]["cam1"][0, 0, 0, 0] = 99
    with pytest.raises(ActorProtocolError, match="collision"):
        ingress(conflicting, True)
    assert extractor.calls == 2


def test_exact_configured_grasp_penalty_is_enforced_before_extraction():
    extractor = _FeatureExtractor()
    ingress = _ingress(extractor, expected_grasp_penalty=-0.07)

    with pytest.raises(ActorProtocolError, match="required in learner mode"):
        ingress(_data(0, penalty=None), False)
    with pytest.raises(ActorProtocolError, match="configured penalty -0.07"):
        ingress(_data(1, penalty=-0.02), False)
    ingress(_data(2, penalty=0.0), False)
    ingress(_data(3, penalty=-0.07), False)

    assert extractor.calls == 4
    assert ingress.status().replay_size == 2


@pytest.mark.parametrize("mode", ["raises", "wrong-shape", "wrong-dtype", "state"])
def test_extractor_fault_is_latched_and_no_partial_route_is_sampleable(mode):
    class BrokenExtractor:
        def __call__(self, observation):
            if mode == "raises":
                raise RuntimeError("scripted GPU fault")
            result = _FeatureExtractor()(observation)
            if mode == "wrong-shape":
                result["cam1"] = np.zeros((1, 4, 4, 511), np.float32)
            elif mode == "wrong-dtype":
                result["cam1"] = result["cam1"].astype(np.float64)
            else:
                result["state"] = result["state"] + np.float32(1.0)
            return result

    raw = _ingress(BrokenExtractor())
    ingress = FaultGatedReplayIngress(raw)

    with pytest.raises(FeatureExtractionError):
        ingress(_data(0), False)

    assert raw.status().replay_size == 0
    assert raw.status().intervention_size == 0
    assert ingress.fault is not None
    assert ingress.fault.error_type == "FeatureExtractionError"
    with pytest.raises(ReplayIngressFaultError, match="permanently faulted"):
        ingress.sample_replay(1)
    with pytest.raises(ReplayIngressFaultError, match="permanently faulted"):
        ingress(_data(1), False)


def test_contract_identity_and_packed_image_sampling_options_are_rejected():
    ingress = _ingress(_FeatureExtractor())
    ingress(_data(0), False)

    assert ingress.feature_encoding_id == "resnet10_frozen_trunk_map_f32_v1"
    assert FEATURE_ENCODING_ID == ingress.feature_encoding_id
    assert FEATURE_AUGMENTATION == ingress.augmentation == "none"
    with pytest.raises(ValueError, match="does not support packed-image"):
        ingress.sample_replay(1, pack_obs_and_next_obs=True)
