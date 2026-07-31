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

    # Three, not four: _data(3)'s O(t) is _data(2)'s O(t+1) under the same
    # observation_id, so it is served from the trunk cache.  The two rejected
    # transitions above fail before extraction and cache nothing.
    assert extractor.calls == 3
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


def _unique_ids(data: dict[str, Any], step: int) -> dict[str, Any]:
    """Rewrite the ids so consecutive transitions never share an observation."""

    transition = data["transition"]
    transition["observation_id"] = f"uncached-{step}-obs"
    transition["next_observation_id"] = f"uncached-{step}-next"
    return data


def test_reused_observation_id_is_encoded_once_and_changes_nothing():
    """O(t) reuse must be invisible in the rings, not merely cheaper."""

    cached_extractor = _FeatureExtractor()
    cached = _ingress(cached_extractor, replay_capacity=8)
    plain_extractor = _FeatureExtractor()
    plain = _ingress(plain_extractor, replay_capacity=8)

    for step in range(4):
        cached(_data(step), False)
        plain(_unique_ids(_data(step), step), False)

    # 2 for the first transition, then 1 each: O(t) is the previous O(t+1).
    assert cached_extractor.calls == 5
    assert plain_extractor.calls == 8
    assert cached.trunk_extractions == 5
    assert cached.observation_cache_hits == 3
    assert plain.observation_cache_hits == 0

    cached_batch = cached.sample_replay(4)
    plain_batch = plain.sample_replay(4)
    for tree in ("observations", "next_observations"):
        for key in ("state", "cam1", "cam2"):
            assert np.array_equal(
                np.asarray(cached_batch[tree][key]),
                np.asarray(plain_batch[tree][key]),
            ), f"{tree}.{key} diverged from the uncached path"


def test_cached_observation_is_not_aliased_by_the_ring():
    """A cache hit must hand out a private copy, like a fresh extraction."""

    ingress = _ingress(_FeatureExtractor(), replay_capacity=8)
    ingress(_data(0), False)
    ingress(_data(1), False)

    entries = list(ingress._feature_cache.values())
    before = [entry["cam1"].copy() for entry in entries]
    batch = ingress.sample_replay(2)
    np.asarray(batch["observations"]["cam1"]).fill(-1.0)
    after = [entry["cam1"] for entry in ingress._feature_cache.values()]
    for expected, actual in zip(before, after):
        assert np.array_equal(expected, actual)


def test_reused_observation_id_with_a_different_state_is_rejected():
    """The id tripwire replaces the extractor-side state equality check."""

    ingress = _ingress(_FeatureExtractor(), replay_capacity=8)
    ingress(_data(0), False)

    forged = _data(1)
    forged["transition"]["observations"]["state"] = np.full(
        (1, 19), 42.0, dtype=np.float32
    )
    with pytest.raises(ActorProtocolError, match="reused for a different state"):
        ingress(forged, False)


def test_a_failed_extraction_caches_only_what_succeeded():
    """The cache memoises per observation, not per transition.

    An earlier revision required both extractions to succeed before caching
    anything, mirroring the ledger's all-or-nothing insert rule.  That rule
    does not belong here: the cache is a memo of a pure function of the
    pixels, keyed by observation_id and guarded by the state tripwire, so a
    correctly encoded entry is never wrong -- and ``prime_observation`` has
    always cached a lone observation with no transition in sight.  What must
    hold is that the observation whose extraction FAILED is not cached.
    """

    class SecondCallFails:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, observation):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("trunk exploded")
            return _FeatureExtractor()(observation)

    ingress = _ingress(SecondCallFails())
    with pytest.raises(FeatureExtractionError):
        ingress(_data(0), False)

    cached = set(ingress._feature_cache)
    assert ("actor-0", "session-0", "observation-0") in cached  # O(t) succeeded
    assert ("actor-0", "session-0", "observation-1") not in cached  # O(t+1) did not
    # And nothing was inserted, which is the invariant that actually matters.
    assert ingress.status().replay_insert_count == 0


def test_priming_makes_the_first_transition_a_single_extraction():
    """O(0) is encoded at the episode boundary, not inside the first step."""

    primed_extractor = _FeatureExtractor()
    primed = _ingress(primed_extractor, replay_capacity=8)
    plain_extractor = _FeatureExtractor()
    plain = _ingress(plain_extractor, replay_capacity=8)

    primed.prime_observation(
        actor_id="actor-0",
        session_id="session-0",
        observation_id="observation-0",
        observation=_observation(0),
    )
    assert primed_extractor.calls == 1

    primed(_data(0), False)
    plain(_data(0), False)

    # Same total work per episode; the primed one just did not spend it here.
    assert primed_extractor.calls == 2
    assert plain_extractor.calls == 2
    assert primed.observation_cache_hits == 1
    assert plain.observation_cache_hits == 0

    primed_batch = primed.sample_replay(1)
    plain_batch = plain.sample_replay(1)
    for tree in ("observations", "next_observations"):
        for key in ("state", "cam1", "cam2"):
            np.testing.assert_array_equal(
                np.asarray(primed_batch[tree][key]),
                np.asarray(plain_batch[tree][key]),
            )


def test_priming_twice_encodes_once_and_rejects_a_blank_id():
    """BeginEpisode may be retried; the trunk must not run again."""

    extractor = _FeatureExtractor()
    ingress = _ingress(extractor)
    for _ in range(3):
        ingress.prime_observation(
            actor_id="actor-0",
            session_id="session-0",
            observation_id="observation-0",
            observation=_observation(0),
        )

    assert extractor.calls == 1
    with pytest.raises(ActorProtocolError, match="observation_id is required"):
        ingress.prime_observation(
            actor_id="actor-0",
            session_id="session-0",
            observation_id="",
            observation=_observation(0),
        )


def test_priming_a_faulted_gated_ingress_is_refused_and_never_latches():
    """Priming inserts nothing, so its own failure must not retire a learner."""

    class BrokenOnPrime(_FeatureExtractor):
        def __call__(self, observation):
            raise RuntimeError("trunk exploded")

    gated = FaultGatedReplayIngress(_ingress(BrokenOnPrime()))
    with pytest.raises(FeatureExtractionError):
        gated.prime_observation(
            actor_id="actor-0",
            session_id="session-0",
            observation_id="observation-0",
            observation=_observation(0),
        )
    assert gated.healthy
    assert gated.fault is None

    healthy = FaultGatedReplayIngress(_ingress(_FeatureExtractor()))
    with pytest.raises(ActorProtocolError):
        healthy(_data(0, penalty=None), False)
    with pytest.raises(ReplayIngressFaultError):
        healthy.prime_observation(
            actor_id="actor-0",
            session_id="session-0",
            observation_id="observation-0",
            observation=_observation(0),
        )


def test_begin_episode_primes_the_sink_and_tolerates_a_plain_callable():
    """The service hook is optional and passes the validated observation."""

    from ur_env.actor_network import (
        ActorSessionService,
        BeginEpisodeCommand,
        ObservationPacket,
        PROTOCOL_VERSION,
    )

    class RecordingSink:
        def __init__(self) -> None:
            self.primed: list[tuple[str, str, str]] = []

        def __call__(self, data, intervened):
            return None

        def prime_observation(
            self, *, actor_id, session_id, observation_id, observation
        ):
            assert set(observation) == {"state", "cam1", "cam2"}
            self.primed.append((actor_id, session_id, observation_id))

    def _begin(service):
        return service.begin_episode(
            BeginEpisodeCommand(
                PROTOCOL_VERSION,
                "actor",
                "run",
                "session",
                0,
                1,
                10_000,
                ObservationPacket("o0", 1_000, _observation(0)),
            )
        )

    sink = RecordingSink()
    _begin(
        ActorSessionService(
            lambda observation, deterministic: (np.zeros(7, np.float32), 0),
            accept_data=sink,
        )
    )
    assert sink.primed == [("actor", "session", "o0")]

    # A sink without the hook must still begin an episode.
    _begin(
        ActorSessionService(
            lambda observation, deterministic: (np.zeros(7, np.float32), 0),
            accept_data=lambda data, intervened: None,
        )
    )
