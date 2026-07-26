"""Receive-server reward and replay tests without robot or gradient updates."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import sys
from typing import Any

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_INFRA_ROOT, ".."))
sys.path.insert(0, _INFRA_ROOT)

from ur_env.actor_network import ActorProtocolError  # noqa: E402
from ur_env.rlpd_receive_server import (  # noqa: E402
    FakeActionRuntime,
    ReplayIngress,
    RewardClassifierError,
    RewardClassifierRuntime,
    RewardTransitionFinalizer,
    ScriptedRewardClassifierRuntime,
    checkpoint_sha256,
    sigmoid_probability,
)


def _observation(value: int) -> dict[str, np.ndarray]:
    return {
        "state": np.full((1, 19), value / 100.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _data(
    *,
    step: int = 0,
    intervened: bool = False,
    reward: float = 0.0,
    done: bool = False,
    truncated: bool = False,
    source_value: int | None = None,
    next_value: int | None = None,
    transition_id: str | None = None,
    session_id: str = "session-0",
) -> dict[str, Any]:
    source_value = step if source_value is None else source_value
    next_value = step + 1 if next_value is None else next_value
    policy_action = np.full(7, step / 100.0, dtype=np.float32)
    executed_action = (
        np.full(7, -step / 100.0, dtype=np.float32)
        if intervened
        else policy_action.copy()
    )
    return {
        "meta": {
            "schema_version": 2,
            "run_id": "run-0",
            "actor_id": "actor-0",
            "session_id": session_id,
            "transition_id": transition_id or f"transition-{step}",
            "env_step": step,
            "timestamp_ns": 1_700_000_000_000_000_000 + step,
            "policy_version": 0,
            "policy_action": policy_action,
            "intervened": int(intervened),
        },
        "transition": {
            "episode_id": 0,
            "step_id": step,
            "observation_id": f"observation-{step}",
            "actions": executed_action,
            "next_observation_id": f"observation-{step + 1}",
            "rewards": reward,
            "masks": 0.0 if done else 1.0,
            "dones": done,
            "truncated": truncated,
            "observations": _observation(source_value),
            "next_observations": _observation(next_value),
        },
    }


def _finalized_data(**kwargs: Any) -> dict[str, Any]:
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([kwargs.pop("probability", 0.1)])
    )
    data, _ = finalizer(_data(**kwargs))
    return data


class _FakeMemoryStore:
    """Small boundary-aware stand-in for dependency-free unit tests."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._first = True
        self.dataset_dict: dict[str, Any] = {}
        self.items: list[dict[str, Any]] = []
        self.started_sequences: list[bool] = []
        self.fail_next_insert = False

    def __len__(self) -> int:
        return min(len(self.items), self._capacity)

    def insert(self, transition: dict[str, Any]) -> None:
        if self.fail_next_insert:
            self.fail_next_insert = False
            raise BufferError("scripted insert failure")
        self.started_sequences.append(bool(self._first))
        self.items.append(copy.deepcopy(transition))
        self._first = bool(transition["dones"])

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        return {"args": args, "kwargs": kwargs, "items": self.items}


class _StoreFactory:
    def __init__(self) -> None:
        self.stores: list[_FakeMemoryStore] = []

    def __call__(self, *, capacity: int, **kwargs: Any) -> _FakeMemoryStore:
        del kwargs
        store = _FakeMemoryStore(capacity)
        self.stores.append(store)
        return store


def test_fake_action_runtime_defaults_to_safe_zero_and_validates_schema():
    runtime = FakeActionRuntime()

    action, version = runtime(_observation(1), deterministic=True)

    np.testing.assert_array_equal(action, np.zeros(7, dtype=np.float32))
    assert version == 0
    assert runtime.sample_count == 1
    malformed = _observation(1)
    malformed["state"] = malformed["state"].astype(np.float64)
    with pytest.raises(ActorProtocolError, match="dtype float32"):
        runtime(malformed, deterministic=True)


def test_fake_action_runtime_uses_script_once_and_fails_closed():
    expected = np.linspace(-0.5, 0.5, 7, dtype=np.float32)
    runtime = FakeActionRuntime([expected], policy_version=7)

    action, version = runtime(_observation(0), deterministic=False)

    np.testing.assert_array_equal(action, expected)
    assert version == 7
    with pytest.raises(RuntimeError, match="sequence exhausted"):
        runtime(_observation(1), deterministic=False)


@pytest.mark.parametrize(
    ("logit", "expected"),
    [(-1_000.0, 0.0), (0.0, 0.5), (1_000.0, 1.0)],
)
def test_sigmoid_probability_is_stable(logit: float, expected: float):
    assert sigmoid_probability(logit) == pytest.approx(expected)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_sigmoid_probability_rejects_non_finite_values(value: float):
    with pytest.raises(RewardClassifierError, match="finite"):
        sigmoid_probability(value)


def test_reward_classifier_validates_checksum_exact_inputs_and_warms_up(tmp_path):
    checkpoint = tmp_path / "checkpoint_150"
    checkpoint.write_bytes(b"test-flax-checkpoint")
    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls: list[dict[str, np.ndarray]] = []

    def loader(sample):
        assert sample["state"].shape == (1, 1)
        assert sample["state"].dtype == np.float32
        assert sample["cam1"].shape == (1, 128, 128, 3)

        def classifier(observation):
            calls.append({key: np.asarray(value) for key, value in observation.items()})
            return np.array(2.0, dtype=np.float32)

        return classifier

    runtime = RewardClassifierRuntime(
        checkpoint_path=str(checkpoint),
        expected_sha256=expected_sha,
        classifier_loader=loader,
        reward_model_id="cube-in-cup-test",
    )
    result = runtime.classify(_observation(23))

    assert runtime.ready
    assert runtime.evaluation_count == 1
    assert checkpoint_sha256(str(checkpoint)) == expected_sha
    assert len(calls) == 2  # one JIT/warmup-equivalent call plus inference
    assert calls[1]["state"].shape == (1, 1)
    assert np.count_nonzero(calls[1]["state"]) == 0
    np.testing.assert_array_equal(calls[1]["cam1"], _observation(23)["cam1"])
    assert result.probability == pytest.approx(sigmoid_probability(2.0))
    assert result.success is True
    assert result.reward_model_id == "cube-in-cup-test"


def test_reward_classifier_rejects_wrong_checkpoint_hash(tmp_path):
    checkpoint = tmp_path / "checkpoint_150"
    checkpoint.write_bytes(b"wrong")

    with pytest.raises(RewardClassifierError, match="SHA256 mismatch"):
        RewardClassifierRuntime(
            checkpoint_path=str(checkpoint),
            expected_sha256="0" * 64,
            classifier_loader=lambda sample: lambda observation: 0.0,
        )


@pytest.mark.parametrize(
    (
        "probability",
        "local_done",
        "local_truncated",
        "expected_reward",
        "expected_done",
        "expected_truncated",
        "expected_mask",
        "expected_success",
    ),
    [
        (0.10, False, False, 0.0, False, False, 1.0, False),
        (0.90, False, False, 1.0, True, False, 0.0, True),
        (0.10, False, True, 0.0, False, True, 1.0, False),
        # A classifier-positive O(t+1) wins over a simultaneous time limit.
        (0.90, False, True, 1.0, True, False, 0.0, True),
        (0.10, True, False, 0.0, True, False, 0.0, False),
    ],
)
def test_reward_finalizer_server_authority(
    probability,
    local_done,
    local_truncated,
    expected_reward,
    expected_done,
    expected_truncated,
    expected_mask,
    expected_success,
):
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime(
            [probability], threshold=0.85, reward_model_id="cube-in-cup-v1"
        )
    )
    data, outcome = finalizer(
        _data(done=local_done, truncated=local_truncated)
    )

    transition = data["transition"]
    assert transition["rewards"] == expected_reward
    assert transition["dones"] is expected_done
    assert transition["truncated"] is expected_truncated
    assert transition["masks"] == expected_mask
    assert bool(transition["classifier_evaluated"])
    assert float(transition["classifier_probability"]) == pytest.approx(probability)
    assert bool(transition["classifier_success"]) is expected_success
    assert outcome.reward == expected_reward
    assert outcome.done is expected_done
    assert outcome.truncated is expected_truncated
    assert outcome.mask == expected_mask
    assert outcome.success is expected_success
    assert outcome.classifier_evaluated is True
    assert outcome.reward_model_id == "cube-in-cup-v1"


def test_reward_finalizer_uses_strict_threshold():
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([0.85], threshold=0.85)
    )

    data, outcome = finalizer(_data())

    assert outcome.success is False
    assert data["transition"]["dones"] is False


def test_reward_finalizer_rejects_provisional_positive_reward_when_negative():
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([0.1], threshold=0.85)
    )

    data, outcome = finalizer(_data(reward=1.0))

    assert data["transition"]["rewards"] == 0.0
    assert outcome.reward == 0.0
    assert outcome.success is False


def test_reward_finalizer_classifier_failure_produces_no_data():
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([RuntimeError("GPU fault")])
    )

    with pytest.raises(RewardClassifierError, match="GPU fault"):
        finalizer(_data())


def test_replay_ingress_routes_all_and_interventions_with_gap_boundaries():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=8,
        intervention_capacity=4,
        store_factory=factory,
    )
    assert factory.stores == [ingress.replay_store, ingress.intervention_store]

    ingress(_finalized_data(step=0, intervened=True), True)
    ingress(_finalized_data(step=1, intervened=False), False)
    ingress(_finalized_data(step=2, intervened=True), True)

    replay, intervention = factory.stores
    assert replay.started_sequences == [True, False, False]
    assert intervention.started_sequences == [True, True]
    assert len(replay.items) == 3
    assert len(intervention.items) == 2
    assert [int(item["intervened"]) for item in replay.items] == [1, 0, 1]
    assert all(item["policy_actions"].shape == (7,) for item in replay.items)
    assert set(ingress._NUMERIC_METADATA).issubset(replay.dataset_dict)

    status = ingress.status()
    assert status.replay_size == 3
    assert status.intervention_size == 2
    assert status.replay_insert_count == 3
    assert status.intervention_insert_count == 2
    assert status.last_transition_id == "transition-2"
    assert status.last_env_step == 2
    assert [record.transition_id for record in ingress.replay_sidecar()] == [
        "transition-0",
        "transition-1",
        "transition-2",
    ]


def test_replay_ingress_uses_truncation_only_as_stack_boundary():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=2,
        store_factory=factory,
    )
    ingress(_finalized_data(step=0, truncated=True), False)

    stored = ingress.replay_store.items[0]
    assert bool(stored["dones"]) is True
    assert bool(stored["terminated"]) is False
    assert bool(stored["truncated"]) is True
    assert float(stored["masks"]) == 1.0


def test_replay_ingress_duplicate_is_idempotent_and_collision_rejected():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=4,
        store_factory=factory,
    )
    original = _finalized_data(step=0, intervened=True)

    ingress(original, True)
    ingress(copy.deepcopy(original), True)

    assert ingress.status().replay_insert_count == 1
    assert ingress.status().intervention_insert_count == 1
    assert len(ingress.replay_store.items) == 1
    assert len(ingress.intervention_store.items) == 1
    conflicting = copy.deepcopy(original)
    conflicting["transition"]["rewards"] = 0.5
    with pytest.raises(ActorProtocolError, match="collision"):
        ingress(conflicting, True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["meta"].__setitem__("timestamp_ns", 42),
        lambda data: data["transition"]["next_observations"]["cam1"].__setitem__(
            (0, 0, 0, 0), 99
        ),
        lambda data: data["transition"].__setitem__(
            "classifier_probability", 0.2
        ),
    ],
)
def test_replay_ingress_id_collision_covers_metadata_and_observation_bytes(
    mutation,
):
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=4,
        store_factory=_StoreFactory(),
    )
    original = _finalized_data(step=0)
    ingress(original, False)
    conflicting = copy.deepcopy(original)
    mutation(conflicting)

    with pytest.raises(ActorProtocolError, match="collision"):
        ingress(conflicting, False)


def test_replay_ingress_retries_only_missing_route_after_partial_failure():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=4,
        store_factory=factory,
    )
    data = _finalized_data(step=0, intervened=True)
    ingress.intervention_store.fail_next_insert = True

    with pytest.raises(BufferError, match="scripted insert failure"):
        ingress(data, True)
    assert len(ingress.replay_store.items) == 1
    assert len(ingress.intervention_store.items) == 0

    ingress(data, True)

    assert len(ingress.replay_store.items) == 1
    assert len(ingress.intervention_store.items) == 1
    assert ingress.status().replay_insert_count == 1
    assert ingress.status().intervention_insert_count == 1


def test_replay_ingress_status_reports_logical_circular_overwrites():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=2,
        intervention_capacity=1,
        store_factory=factory,
        ledger_capacity=2,
    )
    for step in range(3):
        ingress(_finalized_data(step=step, intervened=True), True)

    status = ingress.status()
    assert status.replay_size == 2
    assert status.intervention_size == 1
    assert status.replay_insert_count == 3
    assert status.intervention_insert_count == 3
    assert status.replay_overwrite_count == 1
    assert status.intervention_overwrite_count == 2
    assert len(ingress.replay_sidecar()) == 2
    assert len(ingress.intervention_sidecar()) == 1


def _actual_hil_serl_root() -> str:
    root = os.environ.get(
        "HIL_SERL_ROOT", os.path.join(_REPO_ROOT, "third_party", "hil-serl")
    )
    launcher = os.path.join(
        root, "serl_launcher", "serl_launcher", "data", "data_store.py"
    )
    if not os.path.isfile(launcher):
        pytest.skip("pinned HIL-SERL submodule is not initialized")
    for dependency in ("jax", "flax", "agentlace"):
        pytest.importorskip(dependency)
    return root


def test_actual_upstream_buffers_sample_packed_batch_and_keep_gap_boundaries():
    hil_serl_root = _actual_hil_serl_root()
    ingress = ReplayIngress(
        replay_capacity=32,
        intervention_capacity=16,
        hil_serl_root=hil_serl_root,
    )
    ingress(
        _finalized_data(
            step=0, intervened=True, source_value=10, next_value=11
        ),
        True,
    )
    ingress(
        _finalized_data(
            step=1, intervened=False, source_value=11, next_value=12
        ),
        False,
    )
    ingress(
        _finalized_data(
            step=2, intervened=True, source_value=20, next_value=21
        ),
        True,
    )

    replay_batch = ingress.sample_replay(batch_size=8)
    assert replay_batch["observations"]["cam1"].shape == (
        8,
        2,
        128,
        128,
        3,
    )
    assert replay_batch["observations"]["state"].shape == (8, 1, 19)
    assert replay_batch["policy_actions"].shape == (8, 7)
    assert replay_batch["classifier_probability"].shape == (8,)

    # Both intervention samples are isolated sequences.  Packed frames must be
    # their actual O(t),O(t+1) pairs, never the prior policy-only frame.
    intervention_batch = ingress.sample_intervention(batch_size=256)
    pixels = np.asarray(intervention_batch["observations"]["cam1"])
    pairs = set(
        zip(
            pixels[:, 0, 0, 0, 0].tolist(),
            pixels[:, 1, 0, 0, 0].tolist(),
        )
    )
    assert pairs == {(10, 11), (20, 21)}


def test_actual_upstream_capacity_counts_logical_transitions_not_bootstrap_frames():
    hil_serl_root = _actual_hil_serl_root()
    ingress = ReplayIngress(
        replay_capacity=2,
        intervention_capacity=2,
        hil_serl_root=hil_serl_root,
    )
    for step, value in enumerate((10, 20, 30)):
        ingress(
            _finalized_data(
                step=step,
                intervened=True,
                source_value=value,
                next_value=value + 1,
                session_id=f"isolated-session-{step}",
            ),
            True,
        )

    status = ingress.status()
    assert status.replay_size == 2
    assert status.intervention_size == 2
    assert status.replay_insert_count == 3
    assert status.intervention_insert_count == 3
    assert status.replay_overwrite_count == 1
    assert status.intervention_overwrite_count == 1
    assert np.count_nonzero(ingress.replay_store._is_correct_index) == 2
    assert np.count_nonzero(ingress.intervention_store._is_correct_index) == 2

    for sample in (
        ingress.sample_replay(batch_size=256),
        ingress.sample_intervention(batch_size=256),
    ):
        pixels = np.asarray(sample["observations"]["cam1"])
        pairs = set(
            zip(
                pixels[:, 0, 0, 0, 0].tolist(),
                pixels[:, 1, 0, 0, 0].tolist(),
            )
        )
        assert pairs == {(20, 21), (30, 31)}


@pytest.mark.parametrize(
    "sessions",
    [
        ["continuous"] * 9,
        ["a", "a", "b", "b", "b", "c", "c", "d", "d"],
    ],
)
def test_actual_upstream_logical_capacity_survives_wraps_and_mixed_boundaries(
    sessions,
):
    hil_serl_root = _actual_hil_serl_root()
    ingress = ReplayIngress(
        replay_capacity=2,
        intervention_capacity=2,
        hil_serl_root=hil_serl_root,
    )
    for step, session_id in enumerate(sessions):
        ingress(
            _finalized_data(
                step=step,
                intervened=True,
                source_value=10 + step,
                next_value=11 + step,
                session_id=session_id,
            ),
            True,
        )

    status = ingress.status()
    expected_overwrites = len(sessions) - 2
    assert status.replay_size == status.replay_capacity == 2
    assert status.intervention_size == status.intervention_capacity == 2
    assert status.replay_overwrite_count == expected_overwrites
    assert status.intervention_overwrite_count == expected_overwrites
    expected_pairs = {
        (10 + len(sessions) - 2, 11 + len(sessions) - 2),
        (10 + len(sessions) - 1, 11 + len(sessions) - 1),
    }
    for store, sample in (
        (ingress.replay_store, ingress.sample_replay(batch_size=512)),
        (
            ingress.intervention_store,
            ingress.sample_intervention(batch_size=512),
        ),
    ):
        assert np.count_nonzero(store._is_correct_index) == 2
        pixels = np.asarray(sample["observations"]["cam1"])
        pairs = set(
            zip(
                pixels[:, 0, 0, 0, 0].tolist(),
                pixels[:, 1, 0, 0, 0].tolist(),
            )
        )
        assert pairs == expected_pairs
