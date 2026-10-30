"""Dependency-light checks for the bounded synthetic laptop sender."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


_HERE = Path(__file__).resolve().parent
_INFRA = _HERE.parent
_SCRIPT = _INFRA / "scripts" / "run_fake_e2e_actor.py"
sys.path.insert(0, str(_INFRA))

_SPEC = importlib.util.spec_from_file_location("run_fake_e2e_actor_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _FakeNetwork:
    instances = []

    def __init__(self, target, **kwargs):
        self.target = target
        self.kwargs = kwargs
        self.closed = False
        self.begin_calls = []
        self.step_calls = []
        self._status_calls = 0
        self.instances.append(self)

    def health(self):
        return True, True, "ok"

    def get_buffer_status(self):
        self._status_calls += 1
        insert_count = 7 if self._status_calls == 1 else 9
        return SimpleNamespace(replay_insert_count=insert_count, replay_size=9)

    def begin_episode(self, observation, **kwargs):
        self.begin_calls.append((observation, kwargs))
        return SimpleNamespace(
            action=np.zeros(7, dtype=np.float32),
            policy_version=3,
            round_trip_ms=2.5,
        )

    def step(self, next_observation, **kwargs):
        self.step_calls.append((next_observation, kwargs))
        transition_id = kwargs["data"]["meta"]["transition_id"]
        return SimpleNamespace(
            ack=SimpleNamespace(accepted=True, transition_id=transition_id),
            outcome=SimpleNamespace(terminal=True, success=False),
            action=None,
        )

    def close(self):
        self.closed = True


def _args(*extra: str):
    return _MODULE._parse_args(
        [
            "--expected-start-policy-version",
            "3",
            "--expected-reward-model-id",
            "fake-classifier",
            "--transition-count",
            "2",
            "--run-id",
            "acceptance-run",
            *extra,
        ]
    )


def test_fake_sender_uses_real_contract_shape_and_requires_ack(monkeypatch):
    _FakeNetwork.instances.clear()
    monkeypatch.setattr(_MODULE, "GrpcActorNetwork", _FakeNetwork)

    result = _MODULE.run(_args())
    network = _FakeNetwork.instances[-1]

    assert result["event"] == "fake_e2e_actor_passed"
    assert result["transition_count"] == 2
    assert result["replay_insert_delta"] == 2
    assert result["first_policy_version"] == 3
    assert result["last_policy_version"] == 3
    assert network.kwargs["expected_reward_authority"] == "server_classifier"
    assert network.kwargs["expected_model_id"].endswith("synthetic-e2e-v1")
    assert len(network.begin_calls) == len(network.step_calls) == 2
    assert network.closed
    first_observation = network.begin_calls[0][0]
    assert first_observation["state"].shape == (1, 19)
    assert first_observation["cam1"].shape == (1, 128, 128, 3)
    assert first_observation["cam1"].dtype == np.uint8
    first_data = network.step_calls[0][1]["data"]
    assert first_data["meta"]["auto_success"] is True
    assert first_data["meta"]["operator_success"] is False
    assert first_data["transition"]["grasp_penalty"] == pytest.approx(-0.02)
    assert first_data["transition"]["dones"] is True
    assert network.step_calls[0][1]["request_action"] is False


def test_fake_sender_rejects_policy_lineage_mismatch_and_closes(monkeypatch):
    _FakeNetwork.instances.clear()
    monkeypatch.setattr(_MODULE, "GrpcActorNetwork", _FakeNetwork)

    with pytest.raises(RuntimeError, match="expected 4"):
        _MODULE.run(
            _MODULE._parse_args(
                [
                    "--expected-start-policy-version",
                    "4",
                    "--expected-reward-model-id",
                    "fake-classifier",
                    "--transition-count",
                    "1",
                ]
            )
        )
    assert _FakeNetwork.instances[-1].closed


@pytest.mark.parametrize(
    "extra, match",
    [
        (("--transition-count", "0"), "transition_count"),
        (("--grasp-penalty", "0.1"), "grasp_penalty"),
        (("--timeout-s", "nan"), "timeout_s"),
    ],
)
def test_fake_sender_rejects_invalid_acceptance_arguments(extra, match):
    with pytest.raises(ValueError, match=match):
        _MODULE._validate_args(_args(*extra))
