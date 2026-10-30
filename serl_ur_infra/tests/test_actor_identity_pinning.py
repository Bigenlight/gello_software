"""Fail-closed actor checks for production server/model identity pinning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


_HERE = Path(__file__).resolve().parent
_INFRA = _HERE.parent
sys.path.insert(0, str(_INFRA))

from ur_env.actor_network import (  # noqa: E402
    ActorProtocolError,
    ActorSessionService,
    ActorTransportError,
)
from ur_env.grpc_actor_transport import (  # noqa: E402
    GrpcActorNetwork,
    create_grpc_server,
)


def _observation() -> dict[str, np.ndarray]:
    return {
        "state": np.zeros((1, 19), np.float32),
        "cam1": np.zeros((1, 128, 128, 3), np.uint8),
        "cam2": np.zeros((1, 128, 128, 3), np.uint8),
    }


def _begin(client: GrpcActorNetwork, *, episode_id: int = 0) -> None:
    client.begin_episode(
        _observation(),
        run_id="run",
        session_id=f"session-{episode_id}",
        episode_id=episode_id,
        observation_id=f"observation-{episode_id}",
        timestamp_ns=1_000 + episode_id,
    )


@pytest.mark.parametrize(
    ("config_key", "unexpected_value", "error_field"),
    (
        ("expected_model_id", "other-policy", "model_id"),
        ("expected_reward_authority", "robot", "reward_authority"),
        ("expected_reward_model_id", "other-reward", "reward_model_id"),
    ),
)
def test_from_config_pins_identity_before_begin_episode(
    config_key: str,
    unexpected_value: str,
    error_field: str,
) -> None:
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        model_id="policy-v7",
        reward_authority="server",
        reward_model_id="reward-v3",
    )
    server, port = create_grpc_server(service)
    server.start()
    config = {
        "host": "127.0.0.1",
        "port": port,
        "timeout_s": 1.0,
        config_key: unexpected_value,
    }
    client = GrpcActorNetwork.from_config(
        config,
        actor_id="actor",
        action_shape=(7,),
    )
    try:
        with pytest.raises(ActorProtocolError, match=error_field):
            _begin(client)
        assert service.inference_count == 0
        assert service.observation_accept_count == 0
    finally:
        client.close()
        server.stop(grace=0).wait()


def test_all_matching_pins_are_rechecked_at_each_episode_boundary() -> None:
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        model_id="policy-v7",
        reward_authority="server",
        reward_model_id="reward-v3",
        observation_schema_hash="schema-v2",
    )
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}",
        actor_id="actor",
        timeout_s=1.0,
        expected_model_id="policy-v7",
        expected_reward_authority="server",
        expected_reward_model_id="reward-v3",
        expected_observation_schema_hash="schema-v2",
    )
    try:
        _begin(client)
        assert service.inference_count == 1

        # Simulate a different server process appearing at the same endpoint.
        service._model_id = "policy-v8"
        with pytest.raises(ActorProtocolError, match="model_id"):
            _begin(client, episode_id=1)
        assert service.inference_count == 1
        assert service.observation_accept_count == 1
    finally:
        client.close()
        server.stop(grace=0).wait()


@pytest.mark.parametrize(
    "keyword",
    (
        "expected_model_id",
        "expected_reward_authority",
        "expected_reward_model_id",
    ),
)
def test_empty_identity_pin_is_rejected(keyword: str) -> None:
    with pytest.raises(ValueError, match=keyword):
        GrpcActorNetwork("unused", actor_id="actor", **{keyword: ""})


@pytest.mark.parametrize(
    ("actor_id", "run_id", "detail"),
    (
        ("robot-actor", "fake-run", "actor_id"),
        ("fake-actor", "robot-run", "run_id"),
    ),
)
def test_server_allowlists_gate_before_policy_inference(
    actor_id: str,
    run_id: str,
    detail: str,
) -> None:
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        allowed_actor_ids=("fake-actor",),
        allowed_run_ids=("fake-run",),
    )
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}", actor_id=actor_id, timeout_s=1.0
    )
    try:
        with pytest.raises(ActorTransportError, match=detail):
            client.begin_episode(
                _observation(),
                run_id=run_id,
                session_id="session",
                episode_id=0,
                observation_id="observation",
                timestamp_ns=1_000,
            )
        assert service.inference_count == 0
        assert service.health() == (True, True, "ready")
    finally:
        client.close()
        server.stop(grace=0).wait()


def _load_actor_script():
    path = _INFRA / "scripts" / "run_remote_rlpd_actor.py"
    spec = importlib.util.spec_from_file_location(
        "test_run_remote_rlpd_actor_identity", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actor_cli_parses_and_overrides_identity_pins(monkeypatch) -> None:
    module = _load_actor_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_remote_rlpd_actor.py",
            "--exp-name",
            "task",
            "--expected-model-id",
            "cli-policy",
            "--expected-reward-authority",
            "server",
            "--expected-reward-model-id",
            "cli-reward",
        ],
    )

    args = module._parse_args()
    config = SimpleNamespace(
        NETWORK={
            "expected_model_id": "configured-policy",
            "expected_reward_authority": "robot",
            "expected_reward_model_id": "configured-reward",
        }
    )
    network = module._network_config(config, args)

    assert network["expected_model_id"] == "cli-policy"
    assert network["expected_reward_authority"] == "server"
    assert network["expected_reward_model_id"] == "cli-reward"


def test_actor_cli_leaves_identity_pins_optional(monkeypatch) -> None:
    module = _load_actor_script()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_remote_rlpd_actor.py", "--exp-name", "task"],
    )

    network = module._network_config(
        SimpleNamespace(NETWORK={}), module._parse_args()
    )

    assert "expected_model_id" not in network
    assert "expected_reward_authority" not in network
    assert "expected_reward_model_id" not in network
