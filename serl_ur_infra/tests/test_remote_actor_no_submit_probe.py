"""No-arm actor probe: real observation/inference without replay pollution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


_HERE = Path(__file__).resolve().parent
_INFRA = _HERE.parent
sys.path.insert(0, str(_INFRA))

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ActionResult,
    ActorProtocolError,
    ActorSessionService,
    ServerInfo,
)
from ur_env.actor_smoke import synthetic_observation  # noqa: E402
from ur_env.grpc_actor_transport import (  # noqa: E402
    GrpcActorNetwork,
    create_grpc_server,
)
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
    validate_canonical_observation,
)
from ur_env.remote_actor import run_remote_actor_probe  # noqa: E402


class _ProbeEnv:
    action_space = SimpleNamespace(shape=(7,))

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False

    def reset(self):
        self.reset_calls += 1
        return synthetic_observation(0), {"timestamp_ns": np.int64(12_345)}

    def step(self, action):
        del action
        self.step_calls += 1
        raise AssertionError("no-submit probe must never call env.step")

    def close(self):
        self.closed = True


def _server_info() -> ServerInfo:
    return ServerInfo(
        ready=True,
        protocol_version=PROTOCOL_VERSION,
        schema_version=SCHEMA_VERSION,
        action_dim=7,
        model_id="probe-policy",
        reward_authority="server_classifier",
        reward_model_id="probe-reward",
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
    )


def test_real_grpc_probe_runs_inference_but_inserts_no_replay_transition():
    seen_observations = []
    expected_action = np.array(
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0], dtype=np.float32
    )

    def sample_action(observation, deterministic):
        assert deterministic is False
        seen_observations.append(validate_canonical_observation(observation))
        return expected_action, 7

    service = ActorSessionService(
        sample_action,
        model_id="probe-policy",
        reward_authority="server_classifier",
        reward_model_id="probe-reward",
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
    )
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}",
        actor_id="probe-actor",
        timeout_s=1.0,
        max_response_age_s=2.0,
        expected_observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        expected_model_id="probe-policy",
        expected_reward_authority="server_classifier",
        expected_reward_model_id="probe-reward",
    )
    env = _ProbeEnv()
    try:
        before = service.get_buffer_status()
        summary = run_remote_actor_probe(
            client,
            env,
            actor_id="probe-actor",
            run_id="no-submit-run",
            session_id_factory=lambda: "no-submit-session",
        )
        after = service.get_buffer_status()
    finally:
        client.close()
        server.stop(grace=0).wait()

    assert env.reset_calls == 1
    assert env.step_calls == 0
    assert len(seen_observations) == 1
    assert service.observation_accept_count == 1
    assert service.inference_count == 1
    assert before.replay_insert_count == after.replay_insert_count == 0
    assert before.intervention_insert_count == after.intervention_insert_count == 0
    assert service.replay_items == []
    assert service.intervention_items == []
    assert summary.run_id == "no-submit-run"
    assert summary.session_id == "no-submit-session"
    assert summary.policy_version == 7
    assert summary.server_info.observation_schema_hash == (
        CANONICAL_OBSERVATION_SCHEMA_HASH
    )
    np.testing.assert_array_equal(summary.policy_action, expected_action)


class _InvalidActionNetwork:
    def __init__(self, action) -> None:
        self.action = action
        self.info_calls = 0
        self.begin_calls = 0
        self.step_calls = 0

    def get_server_info(self):
        self.info_calls += 1
        return _server_info()

    def begin_episode(self, observation, **kwargs):
        del observation
        self.begin_calls += 1
        return ActionResult(
            action=self.action,
            policy_version=3,
            session_id=kwargs["session_id"],
            request_id=1,
            request_created_monotonic_ns=99,
            observation_id=kwargs["observation_id"],
            server_inference_ms=1.0,
            round_trip_ms=2.0,
        )

    def step(self, *args, **kwargs):
        del args, kwargs
        self.step_calls += 1
        raise AssertionError("no-submit probe must never call network.step")


@pytest.mark.parametrize(
    "action, match",
    [
        (np.zeros(6, dtype=np.float32), r"shape \(7,\)"),
        (np.full(7, 1.01, dtype=np.float32), r"within \[-1, 1\]"),
    ],
)
def test_probe_rejects_policy_action_with_bad_shape_or_range(action, match):
    network = _InvalidActionNetwork(action)
    env = _ProbeEnv()

    with pytest.raises(ActorProtocolError, match=match):
        run_remote_actor_probe(
            network,
            env,
            actor_id="probe-actor",
            run_id="invalid-action-run",
            session_id_factory=lambda: "invalid-action-session",
        )

    assert network.info_calls == 1
    assert network.begin_calls == 1
    assert network.step_calls == 0
    assert env.reset_calls == 1
    assert env.step_calls == 0


def _load_actor_script():
    path = _INFRA / "scripts" / "run_remote_rlpd_actor.py"
    spec = importlib.util.spec_from_file_location(
        "test_run_remote_rlpd_actor_no_submit", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_arm_main_routes_only_to_probe_and_closes_resources(monkeypatch):
    module = _load_actor_script()
    args = SimpleNamespace(
        exp_name="task",
        ur_config_module=None,
        arm=False,
        fake_env=False,
        mock_policy_noise=0.0,
        actor_id="probe-actor",
    )
    robot_config = SimpleNamespace(DRY_RUN=False)
    config = SimpleNamespace(robot_config=robot_config)
    env = _ProbeEnv()

    class _Network:
        def __init__(self):
            self.closed = False

        def health(self):
            return True, True, "ready"

        def close(self):
            self.closed = True

    network = _Network()
    probe_calls = []

    def probe(network_arg, env_arg, **kwargs):
        probe_calls.append((network_arg, env_arg, kwargs))
        return SimpleNamespace(
            server_info=_server_info(),
            run_id="probe-run",
            policy_version=4,
            policy_action=np.zeros(7, dtype=np.float32),
            server_inference_ms=1.0,
            round_trip_ms=2.0,
        )

    monkeypatch.setattr(module, "_parse_args", lambda: args)
    monkeypatch.setattr(
        module, "_load_config_mapping", lambda unused: {"task": lambda: config}
    )
    monkeypatch.setattr(module, "_build_actor_environment", lambda *_: env)
    monkeypatch.setattr(module, "_network_config", lambda *_: {"type": "grpc"})
    monkeypatch.setattr(module, "create_actor_network", lambda *a, **k: network)
    monkeypatch.setattr(module, "run_remote_actor_probe", probe)
    monkeypatch.setattr(
        module,
        "_mock_policy_transform",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("no-arm probe transformed the server policy action")
        ),
    )
    monkeypatch.setattr(
        module,
        "_build_sidecar_scheduler",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("no-arm probe entered transition sidecar setup")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_remote_actor",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("no-arm main entered production actor loop")
        ),
    )

    assert module.main() == 0

    assert robot_config.DRY_RUN is True
    assert probe_calls == [
        (network, env, {"actor_id": "probe-actor"})
    ]
    assert env.step_calls == 0
    assert env.closed is True
    assert network.closed is True


def test_arm_rejects_fake_environment_before_construction(monkeypatch):
    module = _load_actor_script()
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            arm=True,
            fake_env=True,
            mock_policy_noise=0.0,
        ),
    )

    with pytest.raises(SystemExit, match="cannot be combined"):
        module.main()


def test_no_arm_rejects_mock_action_rewrite(monkeypatch):
    module = _load_actor_script()
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            arm=False,
            fake_env=False,
            mock_policy_noise=0.01,
        ),
    )

    with pytest.raises(SystemExit, match="requires --arm"):
        module.main()
