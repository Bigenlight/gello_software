"""ROS/JAX-free contract tests for the remote actor transport."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    ActorNetworkError,
    ActorProtocolError,
    ActorSessionService,
    BeginEpisodeCommand,
    BufferStatus,
    FailedPreconditionError,
    ObservationPacket,
    PolicyInferenceError,
    StepCommand,
    TransitionOutcome,
)
from ur_env.classifier_sidecar import (  # noqa: E402
    CLASSIFIER_SIDECAR_KEY,
    SIDECAR_TENSOR_KEYS,
    build_sidecar,
)
from ur_env.grpc_actor_transport import (  # noqa: E402
    GrpcActorNetwork,
    create_grpc_server,
    data_from_proto,
    data_to_proto,
    observation_from_proto,
    observation_to_proto,
)
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
    validate_canonical_observation,
)
from ur_env.proto import actor_transport_pb2 as pb  # noqa: E402
from ur_env.remote_actor import (  # noqa: E402
    EnvTimestampAdapter,
    build_data,
    run_remote_actor,
)


def _observation(value: int) -> dict:
    return {
        "state": {
            "q": np.array([value, value + 0.5], dtype=np.float32),
            "flags": np.array([True, False], dtype=np.bool_),
        },
        "images": {
            "wrist": np.full((4, 5, 3), value, dtype=np.uint8),
            "side": np.full((2, 3, 3), value + 1, dtype=np.uint8),
        },
    }


def _canonical_observation(value: int) -> dict:
    """Exactly the three keys validate_canonical_observation permits."""
    return {
        "state": np.full((1, 19), value / 100.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _sidecar(seed: int = 0) -> dict:
    # Full-resolution-shaped, uncropped BGR frames: build_sidecar owns the
    # downscale to the classifier's 128x128, nothing here does.
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, seed % 3] = (seed * 11) % 256
    return build_sidecar({"cam1": frame, "cam2": frame})


def _observation_with_sidecar(value: int) -> dict:
    observation = _canonical_observation(value)
    observation[CLASSIFIER_SIDECAR_KEY] = _sidecar(value)
    return observation


def _assert_observation_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], dict):
            _assert_observation_equal(actual[key], expected[key])
        else:
            assert actual[key].dtype == expected[key].dtype
            assert actual[key].shape == expected[key].shape
            np.testing.assert_array_equal(actual[key], expected[key])


def _data(
    action,
    *,
    env_step=0,
    step_id=0,
    source_timestamp_ns=1_000,
    next_observation_id="o1",
    policy_version=0,
    intervened=False,
    terminal=False,
    truncated=False,
    auto_success=True,
    operator_success=False,
):
    info = {}
    if intervened:
        info = {
            "intervened": 1,
            "intervene_action": np.full(7, -0.25, dtype=np.float32),
            "grasp_penalty": -0.1,
        }
    data = build_data(
        actor_id="actor",
        run_id="run",
        session_id="session",
        transition_id=f"run:{env_step}",
        env_step=env_step,
        timestamp_ns=source_timestamp_ns,
        policy_version=policy_version,
        policy_action=action,
        episode_id=0,
        step_id=step_id,
        observation_id=f"o{step_id}",
        next_observation_id=next_observation_id,
        reward=1.25,
        done=terminal,
        truncated=truncated,
        info=info,
    )
    data["meta"]["auto_success"] = bool(auto_success)
    data["meta"]["operator_success"] = bool(operator_success)
    return data


def test_observation_codec_is_lossless_for_images_and_state():
    expected = _observation(3)
    packet = ObservationPacket("observation-3", 123_456_789, expected)

    decoded = observation_from_proto(observation_to_proto(packet))

    assert decoded.observation_id == packet.observation_id
    assert decoded.timestamp_ns == packet.timestamp_ns
    _assert_observation_equal(decoded.observation, expected)


def test_data_codec_keeps_meta_transition_and_optional_grasp_penalty():
    action = np.linspace(-0.5, 0.5, 7, dtype=np.float32)
    expected = _data(action, intervened=True)

    decoded = data_from_proto(data_to_proto(expected))

    assert set(decoded) == {"meta", "transition"}
    assert decoded["meta"]["timestamp_ns"] == 1_000
    assert decoded["meta"]["intervened"] == 1
    assert decoded["meta"]["auto_success"] is True
    assert decoded["meta"]["operator_success"] is False
    np.testing.assert_array_equal(decoded["meta"]["policy_action"], action)
    np.testing.assert_array_equal(
        decoded["transition"]["actions"], np.full(7, -0.25, np.float32)
    )
    assert decoded["transition"]["grasp_penalty"] == pytest.approx(-0.1)


def test_service_combines_inference_replay_routing_terminal_and_dedupe():
    calls = []

    def sample(observation, deterministic):
        calls.append((observation, deterministic))
        value = 0.1 * len(calls)
        return np.full(7, value, dtype=np.float32), len(calls) - 1

    service = ActorSessionService(sample)
    begin = BeginEpisodeCommand(
        PROTOCOL_VERSION,
        "actor",
        "run",
        "session",
        0,
        1,
        10_000,
        ObservationPacket("o0", 1_000, _observation(0)),
    )
    action0 = service.begin_episode(begin)
    data0 = _data(action0.action)
    step0 = StepCommand(
        PROTOCOL_VERSION,
        "actor",
        "run",
        "session",
        2,
        20_000,
        data0,
        ObservationPacket("o1", 2_000, _observation(1)),
        True,
    )

    reply0 = service.step(step0)
    duplicate = service.step(step0)

    assert reply0.ack.accepted
    assert duplicate.ack.deduplicated
    assert len(service.replay_items) == 1
    assert len(service.intervention_items) == 0
    assert service.observation_accept_count == 2
    assert service.inference_count == 2
    assert len(calls) == 2
    assert service.replay_items[0]["meta"]["timestamp_ns"] == 1_000
    _assert_observation_equal(
        service.replay_items[0]["transition"]["observations"], _observation(0)
    )

    data1 = _data(
        reply0.action.action,
        env_step=1,
        step_id=1,
        source_timestamp_ns=2_000,
        next_observation_id="o2",
        policy_version=reply0.action.policy_version,
        intervened=True,
        terminal=True,
    )
    terminal = service.step(
        StepCommand(
            PROTOCOL_VERSION,
            "actor",
            "run",
            "session",
            3,
            30_000,
            data1,
            ObservationPacket("o2", 3_000, _observation(2)),
            False,
        )
    )

    assert terminal.ack.accepted
    assert terminal.action is None
    assert len(service.replay_items) == 2
    assert len(service.intervention_items) == 1
    assert service.inference_count == 2  # terminal O2 is stored, not inferred


def test_server_finalizer_success_suppresses_requested_action_and_dedupes():
    finalize_calls = []
    sink_calls = []

    def finalize(data, classifier_sidecar):
        finalize_calls.append((data["meta"]["transition_id"], classifier_sidecar))
        transition = data["transition"]
        transition.update(
            rewards=1.0,
            masks=0.0,
            dones=True,
            truncated=False,
        )
        return data, TransitionOutcome(
            transition_id=data["meta"]["transition_id"],
            reward=1.0,
            mask=0.0,
            done=True,
            truncated=False,
            success=True,
            classifier_evaluated=True,
            classifier_probability=0.9,
            classifier_threshold=0.85,
            reward_model_id="cube-in-cup:150",
        )

    def accept(data, intervened):
        sink_calls.append((data, intervened))

    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        reward_authority="server",
        reward_model_id="cube-in-cup:150",
        finalize_transition=finalize,
        accept_data=accept,
    )
    action = service.begin_episode(
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
    command = StepCommand(
        PROTOCOL_VERSION,
        "actor",
        "run",
        "session",
        2,
        20_000,
        _data(action.action),
        ObservationPacket("o1", 2_000, _observation(1)),
        True,
    )

    result = service.step(command)
    duplicate = service.step(command)

    assert result.outcome.success
    assert result.outcome.terminal
    assert result.action is None
    assert duplicate.ack.deduplicated
    assert len(finalize_calls) == 1
    assert finalize_calls[0][1] is None  # no sidecar was attached to this step
    assert len(sink_calls) == 1
    assert sink_calls[0][0]["transition"]["rewards"] == 1.0
    assert service.inference_count == 1


def test_step_strips_the_classifier_sidecar_before_anything_else_sees_it():
    """The reserved key never reaches the policy, replay, or canonical checks."""
    seen_by_policy = []
    seen_by_finalizer = []
    accepted = []

    def sample(observation, deterministic):
        seen_by_policy.append(observation)
        return np.zeros(7, np.float32), 0

    def finalize(data, classifier_sidecar):
        seen_by_finalizer.append(classifier_sidecar)
        transition = data["transition"]
        return data, TransitionOutcome(
            transition_id=data["meta"]["transition_id"],
            reward=float(transition["rewards"]),
            mask=float(transition["masks"]),
            done=bool(transition["dones"]),
            truncated=bool(transition["truncated"]),
            success=False,
            classifier_evaluated=False,
        )

    service = ActorSessionService(
        sample,
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        finalize_transition=finalize,
        accept_data=lambda data, intervened: accepted.append(data),
    )
    action = service.begin_episode(
        BeginEpisodeCommand(
            PROTOCOL_VERSION,
            "actor",
            "run",
            "session",
            0,
            1,
            10_000,
            ObservationPacket("o0", 1_000, _canonical_observation(0)),
        )
    )
    result = service.step(
        StepCommand(
            PROTOCOL_VERSION,
            "actor",
            "run",
            "session",
            2,
            20_000,
            _data(action.action),
            ObservationPacket("o1", 2_000, _observation_with_sidecar(1)),
            True,
        )
    )

    assert result.ack.accepted
    # The finalizer got the validated JPEG tensors, and only the finalizer.
    assert len(seen_by_finalizer) == 1
    assert set(seen_by_finalizer[0]) == set(SIDECAR_TENSOR_KEYS)
    assert seen_by_finalizer[0]["cam1_jpeg"].dtype == np.uint8
    # The policy and replay both see the untouched canonical three-key tree,
    # which is what lets validate_canonical_observation stay exact-key strict.
    for observation in seen_by_policy:
        assert CLASSIFIER_SIDECAR_KEY not in observation
        validate_canonical_observation(observation)
    stored = accepted[0]["transition"]["next_observations"]
    assert CLASSIFIER_SIDECAR_KEY not in stored
    validate_canonical_observation(stored)
    # And the sidecar changes nothing about the schema both peers handshake on.
    assert (
        service.get_server_info().observation_schema_hash
        == CANONICAL_OBSERVATION_SCHEMA_HASH
    )


def test_begin_episode_rejects_a_classifier_sidecar():
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0)
    )
    command = BeginEpisodeCommand(
        PROTOCOL_VERSION,
        "actor",
        "run",
        "session",
        0,
        1,
        10_000,
        ObservationPacket("o0", 1_000, _observation_with_sidecar(0)),
    )

    with pytest.raises(ActorProtocolError, match="reserved 'classifier'"):
        service.begin_episode(command)
    assert service.inference_count == 0


def test_malformed_sidecar_is_a_protocol_error_not_an_internal_fault():
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0)
    )
    action = service.begin_episode(
        BeginEpisodeCommand(
            PROTOCOL_VERSION,
            "actor",
            "run",
            "session",
            0,
            1,
            10_000,
            ObservationPacket("o0", 1_000, _canonical_observation(0)),
        )
    )
    broken = _canonical_observation(1)
    broken[CLASSIFIER_SIDECAR_KEY] = {"cam1_jpeg": np.zeros(4, dtype=np.uint8)}
    command = StepCommand(
        PROTOCOL_VERSION,
        "actor",
        "run",
        "session",
        2,
        20_000,
        _data(action.action),
        ObservationPacket("o1", 2_000, broken),
        True,
    )

    with pytest.raises(ActorProtocolError, match="invalid 'classifier' sidecar"):
        service.step(command)
    # A rejected request must not latch the service into a fault state.
    assert service.health()[1] is True


def test_legacy_single_argument_finalizer_fails_closed_on_a_sidecar():
    """Receive-only harnesses keep working, but may not drop a classification."""

    def legacy_finalize(data):
        transition = data["transition"]
        return data, TransitionOutcome(
            transition_id=data["meta"]["transition_id"],
            reward=float(transition["rewards"]),
            mask=float(transition["masks"]),
            done=bool(transition["dones"]),
            truncated=bool(transition["truncated"]),
            success=False,
            classifier_evaluated=False,
        )

    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        finalize_transition=legacy_finalize,
    )
    action = service.begin_episode(
        BeginEpisodeCommand(
            PROTOCOL_VERSION,
            "actor",
            "run",
            "session",
            0,
            1,
            10_000,
            ObservationPacket("o0", 1_000, _canonical_observation(0)),
        )
    )

    # Without a sidecar the legacy callback is still honoured...
    plain = service.step(
        StepCommand(
            PROTOCOL_VERSION,
            "actor",
            "run",
            "session",
            2,
            20_000,
            _data(action.action),
            ObservationPacket("o1", 2_000, _canonical_observation(1)),
            True,
        )
    )
    assert plain.ack.accepted

    # ...but a sidecar it cannot consume stops the transition instead of being
    # silently discarded.
    with pytest.raises(ActorNetworkError, match="predates the classifier"):
        service.step(
            StepCommand(
                PROTOCOL_VERSION,
                "actor",
                "run",
                "session",
                3,
                30_000,
                _data(
                    plain.action.action,
                    env_step=1,
                    step_id=1,
                    source_timestamp_ns=2_000,
                    next_observation_id="o2",
                ),
                ObservationPacket("o2", 3_000, _observation_with_sidecar(2)),
                True,
            )
        )


def test_sidecar_crosses_the_existing_named_tensor_map_unchanged():
    """No proto change: the sidecar is just two more paths in the tensor map."""
    expected = _observation_with_sidecar(7)
    packet = ObservationPacket("o-side", 123_456_789, expected)

    message = observation_to_proto(packet)
    decoded = observation_from_proto(message)

    paths = {"/".join(tensor.path) for tensor in message.tensors}
    assert paths == {
        "state",
        "cam1",
        "cam2",
        f"{CLASSIFIER_SIDECAR_KEY}/cam1_jpeg",
        f"{CLASSIFIER_SIDECAR_KEY}/cam2_jpeg",
    }
    _assert_observation_equal(decoded.observation, expected)
    # Byte-for-byte identical JPEGs: the actor re-encodes nothing, so the server
    # decodes exactly what the live viewer decoded on real hardware.
    np.testing.assert_array_equal(
        decoded.observation[CLASSIFIER_SIDECAR_KEY]["cam1_jpeg"],
        expected[CLASSIFIER_SIDECAR_KEY]["cam1_jpeg"],
    )


def test_grpc_round_trip_delivers_the_sidecar_to_the_finalizer_only():
    received = []

    def finalize(data, classifier_sidecar):
        received.append(classifier_sidecar)
        transition = data["transition"]
        return data, TransitionOutcome(
            transition_id=data["meta"]["transition_id"],
            reward=float(transition["rewards"]),
            mask=float(transition["masks"]),
            done=bool(transition["dones"]),
            truncated=bool(transition["truncated"]),
            success=False,
            classifier_evaluated=False,
        )

    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        finalize_transition=finalize,
    )
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}", actor_id="actor", timeout_s=5.0
    )
    try:
        action = client.begin_episode(
            _canonical_observation(0),
            run_id="run",
            session_id="session",
            episode_id=0,
            observation_id="o0",
            timestamp_ns=1_000,
        )
        sent = _observation_with_sidecar(1)
        result = client.step(
            sent,
            next_observation_id="o1",
            next_timestamp_ns=2_000,
            data=_data(action.action),
            request_action=True,
        )

        assert result.ack.accepted
        assert result.outcome.classifier_evaluated is False
        assert len(received) == 1
        np.testing.assert_array_equal(
            received[0]["cam2_jpeg"],
            sent[CLASSIFIER_SIDECAR_KEY]["cam2_jpeg"],
        )
        assert CLASSIFIER_SIDECAR_KEY not in (
            service.replay_items[0]["transition"]["next_observations"]
        )
    finally:
        client.close()
        server.stop(grace=0).wait()


def test_transition_pipeline_failure_has_no_ack_and_marks_service_not_ready():
    def fail_finalizer(data):
        raise RuntimeError("classifier unavailable")

    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        finalize_transition=fail_finalizer,
    )
    action = service.begin_episode(
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
    command = StepCommand(
        PROTOCOL_VERSION,
        "actor",
        "run",
        "session",
        2,
        20_000,
        _data(action.action),
        ObservationPacket("o1", 2_000, _observation(1)),
        True,
    )

    with pytest.raises(ActorNetworkError, match="not acknowledged"):
        service.step(command)
    assert service.health()[1] is False
    assert service.replay_items == []
    with pytest.raises(FailedPreconditionError, match="not ready"):
        service.step(command)


def test_buffer_status_provider_is_exposed_without_raw_data():
    expected = BufferStatus(
        replay_size=17,
        replay_capacity=50_000,
        intervention_size=3,
        intervention_capacity=10_000,
        replay_insert_count=19,
        intervention_insert_count=3,
        replay_overwrite_count=2,
        intervention_overwrite_count=0,
        last_transition_id="run:18",
        last_env_step=18,
    )
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        buffer_status_provider=lambda: expected,
    )

    assert service.get_buffer_status() == expected


def test_duplicate_id_with_changed_payload_is_rejected():
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0)
    )
    begin = BeginEpisodeCommand(
        PROTOCOL_VERSION,
        "actor",
        "run",
        "session",
        0,
        1,
        10_000,
        ObservationPacket("o0", 1_000, _observation(0)),
    )
    action = service.begin_episode(begin)
    command = StepCommand(
        PROTOCOL_VERSION,
        "actor",
        "run",
        "session",
        2,
        20_000,
        _data(action.action),
        ObservationPacket("o1", 2_000, _observation(1)),
        True,
    )
    service.step(command)
    changed = StepCommand(
        command.protocol_version,
        command.actor_id,
        command.run_id,
        command.session_id,
        command.request_id,
        command.created_monotonic_ns,
        {**command.data, "transition": {**command.data["transition"], "rewards": 99.0}},
        command.next_observation,
        command.request_action,
    )

    with pytest.raises(ActorProtocolError, match="different content"):
        service.step(changed)


def test_service_rejects_out_of_range_policy_action():
    service = ActorSessionService(
        lambda observation, deterministic: (np.full(7, 1.01, np.float32), 0)
    )
    command = BeginEpisodeCommand(
        PROTOCOL_VERSION,
        "actor",
        "run",
        "session",
        0,
        1,
        10_000,
        ObservationPacket("o0", 1_000, _observation(0)),
    )

    with pytest.raises(PolicyInferenceError, match=r"\[-1, 1\]"):
        service.begin_episode(command)
    assert service.health()[1] is False


def test_mock_store_rejects_at_capacity_without_second_ack():
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        in_memory_capacity=1,
    )
    action = service.begin_episode(
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
    first = service.step(
        StepCommand(
            PROTOCOL_VERSION,
            "actor",
            "run",
            "session",
            2,
            20_000,
            _data(action.action),
            ObservationPacket("o1", 2_000, _observation(1)),
            True,
        )
    )
    with pytest.raises(RuntimeError, match="capacity"):
        service.step(
            StepCommand(
                PROTOCOL_VERSION,
                "actor",
                "run",
                "session",
                3,
                30_000,
                _data(
                    first.action.action,
                    env_step=1,
                    step_id=1,
                    source_timestamp_ns=2_000,
                    next_observation_id="o2",
                    policy_version=first.action.policy_version,
                ),
                ObservationPacket("o2", 3_000, _observation(2)),
                True,
            )
        )
    assert len(service.replay_items) == 1


class _InProcessNetwork:
    def __init__(self, service, actor_id="actor"):
        self.service = service
        self.actor_id = actor_id
        self.run_id = ""
        self.session_id = ""
        self.request_id = 1
        self.clock = 10_000

    def begin_episode(
        self,
        observation,
        *,
        run_id,
        session_id,
        episode_id,
        observation_id,
        timestamp_ns,
        deterministic=False,
    ):
        self.run_id = run_id
        self.session_id = session_id
        self.request_id = 2
        self.clock += 1
        return self.service.begin_episode(
            BeginEpisodeCommand(
                PROTOCOL_VERSION,
                self.actor_id,
                run_id,
                session_id,
                episode_id,
                1,
                self.clock,
                ObservationPacket(observation_id, timestamp_ns, observation),
                deterministic,
            )
        )

    def step(
        self,
        next_observation,
        *,
        next_observation_id,
        next_timestamp_ns,
        data,
        request_action,
        deterministic=False,
    ):
        self.clock += 1
        result = self.service.step(
            StepCommand(
                PROTOCOL_VERSION,
                self.actor_id,
                self.run_id,
                self.session_id,
                self.request_id,
                self.clock,
                data,
                ObservationPacket(
                    next_observation_id, next_timestamp_ns, next_observation
                ),
                request_action,
                deterministic,
            )
        )
        self.request_id += 1
        return result


class _ActionSpace:
    shape = (7,)


class _TwoStepEnv:
    action_space = _ActionSpace()

    def __init__(self):
        self.reset_count = 0
        self.step_count = 0

    def reset(self):
        self.reset_count += 1
        return _observation(0), {"timestamp_ns": np.int64(1_000)}

    def step(self, action):
        self.step_count += 1
        if self.step_count == 1:
            return _observation(1), 0.0, False, False, {
                "timestamp_ns": np.int64(2_000),
                "intervened": 0,
            }
        return _observation(2), 1.0, True, False, {
            "timestamp_ns": np.int64(3_000),
            "intervened": 1,
            "intervene_action": np.full(7, -0.5, np.float32),
        }


def test_actor_uses_source_timestamp_and_waits_for_terminal_ack_before_reset():
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0)
    )
    network = _InProcessNetwork(service)
    env = _TwoStepEnv()

    summary = run_remote_actor(
        network,
        env,
        config=SimpleNamespace(max_steps=2, random_steps=0, buffer_period=0),
        actor_id="actor",
        run_id="run",
        session_id_factory=lambda: "session",
    )

    assert summary.env_steps == 2
    assert summary.intervention_steps == 1
    assert env.reset_count == 1  # final terminal ACK did not start an unused episode
    assert [item["meta"]["env_step"] for item in service.replay_items] == [0, 1]
    assert [item["transition"]["step_id"] for item in service.replay_items] == [0, 1]
    assert [item["meta"]["timestamp_ns"] for item in service.replay_items] == [
        1_000,
        2_000,
    ]
    assert len(service.intervention_items) == 1


def test_timestamp_adapter_preserves_env_timestamp_and_stamps_missing_info():
    class Env:
        def reset(self):
            return {}, {"timestamp_ns": 55}

        def step(self, action):
            return {}, 0.0, False, False, {}

        def close(self):
            pass

    env = EnvTimestampAdapter(Env(), wall_time_ns=lambda: 99)
    assert int(env.reset()[1]["timestamp_ns"]) == 55
    assert int(env.step(None)[4]["timestamp_ns"]) == 99


class _FakeRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "simulated response loss"


class _FakeChannel:
    def __init__(self, begin_handler, step_handler=None):
        self.begin_handler = begin_handler
        self.step_handler = step_handler

    def unary_unary(self, path, request_serializer, response_deserializer):
        del request_serializer, response_deserializer
        if path.endswith("/BeginEpisode"):
            return self.begin_handler
        if path.endswith("/Step") and self.step_handler is not None:
            return self.step_handler
        return lambda request, timeout: None

    def close(self):
        pass


def _action_reply(request, action, *, policy_version=0):
    observation = (
        request.observation
        if isinstance(request, pb.BeginEpisodeRequest)
        else request.next_observation
    )
    return pb.ActionReply(
        ok=True,
        protocol_version=PROTOCOL_VERSION,
        session_id=request.session_id,
        request_id=request.request_id,
        request_created_monotonic_ns=request.created_monotonic_ns,
        observation_id=observation.observation_id,
        action=action,
        policy_version=policy_version,
        server_inference_ms=1.0,
    )


def _outcome_reply(request, **overrides):
    values = {
        "transition_id": request.data.meta.transition_id,
        "reward": request.data.transition.rewards,
        "mask": request.data.transition.masks,
        "done": request.data.transition.dones,
        "truncated": request.data.transition.truncated,
    }
    values.update(overrides)
    return pb.TransitionOutcome(**values)


def test_grpc_client_retries_same_begin_request_once():
    requests = []

    def handler(request, timeout):
        del timeout
        requests.append(request.SerializeToString(deterministic=True))
        if len(requests) == 1:
            raise _FakeRpcError()
        return _action_reply(request, np.zeros(7, np.float32))

    ticks = iter((1_000_000_000, 1_100_000_000))
    client = GrpcActorNetwork(
        "unused",
        actor_id="actor",
        channel=_FakeChannel(handler),
        monotonic_ns=lambda: next(ticks),
    )

    result = client.begin_episode(
        _observation(0),
        run_id="run",
        session_id="session",
        episode_id=0,
        observation_id="o0",
        timestamp_ns=1_000,
    )

    assert len(requests) == 2
    assert requests[0] == requests[1]
    np.testing.assert_array_equal(result.action, np.zeros(7, np.float32))


def test_grpc_client_rejects_stale_action_reply():
    def handler(request, timeout):
        del timeout
        return _action_reply(request, np.zeros(7, np.float32))

    ticks = iter((1_000_000_000, 1_900_000_001))
    client = GrpcActorNetwork(
        "unused",
        actor_id="actor",
        channel=_FakeChannel(handler),
        monotonic_ns=lambda: next(ticks),
        max_response_age_s=0.8,
    )
    with pytest.raises(ActorProtocolError, match="stale"):
        client.begin_episode(
            _observation(0),
            run_id="run",
            session_id="session",
            episode_id=0,
            observation_id="o0",
            timestamp_ns=1_000,
        )


def test_grpc_client_rejects_decreasing_policy_version_after_ack():
    def begin_handler(request, timeout):
        del timeout
        return _action_reply(
            request, np.zeros(7, np.float32), policy_version=2
        )

    def step_handler(request, timeout):
        del timeout
        return pb.StepReply(
            ack=pb.Ack(
                accepted=True,
                transition_id=request.data.meta.transition_id,
                session_id=request.session_id,
                request_id=request.request_id,
            ),
            has_action=True,
            outcome=_outcome_reply(request),
            action=_action_reply(
                request,
                np.zeros(7, np.float32),
                policy_version=1,
            ),
        )

    ticks = iter((1_000_000_000, 1_010_000_000, 1_020_000_000, 1_030_000_000))
    client = GrpcActorNetwork(
        "unused",
        actor_id="actor",
        channel=_FakeChannel(begin_handler, step_handler),
        monotonic_ns=lambda: next(ticks),
    )
    action = client.begin_episode(
        _observation(0),
        run_id="run",
        session_id="session",
        episode_id=0,
        observation_id="o0",
        timestamp_ns=1_000,
    )
    with pytest.raises(ActorProtocolError, match="policy_version"):
        client.step(
            _observation(1),
            next_observation_id="o1",
            next_timestamp_ns=2_000,
            data=_data(action.action, policy_version=2),
            request_action=True,
        )


def test_grpc_client_rejects_outcome_that_disagrees_with_request_reward_mode():
    def begin_handler(request, timeout):
        del timeout
        return _action_reply(request, np.zeros(7, np.float32))

    def step_handler(request, timeout):
        del timeout
        return pb.StepReply(
            ack=pb.Ack(
                accepted=True,
                transition_id=request.data.meta.transition_id,
                session_id=request.session_id,
                request_id=request.request_id,
            ),
            has_action=True,
            outcome=_outcome_reply(
                request,
                success=False,
                classifier_evaluated=True,
                classifier_probability=0.9,
                classifier_threshold=0.5,
                reward_model_id="classifier-v1",
            ),
            action=_action_reply(request, np.zeros(7, np.float32)),
        )

    ticks = iter(
        (1_000_000_000, 1_010_000_000, 1_020_000_000, 1_030_000_000)
    )
    client = GrpcActorNetwork(
        "unused",
        actor_id="actor",
        channel=_FakeChannel(begin_handler, step_handler),
        monotonic_ns=lambda: next(ticks),
    )
    action = client.begin_episode(
        _observation(0),
        run_id="run",
        session_id="session",
        episode_id=0,
        observation_id="o0",
        timestamp_ns=1_000,
    )

    with pytest.raises(
        ActorProtocolError, match="outcome.success does not match"
    ):
        client.step(
            _observation(1),
            next_observation_id="o1",
            next_timestamp_ns=2_000,
            data=_data(action.action, auto_success=True),
            request_action=True,
        )


@pytest.mark.parametrize(
    "bad_action",
    [
        np.zeros(6, np.float32),
        np.array([0, 0, 0, 0, 0, 0, np.nan], np.float32),
        np.array([0, 0, 0, 0, 0, 0, 1.0001], np.float32),
    ],
)
def test_grpc_client_rejects_malformed_policy_action(bad_action):
    def handler(request, timeout):
        del timeout
        return _action_reply(request, bad_action)

    ticks = iter((1_000_000_000, 1_010_000_000))
    client = GrpcActorNetwork(
        "unused",
        actor_id="actor",
        channel=_FakeChannel(handler),
        monotonic_ns=lambda: next(ticks),
    )
    with pytest.raises(ActorProtocolError):
        client.begin_episode(
            _observation(0),
            run_id="run",
            session_id="session",
            episode_id=0,
            observation_id="o0",
            timestamp_ns=1_000,
        )


def test_real_loopback_grpc_begin_and_terminal_step():
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0)
    )
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}", actor_id="actor", timeout_s=1.0
    )
    try:
        assert client.health()[:2] == (True, True)
        info = client.get_server_info()
        assert info.action_dim == 7
        assert info.reward_authority == "local"
        action = client.begin_episode(
            _observation(0),
            run_id="run",
            session_id="session",
            episode_id=0,
            observation_id="o0",
            timestamp_ns=1_000,
        )
        result = client.step(
            _observation(1),
            next_observation_id="o1",
            next_timestamp_ns=2_000,
            data=_data(action.action, terminal=True),
            request_action=False,
        )
        assert result.ack.accepted
        assert result.action is None
        assert result.outcome.done
        assert result.outcome.reward == pytest.approx(1.25)
        assert not result.outcome.classifier_evaluated
        assert len(service.replay_items) == 1
        status = client.get_buffer_status()
        assert status.replay_size == 1
        assert status.intervention_size == 0
        assert status.replay_insert_count == 1
        assert status.last_transition_id == "run:0"
        assert status.last_env_step == 0
    finally:
        client.close()
        server.stop(grace=0).wait()


def test_expected_observation_schema_is_checked_before_begin_episode():
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        observation_schema_hash="server-schema",
    )
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}",
        actor_id="actor",
        timeout_s=1.0,
        expected_observation_schema_hash="laptop-schema",
    )
    try:
        with pytest.raises(ActorProtocolError, match="observation_schema_hash"):
            client.begin_episode(
                _observation(0),
                run_id="run",
                session_id="session",
                episode_id=0,
                observation_id="o0",
                timestamp_ns=1_000,
            )
        assert service.inference_count == 0
        assert service.observation_accept_count == 0
    finally:
        client.close()
        server.stop(grace=0).wait()


def test_grpc_client_accepts_server_classifier_terminal_without_action():
    def finalize(data):
        data["transition"].update(
            rewards=1.0,
            masks=0.0,
            dones=True,
            truncated=False,
        )
        return data, TransitionOutcome(
            transition_id=data["meta"]["transition_id"],
            reward=1.0,
            mask=0.0,
            done=True,
            truncated=False,
            success=True,
            classifier_evaluated=True,
            classifier_probability=0.91,
            classifier_threshold=0.85,
            reward_model_id="cube-in-cup:150",
        )

    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        reward_authority="server",
        reward_model_id="cube-in-cup:150",
        finalize_transition=finalize,
    )
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}", actor_id="actor", timeout_s=1.0
    )
    try:
        action = client.begin_episode(
            _observation(0),
            run_id="run",
            session_id="session",
            episode_id=0,
            observation_id="o0",
            timestamp_ns=1_000,
        )
        result = client.step(
            _observation(1),
            next_observation_id="o1",
            next_timestamp_ns=2_000,
            data=_data(action.action),
            request_action=True,
        )

        assert result.ack.accepted
        assert result.outcome.success
        assert result.outcome.done
        assert result.action is None
        assert service.inference_count == 1
    finally:
        client.close()
        server.stop(grace=0).wait()


def test_protocol_v1_begin_is_rejected_as_incompatible():
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0)
    )
    command = BeginEpisodeCommand(
        "1",
        "actor",
        "run",
        "session",
        0,
        1,
        10_000,
        ObservationPacket("o0", 1_000, _observation(0)),
    )

    with pytest.raises(ActorProtocolError, match="incompatible protocol_version"):
        service.begin_episode(command)
