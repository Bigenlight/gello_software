"""ROS/JAX-free tests for the repeatable remote actor smoke path."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import ActorSessionService  # noqa: E402
from ur_env.actor_smoke import (  # noqa: E402
    IMAGE_SHAPE,
    SummaryDataSink,
    run_mock_smoke,
    synthetic_observation,
)
from ur_env.grpc_actor_transport import (  # noqa: E402
    GrpcActorNetwork,
    create_grpc_server,
)
from ur_env.remote_actor import build_data  # noqa: E402


def test_real_loopback_smoke_sends_normal_and_intervention_terminal_data():
    lines = []
    sink = SummaryDataSink(capacity=4, emit=lines.append)
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        model_id="mock-zero-policy",
        accept_data=sink,
    )
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}",
        actor_id="smoke-test",
        timeout_s=1.0,
        max_response_age_s=2.0,
    )
    try:
        result = run_mock_smoke(
            client,
            actor_id="smoke-test",
            run_id="smoke-run",
            session_id="smoke-session",
            wall_time_ns=lambda: 1_721_800_000_123_456_789,
        )
    finally:
        client.close()
        server.stop(grace=0).wait()

    assert result.transition_ids == ("smoke-run:0", "smoke-run:1")
    assert sink.replay_count == 2
    assert sink.intervention_count == 1
    assert service.replay_items == []
    assert service.intervention_items == []
    assert service.observation_accept_count == 3
    assert service.inference_count == 2
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]
    assert [record["env_step"] for record in records] == [0, 1]
    assert [record["step_id"] for record in records] == [0, 1]
    assert [record["intervened"] for record in records] == [0, 1]
    assert [record["replay_count"] for record in records] == [1, 2]
    assert [record["intervention_count"] for record in records] == [0, 1]
    assert [record["timestamp_ns"] for record in records] == [
        1_721_800_000_123_456_789,
        1_721_800_000_123_456_790,
    ]
    assert records[0]["dones"] is False
    assert records[1]["dones"] is True
    shown = records[0]["observation_tensors"]["shown"]
    image_specs = [spec for spec in shown if spec["path"] in ("cam1", "cam2")]
    assert len(image_specs) == 2
    assert all(spec["dtype"] == "uint8" for spec in image_specs)
    assert all(spec["shape"] == list(IMAGE_SHAPE) for spec in image_specs)
    assert all("array(" not in line and "bytes" not in line for line in lines)


def test_summary_sink_is_bounded_and_escapes_untrusted_tensor_paths():
    lines = []
    sink = SummaryDataSink(
        capacity=1,
        emit=lines.append,
        max_tensor_specs=2,
        max_path_chars=24,
    )
    action = np.zeros(7, np.float32)
    data = build_data(
        actor_id="actor",
        run_id="run",
        session_id="session",
        transition_id="transition",
        env_step=0,
        timestamp_ns=1_000,
        policy_version=0,
        policy_action=action,
        episode_id=0,
        step_id=0,
        observation_id="o0",
        next_observation_id="o1",
        reward=0.0,
        done=False,
        truncated=False,
        info={"intervened": 0},
    )
    observation = synthetic_observation(0)
    observation["aaa\nvery-long-key-and-more"] = np.ones(
        3, np.float32
    )
    data["transition"]["observations"] = observation
    data["transition"]["next_observations"] = synthetic_observation(1)

    sink(data, False)

    assert len(lines) == 1
    assert "\n" not in lines[0]
    record = json.loads(lines[0])
    assert record["observation_tensors"]["total"] == 4
    assert record["observation_tensors"]["omitted"] == 2
    assert all(
        len(spec["path"]) <= 24
        for spec in record["observation_tensors"]["shown"]
    )
    assert any(
        "\n" in spec["path"]
        for spec in record["observation_tensors"]["shown"]
    )
    with pytest.raises(BufferError, match="capacity"):
        sink(data, False)
    assert sink.replay_count == 1
    assert len(lines) == 1
