"""Loopback test for the multi-step receive-server smoke client."""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import ActorSessionService, TransitionOutcome  # noqa: E402
from ur_env.grpc_actor_transport import (  # noqa: E402
    GrpcActorNetwork,
    create_grpc_server,
)
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)
from ur_env.rlpd_receive_smoke import run_receive_smoke  # noqa: E402


def test_receive_smoke_matches_exact_buffer_counters():
    def finalize(data):
        transition = data["transition"]
        return data, TransitionOutcome(
            transition_id=data["meta"]["transition_id"],
            reward=float(transition["rewards"]),
            mask=float(transition["masks"]),
            done=bool(transition["dones"]),
            truncated=bool(transition["truncated"]),
            success=False,
            classifier_evaluated=True,
            classifier_probability=0.1,
            classifier_threshold=0.85,
            reward_model_id="scripted-reward",
        )

    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        reward_authority="server_classifier",
        reward_model_id="scripted-reward",
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        finalize_transition=finalize,
        in_memory_capacity=32,
    )
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}",
        actor_id="receive-smoke-test",
        timeout_s=1.0,
        max_response_age_s=2.0,
        expected_observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
    )
    try:
        result = run_receive_smoke(
            client,
            actor_id="receive-smoke-test",
            steps=10,
            local_episode_steps=4,
            intervention_period=5,
            run_id="receive-smoke-run",
            wall_time_ns=lambda: 1_721_800_000_123_456_789,
        )
    finally:
        client.close()
        server.stop(grace=0).wait()

    assert result.steps == 10
    assert result.episodes == 3
    assert result.interventions == 2
    assert result.classifier_successes == 0
    assert result.replay_insert_delta == 10
    assert result.intervention_insert_delta == 2
    assert result.last_transition_id == "receive-smoke-run:9"
