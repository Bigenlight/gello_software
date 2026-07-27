"""Opt-in fake-data E2E test for the actual local-to-server learner path.

This deliberately expensive test covers the boundary that the default suite
keeps lightweight: a real frozen-trunk hybrid SAC agent serves actions over a
localhost gRPC actor connection, accepted canonical pixels are encoded into
feature replay, CTA updates publish a new policy, and the complete agent is
checkpointed.  It then restarts the server from that checkpoint and repeats
the transport/update/checkpoint cycle.

Run explicitly with::

    RUN_HIL_SERL_FAKE_E2E=1 pytest -q \
        serl_ur_infra/tests/test_actual_fake_data_e2e_learning.py

The fake demo artifact remains an acceptance-test fixture only.  Production
robot serving still rejects it; the CLI exposes a separately fingerprinted,
bounded ``--synthetic-e2e`` acceptance mode for the cross-machine equivalent.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HIL_SERL_FAKE_E2E") != "1",
    reason="set RUN_HIL_SERL_FAKE_E2E=1 to run the actual fake-data E2E test",
)


_HERE = Path(__file__).resolve().parent
_INFRA = _HERE.parent
_REPO = _INFRA.parent


def _observation(value: int) -> dict[str, np.ndarray]:
    """Return one strict canonical raw-pixel policy observation."""

    return {
        "state": np.full((1, 19), value / 255.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _make_ingress(feature_extractor: Any, *, seed: int):
    from ur_env.learner import FaultGatedReplayIngress, FeatureReplayIngress

    raw = FeatureReplayIngress(
        feature_extractor=feature_extractor,
        replay_capacity=4,
        intervention_capacity=2,
        seed=seed,
        available_memory_bytes=16 * 1024**2,
        memory_reserve_bytes=0,
    )
    return raw, FaultGatedReplayIngress(raw)


def _send_terminal_transition(
    client: Any,
    *,
    build_data: Any,
    run_id: str,
    index: int,
    episode_id: int,
    expected_policy_version: int,
    intervened: bool,
) -> None:
    """Exercise one complete actor request/transition/ACK over gRPC."""

    session_id = f"session-{index}"
    observation_id = f"o{index}"
    next_observation_id = f"o{index + 1}"
    timestamp_ns = 1_000_000 + index * 2
    action = client.begin_episode(
        _observation(10 + index),
        run_id=run_id,
        session_id=session_id,
        episode_id=episode_id,
        observation_id=observation_id,
        timestamp_ns=timestamp_ns,
        deterministic=True,
    )
    assert action.policy_version == expected_policy_version
    assert action.action.shape == (7,)
    assert action.action.dtype == np.float32
    assert np.isfinite(action.action).all()
    assert float(action.action[-1]) in (-1.0, 0.0, 1.0)

    info: dict[str, Any] = {
        "intervened": int(intervened),
        "grasp_penalty": -0.02 if intervened else 0.0,
    }
    if intervened:
        executed_action = np.zeros((7,), dtype=np.float32)
        executed_action[-1] = np.float32((-1.0, 0.0, 1.0)[index % 3])
        info["intervene_action"] = executed_action

    data = build_data(
        actor_id="fake-e2e-actor",
        run_id=run_id,
        session_id=session_id,
        transition_id=f"{run_id}:{index}",
        env_step=episode_id,
        timestamp_ns=timestamp_ns,
        policy_version=action.policy_version,
        policy_action=action.action,
        episode_id=episode_id,
        step_id=0,
        observation_id=observation_id,
        next_observation_id=next_observation_id,
        reward=0.0,
        done=True,
        truncated=False,
        info=info,
    )
    result = client.step(
        _observation(11 + index),
        next_observation_id=next_observation_id,
        next_timestamp_ns=timestamp_ns + 1,
        data=data,
        request_action=False,
        deterministic=True,
    )
    assert result.ack.accepted
    assert not result.ack.deduplicated
    assert result.outcome.classifier_evaluated
    assert result.outcome.terminal
    assert result.action is None


def _serve_two_transitions(
    *,
    assembly: Any,
    classifier: Any,
    create_grpc_server: Any,
    grpc_actor_type: Any,
    build_data: Any,
    schema_hash: str,
    run_id: str,
    first_index: int,
    expected_policy_version: int,
) -> None:
    """Bind a real loopback server and fill online/intervention feature RAM."""

    from ur_env.learner import build_actor_service

    service = build_actor_service(assembly=assembly, classifier=classifier)
    server, port = create_grpc_server(service)
    server.start()
    client = grpc_actor_type(
        f"127.0.0.1:{port}",
        actor_id="fake-e2e-actor",
        timeout_s=120.0,
        max_response_age_s=120.0,
        expected_observation_schema_hash=schema_hash,
        expected_model_id=assembly.policy_runtime.model_id,
        expected_reward_authority="server_classifier",
        expected_reward_model_id=classifier.reward_model_id,
    )
    try:
        server_info = client.get_server_info()
        assert server_info.ready
        assert server_info.model_id == assembly.policy_runtime.model_id
        for offset, intervened in enumerate((False, True)):
            _send_terminal_transition(
                client,
                build_data=build_data,
                run_id=run_id,
                index=first_index + offset,
                episode_id=offset,
                expected_policy_version=expected_policy_version,
                intervened=intervened,
            )
        status = client.get_buffer_status()
        assert status.replay_size == 2
        assert status.intervention_size == 1
        assert classifier.evaluation_count == 2
    finally:
        client.close()
        server.stop(0).wait(timeout=10.0)


def test_actual_fake_data_local_to_server_learning_checkpoint_resume(tmp_path):
    # The annotation-only TensorFlow shim and protobuf mode must be installed
    # before importing the real upstream agent and checked-in gRPC bindings.
    sys.path.insert(0, str(_INFRA))
    from ur_env.compat import configure_pure_python_protobuf

    configure_pure_python_protobuf()

    import jax

    from ur_env.grpc_actor_transport import GrpcActorNetwork, create_grpc_server
    from ur_env.learner import (
        CheckpointManager,
        FROZEN_TRUNK_CONTRACT,
        FROZEN_TRUNK_FEATURE_SHAPE,
        FrozenResNet10TrunkExtractor,
        LearnerConfig,
        LearnerFingerprint,
        compose_learner,
        convert_loaded_demos_to_feature_pool,
        create_frozen_trunk_feature_agent,
        load_demo_pickle,
        prepare_learner_state,
        validate_learner_dependencies,
        verify_resnet10_asset,
        write_fake_demo_pickle,
    )
    from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH
    from ur_env.remote_actor import build_data
    from ur_env.rlpd_receive_server import ScriptedRewardClassifierRuntime

    assert jax.default_backend() == "cpu"
    versions = validate_learner_dependencies(include_logging=False)
    assert versions == {
        "jax": "0.5.3",
        "jaxlib": "0.5.3",
        "flax": "0.10.5",
        "distrax": "0.1.5",
        "tensorflow_probability": "0.25.0",
    }

    config = LearnerConfig(
        batch_size=2,
        training_starts=2,
        publish_period=1,
        checkpoint_period=1,
        log_period=1,
        wandb_mode="disabled",
    )
    resnet_source = (
        _REPO
        / "third_party"
        / "hil-serl"
        / "examples"
        / "experiments"
        / "resnet10_params.pkl"
    )
    verify_resnet10_asset(resnet_source)
    resnet_cache = tmp_path / "resnet10_params.pkl"
    demo_path = tmp_path / "canonical_fake_demo.pkl"
    write_fake_demo_pickle(demo_path)
    loaded_demos = load_demo_pickle(demo_path)
    assert len(loaded_demos) == 2

    def create_agent():
        return create_frozen_trunk_feature_agent(
            config=config,
            resnet_source_path=resnet_source,
            resnet_cache_path=resnet_cache,
            validate_versions=True,
        )

    initial_agent = create_agent()
    extractor = FrozenResNet10TrunkExtractor(
        initial_agent, resnet_asset_path=resnet_source
    )
    feature_demos = convert_loaded_demos_to_feature_pool(
        loaded_demos,
        feature_extractor=extractor,
        seed=100,
        extraction_batch_size=2,
    )
    demo_sample = feature_demos.sample(1)
    assert demo_sample["observations"]["cam1"].shape == (
        1,
        *FROZEN_TRUNK_FEATURE_SHAPE,
    )
    assert demo_sample["observations"]["cam1"].dtype == np.float32

    fingerprint = LearnerFingerprint.create(
        config=config,
        resnet_asset_path=resnet_source,
        run_contract={
            "test": "actual-fake-data-local-to-server-e2e-v1",
            "learner_observations": FROZEN_TRUNK_CONTRACT.document(),
            "augmentation": "none",
        },
    )
    manager = CheckpointManager(tmp_path / "checkpoints")
    raw_ingress, ingress = _make_ingress(extractor, seed=101)
    assembly = compose_learner(
        agent_template=initial_agent,
        ingress=ingress,
        offline_demos=feature_demos,
        checkpoint_manager=manager,
        fingerprint=fingerprint,
        config=config,
        parameter_validator=extractor.validate_parameter_invariant,
        candidate_postprocessor=extractor.repin_target_trunk,
    )

    _serve_two_transitions(
        assembly=assembly,
        classifier=ScriptedRewardClassifierRuntime(
            [0.1, 0.9], reward_model_id="fake-e2e-reward-v1"
        ),
        create_grpc_server=create_grpc_server,
        grpc_actor_type=GrpcActorNetwork,
        build_data=build_data,
        schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        run_id="fake-e2e-fresh",
        first_index=0,
        expected_policy_version=0,
    )
    assert assembly.learner.ready
    assert raw_ingress.status().replay_size == 2
    first = assembly.learner.train_once()
    assert first.learner_step == 1
    assert first.gradient_step == 2
    assert first.policy_version == 1
    assert first.published
    assert first.checkpoint_path == str(manager.path_for_step(1))
    assert manager.path_for_step(1).is_dir()

    # Restart from the complete checkpoint before allocating the next live
    # ingress, then prove the restored policy and learner continue together.
    fresh_agent = create_agent()
    fresh_extractor = FrozenResNet10TrunkExtractor(
        fresh_agent, resnet_asset_path=resnet_source
    )
    prepared = prepare_learner_state(
        agent_template=fresh_agent,
        checkpoint_manager=manager,
        fingerprint=fingerprint,
        config=config,
        resume_path=first.checkpoint_path,
    )
    assert prepared.restored_checkpoint is not None
    fresh_extractor.validate_agent_invariant(prepared.agent)
    resumed_raw_ingress, resumed_ingress = _make_ingress(
        fresh_extractor, seed=201
    )
    resumed = compose_learner(
        agent_template=fresh_agent,
        ingress=resumed_ingress,
        offline_demos=feature_demos,
        checkpoint_manager=manager,
        fingerprint=fingerprint,
        config=config,
        prepared_state=prepared,
        parameter_validator=fresh_extractor.validate_parameter_invariant,
        candidate_postprocessor=fresh_extractor.repin_target_trunk,
    )
    assert resumed.learner.learner_step == 1
    assert resumed.learner.gradient_step == 2
    assert resumed.policy_runtime.policy_version == 1

    _serve_two_transitions(
        assembly=resumed,
        classifier=ScriptedRewardClassifierRuntime(
            [0.2, 0.3], reward_model_id="fake-e2e-reward-v1"
        ),
        create_grpc_server=create_grpc_server,
        grpc_actor_type=GrpcActorNetwork,
        build_data=build_data,
        schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        run_id="fake-e2e-resumed",
        first_index=2,
        expected_policy_version=1,
    )
    assert resumed.learner.ready
    assert resumed_raw_ingress.status().replay_size == 2
    second = resumed.learner.train_once()
    assert second.learner_step == 2
    assert second.gradient_step == 4
    assert second.policy_version == 2
    assert second.published
    assert second.checkpoint_path == str(manager.path_for_step(2))
    assert manager.path_for_step(2).is_dir()
    fresh_extractor.validate_agent_invariant(resumed.learner.agent)
