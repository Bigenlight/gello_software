"""Opt-in fake-data E2E test for the actual local-to-server learner path.

This deliberately expensive test covers the boundary that the default suite
keeps lightweight: a real frozen-trunk hybrid SAC agent serves actions over a
localhost gRPC actor connection, accepted canonical pixels are encoded into
feature replay, CTA updates publish a new policy, and the complete agent is
checkpointed.  It then restarts the server from that checkpoint and repeats
the transport/update/checkpoint cycle.

Each server round also drives BOTH halves of the sparse-classification
contract: one transition ships a real classifier sidecar and is scored, the
other ships none and is finalized unclassified.  See
:func:`_serve_two_transitions` for why that mixture is the normal operating
mode rather than a degraded one.

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

#: Timestamp of the first observation of every episode served below.  The
#: service requires ``meta.timestamp_ns`` to be the *source* observation's
#: timestamp, so the whole episode is derived from this one number.
_BASE_TIMESTAMP_NS = 1_000_000


def _observation(value: int) -> dict[str, np.ndarray]:
    """Return one strict canonical raw-pixel policy observation."""

    return {
        "state": np.full((1, 19), value / 255.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _observation_with_sidecar(value: int) -> dict[str, Any]:
    """The canonical observation PLUS the frames the classifier actually scores.

    The server no longer classifies ``transition.next_observations``; it scores
    only the uncropped JPEG pair the actor attaches under the reserved
    ``CLASSIFIER_SIDECAR_KEY``.  The policy observation is cropped by
    ``ur_experiments/cube_in_cup.py::IMAGE_CROP`` while the reward checkpoint
    was trained on full frames, and scoring the cropped view cost recall@0.85
    100% -> 33.3%.  So a transition whose observation carries no sidecar is
    *unclassified* by design, and any test that wants a reward verdict has to
    ship one.

    ``ActorSessionService.step`` splits the reserved key off before canonical
    validation, so what reaches the policy, the finalizer's protected-field diff
    and feature replay is still exactly the three canonical tensors.

    The payload comes from the production ``build_sidecar`` — fed the
    full-resolution-shaped, UNCROPPED BGR frame it expects, since it owns both
    the downscale to the classifier's 128x128 and the JPEG encode — so this test
    fails loudly the day the wire contract moves instead of drifting off it.
    """

    from ur_env.classifier_sidecar import CLASSIFIER_SIDECAR_KEY, build_sidecar

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, value % 3] = np.uint8(value % 256)
    observation: dict[str, Any] = dict(_observation(value))
    observation[CLASSIFIER_SIDECAR_KEY] = build_sidecar(
        {"cam1": frame, "cam2": frame}
    )
    return observation


def _assert_served_action(action: Any, expected_policy_version: int) -> None:
    """Every action the server hands back must be usable by the real actor."""

    assert action is not None
    assert action.policy_version == expected_policy_version
    assert action.action.shape == (7,)
    assert action.action.dtype == np.float32
    assert np.isfinite(action.action).all()
    assert float(action.action[-1]) in (-1.0, 0.0, 1.0)


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


def _send_transition(
    client: Any,
    *,
    build_data: Any,
    classifier: Any,
    run_id: str,
    session_id: str,
    base_value: int,
    step_id: int,
    action: Any,
    intervened: bool,
    terminal: bool,
    expected_probability: float | None,
) -> Any:
    """Exercise one complete actor request/transition/ACK over gRPC.

    ``expected_probability is None`` means "attach no sidecar", and the server
    must then return an *unclassified* outcome.  Any other value attaches a real
    sidecar to O(t+1) and pins the verdict exactly: the finalizer runs with
    ``success_confirmations == 1``, so the probability it reports is the
    instantaneous one the classifier produced, not a smoothed window.

    Returns the action the server served for O(t+1), or ``None`` on a terminal
    step, so the caller can chain the next transition onto it.  Every id is
    derived from ``run_id``/``step_id`` because the service cross-checks all of
    them against its own session state: ``meta.timestamp_ns`` must be the source
    observation's timestamp, and ``env_step``/``step_id`` must advance by
    exactly one.
    """

    observation_id = f"{run_id}-o{step_id}"
    next_observation_id = f"{run_id}-o{step_id + 1}"
    timestamp_ns = _BASE_TIMESTAMP_NS + step_id
    next_value = base_value + step_id + 1
    next_observation = (
        _observation(next_value)
        if expected_probability is None
        else _observation_with_sidecar(next_value)
    )

    info: dict[str, Any] = {
        "intervened": int(intervened),
        "grasp_penalty": -0.02 if intervened else 0.0,
    }
    if intervened:
        executed_action = np.zeros((7,), dtype=np.float32)
        executed_action[-1] = np.float32((-1.0, 0.0, 1.0)[step_id % 3])
        info["intervene_action"] = executed_action

    data = build_data(
        actor_id="fake-e2e-actor",
        run_id=run_id,
        session_id=session_id,
        transition_id=f"{run_id}:{step_id}",
        # One episode per run here, so the run's env_step and the episode's
        # step_id advance together.
        env_step=step_id,
        timestamp_ns=timestamp_ns,
        policy_version=action.policy_version,
        policy_action=action.action,
        # The synthetic E2E case is explicitly classifier-authoritative; real
        # GUI sessions default to MANUAL.
        auto_success=True,
        episode_id=0,
        step_id=step_id,
        observation_id=observation_id,
        next_observation_id=next_observation_id,
        reward=0.0,
        done=terminal,
        truncated=False,
        info=info,
    )
    result = client.step(
        next_observation,
        next_observation_id=next_observation_id,
        next_timestamp_ns=timestamp_ns + 1,
        data=data,
        # The service requires request_action to be false exactly on a
        # terminal/truncated step, and true otherwise.
        request_action=not terminal,
        deterministic=True,
    )
    assert result.ack.accepted
    assert not result.ack.deduplicated
    assert result.outcome.terminal is terminal

    if expected_probability is None:
        # The unclassified contract, field for field, as documented on
        # ``rlpd_receive_server.RewardTransitionFinalizer``.  With no sidecar
        # there is no evidence of success, so reward is forced to 0 and every
        # classifier scalar must be exactly zero/empty — three independent
        # validators reject a "real but unused" probability here.  What still
        # passes through untouched is the locally proposed done/truncated, which
        # is what keeps such a transition an ordinary sample in replay rather
        # than a rejected one.
        assert not result.outcome.classifier_evaluated
        assert result.outcome.reward == 0.0
        assert not result.outcome.success
        assert result.outcome.classifier_probability == 0.0
        assert result.outcome.classifier_threshold == 0.0
        assert result.outcome.reward_model_id == ""
    else:
        success = expected_probability > classifier.threshold
        assert result.outcome.classifier_evaluated
        assert result.outcome.classifier_probability == expected_probability
        assert result.outcome.classifier_threshold == classifier.threshold
        assert result.outcome.success is success
        # Reward authority is the server's in both directions: the actor
        # proposed 0.0 above, and only the classifier can raise it.
        assert result.outcome.reward == (1.0 if success else 0.0)
        assert result.outcome.reward_model_id == classifier.reward_model_id

    if terminal:
        assert result.action is None
        return None
    _assert_served_action(result.action, action.policy_version)
    return result.action


def _serve_two_transitions(
    *,
    assembly: Any,
    classifier: Any,
    create_grpc_server: Any,
    grpc_actor_type: Any,
    build_data: Any,
    schema_hash: str,
    run_id: str,
    base_value: int,
    expected_policy_version: int,
    classified_probability: float,
) -> None:
    """Bind a real loopback server and fill online/intervention feature RAM.

    One episode, two transitions, and they deliberately take DIFFERENT halves of
    the sparse-classification contract:

    * step 0 carries no sidecar, so the server cannot classify it.  That is the
      ordinary case on the rig — the actor attaches frames at roughly 2 Hz and
      only while the arm is stationary (``classifier_sidecar.SidecarScheduler``),
      so at a 10 Hz control loop most transitions arrive unclassified — and such
      a transition must still land in feature replay as a zero-reward sample.
    * step 1 carries a real sidecar and is the terminal, human-intervened
      transition, so the classified path runs end to end: decode -> classify ->
      server-authoritative reward -> intervention routing.

    The sparsity is a design decision, not a defect.  Recomputing the verdict
    only a few times per second is what stops it flickering at 10 Hz (p was
    observed swinging 0.005 -> 1.0 within a single gripper sweep) and lets the
    scene settle after the cube is released.  A test that classified every
    transition would be pinning behaviour we deliberately do not ship.
    """

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

        session_id = f"{run_id}-session"
        action = client.begin_episode(
            _observation(base_value),
            run_id=run_id,
            session_id=session_id,
            episode_id=0,
            observation_id=f"{run_id}-o0",
            timestamp_ns=_BASE_TIMESTAMP_NS,
            deterministic=True,
        )
        _assert_served_action(action, expected_policy_version)

        # Mid-episode step with no sidecar: unclassified, and the server still
        # serves the next action off the accepted observation.
        action = _send_transition(
            client,
            build_data=build_data,
            classifier=classifier,
            run_id=run_id,
            session_id=session_id,
            base_value=base_value,
            step_id=0,
            action=action,
            intervened=False,
            terminal=False,
            expected_probability=None,
        )
        # Terminal, human-intervened step carrying a real sidecar: the one
        # transition in this episode the classifier is allowed to score.
        _send_transition(
            client,
            build_data=build_data,
            classifier=classifier,
            run_id=run_id,
            session_id=session_id,
            base_value=base_value,
            step_id=1,
            action=action,
            intervened=True,
            terminal=True,
            expected_probability=classified_probability,
        )

        status = client.get_buffer_status()
        assert status.replay_size == 2
        assert status.intervention_size == 1
        # Exactly one of the two transitions carried a sidecar, so exactly one
        # classification may have happened.  The scripted classifier is loaded
        # with a single probability, so classifying anything else raises
        # "sequence exhausted" rather than quietly inflating this count.
        assert classifier.evaluation_count == 1
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

    # Above DEFAULT_REWARD_THRESHOLD (0.2), so the classified transition is a
    # success and the finalizer's reward=1/done/mask override is exercised.
    fresh_probability = 0.9
    _serve_two_transitions(
        assembly=assembly,
        classifier=ScriptedRewardClassifierRuntime(
            [fresh_probability], reward_model_id="fake-e2e-reward-v1"
        ),
        create_grpc_server=create_grpc_server,
        grpc_actor_type=GrpcActorNetwork,
        build_data=build_data,
        schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        run_id="fake-e2e-fresh",
        base_value=10,
        expected_policy_version=0,
        classified_probability=fresh_probability,
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

    # Below the threshold this time: the classified transition gets a real
    # verdict of "not a success", which is a different path from the
    # unclassified one above even though both end with reward 0.
    resumed_probability = 0.15
    _serve_two_transitions(
        assembly=resumed,
        classifier=ScriptedRewardClassifierRuntime(
            [resumed_probability], reward_model_id="fake-e2e-reward-v1"
        ),
        create_grpc_server=create_grpc_server,
        grpc_actor_type=GrpcActorNetwork,
        build_data=build_data,
        schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        run_id="fake-e2e-resumed",
        base_value=20,
        expected_policy_version=1,
        classified_probability=resumed_probability,
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
