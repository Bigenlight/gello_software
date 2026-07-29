"""Production learner composition, worker, and localhost service tests."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("jax")
pytest.importorskip("flax")
import jax


_HERE = Path(__file__).resolve().parent
_INFRA = _HERE.parent
sys.path.insert(0, str(_INFRA))
sys.path.insert(0, str(_HERE))

from test_learner_policy_checkpoint import (  # noqa: E402
    _agent,
    _sample_action,
    _transition,
)
from ur_env.actor_network import ActorTransportError  # noqa: E402
from ur_env.actor_smoke import synthetic_observation  # noqa: E402
from ur_env.grpc_actor_transport import (  # noqa: E402
    GrpcActorNetwork,
    create_grpc_server,
)
from ur_env.learner import (  # noqa: E402
    CheckpointCorruptError,
    CheckpointFingerprintError,
    CheckpointManager,
    FROZEN_TRUNK_FEATURE_SHAPE,
    FaultGatedReplayIngress,
    FeatureReplayIngress,
    FeatureTransitionRing,
    LearnerCompositionError,
    LearnerConfig,
    LearnerFingerprint,
    LearnerWorker,
    build_actor_service,
    compose_learner,
    preflight_checkpoint_run,
    prepare_learner_state,
)
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)
from ur_env.remote_actor import build_data  # noqa: E402
from ur_env.rlpd_receive_server import ScriptedRewardClassifierRuntime  # noqa: E402


_FINGERPRINT = LearnerFingerprint(
    document={"test": "production-composition"},
    sha256="0" * 64,
)


class _PoolIngress:
    """Strict in-memory source using the existing canonical pool helper."""

    require_grasp_penalty = True
    observation_representation = "resnet10_frozen_trunk_map_f32_v1"
    augmentation = "none"

    def __init__(self, replay_count: int = 4, intervention_count: int = 1):
        self.replay = _feature_pool(
            [_feature_transition(index) for index in range(replay_count)],
            seed=11,
        )
        self.interventions = _feature_pool(
            [
                _feature_transition(100 + index)
                for index in range(intervention_count)
            ],
            seed=12,
        )

    def status(self):
        return SimpleNamespace(
            replay_size=len(self.replay),
            intervention_size=len(self.interventions),
            replay_insert_count=self.replay.insert_count,
        )

    def sample_replay(self, batch_size: int, **kwargs):
        del kwargs
        return self.replay.sample(batch_size)

    def sample_intervention(self, batch_size: int, **kwargs):
        del kwargs
        return self.interventions.sample(batch_size)


def _feature_transition(value: int) -> dict:
    raw = _transition(value)

    def observation(raw_observation, marker):
        return {
            "state": raw_observation["state"].copy(),
            "cam1": np.full(
                FROZEN_TRUNK_FEATURE_SHAPE, marker, dtype=np.float32
            ),
            "cam2": np.full(
                FROZEN_TRUNK_FEATURE_SHAPE, marker + 1, dtype=np.float32
            ),
        }

    return {
        **raw,
        "observations": observation(raw["observations"], float(value)),
        "next_observations": observation(
            raw["next_observations"], float(value + 1)
        ),
    }


def _feature_pool(transitions, *, seed: int):
    ring = FeatureTransitionRing(max(1, len(transitions)), seed=seed)
    for transition in transitions:
        ring.insert(transition)
    return ring


def _offline_pool():
    return _feature_pool([_feature_transition(200)], seed=13)


def _compose_fresh(
    root: Path,
    *,
    agent=None,
    config: LearnerConfig | None = None,
    ingress=None,
):
    config = config or LearnerConfig(
        batch_size=4,
        training_starts=4,
        publish_period=1,
        checkpoint_period=100,
    )
    return compose_learner(
        agent_template=agent or _agent(),
        ingress=ingress or _PoolIngress(),
        offline_demos=_offline_pool(),
        checkpoint_manager=CheckpointManager(root),
        fingerprint=_FINGERPRINT,
        config=config,
        sample_action=_sample_action,
    )


def test_fresh_composition_starts_at_exact_zero_without_publish(tmp_path):
    agent = _agent()
    assembly = _compose_fresh(tmp_path / "fresh", agent=agent)

    assert assembly.restored_checkpoint is None
    assert assembly.learner.agent is agent
    assert assembly.learner.learner_step == 0
    assert assembly.learner.gradient_step == 0
    assert assembly.learner.policy_version == 0
    assert assembly.policy_runtime.learner_step == 0
    assert assembly.policy_runtime.policy_version == 0
    assert assembly.policy_runtime.snapshot.params is agent.state.params


def test_fresh_composition_rejects_nonzero_agent_and_nonstrict_ingress(
    tmp_path,
):
    with pytest.raises(LearnerCompositionError, match="state.step must be 0"):
        _compose_fresh(tmp_path / "nonzero", agent=_agent(step=2))

    ingress = _PoolIngress()
    ingress.require_grasp_penalty = False
    with pytest.raises(LearnerCompositionError, match="require grasp_penalty"):
        _compose_fresh(tmp_path / "nonstrict", ingress=ingress)

    ingress = _PoolIngress()
    ingress.observation_representation = "raw_pixels_packed_uint8_v1"
    with pytest.raises(
        LearnerCompositionError, match="representation/augmentation"
    ):
        _compose_fresh(tmp_path / "raw-ingress", ingress=ingress)


def test_checkpoint_preflight_refuses_implicit_or_mixed_lineages(tmp_path):
    manager = CheckpointManager(tmp_path / "output")
    assert preflight_checkpoint_run(manager) is None

    incomplete = manager.root / "checkpoint_000000000004"
    incomplete.mkdir()
    with pytest.raises(LearnerCompositionError, match="fresh start refused"):
        preflight_checkpoint_run(manager)

    external = tmp_path / "external" / "checkpoint_000000000008"
    external.mkdir(parents=True)
    with pytest.raises(LearnerCompositionError, match="checkpoint-empty"):
        preflight_checkpoint_run(manager, resume_path=external)

    empty_manager = CheckpointManager(tmp_path / "new-output")
    assert (
        preflight_checkpoint_run(empty_manager, resume_path=external)
        == external.resolve()
    )


def test_resume_restores_exact_counters_rng_without_republish(tmp_path):
    config = LearnerConfig(
        batch_size=4,
        training_starts=4,
        publish_period=2,
        checkpoint_period=4,
    )
    manager = CheckpointManager(tmp_path / "resume")
    saved_rng = jax.random.PRNGKey(77)
    checkpoint = manager.save(
        agent=_agent(step=8, value=0.4),
        learner_step=4,
        gradient_step=8,
        policy_version=2,
        inference_rng=saved_rng,
        fingerprint=_FINGERPRINT,
    )

    template = _agent()
    prepared = prepare_learner_state(
        agent_template=template,
        checkpoint_manager=manager,
        fingerprint=_FINGERPRINT,
        config=config,
        resume_path=checkpoint,
    )
    assembly = compose_learner(
        agent_template=template,
        ingress=_PoolIngress(),
        offline_demos=_offline_pool(),
        checkpoint_manager=manager,
        fingerprint=_FINGERPRINT,
        config=config,
        prepared_state=prepared,
        sample_action=_sample_action,
    )

    assert assembly.restored_checkpoint is not None
    assert assembly.learner.learner_step == 4
    assert assembly.learner.gradient_step == 8
    assert assembly.learner.policy_version == 2
    # A constructor-side republish would incorrectly advance this to 3.
    assert assembly.policy_runtime.policy_version == 2
    assert assembly.policy_runtime.learner_step == 4
    np.testing.assert_array_equal(
        np.asarray(assembly.policy_runtime.inference_rng),
        np.asarray(saved_rng),
    )
    _, served_version = assembly.policy_runtime(
        synthetic_observation(0), deterministic=True
    )
    assert served_version == 2


@pytest.mark.parametrize(
    ("learner_step", "gradient_step", "policy_version", "match"),
    [
        (4, 6, 2, r"learner_step \* cta_ratio"),
        (4, 8, 1, "policy_version"),
        (2, 4, 1, "checkpoint boundary"),
    ],
)
def test_resume_rejects_counter_invariant_drift(
    tmp_path,
    learner_step,
    gradient_step,
    policy_version,
    match,
):
    config = LearnerConfig(
        batch_size=4,
        training_starts=4,
        publish_period=2,
        checkpoint_period=4,
    )
    manager = CheckpointManager(tmp_path / f"bad-{learner_step}-{gradient_step}")
    checkpoint = manager.save(
        agent=_agent(step=gradient_step),
        learner_step=learner_step,
        gradient_step=gradient_step,
        policy_version=policy_version,
        inference_rng=jax.random.PRNGKey(88),
        fingerprint=_FINGERPRINT,
    )

    with pytest.raises(CheckpointCorruptError, match=match):
        compose_learner(
            agent_template=_agent(),
            ingress=_PoolIngress(),
            offline_demos=_offline_pool(),
            checkpoint_manager=manager,
            fingerprint=_FINGERPRINT,
            config=config,
            resume_path=checkpoint,
            sample_action=_sample_action,
        )


def test_worker_reaches_bounded_target_and_publishes_once(tmp_path):
    assembly = _compose_fresh(tmp_path / "worker-complete")
    worker = LearnerWorker(
        assembly.learner,
        replay_insert_count=lambda: (
            assembly.ingress.status().replay_insert_count
        ),
        target_learner_step=1,
        poll_interval=0.001,
    )

    worker.start()
    assert worker.wait(timeout=2.0)
    assert worker.join(timeout=2.0)

    assert worker.status.state == "completed"
    assert worker.status.learner_step == 1
    assert worker.status.gradient_step == 2
    assert worker.status.policy_version == 1
    assert assembly.policy_runtime.policy_version == 1


def test_worker_fault_keeps_last_known_good_policy_callable(tmp_path):
    ingress = _PoolIngress(replay_count=5)
    assembly = _compose_fresh(
        tmp_path / "worker-fault",
        agent=_agent(fail_at=3),
        ingress=ingress,
    )
    worker = LearnerWorker(
        assembly.learner,
        replay_insert_count=lambda: ingress.status().replay_insert_count,
        target_learner_step=2,
        poll_interval=0.001,
    )

    worker.start()
    assert worker.wait(timeout=2.0)
    assert worker.join(timeout=2.0)

    assert worker.status.state == "faulted"
    assert worker.status.learner_step == 1
    assert worker.status.gradient_step == 2
    assert worker.status.policy_version == 1
    assert "non-finite" in worker.status.detail
    action, version = assembly.policy_runtime(
        synthetic_observation(3), deterministic=True
    )
    assert version == 1
    assert np.isfinite(action).all()


def test_resume_refusal_explains_the_intended_one_time_fingerprint_break(
    tmp_path,
):
    """Retiring the 0%-recall classifier breaks resume once, on purpose.

    The classifier SHA, the reward_model_id and the run_contract's
    reward_classifier block all feed the fingerprint, so pre-change checkpoints
    are refused.  The refusal must read as a decision, not as a defect, or the
    next operator will "fix" it by weakening the check.
    """

    manager = CheckpointManager(tmp_path / "reward-epoch")
    checkpoint = manager.save(
        agent=_agent(step=8, value=0.4),
        learner_step=4,
        gradient_step=8,
        policy_version=2,
        inference_rng=jax.random.PRNGKey(77),
        fingerprint=_FINGERPRINT,
    )
    rotated = LearnerFingerprint(
        document={"test": "post-classifier-retirement"},
        sha256="1" * 64,
    )

    with pytest.raises(CheckpointFingerprintError) as excinfo:
        prepare_learner_state(
            agent_template=_agent(),
            checkpoint_manager=manager,
            fingerprint=rotated,
            resume_path=checkpoint,
        )

    message = str(excinfo.value)
    assert "mismatch" in message
    assert "expected and intended" in message
    assert "0% recall" in message
    # The remedy has to be in the message; without it the reflex is to disable
    # the fingerprint check.
    assert "--checkpoint-root" in message
    assert str(checkpoint) in message


def test_build_actor_service_forwards_success_confirmations(
    tmp_path, monkeypatch
):
    """The reward smoothing setting must reach the finalizer, not be dropped.

    ``RewardTransitionFinalizer`` is stubbed rather than inspected so this pins
    *our* wiring (entrypoint flag -> finalizer constructor) without depending on
    the finalizer's private attribute names.
    """

    import ur_env.rlpd_receive_server as receive_server

    captured = {}

    class _Finalizer:
        # Mirrors the real signature: the finalizer's keyword is
        # `confirmations`, the operator-facing flag is --success-confirmations.
        def __init__(self, classifier, *, confirmations=1):
            captured["classifier"] = classifier
            captured["success_confirmations"] = confirmations

        def __call__(self, data):  # pragma: no cover - never invoked here
            raise AssertionError("stub finalizer must not be called")

    monkeypatch.setattr(
        receive_server, "RewardTransitionFinalizer", _Finalizer
    )

    assembly = _compose_fresh(tmp_path / "confirmations")
    classifier = ScriptedRewardClassifierRuntime([0.1])

    build_actor_service(assembly=assembly, classifier=classifier)
    # 1 == no smoothing.  Deliberate operator default: it keeps the server's
    # verdict identical to the live classifier viewer's per-frame probability.
    assert captured["success_confirmations"] == 1
    assert captured["classifier"] is classifier

    build_actor_service(
        assembly=assembly, classifier=classifier, success_confirmations=3
    )
    assert captured["success_confirmations"] == 3


def test_build_actor_service_rejects_invalid_success_confirmations(tmp_path):
    assembly = _compose_fresh(tmp_path / "bad-confirmations")
    classifier = ScriptedRewardClassifierRuntime([0.1])

    # ``True`` is int-like in Python and would silently mean "1"; reject it so
    # a boolean flag can never be mistaken for a confirmation count.
    for value in (0, -1, 1.5, True, "2", None):
        with pytest.raises(
            LearnerCompositionError, match="success_confirmations"
        ):
            build_actor_service(
                assembly=assembly,
                classifier=classifier,
                success_confirmations=value,
            )


def _wait_for_learner_step(worker, expected: int, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.status.learner_step >= expected:
            return
        time.sleep(0.001)
    raise AssertionError(
        f"learner did not reach step {expected}; status={worker.status}"
    )


def test_worker_default_utd_one_waits_for_each_post_warmup_insert(tmp_path):
    assembly = _compose_fresh(tmp_path / "worker-utd-one")
    insert_count = {"value": 3}
    worker = LearnerWorker(
        assembly.learner,
        replay_insert_count=lambda: insert_count["value"],
        target_learner_step=3,
        poll_interval=0.001,
    )

    worker.start()
    time.sleep(0.02)
    assert worker.status.learner_step == 0

    insert_count["value"] = 4
    _wait_for_learner_step(worker, 1)
    time.sleep(0.02)
    assert worker.status.learner_step == 1

    insert_count["value"] = 5
    _wait_for_learner_step(worker, 2)
    time.sleep(0.02)
    assert worker.status.learner_step == 2

    insert_count["value"] = 6
    assert worker.wait(timeout=2.0)
    assert worker.join(timeout=2.0)
    assert worker.status.state == "completed"
    assert worker.status.learner_step == 3
    assert worker.status.gradient_step == 6


def test_worker_utd_two_allows_two_steps_at_warmup_boundary(tmp_path):
    config = LearnerConfig(
        batch_size=4,
        training_starts=4,
        utd_ratio=2,
        publish_period=2,
        checkpoint_period=100,
    )
    assembly = _compose_fresh(tmp_path / "worker-utd-two", config=config)
    worker = LearnerWorker(
        assembly.learner,
        replay_insert_count=lambda: 4,
        target_learner_step=2,
        poll_interval=0.001,
    )

    worker.start()
    assert worker.wait(timeout=2.0)
    assert worker.join(timeout=2.0)
    assert worker.status.state == "completed"
    assert worker.status.learner_step == 2
    assert worker.status.gradient_step == 4


def test_production_service_shares_policy_and_strict_ingress_over_grpc(
    tmp_path,
):
    class Extractor:
        def __call__(self, observation):
            marker = float(observation["cam1"][0, 0, 0, 0])
            return {
                "state": observation["state"].copy(),
                "cam1": np.full(
                    FROZEN_TRUNK_FEATURE_SHAPE, marker, np.float32
                ),
                "cam2": np.full(
                    FROZEN_TRUNK_FEATURE_SHAPE, marker + 1.0, np.float32
                ),
            }

    raw_ingress = FeatureReplayIngress(
        feature_extractor=Extractor(),
        replay_capacity=8,
        intervention_capacity=4,
        available_memory_bytes=16 * 1024**2,
        memory_reserve_bytes=0,
    )
    ingress = FaultGatedReplayIngress(raw_ingress)
    assembly = _compose_fresh(
        tmp_path / "grpc",
        ingress=ingress,
    )
    classifier = ScriptedRewardClassifierRuntime([0.1, 0.1])
    service = build_actor_service(assembly=assembly, classifier=classifier)

    assert service._sample_action is assembly.policy_runtime
    assert service._accept_data is assembly.ingress
    server, port = create_grpc_server(service)
    server.start()
    client = GrpcActorNetwork(
        f"127.0.0.1:{port}",
        actor_id="composition-test",
        timeout_s=1.0,
        max_response_age_s=2.0,
        expected_observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
    )
    try:
        info = client.get_server_info()
        assert info.reward_authority == "server_classifier"
        assert info.model_id == assembly.policy_runtime.model_id
        first = client.begin_episode(
            synthetic_observation(0),
            run_id="composition-run",
            session_id="session-0",
            episode_id=0,
            observation_id="o0",
            timestamp_ns=1_000,
            deterministic=True,
        )
        assert first.policy_version == 0
        accepted = build_data(
            actor_id="composition-test",
            run_id="composition-run",
            session_id="session-0",
            transition_id="composition-run:0",
            env_step=0,
            timestamp_ns=1_000,
            policy_version=first.policy_version,
            policy_action=first.action,
            episode_id=0,
            step_id=0,
            observation_id="o0",
            next_observation_id="o1",
            reward=0.0,
            done=True,
            truncated=False,
            info={"intervened": 0, "grasp_penalty": -0.02},
        )
        result = client.step(
            synthetic_observation(1),
            next_observation_id="o1",
            next_timestamp_ns=1_001,
            data=accepted,
            request_action=False,
            deterministic=True,
        )
        assert result.ack.accepted
        assert client.get_buffer_status().replay_size == 1
        sampled = raw_ingress.sample_replay(1)
        assert sampled["observations"]["cam1"].shape == (1, 1, 4, 4, 512)
        assert sampled["next_observations"]["cam2"].dtype == np.float32

        assembly.policy_runtime.publish(
            assembly.learner.agent.state.params,
            learner_step=1,
        )
        second = client.begin_episode(
            synthetic_observation(2),
            run_id="composition-run",
            session_id="session-1",
            episode_id=1,
            observation_id="o2",
            timestamp_ns=1_002,
            deterministic=True,
        )
        assert second.policy_version == 1
        missing_penalty = build_data(
            actor_id="composition-test",
            run_id="composition-run",
            session_id="session-1",
            transition_id="composition-run:1",
            env_step=1,
            timestamp_ns=1_002,
            policy_version=second.policy_version,
            policy_action=second.action,
            episode_id=1,
            step_id=0,
            observation_id="o2",
            next_observation_id="o3",
            reward=0.0,
            done=True,
            truncated=False,
            info={"intervened": 0},
        )
        with pytest.raises(ActorTransportError, match="grasp_penalty"):
            client.step(
                synthetic_observation(3),
                next_observation_id="o3",
                next_timestamp_ns=1_003,
                data=missing_penalty,
                request_action=False,
                deterministic=True,
            )

        assert raw_ingress.status().replay_size == 1
        assert ingress.fault is not None
        assert "grasp_penalty" in ingress.fault.detail
        assert service.health()[1] is False
        action, version = assembly.policy_runtime(
            synthetic_observation(4), deterministic=True
        )
        assert version == 1
        assert np.isfinite(action).all()
    finally:
        client.close()
        server.stop(grace=0).wait()
