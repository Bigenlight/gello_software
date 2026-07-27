"""Production learner composition, worker, and localhost service tests."""

from __future__ import annotations

from pathlib import Path
import sys
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
from test_rlpd_receive_server import _StoreFactory  # noqa: E402
from ur_env.actor_network import ActorTransportError  # noqa: E402
from ur_env.actor_smoke import synthetic_observation  # noqa: E402
from ur_env.grpc_actor_transport import (  # noqa: E402
    GrpcActorNetwork,
    create_grpc_server,
)
from ur_env.learner import (  # noqa: E402
    CanonicalTransitionPool,
    CheckpointCorruptError,
    CheckpointManager,
    FaultGatedReplayIngress,
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
from ur_env.rlpd_receive_server import (  # noqa: E402
    ReplayIngress,
    ScriptedRewardClassifierRuntime,
)


_FINGERPRINT = LearnerFingerprint(
    document={"test": "production-composition"},
    sha256="0" * 64,
)


class _PoolIngress:
    """Strict in-memory source using the existing canonical pool helper."""

    require_grasp_penalty = True

    def __init__(self, replay_count: int = 4, intervention_count: int = 1):
        self.replay = CanonicalTransitionPool(
            [_transition(index) for index in range(replay_count)], seed=11
        )
        self.interventions = CanonicalTransitionPool(
            [
                _transition(100 + index)
                for index in range(intervention_count)
            ],
            seed=12,
        )

    def status(self):
        return SimpleNamespace(
            replay_size=len(self.replay),
            intervention_size=len(self.interventions),
        )

    def sample_replay(self, batch_size: int, **kwargs):
        del kwargs
        return self.replay.sample(batch_size)

    def sample_intervention(self, batch_size: int, **kwargs):
        del kwargs
        return self.interventions.sample(batch_size)


def _offline_pool() -> CanonicalTransitionPool:
    return CanonicalTransitionPool([_transition(200)], seed=13)


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
    assembly = _compose_fresh(
        tmp_path / "worker-fault",
        agent=_agent(fail_at=3),
    )
    worker = LearnerWorker(
        assembly.learner,
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


def test_production_service_shares_policy_and_strict_ingress_over_grpc(
    tmp_path,
):
    raw_ingress = ReplayIngress(
        replay_capacity=8,
        intervention_capacity=4,
        store_factory=_StoreFactory(),
        learner_mode=True,
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
