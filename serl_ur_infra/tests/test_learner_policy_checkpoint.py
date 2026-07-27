"""Versioned policy, learner fault isolation, checkpoint, and logging tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("jax")
pytest.importorskip("flax")
import flax
from flax.core import freeze
import jax
import jax.numpy as jnp


_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_INFRA, ".."))
sys.path.insert(0, _INFRA)

from ur_env.learner import (  # noqa: E402
    CanonicalTransitionPool,
    CheckpointCorruptError,
    CheckpointExistsError,
    CheckpointFingerprintError,
    CheckpointManager,
    HILSERLLearner,
    JsonlWandbLogger,
    LearnerConfig,
    LearnerFaultError,
    LearnerFingerprint,
    PolicyValidationError,
    RLPDBatchSampler,
    VersionedPolicyRuntime,
)


@flax.struct.dataclass
class _State:
    params: object
    target_params: object
    opt_state: object
    rng: object
    step: int


@dataclass(frozen=True)
class _Agent:
    state: _State
    fail_at_gradient: int | None = None

    def replace(self, **updates):
        return replace(self, **updates)

    def update(self, batch, *, networks_to_update):
        del batch, networks_to_update
        next_step = int(self.state.step) + 1
        value = self.state.params["weight"] + np.float32(0.1)
        if next_step == self.fail_at_gradient:
            value = jnp.full_like(value, jnp.nan)
        params = freeze({"weight": value})
        state = self.state.replace(
            params=params,
            target_params=params,
            opt_state=freeze({"moment": value}),
            rng=jax.random.fold_in(self.state.rng, next_step),
            step=next_step,
        )
        return self.replace(state=state), {"loss": jnp.square(value[0])}


def _agent(*, step: int = 0, value: float = 0.0, fail_at=None) -> _Agent:
    params = freeze({"weight": jnp.array([value], dtype=jnp.float32)})
    return _Agent(
        _State(
            params=params,
            target_params=params,
            opt_state=freeze({"moment": jnp.zeros(1, jnp.float32)}),
            rng=jax.random.PRNGKey(9),
            step=step,
        ),
        fail_at_gradient=fail_at,
    )


def _sample_action(params, observation, seed, deterministic):
    del observation
    action = jnp.zeros(7, dtype=jnp.float32)
    base = jnp.clip(params["weight"][0], -0.8, 0.8)
    if not deterministic:
        base = jnp.clip(base + jax.random.uniform(seed, (), minval=-0.05, maxval=0.05), -1, 1)
    return action.at[0].set(base)


def _observation(value: int) -> dict[str, np.ndarray]:
    return {
        "state": np.zeros((1, 19), np.float32),
        "cam1": np.full((1, 128, 128, 3), value, np.uint8),
        "cam2": np.full((1, 128, 128, 3), value, np.uint8),
    }


def _transition(value: int, penalty: float = -0.02) -> dict:
    return {
        "observations": _observation(value),
        "next_observations": _observation(value + 1),
        "actions": np.zeros(7, np.float32),
        "rewards": np.float32(0),
        "masks": np.float32(1),
        "grasp_penalty": np.float32(penalty),
    }


def _sampler() -> RLPDBatchSampler:
    return RLPDBatchSampler(
        online_replay=CanonicalTransitionPool(
            [_transition(index) for index in range(4)], seed=1
        ),
        offline_demos=CanonicalTransitionPool([_transition(10)], seed=2),
        online_interventions=CanonicalTransitionPool([_transition(20)], seed=3),
        batch_size=4,
        training_starts=4,
        seed=4,
    )


def test_policy_snapshot_is_atomic_monotonic_and_rejects_bad_candidates():
    agent = _agent()
    runtime = VersionedPolicyRuntime(agent, sample_action=_sample_action)
    candidate = freeze({"weight": jnp.array([0.4], jnp.float32)})

    version = runtime.publish(candidate, learner_step=50)
    action, served_version = runtime(_observation(0), deterministic=True)

    assert version == served_version == 1
    assert runtime.snapshot.params is candidate
    assert action.dtype == np.float32
    assert action.shape == (7,)
    assert action[-1] in (-1.0, 0.0, 1.0)
    stochastic, _ = runtime(_observation(0), deterministic=False)
    assert stochastic[0] != action[0]

    before = runtime.snapshot
    with pytest.raises(PolicyValidationError, match="non-finite"):
        runtime.publish(
            freeze({"weight": jnp.array([jnp.nan], jnp.float32)}),
            learner_step=100,
        )
    assert runtime.snapshot is before
    with pytest.raises(PolicyValidationError, match="increase"):
        runtime.publish(candidate, learner_step=50)
    assert runtime.snapshot is before


def test_learner_fault_keeps_last_published_policy():
    config = LearnerConfig(
        batch_size=4,
        training_starts=4,
        publish_period=2,
        checkpoint_period=100,
    )
    agent = _agent(fail_at=5)
    runtime = VersionedPolicyRuntime(agent, sample_action=_sample_action)
    learner = HILSERLLearner(
        agent=agent,
        sampler=_sampler(),
        publisher=runtime,
        config=config,
    )

    learner.train_once()
    second = learner.train_once()
    good_snapshot = runtime.snapshot
    assert second.published
    assert good_snapshot.policy_version == 1
    assert second.gradient_step == 4

    with pytest.raises(LearnerFaultError, match="last known good"):
        learner.train_once()
    assert learner.fault is not None
    assert runtime.snapshot is good_snapshot
    action, version = runtime(_observation(0), deterministic=True)
    assert version == 1
    assert np.isfinite(action).all()


def _fingerprint(config: LearnerConfig) -> LearnerFingerprint:
    return LearnerFingerprint.create(
        config=config,
        resnet_asset_path=Path(_REPO)
        / "third_party"
        / "hil-serl"
        / "examples"
        / "experiments"
        / "resnet10_params.pkl",
    )


def test_checkpoint_roundtrip_counters_rng_and_corruption(tmp_path):
    config = LearnerConfig(
        batch_size=4,
        training_starts=4,
        publish_period=2,
        checkpoint_period=4,
    )
    fingerprint = _fingerprint(config)
    manager = CheckpointManager(tmp_path / "checkpoints")
    agent = _agent(step=4, value=0.4)
    inference_rng = jax.random.PRNGKey(77)

    path = manager.save(
        agent=agent,
        learner_step=2,
        gradient_step=4,
        policy_version=1,
        inference_rng=inference_rng,
        fingerprint=fingerprint,
    )
    restored = manager.load(
        agent_template=_agent(), fingerprint=fingerprint, path=path
    )

    assert restored.learner_step == 2
    assert restored.gradient_step == 4
    assert restored.policy_version == 1
    np.testing.assert_array_equal(restored.inference_rng, inference_rng)
    original_runtime = VersionedPolicyRuntime(
        agent,
        policy_version=1,
        learner_step=2,
        inference_rng=inference_rng,
        sample_action=_sample_action,
    )
    restored_runtime = VersionedPolicyRuntime(
        restored.agent,
        policy_version=restored.policy_version,
        learner_step=restored.learner_step,
        inference_rng=restored.inference_rng,
        sample_action=_sample_action,
    )
    np.testing.assert_array_equal(
        original_runtime(_observation(0), deterministic=True)[0],
        restored_runtime(_observation(0), deterministic=True)[0],
    )
    with pytest.raises(CheckpointExistsError):
        manager.save(
            agent=agent,
            learner_step=2,
            gradient_step=4,
            policy_version=1,
            inference_rng=inference_rng,
            fingerprint=fingerprint,
        )

    state_path = path / "agent_state.msgpack"
    payload = bytearray(state_path.read_bytes())
    payload[-1] ^= 0x01
    state_path.write_bytes(payload)
    with pytest.raises(CheckpointCorruptError, match="checksum"):
        manager.load(agent_template=_agent(), fingerprint=fingerprint, path=path)


def test_checkpoint_rejects_fingerprint_mismatch(tmp_path):
    config = LearnerConfig(
        batch_size=4,
        training_starts=4,
        publish_period=2,
        checkpoint_period=4,
    )
    manager = CheckpointManager(tmp_path)
    path = manager.save(
        agent=_agent(step=4),
        learner_step=2,
        gradient_step=4,
        policy_version=1,
        inference_rng=jax.random.PRNGKey(1),
        fingerprint=_fingerprint(config),
    )
    different = LearnerConfig(
        seed=43,
        batch_size=4,
        training_starts=4,
        publish_period=2,
        checkpoint_period=4,
    )
    with pytest.raises(CheckpointFingerprintError, match="mismatch"):
        manager.load(
            agent_template=_agent(), fingerprint=_fingerprint(different), path=path
        )


class _FakeRun:
    def __init__(self):
        self.records = []
        self.finished = False

    def log(self, record, step):
        self.records.append((record, step))

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self):
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


def test_jsonl_and_wandb_receive_the_same_structured_event(tmp_path):
    wandb = _FakeWandb()
    path = tmp_path / "learner.jsonl"
    logger = JsonlWandbLogger(
        path,
        wandb_dir=tmp_path,
        config={"seed": 42},
        wandb_module=wandb,
    )
    logger.log(
        "learner_update",
        learner_step=3,
        gradient_step=6,
        metrics={"loss": jnp.array(1.25)},
    )
    logger.close()

    record = json.loads(path.read_text().strip())
    assert record["event"] == "learner_update"
    assert record["metrics"]["loss"] == 1.25
    assert wandb.kwargs["mode"] == "offline"
    assert wandb.run.records[0][0] == record
    assert wandb.run.records[0][1] == 3
    assert wandb.run.finished
