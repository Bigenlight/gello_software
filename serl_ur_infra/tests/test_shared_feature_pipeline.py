"""One image encoder per step, shared by replay and the policy.

The laptop uploads one 128x128 image pair; the server must run the frozen
trunk ONCE on it and hand the same feature map to everyone who needs it.
Before this pipeline existed a step ran the trunk six times: twice for O(t),
twice for O(t+1), and twice more inside the policy.

These tests count the encoder calls end to end through the real
``ActorSessionService`` + ``FeatureReplayIngress`` pair, so "shared" is a
measurement rather than a claim.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from test_feature_replay import (  # noqa: E402
    _FeatureExtractor,
    _data,
    _ingress,
    _observation,
)

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    ActorSessionService,
    BeginEpisodeCommand,
    ObservationPacket,
    StepCommand,
    TransitionOutcome,
)
from ur_env.learner import FaultGatedReplayIngress  # noqa: E402
from ur_env.learner.config import FROZEN_TRUNK_FEATURE_SHAPE  # noqa: E402


_BASE_NS = 1_700_000_000_000_000_000


def _finalize(data, sidecar):
    """Echo the transition unchanged; reward authority is not under test."""

    transition = data["transition"]
    return data, TransitionOutcome(
        transition_id=data["meta"]["transition_id"],
        reward=float(transition["rewards"]),
        mask=float(transition["masks"]),
        done=bool(transition["dones"]),
        truncated=bool(transition["truncated"]),
        success=bool(transition["success"]),
        classifier_evaluated=bool(transition["classifier_evaluated"]),
        classifier_probability=float(transition["classifier_probability"]),
        classifier_threshold=float(transition["classifier_threshold"]),
        reward_model_id=str(transition["reward_model_id"]),
    )


class _Harness:
    def __init__(self, *, steps: int) -> None:
        self.extractor = _FeatureExtractor()
        self.raw = _ingress(self.extractor, replay_capacity=16)
        self.ingress = FaultGatedReplayIngress(self.raw)
        self.policy_inputs: list[tuple[int, ...]] = []
        self.service = ActorSessionService(
            self._sample,
            accept_data=self.ingress,
            finalize_transition=_finalize,
        )
        self.service.begin_episode(
            BeginEpisodeCommand(
                PROTOCOL_VERSION,
                "actor-0",
                "run-0",
                "session-0",
                0,
                1,
                10_000,
                ObservationPacket("observation-0", _BASE_NS, _observation(0)),
            )
        )
        self.after_begin = self.extractor.calls
        for index in range(steps):
            self._step(index)

    def _sample(self, observation, deterministic):
        self.policy_inputs.append(
            tuple(int(value) for value in np.asarray(observation["cam1"]).shape)
        )
        return np.zeros(7, dtype=np.float32), 0

    def _step(self, index: int) -> None:
        data = _data(index)
        data["meta"]["policy_action"] = np.zeros(7, dtype=np.float32)
        data["transition"]["actions"] = np.zeros(7, dtype=np.float32)
        self.service.step(
            StepCommand(
                PROTOCOL_VERSION,
                "actor-0",
                "run-0",
                "session-0",
                2 + index,
                10_000,
                data,
                ObservationPacket(
                    f"observation-{index + 1}",
                    _BASE_NS + index + 1,
                    _observation(index + 1),
                ),
                True,
            )
        )


def test_one_observation_is_encoded_per_step():
    """Not two: O(t) is the previous step's O(t+1), served from the cache."""

    steps = 4
    harness = _Harness(steps=steps)

    assert harness.after_begin == 1, "BeginEpisode encodes O(0) exactly once"
    assert harness.extractor.calls - harness.after_begin == steps
    assert harness.raw.trunk_extractions == steps + 1
    # Two hits per step: _encode finds BOTH O(t) and O(t+1) already computed,
    # so the lock this ingress shares with the learner contains no GPU work.
    assert harness.raw.observation_cache_hits == 2 * steps
    assert harness.ingress.status().replay_insert_count == steps


def test_the_policy_is_served_the_shared_feature_not_pixels():
    """A pixel-shaped policy input means the trunk ran a second time."""

    harness = _Harness(steps=3)

    assert harness.policy_inputs, "the policy was never called"
    assert set(harness.policy_inputs) == {FROZEN_TRUNK_FEATURE_SHAPE}
    for shape in harness.policy_inputs:
        assert shape[-3:] != (128, 128, 3)


def test_a_sink_that_cannot_encode_still_serves_pixels():
    """The receive-only server and test doubles keep the old path."""

    inputs: list[tuple[int, ...]] = []

    def sample(observation, deterministic):
        inputs.append(
            tuple(int(value) for value in np.asarray(observation["cam1"]).shape)
        )
        return np.zeros(7, dtype=np.float32), 0

    service = ActorSessionService(
        sample,
        accept_data=lambda data, intervened: None,
        finalize_transition=_finalize,
    )
    service.begin_episode(
        BeginEpisodeCommand(
            PROTOCOL_VERSION,
            "actor-0",
            "run-0",
            "session-0",
            0,
            1,
            10_000,
            ObservationPacket("observation-0", _BASE_NS, _observation(0)),
        )
    )

    assert inputs == [(1, 128, 128, 3)]


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_encoder_calls_scale_one_per_step(steps):
    harness = _Harness(steps=steps)
    assert harness.extractor.calls == steps + 1
