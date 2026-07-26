"""Local-loop tests for server-authoritative reward and termination."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    ActorSessionService,
    BeginEpisodeCommand,
    ObservationPacket,
    StepCommand,
    TransitionOutcome,
)
from ur_env.actor_smoke import synthetic_observation  # noqa: E402
from ur_env.remote_actor import run_remote_actor  # noqa: E402


class _ActionSpace:
    shape = (7,)


class _ServerSuccessThenLocalDoneEnv:
    action_space = _ActionSpace()

    def __init__(self):
        self.reset_count = 0
        self.step_count = 0

    def reset(self):
        self.reset_count += 1
        return synthetic_observation(self.step_count), {
            "timestamp_ns": np.int64(1_000 + self.step_count)
        }

    def step(self, action):
        del action
        self.step_count += 1
        local_done = self.step_count == 2
        return (
            synthetic_observation(self.step_count),
            float(local_done),
            local_done,
            False,
            {
                "timestamp_ns": np.int64(1_000 + self.step_count),
                "intervened": 0,
            },
        )


class _InProcessNetwork:
    def __init__(self, service):
        self._service = service
        self._run_id = ""
        self._session_id = ""
        self._request_id = 1
        self._clock = 10_000

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
        self._run_id = run_id
        self._session_id = session_id
        self._request_id = 2
        self._clock += 1
        return self._service.begin_episode(
            BeginEpisodeCommand(
                PROTOCOL_VERSION,
                "actor",
                run_id,
                session_id,
                episode_id,
                1,
                self._clock,
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
        self._clock += 1
        result = self._service.step(
            StepCommand(
                PROTOCOL_VERSION,
                "actor",
                self._run_id,
                self._session_id,
                self._request_id,
                self._clock,
                data,
                ObservationPacket(
                    next_observation_id,
                    next_timestamp_ns,
                    next_observation,
                ),
                request_action,
                deterministic,
            )
        )
        self._request_id += 1
        return result


def test_actor_resets_on_server_classifier_success_and_keeps_final_values():
    accepted = []

    def finalize(data):
        transition = data["transition"]
        success = int(data["meta"]["env_step"]) == 0
        if success:
            transition.update(
                rewards=1.0,
                masks=0.0,
                dones=True,
                truncated=False,
            )
        return data, TransitionOutcome(
            transition_id=data["meta"]["transition_id"],
            reward=float(transition["rewards"]),
            mask=float(transition["masks"]),
            done=bool(transition["dones"]),
            truncated=bool(transition["truncated"]),
            success=success,
            classifier_evaluated=True,
            classifier_probability=0.9 if success else 0.1,
            classifier_threshold=0.85,
            reward_model_id="scripted-reward",
        )

    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        finalize_transition=finalize,
        accept_data=lambda data, intervened: accepted.append(data),
    )
    sessions = iter(("session-0", "session-1"))
    env = _ServerSuccessThenLocalDoneEnv()

    summary = run_remote_actor(
        _InProcessNetwork(service),
        env,
        config=SimpleNamespace(max_steps=2, random_steps=0, buffer_period=0),
        actor_id="actor",
        run_id="run",
        session_id_factory=lambda: next(sessions),
    )

    assert summary.episodes_started == 2
    assert env.reset_count == 2
    assert service.inference_count == 2
    assert len(accepted) == 2
    assert accepted[0]["transition"]["rewards"] == 1.0
    assert accepted[0]["transition"]["masks"] == 0.0
    assert accepted[0]["transition"]["dones"] is True
    assert accepted[0]["transition"]["truncated"] is False
