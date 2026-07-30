"""Synthetic client for the classifier + replay receive-server milestone."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional
import uuid

import numpy as np

from ur_env.actor_network import ActorNetwork, ActorProtocolError
from ur_env.actor_smoke import synthetic_observation
from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH
from ur_env.remote_actor import build_data


@dataclass(frozen=True)
class ReceiveSmokeResult:
    run_id: str
    steps: int
    episodes: int
    interventions: int
    classifier_successes: int
    replay_insert_delta: int
    intervention_insert_delta: int
    last_transition_id: str


def run_receive_smoke(
    network: ActorNetwork,
    *,
    actor_id: str,
    steps: int = 100,
    local_episode_steps: int = 25,
    intervention_period: int = 10,
    run_id: Optional[str] = None,
    wall_time_ns: Callable[[], int] = time.time_ns,
) -> ReceiveSmokeResult:
    """Exercise inference, classification, ACK, routing, and status counters."""
    if not actor_id:
        raise ValueError("actor_id is required")
    if steps <= 0 or local_episode_steps <= 0 or intervention_period <= 0:
        raise ValueError("smoke counts must be positive")

    alive, ready, detail = network.health()
    if not alive or not ready:
        raise RuntimeError(
            f"receive server is not ready: alive={alive}, ready={ready}, "
            f"detail={detail}"
        )
    server_info = network.get_server_info()
    if server_info.observation_schema_hash != CANONICAL_OBSERVATION_SCHEMA_HASH:
        raise ActorProtocolError("receive server advertises the wrong observation schema")
    if server_info.reward_authority != "server_classifier":
        raise ActorProtocolError(
            "receive smoke requires reward_authority='server_classifier'"
        )
    before = network.get_buffer_status()

    run_id = run_id or f"receive-smoke-{uuid.uuid4().hex}"
    base_timestamp = int(wall_time_ns())
    if base_timestamp <= 0 or base_timestamp > np.iinfo(np.int64).max - steps - 1:
        raise ValueError("wall_time_ns is outside the usable signed-int64 range")

    episode_id = 0
    episode_step = 0
    episodes = 1
    interventions = 0
    classifier_successes = 0
    session_id = uuid.uuid4().hex
    observation_id = f"{session_id}:0"
    source_timestamp = base_timestamp
    action = network.begin_episode(
        synthetic_observation(0),
        run_id=run_id,
        session_id=session_id,
        episode_id=episode_id,
        observation_id=observation_id,
        timestamp_ns=source_timestamp,
        deterministic=True,
    )

    last_transition_id = ""
    for env_step in range(steps):
        next_observation_id = f"{session_id}:{episode_step + 1}"
        next_timestamp = base_timestamp + env_step + 1
        last_transition_id = f"{run_id}:{env_step}"
        intervened = int(env_step % intervention_period == intervention_period - 1)
        interventions += intervened
        local_truncated = bool(
            episode_step + 1 >= local_episode_steps or env_step + 1 == steps
        )
        info = {"intervened": intervened}
        if intervened:
            info["intervene_action"] = np.full(7, -0.25, dtype=np.float32)
        data = build_data(
            actor_id=actor_id,
            run_id=run_id,
            session_id=session_id,
            transition_id=last_transition_id,
            env_step=env_step,
            timestamp_ns=source_timestamp,
            policy_version=action.policy_version,
            policy_action=action.action,
            # This headless receive smoke explicitly covers the automatic
            # classifier reward contract.  Real operator sessions default to
            # MANUAL and stamp their live selection instead.
            auto_success=True,
            episode_id=episode_id,
            step_id=episode_step,
            observation_id=observation_id,
            next_observation_id=next_observation_id,
            reward=0.0,
            done=False,
            truncated=local_truncated,
            info=info,
        )
        result = network.step(
            synthetic_observation(env_step + 1),
            next_observation_id=next_observation_id,
            next_timestamp_ns=next_timestamp,
            data=data,
            request_action=not local_truncated,
            deterministic=True,
        )
        if result.outcome.transition_id != last_transition_id:
            raise ActorProtocolError("receive smoke got the wrong outcome ID")
        classifier_successes += int(result.outcome.success)
        terminal = result.outcome.done or result.outcome.truncated
        if terminal:
            if result.action is not None:
                raise ActorProtocolError("terminal outcome returned an action")
            if env_step + 1 == steps:
                break
            episode_id += 1
            episode_step = 0
            episodes += 1
            session_id = uuid.uuid4().hex
            observation_id = f"{session_id}:0"
            source_timestamp = next_timestamp
            action = network.begin_episode(
                synthetic_observation(env_step + 1),
                run_id=run_id,
                session_id=session_id,
                episode_id=episode_id,
                observation_id=observation_id,
                timestamp_ns=source_timestamp,
                deterministic=True,
            )
            continue

        if result.action is None:
            raise ActorProtocolError("non-terminal outcome returned no action")
        episode_step += 1
        observation_id = next_observation_id
        source_timestamp = next_timestamp
        action = result.action

    after = network.get_buffer_status()
    replay_delta = after.replay_insert_count - before.replay_insert_count
    intervention_delta = (
        after.intervention_insert_count - before.intervention_insert_count
    )
    if replay_delta != steps:
        raise ActorProtocolError(
            f"replay insert delta is {replay_delta}, expected {steps}"
        )
    if intervention_delta != interventions:
        raise ActorProtocolError(
            "intervention insert delta is "
            f"{intervention_delta}, expected {interventions}"
        )
    if (
        after.last_transition_id != last_transition_id
        or after.last_env_step != steps - 1
    ):
        raise ActorProtocolError("buffer status did not advance to the final transition")

    return ReceiveSmokeResult(
        run_id=run_id,
        steps=steps,
        episodes=episodes,
        interventions=interventions,
        classifier_successes=classifier_successes,
        replay_insert_delta=replay_delta,
        intervention_insert_delta=intervention_delta,
        last_transition_id=last_transition_id,
    )
