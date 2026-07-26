"""Robot-laptop loop for one-RPC-per-step remote HIL-SERL execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
import pickle
import time
from typing import Any, Callable, Mapping, Optional
import uuid

import numpy as np

from ur_env.actor_network import (
    SCHEMA_VERSION,
    ActorNetwork,
    ActorProtocolError,
    validate_action,
    validate_counter,
    validate_timestamp_ns,
)


class EnvTimestampAdapter:
    """Attach wall-clock time at the outer environment observation boundary.

    An underlying environment-provided ``info['timestamp_ns']`` wins.  The
    fallback is captured immediately after ``reset``/``step`` returns, keeping
    timestamp generation in the environment adapter rather than in policy
    state or the policy observation vector.
    """

    def __init__(
        self, env: Any, *, wall_time_ns: Callable[[], int] = time.time_ns
    ) -> None:
        self.env = env
        self._wall_time_ns = wall_time_ns

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return observation, self._stamp(info)

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        observation, reward, done, truncated, info = self.env.step(action)
        return observation, reward, done, truncated, self._stamp(info)

    def close(self) -> None:
        self.env.close()

    def _stamp(self, info: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(info)
        timestamp_ns = result.get("timestamp_ns", self._wall_time_ns())
        result["timestamp_ns"] = np.int64(validate_timestamp_ns(timestamp_ns))
        return result


def build_data(
    *,
    actor_id: str,
    run_id: str,
    session_id: str,
    transition_id: str,
    env_step: int,
    timestamp_ns: int,
    policy_version: int,
    policy_action: Any,
    episode_id: int,
    step_id: int,
    observation_id: str,
    next_observation_id: str,
    reward: Any,
    done: bool,
    truncated: bool,
    info: Mapping[str, Any],
    action_shape: tuple[int, ...] = (7,),
) -> dict[str, Any]:
    """Build ``data{meta, transition}`` with action provenance intact.

    ``meta.policy_action`` is the server action before a human override.
    ``transition.actions`` is the action that physically produced the next
    observation.  Every data item goes to replay server-side; items labelled
    ``intervened == 1`` additionally go to the intervention buffer.
    """
    for name, value in (
        ("actor_id", actor_id),
        ("run_id", run_id),
        ("session_id", session_id),
        ("transition_id", transition_id),
        ("observation_id", observation_id),
        ("next_observation_id", next_observation_id),
    ):
        if not isinstance(value, str) or not value:
            raise ActorProtocolError(f"{name} is required")

    requested_action = validate_action(
        policy_action, action_shape=action_shape, name="policy_action"
    )
    has_intervention_action = "intervene_action" in info
    intervened_value = info.get("intervened", int(has_intervention_action))
    if isinstance(intervened_value, np.generic):
        intervened_value = intervened_value.item()
    if intervened_value not in (0, 1, False, True):
        raise ActorProtocolError("intervened must be 0 or 1")
    intervened = int(bool(intervened_value))
    if bool(intervened) != has_intervention_action:
        raise ActorProtocolError(
            "intervened label and intervene_action presence are inconsistent"
        )
    executed_action = validate_action(
        info["intervene_action"] if intervened else requested_action,
        action_shape=action_shape,
        name="executed_action",
    )

    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise ActorProtocolError("reward must be finite")
    if not isinstance(done, (bool, np.bool_)) or not isinstance(
        truncated, (bool, np.bool_)
    ):
        raise ActorProtocolError("done and truncated must be bool")
    if bool(done) and bool(truncated):
        raise ActorProtocolError("a transition cannot be done and truncated")

    transition: dict[str, Any] = {
        "episode_id": validate_counter(episode_id, name="episode_id"),
        "step_id": validate_counter(step_id, name="step_id"),
        "observation_id": observation_id,
        "actions": executed_action,
        "next_observation_id": next_observation_id,
        "rewards": reward_value,
        # Preserve upstream HIL-SERL semantics: truncation resets the episode
        # but is not a Bellman terminal.
        "masks": 0.0 if bool(done) else 1.0,
        "dones": bool(done),
        "truncated": bool(truncated),
    }
    if "grasp_penalty" in info:
        grasp_penalty = float(info["grasp_penalty"])
        if not np.isfinite(grasp_penalty):
            raise ActorProtocolError("grasp_penalty must be finite")
        transition["grasp_penalty"] = grasp_penalty

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "actor_id": actor_id,
            "session_id": session_id,
            "transition_id": transition_id,
            "env_step": validate_counter(env_step, name="env_step"),
            # This is O(t)'s environment timestamp, not O(t+1)'s timestamp.
            "timestamp_ns": validate_timestamp_ns(timestamp_ns),
            "policy_version": validate_counter(
                policy_version, name="policy_version"
            ),
            "policy_action": requested_action,
            "intervened": intervened,
        },
        "transition": transition,
    }


@dataclass(frozen=True)
class ActorRunSummary:
    run_id: str
    env_steps: int
    episodes_started: int
    intervention_steps: int


def _dump_data(
    checkpoint_path: str,
    run_id: str,
    env_step: int,
    replay_data: list[dict[str, Any]],
    intervention_data: list[dict[str, Any]],
) -> None:
    replay_dir = os.path.join(checkpoint_path, "actor_data", run_id, "replay")
    intervention_dir = os.path.join(
        checkpoint_path, "actor_data", run_id, "intervention"
    )
    os.makedirs(replay_dir, exist_ok=True)
    os.makedirs(intervention_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"data_{env_step}.pkl"), "wb") as file:
        pickle.dump(replay_data, file)
    with open(
        os.path.join(intervention_dir, f"data_{env_step}.pkl"), "wb"
    ) as file:
        pickle.dump(intervention_data, file)


def run_remote_actor(
    network: ActorNetwork,
    env: Any,
    *,
    config: Any,
    actor_id: str,
    checkpoint_path: Optional[str] = None,
    run_id: Optional[str] = None,
    session_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> ActorRunSummary:
    """Run synchronous remote inference and lossless transition delivery."""
    max_steps = validate_counter(config.max_steps, name="max_steps")
    if max_steps <= 0:
        raise ValueError("config.max_steps must be positive")
    if int(config.random_steps) != 0:
        raise ValueError(
            "remote actor protocol v1 requires config.random_steps == 0"
        )
    buffer_period = int(getattr(config, "buffer_period", 0))
    if buffer_period < 0:
        raise ValueError("config.buffer_period must be non-negative")
    if buffer_period and not checkpoint_path:
        raise ValueError("checkpoint_path is required when buffer_period > 0")

    run_id = run_id or uuid.uuid4().hex
    if not actor_id or not run_id:
        raise ValueError("actor_id and run_id are required")
    action_shape = tuple(int(dim) for dim in env.action_space.shape)
    if action_shape != (7,):
        raise ValueError(f"protocol v1 requires action shape (7,), got {action_shape}")

    replay_data: list[dict[str, Any]] = []
    intervention_data: list[dict[str, Any]] = []
    total_intervention_steps = 0
    episodes_started = 0
    episode_id = 0
    step_id = 0

    observation, reset_info = env.reset()
    source_timestamp_ns = validate_timestamp_ns(reset_info.get("timestamp_ns"))
    session_id = session_id_factory()
    if not session_id:
        raise ValueError("session_id_factory returned an empty ID")
    observation_id = f"{session_id}:0"
    action_result = network.begin_episode(
        observation,
        run_id=run_id,
        session_id=session_id,
        episode_id=episode_id,
        observation_id=observation_id,
        timestamp_ns=source_timestamp_ns,
    )
    episodes_started += 1

    for env_step in range(max_steps):
        policy_action = validate_action(
            action_result.action,
            action_shape=action_shape,
            name="network policy action",
        )
        next_observation, reward, done, truncated, info = env.step(policy_action)
        next_timestamp_ns = validate_timestamp_ns(info.get("timestamp_ns"))
        next_observation_id = f"{session_id}:{step_id + 1}"
        data = build_data(
            actor_id=actor_id,
            run_id=run_id,
            session_id=session_id,
            transition_id=f"{run_id}:{env_step}",
            env_step=env_step,
            # O(t), deliberately not the just-returned O(t+1) time.
            timestamp_ns=source_timestamp_ns,
            policy_version=action_result.policy_version,
            policy_action=policy_action,
            episode_id=episode_id,
            step_id=step_id,
            observation_id=observation_id,
            next_observation_id=next_observation_id,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
            action_shape=action_shape,
        )
        terminal = bool(done) or bool(truncated)
        result = network.step(
            next_observation,
            next_observation_id=next_observation_id,
            next_timestamp_ns=next_timestamp_ns,
            data=data,
            request_action=not terminal,
        )
        # Reaching here proves the server ACKed this transition. In particular,
        # terminal reset can never happen after an unacknowledged Step.
        replay_data.append(copy.deepcopy(data))
        if data["meta"]["intervened"] == 1:
            intervention_data.append(copy.deepcopy(data))
            total_intervention_steps += 1

        if buffer_period and (env_step + 1) % buffer_period == 0:
            _dump_data(
                checkpoint_path,
                run_id,
                env_step,
                replay_data,
                intervention_data,
            )
            replay_data = []
            intervention_data = []

        if terminal:
            if env_step + 1 >= max_steps:
                break
            episode_id += 1
            step_id = 0
            observation, reset_info = env.reset()
            source_timestamp_ns = validate_timestamp_ns(
                reset_info.get("timestamp_ns")
            )
            session_id = session_id_factory()
            if not session_id:
                raise ValueError("session_id_factory returned an empty ID")
            observation_id = f"{session_id}:0"
            action_result = network.begin_episode(
                observation,
                run_id=run_id,
                session_id=session_id,
                episode_id=episode_id,
                observation_id=observation_id,
                timestamp_ns=source_timestamp_ns,
            )
            episodes_started += 1
            continue

        if result.action is None:
            raise ActorProtocolError("non-terminal Step returned no action")
        observation = next_observation
        observation_id = next_observation_id
        source_timestamp_ns = next_timestamp_ns
        action_result = result.action
        step_id += 1

    if checkpoint_path and replay_data:
        _dump_data(
            checkpoint_path,
            run_id,
            max_steps - 1,
            replay_data,
            intervention_data,
        )
    return ActorRunSummary(
        run_id=run_id,
        env_steps=max_steps,
        episodes_started=episodes_started,
        intervention_steps=total_intervention_steps,
    )
