"""Local RLPD actor loop for UR7e + GELLO.

The learner stays in upstream hil-serl.  This module owns the robot-side
transition boundary so the vendored actor does not need to be patched:

* every executed transition goes to the online replay datastore;
* human-intervention transitions also go to the intervention datastore;
* the pre-override policy action and an explicit 0/1 intervention label are
  retained in the transmitted transition and local pickle files.

The existing upstream learner ignores unknown transition keys, so its SAC
training schema remains unchanged.  A learner can opt into retaining the two
metadata keys later by extending its replay-buffer schema.
"""

from __future__ import annotations

import copy
import glob
import os
import pickle as pkl
from typing import Any, Mapping, MutableMapping, Optional, Protocol

import numpy as np


class TransitionStore(Protocol):
    """The subset of Agentlace's datastore API used by the actor."""

    def insert(self, transition: MutableMapping[str, Any]) -> None:
        ...


class PolicyClient(Protocol):
    def start_episode(self, session_id: Optional[str] = None) -> str:
        ...

    def get_action(
        self,
        observation: Any,
        *,
        observation_timestamp_ns: int,
        deterministic: bool = False,
    ) -> Any:
        ...


class TransitionClient(Protocol):
    def update(self) -> bool:
        ...

    def request(self, request_type: str, payload: dict[str, Any]) -> Any:
        ...


def build_transition(
    *,
    observation: Any,
    policy_action: np.ndarray,
    next_observation: Any,
    reward: Any,
    done: bool,
    info: Mapping[str, Any],
    timestamp_ns: int,
    policy_version: int = -1,
) -> dict[str, Any]:
    """Build one replay transition without losing intervention provenance.

    ``actions`` is always the action that caused ``next_observation``.
    ``policy_actions`` is the pre-override actor output.  The latter is
    counterfactual when ``intervened == 1``.

    GelloIntervention supplies explicit metadata.  The fallbacks retain
    compatibility with the original SpacemouseIntervention contract, where
    only ``info["intervene_action"]`` is present.
    """
    # Use the action sampled at the outermost actor boundary.  The
    # GelloIntervention wrapper normally sits *inside* RelativeFrame, so its
    # info["policy_action"] can be expressed in base-frame units while replay
    # actions must stay in the policy/action-space frame.
    requested_action = np.asarray(policy_action).copy()
    recorded_policy_action = requested_action.copy()

    has_intervention_action = "intervene_action" in info
    intervened = int(info.get("intervened", has_intervention_action))
    if intervened not in (0, 1):
        raise ValueError(f"intervened must be 0 or 1, got {intervened!r}")
    if bool(intervened) != has_intervention_action:
        raise ValueError(
            "inconsistent intervention metadata: intervened="
            f"{intervened}, intervene_action present={has_intervention_action}"
        )

    if intervened:
        executed_action = np.asarray(info["intervene_action"]).copy()
    else:
        executed_action = requested_action.copy()

    if executed_action.shape != recorded_policy_action.shape:
        raise ValueError(
            "executed and policy action shapes differ: "
            f"{executed_action.shape} != {recorded_policy_action.shape}"
        )

    if isinstance(timestamp_ns, (bool, np.bool_)) or not isinstance(
        timestamp_ns, (int, np.integer)
    ):
        raise TypeError("timestamp_ns must be an integer env timestamp")
    timestamp_ns = int(timestamp_ns)
    if timestamp_ns <= 0 or timestamp_ns > np.iinfo(np.int64).max:
        raise ValueError("timestamp_ns must be a positive signed 64-bit value")
    if isinstance(policy_version, (bool, np.bool_)) or not isinstance(
        policy_version, (int, np.integer)
    ):
        raise TypeError("policy_version must be an integer")
    policy_version = int(policy_version)
    if policy_version < -1 or policy_version > np.iinfo(np.int64).max:
        raise ValueError("policy_version must be -1 or a non-negative int64")

    transition = {
        "observations": observation,
        "actions": executed_action,
        "policy_actions": recorded_policy_action,
        "intervened": np.uint8(intervened),
        "next_observations": next_observation,
        "rewards": reward,
        "masks": 1.0 - bool(done),
        "dones": bool(done),
        # Observation-finalization time supplied by the env. It is metadata
        # for correlating/exporting robot data.
        # It deliberately stays outside observations["state"], so SERL never
        # flattens the timestamp into the policy input vector.
        "timestamp_ns": np.int64(timestamp_ns),
        # -1 denotes a locally sampled random exploration action.
        "policy_version": np.int64(policy_version),
    }
    if "grasp_penalty" in info:
        transition["grasp_penalty"] = info["grasp_penalty"]
    return transition


def route_transition(
    transition: MutableMapping[str, Any],
    replay_store: TransitionStore,
    intervention_store: TransitionStore,
) -> None:
    """Apply the original HIL-SERL online/intervention routing rule."""
    replay_store.insert(transition)
    if int(transition["intervened"]) == 1:
        intervention_store.insert(transition)


def _start_step(checkpoint_path: Optional[str]) -> int:
    """Return the first unsaved actor step, tolerating a new checkpoint dir."""
    if not checkpoint_path:
        return 0
    paths = glob.glob(os.path.join(checkpoint_path, "buffer", "transitions_*.pkl"))
    if not paths:
        return 0

    def step_number(path: str) -> int:
        filename = os.path.basename(path)
        return int(filename.removeprefix("transitions_").removesuffix(".pkl"))

    return max(step_number(path) for path in paths) + 1


def _dump_transitions(
    checkpoint_path: str,
    step: int,
    transitions: list[dict[str, Any]],
    intervention_transitions: list[dict[str, Any]],
) -> None:
    buffer_path = os.path.join(checkpoint_path, "buffer")
    intervention_buffer_path = os.path.join(checkpoint_path, "demo_buffer")
    os.makedirs(buffer_path, exist_ok=True)
    os.makedirs(intervention_buffer_path, exist_ok=True)

    with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
        pkl.dump(transitions, f)
    with open(
        os.path.join(intervention_buffer_path, f"transitions_{step}.pkl"), "wb"
    ) as f:
        pkl.dump(intervention_transitions, f)


def run_actor(
    policy_client: PolicyClient,
    replay_store: TransitionStore,
    intervention_store: TransitionStore,
    transition_client: TransitionClient,
    env: Any,
    *,
    config: Any,
    checkpoint_path: Optional[str],
) -> None:
    """Run a robot-only actor with server-side policy inference.

    The policy RPC is synchronous and latency-sensitive. Transition insertion
    is local and lossless; the caller may run Agentlace's asynchronous update
    worker to batch uploads independently.
    """
    import tqdm
    from serl_launcher.utils.timer_utils import Timer

    transitions: list[dict[str, Any]] = []
    intervention_transitions: list[dict[str, Any]] = []
    obs, reset_info = env.reset()
    if "timestamp_ns" not in reset_info:
        raise RuntimeError("env.reset() info is missing timestamp_ns")
    observation_timestamp_ns = int(reset_info["timestamp_ns"])
    policy_client.start_episode()

    timer = Timer()
    running_return = 0.0
    intervention_count = 0
    intervention_steps = 0
    was_intervening = False

    start_step = _start_step(checkpoint_path)
    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                policy_action = env.action_space.sample()
                policy_version = -1
            else:
                result = policy_client.get_action(
                    obs,
                    observation_timestamp_ns=observation_timestamp_ns,
                    deterministic=False,
                )
                policy_action = np.asarray(result.action)
                policy_version = int(result.policy_version)

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(policy_action)
            if "timestamp_ns" not in info:
                raise RuntimeError("env.step() info is missing timestamp_ns")
            transition = build_transition(
                observation=obs,
                policy_action=policy_action,
                next_observation=next_obs,
                reward=reward,
                done=done,
                info=info,
                timestamp_ns=info["timestamp_ns"],
                policy_version=policy_version,
            )
            route_transition(transition, replay_store, intervention_store)
            transitions.append(copy.deepcopy(transition))

            intervened = bool(transition["intervened"])
            if intervened:
                intervention_transitions.append(copy.deepcopy(transition))
                intervention_steps += 1
                if not was_intervening:
                    intervention_count += 1
            was_intervening = intervened

            running_return += reward
            obs = next_obs
            observation_timestamp_ns = int(info["timestamp_ns"])

            if done or truncated:
                stats_info = dict(info)
                for key in (
                    "intervene_action",
                    "policy_action",
                    "intervened",
                    "timestamp_ns",
                    "left",
                    "right",
                ):
                    stats_info.pop(key, None)
                episode_info = stats_info.setdefault("episode", {})
                episode_info["intervention_count"] = intervention_count
                episode_info["intervention_steps"] = intervention_steps
                transition_client.request(
                    "send-stats", {"environment": stats_info}
                )
                pbar.set_description(f"last return: {running_return}")

                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                was_intervening = False

                # Match upstream semantics: flush queued transitions while the
                # robot is between episodes, never in the control-step path.
                transition_client.update()
                obs, reset_info = env.reset()
                if "timestamp_ns" not in reset_info:
                    raise RuntimeError("env.reset() info is missing timestamp_ns")
                observation_timestamp_ns = int(reset_info["timestamp_ns"])
                policy_client.start_episode()

        if step > 0 and config.buffer_period > 0:
            if step % config.buffer_period == 0:
                if not checkpoint_path:
                    raise ValueError(
                        "checkpoint_path is required when buffer_period > 0"
                    )
                _dump_transitions(
                    checkpoint_path,
                    step,
                    transitions,
                    intervention_transitions,
                )
                transitions = []
                intervention_transitions = []

        timer.tock("total")
        if step % config.log_period == 0:
            transition_client.request(
                "send-stats", {"timer": timer.get_average_times()}
            )
