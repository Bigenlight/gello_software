#!/usr/bin/env python3
"""Send bounded synthetic episodes through the real laptop gRPC client.

This is an acceptance tool, not a robot actor and not a demo generator.  It
must target a learner started with ``--synthetic-e2e``.  Every transition is
transported as canonical raw pixels, finalized by the server classifier, and
acknowledged only after feature replay insertion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import uuid

import numpy as np


_INFRA_ROOT = Path(__file__).resolve().parents[1]
if str(_INFRA_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFRA_ROOT))

from ur_env.actor_smoke import synthetic_observation  # noqa: E402
from ur_env.grpc_actor_transport import GrpcActorNetwork  # noqa: E402
from ur_env.learner.config import (  # noqa: E402
    FROZEN_TRUNK_SYNTHETIC_E2E_MODEL_REVISION,
)
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)
from ur_env.remote_actor import build_data  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send fake canonical episodes through the real actor gRPC "
            "transport to a bounded --synthetic-e2e learner."
        )
    )
    parser.add_argument("--target", default="127.0.0.1:50053")
    parser.add_argument("--actor-id", default="fake-e2e-actor")
    parser.add_argument("--run-id", default=f"fake-e2e-{uuid.uuid4().hex}")
    parser.add_argument("--transition-count", type=int, default=100)
    parser.add_argument("--expected-start-policy-version", type=int, required=True)
    parser.add_argument("--expected-reward-model-id", required=True)
    parser.add_argument("--grasp-penalty", type=float, default=-0.02)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-response-age-s", type=float, default=120.0)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("target", "actor_id", "run_id", "expected_reward_model_id"):
        if not isinstance(getattr(args, name), str) or not getattr(args, name):
            raise ValueError(f"{name} is required")
    for name in ("transition_count",):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    version = args.expected_start_policy_version
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("expected_start_policy_version must be non-negative")
    if not math.isfinite(args.grasp_penalty) or args.grasp_penalty > 0.0:
        raise ValueError("grasp_penalty must be finite and non-positive")
    for name in ("timeout_s", "max_response_age_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the bounded sender and return a tensor-free acceptance summary."""

    _validate_args(args)
    network = GrpcActorNetwork(
        args.target,
        actor_id=args.actor_id,
        timeout_s=args.timeout_s,
        max_response_age_s=args.max_response_age_s,
        expected_observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        expected_model_id=FROZEN_TRUNK_SYNTHETIC_E2E_MODEL_REVISION,
        expected_reward_authority="server_classifier",
        expected_reward_model_id=args.expected_reward_model_id,
    )
    base_timestamp = time.time_ns()
    if base_timestamp + args.transition_count * 2 > np.iinfo(np.int64).max:
        raise RuntimeError("acceptance timestamps would overflow int64")

    versions: list[int] = []
    success_count = 0
    begin_round_trip_ms: list[float] = []
    try:
        alive, ready, detail = network.health()
        if not alive or not ready:
            raise RuntimeError(
                "learner service is not ready: "
                f"alive={alive}, ready={ready}, detail={detail}"
            )
        initial_status = network.get_buffer_status()
        for index in range(args.transition_count):
            session_id = f"{args.run_id}-session-{index}"
            observation_id = f"{session_id}:0"
            next_observation_id = f"{session_id}:1"
            timestamp = base_timestamp + index * 2
            action = network.begin_episode(
                synthetic_observation(index),
                run_id=args.run_id,
                session_id=session_id,
                episode_id=index,
                observation_id=observation_id,
                timestamp_ns=timestamp,
                deterministic=True,
            )
            if index == 0 and action.policy_version != (
                args.expected_start_policy_version
            ):
                raise RuntimeError(
                    "server started from policy_version "
                    f"{action.policy_version}, expected "
                    f"{args.expected_start_policy_version}"
                )
            if action.policy_version < args.expected_start_policy_version:
                raise RuntimeError("server policy_version moved backwards")
            versions.append(action.policy_version)
            begin_round_trip_ms.append(action.round_trip_ms)
            transition_id = f"{args.run_id}:{index}"
            data = build_data(
                actor_id=args.actor_id,
                run_id=args.run_id,
                session_id=session_id,
                transition_id=transition_id,
                env_step=index,
                timestamp_ns=timestamp,
                policy_version=action.policy_version,
                policy_action=action.action,
                episode_id=index,
                step_id=0,
                observation_id=observation_id,
                next_observation_id=next_observation_id,
                reward=0.0,
                done=True,
                truncated=False,
                info={"intervened": 0, "grasp_penalty": args.grasp_penalty},
            )
            result = network.step(
                synthetic_observation(index + 1),
                next_observation_id=next_observation_id,
                next_timestamp_ns=timestamp + 1,
                data=data,
                request_action=False,
                deterministic=True,
            )
            if not result.ack.accepted or result.ack.transition_id != transition_id:
                raise RuntimeError(f"transition {transition_id!r} was not accepted")
            if result.action is not None or not result.outcome.terminal:
                raise RuntimeError(
                    f"transition {transition_id!r} was not finalized terminal"
                )
            success_count += int(result.outcome.success)

        final_status = network.get_buffer_status()
        replay_delta = (
            final_status.replay_insert_count
            - initial_status.replay_insert_count
        )
        if replay_delta != args.transition_count:
            raise RuntimeError(
                f"replay accepted {replay_delta} new transitions, expected "
                f"{args.transition_count}"
            )
        return {
            "event": "fake_e2e_actor_passed",
            "target": args.target,
            "run_id": args.run_id,
            "transition_count": args.transition_count,
            "replay_insert_delta": replay_delta,
            "replay_size": final_status.replay_size,
            "classifier_success_count": success_count,
            "first_policy_version": versions[0],
            "last_policy_version": versions[-1],
            "begin_round_trip_ms_max": max(begin_round_trip_ms),
            "begin_round_trip_ms_mean": float(np.mean(begin_round_trip_ms)),
        }
    finally:
        network.close()


def main(argv: list[str] | None = None) -> int:
    result = run(_parse_args(argv))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
