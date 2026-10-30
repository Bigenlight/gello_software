#!/usr/bin/env python3
"""Run a summary-only synthetic test against an RLPD receive server."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))

from ur_env.actor_network import create_actor_network  # noqa: E402
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)
from ur_env.rlpd_receive_smoke import run_receive_smoke  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--actor-id", default=f"{socket.gethostname()}-receive-smoke")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--local-episode-steps", type=int, default=25)
    parser.add_argument("--intervention-period", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--max-response-age-s", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    network = create_actor_network(
        {
            "type": "grpc",
            "host": args.host,
            "port": args.port,
            "timeout_s": args.timeout_s,
            "max_response_age_s": args.max_response_age_s,
            "retry_count": 1,
            "observation_schema_hash": CANONICAL_OBSERVATION_SCHEMA_HASH,
        },
        actor_id=args.actor_id,
        action_shape=(7,),
    )
    try:
        result = run_receive_smoke(
            network,
            actor_id=args.actor_id,
            steps=args.steps,
            local_episode_steps=args.local_episode_steps,
            intervention_period=args.intervention_period,
        )
    finally:
        network.close()
    print(
        json.dumps(
            {
                "event": "rlpd_receive_smoke_passed",
                "run_id": result.run_id,
                "steps": result.steps,
                "episodes": result.episodes,
                "interventions": result.interventions,
                "classifier_successes": result.classifier_successes,
                "replay_insert_delta": result.replay_insert_delta,
                "intervention_insert_delta": result.intervention_insert_delta,
                "last_transition_id": result.last_transition_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
