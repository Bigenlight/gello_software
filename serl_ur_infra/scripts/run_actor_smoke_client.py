#!/usr/bin/env python3
"""Send two synthetic transitions to the remote actor mock server."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))

from ur_env.actor_network import create_actor_network  # noqa: E402
from ur_env.actor_smoke import run_mock_smoke  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--actor-id", default=f"{socket.gethostname()}-smoke")
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--max-response-age-s", type=float, default=3.0)
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
        },
        actor_id=args.actor_id,
        action_shape=(7,),
    )
    try:
        result = run_mock_smoke(network, actor_id=args.actor_id)
    finally:
        network.close()
    print(
        json.dumps(
            {
                "event": "actor_smoke_passed",
                "model_id": result.server_info.model_id,
                "protocol_version": result.server_info.protocol_version,
                "schema_version": result.server_info.schema_version,
                "action_dim": result.server_info.action_dim,
                "run_id": result.run_id,
                "session_id": result.session_id,
                "transition_ids": list(result.transition_ids),
                "begin_round_trip_ms": round(result.begin_round_trip_ms, 3),
                "normal_step_round_trip_ms": round(
                    result.normal_step_round_trip_ms, 3
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
