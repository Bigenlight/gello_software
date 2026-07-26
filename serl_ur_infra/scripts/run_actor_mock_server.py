#!/usr/bin/env python3
"""Loopback-only zero-policy server for validating the actor gRPC contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))

from ur_env.actor_network import ActorSessionService  # noqa: E402
from ur_env.grpc_actor_transport import create_grpc_server  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    if args.host not in (
        "127.0.0.1",
        "localhost",
        "::1",
    ):
        raise ValueError(
            "mock server is loopback-only; use an SSH forward"
        )

    def zero_policy(observation, deterministic):
        del observation, deterministic
        return np.zeros(7, dtype=np.float32), 0

    service = ActorSessionService(zero_policy, model_id="mock-zero-policy")
    server, bound_port = create_grpc_server(
        service, bind_address=f"{args.host}:{args.port}"
    )
    server.start()
    print(
        f"[actor-mock] ready at {args.host}:{bound_port}; "
        "policy=zeros(7), loopback contract test only",
        flush=True,
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2.0).wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
