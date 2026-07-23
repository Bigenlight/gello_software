#!/usr/bin/env python3
"""Container health check for the remote Diffusion gRPC service."""

import os
import sys

import grpc

from policy_server import remote_diffusion_pb2
from policy_server import remote_diffusion_pb2_grpc


def main() -> int:
    target = os.environ.get("HEALTHCHECK_TARGET", "127.0.0.1:50051")
    try:
        with grpc.insecure_channel(target) as channel:
            grpc.channel_ready_future(channel).result(timeout=2.0)
            stub = remote_diffusion_pb2_grpc.RemoteDiffusionStub(channel)
            reply = stub.Health(remote_diffusion_pb2.HealthRequest(), timeout=2.0)
        if not reply.alive or not reply.ready:
            print(
                f"service unhealthy: alive={reply.alive} ready={reply.ready}: {reply.detail}",
                file=sys.stderr,
            )
            return 1
        return 0
    except Exception as exc:  # health checks must convert every failure to nonzero
        print(f"remote diffusion health check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
