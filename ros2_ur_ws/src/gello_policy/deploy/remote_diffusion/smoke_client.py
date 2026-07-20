#!/usr/bin/env python3
"""Send one synthetic observation through the running gRPC policy service."""

import json
import math
import os
import time
import uuid

import cv2
import grpc
import numpy as np

from policy_server import remote_diffusion_pb2 as pb
from policy_server import remote_diffusion_pb2_grpc as pb_grpc


def main() -> int:
    target = os.environ.get("SMOKE_TARGET", "127.0.0.1:50051")
    client_id = "container-smoke-client"
    session_id = str(uuid.uuid4())

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode synthetic JPEG")
    jpeg = encoded.tobytes()

    with grpc.insecure_channel(target) as channel:
        grpc.channel_ready_future(channel).result(timeout=3.0)
        stub = pb_grpc.RemoteDiffusionStub(channel)
        info = stub.GetServerInfo(pb.ServerInfoRequest(), timeout=3.0)
        if not info.ready or info.protocol_version != "1":
            raise RuntimeError(f"server is not compatible/ready: {info}")
        reset = stub.ResetEpisode(
            pb.ResetEpisodeRequest(client_id=client_id, session_id=session_id),
            timeout=3.0,
        )
        if not reset.ok:
            raise RuntimeError(f"reset failed: {reset.detail}")

        now_ns = time.monotonic_ns()
        frame = pb.ImageFrame(
            ros_stamp_ns=now_ns,
            encoding="jpeg",
            width=640,
            height=360,
            data=jpeg,
        )
        request = pb.ObservationRequest(
            protocol_version="1",
            client_id=client_id,
            session_id=session_id,
            request_id=1,
            created_monotonic_ns=now_ns,
            state=[3.106, -1.817, 1.653, -1.618, -1.628, -3.195, 0.0],
            cam1=frame,
            cam2=frame,
        )
        started = time.perf_counter()
        reply = next(stub.StreamActions(iter((request,)), timeout=10.0))
        round_trip_ms = (time.perf_counter() - started) * 1000.0

    if not reply.ok:
        raise RuntimeError(f"inference failed: {reply.error}")
    if reply.request_id != 1 or reply.session_id != session_id:
        raise RuntimeError("response identifiers do not match the request")
    if len(reply.action) != 7 or not all(math.isfinite(value) for value in reply.action):
        raise RuntimeError(f"invalid action: {list(reply.action)}")

    print(
        json.dumps(
            {
                "model_id": info.model_id,
                "checkpoint_revision": info.checkpoint_revision,
                "scheduler": info.scheduler,
                "num_inference_steps": info.num_inference_steps,
                "n_action_steps": info.n_action_steps,
                "action": list(reply.action),
                "chunk_refill": reply.chunk_refill,
                "remaining_chunk_actions": reply.remaining_chunk_actions,
                "preprocess_ms": reply.preprocess_ms,
                "inference_ms": reply.inference_ms,
                "total_server_ms": reply.total_server_ms,
                "round_trip_ms": round_trip_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("PASS: remote Diffusion gRPC inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
