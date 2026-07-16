#!/usr/bin/env python3
"""Dockerized gRPC wrapper around the validated JOINT Diffusion engine.

This service has no ROS imports and no route to the robot controller. The robot
laptop remains the sole owner of observations, action safety, watchdogs, and motion.
"""

from __future__ import annotations

from concurrent import futures
import math
import os
import threading
from typing import Iterator

import grpc
import numpy as np

from .diffusion_server import DiffusionInferenceEngine, resolve_device
from . import remote_diffusion_pb2 as pb
from . import remote_diffusion_pb2_grpc as pb_grpc
from . import zmq_protocol as dimensions


PROTOCOL_VERSION = "1"
DEFAULT_BIND_ADDRESS = "0.0.0.0:50051"
DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_JPEG_BYTES = 4 * 1024 * 1024


class RemoteDiffusionService(pb_grpc.RemoteDiffusionServicer):
    """Single-policy, single-active-session inference service."""

    def __init__(
        self,
        engine: DiffusionInferenceEngine,
        *,
        model_id: str,
        checkpoint_revision: str,
        scheduler: str,
        num_inference_steps: int,
        n_action_steps: int,
        max_jpeg_bytes: int,
    ) -> None:
        self._engine = engine
        self._model_id = model_id
        self._checkpoint_revision = checkpoint_revision
        self._scheduler = scheduler
        self._num_inference_steps = num_inference_steps
        self._n_action_steps = n_action_steps
        self._max_jpeg_bytes = max_jpeg_bytes
        self._lock = threading.Lock()
        self._active_client_id = ""
        self._active_session_id = ""

    def Health(self, request, context):  # noqa: N802 - generated gRPC API
        return pb.HealthReply(alive=True, ready=True, detail="model warm and ready")

    def GetServerInfo(self, request, context):  # noqa: N802
        return pb.ServerInfoReply(
            ready=True,
            protocol_version=PROTOCOL_VERSION,
            model_id=self._model_id,
            checkpoint_revision=self._checkpoint_revision,
            scheduler=self._scheduler,
            num_inference_steps=self._num_inference_steps,
            n_action_steps=self._n_action_steps,
            state_dim=dimensions.STATE_DIM,
            action_dim=dimensions.ACTION_DIM,
            device=self._engine.device,
            resize_height=360,
            resize_width=640,
        )

    def ResetEpisode(self, request, context):  # noqa: N802
        if not request.client_id or not request.session_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "client_id and session_id are required")
        with self._lock:
            self._engine.reset()
            self._active_client_id = request.client_id
            self._active_session_id = request.session_id
        return pb.ResetEpisodeReply(ok=True, detail="policy queues reset")

    def StreamActions(self, request_iterator, context):  # noqa: N802
        for request in request_iterator:
            try:
                yield self._infer_one(request)
            except ValueError as exc:
                # Invalid observation data is a per-request failure, not a process
                # crash. No action is included in a failed reply.
                yield pb.ActionReply(
                    protocol_version=PROTOCOL_VERSION,
                    session_id=request.session_id,
                    request_id=request.request_id,
                    ok=False,
                    error=str(exc),
                )

    def _infer_one(self, request) -> pb.ActionReply:
        self._validate_request(request)
        state = np.asarray(request.state, dtype=np.float64)

        # One lock owns both the active session and the stateful Diffusion queues.
        # It also prevents a second client from interleaving select_action calls.
        with self._lock:
            if (
                request.client_id != self._active_client_id
                or request.session_id != self._active_session_id
            ):
                raise ValueError("session is not active; ResetEpisode must succeed first")
            action = self._engine.act(state, bytes(request.cam1.data), bytes(request.cam2.data))
            metadata = dict(self._engine.last_act_metadata)

        if action.shape != (dimensions.ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError("policy returned an invalid action")

        return pb.ActionReply(
            protocol_version=PROTOCOL_VERSION,
            session_id=request.session_id,
            request_id=request.request_id,
            action=[float(value) for value in action],
            preprocess_ms=metadata["preprocess_ms"],
            inference_ms=metadata["inference_ms"],
            total_server_ms=metadata["total_server_ms"],
            chunk_refill=metadata["chunk_refill"],
            remaining_chunk_actions=metadata["remaining_chunk_actions"],
            ok=True,
        )

    def _validate_request(self, request) -> None:
        if request.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"protocol_version must be {PROTOCOL_VERSION!r}, got {request.protocol_version!r}"
            )
        if not request.client_id or not request.session_id:
            raise ValueError("client_id and session_id are required")
        if request.request_id == 0:
            raise ValueError("request_id must be non-zero")
        if len(request.state) != dimensions.STATE_DIM:
            raise ValueError(f"state must contain {dimensions.STATE_DIM} floats")
        if not all(math.isfinite(value) for value in request.state):
            raise ValueError("state contains a non-finite value")
        for name, frame in (("cam1", request.cam1), ("cam2", request.cam2)):
            if frame.encoding.lower() not in ("jpeg", "jpg"):
                raise ValueError(f"{name} encoding must be jpeg")
            if not frame.data:
                raise ValueError(f"{name} JPEG is empty")
            if len(frame.data) > self._max_jpeg_bytes:
                raise ValueError(
                    f"{name} JPEG exceeds {self._max_jpeg_bytes} byte limit"
                )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"[remote_diffusion] required environment variable {name} is empty")
    return value


def main() -> int:
    checkpoint = _required_env("CHECKPOINT_PATH")
    bind_address = os.environ.get("GRPC_BIND_ADDRESS", DEFAULT_BIND_ADDRESS)
    device = resolve_device(os.environ.get("DIFFUSION_DEVICE", "cuda"))
    scheduler = os.environ.get("DIFFUSION_SCHEDULER", "DDIM")
    num_inference_steps = int(os.environ.get("DIFFUSION_NUM_INFERENCE_STEPS", "10"))
    n_action_steps = int(os.environ.get("DIFFUSION_N_ACTION_STEPS", "32"))
    max_message_bytes = int(
        os.environ.get("GRPC_MAX_MESSAGE_BYTES", str(DEFAULT_MAX_MESSAGE_BYTES))
    )
    max_jpeg_bytes = int(os.environ.get("MAX_JPEG_BYTES", str(DEFAULT_MAX_JPEG_BYTES)))

    engine = DiffusionInferenceEngine(
        checkpoint,
        device,
        n_action_steps,
        num_inference_steps,
        scheduler,
    )
    service = RemoteDiffusionService(
        engine,
        model_id=os.environ.get("MODEL_ID", "unknown"),
        checkpoint_revision=os.environ.get("CHECKPOINT_REVISION", "unknown"),
        scheduler=str(engine.policy.config.noise_scheduler_type),
        num_inference_steps=engine.policy.config.num_inference_steps,
        n_action_steps=engine.policy.config.n_action_steps,
        max_jpeg_bytes=max_jpeg_bytes,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="grpc"),
        options=(
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", max_message_bytes),
        ),
    )
    pb_grpc.add_RemoteDiffusionServicer_to_server(service, server)
    if server.add_insecure_port(bind_address) == 0:
        raise SystemExit(f"[remote_diffusion] failed to bind {bind_address}")
    server.start()
    print(
        f"[remote_diffusion] ready at {bind_address}; model={service._model_id} "
        f"revision={service._checkpoint_revision} device={device}",
        flush=True,
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2.0).wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
