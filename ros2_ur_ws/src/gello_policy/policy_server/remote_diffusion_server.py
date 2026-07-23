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

from . import remote_diffusion_pb2 as pb
from . import remote_diffusion_pb2_grpc as pb_grpc
from . import zmq_protocol as dimensions


PROTOCOL_VERSION = "1"
DEFAULT_BIND_ADDRESS = "0.0.0.0:50051"
DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_JPEG_BYTES = 4 * 1024 * 1024


class RequestValidationError(ValueError):
    """Recoverable protocol/session error raised before policy state is changed."""


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
        resize_height: int = 360,
        resize_width: int = 640,
    ) -> None:
        self._engine = engine
        self._model_id = model_id
        self._checkpoint_revision = checkpoint_revision
        self._scheduler = scheduler
        self._num_inference_steps = num_inference_steps
        self._n_action_steps = n_action_steps
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._max_jpeg_bytes = max_jpeg_bytes
        # Stateful policy execution/reset is serialized independently from the
        # small status/session fields. Health/GetServerInfo must remain responsive
        # while a slow chunk refill owns the inference lock.
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._active_client_id = ""
        self._active_session_id = ""
        self._last_request_id = 0
        self._stream_active = False
        self._ready = True

    def Health(self, request, context):  # noqa: N802 - generated gRPC API
        with self._state_lock:
            ready = self._ready
        detail = "model warm and ready" if ready else "inference failed; restart required"
        return pb.HealthReply(alive=True, ready=ready, detail=detail)

    def GetServerInfo(self, request, context):  # noqa: N802
        with self._state_lock:
            ready = self._ready
        return pb.ServerInfoReply(
            ready=ready,
            protocol_version=PROTOCOL_VERSION,
            model_id=self._model_id,
            checkpoint_revision=self._checkpoint_revision,
            scheduler=self._scheduler,
            num_inference_steps=self._num_inference_steps,
            n_action_steps=self._n_action_steps,
            state_dim=dimensions.STATE_DIM,
            action_dim=dimensions.ACTION_DIM,
            device=self._engine.device,
            resize_height=self._resize_height,
            resize_width=self._resize_width,
        )

    def ResetEpisode(self, request, context):  # noqa: N802
        if not request.client_id or not request.session_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "client_id and session_id are required")
        with self._inference_lock:
            with self._state_lock:
                if not self._ready:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, "server restart required")
                if self._stream_active:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        "cannot reset while an action stream is active",
                    )
            try:
                self._engine.reset()
            except Exception as exc:
                with self._state_lock:
                    self._ready = False
                    self._active_client_id = ""
                    self._active_session_id = ""
                    self._last_request_id = 0
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"policy reset failed; server restart required: {type(exc).__name__}",
                )
            with self._state_lock:
                self._active_client_id = request.client_id
                self._active_session_id = request.session_id
                self._last_request_id = 0
        return pb.ResetEpisodeReply(ok=True, detail="policy queues reset")

    def StreamActions(self, request_iterator, context):  # noqa: N802
        with self._state_lock:
            if not self._ready:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "server restart required")
            if self._stream_active:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "another action stream is active")
            self._stream_active = True
        try:
            for request in request_iterator:
                try:
                    yield self._infer_one(request)
                except RequestValidationError as exc:
                    # Protocol/session validation finishes before engine.act(),
                    # so these failures cannot consume an action or mutate queues.
                    yield pb.ActionReply(
                        protocol_version=PROTOCOL_VERSION,
                        session_id=request.session_id,
                        request_id=request.request_id,
                        ok=False,
                        error=str(exc),
                    )
                except Exception as exc:
                    # The stateful policy queue may be partially changed after an
                    # unexpected inference failure. Refuse all further work until
                    # the process is restarted and warmed from a clean state.
                    with self._state_lock:
                        self._ready = False
                        self._active_client_id = ""
                        self._active_session_id = ""
                    context.abort(
                        grpc.StatusCode.INTERNAL,
                        f"inference failed; server restart required: {type(exc).__name__}",
                    )
        finally:
            with self._state_lock:
                self._stream_active = False

    def _infer_one(self, request) -> pb.ActionReply:
        self._validate_request(request)
        state = np.asarray(request.state, dtype=np.float64)

        with self._inference_lock:
            with self._state_lock:
                if (
                    request.client_id != self._active_client_id
                    or request.session_id != self._active_session_id
                ):
                    raise RequestValidationError(
                        "session is not active; ResetEpisode must succeed first"
                    )
                expected_request_id = self._last_request_id + 1
                if request.request_id != expected_request_id:
                    raise RequestValidationError(
                        f"request_id must be {expected_request_id}, got {request.request_id}"
                    )
            action = self._engine.act(state, bytes(request.cam1.data), bytes(request.cam2.data))
            metadata = dict(self._engine.last_act_metadata)
            if action.shape != (dimensions.ACTION_DIM,) or not np.all(np.isfinite(action)):
                raise RuntimeError("policy returned an invalid action")
            with self._state_lock:
                self._last_request_id = request.request_id

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
            raise RequestValidationError(
                f"protocol_version must be {PROTOCOL_VERSION!r}, got {request.protocol_version!r}"
            )
        if not request.client_id or not request.session_id:
            raise RequestValidationError("client_id and session_id are required")
        if request.request_id == 0:
            raise RequestValidationError("request_id must be non-zero")
        if len(request.state) != dimensions.STATE_DIM:
            raise RequestValidationError(
                f"state must contain {dimensions.STATE_DIM} floats"
            )
        if not all(math.isfinite(value) for value in request.state):
            raise RequestValidationError("state contains a non-finite value")
        for name, frame in (("cam1", request.cam1), ("cam2", request.cam2)):
            if frame.encoding.lower() not in ("jpeg", "jpg"):
                raise RequestValidationError(f"{name} encoding must be jpeg")
            if not frame.data:
                raise RequestValidationError(f"{name} JPEG is empty")
            if len(frame.data) > self._max_jpeg_bytes:
                raise RequestValidationError(
                    f"{name} JPEG exceeds {self._max_jpeg_bytes} byte limit"
                )
            if frame.width <= 0 or frame.height <= 0:
                raise RequestValidationError(f"{name} width and height must be positive")
        if (
            request.cam1.width != request.cam2.width
            or request.cam1.height != request.cam2.height
        ):
            raise RequestValidationError("cam1/cam2 JPEG metadata dimensions must match")


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"[remote_diffusion] required environment variable {name} is empty")
    return value


def main() -> int:
    from .diffusion_server import DiffusionInferenceEngine, resolve_device

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
