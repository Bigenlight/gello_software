"""Non-blocking, latest-observation-only client for remote Diffusion inference.

This module intentionally has no ROS imports.  ROS callbacks submit immutable
observation snapshots and poll results; a single worker owns all gRPC calls.
"""

from dataclasses import dataclass
import math
import threading
import time
import uuid
from typing import Optional, Tuple

from . import remote_diffusion_pb2 as pb


PROTOCOL_VERSION = "1"
STATE_DIM = 7
ACTION_DIM = 7


@dataclass(frozen=True)
class ImageSnapshot:
    ros_stamp_ns: int
    width: int
    height: int
    jpeg: bytes


@dataclass(frozen=True)
class ObservationSnapshot:
    created_monotonic_ns: int
    state: Tuple[float, ...]
    cam1: ImageSnapshot
    cam2: ImageSnapshot


@dataclass(frozen=True)
class ActionResult:
    request_id: int
    action: Tuple[float, ...]
    received_monotonic_ns: int
    observation_age_s: float
    preprocess_ms: float
    inference_ms: float
    total_server_ms: float
    chunk_refill: bool
    remaining_chunk_actions: int


@dataclass(frozen=True)
class ServerContract:
    model_id: str
    checkpoint_revision: str
    scheduler: str
    num_inference_steps: int
    n_action_steps: int
    resize_height: int
    resize_width: int


class RemoteDiffusionWorker:
    """One worker, at most one RPC in flight, and one replaceable pending input."""

    def __init__(
        self,
        stub,
        *,
        client_id: str,
        rpc_deadline_s: float = 0.6,
        max_response_age_s: float = 0.8,
        max_camera_skew_s: float = 0.1,
        max_jpeg_bytes: int = 4 * 1024 * 1024,
        clock_ns=time.monotonic_ns,
    ) -> None:
        if not client_id:
            raise ValueError("client_id is required")
        if rpc_deadline_s <= 0 or max_response_age_s <= 0:
            raise ValueError("deadlines must be positive")
        self._stub = stub
        self._client_id = client_id
        self._rpc_deadline_s = rpc_deadline_s
        self._max_response_age_s = max_response_age_s
        self._max_camera_skew_ns = int(max_camera_skew_s * 1e9)
        self._max_jpeg_bytes = max_jpeg_bytes
        self._clock_ns = clock_ns
        self._condition = threading.Condition()
        self._pending = None  # type: Optional[ObservationSnapshot]
        self._latest_result = None  # type: Optional[ActionResult]
        self._error = None  # type: Optional[str]
        self._session_id = ""
        self._next_request_id = 1
        self._in_flight = False
        self._stopping = False
        self._thread = None  # type: Optional[threading.Thread]

    def get_server_info(self, expected: Optional[ServerContract] = None):
        info = self._stub.GetServerInfo(
            pb.ServerInfoRequest(), timeout=self._rpc_deadline_s
        )
        if not info.ready:
            raise RuntimeError("remote Diffusion server is not ready")
        if info.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"protocol mismatch: expected {PROTOCOL_VERSION}, got {info.protocol_version}"
            )
        if info.state_dim != STATE_DIM or info.action_dim != ACTION_DIM:
            raise RuntimeError(
                f"dimension mismatch: server state/action={info.state_dim}/{info.action_dim}"
            )
        if info.device != "cuda":
            raise RuntimeError(f"server device must be cuda, got {info.device!r}")
        if expected is not None:
            actual = (
                info.model_id, info.checkpoint_revision, info.scheduler,
                info.num_inference_steps, info.n_action_steps,
                info.resize_height, info.resize_width,
            )
            wanted = (
                expected.model_id, expected.checkpoint_revision, expected.scheduler,
                expected.num_inference_steps, expected.n_action_steps,
                expected.resize_height, expected.resize_width,
            )
            if actual != wanted:
                raise RuntimeError(f"server contract mismatch: expected {wanted}, got {actual}")
        return info

    def reset_episode(self, session_id: Optional[str] = None) -> str:
        new_session_id = session_id or uuid.uuid4().hex
        with self._condition:
            if self._in_flight or self._pending is not None:
                raise RuntimeError("cannot reset while inference is pending or in flight")
            reply = self._stub.ResetEpisode(
                pb.ResetEpisodeRequest(
                    client_id=self._client_id, session_id=new_session_id
                ),
                timeout=self._rpc_deadline_s,
            )
            if not reply.ok:
                raise RuntimeError(f"episode reset failed: {reply.detail}")
            self._session_id = new_session_id
            self._pending = None
            self._latest_result = None
            self._error = None
            self._next_request_id = 1
        return new_session_id

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run, name="remote-diffusion", daemon=True
            )
            self._thread.start()

    def close(self, timeout_s: float = 2.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        channel = getattr(self, "_channel", None)
        if channel is not None:
            channel.close()

    def submit(self, observation: ObservationSnapshot) -> None:
        self._validate_observation(observation)
        with self._condition:
            if not self._session_id:
                raise RuntimeError("ResetEpisode must succeed before submit")
            if self._stopping:
                raise RuntimeError("worker is stopping")
            self._pending = observation
            self._condition.notify()

    def has_result(self) -> bool:
        with self._condition:
            return self._latest_result is not None

    def take_result(self, max_age_s: float) -> Optional[ActionResult]:
        """Consume one result, rejecting an action that became stale while polling."""
        with self._condition:
            result = self._latest_result
            self._latest_result = None
        if result is None:
            return None
        age_s = (self._clock_ns() - result.received_monotonic_ns) / 1e9
        if age_s < 0 or age_s > max_age_s:
            raise RuntimeError(f"polled action is stale by {age_s:.3f}s")
        return result

    def error(self) -> Optional[str]:
        with self._condition:
            return self._error

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                observation = self._pending
                self._pending = None
                session_id = self._session_id
                request_id = self._next_request_id
                self._next_request_id += 1
                self._in_flight = True

            try:
                request = self._build_request(observation, session_id, request_id)
                replies = self._stub.StreamActions(
                    iter((request,)), timeout=self._rpc_deadline_s
                )
                reply = next(iter(replies))
                received_ns = self._clock_ns()
                result = self._parse_reply(
                    reply, session_id, request_id, observation, received_ns
                )
            except Exception as exc:  # gRPC errors are transport-specific subclasses.
                with self._condition:
                    self._in_flight = False
                    # A late failure from an old session must not disarm a newer
                    # operator-authorized reset session.
                    if self._session_id == session_id:
                        self._error = f"{type(exc).__name__}: {exc}"
                        self._latest_result = None
                        self._session_id = ""
                        self._pending = None
                continue

            with self._condition:
                self._in_flight = False
                # ResetEpisode invalidates every request from the previous
                # session, including one that was already in flight.
                if self._session_id == session_id:
                    self._latest_result = result
                    self._error = None

    def _build_request(self, observation, session_id, request_id):
        def frame(image):
            return pb.ImageFrame(
                ros_stamp_ns=image.ros_stamp_ns,
                encoding="jpeg",
                width=image.width,
                height=image.height,
                data=image.jpeg,
            )

        return pb.ObservationRequest(
            protocol_version=PROTOCOL_VERSION,
            client_id=self._client_id,
            session_id=session_id,
            request_id=request_id,
            created_monotonic_ns=observation.created_monotonic_ns,
            state=observation.state,
            cam1=frame(observation.cam1),
            cam2=frame(observation.cam2),
        )

    def _parse_reply(self, reply, session_id, request_id, observation, received_ns):
        if not reply.ok:
            raise RuntimeError(f"server rejected observation: {reply.error}")
        if reply.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError("response protocol version mismatch")
        if reply.session_id != session_id or reply.request_id != request_id:
            raise RuntimeError("response session/request ID mismatch")
        action = tuple(float(value) for value in reply.action)
        if len(action) != ACTION_DIM or not all(math.isfinite(value) for value in action):
            raise RuntimeError("response action is invalid")
        age_s = (received_ns - observation.created_monotonic_ns) / 1e9
        if age_s < 0 or age_s > self._max_response_age_s:
            raise RuntimeError(f"stale response age {age_s:.3f}s")
        return ActionResult(
            request_id=request_id,
            action=action,
            received_monotonic_ns=received_ns,
            observation_age_s=age_s,
            preprocess_ms=reply.preprocess_ms,
            inference_ms=reply.inference_ms,
            total_server_ms=reply.total_server_ms,
            chunk_refill=reply.chunk_refill,
            remaining_chunk_actions=reply.remaining_chunk_actions,
        )

    def _validate_observation(self, observation: ObservationSnapshot) -> None:
        if len(observation.state) != STATE_DIM:
            raise ValueError(f"state must contain {STATE_DIM} values")
        if not all(math.isfinite(value) for value in observation.state):
            raise ValueError("state contains a non-finite value")
        if observation.created_monotonic_ns <= 0:
            raise ValueError("created_monotonic_ns must be positive")
        for name, image in (("cam1", observation.cam1), ("cam2", observation.cam2)):
            if image.ros_stamp_ns <= 0 or image.width <= 0 or image.height <= 0:
                raise ValueError(f"{name} timestamp and dimensions must be positive")
            if not image.jpeg:
                raise ValueError(f"{name} JPEG is empty")
            if len(image.jpeg) > self._max_jpeg_bytes:
                raise ValueError(f"{name} JPEG exceeds size limit")
        skew_ns = abs(observation.cam1.ros_stamp_ns - observation.cam2.ros_stamp_ns)
        if skew_ns > self._max_camera_skew_ns:
            raise ValueError("camera timestamp skew exceeds limit")


def create_worker(target: str, **kwargs) -> RemoteDiffusionWorker:
    """Create the production worker while keeping grpc optional for pure tests."""
    import grpc

    from .remote_diffusion_pb2_grpc import RemoteDiffusionStub

    channel = grpc.insecure_channel(target)
    worker = RemoteDiffusionWorker(RemoteDiffusionStub(channel), **kwargs)
    worker._channel = channel  # Keep the channel alive with its worker.
    return worker
