"""gRPC adapter for the transport-neutral HIL-SERL actor contract."""

from __future__ import annotations

from concurrent import futures
import math
import time
from typing import Any, Callable, Mapping, Optional, Tuple

import grpc
import numpy as np

from ur_env.actor_network import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ActionResult,
    ActorNetworkError,
    ActorProtocolError,
    ActorSessionService,
    ActorTransportError,
    BeginEpisodeCommand,
    FailedPreconditionError,
    ObservationPacket,
    PolicyInferenceError,
    ServerInfo,
    StepCommand,
    StepResult,
    TransitionAck,
    copy_observation,
    validate_action,
    validate_counter,
    validate_timestamp_ns,
)
from ur_env.proto import actor_transport_pb2 as pb
from ur_env.proto import actor_transport_pb2_grpc as pb_grpc


DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_TRANSIENT_CODES = (grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.UNAVAILABLE)


def observation_to_proto(packet: ObservationPacket) -> pb.Observation:
    observation = copy_observation(packet.observation)
    message = pb.Observation(
        observation_id=packet.observation_id,
        timestamp_ns=validate_timestamp_ns(packet.timestamp_ns),
    )

    def append(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            for key in sorted(node):
                append(node[key], path + (key,))
            return
        array = np.ascontiguousarray(np.asarray(node))
        tensor = message.tensors.add()
        tensor.path.extend(path)
        tensor.dtype = array.dtype.str
        tensor.shape.extend(int(dim) for dim in array.shape)
        tensor.data = array.tobytes(order="C")

    append(observation, ())
    return message


def observation_from_proto(message: pb.Observation) -> ObservationPacket:
    if not message.observation_id:
        raise ActorProtocolError("observation_id is required")
    timestamp_ns = validate_timestamp_ns(message.timestamp_ns)
    if not message.tensors:
        raise ActorProtocolError("observation has no tensors")
    root: dict[str, Any] = {}
    seen: set[tuple[str, ...]] = set()
    for tensor in message.tensors:
        path = tuple(tensor.path)
        if not path or any(not component for component in path):
            raise ActorProtocolError("tensor path must contain non-empty keys")
        if path in seen:
            raise ActorProtocolError(f"duplicate tensor path {'/'.join(path)!r}")
        seen.add(path)
        try:
            dtype = np.dtype(tensor.dtype)
        except TypeError as exc:
            raise ActorProtocolError(f"invalid tensor dtype {tensor.dtype!r}") from exc
        if dtype.kind not in "biuf":
            raise ActorProtocolError(f"unsupported tensor dtype {dtype}")
        shape = tuple(int(dim) for dim in tensor.shape)
        expected_bytes = math.prod(shape) * dtype.itemsize
        if not shape:
            expected_bytes = dtype.itemsize
        if expected_bytes != len(tensor.data):
            raise ActorProtocolError(
                f"tensor {'/'.join(path)!r} byte length mismatch: "
                f"expected {expected_bytes}, got {len(tensor.data)}"
            )
        array = np.frombuffer(bytes(tensor.data), dtype=dtype).reshape(shape).copy()
        if dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ActorProtocolError(
                f"tensor {'/'.join(path)!r} contains a non-finite value"
            )
        cursor = root
        for component in path[:-1]:
            existing = cursor.get(component)
            if existing is None:
                existing = {}
                cursor[component] = existing
            if not isinstance(existing, dict):
                raise ActorProtocolError("tensor paths collide")
            cursor = existing
        if path[-1] in cursor:
            raise ActorProtocolError("tensor paths collide")
        cursor[path[-1]] = array
    return ObservationPacket(message.observation_id, timestamp_ns, root)


def data_to_proto(data: Mapping[str, Any]) -> pb.Data:
    if not isinstance(data, Mapping) or set(data) != {"meta", "transition"}:
        raise ActorProtocolError("data must contain exactly meta and transition")
    meta = data["meta"]
    transition = data["transition"]
    if not isinstance(meta, Mapping) or not isinstance(transition, Mapping):
        raise ActorProtocolError("data.meta and data.transition must be mappings")
    return pb.Data(
        meta=pb.Meta(
            schema_version=validate_counter(
                meta.get("schema_version"), name="schema_version"
            ),
            run_id=str(meta.get("run_id", "")),
            actor_id=str(meta.get("actor_id", "")),
            session_id=str(meta.get("session_id", "")),
            transition_id=str(meta.get("transition_id", "")),
            env_step=validate_counter(meta.get("env_step"), name="env_step"),
            timestamp_ns=validate_timestamp_ns(meta.get("timestamp_ns")),
            policy_version=validate_counter(
                meta.get("policy_version"), name="policy_version"
            ),
            policy_action=[float(value) for value in np.asarray(meta.get("policy_action")).flat],
            intervened=bool(meta.get("intervened")),
        ),
        transition=pb.Transition(
            episode_id=validate_counter(
                transition.get("episode_id"), name="episode_id"
            ),
            step_id=validate_counter(transition.get("step_id"), name="step_id"),
            observation_id=str(transition.get("observation_id", "")),
            actions=[float(value) for value in np.asarray(transition.get("actions")).flat],
            next_observation_id=str(transition.get("next_observation_id", "")),
            rewards=float(transition.get("rewards")),
            masks=float(transition.get("masks")),
            dones=bool(transition.get("dones")),
            truncated=bool(transition.get("truncated")),
            has_grasp_penalty="grasp_penalty" in transition,
            grasp_penalty=float(transition.get("grasp_penalty", 0.0)),
        ),
    )


def data_from_proto(message: pb.Data) -> dict[str, Any]:
    data = {
        "meta": {
            "schema_version": int(message.meta.schema_version),
            "run_id": message.meta.run_id,
            "actor_id": message.meta.actor_id,
            "session_id": message.meta.session_id,
            "transition_id": message.meta.transition_id,
            "env_step": int(message.meta.env_step),
            "timestamp_ns": int(message.meta.timestamp_ns),
            "policy_version": int(message.meta.policy_version),
            "policy_action": np.asarray(message.meta.policy_action, dtype=np.float32),
            "intervened": int(message.meta.intervened),
        },
        "transition": {
            "episode_id": int(message.transition.episode_id),
            "step_id": int(message.transition.step_id),
            "observation_id": message.transition.observation_id,
            "actions": np.asarray(message.transition.actions, dtype=np.float32),
            "next_observation_id": message.transition.next_observation_id,
            "rewards": float(message.transition.rewards),
            "masks": float(message.transition.masks),
            "dones": bool(message.transition.dones),
            "truncated": bool(message.transition.truncated),
        },
    }
    if message.transition.has_grasp_penalty:
        data["transition"]["grasp_penalty"] = float(
            message.transition.grasp_penalty
        )
    return data


def _action_to_proto(result: ActionResult) -> pb.ActionReply:
    return pb.ActionReply(
        ok=True,
        protocol_version=PROTOCOL_VERSION,
        session_id=result.session_id,
        request_id=result.request_id,
        request_created_monotonic_ns=result.request_created_monotonic_ns,
        observation_id=result.observation_id,
        action=[float(value) for value in result.action.flat],
        policy_version=result.policy_version,
        server_inference_ms=result.server_inference_ms,
    )


class GrpcActorNetwork:
    """Strict synchronous laptop client with one same-ID transient retry."""

    def __init__(
        self,
        target: str,
        *,
        actor_id: str,
        action_shape: Tuple[int, ...] = (7,),
        timeout_s: float = 0.6,
        max_response_age_s: float = 0.8,
        retry_count: int = 1,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        channel: Any = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not target:
            raise ValueError("gRPC target is required")
        if not actor_id:
            raise ValueError("actor_id is required")
        if timeout_s <= 0 or max_response_age_s <= 0:
            raise ValueError("timeouts must be positive")
        if retry_count != 1:
            raise ValueError("protocol v1 requires exactly one transient retry")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        self._target = target
        self._actor_id = actor_id
        self._action_shape = tuple(int(dim) for dim in action_shape)
        self._timeout_s = float(timeout_s)
        self._max_response_age_s = float(max_response_age_s)
        self._retry_count = retry_count
        self._monotonic_ns = monotonic_ns
        self._channel = channel or grpc.insecure_channel(
            target,
            options=(
                ("grpc.max_receive_message_length", int(max_message_bytes)),
                ("grpc.max_send_message_length", int(max_message_bytes)),
            ),
        )
        self._owns_channel = channel is None
        self._stub = pb_grpc.ActorTransportStub(self._channel)
        self._run_id = ""
        self._session_id = ""
        self._next_request_id = 1
        self._last_policy_version = -1
        self._active = False
        self._pending_step: Optional[pb.StepRequest] = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        actor_id: str,
        action_shape: Tuple[int, ...],
    ) -> "GrpcActorNetwork":
        host = str(config.get("host", "127.0.0.1"))
        port = int(config.get("port", 50052))
        if not host or not 0 < port < 65536:
            raise ValueError("NETWORK host/port is invalid")
        return cls(
            f"{host}:{port}",
            actor_id=actor_id,
            action_shape=action_shape,
            timeout_s=float(config.get("timeout_s", 0.6)),
            max_response_age_s=float(config.get("max_response_age_s", 0.8)),
            retry_count=int(config.get("retry_count", 1)),
            max_message_bytes=int(
                config.get("max_message_bytes", DEFAULT_MAX_MESSAGE_BYTES)
            ),
        )

    @property
    def pending_transition_id(self) -> Optional[str]:
        if self._pending_step is None:
            return None
        return self._pending_step.data.meta.transition_id

    def health(self) -> tuple[bool, bool, str]:
        reply, _ = self._call(self._stub.Health, pb.HealthRequest(), "Health")
        return bool(reply.alive), bool(reply.ready), reply.detail

    def get_server_info(self) -> ServerInfo:
        reply, _ = self._call(
            self._stub.GetServerInfo, pb.ServerInfoRequest(), "GetServerInfo"
        )
        info = ServerInfo(
            ready=bool(reply.ready),
            protocol_version=reply.protocol_version,
            schema_version=int(reply.schema_version),
            action_dim=int(reply.action_dim),
            model_id=reply.model_id,
        )
        if not info.ready:
            raise FailedPreconditionError("remote policy service is not ready")
        if info.protocol_version != PROTOCOL_VERSION:
            raise ActorProtocolError(
                f"server protocol_version is {info.protocol_version!r}, "
                f"expected {PROTOCOL_VERSION!r}"
            )
        if info.schema_version != SCHEMA_VERSION:
            raise ActorProtocolError(
                f"server schema_version is {info.schema_version}, "
                f"expected {SCHEMA_VERSION}"
            )
        if info.action_dim != int(np.prod(self._action_shape)):
            raise ActorProtocolError(
                f"server action_dim is {info.action_dim}, "
                f"expected {int(np.prod(self._action_shape))}"
            )
        return info

    def begin_episode(
        self,
        observation: Mapping[str, Any],
        *,
        run_id: str,
        session_id: str,
        episode_id: int,
        observation_id: str,
        timestamp_ns: int,
        deterministic: bool = False,
    ) -> ActionResult:
        if self._pending_step is not None:
            raise FailedPreconditionError(
                "cannot begin an episode while a transition ACK is pending"
            )
        created_ns = validate_timestamp_ns(
            self._monotonic_ns(), name="created_monotonic_ns"
        )
        packet = ObservationPacket(observation_id, timestamp_ns, observation)
        request = pb.BeginEpisodeRequest(
            protocol_version=PROTOCOL_VERSION,
            actor_id=self._actor_id,
            run_id=run_id,
            session_id=session_id,
            episode_id=validate_counter(episode_id, name="episode_id"),
            request_id=1,
            created_monotonic_ns=created_ns,
            observation=observation_to_proto(packet),
            deterministic=bool(deterministic),
        )
        self._active = False
        reply, round_trip_ms = self._call(
            self._stub.BeginEpisode, request, "BeginEpisode", created_ns=created_ns
        )
        result = self._parse_action(
            reply,
            expected_session_id=session_id,
            expected_request_id=1,
            expected_created_ns=created_ns,
            expected_observation_id=observation_id,
            minimum_policy_version=0,
            round_trip_ms=round_trip_ms,
        )
        self._run_id = run_id
        self._session_id = session_id
        self._next_request_id = 2
        self._last_policy_version = result.policy_version
        self._active = True
        return result

    def step(
        self,
        next_observation: Mapping[str, Any],
        *,
        next_observation_id: str,
        next_timestamp_ns: int,
        data: Mapping[str, Any],
        request_action: bool,
        deterministic: bool = False,
    ) -> StepResult:
        if not self._active:
            raise FailedPreconditionError("BeginEpisode must succeed before Step")
        if self._pending_step is not None:
            raise FailedPreconditionError(
                f"transition {self.pending_transition_id!r} is still pending"
            )
        created_ns = validate_timestamp_ns(
            self._monotonic_ns(), name="created_monotonic_ns"
        )
        request = pb.StepRequest(
            protocol_version=PROTOCOL_VERSION,
            actor_id=self._actor_id,
            run_id=self._run_id,
            session_id=self._session_id,
            request_id=self._next_request_id,
            created_monotonic_ns=created_ns,
            data=data_to_proto(data),
            next_observation=observation_to_proto(
                ObservationPacket(
                    next_observation_id, next_timestamp_ns, next_observation
                )
            ),
            request_action=bool(request_action),
            deterministic=bool(deterministic),
        )
        self._pending_step = request
        try:
            reply, round_trip_ms = self._call(
                self._stub.Step, request, "Step", created_ns=created_ns
            )
        except Exception:
            self._active = False
            raise

        ack = TransitionAck(
            accepted=bool(reply.ack.accepted),
            transition_id=reply.ack.transition_id,
            session_id=reply.ack.session_id,
            request_id=int(reply.ack.request_id),
            deduplicated=bool(reply.ack.deduplicated),
            error=reply.ack.error,
        )
        expected_ack = (
            str(request.data.meta.transition_id),
            self._session_id,
            self._next_request_id,
        )
        actual_ack = (ack.transition_id, ack.session_id, ack.request_id)
        if not ack.accepted or actual_ack != expected_ack:
            self._active = False
            raise ActorProtocolError(
                f"transition ACK mismatch/rejection: expected {expected_ack}, "
                f"got {actual_ack}; {ack.error}"
            )

        # The transition is now accepted even if inference for the next action
        # failed. Clear pending before reporting any action-side error.
        self._pending_step = None
        self._next_request_id += 1
        if not request_action:
            self._active = False
            if reply.has_action:
                raise ActorProtocolError("terminal Step unexpectedly returned an action")
            return StepResult(ack=ack, action=None)

        if not reply.has_action:
            self._active = False
            error = reply.action.error or "server accepted data but returned no action"
            raise PolicyInferenceError(error)
        try:
            result = self._parse_action(
                reply.action,
                expected_session_id=self._session_id,
                expected_request_id=request.request_id,
                expected_created_ns=created_ns,
                expected_observation_id=next_observation_id,
                minimum_policy_version=self._last_policy_version,
                round_trip_ms=round_trip_ms,
            )
        except Exception:
            self._active = False
            raise
        self._last_policy_version = result.policy_version
        return StepResult(ack=ack, action=result)

    def close(self) -> None:
        self._active = False
        if self._owns_channel:
            self._channel.close()

    def _call(
        self,
        rpc: Callable[..., Any],
        request: Any,
        name: str,
        *,
        created_ns: Optional[int] = None,
    ) -> tuple[Any, float]:
        started_ns = created_ns or self._monotonic_ns()
        last_error: Optional[grpc.RpcError] = None
        for attempt in range(self._retry_count + 1):
            try:
                reply = rpc(request, timeout=self._timeout_s)
                elapsed_ms = (self._monotonic_ns() - started_ns) / 1e6
                if created_ns is not None and elapsed_ms > self._max_response_age_s * 1000:
                    raise ActorProtocolError(
                        f"{name} reply is stale ({elapsed_ms:.1f} ms)"
                    )
                return reply, elapsed_ms
            except grpc.RpcError as exc:
                last_error = exc
                code = exc.code()
                if code not in _TRANSIENT_CODES or attempt >= self._retry_count:
                    detail = exc.details() if hasattr(exc, "details") else str(exc)
                    raise ActorTransportError(
                        f"{name} RPC failed ({code}): {detail}"
                    ) from exc
        raise ActorTransportError(f"{name} RPC failed: {last_error}")

    def _parse_action(
        self,
        reply: pb.ActionReply,
        *,
        expected_session_id: str,
        expected_request_id: int,
        expected_created_ns: int,
        expected_observation_id: str,
        minimum_policy_version: int,
        round_trip_ms: float,
    ) -> ActionResult:
        if not reply.ok:
            raise PolicyInferenceError(reply.error or "remote policy rejected inference")
        expected = (
            PROTOCOL_VERSION,
            expected_session_id,
            expected_request_id,
            expected_created_ns,
            expected_observation_id,
        )
        actual = (
            reply.protocol_version,
            reply.session_id,
            int(reply.request_id),
            int(reply.request_created_monotonic_ns),
            reply.observation_id,
        )
        if actual != expected:
            raise ActorProtocolError(
                f"action reply identity mismatch: expected {expected}, got {actual}"
            )
        action = validate_action(
            reply.action, action_shape=self._action_shape, name="remote action"
        )
        policy_version = int(reply.policy_version)
        if policy_version < minimum_policy_version:
            raise ActorProtocolError(
                f"policy_version decreased below {minimum_policy_version}"
            )
        inference_ms = float(reply.server_inference_ms)
        if not math.isfinite(inference_ms) or inference_ms < 0:
            raise ActorProtocolError("server_inference_ms is invalid")
        return ActionResult(
            action=action,
            policy_version=policy_version,
            session_id=reply.session_id,
            request_id=int(reply.request_id),
            request_created_monotonic_ns=int(reply.request_created_monotonic_ns),
            observation_id=reply.observation_id,
            server_inference_ms=inference_ms,
            round_trip_ms=round_trip_ms,
        )


class GrpcActorServicer(pb_grpc.ActorTransportServicer):
    def __init__(self, service: ActorSessionService) -> None:
        self._service = service

    def Health(self, request, context):
        alive, ready, detail = self._service.health()
        return pb.HealthReply(alive=alive, ready=ready, detail=detail)

    def GetServerInfo(self, request, context):
        info = self._service.get_server_info()
        return pb.ServerInfoReply(
            ready=info.ready,
            protocol_version=info.protocol_version,
            schema_version=info.schema_version,
            action_dim=info.action_dim,
            model_id=info.model_id,
        )

    def BeginEpisode(self, request, context):
        try:
            command = BeginEpisodeCommand(
                protocol_version=request.protocol_version,
                actor_id=request.actor_id,
                run_id=request.run_id,
                session_id=request.session_id,
                episode_id=int(request.episode_id),
                request_id=int(request.request_id),
                created_monotonic_ns=int(request.created_monotonic_ns),
                observation=observation_from_proto(request.observation),
                deterministic=bool(request.deterministic),
                fingerprint=request.SerializeToString(deterministic=True),
            )
            return _action_to_proto(self._service.begin_episode(command))
        except PolicyInferenceError as exc:
            return pb.ActionReply(ok=False, error=str(exc))
        except Exception as exc:
            _abort(context, exc)

    def Step(self, request, context):
        try:
            command = StepCommand(
                protocol_version=request.protocol_version,
                actor_id=request.actor_id,
                run_id=request.run_id,
                session_id=request.session_id,
                request_id=int(request.request_id),
                created_monotonic_ns=int(request.created_monotonic_ns),
                data=data_from_proto(request.data),
                next_observation=observation_from_proto(request.next_observation),
                request_action=bool(request.request_action),
                deterministic=bool(request.deterministic),
                fingerprint=request.SerializeToString(deterministic=True),
            )
            result = self._service.step(command)
            reply = pb.StepReply(
                ack=pb.Ack(
                    accepted=result.ack.accepted,
                    transition_id=result.ack.transition_id,
                    session_id=result.ack.session_id,
                    request_id=result.ack.request_id,
                    deduplicated=result.ack.deduplicated,
                    error=result.ack.error,
                ),
                has_action=result.action is not None,
            )
            if result.action is not None:
                reply.action.CopyFrom(_action_to_proto(result.action))
            elif result.action_error:
                reply.action.CopyFrom(
                    pb.ActionReply(ok=False, error=result.action_error)
                )
            return reply
        except Exception as exc:
            _abort(context, exc)


def _abort(context: Any, exc: Exception) -> None:
    if isinstance(exc, FailedPreconditionError):
        code = grpc.StatusCode.FAILED_PRECONDITION
    elif isinstance(exc, ActorProtocolError):
        code = grpc.StatusCode.INVALID_ARGUMENT
    elif isinstance(exc, ActorNetworkError):
        code = grpc.StatusCode.INTERNAL
    else:
        code = grpc.StatusCode.INTERNAL
    context.abort(code, str(exc))


def create_grpc_server(
    service: ActorSessionService,
    *,
    bind_address: str = "127.0.0.1:0",
    max_workers: int = 4,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> tuple[Any, int]:
    """Create (but do not start) a gRPC server and return its bound port."""
    if max_workers <= 0 or max_message_bytes <= 0:
        raise ValueError("server limits must be positive")
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="hil-serl-grpc"
        ),
        options=(
            ("grpc.max_receive_message_length", int(max_message_bytes)),
            ("grpc.max_send_message_length", int(max_message_bytes)),
        ),
    )
    pb_grpc.add_ActorTransportServicer_to_server(GrpcActorServicer(service), server)
    port = server.add_insecure_port(bind_address)
    if port == 0:
        raise ActorTransportError(f"failed to bind gRPC server at {bind_address}")
    return server, port
