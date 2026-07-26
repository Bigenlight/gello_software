"""Transport-neutral contract for a remote HIL-SERL actor.

The laptop sends an initial observation with :meth:`begin_episode`.  Every
later RPC combines the transition just executed with its next observation, so
large image tensors cross the network once rather than once for inference and
again for replay insertion.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, replace
import hashlib
import math
import threading
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple

import numpy as np


PROTOCOL_VERSION = "1"
SCHEMA_VERSION = 1


class ActorNetworkError(RuntimeError):
    """Base class for failures after which no action may be executed."""


class ActorTransportError(ActorNetworkError):
    """The remote endpoint did not complete an RPC."""


class ActorProtocolError(ActorNetworkError):
    """A request or reply violated the actor wire contract."""


class FailedPreconditionError(ActorProtocolError):
    """The request is well formed but invalid for the current session."""


class PolicyInferenceError(ActorNetworkError):
    """The server could not produce a safe action."""


@dataclass(frozen=True)
class ObservationPacket:
    observation_id: str
    timestamp_ns: int
    observation: Mapping[str, Any]


@dataclass(frozen=True)
class BeginEpisodeCommand:
    protocol_version: str
    actor_id: str
    run_id: str
    session_id: str
    episode_id: int
    request_id: int
    created_monotonic_ns: int
    observation: ObservationPacket
    deterministic: bool = False
    fingerprint: bytes = b""


@dataclass(frozen=True)
class StepCommand:
    protocol_version: str
    actor_id: str
    run_id: str
    session_id: str
    request_id: int
    created_monotonic_ns: int
    data: Mapping[str, Any]
    next_observation: ObservationPacket
    request_action: bool
    deterministic: bool = False
    fingerprint: bytes = b""


@dataclass(frozen=True)
class ActionResult:
    action: np.ndarray
    policy_version: int
    session_id: str
    request_id: int
    request_created_monotonic_ns: int
    observation_id: str
    server_inference_ms: float
    round_trip_ms: float = 0.0


@dataclass(frozen=True)
class TransitionAck:
    accepted: bool
    transition_id: str
    session_id: str
    request_id: int
    deduplicated: bool = False
    error: str = ""


@dataclass(frozen=True)
class StepResult:
    ack: TransitionAck
    action: Optional[ActionResult]
    action_error: str = ""


@dataclass(frozen=True)
class ServerInfo:
    ready: bool
    protocol_version: str
    schema_version: int
    action_dim: int
    model_id: str


class ActorNetwork(Protocol):
    """Common laptop API implemented by each network transport."""

    def health(self) -> tuple[bool, bool, str]:
        ...

    def get_server_info(self) -> ServerInfo:
        ...

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
        ...

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
        ...

    def close(self) -> None:
        ...


def validate_timestamp_ns(value: Any, *, name: str = "timestamp_ns") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ActorProtocolError(f"{name} must be an integer")
    result = int(value)
    if result <= 0 or result > np.iinfo(np.int64).max:
        raise ActorProtocolError(f"{name} must be a positive signed int64")
    return result


def validate_counter(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ActorProtocolError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result > np.iinfo(np.int64).max:
        raise ActorProtocolError(f"{name} must be a non-negative signed int64")
    return result


def validate_action(
    value: Any,
    *,
    action_shape: Tuple[int, ...],
    name: str = "action",
) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != action_shape:
        raise ActorProtocolError(
            f"{name} must have shape {action_shape}, got {action.shape}"
        )
    if not np.all(np.isfinite(action)):
        raise ActorProtocolError(f"{name} contains a non-finite value")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ActorProtocolError(f"{name} must be within [-1, 1]")
    return action.copy()


def copy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a nested numeric numpy observation."""
    if not isinstance(observation, Mapping) or not observation:
        raise ActorProtocolError("observation must be a non-empty mapping")

    def copy_node(node: Any, path: tuple[str, ...]) -> Any:
        if isinstance(node, Mapping):
            if not node:
                raise ActorProtocolError(
                    f"observation mapping {'/'.join(path) or '<root>'} is empty"
                )
            result: dict[str, Any] = {}
            for key, value in node.items():
                if not isinstance(key, str) or not key:
                    raise ActorProtocolError("observation keys must be non-empty strings")
                result[key] = copy_node(value, path + (key,))
            return result

        array = np.asarray(node)
        if array.dtype.kind not in "biuf":
            raise ActorProtocolError(
                f"observation {'/'.join(path)} has unsupported dtype {array.dtype}"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ActorProtocolError(
                f"observation {'/'.join(path)} contains a non-finite value"
            )
        return np.ascontiguousarray(array).copy()

    return copy_node(observation, ())


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"{")
        for key in sorted(value):
            if not isinstance(key, str):
                raise ActorProtocolError("fingerprinted mapping keys must be strings")
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
            _hash_value(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"a")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _hash_value(digest, item)
        digest.update(b"]")
        return
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
        return
    raise ActorProtocolError(f"cannot fingerprint value of type {type(value).__name__}")


def command_fingerprint(command: Any) -> bytes:
    digest = hashlib.sha256()
    if isinstance(command, BeginEpisodeCommand):
        value = {
            "protocol_version": command.protocol_version,
            "actor_id": command.actor_id,
            "run_id": command.run_id,
            "session_id": command.session_id,
            "episode_id": command.episode_id,
            "request_id": command.request_id,
            "created_monotonic_ns": command.created_monotonic_ns,
            "observation_id": command.observation.observation_id,
            "timestamp_ns": command.observation.timestamp_ns,
            "observation": command.observation.observation,
            "deterministic": command.deterministic,
        }
    elif isinstance(command, StepCommand):
        value = {
            "protocol_version": command.protocol_version,
            "actor_id": command.actor_id,
            "run_id": command.run_id,
            "session_id": command.session_id,
            "request_id": command.request_id,
            "created_monotonic_ns": command.created_monotonic_ns,
            "data": command.data,
            "next_observation_id": command.next_observation.observation_id,
            "next_timestamp_ns": command.next_observation.timestamp_ns,
            "next_observation": command.next_observation.observation,
            "request_action": command.request_action,
            "deterministic": command.deterministic,
        }
    else:
        raise TypeError(f"unsupported command type {type(command).__name__}")
    _hash_value(digest, value)
    return digest.digest()


@dataclass
class _RunState:
    next_env_step: int = 0
    next_episode_id: int = 0
    active_session_id: str = ""


@dataclass
class _SessionState:
    actor_id: str
    run_id: str
    session_id: str
    episode_id: int
    current_observation: ObservationPacket
    expected_request_id: int
    expected_step_id: int
    last_action: np.ndarray
    last_policy_version: int
    active: bool = True


class ActorSessionService:
    """Transport-independent server state, validation, routing, and dedupe."""

    def __init__(
        self,
        sample_action: Callable[[Mapping[str, Any], bool], tuple[Any, int]],
        *,
        action_shape: Tuple[int, ...] = (7,),
        model_id: str = "mock-zero-policy",
        cache_size: int = 2048,
        clock: Callable[[], float] = time.perf_counter,
        accept_data: Optional[Callable[[dict[str, Any], bool], None]] = None,
        in_memory_capacity: int = 256,
    ) -> None:
        if not action_shape or any(int(dim) <= 0 for dim in action_shape):
            raise ValueError("action_shape must contain positive dimensions")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if in_memory_capacity <= 0:
            raise ValueError("in_memory_capacity must be positive")
        self._sample_action = sample_action
        self._action_shape = tuple(int(dim) for dim in action_shape)
        self._model_id = str(model_id)
        self._cache_size = int(cache_size)
        self._clock = clock
        self._in_memory_capacity = int(in_memory_capacity)
        self._accept_data = accept_data or self._accept_data_in_memory
        self._lock = threading.RLock()
        self._ready = True
        self._runs: dict[tuple[str, str], _RunState] = {}
        self._sessions: dict[tuple[str, str], _SessionState] = {}
        self._observations: dict[tuple[str, str, str], ObservationPacket] = {}
        self._reply_cache: OrderedDict[
            tuple[str, str, str, int], tuple[bytes, Any]
        ] = OrderedDict()
        self.replay_items: list[dict[str, Any]] = []
        self.intervention_items: list[dict[str, Any]] = []
        self.observation_accept_count = 0
        self.inference_count = 0

    def health(self) -> tuple[bool, bool, str]:
        with self._lock:
            return True, self._ready, "ready" if self._ready else "policy fault"

    def get_server_info(self) -> ServerInfo:
        with self._lock:
            return ServerInfo(
                ready=self._ready,
                protocol_version=PROTOCOL_VERSION,
                schema_version=SCHEMA_VERSION,
                action_dim=int(np.prod(self._action_shape)),
                model_id=self._model_id,
            )

    def begin_episode(self, command: BeginEpisodeCommand) -> ActionResult:
        fingerprint = command.fingerprint or command_fingerprint(command)
        cache_key = (
            "begin",
            command.actor_id,
            command.session_id,
            command.request_id,
        )
        with self._lock:
            cached = self._cached(cache_key, fingerprint)
            if cached is not None:
                return _copy_action_result(cached)
            self._require_ready()
            self._validate_begin(command)

            run_key = (command.actor_id, command.run_id)
            run = self._runs.setdefault(run_key, _RunState())
            if run.active_session_id:
                raise FailedPreconditionError(
                    f"session {run.active_session_id!r} is still active"
                )
            if command.episode_id != run.next_episode_id:
                raise FailedPreconditionError(
                    f"episode_id must be {run.next_episode_id}, got {command.episode_id}"
                )

            observation = self._validated_observation(command.observation)
            try:
                action, version, inference_ms = self._infer(
                    observation.observation, command.deterministic
                )
            except Exception:
                self._ready = False
                raise
            self._register_observation(
                observation, command.actor_id, command.session_id
            )
            session = _SessionState(
                actor_id=command.actor_id,
                run_id=command.run_id,
                session_id=command.session_id,
                episode_id=command.episode_id,
                current_observation=observation,
                expected_request_id=command.request_id + 1,
                expected_step_id=0,
                last_action=action,
                last_policy_version=version,
            )
            self._sessions[(command.actor_id, command.session_id)] = session
            run.active_session_id = command.session_id
            reply = ActionResult(
                action=action,
                policy_version=version,
                session_id=command.session_id,
                request_id=command.request_id,
                request_created_monotonic_ns=command.created_monotonic_ns,
                observation_id=observation.observation_id,
                server_inference_ms=inference_ms,
            )
            self._store_cache(cache_key, fingerprint, reply)
            return _copy_action_result(reply)

    def step(self, command: StepCommand) -> StepResult:
        fingerprint = command.fingerprint or command_fingerprint(command)
        cache_key = (
            "step",
            command.actor_id,
            command.session_id,
            command.request_id,
        )
        with self._lock:
            cached = self._cached(cache_key, fingerprint)
            if cached is not None:
                result = _copy_step_result(cached)
                return replace(
                    result,
                    ack=replace(result.ack, deduplicated=True),
                )
            self._require_ready()
            self._validate_step_identity(command)
            session_key = (command.actor_id, command.session_id)
            session = self._sessions.get(session_key)
            if session is None or not session.active:
                raise FailedPreconditionError("session is not active")
            if command.run_id != session.run_id:
                raise FailedPreconditionError("run_id does not match active session")
            if command.request_id != session.expected_request_id:
                raise FailedPreconditionError(
                    f"request_id must be {session.expected_request_id}, "
                    f"got {command.request_id}"
                )

            next_observation = self._validated_observation(command.next_observation)
            self._validate_data(command, session, next_observation)
            observation_key = (
                command.actor_id,
                command.session_id,
                next_observation.observation_id,
            )
            if observation_key in self._observations:
                raise ActorProtocolError(
                    f"observation_id {next_observation.observation_id!r} was already accepted"
                )

            data = copy.deepcopy(dict(command.data))
            transition = data["transition"]
            transition["observations"] = copy_observation(
                session.current_observation.observation
            )
            transition["next_observations"] = copy_observation(
                next_observation.observation
            )
            intervened = bool(data["meta"]["intervened"])
            try:
                # One callback owns replay + intervention routing so a real
                # server can make acceptance atomic before ACK is returned.
                self._accept_data(copy.deepcopy(data), intervened)
            except Exception as exc:
                raise ActorNetworkError(
                    f"data sink rejected transition: {type(exc).__name__}: {exc}"
                ) from exc
            self._register_observation(
                next_observation, command.actor_id, command.session_id
            )

            current_key = (
                command.actor_id,
                command.session_id,
                session.current_observation.observation_id,
            )
            self._observations.pop(current_key, None)

            run = self._runs[(session.actor_id, session.run_id)]
            run.next_env_step += 1
            session.current_observation = next_observation
            session.expected_request_id += 1
            session.expected_step_id += 1
            ack = TransitionAck(
                accepted=True,
                transition_id=str(data["meta"]["transition_id"]),
                session_id=session.session_id,
                request_id=command.request_id,
            )

            terminal = bool(transition["dones"] or transition["truncated"])
            if terminal:
                session.active = False
                run.active_session_id = ""
                run.next_episode_id += 1
                self._observations.pop(observation_key, None)
                self._sessions.pop(session_key, None)
                result = StepResult(ack=ack, action=None)
                self._store_cache(cache_key, fingerprint, result)
                return _copy_step_result(result)

            try:
                action, version, inference_ms = self._infer(
                    next_observation.observation, command.deterministic
                )
                if version < session.last_policy_version:
                    raise ActorProtocolError(
                        "policy_version decreased from "
                        f"{session.last_policy_version} to {version}"
                    )
            except Exception as exc:  # transition is already accepted
                self._ready = False
                session.active = False
                run.active_session_id = ""
                self._sessions.pop(session_key, None)
                self._observations.pop(observation_key, None)
                result = StepResult(
                    ack=ack,
                    action=None,
                    action_error=f"policy inference failed: {type(exc).__name__}: {exc}",
                )
                self._store_cache(cache_key, fingerprint, result)
                return _copy_step_result(result)

            session.last_action = action
            session.last_policy_version = version
            action_reply = ActionResult(
                action=action,
                policy_version=version,
                session_id=session.session_id,
                request_id=command.request_id,
                request_created_monotonic_ns=command.created_monotonic_ns,
                observation_id=next_observation.observation_id,
                server_inference_ms=inference_ms,
            )
            result = StepResult(ack=ack, action=action_reply)
            self._store_cache(cache_key, fingerprint, result)
            return _copy_step_result(result)

    def _require_ready(self) -> None:
        if not self._ready:
            raise FailedPreconditionError("policy service is not ready")

    def _accept_data_in_memory(
        self, data: dict[str, Any], intervened: bool
    ) -> None:
        if len(self.replay_items) >= self._in_memory_capacity:
            raise BufferError(
                f"mock replay capacity {self._in_memory_capacity} is full"
            )
        if intervened and len(self.intervention_items) >= self._in_memory_capacity:
            raise BufferError(
                f"mock intervention capacity {self._in_memory_capacity} is full"
            )
        self.replay_items.append(data)
        if intervened:
            self.intervention_items.append(copy.deepcopy(data))

    def _validate_begin(self, command: BeginEpisodeCommand) -> None:
        if command.protocol_version != PROTOCOL_VERSION:
            raise ActorProtocolError(
                f"protocol_version must be {PROTOCOL_VERSION!r}"
            )
        for name, value in (
            ("actor_id", command.actor_id),
            ("run_id", command.run_id),
            ("session_id", command.session_id),
        ):
            if not isinstance(value, str) or not value:
                raise ActorProtocolError(f"{name} is required")
        validate_counter(command.episode_id, name="episode_id")
        if command.request_id != 1:
            raise ActorProtocolError("BeginEpisode request_id must be 1")
        validate_timestamp_ns(
            command.created_monotonic_ns, name="created_monotonic_ns"
        )

    def _validate_step_identity(self, command: StepCommand) -> None:
        if command.protocol_version != PROTOCOL_VERSION:
            raise ActorProtocolError(
                f"protocol_version must be {PROTOCOL_VERSION!r}"
            )
        for name, value in (
            ("actor_id", command.actor_id),
            ("run_id", command.run_id),
            ("session_id", command.session_id),
        ):
            if not isinstance(value, str) or not value:
                raise ActorProtocolError(f"{name} is required")
        if command.request_id <= 1:
            raise ActorProtocolError("Step request_id must be greater than 1")
        validate_timestamp_ns(
            command.created_monotonic_ns, name="created_monotonic_ns"
        )

    def _validated_observation(
        self, packet: ObservationPacket
    ) -> ObservationPacket:
        if not isinstance(packet.observation_id, str) or not packet.observation_id:
            raise ActorProtocolError("observation_id is required")
        timestamp_ns = validate_timestamp_ns(packet.timestamp_ns)
        observation = copy_observation(packet.observation)
        return ObservationPacket(packet.observation_id, timestamp_ns, observation)

    def _register_observation(
        self, packet: ObservationPacket, actor_id: str, session_id: str
    ) -> None:
        key = (actor_id, session_id, packet.observation_id)
        self._observations[key] = packet
        self.observation_accept_count += 1

    def _validate_data(
        self,
        command: StepCommand,
        session: _SessionState,
        next_observation: ObservationPacket,
    ) -> None:
        if not isinstance(command.data, Mapping):
            raise ActorProtocolError("data must be a mapping")
        if set(command.data) != {"meta", "transition"}:
            raise ActorProtocolError("data must contain exactly meta and transition")
        meta = command.data["meta"]
        transition = command.data["transition"]
        if not isinstance(meta, Mapping) or not isinstance(transition, Mapping):
            raise ActorProtocolError("data.meta and data.transition must be mappings")

        schema_version = validate_counter(
            meta.get("schema_version"), name="schema_version"
        )
        if schema_version != SCHEMA_VERSION:
            raise ActorProtocolError(
                f"schema_version must be {SCHEMA_VERSION}, got {schema_version}"
            )
        expected_ids = {
            "run_id": session.run_id,
            "actor_id": session.actor_id,
            "session_id": session.session_id,
        }
        for name, expected in expected_ids.items():
            if meta.get(name) != expected:
                raise ActorProtocolError(f"meta.{name} does not match the session")
        transition_id = meta.get("transition_id")
        if not isinstance(transition_id, str) or not transition_id:
            raise ActorProtocolError("meta.transition_id is required")

        run = self._runs[(session.actor_id, session.run_id)]
        env_step = validate_counter(meta.get("env_step"), name="env_step")
        if env_step != run.next_env_step:
            raise FailedPreconditionError(
                f"env_step must be {run.next_env_step}, got {env_step}"
            )
        timestamp_ns = validate_timestamp_ns(meta.get("timestamp_ns"))
        if timestamp_ns != session.current_observation.timestamp_ns:
            raise ActorProtocolError(
                "meta.timestamp_ns must be the source observation timestamp"
            )
        policy_version = validate_counter(
            meta.get("policy_version"), name="policy_version"
        )
        if policy_version != session.last_policy_version:
            raise ActorProtocolError("meta.policy_version does not match issued action")
        policy_action = validate_action(
            meta.get("policy_action"),
            action_shape=self._action_shape,
            name="meta.policy_action",
        )
        if not np.array_equal(policy_action, session.last_action):
            raise ActorProtocolError("meta.policy_action does not match issued action")
        intervened = meta.get("intervened")
        if isinstance(intervened, np.generic):
            intervened = intervened.item()
        if intervened not in (False, True, 0, 1):
            raise ActorProtocolError("meta.intervened must be 0 or 1")

        episode_id = validate_counter(
            transition.get("episode_id"), name="episode_id"
        )
        step_id = validate_counter(transition.get("step_id"), name="step_id")
        if episode_id != session.episode_id:
            raise ActorProtocolError("transition.episode_id does not match session")
        if step_id != session.expected_step_id:
            raise FailedPreconditionError(
                f"step_id must be {session.expected_step_id}, got {step_id}"
            )
        if transition.get("observation_id") != session.current_observation.observation_id:
            raise ActorProtocolError("transition.observation_id is not the current observation")
        if transition.get("next_observation_id") != next_observation.observation_id:
            raise ActorProtocolError(
                "transition.next_observation_id does not match the attached observation"
            )
        executed_action = validate_action(
            transition.get("actions"),
            action_shape=self._action_shape,
            name="transition.actions",
        )
        if not bool(intervened) and not np.array_equal(executed_action, policy_action):
            raise ActorProtocolError(
                "non-intervention transition.actions must equal policy_action"
            )

        reward = float(transition.get("rewards"))
        mask = float(transition.get("masks"))
        if not math.isfinite(reward) or not math.isfinite(mask):
            raise ActorProtocolError("transition reward/mask must be finite")
        done = transition.get("dones")
        truncated = transition.get("truncated")
        if not isinstance(done, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise ActorProtocolError("transition dones/truncated must be bool")
        if bool(done) and bool(truncated):
            raise ActorProtocolError("transition cannot be both done and truncated")
        expected_mask = 0.0 if bool(done) else 1.0
        if mask != expected_mask:
            raise ActorProtocolError(
                f"transition.masks must be {expected_mask} for dones={bool(done)}"
            )
        if "grasp_penalty" in transition and not math.isfinite(
            float(transition["grasp_penalty"])
        ):
            raise ActorProtocolError("transition.grasp_penalty must be finite")
        terminal = bool(done) or bool(truncated)
        if command.request_action == terminal:
            raise ActorProtocolError(
                "request_action must be false exactly for terminal/truncated steps"
            )

    def _infer(
        self, observation: Mapping[str, Any], deterministic: bool
    ) -> tuple[np.ndarray, int, float]:
        started = self._clock()
        try:
            action_value, version_value = self._sample_action(
                copy_observation(observation), bool(deterministic)
            )
            action = validate_action(
                action_value, action_shape=self._action_shape, name="policy action"
            )
            version = validate_counter(version_value, name="policy_version")
        except Exception as exc:
            raise PolicyInferenceError(
                f"policy callback returned an invalid result: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        inference_ms = (self._clock() - started) * 1000.0
        if not math.isfinite(inference_ms) or inference_ms < 0:
            raise PolicyInferenceError("server inference timing is invalid")
        self.inference_count += 1
        return action, version, inference_ms

    def _cached(
        self, key: tuple[str, str, str, int], fingerprint: bytes
    ) -> Any:
        entry = self._reply_cache.get(key)
        if entry is None:
            return None
        cached_fingerprint, result = entry
        if cached_fingerprint != fingerprint:
            raise ActorProtocolError("duplicate request ID carried different content")
        self._reply_cache.move_to_end(key)
        return result

    def _store_cache(
        self, key: tuple[str, str, str, int], fingerprint: bytes, result: Any
    ) -> None:
        self._reply_cache[key] = (fingerprint, result)
        self._reply_cache.move_to_end(key)
        while len(self._reply_cache) > self._cache_size:
            self._reply_cache.popitem(last=False)


def _copy_action_result(result: ActionResult) -> ActionResult:
    return replace(result, action=np.asarray(result.action, dtype=np.float32).copy())


def _copy_step_result(result: StepResult) -> StepResult:
    action = None if result.action is None else _copy_action_result(result.action)
    return replace(result, action=action)


def create_actor_network(
    config: Mapping[str, Any],
    *,
    actor_id: str,
    action_shape: Tuple[int, ...],
) -> ActorNetwork:
    """Create a transport behind the common API from ``NETWORK`` config."""
    if not isinstance(config, Mapping):
        raise TypeError("NETWORK config must be a mapping")
    network_type = str(config.get("type", "grpc")).strip().lower()
    if network_type == "grpc":
        from ur_env.grpc_actor_transport import GrpcActorNetwork

        return GrpcActorNetwork.from_config(
            config, actor_id=actor_id, action_shape=action_shape
        )
    if network_type == "agentlace":
        raise NotImplementedError(
            "network.type='agentlace' is reserved for a future adapter; "
            "use 'grpc' for the validated v1 path"
        )
    raise ValueError(f"unsupported network.type {network_type!r}")
