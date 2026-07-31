"""Transport-neutral contract for a remote HIL-SERL actor.

The laptop sends an initial observation with :meth:`begin_episode`.  Every
later RPC combines the transition just executed with its next observation, so
large image tensors cross the network once rather than once for inference and
again for replay insertion.

THE CLASSIFIER SIDECAR
----------------------
The reward classifier was trained on UNCROPPED camera frames, while the policy
observation is cropped (``ur_experiments/cube_in_cup.py::IMAGE_CROP``).  Feeding
the policy observation to the classifier costs recall@0.85 100% -> 33.3%, so the
actor additionally ships the classifier its own uncropped JPEG frames.  They
travel inside the existing named-tensor observation map under one reserved key
(``ur_env.classifier_sidecar.CLASSIFIER_SIDECAR_KEY``), which needs no proto
change and does not alter the observation schema hash.

:meth:`ActorSessionService.step` strips that key BEFORE anything else in the
server sees the observation.  Everything downstream — the exact-key check in
``ur_env.observation_schema.validate_canonical_observation``, the policy in
``ur_env.learner.policy``, and replay conversion in
``ur_env.rlpd_receive_server.ReplayIngress`` — therefore keeps operating on the
unchanged canonical tree and needs no relaxation.  Relaxing that exact-key check
instead would have made the schema permissive for every future caller; stripping
one reserved key in one place does not.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, replace
import hashlib
import inspect
import math
import threading
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple

import numpy as np


PROTOCOL_VERSION = "2"
SCHEMA_VERSION = 3

# Resolved on first use, never at import time.  ``ur_env.classifier_sidecar``
# imports ActorProtocolError from this module, so a module-scope import here
# would be circular; a lazy one also keeps the image codec out of processes
# that only want the dataclasses in this file.
_SIDECAR_CONTRACT: Any = None


def _sidecar_contract() -> Any:
    """Return the frozen ``ur_env.classifier_sidecar`` contract module."""
    global _SIDECAR_CONTRACT
    if _SIDECAR_CONTRACT is None:
        from ur_env import classifier_sidecar

        _SIDECAR_CONTRACT = classifier_sidecar
    return _SIDECAR_CONTRACT


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
class TransitionOutcome:
    """Server-authoritative transition values that were accepted by replay."""

    transition_id: str
    reward: float
    mask: float
    done: bool
    truncated: bool
    success: bool
    classifier_evaluated: bool
    classifier_probability: float = 0.0
    classifier_threshold: float = 0.0
    reward_model_id: str = ""

    @property
    def terminal(self) -> bool:
        return self.done or self.truncated


@dataclass(frozen=True)
class StepResult:
    ack: TransitionAck
    outcome: TransitionOutcome
    action: Optional[ActionResult]
    action_error: str = ""


@dataclass(frozen=True)
class ServerInfo:
    ready: bool
    protocol_version: str
    schema_version: int
    action_dim: int
    model_id: str
    reward_authority: str
    reward_model_id: str
    observation_schema_hash: str


@dataclass(frozen=True)
class BufferStatus:
    replay_size: int
    replay_capacity: int
    intervention_size: int
    intervention_capacity: int
    replay_insert_count: int
    intervention_insert_count: int
    replay_overwrite_count: int
    intervention_overwrite_count: int
    last_transition_id: str = ""
    last_env_step: Optional[int] = None
    protocol_version: str = PROTOCOL_VERSION
    schema_version: int = SCHEMA_VERSION


class ActorNetwork(Protocol):
    """Common laptop API implemented by each network transport."""

    def health(self) -> tuple[bool, bool, str]:
        ...

    def get_server_info(self) -> ServerInfo:
        ...

    def get_buffer_status(self) -> BufferStatus:
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


def _values_equal(left: Any, right: Any) -> bool:
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    try:
        _hash_value(left_digest, left)
        _hash_value(right_digest, right)
    except ActorProtocolError:
        return False
    return left_digest.digest() == right_digest.digest()


def _binary_flag(value: Any, *, name: str) -> bool:
    """Return one bool/0/1 protocol flag without accepting truthy values."""

    if isinstance(value, np.generic):
        value = value.item()
    if value not in (False, True, 0, 1):
        raise ActorProtocolError(f"{name} must be 0 or 1")
    return bool(value)


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
        reward_authority: str = "local",
        reward_model_id: str = "",
        observation_schema_hash: str = "",
        cache_size: int = 2048,
        clock: Callable[[], float] = time.perf_counter,
        accept_data: Optional[Callable[[dict[str, Any], bool], None]] = None,
        # Contract: ``finalize_transition(data, classifier_sidecar)`` where
        # ``classifier_sidecar`` is the validated sidecar tensor map for this
        # step's O(t+1) (still JPEG-encoded — the finalizer decodes it), or
        # ``None`` when the actor did not classify this step.  Legacy
        # single-argument finalizers are still accepted; see _bind_finalizer.
        finalize_transition: Optional[
            Callable[..., tuple[dict[str, Any], TransitionOutcome]]
        ] = None,
        buffer_status_provider: Optional[Callable[[], BufferStatus]] = None,
        in_memory_capacity: int = 256,
        allowed_actor_ids: Optional[Tuple[str, ...]] = None,
        allowed_run_ids: Optional[Tuple[str, ...]] = None,
    ) -> None:
        if not action_shape or any(int(dim) <= 0 for dim in action_shape):
            raise ValueError("action_shape must contain positive dimensions")
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if in_memory_capacity <= 0:
            raise ValueError("in_memory_capacity must be positive")
        if not isinstance(reward_authority, str) or not reward_authority:
            raise ValueError("reward_authority is required")
        self._allowed_actor_ids = self._validated_allowlist(
            allowed_actor_ids, name="allowed_actor_ids"
        )
        self._allowed_run_ids = self._validated_allowlist(
            allowed_run_ids, name="allowed_run_ids"
        )
        self._sample_action = sample_action
        self._action_shape = tuple(int(dim) for dim in action_shape)
        self._model_id = str(model_id)
        self._reward_authority = reward_authority
        self._reward_model_id = str(reward_model_id)
        self._observation_schema_hash = str(observation_schema_hash)
        self._cache_size = int(cache_size)
        self._clock = clock
        self._in_memory_capacity = int(in_memory_capacity)
        self._accept_data = accept_data or self._accept_data_in_memory
        self._finalize_transition = self._bind_finalizer(
            finalize_transition or self._finalize_transition_identity
        )
        self._buffer_status_provider = buffer_status_provider
        self._lock = threading.RLock()
        self._ready = True
        self._fault_detail = ""
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
        self._last_transition_id = ""
        self._last_env_step: Optional[int] = None

    @staticmethod
    def _validated_allowlist(
        values: Optional[Tuple[str, ...]], *, name: str
    ) -> Optional[frozenset[str]]:
        if values is None:
            return None
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a non-empty tuple of strings")
        normalized = tuple(values)
        if not normalized or any(
            not isinstance(value, str) or not value for value in normalized
        ):
            raise ValueError(f"{name} must be a non-empty tuple of strings")
        return frozenset(normalized)

    def health(self) -> tuple[bool, bool, str]:
        with self._lock:
            return True, self._ready, "ready" if self._ready else self._fault_detail

    def get_server_info(self) -> ServerInfo:
        with self._lock:
            return ServerInfo(
                ready=self._ready,
                protocol_version=PROTOCOL_VERSION,
                schema_version=SCHEMA_VERSION,
                action_dim=int(np.prod(self._action_shape)),
                model_id=self._model_id,
                reward_authority=self._reward_authority,
                reward_model_id=self._reward_model_id,
                observation_schema_hash=self._observation_schema_hash,
            )

    def get_buffer_status(self) -> BufferStatus:
        with self._lock:
            status = (
                self._buffer_status_provider()
                if self._buffer_status_provider is not None
                else self._in_memory_buffer_status()
            )
            return self._validated_buffer_status(status)

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
            if (
                self._allowed_actor_ids is not None
                and command.actor_id not in self._allowed_actor_ids
            ):
                raise FailedPreconditionError(
                    f"actor_id {command.actor_id!r} is not allowed by this server"
                )
            if (
                self._allowed_run_ids is not None
                and command.run_id not in self._allowed_run_ids
            ):
                raise FailedPreconditionError(
                    f"run_id {command.run_id!r} is not allowed by this server"
                )

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

            # A sidecar on BeginEpisode is a protocol error, not a no-op: the
            # sidecar labels the transition that ENDS on this observation, and
            # BeginEpisode has no transition.  Accepting and dropping it would
            # silently lose a classification the actor believed it had paid for.
            self._reject_classifier_sidecar(command.observation, rpc="BeginEpisode")
            observation = self._validated_observation(command.observation)
            shared_features = self._prime_replay_observation(
                observation, command.actor_id, command.session_id
            )
            try:
                action, version, inference_ms = self._infer(
                    shared_features
                    if shared_features is not None
                    else observation.observation,
                    command.deterministic,
                )
            except Exception as exc:
                self._set_fault(
                    f"policy inference failed: {type(exc).__name__}: {exc}"
                )
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

            # Separate the classifier sidecar FIRST.  From here on nothing in
            # this process — canonical validation, the policy, the finalizer's
            # protected-field diff, replay conversion — can see the reserved
            # key, so the canonical observation contract stays exact-key strict.
            next_packet, classifier_sidecar = self._split_classifier_sidecar(
                command.next_observation
            )
            next_observation = self._validated_observation(next_packet)
            self._validate_data(command, session, next_observation)
            # Run the image encoder ONCE for this step, before anything that
            # wants its output.  _accept_data below finds it already cached
            # (so the trunk does not run again inside the learner-shared
            # lock), and _infer serves the policy from the same tensor.
            shared_features = self._prime_replay_observation(
                next_observation, command.actor_id, command.session_id
            )
            observation_key = (
                command.actor_id,
                command.session_id,
                next_observation.observation_id,
            )
            if observation_key in self._observations:
                raise ActorProtocolError(
                    f"observation_id {next_observation.observation_id!r} was already accepted"
                )

            provisional_data = copy.deepcopy(dict(command.data))
            provisional_transition = provisional_data["transition"]
            provisional_transition["observations"] = copy_observation(
                session.current_observation.observation
            )
            provisional_transition["next_observations"] = copy_observation(
                next_observation.observation
            )
            try:
                finalized = self._finalize_transition(
                    copy.deepcopy(provisional_data), classifier_sidecar
                )
                if not isinstance(finalized, tuple) or len(finalized) != 2:
                    raise ActorProtocolError(
                        "transition finalizer must return (data, outcome)"
                    )
                finalized_data, outcome = finalized
                if not isinstance(finalized_data, Mapping):
                    raise ActorProtocolError(
                        "transition finalizer data must be a mapping"
                    )
                data = copy.deepcopy(dict(finalized_data))
                outcome = self._validate_finalized_transition(
                    provisional_data, data, outcome
                )
                intervened = bool(data["meta"]["intervened"])
                # One callback owns replay + intervention routing so a real
                # server can make acceptance atomic before ACK is returned.
                self._accept_data(copy.deepcopy(data), intervened)
            except Exception as exc:
                self._set_fault(
                    f"transition pipeline failed: {type(exc).__name__}: {exc}"
                )
                raise ActorNetworkError(
                    f"transition was not acknowledged: {type(exc).__name__}: {exc}"
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
            self._last_transition_id = outcome.transition_id
            self._last_env_step = int(data["meta"]["env_step"])
            ack = TransitionAck(
                accepted=True,
                transition_id=str(data["meta"]["transition_id"]),
                session_id=session.session_id,
                request_id=command.request_id,
            )

            if outcome.terminal:
                session.active = False
                run.active_session_id = ""
                run.next_episode_id += 1
                self._observations.pop(observation_key, None)
                self._sessions.pop(session_key, None)
                result = StepResult(ack=ack, outcome=outcome, action=None)
                self._store_cache(cache_key, fingerprint, result)
                return _copy_step_result(result)

            try:
                action, version, inference_ms = self._infer(
                    shared_features
                    if shared_features is not None
                    else next_observation.observation,
                    command.deterministic,
                )
                if version < session.last_policy_version:
                    raise ActorProtocolError(
                        "policy_version decreased from "
                        f"{session.last_policy_version} to {version}"
                    )
            except Exception as exc:  # transition is already accepted
                self._set_fault(
                    f"policy inference failed: {type(exc).__name__}: {exc}"
                )
                session.active = False
                run.active_session_id = ""
                self._sessions.pop(session_key, None)
                self._observations.pop(observation_key, None)
                result = StepResult(
                    ack=ack,
                    outcome=outcome,
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
            result = StepResult(ack=ack, outcome=outcome, action=action_reply)
            self._store_cache(cache_key, fingerprint, result)
            return _copy_step_result(result)

    def _require_ready(self) -> None:
        if not self._ready:
            raise FailedPreconditionError(
                f"actor service is not ready: {self._fault_detail}"
            )

    def _set_fault(self, detail: str) -> None:
        self._ready = False
        self._fault_detail = str(detail) or "actor service fault"

    @staticmethod
    def _bind_finalizer(
        finalizer: Callable[..., tuple[dict[str, Any], TransitionOutcome]]
    ) -> Callable[..., tuple[dict[str, Any], TransitionOutcome]]:
        """Accept the two-argument contract, bridging legacy one-argument ones.

        The finalizer contract grew a second positional argument (the classifier
        sidecar) when the actor started shipping uncropped frames.  Receive-only
        harnesses that predate it still pass ``finalize(data)``; those keep
        working, but they FAIL CLOSED the first time an actor actually sends a
        sidecar rather than silently discarding a classification.
        """
        try:
            signature = inspect.signature(finalizer)
        except (TypeError, ValueError):
            # Builtins / C callables expose no signature.  Assume the current
            # contract rather than downgrading them behind the operator's back.
            return finalizer
        parameters = list(signature.parameters.values())
        takes_var_positional = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        positional_count = sum(
            1
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if takes_var_positional or positional_count >= 2:
            return finalizer

        def legacy_finalizer(
            data: dict[str, Any], classifier_sidecar: Optional[Mapping[str, Any]]
        ) -> tuple[dict[str, Any], TransitionOutcome]:
            if classifier_sidecar is not None:
                raise ActorProtocolError(
                    "this server's transition finalizer predates the classifier "
                    "sidecar and cannot classify the actor's uncropped frames"
                )
            return finalizer(data)

        return legacy_finalizer

    def _reject_classifier_sidecar(
        self, packet: ObservationPacket, *, rpc: str
    ) -> None:
        observation = packet.observation
        if not isinstance(observation, Mapping):
            return  # _validated_observation reports the real shape error
        key = _sidecar_contract().CLASSIFIER_SIDECAR_KEY
        if key in observation:
            raise ActorProtocolError(
                f"{rpc} observation must not carry the reserved {key!r} "
                "classifier sidecar; it belongs on the Step that ends there"
            )

    def _split_classifier_sidecar(
        self, packet: ObservationPacket
    ) -> tuple[ObservationPacket, Optional[dict[str, np.ndarray]]]:
        """Return ``(observation without the sidecar, validated sidecar|None)``.

        Called before ANY other inspection of the observation.  ``validate_``
        ``sidecar`` owns the tensor contract (key set, dtype, rank); a malformed
        sidecar is a protocol error, exactly like a malformed observation, and
        must not be silently downgraded to "no classification this step".
        """
        observation = packet.observation
        if not isinstance(observation, Mapping):
            return packet, None
        contract = _sidecar_contract()
        key = contract.CLASSIFIER_SIDECAR_KEY
        if key not in observation:
            return packet, None
        try:
            sidecar = contract.validate_sidecar(observation[key])
        except ValueError as exc:
            # validate_sidecar speaks ValueError so the actor can call it before
            # any transport exists.  On this side a bad payload is a wire
            # contract violation, and only ActorProtocolError makes the gRPC
            # servicer answer INVALID_ARGUMENT instead of INTERNAL.
            raise ActorProtocolError(f"invalid {key!r} sidecar: {exc}") from exc
        cleaned = {
            name: value for name, value in observation.items() if name != key
        }
        return replace(packet, observation=cleaned), sidecar

    def _finalize_transition_identity(
        self,
        data: dict[str, Any],
        classifier_sidecar: Optional[Mapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], TransitionOutcome]:
        # The identity finalizer has no classifier, so it ignores the sidecar
        # and reports classifier_evaluated=False for every transition.
        del classifier_sidecar
        meta = data["meta"]
        transition = data["transition"]
        auto_success = _binary_flag(
            meta.get("auto_success", False), name="meta.auto_success"
        )
        operator_success = _binary_flag(
            meta.get("operator_success", False), name="meta.operator_success"
        )
        if auto_success and operator_success:
            raise ActorProtocolError(
                "meta.operator_success is forbidden while auto_success is enabled"
            )
        effective_success = operator_success
        if effective_success:
            transition.update(
                rewards=1.0,
                masks=0.0,
                dones=True,
                truncated=False,
            )
        transition["success"] = np.uint8(effective_success)
        transition["classifier_evaluated"] = np.uint8(0)
        transition["classifier_probability"] = 0.0
        transition["classifier_threshold"] = 0.0
        transition["classifier_success"] = np.uint8(0)
        transition["reward_model_id"] = ""
        return data, TransitionOutcome(
            transition_id=str(meta["transition_id"]),
            reward=float(transition["rewards"]),
            mask=float(transition["masks"]),
            done=bool(transition["dones"]),
            truncated=bool(transition["truncated"]),
            success=effective_success,
            classifier_evaluated=False,
        )

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

    def _in_memory_buffer_status(self) -> BufferStatus:
        return BufferStatus(
            replay_size=len(self.replay_items),
            replay_capacity=self._in_memory_capacity,
            intervention_size=len(self.intervention_items),
            intervention_capacity=self._in_memory_capacity,
            replay_insert_count=len(self.replay_items),
            intervention_insert_count=len(self.intervention_items),
            replay_overwrite_count=0,
            intervention_overwrite_count=0,
            last_transition_id=self._last_transition_id,
            last_env_step=self._last_env_step,
        )

    def _validated_buffer_status(self, status: BufferStatus) -> BufferStatus:
        if not isinstance(status, BufferStatus):
            raise ActorProtocolError(
                "buffer status provider must return BufferStatus"
            )
        if status.protocol_version != PROTOCOL_VERSION:
            raise ActorProtocolError(
                "buffer status provider returned incompatible protocol_version "
                f"{status.protocol_version!r}; expected {PROTOCOL_VERSION!r}"
            )
        if status.schema_version != SCHEMA_VERSION:
            raise ActorProtocolError(
                "buffer status provider returned incompatible schema_version "
                f"{status.schema_version}; expected {SCHEMA_VERSION}"
            )
        values = {}
        for name in (
            "replay_size",
            "replay_capacity",
            "intervention_size",
            "intervention_capacity",
            "replay_insert_count",
            "intervention_insert_count",
            "replay_overwrite_count",
            "intervention_overwrite_count",
        ):
            values[name] = validate_counter(getattr(status, name), name=name)
        if values["replay_size"] > values["replay_capacity"]:
            raise ActorProtocolError("replay_size exceeds replay_capacity")
        if values["intervention_size"] > values["intervention_capacity"]:
            raise ActorProtocolError(
                "intervention_size exceeds intervention_capacity"
            )
        if not isinstance(status.last_transition_id, str):
            raise ActorProtocolError("last_transition_id must be a string")
        if (status.last_env_step is None) != (not status.last_transition_id):
            raise ActorProtocolError(
                "last_transition_id and last_env_step must be set together"
            )
        last_env_step = None
        if status.last_env_step is not None:
            last_env_step = validate_counter(
                status.last_env_step, name="last_env_step"
            )
        return BufferStatus(
            **values,
            last_transition_id=status.last_transition_id,
            last_env_step=last_env_step,
        )

    def _validate_finalized_transition(
        self,
        provisional_data: Mapping[str, Any],
        finalized_data: Mapping[str, Any],
        outcome: TransitionOutcome,
    ) -> TransitionOutcome:
        if not isinstance(outcome, TransitionOutcome):
            raise ActorProtocolError(
                "transition finalizer outcome must be TransitionOutcome"
            )
        if set(finalized_data) != {"meta", "transition"}:
            raise ActorProtocolError(
                "finalized data must contain exactly meta and transition"
            )
        meta = finalized_data["meta"]
        transition = finalized_data["transition"]
        if not isinstance(meta, Mapping) or not isinstance(transition, Mapping):
            raise ActorProtocolError(
                "finalized data.meta and data.transition must be mappings"
            )

        provisional_meta = provisional_data["meta"]
        provisional_transition = provisional_data["transition"]
        for name, value in provisional_meta.items():
            if name not in meta or not _values_equal(meta[name], value):
                raise ActorProtocolError(
                    f"transition finalizer changed protected meta.{name}"
                )
        mutable_fields = {"rewards", "masks", "dones", "truncated"}
        for name, value in provisional_transition.items():
            if name in mutable_fields:
                continue
            if name not in transition or not _values_equal(transition[name], value):
                raise ActorProtocolError(
                    f"transition finalizer changed protected transition.{name}"
                )

        transition_id = meta.get("transition_id")
        if outcome.transition_id != transition_id:
            raise ActorProtocolError(
                "outcome.transition_id does not match finalized data"
            )
        reward = float(transition.get("rewards"))
        mask = float(transition.get("masks"))
        if not math.isfinite(reward) or not math.isfinite(mask):
            raise ActorProtocolError("finalized reward/mask must be finite")
        done = transition.get("dones")
        truncated = transition.get("truncated")
        if not isinstance(done, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise ActorProtocolError("finalized dones/truncated must be bool")
        done = bool(done)
        truncated = bool(truncated)
        if done and truncated:
            raise ActorProtocolError(
                "finalized transition cannot be both done and truncated"
            )
        expected_mask = 0.0 if done else 1.0
        if mask != expected_mask:
            raise ActorProtocolError(
                f"finalized masks must be {expected_mask} for dones={done}"
            )

        for name, value in (
            ("outcome.done", outcome.done),
            ("outcome.truncated", outcome.truncated),
            ("outcome.success", outcome.success),
            ("outcome.classifier_evaluated", outcome.classifier_evaluated),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise ActorProtocolError(f"{name} must be bool")
        outcome_reward = float(outcome.reward)
        outcome_mask = float(outcome.mask)
        probability = float(outcome.classifier_probability)
        threshold = float(outcome.classifier_threshold)
        if not all(
            math.isfinite(value)
            for value in (outcome_reward, outcome_mask, probability, threshold)
        ):
            raise ActorProtocolError("outcome contains a non-finite scalar")
        if (
            outcome_reward != reward
            or outcome_mask != mask
            or bool(outcome.done) != done
            or bool(outcome.truncated) != truncated
        ):
            raise ActorProtocolError(
                "outcome reward/mask/terminal flags do not match finalized data"
            )

        evaluated = bool(outcome.classifier_evaluated)
        success = bool(outcome.success)
        transition_evaluated = _binary_flag(
            transition.get("classifier_evaluated", evaluated),
            name="transition.classifier_evaluated",
        )
        classifier_success = _binary_flag(
            transition.get(
                "classifier_success",
                evaluated and probability > threshold,
            ),
            name="transition.classifier_success",
        )
        transition_success = _binary_flag(
            transition.get("success", success), name="transition.success"
        )
        transition_probability = float(
            transition.get("classifier_probability", probability)
        )
        transition_threshold = float(
            transition.get("classifier_threshold", threshold)
        )
        transition_reward_model_id = transition.get(
            "reward_model_id", outcome.reward_model_id
        )
        if not isinstance(transition_reward_model_id, str):
            raise ActorProtocolError(
                "transition.reward_model_id must be a string"
            )
        if not all(
            math.isfinite(value)
            for value in (transition_probability, transition_threshold)
        ):
            raise ActorProtocolError(
                "finalized classifier scalars must be finite"
            )
        if (
            transition_evaluated != evaluated
            or transition_probability != probability
            or transition_threshold != threshold
            or transition_reward_model_id != outcome.reward_model_id
        ):
            raise ActorProtocolError(
                "outcome classifier fields do not match finalized data"
            )
        if evaluated:
            if not 0.0 <= probability <= 1.0:
                raise ActorProtocolError(
                    "classifier_probability must be within [0, 1]"
                )
            if not 0.0 <= threshold <= 1.0:
                raise ActorProtocolError(
                    "classifier_threshold must be within [0, 1]"
                )
            if not isinstance(outcome.reward_model_id, str) or not outcome.reward_model_id:
                raise ActorProtocolError(
                    "reward_model_id is required when classifier was evaluated"
                )
            if self._reward_model_id and outcome.reward_model_id != self._reward_model_id:
                raise ActorProtocolError(
                    "outcome.reward_model_id does not match ServerInfo"
                )
            if classifier_success != (probability > threshold):
                raise ActorProtocolError(
                    "transition.classifier_success must use strict "
                    "probability > threshold"
                )
        elif classifier_success or probability != 0.0 or threshold != 0.0:
            raise ActorProtocolError(
                "unevaluated classifier outcome must have no classifier results"
            )
        elif outcome.reward_model_id:
            raise ActorProtocolError(
                "unevaluated classifier outcome must not name a reward model"
            )

        auto_success = _binary_flag(
            provisional_meta.get("auto_success", False),
            name="meta.auto_success",
        )
        operator_success = _binary_flag(
            provisional_meta.get("operator_success", False),
            name="meta.operator_success",
        )
        if auto_success and operator_success:
            raise ActorProtocolError(
                "meta.operator_success is forbidden while auto_success is enabled"
            )
        effective_success = operator_success or (
            auto_success and classifier_success
        )
        if transition_success != effective_success or success != effective_success:
            raise ActorProtocolError(
                "finalized success does not match operator_success OR "
                "(auto_success AND classifier_success)"
            )

        if success and (reward != 1.0 or not done or truncated or mask != 0.0):
            raise ActorProtocolError(
                "effective success must finalize reward=1, done=true, "
                "truncated=false, mask=0"
            )
        provisional_done = bool(provisional_transition["dones"])
        provisional_truncated = bool(provisional_transition["truncated"])
        if provisional_done and not done:
            raise ActorProtocolError("finalizer cannot clear a local done")
        if provisional_truncated and not (
            truncated or (success and done)
        ):
            raise ActorProtocolError("finalizer cannot clear a local truncation")

        return TransitionOutcome(
            transition_id=str(outcome.transition_id),
            reward=outcome_reward,
            mask=outcome_mask,
            done=done,
            truncated=truncated,
            success=success,
            classifier_evaluated=evaluated,
            classifier_probability=probability,
            classifier_threshold=threshold,
            reward_model_id=outcome.reward_model_id,
        )

    def _validate_begin(self, command: BeginEpisodeCommand) -> None:
        if command.protocol_version != PROTOCOL_VERSION:
            raise ActorProtocolError(
                "incompatible protocol_version "
                f"{command.protocol_version!r}; expected {PROTOCOL_VERSION!r}"
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
                "incompatible protocol_version "
                f"{command.protocol_version!r}; expected {PROTOCOL_VERSION!r}"
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
        auto_success = _binary_flag(
            meta.get("auto_success", False), name="meta.auto_success"
        )
        operator_success = _binary_flag(
            meta.get("operator_success", False),
            name="meta.operator_success",
        )
        if auto_success and operator_success:
            raise ActorProtocolError(
                "meta.operator_success is forbidden while auto_success is enabled"
            )

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

    def _prime_replay_observation(
        self, observation: Any, actor_id: str, session_id: str
    ) -> Any:
        """Encode this observation once, via a sink that shares its result.

        Returns the frozen-trunk features, or ``None`` when the sink does not
        encode -- ``accept_data`` is a plain callable in the receive-only
        server and in tests, and a sink that stores raw observations has
        nothing to share.  Callers fall back to the pixel path on ``None``.

        A failure propagates; see ``FeatureReplayIngress.prime_observation``
        for why that is the safe direction.
        """

        prime = getattr(self._accept_data, "prime_observation", None)
        if not callable(prime):
            return None
        return prime(
            actor_id=actor_id,
            session_id=session_id,
            observation_id=observation.observation_id,
            observation=observation.observation,
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
