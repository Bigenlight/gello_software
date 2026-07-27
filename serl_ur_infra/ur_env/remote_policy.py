"""Request/reply contract for server-side HIL-SERL policy inference.

The learner-facing transition datastore and the latency-sensitive policy
request are intentionally separate.  This module contains no JAX, ROS, or
Agentlace imports; callers provide the request function or action sampler.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping, Optional, Tuple
import uuid

import numpy as np


PROTOCOL_VERSION = "1"
GET_ACTION_REQUEST = "get-action"


def _timestamp_ns(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("observation timestamp must be an integer")
    result = int(value)
    if result <= 0 or result > np.iinfo(np.int64).max:
        raise ValueError("observation timestamp must be a positive int64")
    return result


@dataclass(frozen=True)
class RemoteAction:
    action: np.ndarray
    policy_version: int
    session_id: str
    request_id: int
    observation_timestamp_ns: int
    server_inference_ms: float


class RemotePolicyClient:
    """Strict synchronous client around a custom request function."""

    def __init__(
        self,
        request: Callable[[str, dict[str, Any]], Optional[Mapping[str, Any]]],
        *,
        client_id: str,
        action_shape: Tuple[int, ...],
    ) -> None:
        if not client_id:
            raise ValueError("client_id is required")
        if not action_shape or any(int(dim) <= 0 for dim in action_shape):
            raise ValueError("action_shape must contain positive dimensions")
        self._request = request
        self._client_id = client_id
        self._action_shape = tuple(int(dim) for dim in action_shape)
        self._session_id = ""
        self._next_request_id = 1

    def start_episode(self, session_id: Optional[str] = None) -> str:
        self._session_id = session_id or uuid.uuid4().hex
        self._next_request_id = 1
        return self._session_id

    def get_action(
        self,
        observation: Any,
        *,
        observation_timestamp_ns: int,
        deterministic: bool = False,
    ) -> RemoteAction:
        if not self._session_id:
            raise RuntimeError("start_episode() must be called before get_action()")
        timestamp_ns = _timestamp_ns(observation_timestamp_ns)
        request_id = self._next_request_id
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "client_id": self._client_id,
            "session_id": self._session_id,
            "request_id": request_id,
            "observation_timestamp_ns": np.int64(timestamp_ns),
            "observation": observation,
            "deterministic": bool(deterministic),
        }
        reply = self._request(GET_ACTION_REQUEST, payload)
        if reply is None:
            self._session_id = ""
            raise TimeoutError("remote policy request timed out")
        if not isinstance(reply, Mapping):
            self._session_id = ""
            raise RuntimeError("remote policy returned a non-mapping reply")
        if not bool(reply.get("ok", False)):
            self._session_id = ""
            raise RuntimeError(
                f"remote policy rejected request: {reply.get('error', 'unknown error')}"
            )
        expected = (
            PROTOCOL_VERSION,
            self._session_id,
            request_id,
            timestamp_ns,
        )
        actual = (
            reply.get("protocol_version"),
            reply.get("session_id"),
            int(reply.get("request_id", -1)),
            int(reply.get("observation_timestamp_ns", -1)),
        )
        if actual != expected:
            self._session_id = ""
            raise RuntimeError(
                f"remote policy reply identity mismatch: expected {expected}, got {actual}"
            )

        action = np.asarray(reply.get("action"), dtype=np.float32)
        if action.shape != self._action_shape or not np.all(np.isfinite(action)):
            self._session_id = ""
            raise RuntimeError(
                f"remote policy action must be finite {self._action_shape}, "
                f"got {action.shape}"
            )
        policy_version = int(reply.get("policy_version", -1))
        if policy_version < 0:
            self._session_id = ""
            raise RuntimeError("remote policy_version must be non-negative")
        inference_ms = float(reply.get("server_inference_ms", 0.0))
        if not np.isfinite(inference_ms) or inference_ms < 0:
            self._session_id = ""
            raise RuntimeError("remote server_inference_ms is invalid")

        self._next_request_id += 1
        return RemoteAction(
            action=action.copy(),
            policy_version=policy_version,
            session_id=self._session_id,
            request_id=request_id,
            observation_timestamp_ns=timestamp_ns,
            server_inference_ms=inference_ms,
        )


class RemotePolicyService:
    """Idempotent single-process request handler for an inference runtime."""

    def __init__(
        self,
        sample_action: Callable[[Any, bool], tuple[Any, int]],
        *,
        action_shape: Tuple[int, ...],
        cache_size: int = 1024,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self._sample_action = sample_action
        self._action_shape = tuple(int(dim) for dim in action_shape)
        self._cache_size = cache_size
        self._clock = clock
        self._lock = threading.Lock()
        self._last_request: dict[tuple[str, str], int] = {}
        self._reply_cache: OrderedDict[
            tuple[str, str, int], dict[str, Any]
        ] = OrderedDict()

    def handle(self, request_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if request_type != GET_ACTION_REQUEST:
            return {"ok": False, "error": f"unsupported request {request_type!r}"}
        if not isinstance(payload, Mapping):
            return {"ok": False, "error": "payload must be a mapping"}

        try:
            protocol = str(payload["protocol_version"])
            client_id = str(payload["client_id"])
            session_id = str(payload["session_id"])
            request_id = int(payload["request_id"])
            timestamp_ns = _timestamp_ns(payload["observation_timestamp_ns"])
            observation = payload["observation"]
            deterministic = bool(payload.get("deterministic", False))
            if protocol != PROTOCOL_VERSION:
                raise ValueError(
                    f"protocol_version must be {PROTOCOL_VERSION!r}, got {protocol!r}"
                )
            if not client_id or not session_id:
                raise ValueError("client_id and session_id are required")
            if request_id <= 0:
                raise ValueError("request_id must be positive")
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

        cache_key = (client_id, session_id, request_id)
        session_key = (client_id, session_id)
        with self._lock:
            cached = self._reply_cache.get(cache_key)
            if cached is not None:
                self._reply_cache.move_to_end(cache_key)
                return dict(cached)

            expected_request_id = self._last_request.get(session_key, 0) + 1
            if request_id != expected_request_id:
                return {
                    "ok": False,
                    "error": (
                        f"request_id must be {expected_request_id}, got {request_id}"
                    ),
                }

            started = self._clock()
            try:
                sampled_action, sampled_version = self._sample_action(
                    observation, deterministic
                )
                action = np.asarray(sampled_action, dtype=np.float32)
                version = int(sampled_version)
            except Exception as exc:  # noqa: BLE001 - returned across RPC boundary
                return {
                    "ok": False,
                    "error": f"policy inference failed: {type(exc).__name__}",
                }
            inference_ms = (self._clock() - started) * 1000.0
            if action.shape != self._action_shape or not np.all(np.isfinite(action)):
                return {
                    "ok": False,
                    "error": (
                        f"policy action must be finite {self._action_shape}, "
                        f"got {action.shape}"
                    ),
                }
            if version < 0:
                return {"ok": False, "error": "policy_version must be non-negative"}

            reply = {
                "ok": True,
                "protocol_version": PROTOCOL_VERSION,
                "session_id": session_id,
                "request_id": request_id,
                "observation_timestamp_ns": np.int64(timestamp_ns),
                "action": action.copy(),
                "policy_version": np.int64(version),
                "server_inference_ms": float(inference_ms),
            }
            self._last_request[session_key] = request_id
            self._reply_cache[cache_key] = reply
            while len(self._reply_cache) > self._cache_size:
                self._reply_cache.popitem(last=False)
            return dict(reply)
