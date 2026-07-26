"""Synthetic end-to-end smoke helpers for the remote actor contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from typing import Any, Callable, Mapping, Optional
import uuid

import numpy as np

from ur_env.actor_network import (
    ActorNetwork,
    ActorProtocolError,
    ServerInfo,
    validate_timestamp_ns,
)
from ur_env.remote_actor import build_data


MOCK_MODEL_ID = "mock-zero-policy"
IMAGE_SHAPE = (128, 128, 3)


@dataclass(frozen=True)
class SmokeResult:
    run_id: str
    session_id: str
    transition_ids: tuple[str, str]
    server_info: ServerInfo
    begin_round_trip_ms: float
    normal_step_round_trip_ms: float


class SummaryDataSink:
    """Bounded mock sink retaining metadata summaries, never tensor values."""

    def __init__(
        self,
        *,
        capacity: int = 256,
        emit: Optional[Callable[[str], None]] = print,
        max_tensor_specs: int = 8,
        max_path_chars: int = 96,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_tensor_specs <= 0 or max_path_chars <= 0:
            raise ValueError("summary limits must be positive")
        self._capacity = int(capacity)
        self._emit = emit
        self._max_tensor_specs = int(max_tensor_specs)
        self._max_path_chars = int(max_path_chars)
        self._lock = threading.Lock()
        self.summaries: list[dict[str, Any]] = []
        self.replay_count = 0
        self.intervention_count = 0
        self.log_error_count = 0

    def __call__(self, data: dict[str, Any], intervened: bool) -> None:
        summary = self._summarize(data, intervened)
        with self._lock:
            if self.replay_count >= self._capacity:
                raise BufferError(
                    f"mock summary capacity {self._capacity} is full"
                )
            next_intervention_count = self.intervention_count + int(intervened)
            if next_intervention_count > self._capacity:
                raise BufferError(
                    f"mock intervention capacity {self._capacity} is full"
                )
            self.replay_count += 1
            self.intervention_count = next_intervention_count
            summary["replay_count"] = self.replay_count
            summary["intervention_count"] = self.intervention_count
            self.summaries.append(summary)
            line = json.dumps(summary, sort_keys=True, separators=(",", ":"))

        if self._emit is not None:
            try:
                self._emit(line)
            except (BrokenPipeError, OSError):
                with self._lock:
                    self.log_error_count += 1

    def _summarize(
        self, data: Mapping[str, Any], intervened: bool
    ) -> dict[str, Any]:
        meta = data["meta"]
        transition = data["transition"]
        if bool(meta["intervened"]) != bool(intervened):
            raise ActorProtocolError("sink intervention label mismatch")
        return {
            "event": "transition_accepted",
            "transition_id": _bounded_text(
                meta["transition_id"], self._max_path_chars
            ),
            "run_id": _bounded_text(meta["run_id"], self._max_path_chars),
            "actor_id": _bounded_text(meta["actor_id"], self._max_path_chars),
            "session_id": _bounded_text(
                meta["session_id"], self._max_path_chars
            ),
            "env_step": int(meta["env_step"]),
            "episode_id": int(transition["episode_id"]),
            "step_id": int(transition["step_id"]),
            "timestamp_ns": int(meta["timestamp_ns"]),
            "policy_version": int(meta["policy_version"]),
            "intervened": int(bool(intervened)),
            "observation_id": _bounded_text(
                transition["observation_id"], self._max_path_chars
            ),
            "next_observation_id": _bounded_text(
                transition["next_observation_id"], self._max_path_chars
            ),
            "dones": bool(transition["dones"]),
            "truncated": bool(transition["truncated"]),
            "observation_tensors": self._tensor_summary(
                transition["observations"]
            ),
            "next_observation_tensors": self._tensor_summary(
                transition["next_observations"]
            ),
        }

    def _tensor_summary(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        specs: list[dict[str, Any]] = []

        def visit(node: Any, path: tuple[str, ...]) -> None:
            if isinstance(node, Mapping):
                for key in sorted(node):
                    visit(node[key], path + (str(key),))
                return
            array = np.asarray(node)
            specs.append(
                {
                    "path": _bounded_text(
                        "/".join(path), self._max_path_chars
                    ),
                    "dtype": str(array.dtype),
                    "shape": [int(dim) for dim in array.shape],
                }
            )

        visit(observation, ())
        total = len(specs)
        return {
            "total": total,
            "shown": specs[: self._max_tensor_specs],
            "omitted": max(0, total - self._max_tensor_specs),
        }


def synthetic_observation(index: int) -> dict[str, Any]:
    """Return a small state plus two realistic raw RGB image tensors."""
    image_value = int(index) % 256
    return {
        "state": {
            "tcp_pose": np.linspace(
                -0.25 + index * 0.01,
                0.25 + index * 0.01,
                7,
                dtype=np.float32,
            ),
            "gripper_pos": np.array([index * 0.1], dtype=np.float32),
        },
        "images": {
            "wrist": np.full(IMAGE_SHAPE, image_value, dtype=np.uint8),
            "side": np.full(
                IMAGE_SHAPE, 255 - image_value, dtype=np.uint8
            ),
        },
    }


def run_mock_smoke(
    network: ActorNetwork,
    *,
    actor_id: str,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None,
    wall_time_ns: Callable[[], int] = time.time_ns,
) -> SmokeResult:
    """Send one normal and one terminal intervention transition."""
    if not actor_id:
        raise ValueError("actor_id is required")
    run_id = run_id or f"smoke-{uuid.uuid4().hex}"
    session_id = session_id or uuid.uuid4().hex
    alive, ready, detail = network.health()
    if not alive or not ready:
        raise RuntimeError(
            f"mock server is not ready: alive={alive}, ready={ready}, "
            f"detail={detail}"
        )
    server_info = network.get_server_info()
    if server_info.model_id != MOCK_MODEL_ID:
        raise ActorProtocolError(
            f"smoke requires model_id={MOCK_MODEL_ID!r}, "
            f"got {server_info.model_id!r}"
        )

    timestamp0 = validate_timestamp_ns(wall_time_ns())
    if timestamp0 > np.iinfo(np.int64).max - 2:
        raise ActorProtocolError("smoke timestamp cannot be incremented safely")
    timestamp1 = timestamp0 + 1
    timestamp2 = timestamp0 + 2
    observation_ids = tuple(f"{session_id}:{index}" for index in range(3))
    transition_ids = (f"{run_id}:0", f"{run_id}:1")

    action0 = network.begin_episode(
        synthetic_observation(0),
        run_id=run_id,
        session_id=session_id,
        episode_id=0,
        observation_id=observation_ids[0],
        timestamp_ns=timestamp0,
        deterministic=True,
    )
    _require_zero_action(action0.action, action0.policy_version)

    data0 = build_data(
        actor_id=actor_id,
        run_id=run_id,
        session_id=session_id,
        transition_id=transition_ids[0],
        env_step=0,
        timestamp_ns=timestamp0,
        policy_version=action0.policy_version,
        policy_action=action0.action,
        episode_id=0,
        step_id=0,
        observation_id=observation_ids[0],
        next_observation_id=observation_ids[1],
        reward=0.0,
        done=False,
        truncated=False,
        info={"intervened": 0},
    )
    normal = network.step(
        synthetic_observation(1),
        next_observation_id=observation_ids[1],
        next_timestamp_ns=timestamp1,
        data=data0,
        request_action=True,
        deterministic=True,
    )
    if normal.ack.transition_id != transition_ids[0] or normal.action is None:
        raise ActorProtocolError("normal Step did not return the expected ACK/action")
    _require_zero_action(normal.action.action, normal.action.policy_version)

    data1 = build_data(
        actor_id=actor_id,
        run_id=run_id,
        session_id=session_id,
        transition_id=transition_ids[1],
        env_step=1,
        timestamp_ns=timestamp1,
        policy_version=normal.action.policy_version,
        policy_action=normal.action.action,
        episode_id=0,
        step_id=1,
        observation_id=observation_ids[1],
        next_observation_id=observation_ids[2],
        reward=1.0,
        done=True,
        truncated=False,
        info={
            "intervened": 1,
            "intervene_action": np.full(7, -0.25, dtype=np.float32),
        },
    )
    terminal = network.step(
        synthetic_observation(2),
        next_observation_id=observation_ids[2],
        next_timestamp_ns=timestamp2,
        data=data1,
        request_action=False,
        deterministic=True,
    )
    if terminal.ack.transition_id != transition_ids[1]:
        raise ActorProtocolError("terminal Step returned the wrong ACK")
    if terminal.action is not None:
        raise ActorProtocolError("terminal Step unexpectedly returned an action")

    return SmokeResult(
        run_id=run_id,
        session_id=session_id,
        transition_ids=transition_ids,
        server_info=server_info,
        begin_round_trip_ms=action0.round_trip_ms,
        normal_step_round_trip_ms=normal.action.round_trip_ms,
    )


def _require_zero_action(action: Any, policy_version: int) -> None:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (7,) or not np.array_equal(array, np.zeros(7, np.float32)):
        raise ActorProtocolError("mock server must return exactly zeros(7)")
    if int(policy_version) != 0:
        raise ActorProtocolError("mock server policy_version must be 0")


def _bounded_text(value: Any, max_chars: int) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
