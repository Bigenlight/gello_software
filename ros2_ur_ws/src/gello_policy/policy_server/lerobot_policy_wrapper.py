#!/usr/bin/env python3
"""Common LeRobot checkpoint loading and inference lifecycle.

This module contains no ROS or transport logic.  It uses LeRobot's standard
policy factory so ACT, Diffusion, MultiTaskDiT/Flow-Matching, and future policy
classes can share the same load/reset/select_action path when their checkpoint
uses the robot-side observation/action contract.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


class PolicyInputError(ValueError):
    """Input rejected before ``policy.select_action`` can mutate its queues."""


def parse_config_overrides(values: list[str]) -> dict[str, Any]:
    """Parse repeated ``NAME=JSON`` overrides used before model construction."""
    overrides: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"config override must be NAME=JSON, got {value!r}")
        name, encoded = value.split("=", 1)
        if not name or "." in name:
            raise ValueError(f"config override must name one top-level field: {name!r}")
        try:
            overrides[name] = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in config override {value!r}: {exc}") from exc
    return overrides


def _checkpoint_source(checkpoint: str) -> str:
    path = Path(checkpoint).expanduser()
    return str(path.resolve()) if path.exists() else checkpoint


def _shape(feature: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in feature.shape)


def _kind(feature: Any) -> str:
    feature_type = feature.type
    return str(getattr(feature_type, "value", feature_type)).lower()


class LeRobotPolicyWrapper:
    """Policy-class-independent wrapper around a standard LeRobot checkpoint."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
        external_image_size: tuple[int, int] | None | str = "auto",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is not implicit")

        source = _checkpoint_source(checkpoint)
        config = PreTrainedConfig.from_pretrained(source)
        overrides = dict(config_overrides or {})
        # LeRobot's Diffusion config default may request ImageNet weights while
        # constructing the network. A complete checkpoint does not need that
        # download, and deployments intentionally run with HF offline.
        if (
            str(getattr(config, "type", "")).lower() == "diffusion"
            and "pretrained_backbone_weights" not in overrides
            and hasattr(config, "pretrained_backbone_weights")
        ):
            config.pretrained_backbone_weights = None
        for name, value in overrides.items():
            if not hasattr(config, name):
                raise ValueError(f"checkpoint config has no field {name!r}")
            setattr(config, name, value)
        config.device = device
        config.pretrained_path = source

        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(source, config=config)
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=source,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        if external_image_size == "auto":
            configured = getattr(config, "resize_shape", None)
            self.external_image_size = tuple(map(int, configured)) if configured else None
        else:
            self.external_image_size = external_image_size
        self._resize = v2.Resize(list(self.external_image_size)) if self.external_image_size else None
        self.device = device
        self.checkpoint = source
        self.config = config
        self.input_contract = {
            key: {"kind": _kind(feature), "shape": _shape(feature)}
            for key, feature in config.input_features.items()
        }
        self.output_contract = {
            key: {"kind": _kind(feature), "shape": _shape(feature)}
            for key, feature in config.output_features.items()
        }
        self.task_required = any(
            "token" in type(step).__name__.lower()
            for step in getattr(self.preprocessor, "steps", ())
        ) or self._config_requires_task(config)
        self._validate_action_horizon(config)
        self.last_act_metadata = {
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "total_server_ms": 0.0,
            "chunk_refill": False,
            "remaining_chunk_actions": 0,
        }
        self.reset()

    @staticmethod
    def _config_requires_task(config: Any) -> bool:
        policy_type = str(getattr(config, "type", "")).lower().replace("-", "_")
        objective = str(getattr(config, "objective", "")).lower().replace("-", "_")
        return policy_type in {"multi_task_dit", "flow_matching"} or objective == "flow_matching"

    @staticmethod
    def _validate_action_horizon(config: Any) -> None:
        n_action_steps = int(getattr(config, "n_action_steps", 1))
        if n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
        horizon = getattr(config, "horizon", None)
        n_obs_steps = getattr(config, "n_obs_steps", None)
        if horizon is not None and n_obs_steps is not None:
            available = int(horizon) - int(n_obs_steps) + 1
            if n_action_steps > available:
                raise ValueError(
                    f"n_action_steps ({n_action_steps}) must be <= "
                    f"horizon - n_obs_steps + 1 ({available})"
                )

    def describe(self) -> dict[str, Any]:
        return {
            "policy_type": str(self.config.type),
            "checkpoint": self.checkpoint,
            "device": self.device,
            "input_features": self.input_contract,
            "output_features": self.output_contract,
            "task_required": self.task_required,
            "external_image_size": self.external_image_size,
        }

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def decode_image(self, payload: bytes) -> torch.Tensor:
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise PolicyInputError("cv2.imdecode failed for an observation image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        image = image.contiguous().to(torch.float32).div_(255.0)
        return self._resize(image) if self._resize else image

    @torch.inference_mode()
    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        total_started = time.perf_counter()
        missing = set(self.input_contract) - set(observation)
        if missing:
            raise PolicyInputError(f"missing policy input features: {sorted(missing)}")
        if self.task_required and not observation.get("task"):
            raise PolicyInputError("the policy requires a non-empty task string")

        prepared: dict[str, Any] = {}
        for key, contract in self.input_contract.items():
            value = observation[key]
            if contract["kind"] == "visual":
                if not isinstance(value, (bytes, bytearray, memoryview)):
                    raise PolicyInputError(f"{key} must be encoded image bytes")
                prepared[key] = self.decode_image(bytes(value))
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32)
                if tuple(tensor.shape) != contract["shape"]:
                    raise PolicyInputError(
                        f"{key} shape {tuple(tensor.shape)} != {contract['shape']}"
                    )
                prepared[key] = tensor
        if "task" in observation:
            prepared["task"] = str(observation["task"])

        preprocess_started = time.perf_counter()
        processed = self.preprocessor(prepared)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        inference_started = time.perf_counter()
        queue = getattr(self.policy, "_action_queue", None)
        if queue is None:
            queues = getattr(self.policy, "_queues", {})
            queue = queues.get("action") if hasattr(queues, "get") else None
        # Queue emptiness immediately before select_action is the only reliable
        # indication that this call will refill a standard LeRobot action queue.
        chunk_refill = queue is not None and len(queue) == 0
        action = self.policy.select_action(processed)
        if self.device == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        action = self.postprocessor(action)
        if isinstance(action, Mapping):
            if "action" not in action:
                raise ValueError(f"postprocessor returned mapping without 'action': {action.keys()}")
            action = action["action"]
        result = torch.as_tensor(action).squeeze(0).detach().cpu().numpy().astype(np.float64)
        if result.ndim != 1 or not np.all(np.isfinite(result)):
            raise ValueError(f"invalid policy action: shape={result.shape}, finite={np.all(np.isfinite(result))}")
        remaining = 0
        queue = getattr(self.policy, "_action_queue", None)
        if queue is None:
            queues = getattr(self.policy, "_queues", {})
            queue = queues.get("action") if hasattr(queues, "get") else None
        if queue is not None:
            remaining = len(queue)
        self.last_act_metadata = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "total_server_ms": (time.perf_counter() - total_started) * 1000.0,
            "chunk_refill": chunk_refill,
            "remaining_chunk_actions": remaining,
        }
        return result

    def act(self, state: np.ndarray, cam1_jpeg: bytes, cam2_jpeg: bytes, task: str = "") -> np.ndarray:
        """Adapter for the existing remote gRPC service's fixed robot contract."""
        return self.select_action({
            "observation.state": state,
            "observation.images.cam1": cam1_jpeg,
            "observation.images.cam2": cam2_jpeg,
            "task": task,
        })
