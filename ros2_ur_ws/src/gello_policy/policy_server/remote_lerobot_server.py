#!/usr/bin/env python3
"""Kanu/GPU gRPC server for any compatible LeRobot policy checkpoint.

The validated remote Diffusion transport, session ownership and request-ordering
logic are reused unchanged. Only checkpoint loading/inference is delegated to the
policy-class-independent LeRobot wrapper.
"""

from __future__ import annotations

from concurrent import futures
import json
import os

import grpc
import cv2
import numpy as np

from .remote_diffusion_server import (
    DEFAULT_BIND_ADDRESS,
    DEFAULT_MAX_JPEG_BYTES,
    DEFAULT_MAX_MESSAGE_BYTES,
    RemoteDiffusionService,
    _required_env,
)
from . import remote_diffusion_pb2_grpc as pb_grpc


def _image_size(value: str):
    value = value.strip().lower()
    if value == "auto":
        return "auto"
    if value == "native":
        return None
    try:
        height, width = value.split("x", 1)
        return int(height), int(width)
    except Exception as exc:
        raise SystemExit("EXTERNAL_IMAGE_SIZE must be auto, native, or HxW") from exc


def _sampling_contract(config) -> tuple[str, int]:
    objective = getattr(config, "objective", None)
    method = f"{config.type}:{objective}" if objective else str(config.type)
    steps = getattr(config, "num_inference_steps", None)
    if steps is None:
        steps = getattr(config, "num_integration_steps", 1)
    return method, int(steps or 1)


def _warmup(wrapper, task: str, state: np.ndarray) -> None:
    visual = wrapper.input_contract["observation.images.cam1"]["shape"]
    if len(visual) != 3:
        raise SystemExit(f"camera feature must be CHW, got {visual}")
    height, width = visual[-2], visual[-1]
    ok, encoded = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    if not ok:
        raise SystemExit("failed to create warm-up JPEG")
    jpeg = encoded.tobytes()
    print("[remote_lerobot] warming CUDA/policy with synthetic observations", flush=True)
    wrapper.act(state, jpeg, jpeg, task)
    wrapper.act(state, jpeg, jpeg, task)
    wrapper.reset()
    print("[remote_lerobot] warm-up complete; policy queues reset", flush=True)


def main() -> int:
    from .lerobot_policy_wrapper import LeRobotPolicyWrapper, PolicyInputError
    from .remote_diffusion_server import RequestValidationError

    checkpoint = _required_env("CHECKPOINT_PATH")
    bind_address = os.environ.get("GRPC_BIND_ADDRESS", DEFAULT_BIND_ADDRESS)
    device = os.environ.get("POLICY_DEVICE", "cuda")
    task = os.environ.get("POLICY_TASK", "")
    try:
        overrides = json.loads(os.environ.get("POLICY_CONFIG_OVERRIDES", "{}"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"POLICY_CONFIG_OVERRIDES is not valid JSON: {exc}") from exc
    if not isinstance(overrides, dict):
        raise SystemExit("POLICY_CONFIG_OVERRIDES must be a JSON object")

    wrapper = LeRobotPolicyWrapper(
        checkpoint,
        device=device,
        config_overrides=overrides,
        external_image_size=_image_size(os.environ.get("EXTERNAL_IMAGE_SIZE", "auto")),
    )
    expected_inputs = {
        "observation.state", "observation.images.cam1", "observation.images.cam2",
    }
    if set(wrapper.input_contract) != expected_inputs:
        raise SystemExit(
            "checkpoint is incompatible with robot protocol inputs: "
            f"expected {sorted(expected_inputs)}, got {sorted(wrapper.input_contract)}"
        )
    if wrapper.input_contract["observation.state"]["shape"] != (7,):
        raise SystemExit("checkpoint observation.state must have shape (7,)")
    if wrapper.output_contract.get("action", {}).get("shape") != (7,):
        raise SystemExit("checkpoint action must have shape (7,)")
    if wrapper.task_required and not task:
        raise SystemExit("POLICY_TASK is required by the checkpoint tokenizer")

    try:
        warmup_state = np.asarray(json.loads(os.environ.get(
            "POLICY_WARMUP_STATE",
            "[3.106,-1.817,1.653,-1.618,-1.628,-3.195,0.0]",
        )), dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"POLICY_WARMUP_STATE must be a JSON float array: {exc}") from exc
    if warmup_state.shape != (7,) or not np.all(np.isfinite(warmup_state)):
        raise SystemExit("POLICY_WARMUP_STATE must contain 7 finite values")
    _warmup(wrapper, task, warmup_state)

    sampling_method, sampling_steps = _sampling_contract(wrapper.config)
    n_action_steps = int(getattr(wrapper.config, "n_action_steps", 1))
    resize = wrapper.external_image_size or (0, 0)
    max_message_bytes = int(os.environ.get("GRPC_MAX_MESSAGE_BYTES", DEFAULT_MAX_MESSAGE_BYTES))
    max_jpeg_bytes = int(os.environ.get("MAX_JPEG_BYTES", DEFAULT_MAX_JPEG_BYTES))

    class TaskEngine:
        device = wrapper.device
        policy = wrapper.policy

        @property
        def last_act_metadata(self):
            return wrapper.last_act_metadata

        def reset(self):
            wrapper.reset()

        def act(self, state, cam1, cam2):
            try:
                return wrapper.act(state, cam1, cam2, task)
            except PolicyInputError as exc:
                # Decode/shape/task checks happen before select_action, so the
                # same request_id can safely be retried with corrected input.
                raise RequestValidationError(str(exc)) from exc

    engine = TaskEngine()
    service = RemoteDiffusionService(
        engine,
        model_id=os.environ.get("MODEL_ID", str(wrapper.config.type)),
        checkpoint_revision=os.environ.get("CHECKPOINT_REVISION", "unknown"),
        scheduler=sampling_method,
        num_inference_steps=sampling_steps,
        n_action_steps=n_action_steps,
        resize_height=resize[0],
        resize_width=resize[1],
        max_jpeg_bytes=max_jpeg_bytes,
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="grpc-policy"),
        options=(("grpc.max_receive_message_length", max_message_bytes),
                 ("grpc.max_send_message_length", max_message_bytes)),
    )
    pb_grpc.add_RemoteDiffusionServicer_to_server(service, server)
    if server.add_insecure_port(bind_address) == 0:
        raise SystemExit(f"failed to bind {bind_address}")
    server.start()
    print(json.dumps(wrapper.describe(), indent=2, default=list), flush=True)
    print(f"[remote_lerobot] ready at {bind_address}; sampling={sampling_method}/{sampling_steps}", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2.0).wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
