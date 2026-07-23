#!/usr/bin/env python3
"""Exercise the laptop's production RemoteDiffusionWorker without starting ROS."""

import json
import os
import time

import cv2
import numpy as np

from gello_policy.remote_policy_client import (
    ImageSnapshot,
    ObservationSnapshot,
    create_worker,
)


START_STATE = (3.106, -1.817, 1.653, -1.618, -1.628, -3.195, 0.0)


def make_black_jpeg(height: int, width: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode synthetic black JPEG")
    return encoded.tobytes()


def main() -> int:
    target = os.environ.get("ROUNDTRIP_TARGET", "127.0.0.1:50051")
    timeout_s = float(os.environ.get("ROUNDTRIP_TIMEOUT_S", "15"))
    if timeout_s <= 0:
        raise ValueError("ROUNDTRIP_TIMEOUT_S must be positive")

    worker = create_worker(
        target,
        client_id="laptop-roundtrip-smoke",
        rpc_deadline_s=timeout_s,
        max_response_age_s=timeout_s,
    )
    started = time.perf_counter()
    try:
        # This smoke test discovers the effective policy contract. The real ROS
        # launch still checks every expected_* field before it can arm the robot.
        info = worker.get_server_info()
        session_id = worker.reset_episode()
        worker.start()

        height = int(info.resize_height or os.environ.get("ROUNDTRIP_IMAGE_HEIGHT", "360"))
        width = int(info.resize_width or os.environ.get("ROUNDTRIP_IMAGE_WIDTH", "640"))
        state = tuple(float(value) for value in json.loads(
            os.environ.get("ROUNDTRIP_STATE", json.dumps(START_STATE))
        ))
        if len(state) != 7 or not all(np.isfinite(state)):
            raise ValueError("ROUNDTRIP_STATE must be a JSON array of 7 finite values")
        jpeg = make_black_jpeg(height, width)
        stamp_ns = time.monotonic_ns()
        cam1 = ImageSnapshot(stamp_ns, width, height, jpeg)
        cam2 = ImageSnapshot(stamp_ns, width, height, jpeg)
        worker.submit(
            ObservationSnapshot(stamp_ns, state, cam1, cam2)
        )

        deadline = time.monotonic() + timeout_s
        result = None
        while time.monotonic() < deadline:
            error = worker.error()
            if error is not None:
                raise RuntimeError(f"remote worker failed: {error}")
            result = worker.take_result(max_age_s=timeout_s)
            if result is not None:
                break
            time.sleep(0.01)
        if result is None:
            raise TimeoutError(f"no action received within {timeout_s:g}s")

        print(
            json.dumps(
                {
                    "action": list(result.action),
                    "checkpoint_revision": info.checkpoint_revision,
                    "inference_ms": result.inference_ms,
                    "model_id": info.model_id,
                    "policy_contract": {
                        "scheduler": info.scheduler,
                        "num_inference_steps": info.num_inference_steps,
                        "n_action_steps": info.n_action_steps,
                        "resize": [info.resize_height, info.resize_width],
                    },
                    "observation_age_ms": result.observation_age_s * 1000.0,
                    "preprocess_ms": result.preprocess_ms,
                    "request_id": result.request_id,
                    "round_trip_ms": (time.perf_counter() - started) * 1000.0,
                    "session_id": session_id,
                    "total_server_ms": result.total_server_ms,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
