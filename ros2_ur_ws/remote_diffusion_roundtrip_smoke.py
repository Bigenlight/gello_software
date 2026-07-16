#!/usr/bin/env python3
"""Exercise the laptop's production RemoteDiffusionWorker without starting ROS."""

import json
import os
import time

import cv2
import numpy as np

from gello_policy.remote_diffusion_client import (
    ImageSnapshot,
    ObservationSnapshot,
    ServerContract,
    create_worker,
)


EXPECTED_CONTRACT = ServerContract(
    model_id="Bigenlight/diffusion_banana_in_pot_joint",
    checkpoint_revision=(
        "sha256:d4722b60caee5d76d004a37e16b2d7adecc1668f79703259ea7260fc9c723c57"
    ),
    scheduler="DDIM",
    num_inference_steps=10,
    n_action_steps=32,
    resize_height=360,
    resize_width=640,
)
START_STATE = (3.106, -1.817, 1.653, -1.618, -1.628, -3.195, 0.0)


def make_black_jpeg() -> bytes:
    image = np.zeros(
        (EXPECTED_CONTRACT.resize_height, EXPECTED_CONTRACT.resize_width, 3),
        dtype=np.uint8,
    )
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
        info = worker.get_server_info(EXPECTED_CONTRACT)
        session_id = worker.reset_episode()
        worker.start()

        jpeg = make_black_jpeg()
        stamp_ns = time.monotonic_ns()
        cam1 = ImageSnapshot(stamp_ns, 640, 360, jpeg)
        cam2 = ImageSnapshot(stamp_ns, 640, 360, jpeg)
        worker.submit(
            ObservationSnapshot(stamp_ns, START_STATE, cam1, cam2)
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
