#!/usr/bin/env python3
"""Exercise the laptop's production RemoteDiffusionWorker without starting ROS."""

import json
import math
import os
import statistics
import time

import cv2
import numpy as np

from gello_policy.remote_policy_client import (
    ImageSnapshot,
    ObservationSnapshot,
    create_worker,
)


START_STATE = (3.106, -1.817, 1.653, -1.618, -1.628, -3.195, 0.0)


def positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


def boolean_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0")
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


def percentile(values: list[float], percent: float) -> float:
    """Return a linearly interpolated percentile (NumPy's default convention)."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def latency_summary(records: list[dict]) -> dict:
    fields = (
        "round_trip_ms",
        "preprocess_ms",
        "inference_ms",
        "total_server_ms",
        "observation_age_ms",
    )

    def summarize(samples: list[dict]) -> dict:
        result = {"count": len(samples)}
        for field in fields:
            values = [float(sample[field]) for sample in samples]
            result[field] = {
                "mean": statistics.fmean(values) if values else None,
                "p50": percentile(values, 50) if values else None,
                "p95": percentile(values, 95) if values else None,
                "p99": percentile(values, 99) if values else None,
            }
        return result

    refill = [record for record in records if record["chunk_refill"]]
    non_refill = [record for record in records if not record["chunk_refill"]]
    return {
        "all": summarize(records),
        "refill": summarize(refill),
        "non_refill": summarize(non_refill),
    }


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
    request_count = positive_int_env("ROUNDTRIP_REQUESTS", 1)
    warmup_count = nonnegative_int_env("ROUNDTRIP_WARMUP_REQUESTS", 0)
    include_requests = boolean_env("ROUNDTRIP_INCLUDE_REQUESTS", True)

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

        def send_one(sequence: int, total: int, phase: str) -> dict:
            stamp_ns = time.monotonic_ns()
            cam1 = ImageSnapshot(stamp_ns, width, height, jpeg)
            cam2 = ImageSnapshot(stamp_ns, width, height, jpeg)
            request_started = time.perf_counter()
            worker.submit(ObservationSnapshot(stamp_ns, state, cam1, cam2))

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
                raise TimeoutError(
                    f"no action received for {phase} request {sequence}/{total} "
                    f"within {timeout_s:g}s"
                )
            action = [float(value) for value in result.action]
            if len(action) != 7 or not all(math.isfinite(value) for value in action):
                raise RuntimeError(
                    f"{phase} request {sequence}/{total} returned an invalid action"
                )
            return {
                "action": action,
                "chunk_refill": result.chunk_refill,
                "inference_ms": result.inference_ms,
                "observation_age_ms": result.observation_age_s * 1000.0,
                "preprocess_ms": result.preprocess_ms,
                "remaining_chunk_actions": result.remaining_chunk_actions,
                "request_id": result.request_id,
                "round_trip_ms": (time.perf_counter() - request_started) * 1000.0,
                "sequence": sequence,
                "total_server_ms": result.total_server_ms,
            }

        # Warm-up requests exercise the exact same transport, policy queue, and
        # action validation path, but are intentionally omitted from records and
        # latency statistics.
        for sequence in range(1, warmup_count + 1):
            send_one(sequence, warmup_count, "warm-up")

        records = [
            send_one(sequence, request_count, "measured")
            for sequence in range(1, request_count + 1)
        ]

        contract = {
            "scheduler": info.scheduler,
            "num_inference_steps": info.num_inference_steps,
            "n_action_steps": info.n_action_steps,
            "resize": [info.resize_height, info.resize_width],
        }
        if request_count == 1 and warmup_count == 0 and include_requests:
            # Preserve the original one-line/default result contract.  The two
            # chunk fields expose useful server metadata without changing the
            # existing keys or their meaning.
            record = records[0]
            print(
                json.dumps(
                    {
                        "action": record["action"],
                        "checkpoint_revision": info.checkpoint_revision,
                        "chunk_refill": record["chunk_refill"],
                        "inference_ms": record["inference_ms"],
                        "model_id": info.model_id,
                        "policy_contract": contract,
                        "observation_age_ms": record["observation_age_ms"],
                        "preprocess_ms": record["preprocess_ms"],
                        "remaining_chunk_actions": record["remaining_chunk_actions"],
                        "request_id": record["request_id"],
                        # Retain legacy behavior: default one-shot timing includes
                        # contract discovery and ResetEpisode.
                        "round_trip_ms": (time.perf_counter() - started) * 1000.0,
                        "session_id": session_id,
                        "total_server_ms": record["total_server_ms"],
                    },
                    sort_keys=True,
                )
            )
        else:
            output = {
                "checkpoint_revision": info.checkpoint_revision,
                "model_id": info.model_id,
                "policy_contract": contract,
                "request_count": request_count,
                "session_id": session_id,
                "summary": latency_summary(records),
                "warmup_request_count": warmup_count,
            }
            if include_requests:
                output["requests"] = records
            print(
                json.dumps(
                    output,
                    sort_keys=True,
                )
            )
        return 0
    finally:
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
