#!/usr/bin/env python3
"""ROS-free REQ client for the real IFQL server smoke test."""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import time
from pathlib import Path

import numpy as np
import zmq


def load_obs_assembler(repo_root: Path):
    path = repo_root / "ros2_ur_ws/src/gello_policy/gello_policy/obs_assembler.py"
    spec = importlib.util.spec_from_file_location("ifql_smoke_obs_assembler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5695)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    args = parser.parse_args()

    if args.host != "127.0.0.1" or args.port != 5695:
        parser.error("smoke is restricted to 127.0.0.1:5695")
    if args.count < 1:
        parser.error("--count must be positive")

    obs = load_obs_assembler(args.repo_root)
    states = np.load(args.frames / "states.npy").astype(np.float32)
    frame_dir = args.frames / "frames"
    frames = []
    for index in range(min(args.count, len(states))):
        cam1 = frame_dir / f"{index:05d}_cam1.jpg"
        cam2 = frame_dir / f"{index:05d}_cam2.jpg"
        if not cam1.is_file() or not cam2.is_file():
            raise RuntimeError(f"incomplete extracted frame {index}")
        frames.append((states[index].tolist(), cam1.read_bytes(), cam2.read_bytes()))
    if len(frames) != args.count:
        raise RuntimeError(f"need {args.count} complete frames, found {len(frames)}")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(args.timeout_s * 1000))
    socket.setsockopt(zmq.SNDTIMEO, int(args.timeout_s * 1000))
    socket.connect(f"tcp://{args.host}:{args.port}")

    try:
        reset_start = time.perf_counter()
        socket.send_multipart(obs.build_reset_request())
        reset = obs.parse_reply(socket.recv_multipart())
        reset_ms = (time.perf_counter() - reset_start) * 1000.0
        if reset.get(obs.KEY_OK) is not True:
            raise RuntimeError(f"reset failed: {reset}")
        print(f"[smoke-client] reset_ok={reset_ms:.2f} ms metadata={reset}", flush=True)

        latencies = []
        actions = []
        for index, (state, cam1, cam2) in enumerate(frames):
            start = time.perf_counter()
            socket.send_multipart(obs.build_act_request(state, cam1, cam2))
            reply = obs.parse_action_reply(socket.recv_multipart())
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not all(math.isfinite(value) for value in reply):
                raise RuntimeError(f"non-finite action at frame {index}: {reply}")
            latencies.append(elapsed_ms)
            actions.append(reply)
        print(
            "[smoke-client] acts={} latency_ms(min/p50/p95/max/mean)="
            "{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f} action_shape={} first_action={}".format(
                len(latencies), min(latencies), statistics.median(latencies),
                percentile(latencies, 0.95), max(latencies), statistics.mean(latencies),
                len(actions[0]), [round(value, 5) for value in actions[0]]),
            flush=True,
        )
        refill_indices = list(range(0, len(latencies), 24))
        refill_latencies = [latencies[index] for index in refill_indices]
        queue_hits = [value for index, value in enumerate(latencies) if index not in refill_indices]
        print(
            "[smoke-client] refill_indices={} refill_ms(min/p50/max)={:.2f}/{:.2f}/{:.2f} "
            "queue_hit_p50_ms={:.2f}".format(
                refill_indices, min(refill_latencies), statistics.median(refill_latencies),
                max(refill_latencies), statistics.median(queue_hits) if queue_hits else float("nan")),
            flush=True,
        )
    finally:
        socket.close(0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
