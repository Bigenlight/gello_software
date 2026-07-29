#!/usr/bin/env python3
"""Bounded, gracefully-closed ROS topic liveness/rate probe for HIL arming.

Unlike ``ros2 topic hz`` under an external SIGKILL timeout, this process exits
as soon as it has received a small number of *advancing* samples and always
destroys its reader before returning.  The default subscription QoS is the
same RELIABLE/VOLATILE keep-last profile used by ``URRosBackend``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Iterable


MESSAGE_TYPES = ("joint_state", "compressed_image", "float32")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--type", required=True, choices=MESSAGE_TYPES)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--min-rate", type=float, default=0.0)
    parser.add_argument("--max-stamp-age", type=float, default=1.0)
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.depth < 1:
        parser.error("--depth must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    if not math.isfinite(args.min_rate) or args.min_rate < 0.0:
        parser.error("--min-rate must be finite and non-negative")
    if not math.isfinite(args.max_stamp_age) or args.max_stamp_age <= 0.0:
        parser.error("--max-stamp-age must be finite and positive")
    return args


def _stamp_ns(message: object) -> int | None:
    """Return a ROS header timestamp, or ``None`` for headerless messages."""

    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    try:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None


def _rate_hz(arrivals: Iterable[float]) -> float:
    values = tuple(float(value) for value in arrivals)
    if len(values) < 2:
        raise ValueError("at least two arrivals are required")
    span = values[-1] - values[0]
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("arrival timestamps must advance")
    return (len(values) - 1) / span


def _message_class(name: str):
    if name == "joint_state":
        from sensor_msgs.msg import JointState

        return JointState
    if name == "compressed_image":
        from sensor_msgs.msg import CompressedImage

        return CompressedImage
    if name == "float32":
        from std_msgs.msg import Float32

        return Float32
    raise ValueError(f"unsupported message type: {name}")


def _probe(args: argparse.Namespace) -> tuple[int, float | None, int]:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        print(f"ERROR: ROS Python overlay unavailable: {exc}", file=sys.stderr)
        return 2, None, 0

    try:
        message_class = _message_class(args.type)
    except ImportError as exc:
        print(f"ERROR: ROS message type unavailable: {exc}", file=sys.stderr)
        return 2, None, 0

    sample_times: list[float] = []
    last_stamp: int | None = None
    stale_samples = 0

    class Probe(Node):
        def __init__(self):
            super().__init__(f"hil_topic_probe_{os.getpid()}")
            self.subscription = self.create_subscription(
                message_class, args.topic, self._callback, args.depth
            )

        def _callback(self, message):
            nonlocal last_stamp, stale_samples
            if len(sample_times) >= args.samples:
                return
            arrival = time.monotonic()
            stamp = _stamp_ns(message)
            # Header-bearing streams must advance.  This rejects a lone cached
            # TRANSIENT_LOCAL frame and duplicate delivery as proof of liveness.
            if stamp is not None:
                if last_stamp is not None and stamp <= last_stamp:
                    return
                age_s = (self.get_clock().now().nanoseconds - stamp) / 1e9
                if age_s < -0.5 or age_s > args.max_stamp_age:
                    stale_samples += 1
                    return
                last_stamp = stamp
                sample_times.append(stamp / 1e9)
            else:
                sample_times.append(arrival)

    rclpy.init(args=None)
    node = Probe()
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and len(sample_times) < args.samples:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(node, timeout_sec=min(0.05, remaining))

        if len(sample_times) < args.samples:
            print(
                f"ERROR: {args.topic} delivered {len(sample_times)}/{args.samples} "
                f"fresh advancing samples within {args.timeout:.1f}s "
                f"(stale_rejected={stale_samples})",
                file=sys.stderr,
            )
            return 1, None, len(sample_times)

        rate = _rate_hz(sample_times)
        if rate < args.min_rate:
            print(
                f"ERROR: {args.topic} rate {rate:.3f} Hz is below "
                f"{args.min_rate:.3f} Hz",
                file=sys.stderr,
            )
            return 1, rate, len(sample_times)
        return 0, rate, len(sample_times)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    result, rate, samples = _probe(args)
    if result == 0:
        print(f"topic live: {args.topic} rate={rate:.3f} Hz samples={samples}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
