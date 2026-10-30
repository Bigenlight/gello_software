#!/usr/bin/env python3
"""Require consecutive, valid HIL deadman heartbeats in a demanded state.

This is a LIVENESS PROOF of the deadman channel, not a request that the
operator hold a button: exit 0 only after ``--samples`` consecutive freshly
received messages are well formed (``[engaged in {0,1}, gain in 0.10..1.00]``)
AND carry the state named by ``--require``.

``--require engaged``     the historical mode (the human is driving).
``--require disengaged``  the default for session startup: the GUI is alive and
                          publishing, and the arm is NOT in a state where
                          grabbing GELLO moves it.  A HIL session begins under
                          policy control, so this is the strictly safer proof.

Exit 1 for a malformed heartbeat or the wrong state, 2 when ROS or the topic is
unavailable (including "not enough heartbeats before the timeout").
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Iterable


def validate_sample(data: Iterable[float]) -> tuple[bool, float]:
    values = tuple(float(value) for value in data)
    if len(values) < 2:
        raise ValueError("deadman message must contain [engaged, gain]")
    engaged, gain = values[:2]
    if not math.isfinite(engaged) or not math.isfinite(gain):
        raise ValueError("deadman engaged/gain must be finite")
    if engaged not in (0.0, 1.0):
        raise ValueError("deadman engaged must be exactly 0.0 or 1.0")
    if not 0.10 <= gain <= 1.00:
        raise ValueError("deadman gain must be within [0.10, 1.00]")
    return engaged == 1.0, gain


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/hil/deadman")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--require",
        choices=("engaged", "disengaged"),
        default="engaged",
        help=(
            "state every accepted heartbeat must carry "
            "(default: engaged, the historical behaviour)"
        ),
    )
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    return args


def _wait_for_state(
    topic: str,
    sample_count: int,
    timeout_s: float,
    want_engaged: bool = True,
):
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray
    except ImportError as exc:
        print(f"ERROR: ROS Python overlay unavailable: {exc}", file=sys.stderr)
        return 2, None

    wanted_label = "ENGAGED" if want_engaged else "DISENGAGED"

    class Check(Node):
        def __init__(self):
            super().__init__("hil_deadman_arm_check")
            self.received = 0
            self.gain = None
            self.error = None
            self.create_subscription(Float32MultiArray, topic, self._callback, 10)

        def _callback(self, message):
            if self.error is not None or self.received >= sample_count:
                return
            try:
                engaged, gain = validate_sample(message.data)
            except ValueError as exc:
                self.error = str(exc)
                return
            if engaged != want_engaged:
                if want_engaged:
                    self.error = (
                        "deadman is DISENGAGED; ENGAGE in the HIL GUI first"
                    )
                else:
                    self.error = (
                        "deadman is ENGAGED; DISENGAGE in the HIL GUI first "
                        "(the session starts under policy control; ENGAGE is "
                        "only for intervention)"
                    )
                return
            self.gain = gain
            self.received += 1

    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = Check()
    deadline = time.monotonic() + timeout_s
    try:
        while (
            rclpy.ok()
            and node.error is None
            and node.received < sample_count
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.error is not None:
            print(f"ERROR: {node.error}", file=sys.stderr)
            return 1, None
        if node.received < sample_count:
            print(
                f"ERROR: received {node.received}/{sample_count} valid "
                f"{wanted_label} heartbeats from {topic} within "
                f"{timeout_s:.1f}s",
                file=sys.stderr,
            )
            return 2, None
        return 0, node.gain
    finally:
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    want_engaged = args.require == "engaged"
    result, gain = _wait_for_state(
        args.topic, args.samples, args.timeout, want_engaged
    )
    if result == 0:
        wanted_label = "ENGAGED" if want_engaged else "DISENGAGED"
        print(
            f"deadman {wanted_label} (channel alive): {args.samples} "
            f"consecutive heartbeats, gain={gain:.2f}"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
