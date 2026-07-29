#!/usr/bin/env python3
"""Read one JointState and compare it with HIL RESET_JOINTS safely.

Exit 0 means every joint is within tolerance, 1 means the pose is outside the
tolerance, and 2 means no valid JointState arrived.  Wrist/base joints use the
nearest 2*pi-equivalent target; the elbow intentionally uses its literal value
because its feasible range does not admit an equivalent full turn.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Iterable


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
SHORT_NAMES = ("pan", "lift", "elbow", "w1", "w2", "w3")


def parse_target(value: str) -> tuple[float, ...]:
    target = tuple(float(part) for part in value.split(","))
    if len(target) != len(JOINT_NAMES) or not all(math.isfinite(v) for v in target):
        raise ValueError("target must contain six finite comma-separated radians")
    return target


def branch_safe_deltas(
    current: Iterable[float], target: Iterable[float]
) -> tuple[float, ...]:
    current_values = tuple(float(v) for v in current)
    target_values = tuple(float(v) for v in target)
    if len(current_values) != 6 or len(target_values) != 6:
        raise ValueError("current and target must each contain six joints")
    result = []
    for index, (now, goal) in enumerate(zip(current_values, target_values)):
        raw = now - goal
        result.append(raw if index == 2 else math.remainder(raw, 2.0 * math.pi))
    return tuple(result)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/joint_states")
    parser.add_argument("--target", required=True)
    parser.add_argument("--tolerance", required=True, type=float)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.tolerance) or args.tolerance <= 0.0:
        parser.error("--tolerance must be finite and positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    try:
        args.target = parse_target(args.target)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _read_once(topic: str, timeout_s: float):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        print(f"ERROR: ROS Python overlay unavailable: {exc}", file=sys.stderr)
        return None

    class Once(Node):
        def __init__(self):
            super().__init__("hil_joint_pose_check")
            self.pose = None
            self.create_subscription(JointState, topic, self._callback, 10)

        def _callback(self, message):
            by_name = dict(zip(message.name, message.position))
            if all(name in by_name for name in JOINT_NAMES):
                self.pose = tuple(float(by_name[name]) for name in JOINT_NAMES)

    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = Once()
    deadline = time.monotonic() + timeout_s
    try:
        while rclpy.ok() and node.pose is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.pose
    finally:
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    pose = _read_once(args.topic, args.timeout)
    if pose is None:
        print(f"ERROR: no valid {args.topic} within {args.timeout:.1f}s", file=sys.stderr)
        return 2

    delta = branch_safe_deltas(pose, args.target)
    error = tuple(abs(value) for value in delta)
    worst_index = max(range(6), key=error.__getitem__)
    passed = error[worst_index] <= args.tolerance
    if not args.quiet:
        print("  joint      current      target       safe error")
        for name, now, goal, distance in zip(SHORT_NAMES, pose, args.target, error):
            print(f"  {name:<6} {now:>10.4f}  {goal:>10.4f}  {distance:>10.4f}")
        print(
            f"  => max error {error[worst_index]:.4f} rad ({SHORT_NAMES[worst_index]}), "
            f"tolerance {args.tolerance:.4f}: {'PASS' if passed else 'FAIL'}"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
