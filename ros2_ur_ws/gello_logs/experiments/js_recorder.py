#!/usr/bin/env python3
"""Record /joint_states to CSV for a fixed duration, then compute peak
per-joint speed (rad/s). Used to verify the post-switch streaming window has
no snap. System python + ROS Humble only."""
import argparse
import csv
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

UR = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


class Rec(Node):
    def __init__(self, duration, out):
        super().__init__("js_recorder")
        self.duration = duration
        self.out = out
        self.rows = []          # (t, q0..q5)
        self.t0 = None
        self.sub = self.create_subscription(
            JointState, "/joint_states", self.cb, 50)

    def cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(j in idx for j in UR):
            return
        t = time.monotonic()
        if self.t0 is None:
            self.t0 = t
        q = [float(msg.position[idx[j]]) for j in UR]
        self.rows.append([t - self.t0] + q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rclpy.init()
    node = Rec(a.duration, a.out)
    start = time.monotonic()
    while rclpy.ok() and (time.monotonic() - start) < a.duration:
        rclpy.spin_once(node, timeout_sec=0.05)
    # write CSV
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + UR)
        w.writerows(node.rows)
    # analyze: peak per-joint speed via consecutive samples
    peak = [0.0] * 6
    peak_overall = 0.0
    rows = node.rows
    for k in range(1, len(rows)):
        dt = rows[k][0] - rows[k - 1][0]
        if dt <= 1e-6:
            continue
        for j in range(6):
            v = abs(rows[k][1 + j] - rows[k - 1][1 + j]) / dt
            if v > peak[j]:
                peak[j] = v
    peak_overall = max(peak) if peak else 0.0
    print(f"RECORDED {len(rows)} samples over "
          f"{rows[-1][0] if rows else 0:.2f}s -> {a.out}")
    for j in range(6):
        print(f"  peak_speed[{UR[j]}] = {peak[j]:.4f} rad/s")
    print(f"PEAK_PER_JOINT_SPEED_RAD_S={peak_overall:.4f}")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
