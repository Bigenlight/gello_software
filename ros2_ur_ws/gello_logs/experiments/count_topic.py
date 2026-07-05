#!/usr/bin/env python3
"""Count messages on /forward_position_controller/commands for N seconds.
Robust replacement for `ros2 topic echo` in tests. Prints: COUNT <n>."""
import sys, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
rclpy.init()
n = Node("counter")
cnt = {"n": 0, "last": None}
def cb(m): cnt["n"] += 1; cnt["last"] = list(m.data)
n.create_subscription(Float64MultiArray, "/forward_position_controller/commands", cb, 10)
t_end = time.monotonic() + secs
while rclpy.ok() and time.monotonic() < t_end:
    rclpy.spin_once(n, timeout_sec=0.1)
print(f"COUNT {cnt['n']}")
if cnt["last"]:
    print("LAST " + " ".join(f"{x:.4f}" for x in cnt["last"]))
rclpy.shutdown()
