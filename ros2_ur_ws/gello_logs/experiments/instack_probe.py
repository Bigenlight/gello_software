#!/usr/bin/env python3
"""instack-real-bridge probe.

Publishes synthetic /joint_states (ARRIVED pose q0, 100 Hz, constant) and
/gello/joint_states (30 Hz: hold q0+tremor for 2.0s, then STEP one joint by
+G=0.2 rad) to feed the REAL gello_ur_bridge node under test. Subscribes to
/forward_position_controller/commands and timestamps every message to a CSV
so we can measure the actual per-cycle slew speed and gap-closing time.

No robot, no GELLO hardware. ROS_DOMAIN_ID should be set to 42 (isolation) by
the caller before running this script.
"""

import csv
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

UR_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ARRIVED pose q0 == the frozen move-to-start target (arbitrary but plausible
# UR "home-ish" pose; matches the init_pose used in the real config, minus the
# pan offset, just needs to be a valid interior joint configuration).
Q0 = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

# Which joint gets the drift-gap step, and its magnitude (rad).
STEP_JOINT_IDX = 1  # shoulder_lift_joint
GAP_G = 0.2

TREMOR_STD = 0.01  # rad, small GELLO tremor while holding pre-step
STEP_TIME_S = 2.0  # seconds after start before the step is applied
TOTAL_RUN_S = 6.0

OUT_CSV = "/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/instack_commands.csv"


class InstackProbe(Node):
    def __init__(self) -> None:
        super().__init__("instack_probe")

        self._actual_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._gello_pub = self.create_publisher(JointState, "/gello/joint_states", 10)
        self._cmd_sub = self.create_subscription(
            Float64MultiArray,
            "/forward_position_controller/commands",
            self._on_cmd,
            50,
        )

        self._t0 = time.monotonic()
        self._rng_state = 12345  # trivial LCG for reproducible tremor, no numpy dep

        self._actual_timer = self.create_timer(1.0 / 100.0, self._pub_actual)
        self._gello_timer = self.create_timer(1.0 / 30.0, self._pub_gello)

        self._csv_f = open(OUT_CSV, "w", newline="")
        self._csv_w = csv.writer(self._csv_f)
        self._csv_w.writerow(
            ["t_rel_s", "wall_recv_time_s"]
            + [f"cmd_{j}" for j in UR_JOINT_ORDER]
        )
        self._n_cmds = 0

        self.get_logger().info(
            f"instack_probe started: Q0={Q0} step_joint={UR_JOINT_ORDER[STEP_JOINT_IDX]} "
            f"gap={GAP_G} at t={STEP_TIME_S}s"
        )

    def _tremor(self) -> float:
        # Simple deterministic pseudo-random in [-1,1], scaled by TREMOR_STD.
        self._rng_state = (1103515245 * self._rng_state + 12345) & 0x7FFFFFFF
        u = (self._rng_state / 0x7FFFFFFF) * 2.0 - 1.0
        return u * TREMOR_STD

    def _pub_actual(self) -> None:
        # ARRIVED pose: constant, no tremor (this is the robot's real encoder
        # feedback in the actual deployment -- rock solid at the handover pose).
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(UR_JOINT_ORDER)
        msg.position = list(Q0)
        self._actual_pub.publish(msg)

    def _pub_gello(self) -> None:
        t = time.monotonic() - self._t0
        q = list(Q0)
        if t < STEP_TIME_S:
            # Hold q0 + small tremor on all joints (drift/tremor emulation).
            q = [qi + self._tremor() for qi in q]
        else:
            # STEP: one joint jumps by +GAP_G (models the drift gap revealed at
            # handover), still with small ongoing tremor.
            q = [qi + self._tremor() for qi in q]
            q[STEP_JOINT_IDX] = Q0[STEP_JOINT_IDX] + GAP_G + self._tremor()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(UR_JOINT_ORDER)
        msg.position = q
        self._gello_pub.publish(msg)

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        t = time.monotonic() - self._t0
        row = [f"{t:.6f}", f"{time.monotonic():.6f}"] + [f"{v:.8f}" for v in msg.data]
        self._csv_w.writerow(row)
        self._csv_f.flush()
        self._n_cmds += 1


def main() -> None:
    rclpy.init()
    node = InstackProbe()
    start = time.monotonic()
    try:
        while rclpy.ok() and (time.monotonic() - start) < TOTAL_RUN_S:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.get_logger().info(f"Done: received {node._n_cmds} commands, wrote {OUT_CSV}")
        node._csv_f.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
