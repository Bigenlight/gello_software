#!/usr/bin/env python3
"""ROS2 TEST node that fakes a GELLO leader for robotless RViz testing.

This node has NO hardware dependency. It publishes exactly the same topics a real
``gello_publisher`` would, driving them with a slow, safe sine sweep so you can
verify the whole visualization pipeline (RViz, robot_state_publisher, any bridge)
without a physical GELLO or robot attached.

Use this ONLY for visualizing the pipeline; it does not read any real leader arm.
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

# UR joint names, in order (shared contract with the real gello_publisher).
UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# GELLO/UR calibration start pose (6 arm joints, radians).
START_POSE = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


class FakeGello(Node):
    """Publishes a synthetic GELLO joint state + gripper width for RViz testing."""

    def __init__(self):
        super().__init__("fake_gello")

        # --- Declare + read parameters ---
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("amplitude_rad", 0.4)

        rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value
        self._amplitude = (
            self.get_parameter("amplitude_rad").get_parameter_value().double_value
        )

        if rate_hz <= 0.0:
            self.get_logger().warn(
                f"rate_hz={rate_hz} invalid; falling back to 30.0 Hz."
            )
            rate_hz = 30.0
        self._dt = 1.0 / rate_hz
        self._t = 0.0

        # --- Publishers (same topics as the real gello_publisher) ---
        self._js_pub = self.create_publisher(JointState, "/gello/joint_states", 10)
        self._gripper_pub = self.create_publisher(
            Float32, "/gripper/gripper_client/target_gripper_width_percent", 10
        )

        # --- Timer ---
        self._timer = self.create_timer(self._dt, self._on_timer)

        self.get_logger().info(
            f"fake_gello started (TEST ONLY, no hardware) at {rate_hz:.1f} Hz, "
            f"amplitude={self._amplitude:.3f} rad."
        )

    def _on_timer(self):
        t = self._t

        # Slow sine sweep around the start pose (phase-offset per joint).
        positions = [
            START_POSE[i]
            + self._amplitude * math.sin(2.0 * math.pi * 0.1 * t + i * 0.5)
            for i in range(len(START_POSE))
        ]

        # Gripper oscillates open (0) <-> closed (1).
        gripper_percent = 0.5 + 0.5 * math.sin(2.0 * math.pi * 0.2 * t)

        now = self.get_clock().now().to_msg()

        js_msg = JointState()
        js_msg.header.stamp = now
        js_msg.header.frame_id = "base_link"
        js_msg.name = UR_JOINT_NAMES
        js_msg.position = positions
        self._js_pub.publish(js_msg)

        gripper_msg = Float32()
        gripper_msg.data = float(gripper_percent)
        self._gripper_pub.publish(gripper_msg)

        # Advance simulated time by one tick (never use wall-clock here).
        self._t += self._dt


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FakeGello()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
