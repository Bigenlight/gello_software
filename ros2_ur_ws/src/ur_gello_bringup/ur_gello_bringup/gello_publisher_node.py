#!/usr/bin/env python3
"""ROS2 node that reads the physical GELLO leader arm and publishes its joint state.

The GELLO is a PASSIVE, read-only motion-capture arm. Its Dynamixel motors must
NEVER be powered/torqued. ``DynamixelRobotConfig.make_robot`` initializes torque
OFF; this node only ever reads joints and never enables torque.

Reuses the existing ``gello`` python package (importable in system python3.12).
"""

import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


def _ensure_gello_importable() -> None:
    """Put the gello repo root on sys.path if ``gello`` is not already importable.

    The ``gello`` package is not pip-installed system-wide; it lives at the repo
    root. Candidates, in order: $GELLO_REPO_ROOT, four directories up from this
    file (valid in the source tree / with --symlink-install), and the known
    checkout path on this machine.
    """
    try:
        import gello  # noqa: F401

        return
    except ImportError:
        pass
    candidates = [
        os.environ.get("GELLO_REPO_ROOT", ""),
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        ),
        "/home/theo_lab/gello_software",
    ]
    for root in candidates:
        if root and os.path.isdir(os.path.join(root, "gello")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return


_ensure_gello_importable()

from gello.agents.gello_agent import DynamixelRobotConfig  # noqa: E402

# UR joint names, in order, matching the first 6 values of get_joint_state().
UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class GelloPublisher(Node):
    """Reads the physical GELLO leader and publishes joint state + gripper width."""

    def __init__(self):
        super().__init__("gello_publisher")

        # --- Declare parameters (defaults from configs/rwh_ur.yaml) ---
        self.declare_parameter(
            "port",
            "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO6QK-if00-port0",
        )
        self.declare_parameter("joint_ids", [1, 2, 3, 4, 5, 6])
        self.declare_parameter(
            "joint_offsets", [3.142, 4.712, 1.571, 4.712, 4.712, 3.142]
        )
        self.declare_parameter("joint_signs", [1, 1, -1, 1, 1, 1])
        self.declare_parameter("gripper_config", [7.0, 210.034375, 168.234375])
        self.declare_parameter(
            "start_joints", [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0]
        )
        self.declare_parameter("publish_rate_hz", 30.0)

        # --- Read parameters ---
        port = self.get_parameter("port").get_parameter_value().string_value
        joint_ids = list(
            self.get_parameter("joint_ids").get_parameter_value().integer_array_value
        )
        joint_offsets = list(
            self.get_parameter("joint_offsets")
            .get_parameter_value()
            .double_array_value
        )
        joint_signs = list(
            self.get_parameter("joint_signs").get_parameter_value().integer_array_value
        )
        gripper_config = list(
            self.get_parameter("gripper_config")
            .get_parameter_value()
            .double_array_value
        )
        start_joints = list(
            self.get_parameter("start_joints").get_parameter_value().double_array_value
        )
        publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )

        # gripper_config is (gripper_joint_id:int, open_deg:float, close_deg:float).
        gripper_config_t = (
            int(round(gripper_config[0])),
            gripper_config[1],
            gripper_config[2],
        )

        self.get_logger().info(
            f"Building GELLO config: port={port}, joint_ids={joint_ids}, "
            f"joint_signs={joint_signs}, gripper_config={gripper_config_t}"
        )

        cfg = DynamixelRobotConfig(
            joint_ids=tuple(int(i) for i in joint_ids),
            joint_offsets=tuple(float(o) for o in joint_offsets),
            joint_signs=tuple(int(s) for s in joint_signs),
            gripper_config=gripper_config_t,
        )

        # --- Connect to the physical GELLO (torque stays OFF; read-only) ---
        try:
            self.robot = cfg.make_robot(
                port=port, start_joints=np.array(start_joints)
            )
        except Exception as exc:  # noqa: BLE001 - surface any connect failure clearly
            self.get_logger().fatal(
                f"FATAL: could not connect to GELLO on port '{port}'. "
                f"Check that the arm is plugged in / powered, that the serial port "
                f"is not busy (another process holding it), and that you have "
                f"dialout permission. Underlying error: {exc}"
            )
            raise

        self._expected_len = len(joint_ids) + 1  # 6 arm + 1 gripper = 7

        # --- Publishers ---
        self._js_pub = self.create_publisher(JointState, "/gello/joint_states", 10)
        self._gripper_pub = self.create_publisher(
            Float32, "/gripper/gripper_client/target_gripper_width_percent", 10
        )

        # --- Timer ---
        if publish_rate_hz <= 0.0:
            self.get_logger().warn(
                f"publish_rate_hz={publish_rate_hz} invalid; falling back to 30.0 Hz"
            )
            publish_rate_hz = 30.0
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)

        self.get_logger().info(
            f"gello_publisher started, publishing at {publish_rate_hz:.1f} Hz."
        )

    def _on_timer(self):
        # Read-only: never enables torque.
        try:
            js = self.robot.get_joint_state()
        except Exception as exc:  # noqa: BLE001 - never crash the node on a read glitch
            self.get_logger().warn(
                f"get_joint_state() failed, skipping cycle: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        if js is None or len(js) != self._expected_len:
            self.get_logger().warn(
                f"Unexpected joint state length "
                f"(got {None if js is None else len(js)}, "
                f"expected {self._expected_len}); skipping cycle.",
                throttle_duration_sec=2.0,
            )
            return

        now = self.get_clock().now().to_msg()

        js_msg = JointState()
        js_msg.header.stamp = now
        js_msg.header.frame_id = "base_link"
        js_msg.name = UR_JOINT_NAMES
        js_msg.position = js[:6].tolist()
        self._js_pub.publish(js_msg)

        gripper_msg = Float32()
        gripper_msg.data = float(js[6])
        self._gripper_pub.publish(gripper_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GelloPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # GELLO is passive; nothing to power down. Just clean up ROS resources.
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
