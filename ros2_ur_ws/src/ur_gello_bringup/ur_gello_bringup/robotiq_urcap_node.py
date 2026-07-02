#!/usr/bin/env python3
"""ROS2 node that drives a real Robotiq 2F-85 gripper on a UR via the URCap socket.

Reuses the repo's existing ``gello.robots.robotiq_gripper.RobotiqGripper`` driver,
which talks to the URCap TCP server on the UR controller (default port 63352).

IMPORTANT (build PC vs robot PC):
    On THIS build/test PC there is NO real gripper attached. The ``connect_on_start``
    parameter therefore defaults to ``False``: the node starts, subscribes, and simply
    LOGS the gripper position it *would* command, without ever opening a socket. The
    real gripper runs on the robot PC, where you launch this node with
    ``connect_on_start:=true`` (and the correct ``robot_ip``).

Position convention:
    URCap POS is 0..255 where 0 = fully OPEN and 255 = fully CLOSED. The teleop
    topic sends a width *percent* in [0, 1] where 0 = open and 1 = closed, so the
    mapping is a direct ``pos = round(percent * 255)``.
"""

import os
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# Make the ``gello`` package importable even if it is not already on sys.path.
# gello lives at the repo root: <repo>/gello/... . Candidates, in order:
# $GELLO_REPO_ROOT, four dirs up from this file (valid in the source tree /
# with --symlink-install), and the known checkout path on this machine.
_REPO_ROOT = ""
for _root in (
    os.environ.get("GELLO_REPO_ROOT", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
    "/home/theo_lab/gello_software",
):
    if _root and os.path.isdir(os.path.join(_root, "gello")):
        _REPO_ROOT = _root
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break


class RobotiqUrcapNode(Node):
    """Subscribes to the target gripper width and drives a Robotiq 2F-85 over URCap."""

    def __init__(self):
        super().__init__("robotiq_urcap")

        # --- Declare parameters ---
        self.declare_parameter("robot_ip", "192.168.56.101")
        self.declare_parameter("gripper_port", 63352)
        self.declare_parameter("speed", 255)
        self.declare_parameter("force", 10)
        self.declare_parameter("connect_on_start", False)

        # --- Read parameters ---
        self._robot_ip = (
            self.get_parameter("robot_ip").get_parameter_value().string_value
        )
        self._gripper_port = (
            self.get_parameter("gripper_port").get_parameter_value().integer_value
        )
        self._speed = self.get_parameter("speed").get_parameter_value().integer_value
        self._force = self.get_parameter("force").get_parameter_value().integer_value
        connect_on_start = (
            self.get_parameter("connect_on_start").get_parameter_value().bool_value
        )

        self.gripper = None
        self.connected = False

        if connect_on_start:
            self._connect()
        else:
            self.get_logger().warn(
                "connect_on_start=False: NOT connecting to a physical gripper. "
                "Commands will only be logged (build/test PC has no gripper). "
                "Launch with connect_on_start:=true on the robot PC."
            )

        # --- Subscriber ---
        self._sub = self.create_subscription(
            Float32,
            "/gripper/gripper_client/target_gripper_width_percent",
            self._on_target,
            10,
        )

        self.get_logger().info(
            f"robotiq_urcap started (robot_ip={self._robot_ip}, "
            f"port={self._gripper_port}, speed={self._speed}, force={self._force}, "
            f"connected={self.connected})."
        )

    def _connect(self):
        """Open and activate the URCap gripper socket. Import the driver lazily."""
        try:
            from gello.robots.robotiq_gripper import RobotiqGripper
        except Exception as exc:  # noqa: BLE001 - surface import problems clearly
            self.get_logger().error(
                f"Could not import RobotiqGripper from the gello package "
                f"(is it on PYTHONPATH? repo root tried: '{_REPO_ROOT}'). "
                f"Node will run without a gripper. Error: {exc}"
            )
            self.connected = False
            return

        try:
            gripper = RobotiqGripper()
            gripper.connect(self._robot_ip, port=self._gripper_port)
            gripper.activate()
        except Exception as exc:  # noqa: BLE001 - never crash on a bad connection
            self.get_logger().error(
                f"Failed to connect/activate Robotiq gripper at "
                f"{self._robot_ip}:{self._gripper_port}. Node will keep spinning and "
                f"warn on each command. Error: {exc}"
            )
            self.gripper = None
            self.connected = False
            return

        self.gripper = gripper
        self.connected = True
        self.get_logger().info(
            f"Connected and activated Robotiq gripper at "
            f"{self._robot_ip}:{self._gripper_port}."
        )

    def _on_target(self, msg: Float32):
        # percent in [0, 1]: 0 = open, 1 = closed. URCap POS: 0 = open, 255 = closed.
        percent = float(msg.data)
        pos = int(round(percent * 255))
        pos = max(0, min(255, pos))

        if self.connected and self.gripper is not None:
            try:
                self.gripper.move(pos, self._speed, self._force)
            except Exception as exc:  # noqa: BLE001 - never crash on a write glitch
                self.get_logger().warn(
                    f"gripper.move({pos}) failed: {exc}",
                    throttle_duration_sec=2.0,
                )
        else:
            self.get_logger().info(
                f"[no gripper] would move to POS={pos} "
                f"(percent={percent:.3f}, speed={self._speed}, force={self._force}).",
                throttle_duration_sec=1.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RobotiqUrcapNode()
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
