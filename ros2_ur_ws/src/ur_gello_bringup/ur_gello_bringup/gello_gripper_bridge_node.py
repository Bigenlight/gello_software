#!/usr/bin/env python3
"""GELLO gripper -> Robotiq 2F-85 command_percent streaming bridge.

This node reads the GELLO leader's normalized gripper width and republishes it
as a Robotiq command percent for the Modbus gripper node. GELLO stays PASSIVE:
this node only reads a Float32 topic and publishes a Float32 topic — it never
touches any Dynamixel or robot hardware directly.

Direction invariant (SAFETY-CRITICAL)
-------------------------------------
The GELLO width topic ``/gripper/gripper_client/target_gripper_width_percent``
publishes ``js[6]`` in [0.0, 1.0] where **0.0 = OPEN, 1.0 = CLOSED** (see the
gello DynamixelRobot normalization). The Modbus node's ``~/command_percent`` is
defined with the SAME convention: **0.0 = OPEN, 1.0 = CLOSED** (raw Robotiq
POS = round(percent*255), 0=open/255=closed). Therefore the mapping is a DIRECT
IDENTITY: ``command_percent = clamp(width, 0, 1)`` — NO inversion. This preserves
direction end-to-end so closing your hand on GELLO CLOSES the robot gripper.

The ``invert`` parameter (default **False**) exists only for a future mechanical
recalibration flip; it MUST stay False for this hardware. Inverting it would
command robot-CLOSE when the GELLO hand OPENS — a crush hazard. The startup log
prints the resolved direction so it is verifiable before touching the robot.

Pipeline
--------
On each received width: apply the (identity) map, optional invert, clamp to
[clamp_min, clamp_max], optional EMA smoothing (ema_alpha=1.0 => off), then a
deadband gate that suppresses tiny at-rest changes before publishing. The
authoritative rate-limit toward the single-client Modbus bus is performed by
the gripper node itself, so this bridge just republishes on receive at the
~30 Hz GELLO rate — no timer needed.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class GelloGripperBridge(Node):
    """Map the GELLO gripper width to a Robotiq command_percent (0=open..1=closed)."""

    def __init__(self) -> None:
        super().__init__("gello_gripper_bridge")

        # --- Parameters --------------------------------------------------
        # invert MUST stay False for this robot (see module docstring / SAFETY).
        self.invert = bool(self.declare_parameter("invert", False).value)
        # Deadband (in percent units) suppressing at-rest Dynamixel jitter on the
        # gripper axis; a change smaller than this is not republished.
        self.deadband = float(self.declare_parameter("deadband", 0.02).value)
        # EMA smoothing factor: 1.0 == OFF (no lag; safest so an emergency hand
        # open reaches the gripper immediately). <1 smooths but adds lag.
        self.ema_alpha = float(self.declare_parameter("ema_alpha", 1.0).value)
        self.clamp_min = float(self.declare_parameter("clamp_min", 0.0).value)
        self.clamp_max = float(self.declare_parameter("clamp_max", 1.0).value)
        self.publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 30.0).value
        )
        self._in_topic = str(
            self.declare_parameter(
                "input_topic",
                "/gripper/gripper_client/target_gripper_width_percent",
            ).value
        )
        # Absolute topic so it lands on /robotiq_gripper/command_percent
        # regardless of this node's namespace.
        self._out_topic = str(
            self.declare_parameter(
                "output_topic", "/robotiq_gripper/command_percent"
            ).value
        )

        # --- State -------------------------------------------------------
        self._f: float | None = None  # EMA-filtered value
        self._last_pub: float | None = None  # last published value (deadband ref)

        # --- ROS interfaces ----------------------------------------------
        self._pub = self.create_publisher(Float32, self._out_topic, 10)
        self._sub = self.create_subscription(
            Float32, self._in_topic, self._on_width, 10
        )

        # --- Startup log (state the direction invariant explicitly) ------
        direction = (
            "width 1=OPEN..0=CLOSED (INVERTED!)"
            if self.invert
            else "width 0=OPEN..1=CLOSED -> command_percent 0=open..1=closed"
        )
        self.get_logger().info(
            "gello_gripper_bridge started | "
            f"{direction} (invert={self.invert}) | "
            f"deadband={self.deadband} ema_alpha={self.ema_alpha} "
            f"clamp=[{self.clamp_min},{self.clamp_max}]"
        )
        if self.invert:
            self.get_logger().warn(
                "invert=True: GELLO-open will command robot-CLOSE — CRUSH HAZARD. "
                "This MUST be False for this hardware."
            )
        self.get_logger().info(
            f"subscribing {self._in_topic} -> publishing {self._out_topic}"
        )

    # ---------------------------------------------------------------------
    def _on_width(self, msg: Float32) -> None:
        """Map a GELLO width sample to command_percent and publish (deadbanded)."""
        p = float(msg.data)
        if self.invert:
            p = 1.0 - p
        # Clamp to [clamp_min, clamp_max].
        p = min(max(p, self.clamp_min), self.clamp_max)
        # EMA (ema_alpha=1.0 => identity, no lag).
        if self._f is None:
            self._f = p
        else:
            self._f = (1.0 - self.ema_alpha) * self._f + self.ema_alpha * p
        # DEADBAND gate: skip tiny at-rest changes.
        if (
            self._last_pub is not None
            and abs(self._f - self._last_pub) < self.deadband
        ):
            return
        out = Float32()
        out.data = float(self._f)
        self._pub.publish(out)
        self._last_pub = self._f


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GelloGripperBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
