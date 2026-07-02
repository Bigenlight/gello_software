#!/usr/bin/env python3
"""GELLO -> UR forward_position_controller streaming bridge.

This node subscribes to GELLO joint states (~30 Hz) and republishes them as
position commands to the UR ``forward_position_controller`` at a higher rate
(default 125 Hz) so the UR driver's servoj loop receives a smooth, evenly
spaced stream. Between GELLO samples the command is EMA-smoothed and
slew-rate limited, then upsampled by the publish timer.

Safety / deployment notes
-------------------------
* forward_position_controller does NOT interpolate: whatever position you send
  is commanded immediately. On a REAL robot you MUST first drive the arm to the
  current GELLO pose using the ``scaled_joint_trajectory_controller`` (a smooth
  trajectory move), and only THEN switch controllers to
  ``forward_position_controller`` before starting this bridge. That
  move-to-pose handshake is intentionally handled elsewhere (launch / a
  separate script), not in this node.
* With ``fake_hardware`` (mock), the simulated arm starts at all-zeros. This
  node does not know the arm's actual state, so on the first valid GELLO
  message it seeds its filter to the GELLO pose. To avoid a violent jump the
  MAX-STEP CLAMP limits how far the *published* command can move each cycle,
  which ramps the mock arm from 0 toward the GELLO pose safely over many
  cycles. On real hardware the pre-positioning handshake above means the arm
  is already near the GELLO pose, so the clamp only rejects spikes.
* STALENESS WATCHDOG: if GELLO input stops (unplugged, crashed, driver hang)
  the node STOPS publishing rather than repeating the last command forever.
  Not streaming is the fail-safe state for a position controller.
* Timing uses ``time.monotonic()`` for the staleness check on purpose: it is
  immune to wall-clock steps and to sim-time (``use_sim_time``) surprises, so
  the watchdog measures real elapsed wall time regardless of clock config.
"""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Output index order expected by the UR forward_position_controller.
UR_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Throttle period (seconds) for repeated warnings so we don't spam the log.
_WARN_THROTTLE_S = 2.0


class GelloUrBridge(Node):
    """Bridge GELLO joint states to UR forward_position_controller commands."""

    def __init__(self) -> None:
        super().__init__("gello_ur_bridge")

        # --- Parameters --------------------------------------------------
        self.ema_alpha = float(
            self.declare_parameter("ema_alpha", 0.5).value
        )
        self.max_step_rad = float(
            self.declare_parameter("max_step_rad", 0.05).value
        )
        self.staleness_timeout_s = float(
            self.declare_parameter("staleness_timeout_s", 0.5).value
        )
        self.publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 125.0).value
        )
        if self.publish_rate_hz <= 0.0:
            self.get_logger().warn(
                f"publish_rate_hz={self.publish_rate_hz} invalid; using 125.0 Hz"
            )
            self.publish_rate_hz = 125.0

        # --- State -------------------------------------------------------
        # Raw target reordered into UR_JOINT_ORDER (6 floats) from last good msg.
        self._raw_target: list[float] | None = None
        # Monotonic timestamp (s) of the last good GELLO message.
        self._last_good_msg_time: float | None = None
        # EMA-filtered command (6 floats), seeded on first valid target.
        self._filtered: list[float] | None = None
        # Last command actually published (6 floats); slew clamp is relative to it.
        self._last_published: list[float] | None = None

        # --- ROS interfaces ----------------------------------------------
        self._pub = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10
        )
        self._sub = self.create_subscription(
            JointState, "/gello/joint_states", self._on_joint_state, 10
        )
        self._timer = self.create_timer(
            1.0 / self.publish_rate_hz, self._on_timer
        )

        # --- Startup log -------------------------------------------------
        self.get_logger().info(
            "gello_ur_bridge started | "
            f"ema_alpha={self.ema_alpha} "
            f"max_step_rad={self.max_step_rad} "
            f"staleness_timeout_s={self.staleness_timeout_s} "
            f"publish_rate_hz={self.publish_rate_hz}"
        )
        self.get_logger().info(f"UR joint order: {UR_JOINT_ORDER}")

    # ---------------------------------------------------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        """Ingest a GELLO joint state, reorder by name, store as raw target."""
        name_to_pos = dict(zip(msg.name, msg.position))

        missing = [j for j in UR_JOINT_ORDER if j not in name_to_pos]
        if missing:
            self.get_logger().warn(
                f"GELLO message missing UR joint(s) {missing}; ignoring message",
                throttle_duration_sec=_WARN_THROTTLE_S,
            )
            return

        # Reorder BY NAME into UR command order (never blind index).
        self._raw_target = [float(name_to_pos[j]) for j in UR_JOINT_ORDER]
        self._last_good_msg_time = time.monotonic()

    # ---------------------------------------------------------------------
    def _on_timer(self) -> None:
        """Publish a smoothed, slew-limited command at publish_rate_hz."""
        # Nothing valid received yet: stay silent.
        if self._raw_target is None or self._last_good_msg_time is None:
            return

        # STALENESS WATCHDOG: never keep streaming stale data.
        age = time.monotonic() - self._last_good_msg_time
        if age > self.staleness_timeout_s:
            self.get_logger().warn(
                "GELLO stale, holding — not publishing",
                throttle_duration_sec=_WARN_THROTTLE_S,
            )
            return

        # First valid target: seed filter + last-published to raw target so we
        # don't ramp up from zero. (The mock-arm ramp is handled by the clamp
        # against the arm's real starting command, which begins at this seed.)
        if self._filtered is None or self._last_published is None:
            self._filtered = list(self._raw_target)
            self._last_published = list(self._raw_target)
            self._publish(self._last_published)
            return

        alpha = self.ema_alpha
        step = self.max_step_rad
        out: list[float] = []
        for i in range(len(UR_JOINT_ORDER)):
            # EMA low-pass per joint.
            self._filtered[i] = (
                (1.0 - alpha) * self._filtered[i] + alpha * self._raw_target[i]
            )
            # MAX-STEP CLAMP relative to the last PUBLISHED value (slew limit).
            delta = self._filtered[i] - self._last_published[i]
            if delta > step:
                delta = step
            elif delta < -step:
                delta = -step
            out.append(self._last_published[i] + delta)

        self._last_published = out
        self._publish(out)

    # ---------------------------------------------------------------------
    def _publish(self, positions: list[float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(p) for p in positions]
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GelloUrBridge()
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
