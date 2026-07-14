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

Pause / resume (drop-hazard mitigation)
---------------------------------------
A passive GELLO leader, when the operator lets go, COLLAPSES and its flopping
gripper axis streams straight through this bridge — a crush-or-drop hazard on
the real robot. Two NEW Trigger services gate that:

* ``~/pause``  — UNCONDITIONAL, always succeeds. Sets ``_paused`` and the width
  callback returns BEFORE the pipeline, so NOTHING is republished. The Robotiq
  holds its last commanded position onboard. Not-streaming is the safe state.
* ``~/resume`` — FAIL-CLOSED. Refuses (staying paused + silent) unless a FRESH
  leader sample exists AND an actual gripper position is known. On acceptance it
  seeds the output at the gripper's ACTUAL position (zero jump) and, for
  ``resume_ramp_s`` seconds, slew-limits every published sample toward the live
  leader value (bypassing the deadband so the ramp advances monotonically). Only
  after the ramp window does it revert to the plain deadbanded pass-through.

``~/state`` (String, published at ``state_publish_rate_hz``) reports one of
PAUSED / WAITING / RAMPING / FOLLOWING for the operator UI.
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger


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
        # Actual gripper position feedback (Robotiq modbus node publishes this as
        # Float32 on /robotiq_gripper/position_percent, 0=open..1=closed). Used to
        # SEED resume so the first output equals where the gripper actually is.
        self._actual_topic = str(
            self.declare_parameter(
                "actual_topic", "/robotiq_gripper/position_percent"
            ).value
        )
        # --- Pause/resume params ---
        # A leader sample older than this (s) is STALE — resume refuses on it so a
        # dead/frozen stream can never re-open the drop hazard.
        self.staleness_timeout_s = float(
            self.declare_parameter("staleness_timeout_s", 0.5).value
        )
        # Duration (s) of the slew-limited ramp after a resume.
        self.resume_ramp_s = float(
            self.declare_parameter("resume_ramp_s", 2.0).value
        )
        # Max output change per second (fraction of stroke) during the resume ramp.
        # 0.6 => full 0..1 stroke covered in ~1.67 s, so resume_ramp_s=2.0 always
        # lets a static leader converge inside the window.
        self.resume_slew_per_s = float(
            self.declare_parameter("resume_slew_per_s", 0.6).value
        )
        self.state_publish_rate_hz = float(
            self.declare_parameter("state_publish_rate_hz", 5.0).value
        )
        # Start paused (drop-hazard-safe default is False for standalone runs; the
        # integrated launch may override to true to pre-spawn held).
        self._paused = bool(
            self.declare_parameter("start_paused", False).value
        )

        # --- State -------------------------------------------------------
        self._f: float | None = None  # EMA-filtered value
        self._last_pub: float | None = None  # last published value (deadband ref)
        self._last_pub_time: float = 0.0  # time.monotonic() of last publish
        # Latest leader sample — updated in _on_width ALWAYS (even while paused) so
        # resume can check freshness.
        self._last_rx_time: float | None = None
        self._last_rx_value: float | None = None
        # Actual gripper position (from _actual_topic); None until first feedback.
        self._actual_pos: float | None = None
        # End time (time.monotonic()) of the post-resume slew ramp. 0.0 => no ramp.
        self._ramp_until: float = 0.0

        # --- ROS interfaces ----------------------------------------------
        self._pub = self.create_publisher(Float32, self._out_topic, 10)
        self._sub = self.create_subscription(
            Float32, self._in_topic, self._on_width, 10
        )
        self._actual_sub = self.create_subscription(
            Float32, self._actual_topic, self._on_actual, 10
        )
        self._pause_srv = self.create_service(
            Trigger, "~/pause", self._on_pause
        )
        self._resume_srv = self.create_service(
            Trigger, "~/resume", self._on_resume
        )
        self._state_pub = self.create_publisher(String, "~/state", 10)
        if self.state_publish_rate_hz > 0.0:
            self.create_timer(
                1.0 / self.state_publish_rate_hz, self._publish_state
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
            f"clamp=[{self.clamp_min},{self.clamp_max}] | "
            f"start_paused={self._paused}"
        )
        if self.invert:
            self.get_logger().warn(
                "invert=True: GELLO-open will command robot-CLOSE — CRUSH HAZARD. "
                "This MUST be False for this hardware."
            )
        self.get_logger().info(
            f"subscribing {self._in_topic} -> publishing {self._out_topic} | "
            f"actual feedback {self._actual_topic} | "
            f"pause/resume: ~/pause ~/resume | state: ~/state"
        )

    # ---------------------------------------------------------------------
    def _on_actual(self, msg: Float32) -> None:
        """Record the gripper's actual position (0=open..1=closed) for resume seed."""
        self._actual_pos = float(msg.data)

    # ---------------------------------------------------------------------
    def _on_width(self, msg: Float32) -> None:
        """Map a GELLO width sample to command_percent and publish (deadbanded)."""
        p_raw = float(msg.data)
        now = time.monotonic()
        # ALWAYS record the freshest leader sample — even while paused — so a later
        # ~/resume can gate on its age. This MUST precede the pause early-return.
        self._last_rx_time = now
        self._last_rx_value = p_raw

        # PAUSED: silence output entirely. The Robotiq holds its last commanded
        # position onboard; not-streaming is the safe state (drop hazard mitigated).
        # Return BEFORE the invert/clamp/EMA/deadband pipeline.
        if self._paused:
            return

        p = p_raw
        if self.invert:
            p = 1.0 - p
        # Clamp to [clamp_min, clamp_max].
        p = min(max(p, self.clamp_min), self.clamp_max)
        # EMA (ema_alpha=1.0 => identity, no lag).
        if self._f is None:
            self._f = p
        else:
            self._f = (1.0 - self.ema_alpha) * self._f + self.ema_alpha * p

        # RESUME RAMP: for resume_ramp_s after a resume, slew-limit the published
        # value from the seeded actual position toward the leader target. The
        # deadband gate is DELIBERATELY BYPASSED here: during the ramp we must
        # re-assert an advancing output on every sample to crawl monotonically from
        # the actual position to the leader value; if a deadband skip suppressed a
        # sub-threshold step the output would stall mid-ramp and never converge.
        if now < self._ramp_until:
            prev = self._last_pub if self._last_pub is not None else self._f
            elapsed = now - self._last_pub_time
            max_delta = self.resume_slew_per_s * max(elapsed, 0.0)
            delta = self._f - prev
            if delta > max_delta:
                out_val = prev + max_delta
            elif delta < -max_delta:
                out_val = prev - max_delta
            else:
                out_val = self._f
            out = Float32()
            out.data = float(out_val)
            self._pub.publish(out)
            self._last_pub = out_val
            self._last_pub_time = now
            return

        # DEADBAND gate: skip tiny at-rest changes (normal following).
        if (
            self._last_pub is not None
            and abs(self._f - self._last_pub) < self.deadband
        ):
            return
        out = Float32()
        out.data = float(self._f)
        self._pub.publish(out)
        self._last_pub = self._f
        self._last_pub_time = now

    # ---------------------------------------------------------------------
    def _on_pause(self, request, response):
        """UNCONDITIONAL pause: silence output, always succeed (safe state)."""
        self._paused = True
        response.success = True
        response.message = (
            "paused: gripper output silenced; Robotiq holds last position"
        )
        self.get_logger().info("~/pause: paused (output silenced)")
        return response

    # ---------------------------------------------------------------------
    def _on_resume(self, request, response):
        """FAIL-CLOSED resume: refuse (staying paused + silent) unless a fresh
        leader sample AND an actual gripper position exist. On acceptance, seed at
        the actual position (zero jump) and start the slew ramp. Never publishes."""
        if not self._paused:
            response.success = True
            response.message = "already following (not paused)"
            return response

        now = time.monotonic()
        # Gate (a): fresh leader sample.
        if (
            self._last_rx_time is None
            or (now - self._last_rx_time) > self.staleness_timeout_s
        ):
            age = (
                "never"
                if self._last_rx_time is None
                else f"{now - self._last_rx_time:.2f}s"
            )
            response.success = False
            response.message = (
                f"REFUSED: stale leader (age={age} > "
                f"{self.staleness_timeout_s}s); staying paused & silent"
            )
            self.get_logger().warn(response.message)
            return response

        # Gate (b): actual gripper position known — prefer the measured feedback,
        # fall back to this bridge's own last-published value, else refuse.
        seed = self._actual_pos if self._actual_pos is not None else self._last_pub
        if seed is None:
            response.success = False
            response.message = (
                "REFUSED: no actual gripper position (no position_percent feedback "
                "and nothing published yet); staying paused & silent"
            )
            self.get_logger().warn(response.message)
            return response

        # ACCEPT: seed from the actual position for a zero-jump first output, then
        # ramp. Do NOT publish from here — the next _on_width emits the seeded/
        # slew-limited value.
        self._f = seed
        self._last_pub = seed
        self._last_pub_time = now
        self._ramp_until = now + self.resume_ramp_s
        self._paused = False
        response.success = True
        response.message = (
            f"resumed: seeded at actual={seed:.3f}, ramping to leader over "
            f"{self.resume_ramp_s:.1f}s (slew {self.resume_slew_per_s:.2f}/s)"
        )
        self.get_logger().info(response.message)
        return response

    # ---------------------------------------------------------------------
    def _publish_state(self) -> None:
        """Publish PAUSED / WAITING / RAMPING / FOLLOWING (in precedence order)."""
        now = time.monotonic()
        if self._paused:
            s = "PAUSED"
        elif (
            self._last_rx_time is None
            or (now - self._last_rx_time) > self.staleness_timeout_s
        ):
            s = "WAITING"
        elif now < self._ramp_until:
            s = "RAMPING"
        else:
            s = "FOLLOWING"
        msg = String()
        msg.data = s
        self._state_pub.publish(msg)


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
