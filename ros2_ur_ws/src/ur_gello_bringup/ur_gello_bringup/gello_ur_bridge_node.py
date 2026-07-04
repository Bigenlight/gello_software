#!/usr/bin/env python3
"""GELLO -> UR forward_position_controller streaming bridge.

This node subscribes to GELLO joint states (~30 Hz) and republishes them as
position commands to the UR ``forward_position_controller`` at a higher rate
(configurable via ``publish_rate_hz``; 250 Hz for ur7e in ur7e_gello.yaml) so
the UR driver's servoj loop receives a smooth, evenly spaced stream. Between
GELLO samples the command is EMA-smoothed and slew-rate limited, then upsampled
by the publish timer.

Safety / deployment notes
-------------------------
* forward_position_controller does NOT interpolate: whatever position you send
  is commanded immediately. The driver checks each command as delta/0.002s
  against the joint velocity limit (3.14 rad/s), so a large one-cycle jump is
  rejected ("External Control speed limit").
* NO START-UP SNAP: this node seeds its command from the robot's ACTUAL current
  pose, read from ``/joint_states`` (``joint_states_topic``, published by
  joint_state_broadcaster on both real hardware and mock). On the first cycle
  the published command equals where the arm already is (zero jump); every
  subsequent cycle slews toward the GELLO pose by at most ``max_step_rad``. So
  the arm ramps smoothly from its real pose to the GELLO pose regardless of the
  gap, and no command ever exceeds the per-cycle velocity limit. If
  ``/joint_states`` has not arrived yet the node publishes nothing and waits.
* The ``scaled_joint_trajectory_controller`` move-to-start handshake
  (``gello_move_to_start``, run before this bridge) is still used to gate on the
  External Control program and to STRICT-switch to forward_position_controller;
  with actual-pose seeding it is defense-in-depth for the snap, not the sole
  safeguard.
* STALENESS WATCHDOG: if GELLO input stops (unplugged, crashed, driver hang)
  the node STOPS publishing rather than repeating the last command forever.
  Not streaming is the fail-safe state for a position controller.
* Timing uses ``time.monotonic()`` for the staleness check on purpose: it is
  immune to wall-clock steps and to sim-time (``use_sim_time``) surprises, so
  the watchdog measures real elapsed wall time regardless of clock config.
"""

import math
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


class _OneEuro:
    """Scalar 1-Euro filter (Casiez, Roussel & Vogel 2012) at a FIXED period.

    Adaptive low-pass tuned for teleop input jitter: it low-passes the signal
    with a cutoff that RISES with the (low-passed) signal speed. So when the
    GELLO joint is nearly still it smooths HARD (kills hand tremor + Dynamixel
    encoder/motor noise), and when you move it fast the cutoff opens up so the
    robot tracks with little lag. This beats a fixed EMA, which must trade lag
    for smoothing on *every* sample.

    Tuning: ``min_cutoff`` (Hz) sets the at-rest smoothing — LOWER = smoother /
    less jitter (more lag when slow). ``beta`` sets how fast the cutoff opens
    with speed — HIGHER = snappier on fast moves (less smoothing). ``d_cutoff``
    low-passes the internal speed estimate so jitter does not inflate the cutoff.

    GELLO samples arrive at ~30 Hz while this bridge publishes at 250 Hz. The
    speed estimate must therefore be updated at the GELLO sample cadence, not on
    every publish tick, otherwise each 30 Hz sample step is interpreted as a
    much faster 250 Hz jump and the filter opens up in visible pulses.
    """

    def __init__(self, dt: float, min_cutoff: float, beta: float,
                 d_cutoff: float) -> None:
        self._dt = dt
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._raw_prev: float | None = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        # Smoothing factor of a 1st-order low-pass at the given cutoff (Hz).
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def seed(self, x: float) -> None:
        """Preload the filter state (e.g. from the robot's actual pose)."""
        self._x_prev = x
        self._raw_prev = x
        self._dx_prev = 0.0

    def update_input(self, x: float, dt: float | None) -> None:
        """Update the low-passed input-speed estimate at the source cadence."""
        if self._raw_prev is None or dt is None or dt <= 1e-6:
            self._raw_prev = x
            return
        dx = (x - self._raw_prev) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        self._dx_prev = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._raw_prev = x

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self.seed(x)
            return x
        # Speed-adaptive cutoff: faster motion -> higher cutoff -> less lag.
        cutoff = self._min_cutoff + self._beta * abs(self._dx_prev)
        a = self._alpha(cutoff, self._dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


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
        # Noise gate: ignore GELLO motion smaller than this (rad) so hand tremor
        # and Dynamixel encoder/motor noise do not jitter the arm while holding
        # still. 0.0 = off. A small value (e.g. 0.003-0.01) freezes at-rest
        # jitter with negligible lag on intentional motion.
        self.deadband_rad = float(
            self.declare_parameter("deadband_rad", 0.0).value
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

        # Smoothing filter: "one_euro" (adaptive, best for tremor) or "ema".
        # one_euro filters the GELLO joint values directly with a speed-adaptive
        # cutoff: heavy smoothing at rest (kills tremor), light when moving fast.
        self.filter_type = str(
            self.declare_parameter("filter_type", "ema").value
        ).lower()
        self.one_euro_min_cutoff = float(
            self.declare_parameter("one_euro_min_cutoff", 1.0).value
        )
        self.one_euro_beta = float(
            self.declare_parameter("one_euro_beta", 2.0).value
        )
        self.one_euro_d_cutoff = float(
            self.declare_parameter("one_euro_d_cutoff", 1.0).value
        )
        # Per-joint 1-Euro filters (None when filter_type != "one_euro").
        self._euro: list[_OneEuro] | None = None
        if self.filter_type == "one_euro":
            _dt = 1.0 / self.publish_rate_hz
            self._euro = [
                _OneEuro(_dt, self.one_euro_min_cutoff, self.one_euro_beta,
                         self.one_euro_d_cutoff)
                for _ in range(len(UR_JOINT_ORDER))
            ]

        # --- State -------------------------------------------------------
        # Raw target reordered into UR_JOINT_ORDER (6 floats) from last good msg.
        self._raw_target: list[float] | None = None
        # Monotonic timestamp (s) of the last good GELLO message.
        self._last_good_msg_time: float | None = None
        # Deadband-gated target (6 floats): a joint only updates when GELLO moves
        # more than deadband_rad from this held value, killing at-rest jitter.
        self._gated_target: list[float] | None = None
        # EMA-filtered command (6 floats), seeded on first valid target.
        self._filtered: list[float] | None = None
        # Last command actually published (6 floats); slew clamp is relative to it.
        self._last_published: list[float] | None = None
        # Robot's ACTUAL current pose (6 floats, UR order) from /joint_states.
        # We SEED the command from this (not the GELLO pose) so the first command
        # equals where the arm already is (zero jump), then slew toward GELLO at
        # <= max_step_rad per cycle. This is what prevents the start-up snap that
        # trips the UR "External Control speed limit".
        self._actual_pose: list[float] | None = None

        # --- ROS interfaces ----------------------------------------------
        self._js_topic = str(
            self.declare_parameter("joint_states_topic", "/joint_states").value
        )
        self._pub = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10
        )
        self._sub = self.create_subscription(
            JointState, "/gello/joint_states", self._on_joint_state, 10
        )
        # Robot's actual joint state (published by joint_state_broadcaster, UR
        # names) — real hardware AND mock both publish it.
        self._actual_sub = self.create_subscription(
            JointState, self._js_topic, self._on_actual_joint_state, 10
        )
        self._timer = self.create_timer(
            1.0 / self.publish_rate_hz, self._on_timer
        )

        # --- Startup log -------------------------------------------------
        if self._euro is not None:
            _filt = (
                f"filter=one_euro(min_cutoff={self.one_euro_min_cutoff}Hz "
                f"beta={self.one_euro_beta} d_cutoff={self.one_euro_d_cutoff}Hz)"
            )
        else:
            _filt = f"filter=ema(alpha={self.ema_alpha} deadband_rad={self.deadband_rad})"
        self.get_logger().info(
            "gello_ur_bridge started | "
            f"{_filt} "
            f"max_step_rad={self.max_step_rad} "
            f"staleness_timeout_s={self.staleness_timeout_s} "
            f"publish_rate_hz={self.publish_rate_hz}"
        )
        self.get_logger().info(
            f"UR joint order: {UR_JOINT_ORDER}; seeding from actual "
            f"joint state on {self._js_topic}"
        )

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
        raw_target = [float(name_to_pos[j]) for j in UR_JOINT_ORDER]
        now = time.monotonic()
        prev_time = self._last_good_msg_time
        if self._euro is not None:
            dt = None if prev_time is None else now - prev_time
            for i in range(len(UR_JOINT_ORDER)):
                self._euro[i].update_input(raw_target[i], dt)
        self._raw_target = raw_target
        self._last_good_msg_time = now

    # ---------------------------------------------------------------------
    def _on_actual_joint_state(self, msg: JointState) -> None:
        """Track the robot's ACTUAL current pose (reordered BY NAME).

        Published by joint_state_broadcaster (real hardware and mock). Used only
        to SEED the first command; after seeding the command is driven by the
        GELLO stream + slew clamp.
        """
        name_to_pos = dict(zip(msg.name, msg.position))
        if any(j not in name_to_pos for j in UR_JOINT_ORDER):
            return  # partial / unrelated joint_states; ignore
        self._actual_pose = [float(name_to_pos[j]) for j in UR_JOINT_ORDER]

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

        # First valid target: seed filter + last-published to the ROBOT'S ACTUAL
        # current pose (NOT the GELLO pose). The first published command then
        # equals where the arm already is (zero jump), and every subsequent cycle
        # slews toward the GELLO pose by at most max_step_rad. This is what
        # eliminates the start-up snap that trips the UR speed limit. If the
        # actual pose is not known yet, publish nothing and wait for it.
        if self._filtered is None or self._last_published is None:
            if self._actual_pose is None:
                self.get_logger().warn(
                    f"Waiting for robot joint state on {self._js_topic} to seed "
                    "from the actual pose (not publishing yet)",
                    throttle_duration_sec=_WARN_THROTTLE_S,
                )
                return
            self._filtered = list(self._actual_pose)
            self._last_published = list(self._actual_pose)
            self._gated_target = list(self._raw_target)
            if self._euro is not None:
                for i in range(len(UR_JOINT_ORDER)):
                    self._euro[i].seed(self._actual_pose[i])
            self._publish(self._last_published)
            return

        alpha = self.ema_alpha
        step = self.max_step_rad
        deadband = self.deadband_rad
        use_euro = self._euro is not None
        out: list[float] = []
        for i in range(len(UR_JOINT_ORDER)):
            if use_euro:
                # 1-EURO adaptive low-pass on the GELLO joint value directly:
                # smooths hard when nearly still (kills tremor / Dynamixel noise),
                # opens up when moving fast (low lag). The speed estimate is
                # updated only when new GELLO samples arrive, so 30 Hz sample
                # edges do not look like 250 Hz velocity spikes. No separate
                # deadband needed.
                self._filtered[i] = self._euro[i](self._raw_target[i])
            else:
                # DEADBAND NOISE GATE: only update the held target when GELLO moved
                # more than deadband_rad, so hand tremor / Dynamixel encoder noise
                # is ignored while holding still (0.0 => gate off).
                if abs(self._raw_target[i] - self._gated_target[i]) > deadband:
                    self._gated_target[i] = self._raw_target[i]
                # EMA low-pass per joint toward the (gated) target.
                self._filtered[i] = (
                    (1.0 - alpha) * self._filtered[i] + alpha * self._gated_target[i]
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
