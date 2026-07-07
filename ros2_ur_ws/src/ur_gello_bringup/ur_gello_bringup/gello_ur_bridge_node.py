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
from std_srvs.srv import Trigger

from ur_gello_bringup.angle_utils import circular_dist, wrapped_nearest

# Output index order expected by the UR forward_position_controller.
UR_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Short, UNAMBIGUOUS per-joint labels for the resume alignment report (same order).
UR_JOINT_SHORT = ["pan", "lift", "elbow", "w1", "w2", "w3"]

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
        # SOFT-START: for soft_start_s after every (re)seed the per-cycle slew
        # clamp ramps from a small fraction up to max_step_rad, so the arm eases
        # into closing any residual gap instead of jumping to full slew from a
        # dead stop. Covers ALL (re)seed paths — startup, resume, AND staleness
        # recovery — so none of them can snap. 0.0 = off.
        self.soft_start_s = float(
            self.declare_parameter("soft_start_s", 0.7).value
        )
        # START PAUSED: spawn the bridge before the handshake completes but hold
        # (publish nothing) until gello_move_to_start calls ~/resume after the
        # controller switch. Removes the OnProcessExit cold-start window during
        # which the leader used to keep drifting (widening the handover gap).
        self._paused = bool(
            self.declare_parameter("start_paused", False).value
        )
        # RESUME ALIGNMENT GATE: ~/resume is REFUSED (bridge stays paused) unless
        # the leader and the arm agree within this per-joint tolerance (rad). This
        # hard-gates the operator-resume snap door: without it, resuming after the
        # leader moved while paused would slew the whole accumulated gap at the
        # (soft-started) slew rate. Default 0.05 = 2x the move_to_start chase_tol
        # so the post-handover resume (arm already converged within chase_tol)
        # always passes, while a genuinely misaligned manual resume is refused.
        self.resume_align_tol = float(
            self.declare_parameter("resume_align_tol", 0.05).value
        )
        # Wall-clock (monotonic) of the last (re)seed, for the soft-start ramp.
        self._seed_time: float | None = None
        # Set when the staleness watchdog trips; forces a re-seed (and thus a
        # fresh soft-start) when the stream recovers, so a leader that moved
        # while the stream was dead does not snap on reconnect.
        self._was_stale = False
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
        # Running, globally-continuous ("unwrapped") GELLO target, so a physical
        # crossing of the +/-pi branch cut (or a single-turn Dynamixel raw-tick
        # wrap) never reaches the deadband gate / EMA / 1-euro / slew clamp as a
        # ~2*pi step. Re-anchored to the arm's actual-pose branch at every seed.
        self._unwrapped_target: list[float] | None = None
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
        # PAUSE/RESUME: when paused the timer stops publishing so the robot HOLDS
        # its last commanded pose (forward_position_controller keeps the setpoint).
        # Resume is GATED (see _on_resume): it only proceeds when a fresh leader
        # sample and the arm pose agree within resume_align_tol, so the restart is
        # genuinely jump-free (re-seed from actual + soft-start); a misaligned
        # resume is REFUSED and the bridge stays paused. NOTE: self._paused is
        # initialized from the start_paused parameter above (do not reset here).

        # --- ROS interfaces ----------------------------------------------
        self._js_topic = str(
            self.declare_parameter("joint_states_topic", "/joint_states").value
        )
        # Operator pause/resume of following (driven by the numbered console).
        self._pause_srv = self.create_service(Trigger, "~/pause", self._on_pause)
        self._resume_srv = self.create_service(Trigger, "~/resume", self._on_resume)
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
            f"soft_start_s={self.soft_start_s} "
            f"start_paused={self._paused} "
            f"resume_align_tol={self.resume_align_tol} "
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

        # CONTINUOUS UNWRAP: only ever accumulate the SHORT-way delta from the
        # last unwrapped value, so a branch-cut crossing or a single-turn
        # raw-tick wrap can never appear as a ~2*pi jump to the deadband gate,
        # EMA/1-euro filter, or slew clamp downstream. Identity no-op when no
        # wrap occurs.
        if self._unwrapped_target is None:
            unwrapped = list(raw_target)
        else:
            unwrapped = wrapped_nearest(raw_target, self._unwrapped_target)
        self._unwrapped_target = unwrapped

        now = time.monotonic()
        prev_time = self._last_good_msg_time
        if self._euro is not None:
            dt = None if prev_time is None else now - prev_time
            for i in range(len(UR_JOINT_ORDER)):
                self._euro[i].update_input(unwrapped[i], dt)
        self._raw_target = unwrapped
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
        # PAUSED: stop publishing so the robot holds its last commanded pose.
        if self._paused:
            return

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
            # Force a re-seed on recovery: if the leader moved while the stream
            # was dead, we must NOT resume by closing that gap at full slew. On
            # the next good message the seed branch below re-seeds from the arm's
            # actual pose and restarts the soft-start ramp.
            self._filtered = None
            self._last_published = None
            self._was_stale = True
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
            # RE-ANCHOR the unwrapped target onto the branch NEAREST the robot's
            # actual pose, so the initial (and post-resume / post-stale) seed
            # slew closes only the true SHORT physical gap even when a joint's
            # neutral (e.g. wrist_3 with a ~pi ergonomic offset) sits by the cut
            # and the arm reports the opposite branch. Without this the slew
            # would crawl ~2*pi the LONG way (a full-rev spin in the wrong
            # direction) — a latent bug that existed even before the wraparound
            # offset, since _gated_target seeded from raw while _filtered seeded
            # from actual.
            anchored = wrapped_nearest(self._raw_target, self._actual_pose)
            self._unwrapped_target = anchored
            self._raw_target = anchored
            self._filtered = list(self._actual_pose)
            self._last_published = list(self._actual_pose)
            self._gated_target = list(anchored)
            if self._euro is not None:
                for i in range(len(UR_JOINT_ORDER)):
                    self._euro[i].seed(self._actual_pose[i])
            self._seed_time = time.monotonic()  # start the soft-start ramp
            if self._was_stale:
                self.get_logger().info("GELLO stream recovered; re-seeded from "
                                       "actual pose with soft-start.")
                self._was_stale = False
            self._publish(self._last_published)
            return

        alpha = self.ema_alpha
        # SOFT-START: ramp the effective slew clamp from ~15% up to full over
        # soft_start_s after the last (re)seed, so gap closure eases in instead of
        # snapping to full slew from a standstill. After the ramp, step == max.
        step = self.max_step_rad
        if self.soft_start_s > 0.0 and self._seed_time is not None:
            frac = (time.monotonic() - self._seed_time) / self.soft_start_s
            if frac < 1.0:
                step = self.max_step_rad * (0.15 + 0.85 * max(0.0, frac))
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
    def _on_pause(self, request, response):
        """'2) 정지': stop following; robot holds its last commanded pose."""
        self._paused = True
        response.success = True
        response.message = "Following PAUSED — robot holds position. Resume with proceed."
        self.get_logger().warn("Following PAUSED by operator (robot holds pose).")
        return response

    def _on_resume(self, request, response):
        """'1) 진행': resume following — GATED on leader/arm alignment.

        REFUSES (stays paused, no re-seed, publishes nothing) unless ALL hold:
          (a) a FRESH GELLO sample exists (age <= staleness_timeout_s),
          (b) the arm's ACTUAL pose is known (/joint_states seen), and
          (c) every joint agrees within resume_align_tol: |raw GELLO - actual|.

        This closes the operator-resume snap door with a hard gate, not just the
        soft-start mitigation: resuming after the leader moved while paused would
        otherwise slew the whole accumulated gap. On refusal we report the
        offending joints so the operator can re-align the leader and resume again.
        On success we re-seed from the actual pose (jump-free restart) and let the
        soft-start ramp ease the arm toward the (now-aligned) GELLO pose.
        """
        if not self._paused:
            response.success = True
            response.message = "Already following (was not paused)."
            self.get_logger().info("Resume requested but already following.")
            return response

        # (a) FRESH leader sample required — a stale/dead stream cannot resume.
        age = (
            None if self._last_good_msg_time is None
            else time.monotonic() - self._last_good_msg_time
        )
        if self._raw_target is None or age is None or age > self.staleness_timeout_s:
            age_str = "n/a" if age is None else f"{age:.2f}s"
            response.success = False
            response.message = (
                f"Resume REFUSED — no fresh GELLO sample (age={age_str} > "
                f"{self.staleness_timeout_s:.2f}s). Bridge stays PAUSED (holds "
                "pose). Restore the leader stream, then resume again."
            )
            self.get_logger().warn(response.message)
            return response

        # (b) arm pose must be known to measure the gap.
        if self._actual_pose is None:
            response.success = False
            response.message = (
                "Resume REFUSED — robot actual pose unknown (no /joint_states "
                "yet). Bridge stays PAUSED (holds pose)."
            )
            self.get_logger().warn(response.message)
            return response

        # (c) per-joint alignment gate.
        gap = [
            circular_dist(self._raw_target[i], self._actual_pose[i])
            for i in range(len(UR_JOINT_ORDER))
        ]
        max_gap = max(gap)
        if max_gap > self.resume_align_tol:
            worst = max(range(len(gap)), key=lambda i: gap[i])
            per = ", ".join(
                f"{UR_JOINT_SHORT[i]}={gap[i]:.3f}" for i in range(len(gap))
            )
            response.success = False
            response.message = (
                f"Resume REFUSED — leader/arm misaligned: max {max_gap:.3f} rad "
                f"at {UR_JOINT_SHORT[worst]} > resume_align_tol "
                f"{self.resume_align_tol:.3f} ({per}). Bridge stays PAUSED (holds "
                "pose). Re-align the GELLO leader to the arm, then resume again."
            )
            self.get_logger().warn(response.message)
            return response

        # Aligned: resume + force a re-seed from actual (jump-free) + soft-start.
        self._paused = False
        self._filtered = None
        self._last_published = None
        response.success = True
        response.message = (
            f"Following RESUMED — aligned (max {max_gap:.3f} rad <= "
            f"{self.resume_align_tol:.3f}). Arm re-seeds from its current pose "
            "and soft-starts toward GELLO (rate-limited). Keep clear."
        )
        self.get_logger().warn(response.message)
        return response

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
