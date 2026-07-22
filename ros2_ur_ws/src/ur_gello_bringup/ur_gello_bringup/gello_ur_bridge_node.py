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

import json
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from ur_gello_bringup.angle_utils import (
    circular_dist,
    leader_quasi_still,
    wrapped_nearest,
)
# Pure, rclpy-independent command-pipeline stages + the 1-Euro filter. Keeping
# them in a separate module lets the venv test-suite prove the joint path is
# bit-identical to the legacy interleaved loop (the code tested IS the code run).
# NOTE: ur_kin / eef_delta are imported LAZILY (only when control_mode=="eef")
# so joint mode boots without the analytic-IK dependency stack.
from ur_gello_bringup.bridge_stages import (
    OneEuro,
    command_pipeline,
)

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


class GelloUrBridge(Node):
    """Bridge GELLO joint states to UR forward_position_controller commands."""

    def __init__(self) -> None:
        super().__init__("gello_ur_bridge")

        # --- Parameters --------------------------------------------------
        # CONTROL MODE (launch/yaml, NOT a runtime service): "joint" (default)
        # keeps the legacy joint-passthrough teleop bit-identical; "eef" enables
        # the end-effector delta-control path (needs the config/*_eef.yaml
        # overlay). The eef_* params below are ALWAYS declared (so the overlay
        # can set them) but are only consumed in eef mode.
        self.control_mode = str(
            self.declare_parameter("control_mode", "joint").value
        ).lower()
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
        # RESUME-CHASE gate params (the additive ~/resume_chase service). Unlike
        # the strict ~/resume (which demands the leader already sit within
        # resume_align_tol of the arm), resume_chase authorizes a rate-limited
        # GLIDE across a larger — but bounded — gap, under a quasi-still leader.
        # It never introduces a new publishing path: it only drops the bridge
        # into the existing seed branch (jump-free + soft-started + slew-clamped).
        # Per-joint circular-gap hard cap (rad): above this the service REFUSES.
        self.resume_chase_max_gap = float(
            self.declare_parameter("resume_chase_max_gap", 1.5).value
        )
        # Leader "quasi-still" speed threshold (rad/s) for the resume_chase gate.
        self.resume_chase_still_speed = float(
            self.declare_parameter("resume_chase_still_speed", 0.10).value
        )
        # Window (s) over which the quasi-still speed is measured.
        self.resume_chase_still_window_s = float(
            self.declare_parameter("resume_chase_still_window_s", 0.3).value
        )
        # Rate (Hz) of the ~/state String publisher (operator UI status).
        self.state_publish_rate_hz = float(
            self.declare_parameter("state_publish_rate_hz", 5.0).value
        )
        # Per-joint circular tolerance (rad) below which the state reports
        # FOLLOWING rather than CHASING (i.e. the glide is done).
        self.state_chase_done_tol = float(
            self.declare_parameter("state_chase_done_tol", 0.10).value
        )

        # --- EEF-mode parameters (declared ALWAYS; consumed only in eef mode) --
        # engage gate G4: max_i|_last_published - _actual_pose| must be <= this
        # (the two chains must already agree, else engaging would snap).
        self.anchor_agree_tol = float(
            self.declare_parameter("anchor_agree_tol", 0.02).value
        )
        # engage gate G6: max_i|q_lead_f - _raw_target| must be < this (the EEF
        # leader 1-Euro filter must have positively CONVERGED before we anchor).
        self.filter_settled_tol = float(
            self.declare_parameter("filter_settled_tol", 0.005).value
        )
        # ~/eef/state publish rate (Hz).
        self.eef_state_rate_hz = float(
            self.declare_parameter("eef_state_rate_hz", 10.0).value
        )
        # HOLD -> DISENGAGED latch (s): a continuous HOLD longer than this
        # auto-disengages + invalidates the anchor (anti permanent saturation).
        self.hold_latch_s = float(
            self.declare_parameter("hold_latch_s", 2.0).value
        )
        # Per-tick EEF-stage time budget (us). N consecutive overruns => fail
        # closed to DISENGAGED (timer overrun -> command jitter -> protective stop).
        self.tick_budget_us = float(
            self.declare_parameter("tick_budget_us", 1000.0).value
        )
        self.tick_overrun_limit = int(
            self.declare_parameter("tick_overrun_limit", 5).value
        )
        # EefDeltaController cfg keys (numeric-only; unused in joint mode). Kept
        # as declared params so the eef overlay yaml drives them 1:1.
        self.pos_scale = float(self.declare_parameter("pos_scale", 1.0).value)
        self.r_align_rpy = list(
            self.declare_parameter("r_align_rpy", [0.0, 0.0, 0.0]).value
        )
        self.tool_l_xyz_rpy = list(
            self.declare_parameter(
                "tool_l_xyz_rpy", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ).value
        )
        self.tool_r_xyz_rpy = list(
            self.declare_parameter(
                "tool_r_xyz_rpy", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ).value
        )
        self.eef_v_max = float(self.declare_parameter("v_max", 0.08).value)
        self.eef_w_max = float(self.declare_parameter("w_max", 0.5).value)
        self.sigma_warn = float(self.declare_parameter("sigma_warn", 0.10).value)
        self.sigma_stop = float(self.declare_parameter("sigma_stop", 0.03).value)
        self.gamma_min = float(self.declare_parameter("gamma_min", 0.05).value)
        self.char_length = float(self.declare_parameter("char_length", 0.30).value)
        self.branch_tol = float(self.declare_parameter("branch_tol", 0.25).value)
        self.branch_weights = list(
            self.declare_parameter(
                "branch_weights", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ).value
        )
        self.limit_margin_rad = float(
            self.declare_parameter("limit_margin_rad", 0.05).value
        )
        self.s_floor = float(self.declare_parameter("s_floor", 0.02).value)
        self.lag_max_pose = list(
            self.declare_parameter("lag_max_pose", [0.05, 0.3]).value
        )
        self.max_excursion_m = float(
            self.declare_parameter("max_excursion_m", 0.5).value
        )
        self.ik_backend = str(
            self.declare_parameter("ik_backend", "analytic").value
        )
        # keepout is a nested dict; ROS2 params are flat, so accept a JSON string.
        self.keepout_json = str(self.declare_parameter("keepout_json", "{}").value)

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
        self._euro: list[OneEuro] | None = None
        if self.filter_type == "one_euro":
            _dt = 1.0 / self.publish_rate_hz
            self._euro = [
                OneEuro(_dt, self.one_euro_min_cutoff, self.one_euro_beta,
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
        # Recent leader samples (oldest-first) as (monotonic_ts, unwrapped_pose)
        # for the resume_chase quasi-still gate. Stores the UNWRAPPED target;
        # circular_dist inside leader_quasi_still makes any 2*pi re-anchor
        # discontinuity in the history harmless (short-way distance).
        self._gello_history: deque = deque(maxlen=64)
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
        # ADDITIVE resume-chase service (separate from the frozen ~/resume): a
        # gated glide across a bounded gap under a quasi-still leader.
        self._resume_chase_srv = self.create_service(
            Trigger, "~/resume_chase", self._on_resume_chase
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
        # Operator-UI status topic: PAUSED / WAITING / STALE / CHASING / FOLLOWING.
        self._state_pub = self.create_publisher(String, "~/state", 10)
        _state_hz = self.state_publish_rate_hz if self.state_publish_rate_hz > 0.0 else 5.0
        self._state_timer = self.create_timer(
            1.0 / _state_hz, self._on_state_timer
        )

        # --- EEF mode setup ----------------------------------------------
        # State machine string: BOOTSTRAP (pre-engage, joint passthrough) /
        # ENGAGED (eef delta control) / DISENGAGED (== _paused, anchor void).
        # In joint mode this stays "BOOTSTRAP" and is never used.
        self._eef_state = "BOOTSTRAP"
        # EefDeltaController instance and its q_lead 1-Euro filter bank + the
        # lazily-imported ur_kin / numpy handles. All None in joint mode.
        self._eef = None
        self._euro_lead: list[OneEuro] | None = None
        self._ur_kin = None
        self._np = None
        # Latest EefDeltaController.step() info dict (for ~/eef/state + poses).
        self._eef_info: dict | None = None
        # Latest smoothed leader joint vector (for the ~/eef/leader_pose diag).
        self._eef_leader_q: list[float] | None = None
        # Monotonic ts the current continuous HOLD started (None if not holding).
        self._hold_since: float | None = None
        # Last auto-transition reason string (for the ~/eef/state diagnostic).
        self._eef_auto_reason: str | None = None
        # Consecutive EEF-stage time-budget overruns (fail-closed at the limit).
        self._tick_overruns = 0
        # Publish-tick counter for decimating the PoseStamped diagnostics.
        self._eef_tick = 0
        self._eef_pose_decim = max(1, int(round(self.publish_rate_hz / 30.0)))

        if self.control_mode == "eef":
            # LAZY IMPORT: ur_kin / eef_delta (and their analytic-IK dependency
            # stack) are imported ONLY here, so joint mode never needs them.
            import numpy as np
            from ur_gello_bringup import ur_kin as _ur_kin
            from ur_gello_bringup.eef_delta import EefDeltaController
            self._np = np
            self._ur_kin = _ur_kin
            try:
                keepout = json.loads(self.keepout_json) if self.keepout_json else {}
            except (ValueError, TypeError):
                self.get_logger().warn(
                    f"keepout_json is not valid JSON ({self.keepout_json!r}); "
                    "using empty keepout"
                )
                keepout = {}
            eef_cfg = {
                "pos_scale": self.pos_scale,
                "r_align_rpy": self.r_align_rpy,
                "tool_l_xyz_rpy": self.tool_l_xyz_rpy,
                "tool_r_xyz_rpy": self.tool_r_xyz_rpy,
                "v_max": self.eef_v_max,
                "w_max": self.eef_w_max,
                "sigma_warn": self.sigma_warn,
                "sigma_stop": self.sigma_stop,
                "gamma_min": self.gamma_min,
                "char_length": self.char_length,
                "branch_tol": self.branch_tol,
                "branch_weights": self.branch_weights,
                "limit_margin_rad": self.limit_margin_rad,
                "s_floor": self.s_floor,
                "lag_max_pose": self.lag_max_pose,
                "max_excursion_m": self.max_excursion_m,
                "dt": 1.0 / self.publish_rate_hz,
                "keepout": keepout,
                "ik_backend": self.ik_backend,
            }
            self._eef = EefDeltaController(eef_cfg)
            # DEDICATED q_lead 1-Euro bank (separate from the joint self._euro):
            # always fed / seeded from _raw_target so engage-time filter
            # convergence is checked against the live leader (gate G6).
            _dt = 1.0 / self.publish_rate_hz
            self._euro_lead = [
                OneEuro(_dt, self.one_euro_min_cutoff, self.one_euro_beta,
                        self.one_euro_d_cutoff)
                for _ in range(len(UR_JOINT_ORDER))
            ]
            # Clutch services (fail-closed Trigger, mirror the ~/resume skeleton).
            self._eef_engage_srv = self.create_service(
                Trigger, "~/eef_engage", self._on_eef_engage
            )
            self._eef_disengage_srv = self.create_service(
                Trigger, "~/eef_disengage", self._on_eef_disengage
            )
            self._eef_reclutch_srv = self.create_service(
                Trigger, "~/eef_reclutch", self._on_eef_reclutch
            )
            self._eef_to_joint_srv = self.create_service(
                Trigger, "~/eef_to_joint", self._on_eef_to_joint
            )
            # Diagnostic topics.
            self._eef_state_pub = self.create_publisher(String, "~/eef/state", 10)
            self._eef_leader_pub = self.create_publisher(
                PoseStamped, "~/eef/leader_pose", 10
            )
            self._eef_desired_pub = self.create_publisher(
                PoseStamped, "~/eef/desired_pose", 10
            )
            self._eef_commanded_pub = self.create_publisher(
                PoseStamped, "~/eef/commanded_pose", 10
            )
            _eef_hz = self.eef_state_rate_hz if self.eef_state_rate_hz > 0.0 else 10.0
            self._eef_state_timer = self.create_timer(
                1.0 / _eef_hz, self._on_eef_state_timer
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
            f"control_mode={self.control_mode} "
            f"{_filt} "
            f"max_step_rad={self.max_step_rad} "
            f"soft_start_s={self.soft_start_s} "
            f"start_paused={self._paused} "
            f"resume_align_tol={self.resume_align_tol} "
            f"resume_chase_max_gap={self.resume_chase_max_gap} "
            f"resume_chase_still_speed={self.resume_chase_still_speed} "
            f"resume_chase_still_window_s={self.resume_chase_still_window_s} "
            f"state_publish_rate_hz={self.state_publish_rate_hz} "
            f"state_chase_done_tol={self.state_chase_done_tol} "
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
        _dt = None if prev_time is None else now - prev_time
        if self._euro is not None:
            for i in range(len(UR_JOINT_ORDER)):
                self._euro[i].update_input(unwrapped[i], _dt)
        # DEDICATED EEF leader filter (eef mode only): fed the SAME unwrapped
        # leader stream at the source cadence, kept independent of the joint
        # self._euro so the joint path stays bit-identical.
        if self._euro_lead is not None:
            for i in range(len(UR_JOINT_ORDER)):
                self._euro_lead[i].update_input(unwrapped[i], _dt)
        self._raw_target = unwrapped
        self._last_good_msg_time = now
        # Record the UNWRAPPED pose for the resume_chase quasi-still gate.
        # circular_dist inside leader_quasi_still handles any 2*pi re-anchor
        # discontinuity in this history harmlessly (measures the short-way gap).
        self._gello_history.append((now, list(unwrapped)))

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
            # EEF: a stale leader invalidates the anchor — during the outage the
            # operator may have moved GELLO, so reusing the old anchor would mean
            # a large accumulated delta. Fail closed via the shared helper so this
            # path matches the other auto-disengage paths (hold-latch / tick-
            # overrun / exception): it sets _paused=True, honouring the
            # DISENGAGED==_paused invariant and the plan's "re-engage required
            # after recovery" fail-safe. Without _paused=True the label would say
            # DISENGAGED while the arm silently resumed joint-passthrough chasing.
            if self.control_mode == "eef" and self._eef_state == "ENGAGED":
                self._eef_autodisengage("leader_stale")
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
            # EEF leader filter ALWAYS seeds from the (anchored) raw leader pose,
            # NOT from the arm's actual pose: it tracks the leader, not the robot.
            if self._euro_lead is not None:
                for i in range(len(UR_JOINT_ORDER)):
                    self._euro_lead[i].seed(anchored[i])
            self._seed_time = time.monotonic()  # start the soft-start ramp
            if self._was_stale:
                self.get_logger().info("GELLO stream recovered; re-seeded from "
                                       "actual pose with soft-start.")
                self._was_stale = False
            self._publish(self._last_published)
            return

        # SOFT-START: ramp the effective slew clamp from ~15% up to full over
        # soft_start_s after the last (re)seed, so gap closure eases in instead of
        # snapping to full slew from a standstill. After the ramp, step == max.
        step = self.max_step_rad
        if self.soft_start_s > 0.0 and self._seed_time is not None:
            frac = (time.monotonic() - self._seed_time) / self.soft_start_s
            if frac < 1.0:
                step = self.max_step_rad * (0.15 + 0.85 * max(0.0, frac))

        # ---- THREE-STAGE PIPELINE (bridge_stages, pure + venv-tested) ----
        # (1) filter stage: fill _filtered with the per-joint joint-mode math
        #     (one_euro or ema+deadband) — IDENTICAL arithmetic to the legacy
        #     interleaved loop. (2) eef stage: only in eef mode + ENGAGED, run
        #     the EefDeltaController and OVERWRITE _filtered with its slew-limited
        #     q_cmd (or hold). (3) clamp stage: the legacy slew clamp.
        # In joint mode stage (2) is a no-op, so the output is bit-identical to
        # the old loop (proven in test/test_bridge_stages.py).
        eef_command = None
        if self.control_mode == "eef" and self._eef_state == "ENGAGED":
            # EEF stage runs inside the same try/except so any overrun/exception
            # fails CLOSED to DISENGAGED rather than killing the timer callback.
            try:
                t0 = time.monotonic()
                # DEDICATED leader filter (never the joint self._euro): the eef
                # controller consumes the SMOOTHED leader joint vector.
                q_lead_f = [
                    self._euro_lead[i](self._raw_target[i])
                    for i in range(len(UR_JOINT_ORDER))
                ]
                # step_eff carries the soft-start-ramped clamp so the downstream
                # slew clamp does not additionally bind the eef increment.
                q_cmd, info = self._eef.step(q_lead_f, step)
                self._eef_info = info
                self._eef_leader_q = q_lead_f
                if q_cmd is not None:
                    eef_command = [float(v) for v in q_cmd]
                # HOLD latch: a continuous HOLD longer than hold_latch_s
                # auto-disengages (anti permanent saturation).
                if info.get("state") == "HOLD":
                    if self._hold_since is None:
                        self._hold_since = t0
                    elif (t0 - self._hold_since) > self.hold_latch_s:
                        self._eef_autodisengage("hold_latched "
                                                f"(reason={info.get('reject_reason')})")
                        eef_command = None
                else:
                    self._hold_since = None
                # Time-budget watchdog: N consecutive overruns => fail closed.
                elapsed_us = (time.monotonic() - t0) * 1e6
                if elapsed_us > self.tick_budget_us:
                    self._tick_overruns += 1
                    if self._tick_overruns >= self.tick_overrun_limit:
                        self._eef_autodisengage(
                            f"tick_budget exceeded {self.tick_overrun_limit}x "
                            f"(last {elapsed_us:.0f}us > {self.tick_budget_us:.0f}us)"
                        )
                        eef_command = None
                else:
                    self._tick_overruns = 0
            except Exception as exc:  # noqa: BLE001 fail-closed on ANY eef fault
                self._eef_autodisengage(f"exception: {exc}")
                eef_command = None

        # FAIL-CLOSED: if the EEF stage auto-disengaged this tick it also PAUSED
        # the bridge. Hold the current pose exactly and stop; the next tick's
        # top-level paused check keeps it held (operator must ~/resume / re-engage).
        if self._paused:
            if self._last_published is not None:
                self._publish(self._last_published)
            return

        out = command_pipeline(
            self.control_mode,
            self._eef_state == "ENGAGED",
            self._raw_target,
            self._filtered,
            self._gated_target,
            self._euro,
            self.ema_alpha,
            self.deadband_rad,
            self._last_published,
            step,
            eef_command=eef_command,
        )

        self._last_published = out
        self._publish(out)
        # Decimated EEF diagnostic poses (leader / desired / commanded).
        if self.control_mode == "eef" and self._eef_state == "ENGAGED":
            self._eef_tick += 1
            if self._eef_tick % self._eef_pose_decim == 0:
                self._publish_eef_poses()

    # ---------------------------------------------------------------------
    def _on_pause(self, request, response):
        """'2) 정지': stop following; robot holds its last commanded pose."""
        self._paused = True
        # A pause carries an automatic EEF disengage (anchor invalidated): the
        # same immediate-stop path used by ~/eef_disengage.
        if self.control_mode == "eef" and self._eef is not None:
            self._eef.disengage()
            self._eef_state = "DISENGAGED"
            self._hold_since = None
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
        # EEF: resume returns to BOOTSTRAP (joint passthrough); a fresh
        # ~/eef_engage is required to re-enter delta control.
        if self.control_mode == "eef":
            self._eef_state = "BOOTSTRAP"
        response.success = True
        response.message = (
            f"Following RESUMED — aligned (max {max_gap:.3f} rad <= "
            f"{self.resume_align_tol:.3f}). Arm re-seeds from its current pose "
            "and soft-starts toward GELLO (rate-limited). Keep clear."
        )
        self.get_logger().warn(response.message)
        return response

    def _on_resume_chase(self, request, response):
        """Resume by GLIDING across a bounded gap (additive to ~/resume).

        Unlike the strict ~/resume (leader must already sit within
        resume_align_tol of the arm), resume_chase authorizes the arm to close a
        larger — but HARD-CAPPED — gap by re-seeding into the bridge's EXISTING
        seed branch: the next tick re-anchors the leader target onto the branch
        nearest the arm's actual pose, seeds from that actual pose (zero jump),
        restarts the soft-start ramp, and glides at <= (ramped) max_step_rad per
        cycle. No new publishing path is introduced.

        FAIL CLOSED: refused unless ALL of (a) fresh leader sample, (b) actual
        pose known, (c) leader quasi-still, (d) every joint's circular gap within
        resume_chase_max_gap. On any refusal the bridge stays PAUSED (publishes
        nothing). The ONLY state mutation on acceptance is
        _paused/_filtered/_last_published — the seed branch does the rest.
        """
        if not self._paused:
            response.success = True
            response.message = "Already following (was not paused)."
            self.get_logger().info("Resume-chase requested but already following.")
            return response

        # (a) FRESH leader sample required (same check as ~/resume).
        age = (
            None if self._last_good_msg_time is None
            else time.monotonic() - self._last_good_msg_time
        )
        if self._raw_target is None or age is None or age > self.staleness_timeout_s:
            age_str = "n/a" if age is None else f"{age:.2f}s"
            response.success = False
            response.message = (
                f"Resume-chase REFUSED — no fresh GELLO sample (age={age_str} > "
                f"{self.staleness_timeout_s:.2f}s). Bridge stays PAUSED (holds "
                "pose). Restore the leader stream, then try again."
            )
            self.get_logger().warn(response.message)
            return response

        # (b) arm pose must be known to measure the gap.
        if self._actual_pose is None:
            response.success = False
            response.message = (
                "Resume-chase REFUSED — robot actual pose unknown (no "
                "/joint_states yet). Bridge stays PAUSED (holds pose)."
            )
            self.get_logger().warn(response.message)
            return response

        # (c) leader quasi-still (shared, fail-closed gate). Insufficient history
        # => NOT still => refuse.
        if not leader_quasi_still(
            self._gello_history,
            self.resume_chase_still_window_s,
            self.resume_chase_still_speed,
        ):
            response.success = False
            response.message = (
                "Resume-chase REFUSED — leader is moving (or stillness not yet "
                "established). Hold GELLO steady for ~0.5 s and try again. Bridge "
                "stays PAUSED (holds pose)."
            )
            self.get_logger().warn(response.message)
            return response

        # (d) per-joint circular-gap hard cap.
        gap = [
            circular_dist(self._raw_target[i], self._actual_pose[i])
            for i in range(len(UR_JOINT_ORDER))
        ]
        max_gap = max(gap)
        if max_gap > self.resume_chase_max_gap:
            worst = max(range(len(gap)), key=lambda i: gap[i])
            per = ", ".join(
                f"{UR_JOINT_SHORT[i]}={gap[i]:.3f}" for i in range(len(gap))
            )
            response.success = False
            response.message = (
                f"Resume-chase REFUSED — gap too large: max {max_gap:.3f} rad at "
                f"{UR_JOINT_SHORT[worst]} > resume_chase_max_gap "
                f"{self.resume_chase_max_gap:.3f} ({per}). Bridge stays PAUSED "
                "(holds pose). Hold the GELLO leader closer to the frozen arm "
                "pose, then try again."
            )
            self.get_logger().warn(response.message)
            return response

        # ACCEPTED. Drop into the existing seed branch: the ONLY mutations are
        # these three fields. Do NOT publish here, do NOT touch _seed_time,
        # _unwrapped_target, _gated_target, or the 1-euro filters — the next
        # _on_timer tick re-anchors, seeds from actual (zero jump), and soft-
        # starts the slew-clamped glide.
        self._paused = False
        self._filtered = None
        self._last_published = None
        # EEF: resume_chase also returns to BOOTSTRAP (joint passthrough).
        if self.control_mode == "eef":
            self._eef_state = "BOOTSTRAP"
        # Estimated glide time: gap closed at the sustained slew rate
        # (max_step_rad * publish_rate_hz), plus the soft-start ease-in.
        slew_rate = self.max_step_rad * self.publish_rate_hz
        eta = (max_gap / slew_rate) if slew_rate > 0.0 else float("inf")
        eta += self.soft_start_s
        response.success = True
        response.message = (
            f"Resume-chase ACCEPTED — max gap {max_gap:.3f} rad <= "
            f"{self.resume_chase_max_gap:.3f}; est. glide ~{eta:.1f}s. Arm will "
            "GLIDE to the leader pose — keep clear and hold GELLO still until "
            "FOLLOWING."
        )
        self.get_logger().warn(response.message)
        return response

    # ---------------------------------------------------------------------
    def _on_state_timer(self) -> None:
        """Publish the operator-UI status string at state_publish_rate_hz.

        Precedence: PAUSED > WAITING(no target) > STALE > WAITING(unseeded) >
        CHASING > FOLLOWING. Read-only: never mutates bridge state or publishes
        commands.
        """
        if self._paused:
            state = "PAUSED"
        elif self._raw_target is None or self._last_good_msg_time is None:
            state = "WAITING"
        elif (time.monotonic() - self._last_good_msg_time) > self.staleness_timeout_s:
            state = "STALE"
        elif self._last_published is None:
            state = "WAITING"
        elif max(
            circular_dist(self._last_published[i], self._raw_target[i])
            for i in range(len(UR_JOINT_ORDER))
        ) > self.state_chase_done_tol:
            state = "CHASING"
        else:
            state = "FOLLOWING"
        msg = String()
        msg.data = state
        self._state_pub.publish(msg)

    # =====================================================================
    # EEF mode: automatic transitions, clutch services, diagnostics
    # =====================================================================
    def _eef_autodisengage(self, reason: str) -> None:
        """Fail-closed auto-disengage from within the tick (anchor invalidated).

        Mirrors ~/eef_disengage: PAUSE the bridge (immediate hold on the next
        tick), void the controller anchor, and record the reason for ~/eef/state.
        """
        if self._eef is not None:
            self._eef.disengage()
        self._eef_state = "DISENGAGED"
        self._paused = True
        self._hold_since = None
        self._tick_overruns = 0
        self._eef_auto_reason = reason
        self.get_logger().warn(
            f"EEF auto-disengaged ({reason}); bridge PAUSED (holds pose).",
            throttle_duration_sec=_WARN_THROTTLE_S,
        )

    def _eef_leader_age(self):
        return (
            None if self._last_good_msg_time is None
            else time.monotonic() - self._last_good_msg_time
        )

    def _eef_selftest(self, q_anchor) -> bool:
        """G7: internal IK(FK(q))==q round-trip identity (solver-integrity check).

        Uses OUR fk + OUR ik only, so it is independent of the real robot's DH /
        factory calibration; it catches a coding error or solver-backend mismatch,
        NOT a vendor-DH discrepancy (that is an offline P-1/P6 procedure).
        """
        np = self._np
        uk = self._ur_kin
        try:
            qa = np.asarray(q_anchor, dtype=float).reshape(6)
            T = uk.fk(qa)
            if self.ik_backend == "analytic":
                sols = uk.ik_analytic(T)
                cand = [uk.wrapped_nearest(np.asarray(s, float), qa) for s in sols]
                cand = [c for c in cand if bool(np.all(np.isfinite(c)))]
                if not cand:
                    return False
                best = min(cand, key=lambda c: float(np.max(np.abs(c - qa))))
            else:
                best = uk.ik_numeric(T, qa)
                if best is None:
                    return False
                best = np.asarray(best, dtype=float).reshape(6)
            if float(np.max(np.abs(best - qa))) >= 1e-6:
                return False
            # Pose round-trip: FK(IK(FK(q))) must match FK(q).
            T2 = uk.fk(best)
            xi = uk.se3_log(np.linalg.inv(T2) @ T)
            if float(np.linalg.norm(xi[:3])) >= 1e-4:
                return False
            if float(np.linalg.norm(xi[3:])) >= 1e-4:
                return False
            return True
        except Exception:  # noqa: BLE001 any solver fault -> selftest fails closed
            return False

    def _run_eef_gates(self, run_baseline: bool):
        """Ordered engage/reclutch acceptance gates (fail-closed).

        run_baseline=True  -> full engage sequence G1..G9.
        run_baseline=False -> reclutch: G2,G5,G6,G7,G8,G9 (G3/G4 skipped; the
                              anchor is the frozen command pose _last_published).

        Returns (ok, key, detail, q_anchor, q_lead_f). On failure key is the
        rejection string and q_anchor/q_lead_f are None.
        """
        np = self._np
        uk = self._ur_kin
        n = len(UR_JOINT_ORDER)

        # G1: control mode.
        if self.control_mode != "eef":
            return False, "not_eef_mode", "control_mode != eef", None, None
        # G2: fresh leader sample.
        age = self._eef_leader_age()
        if self._raw_target is None or age is None or age > self.staleness_timeout_s:
            a = "n/a" if age is None else f"{age:.2f}s"
            return False, "leader_stale", f"age={a} > {self.staleness_timeout_s:.2f}s", None, None
        # G3: a command baseline must exist (also the reclutch anchor).
        if self._last_published is None:
            return False, "no_command_baseline", "no _last_published yet", None, None
        # G4 (engage only): actual pose known AND both chains already agree.
        if run_baseline:
            if self._actual_pose is None:
                return False, "chains_disagree", "actual pose unknown", None, None
            agree = max(
                abs(self._last_published[i] - self._actual_pose[i]) for i in range(n)
            )
            if agree > self.anchor_agree_tol:
                return (False, "chains_disagree",
                        f"max|cmd-actual|={agree:.4f} > {self.anchor_agree_tol:.4f}",
                        None, None)
        q_anchor = list(self._last_published)
        # G5: leader quasi-still (shared fail-closed util; reuse chase params).
        if not leader_quasi_still(
            self._gello_history,
            self.resume_chase_still_window_s,
            self.resume_chase_still_speed,
        ):
            return False, "leader_moving", "leader not demonstrably still", None, None
        # G6: EEF leader filter converged (positive proof). One filter step here
        # advances _euro_lead exactly as a tick would; the value is the anchor's
        # q_lead_f.
        q_lead_f = [self._euro_lead[i](self._raw_target[i]) for i in range(n)]
        settle = max(abs(q_lead_f[i] - self._raw_target[i]) for i in range(n))
        if settle >= self.filter_settled_tol:
            return (False, "filter_not_settled",
                    f"max|q_lead_f-raw|={settle:.5f} >= {self.filter_settled_tol:.5f}",
                    None, None)
        # G7: kinematics self-test.
        if not self._eef_selftest(q_anchor):
            return False, "kinematics_selftest", "IK(FK(q))!=q round-trip failed", None, None
        # G8: not singular.
        sig = float(uk.sigma_min(np.asarray(q_anchor, dtype=float), self.char_length))
        if sig <= self.sigma_warn:
            return (False, "singular_anchor",
                    f"sigma_min={sig:.4f} <= sigma_warn={self.sigma_warn:.4f}",
                    None, None)
        # G9: keepout geometry.
        if not uk.keepout_ok(np.asarray(q_anchor, dtype=float), self._eef.keepout):
            return False, "keepout", "anchor pose violates keepout", None, None
        return True, "ok", "", q_anchor, q_lead_f

    def _on_eef_engage(self, request, response):
        """Engage EEF delta control (gates G1..G9). Robot does not move on any
        refusal. On success the anchor is latched (zero-jump)."""
        ok, key, detail, q_anchor, q_lead_f = self._run_eef_gates(run_baseline=True)
        if not ok:
            response.success = False
            response.message = f"eef_engage REFUSED [{key}] — {detail}. Robot did not move."
            self.get_logger().warn(response.message)
            return response
        summary = self._eef.engage(q_anchor, q_lead_f)
        # DO NOT touch _seed_time (§3.5). Anchor is the command chain, not actual.
        self._eef_state = "ENGAGED"
        self._hold_since = None
        self._tick_overruns = 0
        self._eef_auto_reason = None
        p = summary["T_r_anchor"][:3, 3]
        response.success = True
        response.message = (
            f"ENGAGED — anchor p_r0=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
            f"branch={summary['branch0']} sigma_min={summary['sigma_min']:.3f} "
            f"pos_scale={self.pos_scale:.2f}. EEF delta control live."
        )
        self.get_logger().warn(response.message)
        return response

    def _on_eef_disengage(self, request, response):
        """Release EEF control — ALWAYS immediate, no gate. Same path as ~/pause:
        pause + void anchor; the next tick (<=1 period) stops publishing."""
        self._paused = True
        if self._eef is not None:
            self._eef.disengage()
        self._eef_state = "DISENGAGED"
        self._hold_since = None
        self._eef_auto_reason = None
        response.success = True
        response.message = "DISENGAGED — anchor discarded; robot holds (immediate)."
        self.get_logger().warn("EEF DISENGAGED by operator (robot holds pose).")
        return response

    def _on_eef_reclutch(self, request, response):
        """Re-anchor at the current leader/command pose (zero delta) WITHOUT
        going through pause. Re-checks G2,G5,G6,G7,G8,G9."""
        if self._eef_state != "ENGAGED":
            response.success = False
            response.message = "eef_reclutch REFUSED — not ENGAGED (call ~/eef_engage first)."
            self.get_logger().warn(response.message)
            return response
        ok, key, detail, q_anchor, q_lead_f = self._run_eef_gates(run_baseline=False)
        if not ok:
            response.success = False
            response.message = f"eef_reclutch REFUSED [{key}] — {detail}. Robot holds."
            self.get_logger().warn(response.message)
            return response
        # Anchor T_r at the CURRENT command pose so the delta restarts at zero.
        self._eef.reclutch(q_lead_now=q_lead_f, q_cmd_now=self._last_published)
        self._eef_state = "ENGAGED"
        self._hold_since = None
        self._tick_overruns = 0
        self._eef_auto_reason = None
        response.success = True
        response.message = "RE-CLUTCHED — anchor reset at current pose (zero delta)."
        self.get_logger().warn(response.message)
        return response

    def _on_eef_to_joint(self, request, response):
        """Return to joint mode: pause, discard anchor, go to BOOTSTRAP (joint
        passthrough). The operator then physically re-aligns GELLO and calls
        ~/resume (or ~/resume_chase). Reports the per-joint gap to close."""
        self._paused = True
        if self._eef is not None:
            self._eef.disengage()
        self._eef_state = "BOOTSTRAP"
        self._hold_since = None
        self._eef_auto_reason = None
        gap = self._eef_joint_gap()
        if gap is not None:
            per = ", ".join(
                f"{UR_JOINT_SHORT[i]}={gap[i]:.3f}" for i in range(len(gap))
            )
            gap_str = f"per-joint gap (leader vs arm): {per}"
        else:
            gap_str = "per-joint gap unavailable (missing leader/arm pose)"
        response.success = True
        response.message = (
            "EEF->JOINT — paused, anchor discarded (state=BOOTSTRAP, joint "
            "passthrough). Physically align the GELLO leader to the robot pose, "
            f"then call ~/resume or ~/resume_chase. {gap_str}"
        )
        self.get_logger().warn(response.message)
        return response

    def _eef_joint_gap(self):
        """Per-joint circular gap (rad) between the live leader and the arm's
        actual pose — what ~/resume must see within resume_align_tol. None if
        either pose is missing."""
        if self._raw_target is None or self._actual_pose is None:
            return None
        return [
            circular_dist(self._raw_target[i], self._actual_pose[i])
            for i in range(len(UR_JOINT_ORDER))
        ]

    def _on_eef_state_timer(self) -> None:
        """Publish the ~/eef/state JSON diagnostic (eef mode only, read-only)."""
        info = self._eef_info or {}
        gap = self._eef_joint_gap()
        payload = {
            "mode": "eef",
            "state": self._eef_state,
            "reject_reason": info.get("reject_reason"),
            "auto_reason": getattr(self, "_eef_auto_reason", None),
            "sigma_min": info.get("sigma_min"),
            "gamma": info.get("gamma"),
            "ls_scale": info.get("ls_scale"),
            "ik_residual": info.get("ik_residual"),
            "lag_pos": info.get("lag_pos"),
            "lag_rot": info.get("lag_rot"),
            "excursion_m": info.get("excursion"),
            "branch_id": info.get("branch_id"),
            "n_ik_solutions": info.get("n_ik_solutions"),
            "pos_scale": self.pos_scale,
            "joint_gap": gap,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self._eef_state_pub.publish(msg)

    def _mat_to_posestamped(self, T) -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "base_link"
        ps.pose.position.x = float(T[0, 3])
        ps.pose.position.y = float(T[1, 3])
        ps.pose.position.z = float(T[2, 3])
        q = self._ur_kin.mat_to_quat_xyzw(T[:3, :3])
        ps.pose.orientation.x = float(q[0])
        ps.pose.orientation.y = float(q[1])
        ps.pose.orientation.z = float(q[2])
        ps.pose.orientation.w = float(q[3])
        return ps

    def _publish_eef_poses(self) -> None:
        """Publish leader / desired / commanded PoseStamped diagnostics (base
        frame). Called decimated from the tick, ENGAGED only."""
        info = self._eef_info
        if info is None or self._eef_leader_q is None:
            return
        try:
            np = self._np
            T_g = self._ur_kin.fk(np.asarray(self._eef_leader_q, dtype=float)) @ self._eef.T_tool_L
            self._eef_leader_pub.publish(self._mat_to_posestamped(T_g))
            if "T_des" in info:
                self._eef_desired_pub.publish(self._mat_to_posestamped(info["T_des"]))
            if "T_cmd" in info:
                self._eef_commanded_pub.publish(self._mat_to_posestamped(info["T_cmd"]))
        except Exception as exc:  # noqa: BLE001 diagnostics must never break the tick
            self.get_logger().warn(
                f"EEF pose publish skipped: {exc}",
                throttle_duration_sec=_WARN_THROTTLE_S,
            )

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
