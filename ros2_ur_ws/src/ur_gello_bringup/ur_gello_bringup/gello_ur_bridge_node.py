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
* STALENESS WATCHDOGS (TWO, one per input stream): if GELLO input stops
  (unplugged, crashed, driver hang) the node STOPS publishing rather than
  repeating the last command forever. The SAME applies to the robot's own
  ``/joint_states``: if External Control drops, the last actual pose freezes at
  a plausible-looking value while the arm may be moved by hand or by the
  pendant, and since that pose seeds every (re)start of the command chain, a
  later re-seed would command a jump. Both watchdogs stop publishing, force a
  re-seed on recovery, and (in eef mode) auto-disengage with a distinct reason.
  Not streaming is the fail-safe state for a position controller.
* EEF (3D-PEN) MODE: with ``control_mode:=eef`` the leader and the robot live in
  deliberately DIFFERENT joint configurations and only EEF deltas are
  transferred; the arm must NEVER mirror the leader's joint shape. Therefore
  the pre-engage state is ``HOLD`` (arm frozen), not a joint passthrough, and
  the recovery from any fault is ``~/eef_resume`` (re-seed from the actual pose
  into HOLD, no alignment gate) followed by ``~/eef_engage``. Joint passthrough
  is entered ONLY via the alignment-gated ``~/resume`` / ``~/resume_chase`` /
  ``~/eef_to_joint``.
* JOINT_DELTA (START-ANCHORED) MODE: with ``control_mode:=joint_delta`` the arm
  no longer MIRRORS the leader's absolute joints; it moves by how much the
  leader has moved SINCE an anchor latched at ``~/joint_delta_engage``:
  ``q_cmd = q_robot_anchor + gain * (accumulated leader travel)``. The
  pre-engage state is the ordinary joint passthrough (unlike eef's HOLD),
  because joint_delta boots through the SAME alignment-gated handshake as joint
  mode, so the leader and the arm do agree until the operator engages. The mode
  adds one state the other two do not have — ``CLUTCHED`` (the "mouse lift":
  output and accumulator frozen, ``_paused`` untouched, leader free to be
  repositioned) — and it never reuses ``~/resume``'s alignment gate to authorize
  delta streaming, because after any drift that gate is unsatisfiable while
  protecting nothing (see docs/ros2/GELLO_UR7E_JOINT_DELTA_MODE.md).
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
    wrap_to_pi,
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
        # ROBOT-SIDE staleness: max age (s) of the last /joint_states sample
        # before the bridge treats the robot's ACTUAL pose as unknown and stops
        # publishing (see the watchdog in _on_timer). Deliberately a SEPARATE
        # parameter from staleness_timeout_s: the two streams have different
        # cadences and different failure modes — GELLO arrives at ~30 Hz over
        # USB (unplug / driver hang), while joint_state_broadcaster runs at the
        # controller rate (~500 Hz on a real UR) and stops when the External
        # Control program drops or the arm e-stops. One knob per stream lets the
        # robot-side watchdog be tightened without loosening the leader one (or
        # vice versa). Same 0.5 s default, so the documented fail-safe matrix
        # holds out of the box.
        self.actual_staleness_timeout_s = float(
            self.declare_parameter("actual_staleness_timeout_s", 0.5).value
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
        # --- Per-tick EEF-stage time budget: TWO TIERS ---------------------
        # The hazard being guarded is "the EEF stage takes longer than the
        # publish period", because THAT is what turns into timer overrun ->
        # uneven command spacing -> UR protective stop. A budget well below the
        # period is a useful PERFORMANCE target but is not, by itself, a safety
        # threshold — and treating it as one made the watchdog nuisance-trip:
        # measured EefDeltaController.step() on the shipped config runs ~282 us
        # zero-delta but ~2054 us mean / 2249 us p99 / 2512 us max while the
        # leader MOVES, i.e. 99.8% of moving ticks exceed the shipped 1000 us,
        # so the arm auto-disengaged within ~20 ms of any real motion and dumped
        # the operator into the recovery path.
        #
        # SOFT budget (tick_budget_us): performance target. Exceeding it only
        # WARNS (throttled) and is counted + published on ~/eef/state, so the
        # optimization work stays visible instead of being hidden by a raised
        # limit. It NEVER disengages. Default stays 1000 us on purpose.
        self.tick_budget_us = float(
            self.declare_parameter("tick_budget_us", 1000.0).value
        )
        # HARD budget (tick_hard_budget_us): the fail-closed threshold, in the
        # unit that actually matters — a fraction of the publish period.
        # 0.0 (default) => derive as 80% of the period (3200 us at 250 Hz),
        # which is above the measured moving-worst-case (2512 us) yet still
        # catches a stage that genuinely cannot keep up with the control loop.
        # A hard overrun is a PERFORMANCE event, not a safety fault: the tick
        # DEGRADES (holds the arm for that one tick, anchor retained, still
        # ENGAGED and un-paused) and feeds a leaky bucket. Only SUSTAINED
        # slowness — the bucket reaching tick_overrun_limit ticks — fails closed.
        # A transient CPU spike (e.g. the analytic-IK halving loop firing near a
        # singularity for a few ticks) can no longer tear down a healthy session.
        self.tick_hard_budget_us = float(
            self.declare_parameter("tick_hard_budget_us", 0.0).value
        )
        # Leaky-bucket size: ticks of SUSTAINED over-hard before fail-closed.
        # 250 @ 250 Hz ~= 1.0 s of unbroken slowness. Was 5 (consecutive), which
        # a ~20 ms compute hiccup reached and nuisance-disengaged on real HW.
        self.tick_overrun_limit = int(
            self.declare_parameter("tick_overrun_limit", 250).value
        )
        # Bucket drain per healthy tick. 1.0 forgives any non-monotonic spike
        # pattern (recommended); < 1.0 also trips chronic intermittent slowness.
        self.tick_overrun_leak = float(
            self.declare_parameter("tick_overrun_leak", 1.0).value
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

        # --- JOINT_DELTA-mode parameters (declared ALWAYS; consumed only in
        # joint_delta mode, exactly like the eef block above so the overlay yaml
        # config/ur7e_gello_joint_delta.yaml can drive them 1:1). Every one is
        # jd_-prefixed: the GENERIC gate knobs this mode also needs
        # (anchor_agree_tol, filter_settled_tol, hold_latch_s, tick_budget_us,
        # tick_overrun_limit, staleness_timeout_s, resume_chase_still_*,
        # max_step_rad, soft_start_s) are REUSED from the blocks above rather
        # than duplicated — only one control mode is ever live in a process, and
        # duplicating five tolerances would double the tuning surface for no
        # safety gain.
        # Leader->robot joint gain. 0.0 = the robot never moves (the P-stage of
        # the staged bring-up); 1.0 = 1:1 with the leader's travel.
        self.jd_gain = float(self.declare_parameter("jd_gain", 1.0).value)
        # Shrink of ur_kin.JOINT_LIMITS on both sides (rad), same semantics as
        # the eef limit_margin_rad but a SEPARATE knob so the two modes' cages
        # can be tuned independently.
        self.jd_limit_margin_rad = float(
            self.declare_parameter("jd_limit_margin_rad", 0.05).value
        )
        # Per-joint band (rad) around q_robot_anchor the command may never leave.
        # This is the replacement for the joint-mode property "q_cmd == q_leader
        # and the leader is mechanically bounded, so the command is bounded too",
        # which start-anchored delta control structurally loses.
        self.jd_max_excursion_rad = float(
            self.declare_parameter("jd_max_excursion_rad", 1.0).value
        )
        # A leader increment above this (rad) is treated as a teleport, not a
        # human gesture. Tested PRIMARILY on the RAW leader stream at the SOURCE
        # (~30 Hz GELLO) cadence in _on_joint_state — 0.35 rad/sample is ~10.5
        # rad/s of leader speed — and only as a BACKSTOP inside the controller,
        # where the signal is 1-Euro filtered and a low-pass has already smeared
        # any raw teleport below the cap.
        self.jd_leader_jump_max_rad = float(
            self.declare_parameter("jd_leader_jump_max_rad", 0.35).value
        )
        # How long (s) the joint_delta stage HOLDS and re-syncs its accumulator
        # reference after a RAW leader teleport, i.e. how long the 1-Euro leader
        # filter is given to digest the glitch. Must comfortably exceed the
        # filter's settling time; 0.30 s is ~9 GELLO samples / 75 publish ticks.
        # Too long only costs lag (the arm holds); too short lets the filter's
        # tail be accumulated as genuine leader travel.
        self.jd_resync_hold_s = float(
            self.declare_parameter("jd_resync_hold_s", 0.30).value
        )
        # Per-tick increment deadband (rad). OFF (0.0) on purpose: the increments
        # telescope, so accumulation is drift-free by construction and any
        # nonlinearity here CREATES bias instead of removing it.
        self.jd_delta_deadband_rad = float(
            self.declare_parameter("jd_delta_deadband_rad", 0.0).value
        )
        # ~/joint_delta/state publish rate (Hz).
        self.jd_state_rate_hz = float(
            self.declare_parameter("jd_state_rate_hz", 10.0).value
        )
        # When True, ~/joint_delta_start may arm a bridge that has NEVER streamed.
        # Set ONLY by the switch_only no-motion bring-up, where gello_move_to_start
        # performs the STRICT switch to forward_position_controller BEFORE calling
        # this service, so the "controller must already be active" invariant that
        # _has_streamed proxies is guaranteed by call ordering (identical to how
        # ~/eef_resume ships with no such gate). Default False preserves the
        # manual-recovery gate for every existing caller (joint/eef bridges never
        # set it). See gate (a2) in _on_jd_start.
        self.jd_start_allow_unstreamed = bool(
            self.declare_parameter("jd_start_allow_unstreamed", False).value
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
        # Monotonic timestamp (s) of the last /joint_states sample that produced
        # _actual_pose. WITHOUT this the pose silently FREEZES at its last value
        # when External Control drops, and a later re-seed would seed from a pose
        # the arm no longer holds — the entire zero-jump argument rests on
        # _actual_pose being LIVE, so it needs its own watchdog (see _on_timer).
        self._actual_pose_time: float | None = None
        # Set when the robot-side (/joint_states) watchdog trips, so recovery is
        # logged once and always goes through a fresh seed + soft-start.
        self._was_actual_stale = False
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
        # STATE MACHINE (one string, four values — see _eef_hold_active()):
        #   "HOLD"            pre/post-engage 3D-PEN state: the arm HOLDS its
        #                     last commanded pose and does NOT mirror the
        #                     leader's joints. This is what eef mode boots into
        #                     and what ~/eef_resume returns to.
        #   "ENGAGED"         EEF delta control live (anchor latched).
        #   "DISENGAGED"      anchor void; == _paused (bridge publishes nothing).
        #   "JOINT_BOOTSTRAP" joint passthrough: the arm mirrors the leader's
        #                     joints, i.e. the LEGACY bootstrap. Entered ONLY
        #                     from the three alignment-gated services
        #                     (~/resume, ~/resume_chase, ~/eef_to_joint).
        # The pipeline's hold-vs-passthrough choice is DERIVED from this single
        # variable (_eef_hold_active) rather than tracked by a parallel boolean:
        # a separate flag would have to be kept in sync at ~7 mutation sites, and
        # that kind of drift is exactly what produced the "eef mode still mirrors
        # the leader's joints" hazard in the first place. One variable, one
        # invariant, checked in one place.
        # In joint mode this stays "JOINT_BOOTSTRAP" (the honest description of
        # joint mode) and _eef_hold_active() is False by control_mode anyway.
        self._eef_state = "JOINT_BOOTSTRAP"
        # EefDeltaController instance and the SHARED leader 1-Euro filter bank +
        # the lazily-imported ur_kin / numpy handles. All None in joint mode.
        #
        # _euro_lead / _q_lead_f are the LEADER-side filter machinery and are
        # shared verbatim by eef mode and joint_delta mode (the two are mutually
        # exclusive, so exactly one of them ever constructs the bank). Sharing —
        # rather than adding a parallel _euro_jd bank — reuses the seed,
        # cache-invalidation and cache-freshness discipline below as one hardened
        # implementation instead of two.
        self._eef = None
        self._euro_lead: list[OneEuro] | None = None
        # CACHED output of the leader 1-Euro bank, advanced EXACTLY ONCE PER
        # PUBLISH TICK in eef / joint_delta mode (see _on_timer). Both the
        # ENGAGED delta stage and the engage/reclutch gate G6 read this cache, so
        # (a) the filter is never double-stepped inside one tick and (b) G6
        # measures the lag of a filter that has genuinely been running against
        # the live leader. The companion timestamp makes the cache impossible to
        # use stale.
        self._q_lead_f: list[float] | None = None
        self._q_lead_f_time: float | None = None
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
        # Leaky-bucket level of HARD time-budget overruns (fail-closed when it
        # reaches tick_overrun_limit; drains by tick_overrun_leak each good tick).
        self._tick_overruns = 0.0
        # Cumulative SOFT-budget overruns + last measured stage time (us), for
        # ~/eef/state. Diagnostic only — they never gate anything.
        self._tick_soft_overruns = 0
        self._tick_us_last: float | None = None
        # Publish-tick counter for decimating the PoseStamped diagnostics.
        self._eef_tick = 0
        self._eef_pose_decim = max(1, int(round(self.publish_rate_hz / 30.0)))

        # Max age (s) of the cached leader-filter output for it to still be
        # usable by gate G6. It only has to prove "the filter is being stepped by
        # the live tick", so it is a few publish periods (floored at 50 ms for
        # slow publish rates and executor jitter) — NOT staleness_timeout_s,
        # which is 100+ ticks and would happily accept a filter frozen by a pause.
        self._q_lead_f_max_age_s = max(4.0 / self.publish_rate_hz, 0.05)
        # Effective HARD tick budget (us): explicit parameter, else 80% of the
        # publish period. Recomputed by the parameter callback below.
        self._tick_hard_us = self._effective_hard_budget_us()
        # RUNTIME RETUNE of the tick-watchdog knobs only (see _on_set_parameters).
        # Registered after the declarations above so it never fires mid-declare.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # --- JOINT_DELTA state (never used in joint / eef mode) --------------
        # STATE MACHINE (one string, four values):
        #   "JOINT_BOOTSTRAP" pre-engage JOINT PASSTHROUGH — the arm mirrors the
        #                     leader's joints, i.e. exactly joint mode. This is
        #                     what joint_delta boots into (legal, and NOT the eef
        #                     3D-pen HOLD, because joint_delta comes up through
        #                     the SAME alignment-gated move-to-start handshake as
        #                     joint mode) and what any ~/resume returns to.
        #   "ENGAGED"         start-anchored delta control live (anchor latched).
        #   "CLUTCHED"        the MOUSE LIFT: the controller's output AND its
        #                     accumulator are frozen, the anchor is retained, and
        #                     _paused is deliberately UNTOUCHED so the arm keeps
        #                     being commanded its frozen pose while the operator
        #                     repositions the leader. Completed by
        #                     ~/joint_delta_reclutch. This state is why the mode
        #                     cannot reuse eef's release path: ~/eef_disengage
        #                     sets _paused, and the way back (~/resume) is gated
        #                     on a leader/arm alignment that delta mode
        #                     deliberately breaks.
        #   "DISENGAGED"      anchor void; == _paused (bridge publishes nothing).
        # Separate _jd_* bookkeeping (rather than reusing _hold_since /
        # _tick_overruns / _eef_auto_reason) so no eef field is ever written in
        # joint_delta mode. The LEADER filter machinery (_euro_lead / _q_lead_f)
        # IS shared — see its comment above.
        self._jd_state = "JOINT_BOOTSTRAP"
        self._jd = None                     # JointDeltaController
        self._jd_info: dict | None = None   # last step() info dict
        self._jd_hold_since: float | None = None
        self._jd_auto_reason: str | None = None
        self._jd_tick_overruns = 0.0
        self._jd_tick_soft_overruns = 0
        self._jd_tick_us_last: float | None = None
        # RAW-domain leader teleport detector (joint_delta only). _jd_raw_prev is
        # the previous UNWRAPPED raw leader sample at the SOURCE cadence, and
        # _jd_resync_until is the monotonic deadline of the hold-and-re-sync
        # window opened when a teleport is seen. See _on_joint_state.
        self._jd_raw_prev: list[float] | None = None
        self._jd_resync_until: float | None = None
        self._jd_raw_jumps = 0
        # Set the first time the bridge ever PUBLISHES a command. Because
        # _on_timer returns immediately while paused, this latch is a proof that
        # the bridge has been released at least once -- i.e. that the
        # move-to-start handshake completed its STRICT switch to
        # forward_position_controller. ~/joint_delta_start requires it.
        self._has_streamed = False

        if self.control_mode == "eef":
            # 3D PEN: eef mode BOOTS INTO HOLD. Before this the pre-engage state
            # was a joint passthrough, so `control_mode:=eef` had the arm mirror
            # the leader's joint shape from the very first tick until someone
            # called ~/eef_engage — the opposite of the intended "leader and
            # robot permanently live in different joint configurations".
            self._eef_state = "HOLD"
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
            # EEF-NATIVE RE-ARM: the recovery path out of every fault route
            # (leader_stale / hold_latched / tick_budget / exception), all of
            # which end in _eef_autodisengage -> _paused=True. ~/resume and
            # ~/resume_chase are the JOINT-mode recovery and are left untouched.
            self._eef_resume_srv = self.create_service(
                Trigger, "~/eef_resume", self._on_eef_resume
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

        # --- JOINT_DELTA mode setup --------------------------------------
        if self.control_mode == "joint_delta":
            # LAZY IMPORT for symmetry with the eef branch above: the joint
            # path's import graph stays literally unchanged. (joint_delta is
            # stdlib-only — no numpy, no ur_kin — so this costs nothing, unlike
            # eef's analytic-IK stack; the symmetry is the point.)
            from ur_gello_bringup.joint_delta import JointDeltaController
            self._jd = JointDeltaController({
                "gain": self.jd_gain,
                "limit_margin_rad": self.jd_limit_margin_rad,
                "max_excursion_rad": self.jd_max_excursion_rad,
                "leader_jump_max_rad": self.jd_leader_jump_max_rad,
                "delta_deadband_rad": self.jd_delta_deadband_rad,
                "n_joints": len(UR_JOINT_ORDER),
            })
            # SHARED leader 1-Euro bank (see _euro_lead's comment): fed and
            # seeded from _raw_target by the very same ingest/seed/tick code the
            # eef path uses, so gate G6 measures a genuinely running filter here
            # too. Only one of the two modes ever constructs it.
            _dt = 1.0 / self.publish_rate_hz
            self._euro_lead = [
                OneEuro(_dt, self.one_euro_min_cutoff, self.one_euro_beta,
                        self.one_euro_d_cutoff)
                for _ in range(len(UR_JOINT_ORDER))
            ]
            # Clutch services (fail-closed Trigger, mirror the eef skeleton).
            self._jd_engage_srv = self.create_service(
                Trigger, "~/joint_delta_engage", self._on_jd_engage
            )
            self._jd_clutch_srv = self.create_service(
                Trigger, "~/joint_delta_clutch", self._on_jd_clutch
            )
            self._jd_reclutch_srv = self.create_service(
                Trigger, "~/joint_delta_reclutch", self._on_jd_reclutch
            )
            self._jd_disengage_srv = self.create_service(
                Trigger, "~/joint_delta_disengage", self._on_jd_disengage
            )
            self._jd_to_joint_srv = self.create_service(
                Trigger, "~/joint_delta_to_joint", self._on_jd_to_joint
            )
            # JOINT_DELTA-NATIVE ARM/RE-ARM (chase-free): the ONE entry point
            # that both (a) releases a PAUSED bridge straight into ENGAGED delta
            # control with a zero delta, and (b) is the recovery out of every
            # fault route (all of which end in _jd_autodisengage -> _paused).
            # ~/resume / ~/resume_chase stay the JOINT-mode recovery and are
            # untouched; after real drift their alignment gates are unsatisfiable
            # in this mode, which is exactly why a native path must exist.
            self._jd_start_srv = self.create_service(
                Trigger, "~/joint_delta_start", self._on_jd_start
            )
            # Diagnostic topic (JSON). No PoseStamped siblings: joint_delta has
            # no Cartesian quantity to publish.
            self._jd_state_pub = self.create_publisher(
                String, "~/joint_delta/state", 10
            )
            _jd_hz = self.jd_state_rate_hz if self.jd_state_rate_hz > 0.0 else 10.0
            self._jd_state_timer = self.create_timer(
                1.0 / _jd_hz, self._on_jd_state_timer
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
            f"actual_staleness_timeout_s={self.actual_staleness_timeout_s} "
            f"eef_state={self._eef_state} "
            f"hold_when_not_engaged={self._eef_hold_active()} "
            f"jd_state={self._jd_state} "
            f"jd_gain={self.jd_gain} "
            f"jd_max_excursion_rad={self.jd_max_excursion_rad} "
            f"jd_limit_margin_rad={self.jd_limit_margin_rad} "
            f"jd_leader_jump_max_rad={self.jd_leader_jump_max_rad} "
            f"jd_resync_hold_s={self.jd_resync_hold_s} "
            f"publish_rate_hz={self.publish_rate_hz}"
        )
        self.get_logger().info(
            f"UR joint order: {UR_JOINT_ORDER}; seeding from actual "
            f"joint state on {self._js_topic}"
        )

    # ---------------------------------------------------------------------
    def _effective_hard_budget_us(self) -> float:
        """Fail-closed tick budget (us): explicit param, else 80% of the period.

        Deriving it from ``publish_rate_hz`` keeps the threshold expressed in
        the only unit that maps to the actual hazard (missing the control
        deadline), so it stays correct if the publish rate is retuned.
        """
        if self.tick_hard_budget_us > 0.0:
            return self.tick_hard_budget_us
        return 0.8 * (1e6 / self.publish_rate_hz)

    def _on_set_parameters(self, params):
        """Allow the TICK-WATCHDOG knobs to be retuned at RUNTIME.

        Every other parameter here is read once at construction, which is fine
        for values that are chosen before the arm is powered — but the tick
        budget can only be characterised WITH the robot running, and it is not
        exposed as a launch argument, so during bring-up the only way to test a
        value was to stop the launch and re-run the bridge by hand with
        ``-p tick_budget_us:=...``. These three are cheap scalars read fresh
        every tick, so accepting them live is safe and removes that dead end:

            ros2 param set /gello_ur_bridge tick_hard_budget_us 3500.0

        Other parameters are ACCEPTED (so a blanket ``ros2 param set`` does not
        error) but are NOT applied live; they still require a restart.
        """
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            # --- EEF live-tunable controller knobs (A안: the GUI commits
            #     pos_scale at anchor-reset moments via SetParameters). These
            #     scalars are re-read fresh every tick by EefDeltaController.step,
            #     so applying them live takes effect on the next control cycle.
            #     Guarded by `self._eef is not None`: in control_mode:=joint /
            #     joint_delta self._eef is None, so this is a pure no-op there and
            #     joint teleop is byte-for-byte unaffected. We still mirror the
            #     value onto the node attribute so ~/eef/state stays truthful.
            if p.name == "pos_scale":
                self.pos_scale = float(p.value)
                if self._eef is not None:
                    self._eef.pos_scale = float(p.value)
                self.get_logger().info(
                    f"pos_scale set live: {self.pos_scale:.3f} "
                    f"(controller={'yes' if self._eef is not None else 'n/a'})"
                )
                continue
            elif p.name == "v_max":
                self.eef_v_max = float(p.value)
                if self._eef is not None:
                    self._eef.v_max = float(p.value)
                self.get_logger().info(f"v_max set live: {self.eef_v_max:.4f}")
                continue
            elif p.name == "w_max":
                self.eef_w_max = float(p.value)
                if self._eef is not None:
                    self._eef.w_max = float(p.value)
                self.get_logger().info(f"w_max set live: {self.eef_w_max:.4f}")
                continue
            if p.name == "tick_budget_us":
                self.tick_budget_us = float(p.value)
            elif p.name == "tick_hard_budget_us":
                self.tick_hard_budget_us = float(p.value)
                self._tick_hard_us = self._effective_hard_budget_us()
            elif p.name == "tick_overrun_limit":
                self.tick_overrun_limit = int(p.value)
            elif p.name == "tick_overrun_leak":
                self.tick_overrun_leak = float(p.value)
            else:
                continue
            self.get_logger().warn(
                f"tick watchdog retuned live: {p.name}={p.value} "
                f"(soft={self.tick_budget_us:.0f}us hard={self._tick_hard_us:.0f}us "
                f"limit={self.tick_overrun_limit})"
            )
        return SetParametersResult(successful=True)

    # ---------------------------------------------------------------------
    def _eef_hold_active(self) -> bool:
        """DERIVED pipeline flag: hold the last command instead of mirroring.

        Single source of truth for the "3D pen" behaviour: in eef mode every
        state EXCEPT ``JOINT_BOOTSTRAP`` holds. (In ``ENGAGED`` the pipeline's
        engaged branch wins, so this only actually decides ``HOLD`` vs
        ``JOINT_BOOTSTRAP``; ``DISENGAGED`` implies ``_paused``, i.e. nothing is
        published at all.) In joint mode it is ALWAYS False, and the pipeline
        additionally skips the whole eef stage — joint teleop is untouched.
        """
        return self.control_mode == "eef" and self._eef_state != "JOINT_BOOTSTRAP"

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
        # RAW-DOMAIN LEADER TELEPORT DETECTOR (joint_delta only; _jd is None in
        # joint and eef mode, so those paths evaluate one `is not None` and
        # nothing else).
        #
        # WHY IT HAS TO BE HERE AND NOT IN THE CONTROLLER. The controller is fed
        # the 1-EURO-FILTERED leader at the 250 Hz publish tick, and a low-pass
        # SMEARS a raw teleport across many ticks. Measured with the shipped
        # gains (min_cutoff 1.0, beta 2.0, publish 250 Hz, cap 0.35): raw
        # single-sample glitches of 0.3 / 0.5 / 0.8 / 1.0 / 1.2 rad produce ZERO
        # filtered increments above the cap and are accumulated IN FULL — the
        # wrist would sweep up to ~57 deg with the diagnostics reporting a
        # perfectly healthy session. And a glitch big enough to trip the cap
        # (>= ~2 rad) trips it only for the first 2-5 ticks of the smear, after
        # which the sub-threshold tail IS accumulated; because that discard is
        # asymmetric (out-stroke only) a fully transient +2.0 rad glitch left the
        # arm 1.16 rad (-67 deg) off its anchor PERMANENTLY.
        #
        # Detecting on the raw stream at the source cadence fixes both: the
        # window opens BEFORE any tick can consume the glitched sample (callbacks
        # are serialized on one executor), and for its duration the stage calls
        # JointDeltaController.absorb() every tick, which freezes the command and
        # slides q_lead_prev along with the settling filter — so neither the head
        # nor the tail of the smear is ever accumulated.
        if self._jd is not None:
            if self._jd_raw_prev is not None:
                raw_jump = max(
                    abs(wrap_to_pi(unwrapped[i] - self._jd_raw_prev[i]))
                    for i in range(len(UR_JOINT_ORDER))
                )
                if raw_jump > self.jd_leader_jump_max_rad:
                    self._jd_raw_jumps += 1
                    self._jd_resync_until = now + self.jd_resync_hold_s
                    # RE-SEED THE LEADER FILTER ONTO THE NEW READING. Low-passing
                    # ACROSS a discontinuity is meaningless, and the exponential
                    # TAIL it produces is exactly what would still be accumulated
                    # as genuine leader travel once the hold window expires (a
                    # timed window alone left ~0.02 rad of residual offset per
                    # glitch; with the re-seed the residual is bit-exactly zero).
                    # Guarded by `self._jd is not None`, so the bank is never
                    # re-seeded in eef mode.
                    if self._euro_lead is not None:
                        for i in range(len(UR_JOINT_ORDER)):
                            self._euro_lead[i].seed(unwrapped[i])
                        # The bank was just RESET, so any cached output predates
                        # it — same invalidation rule as the seed branch.
                        self._q_lead_f = None
                        self._q_lead_f_time = None
                    self.get_logger().warn(
                        f"LEADER TELEPORT on the RAW GELLO stream: "
                        f"{raw_jump:.3f} rad in one sample > "
                        f"{self.jd_leader_jump_max_rad:.3f} — joint_delta HOLDS "
                        f"and re-syncs for {self.jd_resync_hold_s:.2f}s "
                        f"(total {self._jd_raw_jumps}). The arm does not move.",
                        throttle_duration_sec=_WARN_THROTTLE_S,
                    )
            self._jd_raw_prev = list(unwrapped)
        self._raw_target = unwrapped
        self._last_good_msg_time = now
        # Record the UNWRAPPED pose for the resume_chase quasi-still gate.
        # circular_dist inside leader_quasi_still handles any 2*pi re-anchor
        # discontinuity in this history harmlessly (measures the short-way gap).
        self._gello_history.append((now, list(unwrapped)))

    # ---------------------------------------------------------------------
    def _on_actual_joint_state(self, msg: JointState) -> None:
        """Track the robot's ACTUAL current pose (reordered BY NAME).

        Published by joint_state_broadcaster (real hardware and mock). Used to
        SEED every (re)start of the command chain; after seeding the command is
        driven by the GELLO stream + slew clamp.

        The monotonic receive time is stored alongside the pose so the robot-side
        staleness watchdog in _on_timer can tell "the arm is standing still" from
        "the stream died and this pose is a fossil". time.monotonic() is used for
        the same reason as the GELLO watchdog: immune to wall-clock steps and to
        use_sim_time. Only messages that actually carry all six UR joints refresh
        the timestamp, so a partial/unrelated /joint_states publisher cannot keep
        the watchdog alive.
        """
        name_to_pos = dict(zip(msg.name, msg.position))
        if any(j not in name_to_pos for j in UR_JOINT_ORDER):
            return  # partial / unrelated joint_states; ignore
        self._actual_pose = [float(name_to_pos[j]) for j in UR_JOINT_ORDER]
        self._actual_pose_time = time.monotonic()

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
            # JOINT_DELTA: same argument, and it bites harder here. q_lead_prev
            # is the accumulator's reference; if the operator moved the leader
            # during the outage, the first post-recovery tick would fold the
            # ENTIRE outage motion into one increment. Fail closed from CLUTCHED
            # too — a clutched session's anchor is just as invalid afterwards.
            elif self.control_mode == "joint_delta" and self._jd_state in (
                "ENGAGED", "CLUTCHED"
            ):
                self._jd_autodisengage("leader_stale")
            return

        # ROBOT-SIDE STALENESS WATCHDOG (/joint_states). Symmetric to the GELLO
        # watchdog above and just as load-bearing: _actual_pose is the seed for
        # EVERY (re)start of the command chain, so the whole zero-jump argument
        # assumes it is LIVE. If External Control drops (program stopped, e-stop,
        # driver crash) the last pose FREEZES at its final value while the arm
        # may be moved by hand or by the pendant; seeding from that fossil would
        # command a jump. Fail closed: stop publishing (the controller keeps its
        # last setpoint), force a re-seed on recovery, and — in eef mode — auto-
        # disengage with a DISTINCT reason so ~/eef/state names the real cause.
        # Applies to BOTH modes; in the healthy case (joint_state_broadcaster
        # streaming) it is a compare-and-fall-through, so joint mode is unchanged.
        if self._actual_pose is not None and self._actual_pose_time is not None:
            actual_age = time.monotonic() - self._actual_pose_time
            if actual_age > self.actual_staleness_timeout_s:
                self.get_logger().warn(
                    f"Robot joint state STALE on {self._js_topic} "
                    f"(age={actual_age:.2f}s > "
                    f"{self.actual_staleness_timeout_s:.2f}s) — not publishing. "
                    "Is External Control still running?",
                    throttle_duration_sec=_WARN_THROTTLE_S,
                )
                self._filtered = None
                self._last_published = None
                self._was_actual_stale = True
                if self.control_mode == "eef" and self._eef_state == "ENGAGED":
                    self._eef_autodisengage("robot_joint_state_stale")
                elif self.control_mode == "joint_delta" and self._jd_state in (
                    "ENGAGED", "CLUTCHED"
                ):
                    self._jd_autodisengage("robot_joint_state_stale")
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
                # The filter bank was just RESET, so any cached output predates
                # it. Invalidate the cache (G6 then refuses until the tick below
                # has produced a fresh, post-seed value) — the cache must never
                # be able to certify a filter state that no longer exists.
                self._q_lead_f = None
                self._q_lead_f_time = None
            self._seed_time = time.monotonic()  # start the soft-start ramp
            if self._was_stale:
                self.get_logger().info("GELLO stream recovered; re-seeded from "
                                       "actual pose with soft-start.")
                self._was_stale = False
            if self._was_actual_stale:
                self.get_logger().info(
                    f"Robot joint state recovered on {self._js_topic}; re-seeded "
                    "from actual pose with soft-start."
                )
                self._was_actual_stale = False
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
        # LEADER FILTER, STEPPED EVERY TICK (eef mode only, ENGAGED or not).
        # _on_joint_state only calls update_input(), which advances the SPEED
        # estimate — it never advances the filter output _x_prev. Previously the
        # only caller of __call__ outside the ENGAGED stage was gate G6 itself,
        # so at engage time the filter output was still whatever it was seeded
        # with (usually many seconds old). G6 then compared that seed against the
        # live leader, which made "settled" mean "the operator has not moved the
        # leader by more than ~0.005 rad since the seed" — 0.29 deg, unsatisfiable
        # by a human hand and only ever passed by a frozen mock leader.
        # Stepping here, exactly once per tick, means the cached value is the
        # output of a filter that has been tracking the live leader continuously,
        # so G6's |q_lead_f - raw| really is the filter's residual lag.
        # joint_delta mode steps the SAME bank for the SAME reason (its gate G6
        # and its delta stage both read this cache); the bank is None in joint
        # mode, so joint teleop evaluates one extra string compare and nothing
        # else.
        if (
            self.control_mode in ("eef", "joint_delta")
            and self._euro_lead is not None
        ):
            self._q_lead_f = [
                self._euro_lead[i](self._raw_target[i])
                for i in range(len(UR_JOINT_ORDER))
            ]
            self._q_lead_f_time = time.monotonic()

        eef_command = None
        if self.control_mode == "eef" and self._eef_state == "ENGAGED":
            # EEF stage runs inside the same try/except so any overrun/exception
            # fails CLOSED to DISENGAGED rather than killing the timer callback.
            try:
                t0 = time.monotonic()
                # DEDICATED leader filter (never the joint self._euro): the eef
                # controller consumes the SMOOTHED leader joint vector — the
                # value cached by the single per-tick step above (stepping it
                # again here would double-advance the filter within one tick).
                q_lead_f = self._q_lead_f
                if q_lead_f is None:
                    # Unreachable by construction (the step above always runs in
                    # eef mode before this block, and the seed branch returns
                    # early after invalidating it) — but never feed None to the
                    # controller: fail closed through the except below instead.
                    raise RuntimeError("leader filter cache empty")
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
                # TWO-TIER time-budget watchdog (DEGRADE, then fail closed).
                elapsed_us = (time.monotonic() - t0) * 1e6
                self._tick_us_last = elapsed_us
                # Tier 1 (SOFT): performance target missed. Warn + count only.
                # This is the tier the shipped 1000 us belongs to; it keeps the
                # cost visible without turning every leader motion into an
                # auto-disengage.
                if elapsed_us > self.tick_budget_us:
                    self._tick_soft_overruns += 1
                    self.get_logger().warn(
                        f"EEF stage over SOFT budget: {elapsed_us:.0f}us > "
                        f"{self.tick_budget_us:.0f}us (hard {self._tick_hard_us:.0f}us, "
                        f"period {1e6 / self.publish_rate_hz:.0f}us). Control "
                        "continues; this is a performance signal, not a fault.",
                        throttle_duration_sec=_WARN_THROTTLE_S,
                    )
                # Tier 2 (HARD): the stage ran slow. DEGRADE GRACEFULLY — hold
                # this tick (keep the anchor, stay ENGAGED, do NOT pause) and
                # feed a leaky bucket. A transient spike (e.g. the analytic-IK
                # halving loop firing near a singularity for a few ticks) is
                # forgiven by the leak; only SUSTAINED slowness — the bucket
                # reaching tick_overrun_limit ticks — fails closed via the same
                # _eef_autodisengage as before. eef_command=None makes the
                # command pipeline freeze the arm at last_published for one tick
                # (bridge_stages: engaged + eef_command None -> zero clamp delta).
                if elapsed_us > self._tick_hard_us:
                    self._tick_overruns += 1.0
                    eef_command = None
                    if self._tick_overruns >= self.tick_overrun_limit:
                        self._eef_autodisengage(
                            f"tick_budget SUSTAINED over hard for "
                            f"{self.tick_overrun_limit}x "
                            f"(last {elapsed_us:.0f}us > hard "
                            f"{self._tick_hard_us:.0f}us)"
                        )
                else:
                    self._tick_overruns = max(
                        0.0, self._tick_overruns - self.tick_overrun_leak
                    )
            except Exception as exc:  # noqa: BLE001 fail-closed on ANY eef fault
                self._eef_autodisengage(f"exception: {exc}")
                eef_command = None

        # ---- JOINT_DELTA stage (structural twin of the EEF stage above) ----
        # Runs in CLUTCHED as well as ENGAGED: the controller returns its FROZEN
        # command while clutched, which is exactly the hold the mouse-lift needs,
        # and routing it through the same stage keeps the "one command source per
        # tick" invariant instead of adding a second hold path.
        jd_command = None
        if self.control_mode == "joint_delta" and self._jd_state in (
            "ENGAGED", "CLUTCHED"
        ):
            try:
                t0 = time.monotonic()
                q_lead_f = self._q_lead_f
                if q_lead_f is None:
                    # Unreachable by construction (the per-tick filter step above
                    # always runs in joint_delta mode before this block) — but
                    # never feed None to the controller: fail closed instead.
                    raise RuntimeError("leader filter cache empty")
                # RAW-TELEPORT RESYNC WINDOW (opened by _on_joint_state's
                # raw-domain detector). While it is open we HOLD and slide the
                # accumulator reference along with the still-settling 1-Euro
                # output instead of stepping, so neither the head nor the tail of
                # the smeared glitch is ever accumulated. absorb() returns a HOLD
                # info, so the hold-latch watchdog below still fails closed if a
                # leader keeps teleporting.
                if (
                    self._jd_resync_until is not None
                    and t0 < self._jd_resync_until
                ):
                    q_cmd, info = self._jd.absorb(q_lead_f)
                else:
                    self._jd_resync_until = None
                    # step_eff carries the soft-start-ramped clamp so the
                    # controller pre-clamps with the SAME budget the downstream
                    # clamp_stage uses; the downstream clamp is then a no-op.
                    q_cmd, info = self._jd.step(q_lead_f, step)
                self._jd_info = info
                if q_cmd is not None:
                    jd_command = [float(v) for v in q_cmd]
                # HOLD latch: a continuous HOLD longer than hold_latch_s
                # auto-disengages (anti permanent saturation). Note a saturated
                # joint limit is NOT a HOLD here — the position clamp saturates
                # smoothly and keeps streaming — so this only latches on the
                # genuine faults (BAD_INPUT / ESTOP / a leader stuck jumping).
                if info.get("state") == "HOLD":
                    if self._jd_hold_since is None:
                        self._jd_hold_since = t0
                    elif (t0 - self._jd_hold_since) > self.hold_latch_s:
                        self._jd_autodisengage("hold_latched "
                                               f"(reason={info.get('reject_reason')})")
                        jd_command = None
                else:
                    self._jd_hold_since = None
                # TWO-TIER time-budget watchdog, same tiers as the eef stage.
                elapsed_us = (time.monotonic() - t0) * 1e6
                self._jd_tick_us_last = elapsed_us
                if elapsed_us > self.tick_budget_us:
                    self._jd_tick_soft_overruns += 1
                    self.get_logger().warn(
                        f"JOINT_DELTA stage over SOFT budget: {elapsed_us:.0f}us > "
                        f"{self.tick_budget_us:.0f}us (hard {self._tick_hard_us:.0f}us, "
                        f"period {1e6 / self.publish_rate_hz:.0f}us). Control "
                        "continues; this is a performance signal, not a fault.",
                        throttle_duration_sec=_WARN_THROTTLE_S,
                    )
                # Same DEGRADE-then-sustained-teardown policy as the eef stage.
                # NOTE: JointDeltaController is an accumulator, so the first
                # healthy step() after a degrade window folds the leader motion
                # accumulated during the held ticks into ONE increment — bounded
                # by the slew / max_step_rad clamp (sub-degree for ~1 s windows),
                # which is exactly why the sustained threshold stays ~1 s.
                if elapsed_us > self._tick_hard_us:
                    self._jd_tick_overruns += 1.0
                    jd_command = None
                    if self._jd_tick_overruns >= self.tick_overrun_limit:
                        self._jd_autodisengage(
                            f"tick_budget SUSTAINED over hard for "
                            f"{self.tick_overrun_limit}x "
                            f"(last {elapsed_us:.0f}us > hard "
                            f"{self._tick_hard_us:.0f}us)"
                        )
                else:
                    self._jd_tick_overruns = max(
                        0.0, self._jd_tick_overruns - self.tick_overrun_leak
                    )
            except Exception as exc:  # noqa: BLE001 fail-closed on ANY jd fault
                self._jd_autodisengage(f"exception: {exc}")
                jd_command = None

        # FAIL-CLOSED: if the EEF / JOINT_DELTA stage auto-disengaged this tick it
        # also PAUSED the bridge. Hold the current pose exactly and stop; the next
        # tick's top-level paused check keeps it held (operator must ~/resume /
        # ~/eef_resume / ~/joint_delta_start and re-engage). Mode-agnostic — do
        # not add a per-mode branch here.
        if self._paused:
            if self._last_published is not None:
                self._publish(self._last_published)
            return

        # Mode-resolved "engaged" flag for stage 2 / 2b. The joint and eef
        # branches evaluate the IDENTICAL expression they always did (in joint
        # mode _eef_state is provably never "ENGAGED": the only writer of that
        # value is _on_eef_engage, whose service is not created outside eef mode).
        if self.control_mode == "joint_delta":
            engaged = self._jd_state in ("ENGAGED", "CLUTCHED")
        else:
            engaged = self._eef_state == "ENGAGED"

        out = command_pipeline(
            self.control_mode,
            engaged,
            self._raw_target,
            self._filtered,
            self._gated_target,
            self._euro,
            self.ema_alpha,
            self.deadband_rad,
            self._last_published,
            step,
            eef_command=eef_command,
            # 3D PEN: when not ENGAGED, HOLD instead of mirroring the leader's
            # joints (False in joint mode, and after ~/eef_to_joint).
            hold_when_not_engaged=self._eef_hold_active(),
            joint_delta_command=jd_command,
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
        # Same rule for joint_delta. Deliberately DISENGAGED, not CLUTCHED: the
        # clutch is a keep-the-anchor operator convenience, whereas ~/pause means
        # "stop, I do not know what is happening" and must void the anchor.
        elif self.control_mode == "joint_delta" and self._jd is not None:
            self._jd.disengage()
            self._jd_state = "DISENGAGED"
            self._jd_hold_since = None
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
        # EEF: ~/resume is the JOINT-mode recovery and is left exactly as it was
        # — it returns to JOINT_BOOTSTRAP (joint passthrough), which is legal
        # here precisely BECAUSE of gate (c) above: the leader has just been
        # proven to sit within resume_align_tol of the arm, so the passthrough
        # has almost no gap to close. The EEF-native recovery (no joint
        # alignment required at all) is ~/eef_resume.
        if self.control_mode == "eef":
            self._eef_state = "JOINT_BOOTSTRAP"
        # JOINT_DELTA: a resume ALWAYS lands in absolute passthrough, never back
        # into delta with a stale anchor. That is what keeps gate (c) above a
        # REAL check here — it is measured against a genuinely absolute mapping —
        # instead of a check whose precondition delta mode satisfies by
        # definition. Delta re-entry is only via ~/joint_delta_engage (from
        # streaming) or ~/joint_delta_start (from paused).
        elif self.control_mode == "joint_delta":
            self._jd_state = "JOINT_BOOTSTRAP"
            if self._jd is not None:
                self._jd.disengage()
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
        # EEF: resume_chase also returns to JOINT_BOOTSTRAP (joint passthrough),
        # unchanged — its gates (c)+(d) bound the glide it authorizes.
        if self.control_mode == "eef":
            self._eef_state = "JOINT_BOOTSTRAP"
        # JOINT_DELTA: identical rule to ~/resume — land in absolute passthrough,
        # never back into delta with a stale anchor.
        elif self.control_mode == "joint_delta":
            self._jd_state = "JOINT_BOOTSTRAP"
            if self._jd is not None:
                self._jd.disengage()
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

        Precedence: PAUSED > WAITING(no target) > STALE > STALE_ROBOT >
        WAITING(unseeded) > HOLD > EEF_ENGAGED > DELTA_FOLLOWING >
        DELTA_CLUTCHED > CHASING > FOLLOWING. Read-only: never mutates bridge
        state or publishes commands. No joint-mode or eef-mode label changed when
        the two DELTA_* values were added.

        The last four are the eef-mode additions. CHASING/FOLLOWING compare the
        LEADER's joints to the commanded joints, which is only meaningful while
        the arm is supposed to be converging on the leader's joint pose — i.e. in
        joint mode / JOINT_BOOTSTRAP. In eef mode the leader and the robot live
        in deliberately DIFFERENT joint configurations forever, so that distance
        never shrinks and the old code reported CHASING permanently (both while
        holding and while engaged), which reads as "the arm is on its way
        somewhere" when it is either frozen or under delta control.
        """
        if self._paused:
            state = "PAUSED"
        elif self._raw_target is None or self._last_good_msg_time is None:
            state = "WAITING"
        elif (time.monotonic() - self._last_good_msg_time) > self.staleness_timeout_s:
            state = "STALE"
        elif (
            self._actual_pose_time is not None
            and (time.monotonic() - self._actual_pose_time)
            > self.actual_staleness_timeout_s
        ):
            # Robot-side watchdog tripped: publishing is stopped even though the
            # leader is fine. Distinct label so the operator does not hunt GELLO.
            state = "STALE_ROBOT"
        elif self._last_published is None:
            state = "WAITING"
        elif self._eef_hold_active() and self._eef_state != "ENGAGED":
            # 3D-pen bootstrap: arm deliberately frozen, awaiting ~/eef_engage.
            state = "HOLD"
        elif self.control_mode == "eef" and self._eef_state == "ENGAGED":
            state = "EEF_ENGAGED"
        elif self.control_mode == "joint_delta" and self._jd_state == "ENGAGED":
            # Same argument as EEF_ENGAGED: under start-anchored delta control
            # the leader and the command are SUPPOSED to diverge, so the
            # CHASING/FOLLOWING distance below is meaningless and would read
            # CHASING forever.
            state = "DELTA_FOLLOWING"
        elif self.control_mode == "joint_delta" and self._jd_state == "CLUTCHED":
            state = "DELTA_CLUTCHED"
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
        self._tick_overruns = 0.0
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

    def _actual_pose_fresh(self) -> bool:
        """True iff /joint_states is known AND younger than its timeout.

        Companion to the robot-side watchdog in _on_timer: any decision that
        assumes "the arm is where _actual_pose says it is" must go through this,
        because a dead External Control leaves the last pose frozen and
        plausible-looking forever.
        """
        if self._actual_pose is None or self._actual_pose_time is None:
            return False
        return (
            time.monotonic() - self._actual_pose_time
        ) <= self.actual_staleness_timeout_s

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

        run_baseline=True  -> full engage sequence G0..G9.
        run_baseline=False -> reclutch: G2,G5,G6,G7,G8,G9 (G3/G4 skipped; the
                              anchor is the frozen command pose _last_published).
                              Skipping G3/G4 is sound ONLY inside an already
                              validated ENGAGED session, where the command chain
                              is live and provably equals the arm's pose — which
                              is why ~/eef_reclutch still requires ENGAGED.

        Returns (ok, key, detail, q_anchor, q_lead_f). On failure key is the
        rejection string and q_anchor/q_lead_f are None.
        """
        np = self._np
        uk = self._ur_kin
        n = len(UR_JOINT_ORDER)

        # G1: control mode.
        if self.control_mode != "eef":
            return False, "not_eef_mode", "control_mode != eef", None, None
        # G0 (engage only): the bridge must NOT be paused. Without this, engage
        # succeeds while the bridge publishes nothing — an "ENGAGED but dead arm"
        # whose natural next operator action (~/resume) silently overwrites
        # _eef_state and destroys the engage. Refusing (rather than clearing
        # _paused here) keeps engage a pure, non-motion-causing operation and
        # forces recovery through ~/eef_resume, which does the actual-pose
        # re-seed that makes the restart provably jump-free.
        if run_baseline and self._paused:
            return (False, "bridge_paused",
                    "bridge is PAUSED/DISENGAGED — call ~/eef_resume first "
                    "(re-seeds from the arm's actual pose into HOLD), then "
                    "~/eef_engage",
                    None, None)
        # G2: fresh leader sample.
        age = self._eef_leader_age()
        if self._raw_target is None or age is None or age > self.staleness_timeout_s:
            a = "n/a" if age is None else f"{age:.2f}s"
            return False, "leader_stale", f"age={a} > {self.staleness_timeout_s:.2f}s", None, None
        # G3: a command baseline must exist (also the reclutch anchor).
        if self._last_published is None:
            return False, "no_command_baseline", "no _last_published yet", None, None
        # G4 (engage only): actual pose known, FRESH, and both chains already
        # agree. NOTE this compares _last_published to _actual_pose — BOTH
        # robot-side. It is a command-chain-vs-arm agreement check; nothing here
        # (or anywhere else at engage) compares the LEADER to the robot, which is
        # precisely what makes the 3D-pen handover legal.
        if run_baseline:
            if self._actual_pose is None:
                return False, "chains_disagree", "actual pose unknown", None, None
            if not self._actual_pose_fresh():
                a_age = (
                    "n/a" if self._actual_pose_time is None
                    else f"{time.monotonic() - self._actual_pose_time:.2f}s"
                )
                return (False, "actual_stale",
                        f"/joint_states age={a_age} > "
                        f"{self.actual_staleness_timeout_s:.2f}s",
                        None, None)
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
        # G6: EEF leader filter converged (positive proof), read from the CACHE
        # the publish tick fills exactly once per tick.
        #
        # This gate used to step the filter itself, and that step was the only
        # __call__ the filter ever saw outside the ENGAGED stage (_on_joint_state
        # advances only the speed estimate via update_input, never the output).
        # So the value it compared against the live leader was still essentially
        # the SEED — and with this repo's OneEuro at 250 Hz / min_cutoff 1.0 /
        # beta 2.0 the per-step alpha is 0.0245, making "settled < 0.005 rad"
        # equivalent to "the leader has not moved more than ~0.0051 rad (0.29 deg)
        # since the seed". No human hand satisfies that; only a frozen mock
        # leader did, which is why this never showed up before real hardware.
        # Now the tick steps the filter continuously, so `settle` is the filter's
        # genuine residual lag (~leader_speed * 0.16 s at the default tuning) and
        # the gate means what it says.
        #
        # The cache MUST be fresh: it is only filled while the bridge is actively
        # ticking, so a paused/unseeded bridge leaves it stale (or None) and the
        # gate fails closed instead of certifying a frozen filter.
        q_lead_f = self._q_lead_f
        lead_age = (
            None if self._q_lead_f_time is None
            else time.monotonic() - self._q_lead_f_time
        )
        if q_lead_f is None or lead_age is None or lead_age > self._q_lead_f_max_age_s:
            a = "n/a" if lead_age is None else f"{lead_age:.3f}s"
            return (False, "filter_not_running",
                    f"leader filter not stepping (cache age={a} > "
                    f"{self._q_lead_f_max_age_s:.3f}s) — bridge paused or "
                    "not yet seeded",
                    None, None)
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
        # Defensive copy: the caller hands this to the controller as the anchor's
        # q_lead0, and it must never alias live bridge state.
        return True, "ok", "", q_anchor, list(q_lead_f)

    def _on_eef_engage(self, request, response):
        """Engage EEF delta control (gates G0..G9). Robot does not move on any
        refusal. On success the anchor is latched (zero-jump).

        G0 refuses while the bridge is PAUSED: engaging a bridge that publishes
        nothing produced an "ENGAGED but dead arm" whose obvious next step
        (~/resume) silently overwrote the state. Recover with ~/eef_resume."""
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
        self._tick_overruns = 0.0
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
        going through pause. Re-checks G2,G5,G6,G7,G8,G9.

        DELIBERATELY still requires ENGAGED. Reclutch skips G3/G4, i.e. it never
        verifies that the command chain still agrees with the arm; that is only
        sound inside an already-validated engaged session, where the chain is
        live. Coming out of a fault (DISENGAGED/paused) _last_published may no
        longer be where the arm is, so re-anchoring there would command a jump —
        reclutch would run FEWER gates exactly when more are needed. The recovery
        path is ~/eef_resume (re-seed from the ACTUAL pose into HOLD) followed by
        ~/eef_engage, which runs the full G0..G9 including G3/G4.
        """
        if self._eef_state != "ENGAGED":
            response.success = False
            response.message = (
                f"eef_reclutch REFUSED — not ENGAGED (state={self._eef_state}). "
                "Reclutch skips the command-chain/arm agreement check and is only "
                "valid inside a live engaged session. To recover: ~/eef_resume "
                "then ~/eef_engage."
            )
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
        self._tick_overruns = 0.0
        self._eef_auto_reason = None
        response.success = True
        response.message = "RE-CLUTCHED — anchor reset at current pose (zero delta)."
        self.get_logger().warn(response.message)
        return response

    def _on_eef_to_joint(self, request, response):
        """Return to joint mode: pause, discard anchor, go to JOINT_BOOTSTRAP
        (joint passthrough). The operator then physically re-aligns GELLO and
        calls ~/resume (or ~/resume_chase). Reports the per-joint gap to close.

        This is the ONE service that intentionally LEAVES the 3D-pen hold: it is
        the deliberate hand-back to joint teleop, and the alignment gates on
        ~/resume / ~/resume_chase bound the motion that follows.
        """
        self._paused = True
        if self._eef is not None:
            self._eef.disengage()
        self._eef_state = "JOINT_BOOTSTRAP"
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
            "EEF->JOINT — paused, anchor discarded (state=JOINT_BOOTSTRAP, joint "
            "passthrough). Physically align the GELLO leader to the robot pose, "
            f"then call ~/resume or ~/resume_chase. {gap_str}"
        )
        self.get_logger().warn(response.message)
        return response

    def _on_eef_resume(self, request, response):
        """EEF-NATIVE re-arm: leave PAUSED into a HOLD — no alignment gate.

        This is the missing recovery path. EVERY eef fault route (leader_stale,
        hold_latched, tick_budget, exception) ends in _eef_autodisengage ->
        _paused=True. From there ~/eef_reclutch refuses (not ENGAGED), so the
        only way back used to be ~/resume / ~/resume_chase — and BOTH of those
        drop the bridge into joint passthrough, which in 3D-pen mode means the
        arm slews across the cell toward the leader's joint pose at up to
        max_step_rad * publish_rate_hz per joint, with NONE of the engaged-mode
        protections (keepout, singularity damping, branch lock, excursion cap,
        v_max) — those live only inside EefDeltaController.step(). Their
        alignment gates are what stood between the operator and that motion.

        The EEF-native answer is: do not re-align anything. Re-seed the command
        chain from the arm's ACTUAL pose (so the first published command equals
        where the arm already is — zero jump) and HOLD there. The leader's joint
        configuration is irrelevant, which is the whole point of the 3D pen, and
        no motion is authorized at all: the arm stays exactly where it stopped
        until ~/eef_engage latches a fresh anchor.

        FAIL CLOSED — refused (bridge stays PAUSED, publishes nothing) unless:
          (a) a FRESH GELLO sample exists (mirrors ~/resume's gate (a)), and
          (b) the arm's ACTUAL pose is known AND FRESH (~/resume's gate (b) plus
              the robot-side staleness watchdog: seeding from a frozen pose
              would defeat the zero-jump argument this service rests on).
        There is deliberately NO gate (c): no leader/arm alignment is required,
        because nothing here commands the arm toward the leader.
        """
        if self.control_mode != "eef":
            response.success = False
            response.message = "eef_resume REFUSED — control_mode != eef."
            self.get_logger().warn(response.message)
            return response

        if not self._paused:
            # Not paused: still normalize the state into HOLD if we are sitting
            # in a passthrough bootstrap? NO — that would silently undo an
            # explicit ~/eef_to_joint. Report and change nothing.
            response.success = True
            response.message = (
                f"Already following (was not paused); state={self._eef_state}."
            )
            self.get_logger().info("eef_resume requested but not paused.")
            return response

        # (a) FRESH leader sample required — same check as ~/resume.
        age = self._eef_leader_age()
        if self._raw_target is None or age is None or age > self.staleness_timeout_s:
            age_str = "n/a" if age is None else f"{age:.2f}s"
            response.success = False
            response.message = (
                f"eef_resume REFUSED — no fresh GELLO sample (age={age_str} > "
                f"{self.staleness_timeout_s:.2f}s). Bridge stays PAUSED (holds "
                "pose). Restore the leader stream, then try again."
            )
            self.get_logger().warn(response.message)
            return response

        # (b) arm pose must be known AND fresh — this is the seed.
        if self._actual_pose is None:
            response.success = False
            response.message = (
                "eef_resume REFUSED — robot actual pose unknown (no "
                f"{self._js_topic} yet). Bridge stays PAUSED (holds pose)."
            )
            self.get_logger().warn(response.message)
            return response
        if not self._actual_pose_fresh():
            a_age = time.monotonic() - self._actual_pose_time
            response.success = False
            response.message = (
                f"eef_resume REFUSED — robot joint state STALE (age={a_age:.2f}s "
                f"> {self.actual_staleness_timeout_s:.2f}s on {self._js_topic}). "
                "Bridge stays PAUSED (holds pose). Restart External Control, then "
                "try again."
            )
            self.get_logger().warn(response.message)
            return response

        # ACCEPTED. Land in HOLD, never in JOINT_BOOTSTRAP: with _eef_hold_active
        # True the pipeline holds _last_published every tick, so the arm does not
        # move at all. Mutations mirror ~/resume_chase (drop into the EXISTING
        # seed branch) plus the eef state reset; nothing publishes from here.
        self._paused = False
        self._filtered = None
        self._last_published = None
        if self._eef is not None:
            self._eef.disengage()  # anchor stays void until ~/eef_engage
        self._eef_state = "HOLD"
        self._hold_since = None
        self._tick_overruns = 0.0
        self._eef_auto_reason = None
        response.success = True
        response.message = (
            "EEF RE-ARMED — bridge unpaused into HOLD: the command chain re-seeds "
            "from the arm's CURRENT pose (zero jump) and the arm HOLDS there. No "
            "leader/arm alignment needed and no motion is authorized. Call "
            "~/eef_engage to latch a fresh anchor and resume delta control."
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

    def _eef_reported_state(self) -> str:
        """The state string published on ~/eef/state — never misleading.

        A PAUSED bridge publishes NO commands, so the arm is dead no matter what
        the internal state says; reporting "ENGAGED" there was measured on the
        mock (``{"eef": "ENGAGED", "bridge": "PAUSED"}`` with a 40 mm leader move
        producing ``dp_mm: [0,0,0]``) and it is the single most misleading thing
        the topic can say — the operator sees ENGAGED, moves the leader, nothing
        happens. Gate G0 now makes ENGAGED-while-paused unreachable through the
        services, so this is defence in depth: if it ever happens again it is a
        bug, and it gets reported honestly AND logged rather than displayed.
        """
        if self._paused and self._eef_state == "ENGAGED":
            self.get_logger().error(
                "INVARIANT VIOLATION: _eef_state==ENGAGED while the bridge is "
                "PAUSED (arm receives no commands). Reporting DISENGAGED.",
                throttle_duration_sec=_WARN_THROTTLE_S,
            )
            return "DISENGAGED"
        return self._eef_state

    def _on_eef_state_timer(self) -> None:
        """Publish the ~/eef/state JSON diagnostic (eef mode only, read-only)."""
        info = self._eef_info or {}
        gap = self._eef_joint_gap()
        payload = {
            "mode": "eef",
            "state": self._eef_reported_state(),
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
            # Derived pipeline behaviour, so the operator UI can distinguish
            # "holding (3D pen)" from "mirroring the leader's joints".
            "hold_when_not_engaged": self._eef_hold_active(),
            "paused": self._paused,
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

    # =====================================================================
    # JOINT_DELTA mode: automatic transitions, clutch services, diagnostics
    # =====================================================================
    def _jd_autodisengage(self, reason: str) -> None:
        """Fail-closed auto-disengage from within the tick (anchor invalidated).

        Structural twin of _eef_autodisengage: PAUSE the bridge (immediate hold
        on the next tick), void the controller anchor, record the reason for
        ~/joint_delta/state. Preserves the DISENGAGED == _paused invariant.
        Recovery is ~/joint_delta_start (jd-native, no alignment gate).
        """
        if self._jd is not None:
            self._jd.disengage()
        self._jd_state = "DISENGAGED"
        self._paused = True
        self._jd_hold_since = None
        self._jd_tick_overruns = 0.0
        self._jd_resync_until = None
        self._jd_auto_reason = reason
        self.get_logger().warn(
            f"JOINT_DELTA auto-disengaged ({reason}); bridge PAUSED (holds pose). "
            "Recover with ~/joint_delta_start.",
            throttle_duration_sec=_WARN_THROTTLE_S,
        )

    def _jd_anchor_in_limits(self, q_anchor):
        """G7: the ROBOT-side anchor must itself sit inside the (margined) joint
        limits. Cheap, and it refuses to anchor a pose from which the very first
        command would already be clamped — a saturated start feels identical to
        a broken teleop and is exactly the state the operator cannot diagnose.

        Returns (ok, detail). Reads the limits off the controller (plain floats,
        copied from ur_kin.JOINT_LIMITS) so this mode never imports numpy.
        """
        margin = self._jd.limit_margin
        for i in range(len(UR_JOINT_ORDER)):
            lo, hi = self._jd.joint_limits[i]
            if not (lo + margin <= q_anchor[i] <= hi - margin):
                return False, (
                    f"{UR_JOINT_SHORT[i]}={q_anchor[i]:.3f} outside "
                    f"[{lo + margin:.3f}, {hi - margin:.3f}]"
                )
        return True, ""

    def _run_jd_gates(self, run_baseline: bool):
        """Ordered engage/reclutch acceptance gates (fail-closed).

        run_baseline=True  -> full engage sequence G0..G7.
        run_baseline=False -> reclutch: G1,G2,G3,G5,G6,G7 (G0/G4 skipped; the
                              anchor is the live command pose _last_published).
                              Skipping G4 is sound ONLY inside an already
                              validated session where the command chain is live —
                              which is why ~/joint_delta_reclutch still requires
                              ENGAGED or CLUTCHED.

        Modelled on _run_eef_gates with the KINEMATICS gates deliberately
        dropped: there is no Cartesian mapping in this mode, so an IK self-test
        (eef G7), a singularity test (G8) and a keepout test (G9) would be
        theatre. What replaces them is G7 below plus the controller's own
        per-tick position clamp.

        Returns (ok, key, detail, q_anchor, q_lead_f). On failure key is the
        rejection string and q_anchor/q_lead_f are None.
        """
        n = len(UR_JOINT_ORDER)

        # G1: control mode.
        if self.control_mode != "joint_delta":
            return False, "not_joint_delta_mode", "control_mode != joint_delta", None, None
        # G0 (engage only): the bridge must NOT be paused. Same argument as eef's
        # G0 — engaging a bridge that publishes nothing produces an "ENGAGED but
        # dead arm", and the operator's natural next action (~/resume) would
        # silently overwrite the state. From paused, the correct call is
        # ~/joint_delta_start, which re-seeds from the ACTUAL pose first.
        if run_baseline and self._paused:
            return (False, "bridge_paused",
                    "bridge is PAUSED/DISENGAGED — call ~/joint_delta_start "
                    "instead (re-seeds from the arm's actual pose and engages "
                    "with a zero delta)",
                    None, None)
        # G2: fresh leader sample.
        age = self._eef_leader_age()
        if self._raw_target is None or age is None or age > self.staleness_timeout_s:
            a = "n/a" if age is None else f"{age:.2f}s"
            return False, "leader_stale", f"age={a} > {self.staleness_timeout_s:.2f}s", None, None
        # G3: a command baseline must exist (it IS the anchor).
        if self._last_published is None:
            return False, "no_command_baseline", "no _last_published yet", None, None
        # G4 (engage only): actual pose known, FRESH, and both ROBOT-SIDE chains
        # already agree, so anchoring on the COMMAND chain cannot silently bake a
        # tracking error into the anchor. Nothing here compares the LEADER to the
        # robot — in delta mode that comparison is meaningless.
        if run_baseline:
            if self._actual_pose is None:
                return False, "chains_disagree", "actual pose unknown", None, None
            if not self._actual_pose_fresh():
                a_age = (
                    "n/a" if self._actual_pose_time is None
                    else f"{time.monotonic() - self._actual_pose_time:.2f}s"
                )
                return (False, "actual_stale",
                        f"{self._js_topic} age={a_age} > "
                        f"{self.actual_staleness_timeout_s:.2f}s",
                        None, None)
            agree = max(
                abs(self._last_published[i] - self._actual_pose[i]) for i in range(n)
            )
            if agree > self.anchor_agree_tol:
                return (False, "chains_disagree",
                        f"max|cmd-actual|={agree:.4f} > {self.anchor_agree_tol:.4f}",
                        None, None)
        q_anchor = list(self._last_published)
        # G5: leader quasi-still. NOT optional here and NOT the same role it
        # plays in joint mode: this is the ONE thing preventing a mid-gesture
        # anchor, i.e. preventing the operator's moving hand position from being
        # silently redefined as the origin. Reuses the shared fail-closed util
        # and the resume_chase thresholds.
        if not leader_quasi_still(
            self._gello_history,
            self.resume_chase_still_window_s,
            self.resume_chase_still_speed,
        ):
            return False, "leader_moving", "leader not demonstrably still", None, None
        # G6: leader filter converged, read from the CACHE the publish tick fills
        # exactly once per tick (see _run_eef_gates for the full history of why
        # the cache — and its freshness check — has to be the source here).
        q_lead_f = self._q_lead_f
        lead_age = (
            None if self._q_lead_f_time is None
            else time.monotonic() - self._q_lead_f_time
        )
        if q_lead_f is None or lead_age is None or lead_age > self._q_lead_f_max_age_s:
            a = "n/a" if lead_age is None else f"{lead_age:.3f}s"
            return (False, "filter_not_running",
                    f"leader filter not stepping (cache age={a} > "
                    f"{self._q_lead_f_max_age_s:.3f}s) — bridge paused or "
                    "not yet seeded",
                    None, None)
        settle = max(abs(q_lead_f[i] - self._raw_target[i]) for i in range(n))
        if settle >= self.filter_settled_tol:
            return (False, "filter_not_settled",
                    f"max|q_lead_f-raw|={settle:.5f} >= {self.filter_settled_tol:.5f}",
                    None, None)
        # G7: the anchor itself must be inside the margined joint limits.
        ok, detail = self._jd_anchor_in_limits(q_anchor)
        if not ok:
            return False, "anchor_at_limit", detail, None, None
        # Defensive copy: the caller hands this to the controller as the anchor's
        # q_lead, and it must never alias live bridge state.
        return True, "ok", "", q_anchor, list(q_lead_f)

    def _on_jd_engage(self, request, response):
        """Engage start-anchored joint-delta control (gates G0..G7).

        The robot does not move on any refusal, and it does not move ON SUCCESS
        either: the anchor is latched at the CURRENT command pose with a zero
        delta, so the next tick publishes exactly what it published before.
        """
        ok, key, detail, q_anchor, q_lead_f = self._run_jd_gates(run_baseline=True)
        if not ok:
            response.success = False
            response.message = (
                f"joint_delta_engage REFUSED [{key}] — {detail}. Robot did not move."
            )
            self.get_logger().warn(response.message)
            return response
        self._jd.engage(q_anchor, q_lead_f)
        # DO NOT touch _seed_time: the command is continuous across an engage, so
        # restarting the soft-start ramp would only throttle the arm for no
        # reason (same rule as ~/eef_engage).
        self._jd_state = "ENGAGED"
        self._jd_hold_since = None
        self._jd_tick_overruns = 0.0
        self._jd_auto_reason = None
        # A fresh anchor makes an in-flight raw-teleport resync window moot.
        self._jd_resync_until = None
        response.success = True
        response.message = (
            f"ENGAGED — start-anchored joint delta live, gain={self.jd_gain:.2f}, "
            f"excursion cage +-{self.jd_max_excursion_rad:.2f} rad about the "
            "anchor. The arm now follows how much GELLO MOVES, not where it is."
        )
        self.get_logger().warn(response.message)
        return response

    def _on_jd_clutch(self, request, response):
        """THE MOUSE LIFT — freeze the output and the accumulator, keep streaming.

        ALWAYS immediate, no gate: releasing must never be refused. Crucially it
        does NOT set _paused, so the bridge keeps publishing the frozen pose
        (the arm holds actively) and the operator is free to reposition the
        leader without the robot following. Complete it with
        ~/joint_delta_reclutch, which re-anchors at zero delta.
        """
        if self._jd_state != "ENGAGED":
            response.success = False
            response.message = (
                f"joint_delta_clutch REFUSED — not ENGAGED (state={self._jd_state})."
            )
            self.get_logger().warn(response.message)
            return response
        self._jd.clutch()
        self._jd_state = "CLUTCHED"
        self._jd_hold_since = None
        response.success = True
        response.message = (
            "CLUTCHED — arm HOLDS its current pose and GELLO is free: reposition "
            "the leader, then call ~/joint_delta_reclutch to resume from there "
            "(zero delta, zero jump). The bridge is NOT paused."
        )
        self.get_logger().warn(response.message)
        return response

    def _on_jd_reclutch(self, request, response):
        """Re-anchor at the current leader/command pose (zero delta).

        Legal from ENGAGED (re-centre without letting go) and from CLUTCHED (the
        second half of the mouse lift). Re-checks G1,G2,G3,G5,G6,G7 — in
        particular G5, because the whole point of the reclutch is to adopt the
        leader's CURRENT rest position as the new origin, and adopting it
        mid-gesture would bake the rest of that gesture into the mapping.
        """
        if self._jd_state not in ("ENGAGED", "CLUTCHED"):
            response.success = False
            response.message = (
                f"joint_delta_reclutch REFUSED — not ENGAGED/CLUTCHED "
                f"(state={self._jd_state}). Recover with ~/joint_delta_start."
            )
            self.get_logger().warn(response.message)
            return response
        ok, key, detail, q_anchor, q_lead_f = self._run_jd_gates(run_baseline=False)
        if not ok:
            response.success = False
            response.message = (
                f"joint_delta_reclutch REFUSED [{key}] — {detail}. Robot holds."
            )
            self.get_logger().warn(response.message)
            return response
        self._jd.reclutch(q_lead_now=q_lead_f, q_cmd_now=self._last_published)
        self._jd_state = "ENGAGED"
        self._jd_hold_since = None
        self._jd_tick_overruns = 0.0
        self._jd_auto_reason = None
        # A fresh anchor makes an in-flight raw-teleport resync window moot.
        self._jd_resync_until = None
        response.success = True
        response.message = (
            "RE-CLUTCHED — anchor reset at the current leader/command pose "
            "(delta = 0). Streaming resumes with no jump."
        )
        self.get_logger().warn(response.message)
        return response

    def _on_jd_disengage(self, request, response):
        """Release delta control — ALWAYS immediate, no gate. Same path as
        ~/pause: pause + void anchor; the next tick (<=1 period) stops
        publishing and the arm holds its last commanded pose."""
        self._paused = True
        if self._jd is not None:
            self._jd.disengage()
        self._jd_state = "DISENGAGED"
        self._jd_hold_since = None
        self._jd_auto_reason = None
        response.success = True
        response.message = (
            "DISENGAGED — anchor discarded; robot holds (immediate). Re-arm with "
            "~/joint_delta_start."
        )
        self.get_logger().warn(
            "JOINT_DELTA DISENGAGED by operator (robot holds pose)."
        )
        return response

    def _on_jd_to_joint(self, request, response):
        """Hand back to ABSOLUTE joint teleop: pause, discard the anchor, go to
        JOINT_BOOTSTRAP. The operator then physically re-aligns GELLO to the arm
        and calls ~/resume (or ~/resume_chase), whose alignment gates bound the
        motion that follows.

        This is the escape hatch for the mode's principal residual risk: after
        enough clutch/re-clutch cycles the leader's pose no longer tells you the
        arm's pose. Reporting the per-joint gap here tells the operator exactly
        how far they have to move the leader to get the correspondence back.
        """
        self._paused = True
        if self._jd is not None:
            self._jd.disengage()
        self._jd_state = "JOINT_BOOTSTRAP"
        self._jd_hold_since = None
        self._jd_auto_reason = None
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
            "JOINT_DELTA->JOINT — paused, anchor discarded (state="
            "JOINT_BOOTSTRAP, absolute joint passthrough). Physically align the "
            "GELLO leader to the robot pose, then call ~/resume or "
            f"~/resume_chase. {gap_str}"
        )
        self.get_logger().warn(response.message)
        return response

    def _on_jd_start(self, request, response):
        """CHASE-FREE ARM/RE-ARM: leave PAUSED straight into ENGAGED, zero delta.

        This is the joint_delta answer to ~/eef_resume, and it is also the ONLY
        recovery out of every fault route (leader_stale / robot_joint_state_stale
        / hold_latched / tick_budget / exception), all of which end in
        _jd_autodisengage -> _paused=True. It exists because ~/resume and
        ~/resume_chase are joint-ABSOLUTE recoveries: after real delta drift
        their alignment gates are unsatisfiable, so in this mode they would
        refuse forever while protecting nothing.

        WHY IT IS SAFE WITHOUT THE MOVE-TO-START CHASE. In absolute mode the
        chase exists because the first streamed command is q_leader, which can be
        arbitrarily far from the arm. Here the first streamed command is
        q_robot_anchor := the arm's OWN current pose, with delta == 0 — zero jump
        ALGEBRAICALLY, not within a tolerance. The chase's other two protections
        do not vanish, they relocate and are both present: "do not hand over to a
        moving leader" is gate (c) below (leader_quasi_still), and the
        gross-mispose backstop relocates to streaming time as the controller's
        joint-limit + excursion clamps.

        FAIL CLOSED — refused (bridge stays PAUSED, publishes nothing) unless:
          (a) control_mode == joint_delta and the bridge IS paused and the state
              is JOINT_BOOTSTRAP or DISENGAGED (never a live session),
          (a2) THE BRIDGE HAS STREAMED AT LEAST ONCE SINCE LAUNCH, i.e. the
              move-to-start handshake already ran its STRICT switch and released
              us. See the comment on the gate below — "PAUSED" alone does NOT
              imply the handover happened, and acting on that false premise is a
              full-chase-distance snap.
          (b) a FRESH GELLO sample exists,
          (c) the leader is demonstrably quasi-still (the anti mid-gesture-anchor
              gate — the one protection that has NO analogue in absolute mode),
          (d) the arm's ACTUAL pose is known AND FRESH (it is the anchor AND the
              seed; a frozen /joint_states would defeat the zero-jump argument),
          (e) that actual pose is inside the margined joint limits (G7).

        Gate G6 (filter settled) is deliberately NOT run: while paused the tick
        does not step the leader filter, so the cache is stale by construction
        and G6 could never pass. Instead this service SEEDS the filter from the
        anchored leader pose itself, which makes q_lead_f exactly the anchored
        pose — a stronger statement than "settled".
        """
        if self.control_mode != "joint_delta":
            response.success = False
            response.message = "joint_delta_start REFUSED — control_mode != joint_delta."
            self.get_logger().warn(response.message)
            return response

        if not self._paused:
            response.success = False
            response.message = (
                f"joint_delta_start REFUSED — bridge is not paused (state="
                f"{self._jd_state}). Use ~/joint_delta_engage to anchor a "
                "streaming bridge, or ~/pause first."
            )
            self.get_logger().warn(response.message)
            return response

        if self._jd_state not in ("JOINT_BOOTSTRAP", "DISENGAGED"):
            response.success = False
            response.message = (
                f"joint_delta_start REFUSED — unexpected state {self._jd_state}."
            )
            self.get_logger().warn(response.message)
            return response

        # (a2) THE HANDOVER MUST ALREADY HAVE HAPPENED.
        #
        # This gate exists because "the bridge is PAUSED" does NOT imply "the
        # STRICT switch to forward_position_controller has run": the launch
        # PRE-SPAWNS this node with start_paused:=True at t=6s, so the ENTIRE
        # pre-handshake and chase window is also PAUSED. Without this gate an
        # operator who called ~/joint_delta_start during that window would arm
        # against the arm's PRE-CHASE pose and start streaming it at 250 Hz into
        # an INACTIVE forward_position_controller; gello_move_to_start would then
        # chase the arm to the leader pose and STRICT-activate that controller,
        # which picks up the very next command — the pre-chase pose — and demands
        # the whole chase distance in one 4 ms cycle. That is a violent lurch /
        # UR protective stop, and jd_gain:=0.0 does NOT protect against it (gain
        # 0 pins the command AT the stale anchor, which is exactly the snap
        # value). It is also unrecoverable by the handshake's own
        # /gello_ur_bridge/resume, which early-returns "already following".
        #
        # _has_streamed is the local proof we need and the only one available
        # without a ListControllers round-trip (which cannot be awaited from
        # inside a service callback on this single-threaded executor): _on_timer
        # returns immediately while paused, so a command has been published IFF
        # the bridge was released at least once, and the only thing that releases
        # it at bring-up is the handshake — AFTER the switch, by construction
        # (gello_move_to_start_node.py: "release it now (after the switch, never
        # before, so it can't stream into an inactive controller)").
        #
        # jd_start_allow_unstreamed OPT-OUT: the switch_only no-motion bring-up
        # sets this parameter True. In that path gello_move_to_start calls this
        # service ONLY after a successful STRICT switch to
        # forward_position_controller (it never chases), so the controller is
        # already ACTIVE and the snap this gate protects against cannot occur —
        # the very ordering invariant _has_streamed proxies is guaranteed
        # structurally, exactly as ~/eef_resume relies on it with no gate at all.
        # Default False keeps the manual-recovery gate for every other caller.
        if not self._has_streamed and not self.jd_start_allow_unstreamed:
            response.success = False
            response.message = (
                "joint_delta_start REFUSED — the bridge has never streamed, so "
                "the move-to-start handshake has NOT handed over to "
                "forward_position_controller yet. Arming now would command the "
                "arm's PRE-chase pose the instant the controller activates — a "
                "full-chase-distance snap. Finish the normal handshake first "
                "('1) 진행' → alignment → /gello_ur_bridge/resume), then use "
                "~/joint_delta_engage; ~/joint_delta_start is the RECOVERY path "
                "for a session that already ran. (The switch_only no-motion "
                "bring-up sets jd_start_allow_unstreamed:=True to bypass this "
                "gate safely, since it switches the controller ACTIVE before "
                "calling this service.)"
            )
            self.get_logger().error(response.message)
            return response

        # (b) FRESH leader sample required.
        age = self._eef_leader_age()
        if self._raw_target is None or age is None or age > self.staleness_timeout_s:
            age_str = "n/a" if age is None else f"{age:.2f}s"
            response.success = False
            response.message = (
                f"joint_delta_start REFUSED — no fresh GELLO sample "
                f"(age={age_str} > {self.staleness_timeout_s:.2f}s). Bridge stays "
                "PAUSED (holds pose). Restore the leader stream, then try again."
            )
            self.get_logger().warn(response.message)
            return response

        # (c) leader quasi-still — anti mid-gesture anchor.
        if not leader_quasi_still(
            self._gello_history,
            self.resume_chase_still_window_s,
            self.resume_chase_still_speed,
        ):
            response.success = False
            response.message = (
                "joint_delta_start REFUSED — leader is moving (or stillness not "
                "yet established). Hold GELLO steady for ~0.5 s and try again. "
                "Anchoring mid-gesture would silently redefine your moving hand "
                "position as the origin. Bridge stays PAUSED (holds pose)."
            )
            self.get_logger().warn(response.message)
            return response

        # (d) arm pose known AND fresh — it is both the anchor and the seed.
        if self._actual_pose is None:
            response.success = False
            response.message = (
                "joint_delta_start REFUSED — robot actual pose unknown (no "
                f"{self._js_topic} yet). Bridge stays PAUSED (holds pose)."
            )
            self.get_logger().warn(response.message)
            return response
        if not self._actual_pose_fresh():
            a_age = time.monotonic() - self._actual_pose_time
            response.success = False
            response.message = (
                f"joint_delta_start REFUSED — robot joint state STALE "
                f"(age={a_age:.2f}s > {self.actual_staleness_timeout_s:.2f}s on "
                f"{self._js_topic}). Bridge stays PAUSED (holds pose). Restart "
                "External Control, then try again."
            )
            self.get_logger().warn(response.message)
            return response

        # (e) the anchor must sit inside the margined joint limits.
        ok, detail = self._jd_anchor_in_limits(self._actual_pose)
        if not ok:
            response.success = False
            response.message = (
                f"joint_delta_start REFUSED [anchor_at_limit] — {detail}. Move the "
                "arm away from the limit (pendant/freedrive), then try again."
            )
            self.get_logger().warn(response.message)
            return response

        # ACCEPTED. Do the seed branch's work ATOMICALLY here (no spin in
        # between), so the robot anchor, the leader anchor and the command
        # baseline are all read from the same instant — the near-atomic capture
        # the delta formulation requires. This mirrors _on_timer's seed branch
        # exactly; it is not a second publishing path (nothing publishes here,
        # the next tick does).
        anchored = wrapped_nearest(self._raw_target, self._actual_pose)
        self._unwrapped_target = anchored
        self._raw_target = anchored
        self._filtered = list(self._actual_pose)
        self._last_published = list(self._actual_pose)
        self._gated_target = list(anchored)
        if self._euro is not None:
            for i in range(len(UR_JOINT_ORDER)):
                self._euro[i].seed(self._actual_pose[i])
        for i in range(len(UR_JOINT_ORDER)):
            self._euro_lead[i].seed(anchored[i])
        # The bank was just RESET, so any cached output predates it. The next
        # tick refills the cache before the delta stage reads it.
        self._q_lead_f = None
        self._q_lead_f_time = None
        self._seed_time = time.monotonic()   # start the soft-start ramp
        # Leader anchor == the value the freshly seeded filter now outputs.
        self._jd.engage(list(self._actual_pose), list(anchored))
        self._jd_state = "ENGAGED"
        self._jd_hold_since = None
        self._jd_tick_overruns = 0.0
        self._jd_auto_reason = None
        # A fresh anchor makes an in-flight raw-teleport resync window moot.
        self._jd_resync_until = None
        self._paused = False
        response.success = True
        response.message = (
            "JOINT_DELTA ARMED — anchored at the arm's CURRENT pose with delta=0 "
            f"(gain={self.jd_gain:.2f}). The first published command equals where "
            "the arm already is (zero jump); from now on the arm follows how much "
            "GELLO MOVES. Keep clear."
        )
        self.get_logger().warn(response.message)
        return response

    def _jd_reported_state(self) -> str:
        """The state string published on ~/joint_delta/state — never misleading.

        Same defence-in-depth as _eef_reported_state: a PAUSED bridge publishes
        no commands, so ENGAGED/CLUTCHED there would tell the operator the arm is
        live when it is dead.
        """
        if self._paused and self._jd_state in ("ENGAGED", "CLUTCHED"):
            self.get_logger().error(
                f"INVARIANT VIOLATION: _jd_state=={self._jd_state} while the "
                "bridge is PAUSED (arm receives no commands). Reporting "
                "DISENGAGED.",
                throttle_duration_sec=_WARN_THROTTLE_S,
            )
            return "DISENGAGED"
        return self._jd_state

    def _on_jd_state_timer(self) -> None:
        """Publish the ~/joint_delta/state JSON diagnostic (jd mode only,
        read-only)."""
        info = self._jd_info or {}
        payload = {
            "mode": "joint_delta",
            "state": self._jd_reported_state(),
            "ctrl_state": info.get("state"),
            "reject_reason": info.get("reject_reason"),
            "auto_reason": self._jd_auto_reason,
            "gain": self.jd_gain,
            "delta": info.get("delta"),
            "limited": info.get("limited"),
            "slewed": info.get("slewed"),
            "excursion_rad": info.get("excursion_rad"),
            "max_excursion_rad": self.jd_max_excursion_rad,
            "max_joint_step": info.get("max_joint_step"),
            "jump_absorbed": info.get("jump_absorbed"),
            # Raw-domain teleport detector (the PRIMARY one) and whether its
            # hold-and-resync window is currently open.
            "raw_jumps": self._jd_raw_jumps,
            "resyncing": (
                self._jd_resync_until is not None
                and time.monotonic() < self._jd_resync_until
            ),
            # HOW FAR THE LEADER<->ARM CORRESPONDENCE HAS SILENTLY SHIFTED.
            # Cumulative leader travel (rad, per joint) discarded by the position
            # clamp's anti-windup back-projection since the current anchor. Any
            # nonzero entry means "returning the leader to where you anchored it
            # no longer returns the arm to the anchor" — clutch/reclutch to reset.
            "clamp_discarded": info.get("clamp_discarded"),
            "tick_us": self._jd_tick_us_last,
            "tick_soft_overruns": self._jd_tick_soft_overruns,
            # The mode's principal residual risk made VISIBLE: after any drift
            # the leader's pose no longer tells you the arm's pose, and this is
            # the number that says how far apart they now are.
            "joint_gap": self._eef_joint_gap(),
            "paused": self._paused,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self._jd_state_pub.publish(msg)

    # ---------------------------------------------------------------------
    def _publish(self, positions: list[float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(p) for p in positions]
        self._pub.publish(msg)
        # One-way latch, read ONLY by ~/joint_delta_start (joint_delta mode).
        # Assigning a bool costs nothing and no joint/eef branch reads it.
        self._has_streamed = True


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
