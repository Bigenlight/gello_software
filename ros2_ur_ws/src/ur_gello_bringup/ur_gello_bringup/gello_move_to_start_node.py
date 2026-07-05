#!/usr/bin/env python3
"""One-shot GELLO -> UR move-to-start SAFETY handshake.

Why this node exists
--------------------
The teleop bridge (``gello_ur_bridge``) streams GELLO poses to the UR
``forward_position_controller``. That controller does NOT interpolate: whatever
position it receives is commanded to the servoj loop immediately. If the arm is
sitting at some pose (on real hardware: the driver's current pose; often far
from the GELLO leader's pose) and the bridge suddenly starts publishing the
GELLO pose, the resulting one-cycle jump violates the UR joint-velocity limits
and the robot trips a PROTECTIVE STOP.

To avoid that, on REAL hardware we must first drive the arm SMOOTHLY to the
current GELLO pose using the ``scaled_joint_trajectory_controller`` (which DOES
interpolate a time-parameterised trajectory), and only once the arm has
arrived hand control off to ``forward_position_controller`` for streaming.

This node performs exactly that handshake, once, then shuts down so the bridge
can take over:

    1. Read ONE GELLO joint_states message (reordered BY NAME into UR order).
    2. Send ONE FollowJointTrajectory goal moving the arm to that pose over
       ``trajectory_duration`` seconds; wait for SUCCEEDED.
    3. Switch controllers (STRICT): activate ``forward_position_controller``,
       deactivate ``scaled_joint_trajectory_controller``.
    4. On success: log and rclpy.shutdown().

Fail-safe
---------
On ANY failure (no GELLO message, goal rejected, goal aborted, arrival out of
tolerance, or the controller switch failing) the node logs an error and does
NOT switch controllers, leaving ``scaled_joint_trajectory_controller`` active
and the arm stationary. Streaming never begins from an unsafe state.

GELLO is PASSIVE: this node only READS ``/gello/joint_states``. It never
commands the GELLO Dynamixels (torque stays off).

Start modes
-----------
``start_mode:=gello`` (default) runs a CONVERGENCE-GATED chase: it re-reads the
LIVE leader pose (and the robot's actual ``/joint_states``) each iteration and
drives gap-sized catch-up trajectories, handing over ONLY once the arm agrees
with the live leader within ``chase_tol`` on every joint, sustained
``chase_dwell_s``. (This replaced the original one-shot move to the FIRST-seen
pose, which snapped when streaming began because the leader had drifted/moved
during the approach.) A leader that keeps moving simply delays handover; a
dead/stale leader times out to the fail-safe (no switch).

``start_mode:=init_align`` is an INTERACTIVE, operator-gated flow for extra
safety on unfamiliar hardware. Every stage transition requires an explicit
human authorization via a ROS service (Trigger):

    GATE 1  after the source controller is active, the node WAITS. The operator
            calls ``~/proceed`` to authorize moving the arm to a FIXED, known
            ``init_pose``.
    GATE 2  the arm parks at ``init_pose``; the operator moves the passive GELLO
            leader to match it (live per-joint error is logged). When the
            operator calls ``~/proceed`` the node COMPUTES the per-joint error
            |GELLO - init_pose| and:
              * all joints <= alignment_tolerance  -> hand over (stream).
              * some joint  >  alignment_tolerance -> REFUSE, and REPORT which
                joints are off and by how much (so a large lone offset, e.g. the
                base/shoulder_pan being hard to eyeball, cannot silently drive a
                big follow move). Operator re-aligns, or, if they accept the
                offset, calls ``~/override_follow``.
              * ``~/override_follow`` hands over despite an offset ONLY if every
                joint is within alignment_hard_limit; beyond that it is refused
                even with override. Re-align.
"""

import time
from collections import deque
from statistics import median

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from controller_manager_msgs.srv import ListControllers, SwitchController
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# UR command joint order (identical ur5e & ur7e). Reorder GELLO BY NAME to this.
UR_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Short, UNAMBIGUOUS per-joint labels for the alignment report (same order).
UR_JOINT_SHORT = ["pan", "lift", "elbow", "w1", "w2", "w3"]

# controller_manager_msgs/srv/SwitchController strictness enum.
_STRICT = 2

# How long to block waiting for services / action server to appear.
_DISCOVERY_TIMEOUT_S = 10.0


class GelloMoveToStart(Node):
    """Drive UR to the current GELLO pose, then hand off to the stream controller."""

    def __init__(self) -> None:
        super().__init__("gello_move_to_start")

        # --- Parameters --------------------------------------------------
        self.source_controller = str(
            self.declare_parameter(
                "source_controller", "scaled_joint_trajectory_controller"
            ).value
        )
        self.target_controller = str(
            self.declare_parameter(
                "target_controller", "forward_position_controller"
            ).value
        )
        # RESUME the pre-spawned (paused) streaming bridge after a successful
        # STRICT switch. Default False preserves standalone/mock runs where no
        # bridge exists; the integrated real launch sets resume_bridge:=True (the
        # bridge is started start_paused:=true, so this call is what actually
        # begins teleop). The bridge's ~/resume is alignment-gated, so a
        # just-converged arm (gap <= chase_tol < resume_align_tol) passes.
        self.resume_bridge = bool(
            self.declare_parameter("resume_bridge", False).value
        )
        self.bridge_resume_service = str(
            self.declare_parameter(
                "bridge_resume_service", "/gello_ur_bridge/resume"
            ).value
        )
        self.trajectory_duration = float(
            self.declare_parameter("trajectory_duration", 5.0).value
        )
        self.arrival_tolerance = float(
            self.declare_parameter("arrival_tolerance", 0.05).value
        )

        # --- Convergence-gated handover ("gello" mode) -------------------
        # The leader is a MOVING target: a single captured pose goes stale while
        # the arm approaches (and while the operator may keep moving GELLO). So
        # instead of one blind trajectory to a frozen pose, we CHASE the live
        # pose and only hand over to streaming once the arm's ACTUAL pose agrees
        # with the LIVE leader within chase_tol on every joint, sustained for
        # chase_dwell_s. If the operator keeps moving, the switch simply does not
        # happen (fail direction is "won't start yet", never "moves unexpectedly").
        #
        # INVARIANT: chase_tol MUST be > arrival_tolerance. The catch-up
        # FollowJointTrajectory goal reports SUCCEEDED as soon as the arm is within
        # arrival_tolerance of the command, so a REAL servo that settles in the
        # (chase_tol, arrival_tolerance] dead band would satisfy the trajectory yet
        # NEVER satisfy a tighter gate -> the chase livelocks to chase_timeout_s and
        # teleop never starts. (The fake-hardware mock hides this by arriving
        # exactly.) Default 0.06 > arrival 0.05. The guard below enforces it.
        self.chase_tol = float(self.declare_parameter("chase_tol", 0.06).value)
        # Enforce the invariant so a misconfig can't silently livelock the arm.
        if self.chase_tol <= self.arrival_tolerance:
            bumped = self.arrival_tolerance + 0.01
            self.get_logger().warn(
                f"chase_tol ({self.chase_tol}) <= arrival_tolerance "
                f"({self.arrival_tolerance}): the convergence gate could NEVER pass "
                f"on real hardware (trajectory succeeds within arrival_tolerance but "
                f"the gate demands tighter) -> auto-raising chase_tol to {bumped:.3f}. "
                f"Set chase_tol > arrival_tolerance in the params to silence this."
            )
            self.chase_tol = bumped
        self.chase_dwell_s = float(
            self.declare_parameter("chase_dwell_s", 0.4).value
        )
        # Velocity budget (rad/s) used to SIZE each catch-up trajectory from the
        # measured gap: T = max(gap / chase_v_budget, min_traj_duration). No upper
        # clamp — a big gap gets a proportionally longer (not faster) move, so the
        # JTC spline peak stays well under the 3.14 rad/s protective-stop limit.
        self.chase_v_budget = float(
            self.declare_parameter("chase_v_budget", 0.3).value
        )
        self.min_traj_duration = float(
            self.declare_parameter("min_traj_duration", 0.75).value
        )
        # Per-joint HARD refuse: never auto-chase a gap larger than this. This is
        # a WRAPAROUND / gross-mispose backstop, NOT an approach limit — the arm
        # legitimately starts far (up to ~pi) from the leader, and the catch-up
        # trajectory is duration-sized so even a large approach moves safely. Set
        # above the largest normal approach (~pi) but below a 2*pi (6.28 rad) wrap.
        # Calibrate on real hardware. Default 4.0 rad allows normal approaches and
        # still catches a full-turn wraparound.
        self.chase_hard_limit = float(
            self.declare_parameter("chase_hard_limit", 4.0).value
        )
        # Overall time budget for convergence; <=0 => wait forever.
        self.chase_timeout_s = float(
            self.declare_parameter("chase_timeout_s", 30.0).value
        )
        # A GELLO sample older than this is STALE: a dead stream must never count
        # as "still" and pass the dwell (that would re-open the snap on recovery).
        self.gello_staleness_s = float(
            self.declare_parameter("gello_staleness_s", 0.5).value
        )

        # --- Start-mode (handshake style) --------------------------------
        #   "gello"      : (default) CONVERGENCE-GATED chase — re-reads the live
        #                  leader + actual pose and drives gap-sized catch-ups,
        #                  handing over only when |leader-arm| <= chase_tol on
        #                  every joint, sustained chase_dwell_s. A moving leader
        #                  delays handover; it never streams from a stale gap.
        #   "init_align" : robot first moves to a FIXED, known init_pose; the
        #                  operator then aligns the GELLO leader to that pose;
        #                  streaming starts ONLY once the leader is within
        #                  alignment_tolerance of init_pose for alignment_dwell_s.
        #                  Safer: the arm's first autonomous motion is to a known
        #                  pose, and handover happens with leader ~= follower so
        #                  there is no large slew when streaming begins.
        self.start_mode = str(
            self.declare_parameter("start_mode", "gello").value
        ).strip().lower()
        self.init_pose = [
            float(x)
            for x in self.declare_parameter(
                "init_pose", [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
            ).value
        ]
        # Per-joint tolerance (rad): with every joint within this of init_pose, a
        # ~/proceed call hands over to streaming (init_align mode only).
        self.alignment_tolerance = float(
            self.declare_parameter("alignment_tolerance", 0.15).value
        )
        # Per-joint HARD limit (rad): a ~/override_follow is accepted only if every
        # joint is within this of init_pose; beyond it, handover is refused even
        # with override (protects against a large lone offset, e.g. the base).
        self.alignment_hard_limit = float(
            self.declare_parameter("alignment_hard_limit", 0.5).value
        )
        # Max time (s) to wait at a gate for the operator; <=0 => wait forever.
        self.alignment_timeout = float(
            self.declare_parameter("alignment_timeout", 0.0).value
        )
        if self.start_mode not in ("gello", "init_align"):
            self.get_logger().warn(
                f"Unknown start_mode '{self.start_mode}'; falling back to 'gello'."
            )
            self.start_mode = "gello"
        if self.start_mode == "init_align" and len(self.init_pose) != len(UR_JOINT_ORDER):
            self.get_logger().error(
                f"init_pose has {len(self.init_pose)} values, expected "
                f"{len(UR_JOINT_ORDER)}; falling back to 'gello' mode (fail-safe)."
            )
            self.start_mode = "gello"
        # How long to wait for the source controller to become 'active'. On the
        # real UR the controller_stopper keeps motion controllers inactive until
        # the External Control program is PLAYING on the pendant, so this wait
        # gives the operator time to press Play.
        self.activation_timeout = float(
            self.declare_parameter("activation_timeout", 120.0).value
        )

        # --- State -------------------------------------------------------
        # _gello_target: FIRST complete GELLO pose (frozen; the move target in
        #   "gello" mode). _gello_latest: MOST RECENT GELLO pose (updated every
        #   message; used by the init_align alignment gate).
        self._gello_target: list[float] | None = None
        self._gello_latest: list[float] | None = None
        # Monotonic time of the last complete GELLO message (staleness gate).
        self._last_gello_msg_time: float | None = None
        # Robot's ACTUAL current pose (UR order) from /joint_states — used to
        # measure the true live gap |leader - arm| at the convergence gate.
        self._actual_pose: list[float] | None = None
        # Operator-gate flags (init_align mode), set by the Trigger services.
        self._stage = "init"          # "init" -> "wait_align" -> "streaming"
        self._proceed_init = False    # GATE 1 authorization (move to init pose)
        self._handover = False        # GATE 2 authorization granted (stream)
        self._overridden = False      # handover was via ~/override_follow
        self._abort = False           # operator pressed "2) 정지" -> fail-safe exit
        self._go_home_requested = False  # "4) 홈으로": re-send init-pose trajectory

        # --- ROS interfaces ----------------------------------------------
        self._sub = self.create_subscription(
            JointState, "/gello/joint_states", self._on_joint_state, 10
        )
        # Robot's actual joint state (joint_state_broadcaster; real HW + mock),
        # reordered BY NAME — the convergence gate compares live GELLO to this.
        self._actual_sub = self.create_subscription(
            JointState, "/joint_states", self._on_actual_joint_state, 10
        )
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            f"/{self.source_controller}/follow_joint_trajectory",
        )
        self._switch_client = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        self._list_client = self.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        # Resume client for the pre-spawned paused bridge (created early so its
        # discovery cost is hidden behind the Play-wait). Only USED when
        # resume_bridge is True, after a successful switch.
        self._resume_client = self.create_client(
            Trigger, self.bridge_resume_service
        )
        # Operator-gate services (used only in init_align mode). Resolve to
        # /gello_move_to_start/proceed and /gello_move_to_start/override_follow.
        self._proceed_srv = self.create_service(
            Trigger, "~/proceed", self._on_proceed
        )
        self._override_srv = self.create_service(
            Trigger, "~/override_follow", self._on_override_follow
        )
        self._abort_srv = self.create_service(
            Trigger, "~/abort", self._on_abort
        )
        self._go_home_srv = self.create_service(
            Trigger, "~/go_home", self._on_go_home
        )
        self._check_srv = self.create_service(
            Trigger, "~/check_alignment", self._on_check_alignment
        )

        self.get_logger().info(
            "gello_move_to_start started | "
            f"start_mode={self.start_mode} "
            f"source={self.source_controller} "
            f"target={self.target_controller} "
            f"trajectory_duration={self.trajectory_duration} "
            f"arrival_tolerance={self.arrival_tolerance}"
        )

    # ---------------------------------------------------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        """Track the latest complete GELLO pose; latch the first one as target."""
        name_to_pos = dict(zip(msg.name, msg.position))
        missing = [j for j in UR_JOINT_ORDER if j not in name_to_pos]
        if missing:
            self.get_logger().warn(
                f"GELLO message missing UR joint(s) {missing}; waiting for a "
                "complete message",
                throttle_duration_sec=2.0,
            )
            return
        # Reorder BY NAME into UR command order (never blind index).
        pose = [float(name_to_pos[j]) for j in UR_JOINT_ORDER]
        self._gello_latest = pose
        self._last_gello_msg_time = time.monotonic()
        if self._gello_target is None:
            self._gello_target = pose
            self.get_logger().info(f"Captured GELLO target pose: {pose}")

    # ---------------------------------------------------------------------
    def _on_actual_joint_state(self, msg: JointState) -> None:
        """Track the robot's ACTUAL pose (reordered BY NAME) for the gap check."""
        name_to_pos = dict(zip(msg.name, msg.position))
        if any(j not in name_to_pos for j in UR_JOINT_ORDER):
            return  # partial / unrelated joint_states; ignore
        self._actual_pose = [float(name_to_pos[j]) for j in UR_JOINT_ORDER]

    # ---------------------------------------------------------------------
    def _gello_fresh(self) -> bool:
        """True if a complete GELLO sample arrived within gello_staleness_s."""
        t = self._last_gello_msg_time
        return t is not None and (time.monotonic() - t) <= self.gello_staleness_s

    # ---------------------------------------------------------------------
    def _wait_for_gello_target(self) -> bool:
        """Spin until the first complete GELLO message arrives. True on success."""
        self.get_logger().info(
            "Waiting for first GELLO /gello/joint_states message..."
        )
        while rclpy.ok() and self._gello_target is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._gello_target is not None

    # ---------------------------------------------------------------------
    def _wait_for_source_active(self) -> bool:
        """Block until the source controller reports 'active'.

        On the real UR, ``ur_control.launch.py`` starts
        ``scaled_joint_trajectory_controller`` but the driver's
        controller_stopper holds it INACTIVE until the External Control
        program is running (Play) on the pendant and the reverse interface is
        connected. Sending a trajectory before then is rejected with
        "Controller is not running". So we poll ``list_controllers`` until the
        source controller is active, prompting the operator to press Play.
        Returns True once active, False on timeout.
        """
        if not self._list_client.wait_for_service(timeout_sec=_DISCOVERY_TIMEOUT_S):
            self.get_logger().error(
                "/controller_manager/list_controllers not available; aborting "
                "(fail-safe, no switch)"
            )
            return False

        deadline = time.monotonic() + self.activation_timeout
        warned = False
        while rclpy.ok() and time.monotonic() < deadline:
            future = self._list_client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            resp = future.result()
            if resp is not None:
                for c in resp.controller:
                    if c.name == self.source_controller and c.state == "active":
                        self.get_logger().info(
                            f"{self.source_controller} is ACTIVE; proceeding "
                            "with move-to-start."
                        )
                        return True
            if not warned:
                self.get_logger().warn(
                    f"{self.source_controller} is not active yet. Method A: on "
                    "the pendant START (Play) the External Control program. "
                    "Method B (headless): ensure the pendant is in REMOTE mode + "
                    "Real Robot (the reverse interface auto-connects, no Play "
                    f"needed). Waiting up to {self.activation_timeout:.0f}s..."
                )
                warned = True
            rclpy.spin_once(self, timeout_sec=0.5)

        self.get_logger().error(
            f"{self.source_controller} did not become active within "
            f"{self.activation_timeout:.0f}s; aborting (fail-safe, no switch). "
            "Is the External Control program running on the pendant?"
        )
        return False

    # ---------------------------------------------------------------------
    def _send_trajectory(
        self, target: list[float], label: str, duration: float | None = None
    ) -> bool:
        """Send one FollowJointTrajectory goal to ``target``; wait SUCCEEDED.

        ``duration`` overrides ``trajectory_duration`` (used by the convergence
        chase loop to size each catch-up move from the measured gap).
        """
        assert target is not None
        dur = self.trajectory_duration if duration is None else duration

        if not self._action_client.wait_for_server(
            timeout_sec=_DISCOVERY_TIMEOUT_S
        ):
            self.get_logger().error(
                "FollowJointTrajectory action server "
                f"/{self.source_controller}/follow_joint_trajectory not "
                "available; aborting (fail-safe, no controller switch)"
            )
            return False

        point = JointTrajectoryPoint()
        point.positions = list(target)
        point.velocities = [0.0] * len(UR_JOINT_ORDER)
        sec = int(dur)
        nanosec = int((dur - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)

        traj = JointTrajectory()
        traj.joint_names = list(UR_JOINT_ORDER)
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        # Arrival tolerance per joint (position). Empty path_tolerance = default.
        for name in UR_JOINT_ORDER:
            tol = JointTolerance()
            tol.name = name
            tol.position = self.arrival_tolerance
            goal.goal_tolerance.append(tol)

        self.get_logger().info(
            f"Sending trajectory to {label} over {dur:.2f}s "
            f"via {self.source_controller}..."
        )
        send_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                "Trajectory goal REJECTED; aborting (fail-safe, no switch)"
            )
            return False

        self.get_logger().info("Goal accepted; waiting for arm to arrive...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            self.get_logger().error(
                "No trajectory result returned; aborting (fail-safe, no switch)"
            )
            return False

        # rclpy action status: 4 == STATUS_SUCCEEDED.
        from action_msgs.msg import GoalStatus

        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f"Trajectory did NOT succeed (status={wrapped.status}, "
                f"error_code={wrapped.result.error_code}); aborting "
                "(fail-safe, no switch)"
            )
            return False

        self.get_logger().info(f"Arm arrived at {label} (trajectory SUCCEEDED).")
        return True

    # ---------------------------------------------------------------------
    def _alignment_errors(self) -> list[float] | None:
        """Per-joint |GELLO_latest - init_pose| (rad), or None if no GELLO yet."""
        latest = self._gello_latest
        if latest is None:
            return None
        return [abs(latest[i] - self.init_pose[i]) for i in range(len(UR_JOINT_ORDER))]

    def _alignment_report(self, errs: list[float]) -> str:
        """Human-readable per-joint alignment report (worst joint first)."""
        worst = max(range(len(errs)), key=lambda i: errs[i])
        per = ", ".join(
            f"{UR_JOINT_SHORT[i]}={errs[i]:.2f}" for i in range(len(errs))
        )
        return (
            f"max err {errs[worst]:.3f} rad at {UR_JOINT_SHORT[worst]} "
            f"(tol {self.alignment_tolerance:.2f}, hard {self.alignment_hard_limit:.2f}) "
            f"| per-joint: {per}"
        )

    # ---- Operator-gate Trigger service callbacks (init_align mode) --------
    def _on_proceed(self, request, response):
        """GATE 1: authorize move-to-init. GATE 2: hand over IF within tolerance."""
        if self._stage == "init":
            self._proceed_init = True
            response.success = True
            response.message = "Authorized: moving the arm to the init pose."
            return response
        if self._stage == "wait_align":
            errs = self._alignment_errors()
            if errs is None:
                response.success = False
                response.message = "No GELLO pose yet; cannot check alignment."
                return response
            report = self._alignment_report(errs)
            over_tol = [
                UR_JOINT_ORDER[i] for i, e in enumerate(errs)
                if e > self.alignment_tolerance
            ]
            if not over_tol:
                self._handover = True
                response.success = True
                response.message = f"Aligned within tolerance ({report}). Handing over."
                return response
            over_hard = [
                UR_JOINT_ORDER[i] for i, e in enumerate(errs)
                if e > self.alignment_hard_limit
            ]
            response.success = False
            if over_hard:
                response.message = (
                    f"REFUSED — too far ({report}). Beyond HARD limit: {over_hard}. "
                    "Re-align these joints and call proceed again."
                )
            else:
                response.message = (
                    f"NOT within tolerance ({report}). Off joints: {over_tol}. "
                    "Re-align, OR if you accept this offset call ~/override_follow."
                )
            self.get_logger().warn(f"proceed @ wait_align: {response.message}")
            return response
        response.success = False
        response.message = f"Not awaiting authorization (stage={self._stage})."
        return response

    def _on_override_follow(self, request, response):
        """GATE 2 override: hand over despite an offset, but only within hard limit."""
        if self._stage != "wait_align":
            response.success = False
            response.message = (
                f"override_follow only valid during alignment (stage={self._stage})."
            )
            return response
        errs = self._alignment_errors()
        if errs is None:
            response.success = False
            response.message = "No GELLO pose yet; cannot check alignment."
            return response
        report = self._alignment_report(errs)
        over_hard = [
            UR_JOINT_ORDER[i] for i, e in enumerate(errs)
            if e > self.alignment_hard_limit
        ]
        if over_hard:
            response.success = False
            response.message = (
                f"REFUSED even with override — {over_hard} beyond HARD limit "
                f"{self.alignment_hard_limit:.2f} rad ({report}). Re-align."
            )
            self.get_logger().error(f"override REFUSED: {response.message}")
            return response
        self._handover = True
        self._overridden = True
        response.success = True
        response.message = (
            f"OVERRIDE accepted ({report}). Handing over; the arm will SLEW to the "
            "GELLO pose at the bridge's rate-limited speed. Keep clear."
        )
        self.get_logger().warn(f"override accepted: {response.message}")
        return response

    def _on_abort(self, request, response):
        """'2) 정지': abort the handshake (fail-safe; streaming never starts)."""
        self._abort = True
        response.success = True
        response.message = (
            "ABORT received — handshake will stop; no controller switch, the arm "
            "holds position. (E-STOP / Ctrl-C for a hard stop during motion.)"
        )
        self.get_logger().warn("Operator ABORT: stopping handshake (fail-safe).")
        return response

    def _on_check_alignment(self, request, response):
        """'5) 차이 계산': report GELLO vs init_pose per-joint error (no motion)."""
        errs = self._alignment_errors()
        if errs is None:
            response.success = False
            response.message = "No GELLO pose received yet; cannot compute difference."
            return response
        report = self._alignment_report(errs)
        within_tol = max(errs) <= self.alignment_tolerance
        # success reflects ALIGNMENT (so the console shows ✅ only when aligned).
        response.success = within_tol
        response.message = (
            ("ALIGNED (within tolerance) — " if within_tol else "not yet aligned — ")
            + report
        )
        return response

    def _on_go_home(self, request, response):
        """'4) 홈으로': request re-sending the arm to init_pose (before streaming)."""
        if self._stage == "streaming":
            response.success = False
            response.message = (
                "Already streaming; go_home is unavailable (Ctrl-C and relaunch to "
                "re-home)."
            )
            return response
        self._go_home_requested = True
        response.success = True
        response.message = (
            f"Go-home requested — arm will move to init pose {self.init_pose} "
            f"over {self.trajectory_duration:.0f}s. Keep clear."
        )
        self.get_logger().warn("Operator GO-HOME: re-sending init-pose trajectory.")
        return response

    # ---------------------------------------------------------------------
    def _wait_for_operator(self, flag_name: str, prompt: str) -> bool:
        """Spin (servicing gate calls + logging live error) until ``flag_name`` set."""
        self.get_logger().warn(prompt)
        deadline = (
            None
            if self.alignment_timeout <= 0.0
            else time.monotonic() + self.alignment_timeout
        )
        while rclpy.ok() and not getattr(self, flag_name):
            if self._abort:
                self.get_logger().error(
                    "Aborted by operator; not switching controllers (fail-safe)."
                )
                return False
            # '4) 홈으로': re-send the init-pose trajectory on request (blocks ~5s
            # while the arm moves, then resumes waiting). Only meaningful before
            # streaming, while the scaled trajectory controller is still active.
            if self._go_home_requested:
                self._go_home_requested = False
                self._send_trajectory(self.init_pose, "init pose (go-home)")
                continue
            if deadline is not None and time.monotonic() > deadline:
                self.get_logger().error(
                    f"No operator authorization within {self.alignment_timeout:.0f}s; "
                    "aborting (fail-safe, no switch)."
                )
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
            # During alignment, log the live per-joint error to guide the operator.
            if self._stage == "wait_align":
                errs = self._alignment_errors()
                if errs is not None:
                    self.get_logger().info(
                        "align: " + self._alignment_report(errs),
                        throttle_duration_sec=1.5,
                    )
        return bool(getattr(self, flag_name))

    # ---------------------------------------------------------------------
    def _converge_and_handover(self) -> bool:
        """CHASE the live GELLO pose until the arm agrees with it, then allow
        handover. Returns True only once |live_gello - actual| <= chase_tol on
        every joint, sustained chase_dwell_s. Fail-safe False on abort, timeout,
        or a gap beyond chase_hard_limit.

        This replaces the single frozen-snapshot move: because the leader is a
        moving target, we re-read it every iteration and only hand over when the
        follower has genuinely caught up to where the leader NOW is and the
        leader has settled. If the operator keeps moving GELLO, the gate holds
        (no switch) rather than letting a stale gap snap when streaming begins.
        """
        deadline = (
            None
            if self.chase_timeout_s <= 0.0
            else time.monotonic() + self.chase_timeout_s
        )
        err_samples: deque[float] = deque(maxlen=5)
        dwell_start: float | None = None
        # GELLO message time captured when the dwell begins. Completing the dwell
        # additionally requires that a genuinely NEW sample arrived since then, so
        # a stream that dies at (or just before) convergence cannot pass the dwell
        # on frozen-but-not-yet-stale samples — the freshness window
        # (gello_staleness_s, 0.5s) can otherwise exceed the dwell (0.4s).
        dwell_msg_time: float | None = None
        warned_wait = False
        while rclpy.ok():
            if self._abort:
                self.get_logger().error(
                    "Aborted by operator during convergence; no switch (fail-safe)."
                )
                return False
            if deadline is not None and time.monotonic() > deadline:
                self.get_logger().error(
                    f"Did not converge within {self.chase_timeout_s:.0f}s "
                    "(leader still moving / never settled?); aborting "
                    "(fail-safe, no switch)."
                )
                return False

            rclpy.spin_once(self, timeout_sec=0.05)

            # Need a FRESH leader sample and a known arm pose to measure the gap.
            if (
                not self._gello_fresh()
                or self._gello_latest is None
                or self._actual_pose is None
            ):
                dwell_start = None
                err_samples.clear()
                if not warned_wait:
                    self.get_logger().warn(
                        "Waiting for a fresh GELLO sample + robot /joint_states "
                        "before convergence (a stale/dead leader will NOT hand "
                        "over).",
                        throttle_duration_sec=2.0,
                    )
                    warned_wait = True
                continue

            gap = [
                abs(self._gello_latest[i] - self._actual_pose[i])
                for i in range(len(UR_JOINT_ORDER))
            ]
            max_gap = max(gap)
            worst = max(range(len(gap)), key=lambda i: gap[i])

            # SAFETY: never auto-chase an enormous gap (wraparound / mis-pose).
            if max_gap > self.chase_hard_limit:
                per = ", ".join(
                    f"{UR_JOINT_SHORT[i]}={gap[i]:.2f}" for i in range(len(gap))
                )
                self.get_logger().error(
                    f"Live gap {max_gap:.2f} rad at {UR_JOINT_SHORT[worst]} "
                    f"exceeds chase_hard_limit {self.chase_hard_limit:.2f} rad "
                    f"({per}); REFUSING to auto-chase (fail-safe, no switch). "
                    "Re-pose the GELLO leader closer to the robot and relaunch."
                )
                return False

            if max_gap <= self.chase_tol:
                # Converged this sample — require it to HOLD for the dwell using a
                # median filter so a lone Dynamixel spike can't falsely pass.
                err_samples.append(max_gap)
                if dwell_start is None:
                    dwell_start = time.monotonic()
                    dwell_msg_time = self._last_gello_msg_time
                held = time.monotonic() - dwell_start
                # A LIVE stream (~30 Hz) delivers many new samples over the dwell;
                # a dead stream delivers none. Requiring the message clock to have
                # advanced guarantees the dwell was verified against fresh data,
                # not a frozen pose that merely looks "still".
                new_sample_during_dwell = (
                    self._last_gello_msg_time is not None
                    and dwell_msg_time is not None
                    and self._last_gello_msg_time > dwell_msg_time
                )
                if (
                    len(err_samples) >= 3
                    and median(err_samples) <= self.chase_tol
                    and held >= self.chase_dwell_s
                    and new_sample_during_dwell
                ):
                    self.get_logger().info(
                        f"Converged: max gap {max_gap:.4f} rad held "
                        f"{held:.2f}s (<= {self.chase_tol} for {self.chase_dwell_s}s). "
                        "Handing over to streaming."
                    )
                    return True
                # else keep dwelling (arm holds; no new trajectory)
                continue

            # Not converged: leader is elsewhere (drifted or actively moving).
            # Chase it with a catch-up trajectory sized from the measured gap so
            # the move speed stays within budget regardless of gap magnitude.
            dwell_start = None
            err_samples.clear()
            warned_wait = False
            dur = max(max_gap / self.chase_v_budget, self.min_traj_duration)
            self.get_logger().info(
                f"Chasing live GELLO: gap {max_gap:.3f} rad at "
                f"{UR_JOINT_SHORT[worst]} -> {dur:.2f}s catch-up."
            )
            if not self._send_trajectory(
                list(self._gello_latest), "live GELLO pose", duration=dur
            ):
                return False
            # Loop re-evaluates: the leader may have moved again while we moved.

    # ---------------------------------------------------------------------
    def _switch_controllers(self) -> bool:
        """STRICT switch: activate target, deactivate source. True on success."""
        if not self._switch_client.wait_for_service(
            timeout_sec=_DISCOVERY_TIMEOUT_S
        ):
            self.get_logger().error(
                "/controller_manager/switch_controller service not available; "
                "aborting (fail-safe, source controller left active)"
            )
            return False

        req = SwitchController.Request()
        # Humble uses 'activate'/'deactivate' field names.
        req.activate_controllers = [self.target_controller]
        req.deactivate_controllers = [self.source_controller]
        req.strictness = _STRICT
        req.activate_asap = True
        req.timeout = Duration(sec=5, nanosec=0)

        self.get_logger().info(
            f"Switching controllers (STRICT): activate={self.target_controller}, "
            f"deactivate={self.source_controller}..."
        )
        future = self._switch_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is None or not resp.ok:
            self.get_logger().error(
                "Controller switch FAILED; source controller left active "
                "(fail-safe). Bridge must NOT stream."
            )
            return False

        self.get_logger().info(
            f"Controller switch OK: {self.target_controller} active. "
            "Bridge may now stream."
        )
        return True

    # ---------------------------------------------------------------------
    def run(self) -> bool:
        """Execute the full handshake. True only if the switch succeeded."""
        if not self._wait_for_gello_target():
            self.get_logger().error(
                "Did not obtain a GELLO target; aborting (no switch)."
            )
            return False
        if not self._wait_for_source_active():
            return False

        if self.start_mode == "init_align":
            # GATE 1: operator authorizes the move to the fixed init pose.
            if not self._wait_for_operator(
                "_proceed_init",
                "GATE 1 — ready to move the arm to the INIT POSE "
                f"{self.init_pose}. In the operator console (2nd terminal) press "
                "[1] 진행 when the workspace is clear ([2] 정지 to abort).",
            ):
                return False
            if not self._send_trajectory(self.init_pose, "init pose"):
                return False
            # GATE 2: operator aligns GELLO, then authorizes handover. The
            # ~/proceed and ~/override_follow callbacks set _handover.
            self._stage = "wait_align"
            if not self._wait_for_operator(
                "_handover",
                "GATE 2 — the arm is at the init pose. Now move the GELLO leader to "
                "MATCH it (live per-joint error printed below). In the console press "
                "[1] 진행 to hand over when aligned; if a joint is off you'll get a "
                "report and can re-align, or press [3] 강제 진행 to accept the offset "
                "([2] 정지 to abort).",
            ):
                return False
            self._stage = "streaming"
        else:  # "gello" (default): CHASE the live leader until converged.
            # (Was: one blind trajectory to the FIRST-frozen pose, which snapped
            # when streaming began because the leader had drifted/moved since.)
            if not self._converge_and_handover():
                return False

        if not self._switch_controllers():
            return False
        # Integrated launch: the streaming bridge was pre-spawned PAUSED, so the
        # switch alone does not start teleop — release it now (after the switch,
        # never before, so it can't stream into an inactive controller).
        if self.resume_bridge:
            self._resume_bridge_after_switch()
        return True

    # ---------------------------------------------------------------------
    def _resume_bridge_after_switch(self) -> None:
        """Release the pre-spawned paused bridge to begin streaming.

        The bridge's ~/resume is alignment-gated (|gello-actual| <=
        resume_align_tol); a just-converged arm passes. If resume never
        succeeds, forward_position_controller stays ACTIVE holding the arrived
        pose (safe) but no teleop streams — surfaced loudly with the manual
        recovery command. Handshake still counts as succeeded (switch was OK).
        """
        svc = self.bridge_resume_service
        manual = f"ros2 service call {svc} std_srvs/srv/Trigger"
        if not self._resume_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                f"Bridge resume service {svc} not available; "
                f"{self.target_controller} is ACTIVE and HOLDING but teleop is "
                f"NOT streaming. Recover manually: {manual}"
            )
            return
        for attempt in range(3):
            future = self._resume_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            resp = future.result()
            if resp is not None and resp.success:
                self.get_logger().info(
                    f"Bridge resumed ({resp.message}); teleop is now streaming."
                )
                return
            msg = resp.message if resp is not None else "no response"
            self.get_logger().warn(
                f"Bridge resume attempt {attempt + 1}/3 not accepted: {msg}"
            )
            time.sleep(0.5)
        self.get_logger().error(
            "Bridge did NOT resume after 3 attempts (leader likely not aligned "
            f"within resume_align_tol). {self.target_controller} is ACTIVE and "
            f"HOLDING; no teleop. Align the GELLO leader, then: {manual}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GelloMoveToStart()
    ok = False
    try:
        ok = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if ok:
            node.get_logger().info(
                "Move-to-start handshake complete; shutting down."
            )
        else:
            node.get_logger().error(
                "Move-to-start handshake FAILED; scaled_joint_trajectory_"
                "controller remains active. Do NOT start the bridge."
            )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Propagate a non-zero exit code on failure so a launch on_exit /
        # orchestrator can distinguish a failed handshake from success.
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
