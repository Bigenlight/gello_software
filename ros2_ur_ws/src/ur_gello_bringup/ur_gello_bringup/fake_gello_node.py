#!/usr/bin/env python3
"""ROS2 TEST node that fakes a GELLO leader for robotless RViz testing.

This node has NO hardware dependency. It publishes exactly the same topics a real
``gello_publisher`` would, driving them with a slow, safe sine sweep so you can
verify the whole visualization pipeline (RViz, robot_state_publisher, any bridge)
without a physical GELLO or robot attached.

Use this ONLY for visualizing the pipeline; it does not read any real leader arm.

Beyond the default sine sweep it exposes a small control surface so a test
harness can rehearse the pause / resume-chase feature deterministically without a
real robot (see check_pause_resume_sim.sh):

  * ``~/set_pose`` (Float64MultiArray, 6 or 7 values): jump the output to that
    pose (6 arm joints, optional 7th = gripper) and freeze there (hold mode).
  * ``~/hold``  (Trigger): freeze at the CURRENT instantaneous output pose.
  * ``~/sweep`` (Trigger): resume the sine sweep, re-centred on the currently
    held pose so the output does NOT jump.
  * ``~/collapse`` (Trigger): over ``collapse_duration_s`` glide toward a limp
    ``droop_pose`` (and gripper -> closed), then hold. Emulates a leader that
    goes slack while the bridge is paused; the bridge must ignore it.

In EVERY mode the node keeps publishing continuously at ``rate_hz``.

Beyond the sine sweep, a ``pattern`` parameter selects which deterministic
synthetic motion drives the "moving" mode (the mode the ``~/hold`` /
``~/sweep`` / ``~/collapse`` control surface above rehearses around). This is
for the EEF (end-effector delta) bridge mock harness
(``ur7e_gello_eef_mock.launch.py``, P4/P5 in
``docs/ros2/GELLO_UR7E_EEF_TELEOP_PLAN.md``), so a reviewer can reproduce
zero-jump / gate-reject / singularity behaviour without real hardware:

  * ``sweep``  (DEFAULT, unchanged) -- the original sine sweep around
    ``START_POSE``. Every existing launch / test that does not set
    ``pattern`` keeps today's exact behaviour.
  * ``hold``  -- frozen at ``START_POSE`` from the first tick (gripper frozen
    too). A genuinely still leader, e.g. for the G5 leader-quasi-still gate.
  * ``offset_hold``  -- frozen at ``START_POSE`` plus a constant per-joint
    offset (``offset_hold_delta_rad``), i.e. a leader that reports a pose
    deliberately far from the robot's actual one from the very first
    message. For rehearsing the disagreement / anchor gates (G4) and the
    ``gello_move_to_start`` handshake under a mismatched leader.
  * ``line_xyz``  -- the TCP reciprocates along one Cartesian axis (a sine of
    amplitude ``line_xyz_amplitude_m``, period ``line_xyz_period_s``) around
    ``fk(START_POSE)``, solved back to joint space via ``ur_kin.ik_numeric``.
    A smooth, well-conditioned EEF exercise.
  * ``wrist_singularity``  -- every joint holds at ``START_POSE`` except
    wrist_2 (``q5``), which ramps linearly from its start value to exactly 0
    over ``wrist_singularity_sweep_s`` seconds and then holds there (the UR
    wrist-alignment singularity, wrist_1/wrist_3 axes coincident).
  * ``full_rotation``  -- every joint holds at ``START_POSE`` except wrist_3,
    which turns continuously from the anchor angle through +2*pi (0->360 deg)
    over ``full_rotation_period_s`` seconds and then holds at the +2*pi pose
    (physically identical orientation to the anchor). No wrap-around cap.
  * ``step``  -- one joint (``step_joint_index``, default wrist_1) square-waves
    between ``START_POSE`` and ``START_POSE + step_size_rad`` every
    ``step_period_s`` seconds: a repeated discrete jump for exercising the
    downstream slew-rate clamp / zero-jump behaviour.

Every pattern generates arm joints in ``UR_JOINT_NAMES`` (== UR_JOINT_ORDER)
order. Unknown/misspelled ``pattern`` values fall back to ``sweep`` with a
logged warning (never a silent no-op).
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray
from std_srvs.srv import Trigger

from ur_gello_bringup.ur_kin import fk, ik_numeric

# UR joint names, in order (shared contract with the real gello_publisher).
UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# GELLO/UR calibration start pose (6 arm joints, radians).
START_POSE = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

# Modes the fake leader can be in.
_MODE_SWEEP = "sweep"
_MODE_HOLD = "hold"
_MODE_COLLAPSE = "collapse"

# Supported values of the 'pattern' parameter. "sweep" is the pre-existing
# default sine sweep; the rest are new deterministic EEF-mock test signals.
_PATTERN_SWEEP = "sweep"
_PATTERN_HOLD = "hold"
_PATTERN_OFFSET_HOLD = "offset_hold"
_PATTERN_LINE_XYZ = "line_xyz"
_PATTERN_WRIST_SINGULARITY = "wrist_singularity"
_PATTERN_FULL_ROTATION = "full_rotation"
_PATTERN_STEP = "step"
_VALID_PATTERNS = (
    _PATTERN_SWEEP,
    _PATTERN_HOLD,
    _PATTERN_OFFSET_HOLD,
    _PATTERN_LINE_XYZ,
    _PATTERN_WRIST_SINGULARITY,
    _PATTERN_FULL_ROTATION,
    _PATTERN_STEP,
)

# UR_JOINT_ORDER indices (shared convention with ur_kin / gello_ur_bridge_node).
_WRIST_1 = 3
_WRIST_2 = 4
_WRIST_3 = 5


class FakeGello(Node):
    """Publishes a synthetic GELLO joint state + gripper width for RViz testing."""

    def __init__(self):
        super().__init__("fake_gello")

        # --- Declare + read parameters ---
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("amplitude_rad", 0.4)
        # Seconds to ramp from the current pose to the droop pose on ~/collapse.
        self.declare_parameter("collapse_duration_s", 1.5)
        # Limp "arm went slack" pose the collapse ramps toward.
        self.declare_parameter("droop_pose", [0.0, -0.35, 2.60, -2.60, -1.57, 0.0])

        # --- Pattern selection + per-pattern parameters (all default-off: the
        # default pattern is "sweep", the pre-existing sine sweep) ---
        self.declare_parameter("pattern", _PATTERN_SWEEP)
        self.declare_parameter("offset_hold_delta_rad", 0.5)
        self.declare_parameter("line_xyz_amplitude_m", 0.05)
        self.declare_parameter("line_xyz_period_s", 8.0)
        self.declare_parameter("line_xyz_axis", "x")
        self.declare_parameter("wrist_singularity_sweep_s", 8.0)
        self.declare_parameter("full_rotation_period_s", 8.0)
        self.declare_parameter("step_period_s", 2.0)
        self.declare_parameter("step_size_rad", 0.3)
        self.declare_parameter("step_joint_index", _WRIST_1)

        pattern = self.get_parameter("pattern").get_parameter_value().string_value
        if pattern not in _VALID_PATTERNS:
            self.get_logger().warn(
                f"pattern='{pattern}' is not one of {_VALID_PATTERNS}; "
                "falling back to 'sweep' (the pre-existing default)."
            )
            pattern = _PATTERN_SWEEP
        self._pattern = pattern

        self._offset_hold_delta = (
            self.get_parameter("offset_hold_delta_rad").get_parameter_value().double_value
        )

        self._line_xyz_amplitude_m = (
            self.get_parameter("line_xyz_amplitude_m").get_parameter_value().double_value
        )
        self._line_xyz_period_s = (
            self.get_parameter("line_xyz_period_s").get_parameter_value().double_value
        )
        if self._line_xyz_period_s <= 0.0:
            self.get_logger().warn(
                f"line_xyz_period_s={self._line_xyz_period_s} invalid; using 8.0"
            )
            self._line_xyz_period_s = 8.0
        line_xyz_axis = (
            self.get_parameter("line_xyz_axis").get_parameter_value().string_value
        )
        if line_xyz_axis not in ("x", "y", "z"):
            self.get_logger().warn(
                f"line_xyz_axis='{line_xyz_axis}' invalid; falling back to 'x'."
            )
            line_xyz_axis = "x"
        self._line_xyz_axis_idx = {"x": 0, "y": 1, "z": 2}[line_xyz_axis]

        self._wrist_singularity_sweep_s = (
            self.get_parameter("wrist_singularity_sweep_s")
            .get_parameter_value()
            .double_value
        )
        if self._wrist_singularity_sweep_s <= 0.0:
            self.get_logger().warn(
                "wrist_singularity_sweep_s="
                f"{self._wrist_singularity_sweep_s} invalid; using 8.0"
            )
            self._wrist_singularity_sweep_s = 8.0

        self._full_rotation_period_s = (
            self.get_parameter("full_rotation_period_s").get_parameter_value().double_value
        )
        if self._full_rotation_period_s <= 0.0:
            self.get_logger().warn(
                f"full_rotation_period_s={self._full_rotation_period_s} invalid; "
                "using 8.0"
            )
            self._full_rotation_period_s = 8.0

        self._step_period_s = (
            self.get_parameter("step_period_s").get_parameter_value().double_value
        )
        if self._step_period_s <= 0.0:
            self.get_logger().warn(
                f"step_period_s={self._step_period_s} invalid; using 2.0"
            )
            self._step_period_s = 2.0
        self._step_size_rad = (
            self.get_parameter("step_size_rad").get_parameter_value().double_value
        )
        step_joint_index = (
            self.get_parameter("step_joint_index").get_parameter_value().integer_value
        )
        if not (0 <= step_joint_index < len(START_POSE)):
            self.get_logger().warn(
                f"step_joint_index={step_joint_index} out of range; using {_WRIST_1}."
            )
            step_joint_index = _WRIST_1
        self._step_joint_index = step_joint_index

        # line_xyz IK seed, carried tick-to-tick for continuity; primed lazily
        # (numeric IK on the anchor pose) on first use in _line_xyz_pose().
        self._line_xyz_anchor_T = None
        self._line_xyz_seed = list(START_POSE)

        rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value
        self._amplitude = (
            self.get_parameter("amplitude_rad").get_parameter_value().double_value
        )
        self._collapse_duration_s = (
            self.get_parameter("collapse_duration_s")
            .get_parameter_value()
            .double_value
        )
        droop = list(
            self.get_parameter("droop_pose").get_parameter_value().double_array_value
        )
        if len(droop) != len(START_POSE):
            self.get_logger().warn(
                f"droop_pose has {len(droop)} values (need {len(START_POSE)}); "
                "falling back to a safe default."
            )
            droop = [0.0, -0.35, 2.60, -2.60, -1.57, 0.0]
        self._droop_pose = droop
        if self._collapse_duration_s <= 0.0:
            self.get_logger().warn(
                f"collapse_duration_s={self._collapse_duration_s} invalid; using 1.5"
            )
            self._collapse_duration_s = 1.5

        if rate_hz <= 0.0:
            self.get_logger().warn(
                f"rate_hz={rate_hz} invalid; falling back to 30.0 Hz."
            )
            rate_hz = 30.0
        self._dt = 1.0 / rate_hz
        self._t = 0.0

        # --- Mode state --------------------------------------------------
        # Sweep center: the sine oscillates around this. Default = START_POSE.
        self._sweep_center = list(START_POSE)
        # Gripper sine center (output = center + 0.5*sin(...)), default 0.5.
        self._gripper_center = 0.5
        # Held output (used in HOLD mode).
        self._hold_pose = list(START_POSE)
        self._hold_gripper = 0.5
        # Collapse ramp bookkeeping.
        self._collapse_t0 = 0.0
        self._collapse_from_pose = list(START_POSE)
        self._collapse_from_gripper = 0.5
        # Start in the classic default sweep so existing use is unchanged.
        self._mode = _MODE_SWEEP

        # --- Publishers (same topics as the real gello_publisher) ---
        self._js_pub = self.create_publisher(JointState, "/gello/joint_states", 10)
        self._gripper_pub = self.create_publisher(
            Float32, "/gripper/gripper_client/target_gripper_width_percent", 10
        )

        # --- Control surface (test harness) ------------------------------
        self._set_pose_sub = self.create_subscription(
            Float64MultiArray, "~/set_pose", self._on_set_pose, 10
        )
        self._hold_srv = self.create_service(Trigger, "~/hold", self._on_hold)
        self._sweep_srv = self.create_service(Trigger, "~/sweep", self._on_sweep)
        self._collapse_srv = self.create_service(
            Trigger, "~/collapse", self._on_collapse
        )

        # --- Timer ---
        self._timer = self.create_timer(self._dt, self._on_timer)

        self.get_logger().info(
            f"fake_gello started (TEST ONLY, no hardware) at {rate_hz:.1f} Hz, "
            f"pattern='{self._pattern}', amplitude={self._amplitude:.3f} rad, "
            f"collapse_duration_s={self._collapse_duration_s:.2f}."
        )

    # ---------------------------------------------------------------------
    # Output model
    # ---------------------------------------------------------------------
    def _sweep_pose(self) -> list[float]:
        """Sine-sweep arm pose at the current sim time (around _sweep_center)."""
        t = self._t
        return [
            self._sweep_center[i]
            + self._amplitude * math.sin(2.0 * math.pi * 0.1 * t + i * 0.5)
            for i in range(len(self._sweep_center))
        ]

    def _sweep_gripper(self) -> float:
        """Sine-sweep gripper value at the current sim time (clamped 0..1)."""
        g = self._gripper_center + 0.5 * math.sin(2.0 * math.pi * 0.2 * self._t)
        return min(1.0, max(0.0, g))

    # ---------------------------------------------------------------------
    # Pattern dispatch (feeds the _MODE_SWEEP / "moving" mode).
    #
    # pattern == "sweep" (the default) is BIT-IDENTICAL to the original
    # behaviour: it calls _sweep_pose()/_sweep_gripper() unchanged, so every
    # existing launch/test that never sets `pattern` sees no regression.
    # ---------------------------------------------------------------------
    def _pattern_pose(self) -> list[float]:
        """Arm pose at the current sim time for the SELECTED pattern."""
        if self._pattern == _PATTERN_SWEEP:
            return self._sweep_pose()
        if self._pattern == _PATTERN_HOLD:
            return list(START_POSE)
        if self._pattern == _PATTERN_OFFSET_HOLD:
            return [v + self._offset_hold_delta for v in START_POSE]
        if self._pattern == _PATTERN_LINE_XYZ:
            return self._line_xyz_pose()
        if self._pattern == _PATTERN_WRIST_SINGULARITY:
            return self._wrist_singularity_pose()
        if self._pattern == _PATTERN_FULL_ROTATION:
            return self._full_rotation_pose()
        if self._pattern == _PATTERN_STEP:
            return self._step_pose()
        # Unreachable (pattern validated at startup) -- keep a safe fallback.
        return self._sweep_pose()

    def _pattern_gripper(self) -> float:
        """Gripper value at the current sim time for the SELECTED pattern.

        Only the default "sweep" pattern moves the gripper (unchanged sine);
        every other pattern is an arm-only kinematics exercise and holds the
        gripper still at 0.5 so it never confounds the EEF-mode observations.
        """
        if self._pattern == _PATTERN_SWEEP:
            return self._sweep_gripper()
        return 0.5

    def _line_xyz_pose(self) -> list[float]:
        """TCP reciprocates along one Cartesian axis around fk(START_POSE).

        Solved back to joint space with ur_kin.ik_numeric, seeded from the
        previous tick's solution for continuity. If IK ever fails to converge
        (should not happen for this small, well-conditioned excursion) the
        previous joint solution is held rather than publishing a stale/garbage
        pose.
        """
        if self._line_xyz_anchor_T is None:
            self._line_xyz_anchor_T = fk(START_POSE)
        delta = self._line_xyz_amplitude_m * math.sin(
            2.0 * math.pi * self._t / self._line_xyz_period_s
        )
        t_target = self._line_xyz_anchor_T.copy()
        t_target[self._line_xyz_axis_idx, 3] += delta
        q = ik_numeric(t_target, seed=self._line_xyz_seed)
        if q is None:
            # Hold the last good joint solution rather than jump/NaN.
            return list(self._line_xyz_seed)
        self._line_xyz_seed = list(q)
        return list(q)

    def _wrist_singularity_pose(self) -> list[float]:
        """wrist_2 (q5) ramps linearly from its start value to 0, then holds.

        All other joints stay at START_POSE. wrist_2 == 0 is the UR
        wrist-alignment singularity (wrist_1 / wrist_3 axes coincide).
        """
        p = self._t / self._wrist_singularity_sweep_s
        p = min(1.0, max(0.0, p))
        pose = list(START_POSE)
        pose[_WRIST_2] = START_POSE[_WRIST_2] * (1.0 - p)
        return pose

    def _full_rotation_pose(self) -> list[float]:
        """wrist_3 turns continuously 0->360 deg (no cap), then holds at +2*pi.

        All other joints stay at START_POSE.
        """
        p = self._t / self._full_rotation_period_s
        p = min(1.0, max(0.0, p))
        pose = list(START_POSE)
        pose[_WRIST_3] = START_POSE[_WRIST_3] + 2.0 * math.pi * p
        return pose

    def _step_pose(self) -> list[float]:
        """One joint square-waves between START_POSE and +step_size_rad.

        Toggles every step_period_s seconds -- a repeated discrete jump for
        exercising the downstream slew-rate clamp / zero-jump behaviour.
        """
        level = int(math.floor(self._t / self._step_period_s))
        pose = list(START_POSE)
        if level % 2 == 1:
            pose[self._step_joint_index] += self._step_size_rad
        return pose

    def _current_output(self) -> tuple[list[float], float]:
        """Compute the instantaneous (arm_pose, gripper) for the ACTIVE mode.

        Read-only: does NOT advance time or mutate mode. Used by the control
        services so a mode switch can be made jump-free from wherever we are.
        """
        if self._mode == _MODE_SWEEP:
            return self._pattern_pose(), self._pattern_gripper()
        if self._mode == _MODE_COLLAPSE:
            return self._collapse_output()
        # HOLD (and any unknown mode) -> the frozen values.
        return list(self._hold_pose), self._hold_gripper

    def _collapse_output(self) -> tuple[list[float], float]:
        """Linear ramp from the pose at collapse-start toward the droop pose."""
        elapsed = self._t - self._collapse_t0
        p = elapsed / self._collapse_duration_s
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        pose = [
            self._collapse_from_pose[i]
            + p * (self._droop_pose[i] - self._collapse_from_pose[i])
            for i in range(len(self._droop_pose))
        ]
        gripper = self._collapse_from_gripper + p * (1.0 - self._collapse_from_gripper)
        return pose, min(1.0, max(0.0, gripper))

    # ---------------------------------------------------------------------
    # Control-surface callbacks
    # ---------------------------------------------------------------------
    def _on_set_pose(self, msg: Float64MultiArray) -> None:
        """Jump output to the commanded pose and enter HOLD mode."""
        data = list(msg.data)
        n = len(START_POSE)
        if len(data) not in (n, n + 1):
            self.get_logger().warn(
                f"~/set_pose ignored: got {len(data)} values, expected {n} or "
                f"{n + 1} (6 arm joints, optional 7th gripper)."
            )
            return
        self._hold_pose = [float(v) for v in data[:n]]
        if len(data) == n + 1:
            self._hold_gripper = min(1.0, max(0.0, float(data[n])))
        else:
            # Keep whatever the gripper is showing right now (no jump).
            _, self._hold_gripper = self._current_output()
        self._mode = _MODE_HOLD
        self.get_logger().info(
            "~/set_pose -> HOLD at "
            f"[{', '.join(f'{v:.3f}' for v in self._hold_pose)}] "
            f"gripper={self._hold_gripper:.3f}."
        )

    def _on_hold(self, request, response):
        """Freeze at the current instantaneous output pose."""
        pose, gripper = self._current_output()
        self._hold_pose = pose
        self._hold_gripper = gripper
        self._mode = _MODE_HOLD
        response.success = True
        response.message = (
            "fake_gello HOLDING at "
            f"[{', '.join(f'{v:.3f}' for v in pose)}] gripper={gripper:.3f}."
        )
        self.get_logger().info(response.message)
        return response

    def _on_sweep(self, request, response):
        """Resume the active pattern's "moving" mode.

        For the default `pattern=="sweep"` this is BIT-IDENTICAL to the
        original behaviour: re-centre the sine on the held pose so the output
        does not jump. The other (non-sweep) patterns are pure functions of
        sim time `self._t` (which keeps advancing even while HOLD/COLLAPSE
        freeze the *output*), so resuming them just switches the mode back;
        no re-centring maths applies. Those patterns are meant to be launched
        and observed start-to-finish for EEF-mock testing, not toggled via
        this sine-oriented control surface, so a resume-time jump risk there
        is accepted and logged.
        """
        pose, gripper = self._current_output()
        if self._pattern == _PATTERN_SWEEP:
            t = self._t
            # Choose centers so the sine's current value maps exactly onto the
            # present output: output stays continuous across the switch.
            self._sweep_center = [
                pose[i] - self._amplitude * math.sin(2.0 * math.pi * 0.1 * t + i * 0.5)
                for i in range(len(pose))
            ]
            self._gripper_center = gripper - 0.5 * math.sin(2.0 * math.pi * 0.2 * t)
        self._mode = _MODE_SWEEP
        response.success = True
        response.message = (
            f"fake_gello RESUMING pattern='{self._pattern}' from "
            f"[{', '.join(f'{v:.3f}' for v in pose)}]"
            + (" (continuous)." if self._pattern == _PATTERN_SWEEP else ".")
        )
        self.get_logger().info(response.message)
        return response

    def _on_collapse(self, request, response):
        """Ramp toward the droop pose over collapse_duration_s, then hold."""
        pose, gripper = self._current_output()
        self._collapse_from_pose = pose
        self._collapse_from_gripper = gripper
        self._collapse_t0 = self._t
        self._mode = _MODE_COLLAPSE
        response.success = True
        response.message = (
            f"fake_gello COLLAPSING over {self._collapse_duration_s:.2f}s toward "
            f"[{', '.join(f'{v:.3f}' for v in self._droop_pose)}] (gripper->closed)."
        )
        self.get_logger().info(response.message)
        return response

    # ---------------------------------------------------------------------
    def _on_timer(self):
        # Resolve this tick's output for the active mode.
        if self._mode == _MODE_SWEEP:
            # pattern=="sweep" (default) dispatches straight to _sweep_pose()/
            # _sweep_gripper() -- bit-identical to the pre-`pattern` behaviour.
            positions = self._pattern_pose()
            gripper_percent = self._pattern_gripper()
        elif self._mode == _MODE_COLLAPSE:
            positions, gripper_percent = self._collapse_output()
            # Latch into HOLD once the ramp completes.
            if (self._t - self._collapse_t0) >= self._collapse_duration_s:
                self._hold_pose = list(positions)
                self._hold_gripper = gripper_percent
                self._mode = _MODE_HOLD
        else:  # HOLD
            positions = list(self._hold_pose)
            gripper_percent = self._hold_gripper

        now = self.get_clock().now().to_msg()

        js_msg = JointState()
        js_msg.header.stamp = now
        js_msg.header.frame_id = "base_link"
        js_msg.name = UR_JOINT_NAMES
        js_msg.position = positions
        self._js_pub.publish(js_msg)

        gripper_msg = Float32()
        gripper_msg.data = float(gripper_percent)
        self._gripper_pub.publish(gripper_msg)

        # Advance simulated time by one tick (never use wall-clock here).
        self._t += self._dt


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FakeGello()
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
