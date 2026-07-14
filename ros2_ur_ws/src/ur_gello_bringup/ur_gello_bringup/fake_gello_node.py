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
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray
from std_srvs.srv import Trigger

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
            f"amplitude={self._amplitude:.3f} rad, "
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

    def _current_output(self) -> tuple[list[float], float]:
        """Compute the instantaneous (arm_pose, gripper) for the ACTIVE mode.

        Read-only: does NOT advance time or mutate mode. Used by the control
        services so a mode switch can be made jump-free from wherever we are.
        """
        if self._mode == _MODE_SWEEP:
            return self._sweep_pose(), self._sweep_gripper()
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
        """Resume the sine sweep, re-centred on the held pose (no output jump)."""
        pose, gripper = self._current_output()
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
            "fake_gello SWEEPING, re-centred on "
            f"[{', '.join(f'{v:.3f}' for v in pose)}] (continuous)."
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
            positions = self._sweep_pose()
            gripper_percent = self._sweep_gripper()
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
