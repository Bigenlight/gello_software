"""Automated public-interface validation of the remote policy with mock UR7e."""

from __future__ import annotations

import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from .fake_ur7e_validation_logic import FakeUr7eVerdict


class RemotePolicyFakeUr7eValidator(Node):
    def __init__(self):
        super().__init__("remote_policy_fake_ur7e_validator")
        self.declare_parameter("timeout_s", 45.0)
        self.declare_parameter("tolerance_rad", 0.08)
        self._timeout = float(self.get_parameter("timeout_s").value)
        tolerance = float(self.get_parameter("tolerance_rad").value)
        if self._timeout <= 0:
            raise ValueError("timeout_s must be positive")
        self._verdict = FakeUr7eVerdict(tolerance=tolerance)
        self._started = time.monotonic()
        self._pending = None
        self._hold_requested = False
        self.exit_code = None
        self.create_subscription(
            Float64MultiArray, "/forward_position_controller/commands",
            self._on_command, 100,
        )
        self.create_subscription(JointState, "/joint_states", self._on_joints, 100)
        self.create_subscription(
            String, "/policy_leader_node/state", self._on_policy_state, 10,
        )
        self._start_client = self.create_client(Trigger, "/policy_leader_node/start_execution")
        self._hold_client = self.create_client(Trigger, "/policy_leader_node/hold")
        self.create_timer(0.05, self._tick)

    def _on_command(self, message):
        self._verdict.observe_command(time.monotonic(), message.data)

    def _on_joints(self, message):
        self._verdict.observe_joint_state(message.name, message.position)

    def _on_policy_state(self, message):
        self._verdict.observe_policy_state(message.data)

    def _fail(self, reason):
        if self.exit_code is None:
            self.get_logger().error(f"FAIL: {reason}")
            self.exit_code = 1
            rclpy.shutdown()

    def _call(self, client, label):
        if not client.service_is_ready():
            return False
        self.get_logger().info(f"calling {label}")
        self._pending = (label, client.call_async(Trigger.Request()))
        return True

    def _tick(self):
        now = time.monotonic()
        if self._verdict.failure:
            self._fail(self._verdict.failure)
            return
        if now - self._started > self._timeout:
            self._fail(
                f"validation timed out in phase {self._verdict.phase}; "
                f"post_arm_finite={self._verdict.post_arm_finite}, "
                f"tracking_seen={self._verdict.tracking_seen}"
            )
            return
        if self._pending is not None:
            label, future = self._pending
            if not future.done():
                return
            self._pending = None
            try:
                response = future.result()
            except Exception as exc:
                self._fail(f"{label} call failed: {exc}")
                return
            if not response.success:
                self._fail(f"{label} rejected: {response.message}")
                return
            if label == "start_execution":
                self._verdict.mark_started()
            else:
                self._verdict.mark_hold(now)
            return

        if self._verdict.phase == "initial_hold":
            if self._verdict.initial_hold_ready(now):
                self._call(self._start_client, "start_execution")
        elif self._verdict.phase == "executing":
            if self._verdict.execution_ready_for_hold() and not self._hold_requested:
                self._hold_requested = self._call(self._hold_client, "hold")
        elif self._verdict.passed(now):
            self.get_logger().info(
                "PASS: HOLD stable, start_execution succeeded, finite policy command "
                "tracked by mock joints, and post-hold commands remained stable/continuous; "
                f"policy displacement diagnostic={self._verdict.max_displacement:.6f} rad "
                "(not a pass gate). Roundtrip smoke is the inference gate."
            )
            self.exit_code = 0
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RemotePolicyFakeUr7eValidator()
    try:
        rclpy.spin(node)
    finally:
        code = 1 if node.exit_code is None else node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
