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
"""

import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from controller_manager_msgs.srv import ListControllers, SwitchController
from sensor_msgs.msg import JointState
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
        self.trajectory_duration = float(
            self.declare_parameter("trajectory_duration", 5.0).value
        )
        self.arrival_tolerance = float(
            self.declare_parameter("arrival_tolerance", 0.05).value
        )
        # How long to wait for the source controller to become 'active'. On the
        # real UR the controller_stopper keeps motion controllers inactive until
        # the External Control program is PLAYING on the pendant, so this wait
        # gives the operator time to press Play.
        self.activation_timeout = float(
            self.declare_parameter("activation_timeout", 120.0).value
        )

        # --- State -------------------------------------------------------
        self._gello_target: list[float] | None = None

        # --- ROS interfaces ----------------------------------------------
        self._sub = self.create_subscription(
            JointState, "/gello/joint_states", self._on_joint_state, 10
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

        self.get_logger().info(
            "gello_move_to_start started | "
            f"source={self.source_controller} "
            f"target={self.target_controller} "
            f"trajectory_duration={self.trajectory_duration} "
            f"arrival_tolerance={self.arrival_tolerance}"
        )

    # ---------------------------------------------------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        """Capture the FIRST GELLO message that contains all UR joints (by name)."""
        if self._gello_target is not None:
            return
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
        self._gello_target = [float(name_to_pos[j]) for j in UR_JOINT_ORDER]
        self.get_logger().info(
            f"Captured GELLO target pose: {self._gello_target}"
        )

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
    def _send_trajectory(self) -> bool:
        """Send one FollowJointTrajectory goal to the GELLO pose; wait SUCCEEDED."""
        assert self._gello_target is not None

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
        point.positions = list(self._gello_target)
        point.velocities = [0.0] * len(UR_JOINT_ORDER)
        sec = int(self.trajectory_duration)
        nanosec = int((self.trajectory_duration - sec) * 1e9)
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
            f"Sending trajectory to GELLO pose over {self.trajectory_duration}s "
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

        self.get_logger().info("Arm arrived at GELLO pose (trajectory SUCCEEDED).")
        return True

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
        if not self._send_trajectory():
            return False
        if not self._switch_controllers():
            return False
        return True


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
