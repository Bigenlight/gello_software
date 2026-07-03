#!/usr/bin/env python3
"""Standalone ROS 2 control for a Robotiq 2F-85 wired to a UR (e-Series) tool port.

This node talks to the **Robotiq "Grippers" URCap** TCP socket server that the UR
controller exposes on ``ROBOT_IP:63352`` (PolyScope 5 only — PolyScope X does NOT
expose 63352). It reuses the repo's ``gello.robots.robotiq_gripper.RobotiqGripper``
ASCII-socket driver, so no MODBUS/serial/tool-communication bridge is needed and it
does NOT conflict with the UR ROS 2 arm driver (they use different channels).

It is deliberately GRIPPER-ONLY: it brings up no arm, no ros2_control, nothing else.

Interfaces exposed
------------------
1. Action  ``/robotiq_gripper_controller/gripper_cmd`` (control_msgs/action/GripperCommand)
   The ROS-standard gripper interface (same name/type PickNik's ros2_robotiq_gripper
   and MoveIt use), so this is drop-in for tooling later.
       goal.command.position   float64, METERS gap: 0.0 = fully CLOSED, 0.085 = fully OPEN
       goal.command.max_effort float64, Newtons (0 => use the node's default ``force``)
       result/feedback.position current gap in meters; .stalled True on object contact;
                                .reached_goal True when the commanded gap was reached.
2. Service ``~/set_closed`` (std_srvs/srv/SetBool): data=true => CLOSE, false => OPEN.
   A trivial one-liner for humans:  ros2 service call /robotiq_gripper/set_closed std_srvs/srv/SetBool "{data: true}"
3. Topics (published at ``status_rate_hz``):
       ``/robotiq_gripper/joint_states``    sensor_msgs/JointState  (knuckle joint angle)
       ``~/position_percent``               std_msgs/Float32        (0.0 open .. 1.0 closed)

Position conventions (all conversions live here)
------------------------------------------------
* URCap POS register: 0 = OPEN, 255 = CLOSED (per Robotiq 2F-85 manual).
* GripperCommand.position (meters): 0.0 = CLOSED, stroke = OPEN  (ROS convention).
  => POS = round((1 - clamp(pos_m, 0, stroke) / stroke) * 255)

Build-PC vs robot-PC
--------------------
``connect_on_start`` defaults to False so the node is safe to launch on a PC with no
gripper: it starts, advertises everything, and every command returns a clear "not
connected" message instead of touching a socket. On the robot PC (or any PC that can
reach the UR controller) launch with ``connect_on_start:=true robot_ip:=<UR_IP>``.
"""

import os
import sys
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_srvs.srv import SetBool

# --- Make the ``gello`` package importable (holds the RobotiqGripper driver). ---
# gello lives at the repo root: <repo>/gello/... . Candidates, in order:
# $GELLO_REPO_ROOT, four dirs up from this file (source tree / --symlink-install),
# then the known checkout paths on lab machines.
_REPO_ROOT = ""
for _root in (
    os.environ.get("GELLO_REPO_ROOT", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
    "/home/laptop3/gello_software",
    "/home/theo_lab/gello_software",
):
    if _root and os.path.isdir(os.path.join(_root, "gello")):
        _REPO_ROOT = _root
        if _root not in sys.path:
            sys.path.insert(0, _root)
        break


class RobotiqGripperActionNode(Node):
    """GripperCommand action server + SetBool service driving a 2F-85 over URCap."""

    def __init__(self):
        super().__init__("robotiq_gripper")

        # --- Parameters ---
        self.declare_parameter("robot_ip", "0.0.0.0")
        self.declare_parameter("gripper_port", 63352)
        self.declare_parameter("speed", 150)          # URCap SPE 0-255
        self.declare_parameter("force", 50)           # URCap FOR 0-255 (default when max_effort=0)
        self.declare_parameter("connect_on_start", False)
        self.declare_parameter("auto_calibrate", False)  # URCap already calibrates; True moves the gripper
        self.declare_parameter("stroke_m", 0.085)     # 2F-85 stroke
        self.declare_parameter("max_force_n", 235.0)  # 2F-85 max grip force, for effort->FOR mapping
        self.declare_parameter("knuckle_closed_rad", 0.8)  # joint angle at fully closed (for JointState)
        self.declare_parameter("joint_name", "robotiq_85_left_knuckle_joint")
        self.declare_parameter("status_rate_hz", 5.0)
        self.declare_parameter("move_timeout_s", 3.0)  # max wait for a commanded move to settle

        self._robot_ip = self.get_parameter("robot_ip").value
        self._port = int(self.get_parameter("gripper_port").value)
        self._speed = int(self.get_parameter("speed").value)
        self._force = int(self.get_parameter("force").value)
        self._auto_cal = bool(self.get_parameter("auto_calibrate").value)
        self._stroke = float(self.get_parameter("stroke_m").value)
        self._max_force_n = float(self.get_parameter("max_force_n").value)
        self._knuckle_closed = float(self.get_parameter("knuckle_closed_rad").value)
        self._joint_name = self.get_parameter("joint_name").value
        self._move_timeout = float(self.get_parameter("move_timeout_s").value)
        connect_on_start = bool(self.get_parameter("connect_on_start").value)

        # --- Driver state (guarded by _hw_lock for all socket I/O) ---
        self._gripper = None
        self._connected = False
        self._hw_lock = threading.Lock()
        self._cb_group = ReentrantCallbackGroup()

        # --- Interfaces (advertised regardless of connection state) ---
        self._action_server = ActionServer(
            self,
            GripperCommand,
            "robotiq_gripper_controller/gripper_cmd",
            execute_callback=self._execute_gripper_cmd,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _c: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )
        self._set_closed_srv = self.create_service(
            SetBool, "~/set_closed", self._on_set_closed, callback_group=self._cb_group
        )
        self._js_pub = self.create_publisher(JointState, "~/joint_states", 10)
        self._pct_pub = self.create_publisher(Float32, "~/position_percent", 10)

        rate = float(self.get_parameter("status_rate_hz").value)
        if rate > 0.0:
            self._status_timer = self.create_timer(
                1.0 / rate, self._publish_status, callback_group=self._cb_group
            )

        # --- Connect (in a background thread so a slow/faulted activate never hangs
        #     node startup; the interfaces above are already live). ---
        if connect_on_start:
            threading.Thread(target=self._connect, daemon=True).start()
        else:
            self.get_logger().warn(
                "connect_on_start=False: NOT connecting to a physical gripper. "
                "Commands are accepted but return 'not connected'. Launch with "
                "connect_on_start:=true robot_ip:=<UR_IP> on a PC that can reach the robot."
            )

        self.get_logger().info(
            f"robotiq_gripper up. action=/robotiq_gripper_controller/gripper_cmd "
            f"service=/robotiq_gripper/set_closed  (robot_ip={self._robot_ip}, "
            f"port={self._port}, speed={self._speed}, force={self._force})."
        )

    # ------------------------------------------------------------------ #
    # Connection / activation
    # ------------------------------------------------------------------ #
    def _connect(self):
        try:
            from gello.robots.robotiq_gripper import RobotiqGripper
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Could not import RobotiqGripper from the gello package (PYTHONPATH? "
                f"repo root tried: '{_REPO_ROOT}'). Error: {exc}"
            )
            return
        try:
            g = RobotiqGripper()
            self.get_logger().info(
                f"Connecting to URCap gripper socket at {self._robot_ip}:{self._port} ..."
            )
            g.connect(self._robot_ip, self._port)
            g.activate(auto_calibrate=self._auto_cal)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"Failed to connect/activate gripper at {self._robot_ip}:{self._port}. "
                f"Is the Robotiq 'Grippers' URCap installed & the gripper activated on the "
                f"pendant? (verify: `nc {self._robot_ip} {self._port}` then type `GET STA`). "
                f"Error: {exc}"
            )
            return
        with self._hw_lock:
            self._gripper = g
            self._connected = True
        self.get_logger().info(
            f"Gripper connected & activated at {self._robot_ip}:{self._port}."
        )

    # ------------------------------------------------------------------ #
    # Position <-> unit conversions
    # ------------------------------------------------------------------ #
    def _meters_to_pos(self, pos_m: float) -> int:
        """GripperCommand meters (0=closed .. stroke=open) -> URCap POS (0=open .. 255=closed)."""
        pos_m = max(0.0, min(self._stroke, pos_m))
        pos255 = round((1.0 - pos_m / self._stroke) * 255.0)
        return int(max(0, min(255, pos255)))

    def _pos_to_meters(self, pos255: int) -> float:
        """URCap POS (0=open .. 255=closed) -> gap in meters (0=closed .. stroke=open)."""
        return (1.0 - max(0, min(255, pos255)) / 255.0) * self._stroke

    def _effort_to_force(self, max_effort: float) -> int:
        if max_effort is None or max_effort <= 0.0:
            return self._force
        f = round(max_effort / self._max_force_n * 255.0)
        return int(max(0, min(255, f)))

    # ------------------------------------------------------------------ #
    # GripperCommand action
    # ------------------------------------------------------------------ #
    def _execute_gripper_cmd(self, goal_handle):
        cmd = goal_handle.request.command
        target_pos = self._meters_to_pos(cmd.position)
        force = self._effort_to_force(cmd.max_effort)
        result = GripperCommand.Result()

        if not self._connected or self._gripper is None:
            self.get_logger().warn(
                f"[no gripper] would move to POS={target_pos} "
                f"(pos={cmd.position:.4f} m, force={force}). Aborting goal."
            )
            result.position = cmd.position
            result.reached_goal = False
            result.stalled = False
            goal_handle.abort()
            return result

        # Issue the move.
        with self._hw_lock:
            try:
                self._gripper.move(target_pos, self._speed, force)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"gripper.move({target_pos}) failed: {exc}")
                goal_handle.abort()
                return result

        # Poll until it settles (reaches target, or stops early on an object), with feedback.
        deadline = self.get_clock().now().nanoseconds / 1e9 + self._move_timeout
        stalled = False
        cur = target_pos
        obj = 0
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.position = self._pos_to_meters(cur)
                result.reached_goal = False
                result.stalled = stalled
                return result
            with self._hw_lock:
                try:
                    cur = self._gripper.get_current_position()
                    obj = self._gripper._get_var(self._gripper.OBJ)
                except Exception:  # noqa: BLE001
                    break
            fb = GripperCommand.Feedback()
            fb.position = self._pos_to_meters(cur)
            fb.effort = float(force)
            fb.stalled = obj in (1, 2)
            fb.reached_goal = abs(cur - target_pos) <= 3
            goal_handle.publish_feedback(fb)
            # obj: 0 moving, 1/2 stopped on object, 3 reached target with no object.
            if obj in (1, 2):
                stalled = True
                break
            if obj == 3 or abs(cur - target_pos) <= 3:
                break
            if self.get_clock().now().nanoseconds / 1e9 > deadline:
                break
            time.sleep(0.02)

        result.position = self._pos_to_meters(cur)
        result.effort = float(force)
        result.stalled = stalled
        result.reached_goal = stalled or abs(cur - target_pos) <= 5
        goal_handle.succeed()
        return result

    # ------------------------------------------------------------------ #
    # SetBool convenience service (open/close)
    # ------------------------------------------------------------------ #
    def _on_set_closed(self, request, response):
        target_pos = 255 if request.data else 0  # 255 closed, 0 open
        if not self._connected or self._gripper is None:
            response.success = False
            response.message = (
                f"not connected (would move to POS={target_pos}). Launch with "
                f"connect_on_start:=true robot_ip:=<UR_IP>."
            )
            self.get_logger().warn(response.message)
            return response
        with self._hw_lock:
            try:
                self._gripper.move(target_pos, self._speed, self._force)
                response.success = True
                response.message = f"{'closing' if request.data else 'opening'} (POS={target_pos})"
            except Exception as exc:  # noqa: BLE001
                response.success = False
                response.message = f"gripper.move failed: {exc}"
                self.get_logger().error(response.message)
        return response

    # ------------------------------------------------------------------ #
    # Status publishing
    # ------------------------------------------------------------------ #
    def _publish_status(self):
        if not self._connected or self._gripper is None:
            return
        with self._hw_lock:
            try:
                pos255 = self._gripper.get_current_position()
            except Exception:  # noqa: BLE001
                return
        now = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = now
        js.name = [self._joint_name]
        js.position = [pos255 / 255.0 * self._knuckle_closed]
        self._js_pub.publish(js)
        pct = Float32()
        pct.data = float(pos255 / 255.0)  # 0.0 open .. 1.0 closed
        self._pct_pub.publish(pct)


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = RobotiqGripperActionNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
