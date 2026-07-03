#!/usr/bin/env python3
"""Gripper-only ROS 2 control for a Robotiq 2F-85 on a UR7e, via Modbus-over-tool-comm.

This is the PRIMARY, verified gripper path for the RWH UR7e (whose PolyScope has the
UR "RS485 / tool communication" URCap installed, exposing the tool RS485 line on
``ROBOT_IP:54321``). It drives the gripper with Modbus RTU straight over that TCP
socket — no socat, no ``/tmp/ttyUR``, no C++ build, no extra ROS packages. It brings
up ONLY the gripper (no arm, no ros2_control).

(An alternative node ``robotiq_urcap`` exists for the OTHER PolyScope config — the
Robotiq "Grippers" URCap socket on port 63352 — but that is mutually exclusive with
the RS485 tool-comm URCap and is NOT the current robot's setup.)

Interfaces
----------
* Action  ``/robotiq_gripper_controller/gripper_cmd``  control_msgs/action/GripperCommand
      goal.command.position   meters gap: 0.0 = CLOSED, 0.085 = OPEN (ROS convention)
      goal.command.max_effort Newtons (0 => node default ``force``)
      result/feedback: .position (m), .stalled (object contact), .reached_goal
* Service ``~/set_closed``  std_srvs/srv/SetBool   (data=true => CLOSE, false => OPEN)
* Topics  ``~/joint_states`` (sensor_msgs/JointState), ``~/position_percent`` (Float32,
      0.0 open .. 1.0 closed)
* Sub     ``~/command_percent`` (std_msgs/Float32, 0.0 = OPEN .. 1.0 = CLOSED) — a
      continuous streaming setpoint. Rate-limited + deadbanded, then issues a SINGLE
      non-blocking ``g.move()`` per accepted setpoint (no status-poll loop), so the
      single-client :54321 bus stays quiet while a leader (e.g. GELLO) streams.

Commands (action + service) are serialized against each other via a mutually-exclusive
callback group — the :54321 forwarder is single-client, so concurrent goals must not
interleave. Status polling runs in a separate group.

Powered-off robot: tool voltage is off when the robot is POWER_OFF, so the gripper
won't answer. The node keeps retrying the connection in the background and connects
automatically once the robot is powered on.
"""
import os
import sys
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_srvs.srv import SetBool

# Import the driver whether run from source, symlink-install, or installed share.
try:
    from ur_gello_bringup.robotiq_2f85_modbus import Robotiq2F85, GripperError
except Exception:  # noqa: BLE001 - fall back to a path-relative import
    sys.path.insert(0, os.path.dirname(__file__))
    from robotiq_2f85_modbus import Robotiq2F85, GripperError

_IOERR = (GripperError, OSError)


class RobotiqGripperModbusNode(Node):
    def __init__(self):
        super().__init__("robotiq_gripper")

        # --- Parameters ---
        self.declare_parameter("robot_ip", "192.168.10.11")
        self.declare_parameter("tool_comm_port", 54321)
        self.declare_parameter("serial_port", "")   # non-empty => use serial (/tmp/ttyUR) instead of TCP
        self.declare_parameter("speed", 150)         # rSP 0-255
        self.declare_parameter("force", 50)          # rFR 0-255 (default when max_effort=0)
        self.declare_parameter("connect_on_start", True)
        self.declare_parameter("activate_on_connect", True)
        self.declare_parameter("stroke_m", 0.085)
        self.declare_parameter("max_force_n", 235.0)
        self.declare_parameter("knuckle_closed_rad", 0.8)
        self.declare_parameter("joint_name", "robotiq_85_left_knuckle_joint")
        self.declare_parameter("status_rate_hz", 5.0)
        self.declare_parameter("move_timeout_s", 3.0)
        self.declare_parameter("reconnect_period_s", 3.0)
        # Streaming setpoint (~/command_percent) shaping — protects the single-client bus.
        self.declare_parameter("command_rate_hz", 20.0)   # max accepted writes/sec
        self.declare_parameter("command_deadband", 0.01)  # min |Δpercent| to write

        gp = self.get_parameter
        self._robot_ip = gp("robot_ip").value
        self._port = int(gp("tool_comm_port").value)
        self._serial_port = gp("serial_port").value or None
        self._speed = int(gp("speed").value)
        self._force = int(gp("force").value)
        self._activate = bool(gp("activate_on_connect").value)
        self._stroke = float(gp("stroke_m").value)
        self._max_force_n = float(gp("max_force_n").value)
        self._knuckle_closed = float(gp("knuckle_closed_rad").value)
        self._joint_name = gp("joint_name").value
        self._move_timeout = float(gp("move_timeout_s").value)
        self._reconnect_period = float(gp("reconnect_period_s").value)
        connect_on_start = bool(gp("connect_on_start").value)

        # --- Streaming command state (~/command_percent) ---
        self._cmd_deadband = float(gp("command_deadband").value)
        cmd_rate = float(gp("command_rate_hz").value)
        self._cmd_min_period = (1.0 / cmd_rate) if cmd_rate > 0.0 else 0.0
        self._last_cmd_pct = None     # last percent actually written to the bus
        self._last_cmd_time = 0.0     # time.monotonic() of last accepted write

        # --- Driver state ---
        self._g = None
        self._connected = False
        self._stop = threading.Event()
        self._conn_lock = threading.Lock()   # guards _connecting / _g / _connected commits
        self._connecting = False             # at most ONE reconnect loop at a time

        # Commands (action + service) share one mutually-exclusive group so they never
        # run concurrently on the single-client bus; status polling gets its own group.
        self._cmd_cb = MutuallyExclusiveCallbackGroup()
        self._status_cb = MutuallyExclusiveCallbackGroup()

        # --- Interfaces (advertised regardless of connection) ---
        self._action = ActionServer(
            self, GripperCommand, "robotiq_gripper_controller/gripper_cmd",
            execute_callback=self._execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _c: CancelResponse.ACCEPT,
            callback_group=self._cmd_cb,
        )
        self.create_service(SetBool, "~/set_closed", self._on_set_closed,
                            callback_group=self._cmd_cb)
        # Streaming setpoint subscriber — shares the command group so streaming writes
        # never interleave with an action goal or set_closed on the single-client bus.
        self.create_subscription(Float32, "~/command_percent",
                                 self._on_command_percent, 10,
                                 callback_group=self._cmd_cb)
        self._js_pub = self.create_publisher(JointState, "~/joint_states", 10)
        self._pct_pub = self.create_publisher(Float32, "~/position_percent", 10)

        rate = float(gp("status_rate_hz").value)
        if rate > 0.0:
            self.create_timer(1.0 / rate, self._publish_status,
                              callback_group=self._status_cb)

        transport = (f"serial:{self._serial_port}" if self._serial_port
                     else f"tcp:{self._robot_ip}:{self._port}")
        if connect_on_start:
            self._spawn_reconnect()
        self.get_logger().info(
            f"robotiq_gripper up ({transport}). "
            f"action=/robotiq_gripper_controller/gripper_cmd "
            f"service=/robotiq_gripper/set_closed  speed={self._speed} force={self._force}"
        )

    # ------------------------------------------------------------------ #
    # Connection (retries until the gripper answers, e.g. after power-on)
    # ------------------------------------------------------------------ #
    def _spawn_reconnect(self):
        """Start the reconnect loop iff one isn't already running (single-connection
        forwarder: multiple concurrent loops would storm :54321)."""
        with self._conn_lock:
            if self._connecting or self._connected or self._stop.is_set():
                return
            self._connecting = True
        threading.Thread(target=self._connect_loop, daemon=True).start()

    def _connect_loop(self):
        try:
            while not self._stop.is_set():
                g = None
                try:
                    g = Robotiq2F85(host=self._robot_ip, port=self._port,
                                   serial_port=self._serial_port)
                    g.connect()
                    st = g.read_status()  # proves the gripper is powered & talking
                    if self._activate and not (st["gACT"] == 1 and st["gSTA"] == 3):
                        self.get_logger().info("Activating gripper (auto-cal sweep)...")
                        g.activate()
                    ready = g.read_status()
                    # Commit only if we're not shutting down (else the socket leaks
                    # and starves the next launch's :54321 connection).
                    with self._conn_lock:
                        if self._stop.is_set():
                            g.close()
                            return
                        self._g = g
                        self._connected = True
                    self.get_logger().info(f"Gripper connected & ready: {ready}")
                    return
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(
                        f"Gripper not ready ({exc}). Is the robot POWERED ON? "
                        f"Retrying in {self._reconnect_period:.0f}s.",
                        throttle_duration_sec=10.0,
                    )
                    try:
                        if g is not None:
                            g.close()
                    except Exception:
                        pass
                    self._stop.wait(self._reconnect_period)
        finally:
            with self._conn_lock:
                self._connecting = False

    def _drop_connection(self):
        """Called when I/O fails mid-op: close and let a single reconnect loop retry."""
        with self._conn_lock:
            self._connected = False
            g, self._g = self._g, None
        try:
            if g is not None:
                g.close()
        except Exception:
            pass
        if not self._stop.is_set():
            self._spawn_reconnect()

    # ------------------------------------------------------------------ #
    # Conversions
    # ------------------------------------------------------------------ #
    def _m_to_pos(self, pos_m: float) -> int:
        pos_m = max(0.0, min(self._stroke, pos_m))
        return int(max(0, min(255, round((1.0 - pos_m / self._stroke) * 255.0))))

    def _pos_to_m(self, pos255: int) -> float:
        return (1.0 - max(0, min(255, pos255)) / 255.0) * self._stroke

    def _effort_to_force(self, max_effort: float) -> int:
        if not max_effort or max_effort <= 0.0:
            return self._force
        return int(max(0, min(255, round(max_effort / self._max_force_n * 255.0))))

    # ------------------------------------------------------------------ #
    # GripperCommand action
    # ------------------------------------------------------------------ #
    def _execute(self, goal_handle):
        cmd = goal_handle.request.command
        target = self._m_to_pos(cmd.position)
        force = self._effort_to_force(cmd.max_effort)
        result = GripperCommand.Result()

        g = self._g  # local ref: a concurrent _drop_connection may null self._g
        if not self._connected or g is None:
            self.get_logger().warn(
                f"[no gripper] would move to POS={target}. Robot powered on? Aborting."
            )
            result.position = cmd.position
            result.reached_goal = False
            result.stalled = False
            goal_handle.abort()
            return result

        cur, obj, stalled = target, 0, False
        try:
            g.move(target, self._speed, force)
            deadline = time.time() + self._move_timeout
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    try:
                        g.stop()  # hold current position
                    except _IOERR:
                        self._drop_connection()
                    goal_handle.canceled()
                    result.position = self._pos_to_m(cur)
                    result.effort = float(force)
                    result.stalled = stalled
                    result.reached_goal = False
                    return result
                st = g.read_status()
                cur, obj = st["gPO"], st["gOBJ"]
                fb = GripperCommand.Feedback()
                fb.position = self._pos_to_m(cur)
                fb.effort = float(force)
                fb.stalled = obj in (1, 2)
                fb.reached_goal = abs(cur - target) <= 3
                goal_handle.publish_feedback(fb)
                if obj in (1, 2):
                    stalled = True
                    break
                if obj == 3 or abs(cur - target) <= 3:
                    break
                if time.time() > deadline:
                    break
                time.sleep(0.02)
        except _IOERR as exc:
            self.get_logger().error(f"gripper move failed: {exc}")
            self._drop_connection()
            result.position = self._pos_to_m(cur)
            result.stalled = stalled
            result.reached_goal = False
            goal_handle.abort()
            return result

        result.position = self._pos_to_m(cur)
        result.effort = float(force)
        result.stalled = stalled
        # reached_goal: stopped on an object (stalled), or the gripper reported
        # "at requested position / no object" (gOBJ==3), or within tolerance.
        # (An empty full-close settles near POS 229, not 255, so gOBJ==3 covers it.)
        result.reached_goal = stalled or obj == 3 or abs(cur - target) <= 5
        goal_handle.succeed()
        return result

    # ------------------------------------------------------------------ #
    # SetBool convenience (open/close)
    # ------------------------------------------------------------------ #
    def _on_set_closed(self, request, response):
        target = 255 if request.data else 0
        g = self._g
        if not self._connected or g is None:
            response.success = False
            response.message = f"not connected (would move to POS={target}); robot powered on?"
            return response
        try:
            g.move(target, self._speed, self._force)
            response.success = True
            response.message = f"{'closing' if request.data else 'opening'} (POS={target})"
        except _IOERR as exc:
            response.success = False
            response.message = f"move failed: {exc}"
            self._drop_connection()
        return response

    # ------------------------------------------------------------------ #
    # Streaming setpoint (~/command_percent): 0.0=OPEN .. 1.0=CLOSED
    # ------------------------------------------------------------------ #
    def _on_command_percent(self, msg):
        """Accept a streaming gripper setpoint and issue ONE non-blocking write.

        Rate-limited (>= command_min_period between writes) and deadbanded (ignore
        sub-threshold changes) so a continuous leader stream does not storm the
        single-client :54321 bus. A large jump (>= 0.5) always passes the rate limit
        immediately, so an emergency full-open/full-close is never delayed. No status
        poll here — exactly one FC16 write per accepted setpoint; the driver lock
        serializes it against the status timer and other command callbacks.
        """
        pct = min(max(float(msg.data), 0.0), 1.0)

        # DEADBAND: ignore sub-threshold changes vs the last written setpoint.
        if self._last_cmd_pct is not None and \
                abs(pct - self._last_cmd_pct) < self._cmd_deadband:
            return

        # RATE-LIMIT: throttle bus writes, but let a large jump through immediately.
        now = time.monotonic()
        big_jump = abs(pct - (self._last_cmd_pct or 0.0)) >= 0.5
        if not big_jump and (now - self._last_cmd_time) < self._cmd_min_period:
            return

        g = self._g  # local ref: a concurrent _drop_connection may null self._g
        if not self._connected or g is None:
            self.get_logger().warn(
                "[no gripper] dropped streaming setpoint; robot powered on?",
                throttle_duration_sec=10.0,
            )
            return

        pos = int(round(pct * 255.0))
        try:
            g.move(pos, self._speed, self._force)
        except _IOERR as exc:
            self.get_logger().warn(f"streaming move failed: {exc}")
            self._drop_connection()
            return
        self._last_cmd_pct = pct
        self._last_cmd_time = now

    # ------------------------------------------------------------------ #
    # Status publishing
    # ------------------------------------------------------------------ #
    def _publish_status(self):
        g = self._g
        if not self._connected or g is None:
            return
        try:
            pos255 = g.get_position()
        except _IOERR:
            self._drop_connection()
            return
        now = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = now
        js.name = [self._joint_name]
        js.position = [pos255 / 255.0 * self._knuckle_closed]
        self._js_pub.publish(js)
        pct = Float32()
        pct.data = float(pos255 / 255.0)
        self._pct_pub.publish(pct)

    def destroy_node(self):
        self._stop.set()
        with self._conn_lock:
            g, self._g = self._g, None
            self._connected = False
        try:
            if g is not None:
                g.close()  # clean close so the robot releases :54321
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RobotiqGripperModbusNode()
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
