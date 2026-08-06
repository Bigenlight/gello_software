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
      The rate limiter COALESCES: a setpoint arriving inside the closed window is
      remembered as ``_pending_pct`` and flushed by a timer when the window opens,
      so the LAST sample of a stream is always the one that reaches the gripper.

Commands (action + service + streaming flush) are serialized against each other by
``_bus_cmd_lock`` — the :54321 forwarder is single-client, so concurrent goals must
not interleave. The mutually-exclusive callback group is NOT sufficient on its own:
rclpy runs an action's ``execute_callback`` as a bare executor task
(``rclpy/action/server.py`` ``notify_execute`` -> ``executor.create_task``, and
``Executor.create_task`` appends ``(task, None, None)`` — no entity, hence no
callback group), so under a MultiThreadedExecutor it runs concurrently with every
callback in the group. Measured on Humble: 60 group callbacks fired inside one
1 s ``execute_callback``. Status polling runs in a separate group on purpose and
relies on the driver's own transaction lock.

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
        # Freshest setpoint that has NOT been written yet (None => nothing owed).
        # A rate limiter that drops the newest sample loses the end of every
        # stream; this holds it until the window opens.
        #
        # Written by the subscription and read/cleared by the flush timer, which
        # ARE mutually exclusive (both are real entities in self._cmd_cb, so the
        # executor's can_execute/beginning_execution gate applies). The action's
        # execute_callback is NOT — see the module docstring — so every path that
        # owns the bus takes _bus_cmd_lock before touching this or _last_cmd_pct.
        self._pending_pct = None
        # Held by whoever currently owns the single-client :54321 bus for a
        # COMMAND (action goal, set_closed, streaming flush). The status poll
        # deliberately does not take it: it has its own callback group and the
        # driver's transaction lock already keeps frames from interleaving.
        # The flush only ever tries it non-blockingly, so a long action goal can
        # never block a self._cmd_cb thread (which would starve the action's own
        # cancel handling, since that is served from the same group).
        self._bus_cmd_lock = threading.Lock()

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
        # Coalescing flush timer for the streaming setpoint. It lives in
        # self._cmd_cb so it cannot interleave with the subscription that feeds
        # _pending_pct; _bus_cmd_lock (not the group) is what keeps its write off
        # the bus while an action goal or set_closed owns it. It is a pure no-op
        # when nothing is pending, so the single-client bus stays completely
        # silent at rest.
        self._cmd_flush_timer = None
        if self._cmd_min_period > 0.0:
            self._cmd_flush_timer = self.create_timer(
                self._cmd_min_period / 2.0, self._try_flush_pending,
                callback_group=self._cmd_cb,
            )
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
                        # The activate() auto-calibration sweep leaves the fingers
                        # wherever it ends, so any cached command is now a lie about
                        # where the gripper is. Forget it BEFORE publishing the
                        # connection, so the first streaming sample after this point
                        # can never be swallowed by the deadband. (_pending_pct is
                        # deliberately kept: a desire should survive a reconnect.)
                        self._last_cmd_pct = None
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
            # Invalidate the streaming cache: the reconnect will re-activate, whose
            # auto-cal sweep moves the fingers, and a steady stream of the same
            # value would otherwise never re-assert it past the deadband.
            self._last_cmd_pct = None
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
        # Own the bus for the whole goal. rclpy runs this callback as an executor
        # task OUTSIDE self._cmd_cb (module docstring), so without this a streaming
        # flush lands between our move() and our status polls: the gripper chases
        # the leader instead of the goal, reached_goal never trips, and worse,
        # _last_cmd_pct ends up recording the streaming value while the fingers sit
        # at OUR target -- after which the deadband swallows the leader's next
        # sample and the gripper parks short. That is the exact defect the
        # coalescing rate limiter exists to prevent.
        with self._bus_cmd_lock:
            try:
                g.move(target, self._speed, force)
                # Keep the streaming path's bookkeeping in sync: without this, a
                # later GELLO stream sample that happens to land near the PRE-action
                # _last_cmd_pct (not where this action just moved the gripper) is
                # wrongly treated as "no change" by the command_percent deadband and
                # is silently dropped -- streaming never re-asserts, and the gripper
                # stays wherever this action left it.
                # Dropping _pending_pct is part of the same bookkeeping: a setpoint
                # that was still waiting for its rate-limit window when this goal
                # arrived is now STALE, and flushing it after this move would silently
                # undo the position the action just commanded. (A sample that arrives
                # DURING the goal is fresher than the goal and is deliberately kept.)
                self._pending_pct = None
                self._last_cmd_pct = target / 255.0
                self._last_cmd_time = time.monotonic()
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
        # Own the bus for this command, same as _execute: an action goal is NOT in
        # our callback group (module docstring), so the group alone does not keep
        # the two apart on the single-client :54321 forwarder.
        with self._bus_cmd_lock:
            try:
                g.move(target, self._speed, self._force)
                # Same streaming-resync fix as _execute (see its comment): keep
                # _last_cmd_pct current so a subsequent GELLO stream sample isn't
                # dropped by the command_percent deadband against a stale value, and
                # discard any setpoint still waiting on its rate-limit window so it
                # cannot flush afterwards and undo this move.
                self._pending_pct = None
                self._last_cmd_pct = target / 255.0
                self._last_cmd_time = time.monotonic()
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
        """Record the freshest streaming setpoint, then try to put it on the bus.

        This callback NEVER discards a sample. Discarding the newest sample is what
        made the gripper park short of the commanded position: a ~30 Hz leader
        stream is accepted at ~20 Hz, so the final sample of a release (the one
        that says "fully open") landed inside a closed rate-limit window about half
        the time and was lost forever — nobody re-asserts, and _last_cmd_pct stays
        self-consistent with what was actually written, so nothing ever notices.
        Now the newest desire is always remembered and ``_try_flush_pending`` (also
        driven by a timer) writes it as soon as the window opens.
        """
        self._pending_pct = min(max(float(msg.data), 0.0), 1.0)
        self._try_flush_pending()

    def _try_flush_pending(self):
        """Write ``_pending_pct`` if the bus is allowed to take it right now.

        Called from ``_on_command_percent`` and from the flush timer; both are in
        ``self._cmd_cb`` so this never runs concurrently with itself or with the
        subscription that feeds ``_pending_pct``. An action goal is NOT in that
        group (module docstring), so ``_bus_cmd_lock`` is what keeps this off the
        bus while a goal or ``set_closed`` owns it — tried non-blockingly, so a
        long goal defers the flush instead of blocking a callback-group thread.

        Rate-limited (>= command_min_period between writes) and deadbanded (ignore
        sub-threshold changes) so a continuous leader stream does not storm the
        single-client :54321 bus. A large jump (>= 0.5) always passes the rate limit
        immediately, so an emergency full-open/full-close is never delayed. No status
        poll here — exactly one FC16 write per accepted setpoint; the driver lock
        serializes it against the status timer and other command callbacks.
        """
        if self._pending_pct is None:
            return  # nothing owed => zero bus traffic at rest (single-client bus)
        if not self._bus_cmd_lock.acquire(blocking=False):
            return  # an action goal / set_closed owns the bus; stay pending
        try:
            self._flush_locked()
        finally:
            self._bus_cmd_lock.release()

    def _flush_locked(self):
        """Body of :meth:`_try_flush_pending`, with ``_bus_cmd_lock`` held."""
        pct = self._pending_pct
        if pct is None:
            return

        # DEADBAND: ignore sub-threshold changes vs the last written setpoint.
        # The desire is already on the wire, so nothing is owed any more.
        if self._last_cmd_pct is not None and \
                abs(pct - self._last_cmd_pct) < self._cmd_deadband:
            self._pending_pct = None
            return

        # RATE-LIMIT: throttle bus writes, but let a large jump through immediately.
        # (_last_cmd_pct is None => never written / invalidated by a reconnect, which
        # is NOT a jump from 0.0; the elapsed-time test below lets it straight out.)
        now = time.monotonic()
        big_jump = (self._last_cmd_pct is not None
                    and abs(pct - self._last_cmd_pct) >= 0.5)
        if not big_jump and (now - self._last_cmd_time) < self._cmd_min_period:
            return  # KEEP it pending -- the flush timer will write it

        g = self._g  # local ref: a concurrent _drop_connection may null self._g
        if not self._connected or g is None:
            self.get_logger().warn(
                "[no gripper] streaming setpoint deferred; robot powered on?",
                throttle_duration_sec=10.0,
            )
            # KEEP it pending. There is no bus to storm -- the socket is closed, so
            # this branch issues zero I/O -- and the reconnect loop is the retry.
            # Forgetting here re-creates the whole defect this class of fix exists
            # to kill: the last sample of a release lands during a 1-3 s reconnect,
            # the bridge is silent at rest, and the gripper parks short forever.
            # It cannot go stale either: the subscription keeps overwriting
            # _pending_pct with the leader's live value throughout the outage.
            return

        pos = int(round(pct * 255.0))
        try:
            g.move(pos, self._speed, self._force)
        except _IOERR as exc:
            self.get_logger().warn(f"streaming move failed: {exc}")
            self._drop_connection()
            # Cache AND pending untouched: the cache must only ever reflect a
            # SUCCESSFUL write, and the desire is still owed -- it goes out once
            # the reconnect completes.
            return
        # Only now is the desire discharged. (Nothing fresher can have arrived:
        # the subscription is in the same mutually-exclusive group as this.)
        self._pending_pct = None
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
