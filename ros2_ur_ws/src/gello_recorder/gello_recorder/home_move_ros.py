#!/usr/bin/env python3
"""The ROS half of the GO-HOME / go-to-pose sequence, reusable by any rclpy Node.

``gello_recorder/home_move.py`` is the pure state machine (no ROS, no Qt); it
drives a duck-typed ``ops`` object.  THIS module is the rclpy adapter for it:
the :class:`HomeMoveOps` adapter, the subscriptions / clients / publisher it
reaches into, the freshness-gated ``latest_*`` readers, and the 10 Hz tick.
It was split out of ``task_gui_node.py`` (the task recorder's GO HOME button,
hardware-verified) so the EEF teleop GUI's GO TO START POSE button can run the
identical sequence towards its own target pose instead of re-implementing --
and re-discovering the hazards of -- five asynchronous ROS interfaces.

The sequence, for reference (every deadline and fail-closed rule is in
home_move.py)::

    PAUSING -> SWITCHING_TO_JTC -> MOVING -> OPENING_GRIPPER -> RESTORING_FPC

i.e. force-pause both teleop bridges, hand the joints from
forward_position_controller to scaled_joint_trajectory_controller, drive one
FollowJointTrajectory goal to the target pose, open the Robotiq 2F-85, and
give the joints back to forward_position_controller.

Teleop is deliberately LEFT PAUSED afterwards. Re-engaging is an explicit
operator action (the EEF GUI's ENGAGE button chains eef_resume -> pos_scale ->
eef_engage and re-anchors at the CURRENT pose, so homing cannot break it).
NOTHING in this module ever calls a resume service.

HOW A NODE USES IT
------------------
1. Inherit the mixin AHEAD of the rclpy Node class::

       class MyGuiNode(HomeMoveIoMixin, Node):
           def __init__(self):
               super().__init__("my_gui_node")
               self.install_home_move_io(target_joints=MY_START_POSE)

   The mixin has no ``__init__`` of its own, so it never interferes with the
   Node constructor chain; :meth:`HomeMoveIoMixin.install_home_move_io` is the
   explicit installation step and MUST be called from ``__init__`` (after the
   Node is constructed and BEFORE anything spins it -- it creates a timer that
   fires as soon as an executor picks the node up).  Every graph name is a
   keyword argument with the production value as its default.

2. From the Qt thread, call only the three lock-protected methods
   :meth:`request_go_home`, :meth:`request_stop_home`, :meth:`get_home_status`.

THE NODE PROTOCOL HomeMoveOps DEPENDS ON
-----------------------------------------
:class:`HomeMoveOps` is deliberately intimate with the node -- it reaches into
the node's clients and caches -- and the mixin exists to provide exactly this
surface.  A node that does not use the mixin must provide it by hand
(:class:`HomeMoveIoNode` spells it out as a typing.Protocol):

  methods
    latest_ur_joints() -> list[float] | None   arm pose in UR_JOINT_ORDER,
                                               None when unknown or STALE
    latest_gripper_position() -> float | None  /robotiq_gripper/position_percent,
                                               None when unknown or STALE
    latest_speed_scale() -> float | None       UR speed scaling as a FRACTION in
                                               (0, 1], None when unknown/stale;
                                               must never raise
    get_logger()                               the rclpy logger

  attributes
    _svc_home            dict with keys "arm_pause", "grip_pause"
                         (std_srvs/Trigger clients), "list_controllers"
                         (ListControllers client), "switch_controller"
                         (SwitchController client)
    _home_traj_client    rclpy.action.ActionClient for FollowJointTrajectory
    _gripper_cmd_pub     LONG-LIVED std_msgs/Float32 publisher on the gripper
                         command topic (see HomeMoveOps.command_gripper_open
                         for why long-lived is load-bearing)
    _home_graph_names    dict of the resolved graph names (keys as in
                         install_home_move_io) -- OPTIONAL, only used to word
                         the "service unavailable" messages; the module
                         defaults are used when it is absent

THREADING CONTRACT
------------------
* Every ROS callback here -- the three subscriptions and the 10 Hz home timer
  -- runs on the executor thread that spins the node (a single background
  rclpy spin thread in both GUIs).  The Qt thread only ever calls the public,
  lock-protected methods :meth:`HomeMoveIoMixin.request_go_home`,
  :meth:`HomeMoveIoMixin.request_stop_home` and
  :meth:`HomeMoveIoMixin.get_home_status`.
  SERVICE DONE-CALLBACKS ARE THE EXCEPTION and the reason
  :meth:`HomeMoveOps.pause_bridges` carries a lock: ``add_done_callback`` on an
  ALREADY-RESOLVED future fires INLINE on the calling thread, and the caller of
  ``call_async`` here can be the Qt thread (request_go_home -> request() ->
  _start_pause -> pause_bridges). Two Trigger callbacks can therefore genuinely
  land on two different threads.
* NOTHING here blocks. Every service call is ``call_async`` +
  ``add_done_callback`` and the action is driven by ``send_goal_async`` ->
  ``get_result_async``; ``spin_until_future_complete()`` from inside a callback
  would deadlock the single-threaded executor, so it is never used. (Contrast
  ur_gello_bringup/gello_move_to_start_node.py, which builds the same
  trajectory goal but is a standalone one-shot node and therefore CAN block.)
* ``self._home_lock`` guards HomeMoveController. With a single-threaded spin
  the timer and the done-callbacks cannot actually race each other, but
  :meth:`request_go_home` / :meth:`get_home_status` come from the Qt thread and
  genuinely can, so the lock is real and not ceremonial.
* The done-callbacks handed to HomeMoveController deliberately do NOT take that
  lock. home_move's documented contract is that a ``done_cb`` may fire on any
  thread and does exactly one atomic attribute store, with every transition
  happening inside ``tick()``; taking the lock there would buy nothing and
  could DEADLOCK, because ``add_done_callback`` on an already-resolved future
  fires synchronously -- i.e. potentially underneath ``tick()``, which already
  holds the (non-reentrant) lock. Same reason the "service is absent" fast-fail
  paths below simply call ``done_cb`` inline: ``_begin_step()`` has already
  armed the step by then, so the store lands on the correct step id and the
  next tick reads it.
* ``self._home_state_lock`` guards the three latest-value caches.  It is never
  held together with any other lock (nor is any other lock held while it is
  taken): snapshot under it, release, then act.
"""

import math
import threading
import time
from typing import Callable, List, Optional, Protocol, Sequence

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gello_recorder.spin_health import (
    QOS_DEPTH_GRIPPER,
    QOS_DEPTH_ROBOT_STATE,
    QOS_DEPTH_STATUS,
)
from gello_recorder.home_move import (
    FPC,
    GRIPPER_OPEN_VALUE,
    HOME_JOINTS,
    STJC,
    UR_JOINT_ORDER,
    HomeMoveController,
    reorder_joint_positions,
)

# --- ROS graph names the GO-HOME path talks to ----------------------------- #
# Every one of these is the DEFAULT for the matching install_home_move_io()
# keyword; a node on a differently-named stack overrides the keyword, not the
# constant.
#
# The two teleop bridges' pause services (std_srvs/Trigger, unconditional and
# idempotent per the bridge contract). The task recorder node aliases its base
# class's existing clients for these instead of creating new ones -- see
# install_home_move_io(arm_pause_client=..., gripper_pause_client=...).
ARM_PAUSE_SERVICE = "/gello_ur_bridge/pause"
GRIPPER_PAUSE_SERVICE = "/gello_gripper_bridge/pause"
LIST_CONTROLLERS_SERVICE = "/controller_manager/list_controllers"
SWITCH_CONTROLLER_SERVICE = "/controller_manager/switch_controller"
# The trajectory always goes through the SCALED joint trajectory controller:
# forward_position_controller does not interpolate, so it cannot execute a
# time-parameterised move (see gello_move_to_start_node.py's module docstring).
HOME_TRAJECTORY_ACTION = "/{}/follow_joint_trajectory".format(STJC)
GRIPPER_COMMAND_TOPIC = "/robotiq_gripper/command_percent"
GRIPPER_POSITION_TOPIC = "/robotiq_gripper/position_percent"
JOINT_STATES_TOPIC = "/joint_states"
# The UR driver's speed-scaling broadcaster. READ OFF THIS SYSTEM, not guessed
# (originally verified on Humble at /opt/ros/humble/share/ur_robot_driver/config/
# ur_controllers.yaml -- same relative path and controller/topic names live
# under /opt/ros/jazzy/share/ur_robot_driver/ too, since ur_robot_driver's
# config layout did not change between distros, but this has NOT been
# re-confirmed against an installed Jazzy ur_robot_driver on this machine):
#   * ur_controllers.yaml declares
#     `speed_scaling_state_broadcaster: type: ur_controllers/SpeedScalingStateBroadcaster`
#     and ur_control.launch.py spawns it by that name (so the node name, and
#     hence the topic namespace, is speed_scaling_state_broadcaster);
#   * ur_controllers' speed_scaling_state_broadcaster.hpp declares
#     `rclcpp::Publisher<std_msgs::msg::Float64>` and the shipped library's
#     string table contains the node-relative topic "~/speed_scaling"
#     (version 2.8.1 confirmed on Humble; Jazzy's ur_controllers version not
#     re-checked here).
# => /speed_scaling_state_broadcaster/speed_scaling, std_msgs/msg/Float64,
#    field `data`.
SPEED_SCALING_TOPIC = "/speed_scaling_state_broadcaster/speed_scaling"
# UNITS: that topic carries a PERCENT, not a fraction. The hardware state
# interface (speed_scaling/speed_scaling_factor) is the 0..1 product of the
# pendant slider and the program's target speed fraction, and the broadcaster's
# update() multiplies it by 100.0 before publishing -- confirmed in the shipped
# binary (the mulsd operand in SpeedScalingStateBroadcaster::update is
# 0x4059000000000000, i.e. the double 100.0). HomeMoveController's optional
# speed_scale() hook is specified in FRACTIONS, so latest_speed_scale() divides.
_SPEED_SCALE_PERCENT_PER_FRACTION = 100.0

# ---- staleness windows for the cached sensor reads ------------------------ #
# Every one of these answers the same question: "is this cached value still
# describing the robot NOW, or is it a fossil left behind by a publisher that
# died?" A fossil is worse than no value at all, because the GO-HOME sequencer
# treats a value as evidence and None as "I do not know" (which it handles).
#
# /robotiq_gripper/position_percent comes from robotiq_gripper_modbus_node at
# status_rate_hz = 5.0, and -- the reason this gate exists -- that node
# publishes ONLY while its modbus connection is up (_publish_status returns
# early when not connected). Without a gate, HomeMoveController could confirm
# "gripper open" from a sample taken before the connection dropped, i.e. from a
# gripper nobody can see any more. 2.0 s = 10 nominal periods: long enough that
# a couple of missed reads or a busy spin thread are not mistaken for a dead
# link, short enough that a real disconnect turns into a WARNING inside one
# GRIPPER_OPEN_TIMEOUT_S (3.0 s) window.
_GRIPPER_POSITION_STALE_S = 2.0
# /joint_states arrives from the UR driver at hundreds of Hz, so 1.0 s is a very
# large multiple of the period and cannot fire on jitter. In practice the arm's
# pose going stale means the driver is gone, which is already fatal to a GO
# HOME -- but this value decides the branch-cut-safe target
# (wrapped_nearest(target_joints, current)), and a target computed from a pose
# the arm has since left is precisely the ~2*pi long-way-round hazard. Cheap
# gate, unbounded downside if omitted; HomeMoveController already has the words
# for it ("robot joint states went stale before the move").
_UR_JOINTS_STALE_S = 1.0
# The speed-scaling broadcaster publishes at state_publish_rate = 100.0 Hz, so
# 1.0 s is 100 periods. Stale here is benign by design -- speed_scale() simply
# reports None and HomeMoveController falls back to MIN_ASSUMED_SPEED_SCALE.
_SPEED_SCALE_STALE_S = 1.0

# controller_manager_msgs/srv/SwitchController strictness enum. STRICT means the
# switch either happens completely or not at all -- a partial switch would leave
# the arm with no controller able to move it.
_STRICT = 2
# How long controller_manager may take to realize the switch internally. Same
# value GelloMoveToStart._switch_controllers uses.
#
# COUPLED TO home_move.STEP_TIMEOUT_S, which must be STRICTLY GREATER than this
# (see the essay on that constant for the failure when they were equal).
_SWITCH_TIMEOUT_S = 5.0
# Per-joint arrival tolerance on the home goal (rad). Same 0.05 as the
# move-to-start handshake; the JTC reports SUCCEEDED once inside it.
_GOAL_POSITION_TOLERANCE_RAD = 0.05
# HomeMoveController.tick() cadence. Fast enough that the gripper republish loop
# (which runs one publish per tick) matters, slow enough to be free.
#
# THIS IS WHAT SETS THE GRIPPER RE-ASSERT RATE. home_move's settle window
# (GRIPPER_OPEN_REASSERT_S, 1.0 s) is wall-clock; the number of OPEN commands
# that actually reach the driver inside it is the window divided by this period,
# i.e. ~10 at 0.1 s -- the ~10 Hz for ~1 s that was validated on hardware.
# Slowing this timer thins that stream, so the two constants move together;
# test_home_move.py pins the resulting count so a drift shows up as a failure.
_HOME_TICK_PERIOD_S = 0.1


class HomeMoveIoNode(Protocol):
    """What :class:`HomeMoveOps` needs from its node -- see the module docstring.

    Documentation, not enforcement: HomeMoveOps duck-types the node exactly as
    it always did.  :class:`HomeMoveIoMixin` provides every member below.
    """

    _svc_home: dict
    _home_traj_client: ActionClient
    _gripper_cmd_pub: object

    def latest_ur_joints(self) -> Optional[List[float]]: ...

    def latest_gripper_position(self) -> Optional[float]: ...

    def latest_speed_scale(self) -> Optional[float]: ...

    def get_logger(self): ...


def _service_ready(client) -> bool:
    """``service_is_ready()``, hardened against the shutdown race.

    Same guard as policy_run_gui_node._client_ready: a Ctrl-C/SIGTERM can
    invalidate the rclpy context between the caller's check and this call.
    Reporting "not ready" turns that into a clean GO-HOME refusal instead of an
    exception thrown out of a timer callback (which rclpy propagates all the way
    out of ``spin()``, silently killing the GUI's ROS thread).
    """
    if not rclpy.ok():
        return False
    try:
        return bool(client.service_is_ready())
    except Exception:  # noqa: BLE001 -- shutdown race, not a real fault
        return False


def _action_server_ready(action_client) -> bool:
    """``server_is_ready()`` with the same shutdown-race hardening."""
    if not rclpy.ok():
        return False
    try:
        return bool(action_client.server_is_ready())
    except Exception:  # noqa: BLE001 -- shutdown race, not a real fault
        return False


def _trigger_result(which: str, future) -> tuple:
    """Turn a resolved std_srvs/Trigger future into ``(ok, message)``.

    Never raises: a failed call has to become a readable GO-HOME failure
    message, not an exception inside a done-callback.
    """
    try:
        response = future.result()
    except Exception as exc:  # noqa: BLE001 -- surface, never crash the spin
        return False, "{}: {}".format(which, exc)
    if response is None:
        return False, "{}: no response".format(which)
    ok = bool(response.success)
    message = str(response.message).strip() or ("OK" if ok else "refused")
    return ok, "{}: {}".format(which, message)


def _as_controller_list(value) -> List[str]:
    """Normalize HomeMoveController's controller argument into a name list.

    home_move calls ``switch_controllers(STJC, FPC, cb)`` with plain STRINGS
    (and documents ``deactivate`` as possibly None), while the SwitchController
    request wants lists. Accepting a string, None, or an already-sequence keeps
    this adapter correct if the sequencer's calling convention ever widens.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


class HomeMoveOps:
    """The duck-typed ``ops`` object HomeMoveController drives -- all ROS I/O.

    Deliberately intimate with its node (it reaches into the node's clients and
    caches -- the surface :class:`HomeMoveIoNode` documents and
    :class:`HomeMoveIoMixin` provides): the two are one unit split in half only
    so the sequencing in home_move.py stays ROS-free and laptop-testable. Every
    method here is either non-blocking or a pure cache read.

    Threading: ``current_joints()`` and ``gripper_position()`` are called from
    ``HomeMoveController.request()``, which the Qt thread reaches through
    :meth:`HomeMoveIoMixin.request_go_home`, so they read locked caches.
    Everything else runs on the rclpy spin thread only (from ``tick()`` or from
    a future's done-callback), which is why ``_goal_handle`` / ``_traj_seq``
    need no lock -- documented rather than defended, so a future
    MultiThreadedExecutor change is an obvious thing to re-check.
    """

    def __init__(self, node: "HomeMoveIoNode") -> None:
        self._node = node
        # In-flight FollowJointTrajectory goal handle (spin thread only).
        self._goal_handle = None
        # Monotonic id of the current trajectory attempt (spin thread only).
        # It exists for ONE race: cancel_home_trajectory() can land in the
        # window between send_goal_async() and the goal being accepted, when
        # there is no handle to cancel yet. Bumping the id there lets the
        # late-arriving acceptance recognise itself as superseded and cancel
        # immediately, instead of leaving an unowned trajectory driving the arm.
        self._traj_seq = 0

    def _graph_name(self, key: str, default: str) -> str:
        """The resolved graph name for ``key`` (for messages), or the default.

        The mixin records what it actually created under ``_home_graph_names``;
        a hand-rolled node need not, and then the module default is the best
        available wording.  Messages only -- never used to create anything.
        """
        names = getattr(self._node, "_home_graph_names", None) or {}
        return str(names.get(key, default))

    # ---- clock + cached sensor reads ------------------------------------ #
    def now(self) -> float:
        """Monotonic seconds; every HomeMoveController deadline is measured with it."""
        return time.monotonic()

    def current_joints(self) -> Optional[List[float]]:
        """Latest arm joints in UR_JOINT_ORDER, or None if not known yet."""
        return self._node.latest_ur_joints()

    def gripper_position(self) -> Optional[float]:
        """Latest /robotiq_gripper/position_percent, or None if never received
        (or if the sample has gone stale -- see the node's reader)."""
        return self._node.latest_gripper_position()

    def speed_scale(self) -> Optional[float]:
        """OPTIONAL hook: the UR speed-scaling factor as a FRACTION in (0, 1].

        HomeMoveController reads this once when it enters MOVING, to size the
        move deadline: scaled_joint_trajectory_controller stretches execution by
        the pendant's speed slider, so a perfectly healthy move at 25% takes 4x
        its planned duration and a fixed deadline would cancel it mid-motion and
        strand the arm partway to HOME.

        None means "unknown", which the sequencer handles by assuming
        MIN_ASSUMED_SPEED_SCALE -- so this must NEVER raise, and it is None
        whenever the broadcaster is absent (not every robot config loads it),
        has not published yet, or has gone quiet. See
        :meth:`HomeMoveIoMixin.latest_speed_scale` for the unit conversion
        and the staleness rule.
        """
        return self._node.latest_speed_scale()

    # ---- step 1: pause both teleop bridges ------------------------------- #
    def pause_bridges(self, done_cb: Callable[[bool, str], None]) -> None:
        """Fire BOTH bridge pause Triggers; aggregate into one ``done_cb``.

        Both services are unconditional and idempotent per the bridge contract,
        so firing them is always safe -- in EEF mode the arm bridge additionally
        disengages, and while paused it publishes NOTHING (which is what makes
        deactivating forward_position_controller underneath it safe).

        BOTH are required. An absent service fails fast rather than homing with
        one bridge still live: a running gripper bridge would keep tracking the
        human's GELLO trigger and immediately undo the gripper open, and a
        running arm bridge would fight the trajectory controller.
        """
        clients = [
            ("arm bridge", self._node._svc_home["arm_pause"]),
            ("gripper bridge", self._node._svc_home["grip_pause"]),
        ]
        missing = [name for name, client in clients if not _service_ready(client)]
        if missing:
            done_cb(
                False,
                "pause service(s) unavailable: {} -- the teleop stack looks like "
                "it is not running (bring up the hardware terminal first); "
                "refusing to move the arm".format(", ".join(missing)),
            )
            return

        # NOT spin-thread-only, despite what this comment used to claim.
        # add_done_callback() on an ALREADY-RESOLVED future runs the callback
        # INLINE, on whatever thread is adding it -- and the thread adding it
        # here is whoever called pause_bridges(), i.e. potentially the Qt thread
        # (request_go_home -> HomeMoveController.request() -> _start_pause).
        # So one reply can be aggregated on the Qt thread while the other lands
        # on the spin thread. Unsynchronised, both could see `len(results) ==
        # len(clients)` and call done_cb TWICE, which in HomeMoveController is a
        # second _record() for a step id that may already have moved on.
        # The lock makes the "am I the last one?" test and the store atomic; the
        # _fired flag makes the completion strictly once-only.
        lock = threading.Lock()
        results = {}
        fired = []  # one-element sentinel: appended to exactly once, under lock

        def _one(which: str, future) -> None:
            payload = _trigger_result(which, future)
            with lock:
                if fired:
                    return  # completion already reported; nothing left to say
                results[which] = payload
                if len(results) < len(clients):
                    return
                fired.append(True)
                snapshot = dict(results)
            # done_cb OUTSIDE the lock: it runs HomeMoveController's recorder
            # (and, on the absent-service path above, arbitrary caller code).
            # Holding a lock across it buys nothing and only invents deadlock
            # opportunities.
            ok = all(result[0] for result in snapshot.values())
            message = " | ".join(snapshot[key][1] for key in sorted(snapshot))
            done_cb(ok, message)

        for which, client in clients:
            future = client.call_async(Trigger.Request())
            # Bind `which` per-iteration; a bare closure would capture the loop
            # variable and report both replies under the last name.
            future.add_done_callback(lambda f, w=which: _one(w, f))

    # ---- step 1b / 2 / 5: controller_manager ----------------------------- #
    def get_controller_states(
        self, done_cb: Callable[[bool, bool, bool, str], None]
    ) -> None:
        """Report ``(ok, fpc_active, stjc_active, msg)`` from list_controllers."""
        client = self._node._svc_home["list_controllers"]
        if not _service_ready(client):
            done_cb(
                False,
                False,
                False,
                "{} unavailable -- is the robot driver (controller_manager) "
                "running?".format(
                    self._graph_name("list_controllers", LIST_CONTROLLERS_SERVICE)
                ),
            )
            return

        def _done(future) -> None:
            try:
                response = future.result()
            except Exception as exc:  # noqa: BLE001 -- becomes a GO-HOME failure
                done_cb(False, False, False, "list_controllers failed: {}".format(exc))
                return
            if response is None:
                done_cb(False, False, False, "list_controllers: no response")
                return
            states = {c.name: c.state for c in response.controller}
            fpc_state = states.get(FPC)
            stjc_state = states.get(STJC)
            message = "{}={} {}={}".format(
                FPC, fpc_state or "absent", STJC, stjc_state or "absent"
            )
            done_cb(True, fpc_state == "active", stjc_state == "active", message)

        client.call_async(ListControllers.Request()).add_done_callback(_done)

    def switch_controllers(
        self, activate, deactivate, done_cb: Callable[[bool, str], None]
    ) -> None:
        """STRICT controller switch. ``activate``/``deactivate`` are names or None."""
        client = self._node._svc_home["switch_controller"]
        activate_list = _as_controller_list(activate)
        deactivate_list = _as_controller_list(deactivate)
        label = "activate={} deactivate={}".format(
            activate_list or "[]", deactivate_list or "[]"
        )
        if not _service_ready(client):
            done_cb(
                False,
                "{} unavailable ({}) -- is controller_manager running?".format(
                    self._graph_name("switch_controller", SWITCH_CONTROLLER_SERVICE),
                    label,
                ),
            )
            return

        request = SwitchController.Request()
        # activate_controllers/deactivate_controllers are the field names on
        # both Humble and Jazzy (Foxy-era 'start_controllers'/'stop_controllers'
        # were removed by Jazzy -- using those old names would raise at
        # message-construction time, not silently no-op).
        request.activate_controllers = activate_list
        request.deactivate_controllers = deactivate_list
        request.strictness = _STRICT
        request.activate_asap = True
        request.timeout = Duration(sec=int(_SWITCH_TIMEOUT_S), nanosec=0)

        def _done(future) -> None:
            try:
                response = future.result()
            except Exception as exc:  # noqa: BLE001 -- becomes a GO-HOME failure
                done_cb(False, "switch ({}) failed: {}".format(label, exc))
                return
            if response is None:
                done_cb(False, "switch ({}): no response".format(label))
                return
            ok = bool(response.ok)
            done_cb(
                ok,
                "switch {} {}".format(label, "OK" if ok else "REFUSED (STRICT)"),
            )

        client.call_async(request).add_done_callback(_done)

    # ---- step 3: the home trajectory ------------------------------------- #
    def send_home_trajectory(
        self,
        positions: Sequence[float],
        duration_s: float,
        done_cb: Callable[[bool, str], None],
    ) -> None:
        """Send ONE FollowJointTrajectory goal; ``done_cb`` fires on the RESULT.

        ``positions`` is already the branch-cut-safe target computed by
        HomeMoveController (``wrapped_nearest(target_joints, current)``) -- this
        method must NOT re-wrap or otherwise reinterpret it.

        ok is True only for STATUS_SUCCEEDED; a rejected goal, a missing result
        and any non-succeeded status are all reported as failures with the
        reason attached, because "the arm may not be at HOME" has to reach the
        operator either way.
        """
        action_client = self._node._home_traj_client
        if not _action_server_ready(action_client):
            done_cb(
                False,
                "{} action server unavailable -- is {} loaded, and is the "
                "External Control program PLAYING on the pendant?".format(
                    self._graph_name("trajectory_action", HOME_TRAJECTORY_ACTION),
                    STJC,
                ),
            )
            return

        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        # Come to a full stop at HOME: a non-zero terminal velocity would have
        # the arm arrive still moving, with nothing scheduled after it.
        point.velocities = [0.0] * len(UR_JOINT_ORDER)
        seconds = int(duration_s)
        point.time_from_start = Duration(
            sec=seconds, nanosec=int((float(duration_s) - seconds) * 1e9)
        )

        trajectory = JointTrajectory()
        trajectory.joint_names = list(UR_JOINT_ORDER)
        trajectory.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        # Per-joint arrival tolerance; an empty path_tolerance keeps the
        # controller's own defaults for the in-flight check.
        for name in UR_JOINT_ORDER:
            tolerance = JointTolerance()
            tolerance.name = name
            tolerance.position = _GOAL_POSITION_TOLERANCE_RAD
            goal.goal_tolerance.append(tolerance)

        self._traj_seq += 1
        seq = self._traj_seq
        self._goal_handle = None

        def _on_result(future) -> None:
            if self._goal_handle is not None and seq == self._traj_seq:
                self._goal_handle = None
            try:
                wrapped = future.result()
            except Exception as exc:  # noqa: BLE001 -- becomes a GO-HOME failure
                done_cb(False, "home trajectory result failed: {}".format(exc))
                return
            if wrapped is None:
                done_cb(False, "home trajectory returned no result")
                return
            if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
                done_cb(True, "arm arrived at HOME")
                return
            error_code = getattr(getattr(wrapped, "result", None), "error_code", "?")
            done_cb(
                False,
                "home trajectory did not succeed (status={}, error_code={})".format(
                    wrapped.status, error_code
                ),
            )

        def _on_goal_response(future) -> None:
            try:
                handle = future.result()
            except Exception as exc:  # noqa: BLE001 -- becomes a GO-HOME failure
                done_cb(False, "home trajectory goal send failed: {}".format(exc))
                return
            if handle is None or not handle.accepted:
                done_cb(
                    False,
                    "home trajectory goal REJECTED by {} (is it ACTIVE? on real "
                    "hardware the controller stays inactive until the External "
                    "Control program is PLAYING)".format(STJC),
                )
                return
            if seq != self._traj_seq:
                # Cancelled (or superseded) while we were waiting for the goal
                # to be accepted: cancel it right now. Otherwise this trajectory
                # would keep driving the arm with nobody holding its handle.
                self._node.get_logger().warn(
                    "home trajectory was accepted after it had already been "
                    "cancelled; cancelling the stale goal."
                )
                self._cancel_handle(handle)
                return
            self._goal_handle = handle
            handle.get_result_async().add_done_callback(_on_result)

        action_client.send_goal_async(goal).add_done_callback(_on_goal_response)

    def cancel_home_trajectory(self) -> None:
        """Best-effort cancel of the in-flight home goal (move deadline expired)."""
        # Bump FIRST: this both discards the pending result and arms the
        # "accepted after cancel" branch in _on_goal_response above.
        self._traj_seq += 1
        handle = self._goal_handle
        self._goal_handle = None
        if handle is None:
            return
        self._cancel_handle(handle)

    def _cancel_handle(self, handle) -> None:
        """Async cancel, log-only: there is nothing better to do if it fails."""
        try:
            future = handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001 -- best-effort by contract
            self._node.get_logger().error(
                "home trajectory cancel could not be sent: {}".format(exc)
            )
            return
        future.add_done_callback(self._on_cancel_done)

    def _on_cancel_done(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(
                "home trajectory cancel failed: {}".format(exc)
            )
            return
        accepted = bool(getattr(response, "goals_canceling", []))
        self._node.get_logger().warn(
            "home trajectory cancel {}".format(
                "accepted" if accepted else "NOT accepted (goal already finished?)"
            )
        )

    # ---- step 4: the gripper --------------------------------------------- #
    def command_gripper_open(self) -> None:
        """Publish one OPEN command (0.0) to /robotiq_gripper/command_percent.

        One publish per call by contract; HomeMoveController re-calls this every
        tick while OPENING_GRIPPER, because the topic has no ack and a single
        message can be lost to a subscriber that has not matched yet.

        THE PUBLISHER IS LONG-LIVED (created in install_home_move_io, not here)
        and that is load-bearing, not incidental: a publisher created, used once
        and destroyed loses the message to DDS discovery often enough to measure
        -- 3 of 7 sends on 2026-08-06. By the time a GO HOME runs, this one has
        been matched with the Robotiq driver for the whole life of the GUI.
        Between that and home_move's GRIPPER_OPEN_REASSERT_S settle window, the
        driver sees a stream of identical OPEN commands rather than one message
        it is free to drop.
        """
        message = Float32()
        message.data = float(GRIPPER_OPEN_VALUE)
        self._node._gripper_cmd_pub.publish(message)


class HomeMoveIoMixin:
    """Everything an rclpy Node needs to host a HomeMoveController.

    Mix it in AHEAD of the Node class and call :meth:`install_home_move_io`
    from ``__init__`` (module docstring, "HOW A NODE USES IT").  A mixin rather
    than a free ``install_home_move_io(node, ...)`` function because the
    ``latest_*`` readers, the subscription callbacks and the three Qt-facing
    methods are then REAL methods on the node class -- discoverable, overridable
    and identical in shape to what task_gui_node.py had before the split, which
    kept that node's diff to "delete the block, add the base class and one
    call" -- whereas a function would have to monkey-patch bound methods onto an
    instance.

    No ``__init__`` here, deliberately: rclpy.Node.__init__ takes a node name and
    keyword options, GelloRecorderGuiNode.__init__ takes a dozen recorder
    kwargs, and a cooperative mixin constructor would have to forward all of
    them blind.  An explicit install call from the subclass's __init__ is the
    same amount of typing and far less magic.

    Every attribute this creates is named ``_home_*`` / ``_svc_home`` /
    ``_gripper_cmd_pub`` / ``_speed_scale_*`` -- the exact names
    :class:`HomeMoveOps` reaches into (see :class:`HomeMoveIoNode`).
    """

    def install_home_move_io(
        self,
        *,
        target_joints: Sequence[float] = HOME_JOINTS,
        target_label: str = "HOME",
        arm_pause_client=None,
        gripper_pause_client=None,
        arm_pause_service: str = ARM_PAUSE_SERVICE,
        gripper_pause_service: str = GRIPPER_PAUSE_SERVICE,
        list_controllers_service: str = LIST_CONTROLLERS_SERVICE,
        switch_controller_service: str = SWITCH_CONTROLLER_SERVICE,
        trajectory_action: str = HOME_TRAJECTORY_ACTION,
        gripper_command_topic: str = GRIPPER_COMMAND_TOPIC,
        gripper_position_topic: str = GRIPPER_POSITION_TOPIC,
        joint_states_topic: str = JOINT_STATES_TOPIC,
        speed_scaling_topic: str = SPEED_SCALING_TOPIC,
        tick_period_s: float = _HOME_TICK_PERIOD_S,
    ) -> None:
        """Create the caches, subscriptions, clients, publisher, sequencer and tick.

        Call ONCE from the node's ``__init__``, after the Node is constructed.
        ``target_joints`` goes straight to HomeMoveController (validated there:
        six finite floats in UR_JOINT_ORDER; HOME_JOINTS by default), and so
        does ``target_label`` -- the name of that pose in the operator-facing
        status messages ("HOME" by default; the EEF GUI passes "START POSE").

        ``arm_pause_client`` / ``gripper_pause_client``: pass EXISTING
        std_srvs/Trigger clients for the bridge pause services when the node
        already owns some (the task recorder's base class does) -- identical
        services with identical semantics, so a second client would only add
        graph entities.  When None, clients are created on
        ``arm_pause_service`` / ``gripper_pause_service``.

        Everything else is a graph name with the production value as default.
        The resolved names are kept in ``self._home_graph_names`` (for the
        node's own log line and for HomeMoveOps' "unavailable" messages).

        After this returns, ``self._home_controller`` is the sequencer; a node
        that needs a tuning override (speed budget, timeouts) may replace it
        with ``HomeMoveController(self._home_ops, ...)`` -- still inside
        __init__, before anything spins, so no lock is needed for that swap.
        ⚠️ Re-pass ``target_joints=`` and ``target_label=`` when you do: a
        bare ``HomeMoveController(self._home_ops)`` silently reverts the
        target to HOME_JOINTS, which for the EEF GUI means the arm goes to
        the recorder's HOME instead of the start pose with no error anywhere.
        This method is meant to be called exactly ONCE per node; a second
        call would create a second set of subscriptions and timers.
        """
        # --- latest-value caches ------------------------------------------ #
        # Separate from whatever display caches the host node already keeps
        # (GelloRecorderGuiNode's _state_lock/_ur_q, for instance) on purpose.
        # A display cache happily yields a vector of Nones when a JointState
        # carries no positions. The homing path must be able to say "I do not
        # know where the arm is" and refuse to move, which is exactly what
        # home_move.reorder_joint_positions() returns (None), and reusing that
        # one mapper keeps the target math and the sequencer in agreement about
        # joint order.
        # Each cached value carries the time.monotonic() of its arrival: a value
        # with no freshness stamp cannot be distinguished from a fossil left by
        # a publisher that has since died (see the _*_STALE_S constants).
        self._home_state_lock = threading.Lock()
        self._home_ur_q: Optional[List[float]] = None
        self._home_ur_q_t: Optional[float] = None
        self._home_grip_pos: Optional[float] = None
        self._home_grip_pos_t: Optional[float] = None
        # Speed scaling: cached as the raw published PERCENT plus its receipt
        # time; the conversion to a fraction happens in the reader.
        self._speed_scale_pct: Optional[float] = None
        self._speed_scale_t: Optional[float] = None
        # depth 5, not 20: /joint_states publishes at ~100 Hz, and this node
        # stamps what it receives with time.monotonic() to decide whether the
        # pose is FRESH. A deep queue would hand it a 0.2 s-old sample and let
        # it call that fresh -- the same defect that back-dated the recorded
        # rows (gello_recorder.spin_health). Latest-wins is what a liveness
        # check wants.
        self.create_subscription(
            JointState, joint_states_topic, self._on_home_joint_states,
            QOS_DEPTH_ROBOT_STATE
        )
        self.create_subscription(
            Float32, gripper_position_topic, self._on_home_gripper_position,
            QOS_DEPTH_GRIPPER
        )
        # The broadcaster may simply not be loaded (some robot configs do not
        # spawn it) -- subscribing to a topic nobody publishes is free, and
        # latest_speed_scale() then reports None forever, which is exactly the
        # "unknown" case HomeMoveController already handles.
        self.create_subscription(
            Float64, speed_scaling_topic, self._on_speed_scaling, QOS_DEPTH_STATUS
        )

        # --- control-path clients ----------------------------------------- #
        # NAMED _svc_home, NOT _clients: rclpy.Node uses self._clients
        # internally and shadowing it corrupts the executor and destroy_node
        # (the same gotcha documented on GelloRecorderGuiNode._svc_teleop).
        #
        # The two pause entries may be ALIASES of clients the node already owns
        # rather than new ones -- identical services with identical semantics,
        # so a second client would only add graph entities. The alias is taken
        # once, here, so the coupling to the caller's clients lives in one place.
        if arm_pause_client is None:
            arm_pause_client = self.create_client(Trigger, arm_pause_service)
        if gripper_pause_client is None:
            gripper_pause_client = self.create_client(Trigger, gripper_pause_service)
        self._svc_home = {
            "arm_pause": arm_pause_client,
            "grip_pause": gripper_pause_client,
            "list_controllers": self.create_client(
                ListControllers, list_controllers_service
            ),
            "switch_controller": self.create_client(
                SwitchController, switch_controller_service
            ),
        }
        self._home_traj_client = ActionClient(
            self, FollowJointTrajectory, trajectory_action
        )
        # LONG-LIVED by design -- see HomeMoveOps.command_gripper_open for the
        # measured DDS-discovery loss a per-call publisher suffers.
        self._gripper_cmd_pub = self.create_publisher(
            Float32, gripper_command_topic, 10
        )
        self._home_graph_names = {
            "arm_pause": getattr(arm_pause_client, "srv_name", arm_pause_service),
            "grip_pause": getattr(
                gripper_pause_client, "srv_name", gripper_pause_service
            ),
            "list_controllers": list_controllers_service,
            "switch_controller": switch_controller_service,
            "trajectory_action": trajectory_action,
            "gripper_command_topic": gripper_command_topic,
            "gripper_position_topic": gripper_position_topic,
            "joint_states_topic": joint_states_topic,
            "speed_scaling_topic": speed_scaling_topic,
        }

        # --- the sequencer ------------------------------------------------ #
        self._home_lock = threading.Lock()
        self._home_ops = HomeMoveOps(self)
        # No tuning overrides: HomeMoveController's own defaults are the single
        # source of truth for the speed budget and the timeouts.
        self._home_controller = HomeMoveController(
            self._home_ops, target_joints=target_joints, target_label=target_label
        )
        # Created LAST: its callback reads everything above, and an rclpy timer
        # can fire as soon as it exists once something spins us.
        self._home_timer = self.create_timer(float(tick_period_s), self._home_tick)

    # ---- subscription callbacks (spin thread) ---------------------------- #
    def _on_home_joint_states(self, msg: JointState) -> None:
        """Cache /joint_states remapped BY NAME into UR order.

        /joint_states is NOT published in UR order, so an index-based read
        silently permutes the pose -- and a permuted pose produces a perfectly
        plausible-looking trajectory to the wrong place.
        """
        try:
            positions = reorder_joint_positions(list(msg.name), list(msg.position))
        except Exception as exc:  # noqa: BLE001 -- see below
            # An exception escaping a subscription callback propagates out of
            # rclpy.spin(), where the GUI's spin thread swallows it: the ROS
            # thread would die silently and the GUI would freeze on stale
            # data. Degrade to "pose unknown" (which refuses to home) instead.
            self.get_logger().warn(
                "could not map /joint_states into UR order: {}".format(exc),
                throttle_duration_sec=5.0,
            )
            return
        if positions is None or len(positions) != len(UR_JOINT_ORDER):
            return  # incomplete message: keep the last known-good pose
        now = time.monotonic()
        with self._home_state_lock:
            self._home_ur_q = [float(p) for p in positions]
            self._home_ur_q_t = now

    def _on_home_gripper_position(self, msg: Float32) -> None:
        now = time.monotonic()
        with self._home_state_lock:
            self._home_grip_pos = float(msg.data)
            self._home_grip_pos_t = now

    def _on_speed_scaling(self, msg: Float64) -> None:
        """Cache the raw speed-scaling PERCENT + its receipt time.

        Stored raw and converted in the reader, so the recorded quantity is
        exactly what the broadcaster said and only one place (documented at
        _SPEED_SCALE_PERCENT_PER_FRACTION) knows about the unit.
        """
        now = time.monotonic()
        with self._home_state_lock:
            self._speed_scale_pct = float(msg.data)
            self._speed_scale_t = now

    # ---- cache readers (any thread) -------------------------------------- #
    @staticmethod
    def _fresh(stamp: Optional[float], window_s: float, now: float) -> bool:
        """True iff `stamp` exists and is within `window_s` of `now`.

        A negative age (a stamp from the future) cannot happen with
        time.monotonic() on one machine, but is treated as fresh rather than
        stale so a clock oddity can never make the sequencer refuse to move.
        """
        return stamp is not None and (now - stamp) <= window_s

    def latest_ur_joints(self) -> Optional[List[float]]:
        """Copy of the latest arm pose in UR order, or None if unknown/STALE.

        Stale (older than _UR_JOINTS_STALE_S) counts as unknown: this value
        picks the branch-cut-safe target, so a pose the arm has since left is
        not merely unhelpful, it is the input that produces a plausible-looking
        command to the wrong winding. HomeMoveController refuses to start
        ("no robot joint states received yet") and fails an in-flight move
        ("robot joint states went stale before the move") on None -- both of
        which are the correct reading of a driver that stopped publishing.
        """
        now = time.monotonic()
        with self._home_state_lock:
            if self._home_ur_q is None:
                return None
            if not self._fresh(self._home_ur_q_t, _UR_JOINTS_STALE_S, now):
                return None
            return list(self._home_ur_q)

    def latest_gripper_position(self) -> Optional[float]:
        """Latest /robotiq_gripper/position_percent, or None if unknown/STALE.

        robotiq_gripper_modbus_node publishes this ONLY while its modbus link is
        up, so without the freshness gate the last value before a disconnect
        sits here forever and HomeMoveController would confirm "gripper open"
        from a reading nobody can still see. Returning None instead turns that
        into its warn-only "could not confirm the gripper opened" path, which is
        the honest answer.
        """
        now = time.monotonic()
        with self._home_state_lock:
            if not self._fresh(
                self._home_grip_pos_t, _GRIPPER_POSITION_STALE_S, now
            ):
                return None
            return self._home_grip_pos

    def latest_speed_scale(self) -> Optional[float]:
        """UR speed scaling as a FRACTION in (0, 1], or None if unknown/stale.

        Conversion and validation live here (see SPEED_SCALING_TOPIC's comment
        for how the percent unit was established on this system):

        * nothing received, or older than _SPEED_SCALE_STALE_S -> None;
        * non-finite -> None;
        * <= 0 -> None. Zero is a real reading (slider at zero, or a protective
          stop) but it is not in the contract's (0, 1], and handing the
          sequencer a zero-ish divisor would turn the move deadline into
          effectively no deadline at all. "Unknown" keeps the fallback deadline,
          which still bounds a move that is going nowhere;
        * above 1.0 -> clamped to 1.0, so float noise on a 100% slider cannot
          hand back an out-of-contract value.

        NEVER raises: the caller treats any failure as "unknown" anyway, but
        this is also read from HomeMoveOps.speed_scale() which is documented as
        non-raising.
        """
        now = time.monotonic()
        with self._home_state_lock:
            percent = self._speed_scale_pct
            if not self._fresh(self._speed_scale_t, _SPEED_SCALE_STALE_S, now):
                return None
        if percent is None or not math.isfinite(percent):
            return None
        fraction = percent / _SPEED_SCALE_PERCENT_PER_FRACTION
        if fraction <= 0.0:
            return None
        return min(fraction, 1.0)

    # ---- the 10 Hz sequencer tick (spin thread) -------------------------- #
    def _home_tick(self) -> None:
        """Advance HomeMoveController. The ONLY place tick() is ever called."""
        try:
            with self._home_lock:
                self._home_controller.tick()
        except Exception as exc:  # noqa: BLE001
            # Same reasoning as the subscription callback above: letting this
            # escape kills the whole spin thread, taking the live preview and
            # the recording with it. A stuck sequencer is a stuck BUTTON; a dead
            # spin thread is a dead session.
            self.get_logger().error(
                "GO HOME tick raised (sequencer stalled): {}".format(exc),
                throttle_duration_sec=5.0,
            )

    # ---- public API for the Qt thread (thread-safe) ---------------------- #
    def request_go_home(self) -> bool:
        """Ask for a GO-HOME. False means it was refused (already running, or
        the arm's pose is unknown); the reason shows up in get_home_status()."""
        with self._home_lock:
            return bool(self._home_controller.request())

    def request_stop_home(self) -> bool:
        """Operator STOP for an in-flight GO HOME. False means there was none.

        Same lock, same shape and same thread as :meth:`request_go_home` -- the
        Qt thread calls it from a button handler. HomeMoveController.abort()
        only RECORDS the request; the transition happens on the next tick()
        (~100 ms at 10 Hz), it routes through the fail-closed FPC restore, and
        it never resumes a bridge.

        BEST-EFFORT AND NOT AN E-STOP: all it can do is ask the action server to
        cancel, after which scaled_joint_trajectory_controller decelerates on
        its own schedule -- the arm keeps moving briefly and stops partway
        between wherever it was and HOME.

        A plain direct call on purpose. If abort() were missing this raises
        AttributeError into the Qt handler, which is the loud failure we want;
        a getattr()/hasattr() dance would degrade a missing STOP into a button
        that silently does nothing while the arm keeps moving.
        """
        with self._home_lock:
            return bool(self._home_controller.abort())

    def get_home_status(self) -> dict:
        """Snapshot ``{state, active, message, duration_s}`` for the Qt timer."""
        with self._home_lock:
            return dict(self._home_controller.status())
