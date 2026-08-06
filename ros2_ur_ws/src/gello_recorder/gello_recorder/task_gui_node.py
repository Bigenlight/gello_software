#!/usr/bin/env python3
"""rclpy Node backing the TASK recorder GUI (recorder + a forced GO-HOME path).

Subclass of :class:`GelloRecorderGuiNode` (gello_gui_node.py), exactly like
policy_run_gui_node.py is. It inherits EVERY recording-related behaviour
unchanged -- the always-active subscriptions, the dual-camera preview and state
snapshot, the camera-warmup gate, the start/stop take lifecycle and the teleop
pause/resume panel -- and adds two things on top: the GO HOME / STOP HOME
control path, and the fixed-rate ``synchronized`` sampler the GUI recorder has
never had (see the next section).

THE SYNCHRONIZED SAMPLER (a table the GUI recorder never filled)
----------------------------------------------------------------
:class:`RecordingSession` writes nine HDF5 tables. Eight of them are native-rate
(one row per incoming message) and the base node fills all eight. The ninth,
``synchronized``, is the WIDE time-aligned analysis table: at ``sample_rate_hz``
it stores the latest value of EVERY signal in one row, plus the cam1/cam2 MP4
frame index at that instant -- which is the only thing that lets anyone
cross-reference the videos against the signals offline. Only the headless
recorder (gello_ur_recorder_node.py) ever wrote it, via its own fixed-rate
``_on_sample`` timer; every take recorded through a GUI therefore has an EMPTY
``synchronized`` group. :meth:`TaskRecorderGuiNode._on_sample` here is that
timer, mirroring the headless node's semantics exactly (same
``RecordingSession.write_sample`` call, same value layout, same "not seen yet ->
None -> NaN" convention) with two differences forced by the GUI's multi-take
model:

* it writes ONLY while a take is recording, taking the base's ``_session_lock``
  the same way every base callback does, so a row can never land in a closed or
  absent session and start/stop cannot race the sampler; and
* the cam frame indices are read from the LIVE session rather than cached in the
  node, because each take gets a brand-new RecordingSession whose MP4 writers
  restart at frame 0 -- a node-level cache would leak the previous take's index
  into the first rows of the next one. See :meth:`_cam_frame_index`.

Nothing about which frames reach the MP4s, or when, is touched: the base's
``_on_cam`` remains the only writer of camera frames and is not overridden.

GRIPPER-MODE PROVENANCE (the per-take ``gripper_mode.json`` sidecar)
--------------------------------------------------------------------
The GELLO->Robotiq bridge has an OPT-IN ``discrete_mode`` that folds the
spring-loaded trigger to its endpoints, so the recorded ``grip_cmd`` column
stops being a continuous 0..1 value and becomes BINARY (0.0 = OPEN,
1.0 = CLOSED). ``gello_grip`` (the raw trigger) and ``grip_pos`` (the measured
position) stay continuous either way -- only ``grip_cmd`` changes meaning.

That matters because ``grip_cmd`` is an ACTION CHANNEL for the ACT / Diffusion /
FM + LeRobot export path, which normalises it MEAN_STD-style. Mixing continuous
and binary takes silently shifts the statistics baked into a checkpoint, and
until now NOTHING in a take said which mode produced it: the existing corpus is
continuous, new takes may be binary, and the files are indistinguishable. The
information is free to capture while recording and unrecoverable afterwards.

So every take gets one extra file next to ``vectors.h5``::

    take_07_20260731_204512/gripper_mode.json

built by the pure :func:`build_gripper_mode_record` from two independent
sources, in this order of authority:

* ``/gello_gripper_bridge/discrete_state`` (String) sampled THROUGHOUT the take
  -- what actually happened while these rows were being written. ``DISABLED``
  is the only token that means continuous; ``UNKNOWN`` / ``OPEN`` / ``CLOSED`` /
  ``RAMPING`` all mean the latch was in use. Seeing BOTH families inside one
  take is the mixed-take case, and it is flagged rather than averaged away: it
  is the one shape that would poison a dataset with nothing to notice.
* the bridge's ``discrete_mode`` / ``discrete_open_at`` / ``discrete_close_at``
  PARAMETERS, read once at startup through the same non-blocking
  ``call_async`` + done-callback discipline everything else here uses. They are
  the only source for the thresholds, and the fallback for the mode when the
  topic said nothing. An absent bridge (a policy-deploy stack does not start
  one) or an older bridge without those parameters degrades to "unknown" --
  never to a guess.

The sidecar is written when a take is FINALISED and a failure to write it is
logged, never raised: this file is provenance, and losing a recording to save
its provenance would be an absurd trade. Nothing in ``RecordingSession``, the
HDF5 schema or ``metadata.json`` is touched.

WHAT GO HOME DOES
-----------------
The sequencing (and every deadline, every fail-closed rule) lives in
``gello_recorder/home_move.py``, which is deliberately pure Python so it can be
tested without a robot. THIS module is only the ROS I/O adapter for it: a
duck-typed ``ops`` object plus the lock + timer that drive the state machine.
The sequence it drives is:

    PAUSING -> SWITCHING_TO_JTC -> MOVING -> OPENING_GRIPPER -> RESTORING_FPC

i.e. force-pause both teleop bridges, hand the joints from
forward_position_controller to scaled_joint_trajectory_controller, drive one
FollowJointTrajectory goal to the fixed HOME pose, open the Robotiq 2F-85, and
give the joints back to forward_position_controller.

Teleop is deliberately LEFT PAUSED afterwards. Re-engaging is an explicit
operator action in the separate EEF GUI (its ENGAGE button chains
eef_resume -> pos_scale -> eef_engage and re-anchors at the CURRENT pose, so
homing cannot break it). NOTHING in this module ever calls a resume service.

THREADING CONTRACT (inherited from the base node, and extended here)
--------------------------------------------------------------------
* Every ROS callback -- subscriptions, the 10 Hz home timer and the
  ``sample_rate_hz`` synchronized-sampler timer -- runs on the single background
  rclpy spin thread that task_recorder_gui.py starts. The Qt thread only ever
  calls the public, lock-protected methods :meth:`request_go_home`,
  :meth:`request_stop_home` and :meth:`get_home_status`.
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
"""

import json
import math
import os
import threading
import time
from datetime import datetime
from typing import Callable, List, Optional, Sequence

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from controller_manager_msgs.srv import ListControllers, SwitchController
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64, String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gello_recorder.gello_gui_node import GelloRecorderGuiNode
from gello_recorder.home_move import (
    FPC,
    GRIPPER_OPEN_VALUE,
    STJC,
    UR_JOINT_ORDER,
    HomeMoveController,
    reorder_joint_positions,
)

# --- ROS graph names the GO-HOME path talks to ----------------------------- #
LIST_CONTROLLERS_SERVICE = "/controller_manager/list_controllers"
SWITCH_CONTROLLER_SERVICE = "/controller_manager/switch_controller"
# The trajectory always goes through the SCALED joint trajectory controller:
# forward_position_controller does not interpolate, so it cannot execute a
# time-parameterised move (see gello_move_to_start_node.py's module docstring).
HOME_TRAJECTORY_ACTION = "/{}/follow_joint_trajectory".format(STJC)
GRIPPER_COMMAND_TOPIC = "/robotiq_gripper/command_percent"
GRIPPER_POSITION_TOPIC = "/robotiq_gripper/position_percent"
JOINT_STATES_TOPIC = "/joint_states"
# The UR driver's speed-scaling broadcaster. READ OFF THIS SYSTEM, not guessed:
#   * /opt/ros/humble/share/ur_robot_driver/config/ur_controllers.yaml declares
#     `speed_scaling_state_broadcaster: type: ur_controllers/SpeedScalingStateBroadcaster`
#     and ur_control.launch.py spawns it by that name (so the node name, and
#     hence the topic namespace, is speed_scaling_state_broadcaster);
#   * ur_controllers 2.8.1's speed_scaling_state_broadcaster.hpp declares
#     `rclcpp::Publisher<std_msgs::msg::Float64>` and the shipped library's
#     string table contains the node-relative topic "~/speed_scaling".
# => /speed_scaling_state_broadcaster/speed_scaling, std_msgs/msg/Float64,
#    field `data`.
SPEED_SCALING_TOPIC = "/speed_scaling_state_broadcaster/speed_scaling"
# --- the gripper bridge's discrete-latch channel (dataset provenance) ------- #
# READ-ONLY, both of them: this node never commands the gripper bridge, it only
# records what mode the bridge was in while a take was being written.
GRIPPER_DISCRETE_STATE_TOPIC = "/gello_gripper_bridge/discrete_state"
GRIPPER_BRIDGE_PARAMETERS_SERVICE = "/gello_gripper_bridge/get_parameters"
#: Sidecar file written into each take directory. A SEPARATE FILE on purpose --
#: RecordingSession, the HDF5 schema and metadata.json are shared with the other
#: recorder and must stay byte-identical, and a JSON file next to vectors.h5 is
#: readable by anything (including a human, six months later) without h5py.
GRIPPER_MODE_SIDECAR_NAME = "gripper_mode.json"
_GRIPPER_MODE_SCHEMA = "gello_recorder/gripper_mode/1"
# The bridge parameters that define the mode. discrete_mode alone decides
# continuous-vs-discrete; the two thresholds are the hysteresis band and are
# recorded because a take produced under different thresholds is a different
# labelling of the same trigger sweep.
_GRIPPER_BRIDGE_PARAM_NAMES = (
    "discrete_mode",
    "discrete_open_at",
    "discrete_close_at",
)
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
# (wrapped_nearest(HOME_JOINTS, current)), and a target computed from a pose the
# arm has since left is precisely the ~2*pi long-way-round hazard. Cheap gate,
# unbounded downside if omitted; HomeMoveController already has the words for it
# ("robot joint states went stale before the move").
_UR_JOINTS_STALE_S = 1.0
# The speed-scaling broadcaster publishes at state_publish_rate = 100.0 Hz, so
# 1.0 s is 100 periods. Stale here is benign by design -- speed_scale() simply
# reports None and HomeMoveController falls back to MIN_ASSUMED_SPEED_SCALE.
_SPEED_SCALE_STALE_S = 1.0

# Default rate of the `synchronized` wide-table sampler, matching
# gello_ur_recorder_node's `sample_rate_hz` parameter default.
_DEFAULT_SAMPLE_RATE_HZ = 100.0

# ---- ~/discrete_state vocabulary, classified by what it says about the MODE - #
# DISABLED is the ONLY token that means "continuous value forwarded" -- the
# bridge reports it both for discrete_mode:=false and for "asked for discrete
# but the thresholds were invalid so we fell back", which from a recorded
# take's point of view are the same thing. Everything else means the latch was
# in use, INCLUDING UNKNOWN (discrete, no threshold crossed yet) and RAMPING
# (being added to this topic as this is written).
#
# ANYTHING NOT LISTED maps to "unknown", i.e. "the topic was seen but this token
# does not tell us the mode". That is deliberate and must stay: the vocabulary
# is owned by another module and is actively growing, and a new token silently
# classified as continuous-or-discrete is exactly the mislabelled-dataset
# failure this whole sidecar exists to prevent. Not knowing is recoverable;
# confidently wrong is not.
_DISCRETE_STATE_FAMILY = {
    "DISABLED": "continuous",
    "UNKNOWN": "discrete",
    "OPEN": "discrete",
    "CLOSED": "discrete",
    "RAMPING": "discrete",
}

# What `grip_cmd` MEANS under each resolved mode, spelled out in the sidecar so
# a consumer never has to come back here to find out.
_GRIP_CMD_SEMANTICS = {
    "continuous": (
        "grip_cmd is CONTINUOUS in [0.0, 1.0] (0.0=OPEN, 1.0=CLOSED)"
    ),
    "discrete": (
        "grip_cmd is BINARY: exactly 0.0 (OPEN) or 1.0 (CLOSED). gello_grip "
        "and grip_pos are still continuous."
    ),
    "unknown": (
        "grip_cmd semantics UNKNOWN -- do NOT fold this take into a "
        "mean/std-normalised action channel without checking it by hand"
    ),
}

# How the once-at-startup bridge-parameter read is paced. The bridge may come up
# after this GUI (or never), so the read is a polling timer that fires the
# actual call_async the first time the service is ready and gives up after
# _ATTEMPTS ticks -- bounded, so a stack with no gripper bridge settles on a
# recorded "unknown" instead of probing forever.
_GRIPPER_PARAM_PROBE_PERIOD_S = 2.0
_GRIPPER_PARAM_PROBE_ATTEMPTS = 15

# controller_manager_msgs/srv/SwitchController strictness enum. STRICT means the
# switch either happens completely or not at all -- a partial switch would leave
# the arm with no controller able to move it.
_STRICT = 2
# How long controller_manager may take to realize the switch internally. Same
# value GelloMoveToStart._switch_controllers uses.
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


def discrete_state_family(token) -> str:
    """``"continuous"`` / ``"discrete"`` / ``"unknown"`` for one state token.

    "unknown" means "seen, but says nothing about the mode" -- see
    _DISCRETE_STATE_FAMILY for why an unrecognised token must land there rather
    than be guessed into one of the other two.
    """
    return _DISCRETE_STATE_FAMILY.get(str(token).strip().upper(), "unknown")


def _parameter_scalar(value):
    """One rcl_interfaces ParameterValue -> a plain Python scalar, or None.

    None covers PARAMETER_NOT_SET, which is exactly what a node returns for a
    parameter it never declared -- i.e. what an OLDER gripper bridge answers for
    ``discrete_mode``. Integers are widened to float because the thresholds are
    doubles that a command line can perfectly well hand over as ``0`` or ``1``.
    """
    kind = getattr(value, "type", ParameterType.PARAMETER_NOT_SET)
    if kind == ParameterType.PARAMETER_BOOL:
        return bool(value.bool_value)
    if kind == ParameterType.PARAMETER_DOUBLE:
        return float(value.double_value)
    if kind == ParameterType.PARAMETER_INTEGER:
        return float(value.integer_value)
    return None


def build_gripper_mode_record(state_counts, first_state, last_state, params) -> dict:
    """Assemble the ``gripper_mode.json`` payload. Pure: no ROS, no I/O.

    ``state_counts`` maps each ~/discrete_state token seen DURING THE TAKE to
    how many times it arrived (empty = the topic was never heard);
    ``first_state`` / ``last_state`` are the first and last tokens of the take;
    ``params`` is this node's snapshot of the once-at-startup parameter read.

    HOW THE MODE IS DECIDED, and why in this order:

    * The topic wins, because it is evidence from INSIDE the take, whereas the
      parameters were read once at startup and a bridge can be relaunched in
      the other mode afterwards.
    * BOTH families inside one take -> ``mode="unknown"`` AND
      ``mode_changed_mid_take=True``. This is the one case that would silently
      poison a dataset, so it is reported as not-knowing rather than as
      whichever family happened to arrive last: a consumer filtering on a known
      mode then excludes the take by default, and one that reads the flag sees
      exactly what happened.
    * Only if the topic said nothing usable do the parameters decide.
    * If neither does, the answer is "unknown" -- never a guess.

    ``parameters_agree`` compares the two sources whenever both spoke. False
    means the bridge was restarted into the other mode BETWEEN startup and this
    take: not fatal to the take (the topic is right), but worth seeing.
    """
    counts = {str(token): int(count) for token, count in (state_counts or {}).items()}
    params = dict(params or {})
    families = {discrete_state_family(token) for token in counts}

    mixed = "continuous" in families and "discrete" in families
    topic_mode = None
    if not mixed:
        if "continuous" in families:
            topic_mode = "continuous"
        elif "discrete" in families:
            topic_mode = "discrete"

    param_mode = None
    if params.get("available") and isinstance(params.get("discrete_mode"), bool):
        param_mode = "discrete" if params["discrete_mode"] else "continuous"

    if mixed:
        mode, source = "unknown", "state_topic"
    elif topic_mode is not None:
        mode, source = topic_mode, "state_topic"
    elif param_mode is not None:
        mode, source = param_mode, "parameters"
    else:
        mode, source = "unknown", "none"

    agree = None
    if param_mode is not None and topic_mode is not None:
        agree = param_mode == topic_mode

    return {
        "schema": _GRIPPER_MODE_SCHEMA,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "mode_source": source,
        "mode_changed_mid_take": bool(mixed),
        "grip_cmd_semantics": _GRIP_CMD_SEMANTICS[mode],
        "discrete_open_at": params.get("discrete_open_at"),
        "discrete_close_at": params.get("discrete_close_at"),
        "state_topic": GRIPPER_DISCRETE_STATE_TOPIC,
        "state_topic_seen": bool(counts),
        "state_first": first_state,
        "state_last": last_state,
        "state_counts": counts,
        "parameters": params,
        "parameters_agree": agree,
    }


class HomeMoveOps:
    """The duck-typed ``ops`` object HomeMoveController drives -- all ROS I/O.

    Deliberately intimate with :class:`TaskRecorderGuiNode` (it reaches into the
    node's clients and caches): the two are one unit split in half only so the
    sequencing in home_move.py stays ROS-free and laptop-testable. Every method
    here is either non-blocking or a pure cache read.

    Threading: ``current_joints()`` and ``gripper_position()`` are called from
    ``HomeMoveController.request()``, which the Qt thread reaches through
    :meth:`TaskRecorderGuiNode.request_go_home`, so they read locked caches.
    Everything else runs on the rclpy spin thread only (from ``tick()`` or from
    a future's done-callback), which is why ``_goal_handle`` / ``_traj_seq``
    need no lock -- documented rather than defended, so a future
    MultiThreadedExecutor change is an obvious thing to re-check.
    """

    def __init__(self, node: "TaskRecorderGuiNode") -> None:
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
        :meth:`TaskRecorderGuiNode.latest_speed_scale` for the unit conversion
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
                "running?".format(LIST_CONTROLLERS_SERVICE),
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
                    SWITCH_CONTROLLER_SERVICE, label
                ),
            )
            return

        request = SwitchController.Request()
        # Humble field names (Foxy-era 'start_controllers' is long gone).
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
        HomeMoveController (``wrapped_nearest(HOME_JOINTS, current)``) -- this
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
                    HOME_TRAJECTORY_ACTION, STJC
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

        THE PUBLISHER IS LONG-LIVED (created in the node's __init__, not here)
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


class TaskRecorderGuiNode(GelloRecorderGuiNode):
    """GelloRecorderGuiNode + the GO-HOME/STOP-HOME control path + the
    fixed-rate ``synchronized`` sampler (see module docstring for both).

    ``sample_rate_hz`` is the sampler's rate; it is also exposed as a ROS
    parameter of the same name, matching gello_ur_recorder_node.
    """

    def __init__(
        self,
        cam1_topic: str = "/cam1/cam1/color/image_raw/compressed",
        cam2_topic: str = "/cam2/cam2/color/image_raw/compressed",
        camera_fps: float = 30.0,
        camera_warmup_s: float = 3.0,
        output_root: str = "~/gello_recordings",
        node_name: str = "task_recorder_gui_node",
        sample_rate_hz: float = _DEFAULT_SAMPLE_RATE_HZ,
    ) -> None:
        # The parent wires up every recording subscription, both locks and the
        # take state machine; nothing below touches any of that.
        super().__init__(
            cam1_topic=cam1_topic,
            cam2_topic=cam2_topic,
            camera_fps=camera_fps,
            camera_warmup_s=camera_warmup_s,
            output_root=output_root,
            node_name=node_name,
        )

        # --- synchronized sampler: rate ----------------------------------- #
        # Exposed BOTH ways on purpose: as a ctor kwarg because that is how this
        # node's every other knob arrives (task_recorder_gui.py constructs it
        # with keywords and never touches ROS parameters), and as a ROS
        # parameter named exactly like gello_ur_recorder_node's, so
        # `--ros-args -p sample_rate_hz:=50.0` behaves the same on both
        # recorders. The kwarg is the parameter's DEFAULT, so passing neither
        # gives 100 Hz and passing both lets the command line win.
        declared = self.declare_parameter(
            "sample_rate_hz", float(sample_rate_hz)
        ).value
        try:
            self.sample_rate_hz = float(declared)
        except (TypeError, ValueError):
            self.sample_rate_hz = float(sample_rate_hz)
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            self.get_logger().warn(
                "sample_rate_hz={} is not a positive number; falling back to "
                "{:.1f} Hz".format(declared, _DEFAULT_SAMPLE_RATE_HZ)
            )
            self.sample_rate_hz = _DEFAULT_SAMPLE_RATE_HZ

        # --- GO HOME: latest-value caches --------------------------------- #
        # Separate from the base node's _state_lock/_state fields on purpose.
        # The base's `ur_q` is built for DISPLAY: it happily yields a vector of
        # Nones when a JointState carries no positions. The homing path must be
        # able to say "I do not know where the arm is" and refuse to move, which
        # is exactly what home_move.reorder_joint_positions() returns (None),
        # and reusing that one mapper keeps the target math and the sequencer in
        # agreement about joint order.
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
        self.create_subscription(
            JointState, JOINT_STATES_TOPIC, self._on_home_joint_states, 20
        )
        self.create_subscription(
            Float32, GRIPPER_POSITION_TOPIC, self._on_home_gripper_position, 20
        )
        # The broadcaster may simply not be loaded (some robot configs do not
        # spawn it) -- subscribing to a topic nobody publishes is free, and
        # latest_speed_scale() then reports None forever, which is exactly the
        # "unknown" case HomeMoveController already handles.
        self.create_subscription(
            Float64, SPEED_SCALING_TOPIC, self._on_speed_scaling, 10
        )

        # --- GO HOME: control-path clients -------------------------------- #
        # NAMED _svc_home, NOT _clients: rclpy.Node uses self._clients
        # internally and shadowing it corrupts the executor and destroy_node
        # (the same gotcha documented on the base node's _svc_teleop).
        #
        # The two pause entries are ALIASES of the base node's clients rather
        # than new ones -- identical services with identical semantics, so a
        # second client would only add graph entities. The alias is taken once,
        # here, so the coupling to the base's key names lives in one place.
        self._svc_home = {
            "arm_pause": self._svc_teleop["arm_pause"],
            "grip_pause": self._svc_teleop["grip_pause"],
            "list_controllers": self.create_client(
                ListControllers, LIST_CONTROLLERS_SERVICE
            ),
            "switch_controller": self.create_client(
                SwitchController, SWITCH_CONTROLLER_SERVICE
            ),
        }
        self._home_traj_client = ActionClient(
            self, FollowJointTrajectory, HOME_TRAJECTORY_ACTION
        )
        # The base node already SUBSCRIBES to this topic for recording, so the
        # open command we publish here is captured in the take like any other.
        self._gripper_cmd_pub = self.create_publisher(
            Float32, GRIPPER_COMMAND_TOPIC, 10
        )

        # --- gripper-mode provenance -------------------------------------- #
        # Its OWN lock, and it is never held while _state_lock, _session_lock or
        # _home_lock is held (nor the reverse): the module's rule is one lock at
        # a time, snapshot then act, which is what makes the whole node
        # deadlock-free by construction rather than by inspection.
        #
        # Two things live under it: the LIVE token (for the GUI readout, which
        # polls from the Qt thread) and the PER-TAKE accumulator (written by the
        # spin thread's subscription callback, drained at finalisation). The
        # accumulator is deliberately gated on _take_grip_active rather than on
        # the base's is_recording(), because is_recording() takes _session_lock
        # and taking it from under this one would be exactly the nesting the
        # paragraph above rules out.
        self._grip_mode_lock = threading.Lock()
        self._grip_state: Optional[str] = None
        self._grip_state_t: Optional[float] = None
        self._take_grip_active = False
        self._take_grip_counts = {}
        self._take_grip_first: Optional[str] = None
        self._take_grip_last: Optional[str] = None
        # Replaced wholesale by _finish_gripper_param_probe; shaped like the
        # final record from the start so a take finalised before the probe
        # resolves still writes a well-formed (honestly "unknown") sidecar.
        self._grip_params = {
            "service": GRIPPER_BRIDGE_PARAMETERS_SERVICE,
            "queried": False,
            "available": False,
            "discrete_mode": None,
            "discrete_open_at": None,
            "discrete_close_at": None,
            "note": "not queried yet",
        }
        # Subscribing to a topic nobody publishes is free (same reasoning as the
        # speed-scaling subscription above), and its absence IS the recorded
        # answer: state_topic_seen=false.
        self.create_subscription(
            String, GRIPPER_DISCRETE_STATE_TOPIC, self._on_grip_discrete_state, 10
        )
        self._grip_param_client = self.create_client(
            GetParameters, GRIPPER_BRIDGE_PARAMETERS_SERVICE
        )
        self._grip_param_attempts = 0
        self._grip_param_timer = self.create_timer(
            _GRIPPER_PARAM_PROBE_PERIOD_S, self._probe_gripper_parameters
        )

        # --- GO HOME: the sequencer --------------------------------------- #
        self._home_lock = threading.Lock()
        self._home_ops = HomeMoveOps(self)
        # No tuning overrides: HomeMoveController's own defaults are the single
        # source of truth for the speed budget and the timeouts.
        self._home_controller = HomeMoveController(self._home_ops)
        self._home_timer = self.create_timer(_HOME_TICK_PERIOD_S, self._home_tick)

        # --- the synchronized-table sampler ------------------------------- #
        # Created LAST: its callback reads state the lines above set up, and an
        # rclpy timer can fire as soon as it exists once something spins us.
        # Period expression is verbatim from gello_ur_recorder_node so both
        # recorders clamp a sub-1 Hz request identically.
        self._sample_timer = self.create_timer(
            1.0 / max(1.0, self.sample_rate_hz), self._on_sample
        )

        self.get_logger().info(
            "task_recorder_gui_node up. GO HOME path: {} + {} , {} , {} -> {}. "
            "synchronized sampler @ {:.0f} Hz; speed scaling from {}".format(
                SWITCH_CONTROLLER_SERVICE,
                LIST_CONTROLLERS_SERVICE,
                HOME_TRAJECTORY_ACTION,
                GRIPPER_COMMAND_TOPIC,
                "teleop stays PAUSED (re-engage from the EEF GUI)",
                self.sample_rate_hz,
                SPEED_SCALING_TOPIC,
            )
        )

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
            # rclpy.spin(), where task_recorder_gui._spin_node swallows it: the
            # ROS thread would die silently and the GUI would freeze on stale
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

    # ---- gripper-mode provenance ----------------------------------------- #
    def _on_grip_discrete_state(self, msg: String) -> None:
        """Cache the latest ~/discrete_state token; accumulate it into the take.

        Normalised (stripped + upper-cased) exactly once, here, so every
        consumer downstream -- the sidecar's counts, the family classification
        and the GUI readout -- is looking at the same spelling.
        """
        token = str(msg.data).strip().upper()
        now = time.monotonic()
        with self._grip_mode_lock:
            self._grip_state = token
            self._grip_state_t = now
            if not self._take_grip_active:
                return
            self._take_grip_counts[token] = self._take_grip_counts.get(token, 0) + 1
            if self._take_grip_first is None:
                self._take_grip_first = token
            self._take_grip_last = token

    def _probe_gripper_parameters(self) -> None:
        """Timer: fire the once-only parameter read as soon as it can succeed.

        Polling rather than a single shot because the gripper bridge may come up
        after this GUI. Bounded rather than forever because it may never come up
        at all -- a policy-deploy stack deliberately does not start one -- and
        "unknown" recorded promptly beats a probe that never settles.

        NEVER raises: an exception out of a timer callback propagates out of
        rclpy.spin() and kills the ROS thread, taking the preview, the GO-HOME
        path and the recording with it (same reasoning as _home_tick).
        """
        try:
            if _service_ready(self._grip_param_client):
                # Cancel BEFORE calling: the call is the one-and-only read.
                self._cancel_gripper_param_timer()
                request = GetParameters.Request()
                request.names = list(_GRIPPER_BRIDGE_PARAM_NAMES)
                self._grip_param_client.call_async(request).add_done_callback(
                    self._on_gripper_parameters
                )
                return
            self._grip_param_attempts += 1
            if self._grip_param_attempts < _GRIPPER_PARAM_PROBE_ATTEMPTS:
                return
            self._cancel_gripper_param_timer()
            self._finish_gripper_param_probe(
                False,
                "{} did not appear within {:.0f}s -- no gripper bridge in this "
                "stack (the gripper mode of these takes is UNKNOWN)".format(
                    GRIPPER_BRIDGE_PARAMETERS_SERVICE,
                    _GRIPPER_PARAM_PROBE_PERIOD_S * _GRIPPER_PARAM_PROBE_ATTEMPTS,
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- see the docstring
            self._cancel_gripper_param_timer()
            self._finish_gripper_param_probe(
                False, "gripper parameter probe raised: {}".format(exc)
            )

    def _cancel_gripper_param_timer(self) -> None:
        """Stop the probe timer, idempotently (it fires exactly one read)."""
        timer = self._grip_param_timer
        self._grip_param_timer = None
        if timer is None:
            return
        try:
            timer.cancel()
        except Exception:  # noqa: BLE001 -- shutdown race, nothing to do
            pass

    def _on_gripper_parameters(self, future) -> None:
        """Done-callback for the parameter read. Any failure -> "unknown"."""
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001 -- becomes recorded "unknown"
            self._finish_gripper_param_probe(
                False, "get_parameters failed: {}".format(exc)
            )
            return
        if response is None:
            self._finish_gripper_param_probe(False, "get_parameters: no response")
            return

        values = list(getattr(response, "values", None) or [])
        parsed = {
            name: _parameter_scalar(value)
            for name, value in zip(_GRIPPER_BRIDGE_PARAM_NAMES, values)
        }
        # An older bridge answers PARAMETER_NOT_SET for a parameter it never
        # declared. That is NOT read as "so it must be continuous": the same
        # shape would appear if this service ever moved, and a wrong mode
        # written with confidence is worse than an honest unknown.
        if not isinstance(parsed.get("discrete_mode"), bool):
            self._finish_gripper_param_probe(
                False,
                "the bridge answered but has no discrete_mode parameter (an "
                "older gripper bridge)",
            )
            return
        self._finish_gripper_param_probe(True, "", parsed)

    def _finish_gripper_param_probe(self, available, note, parsed=None) -> None:
        """Store the probe's verdict (once) and say so in the log."""
        parsed = parsed or {}
        with self._grip_mode_lock:
            self._grip_params = {
                "service": GRIPPER_BRIDGE_PARAMETERS_SERVICE,
                "queried": True,
                "available": bool(available),
                "discrete_mode": parsed.get("discrete_mode"),
                "discrete_open_at": parsed.get("discrete_open_at"),
                "discrete_close_at": parsed.get("discrete_close_at"),
                "note": str(note),
            }
        if available:
            self.get_logger().info(
                "gripper bridge parameters: discrete_mode={} open_at={} "
                "close_at={} (recorded in every take's {})".format(
                    parsed.get("discrete_mode"),
                    parsed.get("discrete_open_at"),
                    parsed.get("discrete_close_at"),
                    GRIPPER_MODE_SIDECAR_NAME,
                )
            )
        else:
            self.get_logger().warn(
                "gripper mode unknown from parameters: {}. Takes will still "
                "record what {} reports.".format(note, GRIPPER_DISCRETE_STATE_TOPIC)
            )

    def get_gripper_discrete_status(self) -> dict:
        """Latest ~/discrete_state token + its age, for the Qt refresh tick.

        Age (not a freshness verdict) so the staleness rule stays next to the
        other one in the GUI, where the base window already owns that policy.
        """
        now = time.monotonic()
        with self._grip_mode_lock:
            age = None if self._grip_state_t is None else now - self._grip_state_t
            return {"state": self._grip_state, "age_s": age}

    def _write_gripper_mode_sidecar(self, take_dir) -> Optional[str]:
        """Write this take's gripper_mode.json. Returns the path, or None.

        NEVER RAISES, by design: the sidecar is provenance ABOUT a recording,
        and losing the recording to save its provenance would be an absurd
        trade. Every failure is logged and swallowed.

        Also the one place _take_grip_active is cleared, so the accumulator
        stops growing the moment a take is finalised even if the write itself
        fails.
        """
        try:
            with self._grip_mode_lock:
                self._take_grip_active = False
                counts = dict(self._take_grip_counts)
                first = self._take_grip_first
                last = self._take_grip_last
                params = dict(self._grip_params)
            if not take_dir:
                self.get_logger().error(
                    "no take directory to write {} into".format(
                        GRIPPER_MODE_SIDECAR_NAME
                    )
                )
                return None
            record = build_gripper_mode_record(counts, first, last, params)
            path = os.path.join(take_dir, GRIPPER_MODE_SIDECAR_NAME)
            with open(path, "w") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")
            self.get_logger().info(
                "gripper mode for this take: {} (source={}, mixed={}) -> {}".format(
                    record["mode"],
                    record["mode_source"],
                    record["mode_changed_mid_take"],
                    path,
                )
            )
            return path
        except Exception as exc:  # noqa: BLE001 -- see the docstring
            self.get_logger().error(
                "could not write {} for this take: {} -- the take itself is "
                "intact, but nothing in it records the gripper mode".format(
                    GRIPPER_MODE_SIDECAR_NAME, exc
                )
            )
            return None

    # ---- take lifecycle (base behaviour + the sidecar) -------------------- #
    def start_recording(self) -> str:
        """Base start, then arm a FRESH gripper-mode accumulator.

        Reset AFTER super() so a refused start (cameras not warm, already
        recording) leaves the previous take's evidence alone.
        """
        take_dir = super().start_recording()
        with self._grip_mode_lock:
            self._take_grip_counts = {}
            self._take_grip_first = None
            self._take_grip_last = None
            self._take_grip_active = True
        return take_dir

    def stop_recording(self) -> dict:
        """Base stop, then the sidecar. Return value unchanged from the base.

        Order matters: super() closes the HDF5/MP4 writers and only then is the
        take directory final, so the sidecar lands in a directory nothing else
        is still writing to. super() may raise ("not recording"), in which case
        there is no take to annotate and the raise is passed straight through.
        """
        stats = super().stop_recording()
        self._write_gripper_mode_sidecar(stats.get("session_dir"))
        return stats

    def destroy_node(self):
        """Cover the ONE finalisation path that does not go through stop().

        The base destroy_node closes a still-open session itself (headless
        main(), or a teardown that skipped the GUI's stop). Snapshot the
        directory first so that take gets its sidecar too; if stop_recording()
        already ran there is no session and this is a no-op.
        """
        with self._session_lock:
            session = self._session
        if session is not None:
            self._write_gripper_mode_sidecar(
                getattr(session, "session_dir", None)
            )
        super().destroy_node()

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

    # ---- the synchronized-table sampler (spin thread) -------------------- #
    def _cam_frame_index(self, session, cam_idx: int) -> Optional[int]:
        """Index of the most recent frame written to this take's camN.mp4.

        HOW, and why it is this way. RecordingSession.write_cam1_frame() returns
        the index, but the base node's _on_cam discards it -- and _on_cam must
        stay exactly as it is, because it is the single writer of camera frames
        and other people depend on the recorder it belongs to. So the index is
        read back out of the session instead of being intercepted: each
        Mp4FrameWriter exposes a PUBLIC ``frame_count`` property (the number of
        frames successfully written so far), so ``frame_count - 1`` is the index
        of the latest one. Nothing about which frames are written, or when, is
        touched; this is a pure read.

        Two properties fall out of reading the LIVE session rather than caching
        in the node, and both are wanted:
          * a fresh take gets a fresh RecordingSession whose writers restart at
            0, so the count restarts with it -- a node-level cache would leak
            the previous take's index into the new take's first rows;
          * a frame that failed to decode does not advance frame_count, so this
            keeps reporting the last GOOD index -- identical to the headless
            recorder, which only updates its cache when write_camN_frame()
            returns >= 0.

        Returns None before the first frame of the take, which write_sample
        stores as NaN -- exactly what the headless recorder writes for the same
        situation.

        Caller must hold _session_lock (the session must not be swapped or
        closed underneath this read).
        """
        writer = getattr(
            session, "_cam1_video" if cam_idx == 1 else "_cam2_video", None
        )
        count = getattr(writer, "frame_count", None)
        if not isinstance(count, int):
            # RecordingSession changed shape under us. Loud, because silently
            # writing NaN here would cost every future take its video<->signal
            # cross-reference without anyone noticing.
            self.get_logger().error(
                "cannot read cam{} frame index from RecordingSession "
                "(frame_count missing); the synchronized table will have no "
                "video cross-reference for this take".format(cam_idx),
                throttle_duration_sec=10.0,
            )
            return None
        if count <= 0:
            return None  # no frame yet this take -> NaN, as the headless node does
        return count - 1

    def _on_sample(self) -> None:
        """Write ONE `synchronized` row -- but only while a take is recording.

        Mirrors gello_ur_recorder_node._on_sample: the same
        RecordingSession.write_sample() call with the same argument order, fed
        from the base node's latest-value fields (all of which live under
        _state_lock) plus the two cam frame indices.

        LOCKING follows the base's own pattern exactly and does NOT nest the two
        locks: snapshot under _state_lock, release, then take _session_lock and
        write. Every base callback does state-then-session in that order and
        never holds both, so keeping that shape is what makes this sampler
        deadlock-free by construction. Doing the write INSIDE _session_lock is
        what makes "never write to a closed or absent session" true rather than
        hopeful: stop_recording() clears self._session under this same lock and
        only calls close() after releasing it, so a row either lands in a live
        session or is not written at all.
        """
        try:
            with self._state_lock:
                gello_q = list(self._gello_q)
                gello_qd = list(self._gello_qd)
                gello_grip = self._gello_grip
                cmd = list(self._cmd)
                ur_q = list(self._ur_q)
                ur_qd = list(self._ur_qd)
                ur_eff = list(self._ur_eff)
                grip_cmd = self._grip_cmd
                grip_pos = self._grip_pos
                wrench = list(self._wrench)
                tcp = list(self._tcp)

            with self._session_lock:
                session = self._session
                if session is None:
                    return  # not recording: the sampler is idle between takes
                session.write_sample(
                    gello_q, gello_qd, gello_grip, cmd,
                    ur_q, ur_qd, ur_eff,
                    grip_cmd, grip_pos, wrench, tcp,
                    self._cam_frame_index(session, 1),
                    self._cam_frame_index(session, 2),
                )
        except Exception as exc:  # noqa: BLE001
            # Same reasoning as _home_tick: an exception escaping a timer
            # callback propagates out of rclpy.spin() and kills the ROS thread,
            # which would take the live preview, the GO-HOME path AND the other
            # eight tables down with it. A gap in the synchronized table is a
            # far smaller loss than a dead session, so log (throttled -- this
            # fires at sample_rate_hz) and keep going.
            self.get_logger().error(
                "synchronized sample failed: {}".format(exc),
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


def main(args=None):
    """Headless spin, mirroring gello_gui_node.main() -- mostly for debugging;
    the real entry point is task_recorder_gui.py, which owns the Qt loop."""
    rclpy.init(args=args)
    node = TaskRecorderGuiNode()
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
