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
tested without a robot. The ROS I/O adapter for it -- the duck-typed ``ops``
object, the freshness-gated sensor caches, the controller_manager / action /
gripper clients and the lock + 10 Hz timer that drive the state machine --
lives in ``gello_recorder/home_move_ros.py`` as :class:`HomeMoveIoMixin`, so
the EEF teleop GUI can reuse the identical, hardware-verified path for its GO
TO START POSE button. THIS node only mixes that in and points its two bridge
pause entries at the clients the base recorder node already owns. The sequence
it drives is:

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
  :meth:`request_stop_home` and :meth:`get_home_status` (all three inherited
  from :class:`HomeMoveIoMixin`).
* NOTHING here blocks. Every service call is ``call_async`` +
  ``add_done_callback``; ``spin_until_future_complete()`` from inside a
  callback would deadlock the single-threaded executor, so it is never used.
* The GO HOME locking rules -- why ``_home_lock`` is real and not ceremonial,
  why the done-callbacks must NOT take it, and why two Trigger replies can land
  on two different threads -- are in home_move_ros.py's module docstring
  ("THREADING CONTRACT"). They apply here unchanged.
"""

import json
import math
import os
import threading
import time
from datetime import datetime
from typing import Optional

import rclpy
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from std_msgs.msg import String

from gello_recorder.gello_gui_node import GelloRecorderGuiNode
from gello_recorder.spin_health import QOS_DEPTH_STATUS
from gello_recorder.home_move_ros import HomeMoveIoMixin, _service_ready

# Re-exported for callers that historically imported the GO-HOME ROS surface
# from this module (it lived here until the split into home_move_ros.py).
# New code should import from gello_recorder.home_move_ros directly.
from gello_recorder.home_move_ros import (  # noqa: F401 (re-exports)
    GRIPPER_COMMAND_TOPIC,
    GRIPPER_POSITION_TOPIC,
    HOME_TRAJECTORY_ACTION,
    JOINT_STATES_TOPIC,
    LIST_CONTROLLERS_SERVICE,
    SPEED_SCALING_TOPIC,
    SWITCH_CONTROLLER_SERVICE,
    HomeMoveOps,
)

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


class TaskRecorderGuiNode(HomeMoveIoMixin, GelloRecorderGuiNode):
    """GelloRecorderGuiNode + the GO-HOME/STOP-HOME control path + the
    fixed-rate ``synchronized`` sampler (see module docstring for both).

    The GO HOME control path is :class:`HomeMoveIoMixin` (home_move_ros.py),
    mixed in AHEAD of the recorder node so its ``latest_*`` readers and the
    three Qt-facing methods are this class's own; it has no ``__init__``, so
    ``super().__init__`` below still resolves straight to GelloRecorderGuiNode.

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
        **recorder_kwargs,
    ) -> None:
        # The parent wires up every recording subscription, both locks and the
        # take state machine; nothing below touches any of that.
        # ``**recorder_kwargs`` passes the parent's depth-recording kwargs
        # (cam*_depth_topic / cam*_depth_info_topic / cam*_extrinsics_topic /
        # depth_aligned_to_color -- see GelloRecorderGuiNode) straight through,
        # so task_recorder_gui.main() wires depth exactly like the base GUI.
        super().__init__(
            cam1_topic=cam1_topic,
            cam2_topic=cam2_topic,
            camera_fps=camera_fps,
            camera_warmup_s=camera_warmup_s,
            output_root=output_root,
            node_name=node_name,
            **recorder_kwargs,
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

        # --- GO HOME: caches, clients, sequencer, 10 Hz tick -------------- #
        # All of it comes from HomeMoveIoMixin (home_move_ros.py), with every
        # graph name at its production default, towards HOME_JOINTS.
        #
        # The two pause entries are ALIASES of the base node's clients rather
        # than new ones -- identical services with identical semantics, so a
        # second client would only add graph entities. The alias is taken once,
        # here, so the coupling to the base's key names lives in one place.
        #
        # The base node already SUBSCRIBES to the gripper command topic for
        # recording, so the open command the sequencer publishes is captured in
        # the take like any other.
        self.install_home_move_io(
            arm_pause_client=self._svc_teleop["arm_pause"],
            gripper_pause_client=self._svc_teleop["grip_pause"],
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
            String, GRIPPER_DISCRETE_STATE_TOPIC, self._on_grip_discrete_state,
            QOS_DEPTH_STATUS
        )
        self._grip_param_client = self.create_client(
            GetParameters, GRIPPER_BRIDGE_PARAMETERS_SERVICE
        )
        self._grip_param_attempts = 0
        self._grip_param_timer = self.create_timer(
            _GRIPPER_PARAM_PROBE_PERIOD_S, self._probe_gripper_parameters
        )

        # --- the synchronized-table sampler ------------------------------- #
        # Created LAST: its callback reads state the lines above set up, and an
        # rclpy timer can fire as soon as it exists once something spins us.
        # Period expression is verbatim from gello_ur_recorder_node so both
        # recorders clamp a sub-1 Hz request identically.
        self._sample_timer = self.create_timer(
            1.0 / max(1.0, self.sample_rate_hz), self._on_sample
        )

        names = self._home_graph_names
        self.get_logger().info(
            "task_recorder_gui_node up. GO HOME path: {} + {} , {} , {} -> {}. "
            "synchronized sampler @ {:.0f} Hz; speed scaling from {}".format(
                names["switch_controller"],
                names["list_controllers"],
                names["trajectory_action"],
                names["gripper_command_topic"],
                "teleop stays PAUSED (re-engage from the EEF GUI)",
                self.sample_rate_hz,
                names["speed_scaling_topic"],
            )
        )

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

    # ---- the synchronized-table sampler (spin thread) -------------------- #
    def _cam_frame_index(self, session, cam_idx: int) -> Optional[int]:
        """Index of the most recent frame written to this take's camN.mp4.

        HOW, and why it is this way. The base node's _on_cam does not write the
        frame at all any more -- since 2026-09-14 it SUBMITS it to
        RecordingSession's background writer thread, which returns long before
        the MP4 frame exists, so there is no index to intercept even in
        principle. The index is therefore read back out of the session through
        its public ``latest_frame_index(cam_idx)``, which reports what the MP4
        writer has actually committed. Nothing about which frames are written,
        or when, is touched; this is a pure read.

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
        reader = getattr(session, "latest_frame_index", None)
        if not callable(reader):
            # RecordingSession changed shape under us. Loud, because silently
            # writing NaN here would cost every future take its video<->signal
            # cross-reference without anyone noticing.
            self.get_logger().error(
                "cannot read cam{} frame index from RecordingSession "
                "(latest_frame_index missing); the synchronized table will have "
                "no video cross-reference for this take".format(cam_idx),
                throttle_duration_sec=10.0,
            )
            return None
        idx = reader(cam_idx)
        # None before the first frame of the take -> NaN, as the headless node does.
        return idx if isinstance(idx, int) else None

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
