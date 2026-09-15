#!/usr/bin/env python3
"""rclpy Node backing the interactive GELLO recorder GUI.

Unlike gello_ur_recorder_node.py (which opens its files at construction and records
from launch to Ctrl-C), this node's subscriptions are ALWAYS active -- for the live
preview / state panel -- but nothing is written to disk until start_recording() is
called, and nothing more is written after stop_recording() returns. Multiple
start/stop cycles ("takes") are supported in one node lifetime; the two RealSense
camera ROS2 processes and this node's subscriptions never restart between takes, so
the camera_warmup_s auto-exposure settling cost is only paid once per app launch
(gated on the Start button via cameras_ready()), not once per take.

Thread-safety: every public method here may be called from a different thread than
the one running rclpy.spin(node) (the GUI polls state/preview from Qt timers on the
Qt/main thread while ROS callbacks fire on a background spin thread). Two locks
guard the only two pieces of cross-thread mutable state: `_session_lock` around the
active RecordingSession reference (swapped by start/stop, read+written-through by
every subscription callback), and `_state_lock` around the live preview frames /
latest-value vector fields read by the GUI's polling getters.

Depth (2026-09-14): when BOTH ``cam1_depth_topic`` and ``cam2_depth_topic`` are
given, the node also subscribes to the realsense ``compressedDepth`` streams
(sensor_msgs/CompressedImage, ``16UC1; compressedDepth`` = PNG-in-a-header) plus
each camera's depth CameraInfo and depth->color Extrinsics, and hands the raw
bytes to ``RecordingSession.write_camN_depth_frame`` (-> ``depth.h5``). The
depth path never decodes anything and never touches the preview. Depth frames
are gated by exactly the same rule as color frames (a session is only ever open
once ``cameras_ready()`` is true, so every frame written is post-warm-up) --
there is deliberately no second warm-up clock for depth.

SPIN-THREAD BUDGET + HEADER STAMPS (2026-09-14, the timestamp-artifact fix).
Read :mod:`gello_recorder.spin_health` for the full diagnosis; the three things
this node does about it are:

1. **Stamps.** Every callback whose message has a ``header`` now passes
   ``stamp_s`` (float64 seconds, NaN when absent) into ``RecordingSession``,
   which appends it as a trailing column. ``/joint_states``,
   ``/gello/joint_states``, ``tcp_pose``, ``wrench`` and both colour
   ``CompressedImage`` streams are stamped; ``/forward_position_controller/
   commands`` (``Float64MultiArray``) and the three gripper ``Float32`` topics
   are NOT -- those message types have no header, so for them ``t_rel_s``
   (arrival) remains the only timestamp that exists.
2. **Shallow queues.** The high-rate robot subscriptions dropped from
   depth 100/50 to ``QOS_DEPTH_ROBOT_STATE`` (5). Worst-case staleness is
   ``depth / publish rate``, so 5/100 Hz = 50 ms instead of 1.00 s / 0.50 s.
   This recorder writes whatever arrives, so a deeper queue buys nothing: it
   converts "a row was skipped" into "a row is old", and a skipped row is
   strictly better than a silently mis-stamped one.
3. **No frame I/O on the spin thread.** ``_on_cam`` used to do a 1280x720
   ``cv2.imdecode`` for the preview (measured 9.3 ms) and then hand the same
   JPEG to the MP4 writer, which decoded it AGAIN and encoded it (9.3 + 6.9 ms),
   plus ~1.4 ms per depth frame into HDF5 -- about 1.6 s of work per wall-clock
   second with both cameras and depth on. Now the callback only copies bytes:
   recording goes through ``RecordingSession.submit_cam*`` (one background
   writer thread) and the preview through :class:`PreviewDecoder` (a separate
   latest-wins thread, so the preview can never build a backlog and never
   slows the spin thread down).

The node also publishes the alarm: ``ros_lag_s`` in :meth:`get_state_snapshot`
is ``now - latest /joint_states header stamp``, WARN-logged (throttled to once
per ``ROS_LAG_WARN_PERIOD_S``) above ``ROS_LAG_WARN_S``, and
:meth:`stop_recording` runs :func:`detect_spin_starvation` over the take's own
recorded table rates.
"""

import math
import os
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from std_msgs.msg import Float32, Float64MultiArray, String
from std_srvs.srv import Trigger

try:  # realsense2_camera_msgs ships with realsense-ros; absent => no extrinsics
    from realsense2_camera_msgs.msg import Extrinsics
except ImportError:  # pragma: no cover - depends on the installed overlay
    Extrinsics = None

from gello_recorder.recording_session import RecordingSession
from gello_recorder import take_delete
from gello_recorder.spin_health import (
    QOS_DEPTH_CAMERA,
    QOS_DEPTH_GELLO,
    QOS_DEPTH_GRIPPER,
    QOS_DEPTH_ROBOT_STATE,
    QOS_DEPTH_STATUS,
    ROS_LAG_WARN_PERIOD_S,
    ROS_LAG_WARN_S,
    PreviewDecoder,
    detect_spin_starvation,
    native_rate_table,
)

# QoS realsense-ros 4.58.2 publishes the ONE-SHOT extrinsics with (measured
# 2026-09-14 with `ros2 topic info -v`): RELIABLE + TRANSIENT_LOCAL. It is
# published once at stream start and never again, so a late-joining VOLATILE
# subscriber (rclpy's default depth-10 profile) silently receives nothing --
# the durability MUST match to get the latched sample. The compressedDepth
# image is also RELIABLE + TRANSIENT_LOCAL (identical to the color
# ``compressed`` topic, which the default profile already reads fine), and the
# depth camera_info is RELIABLE + VOLATILE and re-published every frame, so
# those two keep the same plain ``10`` the color subscription uses.
EXTRINSICS_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def depth_topics_for(cam_name: str, aligned: bool) -> Tuple[str, str, str]:
    """Return ``(image, camera_info, extrinsics)`` depth topics for one camera.

    The ONE definition of the realsense-ros 4.x depth topic layout shared by the
    GUI recorders, the headless node and the tests (``gello_recorder_gui``
    re-exports it). With ``aligned=False`` (default -- see ``ALIGN_DEPTH`` in
    ``gello_recorder_gui``) the native 848x480 depth stream is used; with
    ``aligned=True`` the node's ``align_depth.enable`` output (depth resampled
    onto the color image, 1280x720 at the default color profile) is used
    instead, and its camera_info is the color intrinsics realsense publishes
    under the aligned namespace. The extrinsics topic does not depend on
    alignment (it is the same depth->color transform either way).
    """
    if aligned:
        base = "/{0}/{0}/aligned_depth_to_color".format(cam_name)
    else:
        base = "/{0}/{0}/depth".format(cam_name)
    image = base + ("/image_raw/compressedDepth" if aligned
                    else "/image_rect_raw/compressedDepth")
    info = base + "/camera_info"
    extrinsics = "/{0}/{0}/extrinsics/depth_to_color".format(cam_name)
    return image, info, extrinsics


def stamp_to_seconds(stamp) -> float:
    """builtin_interfaces/Time -> float seconds (nan for a zero/absent stamp)."""
    try:
        sec, nsec = int(stamp.sec), int(stamp.nanosec)
    except AttributeError:
        return float("nan")
    if sec == 0 and nsec == 0:
        return float("nan")
    return sec + nsec * 1e-9


UR_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_N = len(UR_JOINT_ORDER)


def _reorder(msg: JointState):
    """Return (pos, vel, eff) lists in UR_JOINT_ORDER, or None if joints missing."""
    idx = {n: i for i, n in enumerate(msg.name)}
    if any(j not in idx for j in UR_JOINT_ORDER):
        return None
    pos = [float(msg.position[idx[j]]) if msg.position else None for j in UR_JOINT_ORDER]
    vel = [float(msg.velocity[idx[j]]) if msg.velocity else None for j in UR_JOINT_ORDER]
    eff = [float(msg.effort[idx[j]]) if msg.effort else None for j in UR_JOINT_ORDER]
    return pos, vel, eff


class GelloRecorderGuiNode(Node):
    def __init__(
        self,
        cam1_topic: str = "/cam1/cam1/color/image_raw/compressed",
        cam2_topic: str = "/cam2/cam2/color/image_raw/compressed",
        camera_fps: float = 30.0,
        camera_warmup_s: float = 3.0,
        output_root: str = "~/gello_recordings",
        node_name: str = "gello_recorder_gui_node",
        cam1_depth_topic: Optional[str] = None,
        cam2_depth_topic: Optional[str] = None,
        cam1_depth_info_topic: Optional[str] = None,
        cam2_depth_info_topic: Optional[str] = None,
        cam1_extrinsics_topic: Optional[str] = None,
        cam2_extrinsics_topic: Optional[str] = None,
        depth_aligned_to_color: bool = False,
    ) -> None:
        super().__init__(node_name)

        self.cam1_topic = cam1_topic
        self.cam2_topic = cam2_topic
        self.camera_fps = camera_fps
        self.camera_warmup_s = camera_warmup_s
        self.output_root = os.path.expanduser(output_root)

        # Depth recording is ON iff BOTH depth image topics are given; the
        # info/extrinsics topics are optional extras (metadata only).
        self.cam1_depth_topic = cam1_depth_topic
        self.cam2_depth_topic = cam2_depth_topic
        self.cam1_depth_info_topic = cam1_depth_info_topic
        self.cam2_depth_info_topic = cam2_depth_info_topic
        self.cam1_extrinsics_topic = cam1_extrinsics_topic
        self.cam2_extrinsics_topic = cam2_extrinsics_topic
        self.depth_aligned_to_color = bool(depth_aligned_to_color)
        self.record_depth = bool(cam1_depth_topic and cam2_depth_topic)

        # --- Recording-session state (guarded by _session_lock) ----------
        self._session_lock = threading.Lock()
        self._session: Optional[RecordingSession] = None
        self._take_index = 0
        # The take most recently STOPPED in this process, offered for
        # one-shot deletion via deletable_take_dir()/delete_last_take().
        # Written by stop_recording() (after close() has drained the
        # writers), cleared only by a successful delete_last_take().
        # start_recording() does NOT touch it: while the next take records,
        # deletable_take_dir() hides it anyway (it returns None whenever a
        # session is open), and stop_recording() then overwrites it with the
        # new take -- so an older take is never offered again once a newer
        # one has been stopped. The one case where leaving it alone is
        # observable: a start_recording() that raises AFTER taking the lock
        # (RecordingSession refused to open its files) keeps the previous
        # take deletable, which is the honest answer since no new take
        # exists.
        self._last_take_dir: Optional[str] = None
        # Per-session "already pushed into this session" flags for the cached
        # depth metadata, keyed by cam index. camera_info is re-published every
        # frame, so without these each frame would re-write the same intrinsics.
        self._depth_info_pushed = {1: False, 2: False}
        self._depth_extrinsics_pushed = {1: False, 2: False}

        # --- Live preview / state-panel data (guarded by _state_lock) ----
        self._state_lock = threading.Lock()
        self._cam1_frame: Optional[np.ndarray] = None
        self._cam2_frame: Optional[np.ndarray] = None
        self._cam1_last_frame_t: Optional[float] = None  # time.monotonic()
        self._cam2_last_frame_t: Optional[float] = None
        # Set ONCE, on each camera's first-ever frame, and NEVER reset across
        # takes -- this is what makes multi-take avoid re-paying camera warmup.
        self._cam1_first_frame_t: Optional[float] = None
        self._cam2_first_frame_t: Optional[float] = None
        # Depth liveness + cached metadata (under _state_lock). The cached
        # dicts are the kwargs for RecordingSession.set_depth_camera_info /
        # set_depth_extrinsics, kept so a session started AFTER the one-shot
        # extrinsics (or the first camera_info) arrived still gets them.
        self._cam1_depth_last_frame_t: Optional[float] = None
        self._cam2_depth_last_frame_t: Optional[float] = None
        self._depth_info_cache = {1: None, 2: None}
        self._depth_extrinsics_cache = {1: None, 2: None}

        self._gello_q = [None] * _N
        self._gello_qd = [None] * _N
        self._gello_q_prev = None
        self._gello_t_prev = None
        self._gello_grip = None
        self._cmd = [None] * _N
        self._ur_q = [None] * _N
        self._ur_qd = [None] * _N
        self._ur_eff = [None] * _N
        self._grip_cmd = None
        self._grip_pos = None
        self._wrench = [None] * 6
        self._tcp = [None] * 7

        # --- spin-thread health (under _state_lock) -----------------------
        # ros_lag_s = ROS-clock now - latest /joint_states header stamp. This is
        # the LIVE form of the 0.900 s artifact: had it been on screen on
        # 2026-09-14, the whole corpus would have been caught while recording.
        # Computed on the spin thread (where get_clock() is already in use) so
        # the Qt poller never touches rcl.
        self._ros_lag_s = None
        self._ros_lag_s_max = None
        self._ros_lag_t = None          # time.monotonic() of the last computation

        # --- live preview decoding (OFF the spin thread) -------------------
        # Latest-wins: a preview frame superseded before it was decoded is
        # dropped, so this thread can never build a backlog and can never push
        # back on the ROS callback. Recording does NOT go through here -- the
        # session's own writer thread owns the MP4 -- so a slow preview cannot
        # cost a recorded frame, and vice versa.
        self._preview = PreviewDecoder(
            self._set_preview_frame, name="recorder-preview-decoder")

        # --- Subscriptions (READ-ONLY, always active) ---------------------
        # QUEUE DEPTHS: the rule is  worst-case staleness = depth / publish rate
        # (gello_recorder.spin_health). The four robot topics publish at 100 Hz
        # or more and were the ones that went stale; everything else publishes
        # slower than the executor's round rate and never queues at all.
        self.create_subscription(
            JointState, "/gello/joint_states", self._on_gello, QOS_DEPTH_GELLO)
        self.create_subscription(
            Float32, "/gripper/gripper_client/target_gripper_width_percent",
            self._on_gello_grip, QOS_DEPTH_GRIPPER)
        self.create_subscription(
            Float64MultiArray, "/forward_position_controller/commands",
            self._on_cmd, QOS_DEPTH_ROBOT_STATE)
        self.create_subscription(
            JointState, "/joint_states", self._on_ur, QOS_DEPTH_ROBOT_STATE)
        self.create_subscription(
            Float32, "/robotiq_gripper/command_percent", self._on_grip_cmd,
            QOS_DEPTH_GRIPPER)
        self.create_subscription(
            Float32, "/robotiq_gripper/position_percent", self._on_grip_pos,
            QOS_DEPTH_GRIPPER)
        self.create_subscription(
            WrenchStamped, "/force_torque_sensor_broadcaster/wrench",
            self._on_wrench, QOS_DEPTH_ROBOT_STATE)
        self.create_subscription(
            PoseStamped, "/tcp_pose_broadcaster/pose", self._on_tcp,
            QOS_DEPTH_ROBOT_STATE)
        self.create_subscription(
            CompressedImage, self.cam1_topic, self._on_cam1, QOS_DEPTH_CAMERA)
        self.create_subscription(
            CompressedImage, self.cam2_topic, self._on_cam2, QOS_DEPTH_CAMERA)

        # --- Depth subscriptions (only when depth recording is on) -------
        if self.record_depth:
            self.create_subscription(
                CompressedImage, self.cam1_depth_topic, self._on_cam1_depth,
                QOS_DEPTH_CAMERA)
            self.create_subscription(
                CompressedImage, self.cam2_depth_topic, self._on_cam2_depth,
                QOS_DEPTH_CAMERA)
            if self.cam1_depth_info_topic:
                self.create_subscription(
                    CameraInfo, self.cam1_depth_info_topic,
                    self._on_cam1_depth_info, QOS_DEPTH_CAMERA)
            if self.cam2_depth_info_topic:
                self.create_subscription(
                    CameraInfo, self.cam2_depth_info_topic,
                    self._on_cam2_depth_info, QOS_DEPTH_CAMERA)
            if Extrinsics is None:
                if self.cam1_extrinsics_topic or self.cam2_extrinsics_topic:
                    self.get_logger().warn(
                        "realsense2_camera_msgs not importable -- depth->color "
                        "extrinsics will NOT be recorded (depth frames still are)")
            else:
                if self.cam1_extrinsics_topic:
                    self.create_subscription(
                        Extrinsics, self.cam1_extrinsics_topic,
                        self._on_cam1_extrinsics, EXTRINSICS_QOS)
                if self.cam2_extrinsics_topic:
                    self.create_subscription(
                        Extrinsics, self.cam2_extrinsics_topic,
                        self._on_cam2_extrinsics, EXTRINSICS_QOS)

        # ================================================================== #
        # TELEOP CONTROL PANEL (added block) -- the FIRST control-path code in
        # this otherwise 100% read-only package. Everything below is guarded by
        # the existing _state_lock and follows the same threading contract as
        # the rest of the node: callbacks fire on the spin thread, the GUI polls
        # copies from the Qt thread. NO blocking waits, ever (call_async +
        # add_done_callback only).
        #
        # Service contract (implemented by other workstreams; we code against
        # the names even when the servers are not running yet):
        #   /gello_ur_bridge/pause        (Trigger, unconditional)
        #   /gello_ur_bridge/resume_chase (Trigger, gated; may succeed=False)
        #   /gello_gripper_bridge/pause   (Trigger)
        #   /gello_gripper_bridge/resume  (Trigger, gated)
        #   /gello_ur_bridge/state        (String, 5 Hz: PAUSED|WAITING|STALE|
        #                                  CHASING|FOLLOWING)
        #   /gello_gripper_bridge/state   (String, 5 Hz: PAUSED|WAITING|
        #                                  RAMPING|FOLLOWING)
        # The gripper side may be entirely absent (sim) -> degrade gracefully.
        #
        # CRITICAL: this dict is named _svc_teleop, NOT _clients -- rclpy.Node
        # uses self._clients internally and shadowing it corrupts the executor
        # and destroy_node (documented in gello_operator_console_node.py).
        self._svc_teleop = {
            "arm_pause": self.create_client(Trigger, "/gello_ur_bridge/pause"),
            "arm_resume_chase": self.create_client(
                Trigger, "/gello_ur_bridge/resume_chase"),
            "grip_pause": self.create_client(
                Trigger, "/gello_gripper_bridge/pause"),
            "grip_resume": self.create_client(
                Trigger, "/gello_gripper_bridge/resume"),
        }
        # Latest state-topic strings + monotonic receipt times (under _state_lock).
        self._teleop_arm_state = None
        self._teleop_arm_state_t = None
        self._teleop_grip_state = None
        self._teleop_grip_state_t = None
        # In-flight request bookkeeping (under _state_lock).
        self._teleop_pending = False
        self._teleop_outstanding = 0
        self._teleop_results = {}
        self._teleop_last_ok = None
        self._teleop_last_msg = ""

        self.create_subscription(
            String, "/gello_ur_bridge/state", self._on_arm_state, QOS_DEPTH_STATUS)
        self.create_subscription(
            String, "/gello_gripper_bridge/state", self._on_grip_state,
            QOS_DEPTH_STATUS)
        # ============ end TELEOP CONTROL PANEL (added block) =============== #

        if self.record_depth:
            depth_desc = (
                f"depth ON (aligned={str(self.depth_aligned_to_color).lower()}) "
                f"cam1_depth={self.cam1_depth_topic} cam2_depth={self.cam2_depth_topic} "
                f"info=({self.cam1_depth_info_topic}, {self.cam2_depth_info_topic}) "
                f"extrinsics=({self.cam1_extrinsics_topic}, {self.cam2_extrinsics_topic})"
            )
        else:
            depth_desc = "depth OFF"
        self.get_logger().info(
            f"gello_recorder_gui_node up. cam1={self.cam1_topic} cam2={self.cam2_topic} "
            f"warmup={self.camera_warmup_s:.1f}s output_root={self.output_root} "
            f"{depth_desc}"
        )

    # ---- public API used by the GUI (thread-safe) ------------------------
    def cameras_ready(self) -> bool:
        return self.warmup_seconds_remaining() <= 0.0

    def warmup_seconds_remaining(self) -> float:
        now = time.monotonic()
        with self._state_lock:
            t1, t2 = self._cam1_first_frame_t, self._cam2_first_frame_t
        if t1 is None or t2 is None:
            return float("inf")
        remaining = self.camera_warmup_s - min(now - t1, now - t2)
        return max(0.0, remaining)

    def is_recording(self) -> bool:
        with self._session_lock:
            return self._session is not None

    def take_index(self) -> int:
        with self._session_lock:
            return self._take_index

    def start_recording(self) -> str:
        if not self.cameras_ready():
            raise RuntimeError("cameras are not warmed up yet -- refusing to start recording")
        with self._session_lock:
            if self._session is not None:
                raise RuntimeError("start_recording() called while already recording")
            self._take_index += 1
            take_idx = self._take_index
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            take_dir = os.path.join(self.output_root, f"take_{take_idx:02d}_{stamp}")
            session = RecordingSession(
                take_dir, camera_fps=self.camera_fps, record_depth=self.record_depth)
            if self.record_depth:
                # Metadata first, while no callback can see the session yet:
                # source topics always, intrinsics/extrinsics for whatever has
                # already arrived (the rest is pushed by the callbacks below
                # the moment it lands -- camera_info commonly does after Start).
                self._depth_info_pushed = {1: False, 2: False}
                self._depth_extrinsics_pushed = {1: False, 2: False}
                session.set_depth_source(1, self.cam1_depth_topic, self.depth_aligned_to_color)
                session.set_depth_source(2, self.cam2_depth_topic, self.depth_aligned_to_color)
                with self._state_lock:
                    info_cache = dict(self._depth_info_cache)
                    ext_cache = dict(self._depth_extrinsics_cache)
                for cam_idx in (1, 2):
                    if info_cache[cam_idx] is not None:
                        session.set_depth_camera_info(cam_idx, **info_cache[cam_idx])
                        self._depth_info_pushed[cam_idx] = True
                    if ext_cache[cam_idx] is not None:
                        session.set_depth_extrinsics(cam_idx, **ext_cache[cam_idx])
                        self._depth_extrinsics_pushed[cam_idx] = True
            self._session = session
        # Per-TAKE maximum, so a stale figure from an earlier take cannot be
        # reported against this one.
        with self._state_lock:
            self._ros_lag_s_max = None
        self.get_logger().info(
            f"Recording started -> {take_dir} "
            f"(depth {'ON' if self.record_depth else 'OFF'})")
        return take_dir

    def stop_recording(self) -> dict:
        with self._session_lock:
            session = self._session
            if session is None:
                raise RuntimeError("stop_recording() called while not recording")
            self._session = None
        # close() (file I/O) happens OUTSIDE the lock -- callbacks already see
        # self._session is None by this point and will no-op, so nothing else
        # touches `session` concurrently from here on. close() also DRAINS the
        # background frame writer before finalising, so the counts below include
        # every frame that was accepted.
        stats = session.close()
        with self._session_lock:
            self._last_take_dir = session.session_dir
        stats["session_dir"] = session.session_dir
        stats.update(self._health_summary(session, stats))
        self.get_logger().info(f"Recording stopped -> {stats}")
        if stats.get("spin_starvation_suspected"):
            self.get_logger().warn(stats["spin_starvation_message"])
        dropped = (stats.get("dropped_frames") or {}).get("total", 0)
        if dropped:
            self.get_logger().warn(
                "{} camera/depth frame(s) were DROPPED (writer queue full) -- "
                "this take is short by that many frames: {}".format(
                    dropped, stats["dropped_frames"]))
        return stats

    def _health_summary(self, session, stats: dict) -> dict:
        """Per-take spin-health verdict, merged into the stop summary.

        Three independent numbers, none of which existed before 2026-09-14:
        what the writer queue had to throw away, how far behind the robot
        state stamps ran, and whether the native tables collapsed onto one
        shared rate (the fingerprint of a starved executor)."""
        rates = native_rate_table(
            stats.get("message_counts") or {}, stats.get("duration_s") or 0.0)
        report = detect_spin_starvation(rates)
        with self._state_lock:
            lag_max = self._ros_lag_s_max
        return {
            # PROVENANCE: a take must say for itself whether depth was on and
            # whether the spin thread was starved while it was written -- those
            # two facts together are what the 2026-09-14 corpus could not
            # answer after the fact.
            "record_depth": bool(self.record_depth),
            "dropped_frames": session.dropped_frames(),
            "native_rates_hz": report["rates_hz"],
            "spin_starvation_suspected": bool(report["suspected"]),
            "spin_starvation_reason": report["reason"],
            "spin_starvation_message": report["message"] or "",
            "ros_lag_s_max": None if lag_max is None else round(float(lag_max), 4),
        }

    def deletable_take_dir(self) -> Optional[str]:
        """The take directory "delete the take I just recorded" may offer.

        Only the take most recently STOPPED in this process, and only while
        nothing is currently recording -- deleting is never offered as an
        option mid-take, so the GUI never has to reason about a delete
        racing a write.
        """
        with self._session_lock:
            return self._last_take_dir if self._session is None else None

    def delete_last_take(self) -> dict:
        """Permanently delete the take from ``deletable_take_dir()``.

        Raises ``RuntimeError`` (never ``take_delete.TakeDeleteError`` and
        never a raw ``OSError``, so the GUI has exactly one exception type
        to catch here) if there is nothing eligible to delete, and lets the
        underlying ``take_delete.TakeDeleteError`` message pass through as a
        ``RuntimeError`` if the stored path somehow fails validation. An
        ``OSError`` out of the actual ``rmtree`` (permissions, a read-only or
        vanished mount) is wrapped the same way: ``_last_take_dir`` is left
        SET in that case, so the take stays offered and the operator can
        retry once the cause is fixed -- a half-deleted folder must not fall
        off the button silently.

        On success, ``_last_take_dir`` is cleared (one-shot: a second call
        with nothing new recorded raises "no take to delete" rather than
        deleting again) and, if the deleted take was still the most recent
        index ever allocated, ``_take_index`` is decremented so the next
        recording REUSES that number -- see
        ``take_delete.next_take_index_after_delete`` for why that is safe
        (the folder name always carries a fresh timestamp, so reuse cannot
        collide with anything).

        Locking: the take is CLAIMED under ``_session_lock`` (eligibility
        check + ``_last_take_dir`` cleared) but the ``rmtree`` itself runs
        OUTSIDE the lock. ``_session_lock`` is also taken by every 100 Hz
        sample callback and every camera frame on the spin thread, so
        holding it across file-system work would stall the whole spin
        thread for the rmtree duration (milliseconds on this SSD, seconds
        on a slow mount) -- and TaskRecorderGuiNode keeps this button
        enabled during a GO HOME, whose sensor caches must stay fresh.
        Claiming first keeps the operation one-shot without the lock: a
        second call sees ``_last_take_dir is None``. If a new take starts
        while the rmtree runs, the counter bookkeeping below still holds
        because ``next_take_index_after_delete`` only decrements when the
        deleted take is STILL the latest index.
        """
        with self._session_lock:
            if self._session is not None:
                raise RuntimeError("cannot delete while recording")
            if self._last_take_dir is None:
                raise RuntimeError("no take to delete")
            take_dir = self._last_take_dir
            self._last_take_dir = None  # claimed; restored below on failure
        try:
            result = take_delete.delete_take_dir(take_dir, self.output_root)
        except (take_delete.TakeDeleteError, OSError) as exc:
            # Validation refused it, or rmtree/scandir failed part-way
            # (EACCES, EROFS, ENOENT under us, ...). Reported, not swallowed
            # -- and the take is put back on offer so the operator can retry
            # once the cause is fixed, unless a newer take has been stopped
            # in the meantime (then the newer one rightly owns the button).
            with self._session_lock:
                if self._last_take_dir is None:
                    self._last_take_dir = take_dir
            if isinstance(exc, take_delete.TakeDeleteError):
                raise RuntimeError(str(exc)) from exc
            raise RuntimeError(
                "could not delete {}: {}".format(take_dir, exc)) from exc
        with self._session_lock:
            self._take_index = take_delete.next_take_index_after_delete(
                self._take_index, take_dir)
            result["take_index_after"] = self._take_index
        self.get_logger().info(
            "Deleted take -> {} ({} files, {} bytes); take counter now {} "
            "(next recording will be take_{:02d}_...)".format(
                result["path"], result["n_files"], result["bytes"],
                result["take_index_after"], result["take_index_after"] + 1)
        )
        return result

    def get_preview_frames(self):
        with self._state_lock:
            return self._cam1_frame, self._cam2_frame

    def get_state_snapshot(self) -> dict:
        now = time.monotonic()
        with self._state_lock:
            cam1_age = None if self._cam1_last_frame_t is None else now - self._cam1_last_frame_t
            cam2_age = None if self._cam2_last_frame_t is None else now - self._cam2_last_frame_t
            # Depth ages are None when depth is off OR no depth frame has
            # arrived yet -- the GUI distinguishes the two via depth_enabled.
            d1 = self._cam1_depth_last_frame_t
            d2 = self._cam2_depth_last_frame_t
            cam1_depth_age = None if d1 is None else now - d1
            cam2_depth_age = None if d2 is None else now - d2
            return {
                "gello_q": list(self._gello_q),
                "gello_qd": list(self._gello_qd),
                "gello_grip": self._gello_grip,
                "cmd": list(self._cmd),
                "ur_q": list(self._ur_q),
                "ur_qd": list(self._ur_qd),
                "ur_eff": list(self._ur_eff),
                "grip_cmd": self._grip_cmd,
                "grip_pos": self._grip_pos,
                "wrench": list(self._wrench),
                "tcp": list(self._tcp),
                "cam1_last_frame_age_s": cam1_age,
                "cam2_last_frame_age_s": cam2_age,
                "depth_enabled": self.record_depth,
                "cam1_depth_last_frame_age_s": cam1_depth_age,
                "cam2_depth_last_frame_age_s": cam2_depth_age,
                # Spin-thread health. ros_lag_s is (ROS now - /joint_states
                # header stamp) as of the last /joint_states callback;
                # ros_lag_age_s says how long ago that was, so "no lag shown"
                # and "no joint states at all" stay distinguishable.
                "ros_lag_s": self._ros_lag_s,
                "ros_lag_s_max": self._ros_lag_s_max,
                "ros_lag_age_s": (None if self._ros_lag_t is None
                                  else now - self._ros_lag_t),
                "ros_lag_warn_s": ROS_LAG_WARN_S,
            }

    # ================================================================== #
    # TELEOP CONTROL PANEL -- public API (thread-safe) + callbacks (added).
    # ================================================================== #
    def _on_arm_state(self, msg: String):
        with self._state_lock:
            self._teleop_arm_state = msg.data
            self._teleop_arm_state_t = time.monotonic()

    def _on_grip_state(self, msg: String):
        with self._state_lock:
            self._teleop_grip_state = msg.data
            self._teleop_grip_state_t = time.monotonic()

    def request_teleop_pause(self) -> bool:
        """Fire the (unconditional) pause services. Non-blocking.

        Pauses the arm bridge (always succeeds) and, if its service is up, the
        gripper bridge. Returns False without firing if a request is already
        in flight or the arm service is unavailable.
        """
        return self._fire_teleop(pause=True)

    def request_teleop_resume(self) -> bool:
        """Fire the (gated) resume services. Non-blocking.

        Requests resume_chase on the arm and resume on the gripper. Either may
        legitimately answer success=False (refused, e.g. leader not still) --
        that is NOT an error; the explanatory message is surfaced via
        get_teleop_status()['last_msg'].
        """
        return self._fire_teleop(pause=False)

    def _fire_teleop(self, pause: bool) -> bool:
        arm_name = "arm_pause" if pause else "arm_resume_chase"
        grip_name = "grip_pause" if pause else "grip_resume"
        arm_cli = self._svc_teleop[arm_name]
        grip_cli = self._svc_teleop[grip_name]
        # service_is_ready() touches rcl graph state -- call it OUTSIDE the lock.
        arm_ready = arm_cli.service_is_ready()
        grip_ready = grip_cli.service_is_ready()
        with self._state_lock:
            if self._teleop_pending:
                return False  # overlapping-request guard
            if not arm_ready:
                self._teleop_last_ok = False
                self._teleop_last_msg = (
                    "arm bridge service unavailable ({})".format(arm_name))
                return False
            self._teleop_pending = True
            self._teleop_outstanding = 1 + (1 if grip_ready else 0)
            self._teleop_results = {}
        # call_async + add_done_callback only -- NEVER spin_until_future_complete.
        arm_future = arm_cli.call_async(Trigger.Request())
        arm_future.add_done_callback(lambda f: self._teleop_done("arm", f))
        if grip_ready:
            grip_future = grip_cli.call_async(Trigger.Request())
            grip_future.add_done_callback(lambda f: self._teleop_done("grip", f))
        return True

    def _teleop_done(self, which: str, future):
        """Done-callback (spin thread): store result; aggregate when all in."""
        ok = False
        msg = ""
        try:
            resp = future.result()
            if resp is not None:
                ok = bool(resp.success)
                msg = str(resp.message)
            else:
                msg = "{}: no response".format(which)
        except Exception as exc:  # noqa: BLE001 -- surface, never crash spin
            msg = "{}: {}".format(which, exc)
        with self._state_lock:
            self._teleop_results[which] = (ok, msg)
            self._teleop_outstanding -= 1
            if self._teleop_outstanding <= 0:
                results = self._teleop_results
                self._teleop_last_ok = all(v[0] for v in results.values())
                self._teleop_last_msg = " | ".join(
                    "{}: {}".format(k, v[1]) for k, v in sorted(results.items()))
                self._teleop_pending = False

    def get_teleop_status(self) -> dict:
        """Copy of the teleop panel state for the Qt polling timer."""
        now = time.monotonic()
        # service readiness touches rcl -- read it before taking the lock.
        grip_svc_ready = (
            self._svc_teleop["grip_pause"].service_is_ready()
            or self._svc_teleop["grip_resume"].service_is_ready())
        with self._state_lock:
            arm_age = (None if self._teleop_arm_state_t is None
                       else now - self._teleop_arm_state_t)
            grip_available = (self._teleop_grip_state_t is not None
                              or grip_svc_ready)
            return {
                "arm_state": self._teleop_arm_state,
                "arm_state_age_s": arm_age,
                "grip_state": self._teleop_grip_state,
                "grip_available": grip_available,
                "pending": self._teleop_pending,
                "last_ok": self._teleop_last_ok,
                "last_msg": self._teleop_last_msg,
            }

    # ---- subscription callbacks -------------------------------------------
    def _on_gello(self, msg: JointState):
        r = _reorder(msg)
        if r is None:
            return
        pos, _, _ = r
        stamp_s = stamp_to_seconds(msg.header.stamp)
        t = time.monotonic()
        with self._state_lock:
            if self._gello_q_prev is not None and self._gello_t_prev is not None:
                dt = t - self._gello_t_prev
                if dt > 1e-6:
                    self._gello_qd = [
                        (pos[i] - self._gello_q_prev[i]) / dt for i in range(_N)
                    ]
            self._gello_q_prev, self._gello_t_prev = pos, t
            self._gello_q = pos
            qd_snapshot = list(self._gello_qd)
        with self._session_lock:
            if self._session is not None:
                self._session.write_gello(pos, qd_snapshot, stamp_s=stamp_s)

    def _on_gello_grip(self, msg: Float32):
        with self._state_lock:
            self._gello_grip = float(msg.data)
            gello_grip, grip_cmd, grip_pos = self._gello_grip, self._grip_cmd, self._grip_pos
        with self._session_lock:
            if self._session is not None:
                self._session.bump("gello_grip")
                self._session.write_gello_grip(gello_grip, grip_cmd, grip_pos)

    def _on_cmd(self, msg: Float64MultiArray):
        d = list(msg.data)
        if len(d) < _N:
            return
        cmd = [float(x) for x in d[:_N]]
        with self._state_lock:
            self._cmd = cmd
        with self._session_lock:
            if self._session is not None:
                self._session.write_cmd(cmd)

    def _on_ur(self, msg: JointState):
        r = _reorder(msg)
        if r is None:
            return
        pos, vel, eff = r
        stamp_s = stamp_to_seconds(msg.header.stamp)
        self._note_ros_lag(stamp_s)
        with self._state_lock:
            self._ur_q, self._ur_qd, self._ur_eff = pos, vel, eff
        with self._session_lock:
            if self._session is not None:
                self._session.write_ur(pos, vel, eff, stamp_s=stamp_s)

    def _note_ros_lag(self, stamp_s: float) -> None:
        """Update ros_lag_s from one /joint_states header stamp (spin thread).

        Both sides come from the ROS clock: the stamp is the driver's, ``now``
        is this node's ``get_clock()``. Mixing in ``time.time()`` would work
        today (use_sim_time is false) and break silently under a sim clock, so
        it is not done. A message with no stamp (sec=nsec=0 -> NaN) teaches us
        nothing and is ignored rather than reported as a huge lag."""
        if stamp_s is None or not math.isfinite(stamp_s):
            return
        try:
            now = self.get_clock().now().nanoseconds * 1e-9
        except Exception:  # noqa: BLE001 - never let a clock read kill the callback
            return
        lag = now - stamp_s
        with self._state_lock:
            self._ros_lag_s = lag
            self._ros_lag_t = time.monotonic()
            if self._ros_lag_s_max is None or lag > self._ros_lag_s_max:
                self._ros_lag_s_max = lag
        if lag > ROS_LAG_WARN_S:
            self.get_logger().warn(
                "ros_lag_s={:.3f}s: /joint_states rows are being stamped "
                "{:.0f} ms after the driver captured them -- the spin thread is "
                "falling behind and recorded robot rows will be stale".format(
                    lag, lag * 1e3),
                throttle_duration_sec=ROS_LAG_WARN_PERIOD_S)

    def _on_grip_cmd(self, msg: Float32):
        with self._state_lock:
            self._grip_cmd = float(msg.data)
            gello_grip, grip_cmd, grip_pos = self._gello_grip, self._grip_cmd, self._grip_pos
        with self._session_lock:
            if self._session is not None:
                self._session.bump("grip_cmd")
                self._session.write_gello_grip(gello_grip, grip_cmd, grip_pos)

    def _on_grip_pos(self, msg: Float32):
        with self._state_lock:
            self._grip_pos = float(msg.data)
            gello_grip, grip_cmd, grip_pos = self._gello_grip, self._grip_cmd, self._grip_pos
        with self._session_lock:
            if self._session is not None:
                self._session.bump("grip_pos")
                self._session.write_gello_grip(gello_grip, grip_cmd, grip_pos)

    def _on_wrench(self, msg: WrenchStamped):
        w = msg.wrench
        wrench6 = [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
        stamp_s = stamp_to_seconds(msg.header.stamp)
        with self._state_lock:
            self._wrench = wrench6
        with self._session_lock:
            if self._session is not None:
                self._session.write_wrench(wrench6, stamp_s=stamp_s)

    def _on_tcp(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        tcp7 = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        stamp_s = stamp_to_seconds(msg.header.stamp)
        with self._state_lock:
            self._tcp = tcp7
        with self._session_lock:
            if self._session is not None:
                self._session.write_tcp(tcp7, stamp_s=stamp_s)

    def _on_cam1(self, msg: CompressedImage):
        self._on_cam(msg, cam_idx=1)

    def _on_cam2(self, msg: CompressedImage):
        self._on_cam(msg, cam_idx=2)

    def _set_preview_frame(self, cam_idx: int, frame) -> None:
        """Sink for :class:`PreviewDecoder` -- runs on the decoder thread.

        The ONLY place ``_cam*_frame`` is written. A payload that failed to
        decode never reaches here, so the pane keeps showing the last good
        frame, exactly as it did when the decode was inline."""
        if frame is None:
            return
        with self._state_lock:
            if cam_idx == 1:
                self._cam1_frame = frame
            else:
                self._cam2_frame = frame

    def _on_cam(self, msg: CompressedImage, cam_idx: int):
        """Colour frame: copy the bytes, note liveness, hand both jobs away.

        NOTHING is decoded here. This callback used to run a 1280x720
        ``cv2.imdecode`` for the preview (9.3 ms measured) and then pass the
        same JPEG to the MP4 writer, which decoded it a SECOND time and encoded
        it (another 16.2 ms) -- 2 cameras x 30 Hz x 25.5 ms = 1.5 s of work per
        wall-clock second on the one thread that also services every robot
        topic. That is the whole timestamp artifact. Now the preview goes to a
        latest-wins decoder thread and the recording to the session's writer
        thread; the spin thread does a memcpy and a couple of lock acquisitions.
        """
        raw = bytes(msg.data)
        stamp_s = stamp_to_seconds(msg.header.stamp)
        now = time.monotonic()
        with self._state_lock:
            if cam_idx == 1:
                self._cam1_last_frame_t = now
                if self._cam1_first_frame_t is None:
                    self._cam1_first_frame_t = now
            else:
                self._cam2_last_frame_t = now
                if self._cam2_first_frame_t is None:
                    self._cam2_first_frame_t = now
        self._preview.submit(cam_idx, raw)
        # Recording only ever starts once cameras_ready() (warmup already
        # elapsed), so every frame written into an active session is good --
        # no per-frame warmup skip needed here (unlike the headless node).
        with self._session_lock:
            if self._session is not None:
                self._session.submit_cam_frame(cam_idx, raw, stamp_s=stamp_s)

    # ---- depth callbacks (only subscribed when record_depth) ---------------
    def _on_cam1_depth(self, msg: CompressedImage):
        self._on_cam_depth(msg, cam_idx=1)

    def _on_cam2_depth(self, msg: CompressedImage):
        self._on_cam_depth(msg, cam_idx=2)

    def _on_cam_depth(self, msg: CompressedImage, cam_idx: int):
        # Raw bytes straight through -- the PNG-in-a-header payload is stored
        # as-is by the depth writer; nothing is decoded on this thread.
        raw = bytes(msg.data)
        stamp_s = stamp_to_seconds(msg.header.stamp)
        now = time.monotonic()
        with self._state_lock:
            if cam_idx == 1:
                self._cam1_depth_last_frame_t = now
            else:
                self._cam2_depth_last_frame_t = now
        # Same gate as _on_cam: a session only exists once cameras_ready() was
        # true, so depth frames are dropped exactly when color frames are
        # (i.e. never, once recording) -- no separate depth warm-up clock.
        with self._session_lock:
            if self._session is not None:
                # Queued, not written: an 848x480 depth PNG is ~740 kB and its
                # HDF5 append measured 1.4 ms -- 84 ms/s for two cameras, on the
                # thread that must service 100 Hz robot topics.
                self._session.submit_cam_depth_frame(cam_idx, raw, stamp_s=stamp_s)

    def _on_cam1_depth_info(self, msg: CameraInfo):
        self._on_depth_info(msg, cam_idx=1)

    def _on_cam2_depth_info(self, msg: CameraInfo):
        self._on_depth_info(msg, cam_idx=2)

    def _on_depth_info(self, msg: CameraInfo, cam_idx: int):
        info = {
            "width": int(msg.width),
            "height": int(msg.height),
            "distortion_model": str(msg.distortion_model),
            "D": [float(x) for x in msg.d],
            "K": [float(x) for x in msg.k],
            "R": [float(x) for x in msg.r],
            "P": [float(x) for x in msg.p],
            "frame_id": str(msg.header.frame_id),
        }
        with self._state_lock:
            self._depth_info_cache[cam_idx] = info
        with self._session_lock:
            if self._session is not None and not self._depth_info_pushed[cam_idx]:
                self._session.set_depth_camera_info(cam_idx, **info)
                self._depth_info_pushed[cam_idx] = True

    def _on_cam1_extrinsics(self, msg):
        self._on_extrinsics(msg, cam_idx=1)

    def _on_cam2_extrinsics(self, msg):
        self._on_extrinsics(msg, cam_idx=2)

    def _on_extrinsics(self, msg, cam_idx: int):
        ext = {
            "rotation": [float(x) for x in msg.rotation],
            "translation": [float(x) for x in msg.translation],
        }
        with self._state_lock:
            self._depth_extrinsics_cache[cam_idx] = ext
        with self._session_lock:
            if self._session is not None and not self._depth_extrinsics_pushed[cam_idx]:
                self._session.set_depth_extrinsics(cam_idx, **ext)
                self._depth_extrinsics_pushed[cam_idx] = True

    def destroy_node(self):
        with self._session_lock:
            session = self._session
            self._session = None
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - best-effort on shutdown
                pass
        preview = getattr(self, "_preview", None)
        if preview is not None:
            try:
                preview.stop()
            except Exception:  # noqa: BLE001 - best-effort on shutdown
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GelloRecorderGuiNode()
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
