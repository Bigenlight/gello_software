#!/usr/bin/env python3
"""Diagnostic recorder for the GELLO -> UR7e teleop pipeline.

Runs ALONGSIDE the teleop (start it in a second terminal, or via run_recorder.sh)
and logs, into a timestamped session sub-folder, as much of the pipeline as it can
observe so you can analyse tracking / lag / vibration offline:

  * GELLO leader arm joints (position) + a finite-difference VELOCITY (rad/s)
  * GELLO gripper width command (0=open..1=closed)
  * Bridge output command to the UR (/forward_position_controller/commands)
  * UR7e ACTUAL joint state: position, velocity AND effort (from /joint_states)
  * Robotiq 2F-85 gripper command + actual position percent
  * TCP force/torque (force_torque_sensor_broadcaster/wrench)
  * TCP Cartesian pose (tcp_pose_broadcaster/pose), if that broadcaster is loaded

Outputs (in <output_root>/session_<YYYYmmdd_HHMMSS>/ unless session_dir is given):
  * vectors.h5         -- ONE HDF5 file holding ALL the vector-signal tables that
                         used to be separate CSVs, each as its own dataset/table:
                           - synchronized      : wide table sampled at sample_rate_hz
                                                  (default 100 Hz), latest value of every
                                                  signal time-aligned (open in pandas via
                                                  pd.read_hdf) -- the main analysis table.
                           - gello_joint_states, ur_joint_states, command,
                             gripper, wrench, tcp_pose : native-rate, full-detail tables.
                           - cam1_frames, cam2_frames : per-frame index tables mapping
                                                  each recorded video frame to its capture
                                                  timestamp (for syncing MP4 <-> signals).
  * cam1.mp4 / cam2.mp4 -- the two RealSense color video streams, one MP4 each.
  * depth.h5           -- ONLY with record_depth:=true: both cameras' RealSense
                         depth streams (raw compressedDepth payloads, one PNG
                         16UC1 mm frame per row) + depth intrinsics and the
                         depth->color extrinsics. See depth_writer.py.
  * metadata.json      -- params, start time, and per-topic message counts on exit.

Depth (2026-09-14): ``record_depth`` (bool, default false) switches on
subscriptions to ``cam1_depth_topic`` / ``cam2_depth_topic`` (CompressedImage,
``16UC1; compressedDepth``), ``cam*_depth_info_topic`` (CameraInfo) and
``cam*_extrinsics_topic`` (realsense2_camera_msgs/Extrinsics; the one-shot
topic is TRANSIENT_LOCAL so it is read with a matching latched QoS).
``depth_aligned_to_color`` only records WHICH depth stream the topics carry --
the topic names themselves are what select the aligned stream (see
``depth_topics_for``). Depth frames are gated by the SAME per-camera color
warm-up clock as the MP4 frames: a depth frame is dropped exactly while that
camera's color frames are being dropped (and before its first color frame).

Subscriptions are defensive: a topic that never publishes simply leaves empty
columns (no error), so this works in sim (fake), arm-only, or full arm+gripper runs.
For EVERYTHING else (speed scaling, IO/status, TF, controller states...) start the
recorder with BAG=true (run_recorder.sh) to also capture a full `ros2 bag -a`.

Header stamps + spin-thread budget (2026-09-14). Every table fed by a STAMPED
message carries a trailing ``stamp_s`` column (float64 seconds from
``msg.header.stamp``, NaN when absent): ``gello_joint_states``,
``ur_joint_states``, ``tcp_pose``, ``wrench``, ``cam1_frames``, ``cam2_frames``.
``command`` (``Float64MultiArray``) and ``gripper`` (three ``Float32`` topics)
get NO stamp column -- those message types have no header at all, so ``t_rel_s``
(arrival) is the only timestamp they can have. The high-rate robot
subscriptions also dropped to depth ``QOS_DEPTH_ROBOT_STATE`` and the camera /
depth frame I/O moved to ``RecordingSession``'s background writer thread.
``metadata.json`` gains ``ros_lag_s_max``, ``spin_starvation_suspected``,
``native_rates_hz`` and ``dropped_frames``. The reason all of that exists is in
:mod:`gello_recorder.spin_health`.

Nothing here commands the robot or GELLO -- it is READ-ONLY (subscriptions only).

The actual table/video-writer file I/O lives in RecordingSession (recording_session.py),
shared with the interactive GUI recorder (gello_recorder_gui.py) -- this node just wires
ROS subscriptions to it and records continuously from construction to Ctrl-C (no
start/stop gating, unlike the GUI's multi-take model).
"""

import json
import math
import os
import time
from datetime import datetime

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from std_msgs.msg import Float32, Float64MultiArray

# Shared with the GUI node: the depth topic layout, the latched-extrinsics QoS
# and the (guarded, may be None) Extrinsics message type live in ONE place.
from gello_recorder.gello_gui_node import (
    EXTRINSICS_QOS,
    Extrinsics,
    depth_topics_for,
    stamp_to_seconds,
)
from gello_recorder.recording_session import RecordingSession
from gello_recorder.paths import default_repo_root
from gello_recorder.spin_health import (
    DEPTH_ON_BANNER,
    QOS_DEPTH_CAMERA,
    QOS_DEPTH_GELLO,
    QOS_DEPTH_GRIPPER,
    QOS_DEPTH_ROBOT_STATE,
    ROS_LAG_WARN_PERIOD_S,
    ROS_LAG_WARN_S,
    detect_spin_starvation,
    native_rate_table,
)

# Canonical UR joint order; GELLO and the UR driver both publish these names.
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


class GelloUrRecorder(Node):
    """Subscribe to the whole teleop pipeline and dump time-aligned HDF5 + MP4 logs."""

    def __init__(self) -> None:
        super().__init__("gello_ur_recorder")

        # --- Parameters --------------------------------------------------
        default_root = os.path.join(
            default_repo_root(),
            "ros2_ur_ws", "gello_logs",
        )
        self.output_root = str(
            self.declare_parameter("output_root", default_root).value
        )
        # If given (e.g. by run_recorder.sh so a rosbag lands in the SAME folder),
        # use it verbatim; otherwise mint session_<timestamp> under output_root.
        session_dir = str(self.declare_parameter("session_dir", "").value)
        self.sample_rate_hz = float(
            self.declare_parameter("sample_rate_hz", 100.0).value
        )
        self.flush_period_s = float(
            self.declare_parameter("flush_period_s", 2.0).value
        )
        self.cam1_topic = str(
            self.declare_parameter(
                "cam1_topic", "/cam1/cam1/color/image_raw/compressed"
            ).value
        )
        self.cam2_topic = str(
            self.declare_parameter(
                "cam2_topic", "/cam2/cam2/color/image_raw/compressed"
            ).value
        )
        self.camera_fps = float(self.declare_parameter("camera_fps", 30.0).value)
        # RealSense color sensors auto-exposure/white-balance for a couple of
        # seconds after the stream starts, often producing a green/washed-out
        # tint -- drop frames until this many seconds after each camera's FIRST
        # received frame, so the green warm-up never lands in the MP4.
        self.camera_warmup_s = float(
            self.declare_parameter("camera_warmup_s", 3.0).value
        )
        # --- Depth (opt-in) ---------------------------------------------
        # Defaults name the UNALIGNED realsense-ros 4.x topics for cam1/cam2;
        # run_recorder.sh overrides all of them when ALIGN_DEPTH=1. The
        # aligned flag is metadata for depth.h5 -- it does not rename topics.
        self.record_depth = bool(
            self.declare_parameter("record_depth", False).value
        )
        self.depth_aligned_to_color = bool(
            self.declare_parameter("depth_aligned_to_color", False).value
        )
        d1_img, d1_info, d1_ext = depth_topics_for("cam1", False)
        d2_img, d2_info, d2_ext = depth_topics_for("cam2", False)
        self.cam1_depth_topic = str(
            self.declare_parameter("cam1_depth_topic", d1_img).value)
        self.cam2_depth_topic = str(
            self.declare_parameter("cam2_depth_topic", d2_img).value)
        self.cam1_depth_info_topic = str(
            self.declare_parameter("cam1_depth_info_topic", d1_info).value)
        self.cam2_depth_info_topic = str(
            self.declare_parameter("cam2_depth_info_topic", d2_info).value)
        self.cam1_extrinsics_topic = str(
            self.declare_parameter("cam1_extrinsics_topic", d1_ext).value)
        self.cam2_extrinsics_topic = str(
            self.declare_parameter("cam2_extrinsics_topic", d2_ext).value)

        if not session_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(self.output_root, f"session_{stamp}")
        self.session_dir = session_dir

        # --- Latest-value store (for the synchronized wide table) --------
        self._gello_q = [None] * _N
        self._gello_qd = [None] * _N          # finite-difference velocity
        self._gello_q_prev = None
        self._gello_t_prev = None
        self._gello_grip = None               # GELLO gripper width 0..1
        self._cmd = [None] * _N               # bridge command to the UR
        self._ur_q = [None] * _N
        self._ur_qd = [None] * _N
        self._ur_eff = [None] * _N
        self._grip_cmd = None                 # /robotiq_gripper/command_percent
        self._grip_pos = None                 # /robotiq_gripper/position_percent
        self._wrench = [None] * 6             # fx fy fz tx ty tz
        self._tcp = [None] * 7                # x y z qx qy qz qw
        # (no cached cam frame index: _on_sample reads it back from the session,
        #  which is the only place that knows what actually reached the MP4)
        self._cam1_first_frame_t = None       # t_rel_s of cam1's first received frame
        self._cam2_first_frame_t = None       # t_rel_s of cam2's first received frame
        # Spin-thread health: ROS now - latest /joint_states header stamp.
        self._ros_lag_s = None
        self._ros_lag_s_max = None

        self._t0 = time.time()

        # --- Recording session: owns vectors.h5 + cam1.mp4 + cam2.mp4 -----
        # (+ depth.h5 when record_depth)
        self._session = RecordingSession(
            self.session_dir, camera_fps=self.camera_fps,
            record_depth=self.record_depth)
        # Depth metadata is pushed at most once per cam per stream (camera_info
        # is re-published every frame; the session is open for the whole run).
        self._depth_info_pushed = {1: False, 2: False}
        self._depth_extrinsics_pushed = {1: False, 2: False}
        if self.record_depth:
            self._session.set_depth_source(
                1, self.cam1_depth_topic, self.depth_aligned_to_color)
            self._session.set_depth_source(
                2, self.cam2_depth_topic, self.depth_aligned_to_color)

        # --- Subscriptions (READ-ONLY) -----------------------------------
        # Queue depths: worst-case staleness is  depth / publish rate  and this
        # recorder writes whatever arrives, so a deep queue only buys age. See
        # gello_recorder.spin_health for the measurement that forced this.
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
        if self.record_depth:
            # compressedDepth + camera_info take the same plain depth-10 QoS
            # as the color topic (measured compatible, see gello_gui_node);
            # the one-shot extrinsics need the latched EXTRINSICS_QOS.
            self.create_subscription(
                CompressedImage, self.cam1_depth_topic, self._on_cam1_depth,
                QOS_DEPTH_CAMERA)
            self.create_subscription(
                CompressedImage, self.cam2_depth_topic, self._on_cam2_depth,
                QOS_DEPTH_CAMERA)
            self.create_subscription(
                CameraInfo, self.cam1_depth_info_topic,
                self._on_cam1_depth_info, QOS_DEPTH_CAMERA)
            self.create_subscription(
                CameraInfo, self.cam2_depth_info_topic,
                self._on_cam2_depth_info, QOS_DEPTH_CAMERA)
            if Extrinsics is None:
                self.get_logger().warn(
                    "realsense2_camera_msgs not importable -- depth->color "
                    "extrinsics will NOT be recorded (depth frames still are)")
            else:
                self.create_subscription(
                    Extrinsics, self.cam1_extrinsics_topic,
                    self._on_cam1_extrinsics, EXTRINSICS_QOS)
                self.create_subscription(
                    Extrinsics, self.cam2_extrinsics_topic,
                    self._on_cam2_extrinsics, EXTRINSICS_QOS)

        # --- Timers ------------------------------------------------------
        self._sample_timer = self.create_timer(
            1.0 / max(1.0, self.sample_rate_hz), self._on_sample)
        self._flush_timer = self.create_timer(self.flush_period_s, self._flush)

        self._write_metadata(final=False, counts={})
        if self.record_depth:
            # Say the bill out loud. Depth is opt-in (ENABLE_DEPTH=1) precisely
            # because it is expensive, and on 2026-09-14 that expense silently
            # back-dated every robot row in a 54-take corpus. The guards are in
            # place now; the cost is not gone.
            self.get_logger().warn(DEPTH_ON_BANNER)
            depth_desc = (
                f"depth ON (aligned={str(self.depth_aligned_to_color).lower()}) "
                f"cam1_depth={self.cam1_depth_topic} cam2_depth={self.cam2_depth_topic}"
            )
        else:
            depth_desc = "depth OFF (RGB only)"
        self.get_logger().info(
            f"gello_ur_recorder logging to {self.session_dir} "
            f"@ {self.sample_rate_hz:.0f} Hz. {depth_desc}. Ctrl-C to stop & finalise."
        )

    # ---- helpers --------------------------------------------------------
    def _t(self):
        return time.time() - self._t0

    # ---- subscription callbacks ----------------------------------------
    def _on_gello(self, msg: JointState):
        r = _reorder(msg)
        if r is None:
            return
        pos, _, _ = r
        t = time.monotonic()
        if self._gello_q_prev is not None and self._gello_t_prev is not None:
            dt = t - self._gello_t_prev
            if dt > 1e-6:
                self._gello_qd = [
                    (pos[i] - self._gello_q_prev[i]) / dt for i in range(_N)
                ]
        self._gello_q_prev, self._gello_t_prev = pos, t
        self._gello_q = pos
        self._session.write_gello(
            pos, self._gello_qd, stamp_s=stamp_to_seconds(msg.header.stamp))

    def _on_gello_grip(self, msg: Float32):
        self._gello_grip = float(msg.data)
        self._session.bump("gello_grip")
        self._session.write_gello_grip(self._gello_grip, self._grip_cmd, self._grip_pos)

    def _on_cmd(self, msg: Float64MultiArray):
        # No stamp: Float64MultiArray has no header (see the module docstring).
        d = list(msg.data)
        if len(d) >= _N:
            self._cmd = [float(x) for x in d[:_N]]
            self._session.write_cmd(self._cmd)

    def _on_ur(self, msg: JointState):
        r = _reorder(msg)
        if r is None:
            return
        pos, vel, eff = r
        stamp_s = stamp_to_seconds(msg.header.stamp)
        self._note_ros_lag(stamp_s)
        self._ur_q, self._ur_qd, self._ur_eff = pos, vel, eff
        self._session.write_ur(pos, vel, eff, stamp_s=stamp_s)

    def _note_ros_lag(self, stamp_s: float) -> None:
        """Track (ROS now - /joint_states header stamp) and warn when it grows.

        Both times come from the ROS clock -- mixing in time.time() would work
        today and break silently under a sim clock. An unstamped message (NaN)
        teaches nothing and is ignored rather than reported as a huge lag."""
        if stamp_s is None or not math.isfinite(stamp_s):
            return
        try:
            now = self.get_clock().now().nanoseconds * 1e-9
        except Exception:  # noqa: BLE001 - a clock read must not kill the callback
            return
        lag = now - stamp_s
        self._ros_lag_s = lag
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
        self._grip_cmd = float(msg.data)
        self._session.bump("grip_cmd")
        self._session.write_gello_grip(self._gello_grip, self._grip_cmd, self._grip_pos)

    def _on_grip_pos(self, msg: Float32):
        self._grip_pos = float(msg.data)
        self._session.bump("grip_pos")
        self._session.write_gello_grip(self._gello_grip, self._grip_cmd, self._grip_pos)

    def _on_wrench(self, msg: WrenchStamped):
        w = msg.wrench
        self._wrench = [w.force.x, w.force.y, w.force.z,
                        w.torque.x, w.torque.y, w.torque.z]
        self._session.write_wrench(
            self._wrench, stamp_s=stamp_to_seconds(msg.header.stamp))

    def _on_tcp(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        self._tcp = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        self._session.write_tcp(
            self._tcp, stamp_s=stamp_to_seconds(msg.header.stamp))

    def _on_cam1(self, msg: CompressedImage):
        self._on_cam(msg, 1)

    def _on_cam2(self, msg: CompressedImage):
        self._on_cam(msg, 2)

    def _on_cam(self, msg: CompressedImage, cam_idx: int):
        """Warm-up gate, then QUEUE the frame -- no decode/encode here.

        The JPEG decode + MP4 encode (measured 16.2 ms per 1280x720 frame) now
        runs on RecordingSession's writer thread. The frame's ``t_rel_s`` is
        still captured on arrival (inside submit_cam_frame), so moving the work
        does not move the timestamp. ``self._cam*_frame_idx`` is gone: the
        synchronized sampler reads the authoritative index back out of the
        session instead, because submit returns before anything is encoded."""
        now = self._t()
        if cam_idx == 1:
            if self._cam1_first_frame_t is None:
                self._cam1_first_frame_t = now
            first = self._cam1_first_frame_t
        else:
            if self._cam2_first_frame_t is None:
                self._cam2_first_frame_t = now
            first = self._cam2_first_frame_t
        if now - first < self.camera_warmup_s:
            return  # discard auto-exposure warm-up frames (often green-tinted)
        self._session.submit_cam_frame(
            cam_idx, bytes(msg.data),
            stamp_s=stamp_to_seconds(msg.header.stamp))

    # ---- depth (only subscribed when record_depth) ---------------------
    def _depth_warmup_over(self, cam_idx: int) -> bool:
        """The COLOR warm-up gate, re-read for a depth frame.

        Reuses the per-camera first-COLOR-frame time set in _on_cam1/_on_cam2
        so depth frames are dropped exactly while that camera's color frames
        are (and before its first color frame arrives) -- one clock, not two.
        """
        first = self._cam1_first_frame_t if cam_idx == 1 else self._cam2_first_frame_t
        if first is None:
            return False
        return (self._t() - first) >= self.camera_warmup_s

    def _on_cam1_depth(self, msg: CompressedImage):
        if self._depth_warmup_over(1):
            self._session.submit_cam_depth_frame(
                1, bytes(msg.data), stamp_s=stamp_to_seconds(msg.header.stamp))

    def _on_cam2_depth(self, msg: CompressedImage):
        if self._depth_warmup_over(2):
            self._session.submit_cam_depth_frame(
                2, bytes(msg.data), stamp_s=stamp_to_seconds(msg.header.stamp))

    def _on_cam1_depth_info(self, msg: CameraInfo):
        self._on_depth_info(msg, 1)

    def _on_cam2_depth_info(self, msg: CameraInfo):
        self._on_depth_info(msg, 2)

    def _on_depth_info(self, msg: CameraInfo, cam_idx: int):
        if self._depth_info_pushed[cam_idx]:
            return
        self._session.set_depth_camera_info(
            cam_idx,
            width=int(msg.width), height=int(msg.height),
            distortion_model=str(msg.distortion_model),
            D=[float(x) for x in msg.d], K=[float(x) for x in msg.k],
            R=[float(x) for x in msg.r], P=[float(x) for x in msg.p],
            frame_id=str(msg.header.frame_id),
        )
        self._depth_info_pushed[cam_idx] = True

    def _on_cam1_extrinsics(self, msg):
        self._on_extrinsics(msg, 1)

    def _on_cam2_extrinsics(self, msg):
        self._on_extrinsics(msg, 2)

    def _on_extrinsics(self, msg, cam_idx: int):
        if self._depth_extrinsics_pushed[cam_idx]:
            return
        self._session.set_depth_extrinsics(
            cam_idx,
            rotation=[float(x) for x in msg.rotation],
            translation=[float(x) for x in msg.translation],
        )
        self._depth_extrinsics_pushed[cam_idx] = True

    # ---- fixed-rate synchronized snapshot ------------------------------
    def _on_sample(self):
        # The cam frame indices are read back from the session (which only
        # advances them on a successful MP4 write) rather than cached here: on
        # the async path submit_cam_frame() returns before the frame exists, so
        # a node-side cache would name a frame that is not in the file yet.
        self._session.write_sample(
            self._gello_q, self._gello_qd, self._gello_grip, self._cmd,
            self._ur_q, self._ur_qd, self._ur_eff,
            self._grip_cmd, self._grip_pos, self._wrench, self._tcp,
            self._session.latest_frame_index(1),
            self._session.latest_frame_index(2),
        )

    def _flush(self):
        self._session.flush()

    # ---- metadata + shutdown -------------------------------------------
    def _write_metadata(self, final: bool, counts: dict, duration_s: float = None):
        duration = round(self._t(), 2) if duration_s is None else duration_s
        meta = {
            "start_wall": datetime.fromtimestamp(self._t0).isoformat(),
            "session_dir": self.session_dir,
            "sample_rate_hz": self.sample_rate_hz,
            "record_depth": self.record_depth,
            "depth_aligned_to_color": self.depth_aligned_to_color,
            "duration_s": duration,
            "message_counts": counts,
            "note": "empty columns => that topic was not publishing this run",
            "finalized": final,
        }
        meta.update(self._health_summary(counts, duration))
        with open(os.path.join(self.session_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _health_summary(self, counts: dict, duration_s: float) -> dict:
        """Spin-health verdict for metadata.json (see gello_recorder.spin_health).

        ``spin_starvation_suspected`` is the 2026-09-14 regression alarm: four
        topics with different publish rates recording at the SAME rate means all
        four were being read out of a full queue, and every row in them is
        stale. ``ros_lag_s_max`` is the same defect measured directly, from the
        /joint_states header stamps."""
        report = detect_spin_starvation(native_rate_table(counts or {}, duration_s))
        lag_max = self._ros_lag_s_max
        return {
            "ros_lag_s_max": None if lag_max is None else round(float(lag_max), 4),
            "ros_lag_warn_s": ROS_LAG_WARN_S,
            "native_rates_hz": report["rates_hz"],
            "spin_starvation_suspected": bool(report["suspected"]),
            "spin_starvation_reason": report["reason"],
            "dropped_frames": self._session.dropped_frames(),
        }

    def destroy_node(self):
        try:
            # close() drains the background frame writer BEFORE finalising, so
            # the counts written below match what is actually in the files.
            stats = self._session.close()
            self._write_metadata(
                final=True,
                counts=stats["message_counts"],
                duration_s=stats["duration_s"],
            )
            health = self._health_summary(
                stats["message_counts"], stats["duration_s"])
            if health["spin_starvation_suspected"]:
                report = detect_spin_starvation(health["native_rates_hz"])
                self.get_logger().warn(
                    report["message"] or health["spin_starvation_reason"])
            dropped = health["dropped_frames"].get("total", 0)
            if dropped:
                self.get_logger().warn(
                    "{} camera/depth frame(s) were DROPPED (writer queue "
                    "full): {}".format(dropped, health["dropped_frames"]))
            self.get_logger().info(
                f"Recorder stopped. {stats['message_counts']} -> {self.session_dir}")
        except Exception:  # noqa: BLE001 - best-effort on shutdown
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GelloUrRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
