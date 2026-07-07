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
  * metadata.json      -- params, start time, and per-topic message counts on exit.

Subscriptions are defensive: a topic that never publishes simply leaves empty
columns (no error), so this works in sim (fake), arm-only, or full arm+gripper runs.
For EVERYTHING else (speed scaling, IO/status, TF, controller states...) start the
recorder with BAG=true (run_recorder.sh) to also capture a full `ros2 bag -a`.

Nothing here commands the robot or GELLO -- it is READ-ONLY (subscriptions only).

The actual table/video-writer file I/O lives in RecordingSession (recording_session.py),
shared with the interactive GUI recorder (gello_recorder_gui.py) -- this node just wires
ROS subscriptions to it and records continuously from construction to Ctrl-C (no
start/stop gating, unlike the GUI's multi-take model).
"""

import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, Float64MultiArray

from gello_recorder.recording_session import RecordingSession

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
            os.environ.get("GELLO_REPO_ROOT", "/home/laptop3/gello_software"),
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
        self._cam1_frame_idx = None           # latest cam1 mp4 frame index
        self._cam2_frame_idx = None           # latest cam2 mp4 frame index
        self._cam1_first_frame_t = None       # t_rel_s of cam1's first received frame
        self._cam2_first_frame_t = None       # t_rel_s of cam2's first received frame

        self._t0 = time.time()

        # --- Recording session: owns vectors.h5 + cam1.mp4 + cam2.mp4 -----
        self._session = RecordingSession(self.session_dir, camera_fps=self.camera_fps)

        # --- Subscriptions (READ-ONLY) -----------------------------------
        self.create_subscription(
            JointState, "/gello/joint_states", self._on_gello, 50)
        self.create_subscription(
            Float32, "/gripper/gripper_client/target_gripper_width_percent",
            self._on_gello_grip, 20)
        self.create_subscription(
            Float64MultiArray, "/forward_position_controller/commands",
            self._on_cmd, 50)
        self.create_subscription(
            JointState, "/joint_states", self._on_ur, 100)
        self.create_subscription(
            Float32, "/robotiq_gripper/command_percent", self._on_grip_cmd, 20)
        self.create_subscription(
            Float32, "/robotiq_gripper/position_percent", self._on_grip_pos, 20)
        self.create_subscription(
            WrenchStamped, "/force_torque_sensor_broadcaster/wrench",
            self._on_wrench, 50)
        self.create_subscription(
            PoseStamped, "/tcp_pose_broadcaster/pose", self._on_tcp, 50)
        self.create_subscription(
            CompressedImage, self.cam1_topic, self._on_cam1, 10)
        self.create_subscription(
            CompressedImage, self.cam2_topic, self._on_cam2, 10)

        # --- Timers ------------------------------------------------------
        self._sample_timer = self.create_timer(
            1.0 / max(1.0, self.sample_rate_hz), self._on_sample)
        self._flush_timer = self.create_timer(self.flush_period_s, self._flush)

        self._write_metadata(final=False, counts={})
        self.get_logger().info(
            f"gello_ur_recorder logging to {self.session_dir} "
            f"@ {self.sample_rate_hz:.0f} Hz. Ctrl-C to stop & finalise."
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
        self._session.write_gello(pos, self._gello_qd)

    def _on_gello_grip(self, msg: Float32):
        self._gello_grip = float(msg.data)
        self._session.bump("gello_grip")
        self._session.write_gello_grip(self._gello_grip, self._grip_cmd, self._grip_pos)

    def _on_cmd(self, msg: Float64MultiArray):
        d = list(msg.data)
        if len(d) >= _N:
            self._cmd = [float(x) for x in d[:_N]]
            self._session.write_cmd(self._cmd)

    def _on_ur(self, msg: JointState):
        r = _reorder(msg)
        if r is None:
            return
        pos, vel, eff = r
        self._ur_q, self._ur_qd, self._ur_eff = pos, vel, eff
        self._session.write_ur(pos, vel, eff)

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
        self._session.write_wrench(self._wrench)

    def _on_tcp(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        self._tcp = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        self._session.write_tcp(self._tcp)

    def _on_cam1(self, msg: CompressedImage):
        now = self._t()
        if self._cam1_first_frame_t is None:
            self._cam1_first_frame_t = now
        if now - self._cam1_first_frame_t < self.camera_warmup_s:
            return  # discard auto-exposure warm-up frames (often green-tinted)
        idx = self._session.write_cam1_frame(bytes(msg.data))
        if idx >= 0:
            self._cam1_frame_idx = idx

    def _on_cam2(self, msg: CompressedImage):
        now = self._t()
        if self._cam2_first_frame_t is None:
            self._cam2_first_frame_t = now
        if now - self._cam2_first_frame_t < self.camera_warmup_s:
            return  # discard auto-exposure warm-up frames (often green-tinted)
        idx = self._session.write_cam2_frame(bytes(msg.data))
        if idx >= 0:
            self._cam2_frame_idx = idx

    # ---- fixed-rate synchronized snapshot ------------------------------
    def _on_sample(self):
        self._session.write_sample(
            self._gello_q, self._gello_qd, self._gello_grip, self._cmd,
            self._ur_q, self._ur_qd, self._ur_eff,
            self._grip_cmd, self._grip_pos, self._wrench, self._tcp,
            self._cam1_frame_idx, self._cam2_frame_idx,
        )

    def _flush(self):
        self._session.flush()

    # ---- metadata + shutdown -------------------------------------------
    def _write_metadata(self, final: bool, counts: dict, duration_s: float = None):
        meta = {
            "start_wall": datetime.fromtimestamp(self._t0).isoformat(),
            "session_dir": self.session_dir,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_s": round(self._t(), 2) if duration_s is None else duration_s,
            "message_counts": counts,
            "note": "empty columns => that topic was not publishing this run",
            "finalized": final,
        }
        with open(os.path.join(self.session_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def destroy_node(self):
        try:
            stats = self._session.close()
            self._write_metadata(
                final=True,
                counts=stats["message_counts"],
                duration_s=stats["duration_s"],
            )
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
