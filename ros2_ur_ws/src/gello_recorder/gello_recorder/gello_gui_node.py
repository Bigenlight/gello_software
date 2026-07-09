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
"""

import os
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, Float64MultiArray

from gello_recorder.recording_session import RecordingSession

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
    ) -> None:
        super().__init__(node_name)

        self.cam1_topic = cam1_topic
        self.cam2_topic = cam2_topic
        self.camera_fps = camera_fps
        self.camera_warmup_s = camera_warmup_s
        self.output_root = os.path.expanduser(output_root)

        # --- Recording-session state (guarded by _session_lock) ----------
        self._session_lock = threading.Lock()
        self._session: Optional[RecordingSession] = None
        self._take_index = 0

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

        # --- Subscriptions (READ-ONLY, always active) ---------------------
        self.create_subscription(JointState, "/gello/joint_states", self._on_gello, 50)
        self.create_subscription(
            Float32, "/gripper/gripper_client/target_gripper_width_percent",
            self._on_gello_grip, 20)
        self.create_subscription(
            Float64MultiArray, "/forward_position_controller/commands", self._on_cmd, 50)
        self.create_subscription(JointState, "/joint_states", self._on_ur, 100)
        self.create_subscription(
            Float32, "/robotiq_gripper/command_percent", self._on_grip_cmd, 20)
        self.create_subscription(
            Float32, "/robotiq_gripper/position_percent", self._on_grip_pos, 20)
        self.create_subscription(
            WrenchStamped, "/force_torque_sensor_broadcaster/wrench", self._on_wrench, 50)
        self.create_subscription(
            PoseStamped, "/tcp_pose_broadcaster/pose", self._on_tcp, 50)
        self.create_subscription(CompressedImage, self.cam1_topic, self._on_cam1, 10)
        self.create_subscription(CompressedImage, self.cam2_topic, self._on_cam2, 10)

        self.get_logger().info(
            f"gello_recorder_gui_node up. cam1={self.cam1_topic} cam2={self.cam2_topic} "
            f"warmup={self.camera_warmup_s:.1f}s output_root={self.output_root}"
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
            session = RecordingSession(take_dir, camera_fps=self.camera_fps)
            self._session = session
        self.get_logger().info(f"Recording started -> {take_dir}")
        return take_dir

    def stop_recording(self) -> dict:
        with self._session_lock:
            session = self._session
            if session is None:
                raise RuntimeError("stop_recording() called while not recording")
            self._session = None
        # close() (file I/O) happens OUTSIDE the lock -- callbacks already see
        # self._session is None by this point and will no-op, so nothing else
        # touches `session` concurrently from here on.
        stats = session.close()
        stats["session_dir"] = session.session_dir
        self.get_logger().info(f"Recording stopped -> {stats}")
        return stats

    def get_preview_frames(self):
        with self._state_lock:
            return self._cam1_frame, self._cam2_frame

    def get_state_snapshot(self) -> dict:
        now = time.monotonic()
        with self._state_lock:
            cam1_age = None if self._cam1_last_frame_t is None else now - self._cam1_last_frame_t
            cam2_age = None if self._cam2_last_frame_t is None else now - self._cam2_last_frame_t
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
            }

    # ---- subscription callbacks -------------------------------------------
    def _on_gello(self, msg: JointState):
        r = _reorder(msg)
        if r is None:
            return
        pos, _, _ = r
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
                self._session.write_gello(pos, qd_snapshot)

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
        with self._state_lock:
            self._ur_q, self._ur_qd, self._ur_eff = pos, vel, eff
        with self._session_lock:
            if self._session is not None:
                self._session.write_ur(pos, vel, eff)

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
        with self._state_lock:
            self._wrench = wrench6
        with self._session_lock:
            if self._session is not None:
                self._session.write_wrench(wrench6)

    def _on_tcp(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        tcp7 = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        with self._state_lock:
            self._tcp = tcp7
        with self._session_lock:
            if self._session is not None:
                self._session.write_tcp(tcp7)

    def _on_cam1(self, msg: CompressedImage):
        self._on_cam(msg, cam_idx=1)

    def _on_cam2(self, msg: CompressedImage):
        self._on_cam(msg, cam_idx=2)

    def _on_cam(self, msg: CompressedImage, cam_idx: int):
        raw = bytes(msg.data)
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        now = time.monotonic()
        with self._state_lock:
            if frame is not None:
                if cam_idx == 1:
                    self._cam1_frame = frame
                else:
                    self._cam2_frame = frame
            if cam_idx == 1:
                self._cam1_last_frame_t = now
                if self._cam1_first_frame_t is None:
                    self._cam1_first_frame_t = now
            else:
                self._cam2_last_frame_t = now
                if self._cam2_first_frame_t is None:
                    self._cam2_first_frame_t = now
        # Recording only ever starts once cameras_ready() (warmup already
        # elapsed), so every frame written into an active session is good --
        # no per-frame warmup skip needed here (unlike the headless node).
        with self._session_lock:
            if self._session is not None:
                if cam_idx == 1:
                    self._session.write_cam1_frame(raw)
                else:
                    self._session.write_cam2_frame(raw)

    def destroy_node(self):
        with self._session_lock:
            session = self._session
            self._session = None
        if session is not None:
            try:
                session.close()
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
