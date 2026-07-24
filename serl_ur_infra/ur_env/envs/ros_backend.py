"""ROS 2 I/O backend for UR7eEnv.

FrankaEnv talks to a Flask robot server over HTTP; we talk to the ROS 2 graph
from ur_gello_bringup directly. This class owns a rclpy node spun in a
background thread and exposes the same conceptual surface FrankaEnv gets from
its server:

    get_joint_state()      <- /joint_states        (POST /getstate analog)
    get_gripper_percent()  <- ~/position_percent
    get_gello_state()      <- /gello/joint_states  (leader, for intervention)
    get_image()            <- /camX/.../compressed (realsense2_camera driver)
    send_joint_command()   -> /forward_position_controller/commands
    send_gripper_percent() -> ~/command_percent

Cameras are consumed the same way as the rest of the UR7e line
(policy_leader_node, recorder, camera_viewer): JPEG CompressedImage topics
published by launch_cameras.sh's realsense2_camera drivers. The backend caches
the newest raw JPEG per camera; decoding happens in the env at step rate.

All getters return the latest message (with reception timestamp) — the env's
fixed-rate step loop does the pacing, so no getter ever blocks.

UNTESTED SKELETON — wiring (QoS, joint name ordering, staleness thresholds)
to be validated together on the real stack.
"""

import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import Float32, Float64MultiArray

    _ROS_AVAILABLE = True
except ImportError:  # allow import (e.g. fake_env on a non-ROS machine)
    _ROS_AVAILABLE = False


# UR joint order used by ur_gello_bringup command path.
UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class URRosBackend:
    def __init__(
        self,
        ros_cfg: Dict[str, str],
        camera_topics: Optional[Dict[str, str]] = None,
        upsampler_cfg: Optional[Dict[str, float]] = None,
        dry_run: bool = True,
    ):
        if not _ROS_AVAILABLE:
            raise RuntimeError("rclpy not available — use fake_env=True")

        self.dry_run = dry_run
        self._lock = threading.Lock()
        self._shutdown = False

        # latest-state caches: (data, wall-clock reception time)
        self._q: Optional[Tuple[np.ndarray, float]] = None          # (6,) robot joints
        self._dq: Optional[Tuple[np.ndarray, float]] = None         # (6,) robot joint vel
        self._gripper: Optional[Tuple[float, float]] = None         # 0.0 open..1.0 closed
        self._gello: Optional[Tuple[np.ndarray, float]] = None      # (7,) leader q + grip
        self._wrench: Optional[Tuple[np.ndarray, float]] = None     # (6,) fx..tz, TCP F/T
        self._tcp_pose: Optional[Tuple[np.ndarray, float]] = None   # (7,) xyz + quat
        self._images: Dict[str, Tuple[bytes, float]] = {}           # name -> raw JPEG

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("ur7e_env_backend")

        self._node.create_subscription(
            JointState, ros_cfg["joint_states_topic"], self._on_joint_states, 10
        )
        self._node.create_subscription(
            JointState, ros_cfg["gello_topic"], self._on_gello, 10
        )
        self._node.create_subscription(
            Float32, ros_cfg["gripper_state_topic"], self._on_gripper, 10
        )
        if "wrench_topic" in ros_cfg:
            self._node.create_subscription(
                WrenchStamped, ros_cfg["wrench_topic"], self._on_wrench, 10
            )
        if "tcp_pose_topic" in ros_cfg:
            self._node.create_subscription(
                PoseStamped, ros_cfg["tcp_pose_topic"], self._on_tcp_pose, 10
            )
        # depth-1 subscriptions: only the newest frame is ever interesting.
        # VERIFY(hw): all subscriptions here use default (reliable) QoS. If a
        # publisher is best-effort the subscription silently never fires —
        # check every topic actually delivers on first bring-up (ros2 topic hz).
        for cam_name, topic in (camera_topics or {}).items():
            self._node.create_subscription(
                CompressedImage,
                topic,
                self._make_image_cb(cam_name),
                1,
            )

        self._cmd_pub = self._node.create_publisher(
            Float64MultiArray, ros_cfg["command_topic"], 10
        )
        self._gripper_pub = self._node.create_publisher(
            Float32, ros_cfg["gripper_command_topic"], 10
        )

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # ---- command upsampler: 10 Hz targets -> slew-limited 250 Hz stream ---- #
        ucfg = upsampler_cfg or {"hz": 250.0, "max_step_rad": 0.002}
        self._up_hz = float(ucfg["hz"])
        self._up_step = float(ucfg["max_step_rad"])
        self._q_target: Optional[np.ndarray] = None   # latest 10 Hz goal
        self._q_stream: Optional[np.ndarray] = None   # what we're publishing now
        self._up_thread = threading.Thread(target=self._upsample_loop, daemon=True)
        self._up_thread.start()

    # ------------------------------------------------------------------ #
    # callbacks                                                           #
    # ------------------------------------------------------------------ #
    def _spin(self):
        rclpy.spin(self._node)

    def _on_joint_states(self, msg: "JointState"):
        # Reorder by name — joint_state_broadcaster order is not guaranteed.
        try:
            idx = [msg.name.index(n) for n in UR_JOINT_NAMES]
        except ValueError:
            return  # not the UR arm (e.g. gripper joint-only message)
        q = np.array([msg.position[i] for i in idx])
        dq = (
            np.array([msg.velocity[i] for i in idx])
            if len(msg.velocity) == len(msg.name)
            else np.zeros(6)
        )
        now = time.monotonic()
        with self._lock:
            self._q = (q, now)
            self._dq = (dq, now)

    def _on_gello(self, msg: "JointState"):
        arr = np.asarray(msg.position, dtype=float)  # 6 joints (+ gripper)
        with self._lock:
            self._gello = (arr, time.monotonic())

    def _on_gripper(self, msg: "Float32"):
        with self._lock:
            self._gripper = (float(msg.data), time.monotonic())

    def _on_wrench(self, msg: "WrenchStamped"):
        w = msg.wrench
        arr = np.array(
            [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
        )
        with self._lock:
            self._wrench = (arr, time.monotonic())

    def _on_tcp_pose(self, msg: "PoseStamped"):
        p, o = msg.pose.position, msg.pose.orientation
        arr = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w])  # xyz + quat (xyzw)
        with self._lock:
            self._tcp_pose = (arr, time.monotonic())

    def _make_image_cb(self, cam_name: str):
        def _cb(msg: "CompressedImage"):
            with self._lock:
                self._images[cam_name] = (bytes(msg.data), time.monotonic())

        return _cb

    # ------------------------------------------------------------------ #
    # getters (never block; return (data, age_seconds))                   #
    # ------------------------------------------------------------------ #
    def _aged(self, slot):
        if slot is None:
            return None, float("inf")
        data, t = slot
        return data, time.monotonic() - t

    def get_joint_state(self):
        with self._lock:
            q, age_q = self._aged(self._q)
            dq, _ = self._aged(self._dq)
        return q, dq, age_q

    def get_gripper_percent(self):
        with self._lock:
            return self._aged(self._gripper)

    def get_gello_state(self):
        with self._lock:
            return self._aged(self._gello)

    def get_wrench(self):
        with self._lock:
            return self._aged(self._wrench)

    def get_tcp_pose(self):
        with self._lock:
            return self._aged(self._tcp_pose)

    def get_image(self, cam_name: str):
        """Returns (raw JPEG bytes or None, age_seconds)."""
        with self._lock:
            return self._aged(self._images.get(cam_name))

    # ------------------------------------------------------------------ #
    # commands                                                            #
    # ------------------------------------------------------------------ #
    def send_joint_command(self, q_cmd: np.ndarray):
        """Update the upsampler's target (does NOT publish directly)."""
        with self._lock:
            self._q_target = np.asarray(q_cmd, dtype=float).reshape(6).copy()

    def reset_command_stream(self):
        """Drop target/stream so the next target re-seeds from measured joints.

        Call on env.reset(): a stale stream from the previous episode must not
        race the new one.
        """
        with self._lock:
            self._q_target = None
            self._q_stream = None

    def _upsample_loop(self):
        """250 Hz worker: slew the published stream toward the latest target.

        Seeding: the stream starts at the *measured* joints, so the first
        published command is jump-free by construction. With no target yet (or
        after reset_command_stream) nothing is published — same "never emit an
        unearned command" rule as eef_delta's DISENGAGED state.

        Once at the target it keeps publishing the held pose; if the env dies
        the robot simply holds position. TODO(together): target-staleness
        policy (stop publishing after N s without a fresh target?), to be
        decided with the other safe-stop cases.
        """
        period = 1.0 / self._up_hz
        while not self._shutdown:
            t0 = time.monotonic()
            with self._lock:
                target = self._q_target
                stream = self._q_stream
                q_meas = self._q[0] if self._q is not None else None
            if target is not None:
                if stream is None:
                    if q_meas is None:
                        time.sleep(period)
                        continue
                    stream = q_meas.copy()
                stream = stream + np.clip(
                    target - stream, -self._up_step, self._up_step
                )
                with self._lock:
                    self._q_stream = stream
                if not self.dry_run:
                    msg = Float64MultiArray()
                    msg.data = [float(v) for v in stream]
                    self._cmd_pub.publish(msg)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def send_gripper_percent(self, fraction: float):
        """Robotiq command_percent convention: 0.0 = OPEN .. 1.0 = CLOSED."""
        if self.dry_run:
            return
        msg = Float32()
        msg.data = float(np.clip(fraction, 0.0, 1.0))
        self._gripper_pub.publish(msg)

    def close(self):
        self._shutdown = True
        self._node.destroy_node()
