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
  * synchronized.csv  -- ONE wide table sampled at sample_rate_hz (default 100 Hz)
                         holding the latest value of every signal, time-aligned.
                         This is the file to open in pandas/Excel for plotting.
  * per-topic CSVs     -- gello_joint_states.csv, ur_joint_states.csv, command.csv,
                         gripper.csv, wrench.csv, tcp_pose.csv (native rates, full detail)
  * metadata.json      -- params, start time, and per-topic message counts on exit.

Subscriptions are defensive: a topic that never publishes simply leaves empty
columns (no error), so this works in sim (fake), arm-only, or full arm+gripper runs.
For EVERYTHING else (speed scaling, IO/status, TF, controller states...) start the
recorder with BAG=true (run_recorder.sh) to also capture a full `ros2 bag -a`.

Nothing here commands the robot or GELLO -- it is READ-ONLY (subscriptions only).
"""

import csv
import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray

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
    """Subscribe to the whole teleop pipeline and dump time-aligned CSV logs."""

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

        if not session_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(self.output_root, f"session_{stamp}")
        self.session_dir = session_dir
        os.makedirs(self.session_dir, exist_ok=True)

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
        self._counts = {}                     # per-topic message counters

        self._t0 = time.time()

        # --- Open CSV writers --------------------------------------------
        self._files = []
        self._sync_w = self._open_csv(
            "synchronized.csv",
            ["t_rel_s", "t_wall"]
            + [f"gello_q{i+1}" for i in range(_N)]
            + [f"gello_qd{i+1}" for i in range(_N)]
            + ["gello_grip"]
            + [f"cmd{i+1}" for i in range(_N)]
            + [f"ur_q{i+1}" for i in range(_N)]
            + [f"ur_qd{i+1}" for i in range(_N)]
            + [f"ur_eff{i+1}" for i in range(_N)]
            + ["grip_cmd", "grip_pos"]
            + ["fx", "fy", "fz", "tx", "ty", "tz"]
            + ["tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"],
        )
        self._gello_w = self._open_csv(
            "gello_joint_states.csv",
            ["t_rel_s"] + [f"q{i+1}" for i in range(_N)] + [f"qd{i+1}" for i in range(_N)],
        )
        self._ur_w = self._open_csv(
            "ur_joint_states.csv",
            ["t_rel_s"]
            + [f"q{i+1}" for i in range(_N)]
            + [f"qd{i+1}" for i in range(_N)]
            + [f"eff{i+1}" for i in range(_N)],
        )
        self._cmd_w = self._open_csv(
            "command.csv", ["t_rel_s"] + [f"cmd{i+1}" for i in range(_N)]
        )
        self._grip_w = self._open_csv(
            "gripper.csv", ["t_rel_s", "gello_grip", "grip_cmd", "grip_pos"]
        )
        self._wrench_w = self._open_csv(
            "wrench.csv", ["t_rel_s", "fx", "fy", "fz", "tx", "ty", "tz"]
        )
        self._tcp_w = self._open_csv(
            "tcp_pose.csv", ["t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw"]
        )

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

        # --- Timers ------------------------------------------------------
        self._sample_timer = self.create_timer(
            1.0 / max(1.0, self.sample_rate_hz), self._on_sample)
        self._flush_timer = self.create_timer(self.flush_period_s, self._flush)

        self._write_metadata(final=False)
        self.get_logger().info(
            f"gello_ur_recorder logging to {self.session_dir} "
            f"@ {self.sample_rate_hz:.0f} Hz. Ctrl-C to stop & finalise."
        )

    # ---- helpers --------------------------------------------------------
    def _open_csv(self, name, header):
        f = open(os.path.join(self.session_dir, name), "w", newline="")
        w = csv.writer(f)
        w.writerow(header)
        self._files.append(f)
        return w

    def _t(self):
        return time.time() - self._t0

    def _bump(self, key):
        self._counts[key] = self._counts.get(key, 0) + 1

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
        self._bump("gello_joint_states")
        self._gello_w.writerow(
            [f"{self._t():.4f}"] + pos
            + [None if v is None else f"{v:.5f}" for v in self._gello_qd]
        )

    def _write_grip(self):
        def f(v):
            return None if v is None else f"{v:.4f}"
        self._grip_w.writerow(
            [f"{self._t():.4f}", f(self._gello_grip),
             f(self._grip_cmd), f(self._grip_pos)])

    def _on_gello_grip(self, msg: Float32):
        self._gello_grip = float(msg.data)
        self._bump("gello_grip")
        self._write_grip()

    def _on_cmd(self, msg: Float64MultiArray):
        d = list(msg.data)
        if len(d) >= _N:
            self._cmd = [float(x) for x in d[:_N]]
            self._bump("command")
            self._cmd_w.writerow([f"{self._t():.4f}"] + self._cmd)

    def _on_ur(self, msg: JointState):
        r = _reorder(msg)
        if r is None:
            return
        pos, vel, eff = r
        self._ur_q, self._ur_qd, self._ur_eff = pos, vel, eff
        self._bump("ur_joint_states")
        self._ur_w.writerow(
            [f"{self._t():.4f}"]
            + [None if v is None else f"{v:.6f}" for v in pos]
            + [None if v is None else f"{v:.6f}" for v in vel]
            + [None if v is None else f"{v:.4f}" for v in eff]
        )

    def _on_grip_cmd(self, msg: Float32):
        self._grip_cmd = float(msg.data)
        self._bump("grip_cmd")
        self._write_grip()

    def _on_grip_pos(self, msg: Float32):
        self._grip_pos = float(msg.data)
        self._bump("grip_pos")
        self._write_grip()

    def _on_wrench(self, msg: WrenchStamped):
        w = msg.wrench
        self._wrench = [w.force.x, w.force.y, w.force.z,
                        w.torque.x, w.torque.y, w.torque.z]
        self._bump("wrench")
        self._wrench_w.writerow(
            [f"{self._t():.4f}"] + [f"{v:.5f}" for v in self._wrench])

    def _on_tcp(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        self._tcp = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        self._bump("tcp_pose")
        self._tcp_w.writerow(
            [f"{self._t():.4f}"] + [f"{v:.6f}" for v in self._tcp])

    # ---- fixed-rate synchronized snapshot ------------------------------
    def _on_sample(self):
        def fmt(v, p=6):
            return None if v is None else f"{v:.{p}f}"
        row = [f"{self._t():.4f}", f"{time.time():.4f}"]
        row += [fmt(v) for v in self._gello_q]
        row += [fmt(v, 5) for v in self._gello_qd]
        row += [fmt(self._gello_grip, 4)]
        row += [fmt(v) for v in self._cmd]
        row += [fmt(v) for v in self._ur_q]
        row += [fmt(v) for v in self._ur_qd]
        row += [fmt(v, 4) for v in self._ur_eff]
        row += [fmt(self._grip_cmd, 4), fmt(self._grip_pos, 4)]
        row += [fmt(v, 5) for v in self._wrench]
        row += [fmt(v) for v in self._tcp]
        self._sync_w.writerow(row)

    def _flush(self):
        for f in self._files:
            f.flush()

    # ---- metadata + shutdown -------------------------------------------
    def _write_metadata(self, final: bool):
        meta = {
            "start_wall": datetime.fromtimestamp(self._t0).isoformat(),
            "session_dir": self.session_dir,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_s": round(self._t(), 2),
            "message_counts": self._counts,
            "note": "empty columns => that topic was not publishing this run",
            "finalized": final,
        }
        with open(os.path.join(self.session_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def destroy_node(self):
        try:
            self._write_metadata(final=True)
            for f in self._files:
                f.flush()
                f.close()
            self.get_logger().info(
                f"Recorder stopped. {self._counts} -> {self.session_dir}")
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
