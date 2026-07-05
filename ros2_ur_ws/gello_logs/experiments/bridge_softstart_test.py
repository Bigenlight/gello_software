#!/usr/bin/env python3
"""Confirm the REAL bridge's soft-start ramp: with a constant 0.3 rad gap the
per-cycle slew should ramp up over soft_start_s instead of jumping to the full
0.0025 rad clamp immediately. Records /forward_position_controller/commands."""
from __future__ import annotations
import json, subprocess, threading, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


class Probe(Node):
    def __init__(self):
        super().__init__("softstart_probe")
        self.t0 = time.monotonic()
        self.rows = []  # (t, cmd0)
        self._js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._gello_pub = self.create_publisher(JointState, "/gello/joint_states", 10)
        self.create_timer(0.01, self._js)      # actual pose = 0
        self.create_timer(1/30, self._gello)   # gello = 0.3 on joint0 (constant gap)
        self.create_subscription(Float64MultiArray,
            "/forward_position_controller/commands", self._cmd, 10)

    def _js(self):
        m = JointState(); m.name = JOINTS; m.position = [0.0]*6
        self._js_pub.publish(m)

    def _gello(self):
        m = JointState(); m.name = JOINTS; m.position = [0.3,0.0,0.0,0.0,0.0,0.0]
        self._gello_pub.publish(m)

    def _cmd(self, msg):
        self.rows.append((time.monotonic()-self.t0, msg.data[0]))


def main():
    rclpy.init()
    node = Probe()
    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True); th.start()
    proc = subprocess.Popen(
        ["ros2","run","ur_gello_bringup","gello_ur_bridge","--ros-args",
         "-p","filter_type:=one_euro","-p","max_step_rad:=0.0025",
         "-p","publish_rate_hz:=250.0","-p","soft_start_s:=0.7"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)
    proc.terminate()
    try: proc.wait(timeout=3)
    except subprocess.TimeoutExpired: proc.kill()

    r = np.array(node.rows)
    # per-cycle speed of joint0 command
    if len(r) > 5:
        t = r[:,0]; q = r[:,1]
        # align to first command (seed) time
        t = t - t[0]
        speed = np.abs(np.diff(q)) * 250.0  # rad/s (approx, 250Hz)
        tm = t[1:]
        def band(a,b):
            m = (tm>=a)&(tm<b)
            return round(float(speed[m].max()),4) if m.any() else None
        out = {
            "peak_speed_0.00_0.10s": band(0.0,0.10),
            "peak_speed_0.10_0.40s": band(0.10,0.40),
            "peak_speed_0.40_0.70s": band(0.40,0.70),
            "peak_speed_0.70s_plus": band(0.70,10),
            "full_clamp_speed": 0.625,
            "n_cmds": len(r),
        }
        print("SOFTSTART " + json.dumps(out))
    else:
        print("SOFTSTART {\"error\": \"too few commands\", \"n\": %d}" % len(r))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
