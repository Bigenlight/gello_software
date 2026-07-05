#!/usr/bin/env python3
"""Scripted /gello/joint_states publisher for real-stack handshake tests.
Scenarios: quiet | abrupt | never_settle. Arm (mock) starts near 0."""
import argparse, math, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# Reachable target pose (arm mock starts at ~0); small so JTC move is quick.
Q0 = [0.3, -0.4, 0.4, -0.4, -0.4, 0.0]


class G(Node):
    def __init__(self, scenario, jump_at=1.5):
        super().__init__("gello_pub")
        self.s = scenario; self.jump_at = jump_at; self.t0 = time.monotonic()
        self.pub = self.create_publisher(JointState, "/gello/joint_states", 10)
        self.create_timer(1/30, self.tick)

    def tick(self):
        t = time.monotonic() - self.t0
        q = list(Q0)
        if self.s == "quiet":
            q[0] += 0.005 * math.sin(2*math.pi*5*t)
        elif self.s == "abrupt":
            q[0] += 0.005 * math.sin(2*math.pi*5*t)
            if t >= self.jump_at:
                q[0] += 0.40           # jump mid-approach
        elif self.s == "never_settle":
            q[0] += 0.20 * math.sin(2*math.pi*t/3.0)
        m = JointState(); m.name = JOINTS; m.position = [float(x) for x in q]
        self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", default="quiet")
    ap.add_argument("--jump-at", type=float, default=1.5,
                    help="abrupt: seconds (gello_pub clock) at which joint0 jumps "
                         "+0.4 rad; set > first-chase-start so it lands mid-approach")
    a = ap.parse_args()
    rclpy.init(); rclpy.spin(G(a.scenario, a.jump_at))


if __name__ == "__main__":
    main()
