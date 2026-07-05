#!/usr/bin/env python3
"""Hardware-free test of the convergence-gated gello_move_to_start handshake.

Runs the REAL modified gello_move_to_start node (as a subprocess) against a mock
robot: a FollowJointTrajectory server that ramps a simulated arm to each goal and
publishes /joint_states, plus list_controllers / switch_controller services that
record IF and WHEN the node hands over and the live |gello - arm| gap at that
instant. A scripted /gello/joint_states leader drives three scenarios:

  quiet         leader holds pose (small tremor)                 -> expect SWITCH, gap tiny
  abrupt        leader JUMPS 0.4 rad mid-approach (the worry)    -> expect SWITCH only AFTER
                                                                    it settles, >=2 chase goals
  never_settle  leader oscillates forever                        -> expect NO SWITCH (timeout)

Usage: python mock_handshake_test.py --scenario abrupt
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys, threading, time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from controller_manager_msgs.msg import ControllerState
from sensor_msgs.msg import JointState

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
Q0 = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]          # leader nominal pose
ARM_START = [Q0[0] - 0.6] + Q0[1:]                   # arm starts 0.6 rad off (joint0)


class MockRobot(Node):
    def __init__(self, scenario: str):
        super().__init__("mock_robot")
        self.scenario = scenario
        self.cb = ReentrantCallbackGroup()
        self.arm = list(ARM_START)
        self.t0 = time.monotonic()
        self.switched = False
        self.t_switch = None
        self.gap_at_switch = None
        self.n_goals = 0

        self._js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._gello_pub = self.create_publisher(JointState, "/gello/joint_states", 10)
        self.create_timer(0.01, self._pub_js, callback_group=self.cb)          # 100 Hz
        self.create_timer(1.0 / 30.0, self._pub_gello, callback_group=self.cb)  # 30 Hz

        self._srv_list = self.create_service(
            ListControllers, "/controller_manager/list_controllers",
            self._on_list, callback_group=self.cb)
        self._srv_switch = self.create_service(
            SwitchController, "/controller_manager/switch_controller",
            self._on_switch, callback_group=self.cb)
        self._action = ActionServer(
            self, FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
            execute_callback=self._execute, callback_group=self.cb)

    def _gello_pose(self) -> list[float]:
        t = time.monotonic() - self.t0
        q = list(Q0)
        if self.scenario == "quiet":
            q[0] += 0.005 * math.sin(2 * math.pi * 5 * t)          # tremor only
        elif self.scenario == "abrupt":
            q[0] += 0.005 * math.sin(2 * math.pi * 5 * t)
            if t >= 1.0:                                            # jump mid-approach
                q[0] += 0.40
        elif self.scenario == "never_settle":
            q[0] += 0.20 * math.sin(2 * math.pi * t / 3.0)          # forever moving
        return q

    def _pub_js(self):
        m = JointState(); m.name = JOINTS; m.position = list(self.arm)
        self._js_pub.publish(m)

    def _pub_gello(self):
        m = JointState(); m.name = JOINTS; m.position = self._gello_pose()
        self._gello_pub.publish(m)

    def _on_list(self, req, resp):
        c = ControllerState()
        c.name = "scaled_joint_trajectory_controller"; c.state = "active"
        resp.controller = [c]
        return resp

    def _on_switch(self, req, resp):
        # This is the HANDOVER instant — record it and the live gap.
        self.switched = True
        self.t_switch = time.monotonic() - self.t0
        g = self._gello_pose()
        self.gap_at_switch = max(abs(g[i] - self.arm[i]) for i in range(6))
        resp.ok = True
        self.get_logger().info(f"SWITCH called at t={self.t_switch:.2f}s "
                               f"gap={self.gap_at_switch:.4f} rad")
        return resp

    def _execute(self, goal_handle):
        self.n_goals += 1
        traj = goal_handle.request.trajectory
        target = list(traj.points[-1].positions)
        d = traj.points[-1].time_from_start
        dur = max(0.05, d.sec + d.nanosec * 1e-9)
        start = list(self.arm); t_start = time.monotonic()
        while True:
            f = min(1.0, (time.monotonic() - t_start) / dur)
            self.arm = [start[i] + f * (target[i] - start[i]) for i in range(6)]
            if f >= 1.0:
                break
            time.sleep(0.01)
        goal_handle.succeed()
        r = FollowJointTrajectory.Result(); r.error_code = 0
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    choices=["quiet", "abrupt", "never_settle"])
    ap.add_argument("--timeout", type=float, default=16.0)
    a = ap.parse_args()

    rclpy.init()
    node = MockRobot(a.scenario)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    # Launch the REAL modified move_to_start node in the same ROS domain.
    proc = subprocess.Popen(
        ["ros2", "run", "ur_gello_bringup", "gello_move_to_start", "--ros-args",
         "-p", "start_mode:=gello",
         "-p", "chase_timeout_s:=8.0",
         "-p", "chase_tol:=0.025",
         "-p", "chase_dwell_s:=0.4",
         "-p", "chase_v_budget:=0.3",
         "-p", "min_traj_duration:=0.5",
         "-p", "activation_timeout:=10.0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t_end = time.monotonic() + a.timeout
    while time.monotonic() < t_end and not node.switched and proc.poll() is None:
        time.sleep(0.05)
    time.sleep(0.3)
    exit_code = proc.poll()
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()

    verdict = {
        "scenario": a.scenario,
        "switched": bool(node.switched),
        "t_switch_s": round(node.t_switch, 2) if node.t_switch else None,
        "gap_at_switch_rad": round(node.gap_at_switch, 4) if node.gap_at_switch is not None else None,
        "n_chase_goals": node.n_goals,
        "node_exit_code": exit_code,
    }
    print("VERDICT " + json.dumps(verdict))
    ex.shutdown(); rclpy.shutdown()


if __name__ == "__main__":
    main()
