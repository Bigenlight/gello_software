"""Isolated ROS-domain smoke. No camera, server, driver, or robot connection.

Invoke with ROS_DOMAIN_ID=231, ROS_LOCALHOST_ONLY=1 and the Jazzy environment.
All inference messages are intercepted by an in-memory socket.
"""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
import rclpy
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
import h5py
import policy_leader_node as module
from leader_recording import instrumented_leader


class FakeSocket:
    def __init__(self):
        self.requests = []
    def send_multipart(self, frames):
        self.requests.append(frames)
    def recv_multipart(self):
        req = json.loads(self.requests[-1][0])
        if req["cmd"] == "reset":
            return [b'{"ok":true,"seed":123,"reset_counter":1}']
        state = req["state"]
        # Force clamp intervention while leaving the base safety implementation intact.
        action = list(state); action[0] += 2
        return [json.dumps({"ok":True,"action":action}).encode()]
    def close(self, *args):
        pass


class RosIntegration(unittest.TestCase):
    def test_real_class_start_clamp_hold(self):
        self.assertEqual(os.environ.get("ROS_DOMAIN_ID"), "231")
        with tempfile.TemporaryDirectory() as root:
            rclpy.init(args=["--ros-args", "-p", "act_port:=59997"])
            node = None
            try:
                cls = instrumented_leader(module.PolicyLeaderNode, module.EXECUTE, root)
                node = cls()
                node._sock.close(0)
                node._sock = FakeSocket()
                node._begin_episode_finalize = lambda: False
                node._live_q = list(node._start_pose)
                node._grip_pos = 0.0
                node._cam1_jpeg = b"fake1"; node._cam2_jpeg = b"fake2"
                node._live_q_t = node._grip_pos_t = node._cam1_t = node._cam2_t = time.monotonic()
                ok, message = node._try_start_execution()
                self.assertTrue(ok, message)
                node._tick_execute()
                response = node._srv_hold(None, Trigger.Response())
                self.assertTrue(response.success)
                path = node._eval_finished[-1].wait(5)
                with h5py.File(path/"recording.h5") as f:
                    value = json.loads(bytes(f["leader_command/payload"][0]))
                    self.assertNotEqual(value["clamp_delta"][0],0)
                    self.assertLessEqual(abs(value["position"][0]-node._live_q[0]),node._max_dev+1e-6)
                    self.assertEqual(len(f["request/payload"]),1)
                meta = json.loads((path/"episode.json").read_text())
                self.assertEqual(meta["metadata"]["reset_reply"]["seed"],123)
            finally:
                if node is not None:
                    node.destroy_node()
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
