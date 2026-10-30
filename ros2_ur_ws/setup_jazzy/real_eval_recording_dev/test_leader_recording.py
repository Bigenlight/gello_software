import json
import tempfile
import unittest
from types import SimpleNamespace
import h5py
from leader_recording import instrumented_leader


class Base:
    def __init__(self):
        self._state = "HOLD"
        self.allowed = True
        self.sent = []
    def _try_start_execution(self):
        if not self.allowed:
            return False, "refused"
        self._state = "EXECUTE"
        return True, "EXECUTE"
    def _srv_hold(self, request, response):
        self._state = "HOLD"
        return response
    def _enter_fault(self, reason):
        self._state = "FAULT"
    def _zmq_roundtrip(self, frames):
        self.sent.append(frames)
        return [b'{"ok":true,"action":[1,2,3,4,5,6,0]}']
    def _publish_arm(self, positions):
        self.arm = positions
    def _publish_gripper(self, value):
        self.grip = value
    def _on_joint_states(self, msg):
        pass
    def _on_fpc_commands(self, msg):
        pass
    def _on_timer(self):
        pass
    def get_logger(self):
        return SimpleNamespace(error=lambda message: None, info=lambda message: None)
    def destroy_node(self):
        pass


class IntegrationTests(unittest.TestCase):
    def test_start_act_hold_two_episodes_wire_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            node = instrumented_leader(Base, "EXECUTE", root)()
            frames = [b'{"state":[0,0,0,0,0,0,0]}', b"jpeg1", b"jpeg2"]
            for _ in range(2):
                self.assertTrue(node._try_start_execution()[0])
                self.assertFalse(node._try_start_execution()[0])
                node._zmq_roundtrip(frames)
                node._publish_arm([1, 2, 3, 4, 5, 5.5])
                node._srv_hold(None, SimpleNamespace())
                path = node._eval_finished[-1].wait(5)
                with h5py.File(path / "recording.h5") as f:
                    self.assertEqual(len(f["request/payload"]), 1)
                    output = json.loads(bytes(f["leader_command/payload"][0]))
                    self.assertEqual(output["position"][-1], 5.5)
                    self.assertEqual(bytes(f["cam1/payload"][0]), b"jpeg1")
            self.assertEqual(node.sent, [frames, frames])
            self.assertNotEqual(node._eval_finished[0].id, node._eval_finished[1].id)

    def test_rejected_and_fault(self):
        with tempfile.TemporaryDirectory() as root:
            node = instrumented_leader(Base, "EXECUTE", root)()
            node.allowed = False
            self.assertFalse(node._try_start_execution()[0])
            path = node._eval_finished[-1].wait(5)
            meta = json.loads((path / "episode.json").read_text())
            self.assertEqual(meta["termination_reason"], "start_rejected")
            self.assertEqual(meta["counts"], {})
            node.allowed = True
            self.assertTrue(node._try_start_execution()[0])
            node._enter_fault("stale camera")
            path = node._eval_finished[-1].wait(5)
            self.assertEqual(json.loads((path / "episode.json").read_text())["termination_reason"], "fault")


if __name__ == "__main__":
    unittest.main()
