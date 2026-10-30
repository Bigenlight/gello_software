"""native_cam1/2 policy (2026-09-23): 15 Hz clock -> not recorded (cam1/cam2 hold the same frames); override "15" -> every other frame; 30 Hz unchanged."""
import os
import random
import time
import tempfile
import unittest
from types import SimpleNamespace
import h5py
from leader_recording import instrumented_leader, native_camera_period_ns
from test_leader_recording import Base


def make_base(rate):
    class CamBase(Base):
        def __init__(self):
            super().__init__()
            self._publish_rate_hz = rate
        def _on_cam1(self, msg):
            self._cam1_ros_stamp_ns = msg.stamp
        def _on_cam2(self, msg):
            self._cam2_ros_stamp_ns = msg.stamp
    return CamBase


def run_episode(rate, n=300, jitter_ms=6, seed=0):
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as root:
        node = instrumented_leader(make_base(rate), "EXECUTE", root)()
        assert node._try_start_execution()[0]
        t0 = 10**18
        for i in range(n):
            stamp = t0 + int(i * 1e9 / 30) + int(rng.uniform(-jitter_ms, jitter_ms) * 1e6)
            node._on_cam1(SimpleNamespace(data=b"j%d" % i, stamp=stamp))
            node._on_cam2(SimpleNamespace(data=b"k%d" % i, stamp=stamp + 3_000_000))
            time.sleep(0.003)  # stay below the writer's 256-slot queue (real cameras are 33 ms apart)
        node._srv_hold(None, SimpleNamespace())
        path = node._eval_finished[-1].wait(5)
        assert node._eval_finished[-1].dropped == 0
        with h5py.File(path / "recording.h5") as f:
            if "native_cam1" not in f:
                return 0, 0, []
            stamps = [int(x) for x in [__import__("json").loads(bytes(p))["source_stamp_ns"] for p in f["native_cam1_meta/payload"]]]
            return len(f["native_cam1/payload"]), len(f["native_cam2/payload"]), stamps


class NativeDecimation(unittest.TestCase):
    def test_15hz_default_drops_native(self):
        n1, n2, _ = run_episode(15.0)
        self.assertEqual((n1, n2), (0, 0))

    def test_15hz_override_keeps_every_other_frame_despite_jitter(self):
        os.environ["REAL_EVAL_NATIVE_CAM_HZ"] = "15"
        try:
            n1, n2, stamps = run_episode(15.0)
        finally:
            del os.environ["REAL_EVAL_NATIVE_CAM_HZ"]
        self.assertEqual((n1, n2), (150, 150))
        gaps = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
        self.assertGreaterEqual(min(gaps), 50.0)
        self.assertLessEqual(max(gaps), 80.0)

    def test_30hz_unchanged(self):
        self.assertEqual(run_episode(30.0)[:2], (300, 300))
        self.assertEqual(native_camera_period_ns(30.0), 0)

    def test_override(self):
        os.environ["REAL_EVAL_NATIVE_CAM_HZ"] = "0"
        try:
            self.assertEqual(run_episode(15.0)[:2], (300, 300))
        finally:
            del os.environ["REAL_EVAL_NATIVE_CAM_HZ"]


if __name__ == "__main__":
    unittest.main()
