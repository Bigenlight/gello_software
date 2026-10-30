import tempfile
import time
import unittest
from unittest.mock import patch
from recording import EpisodeWriter
from test_leader_recording import Base
from leader_recording import instrumented_leader


class FailureTests(unittest.TestCase):
    def test_writer_failure_reported(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("recording.h5py.File", side_effect=OSError("disk full")):
                writer = EpisodeWriter(root)
                with self.assertRaisesRegex(RuntimeError,"disk full"):
                    writer.wait(5)
                self.assertFalse(writer.append("state",{}))

    def test_overflow_is_explicit_and_holds(self):
        with tempfile.TemporaryDirectory() as root:
            node = instrumented_leader(Base,"EXECUTE",root)()
            self.assertTrue(node._try_start_execution()[0])
            writer = node._eval_writer
            with patch.object(writer._queue,"put_nowait",side_effect=__import__('queue').Full):
                self.assertFalse(writer.append("state",{}))
            self.assertEqual(writer.dropped,1)
            node._on_timer()
            self.assertEqual(node._state,"HOLD")
            writer.wait(5)
            self.assertEqual(writer.reason,"recording_error")


if __name__ == "__main__":
    unittest.main()
