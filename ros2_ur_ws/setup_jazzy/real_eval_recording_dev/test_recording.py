import json
import tempfile
import unittest
from pathlib import Path
import h5py
from recording import EpisodeWriter, tag_episode


class RecordingTests(unittest.TestCase):
    def test_boundary_and_tag_history(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EpisodeWriter(root, start_ns=10)
            self.assertFalse(writer.append("state", {"q": [1]}, 9))
            self.assertTrue(writer.append("state", {"q": [2]}, 11))
            self.assertTrue(writer.append_image("cam1", b"exact-jpeg", 12))
            writer.stop(timestamp_ns=20)
            writer.stop("fault", 30)
            self.assertFalse(writer.append("state", {}, 21))
            path = writer.wait(5)
            with h5py.File(path / "recording.h5") as f:
                self.assertEqual(list(f["state/timestamp_ns"]), [11])
                self.assertEqual(bytes(f["cam1/payload"][0]), b"exact-jpeg")
            meta = json.loads((path / "episode.json").read_text())
            self.assertEqual(meta["termination_reason"], "operator_hold")
            path = tag_episode(path, "success")
            path = tag_episode(path, "failure", "corrected after video review")
            data = json.loads((path / "outcome.json").read_text())
            self.assertEqual(len(data["history"]), 2)
            self.assertTrue(path.name.endswith("__failure"))

    def test_zero_action_and_active_tag_refusal(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EpisodeWriter(root)
            with self.assertRaises(ValueError):
                tag_episode(writer.path, "success")
            writer.stop("fault")
            path = writer.wait(5)
            self.assertTrue((path / "recording.h5").exists())
            self.assertEqual(json.loads((path / "episode.json").read_text())["counts"], {})

    def test_identity_and_collision(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EpisodeWriter(root)
            writer.stop()
            path = writer.wait(5)
            target = path.with_name(path.name.rsplit('__', 1)[0] + "__success")
            target.mkdir()
            with self.assertRaises(FileExistsError):
                tag_episode(path, "success")
            self.assertEqual(json.loads((path / "outcome.json").read_text())["outcome"], "unlabeled")


if __name__ == "__main__":
    unittest.main()
