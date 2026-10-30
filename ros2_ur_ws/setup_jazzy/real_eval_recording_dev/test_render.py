import json
import tempfile
import time
import unittest
from pathlib import Path
import cv2
import numpy as np
from recording import EpisodeWriter, tag_episode
from render import render
from viewer_tags import TagPanel, latest_episode


class RenderTests(unittest.TestCase):
    def test_automatic_render_process(self):
        with tempfile.TemporaryDirectory() as root:
            start = time.monotonic_ns()
            writer = EpisodeWriter(root, {"accepted_start": True, "render": True}, start_ns=start)
            jpeg = cv2.imencode(".jpg", np.zeros((48,64,3), np.uint8))[1].tobytes()
            for cam in ("cam1", "cam2"):
                writer.append_image(cam, jpeg, start+1)
            writer.stop(timestamp_ns=start+100000000)
            path = writer.wait(5)
            deadline = time.monotonic()+20
            while time.monotonic()<deadline and (path/".tag.lock").exists():
                time.sleep(.05)
            self.assertFalse((path/".tag.lock").exists())
            self.assertTrue((path/"diagnostic.mp4").exists(), (path/"render.log").read_text())
            self.assertTrue(tag_episode(path,"success").name.endswith("__success"))

    def test_video_decode_and_gui_tag(self):
        with tempfile.TemporaryDirectory() as root:
            start = time.monotonic_ns()
            writer = EpisodeWriter(Path(root)/"run_test"/"episodes", {"accepted_start": True}, start_ns=start)
            jpeg = cv2.imencode(".jpg", np.full((48,64,3), 120, np.uint8))[1].tobytes()
            writer.append("request", {"request_id":1,"control":{"state":[0]*7}},start+1)
            writer.append_image("cam1",jpeg,start+1)
            writer.append_image("cam2",jpeg,start+1)
            writer.append("reply", {"request_id":1,"duration_ns":1000000},start+2)
            writer.append("leader_command", {"request_id":1,"clamp_delta":[0,0,.1,0,0,0],
                                               "policy_action":[0]*7},start+3)
            writer.stop(timestamp_ns=start+100000000)
            path = writer.wait(5)
            render(path)
            for name in ("cam1", "cam2", "diagnostic"):
                cap = cv2.VideoCapture(str(path/(name+".mp4")))
                self.assertTrue(cap.read()[0]); cap.release()
            self.assertEqual(json.loads((path/"summary.json").read_text())["n_clamped_commands"],1)
            panel = TagPanel(root)
            self.assertEqual(panel.draw(1280,800).shape,(145,1280,3))
            self.assertEqual(len(panel.buttons),3)
            panel._tag(path,"success")
            selected = latest_episode(root)
            self.assertTrue(selected[1].name.endswith("__success"))

    def test_render_lock_blocks_tag(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EpisodeWriter(root)
            writer.stop(); path = writer.wait(5)
            (path/".tag.lock").touch()
            with self.assertRaises(FileExistsError):
                tag_episode(path,"failure")


if __name__ == "__main__":
    unittest.main()
