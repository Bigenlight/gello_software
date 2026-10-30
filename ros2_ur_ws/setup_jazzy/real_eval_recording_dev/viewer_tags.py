"""OpenCV panel for labelling the visibly selected completed eval episode."""
import json
import threading
import time
from pathlib import Path
import cv2
import numpy as np
from recording import tag_episode

HEIGHT = 145


def latest_episode(root):
    candidates = []
    for file in Path(root).glob("run_*/episodes/ep_*/episode.json"):
        try:
            meta = json.loads(file.read_text())
            if not meta.get("accepted_start"):
                continue
            candidates.append((meta["start_ns"], file.parent, meta))
        except (OSError, ValueError, KeyError):
            continue
    return max(candidates, default=None, key=lambda x: x[0])


class TagPanel:
    def __init__(self, root):
        self.root = Path(root)
        self.last_check = 0
        self.selected = None
        self.message = "Waiting for a recorded episode"
        self.buttons = []
        self.busy = False

    def draw(self, width, offset):
        if not self.busy and time.monotonic()-self.last_check > 0.5:
            self.selected = latest_episode(self.root)
            self.last_check = time.monotonic()
        strip = np.full((HEIGHT, width, 3), 28, np.uint8)
        self.buttons = []
        ready = False
        line = self.message
        if self.selected:
            _, path, meta = self.selected
            locked = (path / ".tag.lock").exists()
            ready = meta["status"] in ("ready", "invalid", "partial") and not locked and not self.busy
            line = f"EVAL {path.name} | {meta['status']}" + (" | rendering/saving" if locked else "")
        cv2.putText(strip, line[:190], (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235,235,235), 1)
        labels = (("SUCCESS", "success", (50,130,45)), ("FAILURE", "failure", (45,55,160)),
                  ("UNKNOWN", "unknown", (100,100,100)))
        col = (width-80)//3
        for i, (label, outcome, color) in enumerate(labels):
            x1, x2 = 20+i*(col+20), 20+i*(col+20)+col
            cv2.rectangle(strip, (x1,45), (x2,90), color if ready else (60,60,60), -1)
            cv2.putText(strip, label, (x1+15,76), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255,255,255) if ready else (125,125,125), 2)
            if ready:
                self.buttons.append(((x1,offset+45,x2,offset+90), self.selected[1], outcome))
        cv2.putText(strip, self.message[:180], (20,125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,210,230), 1)
        return strip

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.busy:
            return
        for (x1,y1,x2,y2), path, outcome in self.buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Bind the path displayed in this frame. Never relabel a new
                # "latest" episode that appeared after the operator saw it.
                self.busy = True
                self.message = "Saving label..."
                threading.Thread(target=self._tag, args=(path,outcome), daemon=True).start()
                return

    def _tag(self, path, outcome):
        try:
            result = tag_episode(path, outcome, labeler="camera_viewer")
            self.message = f"Saved {outcome}: {result.name}"
            self.last_check = 0
        except Exception as exc:
            self.message = f"Tag failed: {exc}"
        finally:
            self.busy = False
