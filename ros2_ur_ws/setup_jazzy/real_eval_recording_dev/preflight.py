"""Non-actuating recorder dependency and output-directory checks."""
import argparse
import shutil
import tempfile
from pathlib import Path
import cv2
import h5py
import numpy as np


def check(root):
    parent = Path(root).absolute()
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 2 * 1024**3:
        raise RuntimeError("recording requires at least 2 GiB free space")
    with tempfile.TemporaryDirectory(prefix="eval-preflight-", dir=parent) as tmp:
        with h5py.File(Path(tmp) / "test.h5", "w") as f:
            f["test"] = [1]
        video = cv2.VideoWriter(str(Path(tmp) / "test.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
        if not video.isOpened():
            raise RuntimeError("OpenCV MP4 encoder unavailable")
        video.write(np.zeros((64, 64, 3), np.uint8))
        video.release()
        cap = cv2.VideoCapture(str(Path(tmp) / "test.mp4"))
        ok, _ = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("MP4 verification failed")
    print(f"Recorder preflight passed: {root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    check(p.parse_args().root)
