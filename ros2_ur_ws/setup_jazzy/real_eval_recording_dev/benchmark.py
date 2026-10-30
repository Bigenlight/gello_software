"""Synthetic wall-clock recording throughput check; no ROS or robot."""
import json
import tempfile
import time
import cv2
import numpy as np
from recording import EpisodeWriter


with tempfile.TemporaryDirectory() as root:
    rng = np.random.default_rng(42)
    jpeg = cv2.imencode(".jpg", rng.integers(0,256,(720,1280,3),dtype=np.uint8),
                       [cv2.IMWRITE_JPEG_QUALITY,85])[1].tobytes()
    writer = EpisodeWriter(root)
    start = time.monotonic()
    costs = []
    frames = 0
    for tick in range(600):
        target = start + tick/200
        time.sleep(max(0,target-time.monotonic()))
        t = time.perf_counter()
        writer.append("robot_state", {"position":[0]*6,"velocity":[0]*6})
        writer.append("controller_command", {"position":[0]*6})
        if tick*30//200 >= frames:
            frames += 1
            for camera in ("cam1","cam2","native_cam1","native_cam2"):
                writer.append_image(camera,jpeg)
            for stream in ("request","reply","leader_command","gripper_command"):
                writer.append(stream,{"request_id":frames,"values":[0]*7})
        costs.append((time.perf_counter()-t)*1e3)
    writer.stop()
    writer.wait(30)
    print(json.dumps({"duration_s":time.monotonic()-start,"jpeg_bytes":len(jpeg),
                      "frames_per_camera":frames,"dropped":writer.dropped,
                      "enqueue_batch_ms_p50_p95_p99":np.percentile(costs,[50,95,99]).tolist(),
                      "counts":writer.counts},indent=2))
    if writer.dropped:
        raise SystemExit("FAILED: recording overflow")
