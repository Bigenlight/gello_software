"""Render ACT input views and diagnostics off the robot control path."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import cv2
import h5py
import numpy as np
from recording import atomic_json
from frame_io import RecordingFrames, recording_path


def render(path, lock_owned=False):
    path = Path(path)
    lock = path / ".tag.lock"
    if not lock_owned:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    writers = []
    try:
        atomic_json(path / "render_status.json", {"status": "rendering"})
        meta = json.loads((path / "episode.json").read_text())
        with h5py.File(recording_path(path), "r") as f, RecordingFrames(f, path) as frames:
            if not all(k in f for k in ("cam1", "cam2")):
                atomic_json(path / "render_status.json", {"status": "no_images"})
                return
            times = np.asarray(f["cam1/timestamp_ns"])
            times2 = np.asarray(f["cam2/timestamp_ns"])
            if len(times) != len(times2) or not np.array_equal(times, times2):
                raise ValueError("ACT camera streams do not match; recording is incomplete")
            fps, w, h = 30, 640, 360
            names = ("cam1", "cam2", "diagnostic")
            for name, size in zip(names, ((w, h), (w, h), (2*w, h+160))):
                writer = cv2.VideoWriter(str(path / (name + ".partial.mp4")),
                                        cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError("could not open MP4 encoder")
                writers.append(writer)
            start = meta["start_ns"]
            end = meta["stop_ns"]
            count = max(1, int(np.ceil((end-start)/1e9*fps)))
            if count > 30 * 3600:
                raise ValueError("episode exceeds one-hour render bound")
            rows = lambda key: ([json.loads(bytes(x)) for x in f[key + "/payload"]]
                                 if key in f else [])
            commands = rows("leader_command")
            replies = rows("reply")
            command_by_id = {v["request_id"]: v for v in commands}
            requests = rows("request")
            mapping = []
            cache = {}
            for frame_idx in range(count):
                stamp = start + int(frame_idx / fps * 1e9)
                idx = int(np.searchsorted(times, stamp, side="right") - 1)
                panes = []
                for camera in ("cam1", "cam2"):
                    if idx < 0:
                        pane = np.zeros((h, w, 3), np.uint8)
                    else:
                        if (camera, idx) not in cache:
                            image = frames.bgr(camera, int(idx))
                            if image is None:
                                raise ValueError(f"invalid JPEG at {camera}/{idx}")
                            # Preserve aspect ratio; letterbox for display only.
                            scale = min(w/image.shape[1], h/image.shape[0])
                            resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
                            pane = np.zeros((h, w, 3), np.uint8)
                            y, x = (h-resized.shape[0])//2, (w-resized.shape[1])//2
                            pane[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
                            cache[(camera, idx)] = pane
                        pane = cache[(camera, idx)]
                    panes.append(pane)
                cache = {k:v for k,v in cache.items() if k[1] == idx}
                writers[0].write(panes[0]); writers[1].write(panes[1])
                canvas = np.zeros((h+160, 2*w, 3), np.uint8)
                canvas[:h] = np.hstack(panes)
                request = requests[idx] if 0 <= idx < len(requests) else {}
                command = command_by_id.get(request.get("request_id"), {})
                age = (stamp-times[idx])/1e6 if idx >= 0 else None
                lines = [f"{meta['episode_id']}  t={frame_idx/fps:.2f}s  ACT image age_ms={age}",
                         "input state: " + str(request.get("control", {}).get("state", [])),
                         "policy action: " + str(command.get("policy_action", [])),
                         "clamp delta: " + str(command.get("clamp_delta", []))]
                for n, line in enumerate(lines):
                    cv2.putText(canvas, line[:170], (12, h+25+n*32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.49, (255,255,255), 1, cv2.LINE_AA)
                writers[2].write(canvas)
                mapping.append({"video_frame": frame_idx, "monotonic_ns": stamp,
                                "act_image_index": idx, "request_id": request.get("request_id")})
            for writer in writers:
                writer.release()
            writers.clear()
            for name in names:
                os.replace(path / (name + ".partial.mp4"), path / (name + ".mp4"))
            atomic_json(path / "video_frames.json", mapping)
            latencies = [v["duration_ns"]/1e6 for v in replies if "duration_ns" in v]
            deltas = [v["clamp_delta"] for v in commands if v.get("clamp_delta") is not None]
            atomic_json(path / "summary.json", {
                "episode_id": meta["episode_id"], "duration_s": (end-start)/1e9,
                "recording_status": meta["status"], "dropped": meta["dropped"],
                "termination_reason": meta["termination_reason"], "n_commands": len(commands),
                "n_clamped_commands": sum(any(abs(x)>1e-8 for x in d) for d in deltas),
                "roundtrip_ms_p50_p95_p99": np.percentile(latencies,[50,95,99]).tolist() if latencies else [],
                "video_kind": "ACT input replay, constant 30fps with timestamp mapping",
                "frame_storage": frames.provenance()})
        # Separate CPU process after ordinary videos are durable. Optional SVF
        # failure never prevents camera videos or episode tagging from completing.
        md = meta.get("metadata", {})
        if md.get("method") == "svf" and md.get("svf_diagnostic", False):
            try:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
                with (path / "svf_diagnostic.log").open("w") as log:
                    result = subprocess.run([
                        md["diagnostic_python"], str(Path(__file__).with_name("svf_diagnostic.py")),
                        str(path), "--lock-owned"], env=env, stdout=log, stderr=log, timeout=1800)
                if result.returncode:
                    raise RuntimeError(f"SVF renderer exited {result.returncode}; see svf_diagnostic.log")
            except Exception as exc:
                atomic_json(path / "svf_diagnostic_status.json", {"status": "failed", "error": str(exc)})
        atomic_json(path / "render_status.json", {"status": "complete"})
    except Exception as exc:
        atomic_json(path / "render_status.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        for writer in writers:
            writer.release()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("episode")
    p.add_argument("--lock-owned", action="store_true")
    a = p.parse_args()
    render(a.episode, a.lock_owned)
