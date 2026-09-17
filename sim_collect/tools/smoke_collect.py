"""Record two fake-leader takes and verify normal/shutdown finalization.

Run from the checkout with the dedicated simulation Python:
    python -m sim_collect.tools.smoke_collect --output /tmp/gello-sim-smoke
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np

from sim_collect import ipc


def verify_take(path: Path, shutdown: bool) -> dict:
    counts = {}
    with h5py.File(path / "vectors.h5", "r") as vectors:
        metadata = json.loads(vectors.attrs["sim_meta"])
        assert metadata["stopped_at"] and metadata["duration_s"] > 0
        assert metadata["control_mode"] == "eef" and metadata["scene_meta"]["leader_fake"]
        assert not metadata["scene_meta"]["floor"]["texture_used"].startswith("builtin:")
        if shutdown:
            assert metadata["stopped_by"] == "capture shutdown"
        assert "sim_scene" in vectors
        for table in ("ur_joint_states", "synchronized", "sim_control", "sim_mj_state"):
            timestamps = vectors[table]["t_rel_s"][:]
            assert len(timestamps) > 0 and np.isfinite(timestamps).all(), table
            assert (np.diff(timestamps) >= 0).all(), table
            counts[table] = len(timestamps)
        for camera in ("cam1", "cam2"):
            expected = len(vectors[f"{camera}_frames"]["t_rel_s"])
            video = cv2.VideoCapture(str(path / f"{camera}.mp4"))
            decoded = 0
            try:
                while True:
                    ok, frame = video.read()
                    if not ok:
                        break
                    assert frame.shape == (720, 1280, 3)
                    assert frame.std() > 1.0
                    decoded += 1
            finally:
                video.release()
            assert decoded == expected and decoded > 0, (camera, decoded, expected)
            counts[camera] = decoded
    with h5py.File(path / "depth.h5", "r") as depths:
        for camera in ("cam1", "cam2"):
            encoded = depths[camera]["png"]
            assert len(encoded) > 0
            for index in (0, len(encoded) - 1):
                depth = cv2.imdecode(np.asarray(encoded[index], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                assert depth is not None and depth.dtype == np.uint16
                assert depth.shape == (480, 848) and depth.max() > 0
            counts[f"{camera}_depth"] = len(encoded)
    return {"path": str(path), "counts": counts,
            "achieved_fps": metadata["achieved_fps_take"], "problems": metadata["problems"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 2:
        parser.error("--seconds must be finite and at least 2")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for port in (6701, 6702, 6711, 6712):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    root = Path(__file__).resolve().parents[2]
    os.environ["SIM_COLLECT_IPC"] = "tcp"
    env = dict(os.environ, SIM_COLLECT_PY=sys.executable,
               MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), LOG_DIR=str(output / "logs"))
    sim = ipc.Client("sim_rep", timeout_ms=1000)
    capture = ipc.Client("capture_rep", timeout_ms=2000)
    with (output / "launcher.log").open("w") as log:
        proc = subprocess.Popen(
            ["bash", str(root / "sim_collect/run_sim_collect.sh"), "--headless", "--fake-leader",
             "--control-mode", "eef", "--depth", "--root", str(output / "takes")],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        takes = []
        try:
            deadline = time.monotonic() + 90
            status = {}
            while time.monotonic() < deadline:
                assert proc.poll() is None, f"launcher exited; see {output / 'launcher.log'}"
                status = capture.call("get_status")
                renders = status.get("render", {})
                if status.get("scene_ready") and status.get("sim_alive") and all(
                        renders.get(camera, {}).get("n", 0) >= 5 for camera in ("cam1", "cam2")):
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError(f"capture not ready: {status}")
            state = sim.call("get_status")
            assert state["leader_fake"] and not state["viewer"] and state["control_mode"] == "eef"
            assert sim.call("engage")["ok"]
            assert sim.call("get_status")["engaged"]
            for shutdown in (False, True):
                started = capture.call("start_take", note="migration smoke: fake GELLO, EEF, RGB/depth")
                assert started["ok"], started
                takes.append((Path(started["take_dir"]), shutdown))
                time.sleep(args.seconds)
                if shutdown:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=40)
                    assert proc.returncode == 143, proc.returncode
                else:
                    stopped = capture.call("stop_take")
                    assert stopped["ok"], stopped
                    assert sim.call("reclutch")["ok"]
                    assert sim.call("disengage")["ok"]
                    assert not sim.call("get_status")["engaged"]
            report = [verify_take(path, shutdown) for path, shutdown in takes]
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            sim.close()
            capture.close()
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)


if __name__ == "__main__":
    main()
