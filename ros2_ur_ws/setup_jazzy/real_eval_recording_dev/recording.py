"""Episode journal and bounded asynchronous HDF5 writer; no ROS dependency."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def episode_stem(wall_ns, identity):
    date = datetime.fromtimestamp(wall_ns//1_000_000_000, ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
    return f'ep_{date}.{(wall_ns%1_000_000_000)//1000:06d}_KST_{identity}'


class EpisodeWriter:
    """One writer per file. Callbacks enqueue immutable snapshots, never do IO.

    Stop freezes the accepted interval immediately; wait() is for a background
    finalizer, never the control callback. Overflow invalidates the recording.
    """

    def __init__(self, root, metadata=None, capacity=256, start_ns=None):
        self.id = str(uuid.uuid4())
        self.start_wall_ns = time.time_ns()
        # Shared only when explicitly enabled by the registered policy launcher.
        root = os.environ.get('REAL_EVAL_EPISODE_ROOT') or root
        self.path = Path(root) / (episode_stem(self.start_wall_ns, self.id) + "__unlabeled")
        self.path.mkdir(parents=True, exist_ok=False)
        self.start_ns = time.monotonic_ns() if start_ns is None else start_ns
        self.stop_ns = None
        self.reason = None
        self.error = None
        self.dropped = 0
        self.counts = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._done = threading.Event()
        self.metadata = dict(metadata or {})
        if os.environ.get('REAL_EVAL_EPISODE_ROOT'):
            self.metadata['recording_run_dir'] = os.environ['REAL_EVAL_RUN_DIR']
        self._write_status("recording")
        atomic_json(self.path / "outcome.json", {
            "episode_id": self.id, "outcome": "unlabeled", "history": []})
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def _write_status(self, status):
        atomic_json(self.path / "episode.json", {
            "schema": "real-eval-episode/v2", "episode_id": self.id,
            "status": status, "clock": "host_monotonic_ns",
            "accepted_start": bool(self.metadata.get("accepted_start", False)),
            "start_ns": self.start_ns, "stop_ns": self.stop_ns,
            "start_wall_ns": self.start_wall_ns, "display_timezone": "Asia/Seoul",
            "termination_reason": self.reason, "dropped": self.dropped,
            "error": self.error, "counts": self.counts,
            "metadata": self.metadata})

    def append(self, stream, values, timestamp_ns=None):
        if not stream or not stream.replace("_", "").isalnum():
            raise ValueError("stream must be alphanumeric/underscore")
        stamp = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
        # Serialize before enqueue: caller may reuse numpy buffers immediately.
        payload = json.dumps(values, allow_nan=False).encode()
        with self._lock:
            if self._stop.is_set() or self.error or stamp < self.start_ns:
                return False
            try:
                self._queue.put_nowait((stream, stamp, payload))
            except queue.Full:
                self.dropped += 1
                return False
        return True

    def append_image(self, stream, jpeg, timestamp_ns=None):
        # Image streams retain exact supplied payload; no decoding in callback.
        if stream not in ("cam1", "cam2", "native_cam1", "native_cam2"):
            raise ValueError("unknown camera stream")
        stamp = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
        with self._lock:
            if self._stop.is_set() or self.error or stamp < self.start_ns:
                return False
            try:
                self._queue.put_nowait((stream, stamp, bytes(jpeg)))
            except queue.Full:
                self.dropped += 1
                return False
        return True

    def stop(self, reason="operator_hold", timestamp_ns=None):
        with self._lock:
            if self._stop.is_set():
                return
            self.stop_ns = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
            self.reason = reason
            self._stop.set()

    def wait(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError("episode writer still finalizing")
        if self.error:
            raise RuntimeError(self.error)
        return self.path

    def _run(self):
        try:
            with h5py.File(self.path / "recording.partial.h5", "w") as f:
                f.attrs["episode_id"] = self.id
                last_flush = time.monotonic()
                while not (self._stop.is_set() and self._queue.empty()):
                    try:
                        stream, stamp, payload = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        f.flush()
                        continue
                    if self.stop_ns is not None and stamp > self.stop_ns:
                        continue
                    if stream not in f:
                        group = f.create_group(stream)
                        group.create_dataset("timestamp_ns", (0,), maxshape=(None,), dtype="i8")
                        group.create_dataset("payload", (0,), maxshape=(None,),
                                             dtype=h5py.vlen_dtype(np.dtype("uint8")))
                        group.attrs["encoding"] = "jpeg" if stream in ("cam1", "cam2", "native_cam1", "native_cam2") else "json"
                    group = f[stream]
                    index = len(group["timestamp_ns"])
                    for name in ("timestamp_ns", "payload"):
                        group[name].resize(index + 1, axis=0)
                    group["timestamp_ns"][index] = stamp
                    group["payload"][index] = np.frombuffer(payload, dtype=np.uint8)
                    self.counts[stream] = index + 1
                    if stream == "lifecycle" and json.loads(payload).get("event") == "accepted_start":
                        self._write_status("recording")
                    if time.monotonic() - last_flush > 1:
                        f.flush()
                        self._write_status("recording")
                        last_flush = time.monotonic()
                f.flush()
            os.replace(self.path / "recording.partial.h5", self.path / "recording.h5")
            if self.metadata.get('accepted_start'):
                target = self.path.with_name(episode_stem(self.start_wall_ns, self.id) + '__unlabeled')
                os.rename(self.path, target)
                self.path = target
            self._write_status("invalid" if self.dropped or self.error else "ready")
            if self.metadata.get("render", False) and self.counts.get("cam1"):
                # Acquire the same filesystem lock used by tags before spawning.
                # Rendering is out-of-process and never blocks the control loop.
                lock = self.path / ".tag.lock"
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
                try:
                    with (self.path / "render.log").open("w") as log:
                        subprocess.Popen([sys.executable, str(Path(__file__).with_name("render.py")),
                                          str(self.path), "--lock-owned"], stdout=log, stderr=log,
                                         start_new_session=True)
                except Exception:
                    lock.unlink(missing_ok=True)
                    raise
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                self._write_status("partial")
            except OSError:
                pass
        finally:
            self._done.set()


def tag_episode(path, outcome, note="", labeler="operator"):
    """Explicit episode path, locked rename, durable label history.

    Directory suffix is cosmetic. outcome.json is authoritative after a crash.
    A lock collision is reported rather than guessed or stolen.
    """
    outcome = 'partial_success' if outcome == 'success_0.5' else outcome
    if outcome not in ("success", "partial_success", "failure", "unknown", "unlabeled"):
        raise ValueError("invalid outcome")
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError("episode must be a real directory")
    lock = path / ".tag.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    current = path
    try:
        meta = json.loads((path / "episode.json").read_text())
        if meta["status"] not in ("ready", "invalid", "partial"):
            raise ValueError("episode is not finalized")
        identity = str(uuid.UUID(meta["episode_id"]))
        # <stem>[__<scene>]__<outcome>: the optional scene segment (placement_metadata.scene_tag,
        # e.g. 'cube-ep59-take59') is written by the GUI and must survive relabelling.
        parts = path.name.split('__')
        stem, scene = parts[0], parts[1:-1]
        if not stem.startswith('ep_') or not (stem == 'ep_'+identity or stem.endswith('_'+identity)):
            raise ValueError("episode identity/path mismatch")
        target = path.with_name('__'.join([stem] + scene + [outcome]))
        if target != path and target.exists():
            raise FileExistsError(target)
        data = json.loads((path / "outcome.json").read_text())
        score = {'success': 1.0, 'partial_success': 0.5, 'failure': 0.0}.get(outcome)
        data["history"].append({"previous": data["outcome"], "outcome": outcome, "success_score": score,
                                "time_ns": time.time_ns(), "labeler": labeler, "note": note})
        data["outcome"] = outcome
        data["success_score"] = score
        atomic_json(path / "outcome.json", data)
        if target != path:
            os.rename(path, target)
            current = target
        return current
    finally:
        os.close(fd)
        (current / ".tag.lock").unlink()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Label a finalized eval episode")
    parser.add_argument("episode")
    parser.add_argument("outcome", choices=["success", "partial_success", "success_0.5", "failure", "unknown", "unlabeled"])
    parser.add_argument("--note", default="")
    parser.add_argument("--labeler", default="operator")
    args = parser.parse_args()
    print(tag_episode(args.episode, args.outcome, args.note, args.labeler))
