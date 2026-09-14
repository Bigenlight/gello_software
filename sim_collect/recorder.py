"""Take recorder for the sim capture process (DESIGN.md §2.2 / §4).

Wraps :class:`gello_recorder.recording_session.RecordingSession` VERBATIM (its nine
tables, ``t_rel_s`` origin, per-field precision, ``Mp4FrameWriter`` and
``DepthH5Writer``) so a sim take is structurally identical to a real one:

    take_NN_YYYYmmdd_HHMMSS/
        vectors.h5   9 real groups + sim_object_poses / sim_control / sim_leader_filtered
        cam1.mp4     1280x720 @ 30 mp4v (scene camera)
        cam2.mp4     1280x720 @ 30 mp4v (wrist camera)
        depth.h5     cam1/cam2 848x480 uint16 mm PNG (+ camera_info / extrinsics attrs)

Rates (§4.2): ``ur_joint_states`` / ``command`` / ``tcp_pose`` / ``wrench`` on every 2nd
state message (250 -> 125 Hz), ``gripper`` on every 4th (62.5 Hz), ``gello_joint_states``
whenever a new leader sample arrives (``leader_t`` changes; finite-difference ``qd`` over
``leader_t``, first row 0), ``synchronized`` on a 100 Hz wall-clock grid once BOTH
cameras have delivered a frame (so every cell is finite), ``sim_object_poses`` on a
30 Hz grid, ``sim_control`` / ``sim_leader_filtered`` with the 125 Hz group.

The sim extras are additional ``Hdf5TableWriter`` tables opened on the session's OWN
``h5py.File`` (``session._h5``) -- one handle, one file. ``vectors.h5`` gets a file-level
``sim_meta`` JSON attr at start (updated with ``task_success_at_stop`` / duration at
stop). Real files have no file attrs; none of the three consumers reads them.

Thread model: ``on_state`` from the owner's loop, ``on_frames`` from any thread (the
capture process calls it from a frame thread because ``Mp4FrameWriter`` decodes the
JPEG under a released GIL). ``start``/``stop``/``discard_last`` take the lifecycle lock
and wait for in-flight writes; h5py serialises HDF5 calls under its own global lock.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

import weakref

import gello_recorder.hdf5_writer as _h5w
import gello_recorder.recording_session as _rs
from gello_recorder.hdf5_writer import Hdf5TableWriter
from gello_recorder.recording_session import RecordingSession


class BufferedHdf5TableWriter(Hdf5TableWriter):
    """Same on-disk layout as ``Hdf5TableWriter`` (one resizable float64 dataset per
    column, ``columns`` JSON attr), but rows are buffered in memory and written in
    blocks. The real recorder resizes every column dataset per row (~44 µs/column,
    ~3 ms per 250 Hz state message across all tables), which made the sim recorder
    fall behind under CPU contention (measured 78 Hz instead of 125 Hz on
    ``ur_joint_states``). Flush happens every ``FLUSH_EVERY_S`` from ``on_state``, on
    ``SimTakeRecorder.flush()`` and before the session closes."""

    FLUSH_EVERY_S = 1.0
    _instances: "weakref.WeakSet[BufferedHdf5TableWriter]" = weakref.WeakSet()

    def __init__(self, h5file, table_name, header):
        super().__init__(h5file, table_name, header)
        self._buf: List[List[float]] = [[] for _ in self._header]
        self._wlock = threading.Lock()   # writerow (state/frame threads) vs flush_rows
        BufferedHdf5TableWriter._instances.add(self)

    def writerow(self, row: list) -> None:
        if len(row) != len(self._datasets):
            raise ValueError(f"row has {len(row)} values but table has {len(self._datasets)} columns ({self._header})")
        vals = [_h5w._coerce(v) for v in row]
        with self._wlock:
            for col, value in zip(self._buf, vals):
                col.append(value)

    def flush_rows(self) -> None:
        with self._wlock:
            n = len(self._buf[0]) if self._buf else 0
            if n == 0:
                return
            new_len = self._nrows + n
            for dset, col in zip(self._datasets, self._buf):
                dset.resize((new_len,))
                dset[self._nrows:new_len] = np.asarray(col, dtype=np.float64)
                col.clear()
            self._nrows = new_len

    @classmethod
    def flush_all(cls) -> None:
        for w in list(cls._instances):
            try:
                w.flush_rows()
            except Exception:  # dataset already closed (session closed) -> nothing to do
                pass


def _open_buffered_table(h5file, table_name: str, header: list) -> BufferedHdf5TableWriter:
    return BufferedHdf5TableWriter(h5file, table_name, header)


# RecordingSession builds its nine tables through ``open_h5_table`` looked up in its own
# module namespace; rebinding that name (in this process only) swaps in the buffered
# writer without touching the real recorder package.
_rs.open_h5_table = _open_buffered_table
open_h5_table = _open_buffered_table

from sim_collect import cameras as _cams

SIM_COLLECT_VERSION = "0.1.0"
COMPRESSED_DEPTH_HEADER = struct.pack("<iff", 0, 0.0, 0.0)   # 12 bytes, format enum 0

# eef_state string -> code stored in sim_control.eef_state_code (mapping also saved in
# sim_meta so a reader never has to import this file). Unknown strings -> -1.
EEF_STATE_CODES: Dict[str, int] = {
    "DISENGAGED": 0, "ENGAGING": 1, "ENGAGED": 2, "HOLD": 3, "SOFT_START": 4,
    "REJECTED": 5, "FAULT": 6, "PAUSED": 7,
}
_N = 6
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def compressed_depth_payload(png_bytes: bytes) -> bytes:
    """Wrap a PNG the way ``compressed_depth_image_transport`` does (12-byte
    ConfigHeader + PNG) so ``RecordingSession.write_camN_depth_frame`` accepts it."""
    return COMPRESSED_DEPTH_HEADER + bytes(png_bytes)


def _finite_list(v: Any, n: int) -> Optional[List[float]]:
    """``v`` as a list of ``n`` finite floats, or ``None`` when absent / malformed."""
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if a.shape[0] != n or not np.all(np.isfinite(a)):
        return None
    return a.tolist()


def _advance_grid(next_t: float, now: float, period: float) -> float:
    """Next slot of a fixed-period sampling grid. State messages arrive every 4 ms, so
    ``now + period`` would quantise a 10 ms grid to 12 ms (83 Hz); advancing the grid
    itself keeps the mean rate exact. A grid that fell more than one period behind
    (writer stall) re-anchors on ``now`` instead of bursting to catch up."""
    nxt = next_t + period
    if nxt <= now:
        nxt = now + period - ((now - next_t) % period)
    return nxt


def _finite_scalar(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _scrub_paths(obj: Any) -> Any:
    """Strip absolute repo/home paths from strings so ``vectors.h5`` never carries
    ``/home/...`` (make_carrot_raw_stats flags that as leakage)."""
    if isinstance(obj, str):
        if obj.startswith(_REPO_ROOT):
            return os.path.relpath(obj, _REPO_ROOT)
        return obj.replace("/home/", "~/")
    if isinstance(obj, Mapping):
        return {str(k): _scrub_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_paths(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    return obj


def git_commit(repo_root: str = _REPO_ROOT) -> str:
    """``git rev-parse HEAD`` (best-effort, never raises)."""
    try:
        import subprocess
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


class SimTakeRecorder:
    """One recorder per capture process; one :class:`RecordingSession` per take."""

    def __init__(self, *, camera_fps: float = 30.0, sample_rate_hz: float = 100.0,
                 object_pose_hz: float = 30.0, cams: Sequence[str] = _cams.CAMS):
        self.camera_fps = float(camera_fps)
        self.sample_rate_hz = float(sample_rate_hz)
        self.object_pose_hz = float(object_pose_hz)
        self.cams = tuple(cams)
        self._lock = threading.RLock()
        self._last_row_flush = time.monotonic()
        self._cv = threading.Condition(self._lock)
        self._inflight = 0
        self._session: Optional[RecordingSession] = None
        self._take_counter = 0
        self.take_dir: Optional[str] = None
        self.last_take_dir: Optional[str] = None
        self._sim_meta: Dict[str, Any] = {}
        self._reset_take_state()

    # ---- lifecycle -----------------------------------------------------------
    def _reset_take_state(self) -> None:
        self._n_state = 0
        self._last_leader_t: Optional[float] = None
        self._last_leader_q: Optional[List[float]] = None
        self._next_sync_t = 0.0
        self._next_obj_t = 0.0
        self._frame_idx: Dict[str, Optional[int]] = {c: None for c in self.cams}
        self._object_names: Optional[List[str]] = None
        self._obj_w = None
        self._ctrl_w = None
        self._leadf_w = None
        self._last_task_success: Optional[bool] = None
        self._last_msg: Optional[Mapping[str, Any]] = None
        self._last_tick: Optional[int] = None
        self._cap_times: Dict[str, List[float]] = {c: [] for c in self.cams}   # [first, last, n]
        self._fc_w = None
        self._events: List[Dict[str, Any]] = []
        self._warn_last: Dict[str, float] = {}
        self._counts: Dict[str, int] = {"state_msgs": 0, "state_dropped": 0, "frames_dropped": 0,
                                        "write_errors": 0, "missed_ticks": 0, "tick_restarts": 0}

    @property
    def recording(self) -> bool:
        return self._session is not None

    @property
    def take_index(self) -> int:
        return self._take_counter

    def start(self, root: str, note: str = "", sim_meta: Optional[Mapping[str, Any]] = None) -> str:
        """Open ``<root>/take_{NN:02d}_{YYYYmmdd_HHMMSS}/`` and return its path."""
        with self._lock:
            if self._session is not None:
                raise RuntimeError(f"already recording {self.take_dir}")
            os.makedirs(root, exist_ok=True)
            self._take_counter += 1
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            take_dir = os.path.join(root, f"take_{self._take_counter:02d}_{stamp}")
            self._reset_take_state()
            sess = RecordingSession(take_dir, camera_fps=self.camera_fps, record_depth=True)
            for i, cam in enumerate(self.cams, start=1):
                sess.set_depth_source(i, _cams.depth_source_topic(cam), False)
                sess.set_depth_camera_info(i, **_cams.camera_info_dict(cam))
                ext = _cams.extrinsics_dict(cam)
                sess.set_depth_extrinsics(i, ext["rotation"], ext["translation"])
            # Sim extras that do not depend on the object list can open now.
            self._ctrl_w = open_h5_table(
                sess._h5, "sim_control",
                ["t_rel_s", "engaged", "eef_state_code", "pos_scale", "sigma_min",
                 "gamma", "ls_scale", "task_success", "sim_t", "tick"])
            self._leadf_w = open_h5_table(
                sess._h5, "sim_leader_filtered",
                ["t_rel_s"] + [f"qf{i + 1}" for i in range(_N)])
            # When each frame was RENDERED (camN_frames.t_rel_s is the write time, like
            # the real recorder): cam 1/2, mp4 frame_idx, capture time on the session
            # clock, and the sim_t / tick of the state that was rendered.
            self._fc_w = open_h5_table(
                sess._h5, "sim_frame_capture",
                ["t_rel_s", "cam", "frame_idx", "t_capture_rel_s", "sim_t", "tick", "seq"])
            meta = dict(sim_meta or {})
            meta.setdefault("sim_collect_version", SIM_COLLECT_VERSION)
            meta.setdefault("git_commit", git_commit())
            try:
                import mujoco
                meta.setdefault("mujoco_version", mujoco.__version__)
            except Exception:  # noqa: BLE001
                pass
            meta.update({
                "take_name": os.path.basename(take_dir),
                "take_index": self._take_counter,
                "note": str(note or ""),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "camera_fps": self.camera_fps,
                "sample_rate_hz": self.sample_rate_hz,
                "eef_state_codes": dict(EEF_STATE_CODES),
                "depth_camera_info": {c: _cams.camera_info_dict(c) for c in self.cams},
                "depth_extrinsics_depth_to_color": {c: _cams.extrinsics_dict(c) for c in self.cams},
                "depth_source": {c: "mujoco offscreen depth render (uint16 mm PNG); "
                                    "camera_info/extrinsics/source_topic copied from the real D435 "
                                    "so sim and real takes share one sidecar" for c in self.cams},
                "simulated": True,
                "task_success_at_stop": None,
                "duration_s": None,
                "events": [],
            })
            self._sim_meta = _scrub_paths(meta)
            sess._h5.attrs["sim_meta"] = json.dumps(self._sim_meta)
            self._session = sess
            self._last_row_flush = time.monotonic()
            self.take_dir = take_dir
            self.last_take_dir = take_dir
            return take_dir

    def achieved_fps_take(self) -> Dict[str, Optional[float]]:
        """Whole-take frame rate per camera from the workers' capture timestamps
        (``(n-1)/(t_last-t_first)``; ``None`` with fewer than 2 frames). This -- not the
        workers' live sliding window -- is the number that says whether ``camN.mp4``
        (stamped ``camera_fps``) plays at real speed."""
        out: Dict[str, Optional[float]] = {}
        for cam, ct in self._cap_times.items():
            out[cam] = round((ct[2] - 1) / (ct[1] - ct[0]), 2) if ct and ct[2] >= 2 and ct[1] > ct[0] else None
        return out

    def problems(self) -> List[str]:
        """Human-readable list of everything that went wrong in this take so far
        (empty = clean). Surfaced in ``stop_take``, status, preview and ``sim_meta``."""
        c = self._counts
        out = []
        for cam, hz in self.achieved_fps_take().items():
            if hz is not None and self._cap_times[cam][2] >= 30 and hz < 0.97 * self.camera_fps:
                out.append(f"{cam} captured at {hz:.1f} fps but {cam}.mp4 is stamped {self.camera_fps:.0f} "
                           f"(plays {self.camera_fps / hz:.2f}x fast)")
        if c.get("write_errors"):
            out.append(f"{c['write_errors']} write errors (disk full? see log)")
        if c.get("state_dropped"):
            out.append(f"{c['state_dropped']} state messages dropped")
        if c.get("frames_dropped"):
            out.append(f"{c['frames_dropped']} frames rejected by the writers")
        if c.get("missed_ticks"):
            out.append(f"{c['missed_ticks']} sim ticks never reached the recorder")
        if c.get("tick_restarts"):
            out.append(f"sim restarted {c['tick_restarts']}x mid-take")
        return out

    def stop(self, extra_meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Finalise the take: update ``sim_meta`` (merging ``extra_meta``, e.g. the
        capture process's achieved fps), flush buffered rows, close every file, and
        return a summary whose ``ok`` is False when anything failed and whose
        ``problems`` lists every counted anomaly (a take with write errors is NOT
        reported as clean)."""
        with self._lock:
            sess = self._session
            if sess is None:
                raise RuntimeError("not recording")
            self._session = None          # new writes are refused from here on
            while self._inflight > 0:     # ... and in-flight ones drain
                self._cv.wait(timeout=5.0)
            errors: List[str] = []
            try:
                BufferedHdf5TableWriter.flush_all()   # buffered rows -> datasets before the file closes
            except Exception as e:  # noqa: BLE001
                errors.append(f"flush: {type(e).__name__}: {e}")
            self._sim_meta["task_success_at_stop"] = self._last_task_success
            self._sim_meta["duration_s"] = round(sess.t(), 3)
            self._sim_meta["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self._sim_meta["object_names"] = list(self._object_names or [])
            self._sim_meta["recorder_counts"] = dict(self._counts)
            self._sim_meta["events"] = list(self._events)
            self._sim_meta["message_counts"] = dict(getattr(sess, "_counts", {}) or {})
            self._sim_meta["achieved_fps_take"] = self.achieved_fps_take()
            if extra_meta:
                self._sim_meta.update(_scrub_paths(dict(extra_meta)))
                cap = self._sim_meta.get("capture") if isinstance(self._sim_meta.get("capture"), Mapping) else None
                if cap and "achieved_fps" in cap:
                    self._sim_meta["achieved_fps"] = cap["achieved_fps"]
            self._sim_meta["problems"] = self.problems()
            try:
                sess._h5.attrs["sim_meta"] = json.dumps(self._sim_meta)
            except Exception as e:  # noqa: BLE001
                errors.append(f"sim_meta attr: {type(e).__name__}: {e}")
            try:
                result = sess.close()
            except Exception as e:  # noqa: BLE001
                errors.append(f"close: {type(e).__name__}: {e}")
                result = {"duration_s": round(sess.t(), 2), "message_counts": {}}
            problems = self.problems() + errors
            result.update({"take_dir": self.take_dir,
                           "task_success_at_stop": self._last_task_success,
                           "recorder_counts": dict(self._counts),
                           "events": list(self._events),
                           "problems": problems,
                           "ok": not errors})
            if problems:
                print(f"[SimTakeRecorder] take {os.path.basename(self.take_dir or '')} finished with "
                      f"problems: {'; '.join(problems)}", flush=True)
            self.take_dir = None
            return result

    def discard_last(self) -> Dict[str, Any]:
        """Delete the most recently finished take directory. Refused while recording."""
        with self._lock:
            if self._session is not None:
                return {"ok": False, "msg": "stop the take before discarding it"}
            d = self.last_take_dir
            if not d or not os.path.isdir(d):
                return {"ok": False, "msg": "no take to discard"}
            shutil.rmtree(d)
            self.last_take_dir = None
            return {"ok": True, "msg": f"deleted {d}", "deleted": d}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            sess = self._session
            counts = dict(sess._counts) if sess is not None else {}
            return {
                "recording": sess is not None,
                "take_dir": self.take_dir,
                "last_take_dir": self.last_take_dir,
                "take_index": self._take_counter,
                "duration_s": round(sess.t(), 2) if sess is not None else 0.0,
                "frames": {c: counts.get(f"{c}_frames", 0) for c in self.cams},
                "depth_frames": {c: counts.get(f"{c}_depth_frames", 0) for c in self.cams},
                "rows": counts,
                "state_msgs": self._counts.get("state_msgs", 0),
                "counts": dict(self._counts),
                "problems": self.problems() if sess is not None else [],
                "events": list(self._events) if sess is not None else [],
                "task_success": self._last_task_success,
            }

    # ---- data ----------------------------------------------------------------
    def on_state(self, msg: Mapping[str, Any]) -> None:
        """One ``state`` message from the sim (DESIGN §2.1). Never raises on a bad
        message: a table whose inputs are missing/malformed just skips that row."""
        with self._lock:
            sess = self._session
            if sess is None or not isinstance(msg, Mapping):
                return
            self._inflight += 1
        try:
            self._write_state(sess, msg)
            now = time.monotonic()
            if now - self._last_row_flush >= BufferedHdf5TableWriter.FLUSH_EVERY_S:
                self._last_row_flush = now
                with self._lock:
                    BufferedHdf5TableWriter.flush_all()
        except Exception as e:  # noqa: BLE001 - one bad message must not end the take
            self._counts["state_dropped"] += 1
            if isinstance(e, (OSError, RuntimeError)):      # h5py/disk: ENOSPC, closed file ...
                self._counts["write_errors"] += 1
            self._warn("state", f"state message dropped: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._inflight -= 1
                self._cv.notify_all()

    def _write_state(self, sess: RecordingSession, msg: Mapping[str, Any]) -> None:
        self._counts["state_msgs"] += 1
        self._last_msg = msg
        n = self._n_state
        self._n_state += 1

        # tick accounting: gaps = messages the recorder never saw (PUB HWM / slow loop),
        # backwards = the sim restarted mid-take (also an event for sim_meta)
        tick = msg.get("tick")
        if isinstance(tick, (int, np.integer)):
            tick = int(tick)
            if self._last_tick is not None:
                if tick > self._last_tick + 1:
                    self._counts["missed_ticks"] += tick - self._last_tick - 1
                elif tick < self._last_tick:
                    self._counts["tick_restarts"] += 1
                    self.add_event("tick_backwards", tick_before=self._last_tick, tick_after=tick)
            self._last_tick = tick

        q = _finite_list(msg.get("q"), _N)
        qd = _finite_list(msg.get("qd"), _N)
        eff = _finite_list(msg.get("eff"), _N)
        q_cmd = _finite_list(msg.get("q_cmd"), _N)
        tcp_pos = _finite_list(msg.get("tcp_pos"), 3)
        tcp_quat = _finite_list(msg.get("tcp_quat_xyzw"), 4)
        wrench = _finite_list(msg.get("wrench"), 6)
        trigger = _finite_scalar(msg.get("trigger"))
        grip_cmd = _finite_scalar(msg.get("grip_cmd"))
        grip_pos = _finite_scalar(msg.get("grip_pos"))
        task = msg.get("task") if isinstance(msg.get("task"), Mapping) else {}
        if "success" in task:
            self._last_task_success = bool(task.get("success"))

        # -- 125 Hz group: every 2nd message -----------------------------------
        if n % 2 == 0:
            if q is not None and qd is not None and eff is not None:
                sess.write_ur(q, qd, eff)
            if q_cmd is not None:
                sess.write_cmd(q_cmd)
            if tcp_pos is not None and tcp_quat is not None:
                sess.write_tcp(tcp_pos + tcp_quat)
            if wrench is not None:
                sess.write_wrench(wrench)
            self._write_control(sess, msg)
            qf = _finite_list(msg.get("q_lead_f"), _N)
            if qf is not None and self._leadf_w is not None:
                self._leadf_w.writerow([f"{sess.t():.4f}"] + [f"{v:.6f}" for v in qf])

        # -- gripper: every 4th message (62.5 Hz >= 30 Hz) -----------------------
        if n % 4 == 0 and (trigger is not None or grip_cmd is not None or grip_pos is not None):
            sess.write_gello_grip(trigger, grip_cmd, grip_pos)
            for key, v in (("gello_grip", trigger), ("grip_cmd", grip_cmd), ("grip_pos", grip_pos)):
                if v is not None:
                    sess.bump(key)

        # -- leader: on every new leader sample ----------------------------------
        q_lead = _finite_list(msg.get("q_lead_raw"), _N)
        leader_t = _finite_scalar(msg.get("leader_t"))
        if q_lead is not None:
            is_new = (leader_t is None and n % 8 == 0) or (
                leader_t is not None and leader_t != self._last_leader_t)
            if is_new:
                if self._last_leader_q is not None and leader_t is not None \
                        and self._last_leader_t is not None and leader_t - self._last_leader_t > 1e-6:
                    dt = leader_t - self._last_leader_t
                    qd_lead = [(q_lead[i] - self._last_leader_q[i]) / dt for i in range(_N)]
                else:
                    qd_lead = _finite_list(msg.get("qd_lead"), _N) or [0.0] * _N
                sess.write_gello(q_lead, qd_lead)
                self._last_leader_q, self._last_leader_t = q_lead, leader_t

        # -- synchronized: 100 Hz grid, only once both cams have a frame ----------
        now = sess.t()
        if now >= self._next_sync_t and all(self._frame_idx[c] is not None for c in self.cams):
            self._next_sync_t = _advance_grid(self._next_sync_t, now, 1.0 / self.sample_rate_hz)
            if (q is not None and qd is not None and eff is not None and q_cmd is not None
                    and tcp_pos is not None and tcp_quat is not None and wrench is not None
                    and q_lead is not None):
                qd_lead_sync = _finite_list(msg.get("qd_lead"), _N)
                if qd_lead_sync is None:
                    qd_lead_sync = [0.0] * _N
                sess.write_sample(
                    q_lead, qd_lead_sync,
                    0.0 if trigger is None else trigger,
                    q_cmd, q, qd, eff,
                    0.0 if grip_cmd is None else grip_cmd,
                    0.0 if grip_pos is None else grip_pos,
                    wrench, tcp_pos + tcp_quat,
                    self._frame_idx[self.cams[0]], self._frame_idx[self.cams[1]])

        # -- object poses: 30 Hz grid ----------------------------------------------
        if now >= self._next_obj_t:
            self._next_obj_t = _advance_grid(self._next_obj_t, now, 1.0 / self.object_pose_hz)
            self._write_objects(sess, msg)

    def _write_control(self, sess: RecordingSession, msg: Mapping[str, Any]) -> None:
        if self._ctrl_w is None:
            return
        info = msg.get("eef_info") if isinstance(msg.get("eef_info"), Mapping) else {}
        state = str(msg.get("eef_state", info.get("state", "")))
        task = msg.get("task") if isinstance(msg.get("task"), Mapping) else {}

        def f(v: Any, p: int = 6) -> Optional[str]:
            s = _finite_scalar(v)
            return None if s is None else f"{s:.{p}f}"

        self._ctrl_w.writerow([
            f"{sess.t():.4f}",
            1 if msg.get("engaged") else 0,
            EEF_STATE_CODES.get(state, -1),
            f(msg.get("pos_scale"), 4),
            f(info.get("sigma_min")),
            f(info.get("gamma")),
            f(info.get("ls_scale")),
            1 if task.get("success") else 0,
            f(msg.get("sim_t"), 4),
            _finite_scalar(msg.get("tick")),
        ])

    def _write_objects(self, sess: RecordingSession, msg: Mapping[str, Any]) -> None:
        objects = msg.get("objects")
        if not isinstance(objects, Mapping) or not objects:
            return
        if self._object_names is None:
            self._object_names = sorted(str(k) for k in objects)
            cols = ["t_rel_s"]
            for name in self._object_names:
                cols += [f"{name}_{s}" for s in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            self._obj_w = open_h5_table(sess._h5, "sim_object_poses", cols)
        row: List[Any] = [f"{sess.t():.4f}"]
        for name in self._object_names:
            o = objects.get(name) if isinstance(objects.get(name), Mapping) else {}
            pos = _finite_list(o.get("pos"), 3)
            quat = _finite_list(o.get("quat_wxyz"), 4)
            if pos is None or quat is None:
                row += [None] * 7
            else:
                w, x, y, z = quat
                row += [f"{v:.6f}" for v in pos] + [f"{v:.6f}" for v in (x, y, z, w)]
        self._obj_w.writerow(row)

    def on_frames(self, cam: str, t_capture: float, jpeg_bytes: bytes,
                  depth_png: Optional[bytes], *, sim_t: Any = None, tick: Any = None,
                  seq: Any = None) -> Optional[int]:
        """One rendered frame pair for ``cam`` ("cam1"/"cam2"): the colour JPEG goes to
        ``camN.mp4`` via the session's writer (+ ``camN_frames`` row, stamped at WRITE
        time like the real recorder), the depth PNG to ``depth.h5`` (``stamp_s`` =
        ``t_capture``, epoch seconds), and a ``sim_frame_capture`` row records the
        capture time on the session clock plus the rendered state's ``sim_t``/``tick``.
        Returns the mp4 frame index or ``None`` when not recording / rejected."""
        with self._lock:
            sess = self._session
            if sess is None or cam not in self.cams:
                return None
            self._inflight += 1
        try:
            idx = int(self.cams.index(cam)) + 1
            writer = sess.write_cam1_frame if idx == 1 else sess.write_cam2_frame
            frame_idx = writer(bytes(jpeg_bytes))
            if frame_idx >= 0:
                self._frame_idx[cam] = frame_idx
                ct = self._cap_times[cam]
                if not ct:
                    ct[:] = [float(t_capture), float(t_capture), 1]
                else:
                    ct[1] = float(t_capture)
                    ct[2] += 1
                if self._fc_w is not None:
                    self._fc_w.writerow([
                        f"{sess.t():.4f}", idx, frame_idx,
                        f"{float(t_capture) - sess.t0:.4f}",
                        _finite_scalar(sim_t), _finite_scalar(tick), _finite_scalar(seq)])
            else:
                self._counts["frames_dropped"] += 1
                self._warn("frame", f"frame rejected ({cam}): undecodable JPEG")
            if depth_png:
                dwriter = sess.write_cam1_depth_frame if idx == 1 else sess.write_cam2_depth_frame
                if dwriter(compressed_depth_payload(depth_png), stamp_s=float(t_capture)) < 0:
                    self._warn("depth", f"depth frame rejected ({cam}): not a PNG")
            return frame_idx if frame_idx >= 0 else None
        except Exception as e:  # noqa: BLE001
            self._counts["frames_dropped"] += 1
            if isinstance(e, (OSError, RuntimeError)):
                self._counts["write_errors"] += 1
            self._warn("frame", f"frame dropped ({cam}): {type(e).__name__}: {e}")
            return None
        finally:
            with self._lock:
                self._inflight -= 1
                self._cv.notify_all()

    # ---- events / warnings ------------------------------------------------------
    def add_event(self, kind: str, **detail: Any) -> None:
        """Append ``{t_rel_s, kind, ...detail}`` to this take's ``sim_meta["events"]``
        (sim restart, scene change, tick going backwards ...). No-op when not recording."""
        with self._lock:
            sess = self._session
            if sess is None:
                return
            ev = {"t_rel_s": round(sess.t(), 4), "kind": str(kind)}
            ev.update(_scrub_paths(dict(detail)))
            self._events.append(ev)
        print(f"[SimTakeRecorder] event {ev}", flush=True)

    def _warn(self, key: str, text: str, every_s: float = 1.0) -> None:
        """Rate-limited warning (one line per ``key`` per ``every_s``; the count of
        suppressed lines is folded into the next one) -- ENOSPC must not print 155
        lines/s (R2 item 4)."""
        now = time.monotonic()
        last = self._warn_last.get(key)
        self._warn_last[f"{key}#n"] = self._warn_last.get(f"{key}#n", 0) + 1
        if last is not None and now - last < every_s:
            return
        n = int(self._warn_last[f"{key}#n"])
        self._warn_last[key] = now
        self._warn_last[f"{key}#n"] = 0
        suffix = f" (x{n} in the last {every_s:.0f}s)" if n > 1 else ""
        print(f"[SimTakeRecorder] WARNING: {text}{suffix}", flush=True)

    def flush(self) -> None:
        with self._lock:
            if self._session is not None:
                BufferedHdf5TableWriter.flush_all()
                self._session.flush()

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                self.stop()
