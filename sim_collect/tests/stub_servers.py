"""In-process stand-ins for [A] sim_main and [B] capture, to the letter of DESIGN.md §2.

Written by O2 so the GUI can be developed and tested before either real process exists;
kept importable and self-contained on purpose because the integration tests reuse them.

Each stub is a daemon thread that BINDS ITS OWN SOCKET INSIDE ``run()`` — a ZMQ socket
belongs to the thread that polls it, and creating it in ``__init__`` (on the caller's
thread) would hand a live socket across a thread boundary. ``start_and_wait()`` blocks
until the bind actually happened, so a test never races the bind.

The state machines are deliberately thin: they record every request in ``.calls`` and flip
just enough state that the GUI's own logic (two-click confirm, "RESET SCENE is refused
while recording", button enable/disable) has something truthful to read back. Rules that
DESIGN.md assigns to the GUI are NOT enforced here — e.g. ``reset_scene`` while a take is
recording is refused by the GUI, and this stub would happily accept it, which is exactly
what makes that test meaningful.

Usage::

    from sim_collect.tests.stub_servers import start_stubs
    with start_stubs() as s:          # sim + capture + preview publisher
        ...                          # s.sim, s.capture, s.preview
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from sim_collect import ipc

#: preview frame size published by [B] (DESIGN §2.2).
PREVIEW_SIZE = (320, 180)          # (w, h)
PREVIEW_HZ = 10.0


def _bind_with_retry(name: str, timeout: float = 4.0) -> ipc.Server:
    """Bind `name`, tolerating a briefly-occupied endpoint.

    The endpoints are FIXED by the contract, so two test runs (or a leftover process)
    genuinely contend for them. Retrying turns a hard crash into a wait; the caller still
    gets a clear error if the endpoint stays taken.
    """
    deadline = time.time() + timeout
    while True:
        try:
            return ipc.Server(name)
        except Exception:
            if time.time() >= deadline:
                raise
            time.sleep(0.1)


def _bind_pub_with_retry(name: str, timeout: float = 4.0) -> ipc.Publisher:
    """Same retry as :func:`_bind_with_retry`, for the preview PUB socket."""
    deadline = time.time() + timeout
    while True:
        try:
            return ipc.Publisher(name)
        except Exception:
            if time.time() >= deadline:
                raise
            time.sleep(0.1)


class _StubServer(threading.Thread):
    """Common REP-server plumbing: bind in the thread, record calls, stop cleanly."""

    endpoint_name = ""

    def __init__(self, poll_ms: int = 20) -> None:
        super().__init__(daemon=True, name=f"stub-{self.endpoint_name}")
        self._poll_ms = poll_ms
        self._halt = threading.Event()
        self.ready = threading.Event()
        self.error: Optional[BaseException] = None
        self.lock = threading.RLock()
        self.calls: List[Dict[str, Any]] = []

    # -- lifecycle ---------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - trivial loop, exercised by every test
        try:
            srv = _bind_with_retry(self.endpoint_name)
        except BaseException as e:   # report through start_and_wait, not as a thread crash
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        try:
            while not self._halt.is_set():
                srv.poll(self.handle, timeout_ms=self._poll_ms)
        finally:
            srv.close()

    def start_and_wait(self, timeout: float = 8.0) -> "_StubServer":
        self.start()
        if not self.ready.wait(timeout):
            raise RuntimeError(f"stub {self.endpoint_name} did not bind in {timeout}s")
        if self.error is not None:
            raise RuntimeError(f"stub {self.endpoint_name} bind failed: {self.error}")
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(timeout)

    # -- request bookkeeping ------------------------------------------------
    def handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
        cmd = str(req.get("cmd", ""))
        with self.lock:
            self.calls.append(dict(req))
        fn = getattr(self, f"do_{cmd}", None)
        if fn is None:
            return {"ok": False, "msg": f"unknown cmd {cmd!r}"}
        return fn(req)

    def cmds(self) -> List[str]:
        with self.lock:
            return [str(c.get("cmd", "")) for c in self.calls]

    def calls_of(self, cmd: str) -> List[Dict[str, Any]]:
        with self.lock:
            return [dict(c) for c in self.calls if c.get("cmd") == cmd]

    def saw(self, cmd: str) -> bool:
        return bool(self.calls_of(cmd))

    def clear(self) -> None:
        with self.lock:
            self.calls.clear()


class StubSim(_StubServer):
    """`sim_rep` stand-in (DESIGN §2.1). Engage/disengage just flip a flag — every gate
    the real controller runs is out of scope here; the GUI only needs a truthful
    ``engaged``/``eef_state`` to drive its toggle."""

    endpoint_name = "sim_rep"

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.control_mode = "eef"
        self.engaged = False
        self.pos_scale = 1.0
        self.gripper_paused = False
        self.gripper_ramping = False
        self.seed: Optional[int] = 1
        self.reset_count = 0
        self.home_count = 0
        self.sigma_min = 0.124
        self.task_success = False
        self.task_detail = "carrot not in pot"
        self.tick = 0
        self.leader_ok = True
        self.rt_ratio = 1.0
        self.startup_note = "teleported to the leader pose (fake)"
        self.t0 = time.time()
        #: set to make the next command fail, e.g. to exercise the log pane.
        self.fail_next: Optional[str] = None

    # -- helpers ------------------------------------------------------------
    @property
    def eef_state(self) -> str:
        if self.control_mode == "joint":
            return "JOINT_BOOTSTRAP"
        return "ENGAGED" if self.engaged else "DISENGAGED"

    def eef_info(self) -> Dict[str, Any]:
        """Exactly the key set of controller.EefController.info()."""
        return {
            "mode": self.control_mode, "state": self.eef_state, "ctrl_state": self.eef_state,
            "reject_reason": None, "auto_reason": None,
            "last_gate": "engage_ok" if self.engaged else None, "last_gate_detail": None,
            "sigma_min": self.sigma_min, "gamma": 1.0, "ls_scale": 1.0,
            "ik_residual": 0.0, "lag_pos": 0.0, "lag_rot": 0.0, "lag": [0.0, 0.0],
            "excursion": 0.0, "branch_id": 0, "n_ik_solutions": 8,
            "pos_scale": self.pos_scale, "pending_pos_scale": None,
            "step_eff": 0.0025, "soft_start_active": False,
            "leader_age": 0.01, "filter_settle": 0.0, "seeded": True,
        }

    def status(self) -> Dict[str, Any]:
        """Byte-for-byte the key set of sim_main.SimApp.handle()'s get_status reply."""
        with self.lock:
            return {
                "ok": True, "msg": "alive",
                "engaged": self.engaged, "eef_state": self.eef_state,
                "control_mode": self.control_mode, "pos_scale": self.pos_scale,
                "tick": self.tick, "sim_t": round(self.tick * 0.002, 3),
                "leader_ok": self.leader_ok, "leader_fake": True, "viewer": False,
                "gripper_paused": self.gripper_paused,
                "task_success": bool(self.task_success),
                "layout_seed": self.seed, "rt_ratio": self.rt_ratio,
                "eef_info": self.eef_info(),
                "uptime_s": round(time.time() - self.t0, 1),
                "startup_note": self.startup_note,
                "gripper_ramping": self.gripper_ramping,
            }

    def _maybe_fail(self, cmd: str) -> Optional[Dict[str, Any]]:
        if self.fail_next == cmd:
            self.fail_next = None
            return {"ok": False, "msg": f"{cmd} refused (stub fail_next)"}
        return None

    # -- commands -----------------------------------------------------------
    def do_get_status(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.status()

    def do_engage(self, req: Dict[str, Any]) -> Dict[str, Any]:
        bad = self._maybe_fail("engage")
        if bad:
            return bad
        with self.lock:
            self.engaged = True
            self.tick += 1
        return {"ok": True, "key": "engaged", "msg": "engaged", "eef_state": self.eef_state}

    def do_disengage(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.engaged = False
        return {"ok": True, "key": "disengaged", "msg": "disengaged", "eef_state": self.eef_state}

    def do_set_pos_scale(self, req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            v = float(req.get("value"))
        except (TypeError, ValueError):
            return {"ok": False, "msg": f"bad value {req.get('value')!r}"}
        if not 0.0 < v <= 1.0:
            return {"ok": False, "key": "range", "msg": f"pos_scale {v} out of (0, 1]",
                    "pos_scale": self.pos_scale}
        with self.lock:
            self.pos_scale = v
        return {"ok": True, "key": "set", "msg": f"pos_scale={v:.2f}", "pos_scale": v}

    def do_set_control_mode(self, req: Dict[str, Any]) -> Dict[str, Any]:
        mode = req.get("mode")
        if mode not in ("eef", "joint"):
            return {"ok": False, "key": "bad_mode", "msg": f"bad mode {mode!r}",
                    "control_mode": self.control_mode}
        with self.lock:
            if self.engaged:
                return {"ok": False, "key": "engaged",
                        "msg": "cannot change control_mode while ENGAGED",
                        "control_mode": self.control_mode}
            self.control_mode = mode
        return {"ok": True, "key": "set", "msg": f"control_mode={mode}", "control_mode": mode}

    def do_reset_scene(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.engaged = False
            self.seed = req.get("seed")
            self.reset_count += 1
            self.task_success = False
        return {"ok": True, "msg": f"scene reset (seed={self.seed})", "seed": self.seed}

    def do_home(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.engaged = False
            self.home_count += 1
        return {"ok": True, "msg": "homed"}

    def do_gripper_pause(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.gripper_paused = True
        return {"ok": True, "msg": "gripper paused"}

    def do_gripper_resume(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.gripper_paused = False
        return {"ok": True, "msg": "gripper resumed"}

    def do_get_scene_meta(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "msg": "stub scene", "objects": {}, "seed": self.seed}


class StubCapture(_StubServer):
    """`capture_rep` stand-in (DESIGN §2.2). Take dirs are names only — nothing is written.

    The status dict is the key set of ``capture.CaptureApp.status()`` =
    ``recorder.SimTakeRecorder.status()`` + the capture-level health fields. Keep them in
    step with those two functions: the GUI is tested against THIS dict, so a key invented
    here would be a key the GUI reads and the real server never sends.
    """

    endpoint_name = "capture_rep"

    def __init__(self, root: str = "/tmp/sim_collect_stub_takes", fps: float = 30.0,
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.root = root
        self.fps = fps
        self.recording = False
        self.take_index = 0
        self.take_dir: Optional[str] = None    # None while idle, like the real recorder
        self.last_take_dir: Optional[str] = None
        self.discarded: List[str] = []
        self.note: Optional[str] = None
        self.task_success = False
        self.sim_alive = True
        self.scene_ready = True
        self.workers_alive = True
        self.bad_msgs = 0
        self.state_dropped = 0
        self.frames_dropped = 0
        self.write_errors = 0
        self.missed_ticks = 0
        self.tick_restarts = 0
        self.state_msgs_idle = 0
        self.events: List[Dict[str, Any]] = []
        self.render_hz = 30.0
        self.render_ms = 12.6
        self.achieved_fps = 30.0
        self.queue_drops = 0
        self.started_at = time.time()
        self._t0 = 0.0

    def duration(self) -> float:
        return (time.time() - self._t0) if self.recording else 0.0

    def session_counts(self, el: float) -> Dict[str, int]:
        """`rows` in the real reply is ``RecordingSession._counts`` — per-STREAM message
        counters (the key names are the ones asserted in recording_session.py's own
        self-test), plus the frame counters. EMPTY while not recording."""
        n = int(el * self.fps)
        return {
            "gello_joint_states": int(el * 30), "gello_grip": int(el * 30),
            "command": int(el * 125), "ur_joint_states": int(el * 125),
            "wrench": int(el * 125), "tcp_pose": int(el * 125),
            "grip_cmd": int(el * 30), "grip_pos": int(el * 30),
            "cam1_frames": n, "cam2_frames": n,
            "cam1_depth_frames": n, "cam2_depth_frames": n,
        }

    def recorder_counts(self, el: float) -> Dict[str, int]:
        """`counts` = the recorder's own lifetime counters (SimTakeRecorder._counts)."""
        return {"state_msgs": int(el * 250) + self.state_msgs_idle,
                "state_dropped": self.state_dropped, "frames_dropped": self.frames_dropped,
                "write_errors": self.write_errors, "missed_ticks": self.missed_ticks,
                "tick_restarts": self.tick_restarts}

    def problems(self) -> List[str]:
        """Copied from SimTakeRecorder.problems(): empty = clean take."""
        c = self.recorder_counts(self.duration())
        out = []
        if c["write_errors"]:
            out.append(f"{c['write_errors']} write errors (disk full? see log)")
        if c["state_dropped"]:
            out.append(f"{c['state_dropped']} state messages dropped")
        if c["frames_dropped"]:
            out.append(f"{c['frames_dropped']} frames rejected by the writers")
        if c["missed_ticks"]:
            out.append(f"{c['missed_ticks']} sim ticks never reached the recorder")
        if c["tick_restarts"]:
            out.append(f"sim restarted {c['tick_restarts']}x mid-take")
        return out

    def capture_stats(self) -> Dict[str, Any]:
        """Copied from CaptureApp.capture_stats()."""
        cams = ("cam1", "cam2")
        return {
            "achieved_fps": {c: self.achieved_fps for c in cams},
            "worker_queue_drops": {c: self.queue_drops for c in cams},
            "worker_bad_state": {c: 0 for c in cams},
            "render_ms": {c: self.render_ms for c in cams},
            "nominal_fps": self.fps,
            "fps_ok": self.achieved_fps >= 0.97 * self.fps,
        }

    def status(self) -> Dict[str, Any]:
        with self.lock:
            el = self.duration()
            sess = self.session_counts(el) if self.recording else {}
            rec = self.recorder_counts(el)
            cams = ("cam1", "cam2")
            return {
                "ok": True,
                # ---- recorder.SimTakeRecorder.status() ----
                "recording": self.recording,
                "take_dir": self.take_dir,
                "last_take_dir": self.last_take_dir,
                "take_index": self.take_index,
                "duration_s": round(el, 2),
                "frames": {c: sess.get(f"{c}_frames", 0) for c in cams},
                "depth_frames": {c: sess.get(f"{c}_depth_frames", 0) for c in cams},
                "rows": sess,
                "state_msgs": rec["state_msgs"],
                "counts": rec,
                "problems": self.problems() if self.recording else [],
                "events": list(self.events) if self.recording else [],
                "task_success": self.task_success,
                # ---- capture.CaptureApp.status() ----
                "sim_alive": self.sim_alive,
                "bad_msgs": self.bad_msgs,
                "last_tick": int(el * 250),
                "scene_ready": self.scene_ready and self.workers_alive,
                "scene_sha": "b4a6539629748327",
                "workers": {c: self.workers_alive for c in cams},
                "render": {c: {"n": sess.get(f"{c}_frames", 0), "hz": self.render_hz,
                               "render_ms": self.render_ms, "queue_drops": self.queue_drops,
                               "bad_state": 0, "achieved_fps": self.achieved_fps}
                           for c in cams},
                "capture": self.capture_stats(),
                "root": self.root,
                "uptime_s": round(time.time() - self.started_at, 1),
            }

    def do_get_status(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return self.status()

    def do_start_take(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if self.recording:
                return {"ok": False, "msg": f"already recording {self.take_dir}"}
            if not (self.scene_ready and self.workers_alive):
                return {"ok": False, "msg": "scene/render workers not ready"}
            if not self.sim_alive:
                return {"ok": False, "msg": "no state from the sim in the last second"}
            self.take_index += 1
            self.take_dir = "{}/take_{:02d}_{}".format(
                self.root, self.take_index, time.strftime("%Y%m%d_%H%M%S"))
            self.recording = True
            self._t0 = time.time()
            self.note = req.get("note")
            base = self.take_dir.rsplit("/", 1)[-1]
        return {"ok": True, "take_dir": self.take_dir, "take_index": self.take_index,
                "msg": f"recording {base}"}

    def do_stop_take(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if not self.recording:
                return {"ok": False, "msg": "not recording"}
            el = self.duration()
            self.recording = False
            self.last_take_dir = self.take_dir
            base = (self.take_dir or "").rsplit("/", 1)[-1]
            problems = self.problems()
            self.take_dir = None
        return {"ok": not problems, "take_dir": self.last_take_dir, "duration_s": round(el, 3),
                "task_success_at_stop": self.task_success,
                "recorder_counts": self.recorder_counts(el),
                "problems": problems, "events": [],
                "msg": f"saved {base} ({el:.1f}s)"}

    def do_discard_last_take(self, req: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if self.recording:
                return {"ok": False, "msg": "stop the take before discarding it"}
            d = self.last_take_dir
            if not d:
                return {"ok": False, "msg": "no take to discard"}
            self.discarded.append(d)
            self.last_take_dir = None
            self.take_dir = None
        return {"ok": True, "msg": f"deleted {d}", "deleted": d}

    def do_snapshot(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "cam1": solid_jpeg((30, 90, 200)), "cam2": solid_jpeg((30, 200, 90)),
                "recording": self.recording, "take_dir": self.take_dir}


def solid_jpeg(bgr: tuple = (30, 90, 200), size: tuple = PREVIEW_SIZE) -> bytes:
    """One solid-colour JPEG of the real preview size (cv2 wants BGR)."""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = bgr
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:  # pragma: no cover - cv2 always encodes a solid image
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


class StubPreview(threading.Thread):
    """`preview_pub` stand-in: two solid-colour 320×180 JPEGs at 10 Hz (DESIGN §2.2).

    Optionally mirrors a :class:`StubCapture`'s recording state so the GUI's preview
    payload and its polled capture status agree.
    """

    def __init__(self, capture: Optional[StubCapture] = None, hz: float = PREVIEW_HZ,
                 extra: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
        super().__init__(daemon=True, name="stub-preview")
        self.capture = capture
        self.hz = hz
        self.extra = extra
        self.sent = 0
        self._halt = threading.Event()
        self.ready = threading.Event()
        self.error: Optional[BaseException] = None
        self.cam1 = solid_jpeg((30, 90, 200))      # blue-ish "scene"
        self.cam2 = solid_jpeg((30, 200, 90))      # green-ish "wrist"

    def run(self) -> None:  # pragma: no cover - trivial loop
        try:
            pub = _bind_pub_with_retry("preview_pub")
        except BaseException as e:
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        period = 1.0 / self.hz
        try:
            while not self._halt.is_set():
                msg: Dict[str, Any] = {
                    "t": time.time(), "cam1": self.cam1, "cam2": self.cam2,
                    "recording": False, "take_dir": "", "take_index": 0,
                    "frames": {"cam1": 0, "cam2": 0}, "duration_s": 0.0, "rows": {},
                    "sim_alive": True, "scene_ready": True, "task_success": False}
                if self.capture is not None:
                    st = self.capture.status()
                    msg.update({k: st[k] for k in
                                ("recording", "take_dir", "take_index", "frames",
                                 "duration_s", "rows", "sim_alive", "scene_ready",
                                 "task_success")})
                if self.extra is not None:
                    msg.update(self.extra())
                pub.send("preview", msg)
                self.sent += 1
                self._halt.wait(period)
        finally:
            pub.close()

    def start_and_wait(self, timeout: float = 8.0) -> "StubPreview":
        self.start()
        if not self.ready.wait(timeout):
            raise RuntimeError("stub preview publisher did not bind")
        if self.error is not None:
            raise RuntimeError(f"stub preview publisher bind failed: {self.error}")
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(timeout)


class StubStack:
    """The three stubs together; also a context manager."""

    def __init__(self, sim: Optional[StubSim], capture: Optional[StubCapture],
                 preview: Optional[StubPreview]) -> None:
        self.sim = sim
        self.capture = capture
        self.preview = preview

    def stop(self) -> None:
        for s in (self.preview, self.capture, self.sim):
            if s is not None:
                s.stop()

    def __enter__(self) -> "StubStack":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def start_stubs(sim: bool = True, capture: bool = True, preview: bool = True) -> StubStack:
    """Start the requested stubs and return once every socket is bound."""
    s = StubSim().start_and_wait() if sim else None
    c = StubCapture().start_and_wait() if capture else None
    p = StubPreview(capture=c).start_and_wait() if preview else None
    return StubStack(s, c, p)  # type: ignore[arg-type]
