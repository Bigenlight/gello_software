"""Process [B] of sim_collect: cameras + recorder (DESIGN.md §2.2).

    .venv/bin/python -m sim_collect.capture --root ros2_ur_ws/gello_logs/sim [--config ...]

Layout (measured on this machine, see the numbers in DESIGN §1 and in
``sim_collect/tests/test_capture.py``): one loop cannot render two 1280x720 colour +
two 848x480 depth images, JPEG/PNG-encode them, decode+mp4v-encode them again through
``Mp4FrameWriter`` AND append ~14k HDF5 cells/s at the §4.2 table rates inside 33 ms.
So:

* two **render workers** (``multiprocessing`` spawn, one per camera) each own a
  :class:`~sim_collect.cameras.CameraRig`, subscribe to ``state_pub`` themselves
  (latest state wins), render at ``fps`` with their own pacing and push
  ``(jpeg, depth_png, preview_jpeg)`` into a bounded queue (a full queue drops the frame
  -- the recorder never lags behind the sim);
* the **main process** subscribes to ``state_pub`` for the tables (every message,
  decimated by :class:`~sim_collect.recorder.SimTakeRecorder`), serves ``capture_rep``
  and publishes ``preview_pub``; a **frame thread** drains the worker queue into the
  recorder (``cv2`` releases the GIL for the JPEG decode / mp4v write, h5py serialises
  under its own lock).

Robustness: the sim may restart at any time -- ZMQ SUB reconnects on its own; a
``tick`` that jumps backwards re-fetches ``get_scene_meta`` and rebuilds the workers when
the scene changed; a malformed state message is counted and skipped, never fatal.
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import traceback
from typing import Any, Dict, Mapping, Optional

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "ros2_ur_ws", "src", "gello_recorder"),
           os.path.join(_ROOT, "ros2_ur_ws", "src", "ur_gello_bringup")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim_collect import ipc  # noqa: E402
from sim_collect import cameras as _cams  # noqa: E402
from sim_collect.recorder import SimTakeRecorder, _scrub_paths  # noqa: E402

DEFAULT_ROOT = os.environ.get("SIM_COLLECT_OUTPUT_ROOT",
                              os.path.join(_ROOT, "ros2_ur_ws", "gello_logs", "sim"))
PREVIEW_SIZE = (320, 180)
JPEG_QUALITY = 92
STATE_STALE_S = 1.0
FLUSH_EVERY_S = 1.0     # recorder.flush() cadence from the main loop


# --------------------------------------------------------------------------- scene
def resolve_scene(meta: Optional[Mapping[str, Any]], config: Optional[Mapping[str, Any]] = None,
                  scene_xml_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Turn a ``get_scene_meta`` reply into ``{"xml", "assets", "xml_path", "sha"}``.

    Contract keys tried in order: ``scene_xml`` (str) + ``scene_assets`` ({name: bytes}),
    ``scene_xml_path`` (file the sim saved). Fallback: ``sim_collect.scene.build_scene``
    (F1's module, imported lazily; several call shapes are tried because the signature is
    theirs), then ``scene_xml_path`` from the CLI. ``None`` when nothing works."""
    meta = dict(meta or {})
    if isinstance(meta.get("meta"), Mapping):          # sim_main wraps it: {"ok", "msg", "meta"}
        meta = dict(meta["meta"])
    xml = meta.get("scene_xml")
    if isinstance(xml, str) and xml.strip():
        assets = meta.get("scene_assets") or {}
        assets = {str(k): bytes(v) for k, v in dict(assets).items()}
        return _pack_scene(xml, assets, None)
    path = meta.get("scene_xml_path")
    if isinstance(path, str) and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return _pack_scene(fh.read(), {}, path)
    # sim_main's actual reply: config (raw yaml dict) + config_path + layout + xml_sha256.
    # Rebuild the same MJCF here (build_scene is deterministic given config + layout).
    cfg_src = meta.get("config") if isinstance(meta.get("config"), Mapping) else config
    built = _build_scene_fallback(cfg_src, meta.get("layout"), meta.get("config_path"))
    if built is not None:
        want = meta.get("xml_sha256")
        if isinstance(want, str) and want and want != built["xml_sha256"]:
            print(f"[capture] WARNING: rebuilt scene xml_sha256 {built['xml_sha256'][:12]} != sim's "
                  f"{want[:12]}; cameras may not match the physics scene")
            built["sha_mismatch"] = True
        # offscreen render quality for the workers (yaml `render:` block, see CameraRig)
        if isinstance(cfg_src, Mapping):
            built["render"] = dict(cfg_src.get("render") or {})
        return built
    if scene_xml_path and os.path.isfile(scene_xml_path):
        with open(scene_xml_path, "r", encoding="utf-8") as fh:
            return _pack_scene(fh.read(), {}, scene_xml_path)
    return None


def _pack_scene(xml: str, assets: Dict[str, bytes], xml_path: Optional[str]) -> Dict[str, Any]:
    xml_sha = hashlib.sha256(xml.encode("utf-8")).hexdigest()   # same recipe as sim_main's xml_sha256
    h = hashlib.sha256(xml.encode("utf-8"))
    for k in sorted(assets):
        h.update(k.encode()); h.update(assets[k])
    return {"xml": xml, "assets": assets, "xml_path": xml_path, "sha": h.hexdigest()[:16],
            "xml_sha256": xml_sha}


def _coerce_scene_result(obj: Any) -> Optional[Dict[str, Any]]:
    """Accept whatever F1's ``build_scene`` returns: an ``(xml, assets)`` tuple, a
    dm_control ``mjcf.RootElement`` (``to_xml_string`` / ``get_assets``), a dict with the
    contract keys, or a bare XML string."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return _pack_scene(obj, {}, None)
    if isinstance(obj, tuple) and len(obj) >= 2 and isinstance(obj[0], str):
        return _pack_scene(obj[0], {str(k): bytes(v) for k, v in dict(obj[1] or {}).items()}, None)
    if isinstance(obj, Mapping):
        return resolve_scene(obj)
    if hasattr(obj, "to_xml_string"):
        assets = obj.get_assets() if hasattr(obj, "get_assets") else {}
        return _pack_scene(obj.to_xml_string(), {str(k): bytes(v) for k, v in dict(assets).items()}, None)
    for attr in ("xml", "scene_xml"):
        if hasattr(obj, attr) and isinstance(getattr(obj, attr), str):
            assets = getattr(obj, "assets", None) or getattr(obj, "scene_assets", None) or {}
            return _pack_scene(getattr(obj, attr), {str(k): bytes(v) for k, v in dict(assets).items()}, None)
    return None


def _build_scene_fallback(config: Any, layout: Any, config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Rebuild the MJCF with F1's ``scene.build_scene(SceneConfig, layout)``.

    ``config`` may be the raw yaml dict (as ``get_scene_meta`` sends it), a yaml path, or
    ``None`` (then ``config_path`` is loaded). Returns ``None`` when the scene module is
    absent or the build fails -- the caller then tries the other sources."""
    try:
        from sim_collect import scene as scene_mod  # F1's module
    except Exception:  # noqa: BLE001
        return None
    build = getattr(scene_mod, "build_scene", None)
    scene_config = getattr(scene_mod, "SceneConfig", None)
    if build is None:
        return None
    try:
        if isinstance(config, Mapping) and scene_config is not None:
            cfg = scene_config.from_dict(dict(config), path=config_path)
        elif isinstance(config, str) and scene_config is not None:
            cfg = scene_config.load(config)
        elif config is None and config_path and scene_config is not None:
            cfg = scene_config.load(config_path)
        else:
            cfg = config
        if cfg is None:
            return None
        layout = dict(layout) if isinstance(layout, Mapping) and layout else None
        return _coerce_scene_result(build(cfg, layout))
    except Exception as e:  # noqa: BLE001
        print(f"[capture] scene.build_scene failed: {type(e).__name__}: {e}")
        return None


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg if isinstance(cfg, dict) else {}


# ------------------------------------------------------------------- render worker
def _render_worker(cam: str, scene: Dict[str, Any], fps: float, depth_max_m: float,
                   out_q: "mp.Queue", ctrl: "mp.connection.Connection",
                   jpeg_quality: int = JPEG_QUALITY, record_depth: bool = False) -> None:
    """Body of one render process: own rig, own SUB socket, own ``fps`` pacing.

    Lifetime is tied to the parent three ways (R2 found workers outliving capture by
    >12 s holding GL contexts): a ``"stop"`` on the control pipe, EOF on that pipe, or
    ``os.getppid()`` changing (parent gone) all end the loop; the queue's feeder thread
    is detached (``cancel_join_thread``) and the process leaves via ``os._exit(0)``
    after releasing the renderer, so no atexit/feeder join can hang it.

    Pacing: exactly ``fps`` when the render fits the period; when it does not, the
    achieved rate is measured over 2 s windows and reported in every item as
    ``achieved_fps`` plus a loud warning (rate-limited) -- the mp4 is stamped with the
    nominal fps like the real files, so the caller records the achieved rate in
    ``sim_meta``/status rather than silently shipping time-warped video."""
    import cv2
    os.environ.setdefault("MUJOCO_GL", "glfw")
    parent_pid = os.getppid()
    try:
        out_q.cancel_join_thread()
    except Exception:  # noqa: BLE001
        pass
    rig = None
    sub = None
    try:
        rig = _cams.CameraRig(scene["xml"], scene["assets"], xml_path=scene.get("xml_path"),
                              depth_max_m=depth_max_m, render=scene.get("render"))
        sub = ipc.Subscriber("state_pub", "state", conflate=True)   # render newest state only
        ctrl.send({"ready": True, "cam": cam, "nq": rig.nq})
    except Exception as e:  # noqa: BLE001
        try:
            ctrl.send({"ready": False, "cam": cam, "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()})
        except Exception:  # noqa: BLE001
            pass
        os._exit(0)
    period = 1.0 / float(fps)
    next_t = time.perf_counter()
    seq = 0
    dropped = 0
    bad_state = 0
    has_state = False
    sim_t = None
    tick = None
    enc_jpg = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    enc_png = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    # achieved-rate measurement: sliding window over the last 60 completed frames,
    # excluding frame 0 (GL context + first render cost ~280 ms and would poison a
    # fixed 2 s window, which R2/we saw as a spurious 23-26 Hz right after start-up)
    from collections import deque
    frame_times: "deque[float]" = deque(maxlen=60)
    achieved_fps = float(fps)
    last_warn = 0.0
    reason = "stop"
    while True:
        try:
            if ctrl.poll(0):
                cmd = ctrl.recv()
                if cmd == "stop":
                    break
        except (EOFError, OSError):
            reason = "control pipe closed"
            break
        if os.getppid() != parent_pid:
            reason = "parent gone"
            break
        msg = sub.latest()
        if isinstance(msg, Mapping) and "qpos_full" in msg:
            try:
                rig.mirror(msg["qpos_full"], msg.get("qvel_full"))
                has_state = True
                sim_t = msg.get("sim_t")
                tick = msg.get("tick")
            except Exception:  # noqa: BLE001 - wrong scene / malformed: keep last pose
                bad_state += 1
        t_cap = time.time()
        t0 = time.perf_counter()
        try:
            rgb = rig.render_color(cam)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            ok1, jpg = cv2.imencode(".jpg", bgr, enc_jpg)
            if record_depth:   # depth is optional (default off): skip the second render + PNG
                mm = rig.render_depth(cam)
                ok2, png = cv2.imencode(".png", mm, enc_png)
            else:
                ok2, png = False, None
            small = cv2.resize(bgr, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
            ok3, pv = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as e:  # noqa: BLE001
            print(f"[render:{cam}] render failed: {type(e).__name__}: {e}")
            time.sleep(period)
            continue
        now_pc = time.perf_counter()
        if seq > 0:
            frame_times.append(now_pc)
        if len(frame_times) >= 30:
            achieved_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            if achieved_fps < 0.97 * fps and now_pc - last_warn >= 5.0:
                last_warn = now_pc
                print(f"[render:{cam}] WARNING: achieved {achieved_fps:.1f} Hz < nominal {fps:.0f} Hz "
                      f"(render {1e3 * (now_pc - t0):.1f} ms); mp4 is stamped {fps:.0f} fps -> "
                      f"video plays {fps / max(achieved_fps, 1e-6):.2f}x fast. achieved_fps is recorded in sim_meta.")
        item = {"cam": cam, "t": t_cap, "seq": seq, "has_state": has_state,
                "sim_t": sim_t, "tick": tick,
                "jpeg": jpg.tobytes() if ok1 else None,
                "png": png.tobytes() if ok2 else None,
                "preview": pv.tobytes() if ok3 else None,
                "render_ms": 1e3 * (now_pc - t0),
                "achieved_fps": achieved_fps,
                "dropped": dropped, "bad_state": bad_state}
        try:
            out_q.put_nowait(item)
        except queue.Full:
            dropped += 1
            if now_pc - last_warn >= 5.0:
                last_warn = now_pc
                print(f"[render:{cam}] WARNING: frame queue full, {dropped} frames dropped so far "
                      "(recorder cannot keep up)")
        except (ValueError, OSError):   # queue closed by the parent
            reason = "queue closed"
            break
        seq += 1
        next_t += period
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -period:      # fell far behind: re-anchor instead of bursting
            next_t = time.perf_counter()
    try:
        if sub is not None:
            sub.close()
        if rig is not None:
            rig.close()
    except Exception:  # noqa: BLE001
        pass
    print(f"[render:{cam}] exit ({reason}) after {seq} frames, {dropped} dropped", flush=True)
    os._exit(0)


# ----------------------------------------------------------------------- service
class CaptureService:
    """Owns the workers, the recorder, ``capture_rep`` and ``preview_pub``."""

    def __init__(self, root: str = DEFAULT_ROOT, config: Optional[Mapping[str, Any]] = None,
                 *, fps: float = 30.0, preview_hz: float = 10.0, record_depth: Optional[bool] = None,
                 scene_xml_path: Optional[str] = None, sim_timeout_ms: int = 1500,
                 scene: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        self.root = root
        self.config = dict(config or {})
        self.config_path = config_path
        self.fps = float(fps)
        self.preview_hz = float(preview_hz)
        self.scene_xml_path = scene_xml_path
        cams_cfg = self.config.get("cameras") if isinstance(self.config.get("cameras"), Mapping) else {}
        self.depth_max_m = float(cams_cfg.get("depth_max_m", _cams.DEPTH_MAX_M))
        # Depth recording is OPTIONAL and OFF by default (operator decision 2026-09-14):
        # yaml `cameras.record_depth`, overridden by CLI --depth/--no-depth. Off = no
        # depth render, no depth.h5 (3-file take); on = the real recorder's 4-file take.
        cfg_depth = bool(cams_cfg.get("record_depth", False))
        self.record_depth = cfg_depth if record_depth is None else bool(record_depth)
        self.cams = tuple(_cams.CAMS)
        self.recorder = SimTakeRecorder(camera_fps=self.fps, record_depth=self.record_depth)
        self._ctx = mp.get_context("spawn")
        self._frame_q: Optional["mp.Queue"] = None
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._scene: Optional[Dict[str, Any]] = scene
        self._scene_meta: Dict[str, Any] = {}
        self._meta_rig: Optional[_cams.CameraRig] = None
        self._sim = ipc.Client("sim_rep", timeout_ms=sim_timeout_ms)
        self._sub: Optional[ipc.Subscriber] = None
        self._server: Optional[ipc.Server] = None
        self._pub: Optional[ipc.Publisher] = None
        self._previews: Dict[str, Optional[bytes]] = {c: None for c in self.cams}
        self._frame_stats: Dict[str, Dict[str, Any]] = {
            c: {"n": 0, "hz": 0.0, "last_t": None, "render_ms": 0.0, "queue_drops": 0,
                "bad_state": 0, "achieved_fps": 0.0}
            for c in self.cams}
        self._fps_warned_at = 0.0
        self._last_state: Optional[Mapping[str, Any]] = None
        self._last_state_t = 0.0
        self._last_tick: Optional[int] = None
        self._state_msgs = 0
        self._bad_msgs = 0
        self._next_meta_try = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.started_at = time.time()

    # ---- scene / workers ---------------------------------------------------------
    def fetch_scene_meta(self) -> bool:
        """Ask the sim for ``get_scene_meta`` and (re)build the workers when the scene
        differs from the one we render. Returns True when a scene is available."""
        rep = self._sim.call("get_scene_meta")
        if rep.get("ok"):
            meta = dict(rep["meta"]) if isinstance(rep.get("meta"), Mapping) \
                else {k: v for k, v in rep.items() if k not in ("ok", "msg")}
            if "config_path" not in meta and self.config_path:
                meta["config_path"] = self.config_path
            scene = resolve_scene(meta, self.config, self.scene_xml_path)
            self._scene_meta = _scrub_paths({k: v for k, v in meta.items()
                                             if k not in ("scene_xml", "scene_assets")})
        else:
            scene = self._scene or resolve_scene(
                {"config_path": self.config_path} if self.config_path else None,
                self.config, self.scene_xml_path)
            if scene is None:
                return False
        if scene is None:
            print(f"[capture] get_scene_meta reply has no usable scene: keys={sorted(rep)}")
            return self._scene is not None
        if self._scene is None or scene["sha"] != self._scene["sha"] or not self.workers_alive():
            if self._scene is not None and scene["sha"] != self._scene["sha"]:
                print(f"[capture] scene changed {self._scene['sha']} -> {scene['sha']}; rebuilding workers")
                self.recorder.add_event("scene_changed", sha_before=self._scene["sha"], sha_after=scene["sha"])
            elif self._scene is not None:
                self.recorder.add_event("workers_restarted", sha=scene["sha"])
            self._scene = scene
            self._restart_workers()
        return True

    def workers_alive(self) -> bool:
        return bool(self._workers) and all(w["proc"].is_alive() for w in self._workers.values())

    def _restart_workers(self) -> None:
        self._stop_workers()
        assert self._scene is not None
        try:
            self._meta_rig = _cams.CameraRig(self._scene["xml"], self._scene["assets"],
                                             xml_path=self._scene.get("xml_path"),
                                             depth_max_m=self.depth_max_m)
        except Exception as e:  # noqa: BLE001
            print(f"[capture] scene does not compile here: {type(e).__name__}: {e}")
            self._meta_rig = None
            return
        if self._frame_q is None:
            self._frame_q = self._ctx.Queue(maxsize=16)
        for cam in self.cams:
            parent, child = self._ctx.Pipe()
            proc = self._ctx.Process(
                target=_render_worker, name=f"render-{cam}", daemon=True,
                args=(cam, self._scene, self.fps, self.depth_max_m, self._frame_q, child),
                kwargs={"record_depth": self.record_depth})
            proc.start()
            self._workers[cam] = {"proc": proc, "conn": parent, "ready": None}
        deadline = time.time() + 30.0
        for cam, w in self._workers.items():
            while w["ready"] is None and time.time() < deadline:
                if w["conn"].poll(0.1):
                    rep = w["conn"].recv()
                    w["ready"] = bool(rep.get("ready"))
                    if not w["ready"]:
                        print(f"[capture] render worker {cam} failed: {rep.get('error')}\n{rep.get('traceback', '')}")
                elif not w["proc"].is_alive():
                    w["ready"] = False
        print(f"[capture] scene {self._scene['sha']} workers: "
              + ", ".join(f"{c}={'ready' if w['ready'] else 'FAILED'}" for c, w in self._workers.items()))

    def _stop_workers(self) -> None:
        """Stop every render worker for good: cooperative stop, then terminate, then
        kill -- and always join, so no worker outlives the parent (R2 item 2)."""
        for cam, w in list(self._workers.items()):
            try:
                w["conn"].send("stop")
            except Exception:  # noqa: BLE001
                pass
        for cam, w in list(self._workers.items()):
            p = w["proc"]
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
            if p.is_alive():
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
                p.join(timeout=1.0)
            try:
                w["conn"].close()
            except Exception:  # noqa: BLE001
                pass
            if p.is_alive():
                print(f"[capture] WARNING: render worker {cam} (pid {p.pid}) did not die")
        self._workers = {}

    # ---- frames ------------------------------------------------------------------
    def _frame_loop(self) -> None:
        while not self._stop.is_set():
            q = self._frame_q
            if q is None:
                time.sleep(0.05)
                continue
            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                time.sleep(0.05)
                continue
            try:
                self._on_frame(item)
            except Exception as e:  # noqa: BLE001
                print(f"[capture] frame handling failed: {type(e).__name__}: {e}")

    def _on_frame(self, item: Mapping[str, Any]) -> None:
        cam = item.get("cam")
        if cam not in self.cams:
            return
        st = self._frame_stats[cam]
        now = time.time()
        if st["last_t"] is not None:
            dt = now - st["last_t"]
            if dt > 0:
                st["hz"] = 0.9 * st["hz"] + 0.1 / dt if st["hz"] else 1.0 / dt
        st["last_t"] = now
        st["n"] += 1
        st["render_ms"] = 0.9 * st["render_ms"] + 0.1 * float(item.get("render_ms", 0.0))
        st["queue_drops"] = int(item.get("dropped", 0))
        st["bad_state"] = int(item.get("bad_state", 0))
        st["achieved_fps"] = float(item.get("achieved_fps", 0.0))
        if st["achieved_fps"] and st["achieved_fps"] < 0.97 * self.fps and now - self._fps_warned_at >= 5.0:
            self._fps_warned_at = now
            print(f"[capture] WARNING: {cam} renders at {st['achieved_fps']:.1f} Hz, mp4 stamped "
                  f"{self.fps:.0f} fps (video plays {self.fps / st['achieved_fps']:.2f}x fast); "
                  "see sim_meta.achieved_fps")
        if item.get("preview"):
            self._previews[cam] = item["preview"]
        if item.get("jpeg") and self.recorder.recording:
            self.recorder.on_frames(cam, float(item["t"]), item["jpeg"], item.get("png"),
                                    sim_t=item.get("sim_t"), tick=item.get("tick"), seq=item.get("seq"))

    def capture_stats(self) -> Dict[str, Any]:
        """Per-camera worker figures for status / preview / sim_meta."""
        return {
            "achieved_fps": {c: round(s["achieved_fps"], 2) for c, s in self._frame_stats.items()},
            "worker_queue_drops": {c: s["queue_drops"] for c, s in self._frame_stats.items()},
            "worker_bad_state": {c: s["bad_state"] for c, s in self._frame_stats.items()},
            "render_ms": {c: round(s["render_ms"], 1) for c, s in self._frame_stats.items()},
            "nominal_fps": self.fps,
            "fps_ok": all((s["achieved_fps"] == 0.0 or s["achieved_fps"] >= 0.97 * self.fps)
                          for s in self._frame_stats.values()),
        }

    # ---- state -------------------------------------------------------------------
    def _on_state(self, msg: Any) -> None:
        if not isinstance(msg, Mapping):
            self._bad_msgs += 1
            return
        self._state_msgs += 1
        self._last_state = msg
        self._last_state_t = time.time()
        tick = msg.get("tick")
        if isinstance(tick, (int, np.integer)):
            if self._last_tick is not None and tick < self._last_tick - 50:
                print(f"[capture] sim restart detected (tick {self._last_tick} -> {tick}); re-checking scene")
                self._next_meta_try = 0.0
                self.recorder.add_event("sim_restart", tick_before=int(self._last_tick), tick_after=int(tick))
            self._last_tick = int(tick)
        self.recorder.on_state(msg)

    def sim_alive(self) -> bool:
        return (time.time() - self._last_state_t) < STATE_STALE_S

    # ---- REP -----------------------------------------------------------------------
    def _sim_meta_for_take(self, note: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"scene_meta": self._scene_meta,
                                "scene_sha": self._scene["sha"] if self._scene else None,
                                "control_mode": (self._last_state or {}).get("control_mode"),
                                "pos_scale": (self._last_state or {}).get("pos_scale"),
                                "capture_fps": self.fps, "depth_max_m": self.depth_max_m}
        for k in ("robot", "control_mode", "gripper_mode", "pos_scale", "scene_config", "layout_seed",
                  "objects", "chosen_food", "container", "config"):
            if k in self._scene_meta and meta.get(k) is None:
                meta[k] = self._scene_meta[k]
        meta.setdefault("scene_config", self.config)
        rig = self._meta_rig
        if rig is not None:
            try:
                st = self._last_state
                if st is not None and "qpos_full" in st:
                    rig.mirror(st["qpos_full"], st.get("qvel_full"))
                meta["cameras"] = {
                    c: {"pose_at_start": rig.camera_pose(c),
                        "color_render_intrinsics": rig.render_intrinsics(c, depth=False),
                        "depth_render_intrinsics": rig.render_intrinsics(c, depth=True)}
                    for c in self.cams}
            except Exception as e:  # noqa: BLE001
                meta["cameras"] = {"error": f"{type(e).__name__}: {e}"}
        return meta

    def handle(self, req: Mapping[str, Any]) -> Dict[str, Any]:
        cmd = str(req.get("cmd", ""))
        if cmd == "get_status":
            return {"ok": True, **self.status()}
        if cmd == "snapshot":
            return {"ok": True, "cam1": self._previews.get("cam1"), "cam2": self._previews.get("cam2"),
                    "recording": self.recorder.recording, "take_dir": self.recorder.take_dir}
        if cmd == "start_take":
            if self.recorder.recording:
                return {"ok": False, "msg": f"already recording {self.recorder.take_dir}"}
            if self._scene is None or not self.workers_alive():
                return {"ok": False, "msg": "scene/render workers not ready"}
            if not self.sim_alive():
                return {"ok": False, "msg": "no state from the sim in the last second"}
            note = str(req.get("note", ""))
            take_dir = self.recorder.start(self.root, note, self._sim_meta_for_take(note))
            return {"ok": True, "take_dir": take_dir, "take_index": self.recorder.take_index,
                    "msg": f"recording {os.path.basename(take_dir)}"}
        if cmd == "stop_take":
            if not self.recorder.recording:
                return {"ok": False, "msg": "not recording"}
            res = self.recorder.stop(extra_meta={"capture": self.capture_stats()})
            name = os.path.basename(res["take_dir"])
            problems = res.get("problems") or []
            msg = f"saved {name} ({res['duration_s']:.1f}s)"
            if problems:
                msg = f"saved {name} WITH PROBLEMS: " + "; ".join(problems)
            return {**res, "ok": bool(res.get("ok", True)), "msg": msg}
        if cmd == "discard_last_take":
            return self.recorder.discard_last()
        return {"ok": False, "msg": f"unknown cmd {cmd!r}"}

    def status(self) -> Dict[str, Any]:
        rs = self.recorder.status()
        return {
            **rs,
            "sim_alive": self.sim_alive(),
            "state_msgs": self._state_msgs,
            "bad_msgs": self._bad_msgs,
            "last_tick": self._last_tick,
            "scene_ready": self._scene is not None and self.workers_alive(),
            "scene_sha": self._scene["sha"] if self._scene else None,
            "workers": {c: w["proc"].is_alive() for c, w in self._workers.items()},
            "render": {c: {k: (round(v, 2) if isinstance(v, float) else v)
                           for k, v in s.items() if k != "last_t"} for c, s in self._frame_stats.items()},
            "capture": self.capture_stats(),
            "root": self.root,
            "uptime_s": round(time.time() - self.started_at, 1),
        }

    def _publish_preview(self) -> None:
        if self._pub is None:
            return
        rs = self.recorder.status()
        self._pub.send("preview", {
            "t": time.time(), "cam1": self._previews.get("cam1"), "cam2": self._previews.get("cam2"),
            "recording": rs["recording"], "take_dir": rs["take_dir"], "take_index": rs["take_index"],
            "frames": rs["frames"], "duration_s": rs["duration_s"], "rows": rs["rows"],
            "sim_alive": self.sim_alive(), "scene_ready": self._scene is not None and self.workers_alive(),
            "task_success": rs["task_success"],
            "counts": rs.get("counts", {}), "capture": self.capture_stats(),
            "problems": rs.get("problems", []),
        })

    # ---- main loop -----------------------------------------------------------------
    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        self._stop = stop_event or threading.Event()
        self._sub = ipc.Subscriber("state_pub", "state", hwm=4000)   # recorder needs EVERY message
        self._server = ipc.Server("capture_rep")
        self._pub = ipc.Publisher("preview_pub")
        frame_thread = threading.Thread(target=self._frame_loop, name="frames", daemon=True)
        frame_thread.start()
        next_preview = time.time()
        next_flush = time.time() + FLUSH_EVERY_S
        print(f"[capture] root={self.root} fps={self.fps} endpoints: state={ipc.endpoint('state_pub')} "
              f"rep={ipc.endpoint('capture_rep')} preview={ipc.endpoint('preview_pub')}")
        try:
            while not self._stop.is_set():
                now = time.time()
                if (self._scene is None or not self.workers_alive() or self._next_meta_try == 0.0) \
                        and now >= self._next_meta_try:
                    self._next_meta_try = now + 2.0
                    if not self.fetch_scene_meta():
                        print("[capture] waiting for the sim (get_scene_meta) ...")
                # drain every state message (tables need all of them, decimated inside)
                n = 0
                while n < 256 and self._sub.sock.poll(0):
                    msg = self._sub.recv(0)
                    self._on_state(msg)
                    n += 1
                self._server.poll(self.handle, timeout_ms=0)
                if now >= next_preview:
                    next_preview = now + 1.0 / self.preview_hz
                    self._publish_preview()
                if now >= next_flush:
                    # buffered rows -> datasets and h5 -> disk every second, so a hard
                    # crash loses at most ~1 s of tables (R2 item 1)
                    next_flush = now + FLUSH_EVERY_S
                    self.recorder.flush()
                if n == 0:
                    self._sub.sock.poll(2)   # sleep until state or 2 ms
        finally:
            self.close()

    def request_stop(self, why: str = "") -> None:
        """Signal-safe: only sets the stop event; ``run()`` finalises on its way out."""
        if why:
            print(f"[capture] stop requested ({why})", flush=True)
        self._stop.set()

    def close(self) -> None:
        self._stop.set()
        try:
            if self.recorder.recording:
                take_dir = self.recorder.take_dir
                print(f"[capture] shutting down with a take in progress -> finalising {take_dir}", flush=True)
                res = self.recorder.stop(extra_meta={"capture": self.capture_stats(),
                                                     "stopped_by": "capture shutdown"})
                print(f"[capture] take finalised: {res.get('take_dir')} ({res.get('duration_s')}s, "
                      f"frames {res.get('message_counts', {}).get('cam1_frames', 0)}/"
                      f"{res.get('message_counts', {}).get('cam2_frames', 0)}"
                      f"{', PROBLEMS: ' + '; '.join(res['problems']) if res.get('problems') else ''})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[capture] WARNING: finalising the take failed: {type(e).__name__}: {e}", flush=True)
        self._stop_workers()
        for s in (self._sub, self._server, self._pub, self._sim):
            try:
                if s is not None:
                    s.close()
            except Exception:  # noqa: BLE001
                pass
        if self._meta_rig is not None:
            self._meta_rig.close()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="take output root (env SIM_COLLECT_OUTPUT_ROOT)")
    ap.add_argument("--config", default=None, help="scene yaml (for the scene.build_scene fallback)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--depth", dest="record_depth", action="store_true", default=None,
                    help="also render+record depth (848x480 uint16 mm PNGs in depth.h5); default off "
                         "(yaml cameras.record_depth)")
    ap.add_argument("--no-depth", dest="record_depth", action="store_false")
    ap.add_argument("--preview-hz", type=float, default=10.0)
    ap.add_argument("--scene-xml", default=None, help="fallback MJCF file when the sim cannot be asked")
    args = ap.parse_args(argv)
    svc = CaptureService(args.root, load_config(args.config), fps=args.fps,
                         preview_hz=args.preview_hz, record_depth=args.record_depth,
                         scene_xml_path=args.scene_xml, config_path=args.config)
    print(f"[capture] depth recording: {'ON' if svc.record_depth else 'OFF (default; --depth to enable)'}", flush=True)

    # SIGTERM (launcher teardown) and SIGINT both end the loop cooperatively so an
    # in-flight take is finalised (mp4 moov, h5 close) before the process exits.
    def _on_signal(signum, _frame):
        svc.request_stop(f"signal {signal.Signals(signum).name}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)
    try:
        svc.run()
    except KeyboardInterrupt:
        pass
    finally:
        svc.close()
    print("[capture] exit", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
