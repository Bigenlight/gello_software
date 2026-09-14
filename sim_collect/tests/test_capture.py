"""capture.py end-to-end against a FAKE sim: a thread publishes state at 250 Hz and
answers get_scene_meta; the service (in a thread) spawns the two render workers,
records a 3-second take through capture_rep and publishes previews.

Needs a DISPLAY (render workers use MUJOCO_GL=glfw). Scene resolution / status /
robustness tests below run without one.
"""
import os
import sys
import threading
import time

import cv2
import h5py
import numpy as np
import pytest

from sim_collect import capture, ipc
from sim_collect.tests.conftest import needs_display
from sim_collect.tests.f2_testlib import TINY_SCENE_XML, StateStream

TAKE_SECONDS = 3.0


class FakeSim:
    """state_pub at 250 Hz + sim_rep answering get_scene_meta / get_status."""

    def __init__(self, scene_xml=TINY_SCENE_XML, with_xml=True):
        import zmq
        deadline = time.time() + 30.0
        self.pub = self.srv = None
        while True:      # another owner's test may hold the shared tcp ports for a moment
            try:
                self.pub = ipc.Publisher("state_pub")
                self.srv = ipc.Server("sim_rep")
                break
            except zmq.ZMQError as e:
                if self.pub is not None:
                    self.pub.close()
                    self.pub = None
                if time.time() > deadline:
                    raise
                print(f"[FakeSim] {e}; retrying bind")
                time.sleep(1.0)
        self.scene_xml = scene_xml
        self.with_xml = with_xml
        self.stream = StateStream()
        self.stop = threading.Event()
        self.restart_at_tick = None
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def handle(self, req):
        if req.get("cmd") == "get_scene_meta":
            meta = {"ok": True, "model": "f2_tiny", "robot": "tiny", "objects": ["carrot", "pot"],
                    "layout_seed": 7, "config": {"floor": {"texture": "/home/nobody/wood.png"}}}
            if self.with_xml:
                meta["scene_xml"] = self.scene_xml
                meta["scene_assets"] = {}
            return meta
        return {"ok": True, "cmd": req.get("cmd")}

    def _loop(self):
        period = 1.0 / 250.0
        nxt = time.perf_counter()
        while not self.stop.is_set():
            self.srv.poll(self.handle, timeout_ms=0)
            if self.restart_at_tick is not None and self.stream.tick >= self.restart_at_tick:
                self.stream = StateStream()          # tick jumps back to 0 -> "restart"
                self.restart_at_tick = None
            self.pub.send("state", self.stream.next())
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    def close(self):
        self.stop.set()
        self.th.join(2)
        self.pub.close()
        self.srv.close()


def _run_service(svc):
    stop = threading.Event()
    th = threading.Thread(target=svc.run, args=(stop,), daemon=True)
    th.start()
    return stop, th


def test_resolve_scene_contract_keys_and_fallbacks(tmp_path):
    s = capture.resolve_scene({"scene_xml": TINY_SCENE_XML, "scene_assets": {"a.png": b"\x00"}})
    assert s["xml"] == TINY_SCENE_XML and s["assets"] == {"a.png": b"\x00"} and s["xml_path"] is None
    p = tmp_path / "scene.xml"
    p.write_text(TINY_SCENE_XML)
    s2 = capture.resolve_scene({"scene_xml_path": str(p)})
    assert s2["xml_path"] == str(p) and s2["sha"] != s["sha"]   # assets differ -> different sha
    s3 = capture.resolve_scene(None, None, scene_xml_path=str(p))
    assert s3["sha"] == s2["sha"]
    assert capture.resolve_scene({"foo": 1}) is None or hasattr(capture, "scene")
    assert capture._coerce_scene_result((TINY_SCENE_XML, {})) is not None
    assert capture._coerce_scene_result(TINY_SCENE_XML)["sha"] == s2["sha"]
    assert capture._coerce_scene_result(None) is None


def test_resolve_scene_rebuilds_f1_scene_from_get_scene_meta_reply():
    """sim_main's real reply carries config + config_path + layout + xml_sha256 (no XML);
    [B] must rebuild the identical MJCF through scene.build_scene."""
    scene_mod = pytest.importorskip("sim_collect.scene")
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "carrot_in_pot_sim.yaml")
    if not os.path.isfile(cfg_path):
        pytest.skip("F1 config not present")
    cfg = scene_mod.SceneConfig.load(cfg_path)
    layout = scene_mod.sample_layout(cfg, seed=3)
    built = scene_mod.build_scene(cfg, layout)
    import hashlib
    reply = {"ok": True, "msg": "scene meta",
             "meta": {"config": cfg.raw, "config_path": cfg_path, "layout": layout, "layout_seed": 3,
                      "xml_sha256": hashlib.sha256(built.xml.encode()).hexdigest()}}
    s = capture.resolve_scene(reply)
    assert s is not None and s["xml"] == built.xml and not s.get("sha_mismatch")
    assert set(s["assets"]) == set(built.assets)
    # the rebuilt scene compiles here and has the four contract cameras
    from sim_collect import cameras
    rig = cameras.CameraRig(s["xml"], s["assets"])
    for cam in ("cam1", "cam2"):
        assert rig.depth_camera_name(cam) == f"{cam}_depth"
    rig.close()


def test_service_handles_requests_without_workers(tmp_path):
    svc = capture.CaptureService(str(tmp_path), fps=30.0, record_depth=True)   # parity test: 4-file take
    st = svc.handle({"cmd": "get_status"})
    assert st["ok"] and st["recording"] is False and st["scene_ready"] is False and st["sim_alive"] is False
    assert svc.handle({"cmd": "start_take"})["ok"] is False
    assert svc.handle({"cmd": "stop_take"})["ok"] is False
    assert svc.handle({"cmd": "discard_last_take"})["ok"] is False
    assert svc.handle({"cmd": "nope"})["ok"] is False
    snap = svc.handle({"cmd": "snapshot"})
    assert snap["ok"] and snap["cam1"] is None
    svc._on_state("not a dict")
    svc._on_state({"tick": 5})
    svc._on_state({"tick": 100})
    svc._on_state({"tick": 3})          # restart -> schedules a scene re-check
    assert svc._bad_msgs == 1 and svc._state_msgs == 3 and svc._next_meta_try == 0.0
    svc.close()


@needs_display
def test_capture_records_a_take_end_to_end(tmp_path):
    sim = FakeSim()
    svc = capture.CaptureService(str(tmp_path), fps=30.0, preview_hz=10.0, record_depth=True)
    stop, th = _run_service(svc)
    cli = ipc.Client("capture_rep", timeout_ms=3000)
    prev = ipc.Subscriber("preview_pub", "preview")
    try:
        assert ipc.wait_for(cli, 20.0)
        deadline = time.time() + 40.0
        while time.time() < deadline:
            st = cli.call("get_status")
            if st.get("ok") and st.get("scene_ready") and st.get("sim_alive") \
                    and st["render"]["cam1"]["n"] > 5 and st["render"]["cam2"]["n"] > 5:
                break
            time.sleep(0.2)
        else:
            pytest.fail(f"capture never became ready: {st}")
        assert st["workers"] == {"cam1": True, "cam2": True}

        rep = cli.call("start_take", note="e2e")
        assert rep["ok"], rep
        take_dir = rep["take_dir"]
        assert os.path.basename(take_dir).startswith("take_01_")
        assert cli.call("start_take")["ok"] is False           # already recording
        assert cli.call("discard_last_take")["ok"] is False    # refused while recording
        time.sleep(TAKE_SECONDS)
        mid = cli.call("get_status")
        assert mid["recording"] and mid["take_dir"] == take_dir and mid["frames"]["cam1"] > 0
        rep = cli.call("stop_take")
        assert rep["ok"], rep
        assert rep["duration_s"] >= TAKE_SECONDS - 0.2
        # the only problem a loaded machine may legitimately produce is the fps time-warp
        # notice, and then ONLY if the take-level rate really was below 97 % of nominal
        fps_problems = [p for p in rep["problems"] if "captured at" in p]
        assert fps_problems == rep["problems"] and rep["events"] == [], rep
        assert rep["recorder_counts"]["missed_ticks"] == 0
        counts = rep["message_counts"]
        cap = mid["capture"]
        print(f"\n[capture] stop_take -> {counts} render={mid['render']} capture={cap}")
        # The machine may be busy (software GL is shared with whatever else renders), so
        # assert the REPORTING is consistent rather than that this box is idle: the
        # achieved rate is measured, plausible, and fps_ok reflects it honestly.
        for cam in ("cam1", "cam2"):
            assert 15.0 <= cap["achieved_fps"][cam] <= 31.5, cap
            assert cap["worker_queue_drops"][cam] == 0
        slow = min(cap["achieved_fps"].values()) < 0.97 * 30.0
        assert cap["fps_ok"] is (not slow), cap
        if slow:
            print(f"[capture] NOTE: workers below 30 Hz during this run ({cap['achieved_fps']}), reporting verified")
        # whole-take rate from capture timestamps must agree with the frame count / duration
        with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
            import json
            meta_first = json.loads(f.attrs["sim_meta"])
            fc = f["sim_frame_capture"]
            for k, cam in ((1.0, "cam1"), (2.0, "cam2")):
                tc = fc["t_capture_rel_s"][:][fc["cam"][:] == k]
                expect = (len(tc) - 1) / (tc[-1] - tc[0])
                assert abs(meta_first["achieved_fps_take"][cam] - expect) < 0.05, (cam, meta_first["achieved_fps_take"], expect)
                assert 15.0 <= expect <= 31.5
            really_slow = min(meta_first["achieved_fps_take"].values()) < 0.97 * 30.0
            assert bool(fps_problems) == really_slow, (fps_problems, meta_first["achieved_fps_take"])
            assert meta_first["problems"] == rep["problems"]
        print(f"[capture] achieved_fps_take={meta_first['achieved_fps_take']} problems={rep['problems']}")

        # preview stream: jpeg previews of 320x180 + flags
        msg = None
        for _ in range(30):
            msg = prev.recv(500)
            if msg and msg.get("cam1") and msg.get("cam2"):
                break
        assert msg and msg["recording"] is False
        img = cv2.imdecode(np.frombuffer(msg["cam1"], np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (180, 320, 3)
        snap = cli.call("snapshot")
        assert snap["ok"] and snap["cam2"]

        # on disk: 4 files, frame counts vs render rate, tables at rate, depth decodes
        assert sorted(os.listdir(take_dir)) == ["cam1.mp4", "cam2.mp4", "depth.h5", "vectors.h5"]
        with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
            n1 = f["cam1_frames"]["t_rel_s"].shape[0]
            n2 = f["cam2_frames"]["t_rel_s"].shape[0]
            t_ur = f["ur_joint_states"]["t_rel_s"][:]
            t_sync = f["synchronized"]["t_rel_s"][:]
            assert np.all(np.diff(t_ur) >= 0) and np.max(np.diff(t_ur)) < 0.2
            for g in ("cam1_frames", "cam2_frames", "gripper", "gello_joint_states", "command",
                      "tcp_pose", "wrench", "sim_object_poses", "sim_control"):
                t = f[g]["t_rel_s"][:]
                assert len(t) > 2 and np.max(np.diff(t)) < 0.2, g
            rate_ur = (len(t_ur) - 1) / (t_ur[-1] - t_ur[0])
            rate_c1 = (n1 - 1) / (f["cam1_frames"]["t_rel_s"][-1] - f["cam1_frames"]["t_rel_s"][0])
            print(f"[capture] cam1 {n1} frames ({rate_c1:.1f} Hz), cam2 {n2}, ur rows {len(t_ur)} "
                  f"({rate_ur:.1f} Hz), sync rows {len(t_sync)}")
            assert n1 >= 0.8 * 30 * TAKE_SECONDS and n2 >= 0.8 * 30 * TAKE_SECONDS
            assert rate_c1 >= 24.0 and rate_ur >= 100.0
            assert len(t_sync) >= 0.7 * 100 * TAKE_SECONDS
            assert f["sim_object_poses"]["carrot_x"][0] == pytest.approx(-0.45, abs=1e-6)
            import json
            meta = json.loads(f.attrs["sim_meta"])
            assert meta["robot"] == "tiny" and meta["scene_meta"]["layout_seed"] == 7
            assert "/home/" not in json.dumps(meta)
            assert meta["cameras"]["cam1"]["pose_at_start"]["fixed"] is True
        with h5py.File(os.path.join(take_dir, "depth.h5"), "r") as f:
            for cam in ("cam1", "cam2"):
                n = f[cam]["png"].shape[0]
                assert n >= 0.8 * 30 * TAKE_SECONDS
                img = cv2.imdecode(np.asarray(f[cam]["png"][n // 2], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                assert img.shape == (480, 848) and img.dtype == np.uint16 and 0 < img.max() < 10000
        for cam in ("cam1", "cam2"):
            cap = cv2.VideoCapture(os.path.join(take_dir, f"{cam}.mp4"))
            assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720
            ok, fr = cap.read()
            cap.release()
            assert ok and fr.std() > 5.0, "rendered video is flat"

        # discard the finished take, then a sim restart must not break the service
        rep = cli.call("discard_last_take")
        assert rep["ok"] and not os.path.exists(take_dir)
        sim.restart_at_tick = sim.stream.tick + 10
        time.sleep(1.0)
        st = cli.call("get_status")
        assert st["ok"] and st["sim_alive"] and st["scene_ready"]
        rep = cli.call("start_take", note="after restart")
        assert rep["ok"], rep
        time.sleep(0.5)
        sim.restart_at_tick = sim.stream.tick + 10      # restart DURING the take -> event
        time.sleep(0.7)
        rep = cli.call("stop_take")
        assert rep["ok"] and rep["message_counts"].get("cam1_frames", 0) > 5
        assert os.path.basename(rep["take_dir"]).startswith("take_02_")
        kinds = [e["kind"] for e in rep["events"]]
        assert "sim_restart" in kinds and "tick_backwards" in kinds, rep["events"]
        assert rep["recorder_counts"]["tick_restarts"] == 1
        assert any("restarted" in p for p in rep["problems"]), rep["problems"]
        import json
        with h5py.File(os.path.join(rep["take_dir"], "vectors.h5"), "r") as f:
            meta = json.loads(f.attrs["sim_meta"])
            assert [e["kind"] for e in meta["events"]] == kinds
            assert set(meta["achieved_fps"]) == {"cam1", "cam2"}
            assert meta["capture"]["nominal_fps"] == 30.0
            sc = f["sim_control"]
            assert np.isfinite(sc["sim_t"][:]).all() and np.isfinite(sc["tick"][:]).all()
            assert (np.diff(sc["tick"][:]) < 0).sum() == 1       # the restart is visible in the table
    finally:
        stop.set()
        th.join(10)
        cli.close()
        prev.close()
        sim.close()


def _children_of(pid):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as fh:
                fields = fh.read().split()
            if int(fields[3]) == pid:
                out.append(int(d))
        except (OSError, IndexError, ValueError):
            continue
    return out


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except OSError:
        return False


@needs_display
def test_sigterm_finalises_take_and_leaves_no_orphans(tmp_path):
    """R2 items 1+2: SIGTERM mid-take must produce a valid take (mp4 moov, h5 closed)
    and no render worker may outlive the capture process."""
    import signal
    import subprocess
    scene_xml = tmp_path / "tiny.xml"
    scene_xml.write_text(TINY_SCENE_XML)
    root = tmp_path / "takes"
    env = dict(os.environ, SIM_COLLECT_IPC=os.environ.get("SIM_COLLECT_IPC", "tcp"),
               MUJOCO_GL="glfw", PYTHONUNBUFFERED="1")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sim = FakeSim()
    proc = subprocess.Popen([sys.executable, "-m", "sim_collect.capture", "--root", str(root),
                             "--scene-xml", str(scene_xml), "--depth"], cwd=repo, env=env,   # parity: 4-file take
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cli = ipc.Client("capture_rep", timeout_ms=3000)
    out = ""
    try:
        assert ipc.wait_for(cli, 30.0)
        deadline = time.time() + 40.0
        while time.time() < deadline:
            st = cli.call("get_status")
            if st.get("ok") and st.get("scene_ready") and st["render"]["cam1"]["n"] > 5 and st["render"]["cam2"]["n"] > 5:
                break
            time.sleep(0.2)
        else:
            pytest.fail(f"capture never ready: {st}")
        rep = cli.call("start_take", note="sigterm")
        assert rep["ok"], rep
        take_dir = rep["take_dir"]
        time.sleep(2.0)
        assert cli.call("get_status")["frames"]["cam1"] > 20
        workers = _children_of(proc.pid)
        assert len(workers) >= 2, f"expected 2 render workers, saw {workers}"
        t_kill = time.time()
        proc.send_signal(signal.SIGTERM)
        try:
            out = proc.communicate(timeout=30)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            out = proc.communicate()[0]
            pytest.fail("capture did not exit within 30 s of SIGTERM:\n" + out[-3000:])
        t_exit = time.time() - t_kill
        assert proc.returncode == 0, out[-3000:]
        assert "take finalised" in out and take_dir in out, out[-2000:]
        time.sleep(3.0)
        orphans = [p for p in workers if _alive(p)]
        assert orphans == [], f"render workers still alive 3 s after exit: {orphans}\n{out[-2000:]}"
        # the take is valid: closed h5 files, mp4 with a moov atom, rows consistent
        assert sorted(os.listdir(take_dir)) == ["cam1.mp4", "cam2.mp4", "depth.h5", "vectors.h5"]
        with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
            n1 = f["cam1_frames"]["frame_idx"].shape[0]
            assert n1 > 20 and f["ur_joint_states"]["t_rel_s"].shape[0] > 100
            import json
            meta = json.loads(f.attrs["sim_meta"])
            assert meta["stopped_by"] == "capture shutdown" and meta["duration_s"] > 1.5
        with h5py.File(os.path.join(take_dir, "depth.h5"), "r") as f:
            assert f["cam1"]["png"].shape[0] == n1
        cap = cv2.VideoCapture(os.path.join(take_dir, "cam1.mp4"))
        n_video = 0
        while cap.read()[0]:
            n_video += 1
        cap.release()
        assert n_video == n1, (n_video, n1)
        print(f"\n[capture] SIGTERM -> exit in {t_exit:.2f}s rc=0, {n1} frames finalised, workers {workers} all dead")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        for p in _children_of(proc.pid) if proc.poll() is None else []:
            os.kill(p, signal.SIGKILL)
        cli.close()
        sim.close()


def test_capture_default_no_depth(tmp_path):
    """Depth is optional and OFF by default (2026-09-14): the service and its recorder
    default to record_depth=False; yaml cameras.record_depth / CLI --depth flip it."""
    svc = capture.CaptureService(str(tmp_path), fps=30.0)
    try:
        assert svc.record_depth is False and svc.recorder.record_depth is False
    finally:
        svc.close()
    svc = capture.CaptureService(str(tmp_path), {"cameras": {"record_depth": True}}, fps=30.0)
    try:
        assert svc.record_depth is True
    finally:
        svc.close()
    svc = capture.CaptureService(str(tmp_path), {"cameras": {"record_depth": True}}, fps=30.0, record_depth=False)
    try:
        assert svc.record_depth is False   # CLI override wins
    finally:
        svc.close()
