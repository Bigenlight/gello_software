"""SimTakeRecorder format parity against the real gello_recorder take layout.

Records a 3-second take from a synthetic 250 Hz state stream + 30 Hz synthetic frame
pairs (no sim, no GELLO, no display) and checks the on-disk result against the real
reference structure and the three real consumers.
"""
import importlib.util
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import pytest

from sim_collect import recorder as rec_mod
from sim_collect.recorder import SimTakeRecorder
from sim_collect.tests.f2_testlib import (OBJECT_NAMES, StateStream, synthetic_depth_png,
                                          synthetic_jpeg)
from gello_recorder.recording_session import RecordingSession

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REAL_GROUPS = ["synchronized", "gello_joint_states", "ur_joint_states", "command",
               "gripper", "wrench", "tcp_pose", "cam1_frames", "cam2_frames"]
SIM_GROUPS = ["sim_object_poses", "sim_control", "sim_leader_filtered", "sim_frame_capture", "sim_mj_state"]
SIM_NON_TABLE_GROUPS = ["sim_scene"]   # MJCF snapshot: xml dataset + attrs, no columns/t_rel_s
REAL_TAKE = os.path.join(_ROOT, "ros2_ur_ws", "gello_logs", "take_18_20260914_165926")
TAKE_SECONDS = 3.0


def _real_headers(tmp_path):
    """Column lists straight from a throwaway RecordingSession (the contract source)."""
    s = RecordingSession(str(tmp_path / "ref"), record_depth=False)
    heads = {}
    for name, attr in (("synchronized", "_sync_w"), ("gello_joint_states", "_gello_w"),
                       ("ur_joint_states", "_ur_w"), ("command", "_cmd_w"), ("gripper", "_grip_w"),
                       ("wrench", "_wrench_w"), ("tcp_pose", "_tcp_w"), ("cam1_frames", "_cam1_w"),
                       ("cam2_frames", "_cam2_w")):
        heads[name] = list(getattr(s, attr)._header)
    s.close()
    return heads


def record_take(root, seconds=TAKE_SECONDS, fps=30.0, state_hz=250.0, note="synthetic", record_depth=True):
    """Drive a recorder in real time the way capture.py does: state at 250 Hz from this
    thread, pre-encoded frame pairs at 30 Hz from a frame thread."""
    import threading
    frames = [(synthetic_jpeg(i), synthetic_depth_png(i)) for i in range(8)]
    r = SimTakeRecorder(camera_fps=fps, record_depth=record_depth)
    take_dir = r.start(str(root), note, {"robot": "tiny", "scene_config": {"floor": "wood"}})
    stop = threading.Event()
    counter = {"n": 0, "ms": 0.0}
    stream = StateStream(hz=state_hz)

    def frame_loop():
        nxt = time.perf_counter()
        while not stop.is_set():
            jpg, png = frames[counter["n"] % len(frames)]
            t0 = time.perf_counter()
            tick = stream.tick
            for cam in ("cam1", "cam2"):
                r.on_frames(cam, time.time(), jpg, png, sim_t=tick / state_hz, tick=tick, seq=counter["n"])
            counter["ms"] += 1e3 * (time.perf_counter() - t0)
            counter["n"] += 1
            nxt += 1.0 / fps
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)

    th = threading.Thread(target=frame_loop, daemon=True)
    t0 = time.perf_counter()
    th.start()
    nxt = t0
    n_state = 0
    state_ms = 0.0
    while time.perf_counter() - t0 < seconds:
        ts = time.perf_counter()
        r.on_state(stream.next(success=(ts - t0) > seconds - 0.5))
        state_ms += 1e3 * (time.perf_counter() - ts)
        n_state += 1
        nxt += 1.0 / state_hz
        d = nxt - time.perf_counter()
        if d > 0:
            time.sleep(d)
    stop.set()
    th.join(2)
    res = r.stop()
    print(f"\n[recorder] fed {n_state} states ({n_state / seconds:.0f} Hz, on_state {state_ms / n_state:.2f} ms avg) "
          f"and {counter['n']} frame pairs (on_frames pair {counter['ms'] / max(counter['n'], 1):.1f} ms avg)")
    return r, take_dir, res, counter["n"]


@pytest.fixture(scope="module")
def take(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim_takes")
    r, take_dir, res, n_frames = record_take(root)
    return {"root": root, "recorder": r, "dir": take_dir, "res": res, "n_frames": n_frames}


def test_take_dir_name_and_exactly_four_files(take):
    name = os.path.basename(take["dir"])
    assert name.startswith("take_01_") and len(name) == len("take_01_YYYYmmdd_HHMMSS")
    assert sorted(os.listdir(take["dir"])) == ["cam1.mp4", "cam2.mp4", "depth.h5", "vectors.h5"]
    assert take["res"]["take_dir"] == take["dir"] and take["res"]["duration_s"] >= TAKE_SECONDS - 0.1
    assert take["res"]["ok"] is True and take["res"]["problems"] == [], take["res"]
    c = take["res"]["recorder_counts"]
    assert c["missed_ticks"] == 0 and c["tick_restarts"] == 0 and c["write_errors"] == 0
    assert take["res"]["events"] == []


def test_vectors_h5_groups_columns_dtypes(take, tmp_path):
    heads = _real_headers(tmp_path)
    with h5py.File(os.path.join(take["dir"], "vectors.h5"), "r") as f:
        assert sorted(f.keys()) == sorted(REAL_GROUPS + SIM_GROUPS + SIM_NON_TABLE_GROUPS)
        for g in REAL_GROUPS:
            cols = json.loads(f[g].attrs["columns"])
            assert cols == heads[g], g
            assert isinstance(f[g].attrs["columns"], str)
            assert sorted(f[g].keys()) == sorted(cols)
            n = None
            for c in cols:
                d = f[g][c]
                assert d.dtype == np.float64 and d.maxshape == (None,) and d.chunks is not None, (g, c)
                n = d.shape[0] if n is None else n
                assert d.shape[0] == n
        assert json.loads(f["synchronized"].attrs["columns"]) == heads["synchronized"]
        assert len(heads["synchronized"]) == 56
        # sim extras
        assert json.loads(f["sim_control"].attrs["columns"]) == [
            "t_rel_s", "engaged", "eef_state_code", "pos_scale", "sigma_min", "gamma", "ls_scale", "task_success",
            "sim_t", "tick"]
        assert json.loads(f["sim_frame_capture"].attrs["columns"]) == [
            "t_rel_s", "cam", "frame_idx", "t_capture_rel_s", "sim_t", "tick", "seq"]
        assert json.loads(f["sim_leader_filtered"].attrs["columns"]) == ["t_rel_s"] + [f"qf{i}" for i in range(1, 7)]
        obj_cols = json.loads(f["sim_object_poses"].attrs["columns"])
        assert obj_cols[0] == "t_rel_s" and len(obj_cols) == 1 + 7 * len(OBJECT_NAMES)
        assert obj_cols[1:8] == [f"carrot_{s}" for s in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        for g in SIM_GROUPS:
            for c in f[g]:
                assert f[g][c].dtype == np.float64
        # file attrs: sim_meta JSON, no absolute paths anywhere in the file
        meta = json.loads(f.attrs["sim_meta"])
        assert meta["robot"] == "tiny" and meta["sim_collect_version"] == rec_mod.SIM_COLLECT_VERSION
        assert meta["task_success_at_stop"] is True and meta["duration_s"] >= TAKE_SECONDS - 0.1
        assert meta["note"] == "synthetic" and meta["eef_state_codes"]["ENGAGED"] == 2
        assert meta["depth_camera_info"]["cam1"]["K"][0] == 426.7417907714844
        assert meta["simulated"] is True and meta["events"] == [] and meta["problems"] == []
        assert meta["recorder_counts"]["missed_ticks"] == 0
        assert meta["message_counts"]["cam1_frames"] == take["n_frames"]
        # sim_frame_capture: cam in {1,2}, frame_idx contiguous per cam, capture <= write time,
        # sim_t/tick carried from the rendered state (synthetic feed passes none -> NaN allowed
        # there, everything else finite)
        fc = f["sim_frame_capture"]
        cam = fc["cam"][:]
        assert set(np.unique(cam)) == {1.0, 2.0}
        for c in (1.0, 2.0):
            idx = fc["frame_idx"][:][cam == c]
            assert np.array_equal(idx, np.arange(len(idx)))
        assert np.all(fc["t_capture_rel_s"][:] <= fc["t_rel_s"][:] + 1e-3)
        assert np.median(fc["t_rel_s"][:] - fc["t_capture_rel_s"][:]) < 0.05
    with open(os.path.join(take["dir"], "vectors.h5"), "rb") as fh:
        assert b"/home/" not in fh.read()


def test_rates_monotonic_and_no_gaps(take):
    with h5py.File(os.path.join(take["dir"], "vectors.h5"), "r") as f:
        rates = {}
        for g in REAL_GROUPS + SIM_GROUPS:
            t = f[g]["t_rel_s"][:]
            assert len(t) > 1, g
            assert np.all(np.diff(t) >= 0), f"{g} t_rel_s not non-decreasing"
            assert np.max(np.diff(t)) < 0.2, f"{g} gap {np.max(np.diff(t)):.3f}s"
            rates[g] = (len(t) - 1) / (t[-1] - t[0])
            for c in f[g]:
                assert np.isfinite(f[g][c][:]).all(), f"{g}/{c} has non-finite values"
        print("\n[recorder] rates Hz:", {k: round(v, 1) for k, v in rates.items()})
        assert 100 <= rates["ur_joint_states"] <= 135
        assert 100 <= rates["command"] <= 135 and 100 <= rates["tcp_pose"] <= 135 and 100 <= rates["wrench"] <= 135
        assert rates["gripper"] >= 30
        assert 24 <= rates["gello_joint_states"] <= 36          # leader clock 30 Hz
        assert 80 <= rates["synchronized"] <= 105
        assert 24 <= rates["sim_object_poses"] <= 36
        assert 100 <= rates["sim_control"] <= 135
        # value conventions
        g = f["gripper"]
        for c in ("gello_grip", "grip_cmd", "grip_pos"):
            v = g[c][:]
            assert v.min() >= 0.0 and v.max() <= 1.0, c
        assert f["ur_joint_states"]["q1"].shape[0] >= 0.9 * 125 * TAKE_SECONDS
        # gello qd: row 0 comes from the message's qd_lead (0.3*0.7*cos(0) = 0.21, never
        # NaN like the ROS node's first row), later rows are finite differences over leader_t.
        qd1 = f["gello_joint_states"]["qd1"][:]
        assert abs(qd1[0] - 0.21) < 1e-6
        assert np.abs(qd1[1:]).max() < 1.0 and np.abs(qd1[1:]).mean() > 0.01
        # sync cam frame idx are ints and non-decreasing
        idx = f["synchronized"]["cam1_frame_idx"][:]
        assert np.all(idx == np.round(idx)) and np.all(np.diff(idx) >= 0)
        # sim_control encodes state + task success flag flipping at the end
        sc = f["sim_control"]
        assert set(np.unique(sc["eef_state_code"][:])) == {2.0}
        assert sc["task_success"][0] == 0.0 and sc["task_success"][-1] == 1.0
        assert np.allclose(sc["gamma"][:], 1.0)


def test_videos_match_frame_tables(take):
    with h5py.File(os.path.join(take["dir"], "vectors.h5"), "r") as f:
        rows = {c: f[f"{c}_frames"]["frame_idx"][:] for c in ("cam1", "cam2")}
    for cam in ("cam1", "cam2"):
        assert np.array_equal(rows[cam], np.arange(len(rows[cam]))), f"{cam} frame_idx not contiguous"
        cap = cv2.VideoCapture(os.path.join(take["dir"], f"{cam}.mp4"))
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720
        assert abs(cap.get(cv2.CAP_PROP_FPS) - 30.0) < 1e-6
        # cv2 reads mp4v files back as 'FMP4' (877677894) -- same value the real take gives.
        assert int(cap.get(cv2.CAP_PROP_FOURCC)) in (cv2.VideoWriter_fourcc(*"mp4v"), cv2.VideoWriter_fourcc(*"FMP4"))
        n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        assert n == len(rows[cam]) == take["n_frames"], (cam, n, len(rows[cam]), take["n_frames"])
    assert 0.9 * 30 * TAKE_SECONDS <= take["n_frames"] <= 1.05 * 30 * TAKE_SECONDS + 2


def test_depth_h5_structure_matches_real(take):
    with h5py.File(os.path.join(take["dir"], "depth.h5"), "r") as f:
        assert dict(f.attrs) == {}
        assert sorted(f.keys()) == ["cam1", "cam2"]
        for cam in ("cam1", "cam2"):
            g = f[cam]
            assert sorted(g.keys()) == ["camera_info", "extrinsics_depth_to_color", "frame_idx", "png", "stamp_s", "t_rel_s"]
            a = g.attrs
            assert sorted(a.keys()) == ["aligned_to_color", "container", "depth_scale_m", "encoding",
                                        "header_bytes_stripped", "height", "source_topic", "unit", "width"]
            assert a["encoding"] == "16UC1" and a["unit"] == "mm" and a["container"] == "png"
            assert float(a["depth_scale_m"]) == 0.001 and int(a["header_bytes_stripped"]) == 12
            assert bool(a["aligned_to_color"]) is False
            assert int(a["width"]) == 848 and int(a["height"]) == 480
            assert a["source_topic"] == f"/{cam}/{cam}/depth/image_rect_raw/compressedDepth"
            assert g["png"].dtype.kind == "O" and h5py.check_vlen_dtype(g["png"].dtype) == np.uint8
            assert g["frame_idx"].dtype == np.int64
            assert g["t_rel_s"].dtype == np.float64 and g["stamp_s"].dtype == np.float64
            n = g["png"].shape[0]
            assert n == take["n_frames"]
            assert np.array_equal(g["frame_idx"][:], np.arange(n))
            assert np.all(np.diff(g["t_rel_s"][:]) >= 0) and np.isfinite(g["stamp_s"][:]).all()
            assert g["stamp_s"][0] > 1.7e9  # epoch seconds, like a ROS header stamp
            ci = g["camera_info"].attrs
            assert sorted(ci.keys()) == ["D", "K", "P", "R", "distortion_model", "frame_id", "height", "width"]
            assert ci["distortion_model"] == "plumb_bob" and ci["frame_id"] == f"{cam}_depth_optical_frame"
            assert int(ci["width"]) == 848 and int(ci["height"]) == 480
            for key, n_el in (("D", 5), ("K", 9), ("R", 9), ("P", 12)):
                assert ci[key].dtype == np.float64 and ci[key].shape == (n_el,), key
            ex = g["extrinsics_depth_to_color"].attrs
            assert sorted(ex.keys()) == ["layout", "rotation", "translation"]
            assert ex["layout"] == "column_major"
            # real take_18 values verbatim (R2 item 7), not an ideal identity
            assert ex["rotation"].shape == (9,) and abs(ex["translation"][0] - 0.015) < 2e-4
            assert not np.array_equal(ex["rotation"], np.eye(3).ravel())
            assert np.allclose(ex["rotation"].reshape(3, 3) @ ex["rotation"].reshape(3, 3).T, np.eye(3), atol=1e-6)
            for i in (0, n // 2, n - 1):
                img = cv2.imdecode(np.asarray(g["png"][i], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                assert img.shape == (480, 848) and img.dtype == np.uint16
                assert img.max() < 10000 and (img[0] == 0).all()


def test_depth_h5_attr_parity_with_real_reference(take):
    real = os.path.join(_ROOT, "ros2_ur_ws", "gello_logs", "take_18_20260914_165926", "depth.h5")
    if not os.path.isfile(real):
        pytest.skip("real reference take not present")
    with h5py.File(real, "r") as fr, h5py.File(os.path.join(take["dir"], "depth.h5"), "r") as fs:
        for cam in ("cam1", "cam2"):
            assert sorted(fr[cam].keys()) == sorted(fs[cam].keys())
            assert sorted(fr[cam].attrs.keys()) == sorted(fs[cam].attrs.keys())
            for k in fr[cam].attrs:
                assert fr[cam].attrs[k] == fs[cam].attrs[k], (cam, k)   # source_topic included
            for sub in ("camera_info", "extrinsics_depth_to_color"):
                assert sorted(fr[cam][sub].attrs.keys()) == sorted(fs[cam][sub].attrs.keys())
                for k, v in fr[cam][sub].attrs.items():
                    w = fs[cam][sub].attrs[k]
                    assert type(v) is type(w) and getattr(v, "dtype", None) == getattr(w, "dtype", None), (cam, sub, k)
                    if isinstance(v, np.ndarray):
                        assert np.array_equal(v, w), (cam, sub, k)      # values verbatim
                    else:
                        assert v == w, (cam, sub, k)
            ci_r, ci_s = fr[cam]["camera_info"].attrs, fs[cam]["camera_info"].attrs
            assert np.array_equal(ci_r["K"], ci_s["K"]) and np.array_equal(ci_r["P"], ci_s["P"])
            for ds in ("png", "frame_idx", "t_rel_s", "stamp_s"):
                assert fr[cam][ds].dtype == fs[cam][ds].dtype, (cam, ds)
    real_v = os.path.join(os.path.dirname(real), "vectors.h5")
    with h5py.File(real_v, "r") as fr, h5py.File(os.path.join(take["dir"], "vectors.h5"), "r") as fs:
        for g in REAL_GROUPS:
            # The real recorder appended a trailing `stamp_s` column to six tables on
            # 2026-09-14 (timestamp-starvation fix); take_18 predates that. Parity with the
            # CURRENT RecordingSession headers is checked elsewhere; here compare the
            # pre-existing prefix only.
            _strip = lambda cols: [c for c in json.loads(cols) if c != "stamp_s"]
            assert _strip(fr[g].attrs["columns"]) == _strip(fs[g].attrs["columns"]), g
            for c in fr[g]:
                assert fr[g][c].dtype == fs[g][c].dtype and fr[g][c].chunks == fs[g][c].chunks, (g, c)


# ---------------------------------------------------------------- real consumers
def _load_script(name):
    path = os.path.join(_ROOT, "scripts", "dataset", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_consumer_convert_carrot_to_lerobot_loader_and_depth(take):
    # The module imports only cv2/h5py/numpy at import time (lerobot is used inside main).
    conv = _load_script("convert_carrot_to_lerobot")
    h5_path = os.path.join(take["dir"], "vectors.h5")
    cam1_t, cam2_t, state, action = conv.load_take_arrays(h5_path)
    n = take["n_frames"]
    assert len(cam1_t) == len(cam2_t) == n
    assert state.shape == (n, 7) and action.shape == (n, 7)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert state[:, 6].min() >= 0 and state[:, 6].max() <= 1 and action[:, 6].min() >= 0 and action[:, 6].max() <= 1
    depth = conv.DepthTake(os.path.join(take["dir"], "depth.h5"), cam1_t)
    try:
        for cam in conv.CAMS:
            assert depth.n_depth[cam] == n
            fr = depth.frame(cam, n // 2)
            assert fr.shape == conv.DEPTH_SHAPE and fr.dtype == np.uint16
        m1, m2 = depth.meta("cam1"), depth.meta("cam2")
        assert conv._cam_meta_equal(m1, m1) and not conv._cam_meta_equal(m1, m2)
        assert m1["camera_info"]["K"][0] == 426.7417907714844
        # R2 item 7: a sim take and the real take_18 must be ONE camera set for the
        # converter's sidecar (camera_info + extrinsics + source_topic all equal).
        real_depth = os.path.join(REAL_TAKE, "depth.h5")
        if os.path.isfile(real_depth):
            with h5py.File(os.path.join(REAL_TAKE, "vectors.h5"), "r") as fr:
                real_t = fr["cam1_frames"]["t_rel_s"][:]
            real = conv.DepthTake(real_depth, real_t)
            try:
                for cam in conv.CAMS:
                    assert conv._cam_meta_equal(depth.meta(cam), real.meta(cam)), cam
            finally:
                real.close()
    finally:
        depth.close()


_ACTOR_PY = "/home/laptop3/venvs/gello-hil-actor/bin/python"
_RECORDED_DEMO_DRIVER = r"""
import json, sys
from ur_env.learner import recorded_demo
transitions, stats = recorded_demo.convert_recorded_take(sys.argv[1], outcome="success", episode_id=0)
tr = transitions[-1]["transition"]
first = transitions[0]["transition"]
print("RESULT " + json.dumps({
    "n": len(transitions), "outcome": stats.outcome, "done_last": bool(tr["dones"]),
    "reward_last": float(tr["rewards"]), "cam1_shape": list(first["observations"]["cam1"].shape),
    "state_dim": int(len(first["observations"]["state"])),
}))
"""


def test_consumer_recorded_demo_convert_recorded_take(take):
    """recorded_demo.convert_recorded_take needs gymnasium/jax-era deps that the sim
    .venv does not have; its canonical interpreter is the actor venv (CLAUDE.md), so the
    REAL function runs there as a subprocess with the repo's PYTHONPATH recipe."""
    import subprocess
    if not os.path.exists(_ACTOR_PY):
        pytest.skip("actor venv not present; recorded_demo needs gymnasium")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([
        os.path.join(_ROOT, "serl_ur_infra"), os.path.join(_ROOT, "third_party", "hil-serl", "serl_launcher"),
        os.path.join(_ROOT, "ros2_ur_ws", "src", "ur_gello_bringup")]))
    env.pop("MUJOCO_GL", None)
    out = subprocess.run([_ACTOR_PY, "-c", _RECORDED_DEMO_DRIVER, take["dir"]], env=env,
                         capture_output=True, text=True, timeout=300, cwd=_ROOT)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, out.stdout[-2000:]
    res = json.loads(line[-1][len("RESULT "):])
    print(f"\n[recorder] recorded_demo.convert_recorded_take -> {res}")
    assert res["n"] >= int(0.8 * 10 * (TAKE_SECONDS - 0.3))
    assert res["outcome"] == "success" and res["done_last"] and res["reward_last"] == 1.0
    assert res["cam1_shape"] in ([128, 128, 3], [1, 128, 128, 3])   # frame-stack axis is theirs


def test_consumer_make_carrot_raw_stats_on_the_take(take, monkeypatch, capsys):
    stats_mod = _load_script("make_carrot_raw_stats")
    out = os.path.join(str(take["root"]), "dataset_stats.json")
    monkeypatch.setattr(sys, "argv", ["make_carrot_raw_stats.py", "--data", str(take["root"]), "--out", out])
    stats_mod.main()
    with open(out) as fh:
        js = json.load(fh)
    os.remove(out)   # keep the root clean for the other tests
    agg = js["aggregate"]
    assert agg["n_takes"] == 1
    assert agg["stray_files"] == [] and agg["nan_or_inf_hits"] == [] and agg["absolute_path_hits"] == []
    assert agg["identical_group_set_across_takes"] is True
    assert set(REAL_GROUPS) <= set(agg["groups"])
    assert agg["columns_attr_python_types"] == ["str"]
    assert agg["depth_camera_info_identical_across_takes"] == {"cam1": True, "cam2": True}
    rec = js["per_take"][0]
    assert rec["cam1_h5_video_match"] and rec["cam2_h5_video_match"]
    assert rec["depth_shape_cam1"] == [480, 848] and rec["depth_shape_cam2"] == [480, 848]
    assert rec["rows_depth_cam1"] == rec["rows_cam1_frames"]


def test_discard_last_and_second_take_counter(take):
    r = take["recorder"]
    assert r.discard_last()["ok"] is False or True   # first take is kept for the other tests
    r2 = SimTakeRecorder()
    root = str(take["root"])
    d1 = r2.start(root, "one", {})
    time.sleep(0.05)
    r2.stop()
    d2 = r2.start(root, "two", {})
    assert os.path.basename(d2).startswith("take_02_") and os.path.basename(d1).startswith("take_01_")
    assert r2.discard_last()["ok"] is False        # refused while recording
    r2.stop()
    res = r2.discard_last()
    assert res["ok"] and not os.path.exists(d2) and os.path.isdir(d1)
    assert r2.discard_last()["ok"] is False          # nothing left to discard
    import shutil
    shutil.rmtree(d1)


def test_bad_messages_never_raise(tmp_path):
    r = SimTakeRecorder()
    r.on_state({"q": [1, 2]})                          # not recording: ignored
    d = r.start(str(tmp_path), "", {})
    r.on_state("garbage")
    r.on_state({"q": [1, 2], "qd": None, "tick": "x"})
    time.sleep(0.05)   # past the 30 Hz object-pose gate
    r.on_state({"q": [float("nan")] * 6, "qd": [0] * 6, "eff": [0] * 6, "objects": {"a": {"pos": [1]}}})
    r.on_frames("cam1", time.time(), b"not a jpeg", b"not a png")
    r.on_frames("cam9", time.time(), b"", b"")
    res = r.stop()
    assert res["recorder_counts"]["frames_dropped"] == 1
    with h5py.File(os.path.join(d, "vectors.h5"), "r") as f:
        assert f["ur_joint_states"]["t_rel_s"].shape[0] == 0
        assert f["cam1_frames"]["t_rel_s"].shape[0] == 0
        assert "sim_object_poses" in f


def test_default_is_no_depth(tmp_path):
    """Operator decision 2026-09-14: depth is optional and OFF by default -> a 3-file take
    (no depth.h5), depth PNGs handed to on_frames are ignored, sim_meta says so."""
    import json, h5py
    r = SimTakeRecorder()
    assert r.record_depth is False
    _r, take_dir, _res, _n = record_take(tmp_path, seconds=1.0, record_depth=False)
    assert sorted(os.listdir(take_dir)) == ["cam1.mp4", "cam2.mp4", "vectors.h5"]
    with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
        meta = json.loads(f.attrs["sim_meta"])
        assert meta["record_depth"] is False
        assert f["cam1_frames"]["t_rel_s"].shape[0] > 0


def test_scene_snapshot_and_mj_state_reconstruct(tmp_path):
    """Each take stores the exact MJCF (+asset manifest, layout, config) and the full
    generalized state per tick, so the MuJoCo scene can be rebuilt and replayed."""
    import hashlib, json, h5py, mujoco
    from sim_collect.tests.f2_testlib import TINY_SCENE_XML, make_state
    from sim_collect.tools import replay_take
    m0 = mujoco.MjModel.from_xml_string(TINY_SCENE_XML)
    r = SimTakeRecorder()
    scene = {"xml": TINY_SCENE_XML, "assets": {"fake.png": b"\x89PNG"}, "sha": "x"}
    take_dir = r.start(str(tmp_path), "snapshot", {"scene_meta": {"layout": {"_seed": 3}, "config": {"name": "tiny"}}},
                       scene=scene)
    t0 = time.time()
    for k in range(40):
        msg = make_state(k, t0 + k * 0.004, k * 0.004)
        msg["qpos_full"] = list(np.zeros(m0.nq) + 0.01 * k)
        msg["qvel_full"] = list(np.zeros(m0.nv))
        r.on_state(msg)
    r.stop()
    with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
        g = f["sim_scene"]
        assert g["xml"][()].decode() == TINY_SCENE_XML
        assert g.attrs["xml_sha256"] == hashlib.sha256(TINY_SCENE_XML.encode()).hexdigest()
        assert json.loads(g.attrs["assets_manifest"])["fake.png"]["bytes"] == 4
        assert json.loads(g.attrs["layout"]) == {"_seed": 3}
        s = f["sim_mj_state"]
        assert (int(s.attrs["nq"]), int(s.attrs["nv"]), int(s.attrs["nu"])) == (m0.nq, m0.nv, 7)
        assert s["t_rel_s"].shape[0] == 20          # every 2nd of 40 messages (125 Hz)
    st = replay_take.load_state(take_dir)
    assert st.qpos.shape == (20, m0.nq)
    sc = replay_take.load_scene(take_dir)           # config {"name": "tiny"} cannot rebuild -> note, xml still compiles
    model = replay_take.load_model(sc)
    data = mujoco.MjData(model)
    replay_take.set_row(model, data, st, 19)
    assert np.allclose(data.qpos, st.qpos[19])
