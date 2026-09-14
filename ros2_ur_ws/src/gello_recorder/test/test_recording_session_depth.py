"""``RecordingSession`` depth integration (ROS-free).

Two things are pinned. First, ``record_depth=False`` (the default every existing
caller uses) is byte-for-byte the old behaviour: no ``depth.h5``, ``vectors.h5``
keeps exactly its nine tables, and every depth method is a no-op returning
``-1`` / ``None`` without touching the counters. Second, ``record_depth=True``
produces a ``depth.h5`` whose rows stamp ``t_rel_s`` from the same session clock
as ``cam1_frames`` / ``synchronized``, so offline alignment is by ``t_rel_s``
alone -- ``vectors.h5`` gains no table or column for it.
"""

import json
import os
import struct
import time

import cv2
import h5py
import numpy as np
import pytest

from gello_recorder.depth_writer import depth_meta, read_depth_frame
from gello_recorder.recording_session import RecordingSession

NINE_TABLES = sorted([
    "synchronized", "gello_joint_states", "ur_joint_states", "command",
    "gripper", "wrench", "tcp_pose", "cam1_frames", "cam2_frames",
])


def _depth_payload(seed: int, w=48, h=32):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 4000, size=(h, w), dtype=np.uint16)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return struct.pack("<iff", 0, 0.0, 0.0) + buf.tobytes(), arr


def _jpeg(val: int, w=32, h=24) -> bytes:
    frame = np.full((h, w, 3), val, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _vectors_tables(session_dir) -> list:
    with h5py.File(os.path.join(session_dir, "vectors.h5"), "r") as f:
        return sorted(f.keys())


# ---- depth OFF -------------------------------------------------------------------
def test_default_is_depth_off(tmp_path):
    sess = RecordingSession(str(tmp_path / "s"))
    try:
        assert sess.record_depth is False
    finally:
        sess.close()


def test_depth_off_creates_no_depth_file_and_methods_are_noops(tmp_path):
    session_dir = str(tmp_path / "s")
    sess = RecordingSession(session_dir, camera_fps=30.0, record_depth=False)
    payload, _ = _depth_payload(1)

    assert sess.write_cam1_depth_frame(payload) == -1
    assert sess.write_cam2_depth_frame(payload, stamp_s=1.0) == -1
    assert sess.set_depth_source(1, "/cam1/cam1/depth/image_rect_raw/compressedDepth", False) is None
    assert sess.set_depth_camera_info(
        1, width=1, height=1, distortion_model="", D=[], K=[0] * 9, R=[0] * 9,
        P=[0] * 12, frame_id="",
    ) is None
    assert sess.set_depth_extrinsics(2, [0] * 9, [0] * 3) is None
    # A colour frame still works exactly as before.
    assert sess.write_cam1_frame(_jpeg(50)) == 0

    sess.flush()
    result = sess.close()

    assert set(result) == {"duration_s", "message_counts"}
    assert result["message_counts"] == {"cam1_frames": 1}
    assert "cam1_depth_frames" not in result["message_counts"]
    assert "cam2_depth_frames" not in result["message_counts"]
    assert not os.path.exists(os.path.join(session_dir, "depth.h5"))
    assert sorted(os.listdir(session_dir)) == ["cam1.mp4", "vectors.h5"]
    assert _vectors_tables(session_dir) == NINE_TABLES


def test_close_is_safe_on_partial_object_with_depth_attr_missing():
    empty = RecordingSession.__new__(RecordingSession)
    assert empty.close() == {"duration_s": 0.0, "message_counts": {}}
    assert empty.record_depth is False


# ---- depth ON --------------------------------------------------------------------
@pytest.fixture
def depth_session(tmp_path):
    session_dir = str(tmp_path / "take")
    sess = RecordingSession(session_dir, camera_fps=30.0, record_depth=True)
    yield sess, session_dir
    sess.close()


def test_depth_on_writes_both_cams_and_counts(depth_session):
    sess, session_dir = depth_session
    assert sess.record_depth is True

    sess.set_depth_source(1, "/cam1/cam1/depth/image_rect_raw/compressedDepth", False)
    sess.set_depth_source(2, "/cam2/cam2/aligned_depth_to_color/image_raw/compressedDepth", True)
    sess.set_depth_camera_info(
        1, width=48, height=32, distortion_model="plumb_bob", D=[0.0] * 5,
        K=[300, 0, 24, 0, 300, 16, 0, 0, 1], R=list(np.eye(3).ravel()),
        P=[300, 0, 24, 0, 0, 300, 16, 0, 0, 0, 1, 0], frame_id="cam1_depth_optical_frame",
    )
    sess.set_depth_extrinsics(1, list(np.eye(3).ravel()), [0.015, 0.0, 0.0])

    ref1 = []
    for i in range(4):
        payload, arr = _depth_payload(10 + i)
        assert sess.write_cam1_depth_frame(payload, stamp_s=1.7e9 + i) == i
        ref1.append(arr)
    for i in range(2):
        payload, _ = _depth_payload(20 + i)
        assert sess.write_cam2_depth_frame(payload) == i
    # Corrupt payload: -1, and the counter must not move.
    assert sess.write_cam1_depth_frame(b"\x00" * 12 + b"garbage") == -1

    sess.flush()
    result = sess.close()

    assert set(result) == {"duration_s", "message_counts"}
    assert result["message_counts"] == {"cam1_depth_frames": 4, "cam2_depth_frames": 2}

    depth_path = os.path.join(session_dir, "depth.h5")
    assert os.path.exists(depth_path)
    with h5py.File(depth_path, "r") as f:
        assert sorted(f.keys()) == ["cam1", "cam2"]
        assert f["cam1"]["png"].shape[0] == 4
        assert f["cam2"]["png"].shape[0] == 2
        assert list(f["cam1"]["frame_idx"][:]) == [0, 1, 2, 3]
        np.testing.assert_allclose(f["cam1"]["stamp_s"][:], [1.7e9 + i for i in range(4)])
        assert np.isnan(f["cam2"]["stamp_s"][0])
        for i, arr in enumerate(ref1):
            assert np.array_equal(read_depth_frame(f, "cam1", i), arr)

    m1 = depth_meta(depth_path, "cam1")
    assert (m1["width"], m1["height"]) == (48, 32)
    assert m1["aligned_to_color"] is False
    assert m1["camera_info"]["frame_id"] == "cam1_depth_optical_frame"
    assert m1["extrinsics_depth_to_color"]["translation"] == pytest.approx([0.015, 0, 0])
    assert depth_meta(depth_path, "cam2")["aligned_to_color"] is True


def test_depth_does_not_touch_vectors_h5(depth_session):
    sess, session_dir = depth_session
    payload, _ = _depth_payload(1)
    sess.write_cam1_depth_frame(payload)
    sess.write_cam1_frame(_jpeg(80))
    sess.close()

    assert _vectors_tables(session_dir) == NINE_TABLES
    with h5py.File(os.path.join(session_dir, "vectors.h5"), "r") as f:
        cols = json.loads(f["synchronized"].attrs["columns"])
        assert not any("depth" in c for c in cols)
        # cam1_frames gained a trailing `stamp_s` (the COLOUR CompressedImage's
        # own header stamp, 2026-09-14) -- that column has nothing to do with
        # depth, and this test's point is that depth adds nothing to vectors.h5.
        assert json.loads(f["cam1_frames"].attrs["columns"]) == [
            "t_rel_s", "frame_idx", "stamp_s"
        ]


def test_depth_t_rel_s_shares_session_clock(depth_session):
    sess, session_dir = depth_session
    t_before = sess.t()
    sess.write_cam1_frame(_jpeg(10))
    t_rels = []
    for i in range(5):
        payload, _ = _depth_payload(30 + i)
        sess.write_cam1_depth_frame(payload)
        t_rels.append(sess.t())
        time.sleep(0.002)
    sess.write_cam2_frame(_jpeg(20))
    result = sess.close()

    with h5py.File(os.path.join(session_dir, "depth.h5"), "r") as f:
        t_d = f["cam1"]["t_rel_s"][:]
    with h5py.File(os.path.join(session_dir, "vectors.h5"), "r") as f:
        t_c1 = f["cam1_frames"]["t_rel_s"][0]
        t_c2 = f["cam2_frames"]["t_rel_s"][0]

    assert len(t_d) == 5
    assert np.all(np.diff(t_d) >= 0)
    assert t_d[0] >= t_before
    # Same origin as the colour tables: bracketed by the two colour frames written
    # before and after (vectors.h5 rounds to .4f, hence the small tolerance).
    assert t_d[0] >= t_c1 - 1e-4
    assert t_d[-1] <= t_c2 + 1e-4
    # Within the session duration reported by close() (rounded to 2 dp).
    assert t_d[-1] <= result["duration_s"] + 0.01
    assert result["message_counts"]["cam1_depth_frames"] == 5


def test_depth_close_idempotent_and_stats_stable(depth_session):
    sess, _ = depth_session
    payload, _ = _depth_payload(1)
    sess.write_cam2_depth_frame(payload)
    first = sess.close()
    second = sess.close()
    assert first["message_counts"] == second["message_counts"] == {"cam2_depth_frames": 1}
    assert first["duration_s"] >= 0.0
