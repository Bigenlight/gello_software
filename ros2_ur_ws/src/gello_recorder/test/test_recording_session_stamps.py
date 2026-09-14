"""``RecordingSession`` header stamps + the background frame writer (ROS-free).

Two halves, both pinning the 2026-09-14 timestamp-artifact fix:

* every table fed by a STAMPED ROS message carries a trailing ``stamp_s``
  column (and the three fed by header-less message types deliberately do not),
  with every pre-existing column keeping its name and position so readers
  written before the change still work; and
* the frame writer runs off the caller's thread but must not cost a single
  frame or a single millisecond of timestamp accuracy: ``t_rel_s`` is captured
  at SUBMIT time, order is preserved per camera, ``close()`` drains before
  finalising (so "N submitted" == "N in the MP4" == "N rows"), and anything it
  cannot take is counted rather than silently lost.
"""

import json
import os
import struct
import threading
import time

import cv2
import h5py
import numpy as np
import pytest

from gello_recorder.recording_session import RecordingSession

# The tables that gained `stamp_s`, and the ones that must NOT have it.
STAMPED_TABLES = (
    "gello_joint_states", "ur_joint_states", "tcp_pose", "wrench",
    "cam1_frames", "cam2_frames",
)
UNSTAMPED_TABLES = ("command", "gripper", "synchronized")


def _jpeg(val: int, w=32, h=24) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), val, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _depth_payload(seed: int, w=48, h=32) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 4000, size=(h, w), dtype=np.uint16)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return struct.pack("<iff", 0, 0.0, 0.0) + buf.tobytes()


def _cols(session_dir, table):
    with h5py.File(os.path.join(session_dir, "vectors.h5"), "r") as f:
        return json.loads(f[table].attrs["columns"])


# --------------------------------------------------------------------------- #
# 1. schema
# --------------------------------------------------------------------------- #
def test_stamped_tables_gained_exactly_one_trailing_column(tmp_path):
    d = str(tmp_path / "s")
    RecordingSession(d).close()
    expected_tail = {
        "gello_joint_states": ["qd6", "stamp_s"],
        "ur_joint_states": ["eff6", "stamp_s"],
        "tcp_pose": ["qw", "stamp_s"],
        "wrench": ["tz", "stamp_s"],
        "cam1_frames": ["frame_idx", "stamp_s"],
        "cam2_frames": ["frame_idx", "stamp_s"],
    }
    for table, tail in expected_tail.items():
        cols = _cols(d, table)
        assert cols[-2:] == tail, table
        # APPENDED, never inserted: the head of every table is untouched.
        assert cols[0] == "t_rel_s"
        assert cols.count("stamp_s") == 1


def test_the_header_less_message_types_get_no_stamp_column(tmp_path):
    # command  <- std_msgs/Float64MultiArray  (no header)
    # gripper  <- three std_msgs/Float32      (no header)
    # synchronized <- locally sampled, not one message
    d = str(tmp_path / "s")
    RecordingSession(d).close()
    for table in UNSTAMPED_TABLES:
        assert "stamp_s" not in _cols(d, table), table


def test_pre_change_column_order_is_preserved_exactly(tmp_path):
    d = str(tmp_path / "s")
    RecordingSession(d).close()
    assert _cols(d, "ur_joint_states") == (
        ["t_rel_s"] + ["q{}".format(i) for i in range(1, 7)]
        + ["qd{}".format(i) for i in range(1, 7)]
        + ["eff{}".format(i) for i in range(1, 7)] + ["stamp_s"])
    assert _cols(d, "tcp_pose") == [
        "t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw", "stamp_s"]
    assert _cols(d, "wrench") == [
        "t_rel_s", "fx", "fy", "fz", "tx", "ty", "tz", "stamp_s"]
    assert _cols(d, "command") == [
        "t_rel_s"] + ["cmd{}".format(i) for i in range(1, 7)]


def test_no_new_tables(tmp_path):
    d = str(tmp_path / "s")
    RecordingSession(d).close()
    with h5py.File(os.path.join(d, "vectors.h5"), "r") as f:
        assert len(f.keys()) == 9


# --------------------------------------------------------------------------- #
# 2. values + the NaN rule
# --------------------------------------------------------------------------- #
def test_given_stamps_round_trip_at_epoch_precision(tmp_path):
    d = str(tmp_path / "s")
    sess = RecordingSession(d)
    # Real ROS stamps are ~1.7e9 s; a float64 holds that to ~0.2 us, which is
    # why stamp_s is written RAW and not through an f-string.
    stamps = [1757827123.123456789, 1757827123.223456789, 1757827123.323456789]
    for k, st in enumerate(stamps):
        sess.write_ur([0.1] * 6, [0.0] * 6, [1.0] * 6, stamp_s=st)
        sess.write_tcp([0.0] * 7, stamp_s=st)
        sess.write_wrench([0.0] * 6, stamp_s=st)
        sess.write_gello([0.0] * 6, [None] * 6, stamp_s=st)
    sess.close()
    with h5py.File(os.path.join(d, "vectors.h5"), "r") as f:
        for table in ("ur_joint_states", "tcp_pose", "wrench",
                      "gello_joint_states"):
            got = f[table]["stamp_s"][:]
            np.testing.assert_allclose(got, stamps, rtol=0, atol=1e-6)


def test_omitted_stamp_is_nan_not_zero(tmp_path):
    # NaN, never 0.0: a zero would read as "1970" and quietly poison any
    # now-minus-stamp arithmetic. The default must be "unknown".
    d = str(tmp_path / "s")
    sess = RecordingSession(d)
    sess.write_ur([0.0] * 6, [0.0] * 6, [0.0] * 6)
    sess.write_tcp([0.0] * 7)
    sess.write_wrench([0.0] * 6)
    sess.write_gello([0.0] * 6, [None] * 6)
    assert sess.write_cam1_frame(_jpeg(20)) == 0
    sess.close()
    with h5py.File(os.path.join(d, "vectors.h5"), "r") as f:
        for table in ("ur_joint_states", "tcp_pose", "wrench",
                      "gello_joint_states", "cam1_frames"):
            assert np.isnan(f[table]["stamp_s"][0]), table


def test_explicit_nan_stamp_stays_nan(tmp_path):
    d = str(tmp_path / "s")
    sess = RecordingSession(d)
    sess.write_ur([0.0] * 6, [0.0] * 6, [0.0] * 6, stamp_s=float("nan"))
    sess.close()
    with h5py.File(os.path.join(d, "vectors.h5"), "r") as f:
        assert np.isnan(f["ur_joint_states"]["stamp_s"][0])


def test_stamp_does_not_disturb_the_existing_values(tmp_path):
    d = str(tmp_path / "s")
    sess = RecordingSession(d)
    sess.write_ur([1.5] * 6, [0.25] * 6, [None] + [3.0] * 5, stamp_s=1.0)
    sess.close()
    with h5py.File(os.path.join(d, "vectors.h5"), "r") as f:
        g = f["ur_joint_states"]
        assert g["q1"][0] == pytest.approx(1.5)
        assert g["qd6"][0] == pytest.approx(0.25)
        assert np.isnan(g["eff1"][0])       # None still becomes NaN
        assert g["eff2"][0] == pytest.approx(3.0)
        assert g["stamp_s"][0] == pytest.approx(1.0)


def test_colour_frame_stamp_is_recorded_with_its_row(tmp_path):
    d = str(tmp_path / "s")
    sess = RecordingSession(d)
    assert sess.write_cam1_frame(_jpeg(10), stamp_s=1.7e9) == 0
    assert sess.write_cam1_frame(_jpeg(60), stamp_s=1.7e9 + 0.0333) == 1
    # A corrupt payload still logs NOTHING -- no row, no stamp, no counter.
    assert sess.write_cam1_frame(b"not a jpeg", stamp_s=1.7e9 + 0.06) == -1
    res = sess.close()
    with h5py.File(os.path.join(d, "vectors.h5"), "r") as f:
        assert list(f["cam1_frames"]["frame_idx"][:]) == [0, 1]
        np.testing.assert_allclose(
            f["cam1_frames"]["stamp_s"][:], [1.7e9, 1.7e9 + 0.0333])
    assert res["message_counts"]["cam1_frames"] == 2


# --------------------------------------------------------------------------- #
# 3. the background frame writer
# --------------------------------------------------------------------------- #
def _read_back(session_dir, name):
    with h5py.File(os.path.join(session_dir, "vectors.h5"), "r") as f:
        return (list(f[name]["frame_idx"][:]), list(f[name]["t_rel_s"][:]),
                list(f[name]["stamp_s"][:]))


def _mp4_frame_count(path):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), path
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def test_close_drains_so_every_submitted_frame_is_in_the_file(tmp_path):
    d = str(tmp_path / "take")
    sess = RecordingSession(d, camera_fps=30.0)
    n = 60
    for i in range(n):
        assert sess.submit_cam_frame(1, _jpeg(10 + i), stamp_s=1.7e9 + 0.03 * i)
    res = sess.close()

    idx, _, stamps = _read_back(d, "cam1_frames")
    assert idx == list(range(n)), "drain lost or reordered frames"
    assert res["message_counts"]["cam1_frames"] == n
    assert _mp4_frame_count(os.path.join(d, "cam1.mp4")) == n
    np.testing.assert_allclose(stamps, [1.7e9 + 0.03 * i for i in range(n)])
    assert sess.dropped_frames()["total"] == 0


def test_per_camera_ordering_is_preserved_under_interleaving(tmp_path):
    d = str(tmp_path / "take")
    sess = RecordingSession(d, camera_fps=30.0)
    for i in range(30):
        sess.submit_cam_frame(1, _jpeg(10 + i), stamp_s=float(i))
        sess.submit_cam_frame(2, _jpeg(90 - i), stamp_s=float(100 + i))
    sess.close()
    idx1, t1, st1 = _read_back(d, "cam1_frames")
    idx2, t2, st2 = _read_back(d, "cam2_frames")
    assert idx1 == idx2 == list(range(30))
    assert t1 == sorted(t1) and t2 == sorted(t2)
    assert st1 == [float(i) for i in range(30)]
    assert st2 == [float(100 + i) for i in range(30)]


def test_t_rel_s_is_the_arrival_time_not_the_write_time(tmp_path):
    """The whole point: moving the work must NOT move the timestamp.

    A writer stalled for ~0.4 s must still stamp the frame with when it
    ARRIVED -- stamping it at write time is precisely the bug that made every
    ur_joint_states row 0.900 s late.
    """
    d = str(tmp_path / "take")
    sess = RecordingSession(d, camera_fps=30.0)
    gate = threading.Event()
    real = sess._write_cam_frame

    def stalled(cam_idx, payload, t_rel_s, stamp_s):
        gate.wait(5.0)
        return real(cam_idx, payload, t_rel_s, stamp_s)

    sess._write_cam_frame = stalled
    t_submit = sess.t()
    sess.submit_cam_frame(1, _jpeg(30))
    time.sleep(0.4)
    t_release = sess.t()
    gate.set()
    sess.close()

    _, t_rel, _ = _read_back(d, "cam1_frames")
    assert len(t_rel) == 1
    assert t_rel[0] == pytest.approx(t_submit, abs=0.05)
    assert t_rel[0] < t_release - 0.3, (
        "row was stamped when it was WRITTEN, not when it arrived")


def test_submit_never_blocks_and_drops_are_counted(tmp_path):
    d = str(tmp_path / "take")
    sess = RecordingSession(d, camera_fps=30.0, frame_queue_maxsize=2)
    gate = threading.Event()
    real = sess._write_cam_frame

    def stalled(cam_idx, payload, t_rel_s, stamp_s):
        gate.wait(5.0)
        return real(cam_idx, payload, t_rel_s, stamp_s)

    sess._write_cam_frame = stalled
    payload = _jpeg(44)
    accepted = 0
    t0 = time.monotonic()
    for _ in range(40):
        if sess.submit_cam_frame(1, payload):
            accepted += 1
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, "submit_cam_frame blocked the caller ({:.3f}s)".format(
        elapsed)
    drops = sess.dropped_frames()
    assert drops["cam1"] == 40 - accepted > 0
    assert drops["total"] == drops["cam1"]
    assert drops["cam2"] == 0
    gate.set()
    sess.close()
    # Exactness holds for what was ACCEPTED: nothing accepted is ever lost.
    idx, _, _ = _read_back(d, "cam1_frames")
    assert len(idx) == accepted


def test_dropped_frames_is_zero_and_shaped_even_when_unused(tmp_path):
    sess = RecordingSession(str(tmp_path / "s"))
    try:
        assert sess.dropped_frames() == {
            "cam1": 0, "cam2": 0, "cam1_depth": 0, "cam2_depth": 0, "total": 0}
        assert sess.frame_queue_pending() == 0
    finally:
        sess.close()


def test_latest_frame_index_tracks_what_reached_the_mp4(tmp_path):
    d = str(tmp_path / "take")
    sess = RecordingSession(d, camera_fps=30.0)
    assert sess.latest_frame_index(1) is None       # before the first frame
    assert sess.latest_frame_index(2) is None
    sess.submit_cam_frame(1, _jpeg(10))
    sess.submit_cam_frame(1, _jpeg(20))
    sess.flush(drain=True)
    assert sess.latest_frame_index(1) == 1
    assert sess.latest_frame_index(2) is None
    # A corrupt frame must NOT advance it (it never reaches the file).
    sess.submit_cam_frame(1, b"not a jpeg")
    sess.flush(drain=True)
    assert sess.latest_frame_index(1) == 1
    sess.close()


def test_depth_frames_go_through_the_same_writer(tmp_path):
    d = str(tmp_path / "take")
    sess = RecordingSession(d, camera_fps=30.0, record_depth=True)
    sess.set_depth_source(1, "/cam1/cam1/depth/image_rect_raw/compressedDepth",
                          False)
    n = 12
    for i in range(n):
        assert sess.submit_cam_depth_frame(
            1, _depth_payload(700 + i), stamp_s=1.7e9 + i) is True
    res = sess.close()
    assert res["message_counts"]["cam1_depth_frames"] == n
    with h5py.File(os.path.join(d, "depth.h5"), "r") as f:
        assert f["cam1"]["png"].shape[0] == n
        assert list(f["cam1"]["frame_idx"][:]) == list(range(n))
        np.testing.assert_allclose(f["cam1"]["stamp_s"][:],
                                   [1.7e9 + i for i in range(n)])
        t = f["cam1"]["t_rel_s"][:]
        assert np.all(np.diff(t) >= 0)
    assert sess.dropped_frames()["total"] == 0


def test_submitting_depth_with_depth_off_is_a_no_op_not_a_drop(tmp_path):
    # Nothing was ever going to be written, so it is not data loss and must
    # not be reported as such.
    sess = RecordingSession(str(tmp_path / "s"), record_depth=False)
    try:
        assert sess.submit_cam_depth_frame(1, _depth_payload(1)) is False
        assert sess.dropped_frames()["total"] == 0
    finally:
        sess.close()


def test_submitting_after_close_is_refused_and_counted(tmp_path):
    d = str(tmp_path / "take")
    sess = RecordingSession(d, camera_fps=30.0)
    sess.submit_cam_frame(1, _jpeg(10))
    sess.close()
    assert sess.submit_cam_frame(1, _jpeg(20)) is False
    assert sess.dropped_frames()["cam1"] == 1
    # ...and the closed session's files were NOT touched again.
    idx, _, _ = _read_back(d, "cam1_frames")
    assert idx == [0]


def test_vector_rows_written_while_frames_are_in_flight(tmp_path):
    """The two threads share vectors.h5; every row of both must survive."""
    d = str(tmp_path / "take")
    sess = RecordingSession(d, camera_fps=30.0)
    payload = _jpeg(77)
    n_frames, n_rows = 40, 300
    for i in range(n_frames):
        sess.submit_cam_frame(1, payload, stamp_s=float(i))
    for i in range(n_rows):
        sess.write_ur([float(i)] * 6, [0.0] * 6, [0.0] * 6, stamp_s=float(i))
        sess.write_tcp([0.0] * 7, stamp_s=float(i))
    res = sess.close()
    assert res["message_counts"]["ur_joint_states"] == n_rows
    assert res["message_counts"]["tcp_pose"] == n_rows
    assert res["message_counts"]["cam1_frames"] == n_frames
    with h5py.File(os.path.join(d, "vectors.h5"), "r") as f:
        assert f["ur_joint_states"]["q1"].shape[0] == n_rows
        assert f["cam1_frames"]["frame_idx"].shape[0] == n_frames
        np.testing.assert_allclose(f["ur_joint_states"]["stamp_s"][:],
                                   [float(i) for i in range(n_rows)])
    assert _mp4_frame_count(os.path.join(d, "cam1.mp4")) == n_frames


def test_flush_does_not_drain_by_default(tmp_path):
    # The periodic flush runs on the spin thread; blocking it on a frame
    # backlog would re-create the starvation this change removes.
    d = str(tmp_path / "take")
    sess = RecordingSession(d, camera_fps=30.0)
    gate = threading.Event()
    real = sess._write_cam_frame

    def stalled(cam_idx, payload, t_rel_s, stamp_s):
        gate.wait(5.0)
        return real(cam_idx, payload, t_rel_s, stamp_s)

    sess._write_cam_frame = stalled
    for _ in range(5):
        sess.submit_cam_frame(1, _jpeg(30))
    t0 = time.monotonic()
    sess.flush()
    assert time.monotonic() - t0 < 0.5
    assert sess.frame_queue_pending() > 0
    gate.set()
    sess.close()


# --------------------------------------------------------------------------- #
# 4. ragged-row repair (a signal can land inside writerow)
# --------------------------------------------------------------------------- #
def test_close_trims_a_half_written_row(tmp_path):
    """Ctrl-C on the headless recorder is delivered to the thread that is
    writing, so it can interrupt writerow() between two columns. Observed live
    2026-09-14: ur_joint_states ended with t_rel_s/q1/q2 at 4474 rows and its
    other 17 columns at 4473, which makes the last row unreadable by anything
    that takes its row count from column 0."""
    d = str(tmp_path / "take")
    sess = RecordingSession(d)
    for i in range(5):
        sess.write_ur([float(i)] * 6, [0.0] * 6, [0.0] * 6, stamp_s=float(i))
    # Simulate the tear: extend only the first two columns, as an interrupt
    # part-way through writerow() would.
    grp = sess._h5["ur_joint_states"]
    for col in ("t_rel_s", "q1"):
        grp[col].resize((6,))
        grp[col][5] = 99.0
    sess.close()

    with h5py.File(os.path.join(d, "vectors.h5"), "r") as f:
        cols = json.loads(f["ur_joint_states"].attrs["columns"])
        lengths = {c: f["ur_joint_states"][c].shape[0] for c in cols}
        assert set(lengths.values()) == {5}, lengths
        # The five COMPLETE rows are untouched.
        np.testing.assert_allclose(f["ur_joint_states"]["q1"][:],
                                   [float(i) for i in range(5)])


def test_finalize_is_a_no_op_on_a_rectangular_table(tmp_path):
    from gello_recorder.hdf5_writer import open_h5_table

    path = str(tmp_path / "t.h5")
    with h5py.File(path, "w") as f:
        w = open_h5_table(f, "t", ["a", "b"])
        for i in range(3):
            w.writerow([i, i])
        assert w.finalize() == 0
        assert f["t"]["a"].shape[0] == f["t"]["b"].shape[0] == 3
