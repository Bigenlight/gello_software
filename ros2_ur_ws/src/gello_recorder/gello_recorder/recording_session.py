#!/usr/bin/env python3
"""Pure-Python recording-session core for the GELLO -> UR7e diagnostic recorder.

This module owns "one session's worth of files": the shared ``vectors.h5`` (with all
nine signal tables) plus the two ``cam1.mp4`` / ``cam2.mp4`` video writers. It is the
file-I/O half of :class:`~gello_recorder.gello_ur_recorder_node.GelloUrRecorder`,
extracted VERBATIM (same table headers/column order, same per-field float precision,
same ``_bump`` counter-key strings) so that both the ROS2 node and a future
interactive GUI can share it without duplicating the logic.

Deliberately ROS-free / Qt-free / thread-free: it imports only ``h5py``, ``time`` and
the two proven sibling modules (:mod:`gello_recorder.hdf5_writer`,
:mod:`gello_recorder.video_writer`), so it is fully importable and testable
standalone -- run ``python3 recording_session.py`` for the built-in self-test.

Contract notes (why this class does NOT parse ROS messages):
  * Every ``write_*`` method takes ALREADY-COMPUTED values. There is no message
    parsing, no ``_reorder``, and no finite-difference velocity math here -- the
    caller does all of that and hands over plain lists/floats/None.
  * Every method stamps its own row's ``t_rel_s`` via :meth:`t` (this session's OWN
    relative clock), matching the node's per-callback ``f"{self._t():.4f}"`` pattern.
  * ``None`` values are passed straight through to :class:`Hdf5TableWriter`, whose
    ``_coerce`` maps them to NaN -- except where the node already pre-formats a cell
    with an f-string, in which case that exact formatting is preserved here too.
"""

import time

import h5py

from gello_recorder.hdf5_writer import open_h5_table
from gello_recorder.video_writer import Mp4FrameWriter

# Canonical UR joint count -- the node uses len(UR_JOINT_ORDER) == 6 to size every
# per-joint column block. Kept as a local constant so headers below match verbatim.
_N = 6


class RecordingSession:
    """Owns one session's ``vectors.h5`` + ``cam1.mp4`` + ``cam2.mp4`` and writes rows.

    Pure file-I/O: no ROS, no Qt, no threads. Construct once per recording, call the
    ``write_*`` methods (with already-computed values) as data arrives, then
    :meth:`close` to finalise and get back ``{"duration_s", "message_counts"}``.
    """

    def __init__(self, session_dir: str, camera_fps: float = 30.0):
        """Create ``session_dir`` and open every output file for this session.

        Opens ``session_dir/vectors.h5`` (mode 'w') with all nine tables via
        ``open_h5_table`` (same headers/column order as the node), plus two
        ``Mp4FrameWriter`` instances at ``session_dir/cam1.mp4`` and
        ``session_dir/cam2.mp4`` (both at ``camera_fps``). Records this session's OWN
        wall-clock origin in ``self.t0``; :meth:`t` is relative to THIS construction,
        not to any node/global clock, so callers must feed this session's :meth:`t`
        into its own write methods.
        """
        import os

        self.session_dir = session_dir
        os.makedirs(self.session_dir, exist_ok=True)

        # Per-table message counters (keys match the node's _bump() strings verbatim).
        self._counts = {}

        # This session's own relative-time origin.
        self.t0 = time.time()

        # --- Shared HDF5 file: one table/group per signal (headers verbatim) -----
        self._h5 = h5py.File(os.path.join(self.session_dir, "vectors.h5"), "w")
        self._sync_w = open_h5_table(
            self._h5,
            "synchronized",
            ["t_rel_s", "t_wall"]
            + [f"gello_q{i+1}" for i in range(_N)]
            + [f"gello_qd{i+1}" for i in range(_N)]
            + ["gello_grip"]
            + [f"cmd{i+1}" for i in range(_N)]
            + [f"ur_q{i+1}" for i in range(_N)]
            + [f"ur_qd{i+1}" for i in range(_N)]
            + [f"ur_eff{i+1}" for i in range(_N)]
            + ["grip_cmd", "grip_pos"]
            + ["fx", "fy", "fz", "tx", "ty", "tz"]
            + ["tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]
            + ["cam1_frame_idx", "cam2_frame_idx"],
        )
        self._gello_w = open_h5_table(
            self._h5,
            "gello_joint_states",
            ["t_rel_s"] + [f"q{i+1}" for i in range(_N)] + [f"qd{i+1}" for i in range(_N)],
        )
        self._ur_w = open_h5_table(
            self._h5,
            "ur_joint_states",
            ["t_rel_s"]
            + [f"q{i+1}" for i in range(_N)]
            + [f"qd{i+1}" for i in range(_N)]
            + [f"eff{i+1}" for i in range(_N)],
        )
        self._cmd_w = open_h5_table(
            self._h5, "command", ["t_rel_s"] + [f"cmd{i+1}" for i in range(_N)]
        )
        self._grip_w = open_h5_table(
            self._h5, "gripper", ["t_rel_s", "gello_grip", "grip_cmd", "grip_pos"]
        )
        self._wrench_w = open_h5_table(
            self._h5, "wrench", ["t_rel_s", "fx", "fy", "fz", "tx", "ty", "tz"]
        )
        self._tcp_w = open_h5_table(
            self._h5, "tcp_pose", ["t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw"]
        )
        self._cam1_w = open_h5_table(self._h5, "cam1_frames", ["t_rel_s", "frame_idx"])
        self._cam2_w = open_h5_table(self._h5, "cam2_frames", ["t_rel_s", "frame_idx"])

        # --- Video writers (each lazily opens its MP4 on the first frame) --------
        self._cam1_video = Mp4FrameWriter(
            os.path.join(self.session_dir, "cam1.mp4"), fps=camera_fps
        )
        self._cam2_video = Mp4FrameWriter(
            os.path.join(self.session_dir, "cam2.mp4"), fps=camera_fps
        )

    # ---- clock + counters ----------------------------------------------------
    def t(self) -> float:
        """Seconds since this session was constructed (``time.time() - self.t0``)."""
        return time.time() - self.t0

    def _bump(self, key):
        self._counts[key] = self._counts.get(key, 0) + 1

    def bump(self, key: str) -> None:
        """Public counter increment for callers that need topic-specific counts that
        don't map 1:1 to a single write_* call (see write_gello_grip: the node's three
        grip callbacks each bump their OWN key -- "gello_grip"/"grip_cmd"/"grip_pos" --
        even though all three write through the same gripper-table row)."""
        self._bump(key)

    # ---- one write method per table (values already computed by the caller) --
    def write_gello(self, pos: list, qd: list) -> None:
        """gello_joint_states row: ``[t()] + pos + qd``.

        ``pos`` values are appended raw (node passes raw floats); ``qd`` values may be
        ``None`` and are pre-formatted with ``f"{v:.5f}"`` exactly as the node does
        (None passes straight through to become NaN)."""
        self._bump("gello_joint_states")
        self._gello_w.writerow(
            [f"{self.t():.4f}"] + list(pos)
            + [None if v is None else f"{v:.5f}" for v in qd]
        )

    def write_gello_grip(self, gello_grip, grip_cmd, grip_pos) -> None:
        """gripper row: ``[t(), gello_grip, grip_cmd, grip_pos]`` (each ``.4f`` or None).

        Mirrors the node's ``_write_grip()``, which the three grip callbacks each call
        after updating one of the values (caller passes the CURRENT value of all three
        every time). Does NOT bump any counter itself -- ``_write_grip()`` never did
        either; each of the three grip callbacks bumps its OWN key
        ("gello_grip"/"grip_cmd"/"grip_pos") via the public :meth:`bump`, since they
        count distinct incoming topics that all happen to write through this one
        shared gripper-table row."""
        def f(v):
            return None if v is None else f"{v:.4f}"

        self._grip_w.writerow(
            [f"{self.t():.4f}", f(gello_grip), f(grip_cmd), f(grip_pos)]
        )

    def write_cmd(self, cmd: list) -> None:
        """command row: ``[t()] + cmd`` (cmd values appended raw, as the node does)."""
        self._bump("command")
        self._cmd_w.writerow([f"{self.t():.4f}"] + list(cmd))

    def write_ur(self, pos: list, vel: list, eff: list) -> None:
        """ur_joint_states row: ``[t()] + pos + vel + eff``.

        pos/vel formatted ``.6f``, eff formatted ``.4f``; any ``None`` passes through
        as NaN -- verbatim to the node's ``_on_ur``."""
        self._bump("ur_joint_states")
        self._ur_w.writerow(
            [f"{self.t():.4f}"]
            + [None if v is None else f"{v:.6f}" for v in pos]
            + [None if v is None else f"{v:.6f}" for v in vel]
            + [None if v is None else f"{v:.4f}" for v in eff]
        )

    def write_wrench(self, wrench6: list) -> None:
        """wrench row: ``[t()] + wrench6`` (6 floats fx,fy,fz,tx,ty,tz, each ``.5f``)."""
        self._bump("wrench")
        self._wrench_w.writerow(
            [f"{self.t():.4f}"] + [f"{v:.5f}" for v in wrench6]
        )

    def write_tcp(self, tcp7: list) -> None:
        """tcp_pose row: ``[t()] + tcp7`` (7 floats x,y,z,qx,qy,qz,qw, each ``.6f``)."""
        self._bump("tcp_pose")
        self._tcp_w.writerow(
            [f"{self.t():.4f}"] + [f"{v:.6f}" for v in tcp7]
        )

    def write_cam1_frame(self, jpeg_bytes: bytes) -> int:
        """Write one cam1 frame to cam1.mp4 (+ cam1_frames table). No warm-up logic.

        Any warm-up/skip decision is the caller's; this always attempts the write.
        Calls the video writer, and only on success (idx >= 0) appends ``[t(), idx]``
        to cam1_frames and bumps the ``"cam1_frames"`` counter. Returns the frame index
        (or -1 on decode failure, in which case nothing is logged)."""
        now = self.t()
        idx = self._cam1_video.write_compressed(jpeg_bytes)
        if idx >= 0:
            self._bump("cam1_frames")
            self._cam1_w.writerow([f"{now:.4f}", idx])
        return idx

    def write_cam2_frame(self, jpeg_bytes: bytes) -> int:
        """Same as :meth:`write_cam1_frame` for camera 2 / cam2.mp4 / cam2_frames."""
        now = self.t()
        idx = self._cam2_video.write_compressed(jpeg_bytes)
        if idx >= 0:
            self._bump("cam2_frames")
            self._cam2_w.writerow([f"{now:.4f}", idx])
        return idx

    def write_sample(self, gello_q, gello_qd, gello_grip, cmd, ur_q, ur_qd, ur_eff,
                     grip_cmd, grip_pos, wrench, tcp, cam1_frame_idx, cam2_frame_idx) -> None:
        """synchronized (main analysis) row. Layout/precision verbatim to ``_on_sample``.

        Columns 0/1 are ``t_rel_s`` (``.4f``) and ``t_wall`` (``time.time()`` ``.4f``);
        then gello_q ``.6f``, gello_qd ``.5f``, gello_grip ``.4f``, cmd ``.6f``,
        ur_q ``.6f``, ur_qd ``.6f``, ur_eff ``.4f``, grip_cmd/grip_pos ``.4f``,
        wrench ``.5f``, tcp ``.6f``; and finally cam1/cam2 frame indices appended RAW
        (int or None). This method does NOT bump any counter (neither does ``_on_sample``)."""
        def fmt(v, p=6):
            return None if v is None else f"{v:.{p}f}"

        row = [f"{self.t():.4f}", f"{time.time():.4f}"]
        row += [fmt(v) for v in gello_q]
        row += [fmt(v, 5) for v in gello_qd]
        row += [fmt(gello_grip, 4)]
        row += [fmt(v) for v in cmd]
        row += [fmt(v) for v in ur_q]
        row += [fmt(v) for v in ur_qd]
        row += [fmt(v, 4) for v in ur_eff]
        row += [fmt(grip_cmd, 4), fmt(grip_pos, 4)]
        row += [fmt(v, 5) for v in wrench]
        row += [fmt(v) for v in tcp]
        row += [cam1_frame_idx, cam2_frame_idx]
        self._sync_w.writerow(row)

    # ---- flush + finalise ----------------------------------------------------
    def flush(self) -> None:
        """Flush the shared h5py.File to disk (video writers are not flushed here,
        matching the node's ``_flush`` which only touches the HDF5 file)."""
        self._h5.flush()

    def close(self) -> dict:
        """Finalise the session and return ``{"duration_s", "message_counts"}``.

        The snapshot is taken BEFORE any file handle is touched so its duration/counts
        reflect the full session. Then both video writers are closed (idempotent) and
        the HDF5 file is flushed + closed. Safe to call even if construction partially
        failed or nothing was ever written -- always returns a valid dict with
        ``duration_s >= 0`` and a (possibly empty) counts dict."""
        try:
            duration_s = round(self.t(), 2)
        except Exception:  # noqa: BLE001 - t0 may be missing on partial construction
            duration_s = 0.0
        snapshot = {
            "duration_s": duration_s,
            "message_counts": dict(getattr(self, "_counts", {}) or {}),
        }

        for attr in ("_cam1_video", "_cam2_video"):
            vid = getattr(self, attr, None)
            if vid is not None:
                try:
                    vid.close()
                except Exception:  # noqa: BLE001 - best-effort on shutdown
                    pass

        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.flush()
            except Exception:  # noqa: BLE001
                pass
            try:
                h5.close()
            except Exception:  # noqa: BLE001
                pass

        return snapshot


if __name__ == "__main__":
    import json
    import os
    import tempfile

    import cv2
    import numpy as np

    scratch = tempfile.mkdtemp(prefix="recording_session_selftest_")
    session_dir = os.path.join(scratch, "session_selftest")
    print(f"self-test scratch dir : {scratch}")
    print(f"self-test session dir  : {session_dir}")

    sess = RecordingSession(session_dir, camera_fps=30.0)
    assert sess.session_dir == session_dir
    assert sess.t() >= 0.0

    # --- gello_joint_states: first row has qd=all-None (no finite-diff yet) -----
    n_gello = 4
    for i in range(n_gello):
        pos = [float(i) + 0.1 * j for j in range(_N)]
        qd = [None] * _N if i == 0 else [0.01 * (i + j) for j in range(_N)]
        sess.write_gello(pos, qd)

    # --- gripper: exercised like the three grip callbacks (all three current) ---
    # Each callback bumps its OWN key even though all three write the same row --
    # mirrors _on_gello_grip / _on_grip_cmd / _on_grip_pos in the real node.
    n_grip = 3
    for i in range(n_grip):
        sess.write_gello_grip(0.10 * i, 0.20 * i, 0.30 * i)
        sess.bump("gello_grip")
    # A row where one value is still None (e.g. grip_pos never published yet).
    sess.write_gello_grip(0.5, None, None)
    sess.bump("gello_grip")
    n_grip += 1

    # --- command ---------------------------------------------------------------
    n_cmd = 2
    for i in range(n_cmd):
        sess.write_cmd([0.5 + 0.1 * j + i for j in range(_N)])

    # --- ur_joint_states (with a None in effort to exercise pass-through) -------
    n_ur = 3
    for i in range(n_ur):
        pos = [1.0 + 0.1 * j + i for j in range(_N)]
        vel = [0.01 * j + i for j in range(_N)]
        eff = [None if (i == 0 and j == 0) else (2.0 + j + i) for j in range(_N)]
        sess.write_ur(pos, vel, eff)

    # --- wrench ----------------------------------------------------------------
    n_wrench = 2
    for i in range(n_wrench):
        sess.write_wrench([float(i) + 0.5 * k for k in range(6)])

    # --- tcp -------------------------------------------------------------------
    n_tcp = 2
    for i in range(n_tcp):
        sess.write_tcp([0.1 * k + i for k in range(7)])

    # --- camera frames: encode synthetic JPEGs (video_writer's own technique) --
    def make_jpeg(color_channel, val, w=320, h=240):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, color_channel] = val
        ok, buf = cv2.imencode(".jpg", frame)
        assert ok, "cv2.imencode failed"
        return buf.tobytes()

    n_cam1 = 5
    for i in range(n_cam1):
        idx = sess.write_cam1_frame(make_jpeg(0, 30 + i * 20))
        assert idx == i, f"cam1 idx {idx} != {i}"
    # Corrupt frame: must return -1 and NOT log / bump.
    bad = sess.write_cam1_frame(b"not a jpeg")
    assert bad == -1, f"corrupt cam1 frame should return -1, got {bad}"

    n_cam2 = 3
    for i in range(n_cam2):
        idx = sess.write_cam2_frame(make_jpeg(1, 40 + i * 30))
        assert idx == i, f"cam2 idx {idx} != {i}"

    # --- synchronized: mix in a None frame idx (before any frame) --------------
    n_sync = 4
    sync_expected_first = None
    for i in range(n_sync):
        gello_q = [0.3 + 0.1 * j + i for j in range(_N)]
        gello_qd = [None] * _N if i == 0 else [0.02 * (i + j) for j in range(_N)]
        cmd = [0.4 + 0.1 * j + i for j in range(_N)]
        ur_q = [1.5 + 0.1 * j + i for j in range(_N)]
        ur_qd = [0.03 * j + i for j in range(_N)]
        ur_eff = [5.0 + j + i for j in range(_N)]
        wrench = [0.1 * k + i for k in range(6)]
        tcp = [0.2 * k + i for k in range(7)]
        c1 = None if i == 0 else i
        c2 = None if i == 0 else i + 1
        sess.write_sample(
            gello_q, gello_qd, 0.5 * i, cmd, ur_q, ur_qd, ur_eff,
            0.6 * i, 0.7 * i, wrench, tcp, c1, c2,
        )
        if i == 1:
            # Spot-check reference for row index 1 of the synchronized table.
            sync_expected_first = {
                "gello_q1": float(f"{gello_q[0]:.6f}"),
                "ur_eff1": float(f"{ur_eff[0]:.4f}"),
                "cam1_frame_idx": float(c1),
            }

    sess.flush()

    # A couple of representative spot-check references BEFORE close.
    expected_counts = {
        "gello_joint_states": n_gello,
        "gello_grip": n_grip,
        "command": n_cmd,
        "ur_joint_states": n_ur,
        "wrench": n_wrench,
        "tcp_pose": n_tcp,
        "cam1_frames": n_cam1,   # corrupt frame did NOT bump
        "cam2_frames": n_cam2,
    }

    result = sess.close()
    print(f"close() returned      : {json.dumps(result)}")
    assert set(result.keys()) == {"duration_s", "message_counts"}, result
    assert result["duration_s"] >= 0.0, result
    assert result["message_counts"] == expected_counts, (
        f"counts {result['message_counts']} != expected {expected_counts}"
    )

    # --- Reopen vectors.h5 read-only and verify row counts + spot values --------
    h5_path = os.path.join(session_dir, "vectors.h5")
    with h5py.File(h5_path, "r") as f:
        def nrows(table):
            grp = f[table]
            cols = json.loads(grp.attrs["columns"])
            return grp[cols[0]].shape[0]

        assert nrows("gello_joint_states") == n_gello
        assert nrows("gripper") == n_grip
        assert nrows("command") == n_cmd
        assert nrows("ur_joint_states") == n_ur
        assert nrows("wrench") == n_wrench
        assert nrows("tcp_pose") == n_tcp
        assert nrows("cam1_frames") == n_cam1
        assert nrows("cam2_frames") == n_cam2
        assert nrows("synchronized") == n_sync

        # First gello row: qd must be NaN (None passed on the first call).
        assert np.isnan(f["gello_joint_states"]["qd1"][0]), "first gello qd should be NaN"
        # Second gello row: qd formatted at .5f.
        got_qd = f["gello_joint_states"]["qd1"][1]
        assert abs(got_qd - float(f"{0.01 * (1 + 0):.5f}")) < 1e-9, got_qd

        # ur eff first row first col was None -> NaN.
        assert np.isnan(f["ur_joint_states"]["eff1"][0]), "first ur eff1 should be NaN"

        # gripper row 3 had grip_cmd/grip_pos = None -> NaN, gello_grip = 0.5.
        assert abs(f["gripper"]["gello_grip"][3] - 0.5) < 1e-9
        assert np.isnan(f["gripper"]["grip_cmd"][3])
        assert np.isnan(f["gripper"]["grip_pos"][3])

        # synchronized spot-check on row index 1.
        assert sync_expected_first is not None
        assert abs(f["synchronized"]["gello_q1"][1] - sync_expected_first["gello_q1"]) < 1e-9
        assert abs(f["synchronized"]["ur_eff1"][1] - sync_expected_first["ur_eff1"]) < 1e-9
        assert abs(f["synchronized"]["cam1_frame_idx"][1] - sync_expected_first["cam1_frame_idx"]) < 1e-9
        # synchronized row 0 had cam frame idx None -> NaN.
        assert np.isnan(f["synchronized"]["cam1_frame_idx"][0])
        assert np.isnan(f["synchronized"]["cam2_frame_idx"][0])

    # --- Reopen the MP4s and confirm the frame counts match ---------------------
    for cam_path, expected in (
        (os.path.join(session_dir, "cam1.mp4"), n_cam1),
        (os.path.join(session_dir, "cam2.mp4"), n_cam2),
    ):
        cap = cv2.VideoCapture(cam_path)
        assert cap.isOpened(), f"could not open {cam_path}"
        read = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            read += 1
        cap.release()
        assert read == expected, f"{cam_path}: read {read} frames, expected {expected}"

    # --- close() must be idempotent + safe on a partially-built object ----------
    again = sess.close()
    assert again["message_counts"] == expected_counts, again

    empty = RecordingSession.__new__(RecordingSession)
    snap = empty.close()  # nothing was ever constructed
    assert snap == {"duration_s": 0.0, "message_counts": {}}, snap

    print("SELF-TEST OK")
