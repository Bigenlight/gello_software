#!/usr/bin/env python3
"""Pure-Python recording-session core for the GELLO -> UR7e diagnostic recorder.

This module owns "one session's worth of files": the shared ``vectors.h5`` (with all
nine signal tables) plus the two ``cam1.mp4`` / ``cam2.mp4`` video writers, and --
only when ``record_depth=True`` -- a sibling ``depth.h5`` holding the RealSense
compressed-depth PNGs (:mod:`gello_recorder.depth_writer`). It is the file-I/O half
of :class:`~gello_recorder.gello_ur_recorder_node.GelloUrRecorder`, extracted
VERBATIM (same table headers/column order, same per-field float precision, same
``_bump`` counter-key strings) so that both the ROS2 node and a future interactive
GUI can share it without duplicating the logic.

Deliberately ROS-free / Qt-free: it imports only ``h5py``, ``time``, ``threading``
and the four proven sibling modules (:mod:`gello_recorder.hdf5_writer`,
:mod:`gello_recorder.video_writer`, :mod:`gello_recorder.depth_writer`,
:mod:`gello_recorder.spin_health`), so it is fully importable and testable
standalone -- run ``python3 recording_session.py`` for the built-in self-test.

NO LONGER thread-free (2026-09-14). Two threads may now touch one session:

  * the caller's thread (in the recorders: the single rclpy spin thread), which
    writes every VECTOR row synchronously -- those are a handful of float
    appends and cost nothing; and
  * ONE background frame writer (:class:`~gello_recorder.spin_health.FrameWriteQueue`,
    created lazily by the first :meth:`submit_cam_frame` /
    :meth:`submit_cam_depth_frame`) which performs the expensive per-frame I/O:
    JPEG decode + MP4 encode (measured 9.3 + 6.9 ms per 1280x720 frame on
    laptop3) and the ~740 kB depth-PNG HDF5 append (1.4 ms).

That split is the fix for the timestamp artifact described in
:mod:`gello_recorder.spin_health`: with both cameras' colour AND depth handled
inline, the spin thread was doing ~1.6 s of work per wall-clock second, its
round rate fell to 60-69 Hz, and every topic publishing faster than that was
read out of a permanently full queue -- silently back-dating the robot rows.

Locking is deliberately narrow: ``_vec_lock`` around ``vectors.h5`` row appends
(and its flush/close), ``_depth_lock`` around ``depth.h5``, ``_counts_lock``
around the message counters. The MP4 encode holds no lock at all -- each
``Mp4FrameWriter`` is touched only by the frame writer thread (and by
:meth:`close`, after the queue has drained).

HEADER STAMPS (2026-09-14). Every table fed by a STAMPED ROS message carries a
trailing ``stamp_s`` column: float64 seconds taken from ``msg.header.stamp``,
NaN when absent. It is APPENDED, so every pre-existing column keeps its name,
its order and its meaning and old readers are unaffected. The tables that have
it: ``gello_joint_states``, ``ur_joint_states``, ``tcp_pose``, ``wrench``,
``cam1_frames``, ``cam2_frames`` (``depth.h5`` already had one). The tables that
do NOT, because their ROS message type carries no header at all:

  * ``command``   <- ``std_msgs/Float64MultiArray``
  * ``gripper``   <- three ``std_msgs/Float32`` topics
  * ``synchronized`` -- a locally sampled wide row, not one message.

For those three, ``t_rel_s`` (arrival time) is the only timestamp that exists,
which is exactly why the queue depths were shrunk as well: the recorder cannot
repair an old sample it cannot recognise.

Depth is strictly additive: with ``record_depth=False`` (the default) NOTHING about
the on-disk output changes -- ``vectors.h5`` keeps exactly its nine tables, no
``depth.h5`` is created, and the depth ``write_*``/``set_*`` methods are no-ops that
return ``-1`` / ``None`` so callers never have to branch. With depth on, the depth
rows stamp ``t_rel_s`` from the SAME :meth:`t` origin as ``cam1_frames`` /
``synchronized``, so offline alignment is by ``t_rel_s``; ``vectors.h5`` gains no
table or column for it.

Contract notes (why this class does NOT parse ROS messages):
  * Every ``write_*`` method takes ALREADY-COMPUTED values. There is no message
    parsing, no ``_reorder``, and no finite-difference velocity math here -- the
    caller does all of that and hands over plain lists/floats/None.
  * Every method stamps its own row's ``t_rel_s`` via :meth:`t` (this session's OWN
    relative clock), matching the node's per-callback ``f"{self._t():.4f}"`` pattern.
    The ONE exception is a frame handed to :meth:`submit_cam_frame` /
    :meth:`submit_cam_depth_frame`: its ``t_rel_s`` is captured at SUBMIT time
    (i.e. in the ROS callback, on arrival) and carried through the queue, so the
    writer thread's own clock never reaches the data.
  * ``None`` values are passed straight through to :class:`Hdf5TableWriter`, whose
    ``_coerce`` maps them to NaN -- except where the node already pre-formats a cell
    with an f-string, in which case that exact formatting is preserved here too.
"""

import threading
import time

import h5py

from gello_recorder.depth_writer import DepthH5Writer
from gello_recorder.hdf5_writer import open_h5_table
from gello_recorder.spin_health import FRAME_QUEUE_MAXSIZE, FrameWriteQueue
from gello_recorder.video_writer import Mp4FrameWriter

_NAN = float("nan")

#: Work-item kinds carried through the frame queue.
_KIND_COLOR = {1: "cam1", 2: "cam2"}
_KIND_DEPTH = {1: "cam1_depth", 2: "cam2_depth"}

# cam_idx (1 or 2) -> depth.h5 group name / counter-key prefix.
_DEPTH_CAM_NAMES = {1: "cam1", 2: "cam2"}

# Canonical UR joint count -- the node uses len(UR_JOINT_ORDER) == 6 to size every
# per-joint column block. Kept as a local constant so headers below match verbatim.
_N = 6


class RecordingSession:
    """Owns one session's ``vectors.h5`` + ``cam1.mp4`` + ``cam2.mp4`` (+ optional
    ``depth.h5``) and writes rows.

    Pure file-I/O: no ROS, no Qt, no threads. Construct once per recording, call the
    ``write_*`` methods (with already-computed values) as data arrives, then
    :meth:`close` to finalise and get back ``{"duration_s", "message_counts"}``.
    """

    def __init__(self, session_dir: str, camera_fps: float = 30.0,
                 record_depth: bool = False,
                 frame_queue_maxsize: int = FRAME_QUEUE_MAXSIZE):
        """Create ``session_dir`` and open every output file for this session.

        Opens ``session_dir/vectors.h5`` (mode 'w') with all nine tables via
        ``open_h5_table`` (same headers/column order as the node), plus two
        ``Mp4FrameWriter`` instances at ``session_dir/cam1.mp4`` and
        ``session_dir/cam2.mp4`` (both at ``camera_fps``). Records this session's OWN
        wall-clock origin in ``self.t0``; :meth:`t` is relative to THIS construction,
        not to any node/global clock, so callers must feed this session's :meth:`t`
        into its own write methods.

        ``record_depth=True`` additionally opens ``session_dir/depth.h5`` through
        :class:`DepthH5Writer` (groups ``cam1`` / ``cam2``); otherwise ``self._depth``
        is ``None`` and no depth file is ever created.
        """
        import os

        self.session_dir = session_dir
        os.makedirs(self.session_dir, exist_ok=True)

        # Depth is opt-in; keep the attribute present either way so close() and the
        # depth no-op methods can test it without hasattr gymnastics.
        self._depth = None

        # --- threading (see the module docstring) ---------------------------
        # The frame writer is created LAZILY by the first submit_*: a caller
        # that only ever uses the synchronous write_* methods (the self-test,
        # the unit tests, any offline user) gets exactly the old single-threaded
        # behaviour and no thread at all.
        self._frame_queue_maxsize = int(frame_queue_maxsize)
        self._frames = None
        self._closed = False
        self._frames_lock = threading.Lock()
        self._vec_lock = threading.Lock()
        self._depth_lock = threading.Lock()
        self._counts_lock = threading.Lock()
        # Frames lost because the writer queue was full (or already closing),
        # per stream. Reported by dropped_frames(); NOT folded into
        # message_counts, whose shape is part of the on-disk contract.
        self._dropped = {"cam1": 0, "cam2": 0, "cam1_depth": 0, "cam2_depth": 0}

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
        # NOTE on every header below: ``stamp_s`` is APPENDED LAST on purpose.
        # Existing columns keep their names, their order and their meaning, so a
        # reader written before 2026-09-14 sees an unchanged table and a reader
        # written after it gets the publisher's own capture time as well.
        self._gello_w = open_h5_table(
            self._h5,
            "gello_joint_states",
            ["t_rel_s"] + [f"q{i+1}" for i in range(_N)]
            + [f"qd{i+1}" for i in range(_N)] + ["stamp_s"],
        )
        self._ur_w = open_h5_table(
            self._h5,
            "ur_joint_states",
            ["t_rel_s"]
            + [f"q{i+1}" for i in range(_N)]
            + [f"qd{i+1}" for i in range(_N)]
            + [f"eff{i+1}" for i in range(_N)]
            + ["stamp_s"],
        )
        self._cmd_w = open_h5_table(
            self._h5, "command", ["t_rel_s"] + [f"cmd{i+1}" for i in range(_N)]
        )
        self._grip_w = open_h5_table(
            self._h5, "gripper", ["t_rel_s", "gello_grip", "grip_cmd", "grip_pos"]
        )
        self._wrench_w = open_h5_table(
            self._h5, "wrench",
            ["t_rel_s", "fx", "fy", "fz", "tx", "ty", "tz", "stamp_s"]
        )
        self._tcp_w = open_h5_table(
            self._h5, "tcp_pose",
            ["t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw", "stamp_s"]
        )
        self._cam1_w = open_h5_table(
            self._h5, "cam1_frames", ["t_rel_s", "frame_idx", "stamp_s"])
        self._cam2_w = open_h5_table(
            self._h5, "cam2_frames", ["t_rel_s", "frame_idx", "stamp_s"])

        # --- Video writers (each lazily opens its MP4 on the first frame) --------
        self._cam1_video = Mp4FrameWriter(
            os.path.join(self.session_dir, "cam1.mp4"), fps=camera_fps
        )
        self._cam2_video = Mp4FrameWriter(
            os.path.join(self.session_dir, "cam2.mp4"), fps=camera_fps
        )

        # --- Optional depth store (compressedDepth PNGs, one group per cam) ------
        if record_depth:
            self._depth = DepthH5Writer(
                os.path.join(self.session_dir, "depth.h5"),
                cams=tuple(_DEPTH_CAM_NAMES[i] for i in sorted(_DEPTH_CAM_NAMES)),
            )

    @property
    def record_depth(self) -> bool:
        """True when this session was opened with ``record_depth=True`` (a
        ``depth.h5`` exists and the depth methods actually write)."""
        return getattr(self, "_depth", None) is not None

    # ---- clock + counters ----------------------------------------------------
    def t(self) -> float:
        """Seconds since this session was constructed (``time.time() - self.t0``)."""
        return time.time() - self.t0

    def _bump(self, key):
        # Bumped from the caller's thread (vector tables) AND from the frame
        # writer thread (cam*_frames / cam*_depth_frames), hence the lock.
        with self._counts_lock:
            self._counts[key] = self._counts.get(key, 0) + 1

    def bump(self, key: str) -> None:
        """Public counter increment for callers that need topic-specific counts that
        don't map 1:1 to a single write_* call (see write_gello_grip: the node's three
        grip callbacks each bump their OWN key -- "gello_grip"/"grip_cmd"/"grip_pos" --
        even though all three write through the same gripper-table row)."""
        self._bump(key)

    # ---- one write method per table (values already computed by the caller) --
    def write_gello(self, pos: list, qd: list, stamp_s: float = _NAN) -> None:
        """gello_joint_states row: ``[t()] + pos + qd + [stamp_s]``.

        ``pos`` values are appended raw (node passes raw floats); ``qd`` values may be
        ``None`` and are pre-formatted with ``f"{v:.5f}"`` exactly as the node does
        (None passes straight through to become NaN). ``stamp_s`` is the
        ``sensor_msgs/JointState`` ``header.stamp`` in float64 seconds (NaN when the
        message carried no stamp) and is written RAW, never through an f-string:
        an epoch second formatted to 6 dp loses the sub-microsecond digits."""
        self._bump("gello_joint_states")
        with self._vec_lock:
            self._gello_w.writerow(
                [f"{self.t():.4f}"] + list(pos)
                + [None if v is None else f"{v:.5f}" for v in qd]
                + [stamp_s]
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

        with self._vec_lock:
            self._grip_w.writerow(
                [f"{self.t():.4f}", f(gello_grip), f(grip_cmd), f(grip_pos)]
            )

    def write_cmd(self, cmd: list) -> None:
        """command row: ``[t()] + cmd`` (cmd values appended raw, as the node does).

        NO ``stamp_s`` column: the source topic is ``std_msgs/Float64MultiArray``,
        which has no header. ``t_rel_s`` (arrival) is the only timestamp that
        exists for this table -- see the module docstring."""
        self._bump("command")
        with self._vec_lock:
            self._cmd_w.writerow([f"{self.t():.4f}"] + list(cmd))

    def write_ur(self, pos: list, vel: list, eff: list,
                 stamp_s: float = _NAN) -> None:
        """ur_joint_states row: ``[t()] + pos + vel + eff + [stamp_s]``.

        pos/vel formatted ``.6f``, eff formatted ``.4f``; any ``None`` passes through
        as NaN -- verbatim to the node's ``_on_ur``. ``stamp_s`` is the driver's own
        ``header.stamp`` (float64 seconds, NaN if absent): the ONE column that makes
        a queue-delayed row detectable offline."""
        self._bump("ur_joint_states")
        with self._vec_lock:
            self._ur_w.writerow(
                [f"{self.t():.4f}"]
                + [None if v is None else f"{v:.6f}" for v in pos]
                + [None if v is None else f"{v:.6f}" for v in vel]
                + [None if v is None else f"{v:.4f}" for v in eff]
                + [stamp_s]
            )

    def write_wrench(self, wrench6: list, stamp_s: float = _NAN) -> None:
        """wrench row: ``[t()] + wrench6 + [stamp_s]`` (6 floats fx,fy,fz,tx,ty,tz,
        each ``.5f``; ``stamp_s`` = ``WrenchStamped.header.stamp``, NaN if absent)."""
        self._bump("wrench")
        with self._vec_lock:
            self._wrench_w.writerow(
                [f"{self.t():.4f}"] + [f"{v:.5f}" for v in wrench6] + [stamp_s]
            )

    def write_tcp(self, tcp7: list, stamp_s: float = _NAN) -> None:
        """tcp_pose row: ``[t()] + tcp7 + [stamp_s]`` (7 floats x,y,z,qx,qy,qz,qw,
        each ``.6f``; ``stamp_s`` = ``PoseStamped.header.stamp``, NaN if absent)."""
        self._bump("tcp_pose")
        with self._vec_lock:
            self._tcp_w.writerow(
                [f"{self.t():.4f}"] + [f"{v:.6f}" for v in tcp7] + [stamp_s]
            )

    def write_cam1_frame(self, jpeg_bytes: bytes, stamp_s: float = _NAN) -> int:
        """Write one cam1 frame to cam1.mp4 (+ cam1_frames table), SYNCHRONOUSLY.

        Unchanged contract: any warm-up/skip decision is the caller's, this always
        attempts the write, and only on success (idx >= 0) appends
        ``[t(), idx, stamp_s]`` to cam1_frames and bumps the ``"cam1_frames"``
        counter. Returns the frame index (or -1 on decode failure, in which case
        nothing is logged).

        The ROS recorders do NOT call this -- they call :meth:`submit_cam_frame`,
        which does the same work on the background writer thread. This stays for
        offline callers, the self-test and the unit tests, where "decode failed"
        must be answerable in the return value."""
        return self._write_cam_frame(1, jpeg_bytes, self.t(), stamp_s)

    def write_cam2_frame(self, jpeg_bytes: bytes, stamp_s: float = _NAN) -> int:
        """Same as :meth:`write_cam1_frame` for camera 2 / cam2.mp4 / cam2_frames."""
        return self._write_cam_frame(2, jpeg_bytes, self.t(), stamp_s)

    def _write_cam_frame(self, cam_idx: int, jpeg_bytes: bytes,
                         t_rel_s: float, stamp_s: float) -> int:
        """Shared body: decode+encode into camN.mp4, then log the row.

        ``t_rel_s`` is passed IN, never taken here, because on the async path this
        runs on the writer thread long after the frame arrived -- see the module
        docstring. The MP4 encode deliberately holds no lock (one writer thread per
        file); only the two-column table append takes ``_vec_lock``."""
        video = self._cam1_video if cam_idx == 1 else self._cam2_video
        table = self._cam1_w if cam_idx == 1 else self._cam2_w
        idx = video.write_compressed(jpeg_bytes)
        if idx >= 0:
            self._bump(f"cam{cam_idx}_frames")
            with self._vec_lock:
                table.writerow([f"{t_rel_s:.4f}", idx, stamp_s])
        return idx

    # ---- background frame writer (the spin thread never does frame I/O) -----
    def _ensure_frame_writer(self) -> "FrameWriteQueue | None":
        """Create the single background writer on first use (None once closed)."""
        with self._frames_lock:
            if self._closed:
                return None
            frames = self._frames
            if frames is None:
                frames = FrameWriteQueue(
                    self._handle_frame_item,
                    maxsize=self._frame_queue_maxsize,
                    name="recording-frame-writer",
                )
                self._frames = frames
            return frames

    def _handle_frame_item(self, item) -> None:
        """Writer-thread entry point for one ``(kind, payload, t_rel_s, stamp_s)``."""
        kind, payload, t_rel_s, stamp_s = item
        if kind == "cam1":
            self._write_cam_frame(1, payload, t_rel_s, stamp_s)
        elif kind == "cam2":
            self._write_cam_frame(2, payload, t_rel_s, stamp_s)
        elif kind == "cam1_depth":
            self._write_depth_frame_at(1, payload, t_rel_s, stamp_s)
        elif kind == "cam2_depth":
            self._write_depth_frame_at(2, payload, t_rel_s, stamp_s)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown frame kind {kind!r}")

    def _submit(self, kind: str, drop_key: str, payload: bytes,
                stamp_s: float) -> bool:
        """Capture ``t_rel_s`` NOW (arrival) and hand the frame to the writer.

        Never blocks and never raises. Returns False when the frame was dropped
        (queue full, or the session is closing); the drop is counted per stream
        and surfaced by :meth:`dropped_frames`."""
        t_rel_s = self.t()
        frames = self._ensure_frame_writer()
        if frames is None or not frames.submit((kind, payload, t_rel_s, stamp_s)):
            with self._counts_lock:
                self._dropped[drop_key] = self._dropped.get(drop_key, 0) + 1
            return False
        return True

    def submit_cam_frame(self, cam_idx: int, jpeg_bytes: bytes,
                         stamp_s: float = _NAN) -> bool:
        """Queue one colour frame for the background writer. Non-blocking."""
        kind = _KIND_COLOR[cam_idx]
        return self._submit(kind, kind, jpeg_bytes, stamp_s)

    def submit_cam_depth_frame(self, cam_idx: int, data: bytes,
                               stamp_s: float = _NAN) -> bool:
        """Queue one ``compressedDepth`` payload for the background writer.

        A no-op returning False when depth is off -- and it does NOT count as a
        drop, because nothing was ever going to be written."""
        if self._depth is None:
            return False
        kind = _KIND_DEPTH[cam_idx]
        return self._submit(kind, kind, data, stamp_s)

    def latest_frame_index(self, cam_idx: int):
        """Index of the most recent frame actually written to camN.mp4, or None
        before the first one.

        THE authoritative answer on the async path: ``submit_cam_frame`` returns
        before anything is encoded, so a caller that needs "which MP4 frame is
        current" (the ``synchronized`` table's cross-reference) must read it back
        from the writer, which only advances on a successful decode. Reading a
        plain int from another thread needs no lock."""
        video = self._cam1_video if cam_idx == 1 else self._cam2_video
        count = getattr(video, "frame_count", 0)
        if not isinstance(count, int) or count <= 0:
            return None
        return count - 1

    def dropped_frames(self) -> dict:
        """``{"cam1": n, "cam2": n, "cam1_depth": n, "cam2_depth": n, "total": n}``.

        Frames the writer queue refused because it was full (the machine could
        not keep up) or because the session was closing. ZERO is the expected
        value; anything else is a real, counted data loss and belongs in the
        take's summary."""
        lock = getattr(self, "_counts_lock", None)
        if lock is None:
            return {"cam1": 0, "cam2": 0, "cam1_depth": 0, "cam2_depth": 0,
                    "total": 0}
        with lock:
            out = dict(self._dropped)
        out["total"] = sum(out.values())
        return out

    def frame_queue_pending(self) -> int:
        """Frames submitted but not yet written (0 in steady state)."""
        frames = getattr(self, "_frames", None)
        return 0 if frames is None else frames.pending

    # ---- depth (all no-ops returning -1 / None when record_depth is False) ---
    def _write_depth_frame(self, cam_idx: int, data: bytes, stamp_s: float) -> int:
        """Shared body of the two public depth writers. Stamps ``t_rel_s`` from THIS
        session's :meth:`t` (same origin as ``cam1_frames``) BEFORE the write, hands
        the raw ``CompressedImage.data`` to :class:`DepthH5Writer`, and only on success
        (``idx >= 0``) bumps ``"<cam>_depth_frames"`` -- a corrupt payload leaves both
        the file and the counter untouched, exactly like :meth:`write_cam1_frame`."""
        if self._depth is None:
            return -1
        return self._write_depth_frame_at(cam_idx, data, self.t(), stamp_s)

    def _write_depth_frame_at(self, cam_idx: int, data: bytes,
                              t_rel_s: float, stamp_s: float) -> int:
        """Same, with ``t_rel_s`` passed in (the async path captured it on arrival)."""
        if self._depth is None:
            return -1
        cam = _DEPTH_CAM_NAMES[cam_idx]
        with self._depth_lock:
            idx = self._depth.write_compressed_depth(cam, data, t_rel_s, stamp_s)
        if idx >= 0:
            self._bump(f"{cam}_depth_frames")
        return idx

    def write_cam1_depth_frame(self, data: bytes, stamp_s: float = _NAN) -> int:
        """Append one cam1 ``compressedDepth`` payload (12-byte header + PNG, i.e.
        ``CompressedImage.data`` verbatim) to ``depth.h5``. ``stamp_s`` is the ROS
        header stamp in seconds (NaN if unknown). Returns the 0-based depth frame
        index, ``-1`` on a corrupt payload, and ``-1`` (no-op) when depth is off."""
        return self._write_depth_frame(1, data, stamp_s)

    def write_cam2_depth_frame(self, data: bytes, stamp_s: float = _NAN) -> int:
        """Same as :meth:`write_cam1_depth_frame` for camera 2."""
        return self._write_depth_frame(2, data, stamp_s)

    def set_depth_source(self, cam_idx: int, topic: str, aligned_to_color: bool) -> None:
        """Record the depth topic name / aligned flag for cam ``cam_idx`` (1 or 2).
        No-op when depth is off."""
        if self._depth is None:
            return None
        with self._depth_lock:
            self._depth.set_source(_DEPTH_CAM_NAMES[cam_idx], topic, aligned_to_color)
        return None

    def set_depth_camera_info(self, cam_idx: int, **kw) -> None:
        """Forward a ``sensor_msgs/CameraInfo`` (as keyword fields ``width, height,
        distortion_model, D, K, R, P, frame_id``) to
        :meth:`DepthH5Writer.set_camera_info` for cam ``cam_idx``. No-op when depth
        is off."""
        if self._depth is None:
            return None
        with self._depth_lock:
            self._depth.set_camera_info(_DEPTH_CAM_NAMES[cam_idx], **kw)
        return None

    def set_depth_extrinsics(self, cam_idx: int, rotation, translation) -> None:
        """Forward depth->colour extrinsics (rotation[9] column-major, translation[3]
        m) to :meth:`DepthH5Writer.set_extrinsics_depth_to_color`. No-op when depth
        is off."""
        if self._depth is None:
            return None
        with self._depth_lock:
            self._depth.set_extrinsics_depth_to_color(
                _DEPTH_CAM_NAMES[cam_idx], rotation, translation
            )
        return None

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
        with self._vec_lock:
            self._sync_w.writerow(row)

    # ---- flush + finalise ----------------------------------------------------
    def flush(self, drain: bool = False, drain_timeout: float = 1.0) -> None:
        """Flush the shared h5py.File (and ``depth.h5`` when depth is on) to disk.
        Video writers are not flushed here, matching the node's ``_flush`` which only
        touches the HDF5 file(s).

        ``drain`` is OFF by default and that is deliberate. The periodic flush timer
        runs on the rclpy spin thread, and blocking it until a backlog of frames has
        been encoded would re-create -- on the flush timer -- exactly the starvation
        this whole change removes. Crash-resilience does not need an empty queue;
        FINALISATION does, and :meth:`close` always drains. Pass ``drain=True`` from a
        caller that owns its own thread (or from a test) to get an exact snapshot."""
        if drain:
            frames = getattr(self, "_frames", None)
            if frames is not None:
                frames.drain(drain_timeout)
        with self._vec_lock:
            self._h5.flush()
        if self._depth is not None:
            with self._depth_lock:
                self._depth.flush()

    def close(self) -> dict:
        """Finalise the session and return ``{"duration_s", "message_counts"}``.

        ``duration_s`` is read first (it is the recording length, not the
        finalisation length), then the background frame writer is DRAINED and joined
        so every accepted frame is on disk, and only then are the counts snapshotted.
        Both video writers are closed (idempotent), the depth store (if any) is
        closed best-effort, and the HDF5 file is flushed + closed. Safe to call even
        if construction partially failed or nothing was ever written -- always returns
        a valid dict with ``duration_s >= 0`` and a (possibly empty) counts dict. The
        dict SHAPE never changes; with depth on the counts merely gain
        ``"cam1_depth_frames"`` / ``"cam2_depth_frames"``. Dropped frames are
        deliberately NOT in here -- see :meth:`dropped_frames`."""
        try:
            duration_s = round(self.t(), 2)
        except Exception:  # noqa: BLE001 - t0 may be missing on partial construction
            duration_s = 0.0

        # DRAIN FIRST, then snapshot the counts. The queue is emptied before any
        # file handle is touched, so every frame that was accepted is on disk and
        # counted: "N submitted == N written" is exact, and the MP4 frame count
        # still equals the cam*_frames row count.
        frames = getattr(self, "_frames", None)
        if frames is not None:
            try:
                if not frames.close():
                    print("[RecordingSession] WARNING: frame writer did not drain "
                          "within the timeout; {} item(s) may be lost".format(
                              frames.pending))
            except Exception:  # noqa: BLE001 - best-effort on shutdown
                pass

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

        depth = getattr(self, "_depth", None)
        if depth is not None:
            try:
                depth.close()
            except Exception:  # noqa: BLE001 - best-effort on shutdown
                pass

        with getattr(self, "_frames_lock", threading.Lock()):
            self._closed = True

        # Make every table rectangular before the file is sealed. A signal can
        # land inside writerow() -- Ctrl-C on the headless recorder is delivered
        # to the very thread that is writing -- leaving the last row with values
        # in some columns and not others. See Hdf5TableWriter.finalize.
        for writer in (getattr(self, "_sync_w", None), getattr(self, "_gello_w", None),
                       getattr(self, "_ur_w", None), getattr(self, "_cmd_w", None),
                       getattr(self, "_grip_w", None), getattr(self, "_wrench_w", None),
                       getattr(self, "_tcp_w", None), getattr(self, "_cam1_w", None),
                       getattr(self, "_cam2_w", None)):
            if writer is None:
                continue
            try:
                dropped = writer.finalize()
            except Exception:  # noqa: BLE001 - best-effort on shutdown
                continue
            if dropped:
                print("[RecordingSession] NOTE: discarded {} half-written row(s) "
                      "from a table (interrupted mid-writerow)".format(dropped))

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

    # --- Depth OFF (the pass above): no depth.h5, depth methods are no-ops --------
    assert not sess.record_depth
    assert not os.path.exists(os.path.join(session_dir, "depth.h5")), (
        "depth.h5 must NOT be created when record_depth is False"
    )
    assert "cam1_depth_frames" not in result["message_counts"]
    assert "cam2_depth_frames" not in result["message_counts"]

    # --- Depth ON pass: same session API + depth.h5 with both cams -------------
    import struct

    from gello_recorder.depth_writer import (
        COMPRESSED_DEPTH_HEADER_BYTES,
        depth_meta,
        read_depth_frame,
    )

    def make_depth_payload(seed, w=80, h=48):
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 5000, size=(h, w), dtype=np.uint16)
        ok, buf = cv2.imencode(".png", arr)
        assert ok, "cv2.imencode failed to produce a PNG"
        return struct.pack("<iff", 0, 0.0, 0.0) + buf.tobytes(), arr

    depth_dir = os.path.join(scratch, "session_selftest_depth")
    dsess = RecordingSession(depth_dir, camera_fps=30.0, record_depth=True)
    assert dsess.record_depth
    dsess.set_depth_source(1, "/cam1/cam1/depth/image_rect_raw/compressedDepth", False)
    dsess.set_depth_source(2, "/cam2/cam2/aligned_depth_to_color/image_raw/compressedDepth", True)
    dsess.set_depth_camera_info(
        1, width=80, height=48, distortion_model="plumb_bob", D=[0.0] * 5,
        K=[500, 0, 40, 0, 500, 24, 0, 0, 1], R=list(np.eye(3).ravel()),
        P=[500, 0, 40, 0, 0, 500, 24, 0, 0, 0, 1, 0],
        frame_id="cam1_depth_optical_frame",
    )
    dsess.set_depth_extrinsics(1, list(np.eye(3).ravel()), [0.015, 0.0, 0.0])

    # Interleave a colour frame so cam1_frames and cam1 depth share the t() origin.
    assert dsess.write_cam1_frame(make_jpeg(2, 90)) == 0
    n_d1, n_d2 = 4, 2
    ref_d1 = []
    for i in range(n_d1):
        payload, arr = make_depth_payload(300 + i)
        idx = dsess.write_cam1_depth_frame(payload, stamp_s=1.7e9 + 0.033 * i)
        assert idx == i, f"cam1 depth idx {idx} != {i}"
        ref_d1.append(arr)
    for i in range(n_d2):
        payload, _ = make_depth_payload(400 + i)
        assert dsess.write_cam2_depth_frame(payload) == i
    # Corrupt depth payload: -1, no count bump (like the corrupt JPEG above).
    assert dsess.write_cam1_depth_frame(b"definitely not a compressedDepth message") == -1

    dsess.flush()
    dres = dsess.close()
    print(f"close() [depth on]    : {json.dumps(dres)}")
    assert set(dres.keys()) == {"duration_s", "message_counts"}, dres
    assert dres["message_counts"] == {
        "cam1_frames": 1,
        "cam1_depth_frames": n_d1,
        "cam2_depth_frames": n_d2,
    }, dres

    # vectors.h5 must still have exactly the nine tables -- depth adds nothing there.
    with h5py.File(os.path.join(depth_dir, "vectors.h5"), "r") as f:
        assert sorted(f.keys()) == sorted([
            "synchronized", "gello_joint_states", "ur_joint_states", "command",
            "gripper", "wrench", "tcp_pose", "cam1_frames", "cam2_frames",
        ]), sorted(f.keys())
        color_t = f["cam1_frames"]["t_rel_s"][0]

    depth_path = os.path.join(depth_dir, "depth.h5")
    assert os.path.exists(depth_path), "depth.h5 must exist when record_depth=True"
    with h5py.File(depth_path, "r") as f:
        assert sorted(f.keys()) == ["cam1", "cam2"], sorted(f.keys())
        assert f["cam1"]["png"].shape[0] == n_d1
        assert f["cam2"]["png"].shape[0] == n_d2
        assert f["cam1"].attrs["width"] == 80 and f["cam1"].attrs["height"] == 48
        t_d = f["cam1"]["t_rel_s"][:]
        assert np.all(np.diff(t_d) >= 0), "depth t_rel_s must be monotonic"
        assert t_d[0] >= color_t, "depth written after the colour frame -> later t_rel_s"
        assert t_d[-1] <= dres["duration_s"] + 0.01, (t_d[-1], dres["duration_s"])
        assert np.isnan(f["cam2"]["stamp_s"][0])
        for i, arr in enumerate(ref_d1):
            assert np.array_equal(read_depth_frame(f, "cam1", i), arr), i
    dmeta = depth_meta(depth_path, "cam1")
    assert dmeta["source_topic"].endswith("compressedDepth") and dmeta["aligned_to_color"] is False
    assert dmeta["camera_info"]["frame_id"] == "cam1_depth_optical_frame"
    assert dmeta["extrinsics_depth_to_color"]["translation"][0] == 0.015
    assert depth_meta(depth_path, "cam2")["aligned_to_color"] is True

    again_d = dsess.close()  # idempotent with depth on, too
    assert again_d["message_counts"] == dres["message_counts"], again_d

    # --- Header stamps + background writer (2026-09-14) -----------------------
    stamp_dir = os.path.join(scratch, "session_selftest_stamps")
    ssess = RecordingSession(stamp_dir, camera_fps=30.0)
    # Stamped tables take a real epoch second; omitting it must give NaN.
    ssess.write_ur([0.0] * _N, [0.0] * _N, [0.0] * _N, stamp_s=1.7e9)
    ssess.write_ur([0.0] * _N, [0.0] * _N, [0.0] * _N)
    ssess.write_tcp([0.0] * 7, stamp_s=1.7e9)
    ssess.write_wrench([0.0] * 6, stamp_s=1.7e9)
    ssess.write_gello([0.0] * _N, [None] * _N, stamp_s=1.7e9)
    ssess.write_cmd([0.0] * _N)          # Float64MultiArray: no stamp column
    # Async path: submitted, not written -- close() must drain it exactly.
    n_async = 12
    for i in range(n_async):
        assert ssess.submit_cam_frame(1, make_jpeg(0, 20 + i * 5),
                                      stamp_s=1.7e9 + 0.0333 * i) is True
    assert ssess.dropped_frames()["total"] == 0
    sres = ssess.close()
    assert sres["message_counts"]["cam1_frames"] == n_async, sres

    with h5py.File(os.path.join(stamp_dir, "vectors.h5"), "r") as f:
        for table, tail in (
            ("ur_joint_states", "eff6"), ("tcp_pose", "qw"),
            ("wrench", "tz"), ("gello_joint_states", "qd6"),
            ("cam1_frames", "frame_idx"), ("cam2_frames", "frame_idx"),
        ):
            cols = json.loads(f[table].attrs["columns"])
            assert cols[-2:] == [tail, "stamp_s"], (table, cols)
        for table in ("command", "gripper", "synchronized"):
            cols = json.loads(f[table].attrs["columns"])
            assert "stamp_s" not in cols, (table, cols)
        assert abs(f["ur_joint_states"]["stamp_s"][0] - 1.7e9) < 1e-6
        assert np.isnan(f["ur_joint_states"]["stamp_s"][1])
        assert list(f["cam1_frames"]["frame_idx"][:]) == list(range(n_async))
    cap = cv2.VideoCapture(os.path.join(stamp_dir, "cam1.mp4"))
    read = 0
    while cap.read()[0]:
        read += 1
    cap.release()
    assert read == n_async, f"drain lost frames: {read} != {n_async}"

    print("SELF-TEST OK")
