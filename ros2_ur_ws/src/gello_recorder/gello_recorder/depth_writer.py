#!/usr/bin/env python3
"""Standalone depth-frame writer for the GELLO diagnostic recorder's RealSense streams.

Why a separate file, and why not MP4
------------------------------------
Each RealSense camera can publish depth as ``sensor_msgs/msg/CompressedImage`` on
``/<cam>/<cam>/depth/image_rect_raw/compressedDepth`` (``format == '16UC1;
compressedDepth'``, measured 2026-09-14 on realsense-ros 4.58.2: ~29 Hz, 80-130 KB per
message at 848x480; with ``align_depth.enable:=true`` the aligned topic is 1280x720 and
~200 KB). Depth is ``uint16`` millimetres (``depth_scale`` 0.001 m). An MP4 cannot
carry 16-bit single-channel data losslessly, and re-encoding on the hot path would
cost the very CPU the two colour streams already compete for. So the depth stream
gets its OWN ``depth.h5`` next to ``vectors.h5``: the PNG that the camera driver
already produced is stored verbatim, one variable-length ``uint8`` cell per frame,
with the timestamp columns needed to align it offline.

Payload layout (``compressed_depth_image_transport``)
----------------------------------------------------
``data[0:12]`` is a ``ConfigHeader`` -- ``int32`` format enum (observed 0) followed by
two ``float32`` depth-quantisation parameters that are only meaningful for ``32FC1``
(garbage for ``16UC1``). ``data[12:]`` is a complete PNG file (magic
``\\x89PNG\\r\\n\\x1a\\n``) that ``cv2.imdecode(..., cv2.IMREAD_UNCHANGED)`` decodes to a
``uint16 (H, W)`` array. :func:`split_compressed_depth` strips the header and checks
the magic; nothing on the write path decodes the PNG. Width/height are read straight
from the PNG ``IHDR`` chunk (big-endian ``uint32`` pair at byte offset 16..24) so the
group attrs are filled on the first frame without ``cv2``.

File layout (``depth.h5``)
--------------------------
::

    /<cam>/                     attrs: encoding, unit, depth_scale_m, container,
                                       header_bytes_stripped, source_topic,
                                       aligned_to_color, width, height
        png        vlen uint8, shape (N,)   -- the PNG bytes, verbatim
        t_rel_s    float64,    shape (N,)   -- session-relative capture time
        frame_idx  int64,      shape (N,)   -- 0-based running index (== row)
        stamp_s    float64,    shape (N,)   -- ROS header stamp (s), NaN if unknown
    /<cam>/camera_info/         attrs: width, height, distortion_model, frame_id,
                                       D, K, R, P   (sensor_msgs/CameraInfo)
    /<cam>/extrinsics_depth_to_color/
                                attrs: rotation (9, column-major), translation (3, m),
                                       layout='column_major'

The write path depends ONLY on ``h5py`` + ``numpy`` (no rclpy, no cv2), so it is
importable and testable outside a ROS environment. ``cv2`` is imported lazily and only
inside the offline reader helpers (:func:`read_depth_frame`,
:func:`iter_depth_frames`) and the self-test. Run ``python3 depth_writer.py`` for the
built-in self-test.

Scope note: like :class:`~gello_recorder.video_writer.Mp4FrameWriter` this does no
rate matching, no duplication and no dropping -- it strips the header, appends, and
returns the frame index. The caller (``RecordingSession``) owns the relative clock.
"""

import struct

import h5py
import numpy as np

COMPRESSED_DEPTH_HEADER_BYTES = 12
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# PNG: 8-byte signature, then the IHDR chunk = 4-byte length + 4-byte type ("IHDR")
# + 4-byte width + 4-byte height + ...  -> width/height live at bytes 16..24.
_PNG_IHDR_DIMS_OFFSET = 16
_PNG_IHDR_DIMS_END = 24

DEFAULT_CAMS = ("cam1", "cam2")

_VLEN_U8 = h5py.special_dtype(vlen=np.uint8)


def split_compressed_depth(data: bytes) -> "bytes | None":
    """Return the PNG bytes of a ``compressed_depth_image_transport`` payload.

    Pure function, no decode. Returns ``None`` when ``data`` is shorter than the
    12-byte ConfigHeader plus the 8-byte PNG signature, or when the bytes at offset 12
    are not the PNG magic (truncated / foreign payload -- e.g. a colour JPEG that was
    wired to the wrong subscriber).
    """
    if data is None:
        return None
    data = bytes(data)
    start = COMPRESSED_DEPTH_HEADER_BYTES
    if len(data) < start + len(PNG_MAGIC):
        return None
    if data[start:start + len(PNG_MAGIC)] != PNG_MAGIC:
        return None
    return data[start:]


def png_dimensions(png: bytes) -> "tuple[int, int] | None":
    """``(width, height)`` from a PNG's IHDR chunk, or ``None`` if the buffer is too
    short. Reads the two big-endian ``uint32`` at byte offset 16..24; no decode."""
    if png is None or len(png) < _PNG_IHDR_DIMS_END:
        return None
    width, height = struct.unpack(
        ">II", png[_PNG_IHDR_DIMS_OFFSET:_PNG_IHDR_DIMS_END]
    )
    return int(width), int(height)


class DepthH5Writer:
    """Appends compressed-depth PNG payloads to one ``depth.h5``, one group per camera."""

    def __init__(self, path: str, cams=DEFAULT_CAMS):
        """Open ``path`` (h5py mode ``'w'``) and create every per-camera group up front.

        Each ``<cam>`` group gets four resizable datasets (``png`` vlen uint8,
        ``t_rel_s`` float64, ``frame_idx`` int64, ``stamp_s`` float64; all
        ``shape=(0,), maxshape=(None,), chunks=True``) and the fixed metadata attrs.
        ``source_topic`` / ``aligned_to_color`` are placeholders until
        :meth:`set_source`; ``width`` / ``height`` stay 0 until the first frame.
        """
        self._path = path
        self._cams = tuple(cams)
        self._closed = False
        self._h5 = None
        self._groups = {}
        self._dsets = {}     # cam -> (png, t_rel_s, frame_idx, stamp_s)
        self._counts = {}    # cam -> rows written

        self._h5 = h5py.File(path, "w")
        for cam in self._cams:
            grp = self._h5.create_group(cam)
            grp.attrs["encoding"] = "16UC1"
            grp.attrs["unit"] = "mm"
            grp.attrs["depth_scale_m"] = 0.001
            grp.attrs["container"] = "png"
            grp.attrs["header_bytes_stripped"] = COMPRESSED_DEPTH_HEADER_BYTES
            grp.attrs["source_topic"] = ""
            grp.attrs["aligned_to_color"] = False
            grp.attrs["width"] = 0
            grp.attrs["height"] = 0

            png = grp.create_dataset(
                "png", shape=(0,), maxshape=(None,), dtype=_VLEN_U8, chunks=True
            )
            t_rel = grp.create_dataset(
                "t_rel_s", shape=(0,), maxshape=(None,), dtype="float64", chunks=True
            )
            frame_idx = grp.create_dataset(
                "frame_idx", shape=(0,), maxshape=(None,), dtype="int64", chunks=True
            )
            stamp = grp.create_dataset(
                "stamp_s", shape=(0,), maxshape=(None,), dtype="float64", chunks=True
            )
            self._groups[cam] = grp
            self._dsets[cam] = (png, t_rel, frame_idx, stamp)
            self._counts[cam] = 0

    # ---- metadata ------------------------------------------------------------
    def _group(self, cam: str) -> h5py.Group:
        try:
            return self._groups[cam]
        except KeyError:
            raise KeyError(
                f"unknown cam {cam!r}; this writer has {list(self._cams)}"
            ) from None

    def set_source(self, cam: str, topic: str, aligned_to_color: bool) -> None:
        """Record which ROS topic fed this cam and whether it was the
        ``aligned_depth_to_color`` stream (then the PNG is in the COLOUR frame and
        ``camera_info`` describes the colour intrinsics)."""
        grp = self._group(cam)
        grp.attrs["source_topic"] = str(topic)
        grp.attrs["aligned_to_color"] = bool(aligned_to_color)

    def set_camera_info(self, cam: str, *, width: int, height: int,
                        distortion_model: str, D, K, R, P, frame_id: str) -> None:
        """Persist a ``sensor_msgs/CameraInfo`` as attrs on ``<cam>/camera_info``.

        Idempotent: a second call overwrites every attr (the driver republishes the
        same info at frame rate; the caller may forward each one or just the first).
        """
        grp = self._group(cam)
        info = grp.require_group("camera_info")
        info.attrs["width"] = int(width)
        info.attrs["height"] = int(height)
        info.attrs["distortion_model"] = str(distortion_model)
        info.attrs["frame_id"] = str(frame_id)
        info.attrs["D"] = np.asarray(D, dtype=np.float64).reshape(-1)
        info.attrs["K"] = np.asarray(K, dtype=np.float64).reshape(9)
        info.attrs["R"] = np.asarray(R, dtype=np.float64).reshape(9)
        info.attrs["P"] = np.asarray(P, dtype=np.float64).reshape(12)

    def set_extrinsics_depth_to_color(self, cam: str, rotation, translation) -> None:
        """Persist ``realsense2_camera_msgs/Extrinsics`` (depth -> colour) as attrs on
        ``<cam>/extrinsics_depth_to_color``: ``rotation`` float64[9] exactly as
        published (column-major), ``translation`` float64[3] in metres, plus
        ``layout='column_major'`` so a reader never has to guess. Idempotent."""
        grp = self._group(cam)
        ext = grp.require_group("extrinsics_depth_to_color")
        ext.attrs["rotation"] = np.asarray(rotation, dtype=np.float64).reshape(9)
        ext.attrs["translation"] = np.asarray(translation, dtype=np.float64).reshape(3)
        ext.attrs["layout"] = "column_major"

    # ---- hot path ------------------------------------------------------------
    def write_compressed_depth(self, cam: str, data: bytes, t_rel_s: float,
                               stamp_s: float = float("nan")) -> int:
        """Strip the 12-byte header and append the PNG + timestamps for ``cam``.

        Returns the 0-based frame index just written. If the payload has no PNG at
        offset 12 (truncated / wrong-topic message) it prints a one-line WARNING,
        writes nothing, does NOT advance the index and returns ``-1`` -- the same
        contract as ``Mp4FrameWriter.write_compressed`` so callers can skip logging
        that frame. No PNG decode happens here; width/height attrs are filled from the
        IHDR bytes on the first accepted frame.
        """
        png = split_compressed_depth(data)
        if png is None:
            print(
                "[DepthH5Writer] WARNING: compressedDepth payload has no PNG at "
                "offset %d (len=%d, corrupt/incomplete); skipping. cam=%s path=%s"
                % (COMPRESSED_DEPTH_HEADER_BYTES,
                   0 if data is None else len(data), cam, self._path)
            )
            return -1

        grp = self._group(cam)
        png_d, t_d, idx_d, stamp_d = self._dsets[cam]
        idx = self._counts[cam]
        new_len = idx + 1

        if idx == 0:
            dims = png_dimensions(png)
            if dims is not None:
                grp.attrs["width"], grp.attrs["height"] = dims

        png_d.resize((new_len,))
        png_d[idx] = np.frombuffer(png, dtype=np.uint8)
        t_d.resize((new_len,))
        t_d[idx] = float(t_rel_s)
        idx_d.resize((new_len,))
        idx_d[idx] = idx
        stamp_d.resize((new_len,))
        stamp_d[idx] = float(stamp_s)

        self._counts[cam] = new_len
        return idx

    # ---- bookkeeping ---------------------------------------------------------
    def count(self, cam: str) -> int:
        """Number of frames successfully written for ``cam``."""
        self._group(cam)
        return self._counts[cam]

    @property
    def path(self) -> str:
        return self._path

    @property
    def cams(self) -> tuple:
        return self._cams

    def flush(self) -> None:
        """Flush the underlying file to disk (no-op after :meth:`close`)."""
        if self._closed or self._h5 is None:
            return
        self._h5.flush()

    def close(self) -> None:
        """Flush + close the file. Idempotent, and safe on a partially-constructed
        object (``__new__`` without ``__init__``, or ``__init__`` that raised before
        the file was opened) -- mirrors ``RecordingSession.close``."""
        if getattr(self, "_closed", False):
            return
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.flush()
            except Exception:  # noqa: BLE001 - best-effort on shutdown
                pass
            try:
                h5.close()
            except Exception:  # noqa: BLE001
                pass
        self._h5 = None
        self._closed = True


# ---- offline reader helpers (cv2 imported lazily; never on the write path) ---
class _OpenRO:
    """Context manager: accept a path (opened read-only, closed on exit) or an
    already-open ``h5py.File`` / ``h5py.Group`` (used as-is, NOT closed)."""

    def __init__(self, h5_path_or_file):
        self._arg = h5_path_or_file
        self._own = None

    def __enter__(self):
        if isinstance(self._arg, (h5py.File, h5py.Group)):
            return self._arg
        self._own = h5py.File(str(self._arg), "r")
        return self._own

    def __exit__(self, *exc):
        if self._own is not None:
            self._own.close()
            self._own = None
        return False


def _decode_png_u16(png_u8: np.ndarray) -> np.ndarray:
    import cv2  # offline only

    arr = cv2.imdecode(np.asarray(png_u8, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("cv2.imdecode failed on stored depth PNG")
    return arr


def read_depth_frame(h5_path_or_file, cam: str, idx: int) -> np.ndarray:
    """Decode frame ``idx`` of ``cam`` to a ``uint16 (H, W)`` array in millimetres."""
    with _OpenRO(h5_path_or_file) as f:
        return _decode_png_u16(f[cam]["png"][idx])


def iter_depth_frames(h5_path_or_file, cam: str):
    """Yield ``(frame_idx, t_rel_s, stamp_s, uint16 array)`` for every frame of ``cam``
    in write order."""
    with _OpenRO(h5_path_or_file) as f:
        grp = f[cam]
        png_d, t_d, idx_d, stamp_d = (
            grp["png"], grp["t_rel_s"], grp["frame_idx"], grp["stamp_s"]
        )
        n = png_d.shape[0]
        t_all = t_d[:n]
        idx_all = idx_d[:n]
        stamp_all = stamp_d[:n]
        for i in range(n):
            yield (int(idx_all[i]), float(t_all[i]), float(stamp_all[i]),
                   _decode_png_u16(png_d[i]))


def _attrs_to_dict(attrs) -> dict:
    out = {}
    for k, v in attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def depth_meta(h5_path_or_file, cam: str) -> dict:
    """Group attrs of ``cam`` as a plain dict (numpy scalars/arrays converted), plus
    ``n_frames`` and -- when present -- nested ``camera_info`` and
    ``extrinsics_depth_to_color`` dicts."""
    with _OpenRO(h5_path_or_file) as f:
        grp = f[cam]
        meta = _attrs_to_dict(grp.attrs)
        meta["n_frames"] = int(grp["png"].shape[0])
        for sub in ("camera_info", "extrinsics_depth_to_color"):
            if sub in grp:
                meta[sub] = _attrs_to_dict(grp[sub].attrs)
        return meta


def _self_test() -> None:
    """Synthesise uint16 frames, wrap them like the ROS transport does, round-trip."""
    import os
    import tempfile

    import cv2

    scratch = tempfile.mkdtemp(prefix="depth_writer_selftest_")
    path = os.path.join(scratch, "depth.h5")
    print(f"self-test scratch dir : {scratch}")
    print(f"self-test depth file   : {path}")

    width, height = 96, 64

    def make_payload(seed: int) -> "tuple[bytes, np.ndarray]":
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 6000, size=(height, width), dtype=np.uint16)
        arr[: height // 4, :] = 0  # a band of invalid (0) depth, like a real frame
        ok, buf = cv2.imencode(".png", arr)
        assert ok, "cv2.imencode failed to produce a PNG"
        header = struct.pack("<iff", 0, 0.0, 0.0)  # format enum + 2 unused floats
        assert len(header) == COMPRESSED_DEPTH_HEADER_BYTES
        return header + buf.tobytes(), arr

    n1, n2 = 6, 3
    ref1, ref2 = [], []
    w = DepthH5Writer(path)
    w.set_source("cam1", "/cam1/cam1/depth/image_rect_raw/compressedDepth", False)
    w.set_source("cam2", "/cam2/cam2/aligned_depth_to_color/image_raw/compressedDepth", True)
    w.set_camera_info(
        "cam1", width=width, height=height, distortion_model="plumb_bob",
        D=[0, 0, 0, 0, 0], K=[600, 0, 48, 0, 600, 32, 0, 0, 1],
        R=np.eye(3).ravel(), P=[600, 0, 48, 0, 0, 600, 32, 0, 0, 0, 1, 0],
        frame_id="cam1_depth_optical_frame",
    )
    w.set_extrinsics_depth_to_color("cam1", np.eye(3).ravel(), [0.015, 0.0, 0.0])

    for i in range(n1):
        payload, arr = make_payload(100 + i)
        idx = w.write_compressed_depth("cam1", payload, t_rel_s=0.033 * i, stamp_s=1e9 + i)
        assert idx == i, f"expected cam1 idx {i}, got {idx}"
        ref1.append(arr)
    for i in range(n2):
        payload, arr = make_payload(200 + i)
        idx = w.write_compressed_depth("cam2", payload, t_rel_s=0.05 * i)
        assert idx == i, f"expected cam2 idx {i}, got {idx}"
        ref2.append(arr)

    # Corrupt payloads: no crash, -1, no count bump.
    for bad in (b"", b"\x00" * 11, b"\x00" * 12 + b"not a png", make_payload(1)[0][12:]):
        assert w.write_compressed_depth("cam1", bad, 9.9) == -1
    assert w.count("cam1") == n1 and w.count("cam2") == n2

    w.flush()
    w.close()
    w.close()  # idempotent

    with h5py.File(path, "r") as f:
        for cam, n, ref in (("cam1", n1, ref1), ("cam2", n2, ref2)):
            grp = f[cam]
            assert grp["png"].shape[0] == n
            assert grp.attrs["width"] == width and grp.attrs["height"] == height
            assert list(grp["frame_idx"][:]) == list(range(n))
            for i, arr in enumerate(ref):
                got = read_depth_frame(f, cam, i)
                assert got.dtype == np.uint16 and got.shape == (height, width), got.shape
                assert np.array_equal(got, arr), f"{cam}[{i}] round-trip mismatch"
        assert np.isnan(f["cam2"]["stamp_s"][0])
        assert abs(f["cam1"]["stamp_s"][2] - (1e9 + 2)) < 1e-3

    frames = list(iter_depth_frames(path, "cam1"))
    assert [fi for fi, *_ in frames] == list(range(n1))
    assert all(np.array_equal(fr[3], ref1[k]) for k, fr in enumerate(frames))

    meta = depth_meta(path, "cam1")
    assert meta["aligned_to_color"] is False and meta["n_frames"] == n1
    assert meta["camera_info"]["frame_id"] == "cam1_depth_optical_frame"
    assert meta["extrinsics_depth_to_color"]["layout"] == "column_major"
    assert depth_meta(path, "cam2")["aligned_to_color"] is True

    empty = DepthH5Writer.__new__(DepthH5Writer)
    empty.close()  # safe on a never-constructed object

    print(f"round-tripped {n1}+{n2} frames of {width}x{height} uint16 from {path}")
    print("SELF-TEST OK")


if __name__ == "__main__":
    _self_test()
