"""Unit tests for :mod:`gello_recorder.depth_writer` (ROS-free).

What is pinned here is the on-disk contract the offline tools will depend on:
the 12-byte ``compressed_depth_image_transport`` header is stripped and NOTHING
else is touched (the PNG bytes are stored verbatim), width/height come from the
PNG IHDR without decoding, a corrupt payload returns ``-1`` and leaves both the
datasets and the index untouched, and camera_info / extrinsics survive a reopen.

Payloads are built the way the ROS driver builds them -- ``struct.pack("<iff")``
header + ``cv2.imencode('.png', uint16)`` -- so the writer never sees anything a
real ``/<cam>/<cam>/depth/image_rect_raw/compressedDepth`` message would not.
"""

import struct

import cv2
import h5py
import numpy as np
import pytest

from gello_recorder.depth_writer import (
    COMPRESSED_DEPTH_HEADER_BYTES,
    PNG_MAGIC,
    DepthH5Writer,
    depth_meta,
    iter_depth_frames,
    png_dimensions,
    read_depth_frame,
    split_compressed_depth,
)

W, H = 64, 40


def _png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def _frame(seed: int, w=W, h=H) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 8000, size=(h, w), dtype=np.uint16)
    arr[:4, :] = 0
    return arr


def _payload(arr: np.ndarray, fmt: int = 0) -> bytes:
    return struct.pack("<iff", fmt, 0.0, 0.0) + _png(arr)


@pytest.fixture
def writer(tmp_path):
    w = DepthH5Writer(str(tmp_path / "depth.h5"))
    yield w
    w.close()


# ---- pure helpers ----------------------------------------------------------------
def test_header_constant_and_magic():
    assert COMPRESSED_DEPTH_HEADER_BYTES == 12
    assert PNG_MAGIC == b"\x89PNG\r\n\x1a\n"


def test_split_strips_exactly_twelve_bytes():
    arr = _frame(1)
    png = _png(arr)
    assert split_compressed_depth(struct.pack("<iff", 0, 1.5, -2.0) + png) == png


def test_split_rejects_short_or_unmagic_payloads():
    assert split_compressed_depth(b"") is None
    assert split_compressed_depth(b"\x00" * 11) is None
    assert split_compressed_depth(b"\x00" * 12) is None            # header, no PNG
    assert split_compressed_depth(b"\x00" * 12 + b"\xff\xd8JPEG") is None
    assert split_compressed_depth(_png(_frame(2))) is None         # PNG w/o header
    assert split_compressed_depth(None) is None


def test_split_accepts_bytearray_and_memoryview():
    payload = _payload(_frame(3))
    assert split_compressed_depth(bytearray(payload)) == payload[12:]
    assert split_compressed_depth(memoryview(payload)) == payload[12:]


def test_png_dimensions_from_ihdr_without_decode():
    assert png_dimensions(_png(_frame(4, w=123, h=45))) == (123, 45)
    assert png_dimensions(b"\x89PNG\r\n\x1a\n") is None


# ---- writer --------------------------------------------------------------------
def test_groups_and_default_attrs_created_up_front(tmp_path):
    path = str(tmp_path / "depth.h5")
    w = DepthH5Writer(path, cams=("cam1", "cam2"))
    w.close()
    with h5py.File(path, "r") as f:
        assert sorted(f.keys()) == ["cam1", "cam2"]
        for cam in ("cam1", "cam2"):
            g = f[cam]
            assert g.attrs["encoding"] == "16UC1"
            assert g.attrs["unit"] == "mm"
            assert g.attrs["depth_scale_m"] == pytest.approx(0.001)
            assert g.attrs["container"] == "png"
            assert g.attrs["header_bytes_stripped"] == 12
            assert g.attrs["source_topic"] == ""
            assert bool(g.attrs["aligned_to_color"]) is False
            assert g.attrs["width"] == 0 and g.attrs["height"] == 0
            for name, dtype in (("t_rel_s", np.float64), ("frame_idx", np.int64),
                                ("stamp_s", np.float64)):
                assert g[name].shape == (0,)
                assert g[name].maxshape == (None,)
                assert g[name].dtype == dtype
            assert g["png"].shape == (0,)
            assert g["png"].maxshape == (None,)
            assert h5py.check_vlen_dtype(g["png"].dtype) == np.uint8


def test_round_trip_and_verbatim_png(writer):
    frames = [_frame(10 + i) for i in range(5)]
    for i, arr in enumerate(frames):
        assert writer.write_compressed_depth("cam1", _payload(arr), 0.1 * i, 100.0 + i) == i
    assert writer.count("cam1") == 5
    assert writer.count("cam2") == 0
    writer.close()

    with h5py.File(writer.path, "r") as f:
        g = f["cam1"]
        assert g["png"].shape[0] == 5
        assert list(g["frame_idx"][:]) == [0, 1, 2, 3, 4]
        np.testing.assert_allclose(g["t_rel_s"][:], [0.0, 0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(g["stamp_s"][:], [100, 101, 102, 103, 104])
        # Stored bytes are the PNG verbatim (no re-encode).
        assert bytes(g["png"][2]) == _png(frames[2])
        for i, arr in enumerate(frames):
            got = read_depth_frame(f, "cam1", i)
            assert got.dtype == np.uint16 and got.shape == (H, W)
            assert np.array_equal(got, arr)


def test_width_height_filled_on_first_frame_only(writer):
    writer.write_compressed_depth("cam1", _payload(_frame(1, w=96, h=48)), 0.0)
    writer.write_compressed_depth("cam1", _payload(_frame(2, w=32, h=16)), 0.1)
    writer.close()
    m = depth_meta(writer.path, "cam1")
    assert (m["width"], m["height"]) == (96, 48)
    assert m["n_frames"] == 2


def test_corrupt_payload_returns_minus_one_and_writes_nothing(writer, capsys):
    assert writer.write_compressed_depth("cam1", _payload(_frame(1)), 0.0) == 0
    for bad in (b"", b"\x00" * 12, b"\x00" * 12 + b"nope", _png(_frame(2)), None):
        assert writer.write_compressed_depth("cam1", bad, 0.5) == -1
    assert writer.count("cam1") == 1
    assert "WARNING" in capsys.readouterr().out
    # The next good frame gets index 1, not 6.
    assert writer.write_compressed_depth("cam1", _payload(_frame(3)), 1.0) == 1
    writer.close()
    with h5py.File(writer.path, "r") as f:
        assert f["cam1"]["png"].shape[0] == 2
        assert list(f["cam1"]["frame_idx"][:]) == [0, 1]
        np.testing.assert_allclose(f["cam1"]["t_rel_s"][:], [0.0, 1.0])


def test_stamp_defaults_to_nan(writer):
    writer.write_compressed_depth("cam2", _payload(_frame(1)), 0.0)
    writer.close()
    with h5py.File(writer.path, "r") as f:
        assert np.isnan(f["cam2"]["stamp_s"][0])


def test_unknown_cam_raises(writer):
    with pytest.raises(KeyError):
        writer.write_compressed_depth("cam9", _payload(_frame(1)), 0.0)
    with pytest.raises(KeyError):
        writer.count("cam9")


def test_source_camera_info_and_extrinsics_persist_and_overwrite(writer):
    writer.set_source("cam1", "/cam1/cam1/depth/image_rect_raw/compressedDepth", False)
    writer.set_source("cam2", "/cam2/cam2/aligned_depth_to_color/image_raw/compressedDepth", True)
    K = [615.0, 0, 424.0, 0, 615.0, 240.0, 0, 0, 1]
    P = [615.0, 0, 424.0, 0, 0, 615.0, 240.0, 0, 0, 0, 1, 0]
    writer.set_camera_info(
        "cam1", width=848, height=480, distortion_model="plumb_bob",
        D=[0, 0, 0, 0, 0], K=K, R=np.eye(3).ravel(), P=P,
        frame_id="cam1_depth_optical_frame",
    )
    # Second call (driver republishes at frame rate) must overwrite, not fail.
    writer.set_camera_info(
        "cam1", width=848, height=480, distortion_model="plumb_bob",
        D=[0.1, 0, 0, 0, 0], K=K, R=np.eye(3).ravel(), P=P,
        frame_id="cam1_depth_optical_frame",
    )
    rot = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    writer.set_extrinsics_depth_to_color("cam1", rot, [0.0149, -0.0001, 0.0003])
    writer.set_extrinsics_depth_to_color("cam1", rot, [0.0150, 0.0, 0.0])
    writer.close()

    with h5py.File(writer.path, "r") as f:
        ci = f["cam1"]["camera_info"].attrs
        assert ci["width"] == 848 and ci["height"] == 480
        assert ci["distortion_model"] == "plumb_bob"
        assert ci["frame_id"] == "cam1_depth_optical_frame"
        np.testing.assert_allclose(ci["D"], [0.1, 0, 0, 0, 0])
        assert ci["K"].shape == (9,) and ci["R"].shape == (9,) and ci["P"].shape == (12,)
        assert ci["K"].dtype == np.float64
        ex = f["cam1"]["extrinsics_depth_to_color"].attrs
        np.testing.assert_allclose(ex["rotation"], rot)
        np.testing.assert_allclose(ex["translation"], [0.015, 0.0, 0.0])
        assert ex["layout"] == "column_major"
        assert "camera_info" not in f["cam2"]

    m1 = depth_meta(writer.path, "cam1")
    assert m1["source_topic"] == "/cam1/cam1/depth/image_rect_raw/compressedDepth"
    assert m1["aligned_to_color"] is False
    assert m1["camera_info"]["K"] == pytest.approx(K)
    assert m1["extrinsics_depth_to_color"]["layout"] == "column_major"
    m2 = depth_meta(writer.path, "cam2")
    assert m2["aligned_to_color"] is True
    assert "camera_info" not in m2 and "extrinsics_depth_to_color" not in m2


def test_iter_depth_frames_yields_in_order(writer):
    frames = [_frame(20 + i) for i in range(3)]
    for i, arr in enumerate(frames):
        writer.write_compressed_depth("cam2", _payload(arr), 0.25 * i, 7.0 + i)
    writer.close()
    out = list(iter_depth_frames(writer.path, "cam2"))
    assert [o[0] for o in out] == [0, 1, 2]
    assert [o[1] for o in out] == pytest.approx([0.0, 0.25, 0.5])
    assert [o[2] for o in out] == pytest.approx([7.0, 8.0, 9.0])
    for o, arr in zip(out, frames):
        assert np.array_equal(o[3], arr)
    assert list(iter_depth_frames(writer.path, "cam1")) == []


def test_readers_accept_open_file_and_leave_it_open(writer):
    writer.write_compressed_depth("cam1", _payload(_frame(1)), 0.0)
    writer.close()
    with h5py.File(writer.path, "r") as f:
        read_depth_frame(f, "cam1", 0)
        list(iter_depth_frames(f, "cam1"))
        depth_meta(f, "cam1")
        assert f.id.valid  # helpers must not close a caller-owned handle


def test_flush_and_close_idempotent_and_safe_partial(tmp_path):
    path = str(tmp_path / "depth.h5")
    w = DepthH5Writer(path)
    w.write_compressed_depth("cam1", _payload(_frame(1)), 0.0)
    w.flush()
    w.close()
    w.close()
    w.flush()  # after close: no-op, no raise
    with h5py.File(path, "r") as f:
        assert f["cam1"]["png"].shape[0] == 1

    partial = DepthH5Writer.__new__(DepthH5Writer)
    partial.close()
    partial.close()
