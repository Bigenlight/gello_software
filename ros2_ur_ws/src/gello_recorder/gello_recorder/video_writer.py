#!/usr/bin/env python3
"""Standalone MP4 frame writer for the GELLO diagnostic recorder's camera streams.

Each RealSense camera publishes ``sensor_msgs/msg/CompressedImage`` (JPEG-encoded
``.data`` bytes, standard ROS image_transport "compressed" format -- the bytes are a
complete JPEG file). One :class:`Mp4FrameWriter` instance owns ONE camera's single
growing MP4 file: frames are decoded and appended as they arrive, one per ROS
callback (no batching). The writer is opened lazily on the first frame so the camera
resolution is auto-detected from the decoded image -- the caller does not need to
know it ahead of time.

This module is deliberately ROS-free (no rclpy import) so it is unit-testable
standalone -- run ``python3 video_writer.py`` to exercise the built-in self-test.

Scope note: this module does NOT do frame-rate matching / duplication / dropping.
It just decodes -> lazily opens the writer -> writes -> returns the frame index.
Timestamp bookkeeping for offline alignment is handled by the caller separately;
this module only exposes the running frame index so those two can be cross-referenced.
"""

import cv2
import numpy as np


class Mp4FrameWriter:
    """Appends incoming JPEG-compressed frames to a single MP4 file, one camera per instance."""

    def __init__(self, path: str, fps: float = 30.0):
        """Store target path/fps. Do NOT open the cv2.VideoWriter yet -- defer until the
        first frame arrives so we can auto-detect frame width/height from the decoded
        image (avoids requiring the caller to know camera resolution ahead of time).
        Track a running frame_idx counter starting at 0.
        """
        self._path = path
        self._fps = float(fps)
        self._writer = None          # lazily created on first successful decode
        self._frame_idx = 0          # number of frames successfully written so far
        self._closed = False

    def write_compressed(self, jpeg_bytes: bytes) -> int:
        """Decode ``jpeg_bytes`` and append the frame to this camera's MP4.

        On the first successful decode the underlying ``cv2.VideoWriter`` is opened
        using the decoded frame's ``(height, width)``. Returns the 0-indexed frame
        index that was just written (first call -> 0, second -> 1, ...).

        If decode fails (``cv2.imdecode`` returns ``None`` -- corrupt/incomplete JPEG)
        the call is a no-op: it does NOT crash, does NOT advance the index or write a
        frame, logs a warning via ``print`` and returns ``-1`` so the caller can skip
        logging that frame's timestamp. This is intentionally defensive because ROS
        network hiccups can occasionally deliver a truncated/corrupt frame.
        """
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(
                "[Mp4FrameWriter] WARNING: failed to decode JPEG frame "
                "(corrupt/incomplete); skipping. path=%s" % self._path
            )
            return -1

        if self._writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self._path, fourcc, self._fps, (width, height)
            )
            if not self._writer.isOpened():
                # Should not happen with opencv's bundled mp4v encoder, but fail loud
                # rather than silently dropping every frame.
                raise RuntimeError(
                    "cv2.VideoWriter failed to open (fourcc=mp4v, path=%s, "
                    "size=%dx%d, fps=%s)" % (self._path, width, height, self._fps)
                )

        self._writer.write(frame)
        idx = self._frame_idx
        self._frame_idx += 1
        return idx

    def close(self) -> None:
        """Release the underlying ``cv2.VideoWriter`` if it was opened.

        Safe to call when no frame was ever received (nothing to release) and safe to
        call more than once (idempotent).
        """
        if self._closed:
            return
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._closed = True

    @property
    def frame_count(self) -> int:
        """Number of frames successfully written so far."""
        return self._frame_idx


def _self_test() -> None:
    """Synthesize frames, round-trip them through Mp4FrameWriter, and assert."""
    import os
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "selftest.mp4")

    width, height = 320, 240
    n_frames = 10

    writer = Mp4FrameWriter(path, fps=30.0)

    # Feed 10 distinguishable solid-color frames, one at a time.
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Different solid color per frame so they're visually distinguishable.
        frame[:, :, i % 3] = int(25 + (i * 23) % 230)
        ok, buf = cv2.imencode(".jpg", frame)
        assert ok, "cv2.imencode failed to produce a JPEG"
        jpeg_bytes = buf.tobytes()
        idx = writer.write_compressed(jpeg_bytes)
        assert idx == i, "expected frame index %d, got %d" % (i, idx)

    # Corrupt-frame path: must NOT raise, must return -1, must NOT advance the index.
    bad_idx = writer.write_compressed(b"not a jpeg")
    assert bad_idx == -1, "corrupt frame should return -1, got %d" % bad_idx
    assert writer.frame_count == n_frames, (
        "corrupt frame must not advance frame_count (got %d)" % writer.frame_count
    )

    # close() must be idempotent.
    writer.close()
    writer.close()

    assert os.path.exists(path), "output mp4 was not created at %s" % path
    assert os.path.getsize(path) > 0, "output mp4 is 0 bytes at %s" % path

    # Reopen and count how many frames we can read back.
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), "cv2.VideoCapture could not open %s" % path
    read_count = 0
    last_shape = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_shape = frame.shape
        read_count += 1
    cap.release()

    assert read_count == n_frames, (
        "expected to read back %d frames, got %d" % (n_frames, read_count)
    )
    assert last_shape is not None, "no frames were read back"
    h, w = last_shape[:2]
    assert (w, h) == (width, height), (
        "read-back dims %dx%d != expected %dx%d" % (w, h, width, height)
    )

    print("read back %d frames of size %dx%d from %s" % (read_count, w, h, path))
    print("SELF-TEST OK")


if __name__ == "__main__":
    _self_test()
