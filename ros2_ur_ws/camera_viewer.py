#!/usr/bin/env python3
"""Side-by-side live viewer for the two ACT-deploy RealSense cameras.

Standalone rclpy script (NOT part of the colcon build) -- run it directly with
ROS2 Humble sourced:

    python3 camera_viewer.py \
        --cam1-topic /cam1/cam1/color/image_raw/compressed \
        --cam2-topic /cam2/cam2/color/image_raw/compressed \
        --cam1-label "cam1 - SCENE - 147122072740" \
        --cam2-label "cam2 - CLOSE-UP - 243222072700"

Purpose: BEFORE trusting an autonomous real-robot policy deploy, a human needs to
eyeball both camera feeds at once and confirm which physical camera is which
(which one is the wide "scene" view, which is the "close-up") -- if they're
swapped, the policy sees the wrong thing. This shows cam1 on the LEFT and cam2 on
the RIGHT in a single window, each labelled, with a live FPS readout and an
obvious "STALLED" indicator if a feed drops.

Decoding matches the rest of this codebase (see gello_recorder/video_writer.py):
the CompressedImage ``.data`` bytes are a complete JPEG file, decoded straight
with ``cv2.imdecode(np.frombuffer(..., np.uint8), cv2.IMREAD_COLOR)``. We do NOT
use cv_bridge -- it is broken on this machine (NumPy 2.x ABI mismatch,
``_ARRAY_API not found``, its compiled extension targets NumPy 1.x).

This script's ONLY job is to display. It does not start, stop, or otherwise
manage the cameras -- the calling launch_cameras.sh owns that lifecycle and keeps
the cameras running after this viewer exits.
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# A feed is considered "stalled" if no new frame has arrived in this many seconds.
STALL_TIMEOUT_S = 1.0
# Rolling window (in samples) used to compute the displayed FPS. At the nominal
# 30 fps this is ~0.5 s of history -- long enough to be steady, short enough to
# react quickly when a feed hitches.
FPS_WINDOW = 15
# Each pane is rendered at this height; wider/taller frames are scaled to fit so
# hconcat never fails on a height mismatch (both cams are 1280x720 in practice,
# so this is a safety net, not the common path).
PANE_HEIGHT = 720
PANE_WIDTH = 1280


class _CameraFeed:
    """Holds the latest decoded frame + timing bookkeeping for one camera."""

    def __init__(self, label: str):
        self.label = label
        self.frame = None                     # latest decoded BGR image, or None
        self.last_rx_wall = None              # wall-clock time of last frame (s)
        self._rx_times = deque(maxlen=FPS_WINDOW)  # recent arrival timestamps

    def on_frame(self, jpeg_bytes: bytes) -> None:
        """Decode an incoming JPEG frame and update timing.

        Mirrors the repo's decode style. A corrupt/incomplete JPEG (imdecode
        returns None) is dropped silently rather than crashing -- ROS network
        hiccups can occasionally deliver a truncated frame.
        """
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        now = time.monotonic()
        self.frame = frame
        self.last_rx_wall = now
        self._rx_times.append(now)

    def fps(self) -> float:
        """Rolling-average FPS from actual arrival timestamps (0.0 if unknown)."""
        if len(self._rx_times) < 2:
            return 0.0
        span = self._rx_times[-1] - self._rx_times[0]
        if span <= 0.0:
            return 0.0
        return (len(self._rx_times) - 1) / span

    def is_stalled(self) -> bool:
        """True if we've seen a frame before but none in the last STALL_TIMEOUT_S."""
        if self.last_rx_wall is None:
            return False  # never received -> "waiting", handled separately
        return (time.monotonic() - self.last_rx_wall) > STALL_TIMEOUT_S

    def has_frame(self) -> bool:
        return self.frame is not None


def _draw_label(img, text: str, org, scale: float = 0.9, thickness: int = 2,
                color=(255, 255, 255)) -> None:
    """Draw text with a filled dark background box behind it, so it stays legible
    over any camera image. Draws in place on ``img``. ``org`` is the top-left
    corner of the text box (x, y)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    pad = 6
    # Semi-transparent dark rectangle behind the text.
    x2, y2 = x + tw + 2 * pad, y + th + 2 * pad + baseline
    roi = img[y:y2, x:x2]
    if roi.size > 0:
        dark = np.zeros_like(roi)
        cv2.addWeighted(roi, 0.35, dark, 0.65, 0.0, dst=roi)
    cv2.putText(img, text, (x + pad, y + pad + th), font, scale, color,
                thickness, cv2.LINE_AA)


def _fit_pane(frame):
    """Resize a frame to the standard pane size, preserving aspect ratio and
    letterboxing onto a black PANE_HEIGHT x PANE_WIDTH canvas."""
    h, w = frame.shape[:2]
    if (w, h) == (PANE_WIDTH, PANE_HEIGHT):
        return frame
    scale = min(PANE_WIDTH / w, PANE_HEIGHT / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((PANE_HEIGHT, PANE_WIDTH, 3), dtype=np.uint8)
    y0 = (PANE_HEIGHT - new_h) // 2
    x0 = (PANE_WIDTH - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _render_pane(feed: _CameraFeed):
    """Build one pane (PANE_HEIGHT x PANE_WIDTH BGR) for a camera feed, with its
    label + FPS overlay, or a waiting/stalled state."""
    if not feed.has_frame():
        pane = np.zeros((PANE_HEIGHT, PANE_WIDTH, 3), dtype=np.uint8)
        _draw_label(pane, f"waiting for {feed.label}...", (20, 20),
                    scale=0.9, color=(200, 200, 200))
        return pane

    pane = _fit_pane(feed.frame).copy()
    stalled = feed.is_stalled()

    if stalled:
        # Dim the whole pane so a frozen feed reads as obviously dead.
        pane = (pane.astype(np.float32) * 0.4).astype(np.uint8)
        _draw_label(pane, feed.label, (20, 20), scale=0.9)
        _draw_label(pane, "STALLED - no frames", (20, 70), scale=1.1,
                    thickness=3, color=(0, 0, 255))
    else:
        _draw_label(pane, f"{feed.label}   {feed.fps():.1f} fps", (20, 20),
                    scale=0.9)
    return pane


class CameraViewer(Node):
    """Subscribes to both compressed camera topics and stashes latest frames."""

    def __init__(self, args):
        super().__init__("camera_viewer")
        self.cam1 = _CameraFeed(args.cam1_label)
        self.cam2 = _CameraFeed(args.cam2_label)
        # Plain integer queue depth of 10 -- same pattern gello_recorder and
        # gello_policy use for these exact CompressedImage topics.
        self.create_subscription(
            CompressedImage, args.cam1_topic, self._on_cam1, 10)
        self.create_subscription(
            CompressedImage, args.cam2_topic, self._on_cam2, 10)

    def _on_cam1(self, msg: CompressedImage):
        self.cam1.on_frame(bytes(msg.data))

    def _on_cam2(self, msg: CompressedImage):
        self.cam2.on_frame(bytes(msg.data))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Side-by-side live viewer for the two ACT-deploy RealSense "
                    "cameras (cam1 LEFT, cam2 RIGHT).")
    p.add_argument("--cam1-topic", required=True,
                   help="CompressedImage topic for camera 1 (shown LEFT)")
    p.add_argument("--cam2-topic", required=True,
                   help="CompressedImage topic for camera 2 (shown RIGHT)")
    p.add_argument("--cam1-label", required=True,
                   help="Overlay label for camera 1")
    p.add_argument("--cam2-label", required=True,
                   help="Overlay label for camera 2")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    rclpy.init()
    node = CameraViewer(args)

    window = "ACT cameras — cam1 SCENE | cam2 CLOSE-UP"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1600, 450)

    exit_code = 0
    try:
        while rclpy.ok():
            # Service ROS callbacks (non-blocking-ish) so latest frames update,
            # then service the OpenCV GUI event loop. A plain rclpy.spin() would
            # never yield to cv2.waitKey and the window would appear frozen.
            rclpy.spin_once(node, timeout_sec=0.005)

            left = _render_pane(node.cam1)
            right = _render_pane(node.cam2)
            cv2.imshow(window, cv2.hconcat([left, right]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc
                break
            # User clicked the window's X button.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        # Normal Ctrl-C -- exit quietly, no scary traceback.
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001 - best-effort on shutdown
            pass
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
