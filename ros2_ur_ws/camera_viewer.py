#!/usr/bin/env python3
"""Side-by-side live viewer for the two ACT-deploy RealSense cameras.

Standalone rclpy script (NOT part of the colcon build) -- run it directly with
ROS2 Humble sourced:

    python3 camera_viewer.py \
        --cam1-topic /cam1/cam1/color/image_raw/compressed \
        --cam2-topic /cam2/cam2/color/image_raw/compressed \
        --cam1-label "cam1 - SCENE - 151623020789" \
        --cam2-label "cam2 - WRIST - 322743060038"

Purpose: BEFORE trusting an autonomous real-robot policy deploy, a human needs to
eyeball both camera feeds at once and confirm which physical camera is which --
if they're swapped, the policy sees the wrong thing. This shows cam1 on the LEFT
and cam2 on the RIGHT in a single window, each labelled, with a live FPS readout
and an obvious "STALLED" indicator if a feed drops.

Which is which: **cam2 is the WRIST camera**, rigidly mounted on the gripper, so
its whole background sweeps when the arm moves while the fingers hold the same
pixels. cam1 is the fixed third-person tripod view. Earlier docs called cam2 a
"close-up"/"workspace" camera; that was wrong and was corrected 2026-07-28 from
pixel measurements on the recorded dataset. Jogging the arm is the one-command
way to confirm.

Serials were updated 2026-07-28: the pair this file used to name
(147122072740 / 243222072700) is hardware this machine has never enumerated.

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
from std_srvs.srv import Trigger

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


# --- Trigger button bar ------------------------------------------------------
# Two clickable buttons drawn in a strip BELOW the two camera panes. They call
# the running policy_leader_node's std_srvs/Trigger services so the operator can
# arm/pause the autonomous policy WITHOUT a third terminal running
# `ros2 service call ...` by hand.
#
# SAFETY: these trigger REAL robot motion. policy_leader_node is launched AFTER
# this viewer (Terminal 2, run_ur7e_act_real.sh), so at startup the services do
# NOT exist yet -- the buttons stay dim/grey and unclickable until
# service_is_ready() flips true, then light up. A call in flight shows a distinct
# "..." state so the operator can't mash the button and double-fire.

# Full service names (policy_leader_node declares them as ~/... i.e. private).
START_SERVICE = "/policy_leader_node/start_execution"
HOLD_SERVICE = "/policy_leader_node/hold"

BUTTON_STRIP_HEIGHT = 90          # px tall strip appended below the camera row
_SERVICE_CHECK_INTERVAL_S = 0.5   # re-poll service_is_ready() no more often than
_CALL_TIMEOUT_S = 5.0             # an in-flight call is treated as failed after
_RESULT_DISPLAY_S = 4.0           # how long a success/fail result stays shown
_FLASH_S = 0.6                    # duration of the "not ready yet" click flash

# BGR colors (this window renders on a dark background).
_C_STRIP_BG = (25, 25, 25)
_C_WAIT_FILL = (60, 60, 60)       # service absent -> dim grey
_C_WAIT_TEXT = (150, 150, 150)
_C_START_FILL = (170, 120, 30)    # idle START -> blue/teal
_C_HOLD_FILL = (20, 140, 230)     # idle HOLD -> amber/orange
_C_INFLIGHT_TINT = 0.5            # multiply idle fill for the in-flight shade
_C_BORDER = (200, 200, 200)
_C_FLASH_BORDER = (0, 0, 255)     # red border flash on a rejected (not-ready) click
_C_IDLE_TEXT = (255, 255, 255)
_C_INFLIGHT_TEXT = (215, 215, 215)
_C_OK_TEXT = (60, 220, 60)        # green: response.success == True
_C_ERR_TEXT = (60, 60, 255)       # red: failure / exception / timeout


def _put_centered(img, text, rect, scale, color, thickness=2):
    """Draw ``text`` centered inside ``rect`` (x1, y1, x2, y2)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = rect
    cx = x1 + ((x2 - x1) - tw) // 2
    cy = y1 + ((y2 - y1) + th) // 2
    cv2.putText(img, text, (cx, cy), font, scale, color, thickness, cv2.LINE_AA)


class _TriggerButton:
    """One Trigger service button: owns its client, state machine, and result."""

    def __init__(self, node, service_name, label):
        self._client = node.create_client(Trigger, service_name)
        self.service_name = service_name
        self.label = label
        self.ready = False            # mirror of client.service_is_ready()
        self.future = None            # in-flight call_async future, or None
        self.inflight_start = None    # monotonic time the call was issued
        self.result_text = None       # last response/error text, or None
        self.result_ok = False        # last response.success
        self.result_expiry = 0.0      # monotonic time the result stops showing
        self.flash_expiry = 0.0       # monotonic time the not-ready flash ends
        self.rect = (0, 0, 0, 0)      # (x1,y1,x2,y2) in FULL-canvas coordinates

    def in_flight(self):
        return self.future is not None

    def refresh_ready(self):
        # A shutdown signal (Ctrl-C/SIGTERM) can invalidate the rclpy context
        # between the loop's `while rclpy.ok()` check and this call -- treat
        # that race as "not ready" instead of letting it crash the viewer.
        if not rclpy.ok():
            self.ready = False
            return
        try:
            self.ready = self._client.service_is_ready()
        except Exception:  # noqa: BLE001 -- shutdown race, not a real fault
            self.ready = False

    def poll(self, now):
        """Non-blocking: advance an in-flight call to done / timed-out."""
        if self.future is None:
            return
        if self.future.done():
            try:
                resp = self.future.result()
                self.result_ok = bool(resp.success)
                msg = (resp.message or "").strip()
                self.result_text = msg or ("OK" if resp.success else "failed")
            except Exception as exc:  # noqa: BLE001 -- surface any call failure
                self.result_ok = False
                self.result_text = f"call error: {exc}"
            self.future = None
            self.inflight_start = None
            self.result_expiry = now + _RESULT_DISPLAY_S
        elif self.inflight_start is not None and \
                (now - self.inflight_start) > _CALL_TIMEOUT_S:
            # No reply in a reasonable time -> treat as failed so the button
            # doesn't get stuck showing "..." forever.
            try:
                self.future.cancel()
            except Exception:  # noqa: BLE001
                pass
            self.future = None
            self.inflight_start = None
            self.result_ok = False
            self.result_text = f"timed out (>{_CALL_TIMEOUT_S:.0f}s)"
            self.result_expiry = now + _RESULT_DISPLAY_S

    def clicked(self, now):
        """A left-click landed inside this button's rect."""
        if self.in_flight():
            return  # a call is already pending -> ignore (no double-fire)
        if not self.ready:
            self.flash_expiry = now + _FLASH_S  # brief "not ready yet" flash
            return
        self.result_text = None                 # clear any stale result
        self.inflight_start = now
        self.future = self._client.call_async(Trigger.Request())

    def contains(self, x, y):
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


class TriggerButtonBar:
    """Lays out, draws, and routes clicks for the two Trigger buttons."""

    def __init__(self, node):
        self.start_btn = _TriggerButton(node, START_SERVICE, "START EXECUTION")
        self.hold_btn = _TriggerButton(node, HOLD_SERVICE, "HOLD")
        self._buttons = (self.start_btn, self.hold_btn)
        self._last_ready_check = 0.0

    def _layout(self, canvas_w):
        # Buttons live in the strip BELOW the PANE_HEIGHT-tall camera row, so
        # their y coordinates are offset by PANE_HEIGHT in FULL-canvas space.
        strip_y0 = PANE_HEIGHT
        margin, gap = 40, 40
        y1 = strip_y0 + 12
        y2 = strip_y0 + 12 + 46
        half = canvas_w // 2
        self.start_btn.rect = (margin, y1, half - gap // 2, y2)
        self.hold_btn.rect = (half + gap // 2, y1, canvas_w - margin, y2)

    def update(self, now, canvas_w):
        """Refresh layout, availability (throttled), and in-flight calls."""
        self._layout(canvas_w)
        if (now - self._last_ready_check) >= _SERVICE_CHECK_INTERVAL_S:
            self._last_ready_check = now
            for b in self._buttons:
                b.refresh_ready()
        for b in self._buttons:
            b.poll(now)

    def on_mouse(self, event, x, y, flags, param):  # cv2 setMouseCallback sig
        # (x, y) arrive in the shown image's pixel space (WINDOW_NORMAL maps the
        # window scaling back for us), i.e. the same FULL-canvas coords as rects.
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        now = time.monotonic()
        for b in self._buttons:
            if b.contains(x, y):
                b.clicked(now)
                break

    def render_strip(self, canvas_w, now):
        """Build the BUTTON_STRIP_HEIGHT x canvas_w strip to vconcat below."""
        strip = np.full((BUTTON_STRIP_HEIGHT, canvas_w, 3), _C_STRIP_BG,
                        dtype=np.uint8)
        self._draw_button(strip, self.start_btn, _C_START_FILL, now)
        self._draw_button(strip, self.hold_btn, _C_HOLD_FILL, now)
        return strip

    def _draw_button(self, strip, btn, idle_fill, now):
        # rect is FULL-canvas; convert y into strip-local space.
        x1, y1, x2, y2 = btn.rect
        sy1, sy2 = y1 - PANE_HEIGHT, y2 - PANE_HEIGHT
        srect = (x1, sy1, x2, sy2)

        if not btn.ready:
            fill, text_color = _C_WAIT_FILL, _C_WAIT_TEXT
            caption = f"{btn.label} (waiting for launch...)"
        elif btn.in_flight():
            fill = tuple(int(c * _C_INFLIGHT_TINT) for c in idle_fill)
            text_color = _C_INFLIGHT_TEXT
            caption = f"{btn.label} ..."
        else:
            fill, text_color = idle_fill, _C_IDLE_TEXT
            caption = btn.label

        cv2.rectangle(strip, (x1, sy1), (x2, sy2), fill, -1)
        border = _C_FLASH_BORDER if now < btn.flash_expiry else _C_BORDER
        cv2.rectangle(strip, (x1, sy1), (x2, sy2), border, 2)
        _put_centered(strip, caption, srect, 0.9, text_color, 2)

        # Result line (green ok / red fail) just under the button, for a few s.
        if btn.result_text is not None and now < btn.result_expiry:
            color = _C_OK_TEXT if btn.result_ok else _C_ERR_TEXT
            prefix = "OK: " if btn.result_ok else "ERR: "
            line = (prefix + btn.result_text)[:70]
            cv2.putText(strip, line, (x1 + 4, sy2 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


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

    # ASCII only: the Qt highgui backend's window-name lookup breaks on
    # non-ASCII characters (e.g. an em-dash) -- setMouseCallback() then fails
    # with "NULL window handler" even though namedWindow()/imshow() succeeded.
    window = "ACT cameras - cam1 SCENE | cam2 CLOSE-UP"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # Height accounts for the button strip appended below the two camera panes.
    cv2.resizeWindow(window, 1600, 510)
    # The Qt backend doesn't allocate a real window handle until the first
    # imshow() — calling setMouseCallback() any earlier raises "NULL window
    # handler". Show a blank placeholder first so the handle exists.
    cv2.imshow(window, np.zeros((510, 1600, 3), dtype=np.uint8))
    cv2.waitKey(1)

    # The button bar owns the two Trigger service clients; wiring its click
    # handler here (after the window exists) lets the same OpenCV window drive
    # policy_leader_node's ~/start_execution and ~/hold. spin_once() below both
    # dispatches the service replies and updates the camera frames.
    bar = TriggerButtonBar(node)
    cv2.setMouseCallback(window, bar.on_mouse)

    exit_code = 0
    try:
        while rclpy.ok():
            # Service ROS callbacks (non-blocking-ish) so latest frames update,
            # then service the OpenCV GUI event loop. A plain rclpy.spin() would
            # never yield to cv2.waitKey and the window would appear frozen.
            rclpy.spin_once(node, timeout_sec=0.005)

            left = _render_pane(node.cam1)
            right = _render_pane(node.cam2)
            camera_row = cv2.hconcat([left, right])

            now = time.monotonic()
            bar.update(now, camera_row.shape[1])
            strip = bar.render_strip(camera_row.shape[1], now)
            cv2.imshow(window, cv2.vconcat([camera_row, strip]))

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
