#!/usr/bin/env python3
"""Interactive PyQt5 recorder GUI for the GELLO leader-arm -> UR7e teleop pipeline.

This module is the operator-facing front-end. It:

  1. Launches two RealSense camera ROS2 nodes as subprocesses (one per physical
     camera) before/while the window comes up, mirroring ``run_recorder.sh``.
  2. Spins a :class:`GelloRecorderGuiNode` on a background daemon thread with a
     plain (single-threaded) ``rclpy.spin`` so the Qt event loop stays free.
  3. Shows a live dual-camera preview + a robot/GELLO state panel and drives
     start/stop recording from a bottom button bar.

The node (companion module ``gello_gui_node.py``) owns ALL ROS subscriptions,
decoding, warmup gating and recording logic. This file never touches ROS
directly beyond ``rclpy.init``/``spin``/``shutdown`` and the node's documented,
thread-safe getter methods -- Qt widgets are only ever touched from the Qt/main
thread (the QTimers poll the node from that thread).

console_script entry point: ``gello_recorder_gui = gello_recorder.gello_recorder_gui:main``
"""

import os
import signal
import subprocess
import sys
import threading
import time

import rclpy

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# The node lives in the companion module in this same package. Import is done
# lazily inside main() so that -m py_compile / --help style tooling on this file
# doesn't hard-fail while gello_gui_node.py is still being written.


# --------------------------------------------------------------------------- #
# Camera subprocess management
# --------------------------------------------------------------------------- #

# Defaults mirror run_recorder.sh exactly (same env var names + values).
# See launch_cameras.sh for why these changed on 2026-07-28.
DEFAULT_CAM1_SERIAL = "151623020789"   # plain D435
DEFAULT_CAM2_SERIAL = "322743060038"   # D435IF
DEFAULT_CAM1_NAME = "cam1"
DEFAULT_CAM2_NAME = "cam2"
DEFAULT_COLOR_PROFILE = "1280x720x30"

# Shared sizing for the big operator-facing buttons (Start/Stop/Pause/Resume) so
# they're easy to hit one-handed while the other hand holds the GELLO leader.
# Kept separate from color/weight so the armed "Confirm Resume" state can layer
# its own background color on top without losing the size.
_BIG_BUTTON_STYLE = "font-size: 16pt; padding: 14px 22px; min-height: 48px;"


def _launch_realsense(camera_name, serial, color_profile):
    """Launch one realsense2_camera node via ``ros2 launch`` in its own session.

    Returns the ``subprocess.Popen`` handle. The process gets its own process
    group (``start_new_session=True``) so we can later kill the WHOLE group --
    a bare ``terminate()`` on the ``ros2 launch`` wrapper does NOT reliably reap
    the ``realsense2_camera_node`` child it spawns, which leaves orphaned camera
    processes fighting over the USB device on the next run.

    NOTE the ``serial_no`` / ``rgb_camera.color_profile`` argv elements carry
    *embedded* single quotes in the value itself. ``ros2 launch`` type-infers
    bare ``key:=value`` CLI args from their content, so an all-digit serial gets
    coerced to an integer and the node (which declares serial_no as a string)
    dies instantly. Wrapping the value as ``serial_no:='147122072740'`` -- quote
    characters included in the argv string, NOT shell quoting -- forces a string.
    """
    argv = [
        "ros2", "launch", "realsense2_camera", "rs_launch.py",
        "camera_name:={}".format(camera_name),
        "camera_namespace:={}".format(camera_name),
        "serial_no:='{}'".format(serial),
        "rgb_camera.color_profile:='{}'".format(color_profile),
    ]
    # start_new_session=True == preexec_fn=os.setsid: new process group/session.
    return subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _kill_process_group(popen, term_wait_s=3.0):
    """SIGTERM the whole process group of ``popen``; SIGKILL if it lingers."""
    if popen is None:
        return
    if popen.poll() is not None:
        return  # already dead
    try:
        pgid = os.getpgid(popen.pid)
    except (ProcessLookupError, OSError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return
    # Give it a moment to shut down gracefully, then force-kill.
    deadline = time.monotonic() + term_wait_s
    while time.monotonic() < deadline:
        if popen.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


# --------------------------------------------------------------------------- #
# State-panel field layout
# --------------------------------------------------------------------------- #

# (snapshot_key, human_label) pairs, rendered in order in the right-hand panel.
_STATE_FIELDS = [
    ("gello_q", "GELLO q"),
    ("gello_qd", "GELLO qd"),
    ("gello_grip", "GELLO grip"),
    ("ur_q", "UR q"),
    ("ur_qd", "UR qd"),
    ("ur_eff", "UR effort"),
    ("grip_cmd", "Grip cmd"),
    ("grip_pos", "Grip pos"),
    ("wrench", "Wrench"),
    ("tcp", "TCP pose"),
]


def _fmt_value(value):
    """Render a snapshot value (scalar / list / None) as a compact string."""
    if value is None:
        return "—"  # em dash
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "—"
        return "[" + ", ".join(_fmt_scalar(v) for v in value) + "]"
    return _fmt_scalar(value)


def _fmt_scalar(v):
    if v is None:
        return "—"
    try:
        return "{:+.3f}".format(float(v))
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- #
# Teleop control panel (added) -- state -> colour map
# --------------------------------------------------------------------------- #
# Covers both arm (PAUSED|WAITING|STALE|CHASING|FOLLOWING) and gripper
# (PAUSED|WAITING|RAMPING|FOLLOWING) vocabularies. Anything unknown/None -> gray.
_TELEOP_STATE_COLORS = {
    "PAUSED": "#cc3333",     # red
    "STALE": "#cc3333",      # red
    "CHASING": "#dd8800",    # orange
    "RAMPING": "#dd8800",    # orange
    "FOLLOWING": "#22aa22",  # green
    "WAITING": "#888888",    # gray
}
_TELEOP_GRAY = "#888888"
# A state-topic reading older than this many seconds is treated as unknown.
_TELEOP_STATE_STALE_S = 2.0


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    """Operator window: dual camera preview + state panel + record controls."""

    def __init__(self, node, cam1_proc, cam2_proc):
        super().__init__()
        self._node = node
        self._cam1_proc = cam1_proc
        self._cam2_proc = cam2_proc
        self._cameras_killed = False

        self._record_start_wall = None  # time.monotonic() at Start, for elapsed

        # --- Teleop control panel state (added block) --------------------- #
        # Two-click resume confirm gate + last statusBar-shown teleop message.
        self._teleop_resume_armed = False
        self._teleop_last_shown_msg = None

        self.setWindowTitle("GELLO -> UR7e Recorder")
        self._build_ui()

        # --- Timers (all fire on the Qt/main thread) ---------------------- #
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._preview_timer.start(33)   # ~30 Hz

        self._state_timer = QTimer(self)
        self._state_timer.timeout.connect(self._refresh_state)
        self._state_timer.start(100)    # ~10 Hz

        self._control_timer = QTimer(self)
        self._control_timer.timeout.connect(self._refresh_controls)
        self._control_timer.start(200)  # ~5 Hz

        self._refresh_controls()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- Top: cameras (left/center) + state panel (right) ------------- #
        top = QHBoxLayout()
        root.addLayout(top, stretch=1)

        cams_box = QGroupBox("Camera preview")
        cams_layout = QHBoxLayout(cams_box)

        self._cam1_label = self._make_video_pane("cam1")
        self._cam2_label = self._make_video_pane("cam2")
        cams_layout.addWidget(self._cam1_label, stretch=1)
        cams_layout.addWidget(self._cam2_label, stretch=1)
        top.addWidget(cams_box, stretch=3)

        top.addWidget(self._build_state_panel(), stretch=1)

        # --- Teleop control bar (added) -- directly above record controls -- #
        root.addLayout(self._build_teleop_bar())

        # --- Bottom: record control bar ----------------------------------- #
        root.addLayout(self._build_control_bar())

    def _make_video_pane(self, name):
        label = QLabel("{}: no signal".format(name))
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(320, 240)
        label.setStyleSheet(
            "background-color: #202020; color: #999999; border: 1px solid #444;"
        )
        # Let the pane shrink/grow; we scale pixmaps to its current size.
        label.setScaledContents(False)
        return label

    def _build_state_panel(self):
        box = QGroupBox("Robot / GELLO state")
        layout = QVBoxLayout(box)

        # Camera live/stale indicators.
        self._cam1_status = QLabel("cam1: --")
        self._cam2_status = QLabel("cam2: --")
        layout.addWidget(self._cam1_status)
        layout.addWidget(self._cam2_status)

        sep = QLabel("")
        layout.addWidget(sep)

        # One label per state field. Keyed by snapshot key for updates.
        self._state_value_labels = {}
        for key, human in _STATE_FIELDS:
            row = QHBoxLayout()
            name_label = QLabel("{}:".format(human))
            name_label.setMinimumWidth(90)
            value_label = QLabel("—")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setWordWrap(True)
            row.addWidget(name_label)
            row.addWidget(value_label, stretch=1)
            layout.addLayout(row)
            self._state_value_labels[key] = value_label

        layout.addStretch(1)
        return box

    def _build_teleop_bar(self):
        """Teleop pause/resume bar (added block).

        Lets a solo operator pause the leader->robot signal path (single click)
        and resume it (two-click confirm, because resume initiates physical
        robot motion). State labels reflect the two bridge /state topics. All
        wiring is non-blocking: buttons call the node's call_async helpers and
        a QTimer polls get_teleop_status().
        """
        bar = QHBoxLayout()

        title = QLabel("Teleop:")
        title.setStyleSheet("font-weight: bold;")

        self._teleop_arm_label = QLabel("arm: --")
        self._teleop_arm_label.setMinimumWidth(150)
        self._teleop_grip_label = QLabel("gripper: --")
        self._teleop_grip_label.setMinimumWidth(150)

        self._teleop_pause_button = QPushButton("Pause Teleop")
        self._teleop_pause_button.setStyleSheet(_BIG_BUTTON_STYLE)
        self._teleop_pause_button.clicked.connect(self._on_teleop_pause_clicked)

        # Resting label spells out "click twice" up front -- a first-time
        # operator should never be surprised that one click doesn't move the
        # robot. The armed (post-first-click) label in
        # _on_teleop_resume_clicked spells out the second click too.
        self._teleop_resume_button = QPushButton("Resume Teleop (click twice)")
        self._teleop_resume_button.setStyleSheet(_BIG_BUTTON_STYLE)
        self._teleop_resume_button.clicked.connect(self._on_teleop_resume_clicked)
        self._teleop_resume_button.setEnabled(False)

        bar.addWidget(title)
        bar.addSpacing(8)
        bar.addWidget(self._teleop_arm_label)
        bar.addWidget(self._teleop_grip_label)
        bar.addStretch(1)
        bar.addWidget(self._teleop_pause_button)
        bar.addWidget(self._teleop_resume_button)
        return bar

    def _build_control_bar(self):
        bar = QHBoxLayout()

        self._start_button = QPushButton("Start Recording")
        self._start_button.setStyleSheet(_BIG_BUTTON_STYLE)
        self._start_button.clicked.connect(self._on_start_clicked)
        self._stop_button = QPushButton("Stop Recording")
        self._stop_button.setStyleSheet(_BIG_BUTTON_STYLE)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self._stop_button.setEnabled(False)

        self._take_label = QLabel("Take: 0")
        self._elapsed_label = QLabel("")
        self._status_label = QLabel("PREVIEW")
        self._status_label.setStyleSheet("font-weight: bold;")

        bar.addWidget(self._start_button)
        bar.addWidget(self._stop_button)
        bar.addSpacing(20)
        bar.addWidget(self._take_label)
        bar.addSpacing(20)
        bar.addWidget(self._elapsed_label)
        bar.addStretch(1)
        bar.addWidget(self._status_label)
        return bar

    # -------------------------------------------------------------- timers --
    def _refresh_preview(self):
        """~30 Hz: pull the latest decoded frames and blit them to the panes."""
        cam1_frame, cam2_frame = self._node.get_preview_frames()
        self._update_video_pane(self._cam1_label, cam1_frame, "cam1")
        self._update_video_pane(self._cam2_label, cam2_frame, "cam2")

    def _update_video_pane(self, label, frame, name):
        if frame is None:
            label.setText("{}: no signal".format(name))
            label.setPixmap(QPixmap())  # clear any stale frame
            return
        # frame is a numpy uint8 (H, W, 3) BGR array (straight from cv2.imdecode).
        height, width = frame.shape[0], frame.shape[1]
        # BGR -> RGB. Avoid a hard cv2 dependency for a simple channel swap.
        rgb = frame[:, :, ::-1]
        # QImage needs contiguous memory; the reverse-strided view is not, so copy.
        rgb = rgb.copy()
        bytes_per_line = 3 * width
        image = QImage(
            rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    def _refresh_state(self):
        """~10 Hz: update the right-hand state panel + camera live/stale text."""
        snap = self._node.get_state_snapshot()

        for key, _human in _STATE_FIELDS:
            label = self._state_value_labels.get(key)
            if label is not None:
                label.setText(_fmt_value(snap.get(key)))

        self._update_cam_status(
            self._cam1_status, "cam1", snap.get("cam1_last_frame_age_s")
        )
        self._update_cam_status(
            self._cam2_status, "cam2", snap.get("cam2_last_frame_age_s")
        )

    def _update_cam_status(self, label, name, age):
        if age is None:
            label.setText("{}: no frames".format(name))
            label.setStyleSheet("color: #cc3333; font-weight: bold;")
            return
        try:
            age = float(age)
        except (TypeError, ValueError):
            label.setText("{}: ?".format(name))
            label.setStyleSheet("color: #cc3333;")
            return
        if age < 0.5:
            color = "#22aa22"   # green: live
            word = "live"
        elif age < 2.0:
            color = "#dd8800"   # orange: lagging
            word = "lagging"
        else:
            color = "#cc3333"   # red: stale
            word = "STALE"
        label.setText("{}: {} ({:.2f}s)".format(name, word, age))
        label.setStyleSheet("color: {}; font-weight: bold;".format(color))

    def _refresh_controls(self):
        """~5 Hz: drive button enable/disable + status/take/elapsed labels."""
        recording = self._node.is_recording()
        ready = self._node.cameras_ready()
        take_n = self._node.take_index()

        if recording:
            self._start_button.setEnabled(False)
            self._start_button.setText("Recording...")
            self._stop_button.setEnabled(True)
            self._status_label.setText("● RECORDING take {}".format(take_n))
            self._status_label.setStyleSheet("color: #cc3333; font-weight: bold;")
            if self._record_start_wall is not None:
                elapsed = time.monotonic() - self._record_start_wall
                self._elapsed_label.setText("Elapsed: {:.1f}s".format(elapsed))
        else:
            self._stop_button.setEnabled(False)
            self._elapsed_label.setText("")
            self._status_label.setText("PREVIEW")
            self._status_label.setStyleSheet("font-weight: bold;")
            if ready:
                self._start_button.setEnabled(True)
                self._start_button.setText("Start Recording")
            else:
                self._start_button.setEnabled(False)
                remaining = self._node.warmup_seconds_remaining()
                if remaining > 3600 or remaining == float("inf"):
                    self._start_button.setText("Warming up...")
                else:
                    self._start_button.setText(
                        "Warming up... {:.1f}s".format(remaining)
                    )

        # Take counter reflects completed takes; while recording show current.
        if recording:
            self._take_label.setText("Take: {} (recording)".format(take_n))
        else:
            self._take_label.setText("Take: {}".format(take_n))

        # Teleop panel shares this ~5 Hz timer (added block).
        self._refresh_teleop()

    # ---------------------------------------------------- teleop (added) --
    def _refresh_teleop(self):
        """~5 Hz: drive teleop state labels + pause/resume button enablement."""
        st = self._node.get_teleop_status()
        arm_state = st["arm_state"]
        arm_age = st["arm_state_age_s"]
        pending = st["pending"]

        # Treat a missing or stale state-topic reading as unknown (gray).
        arm_live = (
            arm_state is not None
            and arm_age is not None
            and arm_age < _TELEOP_STATE_STALE_S
        )
        self._apply_teleop_label(
            self._teleop_arm_label, "arm",
            arm_state if arm_live else None,
        )

        # Gripper may be entirely absent (sim) -> gray 'gripper: n/a'.
        if not st["grip_available"]:
            self._teleop_grip_label.setText("gripper: n/a")
            self._teleop_grip_label.setStyleSheet(
                "color: {}; font-weight: bold;".format(_TELEOP_GRAY))
        else:
            self._apply_teleop_label(
                self._teleop_grip_label, "gripper", st["grip_state"])

        # Button enablement. Pause is always available (unconditional service)
        # except while a request is in flight. Resume only when the ARM reports
        # PAUSED and nothing is pending. While the two-click confirm is armed we
        # keep the resume button enabled so the confirming click lands.
        self._teleop_pause_button.setEnabled(not pending)
        if self._teleop_resume_armed:
            self._teleop_resume_button.setEnabled(True)
        else:
            self._teleop_resume_button.setEnabled(
                arm_live and arm_state == "PAUSED" and not pending)

        # Surface the latest service reply (incl. a benign refusal) once.
        last_msg = st["last_msg"]
        if last_msg and last_msg != self._teleop_last_shown_msg:
            self._teleop_last_shown_msg = last_msg
            self.statusBar().showMessage("Teleop: {}".format(last_msg), 6000)

    def _apply_teleop_label(self, label, prefix, state):
        if not state:
            label.setText("{}: --".format(prefix))
            label.setStyleSheet(
                "color: {}; font-weight: bold;".format(_TELEOP_GRAY))
            return
        color = _TELEOP_STATE_COLORS.get(state, _TELEOP_GRAY)
        label.setText("{}: {}".format(prefix, state))
        label.setStyleSheet("color: {}; font-weight: bold;".format(color))

    def _on_teleop_pause_clicked(self):
        # Single click: pause is unconditional. Non-blocking call_async in node.
        fired = self._node.request_teleop_pause()
        if fired:
            self.statusBar().showMessage("Teleop: pause requested.", 3000)
        else:
            self.statusBar().showMessage(
                "Teleop: pause not sent (request in flight or service down).",
                3000)

    def _on_teleop_resume_clicked(self):
        # Two-click confirm (no modal dialogs in this codebase): the first click
        # arms an orange 'Confirm' button that reverts after 3 s; the second
        # click within that window actually requests resume (robot WILL move).
        if not self._teleop_resume_armed:
            self._teleop_resume_armed = True
            self._teleop_resume_button.setText("Click AGAIN to Resume (robot will move!)")
            self._teleop_resume_button.setStyleSheet(
                _BIG_BUTTON_STYLE +
                "background-color: #dd8800; color: white; font-weight: bold;")
            QTimer.singleShot(3000, self._disarm_teleop_resume)
            self.statusBar().showMessage(
                "Click again within 3 s to Resume -- the robot WILL move.", 3000)
            return
        self._disarm_teleop_resume()
        fired = self._node.request_teleop_resume()
        if fired:
            self.statusBar().showMessage("Teleop: resume requested...", 3000)
        else:
            self.statusBar().showMessage(
                "Teleop: resume not sent (request in flight or service down).",
                3000)

    def _disarm_teleop_resume(self):
        # Revert the confirm button to its resting look; the ~5 Hz refresh
        # re-drives its enabled state on the next tick.
        self._teleop_resume_armed = False
        self._teleop_resume_button.setText("Resume Teleop (click twice)")
        self._teleop_resume_button.setStyleSheet(_BIG_BUTTON_STYLE)

    # ------------------------------------------------------------- actions --
    def _on_start_clicked(self):
        # Guarded by the button being disabled unless ready & not recording, but
        # re-check to be safe against a race with the 5 Hz control timer.
        if self._node.is_recording() or not self._node.cameras_ready():
            return
        try:
            take_dir = self._node.start_recording()
        except RuntimeError as exc:
            self.statusBar().showMessage("Start failed: {}".format(exc), 5000)
            return
        self._record_start_wall = time.monotonic()
        self.statusBar().showMessage("Recording -> {}".format(take_dir), 5000)
        self._refresh_controls()

    def _on_stop_clicked(self):
        if not self._node.is_recording():
            return
        try:
            meta = self._node.stop_recording()
        except RuntimeError as exc:
            self.statusBar().showMessage("Stop failed: {}".format(exc), 5000)
            return
        self._record_start_wall = None
        self._show_stop_confirmation(meta)
        self._refresh_controls()

    def _show_stop_confirmation(self, meta):
        try:
            duration = meta.get("duration_s")
            session_dir = meta.get("session_dir")
            msg = "Saved take: {} ({:.1f}s)".format(session_dir, float(duration))
        except (AttributeError, TypeError, ValueError):
            msg = "Take saved."
        self.statusBar().showMessage(msg, 8000)

    # -------------------------------------------------------------- close ---
    def _shutdown_cameras(self):
        if self._cameras_killed:
            return
        self._cameras_killed = True
        _kill_process_group(self._cam1_proc)
        _kill_process_group(self._cam2_proc)

    def closeEvent(self, event):
        # 1. Never abandon an in-progress take -- finalize it first.
        try:
            if self._node.is_recording():
                self._node.stop_recording()
        except Exception:  # noqa: BLE001 -- best-effort on shutdown
            pass

        # 2. Kill BOTH camera process groups (SIGTERM then SIGKILL fallback).
        self._shutdown_cameras()

        # 3. Tear ROS down. spin() on the daemon thread will return.
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

        event.accept()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main(args=None):
    # Config from env vars, matching run_recorder.sh's names/defaults.
    cam1_serial = os.environ.get("CAM1_SERIAL", DEFAULT_CAM1_SERIAL)
    cam2_serial = os.environ.get("CAM2_SERIAL", DEFAULT_CAM2_SERIAL)
    cam1_name = os.environ.get("CAM1_NAME", DEFAULT_CAM1_NAME)
    cam2_name = os.environ.get("CAM2_NAME", DEFAULT_CAM2_NAME)
    color_profile = os.environ.get("COLOR_PROFILE", DEFAULT_COLOR_PROFILE)

    # Parse "WxHxFPS" -> fps for the node's camera_fps arg (best-effort).
    camera_fps = 30.0
    try:
        camera_fps = float(color_profile.split("x")[-1])
    except (ValueError, IndexError):
        pass
    camera_warmup_s = float(os.environ.get("CAMERA_WARMUP_S", "4.0"))
    output_root = os.environ.get(
        "RECORDER_OUTPUT_ROOT",
        os.path.join(
            os.environ.get(
                "GELLO_REPO_ROOT",
                os.path.expanduser("~/gello_software"),
            ),
            "ros2_ur_ws", "gello_logs",
        ),
    )

    # The two color topics the realsense nodes publish under their namespace.
    cam1_topic = "/{0}/{0}/color/image_raw/compressed".format(cam1_name)
    cam2_topic = "/{0}/{0}/color/image_raw/compressed".format(cam2_name)

    # --- 1. Launch the two RealSense camera nodes as subprocesses --------- #
    cam1_proc = _launch_realsense(cam1_name, cam1_serial, color_profile)
    cam2_proc = _launch_realsense(cam2_name, cam2_serial, color_profile)

    # --- 2. Bring up ROS + the node --------------------------------------- #
    rclpy.init(args=args)

    # Imported here (not at module top) so this module still py_compiles while
    # gello_gui_node.py is being written in parallel.
    from gello_recorder.gello_gui_node import GelloRecorderGuiNode

    # opencv-python's wheel bundles its own copy of Qt5 (incl. platform plugins)
    # and points QT_QPA_PLATFORM_PLUGIN_PATH at it as a side effect of `import
    # cv2` (pulled in transitively by gello_gui_node -> cv2.imdecode). That
    # bundled copy is a different Qt5 build than the system PyQt5 we actually
    # want, and loading both in one process breaks Qt's xcb platform plugin
    # (and can cause spurious QObject::moveToThread warnings). Clear it so Qt
    # falls back to the system-installed platform plugins before any
    # QApplication is constructed.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    node = GelloRecorderGuiNode(
        cam1_topic=cam1_topic,
        cam2_topic=cam2_topic,
        camera_fps=camera_fps,
        camera_warmup_s=camera_warmup_s,
        output_root=output_root,
    )

    # --- 3. Spin the node on a background daemon thread ------------------- #
    # Plain SingleThreadedExecutor via rclpy.spin -- no MultiThreadedExecutor.
    spin_thread = threading.Thread(
        target=_spin_node, args=(node,), daemon=True
    )
    spin_thread.start()

    # --- 4. Qt event loop on the main thread ------------------------------ #
    app = QApplication(sys.argv if args is None else args)
    window = MainWindow(node, cam1_proc, cam2_proc)
    window.resize(1280, 720)
    window.show()

    # Ctrl-C in the launching terminal is otherwise SILENTLY SWALLOWED: Qt's
    # C++ event loop never yields back to the Python interpreter on its own,
    # so Python's SIGINT handler doesn't get a chance to run (a well-known
    # PyQt gotcha) -- if a KeyboardInterrupt happens to land inside a Qt slot
    # anyway (e.g. a QTimer callback), PyQt5 just prints the traceback and
    # keeps the event loop running, which looks exactly like "Ctrl-C did
    # nothing." Fix: make SIGINT ask Qt to quit, and use a short-interval
    # QTimer purely to tick the interpreter often enough for that handler to
    # actually fire promptly (Qt's own timers don't wake up the signal
    # machinery by themselves).
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    sigint_pump = QTimer()
    sigint_pump.start(200)
    sigint_pump.timeout.connect(lambda: None)

    exit_code = app.exec_()

    # Belt-and-suspenders cleanup in case closeEvent didn't run (it normally
    # does for a window close). These are all idempotent / guarded.
    window._shutdown_cameras()
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    spin_thread.join(timeout=2.0)

    sys.exit(exit_code)


def _spin_node(node):
    """Run rclpy.spin(node); swallow the shutdown-time exception cleanly."""
    try:
        rclpy.spin(node)
    except Exception:  # noqa: BLE001 -- spin raises when shutdown() is called
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
