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
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# The node CLASS lives in the companion module gello_gui_node and is imported
# inside main(). The pure depth-topic helper is re-exported here at module
# level because the headless node and the tests need the SAME single
# definition without going through main().
from gello_recorder.gello_gui_node import depth_topics_for  # noqa: F401 (re-export)
from gello_recorder.camera_defaults import load_default_serials
from gello_recorder import take_tag
from gello_recorder.paths import default_repo_root
from gello_recorder.spin_health import (  # noqa: F401 (DEPTH_ON_BANNER re-export)
    DEPTH_ON_BANNER,
    stop_health_suffix as _stop_health_suffix,
)


# --------------------------------------------------------------------------- #
# Camera subprocess management
# --------------------------------------------------------------------------- #

# Camera serial defaults come from ONE file shared with the shell launchers:
# ros2_ur_ws/camera_serials.env (see gello_recorder.camera_defaults). The
# CAM1_SERIAL / CAM2_SERIAL env vars override it per run. The defaults are a
# PREFERENCE, not a requirement: _resolve_camera_serials() below checks them
# against what pyrealsense2 actually enumerates and falls back (with a loud
# warning) when the configured pair is not plugged in.
#
# serial_no:= matches the DEVICE serial_number, not the ASIC serial that the
# kernel USB descriptor (and therefore journalctl) shows. Binding a serial that
# does not resolve does NOT fail loudly -- the camera node starts and publishes
# nothing -- which is why the pair is checked here instead of trusted blindly.
DEFAULT_CAM1_SERIAL, DEFAULT_CAM2_SERIAL = load_default_serials()
DEFAULT_CAM1_NAME = "cam1"
DEFAULT_CAM2_NAME = "cam2"
DEFAULT_COLOR_PROFILE = "1280x720x30"

# Depth is OPT-IN and OFF by default (RGB-only capture):
#
#     ENABLE_DEPTH=1 ros2 run gello_recorder gello_recorder_gui
#
# Depth adds two 30 Hz subscriptions per camera and a ~6 MB/s HDF5 write. When
# that load landed on the same rclpy spin thread as the robot topics, the
# executor fell behind and robot rows were silently read out of full queues
# (stamped up to ~0.9 s late). Header stamps, small queues, a background frame
# writer and a starvation watchdog (gello_recorder.spin_health) now prevent and
# detect that, but the disk/CPU cost remains -- so depth is asked for per
# session. Only the values in _DEPTH_ON_VALUES turn it on (a typo leaves it
# off, the safe direction). An explicit enable_depth kwarg wins over the env.
_DEPTH_ENV_VAR = "ENABLE_DEPTH"
_DEPTH_ON_VALUES = ("1", "true", "yes", "on")
_DEPTH_DEFAULT = "0"

# Depth->color ALIGNMENT is separately opt-in via ALIGN_DEPTH. In-node
# alignment raised realsense2_camera_node from ~9 % to ~49 % CPU and dropped
# both streams from 30 Hz to ~25 Hz. The recorded intrinsics/extrinsics allow
# the alignment to be done offline instead.
_ALIGN_ENV_VAR = "ALIGN_DEPTH"
_ALIGN_DEFAULT = "0"


def _depth_enabled_from_env():
    """True iff ENABLE_DEPTH opts IN to depth streaming (default: off)."""
    raw = os.environ.get(_DEPTH_ENV_VAR, _DEPTH_DEFAULT)
    return raw.strip().lower() in _DEPTH_ON_VALUES


def _align_depth_from_env():
    """True iff ALIGN_DEPTH opts in to depth->color alignment (default: off)."""
    raw = os.environ.get(_ALIGN_ENV_VAR, _ALIGN_DEFAULT)
    return raw.strip().lower() in _DEPTH_ON_VALUES


def _resolve_camera_serials(want1, want2):
    """Resolve (cam1_serial, cam2_serial) against the live USB bus.

    In-process Python equivalent of resolve_serials() in
    ``ros2_ur_ws/_resolve_camera_serials.sh`` (the shared helper sourced by
    ``launch_cameras.sh`` / ``run_recorder.sh``) -- reimplemented directly
    against pyrealsense2 here rather than shelling out, since this file
    already depends on pyrealsense2 transitively (via the realsense2_camera
    node it launches) and there is no ROS graph to shell into yet at this
    point in startup. Keep the two in sync by hand if the policy changes.

    Enumeration only (``rs.context().query_devices()``) -- this never opens a
    streaming lock, so it is safe to call before the realsense2_camera nodes
    (or anything else) hold the cameras.

    Returns:
        ``(serial1, serial2, warnings)`` where ``warnings`` is a list of
        human-readable strings. An empty list means the configured pair was
        found connected and passed through unchanged (nothing to report).

    Raises:
        ImportError: pyrealsense2 is not importable. Callers should treat
            this as non-fatal and fall back to the configured serials as-is
            -- the realsense2_camera node does its own serial lookup anyway.
        RuntimeError: fewer than 2 RealSense devices are enumerated. Callers
            should treat this as fatal: there is nothing useful to launch.
    """
    import pyrealsense2 as rs  # may raise ImportError -- caller decides

    devs = [
        (d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
        for d in rs.context().query_devices()
    ]

    if len(devs) < 2:
        lines = ["    connected: {} {}".format(n, s) for n, s in devs]
        raise RuntimeError(
            "only {} RealSense device(s) found, need 2\n{}\n"
            "Override explicitly with CAM1_SERIAL=<serial> CAM2_SERIAL=<serial>"
            .format(len(devs), "\n".join(lines))
        )

    present = {s for _, s in devs}
    if want1 in present and want2 in present:
        return want1, want2, []

    warnings = [
        "configured serials ({}, {}) are not both connected".format(want1, want2)
    ]
    warnings += ["    connected: {} {}".format(n, s) for n, s in devs]

    # IMU/IF variants report a name containing "D435I..."; the plain unit
    # reports "D435".
    imu = [s for n, s in devs if "d435i" in n.lower()]
    plain = [s for n, s in devs if "d435i" not in n.lower()]

    if len(plain) == 1 and len(imu) == 1:
        warnings.append(
            "auto-selected by model class: cam1={} (plain D435), "
            "cam2={} (D435IF/i)".format(plain[0], imu[0])
        )
        return plain[0], imu[0], warnings

    # Ambiguous: same model class on both mounts. Order is arbitrary, so say so.
    ordered = sorted(s for _, s in devs)[:2]
    warnings.append(
        "model classes are ambiguous; falling back to serial sort order: "
        "cam1={} cam2={}".format(ordered[0], ordered[1])
    )
    warnings.append("VERIFY THE PANES BEFORE RECORDING -- cam1/cam2 may be swapped")
    return ordered[0], ordered[1], warnings


# Shared sizing for the big operator-facing buttons (Start/Stop/Pause/Resume) so
# they're easy to hit one-handed while the other hand holds the GELLO leader.
# Kept separate from color/weight so the armed "Confirm Resume" state can layer
# its own background color on top without losing the size.
_BIG_BUTTON_STYLE = "font-size: 16pt; padding: 14px 22px; min-height: 48px;"

# "Delete last take" is the deliberate opposite of the big buttons above: a
# rare, destructive, easy-to-fat-finger action, not something reached for on
# every take the way Start/Stop are. Small + muted red (not filled, not bold)
# says "here if you need it, and dangerous" without competing for thumb reach
# or catching the eye the way a primary control would -- an operator scanning
# the bar for Start/Stop should not even register it until they go looking.
_DELETE_BUTTON_STYLE = (
    "font-size: 9pt; padding: 2px 10px; color: #cc3333; "
    "border: 1px solid #cc3333; background-color: transparent;"
)

_DELETE_LAST_TAKE_TOOLTIP = (
    "Permanently deletes the take most recently STOPPED in this window -- "
    "never a take still recording, and never an older take (from a previous "
    "session, or from before an earlier delete). Asks for confirmation "
    "first."
)


def _realsense_argv(camera_name, serial, color_profile, enable_depth=None,
                    align_depth=None):
    """Build the ``ros2 launch`` argv for one realsense2_camera node.

    Split out from :func:`_launch_realsense` so the argv can be asserted on
    without spawning anything -- getting one of these elements wrong fails at
    the camera, not at the call, which is expensive to notice.

    ``enable_depth=None`` defers to :func:`_depth_enabled_from_env`; pass a bool
    to override the env var (see ``_DEPTH_ENV_VAR`` above -- depth is OFF by
    default). ``align_depth=None`` likewise
    defers to :func:`_align_depth_from_env` (``ALIGN_DEPTH``, default off --
    see ``_ALIGN_ENV_VAR`` for the measured cost). The
    ``align_depth.enable:=`` element is emitted ONLY when depth is enabled:
    with depth off the argv is byte-identical to what it was before depth
    recording existed (alignment of a stream that is not running is
    meaningless, and rs_launch.py's own default for it is already false).

    NOTE the ``serial_no`` / ``rgb_camera.color_profile`` argv elements carry
    *embedded* single quotes in the value itself. ``ros2 launch`` type-infers
    bare ``key:=value`` CLI args from their content, so an all-digit serial gets
    coerced to an integer and the node (which declares serial_no as a string)
    dies instantly. Wrapping the value as ``serial_no:='123456789012'`` -- quote
    characters included in the argv string, NOT shell quoting -- forces a string.

    ``enable_depth`` needs no such wrapping: "true"/"false" is exactly what the
    type inference is supposed to read as a bool.

    ``serial_no`` is matched against the DEVICE ``camera_info.serial_number``,
    never the ASIC serial that ``journalctl`` shows (see the note near
    ``DEFAULT_CAM1_SERIAL``).
    """
    if enable_depth is None:
        enable_depth = _depth_enabled_from_env()
    if align_depth is None:
        align_depth = _align_depth_from_env()
    argv = [
        "ros2", "launch", "realsense2_camera", "rs_launch.py",
        "camera_name:={}".format(camera_name),
        "camera_namespace:={}".format(camera_name),
        "serial_no:='{}'".format(serial),
        "rgb_camera.color_profile:='{}'".format(color_profile),
        "enable_depth:={}".format("true" if enable_depth else "false"),
    ]
    if enable_depth:
        argv.append(
            "align_depth.enable:={}".format("true" if align_depth else "false"))
    return argv


def _launch_realsense(camera_name, serial, color_profile, enable_depth=None,
                      align_depth=None):
    """Launch one realsense2_camera node via ``ros2 launch`` in its own session.

    Returns the ``subprocess.Popen`` handle. The process gets its own process
    group (``start_new_session=True``) so we can later kill the WHOLE group --
    a bare ``terminate()`` on the ``ros2 launch`` wrapper does NOT reliably reap
    the ``realsense2_camera_node`` child it spawns, which leaves orphaned camera
    processes fighting over the USB device on the next run.

    Depth is OFF by default (opt in with ``ENABLE_DEPTH=1``) and then recorded
    to depth.h5 by the node; depth->color alignment is separately opt-in via
    ``ALIGN_DEPTH``.
    See :func:`_realsense_argv` for the argv contract.
    """
    argv = _realsense_argv(camera_name, serial, color_profile, enable_depth,
                           align_depth)
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

# Keep the live state table geometrically stable.  Without fixed dimensions,
# QLabel recalculates its size hint every time a sign/digit or wrap point
# changes in a vector.  In the integrated collection window that changing
# minimum width propagates through the horizontal splitter and makes the whole
# EEF pane visibly jitter at the state update rate.
_STATE_PANEL_WIDTH = 430
_STATE_VALUE_WIDTH = 310
_STATE_VALUE_HEIGHT = 38


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
# Teleop control panel -- state -> colour map
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


_RECORDING_BANNER_IDLE_STYLE = (
    "color: white; background-color: #455a64; border: 4px solid #263238; "
    "font-size: 18pt; font-weight: bold; padding: 10px;"
)
_RECORDING_BANNER_ACTIVE_STYLE = (
    "color: white; background-color: #d50000; border: 6px solid #ffeb3b; "
    "font-size: 23pt; font-weight: bold; padding: 12px;"
)
_RECORDING_PANEL_IDLE_STYLE = (
    "QWidget#recorderRecordingPanel {"
    "background-color: #eceff1; border: 5px solid #546e7a; }"
)
_RECORDING_PANEL_ACTIVE_STYLE = (
    "QWidget#recorderRecordingPanel {"
    "background-color: #ffebee; border: 12px solid #ff1744; }"
)


# --------------------------------------------------------------------------- #
# Delete-last-take confirmation sizing
# --------------------------------------------------------------------------- #

def _scan_take_dir(path):
    """Best-effort ``(n_files, total_bytes)`` for a take folder.

    Used ONLY to word the delete confirmation prompt ("this removes N files,
    X.Y MB") -- it is a hint for the operator reading the prompt, not a
    canonical accounting, and the actual delete is done by the node, not by
    this scan. Take folders are flat (mp4s/h5s/json sit directly under the
    take dir, see ``RecordingSession`` / ``GelloRecorderGuiNode.start_recording``),
    so a single ``os.scandir`` pass is enough -- no recursion.

    Swallows per-entry and whole-directory errors rather than raising: a
    partially-wrong count in the confirmation text is a cosmetic problem, and
    is not a reason to block the prompt (or crash it) over something as
    incidental as a file vanishing mid-scan or a permissions hiccup.
    """
    n_files = 0
    total_bytes = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        n_files += 1
                        total_bytes += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        pass
    return n_files, total_bytes


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    """Operator window: dual camera preview + state panel + record controls."""

    def __init__(self, node, cam1_proc, cam2_proc, cam_warnings=None):
        super().__init__()
        self._node = node
        self._cam1_proc = cam1_proc
        self._cam2_proc = cam2_proc
        self._cameras_killed = False
        self._cam_warnings = cam_warnings or []

        self._record_start_wall = None  # time.monotonic() at Start, for elapsed

        # Non-modal "Delete last take?" confirmation, or None when no prompt
        # is open. Kept as a reference (rather than a fire-and-forget popup)
        # for three reasons: a second click on the button must raise the
        # existing prompt instead of stacking a duplicate, Start Recording
        # must be able to close a stale one, and closeEvent must too -- see
        # _on_delete_last_take_clicked / _on_start_clicked / closeEvent.
        self._delete_box = None

        # --- Teleop control panel state ----------------------------------- #
        # Two-click resume confirm gate + last statusBar-shown teleop message.
        self._teleop_resume_armed = False
        self._teleop_last_shown_msg = None

        self.setWindowTitle("GELLO -> UR7e Recorder")
        self._build_ui()
        if self._cam_warnings:
            self._show_camera_warning()

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
        central.setObjectName("recorderRecordingPanel")
        central.setStyleSheet(_RECORDING_PANEL_IDLE_STYLE)
        self._recording_panel = central
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self._recording_banner = QLabel("○ NOT RECORDING — PREVIEW ONLY")
        self._recording_banner.setAlignment(Qt.AlignCenter)
        self._recording_banner.setMinimumHeight(72)
        self._recording_banner.setStyleSheet(_RECORDING_BANNER_IDLE_STYLE)
        root.addWidget(self._recording_banner)

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

        # --- Teleop control bar -- directly above record controls --------- #
        root.addLayout(self._build_teleop_bar())

        # --- Bottom: record control bar ----------------------------------- #
        root.addLayout(self._build_control_bar())

    def _show_camera_warning(self):
        """Make a resolved-but-mismatched camera serial pairing hard to miss.

        Called from __init__ only when ``_resolve_camera_serials()`` (see
        module-level function above) had to override the configured
        CAM1_SERIAL/CAM2_SERIAL -- i.e. the connected pair was not what was
        asked for. The full detail is already on the console (main() prints
        every entry of ``self._cam_warnings``); this adds a permanent,
        can't-miss banner in the GUI itself, since an operator running this
        as a double-clicked GUI app may never see the console.
        """
        detail = [w for w in self._cam_warnings if not w.strip().startswith("connected:")]
        summary = " | ".join(detail) if detail else "camera serials auto-resolved"
        banner = QLabel("CAMERA SERIALS AUTO-RESOLVED (not the configured pair) -- {}".format(summary))
        banner.setStyleSheet(
            "color: white; background-color: #cc3333; font-weight: bold; padding: 3px 8px;"
        )
        banner.setWordWrap(True)
        self.statusBar().addPermanentWidget(banner, stretch=1)

    def _make_video_pane(self, name):
        label = QLabel("{}: no signal".format(name))
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(320, 240)
        label.setStyleSheet(
            "background-color: #202020; color: #999999; border: 1px solid #444;"
        )
        # QLabel's default sizeHint follows its current QPixmap.  Since each
        # preview tick scales a new pixmap to the label's current dimensions,
        # feeding that hint back into the layout can make the two camera panes
        # oscillate by a few pixels forever.  Ignore pixmap-derived hints: the
        # surrounding layout owns the pane geometry and frames only paint into
        # the already-decided rectangle.
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        label.setScaledContents(False)
        return label

    def _build_state_panel(self):
        box = QGroupBox("Robot / GELLO state")
        box.setFixedWidth(_STATE_PANEL_WIDTH)
        layout = QVBoxLayout(box)

        # Camera live/stale indicators (+ one line for the depth streams).
        self._cam1_status = QLabel("cam1: --")
        self._cam2_status = QLabel("cam2: --")
        self._depth_status = QLabel("depth: --")
        # ros_lag = now - /joint_states header stamp. On screen because a
        # starved recorder once wrote robot rows ~0.9 s stale with nothing
        # visible anywhere; this is that number, live.
        self._ros_lag_status = QLabel("ros lag: --")
        layout.addWidget(self._cam1_status)
        layout.addWidget(self._cam2_status)
        layout.addWidget(self._depth_status)
        layout.addWidget(self._ros_lag_status)

        sep = QLabel("")
        layout.addWidget(sep)

        # One label per state field. Keyed by snapshot key for updates.
        self._state_value_labels = {}
        for key, human in _STATE_FIELDS:
            row = QHBoxLayout()
            name_label = QLabel("{}:".format(human))
            name_label.setMinimumWidth(90)
            value_label = QLabel("—")
            value_label.setFixedSize(_STATE_VALUE_WIDTH, _STATE_VALUE_HEIGHT)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setWordWrap(True)
            value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value_label.setStyleSheet("font-family: monospace;")
            row.addWidget(name_label)
            row.addWidget(value_label, stretch=1)
            layout.addLayout(row)
            self._state_value_labels[key] = value_label

        layout.addStretch(1)
        return box

    def _build_teleop_bar(self):
        """Teleop pause/resume bar.

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
        self._start_button.setStyleSheet(
            _BIG_BUTTON_STYLE
            + "background-color: #2e7d32; color: white; font-weight: bold;"
        )
        self._start_button.clicked.connect(self._on_start_clicked)
        self._stop_button = QPushButton("Stop Recording")
        self._stop_button.setStyleSheet(
            _BIG_BUTTON_STYLE
            + "background-color: #c62828; color: white; font-weight: bold;"
        )
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self._stop_button.setEnabled(False)

        self._take_label = QLabel("Take: 0")

        # Small + secondary on purpose -- see _DELETE_BUTTON_STYLE. Right next
        # to the take counter it replaces (the take it can delete IS the one
        # that counter is about to move past), enabled/disabled from
        # _refresh_controls like every other button here, never from the
        # click handler.
        self._delete_last_take_button = QPushButton("Delete last take")
        self._delete_last_take_button.setStyleSheet(_DELETE_BUTTON_STYLE)
        self._delete_last_take_button.setToolTip(_DELETE_LAST_TAKE_TOOLTIP)
        self._delete_last_take_button.setEnabled(False)
        self._delete_last_take_button.clicked.connect(
            self._on_delete_last_take_clicked
        )

        self._tag_combo = QComboBox()
        self._tag_combo.setEditable(True)
        self._tag_combo.addItems(take_tag.preset_tags())
        self._tag_combo.setCurrentIndex(-1)
        self._tag_combo.lineEdit().setPlaceholderText("tag...")
        self._tag_combo.setFixedWidth(125)
        self._tag_combo.setToolTip(
            "Tag the most recently stopped take (appended to its folder "
            "name). Pick a preset (RECORDER_TAG_PRESETS) or type a custom "
            "tag. Multiple tags may be appended."
        )
        self._tag_last_take_button = QPushButton("Tag last take")
        self._tag_last_take_button.setEnabled(False)
        self._tag_last_take_button.clicked.connect(
            self._on_tag_last_take_clicked
        )
        self._tag_combo.lineEdit().returnPressed.connect(
            self._on_tag_last_take_clicked
        )

        self._elapsed_label = QLabel("")
        self._status_label = QLabel("PREVIEW")
        self._status_label.setStyleSheet("font-weight: bold;")

        bar.addWidget(self._start_button)
        bar.addWidget(self._stop_button)
        bar.addSpacing(20)
        bar.addWidget(self._take_label)
        bar.addSpacing(8)
        bar.addWidget(self._delete_last_take_button)
        bar.addWidget(self._tag_combo)
        bar.addWidget(self._tag_last_take_button)
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
        self._update_depth_status(snap)
        self._update_ros_lag_status(snap)

    def _update_ros_lag_status(self, snap):
        """One line: how far behind the robot state stamps are running.

        GREEN under the node's warn threshold, RED over it. Over threshold the
        node is already emitting a throttled WARN; this is the operator-facing
        half of the same alarm, and it is the ONE readout that would have caught
        the 0.900 s timestamp artifact while it was still recordable."""
        lag = snap.get("ros_lag_s")
        warn_s = snap.get("ros_lag_warn_s")
        try:
            warn_s = float(warn_s)
        except (TypeError, ValueError):
            warn_s = 0.15
        age = snap.get("ros_lag_age_s")
        try:
            lag = None if lag is None else float(lag)
        except (TypeError, ValueError):
            lag = None
        if lag is None:
            self._ros_lag_status.setText("ros lag: -- (no /joint_states)")
            self._ros_lag_status.setStyleSheet("color: #888888;")
            return
        try:
            stale = age is not None and float(age) > 2.0
        except (TypeError, ValueError):
            stale = False
        peak = snap.get("ros_lag_s_max")
        try:
            peak_txt = "" if peak is None else " (max {:.2f}s)".format(float(peak))
        except (TypeError, ValueError):
            peak_txt = ""
        if stale:
            self._ros_lag_status.setText(
                "ros lag: {:.3f}s (no update){}".format(lag, peak_txt))
            self._ros_lag_status.setStyleSheet("color: #888888;")
            return
        if lag > warn_s:
            self._ros_lag_status.setText(
                "ros lag: {:.3f}s STALE ROWS{}".format(lag, peak_txt))
            self._ros_lag_status.setStyleSheet(
                "color: #cc3333; font-weight: bold;")
        else:
            self._ros_lag_status.setText(
                "ros lag: {:.3f}s{}".format(lag, peak_txt))
            self._ros_lag_status.setStyleSheet(
                "color: #22aa22; font-weight: bold;")

    def _update_depth_status(self, snap):
        """One line: 'depth: OFF' or 'depth: ON (cam1 x.xxs / cam2 x.xxs)'.

        Colour follows the STALER of the two depth streams with the same
        thresholds as the colour panes; '--' means depth is on but that camera
        has not delivered a single depth frame yet.
        """
        if not snap.get("depth_enabled"):
            self._depth_status.setText("depth: OFF")
            self._depth_status.setStyleSheet("color: #888888;")
            return
        ages = [snap.get("cam1_depth_last_frame_age_s"),
                snap.get("cam2_depth_last_frame_age_s")]
        parts = []
        worst = None
        for name, age in zip(("cam1", "cam2"), ages):
            try:
                age = None if age is None else float(age)
            except (TypeError, ValueError):
                age = None
            parts.append("{} {}".format(name, "--" if age is None else "{:.2f}s".format(age)))
            if age is None:
                worst = float("inf")
            elif worst is None or age > worst:
                worst = age
        if worst is None or worst == float("inf"):
            color = "#cc3333"
        elif worst < 0.5:
            color = "#22aa22"
        elif worst < 2.0:
            color = "#dd8800"
        else:
            color = "#cc3333"
        self._depth_status.setText("depth: ON ({})".format(" / ".join(parts)))
        self._depth_status.setStyleSheet("color: {}; font-weight: bold;".format(color))

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
            elapsed = 0.0
            if self._record_start_wall is not None:
                elapsed = time.monotonic() - self._record_start_wall
                self._elapsed_label.setText("Elapsed: {:.1f}s".format(elapsed))
            self._recording_banner.setText(
                "● RECORDING NOW  —  TAKE {}  —  {:.1f} s".format(take_n, elapsed)
            )
            if self._recording_banner.property("recording") is not True:
                self._recording_banner.setProperty("recording", True)
                self._recording_banner.setStyleSheet(
                    _RECORDING_BANNER_ACTIVE_STYLE
                )
                self._recording_panel.setStyleSheet(
                    _RECORDING_PANEL_ACTIVE_STYLE
                )
        else:
            self._stop_button.setEnabled(False)
            self._elapsed_label.setText("")
            self._status_label.setText("PREVIEW")
            self._status_label.setStyleSheet("font-weight: bold;")
            self._recording_banner.setText("○ NOT RECORDING — PREVIEW ONLY")
            if self._recording_banner.property("recording") is not False:
                self._recording_banner.setProperty("recording", False)
                self._recording_banner.setStyleSheet(
                    _RECORDING_BANNER_IDLE_STYLE
                )
                self._recording_panel.setStyleSheet(
                    _RECORDING_PANEL_IDLE_STYLE
                )
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

        # Delete-last-take is enabled iff the node currently has a deletable
        # take (the take most recently STOPPED in this process -- see
        # gello_gui_node.deletable_take_dir()'s docstring). The node's
        # contract already returns None while recording, but `recording` is
        # repeated here explicitly anyway: every other button in this bar is
        # gated straight off it, and a destructive button should not depend
        # on a single upstream flag staying honest.
        self._delete_last_take_button.setEnabled(
            not recording and self._node.deletable_take_dir() is not None
        )
        can_tag = not recording and self._node.deletable_take_dir() is not None
        self._tag_combo.setEnabled(can_tag)
        self._tag_last_take_button.setEnabled(can_tag)

        # Teleop panel shares this ~5 Hz timer.
        self._refresh_teleop()

    # ------------------------------------------------------------ teleop --
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
        # Belt-and-braces: close any open "Delete last take?" prompt BEFORE a
        # new take can start. The node already refuses delete_last_take() once
        # a new take has begun (deletable_take_dir() stops naming the old
        # path -- see _confirm_delete_last_take's own re-check), so this is
        # not load-bearing for correctness; it is here so a stale confirmation
        # box never sits on screen next to a take it can no longer act on.
        if self._delete_box is not None:
            self._delete_box.close()

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
            counts = meta.get("message_counts") or {}
            # Provenance first: depth on/off is a property of the TAKE, and a
            # take recorded without depth must say so rather than merely omit
            # the depth line (an omission reads as "no depth frames arrived").
            if meta.get("record_depth"):
                msg += " | depth ON (cam1 {} / cam2 {} frames)".format(
                    counts.get("cam1_depth_frames", 0),
                    counts.get("cam2_depth_frames", 0))
            else:
                msg += " | depth OFF (RGB only)"
            msg += _stop_health_suffix(meta)
        except (AttributeError, TypeError, ValueError):
            msg = "Take saved."
        self.statusBar().showMessage(msg, 8000)

    def _on_tag_last_take_clicked(self):
        """Append the selected tag to the latest completed take directory."""
        if self._node.deletable_take_dir() is None:
            return
        tag = self._tag_combo.currentText().strip()
        if not tag:
            self.statusBar().showMessage("Enter or select a tag first.", 5000)
            return
        try:
            result = self._node.tag_last_take(tag)
        except RuntimeError as exc:
            self.statusBar().showMessage("Tag failed: {}".format(exc), 8000)
            return
        self._tag_combo.setCurrentIndex(-1)
        self._tag_combo.setEditText("")
        basename = os.path.basename(os.path.normpath(result["path"]))
        if result.get("changed"):
            message = "Tagged last take -> {}".format(basename)
        else:
            message = "Last take already has tag '{}'".format(result["tag"])
        self.statusBar().showMessage(message, 8000)
        self._refresh_controls()

    # ---------------------------------------------------- delete last take --
    def _on_delete_last_take_clicked(self):
        """Open (or re-raise) the non-modal "Delete last take?" prompt.

        NON-MODAL is the whole point: this codebase never uses a blocking
        QMessageBox.exec_() anywhere the robot can move, because a modal
        dialog freezes the Qt event loop, and Pause Teleop is exactly the
        button an operator may need to hit while this prompt is sitting open.
        So the box is built with Qt.NonModal and shown with show(), and every
        exit path -- Yes, No, Escape, or a programmatic box.close() from
        Start Recording / closeEvent -- is routed through the `finished`
        signal in _on_delete_box_finished rather than a return value, because
        show() (unlike exec_()) returns immediately with nothing to check.
        """
        if self._delete_box is not None:
            # Already open: bring it to the front instead of stacking a
            # second prompt (and a second, possibly stale, captured path).
            self._delete_box.raise_()
            self._delete_box.activateWindow()
            return

        path = self._node.deletable_take_dir()
        if not path:
            # Race with the ~5 Hz refresh that disables this button, or a
            # click that landed the same tick the deletable take rolled
            # over. Nothing to confirm.
            return

        basename = os.path.basename(os.path.normpath(path))
        n_files, total_bytes = _scan_take_dir(path)
        mb = total_bytes / (1024.0 * 1024.0)
        text = (
            "Delete {}? This permanently removes the folder ({} files, "
            "{:.1f} MB). Only the take most recently stopped can be "
            "deleted."
        ).format(basename, n_files, mb)

        box = QMessageBox(self)
        box.setWindowModality(Qt.NonModal)
        # Free the C++ widget once it closes; the `finished` handler has
        # already run by then (it fires inside done(), before close()).
        # Without this every prompt would leave a hidden child until exit.
        box.setAttribute(Qt.WA_DeleteOnClose, True)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Delete last take")
        box.setText(text)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        # No is the default AND the escape button: Enter or Esc must never be
        # able to delete a take, only an explicit click on Yes can.
        box.setDefaultButton(QMessageBox.No)
        box.setEscapeButton(QMessageBox.No)
        # `path` is captured here, at prompt-open time, and carried through
        # to the finished handler -- NOT re-read from the node there -- so the
        # re-check in _confirm_delete_last_take is comparing "what this
        # specific prompt was about" against "what the node considers
        # deletable right now", not comparing the live value against itself.
        box.finished.connect(
            lambda _result, box=box, path=path: self._on_delete_box_finished(box, path)
        )
        self._delete_box = box
        box.show()

    def _on_delete_box_finished(self, box, path):
        """Route every exit from the confirm box through one place.

        `finished` fires for a Yes click, a No click, Escape, AND a
        programmatic box.close() (Start Recording / closeEvent) -- the latter
        reports QMessageBox.Rejected with no button clicked, which is why
        this checks the clicked button explicitly rather than trusting the
        dialog's result code: only an explicit Yes is a confirmation, every
        other exit (including "the box got closed out from under the
        operator") is treated as No.
        """
        self._delete_box = None
        clicked = box.clickedButton()
        if clicked is None or box.standardButton(clicked) != QMessageBox.Yes:
            return
        self._confirm_delete_last_take(path)

    def _confirm_delete_last_take(self, path):
        """Yes was clicked: re-validate against the LIVE node state, then delete.

        WHY RE-CHECK. The prompt can sit open for a while (it is non-modal
        precisely so the operator can keep working while it is up), and in
        that window a new take could start -- which is exactly the take
        deletable_take_dir() will no longer name. Comparing the path this
        prompt was opened for against the CURRENT deletable path (not just
        trusting that Yes was clicked) is what stops a stale confirmation
        from deleting the wrong take, or from clashing with a node that has
        already moved on and would refuse the call anyway.
        """
        if self._node.deletable_take_dir() != path:
            self.statusBar().showMessage(
                "Delete skipped -- the deletable take changed while the "
                "confirmation was open.",
                8000,
            )
            self._refresh_controls()
            return
        try:
            result = self._node.delete_last_take()
        except RuntimeError as exc:
            self.statusBar().showMessage("Delete failed: {}".format(exc), 8000)
            return
        basename = os.path.basename(os.path.normpath(result.get("path", path)))
        n_files = result.get("n_files", 0)
        try:
            mb = float(result.get("bytes", 0)) / (1024.0 * 1024.0)
        except (TypeError, ValueError):
            mb = 0.0
        # take_index_after is the COUNTER (what "Take: N" shows = takes on
        # record); start_recording() increments before naming, so the next
        # folder is counter + 1 -- say that number, not the counter, or the
        # operator reads "Next take: 3" and then sees take_04_... appear.
        counter = result.get("take_index_after")
        try:
            next_take = "{:02d}".format(int(counter) + 1)
        except (TypeError, ValueError):
            next_take = "?"
        self.statusBar().showMessage(
            "Deleted {} ({} files, {:.1f} MB). Take count now {}; next "
            "recording will be take_{}.".format(
                basename, n_files, mb, counter, next_take
            ),
            8000,
        )
        # take_index() may have decreased (the number is reused) -- this is
        # what makes the "Take: N" label on the bar reflect it immediately
        # rather than waiting for the next unrelated ~5 Hz tick.
        self._refresh_controls()

    # -------------------------------------------------------------- close ---
    def _shutdown_cameras(self):
        if self._cameras_killed:
            return
        self._cameras_killed = True
        _kill_process_group(self._cam1_proc)
        _kill_process_group(self._cam2_proc)

    def closeEvent(self, event):
        # 0. Close a stray "Delete last take?" prompt -- belt-and-braces, same
        # reasoning as _on_start_clicked: the window is going away regardless,
        # but a prompt left dangling after close() would otherwise leak a
        # QMessageBox with no window to reparent it to.
        if self._delete_box is not None:
            self._delete_box.close()

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

def _depth_node_kwargs(cam1_name, cam2_name, enable_depth, align_depth):
    """Node ctor kwargs for depth recording: ``{}`` when depth is off.

    Shared by ``main()`` here and by GUIs built on this module so they all wire
    the node from the same :func:`depth_topics_for` names the cameras will
    actually publish under (aligned or not).
    """
    if not enable_depth:
        return {}
    d1_img, d1_info, d1_ext = depth_topics_for(cam1_name, align_depth)
    d2_img, d2_info, d2_ext = depth_topics_for(cam2_name, align_depth)
    return {
        "cam1_depth_topic": d1_img,
        "cam2_depth_topic": d2_img,
        "cam1_depth_info_topic": d1_info,
        "cam2_depth_info_topic": d2_info,
        "cam1_extrinsics_topic": d1_ext,
        "cam2_extrinsics_topic": d2_ext,
        "depth_aligned_to_color": bool(align_depth),
    }


def main(args=None):
    # Config from env vars, matching run_recorder.sh's names/defaults.
    cam1_serial = os.environ.get("CAM1_SERIAL", DEFAULT_CAM1_SERIAL)
    cam2_serial = os.environ.get("CAM2_SERIAL", DEFAULT_CAM2_SERIAL)
    cam1_name = os.environ.get("CAM1_NAME", DEFAULT_CAM1_NAME)
    cam2_name = os.environ.get("CAM2_NAME", DEFAULT_CAM2_NAME)
    color_profile = os.environ.get("COLOR_PROFILE", DEFAULT_COLOR_PROFILE)

    # --- Resolve serials against what is actually on the USB bus ---------- #
    # See _resolve_camera_serials() above for the full policy; this mirrors
    # resolve_serials() in ros2_ur_ws/_resolve_camera_serials.sh. A hard
    # mismatch (< 2 devices) aborts BEFORE any camera subprocess is spawned --
    # launching a realsense2_camera node against a serial we already know is
    # wrong just produces a silently-broken GUI instead of a clear error.
    cam_warnings = []
    try:
        cam1_serial, cam2_serial, cam_warnings = _resolve_camera_serials(
            cam1_serial, cam2_serial
        )
    except ImportError:
        cam_warnings = [
            "pyrealsense2 not importable; using configured serials as-is"
        ]
    except RuntimeError as exc:
        print("### ERROR resolving camera serials: {}".format(exc), file=sys.stderr)
        sys.exit(1)

    for w in cam_warnings:
        print("### WARN {}".format(w), file=sys.stderr)

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
            default_repo_root(),
            "ros2_ur_ws", "gello_logs",
        ),
    )

    # The two color topics the realsense nodes publish under their namespace.
    cam1_topic = "/{0}/{0}/color/image_raw/compressed".format(cam1_name)
    cam2_topic = "/{0}/{0}/color/image_raw/compressed".format(cam2_name)
    # Depth: decided ONCE here from the env and used for both the camera
    # launch argv and the node's subscriptions, so they cannot disagree.
    enable_depth = _depth_enabled_from_env()
    align_depth = _align_depth_from_env()
    # Say the bill OUT LOUD, before the take. Depth is opt-in because it is
    # expensive, and 'expensive' has to be visible at the moment it is chosen.
    if enable_depth:
        print("[gello_recorder] {}".format(DEPTH_ON_BANNER), flush=True)
    else:
        print("[gello_recorder] depth OFF (RGB only) -- "
              "ENABLE_DEPTH=1 to record depth.h5", flush=True)
    depth_kwargs = _depth_node_kwargs(cam1_name, cam2_name, enable_depth, align_depth)

    # --- 1. Launch the two RealSense camera nodes as subprocesses --------- #
    cam1_proc = _launch_realsense(cam1_name, cam1_serial, color_profile,
                                  enable_depth, align_depth)
    cam2_proc = _launch_realsense(cam2_name, cam2_serial, color_profile,
                                  enable_depth, align_depth)

    # --- 2. Bring up ROS + the node --------------------------------------- #
    rclpy.init(args=args)

    # The node class stays a lazy import (see the module-level note next to
    # the depth_topics_for re-export for the history of this pattern).
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
        **depth_kwargs,
    )

    # --- 3. Spin the node on a background daemon thread ------------------- #
    # Plain SingleThreadedExecutor via rclpy.spin -- no MultiThreadedExecutor.
    spin_thread = threading.Thread(
        target=_spin_node, args=(node,), daemon=True
    )
    spin_thread.start()

    # --- 4. Qt event loop on the main thread ------------------------------ #
    app = QApplication(sys.argv if args is None else args)
    window = MainWindow(node, cam1_proc, cam2_proc, cam_warnings=cam_warnings)
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
