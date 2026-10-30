#!/usr/bin/env python3
"""Integrated EEF teleop and persistent GELLO recorder operator GUI.

The recorder and EEF control client intentionally live in separate ROS nodes
inside one process.  The recorder owns the cameras and recording sessions for
the full GUI lifetime; restarting the external GELLO/UR bringup therefore only
makes the EEF pane go offline temporarily.  ROS graph discovery reconnects the
EEF subscriptions and service clients when the bringup returns, without
restarting cameras or losing the recorder window.

Global shortcuts (active anywhere in this window):

* ``R`` toggles recording start/stop.
* ``SPACE`` toggles EEF engage/disengage, preserving its two-press engage gate.
* ``G`` toggles gripper pause/resume, preserving its two-press resume gate.
* ``H`` requests the start pose, preserving its two-press confirmation gate.
"""

import os
import signal
import sys
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gello_recorder import gello_recorder_gui as recorder_gui
from gello_recorder.gello_gui_node import GelloRecorderGuiNode
from gello_recorder.paths import default_repo_root
from gello_recorder.spin_health import DEPTH_ON_BANNER

from ur_gello_bringup.gello_eef_gui_node import (
    EefGuiNode,
    MainWindow as EefWindow,
)


class RecorderPane(recorder_gui.MainWindow):
    """Recorder window embedded without its legacy joint-teleop control bar."""

    def _build_teleop_bar(self):
        from PyQt5.QtWidgets import QHBoxLayout
        return QHBoxLayout()

    def _make_video_pane(self, name):
        pane = super()._make_video_pane(name)
        pane.setMinimumSize(200, 150)
        return pane

    def _refresh_teleop(self):
        return

    def _refresh_controls(self):
        super()._refresh_controls()
        if self._node.is_recording():
            self._start_button.setText("Recording...  [R stops]")
        elif self._start_button.isEnabled():
            self._start_button.setText("Start Recording  [R]")
        self._stop_button.setText("Stop Recording  [R]")


def _compact_css(style):
    """Shrink the EEF standalone-window chrome without changing its colours."""
    replacements = (
        ("font-size: 20pt", "font-size: 13pt"),
        ("font-size: 16pt", "font-size: 11pt"),
        ("font-size: 13pt", "font-size: 9pt"),
        ("min-height: 56px", "min-height: 30px"),
        ("min-height: 38px", "min-height: 24px"),
        ("padding: 18px", "padding: 6px"),
        ("padding: 15px", "padding: 4px"),
        ("padding: 10px 16px", "padding: 3px 7px"),
        ("padding: 9px 14px", "padding: 3px 6px"),
        ("padding-top: 17px", "padding-top: 5px"),
    )
    for old, new in replacements:
        style = style.replace(old, new)
    return style


class CompactEefWindow(EefWindow):
    """EEF controls optimized for embedding beside the camera recorder."""

    def __init__(self, node):
        self._compact_ready = False
        super().__init__(node)
        self._compact_ready = True

        root = self.centralWidget().layout()
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        self._details_button = QPushButton("Show EEF diagnostics")
        self._details_button.setCheckable(True)
        self._details_button.setStyleSheet(
            "font-size: 8pt; padding: 2px 6px; color: #555;"
        )
        self._details_button.toggled.connect(self._toggle_details)
        root.insertWidget(root.indexOf(self._readout_box), self._details_button)
        self._readout_box.hide()
        self._h5.hide()
        self._apply_compact_chrome()

    def _refresh_eef(self):
        super()._refresh_eef()
        if getattr(self, "_compact_ready", False):
            self._apply_compact_chrome()

    def _apply_compact_chrome(self):
        for widget in (
            self._primary,
            self._grip_pause_btn,
            self._grip_resume_btn,
            self._start_pose_btn,
            self._start_pose_stop_btn,
            self._state_label,
            self._grip_discrete_lbl,
        ):
            widget.setStyleSheet(_compact_css(widget.styleSheet()))
        self._h1.setStyleSheet(
            "background-color: #cc3333; color: white; font-weight: bold; "
            "font-size: 9pt; padding: 3px;"
        )
        self._start_pose_info.setStyleSheet(
            "color: #666666; font-size: 8pt;"
        )

    def _toggle_details(self, visible):
        self._readout_box.setVisible(visible)
        self._h5.setVisible(visible)
        self._details_button.setText(
            "Hide EEF diagnostics" if visible else "Show EEF diagnostics"
        )


class IntegratedWindow(QMainWindow):
    """One persistent recorder window with a restart-tolerant EEF side pane."""

    def __init__(self, recorder_node, eef_node, cam1_proc, cam2_proc,
                 cam_warnings, shutdown_callback):
        super().__init__()
        self._recorder_node = recorder_node
        self._eef_node = eef_node
        self._shutdown_callback = shutdown_callback
        self._closed = False

        self.setWindowTitle("GELLO EEF Data Collection — R / SPACE / G / H")
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        hotkeys = QLabel(
            "R: START / STOP RECORDING    |    SPACE: EEF ON / OFF    |    "
            "G: GRIPPER PAUSE / RESUME    |    H H: GO TO START POSE    — "
            "teleop may be restarted while this window and its cameras stay "
            "open"
        )
        hotkeys.setAlignment(Qt.AlignCenter)
        hotkeys.setStyleSheet(
            "background-color: #204060; color: white; font-weight: bold; "
            "font-size: 11pt; padding: 7px;"
        )
        root.addWidget(hotkeys)

        splitter = QSplitter(Qt.Horizontal, self)
        root.addWidget(splitter, stretch=1)

        self._recorder_window = RecorderPane(
            recorder_node, cam1_proc, cam2_proc, cam_warnings
        )
        self._eef_window = CompactEefWindow(eef_node)
        self._recorder_window.setWindowFlags(Qt.Widget)
        self._eef_window.setWindowFlags(Qt.Widget)
        splitter.addWidget(self._recorder_window)

        self._eef_scroll = QScrollArea(self)
        self._eef_scroll.setWidgetResizable(True)
        self._eef_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._eef_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._eef_scroll.setWidget(self._eef_window)
        splitter.addWidget(self._eef_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # Child-window shortcuts are disabled and recreated on this top-level
        # shell so they work regardless of whether focus is in the camera,
        # recorder, EEF readout, or a button.
        self._eef_window._space_shortcut.setEnabled(False)
        self._eef_window._grip_shortcut.setEnabled(False)
        self._eef_window._start_pose_shortcut.setEnabled(False)
        self._record_shortcut = self._shortcut(Qt.Key_R, self._on_record_shortcut)
        self._eef_shortcut = self._shortcut(
            Qt.Key_Space, self._eef_window._on_primary_shortcut
        )
        self._grip_shortcut = self._shortcut(
            Qt.Key_G, self._eef_window._on_grip_toggle_shortcut
        )
        self._start_pose_shortcut = self._shortcut(
            Qt.Key_H, self._eef_window._on_start_pose_shortcut
        )

    def _shortcut(self, key, callback):
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.WindowShortcut)
        shortcut.setAutoRepeat(False)
        shortcut.activated.connect(callback)
        return shortcut

    def _on_record_shortcut(self):
        if self._recorder_node.is_recording():
            self._recorder_window._on_stop_clicked()
            return
        if self._recorder_window._start_button.isEnabled():
            self._recorder_window._on_start_clicked()
            return
        self._recorder_window.statusBar().showMessage(
            "R ignored — cameras are still warming up or unavailable.", 3000
        )

    def closeEvent(self, event):
        if self._closed:
            event.accept()
            return

        # Preserve the EEF GUI's existing action-client safety contract: a
        # running start-pose goal gets STOP first rather than being orphaned.
        if self._eef_window._veto_close_during_start_pose_move(event):
            return

        self._closed = True
        for timer in (
            getattr(self._recorder_window, "_preview_timer", None),
            getattr(self._recorder_window, "_state_timer", None),
            getattr(self._recorder_window, "_control_timer", None),
            getattr(self._eef_window, "_timer", None),
            getattr(self._eef_window, "_vw_timer", None),
        ):
            if timer is not None:
                timer.stop()

        delete_box = getattr(self._recorder_window, "_delete_box", None)
        if delete_box is not None:
            delete_box.close()
        try:
            if self._recorder_node.is_recording():
                self._recorder_node.stop_recording()
        except Exception:  # noqa: BLE001 -- best effort during window teardown
            pass
        self._recorder_window._shutdown_cameras()
        self._shutdown_callback()
        event.accept()


def _camera_config():
    cam1_serial = os.environ.get("CAM1_SERIAL", recorder_gui.DEFAULT_CAM1_SERIAL)
    cam2_serial = os.environ.get("CAM2_SERIAL", recorder_gui.DEFAULT_CAM2_SERIAL)
    cam1_name = os.environ.get("CAM1_NAME", recorder_gui.DEFAULT_CAM1_NAME)
    cam2_name = os.environ.get("CAM2_NAME", recorder_gui.DEFAULT_CAM2_NAME)
    color_profile = os.environ.get(
        "COLOR_PROFILE", recorder_gui.DEFAULT_COLOR_PROFILE
    )

    warnings = []
    try:
        cam1_serial, cam2_serial, warnings = recorder_gui._resolve_camera_serials(
            cam1_serial, cam2_serial
        )
    except ImportError:
        warnings = ["pyrealsense2 not importable; using configured serials as-is"]
    except RuntimeError as exc:
        raise RuntimeError("ERROR resolving camera serials: {}".format(exc)) from exc

    camera_fps = 30.0
    try:
        camera_fps = float(color_profile.split("x")[-1])
    except (ValueError, IndexError):
        pass

    enable_depth = recorder_gui._depth_enabled_from_env()
    align_depth = recorder_gui._align_depth_from_env()
    return {
        "cam1_serial": cam1_serial,
        "cam2_serial": cam2_serial,
        "cam1_name": cam1_name,
        "cam2_name": cam2_name,
        "color_profile": color_profile,
        "camera_fps": camera_fps,
        "camera_warmup_s": float(os.environ.get("CAMERA_WARMUP_S", "4.0")),
        "output_root": os.environ.get(
            "RECORDER_OUTPUT_ROOT",
            os.path.join(default_repo_root(), "ros2_ur_ws", "gello_logs"),
        ),
        "enable_depth": enable_depth,
        "align_depth": align_depth,
        "warnings": warnings,
    }


def _start_node_executor(node):
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    return executor, spin_thread


def main(args=None):
    try:
        config = _camera_config()
    except RuntimeError as exc:
        print("### {}".format(exc), file=sys.stderr)
        sys.exit(1)

    for warning in config["warnings"]:
        print("### WARN {}".format(warning), file=sys.stderr)
    if config["enable_depth"]:
        print("[gello_recorder] {}".format(DEPTH_ON_BANNER), flush=True)
    else:
        print(
            "[gello_recorder] depth OFF (RGB only) -- "
            "ENABLE_DEPTH=1 to record depth.h5",
            flush=True,
        )

    cam1_proc = cam2_proc = None
    executors = []
    spin_threads = []
    recorder_node = eef_node = None
    shutdown_lock = threading.Lock()
    shutdown_done = False

    def shutdown():
        nonlocal shutdown_done
        with shutdown_lock:
            if shutdown_done:
                return
            shutdown_done = True
        for executor in executors:
            try:
                executor.shutdown(timeout_sec=2.0)
            except Exception:  # noqa: BLE001
                pass
        for node in (recorder_node, eef_node):
            if node is not None:
                try:
                    node.destroy_node()
                except Exception:  # noqa: BLE001
                    pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

    try:
        cam1_proc = recorder_gui._launch_realsense(
            config["cam1_name"], config["cam1_serial"], config["color_profile"],
            config["enable_depth"], config["align_depth"]
        )
        cam2_proc = recorder_gui._launch_realsense(
            config["cam2_name"], config["cam2_serial"], config["color_profile"],
            config["enable_depth"], config["align_depth"]
        )

        rclpy.init(args=args)
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

        depth_kwargs = recorder_gui._depth_node_kwargs(
            config["cam1_name"], config["cam2_name"],
            config["enable_depth"], config["align_depth"]
        )
        recorder_node = GelloRecorderGuiNode(
            cam1_topic="/{0}/{0}/color/image_raw/compressed".format(
                config["cam1_name"]
            ),
            cam2_topic="/{0}/{0}/color/image_raw/compressed".format(
                config["cam2_name"]
            ),
            camera_fps=config["camera_fps"],
            camera_warmup_s=config["camera_warmup_s"],
            output_root=config["output_root"],
            node_name="eef_collection_recorder_node",
            **depth_kwargs,
        )
        eef_node = EefGuiNode(node_name="eef_collection_control_node")

        for node in (recorder_node, eef_node):
            executor, spin_thread = _start_node_executor(node)
            executors.append(executor)
            spin_threads.append(spin_thread)

        app = QApplication(sys.argv if args is None else args)
        window = IntegratedWindow(
            recorder_node, eef_node, cam1_proc, cam2_proc,
            config["warnings"], shutdown
        )
        window.resize(1800, 950)
        window.showMaximized()

        signal.signal(signal.SIGINT, lambda *_: window.close())
        exit_code = app.exec_()
        shutdown()
        for spin_thread in spin_threads:
            spin_thread.join(timeout=2.0)
        sys.exit(exit_code)
    finally:
        shutdown()
        recorder_gui._kill_process_group(cam1_proc)
        recorder_gui._kill_process_group(cam2_proc)


if __name__ == "__main__":
    main()
