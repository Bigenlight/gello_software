#!/usr/bin/env python3
"""PyQt5 recorder GUI for real-hardware diffusion-policy inference runs.

Operator flow, one "take" per autonomous policy episode:

  1. Cameras are ALREADY running (launch_cameras.sh, started before the
     diffusion server and before this GUI). This window only subscribes to their
     topics -- it never launches camera subprocesses (a second RealSense driver
     on the same USB device would conflict with the running one).
  2. Operator clicks START EXECUTION -> non-blocking std_srvs/Trigger call to
     /policy_leader_node/start_execution. On response.success == True the window
     AUTOMATICALLY starts a gello_recorder RecordingSession (same
     start_recording() the teleop recorder GUI's Start button makes) -- there is
     no manual "Start Recording" button in this window's normal flow.
  3. HOLD fires /policy_leader_node/hold at any time. HOLD (or a robot fault)
     deliberately does NOT stop the recording -- stopping is ALWAYS an explicit
     operator action, so post-fault footage is preserved.
  4. Stop Recording finalizes the take, then SUCCESS / FAIL light up and
     START EXECUTION stays disabled until one of them is clicked -- an unlabeled
     take is structurally impossible in the normal flow. The chosen label is
     written atomically as label.json into the take directory.

Window structure (dual camera preview + state panel + spin-thread plumbing) is
inherited from gello_recorder_gui.MainWindow; only the control bar, its refresh
logic, and the stop/label actions are overridden. The Trigger-button
ready / in-flight / timeout / result-display state machine is a Qt port of
camera_viewer.py's _TriggerButton (rclpy Futures behave identically whether
polled from a cv2 waitKey loop or a QTimer; completion is serviced by the ROS
spin thread either way).

console_script entry point: ``policy_run_gui = gello_recorder.policy_run_gui:main``
"""

import os
import sys
import threading
import time

import rclpy

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from gello_recorder.gello_recorder_gui import MainWindow, _spin_node

# --- Trigger state-machine tuning: same values as camera_viewer.py ---------- #
_SERVICE_CHECK_INTERVAL_S = 0.5   # re-poll service_is_ready() no more often than
_CALL_TIMEOUT_S = 5.0             # an in-flight call is treated as failed after
_RESULT_DISPLAY_S = 4.0           # how long a success/fail result stays shown


class _TriggerServiceButton:
    """Qt port of camera_viewer.py's _TriggerButton state machine.

    Owns one QPushButton + one result QLabel and drives them through
    waiting-for-service / ready / in-flight / result-display states. All methods
    run on the Qt/main thread (update() is called from a QTimer); the only
    cross-thread objects touched are the rclpy Future's .done()/.result(),
    which is exactly the pattern camera_viewer.py already uses.
    """

    def __init__(self, label, ready_fn, call_fn, button, result_label,
                 on_success=None):
        self.label = label
        self._ready_fn = ready_fn      # thread-safe node method (bool)
        self._call_fn = call_fn        # thread-safe node method -> rclpy Future
        self._button = button
        self._result_label = result_label
        self._on_success = on_success  # called (msg) when response.success

        self.ready = False             # mirror of service_is_ready()
        self._last_ready_check = 0.0   # throttle for the ready poll
        self.future = None             # in-flight call_async future, or None
        self.inflight_start = None     # monotonic time the call was issued
        self.result_text = None        # last response/error text, or None
        self.result_ok = False         # last response.success
        self.result_expiry = 0.0       # monotonic time the result stops showing
        self._gate = False             # external enable gate (window policy)

        button.clicked.connect(self._on_clicked)

    def in_flight(self):
        return self.future is not None

    # ---- click (Qt signal, main thread) --------------------------------- #
    def _on_clicked(self):
        now = time.monotonic()
        if self.in_flight():
            return  # a call is already pending -> ignore (no double-fire)
        if not self.ready or not self._gate:
            return  # button should be disabled anyway; belt and suspenders
        self.result_text = None        # clear any stale result
        self.inflight_start = now
        try:
            self.future = self._call_fn()
        except Exception as exc:  # noqa: BLE001 -- e.g. shutdown race
            self.future = None
            self.inflight_start = None
            self.result_ok = False
            self.result_text = "call error: {}".format(exc)
            self.result_expiry = now + _RESULT_DISPLAY_S

    # ---- periodic update (QTimer, main thread) --------------------------- #
    def update(self, now, gate, gate_hint=None):
        """Refresh availability (throttled), advance an in-flight call, and
        repaint the button/result widgets. ``gate`` is the window-level enable
        policy (e.g. "not while recording"); ``gate_hint`` is a short suffix
        explaining WHY the button is gated off, shown on the disabled button."""
        self._gate = gate
        if (now - self._last_ready_check) >= _SERVICE_CHECK_INTERVAL_S:
            self._last_ready_check = now
            self.ready = bool(self._ready_fn())
        self._poll(now)
        self._update_widgets(now, gate, gate_hint)

    def _poll(self, now):
        """Non-blocking: advance an in-flight call to done / timed-out."""
        if self.future is None:
            return
        if self.future.done():
            try:
                resp = self.future.result()
                ok = bool(resp.success)
                msg = (resp.message or "").strip()
                text = msg or ("OK" if ok else "failed")
            except Exception as exc:  # noqa: BLE001 -- surface any call failure
                ok, text = False, "call error: {}".format(exc)
            self.future = None
            self.inflight_start = None
            self.result_ok, self.result_text = ok, text
            self.result_expiry = now + _RESULT_DISPLAY_S
            if ok and self._on_success is not None:
                self._on_success(text)
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
            self.result_text = "timed out (>{:.0f}s)".format(_CALL_TIMEOUT_S)
            self.result_expiry = now + _RESULT_DISPLAY_S

    def _update_widgets(self, now, gate, gate_hint):
        if self.in_flight():
            self._button.setEnabled(False)
            self._button.setText("{} ...".format(self.label))
        elif not self.ready:
            self._button.setEnabled(False)
            self._button.setText("{} (waiting for launch...)".format(self.label))
        elif not gate:
            self._button.setEnabled(False)
            if gate_hint:
                self._button.setText("{} ({})".format(self.label, gate_hint))
            else:
                self._button.setText(self.label)
        else:
            self._button.setEnabled(True)
            self._button.setText(self.label)

        if self.result_text is not None and now < self.result_expiry:
            if self.result_ok:
                self._result_label.setText("OK: {}".format(self.result_text)[:90])
                self._result_label.setStyleSheet(
                    "color: #22aa22; font-weight: bold;")
            else:
                self._result_label.setText("ERR: {}".format(self.result_text)[:90])
                self._result_label.setStyleSheet(
                    "color: #cc3333; font-weight: bold;")
        else:
            self._result_label.setText("")


class PolicyRunWindow(MainWindow):
    """Diffusion-policy run window: preview/state panel inherited, control bar
    replaced with START EXECUTION / HOLD / Stop Recording / SUCCESS / FAIL."""

    def __init__(self, node):
        # No camera subprocesses in this workflow -> None handles. The base
        # class's _kill_process_group(None) / closeEvent camera teardown are
        # no-ops on None, so the inherited shutdown path stays correct.
        super().__init__(node, None, None)
        self.setWindowTitle("UR7e Diffusion Policy Run Recorder")

    # ------------------------------------------------------------------ UI --
    def _build_control_bar(self):
        # Called from the base __init__'s _build_ui(); window-flow state must be
        # initialized here (attributes cannot be set before QMainWindow init).
        self._pending_label_dir = None   # take dir awaiting SUCCESS/FAIL
        self._current_take_dir = None    # take dir of the active recording

        outer = QVBoxLayout()

        # --- Row 1: policy trigger buttons + result lines ----------------- #
        self._exec_button = QPushButton("START EXECUTION")
        self._exec_button.setMinimumHeight(44)
        self._exec_button.setStyleSheet("font-weight: bold;")
        self._exec_result = QLabel("")
        self._exec_result.setAlignment(Qt.AlignCenter)

        self._hold_button = QPushButton("HOLD")
        self._hold_button.setMinimumHeight(44)
        self._hold_button.setStyleSheet("font-weight: bold;")
        self._hold_result = QLabel("")
        self._hold_result.setAlignment(Qt.AlignCenter)

        self._exec_machine = _TriggerServiceButton(
            "START EXECUTION",
            self._node.start_execution_ready,
            self._node.call_start_execution,
            self._exec_button,
            self._exec_result,
            on_success=self._on_start_execution_success,
        )
        self._hold_machine = _TriggerServiceButton(
            "HOLD",
            self._node.hold_ready,
            self._node.call_hold,
            self._hold_button,
            self._hold_result,
        )

        trigger_row = QHBoxLayout()
        exec_col = QVBoxLayout()
        exec_col.addWidget(self._exec_button)
        exec_col.addWidget(self._exec_result)
        hold_col = QVBoxLayout()
        hold_col.addWidget(self._hold_button)
        hold_col.addWidget(self._hold_result)
        trigger_row.addLayout(exec_col, stretch=1)
        trigger_row.addLayout(hold_col, stretch=1)
        outer.addLayout(trigger_row)

        # --- Row 2: recording / labeling controls + status ---------------- #
        self._stop_button = QPushButton("Stop Recording")
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self._stop_button.setEnabled(False)

        self._success_button = QPushButton("SUCCESS")
        self._success_button.clicked.connect(
            lambda: self._on_label_clicked(True))
        self._success_button.setEnabled(False)
        self._fail_button = QPushButton("FAIL")
        self._fail_button.clicked.connect(
            lambda: self._on_label_clicked(False))
        self._fail_button.setEnabled(False)

        self._take_label = QLabel("Take: 0")
        self._elapsed_label = QLabel("")
        self._status_label = QLabel("IDLE")
        self._status_label.setStyleSheet("font-weight: bold;")

        bar = QHBoxLayout()
        bar.addWidget(self._stop_button)
        bar.addSpacing(20)
        bar.addWidget(self._success_button)
        bar.addWidget(self._fail_button)
        bar.addSpacing(20)
        bar.addWidget(self._take_label)
        bar.addSpacing(20)
        bar.addWidget(self._elapsed_label)
        bar.addStretch(1)
        bar.addWidget(self._status_label)
        outer.addLayout(bar)

        return outer

    # -------------------------------------------------------------- timers --
    def _refresh_controls(self):
        """~5 Hz (base class's _control_timer): drive both trigger state
        machines and the recording/labeling button policy."""
        now = time.monotonic()
        recording = self._node.is_recording()
        cams_ready = self._node.cameras_ready()
        take_n = self._node.take_index()
        awaiting_label = self._pending_label_dir is not None

        # START EXECUTION gate: never while a take is recording, never while a
        # finished take is still unlabeled (structurally prevents an unlabeled
        # take), and never before cameras are ready (start_recording() would
        # refuse and the run would go unrecorded).
        if recording:
            exec_gate, exec_hint = False, "recording"
        elif awaiting_label:
            exec_gate, exec_hint = False, "label last take first"
        elif not cams_ready:
            exec_gate, exec_hint = False, "cameras warming up"
        else:
            exec_gate, exec_hint = True, None
        self._exec_machine.update(now, exec_gate, exec_hint)

        # HOLD is a safety control: available whenever the service exists,
        # including mid-recording. It must NOT auto-stop the recording.
        self._hold_machine.update(now, True)

        self._stop_button.setEnabled(recording)
        self._success_button.setEnabled(awaiting_label)
        self._fail_button.setEnabled(awaiting_label)

        if recording:
            self._status_label.setText("● RECORDING take {}".format(take_n))
            self._status_label.setStyleSheet(
                "color: #cc3333; font-weight: bold;")
            if self._record_start_wall is not None:
                elapsed = now - self._record_start_wall
                self._elapsed_label.setText("Elapsed: {:.1f}s".format(elapsed))
        elif awaiting_label:
            self._status_label.setText(
                "LABEL take {}: SUCCESS or FAIL?".format(take_n))
            self._status_label.setStyleSheet(
                "color: #dd8800; font-weight: bold;")
            self._elapsed_label.setText("")
        else:
            self._status_label.setText("IDLE")
            self._status_label.setStyleSheet("font-weight: bold;")
            self._elapsed_label.setText("")

        if recording:
            self._take_label.setText("Take: {} (recording)".format(take_n))
        else:
            self._take_label.setText("Take: {}".format(take_n))

    # ------------------------------------------------------------- actions --
    def _on_start_execution_success(self, msg):
        """response.success == True from ~/start_execution -> auto-start the
        recording (the only way a recording starts in this window)."""
        if self._node.is_recording() or self._pending_label_dir is not None:
            return  # stale/racy response; never double-start or skip labeling
        if not self._node.cameras_ready():
            self.statusBar().showMessage(
                "start_execution OK but cameras not warmed up -- "
                "NOT recording this run!", 10000)
            return
        try:
            take_dir = self._node.start_recording()
        except RuntimeError as exc:
            self.statusBar().showMessage(
                "start_execution OK but recording failed to start: {}".format(exc),
                10000)
            return
        self._current_take_dir = take_dir
        self._record_start_wall = time.monotonic()
        self.statusBar().showMessage(
            "Policy executing + recording -> {}".format(take_dir), 5000)
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
        # Enter the mandatory labeling state; START EXECUTION stays gated off
        # until SUCCESS or FAIL is clicked.
        take_dir = None
        if isinstance(meta, dict):
            take_dir = meta.get("session_dir")
        self._pending_label_dir = take_dir or self._current_take_dir
        self._current_take_dir = None
        self._show_stop_confirmation(meta)
        self._refresh_controls()

    def _on_label_clicked(self, success):
        take_dir = self._pending_label_dir
        if take_dir is None:
            return
        try:
            label_path = self._node.write_label(take_dir, success)
        except Exception as exc:  # noqa: BLE001 -- keep labeling state so the
            # operator can retry (or fix the disk) instead of losing the verdict
            self.statusBar().showMessage(
                "Label write FAILED (still unlabeled): {}".format(exc), 10000)
            return
        self._pending_label_dir = None
        self.statusBar().showMessage(
            "Labeled {} -> {}".format(
                "SUCCESS" if success else "FAIL", label_path), 8000)
        self._refresh_controls()

    def _on_start_clicked(self):  # pragma: no cover - defensive override
        """Base-class manual record start is not part of this window's flow."""
        return


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main(args=None):
    # Same env-var config surface as gello_recorder_gui.main(), MINUS everything
    # camera-subprocess related (serials/profile): cameras are already running.
    cam1_name = os.environ.get("CAM1_NAME", "cam1")
    cam2_name = os.environ.get("CAM2_NAME", "cam2")
    camera_fps = float(os.environ.get("CAMERA_FPS", "30.0"))
    # Cameras were started (and auto-exposure settled) long before this GUI in
    # the diffusion workflow, so the warmup gate only needs to prove frames are
    # actually flowing to THIS node -- default 1.0s instead of the teleop
    # recorder's 4.0s. Override with CAMERA_WARMUP_S if needed.
    camera_warmup_s = float(os.environ.get("CAMERA_WARMUP_S", "1.0"))
    output_root = os.environ.get(
        "POLICY_RUN_OUTPUT_ROOT",
        os.path.join(
            os.environ.get(
                "GELLO_REPO_ROOT",
                os.path.expanduser("~/gello_software"),
            ),
            "ros2_ur_ws", "gello_logs", "policy_runs",
        ),
    )

    cam1_topic = "/{0}/{0}/color/image_raw/compressed".format(cam1_name)
    cam2_topic = "/{0}/{0}/color/image_raw/compressed".format(cam2_name)

    rclpy.init(args=args)

    # Imported here to mirror gello_recorder_gui.py's lazy-import pattern.
    from gello_recorder.policy_run_gui_node import PolicyRunGuiNode

    # Same cv2-bundled-Qt workaround as gello_recorder_gui.main(): clear the
    # plugin path `import cv2` injected before any QApplication exists.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    node = PolicyRunGuiNode(
        cam1_topic=cam1_topic,
        cam2_topic=cam2_topic,
        camera_fps=camera_fps,
        camera_warmup_s=camera_warmup_s,
        output_root=output_root,
    )

    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv if args is None else args)
    window = PolicyRunWindow(node)
    window.resize(1280, 800)
    window.show()

    exit_code = app.exec_()

    # Belt-and-suspenders cleanup (closeEvent normally already did all this;
    # everything below is idempotent / guarded).
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    spin_thread.join(timeout=2.0)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
