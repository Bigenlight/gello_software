#!/usr/bin/env python3
"""Standalone live reward-classifier monitor.

This intentionally reuses the proven policy/recorder GUI shell: dual camera
preview on the left and live UR7e/GELLO state on the right.  All action,
recording, and policy controls are replaced by a read-only classifier panel.
It can therefore run with only the camera/robot ROS graph; no policy server or
``policy_leader_node`` is required.
"""

import os
import signal
import sys
import threading

import rclpy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from gello_recorder.gello_recorder_gui import MainWindow, _spin_node


class ClassifierViewWindow(MainWindow):
    def __init__(self, node):
        # Cameras are already opened by realsense2_camera.  Subscriptions only.
        super().__init__(node, None, None)
        self.setWindowTitle("UR7e Cube-in-Cup Reward Classifier Monitor")

    def _build_teleop_bar(self):
        """Replace bridge controls with an unmistakable read-only banner."""
        row = QHBoxLayout()
        self._monitor_banner = QLabel(
            "MONITOR ONLY — no policy execution and no robot commands from this UI"
        )
        self._monitor_banner.setAlignment(Qt.AlignCenter)
        self._monitor_banner.setStyleSheet(
            "background-color: #1565c0; color: white; font-weight: bold; "
            "font-size: 12pt; padding: 9px;"
        )
        row.addWidget(self._monitor_banner)
        return row

    def _build_control_bar(self):
        outer = QVBoxLayout()
        box = QGroupBox("Cube-in-cup reward classifier")
        row = QHBoxLayout(box)

        self._classifier_verdict = QLabel("OFFLINE")
        self._classifier_verdict.setAlignment(Qt.AlignCenter)
        self._classifier_verdict.setMinimumWidth(200)

        self._classifier_probability = QLabel("p(success): —")
        self._classifier_probability.setAlignment(Qt.AlignCenter)
        self._classifier_probability.setMinimumWidth(200)
        self._classifier_probability.setStyleSheet(
            "font-weight: bold; font-size: 15pt;"
        )

        self._classifier_diagnostics = QLabel(
            "Waiting for reward_classifier node")
        self._classifier_diagnostics.setWordWrap(True)
        self._classifier_diagnostics.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self._transport_badge = QLabel("REMOTE OFFLINE")
        self._transport_badge.setAlignment(Qt.AlignCenter)
        self._transport_badge.setMinimumWidth(150)
        row.addWidget(self._transport_badge)
        row.addWidget(self._classifier_verdict)
        row.addWidget(self._classifier_probability)
        row.addWidget(self._classifier_diagnostics, stretch=1)
        outer.addWidget(box)

        hint = QLabel(
            "Green SUCCESS means p(success) > threshold. "
            "This display does not terminate an episode or command the robot."
        )
        hint.setStyleSheet("color: #666666;")
        outer.addWidget(hint)
        return outer

    def _refresh_controls(self):
        status = self._node.get_classifier_status()
        age = status.get("status_age_s")
        if age is None or age > 1.0:
            self._set_transport("REMOTE OFFLINE", "#666666")
            self._set_verdict("OFFLINE / STALE", "#666666")
            self._classifier_probability.setText("p(success): —")
            self._classifier_diagnostics.setText(
                status.get("message", "classifier status stale")
            )
            return
        if status.get("remote"):
            self._set_transport(
                "REMOTE CONNECTED" if status.get("ready") else "REMOTE STALE",
                "#2277aa" if status.get("ready") else "#dd8800")
        else:
            self._set_transport("LOCAL", "#5555aa")

        probability = status.get("probability")
        self._classifier_probability.setText(
            "p(success): —"
            if probability is None
            else "p(success): {:.3f}".format(float(probability))
        )

        if not status.get("ready", False):
            self._set_verdict("WAIT / INVALID", "#dd8800")
        elif status.get("success", False):
            self._set_verdict("SUCCESS", "#22aa22")
        else:
            self._set_verdict("FAILURE", "#cc3333")

        details = [status.get("message", "")]
        if status.get("threshold") is not None:
            details.append("threshold={:.2f}".format(float(status["threshold"])))
        if status.get("cam_skew_ms") is not None:
            details.append(
                "camera skew={:.1f} ms".format(float(status["cam_skew_ms"]))
            )
        if status.get("inference_ms") is not None:
            details.append(
                "inference={:.1f} ms".format(float(status["inference_ms"]))
            )
        if status.get("roundtrip_ms") is not None:
            details.append(
                "roundtrip={:.1f} ms".format(float(status["roundtrip_ms"]))
            )
        if status.get("capture_age_ms") is not None:
            details.append(
                "frame age={:.1f} ms".format(float(status["capture_age_ms"]))
            )
        details.append("status age={:.2f} s".format(float(age)))
        self._classifier_diagnostics.setText(
            " | ".join(value for value in details if value)
        )

    def _set_verdict(self, text, color):
        self._classifier_verdict.setText(text)
        self._classifier_verdict.setStyleSheet(
            "background-color: {}; color: white; font-weight: bold; "
            "font-size: 18pt; padding: 12px;".format(color)
        )

    def _set_transport(self, text, color):
        self._transport_badge.setText(text)
        self._transport_badge.setStyleSheet(
            "background-color: {}; color: white; font-weight: bold; "
            "padding: 8px;".format(color)
        )


def main(args=None):
    cam1_name = os.environ.get("CAM1_NAME", "cam1")
    cam2_name = os.environ.get("CAM2_NAME", "cam2")
    cam1_topic = "/{0}/{0}/color/image_raw/compressed".format(cam1_name)
    cam2_topic = "/{0}/{0}/color/image_raw/compressed".format(cam2_name)

    rclpy.init(args=args)
    from gello_recorder.classifier_view_gui_node import ClassifierViewGuiNode

    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    node = ClassifierViewGuiNode(
        cam1_topic=cam1_topic,
        cam2_topic=cam2_topic,
    )
    spin_thread = threading.Thread(
        target=_spin_node, args=(node,), daemon=True
    )
    spin_thread.start()

    app = QApplication(sys.argv if args is None else args)
    window = ClassifierViewWindow(node)
    window.resize(1280, 800)
    window.show()

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    pump = QTimer()
    pump.start(200)
    pump.timeout.connect(lambda: None)
    exit_code = app.exec_()

    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass
    spin_thread.join(timeout=2.0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
