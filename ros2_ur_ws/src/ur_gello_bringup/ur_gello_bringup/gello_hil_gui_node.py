#!/usr/bin/env python3
"""PyQt5 operator GUI for the UR7e+GELLO HIL RL intervention deadman.

This GUI drives the HIL RL env's ``GelloIntervention`` deadman by publishing a
heartbeat on the ROS topic ``/hil/deadman`` -- it is the GUI ALTERNATIVE to the
terminal-focus spacebar deadman. It runs WITHOUT the teleop bridge and WITHOUT
``control_mode:=eef``: it never calls a single bridge service, never talks to the
robot or GELLO, and owns nothing but one publisher. The RL env (the
``RosTopicDeadman`` DeadmanSource in ``serl_ur_infra/ur_env/envs/wrappers.py``)
subscribes and reads the engage/gain signal off the topic.

It is used ALONGSIDE:
  * ``serl_ur_infra/tests/run_rviz_hil.py --deadman topic``  (T4, the RL loop),
  * ``gello_publisher``                                      (T3, the leader),
so the human engages here (or via spacebar), then moves the PHYSICAL GELLO
leader to inject expert intervention.

FROZEN SHARED CONTRACT (do NOT deviate -- the env agent depends on it):
  topic  /hil/deadman        type std_msgs/Float32MultiArray
  data   [engaged, gain]      engaged in {0.0, 1.0}; gain in [0.10, 1.00]
  rate   20 Hz heartbeat      (published continuously, not only on change, so the
                              subscriber's staleness watchdog -- STALE_S=0.5s --
                              stays satisfied; after the first heartbeat, losing
                              the GUI fail-stops the control loop rather than
                              falling back to policy)
  QoS    default reliable, depth 10

ARCHITECTURE (forked verbatim from gello_eef_gui, the proven gello_recorder_gui
pattern -- NOT invented):
  * :class:`HilGuiNode` (an rclpy Node) owns ALL ROS I/O -- here just the
    ``/hil/deadman`` publisher. It is spun on a background daemon thread with a
    plain ``rclpy.spin``.
  * :class:`MainWindow` (Qt, main thread) holds the operator intent
    (``self._engaged`` + the sensitivity slider), publishes the heartbeat from a
    20 Hz QTimer, and polls thread-safe getters from a ~10 Hz QTimer to drive the
    widgets. Qt widgets are touched ONLY from the Qt/main thread. ``publish`` is
    thread-safe, and (as in the eef GUI) node methods are called from Qt slots.

DEADMAN MODEL (differs from the eef mouse GUI -- this is a fork, not the bridge):
  * The big toggle flips a LOCAL ``self._engaged`` bool. There is no bridge, no
    ~/eef/state, no SetParameters: the published ``engaged`` is exactly this bool,
    and the published ``gain`` is exactly the slider value, live every tick.
  * ENGAGE (OFF -> ON) keeps the two-click confirm ("robot WILL move"); DISENGAGE
    (ON -> OFF) is a single click (safe -- the env falls back to the policy).
  * The slider label still says "pending -> applies on next engage": the RL env
    LATCHES the gain at the engage edge (anchor reset), so mid-engage slider moves
    do not retroactively rescale the current intervention -- the wording holds.

console_script entry point: ``gello_hil_gui = ur_gello_bringup.gello_hil_gui_node:main``
"""

import os
import signal
import sys
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# FROZEN CONTRACT constants.
DEADMAN_TOPIC = "/hil/deadman"
GAIN_MIN = 0.10
GAIN_MAX = 1.00
PUBLISH_HZ = 20.0
PUBLISH_PERIOD_MS = int(round(1000.0 / PUBLISH_HZ))  # 50 ms == 20 Hz

# A publish gap longer than this (seconds) means the 20 Hz heartbeat has
# stalled -> the "publishing @20Hz" lamp goes red (mirrors the env's staleness).
_PUB_STALE_S = 0.2

_GRAY = "#888888"
_RED = "#cc3333"
_GREEN = "#22aa22"


# Button chrome: a filled, bordered, rounded, hover-reactive control that reads
# unmistakably as "clickable" -- so it is NOT confused with the flat, square,
# full-width STATUS BANNERS (the H1 and the state label, which stay borderless
# QLabels). Action is colour-coded: ENGAGE green (go), DISENGAGE blue
# (stop/hand back to policy), armed-confirm orange.
def _btn_css(bg, border, big=False):
    size = ("font-size: 16pt; min-height: 56px; padding: 15px;" if big
            else "font-size: 13pt; min-height: 38px; padding: 10px 16px;")
    return (
        f"QPushButton {{ {size} font-weight: bold; border-radius: 9px; "
        f"color: white; background-color: {bg}; border: 2px solid {border}; }} "
        f"QPushButton:hover {{ background-color: {border}; }} "
        f"QPushButton:pressed {{ background-color: {border}; padding-top: 17px; }} "
        f"QPushButton:disabled {{ background-color: #cfd8dc; color: #7a8a94; "
        f"border: 2px solid #b0bec5; }}"
    )


_PRIMARY_ENGAGE = _btn_css("#2e7d32", "#1b5e20", big=True)      # green = go (ENGAGE)
_PRIMARY_DISENGAGE = _btn_css("#1565c0", "#0d47a1", big=True)   # blue = hand back
_PRIMARY_ARMED = _btn_css("#ef6c00", "#e65100", big=True)       # orange = "click again"


# --------------------------------------------------------------------------- #
# Pure, display-free helpers (unit-testable without a Qt window)               #
# --------------------------------------------------------------------------- #
def clamp_gain(gain: float) -> float:
    """Clamp a raw slider gain into the frozen [0.10, 1.00] contract range."""
    g = float(gain)
    if g < GAIN_MIN:
        return GAIN_MIN
    if g > GAIN_MAX:
        return GAIN_MAX
    return g


def build_deadman_data(engaged, gain: float) -> list:
    """Build the frozen ``data`` payload: ``[engaged, gain]``.

    ``engaged`` is coerced to exactly 0.0 or 1.0; ``gain`` is clamped into
    [0.10, 1.00]. This is the single source of truth for the field ORDER and the
    clamping, so it can be unit-tested headlessly (no display, no ROS).
    """
    eng = 1.0 if bool(engaged) else 0.0
    return [eng, clamp_gain(gain)]


def build_deadman_msg(engaged, gain: float) -> Float32MultiArray:
    """Wrap :func:`build_deadman_data` into the frozen Float32MultiArray type."""
    msg = Float32MultiArray()
    msg.data = build_deadman_data(engaged, gain)
    return msg


# --------------------------------------------------------------------------- #
# Node: owns ALL ROS I/O (here, just the /hil/deadman publisher)              #
# --------------------------------------------------------------------------- #
class HilGuiNode(Node):
    """rclpy Node backing the HIL deadman GUI. Owns the /hil/deadman publisher.

    Default reliable QoS, depth 10 (``create_publisher(..., 10)``) -- exactly the
    frozen contract. ``publish_deadman`` is thread-safe and is called from the Qt
    20 Hz QTimer; the freshness getter is called from the Qt refresh timer.
    """

    def __init__(self, node_name: str = "gello_hil_gui_node") -> None:
        super().__init__(node_name)

        self._lock = threading.Lock()
        self._last_pub_t: Optional[float] = None  # monotonic; None => never pub
        self._pub_count = 0

        # Default reliable QoS, depth 10 -- the frozen contract.
        self._pub = self.create_publisher(Float32MultiArray, DEADMAN_TOPIC, 10)

        self.get_logger().info(
            "gello_hil_gui_node up; publishing %s at %.0f Hz (data=[engaged, gain])"
            % (DEADMAN_TOPIC, PUBLISH_HZ))

    def publish_deadman(self, engaged, gain: float) -> None:
        """Publish one heartbeat: data=[float(engaged), clamped(gain)]."""
        self._pub.publish(build_deadman_msg(engaged, gain))
        with self._lock:
            self._last_pub_t = time.monotonic()
            self._pub_count += 1

    # --------------------------------------------------- thread-safe getters
    def get_pub_status(self) -> dict:
        """Copy of the heartbeat health for the Qt polling timer (thread-safe)."""
        now = time.monotonic()
        with self._lock:
            age = None if self._last_pub_t is None else now - self._last_pub_t
            return {"age_s": age, "count": self._pub_count}


# --------------------------------------------------------------------------- #
# Main window (Qt / main thread only)                                          #
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self, node: HilGuiNode):
        super().__init__()
        self._node = node
        # Local operator intent -- the ONLY source of the published `engaged`.
        self._engaged = False
        # Two-click confirm gates, keyed by action id. A gate is 'armed' after
        # the first click and fires on the second (mouse metaphor: no dialogs).
        self._armed = {}

        self.setWindowTitle("GELLO -> UR7e  HIL deadman")
        self._build_ui()

        # 20 Hz heartbeat publisher (the frozen contract rate). Always publishes
        # -- even when DISENGAGED -- so the env's staleness watchdog stays fed.
        self._pub_timer = QTimer(self)
        self._pub_timer.timeout.connect(self._publish_tick)
        self._pub_timer.start(PUBLISH_PERIOD_MS)

        # ~10 Hz widget refresh (drives the lamps / labels, does NO ROS I/O).
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._refresh)
        self._ui_timer.start(100)

        self._refresh()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        # Breathing room so the flat status banners and the rounded buttons never
        # abut (part of the banner-vs-button visual separation).
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # --- H1: persistent red collision-gate banner (always visible) ------
        self._h1 = QLabel(
            "COLLISION GATE OFF — clear the elbow-swing envelope, hand on E-STOP")
        self._h1.setAlignment(Qt.AlignCenter)
        self._h1.setStyleSheet(
            "background-color: #cc3333; color: white; font-weight: bold; "
            "font-size: 12pt; padding: 8px;")
        root.addWidget(self._h1)

        # --- big state indicator (driven by self._engaged) ------------------
        self._state_label = QLabel("DISENGAGED")
        self._state_label.setAlignment(Qt.AlignCenter)
        self._state_label.setStyleSheet(
            "background-color: #888888; color: white; font-weight: bold; "
            "font-size: 20pt; padding: 18px;")
        root.addWidget(self._state_label)

        # --- primary deadman toggle (the whole "button") --------------------
        self._primary = QPushButton("ENGAGE")
        self._primary.setStyleSheet(_PRIMARY_ENGAGE)
        self._primary.clicked.connect(self._on_primary)
        root.addWidget(self._primary)

        # No reclutch / re-arm / gripper rows: this GUI only carries the deadman
        # signal. Engage = hold the deadman down; disengage = release it and the
        # RL policy resumes instantly. The env latches the anchor on the engage
        # edge, so "off -> move GELLO -> on" is the whole workflow.
        self._hint = QLabel(
            "Engage, then move the PHYSICAL GELLO leader to intervene. "
            "Disengage to hand control back to the policy.")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #aa6600;")
        root.addWidget(self._hint)

        # --- sensitivity (DPI) slider = gain --------------------------------
        root.addWidget(self._build_slider_box())

        # --- heartbeat health lamp ------------------------------------------
        root.addWidget(self._build_status_box(), stretch=1)

        # --- H5 static note -------------------------------------------------
        self._h5 = QLabel(
            "H5: the pendant SPEED SLIDER has NO effect on the RL controller's "
            "speed — the real caps live in the env's controller config, not this "
            "GUI. This slider only sets the intervention sensitivity (gain).")
        self._h5.setWordWrap(True)
        self._h5.setStyleSheet("color: #666666; font-style: italic;")
        root.addWidget(self._h5)

    def _build_slider_box(self):
        box = QGroupBox("Sensitivity — gain (DPI), 0.10 fine .. 1.00 1:1")
        lay = QVBoxLayout(box)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(10)    # 0.10 fine
        self._slider.setMaximum(100)   # 1.00 = 1:1 default
        self._slider.setValue(100)
        self._slider.setTickInterval(10)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self._slider)

        self._slider_label = QLabel("")
        self._slider_label.setStyleSheet("font-weight: bold;")
        lay.addWidget(self._slider_label)
        self._update_slider_label()
        return box

    def _build_status_box(self):
        box = QGroupBox("Deadman heartbeat")
        grid = QGridLayout(box)

        self._lamp_pub = QLabel("publishing: --")
        grid.addWidget(self._lamp_pub, 0, 0, 1, 2)

        grid.addWidget(QLabel("topic:"), 1, 0)
        t = QLabel(f"{DEADMAN_TOPIC}  (std_msgs/Float32MultiArray)")
        t.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(t, 1, 1)

        grid.addWidget(QLabel("published data:"), 2, 0)
        self._data_label = QLabel("—")
        self._data_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self._data_label, 2, 1)
        return box

    # -------------------------------------------------------------- slider --
    def _slider_gain(self) -> float:
        """The slider value as a clamped [0.10, 1.00] gain (published directly)."""
        return clamp_gain(self._slider.value() / 100.0)

    def _on_slider(self, _v):
        self._update_slider_label()

    def _update_slider_label(self):
        pending = self._slider_gain()
        if self._engaged:
            self._slider_label.setText(
                f"pending {pending:.2f} -> applies on next engage "
                f"(env latches gain at the engage edge)")
            self._slider_label.setStyleSheet(
                "font-weight: bold; color: #aa6600;")
        else:
            self._slider_label.setText(
                f"pending {pending:.2f} — commits on the next engage")
            self._slider_label.setStyleSheet("font-weight: bold;")

    # ---------------------------------------------------- two-click confirm --
    def _armed_click(self, key, action, warn="the robot WILL move"):
        """First click arms (orange 'click again', 3 s window); second fires."""
        if self._armed.get(key):
            self._armed[key] = False
            action()
            self._refresh()
            return
        self._armed[key] = True
        self.statusBar().showMessage(
            f"Click AGAIN within 3 s — {warn}.", 3000)
        QTimer.singleShot(3000, lambda: self._disarm(key))
        self._refresh()

    def _disarm(self, key):
        if self._armed.get(key):
            self._armed[key] = False
            self._refresh()

    # ------------------------------------------------------------- actions --
    def _on_primary(self):
        if self._engaged:
            # DISENGAGE = safe (env falls back to the policy). Single click.
            self._engaged = False
            self.statusBar().showMessage(
                "Disengaged — deadman released, policy resumes.", 3000)
            self._refresh()
        else:
            # ENGAGE = start intervention -> two-click confirm.
            self._armed_click(
                "primary",
                self._do_engage,
                warn="ENGAGE — the arm will follow the GELLO leader")

    def _do_engage(self):
        self._engaged = True
        gain = self._slider_gain()
        self.statusBar().showMessage(
            f"Engaged — deadman held, gain={gain:.2f}. Move the GELLO leader.",
            4000)

    # ---------------------------------------------------- 20 Hz publish tick --
    def _publish_tick(self):
        """Heartbeat: publish [engaged, gain] EVERY tick (even when disengaged)."""
        self._node.publish_deadman(self._engaged, self._slider_gain())

    # -------------------------------------------------------------- refresh --
    def _refresh(self):
        self._apply_state_indicator()
        self._apply_buttons()
        self._apply_pub_lamp()
        self._update_slider_label()

    def _apply_state_indicator(self):
        if self._engaged:
            color, text = _GREEN, "ENGAGED — intervening (deadman held)"
        else:
            color, text = _GRAY, "DISENGAGED — policy in control"
        self._state_label.setText(text)
        self._state_label.setStyleSheet(
            f"background-color: {color}; color: white; font-weight: bold; "
            "font-size: 20pt; padding: 18px;")

    def _apply_buttons(self):
        if self._engaged:
            self._primary.setText("DISENGAGE (release deadman)")
            primary_style = _PRIMARY_DISENGAGE          # blue = hand back
        else:
            self._primary.setText("ENGAGE  (hold deadman to intervene)")
            primary_style = _PRIMARY_ENGAGE             # green = go
        self._primary.setEnabled(True)

        # Keep the armed (orange) look while the confirm window is open, but only
        # while DISENGAGED (engage is the only confirmable edge); otherwise clear
        # the arm and fall back to the state-based colour/label set above.
        if self._armed.get("primary") and not self._engaged:
            self._primary.setText("Click AGAIN to ENGAGE (robot WILL move)")
            self._primary.setStyleSheet(_PRIMARY_ARMED)
        else:
            self._armed["primary"] = False
            self._primary.setStyleSheet(primary_style)

    def _apply_pub_lamp(self):
        status = self._node.get_pub_status()
        age = status["age_s"]
        gain = self._slider_gain()
        engaged = 1.0 if self._engaged else 0.0
        self._data_label.setText(
            f"[engaged={engaged:.1f}, gain={gain:.2f}]  ({status['count']} sent)")
        if age is None:
            self._lamp_pub.setText("publishing: no heartbeat yet")
            self._lamp_pub.setStyleSheet(f"color: {_RED}; font-weight: bold;")
        elif age <= _PUB_STALE_S:
            self._lamp_pub.setText(
                f"publishing @ ~{PUBLISH_HZ:.0f} Hz — heartbeat healthy")
            self._lamp_pub.setStyleSheet(f"color: {_GREEN}; font-weight: bold;")
        else:
            self._lamp_pub.setText(
                f"publishing: STALLED ({age:.2f}s since last heartbeat)")
            self._lamp_pub.setStyleSheet(f"color: {_RED}; font-weight: bold;")

    def closeEvent(self, event):
        # Stop the QTimers BEFORE tearing down rclpy, so a late publish tick
        # cannot do ROS I/O against an already-shutdown context (which would
        # raise inside a Qt slot during teardown).
        for t in (getattr(self, "_pub_timer", None),
                  getattr(self, "_ui_timer", None)):
            if t is not None:
                t.stop()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
        event.accept()


# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #
def _spin_node(node):
    try:
        rclpy.spin(node)
    except Exception:  # noqa: BLE001 -- spin raises when shutdown() is called
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


def main(args=None):
    rclpy.init(args=args)

    # Defensive: if anything upstream pointed Qt at a bundled plugin dir, clear
    # it so we use the system PyQt5 platform plugins (see recorder GUI note).
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    node = HilGuiNode()

    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv if args is None else args)
    window = MainWindow(node)
    window.resize(760, 620)
    window.show()

    # Make Ctrl-C in the launching terminal actually quit Qt (PyQt gotcha: the
    # C++ loop never yields to Python's SIGINT handler on its own).
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    pump = QTimer()
    pump.start(200)
    pump.timeout.connect(lambda: None)

    exit_code = app.exec_()

    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    spin_thread.join(timeout=2.0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
