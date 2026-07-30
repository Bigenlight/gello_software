#!/usr/bin/env python3
"""PyQt5 operator GUI for the UR7e+GELLO HIL RL intervention deadman.

This GUI drives the HIL RL env's ``GelloIntervention`` deadman by publishing a
heartbeat on the ROS topic ``/hil/deadman`` -- it is the GUI ALTERNATIVE to the
terminal-focus spacebar deadman. It runs WITHOUT the teleop bridge and WITHOUT
``control_mode:=eef``: it never calls a single bridge service, never talks to the
robot or GELLO. Its only direct robot-motion signal is the deadman publisher;
separate operator services select success mode, mark MANUAL success, approve
HOME, and start the next scene. The RL env (the ``RosTopicDeadman``
DeadmanSource in ``serl_ur_infra/ur_env/envs/wrappers.py``) subscribes and reads
the engage/gain signal off the topic.

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
  * :class:`HilGuiNode` (an rclpy Node) owns ALL ROS I/O: the frozen
    ``/hil/deadman`` publisher plus read-only actor-status telemetry and the
    operator's explicit ``/hil/scene_ready`` Trigger client. It is spun on a
    background daemon thread with a plain ``rclpy.spin``.
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
    (ON -> OFF) is a single click. During an active episode it hands back to the
    policy; HOMING/WAIT remains under the actor's episode gate.
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

from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger

from ur_gello_bringup.hil_actor_status import (
    ACTOR_STATUS_TOPIC,
    AUTO_SUCCESS_SERVICE,
    MANUAL_SUCCESS_SERVICE,
    SCENE_READY_SERVICE,
    SCENE_REQUEST_IDLE,
    actor_run_changed,
    actor_banner,
    classifier_verdict_summary,
    engage_button_enabled,
    format_actor_status,
    manual_success_enabled,
    parse_actor_status,
    scene_ready_enabled,
    update_classifier_latch,
    update_terminal_latch,
)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
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

_SCENE_RELEASE_WAIT = "release_wait"
_SCENE_CALL_PENDING = "call_pending"
_SCENE_ACCEPTED_WAIT = "accepted_wait"
_SCENE_RELEASE_DELAY_MS = 100
_SCENE_CALL_TIMEOUT_S = 5.0
_OPERATOR_CALL_TIMEOUT_S = 5.0


# Button chrome: a filled, bordered, rounded, hover-reactive control that reads
# unmistakably as "clickable" -- so it is NOT confused with the flat, square,
# full-width STATUS BANNERS (the H1 and the state label, which stay borderless
# QLabels). Action is colour-coded: ENGAGE green (go), DISENGAGE blue
# (stop/hand back to policy), armed-confirm orange.
def _btn_css(bg, border, big=False):
    # Widget-specific minimum heights are set on the handful of prominent
    # controls below.  Keeping min-height out of QSS avoids Qt counting it once
    # for layout and again with padding when painting, which made labels appear
    # to overlap adjacent buttons on some desktop scale factors.
    size = ("font-size: 14pt; padding: 8px 12px;" if big
            else "font-size: 11pt; padding: 6px 10px;")
    return (
        f"QPushButton {{ {size} font-weight: bold; border-radius: 9px; "
        f"color: white; background-color: {bg}; border: 2px solid {border}; }} "
        f"QPushButton:hover {{ background-color: {border}; }} "
        f"QPushButton:pressed {{ background-color: {border}; }} "
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
# Node: owns ALL ROS I/O                                                        #
# --------------------------------------------------------------------------- #
class HilGuiNode(Node):
    """ROS backing for deadman output, actor display, and scene-ready request.

    Default reliable QoS, depth 10 (``create_publisher(..., 10)``) -- exactly the
    frozen contract. ``publish_deadman`` is thread-safe and is called from the Qt
    20 Hz QTimer; the freshness getter is called from the Qt refresh timer.
    """

    def __init__(self, node_name: str = "gello_hil_gui_node") -> None:
        super().__init__(node_name)

        self._lock = threading.Lock()
        self._last_pub_t: Optional[float] = None  # monotonic; None => never pub
        self._pub_count = 0
        self._actor_status: Optional[dict] = None
        self._actor_status_t: Optional[float] = None
        self._classifier_latch: Optional[dict] = None
        self._terminal_latch = ""
        self._actor_diagnostic = ""
        self._actor_diagnostic_t: Optional[float] = None

        # Default reliable QoS, depth 10 -- the frozen contract.
        self._pub = self.create_publisher(Float32MultiArray, DEADMAN_TOPIC, 10)
        self.create_subscription(
            String, ACTOR_STATUS_TOPIC, self._on_actor_status, 10
        )
        self._scene_ready_client = self.create_client(
            Trigger, SCENE_READY_SERVICE
        )
        self._auto_success_client = self.create_client(
            SetBool, AUTO_SUCCESS_SERVICE
        )
        self._manual_success_client = self.create_client(
            Trigger, MANUAL_SUCCESS_SERVICE
        )

        self.get_logger().info(
            "gello_hil_gui_node up; publishing %s at %.0f Hz "
            "(data=[engaged, gain]); actor status=%s; scene ready=%s; "
            "success mode=%s; manual success=%s"
            % (
                DEADMAN_TOPIC,
                PUBLISH_HZ,
                ACTOR_STATUS_TOPIC,
                SCENE_READY_SERVICE,
                AUTO_SUCCESS_SERVICE,
                MANUAL_SUCCESS_SERVICE,
            )
        )

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

    def _on_actor_status(self, msg: String) -> None:
        """Cache one valid actor status; malformed telemetry never replaces it."""
        now = time.monotonic()
        try:
            status = parse_actor_status(msg.data)
        except Exception as exc:  # noqa: BLE001 -- malformed telemetry is inert
            diagnostic = f"invalid actor status ignored: {exc}"
            with self._lock:
                self._actor_diagnostic = diagnostic
                self._actor_diagnostic_t = now
            self.get_logger().warning(diagnostic)
            return

        with self._lock:
            if actor_run_changed(self._actor_status, status):
                self._classifier_latch = None
                self._terminal_latch = ""
            self._actor_status = status
            self._actor_status_t = now
            self._classifier_latch = update_classifier_latch(
                self._classifier_latch, status
            )
            self._terminal_latch = update_terminal_latch(
                self._terminal_latch, status
            )
            self._actor_diagnostic = ""
            self._actor_diagnostic_t = None

    def get_actor_snapshot(self) -> dict:
        """Thread-safe copy consumed by the Qt refresh timer."""
        now = time.monotonic()
        with self._lock:
            status = (
                None if self._actor_status is None else dict(self._actor_status)
            )
            classifier = (
                None
                if self._classifier_latch is None
                else dict(self._classifier_latch)
            )
            status_age = (
                None
                if self._actor_status_t is None
                else now - self._actor_status_t
            )
            diagnostic_age = (
                None
                if self._actor_diagnostic_t is None
                else now - self._actor_diagnostic_t
            )
            return {
                "status": status,
                "age_s": status_age,
                "classifier_latch": classifier,
                "terminal_latch": self._terminal_latch,
                "diagnostic": self._actor_diagnostic,
                "diagnostic_age_s": diagnostic_age,
            }

    def scene_ready_service_ready(self) -> bool:
        if not rclpy.ok():
            return False
        try:
            return bool(self._scene_ready_client.service_is_ready())
        except Exception:  # noqa: BLE001 -- shutdown race
            return False

    def call_scene_ready(self):
        """Issue the non-blocking Trigger request; Qt polls the returned Future."""
        return self._scene_ready_client.call_async(Trigger.Request())

    @staticmethod
    def _client_ready(client) -> bool:
        if not rclpy.ok():
            return False
        try:
            return bool(client.service_is_ready())
        except Exception:  # noqa: BLE001 -- shutdown race
            return False

    def auto_success_service_ready(self) -> bool:
        return self._client_ready(self._auto_success_client)

    def manual_success_service_ready(self) -> bool:
        return self._client_ready(self._manual_success_client)

    def call_auto_success(self, enabled: bool):
        request = SetBool.Request()
        request.data = bool(enabled)
        return self._auto_success_client.call_async(request)

    def call_manual_success(self):
        return self._manual_success_client.call_async(Trigger.Request())


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
        self._scene_request_state = SCENE_REQUEST_IDLE
        self._scene_future = None
        self._scene_call_started: Optional[float] = None
        self._scene_result_text = ""
        self._scene_result_color = _GRAY
        self._last_actor_state = None
        self._last_actor_status = None
        self._auto_success = False
        self._mode_future = None
        self._mode_call_started: Optional[float] = None
        self._requested_auto_success = False
        self._mode_ack_value = None
        self._manual_success_future = None
        self._manual_success_call_started: Optional[float] = None
        self._manual_success_queued = False
        self._manual_success_target = None
        self._success_run_id = None
        self._success_result_text = ""
        self._success_result_color = _GRAY
        self._closing = False

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
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # --- H1: persistent red collision-gate banner (always visible) ------
        self._h1 = QLabel(
            "COLLISION GATE OFF — clear the elbow-swing envelope, hand on E-STOP")
        self._h1.setAlignment(Qt.AlignCenter)
        self._h1.setStyleSheet(
            "background-color: #cc3333; color: white; font-weight: bold; "
            "font-size: 11pt; padding: 5px;")
        root.addWidget(self._h1)

        # --- big state indicator (actor state first, local intent fallback) -
        self._state_label = QLabel("DISENGAGED")
        self._state_label.setAlignment(Qt.AlignCenter)
        self._state_label.setStyleSheet(
            "background-color: #888888; color: white; font-weight: bold; "
            "font-size: 16pt; padding: 10px;")
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
            "During an episode: ENGAGE to intervene with GELLO; DISENGAGE to "
            "return control to the policy.")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #aa6600;")
        root.addWidget(self._hint)

        # --- actor state + scene-ready control ------------------------------
        root.addWidget(self._build_actor_status_box())

        # --- compact diagnostics row ----------------------------------------
        diagnostics = QHBoxLayout()
        diagnostics.setSpacing(6)
        diagnostics.addWidget(self._build_slider_box(), stretch=1)
        diagnostics.addWidget(self._build_status_box(), stretch=1)
        root.addLayout(diagnostics)

        # --- H5 static note -------------------------------------------------
        self._h5 = QLabel(
            "Gain changes GELLO intervention sensitivity only; robot speed caps "
            "remain in the environment configuration.")
        self._h5.setWordWrap(True)
        self._h5.setStyleSheet(
            "color: #666666; font-size: 9pt; font-style: italic;")
        root.addWidget(self._h5)

    def _build_slider_box(self):
        box = QGroupBox("Sensitivity — gain (DPI), 0.10 fine .. 1.00 1:1")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

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
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

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

    def _build_actor_status_box(self):
        box = QGroupBox("HIL actor / episode operator controls")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        # Spanning buttons used to make both columns expand equally, leaving a
        # very large visual indent before every value.  Keep the key column at
        # its natural width and give all spare width to the value column.
        grid.setColumnMinimumWidth(0, 112)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("state / owner:"), 0, 0)
        self._actor_owner_state = QLabel("actor status: not received")
        self._actor_owner_state.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self._actor_owner_state, 0, 1)

        grid.addWidget(QLabel("run:"), 1, 0)
        self._actor_run = QLabel("—")
        self._actor_run.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self._actor_run, 1, 1)

        grid.addWidget(QLabel("progress:"), 2, 0)
        self._actor_episode = QLabel("episode/step: — / —   env step: —")
        grid.addWidget(self._actor_episode, 2, 1)

        self._classifier_verdict = QLabel("LAST CLASSIFIER: NO RESULT")
        self._classifier_verdict.setAlignment(Qt.AlignCenter)
        self._classifier_verdict.setMinimumHeight(34)
        self._classifier_verdict.setWordWrap(True)
        self._classifier_verdict.setStyleSheet(
            f"color: {_GRAY}; font-size: 16pt; font-weight: bold; padding: 2px;"
        )
        grid.addWidget(self._classifier_verdict, 3, 0, 1, 2)

        grid.addWidget(QLabel("classifier detail:"), 4, 0)
        classifier_col = QVBoxLayout()
        classifier_col.setContentsMargins(0, 0, 0, 0)
        classifier_col.setSpacing(1)
        self._classifier_current = QLabel("current step evaluated: —")
        self._classifier_score = QLabel(
            "p(success): —   threshold: —   last eval env step: —"
        )
        self._classifier_score.setTextInteractionFlags(Qt.TextSelectableByMouse)
        classifier_col.addWidget(self._classifier_current)
        classifier_col.addWidget(self._classifier_score)
        grid.addLayout(classifier_col, 4, 1)

        grid.addWidget(QLabel("success mode:"), 5, 0)
        mode_col = QVBoxLayout()
        mode_col.setContentsMargins(0, 0, 0, 0)
        mode_col.setSpacing(2)
        mode_buttons = QHBoxLayout()
        mode_buttons.setSpacing(6)
        self._manual_mode_button = QPushButton("MANUAL")
        self._auto_mode_button = QPushButton("AUTO (classifier)")
        self._manual_mode_button.setMinimumHeight(34)
        self._auto_mode_button.setMinimumHeight(34)
        self._manual_mode_button.clicked.connect(
            lambda: self._request_success_mode(False)
        )
        self._auto_mode_button.clicked.connect(
            lambda: self._request_success_mode(True)
        )
        mode_buttons.addWidget(self._manual_mode_button)
        mode_buttons.addWidget(self._auto_mode_button)
        self._success_mode_note = QLabel(
            "MANUAL — classifier is display-only; use MARK SUCCESS"
        )
        self._success_mode_note.setWordWrap(True)
        mode_col.addLayout(mode_buttons)
        mode_col.addWidget(self._success_mode_note)
        grid.addLayout(mode_col, 5, 1)

        self._manual_success_button = QPushButton(
            "MARK SUCCESS (current episode)"
        )
        self._manual_success_button.setMinimumHeight(42)
        self._manual_success_button.setStyleSheet(
            _btn_css("#2e7d32", "#1b5e20", big=True)
        )
        self._manual_success_button.clicked.connect(self._on_manual_success)
        grid.addWidget(self._manual_success_button, 6, 0, 1, 2)

        self._success_result = QLabel(self._success_result_text)
        self._success_result.setAlignment(Qt.AlignCenter)
        self._success_result.setWordWrap(True)
        grid.addWidget(self._success_result, 7, 0, 1, 2)

        grid.addWidget(QLabel("terminal:"), 8, 0)
        self._actor_terminal = QLabel("terminal reason: —")
        grid.addWidget(self._actor_terminal, 8, 1)

        grid.addWidget(QLabel("telemetry:"), 9, 0)
        telemetry_col = QVBoxLayout()
        telemetry_col.setContentsMargins(0, 0, 0, 0)
        telemetry_col.setSpacing(1)
        self._actor_age = QLabel("status age: —")
        self._actor_message = QLabel("Waiting for /hil/actor_status")
        self._actor_message.setMinimumHeight(34)
        self._actor_message.setWordWrap(True)
        self._actor_message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        telemetry_col.addWidget(self._actor_age)
        telemetry_col.addWidget(self._actor_message)
        grid.addLayout(telemetry_col, 9, 1)

        self._scene_ready_button = QPushButton(
            "START / NEXT ITERATION (policy)"
        )
        self._scene_ready_button.setMinimumHeight(36)
        self._scene_ready_button.setStyleSheet(
            _btn_css("#1565c0", "#0d47a1")
        )
        self._scene_ready_button.clicked.connect(self._on_scene_ready)
        grid.addWidget(self._scene_ready_button, 10, 0, 1, 2)

        self._scene_result = QLabel("")
        self._scene_result.setAlignment(Qt.AlignCenter)
        self._scene_result.setWordWrap(True)
        grid.addWidget(self._scene_result, 11, 0, 1, 2)
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
            # DISENGAGE is always available. In HOMING/WAIT it releases the
            # deadman without bypassing the independent scene-ready gate.
            self._engaged = False
            status = self._node.get_actor_snapshot()["status"]
            active = (
                status is None
                or status.get("state")
                in {"POLICY_RUNNING", "HUMAN_INTERVENTION", "HOLD"}
            )
            message = (
                "Disengaged — deadman released, policy resumes."
                if active
                else "Disengaged — deadman released; actor gate remains active."
            )
            self.statusBar().showMessage(message, 3000)
            self._refresh()
        else:
            snapshot = self._node.get_actor_snapshot()
            status = snapshot["status"]
            if not engage_button_enabled(status, engaged=False):
                state = status.get("state", "unknown") if status else "unknown"
                self.statusBar().showMessage(
                    f"ENGAGE unavailable while actor state is {state}.", 3000
                )
                self._refresh()
                return
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

    def _set_success_result(self, text: str, color: str) -> None:
        self._success_result_text = str(text)
        self._success_result_color = color

    def _request_success_mode(self, auto_success: bool) -> None:
        """Request MANUAL/AUTO without changing the displayed mode before ACK."""

        if self._mode_future is not None:
            return
        if not self._node.auto_success_service_ready():
            self._set_success_result(
                f"FAILED: {AUTO_SUCCESS_SERVICE} is not available.", _RED
            )
            self._refresh()
            return
        try:
            self._mode_future = self._node.call_auto_success(auto_success)
        except Exception as exc:  # noqa: BLE001 -- ROS/shutdown race
            self._mode_future = None
            self._set_success_result(f"FAILED to set success mode: {exc}", _RED)
            self._refresh()
            return
        self._requested_auto_success = bool(auto_success)
        self._mode_call_started = time.monotonic()
        requested = "AUTO" if auto_success else "MANUAL"
        self._set_success_result(f"Switching success mode to {requested}...", "#aa6600")
        self._refresh()

    def _on_manual_success(self) -> None:
        """Queue one operator success event for the displayed active episode."""

        snapshot = self._node.get_actor_snapshot()
        status = snapshot["status"]
        service_ready = self._node.manual_success_service_ready()
        if not manual_success_enabled(
            status,
            auto_success=self._auto_success,
            request_pending=self._manual_success_future is not None,
            success_queued=self._manual_success_queued,
            service_ready=service_ready,
        ):
            return
        try:
            self._manual_success_future = self._node.call_manual_success()
        except Exception as exc:  # noqa: BLE001 -- ROS/shutdown race
            self._manual_success_future = None
            self._set_success_result(f"FAILED to mark success: {exc}", _RED)
            self._refresh()
            return
        self._manual_success_target = (
            status["run_id"],
            int(status["episode_id"]),
        )
        self._manual_success_call_started = time.monotonic()
        self._set_success_result("MARK SUCCESS request pending...", "#aa6600")
        self._refresh()

    @staticmethod
    def _cancel_future(future) -> None:
        if future is None:
            return
        try:
            future.cancel()
        except Exception:  # noqa: BLE001 -- shutdown race
            pass

    def _poll_success_controls(self, snapshot: dict) -> None:
        """Resolve independent mode/success Futures and episode boundaries."""

        status = snapshot["status"]
        run_id = None if status is None else status.get("run_id")
        if run_id is not None and self._success_run_id is None:
            self._success_run_id = run_id
        elif run_id is not None and run_id != self._success_run_id:
            self._cancel_future(self._mode_future)
            self._cancel_future(self._manual_success_future)
            self._mode_future = None
            self._mode_call_started = None
            self._mode_ack_value = None
            self._manual_success_future = None
            self._manual_success_call_started = None
            self._manual_success_queued = False
            self._manual_success_target = None
            self._success_run_id = run_id
            self._auto_success = bool(status.get("auto_success", False))
            self._set_success_result(
                "New actor run detected; success controls reset.", "#1565c0"
            )

        mode_completed = False
        future = self._mode_future
        if future is not None and future.done():
            try:
                response = future.result()
                accepted = bool(response.success)
                message = (response.message or "").strip()
            except Exception as exc:  # noqa: BLE001 -- surface service error
                accepted = False
                message = f"service error: {exc}"
            self._mode_future = None
            self._mode_call_started = None
            mode_completed = accepted
            if accepted:
                self._auto_success = self._requested_auto_success
                self._mode_ack_value = self._auto_success
                self._manual_success_queued = False
                self._manual_success_target = None
                detail = f" — {message}" if message else ""
                mode = "AUTO" if self._auto_success else "MANUAL"
                self._set_success_result(f"Success mode: {mode}{detail}", _GREEN)
            else:
                self._set_success_result(
                    f"FAILED to set success mode: {message or 'request rejected'}",
                    _RED,
                )
        elif (
            future is not None
            and self._mode_call_started is not None
            and time.monotonic() - self._mode_call_started
            > _OPERATOR_CALL_TIMEOUT_S
        ):
            self._cancel_future(future)
            self._mode_future = None
            self._mode_call_started = None
            self._set_success_result("FAILED: success-mode request timed out.", _RED)

        future = self._manual_success_future
        if future is not None and future.done():
            try:
                response = future.result()
                accepted = bool(response.success)
                message = (response.message or "").strip()
            except Exception as exc:  # noqa: BLE001 -- surface service error
                accepted = False
                message = f"service error: {exc}"
            self._manual_success_future = None
            self._manual_success_call_started = None
            if accepted:
                self._manual_success_queued = True
                detail = f" — {message}" if message else ""
                self._set_success_result(
                    f"SUCCESS queued for the next transition{detail}", _GREEN
                )
            else:
                self._manual_success_target = None
                self._set_success_result(
                    f"FAILED to mark success: {message or 'request rejected'}",
                    _RED,
                )
        elif (
            future is not None
            and self._manual_success_call_started is not None
            and time.monotonic() - self._manual_success_call_started
            > _OPERATOR_CALL_TIMEOUT_S
        ):
            self._cancel_future(future)
            self._manual_success_future = None
            self._manual_success_call_started = None
            self._manual_success_target = None
            self._set_success_result("FAILED: MARK SUCCESS request timed out.", _RED)

        # Actor telemetry is authoritative.  Do not let an older status race
        # backwards over a SetBool reply completed in this same refresh.
        if status is not None and self._mode_future is None and not mode_completed:
            reported_auto = bool(status.get("auto_success", False))
            if self._mode_ack_value is None:
                self._auto_success = reported_auto
            elif reported_auto == self._mode_ack_value:
                self._auto_success = reported_auto
                self._mode_ack_value = None

        if self._manual_success_queued and status is not None:
            active_target = (status["run_id"], int(status["episode_id"]))
            if (
                active_target != self._manual_success_target
                or status.get("state")
                not in {"POLICY_RUNNING", "HUMAN_INTERVENTION", "HOLD"}
            ):
                self._manual_success_queued = False
                self._manual_success_target = None

    def _set_scene_result(self, text: str, color: str) -> None:
        self._scene_result_text = str(text)
        self._scene_result_color = color

    def _on_scene_ready(self) -> None:
        """Release deadman, then submit the actor's current operator gate."""
        snapshot = self._node.get_actor_snapshot()
        status = snapshot["status"]
        service_ready = self._node.scene_ready_service_ready()
        if not scene_ready_enabled(
            status,
            request_state=self._scene_request_state,
            service_ready=service_ready,
        ):
            return
        # Do not wait for the next 50 ms QTimer tick: publish the explicit
        # release immediately, then leave ~100 ms before the Trigger call.
        self._engaged = False
        self._armed["primary"] = False
        try:
            self._node.publish_deadman(False, self._slider_gain())
        except Exception as exc:  # noqa: BLE001 -- shutdown/ROS failure
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._set_scene_result(
                f"FAILED before request: deadman publish error: {exc}", _RED
            )
            self._refresh()
            return

        self._scene_request_state = _SCENE_RELEASE_WAIT
        request_name = (
            "HOME approval"
            if status["state"] == "WAIT_HOME_APPROVAL"
            else "scene-ready"
        )
        self._set_scene_result(
            f"Deadman released; sending {request_name} in 100 ms...", "#aa6600"
        )
        self._refresh()
        QTimer.singleShot(_SCENE_RELEASE_DELAY_MS, self._dispatch_scene_ready)

    def _dispatch_scene_ready(self) -> None:
        if self._closing or self._scene_request_state != _SCENE_RELEASE_WAIT:
            return
        snapshot = self._node.get_actor_snapshot()
        status = snapshot["status"]
        if status is None or status.get("state") not in {
            "WAIT_HOME_APPROVAL",
            "WAIT_SCENE_READY",
        }:
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._set_scene_result(
                "Operator request cancelled: actor left its approval gate.",
                "#aa6600",
            )
            self._refresh()
            return
        if not self._node.scene_ready_service_ready():
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._set_scene_result(
                f"FAILED: {SCENE_READY_SERVICE} is not available.", _RED
            )
            self._refresh()
            return
        try:
            self._scene_future = self._node.call_scene_ready()
        except Exception as exc:  # noqa: BLE001 -- service/shutdown race
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._scene_future = None
            self._set_scene_result(f"FAILED to call scene-ready: {exc}", _RED)
            self._refresh()
            return

        self._scene_request_state = _SCENE_CALL_PENDING
        self._scene_call_started = time.monotonic()
        self._set_scene_result("Scene-ready request pending...", "#aa6600")
        self._refresh()

    def _poll_scene_ready(self, snapshot: dict) -> None:
        status = snapshot["status"]
        state = None if status is None else status.get("state")
        previous_state = self._last_actor_state
        run_changed = bool(
            status is not None
            and actor_run_changed(self._last_actor_status, status)
        )
        if status is not None:
            self._last_actor_status = dict(status)

        if run_changed:
            future = self._scene_future
            if future is not None:
                try:
                    future.cancel()
                except Exception:  # noqa: BLE001 -- shutdown race
                    pass
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._scene_future = None
            self._scene_call_started = None
            previous_state = None
            self._set_scene_result(
                "New actor run detected; previous scene-ready request cleared.",
                "#1565c0",
            )
        self._last_actor_state = state

        if (
            state == "WAIT_HOME_APPROVAL"
            and previous_state != "WAIT_HOME_APPROVAL"
            and self._scene_request_state == SCENE_REQUEST_IDLE
        ):
            self._set_scene_result(
                "Episode ended. Robot is holding. Approve before HOME motion.",
                "#aa6600",
            )

        if (
            state == "WAIT_SCENE_READY"
            and previous_state != "WAIT_SCENE_READY"
            and self._scene_request_state == SCENE_REQUEST_IDLE
        ):
            self._set_scene_result(
                "Robot is HOME. Reset the scene, then start the policy.",
                "#1565c0",
            )

        if (
            self._scene_request_state
            in {_SCENE_CALL_PENDING, _SCENE_ACCEPTED_WAIT}
            and state in {"FAULT", "STOPPED"}
        ):
            future = self._scene_future
            if future is not None:
                try:
                    future.cancel()
                except Exception:  # noqa: BLE001 -- shutdown race
                    pass
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._scene_future = None
            self._scene_call_started = None
            self._set_scene_result(
                f"FAILED: actor entered {state} before policy resumed.", _RED
            )
            return

        # The actor's state is authoritative proof that the accepted request
        # took effect, even if the local Future completion races this refresh.
        if (
            self._scene_request_state
            in {_SCENE_CALL_PENDING, _SCENE_ACCEPTED_WAIT}
            and state == "POLICY_RUNNING"
        ):
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._scene_future = None
            self._scene_call_started = None
            self._set_scene_result(
                "Accepted — actor reports POLICY_RUNNING.", _GREEN
            )
            return

        if (
            self._scene_request_state
            in {_SCENE_CALL_PENDING, _SCENE_ACCEPTED_WAIT}
            and previous_state == "HOMING"
            and state == "WAIT_SCENE_READY"
        ):
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._scene_future = None
            self._scene_call_started = None
            self._set_scene_result(
                "HOME complete. Reset the scene, then start the next iteration.",
                _GREEN,
            )
            return

        if (
            self._scene_request_state == _SCENE_CALL_PENDING
            and state == "HOMING"
        ):
            # WAIT -> HOMING is also actor-authoritative evidence that this
            # request was consumed. Keep the duplicate-click latch until the
            # subsequent POLICY_RUNNING status arrives.
            self._scene_request_state = _SCENE_ACCEPTED_WAIT
            self._scene_future = None
            self._scene_call_started = None
            self._set_scene_result(
                "Accepted — actor is HOMING; waiting for POLICY_RUNNING...",
                _GREEN,
            )
            return

        if self._scene_request_state != _SCENE_CALL_PENDING:
            return
        future = self._scene_future
        if future is None:
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._set_scene_result("FAILED: scene-ready Future is missing.", _RED)
            return

        if future.done():
            try:
                response = future.result()
                accepted = bool(response.success)
                message = (response.message or "").strip()
            except Exception as exc:  # noqa: BLE001 -- surface service error
                accepted = False
                message = f"service error: {exc}"
            self._scene_future = None
            self._scene_call_started = None
            if accepted:
                self._scene_request_state = _SCENE_ACCEPTED_WAIT
                detail = f" ({message})" if message else ""
                self._set_scene_result(
                    f"Accepted{detail}; waiting for actor state change...",
                    _GREEN,
                )
            else:
                self._scene_request_state = SCENE_REQUEST_IDLE
                self._set_scene_result(
                    f"FAILED: {message or 'actor rejected scene-ready'}", _RED
                )
            return

        if (
            self._scene_call_started is not None
            and time.monotonic() - self._scene_call_started
            > _SCENE_CALL_TIMEOUT_S
        ):
            try:
                future.cancel()
            except Exception:  # noqa: BLE001
                pass
            self._scene_future = None
            self._scene_call_started = None
            self._scene_request_state = SCENE_REQUEST_IDLE
            self._set_scene_result(
                f"FAILED: scene-ready reply timed out after "
                f"{_SCENE_CALL_TIMEOUT_S:.0f} s.",
                _RED,
            )

    # ---------------------------------------------------- 20 Hz publish tick --
    def _publish_tick(self):
        """Heartbeat: publish [engaged, gain] EVERY tick (even when disengaged)."""
        self._node.publish_deadman(self._engaged, self._slider_gain())

    # -------------------------------------------------------------- refresh --
    def _refresh(self):
        snapshot = self._node.get_actor_snapshot()
        self._poll_scene_ready(snapshot)
        self._poll_success_controls(snapshot)
        self._apply_state_indicator(snapshot)
        self._apply_buttons(snapshot)
        self._apply_actor_panel(snapshot)
        self._apply_pub_lamp()
        self._update_slider_label()

    def _apply_state_indicator(self, snapshot):
        text, color = actor_banner(
            snapshot["status"], engaged=self._engaged
        )
        self._state_label.setText(text)
        self._state_label.setStyleSheet(
            f"background-color: {color}; color: white; font-weight: bold; "
            "font-size: 16pt; padding: 10px;")

    def _apply_buttons(self, snapshot):
        status = snapshot["status"]
        enabled = engage_button_enabled(status, engaged=self._engaged)
        if self._engaged:
            self._primary.setText("DISENGAGE (release deadman)")
            primary_style = _PRIMARY_DISENGAGE          # blue = hand back
        elif not enabled:
            state = status.get("state", "unknown") if status else "unknown"
            self._primary.setText(f"ENGAGE unavailable while {state}")
            primary_style = _PRIMARY_ENGAGE
        else:
            self._primary.setText("ENGAGE  (hold deadman to intervene)")
            primary_style = _PRIMARY_ENGAGE             # green = go
        self._primary.setEnabled(enabled)

        # Keep the armed (orange) look while the confirm window is open, but only
        # while DISENGAGED (engage is the only confirmable edge); otherwise clear
        # the arm and fall back to the state-based colour/label set above.
        if self._armed.get("primary") and not self._engaged and enabled:
            self._primary.setText("Click AGAIN to ENGAGE (robot WILL move)")
            self._primary.setStyleSheet(_PRIMARY_ARMED)
        else:
            self._armed["primary"] = False
            self._primary.setStyleSheet(primary_style)

    def _apply_actor_panel(self, snapshot):
        status = snapshot["status"]
        formatted = format_actor_status(
            status,
            classifier_latch=snapshot["classifier_latch"],
            terminal_latch=snapshot["terminal_latch"],
            age_s=snapshot["age_s"],
        )
        self._actor_owner_state.setText(formatted["owner_state"])
        self._actor_run.setText(formatted["run"])
        self._actor_episode.setText(formatted["episode"])
        self._classifier_current.setText(formatted["classifier_current"])
        self._classifier_score.setText(formatted["classifier_score"])
        classifier = snapshot["classifier_latch"]
        verdict_text, mode_text, verdict = classifier_verdict_summary(
            classifier, auto_success=self._auto_success
        )
        self._classifier_verdict.setText(verdict_text)
        classifier_color = (
            _GRAY if verdict is None else _GREEN if verdict else "#aa6600"
        )
        self._classifier_verdict.setStyleSheet(
            f"color: {classifier_color}; font-size: 16pt; font-weight: bold; "
            "padding: 2px;"
        )
        self._classifier_score.setStyleSheet(
            f"color: {classifier_color}; font-weight: bold;"
        )
        self._success_mode_note.setText(mode_text)
        self._actor_terminal.setText(formatted["terminal"])
        self._actor_age.setText(formatted["age"])

        diagnostic = snapshot["diagnostic"]
        if diagnostic:
            message = diagnostic
            if formatted["message"]:
                message += f" | last valid: {formatted['message']}"
            self._actor_message.setStyleSheet(
                f"color: {_RED}; font-weight: bold;"
            )
        else:
            message = formatted["message"]
            self._actor_message.setStyleSheet("color: #555555;")
        self._actor_message.setText(message)

        mode_ready = self._node.auto_success_service_ready()
        mode_pending = self._mode_future is not None
        # The selected mode is not an action.  Keeping its button disabled also
        # prevents a redundant SetBool request from racing a queued MARK SUCCESS.
        self._manual_mode_button.setEnabled(
            mode_ready and not mode_pending and self._auto_success
        )
        self._auto_mode_button.setEnabled(
            mode_ready and not mode_pending and not self._auto_success
        )
        if mode_pending:
            pending = "AUTO" if self._requested_auto_success else "MANUAL"
            self._manual_mode_button.setText(f"Switching to {pending}...")
            self._auto_mode_button.setText("Please wait")
        else:
            self._manual_mode_button.setText("MANUAL")
            self._auto_mode_button.setText("AUTO (classifier)")
        self._manual_mode_button.setStyleSheet(
            _btn_css("#2e7d32", "#1b5e20")
            if not self._auto_success
            else _btn_css("#78909c", "#546e7a")
        )
        self._auto_mode_button.setStyleSheet(
            _btn_css("#ef6c00", "#e65100")
            if self._auto_success
            else _btn_css("#78909c", "#546e7a")
        )

        success_ready = self._node.manual_success_service_ready()
        self._manual_success_button.setEnabled(
            manual_success_enabled(
                status,
                auto_success=self._auto_success,
                request_pending=self._manual_success_future is not None,
                success_queued=self._manual_success_queued,
                service_ready=success_ready,
            )
        )
        if self._auto_success:
            success_button_text = "MARK SUCCESS unavailable in AUTO mode"
        elif self._manual_success_future is not None:
            success_button_text = "MARK SUCCESS request pending..."
        elif self._manual_success_queued:
            success_button_text = "SUCCESS QUEUED (current episode)"
        elif status is None or status.get("state") not in {
            "POLICY_RUNNING",
            "HUMAN_INTERVENTION",
            "HOLD",
        }:
            success_button_text = "MARK SUCCESS (waiting for active episode)"
        elif not success_ready:
            success_button_text = "MARK SUCCESS (waiting for service...)"
        else:
            success_button_text = "MARK SUCCESS (current episode)"
        self._manual_success_button.setText(success_button_text)
        self._success_result.setText(self._success_result_text)
        self._success_result.setStyleSheet(
            f"color: {self._success_result_color}; font-weight: bold;"
        )

        state = None if status is None else status.get("state")
        service_ready = self._node.scene_ready_service_ready()
        start_enabled = scene_ready_enabled(
            status,
            request_state=self._scene_request_state,
            service_ready=service_ready,
        )
        self._scene_ready_button.setEnabled(start_enabled)
        if self._scene_request_state == _SCENE_RELEASE_WAIT:
            button_text = "Releasing deadman..."
        elif self._scene_request_state == _SCENE_CALL_PENDING:
            button_text = "Scene-ready request pending..."
        elif self._scene_request_state == _SCENE_ACCEPTED_WAIT:
            button_text = "Accepted — waiting for POLICY_RUNNING..."
        elif state == "WAIT_SCENE_READY" and not service_ready:
            button_text = "START / NEXT ITERATION (waiting for service...)"
        elif state == "WAIT_HOME_APPROVAL" and not service_ready:
            button_text = "APPROVE HOME (waiting for service...)"
        elif state == "WAIT_HOME_APPROVAL":
            button_text = "APPROVE HOME — ROBOT WILL MOVE"
        else:
            button_text = "START / NEXT ITERATION (policy)"
        self._scene_ready_button.setText(button_text)
        self._scene_result.setText(self._scene_result_text)
        self._scene_result.setStyleSheet(
            f"color: {self._scene_result_color}; font-weight: bold;"
        )

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
        self._closing = True
        self._cancel_future(getattr(self, "_scene_future", None))
        self._cancel_future(getattr(self, "_mode_future", None))
        self._cancel_future(getattr(self, "_manual_success_future", None))
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
    window.resize(900, 760)
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
