#!/usr/bin/env python3
"""Mouse-style PyQt5 operator GUI for GELLO -> UR7e EEF (Cartesian delta) teleop.

The whole window behaves like a computer MOUSE for the leader arm:

    click/hold (ENGAGE) = press the mouse button -> the robot TCP starts
                          following the GELLO end-effector delta,
    move                = the arm follows the leader EEF delta (translation
                          scaled by pos_scale; rotation is 1:1, never scaled),
    release (DISENGAGE) = lift the mouse button -> the arm HOLDS its pose and
                          the leader is free to be repositioned,
    lift + replace + click = pick the "mouse" up, put it down somewhere comfy,
                          re-engage from a fresh anchor (RECLUTCH while engaged,
                          or RE-ARM -> ENGAGE after a release),
    DPI slider          = pos_scale (sensitivity / gain k), 0.1 fine .. 1.0 1:1.

ARCHITECTURE (copied from the proven gello_recorder_gui pattern, NOT invented):
  * :class:`EefGuiNode` (an rclpy Node) owns ALL ROS I/O -- the ~/eef/state
    subscription and every service / set-parameter call. It is spun on a
    background daemon thread with a plain ``rclpy.spin``. Every service call is
    ``call_async`` + ``add_done_callback`` under a lock with a ``_pending``
    guard -- NEVER ``spin_until_future_complete`` on the Qt thread.
  * :class:`MainWindow` (Qt, main thread) polls the node's thread-safe getters
    from ~10 Hz QTimers and drives the widgets. Qt widgets are touched ONLY from
    the Qt/main thread.

A안 = COMMIT ON ENGAGE: the sensitivity slider edits a PENDING pos_scale. It is
pushed to the backend (SetParameters) ONLY at anchor-reset moments -- right
before an engage (then engage fires on the set-param success) and on a disengage
so the next HOLD reflects it. pos_scale is NEVER pushed while ENGAGED (that would
step p_des mid-stroke). The ACTIVE value shown comes from the ~/eef/state topic.

console_script entry point: ``gello_eef_gui = ur_gello_bringup.gello_eef_gui_node:main``
"""

import json
import os
import signal
import sys
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node

from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from std_msgs.msg import String
from std_srvs.srv import Trigger

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

BRIDGE = "/gello_ur_bridge"
GRIPPER = "/gello_gripper_bridge"

# A ~/eef/state reading older than this (seconds) is treated as unknown.
_STATE_STALE_S = 2.0

# Per-state indicator colour + one-line meaning. Keys are the ~/eef/state
# "state" field. Anything unknown/missing -> gray "no signal".
_STATE_STYLE = {
    "HOLD": ("#dd8800", "HOLD — armed, arm frozen"),
    "ENGAGED": ("#22aa22", "ENGAGED — tracking"),
    "DISENGAGED": ("#888888", "DISENGAGED — clutched up, arm holds"),
    "JOINT_BOOTSTRAP": ("#3366cc", "JOINT PASSTHROUGH — arm mirrors leader"),
}
_GRAY = "#888888"
_RED = "#cc3333"
_GREEN = "#22aa22"

# Button chrome: a filled, bordered, rounded, hover-reactive control that
# reads unmistakably as "clickable" -- so it is NOT confused with the flat,
# square, full-width STATUS BANNERS (the H1 and the state label, which stay
# borderless QLabels). Action is also colour-coded: ENGAGE green (go),
# DISENGAGE blue (stop/hold), armed-confirm orange, gripper neutral slate.
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
_PRIMARY_DISENGAGE = _btn_css("#1565c0", "#0d47a1", big=True)   # blue = stop/hold
_PRIMARY_ARMED = _btn_css("#ef6c00", "#e65100", big=True)       # orange = "click again"
_BTN = _btn_css("#546e7a", "#37474f")                          # neutral secondary
_BTN_ARMED = _btn_css("#ef6c00", "#e65100")                    # secondary confirm
_GRIP_PAUSE_HAZARD = _btn_css("#cc3333", "#a02020")            # red PAUSE (H2 active)


# --------------------------------------------------------------------------- #
# Node: owns ALL ROS I/O                                                       #
# --------------------------------------------------------------------------- #
class EefGuiNode(Node):
    """rclpy Node backing the EEF mouse GUI. Thread-safe getters + async calls."""

    def __init__(self, node_name: str = "gello_eef_gui_node") -> None:
        super().__init__(node_name)

        self._lock = threading.Lock()

        # Latest ~/eef/state JSON (parsed) + monotonic receipt time.
        self._state: Optional[dict] = None
        self._state_t: Optional[float] = None

        # In-flight service bookkeeping (a single _pending guard is enough for a
        # solo-operator GUI: one deliberate action at a time).
        self._pending = False
        self._last_ok: Optional[bool] = None
        self._last_msg = ""
        # SEPARATE in-flight guard for the gripper Trigger calls, so a bridge
        # motion RPC in flight (engage/disengage/...) can never block the
        # safety-critical Gripper PAUSE (hazard H2). Kept apart from _pending.
        self._grip_pending = False

        # Best-effort read-only v_max / w_max (H5: pendant slider has NO effect,
        # these are the real speed caps; not published on the state topic).
        self._v_max: Optional[float] = None
        self._w_max: Optional[float] = None
        self._vw_pending = False

        # CRITICAL: name this _svc, NOT _clients -- rclpy.Node uses self._clients
        # internally and shadowing it corrupts the executor / destroy_node.
        self._svc = {
            "engage": self.create_client(Trigger, f"{BRIDGE}/eef_engage"),
            "disengage": self.create_client(Trigger, f"{BRIDGE}/eef_disengage"),
            "reclutch": self.create_client(Trigger, f"{BRIDGE}/eef_reclutch"),
            "resume": self.create_client(Trigger, f"{BRIDGE}/eef_resume"),
            "to_joint": self.create_client(Trigger, f"{BRIDGE}/eef_to_joint"),
            "grip_pause": self.create_client(Trigger, f"{GRIPPER}/pause"),
            "grip_resume": self.create_client(Trigger, f"{GRIPPER}/resume"),
        }
        self._set_params = self.create_client(
            SetParameters, f"{BRIDGE}/set_parameters")
        self._get_params = self.create_client(
            GetParameters, f"{BRIDGE}/get_parameters")

        self.create_subscription(String, f"{BRIDGE}/eef/state", self._on_state, 10)

        self.get_logger().info("gello_eef_gui_node up; listening on %s/eef/state"
                               % BRIDGE)

    # ------------------------------------------------------------- ROS in ---
    def _on_state(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            self._state = d
            self._state_t = time.monotonic()

    # --------------------------------------------------- thread-safe getters
    def get_snapshot(self) -> dict:
        """Copy of the panel state for the Qt polling timer (thread-safe)."""
        now = time.monotonic()
        with self._lock:
            age = None if self._state_t is None else now - self._state_t
            state = dict(self._state) if self._state is not None else None
            return {
                "state": state,
                "age_s": age,
                "pending": self._pending,
                "grip_pending": self._grip_pending,
                "last_ok": self._last_ok,
                "last_msg": self._last_msg,
                "v_max": self._v_max,
                "w_max": self._w_max,
            }

    def current_state_name(self) -> Optional[str]:
        """The ~/eef/state "state" string, or None if stale/absent."""
        now = time.monotonic()
        with self._lock:
            if (self._state is None or self._state_t is None
                    or now - self._state_t > _STATE_STALE_S):
                return None
            return self._state.get("state")

    # Test hook: inject a fake ~/eef/state dict (no ROS traffic needed).
    def set_state_for_test(self, d: dict) -> None:
        with self._lock:
            self._state = dict(d)
            self._state_t = time.monotonic()

    # ------------------------------------------------------ service firing --
    def _begin(self) -> bool:
        with self._lock:
            if self._pending:
                return False
            self._pending = True
            return True

    def _finish(self, ok: bool, msg: str) -> None:
        with self._lock:
            self._pending = False
            self._last_ok = ok
            self._last_msg = msg

    def _fire(self, name: str) -> bool:
        """Fire a single Trigger service, non-blocking. Returns False if a
        request is already in flight or the service is unavailable."""
        cli = self._svc[name]
        if not cli.service_is_ready():
            self._finish(False, f"{name}: service unavailable")
            return False
        if not self._begin():
            return False
        fut = cli.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self._trigger_done(name, f))
        return True

    def _trigger_done(self, name: str, future) -> None:
        ok, msg = False, ""
        try:
            resp = future.result()
            if resp is not None:
                ok = bool(resp.success)
                msg = f"{name}: {resp.message}"
            else:
                msg = f"{name}: no response"
        except Exception as exc:  # noqa: BLE001 -- surface, never crash spin
            msg = f"{name}: {exc}"
        self._finish(ok, msg)

    # -- direct single-service actions --------------------------------------
    def request_disengage(self, pos_scale: Optional[float] = None) -> bool:
        """DISENGAGE (ENGAGED -> paused, arm holds).

        If ``pos_scale`` is given, a SetParameters(pos_scale) push is CHAINED in
        the disengage done-callback, AFTER the disengage succeeds (A안
        disengage-commit: 'push pos_scale on a disengage so the next HOLD
        reflects the slider'). It must NOT be fired synchronously next to this
        call: that second call would hit the shared ``_pending`` guard held by
        this in-flight disengage and be a guaranteed silent no-op, and it would
        also race an un-processed disengage into a mid-stroke ``p_des`` step now
        that the backend applies pos_scale LIVE. Chaining it on success avoids
        both: the guard is held across the chain, and the controller is already
        holding (DISENGAGED/paused) before pos_scale moves."""
        cli = self._svc["disengage"]
        if not cli.service_is_ready():
            self._finish(False, "disengage: service unavailable")
            return False
        if not self._begin():
            return False
        fut = cli.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self._disengage_done(f, pos_scale))
        return True

    def _disengage_done(self, future, pos_scale) -> None:
        ok, msg = False, ""
        try:
            resp = future.result()
            if resp is not None:
                ok = bool(resp.success)
                msg = f"disengage: {resp.message}"
            else:
                msg = "disengage: no response"
        except Exception as exc:  # noqa: BLE001 -- surface, never crash spin
            msg = f"disengage: {exc}"
        # No follow-up commit, a failed disengage, or no param service -> done.
        if (not ok) or pos_scale is None \
                or not self._set_params.service_is_ready():
            self._finish(ok, msg)
            return
        # Disengage confirmed (arm holding, paused) -> NOW push pos_scale so the
        # next HOLD reflects the slider. Keep _pending held across the chain.
        req = _double_set_request("pos_scale", float(pos_scale))
        sf = self._set_params.call_async(req)
        sf.add_done_callback(
            lambda g: self._finish(*self._read_set_result(g, "pos_scale")))

    def request_reclutch(self) -> bool:
        return self._fire("reclutch")

    def request_resume(self) -> bool:
        return self._fire("resume")

    def request_to_joint(self) -> bool:
        return self._fire("to_joint")

    def request_grip_pause(self) -> bool:
        return self._grip_fire("grip_pause")

    def request_grip_resume(self) -> bool:
        return self._grip_fire("grip_resume")

    def _grip_fire(self, name: str) -> bool:
        """Fire a GRIPPER Trigger on its OWN in-flight guard (self._grip_pending),
        independent of the bridge-motion _pending guard, so an in-flight
        engage/disengage never disables Gripper PAUSE (hazard H2: the operator
        must be able to stop the gripper the instant teleop leaves ENGAGED)."""
        cli = self._svc[name]
        if not cli.service_is_ready():
            with self._lock:
                self._last_ok = False
                self._last_msg = f"{name}: service unavailable"
            return False
        with self._lock:
            if self._grip_pending:
                return False
            self._grip_pending = True
        fut = cli.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self._grip_done(name, f))
        return True

    def _grip_done(self, name: str, future) -> None:
        ok, msg = False, ""
        try:
            resp = future.result()
            if resp is not None:
                ok, msg = bool(resp.success), f"{name}: {resp.message}"
            else:
                msg = f"{name}: no response"
        except Exception as exc:  # noqa: BLE001 -- surface, never crash spin
            msg = f"{name}: {exc}"
        with self._lock:
            self._grip_pending = False
            self._last_ok = ok
            self._last_msg = msg

    # -- A안 commit-on-engage: push pos_scale, THEN engage on its success -----
    def request_commit_and_engage(self, pos_scale: float) -> bool:
        """HOLD -> ENGAGED in one action: push the pending pos_scale
        (SetParameters, non-blocking) and, ONLY on its success, chain
        eef_engage, so the anchor is latched AFTER pos_scale committed and the
        very first p_des of the stroke already uses the new gain."""
        if not self._begin():
            return False
        self._commit_then_engage(float(pos_scale))
        return True

    def request_reengage(self, pos_scale: float) -> bool:
        """DISENGAGED/paused -> ENGAGED in ONE operator action (the 'off/on = a
        new reference' model): chain eef_resume (re-seed to HOLD, zero jump) ->
        the pos_scale commit -> eef_engage, holding the single _pending guard
        across the whole chain. A bare engage would be refused here (gate G0
        blocks engage while the bridge is paused after a disengage). If resume
        succeeds but engage's gates refuse (e.g. the leader was not held still),
        the bridge is left safely in HOLD and the next toggle click engages."""
        cli = self._svc["resume"]
        if not cli.service_is_ready():
            self._finish(False, "resume: service unavailable")
            return False
        if not self._begin():
            return False
        fut = cli.call_async(Trigger.Request())

        def _after_resume(f):
            ok, msg = False, ""
            try:
                resp = f.result()
                if resp is not None:
                    ok, msg = bool(resp.success), f"resume: {resp.message}"
                else:
                    msg = "resume: no response"
            except Exception as exc:  # noqa: BLE001 -- surface, never crash spin
                msg = f"resume: {exc}"
            if not ok:
                self._finish(False, msg)
                return
            # Resumed (HOLD, zero jump) -> commit pos_scale, then engage.
            self._commit_then_engage(float(pos_scale))

        fut.add_done_callback(_after_resume)
        return True

    def _commit_then_engage(self, pos_scale: float) -> None:
        """Push pos_scale (if the param service is up), then engage on success.
        The caller MUST already hold the _pending guard; this keeps it held
        across the chain and releases it via _finish / _trigger_done at the
        end. Shared by request_commit_and_engage and request_reengage."""
        def _engage_now():
            eng = self._svc["engage"]
            if not eng.service_is_ready():
                self._finish(False, "engage: service unavailable")
                return
            ef = eng.call_async(Trigger.Request())
            ef.add_done_callback(lambda g: self._trigger_done("engage", g))

        if not self._set_params.service_is_ready():
            # No param service: bare engage so the operator is never blocked
            # (pos_scale simply stays at whatever the node already has).
            _engage_now()
            return
        req = _double_set_request("pos_scale", float(pos_scale))
        fut = self._set_params.call_async(req)

        def _after_set(f):
            ok, msg = self._read_set_result(f, "pos_scale")
            if not ok:
                self._finish(False, msg)
                return
            _engage_now()

        fut.add_done_callback(_after_set)

    def push_pos_scale(self, pos_scale: float) -> bool:
        """Push pos_scale with NO chained engage (used on disengage so the next
        HOLD reflects the slider). Non-blocking; refuses while a request is in
        flight."""
        if not self._set_params.service_is_ready():
            return False
        if not self._begin():
            return False
        req = _double_set_request("pos_scale", float(pos_scale))
        fut = self._set_params.call_async(req)
        fut.add_done_callback(
            lambda f: self._finish(*self._read_set_result(f, "pos_scale")))
        return True

    def _read_set_result(self, future, label: str):
        try:
            resp = future.result()
            results = getattr(resp, "results", None)
            if results and all(r.successful for r in results):
                return True, f"{label}: applied"
            reason = ""
            if results:
                reason = "; ".join(r.reason for r in results if r.reason)
            return False, f"{label}: rejected {reason}".strip()
        except Exception as exc:  # noqa: BLE001
            return False, f"{label}: {exc}"

    # -- best-effort read-only v_max / w_max fetch (H5) ---------------------
    def refresh_vw(self) -> None:
        if not rclpy.ok() or not self._get_params.service_is_ready():
            return
        # test-and-set under the lock, mirroring _begin()/_finish(); _vw_done()
        # clears the flag under the same lock.
        with self._lock:
            if self._vw_pending:
                return
            self._vw_pending = True
        req = GetParameters.Request()
        req.names = ["v_max", "w_max"]
        fut = self._get_params.call_async(req)
        fut.add_done_callback(self._vw_done)

    def _vw_done(self, future) -> None:
        v = w = None
        try:
            resp = future.result()
            vals = getattr(resp, "values", None)
            if vals and len(vals) >= 2:
                v = float(vals[0].double_value)
                w = float(vals[1].double_value)
        except Exception:  # noqa: BLE001 -- best-effort, never crash
            pass
        with self._lock:
            if v is not None:
                self._v_max = v
            if w is not None:
                self._w_max = w
            self._vw_pending = False


def _double_set_request(name: str, value: float) -> SetParameters.Request:
    req = SetParameters.Request()
    p = ParameterMsg()
    p.name = name
    p.value = ParameterValue()
    p.value.type = ParameterType.PARAMETER_DOUBLE
    p.value.double_value = float(value)
    req.parameters = [p]
    return req


# --------------------------------------------------------------------------- #
# Main window (Qt / main thread only)                                          #
# --------------------------------------------------------------------------- #
# Live readout rows: (state-topic key, human label, extra hint).
_READOUT_FIELDS = [
    ("excursion_m", "excursion_m", "per-segment, resets each clutch"),
    ("pos_scale", "pos_scale (ACTIVE)", "translation gain; rotation is 1:1"),
    ("sigma_min", "sigma_min", "manipulability"),
    ("gamma", "gamma", "governor throttle"),
    ("joint_gap", "joint_gap", "normal to be large in EEF"),
    ("reject_reason", "reject_reason", "why the last tick HELD"),
    ("branch_id", "branch_id", "IK branch lock"),
    ("paused", "paused", "bridge output frozen"),
]


class MainWindow(QMainWindow):
    def __init__(self, node: EefGuiNode):
        super().__init__()
        self._node = node
        # Two-click confirm gates, keyed by action id. A gate is 'armed' after
        # the first click and fires on the second (mouse metaphor: no dialogs).
        self._armed = {}
        self._last_shown_msg = None
        self._last_left_engaged_prompted = False
        self._prev_state = None

        self.setWindowTitle("GELLO -> UR7e  EEF mouse")
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_eef)
        self._timer.start(100)  # ~10 Hz

        self._vw_timer = QTimer(self)
        self._vw_timer.timeout.connect(self._node.refresh_vw)
        self._vw_timer.start(3000)  # best-effort v_max/w_max poll

        self._refresh_eef()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        # Breathing room so the flat status banners and the rounded buttons
        # never abut (part of the banner-vs-button visual separation).
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

        # --- big state indicator -------------------------------------------
        self._state_label = QLabel("no signal")
        self._state_label.setAlignment(Qt.AlignCenter)
        self._state_label.setStyleSheet(
            "background-color: #888888; color: white; font-weight: bold; "
            "font-size: 20pt; padding: 18px;")
        root.addWidget(self._state_label)

        # --- primary mouse-button toggle -----------------------------------
        self._primary = QPushButton("ENGAGE")
        self._primary.setStyleSheet(_PRIMARY_ENGAGE)
        self._primary.clicked.connect(self._on_primary)
        root.addWidget(self._primary)

        # No reclutch / re-arm / to-joint buttons: the big toggle IS the whole
        # cycle. Each ENGAGE re-anchors at the current pose (a fresh reference),
        # so "off -> move GELLO -> on" is the only workflow needed; from a paused
        # state the toggle chains resume->engage internally. Leaving EEF mode
        # entirely (joint passthrough) is an advanced hand-back done from the
        # operator console, not this mouse GUI.
        self._rearm_reason = QLabel("")
        self._rearm_reason.setStyleSheet("color: #aa6600;")
        root.addWidget(self._rearm_reason)

        # --- gripper row (H2) ----------------------------------------------
        grow = QHBoxLayout()
        self._grip_hint = QLabel("Gripper:")
        self._grip_hint.setStyleSheet("font-weight: bold;")
        self._grip_pause_btn = QPushButton("Gripper PAUSE")
        self._grip_pause_btn.setStyleSheet(_BTN)
        self._grip_pause_btn.clicked.connect(self._on_grip_pause)
        self._grip_resume_btn = QPushButton("Gripper Resume")
        self._grip_resume_btn.setStyleSheet(_BTN)
        self._grip_resume_btn.clicked.connect(self._on_grip_resume)
        grow.addWidget(self._grip_hint)
        grow.addWidget(self._grip_pause_btn)
        grow.addWidget(self._grip_resume_btn)
        grow.addStretch(1)
        root.addLayout(grow)

        # --- sensitivity (DPI) slider = pos_scale (A안) --------------------
        root.addWidget(self._build_slider_box())

        # --- live readout + freshness lamps --------------------------------
        root.addWidget(self._build_readout_box(), stretch=1)

        # --- H5 static note + read-only v_max/w_max ------------------------
        self._h5 = QLabel(
            "H5: the pendant SPEED SLIDER has NO effect on EEF speed — the real "
            "caps are v_max / w_max (below), retune via v_max:= / w_max:=.")
        self._h5.setWordWrap(True)
        self._h5.setStyleSheet("color: #666666; font-style: italic;")
        root.addWidget(self._h5)

    def _build_slider_box(self):
        box = QGroupBox("Sensitivity — pos_scale (DPI). Rotation is 1:1 (NOT scaled)")
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

    def _build_readout_box(self):
        box = QGroupBox("Live ~/eef/state")
        grid = QGridLayout(box)

        # Freshness lamps (row 0).
        self._lamp_state = QLabel("state topic: --")
        self._lamp_leader = QLabel("leader: --")
        self._lamp_arm = QLabel("arm: --")
        grid.addWidget(self._lamp_state, 0, 0)
        grid.addWidget(self._lamp_leader, 0, 1)
        grid.addWidget(self._lamp_arm, 0, 2)

        self._readout = {}
        r = 1
        for key, human, hint in _READOUT_FIELDS:
            name = QLabel(f"{human}:")
            name.setMinimumWidth(150)
            val = QLabel("—")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            hintlbl = QLabel(hint)
            hintlbl.setStyleSheet("color: #888888; font-size: 9pt;")
            grid.addWidget(name, r, 0)
            grid.addWidget(val, r, 1)
            grid.addWidget(hintlbl, r, 2)
            self._readout[key] = val
            r += 1

        self._auto_reason = QLabel("auto_reason: —")
        self._auto_reason.setWordWrap(True)
        grid.addWidget(self._auto_reason, r, 0, 1, 3)
        r += 1
        self._vw_label = QLabel("v_max / w_max: — (read-only; H5)")
        grid.addWidget(self._vw_label, r, 0, 1, 3)
        return box

    # -------------------------------------------------------- slider (A안) --
    def _slider_value(self) -> float:
        return self._slider.value() / 100.0

    def _on_slider(self, _v):
        self._update_slider_label()

    def _update_slider_label(self):
        pending = self._slider_value()
        st = self._node.current_state_name()
        if st == "ENGAGED":
            self._slider_label.setText(
                f"pending {pending:.2f} -> applies on next engage "
                f"(NEVER pushed mid-stroke)")
            self._slider_label.setStyleSheet(
                "font-weight: bold; color: #aa6600;")
        else:
            self._slider_label.setText(
                f"pending {pending:.2f} — commits on the next engage/disengage")
            self._slider_label.setStyleSheet("font-weight: bold;")

    # ---------------------------------------------------- two-click confirm --
    def _armed_click(self, key, action, warn="the robot WILL move"):
        """First click arms (orange 'click again', 3 s window); second fires."""
        if self._armed.get(key):
            self._armed[key] = False
            action()
            self._refresh_eef()
            return
        self._armed[key] = True
        self.statusBar().showMessage(
            f"Click AGAIN within 3 s — {warn}.", 3000)
        QTimer.singleShot(3000, lambda: self._disarm(key))
        self._refresh_eef()

    def _disarm(self, key):
        if self._armed.get(key):
            self._armed[key] = False
            self._refresh_eef()

    # ------------------------------------------------------------- actions --
    def _on_primary(self):
        st = self._node.current_state_name()
        if st == "ENGAGED":
            # DISENGAGE = safe (arm holds). Single click. The pending pos_scale is
            # CHAINED into the disengage done-callback (committed only AFTER the
            # disengage succeeds) so the NEXT reference reflects the slider
            # without a mid-stroke p_des step (A안). It must NOT be pushed
            # synchronously here: the shared _pending guard held by the in-flight
            # disengage would make that push a silent no-op.
            if self._node.request_disengage(self._slider_value()):
                self.statusBar().showMessage(
                    "Disengage requested (pos_scale commits on success).", 3000)
            else:
                self._busy_msg()
        elif st in ("HOLD", "DISENGAGED", "JOINT_BOOTSTRAP"):
            # ENGAGE = start motion -> two-click confirm. From HOLD it is a bare
            # commit+engage; from DISENGAGED/JOINT_BOOTSTRAP the node chains
            # resume->commit->engage (the bridge is paused after a disengage).
            self._armed_click(
                "primary",
                self._do_engage,
                warn="ENGAGE — the arm will start following the leader")
        # else (no ~/eef/state signal): primary is disabled; nothing to do.

    def _do_engage(self):
        # Re-read the state at FIRE time (the 2nd click can land up to 3 s after
        # arming, and the state may have moved).
        st = self._node.current_state_name()
        scale = self._slider_value()
        if st == "HOLD":
            ok = self._node.request_commit_and_engage(scale)
            msg = f"Committing pos_scale={scale:.2f}, engaging..."
        elif st in ("DISENGAGED", "JOINT_BOOTSTRAP"):
            ok = self._node.request_reengage(scale)
            msg = f"Re-arming + committing pos_scale={scale:.2f}, engaging..."
        else:
            self.statusBar().showMessage(
                f"Not engaging — state is now {st}.", 3000)
            return
        if ok:
            self.statusBar().showMessage(msg, 4000)
        else:
            self._busy_msg()

    def _on_grip_pause(self):
        self._fire_and_report(self._node.request_grip_pause,
                              "Gripper pause requested.")

    def _on_grip_resume(self):
        self._armed_click(
            "grip_resume",
            lambda: self._fire_and_report(
                self._node.request_grip_resume, "Gripper resume requested."),
            warn="GRIPPER RESUME — the gripper will seed/ramp to the leader")

    def _fire_and_report(self, fn, ok_msg):
        if fn():
            self.statusBar().showMessage(ok_msg, 3000)
        else:
            self._busy_msg()

    def _busy_msg(self):
        self.statusBar().showMessage(
            "Not sent — a request is in flight or the service is down.", 3000)

    # -------------------------------------------------------------- refresh --
    def _refresh_eef(self):
        snap = self._node.get_snapshot()
        state = snap["state"]
        age = snap["age_s"]
        pending = snap["pending"]
        grip_pending = snap["grip_pending"]
        live = (state is not None and age is not None and age < _STATE_STALE_S)
        st = state.get("state") if (live and state is not None) else None

        self._apply_state_indicator(st, live)
        self._apply_buttons(st, pending, grip_pending)
        self._apply_readout(state, live, age, snap)
        self._apply_gripper_hazard(st)
        self._update_slider_label()

        # Surface the latest service reply once.
        msg = snap["last_msg"]
        if msg and msg != self._last_shown_msg:
            self._last_shown_msg = msg
            self.statusBar().showMessage(msg, 6000)

    def _apply_state_indicator(self, st, live):
        if st in _STATE_STYLE and live:
            color, text = _STATE_STYLE[st]
        else:
            color, text = _GRAY, "no signal — ~/eef/state stale or absent"
        self._state_label.setText(text)
        self._state_label.setStyleSheet(
            f"background-color: {color}; color: white; font-weight: bold; "
            "font-size: 20pt; padding: 18px;")

    def _apply_buttons(self, st, pending, grip_pending=False):
        # The big toggle is the WHOLE mouse button: click = off/on, and every
        # ENGAGE re-anchors at the current pose (a fresh reference), so no
        # separate reclutch/re-arm is needed. In DISENGAGED/JOINT_BOOTSTRAP the
        # click chains resume->engage internally (the bridge is paused after a
        # disengage, so a bare engage would be refused by gate G0).
        can_engage = st in ("HOLD", "DISENGAGED", "JOINT_BOOTSTRAP")
        if st == "ENGAGED":
            self._primary.setText("DISENGAGE (clutch up)")
            self._primary.setEnabled(not pending)
            primary_style = _PRIMARY_DISENGAGE          # blue = stop/hold
        elif st == "HOLD":
            self._primary.setText("ENGAGE  (press to start following)")
            self._primary.setEnabled(not pending)
            primary_style = _PRIMARY_ENGAGE             # green = go
        elif st in ("DISENGAGED", "JOINT_BOOTSTRAP"):
            self._primary.setText("ENGAGE  (new reference here)")
            self._primary.setEnabled(not pending)
            primary_style = _PRIMARY_ENGAGE
        else:
            self._primary.setText("ENGAGE — waiting for ~/eef/state")
            self._primary.setEnabled(False)
            primary_style = _PRIMARY_ENGAGE
        # Keep the armed (orange) look while the confirm window is open, but only
        # while the state still permits engaging; otherwise clear the arm and
        # fall back to the state-based colour/label set above.
        if self._armed.get("primary") and can_engage:
            self._primary.setText("Click AGAIN to ENGAGE (robot WILL move)")
            self._primary.setStyleSheet(_PRIMARY_ARMED)
            self._primary.setEnabled(True)
        else:
            self._armed["primary"] = False
            self._primary.setStyleSheet(primary_style)

        # Gripper buttons ride their OWN guard (grip_pending), NOT the bridge's
        # _pending -- so an in-flight engage/disengage never disables Gripper
        # PAUSE, the H2 safety action when teleop leaves ENGAGED.
        self._grip_pause_btn.setEnabled(not grip_pending)
        self._style_confirmable(
            self._grip_resume_btn, "grip_resume",
            enabled=not grip_pending,
            resting="Gripper Resume", armed="Click AGAIN to resume gripper")

    def _style_confirmable(self, btn, key, enabled, resting, armed):
        if self._armed.get(key):
            btn.setText(armed)
            btn.setStyleSheet(_BTN_ARMED)
            btn.setEnabled(True)
        else:
            btn.setText(resting)
            btn.setStyleSheet(_BTN)
            btn.setEnabled(enabled)

    def _apply_readout(self, state, live, age, snap):
        for key, val in self._readout.items():
            if state is None or not live:
                val.setText("—")
                continue
            v = state.get(key)
            val.setText(_fmt(v))

        auto = state.get("auto_reason") if (state and live) else None
        self._auto_reason.setText(f"auto_reason: {auto if auto else '—'}")
        # RE-ARM helper: echo the auto-disengage reason next to it.
        self._rearm_reason.setText(
            f"last auto-disengage: {auto}" if auto else "")

        # Freshness lamps.
        self._set_lamp(self._lamp_state, "state topic",
                       ok=(live), age=age)
        # Leader / arm freshness inferred from auto_reason (no direct field).
        leader_bad = bool(auto and "leader" in str(auto).lower())
        arm_bad = bool(auto and ("robot" in str(auto).lower()
                                 or "arm" in str(auto).lower()))
        self._set_lamp(self._lamp_leader, "leader",
                       ok=(live and not leader_bad), age=age, bad=leader_bad)
        self._set_lamp(self._lamp_arm, "arm",
                       ok=(live and not arm_bad), age=age, bad=arm_bad)

        # Read-only v_max / w_max (H5).
        v, w = snap["v_max"], snap["w_max"]
        if v is not None and w is not None:
            self._vw_label.setText(
                f"v_max={v:.4f} m/s  w_max={w:.4f} rad/s  (read-only; H5)")
        else:
            self._vw_label.setText("v_max / w_max: — (read-only; H5)")

    def _set_lamp(self, label, name, ok, age, bad=False):
        if age is None:
            label.setText(f"{name}: no signal")
            label.setStyleSheet(f"color: {_RED}; font-weight: bold;")
            return
        if bad:
            label.setText(f"{name}: STALE")
            label.setStyleSheet(f"color: {_RED}; font-weight: bold;")
        elif ok:
            label.setText(f"{name}: fresh ({age:.2f}s)")
            label.setStyleSheet(f"color: {_GREEN}; font-weight: bold;")
        else:
            label.setText(f"{name}: stale ({age:.2f}s)")
            label.setStyleSheet(f"color: {_RED}; font-weight: bold;")

    def _apply_gripper_hazard(self, st):
        # H2: the moment the state LEAVES ENGAGED, warn + highlight Gripper Pause.
        left_engaged = (self._prev_state == "ENGAGED" and st != "ENGAGED")
        if left_engaged and not self._last_left_engaged_prompted:
            self._last_left_engaged_prompted = True
            self.statusBar().showMessage(
                "H2: gripper STILL tracks GELLO — PAUSE the gripper before "
                "repositioning the leader.", 8000)
        if st == "ENGAGED":
            self._last_left_engaged_prompted = False

        highlight = (st is not None and st != "ENGAGED")
        if highlight:
            self._grip_pause_btn.setStyleSheet(_GRIP_PAUSE_HAZARD)
            self._grip_hint.setText(
                "Gripper (H2: still tracking — PAUSE before repositioning):")
        else:
            self._grip_pause_btn.setStyleSheet(_BTN)
            self._grip_hint.setText("Gripper:")
        self._prev_state = st

    def closeEvent(self, event):
        # Stop the QTimers BEFORE tearing down rclpy, so a late _vw_timer /
        # _timer tick cannot do ROS I/O against an already-shutdown context
        # (which would raise inside a Qt slot during teardown).
        for t in (getattr(self, "_timer", None), getattr(self, "_vw_timer", None)):
            if t is not None:
                t.stop()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
        event.accept()


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:+.4f}"
    return str(v)


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

    node = EefGuiNode()

    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv if args is None else args)
    window = MainWindow(node)
    window.resize(760, 900)
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
