"""Headless tests for the EEF GUI's GO TO START POSE row.

Same pattern as gello_recorder/test/test_task_recorder_gui.py (and
test_hil_actor_status.py here): the window methods are called UNBOUND against
plain duck-typed stand-ins, so there is no Qt event loop, no QApplication, no
spinning rclpy node and no robot.  Only the operator-facing decision logic is
under test -- which is exactly the part that must not be found wrong while the
arm is moving.

What is pinned: the gating that keeps a click from starting a motion at the
wrong moment (no pose / move already active / stale ~/eef/state), the two-click
confirm, the single-click STOP and when it is offered, the lock on the primary
ENGAGE button while a move is in flight, the refusal to close the window out
from under a moving arm, the status text/colour mapping, and that a node with
no start pose answers UNAVAILABLE and refuses every request without touching
the home-move I/O.  Exact labels and stylesheet bytes are not pinned.

The module needs the ROS overlay (rclpy, PyQt5, gello_recorder) to import the
GUI module at all; it skips cleanly without one.
"""

import importlib
import importlib.util
import os
import re
import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy", reason="the GUI module imports rclpy")
pytest.importorskip("PyQt5.QtWidgets", reason="the GUI module imports PyQt5")

# gello_recorder.home_move_ros is a NEW module in the sibling package's src/.
# This workspace is a COPY install (symlink_install=False), so until the next
# `colcon build --packages-select gello_recorder` the overlay's gello_recorder
# lacks it and `import ur_gello_bringup.gello_eef_gui_node` would fail here
# even though the code is correct. Same problem test_start_pose.py solves for a
# single file; for a whole package the fix is to put the src/ package directory
# AHEAD of the overlay copy on gello_recorder's search path. A no-op once the
# overlay is current.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RECORDER_SRC = os.path.join(os.path.dirname(_PKG_DIR), "gello_recorder")
try:
    _have_ros_half = importlib.util.find_spec("gello_recorder.home_move_ros") is not None
except ModuleNotFoundError:  # no gello_recorder at all on sys.path
    _have_ros_half = False
if not _have_ros_half and os.path.isdir(os.path.join(_RECORDER_SRC, "gello_recorder")):
    pkg = sys.modules.get("gello_recorder")
    if pkg is not None and hasattr(pkg, "__path__"):
        pkg.__path__.insert(0, os.path.join(_RECORDER_SRC, "gello_recorder"))
    else:
        sys.path.insert(0, _RECORDER_SRC)

gui = pytest.importorskip(
    "ur_gello_bringup.gello_eef_gui_node",
    reason="gello_eef_gui_node needs rclpy + PyQt5 + gello_recorder on the path",
)
home_move = importlib.import_module("gello_recorder.home_move")

IDLE = home_move.HomeMoveState.IDLE
DONE = home_move.HomeMoveState.DONE
FAILED = home_move.HomeMoveState.FAILED
MOVING = home_move.HomeMoveState.MOVING
UNAVAILABLE = gui.START_POSE_UNAVAILABLE

CARROT = (-3.1638, -1.4900, 1.7258, -1.8455, -1.5793, -3.2692)


# --------------------------------------------------------------------------- #
# Colour helpers -- assert the MEANING of a stylesheet, not its exact bytes.
# --------------------------------------------------------------------------- #
def _style_colors(style):
    return [
        tuple(int(hexrgb[i:i + 2], 16) for i in (0, 2, 4))
        for hexrgb in re.findall(r"#([0-9a-fA-F]{6})", style or "")
    ]


def _has_red(style):
    return any(r > 2 * g and r > 2 * b for r, g, b in _style_colors(style))


def _has_orange(style):
    return any(r > g + 40 and g > b + 40 for r, g, b in _style_colors(style))


def _has_green(style):
    return any(g > r + 40 and g > b + 40 for r, g, b in _style_colors(style))


def _is_grey(style):
    return any(abs(r - g) < 8 and abs(g - b) < 8 for r, g, b in _style_colors(style))


# --------------------------------------------------------------------------- #
# Duck-typed stand-ins
# --------------------------------------------------------------------------- #
class _FakeWidget:
    """A QPushButton/QLabel as far as this GUI logic is concerned."""

    def __init__(self, text=""):
        self.text = text
        self.style = ""
        self.enabled = None
        self.tooltip = None

    def setText(self, text):
        self.text = text

    def setStyleSheet(self, style):
        self.style = style

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setToolTip(self, text):
        self.tooltip = text


class _FakeNode:
    """Stand-in for the node half of the contract (the three lock-protected
    methods + the state-name getter the window reads)."""

    def __init__(self, *, state=IDLE, active=False, message="",
                 eef_state="DISENGAGED", accept=True):
        self._status = {
            "state": state,
            "active": active,
            "message": message,
            "duration_s": None,
        }
        self._eef_state = eef_state
        self._accept = accept
        self.go_calls = 0
        self.stop_calls = 0

    def get_home_status(self):
        return dict(self._status)

    def request_go_home(self):
        self.go_calls += 1
        return self._accept

    def request_stop_home(self):
        # HomeMoveController.abort() semantics: False when nothing is in flight.
        self.stop_calls += 1
        return bool(self._status["active"])

    def current_state_name(self):
        return self._eef_state


class _Window:
    """Duck-typed stand-in for MainWindow's widgets and window state."""

    def __init__(self, node, live=True):
        self._node = node
        self._live = live
        self._armed = {}
        self._start_pose_btn = _FakeWidget()
        self._start_pose_stop_btn = _FakeWidget()
        self._start_pose_status = _FakeWidget()
        self._primary = _FakeWidget()
        self._grip_pause_btn = _FakeWidget()
        self._grip_resume_btn = _FakeWidget()
        self._start_pose_last_state = None
        self._start_pose_close_deadline = None
        self.messages = []

    def statusBar(self):
        return SimpleNamespace(showMessage=self._show_message)

    def _show_message(self, text, msec=0):
        self.messages.append(text)

    def feedback(self):
        return " | ".join(self.messages + [self._start_pose_status.text or ""])


def _window(monkeypatch, node, live=True, **fields):
    """A stand-in with the real row logic bound and QTimer captured.

    Returns ``(window, deferred)``; ``deferred`` collects every
    ``QTimer.singleShot(ms, callback)`` the implementation schedules.  The
    stand-in's ``_refresh_eef`` models what the real ~10 Hz refresh does to
    THIS row and to the primary button: it feeds the same status dict to
    ``_apply_start_pose`` and ``_apply_buttons(home_active=...)``.
    """
    deferred = []

    class _FakeQTimer:
        @staticmethod
        def singleShot(ms, callback):
            deferred.append((ms, callback))

    monkeypatch.setattr(gui, "QTimer", _FakeQTimer, raising=False)

    w = _Window(node, live=live)
    w.__dict__.update(fields)
    W = gui.MainWindow

    def _refresh():
        status = node.get_home_status()
        W._apply_buttons(
            w, node.current_state_name() if w._live else None, False, False,
            home_active=bool(status.get("active", False)),
        )
        W._apply_start_pose(w, status, w._live)

    w._refresh_eef = _refresh
    w._disarm = lambda key: W._disarm(w, key)
    w._armed_click = (
        lambda key, action, warn="the robot WILL move":
        W._armed_click(w, key, action, warn=warn))
    w._do_go_start_pose = lambda: W._do_go_start_pose(w)
    w._on_start_pose_stop = lambda: W._on_start_pose_stop(w)
    w._style_confirmable = (
        lambda btn, key, enabled, resting, armed:
        W._style_confirmable(w, btn, key, enabled, resting, armed))
    return w, deferred


def _click(w):
    gui.MainWindow._on_start_pose(w)


def _refresh(w):
    w._refresh_eef()


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #
def test_no_start_pose_means_a_disabled_button_that_says_why(monkeypatch):
    """Fail closed: no START_POSE_CONFIG -> nothing can be clicked, and the
    reason is on screen (a disabled button with no explanation is
    indistinguishable from a broken one)."""
    reason = "START_POSE_CONFIG is not set -- export it to a deploy yaml"
    node = _FakeNode(state=UNAVAILABLE, active=False, message=reason)
    w, _ = _window(monkeypatch, node)

    _refresh(w)

    assert w._start_pose_btn.enabled is False
    assert w._start_pose_stop_btn.enabled is False
    assert reason in w._start_pose_status.text
    assert not _has_red(w._start_pose_status.style)      # config, not a fault
    # And the primary button is NOT locked by an unavailable row.
    assert w._primary.enabled is True


def test_an_active_move_offers_stop_and_locks_engage(monkeypatch):
    """While the arm is being driven: STOP is the only live control here, GO
    is off, and the big ENGAGE toggle is locked -- after the PAUSING step the
    bridge reports DISENGAGED, which the window would otherwise offer to
    re-ENGAGE straight into a running trajectory."""
    node = _FakeNode(state=MOVING, active=True, message="moving to START POSE",
                     eef_state="DISENGAGED")
    w, _ = _window(monkeypatch, node)

    _refresh(w)

    assert w._start_pose_stop_btn.enabled is True
    assert w._start_pose_btn.enabled is False
    assert w._primary.enabled is False
    assert "START POSE" in w._primary.text            # says why
    assert w._grip_resume_btn.enabled is False
    # Gripper PAUSE stays live: idempotent, and the H2 safety action.
    assert w._grip_pause_btn.enabled is True
    assert _has_orange(w._start_pose_status.style)

    # Once the sequence ends, the lock lifts and DISENGAGED is engageable again.
    node._status.update({"state": DONE, "active": False})
    _refresh(w)
    assert w._primary.enabled is True
    assert w._start_pose_btn.enabled is True
    assert w._start_pose_stop_btn.enabled is False


def test_a_stale_eef_state_disables_go(monkeypatch):
    """No live ~/eef/state means no bridge to pause: say so before the click."""
    node = _FakeNode(state=IDLE, active=False, eef_state=None)
    w, _ = _window(monkeypatch, node, live=False)

    _refresh(w)
    assert w._start_pose_btn.enabled is False

    w._live = True
    node._eef_state = "DISENGAGED"
    _refresh(w)
    assert w._start_pose_btn.enabled is True


# --------------------------------------------------------------------------- #
# Two-click confirm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("accepted", [True, False])
def test_first_click_arms_second_fires_exactly_once(monkeypatch, accepted):
    node = _FakeNode(accept=accepted)
    w, deferred = _window(monkeypatch, node)
    _refresh(w)
    rest_text, rest_style = w._start_pose_btn.text, w._start_pose_btn.style

    _click(w)

    assert w._armed.get("start_pose") is True
    assert node.go_calls == 0                          # nothing requested yet
    assert w._start_pose_btn.enabled is True            # confirming click can land
    assert w._start_pose_btn.text != rest_text
    assert _has_orange(w._start_pose_btn.style)
    assert [ms for ms, _cb in deferred] == [3000]      # the confirm window
    assert any("WILL move" in m for m in w.messages)   # and it warns

    _click(w)

    assert node.go_calls == 1                          # exactly once
    assert w._armed.get("start_pose") is False
    assert w._start_pose_btn.text == rest_text
    assert w._start_pose_btn.style == rest_style
    assert w.feedback().strip() != ""


def test_the_scheduled_disarm_restores_the_resting_button(monkeypatch):
    node = _FakeNode()
    w, deferred = _window(monkeypatch, node)
    _refresh(w)
    rest_text, rest_style = w._start_pose_btn.text, w._start_pose_btn.style

    _click(w)
    deferred[0][1]()                                   # the 3 s timer fires

    assert w._armed.get("start_pose") is False
    assert w._start_pose_btn.text == rest_text
    assert w._start_pose_btn.style == rest_style
    assert node.go_calls == 0


def test_an_armed_button_is_dropped_when_the_gate_closes(monkeypatch):
    """The armed window bypasses the gate so the confirming click is not
    swallowed -- but only while the gate still holds."""
    node = _FakeNode()
    w, _ = _window(monkeypatch, node)
    _refresh(w)
    _click(w)
    assert w._armed.get("start_pose") is True

    node._status.update({"state": MOVING, "active": True})   # a move started
    _refresh(w)
    assert w._armed.get("start_pose") is False
    assert w._start_pose_btn.enabled is False


def test_the_confirming_click_rechecks_the_live_state_at_fire_time(monkeypatch):
    """The topic can go stale inside the 3 s window; the refresh's gate is
    bypassed for the armed button, so the fire path checks it itself."""
    node = _FakeNode()
    w, _ = _window(monkeypatch, node)
    _refresh(w)
    _click(w)
    node._eef_state = None                             # stale between the clicks
    _click(w)
    assert node.go_calls == 0
    assert any("stale" in m for m in w.messages)


# --------------------------------------------------------------------------- #
# STOP + refusing to close out from under a moving arm
# --------------------------------------------------------------------------- #
class _FakeCloseEvent:
    def __init__(self):
        self.ignored = False
        self.accepted = False

    def ignore(self):
        self.ignored = True

    def accept(self):
        self.accepted = True


def test_stop_is_single_click_and_only_offered_while_active(monkeypatch):
    node = _FakeNode(state=MOVING, active=True)
    w, deferred = _window(monkeypatch, node)

    assert gui.MainWindow._on_start_pose_stop(w) is True
    assert node.stop_calls == 1                        # first click, no confirm
    assert deferred == []
    assert w.feedback().strip() != ""

    node._status.update({"state": FAILED, "active": False})
    _refresh(w)
    assert w._start_pose_stop_btn.enabled is False
    # A late click after the sequence ended is reported, not silently eaten.
    assert gui.MainWindow._on_start_pose_stop(w) is False
    assert node.stop_calls == 2


def test_closing_mid_move_is_refused_and_turned_into_a_stop(monkeypatch):
    node = _FakeNode(state=MOVING, active=True)
    w, _ = _window(monkeypatch, node)
    event = _FakeCloseEvent()

    assert gui.MainWindow._veto_close_during_start_pose_move(w, event) is True
    assert event.ignored is True and event.accepted is False
    assert node.stop_calls == 1                        # the same abort path
    assert w.feedback().strip() != ""                  # and it says why

    # Once the sequence is terminal a close is an ordinary close.
    node._status.update({"state": FAILED, "active": False})
    event = _FakeCloseEvent()
    assert gui.MainWindow._veto_close_during_start_pose_move(w, event) is False
    assert event.ignored is False
    assert node.stop_calls == 1


def test_a_wedged_sequence_cannot_trap_the_operator(monkeypatch, capsys):
    node = _FakeNode(state=MOVING, active=True)
    w, _ = _window(monkeypatch, node,
                   _start_pose_close_deadline=time.monotonic() - 1.0)
    event = _FakeCloseEvent()

    assert gui.MainWindow._veto_close_during_start_pose_move(w, event) is False
    warning = capsys.readouterr().err
    assert "switch_controllers" in warning
    assert home_move.FPC in warning and home_move.STJC in warning


# --------------------------------------------------------------------------- #
# Status text / colour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "state, active, check",
    [
        (FAILED, False, _has_red),
        (DONE, False, _has_green),
        (MOVING, True, _has_orange),
        (IDLE, False, _is_grey),
        (UNAVAILABLE, False, _is_grey),
    ],
)
def test_status_label_shows_state_message_and_a_distinct_colour(
    monkeypatch, state, active, check
):
    message = "detail for {}".format(state)
    node = _FakeNode(state=state, active=active, message=message)
    w, _ = _window(monkeypatch, node)

    _refresh(w)

    text = w._start_pose_status.text
    assert state in text and message in text, text
    style = w._start_pose_status.style
    assert check(style), style
    assert _has_red(style) is (state == FAILED), style


def test_terminal_outcome_is_announced_once(monkeypatch):
    node = _FakeNode(state=DONE, active=False, message="START POSE reached")
    w, _ = _window(monkeypatch, node)
    _refresh(w)
    _refresh(w)
    assert sum("START POSE reached" in m for m in w.messages) == 1


# --------------------------------------------------------------------------- #
# The node without a pose
# --------------------------------------------------------------------------- #
def test_node_without_a_pose_reports_unavailable_and_refuses_everything():
    """Called UNBOUND against a stand-in with only the two start-pose fields:
    proves the not-installed branch never reaches into the mixin (no
    _home_lock, no controller -- none of that exists without a pose)."""
    reason = "/nope.yaml: file does not exist"
    fake = SimpleNamespace(_start_pose=None, _start_pose_reason=reason,
                           _start_pose_io_installed=False)
    N = gui.EefGuiNode

    status = N.get_home_status(fake)
    assert status == {"state": UNAVAILABLE, "active": False,
                      "message": reason, "duration_s": None}
    assert N.request_go_home(fake) is False
    assert N.request_stop_home(fake) is False
    assert N.start_pose_info(fake) == (None, reason)


# --------------------------------------------------------------------------- #
# The node WITH a pose: construction wires the mixin as designed
# --------------------------------------------------------------------------- #
class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, msg, **kw):
        self.lines.append(("info", msg))

    def warn(self, msg, **kw):
        self.lines.append(("warn", msg))

    def error(self, msg, **kw):
        self.lines.append(("error", msg))


class _Client:
    def __init__(self, srv_type, name):
        self.srv_type = srv_type
        self.srv_name = name


def _stub_node_class():
    """EefGuiNode with every rclpy entity creation stubbed and Node.__init__
    skipped: no context, no DDS participant, nothing on the graph. What runs
    is the node's OWN __init__ -- including resolve_start_pose() and the
    install_home_move_io() call -- against recorded stubs."""

    class Stub(gui.EefGuiNode):
        def __init__(self):
            self.created = {"subs": [], "clients": [], "pubs": [], "timers": []}
            self._logger = _Logger()
            gui.EefGuiNode.__init__(self)

        def create_subscription(self, msg_type, topic, cb, qos, **kw):
            self.created["subs"].append(topic)
            return SimpleNamespace(topic=topic)

        def create_client(self, srv_type, name, **kw):
            client = _Client(srv_type, name)
            self.created["clients"].append(client)
            return client

        def create_publisher(self, msg_type, topic, qos, **kw):
            self.created["pubs"].append(topic)
            return SimpleNamespace(topic=topic, publish=lambda m: None)

        def create_timer(self, period, cb, **kw):
            self.created["timers"].append((period, cb))
            return SimpleNamespace(period=period, cancel=lambda: None)

        def get_logger(self):
            return self._logger

    return Stub


@pytest.fixture
def stubbed_rclpy(monkeypatch):
    import rclpy.node as rclpy_node
    from gello_recorder import home_move_ros

    monkeypatch.setattr(rclpy_node.Node, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(
        home_move_ros, "ActionClient",
        lambda node, action_type, name: SimpleNamespace(action=name))
    return _stub_node_class()


def _write_yaml(tmp_path, joints, gripper=None):
    lines = ["policy_leader_node:", "  ros__parameters:",
             "    start_pose: [{}]".format(", ".join(str(v) for v in joints))]
    if gripper is not None:
        lines.append("    start_gripper: {}".format(gripper))
    path = tmp_path / "deploy.yaml"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_with_a_pose_the_mixin_is_installed_towards_it(
    stubbed_rclpy, monkeypatch, tmp_path
):
    monkeypatch.setenv("START_POSE_CONFIG", _write_yaml(tmp_path, CARROT))
    node = stubbed_rclpy()

    assert node._start_pose_io_installed is True
    assert node.get_home_status()["state"] == IDLE
    ctrl = node._home_controller
    assert ctrl.target_joints == pytest.approx(CARROT)
    assert ctrl.target_label == gui.START_POSE_LABEL       # "START POSE"
    # The gripper pause entry is an ALIAS of the GUI's own client on the same
    # service; the arm pause client is the mixin's own.
    assert node._svc_home["grip_pause"] is node._svc["grip_pause"]
    assert node._svc_home["grip_pause"].srv_name == "/gello_gripper_bridge/pause"
    assert node._svc_home["arm_pause"].srv_name == "/gello_ur_bridge/pause"
    assert node._svc_home["arm_pause"] is not node._svc["resume"]
    # The 10 Hz sequencer tick exists, and the joint-state subscription too.
    assert any(abs(p - 0.1) < 1e-9 for p, _cb in node.created["timers"])
    assert "/joint_states" in node.created["subs"]
    pose, _ = node.start_pose_info()
    assert pose is not None and pose.joints == pytest.approx(CARROT)
    assert any(level == "info" and "GO TO START POSE ready" in msg
               for level, msg in node._logger.lines)
    # A refused request is a plain False (no joint states yet), never a raise.
    assert node.request_go_home() is False
    assert node.get_home_status()["state"] == FAILED


def test_without_a_pose_no_home_move_io_is_created(stubbed_rclpy, monkeypatch):
    monkeypatch.delenv("START_POSE_CONFIG", raising=False)
    node = stubbed_rclpy()

    assert node._start_pose_io_installed is False
    assert not hasattr(node, "_home_controller")
    assert not hasattr(node, "_svc_home")
    assert "/joint_states" not in node.created["subs"]
    assert node.created["timers"] == []
    assert node.get_home_status()["state"] == UNAVAILABLE
    assert "START_POSE_CONFIG" in node.get_home_status()["message"]
    assert node.request_go_home() is False
    assert any(level == "warn" and "disabled" in msg
               for level, msg in node._logger.lines)


def test_a_nonzero_start_gripper_warns_but_does_not_disable(
    stubbed_rclpy, monkeypatch, tmp_path
):
    monkeypatch.setenv("START_POSE_CONFIG", _write_yaml(tmp_path, CARROT, 0.5))
    node = stubbed_rclpy()

    assert node._start_pose_io_installed is True
    assert any(level == "warn" and "start_gripper" in msg
               for level, msg in node._logger.lines)
