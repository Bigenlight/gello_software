"""Headless tests for the task recorder GUI's GO HOME button logic.

The pattern is the one used by ``ur_gello_bringup/test/test_hil_actor_status.py``:
the window methods are called UNBOUND against plain duck-typed stand-ins, so
there is no Qt event loop, no QApplication, no rclpy node and no robot.  Only
the operator-facing decision logic is under test -- which is exactly the part
that must not be discovered to be wrong while the arm is moving.

Deliberately NARROW: what survives here is the two-click confirm, the gating
that keeps a click from starting a motion at the wrong moment, the status the
operator reads afterwards, the neutralized Resume button, the single-click STOP
HOME, and the refusal to close the window out from under a moving arm.  Exact
labels, tooltips and near-duplicate state permutations are not pinned.

Two more sit at the end, for the gripper-mode work: the MIXED-TAKE flag (the one
sidecar verdict whose absence would silently poison a dataset) and the latch
readout's state mapping.  Everything else about the sidecar -- when it is
written, that a write failure is swallowed -- is deliberately not pinned here.

The module skips itself (cleanly, not as a failure) until
``gello_recorder.task_recorder_gui`` lands, so it can be written against the
contract before the implementation exists.
"""

import re
import time
from types import SimpleNamespace

import pytest

gui = pytest.importorskip(
    "gello_recorder.task_recorder_gui",
    reason="task_recorder_gui is not implemented yet",
)

if not hasattr(gui, "TaskRecorderWindow"):  # pragma: no cover - contract guard
    pytest.skip(
        "gello_recorder.task_recorder_gui has no TaskRecorderWindow yet",
        allow_module_level=True,
    )


# The states the node's get_home_status() may report.  Only these three are
# named here; the five in-flight ones (PAUSING, SWITCHING_TO_JTC, MOVING,
# OPENING_GRIPPER, RESTORING_FPC) are interchangeable as far as this GUI is
# concerned -- it branches on the status dict's ``active`` flag, not on which
# step is running -- so "MOVING" stands in for all five below.
IDLE = "IDLE"
DONE = "DONE"
FAILED = "FAILED"


# --------------------------------------------------------------------------- #
# Colour helpers -- assert the MEANING of a stylesheet, not its exact bytes.
# --------------------------------------------------------------------------- #
_NAMED_COLORS = {
    "red": (204, 51, 51),
    "darkred": (139, 0, 0),
    "crimson": (220, 20, 60),
    "orange": (221, 136, 0),
    "darkorange": (255, 140, 0),
    "green": (34, 170, 34),
    "darkgreen": (0, 100, 0),
    "gray": (136, 136, 136),
    "grey": (136, 136, 136),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}


def _style_colors(style):
    style = style or ""
    colors = [
        tuple(int(hexrgb[i:i + 2], 16) for i in (0, 2, 4))
        for hexrgb in re.findall(r"#([0-9a-fA-F]{6})", style)
    ]
    lowered = style.lower()
    for name, rgb in _NAMED_COLORS.items():
        if re.search(r"\b{}\b".format(name), lowered):
            colors.append(rgb)
    return colors


def _has_red(style):
    return any(r > 2 * g and r > 2 * b for r, g, b in _style_colors(style))


def _has_orange(style):
    return any(r > g + 40 and g > b + 40 for r, g, b in _style_colors(style))


def _has_green(style):
    return any(g > r + 40 and g > b + 40 for r, g, b in _style_colors(style))


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


class _Absent:
    """Callable *and* falsy placeholder for anything the stand-in omits.

    Reached only through ``__getattr__``, i.e. for members of the real window
    that this test deliberately does not model.  Falsy so it can never flip a
    gate open by accident, callable so an incidental ``self._something()`` in
    the implementation does not explode the test.
    """

    def __init__(self, name, log):
        self._name = name
        self._log = log

    def __call__(self, *args, **kwargs):
        self._log.append(self._name)
        return None

    def __bool__(self):
        return False


class _FakeHomeNode:
    """Stand-in for the node half of the contract."""

    def __init__(self, *, recording=False, state=IDLE, active=False,
                 message="", duration_s=None, accept=True):
        self._recording = recording
        self._status = {
            "state": state,
            "active": active,
            "message": message,
            "duration_s": duration_s,
        }
        self._accept = accept
        self.go_home_calls = 0
        self.stop_home_calls = 0

    # --- the four methods the contract names -------------------------------
    def is_recording(self):
        return self._recording

    def get_home_status(self):
        return dict(self._status)

    def request_go_home(self):
        self.go_home_calls += 1
        return self._accept

    def request_stop_home(self):
        # HomeMoveController.abort() semantics: False when nothing is in
        # flight, otherwise the request is recorded and the transition happens
        # on the next sequencer tick.
        self.stop_home_calls += 1
        return bool(self._status["active"])

    # --- base-class getters, in case a shared refresh path touches them -----
    def cameras_ready(self):
        return True

    def take_index(self):
        return 0

    def warmup_seconds_remaining(self):
        return 0.0

    def get_teleop_status(self):
        return {
            "arm_state": "PAUSED",
            "arm_state_age_s": 0.1,
            "grip_state": "PAUSED",
            "grip_available": True,
            "pending": False,
            "last_msg": None,
        }


class _HomeWindow:
    """Duck-typed stand-in for TaskRecorderWindow's widgets and window state."""

    def __init__(self, node):
        self._node = node
        self._home_armed = False
        self._home_button = _FakeWidget()
        self._home_status_label = _FakeWidget()
        self._start_button = _FakeWidget()
        self._stop_button = _FakeWidget()
        self._record_start_wall = None
        self._teleop_resume_armed = False
        self.messages = []
        self.calls = []

    def statusBar(self):
        return SimpleNamespace(showMessage=self._show_message)

    def _show_message(self, text, msec=0):
        self.messages.append(text)

    def __getattr__(self, name):
        # Everything modelled is set in __init__, so only unmodelled members
        # arrive here.  Widget-shaped names get a widget; anything else gets a
        # falsy no-op (see _Absent).
        if name.startswith("__"):
            raise AttributeError(name)
        if name.endswith("_button") or name.endswith("_label"):
            widget = _FakeWidget()
            self.__dict__[name] = widget
            return widget
        return _Absent(name, self.__dict__["calls"])

    # --- operator-visible feedback of any shape ----------------------------
    def feedback(self):
        return " | ".join(
            [t for t in self.messages] + [self._home_status_label.text or ""]
        )


def _window(monkeypatch, node, **fields):
    """A stand-in with the real disarm/refresh bound and QTimer captured.

    Returns ``(window, deferred)`` where ``deferred`` collects every
    ``QTimer.singleShot(ms, callback)`` the implementation schedules -- the
    same trick ``test_hil_actor_status.py`` uses for the abort two-click gate.
    """

    deferred = []

    class _FakeQTimer:
        @staticmethod
        def singleShot(ms, callback):
            deferred.append((ms, callback))

    monkeypatch.setattr(gui, "QTimer", _FakeQTimer, raising=False)

    window = _HomeWindow(node)
    window.__dict__.update(fields)
    window._disarm_go_home = lambda: gui.TaskRecorderWindow._disarm_go_home(window)
    window._refresh_home_controls = (
        lambda: gui.TaskRecorderWindow._refresh_home_controls(window)
    )
    return window, deferred


def _resting(window):
    """Establish (and return) the resting button look via the real disarm."""

    gui.TaskRecorderWindow._disarm_go_home(window)
    return window._home_button.text, window._home_button.style


def _click(window):
    gui.TaskRecorderWindow._on_go_home_clicked(window)


def _refresh(window):
    gui.TaskRecorderWindow._refresh_home_controls(window)


# --------------------------------------------------------------------------- #
# Two-click confirm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("accepted", [True, False])
def test_the_first_click_only_arms_and_the_second_fires_exactly_once(
    monkeypatch, accepted
):
    """One click must never move the robot; two must move it once.

    The refused case (``request_go_home()`` -> False) is the same click path:
    the request is still made exactly once, the button still returns to rest,
    and the operator is still told something -- a button press that silently
    does nothing reads as a broken GUI.
    """

    node = _FakeHomeNode(accept=accepted)
    window, deferred = _window(monkeypatch, node)
    rest_text, rest_style = _resting(window)

    _click(window)

    assert window._home_armed is True
    assert node.go_home_calls == 0                   # nothing requested yet
    assert window._home_button.text != rest_text     # and it says so
    assert window._home_button.style != rest_style
    assert [ms for ms, _cb in deferred] == [3000]    # the confirm window

    _click(window)

    assert node.go_home_calls == 1                   # exactly once
    assert window._home_armed is False               # never left armed
    assert window._home_button.text == rest_text
    assert window._home_button.style == rest_style
    assert window.feedback().strip() != ""


def test_the_scheduled_disarm_restores_the_resting_button(monkeypatch):
    """Timing out must cost the operator a click, never move the robot."""

    window, deferred = _window(monkeypatch, _FakeHomeNode())
    rest_text, rest_style = _resting(window)

    _click(window)
    deferred[0][1]()                                 # the 3 s timer fires

    assert window._home_armed is False
    assert window._home_button.text == rest_text
    assert window._home_button.style == rest_style
    assert window._node.go_home_calls == 0


# --------------------------------------------------------------------------- #
# Gating -- what the ~5 Hz refresh is allowed to do to the buttons
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "recording, active, armed, expected",
    [
        (False, False, False, True),   # idle recorder, idle arm -> the only GO case
        (True, False, False, False),   # a take is being recorded
        (False, True, False, False),   # a GO HOME is already running
        # ...but an ARMED button stays live through either, or the operator's
        # confirming click is swallowed by a timer tick and the robot neither
        # moves nor says why.
        (True, True, True, True),
    ],
)
def test_home_button_gate(monkeypatch, recording, active, armed, expected):
    node = _FakeHomeNode(
        recording=recording,
        active=active,
        state="MOVING" if active else IDLE,
    )
    window, _deferred = _window(monkeypatch, node, _home_armed=armed)

    _refresh(window)

    assert window._home_button.enabled is expected


def test_start_recording_is_locked_out_only_while_the_arm_is_being_driven(
    monkeypatch,
):
    """A take must not begin mid-homing: it would record the homing motion.

    The converse matters too -- once the sequence is over the base GUI owns
    that button again, or the operator can never start the next take.
    """

    node = _FakeHomeNode(state="MOVING", active=True)
    window, _deferred = _window(monkeypatch, node)

    _refresh(window)
    assert window._start_button.enabled is False

    # Once the sequence is over this override must not touch the button AT ALL
    # -- the base ``_refresh_controls`` (which ran just before it in the real
    # window, and is not modelled here) owns it. Untouched shows up as None.
    node._status.update({"state": DONE, "active": False})
    window._start_button.enabled = None
    _refresh(window)
    assert window._start_button.enabled is None


# --------------------------------------------------------------------------- #
# Status label
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "state, active, has_colour",
    [
        # FAILED means the arm may be anywhere and the controller may still be
        # STJC -- it must never be mistakable for a clean finish.
        (FAILED, False, _has_red),
        (DONE, False, _has_green),
        ("MOVING", True, _has_orange),
    ],
)
def test_status_label_shows_the_state_the_message_and_a_distinct_colour(
    monkeypatch, state, active, has_colour
):
    message = "detail for {}".format(state)
    window, _deferred = _window(
        monkeypatch,
        _FakeHomeNode(state=state, active=active, message=message),
    )

    _refresh(window)

    text = window._home_status_label.text
    assert state in text, text
    assert message in text, text                     # incl. the post-DONE advice
    style = window._home_status_label.style
    assert has_colour(style), style
    assert _has_red(style) is (state == FAILED), style


# --------------------------------------------------------------------------- #
# The inherited Resume Teleop button
# --------------------------------------------------------------------------- #
def test_resume_teleop_stays_neutralized_across_refresh_ticks(monkeypatch):
    """That button is the JOINT-mode chase resume, and this is an EEF session.

    After a GO HOME the arm is at HOME while the passive leader is wherever the
    operator left it; inside the 1.5 rad/joint gate the arm would glide toward
    the leader's ABSOLUTE joint configuration.  The base's ``_refresh_teleop``
    re-drives (and can re-arm) this button 5 times a second, so the override
    has to win on every single tick.
    """

    window, _deferred = _window(monkeypatch, _FakeHomeNode())

    for _ in range(3):
        # What the base refresh does just before our override runs.
        window._teleop_resume_armed = True
        window._teleop_resume_button.setEnabled(True)

        gui.TaskRecorderWindow._neutralize_teleop_resume(window)

        assert window._teleop_resume_button.enabled is False
        assert window._teleop_resume_armed is False

    # A disabled button with no explanation is indistinguishable from a broken
    # one: it has to point at the EEF GUI's ENGAGE instead.
    explained = "{} {}".format(
        window._teleop_resume_button.text, window._teleop_resume_button.tooltip
    )
    assert "eef" in explained.lower()


# --------------------------------------------------------------------------- #
# STOP HOME + refusing to close out from under a moving arm
# --------------------------------------------------------------------------- #
class _FakeCloseEvent:
    """A QCloseEvent as far as the guard is concerned."""

    def __init__(self):
        self.ignored = False
        self.accepted = False

    def ignore(self):
        self.ignored = True

    def accept(self):
        self.accepted = True


def _stopping_window(monkeypatch, node, deadline=None):
    """Window stand-in with the real STOP-HOME click path bound.

    ``closeEvent`` itself cannot be driven here -- its terminal branch is
    ``super().closeEvent(event)``, and zero-arg ``super()`` refuses a duck-typed
    stand-in -- so what is exercised is the guard it delegates to. The two are
    the same decision: ``closeEvent`` is exactly ``if guard(event): return`` and
    otherwise the base's close (take flush + camera teardown + rclpy shutdown),
    unchanged from the plain recorder.
    """

    window, deferred = _window(monkeypatch, node, _home_close_deadline=deadline)
    window._on_stop_home_clicked = (
        lambda: gui.TaskRecorderWindow._on_stop_home_clicked(window)
    )
    return window, deferred


def _veto_close(window, event):
    return gui.TaskRecorderWindow._veto_close_during_home_move(window, event)


def test_closing_the_window_mid_move_is_refused_and_stops_the_arm(monkeypatch):
    """The X button is the panic reflex, and on its own it does NOTHING.

    A trajectory STJC already accepted survives the client dying, so the arm
    finishes its sweep with the joints left on STJC and teleop silently dead.
    The first close must therefore be refused and converted into the STOP HOME
    the operator actually wanted.
    """

    node = _FakeHomeNode(state="MOVING", active=True)
    window, _deferred = _stopping_window(monkeypatch, node)
    event = _FakeCloseEvent()

    assert _veto_close(window, event) is True
    assert event.ignored is True and event.accepted is False
    assert node.stop_home_calls == 1                  # the same abort path
    assert window.feedback().strip() != ""            # and it says why


def test_closing_when_no_move_is_running_is_handed_to_the_base(monkeypatch):
    """Idle close must be an ordinary close: no veto, no stop, no meddling."""

    node = _FakeHomeNode(state=DONE, active=False)
    window, _deferred = _stopping_window(monkeypatch, node)
    event = _FakeCloseEvent()

    assert _veto_close(window, event) is False        # -> super().closeEvent()
    assert event.ignored is False
    assert node.stop_home_calls == 0


def test_stop_home_fires_on_a_single_click_and_only_while_a_move_is_active(
    monkeypatch,
):
    """A stop must never cost two clicks -- the deliberate inverse of GO HOME."""

    node = _FakeHomeNode(state="MOVING", active=True)
    window, deferred = _stopping_window(monkeypatch, node)

    gui.TaskRecorderWindow._on_stop_home_clicked(window)

    assert node.stop_home_calls == 1                  # first click, not second
    assert deferred == []                             # no confirm window at all
    assert window.feedback().strip() != ""

    # And it is offered only when there is something to stop: the same ~5 Hz
    # refresh that drives GO HOME drives this button, so there is one path.
    _refresh(window)
    assert window._stop_home_button.enabled is True

    node._status.update({"state": FAILED, "active": False})
    _refresh(window)
    assert window._stop_home_button.enabled is False


def test_a_wedged_sequence_cannot_trap_the_operator_in_the_window(
    monkeypatch, capsys
):
    """The refusal is bounded: a stuck sequencer must not hold the GUI hostage.

    Letting go of a possibly-moving arm is bad, so the escape hatch is only
    survivable if it says so loudly and names the way back -- including that
    pressing GO HOME again self-heals the stranded controller.
    """

    node = _FakeHomeNode(state="MOVING", active=True)
    # The grace window opened long enough ago to have expired.
    window, _deferred = _stopping_window(
        monkeypatch, node, deadline=time.monotonic() - 1.0
    )
    event = _FakeCloseEvent()

    assert _veto_close(window, event) is False        # honoured this time
    assert event.ignored is False

    warning = capsys.readouterr().err
    assert "switch_controllers" in warning, warning
    assert "forward_position_controller" in warning, warning
    assert "scaled_joint_trajectory_controller" in warning, warning
    assert "GO HOME" in warning, warning              # the self-healing route


# --------------------------------------------------------------------------- #
# Gripper-mode provenance: the mixed take
# --------------------------------------------------------------------------- #
def test_a_take_whose_gripper_mode_changed_mid_way_is_flagged_not_resolved():
    """The one sidecar verdict whose absence silently poisons a dataset.

    ``grip_cmd`` is an action channel that gets mean/std-normalised downstream,
    and it means two different things in the bridge's two modes (a continuous
    0..1 value, or a latched 0.0/1.0 endpoint).  A take that starts DISABLED and
    later reports OPEN contains BOTH -- and unlike a wholly-continuous or
    wholly-binary take, nothing about the numbers themselves reveals it.

    So it must not be resolved to whichever family arrived last, and the
    startup-time parameters must not be allowed to rescue it: here they say
    "continuous", which is exactly the confident wrong answer.
    """

    node = pytest.importorskip(
        "gello_recorder.task_gui_node",
        reason="task_gui_node needs the ROS overlay on sys.path",
    )
    params = {
        "available": True,
        "discrete_mode": False,
        "discrete_open_at": 0.3,
        "discrete_close_at": 0.7,
    }

    clean = node.build_gripper_mode_record(
        {"DISABLED": 120}, "DISABLED", "DISABLED", params
    )
    assert clean["mode"] == "continuous"
    assert clean["mode_changed_mid_take"] is False
    assert clean["state_topic_seen"] is True
    assert clean["discrete_open_at"] == 0.3          # thresholds carried through

    mixed = node.build_gripper_mode_record(
        {"DISABLED": 40, "UNKNOWN": 2, "OPEN": 78}, "DISABLED", "OPEN", params
    )
    assert mixed["mode_changed_mid_take"] is True
    assert mixed["mode"] == "unknown"                # never one of the two
    assert mixed["mode_source"] == "state_topic"     # the parameters do not win
    assert "UNKNOWN" in mixed["grip_cmd_semantics"]  # and it says so in words

    # No bridge at all (a policy-deploy stack starts none) degrades to unknown
    # rather than to the mode nearly every session happens to run in.
    absent = node.build_gripper_mode_record(
        {}, None, None, {"available": False, "note": "no gripper bridge"}
    )
    assert absent["mode"] == "unknown"
    assert absent["state_topic_seen"] is False
    assert absent["mode_changed_mid_take"] is False


# --------------------------------------------------------------------------- #
# Gripper-mode provenance: the latch readout in the teleop bar
# --------------------------------------------------------------------------- #
def test_the_latch_readout_distinguishes_the_modes_and_never_paints_a_fault():
    """A readout, not an alarm -- and never a stale word left standing.

    None of these states is a failure: DISABLED is the ordinary continuous mode,
    and UNKNOWN only means no threshold has been crossed since the bridge
    started (or since a pause reset the latch).  Red on either would train the
    operator to ignore the bar.  What the readout DOES have to do is separate
    "continuous" from the latched states at a glance, and go quiet -- not
    confident -- when the topic is stale, absent, or speaking a vocabulary this
    window has never heard (it is actively growing on the bridge side).
    """

    words = {}
    for token in ("DISABLED", "UNKNOWN", "OPEN", "CLOSED", "RAMPING"):
        text, color = gui._grip_latch_display(token, 0.1)
        assert not _has_red(color), (token, color)
        words[token] = text

    assert "continuous" in words["DISABLED"].lower()
    assert "OPEN" in words["OPEN"] and "CLOSED" in words["CLOSED"]
    assert len(set(words.values())) == len(words)     # all five distinguishable

    live = gui._grip_latch_display("OPEN", 0.1)
    for state, age in (
        ("OPEN", 5.0),          # stale: older than the ~2 s window
        ("OPEN", None),         # never stamped
        (None, None),           # never received
        ("SOMETHING_NEW", 0.1), # a token from a newer bridge
    ):
        text, color = gui._grip_latch_display(state, age)
        assert (text, color) != live, (state, age)
        assert "OPEN" not in text, (state, age)
        assert not _has_red(color), (state, age)
