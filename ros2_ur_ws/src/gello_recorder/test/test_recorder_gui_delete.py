"""Headless tests for the "Delete last take" button in the recorder GUI.

Same pattern as ``test_task_recorder_gui.py`` (itself borrowed from
``ur_gello_bringup/test/test_hil_actor_status.py``): the window methods are
called UNBOUND against a plain duck-typed stand-in, so there is no Qt event
loop, no QApplication, no rclpy node and no robot -- only the operator-facing
decision logic (when is the button live, does Yes delete exactly once, does a
stale prompt get refused) is under test.

The one extra wrinkle here is the confirmation prompt itself: this codebase
never uses a blocking ``QMessageBox.exec_()`` anywhere the robot can move (see
``gello_recorder_gui``'s module docstring and ``_on_delete_last_take_clicked``),
so the real window builds a NON-MODAL box and drives it entirely off the
``finished`` signal. ``_FakeMessageBox``/``_FakeSignal`` below stand in for
just enough of that Qt surface (construction kwargs, ``finished.connect``,
``clickedButton``/``standardButton``, ``show``/``raise_``/``close``) to
exercise the real click/finished/re-check code paths, with ``QMessageBox``
itself monkeypatched onto the module so no real Qt widget is ever created.

Deliberately NARROW, matching ``test_task_recorder_gui.py``'s stated scope:
button enable/disable, the Yes path, the No/close paths, the stale-path
re-check, and Start Recording closing a stray prompt. Exact tooltip/label
strings are not pinned beyond the substrings the operator relies on.
"""

import os
from types import SimpleNamespace

import pytest

gui = pytest.importorskip(
    "gello_recorder.gello_recorder_gui",
    reason="gello_recorder_gui is not importable (PyQt5/rclpy overlay missing?)",
)


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
    this test deliberately does not model (e.g. ``_refresh_teleop`` when a
    test does not care about the teleop panel). Falsy so it can never flip a
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


class _FakeSignal:
    """Stand-in for a Qt signal: ``connect`` + a synchronous ``emit``.

    Real ``QDialog.finished`` fires synchronously within the call that ends
    the dialog (a button click, or ``close()``), so a synchronous ``emit``
    here is not a simplification of the timing -- it is the timing.
    """

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class _FakeMessageBox:
    """Stand-in for QMessageBox, monkeypatched over ``gui.QMessageBox``.

    Models exactly the surface ``_on_delete_last_take_clicked`` drives
    (construction, the setters, ``finished.connect``) plus what
    ``_on_delete_box_finished`` reads back (``clickedButton``/
    ``standardButton``), and the two ways a prompt can end without a
    click: ``close()`` (Start Recording / closeEvent) and the test's own
    ``click()`` helper (an operator pressing Yes or No).
    """

    Warning = "Warning"
    # Real QMessageBox.StandardButton values (bit flags): plain ints so the
    # production code's ``QMessageBox.Yes | QMessageBox.No`` still works
    # against this stand-in.
    Yes = 0x00004000
    No = 0x00010000

    def __init__(self, parent=None):
        self.parent = parent
        self.modality = None
        self.icon = None
        self.title = None
        self.text = None
        self.buttons = None
        self.default_button = None
        self.escape_button = None
        self.finished = _FakeSignal()
        self._clicked_button = None
        self.shown = False
        self.raised = False
        self.activated = False
        self.closed = False

    def setWindowModality(self, modality):
        self.modality = modality

    def setAttribute(self, attribute, on=True):
        # Qt.WA_DeleteOnClose in production; irrelevant to the logic under test.
        self.attributes = getattr(self, "attributes", {})
        self.attributes[attribute] = on

    def setIcon(self, icon):
        self.icon = icon

    def setWindowTitle(self, title):
        self.title = title

    def setText(self, text):
        self.text = text

    def setStandardButtons(self, buttons):
        self.buttons = buttons

    def setDefaultButton(self, button):
        self.default_button = button

    def setEscapeButton(self, button):
        self.escape_button = button

    def clickedButton(self):
        return self._clicked_button

    def standardButton(self, button):
        # The real QMessageBox maps a QAbstractButton back to its
        # StandardButton; here clickedButton() already IS the standard
        # button token (Yes/No), so mapping is the identity.
        return button

    def show(self):
        self.shown = True

    def raise_(self):
        self.raised = True

    def activateWindow(self):
        self.activated = True

    def close(self):
        # A close with no button pressed reports no clicked button -- the
        # real QMessageBox.close() reports Rejected the same way, which is
        # exactly what makes "only an explicit Yes is a confirmation" true.
        self.closed = True
        self._clicked_button = None
        self.finished.emit(0)  # QDialog.Rejected

    # --- test helper, not part of the real Qt API ---------------------------
    def click(self, button):
        """Simulate an operator clicking ``button`` (Yes or No)."""
        self._clicked_button = button
        self.finished.emit(1 if button == self.Yes else 0)


# --------------------------------------------------------------------------- #
# Fake node + window
# --------------------------------------------------------------------------- #
class _FakeDeleteNode:
    """Stand-in for the node half of the delete-last-take contract."""

    def __init__(self, *, deletable=None, recording=False, cameras_ready=True,
                 take_index=4, delete_result=None, delete_error=None):
        self._deletable = deletable
        self._recording = recording
        self._cameras_ready = cameras_ready
        self._take_index = take_index
        self._delete_result = delete_result
        self._delete_error = delete_error
        self.delete_calls = 0
        self.start_recording_calls = 0

    # --- the delete-last-take contract --------------------------------------
    def deletable_take_dir(self):
        return self._deletable

    def delete_last_take(self):
        self.delete_calls += 1
        if self._delete_error is not None:
            raise RuntimeError(self._delete_error)
        if self._delete_result is not None:
            return dict(self._delete_result)
        return {
            "path": self._deletable,
            "n_files": 3,
            "bytes": 12345,
            "take_index_after": self._take_index - 1,
        }

    # --- base MainWindow._refresh_controls / _on_start_clicked deps --------
    def is_recording(self):
        return self._recording

    def cameras_ready(self):
        return self._cameras_ready

    def take_index(self):
        return self._take_index

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

    def start_recording(self):
        self.start_recording_calls += 1
        self._recording = True
        return "/fake/take_dir"


class _DeleteWindow:
    """Duck-typed stand-in for MainWindow's widgets and window state."""

    def __init__(self, node):
        self._node = node
        self._delete_box = None
        self._delete_last_take_button = _FakeWidget()
        self._start_button = _FakeWidget()
        self._stop_button = _FakeWidget()
        self._take_label = _FakeWidget("Take: 0")
        self._elapsed_label = _FakeWidget()
        self._status_label = _FakeWidget()
        self._record_start_wall = None
        self._teleop_resume_armed = False
        self._teleop_last_shown_msg = None
        self.messages = []
        self.calls = []

    def statusBar(self):
        return SimpleNamespace(showMessage=self._show_message)

    def _show_message(self, text, msec=0):
        self.messages.append(text)

    def __getattr__(self, name):
        # Everything modelled is set in __init__ (or bound explicitly by
        # _window() below), so only unmodelled members arrive here.
        if name.startswith("__"):
            raise AttributeError(name)
        if name.endswith("_button") or name.endswith("_label"):
            widget = _FakeWidget()
            self.__dict__[name] = widget
            return widget
        return _Absent(name, self.__dict__["calls"])

    def feedback(self):
        return " | ".join(self.messages)


def _window(monkeypatch, node):
    """A stand-in with the real delete methods bound and QMessageBox faked.

    Binding ``_on_delete_last_take_clicked`` / ``_on_delete_box_finished`` /
    ``_confirm_delete_last_take`` / ``_refresh_controls`` / ``_on_start_clicked``
    directly onto the instance (rather than relying on ``__getattr__``) is
    required here: ``window`` is a ``_DeleteWindow``, not a ``MainWindow``, so
    when the real method body does ``self._confirm_delete_last_take(path)`` it
    resolves against ``window``'s own attributes -- unbound, that would hit
    ``_Absent`` and silently no-op. This is the same trick
    ``test_task_recorder_gui.py``'s ``_window()`` uses for
    ``_disarm_go_home``/``_refresh_home_controls``.
    """
    monkeypatch.setattr(gui, "QMessageBox", _FakeMessageBox)

    window = _DeleteWindow(node)
    window._on_delete_last_take_clicked = (
        lambda: gui.MainWindow._on_delete_last_take_clicked(window)
    )
    window._on_delete_box_finished = (
        lambda box, path: gui.MainWindow._on_delete_box_finished(window, box, path)
    )
    window._confirm_delete_last_take = (
        lambda path: gui.MainWindow._confirm_delete_last_take(window, path)
    )
    window._refresh_controls = lambda: gui.MainWindow._refresh_controls(window)
    window._on_start_clicked = lambda: gui.MainWindow._on_start_clicked(window)
    return window


def _make_take_dir(tmp_path, name="take_04_20260916_120000", n_files=3, size=1000):
    d = tmp_path / name
    d.mkdir()
    for i in range(n_files):
        (d / "file_{}.bin".format(i)).write_bytes(b"x" * size)
    return str(d)


# --------------------------------------------------------------------------- #
# 1. Button enablement tracks node.deletable_take_dir()
# --------------------------------------------------------------------------- #
def test_button_enabled_iff_deletable_take_dir_is_not_none(monkeypatch, tmp_path):
    take_dir = _make_take_dir(tmp_path)

    node = _FakeDeleteNode(deletable=None)
    window = _window(monkeypatch, node)
    window._refresh_controls()
    assert window._delete_last_take_button.enabled is False

    node._deletable = take_dir
    window._refresh_controls()
    assert window._delete_last_take_button.enabled is True


def test_button_disabled_while_recording_even_if_node_disagrees(monkeypatch, tmp_path):
    """Belt-and-braces: `recording` is checked here directly, not only trusted
    to make deletable_take_dir() return None (which the real node's contract
    promises, but a destructive button should not depend on a single
    upstream flag staying honest)."""

    take_dir = _make_take_dir(tmp_path)
    node = _FakeDeleteNode(deletable=take_dir, recording=True)
    window = _window(monkeypatch, node)

    window._refresh_controls()

    assert window._delete_last_take_button.enabled is False


# --------------------------------------------------------------------------- #
# 2. Yes path
# --------------------------------------------------------------------------- #
def test_click_opens_a_sized_nonmodal_prompt_and_yes_deletes_exactly_once(
    monkeypatch, tmp_path
):
    take_dir = _make_take_dir(tmp_path, n_files=4, size=1000)  # ~3.9 KB
    node = _FakeDeleteNode(
        deletable=take_dir,
        delete_result={
            "path": take_dir, "n_files": 4, "bytes": 4000, "take_index_after": 3,
        },
    )
    window = _window(monkeypatch, node)

    window._on_delete_last_take_clicked()

    box = window._delete_box
    assert box is not None
    assert box.shown is True
    # Non-modal: this is the entire reason the codebase avoids exec_() near
    # anything that can move the robot -- Pause Teleop must stay clickable
    # while this prompt is up.
    import PyQt5.QtCore as QtCore
    assert box.modality == QtCore.Qt.NonModal
    assert box.default_button == _FakeMessageBox.No
    assert box.escape_button == _FakeMessageBox.No
    assert os.path.basename(take_dir) in box.text
    assert "4 files" in box.text
    assert "only the take most recently stopped" in box.text.lower()

    box.click(_FakeMessageBox.Yes)

    assert node.delete_calls == 1
    assert window._delete_box is None            # ready for a future prompt
    assert any("deleted" in m.lower() for m in window.messages)
    assert any(os.path.basename(take_dir) in m for m in window.messages)
    # take_index_after=3 is the COUNTER; the folder that appears next is
    # take_04 (start_recording increments before naming). The message must
    # name the folder the operator will actually see, not the counter.
    assert any("take count now 3" in m.lower() for m in window.messages)
    assert any("take_04" in m for m in window.messages)
    assert not any("next take: 3" in m.lower() for m in window.messages)


def test_delete_failure_shows_the_reason_and_is_not_swallowed(monkeypatch, tmp_path):
    take_dir = _make_take_dir(tmp_path)
    node = _FakeDeleteNode(deletable=take_dir, delete_error="not the last take")
    window = _window(monkeypatch, node)

    window._on_delete_last_take_clicked()
    window._delete_box.click(_FakeMessageBox.Yes)

    assert node.delete_calls == 1
    assert any(
        "delete failed" in m.lower() and "not the last take" in m
        for m in window.messages
    ), window.messages


# --------------------------------------------------------------------------- #
# 3. No / close paths never delete
# --------------------------------------------------------------------------- #
def test_no_click_never_deletes(monkeypatch, tmp_path):
    take_dir = _make_take_dir(tmp_path)
    node = _FakeDeleteNode(deletable=take_dir)
    window = _window(monkeypatch, node)

    window._on_delete_last_take_clicked()
    window._delete_box.click(_FakeMessageBox.No)

    assert node.delete_calls == 0
    assert window._delete_box is None


def test_programmatic_close_never_deletes(monkeypatch, tmp_path):
    """box.close() (as used by Start Recording / closeEvent) reports no
    clicked button -- only an explicit Yes click may confirm."""

    take_dir = _make_take_dir(tmp_path)
    node = _FakeDeleteNode(deletable=take_dir)
    window = _window(monkeypatch, node)

    window._on_delete_last_take_clicked()
    window._delete_box.close()

    assert node.delete_calls == 0
    assert window._delete_box is None


def test_second_click_while_open_raises_instead_of_stacking(monkeypatch, tmp_path):
    take_dir = _make_take_dir(tmp_path)
    node = _FakeDeleteNode(deletable=take_dir)
    window = _window(monkeypatch, node)

    window._on_delete_last_take_clicked()
    first_box = window._delete_box
    window._on_delete_last_take_clicked()

    assert window._delete_box is first_box    # no second prompt created
    assert first_box.raised is True
    assert first_box.activated is True


# --------------------------------------------------------------------------- #
# 4. Stale path is refused
# --------------------------------------------------------------------------- #
def test_yes_is_refused_if_the_deletable_take_changed_while_open(monkeypatch, tmp_path):
    """The prompt was opened for take A; between opening it and clicking Yes,
    a new take started and stopped, so the node's deletable take is now B.
    The captured path (A) must NOT be deleted just because Yes was clicked --
    the re-check compares the CURRENT node state, not the stale snapshot."""

    take_a = _make_take_dir(tmp_path, name="take_04_a")
    take_b = _make_take_dir(tmp_path, name="take_05_b")
    node = _FakeDeleteNode(deletable=take_a)
    window = _window(monkeypatch, node)

    window._on_delete_last_take_clicked()
    box = window._delete_box
    assert os.path.basename(take_a) in box.text

    node._deletable = take_b   # a new take rolled over underneath the prompt

    box.click(_FakeMessageBox.Yes)

    assert node.delete_calls == 0
    assert any("changed" in m.lower() for m in window.messages)


# --------------------------------------------------------------------------- #
# 5. Start Recording closes a stray open box
# --------------------------------------------------------------------------- #
def test_start_recording_closes_an_open_delete_prompt(monkeypatch, tmp_path):
    take_dir = _make_take_dir(tmp_path)
    node = _FakeDeleteNode(deletable=take_dir, recording=False, cameras_ready=True)
    window = _window(monkeypatch, node)

    window._on_delete_last_take_clicked()
    box = window._delete_box
    assert box.closed is False

    window._on_start_clicked()

    assert box.closed is True
    assert node.delete_calls == 0                 # close() is not a Yes
    assert window._delete_box is None
    assert node.start_recording_calls == 1         # Start itself still ran


# --------------------------------------------------------------------------- #
# 6. The node half: GelloRecorderGuiNode.deletable_take_dir / delete_last_take
# --------------------------------------------------------------------------- #
# Same unbound-method trick, one level down: the real node methods run against
# a plain object carrying only the fields they read (_session_lock, _session,
# _last_take_dir, _take_index, output_root, get_logger). No rclpy.init, no
# executor -- these methods never touch ROS, only the lock and the filesystem.
gui_node = pytest.importorskip(
    "gello_recorder.gello_gui_node",
    reason="gello_gui_node is not importable (rclpy overlay missing?)",
)


class _FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warn(self, msg):
        self.lines.append(("warn", msg))


class _NodeHalf:
    """Only what deletable_take_dir/delete_last_take/stop_recording read."""

    def __init__(self, output_root, last_take_dir=None, take_index=0, session=None):
        import threading
        self._session_lock = threading.Lock()
        self._session = session
        self._last_take_dir = last_take_dir
        self._take_index = take_index
        self.output_root = output_root
        self._logger = _FakeLogger()

    def get_logger(self):
        return self._logger

    # bound real methods
    def deletable_take_dir(self):
        return gui_node.GelloRecorderGuiNode.deletable_take_dir(self)

    def delete_last_take(self):
        return gui_node.GelloRecorderGuiNode.delete_last_take(self)

    def take_index(self):
        return gui_node.GelloRecorderGuiNode.take_index(self)


def _node_take(tmp_path, index):
    """A real take_NN_... directory under tmp_path, flat regular files only."""
    name = "take_{:02d}_20260916_12{:02d}00".format(index, index)
    d = tmp_path / name
    d.mkdir()
    for fname in ("cam1.mp4", "cam2.mp4", "vectors.h5", "gripper_mode.json"):
        (d / fname).write_bytes(b"x" * 10)
    return str(d)


def test_node_refuses_to_delete_while_a_session_is_open(tmp_path):
    take = _node_take(tmp_path, 4)
    node = _NodeHalf(str(tmp_path), last_take_dir=take, take_index=5,
                     session=object())          # "recording": any non-None
    assert node.deletable_take_dir() is None    # not even offered
    with pytest.raises(RuntimeError, match="while recording"):
        node.delete_last_take()
    assert os.path.isdir(take)
    assert node._last_take_dir == take          # still remembered for later


def test_node_delete_is_one_shot_and_reuses_the_index(tmp_path):
    take = _node_take(tmp_path, 4)
    node = _NodeHalf(str(tmp_path), last_take_dir=take, take_index=4)
    assert node.deletable_take_dir() == take

    result = node.delete_last_take()

    assert not os.path.exists(take)
    assert result["path"] == os.path.realpath(take)
    assert result["n_files"] == 4               # gripper_mode.json counted too
    assert result["take_index_after"] == 3
    assert node.take_index() == 3               # what "Take: N" is painted from
    assert node.deletable_take_dir() is None    # button goes dark
    with pytest.raises(RuntimeError, match="no take to delete"):
        node.delete_last_take()                 # second Yes cannot delete twice
    assert node._take_index == 3                # and does not decrement again


def test_node_never_deletes_anything_but_last_take_dir(tmp_path):
    """Two finished takes on disk; only the remembered one goes."""
    older = _node_take(tmp_path, 3)
    newest = _node_take(tmp_path, 4)
    node = _NodeHalf(str(tmp_path), last_take_dir=newest, take_index=4)
    node.delete_last_take()
    assert os.path.isdir(older)
    assert not os.path.exists(newest)


def test_node_wraps_a_failed_rmtree_and_keeps_the_take_offered(tmp_path, monkeypatch):
    """An OSError out of the actual delete (EACCES/EROFS/...) must reach the
    GUI as the ONE exception type it catches (RuntimeError -> status bar),
    and must NOT clear _last_take_dir: a half-deleted take that silently
    fell off the button would be unrecoverable from the GUI."""
    take = _node_take(tmp_path, 4)
    node = _NodeHalf(str(tmp_path), last_take_dir=take, take_index=4)

    def _boom(take_dir, output_root):
        raise PermissionError(13, "Permission denied", take_dir)

    monkeypatch.setattr(gui_node.take_delete, "delete_take_dir", _boom)
    with pytest.raises(RuntimeError, match="Permission denied"):
        node.delete_last_take()
    assert node._last_take_dir == take          # still offered for a retry
    assert node._take_index == 4                # counter untouched
    assert os.path.isdir(take)


def test_node_refuses_a_remembered_path_that_no_longer_validates(tmp_path):
    """The remembered path is swapped for a symlink behind the node's back
    (an operator 'tidying' with mv + ln -s): the helper's guard must still
    win, and the message must reach the GUI as RuntimeError."""
    real = _node_take(tmp_path, 4)
    moved = str(tmp_path / "kept_take_04")
    os.rename(real, moved)
    os.symlink(moved, real, target_is_directory=True)
    node = _NodeHalf(str(tmp_path), last_take_dir=real, take_index=4)
    with pytest.raises(RuntimeError, match="symlink"):
        node.delete_last_take()
    assert os.path.isdir(moved)
    assert os.path.islink(real)


def test_stop_recording_remembers_the_take_only_after_close_drained(tmp_path):
    """stop_recording() is where _last_take_dir is written -- and it must be
    written AFTER session.close() returns, i.e. after the writer thread has
    drained, so a delete can never race a frame still being flushed."""
    take = _node_take(tmp_path, 4)
    order = []

    class _Session:
        session_dir = take

        def close(self):
            order.append("close")
            return {"duration_s": 1.0, "message_counts": {}}

        def dropped_frames(self):
            return {"total": 0}

    class _Node(_NodeHalf):
        def _health_summary(self, session, stats):
            order.append(("health", self._last_take_dir))
            return {}

        def stop_recording(self):
            return gui_node.GelloRecorderGuiNode.stop_recording(self)

    node = _Node(str(tmp_path), last_take_dir=None, take_index=4, session=_Session())
    assert node.deletable_take_dir() is None    # open session -> nothing offered
    stats = node.stop_recording()
    assert order[0] == "close"
    assert order[1] == ("health", take)         # set right after close()
    assert stats["session_dir"] == take
    assert node._session is None
    assert node.deletable_take_dir() == take    # now, and only now, offered
