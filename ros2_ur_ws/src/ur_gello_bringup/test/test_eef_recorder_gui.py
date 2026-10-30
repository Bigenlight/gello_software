"""Headless operator-contract tests for the integrated EEF recorder GUI."""

from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("PyQt5.QtWidgets")

gui = pytest.importorskip("ur_gello_bringup.eef_recorder_gui_node")


class _FakeButton:
    def __init__(self, enabled=True):
        self._enabled = enabled

    def isEnabled(self):
        return self._enabled


class _FakeStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, timeout):
        self.messages.append((message, timeout))


class _FakeRecorderWindow:
    def __init__(self, start_enabled=True):
        self._start_button = _FakeButton(start_enabled)
        self.started = 0
        self.stopped = 0
        self.cameras_stopped = 0
        self._delete_box = None
        self._preview_timer = None
        self._state_timer = None
        self._control_timer = None
        self._status = _FakeStatusBar()

    def _on_start_clicked(self):
        self.started += 1

    def _on_stop_clicked(self):
        self.stopped += 1

    def _shutdown_cameras(self):
        self.cameras_stopped += 1

    def statusBar(self):
        return self._status


class _FakeRecorderNode:
    def __init__(self, recording=False):
        self.recording = recording
        self.finalized = 0

    def is_recording(self):
        return self.recording

    def stop_recording(self):
        self.recording = False
        self.finalized += 1


class _FakeEvent:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def _shortcut_shell(recording=False, start_enabled=True):
    return SimpleNamespace(
        _recorder_node=_FakeRecorderNode(recording),
        _recorder_window=_FakeRecorderWindow(start_enabled),
    )


def test_r_starts_when_idle_and_camera_ready():
    shell = _shortcut_shell(recording=False, start_enabled=True)
    gui.IntegratedWindow._on_record_shortcut(shell)
    assert shell._recorder_window.started == 1
    assert shell._recorder_window.stopped == 0


def test_r_stops_immediately_when_recording():
    shell = _shortcut_shell(recording=True, start_enabled=False)
    gui.IntegratedWindow._on_record_shortcut(shell)
    assert shell._recorder_window.started == 0
    assert shell._recorder_window.stopped == 1


def test_r_does_not_start_before_cameras_are_ready():
    shell = _shortcut_shell(recording=False, start_enabled=False)
    gui.IntegratedWindow._on_record_shortcut(shell)
    assert shell._recorder_window.started == 0
    assert "warming up" in shell._recorder_window._status.messages[-1][0]


def test_compact_css_reduces_geometry_without_changing_action_colour():
    original = (
        "QPushButton { font-size: 16pt; min-height: 56px; padding: 15px; "
        "background-color: #2e7d32; } QPushButton:pressed { padding-top: 17px; }"
    )
    compact = gui._compact_css(original)
    assert "font-size: 11pt" in compact
    assert "min-height: 30px" in compact
    assert "padding: 4px" in compact
    assert "padding-top: 5px" in compact
    assert "#2e7d32" in compact


def test_each_node_gets_its_own_single_threaded_executor(monkeypatch):
    executors = []
    threads = []

    class _FakeExecutor:
        def __init__(self):
            self.nodes = []
            executors.append(self)

        def add_node(self, node):
            self.nodes.append(node)

        def spin(self):
            pass

    class _FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self.started = False
            threads.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(gui, "SingleThreadedExecutor", _FakeExecutor)
    monkeypatch.setattr(gui.threading, "Thread", _FakeThread)

    recorder = object()
    eef = object()
    recorder_executor, recorder_thread = gui._start_node_executor(recorder)
    eef_executor, eef_thread = gui._start_node_executor(eef)

    assert recorder_executor is not eef_executor
    assert recorder_executor.nodes == [recorder]
    assert eef_executor.nodes == [eef]
    assert recorder_thread.started and eef_thread.started
    assert recorder_thread.daemon and eef_thread.daemon


def _close_shell(*, recording=True, veto=False):
    recorder_node = _FakeRecorderNode(recording)
    recorder_window = _FakeRecorderWindow()
    shutdown_calls = []
    eef_window = SimpleNamespace(
        _timer=None,
        _vw_timer=None,
        _veto_close_during_start_pose_move=lambda event: veto,
    )
    shell = SimpleNamespace(
        _closed=False,
        _recorder_node=recorder_node,
        _recorder_window=recorder_window,
        _eef_window=eef_window,
        _shutdown_callback=lambda: shutdown_calls.append(True),
    )
    return shell, recorder_node, recorder_window, shutdown_calls


def test_close_finalizes_recording_and_cameras_once():
    shell, node, window, shutdown_calls = _close_shell(recording=True)
    event = _FakeEvent()
    gui.IntegratedWindow.closeEvent(shell, event)

    assert event.accepted
    assert node.finalized == 1
    assert window.cameras_stopped == 1
    assert shutdown_calls == [True]

    second = _FakeEvent()
    gui.IntegratedWindow.closeEvent(shell, second)
    assert second.accepted
    assert node.finalized == 1
    assert window.cameras_stopped == 1
    assert shutdown_calls == [True]


def test_close_veto_keeps_recorder_and_ros_alive_during_start_pose_move():
    shell, node, window, shutdown_calls = _close_shell(recording=True, veto=True)
    event = _FakeEvent()
    gui.IntegratedWindow.closeEvent(shell, event)

    assert not event.accepted
    assert node.finalized == 0
    assert window.cameras_stopped == 0
    assert shutdown_calls == []
