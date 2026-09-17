"""Jazzy console-entrypoint shutdown behavior."""

from rclpy.executors import ExternalShutdownException

import gello_recorder.gello_ur_recorder_node as recorder_module


def test_external_shutdown_finalizes_and_exits_cleanly(monkeypatch):
    events = []

    class FakeNode:
        def destroy_node(self):
            events.append("destroy")

    monkeypatch.setattr(recorder_module.rclpy, "init", lambda args=None: events.append("init"))
    monkeypatch.setattr(recorder_module, "GelloUrRecorder", FakeNode)
    monkeypatch.setattr(
        recorder_module.rclpy,
        "spin",
        lambda node: (_ for _ in ()).throw(ExternalShutdownException()),
    )
    monkeypatch.setattr(recorder_module.rclpy, "ok", lambda: False)
    monkeypatch.setattr(recorder_module.rclpy, "shutdown", lambda: events.append("shutdown"))

    recorder_module.main()

    assert events == ["init", "destroy"]
