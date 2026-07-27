"""Offline guards for the real-HIL acceptance runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).with_name("run_real_hil.py")
_SPEC = importlib.util.spec_from_file_location("run_real_hil_safety", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Node:
    def __init__(self, publishers):
        self.publishers = dict(publishers)
        self.queried = []

    def count_publishers(self, topic):
        self.queried.append(("publishers", topic))
        return self.publishers.get(topic, 1)

    def count_subscribers(self, topic):
        self.queried.append(("subscribers", topic))
        return 1


def test_gripper_preflight_rejects_a_second_command_publisher():
    ros = {
        "command_topic": "/arm/commands",
        "gripper_command_topic": "/robotiq_gripper/command_percent",
    }
    node = _Node({ros["command_topic"]: 1, ros["gripper_command_topic"]: 2})

    assert not _MODULE.preflight_command_topics(
        node, ros, arming=True, gripper_enabled=True
    )
    assert ("publishers", ros["command_topic"]) in node.queried
    assert ("publishers", ros["gripper_command_topic"]) in node.queried


def test_disabled_gripper_does_not_require_ownership_of_its_topic():
    ros = {
        "command_topic": "/arm/commands",
        "gripper_command_topic": "/robotiq_gripper/command_percent",
    }
    node = _Node({ros["command_topic"]: 1, ros["gripper_command_topic"]: 2})

    assert _MODULE.preflight_command_topics(
        node, ros, arming=True, gripper_enabled=False
    )
    assert ("publishers", ros["gripper_command_topic"]) not in node.queried


def test_summary_cannot_pass_when_intervention_checks_are_skipped(capsys):
    result = _MODULE.summarize([{"intervened": 0, "held": 0}])

    assert result == {
        "anchor_latch": None,
        "gain_latch": None,
        "frame_map": None,
        "action_exec": None,
        "held_rate": True,
        "pass": False,
    }
    output = capsys.readouterr().out
    assert "SKIP anchor-latch" in output
    assert "전체: FAIL / 미검증 항목 있음" in output
