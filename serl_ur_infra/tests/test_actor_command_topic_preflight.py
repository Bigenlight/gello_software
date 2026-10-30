"""Regression tests for actor command-topic DDS discovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "serl_ur_infra" / "scripts" / "run_remote_rlpd_actor.py"
_SPEC = importlib.util.spec_from_file_location("actor_command_topic_preflight", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Node:
    def __init__(self, subscribers, *, publishers=1):
        self._subscribers = iter(subscribers)
        self._last_subscribers = 0
        self.publishers = publishers
        self.subscriber_queries = 0

    def count_publishers(self, topic):
        del topic
        return self.publishers

    def count_subscribers(self, topic):
        del topic
        self.subscriber_queries += 1
        self._last_subscribers = next(
            self._subscribers, self._last_subscribers
        )
        return self._last_subscribers


def _inputs(node):
    backend = SimpleNamespace(_node=node)
    env = SimpleNamespace(unwrapped=SimpleNamespace(backend=backend))
    robot_config = SimpleNamespace(
        ROS={"command_topic": "/forward_position_controller/commands"}
    )
    return env, robot_config


def test_preflight_waits_for_late_controller_subscriber():
    node = _Node([0, 0, 1])
    env, config = _inputs(node)

    _MODULE._preflight_command_topics(
        env,
        config,
        discovery_timeout_s=1.0,
        poll_interval_s=0.0,
    )

    assert node.subscriber_queries == 3


def test_preflight_still_rejects_missing_controller_subscriber():
    node = _Node([0])
    env, config = _inputs(node)

    with pytest.raises(SystemExit, match="nothing subscribes"):
        _MODULE._preflight_command_topics(
            env,
            config,
            discovery_timeout_s=0.0,
            poll_interval_s=0.0,
        )


def test_preflight_still_rejects_a_second_command_publisher():
    node = _Node([1], publishers=2)
    env, config = _inputs(node)

    with pytest.raises(SystemExit, match="2 publishers"):
        _MODULE._preflight_command_topics(
            env,
            config,
            discovery_timeout_s=1.0,
            poll_interval_s=0.0,
        )
