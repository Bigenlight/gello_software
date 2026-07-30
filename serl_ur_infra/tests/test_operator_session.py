"""Unit tests for the ROS-free status contract and ROS scene-ready gate."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env import operator_session as operator_session_module  # noqa: E402
from ur_env.operator_session import (  # noqa: E402
    ACTOR_STATUS_TOPIC,
    AUTO_SUCCESS_SERVICE,
    MANUAL_SUCCESS_SERVICE,
    SCENE_READY_SERVICE,
    ActorStatusTracker,
    OWNER_NONE,
    OWNER_HOLD,
    OWNER_POLICY,
    POLICY_RUNNING,
    RosOperatorSession,
    WAIT_HOME_APPROVAL,
    WAIT_SCENE_READY,
    resolve_backend_node,
    resolve_deadman_source,
)


class _String:
    def __init__(self):
        self.data = ""


class _Trigger:
    class Request:
        pass

    class Response:
        def __init__(self):
            self.success = False
            self.message = ""


class _SetBool:
    class Request:
        def __init__(self, *, data=False):
            self.data = data

    class Response:
        def __init__(self):
            self.success = False
            self.message = ""


@pytest.fixture
def ros_message_stubs(monkeypatch):
    std_msgs = ModuleType("std_msgs")
    std_msgs_msg = ModuleType("std_msgs.msg")
    std_msgs_msg.String = _String
    std_msgs.msg = std_msgs_msg
    std_srvs = ModuleType("std_srvs")
    std_srvs_srv = ModuleType("std_srvs.srv")
    std_srvs_srv.SetBool = _SetBool
    std_srvs_srv.Trigger = _Trigger
    std_srvs.srv = std_srvs_srv
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)
    monkeypatch.setitem(sys.modules, "std_srvs", std_srvs)
    monkeypatch.setitem(sys.modules, "std_srvs.srv", std_srvs_srv)


class _Publisher:
    def __init__(self):
        self.messages = []
        self.failure = None

    def publish(self, message):
        if self.failure is not None:
            raise self.failure
        self.messages.append(message)


class _Node:
    def __init__(self):
        self.publisher = _Publisher()
        self.publisher_args = None
        self.service_args = []
        self.callbacks = {}

    def create_publisher(self, message_type, topic, depth):
        self.publisher_args = (message_type, topic, depth)
        return self.publisher

    def create_service(self, service_type, name, callback):
        self.service_args.append((service_type, name))
        self.callbacks[name] = callback
        return SimpleNamespace()


class _Deadman:
    def __init__(self):
        self.engaged = False
        self.failure = None

    def is_engaged(self):
        if self.failure is not None:
            raise self.failure
        return self.engaged

    def fresh_engaged(self):
        return self.is_engaged()


def _status(tracker, *, state=WAIT_SCENE_READY, env_step=-1, **kwargs):
    return tracker.status(
        state=state,
        control_owner=(
            OWNER_NONE
            if state == WAIT_SCENE_READY
            else OWNER_HOLD
            if state == WAIT_HOME_APPROVAL
            else OWNER_POLICY
        ),
        episode_id=0,
        episode_step=0,
        env_step=env_step,
        message="test",
        **kwargs,
    )


def _call(node, name=SCENE_READY_SERVICE):
    return node.callbacks[name](_Trigger.Request(), _Trigger.Response())


def _set_auto(node, enabled):
    return node.callbacks[AUTO_SUCCESS_SERVICE](
        _SetBool.Request(data=enabled), _SetBool.Response()
    )


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def test_sparse_status_retains_last_evaluated_classifier_values():
    tracker = ActorStatusTracker("run")
    scored = _status(
        tracker,
        state=POLICY_RUNNING,
        env_step=3,
        classifier_evaluated=True,
        classifier_probability=0.73,
        classifier_threshold=0.2,
    )
    sparse = _status(
        tracker,
        state=POLICY_RUNNING,
        env_step=4,
        classifier_evaluated=False,
    )

    assert scored.classifier_evaluated is True
    assert sparse.classifier_evaluated is False
    assert sparse.classifier_probability == pytest.approx(0.73)
    assert sparse.classifier_threshold == pytest.approx(0.2)
    assert sparse.classifier_env_step == 3


def test_trigger_is_edge_gated_and_requires_fresh_disengaged(
    ros_message_stubs,
):
    node = _Node()
    deadman = _Deadman()
    session = RosOperatorSession(node, deadman)
    tracker = ActorStatusTracker("run")

    assert node.publisher_args[1:] == (ACTOR_STATUS_TOPIC, 10)
    assert {name for _, name in node.service_args} == {
        SCENE_READY_SERVICE,
        AUTO_SUCCESS_SERVICE,
        MANUAL_SUCCESS_SERVICE,
    }

    early = _call(node)
    assert early.success is False
    assert "not waiting" in early.message

    thread = threading.Thread(
        target=session.wait_for_scene_ready,
        args=(_status(tracker),),
        daemon=True,
    )
    thread.start()
    _wait_until(lambda: session.waiting)

    deadman.engaged = True
    engaged = _call(node)
    assert engaged.success is False
    assert "DISENGAGE" in engaged.message
    assert thread.is_alive()

    deadman.engaged = False
    deadman.failure = RuntimeError("heartbeat stale")
    stale = _call(node)
    assert stale.success is False
    assert "not fresh" in stale.message
    assert thread.is_alive()

    deadman.failure = None
    accepted = _call(node)
    assert accepted.success is True
    thread.join(timeout=1.0)
    assert not thread.is_alive()

    # The accepted edge was consumed. It cannot pre-arm the next wait.
    after = _call(node)
    assert after.success is False
    assert "not waiting" in after.message


def test_trigger_prefers_strict_fresh_state_over_legacy_false(
    ros_message_stubs,
):
    class _NoValidHeartbeat(_Deadman):
        def is_engaged(self):
            return False

        def fresh_engaged(self):
            raise RuntimeError("no valid heartbeat")

    node = _Node()
    session = RosOperatorSession(node, _NoValidHeartbeat())
    tracker = ActorStatusTracker("run")
    thread = threading.Thread(
        target=session.wait_for_scene_ready,
        args=(_status(tracker),),
        daemon=True,
    )
    thread.start()
    _wait_until(lambda: session.waiting)

    response = _call(node)
    assert response.success is False
    assert "not fresh" in response.message
    assert thread.is_alive()
    session._scene_ready.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_home_approval_is_explicit_but_not_blocked_by_deadman_state(
    ros_message_stubs,
):
    node = _Node()
    deadman = _Deadman()
    deadman.engaged = True
    deadman.failure = RuntimeError("heartbeat stale")
    session = RosOperatorSession(node, deadman)
    tracker = ActorStatusTracker("run")
    thread = threading.Thread(
        target=session.wait_for_home_approval,
        args=(_status(tracker, state=WAIT_HOME_APPROVAL),),
        daemon=True,
    )
    thread.start()
    _wait_until(lambda: session.waiting)

    accepted = _call(node)

    assert accepted.success is True
    assert "HOME approved" in accepted.message
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_status_is_schema_v1_json_and_publish_failure_does_not_break_gate(
    ros_message_stubs,
):
    warnings = []
    node = _Node()
    deadman = _Deadman()
    session = RosOperatorSession(node, deadman, warn=warnings.append)
    tracker = ActorStatusTracker("run")
    running = _status(tracker, state=POLICY_RUNNING)

    session.publish(running)
    payload = json.loads(node.publisher.messages[-1].data)
    assert payload == running.as_payload()
    assert payload["schema_version"] == 2
    assert payload["auto_success"] is False

    node.publisher.failure = RuntimeError("publisher down")
    thread = threading.Thread(
        target=session.wait_for_scene_ready,
        args=(_status(tracker),),
        daemon=True,
    )
    thread.start()
    _wait_until(lambda: session.waiting)
    accepted = _call(node)
    assert accepted.success is True
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert any("publish failed" in warning for warning in warnings)


def test_wait_status_is_republished_for_a_late_or_restarted_gui(
    ros_message_stubs, monkeypatch
):
    monkeypatch.setattr(
        operator_session_module, "WAIT_STATUS_REPUBLISH_S", 0.01
    )
    node = _Node()
    session = RosOperatorSession(node, _Deadman())
    tracker = ActorStatusTracker("run")
    wait_status = _status(tracker)

    thread = threading.Thread(
        target=session.wait_for_scene_ready,
        args=(wait_status,),
        daemon=True,
    )
    thread.start()
    _wait_until(lambda: len(node.publisher.messages) >= 2)

    payloads = [json.loads(message.data) for message in node.publisher.messages]
    assert all(payload["state"] == WAIT_SCENE_READY for payload in payloads)
    assert all(payload["run_id"] == "run" for payload in payloads)

    response = _call(node)
    assert response.success is True
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_status_publisher_and_gate_creation_are_mandatory(
    ros_message_stubs,
):
    class _NoStatusNode(_Node):
        def create_publisher(self, message_type, topic, depth):
            del message_type, topic, depth
            raise RuntimeError("status unavailable")

    with pytest.raises(RuntimeError, match="status unavailable"):
        RosOperatorSession(_NoStatusNode(), _Deadman())

    class _NoGateNode(_Node):
        def create_service(self, service_type, name, callback):
            del service_type, name, callback
            raise RuntimeError("gate unavailable")

    with pytest.raises(RuntimeError, match="gate unavailable"):
        RosOperatorSession(_NoGateNode(), _Deadman())


def test_sustained_wait_status_publish_failure_aborts_gate(
    ros_message_stubs, monkeypatch
):
    monkeypatch.setattr(
        operator_session_module, "WAIT_STATUS_REPUBLISH_S", 0.001
    )
    node = _Node()
    node.publisher.failure = RuntimeError("publisher down")
    session = RosOperatorSession(node, _Deadman())
    tracker = ActorStatusTracker("run")

    with pytest.raises(RuntimeError, match="cannot publish /hil/actor_status"):
        session.wait_for_scene_ready(_status(tracker))

    assert session.waiting is False


def test_manual_success_is_active_episode_scoped_and_exactly_once(
    ros_message_stubs,
):
    node = _Node()
    session = RosOperatorSession(node, _Deadman())
    tracker = ActorStatusTracker("run")

    assert session.auto_success is False
    early = _call(node, MANUAL_SUCCESS_SERVICE)
    assert early.success is False
    assert "not been published" in early.message

    session.publish(_status(tracker, state=POLICY_RUNNING, env_step=0))
    accepted = _call(node, MANUAL_SUCCESS_SERVICE)
    assert accepted.success is True
    assert "run=run episode=0" in accepted.message

    same_mode = _set_auto(node, False)
    assert same_mode.success is True
    assert "already MANUAL" in same_mode.message

    duplicate = _call(node, MANUAL_SUCCESS_SERVICE)
    assert duplicate.success is False
    assert "already queued" in duplicate.message

    assert session.consume_operator_success("other-run", 0) is False
    assert session.consume_operator_success("run", 1) is False
    assert session.consume_operator_success("run", 0) is True
    assert session.consume_operator_success("run", 0) is False


def test_auto_mode_is_authoritative_and_mode_edge_clears_pending(
    ros_message_stubs,
):
    node = _Node()
    session = RosOperatorSession(node, _Deadman())
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker, state=POLICY_RUNNING, env_step=0))
    assert _call(node, MANUAL_SUCCESS_SERVICE).success is True

    auto = _set_auto(node, True)
    assert auto.success is True
    assert session.auto_success is True
    assert session.consume_operator_success("run", 0) is False
    assert json.loads(node.publisher.messages[-1].data)["auto_success"] is True

    rejected = _call(node, MANUAL_SUCCESS_SERVICE)
    assert rejected.success is False
    assert "AUTO mode" in rejected.message

    manual = _set_auto(node, False)
    assert manual.success is True
    assert session.auto_success is False
    assert json.loads(node.publisher.messages[-1].data)["auto_success"] is False


@pytest.mark.parametrize("boundary", ["episode", "run", "state"])
def test_status_boundary_discards_unconsumed_operator_success(
    ros_message_stubs, boundary
):
    node = _Node()
    session = RosOperatorSession(node, _Deadman())
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker, state=POLICY_RUNNING, env_step=0))
    assert _call(node, MANUAL_SUCCESS_SERVICE).success is True

    if boundary == "episode":
        status = tracker.status(
            state=POLICY_RUNNING,
            control_owner=OWNER_POLICY,
            episode_id=1,
            episode_step=0,
            env_step=1,
            message="next episode",
        )
    elif boundary == "run":
        status = _status(
            ActorStatusTracker("new-run"),
            state=POLICY_RUNNING,
            env_step=1,
        )
    else:
        status = _status(tracker, state=WAIT_HOME_APPROVAL, env_step=1)
    session.publish(status)

    assert session.consume_operator_success("run", 0) is False


def test_real_wrapper_shape_resolves_existing_backend_node_and_deadman():
    node = object()
    deadman = _Deadman()
    intervention = SimpleNamespace(expert=SimpleNamespace(deadman=deadman))
    wrapped = SimpleNamespace(
        env=SimpleNamespace(env=intervention),
        unwrapped=SimpleNamespace(backend=SimpleNamespace(_node=node)),
    )

    assert resolve_backend_node(wrapped) is node
    assert resolve_deadman_source(wrapped) is deadman
