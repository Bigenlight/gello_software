"""Unit tests for the operator ABORT EPISODE token (ROS-free).

ABORT discards the running episode mid-flight -- after a collision, or when the
policy has wandered somewhere useless -- instead of waiting for task success or
the step limit.  The session layer only latches a one-shot token; the actor
consumes it and drives the robot home.

The token discipline is deliberately identical to MANUAL success (episode
scoped, one shot, cleared on boundaries) with exactly one difference tested
here: abort is accepted in AUTO as well as MANUAL, because an abort is not an
assertion about task success.

ROS is faked the same way ``test_operator_session.py`` fakes it -- ``std_msgs``
and ``std_srvs`` module stubs plus a node that records ``create_service``
callbacks -- so nothing here needs rclpy.
"""

from __future__ import annotations

import os
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.operator_session import (  # noqa: E402
    ABORT_EPISODE_SERVICE,
    AUTO_SUCCESS_SERVICE,
    ACTIVE_CONTROL_STATES,
    ActorStatusTracker,
    HOLD,
    HUMAN_INTERVENTION,
    MANUAL_SUCCESS_SERVICE,
    OWNER_HOLD,
    OWNER_HUMAN,
    OWNER_NONE,
    OWNER_POLICY,
    POLICY_RUNNING,
    RosOperatorSession,
    WAIT_HOME_APPROVAL,
    WAIT_SCENE_READY,
    resolve_follow_controls,
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

    def is_engaged(self):
        return self.engaged

    def fresh_engaged(self):
        return self.is_engaged()


_OWNER_FOR_STATE = {
    POLICY_RUNNING: OWNER_POLICY,
    HUMAN_INTERVENTION: OWNER_HUMAN,
    HOLD: OWNER_HOLD,
    WAIT_HOME_APPROVAL: OWNER_HOLD,
    WAIT_SCENE_READY: OWNER_NONE,
}


def _status(
    tracker,
    *,
    state=POLICY_RUNNING,
    episode_id=0,
    env_step=0,
    terminal_reason="",
    success=False,
):
    return tracker.status(
        state=state,
        control_owner=_OWNER_FOR_STATE[state],
        episode_id=episode_id,
        episode_step=0,
        env_step=env_step,
        terminal_reason=terminal_reason,
        success=success,
        message="test",
    )


def _abort(node):
    return node.callbacks[ABORT_EPISODE_SERVICE](
        _Trigger.Request(), _Trigger.Response()
    )


def _manual_success(node):
    return node.callbacks[MANUAL_SUCCESS_SERVICE](
        _Trigger.Request(), _Trigger.Response()
    )


def _set_auto(node, enabled):
    return node.callbacks[AUTO_SUCCESS_SERVICE](
        _SetBool.Request(data=enabled), _SetBool.Response()
    )


def _session(node=None):
    node = node if node is not None else _Node()
    return node, RosOperatorSession(node, _Deadman())


def test_abort_service_is_advertised_as_a_trigger(ros_message_stubs):
    node, _ = _session()

    assert ABORT_EPISODE_SERVICE == "/hil/abort_episode"
    advertised = dict(
        (name, service_type) for service_type, name in node.service_args
    )
    assert advertised[ABORT_EPISODE_SERVICE] is _Trigger
    # Same service type as MARK SUCCESS: the GUI needs no new message package.
    assert advertised[ABORT_EPISODE_SERVICE] is advertised[MANUAL_SUCCESS_SERVICE]


def test_abort_before_any_status_is_rejected(ros_message_stubs):
    node, session = _session()

    response = _abort(node)

    assert response.success is False
    assert "not been published" in response.message
    assert session.consume_operator_abort("run", 0) is False


@pytest.mark.parametrize("state", sorted(ACTIVE_CONTROL_STATES))
def test_abort_is_accepted_in_every_active_control_state(
    ros_message_stubs, state
):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker, state=state))

    response = _abort(node)

    assert response.success is True
    assert "run=run episode=0" in response.message
    assert session.consume_operator_abort("run", 0) is True


@pytest.mark.parametrize("state", [WAIT_SCENE_READY, WAIT_HOME_APPROVAL])
def test_abort_outside_active_states_is_rejected_and_names_the_state(
    ros_message_stubs, state
):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker, state=state))

    response = _abort(node)

    assert response.success is False
    assert state in response.message
    assert session.consume_operator_abort("run", 0) is False


def test_second_abort_while_one_is_queued_is_rejected(ros_message_stubs):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))

    assert _abort(node).success is True
    duplicate = _abort(node)

    assert duplicate.success is False
    assert "already queued" in duplicate.message
    # The rejected duplicate did not disturb the token that is actually queued.
    assert session.consume_operator_abort("run", 0) is True


@pytest.mark.parametrize("auto", [False, True])
def test_abort_is_accepted_in_both_manual_and_auto_success_modes(
    ros_message_stubs, auto
):
    """Abort is not a success assertion, so the MANUAL/AUTO toggle is irrelevant.

    MARK SUCCESS is refused in AUTO on purpose (the classifier owns success
    there).  Refusing ABORT in AUTO would remove the button in exactly the
    state where a wandering policy most needs stopping.
    """

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _set_auto(node, auto).success is True
    assert session.auto_success is auto

    response = _abort(node)

    assert response.success is True
    assert session.consume_operator_abort("run", 0) is True


def test_manual_success_stays_auto_gated_while_abort_does_not(
    ros_message_stubs,
):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    _set_auto(node, True)

    success = _manual_success(node)
    abort = _abort(node)

    assert success.success is False
    assert "AUTO mode" in success.message
    assert abort.success is True


def test_abort_token_is_consumed_exactly_once(ros_message_stubs):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _abort(node).success is True

    assert session.consume_operator_abort("run", 0) is True
    assert session.consume_operator_abort("run", 0) is False


def test_mismatched_identity_neither_consumes_nor_steals_the_token(
    ros_message_stubs,
):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _abort(node).success is True

    assert session.consume_operator_abort("other-run", 0) is False
    assert session.consume_operator_abort("run", 1) is False
    assert session.consume_operator_abort("run", 7) is False
    # The active episode can still spend its own token afterwards.
    assert session.consume_operator_abort("run", 0) is True


def test_consume_rejects_malformed_identity(ros_message_stubs):
    _, session = _session()

    with pytest.raises(ValueError, match="run_id is required"):
        session.consume_operator_abort("", 0)
    with pytest.raises(ValueError, match="run_id is required"):
        session.consume_operator_abort(None, 0)
    with pytest.raises(ValueError, match="episode_id"):
        session.consume_operator_abort("run", -1)
    with pytest.raises(ValueError, match="episode_id"):
        session.consume_operator_abort("run", True)
    with pytest.raises(ValueError, match="episode_id"):
        session.consume_operator_abort("run", 1.0)


@pytest.mark.parametrize("boundary", ["episode", "run", "state"])
def test_status_boundary_discards_an_unconsumed_abort(
    ros_message_stubs, boundary
):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _abort(node).success is True

    if boundary == "episode":
        crossed = _status(tracker, episode_id=1, env_step=1)
    elif boundary == "run":
        crossed = _status(ActorStatusTracker("new-run"), env_step=1)
    else:
        crossed = _status(tracker, state=WAIT_HOME_APPROVAL, env_step=1)
    session.publish(crossed)

    assert session.consume_operator_abort("run", 0) is False
    assert session.consume_operator_abort("new-run", 0) is False
    assert session.consume_operator_abort("run", 1) is False


def test_abort_token_survives_ordinary_status_publishes(ros_message_stubs):
    """Only boundaries clear the token; steady-state telemetry must not.

    The actor publishes a status every step, and an abort clicked mid-episode
    has to survive until the step that consumes it.
    """

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _abort(node).success is True

    for step in range(1, 5):
        session.publish(_status(tracker, env_step=step))
    session.publish(_status(tracker, state=HUMAN_INTERVENTION, env_step=5))

    assert session.consume_operator_abort("run", 0) is True


def test_abort_and_success_tokens_are_independent(ros_message_stubs):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))

    assert _manual_success(node).success is True
    assert _abort(node).success is True

    # Consuming one must not spend the other.
    assert session.consume_operator_abort("run", 0) is True
    assert session.consume_operator_success("run", 0) is True


def test_abort_can_be_requested_again_in_the_next_episode(ros_message_stubs):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _abort(node).success is True
    assert session.consume_operator_abort("run", 0) is True

    session.publish(_status(tracker, episode_id=1, env_step=1))
    second = _abort(node)

    assert second.success is True
    assert "episode=1" in second.message
    assert session.consume_operator_abort("run", 0) is False
    assert session.consume_operator_abort("run", 1) is True


def test_concurrent_abort_requests_queue_exactly_one_token(ros_message_stubs):
    """Two GUI clicks racing must not produce two consumable tokens."""

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))

    start = threading.Barrier(8)
    results = []

    def _click():
        start.wait()
        results.append(_abort(node).success)

    threads = [threading.Thread(target=_click) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert sum(1 for accepted in results if accepted) == 1
    assert session.consume_operator_abort("run", 0) is True
    assert session.consume_operator_abort("run", 0) is False


# ---- the terminal window: never acknowledge an abort that gets dropped ----- #


@pytest.mark.parametrize("state", sorted(ACTIVE_CONTROL_STATES))
@pytest.mark.parametrize("reason", ["SUCCESS", "TRUNCATED", "MAX_STEPS"])
def test_abort_on_a_terminal_transition_is_rejected_not_acknowledged(
    ros_message_stubs, state, reason
):
    """The one behaviour that must not survive: accept-then-silently-drop.

    The actor publishes its terminal transition while still in an *active*
    control state -- ``_control_state`` maps the action that physically ran to
    POLICY_RUNNING / HUMAN_INTERVENTION / HOLD -- and marks it terminal only
    through ``terminal_reason``.  The next publish is ``WAIT_HOME_APPROVAL``,
    which clears the token on the non-active-state rule in ``publish``.  So an
    abort accepted in this window is guaranteed to be dropped, and the GUI has
    already told the operator the episode would be discarded.

    Operationally this is exactly the click that rejects a false classifier
    SUCCESS.  Dropping it keeps the bad episode and labels it a win.
    """

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(
        _status(
            tracker,
            state=state,
            terminal_reason=reason,
            success=reason == "SUCCESS",
        )
    )

    response = _abort(node)

    assert response.success is False
    assert reason in response.message
    assert "already ended" in response.message
    # The message has to tell the operator what to do instead.
    assert "APPROVE HOME" in response.message
    # Nothing was latched, for this episode or the next one.
    assert session.consume_operator_abort("run", 0) is False
    assert session.consume_operator_abort("run", 1) is False


def test_rejected_terminal_abort_matches_the_actors_publish_sequence(
    ros_message_stubs,
):
    """End-to-end shape of the hole, including the boundary that ate the token.

    Publishing WAIT_HOME_APPROVAL after the terminal is what used to discard an
    accepted token; assert the rejection came from the terminal check itself
    and that the later boundary changes nothing.
    """

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker, state=POLICY_RUNNING, env_step=41))

    # Terminal transition: still POLICY_RUNNING, now carrying SUCCESS.
    session.publish(
        _status(
            tracker,
            state=POLICY_RUNNING,
            env_step=42,
            terminal_reason="SUCCESS",
            success=True,
        )
    )
    rejected = _abort(node)

    # ... then the actor parks the follower and enters the HOME gate.
    session.publish(
        _status(tracker, state=WAIT_HOME_APPROVAL, env_step=42)
    )

    assert rejected.success is False
    assert "SUCCESS" in rejected.message
    assert session.consume_operator_abort("run", 0) is False
    # A click landing after the gate opened is rejected too, by state.
    late = _abort(node)
    assert late.success is False
    assert WAIT_HOME_APPROVAL in late.message


def test_abort_is_accepted_again_once_the_next_episode_clears_the_reason(
    ros_message_stubs,
):
    """The cross-agent contract: a healthy episode must stay abortable.

    ``remote_actor`` clears ``terminal_reason`` back to "" on the first status
    of the next episode, so the terminal rejection must not leak forward and
    disarm ABORT for the rest of the run.
    """

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(
        _status(tracker, terminal_reason="SUCCESS", success=True)
    )
    assert _abort(node).success is False

    session.publish(_status(tracker, state=WAIT_HOME_APPROVAL, env_step=1))
    session.publish(
        _status(tracker, episode_id=1, env_step=1, terminal_reason="")
    )

    response = _abort(node)

    assert response.success is True
    assert "run=run episode=1" in response.message
    assert session.consume_operator_abort("run", 1) is True


def test_abort_queued_before_the_terminal_still_consumes_on_it(
    ros_message_stubs,
):
    """Rejecting new requests must not invalidate an already queued token.

    The actor consumes the abort token and only then publishes the terminal it
    caused (``OPERATOR_ABORT``).  The terminal check belongs to the request
    path alone; ``consume_operator_abort`` and ``publish`` are untouched.
    """

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _abort(node).success is True

    session.publish(
        _status(
            tracker,
            env_step=1,
            terminal_reason="OPERATOR_ABORT",
        )
    )

    assert session.consume_operator_abort("run", 0) is True


# ---- success-token invariants that abort silently depends on --------------- #


def test_manual_success_then_auto_toggle_then_abort_leaves_no_stale_success(
    ros_message_stubs,
):
    """The sequence abort's AUTO correctness rests on.

    Abort is accepted in AUTO.  If a MANUAL-queued success token could survive
    the toggle, the very transition the operator aborted could also carry an
    operator success -- the episode would be discarded and reported a win at
    the same time.  The guard is that every mode edge clears the success token.
    """

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))

    assert _manual_success(node).success is True
    assert _set_auto(node, True).success is True

    response = _abort(node)

    assert response.success is True
    assert session.auto_success is True
    # No stale success survived the mode edge ...
    assert session.consume_operator_success("run", 0) is False
    # ... and it cannot be re-queued while AUTO owns the verdict.
    assert _manual_success(node).success is False
    # The abort itself is intact and consumable exactly once.
    assert session.consume_operator_abort("run", 0) is True
    assert session.consume_operator_abort("run", 0) is False


@pytest.mark.parametrize("start_auto", [False, True])
def test_every_mode_edge_clears_a_queued_success(ros_message_stubs, start_auto):
    """Both directions, not just MANUAL -> AUTO.

    The AUTO -> MANUAL direction cannot be reached through the service today
    (``_on_manual_success`` refuses in AUTO), so the token is planted directly.
    That is the point: the clearing is defence in depth, and a future change
    that made a token reachable in AUTO must not silently make it survive.
    """

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _set_auto(node, start_auto).success is True

    session._operator_success_pending = ("run", 0)
    assert _set_auto(node, not start_auto).success is True

    assert session.consume_operator_success("run", 0) is False


def test_repeating_the_current_mode_is_not_an_edge_and_keeps_the_token(
    ros_message_stubs,
):
    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _manual_success(node).success is True

    repeated = _set_auto(node, False)

    assert repeated.success is True
    assert "already" in repeated.message
    assert session.consume_operator_success("run", 0) is True


@pytest.mark.parametrize("start_auto", [False, True])
def test_mode_edges_deliberately_do_not_clear_a_queued_abort(
    ros_message_stubs, start_auto
):
    """The intentional asymmetry with the success token -- do not "fix" it.

    A MANUAL/AUTO toggle is a statement about who owns the success verdict.  It
    says nothing about whether the episode is ruined, so the collision that
    made the operator hit ABORT is just as true after the toggle.  Clearing the
    abort token here by pattern-matching against ``_on_set_auto_success`` would
    silently discard an abort the operator already saw acknowledged.
    """

    node, session = _session()
    tracker = ActorStatusTracker("run")
    session.publish(_status(tracker))
    assert _set_auto(node, start_auto).success is True
    assert _abort(node).success is True

    assert _set_auto(node, not start_auto).success is True
    assert _set_auto(node, start_auto).success is True

    assert session.consume_operator_abort("run", 0) is True


# ---- resolve_follow_controls: the cross-agent contract --------------------- #


def test_resolve_follow_controls_returns_the_unwrapped_env():
    """An abort must be able to stop the 30 Hz background GELLO follower.

    G32 records that an operator holding GELLO keeps driving the arm through a
    blocking wait because nothing disarms the follower.  ABORT is the "operator
    is holding GELLO during an emergency" case, so the caller needs a handle on
    the base env, not on whatever wrapper the actor happens to hold.
    """

    base = SimpleNamespace(disarm_intervention_follow=lambda: None)
    wrapped = SimpleNamespace(env=SimpleNamespace(env=base), unwrapped=base)

    assert resolve_follow_controls(wrapped) is base


def test_resolve_follow_controls_accepts_a_bare_env_without_unwrapped():
    base = SimpleNamespace(disarm_intervention_follow=lambda: None)

    assert resolve_follow_controls(base) is base


@pytest.mark.parametrize(
    "base",
    [
        SimpleNamespace(),
        SimpleNamespace(disarm_intervention_follow=None),
        SimpleNamespace(disarm_intervention_follow="not callable"),
    ],
)
def test_resolve_follow_controls_rejects_an_env_that_cannot_disarm(base):
    wrapped = SimpleNamespace(unwrapped=base)

    with pytest.raises(RuntimeError, match="follow controls"):
        resolve_follow_controls(wrapped)

    with pytest.raises(RuntimeError, match="follow controls"):
        resolve_follow_controls(base)
