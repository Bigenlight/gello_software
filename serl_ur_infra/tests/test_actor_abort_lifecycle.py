"""Actor lifecycle tests for the operator ABORT EPISODE control.

ABORT discards the episode in flight -- after a collision, say -- and returns
the robot to the start pose without waiting for success or the step limit.  The
three properties these tests pin down are:

* the discarded transition is a TRUNCATION, never a Bellman terminal, so the
  critic bootstraps (``masks=1.0``) instead of learning that the world ends
  wherever the operator gave up;
* the background GELLO follower is disarmed BEFORE the loop enters anything
  that blocks -- the Step RPC and then ``WAIT_HOME_APPROVAL``, which never
  consults the deadman (``docs/testing/08_OPEN_GAPS.md`` G32) -- and a follower
  that will not park is FAIL-CLOSED at ``env.reset``, not homed around;
* the token is read twice per iteration, before ``env.step`` and after, so a
  press landing in the ~412 ms Step RPC window discards the pending policy
  action instead of watching it execute; and
* the ordinary episode boundary is reused unchanged, home-approval gate
  included, rather than short-cutting to the ``max_steps`` exhaustion branch
  that skips it.  ``terminal_reason`` survives that whole boundary and stops at
  the next episode's first status.

The network fake mirrors the real server's reward rules from
``rlpd_receive_server.RewardClassifierRuntime.__call__`` -- in particular that
an effective success OVERWRITES the client's proposal with
``masks=0.0, dones=True, truncated=False``.  That is what makes
``auto_success=False`` on an aborted transition load-bearing rather than
cosmetic, and a fake that simply echoed the client's flags would not show it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import TransitionOutcome  # noqa: E402
from ur_env.operator_session import (  # noqa: E402
    FAULT,
    HOMING,
    POLICY_RUNNING,
    STOPPED,
    WAIT_HOME_APPROVAL,
    WAIT_SCENE_READY,
)
from ur_env.remote_actor import _park_follower, run_remote_actor  # noqa: E402


class _ActionSpace:
    shape = (7,)


def _observation(marker):
    value = np.uint8(marker % 255)
    return {
        "state": np.full((1, 19), float(marker), dtype=np.float32),
        "cam1": np.full((1, 4, 4, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 4, 4, 3), value, dtype=np.uint8),
    }


class _FollowControls:
    """Stand-in for the base ``UR7eEnv`` follower controls.

    Both calls are modelled because they are not equivalent: the real
    ``disarm_intervention_follow`` only locks the follower out, while
    ``await_follower_quiescent`` additionally waits for it to PARK.  Only the
    latter keeps a 30 Hz follower from out-voting ``go_to_reset`` in a
    last-write-wins backend, which is why the actor must prefer it.
    """

    def __init__(self, events, *, quiescent=True):
        self.events = events
        self.disarm_calls = 0
        self.quiesce_calls = 0
        self.quiescent = quiescent

    def disarm_intervention_follow(self):
        self.disarm_calls += 1
        self.events.append("disarm")

    def await_follower_quiescent(self, timeout=None):
        self.quiesce_calls += 1
        self.events.append("quiesce")
        self.disarm_intervention_follow()  # the real one disarms internally
        return self.quiescent


class _DisarmOnlyControls:
    """Older/partial controls: ``resolve_follow_controls``' minimum only."""

    def __init__(self, events):
        self.events = events
        self.disarm_calls = 0

    def disarm_intervention_follow(self):
        self.disarm_calls += 1
        self.events.append("disarm")


class _Env:
    """Fake env exposing the follower controls behind ``unwrapped``.

    ``resolve_follow_controls`` deliberately resolves through ``unwrapped``
    rather than walking the wrapper chain, so the fake has to be shaped the same
    way for the test to exercise the real resolution path.
    """

    action_space = _ActionSpace()

    def __init__(
        self,
        events=None,
        *,
        with_follower=True,
        controls_factory=_FollowControls,
        infos=None,
    ):
        self.events = events if events is not None else []
        self.reset_count = 0
        self.reset_options = []
        self.step_count = 0
        self.infos = list(infos or [])
        self.follow = controls_factory(self.events) if with_follower else None

    @property
    def unwrapped(self):
        return self.follow if self.follow is not None else self

    def reset(self, **kwargs):
        # FAIL-CLOSED, exactly as ``UR7eEnv.reset`` is (ur7e_env.py:1585-1607):
        # it calls ``await_follower_quiescent`` itself and RAISES rather than
        # stream ``go_to_reset``'s 20 Hz target into a backend that a live 30 Hz
        # follower would keep winning.  Without this the fake cannot reproduce
        # the real failure at all, and a test written against it pins fiction --
        # which is precisely what happened: ``_park_follower`` swallows a park
        # timeout, the old fake homed happily, and the test asserted an outcome
        # the robot never produces.  The check reads the flag rather than
        # re-entering ``await_follower_quiescent`` so that the counters and
        # events on ``_FollowControls`` keep meaning "what the ACTOR did".
        if self.follow is not None and not getattr(self.follow, "quiescent", True):
            raise RuntimeError(
                "intervention follower did not confirm it stopped within "
                "2.0s - refusing to reset, because go_to_reset() streams its "
                "target at 20 Hz and a live follower at 30 Hz would win every "
                "write"
            )
        self.reset_count += 1
        self.reset_options.append(kwargs.get("options"))
        self.events.append("reset")
        return _observation(self.reset_count), {
            "timestamp_ns": np.int64(1_000 + self.reset_count)
        }

    def step(self, action):
        index = self.step_count
        self.step_count += 1
        self.events.append("step")
        info = {
            "timestamp_ns": np.int64(2_000 + self.step_count),
            "intervened": 0,
            "held": False,
        }
        if index < len(self.infos):
            info.update(self.infos[index])
        if info.get("intervened") and "intervene_action" not in info:
            info["intervene_action"] = np.asarray(action, dtype=np.float32).copy()
        return _observation(100 + self.step_count), 0.0, False, False, info


class _ServerLikeNetwork:
    """Apply the server's finalization AND session rules to what the client ships.

    Mirrors ``RewardClassifierRuntime.__call__``: server-authoritative reward,
    ``effective_success = operator_success or (auto_success and classifier)``,
    and the success override of ``masks``/``dones``/``truncated``.  A
    non-success proposal passes through untouched, which is the behaviour the
    abort depends on.

    Also mirrors ``ActorSessionService``'s session registry
    (``actor_network.py:612-619`` / ``:802-810``): one live session per run
    key, released ONLY by a terminal Step -- the proto has no close RPC.  This
    is load-bearing.  An earlier abort path ended an episode without shipping
    anything and then reopened; a fake without these two refusals certified it,
    and the real server killed the actor with ``FAILED_PRECONDITION: session
    ... is still active`` on the operator's next START press (2026-07-31).
    """

    def __init__(self, events=None, *, classifier_success=False):
        self.events = events if events is not None else []
        self.classifier_success = bool(classifier_success)
        self.begin_calls = []
        self.step_calls = []
        self.active_session_id = ""
        self.next_episode_id = 0

    @staticmethod
    def _action(version=0):
        return SimpleNamespace(
            action=np.zeros(7, dtype=np.float32),
            policy_version=version,
        )

    def begin_episode(self, observation, **kwargs):
        if self.active_session_id:
            raise RuntimeError(
                f"session {self.active_session_id!r} is still active"
            )
        episode_id = kwargs.get("episode_id")
        if episode_id != self.next_episode_id:
            raise RuntimeError(
                f"episode_id must be {self.next_episode_id}, got {episode_id}"
            )
        self.active_session_id = kwargs.get("session_id") or "unnamed"
        self.begin_calls.append((observation, dict(kwargs)))
        return self._action(len(self.begin_calls) - 1)

    def step(self, next_observation, **kwargs):
        self.events.append("step_rpc")
        self.step_calls.append((next_observation, dict(kwargs)))
        data = kwargs["data"]
        meta = data["meta"]
        transition = data["transition"]
        success = bool(meta["operator_success"]) or (
            bool(meta["auto_success"]) and self.classifier_success
        )
        if success:
            mask, done, truncated = 0.0, True, False
        else:
            mask = float(transition["masks"])
            done = bool(transition["dones"])
            truncated = bool(transition["truncated"])
        outcome = TransitionOutcome(
            transition_id=meta["transition_id"],
            reward=1.0 if success else 0.0,
            mask=mask,
            done=done,
            truncated=truncated,
            success=success,
            classifier_evaluated=False,
            classifier_probability=0.0,
            classifier_threshold=0.0,
            reward_model_id="",
        )
        if done or truncated:
            # The finalized outcome, not the client's proposal, is what
            # releases the session -- same as ``actor_network.py:802-810``.
            self.active_session_id = ""
            self.next_episode_id += 1
        action = None if (done or truncated) else self._action(1)
        return SimpleNamespace(outcome=outcome, action=action)


class _Operator:
    """Immediate (non-blocking) operator session with a ONE-SHOT abort token.

    One-shot exactly like ``RosOperatorSession.consume_operator_abort``: the
    first MATCHING read spends the token and every later read returns False.
    The actor reads the token exactly once per iteration, after ``env.step``,
    so WHICH iteration's read spends it is the behaviour under test.

    ``abort_press_offset`` says WHEN the operator pressed, counted in matching
    reads that go by first:

    * ``0`` (default) -- pressed any time up to the end of the episode's first
      ``env.step`` (scene-ready gate, BeginEpisode, or the step itself).  The
      first iteration's read takes it: a real transition happened and is
      relabelled a truncation.
    * ``1`` -- pressed during the first iteration's blocking Step RPC.  The
      read of that iteration has already returned, so the FOLLOWING iteration
      executes the pending policy action and its read takes the token: the
      abort costs at most one more bounded action, by design.
    """

    auto_success = False

    def __init__(
        self,
        events=None,
        *,
        abort_at=None,
        success_at=None,
        abort_press_offset=0,
    ):
        self.events = events if events is not None else []
        self.statuses = []
        self.wait_statuses = []
        self.home_wait_statuses = []
        self.abort_at = abort_at
        self.success_at = success_at
        self.abort_calls = []
        self.success_calls = []
        self._abort_reads_until_press = int(abort_press_offset)
        self._abort_spent = False

    def publish(self, status):
        self.statuses.append(status)

    def wait_for_scene_ready(self, status):
        self.events.append("wait_scene")
        self.wait_statuses.append(status)
        self.statuses.append(status)

    def wait_for_home_approval(self, status):
        self.events.append("wait_home")
        self.home_wait_statuses.append(status)
        self.statuses.append(status)

    def consume_operator_abort(self, run_id, episode_id):
        self.abort_calls.append((run_id, episode_id))
        if self._abort_spent or (run_id, episode_id) != self.abort_at:
            return False
        if self._abort_reads_until_press > 0:
            # Latched, but not yet: the operator has not pressed at this point
            # in the loop.  A mismatch of timing, not of episode.
            self._abort_reads_until_press -= 1
            return False
        self._abort_spent = True
        return True

    def consume_operator_success(self, run_id, episode_id):
        self.success_calls.append((run_id, episode_id))
        return (run_id, episode_id) == self.success_at


class _LegacyOperator(_Operator):
    """A session predating the abort control: no ``consume_operator_abort``."""

    consume_operator_abort = None


def _config(max_steps):
    return SimpleNamespace(max_steps=max_steps, random_steps=0, buffer_period=0)


def _run(env, network, operator, *, max_steps=2, run_id="run"):
    return run_remote_actor(
        network,
        env,
        config=_config(max_steps),
        actor_id="actor",
        run_id=run_id,
        session_id_factory=iter(("s0", "s1", "s2", "s3")).__next__,
        operator_session=operator,
    )


def _shipped(network, index):
    return network.step_calls[index][1]["data"]


# --------------------------------------------------------------------- #
# The transition an abort produces
# --------------------------------------------------------------------- #


def test_abort_ships_truncated_bootstrap_never_a_bellman_terminal():
    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0))

    _run(env, network, operator)

    transition = _shipped(network, 0)["transition"]
    assert transition["dones"] is False
    assert transition["truncated"] is True
    # masks=1.0 is the whole point: bootstrap from the next state rather than
    # bootstrapping zero, which is what dones=True would mean to the critic.
    assert transition["masks"] == pytest.approx(1.0)
    assert transition["success"] is False
    assert transition["rewards"] == pytest.approx(0.0)

    meta = _shipped(network, 0)["meta"]
    assert meta["auto_success"] is False
    assert meta["operator_success"] is False


def test_abort_requests_no_action_and_reports_operator_abort():
    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0))

    _run(env, network, operator)

    # A terminal Step must not ask for an action it will never execute.
    assert network.step_calls[0][1]["request_action"] is False
    assert operator.home_wait_statuses[0].terminal_reason == "OPERATOR_ABORT"
    # ...but the label stops at the episode boundary: the next episode's first
    # status is clean.  See test_terminal_reason_does_not_decorate_the_next_episode.
    assert operator.wait_statuses[1].terminal_reason == ""
    assert operator.home_wait_statuses[0].success is False


def test_abort_outranks_a_queued_mark_success():
    """ABORT wins, and the queued success token is still spent, not left latched."""

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    # Same (run, episode): the operator queued MARK SUCCESS and then hit ABORT.
    operator = _Operator(events, abort_at=("run", 0), success_at=("run", 0))

    _run(env, network, operator)

    # Consumed -- otherwise the stale click would resolve the NEXT episode.
    assert operator.success_calls == [("run", 0), ("run", 1)]
    meta = _shipped(network, 0)["meta"]
    assert meta["operator_success"] is False  # ...but its value is discarded
    transition = _shipped(network, 0)["transition"]
    assert transition["dones"] is False
    assert transition["truncated"] is True
    assert transition["success"] is False
    assert operator.home_wait_statuses[0].terminal_reason == "OPERATOR_ABORT"


def test_abort_in_auto_mode_is_not_converted_into_a_success():
    """A classifier firing on the abort frame must not resurrect the episode.

    The server overwrites ``masks``/``dones``/``truncated`` on an effective
    success, so forcing ``auto_success=False`` client-side is the only thing
    keeping the truncation.
    """

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events, classifier_success=True)
    operator = _Operator(events, abort_at=("run", 0))
    operator.auto_success = True

    _run(env, network, operator)

    assert _shipped(network, 0)["meta"]["auto_success"] is False
    transition = _shipped(network, 0)["transition"]
    assert transition["truncated"] is True
    assert transition["dones"] is False
    assert transition["masks"] == pytest.approx(1.0)
    assert transition["success"] is False

    # The success token is spent even in AUTO, where nothing else reads it.
    # Otherwise abort's correctness would rest on operator_session clearing the
    # token on every MANUAL/AUTO edge -- an invariant in a different file.  The
    # episode-1 step is AUTO and non-aborted, so it does NOT read the token.
    assert operator.success_calls == [("run", 0)]

    # Control: the same network DOES convert an ordinary AUTO step into a
    # success, so the assertion above is about the abort and not about a fake
    # that can never succeed.
    assert _shipped(network, 1)["meta"]["auto_success"] is True
    assert bool(network.step_calls[1][1]["data"]["transition"]["success"]) is True


# --------------------------------------------------------------------- #
# Safety ordering (G32)
# --------------------------------------------------------------------- #


def test_abort_stops_follower_before_anything_that_can_block():
    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0))

    _run(env, network, operator)

    first_stop = events.index("quiesce")
    # The Step RPC blocks for ~412 ms of the 512 ms loop, and WAIT_HOME_APPROVAL
    # blocks indefinitely and never checks the deadman.  Both must come after.
    assert first_stop < events.index("step_rpc")
    assert first_stop < events.index("wait_home")
    # ...and after the env.step that produced the aborted transition, because
    # the token is only readable once that step has returned.
    assert events.index("step") < first_stop


def test_abort_waits_for_the_follower_to_park_not_merely_disarm():
    """A disarmed-but-still-publishing follower out-votes the HOME move.

    ``go_to_reset`` republishes one target at 20 Hz; a 30 Hz follower wins in a
    last-write-wins backend, the arm never converges, and the run dies with
    "reset did not arrive".  So the abort must WAIT for the park, and it must do
    so before HOME is attempted.

    Counting calls does NOT pin that.  An earlier version of this test asserted
    ``quiesce_calls == 2`` against a fake that returned instantly, so deleting
    the wait and keeping the call would have changed nothing and the test would
    still have passed.  This follower therefore BLOCKS inside the park, and the
    test proves the actor is genuinely stuck behind it -- Step RPC unsent, home
    gate unreached, HOME unstarted -- until it is released.
    """

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0))

    parking = threading.Event()
    parked = threading.Event()
    inner = env.follow.await_follower_quiescent

    def _blocking_park(timeout=None):
        parking.set()
        if not parked.wait(timeout=2.0):
            raise AssertionError("the test never released the park")
        return inner(timeout)

    env.follow.await_follower_quiescent = _blocking_park

    failures = []

    def _drive():
        try:
            _run(env, network, operator, max_steps=1)
        except BaseException as exc:  # surfaced by the assertion below
            failures.append(exc)

    thread = threading.Thread(target=_drive, daemon=True)
    thread.start()
    try:
        assert parking.wait(timeout=2.0), "the abort never parked the follower"
        time.sleep(0.02)
        # Blocked INSIDE the park.  Everything the park exists to protect is
        # still ahead of it.
        assert "quiesce" not in events
        assert "step_rpc" not in events
        assert "wait_home" not in events
        assert env.reset_options[-1] is None  # HOME has not started
    finally:
        parked.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert not failures, failures

    # Released: the ordering holds all the way to HOME, and the park ran twice
    # (abort point, then the shared terminal path).  It is idempotent.
    assert env.follow.quiesce_calls == 2
    assert events.index("quiesce") < events.index("step_rpc")
    assert events.index("quiesce") < events.index("wait_home")
    assert env.reset_options[-1] == {"operator_approved_home": True}


def test_follower_stop_falls_back_to_disarm_when_quiescence_is_unavailable():
    """``resolve_follow_controls`` only promises ``disarm_intervention_follow``."""

    events = []
    env = _Env(events, controls_factory=_DisarmOnlyControls)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0))

    _run(env, network, operator, max_steps=1)

    assert env.follow.disarm_calls == 2  # abort point + shared terminal path
    assert "quiesce" not in events
    assert events.index("disarm") < events.index("wait_home")


def test_follower_that_fails_to_park_faults_instead_of_homing_blind(capsys):
    """Fail-closed: a follower that will not park REFUSES the HOME move.

    ``_park_follower`` only warns -- it is not the layer that decides, and it
    returns nothing for a caller to act on.  The refusal belongs to
    ``UR7eEnv.reset``, which calls ``await_follower_quiescent`` itself and
    raises rather than stream a 20 Hz HOME target against a live 30 Hz
    follower; that exception reaches ``run_remote_actor``'s handler, which
    publishes FAULT and re-raises.

    The previous version of this test asserted the episode was "still discarded
    and homed".  It passed only because the fake env had no such refusal, so it
    pinned behaviour the real robot contradicts -- worse than no test.
    """

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)

    class _WedgingOperator(_Operator):
        def consume_operator_abort(self, run_id, episode_id):
            aborted = super().consume_operator_abort(run_id, episode_id)
            if aborted:
                # The hand is on GELLO and the follower thread wedges.  Setting
                # the flag here rather than up front matters: a follower that
                # never parks would refuse the STARTUP home too, and the abort
                # path would never be reached.
                env.follow.quiescent = False
            return aborted

    operator = _WedgingOperator(events, abort_at=("run", 0))

    with pytest.raises(RuntimeError, match="did not confirm it stopped"):
        _run(env, network, operator, max_steps=1)

    # Warned first, at the abort point, naming the follower as the cause...
    assert "did not park" in capsys.readouterr().err
    # ...the transition is still shipped, so the fault costs no data...
    assert _shipped(network, 0)["transition"]["truncated"] is True
    # ...but nothing moved the arm afterwards, and the operator was told.
    assert {"operator_approved_home": True} not in env.reset_options
    assert operator.statuses[-1].state == FAULT
    assert operator.statuses[-1].terminal_reason == "FAULT"


def test_park_follower_reports_nothing_a_caller_could_act_on():
    """The return value was discarded by both call sites and lied on timeout.

    It returned True even when the follower failed to park.  Rather than make it
    truthful, it is gone: neither caller could honour a False without either
    dropping a legitimate transition (the abort still has one to ship) or
    duplicating ``UR7eEnv.reset``'s refusal badly.  Pinned so it does not come
    back.
    """

    events = []
    wedged = _Env(events)
    wedged.follow.quiescent = False

    assert _park_follower(wedged) is None
    assert _park_follower(_Env(events, controls_factory=_DisarmOnlyControls)) is None
    assert _park_follower(_Env(events, with_follower=False)) is None


def test_abort_survives_an_env_with_no_follower_controls():
    """Fake/bring-up envs have no follower thread and so no hazard to close."""

    events = []
    env = _Env(events, with_follower=False)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0))

    _run(env, network, operator)

    assert "disarm" not in events
    assert "quiesce" not in events
    assert _shipped(network, 0)["transition"]["truncated"] is True
    assert len(operator.home_wait_statuses) == 1


def test_ordinary_terminal_also_stops_the_follower():
    """PIGGYBACK: the shared terminal path, not only abort.

    Judged independently of the abort feature: SUCCESS and EPISODE_LIMIT enter
    the same deadman-blind WAIT_HOME_APPROVAL, so they carry the same G32
    hazard.
    """

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, success_at=("run", 0))

    _run(env, network, operator, max_steps=1)

    assert env.follow.quiesce_calls == 1
    assert events.index("quiesce") < events.index("wait_home")
    assert operator.home_wait_statuses[0].terminal_reason == "SUCCESS"


# --------------------------------------------------------------------- #
# Episode boundary reuse
# --------------------------------------------------------------------- #


def test_abort_traverses_the_home_approval_gate_and_reopens_an_episode():
    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0))

    summary = _run(env, network, operator)

    # The gate is traversed, not skipped: HOME motion only after approval, and
    # the reset carries the operator's approval option.  This is what separates
    # the abort from the max_steps exhaustion branch (gap G31).
    assert len(operator.home_wait_statuses) == 1
    approved = [
        options
        for options in env.reset_options
        if options == {"operator_approved_home": True}
    ]
    assert approved == [{"operator_approved_home": True}]
    assert events.index("wait_home") < events.index("reset", events.index("wait_home"))

    # Episode id advanced and a new episode was opened after the scene reset.
    assert summary.episodes_started == 2
    assert network.begin_calls[1][1]["episode_id"] == 1
    # One read per iteration, each against the episode that was live when it
    # ran.
    assert operator.abort_calls == [("run", 0), ("run", 1)]

    states = [status.state for status in operator.statuses]
    assert HOMING in states
    assert POLICY_RUNNING in states
    assert states[-1] == STOPPED


def test_abort_on_the_final_step_stops_without_reopening_an_episode():
    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0))

    summary = _run(env, network, operator, max_steps=1)

    assert summary.episodes_started == 1
    assert len(operator.home_wait_statuses) == 1
    assert env.reset_options[-1] == {"operator_approved_home": True}
    assert operator.statuses[-1].state == STOPPED


def test_abort_in_a_later_episode_uses_that_episodes_token():
    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 1))

    _run(env, network, operator, max_steps=2)

    # Episode 0 ran to the end of the run without a terminal; episode 1 never
    # started, so only the episode-0 token was offered -- once per iteration --
    # and it never matched.
    assert operator.abort_calls == [("run", 0)] * 2
    assert _shipped(network, 0)["transition"]["truncated"] is False
    assert len(operator.home_wait_statuses) == 0


# --------------------------------------------------------------------- #
# terminal_reason: sticky across the terminal window, gone by the next episode
# --------------------------------------------------------------------- #


def test_terminal_reason_does_not_decorate_the_next_episode():
    """Both halves of the cross-agent contract, in one status sequence.

    NON-EMPTY on the terminal publish and across the whole home-approval window:
    that is where the operator reads WHY the robot stopped, and
    ``operator_session`` rejects a new ABORT request while it is set -- which is
    what stops a press at the home gate from being accepted, acknowledged, and
    then dropped into an episode that is already over.

    EMPTY again on the first status of the NEXT episode: it is a one-shot signal
    to the GUI, and left sticky the GUI's confirmation latch is spent by the
    fresh episode's own WAIT_SCENE_READY/HOMING statuses before that episode has
    taken a single step.
    """

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0))

    _run(env, network, operator, max_steps=2)

    labelled = [
        (status.state, status.terminal_reason) for status in operator.statuses
    ]
    terminal = labelled.index((POLICY_RUNNING, "OPERATOR_ABORT"))
    assert labelled[terminal : terminal + 3] == [
        (POLICY_RUNNING, "OPERATOR_ABORT"),  # the terminal publish itself
        (WAIT_HOME_APPROVAL, "OPERATOR_ABORT"),  # ...and the whole home gate
        (HOMING, "OPERATOR_ABORT"),
    ]
    # First status of the next episode, and everything after it up to the run's
    # own MAX_STEPS stop.
    assert labelled[terminal + 3] == (WAIT_SCENE_READY, "")
    assert labelled[terminal + 3 : terminal + 7] == [
        (WAIT_SCENE_READY, ""),
        (HOMING, ""),
        (POLICY_RUNNING, ""),
        (POLICY_RUNNING, ""),
    ]
    assert operator.wait_statuses[1].terminal_reason == ""


def test_a_second_abort_in_the_same_run_is_labelled_again():
    """The GUI's one-shot confirmation must re-arm between episodes."""

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)

    class _AbortsEveryEpisode(_Operator):
        """One press per episode, each landing in that episode's first step."""

        def __init__(self, inner_events):
            super().__init__(inner_events)
            self._spent = set()

        def consume_operator_abort(self, run_id, episode_id):
            self.abort_calls.append((run_id, episode_id))
            key = (run_id, episode_id)
            if key in self._spent:
                return False
            self._spent.add(key)
            return True

    operator = _AbortsEveryEpisode(events)

    _run(env, network, operator, max_steps=2)

    assert [
        status.terminal_reason for status in operator.home_wait_statuses
    ] == ["OPERATOR_ABORT", "OPERATOR_ABORT"]
    reasons = [status.terminal_reason for status in operator.statuses]
    first = reasons.index("OPERATOR_ABORT")
    last = len(reasons) - 1 - reasons[::-1].index("OPERATOR_ABORT")
    # The label went empty in between, so a GUI latch keyed on the transition
    # to a non-empty terminal_reason fires twice rather than once.
    assert "" in reasons[first:last]


# --------------------------------------------------------------------- #
# Non-abort behaviour is unchanged
# --------------------------------------------------------------------- #


def test_mismatched_token_is_offered_but_not_spent():
    """A token latched on another run/episode must not be stolen."""

    class _CountingOperator(_Operator):
        def __init__(self, events):
            super().__init__(events)
            self.pending = ("other-run", 7)

        def consume_operator_abort(self, run_id, episode_id):
            self.abort_calls.append((run_id, episode_id))
            if (run_id, episode_id) != self.pending:
                return False  # mismatched call does not clear the token
            self.pending = None
            return True

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _CountingOperator(events)

    _run(env, network, operator, max_steps=2)

    assert operator.abort_calls == [("run", 0)] * 2  # one read per iteration
    assert operator.pending == ("other-run", 7)  # still latched
    assert "quiesce" not in events  # no terminal at all, so no follower stop
    for index in range(2):
        assert _shipped(network, index)["transition"]["truncated"] is False
        assert _shipped(network, index)["transition"]["dones"] is False
    assert len(operator.home_wait_statuses) == 0


def test_no_abort_is_byte_identical_to_a_session_without_the_control():
    """An un-pressed abort changes nothing an older session would have done."""

    def run(operator_factory):
        events = []
        env = _Env(events)
        network = _ServerLikeNetwork(events)
        operator = operator_factory(events)
        summary = _run(env, network, operator, max_steps=2)
        return env, network, operator, summary, events

    new_env, new_net, new_op, new_summary, new_events = run(_Operator)
    old_env, old_net, old_op, old_summary, old_events = run(_LegacyOperator)

    assert new_events == old_events
    assert new_env.reset_count == old_env.reset_count
    assert new_env.reset_options == old_env.reset_options
    # Wall-clock RPC latency fields legitimately differ run to run; everything
    # that describes what the actor DID must not.
    timing = {
        "sidecar_round_trip_ms_mean",
        "sidecar_round_trip_ms_max",
        "plain_round_trip_ms_mean",
        "plain_round_trip_ms_max",
    }
    for field in type(new_summary).__dataclass_fields__:
        if field in timing:
            continue
        assert getattr(new_summary, field) == getattr(old_summary, field), field
    assert [status.state for status in new_op.statuses] == [
        status.state for status in old_op.statuses
    ]
    assert [status.terminal_reason for status in new_op.statuses] == [
        status.terminal_reason for status in old_op.statuses
    ]

    assert len(new_net.step_calls) == len(old_net.step_calls)
    for new_call, old_call in zip(new_net.step_calls, old_net.step_calls):
        new_data, old_data = new_call[1]["data"], old_call[1]["data"]
        assert new_call[1]["request_action"] == old_call[1]["request_action"]
        for section in ("meta", "transition"):
            new_section = new_data[section]
            old_section = old_data[section]
            assert set(new_section) == set(old_section), section
            for key, new_value in new_section.items():
                old_value = old_section[key]
                if isinstance(new_value, np.ndarray):
                    assert np.array_equal(new_value, old_value), key
                else:
                    assert new_value == old_value, key
                    assert type(new_value) is type(old_value), key


def test_no_operator_session_never_reaches_the_abort_path():
    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)

    run_remote_actor(
        network,
        env,
        config=_config(1),
        actor_id="actor",
        run_id="run",
        session_id_factory=lambda: "s0",
    )

    assert "quiesce" not in events
    assert _shipped(network, 0)["transition"]["truncated"] is False


# --------------------------------------------------------------------- #
# Latency: "immediate" is bounded by one loop period plus one action
# --------------------------------------------------------------------- #


def test_press_during_the_step_rpc_costs_exactly_one_more_action():
    """A press in the RPC window executes ONE more action, then truncates it.

    This is the accepted trade, not an accident.  The read has to sit after
    ``env.step`` because a terminal Step is the server's only session-close
    signal, and a terminal Step must carry a transition that really happened
    (``transition.actions == policy_action`` for non-intervened rows).  An
    earlier pre-step read that discarded the pending action closed the episode
    with nothing to ship, left the server session latched, and killed the
    actor on the next BeginEpisode (2026-07-31).  The extra action is bounded
    by the governor and the workspace box, and the GUI's deadman release stops
    GELLO follow independently of this read.
    """

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0), abort_press_offset=1)

    _run(env, network, operator, max_steps=2)

    # The pending action DID run -- and its transition is the truncated one.
    assert events.count("step") == 2
    assert env.step_count == 2
    assert operator.abort_calls == [("run", 0)] * 2
    assert len(network.step_calls) == 2
    assert _shipped(network, 0)["transition"]["truncated"] is False
    assert _shipped(network, 1)["transition"]["truncated"] is True
    assert _shipped(network, 1)["transition"]["dones"] is False
    # The episode still ends properly: parked, home gate, HOME.
    assert events.index("quiesce") < events.index("wait_home")
    assert operator.home_wait_statuses[0].terminal_reason == "OPERATOR_ABORT"
    assert operator.home_wait_statuses[0].success is False
    assert env.reset_options[-1] == {"operator_approved_home": True}


def test_deferred_abort_is_a_full_episode_boundary():
    """The RPC-window press closes the episode and the NEXT one opens cleanly.

    The session-aware fake makes this the regression test for the 2026-07-31
    incident: reopening without a shipped terminal Step raises
    ``session ... is still active`` here, exactly like the real server.
    """

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(events, abort_at=("run", 0), abort_press_offset=1)

    summary = _run(env, network, operator, max_steps=3)

    # Iteration 0 stepped episode 0; iteration 1 executed the pending action
    # and truncated it; iteration 2 stepped the NEW episode.
    assert events.count("step") == 3
    assert summary.episodes_started == 2
    assert network.begin_calls[1][1]["episode_id"] == 1
    assert _shipped(network, 2)["transition"]["episode_id"] == 1
    assert _shipped(network, 2)["transition"]["step_id"] == 0
    # The token is one-shot: the new episode's step was not aborted too.
    assert _shipped(network, 2)["transition"]["truncated"] is False
    assert len(operator.home_wait_statuses) == 1


def test_every_abort_ships_the_terminal_step_that_releases_the_session():
    """No abort shape may end an episode with the server session still live.

    Presses ahead of the first step (scene-ready gate, BeginEpisode) are
    honoured after that step, as a truncation of it -- there is no longer a
    shape in which an episode ends having shipped nothing.  The queued MARK
    SUCCESS is spent and discarded on the same terminal.
    """

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)
    operator = _Operator(
        events,
        abort_at=("run", 0),
        success_at=("run", 0),
        abort_press_offset=0,
    )

    _run(env, network, operator, max_steps=1)

    assert operator.success_calls == [("run", 0)]
    assert events.count("step") == 1
    assert len(network.step_calls) == 1
    assert _shipped(network, 0)["transition"]["truncated"] is True
    assert _shipped(network, 0)["meta"]["operator_success"] is False
    # The terminal Step released the server's session slot -- the property the
    # 2026-07-31 incident violated.
    assert network.active_session_id == ""
    assert network.next_episode_id == 1
    # It is still an episode boundary, with the follower parked ahead of the
    # deadman-blind home gate.
    assert events.index("quiesce") < events.index("wait_home")
    assert operator.home_wait_statuses[0].terminal_reason == "OPERATOR_ABORT"
    assert env.reset_options[-1] == {"operator_approved_home": True}


def test_blocking_home_wait_does_not_delay_the_follower_stop():
    """The follower is already parked while the operator gate is still open."""

    events = []
    env = _Env(events)
    network = _ServerLikeNetwork(events)

    entered = threading.Event()
    release = threading.Event()

    class _BlockingOperator(_Operator):
        def wait_for_home_approval(self, status):
            self.events.append("wait_home")
            self.home_wait_statuses.append(status)
            self.statuses.append(status)
            entered.set()
            if not release.wait(timeout=2.0):
                raise RuntimeError("test did not release the home wait")

    operator = _BlockingOperator(events, abort_at=("run", 0))
    thread = threading.Thread(
        target=lambda: _run(env, network, operator, max_steps=1), daemon=True
    )
    thread.start()
    try:
        assert entered.wait(timeout=2.0), "actor never reached the home gate"
        time.sleep(0.02)
        # Held at the terminal pose with the gate open -- and already parked.
        assert env.follow.quiesce_calls >= 1
        assert env.reset_options[-1] is None  # HOME has NOT started yet
    finally:
        release.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert env.reset_options[-1] == {"operator_approved_home": True}
