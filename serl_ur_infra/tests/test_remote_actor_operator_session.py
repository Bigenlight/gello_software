"""Actor state-machine tests for HOME / scene-ready / policy episodes."""

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
    HOLD,
    HOMING,
    HUMAN_INTERVENTION,
    POLICY_RUNNING,
    STOPPED,
    WAIT_HOME_APPROVAL,
    WAIT_SCENE_READY,
)
from ur_env import remote_actor as remote_actor_module  # noqa: E402
from ur_env.remote_actor import run_remote_actor  # noqa: E402


class _ActionSpace:
    shape = (7,)


def _observation(marker):
    value = np.uint8(marker % 255)
    return {
        "state": np.full((1, 19), float(marker), dtype=np.float32),
        "cam1": np.full((1, 4, 4, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 4, 4, 3), value, dtype=np.uint8),
    }


class _Env:
    action_space = _ActionSpace()

    def __init__(self, infos=None, failure=None, reset_failure_at=None):
        self.reset_count = 0
        self.reset_options = []
        self.step_count = 0
        self.infos = list(infos or [])
        self.failure = failure
        self.reset_failure_at = reset_failure_at

    def reset(self, **kwargs):
        self.reset_count += 1
        self.reset_options.append(kwargs.get("options"))
        if self.reset_count == self.reset_failure_at:
            raise RuntimeError("robot HOME failed")
        return _observation(self.reset_count), {
            "timestamp_ns": np.int64(1_000 + self.reset_count)
        }

    def step(self, action):
        if self.failure is not None:
            raise self.failure
        index = self.step_count
        self.step_count += 1
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


def _outcome(
    transition_id,
    *,
    success=False,
    done=False,
    truncated=False,
    evaluated=False,
    probability=0.0,
    threshold=0.0,
):
    return TransitionOutcome(
        transition_id=transition_id,
        reward=1.0 if success else 0.0,
        mask=0.0 if done else 1.0,
        done=bool(done),
        truncated=bool(truncated),
        success=bool(success),
        classifier_evaluated=bool(evaluated),
        classifier_probability=float(probability),
        classifier_threshold=float(threshold),
        reward_model_id="reward" if evaluated else "",
    )


class _Network:
    def __init__(self, outcome_factories):
        self.outcome_factories = list(outcome_factories)
        self.begin_calls = []
        self.step_calls = []

    @staticmethod
    def _action(version=0):
        return SimpleNamespace(
            action=np.zeros(7, dtype=np.float32),
            policy_version=version,
        )

    def begin_episode(self, observation, **kwargs):
        self.begin_calls.append((observation, dict(kwargs)))
        return self._action(len(self.begin_calls) - 1)

    def step(self, next_observation, **kwargs):
        index = len(self.step_calls)
        self.step_calls.append((next_observation, dict(kwargs)))
        transition_id = kwargs["data"]["meta"]["transition_id"]
        outcome = self.outcome_factories[index](transition_id)
        action = None if outcome.terminal else self._action(index + 1)
        return SimpleNamespace(outcome=outcome, action=action)


class _BlockingOperatorSession:
    def __init__(self, waits):
        self.statuses = []
        self.wait_statuses = []
        self.home_wait_statuses = []
        self.entered = [threading.Event() for _ in range(waits)]
        self.release = [threading.Event() for _ in range(waits)]

    def publish(self, status):
        self.statuses.append(status)

    def _wait(self, status, index):
        self.statuses.append(status)
        self.entered[index].set()
        if not self.release[index].wait(timeout=2.0):
            raise RuntimeError("test did not release operator wait")

    def wait_for_scene_ready(self, status):
        index = len(self.wait_statuses) + len(self.home_wait_statuses)
        self.wait_statuses.append(status)
        self._wait(status, index)

    def wait_for_home_approval(self, status):
        index = len(self.wait_statuses) + len(self.home_wait_statuses)
        self.home_wait_statuses.append(status)
        self._wait(status, index)


class _ImmediateOperatorSession:
    def __init__(self):
        self.statuses = []
        self.wait_statuses = []
        self.home_wait_statuses = []

    def publish(self, status):
        self.statuses.append(status)

    def wait_for_scene_ready(self, status):
        self.wait_statuses.append(status)
        self.statuses.append(status)

    def wait_for_home_approval(self, status):
        self.home_wait_statuses.append(status)
        self.statuses.append(status)


class _ManualSuccessOperator(_ImmediateOperatorSession):
    auto_success = False

    def __init__(self):
        super().__init__()
        self.consume_calls = []

    def consume_operator_success(self, run_id, episode_id):
        self.consume_calls.append((run_id, episode_id))
        return len(self.consume_calls) == 1


class _SuccessAwareNetwork(_Network):
    def __init__(self):
        super().__init__([])

    def step(self, next_observation, **kwargs):
        self.step_calls.append((next_observation, dict(kwargs)))
        meta = kwargs["data"]["meta"]
        success = bool(meta["operator_success"])
        outcome = _outcome(
            meta["transition_id"], success=success, done=success
        )
        action = None if outcome.terminal else self._action(1)
        return SimpleNamespace(outcome=outcome, action=action)


def _config(max_steps):
    return SimpleNamespace(max_steps=max_steps, random_steps=0, buffer_period=0)


def _wait(event):
    assert event.wait(timeout=1.0), "actor did not enter expected wait"


def test_wait_has_no_step_or_begin_and_resume_uses_fresh_reset_observation():
    infos = [
        {"held": True, "intervened": 1},
        {"held": False, "intervened": 1},
        {"held": False, "intervened": 0},
    ]
    env = _Env(infos)
    network = _Network(
        [
            lambda tid: _outcome(
                tid,
                evaluated=True,
                probability=0.1,
                threshold=0.2,
            ),
            lambda tid: _outcome(
                tid,
                success=True,
                done=True,
                evaluated=True,
                probability=0.9,
                threshold=0.2,
            ),
            lambda tid: _outcome(tid),
        ]
    )
    operator = _BlockingOperatorSession(waits=3)
    result = {}

    def target():
        result["summary"] = run_remote_actor(
            network,
            env,
            config=_config(3),
            actor_id="actor",
            run_id="run",
            session_id_factory=iter(("s0", "s1")).__next__,
            operator_session=operator,
        )

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        _wait(operator.entered[0])
        assert env.reset_count == 1  # HOME happened
        assert env.step_count == 0
        assert len(network.begin_calls) == 0
        time.sleep(0.03)
        assert env.step_count == 0
        assert len(network.begin_calls) == 0

        operator.release[0].set()
        _wait(operator.entered[1])
        # Terminal pose is held: HOME cannot start before explicit approval.
        assert env.reset_count == 2
        assert env.step_count == 2
        assert len(network.begin_calls) == 1
        time.sleep(0.03)
        assert env.step_count == 2
        assert len(network.begin_calls) == 1

        operator.release[1].set()
        _wait(operator.entered[2])
        assert env.reset_count == 3  # approved HOME completed
        assert env.reset_options[2] == {"operator_approved_home": True}
        assert len(network.begin_calls) == 1

        operator.release[2].set()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    finally:
        for release in operator.release:
            release.set()

    assert result["summary"].episodes_started == 2
    assert env.reset_count == 5  # includes HOME on natural max_steps stop
    assert env.step_count == 3
    assert len(network.begin_calls) == 2
    # BeginEpisode sees reset 2 and reset 4, never either pre-wait HOME image.
    assert float(network.begin_calls[0][0]["state"][0, 0]) == 2.0
    assert float(network.begin_calls[1][0]["state"][0, 0]) == 4.0

    states = [status.state for status in operator.statuses]
    assert states[0:4] == [HOMING, WAIT_SCENE_READY, HOMING, POLICY_RUNNING]
    assert HOLD in states
    assert WAIT_HOME_APPROVAL in states
    assert HUMAN_INTERVENTION in states
    assert states[-1] == STOPPED

    hold_status = next(status for status in operator.statuses if status.state == HOLD)
    assert hold_status.control_owner == "HOLD"  # held wins over intervened
    human_status = next(
        status for status in operator.statuses
        if status.state == HUMAN_INTERVENTION
    )
    assert human_status.control_owner == "HUMAN"
    assert human_status.terminal_reason == "SUCCESS"

    final_policy = [
        status for status in operator.statuses
        if status.state == POLICY_RUNNING and status.env_step == 2
    ][-1]
    assert final_policy.classifier_evaluated is False
    assert final_policy.classifier_probability == pytest.approx(0.9)
    assert final_policy.classifier_threshold == pytest.approx(0.2)
    assert final_policy.classifier_env_step == 1


@pytest.mark.parametrize(
    "terminal_factory, expected_reason",
    [
        (lambda tid: _outcome(tid, truncated=True), "TRUNCATED"),
        (lambda tid: _outcome(tid, done=True), "EPISODE_LIMIT"),
    ],
)
def test_every_server_terminal_homes_and_waits(
    terminal_factory, expected_reason
):
    env = _Env()
    network = _Network([terminal_factory, lambda tid: _outcome(tid)])
    operator = _ImmediateOperatorSession()

    summary = run_remote_actor(
        network,
        env,
        config=_config(2),
        actor_id="actor",
        run_id="run",
        session_id_factory=iter(("s0", "s1")).__next__,
        operator_session=operator,
    )

    assert summary.episodes_started == 2
    assert env.reset_count == 5  # includes HOME on natural max_steps stop
    assert len(operator.wait_statuses) == 2  # startup plus terminal boundary
    assert len(operator.home_wait_statuses) == 1
    assert operator.home_wait_statuses[0].terminal_reason == expected_reason
    # ...but it is cleared once HOME completes, so it never decorates the NEXT
    # episode's WAIT_SCENE_READY.  A sticky reason there let the GUI burn its
    # one-shot abort confirmation on an episode that had not started yet, and
    # it is also what operator_session keys its "too late to abort" rejection
    # on -- that rejection must not stay armed into a healthy episode.
    assert operator.wait_statuses[1].terminal_reason == ""


def test_final_terminal_homes_then_stops_without_another_wait():
    env = _Env()
    network = _Network([lambda tid: _outcome(tid, done=True)])
    operator = _ImmediateOperatorSession()

    run_remote_actor(
        network,
        env,
        config=_config(1),
        actor_id="actor",
        run_id="run",
        session_id_factory=lambda: "s0",
        operator_session=operator,
    )

    assert env.reset_count == 3  # initial HOME, fresh reset, final HOME
    assert len(operator.wait_statuses) == 1
    assert len(operator.home_wait_statuses) == 1
    assert len(network.begin_calls) == 1
    assert operator.statuses[-2].state == HOMING
    assert operator.statuses[-1].state == STOPPED
    assert operator.statuses[-1].terminal_reason == "EPISODE_LIMIT"


def test_actor_exception_publishes_fault():
    env = _Env(failure=RuntimeError("robot step failed"))
    network = _Network([])
    operator = _ImmediateOperatorSession()

    with pytest.raises(RuntimeError, match="robot step failed"):
        run_remote_actor(
            network,
            env,
            config=_config(1),
            actor_id="actor",
            run_id="run",
            session_id_factory=lambda: "s0",
            operator_session=operator,
        )

    assert operator.statuses[-1].state == FAULT
    assert "robot step failed" in operator.statuses[-1].message
    assert operator.statuses[-1].success is False
    assert operator.statuses[-1].terminal_reason == "FAULT"


def test_success_home_failure_reports_fault_not_stale_success():
    env = _Env(reset_failure_at=3)
    network = _Network(
        [lambda tid: _outcome(tid, success=True, done=True)]
    )
    operator = _ImmediateOperatorSession()

    with pytest.raises(RuntimeError, match="robot HOME failed"):
        run_remote_actor(
            network,
            env,
            config=_config(1),
            actor_id="actor",
            run_id="run",
            session_id_factory=lambda: "s0",
            operator_session=operator,
        )

    assert env.reset_count == 3
    assert operator.statuses[-2].state == HOMING
    fault = operator.statuses[-1]
    assert fault.state == FAULT
    assert fault.success is False
    assert fault.terminal_reason == "FAULT"


def test_terminal_home_precedes_optional_pickle_failure(monkeypatch, tmp_path):
    env = _Env()
    network = _Network(
        [lambda tid: _outcome(tid, success=True, done=True)]
    )
    operator = _ImmediateOperatorSession()

    def fail_dump(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(remote_actor_module, "_dump_data", fail_dump)
    config = SimpleNamespace(max_steps=1, random_steps=0, buffer_period=1)
    with pytest.raises(OSError, match="disk full"):
        run_remote_actor(
            network,
            env,
            config=config,
            checkpoint_path=str(tmp_path),
            actor_id="actor",
            run_id="run",
            session_id_factory=lambda: "s0",
            operator_session=operator,
        )

    assert env.reset_count == 3  # startup HOME, fresh O0, terminal HOME
    assert operator.statuses[-2].state == HOMING
    assert operator.statuses[-1].state == FAULT


def test_natural_max_steps_exhaustion_homes_before_stopped():
    env = _Env()
    network = _Network([lambda tid: _outcome(tid)])
    operator = _ImmediateOperatorSession()

    run_remote_actor(
        network,
        env,
        config=_config(1),
        actor_id="actor",
        run_id="run",
        session_id_factory=lambda: "s0",
        operator_session=operator,
    )

    assert env.reset_count == 3
    assert operator.statuses[-2].state == HOMING
    assert operator.statuses[-2].terminal_reason == "MAX_STEPS"
    assert operator.statuses[-1].state == STOPPED


def test_manual_success_token_is_stamped_on_exactly_one_transition():
    env = _Env()
    network = _SuccessAwareNetwork()
    operator = _ManualSuccessOperator()

    run_remote_actor(
        network,
        env,
        config=_config(1),
        actor_id="actor",
        run_id="run",
        session_id_factory=lambda: "s0",
        operator_session=operator,
    )

    assert operator.consume_calls == [("run", 0)]
    meta = network.step_calls[0][1]["data"]["meta"]
    assert meta["auto_success"] is False
    assert meta["operator_success"] is True
    assert operator.home_wait_statuses[0].terminal_reason == "SUCCESS"
