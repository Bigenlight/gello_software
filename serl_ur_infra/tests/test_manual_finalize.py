"""THE parity test: the proxy's MANUAL finalizer against the server's own.

The local policy proxy answers the actor's Step RPC itself, so it has to put a
``TransitionOutcome`` in that reply while the reward model stays on the server.
Everything else in the local-inference phase is plumbing that fails loudly;
this is the one place where a silent, plausible-looking wrong answer would be
trained on.  So the check is not "does the local finalizer look right" but
"does it produce, field for field, what
``ur_env.rlpd_receive_server.RewardTransitionFinalizer`` produces in MANUAL
mode" -- fed from ``ur_env.remote_actor.build_data``, the real producer of the
transitions in question.

No jax anywhere here: it runs in the actor venv, which is where the canonical
suite runs.

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_manual_finalize.py
"""

from __future__ import annotations

import copy
import dataclasses
import os
import sys
from typing import Any, Mapping

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _INFRA_ROOT)

from ur_env.actor_network import (  # noqa: E402
    ActorNetworkError,
    ActorProtocolError,
    ActorSessionService,
    BeginEpisodeCommand,
    ObservationPacket,
    PROTOCOL_VERSION,
    StepCommand,
    TransitionOutcome,
)
from ur_env.local_policy.manual_finalize import (  # noqa: E402
    ManualFinalizerError,
    ManualTransitionFinalizer,
)
from ur_env.remote_actor import build_data  # noqa: E402
from ur_env.rlpd_receive_server import RewardTransitionFinalizer  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


class _NeverClassifier:
    """A classifier that fails the test if the MANUAL path ever calls it.

    This is what "WITHOUT a classifier" means concretely: the server finalizer
    is constructed for real, and the parity claim only holds if it never
    reaches for a model -- which it cannot, because every transition below
    arrives with ``classifier_sidecar=None``.
    """

    ready = True
    threshold = 0.5
    reward_model_id = "never-used"

    def classify(self, frames: Any) -> Any:  # pragma: no cover - must not run
        raise AssertionError(
            "the MANUAL path classified a frame; parity is meaningless"
        )


class _WarningSink:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)


def _observation(value: int) -> dict[str, np.ndarray]:
    return {
        "state": np.full((1, 19), value / 100.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _transition(
    *,
    step: int = 0,
    session_id: str = "session-0",
    reward: float = 0.0,
    done: bool = False,
    truncated: bool = False,
    operator_success: bool = False,
    auto_success: bool = False,
    intervened: bool = False,
    grasp_penalty: float | None = None,
) -> dict[str, Any]:
    """One actor-shaped transition, then completed the way the service does.

    ``build_data`` is the actual producer on the robot, so the battery cannot
    drift into shapes the finalizers never see.  ``ActorSessionService.step``
    attaches ``observations``/``next_observations`` before calling a finalizer;
    that is done here for the same reason.
    """

    info: dict[str, Any] = {}
    if intervened:
        info["intervened"] = 1
        info["intervene_action"] = np.full(7, -0.25, dtype=np.float32)
    if grasp_penalty is not None:
        info["grasp_penalty"] = float(grasp_penalty)
    data = build_data(
        actor_id="actor-0",
        run_id="run-0",
        session_id=session_id,
        transition_id=f"run-0:{session_id}:{step}",
        env_step=step,
        timestamp_ns=1_700_000_000_000_000_000 + step,
        policy_version=7,
        policy_action=np.full(7, step / 100.0, dtype=np.float32),
        auto_success=auto_success,
        operator_success=operator_success,
        episode_id=0,
        step_id=step,
        observation_id=f"observation-{step}",
        next_observation_id=f"observation-{step + 1}",
        reward=reward,
        done=done,
        truncated=truncated,
        info=info,
    )
    data["transition"]["observations"] = _observation(step)
    data["transition"]["next_observations"] = _observation(step + 1)
    return data


#: The battery.  Every case is MANUAL (``auto_success=False``); AUTO has its own
#: test because the two finalizers are SUPPOSED to disagree about it.
PARITY_CASES: dict[str, dict[str, Any]] = {
    # Ordinary mid-episode step: nothing is asserted, reward is already 0.
    "plain": {},
    # A locally proposed POSITIVE reward.  The server discards it; if the local
    # one passed it through, this case is the only thing that would notice.
    "local_reward_discarded": {"reward": 0.73},
    # Gripper penalty: the actor folds a negative number into the reward and
    # also ships it as transition.grasp_penalty (a protected field neither
    # finalizer may touch).
    "gripper_penalty": {"reward": -0.1, "grasp_penalty": -0.1},
    "gripper_penalty_on_intervention": {
        "reward": -0.1,
        "grasp_penalty": -0.1,
        "intervened": True,
    },
    # MANUAL MARK SUCCESS: reward 1, mask 0, done, not truncated.
    "operator_success": {"operator_success": True},
    # The click that lands on the same step as the episode limit or an
    # END EPISODE truncation.  Success must win and clear the truncation.
    "operator_success_over_truncation": {
        "operator_success": True,
        "truncated": True,
    },
    "operator_success_over_done": {"operator_success": True, "done": True},
    "operator_success_while_intervening": {
        "operator_success": True,
        "intervened": True,
    },
    # Terminal without success (MAX_EPISODE_LENGTH lands in dones today, G35).
    "terminal_done": {"done": True},
    # END EPISODE / operator abort: truncated, mask stays 1.
    "truncated_abort": {"truncated": True},
    "truncated_abort_while_intervening": {"truncated": True, "intervened": True},
    # Intervened steps are routed to both buffers; nothing about the reward
    # rules changes, and that has to stay true.
    "intervened": {"intervened": True},
    # A later step of a LATER session: proves neither finalizer's per-session
    # bookkeeping leaks into the values it reports.
    "later_session": {"step": 42, "session_id": "session-9"},
    "later_session_success": {
        "step": 43,
        "session_id": "session-9",
        "operator_success": True,
    },
}


# --------------------------------------------------------------------------- #
# Deep comparison                                                              #
# --------------------------------------------------------------------------- #


def _assert_same(actual: Any, expected: Any, path: str) -> None:
    """Compare value AND concrete type, recursively.

    Type equality is load bearing: ``transition["success"]`` is an
    ``np.uint8`` on the wire and a plain ``0`` would compare equal while
    serializing differently downstream.
    """

    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{path}: not a mapping"
        assert set(actual) == set(expected), (
            f"{path}: key mismatch "
            f"missing={sorted(set(expected) - set(actual))} "
            f"extra={sorted(set(actual) - set(expected))}"
        )
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), f"{path}: not an ndarray"
        assert actual.dtype == expected.dtype, f"{path}: dtype"
        assert actual.shape == expected.shape, f"{path}: shape"
        np.testing.assert_array_equal(actual, expected, err_msg=path)
        return
    assert type(actual) is type(expected), (
        f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    )
    if isinstance(expected, np.generic):
        assert actual.dtype == expected.dtype, f"{path}: scalar dtype"
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _server_finalizer() -> RewardTransitionFinalizer:
    return RewardTransitionFinalizer(_NeverClassifier(), warn=_WarningSink())


# --------------------------------------------------------------------------- #
# Parity                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", sorted(PARITY_CASES))
def test_manual_finalizer_matches_the_server_field_for_field(case):
    kwargs = PARITY_CASES[case]
    server = _server_finalizer()
    local = ManualTransitionFinalizer()
    data = _transition(**kwargs)

    server_data, server_outcome = server(copy.deepcopy(data), None)
    local_data, local_outcome = local(copy.deepcopy(data), None)

    assert isinstance(local_outcome, TransitionOutcome)
    assert dataclasses.asdict(local_outcome) == dataclasses.asdict(server_outcome)
    _assert_same(local_data, server_data, "data")


def test_the_battery_actually_exercises_every_outcome_shape():
    """A parity suite that only ever saw one outcome would prove nothing."""

    local = ManualTransitionFinalizer()
    seen = set()
    for kwargs in PARITY_CASES.values():
        _data, outcome = local(_transition(**kwargs), None)
        seen.add((outcome.reward, outcome.mask, outcome.done, outcome.truncated))
    assert (1.0, 0.0, True, False) in seen, "no success case"
    assert (0.0, 1.0, False, False) in seen, "no ordinary step"
    assert (0.0, 0.0, True, False) in seen, "no unsuccessful terminal"
    assert (0.0, 1.0, False, True) in seen, "no truncation"


def test_parity_holds_when_one_finalizer_instance_sees_the_whole_sequence():
    """Run the battery through ONE pair of instances, in order.

    The per-case test builds a fresh pair each time, which cannot catch state
    that accumulates across transitions (confirmation windows, session
    counters).  This one keeps both instances alive for the whole run.
    """

    server = _server_finalizer()
    local = ManualTransitionFinalizer()
    for index, name in enumerate(sorted(PARITY_CASES)):
        data = _transition(**{"step": index, **PARITY_CASES[name]})
        server_data, server_outcome = server(copy.deepcopy(data), None)
        local_data, local_outcome = local(copy.deepcopy(data), None)
        assert dataclasses.asdict(local_outcome) == dataclasses.asdict(
            server_outcome
        ), name
        _assert_same(local_data, server_data, f"data[{name}]")
    assert local.transition_count == len(PARITY_CASES)
    assert local.classification_count == 0


def test_finalizer_does_not_mutate_the_data_it_was_given():
    local = ManualTransitionFinalizer()
    data = _transition(reward=0.73, operator_success=True)
    before = copy.deepcopy(data)

    finalized, _outcome = local(data, None)

    _assert_same(data, before, "input")
    assert finalized is not data
    assert finalized["transition"]["rewards"] == 1.0


# --------------------------------------------------------------------------- #
# The sidecar, and the divergence it creates on purpose                        #
# --------------------------------------------------------------------------- #


def _sidecar() -> dict[str, Any]:
    from ur_env.classifier_sidecar import build_sidecar

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, 1] = 200
    return build_sidecar({"cam1": frame, "cam2": frame})


def test_a_sidecar_is_accepted_counted_and_left_unevaluated():
    """The proxy cannot score frames, and must not fault on being handed them.

    The actor keeps attaching them because the forwarded raw request is what
    the real server classifies.
    """

    local = ManualTransitionFinalizer()
    data = _transition()

    with_sidecar, outcome = local(copy.deepcopy(data), _sidecar())
    without_sidecar, plain_outcome = local(copy.deepcopy(data), None)

    assert outcome.classifier_evaluated is False
    assert outcome.classifier_probability == 0.0
    assert outcome.classifier_threshold == 0.0
    assert outcome.reward_model_id == ""
    assert dataclasses.asdict(outcome) == dataclasses.asdict(plain_outcome)
    _assert_same(with_sidecar, without_sidecar, "data")
    assert local.sidecar_ignored_count == 1
    assert local.classification_count == 0
    assert local.classifier_degraded is False


def test_reward_fields_still_match_the_server_when_the_server_does_classify():
    """The documented divergence, pinned: telemetry differs, learning does not.

    A step that carried a sidecar is scored by the real server and not by the
    proxy, so ``classifier_*``/``reward_model_id`` diverge BY DESIGN.  The
    fields replay actually trains on must not.
    """

    from ur_env.rlpd_receive_server import ScriptedRewardClassifierRuntime

    classifying_server = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([0.93]), warn=_WarningSink()
    )
    local = ManualTransitionFinalizer()
    data = _transition(operator_success=True)

    server_data, server_outcome = classifying_server(
        copy.deepcopy(data), _sidecar()
    )
    local_data, local_outcome = local(copy.deepcopy(data), _sidecar())

    for field in ("reward", "mask", "done", "truncated", "success"):
        assert getattr(local_outcome, field) == getattr(server_outcome, field), field
    assert server_outcome.classifier_evaluated is True
    assert local_outcome.classifier_evaluated is False
    for key in ("rewards", "masks", "dones", "truncated", "success"):
        _assert_same(
            local_data["transition"][key],
            server_data["transition"][key],
            f"transition.{key}",
        )


# --------------------------------------------------------------------------- #
# AUTO is refused                                                              #
# --------------------------------------------------------------------------- #


def test_auto_success_is_refused_with_a_protocol_error():
    local = ManualTransitionFinalizer()
    data = _transition(auto_success=True)

    with pytest.raises(ManualFinalizerError) as excinfo:
        local(data, None)

    assert isinstance(excinfo.value, ActorProtocolError)
    message = str(excinfo.value)
    assert "MANUAL only" in message
    assert "HIL_POLICY_MODE=remote" in message
    assert local.transition_count == 0


def test_auto_success_is_refused_even_with_a_sidecar_and_a_terminal():
    """No shape of AUTO transition sneaks through: the flag alone decides."""

    local = ManualTransitionFinalizer()
    for kwargs in ({}, {"done": True}, {"truncated": True}, {"intervened": True}):
        with pytest.raises(ManualFinalizerError):
            local(_transition(auto_success=True, **kwargs), _sidecar())


def test_auto_success_refusal_faults_the_session_service():
    """End to end: an AUTO transition stops the actor rather than mis-scoring it."""

    service, begin_action = _service_with_local_finalizer()
    command = _step_command(
        service_action=begin_action, step=0, auto_success=True
    )

    with pytest.raises(ActorNetworkError):
        service.step(command)

    reachable, ready, detail = service.health()
    assert reachable and not ready
    assert "MANUAL only" in detail


def test_both_success_flags_together_are_the_malformed_error_not_the_auto_one():
    """``auto and operator`` is a broken transition, and says so specifically."""

    local = ManualTransitionFinalizer()
    data = _transition()
    data["meta"]["auto_success"] = True
    data["meta"]["operator_success"] = True

    with pytest.raises(ActorProtocolError) as excinfo:
        local(data, None)

    assert not isinstance(excinfo.value, ManualFinalizerError)
    assert "forbidden while auto_success" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Malformed input                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d.pop("meta"), "exactly meta and transition"),
        (lambda d: d.__setitem__("extra", {}), "exactly meta and transition"),
        (lambda d: d["meta"].pop("transition_id"), "meta.transition_id is required"),
        (
            lambda d: d["transition"].pop("next_observations"),
            "next_observations is required",
        ),
        (lambda d: d["meta"].__setitem__("operator_success", 2), "must be 0 or 1"),
    ],
)
def test_malformed_transitions_raise_the_same_way_the_server_does(mutate, expected):
    server = _server_finalizer()
    local = ManualTransitionFinalizer()
    data = _transition()
    mutate(data)

    with pytest.raises(ActorProtocolError) as local_error:
        local(copy.deepcopy(data), None)
    with pytest.raises(ActorProtocolError) as server_error:
        server(copy.deepcopy(data), None)

    assert expected in str(local_error.value)
    assert str(local_error.value) == str(server_error.value)


# --------------------------------------------------------------------------- #
# ActorSessionService accepts what this finalizer produces                     #
# --------------------------------------------------------------------------- #


def _service_with_local_finalizer() -> tuple[ActorSessionService, np.ndarray]:
    """A service wired exactly as the proxy will wire it, one episode open."""

    def sample(observation, deterministic):
        del observation, deterministic
        return np.full(7, 0.25, dtype=np.float32), 3

    service = ActorSessionService(
        sample,
        model_id="hil-serl-local-policy-resnet10-manual-v1",
        reward_authority="server_classifier",
        finalize_transition=ManualTransitionFinalizer(),
    )
    action = service.begin_episode(
        BeginEpisodeCommand(
            PROTOCOL_VERSION,
            "actor-0",
            "run-0",
            "session-0",
            0,
            1,
            10_000,
            ObservationPacket("observation-0", 1_000, _observation(0)),
        )
    )
    return service, action.action


def _step_command(
    *,
    service_action: np.ndarray,
    step: int,
    request_id: int = 2,
    **kwargs: Any,
) -> StepCommand:
    """One Step the service's own ``_validate_data`` will accept.

    The identity fields have to agree with the session the service opened --
    it timestamps the transition with O(t)'s timestamp, pins the policy
    action/version to the one it issued, and requires ``request_action`` to be
    false exactly for LOCALLY terminal steps (a MANUAL success is not one of
    those: it becomes terminal only after this finalizer runs).
    """

    data = _transition(**{"step": step, **kwargs})
    transition = data["transition"]
    data["meta"]["timestamp_ns"] = 1_000
    data["meta"]["policy_action"] = np.asarray(service_action, dtype=np.float32)
    data["meta"]["policy_version"] = 3
    if not data["meta"]["intervened"]:
        transition["actions"] = np.asarray(service_action, dtype=np.float32)
    transition.pop("observations")
    transition.pop("next_observations")
    terminal = bool(transition["dones"]) or bool(transition["truncated"])
    return StepCommand(
        PROTOCOL_VERSION,
        "actor-0",
        "run-0",
        "session-0",
        request_id,
        20_000 + request_id,
        data,
        ObservationPacket(
            f"observation-{step + 1}", 2_000 + step, _observation(step + 1)
        ),
        not terminal,
    )


def test_the_service_binds_this_finalizer_as_a_sidecar_aware_one():
    """``_bind_finalizer`` inspects the signature; getting this wrong is fatal.

    A finalizer it reads as one-argument gets wrapped in a shim that RAISES the
    first time an actor attaches a sidecar -- which the production actor does
    every ~2 Hz, so the mistake would surface as a dead session seconds into
    the first episode rather than as anything resembling a signature bug.
    """

    finalizer = ManualTransitionFinalizer()
    bound = ActorSessionService._bind_finalizer(finalizer)

    assert bound is finalizer

    service, action = _service_with_local_finalizer()
    command = _step_command(service_action=action, step=0)
    packet = command.next_observation
    from ur_env.classifier_sidecar import CLASSIFIER_SIDECAR_KEY

    observation = dict(packet.observation)
    observation[CLASSIFIER_SIDECAR_KEY] = _sidecar()
    result = service.step(
        StepCommand(
            command.protocol_version,
            command.actor_id,
            command.run_id,
            command.session_id,
            command.request_id,
            command.created_monotonic_ns,
            command.data,
            ObservationPacket(
                packet.observation_id, packet.timestamp_ns, observation
            ),
            command.request_action,
        )
    )

    assert result.ack.accepted
    assert result.outcome.classifier_evaluated is False


def test_service_validator_accepts_an_ordinary_local_finalization():
    service, action = _service_with_local_finalizer()

    result = service.step(_step_command(service_action=action, step=0))

    assert result.ack.accepted
    assert result.outcome.reward == 0.0
    assert result.outcome.mask == 1.0
    assert result.outcome.classifier_evaluated is False
    assert result.action is not None
    assert result.action.policy_version == 3


def test_service_validator_accepts_a_manual_success_and_closes_the_episode():
    service, action = _service_with_local_finalizer()

    result = service.step(
        _step_command(service_action=action, step=0, operator_success=True)
    )

    assert result.ack.accepted
    assert result.outcome.success is True
    assert result.outcome.reward == 1.0
    assert result.outcome.mask == 0.0
    assert result.outcome.done is True
    assert result.outcome.truncated is False
    # Terminal: no action comes back and the session is closed.
    assert result.action is None
    reachable, ready, _detail = service.health()
    assert reachable and ready


def test_service_validator_accepts_an_operator_abort_truncation():
    service, action = _service_with_local_finalizer()

    result = service.step(
        _step_command(service_action=action, step=0, truncated=True)
    )

    assert result.ack.accepted
    assert result.outcome.truncated is True
    assert result.outcome.done is False
    assert result.outcome.mask == 1.0
    assert result.outcome.success is False
    assert result.action is None
