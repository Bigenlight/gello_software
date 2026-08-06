"""The server-side opt-in that lets locally-issued policy meta reach replay.

WHY THIS FILE EXISTS
--------------------
``ActorSessionService`` enforces CHAIN OF CUSTODY on every transition: the
``meta.policy_action``/``meta.policy_version`` an actor reports must equal the
action and version THAT service issued for the previous observation.  Correct
when the actor is talking to the server that answered it; fatal when it is not.

In local-inference mode (``ur_env/local_policy/proxy.py``) the proxy runs a real
``ActorSessionService`` of its own, answers the actor from laptop-side
inference, and forwards the raw request bytes to the learner.  The learner then
runs its own stochastic inference over the same observation, cannot reproduce
the proxy's action, and answers ``INVALID_ARGUMENT`` -- so with a strict server
NO online transition ever reaches replay.  That is pinned, deliberately, by
``tests/test_local_policy_proxy.py::
test_forwarded_step_is_rejected_when_the_remote_policy_disagrees``; the first
test here re-pins it so the default can never drift.

``accept_external_policy_meta=True`` closes it.  It does not weaken the
contract into "trust the actor": the equality checks are REPLACED by intrinsic
ones (finite, right shape, within ``[-1, 1]``, executed action still equal to
the claimed policy action on non-intervention steps, version non-decreasing
within the session).  Everything else -- dedup, exactly-once, request_id and
step_id ordering, finalization, replay routing -- is untouched, and the tests
below assert that rather than assume it.

The proxy-side rig is imported from ``test_local_policy_proxy`` rather than
rebuilt: the whole point is that the SAME episode which is rejected today is
accepted with the flag on, and two subtly different harnesses could not show
that.

Run (actor venv)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_external_policy_ingest.py
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Optional

import numpy as np
import pytest

_HERE = Path(os.path.abspath(__file__)).parent
_INFRA_ROOT = _HERE.parent
_REPO_ROOT = _INFRA_ROOT.parent
sys.path.insert(0, str(_INFRA_ROOT))
sys.path.insert(0, str(_HERE))

from ur_env.actor_network import (  # noqa: E402
    EXTERNAL_POLICY_INGEST_ENV_VAR,
    ActorSessionService,
    ActorTransportError,
    is_external_policy_ingest_enabled,
)
from ur_env.grpc_actor_transport import GrpcActorServicer  # noqa: E402
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)

from test_local_policy_proxy import (  # noqa: E402
    LOCAL_ACTION,
    REMOTE_MODEL_ID,
    REWARD_MODEL_ID,
    RUN_ID,
    _ActorDriver,
    _CapturingRemote,
    _ClassifyingFinalizer,
    _FakePolicy,
    _client,
    _drain,
    _proxy,
    _uploader,
)
from ur_env.local_policy.uploader import (  # noqa: E402
    KIND_BEGIN_EPISODE,
    KIND_STEP,
)


#: A policy that disagrees with the proxy's, which is what a real learner is.
DISAGREEING_ACTION = np.zeros(7, dtype=np.float32)


# =========================================================================== #
# Rigs                                                                        #
# =========================================================================== #


class _IngestingRemote(_CapturingRemote):
    """``_CapturingRemote`` with the server-side opt-in switched ON.

    Rebuilds the service rather than mutating a private attribute, because the
    thing under test IS the constructor flag: a service that could be flipped
    after construction would be a different (and worse) contract.
    """

    def __init__(
        self,
        *,
        policy: Optional[_FakePolicy] = None,
        finalizer: Any = None,
        delay_s: float = 0.0,
        reward_model_id: str = REWARD_MODEL_ID,
        observation_schema_hash: str = CANONICAL_OBSERVATION_SCHEMA_HASH,
        accept_external_policy_meta: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            finalizer=finalizer,
            delay_s=delay_s,
            reward_model_id=reward_model_id,
            observation_schema_hash=observation_schema_hash,
        )
        self.service = ActorSessionService(
            sample_action=self.policy,
            model_id=REMOTE_MODEL_ID,
            reward_authority="server_classifier",
            reward_model_id=reward_model_id,
            observation_schema_hash=observation_schema_hash,
            finalize_transition=finalizer,
            accept_external_policy_meta=accept_external_policy_meta,
        )
        self.servicer = GrpcActorServicer(self.service)


class _TamperedNetwork:
    """Pass-through client that edits the outgoing ``data`` mapping.

    The transport does not range-check the transition it is handed (see
    ``grpc_actor_transport.data_to_proto``), so this is how a test states
    "the actor sent something the server must refuse" without hand-rolling a
    second copy of ``_ActorDriver``.
    """

    def __init__(self, network: Any, mutate: Callable[[dict], None]) -> None:
        self._network = network
        self._mutate = mutate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._network, name)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self._mutate(kwargs["data"])
        return self._network.step(*args, **kwargs)


def _episode_through_the_proxy(
    remote: _CapturingRemote, *, after_drain: Optional[Callable[[Any], None]] = None
):
    """Drive begin + 2 steps + a MANUAL success through a proxy at ``remote``.

    Returns ``(uploader, proxy_policy, results)``.  The proxy's policy answers
    ``LOCAL_ACTION``; the remote's answers ``DISAGREEING_ACTION`` -- the shape a
    real learner has, and the shape the strict rule refuses.

    ``after_drain`` runs while the uploader thread is still alive, which is the
    only window in which a test can enqueue anything of its own.
    """

    uploader = _uploader(remote.target)
    policy = _FakePolicy(action=LOCAL_ACTION)
    proxy = _proxy(remote, policy=policy, uploader=uploader)
    proxy.start()
    network = _client(proxy.port)
    results = []
    try:
        driver = _ActorDriver(network)
        results.append(driver.begin())
        results.append(driver.step())
        results.append(driver.step())
        results.append(driver.step(operator_success=True))
        _drain(uploader)
        if after_drain is not None:
            after_drain(uploader)
            _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
    return uploader, policy, results


# =========================================================================== #
# 1. Default off: today's behaviour, pinned                                   #
# =========================================================================== #


def test_the_flag_is_off_by_default_and_only_accepts_a_bool():
    """An env var two processes away must not be able to relax this by "0"."""

    service = ActorSessionService(sample_action=_FakePolicy())
    assert service.accept_external_policy_meta is False

    for value in (1, 0, "1", "true", "", None):
        with pytest.raises(ValueError, match="must be a bool"):
            ActorSessionService(
                sample_action=_FakePolicy(), accept_external_policy_meta=value
            )

    assert (
        ActorSessionService(
            sample_action=_FakePolicy(), accept_external_policy_meta=True
        ).accept_external_policy_meta
        is True
    )


def test_default_off_still_rejects_a_forwarded_step_from_a_disagreeing_policy(
    caplog,
):
    """The gap, re-pinned here so the default cannot drift silently.

    Same scenario as ``test_local_policy_proxy.py``'s pinned gap test, extended
    over a whole episode; owned twice on purpose, because THIS file is what a
    maintainer reads when deciding whether the flag may become the default.  It
    may not.

    Note the cascade in the log: only the FIRST step is refused for
    ``policy_action``.  Once it is gone the server's ``request_id`` sequence has
    a hole, so every later step is refused for a reason that says nothing about
    the real cause -- which is why the flag-on tests below assert
    ``rejected_count == 0`` rather than "the first one got in".
    """

    remote = _CapturingRemote(policy=_FakePolicy(action=DISAGREEING_ACTION))
    remote.start()
    try:
        with caplog.at_level("WARNING"):
            uploader, _policy, results = _episode_through_the_proxy(remote)
    finally:
        remote.stop()

    # The actor's own episode was answered locally and is unaffected.
    np.testing.assert_array_equal(results[0].action, LOCAL_ACTION)
    # Only BeginEpisode (which carries no policy meta) survived the trip.
    assert uploader.uploaded_count == 1
    assert uploader.rejected_count == 3
    assert "meta.policy_action does not match issued action" in caplog.text
    assert "TRANSITION NOT INGESTED" in caplog.text
    assert remote.service.replay_items == []


# =========================================================================== #
# 2. Flag on, end to end: the whole episode lands                             #
# =========================================================================== #


def test_flag_on_ingests_every_forwarded_transition_in_order():
    """The headline: same episode, same disagreeing server policy, accepted."""

    remote = _IngestingRemote(policy=_FakePolicy(action=DISAGREEING_ACTION))
    remote.start()
    try:
        uploader, policy, results = _episode_through_the_proxy(remote)
    finally:
        remote.stop()

    # The actor still saw the PROXY's action, never the learner's.
    np.testing.assert_array_equal(results[0].action, LOCAL_ACTION)
    for result in results[1:3]:
        np.testing.assert_array_equal(result.action.action, LOCAL_ACTION)
    assert policy.calls == 3  # begin + two non-terminal steps

    # Nothing was rejected, nothing diverged, nothing was retried away.
    assert uploader.rejected_count == 0
    assert uploader.divergence_count == 0
    assert uploader.uploaded_count == 4  # one BeginEpisode + three Steps
    assert uploader.last_error == ""

    # And the server really ingested them, in order, carrying the PROXY's
    # action and version -- the exact bytes the strict rule refused.
    items = remote.service.replay_items
    assert len(items) == 3
    assert [item["meta"]["env_step"] for item in items] == [0, 1, 2]
    assert [item["transition"]["step_id"] for item in items] == [0, 1, 2]
    for item in items:
        np.testing.assert_array_equal(
            np.asarray(item["meta"]["policy_action"], dtype=np.float32),
            LOCAL_ACTION,
        )
    status = remote.service.get_buffer_status()
    assert status.replay_insert_count == 3
    assert status.last_env_step == 2

    # The bytes on the wire were the proxy's captures, unchanged.
    assert uploader.captured_payloads(KIND_BEGIN_EPISODE) == remote.payloads(
        KIND_BEGIN_EPISODE
    )
    assert uploader.captured_payloads(KIND_STEP) == remote.payloads(KIND_STEP)


def test_flag_on_manual_success_finalizes_server_side_identically():
    """MANUAL is still the server's call, and it reaches the same verdict.

    The proxy's local finalizer and the server's independent one both see the
    same ``meta.operator_success`` token, so the strict divergence summary
    (reward/mask/done/truncated/success) matches field for field -- which is
    what "the alarm stays quiet on a healthy session" means once transitions
    actually get in.
    """

    finalizer = _ClassifyingFinalizer()
    remote = _IngestingRemote(
        policy=_FakePolicy(action=DISAGREEING_ACTION), finalizer=finalizer
    )
    remote.start()
    try:
        uploader, _policy, results = _episode_through_the_proxy(remote)
    finally:
        remote.stop()

    final = results[-1]
    assert final.action is None
    assert final.outcome.terminal
    assert final.outcome.success is True
    assert final.outcome.reward == 1.0
    assert final.outcome.mask == 0.0

    stored = remote.service.replay_items[-1]["transition"]
    assert float(stored["rewards"]) == 1.0
    assert float(stored["masks"]) == 0.0
    assert bool(stored["dones"]) is True
    assert bool(stored["truncated"]) is False
    assert int(stored["success"]) == 1
    assert uploader.divergence_count == 0
    assert uploader.rejected_count == 0


def test_flag_on_leaves_dedup_and_exactly_once_alone():
    """A replayed identical request is still answered from the cache."""

    remote = _IngestingRemote(policy=_FakePolicy(action=DISAGREEING_ACTION))
    remote.start()

    def _replay_the_first_step(uploader) -> None:
        uploader.enqueue_step(
            uploader.captured_payloads(KIND_STEP)[0],
            local_outcome_summary={
                "transition_id": f"{RUN_ID}:0",
                "done": False,
                "truncated": False,
                "success": False,
                "reward": 0.0,
                "mask": 1.0,
            },
        )

    try:
        uploader, _policy, _results = _episode_through_the_proxy(
            remote, after_drain=_replay_the_first_step
        )
    finally:
        remote.stop()

    assert uploader.uploaded_count == 5
    assert uploader.rejected_count == 0
    assert uploader.divergence_count == 0
    # Deduplicated: the byte-identical retry inserted nothing new, and the
    # cached reply still matched the local outcome (so no false divergence).
    assert len(remote.service.replay_items) == 3


# =========================================================================== #
# 3. Flag on, intrinsic validation still bites                                #
# =========================================================================== #
#
# These drive the flag-on service DIRECTLY over real gRPC (no proxy in front),
# because what is being asserted is what the SERVER refuses, and a proxy would
# only ever produce well-formed meta.


def _direct(remote: _CapturingRemote, **kwargs: Any):
    remote.start()
    return _client(remote.port, **kwargs)


def test_flag_on_accepts_meta_the_server_did_not_issue():
    """The relaxation itself, stated as one fact.

    The server issued ``LOCAL_ACTION`` at version 0; the actor reports a
    different (valid) action at version 7.  Strict mode refuses both; external
    ingest accepts both, and the transition lands.
    """

    remote = _IngestingRemote()
    network = _direct(remote)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.last_action = np.array(
            [0.9, -0.9, 0.5, -0.5, 0.0, 0.25, -1.0], dtype=np.float32
        )
        driver.last_version = 7
        assert driver.step().ack.accepted
    finally:
        network.close()
        remote.stop()

    item = remote.service.replay_items[-1]
    assert int(item["meta"]["policy_version"]) == 7
    np.testing.assert_allclose(
        np.asarray(item["meta"]["policy_action"], dtype=np.float32),
        np.array([0.9, -0.9, 0.5, -0.5, 0.0, 0.25, -1.0], dtype=np.float32),
    )


def test_flag_on_rejects_an_out_of_range_policy_action():
    remote = _IngestingRemote()
    network = _direct(remote)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.last_action = np.full(7, 2.0, dtype=np.float32)
        with pytest.raises(ActorTransportError) as excinfo:
            driver.step()
    finally:
        network.close()
        remote.stop()

    assert "meta.policy_action must be within [-1, 1]" in str(excinfo.value)
    assert "INVALID_ARGUMENT" in str(excinfo.value)
    assert remote.service.replay_items == []


def test_flag_on_rejects_a_non_finite_policy_action():
    remote = _IngestingRemote()
    network = _direct(remote)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.last_action = np.array(
            [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        with pytest.raises(ActorTransportError) as excinfo:
            driver.step()
    finally:
        network.close()
        remote.stop()

    assert "meta.policy_action contains a non-finite value" in str(excinfo.value)
    assert remote.service.replay_items == []


def test_flag_on_rejects_a_policy_version_regression_within_the_session():
    """The watermark is the INCOMING stream's, not the server's own.

    A decrease means two issuers are answering one session (or an old
    episode's meta is being replayed), which would put actions from an unknown
    policy into replay.
    """

    remote = _IngestingRemote()
    network = _direct(remote)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.last_version = 7
        assert driver.step().ack.accepted
        driver.last_version = 7  # equal is fine: versions hold between publishes
        assert driver.step().ack.accepted
        driver.last_version = 6
        with pytest.raises(ActorTransportError) as excinfo:
            driver.step()
    finally:
        network.close()
        remote.stop()

    assert "meta.policy_version decreased within the session: 7 -> 6" in str(
        excinfo.value
    )
    assert len(remote.service.replay_items) == 2


def test_flag_on_rejects_a_negative_policy_version():
    """``validate_counter`` is unchanged; the encoder refuses it even earlier."""

    from ur_env.actor_network import ActorProtocolError

    remote = _IngestingRemote()
    network = _direct(remote)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.last_version = -1
        with pytest.raises((ActorProtocolError, ActorTransportError, ValueError)):
            driver.step()
    finally:
        network.close()
        remote.stop()

    assert remote.service.replay_items == []


def test_flag_on_still_ties_the_executed_action_to_the_claimed_policy_action():
    """The one identity check that survives, and why it matters.

    Without it, external ingest would accept a transition whose stored action
    has no relationship to the policy that supposedly produced it.
    """

    remote = _IngestingRemote()
    network = _direct(remote)
    tampered = _TamperedNetwork(
        network,
        lambda data: data["transition"].__setitem__(
            "actions", np.full(7, 0.5, dtype=np.float32)
        ),
    )
    try:
        driver = _ActorDriver(tampered)
        driver.begin()
        with pytest.raises(ActorTransportError) as excinfo:
            driver.step()
    finally:
        network.close()
        remote.stop()

    assert "non-intervention transition.actions must equal policy_action" in str(
        excinfo.value
    )
    assert remote.service.replay_items == []


def test_flag_on_leaves_request_id_ordering_alone():
    """Nothing about the relaxation touches session sequencing."""

    from ur_env.actor_network import FailedPreconditionError  # noqa: F401

    remote = _IngestingRemote()
    network = _direct(remote)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        assert driver.step().ack.accepted
        # Rewind the step_id the driver will claim: the server must refuse it
        # even though the policy meta is now unconstrained.
        driver._step_id = 0
        with pytest.raises(ActorTransportError) as excinfo:
            driver.step()
    finally:
        network.close()
        remote.stop()

    assert "step_id must be 1" in str(excinfo.value)
    assert len(remote.service.replay_items) == 1


# =========================================================================== #
# 4. The env gate                                                             #
# =========================================================================== #


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_env_gate_accepts_the_same_spellings_as_the_other_server_opt_ins(value):
    assert is_external_policy_ingest_enabled(
        {EXTERNAL_POLICY_INGEST_ENV_VAR: value}
    )


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "onn", "2"])
def test_env_gate_rejects_everything_else(value):
    assert not is_external_policy_ingest_enabled(
        {EXTERNAL_POLICY_INGEST_ENV_VAR: value}
    )


def test_env_gate_is_off_when_the_variable_is_absent():
    assert not is_external_policy_ingest_enabled({})


def test_env_gate_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.delenv(EXTERNAL_POLICY_INGEST_ENV_VAR, raising=False)
    assert not is_external_policy_ingest_enabled()
    monkeypatch.setenv(EXTERNAL_POLICY_INGEST_ENV_VAR, "1")
    assert is_external_policy_ingest_enabled()


def test_the_learner_entrypoint_reads_the_gate_and_passes_it_down():
    """No CLI flag: validate_process_contract compares argv token by token."""

    source = (
        _INFRA_ROOT / "scripts" / "run_rlpd_learner_server.py"
    ).read_text(encoding="utf-8")
    assert "is_external_policy_ingest_enabled()" in source
    assert "accept_external_policy_meta=accept_external_policy_meta" in source
    assert "rlpd_learner_external_policy_ingest" in source
    assert "--external-policy-ingest" not in source
    assert "--accept-external-policy-meta" not in source
    # Also in the run's permanent JSONL, because a REUSED learner's launching
    # environment is long gone and the launcher parses that file.
    assert (
        "accept_external_policy_meta=accept_external_policy_meta," in source
    )


def test_build_actor_service_defaults_to_strict():
    """The learner composition must not become permissive by omission."""

    import inspect

    from ur_env.learner.composition import build_actor_service

    parameter = inspect.signature(build_actor_service).parameters[
        "accept_external_policy_meta"
    ]
    assert parameter.default is False
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


# =========================================================================== #
# 5. The launcher: a 17th optional positional, and nothing else moves         #
# =========================================================================== #


_SERVER_SH = _REPO_ROOT / "ros2_ur_ws" / "run_hil_server.sh"
_SENTINEL = "__unset__"
_BASELINE_POSITIONALS = 14


def _positionals(result) -> list[str]:
    tokens = [
        token.decode() for token in result.ssh_log.read_bytes().split(b"\0") if token
    ]
    return tokens[tokens.index("--") + 1 :]


def test_launcher_forwards_the_env_var_and_leaves_argv_byte_identical(tmp_path):
    from test_hil_server_wandb_mode import (
        _expand_launch_command,
        _rig,
        _split_env_prefix,
    )

    rig = _rig(tmp_path, wandb_mode="offline")
    assert rig["EXTERNAL_INGEST"] == ""

    off_env, off_argv = _split_env_prefix(_expand_launch_command(tmp_path, rig))
    on_env, on_argv = _split_env_prefix(
        _expand_launch_command(tmp_path, dict(rig, EXTERNAL_INGEST="1"))
    )

    # THE constraint: not one argv token added, removed, reordered or rewritten,
    # because run_hil_server.sh re-validates a RUNNING learner against the exact
    # command it would have launched.
    assert on_argv == off_argv
    assert EXTERNAL_POLICY_INGEST_ENV_VAR not in off_env
    assert on_env == {**off_env, EXTERNAL_POLICY_INGEST_ENV_VAR: "1"}


@pytest.mark.parametrize(
    "environment,expected_tail",
    [
        ({}, []),
        ({"HIL_LATENCY_PROFILE": "1"}, ["1"]),
        ({"HIL_PARAMS_EXPORT": "1"}, [_SENTINEL, "1"]),
        ({"HIL_EXTERNAL_POLICY_INGEST": "1"}, [_SENTINEL, _SENTINEL, "1"]),
        ({"HIL_LATENCY_PROFILE": "1", "HIL_PARAMS_EXPORT": "1"}, ["1", "1"]),
        (
            {"HIL_LATENCY_PROFILE": "1", "HIL_EXTERNAL_POLICY_INGEST": "1"},
            ["1", _SENTINEL, "1"],
        ),
        (
            {"HIL_PARAMS_EXPORT": "1", "HIL_EXTERNAL_POLICY_INGEST": "1"},
            [_SENTINEL, "1", "1"],
        ),
        (
            {
                "HIL_LATENCY_PROFILE": "1",
                "HIL_PARAMS_EXPORT": "1",
                "HIL_EXTERNAL_POLICY_INGEST": "1",
            },
            ["1", "1", "1"],
        ),
    ],
)
def test_every_optional_combination_is_marshalled_unambiguously(
    tmp_path, environment, expected_tail
):
    """ssh joins its arguments into ONE remote command line.

    A trailing empty simply disappears (which is why the production default is
    still exactly 14 positionals) but an INTERIOR empty shifts every later
    value one slot left -- so ``HIL_EXTERNAL_POLICY_INGEST=1`` alone would
    arrive as ``HIL_LATENCY_PROFILE=1`` and switch on the wrong feature with no
    error anywhere.  The sentinel is what makes that impossible; this
    parametrisation is the whole truth table.
    """

    from test_hil_server_wandb_mode import _run_launcher

    result = _run_launcher(tmp_path, "--check", **environment)
    # The stub ssh reports "no learner", which --check maps to exit 3.
    assert result.returncode == 3, result.stderr
    positionals = _positionals(result)
    assert len(positionals) == _BASELINE_POSITIONALS + len(expected_tail)
    assert positionals[_BASELINE_POSITIONALS:] == expected_tail
    assert positionals[_BASELINE_POSITIONALS - 1] == "offline"


@pytest.mark.parametrize(
    "extra,expected",
    [
        ([], ("", "", "")),
        (["1"], ("1", "", "")),
        ([_SENTINEL, "1"], ("", "1", "")),
        ([_SENTINEL, _SENTINEL, "1"], ("", "", "1")),
        (["1", "1"], ("1", "1", "")),
        (["1", _SENTINEL, "1"], ("1", "", "1")),
        ([_SENTINEL, "1", "1"], ("", "1", "1")),
        (["1", "1", "1"], ("1", "1", "1")),
    ],
)
def test_the_remote_side_decodes_every_optional_combination(
    tmp_path, extra, expected
):
    """The other half of the sentinel contract, run as the shell runs it."""

    from test_params_export import _remote_positional_prologue

    script = (
        "set -euo pipefail\n"
        + _remote_positional_prologue()
        + '\nprintf "%s\\n%s\\n%s\\n" "$LATENCY_PROFILE" "$PARAMS_EXPORT"'
        ' "$EXTERNAL_INGEST"\n'
    )
    result = subprocess.run(
        [
            "bash",
            "-s",
            "--",
            "check",
            "0",
            "0",
            "run-id",
            "300",
            str(tmp_path),
            sys.executable,
            "50053",
            f"{tmp_path}/runs",
            "abc123",
            "branch",
            "0",
            str(tmp_path),
            "offline",
            *extra,
        ],
        input=script,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert tuple(result.stdout.split("\n")[:3]) == expected


def test_launcher_refuses_a_value_that_would_resplit_on_the_remote_shell(tmp_path):
    from test_hil_server_wandb_mode import _run_launcher

    result = _run_launcher(
        tmp_path,
        "--check",
        HIL_EXTERNAL_POLICY_INGEST="1 ; touch /tmp/pwned-external-ingest",
    )

    assert result.returncode == 3, result.stderr
    assert "ignoring HIL_EXTERNAL_POLICY_INGEST" in result.stderr
    assert len(_positionals(result)) == _BASELINE_POSITIONALS
    assert not os.path.exists("/tmp/pwned-external-ingest")


def test_the_launch_block_forwards_the_env_var_by_name():
    """One expansion, spelled the way ``ur_env/actor_network.py`` reads it."""

    text = _SERVER_SH.read_text(encoding="utf-8")
    assert (
        '${EXTERNAL_INGEST:+HIL_EXTERNAL_POLICY_INGEST="$EXTERNAL_INGEST"} \\'
        in text
    )
    # Never an argv token: validate_process_contract compares argv exactly.
    assert "--external-policy-ingest" not in text
