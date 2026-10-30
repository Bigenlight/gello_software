"""The laptop-side policy runtime: contract first, then real jax.

THREE TIERS, ON PURPOSE
-----------------------
1. Everything that can be wrong WITHOUT jax is pinned in the actor venv, which
   is where the canonical suite runs: the device-selection contract, the
   parameter-holder contract, and the version clamp that stops a misbehaving
   puller from ending an episode.  ``ur_env/local_policy/runtime.py`` imports
   jax only inside functions precisely so this tier exists;
   ``test_module_import_does_not_import_jax`` keeps it that way.
2. ``LocalPolicyRuntime`` itself, driven with a FAKE agent and an injected
   sampler.  Needs jax (PRNG keys, tree validation) but not the network, so it
   runs in seconds under ``/home/laptop3/venvs/hilserl/bin/python`` and skips
   cleanly where jax is absent.
3. The REAL production agent (``create_frozen_trunk_feature_agent``), opt-in
   behind ``RUN_HIL_SERL_ACTUAL_LOCAL_POLICY=1`` following this repo's
   convention for tests that build the network: it is minutes and a gigabyte,
   and it needs ``third_party/hil-serl/examples/experiments/resnet10_params``
   ``.pkl``.

Run tier 1 (canonical suite, no jax)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_local_policy_runtime.py

Run tiers 1+2 (CPU jax)::

    ... /home/laptop3/venvs/hilserl/bin/python -m pytest -q ...

Run tier 3 as well::

    RUN_HIL_SERL_ACTUAL_LOCAL_POLICY=1 ... /home/laptop3/venvs/hilserl/bin/python ...
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import numpy as np
import pytest


_HERE = Path(os.path.abspath(__file__)).parent
_INFRA_ROOT = _HERE.parent
_REPO_ROOT = _INFRA_ROOT.parent
sys.path.insert(0, str(_INFRA_ROOT))

from ur_env.actor_network import (  # noqa: E402
    ActorSessionService,
    BeginEpisodeCommand,
    ObservationPacket,
    PROTOCOL_VERSION,
    PolicyInferenceError,
)
from ur_env.learner.policy import PolicyValidationError  # noqa: E402
from ur_env.local_policy.runtime import (  # noqa: E402
    DEVICE_PLATFORMS,
    LOCAL_POLICY_DEVICE_ENV,
    LOCAL_POLICY_MODEL_ID,
    LocalPolicyError,
    LocalPolicyRuntime,
    MonotonicVersionStamp,
    StaticParamsHolder,
    configure_jax_platform,
    jax_platform_for_device,
    read_params_holder,
)


_HAS_JAX = importlib.util.find_spec("jax") is not None
_needs_jax = pytest.mark.skipif(
    not _HAS_JAX, reason="needs a jax-capable venv (hilserl / gello-local-policy)"
)

_REAL_AGENT_ENV = "RUN_HIL_SERL_ACTUAL_LOCAL_POLICY"
_needs_real_agent = pytest.mark.skipif(
    os.environ.get(_REAL_AGENT_ENV) != "1" or not _HAS_JAX,
    reason=f"set {_REAL_AGENT_ENV}=1 in a jax venv to build the real agent",
)


# =========================================================================== #
# Tier 1: no jax                                                              #
# =========================================================================== #


def test_module_import_does_not_import_jax():
    """The device prologue can only win the race if importing us does not.

    ``configure_jax_platform`` writes ``JAX_PLATFORMS``, which jax reads once,
    at ITS import.  An entrypoint that imports this module first and calls the
    prologue second would silently get the wrong backend, so importing this
    module must not drag jax in.  Checked in a FRESH interpreter: this one may
    already have jax loaded by a sibling test module, so an in-process check
    would be either vacuous or a false alarm depending on collection order.
    """

    probe = (
        "import sys;"
        "import ur_env.local_policy.runtime as m;"
        "import ur_env.local_policy.manual_finalize as f;"
        "print('jax' in sys.modules)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_INFRA_ROOT),
            str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher"),
        ]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", result.stdout


def test_model_id_is_the_pinned_proxy_identity():
    """The operator's proof of which inference path they got."""

    assert LOCAL_POLICY_MODEL_ID == "hil-serl-local-policy-resnet10-manual-v1"
    assert LocalPolicyRuntime.model_id == LOCAL_POLICY_MODEL_ID


# -- device selection ------------------------------------------------------- #


@pytest.mark.parametrize(
    "device, platform",
    [("cpu", "cpu"), ("gpu", "cuda"), ("auto", None), ("GPU", "cuda"), (" cpu ", "cpu")],
)
def test_device_request_maps_to_a_jax_platform(device, platform):
    assert jax_platform_for_device(device) == platform


@pytest.mark.parametrize("device", ["cuda", "rocm", "", "0", "gpu:0"])
def test_an_unknown_device_raises_instead_of_falling_back(device):
    """Silently serving the control loop from the CPU is the failure to avoid."""

    with pytest.raises(LocalPolicyError) as excinfo:
        jax_platform_for_device(device)
    assert LOCAL_POLICY_DEVICE_ENV in str(excinfo.value)


def test_configure_writes_jax_platforms_for_an_explicit_device():
    env = {LOCAL_POLICY_DEVICE_ENV: "gpu"}
    assert configure_jax_platform(env, modules={}) == "cuda"
    assert env["JAX_PLATFORMS"] == "cuda"


def test_configure_leaves_an_operator_export_alone_on_auto():
    """``auto`` is "jax decides", which includes an export the operator made."""

    env = {"JAX_PLATFORMS": "cpu"}
    assert configure_jax_platform(env, modules={}) is None
    assert env["JAX_PLATFORMS"] == "cpu"
    assert DEVICE_PLATFORMS["auto"] is None


def test_configure_refuses_after_jax_is_already_imported():
    env = {LOCAL_POLICY_DEVICE_ENV: "cpu"}
    with pytest.raises(LocalPolicyError) as excinfo:
        configure_jax_platform(env, modules={"jax": object()})
    assert "after jax was imported" in str(excinfo.value)
    assert "JAX_PLATFORMS" not in env


def test_configure_is_a_no_op_after_import_when_the_request_is_auto():
    """Nothing was going to be written, so nothing was lost; do not cry wolf."""

    env: dict[str, str] = {}
    assert configure_jax_platform(env, modules={"jax": object()}) is None
    assert env == {}


# -- the holder contract ---------------------------------------------------- #


class _BrokenHolder:
    def __init__(self, value: Any) -> None:
        self._value = value

    version = 0

    def current(self) -> Any:
        if isinstance(self._value, BaseException):
            raise self._value
        return self._value


@pytest.mark.parametrize(
    "value, expected",
    [
        (RuntimeError("ssh died"), "parameter holder failed"),
        (("params", 3), "must return"),
        (["params", 3, 0.0], "must return"),
        ((None, -1, 0.0), "no parameters yet"),
        (("params", -1, 0.0), "no parameters yet"),
        ((None, 3, 0.0), "no parameters yet"),
        (("params", 1.5, 0.0), "invalid version"),
        (("params", True, 0.0), "invalid version"),
        (("params", 3, "soon"), "must be numeric"),
        (("params", 3, float("nan")), "must be finite"),
    ],
)
def test_a_bad_holder_is_one_policy_inference_error(value, expected):
    """Every way a holder can be wrong surfaces as the same, handled type."""

    with pytest.raises(PolicyInferenceError) as excinfo:
        read_params_holder(_BrokenHolder(value))
    assert expected in str(excinfo.value)


def test_the_real_params_sync_holder_satisfies_this_contract():
    """Cross-check against the holder the proxy will actually be handed.

    ``params_sync.ParamsHolder`` is written by another component to the same
    pinned three-member contract.  A stub agreeing with itself proves nothing;
    this is the seam where a mismatch would only show up on the robot.
    """

    params_sync = pytest.importorskip("ur_env.local_policy.params_sync")

    empty = params_sync.ParamsHolder()
    with pytest.raises(PolicyInferenceError) as excinfo:
        read_params_holder(empty)
    assert "no parameters yet" in str(excinfo.value)

    loaded = params_sync.ParamsHolder({"head": 1}, 12)
    params, version, applied = read_params_holder(loaded)
    assert params == {"head": 1}
    assert version == 12
    assert isinstance(applied, float)

    loaded.swap({"head": 2}, 50)
    assert read_params_holder(loaded)[1] == 50


def test_static_holder_round_trips_and_refuses_a_regression():
    clock = iter([10.0, 20.0, 30.0])
    holder = StaticParamsHolder("v0", 0, clock=lambda: next(clock))

    params, version, applied = holder.current()
    assert (params, version, applied) == ("v0", 0, 10.0)
    assert holder.swap("v5", 5) == 5
    assert holder.current() == ("v5", 5, 20.0)
    assert holder.version == 5
    with pytest.raises(LocalPolicyError):
        holder.swap("v1", 1)
    assert holder.current() == ("v5", 5, 20.0)


def test_static_holder_snapshot_cannot_tear_under_a_concurrent_swap():
    """``current()`` must never mix one version's params with another's."""

    holder = StaticParamsHolder(("params", 0), 0)
    stop = threading.Event()
    torn: list[Any] = []

    def write() -> None:
        version = 0
        while not stop.is_set():
            version += 1
            holder.swap(("params", version), version)

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    try:
        for _ in range(4000):
            params, version, _applied = holder.current()
            if params[1] != version:
                torn.append((params, version))
    finally:
        stop.set()
        writer.join(timeout=5)
    assert not torn


# -- the version clamp ------------------------------------------------------ #


def test_version_stamp_passes_increases_and_gaps_through():
    """Versions arrive with gaps: the puller always skips to the newest blob."""

    stamp = MonotonicVersionStamp()
    assert [stamp.stamp(value) for value in (0, 1, 7, 7, 40)] == [0, 1, 7, 7, 40]
    assert stamp.value == 40
    assert stamp.regression_count == 0


def test_version_stamp_holds_the_number_when_a_holder_regresses():
    warnings: list[str] = []
    stamp = MonotonicVersionStamp(warn=warnings.append)

    stamp.stamp(9)
    assert stamp.stamp(4) == 9
    assert stamp.stamp(0) == 9
    assert stamp.stamp(12) == 12

    assert stamp.regression_count == 2
    assert len(warnings) == 1, "a regression is announced once, not per step"
    assert "OVERSTATES" in warnings[0]


def test_version_stamp_rejects_a_non_counter():
    stamp = MonotonicVersionStamp()
    for bad in (-1, 1.5, True, "3", None):
        with pytest.raises(Exception):
            stamp.stamp(bad)


# =========================================================================== #
# Tier 2: jax, fake agent                                                     #
# =========================================================================== #


class _FakeState:
    def __init__(self, params: Any) -> None:
        self.params = params

    def replace(self, *, params: Any) -> "_FakeState":
        return _FakeState(params)


class _FakeAgent:
    """The smallest object ``LocalPolicyRuntime`` accepts as the agent.

    ``state.params`` is the reference tree every candidate is validated
    against; ``replace``/``sample_actions`` exist so the default sampler path
    is exercised as well as the injected one.
    """

    def __init__(self, params: Any) -> None:
        self.state = _FakeState(params)

    def replace(self, *, state: _FakeState) -> "_FakeAgent":
        agent = _FakeAgent(state.params)
        return agent

    def sample_actions(self, *, observations, seed, argmax):
        del observations, seed
        # A deterministic function OF THE PARAMETERS, so a hot swap is visible.
        bias = float(np.asarray(self.state.params["head"]).sum())
        action = np.full(7, np.tanh(bias) * (0.5 if argmax else 0.25), np.float32)
        action[-1] = 0.0
        return action


def _params(scale: float) -> dict[str, Any]:
    import jax.numpy as jnp

    return {
        "head": jnp.full((2, 3), scale, dtype=jnp.float32),
        "bias": jnp.zeros((3,), dtype=jnp.float32),
    }


def _runtime(
    *,
    holder: Any = None,
    scale: float = 0.1,
    **kwargs: Any,
) -> tuple[LocalPolicyRuntime, Any]:
    agent = _FakeAgent(_params(scale))
    params_holder = (
        StaticParamsHolder(_params(scale), 0) if holder is None else holder
    )
    return LocalPolicyRuntime(agent, params_holder, **kwargs), params_holder


def _observation(value: int = 0) -> dict[str, np.ndarray]:
    from ur_env.learner.policy import canonical_policy_observation

    return canonical_policy_observation(value)


@_needs_jax
def test_construction_warms_both_traces_and_validates_the_action():
    calls: list[bool] = []

    def sampler(params, observation, seed, deterministic):
        calls.append(bool(deterministic))
        assert set(observation) == {"state", "cam1", "cam2"}
        return np.zeros(7, dtype=np.float32)

    runtime, _holder = _runtime(sample_action=sampler)

    assert calls == [True, False], "argmax and stochastic traces both warmed"
    assert runtime.policy_version == 0
    assert runtime.inference_count == 0


@_needs_jax
def test_construction_fails_when_a_smoke_action_is_not_servable():
    """A runtime that exists is a runtime whose health gate may open."""

    def sampler(params, observation, seed, deterministic):
        return np.full(7, 5.0, dtype=np.float32)  # outside [-1, 1]

    with pytest.raises(PolicyValidationError):
        _runtime(sample_action=sampler)


@_needs_jax
def test_construction_rejects_params_that_do_not_fit_the_agent():
    import jax.numpy as jnp

    agent = _FakeAgent(_params(0.1))
    holder = StaticParamsHolder(
        {"head": jnp.zeros((4, 4), jnp.float32), "bias": jnp.zeros((3,), jnp.float32)},
        0,
    )
    with pytest.raises(PolicyValidationError):
        LocalPolicyRuntime(agent, holder)


@_needs_jax
def test_sample_action_matches_the_service_injectable_contract():
    """``(observation, deterministic) -> (float32 (7,), int)``, and it plugs in."""

    runtime, _holder = _runtime()

    action, version = runtime(_observation(3), False)

    assert isinstance(action, np.ndarray)
    assert action.shape == (7,) and action.dtype == np.float32
    assert np.all(np.abs(action) <= 1.0)
    assert isinstance(version, int) and version == 0

    service = ActorSessionService(runtime, model_id=runtime.model_id)
    reply = service.begin_episode(
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
    assert reply.action.shape == (7,)
    assert reply.policy_version == 0
    assert service.get_server_info().model_id == LOCAL_POLICY_MODEL_ID


@_needs_jax
def test_the_returned_action_is_a_private_copy():
    """The caller owns the array it gets; a shared buffer would be poisonable."""

    shared = np.zeros(7, dtype=np.float32)

    def sampler(params, observation, seed, deterministic):
        return shared

    runtime, _holder = _runtime(sample_action=sampler)

    first, _version = runtime(_observation(0), True)
    first[:] = 0.9
    second, _version = runtime(_observation(0), True)

    assert first is not shared and second is not shared
    np.testing.assert_array_equal(second, np.zeros(7, dtype=np.float32))
    np.testing.assert_array_equal(shared, np.zeros(7, dtype=np.float32))


@_needs_jax
def test_a_hot_swap_changes_the_action_on_the_next_call_only():
    """Deterministic proof that the SWAPPED parameters are the ones serving."""

    runtime, holder = _runtime(scale=0.1)
    before, before_version = runtime(_observation(0), True)

    holder.swap(_params(0.9), 7)

    after, after_version = runtime(_observation(0), True)
    again, _version = runtime(_observation(0), True)

    assert before_version == 0 and after_version == 7
    assert not np.allclose(before, after)
    np.testing.assert_allclose(after, again)
    # And the arithmetic is the fake agent's, from the new tree, not a cache.
    expected = np.tanh(float(0.9 * 6)) * 0.5
    assert after[0] == pytest.approx(expected, rel=1e-6)


@_needs_jax
def test_a_call_uses_one_snapshot_even_if_a_swap_lands_mid_call():
    """No tearing: version, params and age all come from the same read."""

    holder = StaticParamsHolder(_params(0.1), 3)
    old_params = holder.current()[0]
    seen_params: list[Any] = []
    calls: list[int] = []

    def sampler(params, observation, seed, deterministic):
        calls.append(1)
        seen_params.append(params)
        # Third call == the first REAL one (two constructor smokes precede it).
        # A swap landing here must not retro-change the reply in flight.
        if len(calls) == 3:
            holder.swap(_params(0.9), 11)
        return np.zeros(7, dtype=np.float32)

    runtime = LocalPolicyRuntime(
        _FakeAgent(_params(0.1)), holder, sample_action=sampler
    )

    _action, version = runtime(_observation(0), False)
    assert version == 3, "the reply carries the version it was serving"
    assert seen_params[-1] is old_params
    assert holder.version == 11

    _action, next_version = runtime(_observation(0), False)
    assert next_version == 11, "the swap takes effect on the NEXT call"
    assert seen_params[-1] is not old_params


@_needs_jax
def test_version_is_never_reported_lower_than_one_already_served():
    warnings: list[str] = []

    class _ScriptedHolder:
        """Reports a scripted version sequence; the last value repeats."""

        def __init__(self, versions: list[int]) -> None:
            self._versions = list(versions)
            self._params = _params(0.1)

        @property
        def version(self) -> int:
            return self._versions[0]

        def current(self):
            value = (
                self._versions.pop(0)
                if len(self._versions) > 1
                else self._versions[0]
            )
            return self._params, value, 100.0

        def swap(self, params, version):  # pragma: no cover - unused
            raise AssertionError("the runtime must never write to the holder")

    # First entry is consumed by the constructor's read, then: 5, 2, 1, 9.
    holder = _ScriptedHolder([5, 5, 2, 1, 9])
    runtime = LocalPolicyRuntime(
        _FakeAgent(_params(0.1)),
        holder,
        warn=warnings.append,
    )

    versions = [runtime(_observation(0), True)[1] for _ in range(4)]

    assert versions == [5, 5, 5, 9], "regressions are held at the last served"
    assert versions == sorted(versions), "a session must never see it decrease"
    assert runtime.version_regression_count == 2
    assert runtime.policy_version == 9
    assert len(warnings) == 1, "announced once, not once per step"


@_needs_jax
def test_a_broken_holder_becomes_a_policy_inference_error_at_call_time():
    runtime, _holder = _runtime()
    runtime._holder = _BrokenHolder(RuntimeError("ssh tunnel died"))

    with pytest.raises(PolicyInferenceError) as excinfo:
        runtime(_observation(0), False)
    assert "parameter holder failed" in str(excinfo.value)


@_needs_jax
def test_a_sampler_failure_becomes_a_policy_inference_error():
    calls: list[int] = []

    def sampler(params, observation, seed, deterministic):
        calls.append(1)
        if len(calls) > 2:  # let the two constructor smokes through
            raise ValueError("XLA said no")
        return np.zeros(7, dtype=np.float32)

    runtime, _holder = _runtime(sample_action=sampler)

    with pytest.raises(PolicyInferenceError) as excinfo:
        runtime(_observation(0), False)
    assert "local policy inference failed" in str(excinfo.value)
    assert "XLA said no" in str(excinfo.value)


@_needs_jax
def test_an_observation_that_is_not_canonical_is_rejected_before_inference():
    calls: list[int] = []

    def sampler(params, observation, seed, deterministic):
        calls.append(1)
        return np.zeros(7, dtype=np.float32)

    runtime, _holder = _runtime(sample_action=sampler)
    before = len(calls)

    with pytest.raises(Exception):
        runtime({"state": np.zeros((1, 19), np.float32)}, False)

    assert len(calls) == before


@_needs_jax
def test_stats_and_latency_gauges_describe_the_call_that_just_happened():
    class _Probe:
        enabled = True

        def __init__(self) -> None:
            self.fields: dict[str, Any] = {}
            self.phases: list[str] = []

        def phase(self, name: str):
            self.phases.append(name)

            class _Scope:
                def __enter__(_self):
                    return _self

                def __exit__(_self, *exc):
                    return False

            return _Scope()

        def set(self, key: str, value: Any) -> None:
            self.fields[key] = value

    probe = _Probe()
    ticks = iter([100.0, 100.0, 130.5, 130.5])
    holder = StaticParamsHolder(_params(0.1), 4, clock=lambda: 100.0)
    runtime = LocalPolicyRuntime(
        _FakeAgent(_params(0.1)),
        holder,
        latency_probe=probe,
        clock=lambda: next(ticks),
    )

    runtime(_observation(0), True)

    stats = runtime.last_inference
    assert stats is not None
    assert stats.params_version == 4
    assert stats.deterministic is True
    assert stats.inference_ms >= 0.0
    assert stats.params_age_s == pytest.approx(0.0)
    assert "local_inference" in probe.phases
    assert probe.fields["params_version"] == 4
    assert probe.fields["params_age_s"] == pytest.approx(0.0)
    assert runtime.inference_count == 1


@_needs_jax
def test_a_probe_that_throws_cannot_break_the_control_path():
    """Profiling observes the control path; it never repairs or breaks it."""

    class _HostileProbe:
        enabled = True

        def phase(self, name: str):
            class _Scope:
                def __enter__(_self):
                    return _self

                def __exit__(_self, *exc):
                    return False

            return _Scope()

        def set(self, key: str, value: Any) -> None:
            raise RuntimeError("disk full")

    runtime, _holder = _runtime(latency_probe=_HostileProbe())

    action, version = runtime(_observation(0), False)
    assert action.shape == (7,) and version == 0


@_needs_jax
def test_validate_candidate_is_the_seam_params_sync_calls_before_swapping():
    """A bad blob must blow up on the puller thread, not in a Step handler."""

    import jax.numpy as jnp

    seen: list[Any] = []
    runtime, _holder = _runtime(parameter_validator=seen.append)

    runtime.validate_candidate(_params(0.5))
    assert len(seen) == 2, "constructor + explicit call"

    with pytest.raises(PolicyValidationError):
        runtime.validate_candidate(
            {"head": jnp.zeros((2, 3), jnp.float32)}  # missing a leaf
        )
    with pytest.raises(PolicyValidationError):
        runtime.validate_candidate(
            {
                "head": jnp.full((2, 3), jnp.nan, jnp.float32),
                "bias": jnp.zeros((3,), jnp.float32),
            }
        )


@_needs_jax
def test_a_failing_parameter_invariant_is_a_policy_validation_error():
    """The frozen trunk check ``params_sync`` grafts with lands here."""

    def validator(params: Any) -> None:
        raise ValueError("frozen ResNet-10 trunk weights changed")

    with pytest.raises(PolicyValidationError) as excinfo:
        _runtime(parameter_validator=validator)
    assert "parameter invariant failed" in str(excinfo.value)


@_needs_jax
def test_the_default_sampler_runs_the_agent_with_the_holders_params():
    """No injected sampler: the production path through ``agent.replace``."""

    runtime, holder = _runtime(scale=0.2)

    action, _version = runtime(_observation(0), True)

    expected = np.tanh(float(0.2 * 6)) * 0.5
    assert action[0] == pytest.approx(expected, rel=1e-6)


@_needs_jax
def test_a_holder_missing_the_contract_is_rejected_at_construction():
    class _NotAHolder:
        pass

    with pytest.raises(TypeError):
        LocalPolicyRuntime(_FakeAgent(_params(0.1)), _NotAHolder())


# =========================================================================== #
# Tier 3: the real production agent                                           #
# =========================================================================== #


@_needs_real_agent
def test_real_agent_serves_actions_and_hot_swaps_deterministically(tmp_path):
    """Build the network the learner serves, then swap two known param sets.

    This is the only test that proves the local runtime and the server runtime
    are the SAME policy: same factory, same canonical observation, same action
    contract -- and that a parameter swap changes the action rather than being
    silently cached by a jit trace.
    """

    import jax

    from ur_env.learner import LearnerConfig
    from ur_env.local_policy.runtime import build_local_policy_agent

    resnet = (
        _REPO_ROOT
        / "third_party"
        / "hil-serl"
        / "examples"
        / "experiments"
        / "resnet10_params.pkl"
    )
    if not resnet.is_file():  # pragma: no cover - environment dependent
        pytest.skip(f"missing {resnet}")

    agent = build_local_policy_agent(
        config=LearnerConfig(),
        resnet_source_path=resnet,
        resnet_cache_path=tmp_path / "verified-resnet10.pkl",
        validate_versions=False,
    )
    baseline = agent.state.params
    holder = StaticParamsHolder(baseline, 0)
    runtime = LocalPolicyRuntime(agent, holder)

    deterministic, version = runtime(_observation(0), True)
    stochastic, _version = runtime(_observation(0), False)

    assert deterministic.shape == (7,) and deterministic.dtype == np.float32
    assert np.all(np.abs(deterministic) <= 1.0)
    assert float(deterministic[-1]) in (-1.0, 0.0, 1.0)
    assert version == 0
    assert runtime.last_inference is not None
    assert runtime.last_inference.inference_ms > 0.0

    # A second, genuinely different parameter set: perturb every trainable leaf
    # while leaving the frozen trunk exactly as it is, which is what the real
    # export/graft produces.
    trunk = ("modules_actor", "encoder", "encoder_cam1", "pretrained_encoder")

    def perturb(path, leaf):
        names = tuple(getattr(entry, "key", "") for entry in path)
        if names[: len(trunk)] == trunk:
            return leaf
        return leaf + 0.05

    perturbed = jax.tree_util.tree_map_with_path(perturb, baseline)
    holder.swap(perturbed, 50)

    swapped, swapped_version = runtime(_observation(0), True)

    assert swapped_version == 50
    assert not np.allclose(deterministic, swapped), "the swap did nothing"
    repeat, _version = runtime(_observation(0), True)
    np.testing.assert_allclose(swapped, repeat)


@_needs_real_agent
def test_real_agent_accepts_the_frozen_trunk_feature_observation(tmp_path):
    """The dual-input encoder is why the same runtime can serve both forms."""

    from ur_env.learner import (
        FrozenResNet10TrunkExtractor,
        LearnerConfig,
    )
    from ur_env.local_policy.runtime import build_local_policy_agent

    resnet = (
        _REPO_ROOT
        / "third_party"
        / "hil-serl"
        / "examples"
        / "experiments"
        / "resnet10_params.pkl"
    )
    if not resnet.is_file():  # pragma: no cover - environment dependent
        pytest.skip(f"missing {resnet}")

    agent = build_local_policy_agent(
        config=LearnerConfig(),
        resnet_source_path=resnet,
        resnet_cache_path=tmp_path / "verified-resnet10.pkl",
        validate_versions=False,
    )
    runtime = LocalPolicyRuntime(agent, StaticParamsHolder(agent.state.params, 0))
    extractor = FrozenResNet10TrunkExtractor(agent, resnet_asset_path=resnet)

    features = extractor(_observation(0))
    action, version = runtime(features, True)

    assert action.shape == (7,) and version == 0
