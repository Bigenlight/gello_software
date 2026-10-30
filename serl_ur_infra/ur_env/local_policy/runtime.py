"""Serve the production policy from laptop3's own GPU.

WHAT THIS IS
------------
:class:`LocalPolicyRuntime` is the ``sample_action`` injectable of
:class:`ur_env.actor_network.ActorSessionService`, exactly like the learner's
:class:`ur_env.learner.policy.VersionedPolicyRuntime` is on the server.  Both
answer the same call -- ``runtime(observation, deterministic) -> (action,
policy_version)`` -- and both serve the SAME network built by
``create_frozen_trunk_feature_agent``.  The proxy is therefore not a different
policy; it is the same policy evaluated on a different machine.

HOW IT DIFFERS FROM ``VersionedPolicyRuntime``, AND WHY IT IS NOT IT
--------------------------------------------------------------------
``VersionedPolicyRuntime`` OWNS its snapshot and mints versions itself
(``publish`` requires a strictly increasing ``learner_step`` and adds exactly
one to the version).  Here the versions are minted somewhere else entirely --
by the learner, published in ``LATEST.json``, and pulled by
``ur_env.local_policy.params_sync`` -- and they arrive with GAPS, because the
puller always skips to the newest blob and never applies a stale one.  So this
runtime does not own parameters at all: it reads them, once per call, from a
holder object (see :class:`ParamsHolder`).  Everything else -- the input
validator, the action validator, the two-trace smoke, the tree validation --
is the server's, imported rather than re-implemented, so the two cannot drift.

VERSION STAMPING IS MONOTONIC, EVEN IF THE HOLDER IS NOT
--------------------------------------------------------
``ActorSessionService.step`` REJECTS a policy_version lower than the one it
last saw ("policy_version decreased from N to M") and that error ends the
session.  A holder is supposed to refuse regressions itself, but a bug there
must not cost an episode on the robot, so the version this runtime stamps is
clamped to never decrease within a process.  A clamp is a lie about which
parameters produced the action, so it is counted
(``version_regression_count``) and warned about the first time it happens.

DEVICE SELECTION IS THE ENTRYPOINT'S JOB
----------------------------------------
jax reads ``JAX_PLATFORMS`` (and ``XLA_PYTHON_CLIENT_PREALLOCATE``) when it is
first imported, so nothing in this module can choose a device: by the time an
agent is built the choice is already made.  :func:`configure_jax_platform`
implements the ``HIL_LOCAL_POLICY_DEVICE`` contract and MUST be called from the
process entrypoint before the first ``import jax`` -- it refuses, loudly,
if jax is already imported.  This module imports jax only inside functions, so
importing it does not itself lose that race.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
import threading
import time
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from ur_env.actor_network import PolicyInferenceError, validate_counter

# ``_validated_policy_input`` is private over there and imported on purpose
# here, together with ``VersionedPolicyRuntime._validated_policy_action``
# below.  Both are the exact checks the SERVER applies to the same tensors, and
# a local copy of either would be a second contract that silently drifts:
#   _validated_policy_input   accepts the canonical pixel observation OR its
#                             frozen-trunk features and rejects anything else.
#   _validated_policy_action  materializes, dtype/shape/range-checks the action
#                             and pins the gripper component to {-1, 0, 1}.
from ur_env.learner.policy import (
    PolicyValidationError,
    VersionedPolicyRuntime,
    _validated_policy_input,
    canonical_policy_observation,
    validate_parameter_tree,
)

_validated_policy_action = VersionedPolicyRuntime._validated_policy_action


#: Advertised as ``ServerInfo.model_id`` by the proxy.  It is deliberately NOT
#: the learner's ``FROZEN_TRUNK_MODEL_REVISION``: the network is the same, but
#: the reward path is not (MANUAL only, no classifier), and the actor's
#: expected-model-id preflight is the operator's proof of which one they got.
LOCAL_POLICY_MODEL_ID = "hil-serl-local-policy-resnet10-manual-v1"

#: ``cpu`` forces the CPU backend; ``gpu`` demands CUDA; ``auto`` (default)
#: leaves the choice to jax, which picks the GPU when the CUDA plugin sees one.
LOCAL_POLICY_DEVICE_ENV = "HIL_LOCAL_POLICY_DEVICE"

#: ``HIL_LOCAL_POLICY_DEVICE`` -> ``JAX_PLATFORMS``.  ``None`` means "do not
#: set it".  Mirrors ``scripts/bench_local_policy.py::DEVICE_PLATFORMS`` so the
#: number the bench printed and the number the proxy serves come from the same
#: backend selection.
DEVICE_PLATFORMS: dict[str, Optional[str]] = {
    "cpu": "cpu",
    "gpu": "cuda",
    "auto": None,
}


class LocalPolicyError(RuntimeError):
    """The local policy runtime cannot be built or cannot serve."""


# --------------------------------------------------------------------------- #
# Device selection (pure python: callable before jax exists in the process)    #
# --------------------------------------------------------------------------- #


def jax_platform_for_device(device: str) -> Optional[str]:
    """Return the ``JAX_PLATFORMS`` value for one device request."""

    if not isinstance(device, str):
        raise LocalPolicyError(f"{LOCAL_POLICY_DEVICE_ENV} must be a string")
    key = device.strip().lower()
    if key not in DEVICE_PLATFORMS:
        raise LocalPolicyError(
            f"{LOCAL_POLICY_DEVICE_ENV} must be one of "
            f"{sorted(DEVICE_PLATFORMS)}, got {device!r}"
        )
    return DEVICE_PLATFORMS[key]


def configure_jax_platform(
    env: Optional[dict[str, str]] = None,
    *,
    modules: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Apply ``HIL_LOCAL_POLICY_DEVICE`` to ``env`` before jax is imported.

    Returns the ``JAX_PLATFORMS`` value that was set, or ``None`` for ``auto``
    (in which case nothing is written and an operator's own export survives).

    An unknown value RAISES rather than falling back to auto: silently running
    the control loop on the CPU because someone typed ``cuda`` instead of
    ``gpu`` is the failure mode this whole phase exists to remove.  Being
    called after jax is imported also raises -- the setting would be ignored,
    and an ignored device request is the same silent failure.
    """

    target = os.environ if env is None else env
    loaded = sys.modules if modules is None else modules
    platform = jax_platform_for_device(str(target.get(LOCAL_POLICY_DEVICE_ENV, "auto")))
    if platform is None:
        return None
    if "jax" in loaded:
        raise LocalPolicyError(
            f"{LOCAL_POLICY_DEVICE_ENV} was applied after jax was imported; "
            "the request would be ignored.  Call configure_jax_platform() at "
            "the very top of the entrypoint, before any jax import."
        )
    target["JAX_PLATFORMS"] = platform
    return platform


# --------------------------------------------------------------------------- #
# Parameter holder contract                                                    #
# --------------------------------------------------------------------------- #


@runtime_checkable
class ParamsHolder(Protocol):
    """The read side of the hot-swappable parameter cell.

    Implemented for real by ``ur_env.local_policy.params_sync``; this runtime
    only ever READS it.  ``swap`` belongs to the puller thread and is declared
    here only so the two halves of the contract stay written down together.
    """

    @property
    def version(self) -> int:
        """Version of the parameters ``current()`` would return right now."""

    def current(self) -> tuple[Any, int, float]:
        """Return ``(params, version, applied_monotonic)`` as one snapshot.

        The tuple must be internally consistent: a swap concurrent with this
        call either happens entirely before it or entirely after it.
        """

    def swap(self, params: Any, version: int, *args: Any, **kwargs: Any) -> Any:
        """Install validated parameters; the writer side, never called here."""


def read_params_holder(holder: ParamsHolder) -> tuple[Any, int, float]:
    """Take ONE consistent snapshot from a holder, or fail as an inference error.

    Split out of :class:`LocalPolicyRuntime` so the contract a foreign holder
    has to satisfy is checkable without jax -- and so every way a holder can be
    wrong (raising, wrong arity, negative version, NaN timestamp) is one
    ``PolicyInferenceError`` rather than an ``AttributeError`` inside a Step
    handler.
    """

    try:
        current = holder.current()
    except Exception as exc:
        raise PolicyInferenceError(
            f"parameter holder failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(current, tuple) or len(current) != 3:
        raise PolicyInferenceError(
            "params holder.current() must return "
            "(params, version, applied_monotonic)"
        )
    params, version, applied = current
    if params is None or (isinstance(version, int) and version < 0):
        # ``params_sync.NO_PARAMS_VERSION`` (-1) is the empty holder.  Serving
        # from it is not a degraded mode, it is impossible, so say which of the
        # two states this is rather than emitting a counter-validation error.
        raise PolicyInferenceError(
            "the parameter holder has no parameters yet (version "
            f"{version!r}); the proxy must not serve, or report SERVING, "
            "until the first parameter blob has been loaded"
        )
    try:
        checked = validate_counter(version, name="params_version")
    except Exception as exc:
        raise PolicyInferenceError(
            f"params holder returned an invalid version: {exc}"
        ) from exc
    try:
        applied_monotonic = float(applied)
    except (TypeError, ValueError) as exc:
        raise PolicyInferenceError(
            "params holder applied_monotonic must be numeric"
        ) from exc
    if not math.isfinite(applied_monotonic):
        raise PolicyInferenceError("params holder applied_monotonic must be finite")
    return params, checked, applied_monotonic


class MonotonicVersionStamp:
    """Clamp reported policy versions so a session can never see one decrease.

    ``ActorSessionService.step`` ends the session on "policy_version decreased
    from N to M".  The puller is supposed to make that impossible; this makes a
    bug there cost a wrong NUMBER instead of an episode on the robot.  Because
    a clamp IS a wrong number -- it credits an action to parameters that did
    not produce it -- it is counted and announced once.
    """

    def __init__(
        self,
        initial: int = 0,
        *,
        warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._value = validate_counter(initial, name="initial version")
        self._warn = warn or _emit_local_policy_warning
        self._lock = threading.Lock()
        self.regression_count = 0

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def stamp(self, version: int) -> int:
        """Return the version to report for parameters at ``version``."""

        checked = validate_counter(version, name="params_version")
        with self._lock:
            if checked >= self._value:
                self._value = checked
                return checked
            served = self._value
            self.regression_count += 1
            first = self.regression_count == 1
        if first:
            self._warn(
                f"parameter holder reported version {checked} after {served} "
                "had already been served; replies keep the higher number "
                "because ActorSessionService ends a session whose "
                "policy_version decreases.  The action was produced by the "
                "parameters the holder returned, so the stamp now OVERSTATES "
                "them -- the param puller is misbehaving."
            )
        return served


class StaticParamsHolder:
    """A minimal thread-safe holder: the bootstrap cell and the test double.

    Real enough for production use as the cell the proxy starts with (initial
    params, version 0) before the puller thread exists, and small enough to be
    the double every test in this package swaps by hand.
    """

    def __init__(
        self,
        params: Any,
        version: int = 0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._params = params
        self._version = validate_counter(version, name="version")
        self._applied_monotonic = float(clock())

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def current(self) -> tuple[Any, int, float]:
        with self._lock:
            return self._params, self._version, self._applied_monotonic

    def swap(self, params: Any, version: int, *args: Any, **kwargs: Any) -> int:
        """Install ``params`` at ``version``; refuses a version regression."""

        del args, kwargs
        checked = validate_counter(version, name="version")
        with self._lock:
            if checked < self._version:
                raise LocalPolicyError(
                    f"parameter version regressed from {self._version} to "
                    f"{checked}"
                )
            self._params = params
            self._version = checked
            self._applied_monotonic = float(self._clock())
            return self._version


# --------------------------------------------------------------------------- #
# Agent construction and smoke                                                 #
# --------------------------------------------------------------------------- #


def build_local_policy_agent(
    *,
    config: Any = None,
    hil_serl_root: Any = None,
    resnet_source_path: Any = None,
    resnet_cache_path: Any = None,
    validate_versions: bool = False,
) -> Any:
    """Build the production agent the learner serves, on this machine.

    Identical call to the one the learner and ``scripts/bench_local_policy.py``
    make, so the parameter tree the server exports fits this agent leaf for
    leaf.  ``validate_versions`` defaults to False for the same reason the
    bench's does: the laptop's venv pins are its own, and the tree-shape
    validation in ``params_sync`` is the check that actually matters here.
    """

    from ur_env.learner import LearnerConfig, create_frozen_trunk_feature_agent

    return create_frozen_trunk_feature_agent(
        config=LearnerConfig() if config is None else config,
        hil_serl_root=hil_serl_root,
        resnet_source_path=resnet_source_path,
        resnet_cache_path=resnet_cache_path,
        validate_versions=validate_versions,
    )


def smoke_policy_traces(
    sample_action: Callable[[Any, Mapping[str, Any], Any, bool], Any],
    params: Any,
) -> None:
    """Materialize BOTH policy traces and validate what they return.

    The deterministic (argmax) and stochastic paths of the production SAC agent
    compile separately, and the feasibility study measured 0.7-1.6 s for the
    first call of each.  Paying that at startup is the whole point: an actor
    whose first Step compiles blows its RPC deadline.

    Same seeds and labels as ``VersionedPolicyRuntime._smoke``, which is the
    server-side original.  It is copied rather than shared because
    ``ur_env/learner/`` is not this component's to refactor; if this file and
    that method ever disagree, that method is right.
    """

    import jax

    observation = canonical_policy_observation()
    for deterministic, seed_value, label in (
        (True, 0, "deterministic policy smoke action"),
        (False, 1, "stochastic policy smoke action"),
    ):
        action = sample_action(
            params,
            observation,
            jax.random.PRNGKey(seed_value),
            deterministic,
        )
        _validated_policy_action(action, name=label)


# --------------------------------------------------------------------------- #
# The runtime                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LocalInferenceStats:
    """What the last served action cost and which parameters produced it."""

    inference_ms: float
    params_version: int
    params_age_s: float
    deterministic: bool


def _emit_local_policy_warning(message: str) -> None:
    """One operator-visible stderr line; not ``logging`` (nothing configures it)."""

    print(f"[local-policy] WARNING: {message}", file=sys.stderr, flush=True)


class LocalPolicyRuntime:
    """Serve ``(action, policy_version)`` from hot-swappable local parameters.

    Construction VALIDATES and SMOKES the holder's current parameters, so a
    successfully constructed runtime is one whose health gate may go SERVING.
    """

    model_id = LOCAL_POLICY_MODEL_ID

    def __init__(
        self,
        agent: Any,
        holder: ParamsHolder,
        *,
        model_id: str = LOCAL_POLICY_MODEL_ID,
        inference_rng: Any | None = None,
        sample_action: Optional[
            Callable[[Any, Mapping[str, Any], Any, bool], Any]
        ] = None,
        parameter_validator: Callable[[Any], None] | None = None,
        latency_probe: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id is required")
        for name in ("current", "version"):
            if not hasattr(holder, name):
                raise TypeError(
                    f"params holder must expose {name!r}; see ParamsHolder"
                )
        if parameter_validator is not None and not callable(parameter_validator):
            raise TypeError("parameter_validator must be callable")

        import jax

        self._agent = agent
        self._holder = holder
        self.model_id = model_id
        self._sample_action = sample_action or self._sample_with_agent
        self._parameter_validator = parameter_validator
        self._clock = clock
        self._warn = warn or _emit_local_policy_warning
        self._lock = threading.Lock()
        self._reference_params = agent.state.params
        self._version_stamp = MonotonicVersionStamp(warn=self._warn)
        self._inference_rng = (
            jax.random.PRNGKey(42) if inference_rng is None else inference_rng
        )
        try:
            jax.random.split(self._inference_rng)
        except Exception as exc:
            raise ValueError("inference_rng must be a valid JAX PRNG key") from exc

        # Opt-in profiling.  A None probe is a shared no-op, exactly as in
        # RewardTransitionFinalizer, so nothing on the control path branches on
        # whether the operator asked for a latency file.
        self._latency = latency_probe if latency_probe is not None else _disabled_probe()

        self.inference_count = 0
        self.last_inference: Optional[LocalInferenceStats] = None

        params, version, _applied = read_params_holder(self._holder)
        self._version_stamp.stamp(version)
        # Validate + smoke BEFORE this constructor returns: "the runtime
        # exists" and "both traces are compiled and produce legal actions" have
        # to be the same fact, or health can advertise SERVING too early.
        self.validate_candidate(params)

    # -- parameters --------------------------------------------------------- #

    def validate_candidate(self, params: Any) -> None:
        """Tree-check, invariant-check and smoke candidate parameters.

        The seam ``params_sync`` calls before it swaps: everything a bad blob
        could do to the control loop happens HERE, on the puller thread, while
        the previous parameters keep serving.
        """

        validate_parameter_tree(params, self._reference_params)
        if self._parameter_validator is not None:
            try:
                self._parameter_validator(params)
            except Exception as exc:
                raise PolicyValidationError(
                    f"parameter invariant failed: {type(exc).__name__}: {exc}"
                ) from exc
        smoke_policy_traces(self._sample_action, params)

    @property
    def policy_version(self) -> int:
        """The version the next reply would carry (monotonic in this process)."""

        return self._version_stamp.value

    @property
    def version_regression_count(self) -> int:
        """Times the holder went backwards and the stamp had to be held."""

        return self._version_stamp.regression_count

    def params_age_s(self) -> float:
        """Seconds since the currently loaded parameters were applied."""

        _params, _version, applied = read_params_holder(self._holder)
        return max(0.0, float(self._clock()) - applied)

    # -- inference ---------------------------------------------------------- #

    def _sample_with_agent(
        self,
        params: Any,
        observation: Mapping[str, Any],
        seed: Any,
        deterministic: bool,
    ) -> Any:
        candidate = self._agent.replace(
            state=self._agent.state.replace(params=params)
        )
        return candidate.sample_actions(
            observations=observation,
            seed=seed,
            argmax=deterministic,
        )

    def __call__(
        self, observation: Mapping[str, Any], deterministic: bool
    ) -> tuple[np.ndarray, int]:
        """The ``sample_action`` injectable of ``ActorSessionService``."""

        canonical = _validated_policy_input(observation)

        import jax

        # ONE read of the holder per call.  A swap that lands halfway through
        # this method changes nothing that has already been decided: the params
        # reference, its version and its age are all taken from this snapshot,
        # so the reply describes exactly the tree that produced it and the new
        # parameters take effect on the NEXT call.
        params, holder_version, applied_monotonic = read_params_holder(self._holder)
        version = self._version_stamp.stamp(holder_version)
        age_s = max(0.0, float(self._clock()) - applied_monotonic)

        with self._lock:
            self._inference_rng, action_rng = jax.random.split(self._inference_rng)

        started = time.perf_counter()
        try:
            with self._latency.phase("local_inference"):
                value = self._sample_action(
                    params, canonical, action_rng, bool(deterministic)
                )
                action = _validated_policy_action(value, name="policy action")
        except Exception as exc:
            if isinstance(exc, PolicyInferenceError):
                raise
            raise PolicyInferenceError(
                f"local policy inference failed: {type(exc).__name__}: {exc}"
            ) from exc
        inference_ms = (time.perf_counter() - started) * 1000.0

        self.inference_count += 1
        # One atomic rebind, so a reader never sees a half-written stat block.
        self.last_inference = LocalInferenceStats(
            inference_ms=inference_ms,
            params_version=version,
            params_age_s=age_s,
            deterministic=bool(deterministic),
        )
        self._record_gauges(version, age_s)
        return action.copy(), version

    def _record_gauges(self, version: int, age_s: float) -> None:
        """Attach the params gauges to this Step's latency record, if profiling.

        Wrapped: profiling observes the control path, it never breaks it.
        """

        try:
            if not getattr(self._latency, "enabled", False):
                return
            self._latency.set("params_version", int(version))
            self._latency.set("params_age_s", round(float(age_s), 3))
        except Exception:  # noqa: BLE001
            pass


def _disabled_probe() -> Any:
    """The shared no-op probe, imported lazily to keep this module light."""

    from ur_env.server_latency import disabled_probe

    return disabled_probe()


__all__ = [
    "DEVICE_PLATFORMS",
    "LOCAL_POLICY_DEVICE_ENV",
    "LOCAL_POLICY_MODEL_ID",
    "LocalInferenceStats",
    "LocalPolicyError",
    "LocalPolicyRuntime",
    "MonotonicVersionStamp",
    "ParamsHolder",
    "StaticParamsHolder",
    "build_local_policy_agent",
    "configure_jax_platform",
    "jax_platform_for_device",
    "read_params_holder",
    "smoke_policy_traces",
]
