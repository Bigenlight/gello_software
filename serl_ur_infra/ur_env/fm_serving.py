"""Serve a rectified-flow (FM) policy over the standard policy-callable contract.

WHAT THIS IS
------------
``ur_env/learner/flow_matching.py`` holds a JAX rectified-flow model trained on
the canonical cube-in-cup demonstrations.  It generates an *action chunk* of
shape ``(B, horizon, 7)`` from *frozen-trunk feature* observations
(``cam1``/``cam2`` ResNet-10 maps ``(B, 1, 4, 4, 512)`` plus ``state``).

The live actor speaks neither of those languages.  It sends the canonical
*pixel* observation (``cam1``/``cam2`` uint8 ``(1, 128, 128, 3)`` + ``state``
float32 ``(1, 19)``) and expects exactly one action back:

    ``policy(observation, deterministic) -> (np.ndarray (7,) float32, int)``

(``ActorSessionService._infer``, ``ur_env/actor_network.py:1505``).

:class:`FmServedPolicy` is the adapter between those two.  It owns the two
translations and nothing else:

    pixels --(FrozenResNet10TrunkExtractor)--> features --(sample_action_chunks)-->
    chunk --(first_action_from_chunk)--> one 7-D action

WHY ONLY THE FIRST ACTION OF THE CHUNK -- LOAD-BEARING
------------------------------------------------------
This is a *receding-horizon* server: every Step re-plans a full ``horizon``-long
chunk from the freshest observation and executes only its first action.  That
keeps the FM policy on the same contract as every other policy in this stack --
one observation in, one action out, no hidden server-side state -- so the actor,
the replay ingress and the recording sink need no changes at all.

Executing the rest of a chunk open-loop ("chunk execution", the mode that makes
diffusion/FM policies cheap) would require per-session state on the server: a
buffer keyed by ``(actor_id, session_id)``, an eviction rule when an episode
ends or an operator aborts, and a decision about what a mid-chunk intervention
invalidates.  gRPC serves Step from several handler threads and the transport
also *caches* replies, so a naive buffer would happily hand a stale queued
action to a re-sent Step.  None of that exists here and it is deliberately out
of scope; ``first_action_from_chunk`` is the single place the selection lives,
so a future chunk-execution mode has exactly one call site to reconsider.

WHY ``deterministic`` IS IGNORED -- INTENTIONAL
-----------------------------------------------
Flow matching integrates a velocity field starting from Gaussian noise
(``flow_matching.py:250``).  There is no "mean action" to return: a determinism
flag could only be honoured by pinning the initial noise, which would freeze the
policy onto one arbitrary sample of its own distribution rather than onto its
expectation.  So the wire flag is read and dropped.  In practice the actor
always sends ``False`` anyway (``ur_env/remote_actor.py`` requests stochastic
actions for online data collection), so this discards nothing that is being
asked for.  Reproducibility is instead served by ``rng_seed``: one server
process with a fixed seed replays the same noise sequence for the same number of
calls, which is as close to reproducible as a stochastic policy can honestly be.

WHY THERE IS NO PER-GROUP CLAMPING HERE
---------------------------------------
``sample_action_chunks`` already clips the whole chunk to ``[-1, 1]``
(``flow_matching.py:263``) and, with ``discretize_gripper=True`` -- its default,
which this module keeps -- forces channel 6 to exactly ``-1.0``/``+1.0``
(``flow_matching.py:264-268``).  Re-clamping here would silently *repair* a
model or an integrator that had started producing out-of-range velocities, and
this stack has already been bitten once by a policy path that quietly rescaled
what it was handed.  ``first_action_from_chunk`` therefore *validates* and
raises instead of fixing: if the invariant the sampler promises is ever broken,
the session fails loudly at the first inference rather than driving a real UR7e
with an action nobody checked.

Note also that the 6 EEF channels are *not* norm-scaled here.  That rule
(``CLAUDE.md``: "no per-axis clip, norm-proportional shrink only") governs the
*intervention* path, where a recorded action is transported through
``RelativeFrame`` rotations.  A policy action goes the other way -- it is
consumed as-is by ``PolicyDeltaController`` in the frame it was produced for --
and it is exactly what the demonstrations were trained on, so rescaling it here
would change the action semantics the model learned.

WHY THERE IS NO ``prime_observation`` HERE -- LOAD-BEARING ABSENCE
------------------------------------------------------------------
``ActorSessionService._prime_replay_observation`` (``actor_network.py:1476``)
probes the *sink* -- not the policy -- for a ``prime_observation`` attribute and
switches the observation the policy receives to pre-encoded features when it
finds one.  This policy must keep receiving *raw pixels*, because its own
extractor is what turns them into features.  This class therefore defines no
such attribute and, deliberately, no delegating ``__getattr__`` that could grow
one by accident.  Do not add either.

IMPORTS
-------
Module level is stdlib + numpy only, so this file imports cleanly inside the
actor venv (``/home/laptop3/venvs/gello-hil-actor/bin/python``), which has no
jax -- the actor-side code and the tests that only exercise
``first_action_from_chunk`` must not drag in an accelerator stack.  jax and
``ur_env.learner.flow_matching`` are imported lazily inside the methods that
actually need them, which is also the convention the learner modules follow
(``frozen_trunk.py``: "Heavy JAX/Flax/HIL-SERL imports remain inside
factories").
"""

from __future__ import annotations

import threading
from typing import Any, Mapping

import numpy as np


#: Artifact ``format`` string written by the FM trainer and checked by
#: ``load_flow_artifact`` (``flow_matching.py:291``).  Mirrored here so a server
#: script can assert the manifest identity without importing jax.
FM_ARTIFACT_FORMAT = "hil-serl-jax-flow-matching"

#: Advertised over the wire.  An actor pinning ``EXPECTED_MODEL_ID`` cannot be
#: pointed at the online SAC learner or at the BC server by accident, and vice
#: versa.  Encodes the corpus and the sampler settings the artifact was built
#: with (horizon 16, 8 Euler steps).
FM_MODEL_ID = "fm-cube-in-cup-raw0731-h16-euler8-v1"

#: FM evaluation runs MANUAL, exactly like BC evaluation: the operator's
#: ``MARK SUCCESS`` is the only reward authority and no classifier checkpoint
#: scores these episodes.  Same string as ``bc_init.BC_REWARD_MODEL_ID`` --
#: identical meaning, so it is deliberately not a different id.
FM_REWARD_MODEL_ID = "operator-manual-success-v1"

#: Action contract of this stack: 6 normalized EEF deltas + 1 gripper channel.
ACTION_DIM = 7
GRIPPER_INDEX = 6

#: The only values channel 6 may carry.  ``+-1`` is what
#: ``discretize_gripper=True`` produces; ``0.0`` is admitted because a
#: zero-initialized or scripted policy legitimately emits it and it is not an
#: unsafe command (it is the neutral gripper delta).
_ALLOWED_GRIPPER_VALUES = (-1.0, 0.0, 1.0)

#: Single-observation frozen-trunk feature shape as
#: ``FrozenResNet10TrunkExtractor`` returns it (``(T=1, 4, 4, 512)``;
#: ``ur_env/learner/config.py:18``).  The FM encoder wants that same tensor with
#: a leading batch axis -- see ``_fm_camera_feature``.
FEATURE_SHAPE = (1, 4, 4, 512)

#: Canonical proprioceptive observation, ``(T=1, 19)`` for a single step.
STATE_SHAPE = (1, 19)

_CAMERA_KEYS = ("cam1", "cam2")
_OBSERVATION_KEYS = frozenset({"cam1", "cam2", "state"})


class FmServingError(RuntimeError):
    """An FM inference did not satisfy the served-policy contract.

    Raised for a malformed action chunk, a feature observation that does not
    match what the FM model was trained on, and any failure inside the
    extractor or the sampler.  ``ActorSessionService._infer`` converts whatever
    escapes the policy callback into ``PolicyInferenceError``, so this type is
    about diagnosis, not control flow: the message always carries the offending
    values.
    """


def first_action_from_chunk(chunk: Any) -> np.ndarray:
    """Select and validate the action the robot will actually execute.

    Accepts ``(horizon, 7)`` or ``(1, horizon, 7)`` -- the unbatched form for
    convenience, the batched form because that is literally what
    ``sample_action_chunks`` returns for ``B=1``.  Returns the first chunk
    entry as a fresh contiguous ``float32`` array of shape ``(7,)``.

    This is the **only** place the chunk-to-action selection lives; see the
    module docstring for why it is the first entry and why nothing here clamps.

    Pure numpy on purpose: a jax array is accepted (numpy reads it through the
    array protocol), but nothing in this function needs jax, so the rejection
    tests run in the actor venv.

    Raises:
        FmServingError: on a wrong shape, a non-finite value, any component
            outside ``[-1, 1]``, or a gripper channel that is not one of
            ``-1.0``/``0.0``/``+1.0``.  The message names the offending values.
    """

    array = np.asarray(chunk)
    shape = tuple(int(dimension) for dimension in array.shape)
    if len(shape) == 3:
        if shape[0] != 1:
            raise FmServingError(
                "action chunk must carry exactly one batch row, got shape "
                f"{shape}; this server plans for a single observation"
            )
    elif len(shape) != 2:
        raise FmServingError(
            f"action chunk must have shape (horizon, {ACTION_DIM}) or "
            f"(1, horizon, {ACTION_DIM}), got shape {shape}"
        )
    if shape[-1] != ACTION_DIM:
        raise FmServingError(
            f"action chunk must have {ACTION_DIM} action channels, got shape "
            f"{shape}"
        )
    if shape[-2] < 1:
        raise FmServingError(
            f"action chunk has an empty horizon, got shape {shape}"
        )

    # ``[..., 0, :]`` is the first horizon entry for both accepted ranks.  The
    # copy is deliberate: the source may alias a device buffer that jax is free
    # to reuse, and the caller receives an array it owns.
    action = np.array(array[..., 0, :], dtype=np.float32).reshape(ACTION_DIM)

    if not np.isfinite(action).all():
        raise FmServingError(
            f"first chunk action is not finite: {action.tolist()}"
        )
    outside = np.flatnonzero(np.abs(action) > 1.0)
    if outside.size:
        offenders = ", ".join(
            f"[{int(index)}]={float(action[index])!r}" for index in outside
        )
        raise FmServingError(
            f"first chunk action escapes [-1, 1]: {offenders} "
            f"(full action {action.tolist()})"
        )
    gripper = float(action[GRIPPER_INDEX])
    if gripper not in _ALLOWED_GRIPPER_VALUES:
        raise FmServingError(
            f"gripper channel [{GRIPPER_INDEX}] must be one of "
            f"{list(_ALLOWED_GRIPPER_VALUES)}, got {gripper!r} "
            f"(full action {action.tolist()})"
        )
    return action


def _fm_camera_feature(value: Any, *, name: str) -> np.ndarray:
    """Shape one extracted camera feature the way the FM encoder wants it.

    ``FrozenResNet10TrunkExtractor.__call__`` returns ``(1, 4, 4, 512)`` for a
    single canonical observation -- the ``T=1`` stacking axis, no batch axis
    (``frozen_trunk.py:701``).  ``_SpatialFeatureEncoder`` requires rank 5,
    ``(B, 1, 4, 4, 512)`` (``flow_matching.py:69-79``).  So the batch axis is
    added here.  An already-batched ``(1, 1, 4, 4, 512)`` passes through, which
    keeps this adapter honest if the extractor is ever called batched.
    """

    array = np.asarray(value, dtype=np.float32)
    shape = tuple(int(dimension) for dimension in array.shape)
    if shape == FEATURE_SHAPE:
        array = array[None]
    elif shape == (1, *FEATURE_SHAPE):
        pass
    else:
        raise FmServingError(
            f"{name} feature must have shape {FEATURE_SHAPE} or "
            f"{(1, *FEATURE_SHAPE)}, got {shape}"
        )
    if not np.isfinite(array).all():
        raise FmServingError(f"{name} feature contains a non-finite value")
    return array


def _fm_state(value: Any) -> np.ndarray:
    """Pass the proprioceptive state through at a rank the FM model accepts.

    The model takes ``(B, D)`` or ``(B, 1, D)`` and squeezes the latter
    (``flow_matching.py:160-163``), and ``sample_action_chunks`` reads the batch
    size off this tensor (``flow_matching.py:248``).  The extractor's ``(1, 19)``
    is therefore already correct as ``B=1``; ``(1, 1, 19)`` is accepted too.
    Anything with a larger leading axis is refused rather than silently served,
    because only one action can come back.
    """

    array = np.asarray(value, dtype=np.float32)
    shape = tuple(int(dimension) for dimension in array.shape)
    if shape != STATE_SHAPE and shape != (1, *STATE_SHAPE):
        raise FmServingError(
            f"state must have shape {STATE_SHAPE} or {(1, *STATE_SHAPE)}, "
            f"got {shape}"
        )
    if not np.isfinite(array).all():
        raise FmServingError("state contains a non-finite value")
    return array


class FmServedPolicy:
    """Adapt canonical pixel observations to an FM policy, one action per call.

    Constructed on the server, next to the artifact:

        >>> model, params, manifest = load_flow_artifact(directory)   # doctest: +SKIP
        >>> policy = FmServedPolicy(                                  # doctest: +SKIP
        ...     model, params, extractor, policy_version=1
        ... )
        >>> action, version = policy(observation, False)              # doctest: +SKIP

    Thread safety: gRPC serves Step from several handler threads.  The PRNG key
    split and ``call_count`` are guarded by one lock, so two concurrent
    inferences can never integrate from the same noise.  The sampling itself
    runs *outside* the lock -- an ODE integration is the expensive part of the
    call and serialising it behind a mutex would add one full inference of
    latency to every concurrent Step.  Each call carries its own split key, and
    ``model.apply`` is a pure function of ``(params, observation, key)``, so
    nothing shared is mutated while it runs.

    Constructing this object touches no jax: the PRNG key is created on the
    first call.  That keeps the module importable, and the class constructible,
    in the actor venv which has no jax.

    NOTE: this class must never grow a ``prime_observation`` attribute, nor a
    delegating ``__getattr__`` -- see the module docstring.
    """

    def __init__(
        self,
        model: Any,
        params: Any,
        feature_extractor: Any,
        *,
        policy_version: int,
        rng_seed: int = 0,
        integration_steps: int | None = None,
        model_id: str = FM_MODEL_ID,
    ) -> None:
        if model is None:
            raise ValueError("model is required")
        if params is None:
            raise ValueError("params is required")
        if not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")
        if isinstance(policy_version, bool) or not isinstance(policy_version, int):
            raise ValueError("policy_version must be an integer")
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
            raise ValueError("rng_seed must be an integer")
        if rng_seed < 0:
            raise ValueError("rng_seed must be non-negative")
        if integration_steps is not None:
            if isinstance(integration_steps, bool) or not isinstance(
                integration_steps, int
            ):
                raise ValueError("integration_steps must be an integer or None")
            if integration_steps <= 0:
                raise ValueError("integration_steps must be positive")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id is required")

        self._model = model
        self._params = params
        self._extractor = feature_extractor
        self._policy_version = policy_version
        self._rng_seed = rng_seed
        #: ``None`` means "use the artifact's own ``model.config``" -- the
        #: sampler resolves that (``flow_matching.py:245``), so the served
        #: number of Euler steps stays whatever the model was documented with
        #: unless an operator deliberately overrides it.
        self._integration_steps = integration_steps
        self._rng: Any = None
        self._sample_jit: Any = None
        self._lock = threading.Lock()

        #: Advertised model identity, read by the server script and sent in
        #: ``GetServerInfo``.
        self.model_id = model_id
        #: Inferences served since construction.  A session that produced no
        #: transitions but a non-zero count means the actor discarded them.
        self.call_count = 0

    @property
    def policy_version(self) -> int:
        """Version reported with every action; constant for a served artifact.

        FM serving has no learner behind it: there is no publication path that
        could swap parameters mid-session, so this never changes and a consumer
        can treat a change as impossible rather than as a race.
        """

        return self._policy_version

    @property
    def integration_steps(self) -> int | None:
        """Euler-step override, or ``None`` to use the model's own config."""

        return self._integration_steps

    @property
    def rng_seed(self) -> int:
        """Seed of this server's noise sequence; see the module docstring."""

        return self._rng_seed

    def _next_rng(self) -> Any:
        """Split off one sampling key, creating the chain on first use."""

        import jax

        with self._lock:
            if self._rng is None:
                self._rng = jax.random.PRNGKey(self._rng_seed)
            self._rng, sample_rng = jax.random.split(self._rng)
            return sample_rng

    def _jitted_sampler(self) -> Any:
        """Build (once) and return the jitted chunk sampler.

        ``sample_action_chunks`` run eagerly pays op-by-op dispatch for every
        Euler step -- measured ~500 ms per action on the serving GPU, against
        a ~100 ms control period.  The observation shapes are fixed by the
        canonical contract, so one trace serves the whole session; the server
        startup smoke absorbs the compilation before ready is advertised.
        """

        with self._lock:
            if self._sample_jit is None:
                import jax

                from ur_env.learner.flow_matching import sample_action_chunks

                model = self._model
                steps = self._integration_steps

                def _sample(params: Any, fm_observation: Any, rng: Any) -> Any:
                    return sample_action_chunks(
                        model,
                        params,
                        fm_observation,
                        rng,
                        integration_steps=steps,
                    )

                self._sample_jit = jax.jit(_sample)
            return self._sample_jit

    def _fm_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Run the frozen trunk and reshape its output for the FM model."""

        try:
            features = self._extractor(observation)
        except Exception as exc:
            raise FmServingError(
                f"frozen-trunk feature extraction failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(features, Mapping):
            raise FmServingError(
                "feature extractor must return a mapping, got "
                f"{type(features).__name__}"
            )
        # ``FlowMatchingPolicy.__call__`` rejects any other key set
        # (``flow_matching.py:143``); checking here names the culprit.
        if set(features) != _OBSERVATION_KEYS:
            raise FmServingError(
                "feature observation keys must be exactly cam1, cam2, state; "
                f"got {sorted(features)}"
            )
        fm_observation: dict[str, Any] = {
            "state": _fm_state(features["state"])
        }
        for camera in _CAMERA_KEYS:
            fm_observation[camera] = _fm_camera_feature(
                features[camera], name=camera
            )
        return fm_observation

    def __call__(
        self, observation: Mapping[str, Any], deterministic: bool
    ) -> tuple[np.ndarray, int]:
        """Serve one action for one canonical pixel observation.

        ``deterministic`` is read off the wire and dropped -- FM sampling starts
        from Gaussian noise and has no deterministic mode to switch into; see
        the module docstring for the full reasoning.
        """

        del deterministic

        import jax

        fm_observation = self._fm_observation(observation)
        sample_rng = self._next_rng()
        try:
            chunk = self._jitted_sampler()(
                self._params, fm_observation, sample_rng
            )
            host_chunk = np.asarray(jax.device_get(chunk))
        except Exception as exc:
            raise FmServingError(
                f"flow-matching sampling failed: {type(exc).__name__}: {exc}"
            ) from exc
        action = first_action_from_chunk(host_chunk)
        with self._lock:
            self.call_count += 1
        return action, self._policy_version


__all__ = [
    "ACTION_DIM",
    "FEATURE_SHAPE",
    "FM_ARTIFACT_FORMAT",
    "FM_MODEL_ID",
    "FM_REWARD_MODEL_ID",
    "GRIPPER_INDEX",
    "STATE_SHAPE",
    "FmServedPolicy",
    "FmServingError",
    "first_action_from_chunk",
]
