"""In-process contract tests for the BC policy server's service assembly.

The BC eval server is not a new server: it is the PRODUCTION
``ur_env.actor_network.ActorSessionService`` wired with a different policy
callable, a different pair of identity strings, and a plain recording sink
instead of a learner ingress.  These tests therefore drive the real service
object directly -- no gRPC, no jax, no robot -- and pin the four properties the
production actor depends on when it talks to that assembly:

1. ``GetServerInfo`` advertises exactly what the actor's handshake pins check
   (``EXPECTED_MODEL_ID`` / ``EXPECTED_REWARD_AUTHORITY`` /
   ``EXPECTED_REWARD_MODEL_ID`` / observation schema hash), so a mismatch is
   caught before the arm moves rather than after.
2. A BeginEpisode + Step round trip is acknowledged and the finalized wrapper
   dict reaches the sink.
3. MANUAL ``MARK SUCCESS`` still works with NO classifier anywhere in the
   process: the service's DEFAULT identity finalizer promotes
   ``meta.operator_success`` to ``rewards=1, masks=0, dones=True``.  This is the
   only success path in BC eval -- there is no reward model to fall back on.
4. A sink WITHOUT ``prime_observation`` leaves the actor's raw uint8 pixels on
   the wire into the policy.  This is the load-bearing half of the dual-input
   guarantee: the production encoder runs the frozen trunk itself when handed
   pixels, so the plain recording sink is mathematically equivalent to the
   learner's feature ingress -- but only as long as nobody teaches that sink to
   prime observations.

Tests 1-4 use a local stand-in sink defined in this file, deliberately: they
assert things about the SERVICE, and must stay green independently of
``ur_env.bc_recording_sink``.  Test 5 is the only one that needs the real sink
and is guarded with ``pytest.importorskip``.

Mirrors ``tests/test_shared_feature_pipeline.py`` (same in-proc
BeginEpisodeCommand/StepCommand driving style) and reuses the production
helpers ``ur_env.actor_smoke.synthetic_observation`` and
``ur_env.remote_actor.build_data`` so the transitions are byte-shaped like the
ones the real actor sends.

Run (from /home/laptop3/gello_software)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
      -p no:cacheprovider serl_ur_infra/tests/test_bc_server_inproc.py
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ActionResult,
    ActorSessionService,
    BeginEpisodeCommand,
    ObservationPacket,
    StepCommand,
    StepResult,
)
from ur_env.actor_smoke import synthetic_observation  # noqa: E402
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)
from ur_env.remote_actor import build_data  # noqa: E402


#: The two identity strings the BC deployment pins.  Hard-coded rather than
#: imported from ``ur_env.learner.bc_init``: these tests are about the actor
#: handshake, and the whole point of a pin is that both ends state it
#: independently.  If the loader module renames them, the operator's
#: ``EXPECTED_MODEL_ID`` env override has to change too, and this file is a
#: second place that has to be updated on purpose.
BC_MODEL_ID = "bc-cube-in-cup-raw0731-bcinit-v1"
BC_REWARD_MODEL_ID = "operator-manual-success-v1"

#: The canonical policy/replay observation keys.  Anything else in the wrapper
#: dict makes the recorded pickle unloadable by ``ur_env/learner/demo.py``.
CANONICAL_OBSERVATION_KEYS = {"state", "cam1", "cam2"}

_BASE_NS = 1_700_000_000_000_000_000


def _stub_policy(observation: Any, deterministic: Any) -> tuple[np.ndarray, int]:
    """Stand in for ``VersionedPolicyRuntime`` -- numpy only, no jax."""

    del observation, deterministic
    return np.zeros(7, dtype=np.float32), 0


class _RecordingSink:
    """Minimal ``accept_data`` sink with the counters the real one exposes.

    Deliberately has NO ``prime_observation`` -- exactly like
    ``ur_env.bc_recording_sink.EpisodeRecordingSink``.  ``hasattr`` on that name
    is what ``ActorSessionService._prime_replay_observation`` branches on.
    """

    def __init__(self) -> None:
        self.items: list[tuple[dict[str, Any], bool]] = []
        self.replay_count = 0
        self.intervention_count = 0

    def __call__(self, data: dict[str, Any], intervened: bool) -> None:
        self.replay_count += 1
        self.intervention_count += int(bool(intervened))
        self.items.append((data, bool(intervened)))


def _service(
    sink: Any,
    *,
    policy: Any = None,
) -> ActorSessionService:
    """Assemble the service exactly the way ``run_bc_policy_server.py`` does."""

    return ActorSessionService(
        policy or _stub_policy,
        model_id=BC_MODEL_ID,
        reward_authority="local",
        reward_model_id=BC_REWARD_MODEL_ID,
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        accept_data=sink,
    )


class _Episode:
    """Drive one episode straight into the service, mirroring ``remote_actor``.

    Request/step/env_step bookkeeping follows the production actor loop: the
    laptop issues BeginEpisode with ``request_id=1`` and then one Step per
    control iteration, and ``request_action`` is ``not provisional_terminal``
    (``remote_actor.run_remote_actor``).
    """

    def __init__(
        self,
        service: ActorSessionService,
        *,
        actor_id: str = "bc-actor",
        run_id: str = "bc-run",
        session_id: str = "bc-session",
        episode_id: int = 0,
    ) -> None:
        self.service = service
        self.actor_id = actor_id
        self.run_id = run_id
        self.session_id = session_id
        self.episode_id = episode_id
        self.observations: list[dict[str, Any]] = [synthetic_observation(0)]
        self._index = 0
        self._request_id = 1
        self._clock = 10_000
        self.begin_result: ActionResult = service.begin_episode(
            BeginEpisodeCommand(
                PROTOCOL_VERSION,
                actor_id,
                run_id,
                session_id,
                episode_id,
                1,
                self._clock,
                ObservationPacket(
                    self.observation_id(0), _BASE_NS, self.observations[0]
                ),
                True,
            )
        )
        self._last_action: ActionResult = self.begin_result

    def observation_id(self, index: int) -> str:
        return f"{self.session_id}:obs-{index}"

    def step(
        self,
        *,
        operator_success: bool = False,
        reward: float = 0.0,
        done: bool = False,
        truncated: bool = False,
    ) -> StepResult:
        index = self._index
        next_index = index + 1
        next_observation = synthetic_observation(next_index)
        self.observations.append(next_observation)
        action = self._last_action
        data = build_data(
            actor_id=self.actor_id,
            run_id=self.run_id,
            session_id=self.session_id,
            transition_id=f"{self.run_id}:{index}",
            env_step=index,
            # O(t)'s timestamp, not O(t+1)'s -- the service checks this.
            timestamp_ns=_BASE_NS + index,
            policy_version=action.policy_version,
            policy_action=action.action,
            operator_success=operator_success,
            episode_id=self.episode_id,
            step_id=index,
            observation_id=self.observation_id(index),
            next_observation_id=self.observation_id(next_index),
            reward=reward,
            done=done,
            truncated=truncated,
            info={"intervened": 0},
        )
        provisional_terminal = bool(done) or bool(truncated)
        self._request_id += 1
        self._clock += 1
        result = self.service.step(
            StepCommand(
                PROTOCOL_VERSION,
                self.actor_id,
                self.run_id,
                self.session_id,
                self._request_id,
                self._clock,
                data,
                ObservationPacket(
                    self.observation_id(next_index),
                    _BASE_NS + next_index,
                    next_observation,
                ),
                not provisional_terminal,
                True,
            )
        )
        self._index = next_index
        if result.action is not None:
            self._last_action = result.action
        return result


# --------------------------------------------------------------------------- #
# 1. GetServerInfo advertises the pins the actor handshake checks
# --------------------------------------------------------------------------- #


def test_get_server_info_advertises_the_actor_handshake_pins() -> None:
    service = _service(_RecordingSink())

    info = service.get_server_info()

    assert info.ready is True
    assert info.model_id == BC_MODEL_ID
    assert info.reward_authority == "local"
    assert info.reward_model_id == BC_REWARD_MODEL_ID
    assert info.observation_schema_hash == CANONICAL_OBSERVATION_SCHEMA_HASH
    assert info.action_dim == 7
    # Equality with the shared constants, not with literals: the actor imports
    # the same names, so a version bump has to move both ends together.
    assert info.protocol_version == PROTOCOL_VERSION
    assert info.schema_version == SCHEMA_VERSION


def test_a_bare_bc_service_is_health_ready_with_no_classifier() -> None:
    """No reward runtime means nothing can degrade, so ``detail`` stays plain."""

    service = _service(_RecordingSink())

    assert service.health() == (True, True, "ready")


# --------------------------------------------------------------------------- #
# 2. BeginEpisode + Step round trip
# --------------------------------------------------------------------------- #


def test_begin_episode_and_step_round_trip_reaches_the_sink() -> None:
    sink = _RecordingSink()
    service = _service(sink)

    episode = _Episode(service)

    assert episode.begin_result.action.shape == (7,)
    assert np.array_equal(episode.begin_result.action, np.zeros(7, np.float32))
    assert episode.begin_result.policy_version == 0
    assert episode.begin_result.observation_id == episode.observation_id(0)
    assert sink.replay_count == 0, "BeginEpisode carries no transition"

    result = episode.step()

    assert result.ack.accepted is True
    assert result.ack.deduplicated is False
    assert result.ack.transition_id == "bc-run:0"
    assert result.ack.session_id == "bc-session"
    assert result.action is not None, "a non-terminal Step must return an action"
    assert np.array_equal(result.action.action, np.zeros(7, np.float32))
    assert service.inference_count == 2  # BeginEpisode + this Step

    assert sink.replay_count == 1
    assert sink.intervention_count == 0
    data, intervened = sink.items[-1]
    assert intervened is False
    assert set(data) == {"meta", "transition"}

    meta = data["meta"]
    assert meta["transition_id"] == "bc-run:0"
    assert meta["run_id"] == "bc-run"
    assert meta["actor_id"] == "bc-actor"
    assert meta["session_id"] == "bc-session"
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["env_step"] == 0
    assert meta["operator_success"] is False
    assert meta["auto_success"] is False

    transition = data["transition"]
    assert transition["episode_id"] == 0
    assert transition["step_id"] == 0
    assert np.array_equal(transition["actions"], np.zeros(7, np.float32))
    assert float(transition["rewards"]) == 0.0
    assert float(transition["masks"]) == 1.0
    assert bool(transition["dones"]) is False
    assert bool(transition["truncated"]) is False
    assert bool(transition["success"]) is False
    assert bool(transition["classifier_evaluated"]) is False
    assert transition["reward_model_id"] == ""
    # The server, not the actor, attaches the observation pair to the wrapper.
    assert set(transition["observations"]) == CANONICAL_OBSERVATION_KEYS
    assert set(transition["next_observations"]) == CANONICAL_OBSERVATION_KEYS
    assert np.array_equal(
        transition["observations"]["cam1"], episode.observations[0]["cam1"]
    )
    assert np.array_equal(
        transition["next_observations"]["cam1"], episode.observations[1]["cam1"]
    )

    outcome = result.outcome
    assert outcome.transition_id == "bc-run:0"
    assert outcome.terminal is False
    assert outcome.classifier_evaluated is False
    assert outcome.reward_model_id == ""


# --------------------------------------------------------------------------- #
# 3. MANUAL MARK SUCCESS with no classifier anywhere
# --------------------------------------------------------------------------- #


def test_operator_success_is_promoted_without_any_classifier() -> None:
    """The default identity finalizer is the whole reward authority here.

    ``ActorSessionService`` was constructed with no ``finalize_transition``, so
    ``_finalize_transition_identity`` runs.  It is what turns the GUI's MANUAL
    ``MARK SUCCESS`` token into a Bellman terminal.  Note the actor sends this
    with local ``done=False, truncated=False`` (the env did not terminate; the
    operator did), hence ``request_action=True`` -- and the service still
    answers with ``action=None`` because the FINALIZED outcome is terminal.
    """

    sink = _RecordingSink()
    service = _service(sink)
    episode = _Episode(service)
    episode.step()

    result = episode.step(operator_success=True)

    outcome = result.outcome
    assert outcome.success is True
    assert outcome.reward == 1.0
    assert outcome.done is True
    assert outcome.truncated is False
    assert outcome.mask == 0.0
    assert outcome.terminal is True
    assert outcome.classifier_evaluated is False
    assert outcome.classifier_probability == 0.0
    assert outcome.classifier_threshold == 0.0
    assert outcome.reward_model_id == ""
    assert result.ack.accepted is True
    assert result.action is None, "a finalized terminal must not return an action"

    assert sink.replay_count == 2
    data, _ = sink.items[-1]
    transition = data["transition"]
    assert float(transition["rewards"]) == 1.0
    assert bool(transition["dones"]) is True
    assert bool(transition["success"]) is True
    assert float(transition["masks"]) == 0.0
    assert bool(transition["truncated"]) is False
    assert bool(transition["classifier_evaluated"]) is False
    assert data["meta"]["operator_success"] is True


def test_without_operator_success_nothing_can_declare_success() -> None:
    """The complement of the test above: no classifier means no auto reward."""

    sink = _RecordingSink()
    service = _service(sink)
    episode = _Episode(service)

    result = episode.step(reward=1.0)

    assert result.outcome.success is False
    assert result.outcome.done is False
    assert result.outcome.terminal is False
    data, _ = sink.items[-1]
    assert bool(data["transition"]["success"]) is False
    # A locally reported reward survives untouched; it just is not a success.
    assert float(data["transition"]["rewards"]) == 1.0


# --------------------------------------------------------------------------- #
# 4. No prime_observation => raw pixels reach the policy
# --------------------------------------------------------------------------- #


def test_a_sink_without_prime_observation_serves_raw_pixels_to_the_policy() -> None:
    """The load-bearing dual-input guarantee for the plain recording sink.

    ``ActorSessionService._prime_replay_observation`` returns ``None`` when the
    sink has no ``prime_observation``, and both ``begin_episode`` and ``step``
    then fall back to ``observation.observation`` -- the actor's own uint8
    pixels.  The production encoder runs the frozen trunk on those itself, so
    the math matches the learner path.  A float32 feature tensor here would mean
    something started priming and the policy is being fed a DIFFERENT input than
    the one it was assembled to consume.
    """

    seen: list[dict[str, Any]] = []

    def recording_policy(observation, deterministic):
        seen.append(observation)
        return _stub_policy(observation, deterministic)

    sink = _RecordingSink()
    assert not hasattr(sink, "prime_observation")

    service = _service(sink, policy=recording_policy)
    episode = _Episode(service)
    episode.step()

    assert len(seen) == 2, "BeginEpisode inference + one non-terminal Step"
    for observation in seen:
        assert set(observation) == CANONICAL_OBSERVATION_KEYS
        for camera in ("cam1", "cam2"):
            tensor = np.asarray(observation[camera])
            assert tensor.dtype == np.uint8, (
                f"{camera} reached the policy as {tensor.dtype}; a float tensor "
                "means the sink primed the observation into trunk features"
            )
            assert tensor.shape == (1, 128, 128, 3)
        state = np.asarray(observation["state"])
        assert state.dtype == np.float32
        assert state.shape == (1, 19)

    # Values, not just shapes: these are the exact frames the actor uploaded.
    assert np.array_equal(seen[0]["cam1"], episode.observations[0]["cam1"])
    assert np.array_equal(seen[0]["cam2"], episode.observations[0]["cam2"])
    assert np.array_equal(seen[1]["cam1"], episode.observations[1]["cam1"])
    assert np.array_equal(seen[1]["cam2"], episode.observations[1]["cam2"])


# --------------------------------------------------------------------------- #
# 5. Real sink integration (skipped until ur_env/bc_recording_sink.py lands)
# --------------------------------------------------------------------------- #


def _real_sink(module: Any, root: Any) -> Any:
    """Build ``EpisodeRecordingSink(root, ...)`` tolerating optional kwargs."""

    factory = getattr(module, "EpisodeRecordingSink", None)
    if factory is None:
        pytest.skip("ur_env.bc_recording_sink has no EpisodeRecordingSink yet")
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        parameters = {}
    optional: dict[str, Any] = {}
    if "artifact_sha256" in parameters:
        optional["artifact_sha256"] = "0" * 64
    if "model_id" in parameters:
        optional["model_id"] = BC_MODEL_ID
    return factory(os.fspath(root), **optional)


def test_the_real_recording_sink_writes_an_episode_pickle(tmp_path) -> None:
    module = pytest.importorskip(
        "ur_env.bc_recording_sink",
        reason="the BC recording sink module has not landed yet",
    )
    root = tmp_path / "bc_eval_run"
    root.mkdir()

    sink = _real_sink(module, root)

    # Construct-time metadata: the record root is stamped with the policy
    # identity BEFORE any transition arrives, so an interrupted run is still
    # attributable.
    metadata = root / "metadata.json"
    assert metadata.is_file(), "the sink must stamp metadata.json at construction"

    # The real sink must not prime either -- that is what keeps pixels flowing.
    assert not hasattr(sink, "prime_observation")

    service = _service(sink)
    episode = _Episode(service)
    episode.step()
    result = episode.step(operator_success=True)
    assert result.outcome.terminal is True

    pickles = sorted(root.glob("**/*.pkl"))
    assert pickles, f"no episode pickle under {root}; found {sorted(root.rglob('*'))}"
    assert getattr(sink, "replay_count", 2) == 2
