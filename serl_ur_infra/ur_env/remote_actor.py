"""Robot-laptop loop for one-RPC-per-step remote HIL-SERL execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
import pickle
import sys
import time
from typing import Any, Callable, Mapping, Optional
import uuid

import numpy as np

from ur_env.actor_network import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ActorNetwork,
    ActorProtocolError,
    ServerInfo,
    validate_action,
    validate_counter,
    validate_timestamp_ns,
)
from ur_env.classifier_sidecar import CLASSIFIER_SIDECAR_KEY, build_sidecar
from ur_env.operator_session import (
    ActorStatusTracker,
    FAULT,
    HOLD,
    HOMING,
    HUMAN_INTERVENTION,
    OWNER_HOLD,
    OWNER_HUMAN,
    OWNER_NONE,
    OWNER_POLICY,
    POLICY_RUNNING,
    STOPPED,
    WAIT_HOME_APPROVAL,
    WAIT_SCENE_READY,
)


class EnvTimestampAdapter:
    """Attach wall-clock time at the outer environment observation boundary.

    An underlying environment-provided ``info['timestamp_ns']`` wins.  The
    fallback is captured immediately after ``reset``/``step`` returns, keeping
    timestamp generation in the environment adapter rather than in policy
    state or the policy observation vector.
    """

    def __init__(
        self, env: Any, *, wall_time_ns: Callable[[], int] = time.time_ns
    ) -> None:
        self.env = env
        self._wall_time_ns = wall_time_ns

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return observation, self._stamp(info)

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        observation, reward, done, truncated, info = self.env.step(action)
        return observation, reward, done, truncated, self._stamp(info)

    def close(self) -> None:
        self.env.close()

    def _stamp(self, info: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(info)
        timestamp_ns = result.get("timestamp_ns", self._wall_time_ns())
        result["timestamp_ns"] = np.int64(validate_timestamp_ns(timestamp_ns))
        return result


def build_data(
    *,
    actor_id: str,
    run_id: str,
    session_id: str,
    transition_id: str,
    env_step: int,
    timestamp_ns: int,
    policy_version: int,
    policy_action: Any,
    policy_actions_synthetic: bool = False,
    auto_success: bool = False,
    operator_success: bool = False,
    episode_id: int,
    step_id: int,
    observation_id: str,
    next_observation_id: str,
    reward: Any,
    done: bool,
    truncated: bool,
    info: Mapping[str, Any],
    action_shape: tuple[int, ...] = (7,),
) -> dict[str, Any]:
    """Build ``data{meta, transition}`` with action provenance intact.

    ``meta.policy_action`` is the server action before a human override.
    ``transition.actions`` is the action that physically produced the next
    observation.  Every data item goes to replay server-side; items labelled
    ``intervened == 1`` additionally go to the intervention buffer.
    """
    for name, value in (
        ("actor_id", actor_id),
        ("run_id", run_id),
        ("session_id", session_id),
        ("transition_id", transition_id),
        ("observation_id", observation_id),
        ("next_observation_id", next_observation_id),
    ):
        if not isinstance(value, str) or not value:
            raise ActorProtocolError(f"{name} is required")

    requested_action = validate_action(
        policy_action, action_shape=action_shape, name="policy_action"
    )
    has_intervention_action = "intervene_action" in info
    intervened_value = info.get("intervened", int(has_intervention_action))
    if isinstance(intervened_value, np.generic):
        intervened_value = intervened_value.item()
    if intervened_value not in (0, 1, False, True):
        raise ActorProtocolError("intervened must be 0 or 1")
    intervened = int(bool(intervened_value))
    if bool(intervened) != has_intervention_action:
        raise ActorProtocolError(
            "intervened label and intervene_action presence are inconsistent"
        )
    executed_action = validate_action(
        info["intervene_action"] if intervened else requested_action,
        action_shape=action_shape,
        name="executed_action",
    )

    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise ActorProtocolError("reward must be finite")
    if not isinstance(done, (bool, np.bool_)) or not isinstance(
        truncated, (bool, np.bool_)
    ):
        raise ActorProtocolError("done and truncated must be bool")
    if bool(done) and bool(truncated):
        raise ActorProtocolError("a transition cannot be done and truncated")
    if not isinstance(auto_success, (bool, np.bool_)):
        raise ActorProtocolError("auto_success must be bool")
    if not isinstance(operator_success, (bool, np.bool_)):
        raise ActorProtocolError("operator_success must be bool")
    auto_success = bool(auto_success)
    operator_success = bool(operator_success)
    if auto_success and operator_success:
        raise ActorProtocolError(
            "operator_success is only valid in MANUAL success mode"
        )

    transition: dict[str, Any] = {
        "episode_id": validate_counter(episode_id, name="episode_id"),
        "step_id": validate_counter(step_id, name="step_id"),
        "observation_id": observation_id,
        "actions": executed_action,
        "next_observation_id": next_observation_id,
        "rewards": reward_value,
        # Preserve upstream HIL-SERL semantics: truncation resets the episode
        # but is not a Bellman terminal.
        "masks": 0.0 if bool(done) else 1.0,
        "dones": bool(done),
        "truncated": bool(truncated),
    }
    if "grasp_penalty" in info:
        grasp_penalty = float(info["grasp_penalty"])
        if not np.isfinite(grasp_penalty):
            raise ActorProtocolError("grasp_penalty must be finite")
        transition["grasp_penalty"] = grasp_penalty

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "actor_id": actor_id,
            "session_id": session_id,
            "transition_id": transition_id,
            "env_step": validate_counter(env_step, name="env_step"),
            # This is O(t)'s environment timestamp, not O(t+1)'s timestamp.
            "timestamp_ns": validate_timestamp_ns(timestamp_ns),
            "policy_version": validate_counter(
                policy_version, name="policy_version"
            ),
            "policy_action": requested_action,
            "intervened": intervened,
            # MANUAL is the production default.  The server remains the sole
            # reward authority and combines these protected inputs with its
            # own classifier result when it finalizes the transition.
            "auto_success": auto_success,
            "operator_success": operator_success,
            # Stamped per transition, not just in the run summary: once these
            # pickles leave the process there is nothing else to distinguish a
            # mock-noise action from a real policy action.
            "policy_actions_synthetic": bool(policy_actions_synthetic),
        },
        "transition": transition,
    }


@dataclass(frozen=True)
class ActorRunSummary:
    run_id: str
    env_steps: int
    episodes_started: int
    intervention_steps: int
    # True when policy actions were rewritten before execution. A run with a
    # mock policy must stay identifiable AFTER the fact: the stored actions are
    # indistinguishable from real policy output once the process exits.
    policy_actions_synthetic: bool = False
    # Classifier-sidecar accounting.  Attachment makes one Step RPC carry two
    # extra encoded frames, and the control loop only has a 100 ms budget, so
    # the cost has to be measurable in the field rather than estimated: the two
    # round-trip series are kept apart on purpose.
    sidecar_attached_steps: int = 0
    sidecar_build_failures: int = 0
    sidecar_round_trip_ms_mean: float = 0.0
    sidecar_round_trip_ms_max: float = 0.0
    plain_round_trip_ms_mean: float = 0.0
    plain_round_trip_ms_max: float = 0.0


@dataclass(frozen=True)
class ActorProbeSummary:
    """Evidence returned by the no-submit real-observation probe.

    A probe intentionally stops after ``BeginEpisode``.  It proves that the
    actor can read one environment observation, pin the server identity/schema,
    and receive a valid policy action without executing that action or creating
    a transition.
    """

    run_id: str
    session_id: str
    observation_id: str
    source_timestamp_ns: int
    server_info: ServerInfo
    policy_version: int
    policy_action: np.ndarray
    server_inference_ms: float
    round_trip_ms: float


def run_remote_actor_probe(
    network: ActorNetwork,
    env: Any,
    *,
    actor_id: str,
    run_id: Optional[str] = None,
    session_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    deterministic: bool = False,
) -> ActorProbeSummary:
    """Validate one real observation and one policy inference, then stop.

    The only actor RPCs issued here are ``GetServerInfo`` and
    ``BeginEpisode``.  In particular, this function never calls ``env.step``
    or ``ActorNetwork.step``; therefore it cannot construct, submit, or ACK a
    replay transition.  This is the safe default used by
    ``run_remote_rlpd_actor.py`` when ``--arm`` is absent.

    ``BeginEpisode`` is still valuable: the production server validates the
    canonical observation before policy inference, and the returned action is
    checked again here for the exact robot action shape and normalized range.
    """

    if not actor_id:
        raise ValueError("actor_id is required")
    action_shape = tuple(int(dim) for dim in env.action_space.shape)
    if action_shape != (7,):
        raise ValueError(
            f"protocol v2 requires action shape (7,), got {action_shape}"
        )

    # GrpcActorNetwork verifies any expected model/reward/schema pins while
    # fetching this object.  Repeat the transport-neutral invariants here so a
    # future ActorNetwork implementation cannot silently weaken probe mode.
    info = network.get_server_info()
    if not info.ready:
        raise ActorProtocolError("remote actor service is not ready")
    if info.protocol_version != PROTOCOL_VERSION:
        raise ActorProtocolError(
            "incompatible server protocol_version "
            f"{info.protocol_version!r}; expected {PROTOCOL_VERSION!r}"
        )
    if info.schema_version != SCHEMA_VERSION:
        raise ActorProtocolError(
            f"server schema_version is {info.schema_version}, "
            f"expected {SCHEMA_VERSION}"
        )
    expected_action_dim = int(np.prod(action_shape))
    if info.action_dim != expected_action_dim:
        raise ActorProtocolError(
            f"server action_dim is {info.action_dim}, "
            f"expected {expected_action_dim}"
        )
    for name, value in (
        ("model_id", info.model_id),
        ("reward_authority", info.reward_authority),
        ("observation_schema_hash", info.observation_schema_hash),
    ):
        if not isinstance(value, str) or not value:
            raise ActorProtocolError(f"server {name} is empty")

    observation, reset_info = env.reset()
    source_timestamp_ns = validate_timestamp_ns(reset_info.get("timestamp_ns"))
    run_id = run_id or f"probe-{uuid.uuid4().hex}"
    session_id = session_id_factory()
    if not session_id:
        raise ValueError("session_id_factory returned an empty ID")
    observation_id = f"{session_id}:0"

    action_result = network.begin_episode(
        observation,
        run_id=run_id,
        session_id=session_id,
        episode_id=0,
        observation_id=observation_id,
        timestamp_ns=source_timestamp_ns,
        deterministic=deterministic,
    )
    policy_action = validate_action(
        action_result.action,
        action_shape=action_shape,
        name="probe policy action",
    )
    policy_version = validate_counter(
        action_result.policy_version, name="policy_version"
    )
    if action_result.session_id != session_id:
        raise ActorProtocolError("probe action session_id does not match request")
    if action_result.request_id != 1:
        raise ActorProtocolError("probe BeginEpisode reply request_id must be 1")
    if action_result.observation_id != observation_id:
        raise ActorProtocolError(
            "probe action observation_id does not match request"
        )
    validate_timestamp_ns(
        action_result.request_created_monotonic_ns,
        name="request_created_monotonic_ns",
    )
    timings: dict[str, float] = {}
    for name, value in (
        ("server_inference_ms", action_result.server_inference_ms),
        ("round_trip_ms", action_result.round_trip_ms),
    ):
        try:
            timing = float(value)
        except (TypeError, ValueError) as exc:
            raise ActorProtocolError(f"probe {name} must be numeric") from exc
        if not math.isfinite(timing) or timing < 0.0:
            raise ActorProtocolError(
                f"probe {name} must be finite and non-negative"
            )
        timings[name] = timing

    return ActorProbeSummary(
        run_id=run_id,
        session_id=session_id,
        observation_id=observation_id,
        source_timestamp_ns=source_timestamp_ns,
        server_info=info,
        policy_version=policy_version,
        policy_action=policy_action,
        server_inference_ms=timings["server_inference_ms"],
        round_trip_ms=timings["round_trip_ms"],
    )


class _RoundTripStats:
    """Streaming count/mean/max of Step RPC latency.

    Deliberately O(1) in memory: ``config.max_steps`` defaults to 1e6, and
    keeping every sample just to compute a percentile would cost more RAM than
    the actor's whole working set.  Mean and max are enough to answer the
    question this exists for -- "does attaching the sidecar blow the 100 ms
    step budget?".
    """

    __slots__ = ("count", "total_ms", "max_ms")

    def __init__(self) -> None:
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0

    def add(self, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        if elapsed_ms > self.max_ms:
            self.max_ms = elapsed_ms

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


def resolve_camera_frame_source(
    env: Any,
) -> Optional[Callable[[], Mapping[str, np.ndarray]]]:
    """Find ``last_camera_frames`` on the environment, or return ``None``.

    Walks ``unwrapped`` first and then the ``.env`` chain because the actor is
    handed a stack of wrappers (EnvTimestampAdapter -> RecordEpisodeStatistics
    -> GripperPenalty -> Chunking -> SERLObs -> ...), and gymnasium 1.x no
    longer forwards arbitrary attributes through ``Wrapper.__getattr__``.

    Returning ``None`` rather than raising is intentional: a fake env and the
    test stubs have no camera, and the sidecar is an optional enrichment of the
    observation, never a precondition for driving the robot.
    """

    accessor = getattr(getattr(env, "unwrapped", None), "last_camera_frames", None)
    if callable(accessor):
        return accessor
    # Fall back to walking the chain by hand: EnvTimestampAdapter is not a
    # gymnasium Wrapper, so ``unwrapped`` is not guaranteed to be defined all
    # the way down in every composition.
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        accessor = getattr(current, "last_camera_frames", None)
        if callable(accessor):
            return accessor
        current = getattr(current, "env", None)
    return None


def _dump_data(
    checkpoint_path: str,
    run_id: str,
    env_step: int,
    replay_data: list[dict[str, Any]],
    intervention_data: list[dict[str, Any]],
) -> None:
    replay_dir = os.path.join(checkpoint_path, "actor_data", run_id, "replay")
    intervention_dir = os.path.join(
        checkpoint_path, "actor_data", run_id, "intervention"
    )
    os.makedirs(replay_dir, exist_ok=True)
    os.makedirs(intervention_dir, exist_ok=True)
    with open(os.path.join(replay_dir, f"data_{env_step}.pkl"), "wb") as file:
        pickle.dump(replay_data, file)
    with open(
        os.path.join(intervention_dir, f"data_{env_step}.pkl"), "wb"
    ) as file:
        pickle.dump(intervention_data, file)


class _OperatorReporter:
    """Best-effort status reporting around a mandatory scene-ready wait."""

    def __init__(self, session: Optional[Any], run_id: str) -> None:
        self.session = session
        self.tracker = ActorStatusTracker(run_id)
        self.episode_id = 0
        self.episode_step = 0
        self.env_step = -1
        self.success = False
        self.terminal_reason = ""

    def position(self, *, episode_id: int, episode_step: int, env_step: int) -> None:
        self.episode_id = int(episode_id)
        self.episode_step = int(episode_step)
        self.env_step = int(env_step)

    def status(
        self,
        state: str,
        control_owner: str,
        *,
        message: str,
        outcome: Optional[Any] = None,
        success: Optional[bool] = None,
        terminal_reason: Optional[str] = None,
    ) -> Any:
        if success is not None:
            self.success = bool(success)
        if terminal_reason is not None:
            self.terminal_reason = str(terminal_reason)
        evaluated = bool(
            outcome is not None and outcome.classifier_evaluated
        )
        return self.tracker.status(
            state=state,
            control_owner=control_owner,
            episode_id=self.episode_id,
            episode_step=self.episode_step,
            env_step=self.env_step,
            classifier_evaluated=evaluated,
            classifier_probability=(
                outcome.classifier_probability if evaluated else None
            ),
            classifier_threshold=(
                outcome.classifier_threshold if evaluated else None
            ),
            success=self.success,
            terminal_reason=self.terminal_reason,
            message=message,
        )

    def publish(
        self,
        state: str,
        control_owner: str,
        *,
        message: str,
        outcome: Optional[Any] = None,
        success: Optional[bool] = None,
        terminal_reason: Optional[str] = None,
    ) -> None:
        if self.session is None:
            return
        try:
            status = self.status(
                state,
                control_owner,
                message=message,
                outcome=outcome,
                success=success,
                terminal_reason=terminal_reason,
            )
            self.session.publish(status)
        except Exception as exc:  # status cannot stop the robot loop
            print(
                f"[remote-actor] WARNING: actor status publish failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def wait_for_scene_ready(self, *, message: str) -> None:
        if self.session is None:
            return
        # Unlike status publication, failure of this call is fatal: an armed
        # topic actor may not silently bypass its operator gate.
        status = self.status(
            WAIT_SCENE_READY,
            OWNER_NONE,
            message=message,
        )
        self.session.wait_for_scene_ready(status)

    def wait_for_home_approval(self, *, message: str) -> None:
        if self.session is None:
            return
        status = self.status(
            WAIT_HOME_APPROVAL,
            OWNER_HOLD,
            message=message,
        )
        self.session.wait_for_home_approval(status)


def _control_state(info: Mapping[str, Any], intervened: bool) -> tuple[str, str]:
    """Map the action that physically ran to status state/owner."""

    if bool(info.get("held", False)):
        return HOLD, OWNER_HOLD
    if intervened:
        return HUMAN_INTERVENTION, OWNER_HUMAN
    return POLICY_RUNNING, OWNER_POLICY


def _terminal_reason(outcome: Any) -> str:
    if bool(outcome.success):
        return "SUCCESS"
    if bool(outcome.truncated):
        return "TRUNCATED"
    return "EPISODE_LIMIT"


def run_remote_actor(
    network: ActorNetwork,
    env: Any,
    *,
    config: Any,
    actor_id: str,
    checkpoint_path: Optional[str] = None,
    run_id: Optional[str] = None,
    session_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    policy_action_transform: Optional[Callable[[Any], Any]] = None,
    sidecar_scheduler: Optional[Any] = None,
    operator_session: Optional[Any] = None,
) -> ActorRunSummary:
    """Run the actor, reporting FAULT/STOPPED when an operator session exists."""

    resolved_run_id = run_id or uuid.uuid4().hex
    reporter = _OperatorReporter(operator_session, resolved_run_id)
    try:
        return _run_remote_actor_impl(
            network,
            env,
            config=config,
            actor_id=actor_id,
            checkpoint_path=checkpoint_path,
            run_id=resolved_run_id,
            session_id_factory=session_id_factory,
            policy_action_transform=policy_action_transform,
            sidecar_scheduler=sidecar_scheduler,
            operator_session=operator_session,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        reporter.publish(
            STOPPED,
            OWNER_NONE,
            message="actor interrupted by operator",
        )
        raise
    except BaseException as exc:
        reporter.publish(
            FAULT,
            OWNER_NONE,
            message=f"{type(exc).__name__}: {exc}",
            success=False,
            terminal_reason="FAULT",
        )
        raise


def _run_remote_actor_impl(
    network: ActorNetwork,
    env: Any,
    *,
    config: Any,
    actor_id: str,
    checkpoint_path: Optional[str],
    run_id: str,
    session_id_factory: Callable[[], str],
    policy_action_transform: Optional[Callable[[Any], Any]],
    sidecar_scheduler: Optional[Any],
    operator_session: Optional[Any],
    reporter: _OperatorReporter,
) -> ActorRunSummary:
    """Run synchronous remote inference and lossless transition delivery.

    ``policy_action_transform`` rewrites the server's action before it is used.
    It is applied at the single point where the action enters the loop, so the
    executed action and the stored action stay the same object -- the buffer
    invariant that makes stored transitions trainable.  Its only intended use
    is bring-up against a zero-action server, where the robot would otherwise
    never move and the policy->robot->intervention path could not be exercised.

    ``sidecar_scheduler`` is a ``classifier_sidecar.SidecarScheduler`` (or any
    object with the same three methods).  When it says yes, the outgoing
    next-observation carries an extra ``classifier`` entry built from the
    cameras' UNCROPPED frames, so the reward classifier scores the distribution
    it was trained on instead of the policy's measured crop.  ``None`` disables
    attachment entirely and the loop behaves exactly as before.
    """
    max_steps = validate_counter(config.max_steps, name="max_steps")
    if max_steps <= 0:
        raise ValueError("config.max_steps must be positive")
    if int(config.random_steps) != 0:
        raise ValueError(
            "remote actor protocol v2 requires config.random_steps == 0"
        )
    buffer_period = int(getattr(config, "buffer_period", 0))
    if buffer_period < 0:
        raise ValueError("config.buffer_period must be non-negative")
    if buffer_period and not checkpoint_path:
        raise ValueError("checkpoint_path is required when buffer_period > 0")

    if not actor_id or not run_id:
        raise ValueError("actor_id and run_id are required")
    action_shape = tuple(int(dim) for dim in env.action_space.shape)
    if action_shape != (7,):
        raise ValueError(f"protocol v2 requires action shape (7,), got {action_shape}")

    frame_source: Optional[Callable[[], Mapping[str, np.ndarray]]] = None
    if sidecar_scheduler is not None:
        frame_source = resolve_camera_frame_source(env)
        if frame_source is None:
            # Loud, once, at start-up rather than a silent no-op for the whole
            # run: an operator who asked for classifier rewards must not
            # discover at analysis time that none were ever produced.
            print(
                "[remote-actor] classifier sidecar requested, but this "
                "environment exposes no last_camera_frames() -- no sidecar "
                "will be attached and the server cannot score reward.",
                flush=True,
            )

    replay_data: list[dict[str, Any]] = []
    intervention_data: list[dict[str, Any]] = []
    total_intervention_steps = 0
    episodes_started = 0
    episode_id = 0
    step_id = 0
    sidecar_attached_steps = 0
    sidecar_build_failures = 0
    sidecar_latency = _RoundTripStats()
    plain_latency = _RoundTripStats()

    # A real armed topic actor does not obtain or execute a policy action just
    # because the launcher handed off the controller.  First HOME, then wait for
    # a fresh-DISENGAGED /hil/scene_ready edge.  The reset after that edge is
    # intentional: it gives BeginEpisode camera/state data captured after the
    # operator changed the scene, rather than the pre-wait HOME observation.
    reporter.position(episode_id=0, episode_step=0, env_step=-1)
    if operator_session is not None:
        reporter.publish(
            HOMING,
            OWNER_NONE,
            message="moving to HOME before the first policy episode",
            success=False,
            terminal_reason="",
        )
        env.reset()  # HOME result is deliberately discarded (pre-scene-reset)
        reporter.wait_for_scene_ready(
            message="HOME; reset the scene, DISENGAGE, then press Start/Resume"
        )
        reporter.publish(
            HOMING,
            OWNER_NONE,
            message="scene ready accepted; refreshing HOME observation",
            success=False,
            terminal_reason="",
        )
    observation, reset_info = env.reset()
    if sidecar_scheduler is not None:
        sidecar_scheduler.reset()
    source_timestamp_ns = validate_timestamp_ns(reset_info.get("timestamp_ns"))
    session_id = session_id_factory()
    if not session_id:
        raise ValueError("session_id_factory returned an empty ID")
    observation_id = f"{session_id}:0"
    # NOTE: BeginEpisode never carries a sidecar.  O0 is the observation the
    # first action is computed from; it is no transition's next_observations,
    # so a classifier verdict on it could not be attached to any reward.
    action_result = network.begin_episode(
        observation,
        run_id=run_id,
        session_id=session_id,
        episode_id=episode_id,
        observation_id=observation_id,
        timestamp_ns=source_timestamp_ns,
    )
    episodes_started += 1
    reporter.publish(
        POLICY_RUNNING,
        OWNER_POLICY,
        message="policy episode running",
        success=False,
        terminal_reason="",
    )

    for env_step in range(max_steps):
        policy_action = validate_action(
            action_result.action,
            action_shape=action_shape,
            name="network policy action",
        )
        if policy_action_transform is not None:
            # Re-validate: the transform is caller-supplied, and an out-of-range
            # or wrong-dtype action must fail here rather than reach the robot.
            policy_action = validate_action(
                policy_action_transform(policy_action),
                action_shape=action_shape,
                name="transformed policy action",
            )
        next_observation, reward, done, truncated, info = env.step(policy_action)
        next_timestamp_ns = validate_timestamp_ns(info.get("timestamp_ns"))
        next_observation_id = f"{session_id}:{step_id + 1}"
        auto_success = bool(
            getattr(operator_session, "auto_success", False)
            if operator_session is not None
            else False
        )
        operator_success = False
        if operator_session is not None and not auto_success:
            consume_operator_success = getattr(
                operator_session, "consume_operator_success", None
            )
            if callable(consume_operator_success):
                operator_success = bool(
                    consume_operator_success(run_id, episode_id)
                )
        data = build_data(
            actor_id=actor_id,
            run_id=run_id,
            session_id=session_id,
            transition_id=f"{run_id}:{env_step}",
            env_step=env_step,
            # O(t), deliberately not the just-returned O(t+1) time.
            timestamp_ns=source_timestamp_ns,
            policy_version=action_result.policy_version,
            policy_action=policy_action,
            policy_actions_synthetic=policy_action_transform is not None,
            auto_success=auto_success,
            operator_success=operator_success,
            episode_id=episode_id,
            step_id=step_id,
            observation_id=observation_id,
            next_observation_id=next_observation_id,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
            action_shape=action_shape,
        )
        provisional_terminal = bool(done) or bool(truncated)

        # ---- classifier sidecar ---------------------------------------- #
        # The classifier deliberately does NOT run on every step.  Sparse
        # evaluation is the intended behaviour, not a bandwidth compromise:
        # scoring at ~2 Hz instead of 10 Hz stops the success verdict from
        # flickering frame to frame, and it gives the scene a moment to settle
        # after the cube is released before we ask "did that succeed?".  The
        # scheduler additionally gates on the arm being stationary, so the
        # frame we ship is not a motion-blurred one.
        attached = False
        outgoing_observation = next_observation
        if frame_source is not None and sidecar_scheduler.should_attach(
            next_observation.get("state"), provisional_terminal
        ):
            try:
                # build_sidecar takes the UNCROPPED full-resolution BGR frames
                # and does the 128x128 resize + re-encode itself.  The resize
                # has to happen here on the laptop: the camera's own JPEG is
                # ~200 KiB per frame at the nodes' quality=95, and 400 KiB per
                # attachment does not fit a 13 Mbit/s link inside a 100 ms step.
                sidecar = build_sidecar(frame_source())
            except Exception as exc:  # noqa: BLE001 - any build failure is equivalent
                # Fail SOFT.  A malformed or missing frame costs this step its
                # reward (the server reports classifier_evaluated=False), which
                # is strictly better than aborting an episode mid-motion with
                # the arm under power.
                sidecar_build_failures += 1
                if sidecar_build_failures == 1 or sidecar_build_failures % 50 == 0:
                    print(
                        "[remote-actor] classifier sidecar not built "
                        f"({sidecar_build_failures}x): "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            else:
                # Shallow copy, and ONLY for the wire.  ``next_observation``
                # itself must stay canonical: it is deep-copied into the local
                # backup pickle below, and ur_env/learner/demo.py validates
                # those observations strictly -- an extra key makes the whole
                # dump unloadable as demo data.  It is also the next step's
                # ``observations``, so a mutation here would leak forward.
                outgoing_observation = dict(next_observation)
                outgoing_observation[CLASSIFIER_SIDECAR_KEY] = sidecar
                attached = True

        rpc_started = time.perf_counter()
        result = network.step(
            outgoing_observation,
            next_observation_id=next_observation_id,
            next_timestamp_ns=next_timestamp_ns,
            data=data,
            request_action=not provisional_terminal,
        )
        rpc_ms = (time.perf_counter() - rpc_started) * 1000.0
        if attached:
            sidecar_attached_steps += 1
            sidecar_latency.add(rpc_ms)
        else:
            plain_latency.add(rpc_ms)
        # Reaching here proves the server ACKed this transition. In particular,
        # terminal reset can never happen after an unacknowledged Step.
        outcome = result.outcome
        if outcome.transition_id != data["meta"]["transition_id"]:
            raise ActorProtocolError(
                "transition outcome ID does not match the accepted data"
            )
        if sidecar_scheduler is not None:
            # Fed on EVERY step, not only attached ones.  The outcome carries
            # ``classifier_evaluated``, so the scheduler can tell an unattached
            # step from an attached one that the server declined to score --
            # information it needs to decide whether to escalate.  Withholding
            # the unattached outcomes would hide the run's actual duty cycle
            # from it.
            sidecar_scheduler.note_outcome(outcome)
        transition = data["transition"]
        transition["rewards"] = float(outcome.reward)
        transition["masks"] = float(outcome.mask)
        transition["dones"] = bool(outcome.done)
        transition["truncated"] = bool(outcome.truncated)
        transition["success"] = bool(outcome.success)
        transition["classifier_evaluated"] = bool(
            outcome.classifier_evaluated
        )
        if outcome.classifier_evaluated:
            transition["classifier_probability"] = float(
                outcome.classifier_probability
            )
            transition["classifier_threshold"] = float(
                outcome.classifier_threshold
            )
            transition["reward_model_id"] = outcome.reward_model_id
        terminal = bool(outcome.done) or bool(outcome.truncated)
        intervened = data["meta"]["intervened"] == 1
        if intervened:
            total_intervention_steps += 1
        reason = _terminal_reason(outcome) if terminal else ""
        reporter.position(
            episode_id=episode_id,
            episode_step=step_id,
            env_step=env_step,
        )
        control_state, control_owner = _control_state(info, intervened)
        reporter.publish(
            control_state,
            control_owner,
            message=(
                f"terminal transition acknowledged: {reason}"
                if terminal
                else "transition acknowledged"
            ),
            outcome=outcome,
            success=bool(outcome.success),
            terminal_reason=reason,
        )
        final_step = terminal and env_step + 1 >= max_steps
        if terminal:
            if not final_step:
                episode_id += 1
                step_id = 0
                reporter.position(
                    episode_id=episode_id,
                    episode_step=0,
                    env_step=env_step,
                )
            if operator_session is not None:
                reporter.wait_for_home_approval(
                    message=(
                        f"{reason}; robot is holding at the terminal pose. "
                        "Press APPROVE HOME to allow HOME motion."
                    )
                )
                reporter.publish(
                    HOMING,
                    OWNER_NONE,
                    message=(
                        f"{reason}; returning HOME before stop"
                        if final_step
                        else f"{reason}; returning HOME"
                    ),
                )
                # HOME is a motion-safety operation and therefore precedes all
                # optional local deepcopy/pickle I/O.  Disk or memory failure
                # must not strand a successful episode at its terminal pose.
                env.reset(options={"operator_approved_home": True})

        if checkpoint_path:
            # Optional local backup is replay-ready.  This does not resend
            # images: it only materializes the two observations already held
            # by the laptop before writing the local pickle.  On a terminal
            # operator run the arm is already HOME before this block executes.
            transition["observations"] = copy.deepcopy(observation)
            transition["next_observations"] = copy.deepcopy(next_observation)
            replay_data.append(copy.deepcopy(data))
            if intervened:
                intervention_data.append(copy.deepcopy(data))

        if buffer_period and (env_step + 1) % buffer_period == 0:
            _dump_data(
                checkpoint_path,
                run_id,
                env_step,
                replay_data,
                intervention_data,
            )
            replay_data = []
            intervention_data = []

        if terminal:
            if final_step:
                break

            if operator_session is not None:
                reporter.wait_for_scene_ready(
                    message=(
                        f"HOME after {reason}; reset the scene, DISENGAGE, "
                        "then press Resume"
                    )
                )
                reporter.publish(
                    HOMING,
                    OWNER_NONE,
                    message="scene ready accepted; refreshing HOME observation",
                )
            observation, reset_info = env.reset()
            if sidecar_scheduler is not None:
                # Per-episode reset: the interval counter and any escalation
                # state are about "how long since this episode last checked",
                # and carrying them across a reset would put the first query of
                # the new episode at an arbitrary offset.
                sidecar_scheduler.reset()
            source_timestamp_ns = validate_timestamp_ns(
                reset_info.get("timestamp_ns")
            )
            session_id = session_id_factory()
            if not session_id:
                raise ValueError("session_id_factory returned an empty ID")
            observation_id = f"{session_id}:0"
            action_result = network.begin_episode(
                observation,
                run_id=run_id,
                session_id=session_id,
                episode_id=episode_id,
                observation_id=observation_id,
                timestamp_ns=source_timestamp_ns,
            )
            episodes_started += 1
            reporter.publish(
                POLICY_RUNNING,
                OWNER_POLICY,
                message="policy episode running",
                success=False,
                terminal_reason="",
            )
            continue

        if result.action is None:
            raise ActorProtocolError("non-terminal Step returned no action")
        observation = next_observation
        observation_id = next_observation_id
        source_timestamp_ns = next_timestamp_ns
        action_result = result.action
        step_id += 1

    else:
        # Natural max_steps exhaustion is rare in production (default 1M), but
        # it is still an intentional stop rather than a fault.  Leave an armed
        # robot at HOME just as the final-terminal path does.
        if operator_session is not None:
            reporter.publish(
                HOMING,
                OWNER_NONE,
                message="max_steps reached; returning HOME before stop",
                success=False,
                terminal_reason="MAX_STEPS",
            )
            env.reset()

    if checkpoint_path and replay_data:
        _dump_data(
            checkpoint_path,
            run_id,
            max_steps - 1,
            replay_data,
            intervention_data,
        )
    reporter.publish(
        STOPPED,
        OWNER_NONE,
        message="actor run stopped",
    )
    return ActorRunSummary(
        run_id=run_id,
        env_steps=max_steps,
        episodes_started=episodes_started,
        intervention_steps=total_intervention_steps,
        policy_actions_synthetic=policy_action_transform is not None,
        sidecar_attached_steps=sidecar_attached_steps,
        sidecar_build_failures=sidecar_build_failures,
        sidecar_round_trip_ms_mean=sidecar_latency.mean_ms,
        sidecar_round_trip_ms_max=sidecar_latency.max_ms,
        plain_round_trip_ms_mean=plain_latency.mean_ms,
        plain_round_trip_ms_max=plain_latency.max_ms,
    )
