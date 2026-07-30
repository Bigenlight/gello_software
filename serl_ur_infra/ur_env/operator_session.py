"""ROS operator boundary for starting and resetting a real HIL actor.

The policy loop deliberately knows nothing about ROS message classes.  It
hands :class:`ActorStatus` values to ``RosOperatorSession`` and blocks on one
edge-triggered ``/hil/scene_ready`` gate.  ROS imports are lazy so importing
``ur_env.remote_actor`` continues to work in the non-ROS learner/test venv.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import sys
import threading
from typing import Any, Callable, Optional


STATUS_SCHEMA_VERSION = 2
ACTOR_STATUS_TOPIC = "/hil/actor_status"
SCENE_READY_SERVICE = "/hil/scene_ready"
AUTO_SUCCESS_SERVICE = "/hil/set_auto_success"
MANUAL_SUCCESS_SERVICE = "/hil/manual_success"
WAIT_STATUS_REPUBLISH_S = 0.5
WAIT_STATUS_MAX_CONSECUTIVE_FAILURES = 3

HOMING = "HOMING"
WAIT_HOME_APPROVAL = "WAIT_HOME_APPROVAL"
WAIT_SCENE_READY = "WAIT_SCENE_READY"
POLICY_RUNNING = "POLICY_RUNNING"
HUMAN_INTERVENTION = "HUMAN_INTERVENTION"
HOLD = "HOLD"
FAULT = "FAULT"
STOPPED = "STOPPED"

ACTOR_STATES = frozenset(
    {
        HOMING,
        WAIT_HOME_APPROVAL,
        WAIT_SCENE_READY,
        POLICY_RUNNING,
        HUMAN_INTERVENTION,
        HOLD,
        FAULT,
        STOPPED,
    }
)
ACTIVE_CONTROL_STATES = frozenset(
    {POLICY_RUNNING, HUMAN_INTERVENTION, HOLD}
)

OWNER_NONE = "NONE"
OWNER_POLICY = "POLICY"
OWNER_HUMAN = "HUMAN"
OWNER_HOLD = "HOLD"
CONTROL_OWNERS = frozenset(
    {OWNER_NONE, OWNER_POLICY, OWNER_HUMAN, OWNER_HOLD}
)


@dataclass(frozen=True)
class ActorStatus:
    """Versioned payload published on ``/hil/actor_status``."""

    schema_version: int
    state: str
    control_owner: str
    run_id: str
    episode_id: int
    episode_step: int
    env_step: int
    classifier_evaluated: bool
    classifier_probability: float
    classifier_threshold: float
    classifier_env_step: int
    success: bool
    terminal_reason: str
    message: str
    auto_success: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != STATUS_SCHEMA_VERSION:
            raise ValueError(
                f"status schema_version must be {STATUS_SCHEMA_VERSION}"
            )
        if self.state not in ACTOR_STATES:
            raise ValueError(f"unknown actor state {self.state!r}")
        if self.control_owner not in CONTROL_OWNERS:
            raise ValueError(f"unknown control owner {self.control_owner!r}")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("status run_id is required")
        for name in ("episode_id", "episode_step"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"status {name} must be a non-negative int")
        for name in ("env_step", "classifier_env_step"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < -1:
                raise ValueError(f"status {name} must be an int >= -1")
        for name in ("classifier_probability", "classifier_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"status {name} must be finite and in [0, 1]")
        if not isinstance(self.classifier_evaluated, bool):
            raise ValueError("status classifier_evaluated must be bool")
        if not isinstance(self.success, bool):
            raise ValueError("status success must be bool")
        if not isinstance(self.auto_success, bool):
            raise ValueError("status auto_success must be bool")
        if not isinstance(self.terminal_reason, str):
            raise ValueError("status terminal_reason must be str")
        if not isinstance(self.message, str):
            raise ValueError("status message must be str")

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


class ActorStatusTracker:
    """Build complete statuses while retaining the last scored classifier data.

    An unscored transition sets only ``classifier_evaluated`` back to false.
    The most recent evaluated probability, threshold and global env step remain
    visible, which lets the GUI distinguish "not scored this tick" from "the
    classifier has never produced a usable result".
    """

    def __init__(self, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id is required")
        self.run_id = run_id
        self._probability = 0.0
        self._threshold = 0.0
        self._classifier_env_step = -1

    def status(
        self,
        *,
        state: str,
        control_owner: str,
        episode_id: int,
        episode_step: int,
        env_step: int,
        classifier_evaluated: bool = False,
        classifier_probability: Optional[float] = None,
        classifier_threshold: Optional[float] = None,
        success: bool = False,
        terminal_reason: str = "",
        message: str = "",
        auto_success: bool = False,
    ) -> ActorStatus:
        evaluated = bool(classifier_evaluated)
        if evaluated:
            if classifier_probability is None or classifier_threshold is None:
                raise ValueError(
                    "evaluated classifier status requires probability and threshold"
                )
            probability = float(classifier_probability)
            threshold = float(classifier_threshold)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("classifier_probability must be in [0, 1]")
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("classifier_threshold must be in [0, 1]")
            if int(env_step) < 0:
                raise ValueError("evaluated classifier status requires env_step >= 0")
            self._probability = probability
            self._threshold = threshold
            self._classifier_env_step = int(env_step)

        return ActorStatus(
            schema_version=STATUS_SCHEMA_VERSION,
            state=state,
            control_owner=control_owner,
            run_id=self.run_id,
            episode_id=int(episode_id),
            episode_step=int(episode_step),
            env_step=int(env_step),
            classifier_evaluated=evaluated,
            classifier_probability=self._probability,
            classifier_threshold=self._threshold,
            classifier_env_step=self._classifier_env_step,
            success=bool(success),
            terminal_reason=str(terminal_reason),
            message=str(message),
            auto_success=bool(auto_success),
        )


def resolve_deadman_source(env: Any) -> Any:
    """Return the deadman object embedded in a GelloIntervention wrapper."""

    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        expert = getattr(current, "expert", None)
        deadman = getattr(expert, "deadman", None)
        if callable(getattr(deadman, "is_engaged", None)):
            return deadman
        current = getattr(current, "env", None)
    raise RuntimeError(
        "armed topic actor has no reachable GelloIntervention deadman source"
    )


def resolve_backend_node(env: Any) -> Any:
    """Return the existing UR backend node whose executor already spins."""

    base = getattr(env, "unwrapped", None)
    backend = getattr(base, "backend", None)
    node = getattr(backend, "_node", None)
    if node is None:
        raise RuntimeError("armed topic actor has no reachable backend ROS node")
    return node


class RosOperatorSession:
    """ROS operator gates plus episode-scoped success controls.

    Success mode starts in MANUAL.  ``/hil/manual_success`` queues one token
    against the latest active ``(run_id, episode_id)``; the actor consumes that
    token exactly once with :meth:`consume_operator_success`.  The token is
    cleared on mode changes and actor run/episode/terminal boundaries so an
    operator click cannot leak into a later episode.
    """

    def __init__(
        self,
        node: Any,
        deadman: Any,
        *,
        warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        if node is None:
            raise ValueError("node is required")
        if not callable(getattr(deadman, "is_engaged", None)):
            raise ValueError("deadman must provide is_engaged()")
        self._node = node
        self._deadman = deadman
        self._warn = warn or (
            lambda message: print(
                f"[operator-session] WARNING: {message}",
                file=sys.stderr,
                flush=True,
            )
        )
        self._lock = threading.Lock()
        self._scene_ready = threading.Event()
        self._waiting = False
        self._waiting_state: Optional[str] = None
        self._auto_success = False
        self._operator_success_pending: Optional[tuple[str, int]] = None
        self._latest_status: Optional[ActorStatus] = None
        self._status_failures = 0
        self._publisher = None
        self._string_type = None

        # The WAIT gate is operable only when the GUI can discover its state,
        # so publisher construction is mandatory just like service creation.
        # Individual runtime publishes remain best-effort during policy motion;
        # WAIT below escalates only sustained consecutive failures.
        from std_msgs.msg import String

        self._string_type = String
        self._publisher = node.create_publisher(
            String, ACTOR_STATUS_TOPIC, 10
        )

        # The gate is mandatory.  Import or service-creation failure propagates
        # and prevents an armed topic actor from entering its policy loop.
        from std_srvs.srv import SetBool, Trigger

        self._scene_ready_service = node.create_service(
            Trigger, SCENE_READY_SERVICE, self._on_scene_ready
        )
        self._auto_success_service = node.create_service(
            SetBool, AUTO_SUCCESS_SERVICE, self._on_set_auto_success
        )
        self._manual_success_service = node.create_service(
            Trigger, MANUAL_SUCCESS_SERVICE, self._on_manual_success
        )

    @classmethod
    def from_environment(
        cls,
        env: Any,
        *,
        warn: Optional[Callable[[str], None]] = None,
    ) -> "RosOperatorSession":
        return cls(
            resolve_backend_node(env),
            resolve_deadman_source(env),
            warn=warn,
        )

    @property
    def waiting(self) -> bool:
        with self._lock:
            return self._waiting

    @property
    def auto_success(self) -> bool:
        """Whether classifier success may automatically end an episode."""

        with self._lock:
            return self._auto_success

    def consume_operator_success(self, run_id: str, episode_id: int) -> bool:
        """Consume the matching MANUAL success token exactly once.

        A mismatched call returns false without stealing the token belonging to
        the active episode.  Normal status publication clears stale tokens when
        the actor crosses a run, episode, or active-control boundary.
        """

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id is required")
        if (
            not isinstance(episode_id, int)
            or isinstance(episode_id, bool)
            or episode_id < 0
        ):
            raise ValueError("episode_id must be a non-negative int")
        target = (run_id, episode_id)
        with self._lock:
            if self._operator_success_pending != target:
                return False
            self._operator_success_pending = None
            return True

    def publish(self, status: ActorStatus) -> bool:
        """Publish one complete JSON status and report whether it succeeded."""

        if not isinstance(status, ActorStatus):
            raise TypeError("status must be ActorStatus")
        if self._publisher is None or self._string_type is None:
            return False
        with self._lock:
            previous = self._latest_status
            crossed_boundary = bool(
                previous is not None
                and (
                    previous.run_id != status.run_id
                    or previous.episode_id != status.episode_id
                )
            )
            if crossed_boundary or status.state not in ACTIVE_CONTROL_STATES:
                self._operator_success_pending = None
            status = replace(status, auto_success=self._auto_success)
            self._latest_status = status
        try:
            message = self._string_type()
            message.data = json.dumps(
                status.as_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            self._publisher.publish(message)
            return True
        except Exception as exc:  # noqa: BLE001 - status is best effort
            self._status_failures += 1
            if self._status_failures == 1 or self._status_failures % 100 == 0:
                self._warn(
                    f"{ACTOR_STATUS_TOPIC} publish failed "
                    f"({self._status_failures}x): {exc}"
                )
            return False

    def _on_set_auto_success(self, request: Any, response: Any) -> Any:
        """Set MANUAL/AUTO mode; every mode edge cancels a queued click."""

        requested = bool(request.data)
        with self._lock:
            changed = requested != self._auto_success
            self._auto_success = requested
            if changed:
                self._operator_success_pending = None
            latest = self._latest_status
        response.success = True
        mode_message = (
            "AUTO: classifier success may end the episode"
            if requested
            else "MANUAL: only operator success may end the episode"
        )
        response.message = mode_message if changed else f"already {mode_message}"
        # Publish the authoritative mode immediately so a newly opened GUI and
        # the client that made this request do not wait for another policy step.
        if latest is not None:
            self.publish(latest)
        return response

    def _on_manual_success(self, _request: Any, response: Any) -> Any:
        """Queue one success token for the latest active MANUAL episode."""

        with self._lock:
            status = self._latest_status
            if self._auto_success:
                response.success = False
                response.message = "manual success is disabled in AUTO mode"
                return response
            if status is None:
                response.success = False
                response.message = "actor status has not been published"
                return response
            if status.state not in ACTIVE_CONTROL_STATES:
                response.success = False
                response.message = (
                    f"actor is not in an active episode ({status.state})"
                )
                return response
            target = (status.run_id, status.episode_id)
            if self._operator_success_pending is not None:
                response.success = False
                response.message = "operator success is already queued"
                return response
            self._operator_success_pending = target

        response.success = True
        response.message = (
            "operator success queued for next transition: "
            f"run={target[0]} episode={target[1]}"
        )
        return response

    def wait_for_scene_ready(self, status: ActorStatus) -> None:
        """Advertise WAIT until consuming exactly one accepted Trigger edge.

        The status topic deliberately remains ordinary volatile telemetry, so
        repeat the current WAIT snapshot while blocked.  A GUI that starts or
        restarts after the actor reached this gate can therefore recover the
        START/NEXT button without any policy step or BeginEpisode call.
        """

        self._wait_for_operator(status, expected_state=WAIT_SCENE_READY)

    def wait_for_home_approval(self, status: ActorStatus) -> None:
        """Hold at the terminal pose until the operator approves HOME motion."""

        self._wait_for_operator(status, expected_state=WAIT_HOME_APPROVAL)

    def _wait_for_operator(
        self, status: ActorStatus, *, expected_state: str
    ) -> None:
        if status.state != expected_state:
            raise ValueError(f"wait status must use {expected_state}")
        with self._lock:
            if self._waiting:
                raise RuntimeError("operator wait is already active")
            self._scene_ready.clear()
            self._waiting = True
            self._waiting_state = expected_state
        try:
            # Set _waiting before publishing so a GUI reacting to this exact
            # status cannot race into an "actor is not waiting" rejection.
            consecutive_failures = 0
            while True:
                if self.publish(status):
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if (
                        consecutive_failures
                        >= WAIT_STATUS_MAX_CONSECUTIVE_FAILURES
                    ):
                        raise RuntimeError(
                            f"cannot publish {ACTOR_STATUS_TOPIC} while "
                            "waiting for scene ready"
                        )
                if self._scene_ready.wait(WAIT_STATUS_REPUBLISH_S):
                    break
        finally:
            with self._lock:
                self._waiting = False
                self._waiting_state = None
                self._scene_ready.clear()

    def _on_scene_ready(self, _request: Any, response: Any) -> Any:
        """Trigger callback: fresh DISENGAGED only; never latch an early call."""

        with self._lock:
            if not self._waiting:
                response.success = False
                response.message = "actor is not waiting for an operator request"
                return response
            if self._scene_ready.is_set():
                response.success = False
                response.message = "operator request already accepted"
                return response
            waiting_state = self._waiting_state

        if waiting_state == WAIT_HOME_APPROVAL:
            with self._lock:
                if (
                    not self._waiting
                    or self._waiting_state != WAIT_HOME_APPROVAL
                    or self._scene_ready.is_set()
                ):
                    response.success = False
                    response.message = "actor is no longer accepting HOME approval"
                    return response
                self._scene_ready.set()
            response.success = True
            response.message = "HOME approved; robot will move"
            return response

        try:
            fresh_engaged = getattr(self._deadman, "fresh_engaged", None)
            engaged = bool(
                fresh_engaged()
                if callable(fresh_engaged)
                else self._deadman.is_engaged()
            )
        except Exception as exc:  # stale/missing heartbeat is fail-closed
            response.success = False
            response.message = f"deadman is not fresh: {exc}"
            return response
        if engaged:
            response.success = False
            response.message = "DISENGAGE the deadman before resuming policy"
            return response

        # Re-check after the deadman call: wait teardown could have raced while
        # that call acquired the source's own lock.
        with self._lock:
            if (
                not self._waiting
                or self._waiting_state != WAIT_SCENE_READY
                or self._scene_ready.is_set()
            ):
                response.success = False
                response.message = "actor is no longer accepting scene ready"
                return response
            self._scene_ready.set()
        response.success = True
        response.message = "scene ready accepted; policy episode will start"
        return response
