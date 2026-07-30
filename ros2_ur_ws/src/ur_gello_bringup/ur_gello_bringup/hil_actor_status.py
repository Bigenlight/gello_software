"""Pure helpers for the HIL actor-status operator panel.

This module deliberately imports neither ROS nor Qt.  The actor status wire is
display-only telemetry; the independent ``/hil/deadman`` safety contract stays
in :mod:`gello_hil_gui_node` and must not depend on this parser.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional


SCHEMA_VERSION = 2
ACTOR_STATUS_TOPIC = "/hil/actor_status"
SCENE_READY_SERVICE = "/hil/scene_ready"
AUTO_SUCCESS_SERVICE = "/hil/set_auto_success"
MANUAL_SUCCESS_SERVICE = "/hil/manual_success"

ACTOR_STATES = frozenset(
    {
        "HOMING",
        "WAIT_HOME_APPROVAL",
        "WAIT_SCENE_READY",
        "POLICY_RUNNING",
        "HUMAN_INTERVENTION",
        "HOLD",
        "FAULT",
        "STOPPED",
    }
)
CONTROL_OWNERS = frozenset({"NONE", "POLICY", "HUMAN", "HOLD"})
REQUIRED_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "control_owner",
        "run_id",
        "episode_id",
        "episode_step",
        "env_step",
        "classifier_evaluated",
        "classifier_probability",
        "classifier_threshold",
        "classifier_env_step",
        "success",
        "auto_success",
        "terminal_reason",
        "message",
    }
)

# These are the states in which a new human intervention makes sense.  An
# already-engaged deadman may always be released, including in every other
# state; ``engage_button_enabled`` encodes that asymmetry explicitly.
ACTIVE_CONTROL_STATES = frozenset(
    {"POLICY_RUNNING", "HUMAN_INTERVENTION", "HOLD"}
)

SCENE_REQUEST_IDLE = "idle"


def actor_run_changed(
    previous: Optional[Mapping[str, Any]], current: Mapping[str, Any]
) -> bool:
    """Return true when telemetry switches between two non-empty run IDs.

    Classifier and terminal values are intentionally latched across episode
    boundaries. They must not leak into a newly started actor process, though.
    """

    if previous is None:
        return False
    previous_run = str(previous.get("run_id") or "")
    current_run = str(current.get("run_id") or "")
    return bool(previous_run and current_run and previous_run != current_run)


def _optional_int(
    payload: Mapping[str, Any], name: str, *, minimum: int = 0
) -> Optional[int]:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or null")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _optional_probability(
    payload: Mapping[str, Any], name: str
) -> Optional[float]:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return result


def _required_string(
    payload: Mapping[str, Any], name: str, *, nonempty: bool = False
) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if nonempty and not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _bool(payload: Mapping[str, Any], name: str, default: bool = False) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def parse_actor_status(raw: str) -> dict[str, Any]:
    """Parse and validate one schema-v2 ``/hil/actor_status`` JSON payload.

    Schema v2 is a fixed, complete payload.  Rejecting missing identity,
    counters, or owner fields prevents malformed telemetry from replacing a
    known-good snapshot or leaking classifier/terminal latches across runs.
    Unknown extra fields are ignored for forward-compatible diagnostics.

    ``ValueError`` is the sole public failure mode.  The ROS callback catches it
    and retains the last valid snapshot, so malformed telemetry cannot crash the
    GUI or disturb the deadman publisher.
    """

    if not isinstance(raw, str):
        raise ValueError("actor status payload must be a JSON string")
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("actor status must be a JSON object")

    missing = sorted(REQUIRED_STATUS_FIELDS.difference(payload))
    if missing:
        raise ValueError(
            "actor status missing required fields: " + ", ".join(missing)
        )

    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {schema_version!r}; expected "
            f"{SCHEMA_VERSION}"
        )

    state = payload.get("state")
    if state not in ACTOR_STATES:
        raise ValueError(
            f"unknown actor state {state!r}; expected one of "
            + ", ".join(sorted(ACTOR_STATES))
        )

    control_owner = _required_string(payload, "control_owner")
    if control_owner not in CONTROL_OWNERS:
        raise ValueError(f"unknown control_owner {control_owner!r}")
    run_id = _required_string(payload, "run_id", nonempty=True)
    episode_id = _optional_int(payload, "episode_id")
    episode_step = _optional_int(payload, "episode_step")
    env_step = _optional_int(payload, "env_step", minimum=-1)
    classifier_env_step = _optional_int(
        payload, "classifier_env_step", minimum=-1
    )
    if None in (episode_id, episode_step, env_step, classifier_env_step):
        raise ValueError("actor status counters must not be null")

    evaluated = _bool(payload, "classifier_evaluated")
    probability = _optional_probability(payload, "classifier_probability")
    threshold = _optional_probability(payload, "classifier_threshold")
    if probability is None or threshold is None:
        raise ValueError("classifier probability and threshold must not be null")
    success = _bool(payload, "success")
    auto_success = _bool(payload, "auto_success")
    terminal_reason = _required_string(payload, "terminal_reason")
    message = _required_string(payload, "message")

    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "control_owner": control_owner,
        "run_id": run_id,
        "episode_id": episode_id,
        "episode_step": episode_step,
        "env_step": env_step,
        "classifier_evaluated": evaluated,
        "classifier_probability": probability,
        "classifier_threshold": threshold,
        "classifier_env_step": classifier_env_step,
        "success": success,
        "auto_success": auto_success,
        "terminal_reason": terminal_reason,
        "message": message,
    }


def update_classifier_latch(
    previous: Optional[Mapping[str, Any]], status: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    """Keep the most recent *evaluated* classifier result across sparse steps."""

    probability = status.get("classifier_probability")
    threshold = status.get("classifier_threshold")
    classifier_env_step = status.get("classifier_env_step")
    has_carried_result = (
        isinstance(classifier_env_step, int)
        and not isinstance(classifier_env_step, bool)
        and classifier_env_step >= 0
    )
    if (
        not bool(status.get("classifier_evaluated", False))
        and not has_carried_result
    ):
        return None if previous is None else dict(previous)
    if probability is None or threshold is None:
        # Parsed messages cannot reach this branch, but keeping this helper
        # total makes it safe to use in tests and future non-ROS callers.
        return None if previous is None else dict(previous)
    if classifier_env_step is None:
        classifier_env_step = status.get("env_step")
    return {
        "probability": float(probability),
        "threshold": float(threshold),
        "env_step": classifier_env_step,
    }


def update_terminal_latch(
    previous: str, status: Mapping[str, Any]
) -> str:
    """Retain a terminal verdict through HOMING/WAIT, clear on active policy."""

    reason = str(status.get("terminal_reason") or "")
    if reason:
        return reason
    if status.get("state") in {"POLICY_RUNNING", "HUMAN_INTERVENTION"}:
        return ""
    return previous


def engage_button_enabled(
    status: Optional[Mapping[str, Any]], *, engaged: bool
) -> bool:
    """Whether the primary deadman button may act.

    With no actor status, return the legacy behaviour (enabled).  Releasing an
    already-engaged deadman is always allowed.  Only a new ENGAGE is gated by
    actor state.
    """

    if engaged or status is None:
        return True
    return status.get("state") in ACTIVE_CONTROL_STATES


def scene_ready_enabled(
    status: Optional[Mapping[str, Any]],
    *,
    request_state: str,
    service_ready: bool,
) -> bool:
    """True only for one idle request at an operator approval gate."""

    return bool(
        status is not None
        and status.get("state") in {"WAIT_HOME_APPROVAL", "WAIT_SCENE_READY"}
        and request_state == SCENE_REQUEST_IDLE
        and service_ready
    )


def classifier_verdict_summary(
    classifier_latch: Optional[Mapping[str, Any]], *, auto_success: bool
) -> tuple[str, str, Optional[bool]]:
    """Return prominent classifier text, mode context, and strict verdict.

    The status stream is sparse, so the wording deliberately says ``LAST``.
    The third value is ``None`` before any evaluated result, otherwise the
    strict server rule ``probability > threshold``.
    """

    mode_text = (
        "AUTO — classifier success may end the episode"
        if auto_success
        else "MANUAL — classifier is display-only; use MARK SUCCESS"
    )
    if classifier_latch is None:
        return "LAST CLASSIFIER: NO RESULT", mode_text, None
    probability = float(classifier_latch["probability"])
    threshold = float(classifier_latch["threshold"])
    success = probability > threshold
    verdict = "SUCCESS" if success else "NOT SUCCESS"
    return (
        f"LAST CLASSIFIER: {verdict}   p(success)={probability:.3f}",
        mode_text,
        success,
    )


def manual_success_enabled(
    status: Optional[Mapping[str, Any]],
    *,
    auto_success: bool,
    request_pending: bool,
    success_queued: bool,
    service_ready: bool,
) -> bool:
    """Enable MARK SUCCESS only for one active MANUAL episode request."""

    return bool(
        status is not None
        and status.get("state") in ACTIVE_CONTROL_STATES
        and not auto_success
        and not request_pending
        and not success_queued
        and service_ready
    )


def actor_banner(
    status: Optional[Mapping[str, Any]], *, engaged: bool
) -> tuple[str, str]:
    """Return ``(text, colour)`` for the large state banner.

    Actor state is authoritative whenever available.  The exact legacy banner
    text is preserved before the first status so existing preflight behaviour
    remains unchanged.
    """

    if status is None:
        if engaged:
            return "ENGAGED — intervening (deadman held)", "#22aa22"
        return "DISENGAGED — policy in control", "#888888"

    state = status["state"]
    banners = {
        "HOMING": ("HOMING — robot returning to reset", "#dd8800"),
        "WAIT_HOME_APPROVAL": (
            "WAIT_HOME_APPROVAL — robot holding; approve before HOME motion",
            "#dd8800",
        ),
        "WAIT_SCENE_READY": (
            "WAIT_SCENE_READY — reset scene, then START / NEXT ITERATION",
            "#1565c0",
        ),
        "POLICY_RUNNING": ("POLICY_RUNNING — policy in control", "#2277aa"),
        "HUMAN_INTERVENTION": (
            "HUMAN_INTERVENTION — GELLO/deadman in control",
            "#22aa22",
        ),
        "HOLD": ("HOLD — robot command held", "#dd8800"),
        "FAULT": ("FAULT — actor stopped; operator attention required", "#cc3333"),
        "STOPPED": ("STOPPED — actor not running", "#666666"),
    }
    return banners[state]


def _counter(value: Any) -> str:
    return "—" if value is None else str(value)


def format_actor_status(
    status: Optional[Mapping[str, Any]],
    *,
    classifier_latch: Optional[Mapping[str, Any]],
    terminal_latch: str,
    age_s: Optional[float],
) -> dict[str, str]:
    """Format the actor panel without touching any Qt objects."""

    if status is None:
        return {
            "owner_state": "actor status: not received (legacy/preflight mode)",
            "run": "—",
            "episode": "episode/step: — / —   env step: —",
            "classifier_current": "current step evaluated: —",
            "classifier_score": "p(success): —   threshold: —   last eval env step: —",
            "terminal": "terminal reason: —",
            "age": "status age: —",
            "message": "Waiting for /hil/actor_status",
        }

    evaluated = bool(status.get("classifier_evaluated", False))
    current = "YES" if evaluated else "NO (sparse/unscored step)"
    if classifier_latch is None:
        score = (
            "verdict: —   p(success): —   threshold: —   "
            "last eval env step: —"
        )
    else:
        probability = float(classifier_latch["probability"])
        threshold = float(classifier_latch["threshold"])
        # The server contract is strict p > threshold, not >=.
        verdict = "SUCCESS" if probability > threshold else "NOT SUCCESS"
        score = (
            f"verdict: {verdict}   p(success): {probability:.3f}   "
            f"threshold: {threshold:.3f}   "
            f"last eval env step: {_counter(classifier_latch.get('env_step'))}"
        )
    owner = status.get("control_owner") or "—"
    run_id = status.get("run_id") or "—"
    terminal = terminal_latch or "—"
    age = "—" if age_s is None else f"{max(0.0, float(age_s)):.2f} s"
    return {
        "owner_state": f"{status['state']}   owner: {owner}",
        "run": str(run_id),
        "episode": (
            f"episode/step: {_counter(status.get('episode_id'))} / "
            f"{_counter(status.get('episode_step'))}   "
            f"env step: {_counter(status.get('env_step'))}"
        ),
        "classifier_current": f"current step evaluated: {current}",
        "classifier_score": score,
        "terminal": f"terminal reason: {terminal}",
        "age": f"status age: {age}",
        "message": str(status.get("message") or ""),
    }
