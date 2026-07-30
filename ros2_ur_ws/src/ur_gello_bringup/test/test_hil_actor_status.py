"""Headless tests for HIL actor-status parsing and GUI decisions."""

import json
import time
from types import SimpleNamespace

import pytest

from ur_gello_bringup.hil_actor_status import (
    SCENE_REQUEST_IDLE,
    actor_run_changed,
    actor_banner,
    classifier_verdict_summary,
    engage_button_enabled,
    format_actor_status,
    manual_success_enabled,
    parse_actor_status,
    scene_ready_enabled,
    update_classifier_latch,
    update_terminal_latch,
)


def _status(**overrides):
    return parse_actor_status(json.dumps(_payload(**overrides)))


def _payload(**overrides):
    payload = {
        "schema_version": 2,
        "state": "POLICY_RUNNING",
        "control_owner": "POLICY",
        "run_id": "run-1",
        "episode_id": 2,
        "episode_step": 7,
        "env_step": 42,
        "classifier_evaluated": True,
        "classifier_probability": 0.91,
        "classifier_threshold": 0.20,
        "classifier_env_step": 42,
        "success": False,
        "auto_success": False,
        "terminal_reason": "",
        "message": "running",
    }
    payload.update(overrides)
    return payload


def test_parse_schema_v2_complete_payload():
    status = _status()

    assert status["state"] == "POLICY_RUNNING"
    assert status["control_owner"] == "POLICY"
    assert status["episode_id"] == 2
    assert status["classifier_evaluated"] is True
    assert status["classifier_probability"] == pytest.approx(0.91)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("not-json", "invalid JSON"),
        ("[]", "JSON object"),
        (json.dumps(_payload(schema_version=1)), "schema_version"),
        (json.dumps(_payload(state="UNKNOWN")), "actor state"),
        (json.dumps(_payload(run_id="")), "non-empty"),
        (json.dumps(_payload(classifier_probability=None)), "must not be null"),
        (json.dumps(_payload(classifier_probability=1.5)), r"within \[0, 1\]"),
    ],
)
def test_malformed_payload_raises_one_diagnostic_error(raw, expected):
    with pytest.raises(ValueError, match=expected):
        parse_actor_status(raw)


def test_fixed_schema_rejects_missing_identity_and_counter_fields():
    incomplete = _payload()
    del incomplete["run_id"]
    del incomplete["episode_step"]

    with pytest.raises(ValueError, match="missing required fields"):
        parse_actor_status(json.dumps(incomplete))


def test_schema_v2_requires_boolean_auto_success():
    missing = _payload()
    del missing["auto_success"]

    with pytest.raises(ValueError, match="auto_success"):
        parse_actor_status(json.dumps(missing))
    with pytest.raises(ValueError, match="boolean"):
        parse_actor_status(json.dumps(_payload(auto_success="AUTO")))


def test_sparse_unscored_step_keeps_last_classifier_result():
    first = _status()
    latch = update_classifier_latch(None, first)
    sparse = _status(
        env_step=43,
        episode_step=8,
        classifier_evaluated=False,
        classifier_probability=0.91,
        classifier_threshold=0.20,
        classifier_env_step=42,
    )

    retained = update_classifier_latch(latch, sparse)

    assert retained == latch
    assert retained["probability"] == pytest.approx(0.91)
    assert retained["env_step"] == 42

    formatted = format_actor_status(
        sparse,
        classifier_latch=retained,
        terminal_latch="",
        age_s=0.125,
    )
    assert "NO (sparse/unscored step)" in formatted["classifier_current"]
    assert "verdict: SUCCESS" in formatted["classifier_score"]
    assert "0.910" in formatted["classifier_score"]
    assert "last eval env step: 42" in formatted["classifier_score"]
    assert formatted["age"] == "status age: 0.12 s"


def test_late_join_sparse_status_recovers_actor_carried_classifier_result():
    sparse = _status(
        env_step=11,
        classifier_evaluated=False,
        classifier_probability=0.83,
        classifier_threshold=0.50,
        classifier_env_step=10,
    )

    latch = update_classifier_latch(None, sparse)

    assert latch["probability"] == pytest.approx(0.83)
    assert latch["threshold"] == pytest.approx(0.50)
    assert latch["env_step"] == 10


def test_classifier_verdict_uses_server_strict_greater_than_rule():
    boundary = _status(
        classifier_probability=0.2,
        classifier_threshold=0.2,
    )
    latch = update_classifier_latch(None, boundary)
    formatted = format_actor_status(
        boundary,
        classifier_latch=latch,
        terminal_latch="",
        age_s=0.0,
    )

    assert "verdict: NOT SUCCESS" in formatted["classifier_score"]

    prominent, mode, verdict = classifier_verdict_summary(
        latch, auto_success=False
    )
    assert "LAST CLASSIFIER: NOT SUCCESS" in prominent
    assert "p(success)=0.200" in prominent
    assert "MANUAL" in mode
    assert "display-only" in mode
    assert verdict is False


def test_prominent_classifier_stays_visible_in_manual_and_auto_modes():
    latch = {"probability": 0.91, "threshold": 0.20, "env_step": 42}

    manual = classifier_verdict_summary(latch, auto_success=False)
    auto = classifier_verdict_summary(latch, auto_success=True)

    assert manual[0] == auto[0]
    assert "SUCCESS" in manual[0]
    assert manual[2] is True
    assert "display-only" in manual[1]
    assert "may end the episode" in auto[1]


def test_manual_success_button_gate_is_manual_active_and_one_shot_only():
    active = _status(auto_success=False)
    base = dict(
        status=active,
        auto_success=False,
        request_pending=False,
        success_queued=False,
        service_ready=True,
    )
    assert manual_success_enabled(**base)
    assert not manual_success_enabled(**dict(base, auto_success=True))
    assert not manual_success_enabled(**dict(base, request_pending=True))
    assert not manual_success_enabled(**dict(base, success_queued=True))
    assert not manual_success_enabled(**dict(base, service_ready=False))
    assert not manual_success_enabled(
        **dict(base, status=dict(active, state="WAIT_SCENE_READY"))
    )


def test_startup_status_accepts_actor_minus_one_env_step_sentinels():
    startup = _status(
        state="HOMING",
        env_step=-1,
        classifier_evaluated=False,
        classifier_probability=0.0,
        classifier_threshold=0.0,
        classifier_env_step=-1,
    )

    assert startup["env_step"] == -1
    assert startup["classifier_env_step"] == -1
    assert update_classifier_latch(None, startup) is None


def test_home_approval_state_is_valid_and_uses_the_operator_button_gate():
    waiting = _status(
        state="WAIT_HOME_APPROVAL",
        control_owner="HOLD",
        classifier_evaluated=False,
        terminal_reason="TRUNCATED",
    )

    assert scene_ready_enabled(
        waiting, request_state=SCENE_REQUEST_IDLE, service_ready=True
    )
    text, _ = actor_banner(waiting, engaged=False)
    assert text.startswith("WAIT_HOME_APPROVAL")


def test_terminal_latches_through_wait_and_clears_when_policy_runs():
    terminal = _status(
        state="HOMING", terminal_reason="SUCCESS", success=True
    )
    latched = update_terminal_latch("", terminal)
    assert latched == "SUCCESS"

    wait = _status(
        state="WAIT_SCENE_READY",
        classifier_evaluated=False,
        terminal_reason="",
    )
    assert update_terminal_latch(latched, wait) == "SUCCESS"
    assert update_terminal_latch(latched, _status()) == ""


def test_actor_state_has_banner_priority_but_no_status_keeps_legacy_text():
    text, _ = actor_banner(
        _status(
            state="WAIT_SCENE_READY",
            classifier_evaluated=False,
        ),
        engaged=False,
    )
    assert text.startswith("WAIT_SCENE_READY")
    assert "policy in control" not in text

    assert actor_banner(None, engaged=False)[0] == "DISENGAGED — policy in control"
    assert actor_banner(None, engaged=True)[0].startswith("ENGAGED")


def test_engage_gate_allows_legacy_active_and_release_but_not_wait_or_homing():
    assert engage_button_enabled(None, engaged=False)
    assert engage_button_enabled(_status(), engaged=False)

    wait = _status(
        state="WAIT_SCENE_READY",
        classifier_evaluated=False,
    )
    homing = dict(wait, state="HOMING")
    assert not engage_button_enabled(wait, engaged=False)
    assert not engage_button_enabled(homing, engaged=False)
    assert engage_button_enabled(wait, engaged=True)  # safe release remains possible


def test_scene_ready_is_one_idle_ready_request_only_in_wait_state():
    wait = _status(
        state="WAIT_SCENE_READY",
        classifier_evaluated=False,
    )
    assert scene_ready_enabled(
        wait, request_state=SCENE_REQUEST_IDLE, service_ready=True
    )
    assert not scene_ready_enabled(
        wait, request_state="pending", service_ready=True
    )
    assert not scene_ready_enabled(
        wait, request_state=SCENE_REQUEST_IDLE, service_ready=False
    )
    assert not scene_ready_enabled(
        _status(), request_state=SCENE_REQUEST_IDLE, service_ready=True
    )


def test_new_actor_run_is_a_latch_reset_boundary():
    previous = _status(run_id="run-old")
    current = _status(
        run_id="run-new",
        state="HOMING",
        classifier_evaluated=False,
        classifier_probability=0.0,
        classifier_threshold=0.0,
        classifier_env_step=-1,
    )

    assert actor_run_changed(previous, current)
    assert not actor_run_changed(current, dict(current))
    assert not actor_run_changed(None, current)


def test_gui_scene_request_is_cleared_when_actor_run_changes():
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    class _Future:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class _WindowState:
        def _set_scene_result(self, text, color):
            self.result = (text, color)

    future = _Future()
    window = _WindowState()
    window._scene_request_state = gui_module._SCENE_ACCEPTED_WAIT
    window._scene_future = future
    window._scene_call_started = 123.0
    window._last_actor_state = "HOMING"
    window._last_actor_status = _status(run_id="run-old", state="HOMING")

    gui_module.MainWindow._poll_scene_ready(
        window,
        {
            "status": _status(
                run_id="run-new",
                state="WAIT_SCENE_READY",
                control_owner="NONE",
                classifier_evaluated=False,
            )
        },
    )

    assert future.cancelled is True
    assert window._scene_request_state == SCENE_REQUEST_IDLE
    assert window._scene_future is None
    assert window._scene_call_started is None
    assert window._last_actor_status["run_id"] == "run-new"
    assert "Robot is HOME" in window.result[0]


class _SceneFuture:
    def __init__(self, *, done=False, response=None, failure=None):
        self._done = done
        self._response = response
        self._failure = failure
        self.cancelled = False

    def done(self):
        return self._done

    def result(self):
        if self._failure is not None:
            raise self._failure
        return self._response

    def cancel(self):
        self.cancelled = True


class _SceneWindowState:
    def _set_scene_result(self, text, color):
        self.result = (text, color)


def _scene_window(gui_module, request_state, future=None):
    window = _SceneWindowState()
    window._scene_request_state = request_state
    window._scene_future = future
    window._scene_call_started = time.monotonic()
    window._last_actor_state = "WAIT_SCENE_READY"
    window._last_actor_status = _status(
        state="WAIT_SCENE_READY",
        control_owner="NONE",
        classifier_evaluated=False,
    )
    return window


def test_gui_home_approval_completes_at_home_then_reenables_next_iteration():
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    window = _scene_window(gui_module, gui_module._SCENE_ACCEPTED_WAIT)
    window._last_actor_state = "HOMING"
    window._last_actor_status = _status(
        state="HOMING", control_owner="NONE", classifier_evaluated=False
    )
    home = _status(
        state="WAIT_SCENE_READY",
        control_owner="NONE",
        classifier_evaluated=False,
        terminal_reason="TRUNCATED",
    )

    gui_module.MainWindow._poll_scene_ready(window, {"status": home})

    assert window._scene_request_state == SCENE_REQUEST_IDLE
    assert "HOME complete" in window.result[0]


@pytest.mark.parametrize(
    "response, expected_state, expected_text",
    [
        (
            SimpleNamespace(success=True, message="accepted"),
            "accepted_wait",
            "Accepted",
        ),
        (
            SimpleNamespace(success=False, message="rejected"),
            SCENE_REQUEST_IDLE,
            "FAILED: rejected",
        ),
    ],
)
def test_gui_scene_future_accept_and_reject_paths(
    response, expected_state, expected_text
):
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    future = _SceneFuture(done=True, response=response)
    window = _scene_window(gui_module, gui_module._SCENE_CALL_PENDING, future)

    gui_module.MainWindow._poll_scene_ready(
        window, {"status": dict(window._last_actor_status)}
    )

    assert window._scene_request_state == expected_state
    assert window._scene_future is None
    assert expected_text in window.result[0]


def test_gui_scene_future_exception_is_reported_and_recoverable():
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    future = _SceneFuture(done=True, failure=RuntimeError("service lost"))
    window = _scene_window(gui_module, gui_module._SCENE_CALL_PENDING, future)

    gui_module.MainWindow._poll_scene_ready(
        window, {"status": dict(window._last_actor_status)}
    )

    assert window._scene_request_state == SCENE_REQUEST_IDLE
    assert window._scene_future is None
    assert "service error: service lost" in window.result[0]


@pytest.mark.parametrize("state", ["FAULT", "STOPPED"])
def test_gui_fault_or_stop_clears_pending_scene_request(state):
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    future = _SceneFuture(done=False)
    window = _scene_window(gui_module, gui_module._SCENE_CALL_PENDING, future)
    status = _status(
        state=state,
        control_owner="NONE",
        classifier_evaluated=False,
        terminal_reason="FAULT" if state == "FAULT" else "",
    )

    gui_module.MainWindow._poll_scene_ready(window, {"status": status})

    assert future.cancelled is True
    assert window._scene_request_state == SCENE_REQUEST_IDLE
    assert f"actor entered {state}" in window.result[0]


def test_gui_scene_future_timeout_cancels_and_reenables_request():
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    future = _SceneFuture(done=False)
    window = _scene_window(gui_module, gui_module._SCENE_CALL_PENDING, future)
    window._scene_call_started = (
        time.monotonic() - gui_module._SCENE_CALL_TIMEOUT_S - 0.1
    )

    gui_module.MainWindow._poll_scene_ready(
        window, {"status": dict(window._last_actor_status)}
    )

    assert future.cancelled is True
    assert window._scene_request_state == SCENE_REQUEST_IDLE
    assert "timed out" in window.result[0]
