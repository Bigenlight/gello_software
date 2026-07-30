"""Headless tests for HIL actor-status parsing and GUI decisions."""

import json
import time
from types import SimpleNamespace

import pytest

from ur_gello_bringup.hil_actor_status import (
    ABORT_EPISODE_SERVICE,
    SCENE_REQUEST_IDLE,
    abort_episode_enabled,
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


# --------------------------------------------------------------------------- #
# ABORT EPISODE                                                                #
# --------------------------------------------------------------------------- #
def test_abort_service_name_is_the_frozen_contract():
    assert ABORT_EPISODE_SERVICE == "/hil/abort_episode"


def test_end_episode_operator_strings_do_not_overclaim():
    """The operator ACTS on these four strings, so lock what they may say.

    Nothing is discarded: the transport has no cancel/retract RPC and the
    server inserts into replay inside the Step handler before it Acks.  And the
    deadman does not stop the policy -- it gates only the GELLO follower.
    """

    from ur_gello_bringup import gello_hil_gui_node as gui_module

    captions = (
        gui_module._ABORT_BUTTON_TEXT,
        gui_module._ABORT_ARMED_TEXT,
        gui_module._ABORT_CONFIRM_WARN,
        gui_module._ABORT_RELEASE_TEXT,
    )
    for text in captions:
        low = text.lower()
        assert "discard" not in low, text
        assert "cannot be recovered" not in low, text
        assert "cannot undo" not in low, text
        assert "arm stopped" not in low, text
        assert "immediately" not in low, text

    # ... and each must positively carry its own truth.
    assert gui_module._ABORT_BUTTON_TEXT == "END EPISODE (truncate & re-home)"
    assert "data stays" in gui_module._ABORT_ARMED_TEXT
    assert "cannot be withdrawn" in gui_module._ABORT_CONFIRM_WARN
    assert "NO success" in gui_module._ABORT_CONFIRM_WARN
    assert "replay buffer" in gui_module._ABORT_CONFIRM_WARN
    assert "GELLO following stopped" in gui_module._ABORT_RELEASE_TEXT
    assert "policy stops within one step" in gui_module._ABORT_RELEASE_TEXT

    # The armed caption sits on the button, so it must stay button-sized.
    assert len(gui_module._ABORT_ARMED_TEXT) <= 60


def test_abort_button_gate_is_active_episode_and_one_request_only():
    base = dict(
        status=_status(),
        request_pending=False,
        service_ready=True,
    )
    assert abort_episode_enabled(**base)
    assert not abort_episode_enabled(**dict(base, request_pending=True))
    assert not abort_episode_enabled(**dict(base, service_ready=False))
    assert not abort_episode_enabled(**dict(base, status=None))
    for state in ("WAIT_SCENE_READY", "WAIT_HOME_APPROVAL", "HOMING",
                  "FAULT", "STOPPED"):
        assert not abort_episode_enabled(
            **dict(base, status=dict(base["status"], state=state))
        ), state
    for state in ("POLICY_RUNNING", "HUMAN_INTERVENTION", "HOLD"):
        assert abort_episode_enabled(
            **dict(base, status=dict(base["status"], state=state))
        ), state


def test_abort_is_legal_in_auto_mode_unlike_mark_success():
    """Abort is not a success assertion, so AUTO must not take it away."""

    auto = _status(auto_success=True)

    assert not manual_success_enabled(
        auto,
        auto_success=True,
        request_pending=False,
        success_queued=False,
        service_ready=True,
    )
    assert abort_episode_enabled(auto, request_pending=False, service_ready=True)


class _AbortWindowState:
    """Duck-typed stand-in: exercises MainWindow logic without a Qt window."""

    def __init__(self, **fields):
        self._abort_future = None
        self._abort_call_started = None
        self._abort_release_pending = False
        self._abort_queued = False
        self._abort_target = None
        self._abort_confirmed_target = None
        self._abort_last_status = None
        self.result = ("", "")
        self.__dict__.update(fields)

    def _set_abort_result(self, text, color):
        self.result = (text, color)

    @staticmethod
    def _cancel_future(future):
        if future is not None:
            future.cancel()


def _poll_abort(window, status):
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    gui_module.MainWindow._poll_abort(window, {"status": status})


def test_abort_future_acceptance_latches_a_one_shot_queued_flag():
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    active = _status()
    window = _AbortWindowState(
        _abort_future=_SceneFuture(
            done=True, response=SimpleNamespace(success=True, message="queued")
        ),
        _abort_call_started=time.monotonic(),
        _abort_target=(active["run_id"], active["episode_id"]),
    )

    _poll_abort(window, active)

    assert window._abort_future is None
    assert window._abort_release_pending is False
    assert window._abort_queued is True
    assert "END EPISODE queued" in window.result[0]
    # The Ack must NOT be read as "the episode is gone from replay".
    assert "discard" not in window.result[0].lower()

    # The latch is what keeps a second Trigger from racing the actor.  With the
    # future resolved and nothing mid-release, ``_abort_queued`` is now the ONLY
    # thing that can make a request pending -- so toggling it and re-reading the
    # REAL ``_abort_request_pending`` proves the wiring, instead of hand-feeding
    # the flag into ``abort_episode_enabled`` (which would only re-test that).
    pending = gui_module.MainWindow._abort_request_pending
    assert pending(window) is True
    window._abort_queued = False
    assert pending(window) is False
    window._abort_queued = True
    assert not abort_episode_enabled(
        active, request_pending=pending(window), service_ready=True
    )


def test_abort_rejection_and_service_error_are_surfaced_and_recoverable():
    rejected = _AbortWindowState(
        _abort_future=_SceneFuture(
            done=True,
            response=SimpleNamespace(success=False, message="not active"),
        ),
        _abort_call_started=time.monotonic(),
    )
    _poll_abort(rejected, _status())
    assert rejected._abort_queued is False
    assert "FAILED to end episode: not active" in rejected.result[0]

    raised = _AbortWindowState(
        _abort_future=_SceneFuture(done=True, failure=RuntimeError("gone")),
        _abort_call_started=time.monotonic(),
    )
    _poll_abort(raised, _status())
    assert "service error: gone" in raised.result[0]
    # Recoverable: nothing pending, so the button comes back.
    assert abort_episode_enabled(
        _status(), request_pending=False, service_ready=True
    )


def test_abort_future_timeout_cancels_and_reenables_the_button():
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    future = _SceneFuture(done=False)
    window = _AbortWindowState(
        _abort_future=future,
        _abort_call_started=(
            time.monotonic() - gui_module._OPERATOR_CALL_TIMEOUT_S - 0.1
        ),
    )

    _poll_abort(window, _status())

    assert future.cancelled is True
    assert window._abort_future is None
    assert "timed out" in window.result[0]
    assert "deadman stayed released" in window.result[0]


def test_abort_queued_latch_clears_at_the_episode_boundary():
    active = _status()
    window = _AbortWindowState(
        _abort_queued=True,
        _abort_target=(active["run_id"], active["episode_id"]),
    )

    # Still the same active episode -> latch holds.
    _poll_abort(window, active)
    assert window._abort_queued is True

    # Actor consumed it and left the active states -> latch releases.
    _poll_abort(
        window,
        _status(
            state="WAIT_HOME_APPROVAL",
            control_owner="HOLD",
            classifier_evaluated=False,
            terminal_reason="OPERATOR_ABORT",
        ),
    )
    assert window._abort_queued is False
    assert window._abort_target is None


def _aborted_status(**overrides):
    return _status(
        state="WAIT_HOME_APPROVAL",
        control_owner="HOLD",
        classifier_evaluated=False,
        terminal_reason="OPERATOR_ABORT",
        **overrides,
    )


def test_abort_outcome_is_read_from_terminal_reason_once_per_episode():
    """The confirmation fires on the EDGE into OPERATOR_ABORT, once."""

    aborted = _aborted_status()
    window = _AbortWindowState(_abort_last_status=_status())

    _poll_abort(window, aborted)
    assert "OPERATOR_ABORT" in window.result[0]
    assert "truncated" in window.result[0]
    # It must not claim the data went away -- nothing is retractable.
    assert "discard" not in window.result[0].lower()
    assert window._abort_confirmed_target == ("run-1", 2)

    window.result = ("", "")
    _poll_abort(window, aborted)
    assert window.result == ("", "")  # not rewritten every 100 ms tick

    # The actor clears terminal_reason on the next episode's first status, so
    # episode 3's own end is a fresh edge and confirms again.
    _poll_abort(window, _status(episode_id=3))
    assert window.result == ("", "")
    _poll_abort(window, _aborted_status(episode_id=3))
    assert "OPERATOR_ABORT" in window.result[0]
    assert window._abort_confirmed_target == ("run-1", 3)


def test_stale_terminal_reason_cannot_burn_the_next_episodes_confirmation():
    """Robustness: a sticky OPERATOR_ABORT must not confirm episode N+1.

    If the actor ever carries the reason across the boundary again, latching on
    the LEVEL would consume the one-shot on episode 3 before it starts, and the
    operator's second end-episode of the run would silently get no confirmation.
    """

    window = _AbortWindowState(_abort_last_status=_status())
    _poll_abort(window, _aborted_status())
    assert window._abort_confirmed_target == ("run-1", 2)

    window.result = ("", "")
    stale = _status(episode_id=3, terminal_reason="OPERATOR_ABORT")
    _poll_abort(window, stale)  # episode 3 running, reason left over
    assert window.result == ("", "")
    assert window._abort_confirmed_target == ("run-1", 2)

    # ... and when the operator really ends episode 3, they are still told --
    # the request they made for THAT episode is the second, independent path.
    window._abort_target = ("run-1", 3)
    _poll_abort(window, _aborted_status(episode_id=3))
    assert "OPERATOR_ABORT" in window.result[0]
    assert window._abort_confirmed_target == ("run-1", 3)


def test_unrequested_terminal_reason_seen_first_is_not_reported():
    """A GUI started mid-run must not invent a confirmation it never saw."""

    window = _AbortWindowState()  # no previous status, no request of our own

    _poll_abort(window, _aborted_status())

    assert window.result == ("", "")
    assert window._abort_confirmed_target is None


def test_new_actor_run_clears_a_pending_abort():
    future = _SceneFuture(done=False)
    window = _AbortWindowState(
        _abort_future=future,
        _abort_call_started=time.monotonic(),
        _abort_release_pending=True,
        _abort_queued=True,
        _abort_target=("run-old", 2),
        _abort_last_status=_status(run_id="run-old"),
    )

    _poll_abort(
        window,
        _status(
            run_id="run-new",
            state="HOMING",
            classifier_evaluated=False,
            classifier_env_step=-1,
        ),
    )

    assert future.cancelled is True
    assert window._abort_future is None
    assert window._abort_release_pending is False
    assert window._abort_queued is False
    assert window._abort_target is None
    assert "END EPISODE control reset" in window.result[0]


class _FakeAbortNode:
    """Records the ORDER of deadman publishes vs. the abort Trigger."""

    def __init__(self, status, *, service_ready=True):
        self.status = status
        self.service_ready = service_ready
        self.calls = []

    def get_actor_snapshot(self):
        return {"status": self.status}

    def publish_deadman(self, engaged, gain):
        self.calls.append(("publish_deadman", bool(engaged), float(gain)))

    def abort_service_ready(self):
        return self.service_ready

    def call_abort_episode(self):
        self.calls.append(("call_abort_episode",))
        return _SceneFuture(done=False)


def _abort_click_window(monkeypatch, node):
    from ur_gello_bringup import gello_hil_gui_node as gui_module

    deferred = []

    class _FakeQTimer:
        @staticmethod
        def singleShot(ms, callback):
            deferred.append((ms, callback))

    monkeypatch.setattr(gui_module, "QTimer", _FakeQTimer)

    window = _AbortWindowState(
        _node=node,
        _engaged=True,
        _armed={"primary": True},
        _closing=False,
    )
    window._slider_gain = lambda: 1.0
    window._refresh = lambda: None
    window._abort_request_pending = (
        lambda: gui_module.MainWindow._abort_request_pending(window)
    )
    window._dispatch_abort_episode = (
        lambda: gui_module.MainWindow._dispatch_abort_episode(window)
    )
    return gui_module, window, deferred


def test_abort_click_releases_the_deadman_before_calling_the_service(
    monkeypatch,
):
    """SAFETY: the 30 Hz follower stops in ~33 ms; the actor takes ~512 ms."""

    node = _FakeAbortNode(_status(state="HUMAN_INTERVENTION",
                                  control_owner="HUMAN"))
    gui_module, window, deferred = _abort_click_window(monkeypatch, node)

    gui_module.MainWindow._do_abort_episode(window)

    # The deadman release happened synchronously, and NOTHING else has run yet.
    assert node.calls == [("publish_deadman", False, 1.0)]
    assert window._engaged is False
    assert window._armed["primary"] is False
    assert window._abort_release_pending is True
    assert window._abort_target == ("run-1", 2)
    assert "Deadman released" in window.result[0]

    # Only afterwards, on the deferred tick, is the Trigger sent.
    assert [ms for ms, _ in deferred] == [gui_module._SCENE_RELEASE_DELAY_MS]
    deferred[0][1]()
    assert node.calls[-1] == ("call_abort_episode",)
    assert window._abort_release_pending is False
    assert window._abort_future is not None


def test_abort_dispatch_is_skipped_but_deadman_stays_released(monkeypatch):
    node = _FakeAbortNode(_status(state="HUMAN_INTERVENTION",
                                  control_owner="HUMAN"))
    gui_module, window, deferred = _abort_click_window(monkeypatch, node)

    gui_module.MainWindow._do_abort_episode(window)
    # The episode ended on its own during the 100 ms release window.
    node.status = _status(
        state="WAIT_HOME_APPROVAL",
        control_owner="HOLD",
        classifier_evaluated=False,
        terminal_reason="TRUNCATED",
    )
    deferred[0][1]()

    assert node.calls == [("publish_deadman", False, 1.0)]
    assert window._abort_future is None
    assert "deadman stayed released" in window.result[0]
    assert window._engaged is False


def test_abort_deadman_publish_failure_aborts_the_request_not_the_release(
    monkeypatch,
):
    node = _FakeAbortNode(_status(state="HUMAN_INTERVENTION",
                                  control_owner="HUMAN"))

    def _boom(engaged, gain):
        raise RuntimeError("rclpy is down")

    node.publish_deadman = _boom
    gui_module, window, deferred = _abort_click_window(monkeypatch, node)

    gui_module.MainWindow._do_abort_episode(window)

    # Intent is still recorded locally, so the 20 Hz tick keeps publishing
    # engaged=False and the env's staleness watchdog fail-stops regardless.
    assert window._engaged is False
    assert window._abort_release_pending is False
    assert deferred == []
    assert "deadman publish error" in window.result[0]


# --------------------------------------------------------------------------- #
# The real click entry point: gate pre-check + two-click confirm               #
# --------------------------------------------------------------------------- #
def _abort_confirm_window(monkeypatch, node):
    """``_abort_click_window`` plus the real ``_armed_click`` machinery."""

    gui_module, window, deferred = _abort_click_window(monkeypatch, node)
    window.fired = 0
    window.hints = []

    def _fire():
        window.fired += 1

    def _show(text, msec=0):
        window.hints.append(text)

    window._do_abort_episode = _fire
    window.statusBar = lambda: SimpleNamespace(showMessage=_show)
    window._disarm = lambda key: gui_module.MainWindow._disarm(window, key)
    window._armed_click = (
        lambda key, action, warn="the robot WILL move":
        gui_module.MainWindow._armed_click(window, key, action, warn=warn)
    )
    return gui_module, window


def test_abort_click_needs_two_clicks_and_the_warning_is_true(monkeypatch):
    """POLICY_RUNNING: the case the abort gate exists for, and the case where
    'arm stopped' would be a lie."""

    gui_module, window = _abort_confirm_window(
        monkeypatch, _FakeAbortNode(_status())
    )

    gui_module.MainWindow._on_abort_episode(window)

    assert window.fired == 0  # first click only ARMS
    assert window._armed["abort_episode"] is True
    warning = window.hints[-1]
    assert "NO success" in warning
    assert "cannot be withdrawn" in warning
    # BLOCKER 1: the transport has no cancel/retract RPC, so the confirm text
    # must never promise that anything is thrown away or recoverable.
    assert "discard" not in warning.lower()
    assert "cannot be recovered" not in warning

    gui_module.MainWindow._on_abort_episode(window)

    assert window.fired == 1
    assert window._armed["abort_episode"] is False


@pytest.mark.parametrize(
    "node_kwargs, status_kwargs, window_fields",
    [
        ({"service_ready": False}, {}, {}),
        ({}, {"state": "WAIT_HOME_APPROVAL", "control_owner": "HOLD",
              "classifier_evaluated": False}, {}),
        ({}, {}, {"_abort_queued": True}),
        ({}, {}, {"_abort_release_pending": True}),
    ],
)
def test_abort_click_pre_check_blocks_arming(
    monkeypatch, node_kwargs, status_kwargs, window_fields
):
    """The gate runs BEFORE the two-click arm, so a blocked click is silent."""

    node = _FakeAbortNode(_status(**status_kwargs), **node_kwargs)
    gui_module, window = _abort_confirm_window(monkeypatch, node)
    window.__dict__.update(window_fields)

    gui_module.MainWindow._on_abort_episode(window)

    assert window.fired == 0
    assert window.hints == []
    assert window._armed.get("abort_episode") is not True


def test_abort_click_without_actor_status_is_a_no_op(monkeypatch):
    gui_module, window = _abort_confirm_window(
        monkeypatch, _FakeAbortNode(None)
    )

    gui_module.MainWindow._on_abort_episode(window)

    assert window.fired == 0
    assert window.hints == []


def test_armed_abort_does_not_fire_after_the_episode_already_ended(monkeypatch):
    """The pre-check re-runs on the confirming click, not just the arming one."""

    node = _FakeAbortNode(_status())
    gui_module, window = _abort_confirm_window(monkeypatch, node)

    gui_module.MainWindow._on_abort_episode(window)
    assert window._armed["abort_episode"] is True

    node.status = _status(
        state="WAIT_HOME_APPROVAL",
        control_owner="HOLD",
        classifier_evaluated=False,
        terminal_reason="TRUNCATED",
    )
    gui_module.MainWindow._on_abort_episode(window)

    assert window.fired == 0


def test_abort_post_click_label_never_claims_the_arm_stopped(monkeypatch):
    """BLOCKER 2: the deadman gates the follower only; the policy path never
    reads it, so the arm keeps moving for up to one actor loop period."""

    node = _FakeAbortNode(_status())  # POLICY_RUNNING == policy is driving
    gui_module, window, _deferred = _abort_click_window(monkeypatch, node)

    gui_module.MainWindow._do_abort_episode(window)

    label = window.result[0]
    assert "GELLO following stopped" in label
    assert "arm stopped" not in label
    assert "immediately" not in label.lower()
    assert "policy stops within one step" in label
