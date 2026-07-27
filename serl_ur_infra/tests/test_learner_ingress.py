"""Process-local fault gate between live replay acceptance and sampling."""

from __future__ import annotations

from types import SimpleNamespace
import threading

import pytest

from ur_env.learner import (
    FaultGatedReplayIngress,
    ReplayIngressFaultError,
    ReplayIngressView,
)


class _ScriptedIngress:
    def __init__(self) -> None:
        self.items: list[tuple[dict, bool]] = []
        self.accept_calls = 0
        self.replay_sample_calls = 0
        self.intervention_sample_calls = 0
        self.fail_accept = False
        self.accept_entered = threading.Event()
        self.release_accept = threading.Event()
        self.block_accept = False

    def __call__(self, data, intervened):
        self.accept_calls += 1
        self.accept_entered.set()
        if self.block_accept:
            if not self.release_accept.wait(timeout=2.0):
                raise TimeoutError("test did not release acceptance")
        # Model ReplayIngress's intentional partial-route retry behavior: the
        # replay side may already contain data when intervention routing fails.
        self.items.append((data, intervened))
        if self.fail_accept:
            raise BufferError("intervention route failed")

    def status(self):
        return SimpleNamespace(
            replay_size=len(self.items),
            intervention_size=sum(intervened for _, intervened in self.items),
        )

    def sample_replay(self, batch_size, **kwargs):
        self.replay_sample_calls += 1
        return {"route": "replay", "batch_size": batch_size, "kwargs": kwargs}

    def sample_intervention(self, batch_size, **kwargs):
        self.intervention_sample_calls += 1
        return {
            "route": "intervention",
            "batch_size": batch_size,
            "kwargs": kwargs,
        }


def test_successful_accept_and_sampling_delegate_through_one_facade():
    raw = _ScriptedIngress()
    gated = FaultGatedReplayIngress(raw)
    data = {"transition": "one"}

    gated(data, True)

    assert gated.healthy
    assert gated.fault is None
    assert gated.status().replay_size == 1
    assert gated.sample_replay(4, packed=True) == {
        "route": "replay",
        "batch_size": 4,
        "kwargs": {"packed": True},
    }
    assert gated.sample_intervention(2)["route"] == "intervention"
    assert raw.items == [(data, True)]


def test_accept_fault_is_latched_before_waiting_sample_can_touch_raw_ingress():
    raw = _ScriptedIngress()
    raw.block_accept = True
    raw.fail_accept = True
    gated = FaultGatedReplayIngress(raw)
    accept_errors: list[BaseException] = []
    sample_errors: list[BaseException] = []
    sample_attempted = threading.Event()

    def accept():
        try:
            gated({"transition": "partial"}, True)
        except BaseException as exc:
            accept_errors.append(exc)

    def sample():
        sample_attempted.set()
        try:
            gated.sample_replay(1)
        except BaseException as exc:
            sample_errors.append(exc)

    accept_thread = threading.Thread(target=accept)
    sample_thread = threading.Thread(target=sample)
    accept_thread.start()
    assert raw.accept_entered.wait(timeout=1.0)
    sample_thread.start()
    assert sample_attempted.wait(timeout=1.0)

    # Sampling is waiting on the facade lock, not entering the raw store while
    # the two-route acceptance operation is in flight.
    sample_thread.join(timeout=0.05)
    assert sample_thread.is_alive()
    assert raw.replay_sample_calls == 0

    raw.release_accept.set()
    accept_thread.join(timeout=1.0)
    sample_thread.join(timeout=1.0)

    assert not accept_thread.is_alive()
    assert not sample_thread.is_alive()
    assert len(accept_errors) == 1
    assert isinstance(accept_errors[0], BufferError)
    assert len(sample_errors) == 1
    assert isinstance(sample_errors[0], ReplayIngressFaultError)
    assert raw.replay_sample_calls == 0
    assert gated.fault is not None
    assert gated.fault.error_type == "BufferError"
    assert gated.fault.detail == "intervention route failed"


def test_fault_is_permanent_but_status_remains_available_for_diagnostics():
    raw = _ScriptedIngress()
    raw.fail_accept = True
    gated = FaultGatedReplayIngress(raw)

    with pytest.raises(BufferError, match="intervention route failed"):
        gated({"transition": "partial"}, True)

    assert not gated.healthy
    assert gated.status().replay_size == 1
    with pytest.raises(ReplayIngressFaultError, match="permanently faulted"):
        gated.ensure_healthy()
    with pytest.raises(ReplayIngressFaultError, match="BufferError"):
        gated.sample_intervention(1)
    with pytest.raises(ReplayIngressFaultError, match="BufferError"):
        gated({"transition": "retry"}, True)
    assert raw.accept_calls == 1
    assert raw.intervention_sample_calls == 0


def test_replay_ingress_view_checks_optional_learner_health_contract():
    raw = _ScriptedIngress()
    gated = FaultGatedReplayIngress(raw)
    replay = ReplayIngressView(gated, "replay")
    intervention = ReplayIngressView(gated, "intervention")

    gated({"transition": "one"}, True)
    assert len(replay) == 1
    assert len(intervention) == 1
    assert replay.sample(3)["batch_size"] == 3

    raw.fail_accept = True
    with pytest.raises(BufferError):
        gated({"transition": "partial"}, True)
    with pytest.raises(ReplayIngressFaultError, match="permanently faulted"):
        len(replay)
    with pytest.raises(ReplayIngressFaultError, match="permanently faulted"):
        intervention.sample(1)


def test_replay_ingress_view_remains_compatible_without_health_hook():
    raw = _ScriptedIngress()
    replay = ReplayIngressView(raw, "replay")

    raw({"transition": "one"}, False)

    assert len(replay) == 1
    assert replay.sample(2)["route"] == "replay"
