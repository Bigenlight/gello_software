"""Deterministic concurrency tests for actor-priority GPU scheduling."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time


_INFRA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_INFRA))

from ur_env.learner.scheduling import ActorInferenceGate  # noqa: E402


def _wait_until(predicate, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


def test_waiting_actor_runs_before_next_learner_update():
    gate = ActorInferenceGate()
    first_learner_entered = threading.Event()
    release_first_learner = threading.Event()
    actor_entered = threading.Event()
    release_actor = threading.Event()
    second_learner_attempted = threading.Event()
    second_learner_entered = threading.Event()
    order: list[str] = []

    def first_learner() -> None:
        with gate.learner_update():
            first_learner_entered.set()
            assert release_first_learner.wait(timeout=2.0)

    def actor() -> None:
        with gate.actor_request():
            order.append("actor")
            actor_entered.set()
            assert release_actor.wait(timeout=2.0)

    def second_learner() -> None:
        second_learner_attempted.set()
        with gate.learner_update():
            order.append("learner")
            second_learner_entered.set()

    threads = [
        threading.Thread(target=first_learner, daemon=True),
        threading.Thread(target=actor, daemon=True),
        threading.Thread(target=second_learner, daemon=True),
    ]
    threads[0].start()
    assert first_learner_entered.wait(timeout=1.0)

    threads[1].start()
    assert _wait_until(lambda: gate.actor_waiters == 1)

    threads[2].start()
    assert second_learner_attempted.wait(timeout=1.0)
    release_first_learner.set()

    assert actor_entered.wait(timeout=1.0)
    assert not second_learner_entered.wait(timeout=0.05)
    release_actor.set()
    assert second_learner_entered.wait(timeout=1.0)

    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    assert order == ["actor", "learner"]
    assert gate.actor_waiters == 0
    assert gate.active_actor_requests == 0
    assert not gate.learner_update_active

