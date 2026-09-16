"""Cooperative GPU scheduling between actor inference and learner updates."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Iterator


class ActorInferenceGate:
    """Give waiting actor RPCs priority at learner update boundaries.

    A running JAX update cannot be preempted.  Once it finishes, however, an
    actor which arrived in the meantime must get the GPU before the learner
    starts another update.  The waiter count is part of the condition rather
    than an ``Event`` checked by the learner so the check/start race cannot let
    a new update jump ahead of an already waiting actor.

    The actor context is intentionally broad: it covers frozen-trunk encoding,
    transition acceptance and policy inference.  The learner context covers one
    optimizer call and its synchronous validation.  Consequently the maximum
    scheduling delay introduced by the learner is one optimizer call, not one
    complete critic+actor CTA cycle or an unbounded UTD burst.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._actor_waiters = 0
        self._active_actor_requests = 0
        self._learner_update_active = False

    @property
    def actor_waiters(self) -> int:
        with self._condition:
            return self._actor_waiters

    @property
    def active_actor_requests(self) -> int:
        with self._condition:
            return self._active_actor_requests

    @property
    def learner_update_active(self) -> bool:
        with self._condition:
            return self._learner_update_active

    @contextmanager
    def actor_request(self) -> Iterator[None]:
        """Wait for the current learner update, then block subsequent ones."""

        with self._condition:
            self._actor_waiters += 1
            try:
                while self._learner_update_active:
                    self._condition.wait()
                self._active_actor_requests += 1
            finally:
                self._actor_waiters -= 1
            self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._active_actor_requests -= 1
                self._condition.notify_all()

    @contextmanager
    def learner_update(self) -> Iterator[None]:
        """Run one update only when no actor is active or already waiting."""

        with self._condition:
            while (
                self._learner_update_active
                or self._actor_waiters
                or self._active_actor_requests
            ):
                self._condition.wait()
            self._learner_update_active = True
        try:
            yield
        finally:
            with self._condition:
                self._learner_update_active = False
                self._condition.notify_all()


__all__ = ["ActorInferenceGate"]
