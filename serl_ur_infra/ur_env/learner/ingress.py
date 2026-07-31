"""Fault-gated access to one live learner replay ingress.

``ReplayIngress`` deliberately retains enough route state to finish a partial
replay/intervention insert when its caller retries.  The actor service instead
fails closed after an acceptance exception, so a concurrent learner must not
sample a replay-only half of that failed route.  This facade supplies that
process-local boundary without changing the receive-only ingress contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Mapping, Protocol


class ReplayIngressLike(Protocol):
    """The live-ingress operations used by the learner composition root."""

    def __call__(self, data: dict[str, Any], intervened: bool) -> None:
        ...

    def status(self) -> Any:
        ...

    def sample_replay(self, batch_size: int, **kwargs: Any) -> Mapping[str, Any]:
        ...

    def sample_intervention(
        self, batch_size: int, **kwargs: Any
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class ReplayIngressFault:
    """Bounded diagnostic for the first failed acceptance operation."""

    error_type: str
    detail: str


class ReplayIngressFaultError(RuntimeError):
    """Learner access was rejected after replay acceptance faulted."""


class FaultGatedReplayIngress:
    """Serialize acceptance/sampling and permanently gate a failed ingress.

    The first exception raised by the wrapped acceptance callback is re-raised
    unchanged so ``ActorSessionService`` keeps its existing fail-stop behavior.
    Before releasing the facade lock, a bounded fault record is installed.
    Subsequent acceptance and learner sampling then fail explicitly without
    touching the wrapped ingress.  ``status()`` remains available for
    diagnostics even in that fault state.
    """

    _MAX_FAULT_DETAIL = 2_000

    def __init__(self, ingress: ReplayIngressLike) -> None:
        for name in (
            "__call__",
            "status",
            "sample_replay",
            "sample_intervention",
        ):
            if not callable(getattr(ingress, name, None)):
                raise TypeError(f"ingress must provide callable {name}()")
        self._ingress = ingress
        self._lock = threading.RLock()
        self._fault: ReplayIngressFault | None = None

    @property
    def require_grasp_penalty(self) -> bool:
        """Expose the strict learner contract without exposing raw stores."""

        return getattr(self._ingress, "require_grasp_penalty", False) is True

    @property
    def observation_representation(self) -> str | None:
        return getattr(self._ingress, "observation_representation", None)

    @property
    def augmentation(self) -> str | None:
        return getattr(self._ingress, "augmentation", None)

    @property
    def fault(self) -> ReplayIngressFault | None:
        with self._lock:
            return self._fault

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._fault is None

    def ensure_healthy(self) -> None:
        """Fail explicitly if an acceptance operation previously failed."""

        with self._lock:
            self._raise_if_faulted()

    def __call__(self, data: dict[str, Any], intervened: bool) -> None:
        with self._lock:
            self._raise_if_faulted()
            try:
                self._ingress(data, intervened)
            except Exception as exc:
                if self._fault is None:
                    detail = str(exc) or "replay ingress acceptance failed"
                    self._fault = ReplayIngressFault(
                        error_type=type(exc).__name__,
                        detail=detail[: self._MAX_FAULT_DETAIL],
                    )
                raise

    def status(self) -> Any:
        """Return diagnostic status regardless of the acceptance fault state."""

        with self._lock:
            return self._ingress.status()

    def prime_observation(self, **kwargs: Any) -> None:
        """Warm the wrapped ingress's trunk cache under the same lock.

        Unlike ``__call__`` this inserts nothing, so a failure cannot leave a
        replay-only half of a route behind.  It therefore propagates the
        exception *without* installing a permanent fault -- latching here would
        let a transient extractor error retire an otherwise healthy learner.
        A caller holding an already-faulted ingress is still refused.
        """

        prime = getattr(self._ingress, "prime_observation", None)
        if not callable(prime):
            return
        with self._lock:
            self._raise_if_faulted()
            prime(**kwargs)

    def sample_replay(self, batch_size: int, **kwargs: Any) -> Mapping[str, Any]:
        with self._lock:
            self._raise_if_faulted()
            return self._ingress.sample_replay(batch_size=batch_size, **kwargs)

    def sample_intervention(
        self, batch_size: int, **kwargs: Any
    ) -> Mapping[str, Any]:
        with self._lock:
            self._raise_if_faulted()
            return self._ingress.sample_intervention(
                batch_size=batch_size, **kwargs
            )

    def _raise_if_faulted(self) -> None:
        fault = self._fault
        if fault is not None:
            raise ReplayIngressFaultError(
                "replay ingress acceptance is permanently faulted: "
                f"{fault.error_type}: {fault.detail}"
            )
