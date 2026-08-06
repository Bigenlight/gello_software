"""The proxy's transition finalizer: the server's MANUAL branch, offline.

WHAT IT IS
----------
The ``finalize_transition`` injectable of
:class:`ur_env.actor_network.ActorSessionService`, with the same two-positional
signature the server's :class:`ur_env.rlpd_receive_server
.RewardTransitionFinalizer` has (``_bind_finalizer`` inspects it to decide
whether a finalizer predates the classifier sidecar; a one-argument callable
here would fail closed on the actor's first sidecar).

It exists because the proxy answers the actor's Step RPC LOCALLY and must put a
``TransitionOutcome`` in that reply, while the reward model lives on the server.
Its whole job is to be indistinguishable from the server's finalizer in the one
mode the proxy supports, and ``tests/test_manual_finalize.py`` asserts that
field for field against the real thing.

WHY MANUAL IS EXACTLY REPRODUCIBLE WITHOUT A CLASSIFIER
-------------------------------------------------------
In MANUAL, ``RewardTransitionFinalizer`` computes

    effective_success = operator_success or (auto_success and classifier_success)

with ``auto_success`` false, so the classifier term is dead: success is the
operator's one-shot ``MARK SUCCESS`` token in ``meta.operator_success`` and
nothing else.  Reward is then ``1.0 if effective_success else 0.0`` -- the
locally proposed reward is DISCARDED either way, which is what makes this
reproducible rather than approximated.  ``dones``/``truncated``/``masks`` pass
through untouched unless success overrides them (success wins over a
simultaneous local truncation, matching one-positive HIL-SERL behaviour).

AUTO IS REFUSED, LOUDLY
-----------------------
``meta.auto_success == 1`` means the operator handed success authority to a
classifier this process does not have.  Silently finalizing it MANUAL-style
would under-claim success forever: episodes would only ever end on the step
limit, and nobody would be told why.  So it raises
:class:`~ur_env.actor_network.ActorProtocolError`, which
``ActorSessionService.step`` turns into a permanent service fault and a failed
RPC -- the actor stops on the first AUTO transition instead of running a
session whose rewards are quietly wrong.

THE CLASSIFIER SIDECAR IS ACCEPTED AND IGNORED
----------------------------------------------
The actor keeps attaching uncropped frames at ~2 Hz and must keep doing so: the
raw request bytes are forwarded to the real server, which classifies them, and
its verdict is what ends up in replay.  Rejecting the sidecar here would break
that; classifying it here is impossible.  So the sidecar is counted and
dropped, and every transition this finalizer produces is marked
``classifier_evaluated=0``.

*** THE KNOWN DIVERGENCE, FOR WHOEVER COMPARES THE TWO OUTCOMES ***
On a step that carried a sidecar, the server's forwarded outcome will report
``classifier_evaluated=1`` with a real probability/threshold/reward_model_id
while this one reports the unevaluated triple.  In MANUAL that difference
CANNOT reach ``rewards``/``masks``/``dones``/``truncated``/``success`` -- those
must match exactly, and a divergence in any of them is a real bug.  A
divergence checker must therefore compare the reward/terminal fields strictly
and treat the classifier telemetry fields as expected-to-differ whenever the
step carried a sidecar.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Callable, Mapping, MutableMapping, Optional

import numpy as np

from ur_env.actor_network import ActorProtocolError, TransitionOutcome

# The server's own scalar validators, imported rather than re-written: they
# decide what "0 or 1", "finite float" and "required text" mean on this wire,
# and a second opinion about that is exactly what a parity test cannot catch.
from ur_env.rlpd_receive_server import (
    _binary_flag,
    _finite_float,
    _required_text,
)


class ManualFinalizerError(ActorProtocolError):
    """This transition cannot be finalized without the server's classifier.

    An ``ActorProtocolError`` subclass on purpose: it is the type every
    finalizer in this repo raises for a contract violation, so the gRPC layer
    maps it the same way and ``ActorSessionService`` faults the service on it
    the same way.  The distinct class only exists so a log or a test can say
    "the proxy refused AUTO" without matching on message text.
    """


class ManualTransitionFinalizer:
    """Finalize one transition the way the server would in MANUAL mode.

    Stateless with respect to the outcome: every field it writes depends only
    on the transition in front of it.  The counters exist for health/telemetry
    and are deliberately named the way ``ActorSessionService._ready_detail``
    reads them, so a proxy's Health line says ``ready`` rather than tripping
    over a missing attribute.
    """

    def __init__(
        self,
        *,
        latency_probe: Optional[Any] = None,
        warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._latency = (
            latency_probe if latency_probe is not None else _disabled_probe()
        )
        self._warn = warn
        self._lock = threading.Lock()
        self._transition_count = 0
        self._success_count = 0
        self._sidecar_ignored_count = 0

    # -- telemetry ---------------------------------------------------------- #

    @property
    def transition_count(self) -> int:
        with self._lock:
            return self._transition_count

    @property
    def success_count(self) -> int:
        """Transitions finalized with an effective (operator) success."""

        with self._lock:
            return self._success_count

    @property
    def sidecar_ignored_count(self) -> int:
        """Sidecars this proxy could not score; the server scores them."""

        with self._lock:
            return self._sidecar_ignored_count

    @property
    def classification_count(self) -> int:
        """Always 0: this finalizer has no classifier, by construction."""

        return 0

    @property
    def classifier_fault_count(self) -> int:
        """Always 0.  "No classifier here" is not "the classifier broke"."""

        return 0

    @property
    def last_classifier_fault(self) -> str:
        return ""

    @property
    def classifier_degraded(self) -> bool:
        return False

    # -- the contract ------------------------------------------------------- #

    def __call__(
        self,
        data: dict[str, Any],
        classifier_sidecar: Optional[Mapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], TransitionOutcome]:
        # Keep the two positional parameters: ActorSessionService._bind_finalizer
        # inspects this signature.  The thin wrapper matches
        # RewardTransitionFinalizer so the profiled region is the whole
        # finalization without reindenting it.
        with self._latency.phase("local_finalize"):
            return self._finalize(data, classifier_sidecar)

    def _finalize(
        self,
        data: dict[str, Any],
        classifier_sidecar: Optional[Mapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], TransitionOutcome]:
        finalized = copy.deepcopy(data)
        if set(finalized) != {"meta", "transition"}:
            raise ActorProtocolError("data must contain exactly meta and transition")
        meta = finalized["meta"]
        transition = finalized["transition"]
        if not isinstance(meta, MutableMapping) or not isinstance(
            transition, MutableMapping
        ):
            raise ActorProtocolError("data.meta and data.transition must be mappings")
        transition_id = _required_text(
            meta.get("transition_id"), name="meta.transition_id"
        )
        auto_success = _binary_flag(
            meta.get("auto_success", False), name="meta.auto_success"
        )
        operator_success = _binary_flag(
            meta.get("operator_success", False),
            name="meta.operator_success",
        )
        if auto_success and operator_success:
            # Checked before the AUTO refusal below so a transition that is
            # malformed in BOTH ways reports the malformation, which is the
            # more specific fault.
            raise ActorProtocolError(
                "meta.operator_success is forbidden while auto_success is enabled"
            )
        if auto_success:
            raise ManualFinalizerError(
                "the local policy proxy is MANUAL only and cannot evaluate the "
                "reward classifier, but this transition arrived with "
                "meta.auto_success=1.  Turn AUTO off in the GUI "
                "(/hil/set_auto_success), or run with HIL_POLICY_MODE=remote "
                "so the server owns success authority again."
            )
        if "next_observations" not in transition:
            # Same guard the server keeps: ActorSessionService attaches it and
            # the forwarded request needs it, so its absence means the pipeline
            # upstream of here is already broken.
            raise ActorProtocolError(
                "transition.next_observations is required before reward finalization"
            )

        # No classifier: the AUTO term of the server's success rule is dead and
        # every classifier field is the unevaluated constant.
        probability = 0.0
        threshold = 0.0
        classifier_success = False
        evaluated = False
        reward_model_id = ""

        effective_success = operator_success
        transition["rewards"] = 1.0 if effective_success else 0.0
        if effective_success:
            transition["masks"] = 0.0
            transition["dones"] = True
            transition["truncated"] = False

        reward = _finite_float(transition.get("rewards"), name="transition.rewards")
        mask = _finite_float(transition.get("masks"), name="transition.masks")
        done = bool(transition.get("dones"))
        truncated = bool(transition.get("truncated"))

        transition["classifier_evaluated"] = np.uint8(evaluated)
        transition["classifier_probability"] = float(probability)
        transition["classifier_threshold"] = float(threshold)
        transition["classifier_success"] = np.uint8(classifier_success)
        transition["success"] = np.uint8(effective_success)
        transition["reward_model_id"] = reward_model_id

        with self._lock:
            self._transition_count += 1
            if effective_success:
                self._success_count += 1
            if classifier_sidecar is not None:
                self._sidecar_ignored_count += 1

        return finalized, TransitionOutcome(
            transition_id=transition_id,
            reward=reward,
            mask=mask,
            done=done,
            truncated=truncated,
            success=effective_success,
            classifier_evaluated=evaluated,
            classifier_probability=probability,
            classifier_threshold=threshold,
            reward_model_id=reward_model_id,
        )


def _disabled_probe() -> Any:
    """The shared no-op probe, imported lazily to keep this module light."""

    from ur_env.server_latency import disabled_probe

    return disabled_probe()


__all__ = [
    "ManualFinalizerError",
    "ManualTransitionFinalizer",
]
