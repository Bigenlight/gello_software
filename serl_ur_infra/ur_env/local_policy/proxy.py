"""The laptop-side ActorTransport server: local actions, forwarded transitions.

WHAT THIS IS
------------
One process on laptop3 that speaks the SAME wire protocol as the learner
server, so ``remote_actor.py`` connects to it with zero changes and cannot tell
the difference at the transport level.  It answers the ACTION half of every RPC
itself (local inference, ~2 ms on this laptop's GPU instead of a ~156 ms
blocking round trip to junhyeong_ai) and hands the TRANSITION half to
:class:`~ur_env.local_policy.uploader.TransitionUploader`, which replays the
request bytes to the real server off the control loop.

Nothing here is a new protocol implementation.  The pieces are:

* :class:`ur_env.actor_network.ActorSessionService` -- the real one, with the
  real session/dedup/validation rules.  Two injectables differ from the
  server's composition: ``sample_action`` is
  :class:`~ur_env.local_policy.runtime.LocalPolicyRuntime` and
  ``finalize_transition`` is
  :class:`~ur_env.local_policy.manual_finalize.ManualTransitionFinalizer`.
* :class:`ur_env.grpc_actor_transport.GrpcActorServicer` -- the real handler
  bodies, subclassed only to (a) gate readiness and (b) capture raw bytes.
* :class:`ur_env.server_latency.ServerLatencyProbe` -- the real probe,
  subclassed to stamp the proxy-only gauges on every RPC record.

THE RAW-CAPTURE SEAM, AND WHY IT IS THIS ONE
--------------------------------------------
The server dedups a retried request by fingerprinting the bytes it already
answered.  Anything that re-derives the request -- rebuilding it from
dataclasses, re-encoding an observation, re-stamping a timestamp -- risks a
different fingerprint, which turns a harmless retry into a rejected transition.
So the forwarder must send the bytes the actor produced, unchanged.

Two seams could produce them:

(i)  register the ``BeginEpisode``/``Step`` method handlers with
     ``request_deserializer=None``.  grpc then hands the behaviour the payload
     exactly as it arrived; this module parses it itself and delegates the
     parsed message to the unmodified servicer.  These ARE the wire bytes, by
     construction, and it is the exact mirror of the uploader's
     ``request_serializer=None`` on the sending side.
(ii) let the generated deserializer run and call ``SerializeToString()`` on the
     parsed message at the servicer seam.

(i) is what this module does.  (ii) round-trips byte-identically in practice
for these messages -- same protobuf runtime, no map fields, ascending field
order -- but "in practice" is the whole objection: it is a property of the
runtime, not of the contract, and it is silently false for a message carrying
unknown fields (a half-upgraded peer), which is precisely the situation where a
fingerprint mismatch would be hardest to diagnose.  ``tests/test_local_policy``
``_proxy.py`` asserts the round trip holds anyway, so a future maintainer can
see which property they would be leaning on if they switched.

The cost of (i) is that ``create_grpc_server`` cannot be reused verbatim: it
calls the generated ``add_ActorTransportServicer_to_server``, which pins the
deserializers.  :func:`create_proxy_grpc_server` therefore builds the same
server with the same options and registers the same five methods, two of them
raw.  ``ur_env/grpc_actor_transport.py`` is not modified, and a test asserts
this module's method set still equals the generated stub's.

WHAT THE PROXY DOES NOT DO
--------------------------
It stores nothing.  ``accept_data`` is :func:`discard_transition`, a no-op:
replay, the classifier and training all live on the server, and the bytes that
feed them are already in the uploader's queue by the time the actor gets its
reply.  Deliberately, that no-op does NOT expose ``prime_observation``, so
``ActorSessionService`` takes its pixel path and the local policy runs the
frozen trunk itself -- the shared-feature optimisation exists to stop the
SERVER encoding the same image twice (policy + replay), and here there is no
second consumer to share with.

It also has no reward model.  ``reward_authority`` / ``reward_model_id`` in the
proxy's ``GetServerInfo`` are MIRRORED from the real server precisely because
the server really is the reward authority for everything that lands in replay;
only ``model_id`` is the proxy's own, so an operator's expected-model-id
preflight is proof of which mode they are running.  (There is no threshold
field in ``ServerInfoReply`` to mirror -- the reward threshold reaches the actor
only inside a per-transition ``TransitionOutcome``, and this proxy's outcomes
are all unevaluated by construction.  See ``manual_finalize``.)

MANUAL ONLY
-----------
``ManualTransitionFinalizer`` raises on ``meta.auto_success``; that propagates
through ``ActorSessionService.step``, which faults the service permanently and
fails the RPC.  The proxy cannot evaluate the classifier and must never quietly
under-claim success authority.

IMPORTS
-------
grpc + stdlib + this repo.  No jax at import time (the runtime module is
imported by the entrypoint, and it keeps jax inside functions), so this module
loads in the actor venv and its whole contract is testable there with an
injected ``sample_action``.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, replace
import logging
import os
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import grpc

from ur_env.actor_network import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ActorSessionService,
    ActorTransportError,
    BufferStatus,
    ServerInfo,
)
from ur_env.grpc_actor_transport import (
    DEFAULT_MAX_MESSAGE_BYTES,
    GrpcActorNetwork,
    GrpcActorServicer,
)
from ur_env.latency_profile import LatencyProfiler
from ur_env.local_policy.manual_finalize import ManualTransitionFinalizer
from ur_env.local_policy.uploader import (
    BEGIN_EPISODE_METHOD,
    DEFAULT_LATENCY_PROFILE_DIR,
    DEFAULT_TARGET as DEFAULT_REMOTE_TARGET,
    STEP_METHOD,
    TransitionUploader,
)
from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH
from ur_env.proto import actor_transport_pb2 as pb
from ur_env.server_latency import ServerLatencyProbe

__all__ = [
    "DEFAULT_BIND_HOST",
    "DEFAULT_PROXY_PORT",
    "DEFAULT_REMOTE_TARGET",
    "GATE_PARAMS_LOADED",
    "GATE_REMOTE_VERIFIED",
    "GATE_SMOKE_PASSED",
    "LATENCY_ROLE",
    "PROXY_PORT_ENV_VAR",
    "SERVICE_NAME",
    "LocalPolicyProxy",
    "LocalPolicyProxyServicer",
    "ProxyError",
    "ProxyLatencyProbe",
    "ProxyReadiness",
    "RemoteBufferStatusMirror",
    "RemoteContract",
    "RemoteHandshakeError",
    "RemoteLink",
    "ValidatingParamsSwap",
    "compose_health_detail",
    "create_proxy_grpc_server",
    "discard_transition",
    "local_outcome_summary",
    "proxy_method_handlers",
    "resolve_proxy_port",
]

_LOGGER = logging.getLogger(__name__)

#: The proxy's own listener.  Loopback only, and a different port from the
#: tunnel-local end of the ssh forward to the real server (50153) so both can be
#: up at once and an operator can tell from ``ss -ltnp`` which is which.
DEFAULT_PROXY_PORT = 50253
PROXY_PORT_ENV_VAR = "HIL_LOCAL_POLICY_PORT"
DEFAULT_BIND_HOST = "127.0.0.1"

#: ``role`` written into every latency record from the proxy's RPC handlers.
LATENCY_ROLE = "proxy"

#: Derived from the method paths rather than written out again: the uploader
#: dials these exact strings and a test pins them against the generated stub, so
#: there is exactly one place in this package where the service name lives.
SERVICE_NAME = BEGIN_EPISODE_METHOD.rsplit("/", 1)[0].lstrip("/")

_BEGIN_EPISODE_METHOD_NAME = BEGIN_EPISODE_METHOD.rsplit("/", 1)[-1]
_STEP_METHOD_NAME = STEP_METHOD.rsplit("/", 1)[-1]

#: Health gates.  All three must be satisfied before the proxy reports SERVING;
#: the names appear verbatim in the ``detail`` string an operator reads.
GATE_REMOTE_VERIFIED = "remote_verified"
GATE_PARAMS_LOADED = "params_loaded"
GATE_SMOKE_PASSED = "smoke_passed"

#: How long a mirrored buffer status may be reused before a refresh is kicked
#: off in the background.  Nothing on the control path reads it.
DEFAULT_BUFFER_STATUS_REFRESH_S = 2.0

#: Identity cache for BeginEpisode forwarding.  Sized for "a handful of
#: episodes", not for a session: it only has to outlive one actor retry.
_BEGIN_FORWARD_CACHE = 64


class ProxyError(RuntimeError):
    """The local policy proxy cannot be assembled or cannot serve."""


class RemoteHandshakeError(ProxyError):
    """The real server is unreachable or incompatible with this proxy."""


def resolve_proxy_port(
    value: Optional[int] = None, *, env: Optional[Mapping[str, str]] = None
) -> int:
    """Explicit value > ``HIL_LOCAL_POLICY_PORT`` > :data:`DEFAULT_PROXY_PORT`.

    A malformed env value RAISES rather than falling back: unlike a poll
    interval, a port an operator believes they set and did not is a proxy the
    actor cannot find, and the failure would surface as a connection refusal
    far from its cause.  ``0`` is allowed and means "let the kernel choose",
    which is what the tests bind with.
    """

    if value is not None:
        port = int(value)
    else:
        source = os.environ if env is None else env
        text = str(source.get(PROXY_PORT_ENV_VAR, "") or "").strip()
        if not text:
            return DEFAULT_PROXY_PORT
        try:
            port = int(text)
        except ValueError as exc:
            raise ProxyError(
                f"{PROXY_PORT_ENV_VAR}={text!r} is not an integer"
            ) from exc
    if not 0 <= port < 65536:
        raise ProxyError(f"proxy port {port} is out of range")
    return port


# --------------------------------------------------------------------------- #
# The sink: the proxy stores nothing                                           #
# --------------------------------------------------------------------------- #


def discard_transition(data: Mapping[str, Any], intervened: bool) -> None:
    """``accept_data`` for a server that owns no replay.

    ``ActorSessionService`` requires SOME sink and its default is an in-memory
    list with a 256-entry capacity that raises ``BufferError`` when full -- a
    mock's behaviour, and on this path it would turn a long session into a
    failed Step RPC for no reason.  Discarding is the honest implementation:
    the transition's authoritative copy is the raw request bytes, which the
    caller has already handed to the uploader, and the buffer that matters is
    the server's.

    Deliberately a plain function with no ``prime_observation`` attribute; see
    this module's docstring.
    """

    del data, intervened


def local_outcome_summary(outcome: Any) -> dict[str, Any]:
    """The divergence-check summary for one locally finalized transition.

    STRICTLY the reward/terminal fields plus the id, and never the classifier
    telemetry.  Under MANUAL the local finalizer reproduces the server's branch
    exactly for these five values, so any difference is a real parity bug.  The
    classifier fields are EXPECTED to differ whenever the step carried a
    sidecar: the server scores it and reports ``classifier_evaluated=1``, while
    the proxy has no classifier and reports the unevaluated triple.  Including
    them here would make the alarm fire ~2x a second on a healthy session and
    train the operator to ignore it.
    """

    return {
        "transition_id": str(outcome.transition_id),
        "done": bool(outcome.done),
        "truncated": bool(outcome.truncated),
        "success": bool(outcome.success),
        "reward": float(outcome.reward),
        "mask": float(outcome.mask),
    }


# --------------------------------------------------------------------------- #
# Readiness gate                                                               #
# --------------------------------------------------------------------------- #


class ProxyReadiness:
    """SERVING iff every named gate is satisfied and nothing has faulted.

    ``ActorSessionService`` starts ready and only ever leaves that state on a
    fault, which is right for a server whose collaborators were all built
    before it.  The proxy's are not: parameters arrive over ssh and the remote
    contract over gRPC, both after the process exists.  This is the AND term
    that keeps Health honest in between, and the reason the gates are named is
    that "not ready" without "waiting for what" is an operator's dead end.
    """

    def __init__(self, gates: Iterable[str] = ()) -> None:
        names = tuple(str(name) for name in gates)
        if not names:
            names = (GATE_REMOTE_VERIFIED, GATE_PARAMS_LOADED, GATE_SMOKE_PASSED)
        self._lock = threading.Lock()
        self._pending = list(names)
        self._fault = ""
        self._notes: dict[str, str] = {}

    def satisfy(self, gate: str, note: str = "") -> None:
        """Mark one gate done.  Unknown gate names are a wiring bug and raise."""

        with self._lock:
            if gate not in self._pending and gate not in self._notes:
                raise ProxyError(f"unknown readiness gate {gate!r}")
            if gate in self._pending:
                self._pending.remove(gate)
            if note:
                self._notes[gate] = str(note)
            else:
                self._notes.setdefault(gate, "")

    def fault(self, detail: str) -> None:
        """Latch a permanent not-ready.  The first fault is the one reported."""

        with self._lock:
            if not self._fault:
                self._fault = str(detail) or "proxy fault"

    @property
    def ready(self) -> bool:
        with self._lock:
            return not self._pending and not self._fault

    @property
    def pending(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._pending)

    @property
    def notes_text(self) -> str:
        """``gate=note`` for every gate that recorded one; ``""`` if none.

        These are the boot facts an operator wants on the Health line -- which
        parameter version is serving, which server was verified -- and they are
        appended to, never substituted for, the service's own detail (which is
        where a degraded-classifier marker would appear).
        """

        with self._lock:
            return "; ".join(
                f"{gate}={note}"
                for gate, note in sorted(self._notes.items())
                if note
            )

    def state(self) -> tuple[bool, str]:
        """``(ready, detail)`` -- one consistent snapshot for a Health reply."""

        with self._lock:
            if self._fault:
                return False, f"local policy proxy FAULTED: {self._fault}"
            if self._pending:
                return False, (
                    "local policy proxy is starting; waiting for "
                    + ", ".join(self._pending)
                )
        return True, "ready"


def compose_health_detail(
    service_detail: str, readiness: ProxyReadiness
) -> tuple[bool, str]:
    """Merge the service's Health detail with the proxy's readiness gate.

    One function so the gRPC reply and :meth:`LocalPolicyProxy.health` can never
    tell an operator two different stories about the same process.
    """

    gate_ready, gate_detail = readiness.state()
    if not gate_ready:
        return False, f"{gate_detail}; service: {service_detail}"
    notes = readiness.notes_text
    return True, (f"{service_detail} [{notes}]" if notes else service_detail)


# --------------------------------------------------------------------------- #
# The real server, as seen from the proxy                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RemoteContract:
    """What the real server told the proxy about itself at startup."""

    target: str
    model_id: str
    reward_authority: str
    reward_model_id: str
    observation_schema_hash: str
    protocol_version: str = PROTOCOL_VERSION
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_server_info(cls, info: ServerInfo, *, target: str) -> "RemoteContract":
        return cls(
            target=str(target),
            model_id=info.model_id,
            reward_authority=info.reward_authority,
            reward_model_id=info.reward_model_id,
            observation_schema_hash=info.observation_schema_hash,
            protocol_version=info.protocol_version,
            schema_version=int(info.schema_version),
        )


class RemoteLink:
    """The proxy's read-only client for the real server.

    Reuses :class:`~ur_env.grpc_actor_transport.GrpcActorNetwork` rather than a
    second client: everything the handshake has to check -- protocol version,
    schema version, action dim, observation schema hash, readiness -- is
    already implemented there and is exactly what the ACTOR would have checked
    in remote mode.  Doing it again by hand would be a second opinion about
    compatibility, which is how two peers end up disagreeing quietly.

    Only the stateless RPCs are used (``health`` / ``get_server_info`` /
    ``get_buffer_status``); ``begin_episode``/``step`` are never called, so none
    of the client's session state is ever touched and calls from the mirror's
    refresh thread cannot interleave with anything.
    """

    def __init__(
        self,
        target: str = DEFAULT_REMOTE_TARGET,
        *,
        actor_id: str = "hil-local-policy-proxy",
        action_shape: Tuple[int, ...] = (7,),
        timeout_s: float = 10.0,
        # Not latency critical and not on the control path: a handshake that
        # took 900 ms is fine, and rejecting it would only make the proxy
        # refuse to start on a busy server.
        max_response_age_s: float = 60.0,
        expected_observation_schema_hash: Optional[str] = (
            CANONICAL_OBSERVATION_SCHEMA_HASH
        ),
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        network: Any = None,
    ) -> None:
        self._target = str(target)
        self._owns_network = network is None
        self._network = network or GrpcActorNetwork(
            self._target,
            actor_id=actor_id,
            action_shape=tuple(int(dim) for dim in action_shape),
            timeout_s=float(timeout_s),
            max_response_age_s=float(max_response_age_s),
            max_message_bytes=int(max_message_bytes),
            expected_observation_schema_hash=expected_observation_schema_hash,
        )
        self._lock = threading.Lock()

    @property
    def target(self) -> str:
        return self._target

    def verify(self) -> RemoteContract:
        """Health + GetServerInfo, converted into one contract or one error.

        Every failure becomes :class:`RemoteHandshakeError` so the entrypoint
        has a single thing to catch and one exit code to report; the original
        message is preserved, because "which check failed" is the whole content
        of a handshake failure.
        """

        with self._lock:
            try:
                alive, ready, detail = self._network.health()
            except Exception as exc:
                raise RemoteHandshakeError(
                    f"cannot reach the real server at {self._target}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not alive or not ready:
                raise RemoteHandshakeError(
                    f"real server at {self._target} is not ready "
                    f"(alive={alive} ready={ready}): {detail}"
                )
            try:
                info = self._network.get_server_info()
            except Exception as exc:
                raise RemoteHandshakeError(
                    f"real server at {self._target} is incompatible with this "
                    f"proxy: {type(exc).__name__}: {exc}"
                ) from exc
        for name, value in (
            ("model_id", info.model_id),
            ("reward_authority", info.reward_authority),
            ("observation_schema_hash", info.observation_schema_hash),
        ):
            if not isinstance(value, str) or not value:
                raise RemoteHandshakeError(f"real server {name} is empty")
        return RemoteContract.from_server_info(info, target=self._target)

    def buffer_status(self) -> BufferStatus:
        with self._lock:
            return self._network.get_buffer_status()

    def close(self) -> None:
        if not self._owns_network:
            return
        try:
            self._network.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass


class RemoteBufferStatusMirror:
    """``GetBufferStatus`` on the proxy = the REAL server's buffer status.

    WHY PASS-THROUGH AND NOT A LOCAL STUB
    -------------------------------------
    The proxy holds no replay, so every number in a locally-invented
    ``BufferStatus`` would be a fabrication, and ``BufferStatusReply`` has no
    free-text field in which to mark it as one -- an all-zero stub is
    indistinguishable from a server whose buffers really are empty.  The
    question an operator asks this RPC ("how much has the learner ingested?")
    has exactly one true answer and it is on the other host.

    WHY IT NEVER BLOCKS
    -------------------
    ``ActorSessionService.get_buffer_status`` calls its provider while holding
    the service lock -- the same lock ``step`` needs.  A network round trip in
    there would let a diagnostic RPC from a second terminal stall the arm's
    control loop for as long as the remote takes to answer.  So this provider
    does no I/O: it returns the last snapshot and, when that snapshot is older
    than ``refresh_after_s``, kicks a single-flight background refresh.  The
    proxy primes it during the startup handshake, so a snapshot exists from
    boot.

    Staleness is not reported and does not need to be: in local-inference mode
    the server's counters lag the actor by the upload backlog anyway, which is
    a bigger and more interesting lag than this cache's.  What IS reported is
    the absence of any snapshot at all -- that raises, so "the proxy cannot
    reach the real server" is never mistaken for "the buffers are empty".
    """

    def __init__(
        self,
        link: Any,
        *,
        refresh_after_s: float = DEFAULT_BUFFER_STATUS_REFRESH_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._link = link
        self._refresh_after_s = float(refresh_after_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._status: Optional[BufferStatus] = None
        self._fetched_at = 0.0
        self._refreshing = False
        self._last_error = ""
        self.refresh_count = 0

    def refresh(self) -> BufferStatus:
        """Query the remote now, on the caller's thread.  Raises on failure."""

        status = self._link.buffer_status()
        with self._lock:
            self._status = status
            self._fetched_at = self._clock()
            self._last_error = ""
            self.refresh_count += 1
        return status

    def __call__(self) -> BufferStatus:
        with self._lock:
            status = self._status
            age = self._clock() - self._fetched_at
            stale = status is None or age >= self._refresh_after_s
            start_refresh = stale and not self._refreshing
            if start_refresh:
                self._refreshing = True
            error = self._last_error
        if start_refresh:
            threading.Thread(
                target=self._refresh_in_background,
                name="hil-proxy-buffer-status",
                daemon=True,
            ).start()
        if status is None:
            raise ActorTransportError(
                "the local policy proxy holds no replay of its own and has no "
                f"buffer status from {getattr(self._link, 'target', 'the real server')} "
                "yet; a refresh has been scheduled"
                + (f" (last error: {error})" if error else "")
            )
        return status

    def _refresh_in_background(self) -> None:
        try:
            self.refresh()
        except Exception as exc:  # noqa: BLE001 - a diagnostic never raises here
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            _LOGGER.warning(
                "proxy could not refresh the real server's buffer status: %s",
                exc,
            )
        finally:
            with self._lock:
                self._refreshing = False


# --------------------------------------------------------------------------- #
# Parameters                                                                   #
# --------------------------------------------------------------------------- #


class ValidatingParamsSwap:
    """``swap_callback`` for ``ParamsSyncClient``: validate, THEN publish.

    ``ParamsSyncClient`` deserializes and tree-checks a blob before it gets
    here; what this adds is the check only the serving process can make -- the
    candidate is smoked on the real agent (both traces) by
    ``LocalPolicyRuntime.validate_candidate`` before the inference path can
    ever see it.  A raise leaves the applied version unchanged, so the current
    parameters keep serving and the next poll retries the same blob.

    The validator is installed AFTER construction on purpose.  The very first
    blob has to land in the holder before ``LocalPolicyRuntime`` can exist at
    all (its constructor reads the holder), and that bootstrap blob is then
    validated and smoked by that same constructor -- so no parameters ever
    serve unvalidated, and there is no chicken-and-egg dance in the entrypoint.
    """

    def __init__(
        self, holder: Any, validator: Optional[Callable[[Any], None]] = None
    ) -> None:
        if not callable(getattr(holder, "swap", None)):
            raise TypeError("holder must expose swap(params, version)")
        self._holder = holder
        self._lock = threading.Lock()
        self._validator = validator
        self.validated_count = 0

    def set_validator(self, validator: Optional[Callable[[Any], None]]) -> None:
        if validator is not None and not callable(validator):
            raise TypeError("validator must be callable")
        with self._lock:
            self._validator = validator

    @property
    def validator(self) -> Optional[Callable[[Any], None]]:
        with self._lock:
            return self._validator

    def __call__(self, params: Any, version: int) -> None:
        validator = self.validator
        if validator is not None:
            validator(params)
            with self._lock:
                self.validated_count += 1
        self._holder.swap(params, version)


# --------------------------------------------------------------------------- #
# Latency                                                                      #
# --------------------------------------------------------------------------- #


class _GaugeScope:
    """Wrap one RPC scope so the proxy's gauges land on its record."""

    __slots__ = ("_inner", "_gauges", "_record")

    def __init__(self, inner: Any, gauges: Callable[[], Mapping[str, Any]]) -> None:
        self._inner = inner
        self._gauges = gauges
        self._record: Any = None

    def __enter__(self) -> Any:
        record = self._inner.__enter__()
        self._record = record
        try:
            if getattr(record, "enabled", False):
                for key, value in self._gauges().items():
                    record.set(key, value)
        except Exception:  # noqa: BLE001 - profiling never fails a handler
            pass
        return record

    def __exit__(self, exc_type, exc, tb) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)


class ProxyLatencyProbe(ServerLatencyProbe):
    """``ServerLatencyProbe`` with role ``proxy`` and the proxy-only gauges.

    The base class already gives ``total_ms`` (its ``total`` phase),
    ``request_decode_ms`` / ``service_step_ms`` / ``response_build_ms`` (the
    servicer's phases), and it is the object
    :class:`~ur_env.local_policy.runtime.LocalPolicyRuntime` and
    :class:`~ur_env.local_policy.manual_finalize.ManualTransitionFinalizer`
    write ``local_inference_ms`` / ``local_finalize_ms`` into -- they call
    ``phase()``, which finds this thread's in-flight record.

    What it cannot give is anything the RPC handler does not know about: the
    forward-queue depth, and (for a Step whose inference raised before it could
    report them) the parameter version and age.  Those are read from callables
    at scope entry.  ``params_version``/``params_age_s`` are stamped here as a
    floor and OVERWRITTEN by the runtime's own per-call values when inference
    succeeds -- last write wins, and the runtime's is the authoritative one
    because it describes the exact snapshot that produced the action.
    """

    def __init__(
        self,
        profiler: Any,
        *,
        gauges: Optional[Callable[[], Mapping[str, Any]]] = None,
    ) -> None:
        super().__init__(profiler)
        self._gauges = gauges

    @classmethod
    def from_env(
        cls,
        out_path: Optional[Any] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
        gauges: Optional[Callable[[], Mapping[str, Any]]] = None,
    ) -> "ProxyLatencyProbe":
        """Enabled iff ``HIL_LATENCY_PROFILE`` is truthy in ``env``."""

        return cls(
            LatencyProfiler.from_env(
                LATENCY_ROLE,
                DEFAULT_LATENCY_PROFILE_DIR if out_path is None else out_path,
                env=env,
            ),
            gauges=gauges,
        )

    def set_gauges(self, gauges: Optional[Callable[[], Mapping[str, Any]]]) -> None:
        """Install the gauge source once the objects it reads exist."""

        self._gauges = gauges

    def rpc(self, rpc: str) -> Any:
        scope = super().rpc(rpc)
        if not self.enabled or self._gauges is None:
            return scope
        return _GaugeScope(scope, self._gauges)


# --------------------------------------------------------------------------- #
# The servicer                                                                 #
# --------------------------------------------------------------------------- #


class LocalPolicyProxyServicer(GrpcActorServicer):
    """The production servicer, plus readiness gating and raw-bytes capture.

    Everything that answers an actor is inherited.  This class adds exactly
    three behaviours and nothing else:

    * ``Health``/``GetServerInfo`` are ANDed with the readiness gate, so the
      proxy cannot advertise SERVING before the remote is verified, the first
      parameters are loaded and both policy traces are compiled.
    * ``BeginEpisodeRaw``/``StepRaw`` take the request as raw bytes, parse it,
      delegate to the inherited handler, and -- only if that handler produced a
      reply -- hand those same bytes to the uploader.
    * A forwarding lock keeps enqueue order equal to handling order even if the
      gRPC pool ever runs two handlers at once.

    ORDER OF OPERATIONS, AND WHY
    ----------------------------
    The bytes are enqueued AFTER the local handler returns, never before.
    Before would forward transitions the local service rejected (which the
    server would then accept, so replay would contain a step the actor never
    completed) and would have no local outcome to compare against.  A handler
    that aborts therefore forwards nothing, which is the correct direction:
    the actor sees the failure immediately and the two sides stay in agreement.

    A DEDUPLICATED reply is not forwarded again either.  The actor's one
    transient retry re-sends bytes the proxy already queued; forwarding them
    twice would be harmless at the server (identical bytes, identical
    fingerprint, dedup) but would make every backlog and upload count wrong.
    """

    def __init__(
        self,
        service: ActorSessionService,
        *,
        uploader: Any,
        readiness: Optional[ProxyReadiness] = None,
        latency_probe: Optional[Any] = None,
    ) -> None:
        super().__init__(service, latency_probe=latency_probe)
        if not callable(getattr(uploader, "enqueue_step", None)):
            raise TypeError("uploader must expose enqueue_step/enqueue_begin_episode")
        self._uploader = uploader
        self._readiness = readiness if readiness is not None else ProxyReadiness()
        self._forward_lock = threading.Lock()
        self._begin_forwarded: "collections.OrderedDict[tuple, None]" = (
            collections.OrderedDict()
        )
        self.forwarded_begin_count = 0
        self.forwarded_step_count = 0
        self.skipped_duplicate_count = 0
        self.forward_failure_count = 0

    # -- readiness ---------------------------------------------------------- #

    @property
    def readiness(self) -> ProxyReadiness:
        return self._readiness

    def Health(self, request, context):
        reply = super().Health(request, context)
        gate_ready, detail = compose_health_detail(reply.detail, self._readiness)
        return pb.HealthReply(
            alive=reply.alive,
            ready=bool(reply.ready and gate_ready),
            detail=detail,
        )

    def GetServerInfo(self, request, context):
        reply = super().GetServerInfo(request, context)
        if not self._readiness.ready:
            # The actor's client turns ready=false into FailedPreconditionError
            # with its own message, so nothing is lost by not having a detail
            # field here.
            reply.ready = False
        return reply

    # -- raw capture -------------------------------------------------------- #

    def BeginEpisodeRaw(self, raw: bytes, context):
        request = self._parse(raw, pb.BeginEpisodeRequest, "BeginEpisodeRequest", context)
        with self._forward_lock:
            reply = super().BeginEpisode(request, context)
            if reply is not None and reply.ok:
                key = (
                    request.actor_id,
                    request.session_id,
                    int(request.request_id),
                )
                if key in self._begin_forwarded:
                    self.skipped_duplicate_count += 1
                else:
                    self._remember_begin(key)
                    self._forward(
                        context,
                        self._uploader.enqueue_begin_episode,
                        raw,
                        None,
                    )
                    self.forwarded_begin_count += 1
        return reply

    def StepRaw(self, raw: bytes, context):
        request = self._parse(raw, pb.StepRequest, "StepRequest", context)
        with self._forward_lock:
            reply = super().Step(request, context)
            if reply is None:
                return reply
            if not reply.ack.accepted:
                # The inherited handler aborts rather than returning a rejected
                # ACK, so this is unreachable today; forwarding a transition the
                # local service refused would be the one unrecoverable mistake
                # here, so it is checked rather than assumed.
                return reply
            if reply.ack.deduplicated:
                self.skipped_duplicate_count += 1
                return reply
            summary = (
                local_outcome_summary(reply.outcome)
                if reply.HasField("outcome")
                else None
            )
            self._forward(context, self._uploader.enqueue_step, raw, summary)
            self.forwarded_step_count += 1
        return reply

    # -- internals ---------------------------------------------------------- #

    def _parse(self, raw: Any, message_type: Any, name: str, context) -> Any:
        if isinstance(raw, (bytearray, memoryview)):
            raw = bytes(raw)
        if not isinstance(raw, bytes):
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"{name} handler expected raw bytes, got {type(raw).__name__}; "
                "the proxy's method handler is misregistered",
            )
        try:
            return message_type.FromString(raw)
        except Exception as exc:  # noqa: BLE001 - a bad frame is the peer's fault
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"could not parse {name}: {type(exc).__name__}: {exc}",
            )

    def _remember_begin(self, key: tuple) -> None:
        self._begin_forwarded[key] = None
        while len(self._begin_forwarded) > _BEGIN_FORWARD_CACHE:
            self._begin_forwarded.popitem(last=False)

    def _forward(
        self,
        context: Any,
        enqueue: Callable[..., int],
        raw: bytes,
        summary: Optional[Mapping[str, Any]],
    ) -> None:
        """Hand the bytes to the uploader, or fail the RPC saying so.

        ``enqueue`` appends to a deque; the only way it raises is a wiring bug
        (wrong type).  That is still worth aborting for: the local service has
        already accepted this transition, so a swallowed exception here would
        mean a transition that exists on the robot's timeline and nowhere else.
        Failing the RPC tells the operator immediately.
        """

        try:
            enqueue(raw, local_outcome_summary=summary)
        except Exception as exc:  # noqa: BLE001
            self.forward_failure_count += 1
            _LOGGER.exception("the proxy could not queue a request for upload")
            context.abort(
                grpc.StatusCode.INTERNAL,
                "the local policy proxy accepted this transition but could not "
                f"queue it for the real server: {type(exc).__name__}: {exc}",
            )


# --------------------------------------------------------------------------- #
# Server wiring                                                                #
# --------------------------------------------------------------------------- #


def proxy_method_handlers(servicer: LocalPolicyProxyServicer) -> dict:
    """The five ActorTransport methods, two of them with raw requests.

    ``request_deserializer=None`` is the whole point for ``BeginEpisode`` and
    ``Step``: grpc hands the behaviour the payload exactly as it arrived, which
    is what the uploader must replay.  The other three use the generated
    codecs, because nothing forwards them.

    This dict is the reason ``create_grpc_server`` is not reused; a test pins
    its keys against the generated stub so a new RPC cannot appear in the proto
    and silently go unserved by the proxy.
    """

    return {
        "Health": grpc.unary_unary_rpc_method_handler(
            servicer.Health,
            request_deserializer=pb.HealthRequest.FromString,
            response_serializer=pb.HealthReply.SerializeToString,
        ),
        "GetServerInfo": grpc.unary_unary_rpc_method_handler(
            servicer.GetServerInfo,
            request_deserializer=pb.ServerInfoRequest.FromString,
            response_serializer=pb.ServerInfoReply.SerializeToString,
        ),
        "GetBufferStatus": grpc.unary_unary_rpc_method_handler(
            servicer.GetBufferStatus,
            request_deserializer=pb.BufferStatusRequest.FromString,
            response_serializer=pb.BufferStatusReply.SerializeToString,
        ),
        _BEGIN_EPISODE_METHOD_NAME: grpc.unary_unary_rpc_method_handler(
            servicer.BeginEpisodeRaw,
            request_deserializer=None,
            response_serializer=pb.ActionReply.SerializeToString,
        ),
        _STEP_METHOD_NAME: grpc.unary_unary_rpc_method_handler(
            servicer.StepRaw,
            request_deserializer=None,
            response_serializer=pb.StepReply.SerializeToString,
        ),
    }


def create_proxy_grpc_server(
    servicer: LocalPolicyProxyServicer,
    *,
    bind_address: str = "127.0.0.1:0",
    max_workers: int = 4,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> tuple[Any, int]:
    """Create (but do not start) the proxy's gRPC server and return its port.

    Deliberately the same shape, thread-pool size and message limits as
    ``ur_env.grpc_actor_transport.create_grpc_server``; only the handler
    registration differs.
    """

    from concurrent import futures

    if max_workers <= 0 or max_message_bytes <= 0:
        raise ValueError("server limits must be positive")
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="hil-local-proxy"
        ),
        options=(
            ("grpc.max_receive_message_length", int(max_message_bytes)),
            ("grpc.max_send_message_length", int(max_message_bytes)),
        ),
    )
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                SERVICE_NAME, proxy_method_handlers(servicer)
            ),
        )
    )
    port = server.add_insecure_port(bind_address)
    if port == 0:
        raise ActorTransportError(
            f"failed to bind the local policy proxy at {bind_address}"
        )
    return server, port


# --------------------------------------------------------------------------- #
# The assembly                                                                 #
# --------------------------------------------------------------------------- #


class LocalPolicyProxy:
    """Everything the proxy is, as one startable/stoppable object.

    Construction binds the port (so ``port`` is known before ``start``) but
    starts nothing.  ``start()`` brings up the uploader, the parameter sync
    thread (when one was supplied) and the gRPC listener, in that order: the
    two background paths must be running before an actor can connect, or the
    first Step would queue against a stopped uploader.

    ``stop()`` reverses it and DRAINS: the gRPC server goes down first so no
    new transition can arrive, then the uploader is given a bounded chance to
    empty its queue.  Its return value is the shutdown's honesty -- ``False``
    means transitions are still buffered and the run's replay is incomplete.
    """

    def __init__(
        self,
        sample_action: Callable[..., tuple[Any, int]],
        *,
        remote: RemoteContract,
        uploader: Optional[Any] = None,
        finalize_transition: Optional[Callable[..., Any]] = None,
        readiness: Optional[ProxyReadiness] = None,
        latency_probe: Optional[Any] = None,
        buffer_status_provider: Optional[Callable[[], BufferStatus]] = None,
        params_sync: Optional[Any] = None,
        params_holder: Optional[Any] = None,
        model_id: Optional[str] = None,
        action_shape: Tuple[int, ...] = (7,),
        host: str = DEFAULT_BIND_HOST,
        port: int = DEFAULT_PROXY_PORT,
        bind_address: Optional[str] = None,
        max_workers: int = 4,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        allowed_actor_ids: Optional[Tuple[str, ...]] = None,
        allowed_run_ids: Optional[Tuple[str, ...]] = None,
    ) -> None:
        if not isinstance(remote, RemoteContract):
            raise TypeError("remote must be a verified RemoteContract")
        if model_id is None:
            from ur_env.local_policy.runtime import LOCAL_POLICY_MODEL_ID

            model_id = LOCAL_POLICY_MODEL_ID
        self._remote = remote
        self._readiness = readiness if readiness is not None else ProxyReadiness()
        self._latency = latency_probe
        self._uploader = (
            uploader if uploader is not None else TransitionUploader(remote.target)
        )
        self._params_sync = params_sync
        self._params_holder = params_holder
        self._finalizer = (
            finalize_transition
            if finalize_transition is not None
            else ManualTransitionFinalizer(
                **({} if latency_probe is None else {"latency_probe": latency_probe})
            )
        )
        self._service = ActorSessionService(
            sample_action=sample_action,
            action_shape=tuple(int(dim) for dim in action_shape),
            model_id=str(model_id),
            # MIRRORED, not invented: the server owns reward authority for
            # everything that reaches replay, and an actor pinned to the
            # server's values must keep working unchanged in local mode.
            reward_authority=remote.reward_authority,
            reward_model_id=remote.reward_model_id,
            observation_schema_hash=remote.observation_schema_hash,
            accept_data=discard_transition,
            finalize_transition=self._finalizer,
            buffer_status_provider=buffer_status_provider,
            allowed_actor_ids=allowed_actor_ids,
            allowed_run_ids=allowed_run_ids,
        )
        self._servicer = LocalPolicyProxyServicer(
            self._service,
            uploader=self._uploader,
            readiness=self._readiness,
            latency_probe=latency_probe,
        )
        if bind_address is None:
            bind_address = (
                f"[{host}]:{int(port)}" if ":" in host else f"{host}:{int(port)}"
            )
        self._server, self._port = create_proxy_grpc_server(
            self._servicer,
            bind_address=bind_address,
            max_workers=max_workers,
            max_message_bytes=max_message_bytes,
        )
        if isinstance(latency_probe, ProxyLatencyProbe):
            latency_probe.set_gauges(self.latency_gauges)
        self._started = False
        self._stopped = False

    # -- introspection ------------------------------------------------------ #

    @property
    def port(self) -> int:
        return self._port

    @property
    def service(self) -> ActorSessionService:
        return self._service

    @property
    def servicer(self) -> LocalPolicyProxyServicer:
        return self._servicer

    @property
    def uploader(self) -> Any:
        return self._uploader

    @property
    def finalizer(self) -> Any:
        """The transition finalizer, for health counters and telemetry."""

        return self._finalizer

    @property
    def readiness(self) -> ProxyReadiness:
        return self._readiness

    @property
    def remote(self) -> RemoteContract:
        return self._remote

    def server_info(self) -> ServerInfo:
        """What ``GetServerInfo`` would answer right now (gate included)."""

        info = self._service.get_server_info()
        if info.ready and not self._readiness.ready:
            return replace(info, ready=False)
        return info

    def health(self) -> tuple[bool, bool, str]:
        alive, ready, detail = self._service.health()
        gate_ready, composed = compose_health_detail(detail, self._readiness)
        return alive, bool(ready and gate_ready), composed

    def latency_gauges(self) -> dict[str, Any]:
        """Gauges stamped on every profiled RPC record.  Never raises."""

        gauges: dict[str, Any] = {}
        try:
            gauges["queue_depth"] = int(self._uploader.backlog_depth)
        except Exception:  # noqa: BLE001
            pass
        holder = self._params_holder
        if holder is not None:
            try:
                params, version, applied = holder.current()
                del params
                gauges["params_version"] = int(version)
                gauges["params_age_s"] = round(
                    max(0.0, time.monotonic() - float(applied)), 3
                )
            except Exception:  # noqa: BLE001
                pass
        return gauges

    def metrics(self) -> dict[str, Any]:
        """One snapshot for a log line or an operator's status probe."""

        # Through health() so a status line and a Health RPC can never tell an
        # operator two different stories about the same process.
        _alive, ready, detail = self.health()
        metrics: dict[str, Any] = {
            "port": self._port,
            "ready": ready,
            "detail": detail,
            "remote_target": self._remote.target,
            "model_id": self._service.get_server_info().model_id,
            "inference_count": int(self._service.inference_count),
            "forwarded_begin": self._servicer.forwarded_begin_count,
            "forwarded_step": self._servicer.forwarded_step_count,
            "skipped_duplicate": self._servicer.skipped_duplicate_count,
        }
        try:
            metrics["uploader"] = self._uploader.metrics()
        except Exception:  # noqa: BLE001
            pass
        if self._params_sync is not None:
            try:
                metrics["params_version"] = int(self._params_sync.applied_version)
                # ``stats`` is a property on ParamsSyncClient; tolerate a
                # callable one so a future sync client is not a silent gap.
                stats = self._params_sync.stats
                metrics["params_sync"] = dict(stats() if callable(stats) else stats)
            except Exception:  # noqa: BLE001
                pass
        elif self._params_holder is not None:
            try:
                metrics["params_version"] = int(self._params_holder.version)
            except Exception:  # noqa: BLE001
                pass
        return metrics

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> int:
        """Start the background paths and the listener.  Returns the port."""

        if self._started:
            return self._port
        self._uploader.start()
        if self._params_sync is not None:
            self._params_sync.start()
        self._server.start()
        self._started = True
        return self._port

    def stop(
        self,
        *,
        drain: bool = True,
        grace_s: float = 2.0,
        drain_timeout_s: float = 30.0,
    ) -> bool:
        """Stop serving and (optionally) drain.  ``True`` iff nothing is left.

        Idempotent.  The gRPC server is stopped first so the queue stops
        growing while it is being emptied; a ``False`` return is the caller's
        cue that this shutdown lost nothing yet but is not clean.
        """

        if self._stopped:
            return not self._uploader.backlog_depth
        self._stopped = True
        try:
            self._server.stop(grace=float(grace_s)).wait(
                timeout=max(1.0, float(grace_s) + 5.0)
            )
        except Exception:  # noqa: BLE001 - teardown must not raise
            _LOGGER.exception("the proxy's gRPC server did not stop cleanly")
        if self._params_sync is not None:
            try:
                self._params_sync.stop()
            except Exception:  # noqa: BLE001
                _LOGGER.exception("the parameter sync client did not stop cleanly")
        drained = self._uploader.stop(drain=drain, timeout_s=float(drain_timeout_s))
        if self._latency is not None:
            try:
                self._latency.close()
            except Exception:  # noqa: BLE001
                pass
        return bool(drained)

    def __enter__(self) -> "LocalPolicyProxy":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop(drain=exc_type is None)
        return False
