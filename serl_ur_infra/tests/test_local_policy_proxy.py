"""End-to-end tests for the local-inference proxy: actor -> proxy -> server.

WHAT IS REAL HERE AND WHAT IS NOT
--------------------------------
Almost everything is real.  The client is this repo's own
:class:`~ur_env.grpc_actor_transport.GrpcActorNetwork`, driven the way
``remote_actor.py`` drives it.  The proxy is the production
:class:`~ur_env.local_policy.proxy.LocalPolicyProxy` over a real gRPC socket,
with the real :class:`~ur_env.local_policy.manual_finalize
.ManualTransitionFinalizer` and the real
:class:`~ur_env.local_policy.uploader.TransitionUploader`.  The "server" is this
repo's :class:`~ur_env.grpc_actor_transport.GrpcActorServicer` in front of a
real :class:`~ur_env.actor_network.ActorSessionService` on a second in-process
socket, with a request deserializer that records the bytes it was handed -- so
byte identity is asserted on what gRPC delivered, not on a re-serialization.

Exactly one thing is faked: ``sample_action``.  Injecting it is what keeps the
whole path testable in the actor venv, which has no jax; the real
:class:`~ur_env.local_policy.runtime.LocalPolicyRuntime` satisfies the same
contract and is covered by ``tests/test_local_policy_runtime.py``.  One
jax-gated test at the bottom swaps a real runtime in and repeats the episode.

*** A LOAD-BEARING CAVEAT ABOUT THE FAKE SERVER'S POLICY ***
``ActorSessionService._validate_data`` requires ``meta.policy_action`` and
``meta.policy_version`` to equal the action and version THAT SERVICE issued for
the previous observation.  A forwarded transition carries the PROXY's action, so
a real learner server -- which runs its own inference over the same forwarded
request -- accepts it only when both sides produce identical values.  Most tests
here therefore give the fake server the same trivial policy as the proxy;
``test_forwarded_step_is_rejected_when_the_remote_policy_disagrees`` pins what
happens when they differ, because that is what a real server will do and it must
not be discovered on a robot.

Run (actor venv, no jax -- everything except the last test)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_local_policy_proxy.py

Run the jax-gated one as well with ``/home/laptop3/venvs/hilserl/bin/python``.
"""

from __future__ import annotations

from concurrent import futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional

import grpc
import numpy as np
import pytest

_HERE = Path(os.path.abspath(__file__)).parent
_INFRA_ROOT = _HERE.parent
_REPO_ROOT = _INFRA_ROOT.parent
sys.path.insert(0, str(_INFRA_ROOT))

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ActorSessionService,
    ActorTransportError,
    FailedPreconditionError,
    TransitionOutcome,
)
from ur_env.classifier_sidecar import CLASSIFIER_SIDECAR_KEY  # noqa: E402
from ur_env.grpc_actor_transport import (  # noqa: E402
    DEFAULT_MAX_MESSAGE_BYTES,
    GrpcActorNetwork,
    GrpcActorServicer,
)
from ur_env.latency_profile import LatencyProfiler  # noqa: E402
from ur_env.local_policy.params_sync import (  # noqa: E402
    LATEST_FILENAME,
    LocalDirectoryFetcher,
    ParamsHolder,
    ParamsSyncClient,
    blob_filename,
)
from ur_env.local_policy.proxy import (  # noqa: E402
    GATE_PARAMS_LOADED,
    GATE_REMOTE_VERIFIED,
    GATE_SMOKE_PASSED,
    LATENCY_ROLE,
    SERVICE_NAME,
    LocalPolicyProxy,
    LocalPolicyProxyServicer,
    ProxyError,
    ProxyLatencyProbe,
    ProxyReadiness,
    RemoteBufferStatusMirror,
    RemoteContract,
    RemoteHandshakeError,
    RemoteLink,
    ValidatingParamsSwap,
    discard_transition,
    local_outcome_summary,
    proxy_method_handlers,
    resolve_proxy_port,
)
from ur_env.local_policy.uploader import (  # noqa: E402
    KIND_BEGIN_EPISODE,
    KIND_STEP,
    TransitionUploader,
)
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)
from ur_env.proto import actor_transport_pb2 as pb  # noqa: E402
from ur_env.proto import actor_transport_pb2_grpc as pb_grpc  # noqa: E402


ACTOR_ID = "laptop3-actor"
RUN_ID = "run-local-proxy"
SESSION_ID = "session-0"
REWARD_MODEL_ID = "fake-reward-model-v1"
REMOTE_MODEL_ID = "fake-learner-policy"

#: The proxy's fake policy answers this; the actor must see exactly it.
LOCAL_ACTION = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

_HAS_JAX = importlib.util.find_spec("jax") is not None
_needs_jax = pytest.mark.skipif(
    not _HAS_JAX, reason="needs a jax-capable venv (hilserl / gello-local-policy)"
)


# =========================================================================== #
# Observations, sidecars, policies                                            #
# =========================================================================== #


def _observation(value: int) -> dict:
    """Small but structurally real: nested state + two image tensors."""

    return {
        "state": np.full((1, 6), value / 100.0, dtype=np.float32),
        "cam1": np.full((1, 4, 4, 3), value % 256, dtype=np.uint8),
        "cam2": np.full((1, 4, 4, 3), (255 - value) % 256, dtype=np.uint8),
    }


def _sidecar() -> dict:
    """A structurally valid sidecar.  ``validate_sidecar`` never decodes."""

    return {
        "cam1_jpeg": np.frombuffer(b"\xff\xd8\xff\xe0jpeg-cam1", dtype=np.uint8),
        "cam2_jpeg": np.frombuffer(b"\xff\xd8\xff\xe0jpeg-cam2", dtype=np.uint8),
    }


class _FakePolicy:
    """``sample_action`` with a settable action/version, for both sides.

    Matches ``LocalPolicyRuntime.__call__``: ``(observation, deterministic) ->
    (action, policy_version)``.
    """

    def __init__(self, action: np.ndarray = LOCAL_ACTION, version: int = 0) -> None:
        self.action = np.asarray(action, dtype=np.float32).copy()
        self.version = int(version)
        self.calls = 0
        self.last_deterministic: Optional[bool] = None

    def __call__(self, observation, deterministic):
        del observation
        self.calls += 1
        self.last_deterministic = bool(deterministic)
        return self.action.copy(), self.version


class _HolderPolicy(_FakePolicy):
    """Reports the version currently in a ``ParamsHolder`` -- the real rule."""

    def __init__(self, holder: Any, action: np.ndarray = LOCAL_ACTION) -> None:
        super().__init__(action=action)
        self._holder = holder

    def __call__(self, observation, deterministic):
        params, version, _applied = self._holder.current()
        assert params is not None, "the proxy must not serve without parameters"
        self.version = int(version)
        return super().__call__(observation, deterministic)


class _ClassifyingFinalizer:
    """A fake server finalizer that DOES evaluate a classifier, MANUAL-style.

    Reproduces the fields ``RewardTransitionFinalizer`` writes for a scored step
    under MANUAL: the classifier verdict is recorded and reported, and it does
    NOT make the transition successful (only the operator token can).  That is
    the exact shape the pinned divergence rule exists for.
    """

    def __init__(self, probability: float = 0.9, threshold: float = 0.5) -> None:
        self.probability = float(probability)
        self.threshold = float(threshold)
        self.evaluated = 0

    def __call__(self, data, classifier_sidecar=None):
        meta = data["meta"]
        transition = data["transition"]
        operator_success = bool(meta.get("operator_success", False))
        evaluated = classifier_sidecar is not None
        if evaluated:
            self.evaluated += 1
        probability = self.probability if evaluated else 0.0
        threshold = self.threshold if evaluated else 0.0
        transition["rewards"] = 1.0 if operator_success else 0.0
        if operator_success:
            transition["masks"] = 0.0
            transition["dones"] = True
            transition["truncated"] = False
        transition["classifier_evaluated"] = np.uint8(evaluated)
        transition["classifier_probability"] = probability
        transition["classifier_threshold"] = threshold
        transition["classifier_success"] = np.uint8(
            evaluated and probability > threshold
        )
        transition["success"] = np.uint8(operator_success)
        transition["reward_model_id"] = REWARD_MODEL_ID if evaluated else ""
        return data, TransitionOutcome(
            transition_id=str(meta["transition_id"]),
            reward=float(transition["rewards"]),
            mask=float(transition["masks"]),
            done=bool(transition["dones"]),
            truncated=bool(transition["truncated"]),
            success=operator_success,
            classifier_evaluated=evaluated,
            classifier_probability=probability,
            classifier_threshold=threshold,
            reward_model_id=REWARD_MODEL_ID if evaluated else "",
        )


# =========================================================================== #
# The fake remote server                                                      #
# =========================================================================== #


class _CapturingRemote:
    """The repo's servicer + service, recording the raw bytes it receives."""

    def __init__(
        self,
        *,
        policy: Optional[_FakePolicy] = None,
        finalizer: Any = None,
        delay_s: float = 0.0,
        reward_model_id: str = REWARD_MODEL_ID,
        observation_schema_hash: str = CANONICAL_OBSERVATION_SCHEMA_HASH,
    ) -> None:
        self.policy = policy if policy is not None else _FakePolicy()
        self.delay_s = float(delay_s)
        self.lock = threading.Lock()
        self.received: list[tuple[str, bytes]] = []
        self.service = ActorSessionService(
            sample_action=self.policy,
            model_id=REMOTE_MODEL_ID,
            reward_authority="server_classifier",
            reward_model_id=reward_model_id,
            observation_schema_hash=observation_schema_hash,
            finalize_transition=finalizer,
        )
        self.servicer = GrpcActorServicer(self.service)
        self._server: Any = None
        self.port = 0

    # -- capture ------------------------------------------------------------ #

    def _recording(self, kind: str, parse: Callable[[bytes], Any]):
        def deserialize(raw: bytes):
            with self.lock:
                self.received.append((kind, bytes(raw)))
            if self.delay_s:
                time.sleep(self.delay_s)
            return parse(raw)

        return deserialize

    def payloads(self, kind: Optional[str] = None) -> list[bytes]:
        with self.lock:
            return [
                payload
                for item_kind, payload in self.received
                if kind is None or item_kind == kind
            ]

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> int:
        handlers = {
            "Health": grpc.unary_unary_rpc_method_handler(
                self.servicer.Health,
                request_deserializer=pb.HealthRequest.FromString,
                response_serializer=pb.HealthReply.SerializeToString,
            ),
            "GetServerInfo": grpc.unary_unary_rpc_method_handler(
                self.servicer.GetServerInfo,
                request_deserializer=pb.ServerInfoRequest.FromString,
                response_serializer=pb.ServerInfoReply.SerializeToString,
            ),
            "GetBufferStatus": grpc.unary_unary_rpc_method_handler(
                self.servicer.GetBufferStatus,
                request_deserializer=pb.BufferStatusRequest.FromString,
                response_serializer=pb.BufferStatusReply.SerializeToString,
            ),
            "BeginEpisode": grpc.unary_unary_rpc_method_handler(
                self.servicer.BeginEpisode,
                request_deserializer=self._recording(
                    KIND_BEGIN_EPISODE, pb.BeginEpisodeRequest.FromString
                ),
                response_serializer=pb.ActionReply.SerializeToString,
            ),
            "Step": grpc.unary_unary_rpc_method_handler(
                self.servicer.Step,
                request_deserializer=self._recording(
                    KIND_STEP, pb.StepRequest.FromString
                ),
                response_serializer=pb.StepReply.SerializeToString,
            ),
        }
        server = grpc.server(
            futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="fake-remote"
            ),
            options=(
                ("grpc.max_receive_message_length", DEFAULT_MAX_MESSAGE_BYTES),
                ("grpc.max_send_message_length", DEFAULT_MAX_MESSAGE_BYTES),
            ),
        )
        server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler(SERVICE_NAME, handlers),)
        )
        self.port = server.add_insecure_port("127.0.0.1:0")
        assert self.port, "failed to bind the fake remote"
        server.start()
        self._server = server
        return self.port

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop(grace=0).wait()
            self._server = None

    @property
    def target(self) -> str:
        return f"127.0.0.1:{self.port}"


# =========================================================================== #
# The proxy under test, and the actor that drives it                          #
# =========================================================================== #


class _RecordingUploader(TransitionUploader):
    """The real uploader, remembering what the proxy handed it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.captured: list[tuple[str, bytes, Optional[dict]]] = []
        self._capture_lock = threading.Lock()

    def enqueue(self, kind, payload, *, local_outcome_summary=None):
        with self._capture_lock:
            self.captured.append(
                (
                    kind,
                    bytes(payload),
                    dict(local_outcome_summary) if local_outcome_summary else None,
                )
            )
        return super().enqueue(
            kind, payload, local_outcome_summary=local_outcome_summary
        )

    def captured_payloads(self, kind: Optional[str] = None) -> list[bytes]:
        with self._capture_lock:
            return [
                payload
                for item_kind, payload, _ in self.captured
                if kind is None or item_kind == kind
            ]


def _uploader(target: str, **kwargs: Any) -> _RecordingUploader:
    kwargs.setdefault("backoff_initial_s", 0.01)
    kwargs.setdefault("backoff_max_s", 0.05)
    kwargs.setdefault("timeout_s", 5.0)
    # env={} so a developer shell with HIL_LATENCY_PROFILE=1 cannot turn these
    # tests into a file-writing suite.
    kwargs.setdefault("env", {})
    return _RecordingUploader(target, **kwargs)


def _ready_gate() -> ProxyReadiness:
    readiness = ProxyReadiness()
    for gate in (GATE_REMOTE_VERIFIED, GATE_PARAMS_LOADED, GATE_SMOKE_PASSED):
        readiness.satisfy(gate)
    return readiness


def _proxy(
    remote: _CapturingRemote,
    *,
    policy: Any = None,
    uploader: Any = None,
    readiness: Optional[ProxyReadiness] = None,
    **kwargs: Any,
) -> LocalPolicyProxy:
    """A proxy on an ephemeral port, ready unless the test says otherwise.

    The contract comes from a REAL handshake against the fake remote, so every
    test also exercises ``RemoteLink.verify``.
    """

    link = RemoteLink(remote.target)
    try:
        contract = link.verify()
    finally:
        link.close()
    return LocalPolicyProxy(
        policy if policy is not None else _FakePolicy(),
        remote=contract,
        uploader=uploader if uploader is not None else _uploader(remote.target),
        readiness=readiness if readiness is not None else _ready_gate(),
        port=0,
        **kwargs,
    )


def _client(port: int, **kwargs: Any) -> GrpcActorNetwork:
    kwargs.setdefault("timeout_s", 10.0)
    kwargs.setdefault("max_response_age_s", 60.0)
    return GrpcActorNetwork(f"127.0.0.1:{port}", actor_id=ACTOR_ID, **kwargs)


class _ActorDriver:
    """Drive one episode the way ``remote_actor.py`` drives a server.

    Threads every identity the service checks: the observation chain, the source
    observation's timestamp, ``env_step``/``step_id``, and the action and version
    the proxy last issued.
    """

    def __init__(
        self,
        network: GrpcActorNetwork,
        *,
        run_id: str = RUN_ID,
        session_id: str = SESSION_ID,
        episode_id: int = 0,
        observation_factory: Callable[[int], dict] = _observation,
    ) -> None:
        self.network = network
        self.run_id = run_id
        self.session_id = session_id
        self.episode_id = episode_id
        self.env_step = 0
        self._observation = observation_factory
        self._step_id = 0
        self._obs_seq = 0
        self._current_id = "o0"
        self._current_ts = 1_000
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_version = 0

    def begin(self):
        result = self.network.begin_episode(
            self._observation(self._obs_seq),
            run_id=self.run_id,
            session_id=self.session_id,
            episode_id=self.episode_id,
            observation_id=self._current_id,
            timestamp_ns=self._current_ts,
        )
        self.last_action = np.asarray(result.action, dtype=np.float32)
        self.last_version = int(result.policy_version)
        return result

    def step(
        self,
        *,
        operator_success: bool = False,
        auto_success: bool = False,
        done: bool = False,
        truncated: bool = False,
        intervened: bool = False,
        sidecar: bool = False,
    ):
        source_id = self._current_id
        source_ts = self._current_ts
        self._obs_seq += 1
        next_id = f"o{self._obs_seq}"
        next_ts = 1_000 + self._obs_seq
        terminal = bool(done) or bool(truncated)
        data = {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "actor_id": ACTOR_ID,
                "session_id": self.session_id,
                "transition_id": f"{self.run_id}:{self.env_step}",
                "env_step": self.env_step,
                "timestamp_ns": source_ts,
                "policy_version": self.last_version,
                "policy_action": self.last_action,
                "intervened": bool(intervened),
                "auto_success": bool(auto_success),
                "operator_success": bool(operator_success),
            },
            "transition": {
                "episode_id": self.episode_id,
                "step_id": self._step_id,
                "observation_id": source_id,
                "actions": self.last_action.copy(),
                "next_observation_id": next_id,
                "rewards": 0.0,
                "masks": 0.0 if done else 1.0,
                "dones": bool(done),
                "truncated": bool(truncated),
            },
        }
        next_observation = self._observation(self._obs_seq)
        if sidecar:
            next_observation[CLASSIFIER_SIDECAR_KEY] = _sidecar()
        result = self.network.step(
            next_observation,
            next_observation_id=next_id,
            next_timestamp_ns=next_ts,
            data=data,
            request_action=not terminal,
        )
        self._step_id += 1
        self.env_step += 1
        self._current_id = next_id
        self._current_ts = next_ts
        if result.action is not None:
            self.last_action = np.asarray(result.action.action, dtype=np.float32)
            self.last_version = int(result.action.policy_version)
        return result


def _drain(uploader: TransitionUploader, timeout_s: float = 20.0) -> None:
    assert uploader.wait_until_idle(timeout_s), (
        f"uploader did not drain: {uploader.metrics()}"
    )


# =========================================================================== #
# Wiring: the raw-capture seam                                                #
# =========================================================================== #


def _handlers() -> dict:
    """The handler dict, built without a live service (names/codecs only)."""

    return proxy_method_handlers(object.__new__(LocalPolicyProxyServicer))


def test_proxy_serves_every_method_the_generated_stub_dials():
    """A new RPC in the proto must not silently go unserved by the proxy."""

    class _PathProbe:
        def __init__(self) -> None:
            self.methods: set[str] = set()

        def unary_unary(self, method, **kwargs):
            del kwargs
            self.methods.add(method.rsplit("/", 1)[-1])
            return None

    probe = _PathProbe()
    pb_grpc.ActorTransportStub(probe)
    assert set(_handlers()) == probe.methods


def test_only_the_forwarded_methods_take_raw_requests():
    """``request_deserializer=None`` exactly where bytes must survive."""

    handlers = _handlers()
    raw = {
        name
        for name, handler in handlers.items()
        if handler.request_deserializer is None
    }
    assert raw == {"BeginEpisode", "Step"}
    for name, handler in handlers.items():
        assert handler.response_serializer is not None, name
        assert handler.unary_unary is not None, name


def test_step_request_reserialization_round_trips_byte_identically():
    """Documents the property the REJECTED seam (ii) would have leaned on.

    Parsing and re-serializing a representative StepRequest reproduces the
    original bytes in this protobuf runtime, with and without
    ``deterministic=True``.  The proxy still does not rely on it -- see the
    module docstring of ``ur_env/local_policy/proxy.py`` -- but a maintainer
    weighing the two seams should be able to read the measurement rather than
    re-derive it.
    """

    remote = _CapturingRemote()
    remote.start()
    proxy = _proxy(remote)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.step(sidecar=True)
        payloads = proxy.uploader.captured_payloads(KIND_STEP)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    assert payloads
    for raw in payloads:
        parsed = pb.StepRequest.FromString(raw)
        assert parsed.SerializeToString() == raw
        assert parsed.SerializeToString(deterministic=True) == raw


# =========================================================================== #
# The full episode                                                            #
# =========================================================================== #


def test_full_episode_local_actions_forwarded_bytes_no_divergence():
    """The headline path: local replies, remote ingest, zero divergence."""

    remote = _CapturingRemote()
    remote.start()
    policy = _FakePolicy()
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, policy=policy, uploader=uploader)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        begin = driver.begin()
        # The actor got the PROXY's action, not the remote's inference.
        np.testing.assert_array_equal(begin.action, LOCAL_ACTION)
        assert begin.policy_version == 0

        for _ in range(2):
            result = driver.step()
            assert result.ack.accepted
            assert not result.outcome.terminal
            assert result.outcome.reward == 0.0
            assert result.outcome.mask == 1.0
            assert result.outcome.classifier_evaluated is False
            np.testing.assert_array_equal(result.action.action, LOCAL_ACTION)

        # MANUAL MARK SUCCESS: the operator token is the only success path.
        final = driver.step(operator_success=True)
        assert final.action is None
        assert final.outcome.terminal
        assert final.outcome.success is True
        assert final.outcome.done is True
        assert final.outcome.truncated is False
        assert final.outcome.reward == 1.0
        assert final.outcome.mask == 0.0

        _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    # The proxy answered every action itself: begin + two non-terminal steps.
    assert policy.calls == 3

    # The server received the bytes the proxy captured, in order, unchanged.
    assert uploader.captured_payloads(KIND_BEGIN_EPISODE) == remote.payloads(
        KIND_BEGIN_EPISODE
    )
    assert uploader.captured_payloads(KIND_STEP) == remote.payloads(KIND_STEP)
    assert [
        pb.StepRequest.FromString(raw).request_id
        for raw in remote.payloads(KIND_STEP)
    ] == [2, 3, 4]

    # And it really ingested them.
    assert len(remote.service.replay_items) == 3
    assert uploader.uploaded_count == 4  # one BeginEpisode + three Steps
    assert uploader.divergence_count == 0
    assert uploader.rejected_count == 0
    assert proxy.servicer.forwarded_step_count == 3
    assert proxy.servicer.forwarded_begin_count == 1
    # Nothing was stored locally: the proxy is not a replay buffer.
    assert proxy.service.replay_items == []


def test_local_outcome_summary_is_strictly_reward_and_terminal_fields():
    """The pinned cross-agent rule, as a unit fact about the summary."""

    outcome = pb.TransitionOutcome(
        transition_id="t1",
        reward=1.0,
        mask=0.0,
        done=True,
        truncated=False,
        success=True,
        classifier_evaluated=True,
        classifier_probability=0.9,
        classifier_threshold=0.5,
        reward_model_id=REWARD_MODEL_ID,
    )
    summary = local_outcome_summary(outcome)
    assert set(summary) == {
        "transition_id",
        "done",
        "truncated",
        "success",
        "reward",
        "mask",
    }
    assert summary["reward"] == 1.0
    assert summary["success"] is True


def test_sidecar_step_is_not_a_divergence():
    """A scored step differs in classifier telemetry, and that is EXPECTED.

    The proxy has no classifier, so its outcome is unevaluated while the
    server's is evaluated.  Under MANUAL neither can move reward or the terminal
    flags, so the strict summary must report no divergence -- otherwise the
    alarm would fire twice a second on a perfectly healthy session.
    """

    finalizer = _ClassifyingFinalizer()
    remote = _CapturingRemote(finalizer=finalizer)
    remote.start()
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, uploader=uploader)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        scored = driver.step(sidecar=True)
        # The proxy told the actor "nobody scored this".
        assert scored.outcome.classifier_evaluated is False
        assert scored.outcome.classifier_probability == 0.0
        assert scored.outcome.reward_model_id == ""
        _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    # The server DID score it...
    assert finalizer.evaluated == 1
    stored = remote.service.replay_items[-1]["transition"]
    assert int(stored["classifier_evaluated"]) == 1
    assert float(stored["classifier_probability"]) == pytest.approx(0.9)
    # ...and that is not a divergence.
    assert uploader.divergence_count == 0
    assert proxy.finalizer.sidecar_ignored_count == 1


def test_reward_divergence_is_detected():
    """The alarm still fires for the fields that must match.

    The divergence is injected by replaying an already-delivered transition with
    a WRONG local summary: the server answers identically (dedup), so the only
    thing that changed is the local verdict -- which is exactly the shape of a
    finalizer-parity bug, and the only shape a real service will let us build
    (its own validator refuses to emit a success without an operator token).
    """

    remote = _CapturingRemote()
    remote.start()
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, uploader=uploader)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.step()
        _drain(uploader)
        kind, payload, summary = uploader.captured[-1]
        assert kind == KIND_STEP
        assert summary == {
            "transition_id": f"{RUN_ID}:0",
            "done": False,
            "truncated": False,
            "success": False,
            "reward": 0.0,
            "mask": 1.0,
        }
        uploader.enqueue_step(
            payload,
            local_outcome_summary={**summary, "success": True, "reward": 1.0},
        )
        _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    assert uploader.divergence_count == 1


# =========================================================================== #
# AUTO is refused                                                             #
# =========================================================================== #


def test_auto_success_faults_the_proxy_and_forwards_nothing():
    """MANUAL-only, fail-closed: the session ends rather than under-claiming."""

    remote = _CapturingRemote()
    remote.start()
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, uploader=uploader)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.step()
        with pytest.raises(ActorTransportError) as excinfo:
            driver.step(auto_success=True)
        assert "auto_success" in str(excinfo.value)

        # The service is permanently faulted; health says so.
        alive, ready, detail = proxy.health()
        assert alive is True
        assert ready is False
        assert "auto_success" in detail
        _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    # One BeginEpisode and ONE step were forwarded; the AUTO transition was not.
    assert proxy.servicer.forwarded_step_count == 1
    assert [
        pb.StepRequest.FromString(raw).request_id
        for raw in remote.payloads(KIND_STEP)
    ] == [2]
    assert len(remote.service.replay_items) == 1


# =========================================================================== #
# Health gating and ServerInfo mirroring                                      #
# =========================================================================== #


def test_health_is_not_serving_until_every_gate_is_satisfied():
    remote = _CapturingRemote()
    remote.start()
    readiness = ProxyReadiness()
    proxy = _proxy(remote, readiness=readiness)
    proxy.start()
    network = _client(proxy.port)
    try:
        alive, ready, detail = network.health()
        assert alive is True
        assert ready is False
        for gate in (GATE_REMOTE_VERIFIED, GATE_PARAMS_LOADED, GATE_SMOKE_PASSED):
            assert gate in detail
        # An actor that dialled now would be refused rather than served.
        with pytest.raises(FailedPreconditionError):
            network.get_server_info()

        readiness.satisfy(GATE_REMOTE_VERIFIED, remote.target)
        _alive, ready, detail = network.health()
        assert ready is False
        assert GATE_REMOTE_VERIFIED not in detail
        assert GATE_PARAMS_LOADED in detail

        readiness.satisfy(GATE_PARAMS_LOADED, "v7")
        readiness.satisfy(GATE_SMOKE_PASSED)
        _alive, ready, detail = network.health()
        assert ready is True
        assert "v7" in detail
        assert network.get_server_info().ready is True
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()


def test_server_info_mirrors_the_remote_reward_fields_but_not_its_model_id():
    remote = _CapturingRemote()
    remote.start()
    proxy = _proxy(remote)
    proxy.start()
    network = _client(proxy.port)
    try:
        info = network.get_server_info()
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    from ur_env.local_policy.runtime import LOCAL_POLICY_MODEL_ID

    assert info.model_id == LOCAL_POLICY_MODEL_ID
    assert info.model_id != REMOTE_MODEL_ID
    assert info.reward_authority == "server_classifier"
    assert info.reward_model_id == REWARD_MODEL_ID
    assert info.observation_schema_hash == CANONICAL_OBSERVATION_SCHEMA_HASH
    assert info.protocol_version == PROTOCOL_VERSION
    assert info.schema_version == SCHEMA_VERSION
    assert info.action_dim == 7


def test_readiness_rejects_an_unknown_gate():
    with pytest.raises(ProxyError):
        ProxyReadiness().satisfy("no-such-gate")


def test_readiness_fault_is_permanent_and_first_wins():
    readiness = _ready_gate()
    assert readiness.ready is True
    readiness.fault("first")
    readiness.fault("second")
    ready, detail = readiness.state()
    assert ready is False
    assert "first" in detail and "second" not in detail


# =========================================================================== #
# The handshake                                                               #
# =========================================================================== #


def test_handshake_reads_the_real_contract():
    remote = _CapturingRemote()
    remote.start()
    link = RemoteLink(remote.target)
    try:
        contract = link.verify()
    finally:
        link.close()
        remote.stop()
    assert isinstance(contract, RemoteContract)
    assert contract.target == remote.target
    assert contract.model_id == REMOTE_MODEL_ID
    assert contract.reward_model_id == REWARD_MODEL_ID
    assert contract.observation_schema_hash == CANONICAL_OBSERVATION_SCHEMA_HASH


def test_handshake_refuses_an_unreachable_server():
    # Bind and immediately release a port so the connection is refused rather
    # than hanging on a firewalled address.
    remote = _CapturingRemote()
    port = remote.start()
    remote.stop()
    link = RemoteLink(f"127.0.0.1:{port}", timeout_s=1.0)
    try:
        with pytest.raises(RemoteHandshakeError) as excinfo:
            link.verify()
    finally:
        link.close()
    assert "cannot reach" in str(excinfo.value)


def test_handshake_refuses_a_foreign_observation_schema():
    remote = _CapturingRemote(observation_schema_hash="not-our-schema")
    remote.start()
    link = RemoteLink(remote.target)
    try:
        with pytest.raises(RemoteHandshakeError) as excinfo:
            link.verify()
    finally:
        link.close()
        remote.stop()
    assert "observation_schema_hash" in str(excinfo.value)


# =========================================================================== #
# GetBufferStatus                                                             #
# =========================================================================== #


def test_get_buffer_status_mirrors_the_real_server():
    remote = _CapturingRemote()
    remote.start()
    link = RemoteLink(remote.target)
    mirror = RemoteBufferStatusMirror(link)
    mirror.refresh()
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, uploader=uploader, buffer_status_provider=mirror)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.step()
        driver.step(operator_success=True)
        _drain(uploader)
        # Force a synchronous refresh so the assertion is not racing the
        # background one; the provider itself never blocks.
        mirror.refresh()
        status = network.get_buffer_status()
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()
        link.close()

    assert status.replay_size == 2
    assert status.replay_insert_count == 2
    assert status.last_env_step == 1
    # The proxy answered from the remote, not from its own (empty) service.
    assert proxy.service.replay_items == []


def test_buffer_status_provider_never_blocks_and_says_so_when_empty():
    """No snapshot must be an error, never a fabricated set of zeroes."""

    class _SlowLink:
        target = "127.0.0.1:1"

        def __init__(self) -> None:
            self.calls = 0
            self.started = threading.Event()
            self.release = threading.Event()

        def buffer_status(self):
            self.calls += 1
            self.started.set()
            self.release.wait(5.0)
            raise ActorTransportError("remote is down")

    link = _SlowLink()
    mirror = RemoteBufferStatusMirror(link)
    started = time.monotonic()
    with pytest.raises(ActorTransportError) as excinfo:
        mirror()
    # The provider returned immediately even though the link is blocked.
    assert time.monotonic() - started < 1.0
    assert "holds no replay of its own" in str(excinfo.value)
    assert link.started.wait(5.0)
    link.release.set()


# =========================================================================== #
# Parameter hot-swap                                                          #
# =========================================================================== #


def _write_export(
    directory: Path, version: int, payload: bytes, learner_step: int
) -> None:
    """Write one blob + LATEST.json in the pinned export format."""

    directory.mkdir(parents=True, exist_ok=True)
    name = blob_filename(version)
    (directory / name).write_bytes(payload)
    manifest = {
        "schema": 1,
        "version": int(version),
        "learner_step": int(learner_step),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "filename": name,
    }
    (directory / LATEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _sync_client(export: Path, holder: ParamsHolder, swap: Any) -> ParamsSyncClient:
    return ParamsSyncClient(
        LocalDirectoryFetcher(export),
        str(export),
        lambda blob: {"weights": blob},
        holder=holder,
        swap_callback=swap,
        poll_interval_s=0.05,
        profiler=LatencyProfiler("paramsync", None, enabled=False),
    )


def test_params_hot_swap_mid_episode_raises_the_served_version(tmp_path):
    """A new blob mid-episode changes the version the actor is told, upwards."""

    export = tmp_path / "params_live"
    _write_export(export, 0, b"params-v0", learner_step=0)

    holder = ParamsHolder()
    validated: list[Any] = []
    # Constructed WITHOUT a validator and given one after the bootstrap blob
    # lands -- the production order, because LocalPolicyRuntime cannot exist
    # until the holder has parameters, and it validates that first blob itself.
    swap = ValidatingParamsSwap(holder)
    sync = _sync_client(export, holder, swap)
    assert sync.poll_once() is True
    assert holder.version == 0
    swap.set_validator(validated.append)

    remote = _CapturingRemote()
    remote.start()
    uploader = _uploader(remote.target)
    proxy = _proxy(
        remote,
        policy=_HolderPolicy(holder),
        uploader=uploader,
        params_holder=holder,
        params_sync=sync,
    )
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        assert driver.begin().policy_version == 0
        assert driver.step().action.policy_version == 0
        # Deliver what has been produced so far BEFORE the versions move, so
        # the fake server's own policy version can follow deterministically.
        _drain(uploader)

        _write_export(export, 3, b"params-v3-longer", learner_step=150)
        assert sync.poll_once() is True
        assert holder.version == 3
        # The remote must move with it or it would reject the forwarded
        # transition's meta.policy_version; see the module docstring.
        remote.policy.version = 3

        assert driver.step().action.policy_version == 3
        assert driver.step(operator_success=True).outcome.success is True
        _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    assert [
        item["meta"]["policy_version"] for item in remote.service.replay_items
    ] == [0, 0, 3]
    assert uploader.divergence_count == 0
    assert uploader.rejected_count == 0
    # The swap seam validated the candidate before it could serve.
    assert validated == [{"weights": b"params-v3-longer"}]
    assert swap.validated_count == 1
    metrics = proxy.metrics()
    assert metrics["params_version"] == 3
    assert metrics["params_sync"]["applied"] == 2
    assert metrics["forwarded_step"] == 3


def test_validating_swap_refuses_a_candidate_the_runtime_rejects(tmp_path):
    """A candidate that fails validation never reaches the inference path."""

    export = tmp_path / "params_live"
    _write_export(export, 0, b"good", learner_step=0)
    holder = ParamsHolder()
    swap = ValidatingParamsSwap(holder)
    sync = _sync_client(export, holder, swap)
    assert sync.poll_once() is True

    def _reject(params):
        del params
        raise ValueError("smoke failed")

    swap.set_validator(_reject)
    _write_export(export, 1, b"bad", learner_step=50)
    assert sync.poll_once() is False
    assert holder.version == 0
    assert holder.current().params == {"weights": b"good"}
    assert sync.stats["swap_errors"] == 1


# =========================================================================== #
# Shutdown                                                                    #
# =========================================================================== #


def test_stop_drains_the_backlog_against_a_slow_remote():
    remote = _CapturingRemote(delay_s=0.05)
    remote.start()
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, uploader=uploader)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        for _ in range(5):
            driver.step()
        driver.step(operator_success=True)
        # The control loop outran the uploader: that is the whole point.
        assert uploader.backlog_depth > 0
    finally:
        network.close()

    try:
        assert proxy.stop(drain=True, drain_timeout_s=30.0) is True
        assert uploader.backlog_depth == 0
        assert uploader.uploaded_count == 7  # begin + 6 steps
        assert len(remote.service.replay_items) == 6
    finally:
        remote.stop()


def test_stop_reports_an_incomplete_drain_when_the_remote_is_gone():
    """A shutdown that could not deliver everything must SAY so.

    The remote is stopped BEFORE the episode is driven, not after.  That
    ordering is load-bearing rather than incidental: the uploader is a
    background thread racing an in-process remote with no delay, so stopping it
    afterwards left it a real chance to have delivered both items already --
    in which case ``stop()`` returns True *correctly* and the assertion below
    fails for a reason that has nothing to do with the behaviour under test.
    Measured at 3 failures in 15 runs before this was pinned.  Killing the
    remote first makes delivery impossible, so an incomplete drain is the only
    outcome the production code is allowed to report.

    The handshake in ``_proxy`` still runs against a live remote, so this is
    the real "the tunnel died mid-session" shape: a proxy that verified a
    server it can no longer reach.  ``test_stop_drains_the_backlog_against_a
    _slow_remote`` pins the opposite direction with an explicit backlog guard.
    """

    remote = _CapturingRemote()
    remote.start()
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, uploader=uploader)  # handshake, while it is alive
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        remote.stop()  # the tunnel dies before anything can be forwarded
        # The proxy still answers the actor locally -- that is the point of
        # local inference -- so the episode proceeds and the queue only grows.
        driver.begin()
        driver.step()
    finally:
        network.close()

    assert proxy.stop(drain=True, drain_timeout_s=1.0) is False
    assert uploader.backlog_depth > 0


def test_stop_is_idempotent():
    remote = _CapturingRemote()
    remote.start()
    proxy = _proxy(remote)
    proxy.start()
    try:
        assert proxy.stop(drain=True, drain_timeout_s=5.0) is True
        assert proxy.stop(drain=True, drain_timeout_s=5.0) is True
    finally:
        remote.stop()


# =========================================================================== #
# The forwarding gap this phase has NOT closed                                #
# =========================================================================== #


def test_forwarded_step_is_rejected_when_the_remote_policy_disagrees():
    """PINS A KNOWN GAP -- read this before trusting the transition path.

    ``ActorSessionService._validate_data`` requires ``meta.policy_action`` to
    equal the action THAT service issued for the previous observation.  A
    forwarded request carries the PROXY's action, and the real learner server
    runs its own (stochastic, different-backend) inference over the same
    request, so in production the two cannot match.  The server then answers
    INVALID_ARGUMENT and the uploader ABANDONS the transition after its
    permanent-retry budget: it is counted in ``rejected_count`` and logged as
    ``TRANSITION NOT INGESTED``, never silently lost, but it does not reach
    replay.

    This test exists so the failure is a pinned, named fact rather than a
    surprise on the robot.  Closing it needs a server-side change, which is
    outside this component.
    """

    remote = _CapturingRemote(policy=_FakePolicy(action=np.zeros(7, np.float32)))
    remote.start()
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, policy=_FakePolicy(action=LOCAL_ACTION), uploader=uploader)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        # The actor's own episode is unaffected: local inference answered it.
        np.testing.assert_array_equal(driver.begin().action, LOCAL_ACTION)
        assert driver.step().ack.accepted
        _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    assert uploader.uploaded_count == 1  # the BeginEpisode went through
    assert uploader.rejected_count == 1  # the Step did not
    assert "policy_action" in uploader.last_error
    assert not remote.service.replay_items


# =========================================================================== #
# Latency records                                                             #
# =========================================================================== #


def test_proxy_latency_records_carry_the_pinned_fields(tmp_path):
    remote = _CapturingRemote()
    remote.start()
    holder = ParamsHolder({"weights": b"x"}, 4)
    probe = ProxyLatencyProbe.from_env(
        tmp_path / "proxy.jsonl", env={"HIL_LATENCY_PROFILE": "1"}
    )
    uploader = _uploader(remote.target)
    proxy = _proxy(
        remote,
        policy=_HolderPolicy(holder),
        uploader=uploader,
        latency_probe=probe,
        params_holder=holder,
    )
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.step()
        driver.step(operator_success=True)
        _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    rows = [
        json.loads(line)
        for line in (tmp_path / "proxy.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows, "profiling was enabled but nothing was written"
    assert {row["role"] for row in rows} == {LATENCY_ROLE}
    steps = [row for row in rows if row.get("rpc") == "step"]
    assert len(steps) == 2
    for row in steps:
        assert row["total_ms"] >= 0.0
        assert row["queue_depth"] >= 0
        assert row["params_version"] == 4
        assert row["params_age_s"] >= 0.0
        # Contributed from inside the service by the REAL finalizer.
        assert row["local_finalize_ms"] >= 0.0
        assert row["transition_id"].startswith(RUN_ID)
    begins = [row for row in rows if row.get("rpc") == "begin_episode"]
    assert len(begins) == 1
    assert begins[0]["queue_depth"] >= 0
    assert begins[0]["total_ms"] >= 0.0


def test_disabled_probe_writes_nothing(tmp_path):
    remote = _CapturingRemote()
    remote.start()
    probe = ProxyLatencyProbe.from_env(tmp_path / "proxy.jsonl", env={})
    assert probe.enabled is False
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, uploader=uploader, latency_probe=probe)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(network)
        driver.begin()
        driver.step(operator_success=True)
        _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()
    assert not (tmp_path / "proxy.jsonl").exists()


# =========================================================================== #
# Small units                                                                 #
# =========================================================================== #


def test_discard_transition_has_no_prime_observation():
    """The pixel path is deliberate: there is no second consumer to share with."""

    assert not hasattr(discard_transition, "prime_observation")
    assert discard_transition({"meta": {}, "transition": {}}, True) is None


def test_resolve_proxy_port_precedence():
    assert resolve_proxy_port(None, env={}) == 50253
    assert resolve_proxy_port(None, env={"HIL_LOCAL_POLICY_PORT": "50999"}) == 50999
    assert resolve_proxy_port(0, env={"HIL_LOCAL_POLICY_PORT": "50999"}) == 0
    with pytest.raises(ProxyError):
        resolve_proxy_port(None, env={"HIL_LOCAL_POLICY_PORT": "not-a-port"})
    with pytest.raises(ProxyError):
        resolve_proxy_port(70000)


def test_module_import_does_not_import_jax():
    """The entrypoint's device prologue must be able to win the jax race."""

    probe = (
        "import sys;"
        "import ur_env.local_policy.proxy as p;"
        "import scripts.run_local_policy_proxy as e;"
        "print('jax' in sys.modules)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_INFRA_ROOT),
            str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher"),
        ]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_INFRA_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", result.stdout


# =========================================================================== #
# The entrypoint's pure parts                                                 #
# =========================================================================== #


def _entrypoint():
    import importlib

    sys.path.insert(0, str(_INFRA_ROOT))
    return importlib.import_module("scripts.run_local_policy_proxy")


def test_entrypoint_resolves_the_params_directory():
    module = _entrypoint()
    args = module.parse_args(["--params-run-root", "/home/user/runs/run-1"])
    assert module.resolve_params_dir(args, env={}) == (
        "/home/user/runs/run-1/params_live"
    )
    args = module.parse_args(["--params-remote-dir", "/exports/params_live"])
    assert module.resolve_params_dir(args, env={}) == "/exports/params_live"
    args = module.parse_args([])
    assert module.resolve_params_dir(
        args, env={"HIL_PARAMS_REMOTE_DIR": "/from/env"}
    ) == "/from/env"
    with pytest.raises(SystemExit):
        module.resolve_params_dir(module.parse_args([]), env={})


def test_entrypoint_refuses_a_non_loopback_bind():
    module = _entrypoint()
    with pytest.raises(SystemExit):
        module.parse_args(["--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        module.parse_args(
            ["--params-remote-dir", "/x", "--params-run-root", "/y"]
        )


def test_entrypoint_waits_for_the_first_params_blob():
    module = _entrypoint()

    class _Sync:
        remote_dir = "/exports/params_live"
        manifest_path = "/exports/params_live/LATEST.json"

        def __init__(self, succeed_after: int) -> None:
            self.calls = 0
            self._succeed_after = succeed_after
            self.applied_version = -1

        def poll_once(self) -> bool:
            self.calls += 1
            if self.calls >= self._succeed_after:
                self.applied_version = 0
                return True
            return False

    sync = _Sync(succeed_after=3)
    ticks = [0.0]

    def _clock() -> float:
        return ticks[0]

    def _sleep(seconds: float) -> None:
        ticks[0] += seconds

    assert (
        module.wait_for_initial_params(
            sync, timeout_s=10.0, poll_interval_s=1.0, sleep=_sleep, clock=_clock
        )
        == 0
    )
    assert sync.calls == 3

    never = _Sync(succeed_after=10_000)
    with pytest.raises(SystemExit) as excinfo:
        module.wait_for_initial_params(
            never, timeout_s=2.0, poll_interval_s=1.0, sleep=_sleep, clock=_clock
        )
    assert "HIL_PARAMS_EXPORT=1" in str(excinfo.value)


# =========================================================================== #
# Full stack with a real LocalPolicyRuntime (jax)                             #
# =========================================================================== #


@_needs_jax
def test_full_episode_with_a_real_local_policy_runtime():
    """The same episode, with ``LocalPolicyRuntime`` instead of the fake.

    A tiny hand-built agent stands in for the production network (that one is
    minutes and a gigabyte; ``tests/test_local_policy_runtime.py`` covers it
    behind ``RUN_HIL_SERL_ACTUAL_LOCAL_POLICY=1``).  What is exercised here is
    the seam this component owns: a real runtime object as
    ``ActorSessionService``'s ``sample_action``, validating and serving real
    canonical observations over the real transport, with its version coming
    from a real ``ParamsHolder``.
    """

    import jax.numpy as jnp

    from ur_env.learner.policy import canonical_policy_observation
    from ur_env.local_policy.runtime import LocalPolicyRuntime

    class _State:
        def __init__(self, params):
            self.params = params

        def replace(self, params):
            return _State(params)

    class _Agent:
        def __init__(self, params):
            self.state = _State(params)

        def replace(self, state):
            return _Agent(state.params)

    base = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _sample(params, observation, seed, deterministic):
        del seed, deterministic
        # Reads the observation so a broken tensor would show up here, and the
        # parameters so a swap actually changes the action.
        assert set(observation) == {"state", "cam1", "cam2"}
        scale = float(np.asarray(params["scale"])[0])
        action = base.copy()
        action[:3] = action[:3] * scale
        return jnp.asarray(action)

    params = {"scale": jnp.asarray([0.5], dtype=jnp.float32)}
    holder = ParamsHolder(params, 2)
    runtime = LocalPolicyRuntime(_Agent(params), holder, sample_action=_sample)

    expected = base.copy()
    expected[:3] = expected[:3] * 0.5
    remote = _CapturingRemote(policy=_FakePolicy(action=expected, version=2))
    remote.start()
    uploader = _uploader(remote.target)
    proxy = _proxy(remote, policy=runtime, uploader=uploader, params_holder=holder)
    proxy.start()
    network = _client(proxy.port)
    try:
        driver = _ActorDriver(
            network,
            observation_factory=lambda value: canonical_policy_observation(
                value % 256
            ),
        )
        begin = driver.begin()
        assert begin.policy_version == 2
        np.testing.assert_allclose(begin.action, expected, rtol=0, atol=0)
        assert driver.step().action is not None
        assert driver.step(operator_success=True).outcome.success is True
        _drain(uploader)
    finally:
        network.close()
        proxy.stop(drain=False)
        remote.stop()

    assert runtime.inference_count == 2  # begin + one non-terminal step
    assert uploader.divergence_count == 0
    assert uploader.rejected_count == 0
    assert len(remote.service.replay_items) == 2
