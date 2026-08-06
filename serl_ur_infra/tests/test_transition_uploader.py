"""Contract tests for the raw-bytes transition forwarder.

The fake remote here is not a mock: it is this repo's own
:class:`~ur_env.grpc_actor_transport.GrpcActorServicer` in front of a real
:class:`~ur_env.actor_network.ActorSessionService`, served over a real
in-process gRPC socket, with one thin recording/fault-injection layer around the
two RPCs the uploader uses.  That matters because the properties under test --
byte identity, dedup on retry, ACK semantics -- are *server* behaviours; a
hand-written stub would let the uploader pass against a server that does not
exist.

Two tests deliberately use a hand-written servicer instead: the one that proves
the remote ACTION is never read (a real service cannot be made to return a
malformed action) and the one that proves a refused ``BeginEpisode`` is not
retried forever.

No jax anywhere; runs in the actor venv:

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:..." \\
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_transition_uploader.py
"""

from __future__ import annotations

from concurrent import futures
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Optional

import grpc
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ActorSessionService,
    ObservationPacket,
)
from ur_env.grpc_actor_transport import (  # noqa: E402
    DEFAULT_MAX_MESSAGE_BYTES,
    GrpcActorServicer,
    data_to_proto,
    observation_to_proto,
)
from ur_env.latency_profile import LatencyProfiler  # noqa: E402
from ur_env.local_policy.uploader import (  # noqa: E402
    BEGIN_EPISODE_METHOD,
    KIND_BEGIN_EPISODE,
    KIND_STEP,
    LATENCY_ROLE,
    STEP_METHOD,
    TransitionUploader,
)
from ur_env.proto import actor_transport_pb2 as pb  # noqa: E402
from ur_env.proto import actor_transport_pb2_grpc as pb_grpc  # noqa: E402


ACTOR_ID = "actor"
RUN_ID = "run"
SESSION_ID = "session"
_ZERO_ACTION = np.zeros(7, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Request construction: a chain of requests a real server accepts.
# --------------------------------------------------------------------------- #


def _observation(value: int) -> dict:
    """Small but structurally real: nested state + two image tensors."""

    return {
        "state": np.full((1, 6), value / 100.0, dtype=np.float32),
        "cam1": np.full((1, 4, 4, 3), value % 256, dtype=np.uint8),
        "cam2": np.full((1, 4, 4, 3), (255 - value) % 256, dtype=np.uint8),
    }


class _EpisodeRequests:
    """Builds byte-stable BeginEpisode/Step requests for one episode.

    Every identity the server checks (request_id, env_step, step_id, the
    observation chain, the timestamp of the *source* observation) is threaded
    through here, so the produced bytes are exactly what a live actor would put
    on the wire.
    """

    def __init__(
        self,
        *,
        episode_id: int = 0,
        env_step: int = 0,
        observation_seq: int = 0,
        session_id: str = SESSION_ID,
    ) -> None:
        self.episode_id = episode_id
        self.env_step = env_step
        self.session_id = session_id
        self._obs_seq = observation_seq
        self._step_id = 0
        self._request_id = 1
        self._current_id = f"o{self._obs_seq}"
        self._current_ts = 1_000 + self._obs_seq

    def begin(self) -> bytes:
        request = pb.BeginEpisodeRequest(
            protocol_version=PROTOCOL_VERSION,
            actor_id=ACTOR_ID,
            run_id=RUN_ID,
            session_id=self.session_id,
            episode_id=self.episode_id,
            request_id=1,
            created_monotonic_ns=10_000 + self._obs_seq,
            observation=observation_to_proto(
                ObservationPacket(
                    self._current_id, self._current_ts, _observation(self._obs_seq)
                )
            ),
            deterministic=False,
        )
        self._request_id = 2
        return request.SerializeToString()

    def step(
        self,
        *,
        done: bool = False,
        truncated: bool = False,
        operator_success: bool = False,
        reward: float = 0.0,
    ) -> tuple[bytes, dict]:
        """Return ``(payload, local_outcome_summary)`` for the next transition."""

        terminal = bool(done) or bool(truncated)
        source_id = self._current_id
        source_ts = self._current_ts
        self._obs_seq += 1
        next_id = f"o{self._obs_seq}"
        next_ts = 1_000 + self._obs_seq
        transition_id = f"{RUN_ID}:{self.env_step}"
        mask = 0.0 if done else 1.0
        data = {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "run_id": RUN_ID,
                "actor_id": ACTOR_ID,
                "session_id": self.session_id,
                "transition_id": transition_id,
                "env_step": self.env_step,
                "timestamp_ns": source_ts,
                "policy_version": 0,
                "policy_action": _ZERO_ACTION,
                "intervened": False,
                "auto_success": False,
                "operator_success": bool(operator_success),
            },
            "transition": {
                "episode_id": self.episode_id,
                "step_id": self._step_id,
                "observation_id": source_id,
                "actions": _ZERO_ACTION,
                "next_observation_id": next_id,
                "rewards": float(reward),
                "masks": mask,
                "dones": bool(done),
                "truncated": bool(truncated),
            },
        }
        request = pb.StepRequest(
            protocol_version=PROTOCOL_VERSION,
            actor_id=ACTOR_ID,
            run_id=RUN_ID,
            session_id=self.session_id,
            request_id=self._request_id,
            created_monotonic_ns=20_000 + self._obs_seq,
            data=data_to_proto(data),
            next_observation=observation_to_proto(
                ObservationPacket(next_id, next_ts, _observation(self._obs_seq))
            ),
            request_action=not terminal,
            deterministic=False,
        )
        self._request_id += 1
        self._step_id += 1
        self.env_step += 1
        self._current_id = next_id
        self._current_ts = next_ts
        # What the server's identity finalizer will report for these bytes.
        summary = {
            "transition_id": transition_id,
            "done": bool(done) or bool(operator_success),
            "truncated": bool(truncated) and not operator_success,
            "success": bool(operator_success),
            "reward": 1.0 if operator_success else float(reward),
            "mask": 0.0 if (done or operator_success) else 1.0,
        }
        return request.SerializeToString(), summary


# --------------------------------------------------------------------------- #
# The fake remote: this repo's servicer + service, plus recording and faults.
# --------------------------------------------------------------------------- #


class _Recorder:
    """Per-RPC log, concurrency gauge and fault queues."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[tuple[str, bytes, bytes]] = []
        self.delay_s = 0.0
        self.in_flight = 0
        self.max_in_flight = 0
        self.faults_before: list[tuple[str, Any, str]] = []
        self.faults_after: list[tuple[str, Any, str]] = []

    @property
    def kinds(self) -> list[str]:
        with self.lock:
            return [kind for kind, _, _ in self.calls]

    @property
    def payloads(self) -> list[bytes]:
        with self.lock:
            return [payload for _, payload, _ in self.calls]

    def transition_ids(self) -> list[str]:
        ids = []
        with self.lock:
            calls = list(self.calls)
        for kind, payload, _ in calls:
            if kind == KIND_STEP:
                ids.append(pb.StepRequest.FromString(payload).data.meta.transition_id)
            else:
                ids.append("<begin>")
        return ids

    def enter(self, kind: str, request: Any) -> None:
        with self.lock:
            self.calls.append(
                (
                    kind,
                    request.SerializeToString(),
                    request.SerializeToString(deterministic=True),
                )
            )
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.delay_s:
            time.sleep(self.delay_s)

    def exit(self) -> None:
        with self.lock:
            self.in_flight -= 1

    def take(self, queue: list, kind: str) -> Optional[tuple[Any, str]]:
        with self.lock:
            for index, (wanted, code, detail) in enumerate(queue):
                if wanted == kind:
                    queue.pop(index)
                    return code, detail
        return None


class _RecordingServicer(GrpcActorServicer):
    """The production servicer, wrapped so a test can watch and break it."""

    def __init__(self, service: ActorSessionService, recorder: _Recorder) -> None:
        super().__init__(service)
        self._recorder = recorder

    def BeginEpisode(self, request, context):
        return self._wrap(KIND_BEGIN_EPISODE, request, context, super().BeginEpisode)

    def Step(self, request, context):
        return self._wrap(KIND_STEP, request, context, super().Step)

    def _wrap(self, kind, request, context, delegate):
        self._recorder.enter(kind, request)
        try:
            fault = self._recorder.take(self._recorder.faults_before, kind)
            if fault is not None:
                context.abort(*fault)
            reply = delegate(request, context)
            fault = self._recorder.take(self._recorder.faults_after, kind)
            if fault is not None:
                # The service already committed this transition; the failure is
                # on the way back, which is exactly when dedup has to work.
                context.abort(*fault)
            return reply
        finally:
            self._recorder.exit()


class _Remote:
    """A startable/stoppable in-process gRPC server on a stable port."""

    def __init__(self, servicer: Any, port: int = 0) -> None:
        self._servicer = servicer
        self.port = port
        self._server = None

    def start(self) -> int:
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="fake-remote"),
            options=(
                ("grpc.max_receive_message_length", DEFAULT_MAX_MESSAGE_BYTES),
                ("grpc.max_send_message_length", DEFAULT_MAX_MESSAGE_BYTES),
            ),
        )
        pb_grpc.add_ActorTransportServicer_to_server(self._servicer, server)
        bound = server.add_insecure_port(f"127.0.0.1:{self.port}")
        assert bound, "failed to bind the fake remote"
        server.start()
        self._server = server
        self.port = bound
        return bound

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop(grace=0).wait()
            self._server = None


def _service(**kwargs: Any) -> ActorSessionService:
    return ActorSessionService(
        lambda observation, deterministic: (_ZERO_ACTION.copy(), 0), **kwargs
    )


def _uploader(port: int, **kwargs: Any) -> TransitionUploader:
    kwargs.setdefault("backoff_initial_s", 0.01)
    kwargs.setdefault("backoff_max_s", 0.05)
    kwargs.setdefault("timeout_s", 5.0)
    # env={} so a developer shell with HIL_LATENCY_PROFILE=1 cannot turn these
    # tests into a file-writing suite.
    kwargs.setdefault("env", {})
    return TransitionUploader(f"127.0.0.1:{port}", **kwargs)


def _drain(uploader: TransitionUploader, timeout_s: float = 15.0) -> None:
    assert uploader.wait_until_idle(timeout_s), (
        f"uploader did not drain: {uploader.metrics()}"
    )


# --------------------------------------------------------------------------- #
# Wire plumbing
# --------------------------------------------------------------------------- #


def test_method_paths_match_the_generated_stub():
    """The uploader bypasses the stub; it must still dial the same methods."""

    class _PathProbe:
        def __init__(self) -> None:
            self.paths: dict[str, str] = {}

        def unary_unary(self, method, **kwargs):
            self.paths[method.rsplit("/", 1)[-1]] = method
            return None

    probe = _PathProbe()
    pb_grpc.ActorTransportStub(probe)

    assert probe.paths["BeginEpisode"] == BEGIN_EPISODE_METHOD
    assert probe.paths["Step"] == STEP_METHOD


# --------------------------------------------------------------------------- #
# Delivery: bytes, order, one at a time
# --------------------------------------------------------------------------- #


def test_delivers_the_enqueued_bytes_verbatim_and_in_order():
    recorder = _Recorder()
    service = _service()
    remote = _Remote(_RecordingServicer(service, recorder))
    port = remote.start()
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    sent: list[bytes] = []
    try:
        uploader.start()
        begin = episode.begin()
        sent.append(begin)
        uploader.enqueue_begin_episode(begin)
        for index in range(3):
            payload, summary = episode.step(done=index == 2)
            sent.append(payload)
            uploader.enqueue_step(payload, local_outcome_summary=summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    # Byte identity, twice over: the wire payload the server parsed, and the
    # deterministic fingerprint it dedups on.
    assert recorder.payloads == sent
    with recorder.lock:
        fingerprints = [fingerprint for _, _, fingerprint in recorder.calls]
    assert fingerprints == sent
    assert recorder.kinds == [KIND_BEGIN_EPISODE, KIND_STEP, KIND_STEP, KIND_STEP]
    assert recorder.transition_ids() == ["<begin>", "run:0", "run:1", "run:2"]
    # The real service really ingested them, in order.
    assert [item["meta"]["transition_id"] for item in service.replay_items] == [
        "run:0",
        "run:1",
        "run:2",
    ]
    assert uploader.uploaded_count == 4
    assert uploader.divergence_count == 0
    assert uploader.rejected_count == 0
    assert uploader.backlog_depth == 0


def test_only_one_request_is_ever_in_flight():
    recorder = _Recorder()
    recorder.delay_s = 0.03
    remote = _Remote(_RecordingServicer(_service(), recorder))
    port = remote.start()
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        for _ in range(5):
            payload, summary = episode.step()
            uploader.enqueue_step(payload, local_outcome_summary=summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert recorder.max_in_flight == 1
    assert recorder.transition_ids() == [
        "<begin>",
        "run:0",
        "run:1",
        "run:2",
        "run:3",
        "run:4",
    ]


def test_episode_boundaries_are_preserved_across_a_second_episode():
    """The queue is one stream: episode 1's BeginEpisode may not overtake."""

    recorder = _Recorder()
    service = _service()
    remote = _Remote(_RecordingServicer(service, recorder))
    port = remote.start()
    uploader = _uploader(port)
    first = _EpisodeRequests(episode_id=0, env_step=0, observation_seq=0)
    try:
        uploader.start()
        uploader.enqueue_begin_episode(first.begin())
        for index in range(2):
            payload, summary = first.step(done=index == 1)
            uploader.enqueue_step(payload, local_outcome_summary=summary)
        second = _EpisodeRequests(
            episode_id=1,
            env_step=first.env_step,
            observation_seq=50,
            session_id="session-2",
        )
        uploader.enqueue_begin_episode(second.begin())
        payload, summary = second.step(operator_success=True)
        uploader.enqueue_step(payload, local_outcome_summary=summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert recorder.transition_ids() == [
        "<begin>",
        "run:0",
        "run:1",
        "<begin>",
        "run:2",
    ]
    assert uploader.uploaded_count == 5
    assert uploader.divergence_count == 0
    # The MANUAL success really landed as a success on the server side.
    assert bool(service.replay_items[-1]["transition"]["success"]) is True


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def test_transient_failure_retries_identical_bytes_and_the_server_dedups():
    recorder = _Recorder()
    service = _service()
    # UNAVAILABLE *after* the service committed the transition: the reply is
    # lost on the way back, so the retry meets a server that already has it.
    recorder.faults_after.append(
        (KIND_STEP, grpc.StatusCode.UNAVAILABLE, "connection reset by peer")
    )
    remote = _Remote(_RecordingServicer(service, recorder))
    port = remote.start()
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        first, first_summary = episode.step()
        uploader.enqueue_step(first, local_outcome_summary=first_summary)
        second, second_summary = episode.step(done=True)
        uploader.enqueue_step(second, local_outcome_summary=second_summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    # Four RPCs for three requests: the failed Step was sent twice.
    assert recorder.transition_ids() == ["<begin>", "run:0", "run:0", "run:1"]
    payloads = recorder.payloads
    assert payloads[1] == payloads[2] == first
    # Dedup: identical bytes -> identical fingerprint -> one replay row.
    assert [item["meta"]["transition_id"] for item in service.replay_items] == [
        "run:0",
        "run:1",
    ]
    assert uploader.retry_count == 1
    assert uploader.uploaded_count == 3
    assert uploader.rejected_count == 0
    assert uploader.divergence_count == 0


def test_a_restarted_remote_loses_nothing_and_keeps_the_order():
    recorder = _Recorder()
    service = _service()
    servicer = _RecordingServicer(service, recorder)
    remote = _Remote(servicer)
    port = remote.start()
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        first, first_summary = episode.step()
        uploader.enqueue_step(first, local_outcome_summary=first_summary)
        _drain(uploader, timeout_s=10.0)

        # Kill the remote with a full queue behind it.  The service object
        # survives (this is a dropped tunnel, not a lost learner).
        remote.stop()
        pending = []
        for index in range(3):
            payload, summary = episode.step(done=index == 2)
            pending.append(payload)
            uploader.enqueue_step(payload, local_outcome_summary=summary)
        deadline = time.monotonic() + 3.0
        while uploader.retry_count == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert uploader.retry_count > 0, uploader.metrics()
        assert uploader.backlog_depth == 3

        remote.start()  # same port, same service
        _drain(uploader, timeout_s=20.0)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert [item["meta"]["transition_id"] for item in service.replay_items] == [
        "run:0",
        "run:1",
        "run:2",
        "run:3",
    ]
    assert uploader.rejected_count == 0
    assert uploader.backlog_depth == 0


def test_a_permanently_refused_request_does_not_block_the_queue():
    recorder = _Recorder()
    service = _service()
    # A non-transient status on the way back, twice: one more attempt than the
    # allowance, so the item is abandoned rather than retried forever.  The
    # service itself accepted the transition both times (dedup on the second),
    # which is why the chain behind it stays valid.
    for _ in range(2):
        recorder.faults_after.append(
            (KIND_STEP, grpc.StatusCode.INVALID_ARGUMENT, "malformed request")
        )
    remote = _Remote(_RecordingServicer(service, recorder))
    port = remote.start()
    uploader = _uploader(port, permanent_retry_count=1)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        first, first_summary = episode.step()
        uploader.enqueue_step(first, local_outcome_summary=first_summary)
        second, second_summary = episode.step()
        uploader.enqueue_step(second, local_outcome_summary=second_summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    # One retry of the refused item, then it is abandoned and the queue moves.
    assert uploader.rejected_count == 1
    assert uploader.uploaded_count == 2  # the BeginEpisode and the next Step
    assert uploader.backlog_depth == 0
    assert "INVALID_ARGUMENT" in uploader.last_error
    assert recorder.transition_ids() == ["<begin>", "run:0", "run:0", "run:1"]
    assert [item["meta"]["transition_id"] for item in service.replay_items] == [
        "run:0",
        "run:1",
    ]


def test_worker_survives_a_reply_that_makes_no_sense():
    """A malformed reply costs one item, never the thread."""

    class _NonsenseServicer(pb_grpc.ActorTransportServicer):
        def __init__(self) -> None:
            self.seen: list[str] = []

        def BeginEpisode(self, request, context):
            self.seen.append(KIND_BEGIN_EPISODE)
            return pb.ActionReply(ok=True, protocol_version=PROTOCOL_VERSION)

        def Step(self, request, context):
            self.seen.append(request.data.meta.transition_id)
            # No ack, no outcome: every optional field missing at once.
            return pb.StepReply()

    servicer = _NonsenseServicer()
    remote = _Remote(servicer)
    port = remote.start()
    uploader = _uploader(port, permanent_retry_count=0)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        for _ in range(2):
            payload, summary = episode.step()
            uploader.enqueue_step(payload, local_outcome_summary=summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert servicer.seen == [KIND_BEGIN_EPISODE, "run:0", "run:1"]
    # accepted=false on an empty ack: both steps are refused, neither wedges.
    assert uploader.rejected_count == 2
    assert uploader.uploaded_count == 1
    assert uploader.running is False


# --------------------------------------------------------------------------- #
# The reply: action discarded, outcome compared
# --------------------------------------------------------------------------- #


class _HostileActionServicer(pb_grpc.ActorTransportServicer):
    """Answers with an action no actor client would accept.

    Wrong protocol version, wrong session, wrong request id, wrong dimension,
    NaN values, absurd inference time.  ``GrpcActorNetwork`` raises on every one
    of these; the uploader must not look at any of them.
    """

    def __init__(self, outcome_overrides: Optional[dict] = None) -> None:
        self.overrides = dict(outcome_overrides or {})
        self.steps: list[str] = []

    def BeginEpisode(self, request, context):
        return pb.ActionReply(
            ok=True,
            protocol_version="not-a-version",
            session_id="somebody-else",
            request_id=999,
            observation_id="wrong",
            action=[float("nan")] * 3,
            policy_version=7,
            server_inference_ms=-1.0,
        )

    def Step(self, request, context):
        meta = request.data.meta
        transition = request.data.transition
        self.steps.append(meta.transition_id)
        outcome = dict(
            transition_id=meta.transition_id,
            reward=float(transition.rewards),
            mask=float(transition.masks),
            done=bool(transition.dones),
            truncated=bool(transition.truncated),
            success=False,
            classifier_evaluated=False,
        )
        outcome.update(self.overrides)
        return pb.StepReply(
            ack=pb.Ack(
                accepted=True,
                transition_id=meta.transition_id,
                session_id=request.session_id,
                request_id=request.request_id,
            ),
            has_action=True,
            action=pb.ActionReply(
                ok=True,
                protocol_version="not-a-version",
                session_id="somebody-else",
                request_id=999,
                observation_id="wrong",
                action=[float("nan")] * 3,
                policy_version=0,
                server_inference_ms=float("inf"),
            ),
            outcome=pb.TransitionOutcome(**outcome),
        )


def test_the_remote_action_is_discarded_entirely():
    servicer = _HostileActionServicer()
    remote = _Remote(servicer)
    port = remote.start()
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        for _ in range(2):
            payload, summary = episode.step()
            uploader.enqueue_step(payload, local_outcome_summary=summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert servicer.steps == ["run:0", "run:1"]
    assert uploader.uploaded_count == 3
    assert uploader.rejected_count == 0
    assert uploader.divergence_count == 0


def test_outcome_divergence_is_logged_loudly_and_counted(caplog):
    servicer = _HostileActionServicer(outcome_overrides={"truncated": True})
    remote = _Remote(servicer)
    port = remote.start()
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        payload, summary = episode.step()
        assert summary["truncated"] is False
        with caplog.at_level(logging.WARNING, logger="ur_env.local_policy.uploader"):
            uploader.enqueue_step(payload, local_outcome_summary=summary)
            _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert uploader.divergence_count == 1
    # Still delivered: a divergence is an alarm, not a delivery failure.
    assert uploader.uploaded_count == 2
    assert uploader.rejected_count == 0
    messages = [record.getMessage() for record in caplog.records]
    loud = [text for text in messages if "OUTCOME DIVERGENCE" in text]
    assert len(loud) == 1
    assert "run:0" in loud[0]
    assert "truncated" in loud[0]


def test_a_matching_outcome_is_not_a_divergence():
    """The server's own finalizer against the local summary: identical."""

    recorder = _Recorder()
    remote = _Remote(_RecordingServicer(_service(), recorder))
    port = remote.start()
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        for kwargs in ({}, {"reward": 0.5}, {"operator_success": True}):
            payload, summary = episode.step(**kwargs)
            uploader.enqueue_step(payload, local_outcome_summary=summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert uploader.divergence_count == 0
    assert uploader.uploaded_count == 4


def test_a_begin_episode_the_server_refuses_is_not_retried_forever():
    class _RefusingServicer(pb_grpc.ActorTransportServicer):
        def __init__(self) -> None:
            self.count = 0

        def BeginEpisode(self, request, context):
            self.count += 1
            return pb.ActionReply(ok=False, error="policy is faulted")

    servicer = _RefusingServicer()
    remote = _Remote(servicer)
    port = remote.start()
    uploader = _uploader(port)
    try:
        uploader.start()
        uploader.enqueue_begin_episode(_EpisodeRequests().begin())
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert servicer.count == 1
    assert uploader.rejected_count == 1
    assert uploader.uploaded_count == 0
    assert "policy is faulted" in uploader.last_error


# --------------------------------------------------------------------------- #
# Queue lifecycle and metrics
# --------------------------------------------------------------------------- #


def test_stop_with_drain_blocks_until_the_queue_is_empty():
    recorder = _Recorder()
    recorder.delay_s = 0.02
    service = _service()
    remote = _Remote(_RecordingServicer(service, recorder))
    port = remote.start()
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        for _ in range(5):
            payload, summary = episode.step()
            uploader.enqueue_step(payload, local_outcome_summary=summary)
        assert uploader.backlog_depth > 0
        assert uploader.stop(drain=True, timeout_s=20.0) is True
    finally:
        remote.stop()

    assert uploader.backlog_depth == 0
    assert uploader.queued_bytes == 0
    assert uploader.uploaded_count == 6
    assert len(service.replay_items) == 5
    assert uploader.running is False


def test_stop_without_drain_keeps_the_backlog_for_the_next_start():
    recorder = _Recorder()
    service = _service()
    servicer = _RecordingServicer(service, recorder)
    remote = _Remote(servicer)
    port = remote.start()
    remote.stop()  # nothing is listening yet
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        for index in range(3):
            payload, summary = episode.step(done=index == 2)
            uploader.enqueue_step(payload, local_outcome_summary=summary)
        deadline = time.monotonic() + 3.0
        while uploader.retry_count == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert uploader.stop(drain=False, timeout_s=5.0) is False
        assert uploader.backlog_depth == 4
        assert uploader.running is False

        remote.start()
        uploader.start()
        _drain(uploader, timeout_s=20.0)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert recorder.transition_ids() == ["<begin>", "run:0", "run:1", "run:2"]
    assert uploader.uploaded_count == 4


def test_backlog_metrics_move_with_the_queue():
    recorder = _Recorder()
    recorder.delay_s = 0.02
    remote = _Remote(_RecordingServicer(_service(), recorder))
    port = remote.start()
    uploader = _uploader(port)
    episode = _EpisodeRequests()
    try:
        assert uploader.backlog_depth == 0
        assert uploader.oldest_age_s == 0.0
        assert uploader.metrics()["running"] is False

        begin = episode.begin()
        uploader.enqueue_begin_episode(begin)
        payloads = [begin]
        for _ in range(3):
            payload, summary = episode.step()
            payloads.append(payload)
            uploader.enqueue_step(payload, local_outcome_summary=summary)

        assert uploader.backlog_depth == 4
        assert uploader.queued_bytes == sum(len(item) for item in payloads)
        assert uploader.high_water_mark == 4
        time.sleep(0.05)
        assert uploader.oldest_age_s >= 0.05

        uploader.start()
        _drain(uploader)
        snapshot = uploader.metrics()
        assert snapshot["backlog_depth"] == 0
        assert snapshot["queued_bytes"] == 0
        assert snapshot["oldest_age_s"] == 0.0
        assert snapshot["enqueued_count"] == 4
        assert snapshot["uploaded_count"] == 4
        assert snapshot["high_water_mark"] == 4
    finally:
        uploader.stop(drain=True)
        remote.stop()


def test_high_water_warning_fires_once_per_excursion(caplog):
    recorder = _Recorder()
    remote = _Remote(_RecordingServicer(_service(), recorder))
    port = remote.start()
    uploader = _uploader(port, high_water_depth=4)
    episode = _EpisodeRequests()
    try:
        with caplog.at_level(logging.WARNING, logger="ur_env.local_policy.uploader"):
            uploader.enqueue_begin_episode(episode.begin())
            for _ in range(6):
                payload, summary = episode.step()
                uploader.enqueue_step(payload, local_outcome_summary=summary)
            first_excursion = _high_water_warnings(caplog)
            assert uploader.backlog_depth == 7

            # Drain below the low-water mark, then climb again.
            uploader.start()
            _drain(uploader)
            uploader.stop(drain=True)
            for _ in range(6):
                payload, summary = episode.step()
                uploader.enqueue_step(payload, local_outcome_summary=summary)
            second_excursion = _high_water_warnings(caplog)
    finally:
        uploader.stop(drain=False)
        remote.stop()

    assert first_excursion == 1, "one warning per excursion, not one per item"
    assert second_excursion == 2, "a new excursion warns again"


def _high_water_warnings(caplog) -> int:
    return sum(
        1
        for record in caplog.records
        if "high-water mark" in record.getMessage()
    )


def test_enqueue_rejects_wiring_mistakes_but_not_runtime_conditions():
    uploader = _uploader(1)  # never started, never dialled
    episode = _EpisodeRequests()
    with pytest.raises(ValueError, match="kind must be one of"):
        uploader.enqueue("nonsense", b"x")
    with pytest.raises(TypeError, match="serialized request bytes"):
        uploader.enqueue_step(pb.StepRequest())
    with pytest.raises(ValueError, match="payload is empty"):
        uploader.enqueue_step(b"")

    # A stopped uploader still accepts work: refusing it here would drop a
    # transition inside the proxy's Step handler.
    assert uploader.enqueue_begin_episode(episode.begin()) == 0
    assert uploader.backlog_depth == 1
    assert uploader.stop(drain=True, timeout_s=0.5) is False


def test_stop_and_close_are_idempotent():
    recorder = _Recorder()
    remote = _Remote(_RecordingServicer(_service(), recorder))
    port = remote.start()
    uploader = _uploader(port)
    try:
        uploader.start()
        uploader.start()  # second start is a no-op
        uploader.enqueue_begin_episode(_EpisodeRequests().begin())
        _drain(uploader)
        assert uploader.stop(drain=True) is True
        assert uploader.stop(drain=True) is True
        uploader.close()
        uploader.close()
    finally:
        remote.stop()

    assert uploader.uploaded_count == 1
    assert uploader.running is False


# --------------------------------------------------------------------------- #
# Latency records
# --------------------------------------------------------------------------- #


def test_latency_records_carry_the_specified_fields(tmp_path):
    path = tmp_path / "uploader.jsonl"
    profiler = LatencyProfiler(LATENCY_ROLE, path)
    servicer = _HostileActionServicer(outcome_overrides={"truncated": True})
    remote = _Remote(servicer)
    port = remote.start()
    uploader = _uploader(port, profiler=profiler)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        payload, summary = episode.step()
        uploader.enqueue_step(payload, local_outcome_summary=summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()
        profiler.close()

    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == 2
    assert {row["role"] for row in rows} == {LATENCY_ROLE}
    begin_row, step_row = rows
    assert begin_row["kind"] == KIND_BEGIN_EPISODE
    assert step_row["kind"] == KIND_STEP
    for row in rows:
        assert row["upload_rpc_ms"] >= 0.0
        assert row["backlog_depth"] >= 1
        assert row["oldest_backlog_s"] >= 0.0
        assert row["attempts"] == 1
        assert row["delivered"] is True
        assert isinstance(row["outcome_divergence"], bool)
    assert begin_row["outcome_divergence"] is False
    assert step_row["outcome_divergence"] is True
    assert step_row["transition_id"] == "run:0"
    # The profiler owns "seq"; the uploader's own counter must not shadow it.
    assert [row["seq"] for row in rows] == [0, 1]
    assert [row["upload_seq"] for row in rows] == [0, 1]


def test_profiling_is_off_by_default_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    remote = _Remote(_RecordingServicer(_service(), recorder))
    port = remote.start()
    # env={} is the default in this module's factory; assert the real default
    # too by constructing without it.
    uploader = TransitionUploader(
        f"127.0.0.1:{port}", env={}, backoff_initial_s=0.01, backoff_max_s=0.05
    )
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        payload, summary = episode.step()
        uploader.enqueue_step(payload, local_outcome_summary=summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()

    assert list(tmp_path.rglob("*.jsonl")) == []
    assert uploader.uploaded_count == 2


def test_retry_time_accumulates_into_one_upload_rpc_ms(tmp_path):
    path = tmp_path / "uploader.jsonl"
    profiler = LatencyProfiler(LATENCY_ROLE, path)
    recorder = _Recorder()
    recorder.faults_after.append(
        (KIND_STEP, grpc.StatusCode.DEADLINE_EXCEEDED, "slow link")
    )
    remote = _Remote(_RecordingServicer(_service(), recorder))
    port = remote.start()
    uploader = _uploader(port, profiler=profiler)
    episode = _EpisodeRequests()
    try:
        uploader.start()
        uploader.enqueue_begin_episode(episode.begin())
        payload, summary = episode.step()
        uploader.enqueue_step(payload, local_outcome_summary=summary)
        _drain(uploader)
    finally:
        uploader.stop(drain=True)
        remote.stop()
        profiler.close()

    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    step_row = rows[-1]
    assert step_row["attempts"] == 2
    assert step_row["delivered"] is True
    assert step_row["upload_rpc_ms"] > 0.0
