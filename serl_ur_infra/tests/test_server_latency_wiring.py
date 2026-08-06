"""Server-side latency profiling: what a Step RPC leaves behind, and when.

The sink itself is tested in ``test_latency_profile.py``.  This file is about
the WIRING -- that the phases named in the design spec are attached to the code
that actually performs them, that the queue-pressure gauge counts real overlap,
that a dead classifier is visible in the file, and above all that a server
started WITHOUT ``HIL_LATENCY_PROFILE`` behaves exactly as it did before this
existed and writes nothing at all.

No robot, no learner, no gRPC server: the servicer is driven directly with
protobuf requests, which is the only way to make the concurrency test
deterministic (a barrier inside a fake service) rather than a race the machine
may or may not lose.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    ActionResult,
    ActorSessionService,
    StepResult,
    TransitionAck,
    TransitionOutcome,
)
from ur_env.classifier_sidecar import (  # noqa: E402
    CLASSIFIER_SIDECAR_KEY,
    build_sidecar,
)
from ur_env.grpc_actor_transport import (  # noqa: E402
    GrpcActorServicer,
    data_to_proto,
    observation_to_proto,
)
from ur_env.actor_network import ObservationPacket  # noqa: E402
from ur_env.proto import actor_transport_pb2 as pb  # noqa: E402
from ur_env.remote_actor import build_data  # noqa: E402
from ur_env.rlpd_receive_server import (  # noqa: E402
    RewardTransitionFinalizer,
    ScriptedRewardClassifierRuntime,
)
from ur_env.server_latency import (  # noqa: E402
    BEGIN_EPISODE_RPC,
    STEP_RPC,
    ServerLatencyProbe,
    wrap_ingress_sink,
)


_ENABLED_ENV = {"HIL_LATENCY_PROFILE": "1"}
_DISABLED_ENV: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# fixtures shared with the existing server-path suites
# --------------------------------------------------------------------------- #


def _observation(value: int) -> dict:
    """The three canonical keys, sized like the real policy observation."""

    return {
        "state": np.full((1, 19), value / 100.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _sidecar(seed: int = 0) -> dict:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, seed % 3] = (seed * 11) % 256
    return build_sidecar({"cam1": frame, "cam2": frame})


def _observation_with_sidecar(value: int) -> dict:
    observation = _observation(value)
    observation[CLASSIFIER_SIDECAR_KEY] = _sidecar(value)
    return observation


def _data(action, *, step_id: int = 0, intervened: bool = False):
    info = (
        {
            "intervened": 1,
            "intervene_action": np.full(7, -0.25, dtype=np.float32),
            "grasp_penalty": -0.1,
        }
        if intervened
        else {}
    )
    data = build_data(
        actor_id="actor",
        run_id="run",
        session_id="session",
        transition_id=f"run:{step_id}",
        env_step=step_id,
        # meta.timestamp_ns must be the SOURCE observation's timestamp, and
        # observation "oN" is stamped 1_000 + N throughout this file.
        timestamp_ns=1_000 + step_id,
        policy_version=0,
        policy_action=action,
        # 0: ActorSessionService requires the first episode of a run to be 0,
        # and BeginEpisode above opened exactly that one.
        episode_id=0,
        step_id=step_id,
        observation_id=f"o{step_id}",
        next_observation_id=f"o{step_id + 1}",
        reward=0.0,
        done=False,
        truncated=False,
        info=info,
    )
    data["meta"]["auto_success"] = True
    data["meta"]["operator_success"] = False
    return data


def _begin_request(request_id: int = 1) -> pb.BeginEpisodeRequest:
    return pb.BeginEpisodeRequest(
        protocol_version=PROTOCOL_VERSION,
        actor_id="actor",
        run_id="run",
        session_id="session",
        episode_id=0,
        request_id=request_id,
        created_monotonic_ns=10_000,
        observation=observation_to_proto(
            ObservationPacket("o0", 1_000, _observation(0))
        ),
        deterministic=False,
    )


def _step_request(
    action,
    *,
    request_id: int,
    step_id: int = 0,
    with_sidecar: bool = True,
    intervened: bool = False,
    protocol_version: str = PROTOCOL_VERSION,
) -> pb.StepRequest:
    next_value = step_id + 1
    observation = (
        _observation_with_sidecar(next_value)
        if with_sidecar
        else _observation(next_value)
    )
    return pb.StepRequest(
        protocol_version=protocol_version,
        actor_id="actor",
        run_id="run",
        session_id="session",
        request_id=request_id,
        created_monotonic_ns=20_000 + request_id,
        data=data_to_proto(
            _data(action, step_id=step_id, intervened=intervened)
        ),
        next_observation=observation_to_proto(
            ObservationPacket(f"o{next_value}", 1_000 + next_value, observation)
        ),
        request_action=True,
        deterministic=False,
    )


class _Context:
    """A gRPC context whose abort() is an immediate, loud test failure."""

    def __init__(self) -> None:
        self.aborted: list[tuple] = []

    def abort(self, code, details):  # noqa: D401 - gRPC's own signature
        self.aborted.append((code, details))
        raise AssertionError(f"handler aborted: {code} {details}")


class _RecordingIngress:
    """A sink shaped like FaultGatedReplayIngress: accepts, and primes.

    ``prime_observation`` returning None is the receive-only shape (no shared
    trunk feature), which keeps the classifier on its sidecar pixels -- exactly
    the path a server without the frozen-trunk extractor takes.
    """

    def __init__(self, *, prime_delay_s: float = 0.0) -> None:
        self.accepted: list[tuple] = []
        self.primed: list[str] = []
        self._prime_delay_s = prime_delay_s

    def __call__(self, data, intervened) -> None:
        self.accepted.append((data["meta"]["transition_id"], bool(intervened)))

    def prime_observation(self, **kwargs):
        if self._prime_delay_s:
            time.sleep(self._prime_delay_s)
        self.primed.append(kwargs["observation_id"])
        return None

    def status(self):  # pragma: no cover - only for buffer_status_provider
        raise AssertionError("status() is not part of this test")


def _service(probe, *, classifier=None, ingress=None) -> ActorSessionService:
    """Compose the service the way ``build_actor_service`` composes it."""

    classifier = classifier or ScriptedRewardClassifierRuntime([0.1] * 8)
    ingress = ingress if ingress is not None else _RecordingIngress()
    return ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        reward_authority="server_classifier",
        reward_model_id=classifier.reward_model_id,
        finalize_transition=RewardTransitionFinalizer(
            classifier, warn=lambda message: None, latency_probe=probe
        ),
        accept_data=wrap_ingress_sink(ingress, probe),
    )


def _run_one_step(servicer, context, *, request_id: int = 2):
    """BeginEpisode + one Step, the shortest complete production round."""

    begin = servicer.BeginEpisode(_begin_request(), context)
    action = np.asarray(begin.action, dtype=np.float32)
    return servicer.Step(
        _step_request(action, request_id=request_id, intervened=True), context
    )


def _records(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _enabled_probe(tmp_path, name: str = "latency_server.jsonl"):
    return ServerLatencyProbe.from_env(tmp_path / name, env=_ENABLED_ENV)


# --------------------------------------------------------------------------- #
# enabled: the phases exist where the work is
# --------------------------------------------------------------------------- #


def test_step_record_carries_every_mapped_phase_and_the_join_key(tmp_path):
    probe = _enabled_probe(tmp_path)
    ingress = _RecordingIngress()
    servicer = GrpcActorServicer(
        _service(probe, ingress=ingress), latency_probe=probe
    )
    context = _Context()

    result = _run_one_step(servicer, context)
    probe.close()

    assert result.ack.accepted
    assert ingress.accepted == [("run:0", True)]
    records = _records(probe.path)
    assert [record["rpc"] for record in records] == [
        BEGIN_EPISODE_RPC,
        STEP_RPC,
    ]
    step = records[1]

    # The join key the analyzer needs, plus the ids that make a row readable
    # without one.
    assert step["transition_id"] == "run:0"
    assert step["request_id"] == 2
    assert step["run_id"] == "run"
    assert step["episode_id"] == 0
    assert step["env_step"] == 0
    assert step["intervened"] is True
    assert step["accepted"] is True
    assert step["deduplicated"] is False
    assert step["terminal"] is False
    assert step["schema"] == 1
    assert step["role"] == "server"
    assert step["seq"] == 1
    assert step["t_epoch"] > 0.0

    for phase in (
        "total_ms",
        "request_decode_ms",
        "service_step_ms",
        "response_build_ms",
        "trunk_encode_ms",
        "reward_finalize_ms",
        "classifier_ms",
        "replay_insert_ms",
    ):
        assert phase in step, f"missing phase {phase}: {sorted(step)}"
        assert step[phase] >= 0.0

    # policy_inference_ms is NOT re-timed here: it is the number the service
    # already measured and shipped to the actor in the same reply.
    assert step["policy_inference_ms"] == pytest.approx(
        result.action.server_inference_ms, abs=1e-3
    )

    # The phases are nested, not disjoint, and the file must be readable that
    # way.  This is the invariant that catches a timer attached to the wrong
    # region.
    assert step["classifier_ms"] <= step["reward_finalize_ms"] + 1e-6
    assert step["reward_finalize_ms"] <= step["service_step_ms"] + 1e-6
    assert step["trunk_encode_ms"] <= step["service_step_ms"] + 1e-6
    assert step["replay_insert_ms"] <= step["service_step_ms"] + 1e-6
    assert step["service_step_ms"] <= step["total_ms"] + 1e-6
    assert step["request_decode_ms"] <= step["total_ms"] + 1e-6


def test_trunk_encode_phase_measures_the_priming_call(tmp_path):
    """The phase must follow prime_observation, not merely exist."""

    probe = _enabled_probe(tmp_path)
    ingress = _RecordingIngress(prime_delay_s=0.02)
    servicer = GrpcActorServicer(
        _service(probe, ingress=ingress), latency_probe=probe
    )

    _run_one_step(servicer, _Context())
    probe.close()

    step = _records(probe.path)[1]
    # Two primings happen per round (BeginEpisode's observation and the Step's
    # next observation) but only the Step's is inside this record's scope.
    assert ingress.primed == ["o0", "o1"]
    assert step["trunk_encode_ms"] >= 18.0
    assert step["service_step_ms"] >= step["trunk_encode_ms"]


def test_begin_episode_record_is_total_only(tmp_path):
    probe = _enabled_probe(tmp_path)
    servicer = GrpcActorServicer(_service(probe), latency_probe=probe)

    servicer.BeginEpisode(_begin_request(), _Context())
    probe.close()

    (begin,) = _records(probe.path)
    assert begin["rpc"] == BEGIN_EPISODE_RPC
    assert begin["run_id"] == "run"
    assert begin["episode_id"] == 0
    assert begin["request_id"] == 1
    assert begin["total_ms"] >= 0.0
    # Deliberately lightweight: no transition to key on, no concurrency gauge
    # (that one counts Step handlers only), and none of the Step breakdown.
    assert "concurrent_rpcs" not in begin
    assert "transition_id" not in begin
    for absent in (
        "request_decode_ms",
        "service_step_ms",
        "response_build_ms",
        "reward_finalize_ms",
        "classifier_ms",
        "replay_insert_ms",
        "policy_inference_ms",
    ):
        assert absent not in begin
    # trunk_encode IS here, and truthfully so: BeginEpisode primes its
    # observation through the same sink, and that encode is real cost of this
    # RPC.  The Step record must not be charged for it.
    assert begin["trunk_encode_ms"] >= 0.0


def test_unclassified_step_reports_no_classifier_phase(tmp_path):
    """Most transitions carry no sidecar; their rows must say so by absence."""

    probe = _enabled_probe(tmp_path)
    servicer = GrpcActorServicer(_service(probe), latency_probe=probe)
    context = _Context()

    begin = servicer.BeginEpisode(_begin_request(), context)
    servicer.Step(
        _step_request(
            np.asarray(begin.action, dtype=np.float32),
            request_id=2,
            with_sidecar=False,
        ),
        context,
    )
    probe.close()

    step = _records(probe.path)[1]
    assert "classifier_ms" not in step
    assert step["classifier_evaluated"] is False
    assert "reward_finalize_ms" in step


# --------------------------------------------------------------------------- #
# the concurrency gauge
# --------------------------------------------------------------------------- #


class _BarrierService:
    """A service whose step() parks until two handlers are inside it."""

    def __init__(self, parties: int) -> None:
        self._barrier = threading.Barrier(parties, timeout=5.0)
        self.calls = 0

    def step(self, command) -> StepResult:
        self._barrier.wait()
        self.calls += 1
        transition_id = str(command.data["meta"]["transition_id"])
        return StepResult(
            ack=TransitionAck(
                accepted=True,
                transition_id=transition_id,
                session_id=command.session_id,
                request_id=command.request_id,
            ),
            outcome=TransitionOutcome(
                transition_id=transition_id,
                reward=0.0,
                mask=1.0,
                done=False,
                truncated=False,
                success=False,
                classifier_evaluated=False,
            ),
            action=ActionResult(
                action=np.zeros(7, np.float32),
                policy_version=0,
                session_id=command.session_id,
                request_id=command.request_id,
                request_created_monotonic_ns=command.created_monotonic_ns,
                observation_id=command.next_observation.observation_id,
                server_inference_ms=1.5,
            ),
        )


def test_concurrent_rpcs_counts_overlapping_step_handlers(tmp_path):
    probe = _enabled_probe(tmp_path)
    servicer = GrpcActorServicer(_BarrierService(2), latency_probe=probe)
    action = np.zeros(7, np.float32)
    requests = [
        _step_request(action, request_id=2, step_id=0),
        _step_request(action, request_id=3, step_id=1),
    ]
    errors: list[BaseException] = []

    def call(index: int) -> None:
        try:
            servicer.Step(requests[index], _Context())
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(index,)) for index in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    probe.close()

    assert not errors
    records = _records(probe.path)
    assert len(records) == 2
    # The gauge is read AT ENTRY and includes the handler reporting it, so the
    # barrier makes exactly one row say 1 (it arrived alone) and one say 2 (it
    # arrived while the first was still inside).  Rows land in commit order,
    # which is not entry order, hence sorted().
    assert sorted(record["concurrent_rpcs"] for record in records) == [1, 2]
    assert probe.max_concurrent_steps == 2
    # And the counter comes back down, or every later row would inherit a
    # permanently inflated queue depth.
    assert probe.active_steps == 0


def test_sequential_steps_report_a_gauge_of_one(tmp_path):
    probe = _enabled_probe(tmp_path)
    servicer = GrpcActorServicer(_service(probe), latency_probe=probe)
    context = _Context()

    begin = servicer.BeginEpisode(_begin_request(), context)
    action = np.asarray(begin.action, dtype=np.float32)
    for index in range(3):
        servicer.Step(
            _step_request(action, request_id=2 + index, step_id=index),
            context,
        )
    probe.close()

    steps = [record for record in _records(probe.path) if record["rpc"] == STEP_RPC]
    assert [record["concurrent_rpcs"] for record in steps] == [1, 1, 1]
    assert probe.max_concurrent_steps == 1


# --------------------------------------------------------------------------- #
# disabled: nothing happens
# --------------------------------------------------------------------------- #


def _normalized_reply(reply: pb.StepReply) -> bytes:
    """Serialize a StepReply with the one wall-clock field zeroed."""

    copy = pb.StepReply()
    copy.CopyFrom(reply)
    copy.action.server_inference_ms = 0.0
    return copy.SerializeToString(deterministic=True)


def test_disabled_probe_writes_nothing_and_replies_identically(tmp_path):
    out_dir = tmp_path / "quiet"
    out_dir.mkdir()
    disabled = ServerLatencyProbe.from_env(
        out_dir / "latency_server.jsonl", env=_DISABLED_ENV
    )
    enabled = _enabled_probe(tmp_path)

    assert disabled.enabled is False
    assert disabled.path is None

    quiet_servicer = GrpcActorServicer(_service(disabled), latency_probe=disabled)
    loud_servicer = GrpcActorServicer(_service(enabled), latency_probe=enabled)
    quiet = _run_one_step(quiet_servicer, _Context())
    loud = _run_one_step(loud_servicer, _Context())
    disabled.close()
    enabled.close()

    # Byte-identical replies: profiling observes the handler, it never edits it.
    assert _normalized_reply(quiet) == _normalized_reply(loud)
    # Not one filesystem entry: no directory creation, no empty file, nothing
    # to clean up on a server that never asked to be profiled.
    assert list(out_dir.iterdir()) == []
    assert _records(enabled.path)


def test_servicer_without_a_probe_still_serves(tmp_path):
    """The default argument is the production default: no probe at all."""

    servicer = GrpcActorServicer(_service(None))

    result = _run_one_step(servicer, _Context())

    assert result.ack.accepted
    assert list(tmp_path.iterdir()) == []


def test_disabled_path_never_reads_the_profiling_clock(tmp_path, monkeypatch):
    """"Near-zero overhead" stated as a fact, not as a hope.

    The sink's ``time`` module is swapped for a counting proxy, so this counts
    only clock reads the PROFILER makes -- the service's own
    ``server_inference_ms`` timer uses its own reference and is invisible here.
    """

    import ur_env.latency_profile as latency_profile

    reads = {"count": 0}
    real = latency_profile.time

    def perf_counter() -> float:
        reads["count"] += 1
        return real.perf_counter()

    monkeypatch.setattr(
        latency_profile,
        "time",
        SimpleNamespace(
            perf_counter=perf_counter, time=real.time, monotonic=real.monotonic
        ),
    )

    disabled = ServerLatencyProbe.from_env(
        tmp_path / "unused.jsonl", env=_DISABLED_ENV
    )
    _run_one_step(
        GrpcActorServicer(_service(disabled), latency_probe=disabled), _Context()
    )
    assert reads["count"] == 0

    # The same round with profiling on proves the counter was watching the
    # right clock all along.
    enabled = _enabled_probe(tmp_path)
    _run_one_step(
        GrpcActorServicer(_service(enabled), latency_probe=enabled), _Context()
    )
    enabled.close()
    assert reads["count"] > 0


def test_wrap_ingress_sink_is_identity_when_profiling_is_off():
    ingress = _RecordingIngress()
    disabled = ServerLatencyProbe.from_env(env=_DISABLED_ENV)

    assert wrap_ingress_sink(ingress, disabled) is ingress
    assert wrap_ingress_sink(ingress, None) is ingress


def test_wrap_ingress_sink_forwards_prime_observation(tmp_path):
    """Load-bearing: the service DISCOVERS priming with getattr.

    A wrapper that forgot ``prime_observation`` would not raise -- it would
    silently stop sharing the frozen-trunk feature, so the encoder would run
    twice per step and the classifier would score different pixels than the
    policy.
    """

    probe = _enabled_probe(tmp_path)
    ingress = _RecordingIngress()

    wrapped = wrap_ingress_sink(ingress, probe)

    assert wrapped is not ingress
    assert callable(getattr(wrapped, "prime_observation", None))
    assert wrapped.prime_observation(observation_id="o7") is None
    assert ingress.primed == ["o7"]

    # A sink that does NOT prime must not GROW the attribute either: the
    # service reads its presence as "this ingress shares trunk features", so an
    # unconditional wrapper would flip a receive-only server into the shared
    # path and hand the policy a feature nothing computed.
    plain_wrapped = wrap_ingress_sink(lambda data, intervened: None, probe)
    assert getattr(plain_wrapped, "prime_observation", None) is None


# --------------------------------------------------------------------------- #
# the classifier is dead: say so in the file
# --------------------------------------------------------------------------- #


def test_classifier_degraded_flag_appears_once_the_runtime_faults(tmp_path):
    """A latency file with classifier_ms ~ 0 must distinguish its two causes.

    ``ScriptedRewardClassifierRuntime`` latches itself not-ready on its first
    failure, exactly like ``RewardClassifierRuntime`` does in production, so
    every later scored step is degraded too.
    """

    probe = _enabled_probe(tmp_path)
    classifier = ScriptedRewardClassifierRuntime([RuntimeError("cuda fell over")])
    servicer = GrpcActorServicer(
        _service(probe, classifier=classifier), latency_probe=probe
    )
    context = _Context()

    begin = servicer.BeginEpisode(_begin_request(), context)
    action = np.asarray(begin.action, dtype=np.float32)
    first = servicer.Step(_step_request(action, request_id=2), context)
    second = servicer.Step(
        _step_request(action, request_id=3, step_id=1), context
    )
    probe.close()

    # The session survived the fault: transitions are degraded to unclassified,
    # they do not end the run (commit 4a14c30).
    assert first.ack.accepted and second.ack.accepted
    assert first.outcome.classifier_evaluated is False

    steps = [record for record in _records(probe.path) if record["rpc"] == STEP_RPC]
    assert len(steps) == 2
    for step in steps:
        assert step["classifier_degraded"] is True
        assert step["classifier_evaluated"] is False
        assert "classifier_ms" in step


def test_healthy_classifier_leaves_the_degraded_flag_off(tmp_path):
    probe = _enabled_probe(tmp_path)
    servicer = GrpcActorServicer(_service(probe), latency_probe=probe)

    _run_one_step(servicer, _Context())
    probe.close()

    step = _records(probe.path)[1]
    assert "classifier_degraded" not in step
    assert step["classifier_evaluated"] is True


# --------------------------------------------------------------------------- #
# failure paths
# --------------------------------------------------------------------------- #


def test_failed_step_still_commits_a_row_naming_the_error(tmp_path):
    """The handler that raised is exactly the row an investigation wants."""

    probe = _enabled_probe(tmp_path)
    servicer = GrpcActorServicer(_service(probe), latency_probe=probe)
    context = _Context()

    servicer.BeginEpisode(_begin_request(), context)
    with pytest.raises(AssertionError, match="handler aborted"):
        servicer.Step(
            _step_request(
                np.zeros(7, np.float32),
                request_id=2,
                protocol_version="v0-not-a-protocol",
            ),
            context,
        )
    probe.close()

    step = _records(probe.path)[1]
    assert step["rpc"] == STEP_RPC
    # Ids are read off the request before anything can fail, so a decode-time
    # death is still joinable against the actor's row for the same transition.
    assert step["transition_id"] == "run:0"
    assert step["error"]
    assert step["total_ms"] >= 0.0
    assert "service_step_ms" not in step


def test_a_broken_sink_degrades_profiling_without_touching_the_reply(tmp_path):
    """An unwritable path must cost a warning, never a transition."""

    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not a directory\n")
    probe = ServerLatencyProbe.from_env(
        blocked / "logs" / "latency_server.jsonl", env=_ENABLED_ENV
    )
    servicer = GrpcActorServicer(_service(probe), latency_probe=probe)

    result = _run_one_step(servicer, _Context())
    probe.close()

    assert result.ack.accepted
    assert probe.profiler.degraded is True


def test_phase_outside_a_handler_is_dropped(tmp_path):
    """The learner thread also calls into these objects; it must not emit rows."""

    probe = _enabled_probe(tmp_path)

    with probe.phase("replay_insert"):
        pass
    probe.set("transition_id", "never-happened")
    probe.close()

    assert not os.path.exists(probe.path)


# --------------------------------------------------------------------------- #
# composition: the probe reaches the two collaborators
# --------------------------------------------------------------------------- #


def test_build_actor_service_wires_the_probe_into_finalizer_and_sink(tmp_path):
    from ur_env.learner.composition import build_actor_service

    probe = _enabled_probe(tmp_path, "composed.jsonl")
    ingress = _RecordingIngress()
    classifier = ScriptedRewardClassifierRuntime([0.1] * 4)
    assembly = SimpleNamespace(
        policy_runtime=_CallablePolicy(), ingress=ingress
    )

    service = build_actor_service(
        assembly=assembly, classifier=classifier, latency_probe=probe
    )
    servicer = GrpcActorServicer(service, latency_probe=probe)

    _run_one_step(servicer, _Context())
    probe.close()

    step = _records(probe.path)[1]
    for phase in ("reward_finalize_ms", "classifier_ms", "replay_insert_ms"):
        assert phase in step, f"{phase} did not survive build_actor_service"
    assert ingress.accepted == [("run:0", True)]
    assert ingress.primed == ["o0", "o1"]


def test_build_actor_service_without_a_probe_leaves_the_ingress_bare():
    from ur_env.learner.composition import build_actor_service

    ingress = _RecordingIngress()
    classifier = ScriptedRewardClassifierRuntime([0.1] * 4)
    assembly = SimpleNamespace(policy_runtime=_CallablePolicy(), ingress=ingress)

    service = build_actor_service(assembly=assembly, classifier=classifier)

    # Not a wrapper: a server that is not profiling keeps the call graph it had
    # before this feature existed.
    assert service._accept_data is ingress


class _CallablePolicy:
    model_id = "test-policy"

    def __call__(self, observation, deterministic):
        return np.zeros(7, np.float32), 0


# --------------------------------------------------------------------------- #
# the launcher: an env var, never an argv token
# --------------------------------------------------------------------------- #
#
# run_hil_server.sh re-validates a RUNNING learner against the exact command it
# would have launched, comparing argv values by string equality through /proc.
# Server-side opt-in therefore has to ride on the environment: one extra argv
# token would make every profiled learner fail its own reuse check.  These
# tests expand the launcher's own nohup block (the helpers belong to
# test_hil_server_wandb_mode.py, which models that block already) rather than
# restating what it should contain.


sys.path.insert(0, _HERE)


def test_launcher_forwards_the_env_var_and_leaves_argv_byte_identical(tmp_path):
    from test_hil_server_wandb_mode import (
        _expand_launch_command,
        _rig,
        _split_env_prefix,
    )

    rig = _rig(tmp_path, wandb_mode="offline")
    assert rig["LATENCY_PROFILE"] == ""

    off_env, off_argv = _split_env_prefix(_expand_launch_command(tmp_path, rig))
    on_env, on_argv = _split_env_prefix(
        _expand_launch_command(tmp_path, dict(rig, LATENCY_PROFILE="1"))
    )

    # THE constraint: not one token added, removed, reordered or rewritten.
    assert on_argv == off_argv
    assert "HIL_LATENCY_PROFILE" not in off_env
    assert on_env == {**off_env, "HIL_LATENCY_PROFILE": "1"}


def test_launcher_marshals_the_value_to_the_learner_host_only_when_set(tmp_path):
    from test_hil_server_wandb_mode import _run_launcher

    def positionals(result) -> list[str]:
        tokens = [
            token.decode()
            for token in result.ssh_log.read_bytes().split(b"\0")
            if token
        ]
        return tokens[tokens.index("--") + 1 :]

    unset_dir = tmp_path / "unset"
    on_dir = tmp_path / "on"
    unset_dir.mkdir()
    on_dir.mkdir()

    # Pinned, not incidental: the launcher's default run id is
    # `cube_in_cup_real_$(date -u +...%H%M%S)`, so two invocations that straddle
    # a second boundary disagree in positional 4 and the comparison below fails
    # for a reason that has nothing to do with latency profiling.  Measured at
    # roughly one run in ten before it was pinned.
    fixed = {"HIL_RUN_ID": "cube_in_cup_real_20260101_000000"}

    unset = _run_launcher(unset_dir, "--check", **fixed)
    assert unset.returncode == 3, unset.stderr
    # 14 positionals is the pre-existing contract; an unset profile adds none,
    # because ssh joins these into one command line and an empty trailing
    # argument does not survive the trip (hence ${15:-} on the remote side).
    assert len(positionals(unset)) == 14

    on = _run_launcher(on_dir, "--check", HIL_LATENCY_PROFILE="1", **fixed)
    assert on.returncode == 3, on.stderr
    assert positionals(on) == positionals(unset) + ["1"]


def test_launcher_refuses_to_marshal_a_value_that_would_resplit(tmp_path):
    """These positionals cross an ssh command line; whitespace would split."""

    from test_hil_server_wandb_mode import _run_launcher

    result = _run_launcher(
        tmp_path, "--check", HIL_LATENCY_PROFILE="1 ; touch /tmp/pwned"
    )

    assert result.returncode == 3, result.stderr
    assert "ignoring HIL_LATENCY_PROFILE" in result.stderr
    tokens = [
        token.decode()
        for token in result.ssh_log.read_bytes().split(b"\0")
        if token
    ]
    assert len(tokens[tokens.index("--") + 1 :]) == 14
    assert not os.path.exists("/tmp/pwned")
