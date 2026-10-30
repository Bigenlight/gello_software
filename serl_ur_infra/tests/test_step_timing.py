"""Unit contract for ``ur_env/step_timing.py`` -- the OPT-IN step instrumentation.

WHAT THIS MODULE IS AND WHY IT NEEDS ITS OWN TEST FILE
------------------------------------------------------
``step_timing`` sits in the request path of a live UR7e evaluation session.
Everything it does is measurement, so every property worth pinning is a
NEGATIVE one: it must not change what the server returns, must not raise when
the disk is full, must not spam a session's stderr, and -- the sharp one --
must not shadow an attribute the server probes for with ``getattr``.

Three of those are easy to get wrong in a way no smoke test would catch:

* **``TimingSink`` must refuse to wrap a sink that defines
  ``prime_observation``.**  The server decides whether to hand the policy
  feature-primed observations with ``getattr(sink, "prime_observation", None)``
  (the same probe ``tests/test_bc_inference_log.py`` and
  ``tests/test_bc_server_inproc.py`` pin from the other side).  A transparent
  wrapper that forgot to forward it would silently serve RAW pixels to a policy
  trained on primed features -- a wrong-but-running rollout, the most expensive
  failure mode this stack has.  Refusing at CONSTRUCTION is the only place the
  mistake is still cheap.
* **``ServiceStepTimingProxy`` must implement exactly five methods and define
  no ``__getattr__``.**  A catch-all would make a servicer that grew a sixth
  RPC keep working -- untimed and unnoticed.  ``AttributeError`` is the desired
  behaviour, not a gap.
* **The writer must fail OPEN and stay quiet.**  A full disk costs a log line,
  never an episode with the arm mid-reach, and the warning is printed once per
  instance so a failing disk cannot bury the operator's real console output.

The frozen server record schema is restated here as ``SERVER_RECORD_KEYS`` and
asserted as an EXACT key set: the two consumers of ``timing.jsonl``
(``scripts/analyze_bc_rollout.py`` and a human reading it) both key off these
names, so an added or renamed field is a contract change that should be seen
here first.

No network, no grpc import, no jax: the stand-ins for ``StepCommand`` /
``StepResult`` are ``SimpleNamespace`` objects with the attribute shapes the
proxy reads, which is also proof the proxy is duck-typed rather than bound to
``ur_env.actor_network``.

Run (from ``/home/laptop3/gello_software``)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
      -p no:cacheprovider serl_ur_infra/tests/test_step_timing.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

_HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(_HERE.parent))

from ur_env.step_timing import (  # noqa: E402
    SERVER_TIMING_FILENAME,
    STEP_TIMING_ENV,
    FailOpenJsonlWriter,
    ServiceStepTimingProxy,
    StepTimingRecorder,
    TimingSink,
    step_timing_enabled,
)


#: The frozen server-side record.  Restated, not imported: this list is the
#: contract, and a test that read it from the module under test would agree
#: with any change the module made to itself.
SERVER_RECORD_KEYS = frozenset(
    {
        "ts",
        "kind",
        "run_id",
        "episode_id",
        "step_id",
        "env_step",
        "transition_id",
        "handler_ms",
        "infer_ms",
        "sink_ms",
        "overhead_ms",
        "terminal",
        "deduplicated",
        "error",
    }
)

#: Long enough that ``handler_ms`` cannot round to zero on any clock, short
#: enough that the whole file stays well under a second.
_WORK_S = 0.003


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _lines(path: Path) -> list[dict[str, Any]]:
    """Every JSON object in a jsonl file, in order."""

    assert path.is_file(), f"expected a jsonl file at {path}"
    out = []
    for number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        assert raw.strip(), f"{path}:{number} is blank; one record per line"
        out.append(json.loads(raw))
    return out


def _writer(tmp_path: Path, name: str = "timing.jsonl") -> FailOpenJsonlWriter:
    return FailOpenJsonlWriter(tmp_path / name)


def _step_command(
    *,
    run_id: str = "run-a",
    episode_id: int = 3,
    step_id: int = 7,
    env_step: int = 11,
    transition_id: str = "run-a:11",
) -> SimpleNamespace:
    """A ``StepCommand`` stand-in: only the fields the proxy reads.

    Field placement mirrors ``actor_network.StepCommand`` -- ``env_step`` and
    ``transition_id`` live in ``data["meta"]``, ``episode_id`` and ``step_id``
    in ``data["transition"]`` -- because that split is exactly what the
    extraction has to get right.
    """

    return SimpleNamespace(
        run_id=run_id,
        data={
            "meta": {"env_step": env_step, "transition_id": transition_id},
            "transition": {"episode_id": episode_id, "step_id": step_id},
        },
    )


def _step_result(
    *,
    inference_ms: float | None = 6.5,
    terminal: bool = False,
    deduplicated: bool = False,
) -> SimpleNamespace:
    """A ``StepResult`` stand-in.  ``action=None`` is the terminal shape."""

    action = (
        None
        if inference_ms is None
        else SimpleNamespace(server_inference_ms=inference_ms)
    )
    return SimpleNamespace(
        ack=SimpleNamespace(deduplicated=deduplicated),
        outcome=SimpleNamespace(terminal=terminal),
        action=action,
    )


class _StubService:
    """The five-method service surface, with recording counters.

    ``before_return`` is a hook the sink test uses to make the sink run INSIDE
    the handler span, which is the only arrangement in which ``sink_ms`` is a
    piece of ``handler_ms`` rather than a number from a different request.
    """

    def __init__(
        self,
        *,
        result: Any = None,
        raises: BaseException | None = None,
        before_return: Any = None,
        work_s: float = _WORK_S,
    ) -> None:
        self.result = result
        self.raises = raises
        self.before_return = before_return
        self.work_s = work_s
        self.calls: list[tuple[str, Any]] = []
        self.health_value = (True, True, "stub is ready")
        self.server_info_value = SimpleNamespace(ready=True, model_id="stub-model")
        self.buffer_status_value = SimpleNamespace(replay_size=17)

    def health(self):
        self.calls.append(("health", None))
        return self.health_value

    def get_server_info(self):
        self.calls.append(("get_server_info", None))
        return self.server_info_value

    def get_buffer_status(self):
        self.calls.append(("get_buffer_status", None))
        return self.buffer_status_value

    def begin_episode(self, command):
        self.calls.append(("begin_episode", command))
        time.sleep(self.work_s)
        if self.raises is not None:
            raise self.raises
        return self.result

    def step(self, command):
        self.calls.append(("step", command))
        time.sleep(self.work_s)
        if self.before_return is not None:
            self.before_return(command)
        if self.raises is not None:
            raise self.raises
        return self.result


class _PrimingSink:
    """A sink that DOES define ``prime_observation`` -- the forbidden inner."""

    def __call__(self, data, intervened):
        return None

    def prime_observation(self, observation):
        return observation


class _RecordingSink:
    """An ordinary sink: callable, and deliberately without priming."""

    def __init__(self, *, returns: Any = None, raises: BaseException | None = None):
        self.returns = returns
        self.raises = raises
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, data, intervened):
        self.calls.append((data, intervened))
        time.sleep(_WORK_S)
        if self.raises is not None:
            raise self.raises
        return self.returns


# --------------------------------------------------------------------------- #
# 1. Module constants and the opt-in switch                                     #
# --------------------------------------------------------------------------- #
def test_the_opt_in_names_are_pinned():
    """Both names are typed by operators and read by the analyzer."""

    assert STEP_TIMING_ENV == "HIL_STEP_TIMING"
    assert SERVER_TIMING_FILENAME == "timing.jsonl"


@pytest.mark.parametrize(
    "value",
    ["1", " 1 ", "true", "TRUE", "True", " true\n", "yes", "YES", "Yes"],
)
def test_step_timing_enabled_accepts_the_three_true_spellings(value):
    assert step_timing_enabled(value) is True


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "0",
        "no",
        "NO",
        "off",
        "false",
        "FALSE",
        "2",
        "yes please",
        "truthy",
        "garbage",
    ],
)
def test_step_timing_enabled_is_off_for_everything_else(value):
    """Default OFF is load-bearing: a typo must not instrument a session."""

    assert step_timing_enabled(value) is False


def test_step_timing_enabled_returns_a_real_bool():
    """Callers branch on it; a truthy string would also pass ``if``."""

    assert isinstance(step_timing_enabled("1"), bool)
    assert isinstance(step_timing_enabled("nope"), bool)


# --------------------------------------------------------------------------- #
# 2. FailOpenJsonlWriter                                                        #
# --------------------------------------------------------------------------- #
def test_writer_appends_one_json_object_per_write(tmp_path):
    writer = _writer(tmp_path)
    records = [{"i": index, "kind": "step", "note": f"row {index}"} for index in range(4)]

    for record in records:
        assert writer.write(record) is True
    writer.close()

    assert writer.write_count == len(records)
    assert writer.error_count == 0
    assert _lines(tmp_path / "timing.jsonl") == records


def test_writer_creates_missing_parent_directories_on_first_write(tmp_path):
    """The operator names a path; the directory tree is the writer's problem."""

    path = tmp_path / "deep" / "deeper" / "timing.jsonl"
    writer = FailOpenJsonlWriter(path)

    assert not path.parent.exists(), "construction must touch no filesystem"

    assert writer.write({"kind": "step"}) is True
    writer.close()

    assert path.is_file()
    assert _lines(path) == [{"kind": "step"}]


def test_writer_is_lazy_so_a_session_that_never_writes_leaves_no_file(tmp_path):
    """An enabled-but-idle writer must not litter the record root."""

    path = tmp_path / "timing.jsonl"
    writer = FailOpenJsonlWriter(path)
    writer.close()

    assert not path.exists()
    assert writer.write_count == 0
    assert writer.error_count == 0


def test_writer_appends_rather_than_truncating_an_existing_file(tmp_path):
    """A re-run against the same path adds to the record, never destroys it."""

    path = tmp_path / "timing.jsonl"
    first = FailOpenJsonlWriter(path)
    first.write({"run": 1})
    first.close()

    second = FailOpenJsonlWriter(path)
    second.write({"run": 2})
    second.close()

    assert _lines(path) == [{"run": 1}, {"run": 2}]


def test_writer_swallows_io_errors_and_warns_exactly_once(tmp_path, capsys):
    """Fail OPEN: a bad path costs records and one line of stderr, never a raise.

    The unwritable path is a parent that already exists as a FILE, which is the
    cheapest reproduction of "this directory cannot be created" that needs no
    permission games and behaves identically as root.
    """

    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory\n", encoding="utf-8")
    writer = FailOpenJsonlWriter(blocker / "timing.jsonl")
    capsys.readouterr()  # drop anything emitted before the writer existed

    assert writer.write({"kind": "step", "n": 0}) is False
    first = capsys.readouterr().err
    assert first.strip(), "the first loss must be reported on stderr"

    for index in range(1, 5):
        assert writer.write({"kind": "step", "n": index}) is False
    later = capsys.readouterr().err
    assert later == "", (
        "only ONE warning per writer instance: a failing disk must not bury "
        f"the operator's console, got:\n{later}"
    )

    writer.close()  # must not raise either
    assert writer.write_count == 0
    assert writer.error_count == 5, "every lost record has to be counted"


def test_writer_counts_an_unserializable_record_as_a_loss(tmp_path):
    """A record the JSON encoder rejects is a loss, not an exception."""

    writer = _writer(tmp_path)
    assert writer.write({"good": 1}) is True

    assert writer.write({"bad": object()}) is False
    assert writer.error_count == 1

    assert writer.write({"good": 2}) is True
    writer.close()
    assert writer.write_count == 2
    assert _lines(tmp_path / "timing.jsonl") == [{"good": 1}, {"good": 2}]


def test_writer_close_is_idempotent(tmp_path):
    writer = _writer(tmp_path)
    writer.write({"kind": "step"})

    writer.close()
    writer.close()

    assert writer.error_count == 0


# --------------------------------------------------------------------------- #
# 3. StepTimingRecorder                                                         #
# --------------------------------------------------------------------------- #
def test_note_without_an_open_scope_is_a_silent_no_op(tmp_path):
    """A sink that runs outside any request must not decorate the next one."""

    writer = _writer(tmp_path)
    recorder = StepTimingRecorder(writer)

    recorder.note("sink_ms", 99.0)  # no begin() has happened
    recorder.emit({"kind": "step", "handler_ms": 1.0})
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert "sink_ms" not in record or record["sink_ms"] is None
    assert record["handler_ms"] == 1.0


def test_emit_merges_the_notes_taken_since_begin(tmp_path):
    writer = _writer(tmp_path)
    recorder = StepTimingRecorder(writer)

    recorder.begin()
    recorder.note("sink_ms", 2.5)
    recorder.emit({"kind": "step", "handler_ms": 10.0})
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["sink_ms"] == pytest.approx(2.5)
    assert record["handler_ms"] == pytest.approx(10.0)
    assert record["kind"] == "step"


def test_emit_closes_the_scope_so_a_span_is_never_reported_twice(tmp_path):
    """The span belongs to ONE request.  Leaking it would invent a sink call."""

    writer = _writer(tmp_path)
    recorder = StepTimingRecorder(writer)

    recorder.begin()
    recorder.note("sink_ms", 4.0)
    recorder.emit({"kind": "step", "handler_ms": 10.0})
    recorder.emit({"kind": "step", "handler_ms": 11.0})
    writer.close()

    first, second = _lines(tmp_path / "timing.jsonl")
    assert first["sink_ms"] == pytest.approx(4.0)
    assert "sink_ms" not in second or second["sink_ms"] is None


def test_begin_discards_a_stale_scope(tmp_path):
    """A handler that died before emitting must not bleed into the next one."""

    writer = _writer(tmp_path)
    recorder = StepTimingRecorder(writer)

    recorder.begin()
    recorder.note("sink_ms", 7.0)
    recorder.begin()  # previous request never emitted
    recorder.emit({"kind": "step", "handler_ms": 1.0})
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert "sink_ms" not in record or record["sink_ms"] is None


def test_emit_writes_one_line_per_call(tmp_path):
    writer = _writer(tmp_path)
    recorder = StepTimingRecorder(writer)

    for index in range(3):
        recorder.begin()
        recorder.emit({"kind": "step", "n": index})
    writer.close()

    assert [row["n"] for row in _lines(tmp_path / "timing.jsonl")] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# 4. TimingSink                                                                 #
# --------------------------------------------------------------------------- #
def test_timing_sink_delegates_and_returns_the_inner_value(tmp_path):
    sentinel = object()
    inner = _RecordingSink(returns=sentinel)
    recorder = StepTimingRecorder(_writer(tmp_path))
    sink = TimingSink(inner, recorder)
    data = {"meta": {"transition_id": "run:0"}}

    returned = sink(data, True)

    assert returned is sentinel, "the wrapper observes; it must not repair"
    assert inner.calls == [(data, True)]
    assert inner.calls[0][0] is data, "the sink must get the SAME object"


def test_timing_sink_notes_its_span_into_the_open_scope(tmp_path):
    writer = _writer(tmp_path)
    recorder = StepTimingRecorder(writer)
    sink = TimingSink(_RecordingSink(), recorder)

    recorder.begin()
    sink({"meta": {}}, False)
    recorder.emit({"kind": "step", "handler_ms": 100.0})
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["sink_ms"] is not None
    assert record["sink_ms"] > 0.0


def test_timing_sink_refuses_a_sink_that_defines_prime_observation(tmp_path):
    """The whole reason this class validates anything.

    ``ActorSessionService`` probes the sink with
    ``getattr(sink, "prime_observation", None)``.  Wrapping a priming sink in
    something that does not forward it serves RAW pixels to a policy trained on
    primed features -- wrong, and running.  Fail at construction instead.
    """

    recorder = StepTimingRecorder(_writer(tmp_path))

    with pytest.raises(ValueError) as excinfo:
        TimingSink(_PrimingSink(), recorder)

    assert "prime_observation" in str(excinfo.value), (
        "the message must name the attribute so the operator can act on it"
    )


def test_timing_sink_accepts_a_sink_whose_prime_observation_is_not_callable(tmp_path):
    """The guard is about a callable, not about the name existing.

    INTERPRETATION (spec: "if getattr(inner, 'prime_observation', None) is
    callable"): a sink carrying a non-callable attribute of that name is not
    something the server would ever call, so wrapping it hides nothing.
    """

    inner = _RecordingSink()
    inner.prime_observation = None
    recorder = StepTimingRecorder(_writer(tmp_path))

    sink = TimingSink(inner, recorder)  # must not raise

    assert sink({"meta": {}}, False) is None


def test_timing_sink_does_not_advertise_prime_observation(tmp_path):
    """The probe must see nothing -- which is what makes the guard above sound."""

    recorder = StepTimingRecorder(_writer(tmp_path))
    sink = TimingSink(_RecordingSink(), recorder)

    assert getattr(sink, "prime_observation", None) is None
    assert not hasattr(sink, "prime_observation")


def test_timing_sink_has_no_catch_all_getattr(tmp_path):
    """A transparent proxy here would resurrect the priming hazard by accident."""

    recorder = StepTimingRecorder(_writer(tmp_path))
    sink = TimingSink(_RecordingSink(), recorder)

    with pytest.raises(AttributeError):
        getattr(sink, "nonexistent_attr")


def test_timing_sink_lets_the_inner_exception_through(tmp_path):
    """Observation never swallows a sink failure -- the server owns that call."""

    boom = RuntimeError("disk sink exploded")
    inner = _RecordingSink(raises=boom)
    writer = _writer(tmp_path)
    recorder = StepTimingRecorder(writer)
    sink = TimingSink(inner, recorder)

    recorder.begin()
    with pytest.raises(RuntimeError, match="disk sink exploded"):
        sink({"meta": {}}, False)

    # ...and the span it did spend is still attributed to this request.
    recorder.emit({"kind": "step", "handler_ms": 50.0})
    writer.close()
    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["sink_ms"] is not None


# --------------------------------------------------------------------------- #
# 5. ServiceStepTimingProxy -- passthrough surface                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method, attribute",
    [
        ("health", "health_value"),
        ("get_server_info", "server_info_value"),
        ("get_buffer_status", "buffer_status_value"),
    ],
)
def test_cheap_rpcs_pass_straight_through_and_write_nothing(
    tmp_path, method, attribute
):
    """These three are off the control loop; timing them would only add rows."""

    service = _StubService()
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    returned = getattr(proxy, method)()

    assert returned is getattr(service, attribute)
    assert service.calls == [(method, None)]
    assert writer.write_count == 0
    assert not (tmp_path / "timing.jsonl").exists()


def test_the_proxy_implements_exactly_the_five_service_methods(tmp_path):
    """No ``__getattr__``: a servicer that grew a sixth call must fail loudly."""

    proxy = ServiceStepTimingProxy(_StubService(), StepTimingRecorder(_writer(tmp_path)))

    for name in (
        "health",
        "get_server_info",
        "get_buffer_status",
        "begin_episode",
        "step",
    ):
        assert callable(getattr(proxy, name)), f"{name} must be implemented"

    with pytest.raises(AttributeError):
        getattr(proxy, "some_future_rpc")


# --------------------------------------------------------------------------- #
# 6. ServiceStepTimingProxy -- the step record                                   #
# --------------------------------------------------------------------------- #
def test_step_emits_one_record_carrying_the_whole_frozen_schema(tmp_path):
    service = _StubService(
        result=_step_result(inference_ms=6.5, terminal=False, deduplicated=False)
    )
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))
    command = _step_command()

    returned = proxy.step(command)
    writer.close()

    assert returned is service.result, "the proxy observes; it must not repair"
    (record,) = _lines(tmp_path / "timing.jsonl")
    assert set(record) == set(SERVER_RECORD_KEYS), (
        "the timing.jsonl record shape is frozen; reconcile any added or "
        f"renamed field.  extra={sorted(set(record) - SERVER_RECORD_KEYS)} "
        f"missing={sorted(SERVER_RECORD_KEYS - set(record))}"
    )
    assert record["kind"] == "step"
    assert record["error"] is None
    assert isinstance(record["ts"], (int, float))


def test_step_extracts_every_identifier_from_the_command(tmp_path):
    """The join key of the whole feature: these five are what pair the two logs."""

    service = _StubService(result=_step_result())
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    proxy.step(
        _step_command(
            run_id="bc_eval_20260806",
            episode_id=4,
            step_id=9,
            env_step=42,
            transition_id="bc_eval_20260806:42",
        )
    )
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["run_id"] == "bc_eval_20260806"
    assert record["episode_id"] == 4
    assert record["step_id"] == 9
    assert record["env_step"] == 42
    assert record["transition_id"] == "bc_eval_20260806:42"


def test_step_identifier_extraction_is_defensive(tmp_path):
    """A command without ``data`` yields nulls, not a lost record.

    ``BeginEpisodeCommand`` has no ``data`` at all and a future command could
    drop a key; the row still has to be written, because a timing log that
    disappears on the odd request is worse than one with a null in it.
    """

    service = _StubService(result=_step_result())
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    proxy.step(SimpleNamespace(run_id="run-a"))
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["run_id"] == "run-a"
    for field in ("episode_id", "step_id", "env_step", "transition_id"):
        assert record[field] is None, f"{field} must be null when unavailable"
    assert record["handler_ms"] > 0.0


def test_step_reports_the_inference_span_the_server_measured(tmp_path):
    service = _StubService(result=_step_result(inference_ms=6.5))
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    proxy.step(_step_command())
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["infer_ms"] == pytest.approx(6.5)


def test_a_terminal_step_has_no_action_and_therefore_no_inference_span(tmp_path):
    """``request_action=False`` means the server never ran the policy."""

    service = _StubService(result=_step_result(inference_ms=None, terminal=True))
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    proxy.step(_step_command())
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["infer_ms"] is None
    assert record["terminal"] is True
    # ...and the arithmetic must treat the absence as zero, not as a hole.
    assert record["overhead_ms"] == pytest.approx(record["handler_ms"])


def test_step_propagates_terminal_and_deduplicated_from_the_result(tmp_path):
    """``deduplicated`` is why the row count is the RPC count, not the step count."""

    service = _StubService(result=_step_result(terminal=True, deduplicated=True))
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    proxy.step(_step_command())
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["terminal"] is True
    assert record["deduplicated"] is True


def test_a_result_without_outcome_or_ack_yields_nulls_not_a_crash(tmp_path):
    service = _StubService(result=SimpleNamespace())
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    proxy.step(_step_command())
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["terminal"] is None
    assert record["deduplicated"] is None
    assert record["infer_ms"] is None


def test_the_sink_span_is_subtracted_out_of_the_handler_remainder(tmp_path):
    """The point of the whole module: name the remainder instead of hiding it.

    The sink is wired the way production wires it -- the SAME ``TimingSink``
    instance the service calls inside its own handler, sharing the recorder
    with the proxy -- so ``sink_ms`` is genuinely a piece of ``handler_ms``.
    """

    writer = _writer(tmp_path)
    recorder = StepTimingRecorder(writer)
    inner = _RecordingSink()
    sink = TimingSink(inner, recorder)

    def _run_sink(command):
        sink(command.data, False)

    service = _StubService(
        result=_step_result(inference_ms=6.5), before_return=_run_sink
    )
    proxy = ServiceStepTimingProxy(service, recorder)

    proxy.step(_step_command())
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert len(inner.calls) == 1, "the service really did run the sink"
    assert record["sink_ms"] is not None and record["sink_ms"] > 0.0
    assert record["handler_ms"] > record["sink_ms"], (
        "the sink ran inside the handler, so its span cannot exceed it"
    )
    assert record["overhead_ms"] == pytest.approx(
        record["handler_ms"] - record["infer_ms"] - record["sink_ms"], abs=1e-9
    )


def test_a_step_that_ran_no_sink_reports_a_null_sink_span(tmp_path):
    """Null, not zero: "not measured" and "took no time" are different claims."""

    service = _StubService(result=_step_result(inference_ms=6.5))
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    proxy.step(_step_command())
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["sink_ms"] is None
    assert record["overhead_ms"] == pytest.approx(
        record["handler_ms"] - record["infer_ms"], abs=1e-9
    )


def test_a_raising_step_still_emits_its_record_and_re_raises(tmp_path):
    """The rows an operator investigating a stall most wants are the failures."""

    service = _StubService(raises=RuntimeError("server blew up"))
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    with pytest.raises(RuntimeError, match="server blew up"):
        proxy.step(_step_command())
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["error"] == "RuntimeError", (
        "the exception CLASS is the honest label; the message can carry PII "
        "and is not part of the schema"
    )
    assert record["kind"] == "step"
    assert record["handler_ms"] > 0.0
    assert record["transition_id"] == "run-a:11", (
        "a failed step must still be identifiable"
    )


def test_each_step_call_emits_exactly_one_record(tmp_path):
    service = _StubService(result=_step_result())
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    for env_step in range(3):
        proxy.step(_step_command(env_step=env_step, transition_id=f"run-a:{env_step}"))
    writer.close()

    records = _lines(tmp_path / "timing.jsonl")
    assert [row["env_step"] for row in records] == [0, 1, 2]
    assert all(row["kind"] == "step" for row in records)


# --------------------------------------------------------------------------- #
# 7. ServiceStepTimingProxy -- the begin_episode record                          #
# --------------------------------------------------------------------------- #
def test_begin_episode_emits_its_own_kind_with_the_same_schema(tmp_path):
    """One row per BeginEpisode, so a slow episode start is visible too.

    INTERPRETATION (spec §2: "begin_episode emits kind='begin_episode' with
    nulls for step-only fields"): the identifier extraction is defined against
    ``command.data``, which ``BeginEpisodeCommand`` does not have, so
    ``episode_id`` is accepted as either null or the command's own value --
    everything genuinely step-scoped is required to be null.
    """

    action = SimpleNamespace(server_inference_ms=3.25)
    service = _StubService(result=action)
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))
    command = SimpleNamespace(run_id="run-a", episode_id=2)

    returned = proxy.begin_episode(command)
    writer.close()

    assert returned is action
    (record,) = _lines(tmp_path / "timing.jsonl")
    assert set(record) == set(SERVER_RECORD_KEYS), (
        "both kinds share one frozen schema so the analyzer can read the file "
        "with a single reader"
    )
    assert record["kind"] == "begin_episode"
    assert record["run_id"] == "run-a"
    assert record["handler_ms"] > 0.0
    assert record["error"] is None
    for field in ("step_id", "env_step", "transition_id", "sink_ms"):
        assert record[field] is None, f"{field} is step-scoped and must be null"
    for field in ("terminal", "deduplicated"):
        assert record[field] is None, f"BeginEpisode has no {field}"
    assert record["episode_id"] in (None, 2)


def test_a_raising_begin_episode_still_emits_its_record_and_re_raises(tmp_path):
    service = _StubService(raises=ValueError("observation rejected"))
    writer = _writer(tmp_path)
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    with pytest.raises(ValueError, match="observation rejected"):
        proxy.begin_episode(SimpleNamespace(run_id="run-a", episode_id=0))
    writer.close()

    (record,) = _lines(tmp_path / "timing.jsonl")
    assert record["kind"] == "begin_episode"
    assert record["error"] == "ValueError"


def test_a_dead_writer_never_breaks_serving(tmp_path):
    """End-to-end fail-open: an unwritable log must not reach the caller.

    This is the property that lets the instrumentation be left on during a real
    session.  It is asserted through the proxy rather than the writer alone
    because that is the path a live Step RPC actually takes.
    """

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory\n", encoding="utf-8")
    writer = FailOpenJsonlWriter(blocker / "timing.jsonl")
    service = _StubService(result=_step_result())
    proxy = ServiceStepTimingProxy(service, StepTimingRecorder(writer))

    returned = proxy.step(_step_command())

    assert returned is service.result
    assert writer.error_count == 1
    assert writer.write_count == 0
