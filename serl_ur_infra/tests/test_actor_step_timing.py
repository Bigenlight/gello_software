"""Actor-side step timing: one jsonl row per completed Step RPC, or no file.

WHY THIS IS AN INTEGRATION TEST AND NOT A UNIT TEST
---------------------------------------------------
The server half of this feature (``ur_env/step_timing.py``) can be tested in
isolation because it is a wrapper.  The actor half cannot: its whole content is
WHERE the ``time.perf_counter()`` calls sit inside ``_run_remote_actor_impl``'s
loop, and every property worth pinning is a statement about the loop --

* one row per COMPLETED Step RPC, so the row count is evidence of how far the
  session actually got rather than of how many iterations were attempted;
* ``env_step`` strictly increasing across the whole run, because it is the run's
  global counter (``step_id`` is the one that restarts per episode) and the
  analyzer joins on ``transition_id = f"{run_id}:{env_step}"``;
* ``post_prev_ms`` null on the first row only -- it measures the gap BETWEEN
  iterations, which is where the pickle dumps, the operator waits and the next
  ``env.reset()`` land.  That gap is the actual subject: 2026-07-30 measured the
  production actor at 512 ms per iteration while ``env.step`` paced itself at
  100 ms, and this file is how the remaining ~412 ms gets attributed;
* and, with ``step_timing_path=None``, NOTHING on disk.  Default-off is
  load-bearing: a robot evaluation must not change shape because a measurement
  exists.

THE FAKES ARE IMPORTED, NOT COPIED
----------------------------------
``tests/test_actor_abort_lifecycle.py`` already owns a fake env / server-like
network / operator session that reproduce the real server's session registry and
reward finalization.  Re-deriving them here would produce a second, weaker set
that drifts; the timing additions are a subclass of its network that also carries
``server_inference_ms`` and ``round_trip_ms``, which the abort fakes have no
reason to model.

WHAT IS DELIBERATELY NOT ASSERTED
---------------------------------
Phase arithmetic.  The phases overlap unmeasured code (validation, the abort
read, the reporter publishes), so ``loop_ms >= env_step_ms + rpc_ms`` is NOT a
theorem and asserting it would pin the current statement order rather than the
contract.  What is asserted is that each phase is either a non-negative number
or an honest null.

Run (from ``/home/laptop3/gello_software``)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
      -p no:cacheprovider serl_ur_infra/tests/test_actor_step_timing.py
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(_HERE.parent))
# The shared fakes live in a sibling test module; under pytest's prepend import
# mode this directory is already on sys.path, but say so explicitly so the file
# also works under -p no:cacheprovider from any cwd.
sys.path.insert(0, str(_HERE))

from ur_env.remote_actor import run_remote_actor  # noqa: E402

from test_actor_abort_lifecycle import (  # noqa: E402
    _Env,
    _Operator,
    _ServerLikeNetwork,
    _config,
)


#: The frozen actor-side record.  Restated rather than imported: this list IS
#: the contract that ``scripts/analyze_bc_rollout.py --actor-timing`` reads.
ACTOR_RECORD_KEYS = frozenset(
    {
        "ts",
        "run_id",
        "episode_id",
        "step_id",
        "env_step",
        "transition_id",
        "env_step_ms",
        "build_ms",
        "sidecar_ms",
        "rpc_ms",
        "round_trip_ms",
        "server_inference_ms",
        "loop_ms",
        "post_prev_ms",
        "attached_sidecar",
        "intervened",
        "terminal",
    }
)

#: Phases that are a measurement of code that always runs.
ALWAYS_MEASURED = ("env_step_ms", "build_ms", "rpc_ms", "loop_ms")

RUN_ID = "run"
SERVER_INFERENCE_MS = 4.25
SERVER_ROUND_TRIP_MS = 9.5


class _TimedNetwork(_ServerLikeNetwork):
    """The abort suite's server-like fake, plus the two server-reported spans.

    ``ActionResult`` carries ``server_inference_ms`` and ``round_trip_ms``; the
    abort fakes omit them because nothing there reads them.  Pinning real values
    here is what proves the actor PLUMBS them rather than writing null.
    """

    @staticmethod
    def _action(version=0):
        return SimpleNamespace(
            action=np.zeros(7, dtype=np.float32),
            policy_version=version,
            server_inference_ms=SERVER_INFERENCE_MS,
            round_trip_ms=SERVER_ROUND_TRIP_MS,
        )


def _rows(path: Path) -> list[dict[str, Any]]:
    assert path.is_file(), f"expected the actor step-timing log at {path}"
    out = []
    for number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        assert raw.strip(), f"{path}:{number} is blank; one record per line"
        out.append(json.loads(raw))
    return out


def _supports_step_timing() -> bool:
    return "step_timing_path" in inspect.signature(run_remote_actor).parameters


def _run(
    *,
    step_timing_path: str | None,
    max_steps: int = 4,
    abort_press_offset: int = 2,
    infos=None,
):
    """One short run: three steps, an operator ABORT, then a fresh episode.

    ``abort_press_offset=2`` spends the one-shot token on the THIRD read, so the
    terminal lands mid-run rather than on the first or last row -- the only
    arrangement in which "terminal is True on exactly one row" and "the episode
    counter advanced" are both observable.
    """

    assert _supports_step_timing(), (
        "run_remote_actor has no step_timing_path parameter -- the actor half "
        "of the step-timing feature (spec section 3) is not wired up yet"
    )
    events: list[str] = []
    env = _Env(events, infos=infos)
    network = _TimedNetwork(events)
    operator = _Operator(
        events, abort_at=(RUN_ID, 0), abort_press_offset=abort_press_offset
    )
    summary = run_remote_actor(
        network,
        env,
        config=_config(max_steps),
        actor_id="actor",
        run_id=RUN_ID,
        session_id_factory=iter(("s0", "s1", "s2", "s3")).__next__,
        operator_session=operator,
        step_timing_path=step_timing_path,
    )
    return summary, network, env


# --------------------------------------------------------------------------- #
# 1. The file exists and its rows are the Step RPCs                             #
# --------------------------------------------------------------------------- #
def test_one_row_per_completed_step_rpc(tmp_path):
    path = tmp_path / "actor_timing.jsonl"

    _summary, network, _env = _run(step_timing_path=str(path))

    rows = _rows(path)
    assert len(rows) == len(network.step_calls) == 4, (
        "the row count is the count of Step RPCs that came back, which is what "
        "makes the log evidence of progress rather than of intent"
    )


def test_every_row_carries_the_whole_frozen_schema(tmp_path):
    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    for index, row in enumerate(_rows(path)):
        assert set(row) == set(ACTOR_RECORD_KEYS), (
            f"row {index}: the actor step-timing schema is frozen; "
            f"extra={sorted(set(row) - ACTOR_RECORD_KEYS)} "
            f"missing={sorted(ACTOR_RECORD_KEYS - set(row))}"
        )


def test_the_log_parent_directory_is_created_for_the_operator(tmp_path):
    """The operator names a path under gello_logs/; the tree is ours to make."""

    path = tmp_path / "gello_logs" / "step_timing" / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    assert _rows(path)


# --------------------------------------------------------------------------- #
# 2. Identity: what the analyzer joins on                                       #
# --------------------------------------------------------------------------- #
def test_env_step_is_the_runs_global_strictly_increasing_counter(tmp_path):
    """``step_id`` restarts per episode; ``env_step`` must not."""

    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    rows = _rows(path)
    env_steps = [row["env_step"] for row in rows]
    assert env_steps == sorted(set(env_steps)), (
        f"env_step must be strictly increasing across the run, got {env_steps}"
    )
    assert env_steps == [0, 1, 2, 3]
    # The episode really did roll over -- otherwise "global" is untested.
    assert [row["episode_id"] for row in rows] == [0, 0, 0, 1]
    assert [row["step_id"] for row in rows] == [0, 1, 2, 0]


def test_transition_id_is_the_wire_id_the_analyzer_joins_on(tmp_path):
    """``f"{run_id}:{env_step}"`` -- the same string ``build_data`` ships."""

    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    rows = _rows(path)
    for row in rows:
        assert row["run_id"] == RUN_ID
        assert row["transition_id"] == f"{RUN_ID}:{row['env_step']}"


def test_the_timing_ids_agree_with_what_went_on_the_wire(tmp_path):
    """Pinned against the network fake, not against this file's arithmetic."""

    path = tmp_path / "actor_timing.jsonl"

    _summary, network, _env = _run(step_timing_path=str(path))

    shipped = [
        call[1]["data"]["meta"]["transition_id"] for call in network.step_calls
    ]
    assert [row["transition_id"] for row in _rows(path)] == shipped


# --------------------------------------------------------------------------- #
# 3. The measurements themselves                                                #
# --------------------------------------------------------------------------- #
def test_every_phase_is_a_non_negative_number_or_an_honest_null(tmp_path):
    """Null means "not measured"; zero would mean "took no time"."""

    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    numeric = (
        "env_step_ms",
        "build_ms",
        "sidecar_ms",
        "rpc_ms",
        "round_trip_ms",
        "server_inference_ms",
        "loop_ms",
        "post_prev_ms",
    )
    for index, row in enumerate(_rows(path)):
        for field in numeric:
            value = row[field]
            assert value is None or isinstance(value, (int, float)), (
                f"row {index}: {field} must be a number or null, got {value!r}"
            )
            if isinstance(value, (int, float)):
                assert value >= 0.0, f"row {index}: {field} is negative ({value})"
        for field in ALWAYS_MEASURED:
            assert row[field] is not None, (
                f"row {index}: {field} measures code that always runs, so a "
                "null there is a wiring bug, not an absent measurement"
            )


def test_rpc_ms_is_measured_on_every_row(tmp_path):
    """The Step RPC is the span this whole feature exists to explain."""

    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    for index, row in enumerate(_rows(path)):
        assert row["rpc_ms"] > 0.0, f"row {index} claims a free Step RPC"


def test_post_prev_ms_is_null_on_the_first_row_and_numeric_afterwards(tmp_path):
    """It is the gap BETWEEN iterations; before the first there is no gap.

    This is where the operator waits, the episode boundary and the local pickle
    dump are accounted for -- the ~412 ms that sits outside ``env.step``.
    """

    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    rows = _rows(path)
    assert rows[0]["post_prev_ms"] is None, (
        "there is no previous iteration to measure a gap from"
    )
    for index, row in enumerate(rows[1:], start=1):
        assert isinstance(row["post_prev_ms"], (int, float)), (
            f"row {index} must report the gap since the previous emit"
        )
        assert row["post_prev_ms"] >= 0.0


def test_the_server_reported_spans_are_plumbed_through(tmp_path):
    """``server_inference_ms``/``round_trip_ms`` come off the ActionResult.

    INTERPRETATION: both are read from the action THIS Step returned, so a
    terminal Step -- which requests no action -- legitimately reports null.  The
    assertion is therefore "the value reaches the log at all", not "every row
    has one"; pinning the terminal row to a number would pin a different
    (and wrong) reading of the schema.
    """

    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    rows = _rows(path)
    inference = [row["server_inference_ms"] for row in rows]
    round_trip = [row["round_trip_ms"] for row in rows]
    assert any(value is not None for value in inference), (
        "no row carried the server's own inference span; the field is a "
        "constant null and the analyzer's phase table would be empty"
    )
    for value in inference:
        assert value is None or value == pytest.approx(SERVER_INFERENCE_MS)
    for value in round_trip:
        assert value is None or value == pytest.approx(SERVER_ROUND_TRIP_MS)


def test_ts_is_a_wall_clock_stamp_on_every_row(tmp_path):
    """The rows are read alongside logs from two other hosts; they need a clock."""

    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    rows = _rows(path)
    stamps = [row["ts"] for row in rows]
    assert all(isinstance(value, (int, float)) for value in stamps)
    assert stamps == sorted(stamps), "rows are appended in loop order"
    # A plausible-epoch check: a perf_counter leaking in here would be tiny.
    assert stamps[0] > 1_600_000_000, "ts must be a wall clock, not a monotonic"


# --------------------------------------------------------------------------- #
# 4. The flags the row carries about ITSELF                                     #
# --------------------------------------------------------------------------- #
def test_terminal_is_true_on_exactly_the_row_that_ended_the_episode(tmp_path):
    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    rows = _rows(path)
    terminal = [bool(row["terminal"]) for row in rows]
    assert terminal == [False, False, True, False], (
        "the operator ABORT ended env_step 2, and the next episode's first "
        f"step is not terminal; got {terminal}"
    )


def test_intervened_reflects_what_the_transition_reported(tmp_path):
    """Human-driven steps are the slow ones; the split has to be readable."""

    path = tmp_path / "actor_timing.jsonl"

    _run(
        step_timing_path=str(path),
        infos=[{}, {"intervened": 1}, {}, {}],
    )

    rows = _rows(path)
    assert [bool(row["intervened"]) for row in rows] == [False, True, False, False]


def test_attached_sidecar_is_false_when_no_scheduler_is_wired(tmp_path):
    """A classifier sidecar makes one Step carry two images; say which ones did."""

    path = tmp_path / "actor_timing.jsonl"

    _run(step_timing_path=str(path))

    rows = _rows(path)
    assert all(row["attached_sidecar"] is False for row in rows)
    assert all(row["sidecar_ms"] is None for row in rows), (
        "no sidecar was built, so its span is absent rather than zero"
    )


# --------------------------------------------------------------------------- #
# 5. Default OFF                                                                #
# --------------------------------------------------------------------------- #
def test_without_a_path_the_run_writes_nothing_at_all(tmp_path):
    """The instrumentation is opt-in; an ordinary evaluation leaves no trace."""

    _summary, network, _env = _run(step_timing_path=None)

    assert len(network.step_calls) == 4, "the run itself must be unaffected"
    leftovers = sorted(str(item) for item in tmp_path.rglob("*"))
    assert leftovers == [], (
        f"step_timing_path=None must create no files anywhere: {leftovers}"
    )


def test_the_instrumented_and_uninstrumented_runs_ship_identical_transitions(
    tmp_path,
):
    """Measuring must not change the data.

    Compared on the wire payload rather than on the summary: the transition is
    what the learner keeps, so this is the assertion that the diagnostic is
    genuinely passive.
    """

    def _shipped(network):
        return [
            (
                call[1]["data"]["meta"]["transition_id"],
                call[1]["data"]["meta"]["env_step"],
                call[1]["data"]["transition"]["episode_id"],
                call[1]["data"]["transition"]["step_id"],
                call[1]["data"]["transition"]["dones"],
                call[1]["data"]["transition"]["truncated"],
                call[1]["request_action"],
            )
            for call in network.step_calls
        ]

    _s1, timed_network, _e1 = _run(
        step_timing_path=str(tmp_path / "on" / "actor_timing.jsonl")
    )
    _s2, plain_network, _e2 = _run(step_timing_path=None)

    assert _shipped(timed_network) == _shipped(plain_network)


# --------------------------------------------------------------------------- #
# 6. Fail-open, through the real loop                                           #
# --------------------------------------------------------------------------- #
def test_an_unwritable_timing_path_does_not_take_the_session_down(tmp_path):
    """A full disk costs a log, never an episode with the arm under power."""

    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory\n", encoding="utf-8")

    _summary, network, _env = _run(
        step_timing_path=str(blocker / "actor_timing.jsonl")
    )

    assert len(network.step_calls) == 4, (
        "the run must complete exactly as it would with timing off"
    )
    assert blocker.read_text(encoding="utf-8").startswith("i am a file")
