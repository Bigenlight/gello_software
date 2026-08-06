"""Contract tests for ``scripts/analyze_hil_latency.py``.

WHY THE FIXTURE NUMBERS LOOK ARTIFICIAL
---------------------------------------
Every distribution below is chosen so that ``numpy``'s linear-interpolation
percentiles land on values a reader can verify by hand, because the point of
this analyzer is that an operator will quote its p99 in a status document.  For
101 samples the percentile index is ``q/100 * (n - 1) = q``, so a run of
``1 .. 101`` has p50 = 51, p90 = 91, p99 = 100 exactly -- no tolerance, no
"approximately".  A test that only checked ``p50 <= p90 <= p99`` would pass just
as happily on a sorted-wrong array.

WHY THE TWO HOSTS GET DELIBERATELY INCOMPATIBLE CLOCKS
------------------------------------------------------
The actor fixture stamps ``t_epoch`` near 1e6 and the server fixture near 5e6.
If any code path ever subtracted one host's timestamp from the other's, the
derived numbers would be absurd and these tests would fail loudly.  The one
cross-host quantity -- ``network_plus_queue_ms`` -- is a difference of two
*durations* and is pinned to an exact constant here.

WHAT IS DELIBERATELY NOT PINNED
-------------------------------
Prose.  Assertions match on section titles, row labels and numbers; the wording
of the caveats around them is free to change.  The exception is the three
degradation phrases (``server data absent`` etc.), which an operator greps for.

Run (from ``/home/laptop3/gello_software``)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_analyze_hil_latency.py
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import pytest


_HERE = Path(os.path.abspath(__file__)).parent
_SERL_UR_INFRA = _HERE.parent
_REPO_ROOT = _SERL_UR_INFRA.parent
_SCRIPT = _SERL_UR_INFRA / "scripts" / "analyze_hil_latency.py"


def _load_module():
    """Import the CLI from its path: ``scripts/`` is not a package."""

    assert _SCRIPT.is_file(), f"pinned CLI is missing: {_SCRIPT}"
    spec = importlib.util.spec_from_file_location("_analyze_hil_latency", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


# --------------------------------------------------------------------------- #
# Fixture ground truth                                                          #
# --------------------------------------------------------------------------- #

N_ACTOR = 101              # actor loop iterations
N_SERVER_MATCHED = 100     # server Step records that share a transition_id
N_SERVER_EXTRA = 2         # server Steps whose actor line was never written
N_LEARNER_UPDATES = 20     # learner_update events, each 100 ms long
N_OVERLAPPED = 20          # server Steps that overlap one of those windows

RUN_ID = "cube_in_cup_real_20260806_101500"
SERVER_TOTAL_MS = 50.0     # constant, so network+queue is exactly step_rpc - 50
ACTOR_T0 = 1_000_000.0     # laptop3 epoch (fictional)
SERVER_T0 = 5_000_000.0    # GPU host epoch (fictional, and 4e6 s away)
UTD_RATIO = 7              # NOT the production 10, on purpose


def _transition_id(index: int) -> str:
    return f"{RUN_ID}:0:{index}"


def _iter_interval_ms(index: int) -> float:
    """50x 90 ms, 30x 150 ms, 15x 300 ms, 6x 600 ms -- 101 samples.

    Sorted, index 50 is 150, index 90 is 300, index 99 is 600, so p50/p90/p99
    are those three values exactly.  Over-budget counts are 51 / 21 / 6.
    """

    if index < 50:
        return 90.0
    if index < 80:
        return 150.0
    if index < 95:
        return 300.0
    return 600.0


def write_actor(path: Path, *, count: int = N_ACTOR) -> Path:
    lines = []
    for index in range(count):
        record: dict[str, Any] = {
            "schema": 1,
            "role": "actor",
            "seq": index,
            "t_epoch": ACTOR_T0 + index,
            "run_id": RUN_ID,
            "episode_id": 0,
            "env_step": index,
            "transition_id": _transition_id(index),
            "request_id": f"req-{index}",
            "intervened": index < 40,
            "sidecar_attached": index % 5 == 0,
            "env_step_ms": float(index + 1),          # 1 .. 101
            "transition_build_ms": 2.0,
            "step_rpc_ms": 100.0 + index,             # 100 .. 200
            "iter_interval_ms": _iter_interval_ms(index),
            "server_inference_ms": 12.0,
        }
        if index % 5 == 0:
            record["sidecar_encode_ms"] = 8.0
        if index < 40:
            record["intervention_saturation"] = round(0.01 * index, 4)
            record["intervention_saturated"] = index < 10
            record["intervention_follow_ticks"] = 15
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_server(path: Path) -> Path:
    lines = [
        json.dumps(
            {
                "schema": 1,
                "role": "server",
                "seq": 0,
                "t_epoch": SERVER_T0 - 10.0,
                "rpc": "begin_episode",
                "run_id": RUN_ID,
                "episode_id": 0,
                "total_ms": 57.7,
            }
        )
    ]
    for index in range(N_SERVER_MATCHED):
        lines.append(
            json.dumps(
                {
                    "schema": 1,
                    "role": "server",
                    "seq": index + 1,
                    # 0.5 s apart, so a 100 ms learner window can only ever
                    # touch the one record it was built for.
                    "t_epoch": SERVER_T0 + 0.5 * index,
                    "transition_id": _transition_id(index),
                    "request_id": f"req-{index}",
                    "concurrent_rpcs": 1,
                    "request_decode_ms": 3.0,
                    "trunk_encode_ms": 4.0,
                    "classifier_ms": 0.0,
                    "reward_finalize_ms": 1.0,
                    "replay_insert_ms": 2.0,
                    "policy_inference_ms": 30.0,
                    "response_build_ms": 5.0,
                    "total_ms": SERVER_TOTAL_MS,
                }
            )
        )
    for extra in range(N_SERVER_EXTRA):
        lines.append(
            json.dumps(
                {
                    "schema": 1,
                    "role": "server",
                    "seq": N_SERVER_MATCHED + 1 + extra,
                    # Far away in time: these must not land in the overlap split.
                    "t_epoch": SERVER_T0 + 10_000.0 + extra,
                    "transition_id": f"{RUN_ID}:9:{extra}",
                    "concurrent_rpcs": 2,
                    "total_ms": SERVER_TOTAL_MS,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_learner(path: Path, *, utd_ratio: int | None = UTD_RATIO) -> Path:
    ready: dict[str, Any] = {
        "event": "learner_process_ready",
        "time_ns": int((SERVER_T0 - 60.0) * 1e9),
        "learner_step": 0,
        "policy_version": 1,
    }
    if utd_ratio is not None:
        ready["utd_ratio"] = utd_ratio
        ready["learner"] = {"log_period": 20, "batch_size": 256}
    lines = [json.dumps(ready)]
    for index in range(N_LEARNER_UPDATES):
        # Window is [end - 100 ms, end] with end = record start + 20 ms, so it
        # overlaps exactly the server Step whose own window is [start, +50 ms].
        end_s = SERVER_T0 + 0.5 * index + 0.02
        lines.append(
            json.dumps(
                {
                    "event": "learner_update",
                    "time_ns": int(end_s * 1e9),
                    "learner_step": index + 1,
                    "gradient_step": 2 * (index + 1),
                    "policy_version": 1 + index // 10,
                    "metrics": {
                        "timing/sample_ms": 5.0,
                        "timing/critic_update_ms": float(index + 1),
                        "timing/full_update_ms": 90.0,
                        "timing/learner_step_ms": 100.0,
                        "buffer/replay_size": float(300 + index),
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Report navigation helpers                                                     #
# --------------------------------------------------------------------------- #


def _sections(actor: Path, server: Path | None = None, learner: Path | None = None):
    sections, *_ = analyzer.build_report(actor, server, learner)
    return sections


def _section(sections: Sequence[Any], needle: str):
    for section in sections:
        if needle.lower() in section.title.lower():
            return section
    raise AssertionError(
        f"no section matching {needle!r}; have "
        f"{[s.title for s in sections]}"
    )


def _rows(section: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for block in section.blocks:
        if isinstance(block, analyzer.Table):
            for row in block.rows:
                out.append(dict(zip(block.headers, row)))
    return out


def _row(section: Any, label: str) -> dict[str, str]:
    for row in _rows(section):
        first = next(iter(row.values()), "")
        if first == label:
            return row
    raise AssertionError(
        f"no row labelled {label!r}; have "
        f"{[next(iter(r.values()), '') for r in _rows(section)]}"
    )


def _notes(section: Any) -> str:
    return "\n".join(
        block.text for block in section.blocks if isinstance(block, analyzer.Note)
    )


def _console(sections: Sequence[Any]) -> str:
    return analyzer._render_console(sections)


# --------------------------------------------------------------------------- #
# 1. per-phase percentiles                                                      #
# --------------------------------------------------------------------------- #


def test_actor_phase_percentiles_are_exact(tmp_path: Path) -> None:
    """``env_step_ms`` = 1..101 -> mean 51, p50 51, p90 91, p99 100, max 101."""

    sections = _sections(write_actor(tmp_path / "actor.jsonl"))
    row = _row(_section(sections, "PER-PHASE"), "env_step_ms")
    assert row["count"] == str(N_ACTOR)
    assert row["mean"] == "51.000"
    assert row["p50"] == "51.000"
    assert row["p90"] == "91.000"
    assert row["p99"] == "100.000"
    assert row["max"] == "101.000"


def test_sparse_phase_counts_only_the_records_that_have_it(tmp_path: Path) -> None:
    """``sidecar_encode_ms`` exists on 21 of 101 lines and must not be zero-filled."""

    sections = _sections(write_actor(tmp_path / "actor.jsonl"))
    row = _row(_section(sections, "PER-PHASE"), "sidecar_encode_ms")
    assert row["count"] == "21"
    assert row["mean"] == "8.000"
    assert row["max"] == "8.000"


def test_passthrough_ms_key_is_labelled_not_silently_a_phase(tmp_path: Path) -> None:
    sections = _sections(write_actor(tmp_path / "actor.jsonl"))
    row = _row(_section(sections, "PER-PHASE"), "server_inference_ms")
    assert row["count"] == str(N_ACTOR)
    assert "server" in row["note"].lower()


def test_marks_are_excluded_from_the_phase_table(tmp_path: Path) -> None:
    """``mark()`` writes ``*_at_ms`` offsets; those are not durations."""

    path = tmp_path / "actor.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "role": "actor",
                "seq": 0,
                "t_epoch": ACTOR_T0,
                "transition_id": _transition_id(0),
                "env_step_ms": 10.0,
                "response_seen_at_ms": 7.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    section = _section(_sections(path), "PER-PHASE")
    labels = [next(iter(row.values()), "") for row in _rows(section)]
    assert "env_step_ms" in labels
    assert "response_seen_at_ms" not in labels
    assert "response_seen_at_ms" in _notes(_section(_sections(path), "INPUTS"))


def test_unknown_phase_keys_are_discovered_not_hardcoded(tmp_path: Path) -> None:
    """A phase a future wiring commit adds must appear without editing the CLI."""

    path = tmp_path / "actor.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "role": "actor",
                "seq": 0,
                "t_epoch": ACTOR_T0,
                "brand_new_phase_ms": 3.5,
                "some_unknown_field": {"nested": [1, 2]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = _row(_section(_sections(path), "PER-PHASE"), "brand_new_phase_ms")
    assert row["count"] == "1"
    assert row["p99"] == "3.500"


def test_server_phase_table_and_begin_episode_are_separate(tmp_path: Path) -> None:
    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"), write_server(tmp_path / "server.jsonl")
    )
    section = _section(sections, "PER-PHASE")
    captions = [
        block.caption
        for block in section.blocks
        if isinstance(block, analyzer.Table) and block.caption
    ]
    assert any("Step" in caption for caption in captions)
    assert any("BeginEpisode" in caption for caption in captions)
    # The 102 Step records carry total_ms; the single BeginEpisode record does
    # not get folded into them.
    step_table = next(
        block
        for block in section.blocks
        if isinstance(block, analyzer.Table) and block.caption and "Step" in block.caption
    )
    total_row = dict(
        zip(step_table.headers, next(r for r in step_table.rows if r[0] == "total_ms"))
    )
    assert total_row["count"] == str(N_SERVER_MATCHED + N_SERVER_EXTRA)
    assert total_row["max"] == "50.000"


# --------------------------------------------------------------------------- #
# 2. join + network_plus_queue                                                  #
# --------------------------------------------------------------------------- #


def test_join_coverage_counts(tmp_path: Path) -> None:
    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"), write_server(tmp_path / "server.jsonl")
    )
    section = _section(sections, "CROSS-HOST JOIN")
    actor_row = _row(section, "actor")
    assert actor_row["records"] == str(N_ACTOR)
    assert actor_row["with transition_id"] == str(N_ACTOR)
    assert actor_row["matched"] == str(N_SERVER_MATCHED)
    assert actor_row["coverage"] == "99.0%"

    server_row = _row(section, "server (Step)")
    assert server_row["records"] == str(N_SERVER_MATCHED + N_SERVER_EXTRA)
    assert server_row["with transition_id"] == str(N_SERVER_MATCHED + N_SERVER_EXTRA)
    assert server_row["matched"] == str(N_SERVER_MATCHED)


def test_network_plus_queue_is_step_rpc_minus_server_total(tmp_path: Path) -> None:
    """step_rpc = 100..199 over the matched 100; server total = 50 flat.

    So the derived values are 50..149: mean 99.5, p50 99.5 (index 49.5),
    p90 139.1 (index 89.1), p99 148.01 (index 98.01), max 149.
    """

    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"), write_server(tmp_path / "server.jsonl")
    )
    section = _section(sections, "CROSS-HOST JOIN")
    row = _row(section, "network_plus_queue_ms")
    assert row["count"] == str(N_SERVER_MATCHED)
    assert row["mean"] == "99.500"
    assert row["p50"] == "99.500"
    assert row["p90"] == "139.100"
    assert row["p99"] == "148.010"
    assert row["max"] == "149.000"


def test_join_ignores_the_two_hosts_wall_clocks(tmp_path: Path) -> None:
    """The fixture's epochs are 4e6 s apart; nothing derived may reflect that."""

    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"), write_server(tmp_path / "server.jsonl")
    )
    row = _row(_section(sections, "CROSS-HOST JOIN"), "network_plus_queue_ms")
    assert abs(float(row["max"])) < 1e6


def test_duplicate_server_transition_ids_are_reported(tmp_path: Path) -> None:
    server = write_server(tmp_path / "server.jsonl")
    duplicate = json.loads(server.read_text(encoding="utf-8").splitlines()[1])
    duplicate["seq"] = 9999
    with open(server, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(duplicate) + "\n")
    sections = _sections(write_actor(tmp_path / "actor.jsonl"), server)
    assert "1 duplicate transition_id" in _notes(_section(sections, "CROSS-HOST JOIN"))


# --------------------------------------------------------------------------- #
# 3. loop budget                                                                #
# --------------------------------------------------------------------------- #


def test_loop_budget_percentiles_and_fractions(tmp_path: Path) -> None:
    sections = _sections(write_actor(tmp_path / "actor.jsonl"))
    section = _section(sections, "LOOP BUDGET")
    row = _row(section, "iter_interval_ms")
    assert row["count"] == str(N_ACTOR)
    assert row["p50"] == "150.000"
    assert row["p90"] == "300.000"
    assert row["p99"] == "600.000"
    assert row["max"] == "600.000"

    over_100 = _row(section, "> 100 ms")
    assert over_100["iterations"] == "51"      # 30 + 15 + 6
    assert over_100["of"] == str(N_ACTOR)
    assert over_100["fraction"] == "50.5%"
    assert _row(section, "> 200 ms")["iterations"] == "21"
    assert _row(section, "> 500 ms")["iterations"] == "6"


def test_loop_budget_reports_implied_rate(tmp_path: Path) -> None:
    sections = _sections(write_actor(tmp_path / "actor.jsonl"))
    section = _section(sections, "LOOP BUDGET")
    assert _row(section, "from p50")["Hz"] == "6.667"       # 1000 / 150
    assert _row(section, "from p99")["Hz"] == "1.667"       # 1000 / 600


def test_loop_budget_degrades_without_iter_interval(tmp_path: Path) -> None:
    path = tmp_path / "actor.jsonl"
    path.write_text(
        json.dumps({"schema": 1, "role": "actor", "seq": 0, "env_step_ms": 5.0}) + "\n",
        encoding="utf-8",
    )
    section = _section(_sections(path), "LOOP BUDGET")
    assert "iter_interval_ms absent" in _notes(section)


# --------------------------------------------------------------------------- #
# 4. contention                                                                 #
# --------------------------------------------------------------------------- #


def test_contention_split_uses_only_server_side_clock(tmp_path: Path) -> None:
    """20 crafted windows overlap 20 Step RPCs, and only those 20."""

    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"),
        write_server(tmp_path / "server.jsonl"),
        write_learner(tmp_path / "learner.jsonl"),
    )
    section = _section(sections, "CONTENTION")
    notes = _notes(section)
    assert f"{N_OVERLAPPED}/{N_SERVER_MATCHED + N_SERVER_EXTRA} Step RPCs" in notes

    during = _row(section, "step_rpc_ms / learner update in flight")
    idle = _row(section, "step_rpc_ms / learner idle")
    assert during["count"] == str(N_OVERLAPPED)
    assert idle["count"] == str(N_SERVER_MATCHED - N_OVERLAPPED)
    # step_rpc for the first 20 matched pairs is 100..119, the rest 120..199.
    assert during["mean"] == "109.500"
    assert during["max"] == "119.000"
    assert idle["mean"] == "159.500"


def test_learner_own_timing_stats_are_reported(tmp_path: Path) -> None:
    """``timing/*`` lives one level down under ``metrics`` -- read it there."""

    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"),
        write_server(tmp_path / "server.jsonl"),
        write_learner(tmp_path / "learner.jsonl"),
    )
    section = _section(sections, "CONTENTION")
    step = _row(section, "timing/learner_step_ms")
    assert step["count"] == str(N_LEARNER_UPDATES)
    assert step["mean"] == "100.000"
    critic = _row(section, "timing/critic_update_ms")
    assert critic["count"] == str(N_LEARNER_UPDATES)
    assert critic["p50"] == "10.500"      # 1..20, index 0.5*19 = 9.5


def test_utd_ratio_is_read_from_the_file_never_hardcoded(tmp_path: Path) -> None:
    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"),
        write_server(tmp_path / "server.jsonl"),
        write_learner(tmp_path / "learner.jsonl", utd_ratio=UTD_RATIO),
    )
    notes = _notes(_section(sections, "CONTENTION"))
    assert f"utd_ratio (read from learner.jsonl): {UTD_RATIO}" in notes
    assert "log_period = 20" in notes


def test_utd_ratio_absent_says_so_instead_of_guessing(tmp_path: Path) -> None:
    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"),
        write_server(tmp_path / "server.jsonl"),
        write_learner(tmp_path / "learner.jsonl", utd_ratio=None),
    )
    notes = _notes(_section(sections, "CONTENTION"))
    assert "not present in this file" in notes
    assert "NOT assumed" in notes


def test_contention_degrades_without_server_file(tmp_path: Path) -> None:
    """Learner stats still come out; the split says why it cannot happen."""

    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"),
        None,
        write_learner(tmp_path / "learner.jsonl"),
    )
    section = _section(sections, "CONTENTION")
    assert analyzer.SERVER_ABSENT in _notes(section)
    assert _row(section, "timing/learner_step_ms")["count"] == str(N_LEARNER_UPDATES)


def test_contention_degrades_without_learner_file(tmp_path: Path) -> None:
    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"), write_server(tmp_path / "server.jsonl")
    )
    assert analyzer.LEARNER_ABSENT in _notes(_section(sections, "CONTENTION"))


# --------------------------------------------------------------------------- #
# 5. intervention                                                               #
# --------------------------------------------------------------------------- #


def test_intervention_counts_and_saturation(tmp_path: Path) -> None:
    sections = _sections(write_actor(tmp_path / "actor.jsonl"))
    section = _section(sections, "INTERVENTION")
    intervened = _row(section, "intervened steps")
    assert intervened["n"] == "40"
    assert intervened["of"] == str(N_ACTOR)

    saturated = _row(section, "saturated (of intervened)")
    assert saturated["n"] == "10"
    assert saturated["of"] == "40"
    assert saturated["fraction"] == "25.0%"

    sidecar = _row(section, "sidecar attached")
    assert sidecar["n"] == "21"

    magnitude = _row(section, "intervention_saturation")
    assert magnitude["count"] == "40"          # only the intervened lines carry it
    assert magnitude["max"] == "0.390"         # 0.01 * 39


def test_intervention_section_degrades_when_fields_absent(tmp_path: Path) -> None:
    path = tmp_path / "actor.jsonl"
    path.write_text(
        json.dumps({"schema": 1, "role": "actor", "seq": 0, "env_step_ms": 5.0}) + "\n",
        encoding="utf-8",
    )
    assert "no intervention fields" in _notes(_section(_sections(path), "INTERVENTION"))


# --------------------------------------------------------------------------- #
# Robustness                                                                    #
# --------------------------------------------------------------------------- #


def test_truncated_last_line_is_tolerated(tmp_path: Path) -> None:
    """A SIGKILLed session leaves a half line; every other line still counts."""

    path = write_actor(tmp_path / "actor.jsonl")
    with open(path, "a", encoding="utf-8") as stream:
        stream.write('{"schema":1,"role":"actor","seq":101,"env_step')  # no newline
    load = analyzer.load_jsonl(path)
    assert load.truncated_tail is True
    assert load.bad_lines == 0
    assert load.count == N_ACTOR

    sections = _sections(path)
    assert _row(_section(sections, "PER-PHASE"), "env_step_ms")["count"] == str(N_ACTOR)
    assert "truncated last line" in _notes(_section(sections, "INPUTS"))


def test_corrupt_middle_line_is_counted_separately(tmp_path: Path) -> None:
    path = tmp_path / "actor.jsonl"
    good = json.dumps({"schema": 1, "role": "actor", "seq": 0, "env_step_ms": 1.0})
    tail = json.dumps({"schema": 1, "role": "actor", "seq": 2, "env_step_ms": 3.0})
    path.write_text(f"{good}\n<<< not json >>>\n{tail}\n", encoding="utf-8")
    load = analyzer.load_jsonl(path)
    assert load.bad_lines == 1
    assert load.truncated_tail is False
    assert load.count == 2


def test_empty_actor_file_produces_a_report_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "actor.jsonl"
    path.write_text("", encoding="utf-8")
    sections = _sections(path)
    text = _console(sections)
    assert analyzer.ACTOR_ABSENT in text
    # Every section still renders.
    for needle in ("INPUTS", "PER-PHASE", "CROSS-HOST JOIN", "LOOP BUDGET",
                   "CONTENTION", "INTERVENTION"):
        _section(sections, needle)


def test_missing_server_file_degrades_with_the_grep_phrase(tmp_path: Path) -> None:
    sections = _sections(
        write_actor(tmp_path / "actor.jsonl"), tmp_path / "nope.jsonl"
    )
    text = _console(sections)
    assert analyzer.SERVER_ABSENT in text
    assert "file not found" in text
    assert "reused" in text.lower()


def test_records_with_no_useful_fields_do_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "actor.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({}),
                json.dumps({"schema": 1, "role": "actor"}),
                json.dumps([1, 2, 3]),
                json.dumps("a bare string"),
                json.dumps({"env_step_ms": None, "step_rpc_ms": "not a number"}),
                json.dumps({"env_step_ms": float("inf")}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    load = analyzer.load_jsonl(path)
    assert load.bad_lines == 2          # the list and the bare string
    text = _console(_sections(path))
    assert "PER-PHASE" in text


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #


def _run(*args: Any) -> subprocess.CompletedProcess:
    assert _SCRIPT.is_file()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *[str(a) for a in args]],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(_REPO_ROOT),
    )


def test_cli_full_run_exits_zero_and_prints_every_section(tmp_path: Path) -> None:
    result = _run(
        "--actor", write_actor(tmp_path / "actor.jsonl"),
        "--server", write_server(tmp_path / "server.jsonl"),
        "--learner", write_learner(tmp_path / "learner.jsonl"),
    )
    assert result.returncode == 0, result.stderr
    for needle in ("0. INPUTS", "1. PER-PHASE", "2. CROSS-HOST JOIN",
                   "3. LOOP BUDGET", "4. LEARNER CONTENTION", "5. INTERVENTION"):
        assert needle in result.stdout, result.stdout
    assert "network_plus_queue_ms" in result.stdout


def test_cli_writes_markdown_with_out(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "latency.md"
    result = _run(
        "--actor", write_actor(tmp_path / "actor.jsonl"),
        "--server", write_server(tmp_path / "server.jsonl"),
        "--out", out,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# ")
    assert "| phase |" in text
    assert "network_plus_queue_ms" in text


def test_cli_missing_actor_file_is_a_clean_error(tmp_path: Path) -> None:
    result = _run("--actor", tmp_path / "does_not_exist.jsonl")
    assert result.returncode == 2
    assert "not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_missing_server_file_still_exits_zero(tmp_path: Path) -> None:
    result = _run(
        "--actor", write_actor(tmp_path / "actor.jsonl"),
        "--server", tmp_path / "absent.jsonl",
    )
    assert result.returncode == 0, result.stderr
    assert analyzer.SERVER_ABSENT in result.stdout
    assert "warning:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_empty_actor_file_exits_zero(tmp_path: Path) -> None:
    path = tmp_path / "actor.jsonl"
    path.write_text("", encoding="utf-8")
    result = _run("--actor", path)
    assert result.returncode == 0, result.stderr
    assert analyzer.ACTOR_ABSENT in result.stdout
    assert "Traceback" not in result.stderr


def test_cli_requires_actor() -> None:
    result = _run("--server", "whatever.jsonl")
    assert result.returncode == 2
    assert "--actor" in result.stderr


# --------------------------------------------------------------------------- #
# LOCAL INFERENCE MODE (sections 6-8)                                           #
#                                                                               #
# Same fixture discipline as above: every distribution is built so that         #
# numpy's linear-interpolation percentile lands ON a sample, so the assertions  #
# are exact constants an operator could recompute by hand.  The three roles     #
# (proxy / uploader / paramsync) and their field names are PINNED by the local  #
# inference design spec -- these fixtures are the analyzer's copy of that       #
# contract, so a wiring commit that renames a field fails here first.           #
# --------------------------------------------------------------------------- #

N_PROXY = 101              # Step replies served locally
N_UPLOADER = 50            # transitions forwarded to the real server
N_PARAMSYNC = 40           # LATEST.json polls
N_PARAMSYNC_FETCHES = 5    # of which actually pulled a blob
DIVERGENT_INDICES = (7, 23, 41)


def write_proxy(path: Path, *, count: int = N_PROXY) -> Path:
    """local_inference 0.1..10.1 ms, queue 0..10, age 0..5 s, version 7..11.

    ``local_inference_ms`` spans the feasibility study's own range (2 ms GPU,
    9.6 ms CPU) on purpose: a fixture that pinned percentiles with 1..101 ms
    would read like a claim that local inference costs 100 ms.
    """

    lines = []
    for index in range(count):
        lines.append(
            json.dumps(
                {
                    "schema": 1,
                    "role": "proxy",
                    "seq": index,
                    "t_epoch": ACTOR_T0 + index,
                    "run_id": RUN_ID,
                    "episode_id": 0,
                    "transition_id": _transition_id(index),
                    "local_inference_ms": round((index + 1) / 10.0, 4),
                    "local_finalize_ms": 0.5,
                    "total_ms": 12.0,
                    "queue_depth": index // 10,
                    "params_version": 7 + index // 25,
                    "params_age_s": float(index % 6),
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_uploader(
    path: Path,
    *,
    count: int = N_UPLOADER,
    divergent: Sequence[int] = DIVERGENT_INDICES,
) -> Path:
    """upload_rpc 100..198 ms, backlog 0..49, oldest 0..24.5 s."""

    lines = []
    for index in range(count):
        lines.append(
            json.dumps(
                {
                    "schema": 1,
                    "role": "uploader",
                    "seq": index,
                    "t_epoch": ACTOR_T0 + index,
                    "transition_id": _transition_id(index),
                    "upload_rpc_ms": 100.0 + 2.0 * index,
                    "backlog_depth": index,
                    "oldest_backlog_s": round(0.5 * index, 3),
                    "outcome_divergence": index in tuple(divergent),
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_paramsync(
    path: Path, *, count: int = N_PARAMSYNC, versions: Sequence[int] | None = None
) -> Path:
    """40 polls, every 8th a real fetch; staleness 0..9 s, version 7..11."""

    lines = []
    for index in range(count):
        record: dict[str, Any] = {
            "schema": 1,
            "role": "paramsync",
            "seq": index,
            "t_epoch": ACTOR_T0 + index,
            "poll_ms": 1.0,
            "staleness_s": float(index % 10),
            "version": (
                versions[index] if versions is not None else 7 + index // 8
            ),
        }
        if index % 8 == 0:
            record["fetch_ms"] = 3000.0 + 200.0 * (index // 8)
            record["load_ms"] = 120.0
            record["swap_ms"] = 0.4
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sections_local(
    actor: Path,
    *,
    server: Path | None = None,
    learner: Path | None = None,
    proxy: Path | None = None,
    uploader: Path | None = None,
    paramsync: Path | None = None,
):
    sections, *_ = analyzer.build_report(
        actor, server, learner, proxy, uploader, paramsync
    )
    return sections


# --------------------------------------------------------------------------- #
# 6. proxy                                                                      #
# --------------------------------------------------------------------------- #


def test_proxy_phase_percentiles_are_exact(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"), proxy=write_proxy(tmp_path / "p.jsonl")
    )
    section = _section(sections, "POLICY PROXY")
    row = _row(section, "local_inference_ms")
    assert row["count"] == str(N_PROXY)
    assert row["mean"] == "5.100"
    assert row["p50"] == "5.100"
    assert row["p90"] == "9.100"
    assert row["p99"] == "10.000"
    assert row["max"] == "10.100"

    finalize = _row(section, "local_finalize_ms")
    assert finalize["count"] == str(N_PROXY)
    assert finalize["max"] == "0.500"

    total = _row(section, "total_ms")
    assert total["count"] == str(N_PROXY)
    assert "contains" in total["note"]


def test_proxy_gauges_are_not_in_the_phase_table(tmp_path: Path) -> None:
    """queue_depth / params_age_s are levels, not durations."""

    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"), proxy=write_proxy(tmp_path / "p.jsonl")
    )
    section = _section(sections, "POLICY PROXY")
    phase_table = next(
        block
        for block in section.blocks
        if isinstance(block, analyzer.Table) and block.headers[0] == "phase"
    )
    assert [row[0] for row in phase_table.rows] == [
        "total_ms",
        "local_inference_ms",
        "local_finalize_ms",
    ]

    depth = _row(section, "queue_depth")
    assert depth["count"] == str(N_PROXY)
    assert depth["p50"] == "5.000"
    assert depth["max"] == "10.000"

    age = _row(section, "params_age_s")
    assert age["count"] == str(N_PROXY)
    assert age["p50"] == "2.000"
    assert age["max"] == "5.000"


def test_proxy_params_version_progression(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"), proxy=write_proxy(tmp_path / "p.jsonl")
    )
    section = _section(sections, "POLICY PROXY")
    assert _row(section, "first")["value"] == "7"
    assert _row(section, "last")["value"] == "11"
    assert _row(section, "distinct")["value"] == "5"
    assert _row(section, "advances")["value"] == "4"
    assert _row(section, "regressions")["value"] == "0"
    assert "WARNING" not in _notes(section)


def test_proxy_version_regression_is_flagged(tmp_path: Path) -> None:
    """The proxy contracts to stamp monotonically; a decrease must be loud."""

    path = tmp_path / "p.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema": 1,
                    "role": "proxy",
                    "seq": index,
                    "local_inference_ms": 2.0,
                    "params_version": version,
                }
            )
            for index, version in enumerate((9, 10, 8, 11))
        )
        + "\n",
        encoding="utf-8",
    )
    section = _section(
        _sections_local(write_actor(tmp_path / "actor.jsonl"), proxy=path),
        "POLICY PROXY",
    )
    assert _row(section, "regressions")["value"] == "1"
    assert _row(section, "advances")["value"] == "2"
    notes = _notes(section)
    assert "WARNING: params_version went BACKWARDS 1 time(s)" in notes


def test_proxy_high_water_queue_is_called_out(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema": 1,
                    "role": "proxy",
                    "seq": index,
                    "local_inference_ms": 2.0,
                    "queue_depth": depth,
                }
            )
            for index, depth in enumerate((1, 501, 3))
        )
        + "\n",
        encoding="utf-8",
    )
    notes = _notes(
        _section(
            _sections_local(write_actor(tmp_path / "actor.jsonl"), proxy=path),
            "POLICY PROXY",
        )
    )
    assert "WARNING: queue_depth reached 501" in notes
    assert str(analyzer.BACKLOG_HIGH_WATER) in notes


# --------------------------------------------------------------------------- #
# 7. uploader                                                                   #
# --------------------------------------------------------------------------- #


def test_uploader_rpc_and_backlog_summary(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"),
        uploader=write_uploader(tmp_path / "u.jsonl"),
    )
    section = _section(sections, "UPLOADER")
    rpc = _row(section, "upload_rpc_ms")
    assert rpc["count"] == str(N_UPLOADER)
    assert rpc["mean"] == "149.000"
    assert rpc["p50"] == "149.000"
    assert rpc["p90"] == "188.200"
    assert rpc["p99"] == "197.020"
    assert rpc["max"] == "198.000"

    depth = _row(section, "backlog_depth")
    assert depth["count"] == str(N_UPLOADER)
    assert depth["p50"] == "24.500"
    assert depth["max"] == "49.000"

    oldest = _row(section, "oldest_backlog_s")
    assert oldest["max"] == "24.500"

    notes = _notes(section)
    assert f"{N_UPLOADER} uploads recorded" in notes
    assert "backlog_depth max 49" in notes
    assert "oldest entry 24.500 s" in notes


def test_uploader_divergence_is_loud(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"),
        uploader=write_uploader(tmp_path / "u.jsonl"),
    )
    section = _section(sections, "UPLOADER")
    row = _row(section, "outcome_divergence")
    assert row["n"] == str(len(DIVERGENT_INDICES))
    assert row["of"] == str(N_UPLOADER)
    assert row["fraction"] == "6.0%"

    notes = _notes(section)
    assert analyzer.DIVERGENCE_BANNER in notes
    assert f"on {len(DIVERGENT_INDICES)}/{N_UPLOADER} forwarded transitions" in notes
    # and it survives into what an operator actually reads
    assert analyzer.DIVERGENCE_BANNER in _console(sections)


def test_uploader_zero_divergence_says_zero_not_silence(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"),
        uploader=write_uploader(tmp_path / "u.jsonl", divergent=()),
    )
    section = _section(sections, "UPLOADER")
    assert _row(section, "outcome_divergence")["n"] == "0"
    notes = _notes(section)
    assert analyzer.DIVERGENCE_BANNER not in notes
    assert f"outcome_divergence: 0/{N_UPLOADER}" in notes


def test_uploader_without_the_divergence_field_says_unknown(tmp_path: Path) -> None:
    """Absent parity data must never read as proven parity."""

    path = tmp_path / "u.jsonl"
    path.write_text(
        json.dumps(
            {"schema": 1, "role": "uploader", "seq": 0, "upload_rpc_ms": 150.0}
        )
        + "\n",
        encoding="utf-8",
    )
    notes = _notes(
        _section(
            _sections_local(write_actor(tmp_path / "actor.jsonl"), uploader=path),
            "UPLOADER",
        )
    )
    assert "UNKNOWN for this session, not proven" in notes


def test_uploader_high_water_backlog_is_called_out(tmp_path: Path) -> None:
    path = tmp_path / "u.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema": 1,
                    "role": "uploader",
                    "seq": index,
                    "upload_rpc_ms": 150.0,
                    "backlog_depth": depth,
                    "outcome_divergence": False,
                }
            )
            for index, depth in enumerate((10, 900))
        )
        + "\n",
        encoding="utf-8",
    )
    notes = _notes(
        _section(
            _sections_local(write_actor(tmp_path / "actor.jsonl"), uploader=path),
            "UPLOADER",
        )
    )
    assert f"exceeded the {analyzer.BACKLOG_HIGH_WATER} high-water mark" in notes
    assert "Nothing was dropped" in notes


# --------------------------------------------------------------------------- #
# 8. param sync                                                                 #
# --------------------------------------------------------------------------- #


def test_paramsync_phase_tables(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"),
        paramsync=write_paramsync(tmp_path / "s.jsonl"),
    )
    section = _section(sections, "PARAM SYNC")
    poll = _row(section, "poll_ms")
    assert poll["count"] == str(N_PARAMSYNC)
    assert poll["mean"] == "1.000"

    fetch = _row(section, "fetch_ms")
    assert fetch["count"] == str(N_PARAMSYNC_FETCHES)
    assert fetch["mean"] == "3400.000"
    assert fetch["p50"] == "3400.000"
    assert fetch["p90"] == "3720.000"
    assert fetch["max"] == "3800.000"

    assert _row(section, "load_ms")["count"] == str(N_PARAMSYNC_FETCHES)
    assert _row(section, "swap_ms")["max"] == "0.400"

    notes = _notes(section)
    assert f"{N_PARAMSYNC} polls, {N_PARAMSYNC_FETCHES} carried a fetch" in notes


def test_paramsync_staleness_and_version(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"),
        paramsync=write_paramsync(tmp_path / "s.jsonl"),
    )
    section = _section(sections, "PARAM SYNC")
    stale = _row(section, "staleness_s")
    assert stale["count"] == str(N_PARAMSYNC)
    assert stale["mean"] == "4.500"
    assert stale["p50"] == "4.500"
    assert stale["max"] == "9.000"

    assert _row(section, "first")["value"] == "7"
    assert _row(section, "last")["value"] == "11"
    assert _row(section, "advances")["value"] == "4"
    assert _row(section, "regressions")["value"] == "0"


def test_paramsync_version_regression_is_flagged(tmp_path: Path) -> None:
    versions = [7 + index // 8 for index in range(N_PARAMSYNC)]
    versions[20] = 3          # a blob older than the loaded one got applied
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"),
        paramsync=write_paramsync(tmp_path / "s.jsonl", versions=versions),
    )
    section = _section(sections, "PARAM SYNC")
    assert _row(section, "regressions")["value"] == "1"
    assert "WARNING: version went BACKWARDS 1 time(s)" in _notes(section)


# --------------------------------------------------------------------------- #
# Section 3 must be re-read in local mode                                       #
# --------------------------------------------------------------------------- #


def test_loop_budget_flags_the_local_hop_when_proxy_is_present(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"), proxy=write_proxy(tmp_path / "p.jsonl")
    )
    notes = _notes(_section(sections, "LOOP BUDGET"))
    assert analyzer.LOCAL_HOP_PHRASE in notes
    assert "127.0.0.1" in notes
    assert f"{N_PROXY} records" in notes


def test_loop_budget_says_nothing_about_local_mode_without_proxy(tmp_path: Path) -> None:
    notes = _notes(
        _section(_sections(write_actor(tmp_path / "actor.jsonl")), "LOOP BUDGET")
    )
    assert analyzer.LOCAL_HOP_PHRASE not in notes


# --------------------------------------------------------------------------- #
# Degradation + purity of the remote-mode report                                #
# --------------------------------------------------------------------------- #


def test_remote_mode_report_states_local_mode_was_not_given(tmp_path: Path) -> None:
    sections = _sections(write_actor(tmp_path / "actor.jsonl"))
    section = _section(sections, "LOCAL INFERENCE MODE")
    assert analyzer.LOCAL_MODE_NOT_GIVEN in _notes(section)
    # ... and the three per-role sections are NOT rendered.
    titles = [s.title for s in sections]
    assert not any("POLICY PROXY" in title for title in titles)
    assert not any("UPLOADER" in title for title in titles)
    assert not any("PARAM SYNC" in title for title in titles)


def test_absent_local_files_degrade_with_grep_phrases(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"),
        proxy=tmp_path / "no_proxy.jsonl",
        uploader=tmp_path / "no_uploader.jsonl",
        paramsync=tmp_path / "no_paramsync.jsonl",
    )
    text = _console(sections)
    for phrase in (
        analyzer.PROXY_ABSENT,
        analyzer.UPLOADER_ABSENT,
        analyzer.PARAMSYNC_ABSENT,
    ):
        assert phrase in text
    assert text.count("file not found") >= 3


def test_one_local_file_does_not_degrade_the_other_two_silently(tmp_path: Path) -> None:
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"), proxy=write_proxy(tmp_path / "p.jsonl")
    )
    assert _row(_section(sections, "POLICY PROXY"), "local_inference_ms")["count"] == str(
        N_PROXY
    )
    assert analyzer.UPLOADER_ABSENT in _notes(_section(sections, "UPLOADER"))
    assert analyzer.PARAMSYNC_ABSENT in _notes(_section(sections, "PARAM SYNC"))
    assert "not given" in _notes(_section(sections, "PARAM SYNC"))


def test_empty_local_files_are_not_mistaken_for_absent_flags(tmp_path: Path) -> None:
    for name in ("p.jsonl", "u.jsonl", "s.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")
    sections = _sections_local(
        write_actor(tmp_path / "actor.jsonl"),
        proxy=tmp_path / "p.jsonl",
        uploader=tmp_path / "u.jsonl",
        paramsync=tmp_path / "s.jsonl",
    )
    text = _console(sections)
    assert "held no parsable records" in text
    assert "HIL_LATENCY_PROFILE=1 was in ITS process environment" in text


def test_local_flags_do_not_change_the_existing_sections(tmp_path: Path) -> None:
    """The remote-mode half of the report must be byte-identical either way."""

    actor = write_actor(tmp_path / "actor.jsonl")
    server = write_server(tmp_path / "server.jsonl")
    learner = write_learner(tmp_path / "learner.jsonl")
    plain = _sections(actor, server, learner)
    local = _sections_local(
        actor,
        server=server,
        learner=learner,
        proxy=write_proxy(tmp_path / "p.jsonl"),
        uploader=write_uploader(tmp_path / "u.jsonl"),
        paramsync=write_paramsync(tmp_path / "s.jsonl"),
    )
    for needle in ("PER-PHASE", "CROSS-HOST JOIN", "CONTENTION", "INTERVENTION"):
        assert _console([_section(plain, needle)]) == _console(
            [_section(local, needle)]
        ), needle


def test_local_records_with_junk_fields_do_not_crash(tmp_path: Path) -> None:
    for name in ("p.jsonl", "u.jsonl", "s.jsonl"):
        (tmp_path / name).write_text(
            "\n".join(
                [
                    json.dumps({}),
                    json.dumps({"local_inference_ms": None, "queue_depth": "deep"}),
                    json.dumps({"params_version": float("nan"), "version": "v3"}),
                    json.dumps({"outcome_divergence": "maybe"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    text = _console(
        _sections_local(
            write_actor(tmp_path / "actor.jsonl"),
            proxy=tmp_path / "p.jsonl",
            uploader=tmp_path / "u.jsonl",
            paramsync=tmp_path / "s.jsonl",
        )
    )
    assert "6. LOCAL MODE" in text
    assert "8. LOCAL MODE" in text


# --------------------------------------------------------------------------- #
# CLI -- local mode                                                             #
# --------------------------------------------------------------------------- #


def test_cli_local_mode_prints_sections_six_to_eight(tmp_path: Path) -> None:
    result = _run(
        "--actor", write_actor(tmp_path / "actor.jsonl"),
        "--proxy", write_proxy(tmp_path / "p.jsonl"),
        "--uploader", write_uploader(tmp_path / "u.jsonl"),
        "--paramsync", write_paramsync(tmp_path / "s.jsonl"),
    )
    assert result.returncode == 0, result.stderr
    for needle in (
        "6. LOCAL MODE -- POLICY PROXY",
        "7. LOCAL MODE -- TRANSITION UPLOADER",
        "8. LOCAL MODE -- PARAM SYNC",
        "local_inference_ms",
        "params_age_s",
        "backlog_depth",
        "staleness_s",
        analyzer.DIVERGENCE_BANNER,
        analyzer.LOCAL_HOP_PHRASE,
    ):
        assert needle in result.stdout, needle
    # the remote-mode sections are still all there
    for needle in ("0. INPUTS", "1. PER-PHASE", "3. LOOP BUDGET", "5. INTERVENTION"):
        assert needle in result.stdout, needle
    assert "Traceback" not in result.stderr


def test_cli_missing_local_files_still_exit_zero(tmp_path: Path) -> None:
    result = _run(
        "--actor", write_actor(tmp_path / "actor.jsonl"),
        "--proxy", tmp_path / "absent_proxy.jsonl",
        "--uploader", tmp_path / "absent_uploader.jsonl",
        "--paramsync", tmp_path / "absent_paramsync.jsonl",
    )
    assert result.returncode == 0, result.stderr
    assert analyzer.PROXY_ABSENT in result.stdout
    assert analyzer.UPLOADER_ABSENT in result.stdout
    assert analyzer.PARAMSYNC_ABSENT in result.stdout
    assert "warning: --proxy" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_local_mode_markdown_carries_the_new_sections(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "local.md"
    result = _run(
        "--actor", write_actor(tmp_path / "actor.jsonl"),
        "--proxy", write_proxy(tmp_path / "p.jsonl"),
        "--uploader", write_uploader(tmp_path / "u.jsonl"),
        "--paramsync", write_paramsync(tmp_path / "s.jsonl"),
        "--out", out,
    )
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "## 6. LOCAL MODE -- POLICY PROXY (laptop3)" in text
    assert "| gauge |" in text
    assert "| progression | value |" in text
    assert analyzer.DIVERGENCE_BANNER in text
