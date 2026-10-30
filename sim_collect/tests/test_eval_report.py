"""Tests for sim_collect/eval/report.py on synthetic run_eval output dirs.

No MuJoCo, no torch, no socket: the report tool is stdlib-only by design, and
these tests build ``episodes.jsonl`` / ``summary.json`` by hand so the exact
shape of a missing/garbled field can be pinned.

    cd /home/laptop3/gello_software && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
      .venv/bin/python -m pytest -q -p no:cacheprovider \\
      sim_collect/tests/test_eval_report.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from sim_collect.eval import report as R


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def write_run(
    root: Path,
    name: str,
    episodes,
    summary: dict | None = None,
    *,
    raw_lines: list[str] | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / R.EPISODES_JSONL, "w", encoding="utf-8") as stream:
        for ep in episodes:
            stream.write(json.dumps(ep) + "\n")
        for line in raw_lines or []:
            stream.write(line + "\n")
    if summary is not None:
        (run_dir / R.SUMMARY_JSON).write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def simple_episodes(outcomes, t0: float = 5.0):
    """One episode per outcome, seeds 0..n-1, successes carrying a t_success_s."""
    out = []
    for seed, outcome in enumerate(outcomes):
        row = {"seed": seed, "outcome": outcome, "n_steps": 100 + seed,
               "wall_s": 12.5 + seed}
        if outcome == "success":
            row["t_success_s"] = t0 + seed
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_wilson_matches_closed_form_at_half():
    lo, hi = R.wilson_interval(10, 20)
    # Textbook Wilson for 10/20: 0.5 +- ~0.2050 (symmetric at p = 0.5).
    assert lo == pytest.approx(0.2993, abs=5e-4)
    assert hi == pytest.approx(0.7007, abs=5e-4)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-12)


def test_wilson_never_collapses_at_the_extremes():
    """The reason Wilson is used at all: 0/20 and 20/20 must not be points."""
    lo0, hi0 = R.wilson_interval(0, 20)
    lo1, hi1 = R.wilson_interval(20, 20)
    assert lo0 == 0.0 and 0.1 < hi0 < 0.2
    assert 0.8 < lo1 < 0.9 and hi1 == 1.0


def test_wilson_empty_is_vacuous():
    assert R.wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_values():
    assert R.mcnemar_exact(0, 0) == 1.0
    assert R.mcnemar_exact(3, 3) == pytest.approx(1.0)
    # 6 discordant pairs all one way: 2 * (1/64) = 0.03125 -- the threshold case
    # the report text quotes.
    assert R.mcnemar_exact(6, 0) == pytest.approx(0.03125)
    assert R.mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert R.mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert R.mcnemar_exact(4, 1) == pytest.approx(2 * (1 + 5) / 32)


# ---------------------------------------------------------------------------
# normalization / loading
# ---------------------------------------------------------------------------
def test_normalize_prefers_explicit_success_bool():
    ep = R.normalize_episode({"seed": 3, "success": True})
    assert ep["outcome"] == "success" and ep["success"] is True
    ep = R.normalize_episode({"seed": 3, "success": False})
    assert ep["outcome"] == "failure" and ep["success"] is False


def test_normalize_unknown_outcome_is_not_a_failure():
    ep = R.normalize_episode({"seed": 1, "n_steps": 4})
    assert ep["outcome"] == R.UNKNOWN
    assert ep["success"] is None


def test_normalize_accepts_alternate_spellings_and_dict_detail():
    ep = R.normalize_episode({
        "episode_seed": 7, "result": "timeout", "steps": 600,
        "time_to_success_s": None, "failure_reason": {"why": "carrot outside"},
        "elapsed_s": 31.25,
    })
    assert ep["seed"] == 7 and ep["outcome"] == "timeout" and ep["n_steps"] == 600
    assert ep["success"] is False
    assert ep["wall_s"] == pytest.approx(31.25)
    assert ep["detail"] == '{"why":"carrot outside"}'


def test_load_run_counts_sr_ci_and_times(tmp_path):
    run = write_run(
        tmp_path, "a",
        simple_episodes(["success", "success", "timeout", "fault", "success"]),
        {"policy": "zmq://127.0.0.1:5593", "checkpoint": "ckpt_40k",
         "git_commit": "deadbeef", "config_sha": "0" * 64},
    )
    loaded = R.load_run(run)
    assert loaded["n_episodes"] == 5 and loaded["n_scored"] == 5
    assert loaded["successes"] == 3
    assert loaded["sr"] == pytest.approx(0.6)
    assert loaded["counts"] == {"success": 3, "timeout": 1, "fault": 1}
    assert loaded["t_success"]["n"] == 3
    # successes are seeds 0, 1, 4 -> t = 5, 6, 9
    assert loaded["t_success"]["median"] == pytest.approx(6.0)
    assert loaded["t_success"]["mean"] == pytest.approx(20 / 3)
    assert loaded["t_success"]["min"] == pytest.approx(5.0)
    assert loaded["t_success"]["max"] == pytest.approx(9.0)
    lo, hi = loaded["ci"]
    assert 0.0 < lo < 0.6 < hi < 1.0
    assert loaded["checkpoint"] == "ckpt_40k"
    assert loaded["errors"] == []


def test_load_run_survives_malformed_lines_and_missing_summary(tmp_path):
    run = write_run(
        tmp_path, "broken",
        simple_episodes(["success", "timeout"]),
        None,
        raw_lines=["{not json", "[1,2,3]", "", "   "],
    )
    loaded = R.load_run(run)
    assert loaded["n_episodes"] == 2
    assert loaded["sr"] == pytest.approx(0.5)
    joined = " ".join(loaded["errors"])
    assert "malformed" in joined and "summary.json" in joined


def test_unknown_outcomes_leave_the_denominator(tmp_path):
    run = write_run(tmp_path, "u", [
        {"seed": 0, "outcome": "success"},
        {"seed": 1, "outcome": "timeout"},
        {"seed": 2},                       # no usable outcome
    ])
    loaded = R.load_run(run)
    assert loaded["n_episodes"] == 3
    assert loaded["n_scored"] == 2 and loaded["n_unknown"] == 1
    assert loaded["sr"] == pytest.approx(0.5)   # NOT 1/3


def test_time_to_success_is_derived_only_when_absent(tmp_path):
    run = write_run(
        tmp_path, "d",
        [{"seed": 0, "outcome": "success", "n_steps": 300},
         {"seed": 1, "outcome": "success", "n_steps": 60, "t_success_s": 1.0},
         {"seed": 2, "outcome": "timeout", "n_steps": 600}],
        {"control_hz": 30.0},
    )
    loaded = R.load_run(run)
    assert loaded["t_success"]["derived"] == 1
    times = sorted(ep["t_success_s"] for ep in loaded["episodes"] if ep["success"])
    assert times == pytest.approx([1.0, 10.0])
    # the timed-out episode is never given a derived time
    assert loaded["episodes"][2]["t_success_s"] is None


def test_missing_episodes_file_is_reported_not_raised(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    loaded = R.load_run(empty)
    assert loaded["n_episodes"] == 0 and loaded["sr"] is None
    assert any("episodes.jsonl" in e for e in loaded["errors"])


# ---------------------------------------------------------------------------
# cross-run views
# ---------------------------------------------------------------------------
def test_per_seed_table_marks_missing_seeds_apart_from_failures(tmp_path):
    a = write_run(tmp_path, "a", simple_episodes(["success", "timeout", "success"]))
    b = write_run(tmp_path, "b", [{"seed": 0, "outcome": "fault"},
                                  {"seed": 1, "outcome": "success"}])
    report = R.build_report([a, b])
    seeds, by_run = report["seeds"], report["per_seed"]
    assert seeds == [0, 1, 2]
    assert by_run["a"][2] == "success"
    assert 2 not in by_run["b"]             # never attempted, not a failure
    markdown = R.render_markdown(report)
    assert "## 3. per-seed outcomes" in markdown
    assert "✓" in markdown


def test_duplicate_labels_are_disambiguated(tmp_path):
    (tmp_path / "x1").mkdir()
    (tmp_path / "x2").mkdir()
    a = write_run(tmp_path / "x1", "run", simple_episodes(["success"]))
    b = write_run(tmp_path / "x2", "run", simple_episodes(["timeout"]))
    report = R.build_report([a, b])
    labels = [r["label"] for r in report["runs"]]
    assert labels[0] != labels[1]
    assert set(report["per_seed"]) == set(labels)


def test_pairwise_compare_pairs_on_seeds(tmp_path):
    a = write_run(tmp_path, "a", [
        {"seed": 0, "outcome": "success"},
        {"seed": 1, "outcome": "success"},
        {"seed": 2, "outcome": "timeout"},
        {"seed": 3, "outcome": "timeout"},
    ])
    b = write_run(tmp_path, "b", [
        {"seed": 0, "outcome": "success"},
        {"seed": 1, "outcome": "fault"},
        {"seed": 2, "outcome": "success"},
        {"seed": 4, "outcome": "success"},      # not in a
    ])
    report = R.build_report([a, b], compare=True)
    assert len(report["comparisons"]) == 1
    cmp = report["comparisons"][0]
    assert cmp["shared_seeds"] == 3
    assert (cmp["both"], cmp["a_only"], cmp["b_only"], cmp["neither"]) == (1, 1, 1, 0)
    assert cmp["a_only_seeds"] == [1] and cmp["b_only_seeds"] == [2]
    assert cmp["only_in_a"] == [3] and cmp["only_in_b"] == [4]
    assert cmp["mcnemar_p"] == pytest.approx(1.0)
    assert cmp["sr_delta"] == pytest.approx(0.75 - 0.5)
    markdown = R.render_markdown(report)
    assert "## 4. paired comparison" in markdown
    assert "seed lists differ" in markdown


def test_compare_section_absent_without_flag(tmp_path):
    a = write_run(tmp_path, "a", simple_episodes(["success"]))
    b = write_run(tmp_path, "b", simple_episodes(["timeout"]))
    assert "## 4. paired comparison" not in R.render_markdown(R.build_report([a, b]))


# ---------------------------------------------------------------------------
# rendering / CLI
# ---------------------------------------------------------------------------
def test_markdown_has_every_section_and_the_ci_caveat(tmp_path):
    run = write_run(tmp_path, "solo", simple_episodes(["success", "timeout"]),
                    {"checkpoint": "ckpt_80k", "git_commit": "abc1234"})
    markdown = R.render_markdown(R.build_report([run]))
    for heading in ("# sim eval report", "## 1. success rate", "## 2. outcome breakdown",
                    "## 3. per-seed outcomes", "## 5. provenance"):
        assert heading in markdown
    assert "Wilson" in markdown and "Overlapping" in markdown
    assert "ckpt_80k" in markdown and "abc1234" in markdown
    assert "50.0%" in markdown


def test_markdown_renders_with_zero_usable_outcomes(tmp_path):
    run = write_run(tmp_path, "weird", [{"foo": 1}, {"bar": 2}])
    markdown = R.render_markdown(R.build_report([run]))
    assert "n/a" in markdown          # SR is n/a, nothing raised
    assert "no usable outcome" in markdown


def test_main_writes_markdown_and_csvs(tmp_path, capsys):
    a = write_run(tmp_path, "a", simple_episodes(["success", "timeout", "success"]),
                  {"checkpoint": "A"})
    b = write_run(tmp_path, "b", simple_episodes(["success", "fault", "fault"]),
                  {"checkpoint": "B"})
    out = tmp_path / "out"
    rc = R.main([str(a), str(b), "--label", "A", "--label", "B",
                 "--compare", "--out", str(out), "--json", str(out / "report.json"),
                 "--quiet"])
    assert rc == 0
    assert (out / "report.md").is_file()
    assert (out / "runs.csv").is_file()
    assert (out / "per_seed.csv").is_file()
    assert json.loads((out / "report.json").read_text())["runs"]

    with open(out / "episodes.csv", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {r["run"] for r in rows} == {"A", "B"}
    a_success = [r for r in rows if r["run"] == "A" and r["outcome"] == "success"]
    assert len(a_success) == 2 and all(r["success"] == "1" for r in a_success)

    with open(out / "runs.csv", encoding="utf-8") as stream:
        run_rows = {r["run"]: r for r in csv.DictReader(stream)}
    assert float(run_rows["A"]["sr"]) == pytest.approx(2 / 3)
    assert run_rows["B"]["n_fault"] == "2"
    assert 0.0 < float(run_rows["A"]["ci_lo"]) < float(run_rows["A"]["ci_hi"]) <= 1.0

    with open(out / "per_seed.csv", encoding="utf-8") as stream:
        seed_rows = list(csv.DictReader(stream))
    assert [r["seed"] for r in seed_rows] == ["0", "1", "2"]
    assert seed_rows[1]["A"] == "timeout" and seed_rows[1]["B"] == "fault"

    # --quiet keeps stdout clean so the tool can be piped
    assert capsys.readouterr().out == ""


def test_main_prints_markdown_by_default(tmp_path, capsys):
    run = write_run(tmp_path, "a", simple_episodes(["success"]))
    assert R.main([str(run)]) == 0
    assert "# sim eval report" in capsys.readouterr().out


def test_main_refuses_when_nothing_is_readable(tmp_path, capsys):
    assert R.main([str(tmp_path / "does_not_exist")]) == 2
    assert "no episodes were readable" in capsys.readouterr().err


def test_main_rejects_more_labels_than_runs(tmp_path):
    run = write_run(tmp_path, "a", simple_episodes(["success"]))
    assert R.main([str(run), "--label", "x", "--label", "y"]) == 2


def test_report_module_is_importable_without_heavy_deps():
    """It must stay stdlib-only: no numpy/mujoco/torch/zmq at import time."""
    source = Path(R.__file__).read_text(encoding="utf-8")
    for banned in ("import numpy", "import mujoco", "import torch", "import zmq", "import cv2"):
        assert banned not in source, banned
    assert math.isfinite(R.Z95)


# ---------------------------------------------------------------------------
# alignment with what run_eval actually writes (fields read from run_eval.py)
# ---------------------------------------------------------------------------
def test_checkpoint_is_found_inside_policy_meta(tmp_path):
    """run_eval.summarize puts the RESET v2 fields under `policy_meta`."""
    run = write_run(tmp_path, "pm", simple_episodes(["success"]),
                    {"policy": "zmq://127.0.0.1:5593",
                     "policy_meta": {"policy_type": "fm", "checkpoint": "/k/last/pretrained_model"}})
    assert R.load_run(run)["checkpoint"] == "/k/last/pretrained_model"


def test_run_eval_outcome_vocabulary_round_trips(tmp_path):
    """success / timeout / fault / failure are world.Outcome's four literals."""
    run = write_run(tmp_path, "vocab", [
        {"episode": "s00", "seed": 0, "outcome": "success", "n_steps": 120, "t_success_s": 4.0},
        {"episode": "s01", "seed": 1, "outcome": "timeout", "n_steps": 600},
        {"episode": "s02", "seed": 2, "outcome": "fault", "n_steps": 3, "fault": "reset: Timeout"},
        {"episode": "s03", "seed": 3, "outcome": "failure", "n_steps": 88, "failure": "carrot off table"},
    ])
    loaded = R.load_run(run)
    assert loaded["counts"] == {"success": 1, "timeout": 1, "fault": 1, "failure": 1}
    assert loaded["sr"] == pytest.approx(0.25)
    # fault/failure text wins over the (usually empty) task detail, as in summary.md
    details = {ep["episode"]: ep["detail"] for ep in loaded["episodes"]}
    assert details["s02"] == "reset: Timeout"
    assert details["s03"] == "carrot off table"


def test_detail_prefers_fault_over_detail(tmp_path):
    ep = R.normalize_episode({"outcome": "fault", "fault": "act: timeout",
                              "detail": "food not in container"})
    assert ep["detail"] == "act: timeout"


def test_replay_runs_without_seeds_pair_on_episode_id(tmp_path):
    """A replay run has one episode per take and seed=None -- the id is the key."""
    a = write_run(tmp_path, "ra", [
        {"episode": "take_01", "seed": None, "outcome": "success"},
        {"episode": "take_02", "seed": None, "outcome": "timeout"},
    ])
    b = write_run(tmp_path, "rb", [
        {"episode": "take_01", "seed": None, "outcome": "timeout"},
        {"episode": "take_02", "seed": None, "outcome": "timeout"},
    ])
    report = R.build_report([a, b], compare=True)
    assert report["seeds"] == ["take_01", "take_02"]
    cmp = report["comparisons"][0]
    assert cmp["shared_seeds"] == 2
    assert (cmp["both"], cmp["a_only"], cmp["b_only"], cmp["neither"]) == (0, 1, 0, 1)
    assert cmp["a_only_seeds"] == ["take_01"]
    # mixed int/str keys across runs must still sort without raising
    R.render_markdown(R.build_report([a, write_run(tmp_path, "rc", simple_episodes(["success"]))]))


def test_episodes_csv_carries_the_episode_id(tmp_path):
    run = write_run(tmp_path, "c", [{"episode": "s07", "seed": 7, "outcome": "success",
                                     "t_success_s": 3.5, "wall_s": 9.0}])
    out = tmp_path / "out"
    assert R.main([str(run), "--out", str(out), "--quiet"]) == 0
    with open(out / "episodes.csv", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["episode"] == "s07" and row["seed"] == "7"
    assert row["t_success_s"].startswith("3.5") and row["t_success_derived"] == "0"
