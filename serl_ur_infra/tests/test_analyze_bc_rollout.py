"""CLI contract tests for ``scripts/analyze_bc_rollout.py``.

WHAT THE ANALYZER IS FOR
------------------------
A BC evaluation run leaves two piles of evidence on two different machines:
the SERVED side (what the policy server wrote -- ``metadata.json``,
``actions.jsonl``, ``<run_id>/episode_XXXX.pkl``, and, when logging was on,
``inference.jsonl``) and the ROBOT side (what the laptop recorded).  The
analyzer is the thing an operator runs afterwards to find out what happened,
so its job is to produce a report even when half the evidence is missing --
a rollout whose ``inference.jsonl`` was disabled, or whose robot recording was
never started, must still be summarised, with the gap named rather than
silently rendered as a zero.

WHY THE FIXTURE IS BUILT BY THE REAL SINK
-----------------------------------------
The served directory here is written by
``ur_env.bc_recording_sink.EpisodeRecordingSink`` itself, fed transitions
shaped like the ones ``ur_env/remote_actor.py::build_data`` puts on the wire
(the same fixture style as ``tests/test_bc_recording_sink.py``).  A hand-rolled
directory would only prove the analyzer agrees with this test file's guess at
the layout; letting the sink write it means the two modules are pinned to each
other.  The counts below are therefore ground truth, not restated
expectations:

    2 episodes / 5 transitions / 1 success
      episode 0 -- 3 steps, terminal ``dones``, operator MARK SUCCESS
      episode 1 -- 2 steps, terminal ``truncated`` (END EPISODE), no success

The h5 / robot-side path is deliberately NOT exercised here: those fixtures
belong to the recorder work.  What is pinned is only that omitting ``--robot``
yields a report that says so.

Assertions are about structure and behaviour, never about exact wording: the
report's key names are matched against candidate spellings and its numbers
against the fixture, so the analyzer's phrasing stays free to change.

Run (from ``/home/laptop3/gello_software``)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
      -p no:cacheprovider serl_ur_infra/tests/test_analyze_bc_rollout.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pytest


_HERE = Path(os.path.abspath(__file__)).parent
_SERL_UR_INFRA = _HERE.parent
_REPO_ROOT = _SERL_UR_INFRA.parent
_SERL_LAUNCHER = _REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher"
_SCRIPT = _SERL_UR_INFRA / "scripts" / "analyze_bc_rollout.py"

sys.path.insert(0, str(_SERL_UR_INFRA))

from ur_env.actor_network import SCHEMA_VERSION  # noqa: E402
from ur_env.bc_recording_sink import EpisodeRecordingSink  # noqa: E402


ARTIFACT_SHA = "8ffcfac5" + "0" * 56
MODEL_ID = "bc-cube-in-cup-raw0731-bcinit-v1"
RUN_ID = "bc_eval_20260731_000000"
POLICY_VERSION = 1

ACTION_DIM = 7
STATE_DIM = 19

#: Ground truth of the fixture below.
EXPECTED_EPISODES = 2
EXPECTED_STEPS = 5
EXPECTED_SUCCESSES = 1

ACTION = np.array([0.1, -0.1, 0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Served-directory fixture (written by the production sink).                    #
# --------------------------------------------------------------------------- #
def _observation(value: int = 0) -> dict[str, np.ndarray]:
    """Canonical observation: cam1/cam2 uint8 (1,128,128,3), state f32 (1,19)."""

    return {
        "cam1": np.full((1, 128, 128, 3), value % 256, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), (255 - value) % 256, dtype=np.uint8),
        "state": np.full((1, STATE_DIM), value / 100.0, dtype=np.float32),
    }


def _data(
    *,
    episode_id: int,
    step_id: int,
    dones: bool = False,
    truncated: bool = False,
    success: bool = False,
    intervened: bool = False,
) -> dict[str, Any]:
    """One ``{"meta", "transition"}`` wrapper exactly as the actor sends it.

    Field placement mirrors ``ActorSessionService._validate_data``.  A success
    is spelled the way the service's identity finalizer spells it on the
    MANUAL path -- ``operator_success`` in meta promoted to ``rewards=1``,
    ``masks=0``, ``dones=True``, ``success=1`` -- so an analyzer that reads any
    one of those fields sees the same single success.
    """

    assert not (dones and truncated), "the wire contract forbids done AND truncated"
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "actor_id": "laptop3-bc-eval",
        "session_id": f"{RUN_ID}:{episode_id}",
        "transition_id": f"{RUN_ID}:{episode_id}:{step_id}",
        "env_step": int(step_id),
        "timestamp_ns": 1_700_000_000_000_000_000 + int(step_id),
        "policy_version": POLICY_VERSION,
        "policy_action": ACTION.copy(),
        "intervened": int(bool(intervened)),
        "auto_success": 0,
        "operator_success": int(bool(success)),
    }
    transition = {
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "observation_id": f"{RUN_ID}:{episode_id}:obs{step_id}",
        "next_observation_id": f"{RUN_ID}:{episode_id}:obs{step_id + 1}",
        "observations": _observation(step_id),
        "next_observations": _observation(step_id + 1),
        "actions": ACTION.copy(),
        "rewards": 1.0 if success else 0.0,
        "masks": 0.0 if dones else 1.0,
        "dones": bool(dones),
        "truncated": bool(truncated),
        "grasp_penalty": 0.0,
        "success": np.uint8(1 if success else 0),
        "classifier_evaluated": np.uint8(0),
        "classifier_probability": 0.0,
        "classifier_threshold": 0.0,
        "classifier_success": np.uint8(0),
        "reward_model_id": "",
    }
    return {"meta": meta, "transition": transition}


def _inference_line(step: int) -> str:
    """One log line in the shape ``InferenceLoggingPolicy`` writes."""

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "latency_ms": 12.5 + step,
        "deterministic": True,
        "policy_version": POLICY_VERSION,
        "action": [float(value) for value in ACTION],
        "state": [round(step / 100.0, 4)] * STATE_DIM,
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _build_served_dir(root: Path, *, with_inference: bool = True) -> Path:
    """Write the 2-episode / 5-step / 1-success rollout with the real sink."""

    root.mkdir(parents=True, exist_ok=True)
    sink = EpisodeRecordingSink(root, artifact_sha256=ARTIFACT_SHA, model_id=MODEL_ID)

    # Episode 0: three steps, operator MARK SUCCESS on the last one.
    for step in range(3):
        last = step == 2
        sink(
            _data(episode_id=0, step_id=step, dones=last, success=last),
            False,
        )
    # Episode 1: two steps, END EPISODE truncation, no success.
    for step in range(2):
        last = step == 1
        sink(
            _data(episode_id=1, step_id=step, truncated=last, intervened=last),
            last,
        )

    assert sink.replay_count == EXPECTED_STEPS
    assert (root / "metadata.json").is_file()
    assert (root / "actions.jsonl").is_file()
    assert (root / RUN_ID / "episode_0000.pkl").is_file()
    assert (root / RUN_ID / "episode_0001.pkl").is_file()

    if with_inference:
        lines = [_inference_line(step) for step in range(EXPECTED_STEPS)]
        (root / "inference.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return root


# --------------------------------------------------------------------------- #
# Running the CLI                                                              #
# --------------------------------------------------------------------------- #
def _analyzer_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_SERL_UR_INFRA), str(_SERL_LAUNCHER), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env


def _run_analyzer(*args: Any, expect_success: bool = True):
    """Run the CLI out-of-process with THIS interpreter.

    ``sys.executable`` is the actor venv python under the canonical run
    command, which is the interpreter an operator uses for everything in this
    stack -- so a dependency the analyzer needs but that venv lacks fails here
    instead of on the operator's terminal.
    """

    assert _SCRIPT.is_file(), f"pinned CLI is missing: {_SCRIPT}"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), *[str(item) for item in args]],
        capture_output=True,
        text=True,
        timeout=300,
        env=_analyzer_env(),
        cwd=str(_REPO_ROOT),
    )
    if expect_success:
        assert result.returncode == 0, (
            f"analyzer exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def _read_report(out_dir: Path) -> dict[str, Any]:
    report_path = out_dir / "report.json"
    assert report_path.is_file(), f"expected a machine-readable report at {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(report, dict), "report.json must hold a JSON object"
    return report


def _read_markdown(out_dir: Path) -> str:
    markdown_path = out_dir / "rollout_report.md"
    assert markdown_path.is_file(), f"expected the operator report at {markdown_path}"
    text = markdown_path.read_text(encoding="utf-8")
    assert text.strip(), "rollout_report.md must not be empty"
    return text


# --------------------------------------------------------------------------- #
# Tolerant readers: pin the numbers, not the key spellings.                     #
# --------------------------------------------------------------------------- #
def _walk(node: Any):
    """Yield every ``(key, value)`` pair in the report, at any depth."""

    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key), value
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _values_for(report: dict[str, Any], names: Iterable[str]) -> list[Any]:
    wanted = {name.lower() for name in names}
    return [value for key, value in _walk(report) if key.lower() in wanted]


def _counts_for(report: dict[str, Any], names: Iterable[str], label: str) -> list[float]:
    """Every value under ``names`` read as a count.

    A container counts as its length, so a report that lists two episode
    objects under ``"episodes"`` satisfies the same assertion as one that
    writes ``"episode_count": 2``.
    """

    values = _values_for(report, names)
    assert values, (
        f"report.json must state {label}; looked for keys {sorted(set(names))}, "
        f"found {sorted({key for key, _ in _walk(report)})}"
    )
    counts: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            counts.append(float(value))
        elif isinstance(value, (list, dict)):
            counts.append(float(len(value)))
    return counts


def _entries_containing(report: dict[str, Any], fragment: str) -> list[tuple[str, Any]]:
    return [
        (key, value) for key, value in _walk(report) if fragment in key.lower()
    ]


#: Keys that identify a per-episode record.
_EPISODE_ID_KEYS = frozenset({"episode_id", "episode", "episode_index"})

#: How a per-episode record can spell "this one succeeded", in priority order:
#: an explicit boolean, a verdict word, then the episode's own reward.
_EPISODE_SUCCESS_BOOL_KEYS = frozenset({"success", "succeeded", "is_success"})
_EPISODE_VERDICT_KEYS = frozenset(
    {"outcome", "result", "verdict", "terminal_reason", "status"}
)
_EPISODE_REWARD_KEYS = frozenset({"reward_sum", "rewards", "reward", "return"})
_SUCCESS_WORDS = ("success", "succeeded", "successful")


def _episode_groups(report: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Every list in the report that looks like a list of per-episode records."""

    groups = []
    for _, value in _walk(report):
        if not isinstance(value, list) or not value:
            continue
        if all(
            isinstance(item, dict)
            and any(str(key).lower() in _EPISODE_ID_KEYS for key in item)
            for item in value
        ):
            groups.append(value)
    return groups


def _episode_succeeded(entry: dict[str, Any]) -> bool:
    lowered = {str(key).lower(): value for key, value in entry.items()}
    for key in _EPISODE_SUCCESS_BOOL_KEYS:
        value = lowered.get(key)
        if isinstance(value, bool):
            return value
    for key in _EPISODE_VERDICT_KEYS:
        value = lowered.get(key)
        if isinstance(value, str):
            text = value.strip().lower()
            return any(text.startswith(word) for word in _SUCCESS_WORDS)
    for key in _EPISODE_REWARD_KEYS:
        value = lowered.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value > 0
    return False


def _success_signals(report: dict[str, Any]) -> list[float]:
    """Every honest way the report can say how many episodes succeeded.

    A total under one of the count keys, or one per-episode verdict each.
    Returned as candidate counts; the caller asserts the fixture's number is
    among them.  Reporting *every* episode as a success (or none) yields no
    candidate equal to 1 and therefore still fails.
    """

    signals = [
        float(value)
        for value in _values_for(
            report,
            (
                "success_count",
                "successes",
                "n_successes",
                "num_successes",
                "n_success",
                "num_success",
                "success_total",
                "successful_episodes",
                "success_episodes",
                "episodes_successful",
            ),
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    signals += [
        float(sum(1 for entry in group if _episode_succeeded(entry)))
        for group in _episode_groups(report)
    ]
    return signals


#: Key names that answer "is this section present?".
_PRESENCE_KEYS = frozenset(
    {
        "present",
        "recorded",
        "available",
        "found",
        "exists",
        "enabled",
        "logged",
        "attached",
    }
)

#: Deliberately generous fallback vocabulary.  Checked only after the
#: structural forms above fail, so a report that answers with a boolean or a
#: null never depends on wording at all.
_ABSENCE_WORDS = (
    "absent",
    "missing",
    "not recorded",
    "not_recorded",
    "not-recorded",
    "no record",
    "unavailable",
    "disabled",
    "none",
    "null",
)


def _absence_verdict(entries: list[tuple[str, Any]]) -> bool:
    """Does this group of report entries say "the evidence was not there"?

    A presence flag, wherever one exists, is AUTHORITATIVE.  That matters: a
    section that claims ``present: true`` while every count beneath it happens
    to be zero is precisely the "empty but present" report this file exists to
    reject, and a purely form-based reading would accept it because the zeros
    look like absence.  The generous reading below is only the fallback for a
    report that carries no flag at all.
    """

    flags: list[bool] = []
    for _key, value in entries:
        if value is None:
            flags.append(True)
        elif isinstance(value, bool):
            flags.append(not value)
        elif isinstance(value, dict):
            for inner_key, inner in value.items():
                if str(inner_key).lower() in _PRESENCE_KEYS:
                    flags.append(_signals_absence(inner))
    if flags:
        return all(flags)
    return any(_signals_absence(value) for _, value in entries)


def _signals_absence(value: Any) -> bool:
    """True when ``value`` says "this evidence was not there"."""

    if value is None or value is False:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        return (not text) or any(word in text for word in _ABSENCE_WORDS)
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    if isinstance(value, dict):
        if not value:
            return True
        for key, inner in value.items():
            if str(key).lower() in _PRESENCE_KEYS and not inner:
                return True
            if str(key).lower() in {"status", "state", "reason", "note"}:
                if isinstance(inner, str) and any(
                    word in inner.lower() for word in _ABSENCE_WORDS
                ):
                    return True
        # A section that carries nothing but empty/zero fields is absence too.
        return all(_signals_absence(inner) for inner in value.values())
    return False


# --------------------------------------------------------------------------- #
# 1. Artifacts and their locations                                              #
# --------------------------------------------------------------------------- #
def test_default_out_dir_is_analysis_under_served(tmp_path):
    served = _build_served_dir(tmp_path / "bc_eval")

    _run_analyzer("--served", served)

    out_dir = served / "analysis"
    assert out_dir.is_dir(), "--out defaults to <served>/analysis"
    _read_report(out_dir)
    _read_markdown(out_dir)


def test_out_flag_redirects_both_artifacts_and_creates_the_directory(tmp_path):
    served = _build_served_dir(tmp_path / "bc_eval")
    out_dir = tmp_path / "elsewhere" / "analysis"

    _run_analyzer("--served", served, "--out", out_dir)

    _read_report(out_dir)
    _read_markdown(out_dir)
    assert not (served / "analysis").exists(), (
        "--out must move the artifacts, not copy them into the served dir too"
    )


# --------------------------------------------------------------------------- #
# 2. The numbers must match the fixture the sink actually wrote.                #
# --------------------------------------------------------------------------- #
def test_report_counts_match_the_recorded_rollout(tmp_path):
    served = _build_served_dir(tmp_path / "bc_eval")
    out_dir = tmp_path / "out"

    _run_analyzer("--served", served, "--out", out_dir)
    report = _read_report(out_dir)

    episodes = _counts_for(
        report,
        (
            "episode_count",
            "episodes",
            "n_episodes",
            "num_episodes",
            "total_episodes",
            "episodes_recorded",
            "recorded_episodes",
        ),
        "the number of episodes",
    )
    assert float(EXPECTED_EPISODES) in episodes, (
        f"expected {EXPECTED_EPISODES} episodes somewhere in the report, "
        f"saw {sorted(set(episodes))}"
    )

    steps = _counts_for(
        report,
        (
            "step_count",
            "steps",
            "n_steps",
            "num_steps",
            "total_steps",
            "transitions",
            "transition_count",
            "total_transitions",
            "transitions_recorded",
            "replay_count",
        ),
        "the number of steps",
    )
    # Either a total is reported, or the per-episode counts add up to one.
    assert float(EXPECTED_STEPS) in steps or sum(steps) == float(EXPECTED_STEPS), (
        f"expected {EXPECTED_STEPS} transitions, saw {sorted(set(steps))}"
    )

    successes = _success_signals(report)
    assert successes, (
        "report.json must say which episodes succeeded -- as a count, or as a "
        "per-episode verdict/reward; found keys "
        f"{sorted({key for key, _ in _walk(report)})}"
    )
    rates = [
        float(value)
        for value in _values_for(
            report, ("success_rate", "success_ratio", "successes_fraction")
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    assert float(EXPECTED_SUCCESSES) in successes or any(
        abs(rate - EXPECTED_SUCCESSES / EXPECTED_EPISODES) < 1e-6 for rate in rates
    ), (
        f"expected {EXPECTED_SUCCESSES} success out of {EXPECTED_EPISODES} "
        f"episodes, saw counts={sorted(set(successes))} rates={sorted(set(rates))}"
    )


def test_markdown_carries_an_episode_table(tmp_path):
    served = _build_served_dir(tmp_path / "bc_eval")
    out_dir = tmp_path / "out"

    _run_analyzer("--served", served, "--out", out_dir)
    markdown = _read_markdown(out_dir)

    assert "episode" in markdown.lower(), "the operator report must name episodes"
    rows = [
        line
        for line in markdown.splitlines()
        if line.strip().startswith("|") and line.count("|") >= 2
    ]
    assert len(rows) >= 3, (
        "expected a markdown table (header + separator + one row per episode), "
        f"found {len(rows)} table lines"
    )
    separators = [
        row
        for row in rows
        if "-" in row and set(row.replace("|", "").replace(" ", "")) <= set("-:")
    ]
    assert separators, "a markdown table needs a header separator row"


# --------------------------------------------------------------------------- #
# 3. Missing evidence is named, not silently zeroed.                            #
# --------------------------------------------------------------------------- #
def test_missing_inference_log_still_produces_a_report_that_says_so(tmp_path):
    without = _build_served_dir(tmp_path / "no_log", with_inference=False)
    assert not (without / "inference.jsonl").exists()
    out_without = tmp_path / "out_without"

    _run_analyzer("--served", without, "--out", out_without)
    report_without = _read_report(out_without)
    _read_markdown(out_without)

    # The rollout itself must still be summarised.
    episodes = _counts_for(
        report_without,
        ("episode_count", "episodes", "n_episodes", "num_episodes", "total_episodes"),
        "the number of episodes",
    )
    assert float(EXPECTED_EPISODES) in episodes

    entries = _entries_containing(report_without, "inference")
    assert entries, (
        "report.json must carry a key about the inference log even when the "
        f"log is absent; keys were {sorted({k for k, _ in _walk(report_without)})}"
    )
    assert _absence_verdict(entries), (
        "the inference section must mark the log as not recorded, not report "
        f"it as an empty-but-present log: {entries}"
    )

    # Differential: the same rollout WITH the log must not report the same
    # thing, otherwise the marker above is a constant rather than a finding.
    with_log = _build_served_dir(tmp_path / "with_log", with_inference=True)
    out_with = tmp_path / "out_with"
    _run_analyzer("--served", with_log, "--out", out_with)
    report_with = _read_report(out_with)

    entries_with = _entries_containing(report_with, "inference")
    assert entries_with
    assert not _absence_verdict(entries_with), (
        "a rollout that DID log inference must not be reported as missing it"
    )


def test_robot_section_is_marked_absent_when_the_flag_is_omitted(tmp_path):
    served = _build_served_dir(tmp_path / "bc_eval")
    out_dir = tmp_path / "out"

    _run_analyzer("--served", served, "--out", out_dir)
    report = _read_report(out_dir)

    entries = _entries_containing(report, "robot")
    assert entries, (
        "report.json must say something about the robot-side recording even "
        f"when --robot was not given; keys were "
        f"{sorted({k for k, _ in _walk(report)})}"
    )
    assert _absence_verdict(entries), (
        f"the robot section must be marked absent, got {entries}"
    )


# --------------------------------------------------------------------------- #
# 4. Damaged evidence must not take the report down with it.                    #
# --------------------------------------------------------------------------- #
def test_malformed_jsonl_lines_are_skipped_without_crashing(tmp_path):
    served = _build_served_dir(tmp_path / "bc_eval")

    # A run killed mid-append leaves exactly this: a half-written last line.
    # Blank lines and outright garbage are the cheap way to prove the reader
    # is line-tolerant rather than json.load()-on-the-whole-file.
    for name in ("actions.jsonl", "inference.jsonl"):
        with open(served / name, "a", encoding="utf-8") as stream:
            stream.write("\n")
            stream.write("this is not json at all\n")
            stream.write('{"ts": "2026-07-31T00:00:00+00:00", "latency_ms":\n')

    out_dir = tmp_path / "out"
    _run_analyzer("--served", served, "--out", out_dir)
    report = _read_report(out_dir)
    _read_markdown(out_dir)

    episodes = _counts_for(
        report,
        ("episode_count", "episodes", "n_episodes", "num_episodes", "total_episodes"),
        "the number of episodes",
    )
    assert float(EXPECTED_EPISODES) in episodes, (
        "the intact episodes must survive a torn jsonl line"
    )


# --------------------------------------------------------------------------- #
# 5. CLI surface                                                                #
# --------------------------------------------------------------------------- #
def test_served_is_required(tmp_path):
    result = _run_analyzer(expect_success=False)

    assert result.returncode != 0, "--served is required; running bare must fail"


def test_a_served_directory_that_does_not_exist_is_rejected(tmp_path):
    missing = tmp_path / "not_a_rollout"

    result = _run_analyzer("--served", missing, expect_success=False)

    assert result.returncode != 0, (
        "a served dir that is not there must fail loudly, not write an "
        "empty report that looks like a rollout with no episodes"
    )
    assert not (missing / "analysis").exists()


@pytest.mark.parametrize("flag", ["--served", "--robot", "--out"])
def test_pinned_flags_are_accepted(tmp_path, flag):
    """The three flag names the runbook and the launcher will use."""

    result = _run_analyzer("--help", expect_success=False)

    # --help exits 0 on argparse; assert on the text of the CLI's own usage,
    # which is the one string a CLI contract legitimately owns.
    assert result.returncode == 0, result.stderr
    assert flag in result.stdout
