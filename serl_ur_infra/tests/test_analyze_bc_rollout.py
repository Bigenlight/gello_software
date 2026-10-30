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


# --------------------------------------------------------------------------- #
# 6. Step timing -- the OPT-IN latency breakdown                                #
# --------------------------------------------------------------------------- #
# A BC evaluation can be run with HIL_STEP_TIMING=1, which leaves a second pair
# of logs: ``timing.jsonl`` in the served dir (written by the SERVER, one row per
# Step RPC and one per BeginEpisode) and a separate actor-side jsonl on laptop3
# (one row per loop iteration).  Neither exists by default.
#
# The number that needs BOTH files is ``wire_ms = actor.rpc_ms -
# server.handler_ms``: everything the actor measured includes the server's own
# handler span, so subtracting it leaves the tunnel and the framing.  That
# subtraction is the only arithmetic the analyzer performs across hosts, and the
# fixture below is built so its answer is known by construction rather than
# recomputed by the test.
#
#     transition          server handler_ms    actor rpc_ms    wire_ms
#     <RUN_ID>:0:0                    10.0            15.0        5.0
#     <RUN_ID>:0:1                    20.0            26.0        6.0
#     <RUN_ID>:0:2 (terminal)         30.0            37.0        7.0
#                                                     mean        6.0
#
# The third server row is terminal and therefore has ``infer_ms``/``sink_ms``
# null (a terminal Step requests no action and runs no sink), which is what
# makes the per-phase sample counts differ from the row count -- the property
# a "count == lines" implementation would get wrong.

#: Server rows: handler_ms, infer_ms, sink_ms, terminal.
_TIMING_STEPS = (
    (10.0, 4.0, 1.0, False),
    (20.0, 5.0, 2.0, False),
    (30.0, None, None, True),
)
#: Actor rpc_ms per step, in the same order.
_ACTOR_RPC_MS = (15.0, 26.0, 37.0)
#: The known answers.
_WIRE_MS = tuple(
    rpc - step[0] for rpc, step in zip(_ACTOR_RPC_MS, _TIMING_STEPS)
)
_WIRE_MEAN = sum(_WIRE_MS) / len(_WIRE_MS)
#: BeginEpisode is timed too, but it is not a step and must not pollute the
#: step phases; its handler_ms is deliberately below every step's.
_BEGIN_EPISODE_MS = 12.0

_TIMING_TRANSITION_IDS = tuple(
    f"{RUN_ID}:0:{index}" for index in range(len(_TIMING_STEPS))
)

#: The one stable string the "we have no step timing" line has to contain: the
#: environment variable an operator would set to get it next time.  Asserting on
#: the switch rather than on the prose keeps the wording free to change.
NOT_RECORDED_SUBSTRING = "HIL_STEP_TIMING"


def _server_timing_lines() -> list[str]:
    """``timing.jsonl`` in the frozen server schema (3 steps + 1 BeginEpisode)."""

    rows: list[dict[str, Any]] = []
    for index, (handler, infer, sink, terminal) in enumerate(_TIMING_STEPS):
        rows.append(
            {
                "ts": 1_800_000_000.0 + index,
                "kind": "step",
                "run_id": RUN_ID,
                "episode_id": 0,
                "step_id": index,
                "env_step": index,
                "transition_id": _TIMING_TRANSITION_IDS[index],
                "handler_ms": handler,
                "infer_ms": infer,
                "sink_ms": sink,
                "overhead_ms": handler - (infer or 0.0) - (sink or 0.0),
                "terminal": terminal,
                "deduplicated": False,
                "error": None,
            }
        )
    rows.append(
        {
            "ts": 1_799_999_999.0,
            "kind": "begin_episode",
            "run_id": RUN_ID,
            "episode_id": None,
            "step_id": None,
            "env_step": None,
            "transition_id": None,
            "handler_ms": _BEGIN_EPISODE_MS,
            "infer_ms": None,
            "sink_ms": None,
            "overhead_ms": _BEGIN_EPISODE_MS,
            "terminal": None,
            "deduplicated": None,
            "error": None,
        }
    )
    return [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]


def _actor_timing_lines() -> list[str]:
    """The laptop3 side, in the frozen actor schema, with matching ids."""

    rows = []
    for index, rpc in enumerate(_ACTOR_RPC_MS):
        terminal = _TIMING_STEPS[index][3]
        rows.append(
            {
                "ts": 1_800_000_000.0 + index,
                "run_id": RUN_ID,
                "episode_id": 0,
                "step_id": index,
                "env_step": index,
                "transition_id": _TIMING_TRANSITION_IDS[index],
                "env_step_ms": 100.0 + index,
                "build_ms": 1.0 + index,
                "sidecar_ms": None,
                "rpc_ms": rpc,
                "round_trip_ms": rpc - 1.0,
                "server_inference_ms": _TIMING_STEPS[index][1],
                "loop_ms": 200.0 + index,
                "post_prev_ms": None if index == 0 else 50.0,
                "attached_sidecar": False,
                "intervened": False,
                "terminal": terminal,
            }
        )
    return [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]


def _write_step_timing(served: Path) -> None:
    (served / "timing.jsonl").write_text(
        "\n".join(_server_timing_lines()) + "\n", encoding="utf-8"
    )


def _write_actor_timing(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_actor_timing_lines()) + "\n", encoding="utf-8")
    return path


def _latency(report: dict[str, Any]) -> dict[str, Any]:
    latency = report.get("latency")
    assert isinstance(latency, dict), (
        "report.json must carry a `latency` section whenever step timing is "
        f"part of the CLI; found top-level keys {sorted(report)}"
    )
    for side in ("server", "actor", "joined"):
        assert side in latency, f"latency must name the `{side}` side"
    return latency


def _stats(node: Any, name: str) -> dict[str, Any]:
    """The count/mean/p50/p95/max block for one phase, wherever it is nested.

    Tolerant on purpose: the assertions below are about the NUMBERS, so the
    analyzer stays free to group its phases under ``phases``, inline them, or
    rename the grouping.
    """

    for key, value in _walk(node):
        if key == name and isinstance(value, dict) and "count" in value:
            return value
    raise AssertionError(
        f"no per-phase stats found for {name!r}; keys present: "
        f"{sorted({key for key, _ in _walk(node)})}"
    )


def _looks_absent(node: Any) -> bool:
    """True when this side of the latency report is explicitly marked missing."""

    if node is None or node == {}:
        return True
    if not isinstance(node, dict):
        return False
    for key, value in _walk(node):
        if key.lower() in {"present", "recorded", "found", "available"}:
            if value is False:
                return True
    return False


def test_step_timing_logs_produce_a_joined_latency_breakdown(tmp_path):
    """Both sides present: the report must split rpc time from handler time."""

    served = _build_served_dir(tmp_path / "bc_eval")
    _write_step_timing(served)
    actor_timing = _write_actor_timing(tmp_path / "laptop3" / "step_timing.jsonl")
    out_dir = tmp_path / "out"

    _run_analyzer(
        "--served", served, "--actor-timing", actor_timing, "--out", out_dir
    )
    report = _read_report(out_dir)
    latency = _latency(report)

    joined = latency["joined"]
    wire = _stats(joined, "wire_ms")
    assert wire["count"] == len(_ACTOR_RPC_MS), (
        "all three actor rows share a transition_id with a server step row, so "
        "all three must join"
    )
    assert wire["mean"] == pytest.approx(_WIRE_MEAN), (
        "wire_ms is actor rpc_ms minus server handler_ms, averaged: "
        f"expected {_WIRE_MEAN}, fixture wires are {list(_WIRE_MS)}"
    )
    assert wire["max"] == pytest.approx(max(_WIRE_MS))


def test_step_timing_phase_counts_follow_the_samples_not_the_line_count(tmp_path):
    """A terminal step has no inference and no sink; the counts must show it."""

    served = _build_served_dir(tmp_path / "bc_eval")
    _write_step_timing(served)
    actor_timing = _write_actor_timing(tmp_path / "laptop3" / "step_timing.jsonl")
    out_dir = tmp_path / "out"

    _run_analyzer(
        "--served", served, "--actor-timing", actor_timing, "--out", out_dir
    )
    latency = _latency(_read_report(out_dir))

    server = latency["server"]
    handler = _stats(server, "handler_ms")
    # INTERPRETATION: the frozen spec does not say whether the BeginEpisode row
    # joins the step phases, so both readings are accepted -- but the maximum
    # pins that the step rows are all there either way.
    assert handler["count"] in (len(_TIMING_STEPS), len(_TIMING_STEPS) + 1), (
        f"expected 3 step rows (or 4 including BeginEpisode), got {handler['count']}"
    )
    assert handler["max"] == pytest.approx(30.0)

    infer = _stats(server, "infer_ms")
    assert infer["count"] == 2, "only the two non-terminal rows ran the policy"
    assert infer["mean"] == pytest.approx(4.5)

    sink = _stats(server, "sink_ms")
    assert sink["count"] == 2, "only the two non-terminal rows reached the sink"
    assert sink["mean"] == pytest.approx(1.5)

    actor = latency["actor"]
    rpc = _stats(actor, "rpc_ms")
    assert rpc["count"] == len(_ACTOR_RPC_MS)
    assert rpc["mean"] == pytest.approx(sum(_ACTOR_RPC_MS) / len(_ACTOR_RPC_MS))


def test_the_markdown_carries_the_latency_subsection(tmp_path):
    """Under section 2 -- the operator reads latency next to the inference log."""

    served = _build_served_dir(tmp_path / "bc_eval")
    _write_step_timing(served)
    actor_timing = _write_actor_timing(tmp_path / "laptop3" / "step_timing.jsonl")
    out_dir = tmp_path / "out"

    _run_analyzer(
        "--served", served, "--actor-timing", actor_timing, "--out", out_dir
    )
    markdown = _read_markdown(out_dir)

    assert "Latency breakdown (step timing)" in markdown
    heading = markdown.index("Latency breakdown (step timing)")
    section_two = markdown.index("## 2. Inference log")
    assert heading > section_two, (
        "the latency breakdown is a subsection of the inference-log section"
    )
    section_three = markdown.find("## 3.")
    if section_three != -1:
        assert heading < section_three, (
            "the latency breakdown must not drift into the robot-side section"
        )


def test_a_rollout_without_step_timing_says_so_and_still_reports(tmp_path):
    """Default OFF: neither log exists, and the analyzer must not crash on it."""

    served = _build_served_dir(tmp_path / "bc_eval")
    assert not (served / "timing.jsonl").exists(), "fixture sanity check"
    out_dir = tmp_path / "out"

    result = _run_analyzer("--served", served, "--out", out_dir)
    assert result.returncode == 0

    latency = _latency(_read_report(out_dir))
    assert _looks_absent(latency["server"]), (
        "a missing timing.jsonl must be marked absent, not rendered as zeros: "
        f"{latency['server']}"
    )
    assert _looks_absent(latency["actor"]), (
        f"no --actor-timing was given: {latency['actor']}"
    )

    markdown = _read_markdown(out_dir)
    assert NOT_RECORDED_SUBSTRING in markdown, (
        "the operator has to be told HOW to record this next time, not just "
        "that it is missing"
    )


def test_an_actor_timing_path_that_does_not_exist_is_survivable(tmp_path):
    """A mistyped path must degrade to "absent", never take the report down."""

    served = _build_served_dir(tmp_path / "bc_eval")
    _write_step_timing(served)
    out_dir = tmp_path / "out"

    result = _run_analyzer(
        "--served",
        served,
        "--actor-timing",
        tmp_path / "nope" / "missing.jsonl",
        "--out",
        out_dir,
    )
    assert result.returncode == 0

    latency = _latency(_read_report(out_dir))
    assert _looks_absent(latency["actor"])
    # ...and the half that IS there is still summarised.
    assert _stats(latency["server"], "handler_ms")["count"] >= len(_TIMING_STEPS)


def test_torn_step_timing_lines_are_skipped_without_crashing(tmp_path):
    """These logs are appended to live; a Ctrl-C leaves a half-written line."""

    served = _build_served_dir(tmp_path / "bc_eval")
    _write_step_timing(served)
    actor_timing = _write_actor_timing(tmp_path / "laptop3" / "step_timing.jsonl")
    for path in (served / "timing.jsonl", actor_timing):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write("\n")
            stream.write("not json at all\n")
            stream.write('{"kind": "step", "handler_ms":\n')
    out_dir = tmp_path / "out"

    _run_analyzer(
        "--served", served, "--actor-timing", actor_timing, "--out", out_dir
    )
    latency = _latency(_read_report(out_dir))

    assert _stats(latency["joined"], "wire_ms")["mean"] == pytest.approx(_WIRE_MEAN), (
        "the intact rows must still join and average to the same number"
    )


def test_the_actor_timing_flag_is_pinned(tmp_path):
    """The runbook and the launcher both type this name."""

    result = _run_analyzer("--help", expect_success=False)

    assert result.returncode == 0, result.stderr
    assert "--actor-timing" in result.stdout
