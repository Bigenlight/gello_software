#!/usr/bin/env python3
"""Offline analyzer for the opt-in HIL-SERL per-step latency logs.

WHAT THIS IS
------------
``HIL_LATENCY_PROFILE=1`` makes each host of the HIL loop write its own JSONL
sample store (``ur_env/latency_profile.py``):

  (a) the ACTOR on laptop3 -- one line per control-loop iteration, with the
      phases of that iteration (``env_step_ms``, ``transition_build_ms``,
      ``sidecar_encode_ms``, ``step_rpc_ms``, ``iter_interval_ms`` ...), the ids
      needed to correlate it (``transition_id``, ``request_id``, ``run_id``,
      ``episode_id``, ``env_step``) and the values it merely copied
      (``server_inference_ms`` from the Step response, the intervention
      saturation fields out of ``info``).

  (b) the SERVER on junhyeong_ai -- one line per Step RPC, with the phases of
      the handler (``request_decode_ms``, ``classifier_ms``, ``trunk_encode_ms``,
      ``reward_finalize_ms``, ``replay_insert_ms``, ``policy_inference_ms``,
      ``response_build_ms``, ``total_ms``), the same ``transition_id``, and the
      ``concurrent_rpcs`` gauge.  ``BeginEpisode`` gets its own record, tagged
      ``rpc: "begin_episode"``.

This script joins the two offline and reports p50/p90/p99/max per phase, so the
question ``HIL_SERL_REAL_ROBOT_STATUS_AND_NEXT_KO.md`` §8 P1 asks -- *which*
phase owns the 512 ms loop period, and how much of it is contention with the
learner -- has an answer made of samples rather than of two streaming means.

LOCAL INFERENCE MODE (``HIL_POLICY_MODE=local``) ADDS THREE MORE WRITERS
------------------------------------------------------------------------
When the policy runs on laptop3 the same profiler is used by three new
components, each with its own ``role`` and its own file in the same directory
(``serl_ur_infra/HIL_LOCAL_INFERENCE_KO.md``):

  (c) the PROXY -- one line per Step reply it served locally
      (``local_inference_ms``, ``local_finalize_ms``, ``total_ms``,
      ``queue_depth``, ``params_version``, ``params_age_s``).

  (d) the UPLOADER -- one line per transition forwarded to the real server
      (``upload_rpc_ms``, ``backlog_depth``, ``oldest_backlog_s``,
      ``outcome_divergence``).

  (e) the PARAM SYNC -- one line per poll of the learner's exported params
      (``poll_ms``, ``fetch_ms``, ``load_ms``, ``swap_ms``, ``version``,
      ``staleness_s``).

They are optional in exactly the way ``--server`` is: pass them and sections
6-8 appear, leave them out and the report is the remote-mode report it always
was.  🛑 **In local mode the actor's ``step_rpc_ms`` no longer measures the GPU
server** -- it measures the hop to the proxy on 127.0.0.1, and section 3 says
so whenever ``--proxy`` is present.

It trains nothing, touches no hardware, opens no socket, imports no jax and
never writes into a run root unless ``--out`` says to: stdlib + numpy.

USAGE
-----
::

    PY=/home/laptop3/venvs/gello-hil-actor/bin/python

    # actor side only -- everything that does not need the server's file
    $PY serl_ur_infra/scripts/analyze_hil_latency.py \\
        --actor ros2_ur_ws/gello_logs/hil_latency/20260806_101500_actor_12345.jsonl

    # full picture: pull the server's two files back first
    scp junhyeong_ai:'~/hil-serl-data/runs/<run>/logs/latency_server.jsonl' /tmp/
    scp junhyeong_ai:'~/hil-serl-data/runs/<run>/logs/learner.jsonl'        /tmp/
    $PY serl_ur_infra/scripts/analyze_hil_latency.py \\
        --actor   ros2_ur_ws/gello_logs/hil_latency/<file>.jsonl \\
        --server  /tmp/latency_server.jsonl \\
        --learner /tmp/learner.jsonl \\
        --out     /tmp/hil_latency_report.md

    # local inference mode -- the three extra files live next to the actor's
    $PY serl_ur_infra/scripts/analyze_hil_latency.py \\
        --actor     ros2_ur_ws/gello_logs/hil_latency/<utc>_actor_<pid>.jsonl \\
        --proxy     ros2_ur_ws/gello_logs/hil_latency/<utc>_proxy_<pid>.jsonl \\
        --uploader  ros2_ur_ws/gello_logs/hil_latency/<utc>_uploader_<pid>.jsonl \\
        --paramsync ros2_ur_ws/gello_logs/hil_latency/<utc>_paramsync_<pid>.jsonl

THE CLOCK RULE -- READ THIS BEFORE TRUSTING SECTION 2
-----------------------------------------------------
laptop3 and the GPU server do not share a clock and nothing here pretends they
do.  **No timestamp is ever subtracted across hosts.**  Every ``*_ms`` value in
either file is a ``time.perf_counter`` delta taken entirely inside the process
that wrote it, and the one cross-host quantity this report produces is a
subtraction of two such *durations*::

    network_plus_queue_ms = step_rpc_ms (actor)  -  total_ms (server)

which needs no common epoch.  ``t_epoch`` is used for exactly one thing --
overlapping SERVER records against SERVER-host ``learner.jsonl`` events in
section 4 -- and both of those come off the same machine's ``time.time``.

WHY EVERY SECTION DEGRADES INSTEAD OF FAILING
---------------------------------------------
The server side only logs when the learner *process* was started with the env
var, and ``run_hil_server.sh`` reuses a live learner
(``HIL_SERVER_RESULT=reused``).  So the ordinary case is an operator who
enabled profiling on T3, got a fine actor file, and has no server file at all --
that must produce a shorter report, not a stack trace.  Same for a session
killed mid-flush (truncated last line), a field a future wiring commit renames,
and a run with no interventions.  Phases are discovered from the data (*any*
key ending in ``_ms``), never from a hardcoded list, so a phase B or C adds
later shows up here without editing this file.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Contract constants                                                            #
# --------------------------------------------------------------------------- #

#: Nominal control-loop target.  The actor is *measured* at 1.95 Hz; 10 Hz is
#: what the env's own pacing and the action semantics were designed around.
TARGET_HZ = 10.0
TARGET_PERIOD_MS = 1000.0 / TARGET_HZ

#: Loop-budget buckets reported as "fraction of iterations slower than".
BUDGET_THRESHOLDS_MS = (100.0, 200.0, 500.0)

#: Percentiles reported for every phase.  ``numpy`` linear interpolation.
PERCENTILES = (50.0, 90.0, 99.0)

#: Preferred ordering so the console table reads like the loop runs.  Anything
#: not listed still appears -- appended, sorted -- which is how a phase added by
#: a later wiring commit becomes visible without touching this file.
ACTOR_PHASE_ORDER = (
    "iter_interval_ms",
    "obs_prepare_ms",
    "env_step_ms",
    "transition_build_ms",
    "sidecar_encode_ms",
    "step_rpc_ms",
    "server_inference_ms",
)
SERVER_PHASE_ORDER = (
    "total_ms",
    "request_decode_ms",
    "service_step_ms",
    "trunk_encode_ms",
    "classifier_ms",
    "reward_finalize_ms",
    "replay_insert_ms",
    "policy_inference_ms",
    "response_build_ms",
)

#: Local-inference mode (``HIL_POLICY_MODE=local``).  Same rule as above: these
#: are a reading order, not a filter -- a field the proxy adds later still shows
#: up, appended and sorted.
PROXY_PHASE_ORDER = (
    "total_ms",
    "local_inference_ms",
    "local_finalize_ms",
)
UPLOADER_PHASE_ORDER = ("upload_rpc_ms",)
PARAMSYNC_PHASE_ORDER = (
    "poll_ms",
    "fetch_ms",
    "load_ms",
    "swap_ms",
)

#: Gauges (not durations, so they never belong in a phase table) reported with
#: the phase stats machinery anyway, because "how deep did the queue get" is
#: the same kind of question as "how long did the RPC take".
PROXY_GAUGE_ORDER = ("queue_depth", "params_age_s")
UPLOADER_GAUGE_ORDER = ("backlog_depth", "oldest_backlog_s")
PARAMSYNC_GAUGE_ORDER = ("staleness_s",)

#: The proxy's forward queue is unbounded in RAM by design (a dropped
#: transition is worse than a fat queue), so the only defence is noticing.
#: This is the threshold the uploader itself warns at.
BACKLOG_HIGH_WATER = 500

#: Feasibility-study anchors (2026-08-06, measured -- NOT measurements of the
#: file being analyzed).  Printed next to the tables so a reader can tell at a
#: glance whether a session behaved like the study predicted.
ANCHOR_LOCAL_INFERENCE_MS = "2.0 ms GPU / 9.6 ms CPU"
ANCHOR_REMOTE_STEP_MS = 156.1
ANCHOR_PARAMS_BYTES = 12_383_812
ANCHOR_PUBLISH_PERIOD_S = 5.77
ANCHOR_LINK_MB_S = 3.6

#: Keys that end in ``_ms`` but are *not* a duration measured by the writer.
#: They stay in the table (they are useful) with the provenance spelled out,
#: because summing the actor's phases and finding more than ``iter_interval_ms``
#: is otherwise a mystery.
PASSTHROUGH_MS_KEYS = {
    "server_inference_ms": "copied from the Step response (measured on the server)",
}

#: ``iter_interval_ms`` is a *period*, not a phase of the iteration: it spans
#: consecutive loop starts and therefore contains all the other phases.
NON_ADDITIVE_MS_KEYS = {
    "iter_interval_ms": "loop period (start-to-start), not a component phase",
    "total_ms": "handler entry->exit; contains the phases below it",
    "service_step_ms": "contains the session-service phases below it",
}

#: ``LatencyRecord.mark()`` writes offsets from record start, not durations.
#: They are excluded from the phase tables and listed separately.
MARK_SUFFIX = "_at_ms"
MS_SUFFIX = "_ms"

#: The tag agent C puts on the BeginEpisode record.
BEGIN_EPISODE_RPC = "begin_episode"

#: What a missing server file must say, in those words, everywhere.
SERVER_ABSENT = "server data absent"
LEARNER_ABSENT = "learner data absent"
ACTOR_ABSENT = "actor data absent"
PROXY_ABSENT = "proxy data absent"
UPLOADER_ABSENT = "uploader data absent"
PARAMSYNC_ABSENT = "paramsync data absent"

#: Printed when none of the three local-mode files was given at all, which is
#: the ordinary remote-mode session -- not a degradation, just a fact.
LOCAL_MODE_NOT_GIVEN = "local-mode data not given"

#: Section 3 says this, verbatim, whenever --proxy has records: in local mode
#: the actor's Step round trip is a loopback call, and reading it as "the
#: server got faster" would be the single easiest mistake to make with this
#: report.
LOCAL_HOP_PHRASE = "step_rpc_ms measures the LOCAL hop"

#: The divergence callout.  An operator greps for this; it is deliberately
#: ugly.
DIVERGENCE_BANNER = "!!! OUTCOME DIVERGENCE"


# --------------------------------------------------------------------------- #
# Tiny document model: one build pass, two renderers                            #
# --------------------------------------------------------------------------- #


class Note:
    """A free-text line in a section."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = str(text)


class Table:
    """A table of pre-formatted strings."""

    __slots__ = ("headers", "rows", "caption")

    def __init__(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
        caption: str | None = None,
    ) -> None:
        self.headers = [str(h) for h in headers]
        self.rows = [[("" if c is None else str(c)) for c in row] for row in rows]
        self.caption = caption


class Section:
    __slots__ = ("title", "blocks")

    def __init__(self, title: str, blocks: Sequence[Any] | None = None) -> None:
        self.title = str(title)
        self.blocks: list[Any] = list(blocks or [])

    def note(self, text: str) -> "Section":
        self.blocks.append(Note(text))
        return self

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
        caption: str | None = None,
    ) -> "Section":
        self.blocks.append(Table(headers, rows, caption))
        return self


def _render_console(sections: Sequence[Section]) -> str:
    out: list[str] = []
    for section in sections:
        out.append("")
        out.append("=" * 78)
        out.append(section.title)
        out.append("=" * 78)
        previous_was_note = False
        for block in section.blocks:
            if isinstance(block, Note):
                # Consecutive notes stay together: a run of one-line facts
                # reads as a block, not as six paragraphs.
                if not previous_was_note and out and out[-1] != "":
                    out.append("")
                out.append(block.text)
                previous_was_note = True
                continue
            if isinstance(block, Table):
                out.append("")
                if block.caption:
                    out.append(block.caption)
                out.extend(_console_table(block))
                out.append("")
            previous_was_note = False
        while out and out[-1] == "":
            out.pop()
    out.append("")
    return "\n".join(out) + "\n"


def _console_table(table: Table) -> list[str]:
    cols = len(table.headers)
    widths = [len(h) for h in table.headers]
    for row in table.rows:
        for index in range(cols):
            cell = row[index] if index < len(row) else ""
            widths[index] = max(widths[index], len(cell))

    def _line(cells: Sequence[str]) -> str:
        parts = []
        for index in range(cols):
            cell = cells[index] if index < len(cells) else ""
            # First column is a label (left), the rest are numbers (right).
            parts.append(cell.ljust(widths[index]) if index == 0
                         else cell.rjust(widths[index]))
        return "  ".join(parts).rstrip()

    lines = [_line(table.headers), "  ".join("-" * w for w in widths)]
    if not table.rows:
        lines.append("(no rows)")
    for row in table.rows:
        lines.append(_line(row))
    return lines


def _render_markdown(sections: Sequence[Section], title: str) -> str:
    out: list[str] = [f"# {title}", ""]
    for section in sections:
        out.append(f"## {section.title}")
        out.append("")
        for block in section.blocks:
            if isinstance(block, Note):
                # Console notes are space-padded for column alignment; four
                # leading spaces would make markdown render them as a code
                # block, so the padding is a console-only affordance.
                out.append(block.text.lstrip() or "&nbsp;")
                out.append("")
            elif isinstance(block, Table):
                if block.caption:
                    out.append(f"**{block.caption}**")
                    out.append("")
                out.append("| " + " | ".join(block.headers) + " |")
                out.append("|" + "|".join("---" for _ in block.headers) + "|")
                if not block.rows:
                    out.append("| " + " | ".join("-" for _ in block.headers) + " |")
                for row in block.rows:
                    cells = list(row) + [""] * (len(block.headers) - len(row))
                    out.append("| " + " | ".join(cells) + " |")
                out.append("")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Input                                                                         #
# --------------------------------------------------------------------------- #


class JsonlLoad:
    """One JSONL file, read defensively.

    ``truncated_tail`` is the ordinary shape of a session that was SIGKILLed
    with a partial line in the buffer; it is reported, never fatal.  A bad line
    anywhere *else* means something wrote garbage into the middle of the file,
    which is a different (louder) fact, so the two are counted separately.
    """

    __slots__ = (
        "path",
        "records",
        "lines",
        "bad_lines",
        "truncated_tail",
        "present",
        "error",
    )

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.records: list[dict] = []
        self.lines = 0
        self.bad_lines = 0
        self.truncated_tail = False
        self.present = False
        self.error: str | None = None

    @property
    def count(self) -> int:
        return len(self.records)


def load_jsonl(path: str | os.PathLike | None) -> JsonlLoad:
    """Read a JSONL file into a :class:`JsonlLoad`.  Never raises."""

    if path is None:
        return JsonlLoad(None)
    resolved = Path(os.path.expanduser(os.fspath(path)))
    load = JsonlLoad(resolved)
    if not resolved.exists():
        load.error = "file not found"
        return load
    if resolved.is_dir():
        load.error = "path is a directory"
        return load
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        load.error = f"unreadable: {exc}"
        return load
    load.present = True

    raw_lines = text.splitlines()
    last_content_index = -1
    for index, line in enumerate(raw_lines):
        if line.strip():
            last_content_index = index
    for index, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            continue
        load.lines += 1
        try:
            record = json.loads(stripped)
        except (ValueError, TypeError):
            if index == last_content_index:
                load.truncated_tail = True
            else:
                load.bad_lines += 1
            continue
        if isinstance(record, dict):
            load.records.append(record)
        else:
            load.bad_lines += 1
    return load


# --------------------------------------------------------------------------- #
# Numeric helpers                                                               #
# --------------------------------------------------------------------------- #


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return False


def _number(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    return float(value) if _is_number(value) else None


def _truthy(value: Any) -> bool:
    """Loose truth for wire flags: ``True``/``1``/``1.0``/``"1"``/``"true"``."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def summarize(values: Iterable[Any]) -> dict | None:
    """count/mean/p50/p90/p99/max/min over the finite numbers in ``values``.

    ``None`` when nothing usable is there, so every caller has to decide what
    an empty phase renders as instead of silently printing a zero.
    """

    data = [float(v) for v in values if _is_number(v)]
    if not data:
        return None
    array = np.asarray(data, dtype=float)
    p50, p90, p99 = (float(x) for x in np.percentile(array, PERCENTILES))
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "max": float(array.max()),
        "min": float(array.min()),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _pct(numerator: float, denominator: float) -> str:
    if not denominator:
        return "-"
    return f"{100.0 * numerator / denominator:.1f}%"


STAT_HEADERS = ("count", "mean", "p50", "p90", "p99", "max")


def _stat_cells(stats: Mapping[str, Any] | None) -> list[str]:
    if stats is None:
        return ["0", "-", "-", "-", "-", "-"]
    return [
        str(stats["count"]),
        _fmt(stats["mean"]),
        _fmt(stats["p50"]),
        _fmt(stats["p90"]),
        _fmt(stats["p99"]),
        _fmt(stats["max"]),
    ]


# --------------------------------------------------------------------------- #
# Phase discovery                                                               #
# --------------------------------------------------------------------------- #


def phase_keys(records: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> list[str]:
    """Every ``*_ms`` key present, preferred order first then the rest sorted.

    Deliberately data-driven: this analyzer must survive a wiring commit that
    renames or adds a phase, and a hardcoded list would silently drop it.
    """

    found: set[str] = set()
    for record in records:
        for key, value in record.items():
            if not isinstance(key, str):
                continue
            if key.endswith(MARK_SUFFIX) or not key.endswith(MS_SUFFIX):
                continue
            if _is_number(value):
                found.add(key)
    ordered = [key for key in preferred if key in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def mark_keys(records: Sequence[Mapping[str, Any]]) -> list[str]:
    found: set[str] = set()
    for record in records:
        for key, value in record.items():
            if isinstance(key, str) and key.endswith(MARK_SUFFIX) and _is_number(value):
                found.add(key)
    return sorted(found)


def phase_table(
    records: Sequence[Mapping[str, Any]],
    preferred: Sequence[str],
    caption: str,
) -> Table:
    keys = phase_keys(records, preferred)
    rows: list[list[str]] = []
    for key in keys:
        stats = summarize(record.get(key) for record in records)
        label = key
        note = PASSTHROUGH_MS_KEYS.get(key) or NON_ADDITIVE_MS_KEYS.get(key) or ""
        rows.append([label] + _stat_cells(stats) + [note])
    return Table(("phase", *STAT_HEADERS, "note"), rows, caption)


def gauge_table(
    records: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    caption: str,
) -> Table | None:
    """Stats for named non-duration fields; ``None`` when none of them exist.

    A gauge that no record carries produces no row at all rather than a row of
    dashes: "the writer never logged this" and "the writer logged zeros" are
    different facts and must not render the same.
    """

    rows: list[list[str]] = []
    for key in keys:
        stats = summarize(record.get(key) for record in records)
        if stats is None:
            continue
        rows.append([key] + _stat_cells(stats))
    if not rows:
        return None
    return Table(("gauge", *STAT_HEADERS), rows, caption)


# --------------------------------------------------------------------------- #
# Version progression (params blobs)                                            #
# --------------------------------------------------------------------------- #


class VersionProgression:
    """How a monotonic version counter actually moved through a file.

    ``regressions`` is the load-bearing number: the local proxy pins its
    ``params_version`` stamping monotonic within a session (it holds the old
    number while a fetch is in flight), and the param sync refuses to apply a
    blob older than the loaded one.  A regression here therefore means one of
    those two guards did not hold, which a mean would hide completely.
    """

    __slots__ = ("values", "first", "last", "distinct", "advances", "regressions", "missing")

    def __init__(self, values: Sequence[float], missing: int) -> None:
        self.values = list(values)
        self.first = self.values[0] if self.values else None
        self.last = self.values[-1] if self.values else None
        self.distinct = len(set(self.values))
        self.advances = sum(
            1 for a, b in zip(self.values, self.values[1:]) if b > a
        )
        self.regressions = sum(
            1 for a, b in zip(self.values, self.values[1:]) if b < a
        )
        self.missing = int(missing)


def version_progression(
    records: Sequence[Mapping[str, Any]], key: str
) -> VersionProgression:
    values: list[float] = []
    missing = 0
    for record in records:
        value = _number(record, key)
        if value is None:
            missing += 1
            continue
        values.append(value)
    return VersionProgression(values, missing)


def version_table(prog: VersionProgression, caption: str) -> Table:
    return Table(
        ("progression", "value"),
        [
            ["first", _fmt(prog.first, 0)],
            ["last", _fmt(prog.last, 0)],
            ["distinct", str(prog.distinct)],
            ["advances", str(prog.advances)],
            ["regressions", str(prog.regressions)],
            ["records without the field", str(prog.missing)],
        ],
        caption,
    )


# --------------------------------------------------------------------------- #
# Record partitioning                                                           #
# --------------------------------------------------------------------------- #


def split_server_records(
    records: Sequence[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """(step records, begin-episode records, everything else)."""

    steps: list[Mapping[str, Any]] = []
    begins: list[Mapping[str, Any]] = []
    other: list[Mapping[str, Any]] = []
    for record in records:
        rpc = record.get("rpc")
        if isinstance(rpc, str) and rpc.strip().lower() == BEGIN_EPISODE_RPC:
            begins.append(record)
        elif record.get("transition_id") is not None or rpc in (None, "step", "Step"):
            steps.append(record)
        else:
            other.append(record)
    return steps, begins, other


def index_by_transition(
    records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Mapping[str, Any]], int, int]:
    """transition_id -> first record; also (duplicates, records with no id)."""

    index: dict[str, Mapping[str, Any]] = {}
    duplicates = 0
    missing = 0
    for record in records:
        key = record.get("transition_id")
        if not isinstance(key, str) or not key:
            missing += 1
            continue
        if key in index:
            duplicates += 1
            continue
        index[key] = record
    return index, duplicates, missing


# --------------------------------------------------------------------------- #
# Section 0 -- inputs                                                           #
# --------------------------------------------------------------------------- #


def _load_note(label: str, load: JsonlLoad, optional: bool) -> str:
    if load.path is None:
        return f"{label}: not given"
    if load.error is not None:
        prefix = "" if optional else "ERROR "
        return f"{prefix}{label}: {load.path} -- {load.error}"
    bits = [f"{load.count} records", f"{load.lines} lines"]
    if load.bad_lines:
        bits.append(f"{load.bad_lines} unparsable line(s) mid-file")
    if load.truncated_tail:
        bits.append("truncated last line (ignored)")
    return f"{label}: {load.path} -- " + ", ".join(bits)


def _schema_roles(load: JsonlLoad) -> str:
    schemas = sorted({str(r.get("schema")) for r in load.records if "schema" in r})
    roles = sorted({str(r.get("role")) for r in load.records if "role" in r})
    parts = []
    if schemas:
        parts.append("schema=" + ",".join(schemas))
    if roles:
        parts.append("role=" + ",".join(roles))
    return " ".join(parts)


def section_inputs(
    actor: JsonlLoad,
    server: JsonlLoad,
    learner: JsonlLoad,
    proxy: JsonlLoad | None = None,
    uploader: JsonlLoad | None = None,
    paramsync: JsonlLoad | None = None,
) -> Section:
    section = Section("0. INPUTS")
    section.note(_load_note("actor  ", actor, optional=False))
    detail = _schema_roles(actor)
    if detail:
        section.note(f"         {detail}")
    section.note(_load_note("server ", server, optional=True))
    detail = _schema_roles(server)
    if detail:
        section.note(f"         {detail}")
    section.note(_load_note("learner", learner, optional=True))

    # Local-mode inputs are listed only when at least one was given, so a
    # remote-mode report keeps the shape operators already know.
    local = [
        ("proxy    ", proxy),
        ("uploader ", uploader),
        ("paramsync", paramsync),
    ]
    if any(load is not None and load.path is not None for _, load in local):
        for label, load in local:
            section.note(_load_note(label, load if load is not None else JsonlLoad(None), optional=True))

    if actor.records:
        runs = sorted({str(r["run_id"]) for r in actor.records if r.get("run_id")})
        episodes = sorted(
            {r["episode_id"] for r in actor.records if _is_number(r.get("episode_id"))}
        )
        if runs:
            section.note(f"run_id(s): {', '.join(runs)}")
        if episodes:
            section.note(
                f"episodes: {len(episodes)} "
                f"(id {min(episodes):g}..{max(episodes):g})"
            )
    marks = mark_keys(actor.records) + mark_keys(server.records)
    if marks:
        section.note(
            "point-in-time marks present (offsets from record start, not "
            "durations; excluded from the phase tables): " + ", ".join(sorted(set(marks)))
        )
    section.note(
        "Durations are perf_counter deltas taken inside the writing process. "
        "No timestamp is subtracted across hosts anywhere in this report."
    )
    return section


# --------------------------------------------------------------------------- #
# Section 1 -- per-phase tables                                                 #
# --------------------------------------------------------------------------- #


def section_phases(
    actor: JsonlLoad,
    server_steps: Sequence[Mapping[str, Any]],
    server_begins: Sequence[Mapping[str, Any]],
    server: JsonlLoad,
) -> Section:
    section = Section("1. PER-PHASE LATENCY (ms)")
    if actor.records:
        section.blocks.append(
            phase_table(
                actor.records,
                ACTOR_PHASE_ORDER,
                caption=f"ACTOR (laptop3) -- {len(actor.records)} loop iterations",
            )
        )
        section.note(
            "env_step_ms includes the env's own 100 ms pacing, so it has a floor, "
            "not a zero. step_rpc_ms is the blocking Step round trip and sits "
            "OUTSIDE env.step -- that gap is what §8 P1 is about."
        )
    else:
        section.note(f"{ACTOR_ABSENT} -- no actor phase table.")

    if not server.present:
        section.note(
            f"{SERVER_ABSENT} -- no server phase table. "
            "The server only logs when the learner PROCESS was started with "
            "HIL_LATENCY_PROFILE=1; a learner that run_hil_server.sh REUSED "
            "(HIL_SERVER_RESULT=reused) writes nothing however T3 was launched."
        )
        return section

    section.blocks.append(
        phase_table(
            server_steps,
            SERVER_PHASE_ORDER,
            caption=f"SERVER (GPU host) Step -- {_plural(len(server_steps), 'RPC')}",
        )
    )
    if server_begins:
        section.blocks.append(
            phase_table(
                server_begins,
                SERVER_PHASE_ORDER,
                caption=(
                    "SERVER BeginEpisode -- "
                    f"{_plural(len(server_begins), 'RPC')}"
                ),
            )
        )
    gauge = summarize(record.get("concurrent_rpcs") for record in server_steps)
    if gauge is not None:
        section.table(
            ("gauge", *STAT_HEADERS),
            [["concurrent_rpcs"] + _stat_cells(gauge)],
            caption="SERVER queue-pressure proxy (active Step handlers at entry)",
        )
        section.note(
            "concurrent_rpcs > 1 means handlers overlapped (--max-workers 4). "
            "True queue wait is not measurable without a shared clock; this is "
            "the proxy."
        )
    degraded = sum(1 for r in server_steps if _truthy(r.get("classifier_degraded")))
    if degraded:
        section.note(
            f"classifier_degraded on {degraded}/{len(server_steps)} Step records "
            f"({_pct(degraded, len(server_steps))}) -- those transitions were "
            "demoted to unscored, so classifier_ms there is not a scoring cost."
        )
    return section


# --------------------------------------------------------------------------- #
# Section 2 -- cross-host join                                                  #
# --------------------------------------------------------------------------- #


class JoinResult:
    __slots__ = ("pairs", "actor_with_id", "server_with_id", "duplicates", "derived")

    def __init__(self) -> None:
        self.pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        self.actor_with_id = 0
        self.server_with_id = 0
        self.duplicates = 0
        self.derived: list[float] = []


def join_by_transition(
    actor_records: Sequence[Mapping[str, Any]],
    server_steps: Sequence[Mapping[str, Any]],
) -> JoinResult:
    result = JoinResult()
    index, duplicates, _ = index_by_transition(server_steps)
    result.duplicates = duplicates
    result.server_with_id = len(index)
    for record in actor_records:
        key = record.get("transition_id")
        if not isinstance(key, str) or not key:
            continue
        result.actor_with_id += 1
        server = index.get(key)
        if server is None:
            continue
        result.pairs.append((record, server))
        rpc_ms = _number(record, "step_rpc_ms")
        total_ms = _number(server, "total_ms")
        if rpc_ms is not None and total_ms is not None:
            result.derived.append(rpc_ms - total_ms)
    return result


def section_join(
    actor: JsonlLoad, server: JsonlLoad, join: JoinResult, server_steps: Sequence[Mapping[str, Any]]
) -> Section:
    section = Section("2. CROSS-HOST JOIN -- network + queue")
    if not server.present:
        section.note(
            f"{SERVER_ABSENT} -- network+queue cannot be derived. "
            "step_rpc_ms alone still contains the server's whole handler time, "
            "so do NOT read it as wire time."
        )
        return section
    if not actor.records:
        section.note(f"{ACTOR_ABSENT} -- nothing to join against.")
        return section

    section.table(
        ("side", "records", "with transition_id", "matched", "coverage"),
        [
            [
                "actor",
                str(len(actor.records)),
                str(join.actor_with_id),
                str(len(join.pairs)),
                _pct(len(join.pairs), join.actor_with_id),
            ],
            [
                "server (Step)",
                str(len(server_steps)),
                str(join.server_with_id),
                str(len(join.pairs)),
                _pct(len(join.pairs), join.server_with_id),
            ],
        ],
        caption="join coverage (inner join on transition_id)",
    )
    if join.duplicates:
        section.note(
            f"{join.duplicates} duplicate transition_id(s) on the server side; "
            "the first record won each time."
        )
    if not join.pairs:
        section.note(
            "0 matched pairs -- the two files are from different sessions, or "
            "one side never wrote transition_id."
        )
        return section

    stats = summarize(join.derived)
    if stats is None:
        section.note(
            "matched pairs carry no step_rpc_ms/total_ms pair, so "
            "network_plus_queue_ms cannot be derived."
        )
        return section
    section.table(
        ("derived", *STAT_HEADERS, "min"),
        [["network_plus_queue_ms"] + _stat_cells(stats) + [_fmt(stats["min"])]],
        caption="network_plus_queue_ms = step_rpc_ms (actor) - total_ms (server)",
    )
    negatives = sum(1 for value in join.derived if value < 0.0)
    if negatives:
        section.note(
            f"WARNING: {negatives}/{len(join.derived)} pairs are negative. Both "
            "sides are same-host durations, so a negative is not clock skew -- "
            "it means the server's total_ms spans more than the actor's Step "
            "call did (e.g. work after the response was handed to gRPC)."
        )
    rpc = summarize(_number(a, "step_rpc_ms") for a, _ in join.pairs)
    srv = summarize(_number(s, "total_ms") for _, s in join.pairs)
    section.table(
        ("component", *STAT_HEADERS),
        [
            ["step_rpc_ms (actor)"] + _stat_cells(rpc),
            ["total_ms (server)"] + _stat_cells(srv),
            ["network_plus_queue_ms"] + _stat_cells(stats),
        ],
        caption="the same matched pairs, side by side",
    )
    section.note(
        "network+queue lumps together the wire, the SSH tunnel, gRPC "
        "serialization on both ends and any handler-queue wait -- it is a "
        "residual, not a measured hop."
    )
    return section


# --------------------------------------------------------------------------- #
# Section 3 -- loop budget                                                      #
# --------------------------------------------------------------------------- #


def section_loop_budget(actor: JsonlLoad, proxy: JsonlLoad | None = None) -> Section:
    section = Section(f"3. LOOP BUDGET vs {TARGET_HZ:g} Hz ({TARGET_PERIOD_MS:g} ms)")
    local_mode = bool(proxy is not None and proxy.records)
    if not actor.records:
        section.note(f"{ACTOR_ABSENT} -- no loop budget.")
        if local_mode:
            section.note(_local_hop_note(proxy))
        return section
    values = [
        v
        for v in (_number(r, "iter_interval_ms") for r in actor.records)
        if v is not None
    ]
    if not values:
        section.note(
            "iter_interval_ms absent from the actor log -- the loop period was "
            "not recorded, so this section has nothing to report."
        )
        return section
    stats = summarize(values)
    section.table(
        ("metric", *STAT_HEADERS),
        [["iter_interval_ms"] + _stat_cells(stats)],
        caption=f"loop period over {len(values)} intervals",
    )
    section.table(
        ("rate", "Hz"),
        [
            ["from mean", _fmt(1000.0 / stats["mean"] if stats["mean"] else None, 3)],
            ["from p50", _fmt(1000.0 / stats["p50"] if stats["p50"] else None, 3)],
            ["from p99", _fmt(1000.0 / stats["p99"] if stats["p99"] else None, 3)],
        ],
        caption="implied loop rate",
    )
    rows = []
    total = len(values)
    for threshold in BUDGET_THRESHOLDS_MS:
        over = sum(1 for value in values if value > threshold)
        rows.append([f"> {threshold:g} ms", str(over), str(total), _pct(over, total)])
    section.table(
        ("bucket", "iterations", "of", "fraction"),
        rows,
        caption="iterations slower than the budget",
    )
    section.note(
        f"The {TARGET_PERIOD_MS:g} ms target is the nominal control rate the "
        "action semantics were designed around (ACTION_SCALE x 10 Hz). The "
        "production actor was previously measured at 1.95 Hz (512 ms mean, "
        "854 ms max) on the OLD server -- that number is a comparison anchor, "
        "not a claim about this file."
    )
    if local_mode:
        section.note(_local_hop_note(proxy))
    return section


def _local_hop_note(proxy: JsonlLoad | None) -> str:
    """Why section 3 must be read differently once --proxy is present."""

    count = len(proxy.records) if proxy is not None else 0
    return (
        f"LOCAL INFERENCE MODE: --proxy carries {_plural(count, 'record')}, so "
        f"{LOCAL_HOP_PHRASE} -- the actor dialled 127.0.0.1 and the reply came "
        "from the local policy proxy, NOT from the GPU server. Do not compare "
        f"this file's step_rpc_ms with a remote-mode run's and call it a "
        f"speedup of the server: the server's Step (anchor "
        f"{ANCHOR_REMOTE_STEP_MS:g} ms) moved off the control loop into the "
        "uploader, section 7."
    )


# --------------------------------------------------------------------------- #
# Section 4 -- learner contention                                               #
# --------------------------------------------------------------------------- #


def _learner_epoch_s(record: Mapping[str, Any]) -> float | None:
    """Epoch seconds from whatever the learner logger stamped.

    ``ur_env/learner/logging.py`` writes ``time_ns`` (``time.time_ns()``); the
    other spellings are accepted so a future logger change does not silently
    empty this section.
    """

    for key, scale in (
        ("time_ns", 1e-9),
        ("t_epoch", 1.0),
        ("timestamp_ns", 1e-9),
        ("time_s", 1.0),
        ("timestamp", 1.0),
    ):
        value = record.get(key)
        if _is_number(value):
            return float(value) * scale
    return None


def _learner_timing_fields(record: Mapping[str, Any]) -> dict[str, float]:
    """``timing/*`` values, top-level or nested under ``metrics``.

    ``LearnerRuntime`` builds one flat ``metrics`` dict and hands it to
    ``logger.log(..., metrics=metrics)``, so on disk the keys live one level
    down; a flattened writer would put them at the top.  Both are read.
    """

    out: dict[str, float] = {}
    for key, value in record.items():
        if isinstance(key, str) and key.startswith("timing/") and _is_number(value):
            out[key] = float(value)
    nested = record.get("metrics")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            if isinstance(key, str) and key.startswith("timing/") and _is_number(value):
                out.setdefault(key, float(value))
    return out


def _find_nested(record: Any, wanted: str, depth: int = 0) -> list[Any]:
    """Every value stored under key ``wanted`` anywhere in a record."""

    if depth > 6:
        return []
    found: list[Any] = []
    if isinstance(record, Mapping):
        for key, value in record.items():
            if key == wanted:
                found.append(value)
            else:
                found.extend(_find_nested(value, wanted, depth + 1))
    elif isinstance(record, (list, tuple)):
        for item in record:
            found.extend(_find_nested(item, wanted, depth + 1))
    return found


def learner_config_value(records: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for record in records:
        for value in _find_nested(record, key):
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                if value not in values:
                    values.append(value)
    return values


class LearnerWindows:
    """Learner-update intervals on the SERVER clock, queryable by overlap.

    The logger stamps ``time_ns`` when the line is written, which is *after*
    the update, so the window is ``[t - learner_step_ms, t]``.
    """

    __slots__ = ("starts", "ends", "_prefix_max_end", "count")

    def __init__(self, windows: Sequence[tuple[float, float]]) -> None:
        ordered = sorted(windows, key=lambda w: w[0])
        self.starts = [w[0] for w in ordered]
        self.ends = [w[1] for w in ordered]
        self.count = len(ordered)
        running = -math.inf
        self._prefix_max_end: list[float] = []
        for end in self.ends:
            running = max(running, end)
            self._prefix_max_end.append(running)

    def overlaps(self, t0: float, t1: float) -> bool:
        if not self.count:
            return False
        index = bisect.bisect_right(self.starts, t1)
        if index == 0:
            return False
        return self._prefix_max_end[index - 1] >= t0


def build_learner_windows(records: Sequence[Mapping[str, Any]]) -> LearnerWindows:
    windows: list[tuple[float, float]] = []
    for record in records:
        if record.get("event") != "learner_update":
            continue
        end = _learner_epoch_s(record)
        if end is None:
            continue
        timing = _learner_timing_fields(record)
        duration_ms = timing.get("timing/learner_step_ms")
        if duration_ms is None:
            duration_ms = timing.get("timing/full_update_ms")
        if duration_ms is None:
            duration_ms = 0.0
        windows.append((end - max(duration_ms, 0.0) / 1000.0, end))
    return LearnerWindows(windows)


def section_contention(
    server: JsonlLoad,
    server_steps: Sequence[Mapping[str, Any]],
    learner: JsonlLoad,
    join: JoinResult,
) -> Section:
    section = Section("4. LEARNER CONTENTION (server host)")

    if not learner.present:
        section.note(
            f"{LEARNER_ABSENT} -- pass --learner <run_root>/logs/learner.jsonl "
            "to split the RPC stats by learner activity."
        )
    else:
        updates = [r for r in learner.records if r.get("event") == "learner_update"]
        section.note(
            f"learner.jsonl: {len(learner.records)} events, "
            f"{len(updates)} learner_update"
        )
        utd = learner_config_value(learner.records, "utd_ratio")
        section.note(
            "utd_ratio (read from learner.jsonl): "
            + (", ".join(str(v) for v in utd) if utd else
               "not present in this file -- NOT assumed")
        )
        log_period = learner_config_value(learner.records, "log_period")
        if log_period:
            section.note(
                "log_period = "
                + ", ".join(str(v) for v in log_period)
                + " -- learner_update is logged every log_period learner steps, so "
                "the overlap below is a LOWER BOUND on real learner activity."
            )
        else:
            section.note(
                "log_period not in this file: learner_update is SAMPLED "
                "(LearnerRuntime logs every log_period steps), so 'in flight' "
                "below is a lower bound."
            )
        steps = [r.get("learner_step") for r in updates if _is_number(r.get("learner_step"))]
        if steps:
            section.note(
                f"learner_step {min(steps):g}..{max(steps):g}"
            )

        timing_keys = sorted({k for r in updates for k in _learner_timing_fields(r)})
        rows = []
        for key in timing_keys:
            stats = summarize(
                _learner_timing_fields(r).get(key) for r in updates
            )
            rows.append([key] + _stat_cells(stats))
        if rows:
            section.table(
                ("learner phase", *STAT_HEADERS),
                rows,
                caption="the learner's own timing/* (ms), as it already logs them",
            )

    if not server.present:
        section.note(
            f"{SERVER_ABSENT} -- the overlap split needs SERVER-side t_epoch. "
            "The actor's t_epoch is a different host's clock and is never used "
            "for this."
        )
        return section
    if not learner.present:
        return section

    windows = build_learner_windows(learner.records)
    if not windows.count:
        section.note(
            "no usable learner_update windows (missing timestamp or timing) -- "
            "no split."
        )
        return section

    in_flight_ids: set[int] = set()
    classified = 0
    for record in server_steps:
        t0 = _number(record, "t_epoch")
        if t0 is None:
            continue
        classified += 1
        total_ms = _number(record, "total_ms") or 0.0
        if windows.overlaps(t0, t0 + max(total_ms, 0.0) / 1000.0):
            in_flight_ids.add(id(record))
    if not classified:
        section.note(
            "server records carry no t_epoch -- cannot overlap them with the "
            "learner's events."
        )
        return section

    section.note(
        f"{len(in_flight_ids)}/{classified} Step RPCs overlapped a "
        f"learner_update window ({_pct(len(in_flight_ids), classified)}), out of "
        f"{windows.count} windows."
    )

    def _split(pairs, getter):
        during = [getter(x) for x in pairs if id(x[1]) in in_flight_ids]
        idle = [getter(x) for x in pairs if id(x[1]) not in in_flight_ids]
        return summarize(during), summarize(idle)

    server_pairs = [(None, record) for record in server_steps]
    during, idle = _split(server_pairs, lambda x: _number(x[1], "total_ms"))
    rows = [
        ["total_ms / learner update in flight"] + _stat_cells(during),
        ["total_ms / learner idle"] + _stat_cells(idle),
    ]
    if join.pairs:
        during_rpc, idle_rpc = _split(join.pairs, lambda x: _number(x[0], "step_rpc_ms"))
        rows.append(
            ["step_rpc_ms / learner update in flight"] + _stat_cells(during_rpc)
        )
        rows.append(["step_rpc_ms / learner idle"] + _stat_cells(idle_rpc))
    else:
        section.note(
            "step_rpc_ms is not split: no actor<->server matched pairs "
            "(section 2)."
        )
    section.table(
        ("split", *STAT_HEADERS),
        rows,
        caption="RPC cost with and without a learner update in flight",
    )
    section.note(
        "Classification uses ONLY the server's own t_epoch against the server "
        "host's learner.jsonl -- same machine, same clock. The actor's "
        "step_rpc_ms is then carried over through the transition_id join, not "
        "through a timestamp comparison."
    )
    return section


# --------------------------------------------------------------------------- #
# Section 5 -- intervention                                                     #
# --------------------------------------------------------------------------- #


def section_intervention(actor: JsonlLoad) -> Section:
    section = Section("5. INTERVENTION / SATURATION (actor)")
    records = actor.records
    if not records:
        section.note(f"{ACTOR_ABSENT} -- nothing to report.")
        return section

    has_intervened = any("intervened" in r for r in records)
    has_saturation = any("intervention_saturation" in r for r in records)
    has_saturated = any("intervention_saturated" in r for r in records)
    has_ticks = any("intervention_follow_ticks" in r for r in records)
    has_sidecar = any("sidecar_attached" in r for r in records)

    if not any((has_intervened, has_saturation, has_saturated, has_ticks, has_sidecar)):
        section.note(
            "no intervention fields in this actor log -- either the run had no "
            "interventions or the wiring did not copy info[] through."
        )
        return section

    total = len(records)
    intervened = [r for r in records if _truthy(r.get("intervened"))]
    rows = []
    if has_intervened:
        rows.append(
            ["intervened steps", str(len(intervened)), str(total),
             _pct(len(intervened), total)]
        )
    if has_sidecar:
        sidecar = sum(1 for r in records if _truthy(r.get("sidecar_attached")))
        rows.append(["sidecar attached", str(sidecar), str(total), _pct(sidecar, total)])
    if has_saturated:
        saturated = [r for r in records if _truthy(r.get("intervention_saturated"))]
        denominator = len(intervened) if has_intervened and intervened else total
        label = "of intervened" if (has_intervened and intervened) else "of all steps"
        rows.append(
            [f"saturated ({label})", str(len(saturated)), str(denominator),
             _pct(len(saturated), denominator)]
        )
    if rows:
        section.table(("counter", "n", "of", "fraction"), rows, caption="counts")

    stat_rows = []
    if has_saturation:
        stat_rows.append(
            ["intervention_saturation"]
            + _stat_cells(summarize(r.get("intervention_saturation") for r in records))
        )
    if has_ticks:
        stat_rows.append(
            ["intervention_follow_ticks"]
            + _stat_cells(
                summarize(r.get("intervention_follow_ticks") for r in records)
            )
        )
    if stat_rows:
        section.table(
            ("field", *STAT_HEADERS), stat_rows, caption="distributions (as logged)"
        )

    if has_intervened and any(_number(r, "iter_interval_ms") is not None for r in records):
        during = summarize(
            _number(r, "iter_interval_ms") for r in records if _truthy(r.get("intervened"))
        )
        without = summarize(
            _number(r, "iter_interval_ms")
            for r in records
            if not _truthy(r.get("intervened"))
        )
        section.table(
            ("split", *STAT_HEADERS),
            [
                ["iter_interval_ms / intervened"] + _stat_cells(during),
                ["iter_interval_ms / policy"] + _stat_cells(without),
            ],
            caption="loop period with and without a human on the leader",
        )
    if has_sidecar and any(_number(r, "step_rpc_ms") is not None for r in records):
        with_sc = summarize(
            _number(r, "step_rpc_ms")
            for r in records
            if _truthy(r.get("sidecar_attached"))
        )
        without_sc = summarize(
            _number(r, "step_rpc_ms")
            for r in records
            if not _truthy(r.get("sidecar_attached"))
        )
        section.table(
            ("split", *STAT_HEADERS),
            [
                ["step_rpc_ms / sidecar attached"] + _stat_cells(with_sc),
                ["step_rpc_ms / plain"] + _stat_cells(without_sc),
            ],
            caption="the classifier sidecar's cost on the wire",
        )

    section.note(
        "A saturated intervention step means the RECORDED action understates "
        "the arm's real motion (the displacement budget was removed from the "
        "intervention control path). That is the data-integrity half of §8 P1, "
        "and it is copied from info[] here, not recomputed."
    )
    return section


# --------------------------------------------------------------------------- #
# Sections 6-8 -- local inference mode                                          #
# --------------------------------------------------------------------------- #


def _absent_note(load: JsonlLoad, phrase: str, what: str) -> str:
    """One line explaining an empty local-mode input, whichever way it failed."""

    if load.path is None:
        return f"{phrase} -- not given ({what})."
    if load.error is not None:
        return f"{phrase} -- {load.path}: {load.error}."
    return (
        f"{phrase} -- {load.path} held no parsable records. The component only "
        "writes when HIL_LATENCY_PROFILE=1 was in ITS process environment."
    )


def section_local_proxy(proxy: JsonlLoad) -> Section:
    section = Section("6. LOCAL MODE -- POLICY PROXY (laptop3)")
    if not proxy.records:
        section.note(
            _absent_note(
                proxy,
                PROXY_ABSENT,
                "pass --proxy <...>_proxy_<pid>.jsonl for the local action path",
            )
        )
        return section

    records = proxy.records
    section.blocks.append(
        phase_table(
            records,
            PROXY_PHASE_ORDER,
            caption=f"PROXY -- {len(records)} Step replies served locally",
        )
    )
    section.note(
        "local_inference_ms is the policy forward pass on laptop3 "
        f"(feasibility anchor {ANCHOR_LOCAL_INFERENCE_MS}); local_finalize_ms "
        "is the MANUAL finalizer that replaces the server's reward branch. "
        "Both are perf_counter deltas inside the proxy process."
    )

    gauges = gauge_table(
        records,
        PROXY_GAUGE_ORDER,
        caption="proxy gauges (levels at reply time, not durations)",
    )
    if gauges is not None:
        section.blocks.append(gauges)
        backlog = summarize(record.get("queue_depth") for record in records)
        if backlog is not None and backlog["max"] > BACKLOG_HIGH_WATER:
            section.note(
                f"WARNING: queue_depth reached {backlog['max']:.0f}, over the "
                f"{BACKLOG_HIGH_WATER} high-water mark -- the forward queue is "
                "unbounded in RAM by design (dropping a transition is worse), "
                "so a queue this deep is a fact to act on, not a limit that "
                "protected you."
            )
        age = summarize(record.get("params_age_s") for record in records)
        if age is not None:
            section.note(
                "params_age_s is how old the loaded params were when that "
                "reply was served. It has a floor of roughly one transfer: "
                f"the learner publishes every ~{ANCHOR_PUBLISH_PERIOD_S:g} s "
                f"and {ANCHOR_PARAMS_BYTES / 1e6:.1f} MB over a "
                f"~{ANCHOR_LINK_MB_S:g} MB/s link is ~"
                f"{ANCHOR_PARAMS_BYTES / 1e6 / ANCHOR_LINK_MB_S:.1f} s of it. "
                "The actor is ALWAYS acting on a slightly old policy here; "
                "this row says how old."
            )

    prog = version_progression(records, "params_version")
    if prog.values:
        section.blocks.append(
            version_table(prog, caption="params_version as the proxy stamped it")
        )
        if prog.regressions:
            section.note(
                f"WARNING: params_version went BACKWARDS {prog.regressions} "
                "time(s). The proxy is contracted to stamp monotonically "
                "within a session (it holds the old number while a fetch is in "
                "flight), so this is a contract violation, not a slow link."
            )
    else:
        section.note(
            "no params_version field in the proxy log -- cannot tell which "
            "policy served these replies."
        )
    return section


def section_local_uploader(uploader: JsonlLoad) -> Section:
    section = Section("7. LOCAL MODE -- TRANSITION UPLOADER (laptop3 -> server)")
    if not uploader.records:
        section.note(
            _absent_note(
                uploader,
                UPLOADER_ABSENT,
                "pass --uploader <...>_uploader_<pid>.jsonl for the forward path",
            )
        )
        return section

    records = uploader.records
    section.note(
        f"{_plural(len(records), 'upload')} recorded (one record per forwarded "
        "request). This path is OFF the control loop: its latency costs "
        "freshness of the server's replay, never a control-loop millisecond."
    )
    section.blocks.append(
        phase_table(
            records,
            UPLOADER_PHASE_ORDER,
            caption=f"UPLOADER -- {_plural(len(records), 'forwarded request')}",
        )
    )
    gauges = gauge_table(
        records,
        UPLOADER_GAUGE_ORDER,
        caption="backlog over time (level at the moment each upload was recorded)",
    )
    if gauges is not None:
        section.blocks.append(gauges)
        depth = summarize(record.get("backlog_depth") for record in records)
        oldest = summarize(record.get("oldest_backlog_s") for record in records)
        if depth is not None:
            section.note(
                f"backlog_depth max {depth['max']:.0f}"
                + (
                    f", oldest entry {oldest['max']:.3f} s"
                    if oldest is not None
                    else ""
                )
                + ". The server still ingests every forwarded transition at "
                f"its usual cost (anchor {ANCHOR_REMOTE_STEP_MS:g} ms per "
                "Step), so during an episode the backlog GROWS and it drains "
                "on the WAIT screens. A backlog that never returns to ~0 "
                "between episodes is the signal that this design's assumption "
                "broke."
            )
            if depth["max"] > BACKLOG_HIGH_WATER:
                section.note(
                    f"WARNING: backlog_depth exceeded the {BACKLOG_HIGH_WATER} "
                    "high-water mark. Nothing was dropped (the queue is "
                    "unbounded on purpose) -- but RAM on laptop3 is the only "
                    "thing bounding it."
                )

    divergent = [r for r in records if _truthy(r.get("outcome_divergence"))]
    has_field = any("outcome_divergence" in r for r in records)
    if not has_field:
        section.note(
            "no outcome_divergence field in this log -- the parity alarm "
            "between the proxy's LOCAL MANUAL verdict and the server's own "
            "outcome was not recorded, so parity is UNKNOWN for this session, "
            "not proven."
        )
        return section
    section.table(
        ("counter", "n", "of", "fraction"),
        [
            [
                "outcome_divergence",
                str(len(divergent)),
                str(len(records)),
                _pct(len(divergent), len(records)),
            ]
        ],
        caption="local verdict vs the server's returned outcome",
    )
    if divergent:
        section.note(
            f"{DIVERGENCE_BANNER} on {len(divergent)}/{len(records)} forwarded "
            f"transitions ({_pct(len(divergent), len(records))})"
        )
        section.note(
            "The proxy replicates the server's MANUAL finalizer; a divergence "
            "means the two disagreed about done/truncated/masks/reward for a "
            "transition the ACTOR already acted on. The server's copy is what "
            "the learner trained on, so the data is not corrupt -- but the "
            "actor saw a different episode than the learner did, and that gap "
            "is exactly what this alarm exists to catch. Investigate before "
            "trusting the session."
        )
    else:
        section.note(
            f"outcome_divergence: 0/{len(records)} -- the local MANUAL verdict "
            "matched the server on every forwarded transition in this file."
        )
    return section


def section_local_paramsync(paramsync: JsonlLoad) -> Section:
    section = Section("8. LOCAL MODE -- PARAM SYNC (server -> laptop3)")
    if not paramsync.records:
        section.note(
            _absent_note(
                paramsync,
                PARAMSYNC_ABSENT,
                "pass --paramsync <...>_paramsync_<pid>.jsonl for the params path",
            )
        )
        return section

    records = paramsync.records
    section.blocks.append(
        phase_table(
            records,
            PARAMSYNC_PHASE_ORDER,
            caption=f"PARAM SYNC -- {_plural(len(records), 'poll')}",
        )
    )
    fetches = sum(1 for r in records if _number(r, "fetch_ms") is not None)
    swaps = sum(1 for r in records if _number(r, "swap_ms") is not None)
    section.note(
        f"{_plural(len(records), 'poll')}, {fetches} carried a fetch and "
        f"{swaps} a swap: poll_ms is paid every cycle, fetch/load/swap only "
        "when the learner published a newer version. All three are background "
        "work -- none of them is in the control loop."
    )
    gauges = gauge_table(
        records,
        PARAMSYNC_GAUGE_ORDER,
        caption="staleness of the applied params (s)",
    )
    if gauges is not None:
        section.blocks.append(gauges)
        section.note(
            "staleness_s = now - the LATEST.json timestamp of the version "
            "actually applied. It cannot be zero: the blob has to cross the "
            "link first. Compare it with the learner's publish period "
            f"(anchor ~{ANCHOR_PUBLISH_PERIOD_S:g} s) -- staleness of about "
            "one period is the design working, staleness that GROWS means the "
            "link cannot keep up with the learner and the local policy is "
            "falling behind."
        )

    prog = version_progression(records, "version")
    if prog.values:
        section.blocks.append(
            version_table(prog, caption="params version seen by the sync client")
        )
        if prog.regressions:
            section.note(
                f"WARNING: version went BACKWARDS {prog.regressions} time(s). "
                "The sync client is contracted to skip-to-newest and refuse an "
                "older blob, so this means either a learner restart reset the "
                "counter (check the run root) or the refusal did not hold."
            )
    else:
        section.note("no version field in the paramsync log -- progression unknown.")
    return section


def section_local_mode_absent(
    proxy: JsonlLoad, uploader: JsonlLoad, paramsync: JsonlLoad
) -> Section:
    """Placeholder section for the ordinary remote-mode report.

    Rendering three empty sections on every remote run would train operators
    to scroll past the end of the report, so a remote run gets one line
    instead -- but it does get that line, because "this report says nothing
    about local mode" is itself worth stating on a page someone will paste
    into a status document.
    """

    section = Section("6. LOCAL INFERENCE MODE")
    section.note(
        f"{LOCAL_MODE_NOT_GIVEN} -- no --proxy / --uploader / --paramsync. "
        "This is the remote-mode report: the actor's step_rpc_ms is the round "
        "trip to the GPU server. Pass those three files (written when "
        "HIL_POLICY_MODE=local and HIL_LATENCY_PROFILE=1) to get sections "
        "6-8: local inference, forward backlog and params staleness."
    )
    return section


def local_mode_sections(
    proxy: JsonlLoad, uploader: JsonlLoad, paramsync: JsonlLoad
) -> list[Section]:
    if all(load.path is None for load in (proxy, uploader, paramsync)):
        return [section_local_mode_absent(proxy, uploader, paramsync)]
    return [
        section_local_proxy(proxy),
        section_local_uploader(uploader),
        section_local_paramsync(paramsync),
    ]


# --------------------------------------------------------------------------- #
# Assembly                                                                      #
# --------------------------------------------------------------------------- #


def build_sections(
    actor: JsonlLoad,
    server: JsonlLoad,
    learner: JsonlLoad,
    proxy: JsonlLoad | None = None,
    uploader: JsonlLoad | None = None,
    paramsync: JsonlLoad | None = None,
) -> list[Section]:
    proxy = proxy if proxy is not None else JsonlLoad(None)
    uploader = uploader if uploader is not None else JsonlLoad(None)
    paramsync = paramsync if paramsync is not None else JsonlLoad(None)
    server_steps, server_begins, _other = split_server_records(server.records)
    join = join_by_transition(actor.records, server_steps)
    sections = [
        section_inputs(actor, server, learner, proxy, uploader, paramsync),
        section_phases(actor, server_steps, server_begins, server),
        section_join(actor, server, join, server_steps),
        section_loop_budget(actor, proxy),
        section_contention(server, server_steps, learner, join),
        section_intervention(actor),
    ]
    sections.extend(local_mode_sections(proxy, uploader, paramsync))
    return sections


def build_report(
    actor_path: str | os.PathLike,
    server_path: str | os.PathLike | None = None,
    learner_path: str | os.PathLike | None = None,
    proxy_path: str | os.PathLike | None = None,
    uploader_path: str | os.PathLike | None = None,
    paramsync_path: str | os.PathLike | None = None,
) -> tuple[
    list[Section], JsonlLoad, JsonlLoad, JsonlLoad, JsonlLoad, JsonlLoad, JsonlLoad
]:
    actor = load_jsonl(actor_path)
    server = load_jsonl(server_path)
    learner = load_jsonl(learner_path)
    proxy = load_jsonl(proxy_path)
    uploader = load_jsonl(uploader_path)
    paramsync = load_jsonl(paramsync_path)
    sections = build_sections(actor, server, learner, proxy, uploader, paramsync)
    return sections, actor, server, learner, proxy, uploader, paramsync


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_hil_latency.py",
        description=(
            "Join the actor's and the server's HIL_LATENCY_PROFILE JSONL logs "
            "and report per-phase p50/p90/p99/max, the derived network+queue "
            "time, the loop budget against 10 Hz, and learner contention. In "
            "local inference mode (HIL_POLICY_MODE=local) the proxy, uploader "
            "and param-sync files add sections 6-8."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  analyze_hil_latency.py --actor gello_logs/hil_latency/<f>.jsonl\n"
            "  analyze_hil_latency.py --actor <f>.jsonl --server latency_server.jsonl \\\n"
            "      --learner learner.jsonl --out /tmp/hil_latency_report.md\n"
            "  analyze_hil_latency.py --actor <f>.jsonl --proxy <p>.jsonl \\\n"
            "      --uploader <u>.jsonl --paramsync <s>.jsonl\n"
        ),
    )
    parser.add_argument(
        "--actor",
        required=True,
        help=(
            "actor-side jsonl written on laptop3 "
            "(ros2_ur_ws/gello_logs/hil_latency/*.jsonl)"
        ),
    )
    parser.add_argument(
        "--server",
        default=None,
        help=(
            "server-side jsonl (<run_root>/logs/latency_server.jsonl). Optional: "
            "a REUSED learner never wrote one"
        ),
    )
    parser.add_argument(
        "--learner",
        default=None,
        help="the learner's own <run_root>/logs/learner.jsonl (for contention)",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help=(
            "LOCAL MODE: the policy proxy's jsonl "
            "(gello_logs/hil_latency/*_proxy_*.jsonl) -- local inference, "
            "queue depth, params age/version. Section 6"
        ),
    )
    parser.add_argument(
        "--uploader",
        default=None,
        help=(
            "LOCAL MODE: the forwarding uploader's jsonl (*_uploader_*.jsonl) "
            "-- upload RPC, backlog, outcome divergence. Section 7"
        ),
    )
    parser.add_argument(
        "--paramsync",
        default=None,
        help=(
            "LOCAL MODE: the param sync client's jsonl (*_paramsync_*.jsonl) "
            "-- poll/fetch/load/swap and staleness. Section 8"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="also write the report as markdown to this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    actor_path = Path(os.path.expanduser(args.actor))
    if not actor_path.exists():
        print(f"error: --actor file not found: {actor_path}", file=sys.stderr)
        return 2
    if actor_path.is_dir():
        print(f"error: --actor is a directory, not a file: {actor_path}", file=sys.stderr)
        return 2

    sections, actor, server, learner, proxy, uploader, paramsync = build_report(
        actor_path,
        args.server,
        args.learner,
        args.proxy,
        args.uploader,
        args.paramsync,
    )

    for label, load in (
        ("--server", server),
        ("--learner", learner),
        ("--proxy", proxy),
        ("--uploader", uploader),
        ("--paramsync", paramsync),
    ):
        if load.path is not None and load.error is not None:
            print(
                f"warning: {label} {load.path}: {load.error} "
                "-- that section degrades",
                file=sys.stderr,
            )

    sys.stdout.write(_render_console(sections))

    if args.out:
        out_path = Path(os.path.expanduser(args.out))
        try:
            if out_path.parent and str(out_path.parent):
                out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                _render_markdown(sections, "HIL-SERL latency report"),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"error: cannot write --out {out_path}: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {out_path}", file=sys.stderr)

    if not actor.records:
        print(
            f"note: {ACTOR_ABSENT} -- {actor_path} held no parsable records",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
