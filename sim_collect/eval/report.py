#!/usr/bin/env python3
"""Aggregate one or many ``run_eval`` output dirs into a markdown + CSV report.

WHAT THIS IS
------------
``sim_collect/eval/run_eval.py`` leaves, per run directory (DESIGN.md §2.3)::

    <run>/episodes.jsonl   one JSON object per episode: seed, outcome, n_steps,
                           t_success_s, detail, policy meta, wall time
    <run>/summary.json     SR, CI, per-outcome counts, config sha, git commit,
                           checkpoint id
    <run>/ep_<seed>.h5     per-episode state (not read here)
    <run>/ep_<seed>_cam1.mp4 (only with --video; not read here)

This script re-derives the aggregate numbers from ``episodes.jsonl`` (the raw
evidence) rather than trusting ``summary.json`` (a convenience copy), uses
``summary.json`` only for identity fields, and puts several runs side by side so
checkpoint A and checkpoint B can be compared **on the same seeds**.

    # one run to stdout
    .venv/bin/python -m sim_collect.eval.report sim_collect/eval/runs/act_40k

    # two checkpoints, paired comparison, files on disk
    .venv/bin/python -m sim_collect.eval.report \\
        sim_collect/eval/runs/act_40k sim_collect/eval/runs/act_80k \\
        --label 40k --label 80k --compare --out sim_collect/eval/runs/compare_act

WHAT THE NUMBERS MEAN -- AND DO NOT
-----------------------------------
* **SR** = successes / episodes whose outcome could be determined.  Episodes
  whose row carries no usable outcome are counted separately as ``unknown`` and
  are kept OUT of the denominator, so a malformed line cannot silently deflate
  the success rate.  Their count is printed in the report.
* **Wilson 95 % CI** is a *binomial* interval.  At the default 20 seeds it is
  wide: 10/20 gives roughly [0.30, 0.70].  Two checkpoints whose intervals
  overlap are not distinguished by this run -- say so, do not pick a winner.
* **Comparisons are paired** (the same seed list, the same scene layouts), so
  the honest test is McNemar's exact test on the discordant seeds, not two
  independent proportions.  It is reported as ``p`` and it, too, is weak at 20
  seeds: it needs ~6 discordant pairs all pointing the same way to reach 0.05.
* Diffusion and FM sample stochastically: re-running the same checkpoint on the
  same seeds does not have to reproduce the same episodes.  The scripted/replay/
  zero policies do.

Nothing here touches MuJoCo, torch or a socket: stdlib only, so it runs in any
interpreter (``.venv`` is the documented one).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

# --- filenames, mirrored from DESIGN §2.3 (not guessed) ----------------------
EPISODES_JSONL = "episodes.jsonl"
SUMMARY_JSON = "summary.json"

#: z for a two-sided 95 % interval.
Z95 = 1.959963984540054

#: the outcome string that counts as a success.  Everything else is a failure.
SUCCESS = "success"
UNKNOWN = "unknown"

#: fallback control rate when an episode row has no t_success_s and the run
#: summary does not state one (run_eval steps the world at 30 Hz, DESIGN §2).
DEFAULT_CONTROL_HZ = 30.0


# =============================================================================
# statistics
# =============================================================================
def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because the interesting cases here are
    exactly the ones where it breaks: 0/20 and 20/20, where Wald gives a
    zero-width interval and Wilson does not.  ``n == 0`` yields the vacuous
    ``(0.0, 1.0)``.
    """

    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    ``b`` = seeds where A succeeded and B failed, ``c`` = the other way round.
    Concordant seeds carry no information about the difference and are excluded
    by construction.  With no discordant pairs the p-value is 1.0.
    """

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


# =============================================================================
# loading
# =============================================================================
def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    f = _as_float(value)
    if f is None:
        return None
    return int(f) if float(f).is_integer() else None


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    """First key present with a non-None value (spelling drift is expected)."""

    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def normalize_episode(row: Mapping[str, Any]) -> dict[str, Any]:
    """One ``episodes.jsonl`` object -> the fields this report needs.

    Every field is optional.  ``outcome`` falls back to a boolean ``success``
    field and then to ``unknown``; ``success`` prefers an explicit boolean and
    only then the literal outcome string.
    """

    outcome_raw = _first(row, "outcome", "result", "status")
    success_raw = _first(row, "success", "succeeded")

    outcome = str(outcome_raw).strip() if outcome_raw is not None else ""
    if not outcome:
        if isinstance(success_raw, bool):
            outcome = SUCCESS if success_raw else "failure"
        else:
            outcome = UNKNOWN

    if isinstance(success_raw, bool):
        success: bool | None = success_raw
    elif outcome == UNKNOWN:
        success = None
    else:
        success = outcome == SUCCESS

    # Same precedence as run_eval's own summary.md: the specific cause first,
    # the task detail only when there is no cause.
    detail = _first(row, "fault", "failure", "detail", "failure_reason", "reason", "note")
    if isinstance(detail, (dict, list)):
        detail = json.dumps(detail, sort_keys=True, separators=(",", ":"))

    seed = _as_int(_first(row, "seed", "episode_seed"))
    episode = _first(row, "episode", "episode_id", "id")
    # A replay run has no seed (one episode per recorded take), so the episode id
    # is the pairing key there.  Without this every replay row would collapse
    # into one anonymous bucket.
    key = seed if seed is not None else (str(episode) if episode is not None else None)

    return {
        "seed": seed,
        "episode": None if episode is None else str(episode),
        "key": key,
        "outcome": outcome,
        "success": success,
        "n_steps": _as_int(_first(row, "n_steps", "steps", "step_count")),
        "t_success_s": _as_float(_first(row, "t_success_s", "t_success", "time_to_success_s")),
        "wall_s": _as_float(_first(row, "wall_s", "wall_time_s", "wall_time", "elapsed_s")),
        "detail": "" if detail is None else str(detail),
        "policy": _first(row, "policy", "policy_id", "policy_meta"),
    }


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Rows + count of lines that were not a JSON object (never raises on those)."""

    rows: list[dict[str, Any]] = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            obj = json.load(stream)
    except FileNotFoundError:
        return {}, "missing"
    except (OSError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    return (obj, None) if isinstance(obj, dict) else ({}, "not a JSON object")


def _label_for(run_dir: Path, summary: Mapping[str, Any]) -> str:
    for key in ("label", "run_name", "name"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return run_dir.name or str(run_dir)


def _checkpoint_of(summary: Mapping[str, Any]) -> str:
    """Checkpoint identity, top level or inside ``policy_meta``.

    ``run_eval.summarize`` puts the policy's own metadata (which is where the
    RESET v2 ``checkpoint`` field lands) under ``policy_meta``, not at the top.
    """

    meta = summary.get("policy_meta")
    sources: list[Mapping[str, Any]] = [summary]
    if isinstance(meta, Mapping):
        sources.append(meta)
    for source in sources:
        for key in ("checkpoint", "checkpoint_id", "checkpoint_path", "model", "model_id"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _sort_key(value: Any) -> tuple[int, Any]:
    """Sort ints before strings; a run keyed by take name must not blow up."""

    return (1, str(value)) if isinstance(value, str) else (0, value)


def load_run(run_dir: str | os.PathLike[str], label: str | None = None) -> dict[str, Any]:
    """Read one run dir.  Never raises: a broken run becomes an ``errors`` list."""

    path = Path(run_dir).expanduser()
    errors: list[str] = []
    summary, summary_error = _read_json(path / SUMMARY_JSON)
    if summary_error and summary_error != "missing":
        errors.append(f"{SUMMARY_JSON}: {summary_error}")
    elif summary_error == "missing":
        errors.append(f"no {SUMMARY_JSON} (identity fields unavailable)")

    episodes_path = path / EPISODES_JSONL
    bad_lines = 0
    if episodes_path.is_file():
        try:
            raw_rows, bad_lines = _read_jsonl(episodes_path)
        except OSError as exc:
            raw_rows = []
            errors.append(f"{EPISODES_JSONL}: {type(exc).__name__}: {exc}")
    else:
        raw_rows = []
        errors.append(f"no {EPISODES_JSONL} — this run contributes no episodes")
    if bad_lines:
        errors.append(f"{bad_lines} malformed line(s) in {EPISODES_JSONL} skipped")

    hz = _as_float(_first(summary, "control_hz", "hz", "rate_hz")) or DEFAULT_CONTROL_HZ
    episodes = [normalize_episode(row) for row in raw_rows]

    # Only derive a time-to-success when the episode succeeded and the writer did
    # not record one; mark it so the report can footnote the derived values.
    derived = 0
    for ep in episodes:
        ep["t_derived"] = False
        if ep["success"] and ep["t_success_s"] is None and ep["n_steps"] is not None and hz > 0:
            ep["t_success_s"] = ep["n_steps"] / hz
            ep["t_derived"] = True
            derived += 1

    counts: dict[str, int] = {}
    for ep in episodes:
        counts[ep["outcome"]] = counts.get(ep["outcome"], 0) + 1

    scored = [ep for ep in episodes if ep["success"] is not None]
    successes = sum(1 for ep in scored if ep["success"])
    n_scored = len(scored)
    lo, hi = wilson_interval(successes, n_scored)

    times = sorted(ep["t_success_s"] for ep in scored
                   if ep["success"] and ep["t_success_s"] is not None)

    return {
        "dir": str(path),
        "label": label or _label_for(path, summary),
        "summary": summary,
        "checkpoint": _checkpoint_of(summary),
        "policy": str(summary.get("policy") or summary.get("policy_id") or ""),
        "git_commit": str(summary.get("git_commit") or ""),
        "config_sha": str(summary.get("config_sha") or summary.get("config_sha256") or ""),
        "control_hz": hz,
        "episodes": episodes,
        "n_episodes": len(episodes),
        "n_scored": n_scored,
        "n_unknown": len(episodes) - n_scored,
        "successes": successes,
        "sr": (successes / n_scored) if n_scored else None,
        "ci": (lo, hi),
        "counts": counts,
        "t_success": {
            "n": len(times),
            "mean": statistics.fmean(times) if times else None,
            "median": statistics.median(times) if times else None,
            "min": times[0] if times else None,
            "max": times[-1] if times else None,
            "derived": derived,
        },
        "errors": errors,
    }


# =============================================================================
# cross-run views
# =============================================================================
def per_seed_table(runs: Sequence[Mapping[str, Any]]) -> tuple[list[int | None], dict[str, dict[Any, str]]]:
    """(sorted seed list, {label: {seed: outcome}}).

    A seed a run never attempted shows as an empty cell, which is the point of
    the table: a missing seed and a failed seed must not look the same.
    """

    by_run: dict[str, dict[Any, str]] = {}
    seeds: set[Any] = set()
    for run in runs:
        cell: dict[Any, str] = {}
        for index, ep in enumerate(run["episodes"]):
            key = ep["key"] if ep["key"] is not None else f"row{index}"
            seeds.add(key)
            # A duplicate seed inside one run is a writer bug; keep both visible.
            cell[key] = f"{cell[key]} / {ep['outcome']}" if key in cell else ep["outcome"]
        by_run[run["label"]] = cell
    ints = sorted(s for s in seeds if isinstance(s, int))
    rest = sorted(str(s) for s in seeds if not isinstance(s, int))
    return ints + rest, by_run  # type: ignore[return-value]


def pairwise_compare(run_a: Mapping[str, Any], run_b: Mapping[str, Any]) -> dict[str, Any]:
    """Paired (same-seed) comparison of two runs."""

    def by_seed(run: Mapping[str, Any]) -> dict[Any, bool | None]:
        out: dict[Any, bool | None] = {}
        for ep in run["episodes"]:
            if ep["key"] is not None:
                out[ep["key"]] = ep["success"]
        return out

    a, b = by_seed(run_a), by_seed(run_b)
    shared = sorted(set(a) & set(b), key=_sort_key)
    both = a_only = b_only = neither = 0
    a_only_seeds: list[Any] = []
    b_only_seeds: list[Any] = []
    for seed in shared:
        sa, sb = a[seed], b[seed]
        if sa is None or sb is None:
            continue
        if sa and sb:
            both += 1
        elif sa and not sb:
            a_only += 1
            a_only_seeds.append(seed)
        elif sb and not sa:
            b_only += 1
            b_only_seeds.append(seed)
        else:
            neither += 1

    sr_a, sr_b = run_a["sr"], run_b["sr"]
    return {
        "a": run_a["label"],
        "b": run_b["label"],
        "shared_seeds": len(shared),
        "paired_scored": both + a_only + b_only + neither,
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "a_only_seeds": a_only_seeds,
        "b_only_seeds": b_only_seeds,
        "sr_a": sr_a,
        "sr_b": sr_b,
        "sr_delta": (sr_b - sr_a) if (sr_a is not None and sr_b is not None) else None,
        "mcnemar_p": mcnemar_exact(a_only, b_only),
        "only_in_a": sorted(set(a) - set(b), key=_sort_key),
        "only_in_b": sorted(set(b) - set(a), key=_sort_key),
    }


def build_report(
    run_dirs: Sequence[str | os.PathLike[str]],
    labels: Sequence[str] | None = None,
    compare: bool = False,
) -> dict[str, Any]:
    labels = list(labels or [])
    runs: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for index, run_dir in enumerate(run_dirs):
        label = labels[index] if index < len(labels) else None
        run = load_run(run_dir, label)
        # Distinct labels are load-bearing: they key the per-seed columns.
        base = run["label"]
        if base in seen:
            seen[base] += 1
            run["label"] = f"{base}#{seen[base]}"
        else:
            seen[base] = 1
        runs.append(run)

    seeds, by_run = per_seed_table(runs)
    comparisons: list[dict[str, Any]] = []
    if compare and len(runs) >= 2:
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                comparisons.append(pairwise_compare(runs[i], runs[j]))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
        "seeds": seeds,
        "per_seed": by_run,
        "comparisons": comparisons,
    }


# =============================================================================
# rendering
# =============================================================================
def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def _signed_pct(value: float | None, digits: int = 1) -> str:
    """A delta needs its sign visible; "40.0%" reads like a level, not a change."""

    if value is None:
        return "n/a"
    return f"{100.0 * value:+.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    count = 0
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
        count += 1
    if count == 0:
        out.append("| " + " | ".join("-" for _ in headers) + " |")
    return out


def _cell(outcome: str) -> str:
    if outcome == SUCCESS:
        return "✓"
    if outcome == UNKNOWN:
        return "?"
    return outcome


def render_markdown(report: Mapping[str, Any]) -> str:
    runs = report["runs"]
    lines: list[str] = []
    add = lines.append

    add("# sim eval report")
    add("")
    add(f"generated `{report['generated_at_utc']}` by `sim_collect/eval/report.py`")
    add("")

    # ---- 1. headline ------------------------------------------------------
    add("## 1. success rate")
    add("")
    rows = []
    for run in runs:
        lo, hi = run["ci"]
        rows.append((
            run["label"],
            run["policy"] or "n/a",
            run["checkpoint"] or "n/a",
            f"{run['successes']}/{run['n_scored']}",
            _pct(run["sr"]),
            f"[{_pct(lo)}, {_pct(hi)}]",
            _num(run["t_success"]["mean"]),
            _num(run["t_success"]["median"]),
        ))
    lines.extend(_md_table(
        ["run", "policy", "checkpoint", "success", "SR", "Wilson 95 % CI",
         "mean t_succ (s)", "median t_succ (s)"],
        rows,
    ))
    add("")

    notes: list[str] = []
    for run in runs:
        if run["n_unknown"]:
            notes.append(
                f"`{run['label']}`: {run['n_unknown']} episode(s) had no usable outcome "
                "and are excluded from the SR denominator"
            )
        if run["t_success"]["derived"]:
            notes.append(
                f"`{run['label']}`: {run['t_success']['derived']} time(s)-to-success were "
                f"derived as n_steps / {run['control_hz']:g} Hz because the row had no "
                "`t_success_s`"
            )
        for error in run["errors"]:
            notes.append(f"`{run['label']}`: {error}")
    if notes:
        add("Notes:")
        add("")
        for note in notes:
            add(f"- {note}")
        add("")

    add("With 20 seeds the Wilson interval is about ±0.20 wide at SR 0.5. Overlapping "
        "intervals mean **this run did not separate the two**, not that they are equal.")
    add("")

    # ---- 2. outcomes ------------------------------------------------------
    add("## 2. outcome breakdown")
    add("")
    outcomes: list[str] = []
    for run in runs:
        for name in run["counts"]:
            if name not in outcomes:
                outcomes.append(name)
    outcomes.sort(key=lambda name: (name != SUCCESS, name))
    lines.extend(_md_table(
        ["outcome"] + [run["label"] for run in runs],
        [[name] + [run["counts"].get(name, 0) for run in runs] for name in outcomes],
    ))
    add("")

    # ---- 3. per seed ------------------------------------------------------
    add("## 3. per-seed outcomes")
    add("")
    add("`✓` = success, `?` = no usable outcome, empty = the run never attempted that seed.")
    add("")
    per_seed = report["per_seed"]
    lines.extend(_md_table(
        ["seed"] + [run["label"] for run in runs],
        [[seed] + [_cell(per_seed[run["label"]][seed]) if seed in per_seed[run["label"]] else ""
                   for run in runs]
         for seed in report["seeds"]],
    ))
    add("")

    # ---- 4. comparison ----------------------------------------------------
    if report["comparisons"]:
        add("## 4. paired comparison")
        add("")
        add("Seeds are fixed, so A and B saw the **same scene layouts**; the paired test "
            "(McNemar exact, two-sided) is the one that uses that. `p` is not a verdict: "
            "at 20 seeds it needs ~6 discordant seeds all one way to reach 0.05.")
        add("")
        lines.extend(_md_table(
            ["A", "B", "SR A", "SR B", "Δ (B−A)", "shared seeds",
             "both", "A only", "B only", "neither", "McNemar p"],
            [[
                cmp["a"], cmp["b"], _pct(cmp["sr_a"]), _pct(cmp["sr_b"]),
                _signed_pct(cmp["sr_delta"]),
                cmp["shared_seeds"], cmp["both"], cmp["a_only"], cmp["b_only"],
                cmp["neither"], f"{cmp['mcnemar_p']:.3f}",
            ] for cmp in report["comparisons"]],
        ))
        add("")
        for cmp in report["comparisons"]:
            if cmp["a_only_seeds"] or cmp["b_only_seeds"]:
                add(f"- `{cmp['a']}` vs `{cmp['b']}`: only `{cmp['a']}` solved "
                    f"{cmp['a_only_seeds'] or '[]'}; only `{cmp['b']}` solved "
                    f"{cmp['b_only_seeds'] or '[]'} — replay those seeds before believing Δ.")
            if cmp["only_in_a"] or cmp["only_in_b"]:
                add(f"- ⚠️ seed lists differ: `{cmp['a']}`-only {cmp['only_in_a'] or '[]'}, "
                    f"`{cmp['b']}`-only {cmp['only_in_b'] or '[]'}. The paired columns use "
                    "the intersection.")
        add("")

    # ---- 5. provenance ----------------------------------------------------
    add("## 5. provenance")
    add("")
    lines.extend(_md_table(
        ["run", "dir", "git commit", "config sha", "episodes"],
        [[run["label"], f"`{run['dir']}`", run["git_commit"] or "n/a",
          (run["config_sha"][:12] + "…") if len(run["config_sha"]) > 12 else (run["config_sha"] or "n/a"),
          run["n_episodes"]] for run in runs],
    ))
    add("")
    return "\n".join(lines) + "\n"


# =============================================================================
# CSV
# =============================================================================
EPISODE_CSV_FIELDS = [
    "run", "run_dir", "episode", "seed", "outcome", "success", "n_steps",
    "t_success_s", "t_success_derived", "wall_s", "detail",
]


def episode_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in report["runs"]:
        for ep in run["episodes"]:
            rows.append({
                "run": run["label"],
                "run_dir": run["dir"],
                "episode": ep["episode"] or "",
                "seed": "" if ep["seed"] is None else ep["seed"],
                "outcome": ep["outcome"],
                "success": "" if ep["success"] is None else int(ep["success"]),
                "n_steps": "" if ep["n_steps"] is None else ep["n_steps"],
                "t_success_s": "" if ep["t_success_s"] is None else f"{ep['t_success_s']:.4f}",
                "t_success_derived": int(bool(ep["t_derived"])),
                "wall_s": "" if ep["wall_s"] is None else f"{ep['wall_s']:.3f}",
                "detail": ep["detail"],
            })
    return rows


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_runs_csv(path: Path, report: Mapping[str, Any]) -> None:
    outcomes: list[str] = []
    for run in report["runs"]:
        for name in run["counts"]:
            if name not in outcomes:
                outcomes.append(name)
    outcomes.sort()
    fields = ["run", "run_dir", "policy", "checkpoint", "episodes", "scored",
              "unknown", "successes", "sr", "ci_lo", "ci_hi",
              "t_success_mean_s", "t_success_median_s", "git_commit"] + \
             [f"n_{name}" for name in outcomes]
    rows = []
    for run in report["runs"]:
        lo, hi = run["ci"]
        row = {
            "run": run["label"], "run_dir": run["dir"], "policy": run["policy"],
            "checkpoint": run["checkpoint"], "episodes": run["n_episodes"],
            "scored": run["n_scored"], "unknown": run["n_unknown"],
            "successes": run["successes"],
            "sr": "" if run["sr"] is None else f"{run['sr']:.6f}",
            "ci_lo": f"{lo:.6f}", "ci_hi": f"{hi:.6f}",
            "t_success_mean_s": "" if run["t_success"]["mean"] is None else f"{run['t_success']['mean']:.4f}",
            "t_success_median_s": "" if run["t_success"]["median"] is None else f"{run['t_success']['median']:.4f}",
            "git_commit": run["git_commit"],
        }
        for name in outcomes:
            row[f"n_{name}"] = run["counts"].get(name, 0)
        rows.append(row)
    write_csv(path, fields, rows)


def write_per_seed_csv(path: Path, report: Mapping[str, Any]) -> None:
    labels = [run["label"] for run in report["runs"]]
    fields = ["seed"] + labels
    per_seed = report["per_seed"]
    rows = [{"seed": seed, **{label: per_seed[label].get(seed, "") for label in labels}}
            for seed in report["seeds"]]
    write_csv(path, fields, rows)


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sim_collect.eval.report",
        description="Aggregate run_eval output dirs into a markdown + CSV report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  report.py sim_collect/eval/runs/act_40k\n"
            "  report.py runs/a runs/b --label 40k --label 80k --compare "
            "--out runs/compare\n"
        ),
    )
    parser.add_argument("runs", nargs="+", help="run_eval output directories")
    parser.add_argument("--label", action="append", default=[],
                        help="column name for the matching run (repeatable, positional)")
    parser.add_argument("--out", default=None,
                        help="directory for report.md / episodes.csv / per_seed.csv / runs.csv")
    parser.add_argument("--md", default=None, help="explicit markdown path (overrides --out)")
    parser.add_argument("--csv", default=None, help="explicit episodes-CSV path (overrides --out)")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="also dump the raw aggregate as JSON")
    parser.add_argument("--compare", action="store_true",
                        help="add the paired (same-seed) comparison section")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the markdown to stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.label and len(args.label) > len(args.runs):
        print("error: more --label values than run directories", file=sys.stderr)
        return 2

    report = build_report(args.runs, args.label, compare=args.compare)
    if not any(run["n_episodes"] for run in report["runs"]):
        for run in report["runs"]:
            for error in run["errors"]:
                print(f"warning: {run['label']}: {error}", file=sys.stderr)
        print("error: no episodes were readable in any run directory", file=sys.stderr)
        return 2

    markdown = render_markdown(report)

    out_dir = Path(args.out).expanduser() if args.out else None
    md_path = Path(args.md).expanduser() if args.md else (out_dir / "report.md" if out_dir else None)
    csv_path = Path(args.csv).expanduser() if args.csv else (out_dir / "episodes.csv" if out_dir else None)

    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        print(f"wrote {md_path}", file=sys.stderr)
    if csv_path is not None:
        write_csv(csv_path, EPISODE_CSV_FIELDS, episode_rows(report))
        print(f"wrote {csv_path}", file=sys.stderr)
    if out_dir is not None:
        write_per_seed_csv(out_dir / "per_seed.csv", report)
        write_runs_csv(out_dir / "runs.csv", report)
        print(f"wrote {out_dir / 'per_seed.csv'}", file=sys.stderr)
        print(f"wrote {out_dir / 'runs.csv'}", file=sys.stderr)
    if args.json_path:
        json_path = Path(args.json_path).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {json_path}", file=sys.stderr)

    if not args.quiet:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
