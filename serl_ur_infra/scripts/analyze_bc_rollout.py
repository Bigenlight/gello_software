#!/usr/bin/env python3
"""Offline post-rollout analyzer for a BC evaluation run.

WHAT THIS IS
------------
A BC rollout leaves its evidence in two places that never talk to each other:

  (a) the SERVER-side record root written by
      ``ur_env/bc_recording_sink.py::EpisodeRecordingSink`` -- ``metadata.json``,
      the append-only ``actions.jsonl``, one ``<run_id>/episode_XXXX.pkl`` per
      completed episode, and (when the BC server was started without
      ``--no-inference-log``) ``inference.jsonl``.

  (b) the LAPTOP3 robot-side diagnostic recording written by
      ``ros2_ur_ws/src/gello_recorder`` -- ``vectors.h5`` (the nine
      ``Hdf5TableWriter`` tables) plus, optionally, a ``status.jsonl``.

This script merges them into one human-readable report.  It trains nothing,
touches no hardware, opens no socket and imports no jax: stdlib + numpy, and
h5py only when ``--robot`` is given.

USAGE
-----
    PY=/home/laptop3/venvs/gello-hil-actor/bin/python

    # server-side only; writes <served>/analysis/{report.json,rollout_report.md}
    $PY serl_ur_infra/scripts/analyze_bc_rollout.py \
        --served ~/bc_rollouts/run_20260731_2200

    # merge in the laptop3 robot-side recording and pick an output dir
    $PY serl_ur_infra/scripts/analyze_bc_rollout.py \
        --served ~/bc_rollouts/run_20260731_2200 \
        --robot  ros2_ur_ws/gello_logs/session_20260731_220145 \
        --out    /tmp/bc_report

    # add a per-step deep dive for one episode
    $PY serl_ur_infra/scripts/analyze_bc_rollout.py \
        --served ~/bc_rollouts/run_20260731_2200 --episode 3

THE TIME BASE -- READ THIS BEFORE TRUSTING SECTION 4
----------------------------------------------------
The three clocks in play are NOT the same clock:

  * ``actions.jsonl`` ``ts`` is ``datetime.now(timezone.utc)`` taken inside the
    BC policy SERVER process (``EpisodeRecordingSink.__call__``).  When the
    server runs on the GPU box, this is a FOREIGN host's clock and its skew
    against laptop3 is unbounded and uncorrected here.
  * ``meta.timestamp_ns`` inside each episode pickle is ``time.time_ns()``
    stamped by ``remote_actor.EnvTimestampAdapter`` in the ACTOR process on
    laptop3 -- the same host, and the same ``time.time`` epoch, as the
    recorder.  This analyzer therefore PREFERS the pickle timestamps as the
    alignment anchor and only falls back to ``actions.jsonl`` ts.
  * ``vectors.h5`` stores ``t_rel_s`` = ``time.time() - t0`` where ``t0`` is
    fixed at ``RecordingSession.__init__``.  It is a RELATIVE clock and every
    table uses it.  The only absolute anchor inside the file is the
    ``synchronized`` table's ``t_wall`` column (raw ``time.time()``), so the
    epoch origin is recovered as ``median(t_wall - t_rel_s)``.  If the
    ``synchronized`` table is absent or empty, the recorder's own
    ``metadata.json`` ``start_wall`` (a NAIVE LOCAL-TIME isoformat string from
    ``datetime.fromtimestamp``) is used instead, which additionally assumes the
    reader's local timezone matches the recorder's.

Whichever of those held, the report states it in prose in section 4 so a reader
never has to come back here to find out what was assumed.

CONSTRAINTS HONOURED HERE
-------------------------
  * Pixels never enter the report.  Episode pickles are loaded ONE AT A TIME and
    dropped immediately; only observation SHAPES and DTYPES are reported.
  * Every missing input degrades to a clearly-marked "not recorded" section.
    Malformed jsonl lines are skipped and counted, never fatal.
  * Nothing is written outside ``--out``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import pickle
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Filenames -- mirrored from the writers, not guessed.
# ---------------------------------------------------------------------------

#: bc_recording_sink.METADATA_FILENAME
SERVED_METADATA = "metadata.json"
#: bc_recording_sink.ACTIONS_FILENAME
SERVED_ACTIONS = "actions.jsonl"
#: written by the BC server's InferenceLoggingPolicy (sibling work); optional.
SERVED_INFERENCE = "inference.jsonl"
#: RecordingSession opens exactly this name under its session dir.
ROBOT_VECTORS = "vectors.h5"
#: operator/status stream that rides alongside the h5; optional.
ROBOT_STATUS = "status.jsonl"
#: gello_ur_recorder_node._write_metadata target.
ROBOT_METADATA = "metadata.json"

#: bc_recording_sink.ACTION_DIM -- 6 pose deltas + gripper.
ACTION_DIM = 7
#: recording_session._N -- the UR has six joints and every column block is 6 wide.
UR_JOINTS = 6

#: A recorded gripper command counts as "changed" past this delta.  Actions are
#: stored as float32 casts, so anything at 1e-6 is quantisation, not a command.
GRIPPER_CHANGE_EPS = 1e-3
#: Above/below this the gripper command is read as a commanded close/open.
GRIPPER_STATE_THRESHOLD = 0.5
#: float32 is the action dtype on both sides, so cross-checking any tighter
#: than this compares rounding noise rather than actions.
ACTION_MATCH_ATOL = 1e-6

NOT_RECORDED = "**NOT RECORDED**"


# ---------------------------------------------------------------------------
# Small formatting / parsing helpers
# ---------------------------------------------------------------------------


def _num(value: Any, digits: int = 3) -> str:
    """Render a number for a markdown cell; anything unusable becomes ``n/a``."""

    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.{digits}f}"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """Emit a GitHub-flavoured markdown table (no padding: diffs stay small)."""

    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    count = 0
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
        count += 1
    if count == 0:
        out.append("| " + " | ".join("-" for _ in headers) + " |")
    return out


def _parse_ts(value: Any) -> float | None:
    """Best-effort epoch-seconds from the several ts spellings in play.

    The sink writes ``datetime.now(timezone.utc).isoformat()``; a sibling writer
    may well use a bare float.  Numeric values are disambiguated by magnitude
    (s / ms / us / ns) rather than by trusting a field name.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.generic)):
        f = float(value)
        if not math.isfinite(f):
            return None
        magnitude = abs(f)
        if magnitude >= 1e17:      # nanoseconds
            return f / 1e9
        if magnitude >= 1e14:      # microseconds
            return f / 1e6
        if magnitude >= 1e11:      # milliseconds
            return f / 1e3
        return f
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        # ``Z`` is legal ISO-8601 but fromisoformat only learned it in 3.11.
        candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            # Naive: the writer used local time (recorder metadata does).
            parsed = parsed.astimezone()
        return parsed.timestamp()
    return None


def _epoch_text(epoch: float | None) -> str:
    if epoch is None or not math.isfinite(epoch):
        return "n/a"
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds")


def _short_sha(value: Any, width: int = 12) -> str:
    text = str(value or "").strip()
    if not text:
        return "n/a"
    return text[:width]


def _finite(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _percentile(values: np.ndarray, q: float) -> float | None:
    values = _finite(values)
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _read_jsonl(path: Path) -> dict[str, Any]:
    """Read a jsonl file into rows, counting (never raising on) bad lines.

    A truncated final line is the normal shape of a file that was being
    appended to when the process died, so it is a skip, not an error.
    """

    result: dict[str, Any] = {
        "path": str(path),
        "present": False,
        "rows": [],
        "bad_lines": 0,
        "bad_line_numbers": [],
        "error": None,
    }
    if not path.is_file():
        return result
    result["present"] = True
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except ValueError:
                    result["bad_lines"] += 1
                    if len(result["bad_line_numbers"]) < 20:
                        result["bad_line_numbers"].append(number)
                    continue
                if not isinstance(obj, dict):
                    result["bad_lines"] += 1
                    if len(result["bad_line_numbers"]) < 20:
                        result["bad_line_numbers"].append(number)
                    continue
                rows.append(obj)
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["rows"] = rows
    return result


def _read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path), "present": False, "data": None, "error": None
    }
    if not path.is_file():
        return result
    result["present"] = True
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    if not isinstance(data, dict):
        result["error"] = f"expected a JSON object, got {type(data).__name__}"
        return result
    result["data"] = data
    return result


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# (a) served side -- actions.jsonl
# ---------------------------------------------------------------------------


def _episode_key(run_id: Any, episode_id: Any) -> tuple[str, str]:
    return (str(run_id), str(episode_id))


def summarize_actions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold ``actions.jsonl`` rows into per-episode statistics.

    Rows whose shape does not match the sink's contract are counted as
    ``skipped`` rather than aborting the report: a partially-written line is
    exactly what a killed session leaves behind.
    """

    episodes: dict[tuple[str, str], dict[str, Any]] = {}
    skipped = 0
    totals = {"rows": 0, "intervened": 0, "reward_sum": 0.0}

    for row in rows:
        try:
            actions = np.asarray(row["actions"], dtype=np.float64).reshape(-1)
            if actions.shape != (ACTION_DIM,) or not np.all(np.isfinite(actions)):
                raise ValueError("action shape/finiteness")
            key = _episode_key(row.get("run_id", ""), row.get("episode_id", ""))
            step_id = int(row["step_id"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        entry = episodes.setdefault(
            key,
            {
                "run_id": key[0],
                "episode_id": key[1],
                "steps": 0,
                "intervened_steps": 0,
                "success": False,
                "dones": False,
                "truncated": False,
                "reward_sum": 0.0,
                "step_ids": [],
                "env_steps": [],
                "ts": [],
                "_actions": [],
            },
        )
        entry["steps"] += 1
        entry["step_ids"].append(step_id)
        entry["_actions"].append(actions)
        entry["intervened_steps"] += int(bool(row.get("intervened", False)))
        entry["success"] = entry["success"] or bool(row.get("success", False))
        entry["dones"] = entry["dones"] or bool(row.get("dones", False))
        entry["truncated"] = entry["truncated"] or bool(row.get("truncated", False))
        try:
            entry["reward_sum"] += float(row.get("rewards", 0.0))
        except (TypeError, ValueError):
            pass
        try:
            entry["env_steps"].append(int(row["env_step"]))
        except (KeyError, TypeError, ValueError):
            pass
        epoch = _parse_ts(row.get("ts"))
        if epoch is not None:
            entry["ts"].append(epoch)

        totals["rows"] += 1
        totals["intervened"] += int(bool(row.get("intervened", False)))

    for entry in episodes.values():
        stacked = np.stack(entry.pop("_actions")) if entry["steps"] else np.zeros((0, ACTION_DIM))
        entry.update(_action_group_stats(stacked))
        ts = sorted(entry.pop("ts"))
        entry["ts_first"] = ts[0] if ts else None
        entry["ts_last"] = ts[-1] if ts else None
        entry["ts_duration_s"] = (ts[-1] - ts[0]) if len(ts) >= 2 else None
        step_ids = entry.pop("step_ids")
        entry["step_id_min"] = min(step_ids) if step_ids else None
        entry["step_id_max"] = max(step_ids) if step_ids else None
        env_steps = entry.pop("env_steps")
        entry["env_step_min"] = min(env_steps) if env_steps else None
        entry["env_step_max"] = max(env_steps) if env_steps else None
        entry["outcome"] = _outcome(entry)
        totals["reward_sum"] += entry["reward_sum"]

    return {
        "episodes": episodes,
        "skipped_rows": skipped,
        "totals": totals,
    }


def _action_group_stats(stacked: np.ndarray) -> dict[str, Any]:
    """Mean/max L2 norm for the translation and rotation action groups.

    The 7-D contract is ``[dx, dy, dz, drx, dry, drz, gripper]``; the norm is
    taken per group because a per-axis maximum is not rotation invariant (the
    same reason ``RelativeFrame`` forces norm-proportional scaling upstream).
    """

    if stacked.size == 0:
        return {
            "trans_norm_mean": None, "trans_norm_max": None,
            "rot_norm_mean": None, "rot_norm_max": None,
            "gripper_changes": 0, "gripper_state_flips": 0,
            "gripper_min": None, "gripper_max": None,
        }
    trans = np.linalg.norm(stacked[:, 0:3], axis=1)
    rot = np.linalg.norm(stacked[:, 3:6], axis=1)
    grip = stacked[:, 6]
    if grip.size >= 2:
        deltas = np.abs(np.diff(grip))
        changes = int(np.count_nonzero(deltas > GRIPPER_CHANGE_EPS))
        state = np.where(
            grip > GRIPPER_STATE_THRESHOLD, 1,
            np.where(grip < -GRIPPER_STATE_THRESHOLD, -1, 0),
        )
        flips = int(np.count_nonzero(np.diff(state) != 0))
    else:
        changes = 0
        flips = 0
    return {
        "trans_norm_mean": float(trans.mean()),
        "trans_norm_max": float(trans.max()),
        "rot_norm_mean": float(rot.mean()),
        "rot_norm_max": float(rot.max()),
        "gripper_changes": changes,
        "gripper_state_flips": flips,
        "gripper_min": float(grip.min()),
        "gripper_max": float(grip.max()),
    }


def _outcome(entry: Mapping[str, Any]) -> str:
    """Outcome precedence mirrors the actor's own terminal-reason precedence."""

    if entry.get("success"):
        return "success"
    if entry.get("truncated"):
        return "truncated"
    if entry.get("dones"):
        return "done"
    return "open"


# ---------------------------------------------------------------------------
# (a) served side -- episode pickles
# ---------------------------------------------------------------------------


def _describe_observation(obs: Any) -> Any:
    """Shapes and dtypes ONLY.  Pixels must never reach the report."""

    if isinstance(obs, Mapping):
        described = {}
        for key, value in obs.items():
            array = np.asarray(value)
            described[str(key)] = {
                "shape": [int(d) for d in array.shape],
                "dtype": str(array.dtype),
            }
        return described
    array = np.asarray(obs)
    return {"<array>": {"shape": [int(d) for d in array.shape], "dtype": str(array.dtype)}}


def scan_episode_pickle(path: Path, *, deep_dive_key: tuple[str, str] | None) -> dict[str, Any]:
    """Load ONE episode pickle, extract a summary, drop it again.

    The list held by a pickle carries every observation and next_observation of
    the episode, i.e. the pixels.  Nothing derived from them but shape/dtype
    survives this function, and the list itself is released before returning so
    at most one episode's images are resident at a time.
    """

    summary: dict[str, Any] = {
        "path": str(path),
        "items": 0,
        "error": None,
        "run_id": None,
        "episode_id": None,
        "timestamp_ns_first": None,
        "timestamp_ns_last": None,
        "duration_s": None,
        "policy_versions": [],
        "schema_versions": [],
        "intervened_items": 0,
        "auto_success_items": 0,
        "operator_success_items": 0,
        "synthetic_policy_actions": False,
        "observation_spec": None,
        "deep_dive_rows": None,
    }
    try:
        with open(path, "rb") as stream:
            items = pickle.load(stream)
    except Exception as exc:  # noqa: BLE001 - a bad pickle must not end the run
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    try:
        if not isinstance(items, list):
            summary["error"] = f"expected a list of wrappers, got {type(items).__name__}"
            return summary
        summary["items"] = len(items)
        stamps: list[int] = []
        policy_versions: set[int] = set()
        schema_versions: set[Any] = set()
        deep_rows: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            meta = item.get("meta") or {}
            transition = item.get("transition") or {}
            if summary["run_id"] is None:
                summary["run_id"] = str(meta.get("run_id", ""))
            if summary["episode_id"] is None and "episode_id" in transition:
                episode_id = transition["episode_id"]
                if isinstance(episode_id, np.generic):
                    episode_id = episode_id.item()
                summary["episode_id"] = str(episode_id)
            try:
                stamps.append(int(meta["timestamp_ns"]))
            except (KeyError, TypeError, ValueError):
                pass
            try:
                policy_versions.add(int(meta["policy_version"]))
            except (KeyError, TypeError, ValueError):
                pass
            if "schema_version" in meta:
                schema_versions.add(meta["schema_version"])
            summary["intervened_items"] += int(bool(meta.get("intervened", False)))
            summary["auto_success_items"] += int(bool(meta.get("auto_success", False)))
            summary["operator_success_items"] += int(
                bool(meta.get("operator_success", False))
            )
            if bool(meta.get("policy_actions_synthetic", False)):
                summary["synthetic_policy_actions"] = True
            if index == 0 and "observations" in transition:
                summary["observation_spec"] = _describe_observation(
                    transition["observations"]
                )
            if deep_dive_key is not None:
                deep_rows.append(
                    {
                        "step_id": transition.get("step_id"),
                        "env_step": meta.get("env_step"),
                        "timestamp_ns": meta.get("timestamp_ns"),
                        "actions": _action_list(transition.get("actions")),
                        "policy_action": _action_list(meta.get("policy_action")),
                        "intervened": bool(meta.get("intervened", False)),
                        "rewards": _float_or_none(transition.get("rewards")),
                        "masks": _float_or_none(transition.get("masks")),
                        "dones": bool(transition.get("dones", False)),
                        "truncated": bool(transition.get("truncated", False)),
                    }
                )
        if stamps:
            summary["timestamp_ns_first"] = min(stamps)
            summary["timestamp_ns_last"] = max(stamps)
            summary["duration_s"] = (max(stamps) - min(stamps)) / 1e9
        summary["policy_versions"] = sorted(policy_versions)
        summary["schema_versions"] = sorted(str(v) for v in schema_versions)
        if deep_dive_key is not None:
            summary["deep_dive_rows"] = deep_rows
    finally:
        # Release the pixels before the caller moves to the next episode.
        del items
        gc.collect()
    return summary


def _action_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return [float(v) for v in array]


def _float_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def scan_served_pickles(
    served: Path, *, deep_dive: tuple[str, str] | None
) -> dict[str, Any]:
    """Walk ``<served>/<run_id>/episode_*.pkl`` one file at a time."""

    found: dict[tuple[str, str], dict[str, Any]] = {}
    paths = sorted(served.glob("*/episode_*.pkl"))
    for path in paths:
        run_id = path.parent.name
        stem = path.stem  # episode_XXXX
        file_episode = stem[len("episode_"):]
        wants_deep = (
            deep_dive is not None
            and str(deep_dive[1]) in {file_episode, file_episode.lstrip("0") or "0"}
            and (deep_dive[0] in ("", run_id))
        )
        summary = scan_episode_pickle(path, deep_dive_key=deep_dive if wants_deep else None)
        # The filename is authoritative for keying: a pickle that failed to load
        # still has to appear in the table under the episode it belongs to.
        key = _episode_key(run_id, summary.get("episode_id") or _int_or_text(file_episode))
        found[key] = summary
    return {"episodes": found, "files": [str(p) for p in paths]}


def _int_or_text(text: str) -> Any:
    try:
        return int(text)
    except ValueError:
        return text


# ---------------------------------------------------------------------------
# (a) served side -- inference.jsonl
# ---------------------------------------------------------------------------


def summarize_inference(
    inference: Mapping[str, Any], action_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Latency percentiles plus a nearest-timestamp join back to the executed action.

    The join is deliberately reported as an aggregate fraction: a per-row diff
    of "requested vs executed" is dominated by human intervention rewriting the
    action, which is correct behaviour, not an anomaly.
    """

    result: dict[str, Any] = {
        "present": bool(inference.get("present")),
        "path": inference.get("path"),
        "error": inference.get("error"),
        "bad_lines": int(inference.get("bad_lines", 0)),
        "lines": 0,
        "latency_ms": {"count": 0, "p50": None, "p95": None, "max": None, "mean": None},
        "policy_versions": [],
        "deterministic_true": 0,
        "deterministic_false": 0,
        "state_present": 0,
        "state_dims": [],
        "cross_check": None,
        "ts_first": None,
        "ts_last": None,
    }
    rows = list(inference.get("rows") or [])
    if not result["present"]:
        return result
    result["lines"] = len(rows)

    latencies: list[float] = []
    versions: set[int] = set()
    state_dims: set[int] = set()
    stamps: list[float] = []
    parsed: list[tuple[float | None, np.ndarray | None]] = []

    for row in rows:
        latency = _float_or_none(row.get("latency_ms"))
        if latency is not None:
            latencies.append(latency)
        try:
            versions.add(int(row["policy_version"]))
        except (KeyError, TypeError, ValueError):
            pass
        deterministic = row.get("deterministic")
        if isinstance(deterministic, bool):
            result["deterministic_true"] += int(deterministic)
            result["deterministic_false"] += int(not deterministic)
        state = row.get("state")
        if state is not None:
            try:
                state_dims.add(int(np.asarray(state).reshape(-1).shape[0]))
                result["state_present"] += 1
            except (TypeError, ValueError):
                pass
        epoch = _parse_ts(row.get("ts"))
        if epoch is not None:
            stamps.append(epoch)
        action = _action_list(row.get("action"))
        parsed.append(
            (epoch, np.asarray(action, dtype=np.float64) if action else None)
        )

    latency_array = np.asarray(latencies, dtype=np.float64)
    finite = _finite(latency_array)
    result["latency_ms"] = {
        "count": int(finite.size),
        "p50": _percentile(finite, 50.0),
        "p95": _percentile(finite, 95.0),
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
    }
    result["policy_versions"] = sorted(versions)
    result["state_dims"] = sorted(state_dims)
    if stamps:
        result["ts_first"] = min(stamps)
        result["ts_last"] = max(stamps)

    result["cross_check"] = _cross_check(parsed, action_rows)
    return result


def _cross_check(
    inference_rows: Sequence[tuple[float | None, np.ndarray | None]],
    action_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Nearest-ts join of inference lines onto actions.jsonl rows.

    Both sides are stamped inside the same server process, so the join is a
    within-host comparison and the |dt| distribution is a sanity check on the
    join itself.  An exact match means the executed action WAS the inferred
    action; a mismatch on an ``intervened`` row is the human, not a bug, so the
    non-intervened fraction is reported separately.
    """

    out: dict[str, Any] = {
        "usable_inference_rows": 0,
        "paired": 0,
        "exact_match": 0,
        "match_fraction": None,
        "paired_non_intervened": 0,
        "exact_match_non_intervened": 0,
        "match_fraction_non_intervened": None,
        "median_abs_dt_s": None,
        "max_abs_dt_s": None,
        "note": None,
    }

    catalogue: list[tuple[float, np.ndarray, bool]] = []
    for row in action_rows:
        epoch = _parse_ts(row.get("ts"))
        action = _action_list(row.get("actions"))
        if epoch is None or action is None or len(action) != ACTION_DIM:
            continue
        catalogue.append(
            (epoch, np.asarray(action, dtype=np.float64), bool(row.get("intervened", False)))
        )
    if not catalogue:
        out["note"] = "no actions.jsonl rows carried both a parsable ts and a 7-D action"
        return out
    catalogue.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in catalogue], dtype=np.float64)

    deltas: list[float] = []
    for epoch, action in inference_rows:
        if epoch is None or action is None or action.shape != (ACTION_DIM,):
            continue
        out["usable_inference_rows"] += 1
        index = int(np.searchsorted(times, epoch))
        candidates = [i for i in (index - 1, index) if 0 <= i < times.size]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(times[i] - epoch))
        _, executed, intervened = catalogue[best]
        deltas.append(abs(float(times[best] - epoch)))
        out["paired"] += 1
        matched = bool(
            np.allclose(
                np.asarray(action, dtype=np.float32),
                np.asarray(executed, dtype=np.float32),
                rtol=0.0,
                atol=ACTION_MATCH_ATOL,
            )
        )
        out["exact_match"] += int(matched)
        if not intervened:
            out["paired_non_intervened"] += 1
            out["exact_match_non_intervened"] += int(matched)

    if out["paired"]:
        out["match_fraction"] = out["exact_match"] / out["paired"]
    if out["paired_non_intervened"]:
        out["match_fraction_non_intervened"] = (
            out["exact_match_non_intervened"] / out["paired_non_intervened"]
        )
    if deltas:
        array = np.asarray(deltas, dtype=np.float64)
        out["median_abs_dt_s"] = float(np.median(array))
        out["max_abs_dt_s"] = float(array.max())
    return out


# ---------------------------------------------------------------------------
# (b) robot side -- vectors.h5 + status.jsonl
# ---------------------------------------------------------------------------


def _resolve_robot_paths(robot: Path) -> dict[str, Path | None]:
    """The recorder's session dir may be given directly or one level up.

    ``run_hil_*`` layouts nest the session under ``robot/``; a raw
    ``gello_logs/session_*`` directory does not.  Both are accepted.
    """

    return {
        "vectors": _first_existing(robot / "robot" / ROBOT_VECTORS, robot / ROBOT_VECTORS),
        "status": _first_existing(robot / "robot" / ROBOT_STATUS, robot / ROBOT_STATUS),
        "metadata": _first_existing(
            robot / "robot" / ROBOT_METADATA, robot / ROBOT_METADATA
        ),
    }


def _table_columns(group: Any) -> list[str]:
    """``Hdf5TableWriter`` stores the true column ORDER in ``attrs['columns']``."""

    raw = group.attrs.get("columns")
    if raw is not None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            columns = json.loads(raw)
            if isinstance(columns, list):
                return [str(c) for c in columns]
        except ValueError:
            pass
    return sorted(str(k) for k in group.keys())


def _column(group: Any, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    try:
        return np.asarray(group[name][()], dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001 - a corrupt dataset degrades to absent
        return None


def read_robot(robot: Path) -> dict[str, Any]:
    """Open ``vectors.h5`` read-only and pull out the columns the report needs.

    h5py is imported HERE and nowhere else so the server-only path has no h5py
    dependency at all.
    """

    paths = _resolve_robot_paths(robot)
    out: dict[str, Any] = {
        "root": str(robot),
        "vectors_path": str(paths["vectors"]) if paths["vectors"] else None,
        "status_path": str(paths["status"]) if paths["status"] else None,
        "metadata_path": str(paths["metadata"]) if paths["metadata"] else None,
        "error": None,
        "tables": {},
        "epoch0": None,
        "epoch0_source": None,
        "t_rel_min": None,
        "t_rel_max": None,
        "metadata": None,
        "series": {},
    }

    if paths["metadata"] is not None:
        meta = _read_json(paths["metadata"])
        out["metadata"] = meta.get("data")
        if meta.get("error"):
            out["metadata"] = {"_error": meta["error"]}

    if paths["vectors"] is None:
        out["error"] = f"no {ROBOT_VECTORS} under {robot} (looked in ./ and ./robot/)"
        return out

    try:
        import h5py  # noqa: PLC0415 - lazy on purpose; see docstring
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"h5py unavailable: {type(exc).__name__}: {exc}"
        return out

    try:
        with h5py.File(paths["vectors"], "r") as handle:
            for name in handle:
                group = handle[name]
                if not hasattr(group, "keys"):
                    continue
                columns = _table_columns(group)
                first = _column(group, columns[0]) if columns else None
                rows = int(first.size) if first is not None else 0
                t_rel = _column(group, "t_rel_s")
                entry = {
                    "columns": columns,
                    "rows": rows,
                    "t_rel_min": float(t_rel.min()) if t_rel is not None and t_rel.size else None,
                    "t_rel_max": float(t_rel.max()) if t_rel is not None and t_rel.size else None,
                }
                out["tables"][name] = entry

                # Only the columns the report actually consumes are held.
                if name == "ur_joint_states" and t_rel is not None:
                    speeds = [
                        _column(group, f"qd{i + 1}") for i in range(UR_JOINTS)
                    ]
                    speeds = [s for s in speeds if s is not None and s.size == t_rel.size]
                    if speeds:
                        out["series"]["ur_qd"] = (t_rel, np.stack(speeds, axis=1))
                elif name == "wrench" and t_rel is not None:
                    force = [_column(group, c) for c in ("fx", "fy", "fz")]
                    torque = [_column(group, c) for c in ("tx", "ty", "tz")]
                    if all(c is not None and c.size == t_rel.size for c in force):
                        out["series"]["wrench_force"] = (t_rel, np.stack(force, axis=1))
                    if all(c is not None and c.size == t_rel.size for c in torque):
                        out["series"]["wrench_torque"] = (t_rel, np.stack(torque, axis=1))
                elif name == "gripper" and t_rel is not None:
                    for column_name in ("grip_pos", "grip_cmd", "gello_grip"):
                        column = _column(group, column_name)
                        if column is not None and column.size == t_rel.size:
                            out["series"][f"gripper_{column_name}"] = (t_rel, column)
                elif name == "synchronized" and t_rel is not None:
                    t_wall = _column(group, "t_wall")
                    if t_wall is not None and t_wall.size == t_rel.size and t_rel.size:
                        # THE ANCHOR: t_rel_s is relative to RecordingSession.t0;
                        # t_wall is raw time.time().  Their difference IS t0.
                        offsets = t_wall - t_rel
                        offsets = offsets[np.isfinite(offsets)]
                        if offsets.size:
                            out["epoch0"] = float(np.median(offsets))
                            out["epoch0_source"] = "synchronized.t_wall - synchronized.t_rel_s"
    except Exception as exc:  # noqa: BLE001 - an unreadable h5 degrades, never crashes
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    spans = [
        (entry["t_rel_min"], entry["t_rel_max"])
        for entry in out["tables"].values()
        if entry["t_rel_min"] is not None
    ]
    if spans:
        out["t_rel_min"] = min(s[0] for s in spans)
        out["t_rel_max"] = max(s[1] for s in spans)

    if out["epoch0"] is None and isinstance(out.get("metadata"), Mapping):
        start_wall = _parse_ts(out["metadata"].get("start_wall"))
        if start_wall is not None:
            out["epoch0"] = start_wall
            out["epoch0_source"] = (
                "recorder metadata.json start_wall (naive local time; assumes this "
                "reader's timezone matches the recorder's)"
            )
    return out


def robot_window_stats(
    robot: Mapping[str, Any], start_epoch: float | None, end_epoch: float | None
) -> dict[str, Any]:
    """Slice every consumed series to one episode's wall-clock window."""

    out: dict[str, Any] = {
        "aligned": False,
        "t_rel_start": None,
        "t_rel_end": None,
        "ur_rows": 0,
        "ur_speed_max_abs": None,
        "ur_speed_rms": None,
        "wrench_rows": 0,
        "wrench_force_max": None,
        "wrench_torque_max": None,
        "gripper_source": None,
        "gripper_travel": None,
        "gripper_range": None,
    }
    epoch0 = robot.get("epoch0")
    if epoch0 is None or start_epoch is None or end_epoch is None:
        return out
    lo = start_epoch - epoch0
    hi = end_epoch - epoch0
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
        return out
    out["aligned"] = True
    out["t_rel_start"] = lo
    out["t_rel_end"] = hi

    series = robot.get("series") or {}

    pair = series.get("ur_qd")
    if pair is not None:
        t_rel, values = pair
        mask = (t_rel >= lo) & (t_rel <= hi)
        window = values[mask]
        finite = window[np.isfinite(window)]
        out["ur_rows"] = int(mask.sum())
        if finite.size:
            out["ur_speed_max_abs"] = float(np.abs(finite).max())
            out["ur_speed_rms"] = float(np.sqrt(np.mean(np.square(finite))))

    for key, target in (("wrench_force", "wrench_force_max"),
                        ("wrench_torque", "wrench_torque_max")):
        pair = series.get(key)
        if pair is None:
            continue
        t_rel, values = pair
        mask = (t_rel >= lo) & (t_rel <= hi)
        window = values[mask]
        if key == "wrench_force":
            out["wrench_rows"] = int(mask.sum())
        if window.size:
            magnitude = np.linalg.norm(np.nan_to_num(window, nan=0.0), axis=1)
            if magnitude.size:
                out[target] = float(magnitude.max())

    for column_name in ("grip_pos", "grip_cmd", "gello_grip"):
        pair = series.get(f"gripper_{column_name}")
        if pair is None:
            continue
        t_rel, values = pair
        mask = (t_rel >= lo) & (t_rel <= hi)
        window = values[mask]
        finite = window[np.isfinite(window)]
        if finite.size < 2:
            continue
        out["gripper_source"] = column_name
        out["gripper_travel"] = float(np.abs(np.diff(finite)).sum())
        out["gripper_range"] = float(finite.max() - finite.min())
        break

    return out


def summarize_status(status: Mapping[str, Any], *, max_entries: int = 200) -> dict[str, Any]:
    """Compress ``status.jsonl`` into a per-topic state-CHANGE timeline."""

    out: dict[str, Any] = {
        "present": bool(status.get("present")),
        "path": status.get("path"),
        "error": status.get("error"),
        "bad_lines": int(status.get("bad_lines", 0)),
        "lines": 0,
        "topics": [],
        "timeline": [],
        "timeline_truncated": False,
    }
    if not out["present"]:
        return out
    rows = list(status.get("rows") or [])
    out["lines"] = len(rows)

    last: dict[str, str] = {}
    timeline: list[dict[str, Any]] = []
    topics: set[str] = set()
    for row in rows:
        topic = str(row.get("topic", ""))
        topics.add(topic)
        parsed = row.get("parsed")
        value = row.get("raw") if parsed is None else parsed
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
        if last.get(topic) == text:
            continue
        previous = last.get(topic)
        last[topic] = text
        timeline.append(
            {
                "ts": row.get("ts"),
                "epoch": _parse_ts(row.get("ts")),
                "topic": topic,
                "from": previous,
                "to": text,
            }
        )
    out["topics"] = sorted(topics)
    if len(timeline) > max_entries:
        out["timeline_truncated"] = True
        out["timeline"] = timeline[:max_entries]
    else:
        out["timeline"] = timeline
    return out


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    served = Path(args.served).expanduser().resolve()

    metadata = _read_json(served / SERVED_METADATA)
    actions = _read_jsonl(served / SERVED_ACTIONS)
    inference_raw = _read_jsonl(served / SERVED_INFERENCE)

    action_summary = summarize_actions(actions["rows"])

    deep_dive: tuple[str, str] | None = None
    if args.episode is not None:
        deep_dive = ("", str(args.episode))

    pickles = scan_served_pickles(served, deep_dive=deep_dive)
    inference = summarize_inference(inference_raw, actions["rows"])

    robot: dict[str, Any] | None = None
    status: dict[str, Any] | None = None
    if args.robot:
        robot_root = Path(args.robot).expanduser().resolve()
        robot = read_robot(robot_root)
        status_path = robot.get("status_path")
        status = summarize_status(
            _read_jsonl(Path(status_path)) if status_path else {"present": False,
                                                               "path": None}
        )

    # ---- merge per-episode views -------------------------------------------
    keys = sorted(
        set(action_summary["episodes"]) | set(pickles["episodes"]),
        key=lambda k: (k[0], _sort_key(k[1])),
    )
    episodes: list[dict[str, Any]] = []
    for key in keys:
        from_actions = action_summary["episodes"].get(key, {})
        from_pickle = pickles["episodes"].get(key, {})
        # Wall duration: the pickle's meta.timestamp_ns is the ACTOR's clock and
        # is the only per-step stamp taken on the robot host; actions.jsonl ts is
        # the sink's (server) clock and is the fallback.
        if from_pickle.get("duration_s") is not None:
            duration = from_pickle["duration_s"]
            duration_source = "pickle meta.timestamp_ns"
        elif from_actions.get("ts_duration_s") is not None:
            duration = from_actions["ts_duration_s"]
            duration_source = "actions.jsonl ts"
        else:
            duration = None
            duration_source = None

        start_epoch = end_epoch = None
        epoch_source = None
        if from_pickle.get("timestamp_ns_first") is not None:
            start_epoch = from_pickle["timestamp_ns_first"] / 1e9
            end_epoch = from_pickle["timestamp_ns_last"] / 1e9
            epoch_source = "pickle meta.timestamp_ns (actor/laptop3 clock)"
        elif from_actions.get("ts_first") is not None:
            start_epoch = from_actions["ts_first"]
            end_epoch = from_actions["ts_last"]
            epoch_source = "actions.jsonl ts (sink/server clock)"

        entry = {
            "run_id": key[0],
            "episode_id": key[1],
            "steps": from_actions.get("steps", 0),
            "pickle_items": from_pickle.get("items", 0),
            "pickle_present": bool(from_pickle),
            "pickle_error": from_pickle.get("error"),
            "outcome": from_actions.get("outcome", "unknown (no actions.jsonl rows)"),
            "intervened_steps": from_actions.get("intervened_steps", 0),
            "reward_sum": from_actions.get("reward_sum"),
            "trans_norm_mean": from_actions.get("trans_norm_mean"),
            "trans_norm_max": from_actions.get("trans_norm_max"),
            "rot_norm_mean": from_actions.get("rot_norm_mean"),
            "rot_norm_max": from_actions.get("rot_norm_max"),
            "gripper_changes": from_actions.get("gripper_changes"),
            "gripper_state_flips": from_actions.get("gripper_state_flips"),
            "duration_s": duration,
            "duration_source": duration_source,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "epoch_source": epoch_source,
            "policy_versions": from_pickle.get("policy_versions", []),
            "observation_spec": from_pickle.get("observation_spec"),
            "synthetic_policy_actions": from_pickle.get("synthetic_policy_actions", False),
            "env_step_min": from_actions.get("env_step_min"),
            "env_step_max": from_actions.get("env_step_max"),
            "robot_window": None,
        }
        if robot is not None and not robot.get("error"):
            entry["robot_window"] = robot_window_stats(robot, start_epoch, end_epoch)
        episodes.append(entry)

    deep_dive_rows = None
    deep_dive_note = None
    if args.episode is not None:
        matches = [
            summary
            for key, summary in pickles["episodes"].items()
            if summary.get("deep_dive_rows") is not None
        ]
        if matches:
            deep_dive_rows = matches[0]["deep_dive_rows"]
        else:
            deep_dive_note = (
                f"no episode pickle matched --episode {args.episode}; a deep dive "
                "needs the pickle (actions.jsonl has no per-step policy action)"
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "served_dir": str(served),
        "out_dir": None,
        "metadata": {
            "present": metadata["present"],
            "error": metadata["error"],
            "data": metadata["data"],
        },
        "actions_log": {
            "present": actions["present"],
            "path": actions["path"],
            "error": actions["error"],
            "rows": len(actions["rows"]),
            "bad_lines": actions["bad_lines"],
            "bad_line_numbers": actions["bad_line_numbers"],
            "unusable_rows": action_summary["skipped_rows"],
        },
        "totals": _totals(episodes, action_summary["totals"]),
        "episodes": episodes,
        "inference": inference,
        "robot": _robot_report_view(robot),
        "status": status,
        "deep_dive": {
            "episode": None if args.episode is None else str(args.episode),
            "rows": deep_dive_rows,
            "note": deep_dive_note,
            "max_rows": args.deep_dive_rows,
        },
    }
    return report


def _totals(
    episodes: Sequence[Mapping[str, Any]], action_totals: Mapping[str, Any]
) -> dict[str, Any]:
    """Run-level tallies.

    ``episodes_successful`` / ``success_rate`` are the headline numbers of a BC
    evaluation, so they are stated once, explicitly, rather than left for a
    reader to recount from the per-episode table.  ``success_rate`` counts an
    ``open`` episode (no terminal row ever logged) in the denominator: dropping
    it would quietly flatter a run that died mid-episode.
    """

    outcomes = [entry.get("outcome") for entry in episodes]
    successful = sum(1 for outcome in outcomes if outcome == "success")
    return {
        "episodes": len(episodes),
        "episodes_with_pickle": sum(1 for e in episodes if e["pickle_present"]),
        "episodes_successful": successful,
        "episodes_truncated": sum(1 for o in outcomes if o == "truncated"),
        "episodes_done": sum(1 for o in outcomes if o == "done"),
        "episodes_open": sum(1 for o in outcomes if o == "open"),
        "success_rate": (successful / len(episodes)) if episodes else None,
        "transitions": action_totals["rows"],
        "interventions": action_totals["intervened"],
        "reward_sum": action_totals["reward_sum"],
        "runs": sorted({e["run_id"] for e in episodes}),
    }


def _robot_report_view(robot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Strip the raw numpy series before the report is serialised."""

    if robot is None:
        return None
    view = {k: v for k, v in robot.items() if k != "series"}
    view["series_available"] = sorted((robot.get("series") or {}).keys())
    return view


def _sort_key(episode_id: str) -> tuple[int, Any]:
    try:
        return (0, int(episode_id))
    except (TypeError, ValueError):
        return (1, str(episode_id))


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: Mapping[str, Any], args: argparse.Namespace) -> str:
    lines: list[str] = []
    add = lines.append

    # ---- 1. header ---------------------------------------------------------
    metadata = report["metadata"]
    data = metadata.get("data") or {}
    totals = report["totals"]
    add("# BC rollout report")
    add("")
    add(f"generated `{report['generated_at_utc']}` by `analyze_bc_rollout.py`")
    add("")
    if not metadata["present"]:
        add(f"{NOT_RECORDED} — no `{SERVED_METADATA}` in the served dir; "
            "model identity is unknown.")
        add("")
    elif metadata["error"]:
        add(f"{NOT_RECORDED} — `{SERVED_METADATA}` unreadable: {metadata['error']}")
        add("")

    header_rows = [
        ("model_id", data.get("model_id") or "n/a"),
        ("artifact sha256 (short)", _short_sha(data.get("artifact_sha256"))),
        ("record root created", data.get("created_at_utc") or "n/a"),
        ("served dir", f"`{report['served_dir']}`"),
        ("run_id(s)", ", ".join(totals["runs"]) or "n/a"),
        ("episodes", totals["episodes"]),
        ("episodes with a pickle", totals["episodes_with_pickle"]),
        (
            "**successful episodes**",
            f"**{totals['episodes_successful']} / {totals['episodes']}**"
            f" (success rate {_num(totals['success_rate'], 3)})",
        ),
        (
            "other outcomes",
            f"truncated {totals['episodes_truncated']}, "
            f"done-without-success {totals['episodes_done']}, "
            f"open {totals['episodes_open']}",
        ),
        ("transitions", totals["transitions"]),
        ("intervention steps", totals["interventions"]),
        ("reward sum", _num(totals["reward_sum"])),
    ]
    lines.extend(_md_table(["field", "value"], header_rows))
    add("")

    actions_log = report["actions_log"]
    if not actions_log["present"]:
        add(f"{NOT_RECORDED} — no `{SERVED_ACTIONS}`; every per-step statistic "
            "below is unavailable.")
        add("")
    else:
        notes = []
        if actions_log["bad_lines"]:
            numbers = ", ".join(str(n) for n in actions_log["bad_line_numbers"])
            notes.append(
                f"{actions_log['bad_lines']} malformed line(s) skipped "
                f"(line numbers: {numbers or 'n/a'})"
            )
        if actions_log["unusable_rows"]:
            notes.append(
                f"{actions_log['unusable_rows']} row(s) skipped for a bad "
                "action/step_id field"
            )
        if actions_log["error"]:
            notes.append(f"read error: {actions_log['error']}")
        if notes:
            add("> parse warnings: " + "; ".join(notes))
            add("")

    synthetic = [e for e in report["episodes"] if e.get("synthetic_policy_actions")]
    if synthetic:
        add(f"> ⚠️ {len(synthetic)} episode(s) carry "
            "`policy_actions_synthetic=true` — those actions were rewritten "
            "before execution and are NOT policy output.")
        add("")

    # ---- 2. per-episode table ---------------------------------------------
    add("## 1. Episodes")
    add("")
    if not report["episodes"]:
        add(f"{NOT_RECORDED} — no episodes found in `{SERVED_ACTIONS}` or as "
            "`<run_id>/episode_*.pkl`.")
        add("")
    else:
        multi_run = len(totals["runs"]) > 1
        rows = []
        for entry in report["episodes"]:
            label = (
                f"{entry['run_id']}/{entry['episode_id']}"
                if multi_run
                else entry["episode_id"]
            )
            duration = _num(entry["duration_s"], 2)
            if entry["duration_source"] == "actions.jsonl ts":
                duration += " *"
            rows.append(
                [
                    label,
                    entry["steps"],
                    duration,
                    entry["outcome"],
                    entry["intervened_steps"],
                    f"{_num(entry['trans_norm_mean'])} / {_num(entry['trans_norm_max'])}",
                    f"{_num(entry['rot_norm_mean'])} / {_num(entry['rot_norm_max'])}",
                    entry["gripper_changes"],
                ]
            )
        lines.extend(
            _md_table(
                [
                    "episode", "steps", "wall s", "outcome", "interv",
                    "‖a[0:3]‖ mean/max", "‖a[3:6]‖ mean/max", "grip Δ",
                ],
                rows,
            )
        )
        add("")
        add("Wall duration comes from the episode pickle's `meta.timestamp_ns` "
            "(stamped by the actor on laptop3). A `*` marks a fallback to "
            "`actions.jsonl` `ts`, which is the SERVER's clock.")
        add("")
        add(f"`grip Δ` counts consecutive steps whose gripper command `a[6]` "
            f"moved by more than {GRIPPER_CHANGE_EPS}; `outcome` follows the "
            "actor's precedence success > truncated > done, and `open` means no "
            "terminal row was ever logged for that episode.")
        add("")

        missing = [e for e in report["episodes"] if not e["pickle_present"]]
        if missing:
            add(f"> {NOT_RECORDED} — {len(missing)} episode(s) have no pickle: "
                + ", ".join(str(e["episode_id"]) for e in missing[:20])
                + (" …" if len(missing) > 20 else "")
                + ". The sink only writes a pickle when the terminal transition "
                "arrives, so a run that died mid-episode loses it (the "
                "`actions.jsonl` rows survive).")
            add("")
        broken = [e for e in report["episodes"] if e["pickle_error"]]
        for entry in broken:
            add(f"> pickle for episode {entry['episode_id']} did not load: "
                f"{entry['pickle_error']}")
        if broken:
            add("")

        spec_entry = next(
            (e for e in report["episodes"] if e.get("observation_spec")), None
        )
        if spec_entry is not None:
            add("### Observation spec (shapes only — pixels are never loaded into "
                "this report)")
            add("")
            lines.extend(
                _md_table(
                    ["key", "shape", "dtype"],
                    [
                        [key, "×".join(str(d) for d in value["shape"]), value["dtype"]]
                        for key, value in sorted(spec_entry["observation_spec"].items())
                    ],
                )
            )
            add("")
            add(f"(from the first transition of episode {spec_entry['episode_id']})")
            add("")

    # ---- 3. inference ------------------------------------------------------
    add("## 2. Inference log")
    add("")
    inference = report["inference"]
    if not inference["present"]:
        add(f"{NOT_RECORDED} — no `{SERVED_INFERENCE}` in the served dir "
            "(the BC server was started with `--no-inference-log`, or this "
            "rollout predates the inference logger).")
        add("")
    else:
        latency = inference["latency_ms"]
        rows = [
            ("lines", inference["lines"]),
            ("malformed lines skipped", inference["bad_lines"]),
            ("latency samples", latency["count"]),
            ("latency p50 (ms)", _num(latency["p50"], 2)),
            ("latency p95 (ms)", _num(latency["p95"], 2)),
            ("latency max (ms)", _num(latency["max"], 2)),
            ("latency mean (ms)", _num(latency["mean"], 2)),
            ("policy_version set",
             ", ".join(str(v) for v in inference["policy_versions"]) or "n/a"),
            ("deterministic true/false",
             f"{inference['deterministic_true']} / {inference['deterministic_false']}"),
            ("lines carrying state",
             f"{inference['state_present']} "
             f"(dims: {inference['state_dims'] or 'n/a'})"),
        ]
        lines.extend(_md_table(["field", "value"], rows))
        add("")

        cross = inference["cross_check"] or {}
        add("### Inferred vs executed action")
        add("")
        if cross.get("note"):
            add(f"{NOT_RECORDED} — {cross['note']}")
            add("")
        else:
            rows = [
                ("usable inference rows", cross.get("usable_inference_rows")),
                ("paired to an actions.jsonl row", cross.get("paired")),
                ("exact action match", cross.get("exact_match")),
                ("match fraction", _num(cross.get("match_fraction"), 4)),
                ("match fraction, non-intervened only",
                 _num(cross.get("match_fraction_non_intervened"), 4)),
                ("non-intervened pairs", cross.get("paired_non_intervened")),
                ("join |Δt| median (s)", _num(cross.get("median_abs_dt_s"), 4)),
                ("join |Δt| max (s)", _num(cross.get("max_abs_dt_s"), 4)),
            ]
            lines.extend(_md_table(["field", "value"], rows))
            add("")
            add("Join is nearest-timestamp; both sides are stamped in the same "
                "server process, so `|Δt|` measures the join, not clock skew. "
                "A mismatch on an `intervened` row is the human overriding the "
                "policy — expected, which is why the non-intervened fraction is "
                "reported separately. Per-row diffs are deliberately omitted.")
            add("")

    # ---- 4. robot side -----------------------------------------------------
    add("## 3. Robot side (laptop3 recorder)")
    add("")
    robot = report["robot"]
    if robot is None:
        add(f"{NOT_RECORDED} — `--robot` was not given, so no `vectors.h5` was "
            "read. Server-side numbers above are unaffected.")
        add("")
    elif robot.get("error"):
        add(f"{NOT_RECORDED} — robot recording unusable: {robot['error']}")
        add("")
    else:
        add("### Time base and alignment assumption")
        add("")
        add("Every `vectors.h5` table stamps `t_rel_s = time.time() - t0`, where "
            "`t0` is fixed when `RecordingSession` is constructed — a RELATIVE "
            "clock with no absolute meaning on its own. This report recovers the "
            "epoch origin as follows:")
        add("")
        add(f"- **epoch origin source**: {robot.get('epoch0_source') or 'NONE FOUND'}")
        add(f"- **epoch origin**: {_epoch_text(robot.get('epoch0'))}")
        add("")
        if robot.get("epoch0") is None:
            add(f"{NOT_RECORDED} — neither a `synchronized` table with `t_wall` "
                "nor a recorder `metadata.json` `start_wall` was found, so no "
                "episode window can be aligned. Table spans below are in "
                "`t_rel_s` only.")
            add("")
        else:
            add("**Assumption, stated so it can be challenged:** the episode "
                "windows below are aligned by comparing that epoch origin to the "
                "episode pickles' `meta.timestamp_ns`, which is `time.time_ns()` "
                "taken in the ACTOR process on laptop3 — the same host and the "
                "same epoch as the recorder, so no cross-host skew enters. When "
                "an episode has no pickle the fallback is `actions.jsonl` `ts`, "
                "stamped in the BC SERVER process; if that server ran on another "
                "host, the skew between the two machines is uncorrected and "
                "unbounded, and those windows should be read as approximate.")
            add("")

        span_rows = []
        for name, entry in sorted(robot["tables"].items()):
            span_rows.append(
                [
                    name,
                    entry["rows"],
                    _num(entry["t_rel_min"], 2),
                    _num(entry["t_rel_max"], 2),
                    len(entry["columns"]),
                ]
            )
        add("### Recording span")
        add("")
        lines.extend(
            _md_table(
                ["table", "rows", "t_rel_s min", "t_rel_s max", "columns"], span_rows
            )
        )
        add("")
        if robot.get("t_rel_min") is not None:
            add(f"Overall span: `t_rel_s` "
                f"{_num(robot['t_rel_min'], 2)} → {_num(robot['t_rel_max'], 2)} "
                f"({_num((robot['t_rel_max'] - robot['t_rel_min']), 2)} s)")
            if robot.get("epoch0") is not None:
                add("")
                add(f"In wall clock: {_epoch_text(robot['epoch0'] + robot['t_rel_min'])}"
                    f" → {_epoch_text(robot['epoch0'] + robot['t_rel_max'])}")
            add("")
        recorder_meta = robot.get("metadata")
        if isinstance(recorder_meta, Mapping) and "_error" not in recorder_meta:
            add(f"Recorder `metadata.json`: duration_s="
                f"{recorder_meta.get('duration_s', 'n/a')}, "
                f"finalized={recorder_meta.get('finalized', 'n/a')}, "
                f"sample_rate_hz={recorder_meta.get('sample_rate_hz', 'n/a')}")
            add("")

        add("### Per-episode robot-side window")
        add("")
        windows = [
            e for e in report["episodes"]
            if e.get("robot_window") and e["robot_window"].get("aligned")
        ]
        if not windows:
            add(f"{NOT_RECORDED} — no episode could be aligned onto the robot "
                "recording (missing epoch origin, or no episode carried a "
                "parsable timestamp).")
            add("")
        else:
            rows = []
            for entry in windows:
                window = entry["robot_window"]
                rows.append(
                    [
                        entry["episode_id"],
                        f"{_num(window['t_rel_start'], 2)}–{_num(window['t_rel_end'], 2)}",
                        window["ur_rows"],
                        _num(window["ur_speed_max_abs"], 4),
                        _num(window["ur_speed_rms"], 4),
                        _num(window["wrench_force_max"], 3),
                        _num(window["wrench_torque_max"], 3),
                        f"{_num(window['gripper_travel'], 2)}"
                        + (f" ({window['gripper_source']})"
                           if window["gripper_source"] else ""),
                    ]
                )
            lines.extend(
                _md_table(
                    [
                        "episode", "t_rel_s window", "ur rows",
                        "max |qd| rad/s", "RMS qd rad/s",
                        "max |F| N", "max |T| Nm", "gripper travel",
                    ],
                    rows,
                )
            )
            add("")
            add("`max |qd|` is the largest single-joint speed magnitude in the "
                "window; `RMS qd` is the root-mean-square over all six joints and "
                "all rows in the window (both from `ur_joint_states.qd1..qd6`). "
                "`max |F|`/`max |T|` are the largest force and torque vector "
                "magnitudes from the `wrench` table. `gripper travel` is the "
                "summed absolute step-to-step change of the named `gripper` "
                "column (`grip_pos` preferred, falling back to `grip_cmd` then "
                "`gello_grip`).")
            add("")

        unaligned = [
            e for e in report["episodes"]
            if e.get("robot_window") and not e["robot_window"].get("aligned")
        ]
        if unaligned:
            add(f"> {len(unaligned)} episode(s) had no alignable window: "
                + ", ".join(str(e["episode_id"]) for e in unaligned[:20])
                + (" …" if len(unaligned) > 20 else ""))
            add("")

    # ---- 4b. status timeline ----------------------------------------------
    add("## 4. Status timeline")
    add("")
    status = report["status"]
    if status is None:
        add(f"{NOT_RECORDED} — `--robot` was not given.")
        add("")
    elif not status["present"]:
        add(f"{NOT_RECORDED} — no `{ROBOT_STATUS}` beside the robot recording.")
        add("")
    else:
        add(f"{status['lines']} line(s), {status['bad_lines']} malformed skipped, "
            f"topics: {', '.join(status['topics']) or 'n/a'}")
        add("")
        if not status["timeline"]:
            add(f"{NOT_RECORDED} — no state CHANGES were observed (the log holds "
                "only repeats of one value per topic).")
            add("")
        else:
            lines.extend(
                _md_table(
                    ["ts", "topic", "from", "to"],
                    [
                        [
                            entry["ts"],
                            entry["topic"],
                            "—" if entry["from"] is None else f"`{entry['from']}`",
                            f"`{entry['to']}`",
                        ]
                        for entry in status["timeline"]
                    ],
                )
            )
            add("")
            if status["timeline_truncated"]:
                add("> timeline truncated; only state CHANGES are listed, repeats "
                    "are collapsed.")
                add("")

    # ---- 5. deep dive ------------------------------------------------------
    deep = report["deep_dive"]
    if deep["episode"] is not None:
        add(f"## 5. Deep dive — episode {deep['episode']}")
        add("")
        if deep["note"]:
            add(f"{NOT_RECORDED} — {deep['note']}")
            add("")
        elif not deep["rows"]:
            add(f"{NOT_RECORDED} — the pickle for episode {deep['episode']} held "
                "no readable transitions.")
            add("")
        else:
            rows = deep["rows"][: max(1, int(deep["max_rows"]))]
            table = []
            for row in rows:
                action = row["actions"] or [None] * ACTION_DIM
                table.append(
                    [
                        row["step_id"],
                        row["env_step"],
                        *[_num(v, 4) for v in action[:ACTION_DIM]],
                        "yes" if row["intervened"] else "",
                        _num(row["rewards"], 3),
                        _num(row["masks"], 1),
                        "D" if row["dones"] else ("T" if row["truncated"] else ""),
                    ]
                )
            lines.extend(
                _md_table(
                    [
                        "step", "env_step", "dx", "dy", "dz", "drx", "dry", "drz",
                        "grip", "interv", "reward", "mask", "term",
                    ],
                    table,
                )
            )
            add("")
            add(f"Showing {len(rows)} of {len(deep['rows'])} transitions "
                f"(cap `--deep-dive-rows {deep['max_rows']}`). Columns are the "
                "EXECUTED action `transition.actions`; `meta.policy_action` (the "
                "action requested before any human override) is preserved in "
                "`report.json` under `deep_dive.rows[].policy_action`.")
            add("")
            if len(deep["rows"]) > len(rows):
                add("> deep-dive table truncated.")
                add("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a BC rollout's server-side record root and (optionally) the "
            "laptop3 robot-side recording into one report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  analyze_bc_rollout.py --served ~/bc_rollouts/run_2200\n"
            "  analyze_bc_rollout.py --served ~/bc_rollouts/run_2200 "
            "--robot ros2_ur_ws/gello_logs/session_20260731_220145\n"
            "  analyze_bc_rollout.py --served ~/bc_rollouts/run_2200 --episode 3\n"
        ),
    )
    parser.add_argument(
        "--served",
        required=True,
        help="record root written by ur_env/bc_recording_sink.py",
    )
    parser.add_argument(
        "--robot",
        default=None,
        help=(
            "laptop3 recorder session dir; vectors.h5/status.jsonl are looked up "
            "both directly under it and under its robot/ subdirectory"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: <served>/analysis)",
    )
    parser.add_argument(
        "--episode",
        default=None,
        help="episode id to deep-dive (per-step action components into the md)",
    )
    parser.add_argument(
        "--deep-dive-rows",
        type=int,
        default=200,
        help="cap on deep-dive table rows in the markdown (default: 200)",
    )
    parser.add_argument(
        "--print",
        dest="print_md",
        action="store_true",
        help="also write the markdown report to stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    served = Path(args.served).expanduser()
    if not served.is_dir():
        print(f"error: --served is not a directory: {served}", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.out).expanduser()
        if args.out
        else served.resolve() / "analysis"
    )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create --out {out_dir}: {exc}", file=sys.stderr)
        return 2

    try:
        report = build_report(args)
    except Exception:  # noqa: BLE001 - the analyzer must never die on bad input
        traceback.print_exc()
        print(
            "error: report assembly failed; nothing was written", file=sys.stderr
        )
        return 1
    report["out_dir"] = str(out_dir)

    markdown = render_markdown(report, args)

    json_path = out_dir / "report.json"
    md_path = out_dir / "rollout_report.md"
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")
    with open(md_path, "w", encoding="utf-8") as stream:
        stream.write(markdown)

    if args.print_md:
        sys.stdout.write(markdown)
    print(f"wrote {json_path}", file=sys.stderr)
    print(f"wrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
