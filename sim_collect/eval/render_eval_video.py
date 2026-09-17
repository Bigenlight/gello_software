#!/usr/bin/env python
"""Render 30 Hz IFQL evaluation diagnostics from an existing eval run.

The renderer never runs a policy or re-simulates physics.  It reads
``episodes.jsonl`` and ``ep_<episode>.h5``.  When ``--policy-log-dir`` is
provided, the matching trusted-local ``ep_<reset_counter:04d>.npz`` supplies
the exact JPEG observations and per-request chunk indices; otherwise cam1 and
cam2 are rendered kinematically from the recorded H5 state.

Policy logs contain object arrays, so loading them requires
``numpy.load(..., allow_pickle=True)``.  Only use a log directory produced
locally by a trusted IFQL server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

FPS = 30
CAM_SIZE = (640, 360)
DASHBOARD_SIZE = (CAM_SIZE[0] * 2, 360)
# The two policy cameras stay pixel-for-pixel 640x360.  Diagnostics live in a
# separate lower dashboard rather than shrinking either observation.
OUTPUT_SIZE = (CAM_SIZE[0] * 2, CAM_SIZE[1] + DASHBOARD_SIZE[1])

OUTCOME_BGR = {
    "success": (70, 210, 80),
    "timeout": (70, 90, 235),
    "failure": (60, 60, 230),
    "fault": (0, 170, 255),
}
Q_FIELDS = ("K", "q_chosen", "q_mean", "q_std", "q_min", "q_max", "q_spread", "q_argmax")
Q_HEADS_FIELD = "q_heads"
LATENCY_FIELDS = ("encode_ms", "sample_ms", "refill_ms")


class JoinError(ValueError):
    """Policy log metadata cannot be joined without guessing or shifting indices."""


@dataclass(frozen=True)
class EpisodeMeta:
    episode: str
    seed: Optional[int]
    outcome: str
    n_steps: int
    detail: str
    failure: Optional[str]
    fault: Optional[str]
    policy: Mapping[str, Any]
    policy_reset: Mapping[str, Any]


@dataclass(frozen=True)
class Diagnostic:
    request_idx: int
    decision_idx: int
    chunk_id: int
    chunk_step: int
    boundary: bool
    q: Optional[Mapping[str, Any]] = None
    latency: Optional[Mapping[str, float]] = None
    sidecar_refill: Optional[int] = None
    sidecar_updated: bool = False


@dataclass
class PolicyEpisode:
    path: Path
    request_idx: np.ndarray
    cam1_jpeg: np.ndarray
    cam2_jpeg: np.ndarray
    chunk_id: np.ndarray
    chunk_step: np.ndarray
    chunk_t: np.ndarray
    state: Optional[np.ndarray]
    action: Optional[np.ndarray]
    diagnostics: List[Diagnostic]
    notes: List[str] = field(default_factory=list)


@dataclass
class H5Episode:
    path: Path
    xml: str
    config: Mapping[str, Any]
    layout: Mapping[str, Any]
    t_rel_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    frame_rows: np.ndarray


def _strict_int(value: Any, name: str, *, minimum: Optional[int] = None,
                maximum: Optional[int] = None, line: Any = "?") -> int:
    """Parse a lossless schema integer.

    Accepted values are Python/JSON integers and NumPy integer scalars only.
    Booleans, floats (including ``1.0``), and numeric strings are rejected so
    indices can never be silently truncated or coerced.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise JoinError(
            f"refill_stats line {line}: {name} must be an integer token "
            f"(not bool, float, or string), got {value!r}"
        )
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise JoinError(f"refill_stats line {line}: {name}={parsed} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise JoinError(f"refill_stats line {line}: {name}={parsed} must be <= {maximum}")
    return parsed


_SIDECAR_DISCRETE_MINIMUMS = {
    "schema_version": 1,
    "reset_counter": 1,
    "request_idx": 0,
    "t_index": 0,
    "decision_idx": 0,
    "refill": 1,
    "K": 1,
    "q_argmax": 0,
}


def _validate_discrete_sidecar_fields(row: Mapping[str, Any], *, require_reset: bool = False) -> Dict[str, Any]:
    """Return a copy with every present discrete sidecar field strictly validated."""
    out = dict(row)
    line = out.get("_line", "?")
    if require_reset and "reset_counter" not in out:
        raise JoinError(f"refill_stats line {line}: missing reset_counter")
    for name, minimum in _SIDECAR_DISCRETE_MINIMUMS.items():
        if name in out and out[name] is not None:
            out[name] = _strict_int(out[name], name, minimum=minimum, line=line)
    if out.get("q_argmax") is not None and out.get("K") is not None and out["q_argmax"] >= out["K"]:
        raise JoinError(
            f"refill_stats line {line}: q_argmax={out['q_argmax']} outside K={out['K']} candidates"
        )
    return out


def load_episodes_jsonl(path_or_run_dir: os.PathLike[str] | str) -> Dict[str, EpisodeMeta]:
    path = Path(path_or_run_dir)
    if path.is_dir():
        path = path / "episodes.jsonl"
    episodes: Dict[str, EpisodeMeta] = {}
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                episode = str(row["episode"])
                n_steps = _strict_int(row["n_steps"], "n_steps", minimum=1, line=lineno)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid episode row: {exc}") from exc
            if episode in episodes:
                raise ValueError(f"{path}:{lineno}: duplicate episode {episode!r}")
            if n_steps <= 0:
                raise ValueError(f"{path}:{lineno}: n_steps must be positive, got {n_steps}")
            policy = row.get("policy") if isinstance(row.get("policy"), dict) else {}
            reset = row.get("policy_reset") if isinstance(row.get("policy_reset"), dict) else {}
            if not reset and isinstance(policy.get("reset_reply"), dict):
                reset = policy["reset_reply"]
            seed = row.get("seed")
            episodes[episode] = EpisodeMeta(
                episode=episode,
                seed=None if seed is None else _strict_int(seed, "seed", minimum=0, line=lineno),
                outcome=str(row.get("outcome") or "unknown"),
                n_steps=n_steps,
                detail=str(row.get("detail") or ""),
                failure=row.get("failure"),
                fault=row.get("fault"),
                policy=policy,
                policy_reset=reset,
            )
    if not episodes:
        raise ValueError(f"{path}: no episode rows")
    return episodes


def _as_int_vector(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise JoinError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.dtype.kind not in "iu":
        raise JoinError(f"{name} must use an integer dtype, got {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def _policy_reset_counter(meta: EpisodeMeta, override: Optional[int]) -> int:
    recorded = meta.policy_reset.get("reset_counter")
    if override is not None:
        return _strict_int(override, "reset_counter override", minimum=1, line=f"episode {meta.episode}")
    if recorded is None:
        raise JoinError(
            f"episode {meta.episode}: no reset_counter in episodes.jsonl; pass an explicit "
            "--reset-map EPISODE:COUNTER (indices are never shifted or inferred)"
        )
    return _strict_int(recorded, "reset_counter", minimum=1, line=f"episode {meta.episode}")


def _load_refill_rows(path: Path, reset_counter: int) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JoinError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            row["_line"] = lineno
            row = _validate_discrete_sidecar_fields(row, require_reset=True)
            if row["reset_counter"] == reset_counter:
                rows.append(row)
    return rows


def _validate_policy_indices(
    request_idx: np.ndarray,
    chunk_t: np.ndarray,
    chunk_id: np.ndarray,
    chunk_step: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    n = len(request_idx)
    if n == 0:
        raise JoinError("policy NPZ has zero requests")
    if len(chunk_id) != n or len(chunk_step) != n:
        raise JoinError(
            f"per-request lengths differ: request_idx={n}, chunk_id={len(chunk_id)}, chunk_step={len(chunk_step)}"
        )
    if request_idx[0] != 0 or not np.array_equal(request_idx, np.arange(n)):
        raise JoinError("request_idx must be exactly 0..N-1; renderer will not shift or compact it")
    if len(chunk_t) == 0 or chunk_t[0] != 0 or np.any(np.diff(chunk_t) <= 0):
        raise JoinError("chunk_t must be strictly increasing and start at request_idx 0")
    if np.any(chunk_t < 0) or np.any(chunk_t >= n):
        raise JoinError(f"chunk_t contains a request outside 0..{n - 1}")

    boundary_pos = np.flatnonzero(np.r_[True, chunk_id[1:] != chunk_id[:-1]])
    boundary_requests = request_idx[boundary_pos]
    if not np.array_equal(boundary_requests, chunk_t):
        raise JoinError(
            f"chunk_t {chunk_t.tolist()} does not equal exact chunk_id boundaries {boundary_requests.tolist()}"
        )
    if not np.array_equal(chunk_id[boundary_pos], np.arange(len(boundary_pos))):
        raise JoinError("chunk_id must be contiguous 0..D-1 at boundaries")
    for lo, hi in zip(boundary_pos, np.r_[boundary_pos[1:], n]):
        want = np.arange(hi - lo)
        if not np.array_equal(chunk_step[lo:hi], want):
            raise JoinError(f"chunk_step for chunk_id {chunk_id[lo]} is not exactly 0..{len(want) - 1}")
    return boundary_pos, {int(req): i for i, req in enumerate(boundary_requests)}


def _normalise_q_heads(row: Mapping[str, Any]) -> Optional[List[List[float]]]:
    """Validate optional raw critic-head scores without inferring missing data.

    A v2 sidecar may omit this field.  When it is present, it is the exact
    ``[ensemble_head][candidate]`` tensor evaluated for this refill.  Keeping
    the candidate axis lets the dashboard derive Q1/Q2 for the *chosen*
    candidate from the existing ``q_argmax`` rather than inventing a value.
    """
    raw = row.get(Q_HEADS_FIELD)
    if raw is None:
        return None
    try:
        heads = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise JoinError(f"refill_stats line {row.get('_line', '?')}: q_heads must be numeric") from exc
    if heads.ndim != 2 or heads.shape[0] < 1 or heads.shape[1] < 1 or not np.all(np.isfinite(heads)):
        raise JoinError(
            f"refill_stats line {row.get('_line', '?')}: q_heads must have finite [E][K] shape, got {heads.shape}"
        )
    line = row.get("_line", "?")
    k = _strict_int(row["K"], "K", minimum=1, line=line) if row.get("K") is not None else None
    if k is not None and k != heads.shape[1]:
        raise JoinError(
            f"refill_stats line {row.get('_line', '?')}: K={row['K']} but q_heads has {heads.shape[1]} candidates"
        )
    if row.get("q_argmax") is None:
        raise JoinError(f"refill_stats line {row.get('_line', '?')}: q_heads requires q_argmax")
    chosen = _strict_int(row["q_argmax"], "q_argmax", minimum=0, line=line)
    if chosen < 0 or chosen >= heads.shape[1]:
        raise JoinError(
            f"refill_stats line {row.get('_line', '?')}: q_argmax={chosen} outside q_heads candidate axis"
        )
    if row.get("q_chosen") is None:
        raise JoinError(f"refill_stats line {row.get('_line', '?')}: q_heads requires aggregate q_chosen")
    return heads.astype(float).tolist()


def _q_values_from_row(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    values: Dict[str, Any] = {k: row[k] for k in Q_FIELDS if k in row and row[k] is not None}
    heads = _normalise_q_heads(row)
    if heads is not None:
        values[Q_HEADS_FIELD] = heads
    # K alone describes sampler configuration, not an observed critic score.
    return values if "q_chosen" in values else None


def _join_diagnostics(
    request_idx: np.ndarray,
    chunk_t: np.ndarray,
    chunk_id: np.ndarray,
    chunk_step: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Diagnostic], List[str]]:
    boundary_pos, boundary_by_request = _validate_policy_indices(request_idx, chunk_t, chunk_id, chunk_step)
    by_request: Dict[int, Mapping[str, Any]] = {}
    notes: List[str] = []
    used_legacy = False
    for raw_row in rows:
        row = _validate_discrete_sidecar_fields(raw_row)
        explicit = row.get("request_idx")
        legacy = row.get("t_index")
        if explicit is None and legacy is None:
            raise JoinError(f"refill_stats line {row.get('_line', '?')}: missing request_idx and legacy t_index")
        if explicit is not None and legacy is not None and explicit != legacy:
            raise JoinError(
                f"refill_stats line {row.get('_line', '?')}: request_idx={explicit} disagrees with t_index={legacy}"
            )
        req = explicit if explicit is not None else legacy
        used_legacy |= explicit is None
        if req not in boundary_by_request:
            raise JoinError(
                f"refill_stats line {row.get('_line', '?')}: request_idx {req} is not an exact chunk_t/chunk_id boundary"
            )
        if req in by_request:
            raise JoinError(f"duplicate refill diagnostic for request_idx {req}")
        expected_decision = boundary_by_request[req]
        if row.get("decision_idx") is not None and row["decision_idx"] != expected_decision:
            raise JoinError(
                f"refill_stats line {row.get('_line', '?')}: decision_idx={row['decision_idx']} "
                f"but exact boundary ordinal is {expected_decision}"
            )
        by_request[req] = row
    if used_legacy:
        notes.append("legacy refill_stats join: t_index used as request_idx; refill was label-only (no offset applied)")

    diagnostics: List[Diagnostic] = []
    held: Optional[Mapping[str, Any]] = None
    for pos, req in enumerate(request_idx):
        req_i = int(req)
        row = by_request.get(req_i)
        if row is not None:
            held = row
        q = None
        latency = None
        sidecar_refill = None
        # decision_idx always describes the chunk currently being displayed.
        # A missing sidecar row may leave an older Q value held on screen, but
        # must never make the new chunk inherit the older decision label.
        decision_idx = int(chunk_id[pos])
        if held is not None:
            q = _q_values_from_row(held)
            latency_values = {k: float(held[k]) for k in LATENCY_FIELDS if held.get(k) is not None}
            latency = latency_values or None
            sidecar_refill = held["refill"] if held.get("refill") is not None else None
        diagnostics.append(
            Diagnostic(
                request_idx=req_i,
                decision_idx=decision_idx,
                chunk_id=int(chunk_id[pos]),
                chunk_step=int(chunk_step[pos]),
                boundary=pos in set(boundary_pos.tolist()),
                q=q,
                latency=latency,
                sidecar_refill=sidecar_refill,
                sidecar_updated=row is not None,
            )
        )
    return diagnostics, notes


def load_policy_episode(
    log_dir: os.PathLike[str] | str,
    meta: EpisodeMeta,
    *,
    reset_counter: Optional[int] = None,
) -> PolicyEpisode:
    """Load one trusted-local IFQL policy log and strictly join its Q sidecar."""
    log_path = Path(log_dir).expanduser().resolve()
    counter = _policy_reset_counter(meta, reset_counter)
    npz_path = log_path / f"ep_{counter:04d}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"episode {meta.episode}: policy log not found: {npz_path}")

    # allow_pickle is required by the JPEG object arrays.  The CLI and README make
    # the trust boundary explicit; never point this at an untrusted downloaded NPZ.
    with np.load(npz_path, allow_pickle=True) as z:
        required = ("cam1_jpeg", "cam2_jpeg", "chunk_id", "chunk_step", "chunk_t")
        missing = [k for k in required if k not in z]
        if missing:
            raise JoinError(f"{npz_path}: missing required arrays {missing}")
        cam1 = np.asarray(z["cam1_jpeg"], dtype=object)
        cam2 = np.asarray(z["cam2_jpeg"], dtype=object)
        chunk_id = _as_int_vector(z["chunk_id"], "chunk_id")
        chunk_step = _as_int_vector(z["chunk_step"], "chunk_step")
        chunk_t = _as_int_vector(z["chunk_t"], "chunk_t")
        has_request_idx = "request_idx" in z
        request_idx = _as_int_vector(z["request_idx"], "request_idx") if has_request_idx else np.arange(len(cam1))
        state = np.asarray(z["state"]) if "state" in z else None
        action = np.asarray(z["action"]) if "action" in z else None
        npz_counter = (
            _strict_int(np.asarray(z["reset_counter"]).item(), "NPZ reset_counter", minimum=1, line=npz_path)
            if "reset_counter" in z else None
        )
    if npz_counter is not None and npz_counter != counter:
        raise JoinError(f"{npz_path}: reset_counter={npz_counter}, expected {counter}")
    n = len(request_idx)
    if len(cam1) != n or len(cam2) != n:
        raise JoinError(f"JPEG lengths differ: request_idx={n}, cam1={len(cam1)}, cam2={len(cam2)}")
    if n != meta.n_steps:
        raise JoinError(
            f"episode {meta.episode}: policy requests={n} but episodes.jsonl n_steps={meta.n_steps}; "
            "refusing to trim, pad, or shift"
        )
    for name, arr in (("state", state), ("action", action)):
        if arr is not None and (arr.ndim != 2 or len(arr) != n or arr.shape[1] != 7):
            raise JoinError(f"{npz_path}: {name} must have shape ({n}, 7), got {arr.shape}")
    rows = _load_refill_rows(log_path / "refill_stats.jsonl", counter)
    diagnostics, notes = _join_diagnostics(request_idx, chunk_t, chunk_id, chunk_step, rows)
    if not has_request_idx:
        notes.insert(0, "legacy policy NPZ: request_idx absent; exact array position 0..N-1 used")
    if not rows:
        notes.append("no matching refill_stats rows: Q and latency are explicitly unavailable")
    return PolicyEpisode(npz_path, request_idx, cam1, cam2, chunk_id, chunk_step, chunk_t,
                         state, action, diagnostics, notes)


def _nearest_rows(times: np.ndarray, n_frames: int, fps: int) -> np.ndarray:
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError("sim_mj_state/t_rel_s must be a non-empty finite vector")
    if np.any(np.diff(times) <= 0):
        raise ValueError("sim_mj_state/t_rel_s must be strictly increasing")
    targets = np.arange(n_frames, dtype=np.float64) / float(fps)
    if times[0] > 1e-6 or times[-1] + 1e-9 < targets[-1]:
        raise ValueError(
            f"H5 state coverage [{times[0]:.6f}, {times[-1]:.6f}] does not cover video [0, {targets[-1]:.6f}]"
        )
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(times[left] - targets) <= np.abs(times[right] - targets)
    return np.where(choose_left, left, right).astype(np.int64)


def load_h5_episode(run_dir: os.PathLike[str] | str, meta: EpisodeMeta, fps: int = FPS) -> H5Episode:
    path = Path(run_dir) / f"ep_{meta.episode}.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        if "sim_scene" not in f or "sim_mj_state" not in f:
            raise KeyError(f"{path}: missing /sim_scene or /sim_mj_state")
        scene = f["sim_scene"]
        xml_value = scene["xml"][()]
        xml = xml_value.decode() if isinstance(xml_value, bytes) else str(xml_value)
        config = json.loads(str(scene.attrs.get("config", "{}")))
        layout = json.loads(str(scene.attrs.get("layout", "{}")))
        state = f["sim_mj_state"]
        nq, nv, nu = int(state.attrs["nq"]), int(state.attrs["nv"]), int(state.attrs["nu"])
        t = np.asarray(state["t_rel_s"][:], dtype=np.float64)
        qpos = np.stack([state[f"qpos{i}"][:] for i in range(nq)], axis=1)
        qvel = np.stack([state[f"qvel{i}"][:] for i in range(nv)], axis=1)
        ctrl = np.stack([state[f"ctrl{i}"][:] for i in range(nu)], axis=1)
    rows = _nearest_rows(t, meta.n_steps, fps)
    return H5Episode(path, xml, config, layout, t, qpos, qvel, ctrl, rows)


def _rebuild_assets(h5: H5Episode) -> Tuple[Dict[str, bytes], List[str]]:
    notes: List[str] = []
    assets: Dict[str, bytes] = {}
    if not h5.config:
        return assets, ["H5 sim_scene has no config; compiling recorded XML without external assets"]
    try:
        from sim_collect.scene import SceneConfig, build_scene

        built = build_scene(SceneConfig.from_dict(dict(h5.config)), dict(h5.layout) or None)
        assets = dict(built.assets)
        recorded_sha = hashlib.sha256(h5.xml.encode()).hexdigest()
        rebuilt_sha = hashlib.sha256(built.xml.encode()).hexdigest()
        if recorded_sha != rebuilt_sha:
            notes.append("rebuilt scene XML differs; recorded XML is used with rebuilt assets")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"scene asset rebuild failed: {type(exc).__name__}: {exc}")
    return assets, notes


def _decode_jpeg(value: Any, name: str) -> np.ndarray:
    import cv2

    if isinstance(value, np.ndarray):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{name}: expected JPEG bytes, got {type(value).__name__}")
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{name}: JPEG decode failed")
    if image.shape[1::-1] != CAM_SIZE:
        image = cv2.resize(image, CAM_SIZE, interpolation=cv2.INTER_AREA)
    return image


def _put_text(image: np.ndarray, text: str, xy: Tuple[int, int], scale: float = 0.48,
              color: Tuple[int, int, int] = (225, 230, 235), thickness: int = 1) -> None:
    import cv2

    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "missing"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _chunk_status(diag: Diagnostic) -> str:
    """Return a truthful status label for the current request.

    Q is measured only when a sidecar row is present at a chunk boundary. A
    previous value may remain visible for continuity, but that is not a critic
    update for the new chunk.
    """
    if diag.boundary:
        if diag.sidecar_updated and diag.q is not None:
            return "NEW CHUNK / CRITIC UPDATED"
        if diag.q is not None:
            return "NEW CHUNK / Q NOT LOGGED - PREVIOUS Q HELD"
        return "NEW CHUNK / Q NOT LOGGED"
    if diag.q is not None:
        return "PREVIOUS CHUNK Q HELD"
    return "NO Q LOG ATTACHED"


def _q_history_values(history: Sequence[Diagnostic]) -> np.ndarray:
    """Return held q_chosen values; NaN means no logged critic value."""
    return _q_history_series(history, "aggregate")


def _chosen_q_heads(q: Optional[Mapping[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """Return Q1/Q2 for the aggregate-selected candidate, if the row logged them."""
    if q is None or q.get(Q_HEADS_FIELD) is None or q.get("q_argmax") is None:
        return None, None
    heads = np.asarray(q[Q_HEADS_FIELD], dtype=np.float64)
    try:
        choice = _strict_int(q["q_argmax"], "q_argmax", minimum=0, line="diagnostic")
    except JoinError:
        return None, None
    if heads.ndim != 2 or choice < 0 or choice >= heads.shape[1]:
        # This is defensive only: sidecars are validated before entering a
        # Diagnostic.  Do not substitute another candidate if malformed data
        # somehow reaches a caller directly.
        return None, None
    q1 = float(heads[0, choice]) if heads.shape[0] >= 1 else None
    q2 = float(heads[1, choice]) if heads.shape[0] >= 2 else None
    return q1, q2


def _q_history_series(history: Sequence[Diagnostic], kind: str) -> np.ndarray:
    """Return one held critic trace; unavailable heads are represented by NaN."""
    values = np.full(len(history), np.nan, dtype=np.float64)
    for i, item in enumerate(history):
        if item.q is None:
            continue
        if kind == "aggregate" and item.q.get("q_chosen") is not None:
            values[i] = float(item.q["q_chosen"])
        elif kind in {"q1", "q2"}:
            q1, q2 = _chosen_q_heads(item.q)
            value = q1 if kind == "q1" else q2
            if value is not None:
                values[i] = value
    return values


def _step_trace_points(
    values: np.ndarray,
    point_fn: Any,
) -> List[List[Tuple[int, int]]]:
    """Build finite zero-order-hold paths without diagonal boundary interpolation."""
    segments: List[List[Tuple[int, int]]] = []
    segment: List[Tuple[int, int]] = []
    previous_value: Optional[float] = None
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            if segment:
                segments.append(segment)
                segment = []
            previous_value = None
            continue
        value = float(value)
        current = point_fn(idx, value)
        if not segment:
            segment = [current]
        else:
            # Hold the old score through the interval, then jump vertically at
            # the new sample.  A plain polyline would imply a critic update
            # happened gradually before the chunk boundary.
            x, _ = current
            segment.append((x, point_fn(idx - 1, float(previous_value))[1]))
            segment.append(current)
        previous_value = value
    if segment:
        segments.append(segment)
    return segments


def _draw_q_graph(dashboard: np.ndarray, history: Sequence[Diagnostic], fps: int) -> None:
    """Draw selected Q1/Q2 and aggregate Q over the episode's elapsed time."""
    import cv2

    x0, y0, x1, y1 = 14, 52, 790, 340
    cv2.rectangle(dashboard, (x0, y0), (x1, y1), (62, 68, 78), 1)
    _put_text(dashboard, "CHUNK CRITIC SCORES  (not p(success))", (x0 + 10, y0 + 22), 0.52,
              (195, 210, 225), 1)
    legend = (("Q1 chosen", "q1", (95, 225, 95)),
              ("Q2 chosen", "q2", (230, 130, 220)),
              ("aggregate q_chosen", "aggregate", (80, 215, 255)))
    for j, (label, _, color) in enumerate(legend):
        lx = x0 + 12 + j * 205
        cv2.line(dashboard, (lx, y0 + 36), (lx + 18, y0 + 36), color, 2, cv2.LINE_AA)
        _put_text(dashboard, label, (lx + 24, y0 + 40), 0.36, color)

    series = {key: _q_history_series(history, key) for _, key, _ in legend}
    finite_values = [values[np.isfinite(values)] for values in series.values() if np.any(np.isfinite(values))]
    if not finite_values:
        _put_text(dashboard, "UNAVAILABLE / Q NOT LOGGED", (x0 + 220, y0 + 155), 0.72, (80, 150, 255), 2)
        _put_text(dashboard, "No raw Q1/Q2 or aggregate score was fabricated.", (x0 + 185, y0 + 184),
                  0.43, (160, 175, 190))
        return

    all_values = np.concatenate(finite_values)
    q_min = float(np.min(all_values))
    q_max = float(np.max(all_values))
    if np.isclose(q_min, q_max):
        pad = max(0.05, abs(q_min) * 0.05)
        q_min -= pad
        q_max += pad
    else:
        pad = 0.08 * (q_max - q_min)
        q_min -= pad
        q_max += pad
    plot_l, plot_r = x0 + 48, x1 - 14
    plot_t, plot_b = y0 + 57, y1 - 29
    cv2.line(dashboard, (plot_l, plot_b), (plot_r, plot_b), (78, 84, 94), 1)
    cv2.line(dashboard, (plot_l, plot_t), (plot_l, plot_b), (78, 84, 94), 1)
    n_total = max(2, len(history))

    def point(idx: int, value: float) -> Tuple[int, int]:
        x = int(round(plot_l + idx * (plot_r - plot_l) / (n_total - 1)))
        y = int(round(plot_b - (value - q_min) * (plot_b - plot_t) / (q_max - q_min)))
        return x, y

    # Faint vertical rules make the refill semantics visible without inventing
    # a continuous value between decisions.  Values themselves are held from
    # the sidecar score at one chunk boundary to the next.
    for idx, item in enumerate(history):
        if item.boundary:
            x, _ = point(idx, q_min)
            cv2.line(dashboard, (x, plot_t), (x, plot_b), (48, 54, 64), 1)

    for _, key, color in legend:
        values = series[key]
        segments = _step_trace_points(values, point)
        for points in segments:
            if len(points) > 1:
                cv2.polylines(dashboard, [np.asarray(points, np.int32)], False, color, 2, cv2.LINE_AA)
            else:
                cv2.circle(dashboard, points[0], 2, color, -1, cv2.LINE_AA)
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite):
            current = point(int(finite[-1]), float(values[finite[-1]]))
            cv2.circle(dashboard, current, 4, color, -1, cv2.LINE_AA)
            cv2.circle(dashboard, current, 6, (28, 31, 38), 1, cv2.LINE_AA)
    _put_text(dashboard, f"{q_max:+.3f}", (x0 + 4, plot_t + 7), 0.34, (150, 160, 175))
    _put_text(dashboard, f"{q_min:+.3f}", (x0 + 4, plot_b), 0.34, (150, 160, 175))
    _put_text(dashboard, "0.0s", (plot_l, y1 - 7), 0.34, (150, 160, 175))
    _put_text(dashboard, f"{(len(history) - 1) / fps:.1f}s", (plot_r - 48, y1 - 7), 0.34,
              (150, 160, 175))
    _put_text(dashboard, "vertical rules = chunk boundaries; values hold until next boundary",
              (plot_l + 98, y1 - 7), 0.33, (125, 138, 150))


def draw_dashboard(meta: EpisodeMeta, diag: Diagnostic, gripper: Optional[float], source: str,
                   history: Optional[Sequence[Diagnostic]] = None, fps: int = FPS,
                   synthetic: bool = False) -> np.ndarray:
    import cv2

    w, h = DASHBOARD_SIZE
    panel = np.full((h, w, 3), (22, 25, 31), dtype=np.uint8)
    accent = OUTCOME_BGR.get(meta.outcome, (180, 180, 180))
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (75, 80, 90), 1)
    if synthetic:
        # Reserved dashboard banner: cameras and the graph/info regions remain
        # pixel-for-pixel untouched by the warning.
        cv2.rectangle(panel, (0, 0), (w, 35), (20, 20, 185), -1)
        _put_text(
            panel,
            f"SYNTHETIC DIAGNOSTICS - NOT AN EXPERIMENT RESULT  |  {meta.outcome.upper()} "
            f"ep {meta.episode}  |  {source}",
            (45, 24), 0.55, (255, 255, 255), 2,
        )
    else:
        cv2.rectangle(panel, (0, 0), (w, 35), accent, -1)
        _put_text(panel, f"{meta.outcome.upper()}  episode {meta.episode}  |  {source}", (14, 24), 0.62,
                  (10, 10, 10), 2)
    status = _chunk_status(diag)
    if diag.boundary:
        cv2.rectangle(panel, (806, 43), (1265, 68), (0, 185, 255), -1)
        _put_text(panel, status, (817, 61), 0.40, (20, 20, 20), 1)
    else:
        _put_text(panel, status, (817, 61), 0.42, (160, 170, 180))

    _draw_q_graph(panel, history if history is not None else [diag], fps)

    x = 817
    _put_text(panel, "CURRENT CHUNK", (x, 94), 0.52, (115, 210, 255), 1)
    q = diag.q
    q1, q2 = _chosen_q_heads(q)
    if q is None:
        _put_text(panel, "aggregate: MISSING", (x, 119), 0.52, (80, 150, 255), 2)
    else:
        _put_text(panel, f"aggregate {_fmt(q.get('q_chosen'))}", (x, 119), 0.52, (80, 215, 255), 1)
    _put_text(panel, f"Q1 chosen {_fmt(q1)}", (x, 143), 0.47, (95, 225, 95))
    _put_text(panel, f"Q2 chosen {_fmt(q2)}", (x, 165), 0.47, (230, 130, 220))
    disagreement = None if q1 is None or q2 is None else abs(q1 - q2)
    _put_text(panel, f"|Q1-Q2| {_fmt(disagreement)}", (x, 187), 0.45, (195, 205, 215))
    if q is not None:
        _put_text(panel, f"range [{_fmt(q.get('q_min'))}, {_fmt(q.get('q_max'))}]", (x, 209), 0.42)
        _put_text(panel, f"spread {_fmt(q.get('q_spread'))}  std {_fmt(q.get('q_std'))}", (x, 229), 0.42)
        _put_text(panel, f"K {q.get('K', 'missing')}  selected candidate {q.get('q_argmax', 'missing')}",
                  (x, 250), 0.43)
    latency = diag.latency or {}
    _put_text(panel, "LATENCY (measured at refill, then held)", (x, 274), 0.42, (115, 210, 255))
    _put_text(panel, f"encode {_fmt(latency.get('encode_ms'), 1)} ms  sample {_fmt(latency.get('sample_ms'), 1)} ms",
              (x, 294), 0.42)
    refill_label = str(diag.sidecar_refill) if diag.sidecar_refill is not None else "missing"
    _put_text(panel, f"refill {_fmt(latency.get('refill_ms'), 1)} ms  global refill {refill_label}",
              (x, 314), 0.42)
    _put_text(panel, f"request {diag.request_idx}  decision {diag.decision_idx}  chunk {diag.chunk_id}/{diag.chunk_step}",
              (x, 334), 0.40)
    _put_text(panel, f"gripper {_fmt(gripper)}", (x + 310, 334), 0.40)
    detail = meta.fault or meta.failure or meta.detail or "(no detail)"
    detail_lines = textwrap.wrap(str(detail), width=70)[:1]
    for j, line in enumerate(detail_lines):
        _put_text(panel, line, (14, 355 - 17 * j), 0.40, (190, 195, 205))
    return panel


def _synthetic_diagnostics(n_steps: int) -> List[Diagnostic]:
    out: List[Diagnostic] = []
    held: Optional[Dict[str, Any]] = None
    for i in range(n_steps):
        decision = i // 24
        boundary = i % 24 == 0
        if boundary:
            k = 32
            candidates = np.arange(k, dtype=np.float64)
            # These are deliberately two distinct ensemble heads.  They are
            # only a layout fixture, never a claim about an evaluation run.
            q1 = -0.54 + 0.11 * np.sin(decision * 0.73 + candidates * 0.31)
            q2 = -0.43 + 0.13 * np.cos(decision * 0.57 + candidates * 0.27)
            aggregate = 0.5 * (q1 + q2)
            chosen = int(np.argmax(aggregate))
            held = {
                "K": k, "q_chosen": float(aggregate[chosen]), "q_mean": float(np.mean(aggregate)),
                "q_std": float(np.std(aggregate)), "q_min": float(np.min(aggregate)),
                "q_max": float(np.max(aggregate)), "q_spread": float(np.ptp(aggregate)),
                "q_argmax": chosen, "q_heads": [q1.tolist(), q2.tolist()],
                "encode_ms": 17.0 + decision % 4, "sample_ms": 23.0 + decision % 5,
                "refill_ms": 42.0 + decision % 6,
            }
        q = {key: held[key] for key in (*Q_FIELDS, Q_HEADS_FIELD)} if held else None
        latency = {k: float(held[k]) for k in LATENCY_FIELDS} if held else None
        out.append(Diagnostic(i, decision, decision, i % 24, boundary, q, latency,
                              sidecar_refill=decision + 1, sidecar_updated=boundary))
    return out


def _gripper(policy: Optional[PolicyEpisode], h5: H5Episode, frame: int) -> Optional[float]:
    if policy is not None:
        if policy.state is not None:
            return float(policy.state[frame, 6])
        if policy.action is not None:
            return float(policy.action[frame, 6])
    row = int(h5.frame_rows[frame])
    if h5.ctrl.shape[1] >= 7 and np.isfinite(h5.ctrl[row, 6]):
        return float(np.clip(h5.ctrl[row, 6] / 255.0, 0.0, 1.0))
    return None


def _overlay_camera_label(frame: np.ndarray, label: str, meta: EpisodeMeta, frame_idx: int, fps: int) -> None:
    import cv2

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    _put_text(frame, f"{label}  seed {meta.seed}  t={frame_idx / fps:5.2f}s", (10, 21), 0.52,
              (255, 255, 255), 1)


def _encode_h264(temp_input: Path, output: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to produce H.264/yuv420p output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.stem}.", suffix=".h264.mp4",
                                     dir=output.parent, delete=False) as fh:
        encoded = Path(fh.name)
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(temp_input), "-r", str(fps),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)],
            check=True,
        )
        encoded.chmod(0o644)
        os.replace(encoded, output)
    finally:
        encoded.unlink(missing_ok=True)


def render_episode(
    run_dir: os.PathLike[str] | str,
    meta: EpisodeMeta,
    out_dir: os.PathLike[str] | str,
    *,
    policy_log_dir: Optional[os.PathLike[str] | str] = None,
    reset_counter: Optional[int] = None,
    synthetic_diagnostics: bool = False,
    fps: int = FPS,
) -> Path:
    import cv2

    if fps != FPS:
        raise ValueError(f"diagnostic video contract is fixed at {FPS} fps")
    h5 = load_h5_episode(run_dir, meta, fps=fps)
    policy = None
    if policy_log_dir is not None:
        policy = load_policy_episode(policy_log_dir, meta, reset_counter=reset_counter)
        for note in policy.notes:
            print(f"[render_eval_video] ep {meta.episode}: {note}", flush=True)
    diagnostics = _synthetic_diagnostics(meta.n_steps) if synthetic_diagnostics else (
        policy.diagnostics if policy is not None else [
            Diagnostic(i, -1, -1, -1, False) for i in range(meta.n_steps)
        ]
    )
    source = "policy JPEG" if policy is not None else "H5 kinematic replay"

    rig = None
    if policy is None:
        from sim_collect.cameras import CameraRig

        assets, notes = _rebuild_assets(h5)
        for note in notes:
            print(f"[render_eval_video] ep {meta.episode}: {note}", flush=True)
        rig = CameraRig(h5.xml, assets or None, color_size=CAM_SIZE, cams=("cam1", "cam2"),
                        render=h5.config.get("render") if isinstance(h5.config, dict) else None)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_synthetic" if synthetic_diagnostics else ""
    final_path = out / f"ep_{meta.episode}_{meta.outcome}_diagnostic{suffix}.mp4"
    fd, temp_name = tempfile.mkstemp(prefix=f".{final_path.stem}.", suffix=".mp4v.mp4", dir=out)
    os.close(fd)
    temp_path = Path(temp_name)
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, OUTPUT_SIZE)
    if not writer.isOpened():
        temp_path.unlink(missing_ok=True)
        if rig is not None:
            rig.close()
        raise RuntimeError(f"OpenCV VideoWriter could not open {temp_path}")
    try:
        for i in range(meta.n_steps):
            if policy is not None:
                cam1 = _decode_jpeg(policy.cam1_jpeg[i], f"ep {meta.episode} cam1 request {i}")
                cam2 = _decode_jpeg(policy.cam2_jpeg[i], f"ep {meta.episode} cam2 request {i}")
            else:
                row = int(h5.frame_rows[i])
                rig.mirror(h5.qpos[row], h5.qvel[row])
                cam1 = cv2.cvtColor(rig.render_color("cam1"), cv2.COLOR_RGB2BGR)
                cam2 = cv2.cvtColor(rig.render_color("cam2"), cv2.COLOR_RGB2BGR)
            _overlay_camera_label(cam1, "cam1", meta, i, fps)
            _overlay_camera_label(cam2, "cam2", meta, i, fps)
            dashboard = draw_dashboard(meta, diagnostics[i], _gripper(policy, h5, i), source,
                                       diagnostics[:i + 1], fps, synthetic=synthetic_diagnostics)
            composed = np.vstack((np.hstack((cam1, cam2)), dashboard))
            writer.write(composed)
    finally:
        writer.release()
        if rig is not None:
            rig.close()
    try:
        _encode_h264(temp_path, final_path, fps)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"[render_eval_video] ep {meta.episode}: {meta.n_steps} frames -> {final_path}", flush=True)
    return final_path


def _parse_reset_map(spec: Optional[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not spec:
        return out
    for item in spec.split(","):
        episode, sep, counter = item.strip().partition(":")
        if not sep or not episode or not counter:
            raise ValueError(f"invalid --reset-map item {item!r}; expected EPISODE:COUNTER")
        if episode in out:
            raise ValueError(f"duplicate --reset-map episode {episode!r}")
        if not counter.isascii() or not counter.isdecimal() or int(counter) <= 0:
            raise ValueError(f"invalid --reset-map counter {counter!r}; expected a positive decimal integer")
        out[episode] = int(counter)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="directory containing episodes.jsonl and ep_<episode>.h5")
    ap.add_argument("--episodes", default=None, help="comma-separated episode ids (default: all JSONL rows)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--policy-log-dir", default=None,
                    help="trusted local IFQL directory with ep_<reset_counter:04d>.npz and refill_stats.jsonl")
    ap.add_argument("--reset-map", default=None,
                    help="explicit legacy mapping EPISODE:RESET_COUNTER[,..]; no index shifting is inferred")
    ap.add_argument("--synthetic-diagnostics", action="store_true",
                    help="watermarked fake Q/latency panel for layout demonstration only")
    args = ap.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "glfw")
    episodes = load_episodes_jsonl(args.run_dir)
    selected = list(episodes) if not args.episodes else [x.strip() for x in args.episodes.split(",") if x.strip()]
    unknown = [x for x in selected if x not in episodes]
    if unknown:
        ap.error(f"episodes not found in episodes.jsonl: {unknown}")
    reset_map = _parse_reset_map(args.reset_map)
    unused = sorted(set(reset_map) - set(selected))
    if unused:
        ap.error(f"--reset-map contains unselected episodes: {unused}")
    for episode in selected:
        render_episode(args.run_dir, episodes[episode], args.out,
                       policy_log_dir=args.policy_log_dir, reset_counter=reset_map.get(episode),
                       synthetic_diagnostics=args.synthetic_diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
