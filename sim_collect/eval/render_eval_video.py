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
PANEL_SIZE = (480, 360)
OUTPUT_SIZE = (CAM_SIZE[0] * 2 + PANEL_SIZE[0], CAM_SIZE[1])

OUTCOME_BGR = {
    "success": (70, 210, 80),
    "timeout": (70, 90, 235),
    "failure": (60, 60, 230),
    "fault": (0, 170, 255),
}
Q_FIELDS = ("K", "q_chosen", "q_mean", "q_std", "q_min", "q_max", "q_spread", "q_argmax")
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
                n_steps = int(row["n_steps"])
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
                seed=None if seed is None else int(seed),
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
        if arr.dtype.kind not in "f" or not np.all(np.isfinite(arr)) or not np.all(arr == np.floor(arr)):
            raise JoinError(f"{name} must contain integers")
    return arr.astype(np.int64, copy=False)


def _policy_reset_counter(meta: EpisodeMeta, override: Optional[int]) -> int:
    recorded = meta.policy_reset.get("reset_counter")
    if override is not None:
        if int(override) <= 0:
            raise JoinError(f"episode {meta.episode}: reset counter override must be positive")
        return int(override)
    if recorded is None:
        raise JoinError(
            f"episode {meta.episode}: no reset_counter in episodes.jsonl; pass an explicit "
            "--reset-map EPISODE:COUNTER (indices are never shifted or inferred)"
        )
    return int(recorded)


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
            if int(row.get("reset_counter", -1)) == reset_counter:
                row["_line"] = lineno
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
    for row in rows:
        explicit = row.get("request_idx")
        legacy = row.get("t_index")
        if explicit is None and legacy is None:
            raise JoinError(f"refill_stats line {row.get('_line', '?')}: missing request_idx and legacy t_index")
        if explicit is not None and legacy is not None and int(explicit) != int(legacy):
            raise JoinError(
                f"refill_stats line {row.get('_line', '?')}: request_idx={explicit} disagrees with t_index={legacy}"
            )
        req = int(explicit if explicit is not None else legacy)
        used_legacy |= explicit is None
        if req not in boundary_by_request:
            raise JoinError(
                f"refill_stats line {row.get('_line', '?')}: request_idx {req} is not an exact chunk_t/chunk_id boundary"
            )
        if req in by_request:
            raise JoinError(f"duplicate refill diagnostic for request_idx {req}")
        expected_decision = boundary_by_request[req]
        if row.get("decision_idx") is not None and int(row["decision_idx"]) != expected_decision:
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
            q_values = {k: held[k] for k in Q_FIELDS if k in held and held[k] is not None}
            q = q_values if "q_chosen" in q_values else None
            latency_values = {k: float(held[k]) for k in LATENCY_FIELDS if held.get(k) is not None}
            latency = latency_values or None
            sidecar_refill = int(held["refill"]) if held.get("refill") is not None else None
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
        npz_counter = int(np.asarray(z["reset_counter"]).item()) if "reset_counter" in z else None
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
    values = np.full(len(history), np.nan, dtype=np.float64)
    for i, item in enumerate(history):
        if item.q is not None and item.q.get("q_chosen") is not None:
            values[i] = float(item.q["q_chosen"])
    return values


def _draw_q_graph(panel: np.ndarray, history: Sequence[Diagnostic], fps: int) -> None:
    """Draw q_chosen over elapsed episode time in a compact fixed panel area."""
    import cv2

    x0, y0, x1, y1 = 238, 72, 468, 166
    cv2.rectangle(panel, (x0, y0), (x1, y1), (62, 68, 78), 1)
    _put_text(panel, "q_chosen over time", (x0 + 7, y0 + 15), 0.38, (180, 190, 205))
    values = _q_history_values(history)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        _put_text(panel, "UNAVAILABLE", (x0 + 55, y0 + 47), 0.50, (80, 150, 255), 2)
        _put_text(panel, "Q NOT LOGGED", (x0 + 55, y0 + 68), 0.38, (80, 150, 255), 1)
        return

    q_min = float(np.nanmin(values))
    q_max = float(np.nanmax(values))
    if np.isclose(q_min, q_max):
        pad = max(0.05, abs(q_min) * 0.05)
        q_min -= pad
        q_max += pad
    else:
        pad = 0.08 * (q_max - q_min)
        q_min -= pad
        q_max += pad
    plot_l, plot_r = x0 + 7, x1 - 7
    plot_t, plot_b = y0 + 22, y1 - 17
    cv2.line(panel, (plot_l, plot_b), (plot_r, plot_b), (78, 84, 94), 1)
    n_total = max(2, len(history))

    def point(idx: int, value: float) -> Tuple[int, int]:
        x = int(round(plot_l + idx * (plot_r - plot_l) / (n_total - 1)))
        y = int(round(plot_b - (value - q_min) * (plot_b - plot_t) / (q_max - q_min)))
        return x, y

    # Do not bridge gaps where no Q was logged. A normal held trace is
    # continuous because each Diagnostic contains the explicitly held value.
    segments: List[List[Tuple[int, int]]] = []
    segment: List[Tuple[int, int]] = []
    for idx, value in enumerate(values):
        if np.isfinite(value):
            segment.append(point(idx, float(value)))
        elif segment:
            segments.append(segment)
            segment = []
    if segment:
        segments.append(segment)
    for points in segments:
        if len(points) > 1:
            cv2.polylines(panel, [np.asarray(points, np.int32)], False, (80, 215, 255), 1,
                          cv2.LINE_AA)
        else:
            cv2.circle(panel, points[0], 1, (80, 215, 255), -1, cv2.LINE_AA)
    current_idx = int(finite[-1])
    current = point(current_idx, float(values[current_idx]))
    cv2.circle(panel, current, 3, (80, 90, 255), -1, cv2.LINE_AA)
    _put_text(panel, f"{q_max:+.2f}", (x0 + 3, plot_t + 7), 0.28, (145, 155, 170))
    _put_text(panel, f"{q_min:+.2f}", (x0 + 3, plot_b), 0.28, (145, 155, 170))
    _put_text(panel, f"0s", (plot_l, y1 - 3), 0.28, (145, 155, 170))
    _put_text(panel, f"{(len(history) - 1) / fps:.1f}s", (plot_r - 34, y1 - 3), 0.28,
              (145, 155, 170))


def draw_panel(meta: EpisodeMeta, diag: Diagnostic, gripper: Optional[float], source: str,
               history: Optional[Sequence[Diagnostic]] = None, fps: int = FPS) -> np.ndarray:
    import cv2

    w, h = PANEL_SIZE
    panel = np.full((h, w, 3), (22, 25, 31), dtype=np.uint8)
    accent = OUTCOME_BGR.get(meta.outcome, (180, 180, 180))
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (75, 80, 90), 1)
    cv2.rectangle(panel, (0, 0), (w, 34), accent, -1)
    _put_text(panel, f"{meta.outcome.upper()}  episode {meta.episode}", (12, 23), 0.58, (10, 10, 10), 2)
    status = _chunk_status(diag)
    if diag.boundary:
        cv2.rectangle(panel, (0, 34), (w, 56), (0, 185, 255), -1)
        _put_text(panel, status, (12, 50), 0.38, (20, 20, 20), 1)
    else:
        _put_text(panel, status, (12, 50), 0.42, (160, 170, 180))

    _put_text(panel, "CRITIC SCORE (not p(success))", (12, 76), 0.52, (115, 210, 255), 1)
    q = diag.q
    if q is None:
        _put_text(panel, "Q: MISSING", (12, 101), 0.54, (80, 150, 255), 2)
        q_lines = ["mean/range: missing", "spread/std: missing"]
        q_line_y = 124
    else:
        q_lines = [
            f"chosen {_fmt(q.get('q_chosen'))}",
            f"mean {_fmt(q.get('q_mean'))}",
            f"range [{_fmt(q.get('q_min'))}, {_fmt(q.get('q_max'))}]",
            f"spread {_fmt(q.get('q_spread'))}  std {_fmt(q.get('q_std'))}",
        ]
        q_line_y = 94
    for j, line in enumerate(q_lines):
        _put_text(panel, line, (12, q_line_y + 18 * j), 0.40)

    _draw_q_graph(panel, history if history is not None else [diag], fps)

    latency = diag.latency or {}
    _put_text(panel, "LATENCY (chunk boundary measurement)", (12, 176), 0.45, (115, 210, 255))
    _put_text(panel,
              f"encode {_fmt(latency.get('encode_ms'), 1)} ms   sample {_fmt(latency.get('sample_ms'), 1)} ms",
              (12, 198), 0.45)
    refill_label = str(diag.sidecar_refill) if diag.sidecar_refill is not None else "missing"
    _put_text(panel,
              f"refill {_fmt(latency.get('refill_ms'), 1)} ms   global refill {refill_label}",
              (12, 219), 0.45)

    k = q.get("K") if q else None
    _put_text(panel,
              f"K {k if k is not None else 'missing'}  decision {diag.decision_idx}  request {diag.request_idx}",
              (12, 245), 0.47)
    _put_text(panel, f"chunk_id {diag.chunk_id}  chunk_step {diag.chunk_step}", (12, 266), 0.47)
    _put_text(panel, f"gripper {_fmt(gripper)}", (12, 287), 0.47)
    _put_text(panel, f"camera source: {source}", (12, 309), 0.42, (170, 180, 190))
    detail = meta.fault or meta.failure or meta.detail or "(no detail)"
    detail_lines = textwrap.wrap(str(detail), width=53)[:2]
    for j, line in enumerate(detail_lines):
        _put_text(panel, line, (12, 331 + 18 * j), 0.40, (190, 195, 205))
    return panel


def _synthetic_diagnostics(n_steps: int) -> List[Diagnostic]:
    out: List[Diagnostic] = []
    held: Optional[Dict[str, Any]] = None
    for i in range(n_steps):
        decision = i // 24
        boundary = i % 24 == 0
        if boundary:
            centre = -0.45 + 0.12 * np.sin(decision * 0.8)
            spread = 0.18 + 0.03 * (decision % 3)
            held = {
                "K": 32, "q_chosen": centre + spread / 2, "q_mean": centre,
                "q_std": spread / 3, "q_min": centre - spread / 2,
                "q_max": centre + spread / 2, "q_spread": spread, "q_argmax": 7,
                "encode_ms": 17.0 + decision % 4, "sample_ms": 23.0 + decision % 5,
                "refill_ms": 42.0 + decision % 6,
            }
        q = {k: held[k] for k in Q_FIELDS} if held else None
        latency = {k: float(held[k]) for k in LATENCY_FIELDS} if held else None
        out.append(Diagnostic(i, decision, decision, i % 24, boundary, q, latency,
                              sidecar_updated=boundary))
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
            panel = draw_panel(meta, diagnostics[i], _gripper(policy, h5, i), source,
                               diagnostics[:i + 1], fps)
            composed = np.hstack((cam1, cam2, panel))
            if synthetic_diagnostics:
                # Keep the diagnostic panel fully readable. The watermark is
                # deliberately confined to the two camera panes.
                cv2.rectangle(composed, (0, 326), (CAM_SIZE[0] * 2, 360), (10, 10, 10), -1)
                _put_text(composed, "SYNTHETIC DIAGNOSTICS - NOT AN EXPERIMENT RESULT",
                          (205, 350), 0.72, (40, 70, 255), 2)
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
