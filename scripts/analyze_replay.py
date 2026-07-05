#!/usr/bin/env python3
"""Deterministic trajectory metrics for a recorded GELLO/UR session source.

Prints a single JSON object to stdout. Used by the replay-verification workflow
so each agent gets identical, reproducible numbers to interpret (motion range,
peak joint speed, jerk proxy, discontinuity/vibration count, NaN count). For
``--source smooth-command`` it also reports the same metrics AFTER applying the
replay tool's accel-limited smoothing, so the vibration reduction is quantified.

No ROS2, no robot, no GELLO hardware needed — pure CSV -> numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from ur_command_smoothing import accel_limited_command  # noqa: E402

SOURCES = {
    "ur": ("ur_joint_states.csv", "q"),
    "command": ("command.csv", "cmd"),
    "smooth-command": ("command.csv", "cmd"),
    "gello": ("gello_joint_states.csv", "q"),
}
STEP_THRESH_RAD = 0.02  # per-sample jump above this = discontinuity/vibration proxy


def metrics(t: np.ndarray, q: np.ndarray) -> dict:
    dt = np.diff(t)
    dt_med = float(np.median(dt)) if len(dt) else float("nan")
    rate_hz = 1.0 / dt_med if dt_med and np.isfinite(dt_med) and dt_med > 0 else float("nan")
    safe_dt = np.where(dt > 1e-9, dt, np.nan)
    vel = np.diff(q, axis=0) / safe_dt[:, None]
    peak_speed = float(np.nanmax(np.abs(vel))) if vel.size else 0.0
    jerk = np.diff(vel, axis=0)
    jerk_metric = float(np.nanmean(np.abs(jerk))) if jerk.size else 0.0
    steps = np.abs(np.diff(q, axis=0))
    n_large_steps = int((steps > STEP_THRESH_RAD).sum())
    per_joint_range = [float(np.nanmax(q[:, i]) - np.nanmin(q[:, i])) for i in range(q.shape[1])]
    return {
        "rate_hz": round(rate_hz, 2),
        "peak_speed_rad_s": round(peak_speed, 4),
        "jerk_metric": round(jerk_metric, 4),
        "n_large_steps": n_large_steps,
        "per_joint_range_rad": [round(v, 4) for v in per_joint_range],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    args = ap.parse_args()

    filename, prefix = SOURCES[args.source]
    csv_path = args.session / filename
    df = pd.read_csv(csv_path)
    cols = [f"{prefix}{i}" for i in range(1, 7)]
    t = df["t_rel_s"].to_numpy(dtype=float)
    q = df[cols].to_numpy(dtype=float)
    finite = np.isfinite(t) & np.isfinite(q).all(axis=1)
    n_nan = int((~finite).sum())
    t, q = t[finite], q[finite]
    t = t - t[0]

    out = {
        "session": args.session.name,
        "source": args.source,
        "csv_file": filename,
        "n_samples": int(len(t)),
        "duration_s": round(float(t[-1]), 2) if len(t) else 0.0,
        "n_nan": n_nan,
        "raw": metrics(t, q),
    }
    if args.source == "smooth-command":
        qs, _ = accel_limited_command(q, 250.0, 0.0025, 8.0)
        out["smoothed"] = metrics(t, qs)

        def _reduction(key: str) -> float:
            r = out["raw"][key]
            s = out["smoothed"][key]
            return round(100.0 * (r - s) / r, 1) if r else 0.0

        # jerk is the primary vibration proxy; peak speed and discontinuity
        # count are secondary. (n_large_steps alone is ~0 at 250 Hz, so basing
        # the headline reduction on it always read 0 — misleading.)
        out["jerk_reduction_pct"] = _reduction("jerk_metric")
        out["peak_speed_reduction_pct"] = _reduction("peak_speed_rad_s")
        out["vibration_reduction_pct"] = out["jerk_reduction_pct"]

    print(json.dumps(out))


if __name__ == "__main__":
    main()
