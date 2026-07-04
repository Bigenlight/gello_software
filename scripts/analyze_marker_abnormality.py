#!/usr/bin/env python3
"""Rank marker windows by command velocity/acceleration/jerk metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ur_command_smoothing import accel_limited_command


def peak_percentile(rolling_peak: np.ndarray, idx: int) -> tuple[float, int, int]:
    finite = rolling_peak[np.isfinite(rolling_peak)]
    value = rolling_peak[idx]
    percentile = 100.0 * float(np.mean(finite <= value))
    rank = int(np.sum(finite > value) + 1)
    return percentile, rank, len(finite)


def derivatives(q: np.ndarray, rate_hz: float) -> dict[str, np.ndarray]:
    v = np.r_[np.full((1, q.shape[1]), np.nan), np.diff(q, axis=0) * rate_hz]
    a = np.r_[np.full((1, q.shape[1]), np.nan), np.diff(v, axis=0) * rate_hz]
    j = np.r_[np.full((1, q.shape[1]), np.nan), np.diff(a, axis=0) * rate_hz]
    step = np.r_[np.full((1, q.shape[1]), np.nan), np.abs(np.diff(q, axis=0))]
    return {"vel": v, "acc": a, "jerk": j, "step": step}


def rolling_peaks(signals: dict[str, np.ndarray], window_samples: int) -> dict[str, np.ndarray]:
    peaks = {}
    for name, values in signals.items():
        per_sample = np.nanmax(np.abs(values), axis=1)
        peaks[name] = (
            pd.Series(per_sample)
            .rolling(window=window_samples, min_periods=1)
            .max()
            .to_numpy(float)
        )
    return peaks


def analyze(args: argparse.Namespace) -> None:
    session = args.session
    command = pd.read_csv(session / "command.csv")
    markers = pd.read_csv(session / f"markers_{args.marker_source}.csv")

    t = command["t_rel_s"].to_numpy(float)
    q = command[[f"cmd{i}" for i in range(1, 7)]].to_numpy(float)
    smooth_q, _ = accel_limited_command(
        q,
        args.rate_hz,
        args.max_step_rad,
        args.max_accel_rad_s2,
    )

    window_samples = max(1, int(round(args.window_s * args.rate_hz)))
    raw_peaks = rolling_peaks(derivatives(q, args.rate_hz), window_samples)
    smooth_peaks = rolling_peaks(derivatives(smooth_q, args.rate_hz), window_samples)

    marker_t_col = (
        "source_sample_t_rel_s"
        if "source_sample_t_rel_s" in markers
        else "source_t_rel_s"
    )
    rows = []
    for marker_idx, marker in markers.iterrows():
        marker_t = float(marker[marker_t_col])
        idx = int(np.searchsorted(t, marker_t, side="right") - 1)
        idx = max(0, min(idx, len(t) - 1))
        row = {"marker": marker_idx + 1, "t_s": marker_t}
        for metric in ("vel", "acc", "jerk", "step"):
            raw_pct, raw_rank, total = peak_percentile(raw_peaks[metric], idx)
            smooth_pct, smooth_rank, _ = peak_percentile(smooth_peaks[metric], idx)
            row[f"raw_{metric}"] = raw_peaks[metric][idx]
            row[f"raw_{metric}_pct"] = raw_pct
            row[f"raw_{metric}_rank"] = raw_rank
            row[f"smooth_{metric}"] = smooth_peaks[metric][idx]
            row[f"smooth_{metric}_pct"] = smooth_pct
            row[f"smooth_{metric}_rank"] = smooth_rank
            row["windows"] = total
        rows.append(row)

    df = pd.DataFrame(rows)
    print(
        df[
            [
                "marker",
                "t_s",
                "raw_vel_pct",
                "smooth_vel_pct",
                "raw_acc_pct",
                "smooth_acc_pct",
                "raw_jerk_pct",
                "smooth_jerk_pct",
                "raw_step_pct",
                "smooth_step_pct",
            ]
        ]
        .round(1)
        .to_string(index=False)
    )

    print("\nTop-window counts among markers:")
    for metric in ("vel", "acc", "jerk", "step"):
        raw_top = int((df[f"raw_{metric}_pct"] >= args.top_percentile).sum())
        smooth_top = int((df[f"smooth_{metric}_pct"] >= args.top_percentile).sum())
        print(
            f"{metric:>5}: raw {raw_top}/{len(df)} >= p{args.top_percentile:g}, "
            f"smooth {smooth_top}/{len(df)} >= p{args.top_percentile:g}"
        )

    print("\nPeak value means over marked windows:")
    for metric in ("vel", "acc", "jerk", "step"):
        raw_mean = float(df[f"raw_{metric}"].mean())
        smooth_mean = float(df[f"smooth_{metric}"].mean())
        reduction = 100.0 * (1.0 - smooth_mean / raw_mean) if raw_mean else 0.0
        print(f"{metric:>5}: raw={raw_mean:.6g}, smooth={smooth_mean:.6g}, reduction={reduction:.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--marker-source", default="smooth-command")
    parser.add_argument("--window-s", type=float, default=0.6)
    parser.add_argument("--rate-hz", type=float, default=250.0)
    parser.add_argument("--max-step-rad", type=float, default=0.0025)
    parser.add_argument("--max-accel-rad-s2", type=float, default=8.0)
    parser.add_argument("--top-percentile", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
