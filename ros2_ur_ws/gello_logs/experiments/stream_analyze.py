#!/usr/bin/env python3
"""Analyze a stream_*.csv (/joint_states during post-switch streaming) and
report snap-relevant metrics per joint:
  raw_peak    = max |dq/dt| over adjacent 2ms samples (noise-sensitive)
  win_peak_50 = max |dq/dt| over a 50ms sliding window (rejects 1-sample spikes;
                a real catch-up SWEEP shows here, oscillation jitter averages out)
  excursion   = max(q) - min(q) over the whole window (net travel; a SNAP moves
                the joint a long way, an oscillation stays put)
A snap = large win_peak_50 AND large excursion. Jitter = large raw_peak only."""
import csv
import sys

UR = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def analyze(path, win_s=0.05):
    rows = list(csv.reader(open(path)))[1:]
    t = [float(r[0]) for r in rows]
    q = [[float(r[1 + j]) for j in range(6)] for r in rows]
    n = len(rows)
    raw = [0.0] * 6
    winp = [0.0] * 6
    exc = [0.0] * 6
    for j in range(6):
        col = [q[k][j] for k in range(n)]
        exc[j] = max(col) - min(col)
        for k in range(1, n):
            dt = t[k] - t[k - 1]
            if dt > 1e-6:
                raw[j] = max(raw[j], abs(col[k] - col[k - 1]) / dt)
        # sliding window speed
        lo = 0
        for hi in range(n):
            while t[hi] - t[lo] > win_s and lo < hi:
                lo += 1
            dt = t[hi] - t[lo]
            if dt >= win_s * 0.5:
                v = abs(col[hi] - col[lo]) / dt
                winp[j] = max(winp[j], v)
    return raw, winp, exc


def main():
    path = sys.argv[1]
    raw, winp, exc = analyze(path)
    print(f"=== {path} ({win_s_str()}) ===")
    for j in range(6):
        print(f"  {UR[j]:14s} raw_peak={raw[j]:.3f}  win50_peak={winp[j]:.3f}"
              f"  excursion={exc[j]:.4f} rad")
    print(f"MAX_RAW_PEAK_RAD_S={max(raw):.4f}")
    print(f"MAX_WIN50_PEAK_RAD_S={max(winp):.4f}")
    print(f"MAX_EXCURSION_RAD={max(exc):.4f}")


def win_s_str():
    return "50ms sliding window"


if __name__ == "__main__":
    main()
