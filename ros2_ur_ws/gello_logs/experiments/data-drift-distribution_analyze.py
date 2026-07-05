#!/usr/bin/env python3
"""data-drift-distribution experiment.

For both recorded GELLO sessions, slide a 5s window across the whole
recording (step = one sample, i.e. dense sliding window at native ~30Hz
sample rate) and compute per-joint peak-to-peak (max-min) drift within
each window. Report per-joint distribution (median, 90th pct, max) in
rad and deg. Also find the single quietest 5s window (minimum total
motion, defined as sum of per-joint ptp across all 6 joints) as a proxy
for "operator holding still during move-to-start", and report its
per-joint drift.

Then feed median and 90th-pct drift (max over joints, since the bridge
mechanism is per-joint and the worst joint sets the visible snap) into
the harness sim_bridge_snap.simulate() to predict resulting snap peak
speed and duration for both filters (one_euro default, ema deadband).

Outputs:
  - gello_logs/experiments/data-drift.csv  (per-window per-joint ptp table
    is large; instead we write the SUMMARY table: one row per
    session x joint with median/p90/max, plus quietest-window rows,
    plus harness-prediction rows)
  - gello_logs/experiments/data-drift_windows_<session>.csv (raw sliding
    window per-joint ptp, for provenance / plotting)
  - gello_logs/experiments/data-drift_hist.png (histogram of window ptp
    per session, max-over-joints)
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/theo/gello_software")
SESS = {
    "session_20260703_165530": REPO / "ros2_ur_ws/gello_logs/session_20260703_165530/gello_joint_states.csv",
    "session_20260703_171323": REPO / "ros2_ur_ws/gello_logs/session_20260703_171323/gello_joint_states.csv",
}
OUT_DIR = REPO / "ros2_ur_ws/gello_logs/experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_S = 5.0
JOINTS = [f"q{i}" for i in range(1, 7)]

sys.path.insert(0, str(REPO / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("sim_bridge_snap", REPO / "scripts/sim_bridge_snap.py")
sim_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_mod)
simulate = sim_mod.simulate


def sliding_window_ptp(df: pd.DataFrame, window_s: float):
    """Dense sliding window (every sample as a window start) -> per-joint ptp array (n_windows, n_joints)."""
    t = df["t_rel_s"].to_numpy()
    q = df[JOINTS].to_numpy()
    n = len(t)
    starts = []
    ptp = []
    i0 = 0
    # For each candidate start index i0, window = [t[i0], t[i0]+window_s]
    for i0 in range(n):
        t_end = t[i0] + window_s
        if t_end > t[-1]:
            break
        i1 = np.searchsorted(t, t_end, side="right") - 1
        if i1 <= i0:
            continue
        seg = q[i0:i1 + 1]
        this_ptp = seg.max(axis=0) - seg.min(axis=0)
        starts.append(t[i0])
        ptp.append(this_ptp)
    return np.array(starts), np.array(ptp)  # (n_windows,), (n_windows, 6)


summary_rows = []
quiet_rows = []
window_dfs = {}

for sess_name, csv_path in SESS.items():
    df = pd.read_csv(csv_path)
    duration = df["t_rel_s"].iloc[-1] - df["t_rel_s"].iloc[0]
    n_samples = len(df)
    print(f"[{sess_name}] duration={duration:.2f}s n_samples={n_samples} "
          f"nominal_rate={n_samples/duration:.1f}Hz")

    starts, ptp = sliding_window_ptp(df, WINDOW_S)
    n_windows = len(starts)
    print(f"  -> {n_windows} sliding 5s windows")

    wdf = pd.DataFrame(ptp, columns=[f"{j}_ptp_rad" for j in JOINTS])
    wdf.insert(0, "window_start_t_rel_s", starts)
    wdf["total_ptp_rad_sum"] = ptp.sum(axis=1)
    wdf["max_joint_ptp_rad"] = ptp.max(axis=1)
    window_dfs[sess_name] = wdf
    wdf.to_csv(OUT_DIR / f"data-drift_windows_{sess_name}.csv", index=False)

    for ji, j in enumerate(JOINTS):
        col = ptp[:, ji]
        summary_rows.append({
            "session": sess_name,
            "joint": j,
            "n_windows": n_windows,
            "median_rad": float(np.median(col)),
            "p90_rad": float(np.percentile(col, 90)),
            "max_rad": float(np.max(col)),
            "median_deg": float(np.degrees(np.median(col))),
            "p90_deg": float(np.degrees(np.percentile(col, 90))),
            "max_deg": float(np.degrees(np.max(col))),
        })

    # quietest window = min total motion (sum of per-joint ptp)
    qi = int(np.argmin(wdf["total_ptp_rad_sum"].to_numpy()))
    qrow = wdf.iloc[qi]
    print(f"  quietest window start t={qrow['window_start_t_rel_s']:.3f}s "
          f"total_ptp_sum={qrow['total_ptp_rad_sum']:.4f} rad "
          f"max_joint_ptp={qrow['max_joint_ptp_rad']:.4f} rad")
    for j in JOINTS:
        quiet_rows.append({
            "session": sess_name,
            "joint": j,
            "quietest_window_start_t_rel_s": float(qrow["window_start_t_rel_s"]),
            "ptp_rad": float(qrow[f"{j}_ptp_rad"]),
            "ptp_deg": float(np.degrees(qrow[f"{j}_ptp_rad"])),
        })

summary_df = pd.DataFrame(summary_rows)
quiet_df = pd.DataFrame(quiet_rows)

print("\n=== Per-joint 5s-window drift distribution (rad) ===")
print(summary_df.to_string(index=False))

print("\n=== Quietest 5s window per session (per-joint drift) ===")
print(quiet_df.to_string(index=False))

# ---- Aggregate across both sessions: overall median / p90 / max of
# max-over-joints-per-window (this is what actually determines "does ANY
# joint snap visibly" since the bridge clamps each joint independently and
# a human will visually notice whichever joint moves the most).
all_max_joint_ptp = np.concatenate([
    window_dfs[s]["max_joint_ptp_rad"].to_numpy() for s in SESS
])
overall_median = float(np.median(all_max_joint_ptp))
overall_p90 = float(np.percentile(all_max_joint_ptp, 90))
overall_max = float(np.max(all_max_joint_ptp))
print(f"\n=== Pooled across both sessions: max-joint-ptp per 5s window ===")
print(f"median={overall_median:.4f} rad ({math.degrees(overall_median):.2f} deg)")
print(f"p90   ={overall_p90:.4f} rad ({math.degrees(overall_p90):.2f} deg)")
print(f"max   ={overall_max:.4f} rad ({math.degrees(overall_max):.2f} deg)")

# Also pooled quietest-window max-joint-ptp (best proxy for "hold still")
quiet_max_joint = quiet_df.groupby("session")["ptp_rad"].max()
print("\nQuietest-window max-joint ptp per session (rad):")
print(quiet_max_joint.to_string())
best_quiet = float(quiet_max_joint.min())
print(f"Best ('most-still') quietest-window max-joint ptp across sessions: "
      f"{best_quiet:.4f} rad ({math.degrees(best_quiet):.2f} deg)")

# ---- Feed median / p90 drift into the harness for both filters ----
harness_rows = []
for label, G in [
    ("median_pooled", overall_median),
    ("p90_pooled", overall_p90),
    ("max_pooled", overall_max),
    ("quietest_window_best", best_quiet),
]:
    for filt in ("one_euro", "ema"):
        res = simulate(G, filter_type=filt, max_step_rad=0.0025,
                        publish_rate_hz=250.0, sim_s=3.0)
        res["drift_label"] = label
        harness_rows.append(res)

harness_df = pd.DataFrame(harness_rows)
print("\n=== Harness predictions (max_step_rad=0.0025, rate=250Hz) ===")
print(harness_df.to_string(index=False))

# ---- Write consolidated data-drift.csv ----
# Combine: per-joint summary stats, quietest-window rows, pooled stats, and
# harness predictions, each tagged by a 'record_type' column so it's one file.
out_rows = []
for r in summary_rows:
    r2 = dict(r)
    r2["record_type"] = "per_joint_summary"
    out_rows.append(r2)
for r in quiet_rows:
    r2 = dict(r)
    r2["record_type"] = "quietest_window"
    out_rows.append(r2)
out_rows.append({"record_type": "pooled_max_joint_ptp", "session": "both",
                  "joint": "max_over_joints", "median_rad": overall_median,
                  "p90_rad": overall_p90, "max_rad": overall_max,
                  "median_deg": math.degrees(overall_median),
                  "p90_deg": math.degrees(overall_p90),
                  "max_deg": math.degrees(overall_max)})
for r in harness_rows:
    r2 = dict(r)
    r2["record_type"] = "harness_prediction"
    out_rows.append(r2)

out_df = pd.DataFrame(out_rows)
out_csv = OUT_DIR / "data-drift.csv"
out_df.to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")

# ---- Histogram plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (sess_name, wdf) in zip(axes, window_dfs.items()):
    ax.hist(wdf["max_joint_ptp_rad"], bins=40, color="steelblue", alpha=0.85)
    ax.axvline(np.median(wdf["max_joint_ptp_rad"]), color="orange", ls="--", label="median")
    ax.axvline(np.percentile(wdf["max_joint_ptp_rad"], 90), color="red", ls="--", label="p90")
    ax.axvline(0.1, color="green", ls=":", label="0.1 rad threshold")
    ax.set_title(sess_name, fontsize=9)
    ax.set_xlabel("max-over-joints 5s-window ptp (rad)")
    ax.set_ylabel("count of windows")
    ax.legend(fontsize=7)
fig.suptitle("Distribution of 5s-window peak-to-peak drift (max over 6 joints)")
fig.tight_layout()
png_path = OUT_DIR / "data-drift_hist.png"
fig.savefig(png_path, dpi=140)
print(f"Wrote {png_path}")
