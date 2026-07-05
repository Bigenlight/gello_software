#!/usr/bin/env python3
"""fix1-refresh-target experiment.

Validates FIX #1 from STARTUP_JERK_DIAGNOSIS.md: refresh the move-to-start
target to the LIVE gello pose immediately before sending the 5s trajectory
(instead of the pose latched at t=0, which can age through the whole
Play-press wait, potentially minutes).

Baseline: gap G drawn from the full-window drift regime measured in the CSVs
  (0.15-0.30 rad range; includes Play-wait + 5s move + arrival slack).
Fixed:    gap G drawn from the residual 5s-move-only drift regime
  (0.05-0.10 rad; only the trajectory-duration window remains, Play-wait
  aging is eliminated by the one-line re-latch fix).

For each G we run the REAL bridge math (_OneEuro import from the actual node)
via sim_bridge_snap.simulate(), with default production params:
  filter_type=one_euro, max_step_rad=0.0025, publish_rate_hz=250 Hz,
  min_cutoff=1.0, beta=2.0, d_cutoff=1.0 (yaml defaults),
  tremor_std=0.003 rad, noise_std=0.0015 rad (per diagnosis doc: hand tremor
  Gaussian sigma=0.003 rad + ~0.00153 rad Dynamixel quantization noise),
  continuing to jitter around the frozen+G raw target during the sim window
  (models the leader still drifting/trembling live while bridge streams).
"""
import sys
sys.path.insert(0, "/home/theo/gello_software/scripts")
import sim_bridge_snap as sbs
from sim_bridge_snap import simulate
import numpy as np
import csv
import math

CEILING = 0.0025 * 250.0  # 0.625 rad/s


def traced_simulate(drift, *, filter_type="one_euro", max_step_rad=0.0025,
                     publish_rate_hz=250.0, ema_alpha=0.4, deadband_rad=0.004,
                     min_cutoff=1.0, beta=2.0, d_cutoff=1.0,
                     gello_rate_hz=30.0, tremor_std=0.0, noise_std=0.0,
                     sim_s=3.0, rng=None):
    """Exact copy of sim_bridge_snap.simulate()'s loop, but also returns the
    full published-command trace so we can compute a physically-motivated
    'time spent in the high-speed catch-up sweep' duration metric (the
    built-in settle_s is a 5%-relative-band metric that gets swamped by
    tremor/noise floor for small drift and is not a clean duration proxy)."""
    rng = rng or np.random.default_rng(0)
    dt_pub = 1.0 / publish_rate_hz
    n = int(sim_s * publish_rate_hz)
    frozen = 0.0
    euro = sbs._OneEuro(dt_pub, min_cutoff, beta, d_cutoff)
    euro.seed(frozen)
    filtered = frozen
    last_pub = frozen
    gated = frozen
    pub_per_gello = max(1, int(round(publish_rate_hz / gello_rate_hz)))
    raw_target = frozen + drift
    last_gello_input = None
    published = []
    first = True
    for k in range(n):
        if k % pub_per_gello == 0:
            j = rng.normal(0, tremor_std) + rng.normal(0, noise_std)
            raw_target = frozen + drift + j
            if euro is not None:
                dt_g = None if last_gello_input is None else pub_per_gello * dt_pub
                euro.update_input(raw_target, dt_g)
            last_gello_input = raw_target
        if first:
            gated = raw_target
            first = False
        if filter_type == "one_euro":
            filtered = euro(raw_target)
        else:
            if abs(raw_target - gated) > deadband_rad:
                gated = raw_target
            filtered = (1.0 - ema_alpha) * filtered + ema_alpha * gated
        delta = filtered - last_pub
        delta = max(-max_step_rad, min(max_step_rad, delta))
        last_pub = last_pub + delta
        published.append(last_pub)
    published = np.array(published)
    speed = np.abs(np.diff(published)) * publish_rate_hz
    return published, speed

COMMON = dict(
    filter_type="one_euro",
    max_step_rad=0.0025,
    publish_rate_hz=250.0,
    min_cutoff=1.0,
    beta=2.0,
    d_cutoff=1.0,
    gello_rate_hz=30.0,
    tremor_std=0.003,
    noise_std=0.0015,
    sim_s=3.0,
)

BASELINE_G = [0.15, 0.20, 0.25, 0.30]   # full Play-wait+5s-window drift (measured range + doc's swept ceiling)
FIX_G      = [0.05, 0.075, 0.10]        # residual 5s-only drift after fix #1

rows = []


def run_group(label, g_list, rng_seed_base):
    for i, g in enumerate(g_list):
        rng = np.random.default_rng(rng_seed_base + i)
        r = simulate(g, rng=rng, **COMMON)
        # Re-run with an identical rng (same seed) to get the traced arrays
        # for the duration-at-high-speed metric (deterministic given the seed).
        rng2 = np.random.default_rng(rng_seed_base + i)
        published, speed = traced_simulate(g, rng=rng2, **COMMON)
        theoretical_slew_s = abs(g) / CEILING
        below_ceiling = r["peak_speed_rad_s"] < CEILING - 1e-9
        # Physical "snap duration": time spent moving at >=90% of the slew
        # ceiling (the visually-obvious fast sweep), independent of the
        # noise-floor-sensitive 5%-relative settle_s metric.
        dt_pub = 1.0 / COMMON["publish_rate_hz"]
        n_fast = int(np.sum(speed >= 0.9 * CEILING))
        fast_duration_s = round(n_fast * dt_pub, 4)
        rows.append({
            "scenario": label,
            "drift_rad": r["drift_rad"],
            "drift_deg": r["drift_deg"],
            "peak_speed_rad_s": r["peak_speed_rad_s"],
            "peak_speed_deg_s": r["peak_speed_deg_s"],
            "settle_s": r["settle_s"],
            "theoretical_slew_s": round(theoretical_slew_s, 4),
            "fast_sweep_duration_s": fast_duration_s,
            "below_0.625_ceiling": below_ceiling,
            "final_err_rad": r["final_err_rad"],
        })


run_group("baseline_full_window", BASELINE_G, rng_seed_base=100)
run_group("fix1_5s_only", FIX_G, rng_seed_base=200)

out_csv = "/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/fix1.csv"
fieldnames = list(rows[0].keys())
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# --- Summary stats ---
base_rows = [r for r in rows if r["scenario"] == "baseline_full_window"]
fix_rows = [r for r in rows if r["scenario"] == "fix1_5s_only"]

base_settle = [r["settle_s"] for r in base_rows]
fix_settle = [r["settle_s"] for r in fix_rows]
base_peak = [r["peak_speed_rad_s"] for r in base_rows]
fix_peak = [r["peak_speed_rad_s"] for r in fix_rows]

print("=== baseline (full Play-wait + 5s window drift) ===")
for r in base_rows:
    print(r)
print()
print("=== fix1 (residual 5s-only drift) ===")
for r in fix_rows:
    print(r)
print()

mean_base_settle = sum(base_settle) / len(base_settle)
mean_fix_settle = sum(fix_settle) / len(fix_settle)
mean_base_peak = sum(base_peak) / len(base_peak)
mean_fix_peak = sum(fix_peak) / len(fix_peak)
base_fast = [r["fast_sweep_duration_s"] for r in base_rows]
fix_fast = [r["fast_sweep_duration_s"] for r in fix_rows]
mean_base_fast = sum(base_fast) / len(base_fast)
mean_fix_fast = sum(fix_fast) / len(fix_fast)
base_theo = [r["theoretical_slew_s"] for r in base_rows]
fix_theo = [r["theoretical_slew_s"] for r in fix_rows]
mean_base_theo = sum(base_theo) / len(base_theo)
mean_fix_theo = sum(fix_theo) / len(fix_theo)

reduction_settle_pct = 100.0 * (1 - mean_fix_settle / mean_base_settle)
reduction_peak_pct = 100.0 * (1 - mean_fix_peak / mean_base_peak)
reduction_fast_pct = 100.0 * (1 - mean_fix_fast / mean_base_fast)
reduction_theo_pct = 100.0 * (1 - mean_fix_theo / mean_base_theo)

n_fix_below_ceiling = sum(1 for r in fix_rows if r["below_0.625_ceiling"])
n_base_below_ceiling = sum(1 for r in base_rows if r["below_0.625_ceiling"])

print(f"mean baseline settle_s (5%-band metric) = {mean_base_settle:.4f}  (range {min(base_settle):.3f}-{max(base_settle):.3f})")
print(f"mean fix1     settle_s (5%-band metric) = {mean_fix_settle:.4f}  (range {min(fix_settle):.3f}-{max(fix_settle):.3f})")
print(f"  -> CAVEAT: settle_s is a relative-5%-band metric; eps=0.05*|G| shrinks")
print(f"     proportionally with G, so for small residual G it becomes comparable")
print(f"     to the tremor/noise floor (std=0.003+0.0015 rad) and the metric stops")
print(f"     being a clean duration proxy -- it does NOT show the expected drop.")
print(f"  snap-duration 'reduction' by this metric: {reduction_settle_pct:+.1f}% (NOT meaningful, see caveat)")
print()
print(f"mean baseline theoretical_slew_s (G/0.625, pure clamp-ramp time) = {mean_base_theo:.4f} s")
print(f"mean fix1     theoretical_slew_s (G/0.625, pure clamp-ramp time) = {mean_fix_theo:.4f} s")
print(f"theoretical ramp-time reduction: {reduction_theo_pct:.1f}%")
print()
print(f"mean baseline fast_sweep_duration_s (time at >=90% ceiling speed) = {mean_base_fast:.4f} s")
print(f"mean fix1     fast_sweep_duration_s (time at >=90% ceiling speed) = {mean_fix_fast:.4f} s")
print(f"fast-sweep duration reduction (physical duration metric): {reduction_fast_pct:.1f}%")
print()
print(f"mean baseline peak_speed_rad_s = {mean_base_peak:.4f}  ({math.degrees(mean_base_peak):.1f} deg/s)")
print(f"mean fix1     peak_speed_rad_s = {mean_fix_peak:.4f}  ({math.degrees(mean_fix_peak):.1f} deg/s)")
print(f"peak-speed reduction: {reduction_peak_pct:.1f}%")
print()
print(f"ceiling = {CEILING} rad/s ({math.degrees(CEILING):.1f} deg/s)")
print(f"baseline scenarios pegged AT/near ceiling (>=0.625, clamp binding): {len(base_rows)-n_base_below_ceiling}/{len(base_rows)}")
print(f"fix1     scenarios BELOW ceiling (clamp no longer binding): {n_fix_below_ceiling}/{len(fix_rows)}")
print()
print(f"CSV written: {out_csv}")
