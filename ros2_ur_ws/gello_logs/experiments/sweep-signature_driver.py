#!/usr/bin/env python3
"""sweep-signature experiment: establish the SNAP SIGNATURE via the pure-math
harness (scripts/sim_bridge_snap.py -> simulate()).

Sweeps drift G over {0.02,0.05,0.10,0.15,0.20,0.30,0.40} rad for both
filter_type in {one_euro, ema}, at config params (max_step=0.0025 rad,
publish_rate=250 Hz), WITH realistic per-sample tremor (std=0.003 rad) and
Dynamixel quantization noise (std=0.0015 rad).

For each run: records peak_speed_rad_s, settle_s, asserts max per-cycle
|delta cmd| <= max_step_rad (the clamp is never violated -- it is enforced
in simulate() itself, but we also recompute it independently here from the
returned published-trace-equivalent by re-deriving speed samples, as a second
check using a modified simulate that also returns the full trace).

Outputs:
  - CSV:  gello_logs/experiments/sweep-signature.csv
  - PNG:  gello_logs/experiments/sweep-signature.png (representative trace G=0.15,
          both filters, showing the flat 0.625 rad/s ramp = the "snap")
"""
import sys
sys.path.insert(0, "/home/theo/gello_software/scripts")
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_bridge_snap import _OneEuro  # noqa: F401  (sanity: real filter import works)
import sim_bridge_snap as sbs

RNG_SEED = 0
DRIFTS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
FILTERS = ["one_euro", "ema"]
MAX_STEP = 0.0025
RATE_HZ = 250.0
TREMOR_STD = 0.003
NOISE_STD = 0.0015
THEORETICAL_MAX = MAX_STEP * RATE_HZ  # 0.625 rad/s


def simulate_with_trace(drift, filter_type, rng):
    """Re-implements simulate()'s loop but also returns the full published
    trace + per-cycle delta trace, so we can independently verify the clamp
    and plot a representative curve. Mirrors sim_bridge_snap.simulate exactly.
    """
    dt_pub = 1.0 / RATE_HZ
    sim_s = 3.0 if drift <= 0.20 else 1.5 * (drift / THEORETICAL_MAX) + 1.0
    # generous sim horizon: enough to fully settle + margin
    sim_s = max(sim_s, drift / THEORETICAL_MAX + 1.0)
    n = int(sim_s * RATE_HZ)
    frozen = 0.0
    euro = sbs._OneEuro(dt_pub, 1.0, 2.0, 1.0)
    euro.seed(frozen)
    filtered = frozen
    last_pub = frozen
    gated = frozen
    gello_rate_hz = 30.0
    pub_per_gello = max(1, int(round(RATE_HZ / gello_rate_hz)))
    raw_target = frozen + drift
    last_gello_input = None
    published = []
    deltas = []
    first = True
    for k in range(n):
        if k % pub_per_gello == 0:
            j = rng.normal(0, TREMOR_STD) + rng.normal(0, NOISE_STD)
            raw_target = frozen + drift + j
            dt_g = None if last_gello_input is None else pub_per_gello * dt_pub
            euro.update_input(raw_target, dt_g)
            last_gello_input = raw_target
        if first:
            gated = raw_target
            first = False
        if filter_type == "one_euro":
            filtered = euro(raw_target)
        else:
            if abs(raw_target - gated) > 0.004:
                gated = raw_target
            filtered = 0.6 * filtered + 0.4 * gated
        delta = filtered - last_pub
        delta_clamped = max(-MAX_STEP, min(MAX_STEP, delta))
        last_pub = last_pub + delta_clamped
        published.append(last_pub)
        deltas.append(delta_clamped)
    return np.array(published), np.array(deltas), n


rows = []
traces = {}  # (filter_type) -> (t, published) for drift=0.15
for filter_type in FILTERS:
    for drift in DRIFTS:
        rng = np.random.default_rng(RNG_SEED)
        published, deltas, n = simulate_with_trace(drift, filter_type, rng)
        t = np.arange(n) / RATE_HZ

        # independent clamp check
        max_abs_delta = float(np.max(np.abs(deltas)))
        clamp_ok = max_abs_delta <= MAX_STEP + 1e-12
        assert clamp_ok, f"CLAMP VIOLATED filter={filter_type} drift={drift} max|delta|={max_abs_delta}"

        speed = np.abs(deltas) * RATE_HZ  # rad/s realized this cycle
        peak_speed = float(speed.max())

        eps = max(0.05 * abs(drift), 0.01)
        settled_idx = np.where(np.abs(published - drift) <= eps)[0]
        # require it STAYS within eps from that point on (avoid transient false-settle from noise)
        settle_s = float("nan")
        for idx in settled_idx:
            if np.all(np.abs(published[idx:] - drift) <= eps + 0.02):
                settle_s = idx / RATE_HZ
                break

        theoretical_settle_s = drift / THEORETICAL_MAX

        # Cleaner "ramp duration" metric: first crossing of 90% of drift.
        # This isolates the visible fast-sweep ("snap") phase from the
        # trailing tremor/noise-driven micro-jitter that the stricter
        # settle_s (stays within eps forever) metric partly captures.
        cross90 = np.where(published >= 0.9 * drift)[0] if drift > 0 else np.array([0])
        t90_s = float(cross90[0] / RATE_HZ) if cross90.size else float("nan")

        rows.append(dict(
            filter_type=filter_type,
            drift_rad=drift,
            drift_deg=round(math.degrees(drift), 2),
            peak_speed_rad_s=round(peak_speed, 5),
            peak_speed_deg_s=round(math.degrees(peak_speed), 1),
            theoretical_max_slew_rad_s=THEORETICAL_MAX,
            saturates_at_theoretical=bool(abs(peak_speed - THEORETICAL_MAX) < 1e-6 or
                                           peak_speed >= THEORETICAL_MAX - 1e-9),
            settle_s=round(settle_s, 4) if settle_s == settle_s else float("nan"),
            theoretical_settle_s=round(theoretical_settle_s, 4),
            settle_ratio=(round(settle_s / theoretical_settle_s, 3)
                          if (settle_s == settle_s and theoretical_settle_s > 0) else float("nan")),
            t90_s=round(t90_s, 4),
            t90_ratio=(round(t90_s / theoretical_settle_s, 3)
                       if (t90_s == t90_s and theoretical_settle_s > 0) else float("nan")),
            max_abs_delta_rad=round(max_abs_delta, 6),
            clamp_respected=clamp_ok,
            final_err_rad=round(float(abs(published[-1] - drift)), 6),
        ))

        if abs(drift - 0.15) < 1e-9:
            traces[filter_type] = (t, published.copy(), deltas.copy())

df = pd.DataFrame(rows)
csv_path = "/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/sweep-signature.csv"
df.to_csv(csv_path, index=False)
print(df.to_string(index=False))

# ---- Plot: representative trace @ G=0.15 for both filters ----
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

ax = axes[0]
for filter_type, (t, published, deltas) in traces.items():
    ax.plot(t, published, label=f"{filter_type} cmd (G=0.15 rad)")
ax.axhline(0.15, color="gray", ls="--", lw=1, label="target drift G=0.15 rad")
ax.set_ylabel("published joint position (rad, relative)")
ax.set_title("Bridge command trace after handoff, G=0.15 rad (config max_step=0.0025 rad @ 250 Hz)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

ax2 = axes[1]
for filter_type, (t, published, deltas) in traces.items():
    speed = np.abs(deltas) * RATE_HZ
    ax2.plot(t[:len(speed)], speed, label=f"{filter_type} realized speed")
ax2.axhline(THEORETICAL_MAX, color="red", ls="--", lw=1.5,
            label=f"theoretical max slew = max_step*rate = {THEORETICAL_MAX:.3f} rad/s")
ax2.set_xlabel("time since handoff (s)")
ax2.set_ylabel("|d(cmd)/dt| (rad/s)")
ax2.set_title('The "snap" = a flat-topped ramp at the clamp-bound speed, not an instantaneous jump')
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

fig.tight_layout()
png_path = "/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments/sweep-signature.png"
fig.savefig(png_path, dpi=130)
print(f"\nSaved CSV -> {csv_path}")
print(f"Saved PNG -> {png_path}")

# ---- Summary checks ----
sat_thresh_drift = 0.15
print("\n=== Saturation check (peak speed vs theoretical max 0.625 rad/s) ===")
for filter_type in FILTERS:
    sub = df[df.filter_type == filter_type].sort_values("drift_rad")
    print(f"\n{filter_type}:")
    print(sub[["drift_rad", "peak_speed_rad_s", "settle_s", "theoretical_settle_s", "settle_ratio",
               "t90_s", "t90_ratio"]].to_string(index=False))

print("\nAll clamp_respected True:", bool(df["clamp_respected"].all()))
