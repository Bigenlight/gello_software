#!/usr/bin/env python3
"""fix2-catchup-and-softstart experiment.

Copies the validated `simulate()` math from scripts/sim_bridge_snap.py (which
itself imports the REAL _OneEuro filter from gello_ur_bridge_node.py) and adds
two candidate fixes for the post-handoff "snap":

  FIX #2 (catch-up trajectory): at t=0, instead of seeding the streaming bridge
    directly from the robot's actual arrival pose and clamping the raw gap G
    with max_step_rad forever, first issue ONE short, smooth catch-up motion
    (a raised-cosine / versine position profile, i.e. what a
    scaled_joint_trajectory_controller point-to-point move looks like) that
    covers most of the gap in `catchup_duration_s`. Because the live GELLO
    keeps drifting/trembling during that window, a small residual gap
    `catchup_residual_rad` (~0.02 rad) remains when normal streaming (One-Euro
    filter + max_step_rad clamp) takes over.

  SOFT-START (time-varying clamp): no catch-up trajectory at all -- the gap G
    is the full drift, exactly like baseline -- but max_step_rad(t) ramps
    linearly from `softstart_min_step_rad` up to the nominal `max_step_rad`
    over `softstart_ramp_s` seconds instead of being full-value from cycle 0.

Both are compared against the unmodified baseline (single-shot max_step clamp
of the full drift, no catch-up, no ramp) reproduced verbatim from
scripts/sim_bridge_snap.py::simulate().
"""
from __future__ import annotations
import argparse, importlib.util, json, math
from pathlib import Path
import numpy as np

NODE = Path("/home/theo/gello_software/ros2_ur_ws/src/ur_gello_bringup/"
            "ur_gello_bringup/gello_ur_bridge_node.py")
spec = importlib.util.spec_from_file_location("bridge_node", NODE)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    src = NODE.read_text()
    start = src.index("class _OneEuro")
    end = src.index("class GelloUrBridge")
    ns: dict = {"math": math}
    exec(compile(src[start:end], str(NODE), "exec"), ns)
    mod = type("m", (), ns)
_OneEuro = mod._OneEuro


def _streaming_phase(drift, *, filter_type, publish_rate_hz, ema_alpha, deadband_rad,
                      min_cutoff, beta, d_cutoff, gello_rate_hz, tremor_std, noise_std,
                      rng, n, seed_pos, max_step_fn, t0_s=0.0):
    """Run n cycles of the real _on_timer streaming math starting from seed_pos,
    with live target = seed_pos + drift (+ jitter), and a (possibly
    time-varying) max_step given by max_step_fn(cycle_index) -> rad/cycle.
    Returns arrays of published position and time (s, absolute)."""
    dt_pub = 1.0 / publish_rate_hz
    euro = _OneEuro(dt_pub, min_cutoff, beta, d_cutoff)
    euro.seed(seed_pos)
    filtered = seed_pos
    last_pub = seed_pos
    gated = seed_pos
    pub_per_gello = max(1, int(round(publish_rate_hz / gello_rate_hz)))
    raw_target = seed_pos + drift
    last_gello_input = None
    published = np.empty(n)
    times = np.empty(n)
    first = True
    for k in range(n):
        if k % pub_per_gello == 0:
            j = rng.normal(0, tremor_std) + rng.normal(0, noise_std)
            raw_target = seed_pos + drift + j
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
        step = max_step_fn(k)
        delta = max(-step, min(step, delta))
        last_pub = last_pub + delta
        published[k] = last_pub
        times[k] = t0_s + k * dt_pub
    return published, times


def _metrics(published, times, target_final, publish_rate_hz, label):
    speed = np.abs(np.diff(published)) * publish_rate_hz
    peak_speed = float(speed.max()) if speed.size else 0.0
    peak_idx = int(np.argmax(speed)) if speed.size else 0
    eps = 0.05 * abs(target_final) if target_final else 0.01
    settled = np.where(np.abs(published - target_final) <= eps)[0]
    settle_s = float(times[settled[0]] - times[0]) if settled.size else float("nan")
    return {
        "label": label,
        "peak_speed_rad_s": round(peak_speed, 4),
        "peak_speed_deg_s": round(math.degrees(peak_speed), 1),
        "peak_speed_t_s": round(float(times[peak_idx]), 4),
        "settle_s": round(settle_s, 3),
        "final_err_rad": round(float(abs(published[-1] - target_final)), 5),
    }


def simulate_baseline(drift, *, filter_type="one_euro", max_step_rad=0.0025,
                       publish_rate_hz=250.0, ema_alpha=0.4, deadband_rad=0.004,
                       min_cutoff=1.0, beta=2.0, d_cutoff=1.0, gello_rate_hz=30.0,
                       tremor_std=0.0, noise_std=0.0, sim_s=3.0, seed=0):
    """Unmodified bridge behaviour: seed from actual arrival pose (0.0), full
    max_step_rad clamp from cycle 0. Faithful reproduction of
    scripts/sim_bridge_snap.py::simulate()."""
    rng = np.random.default_rng(seed)
    n = int(sim_s * publish_rate_hz)
    published, times = _streaming_phase(
        drift, filter_type=filter_type, publish_rate_hz=publish_rate_hz,
        ema_alpha=ema_alpha, deadband_rad=deadband_rad, min_cutoff=min_cutoff,
        beta=beta, d_cutoff=d_cutoff, gello_rate_hz=gello_rate_hz,
        tremor_std=tremor_std, noise_std=noise_std, rng=rng, n=n,
        seed_pos=0.0, max_step_fn=lambda k: max_step_rad, t0_s=0.0)
    m = _metrics(published, times, drift, publish_rate_hz, "baseline")
    m["theoretical_max_slew_rad_s"] = round(max_step_rad * publish_rate_hz, 4)
    return published, times, m


def simulate_fix2_catchup(drift, *, filter_type="one_euro", max_step_rad=0.0025,
                           publish_rate_hz=250.0, ema_alpha=0.4, deadband_rad=0.004,
                           min_cutoff=1.0, beta=2.0, d_cutoff=1.0, gello_rate_hz=30.0,
                           tremor_std=0.0, noise_std=0.0, sim_s=3.0, seed=0,
                           catchup_duration_s=0.4, catchup_residual_rad=0.02):
    """FIX #2: after move-to-start, compare arrival pose (0.0) to the live
    GELLO pose (0.0+drift). If the gap exceeds a threshold, issue one smooth
    catch-up trajectory (raised-cosine / versine profile -- the same kind of
    smooth point-to-point motion a joint_trajectory_controller executes) that
    covers (drift - catchup_residual_rad) over catchup_duration_s. This is a
    SEPARATE, bounded, smooth motion -- not the per-cycle max_step_rad
    streaming clamp -- so its peak speed is governed by profile duration, not
    by the streaming slew limit. Once the catch-up trajectory completes,
    normal streaming (filter + max_step_rad clamp) takes over, but now the
    live GELLO has kept drifting during catchup_duration_s, so a residual gap
    of ~catchup_residual_rad remains for the streaming phase to close.
    """
    rng = np.random.default_rng(seed)
    dt_pub = 1.0 / publish_rate_hz
    n_catchup = max(1, int(round(catchup_duration_s * publish_rate_hz)))
    catchup_span = drift - catchup_residual_rad
    # raised-cosine (versine) position profile: s(t)=0.5*(1-cos(pi*t/T))*span
    t_c = (np.arange(n_catchup) + 1) * dt_pub
    pos_c = 0.5 * (1.0 - np.cos(math.pi * np.clip(t_c / catchup_duration_s, 0, 1))) * catchup_span
    times_c = t_c
    # streaming phase, starting AFTER catch-up, seeded from wherever catch-up ended
    seed_pos = float(pos_c[-1])
    n_stream = int(sim_s * publish_rate_hz) - n_catchup
    published_s, times_s = _streaming_phase(
        catchup_residual_rad, filter_type=filter_type, publish_rate_hz=publish_rate_hz,
        ema_alpha=ema_alpha, deadband_rad=deadband_rad, min_cutoff=min_cutoff,
        beta=beta, d_cutoff=d_cutoff, gello_rate_hz=gello_rate_hz,
        tremor_std=tremor_std, noise_std=noise_std, rng=rng, n=n_stream,
        seed_pos=seed_pos, max_step_fn=lambda k: max_step_rad, t0_s=catchup_duration_s)
    published = np.concatenate([pos_c, published_s])
    times = np.concatenate([times_c, times_s])
    m = _metrics(published, times, drift, publish_rate_hz, "fix2_catchup")
    # also report catch-up-phase-only peak speed to show its motion is smooth
    speed_c = np.abs(np.diff(pos_c)) * publish_rate_hz
    m["catchup_peak_speed_rad_s"] = round(float(speed_c.max()) if speed_c.size else 0.0, 4)
    m["catchup_peak_speed_deg_s"] = round(math.degrees(m["catchup_peak_speed_rad_s"]), 1)
    m["catchup_duration_s"] = catchup_duration_s
    m["catchup_residual_rad"] = catchup_residual_rad
    return published, times, m


def simulate_softstart(drift, *, filter_type="one_euro", max_step_rad=0.0025,
                        publish_rate_hz=250.0, ema_alpha=0.4, deadband_rad=0.004,
                        min_cutoff=1.0, beta=2.0, d_cutoff=1.0, gello_rate_hz=30.0,
                        tremor_std=0.0, noise_std=0.0, sim_s=3.0, seed=0,
                        softstart_ramp_s=1.0, softstart_min_step_rad=0.0002):
    """SOFT-START: no catch-up trajectory -- gap is the full drift, exactly
    like baseline -- but max_step_rad(t) ramps linearly from
    softstart_min_step_rad up to max_step_rad over softstart_ramp_s seconds,
    then holds at max_step_rad."""
    rng = np.random.default_rng(seed)
    n = int(sim_s * publish_rate_hz)
    ramp_n = max(1, int(round(softstart_ramp_s * publish_rate_hz)))

    def max_step_fn(k):
        if k >= ramp_n:
            return max_step_rad
        frac = k / ramp_n
        return softstart_min_step_rad + frac * (max_step_rad - softstart_min_step_rad)

    published, times = _streaming_phase(
        drift, filter_type=filter_type, publish_rate_hz=publish_rate_hz,
        ema_alpha=ema_alpha, deadband_rad=deadband_rad, min_cutoff=min_cutoff,
        beta=beta, d_cutoff=d_cutoff, gello_rate_hz=gello_rate_hz,
        tremor_std=tremor_std, noise_std=noise_std, rng=rng, n=n,
        seed_pos=0.0, max_step_fn=max_step_fn, t0_s=0.0)
    m = _metrics(published, times, drift, publish_rate_hz, "softstart")
    m["softstart_ramp_s"] = softstart_ramp_s
    m["softstart_min_step_rad"] = softstart_min_step_rad
    return published, times, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=float, default=0.2)
    ap.add_argument("--filter", dest="filter_type", default="one_euro")
    ap.add_argument("--max-step", dest="max_step_rad", type=float, default=0.0025)
    ap.add_argument("--rate", dest="publish_rate_hz", type=float, default=250.0)
    ap.add_argument("--tremor", dest="tremor_std", type=float, default=0.0)
    ap.add_argument("--noise", dest="noise_std", type=float, default=0.0)
    ap.add_argument("--sim-s", dest="sim_s", type=float, default=3.0)
    ap.add_argument("--catchup-duration", type=float, default=0.4)
    ap.add_argument("--catchup-residual", type=float, default=0.02)
    ap.add_argument("--softstart-ramp", type=float, default=1.0)
    ap.add_argument("--softstart-min-step", type=float, default=0.0002)
    ap.add_argument("--out-dir", default="/home/theo/gello_software/ros2_ur_ws/gello_logs/experiments")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    common = dict(filter_type=a.filter_type, max_step_rad=a.max_step_rad,
                   publish_rate_hz=a.publish_rate_hz, tremor_std=a.tremor_std,
                   noise_std=a.noise_std, sim_s=a.sim_s, seed=a.seed)

    pub_b, t_b, m_b = simulate_baseline(a.drift, **common)
    pub_f, t_f, m_f = simulate_fix2_catchup(
        a.drift, catchup_duration_s=a.catchup_duration,
        catchup_residual_rad=a.catchup_residual, **common)
    pub_s, t_s, m_s = simulate_softstart(
        a.drift, softstart_ramp_s=a.softstart_ramp,
        softstart_min_step_rad=a.softstart_min_step, **common)

    print(json.dumps({"baseline": m_b, "fix2_catchup": m_f, "softstart": m_s}, indent=2))

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: union of time grids (all share dt_pub, baseline/softstart start at
    # t=0 for n samples; fix2 spans catchup + streaming for the same n samples
    # at same dt so lengths match)
    import pandas as pd
    df = pd.DataFrame({
        "t_s": t_b,
        "baseline_q": pub_b,
        "fix2_catchup_q": pub_f,
        "softstart_q": pub_s,
    })
    df.to_csv(out_dir / "fix2.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax = axes[0]
    ax.plot(t_b, pub_b, label=f"baseline (G={a.drift} rad)", color="tab:red")
    ax.plot(t_f, pub_f, label=f"fix2 catch-up (G={a.drift}->resid {a.catchup_residual} rad)", color="tab:green")
    ax.plot(t_s, pub_s, label=f"soft-start (G={a.drift} rad, ramp {a.softstart_ramp}s)", color="tab:blue")
    ax.axhline(a.drift, color="gray", ls=":", lw=1, label="target (drift)")
    ax.set_ylabel("commanded joint position (rad, rel. to arrival pose)")
    ax.set_title("Post-handoff commanded position: baseline vs fix2 catch-up vs soft-start")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    speed_b = np.abs(np.diff(pub_b)) * a.publish_rate_hz
    speed_f = np.abs(np.diff(pub_f)) * a.publish_rate_hz
    speed_s = np.abs(np.diff(pub_s)) * a.publish_rate_hz
    ax2.plot(t_b[1:], speed_b, color="tab:red", label="baseline")
    ax2.plot(t_f[1:], speed_f, color="tab:green", label="fix2 catch-up")
    ax2.plot(t_s[1:], speed_s, color="tab:blue", label="soft-start")
    ax2.axhline(1.0, color="k", ls="--", lw=1, label="1 rad/s FPC-jump discriminator")
    ax2.axhline(a.max_step_rad * a.publish_rate_hz, color="gray", ls=":", lw=1,
                label=f"nominal max slew ({a.max_step_rad*a.publish_rate_hz:.3f} rad/s)")
    ax2.set_xlabel("time since handoff (s)")
    ax2.set_ylabel("|d(command)/dt| (rad/s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fix2.png", dpi=140)
    print(f"wrote {out_dir/'fix2.csv'} and {out_dir/'fix2.png'}")
