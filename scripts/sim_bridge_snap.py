#!/usr/bin/env python3
"""Faithful offline sim of the gello_ur_bridge post-handoff streaming, to test
the 'startup snap' mechanism WITHOUT ROS/robot/GELLO.

It imports the REAL _OneEuro filter from gello_ur_bridge_node.py and replicates
_on_timer's per-joint math (one_euro OR ema+deadband, then max_step clamp) exactly.

Scenario modelled (single joint, per-joint mechanism):
  - Arm has just arrived at the FROZEN move-to-start target -> we seed the bridge
    from the actual pose = frozen_target (ref 0.0), exactly as the node does.
  - The LIVE GELLO is now at frozen_target + drift (+ per-sample tremor/noise),
    because the human/leader drifted during the ~5 s trajectory.
  - Stream at publish_rate_hz; GELLO samples arrive at gello_rate_hz.
  - Measure the published command's PEAK slew speed (rad/s) and settle time.
"""
from __future__ import annotations
import argparse, importlib.util, json, math
from pathlib import Path
import numpy as np

NODE = Path("/home/theo/gello_software/ros2_ur_ws/src/ur_gello_bringup/"
            "ur_gello_bringup/gello_ur_bridge_node.py")
spec = importlib.util.spec_from_file_location("bridge_node", NODE)
mod = importlib.util.module_from_spec(spec)
# The node imports rclpy at module load; guard by stubbing if unavailable.
import sys
try:
    spec.loader.exec_module(mod)
except Exception:
    # Fall back: exec only the _OneEuro class source (pure, no rclpy).
    src = NODE.read_text()
    start = src.index("class _OneEuro")
    end = src.index("class GelloUrBridge")
    ns: dict = {"math": math}
    exec(compile(src[start:end], str(NODE), "exec"), ns)
    mod = type("m", (), ns)
_OneEuro = mod._OneEuro


def simulate(drift, *, filter_type="one_euro", max_step_rad=0.0025,
             publish_rate_hz=250.0, ema_alpha=0.4, deadband_rad=0.004,
             min_cutoff=1.0, beta=2.0, d_cutoff=1.0,
             gello_rate_hz=30.0, tremor_std=0.0, noise_std=0.0,
             sim_s=3.0, rng=None):
    rng = rng or np.random.default_rng(0)
    dt_pub = 1.0 / publish_rate_hz
    n = int(sim_s * publish_rate_hz)
    frozen = 0.0                 # arm arrived here; bridge seeds here
    euro = _OneEuro(dt_pub, min_cutoff, beta, d_cutoff)
    euro.seed(frozen)
    filtered = frozen
    last_pub = frozen
    gated = frozen               # ema deadband held target (seed to live below on first)
    # live gello sits at frozen+drift, jittered per GELLO sample
    pub_per_gello = max(1, int(round(publish_rate_hz / gello_rate_hz)))
    raw_target = frozen + drift
    last_gello_input = None
    published = []
    first = True
    for k in range(n):
        # New GELLO sample at gello cadence
        if k % pub_per_gello == 0:
            j = rng.normal(0, tremor_std) + rng.normal(0, noise_std)
            raw_target = frozen + drift + j
            if euro is not None:
                dt_g = None if last_gello_input is None else pub_per_gello * dt_pub
                euro.update_input(raw_target, dt_g)
            last_gello_input = raw_target
        if first:
            # mirror node: seed gated_target to raw on first cycle
            gated = raw_target
            first = False
        if filter_type == "one_euro":
            filtered = euro(raw_target)
        else:  # ema + deadband
            if abs(raw_target - gated) > deadband_rad:
                gated = raw_target
            filtered = (1.0 - ema_alpha) * filtered + ema_alpha * gated
        delta = filtered - last_pub
        delta = max(-max_step_rad, min(max_step_rad, delta))
        last_pub = last_pub + delta
        published.append(last_pub)
    published = np.array(published)
    speed = np.abs(np.diff(published)) * publish_rate_hz  # rad/s per cycle
    peak_speed = float(speed.max()) if speed.size else 0.0
    # settle: first time within 5% of drift and stays
    eps = 0.05 * abs(drift) if drift else 0.01
    settled = np.where(np.abs(published - drift) <= eps)[0]
    settle_s = float(settled[0] / publish_rate_hz) if settled.size else float("nan")
    return {
        "drift_rad": round(drift, 4), "drift_deg": round(math.degrees(drift), 2),
        "filter_type": filter_type, "max_step_rad": max_step_rad,
        "publish_rate_hz": publish_rate_hz,
        "peak_speed_rad_s": round(peak_speed, 4),
        "peak_speed_deg_s": round(math.degrees(peak_speed), 1),
        "settle_s": round(settle_s, 3),
        "final_err_rad": round(float(abs(published[-1] - drift)), 5),
        "theoretical_max_slew_rad_s": round(max_step_rad * publish_rate_hz, 4),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=float, default=0.2)
    ap.add_argument("--filter", dest="filter_type", default="one_euro")
    ap.add_argument("--max-step", dest="max_step_rad", type=float, default=0.0025)
    ap.add_argument("--rate", dest="publish_rate_hz", type=float, default=250.0)
    ap.add_argument("--tremor", dest="tremor_std", type=float, default=0.0)
    ap.add_argument("--noise", dest="noise_std", type=float, default=0.0)
    ap.add_argument("--sim-s", dest="sim_s", type=float, default=3.0)
    a = ap.parse_args()
    print(json.dumps(simulate(
        a.drift, filter_type=a.filter_type, max_step_rad=a.max_step_rad,
        publish_rate_hz=a.publish_rate_hz, tremor_std=a.tremor_std,
        noise_std=a.noise_std, sim_s=a.sim_s)))
