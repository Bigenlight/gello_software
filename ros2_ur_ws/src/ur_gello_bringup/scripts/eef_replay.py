#!/usr/bin/env python3
"""Offline replay of a recorded GELLO -> UR7e session through the FULL
``ur_kin`` + ``eef_delta`` chain (P2 measurement: zero real-robot time).

Reads a ``gello_recorder`` session (``vectors.h5``: ``gello_q`` / ``ur_q`` /
``cmd`` columns, see ``gello_recorder.recording_session``) and drives
``EefDeltaController`` with the RECORDED GELLO trajectory exactly as the real
bridge would, tick-by-tick, so we can measure how a real human's motion would
have fared under the eef governor/IK/safety stack -- reject rate, singularity
proximity, step/velocity/scale behaviour -- WITHOUT ever touching the robot.

Output: one CSV row per replayed tick (Cartesian velocity, joint step, sigma_min,
gamma, line-search scale, reject reason, ...), a JSON summary (hold/reject rate,
sigma_min extrema), and, if matplotlib is available, a PNG plot.

Standalone by design -- NO rclpy, NO ROS message types. h5py/PyYAML/matplotlib
are all imported lazily (inside the functions that need them), so this module
imports and answers ``--help`` even in an environment that has none of them
installed (e.g. the venv used for the pure-module test suite) and even when no
recorder output exists on disk yet.

Usage
-----
    python3 eef_replay.py /path/to/session_20260101_120000 --plot
    python3 eef_replay.py /path/to/vectors.h5 --out /tmp/replay.csv

The eef controller config is built from config/ur7e_gello.yaml (base) +
config/ur7e_gello_eef.yaml (overlay) by default -- the SAME two files the real
bridge loads -- so a replay run reflects the currently-configured tuning
unless overridden on the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# ur_gello_bringup/ is a sibling of scripts/ (this file's directory); import the
# real deploy modules from there so this script and the robot path can never
# drift apart (same pattern as gello_policy/scripts/offline_ensemble.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from ur_gello_bringup.eef_delta import EefDeltaController  # noqa: E402
from ur_gello_bringup.ur_kin import fk, se3_log  # noqa: E402

_N = 6
DEFAULT_BASE_YAML = os.path.join(_PKG, "config", "ur7e_gello.yaml")
DEFAULT_EEF_YAML = os.path.join(_PKG, "config", "ur7e_gello_eef.yaml")

CSV_FIELDS = [
    "t_rel_s",
    "dt_s",
    "state",
    "reject_reason",
    "sigma_min",
    "gamma",
    "ls_scale",
    "ik_residual",
    "lag_pos_m",
    "lag_rot_rad",
    "excursion_m",
    "branch_id",
    "n_ik_solutions",
    "cart_lin_vel_mps",
    "cart_ang_vel_radps",
    "max_joint_step_rad",
]


# --------------------------------------------------------------------------- #
# Config: base + eef-overlay ROS yaml -> EefDeltaController cfg               #
# --------------------------------------------------------------------------- #
def _load_yaml_params(path: str, node_name: str = "gello_ur_bridge") -> dict:
    """Best-effort ``<node_name>: ros__parameters:`` loader.

    Returns ``{}`` if the file is missing, unparsable, or PyYAML isn't
    installed -- mirrors ``ur_kin.load_dh``'s "optional yaml" tolerance so this
    script degrades to EefDeltaController's own built-in defaults rather than
    ever raising just because a yaml is absent.
    """
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml
    except ImportError:
        print(f"[warn] PyYAML not installed; ignoring {path}", file=sys.stderr)
        return {}
    with open(path, "r") as f:
        doc = yaml.safe_load(f) or {}
    node = doc.get(node_name) or {}
    return dict(node.get("ros__parameters") or {})


# EefDeltaController cfg keys spelled IDENTICALLY in the yaml (see
# config/ur7e_gello_eef.yaml). eef_v_max/eef_w_max are the two renamed keys,
# handled separately below.
_DIRECT_CFG_KEYS = (
    "pos_scale",
    "r_align_rpy",
    "tool_l_xyz_rpy",
    "tool_r_xyz_rpy",
    "sigma_warn",
    "sigma_stop",
    "gamma_min",
    "char_length",
    "branch_tol",
    "branch_weights",
    "limit_margin_rad",
    "s_floor",
    "lag_max_pose",
    "max_excursion_m",
    "keepout",
    "ik_backend",
)


def build_eef_cfg(base_yaml: str, eef_yaml: str, overrides: dict | None = None):
    """Merge base + eef-overlay ROS params -> ``(EefDeltaController cfg, raw params)``.

    ``raw params`` is returned too since a couple of node-level (not
    EefDeltaController) knobs -- ``max_step_rad``, ``publish_rate_hz`` -- come
    from the SAME yaml and are needed by the replay loop itself (step budget,
    default dt).
    """
    params: dict = {}
    params.update(_load_yaml_params(base_yaml))
    params.update(_load_yaml_params(eef_yaml))
    if overrides:
        params.update(overrides)

    cfg: dict = {}
    for k in _DIRECT_CFG_KEYS:
        if k in params:
            cfg[k] = params[k]
    if "eef_v_max" in params:
        cfg["v_max"] = params["eef_v_max"]
    if "eef_w_max" in params:
        cfg["w_max"] = params["eef_w_max"]
    return cfg, params


# --------------------------------------------------------------------------- #
# HDF5 loading (h5py imported lazily -- only touched once we actually read)   #
# --------------------------------------------------------------------------- #
def _find_h5(path: str) -> str:
    if os.path.isdir(path):
        cand = os.path.join(path, "vectors.h5")
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"no vectors.h5 found under session dir {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _read_table(h5file, name: str):
    """Return ``(dict[col] = np.ndarray, columns)`` for one table, or
    ``(None, [])`` if the table doesn't exist (a signal that never published,
    see recording_session.py -- every table is created up front regardless, so
    "doesn't exist" only happens for a corrupt/foreign file)."""
    if name not in h5file:
        return None, []
    grp = h5file[name]
    cols = json.loads(grp.attrs["columns"])
    data = {c: np.asarray(grp[c][:]) for c in cols}
    return data, cols


def _asof_merge(t_ref: np.ndarray, other: dict | None, prefix: str, n: int = _N) -> np.ndarray:
    """Backward (as-of) nearest-time merge of ``other``'s ``prefix{1..n}``
    columns onto ``t_ref``: row ``i`` gets the LATEST ``other`` row whose
    ``t_rel_s <= t_ref[i]`` (NaN row if none yet). Used only as the fallback
    path when the recorder's pre-aligned ``synchronized`` table is absent."""
    out = np.full((len(t_ref), n), np.nan)
    if other is None:
        return out
    ot = other["t_rel_s"]
    j = -1
    for i, ti in enumerate(t_ref):
        while j + 1 < len(ot) and ot[j + 1] <= ti:
            j += 1
        if j >= 0:
            out[i] = [other[f"{prefix}{k + 1}"][j] for k in range(n)]
    return out


def load_session(h5_path: str):
    """Load ``(t_rel_s, gello_q[N,6], ur_q[N,6], cmd[N,6])`` from ``vectors.h5``.

    Prefers the recorder's ``synchronized`` wide table (already time-aligned,
    latest-value-hold, see ``RecordingSession.write_sample``); falls back to an
    as-of merge of the three native-rate ``gello_joint_states`` /
    ``ur_joint_states`` / ``command`` tables onto the GELLO timeline if
    ``synchronized`` is missing or empty.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        sync, _ = _read_table(f, "synchronized")
        if sync is not None and len(sync.get("t_rel_s", [])) > 0:
            t = sync["t_rel_s"]
            gello_q = np.stack([sync[f"gello_q{i + 1}"] for i in range(_N)], axis=1)
            ur_q = np.stack([sync[f"ur_q{i + 1}"] for i in range(_N)], axis=1)
            cmd = np.stack([sync[f"cmd{i + 1}"] for i in range(_N)], axis=1)
            return t, gello_q, ur_q, cmd

        gello, _ = _read_table(f, "gello_joint_states")
        if gello is None or len(gello.get("t_rel_s", [])) == 0:
            raise ValueError(
                f"{h5_path}: neither a populated 'synchronized' table nor a "
                "'gello_joint_states' table to fall back to"
            )
        ur, _ = _read_table(f, "ur_joint_states")
        cmdt, _ = _read_table(f, "command")

        t = gello["t_rel_s"]
        gello_q = np.stack([gello[f"q{i + 1}"] for i in range(_N)], axis=1)
        ur_q = _asof_merge(t, ur, "q")
        cmd = _asof_merge(t, cmdt, "cmd")
        return t, gello_q, ur_q, cmd


# --------------------------------------------------------------------------- #
# Replay: drive EefDeltaController tick-by-tick over the recorded trajectory  #
# --------------------------------------------------------------------------- #
def replay(t, gello_q, ur_q, cmd, cfg: dict, step_eff: float,
           dt_mode: str = "recorded", fixed_dt: float = 0.004):
    """Run the recorded GELLO trajectory through a fresh ``EefDeltaController``.

    Engages once, anchored at the first row with BOTH a finite UR pose (the
    robot side of the anchor) and a finite GELLO pose (the leader side);
    every later finite-GELLO row is one ``ctrl.step()`` tick. Rows before the
    anchor, or with a still-NaN GELLO reading, are skipped (nothing to
    replay: the real bridge wouldn't have a command either).

    ``dt_mode="recorded"`` uses each row's own ``t_rel_s`` delta (matches real
    jitter -- and feeds ``EefDeltaController.dt``, which the anisotropic-gain
    rate limiter uses directly, so this is not merely cosmetic); ``"fixed"``
    uses ``fixed_dt`` for every tick.

    Returns ``list[dict]`` (one row per tick, fields = ``CSV_FIELDS``).
    """
    n = len(t)
    anchor_idx = None
    for i in range(n):
        if np.all(np.isfinite(ur_q[i])) and np.all(np.isfinite(gello_q[i])):
            anchor_idx = i
            break
    if anchor_idx is None:
        raise ValueError(
            "no row with both a finite UR pose and a finite GELLO pose to "
            "anchor on -- nothing to replay"
        )

    ctrl = EefDeltaController(dict(cfg))
    ctrl.engage(ur_q[anchor_idx], gello_q[anchor_idx])

    T_prev = fk(ur_q[anchor_idx]) @ ctrl.T_tool_R
    q_prev = ur_q[anchor_idx].copy()
    t_prev = float(t[anchor_idx])

    rows = []
    for i in range(anchor_idx + 1, n):
        if not np.all(np.isfinite(gello_q[i])):
            continue

        ti = float(t[i])
        if dt_mode == "recorded":
            dt = ti - t_prev
            if not math.isfinite(dt) or dt <= 0.0:
                dt = fixed_dt
        else:
            dt = fixed_dt
        ctrl.dt = dt

        q_cmd, info = ctrl.step(gello_q[i], step_eff)
        T_cmd = info["T_cmd"]

        xi = se3_log(np.linalg.inv(T_prev) @ T_cmd)
        lin_v = float(np.linalg.norm(xi[:3])) / dt if dt > 0 else float("nan")
        ang_v = float(np.linalg.norm(xi[3:])) / dt if dt > 0 else float("nan")
        max_step = float(np.max(np.abs(q_cmd - q_prev))) if q_cmd is not None else float("nan")

        rows.append({
            "t_rel_s": ti,
            "dt_s": dt,
            "state": info.get("state"),
            "reject_reason": info.get("reject_reason") or "",
            "sigma_min": info.get("sigma_min", float("nan")),
            "gamma": info.get("gamma", float("nan")),
            "ls_scale": info.get("ls_scale", float("nan")),
            "ik_residual": info.get("ik_residual", float("nan")),
            "lag_pos_m": info.get("lag_pos", float("nan")),
            "lag_rot_rad": info.get("lag_rot", float("nan")),
            "excursion_m": info.get("excursion", float("nan")),
            "branch_id": info.get("branch_id", -1),
            "n_ik_solutions": info.get("n_ik_solutions", 0),
            "cart_lin_vel_mps": lin_v,
            "cart_ang_vel_radps": ang_v,
            "max_joint_step_rad": max_step,
        })

        T_prev = T_cmd
        t_prev = ti
        if q_cmd is not None:
            q_prev = q_cmd

    return rows


def summarize(rows: list) -> dict:
    """Reject-rate / singularity-proximity summary of a :func:`replay` run."""
    n = len(rows)
    n_hold = sum(1 for r in rows if r["state"] == "HOLD")
    reasons: dict = {}
    for r in rows:
        rr = r["reject_reason"]
        if rr:
            reasons[rr] = reasons.get(rr, 0) + 1
    sigmas = [r["sigma_min"] for r in rows if math.isfinite(r["sigma_min"])]
    gammas = [r["gamma"] for r in rows if math.isfinite(r["gamma"])]
    lin_vs = [r["cart_lin_vel_mps"] for r in rows if math.isfinite(r["cart_lin_vel_mps"])]
    ang_vs = [r["cart_ang_vel_radps"] for r in rows if math.isfinite(r["cart_ang_vel_radps"])]
    return {
        "n_ticks": n,
        "n_hold": n_hold,
        "hold_rate": (n_hold / n) if n else 0.0,
        "reject_reason_counts": reasons,
        "sigma_min_min": min(sigmas) if sigmas else None,
        "sigma_min_mean": (sum(sigmas) / len(sigmas)) if sigmas else None,
        "gamma_min": min(gammas) if gammas else None,
        "cart_lin_vel_mps_max": max(lin_vs) if lin_vs else None,
        "cart_ang_vel_radps_max": max(ang_vs) if ang_vs else None,
    }


# --------------------------------------------------------------------------- #
# Output: CSV + summary JSON + optional plot                                  #
# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: list) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def maybe_plot(rows: list, png_path: str) -> bool:
    """Best-effort PNG plot (sigma_min / gamma / Cartesian speed / rejects vs
    time). Returns False (and prints a notice) if matplotlib isn't installed
    instead of raising -- plotting is a "nice to have", never a hard
    requirement of this script."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed; skipping --plot", file=sys.stderr)
        return False

    if not rows:
        print("[info] no rows to plot", file=sys.stderr)
        return False

    t = [r["t_rel_s"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(t, [r["sigma_min"] for r in rows], label="sigma_min")
    axes[0].axhline(0.0, color="grey", linewidth=0.5)
    axes[0].set_ylabel("sigma_min")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, [r["gamma"] for r in rows], label="gamma", color="tab:orange")
    axes[1].plot(t, [r["ls_scale"] for r in rows], label="ls_scale", color="tab:green")
    axes[1].set_ylabel("gamma / ls_scale")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, [r["cart_lin_vel_mps"] for r in rows], label="lin vel (m/s)")
    axes[2].plot(t, [r["cart_ang_vel_radps"] for r in rows], label="ang vel (rad/s)")
    axes[2].set_ylabel("Cartesian vel")
    axes[2].legend(loc="upper right")

    hold_t = [ti for ti, r in zip(t, rows) if r["state"] == "HOLD"]
    axes[3].scatter(hold_t, [1] * len(hold_t), marker="|", color="tab:red", label="HOLD")
    axes[3].set_ylim(0, 2)
    axes[3].set_yticks([])
    axes[3].set_xlabel("t_rel_s")
    axes[3].set_ylabel("reject")
    axes[3].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline replay of a gello_recorder session through ur_kin + "
                     "eef_delta (no robot, no ROS).",
    )
    ap.add_argument("session", help="recorder session dir OR a direct path to vectors.h5")
    ap.add_argument("--out", default=None,
                     help="output CSV path (default: <session_dir>/eef_replay.csv)")
    ap.add_argument("--base-yaml", default=DEFAULT_BASE_YAML,
                     help="base ROS params yaml (default: config/ur7e_gello.yaml)")
    ap.add_argument("--eef-yaml", default=DEFAULT_EEF_YAML,
                     help="eef overlay ROS params yaml (default: config/ur7e_gello_eef.yaml)")
    ap.add_argument("--step-eff", type=float, default=None,
                     help="max joint step per tick (rad); default = max_step_rad from --base-yaml")
    ap.add_argument("--dt-mode", choices=["recorded", "fixed"], default="recorded",
                     help="per-tick dt: recorded t_rel_s deltas, or a --fixed-dt constant")
    ap.add_argument("--fixed-dt", type=float, default=None,
                     help="dt (s) for --dt-mode fixed; default = 1/publish_rate_hz from --base-yaml")
    ap.add_argument("--plot", action="store_true",
                     help="also save a PNG plot next to the CSV (needs matplotlib; skipped if absent)")
    args = ap.parse_args(argv)

    h5_path = _find_h5(args.session)
    cfg, params = build_eef_cfg(args.base_yaml, args.eef_yaml)

    step_eff = args.step_eff
    if step_eff is None:
        step_eff = float(params.get("max_step_rad", 0.0025))
    publish_rate_hz = float(params.get("publish_rate_hz", 250.0))
    fixed_dt = args.fixed_dt if args.fixed_dt is not None else (1.0 / publish_rate_hz)
    cfg.setdefault("dt", fixed_dt)

    t, gello_q, ur_q, cmd = load_session(h5_path)
    rows = replay(t, gello_q, ur_q, cmd, cfg, step_eff,
                  dt_mode=args.dt_mode, fixed_dt=fixed_dt)

    out_csv = args.out or os.path.join(os.path.dirname(h5_path), "eef_replay.csv")
    write_csv(out_csv, rows)
    print(f"wrote {len(rows)} rows -> {out_csv}")

    summary = summarize(rows)
    summary_path = os.path.splitext(out_csv)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote summary -> {summary_path}")

    if args.plot:
        png_path = os.path.splitext(out_csv)[0] + ".png"
        if maybe_plot(rows, png_path):
            print(f"wrote plot -> {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
