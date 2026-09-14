#!/usr/bin/env python3
"""Derive the EEF-delta deploy envelope for ``carrot_in_pot`` from the raw takes.

A2 of ``ros2_ur_ws/src/gello_policy/CARROT_EEF_DESIGN.md``. Reads every
``take_*/vectors.h5`` under ``--data``, resamples onto the cam1 master clock
exactly like ``scripts/dataset/convert_carrot_to_lerobot.py`` (so the numbers
describe the frames the policy is actually trained on), and writes a provenance
JSON plus the ROS-parameter yaml snippet that A6 pastes into
``config/carrot_eef_{act,fm}.yaml``.

WHY a separate tool instead of eyeballing the dataset: every safety clamp in the
deploy node (§3 of the design) is a number that must be traceable to the takes it
came from. The banana equivalent is ``config/joint_safety_limits.json``; this file
keeps the same schema style and the same 1.2x expansion formula, and additionally
carries the Cartesian box and the per-tick / lead delta budgets that the EEF path
needs.

The stale joint cache
---------------------
In this recording session ``ur_joint_states`` is stamped ~0.90 s STALE while
``command`` and the cameras are fresh (integrator's measurement: ``FK(q(t+0.90))``
matches ``TCP(command(t))`` to ~1 mm). A constant time shift does not move the
joint envelope, the workspace box or the 1-second motion statistic. It DOES move
two things:

* **frame 0** -- the arm is already moving there, so the pose at the instant the
  first image was taken is ``q(t0 + tau_q)``, 21 mm of TCP away from the naive
  ``q(t0)``. ``start_pose`` is therefore the de-staled one.
* **the single-frame motion** -- not because the shift changes the trajectory but
  because it changes which raw cache sample a 30 Hz frame snaps to. A1's converter
  de-stales by LINEAR INTERPOLATION, so this tool does too: ``frame_1`` (and hence
  ``eef_max_step_*``) is measured on ``q_interp(cam1_t + tau_q)``, which is exactly
  the series the dataset action is built from. The nearest-neighbour and tau=0
  variants are kept beside it as audit blocks -- the nearest one is visibly noisier
  (8.8 % of its frames repeat a sample), which is why interpolation won.

Explaining the lag itself is A7's row, not this one.

All kinematics go through ``gello_policy.eef_space`` (which goes through
``ur_gello_bringup.ur_kin``) -- the arithmetic is NEVER re-implemented here, so a
change to the TCP definition or the action convention moves these limits with it.

Interpreter: system ``python3`` (numpy + h5py), no ROS sourcing required::

    PYTHONPATH=<wt>/ros2_ur_ws/src/ur_gello_bringup:<wt>/ros2_ur_ws/src/gello_policy \
      python3 scripts/dataset/make_carrot_eef_limits.py \
        --data /home/laptop3/youngwoong_ws/Put_carrot_in_pot \
        --out  <wt>/ros2_ur_ws/src/gello_policy/config/carrot_eef_limits.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
# Allow running without PYTHONPATH set, as long as the worktree layout is intact.
for _p in (
    os.path.join(_REPO_ROOT, "ros2_ur_ws", "src", "ur_gello_bringup"),
    os.path.join(_REPO_ROOT, "ros2_ur_ws", "src", "gello_policy"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from gello_policy import eef_space  # noqa: E402  (needs the sys.path fix above)
from ur_gello_bringup import ur_kin  # noqa: E402

DEFAULT_DATA = "/home/laptop3/youngwoong_ws/Put_carrot_in_pot"
DEFAULT_OUT = os.path.join(
    _REPO_ROOT, "ros2_ur_ws", "src", "gello_policy", "config", "carrot_eef_limits.json"
)

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

# Envelope expansion factor, identical to config/joint_safety_limits.json (banana).
EXPAND = 1.2
# Cartesian margins added to the observed TCP box (m). z gets a smaller one and a
# hard floor at 0 because "below the table" is the one direction that is not a
# recoverable OOD excursion.
WS_MARGIN_M = 0.03
WS_Z_MARGIN_M = 0.01
WS_Z_FLOOR_M = 0.0
# Per-tick delta budget = this many times the largest 1-frame achieved motion.
STEP_SAFETY_FACTOR = 1.5
# Delta-statistics horizons in master-clock frames (30 fps): one control tick and
# one second (~ an ACT chunk), the latter being the natural scale for the lead clamp.
H_STEP = 1
H_LEAD = 30

# Joint-cache shift search (see the module docstring).
TAU_GRID = np.round(np.arange(0.0, 1.5001, 0.01), 4)
TAU_MIN_OK, TAU_MAX_OK = 0.6, 1.2       # integrator's acceptance window
TAU_RESIDUAL_MAX_M = 0.003              # median |TCP(q shifted) - TCP(cmd)| must beat this

# Every stored number is quantised to this grid, OUTWARD (lo down, hi up), so that
# a reader can re-derive joint_limits/workspace from the JSON's own joint_raw_*/
# workspace_raw and get byte-identical values. The banana joint_safety_limits.json
# does NOT do this -- it printed 4-decimal raw values but derived its limits from
# the unrounded ones, so recomputing from that file is off by up to 1e-4 rad.
GRID = 1e-4


# --------------------------------------------------------------------------- #
# resampling -- semantics copied verbatim from convert_carrot_to_lerobot.py
# (nearest_idx / ffill_bfill / load_take_arrays), so the frames analysed here are
# byte-for-byte the frames the LeRobot dataset holds.
# --------------------------------------------------------------------------- #
def nearest_idx(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """For each query timestamp, index of the nearest src sample (src_t sorted asc)."""
    j = np.searchsorted(src_t, query_t)
    j = np.clip(j, 1, len(src_t) - 1)
    left = src_t[j - 1]
    right = src_t[j]
    pick_left = (query_t - left) <= (right - query_t)
    out = np.where(pick_left, j - 1, j)
    return np.clip(out, 0, len(src_t) - 1)


def ffill_bfill(v: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaNs."""
    v = v.copy()
    n = len(v)
    last = np.nan
    for i in range(n):
        if np.isnan(v[i]):
            v[i] = last
        else:
            last = v[i]
    nxt = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(v[i]):
            v[i] = nxt
        else:
            nxt = v[i]
    return v


class Take:
    """The raw streams of one ``vectors.h5`` plus the master-clock resampling.

    Only what A2 needs: the cam1 master clock, the measured joints, the commanded
    joints (used ONLY to measure the cache shift -- never to define an action;
    design §1) and the gripper.
    """

    def __init__(self, h5_path: str) -> None:
        with h5py.File(h5_path, "r") as f:
            self.cam1_t = f["cam1_frames"]["t_rel_s"][:].astype(float)
            self.ur_t = f["ur_joint_states"]["t_rel_s"][:].astype(float)
            self.q_raw = np.stack(
                [f["ur_joint_states"][f"q{k + 1}"][:] for k in range(6)], axis=1
            ).astype(float)
            if "command" in f:
                self.cmd_t = f["command"]["t_rel_s"][:].astype(float)
                self.cmd_raw = np.stack(
                    [f["command"][f"cmd{k + 1}"][:] for k in range(6)], axis=1
                ).astype(float)
            else:
                self.cmd_t = np.zeros(0)
                self.cmd_raw = np.zeros((0, 6))
            self.grip_t = f["gripper"]["t_rel_s"][:].astype(float)
            self.grip_pos_raw = f["gripper"]["grip_pos"][:].astype(float)
            self.grip_cmd_raw = ffill_bfill(f["gripper"]["grip_cmd"][:]).astype(float)

    @property
    def n(self) -> int:
        return len(self.cam1_t)

    def q_at(self, tau: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Joints on the master clock shifted by ``tau`` s, plus an in-range mask.

        The mask is False where ``cam1_t + tau`` falls outside the joint record;
        ``nearest_idx`` would silently clamp there and invent a frozen arm.
        """
        tq = self.cam1_t + tau
        ok = (tq >= self.ur_t[0]) & (tq <= self.ur_t[-1])
        return self.q_raw[nearest_idx(self.ur_t, tq)], ok

    def q_interp_at(self, tau: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Like :meth:`q_at` but LINEARLY interpolated instead of nearest-neighbour.

        This is the PRIMARY de-staling path: A1's converter interpolates too, so
        the deltas measured here are the actions the policy is trained to emit.
        :meth:`q_at` (nearest) is kept only to show how much single-frame motion
        the snap-to-sample quantisation of a jittery 66 Hz cache invents.
        """
        tq = self.cam1_t + tau
        ok = (tq >= self.ur_t[0]) & (tq <= self.ur_t[-1])
        q = np.stack([np.interp(tq, self.ur_t, self.q_raw[:, k]) for k in range(6)], axis=1)
        return q, ok

    def gripper(self) -> Tuple[np.ndarray, np.ndarray]:
        """(grip_pos, grip_cmd) on the master clock (unshifted, as the converter does)."""
        j = nearest_idx(self.grip_t, self.cam1_t)
        return self.grip_pos_raw[j], self.grip_cmd_raw[j]


def load_take_arrays(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (q[N,6], grip_pos[N], grip_cmd[N]) on the cam1 master timeline.

    Same clock and same nearest-timestamp / ffill-bfill rules as the joint-space
    converter's ``load_take_arrays``; the commanded joints are deliberately NOT
    returned (design §1: ``command`` leads the measured joints by ~0.9 s in this
    session, so the action is defined from ACHIEVED motion only).
    """
    tk = Take(h5_path)
    q, _ = tk.q_at(0.0)
    grip_pos, grip_cmd = tk.gripper()
    return q, grip_pos, grip_cmd


# --------------------------------------------------------------------------- #
# stale-joint-cache shift
# --------------------------------------------------------------------------- #
def estimate_joint_shift(
    take: Take,
    tau_grid: np.ndarray = TAU_GRID,
) -> Tuple[Optional[float], float, int]:
    """Measure how far the joint cache lags the (fresh) command stream, in seconds.

    Returns ``(tau, residual_m, n_frames)`` where ``tau`` minimises the median
    ``|TCP(q(t + tau)) - TCP(command(t))|`` over the master frames for which the
    shifted query is inside the joint record. ``(None, nan, 0)`` when the take has
    no command stream (synthetic fixtures).

    FK is evaluated once per RAW sample and then indexed, so the whole sweep costs
    ~2 x n_raw forward kinematics per take rather than ``len(tau_grid) x n``.
    """
    if take.cmd_raw.shape[0] == 0 or take.n == 0:
        return None, float("nan"), 0

    p_ur = np.array([eef_space.tcp_pose(q)[:3, 3] for q in take.q_raw])
    p_cmd = np.array([eef_space.tcp_pose(q)[:3, 3] for q in take.cmd_raw])
    p_cmd_master = p_cmd[nearest_idx(take.cmd_t, take.cam1_t)]

    best_tau: Optional[float] = None
    best_res = float("inf")
    best_n = 0
    for tau in tau_grid:
        tq = take.cam1_t + float(tau)
        ok = (tq >= take.ur_t[0]) & (tq <= take.ur_t[-1])
        if int(ok.sum()) < 10:
            continue
        j = nearest_idx(take.ur_t, tq[ok])
        res = float(np.median(np.linalg.norm(p_ur[j] - p_cmd_master[ok], axis=1)))
        if res < best_res:
            best_tau, best_res, best_n = float(tau), res, int(ok.sum())
    return best_tau, best_res, best_n


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def ceil_to(x: float, step: float) -> float:
    """Round x UP to the next multiple of ``step`` (safety numbers never round down)."""
    # round() before ceil() so 0.045/0.01 == 4.499999999 does not become 0.05 -> 0.05
    # by luck and 0.06 by bad luck; 9 decimals is far below any real measurement noise.
    return round(math.ceil(round(x / step, 9)) * step, 9)


def floor_to(x: float, step: float) -> float:
    """Round x DOWN to the previous multiple of ``step``."""
    return round(math.floor(round(x / step, 9)) * step, 9)


def outward(lo, hi, step: float = GRID) -> Tuple[np.ndarray, np.ndarray]:
    """Quantise an interval onto the grid so it can only grow, never shrink."""
    lo = np.array([floor_to(float(v), step) for v in np.asarray(lo).reshape(-1)])
    hi = np.array([ceil_to(float(v), step) for v in np.asarray(hi).reshape(-1)])
    return lo, hi


def expand_symmetric(lo: np.ndarray, hi: np.ndarray, factor: float = EXPAND) -> Tuple[np.ndarray, np.ndarray]:
    """``factor`` x expansion about the centre of [lo, hi] (the banana formula)."""
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return centre - factor * half, centre + factor * half


def _round_list(v: Sequence[float], nd: int) -> List[float]:
    return [round(float(x), nd) for x in v]


def _stats(v: np.ndarray) -> Dict[str, float]:
    """p50 / p99 / max of a magnitude array, rounded to 1e-6 (um / urad)."""
    if v.size == 0:
        return {"p50": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": round(float(np.percentile(v, 50)), 6),
        "p99": round(float(np.percentile(v, 99)), 6),
        "max": round(float(np.max(v)), 6),
    }


def git_commit(root: str) -> str:
    """``<sha>`` or ``<sha>-dirty`` for the worktree holding this script."""
    try:
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception as exc:  # pragma: no cover - provenance must never fail the run
        return f"<unavailable: {exc}>"


# --------------------------------------------------------------------------- #
# per-take analysis
# --------------------------------------------------------------------------- #
def delta_magnitudes(
    q: np.ndarray,
    horizon: int,
    valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(|dp|, |drot|) of the ACHIEVED motion over ``horizon`` frames.

    Uses :func:`eef_space.delta_action` -- the very function that defines the
    dataset action -- so the clamps derived from these numbers are expressed in
    the same units the policy emits. The gripper slot is irrelevant here.
    ``valid`` (optional) marks frames whose joints are real rather than clamped at
    the edge of the record; a pair is used only when BOTH ends are valid.
    """
    n = len(q)
    if n <= horizon:
        return np.zeros(0), np.zeros(0)
    if valid is None:
        use = np.ones(n - horizon, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        use = valid[:n - horizon] & valid[horizon:]
    dp: List[float] = []
    drot: List[float] = []
    for t in range(n - horizon):
        if not use[t]:
            continue
        a = eef_space.delta_action(q[t], q[t + horizon], 0.0)
        dp.append(float(np.linalg.norm(a[:3])))
        drot.append(float(np.linalg.norm(a[3:6])))
    return np.array(dp), np.array(drot)


def analyse(
    data_root: str,
    branch_stride: int = 1,
    limit: int | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Read every take and return the raw measurements (no rounding/formatting)."""
    takes = sorted(t for t in glob.glob(os.path.join(data_root, "take_*")) if os.path.isdir(t))
    if limit:
        takes = takes[:limit]
    if not takes:
        raise SystemExit(f"[!] no takes found under {data_root}")

    q0_list: List[np.ndarray] = []
    q0_destaled: List[np.ndarray] = []
    grip0_list: List[float] = []
    j_min = np.full(6, np.inf)
    j_max = np.full(6, -np.inf)
    p_min = np.full(3, np.inf)
    p_max = np.full(3, -np.inf)
    gp_min, gp_max = np.inf, -np.inf
    gc_min, gc_max = np.inf, -np.inf
    dp1s: List[np.ndarray] = []
    dr1s: List[np.ndarray] = []
    dp1u: List[np.ndarray] = []
    dr1u: List[np.ndarray] = []
    dp1i: List[np.ndarray] = []
    dr1i: List[np.ndarray] = []
    dp30: List[np.ndarray] = []
    dr30: List[np.ndarray] = []
    q_branch_pool: List[np.ndarray] = []
    taus: List[Optional[float]] = []
    residuals: List[float] = []
    n_frames = 0

    for tk in takes:
        take = Take(os.path.join(tk, "vectors.h5"))
        q, _ = take.q_at(0.0)                       # the converter's own resampling
        grip_pos, grip_cmd = take.gripper()
        n = take.n
        n_frames += n
        q0_list.append(q[0].copy())
        grip0_list.append(float(grip_pos[0]))

        # Envelopes / start pose / workspace: a pure time shift cannot move these,
        # so they stay on the converter's resampling (= the dataset's own frames).
        j_min = np.minimum(j_min, q.min(axis=0))
        j_max = np.maximum(j_max, q.max(axis=0))
        gp_min = min(gp_min, float(grip_pos.min()))
        gp_max = max(gp_max, float(grip_pos.max()))
        gc_min = min(gc_min, float(grip_cmd.min()))
        gc_max = max(gc_max, float(grip_cmd.max()))

        p = np.array([eef_space.tcp_pose(qi)[:3, 3] for qi in q])
        p_min = np.minimum(p_min, p.min(axis=0))
        p_max = np.maximum(p_max, p.max(axis=0))

        # 1-frame motion: on the DE-STALED joints (module docstring).
        tau, res, _ = estimate_joint_shift(take)
        taus.append(tau)
        residuals.append(res)
        q_sh, ok_sh = take.q_at(tau if tau is not None else 0.0)
        a, b = delta_magnitudes(q_sh, H_STEP, ok_sh)   # audit: de-staled, nearest
        dp1s.append(a)
        dr1s.append(b)
        a, b = delta_magnitudes(q, H_STEP)            # audit: converter resampling
        dp1u.append(a)
        dr1u.append(b)
        # PRIMARY: de-staled by linear interpolation == A1's converter recipe.
        q_in, ok_in = take.q_interp_at(tau if tau is not None else 0.0)
        if not bool(ok_in[0]):
            raise SystemExit(f"[!] {os.path.basename(tk)}: frame 0 has no joints at "
                             f"cam1_t[0] + tau_q -- cannot de-stale the start pose")
        q0_destaled.append(q_in[0].copy())
        a, b = delta_magnitudes(q_in, H_STEP, ok_in)
        dp1i.append(a)
        dr1i.append(b)
        # 1 s horizon: unaffected by a constant shift, so keep the dataset frames.
        a, b = delta_magnitudes(q, H_LEAD)
        dp30.append(a)
        dr30.append(b)

        q_branch_pool.append(q[::max(1, branch_stride)])
        if verbose:
            tt = "n/a" if tau is None else f"{tau:.2f}s res {res * 1000:.1f}mm"
            print(f"[take] {os.path.basename(tk)}: {n} frames, tau_q {tt}", flush=True)

    # ---- start pose -------------------------------------------------------- #
    # Unwrap every take's frame-0 to the FIRST take's frame-0 before averaging:
    # wrist_3 sits near -pi here, so a take recorded at +pi would otherwise drag
    # the mean to ~0 -- a pose the arm has never been in.
    ref = q0_list[0]
    q0 = np.array([ur_kin.wrapped_nearest(v, ref) for v in q0_list])
    start_unshifted = q0.mean(axis=0)

    # Frame 0 is the ONE place a constant time shift is not invariant: the arm is
    # already moving, so q(t0) and q(t0 + tau_q) are different poses. The de-staled
    # one is the arm's REAL pose when the first image was taken, which is what the
    # policy's frame-0 observation contains -- so that is start_pose. The naive value
    # is kept for reference only.
    ref_d = q0_destaled[0]
    q0d = np.array([ur_kin.wrapped_nearest(v, ref_d) for v in q0_destaled])
    start_pose = q0d.mean(axis=0)
    start_std = q0d.std(axis=0)
    start_dev = np.max(np.abs(q0d - start_pose), axis=0)

    start_branch = int(ur_kin.branch_id(start_pose))

    # ---- branch check ------------------------------------------------------ #
    off: List[Dict[str, Any]] = []   # first few offenders, for the JSON
    n_off = 0                        # total count (off may be truncated)
    checked = 0
    for tk, qs in zip(takes, q_branch_pool):
        for i, qi in enumerate(qs):
            checked += 1
            bid = int(ur_kin.branch_id(qi))
            if bid != start_branch:
                n_off += 1
                if len(off) < 20:
                    off.append({"take": os.path.basename(tk),
                                "row": i * branch_stride, "branch": bid})

    return {
        "takes": [os.path.basename(t) for t in takes],
        "n_takes": len(takes),
        "n_frames": n_frames,
        "start_pose": start_pose,
        "start_pose_std": start_std,
        "start_pose_max_dev": start_dev,
        "start_pose_unshifted": start_unshifted,
        "start_pose_unshifted_std": q0.std(axis=0),
        "start_pose_unshifted_max_dev": np.max(np.abs(q0 - start_unshifted), axis=0),
        "start_branch": start_branch,
        "branch_checked": checked,
        "branch_stride": branch_stride,
        "branch_off": off,
        "branch_n_off": n_off,
        "start_gripper": float(np.median(grip0_list)),
        "joint_raw_min": j_min,
        "joint_raw_max": j_max,
        "gripper_pos_raw_min": gp_min,
        "gripper_pos_raw_max": gp_max,
        "gripper_cmd_raw_min": gc_min,
        "gripper_cmd_raw_max": gc_max,
        "ws_min": p_min,
        "ws_max": p_max,
        "tau_q": taus,
        "tau_residual_m": residuals,
        "dp1": np.concatenate(dp1i) if dp1i else np.zeros(0),
        "dr1": np.concatenate(dr1i) if dr1i else np.zeros(0),
        "dp1_destaled_nearest": np.concatenate(dp1s) if dp1s else np.zeros(0),
        "dr1_destaled_nearest": np.concatenate(dr1s) if dr1s else np.zeros(0),
        "dp1_unshifted": np.concatenate(dp1u) if dp1u else np.zeros(0),
        "dr1_unshifted": np.concatenate(dr1u) if dr1u else np.zeros(0),
        "dp30": np.concatenate(dp30) if dp30 else np.zeros(0),
        "dr30": np.concatenate(dr30) if dr30 else np.zeros(0),
    }


def check_shifts(raw: Dict[str, Any]) -> List[str]:
    """Integrator's acceptance test on the measured per-take cache shift."""
    problems: List[str] = []
    for name, tau, res in zip(raw["takes"], raw.get("tau_q", []), raw.get("tau_residual_m", [])):
        if tau is None:
            continue
        if not (TAU_MIN_OK <= tau <= TAU_MAX_OK):
            problems.append(f"{name}: tau_q {tau:.2f}s outside [{TAU_MIN_OK}, {TAU_MAX_OK}]")
        if not (res < TAU_RESIDUAL_MAX_M):
            problems.append(f"{name}: residual {res * 1000:.1f}mm >= "
                            f"{TAU_RESIDUAL_MAX_M * 1000:.0f}mm")
    return problems


# --------------------------------------------------------------------------- #
# JSON + yaml assembly
# --------------------------------------------------------------------------- #
def _yaml_list(v: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(x):.4f}" for x in v) + "]"


def build_yaml_snippet(doc: Dict[str, Any], indent: str = "    ") -> str:
    """The exact lines A6 pastes under ``policy_leader_node: ros__parameters:``."""
    lines = [
        "# --- carrot_in_pot EEF-delta envelope"
        " (scripts/dataset/make_carrot_eef_limits.py) ---",
        f"# source: {doc['source_data']}"
        f" ({doc['n_takes']} takes, {doc['n_frames']} frames)",
        f"# generated {doc['generated_at']} from git {doc['git_commit']}",
        "# start_pose is DE-STALED (joints interpolated at cam1_t[0] + tau_q): the arm's"
        " real pose when the first camera frame was taken.",
        f"start_pose: {_yaml_list(doc['start_pose'])}",
        "# reference only, do NOT paste -- the naive tau=0 frame-0 mean, 21 mm of TCP away:",
        f"# start_pose_unshifted: {_yaml_list(doc['start_pose_unshifted'])}",
        f"start_gripper: {doc['start_gripper']:.4f}",
        f"joint_limits_lo: {_yaml_list(doc['joint_limits_lo'])}",
        f"joint_limits_hi: {_yaml_list(doc['joint_limits_hi'])}",
        f"workspace_lo: {_yaml_list(doc['workspace_lo'])}",
        f"workspace_hi: {_yaml_list(doc['workspace_hi'])}",
        f"eef_max_step_m: {doc['eef_max_step_m']:.4f}",
        f"eef_max_step_rad: {doc['eef_max_step_rad']:.4f}",
        f"eef_max_lead_m: {doc['eef_max_lead_m']:.4f}",
        f"eef_max_lead_rad: {doc['eef_max_lead_rad']:.4f}",
    ]
    return "\n".join(indent + ln for ln in lines)


def build_document(raw: Dict[str, Any], data_root: str) -> Dict[str, Any]:
    """Turn the raw measurements into the rounded, documented JSON document."""
    # Quantise the observed envelopes outward FIRST, then derive: everything the
    # JSON prints is then exactly what the printed inputs produce (see GRID).
    q_lo, q_hi = outward(raw["joint_raw_min"], raw["joint_raw_max"])
    j_lo, j_hi = outward(*expand_symmetric(q_lo, q_hi))

    ws_min, ws_max = outward(raw["ws_min"], raw["ws_max"])
    ws_lo = ws_min - WS_MARGIN_M
    ws_hi = ws_max + WS_MARGIN_M
    ws_lo[2] = max(float(ws_min[2]) - WS_Z_MARGIN_M, WS_Z_FLOOR_M)

    s1p, s1r = _stats(raw["dp1"]), _stats(raw["dr1"])
    s1pu, s1ru = _stats(raw.get("dp1_unshifted", np.zeros(0))), \
        _stats(raw.get("dr1_unshifted", np.zeros(0)))
    s1pn, s1rn = _stats(raw.get("dp1_destaled_nearest", np.zeros(0))), \
        _stats(raw.get("dr1_destaled_nearest", np.zeros(0)))
    s30p, s30r = _stats(raw["dp30"]), _stats(raw["dr30"])

    taus = [t for t in raw.get("tau_q", []) if t is not None]
    residuals = [r for t, r in zip(raw.get("tau_q", []), raw.get("tau_residual_m", []))
                 if t is not None]

    doc: Dict[str, Any] = {
        "note": (
            "EEF-delta deploy envelope for carrot_in_pot (design row A2). Joint envelope: "
            f"{EXPAND}x expanded symmetric about the centre of the OBSERVED state q only "
            "(unlike the banana joint_safety_limits.json, the recorded `command` stream is "
            "excluded from the envelope -- in this EEF-mode session it leads the measured "
            "joints by ~0.9 s, so it is not a pose the arm ever held). UR7e hard limits "
            "+/-2pi (elbow +/-pi); these are well inside. Workspace box = observed TCP "
            f"min/max +/- {WS_MARGIN_M} m, except z_lo = max(raw_z_min - {WS_Z_MARGIN_M}, "
            f"{WS_Z_FLOOR_M}) (below the table is not a recoverable excursion). "
            f"eef_max_step_* = {STEP_SAFETY_FACTOR}x the largest single-frame achieved "
            "motion, rounded up (0.001 m / 0.01 rad); eef_max_lead_* = the median 30-frame "
            "(1 s) achieved motion, rounded up to 0.01. All motion statistics come from "
            "gello_policy.eef_space.delta_action on ACHIEVED joints, never from `command`. "
            "start_pose and delta_stats.frame_1 are on DE-STALED joints (linearly "
            "interpolated at cam1_t + tau_q, matching A1's converter); the joint envelope, "
            "the workspace box and frame_30 stay on the converter's tau=0 resampling, which "
            "a constant shift leaves alone and which is the wider (conservative) set because "
            "shifting drops the last 7.8% of frames off the end of the joint record. "
            f"All intervals are quantised OUTWARD onto a {GRID} grid before the derived "
            "values are computed, so joint_limits_* / workspace_* can be re-derived exactly "
            "from the joint_raw_* / workspace_raw printed in this same file."
        ),
        "source_data": os.path.abspath(data_root),
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(_REPO_ROOT),
        "generator": "scripts/dataset/make_carrot_eef_limits.py",
        "action_space_id": eef_space.ACTION_SPACE_ID,
        "tool_z_m": eef_space.TOOL_Z_M,
        "n_takes": raw["n_takes"],
        "n_frames": raw["n_frames"],
        "takes": raw["takes"],
        "joint_order": JOINT_ORDER,

        "start_pose": _round_list(raw["start_pose"], 4),
        "start_pose_std": _round_list(raw["start_pose_std"], 4),
        "start_pose_max_dev": _round_list(raw["start_pose_max_dev"], 4),
        "start_pose_note": (
            "Mean over all takes of the arm's REAL pose at the instant the first camera "
            "frame was taken: joints linearly interpolated at cam1_t[0] + tau_q (the same "
            "de-staling A1's converter applies), each take first brought to the literal "
            "value nearest take 1's by ur_kin.wrapped_nearest so a +/-2pi split on wrist_3 "
            "cannot corrupt the mean. std / max_dev are on this same basis. The naive "
            "tau=0 value is kept as start_pose_unshifted for reference."
        ),
        "start_gripper": round(raw["start_gripper"], 4),
        "start_tcp": _round_list(eef_space.tcp_pose(raw["start_pose"])[:3, 3], 4),
        "start_pose_unshifted": _round_list(raw["start_pose_unshifted"], 4),
        "start_pose_unshifted_std": _round_list(raw["start_pose_unshifted_std"], 4),
        "start_pose_unshifted_max_dev": _round_list(raw["start_pose_unshifted_max_dev"], 4),
        "start_tcp_unshifted": _round_list(
            eef_space.tcp_pose(raw["start_pose_unshifted"])[:3, 3], 4),
        "start_pose_unshifted_note": (
            "REFERENCE ONLY -- frame-0 joints read at cam1_t with no de-staling, i.e. the "
            "arm's pose ~0.89 s BEFORE the first camera frame. Frame 0 is the one quantity "
            "in this file that a constant time shift does NOT leave alone (the arm is "
            "already moving), and the two differ by up to 0.047 rad / 21 mm of TCP. Do not "
            "put this in the yaml: the pose the arm holds at arming must be the frame-0 "
            "state the policy was trained on, which is the de-staled one above."
        ),

        "start_branch": raw["start_branch"],
        "start_branch_note": (
            "ur_kin.branch_id(start_pose); policy_leader_node locks IK to this closed-form "
            f"branch. Verified on {raw['branch_checked']} sampled frames "
            f"(stride {raw['branch_stride']}): {raw['branch_n_off']} off-branch."
        ),
        "branch_frames_checked": raw["branch_checked"],
        "branch_frames_off": raw["branch_n_off"],

        "joint_state_shift_s": [None if t is None else round(t, 3)
                                for t in raw.get("tau_q", [])],
        "joint_state_shift_median_s": (round(float(np.median(taus)), 3) if taus else None),
        "joint_state_shift_residual_m": [round(float(r), 5) for r in residuals],
        "joint_state_shift_note": (
            "Per take: the shift tau that minimises median |TCP(q(t+tau)) - TCP(command(t))| "
            f"over tau in [{TAU_GRID[0]:.2f}, {TAU_GRID[-1]:.2f}] s step "
            f"{TAU_GRID[1] - TAU_GRID[0]:.2f}. The ur_joint_states cache is stamped stale by "
            "this much while command and the cameras are fresh. Accepted range "
            f"[{TAU_MIN_OK}, {TAU_MAX_OK}] s with residual < {TAU_RESIDUAL_MAX_M * 1000:.0f} "
            "mm. Only delta_stats.frame_1 uses the shifted joints (the shift changes which "
            "cache sample a 30 Hz frame snaps to, which inflates single-frame motion); the "
            "joint envelope, start pose, workspace box and frame_30 statistics are invariant "
            "to a constant time shift and stay on the converter's own resampling. Why the "
            "cache is stale is A7's row, not this one."
        ),

        "joint_raw_min": _round_list(q_lo, 4),
        "joint_raw_max": _round_list(q_hi, 4),
        "joint_limits_lo": _round_list(j_lo, 4),
        "joint_limits_hi": _round_list(j_hi, 4),
        "gripper_pos_raw_min": round(raw["gripper_pos_raw_min"], 4),
        "gripper_pos_raw_max": round(raw["gripper_pos_raw_max"], 4),
        "gripper_cmd_raw_min": round(raw["gripper_cmd_raw_min"], 4),
        "gripper_cmd_raw_max": round(raw["gripper_cmd_raw_max"], 4),

        "workspace_raw": {
            "min": _round_list(ws_min, 4),
            "max": _round_list(ws_max, 4),
            "axes": ["x", "y", "z"],
            "note": f"observed TCP min/max, quantised outward to {GRID} m",
        },
        "workspace_lo": _round_list(ws_lo, 4),
        "workspace_hi": _round_list(ws_hi, 4),

        "delta_stats": {
            "units": {"dp": "m", "drot": "rad"},
            "source": "eef_space.delta_action on achieved joints, cam1 master clock (30 fps)",
            "frame_1": {
                "n": int(raw["dp1"].size),
                "joints": "cam1_t + tau_q, LINEAR interpolation (A1's converter recipe); "
                          "drives eef_max_step_*",
                "dp": s1p,
                "drot": s1r,
                "jitter_note": (
                    "Why interpolation and not nearest: the joint cache runs at ~66.7 Hz "
                    "with a jittery dt (median 15.8 ms, p99 25.9 ms), so nearest-timestamp "
                    "resampling at 30 Hz snaps to a 2- or 3-sample gap, and tau_q = 0.89 s "
                    "lands the master grid on a bad phase where 8.8% of frames repeat a "
                    "sample (zero motion) and 24% jump 3. That inflates the single-frame "
                    "motion to p99 7.9 / max 11.5 mm (see frame_1_destaled_nearest) although "
                    "the arm never moved that fast. Interpolating the same shifted grid gives "
                    "p99 6.6 / max 8.9 mm, and it is what A1's converter does -- so these are "
                    "the actions the policy is actually trained to emit, and the clamp is "
                    "sized on them. REGENERATE this file if A1 changes the resampling again."
                ),
            },
            "frame_1_destaled_nearest": {
                "n": int(raw.get("dp1_destaled_nearest", np.zeros(0)).size),
                "joints": "cam1_t + tau_q, nearest-timestamp; audit only, nothing derived",
                "dp": s1pn,
                "drot": s1rn,
            },
            "frame_1_unshifted": {
                "n": int(raw.get("dp1_unshifted", np.zeros(0)).size),
                "joints": "converter resampling (tau=0); audit only, nothing is derived from it",
                "dp": s1pu,
                "drot": s1ru,
            },
            "frame_30": {
                "n": int(raw["dp30"].size),
                "joints": "converter resampling (tau=0); invariant to a constant shift",
                "dp": s30p,
                "drot": s30r,
            },
        },

        "eef_max_step_m": ceil_to(STEP_SAFETY_FACTOR * s1p["max"], 0.001),
        "eef_max_step_rad": ceil_to(STEP_SAFETY_FACTOR * s1r["max"], 0.01),
        "eef_max_lead_m": ceil_to(s30p["p50"], 0.01),
        "eef_max_lead_rad": ceil_to(s30r["p50"], 0.01),
    }
    if raw["branch_off"]:
        doc["branch_off_examples"] = raw["branch_off"][:20]

    doc["yaml_snippet"] = build_yaml_snippet(doc)
    return doc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=DEFAULT_DATA, help="raw take_* root")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSON path")
    ap.add_argument("--branch-stride", type=int, default=1,
                    help="check ur_kin.branch_id on every Nth frame (1 = all, ~0.6 ms/frame)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N takes (debug)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--allow-branch-mix", action="store_true",
                    help="do not fail when some frames are off the start branch")
    ap.add_argument("--allow-shift-outlier", action="store_true",
                    help="do not fail when a take's measured joint-cache shift is out of range")
    args = ap.parse_args(argv)

    raw = analyse(args.data, branch_stride=max(1, args.branch_stride),
                  limit=args.limit, verbose=not args.quiet)
    doc = build_document(raw, args.data)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    shift_problems = check_shifts(raw)
    print(f"\n[ok] wrote {args.out}")
    print(f"     {doc['n_takes']} takes / {doc['n_frames']} frames, branch {doc['start_branch']}, "
          f"{doc['branch_frames_off']}/{doc['branch_frames_checked']} off-branch")
    print(f"     joint-cache shift tau_q median {doc['joint_state_shift_median_s']} s, "
          f"{len(shift_problems)} take(s) out of range")
    print("\n--- yaml snippet (paste under policy_leader_node: ros__parameters:) ---")
    print(doc["yaml_snippet"])

    rc = 0
    if doc["branch_frames_off"] and not args.allow_branch_mix:
        print("\n[!] some frames are NOT on the start branch -- see branch_off_examples "
              "in the JSON. Branch-locked IK at deploy would refuse them.", file=sys.stderr)
        rc = 2
    if shift_problems and not args.allow_shift_outlier:
        print("\n[!] joint-cache shift out of the accepted window:", file=sys.stderr)
        for p in shift_problems[:10]:
            print(f"    {p}", file=sys.stderr)
        rc = 3
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
