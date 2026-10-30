"""Thin, tested IK/FK wrapper over `ur_gello_bringup.ur_kin` for the eval oracle.

Conventions (DESIGN.md §1.2 / §1.4): MuJoCo world frame == ROS base_link; the
TCP is the flange (`ur_kin.fk`) pushed 0.174 m along tool +Z (`T_TOOL`, the
measured Robotiq 2F-85 endpoint, `ur7e_gello_eef.yaml` tool_r_xyz_rpy). Joint
order is UR order; angles are literal (unwrapped) radians on the carrot data's
-pi shoulder_pan branch.

`tcp_to_joints` is the branch-locked selection idea of
`eef_delta.EefDeltaController._ik_branchlock`, reduced to what a scripted
policy needs: enumerate the closed-form solutions of the FLANGE pose, keep the
ones on the requested closed-form branch (the home pose's branch by default),
re-anchor each to the seed with `wrapped_nearest` (adds 2*pi*k only, never
changes branch), drop the ones outside the joint envelope, and return the one
nearest the seed. If no on-branch solution survives the envelope the search
falls back to every branch (the oracle prefers finishing the task over branch
purity; the real deploy's max_dev clamp still bounds any jump).

Default envelope = the carrot deploy envelope
(`gello_policy/config/carrot_eef_limits.json` joint_limits_lo/hi, generated
2026-09-14 from 54 real takes; provenance in that file). The yaml `eval:` block
F1 adds carries the same numbers; callers pass them explicitly.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ur_gello_bringup import ur_kin

TOOL_Z_M = 0.174
T_TOOL = ur_kin.xyz_rpy_to_T([0.0, 0.0, TOOL_Z_M, 0.0, 0.0, 0.0])
_T_TOOL_INV = np.linalg.inv(T_TOOL)

# carrot_eef_limits.json (2026-09-14, 54 takes / 18557 frames), 1.2x expanded
# symmetric envelope of the observed joints. J1 is on the -pi branch.
DEFAULT_JOINT_LIMITS_LO = np.array([-3.8049, -1.8055, 1.2649, -2.6913, -1.9309, -4.6440])
DEFAULT_JOINT_LIMITS_HI = np.array([-2.5787, -0.8823, 2.1442, -1.4379, -1.2014, -1.8222])

Limits = Tuple[Sequence[float], Sequence[float]]


def default_limits() -> Tuple[np.ndarray, np.ndarray]:
    return DEFAULT_JOINT_LIMITS_LO.copy(), DEFAULT_JOINT_LIMITS_HI.copy()


def _as_limits(limits: Optional[Limits]) -> Tuple[np.ndarray, np.ndarray]:
    if limits is None:
        lo, hi = ur_kin.JOINT_LIMITS[:, 0], ur_kin.JOINT_LIMITS[:, 1]
    else:
        lo = np.asarray(limits[0], dtype=float).reshape(6)
        hi = np.asarray(limits[1], dtype=float).reshape(6)
    if np.any(hi <= lo):
        raise ValueError(f"joint limits must satisfy lo < hi per joint: lo={lo}, hi={hi}")
    return lo, hi


def within_limits(q: np.ndarray, limits: Optional[Limits], margin: float = 0.0) -> bool:
    lo, hi = _as_limits(limits)
    q = np.asarray(q, dtype=float).reshape(6)
    return bool(np.all(q >= lo + margin) and np.all(q <= hi - margin))


def tcp_pose_from_q(q: Sequence[float]) -> np.ndarray:
    """4x4 TCP pose in the world/base frame: fk(q) @ T_TOOL."""
    return ur_kin.fk(np.asarray(q, dtype=float).reshape(6)) @ T_TOOL


def flange_from_tcp(T_tcp: np.ndarray) -> np.ndarray:
    return np.asarray(T_tcp, dtype=float) @ _T_TOOL_INV


def tcp_to_joints(T_tcp: np.ndarray, q_seed: Sequence[float], limits: Optional[Limits] = None,
                  branch: Optional[int] = None, margin: float = 0.0,
                  allow_branch_fallback: bool = True) -> Optional[np.ndarray]:
    """Analytic IK of a TCP pose, nearest `q_seed` on `branch`, inside `limits`.

    `branch` = closed-form branch id (`ur_kin.branch_id`) to lock to; None locks
    to the seed's own branch. `limits` = (lo, hi) 6-vectors of literal radians
    (None = the UR hard limits). Returns None when nothing is reachable.
    """
    q_seed = np.asarray(q_seed, dtype=float).reshape(6)
    lo, hi = _as_limits(limits)
    if branch is None:
        branch = ur_kin.branch_id(q_seed)
    T_flange = flange_from_tcp(T_tcp)
    tagged = ur_kin._ik_analytic_tagged(T_flange, prefer_branch=branch if branch >= 0 else None)
    if not tagged:
        return None

    def pick(pool):
        best, best_d = None, float("inf")
        for _bid, q in pool:
            qa = ur_kin.wrapped_nearest(np.asarray(q, dtype=float), q_seed)
            if not (np.all(qa >= lo + margin) and np.all(qa <= hi - margin)):
                continue
            d = float(np.max(np.abs(qa - q_seed)))
            if d < best_d:
                best, best_d = qa, d
        return best

    on_branch = [(b, q) for b, q in tagged if b == branch]
    q = pick(on_branch) if on_branch else None
    if q is None and allow_branch_fallback:
        q = pick(tagged)
    return q


def interpolate_joint(q_from: Sequence[float], q_to: Sequence[float], n: int) -> np.ndarray:
    """`n` joint-space waypoints from `q_from` (exclusive) to `q_to` (inclusive), shape (n, 6).

    n == 0 -> empty array; n == 1 -> just `q_to`."""
    a = np.asarray(q_from, dtype=float).reshape(6)
    b = np.asarray(q_to, dtype=float).reshape(6)
    n = int(n)
    if n <= 0:
        return np.zeros((0, 6))
    s = np.linspace(0.0, 1.0, n + 1)[1:, None]
    return a[None, :] + s * (b - a)[None, :]


def interpolate_pose(T_from: np.ndarray, T_to: np.ndarray, n: int) -> list:
    """`n` SE(3) waypoints (position linear, rotation geodesic), `T_from` exclusive, `T_to` inclusive."""
    T_from = np.asarray(T_from, dtype=float)
    T_to = np.asarray(T_to, dtype=float)
    n = int(n)
    if n <= 0:
        return []
    R0 = T_from[:3, :3]
    w = ur_kin.so3_log(R0.T @ T_to[:3, :3])
    p0, p1 = T_from[:3, 3], T_to[:3, 3]
    out = []
    for s in np.linspace(0.0, 1.0, n + 1)[1:]:
        T = np.eye(4)
        T[:3, :3] = R0 @ ur_kin.so3_exp(s * w)
        T[:3, 3] = p0 + s * (p1 - p0)
        out.append(T)
    return out


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[float, float]:
    """(position error [m], rotation error [rad]) between two poses."""
    dp = float(np.linalg.norm(np.asarray(T_a)[:3, 3] - np.asarray(T_b)[:3, 3]))
    dR = float(np.linalg.norm(ur_kin.so3_log(np.asarray(T_a)[:3, :3].T @ np.asarray(T_b)[:3, :3])))
    return dp, dR


__all__ = [
    "T_TOOL", "TOOL_Z_M", "DEFAULT_JOINT_LIMITS_LO", "DEFAULT_JOINT_LIMITS_HI", "default_limits",
    "tcp_pose_from_q", "flange_from_tcp", "tcp_to_joints", "interpolate_joint", "interpolate_pose",
    "pose_error", "within_limits",
]
