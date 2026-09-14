#!/usr/bin/env python3
"""Single source of truth for the EEF-delta policy space (carrot_in_pot, 2026-09-14).

BOTH the dataset converter (py3.12, lr_env, no ROS) and the deploy node
(py3.10, rclpy) import THIS module so that "what the policy was trained on" and
"what the robot executes" are the same arithmetic. Pure numpy + ur_kin
(``ur_gello_bringup.ur_kin`` is ROS-free; put ``ros2_ur_ws/src/ur_gello_bringup``
on PYTHONPATH outside the ROS overlay).

Definitions
-----------
TCP    : flange (ur_kin.fk, base_link -> tool0) translated by +0.174 m along the
         flange +Z (same ``tool_r`` as ur7e_gello_eef.yaml; matches the recorder's
         ``tcp_pose`` table to ~1 mm, verified on take_01).
state  : (16,) float32
         [q1..q6 (rad, UR order, as observed),
          tcp_x, tcp_y, tcp_z (m, base frame),
          r11, r21, r31, r12, r22, r32  (6-D rotation = first two COLUMNS of R_tcp,
                                          column-major; continuous, no rotvec flip
                                          -- the carrot TCP sits ~175 deg from
                                          identity so a rotvec would wrap),
          grip_pos (0=open .. 1=closed)]
action : (7,) float32
         [dx, dy, dz (m, base frame), drx, dry, drz (rotation vector, rad, base
          frame), grip_cmd (0=open .. 1=closed, ABSOLUTE)]
         with, per dataset frame t,
             dp  = p_tcp(command_t) - p_tcp(q_t)
             dR  = R_tcp(command_t) @ R_tcp(q_t).T          (left-multiplied, base frame)
             drot = so3_log(dR)
         i.e. "where the bridge is telling the arm to go, relative to where the
         arm is right now". Deploy inverts this exactly:
             p_target = p_live + dp ;  R_target = so3_exp(drot) @ R_live
         then analytic IK on the flange pose with branch/limit filtering and
         nearest-to-live selection (:func:`ik_tcp_nearest`).

Nothing in here scales, clips or filters -- clamps are separate helpers so the
converter never applies them and the node always does.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ur_gello_bringup import ur_kin

TOOL_Z_M = 0.174
T_TOOL = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, TOOL_Z_M],
     [0.0, 0.0, 0.0, 1.0]], dtype=float,
)
T_TOOL_INV = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, -TOOL_Z_M],
     [0.0, 0.0, 0.0, 1.0]], dtype=float,
)

STATE_NAMES = [
    "ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6",
    "tcp_x", "tcp_y", "tcp_z",
    "tcp_r11", "tcp_r21", "tcp_r31", "tcp_r12", "tcp_r22", "tcp_r32",
    "grip_pos",
]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "grip_cmd"]
STATE_DIM = len(STATE_NAMES)    # 16
ACTION_DIM = len(ACTION_NAMES)  # 7
ACTION_SPACE_ID = "eef_delta_v1"  # bump if ANY definition above changes


# --------------------------------------------------------------------------- #
# kinematics
# --------------------------------------------------------------------------- #
def tcp_pose(q) -> np.ndarray:
    """base_link -> TCP homogeneous transform (4x4) for joint vector q (6,)."""
    return ur_kin.fk(np.asarray(q, dtype=float).reshape(6)) @ T_TOOL


def rot_to_6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, column-major: [r11, r21, r31, r12, r22, r32]."""
    R = np.asarray(R, dtype=float)
    return np.concatenate([R[:, 0], R[:, 1]])


def rot_from_6d(v) -> np.ndarray:
    """Gram-Schmidt inverse of :func:`rot_to_6d` (Zhou et al. 2019)."""
    v = np.asarray(v, dtype=float).reshape(6)
    a1, a2 = v[:3], v[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


# --------------------------------------------------------------------------- #
# state / action
# --------------------------------------------------------------------------- #
def state_vector(q, grip_pos: float) -> np.ndarray:
    """(16,) float64 observation.state for joint vector q and gripper position."""
    T = tcp_pose(q)
    return np.concatenate([
        np.asarray(q, dtype=float).reshape(6),
        T[:3, 3],
        rot_to_6d(T[:3, :3]),
        [float(grip_pos)],
    ])


def delta_action(q, q_cmd, grip_cmd: float) -> np.ndarray:
    """(7,) float64 action for one dataset frame: TCP(command) relative to TCP(q)."""
    T_q = tcp_pose(q)
    T_c = tcp_pose(q_cmd)
    dp = T_c[:3, 3] - T_q[:3, 3]
    dR = T_c[:3, :3] @ T_q[:3, :3].T
    drot = ur_kin.so3_log(dR)
    return np.concatenate([dp, drot, [float(grip_cmd)]])


def apply_delta(q_live, action) -> np.ndarray:
    """Target TCP pose (4x4) = live TCP(q_live) advanced by action[:6]."""
    a = np.asarray(action, dtype=float).reshape(-1)
    T = tcp_pose(q_live)
    out = np.eye(4)
    out[:3, :3] = ur_kin.so3_exp(a[3:6]) @ T[:3, :3]
    out[:3, 3] = T[:3, 3] + a[:3]
    return out


# --------------------------------------------------------------------------- #
# inverse kinematics (deploy side, also used by the round-trip tests)
# --------------------------------------------------------------------------- #
def ik_tcp_nearest(
    T_tcp: np.ndarray,
    q_seed,
    branch: Optional[int] = None,
    limit_margin: float = 0.0,
) -> Tuple[Optional[np.ndarray], dict]:
    """Invert a TCP target; return (q, info) with q nearest to q_seed, or (None, info).

    Order (mirrors ur_gello_bringup.eef_delta.EefDeltaController._ik_branchlock):
      analytic enumerate on the flange pose -> keep the requested closed-form
      branch when it has solutions (skip the filter near the elbow / wrist_2 merge
      points where labels collapse) -> unwrap every candidate to the seed
      (wrapped_nearest) -> drop candidates outside the UR joint limits when any
      remain -> pick argmin of max|q - seed|.
    info: {"n_solutions", "branch", "on_branch", "max_dev"}.
    """
    q_seed = np.asarray(q_seed, dtype=float).reshape(6)
    T_flange = np.asarray(T_tcp, dtype=float) @ T_TOOL_INV
    merge = (abs(float(np.remainder(q_seed[2] + np.pi, 2 * np.pi) - np.pi)) < 0.05
             or abs(float(np.remainder(q_seed[4] + np.pi, 2 * np.pi) - np.pi)) < 0.05)
    prefer = None if (merge or branch is None or branch < 0) else int(branch)
    tagged = ur_kin._ik_analytic_tagged(T_flange, prefer)
    info = {"n_solutions": len(tagged), "branch": -1, "on_branch": False, "max_dev": float("nan")}
    if not tagged:
        return None, info
    pool = tagged
    if prefer is not None:
        on = [(b, q) for (b, q) in tagged if b == prefer]
        if on:
            pool = on
            info["on_branch"] = True
    reanch = [(b, ur_kin.wrapped_nearest(q, q_seed)) for (b, q) in pool]
    limited = [t for t in reanch if ur_kin.within_joint_limits(t[1], margin=limit_margin)]
    if limited:
        reanch = limited
    bid, best = min(reanch, key=lambda t: float(np.max(np.abs(t[1] - q_seed))))
    info["branch"] = int(bid)
    info["max_dev"] = float(np.max(np.abs(best - q_seed)))
    return best, info


# --------------------------------------------------------------------------- #
# clamps (deploy side ONLY -- the converter must never call these)
# --------------------------------------------------------------------------- #
def clamp_delta(action, max_pos_m: float, max_rot_rad: float) -> Tuple[np.ndarray, bool, bool]:
    """Scale the translation / rotation parts of a delta DOWN to the given norms.

    Norm-proportional (direction preserving), never per-axis -- see CLAUDE.md
    "개입 액션을 축별 np.clip으로 자르지 말 것". Returns (action, pos_clamped, rot_clamped).
    """
    a = np.array(action, dtype=float).reshape(-1).copy()
    pc = rc = False
    n = float(np.linalg.norm(a[:3]))
    if n > max_pos_m > 0.0:
        a[:3] *= max_pos_m / n
        pc = True
    r = float(np.linalg.norm(a[3:6]))
    if r > max_rot_rad > 0.0:
        a[3:6] *= max_rot_rad / r
        rc = True
    return a, pc, rc


def clamp_position_box(p, lo, hi) -> Tuple[np.ndarray, bool]:
    """Axis-wise clip of a TCP position to the workspace box [lo, hi] (m)."""
    p = np.asarray(p, dtype=float).reshape(3)
    lo = np.asarray(lo, dtype=float).reshape(3)
    hi = np.asarray(hi, dtype=float).reshape(3)
    c = np.minimum(np.maximum(p, lo), hi)
    return c, bool(np.any(c != p))
