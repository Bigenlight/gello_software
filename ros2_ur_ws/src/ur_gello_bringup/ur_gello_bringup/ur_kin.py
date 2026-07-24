"""Pure-numpy kinematics for a UR7e (mirrors UR5e nominal DH).

This module is intentionally free of any ROS / rclpy dependency: it is the shared
math core called by the GELLO->UR bridge (eef_delta.py) and by tests. Only numpy
is required; pyyaml is used *optionally* by the DH loader (falls back to the
built-in nominal values when absent).

Conventions (fixed by the project grounding, do not silently change)
--------------------------------------------------------------------
* UR joint order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3].
* Standard (distal) Denavit-Hartenberg, per-joint (d_i, a_i, alpha_i):

      i :   d        a        alpha
      1 :   d1       0        +pi/2
      2 :   0        a2       0
      3 :   0        a3       0
      4 :   d4       0        +pi/2
      5 :   d5       0        -pi/2
      6 :   d6       0        0

  with the UR7e (== UR5e) nominal values
      d1=0.1625, a2=-0.425, a3=-0.3922, d4=0.1333, d5=0.0997, d6=0.0996.
  A_i = Rot_z(theta_i) . Trans_z(d_i) . Trans_x(a_i) . Rot_x(alpha_i)
  and fk(q) = A_1 A_2 A_3 A_4 A_5 A_6  ==  base_link -> tool0 (flange).

* Rotations are carried as 3x3 matrices on the control path. Quaternions are
  xyzw (scalar-last, ROS convention) and used only for diagnostics/conversion.

* Geometric Jacobian is [Jv; Jw] in the base frame (top 3 rows linear velocity
  of the tool0 origin, bottom 3 rows angular velocity).

The analytic IK is a from-scratch numpy vendoring of the closed-form UR inverse
(the Andersen / Hawkins solution structure). Each closed-form candidate is
polished by a couple of Newton steps (which only remove float error, never
change branch) and FK-verified before being returned, so a returned solution is
guaranteed to reproduce the target pose.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Nominal DH + limit constants (source: ur_description default_kinematics.yaml; #
# re-verify on the real robot PC -- see config/ur7e_dh.yaml).                   #
# --------------------------------------------------------------------------- #
_D1 = 0.1625
_A2 = -0.425
_A3 = -0.3922
_D4 = 0.1333
_D5 = 0.0997
_D6 = 0.0996

# Per-joint DH arrays in UR joint order.
D = np.array([_D1, 0.0, 0.0, _D4, _D5, _D6], dtype=float)
A = np.array([0.0, _A2, _A3, 0.0, 0.0, 0.0], dtype=float)
ALPHA = np.array([math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0], dtype=float)

# Joint limits: elbow (index 2) is physically +/-pi (shoulder_lift interference,
# ros-industrial/universal_robot#265); the other five are +/-2pi. Re-confirm on
# the real robot PC. Velocity limit is pi rad/s (180 deg/s) on every joint.
JOINT_LIMITS = np.array(
    [
        [-2 * math.pi, 2 * math.pi],
        [-2 * math.pi, 2 * math.pi],
        [-math.pi, math.pi],  # elbow: do NOT unwrap past +/-pi
        [-2 * math.pi, 2 * math.pi],
        [-2 * math.pi, 2 * math.pi],
        [-2 * math.pi, 2 * math.pi],
    ],
    dtype=float,
)
VELOCITY_LIMITS = np.full(6, math.pi, dtype=float)

# Branch weighting and Jacobian conditioning constants (grounding).
W_BRANCH = np.diag([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])
CHAR_LENGTH = 0.30

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Small SO(3)/SE(3) utilities                                                  #
# --------------------------------------------------------------------------- #
def _skew(w: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ]
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Exponential map so(3) -> SO(3) (Rodrigues), 3-vector -> 3x3 rotation."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    K = _skew(w)
    if theta < 1e-9:
        # 2nd-order series; stable as theta -> 0.
        return np.eye(3) + K + 0.5 * (K @ K)
    K = K / theta
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map SO(3) -> so(3), 3x3 rotation -> 3-vector (axis*angle).

    Routed through the (Shepperd) quaternion so it stays well-conditioned at
    every angle, including theta -> 0 and theta -> pi where the classic
    trace/antisymmetric formula loses accuracy."""
    R = np.asarray(R, dtype=float)
    q = mat_to_quat_xyzw(R)  # [x, y, z, w], unit
    vec = q[:3]
    w = q[3]
    if w < 0.0:  # canonical hemisphere -> theta in [0, pi]
        vec = -vec
        w = -w
    nv = float(np.linalg.norm(vec))
    if nv < 1e-12:
        return np.zeros(3)
    theta = 2.0 * math.atan2(nv, w)
    return (theta / nv) * vec


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map se(3) -> SE(3). xi = (v, w) 6-vector -> 4x4 transform."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    v = xi[:3]
    w = xi[3:]
    theta = float(np.linalg.norm(w))
    R = so3_exp(w)
    T = np.eye(4)
    T[:3, :3] = R
    if theta < 1e-9:
        V = np.eye(3) + 0.5 * _skew(w) + (1.0 / 6.0) * (_skew(w) @ _skew(w))
    else:
        K = _skew(w) / theta
        V = (
            np.eye(3)
            + (1.0 - math.cos(theta)) / theta * K
            + (theta - math.sin(theta)) / theta * (K @ K)
        )
    T[:3, 3] = V @ v
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithm map SE(3) -> se(3). 4x4 transform -> (v, w) 6-vector."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    p = T[:3, 3]
    w = so3_log(R)
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        Vinv = np.eye(3) - 0.5 * _skew(w)
    else:
        K = _skew(w) / theta
        # Vinv = I - 1/2 K*theta + (1 - theta*cot(theta/2)/2)/theta^2 * (K*theta)^2
        half = theta / 2.0
        Vinv = (
            np.eye(3)
            - 0.5 * _skew(w)
            + (1.0 / theta**2) * (1.0 - (theta * math.cos(half)) / (2.0 * math.sin(half)))
            * (_skew(w) @ _skew(w))
        )
    v = Vinv @ p
    return np.concatenate([v, w])


# --------------------------------------------------------------------------- #
# Rotation <-> quaternion (xyzw, ROS scalar-last)                              #
# --------------------------------------------------------------------------- #
def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit quaternion [x, y, z, w] (scalar-last)."""
    R = np.asarray(R, dtype=float)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [x, y, z, w] (scalar-last) -> 3x3 rotation."""
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rpy_to_mat(rpy) -> np.ndarray:
    """Fixed-axis roll-pitch-yaw (X, Y, Z) -> R = Rz(yaw) Ry(pitch) Rx(roll).

    Shared by eef_delta (R_align / tool transforms) and by the keepout tool
    resolution below, so there is exactly ONE rpy convention in the package."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def xyz_rpy_to_T(vec6) -> np.ndarray:
    """[x, y, z, roll, pitch, yaw] -> 4x4 homogeneous transform."""
    v = np.asarray(vec6, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = rpy_to_mat(v[3:])
    T[:3, 3] = v[:3]
    return T


# --------------------------------------------------------------------------- #
# Forward kinematics, link origins, Jacobian, conditioning                    #
# --------------------------------------------------------------------------- #
# cos/sin of the (constant) DH twist angles, hoisted out of the hot path.
_CA = np.cos(ALPHA)
_SA = np.sin(ALPHA)


def _dh_into(M: np.ndarray, theta: float, d: float, a: float,
             ca: float, sa: float) -> np.ndarray:
    """Fill `M` (4x4) with the standard-DH link transform. Element-wise stores
    into a preallocated buffer: np.array() on a nested list was ~15% of the
    whole control tick, and this is called 6x per FK and ~28x per analytic IK."""
    ct, st = math.cos(theta), math.sin(theta)
    M[0, 0] = ct
    M[0, 1] = -st * ca
    M[0, 2] = st * sa
    M[0, 3] = a * ct
    M[1, 0] = st
    M[1, 1] = ct * ca
    M[1, 2] = -ct * sa
    M[1, 3] = a * st
    M[2, 0] = 0.0
    M[2, 1] = sa
    M[2, 2] = ca
    M[2, 3] = d
    M[3, 0] = 0.0
    M[3, 1] = 0.0
    M[3, 2] = 0.0
    M[3, 3] = 1.0
    return M


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard-DH link transform (kept as the general-alpha public form)."""
    return _dh_into(np.empty((4, 4)), theta, d, a, math.cos(alpha), math.sin(alpha))


def _inv_rigid(T: np.ndarray) -> np.ndarray:
    """Inverse of a HOMOGENEOUS (rigid) 4x4: [[R^T, -R^T p], [0, 1]].

    Mathematically identical to np.linalg.inv for a rigid transform but ~2x
    cheaper (no LU): the analytic IK below inverts 28 such matrices per call,
    which measured as ~23% of a moving control tick."""
    R = T[:3, :3]
    Ti = np.empty((4, 4))
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ T[:3, 3])
    Ti[3, 0] = 0.0
    Ti[3, 1] = 0.0
    Ti[3, 2] = 0.0
    Ti[3, 3] = 1.0
    return Ti


def _fk_frames(q: np.ndarray) -> List[np.ndarray]:
    """Cumulative transforms base->frame_i for i = 0..6 (7 matrices)."""
    q = np.asarray(q, dtype=float).reshape(6)
    frames = [np.eye(4)]
    T = np.eye(4)
    M = np.empty((4, 4))
    for i in range(6):
        T = T @ _dh_into(M, q[i], D[i], A[i], _CA[i], _SA[i])
        frames.append(T)
    return frames


def fk(q: np.ndarray) -> np.ndarray:
    """base_link -> tool0 (flange) homogeneous transform, 4x4.

    Does not build the intermediate frame list (this is the single most-called
    function on the control path -- ~34 calls per tick through the analytic IK's
    verify step)."""
    q = np.asarray(q, dtype=float).reshape(6)
    T = np.eye(4)
    M = np.empty((4, 4))
    for i in range(6):
        T = T @ _dh_into(M, q[i], D[i], A[i], _CA[i], _SA[i])
    return T


def link_origins(q: np.ndarray, T_tool: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Origins (base frame, 3-vectors) of frames 1..6:
    shoulder, elbow, wrist_1, wrist_2, wrist_3, flange(tool0).

    NOTE the last entry is the FLANGE, not the tool centre point: with a
    Robotiq 2F-85 the actual endpoint is another ~0.174 m along the flange +Z,
    and that is precisely the part that reaches the table.  Pass ``T_tool``
    (flange -> TCP, 4x4) to append the TCP origin as a 7th point; ``None``
    (the default) preserves the historical 6-point flange-only behaviour."""
    frames = _fk_frames(q)
    out = [frames[i][:3, 3].copy() for i in range(1, 7)]
    if T_tool is not None:
        T_tool = np.asarray(T_tool, dtype=float).reshape(4, 4)
        out.append((frames[6] @ T_tool)[:3, 3].copy())
    return out


def jacobian(q: np.ndarray) -> np.ndarray:
    """Base-frame geometric Jacobian [Jv; Jw], 6x6."""
    frames = _fk_frames(q)
    o_end = frames[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z = frames[i][:3, 2]  # axis of joint i+1 = z of frame i
        o = frames[i][:3, 3]
        J[:3, i] = np.cross(z, o_end - o)
        J[3:, i] = z
    return J


def sigma_min(q: np.ndarray, L: float = CHAR_LENGTH) -> float:
    """Minimum singular value of the dimensionless weighted Jacobian.

    J_w = diag(1/L, 1/L, 1/L, 1, 1, 1) . J_geom ; returns min(svd(J_w))."""
    J = jacobian(q)
    scale = np.diag([1.0 / L, 1.0 / L, 1.0 / L, 1.0, 1.0, 1.0])
    Jw = scale @ J
    return float(np.linalg.svd(Jw, compute_uv=False)[-1])


# --------------------------------------------------------------------------- #
# Joint limits / keepout                                                       #
# --------------------------------------------------------------------------- #
def within_joint_limits(q: np.ndarray, margin: float = 0.0) -> bool:
    """True iff every joint is inside [lo+margin, hi-margin]."""
    q = np.asarray(q, dtype=float).reshape(6)
    lo = JOINT_LIMITS[:, 0] + margin
    hi = JOINT_LIMITS[:, 1] - margin
    return bool(np.all(q >= lo) and np.all(q <= hi))


# Samples per arm link segment for the geometric keepout sweep. Odd so the exact
# midpoint of every link is tested (see keepout_ok / _link_segment_samples).
_KEEPOUT_SAMPLES_PER_LINK = 17


def _link_segment_samples(
    q: np.ndarray,
    n: int = _KEEPOUT_SAMPLES_PER_LINK,
    T_tool: Optional[np.ndarray] = None,
):
    """Points sampled along every ARM link segment (shoulder->elbow->wrist_1->
    wrist_2->wrist_3->flange), including both endpoints and the midpoint.

    With ``T_tool`` given the flange->TCP segment (the gripper body) is swept
    too, so a tool that dips into a keepout region is seen along its whole
    length and not merely at its tip.

    The base->shoulder pedestal segment is deliberately excluded: it is the
    robot's own column and is expected to sit on the base axis (so it must not
    trip a base_cylinder pedestal-protection constraint)."""
    origins = link_origins(q, T_tool)  # frames 1..6 (..TCP if a tool is given)
    ts = np.linspace(0.0, 1.0, n)
    samples = []
    for a, b in zip(origins[:-1], origins[1:]):
        seg = b - a
        if float(np.dot(seg, seg)) < 1e-24:
            # Degenerate (e.g. an identity tool transform): the endpoint is
            # already in the list via the previous segment.
            samples.append(a.copy())
            continue
        for t in ts:
            samples.append(a + t * seg)
    return samples


def _resolve_tool(cfg: dict, T_tool: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """flange->TCP transform for a keepout check, or None.

    Precedence: explicit ``T_tool`` argument, then ``cfg['tool']`` (4x4), then
    ``cfg['tool_xyz_rpy']`` (6-vector).  Absent everywhere -> None (flange-only,
    the historical behaviour)."""
    if T_tool is not None:
        return np.asarray(T_tool, dtype=float).reshape(4, 4)
    if isinstance(cfg, dict):
        if cfg.get("tool") is not None:
            return np.asarray(cfg["tool"], dtype=float).reshape(4, 4)
        if cfg.get("tool_xyz_rpy") is not None:
            return xyz_rpy_to_T(cfg["tool_xyz_rpy"])
    return None


# Keys that actually impose a constraint. Anything else in the cfg (tool,
# margin, link_radius, ...) only *modifies* a constraint, so a cfg holding none
# of these is a no-op no matter what else it carries.
_KEEPOUT_CONSTRAINT_KEYS = ("floor_z", "base_cylinder", "cylinder", "halfplane")


def keepout_ok(q: np.ndarray, cfg: dict, T_tool: Optional[np.ndarray] = None) -> bool:
    """Geometric keepout check.

    For cylinder / base_cylinder / halfplane constraints the whole arm link
    *segments* are sampled (not just the link origins): a link that pierces a
    keepout cylinder with both endpoints outside would otherwise read as a false
    "safe". floor_z stays an endpoints-only check because z is linear along a
    segment, so an endpoint is always the extremum.

    THE TOOL.  ``link_origins`` stops at the flange (tool0).  A real end
    effector (Robotiq 2F-85: ~0.174 m along flange +Z) hangs past it and is the
    part that actually reaches the table, so the check is blind to it unless the
    flange->TCP transform is supplied -- via the ``T_tool`` argument, or via
    ``cfg['tool']`` / ``cfg['tool_xyz_rpy']``.  When supplied, the TCP point is
    added to the checked set and the flange->TCP segment is swept.

    LINK RADIUS.  All sampled points are link *centrelines*; a UR7e upper arm is
    ~0.09 m in diameter and a gripper is wider still.  ``cfg['link_radius']``
    inflates the arm by that amount inside every test, so an operator setting
    ``floor_z`` states the physical table height rather than hand-computing
    ``table + tool_length + link_radius`` (and getting it wrong).

    cfg keys (all optional; absent -> that constraint is not enforced, so an
    EMPTY cfg is always a no-op):
      floor_z   : float   -- every checked point must have z >= floor_z + inflate.
      margin    : float   -- safety margin (default 0.0).
      link_radius : float -- link/tool radius (default 0.0); added to `margin`
                       in every test.  inflate = margin + link_radius.
      tool / tool_xyz_rpy : flange->TCP transform (4x4 / [x,y,z,r,p,y]); see
                       THE TOOL above.  Overridden by the `T_tool` argument.
      base_cylinder : {radius, height} -- no sampled arm point (up to `height`)
                       may lie inside the vertical cylinder of that radius around
                       the base z-axis (protects the pedestal).
      cylinder  : {point:[x,y,z], axis:[x,y,z], radius} -- keepout cylinder;
                       no sampled arm point may be within `radius` of that line.
      halfplane : {point:[x,y,z], normal:[x,y,z]} -- every sampled arm point must
                       be on the +normal side: dot(pt-point, normal) >= inflate.
    """
    if not cfg or not any(k in cfg for k in _KEEPOUT_CONSTRAINT_KEYS):
        return True  # nothing to enforce -- never touch FK

    margin = float(cfg.get("margin", 0.0))
    radius = float(cfg.get("link_radius", 0.0))
    inflate = margin + radius
    tool = _resolve_tool(cfg, T_tool)
    origins = link_origins(q, tool)

    if "floor_z" in cfg:
        fz = float(cfg["floor_z"])
        for o in origins:
            if o[2] < fz + inflate:
                return False

    # The remaining constraints are non-linear (or vertex-sensitive) along a
    # link, so sweep sampled points down each arm segment.
    need_sweep = ("base_cylinder" in cfg) or ("cylinder" in cfg) or ("halfplane" in cfg)
    samples = _link_segment_samples(q, T_tool=tool) if need_sweep else []

    if "base_cylinder" in cfg:
        bc = cfg["base_cylinder"]
        r = float(bc["radius"])
        h = float(bc["height"])
        for o in samples:
            # The height test is inflated too: a centreline just above `h` still
            # has material below it once the link has a radius.
            if o[2] <= h + radius and math.hypot(o[0], o[1]) < r + inflate:
                return False

    if "cylinder" in cfg:
        cy = cfg["cylinder"]
        p = np.asarray(cy["point"], dtype=float)
        axis = np.asarray(cy["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        r = float(cy["radius"])
        for o in samples:
            d = o - p
            perp = d - np.dot(d, axis) * axis
            if np.linalg.norm(perp) < r + inflate:
                return False

    if "halfplane" in cfg:
        hp = cfg["halfplane"]
        p = np.asarray(hp["point"], dtype=float)
        n = np.asarray(hp["normal"], dtype=float)
        n = n / np.linalg.norm(n)
        for o in samples:
            if np.dot(o - p, n) < inflate:
                return False

    return True


# --------------------------------------------------------------------------- #
# Numeric IK (damped least squares Newton)                                     #
# --------------------------------------------------------------------------- #
def _pose_error(T_cur: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """6-vector [dp; dw] consistent with the geometric Jacobian [Jv; Jw]."""
    dp = T_target[:3, 3] - T_cur[:3, 3]
    dR = T_target[:3, :3] @ T_cur[:3, :3].T
    dw = so3_log(dR)
    return np.concatenate([dp, dw])


def ik_numeric(
    T_target: np.ndarray,
    seed: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-8,
    damping: float = 1e-3,
) -> Optional[np.ndarray]:
    """Damped-least-squares Newton IK starting from `seed`. None on failure.

    tol contract: a returned solution is guaranteed to satisfy
    ``||pose_error|| < tol`` -- the *requested* tol, with no hidden 1e-6 floor.
    If the iteration cannot reach `tol` within `max_iter` (e.g. an unreachable
    target, or a near-singular pose where the damped step plateaus above tol)
    the function returns None rather than silently handing back a coarser
    solution. The only hard limit is float precision (~1e-12); ask below that
    and you will simply get None."""
    q = np.asarray(seed, dtype=float).reshape(6).copy()
    T_target = np.asarray(T_target, dtype=float)
    for _ in range(max_iter):
        T_cur = fk(q)
        e = _pose_error(T_cur, T_target)
        if np.linalg.norm(e) < tol:
            return q
        J = jacobian(q)
        # dq = J^T (J J^T + lambda^2 I)^-1 e
        JJt = J @ J.T + (damping**2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)
        # Cap step for stability near singularities.
        step = np.linalg.norm(dq)
        if step > 0.5:
            dq *= 0.5 / step
        q = q + dq
    T_cur = fk(q)
    # Gate on the *requested* tol (not max(tol, 1e-6)): never return a solution
    # that fails to meet what the caller asked for.
    if np.linalg.norm(_pose_error(T_cur, T_target)) < tol:
        return q
    return None


# --------------------------------------------------------------------------- #
# Analytic IK (closed form, all 8 branches)                                    #
# --------------------------------------------------------------------------- #
def _clamp_unit(x: float) -> Optional[float]:
    if x > 1.0:
        if x > 1.0 + 1e-9:
            return None
        return 1.0
    if x < -1.0:
        if x < -1.0 - 1e-9:
            return None
        return -1.0
    return x


def _branch_bits(i1: int, i5: int, i3: int) -> int:
    """Deterministic 3-bit branch id: bit2 = theta1 sel, bit1 = theta5 sel,
    bit0 = theta3 sel (each 0/1)."""
    return (i1 << 2) | (i5 << 1) | i3


def _ik_analytic_raw(T: np.ndarray):
    """Yield (branch_id, q) closed-form candidates (unpolished, unverified)."""
    d1, a2, a3, d4, d5, d6 = _D1, _A2, _A3, _D4, _D5, _D6
    T = np.asarray(T, dtype=float)

    # ---- theta1 : two branches ----
    p05 = T @ np.array([0.0, 0.0, -d6, 1.0])
    p05x, p05y = p05[0], p05[1]
    R05 = math.hypot(p05x, p05y)
    if R05 < abs(d4) - 1e-9:
        return
    ratio = _clamp_unit(d4 / R05) if R05 > _EPS else None
    if ratio is None:
        return
    psi = math.atan2(p05y, p05x)
    phi = math.acos(ratio)
    theta1_opts = [psi + phi + math.pi / 2.0, psi - phi + math.pi / 2.0]

    p06x, p06y = T[0, 3], T[1, 3]

    for i1, t1 in enumerate(theta1_opts):
        s1, c1 = math.sin(t1), math.cos(t1)
        # T01 depends only on theta1 -- hoisted out of the theta5 loop.
        T01_inv = _inv_rigid(_dh(t1, D[0], A[0], ALPHA[0]))

        # ---- theta5 : two branches ----
        c5arg = _clamp_unit((p06x * s1 - p06y * c1 - d4) / d6)
        if c5arg is None:
            continue
        a5 = math.acos(c5arg)
        for i5, t5 in enumerate((a5, -a5)):
            s5 = math.sin(t5)

            # ---- theta6 ----
            if abs(s5) < 1e-8:
                t6 = 0.0  # wrist singular: theta6 free, arbitrary pick
            else:
                num = (-T[0, 1] * s1 + T[1, 1] * c1) / s5
                den = (T[0, 0] * s1 - T[1, 0] * c1) / s5
                t6 = math.atan2(num, den)

            # ---- theta2, theta3, theta4 from planar sub-problem ----
            # All of these are rigid transforms, so _inv_rigid == np.linalg.inv
            # (exactly, to float rounding) at half the cost.
            T45 = _dh(t5, D[4], A[4], ALPHA[4])
            T56 = _dh(t6, D[5], A[5], ALPHA[5])
            T14 = T01_inv @ T @ _inv_rigid(T56) @ _inv_rigid(T45)
            p13 = T14 @ np.array([0.0, -d4, 0.0, 1.0])
            p13x, p13y = p13[0], p13[1]
            norm2 = p13x * p13x + p13y * p13y
            c3arg = _clamp_unit((norm2 - a2 * a2 - a3 * a3) / (2.0 * a2 * a3))
            if c3arg is None:
                continue
            a3ang = math.acos(c3arg)
            norm = math.sqrt(norm2)
            for i3, t3 in enumerate((a3ang, -a3ang)):
                # theta2
                denom = norm if norm > _EPS else _EPS
                t2 = math.atan2(-p13y, -p13x) - math.asin(
                    max(-1.0, min(1.0, -a3 * math.sin(t3) / denom))
                )
                # theta4 from T34 = inv(T23) inv(T12) T14
                T12 = _dh(t2, D[1], A[1], ALPHA[1])
                T23 = _dh(t3, D[2], A[2], ALPHA[2])
                T34 = _inv_rigid(T23) @ _inv_rigid(T12) @ T14
                t4 = math.atan2(T34[1, 0], T34[0, 0])
                q = np.array([t1, t2, t3, t4, t5, t6])
                yield _branch_bits(i1, i5, i3), q


_TWO_PI = 2.0 * math.pi


def _wrap_pi(x):
    """Wrap to (-pi, pi]. Vectorised form of [math.remainder(v, 2pi) ...];
    verified bit-for-bit identical to the scalar loop over 2e5 random samples
    plus the +/-pi, +/-3pi, +/-2pi and signed-zero edge cases (np.round is
    round-half-to-even, matching IEEE remainder)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return x - _TWO_PI * np.round(x / _TWO_PI)


# Verify tolerance for an accepted closed-form candidate. The raw closed form is
# accurate to ~1e-15 (measured), so this is only a paranoia check; a candidate
# that fails it is a genuinely degenerate branch and is dropped.
_ANALYTIC_VERIFY_TOL = 1e-8
# Below this the raw candidate is accepted as-is (no Newton polish); the polish
# was pure overhead on the 250 Hz hot path since the closed form already lands
# ~1e-15. Only if a candidate is somehow coarser than this do we polish it.
_ANALYTIC_POLISH_TRIGGER = 1e-10


def _polish_verify(T_target: np.ndarray, q_raw: np.ndarray) -> Optional[np.ndarray]:
    """Wrap a raw closed-form candidate to (-pi, pi], FK-verify, and only run a
    Newton polish if the raw pose error exceeds _ANALYTIC_POLISH_TRIGGER.

    Returns the accepted (wrapped) solution or None if it cannot be made to
    reproduce the target to _ANALYTIC_VERIFY_TOL. Polish never changes branch."""
    q = _wrap_pi(q_raw)
    err = float(np.linalg.norm(_pose_error(fk(q), T_target)))
    if err > _ANALYTIC_POLISH_TRIGGER:
        qp = ik_numeric(T_target, q, max_iter=8, tol=1e-12)
        # ik_numeric may now legitimately return None (tight tol); keep the raw
        # candidate in that case and let the final verify decide.
        if qp is not None and np.max(np.abs(_wrap_pi(qp - q))) <= 0.05:
            q2 = _wrap_pi(qp)
            err2 = float(np.linalg.norm(_pose_error(fk(q2), T_target)))
            if err2 < err:
                q, err = q2, err2
    # `err` is already the pose error of the CURRENT q (the polish branch keeps
    # it in step), so the final verify needs no second fk/_pose_error. The raw
    # closed form lands ~1e-15, i.e. the polish branch is essentially never
    # taken, so this removed a full duplicate FK per candidate (up to 8 per IK).
    if err > _ANALYTIC_VERIFY_TOL:
        return None
    return q


def _dedup_by_branch(tagged):
    """Deduplicate (same physical pose from different nominal branches, e.g. at
    a singularity) keeping the lowest branch id; then order by branch id."""
    unique = []
    if not tagged:
        return unique
    # One vectorised (n,6) compare against all accepted solutions instead of a
    # Python loop of per-pair _wrap_pi calls (this was ~25% of an analytic IK).
    seen = np.empty((len(tagged), 6))
    n = 0
    for bid, q in sorted(tagged, key=lambda t: t[0]):
        if n and np.min(np.max(np.abs(_wrap_pi(seen[:n] - q)), axis=1)) < 1e-6:
            continue
        seen[n] = q
        n += 1
        unique.append((bid, q))
    return unique


def _ik_analytic_tagged(T_target: np.ndarray, prefer_branch: Optional[int] = None):
    """All closed-form solutions as (branch_id, q), wrapped to (-pi, pi] and
    FK-verified. Deterministically ordered by branch id (theta1, theta5, theta3
    selectors). Shared core of ik_analytic() and branch_id().

    `prefer_branch` is a pure COST optimisation for a branch-locked caller: the
    closed form tags each candidate with its branch before any verification, so
    when the caller is going to discard every off-branch solution anyway there
    is no reason to FK-verify all 8 (the verify is ~40% of this function).  If
    the preferred branch yields at least one verified solution, only those are
    returned; otherwise the full enumeration is verified and returned exactly as
    if `prefer_branch` had not been given.  The SOLUTION SET for the preferred
    branch is identical either way -- only solutions the caller was going to
    drop are skipped, and the not-found case falls back, so a caller can never
    see fewer options than it would have acted on."""
    T_target = np.asarray(T_target, dtype=float)
    raw = list(_ik_analytic_raw(T_target))

    if prefer_branch is not None:
        pref = []
        for bid, q in raw:
            if bid != prefer_branch:
                continue
            qv = _polish_verify(T_target, q)
            if qv is not None:
                pref.append((bid, qv))
        if pref:
            return _dedup_by_branch(pref)
        # Preferred branch unavailable -> fall through to the full enumeration.

    tagged = []
    for bid, q in raw:
        qv = _polish_verify(T_target, q)
        if qv is None:
            continue
        tagged.append((bid, qv))
    return _dedup_by_branch(tagged)


def ik_analytic(T_target: np.ndarray) -> List[np.ndarray]:
    """All closed-form solutions (up to 8) as wrapped-to-(-pi,pi] 6-vectors.

    The raw closed form already reproduces the pose to ~1e-15, so candidates are
    accepted directly after an FK verify (a Newton polish is invoked only in the
    rare case a candidate lands coarser than 1e-10). Deterministically ordered
    by branch id (theta1 sel, theta5 sel, theta3 sel)."""
    return [q for _, q in _ik_analytic_tagged(T_target)]


def wrapped_nearest(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Add 2*pi*k per joint to bring `q` to the *literal* (non-circular) value
    nearest `ref`, undoing the (-pi, pi] wrap of ik_analytic so branch locking
    can compare in true joint space.

    The elbow (index 2) is NOT unwrapped: its physical range is +/-pi, so an
    unwrap there would leave the feasible set (and hide a real branch difference)."""
    q = np.asarray(q, dtype=float).reshape(6)
    ref = np.asarray(ref, dtype=float).reshape(6)
    out = q + _TWO_PI * np.round((ref - q) / _TWO_PI)
    out[2] = q[2]  # elbow: never unwrap
    return out


def branch_id(q: np.ndarray) -> int:
    """Closed-form branch label of a configuration (0..7).

    NOT a sign heuristic: q's pose is inverted analytically and q is matched
    (circularly, per joint) to the nearest returned closed-form solution; that
    solution's branch selector (theta1 sel, theta5 sel, theta3 sel) is returned.
    Deterministic for a given q."""
    q = np.asarray(q, dtype=float).reshape(6)
    tagged = _ik_analytic_tagged(fk(q))
    if not tagged:
        return -1
    best_bid, best_d = -1, float("inf")
    for bid, sol in tagged:
        d = float(np.max(np.abs(_wrap_pi(sol - q))))
        if d < best_d:
            best_d, best_bid = d, bid
    return best_bid


# --------------------------------------------------------------------------- #
# Optional DH loader (pyyaml optional; falls back to nominal built-ins)         #
# --------------------------------------------------------------------------- #
def load_dh(path: Optional[str] = None) -> dict:
    """Load DH/limit params from config/ur7e_dh.yaml (pyyaml optional).

    INFORMATIONAL ONLY. fk / ik / jacobian use the module-level constants
    (D, A, ALPHA, JOINT_LIMITS, ...), NOT this dict; it exists for tooling,
    inspection and real-robot re-calibration checks. Wiring it into the
    kinematics would require re-deriving the module constants at import time and
    is intentionally not done.

    The return shape is normalised to a single FLAT schema regardless of whether
    the yaml nests the DH block under a ``dh:`` key or lists d/a/alpha at the top
    level, and regardless of whether pyyaml is present:

        {d, a, alpha, joint_limits, velocity_limits, char_length, branch_weights}
    """
    defaults = {
        "d": D.tolist(),
        "a": A.tolist(),
        "alpha": ALPHA.tolist(),
        "joint_limits": JOINT_LIMITS.tolist(),
        "velocity_limits": VELOCITY_LIMITS.tolist(),
        "char_length": CHAR_LENGTH,
        "branch_weights": np.diag(W_BRANCH).tolist(),
    }
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "ur7e_dh.yaml"
        )
    try:
        import yaml  # optional
    except Exception:
        return defaults
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return defaults
    if not isinstance(data, dict):
        return defaults

    # Normalise: flatten a nested {'dh': {d, a, alpha}} block and copy any
    # top-level overrides, falling back to the nominal defaults for anything the
    # file omits. Never return the raw (possibly nested) yaml.
    out = dict(defaults)
    dh_block = data.get("dh") if isinstance(data.get("dh"), dict) else {}
    for k in ("d", "a", "alpha"):
        if k in dh_block:
            out[k] = dh_block[k]
        elif k in data:
            out[k] = data[k]
    for k in ("joint_limits", "velocity_limits", "char_length", "branch_weights"):
        if k in data:
            out[k] = data[k]
    return out
