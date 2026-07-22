"""Anchor / delta / governor controller for GELLO -> UR7e end-effector teleop.

Pure math (numpy only, NO rclpy). All kinematics come from
``ur_gello_bringup.ur_kin``; this module only implements the *policy* layer:

    engage  -> latch an anchor (leader pose <-> robot command pose) once,
    step    -> map the *world/left* leader delta onto the robot TCP, run an
               SE(3) reference governor with an anisotropic manipulability gain
               and an analytic (continuous) line-search, invert with a
               branch-locked analytic IK, and gate the result through a stack of
               safety acceptance tests (branch, joint limits, step, keepout,
               excursion).  Any failure -> HOLD (freeze T_cmd, return the last
               commanded joints, publish a reject_reason).

Coordinate / multiplication conventions (the whole point of this module -- get
them wrong and (b)/(c)/(d) in the test-suite fail):

    T_g   = fk(q_lead_f) @ T_tool_L        # leader EEF (TCP), UR base frame
    T_r   = fk(q_r)      @ T_tool_R        # robot TCP,        UR base frame
      (same fk on both sides so kinematic error cancels in the anchor-delta)

    R_delta = R_align @ (R_g @ R_g_anchor^T) @ R_align^T      # WORLD/left frame
    R_des   = R_delta @ R_r_anchor
    p_des   = p_r_anchor + pos_scale * R_align @ (p_g - p_g_anchor)

Note ``R_g @ R_g_anchor^T`` (world/left), *not* ``R_g_anchor^T @ R_g`` (body);
and rot_scale is *not* a parameter -- any value != 1.0 is rejected.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ur_gello_bringup.ur_kin import (
    fk,
    sigma_min,
    ik_numeric,
    ik_analytic,
    branch_id,
    within_joint_limits,
    keepout_ok,
    se3_log,
    se3_exp,
    wrapped_nearest,
    W_BRANCH,
    CHAR_LENGTH,
)

# Branch-tagged analytic solutions in a single call (cheaper than branch_id per
# solution).  Private helper of ur_kin; fall back to branch_id if it ever moves.
try:  # pragma: no cover - import guard
    from ur_gello_bringup.ur_kin import _ik_analytic_tagged as _tagged_solver
except Exception:  # pragma: no cover
    _tagged_solver = None


REJECT_NO_IK = "NO_IK"
REJECT_BRANCH_JUMP = "BRANCH_JUMP"
REJECT_JOINT_LIMIT = "JOINT_LIMIT"
REJECT_GEOM_KEEPOUT = "GEOM_KEEPOUT"
REJECT_EXCURSION = "EXCURSION"
REJECT_STEP_FLOOR = "STEP_FLOOR"  # line-search collapsed below s_floor
REJECT_BAD_INPUT = "BAD_INPUT"    # non-finite (inf/nan) leader joints or step budget
REJECT_ESTOP = "ESTOP"            # non-positive step budget -> hard stop (freeze)

_DEFAULTS = {
    "pos_scale": 1.0,
    "r_align_rpy": (0.0, 0.0, 0.0),
    "tool_l_xyz_rpy": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "tool_r_xyz_rpy": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "v_max": 0.08,
    "w_max": 0.5,
    "sigma_warn": 0.10,
    "sigma_stop": 0.03,
    "gamma_min": 0.05,
    "char_length": CHAR_LENGTH,
    "branch_tol": 0.25,
    "branch_weights": np.diag(W_BRANCH).astype(float).tolist(),
    "limit_margin_rad": 0.05,
    "s_floor": 0.02,
    "lag_max_pose": (0.05, 0.3),
    "max_excursion_m": 0.5,
    "dt": 0.008,
    "keepout": {},
    "ik_backend": "analytic",
}

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def _rpy_to_mat(rpy) -> np.ndarray:
    """Fixed-axis roll-pitch-yaw (X,Y,Z) -> R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def _xyz_rpy_to_T(vec6) -> np.ndarray:
    v = np.asarray(vec6, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = _rpy_to_mat(v[3:])
    T[:3, 3] = v[:3]
    return T


def _inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def _wrap(x) -> np.ndarray:
    return np.array([math.remainder(float(v), _TWO_PI) for v in np.atleast_1d(x)])


# --------------------------------------------------------------------------- #
# Controller                                                                   #
# --------------------------------------------------------------------------- #
class EefDeltaController:
    """Anchor + world-frame delta + SE(3) reference governor (see module doc)."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = dict(cfg or {})

        # rot_scale is forbidden: the rotation delta is applied at unit gain.
        if "rot_scale" in cfg and float(cfg["rot_scale"]) != 1.0:
            raise ValueError("rot_scale is not a supported parameter (must be 1.0)")

        g = dict(_DEFAULTS)
        g.update(cfg)
        self.cfg = g

        self.pos_scale = float(g["pos_scale"])
        self.R_align = _rpy_to_mat(g["r_align_rpy"])
        self.T_tool_L = _xyz_rpy_to_T(g["tool_l_xyz_rpy"])
        self.T_tool_R = _xyz_rpy_to_T(g["tool_r_xyz_rpy"])
        self.T_tool_R_inv = _inv_se3(self.T_tool_R)

        self.v_max = float(g["v_max"])
        self.w_max = float(g["w_max"])
        self.sigma_warn = float(g["sigma_warn"])
        self.sigma_stop = float(g["sigma_stop"])
        self.gamma_min = float(g["gamma_min"])
        self.char_length = float(g["char_length"])
        self.branch_tol = float(g["branch_tol"])
        self.branch_w = np.asarray(g["branch_weights"], dtype=float).reshape(6)
        self.limit_margin = float(g["limit_margin_rad"])
        self.s_floor = float(g["s_floor"])
        lag = g["lag_max_pose"]
        self.lag_pos = float(lag[0])
        self.lag_rot = float(lag[1])
        self.max_excursion = float(g["max_excursion_m"])
        self.dt = float(g["dt"])
        self.keepout = dict(g["keepout"] or {})
        self.backend = str(g["ik_backend"])

        # Anchor / running state. q_ik_prev / q_cmd are None until the first
        # engage(): a controller that has never engaged must NOT emit a joint
        # vector (returning zeros(6) is a foot-gun -- a consumer that streams
        # without checking state would command a zero-pose lurch).
        self.state = "DISENGAGED"
        self.q_ik_prev: Optional[np.ndarray] = None
        self.q_cmd: Optional[np.ndarray] = None
        self.T_cmd = np.eye(4)
        self.T_r_anchor = np.eye(4)
        self.R_r_anchor = np.eye(3)
        self.p_r_anchor = np.zeros(3)
        self.R_g_anchor = np.eye(3)
        self.p_g_anchor = np.zeros(3)
        self.branch0 = -1

    # ------------------------------------------------------------------ #
    # Engage / reclutch / disengage                                      #
    # ------------------------------------------------------------------ #
    def _set_anchor(self, q_anchor, q_lead_anchor) -> None:
        q_anchor = np.asarray(q_anchor, dtype=float).reshape(6).copy()
        q_lead = np.asarray(q_lead_anchor, dtype=float).reshape(6).copy()

        self.T_r_anchor = fk(q_anchor) @ self.T_tool_R
        self.R_r_anchor = self.T_r_anchor[:3, :3].copy()
        self.p_r_anchor = self.T_r_anchor[:3, 3].copy()

        T_g = fk(q_lead) @ self.T_tool_L
        self.R_g_anchor = T_g[:3, :3].copy()
        self.p_g_anchor = T_g[:3, 3].copy()

        self.T_cmd = self.T_r_anchor.copy()
        self.q_ik_prev = q_anchor.copy()
        self.q_cmd = q_anchor.copy()
        self.branch0 = branch_id(q_anchor) if self.backend == "analytic" else -1

    def engage(self, q_anchor, q_lead_f_anchor) -> dict:
        """Latch the anchor at engage-time (t0).  Returns an anchor summary."""
        self._set_anchor(q_anchor, q_lead_f_anchor)
        self.state = "ENGAGED"
        return {
            "state": self.state,
            "q_anchor": self.q_ik_prev.copy(),
            "branch0": self.branch0,
            "T_r_anchor": self.T_r_anchor.copy(),
            "T_g_anchor": (fk(np.asarray(q_lead_f_anchor, float).reshape(6))
                           @ self.T_tool_L),
            "sigma_min": sigma_min(self.q_ik_prev, self.char_length),
        }

    def reclutch(self, q_lead_now, q_cmd_now) -> dict:
        """Re-anchor at the current leader/robot pose (fresh, zero delta)."""
        self._set_anchor(q_cmd_now, q_lead_now)
        self.state = "ENGAGED"
        return {
            "state": self.state,
            "q_anchor": self.q_ik_prev.copy(),
            "branch0": self.branch0,
        }

    def disengage(self) -> None:
        self.state = "DISENGAGED"

    # ------------------------------------------------------------------ #
    # Branch-locked IK                                                    #
    # ------------------------------------------------------------------ #
    def _reanchor(self, sol: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Undo ik_analytic's (-pi, pi] wrap: add 2*pi*k per joint to bring `sol`
        to the literal value nearest `seed` (elbow idx 2 left alone). Shared
        implementation lives in ur_kin.wrapped_nearest."""
        return wrapped_nearest(sol, seed)

    def _wnorm(self, dq: np.ndarray) -> float:
        return float(math.sqrt(np.sum(self.branch_w * (dq * dq))))

    def _tagged(self, T_flange):
        if _tagged_solver is not None:
            return _tagged_solver(T_flange)
        return [(branch_id(s), s) for s in ik_analytic(T_flange)]

    def _ik_branchlock(self, T_tcp: np.ndarray):
        """Invert a *TCP* target with branch locking.

        Returns (q or None, n_solutions).  Order (grounding): analytic enumerate
        -> branch0 filter (suspended at an elbow/wrist merge point) -> re-anchor
        to q_ik_prev -> hard-limit filter -> argmin ||.||_W.  Acceptance vs
        branch_tol is done by the caller."""
        T_flange = T_tcp @ self.T_tool_R_inv
        seed = self.q_ik_prev

        if self.backend != "analytic":
            q = ik_numeric(T_flange, seed)
            if q is None:
                return None, 0
            return q, 1

        tagged = self._tagged(T_flange)
        if not tagged:
            return None, 0
        n = len(tagged)

        # Merge point: elbow (q3, idx2) ~ 0 or wrist_2 (q5, idx4) ~ 0.  There the
        # branch labels collapse, so suspend the branch filter and rely on
        # branch_tol alone.
        merge = (abs(math.remainder(float(seed[2]), _TWO_PI)) < 0.05
                 or abs(math.remainder(float(seed[4]), _TWO_PI)) < 0.05)

        # Branch filter BEFORE unwrap (order matters).
        pool = tagged
        if not merge and self.branch0 >= 0:
            on_branch = [(bid, q) for (bid, q) in tagged if bid == self.branch0]
            if on_branch:
                pool = on_branch

        # Re-anchor each candidate to the seed.
        reanch = [self._reanchor(q, seed) for (_, q) in pool]

        # Hard-limit filter (margin 0). The 0.05 margin is applied later as an
        # acceptance test so that a marginal violation reports JOINT_LIMIT rather
        # than silently vanishing here.
        limited = [q for q in reanch if within_joint_limits(q, margin=0.0)]
        if limited:
            reanch = limited

        best = min(reanch, key=lambda q: self._wnorm(q - seed))
        return best, n

    # ------------------------------------------------------------------ #
    # Step                                                               #
    # ------------------------------------------------------------------ #
    def _hold(self, info: dict, reason: str):
        info["state"] = "HOLD"
        info["reject_reason"] = reason
        info["branch_id"] = self.branch0
        info["excursion"] = float(np.linalg.norm(self.T_cmd[:3, 3] - self.p_r_anchor))
        return self.q_ik_prev.copy(), info

    def step(self, q_lead_f, step_eff: float):
        """Advance one control tick.

        Returns (q_cmd, info).  On any acceptance failure the arm HOLDs:
        q_cmd == previous joints, T_cmd frozen, info['reject_reason'] set.

        Before the first engage() q_cmd is None: a controller that has never
        engaged returns (None, info) rather than a zero-pose vector.  Non-finite
        inputs (A1) and a non-positive step budget (A2) are trapped up front and
        routed to HOLD before any trig/FK runs."""
        # ---- DISENGAGED: never emit an unearned joint vector ---- #
        if self.state == "DISENGAGED":
            info = {
                "state": "DISENGAGED",
                "reject_reason": None,
                "branch_id": self.branch0,
                "excursion": float(np.linalg.norm(self.T_cmd[:3, 3] - self.p_r_anchor)),
                "T_des": self.T_cmd.copy(),
                "T_cmd": self.T_cmd.copy(),
            }
            return (self.q_cmd.copy() if self.q_cmd is not None else None), info

        q_prev = self.q_ik_prev  # engaged -> guaranteed not None

        info = {
            "state": self.state,
            "reject_reason": None,
            "sigma_min": sigma_min(q_prev, self.char_length),
            "gamma": 1.0,
            "ls_scale": 0.0,
            "ik_residual": 0.0,
            "lag_pos": 0.0,
            "lag_rot": 0.0,
            "excursion": float(np.linalg.norm(self.T_cmd[:3, 3] - self.p_r_anchor)),
            "branch_id": self.branch0,
            "n_ik_solutions": 0,
            "T_des": self.T_cmd.copy(),
            "T_cmd": self.T_cmd.copy(),
        }

        # ---- A1: sanitize non-finite inputs BEFORE any trig/FK (math.cos(inf)
        #      raises ValueError; nan would otherwise slip through the gates). ----
        q_arr = np.asarray(q_lead_f, dtype=float).reshape(6)
        step_eff = float(step_eff)
        if not bool(np.all(np.isfinite(q_arr))) or not math.isfinite(step_eff):
            return self._hold(info, REJECT_BAD_INPUT)
        q_lead_f = q_arr

        # ---- A2: a non-positive step budget is a HARD STOP, not a "freeze that
        #      disables the limiter". Freeze T_cmd, hold at the last joints. A
        #      non-positive budget must never reach (and so bypass) the step-cap
        #      acceptance gate below. ----
        if step_eff <= 0.0:
            return self._hold(info, REJECT_ESTOP)

        # ---- world/left delta -> desired TCP pose ---- #
        T_g = fk(q_lead_f) @ self.T_tool_L
        R_g = T_g[:3, :3]
        p_g = T_g[:3, 3]
        R_delta = self.R_align @ (R_g @ self.R_g_anchor.T) @ self.R_align.T
        R_des = R_delta @ self.R_r_anchor
        p_des = self.p_r_anchor + self.pos_scale * (self.R_align @ (p_g - self.p_g_anchor))
        T_des = np.eye(4)
        T_des[:3, :3] = R_des
        T_des[:3, 3] = p_des

        # ---- anti-windup: keep T_des within lag_max of the (possibly frozen)
        #      T_cmd so a runaway leader cannot build up unbounded error. ----
        xi_lag = se3_log(_inv_se3(self.T_cmd) @ T_des)
        v_l, w_l = xi_lag[:3].copy(), xi_lag[3:].copy()
        nvl, nwl = float(np.linalg.norm(v_l)), float(np.linalg.norm(w_l))
        if nvl > self.lag_pos:
            v_l *= self.lag_pos / nvl
        if nwl > self.lag_rot:
            w_l *= self.lag_rot / nwl
        T_des = self.T_cmd @ se3_exp(np.concatenate([v_l, w_l]))
        info["T_des"] = T_des.copy()
        info["lag_pos"] = float(np.linalg.norm(v_l))
        info["lag_rot"] = float(np.linalg.norm(w_l))

        # ---- governor increment ---- #
        xi_raw = se3_log(_inv_se3(self.T_cmd) @ T_des)
        v, w = xi_raw[:3], xi_raw[3:]
        nv, nw = float(np.linalg.norm(v)), float(np.linalg.norm(w))

        # Exactly-zero delta -> hold pose bit-for-bit (zero-jump guarantee).
        if nv < 1e-11 and nw < 1e-11:
            info["state"] = "ENGAGED"
            info["ls_scale"] = 1.0
            return q_prev.copy(), info

        sigma_prev = info["sigma_min"]
        denom = self.sigma_warn - self.sigma_stop
        gamma_thr = 1.0 if denom <= 0 else (sigma_prev - self.sigma_stop) / denom
        gamma_thr = min(1.0, max(self.gamma_min, gamma_thr))

        def rate_scale(gamma):
            sc = 1.0
            if nv > 1e-12:
                sc = min(sc, gamma * self.v_max * self.dt / nv)
            if nw > 1e-12:
                sc = min(sc, gamma * self.w_max * self.dt / nw)
            return sc

        # ---- asymmetric gamma: a move that *increases* sigma_min (escapes the
        #      singularity) is passed unthrottled (gamma=1), preventing a
        #      permanent lock-up at sigma_min < sigma_stop. ----
        gamma_eff = gamma_thr
        if gamma_thr < 1.0:
            probe_target = self.T_cmd @ se3_exp(xi_raw * rate_scale(1.0))
            q_probe, _ = self._ik_branchlock(probe_target)
            if q_probe is not None and sigma_min(q_probe, self.char_length) > sigma_prev + 1e-9:
                gamma_eff = 1.0
        info["gamma"] = gamma_eff

        xi_lim = xi_raw * rate_scale(gamma_eff)

        # ---- analytic (continuous) line search for the step scale s ---- #
        q_try0, n_sol = self._ik_branchlock(self.T_cmd @ se3_exp(xi_lim))
        info["n_ik_solutions"] = n_sol
        if q_try0 is None:
            return self._hold(info, REJECT_NO_IK)

        # step_eff > 0 is guaranteed here (A2 hard-stops non-positive budgets).
        max_dq0 = float(np.max(np.abs(q_try0 - q_prev)))
        if max_dq0 > 1e-12:
            s = min(1.0, 0.9 * step_eff / max_dq0)
        else:
            s = 1.0

        q_try = None
        for _ in range(3):
            cand, n_sol = self._ik_branchlock(self.T_cmd @ se3_exp(s * xi_lim))
            if cand is None:
                s *= 0.5
                if s < self.s_floor:
                    break
                continue
            max_dq = float(np.max(np.abs(cand - q_prev)))
            q_try = cand
            if max_dq <= step_eff * (1.0 + 1e-9) or max_dq < 1e-12:
                break
            # Overshoot from IK nonlinearity: shrink proportionally and retry.
            s = 0.9 * s * step_eff / max_dq
            if s < self.s_floor:
                q_try = None
                break

        if q_try is None or s < self.s_floor:
            reason = REJECT_NO_IK if q_try is None and n_sol == 0 else REJECT_STEP_FLOOR
            return self._hold(info, reason)
        info["ls_scale"] = s
        info["n_ik_solutions"] = n_sol

        # ---- acceptance stack ---- #
        dq = q_try - q_prev
        if self._wnorm(dq) > self.branch_tol:
            return self._hold(info, REJECT_BRANCH_JUMP)
        if not within_joint_limits(q_try, margin=self.limit_margin):
            return self._hold(info, REJECT_JOINT_LIMIT)
        if float(np.max(np.abs(dq))) > step_eff * (1.0 + 1e-6):
            return self._hold(info, REJECT_STEP_FLOOR)
        if not keepout_ok(q_try, self.keepout):
            return self._hold(info, REJECT_GEOM_KEEPOUT)

        T_cmd_new = fk(q_try) @ self.T_tool_R
        excursion = float(np.linalg.norm(T_cmd_new[:3, 3] - self.p_r_anchor))
        if excursion > self.max_excursion:
            return self._hold(info, REJECT_EXCURSION)

        # ---- commit (keep invariant  T_cmd == fk(q_ik_prev) @ T_tool_R) ---- #
        self.q_ik_prev = q_try.copy()
        self.q_cmd = q_try.copy()
        self.T_cmd = T_cmd_new
        info["state"] = "ENGAGED"
        info["excursion"] = excursion
        info["branch_id"] = self.branch0
        info["ik_residual"] = float(
            np.linalg.norm(se3_log(_inv_se3(T_cmd_new) @ T_des))
        )
        info["T_cmd"] = T_cmd_new.copy()
        return q_try.copy(), info
