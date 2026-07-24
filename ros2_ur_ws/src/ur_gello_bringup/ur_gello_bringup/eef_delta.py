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
    rpy_to_mat,
    xyz_rpy_to_T,
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
# BRANCH_JUMP is now reserved for a GENUINE branch change: the accepted-by-IK
# candidate carries a different closed-form branch id than the anchor.  The
# large-but-same-branch case (the ||dq||_W > branch_tol size test) reports
# JOINT_JUMP -- both come from the same gate, but an operator reading
# reject_reason on the real robot must not be told "branch jump" when the truth
# is "that step was too big".
REJECT_BRANCH_JUMP = "BRANCH_JUMP"
REJECT_JOINT_JUMP = "JOINT_JUMP"   # ||dq||_W > branch_tol on the anchor's OWN branch
REJECT_JOINT_LIMIT = "JOINT_LIMIT"
REJECT_GEOM_KEEPOUT = "GEOM_KEEPOUT"
REJECT_EXCURSION = "EXCURSION"
REJECT_STEP_FLOOR = "STEP_FLOOR"  # line-search collapsed below s_floor
REJECT_STEP_CAP = "STEP_CAP"      # accepted s, but max|dq| still over step_eff
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

# Budget-fitting iterations of the line search AFTER the solvable-scale probe.
_LINE_SEARCH_ITERS = 3

# --- sigma_min (SVD) decimation, see _sigma_min_cached ---------------------- #
# Lipschitz bound of sigma_min w.r.t. max-norm joint travel. MEASURED max over
# 30k random configurations x 4 step sizes: 3.502 per rad. 12.0 is a 3.4x
# safety margin on that measurement.
_SIGMA_LIPSCHITZ = 12.0
# Hard cap on how many ticks a cached sigma_min may be reused, independent of
# the travel bound (docs/ros2/GELLO_UR7E_EEF_TELEOP_PLAN.md 6.6 asks for ~10).
_SIGMA_MAX_STALE_TICKS = 10

# Re-test the asymmetric-gamma escape probe every N ticks while it keeps
# failing (see step()). 8 ticks = 32 ms at 250 Hz.
_ESCAPE_PROBE_PERIOD = 8


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
# Single source of truth for the rpy convention lives in ur_kin; these aliases
# keep the historical private names working for anything importing them.
_rpy_to_mat = rpy_to_mat
_xyz_rpy_to_T = xyz_rpy_to_T


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
        # Teach the keepout check about the end effector.  link_origins() stops
        # at the flange, so without this the 0.174 m Robotiq 2F-85 -- the part
        # that actually reaches the table -- is invisible to floor_z & friends.
        # Setting it on the dict (a private copy; the caller's cfg is never
        # mutated) means every keepout_ok(q, ctrl.keepout) call site is
        # tool-aware, not just this module's.  An operator-supplied tool wins.
        self.keepout.setdefault("tool", self.T_tool_R.copy())
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
        self._last_ik_branch = -1
        self._escape_skip = 0
        self._sigma_cache: Optional[float] = None
        self._sigma_stale_ticks = 0
        self._sigma_stale_travel = 0.0

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
        # A fresh anchor invalidates any decimated sigma_min / escape verdict.
        self._escape_skip = 0
        self._sigma_cache = None
        self._sigma_stale_ticks = 0
        self._sigma_stale_travel = 0.0

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

    def _sigma_min_cached(self, q: np.ndarray) -> float:
        """sigma_min(q), with the 6x6 SVD skipped while the cached value is
        PROVABLY still above sigma_warn.

        Why this is exact for the control decision, not merely "close enough":
        the only use of sigma_min on the accepted path is gamma_thr, and
        gamma_thr saturates at 1.0 for every sigma >= sigma_warn.  The cache is
        reused only when

            c - LIP * travel > sigma_warn

        where `travel` is the accumulated max-norm joint motion since the SVD
        was taken (a sum of per-tick max-norms, itself an upper bound on the
        true max-norm displacement by the triangle inequality).  That inequality
        certifies BOTH that the cached c and the true sigma exceed sigma_warn,
        so gamma_thr is 1.0 either way and the emitted command is bit-identical
        to the undecimated controller.  The escape probe (:gamma_thr < 1.0) is
        likewise not entered in either case, so it cannot be skipped by
        staleness.  Near the singularity -- where the value actually steers the
        throttle -- the guard fails and the SVD runs every tick.

        Only info['sigma_min'] is affected, and only as a diagnostic; the
        companion field info['sigma_stale_ticks'] reports the staleness."""
        c = self._sigma_cache
        if (
            c is not None
            and self._sigma_stale_ticks < _SIGMA_MAX_STALE_TICKS
            and c - _SIGMA_LIPSCHITZ * self._sigma_stale_travel > self.sigma_warn
        ):
            return c
        s = sigma_min(q, self.char_length)
        self._sigma_cache = s
        self._sigma_stale_ticks = 0
        self._sigma_stale_travel = 0.0
        return s

    def _jump_reason(self) -> str:
        """Truthful reason for the ``||dq||_W > branch_tol`` refusal.

        BRANCH_JUMP only if the candidate really is on a different closed-form
        branch than the anchor; otherwise JOINT_JUMP (a same-branch step that is
        simply too large).  With the numeric backend there is no branch label at
        all (branch0 == -1), so it can only ever be JOINT_JUMP.

        Uses the branch tag `_ik_branchlock` already carried out of the closed-
        form enumeration, so this costs nothing: calling branch_id() here would
        re-run a whole 8-branch analytic IK (measured +461 us) on a path that is
        latency-sensitive precisely because it repeats every tick while the
        operator holds against the gate."""
        if self.branch0 < 0 or self._last_ik_branch < 0:
            return REJECT_JOINT_JUMP
        if self._last_ik_branch != self.branch0:
            return REJECT_BRANCH_JUMP
        return REJECT_JOINT_JUMP

    def _tagged(self, T_flange, prefer_branch=None):
        if _tagged_solver is not None:
            return _tagged_solver(T_flange, prefer_branch)
        return [(branch_id(s), s) for s in ik_analytic(T_flange)]

    def _ik_branchlock(self, T_tcp: np.ndarray):
        """Invert a *TCP* target with branch locking.

        Returns (q or None, n_solutions).  Order (grounding): analytic enumerate
        -> branch0 filter (suspended at an elbow/wrist merge point) -> re-anchor
        to q_ik_prev -> hard-limit filter -> argmin ||.||_W.  Acceptance vs
        branch_tol is done by the caller.

        Side channel: ``self._last_ik_branch`` is set to the closed-form branch
        id of the returned candidate (-1 if none / numeric backend), so callers
        can attribute a refusal without paying for a second analytic IK.  The
        (q, n) return shape is deliberately unchanged."""
        self._last_ik_branch = -1
        T_flange = T_tcp @ self.T_tool_R_inv
        seed = self.q_ik_prev

        if self.backend != "analytic":
            q = ik_numeric(T_flange, seed)
            if q is None:
                return None, 0
            return q, 1

        # Merge point: elbow (q3, idx2) ~ 0 or wrist_2 (q5, idx4) ~ 0.  There the
        # branch labels collapse, so suspend the branch filter and rely on
        # branch_tol alone.
        merge = (abs(math.remainder(float(seed[2]), _TWO_PI)) < 0.05
                 or abs(math.remainder(float(seed[4]), _TWO_PI)) < 0.05)

        # Tell the solver which branch we are going to keep, so it can skip
        # FK-verifying the 7 candidates this filter would discard anyway. It
        # falls back to the full enumeration whenever that branch has no
        # solution, so the outcome is unchanged (see _ik_analytic_tagged).
        prefer = None if (merge or self.branch0 < 0) else self.branch0
        tagged = self._tagged(T_flange, prefer)
        if not tagged:
            return None, 0
        n = len(tagged)

        # Branch filter BEFORE unwrap (order matters).
        pool = tagged
        if not merge and self.branch0 >= 0:
            on_branch = [(bid, q) for (bid, q) in tagged if bid == self.branch0]
            if on_branch:
                pool = on_branch

        # Re-anchor each candidate to the seed, keeping its branch tag alongside
        # (wrapped_nearest only adds 2*pi*k, which never changes the branch).
        reanch = [(bid, self._reanchor(q, seed)) for (bid, q) in pool]

        # Hard-limit filter (margin 0). The 0.05 margin is applied later as an
        # acceptance test so that a marginal violation reports JOINT_LIMIT rather
        # than silently vanishing here.
        limited = [t for t in reanch if within_joint_limits(t[1], margin=0.0)]
        if limited:
            reanch = limited

        best_bid, best = min(reanch, key=lambda t: self._wnorm(t[1] - seed))
        self._last_ik_branch = int(best_bid)
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
            "sigma_min": self._sigma_min_cached(q_prev),
            "sigma_stale_ticks": self._sigma_stale_ticks,
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
        # The probe costs a full analytic IK + a second SVD, so it is entered
        # only when gamma_thr < 1.0 -- i.e. only when sigma_min has already
        # fallen into the warn band and the throttle is actually biting.  Above
        # sigma_warn gamma_thr saturates at 1.0 and there is nothing to escape,
        # so the probe never runs on the well-conditioned hot path.
        # While the probe keeps FAILING the verdict is re-tested only every
        # _ESCAPE_PROBE_PERIOD ticks. Skipping it leaves gamma_eff == gamma_thr,
        # the THROTTLED value -- so a skipped probe can only ever make the arm
        # slower, never faster: the optimisation is safety-monotone, and it
        # cannot turn a HOLD into a motion. The only cost is up to
        # (period-1) ticks = 28 ms at 250 Hz of extra throttling after the
        # operator reverses out of the singularity. A probe that SUCCEEDS resets
        # the counter, so an escape in progress is re-checked every tick (and is
        # nearly free, since its solution is reused by the line search below).
        gamma_eff = gamma_thr
        q_escape = None
        n_escape = 0
        if gamma_thr < 1.0:
            if self._escape_skip > 0:
                self._escape_skip -= 1
            else:
                probe_target = self.T_cmd @ se3_exp(xi_raw * rate_scale(1.0))
                q_probe, n_probe = self._ik_branchlock(probe_target)
                if (q_probe is not None
                        and sigma_min(q_probe, self.char_length) > sigma_prev + 1e-9):
                    gamma_eff = 1.0
                    # Escape accepted -> gamma_eff is 1.0, so the line search's
                    # own full-scale probe below asks for exactly this same
                    # target. Carry the solution over instead of re-solving it.
                    q_escape, n_escape = q_probe, n_probe
                    self._escape_skip = 0
                else:
                    self._escape_skip = _ESCAPE_PROBE_PERIOD - 1
        info["gamma"] = gamma_eff

        xi_lim = xi_raw * rate_scale(gamma_eff)

        # ---- analytic (continuous) line search for the step scale s ---- #
        # Probe the FULL rate-limited increment first.  If it has no IK solution
        # that is NOT grounds to give up: near a singularity the reachable set
        # shrinks continuously, so a fraction of the very same increment is
        # routinely solvable (measured: s=1.00 -> None, s=0.50 -> ok,
        # s=0.10 -> ok and inside the step budget).  Returning NO_IK straight
        # off the full-scale probe skipped the halving loop below entirely and
        # dead-stopped the arm permanently (1936/2000 ticks rejected, 0.8 mm
        # delivered of a requested 10 mm).  Halve down to s_floor looking for
        # the largest solvable scale and let the normal budget line search take
        # over from there.  Fail-closed is preserved: nothing solvable down to
        # s_floor still HOLDs with NO_IK, and no acceptance gate is relaxed.
        s_probe = 1.0
        q_try0 = None
        n_sol = 0
        if q_escape is not None:
            # Bit-identical target, already solved by the escape probe above.
            q_try0, n_sol = q_escape, n_escape
        while q_try0 is None:
            q_try0, n_sol = self._ik_branchlock(self.T_cmd @ se3_exp(s_probe * xi_lim))
            if q_try0 is not None:
                break
            if s_probe <= self.s_floor:
                break
            # Clamp the last halving to s_floor EXACTLY.  Plain halving steps
            # 0.03125 -> 0.015625 and so jumps straight over s_floor=0.02 -- and
            # 0.02 was measured to be the largest solvable scale at a real stall
            # point, so the naive sequence declares NO_IK on a feasible target.
            s_probe = max(self.s_floor, 0.5 * s_probe)
        info["n_ik_solutions"] = n_sol
        info["ls_probe_scale"] = s_probe if q_try0 is not None else 0.0
        if q_try0 is None:
            return self._hold(info, REJECT_NO_IK)

        # step_eff > 0 is guaranteed here (A2 hard-stops non-positive budgets).
        # max_dq0 is the joint cost of `s_probe * xi_lim`, so the budget-fitting
        # scale extrapolates from s_probe (identical to the old expression when
        # the full-scale probe succeeded and s_probe == 1.0).
        max_dq0 = float(np.max(np.abs(q_try0 - q_prev)))
        if max_dq0 > 1e-12:
            s = min(s_probe, 0.9 * s_probe * step_eff / max_dq0)
        else:
            s = s_probe

        # `s` is a FRACTION of an increment that the rate limiter and gamma have
        # already shrunk (at gamma_min the full increment is v_max*dt*gamma_min
        # = 32 um), so it is not a physical floor on anything.  Vetoing an
        # otherwise-valid candidate just because that fraction landed under
        # s_floor is what produced the second permanent stall: measured at the
        # dead-stop, the search found a candidate costing 0.00136 rad of the
        # 0.0025 rad budget -- on branch, in limits, keepout clear -- and threw
        # it away because s was 0.0162 < 0.02.  s_floor now bounds how far the
        # search may SHRINK (below); whether a candidate is safe is decided by
        # the acceptance stack, which is unchanged and still runs on every
        # candidate.  Fail-closed is kept for the two cases that really are
        # failures: no IK anywhere (NO_IK) and an increment that still overshoots
        # the joint budget once the search has shrunk to s_floor (STEP_FLOOR).
        q_try = None
        q_try_branch = -1
        n_sol0 = n_sol
        first = True
        for _ in range(_LINE_SEARCH_ITERS):
            if first and s == s_probe:
                # The budget did not bind, so the first line-search target is
                # BIT-IDENTICAL to the probe target already solved above.
                # Re-solving it cost a second full 8-branch analytic IK (~760 us,
                # ~40% of a moving tick) for a guaranteed-identical answer.
                cand, n_sol = q_try0, n_sol0
            else:
                cand, n_sol = self._ik_branchlock(self.T_cmd @ se3_exp(s * xi_lim))
            first = False
            if cand is None:
                s *= 0.5
                if s < self.s_floor:
                    break
                continue
            max_dq = float(np.max(np.abs(cand - q_prev)))
            q_try = cand
            # Pin the branch tag to THIS candidate: a later probe that returns
            # None would otherwise reset the side channel to -1 while q_try still
            # holds this (perfectly good) candidate.
            q_try_branch = self._last_ik_branch
            if max_dq <= step_eff * (1.0 + 1e-9) or max_dq < 1e-12:
                break
            # Overshoot from IK nonlinearity: shrink proportionally and retry.
            s = 0.9 * s * step_eff / max_dq
            if s < self.s_floor:
                q_try = None
                break

        if q_try is None:
            reason = REJECT_NO_IK if n_sol == 0 else REJECT_STEP_FLOOR
            return self._hold(info, reason)
        info["ls_scale"] = s
        info["n_ik_solutions"] = n_sol

        # ---- acceptance stack ---- #
        dq = q_try - q_prev
        if self._wnorm(dq) > self.branch_tol:
            # The gate itself is a weighted joint-STEP-SIZE test, not a branch
            # test -- it fires just as readily on a large step that never left
            # the anchor's branch.  The HOLD is right either way; only the
            # attribution has to be honest, so report which of the two it was
            # (free: the branch tag came out of the IK that produced q_try).
            self._last_ik_branch = q_try_branch
            return self._hold(info, self._jump_reason())
        if not within_joint_limits(q_try, margin=self.limit_margin):
            return self._hold(info, REJECT_JOINT_LIMIT)
        if float(np.max(np.abs(dq))) > step_eff * (1.0 + 1e-6):
            return self._hold(info, REJECT_STEP_CAP)
        if not keepout_ok(q_try, self.keepout):
            return self._hold(info, REJECT_GEOM_KEEPOUT)

        T_cmd_new = fk(q_try) @ self.T_tool_R
        excursion = float(np.linalg.norm(T_cmd_new[:3, 3] - self.p_r_anchor))
        if excursion > self.max_excursion:
            return self._hold(info, REJECT_EXCURSION)

        # ---- commit (keep invariant  T_cmd == fk(q_ik_prev) @ T_tool_R) ---- #
        # Accumulate the joint travel that the decimated sigma_min is now stale
        # by (per-tick max-norms sum to an upper bound on the total).  A HOLD
        # moves nothing, so only accepted steps age the cache.
        self._sigma_stale_ticks += 1
        self._sigma_stale_travel += float(np.max(np.abs(dq)))
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
