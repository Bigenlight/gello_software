"""Task-space delta -> safe joint command, for the RL policy path.

This is the UR-side replacement for what Franka's impedance controller absorbs
for free. It is deliberately the same shape as the *second half* of
ur_gello_bringup.eef_delta.EefDeltaController.step() (governor -> IK ->
acceptance gates -> HOLD), minus the leader/anchor mapping: the RL policy
already gives us a per-step task-space increment, so there is nothing to
anchor.

    xi (6,) = [dpos (m), drot rotvec (rad)]      # already scaled by ACTION_SCALE
                                                 # WORLD/base frame (FrankaEnv
                                                 # convention; RelativeFrame
                                                 # gives the policy EEF frame)
    T_des   = (p_cmd + dpos,  exp(drot) @ R_cmd)
    q_cmd   = branch-continuous IK(T_des), gated; on any doubt -> HOLD

TODO(together): replace the simplified internals below with the real
EefDeltaController machinery (sigma_min throttle with asymmetric escape,
analytic branch-lock, keepout, line search). Cleanest path is probably
refactoring eef_delta so its post-anchor pipeline is callable with a T_des
directly — then this file shrinks to a thin adapter and both the policy path
and GELLO intervention share one gate stack.

UNTESTED SKELETON — do not run against real hardware until reviewed.
"""

from typing import Optional, Tuple

import numpy as np

try:
    from ur_gello_bringup.ur_kin import (  # noqa: F401
        fk,
        ik_numeric,
        so3_exp,
        within_joint_limits,
    )

    _KIN_AVAILABLE = True
except ImportError:
    _KIN_AVAILABLE = False


def _inv_se3(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


class PolicyDeltaController:
    def __init__(self, governor_cfg: dict, hz: float):
        if not _KIN_AVAILABLE:
            raise RuntimeError(
                "ur_gello_bringup not importable — source the ros2_ur_ws overlay"
            )
        self.v_max = float(governor_cfg["v_max"])
        self.w_max = float(governor_cfg["w_max"])
        self.dq_step_max = float(governor_cfg["dq_step_max"])
        self.dt = 1.0 / hz

        self.T_cmd: Optional[np.ndarray] = None
        self.q_cmd: Optional[np.ndarray] = None

    def reset(self, q_now: np.ndarray):
        """Latch the command state to the robot's actual pose (call on env.reset)."""
        q_now = np.asarray(q_now, dtype=float).reshape(6)
        self.q_cmd = q_now.copy()
        self.T_cmd = fk(q_now)

    def step(self, xi: np.ndarray) -> Tuple[np.ndarray, dict]:
        """One control tick. Returns (q_cmd, info); HOLDs on any doubt."""
        assert self.T_cmd is not None, "call reset() before step()"
        info = {"held": False, "reject_reason": None}

        xi = np.asarray(xi, dtype=float).reshape(6)
        if not np.all(np.isfinite(xi)):
            return self._hold(info, "BAD_INPUT")

        # ---- governor: per-tick rate cap in task space ---- #
        v, w = xi[:3].copy(), xi[3:].copy()
        nv, nw = float(np.linalg.norm(v)), float(np.linalg.norm(w))
        scale = 1.0
        if nv > 1e-12:
            scale = min(scale, self.v_max * self.dt / nv)
        if nw > 1e-12:
            scale = min(scale, self.w_max * self.dt / nw)
        scale = min(scale, 1.0)
        v, w = v * scale, w * scale

        # ---- WORLD-frame increment, FrankaEnv semantics ---- #
        # position shifts along BASE axes; rotation is a base-frame rotvec
        # applied about the TCP point (orientation-only left multiply).
        # NOT T_cmd @ exp(xi): right-multiplication would mean tool-frame
        # deltas, and the hil-serl stack (RelativeFrame wrapper) already
        # assumes the raw env takes base-frame actions.
        T_des = self.T_cmd.copy()
        T_des[:3, 3] = self.T_cmd[:3, 3] + v
        T_des[:3, :3] = so3_exp(w) @ self.T_cmd[:3, :3]

        # ---- IK, seeded at previous command (branch continuity via seed) ---- #
        q_sol = ik_numeric(T_des, self.q_cmd)
        if q_sol is None:
            return self._hold(info, "NO_IK")
        if not within_joint_limits(q_sol, margin=0.0):
            return self._hold(info, "JOINT_LIMIT")

        # ---- per-tick joint step acceptance gate ---- #
        dq = np.abs(q_sol - self.q_cmd)
        if float(np.max(dq)) > self.dq_step_max:
            return self._hold(info, "STEP_LIMIT")

        self.T_cmd = T_des
        self.q_cmd = q_sol
        return q_sol.copy(), info

    def _hold(self, info: dict, reason: str):
        info["held"] = True
        info["reject_reason"] = reason
        return self.q_cmd.copy(), info

    # convenience for wrappers that need the commanded TCP pose
    def tcp_cmd(self) -> np.ndarray:
        return self.T_cmd.copy()
