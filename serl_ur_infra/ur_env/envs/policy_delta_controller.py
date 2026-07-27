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

from typing import Callable, Optional, Tuple

import numpy as np

try:
    from ur_gello_bringup.ur_kin import (  # noqa: F401
        fk,
        ik_numeric,
        so3_exp,
        so3_log,
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
    def __init__(
        self,
        governor_cfg: dict,
        hz: float,
        clip_pose: Optional[Callable[[np.ndarray], Tuple[np.ndarray, bool]]] = None,
    ):
        """clip_pose: optional workspace-box hook, ``T (4,4) -> (T, clipped)``.

        UR7eEnv passes ``UR7eEnv._clip_command_pose`` here. The env owns the
        box (it is a config/task property); the controller owns WHEN it is
        applied, because only the controller knows the command state that the
        clamp has to be written back into. Left None the controller behaves
        exactly as before — useful for unit tests and for configs with no
        measured box.
        """
        if not _KIN_AVAILABLE:
            raise RuntimeError(
                "ur_gello_bringup not importable — source the ros2_ur_ws overlay"
            )
        self.v_max = float(governor_cfg["v_max"])
        self.w_max = float(governor_cfg["w_max"])
        self.dq_step_max = float(governor_cfg["dq_step_max"])
        self.dt = 1.0 / hz
        self.clip_pose = clip_pose

        self.T_cmd: Optional[np.ndarray] = None
        self.q_cmd: Optional[np.ndarray] = None

    def reset(self, q_now: np.ndarray):
        """Latch the command state to the robot's actual pose (call on env.reset)."""
        q_now = np.asarray(q_now, dtype=float).reshape(6)
        self.q_cmd = q_now.copy()
        self.T_cmd = fk(q_now)

    def _govern(self, v: np.ndarray, w: np.ndarray):
        """Per-tick rate cap in task space. Returns (v, w, scale)."""
        nv, nw = float(np.linalg.norm(v)), float(np.linalg.norm(w))
        scale = 1.0
        if nv > 1e-12:
            scale = min(scale, self.v_max * self.dt / nv)
        if nw > 1e-12:
            scale = min(scale, self.w_max * self.dt / nw)
        scale = min(scale, 1.0)
        return v * scale, w * scale, scale

    def _integrate(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """T_cmd + (v, w) in WORLD/base frame, FrankaEnv semantics.

        Position shifts along BASE axes; rotation is a base-frame rotvec
        applied about the TCP point (orientation-only left multiply).
        NOT T_cmd @ exp(xi): right-multiplication would mean tool-frame
        deltas, and the hil-serl stack (RelativeFrame wrapper) already
        assumes the raw env takes base-frame actions.
        """
        T = self.T_cmd.copy()
        T[:3, 3] = self.T_cmd[:3, 3] + v
        T[:3, :3] = so3_exp(w) @ self.T_cmd[:3, :3]
        return T

    def _net(self, T_des: np.ndarray):
        """The (v, w) that takes T_cmd to T_des — inverse of _integrate."""
        dp = T_des[:3, 3] - self.T_cmd[:3, 3]
        dw = so3_log(T_des[:3, :3] @ self.T_cmd[:3, :3].T)
        return dp, dw

    def step(self, xi: np.ndarray) -> Tuple[np.ndarray, dict]:
        """One control tick. Returns (q_cmd, info); HOLDs on any doubt."""
        assert self.T_cmd is not None, "call reset() before step()"
        info = {"held": False, "reject_reason": None, "clipped": False}

        xi = np.asarray(xi, dtype=float).reshape(6)
        if not np.all(np.isfinite(xi)):
            return self._hold(info, "BAD_INPUT")

        # ---- governor: per-tick rate cap in task space ---- #
        v, w, _ = self._govern(xi[:3].copy(), xi[3:].copy())

        # ---- WORLD-frame increment, FrankaEnv semantics ---- #
        T_des = self._integrate(v, w)

        # ---- workspace safety box, on the COMMANDED pose ---- #
        # Upstream order (FrankaEnv.step): build nextpos from the current pose,
        # clip_safety_box(nextpos), send. Ours is the same order, one level
        # deeper: we clip BEFORE IK so a clamped target still gets solved and
        # executed. Clipping after IK would mean either sending an unclamped
        # joint command or holding, and holding at the wall is exactly the
        # STEP_LIMIT storm the line search below exists to avoid.
        if self.clip_pose is not None:
            T_des, info["clipped"] = self.clip_pose(T_des)

        # ---- ANTI-WINDUP: the clamp must land in the integrator state ---- #
        # This is the one real difference from upstream and it is not optional.
        # FrankaEnv rebuilds nextpos from `self.currpos`, the MEASURED pose, so
        # its integrator is the robot itself and a clamped command simply
        # doesn't move the arm — nothing accumulates. This controller instead
        # integrates its OWN output (T_cmd), because the UR runs a stiff
        # position controller and re-seeding from measured joints every tick
        # would inject tracking error back into the target.
        #
        # So if we clamped the outgoing command but kept the unclamped pose in
        # T_cmd, every subsequent tick would restart from a target already
        # outside the box: the command state would run away at ACTION_SCALE per
        # step (0.01 m at 10 Hz = 0.1 m/s, unbounded) while the arm sat still at
        # the wall. The moment the policy reversed, the arm would do nothing for
        # however many seconds it took to unwind that phantom excursion, and
        # then lurch. Classic integrator windup, with a robot on the end.
        #
        # Writing the CLAMPED pose back into T_cmd (below: `self.T_cmd = T_des`)
        # makes the wall an actual hard stop: T_cmd can never leave the box, so
        # the very next opposing action produces motion on the very next tick.
        # It also keeps tcp_cmd() — the GELLO intervention anchor — inside the
        # box, so an intervention that starts at the wall is anchored to a pose
        # the arm can actually be at.
        #
        # Re-govern the NET step (T_cmd -> clamped T_des) rather than the
        # requested one. Identity while T_cmd is inside the box (clipping only
        # ever shortens the step). It matters when T_cmd starts OUTSIDE the box
        # — a reset pose outside the box, or a box narrowed between runs — where
        # the clamp is a jump of arbitrary size rather than a delta: this turns
        # the return into a rate-limited approach at v_max/w_max instead of one
        # unbounded lunge.
        v, w = self._net(T_des)
        v, w, scale = self._govern(v, w)
        if scale < 1.0:
            T_des = self._integrate(v, w)

        # ---- IK, seeded at previous command (branch continuity via seed) ---- #
        q_sol = ik_numeric(T_des, self.q_cmd)
        if q_sol is None:
            return self._hold(info, "NO_IK")
        if not within_joint_limits(q_sol, margin=0.0):
            return self._hold(info, "JOINT_LIMIT")

        # ---- line search: shrink the task step until the joint step fits ---- #
        # The proven eef_delta controller does exactly this. The earlier code
        # HARD-REJECTED the whole tick to HOLD the instant max|dq| > dq_step_max.
        # That is a self-reinforcing STEP_LIMIT storm: a rejected tick freezes
        # T_cmd while the leader/policy keeps demanding the full (growing) anchor
        # error, so the same over-budget step is re-rejected every tick and the
        # arm delivers ~2% of the intended motion (measured, well-conditioned
        # pose sigma=0.56 — not a singularity). Shrinking the task-space step to
        # a smaller FEASIBLE one keeps the arm flowing as a rate-limited chase
        # (measured: 98% held / 2% tracked -> 0% held / 47% tracked, max|dq|
        # stays under dq_step_max). dq is ~linear in the step at a well-
        # conditioned pose, so this converges in 1-2 iterations.
        #
        # NOTE it shrinks the NET step (v, w as recomputed above), not the
        # requested one. With a clamp in play those differ, and shrinking the
        # request would rebuild the same out-of-box target every iteration and
        # never converge — the arm would freeze at the wall instead of sliding
        # along it. Shrinking the net step interpolates toward T_cmd, which is
        # inside the box, so no re-clip is needed inside the loop.
        for _ in range(3):
            if float(np.max(np.abs(q_sol - self.q_cmd))) <= self.dq_step_max:
                break
            s = 0.9 * self.dq_step_max / float(np.max(np.abs(q_sol - self.q_cmd)))
            v, w = v * s, w * s
            T_des = self._integrate(v, w)
            q_sol = ik_numeric(T_des, self.q_cmd)
            if q_sol is None or not within_joint_limits(q_sol, margin=0.0):
                return self._hold(info, "NO_IK")
        else:
            # still over budget after shrinking (e.g. genuinely near-singular
            # Jacobian): HOLD is the correct, rare fallback the gate is for.
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
