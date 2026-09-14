#!/usr/bin/env python3
"""Cartesian (EEF) delta execution for the carrot policy -- pure math, no rclpy.

This is the whole of CARROT_EEF_DESIGN.md §3 items 3-5 in one testable class, so
the safety-critical arithmetic can be exercised without ROS, a robot, or a policy
server (``test/test_eef_executor.py``).  ``policy_leader_node`` owns only the ROS
plumbing around it: assemble the 16-D state, ZMQ the server, call :meth:`step`,
then run the UNCHANGED joint envelope / max-deviation clamps and publish.

Per 30 Hz tick, given the live joint vector and a 7-D action
``[dx, dy, dz, drx, dry, drz, grip_cmd]`` (base frame, metres / rotation vector):

  1. **Re-anchor** when the server says ``chunk_start`` (the network re-ran this
     tick) or on the first tick after arming: ``anchor = target = TCP(live_q)``.
  2. **Accumulate on the TARGET**, never on the live pose:
     ``target <- target (+) clamp_delta(delta)``.
     Rationale (design §3.3): a chunked policy predicts a trajectory from the one
     observation it saw.  Re-applying each delta to a *lagging* live pose would
     shrink every step by whatever the arm has not caught up yet, so a 30-step
     chunk would execute a fraction of the motion it was trained to produce.
  3. **Clamps, in this order, all norm-proportional** (never per-axis on a delta
     -- CLAUDE.md: an axis-wise clip bends the commanded direction, and a frame
     rotation does not preserve per-axis maxima):
       (a) per-tick delta   -- ``eef_space.clamp_delta(d, max_step_m, max_step_rad)``
       (b) lead             -- the target may never run away from a stalled arm:
                               ``|p_target - p_live| <= max_lead_m`` and
                               ``|so3_log(R_target R_live^T)| <= max_lead_rad``,
                               scaled down as a vector (same shape of code as
                               ``clamp_delta``; it IS ``clamp_delta`` applied to the
                               lead 6-vector).
       (c) workspace box    -- ``eef_space.clamp_position_box`` (axis-wise on a
                               POSITION, which is what a box is; unlike a delta a
                               position has no direction to preserve).
     A clamp is written BACK into the accumulator.  If it were only applied to the
     command, a stalled arm would let ``target`` integrate away unseen and then
     snap forward the moment the arm freed itself.
  4. **IK** ``eef_space.ik_tcp_nearest(target, live_q, branch=start_branch)`` with
     ``start_branch = ur_kin.branch_id(start_pose)`` computed once (§3.5).  On
     failure return ``None`` -- the caller holds ``live_q`` for that tick -- and
     count; ``ik_fail_fault_n`` CONSECUTIVE failures is a FAULT (the caller acts,
     this class only reports, so the fault policy stays in one place: the node).

Nothing here talks to ROS, zmq, or the clock; every decision is a pure function of
(config, accumulated target, this tick's live_q/action/chunk_start).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from gello_policy import eef_space
from ur_gello_bringup import ur_kin

# Design §3 defaults.  Duplicated as ROS parameter defaults in policy_leader_node
# (a ROS declare_parameter cannot take them from here without importing rclpy into
# this module's test path); test_eef_executor asserts the two agree.
DEFAULT_MAX_STEP_M = 0.02
DEFAULT_MAX_STEP_RAD = 0.10
DEFAULT_MAX_LEAD_M = 0.06
DEFAULT_MAX_LEAD_RAD = 0.30
DEFAULT_WORKSPACE_LO = (0.35, -0.30, 0.00)
DEFAULT_WORKSPACE_HI = (0.78, 0.45, 0.40)
DEFAULT_IK_FAIL_FAULT_N = 10


def apply_delta_pose(T: np.ndarray, action) -> np.ndarray:
    """Advance a TCP POSE by a delta -- thin alias of ``eef_space.apply_delta_pose``
    (moved into the single-source module by the integrator; kept here so callers
    and tests keep working)."""
    return eef_space.apply_delta_pose(T, action)


@dataclasses.dataclass(frozen=True)
class EefExecutorConfig:
    """Validated configuration for :class:`EefDeltaExecutor` (design §3 params).

    Field names deliberately drop the ``eef_`` prefix the ROS parameters carry
    (``eef_max_step_m`` -> ``max_step_m``): the prefix disambiguates inside a flat
    ROS parameter namespace shared with the joint path, and is noise here.  The
    ROS names are the contract with A6 and must not be renamed.

    ``start_branch`` is optional: when None it is derived from ``start_pose`` with
    ``ur_kin.branch_id`` (design §3.5 -- "computed once at arming"; it is a pure
    function of a constructor parameter, so computing it once at construction is
    the same value, and doing it early surfaces an unreachable start pose before
    the operator arms rather than on the first executing tick).
    """

    start_pose: Tuple[float, ...]
    max_step_m: float = DEFAULT_MAX_STEP_M
    max_step_rad: float = DEFAULT_MAX_STEP_RAD
    max_lead_m: float = DEFAULT_MAX_LEAD_M
    max_lead_rad: float = DEFAULT_MAX_LEAD_RAD
    workspace_lo: Tuple[float, float, float] = DEFAULT_WORKSPACE_LO
    workspace_hi: Tuple[float, float, float] = DEFAULT_WORKSPACE_HI
    ik_fail_fault_n: int = DEFAULT_IK_FAIL_FAULT_N
    start_branch: Optional[int] = None
    ik_limit_margin: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_pose", tuple(float(x) for x in self.start_pose))
        object.__setattr__(self, "workspace_lo", tuple(float(x) for x in self.workspace_lo))
        object.__setattr__(self, "workspace_hi", tuple(float(x) for x in self.workspace_hi))
        if len(self.start_pose) != 6:
            raise ValueError(f"start_pose must have 6 entries, got {len(self.start_pose)}")
        for name in ("workspace_lo", "workspace_hi"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"{name} must have 3 entries, got {len(getattr(self, name))}")
        if any(hi < lo for lo, hi in zip(self.workspace_lo, self.workspace_hi)):
            raise ValueError(
                f"workspace_lo must be <= workspace_hi elementwise "
                f"({self.workspace_lo} vs {self.workspace_hi})"
            )
        # A non-positive clamp is silently "disabled" inside eef_space.clamp_delta
        # (`n > max > 0`). Refuse it here: a disabled Cartesian clamp is exactly the
        # kind of safety hole that must never be reachable by a typo in a yaml.
        for name in ("max_step_m", "max_step_rad", "max_lead_m", "max_lead_rad"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if int(self.ik_fail_fault_n) < 1:
            raise ValueError(f"ik_fail_fault_n must be >= 1, got {self.ik_fail_fault_n}")
        object.__setattr__(self, "ik_fail_fault_n", int(self.ik_fail_fault_n))


class EefDeltaExecutor:
    """Accumulate EEF deltas onto a target pose and turn it into a joint target.

    Stateful across ticks (that is the point): it owns the chunk anchor, the
    accumulated target pose and the consecutive-IK-failure counter.  Thread
    confinement is the caller's job -- the node drives it from its single timer
    callback, like every other piece of that node's state.
    """

    def __init__(self, cfg: EefExecutorConfig):
        self.cfg = cfg
        if cfg.start_branch is None:
            branch = int(ur_kin.branch_id(np.asarray(cfg.start_pose, dtype=float)))
        else:
            branch = int(cfg.start_branch)
        self._start_branch = branch
        self._anchor: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        self._ik_fail_streak = 0
        self._ticks = 0

    # ---- introspection ---------------------------------------------------
    @property
    def start_branch(self) -> int:
        """Closed-form IK branch every target is locked to (design §1: always 2)."""
        return self._start_branch

    @property
    def target_pose(self) -> Optional[np.ndarray]:
        """Current accumulated TCP target (4x4), or None before the first tick."""
        return None if self._target is None else self._target.copy()

    @property
    def anchor_pose(self) -> Optional[np.ndarray]:
        """TCP pose the current chunk was anchored on (4x4), or None."""
        return None if self._anchor is None else self._anchor.copy()

    @property
    def ik_fail_streak(self) -> int:
        """Consecutive IK failures; ``>= cfg.ik_fail_fault_n`` means FAULT."""
        return self._ik_fail_streak

    # ---- lifecycle -------------------------------------------------------
    def reset(self, live_q: Optional[Sequence[float]] = None) -> None:
        """Forget the chunk (call on arming and on ``~/hold``).

        With ``live_q`` the anchor is taken immediately; without it the next
        :meth:`step` anchors on its own ``live_q``.  Both are correct: after a
        RESET the server's queue is empty, so its first reply carries
        ``chunk_start=True`` and re-anchors anyway.  The explicit form exists so a
        node that arms and only starts ticking later still has a defined target,
        and so the state is never half-initialised.
        """
        self._ik_fail_streak = 0
        self._ticks = 0
        if live_q is None:
            self._anchor = None
            self._target = None
        else:
            T = eef_space.tcp_pose(live_q)
            self._anchor = T.copy()
            self._target = T.copy()

    # ---- per-tick --------------------------------------------------------
    def step(
        self,
        live_q: Sequence[float],
        action: Sequence[float],
        chunk_start: bool = False,
    ) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
        """One EXECUTE tick.  Returns ``(q_target | None, info)``.

        ``None`` means "IK found nothing for this target": the caller must publish
        the live pose (hold) for this tick and FAULT once ``info["ik_fault"]`` is
        True.  ``info`` carries every clamp flag, the raw and clamped norms, and
        the IK diagnostics, so a log line can explain any motion the operator sees.
        """
        live_q = np.asarray(live_q, dtype=float).reshape(6)
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size < 6:
            raise ValueError(f"eef action needs >= 6 entries, got {a.size}")

        T_live = eef_space.tcp_pose(live_q)

        # --- (1) chunk anchor --------------------------------------------
        re_anchored = bool(chunk_start) or self._target is None
        if re_anchored:
            self._anchor = T_live.copy()
            self._target = T_live.copy()

        # --- (2) per-tick delta clamp, then accumulate ON THE TARGET ------
        delta_pos_norm = float(np.linalg.norm(a[:3]))
        delta_rot_norm = float(np.linalg.norm(a[3:6]))
        delta, clamped_step_pos, clamped_step_rot = eef_space.clamp_delta(
            a[:6], self.cfg.max_step_m, self.cfg.max_step_rad
        )
        self._target = apply_delta_pose(self._target, delta)

        # --- (3) lead clamp: the target may not run away from a stalled arm
        # Expressed as a delta from live to target and pushed through the SAME
        # norm-proportional helper the per-tick clamp uses, so the direction of
        # the lead is preserved exactly like the direction of a delta.
        lead = np.empty(6)
        lead[:3] = self._target[:3, 3] - T_live[:3, 3]
        lead[3:6] = ur_kin.so3_log(self._target[:3, :3] @ T_live[:3, :3].T)
        lead_pos_norm = float(np.linalg.norm(lead[:3]))
        lead_rot_norm = float(np.linalg.norm(lead[3:6]))
        lead_c, clamped_lead_pos, clamped_lead_rot = eef_space.clamp_delta(
            lead, self.cfg.max_lead_m, self.cfg.max_lead_rad
        )
        if clamped_lead_pos or clamped_lead_rot:
            # Rebuild ONLY when a clamp engaged: a no-op rebuild would still round
            # trip the rotation through log/exp and inject drift every single tick.
            self._target = apply_delta_pose(T_live, lead_c)

        # --- (4) workspace box on the target POSITION ---------------------
        p_box, clamped_box = eef_space.clamp_position_box(
            self._target[:3, 3], self.cfg.workspace_lo, self.cfg.workspace_hi
        )
        if clamped_box:
            self._target[:3, 3] = p_box

        # --- (5) IK, branch-locked to the start pose ----------------------
        q_target, ik_info = eef_space.ik_tcp_nearest(
            self._target,
            live_q,
            branch=self._start_branch,
            limit_margin=self.cfg.ik_limit_margin,
        )
        if q_target is None:
            self._ik_fail_streak += 1
        else:
            self._ik_fail_streak = 0
        self._ticks += 1

        info: Dict[str, object] = {
            "tick": self._ticks,
            "chunk_start": bool(chunk_start),
            "re_anchored": re_anchored,
            "delta_pos_norm": delta_pos_norm,
            "delta_rot_norm": delta_rot_norm,
            "delta_pos_norm_clamped": float(np.linalg.norm(delta[:3])),
            "delta_rot_norm_clamped": float(np.linalg.norm(delta[3:6])),
            "clamped_step_pos": bool(clamped_step_pos),
            "clamped_step_rot": bool(clamped_step_rot),
            "lead_pos_norm": lead_pos_norm,
            "lead_rot_norm": lead_rot_norm,
            "lead_pos_norm_clamped": float(np.linalg.norm(lead_c[:3])),
            "lead_rot_norm_clamped": float(np.linalg.norm(lead_c[3:6])),
            "clamped_lead_pos": bool(clamped_lead_pos),
            "clamped_lead_rot": bool(clamped_lead_rot),
            "clamped_box": bool(clamped_box),
            "target_pos": tuple(float(v) for v in self._target[:3, 3]),
            "live_pos": tuple(float(v) for v in T_live[:3, 3]),
            "start_branch": self._start_branch,
            "ik_ok": q_target is not None,
            "ik_branch": int(ik_info["branch"]),
            "ik_on_branch": bool(ik_info["on_branch"]),
            "ik_n_solutions": int(ik_info["n_solutions"]),
            "ik_max_dev": float(ik_info["max_dev"]),
            "ik_fail_streak": self._ik_fail_streak,
            "ik_fault": self._ik_fail_streak >= self.cfg.ik_fail_fault_n,
        }
        info["any_clamped"] = bool(
            clamped_step_pos
            or clamped_step_rot
            or clamped_lead_pos
            or clamped_lead_rot
            or clamped_box
        )
        return q_target, info

    # ---- logging helper --------------------------------------------------
    @staticmethod
    def clamp_reasons(info: Dict[str, object]) -> list:
        """Human-readable list of the clamps that engaged this tick (may be empty)."""
        reasons = []
        if info.get("clamped_step_pos"):
            reasons.append(f"eef_step_m ({info['delta_pos_norm']:.4f} m)")
        if info.get("clamped_step_rot"):
            reasons.append(f"eef_step_rad ({info['delta_rot_norm']:.4f} rad)")
        if info.get("clamped_lead_pos"):
            reasons.append(f"eef_lead_m ({info['lead_pos_norm']:.4f} m)")
        if info.get("clamped_lead_rot"):
            reasons.append(f"eef_lead_rad ({info['lead_rot_norm']:.4f} rad)")
        if info.get("clamped_box"):
            reasons.append("workspace_box")
        return reasons
