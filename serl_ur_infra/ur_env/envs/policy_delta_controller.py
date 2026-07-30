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

The tick length is a per-call argument (``step(xi, dt=...)``), not a fixed
``1/hz``: the policy path runs at 10 Hz but GELLO intervention sub-steps the
same gate stack at 30 Hz, and the governor's caps are rate caps, so they have
to know how long the tick they are capping is. ``dt=None`` reproduces the
10 Hz behaviour bit for bit. See ``step()`` for why the controller never
measures dt off the wall clock.

Three callers, one gate stack, and — since 2026-07-30 — TWO THREADS. Policy
path: ``UR7eEnv._apply_action`` (``dt=None``). In-window intervention path:
``GelloIntervention.substep`` (``dt=1/30``), paced by
``UR7eEnv._drive_intervention_substeps``. Background intervention path:
``UR7eEnv._follow_tick`` (``dt=1/30``) on a daemon thread that follows the
leader while the RL loop is off doing gRPC. All three enter through
``UR7eEnv._emit_arm_command``, which admits one owner at a time; this class
locks itself as well, for the measured reasons in ``__init__``. The ordered call
chain of the intervention paths is written out in the ``wrappers.py`` module
docstring.

TODO(together): replace the simplified internals below with the real
EefDeltaController machinery (sigma_min throttle with asymmetric escape,
analytic branch-lock, keepout, line search). Cleanest path is probably
refactoring eef_delta so its post-anchor pipeline is callable with a T_des
directly — then this file shrinks to a thin adapter and both the policy path
and GELLO intervention share one gate stack.

REAL-HARDWARE STATUS (supersedes the "UNTESTED SKELETON — do not run against
real hardware" line this file used to carry): this gate stack has driven the real
UR7e. The HIL intervention runner passed 2026-07-28, the first full
policy+intervention E2E ran 2026-07-29, and the 30 Hz intervention sub-step path
passed ARMED 2026-07-30 (dp_ratio median 1.000, held 0% over 120 intervened
steps; docs/testing/04_HIL_INTERVENTION.md §9). Two things are still NOT proven
on hardware: the TODO above (the real eef_delta machinery is not in here,
08_OPEN_GAPS.md G2), and the governor's rate-cap truncation — ``governed`` was
False in every window of the 07-30 run, so that branch has only ever been
exercised by tests/test_governor_dt.py.
"""

import threading
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
        # NOMINAL tick length. It is the default for step(dt=None) and the
        # reference the per-call dt is scaled against; it is NOT a measurement.
        # See step() for why nothing here ever samples the wall clock.
        self.dt = 1.0 / hz
        self.clip_pose = clip_pose

        # ---- THREAD SAFETY (measured, not defensive) --------------------- #
        # Two threads reach this object once human intervention is followed in
        # the background: the RL step thread and the 30 Hz follower.  UR7eEnv's
        # ownership mux is the primary guarantee that only one of them commands
        # at a time, but a controller whose invariants depend on a caller's
        # discipline is a controller that will eventually be driven by two.
        # Reproduced against this class + the real ur_kin, 2026-07-30:
        #
        #  * (T_cmd, q_cmd) DISAGREED on 12,143 of 108,949 samples (11.1%).
        #    They were two separate stores and ``_integrate`` re-read
        #    ``self.T_cmd`` three times, so IK could be seeded with one tick's
        #    q against another tick's T.  Branch continuity is exactly what that
        #    seed provides; losing it puts a wrist a full turn away, and the
        #    250 Hz upsampler rate-limits but does not veto — it executes it as a
        #    ~10 s blind sweep, i.e. the cable-winding hazard H3.
        #  * 25 of 130,028 HOLD ticks PUBLISHED MOTION, up to 0.2047 rad
        #    (3.3x dq_step_max), because ``_hold`` returned ``self.q_cmd`` — a
        #    field the other thread had meanwhile overwritten.  The operator's
        #    display says HOLD while the arm moves.
        #  * The ``dq_step_max`` gate leaked on 6.08% of ticks for the same
        #    reason: it is measured against ``self.q_cmd``.
        #
        # The fix is structural, not a re-order: one re-entrant lock covers each
        # public entry point end to end, ``T_cmd``/``q_cmd`` are only ever
        # replaced together by ``_commit``, and ``_hold`` returns the command
        # this controller last actually PUBLISHED rather than re-reading a field.
        self._lock = threading.RLock()

        self.T_cmd: Optional[np.ndarray] = None
        self.q_cmd: Optional[np.ndarray] = None
        #: The joint vector this controller last handed a caller. Held apart from
        #: ``q_cmd`` so a HOLD re-issues what was really published, and cannot be
        #: turned into motion by whatever last wrote ``q_cmd``.
        self._last_issued: Optional[np.ndarray] = None

    def _commit(self, T_cmd: np.ndarray, q_cmd: np.ndarray) -> np.ndarray:
        """Replace the command state ATOMICALLY. Caller holds ``self._lock``.

        The single writer of ``T_cmd``/``q_cmd``/``_last_issued``. Keeping it to
        one place is what makes "the pair is always from the same tick" a
        property of the class rather than a property of every call site
        remembering to assign both.
        """

        self.T_cmd = T_cmd
        self.q_cmd = q_cmd
        self._last_issued = q_cmd.copy()
        return self._last_issued.copy()

    def reset(self, q_now: np.ndarray):
        """Latch the command state to the robot's actual pose (call on env.reset)."""
        q_now = np.asarray(q_now, dtype=float).reshape(6)
        with self._lock:
            self._commit(fk(q_now), q_now.copy())

    def _govern(self, v: np.ndarray, w: np.ndarray, dt: float):
        """Per-tick rate cap in task space. Returns (v, w, scale).

        ``dt`` is the length of the tick this (v, w) is meant to cover, in
        seconds. It is an argument rather than ``self.dt`` because the caller
        decides the tick rate: the policy path runs at ``1/hz`` while the
        intervention sub-step driver runs the same controller at 30 Hz, and a
        cap of ``v_max * self.dt`` applied to a 1/30 s tick would let three
        sub-steps travel 3x the 10 Hz budget (0.045 m instead of 0.015 m).
        """
        nv, nw = float(np.linalg.norm(v)), float(np.linalg.norm(w))
        scale = 1.0
        if nv > 1e-12:
            scale = min(scale, self.v_max * dt / nv)
        if nw > 1e-12:
            scale = min(scale, self.w_max * dt / nw)
        scale = min(scale, 1.0)
        return v * scale, w * scale, scale

    def _resolve_dt(self, dt: Optional[float]) -> Tuple[float, float]:
        """(dt, dq_step_max) for one call. ``None`` -> the nominal tick, verbatim.

        WHY dq_step_max SCALES WITH dt.  ``GOVERNOR["dq_step_max"]`` is written
        per tick but is specified as a RATE: config.py:106-109 pins 0.0625 rad at
        10 Hz as "the same rate ceiling" as the 250 Hz upsampler's 0.0025
        rad/tick, i.e. 0.625 rad/s, and config.py:66-72 requires the three speed
        layers (ACTION_SCALE / GOVERNOR / UPSAMPLER) to stay in a fixed ratio
        because "raise one alone and the next silently truncates it". Holding
        0.0625 rad fixed while the tick shrinks to 1/30 s would raise the joint
        rate to 1.875 rad/s — 3x past the ceiling the UPSAMPLER can actually
        emit (ros_backend streams max_step_rad per 250 Hz tick), so the extra
        would not reach the arm as motion; it would accumulate as command-vs-
        measured lag, which is the exact stiffness the 30 Hz sub-stepping exists
        to remove. Scaling the gate by dt/self.dt keeps the rate invariant, so
        the line search below still means "one tick's worth of joint motion".

        ``dt is None`` returns ``self.dq_step_max`` unmultiplied so the policy
        path is bit-identical to the pre-sub-step controller, not merely equal
        to within a float multiply by 1.0.
        """
        if dt is None:
            return self.dt, self.dq_step_max
        dt = float(dt)
        # Reject non-finite/non-positive rather than clamping: dt drives a
        # SAFETY cap, and a silently-substituted default would be a cap the
        # caller never asked for. dt<=0 collapses the cap to zero (arm frozen),
        # NaN makes every comparison False (cap effectively removed).
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}")
        return dt, self.dq_step_max * (dt / self.dt)

    def _integrate(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """T_cmd + (v, w) in WORLD/base frame, FrankaEnv semantics.

        Position shifts along BASE axes; rotation is a base-frame rotvec
        applied about the TCP point (orientation-only left multiply).
        NOT T_cmd @ exp(xi): right-multiplication would mean tool-frame
        deltas, and the hil-serl stack (RelativeFrame wrapper) already
        assumes the raw env takes base-frame actions.

        ONE read of ``self.T_cmd``, into a local. The three separate reads this
        used to do could each land on a different tick's matrix if another
        thread committed in between — a base pose assembled from two ticks, and
        the resulting IK seed mismatch is the wrist-flip hazard documented in
        ``__init__``. ``step`` holds the lock across the whole call, so this is
        belt-and-braces; it is written this way so it stays correct if that ever
        stops being true.
        """
        base = self.T_cmd
        T = base.copy()
        T[:3, 3] = base[:3, 3] + v
        T[:3, :3] = so3_exp(w) @ base[:3, :3]
        return T

    def _net(self, T_des: np.ndarray):
        """The (v, w) that takes T_cmd to T_des — inverse of _integrate."""
        base = self.T_cmd
        dp = T_des[:3, 3] - base[:3, 3]
        dw = so3_log(T_des[:3, :3] @ base[:3, :3].T)
        return dp, dw

    def step(
        self, xi: np.ndarray, dt: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """One control tick. Returns (q_cmd, info); HOLDs on any doubt.

        ``dt`` — length of THIS tick in seconds, for the governor's rate cap
        and the joint-step gate. ``None`` (the default, and what UR7eEnv.step
        passes at ur7e_env.py:501) means the nominal ``1/hz``, handled so the
        policy path stays bit-identical to before this parameter existed.
        The 30 Hz intervention sub-step driver passes ``dt=1/30``.

        WHY THE CALLER PASSES dt INSTEAD OF THE CONTROLLER MEASURING IT.
        A ``time.monotonic()`` diff inside step() looks strictly better and is
        wrong here, because the governor is a hardware safety net (config.py:87-98)
        and wall-clock dt makes that net a function of unrelated load:

        * Real steps have been measured stretching to ~1.12 s under learner
          contention (CLAUDE.md; the 0.6 s Step-RPC deadline blew past). A
          measured dt of 1.12 s would hand that tick a 0.168 m / 0.84 rad cap —
          11x the intended budget — precisely when the system is least healthy.
        * Jitter the other way silently CUTS policy actions that used to pass,
          so identical (obs, action) pairs would produce different motion run to
          run and the stored-action invariant (config.py:55-63) would break in a
          way no log could explain.

        Explicit dt keeps the cap a property of the control schedule the caller
        actually implements, and keeps it reproducible in replay.

        ATOMIC. The whole body runs under ``self._lock``, so no other thread can
        observe — or seed IK from — a half-updated command state, and the
        ``dq_step_max`` gate below is measured against a ``q_cmd`` nobody can
        replace while it is being measured (see ``__init__`` for the 11.1% /
        6.08% / 25-moving-HOLDs reproduction that made this mandatory). Re-entrant
        so ``_hold`` and ``_commit`` can be called from inside it.
        """
        with self._lock:
            return self._step_locked(xi, dt)

    def _step_locked(
        self, xi: np.ndarray, dt: Optional[float]
    ) -> Tuple[np.ndarray, dict]:
        assert self.T_cmd is not None, "call reset() before step()"
        # governed/governed_scale: the governor used to shrink actions with no
        # trace at all — `scale` was computed and dropped on the floor, and
        # info["clipped"] covers only the workspace box, a DIFFERENT gate. A
        # silently rate-capped action is the same undiagnosable failure as a
        # silently clamped one (ur7e_env.py:699-704): big action, little motion,
        # nothing in the reward to explain it. It is also the only observable
        # for the ACTION_SCALE*HZ < v_max/w_max headroom invariant
        # (config.py:55-63) being violated at run time, i.e. for the stored
        # transitions overstating the motion that actually happened.
        # These are ADDITIVE: held/reject_reason/clipped keep their exact
        # meaning, and ur7e_env.step's info.update(ctrl_info) forwards the new
        # keys unchanged.
        info = {
            "held": False,
            "reject_reason": None,
            "clipped": False,
            "governed": False,
            "governed_scale": 1.0,
        }
        dt, dq_step_max = self._resolve_dt(dt)

        xi = np.asarray(xi, dtype=float).reshape(6)
        if not np.all(np.isfinite(xi)):
            return self._hold(info, "BAD_INPUT")

        # ---- governor: per-tick rate cap in task space ---- #
        v, w, scale_req = self._govern(xi[:3].copy(), xi[3:].copy(), dt)

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
        v, w, scale_net = self._govern(v, w, dt)
        if scale_net < 1.0:
            T_des = self._integrate(v, w)

        # Report the TIGHTER of the two governor passes. Both are the same rate
        # cap: the first bounds what the policy/leader asked for, the second
        # bounds the post-clamp net step (only ever binds when T_cmd starts
        # outside the box, see above). The line search below is deliberately NOT
        # folded in — that is the joint-step gate, a different limit, whose
        # terminal case already reports reject_reason="STEP_LIMIT".
        # NOT observed on real hardware: the 2026-07-30 ARMED run reported
        # governed=0 in every window, so the truncation this reports is covered
        # by tests only (see the module docstring's REAL-HARDWARE STATUS).
        info["governed_scale"] = float(min(scale_req, scale_net))
        info["governed"] = info["governed_scale"] < 1.0

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
        #
        # dq_step_max here is the dt-SCALED budget from _resolve_dt (identical
        # to self.dq_step_max on the dt=None policy path); see that docstring
        # for why a shorter tick must get a proportionally smaller joint step.
        for _ in range(3):
            if float(np.max(np.abs(q_sol - self.q_cmd))) <= dq_step_max:
                break
            s = 0.9 * dq_step_max / float(np.max(np.abs(q_sol - self.q_cmd)))
            v, w = v * s, w * s
            T_des = self._integrate(v, w)
            q_sol = ik_numeric(T_des, self.q_cmd)
            if q_sol is None or not within_joint_limits(q_sol, margin=0.0):
                return self._hold(info, "NO_IK")
        else:
            # still over budget after shrinking (e.g. genuinely near-singular
            # Jacobian): HOLD is the correct, rare fallback the gate is for.
            return self._hold(info, "STEP_LIMIT")

        return self._commit(T_des, q_sol), info

    def _hold(self, info: dict, reason: str):
        """HOLD: re-issue the previous command, unchanged.

        Leaves governed/governed_scale as recorded: they describe what the rate
        cap did to the REQUEST, which is still the truth even though the tick
        ended in a HOLD. Consumers read `held` first — it means no new command
        was issued at all.

        Returns ``_last_issued`` and NOT ``self.q_cmd``. The docstring above has
        always claimed "unchanged", and with one caller the two were the same
        vector; with two threads they are not. Reproduced 2026-07-30: 25 of
        130,028 HOLD ticks published a command up to 0.2047 rad (3.3x
        dq_step_max) away from the previous one, because ``q_cmd`` had been
        rewritten by the other caller in between. A HOLD that moves the arm
        while the GUI shows HOLD is the worst failure this class can have, so it
        re-issues a value nothing else writes.
        """
        info["held"] = True
        info["reject_reason"] = reason
        with self._lock:
            return self._last_issued.copy(), info

    # convenience for wrappers that need the commanded TCP pose
    def tcp_cmd(self) -> np.ndarray:
        with self._lock:
            return self.T_cmd.copy()

    def joint_cmd(self) -> np.ndarray:
        """The joint vector last published, i.e. what a HOLD would re-issue."""
        with self._lock:
            return self._last_issued.copy()
