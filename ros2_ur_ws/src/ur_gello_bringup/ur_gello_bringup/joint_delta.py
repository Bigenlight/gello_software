"""Start-anchored JOINT-space delta controller for GELLO -> UR7e teleop.

Pure math (stdlib ``math`` + the stdlib-only ``angle_utils``), NO rclpy, NO
numpy, NO kinematics. Where :mod:`ur_gello_bringup.eef_delta` maps an
end-effector POSE delta through FK/IK, this module maps a JOINT delta with
nothing but arithmetic -- so joint_delta mode never pulls in the analytic-IK
dependency stack, exactly like joint mode.

    engage    -> latch (q_robot_anchor, q_lead_anchor); delta := 0
    step      -> delta += wrap_to_pi(q_lead - q_lead_prev)        (per joint)
                 q     := q_robot_anchor + gain * delta
                 q     := clamp(q, joint limits, excursion band)  (anti-windup)
                 q     := q_cmd_prev + clip(q - q_cmd_prev, +-step_eff)
    clutch    -> freeze the output AND the accumulator (operator repositions
                 the leader without moving the robot -- the "mouse lift")
    reclutch  -> re-anchor at (leader now, command now); delta := 0
    disengage -> freeze, anchor void

Conventions that are the whole point of this module (get them wrong and (c)/(d)
in the test-suite fail):

  * THE INCREMENT IS WRAPPED PER TICK, NOT AGAINST THE ANCHOR.

        d[i]      = wrap_to_pi(q_lead[i] - q_lead_prev[i])   # <-- correct
        delta[i] += d[i]

    NOT ``delta[i] = wrap_to_pi(q_lead[i] - q_lead_anchor[i])``. The absolute
    form FOLDS at +-pi: a deliberate 200 deg leader excursion would come back as
    -160 deg and command the robot BACKWARDS. Per-tick increments telescope
    exactly (the sum of the increments IS the total travel), are unbounded in
    range, and are branch-cut safe on every tick -- including a crossing that
    happens long AFTER the anchor was latched, which is why the guard has to be
    stateless-per-tick rather than a one-off fix at engage time.

  * BECAUSE THE INCREMENTS TELESCOPE, THE ACCUMULATOR IS DRIFT-FREE **SO LONG AS
    NO POSITION CLAMP SATURATES**. ``delta_deadband_rad`` therefore defaults to
    0.0: any nonlinearity applied to the increment (a deadband above all)
    CREATES a bias that integrates into drift instead of removing one. Jitter is
    fought upstream (the bridge's 1-Euro leader filter) and at the output (the
    position and slew clamps), never inside the accumulator.

  * POSITION CLAMPS BACK-PROJECT INTO THE ACCUMULATOR (anti-windup), THE SLEW
    CLAMP DOES NOT. Saturating at a joint limit without back-projection would
    let ``delta`` wind up arbitrarily far past the limit, and the operator's
    return stroke would then move nothing until the whole windup had been paid
    back -- a dead zone that feels exactly like a broken teleop. Slew binding,
    on the other hand, is transient lag that must be caught up on later ticks,
    so rewriting the accumulator there would silently DISCARD leader motion.

    THE PRICE OF THAT CHOICE, STATED HONESTLY (it used to be mis-stated here as
    unconditional drift-freedom): back-projection DISCARDS the leader travel
    that went past the clamp, so once a position clamp has bound, returning the
    leader EXACTLY to its anchor pose no longer returns the command exactly to
    ``q_robot_anchor`` -- it lands short by the discarded overtravel, in the
    OPPOSITE direction. Worked example with the shipped ``gain=1.0`` and
    ``max_excursion_rad=1.0``: leader +1.5 rad out (command saturates at +1.0,
    0.5 rad discarded), leader back to 0.0 -> command settles at -0.5 rad and
    STAYS there. The trade is deliberate and it is BOUNDED (the command can
    never leave the anchor +- max_excursion cage) and OBSERVABLE:

        info["limited"][i]          True on every tick the clamp bound
        info["clamp_discarded"][i]  cumulative rad of leader travel discarded

    Both are republished on the node's ~/joint_delta/state. The operator remedy
    is the one this mode already ships: clutch -> reposition -> reclutch (which
    re-anchors and zeroes both the accumulator and the discard counters), or
    ~/joint_delta_to_joint to hand back to absolute mapping. Neither of the two
    alternatives is better: pure windup keeps the correspondence but buys an
    UNBOUNDED return-stroke dead zone, and a HOLD-on-saturation freezes all six
    joints because one of them reached its cage.

  * THE TELEPORT GUARD IS A BACKSTOP, NOT THE PRIMARY DETECTOR. ``step()`` sees
    whatever the caller feeds it, which in the bridge is the 1-EURO-FILTERED
    leader at the 250 Hz publish tick -- a low-pass smears a raw teleport across
    many ticks, so a raw glitch of ~1.2 rad or less never produces a single
    filtered increment above ``leader_jump_max_rad`` and is accumulated in full.
    Worse, a glitch big enough to trip it trips it only for the first few ticks
    of the smear and then the sub-threshold tail is accumulated, which is an
    ASYMMETRIC (out-stroke only) discard and therefore a permanent offset. The
    primary detector must therefore run on the RAW leader stream at the SOURCE
    cadence; the node does exactly that in ``_on_joint_state`` and drives the
    resulting hold window through :meth:`JointDeltaController.absorb`.
"""

from __future__ import annotations

import math
from typing import List, Optional

from ur_gello_bringup.angle_utils import wrap_to_pi


REJECT_BAD_INPUT = "BAD_INPUT"      # non-finite leader joints or step budget
REJECT_ESTOP = "ESTOP"              # non-positive step budget -> hard freeze
REJECT_LEADER_JUMP = "LEADER_JUMP"  # single-tick leader increment over the cap
# Driven by the CALLER (the node's raw-domain teleport detector), not by step():
# hold the output and re-sync the accumulator reference while the upstream filter
# digests a raw leader teleport. See absorb().
REJECT_LEADER_RESYNC = "LEADER_RESYNC"

_TWO_PI = 2.0 * math.pi

_DEFAULTS = {
    "gain": 1.0,
    "limit_margin_rad": 0.05,
    # VERBATIM copy of ur_kin.JOINT_LIMITS (elbow, index 2, is physically +-pi:
    # shoulder_lift interference, ros-industrial/universal_robot#265). Copied as
    # a plain tuple rather than imported so this module stays numpy-free;
    # test_joint_delta.test_p asserts the two never drift apart.
    "joint_limits": (
        (-_TWO_PI, _TWO_PI),
        (-_TWO_PI, _TWO_PI),
        (-math.pi, math.pi),
        (-_TWO_PI, _TWO_PI),
        (-_TWO_PI, _TWO_PI),
        (-_TWO_PI, _TWO_PI),
    ),
    "max_excursion_rad": 1.0,
    "leader_jump_max_rad": 0.35,
    "delta_deadband_rad": 0.0,
    "n_joints": 6,
}


def _as_float_list(values, n: int) -> Optional[List[float]]:
    """Coerce a length-n sequence to a list of floats, or None if impossible.

    Deliberately total (never raises): it runs BEFORE the sanitize gate, on the
    DISENGAGED/CLUTCHED short-circuit path too, where a malformed input must
    still produce a well-formed ``info`` dict rather than an exception.
    """
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(out) != n:
        return None
    return out


class JointDeltaController:
    """Anchor + per-tick joint increment + clamps (see the module docstring)."""

    def __init__(self, cfg: Optional[dict] = None):
        g = dict(_DEFAULTS)
        g.update(dict(cfg or {}))
        self.cfg = g

        self.n = int(g["n_joints"])
        self.gain = float(g["gain"])
        self.limit_margin = float(g["limit_margin_rad"])
        self.max_excursion = float(g["max_excursion_rad"])
        self.leader_jump_max = float(g["leader_jump_max_rad"])
        self.delta_deadband = float(g["delta_deadband_rad"])
        self.joint_limits = [
            (float(lo), float(hi)) for (lo, hi) in g["joint_limits"]
        ]

        # Validation up front (mirrors eef_delta's rot_scale guard): a bad gain
        # or a non-positive cage is a CONFIG bug, and failing at construction is
        # far better than discovering it mid-teleop.
        if not (0.0 <= self.gain <= 2.0):
            raise ValueError(
                f"gain must be in [0.0, 2.0] (got {self.gain}); 0.0 = the robot "
                "never moves (staged bring-up), 1.0 = 1:1"
            )
        if self.max_excursion <= 0.0:
            raise ValueError(
                f"max_excursion_rad must be > 0 (got {self.max_excursion})"
            )
        if self.leader_jump_max <= 0.0:
            raise ValueError(
                f"leader_jump_max_rad must be > 0 (got {self.leader_jump_max})"
            )
        if self.delta_deadband < 0.0:
            raise ValueError(
                f"delta_deadband_rad must be >= 0 (got {self.delta_deadband})"
            )
        if self.limit_margin < 0.0:
            raise ValueError(
                f"limit_margin_rad must be >= 0 (got {self.limit_margin})"
            )
        if len(self.joint_limits) != self.n:
            raise ValueError(
                f"joint_limits must have {self.n} entries "
                f"(got {len(self.joint_limits)})"
            )
        for i, (lo, hi) in enumerate(self.joint_limits):
            if not (lo < hi):
                raise ValueError(f"joint_limits[{i}] is not lo < hi: ({lo}, {hi})")

        # Anchor / running state. q_cmd is None until the first engage(): a
        # controller that has never engaged must NOT emit a joint vector (same
        # foot-gun argument as eef_delta.py -- a consumer that streams without
        # checking state would command a zero-pose lurch).
        self.state = "DISENGAGED"          # DISENGAGED | ENGAGED | CLUTCHED
        self.q_robot_anchor: Optional[List[float]] = None
        self.q_lead_anchor: Optional[List[float]] = None   # diagnostics only
        self.q_lead_prev: Optional[List[float]] = None     # accumulator ref
        self.delta: List[float] = [0.0] * self.n
        self.q_cmd: Optional[List[float]] = None
        self.jump_absorbed = 0
        # Cumulative leader travel (rad, per joint) THROWN AWAY by the position
        # clamp's anti-windup back-projection since the current anchor. This is
        # exactly how far the leader<->arm correspondence has silently shifted;
        # see the module docstring. Zeroed by every (re)anchor.
        self.clamp_discarded: List[float] = [0.0] * self.n

    # ------------------------------------------------------------------ #
    # Engage / clutch / reclutch / disengage                              #
    # ------------------------------------------------------------------ #
    def _set_anchor(self, q_robot_anchor, q_lead_anchor) -> None:
        q_r = _as_float_list(q_robot_anchor, self.n)
        q_l = _as_float_list(q_lead_anchor, self.n)
        if q_r is None or q_l is None:
            raise ValueError(
                f"anchor vectors must both be {self.n} finite-length sequences "
                f"of numbers (got {q_robot_anchor!r} / {q_lead_anchor!r})"
            )
        self.q_robot_anchor = q_r
        self.q_lead_anchor = list(q_l)
        self.q_lead_prev = list(q_l)
        self.delta = [0.0] * self.n
        # A fresh anchor IS a fresh leader<->arm correspondence, so the record of
        # how much the old one had drifted must not survive it.
        self.clamp_discarded = [0.0] * self.n
        # ZERO JUMP IS ALGEBRAIC, NOT A TOLERANCE: q_cmd starts AT the anchor and
        # the first step() with an unmoved leader adds exactly 0.0 to it.
        self.q_cmd = list(q_r)

    def engage(self, q_anchor, q_lead_anchor) -> dict:
        """Latch the anchor at engage-time (t0). Returns an anchor summary.

        ``q_anchor`` is the ROBOT-side anchor and must come from the COMMAND
        chain (the bridge's ``_last_published``), never from a raw actual-pose
        reading, so the command stream stays continuous across the engage --
        same rule as :meth:`EefDeltaController.engage`.
        """
        self._set_anchor(q_anchor, q_lead_anchor)
        self.state = "ENGAGED"
        return {
            "state": self.state,
            "q_robot_anchor": list(self.q_robot_anchor),
            "q_lead_anchor": list(self.q_lead_anchor),
            "gain": self.gain,
        }

    def clutch(self) -> dict:
        """Freeze the output AND the accumulator (the "mouse lift").

        The operator lets go / repositions the leader; the robot keeps being
        commanded its frozen pose and the leader's motion is NOT accumulated, so
        nothing is owed on re-engage. Legal only from ENGAGED; from DISENGAGED it
        is a no-op (fail-safe -- clutching a void anchor must not fabricate one).
        """
        if self.state == "ENGAGED":
            self.state = "CLUTCHED"
        return {"state": self.state, "gain": self.gain}

    def reclutch(self, q_lead_now, q_cmd_now) -> dict:
        """Re-anchor at the current leader/command pose (fresh, zero delta).

        This is what COMPLETES the mouse lift: the leader's new resting position
        becomes the new origin, and because ``delta`` is reset the next step()
        returns exactly ``q_cmd_now``. Legal from ENGAGED and from CLUTCHED.
        """
        self._set_anchor(q_cmd_now, q_lead_now)
        self.state = "ENGAGED"
        return {
            "state": self.state,
            "q_robot_anchor": list(self.q_robot_anchor),
            "q_lead_anchor": list(self.q_lead_anchor),
            "gain": self.gain,
        }

    def disengage(self) -> None:
        self.state = "DISENGAGED"

    # ------------------------------------------------------------------ #
    # Step                                                                #
    # ------------------------------------------------------------------ #
    def _excursion(self) -> float:
        """max_i |q_cmd[i] - q_robot_anchor[i]| (0.0 before the first engage)."""
        if self.q_cmd is None or self.q_robot_anchor is None:
            return 0.0
        return max(
            abs(self.q_cmd[i] - self.q_robot_anchor[i]) for i in range(self.n)
        )

    def _info(self, state: str, q_lead: Optional[List[float]]) -> dict:
        """The STABLE key set returned on every path (see the module docstring
        of the node's ~/joint_delta/state diagnostic)."""
        return {
            "state": state,
            "reject_reason": None,
            "gain": self.gain,
            "delta": list(self.delta),
            "limited": [False] * self.n,
            "slewed": [False] * self.n,
            "excursion_rad": self._excursion(),
            "max_joint_step": 0.0,
            "jump_absorbed": self.jump_absorbed,
            "clamp_discarded": list(self.clamp_discarded),
            "q_lead": None if q_lead is None else list(q_lead),
            "q_robot_anchor": (
                None if self.q_robot_anchor is None else list(self.q_robot_anchor)
            ),
            "q_lead_anchor": (
                None if self.q_lead_anchor is None else list(self.q_lead_anchor)
            ),
        }

    def _hold(self, info: dict, reason: str):
        """Freeze: return the last command, mutate NOTHING (delta / anchors /
        q_cmd all stay exactly as they were), mirroring eef_delta's frozen
        T_cmd."""
        info["state"] = "HOLD"
        info["reject_reason"] = reason
        return (list(self.q_cmd) if self.q_cmd is not None else None), info

    def step(self, q_lead, step_eff: float):
        """Advance one control tick.

        Returns ``(q_cmd, info)``. On any acceptance failure the arm HOLDs:
        q_cmd == the previous command, the accumulator is frozen, and
        ``info['reject_reason']`` names the cause. ``info['state']`` is the
        TRANSIENT tick outcome (``HOLD``); ``self.state`` stays ``ENGAGED``.

        Before the first engage() q_cmd is None: a controller that has never
        engaged returns ``(None, info)`` rather than a zero-pose vector.
        """
        n = self.n
        q_in = _as_float_list(q_lead, n)

        # ---- DISENGAGED / CLUTCHED: output frozen, accumulator frozen ---- #
        # q_lead_prev is deliberately NOT advanced here: reclutch() resets it
        # anyway, and disengage() invalidates the anchor entirely, so advancing
        # it would only create a way to silently carry leader motion across a
        # release.
        if self.state != "ENGAGED":
            info = self._info(self.state, q_in)
            return (list(self.q_cmd) if self.q_cmd is not None else None), info

        info = self._info("ENGAGED", q_in)

        # ---- A1: sanitize non-finite inputs BEFORE any arithmetic ---- #
        try:
            step_eff = float(step_eff)
        except (TypeError, ValueError):
            return self._hold(info, REJECT_BAD_INPUT)
        if q_in is None or not all(math.isfinite(v) for v in q_in):
            return self._hold(info, REJECT_BAD_INPUT)
        if not math.isfinite(step_eff):
            return self._hold(info, REJECT_BAD_INPUT)

        # ---- A2: a non-positive step budget is a HARD STOP. It must never
        #      reach (and so bypass) the slew clamp below. NOTE this is a MODULE
        #      guard, not a node feature: the bridge's soft-start ramp is
        #      max_step_rad*(0.15 + 0.85*max(0, frac)), which is strictly
        #      positive for any positive max_step_rad, so REJECT_ESTOP is
        #      unreachable from gello_ur_bridge_node as shipped. It stays because
        #      this is a public, independently testable controller and a caller
        #      that CAN pass 0.0 (a replay harness, a future e-stop hook) must
        #      freeze rather than divide the arm loose from the slew clamp. ----
        if step_eff <= 0.0:
            return self._hold(info, REJECT_ESTOP)

        # ---- per-tick, wrap-guarded leader increment (the 2*pi guard) ---- #
        d = [wrap_to_pi(q_in[i] - self.q_lead_prev[i]) for i in range(n)]

        # ---- jump guard (BACKSTOP ONLY -- the primary teleport detector is the
        #      node's RAW-domain one feeding absorb(); see the module docstring
        #      and absorb() for why a guard on the FILTERED stream cannot be the
        #      primary): a single-tick increment this large is a teleport (USB
        #      glitch, encoder multi-turn re-read, a dropped burst of samples),
        #      not a human gesture. ABSORB it -- shift the mapping by updating
        #      q_lead_prev without accumulating -- rather than latching it:
        #      latching would deadlock, because the very next tick would measure
        #      the same (or a larger) gap and reject again forever. ----
        if max(abs(v) for v in d) > self.leader_jump_max:
            self.q_lead_prev = list(q_in)
            self.jump_absorbed += 1
            info["jump_absorbed"] = self.jump_absorbed
            return self._hold(info, REJECT_LEADER_JUMP)

        # ---- deadband (OFF by default; see the module docstring) ---- #
        if self.delta_deadband > 0.0:
            d = [0.0 if abs(v) < self.delta_deadband else v for v in d]

        limited = [False] * n
        slewed = [False] * n
        q_out: List[float] = []
        for i in range(n):
            # ---- accumulate + map ---- #
            self.delta[i] += d[i]
            target = self.q_robot_anchor[i] + self.gain * self.delta[i]

            # ---- position clamp (joint limit AND excursion cage) with
            #      anti-windup back-projection into the accumulator ---- #
            lo = max(
                self.joint_limits[i][0] + self.limit_margin,
                self.q_robot_anchor[i] - self.max_excursion,
            )
            hi = min(
                self.joint_limits[i][1] - self.limit_margin,
                self.q_robot_anchor[i] + self.max_excursion,
            )
            q = target
            if q < lo:
                q = lo
            elif q > hi:
                q = hi
            if q != target:
                limited[i] = True
                if self.gain > 0.0:
                    # Back-project so the operator's RETURN stroke tracks on the
                    # very next tick instead of through a windup dead zone.
                    new_delta = (q - self.q_robot_anchor[i]) / self.gain
                    # ACCOUNT for what that throws away: this is leader travel
                    # the arm will never get back, i.e. the exact amount by which
                    # the leader<->arm correspondence just shifted. Silent before;
                    # now published on ~/joint_delta/state.
                    self.clamp_discarded[i] += abs(self.delta[i] - new_delta)
                    self.delta[i] = new_delta

            # ---- slew clamp: SAME arithmetic and branch order as
            #      bridge_stages.clamp_stage, so the downstream clamp in the
            #      pipeline is a no-op on our output. NO back-projection here:
            #      slew binding is transient lag that must be caught up. ---- #
            dq = q - self.q_cmd[i]
            if dq > step_eff:
                dq = step_eff
                slewed[i] = True
            elif dq < -step_eff:
                dq = -step_eff
                slewed[i] = True
            q_out.append(self.q_cmd[i] + dq)

        max_joint_step = max(abs(q_out[i] - self.q_cmd[i]) for i in range(n))

        # ---- commit ---- #
        self.q_lead_prev = list(q_in)
        self.q_cmd = q_out
        info["state"] = "ENGAGED"
        info["delta"] = list(self.delta)
        info["limited"] = limited
        info["slewed"] = slewed
        info["excursion_rad"] = self._excursion()
        info["max_joint_step"] = max_joint_step
        info["clamp_discarded"] = list(self.clamp_discarded)
        return list(q_out), info

    def absorb(self, q_lead):
        """HOLD the output and RE-SYNC the accumulator reference to ``q_lead``.

        The caller-driven half of the teleport guard. ``step()``'s own
        ``leader_jump_max_rad`` test is a backstop only, because in the bridge
        ``step()`` is fed the 1-EURO-FILTERED leader: a low-pass smears a raw
        teleport over many ticks, so the practical glitch band (<= ~1.2 rad raw,
        with the shipped filter gains) never produces a single filtered increment
        above the cap and is accumulated in full, while a bigger glitch trips the
        cap only for the first few ticks of the smear and then has its tail
        accumulated -- an ASYMMETRIC discard that leaves a permanent offset.

        So the node detects teleports on the RAW leader stream at the SOURCE
        cadence and, for a short window afterwards, calls THIS instead of
        ``step()`` on every tick. Each call freezes the command and slides
        ``q_lead_prev`` to the current (still-settling) filtered value, so
        NOTHING of the glitch -- head or tail -- is ever accumulated, and at the
        end of the window the mapping is anchored to wherever the filter landed.
        No residual, symmetric by construction.

        Returns the same ``(q_cmd, info)`` shape as :meth:`step`, with
        ``info['state'] == 'HOLD'`` and ``reject_reason == 'LEADER_RESYNC'``, so
        the node's HOLD-latch watchdog still fires if a leader keeps teleporting
        (fail closed) and the diagnostic topic names the real cause.

        Costs one window's worth of genuine leader motion on ALL joints, exactly
        like ``step()``'s own guard: bounded, in the safe direction (the arm
        lags, it never lurches), and only reachable from a genuine raw teleport.
        """
        q_in = _as_float_list(q_lead, self.n)
        if self.state != "ENGAGED":
            info = self._info(self.state, q_in)
            return (list(self.q_cmd) if self.q_cmd is not None else None), info
        info = self._info("ENGAGED", q_in)
        if q_in is None or not all(math.isfinite(v) for v in q_in):
            # Never slide the reference onto garbage.
            return self._hold(info, REJECT_BAD_INPUT)
        self.q_lead_prev = list(q_in)
        return self._hold(info, REJECT_LEADER_RESYNC)
