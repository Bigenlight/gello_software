"""Pure (rclpy-independent) command-pipeline stages for the GELLO->UR bridge.

The bridge's ``_on_timer`` used to run a single per-joint loop that INTERLEAVED
the smoothing filter and the slew clamp:

    for i in range(6):
        filtered[i] = <one_euro or ema+deadband>(raw_target[i])
        out.append(last_published[i] + clamp(filtered[i]-last_published[i], step))

To add an end-effector (EEF) delta-control path WITHOUT regressing the existing
joint-passthrough teleop, that loop is split into three ordered stages:

    (1) filter stage   -> fill ``filtered`` (per-joint, joint-mode math)
    (2) eef stage       -> ONLY when control_mode=="eef": when ENGAGED overwrite
                           ``filtered`` with the EEF controller's slew-limited
                           q_cmd (or hold on None); when NOT engaged, hold the
                           last published command iff ``hold_when_not_engaged``
                           (the 3D-pen bootstrap). A NO-OP in joint mode.
    (2b) joint_delta stage -> ONLY when control_mode=="joint_delta": the exact
                           same overwrite/hold shape as (2), fed by the
                           JointDeltaController's already-slew-limited q_cmd.
                           A NO-OP in joint AND in eef mode.
    (3) clamp stage     -> the legacy slew clamp

Because stage (1) and stage (3) touch each joint independently (no cross-joint
coupling, and the clamp reads only ``filtered[i]``/``last_published[i]``),
running "all filters, then all clamps" is **bit-identical** to the old
interleaved loop. In joint mode stage (2) is skipped, so joint teleop is proven
bit-identical (see test/test_bridge_stages.py) — the top safety requirement.

This module has NO rclpy / ROS dependency (only the stdlib ``math``) so it can
be unit-tested in a plain venv, i.e. the code that is tested IS the code the
node runs.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


class OneEuro:
    """Scalar 1-Euro filter (Casiez, Roussel & Vogel 2012) at a FIXED period.

    Adaptive low-pass tuned for teleop input jitter: it low-passes the signal
    with a cutoff that RISES with the (low-passed) signal speed. So when the
    GELLO joint is nearly still it smooths HARD (kills hand tremor + Dynamixel
    encoder/motor noise), and when you move it fast the cutoff opens up so the
    robot tracks with little lag.

    ``min_cutoff`` (Hz) sets the at-rest smoothing (LOWER = smoother / more lag
    when slow). ``beta`` sets how fast the cutoff opens with speed (HIGHER =
    snappier on fast moves). ``d_cutoff`` low-passes the internal speed estimate
    so jitter does not inflate the cutoff.

    GELLO samples arrive at ~30 Hz while the bridge publishes at 250 Hz, so the
    speed estimate is updated at the GELLO sample cadence (``update_input``) and
    NOT on every publish tick, otherwise each 30 Hz sample step would look like a
    much faster 250 Hz jump and the filter would open up in visible pulses.

    Lives here (pure module) rather than in the node so both the node and the
    venv bit-identity tests use the exact same filter implementation.
    """

    def __init__(self, dt: float, min_cutoff: float, beta: float,
                 d_cutoff: float) -> None:
        self._dt = dt
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._raw_prev: Optional[float] = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        # Smoothing factor of a 1st-order low-pass at the given cutoff (Hz).
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def seed(self, x: float) -> None:
        """Preload the filter state (e.g. from the robot's actual pose)."""
        self._x_prev = x
        self._raw_prev = x
        self._dx_prev = 0.0

    def update_input(self, x: float, dt: Optional[float]) -> None:
        """Update the low-passed input-speed estimate at the source cadence."""
        if self._raw_prev is None or dt is None or dt <= 1e-6:
            self._raw_prev = x
            return
        dx = (x - self._raw_prev) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        self._dx_prev = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._raw_prev = x

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self.seed(x)
            return x
        # Speed-adaptive cutoff: faster motion -> higher cutoff -> less lag.
        cutoff = self._min_cutoff + self._beta * abs(self._dx_prev)
        a = self._alpha(cutoff, self._dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


def filter_stage_joint(
    raw_target: Sequence[float],
    filtered: List[float],
    gated_target: List[float],
    euro: Optional[Sequence] ,
    ema_alpha: float,
    deadband: float,
) -> List[float]:
    """Per-joint smoothing filter — the EXACT arithmetic of the legacy loop.

    Mutates ``filtered`` (and ``gated_target`` on the EMA path) IN PLACE, exactly
    as the old interleaved loop did, and returns ``filtered``.

    * ``euro`` not None  -> 1-Euro path: ``filtered[i] = euro[i](raw_target[i])``.
      Each ``euro[i]`` is a stateful callable (an :class:`OneEuro`); calling it
      advances its internal state, so it must be called exactly ONCE per joint
      per tick — same as the reference loop.
    * ``euro`` is None   -> EMA + deadband path: a joint's held ``gated_target``
      only updates when the raw target moved more than ``deadband``; then the
      output is an EMA toward that gated value.

    No cross-joint coupling: each joint is a function of its own state only, so
    calling this for all joints then :func:`clamp_stage` for all joints is
    bit-identical to interleaving the two per joint.
    """
    use_euro = euro is not None
    n = len(raw_target)
    for i in range(n):
        if use_euro:
            filtered[i] = euro[i](raw_target[i])
        else:
            if abs(raw_target[i] - gated_target[i]) > deadband:
                gated_target[i] = raw_target[i]
            filtered[i] = (1.0 - ema_alpha) * filtered[i] + ema_alpha * gated_target[i]
    return filtered


def clamp_stage(
    filtered: Sequence[float],
    last_published: Sequence[float],
    step: float,
) -> List[float]:
    """Per-joint slew (max-step) clamp relative to the last PUBLISHED value.

    ``out[i] = last_published[i] + clip(filtered[i] - last_published[i], +-step)``.
    Byte-for-byte the legacy clamp; branch order (``>`` then ``elif <``) is
    preserved so the boundary rounding is identical.
    """
    out: List[float] = []
    n = len(filtered)
    for i in range(n):
        delta = filtered[i] - last_published[i]
        if delta > step:
            delta = step
        elif delta < -step:
            delta = -step
        out.append(last_published[i] + delta)
    return out


def command_pipeline(
    control_mode: str,
    engaged: bool,
    raw_target: Sequence[float],
    filtered: List[float],
    gated_target: List[float],
    euro: Optional[Sequence],
    ema_alpha: float,
    deadband: float,
    last_published: Sequence[float],
    step: float,
    eef_command: Optional[Sequence[float]] = None,
    hold_when_not_engaged: bool = False,
    joint_delta_command: Optional[Sequence[float]] = None,
) -> List[float]:
    """Run the full three-stage command pipeline for one publish tick.

    Stage 1 (filter) ALWAYS runs (fills ``filtered`` with the joint-mode math).
    Stage 2 (eef) runs ONLY when ``control_mode == "eef"``:
      * ``engaged`` and ``eef_command`` not None -> OVERWRITE ``filtered`` with
        the EEF controller's already-slew-limited q_cmd; the downstream clamp
        then does not bind (``step`` is passed to the EEF controller too).
      * ``engaged`` and ``eef_command`` is None  -> HOLD: set ``filtered`` to
        ``last_published`` so the clamp delta is exactly zero (arm freezes).
      * NOT ``engaged`` and ``hold_when_not_engaged`` -> HOLD as well. This is
        the "3D pen" bootstrap: the leader and the robot deliberately live in
        DIFFERENT joint configurations, so mirroring the leader's joints before
        the clutch is engaged would sweep the arm across the cell. With this
        flag the pre-engage state is a genuine HOLD, not a joint passthrough.
      * NOT ``engaged`` and NOT ``hold_when_not_engaged`` -> joint passthrough
        (the legacy eef BOOTSTRAP behaviour, still used after ``~/eef_to_joint``
        so the operator can re-align the leader and hand back to joint teleop).
    Stage 2b (joint_delta) runs ONLY when ``control_mode == "joint_delta"``, as
    an ``elif`` AFTER stage 2 so the eef branch's evaluated instructions are
    byte-identical to before this mode existed:
      * ``engaged`` and ``joint_delta_command`` not None -> OVERWRITE
        ``filtered`` with the JointDeltaController's already-slew-limited q_cmd.
        ``engaged`` is True for BOTH its ENGAGED and CLUTCHED node states: in
        CLUTCHED the controller returns its frozen command, which is exactly the
        hold the clutch is supposed to produce.
      * ``engaged`` and ``joint_delta_command`` is None -> HOLD.
      * NOT ``engaged`` (JOINT_BOOTSTRAP, i.e. before ``~/joint_delta_engage``)
        -> joint passthrough. Unlike eef mode there is no "3D pen" framing here:
        the leader and the robot ARE in the same joint configuration before the
        anchor is latched (joint_delta mode boots through the normal
        alignment-gated handshake), so passthrough is the honest bootstrap.
    Stage 3 (clamp) ALWAYS runs.

    ``hold_when_not_engaged`` DEFAULTS TO FALSE, and stages 2/2b are skipped
    entirely in joint mode, so the joint path is untouched: the result equals
    ``clamp_stage(filter_stage_joint(...), last_published, step)`` — bit-identical
    to the legacy interleaved loop regardless of any ``eef_command`` /
    ``joint_delta_command`` argument.

    NOTE stage 1 keeps running even while holding, on purpose: the joint filter
    bank (``euro`` / ``filtered`` / ``gated_target``) must stay live so a later
    return to joint passthrough starts from a current filter state rather than a
    stale one. Its output is simply discarded while stage 2 holds.
    """
    filter_stage_joint(raw_target, filtered, gated_target, euro, ema_alpha, deadband)
    if control_mode == "eef":
        if engaged:
            if eef_command is not None:
                for i in range(len(filtered)):
                    filtered[i] = float(eef_command[i])
            else:
                for i in range(len(filtered)):
                    filtered[i] = last_published[i]
        elif hold_when_not_engaged:
            for i in range(len(filtered)):
                filtered[i] = last_published[i]
    elif control_mode == "joint_delta":
        if engaged:
            if joint_delta_command is not None:
                for i in range(len(filtered)):
                    filtered[i] = float(joint_delta_command[i])
            else:
                for i in range(len(filtered)):
                    filtered[i] = last_published[i]
    return clamp_stage(filtered, last_published, step)
