"""Pose-independence ("3D pen") tests for ur_gello_bringup.eef_delta.

Sibling of ``test_eef_delta.py`` (same fixtures/naming/tolerance style); this
module tests ONE property that ``test_eef_delta.py`` never exercises, because
every case there engages with ``q_lead == q_robot``:

    THE PROPERTY (P).  For any valid leader configuration ``q_lead`` and any
    valid robot configuration ``q_robot`` -- *arbitrarily different* -- after
    ``engage(q_robot, q_lead)`` a leader EEF displacement ``d`` (translation
    and/or rotation, expressed in the UR base frame) produces a robot EEF
    displacement of the SAME magnitude and the SAME base-frame direction, to
    within tolerance, INDEPENDENT of how different ``q_lead`` and ``q_robot``
    are.  GELLO is a 3D pen: only the delta crosses over, never the shape.

Letters map to the sub-claims:

  (P1) zero-jump at engage for deliberately MISMATCHED pairs (quantified).
  (P2) translation delta reproduced in magnitude AND base-frame direction.
  (P3) rotation delta reproduced in angle AND base-frame axis.
  (P4) leader-invariance: one robot pose + one world delta, 24 leader poses
       spanning |dq| 0 -> 7 rad -> byte-for-byte-equal robot commands.
  (P5) robot-invariance: one leader delta, many robot poses -> identical robot
       EEF displacement (while the JOINTS differ completely).
  (P6) translation/rotation decoupling (no cross-talk either way).
  (P7) no mirroring: the leader/robot joint-space mismatch does NOT shrink.
  (P8) the SAME property under the real rate-limited config over many ticks.
  (P9) boundary map I  -- per-tick Cartesian increment vs the 9 gates.
  (P10) boundary map II -- reject rate tracks ROBOT conditioning, not mismatch.
  (P11) documented failure: the arm can dead-stop mid-delta at a singularity
        it drives itself into (see the FINDING notes in that test).

Tolerance style follows the existing suite: exact-ish claims at 1e-9, and
"materially different" guards at coarse thresholds so a regression cannot be
hidden by a tighter tolerance.
"""

import math

import numpy as np
import pytest

from ur_gello_bringup.eef_delta import EefDeltaController
from ur_gello_bringup.ur_kin import (
    fk,
    ik_numeric,
    se3_exp,
    se3_log,
    sigma_min,
    so3_exp,
    so3_log,
    within_joint_limits,
)


# --------------------------------------------------------------------------- #
# Helpers / fixtures                                                           #
# --------------------------------------------------------------------------- #
def _inv(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def _rand_q(rng):
    """Same convention as test_eef_delta._rand_q (elbow kept inside +/-pi)."""
    q = rng.uniform(-2.5, 2.5, 6)
    q[2] = rng.uniform(-2.9, 2.9)
    return q


def _sample_q(rng, sigma_floor=0.15, tries=5000):
    """A *well-conditioned, non-singular, in-limits* configuration.

    Rejection-samples until sigma_min is above `sigma_floor`, the joints have
    0.3 rad of limit headroom, and the TCP is clear of the pedestal/floor -- so
    that a rejected step in these tests can never be blamed on a degenerate
    sample.  Deterministic given `rng`."""
    for _ in range(tries):
        q = _rand_q(rng)
        if sigma_min(q) <= sigma_floor or not within_joint_limits(q, 0.3):
            continue
        T = fk(q)
        if T[2, 3] > 0.10 and math.hypot(T[0, 3], T[1, 3]) > 0.25:
            return q
    raise AssertionError("could not sample a well-conditioned configuration")


# Governor/limiter deliberately disabled: (P2)-(P7) are about the *mapping*
# leader-delta -> robot-delta, so the rate limiter, anti-windup and excursion
# cap are opened up and the whole delta is applied in one tick.  (P8) re-runs
# the same property through the REAL default config instead.
_FREE_CFG = dict(
    dt=1.0,
    v_max=1e6,
    w_max=1e6,
    lag_max_pose=[100.0, 100.0],
    max_excursion_m=10.0,
)


def _free_cfg(**over):
    cfg = dict(_FREE_CFG)
    cfg.update(over)
    return cfg


def _leader_q_for_translation(q_lead, d):
    """Leader joints whose TCP is `q_lead`'s TCP translated by `d` (base frame),
    orientation unchanged.  None if the leader cannot reach it."""
    T = fk(q_lead)
    T_t = T.copy()
    T_t[:3, 3] = T[:3, 3] + np.asarray(d, dtype=float)
    return ik_numeric(T_t, q_lead, max_iter=100, tol=1e-11)


def _leader_q_for_rotation(q_lead, w):
    """Leader joints whose TCP is `q_lead`'s TCP rotated by the WORLD-frame
    axis-angle `w`, position unchanged.  None if unreachable."""
    T = fk(q_lead)
    T_t = T.copy()
    T_t[:3, :3] = so3_exp(np.asarray(w, dtype=float)) @ T[:3, :3]
    return ik_numeric(T_t, q_lead, max_iter=100, tol=1e-11)


def _tcp(c, q):
    """Robot TCP pose for joints `q` under controller `c`'s tool transform."""
    return fk(q) @ c.T_tool_R


def _robot_delta(c, q_cmd):
    """(translation, world-frame axis-angle) of the robot TCP vs its anchor."""
    T = _tcp(c, q_cmd)
    return T[:3, 3] - c.p_r_anchor, so3_log(T[:3, :3] @ c.R_r_anchor.T)


def _settle(c, q_lead_target, step_eff, n_ticks, stable=40):
    """Hold the leader still at `q_lead_target` and let the governor converge.

    Stops early once the command has been bit-for-bit unchanged for `stable`
    consecutive ticks (`stable=0` disables the early exit and always runs the
    full `n_ticks`).  A permanently-HOLDing controller also settles, so the
    early exit is safe for stall cases too -- the accumulated reject reasons
    are returned either way.

    Returns (q_cmd, {reject_reason: count})."""
    reasons = {}
    q_cmd = None
    unchanged = 0
    for _ in range(n_ticks):
        q_prev = q_cmd
        q_cmd, info = c.step(q_lead_target, step_eff=step_eff)
        rr = info["reject_reason"]
        if rr:
            reasons[rr] = reasons.get(rr, 0) + 1
        if stable:
            unchanged = (
                unchanged + 1
                if q_prev is not None and np.array_equal(q_prev, q_cmd)
                else 0
            )
            if unchanged >= stable:
                break
    return q_cmd, reasons


# Deterministic mismatched (q_robot, q_lead) pairs.  Half of them walk a
# controlled mismatch ladder away from q_robot (0 rad -> ~4 rad, i.e. the leader
# is progressively LESS like the robot), half are drawn fully independently so
# the leader is in no way related to the robot.  Seeded -> reproducible.
def _mismatched_pairs(seed=20260722, n_independent=12):
    rng = np.random.default_rng(seed)
    pairs = []
    q_robot = _sample_q(rng)
    direction = rng.normal(size=6)
    direction /= np.linalg.norm(direction)
    for scale in (0.0, 0.02, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0):
        q_lead = q_robot + scale * direction
        q_lead[2] = math.remainder(q_lead[2], 2 * math.pi)  # elbow stays +/-pi
        pairs.append((q_robot.copy(), q_lead))
    for _ in range(n_independent):
        pairs.append((_sample_q(rng), _sample_q(rng)))
    return pairs


PAIRS = _mismatched_pairs()


def _mismatch(q_robot, q_lead):
    """(joint-space, TCP-position, TCP-orientation) mismatch of a pair."""
    T_r, T_l = fk(q_robot), fk(q_lead)
    return (
        float(np.linalg.norm(q_lead - q_robot)),
        float(np.linalg.norm(T_l[:3, 3] - T_r[:3, 3])),
        float(np.linalg.norm(so3_log(T_l[:3, :3] @ T_r[:3, :3].T))),
    )


def test_pairs_are_genuinely_mismatched():
    """Guard: the shared PAIRS set really does span small -> huge mismatch, so
    the tests below are not accidentally re-testing q_lead == q_robot."""
    dq = [_mismatch(a, b)[0] for a, b in PAIRS]
    dp = [_mismatch(a, b)[1] for a, b in PAIRS]
    dr = [_mismatch(a, b)[2] for a, b in PAIRS]
    assert min(dq) == pytest.approx(0.0, abs=1e-12)  # the coincident rung
    assert max(dq) > 4.0, f"largest joint mismatch only {max(dq)} rad"
    assert max(dp) > 0.5, f"largest TCP-position mismatch only {max(dp)} m"
    assert max(dr) > 2.0, f"largest TCP-orientation mismatch only {max(dr)} rad"


# --------------------------------------------------------------------------- #
# (P1) zero-jump at engage, for MISMATCHED pairs                               #
# --------------------------------------------------------------------------- #
def test_p1_zero_jump_at_engage_under_mismatch():
    """engage(q_robot, q_lead) with q_lead arbitrarily unlike q_robot, then a
    first tick at the SAME leader value, must leave the robot bit-for-bit put.

    test_eef_delta.test_a covers this only for random pairs at default cfg; the
    point here is that the *anchor* absorbs the whole leader/robot disagreement,
    so the size of the mismatch never leaks into the first command."""
    worst_q = 0.0
    worst_p = 0.0
    worst_r = 0.0
    for q_robot, q_lead in PAIRS:
        c = EefDeltaController(_free_cfg())
        c.engage(q_robot, q_lead)
        q_cmd, info = c.step(q_lead, step_eff=100.0)
        assert info["state"] == "ENGAGED", info["reject_reason"]
        worst_q = max(worst_q, float(np.max(np.abs(q_cmd - q_robot))))
        dp, dw = _robot_delta(c, q_cmd)
        worst_p = max(worst_p, float(np.linalg.norm(dp)))
        worst_r = max(worst_r, float(np.linalg.norm(dw)))
    # Exact, not merely small: the zero-delta shortcut must hold the pose.
    assert worst_q == 0.0, f"engage jump {worst_q} rad"
    assert worst_p < 1e-12, f"engage TCP position jump {worst_p} m"
    assert worst_r < 1e-12, f"engage TCP orientation jump {worst_r} rad"


def test_p1b_zero_jump_holds_for_a_still_leader():
    """A leader that is merely *still* (not bit-identical) after engage must
    likewise not walk the robot: 200 ticks at the anchor value, mismatched."""
    for q_robot, q_lead in PAIRS[:8]:
        c = EefDeltaController({})  # REAL defaults, real step budget
        c.engage(q_robot, q_lead)
        q_cmd, reasons = _settle(c, q_lead.copy(), step_eff=0.0025, n_ticks=200)
        assert reasons == {}, reasons
        assert float(np.max(np.abs(q_cmd - q_robot))) == 0.0


# --------------------------------------------------------------------------- #
# (P2) translation delta: magnitude AND base-frame direction                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "d",
    [
        np.array([0.005, 0.0, 0.0]),
        np.array([0.0, -0.004, 0.0]),
        np.array([0.0, 0.0, 0.006]),
        np.array([0.003, -0.002, 0.0035]),
    ],
)
def test_p2_translation_delta_is_pose_independent(d):
    """Robot TCP displacement == leader TCP displacement, as a base-frame
    VECTOR (so both magnitude and direction), for every mismatched pair."""
    worst = 0.0
    n = 0
    for q_robot, q_lead in PAIRS:
        q_lead_t = _leader_q_for_translation(q_lead, d)
        assert q_lead_t is not None, "leader IK failed -- bad test sample"
        c = EefDeltaController(_free_cfg())
        c.engage(q_robot, q_lead)
        q_cmd, info = c.step(q_lead_t, step_eff=100.0)
        assert info["state"] == "ENGAGED", (
            f"rejected {info['reject_reason']} at mismatch {_mismatch(q_robot, q_lead)}"
        )
        dp, _ = _robot_delta(c, q_cmd)
        worst = max(worst, float(np.linalg.norm(dp - d)))
        n += 1
    assert n == len(PAIRS)
    assert worst < 1e-9, f"translation delta error {worst} m (tol 1e-9)"


def test_p2b_translation_error_does_not_grow_with_mismatch():
    """The residual must show NO trend against the size of the mismatch: a
    controller that leaked the leader's shape would degrade as |dq| grows."""
    d = np.array([0.004, 0.003, -0.002])
    rows = []
    for q_robot, q_lead in PAIRS:
        q_lead_t = _leader_q_for_translation(q_lead, d)
        c = EefDeltaController(_free_cfg())
        c.engage(q_robot, q_lead)
        q_cmd, info = c.step(q_lead_t, step_eff=100.0)
        assert info["state"] == "ENGAGED", info["reject_reason"]
        dp, _ = _robot_delta(c, q_cmd)
        rows.append((_mismatch(q_robot, q_lead)[0], float(np.linalg.norm(dp - d))))
    lo = [e for m, e in rows if m < 1.0]
    hi = [e for m, e in rows if m > 3.0]
    assert lo and hi, "need both near-coincident and far-mismatched samples"
    # Both regimes are at float-noise level; the far one is not systematically
    # worse (a 1e5 headroom, so this cannot fail on numerical luck alone).
    assert max(hi) < 1e-9 and max(lo) < 1e-9
    assert max(hi) < max(lo) + 1e-9


# --------------------------------------------------------------------------- #
# (P3) rotation delta: angle AND base-frame axis                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "w",
    [
        np.array([0.02, 0.0, 0.0]),
        np.array([0.0, 0.015, 0.0]),
        np.array([0.0, 0.0, -0.025]),
        np.array([0.01, -0.015, 0.008]),
    ],
)
def test_p3_rotation_delta_is_pose_independent(w):
    """The robot TCP rotates by the same world-frame axis-angle as the leader,
    for every mismatched pair -- the rotation delta is applied in the WORLD
    frame (see the module docstring of eef_delta), so it must not depend on
    either arm's orientation."""
    worst_vec = 0.0
    worst_ang = 0.0
    for q_robot, q_lead in PAIRS:
        q_lead_t = _leader_q_for_rotation(q_lead, w)
        assert q_lead_t is not None, "leader IK failed -- bad test sample"
        c = EefDeltaController(_free_cfg())
        c.engage(q_robot, q_lead)
        q_cmd, info = c.step(q_lead_t, step_eff=100.0)
        assert info["state"] == "ENGAGED", (
            f"rejected {info['reject_reason']} at mismatch {_mismatch(q_robot, q_lead)}"
        )
        _, dw = _robot_delta(c, q_cmd)
        worst_vec = max(worst_vec, float(np.linalg.norm(dw - w)))
        worst_ang = max(
            worst_ang,
            abs(float(np.linalg.norm(dw)) - float(np.linalg.norm(w))),
        )
    assert worst_ang < 1e-9, f"rotation magnitude error {worst_ang} rad"
    assert worst_vec < 1e-9, f"rotation axis+angle error {worst_vec} rad"


# --------------------------------------------------------------------------- #
# (P4) leader-invariance: identical robot command for every leader pose        #
# --------------------------------------------------------------------------- #
def test_p4_same_world_delta_gives_identical_command_for_every_leader():
    """The sharpest form of the property.

    ONE robot configuration, ONE world-frame delta (translation + rotation).
    Feed it through 20 completely different leader configurations.  Every one
    must yield the SAME commanded joint vector -- not merely the same TCP
    displacement.  If any of the leader's own shape leaked through, the spread
    across leaders would be non-zero."""
    rng = np.random.default_rng(4242)
    q_robot = _sample_q(rng)
    d = np.array([0.004, -0.003, 0.002])
    w = np.array([0.01, 0.008, -0.012])

    leaders = [_sample_q(rng) for _ in range(20)]
    # Plus a few that ARE close to the robot, to span the whole mismatch range.
    leaders += [q_robot + s * np.array([0.1, -0.2, 0.15, 0.3, -0.1, 0.2])
                for s in (0.0, 1.0, 5.0)]

    cmds = []
    for q_lead in leaders:
        T = fk(q_lead)
        T_t = np.eye(4)
        T_t[:3, :3] = so3_exp(w) @ T[:3, :3]
        T_t[:3, 3] = T[:3, 3] + d
        q_lead_t = ik_numeric(T_t, q_lead, max_iter=100, tol=1e-11)
        if q_lead_t is None:
            continue  # this leader physically cannot make that motion
        c = EefDeltaController(_free_cfg())
        c.engage(q_robot, q_lead)
        q_cmd, info = c.step(q_lead_t, step_eff=100.0)
        assert info["state"] == "ENGAGED", info["reject_reason"]
        cmds.append(q_cmd)

    assert len(cmds) >= 15, f"only {len(cmds)} usable leaders"
    cmds = np.asarray(cmds)
    spread = float(np.max(np.max(cmds, axis=0) - np.min(cmds, axis=0)))
    assert spread < 1e-9, f"robot command varies by {spread} rad across leaders"
    # And it did move (otherwise "all identical" would be trivially true).
    assert float(np.max(np.abs(cmds[0] - q_robot))) > 1e-3


# --------------------------------------------------------------------------- #
# (P5) robot-invariance: same EEF displacement from every robot pose           #
# --------------------------------------------------------------------------- #
def test_p5_same_eef_displacement_from_every_robot_pose():
    """ONE leader motion, 20 unrelated robot configurations.  The robot EEF
    displacement is identical everywhere; the JOINT displacements are not --
    which is exactly the 3D-pen contract (the shape is free, the tip is not)."""
    rng = np.random.default_rng(909)
    q_lead = _sample_q(rng)
    d = np.array([0.005, 0.002, -0.004])
    w = np.array([-0.01, 0.006, 0.009])
    T = fk(q_lead)
    T_t = np.eye(4)
    T_t[:3, :3] = so3_exp(w) @ T[:3, :3]
    T_t[:3, 3] = T[:3, 3] + d
    q_lead_t = ik_numeric(T_t, q_lead, max_iter=100, tol=1e-11)
    assert q_lead_t is not None

    dps, dws, dqs = [], [], []
    for _ in range(20):
        q_robot = _sample_q(rng)
        c = EefDeltaController(_free_cfg())
        c.engage(q_robot, q_lead)
        q_cmd, info = c.step(q_lead_t, step_eff=100.0)
        if info["state"] != "ENGAGED":
            continue
        dp, dw = _robot_delta(c, q_cmd)
        dps.append(dp)
        dws.append(dw)
        dqs.append(q_cmd - q_robot)

    assert len(dps) >= 15, f"only {len(dps)} robot poses accepted"
    dps, dws, dqs = np.asarray(dps), np.asarray(dws), np.asarray(dqs)
    assert float(np.max(np.abs(dps - d))) < 1e-9
    assert float(np.max(np.abs(dws - w))) < 1e-9
    # The JOINT response is genuinely configuration-dependent (so the Cartesian
    # agreement above is a real result, not an artefact of identical arms).
    joint_spread = float(np.max(np.max(dqs, axis=0) - np.min(dqs, axis=0)))
    assert joint_spread > 0.01, (
        f"joint responses only span {joint_spread} rad -- robots too similar"
    )


# --------------------------------------------------------------------------- #
# (P6) translation / rotation decoupling                                       #
# --------------------------------------------------------------------------- #
def test_p6_pure_translation_induces_no_robot_rotation():
    worst = 0.0
    for q_robot, q_lead in PAIRS:
        q_lead_t = _leader_q_for_translation(q_lead, np.array([0.005, -0.003, 0.004]))
        c = EefDeltaController(_free_cfg())
        c.engage(q_robot, q_lead)
        q_cmd, info = c.step(q_lead_t, step_eff=100.0)
        assert info["state"] == "ENGAGED", info["reject_reason"]
        _, dw = _robot_delta(c, q_cmd)
        worst = max(worst, float(np.linalg.norm(dw)))
    assert worst < 1e-9, f"spurious robot rotation {worst} rad"


def test_p6b_pure_rotation_induces_no_robot_translation():
    """A leader rotation about its own TCP origin rotates the robot TCP about
    the ROBOT's TCP origin (p_des is untouched) -- pen-like, and again
    independent of either arm's configuration."""
    worst = 0.0
    for q_robot, q_lead in PAIRS:
        q_lead_t = _leader_q_for_rotation(q_lead, np.array([0.012, -0.009, 0.015]))
        c = EefDeltaController(_free_cfg())
        c.engage(q_robot, q_lead)
        q_cmd, info = c.step(q_lead_t, step_eff=100.0)
        assert info["state"] == "ENGAGED", info["reject_reason"]
        dp, _ = _robot_delta(c, q_cmd)
        worst = max(worst, float(np.linalg.norm(dp)))
    assert worst < 1e-9, f"spurious robot translation {worst} m"


# --------------------------------------------------------------------------- #
# (P7) no mirroring: the joint-space mismatch must NOT shrink                  #
# --------------------------------------------------------------------------- #
def test_p7_robot_does_not_mirror_leader_shape():
    """Drive a real delta and check the arms do not converge in joint space.

    A mirroring (joint-copy) controller would drive ||q_cmd - q_lead|| toward
    zero; a pen-like controller leaves it essentially where it was, changing it
    by no more than the robot's own (small) joint motion."""
    d = np.array([0.02, 0.01, -0.015])
    for q_robot, q_lead in PAIRS:
        if _mismatch(q_robot, q_lead)[0] < 1.0:
            continue  # a near-coincident pair says nothing about mirroring
        q_lead_t = _leader_q_for_translation(q_lead, d)
        c = EefDeltaController({})  # real defaults, rate limited
        c.engage(q_robot, q_lead)
        q_cmd, reasons = _settle(c, q_lead_t, step_eff=0.0025, n_ticks=600)
        # Both distances are measured against the SAME (final) leader pose, so
        # the comparison isolates the ROBOT's contribution.
        before = float(np.linalg.norm(q_lead_t - q_robot))
        after = float(np.linalg.norm(q_lead_t - q_cmd))
        moved = float(np.linalg.norm(q_cmd - q_robot))
        # The mismatch can only change by as much as the arm actually moved
        # (triangle inequality) -- i.e. it is not being *driven* to zero.
        assert abs(after - before) <= moved + 1e-9
        assert after > 0.5 * before, (
            f"mismatch collapsed {before} -> {after}: looks like mirroring"
        )
        assert moved > 1e-3, "robot did not move -- test would be vacuous"


# --------------------------------------------------------------------------- #
# (P8) same property under the REAL rate-limited config, over many ticks       #
# --------------------------------------------------------------------------- #
def test_p8_property_survives_the_real_governor():
    """(P2)/(P3) with EefDeltaController's shipped defaults (dt=8 ms,
    v_max=0.08 m/s, w_max=0.5 rad/s, lag 0.05 m/0.3 rad, step budget 2.5 mrad)
    and the delta accumulated over 1500 ticks instead of one free step."""
    d = np.array([0.012, -0.008, 0.010])
    w = np.array([0.03, 0.02, -0.025])
    ok = 0
    worst_p = 0.0
    worst_w = 0.0
    stalled = []
    for q_robot, q_lead in PAIRS:
        T = fk(q_lead)
        T_t = np.eye(4)
        T_t[:3, :3] = so3_exp(w) @ T[:3, :3]
        T_t[:3, 3] = T[:3, 3] + d
        q_lead_t = ik_numeric(T_t, q_lead, max_iter=100, tol=1e-11)
        if q_lead_t is None:
            continue
        c = EefDeltaController({})
        c.engage(q_robot, q_lead)
        q_cmd, reasons = _settle(c, q_lead_t, step_eff=0.0025, n_ticks=1500)
        if reasons:
            stalled.append((_mismatch(q_robot, q_lead)[0], reasons))
            continue
        dp, dw = _robot_delta(c, q_cmd)
        worst_p = max(worst_p, float(np.linalg.norm(dp - d)))
        worst_w = max(worst_w, float(np.linalg.norm(dw - w)))
        ok += 1

    assert ok >= int(0.8 * len(PAIRS)), (
        f"only {ok}/{len(PAIRS)} pairs completed cleanly; stalls: {stalled}"
    )
    assert worst_p < 1e-9, f"rate-limited translation error {worst_p} m"
    assert worst_w < 1e-9, f"rate-limited rotation error {worst_w} rad"


# --------------------------------------------------------------------------- #
# (P9) boundary map I: per-tick Cartesian increment vs the acceptance gates    #
# --------------------------------------------------------------------------- #
def test_p9_boundary_single_tick_increment():
    """Map WHERE the property stops holding as a function of the size of the
    increment demanded in ONE tick (governor off, so the whole delta is asked
    for at once).

    FINDING (documented, not asserted-as-desirable): the first gate to fire is
    BRANCH_JUMP, and it fires on a step that is on the SAME branch -- the gate
    is ``_wnorm(dq) > branch_tol`` (a weighted joint-step magnitude), so its
    reject_reason misattributes a large-but-continuous step to a branch flip.
    See test_p9b."""
    rng = np.random.default_rng(5)
    robots = [_sample_q(rng, sigma_floor=0.25) for _ in range(8)]
    q_lead = PAIRS[-1][1]
    unit = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)

    table = {}
    for mag in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        q_lead_t = _leader_q_for_translation(q_lead, mag * unit)
        if q_lead_t is None:
            continue
        tally = {}
        for q_robot in robots:
            c = EefDeltaController(_free_cfg())
            c.engage(q_robot, q_lead)
            _, info = c.step(q_lead_t, step_eff=100.0)
            key = info["reject_reason"] or "OK"
            tally[key] = tally.get(key, 0) + 1
        table[mag] = tally

    # Below ~2 cm in a single tick every well-conditioned robot accepts.
    assert table[0.005] == {"OK": len(robots)}
    assert table[0.010] == {"OK": len(robots)}
    assert table[0.020].get("OK", 0) == len(robots)
    # By 10 cm in one tick nothing is accepted any more ...
    assert table[0.10].get("OK", 0) == 0
    # ... and the gate that stops it is JOINT_JUMP (the weighted-step gate),
    # not a limit/keepout/excursion gate.  NOTE: this reason used to be called
    # BRANCH_JUMP, which was a misattribution -- the candidate is on the
    # anchor's own branch and the violation is one of SIZE.  BRANCH_JUMP now
    # means an actual branch change.  See test_p9b.
    assert table[0.10].get("JOINT_JUMP", 0) >= len(robots) - 1
    # The transition is monotone in the increment size.
    accepted = [table[m].get("OK", 0) for m in sorted(table)]
    assert accepted == sorted(accepted, reverse=True), accepted
    # Under the REAL rate limiter one tick can only ask for v_max*dt = 0.64 mm,
    # i.e. ~30x inside the boundary measured above -- which is why (P8) passes.
    assert 0.08 * 0.008 < 0.005


def test_p9b_branch_jump_reason_can_fire_without_a_branch_change():
    """FINDING (report, do NOT fix here): at the (P9) boundary the rejected
    candidate carries the SAME branch id as the anchor.  ``BRANCH_JUMP`` is
    therefore emitted for a large-but-same-branch step.  Pinned here so the
    misattribution is visible in the suite rather than surprising an operator
    reading reject_reason on the real robot."""
    from ur_gello_bringup.ur_kin import branch_id

    rng = np.random.default_rng(5)
    q_robot = _sample_q(rng, sigma_floor=0.25)
    q_lead = PAIRS[-1][1]
    unit = np.array([1.0, 0.5, -0.3])
    unit /= np.linalg.norm(unit)
    q_lead_t = _leader_q_for_translation(q_lead, 0.10 * unit)

    c = EefDeltaController(_free_cfg())
    c.engage(q_robot, q_lead)
    q_cmd, info = c.step(q_lead_t, step_eff=100.0)
    assert info["state"] == "HOLD"
    # This reason used to be reported as BRANCH_JUMP, which was a
    # misattribution: the refused candidate is on the anchor's own branch and
    # the violation is one of SIZE.  The reason is now truthfully JOINT_JUMP,
    # and BRANCH_JUMP is reserved for a genuine branch change.  The assertions
    # below are what proved the old name wrong, so they are kept as the
    # regression guard for the new one.
    assert info["reject_reason"] == "JOINT_JUMP"
    assert np.allclose(q_cmd, q_robot)  # held, no jump -- the safety part is fine

    # The candidate the gate refused is on the anchor's own branch.
    q_try, n_sol = c._ik_branchlock(info["T_des"])
    assert q_try is not None and n_sol > 0
    assert branch_id(q_try) == c.branch0, "expected a same-branch candidate"
    assert c._wnorm(q_try - q_robot) > c.branch_tol  # it is a SIZE violation


# --------------------------------------------------------------------------- #
# (P10) boundary map II: rejection tracks ROBOT conditioning, not mismatch     #
# --------------------------------------------------------------------------- #
def test_p10_rejection_tracks_robot_conditioning_not_mismatch():
    """The decisive boundary result: whether a delta is accepted depends on the
    ROBOT's own conditioning (sigma_min), and not at all on how different the
    leader is.  Sampled over sigma_min bands with a fixed 1 cm delta."""
    d = 0.01 * np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    bands = [(0.0, 0.03), (0.03, 0.10), (0.10, 0.25), (0.25, 10.0)]
    accept_rate = {}
    reasons_by_band = {}
    mismatch_of_rejects = []
    mismatch_of_accepts = []

    for lo, hi in bands:
        rng = np.random.default_rng(31 + int(lo * 1000))
        n_ok = n_tot = 0
        reasons = {}
        for _ in range(400):
            if n_tot >= 20:
                break
            q_robot = _rand_q(rng)
            s = sigma_min(q_robot)
            if not (lo < s <= hi) or not within_joint_limits(q_robot, 0.3):
                continue
            T = fk(q_robot)
            if T[2, 3] <= 0.10 or math.hypot(T[0, 3], T[1, 3]) <= 0.25:
                continue
            q_lead = _sample_q(rng)
            q_lead_t = _leader_q_for_translation(q_lead, d)
            if q_lead_t is None:
                continue
            n_tot += 1
            c = EefDeltaController(_free_cfg())
            c.engage(q_robot, q_lead)
            _, info = c.step(q_lead_t, step_eff=100.0)
            mm = _mismatch(q_robot, q_lead)[0]
            if info["state"] == "ENGAGED":
                n_ok += 1
                mismatch_of_accepts.append(mm)
            else:
                reasons[info["reject_reason"]] = reasons.get(info["reject_reason"], 0) + 1
                mismatch_of_rejects.append(mm)
        assert n_tot >= 10, f"band ({lo},{hi}] under-sampled: {n_tot}"
        accept_rate[(lo, hi)] = n_ok / n_tot
        reasons_by_band[(lo, hi)] = reasons

    # Strong, monotone-ish dependence on the ROBOT's conditioning.
    assert accept_rate[(0.25, 10.0)] == 1.0, accept_rate
    assert accept_rate[(0.10, 0.25)] >= 0.8, accept_rate
    assert accept_rate[(0.0, 0.03)] < 0.75, (
        f"near-singular band was expected to reject: {accept_rate}"
    )
    assert accept_rate[(0.0, 0.03)] < accept_rate[(0.25, 10.0)]
    # The only reasons seen at the boundary are the joint-step / branch / IK
    # gates -- never a limit, keepout or excursion gate, which is the point.
    # (JOINT_JUMP and STEP_CAP were split out of the old, overloaded
    # BRANCH_JUMP / STEP_FLOOR reasons; see test_p9b.)
    seen = set()
    for r in reasons_by_band.values():
        seen |= set(r)
    assert seen <= {
        "JOINT_JUMP",
        "BRANCH_JUMP",
        "NO_IK",
        "STEP_FLOOR",
        "STEP_CAP",
    }, seen

    # ... and essentially NO dependence on the leader/robot mismatch: rejected
    # and accepted samples come from the same mismatch distribution.
    assert mismatch_of_rejects, "no rejects collected -- boundary not reached"
    m_rej = float(np.mean(mismatch_of_rejects))
    m_ok = float(np.mean(mismatch_of_accepts))
    assert abs(m_rej - m_ok) < 1.0, (
        f"mean mismatch differs between rejects ({m_rej}) and accepts ({m_ok}) "
        "-- the mismatch would then be influencing acceptance"
    )


# --------------------------------------------------------------------------- #
# (P11) documented failure: mid-delta dead-stop at a self-inflicted singularity #
# --------------------------------------------------------------------------- #
def test_p11_documented_dead_stop_when_the_delta_crosses_a_singularity():
    """The property can FAIL -- but always for a robot-side reason, never a
    mismatch reason.  That distinction is the point of this test.

    If executing the (pose-independent, correct) T_des takes the robot through
    one of its OWN singularities, the arm stops partway and never resumes: the
    leader is holding still at a target the governor refuses forever, so the
    robot EEF displacement ends up a fraction of the leader's.

    HISTORY -- this test originally documented a DEFECT.  ``step()`` probed the
    full rate-limited increment and returned NO_IK immediately if that probe
    failed, so the halving line-search below it was never entered even when a
    smaller increment was perfectly solvable; the arm dead-stopped with feasible
    motion still available.  That is fixed (the probe now halves down to
    ``s_floor``; see test_d1 / test_d1b in test_eef_delta.py).

    What this test now pins is the RESIDUAL, legitimate stall: past ~0.8 mm the
    requested target has no IK solution on any branch at any scale the line
    search may try, i.e. it is outside the reachable workspace.  Stopping is the
    correct behaviour, and the assertions below are what distinguish it from a
    regression back to the old short-circuit.
    """
    # A configuration whose 1 cm delta drives the elbow through ~0.
    rng = np.random.default_rng(41)
    q_robot = None
    for _ in range(5000):
        q = _rand_q(rng)
        if 0.02 < sigma_min(q) <= 0.03 and within_joint_limits(q, 0.2):
            T = fk(q)
            if T[2, 3] > 0.05 and math.hypot(T[0, 3], T[1, 3]) > 0.2:
                q_robot = q
                break
    assert q_robot is not None
    q_lead = _sample_q(rng)
    d = 0.01 * np.array([0.6, 0.6, 0.5292])
    q_lead_t = _leader_q_for_translation(q_lead, d)
    assert q_lead_t is not None

    c = EefDeltaController({})
    c.engage(q_robot, q_lead)
    # stable=0: no early exit -- the point is that the stall is PERMANENT.
    q_cmd, reasons = _settle(c, q_lead_t, step_eff=0.0025, n_ticks=2000, stable=0)

    # It stalls, permanently, on a robot-side gate.
    assert reasons, "expected this configuration to stall"
    assert set(reasons) <= {"NO_IK", "STEP_FLOOR"}, reasons
    assert sum(reasons.values()) > 1000, reasons
    # And the delivered displacement really is short of the leader's.
    dp, _ = _robot_delta(c, q_cmd)
    assert float(np.linalg.norm(dp)) < 0.5 * float(np.linalg.norm(d))
    # The arm walked itself INTO the singularity (it did not start there).
    assert sigma_min(q_cmd) < sigma_min(q_robot)

    if "NO_IK" in reasons:
        # This block used to pin a DEFECT: the full-scale probe returning None
        # short-circuited straight to HOLD, so the halving loop underneath was
        # never entered even though a smaller increment was perfectly solvable.
        # That is fixed -- the probe now halves down to s_floor.
        #
        # What remains here is a GENUINE stall, and the assertions below are
        # what tell the two apart: at the stall point NO scale down to s_floor
        # has an IK solution on any branch, i.e. the requested target is simply
        # outside the reachable workspace.  Stopping is correct.  (The fix
        # itself is covered by test_d1 / test_d1b in test_eef_delta.py, which
        # assert step() advances while the full-scale probe fails.)
        _, info = c.step(q_lead_t, step_eff=0.0025)
        assert info["reject_reason"] == "NO_IK"
        assert info["n_ik_solutions"] == 0
        xi = se3_log(_inv(c.T_cmd) @ info["T_des"])
        nv = float(np.linalg.norm(xi[:3]))
        nw = float(np.linalg.norm(xi[3:]))
        scale = 1.0
        if nv > 1e-12:
            scale = min(scale, info["gamma"] * c.v_max * c.dt / nv)
        if nw > 1e-12:
            scale = min(scale, info["gamma"] * c.w_max * c.dt / nw)
        xi_lim = xi * scale
        # Unreachable at every scale the line search is allowed to try, so the
        # HOLD is geometry -- not the solver giving up early.
        s = 1.0
        while s >= c.s_floor:
            q_s, n_s = c._ik_branchlock(c.T_cmd @ se3_exp(s * xi_lim))
            assert q_s is None and n_s == 0, (
                f"scale {s} was solvable -- this is the old short-circuit "
                "defect, not a genuine end-of-reach stall"
            )
            s *= 0.5
