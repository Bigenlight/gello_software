"""Property tests for the START-ANCHORED JOINT-DELTA controller.

Each test is lettered and asserts ONE precise numeric invariant of
``ur_gello_bringup.joint_delta.JointDeltaController``:

  (a) ZERO JUMP AT ENGAGE — the first step() after engage with an unmoved
      leader returns q_robot_anchor EXACTLY (algebra, not a tolerance).
  (b) LINEARITY — with the clamps out of the way the command is exactly
      q_robot_anchor + gain*(leader travel), for several gains.
  (c) ACCUMULATION PAST PI (regression) — a >pi leader excursion accumulates
      monotonically, and the naive "wrap against the anchor" formula is shown
      to DIVERGE from it. This is the single most important property in the
      module: the naive form folds and would drive the arm backwards.
  (d) BRANCH CUT — a leader anchored at +3.10 stepping across to -3.10 produces
      small, continuous commands and a total delta of ~+0.083, never ~+-2pi.
  (e) CLUTCH freezes the output while the leader keeps moving.
  (f) RECLUTCH re-anchors at zero delta with no jump.
  (g) DISENGAGE freezes; a never-engaged controller returns (None, info).
  (h) JOINT-LIMIT SATURATION — the elbow saturates at +pi (NOT +2pi).
  (i) ANTI-WINDUP — after long saturation the return stroke tracks on the very
      next tick, with no dead zone.
  (j) EXCURSION CAGE — the command never leaves anchor +- max_excursion_rad.
  (k) SLEW — the controller's own clamp makes bridge_stages.clamp_stage a no-op.
  (l) ESTOP / BAD_INPUT — HOLD with the right reason, state stays ENGAGED.
  (m) LEADER JUMP — absorbed, no motion, no residual offset afterwards.
  (n) GAIN 0.0 — the arm never moves (the staged bring-up's guarantee).
  (o) INFO SHAPE — the same key set on every path.
  (p) DEFAULT LIMITS — identical to ur_kin.JOINT_LIMITS (anti copy-drift).
  (q) CFG VALIDATION — bad config raises ValueError at construction.
  (r) NO DRIFT — out and back over thousands of noisy ticks returns exactly to
      the anchor (the telescoping property that justifies deadband = 0).
  (s) CLAMP SATURATION SHIFTS THE MAPPING — the documented PRICE of (i)'s
      anti-windup: once a position clamp has bound, out-and-back does NOT return
      to the anchor. Pins the exact displacement and the clamp_discarded
      accounting that makes it visible.
  (t) ABSORB — the caller-driven resync: HOLDs, slides q_lead_prev, accumulates
      nothing, and leaves no residual once stepping resumes.
  (u) RESYNC WINDOW END-TO-END — a raw leader teleport pushed through the REAL
      1-Euro filter at the real cadences moves the arm not at all when the node's
      resync window is honoured, and moves it a long way when it is not.
  (v) RE-ANCHOR CLEARS THE DISCARD LEDGER — clutch/reclutch is the documented
      remedy for (s), so it must actually reset the accounting.

Pure module — no rclpy — so it runs in a plain venv (the code tested IS the code
the node runs).
"""

import math
import random

import pytest

from ur_gello_bringup.angle_utils import wrap_to_pi
from ur_gello_bringup.bridge_stages import OneEuro, clamp_stage
from ur_gello_bringup.joint_delta import (
    REJECT_BAD_INPUT,
    REJECT_ESTOP,
    REJECT_LEADER_JUMP,
    REJECT_LEADER_RESYNC,
    JointDeltaController,
)

N = 6

# A generic, limit-free-ish anchor pair used by most tests.
Q_ROBOT = [0.10, -1.20, 1.30, -1.60, -1.55, 0.20]
Q_LEAD = [0.50, -0.90, 0.70, -1.10, -1.40, 3.05]

# "Get out of the way" config: huge cage, no jump cap in practice.
WIDE = {"max_excursion_rad": 100.0, "leader_jump_max_rad": 100.0}

BIG_STEP = 1e6  # slew clamp effectively disabled


def _engaged(cfg=None, q_robot=None, q_lead=None):
    c = JointDeltaController(dict(WIDE, **(cfg or {})))
    c.engage(list(q_robot or Q_ROBOT), list(q_lead or Q_LEAD))
    return c


# --------------------------------------------------------------------------- #
# (a) zero jump at engage                                                      #
# --------------------------------------------------------------------------- #
def test_a_zero_jump_at_engage():
    c = _engaged()
    q, info = c.step(list(Q_LEAD), step_eff=1.0)
    assert info["state"] == "ENGAGED"
    worst = max(abs(q[i] - Q_ROBOT[i]) for i in range(N))
    assert worst < 1e-12, f"engage is not zero-jump: worst={worst}"
    assert q == Q_ROBOT  # exact, not merely close


# --------------------------------------------------------------------------- #
# (b) linearity                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gain", [0.25, 0.5, 1.0])
def test_b_linearity_matches_formula(gain):
    """q_cmd == q_robot_anchor + gain*(q_lead - q_lead_anchor) on a smooth,
    small-increment trajectory (no wrap involved, clamps out of the way)."""
    c = _engaged({"gain": gain})
    rng = random.Random(1234 + int(gain * 1000))
    q_lead = list(Q_LEAD)
    for _ in range(200):
        for i in range(N):
            q_lead[i] += rng.uniform(-0.01, 0.01)
        q, info = c.step(list(q_lead), step_eff=BIG_STEP)
        for i in range(N):
            expect = Q_ROBOT[i] + gain * (q_lead[i] - Q_LEAD[i])
            assert abs(q[i] - expect) < 1e-9, f"joint {i}: {q[i]} != {expect}"
        assert not any(info["limited"])
        assert not any(info["slewed"])


# --------------------------------------------------------------------------- #
# (c) accumulation past pi (THE regression guard for the wrap convention)      #
# --------------------------------------------------------------------------- #
def test_c_accumulation_beyond_pi_regression():
    """A +1.6*pi wrist_3 excursion must accumulate MONOTONICALLY, and the naive
    'wrap the total against the anchor' formula must be shown to diverge.

    The naive form folds at +-pi: at +1.6*pi it reports -0.4*pi, i.e. it would
    command the arm BACKWARDS through nearly half a revolution.
    """
    c = _engaged()
    total = 1.6 * math.pi
    ticks = 400
    inc = total / ticks
    q_lead = list(Q_LEAD)
    prev_delta = 0.0
    for _ in range(ticks):
        q_lead[5] += inc
        q, info = c.step(list(q_lead), step_eff=BIG_STEP)
        assert info["delta"][5] >= prev_delta - 1e-12, "delta folded / went backwards"
        prev_delta = info["delta"][5]

    assert abs(c.delta[5] - total) < 1e-9, f"delta[5]={c.delta[5]} != {total}"
    assert abs(c.q_cmd[5] - (Q_ROBOT[5] + total)) < 1e-9

    # The naive absolute-wrap formula, hand-computed, and the divergence.
    naive_delta = wrap_to_pi(q_lead[5] - Q_LEAD[5])
    assert abs(naive_delta - total) > 1.0, (
        "the naive formula did not diverge — this test no longer guards anything"
    )
    assert naive_delta < 0.0 < total, "naive formula should have folded negative"


# --------------------------------------------------------------------------- #
# (d) branch cut at wrist_3                                                    #
# --------------------------------------------------------------------------- #
def test_d_branch_cut_wrist3():
    """Leader anchored at +3.10 stepping across the cut to -3.10 (physically
    +0.0832 rad the SHORT way) must produce small, continuous commands."""
    q_lead = list(Q_LEAD)
    q_lead[5] = 3.10
    c = _engaged(q_lead=q_lead)

    ticks = 20
    start, end = 3.10, 3.10 + (2.0 * math.pi - 6.20)  # walks past +pi
    prev = list(c.q_cmd)
    for k in range(1, ticks + 1):
        q_lead[5] = start + (end - start) * k / ticks
        # Represent it the way a real leader would: wrapped into (-pi, pi].
        q_lead[5] = wrap_to_pi(q_lead[5])
        q, _ = c.step(list(q_lead), step_eff=BIG_STEP)
        for i in range(N):
            assert abs(q[i] - prev[i]) < 0.05, (
                f"tick {k} joint {i}: per-tick jump {abs(q[i] - prev[i])}"
            )
        prev = list(q)

    total = 2.0 * math.pi - 6.20
    assert abs(c.delta[5] - total) < 1e-9, f"delta[5]={c.delta[5]} != {total}"
    assert abs(c.delta[5]) < 0.2  # certainly not ~2*pi


# --------------------------------------------------------------------------- #
# (e) clutch freezes                                                           #
# --------------------------------------------------------------------------- #
def test_e_clutch_freezes_output():
    c = _engaged()
    q_lead = list(Q_LEAD)
    for _ in range(10):
        q_lead[0] += 0.01
        c.step(list(q_lead), step_eff=BIG_STEP)
    frozen = list(c.q_cmd)
    delta_at_clutch = list(c.delta)

    c.clutch()
    for _ in range(50):
        q_lead[0] += 0.02
        q, info = c.step(list(q_lead), step_eff=BIG_STEP)
        assert info["state"] == "CLUTCHED"
        assert q == frozen
    # The accumulator is frozen too: nothing is owed on re-engage.
    assert c.delta == delta_at_clutch


# --------------------------------------------------------------------------- #
# (f) reclutch = zero delta, zero jump                                         #
# --------------------------------------------------------------------------- #
def test_f_reclutch_zero_delta_continuity():
    c = _engaged()
    q_lead = list(Q_LEAD)
    c.step(list(q_lead), step_eff=BIG_STEP)
    c.clutch()
    q_cmd_now = list(c.q_cmd)

    # Operator repositions the leader by 0.8 rad while clutched.
    q_lead[1] += 0.8
    c.step(list(q_lead), step_eff=BIG_STEP)

    c.reclutch(q_lead_now=list(q_lead), q_cmd_now=q_cmd_now)
    assert c.delta == [0.0] * N
    q, info = c.step(list(q_lead), step_eff=BIG_STEP)
    assert info["state"] == "ENGAGED"
    assert q == q_cmd_now  # exact continuity


# --------------------------------------------------------------------------- #
# (g) disengage freezes; never-engaged emits nothing                           #
# --------------------------------------------------------------------------- #
def test_g_disengage_freezes():
    c = _engaged()
    q_lead = list(Q_LEAD)
    q_lead[2] += 0.05
    last, _ = c.step(list(q_lead), step_eff=BIG_STEP)

    c.disengage()
    q_lead[2] += 0.5
    q, info = c.step(list(q_lead), step_eff=BIG_STEP)
    assert info["state"] == "DISENGAGED"
    assert q == last

    fresh = JointDeltaController()
    q, info = fresh.step(list(Q_LEAD), step_eff=1.0)
    assert q is None, "a controller that never engaged must not emit joints"
    assert info["state"] == "DISENGAGED"


# --------------------------------------------------------------------------- #
# (h) joint-limit saturation, elbow at +-pi (NOT +-2pi)                        #
# --------------------------------------------------------------------------- #
def test_h_joint_limit_saturation_elbow():
    margin = 0.05
    q_robot = list(Q_ROBOT)
    q_robot[2] = 3.0  # elbow already near its +pi limit
    c = _engaged({"limit_margin_rad": margin}, q_robot=q_robot)

    q_lead = list(Q_LEAD)
    for _ in range(400):
        q_lead[2] += 0.01
        q, info = c.step(list(q_lead), step_eff=BIG_STEP)

    expect = math.pi - margin
    assert abs(q[2] - expect) < 1e-12, f"elbow saturated at {q[2]}, want {expect}"
    assert info["limited"][2] is True
    assert q[2] < 2 * math.pi, "elbow must NOT be allowed past +pi"


# --------------------------------------------------------------------------- #
# (i) anti-windup: immediate return, no dead zone                              #
# --------------------------------------------------------------------------- #
def test_i_antiwindup_immediate_return():
    gain = 1.0
    q_robot = list(Q_ROBOT)
    q_robot[2] = 3.0
    c = _engaged({"gain": gain, "limit_margin_rad": 0.05}, q_robot=q_robot)

    q_lead = list(Q_LEAD)
    for _ in range(100):
        q_lead[2] += 0.02      # push hard into the limit for a long time
        c.step(list(q_lead), step_eff=BIG_STEP)
    saturated = c.q_cmd[2]

    q_lead[2] -= 0.01          # the return stroke
    q, _ = c.step(list(q_lead), step_eff=BIG_STEP)
    moved = saturated - q[2]
    assert abs(moved - gain * 0.01) < 1e-9, (
        f"return stroke moved {moved}, expected {gain * 0.01} (windup dead zone?)"
    )


# --------------------------------------------------------------------------- #
# (j) excursion cage                                                           #
# --------------------------------------------------------------------------- #
def test_j_excursion_band():
    band = 0.2
    c = _engaged({"max_excursion_rad": band, "leader_jump_max_rad": 100.0})
    q_lead = list(Q_LEAD)
    for sign in (+1.0, -1.0):
        for _ in range(500):
            for i in range(N):
                q_lead[i] += sign * 0.02
            q, info = c.step(list(q_lead), step_eff=BIG_STEP)
            for i in range(N):
                assert abs(q[i] - Q_ROBOT[i]) <= band + 1e-12, (
                    f"joint {i} left the cage: {abs(q[i] - Q_ROBOT[i])} > {band}"
                )
    assert info["excursion_rad"] <= band + 1e-12


# --------------------------------------------------------------------------- #
# (k) the downstream clamp_stage never binds                                   #
# --------------------------------------------------------------------------- #
def test_k_slew_matches_clamp_stage():
    """The controller pre-clamps with the SAME step_eff the bridge pipeline then
    clamps with, so clamp_stage(q_cmd, last_published, step) == q_cmd."""
    step = 0.0025
    c = _engaged()
    rng = random.Random(99)
    q_lead = list(Q_LEAD)
    for _ in range(500):
        for i in range(N):
            q_lead[i] += rng.uniform(-0.05, 0.05)   # deliberately faster than step
        last_published = list(c.q_cmd)
        q, info = c.step(list(q_lead), step_eff=step)
        assert info["max_joint_step"] <= step * (1.0 + 1e-12)
        assert clamp_stage(q, last_published, step) == q


# --------------------------------------------------------------------------- #
# (l) estop / bad input                                                        #
# --------------------------------------------------------------------------- #
def test_l_estop_and_bad_input():
    c = _engaged()
    q_lead = list(Q_LEAD)
    q_lead[0] += 0.01
    good, _ = c.step(list(q_lead), step_eff=BIG_STEP)

    q, info = c.step(list(q_lead), step_eff=0.0)
    assert info["state"] == "HOLD"
    assert info["reject_reason"] == REJECT_ESTOP
    assert q == good
    assert c.state == "ENGAGED"

    for bad in (float("nan"), float("inf")):
        broken = list(q_lead)
        broken[3] = bad
        q, info = c.step(broken, step_eff=BIG_STEP)
        assert info["state"] == "HOLD"
        assert info["reject_reason"] == REJECT_BAD_INPUT
        assert q == good
        assert c.state == "ENGAGED"

    q, info = c.step(list(q_lead), step_eff=float("nan"))
    assert info["reject_reason"] == REJECT_BAD_INPUT
    assert q == good


# --------------------------------------------------------------------------- #
# (m) leader jump absorbed                                                     #
# --------------------------------------------------------------------------- #
def test_m_leader_jump_absorbed():
    """The BACKSTOP guard's contract, stated precisely.

    "Absorbed" means the MAPPING SHIFTS by the jump and the arm does not move.
    It does NOT mean the leader<->arm correspondence is preserved: after this
    test the leader sits 2.0 rad from where the same arm pose used to put it.
    That is only harmless when the guard fires SYMMETRICALLY (a transient glitch
    trips it on the way out and again on the way back, netting zero) — which is
    exactly what a 1-Euro-filtered input destroys, and why the PRIMARY teleport
    detector runs on the raw stream instead (test_u).
    """
    cap = 0.35
    c = _engaged({"leader_jump_max_rad": cap})
    q_lead = list(Q_LEAD)
    c.step(list(q_lead), step_eff=BIG_STEP)
    before = list(c.q_cmd)

    q_lead[0] += 2.0                       # teleport
    q, info = c.step(list(q_lead), step_eff=BIG_STEP)
    assert q == before
    assert info["reject_reason"] == REJECT_LEADER_JUMP
    assert info["jump_absorbed"] == 1
    assert c.state == "ENGAGED"

    # The tick AFTER resumes tracking with NO residual offset from the jump:
    # the mapping shifted, so a normal 0.01 increment moves exactly 0.01.
    q_lead[0] += 0.01
    q, info = c.step(list(q_lead), step_eff=BIG_STEP)
    assert abs((q[0] - before[0]) - 0.01) < 1e-9
    assert info["reject_reason"] is None


# --------------------------------------------------------------------------- #
# (n) gain 0.0 never moves                                                     #
# --------------------------------------------------------------------------- #
def test_n_gain_zero_never_moves():
    c = _engaged({"gain": 0.0})
    rng = random.Random(5)
    q_lead = list(Q_LEAD)
    for _ in range(500):
        for i in range(N):
            q_lead[i] += rng.uniform(-0.05, 0.05)
        q, _ = c.step(list(q_lead), step_eff=BIG_STEP)
        assert q == Q_ROBOT


# --------------------------------------------------------------------------- #
# (o) info key set is stable on every path                                     #
# --------------------------------------------------------------------------- #
_INFO_KEYS = {
    "state", "reject_reason", "gain", "delta", "limited", "slewed",
    "excursion_rad", "max_joint_step", "jump_absorbed", "clamp_discarded",
    "q_lead", "q_robot_anchor", "q_lead_anchor",
}


def test_o_info_shape_stable():
    fresh = JointDeltaController()
    _, info = fresh.step(list(Q_LEAD), step_eff=1.0)          # DISENGAGED
    assert set(info) == _INFO_KEYS

    c = _engaged()
    _, info = c.step(list(Q_LEAD), step_eff=BIG_STEP)          # ENGAGED
    assert set(info) == _INFO_KEYS
    _, info = c.step(list(Q_LEAD), step_eff=0.0)               # HOLD
    assert set(info) == _INFO_KEYS
    c.clutch()
    _, info = c.step(list(Q_LEAD), step_eff=BIG_STEP)          # CLUTCHED
    assert set(info) == _INFO_KEYS
    c.disengage()
    _, info = c.step(list(Q_LEAD), step_eff=BIG_STEP)          # DISENGAGED
    assert set(info) == _INFO_KEYS


# --------------------------------------------------------------------------- #
# (p) default limits == ur_kin.JOINT_LIMITS (anti copy-drift)                   #
# --------------------------------------------------------------------------- #
def test_p_default_limits_match_ur_kin():
    """The controller copies the limits as plain floats so joint_delta mode never
    imports numpy. This is the ONE test that imports ur_kin, and it exists purely
    so the copy can never silently drift from the original."""
    from ur_gello_bringup.ur_kin import JOINT_LIMITS

    c = JointDeltaController()
    assert c.joint_limits == [tuple(row) for row in JOINT_LIMITS.tolist()]
    # And spell out the property that actually matters on this arm.
    assert c.joint_limits[2] == (-math.pi, math.pi), "elbow must be +-pi"


# --------------------------------------------------------------------------- #
# (q) cfg validation                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cfg",
    [
        {"gain": -0.1},
        {"gain": 2.5},
        {"max_excursion_rad": 0.0},
        {"max_excursion_rad": -1.0},
        {"leader_jump_max_rad": 0.0},
        {"delta_deadband_rad": -0.01},
        {"limit_margin_rad": -0.01},
        {"joint_limits": ((-1.0, 1.0),)},
        {"joint_limits": [(1.0, -1.0)] * 6},
    ],
)
def test_q_cfg_validation(cfg):
    with pytest.raises(ValueError):
        JointDeltaController(cfg)


# --------------------------------------------------------------------------- #
# (r) telescoping => no drift                                                  #
# --------------------------------------------------------------------------- #
def test_r_no_drift_telescoping():
    """Out and back over thousands of NOISY ticks returns to the anchor exactly.

    This is what justifies jd_delta_deadband_rad = 0.0: the per-tick increments
    telescope, so the accumulation itself contributes no drift. (Re-run this with
    a deadband and it fails — which is the point.)
    """
    c = _engaged()
    rng = random.Random(20260722)
    q_lead = list(Q_LEAD)
    walk = [0.0] * N
    for _ in range(2500):
        for i in range(N):
            d = rng.uniform(-0.02, 0.02)
            walk[i] += d
            q_lead[i] += d
        c.step(list(q_lead), step_eff=BIG_STEP)
    # Walk back to EXACTLY the anchor leader pose, in many small steps.
    for k in range(2500, 0, -1):
        for i in range(N):
            q_lead[i] = Q_LEAD[i] + walk[i] * (k - 1) / 2500.0
        c.step(list(q_lead), step_eff=BIG_STEP)

    worst_delta = max(abs(v) for v in c.delta)
    worst_cmd = max(abs(c.q_cmd[i] - Q_ROBOT[i]) for i in range(N))
    assert worst_delta < 1e-9, f"accumulator drifted: {worst_delta}"
    assert worst_cmd < 1e-9, f"command drifted: {worst_cmd}"


# --------------------------------------------------------------------------- #
# (s) the PRICE of anti-windup: clamp saturation shifts the mapping            #
# --------------------------------------------------------------------------- #
def test_s_clamp_saturation_shifts_the_mapping():
    """(r) holds ONLY while no position clamp saturates. Pin the exception.

    The excursion cage back-projects into the accumulator so the return stroke
    tracks immediately (test i) — and the price is that the overtravel is
    DISCARDED, so bringing the leader back exactly to its anchor pose leaves the
    command displaced by that discarded amount, in the OPPOSITE direction, for
    good. This was previously undocumented AND untested: test_r only ever ran
    with WIDE (cage 100), so it could not saturate.

    The behaviour is deliberate (see the joint_delta module docstring: the
    alternatives are an unbounded return-stroke dead zone, or freezing all six
    joints because one hit its cage). What this test guarantees is that it stays
    BOUNDED by the cage and ACCOUNTED FOR in clamp_discarded.
    """
    cage = 1.0
    over = 0.5
    c = JointDeltaController({
        "gain": 1.0, "max_excursion_rad": cage, "leader_jump_max_rad": 100.0,
    })
    c.engage([0.0] * N, [0.0] * N)

    q = 0.0
    while q < cage + over - 1e-9:                 # push out past the cage
        q += 0.001
        c.step([q] + [0.0] * (N - 1), step_eff=BIG_STEP)
    assert c.q_cmd[0] == pytest.approx(cage, abs=1e-9)
    assert c.clamp_discarded[0] == pytest.approx(over, abs=1e-6)

    while q > 1e-9:                               # ... and come all the way back
        q -= 0.001
        c.step([max(q, 0.0)] + [0.0] * (N - 1), step_eff=BIG_STEP)
    _, info = c.step([0.0] * N, step_eff=BIG_STEP)

    # NOT the anchor: short by exactly the discarded overtravel, opposite sign.
    assert c.q_cmd[0] == pytest.approx(-over, abs=1e-6)
    # Bounded by the cage on every joint, always.
    assert max(abs(v) for v in c.q_cmd) <= cage + 1e-9
    # ... and the loss is REPORTED, which is what makes it diagnosable.
    assert info["clamp_discarded"][0] == pytest.approx(over, abs=1e-6)
    assert all(info["clamp_discarded"][i] == 0.0 for i in range(1, N))


# --------------------------------------------------------------------------- #
# (t) absorb(): caller-driven resync — HOLD, slide the reference, accumulate 0  #
# --------------------------------------------------------------------------- #
def test_t_absorb_holds_and_resyncs():
    c = _engaged()
    before = list(c.q_cmd)
    delta_before = list(c.delta)

    moved = [Q_LEAD[i] + 0.4 for i in range(N)]
    q_out, info = c.absorb(moved)

    assert q_out == before, "absorb() must FREEZE the command"
    assert info["state"] == "HOLD"
    assert info["reject_reason"] == REJECT_LEADER_RESYNC
    assert set(info) == _INFO_KEYS
    assert c.delta == delta_before, "absorb() must accumulate NOTHING"
    assert c.state == "ENGAGED", "absorb() is a tick outcome, not a state change"
    assert c.q_lead_prev == moved, "absorb() must slide the reference"

    # Stepping resumes from the RESYNCED reference: no residual from the jump.
    c.step(list(moved), step_eff=BIG_STEP)
    assert c.q_cmd == pytest.approx(before, abs=1e-12)
    nxt = [moved[i] + 0.05 for i in range(N)]
    c.step(nxt, step_eff=BIG_STEP)
    assert c.q_cmd == pytest.approx([v + 0.05 for v in before], abs=1e-12)

    # Non-finite input must never be slid onto the reference.
    ref = list(c.q_lead_prev)
    _, info = c.absorb([float("nan")] + list(nxt[1:]))
    assert info["reject_reason"] == REJECT_BAD_INPUT
    assert c.q_lead_prev == ref

    # Frozen states short-circuit exactly like step().
    c.clutch()
    q_out, info = c.absorb([v + 1.0 for v in nxt])
    assert info["state"] == "CLUTCHED"
    assert q_out == c.q_cmd


# --------------------------------------------------------------------------- #
# (u) the resync window, end-to-end through the REAL filter and cadences       #
# --------------------------------------------------------------------------- #
def _teleport_sim(glitch_rad, use_resync_window, resync_hold_s=0.30):
    """Replicate the bridge's joint_delta signal chain exactly.

    250 Hz publish tick, ~30 Hz GELLO samples, the shipped OneEuro gains and the
    shipped max_step_rad; a single bad leader sample on joint 0 that then returns
    to the true value. ``use_resync_window`` models the node's RAW-domain
    detector exactly: on a raw increment over the cap it RE-SEEDS the leader
    filter onto the new reading (no smoothing across a discontinuity, hence no
    exponential tail left to be accumulated once the window expires) and calls
    absorb() instead of step() for jd_resync_hold_s.
    """
    dt_pub, dt_lead = 1.0 / 250.0, 1.0 / 30.0
    ticks_per_sample = 250 // 30
    cap = 0.35
    euro = [OneEuro(dt_pub, 1.0, 2.0, 1.0) for _ in range(N)]
    for e in euro:
        e.seed(0.0)
    c = JointDeltaController({
        "gain": 1.0, "max_excursion_rad": 100.0, "leader_jump_max_rad": cap,
    })
    c.engage([0.0] * N, [0.0] * N)

    now = 0.0
    resync_until = None
    raw_prev = [0.0] * N
    raw = [0.0] * N
    peak = 0.0

    def feed(nsamples):
        nonlocal now, resync_until, raw_prev, peak
        for _ in range(nsamples):
            if use_resync_window:
                jump = max(abs(wrap_to_pi(raw[i] - raw_prev[i])) for i in range(N))
            else:
                jump = 0.0
            raw_prev = list(raw)
            for i, e in enumerate(euro):
                e.update_input(raw[i], dt_lead)
            if jump > cap:
                resync_until = now + resync_hold_s
                for i, e in enumerate(euro):
                    e.seed(raw[i])
            for _ in range(ticks_per_sample):
                q_lead_f = [euro[i](raw[i]) for i in range(N)]
                if resync_until is not None and now < resync_until:
                    c.absorb(q_lead_f)
                else:
                    resync_until = None
                    c.step(q_lead_f, step_eff=0.0025)
                peak = max(peak, max(abs(v) for v in c.q_cmd))
                now += dt_pub

    feed(30)
    raw[0] = glitch_rad          # ONE bad sample
    feed(1)
    raw[0] = 0.0                 # ... reading returns to the truth
    feed(400)
    return c, peak


@pytest.mark.parametrize("glitch", [0.5, 1.0, 2.0, math.pi])
def test_u_resync_window_neutralises_a_raw_teleport(glitch):
    """WITH the node's raw-domain window the arm does not move at all; WITHOUT it
    the same glitch either sweeps the arm (small glitches never trip the
    filtered-domain backstop) or leaves a permanent offset (big ones trip it
    asymmetrically across the filter's smear). This is the regression guard for
    "the teleport cap must be tested on the RAW stream".
    """
    guarded, guarded_peak = _teleport_sim(glitch, use_resync_window=True)
    assert guarded_peak < 1e-9, f"resync window leaked motion: peak={guarded_peak}"
    assert max(abs(v) for v in guarded.q_cmd) < 1e-9, (
        f"resync window left a residual: {guarded.q_cmd}"
    )

    naive, naive_peak = _teleport_sim(glitch, use_resync_window=False)
    residual = max(abs(v) for v in naive.q_cmd)
    # In BOTH bands the unguarded chain sweeps the arm a long way while the
    # glitch is being digested -- that transient IS the hazard.
    assert naive_peak > 0.05, (
        f"expected the unguarded chain to sweep the arm (peak={naive_peak}); if "
        "this fails the filter gains changed and the threshold analysis in "
        "joint_delta.py / the yaml must be redone"
    )
    if glitch <= 1.2:
        # The filtered-domain backstop never even FIRES in this band: the whole
        # glitch is accumulated as if it were genuine leader travel. It happens
        # to telescope back to zero here only because the glitch fully reverses.
        assert naive.jump_absorbed == 0
    else:
        # Here it fires -- but only across the first ticks of the smear, so the
        # discard is asymmetric and leaves a PERMANENT offset.
        assert naive.jump_absorbed > 0
        assert residual > 0.1, (
            f"expected a permanent offset from the asymmetric absorb, got {residual}"
        )


# --------------------------------------------------------------------------- #
# (v) re-anchoring clears the discard ledger (it is the documented remedy)     #
# --------------------------------------------------------------------------- #
def test_v_reanchor_clears_clamp_discarded():
    c = JointDeltaController({
        "gain": 1.0, "max_excursion_rad": 0.2, "leader_jump_max_rad": 100.0,
    })
    c.engage([0.0] * N, [0.0] * N)
    q = 0.0
    for _ in range(600):
        q += 0.001
        c.step([q] + [0.0] * (N - 1), step_eff=BIG_STEP)
    assert c.clamp_discarded[0] > 0.3

    c.clutch()
    c.reclutch(q_lead_now=[q] + [0.0] * (N - 1), q_cmd_now=list(c.q_cmd))
    assert c.clamp_discarded == [0.0] * N
    assert c.delta == [0.0] * N
    _, info = c.step([q] + [0.0] * (N - 1), step_eff=BIG_STEP)
    assert info["clamp_discarded"] == [0.0] * N
