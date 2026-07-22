"""Tests for ur_gello_bringup.eef_delta (anchor / delta / SE(3) governor).

Every letter maps to the task spec:

  (a) zero-jump: first step after engage == q_anchor to 1e-9 (500 pairs).
  (b) joint-equivalence: tool_l==tool_r, leader==robot, R_align=I, rate limit
      off -> the whole trajectory is reproduced to 1e-6.
  (c) rotation multiplication-order regression: flipping R_delta to the body
      order R_g_anchor^T @ R_g breaks (b).
  (d) 0->360 continuous rotation: T_des continuous through pi (no axis flip, no
      cap), zero rejects, exact return to the anchor orientation at 360 deg.
  (e) R_align = RotZ(90): a +x leader translation becomes a +y robot translation.
  (f) asymmetric gamma escape: at sigma_min < sigma_stop, the increment that
      moves *away* from the singularity is passed unthrottled.
  (g) analytic scale: at low manipulability the step scale stays finite & nonzero
      (no dead stop).
  (h) wrist-singularity branch lock: a ~pi branch flip is rejected (analytic).
  (i) disengage freezes the anchor; reclutch is fresh (zero delta).
  (j) anti-windup: while HOLDing, ||T_cmd (-) T_des|| stays under lag_max_pose.
  (k) each reject_reason is a distinct string (forced scenarios).
"""

import math

import numpy as np
import pytest

from ur_gello_bringup.eef_delta import EefDeltaController
from ur_gello_bringup.ur_kin import (
    fk,
    link_origins,
    sigma_min,
    ik_analytic,
    branch_id,
    se3_log,
    so3_log,
)


def _inv(T):
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def _rand_q(rng):
    q = rng.uniform(-2.5, 2.5, 6)
    q[2] = rng.uniform(-2.9, 2.9)  # elbow within +/-pi
    return q


# --------------------------------------------------------------------------- #
# (a) zero-jump                                                                #
# --------------------------------------------------------------------------- #
def test_a_zero_jump_500_pairs():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        q_anchor = _rand_q(rng)
        q_lead = _rand_q(rng)
        c = EefDeltaController({"dt": 0.01})
        c.engage(q_anchor, q_lead)
        # Step at t0 with the SAME leader value used for the anchor -> zero delta.
        q_cmd, info = c.step(q_lead, step_eff=0.2)
        worst = max(worst, float(np.max(np.abs(q_cmd - q_anchor))))
        assert info["state"] == "ENGAGED"
    assert worst < 1e-9, f"zero-jump worst error {worst}"


# --------------------------------------------------------------------------- #
# (b) joint equivalence                                                        #
# --------------------------------------------------------------------------- #
def test_b_joint_equivalence_full_trajectory():
    tool = [0.01, 0.02, 0.05, 0.1, -0.2, 0.3]
    cfg = dict(dt=0.01, v_max=1e6, w_max=1e6,
               tool_l_xyz_rpy=tool, tool_r_xyz_rpy=tool, pos_scale=1.0)
    c = EefDeltaController(cfg)
    q0 = np.array([0.3, -1.2, 1.1, -0.7, 1.3, 0.4])
    c.engage(q0, q0)

    rng = np.random.default_rng(2)
    q = q0.copy()
    traj = [q0.copy()]
    for _ in range(60):
        q = q + rng.uniform(-0.008, 0.008, 6)
        traj.append(q.copy())

    worst = 0.0
    for qt in traj:
        q_cmd, info = c.step(qt, step_eff=100.0)
        assert info["state"] == "ENGAGED", info["reject_reason"]
        worst = max(worst, float(np.max(np.abs(q_cmd - qt))))
    assert worst < 1e-6, f"trajectory tracking error {worst}"


# --------------------------------------------------------------------------- #
# (c) rotation multiplication-order regression                                #
# --------------------------------------------------------------------------- #
def test_c_rotation_order_regression():
    """The correct world-frame order R_g @ R_g_anchor^T reproduces the leader
    orientation; the flipped body order R_g_anchor^T @ R_g does NOT."""
    q0 = np.array([0.3, -1.2, 1.1, -0.7, 1.3, 0.4])
    qt = np.array([0.9, -0.7, 0.6, 0.4, -0.8, 1.1])  # non-commuting orientation
    R_g_anchor = fk(q0)[:3, :3]
    R_g = fk(qt)[:3, :3]
    R_r_anchor = R_g_anchor  # tool_l == tool_r, coincident anchor (b-setup)

    R_correct = (R_g @ R_g_anchor.T) @ R_r_anchor          # world/left (correct)
    R_flipped = (R_g_anchor.T @ R_g) @ R_r_anchor          # body (wrong)
    R_expected = R_g                                       # == fk(qt) rotation

    assert np.max(np.abs(R_correct - R_expected)) < 1e-12
    # The flipped order is a materially different rotation (non-commuting case).
    ang = float(np.linalg.norm(so3_log(R_flipped @ R_expected.T)))
    assert ang > 0.5, f"flipped order should differ, angle {ang}"
    assert np.max(np.abs(R_flipped - R_expected)) > 0.1


# --------------------------------------------------------------------------- #
# (d) 0 -> 360 continuous rotation                                            #
# --------------------------------------------------------------------------- #
def test_d_full_turn_continuous_no_cap():
    cfg = dict(dt=0.02, v_max=1e6, w_max=1e6)
    c = EefDeltaController(cfg)
    # wrist_3 low so the mirrored +2pi stays inside +/-2pi joint limits.
    qa = np.array([0.2, -1.2, 1.0, -0.6, 1.3, -3.0])
    c.engage(qa, qa)

    N = 400
    rejects = 0
    prev_R = None
    max_tick_ang = 0.0
    half_ang = None
    info = None
    for k in range(N + 1):
        th = 2.0 * math.pi * k / N
        ql = qa.copy()
        ql[5] = qa[5] + th
        _, info = c.step(ql, step_eff=100.0)
        if info["state"] != "ENGAGED":
            rejects += 1
        R_des = info["T_des"][:3, :3]
        if prev_R is not None:
            max_tick_ang = max(
                max_tick_ang, float(np.linalg.norm(so3_log(R_des @ prev_R.T)))
            )
        if k == N // 2:
            half_ang = float(np.linalg.norm(so3_log(R_des @ c.R_r_anchor.T)))
        prev_R = R_des

    assert rejects == 0, f"{rejects} rejects during a full turn"
    # Continuity: every tick's incremental rotation is small (no jump at pi).
    assert max_tick_ang < 0.02
    # No cap: at 180 deg the delta really is ~pi away from the anchor.
    assert abs(half_ang - math.pi) < 1e-6, f"halfway delta {half_ang} != pi (capped?)"
    # Exact return to the anchor orientation at 360 deg.
    back = float(np.linalg.norm(so3_log(info["T_des"][:3, :3] @ c.R_r_anchor.T)))
    assert back < 1e-9, f"did not return to anchor orientation: {back}"


# --------------------------------------------------------------------------- #
# (e) R_align = RotZ(90): leader +x -> robot +y                               #
# --------------------------------------------------------------------------- #
def test_e_r_align_maps_x_to_y():
    cfg = dict(dt=0.01, v_max=1e6, w_max=1e6,
               r_align_rpy=[0, 0, math.pi / 2], pos_scale=1.0)
    c = EefDeltaController(cfg)
    qa = np.array([0.2, -1.0, 1.0, -0.5, 1.2, 0.3])
    c.engage(qa, qa)

    ql = qa + np.array([0.02, 0, 0, 0, 0, 0])
    d_g = fk(ql)[:3, 3] - fk(qa)[:3, 3]           # leader TCP displacement
    _, info = c.step(ql, step_eff=100.0)
    d_r = info["T_des"][:3, 3] - c.p_r_anchor      # commanded robot displacement

    # RotZ(90): [dx,dy,dz] -> [-dy, dx, dz].
    assert np.allclose(d_r, [-d_g[1], d_g[0], d_g[2]], atol=1e-9)
    # Concretely: leader's x-component shows up as the robot's y-component.
    assert abs(d_r[1] - d_g[0]) < 1e-9
    assert abs(d_r[0] + d_g[1]) < 1e-9


# --------------------------------------------------------------------------- #
# (f) asymmetric gamma escape from a locked-up singularity                    #
# --------------------------------------------------------------------------- #
def test_f_asymmetric_gamma_escape():
    qa = np.array([0.3, -1.1, 1.0, -0.6, 0.04, 0.4])  # wrist_2 ~0 -> sigma < sigma_stop
    assert sigma_min(qa) < 0.03  # genuinely locked

    def run(direction):
        cfg = dict(dt=0.05, v_max=0.08, w_max=0.5)
        c = EefDeltaController(cfg)
        c.engage(qa, qa)
        ql = qa.copy()
        ql[4] = qa[4] + direction * 0.1  # +away from singularity, -toward it
        q_cmd, info = c.step(ql, step_eff=0.5)
        return float(np.max(np.abs(q_cmd - qa))), info, sigma_min(q_cmd)

    away_dq, away_info, away_sig = run(+1)
    toward_dq, toward_info, _ = run(-1)

    # Away move is unthrottled (gamma forced to 1) and actually escapes.
    assert away_info["gamma"] == pytest.approx(1.0)
    assert away_sig > sigma_min(qa)
    # Toward move stays throttled at gamma_min.
    assert toward_info["gamma"] == pytest.approx(0.05)
    # The asymmetry gives the escaping increment a much larger step.
    assert away_dq > 10.0 * toward_dq
    assert away_info["state"] == "ENGAGED"


# --------------------------------------------------------------------------- #
# (g) analytic scale stays finite & nonzero at low manipulability             #
# --------------------------------------------------------------------------- #
def test_g_analytic_scale_no_dead_stop():
    cfg = dict(dt=0.05, v_max=0.08, w_max=0.5)
    c = EefDeltaController(cfg)
    qa = np.array([0.3, -1.1, 1.0, -0.6, 0.08, 0.4])  # low sigma (warn band)
    s = sigma_min(qa)
    assert cfg and 0.03 < s < 0.10
    c.engage(qa, qa)
    ql = qa.copy()
    ql[4] = qa[4] - 0.05  # move further toward the singularity (throttled)
    q_cmd, info = c.step(ql, step_eff=0.5)

    assert info["state"] == "ENGAGED"
    assert 0.0 < info["gamma"] < 1.0          # throttled but not zero
    assert info["ls_scale"] > 0.0             # finite, nonzero scale
    assert float(np.max(np.abs(q_cmd - qa))) > 1e-5  # it actually moves


# --------------------------------------------------------------------------- #
# (h) wrist-singularity branch lock rejects a ~pi flip                         #
# --------------------------------------------------------------------------- #
def test_h_branch_lock_rejects_wrist_flip():
    if not ik_analytic(fk(np.array([0.3, -1.1, 1.0, -0.6, 0.15, 0.4]))):
        pytest.skip("analytic IK unavailable")

    qa = np.array([0.3, -1.1, 1.0, -0.6, 0.15, 0.4])
    # Wrist-flip partner: wrist_2 negated, wrist_1 & wrist_3 shifted by pi.
    qb = qa.copy()
    qb[3] = qa[3] + math.pi
    qb[4] = -qa[4]
    qb[5] = qa[5] + math.pi
    # The refused motion is a genuine >3 rad branch jump on a different branch.
    assert np.linalg.norm(qb - qa) >= 3.0
    assert branch_id(qb) != branch_id(qa)

    # Rate limit / anti-windup off so the controller attempts the whole flip.
    cfg = dict(dt=1.0, v_max=1e6, w_max=1e6, lag_max_pose=[100.0, 100.0])
    c = EefDeltaController(cfg)
    c.engage(qa, qa)
    q_cmd, info = c.step(qb, step_eff=100.0)

    assert info["state"] == "HOLD"
    assert info["reject_reason"] == "BRANCH_JUMP"
    assert np.allclose(q_cmd, qa)  # held at the anchor, no jump


# --------------------------------------------------------------------------- #
# (i) disengage freezes; reclutch is fresh                                    #
# --------------------------------------------------------------------------- #
def test_i_disengage_and_reclutch():
    cfg = dict(dt=0.02, v_max=1e6, w_max=1e6)
    c = EefDeltaController(cfg)
    qa = np.array([0.3, -1.1, 1.0, -0.6, 1.2, 0.4])
    c.engage(qa, qa)

    # Advance a bit.
    ql = qa.copy()
    last = None
    for _ in range(5):
        ql = ql + np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0])
        last, info = c.step(ql, step_eff=100.0)
    assert not np.allclose(last, qa)  # it moved

    # Disengage: further leader motion must not advance the command.
    c.disengage()
    ql_far = ql + np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    q_dis, info = c.step(ql_far, step_eff=100.0)
    assert info["state"] == "DISENGAGED"
    assert np.allclose(q_dis, last)  # frozen, no advance

    # Reclutch at the current leader/robot pose -> fresh, zero delta.
    c.reclutch(ql_far, last)
    q_re, info = c.step(ql_far, step_eff=100.0)
    assert info["state"] == "ENGAGED"
    assert np.allclose(q_re, last, atol=1e-9)  # delta 0 at reclutch pose


# --------------------------------------------------------------------------- #
# (j) anti-windup bounds the T_cmd -> T_des lag while HOLDing                  #
# --------------------------------------------------------------------------- #
def test_j_anti_windup_bounds_lag():
    lag_pos, lag_rot = 0.05, 0.3
    cfg = dict(dt=1.0, v_max=1e6, w_max=1e6,
               lag_max_pose=[lag_pos, lag_rot], max_excursion_m=0.01, pos_scale=2.0)
    c = EefDeltaController(cfg)
    qa = np.array([0.3, -1.1, 1.0, -0.6, 1.2, 0.4])
    c.engage(qa, qa)

    # Persistently drag the leader far away; the excursion cap keeps HOLDing.
    ql = qa.copy()
    saw_hold = False
    for _ in range(15):
        ql = ql + np.array([0.15, 0.05, 0.0, 0.0, 0.0, 0.05])
        _, info = c.step(ql, step_eff=100.0)
        if info["state"] == "HOLD":
            saw_hold = True
        xi = se3_log(_inv(info["T_cmd"]) @ info["T_des"])
        assert np.linalg.norm(xi[:3]) <= lag_pos + 1e-9
        assert np.linalg.norm(xi[3:]) <= lag_rot + 1e-9
    assert saw_hold


# --------------------------------------------------------------------------- #
# (k) every reject_reason is a distinct string                                #
# --------------------------------------------------------------------------- #
def _reason_no_ik():
    cfg = dict(dt=1.0, v_max=1e6, w_max=1e6, lag_max_pose=[100.0, 100.0], pos_scale=3.0)
    c = EefDeltaController(cfg)
    qa = np.array([0.2, -1.0, 1.0, -0.6, 1.2, 0.3])
    c.engage(qa, qa)
    _, info = c.step(np.array([0.0, -0.2, 0.2, -0.2, 0.5, 0.0]), step_eff=100.0)
    return info


def _reason_branch_jump():
    cfg = dict(dt=1.0, v_max=1e6, w_max=1e6, lag_max_pose=[100.0, 100.0])
    c = EefDeltaController(cfg)
    qa = np.array([0.3, -1.1, 1.0, -0.6, 0.15, 0.4])
    c.engage(qa, qa)
    qb = qa.copy()
    qb[3] += math.pi
    qb[4] = -qa[4]
    qb[5] += math.pi
    _, info = c.step(qb, step_eff=100.0)
    return info


def _reason_joint_limit():
    cfg = dict(dt=1.0, v_max=1e6, w_max=1e6, lag_max_pose=[100.0, 100.0])
    c = EefDeltaController(cfg)
    qa = np.array([0.2, -1.0, math.pi - 0.12, -0.6, 1.2, 0.3])
    c.engage(qa, qa)
    ql = qa.copy()
    ql[2] = math.pi - 0.02  # past the 0.05 limit margin
    _, info = c.step(ql, step_eff=100.0)
    return info


def _reason_keepout():
    qa = np.array([0.0, -0.3, 0.9, -0.6, 1.0, 0.0])
    ql = qa.copy()
    ql[1] = qa[1] + 0.05
    floor = min(o[2] for o in link_origins(qa)) - 0.005
    cfg = dict(dt=1.0, v_max=1e6, w_max=1e6, lag_max_pose=[100.0, 100.0],
               keepout={"floor_z": floor})
    c = EefDeltaController(cfg)
    c.engage(qa, qa)
    _, info = c.step(ql, step_eff=100.0)
    return info


def _reason_excursion():
    cfg = dict(dt=1.0, v_max=1e6, w_max=1e6, lag_max_pose=[100.0, 100.0],
               max_excursion_m=0.02)
    c = EefDeltaController(cfg)
    qa = np.array([0.2, -1.0, 1.0, -0.6, 1.2, 0.3])
    c.engage(qa, qa)
    _, info = c.step(qa + np.array([0.05, 0, 0, 0, 0, 0]), step_eff=100.0)
    return info


def test_k_reject_reasons_distinct():
    infos = {
        "NO_IK": _reason_no_ik(),
        "BRANCH_JUMP": _reason_branch_jump(),
        "JOINT_LIMIT": _reason_joint_limit(),
        "GEOM_KEEPOUT": _reason_keepout(),
        "EXCURSION": _reason_excursion(),
    }
    reasons = {}
    for expected, info in infos.items():
        assert info["state"] == "HOLD", f"{expected}: expected HOLD, got {info['state']}"
        assert info["reject_reason"] == expected, (
            f"expected {expected}, got {info['reject_reason']}"
        )
        reasons[expected] = info["reject_reason"]
    # All five are distinct strings.
    assert len(set(reasons.values())) == 5


# --------------------------------------------------------------------------- #
# (A1) non-finite input -> HOLD (no crash), reason BAD_INPUT                   #
# --------------------------------------------------------------------------- #
def test_a1_nonfinite_leader_joint_holds():
    qa = np.array([0.3, -1.1, 1.0, -0.6, 1.2, 0.4])
    for bad in (math.inf, -math.inf, math.nan):
        for j in range(6):
            c = EefDeltaController()
            c.engage(qa, qa)
            ql = qa.copy()
            ql[j] = bad
            q_cmd, info = c.step(ql, step_eff=0.2)  # must NOT raise (math.cos(inf))
            assert info["state"] == "HOLD"
            assert info["reject_reason"] == "BAD_INPUT"
            assert np.allclose(q_cmd, qa)                 # frozen at the anchor
            assert np.allclose(info["T_cmd"], c.T_r_anchor)  # T_cmd frozen


def test_a1_nonfinite_step_budget_holds():
    qa = np.array([0.3, -1.1, 1.0, -0.6, 1.2, 0.4])
    for bad in (math.inf, math.nan):
        c = EefDeltaController()
        c.engage(qa, qa)
        q_cmd, info = c.step(qa + np.array([0, 0.01, 0, 0, 0, 0]), step_eff=bad)
        assert info["state"] == "HOLD"
        assert info["reject_reason"] == "BAD_INPUT"
        assert np.allclose(q_cmd, qa)


# --------------------------------------------------------------------------- #
# (A2) step_eff <= 0 is a HARD STOP, never a limiter-disabling freeze          #
# --------------------------------------------------------------------------- #
def test_a2_nonpositive_step_is_hard_stop():
    # Near-singular pose: a disabled limiter would allow a huge jump here.
    qa = np.array([0.3, -1.1, 1.0, -0.6, 0.04, 0.4])
    ql = qa.copy()
    ql[4] = qa[4] + 0.1  # a leader move that WOULD drive a large joint change

    for bad_budget in (0.0, -1.0, -1e-9):
        c = EefDeltaController(dict(dt=0.05))
        c.engage(qa, qa)
        q_cmd, info = c.step(ql, step_eff=bad_budget)
        assert info["state"] == "HOLD"
        assert info["reject_reason"] == "ESTOP"
        assert np.allclose(q_cmd, qa)  # bit-for-bit no motion

    # And it must never move MORE than a small positive budget would.
    c0 = EefDeltaController(dict(dt=0.05))
    c0.engage(qa, qa)
    q_zero, _ = c0.step(ql, step_eff=0.0)
    c1 = EefDeltaController(dict(dt=0.05))
    c1.engage(qa, qa)
    q_small, _ = c1.step(ql, step_eff=0.02)
    jump_zero = float(np.max(np.abs(q_zero - qa)))
    jump_small = float(np.max(np.abs(q_small - qa)))
    assert jump_zero == 0.0
    assert jump_zero <= jump_small  # zero budget never produces a bigger jump


# --------------------------------------------------------------------------- #
# (A3) step() before engage must not emit a joint vector (no zeros(6) lurch)   #
# --------------------------------------------------------------------------- #
def test_a3_pre_engage_step_returns_none_not_zeros():
    c = EefDeltaController()
    q_cmd, info = c.step(np.array([0.3, -1.1, 1.0, -0.6, 1.2, 0.4]), step_eff=0.2)
    assert info["state"] == "DISENGAGED"
    assert q_cmd is None            # not zeros(6), not any joint vector
    # The internal anchor state is likewise unset until engage.
    assert c.q_cmd is None and c.q_ik_prev is None


# --------------------------------------------------------------------------- #
# (C1) small-but-nonzero leader delta right after engage -> no jump            #
#      (the real safety property; the 500 exact-zero pairs are vacuous)        #
# --------------------------------------------------------------------------- #
def test_c1_small_nonzero_delta_no_jump():
    rng = np.random.default_rng(5)
    worst = 0.0
    n_moved = 0
    for _ in range(80):
        qa = _rand_q(rng)
        c = EefDeltaController({"dt": 0.01})
        c.engage(qa, qa)
        ql = qa + rng.uniform(-1e-3, 1e-3, 6)  # tiny but genuinely nonzero
        q_cmd, info = c.step(ql, step_eff=0.2)
        if info["state"] != "ENGAGED":
            continue
        jump = float(np.max(np.abs(q_cmd - qa)))
        worst = max(worst, jump)
        if jump > 1e-12:
            n_moved += 1
    # It really did move (so this is NOT the exact-zero shortcut) ...
    assert n_moved > 0, "no accepted motion -> test would be vacuous"
    # ... yet a tiny leader delta only ever produced a tiny joint delta.
    assert worst < 5e-3, f"small delta produced a {worst} rad jump"


# --------------------------------------------------------------------------- #
# (C2) R_align != I with a real leader rotation: full conjugation is applied   #
# --------------------------------------------------------------------------- #
def test_c2_r_align_rotation_conjugation():
    cfg = dict(dt=0.02, v_max=1e6, w_max=1e6, r_align_rpy=[0, 0, math.pi / 2])
    c = EefDeltaController(cfg)
    qa = np.array([0.2, -1.0, 1.0, -0.5, 1.2, 0.3])
    c.engage(qa, qa)
    ql = qa + np.array([0.0, 0.0, 0.0, 0.03, -0.02, 0.025])  # small wrist rotation
    _, info = c.step(ql, step_eff=100.0)
    assert info["state"] == "ENGAGED"

    R_align = c.R_align
    R_g = (fk(ql) @ c.T_tool_L)[:3, :3]
    leader_rot = float(np.linalg.norm(so3_log(R_g @ c.R_g_anchor.T)))
    assert leader_rot > 1e-3, "leader rotation must be non-trivial for this test"

    # T_des rotation == R_align (R_g R_g_anchor^T) R_align^T R_r_anchor exactly.
    R_expected = R_align @ (R_g @ c.R_g_anchor.T) @ R_align.T @ c.R_r_anchor
    assert np.max(np.abs(info["T_des"][:3, :3] - R_expected)) < 1e-9

    # Regression guard: dropping the R_align^T conjugate gives a DIFFERENT answer
    # (so the test would fail if the conjugation were removed).
    R_no_conj = R_align @ (R_g @ c.R_g_anchor.T) @ c.R_r_anchor
    assert np.max(np.abs(R_no_conj - R_expected)) > 1e-6


# --------------------------------------------------------------------------- #
# (C3) tool_l != tool_r: achieved robot TCP tracks the commanded desired TCP   #
#      (guards the T_tool_R inverse / swap)                                    #
# --------------------------------------------------------------------------- #
def test_c3_tool_l_ne_tool_r_tcp_tracks_desired():
    tool_l = [0.01, 0.02, 0.05, 0.1, -0.2, 0.3]
    tool_r = [-0.03, 0.04, 0.02, -0.15, 0.25, -0.1]
    cfg = dict(dt=0.02, v_max=1e6, w_max=1e6, lag_max_pose=[100.0, 100.0],
               tool_l_xyz_rpy=tool_l, tool_r_xyz_rpy=tool_r)
    c = EefDeltaController(cfg)
    qa = np.array([0.3, -1.1, 1.0, -0.6, 1.2, 0.4])
    c.engage(qa, qa)
    ql = qa + np.array([0.05, 0.03, -0.02, 0.04, 0.05, -0.03])
    q_cmd, info = c.step(ql, step_eff=100.0)
    assert info["state"] == "ENGAGED", info["reject_reason"]

    # fk(q_cmd) @ T_tool_R must reproduce the commanded desired TCP. A swapped
    # T_tool_R / inverse would leave a residual T_tool_R^2 offset here.
    achieved = fk(q_cmd) @ c.T_tool_R
    assert np.allclose(achieved[:3, 3], info["T_des"][:3, 3], atol=1e-6)
    assert np.max(np.abs(achieved[:3, :3] - info["T_des"][:3, :3])) < 1e-6
    assert float(np.max(np.abs(q_cmd - qa))) > 1e-3  # it actually moved


# --------------------------------------------------------------------------- #
# (C4) rot_scale != 1.0 is rejected at construction; 1.0 / absent accepted     #
# --------------------------------------------------------------------------- #
def test_c4_rot_scale_must_be_unity():
    with pytest.raises(ValueError):
        EefDeltaController({"rot_scale": 1.5})
    with pytest.raises(ValueError):
        EefDeltaController({"rot_scale": 0.0})
    # 1.0 and absent are both fine.
    EefDeltaController({"rot_scale": 1.0})
    EefDeltaController({})


# --------------------------------------------------------------------------- #
# (C5) new reject reasons are distinct strings, alongside the existing five    #
# --------------------------------------------------------------------------- #
def test_c5_all_reject_reasons_distinct():
    qa = np.array([0.3, -1.1, 1.0, -0.6, 1.2, 0.4])

    c = EefDeltaController()
    c.engage(qa, qa)
    ql = qa.copy()
    ql[0] = math.inf
    _, info_bad = c.step(ql, step_eff=0.2)

    c = EefDeltaController()
    c.engage(qa, qa)
    _, info_estop = c.step(qa + np.array([0, 0.01, 0, 0, 0, 0]), step_eff=0.0)

    assert info_bad["reject_reason"] == "BAD_INPUT"
    assert info_estop["reject_reason"] == "ESTOP"

    all_reasons = {
        info_bad["reject_reason"],
        info_estop["reject_reason"],
        _reason_no_ik()["reject_reason"],
        _reason_branch_jump()["reject_reason"],
        _reason_joint_limit()["reject_reason"],
        _reason_keepout()["reject_reason"],
        _reason_excursion()["reject_reason"],
    }
    # All seven reject reasons are distinct, non-empty strings.
    assert len(all_reasons) == 7
    assert all(isinstance(r, str) and r for r in all_reasons)
