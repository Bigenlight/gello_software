"""Tests for ur_gello_bringup.ur_kin (pure-numpy UR7e kinematics).

Covers (a) FK<->analytic-IK round trip, (b) numeric<->analytic cross-check,
(c) Jacobian vs central difference, (d) sigma_min at/away-from singularity,
(e) branch_id continuity + determinism, (f) joint-limit boundaries,
(g) SE(3)/SO(3) log/exp round trips incl. theta->0 and theta->pi,
(h) matrix<->quaternion(xyzw) round trip and ROS ordering,
(i) keepout floor reject/accept, (j) wrist-singularity >=3 rad branch jump,
(k) worst-case tick timing (informational print).

All configurations sampled here are kept inside joint limits and away from
singularities unless the test is explicitly about a singularity.
"""

import math
import time

import numpy as np
import pytest

from ur_gello_bringup.ur_kin import (
    fk,
    jacobian,
    sigma_min,
    ik_numeric,
    ik_analytic,
    branch_id,
    wrapped_nearest,
    within_joint_limits,
    link_origins,
    keepout_ok,
    se3_log,
    se3_exp,
    so3_log,
    so3_exp,
    mat_to_quat_xyzw,
    quat_xyzw_to_mat,
    load_dh,
    _ik_analytic_tagged,
    _pose_error,
    JOINT_LIMITS,
    CHAR_LENGTH,
)


def _wrap(x):
    return np.array([math.remainder(v, 2.0 * math.pi) for v in np.atleast_1d(x)])


def _rand_q(rng, elbow_lim=2.9):
    """A random configuration inside limits; elbow kept within +/-pi."""
    q = rng.uniform(-2.5, 2.5, 6)
    q[2] = rng.uniform(-elbow_lim, elbow_lim)
    return q


def _rand_nonsingular_q(rng):
    """Random config that is comfortably away from wrist/elbow singularities."""
    while True:
        q = _rand_q(rng)
        # keep wrist_2 away from 0/pi and elbow away from 0/pi
        if abs(math.sin(q[4])) < 0.35:
            continue
        if abs(math.sin(q[2])) < 0.2:
            continue
        if sigma_min(q) < 0.05:
            continue
        return q


# --------------------------------------------------------------------------- #
# (a) FK <-> analytic IK round trip                                            #
# --------------------------------------------------------------------------- #
def test_a_fk_analytic_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = _rand_nonsingular_q(rng)
        T = fk(q)
        sols = ik_analytic(T)
        assert len(sols) >= 1
        # q must be present (circularly, per joint) among the analytic solutions.
        d = min(np.max(np.abs(_wrap(s - q))) for s in sols)
        assert d < 1e-9, f"q not recovered, closest err {d}"
        # every returned solution must reproduce the pose.
        for s in sols:
            assert np.linalg.norm(fk(s)[:3, 3] - T[:3, 3]) < 1e-9
            assert np.max(np.abs(fk(s)[:3, :3] - T[:3, :3])) < 1e-8


def test_a_analytic_enumerates_eight():
    """A generic well-conditioned pose has the full set of 8 branches."""
    rng = np.random.default_rng(7)
    got_eight = 0
    for _ in range(50):
        q = _rand_nonsingular_q(rng)
        sols = ik_analytic(fk(q))
        if len(sols) == 8:
            got_eight += 1
    assert got_eight > 40, f"only {got_eight}/50 poses enumerated 8 branches"


# --------------------------------------------------------------------------- #
# (b) numeric <-> analytic cross-check                                         #
# --------------------------------------------------------------------------- #
def test_b_numeric_matches_an_analytic_solution():
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = _rand_nonsingular_q(rng)
        T = fk(q)
        qn = ik_numeric(T, q + rng.uniform(-0.1, 0.1, 6))
        assert qn is not None
        sols = ik_analytic(T)
        d = min(np.max(np.abs(_wrap(qn - s))) for s in sols)
        assert d < 1e-6, f"numeric soln not in analytic set, err {d}"


def test_b_numeric_roundtrip_from_seed():
    rng = np.random.default_rng(11)
    for _ in range(100):
        q = _rand_nonsingular_q(rng)
        qn = ik_numeric(fk(q), q + rng.uniform(-0.05, 0.05, 6), tol=1e-10)
        assert qn is not None
        assert np.max(np.abs(_wrap(qn - q))) < 1e-8


# --------------------------------------------------------------------------- #
# (c) Jacobian vs central difference                                          #
# --------------------------------------------------------------------------- #
def test_c_jacobian_central_difference():
    rng = np.random.default_rng(2)
    h = 1e-6
    for _ in range(50):
        q = _rand_q(rng)
        J = jacobian(q)
        Jfd = np.zeros((6, 6))
        for i in range(6):
            qp = q.copy()
            qm = q.copy()
            qp[i] += h
            qm[i] -= h
            Tp, Tm = fk(qp), fk(qm)
            Jfd[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
            Jfd[3:, i] = so3_log(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)
        assert np.max(np.abs(J - Jfd)) < 1e-6


# --------------------------------------------------------------------------- #
# (d) sigma_min small at singularity, large in generic pose                   #
# --------------------------------------------------------------------------- #
def test_d_sigma_min_wrist_singularity():
    # wrist singularity: wrist_2 (index 4) == 0 -> rank drops.
    q_sing = np.array([0.1, -1.0, 1.0, 0.5, 0.0, 0.3])
    q_gen = np.array([0.3, -1.2, 1.1, -0.7, 1.3, 0.4])
    assert sigma_min(q_sing) < 1e-6
    assert sigma_min(q_gen) > 0.2
    assert sigma_min(q_sing) < sigma_min(q_gen)


def test_d_sigma_min_elbow_singularity():
    # elbow straight (index 2 == 0) is a boundary singularity.
    q_elbow = np.array([0.2, -0.8, 0.0, 0.4, 0.9, 0.1])
    q_gen = np.array([0.3, -1.2, 1.1, -0.7, 1.3, 0.4])
    assert sigma_min(q_elbow) < sigma_min(q_gen)


def test_d_sigma_min_uses_char_length_default():
    q = np.array([0.3, -1.2, 1.1, -0.7, 1.3, 0.4])
    assert sigma_min(q) == pytest.approx(sigma_min(q, L=CHAR_LENGTH))


# --------------------------------------------------------------------------- #
# (e) branch_id: invariance along a same-branch trajectory + determinism      #
# --------------------------------------------------------------------------- #
def test_e_branch_id_constant_on_smooth_trajectory():
    q0 = np.array([0.3, -1.2, 1.1, -0.7, 1.3, 0.4])
    b0 = branch_id(q0)
    assert 0 <= b0 <= 7
    # small smooth perturbations stay on the same branch.
    rng = np.random.default_rng(3)
    for _ in range(30):
        q = q0 + rng.uniform(-0.05, 0.05, 6)
        assert branch_id(q) == b0


def test_e_branch_id_deterministic_and_ordered():
    q = np.array([0.3, -1.2, 1.1, -0.7, 1.3, 0.4])
    T = fk(q)
    # ik_analytic returns branches in a fixed, deterministic order.
    a = ik_analytic(T)
    b = ik_analytic(T)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.allclose(x, y)
    # branch_id is deterministic.
    assert branch_id(q) == branch_id(q)


def test_e_distinct_branches_have_distinct_ids():
    q = np.array([0.3, -1.2, 1.1, -0.7, 1.3, 0.4])
    ids = {branch_id(s) for s in ik_analytic(fk(q))}
    # 8 distinct solutions -> 8 distinct branch labels.
    assert len(ids) == len(ik_analytic(fk(q)))


# --------------------------------------------------------------------------- #
# (f) within_joint_limits boundaries                                          #
# --------------------------------------------------------------------------- #
def test_f_joint_limits_elbow_boundary():
    q = np.zeros(6)
    q[2] = math.pi - 1e-9
    assert within_joint_limits(q)
    q[2] = math.pi + 1e-6
    assert not within_joint_limits(q)
    q[2] = -math.pi + 1e-9
    assert within_joint_limits(q)
    q[2] = -math.pi - 1e-6
    assert not within_joint_limits(q)


def test_f_joint_limits_other_axes_boundary():
    for j in (0, 1, 3, 4, 5):
        q = np.zeros(6)
        q[j] = 2 * math.pi - 1e-9
        assert within_joint_limits(q), f"axis {j} at +2pi-eps should pass"
        q[j] = 2 * math.pi + 1e-6
        assert not within_joint_limits(q), f"axis {j} above +2pi should fail"


def test_f_joint_limits_margin():
    q = np.zeros(6)
    q[2] = math.pi - 0.05
    assert within_joint_limits(q, margin=0.0)
    assert not within_joint_limits(q, margin=0.1)


# --------------------------------------------------------------------------- #
# (g) SE(3)/SO(3) log/exp round trips                                         #
# --------------------------------------------------------------------------- #
def test_g_se3_log_exp_roundtrip():
    rng = np.random.default_rng(4)
    for _ in range(500):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        ang = rng.uniform(0.0, math.pi - 1e-6)
        w = axis * ang
        v = rng.uniform(-1.0, 1.0, 3)
        xi = np.concatenate([v, w])
        T = se3_exp(xi)
        assert np.max(np.abs(se3_exp(se3_log(T)) - T)) < 1e-9
        assert np.max(np.abs(se3_log(T) - xi)) < 1e-9


def test_g_so3_roundtrip_and_edge_angles():
    rng = np.random.default_rng(5)
    for _ in range(500):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        ang = rng.uniform(0.0, math.pi - 1e-6)
        w = axis * ang
        R = so3_exp(w)
        assert np.max(np.abs(so3_exp(so3_log(R)) - R)) < 1e-9
    # theta -> 0
    w_small = np.array([1e-10, -2e-10, 3e-11])
    assert np.max(np.abs(so3_log(so3_exp(w_small)) - w_small)) < 1e-12
    # theta -> pi (each principal axis)
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        w = axis * (math.pi - 1e-7)
        R = so3_exp(w)
        assert np.max(np.abs(so3_exp(so3_log(R)) - R)) < 1e-8


def test_g_se3_exp_identity():
    assert np.allclose(se3_exp(np.zeros(6)), np.eye(4))
    assert np.allclose(so3_exp(np.zeros(3)), np.eye(3))


# --------------------------------------------------------------------------- #
# (h) matrix <-> quaternion (xyzw / ROS scalar-last)                          #
# --------------------------------------------------------------------------- #
def test_h_mat_quat_roundtrip():
    rng = np.random.default_rng(6)
    for _ in range(500):
        w = rng.uniform(-3, 3, 3)
        R = so3_exp(w)
        q = mat_to_quat_xyzw(R)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12
        assert np.max(np.abs(quat_xyzw_to_mat(q) - R)) < 1e-12


def test_h_quat_is_scalar_last_xyzw():
    # +90 deg about base z -> quaternion (0, 0, sin45, cos45) in xyzw order.
    R = so3_exp(np.array([0.0, 0.0, math.pi / 2]))
    q = mat_to_quat_xyzw(R)
    assert q[0] == pytest.approx(0.0, abs=1e-12)
    assert q[1] == pytest.approx(0.0, abs=1e-12)
    assert q[2] == pytest.approx(math.sin(math.pi / 4), abs=1e-12)  # z
    assert q[3] == pytest.approx(math.cos(math.pi / 4), abs=1e-12)  # w (last)


# --------------------------------------------------------------------------- #
# (i) keepout                                                                  #
# --------------------------------------------------------------------------- #
def test_i_keepout_floor_reject_and_accept():
    cfg = {"floor_z": -0.05, "margin": 0.0}
    # A folded-down pose that drives links well below the floor plane.
    q_low = np.array([0.0, 0.6, 1.5, 0.5, 0.0, 0.0])
    origins = link_origins(q_low)
    assert min(o[2] for o in origins) < -0.05  # sanity: really below floor
    assert not keepout_ok(q_low, cfg)
    # A raised pose that keeps every link above the floor.
    q_ok = np.array([0.0, -1.6, 1.2, -1.2, -1.57, 0.0])
    assert min(o[2] for o in link_origins(q_ok)) > -0.05
    assert keepout_ok(q_ok, cfg)


def test_i_keepout_margin_tightens_floor():
    # A pose whose lowest link sits just above the plane: passes at margin 0,
    # fails once a margin lifts the effective floor above it.
    q = np.array([0.0, 0.35, 1.5, 0.3, 0.0, 0.0])
    z_min = min(o[2] for o in link_origins(q))
    floor = z_min - 0.02
    assert keepout_ok(q, {"floor_z": floor, "margin": 0.0})
    assert not keepout_ok(q, {"floor_z": floor, "margin": 0.1})


def test_i_keepout_halfplane_and_base_cylinder():
    # Half-plane keepout: require every link origin on the +z side of a plane
    # below the base. The above-floor pose satisfies it, the folded-down one
    # violates it.
    hp = {"halfplane": {"point": [0, 0, -0.05], "normal": [0, 0, 1]}}
    q_ok = np.array([0.0, -1.6, 1.2, -1.2, -1.57, 0.0])
    q_low = np.array([0.0, 0.6, 1.5, 0.5, 0.0, 0.0])
    assert keepout_ok(q_ok, hp)
    assert not keepout_ok(q_low, hp)
    # Base cylinder (r=0.05, h=0.1): the shoulder sits on the base axis but
    # above the cylinder top (z=0.162 > 0.1), so a raised arm stays clear.
    cfg2 = {"base_cylinder": {"radius": 0.05, "height": 0.1}}
    assert keepout_ok(q_ok, cfg2)
    # A pose whose forearm passes through the base cylinder is rejected.
    cfg3 = {"base_cylinder": {"radius": 0.30, "height": 1.0}}
    assert not keepout_ok(q_ok, cfg3)


# --------------------------------------------------------------------------- #
# (j) wrist-singularity: unweighted nearest branch jumps >= 3 rad             #
# --------------------------------------------------------------------------- #
def test_j_wrist_singularity_branch_jump():
    """Sweeping smoothly through the wrist singularity, an UNWEIGHTED
    nearest-of-8 branch selector jumps >= 3.0 rad -- the very failure the
    branch weighting W exists to prevent."""
    import itertools

    base = np.array([0.4, -1.1, 1.0, -0.6, 0.05, 0.5])
    sols = ik_analytic(fk(base))
    assert len(sols) == 8
    # The wrist-flip partner pair: the two branches farthest apart in joint
    # space at this near-singular pose (they differ ~pi in wrist_1 & wrist_3).
    i, j = max(
        itertools.combinations(range(len(sols)), 2),
        key=lambda ij: np.linalg.norm(_wrap(sols[ij[0]] - sols[ij[1]])),
    )
    a, b = sols[i], sols[j]
    assert np.linalg.norm(_wrap(a - b)) >= 3.0
    # Seed placed near the circular midpoint -> the nearest branch switches as
    # the pose sweeps through q5 = 0.
    seed = a + _wrap(b - a) / 2.0

    prev = None
    max_jump = 0.0
    for q5 in np.linspace(0.12, -0.12, 241):
        q = base.copy()
        q[4] = q5
        ss = ik_analytic(fk(q))
        if len(ss) < 2:
            continue
        nearest = min(ss, key=lambda s: np.linalg.norm(_wrap(s - seed)))
        if prev is not None:
            max_jump = max(max_jump, float(np.linalg.norm(_wrap(nearest - prev))))
        prev = nearest
    assert max_jump >= 3.0, f"expected >=3 rad branch jump, saw {max_jump}"


# --------------------------------------------------------------------------- #
# (k) worst-case tick timing (informational, no hard assert)                  #
# --------------------------------------------------------------------------- #
def _near_singular_q(rng):
    """A config deliberately parked next to the wrist/elbow singularity -- the
    worst case for the analytic enumerate + DLS refine hot path."""
    q = _rand_q(rng)
    q[4] = rng.uniform(-0.01, 0.01)  # wrist_2 ~ 0 -> ill-conditioned
    return q


def _hot_path_tick(T, seed):
    """The analytic branch-lock hot path as eef_delta actually runs it:
    closed-form enumerate (tagged) + Jacobian conditioning + wrapped-nearest
    branch selection. NO redundant full ik_numeric DLS solve -- that per-tick
    refine was removed (B3); the closed form already lands ~1e-15."""
    tagged = _ik_analytic_tagged(T)
    _ = sigma_min(seed)
    if tagged:
        _ = min((wrapped_nearest(q, seed) for _, q in tagged),
                key=lambda s: float(np.max(np.abs(s - seed))))


def test_k_worst_case_tick_timing(capsys):
    """Worst-case control-tick timing, INCLUDING near-singular and unreachable
    poses (the cases that used to blow the 250 Hz / 4 ms budget). Informational:
    prints and soft-warns, does not hard-assert on wall-clock (CI-machine
    dependent)."""
    BUDGET_MS = 4.0  # 250 Hz control loop
    rng = np.random.default_rng(9)

    # A worst-case mix: generic, near-singular, and outright unreachable targets.
    cases = []
    for _ in range(150):
        cases.append(fk(_rand_nonsingular_q(rng)))
    for _ in range(150):
        cases.append(fk(_near_singular_q(rng)))
    for _ in range(50):
        Tu = np.eye(4)
        Tu[:3, 3] = [rng.uniform(1.5, 3.0), rng.uniform(1.5, 3.0), 1.0]  # out of reach
        cases.append(Tu)

    # Warm up (first calls pay one-time numpy/import costs, not per-tick cost).
    for _ in range(30):
        _hot_path_tick(cases[0], np.zeros(6))

    worst = 0.0
    worst_kind = ""
    for idx, T in enumerate(cases):
        seed = rng.uniform(-0.3, 0.3, 6)
        t0 = time.perf_counter()
        _hot_path_tick(T, seed)
        dt = (time.perf_counter() - t0) * 1e3
        if dt > worst:
            worst = dt
            worst_kind = ("generic" if idx < 150 else
                          "near-singular" if idx < 300 else "unreachable")

    with capsys.disabled():
        msg = (f"[timing] worst-case tick = {worst:.3f} ms "
               f"(on a {worst_kind} pose; budget {BUDGET_MS:.1f} ms @250Hz)")
        if worst > BUDGET_MS:
            msg += "  WARNING: exceeds budget"
        print("\n" + msg)


# --------------------------------------------------------------------------- #
# DH loader (optional pyyaml)                                                  #
# --------------------------------------------------------------------------- #
def test_load_dh_returns_dict():
    cfg = load_dh()
    assert isinstance(cfg, dict)
    assert len(cfg) > 0


# --------------------------------------------------------------------------- #
# (B1) +/-2pi unwrap: branch lock re-anchors to the LITERAL nearest solution   #
# --------------------------------------------------------------------------- #
def test_b1_wrapped_nearest_literal_recovery():
    """ik_analytic wraps every solution to (-pi, pi]; wrapped_nearest must undo
    that toward the seed with a LITERAL (non-circular) distance, so a pose whose
    joints live in (pi, 2pi] is recovered exactly -- not hidden behind a
    circular comparison."""
    q = np.array([4.0, 4.5, 1.0, 3.5, 4.2, 5.5])  # non-elbow joints in (pi, 2pi]
    assert all(math.pi < q[i] <= 2 * math.pi for i in (0, 1, 3, 4, 5))

    sols = ik_analytic(fk(q))
    assert len(sols) >= 1
    # Every wrapped solution is inside (-pi, pi], so a raw literal comparison is
    # far from q (this is exactly the trap circular comparison would hide).
    d_raw = min(float(np.max(np.abs(s - q))) for s in sols)
    assert d_raw > 1.0, f"expected wrapped solutions far from q, got {d_raw}"

    # Re-anchoring (the branch-lock selector step) restores the literal value.
    best = min((wrapped_nearest(s, q) for s in sols),
               key=lambda s: float(np.max(np.abs(s - q))))
    d_literal = float(np.max(np.abs(best - q)))
    assert d_literal < 1e-9, f"literal recovery failed, err {d_literal}"
    # And it must still be a valid IK solution for the pose.
    assert np.linalg.norm(_pose_error(fk(best), fk(q))) < 1e-8


def test_b1_wrapped_nearest_leaves_elbow():
    # The elbow (index 2) must never be unwrapped past +/-pi.
    sol = np.array([0.0, 0.0, 2.5, 0.0, 0.0, 0.0])
    ref = np.array([0.0, 0.0, 2.5 - 2 * math.pi, 0.0, 0.0, 0.0])
    out = wrapped_nearest(sol, ref)
    assert out[2] == pytest.approx(2.5)  # unchanged despite ref being 2pi away


# --------------------------------------------------------------------------- #
# (B2) absolute FK ground truth (independently hand-derived UR5e/UR7e values)  #
# --------------------------------------------------------------------------- #
def test_b2_fk_ground_truth_zero_pose():
    """fk(0) cross-checked against the canonical UR5e/UR7e zero-config pose,
    derived independently from the DH table (NOT copied from fk output):
        x = a2 + a3 = -0.8172,  y = -(d4 + d6) = -0.2329,  z = d1 - d5 = 0.0628.
    This is the only guard against a DH transcription typo (round-trip tests
    stay green even with a wrong-but-consistent DH constant)."""
    T = fk(np.zeros(6))
    assert np.allclose(T[:3, 3], [-0.8172, -0.2329, 0.0628], atol=1e-9)
    # Orientation at zero: tool0 x-axis along base x, with a Rx(90) twist.
    R_expected = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0]])
    assert np.allclose(T[:3, :3], R_expected, atol=1e-12)


def test_b2_fk_ground_truth_base_rotated():
    """theta1 = +90 deg rotates the whole zero pose about base z:
        Rz(90) @ (-0.8172, -0.2329, 0.0628) = (0.2329, -0.8172, 0.0628)."""
    T = fk(np.array([math.pi / 2, 0, 0, 0, 0, 0]))
    assert np.allclose(T[:3, 3], [0.2329, -0.8172, 0.0628], atol=1e-9)


# --------------------------------------------------------------------------- #
# (B4) keepout segment sampling: a link piercing a cylinder is caught          #
# --------------------------------------------------------------------------- #
def test_b4_keepout_cylinder_midlink_piercing():
    """Both endpoints of a link lie OUTSIDE a keepout cylinder but the link
    passes straight through it. Sampling only the 6 link origins (old behavior)
    reads 'safe'; sampling the segments must reject."""
    q = np.array([0.3, -1.0, 1.0, -0.7, 1.2, 0.4])
    origins = link_origins(q)
    a, b = origins[1], origins[2]           # the elbow -> wrist_1 link
    m = 0.5 * (a + b)
    seg_u = (b - a) / np.linalg.norm(b - a)
    # A cylinder axis perpendicular to the link, threaded through its midpoint.
    tmp = np.array([0.0, 0.0, 1.0]) if abs(seg_u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    axis = np.cross(seg_u, tmp)
    axis = axis / np.linalg.norm(axis)
    # Radius chosen just under the nearest ORIGIN so all 6 origins stay outside.
    perps = [float(np.linalg.norm((o - m) - np.dot(o - m, axis) * axis)) for o in origins]
    r = min(perps) - 1e-6
    assert r > 0
    cy = {"point": m.tolist(), "axis": axis.tolist(), "radius": r}

    def origins_only_ok():
        for o in origins:
            d = o - m
            perp = d - np.dot(d, axis) * axis
            if np.linalg.norm(perp) < r:
                return False
        return True

    assert origins_only_ok()                         # old origins-only: SAFE (wrong)
    assert not keepout_ok(q, {"cylinder": cy})       # segment sweep: REJECT (right)


def test_b4_keepout_cylinder_fully_clear_still_ok():
    # A far-away cylinder must not spuriously reject after the sampling change.
    q = np.array([0.3, -1.0, 1.0, -0.7, 1.2, 0.4])
    cy = {"point": [10.0, 10.0, 10.0], "axis": [0, 0, 1], "radius": 0.1}
    assert keepout_ok(q, {"cylinder": cy})


# --------------------------------------------------------------------------- #
# (B5) load_dh: single normalized FLAT schema with concrete values            #
# --------------------------------------------------------------------------- #
def test_b5_load_dh_flat_schema_and_values():
    cfg = load_dh()
    for k in ("d", "a", "alpha", "joint_limits", "velocity_limits",
              "char_length", "branch_weights"):
        assert k in cfg, f"missing key {k}"
    assert "dh" not in cfg, "schema must be flattened (no nested 'dh' block)"

    assert cfg["d"][0] == pytest.approx(0.1625)
    assert cfg["d"][3] == pytest.approx(0.1333)
    assert cfg["d"][5] == pytest.approx(0.0996)
    assert cfg["a"][1] == pytest.approx(-0.425)
    assert cfg["a"][2] == pytest.approx(-0.3922)
    assert cfg["alpha"][0] == pytest.approx(math.pi / 2)
    assert cfg["alpha"][4] == pytest.approx(-math.pi / 2)
    assert cfg["char_length"] == pytest.approx(0.30)
    lo, hi = cfg["joint_limits"][2]
    assert lo == pytest.approx(-math.pi) and hi == pytest.approx(math.pi)  # elbow +/-pi
    assert list(cfg["branch_weights"]) == [2.0, 2.0, 1.5, 1.0, 1.0, 0.5]


# --------------------------------------------------------------------------- #
# (B6) ik_numeric tol contract: no silent 1e-6 floor                          #
# --------------------------------------------------------------------------- #
def test_b6_ik_numeric_unreachable_returns_none():
    T = np.eye(4)
    T[:3, 3] = [5.0, 0.0, 0.0]  # far outside the ~0.82 m reach
    assert ik_numeric(T, np.zeros(6)) is None


def test_b6_ik_numeric_tol_not_floored_at_1e6():
    """Reviewer's reproduction: a near-singular target where DLS plateaus at
    ~2e-8 within max_iter. The OLD gate max(tol, 1e-6) silently returned that
    ~2e-8 solution for a tol=1e-9 request; the contract requires either a
    solution meeting the REQUESTED tol, or None."""
    q = np.array([-1.301907, -0.406081, 1.523366, 0.763349, -0.002459, -0.530849])
    seed = np.array([-1.206123, -0.265087, 1.580806, 0.994044, 0.02298, -0.671343])
    T = fk(q)
    qn = ik_numeric(T, seed, tol=1e-9)
    assert qn is None or np.linalg.norm(_pose_error(fk(qn), T)) <= 1e-9


def test_b6_ik_numeric_returned_solution_always_meets_tol():
    rng = np.random.default_rng(21)
    for _ in range(60):
        q = _rand_nonsingular_q(rng)
        T = fk(q)
        tol = 1e-10
        qn = ik_numeric(T, q + rng.uniform(-0.05, 0.05, 6), tol=tol)
        if qn is not None:
            assert np.linalg.norm(_pose_error(fk(qn), T)) <= tol


# --------------------------------------------------------------------------- #
# (C6) branch_id: the 3 returned bits ARE the (theta1, theta5, theta3) sels    #
# --------------------------------------------------------------------------- #
def test_c6_branch_id_bits_are_real_selectors():
    q = np.array([0.3, -1.2, 1.1, -0.7, 1.3, 0.4])
    tagged = _ik_analytic_tagged(fk(q))
    assert len(tagged) == 8
    # All 8 bit-combinations appear exactly once -> the 3 bits are independent
    # selectors, not a nearest-label collision.
    assert sorted(b for b, _ in tagged) == list(range(8))

    for bid, sol in tagged:
        # branch_id of a solution reproduces its own enumerated tag.
        assert branch_id(sol) == bid
        # bit0 = theta3 selector (+a3ang -> 0, -a3ang -> 1).
        assert (bid & 1) == (0 if sol[2] >= 0 else 1)
        # bit1 = theta5 selector (+acos -> 0, -acos -> 1).
        assert ((bid >> 1) & 1) == (0 if sol[4] >= 0 else 1)

    # bit2 = theta1 selector: the two theta1 branches partition the 8 solutions
    # into two groups, each sharing one theta1 value.
    by_t1 = {}
    for bid, sol in tagged:
        by_t1.setdefault((bid >> 2) & 1, []).append(sol[0])
    assert set(by_t1.keys()) == {0, 1}
    for vals in by_t1.values():
        assert max(vals) - min(vals) < 1e-6
