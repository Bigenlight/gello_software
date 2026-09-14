#!/usr/bin/env python3
"""Adversarial tests for ``gello_policy.eef_space`` (A7, 2026-09-14).

WHY these tests exist
---------------------
``eef_space.py`` is the SINGLE SOURCE OF TRUTH shared by the dataset converter
(py3.12, lr_env, no ROS) and the deploy node (py3.10, rclpy). If its arithmetic
and the deploy node's arithmetic ever disagree, nothing crashes -- the robot
just executes a different action space than the one the policy was trained on.
So every convention is pinned here explicitly rather than assumed:

* the 6-D rotation is the first two COLUMNS of R (not rows), and its
  Gram-Schmidt inverse must reproduce R exactly;
* the action rotation is LEFT-multiplied in the base frame
  (``dR = R_b @ R_a.T``), and ``apply_delta`` must invert exactly that;
* ``ik_tcp_nearest`` must mirror ``ur_gello_bringup.eef_delta`` closely enough
  that the branch handling near the elbow / wrist_2 merge points agrees;
* the clamps must be norm-proportional (direction preserving) -- see CLAUDE.md
  "개입 액션을 축별 np.clip으로 자르지 말 것" -- and their edge semantics
  (zero cap, NaN) must be pinned, because both are silently FAIL-OPEN.

Run (system python3, no ROS overlay needed -- both packages are ROS-free)::

    cd /home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy
    PYTHONPATH=/home/laptop3/gello_software_humble/ros2_ur_ws/src/ur_gello_bringup:\
/home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy \
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
      -p no:cacheprovider test/test_eef_space.py

Tests that need the raw corpus skip themselves when it is absent.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from gello_policy import eef_space as es
from ur_gello_bringup import ur_kin

# --------------------------------------------------------------------------- #
# raw-take fixtures                                                           #
# --------------------------------------------------------------------------- #
RAW_ROOT = "/home/laptop3/youngwoong_ws/Put_carrot_in_pot"
# One early, one middle, one LATE take (the late ones are the tail of the
# session, where the operator was fastest and the arm sat on the v_max
# governor most often -- the worst case for the round trip).
TAKE_GLOBS = ("take_01_", "take_30_", "take_55_")


def _find_take(prefix: str):
    if not os.path.isdir(RAW_ROOT):
        return None
    for name in sorted(os.listdir(RAW_ROOT)):
        if name.startswith(prefix):
            p = os.path.join(RAW_ROOT, name, "vectors.h5")
            if os.path.isfile(p):
                return p
    return None


def _load_q(path: str) -> np.ndarray:
    """(N, 6) measured joint trajectory of one take."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r") as f:
        g = f["ur_joint_states"]
        return np.stack([np.asarray(g[f"q{i}"]) for i in range(1, 7)], axis=1)


@pytest.fixture(scope="module")
def takes():
    out = []
    for pre in TAKE_GLOBS:
        p = _find_take(pre)
        if p is not None:
            out.append((pre.rstrip("_"), _load_q(p)))
    if len(out) < 3:
        pytest.skip(f"need >=3 raw takes under {RAW_ROOT}; found {len(out)}")
    return out


def _sample_pairs(Q: np.ndarray, n: int = 60):
    """n (q_t, q_{t+1}) pairs spread over the take."""
    idx = np.linspace(0, len(Q) - 2, n).astype(int)
    return [(Q[i], Q[i + 1]) for i in idx]


def _rand_rot(rng) -> np.ndarray:
    return ur_kin.so3_exp(rng.normal(size=3) * rng.uniform(0.0, math.pi))


# --------------------------------------------------------------------------- #
# 1. contract: dims, names, ids, tool transform                               #
# --------------------------------------------------------------------------- #
def test_state_and_action_dims_and_names():
    assert es.STATE_DIM == 16 and len(es.STATE_NAMES) == 16
    assert es.ACTION_DIM == 7 and len(es.ACTION_NAMES) == 7
    assert es.ACTION_SPACE_ID == "eef_delta_v1"
    # DESIGN §2 order, verbatim.
    assert es.STATE_NAMES == [
        "ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6",
        "tcp_x", "tcp_y", "tcp_z",
        "tcp_r11", "tcp_r21", "tcp_r31", "tcp_r12", "tcp_r22", "tcp_r32",
        "grip_pos",
    ]
    assert es.ACTION_NAMES == ["dx", "dy", "dz", "drx", "dry", "drz", "grip_cmd"]
    assert len(set(es.STATE_NAMES)) == 16 and len(set(es.ACTION_NAMES)) == 7


def test_tool_transform_is_self_consistent():
    assert es.TOOL_Z_M == pytest.approx(0.174)
    np.testing.assert_allclose(es.T_TOOL @ es.T_TOOL_INV, np.eye(4), atol=1e-15)
    # pure +Z translation, no rotation
    np.testing.assert_allclose(es.T_TOOL[:3, :3], np.eye(3), atol=0.0)
    np.testing.assert_allclose(es.T_TOOL[:3, 3], [0.0, 0.0, es.TOOL_Z_M], atol=0.0)


def test_tcp_pose_is_fk_plus_tool_along_flange_z():
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    T_f = ur_kin.fk(q)
    T_t = es.tcp_pose(q)
    np.testing.assert_allclose(T_t[:3, :3], T_f[:3, :3], atol=1e-15)
    np.testing.assert_allclose(
        T_t[:3, 3], T_f[:3, 3] + es.TOOL_Z_M * T_f[:3, 2], atol=1e-15
    )


# --------------------------------------------------------------------------- #
# 2. the 6-D rotation                                                         #
# --------------------------------------------------------------------------- #
def test_rot_to_6d_takes_columns_not_rows():
    # A matrix whose rows and columns differ, so a row/column mix-up is fatal.
    R = ur_kin.so3_exp(np.array([0.3, -0.7, 1.1]))
    v = es.rot_to_6d(R)
    assert v.shape == (6,)
    np.testing.assert_allclose(v, [R[0, 0], R[1, 0], R[2, 0],
                                   R[0, 1], R[1, 1], R[2, 1]], atol=0.0)
    # and the names say the same thing
    assert es.STATE_NAMES[9:15] == [
        "tcp_r11", "tcp_r21", "tcp_r31", "tcp_r12", "tcp_r22", "tcp_r32"]


def test_rot_from_6d_inverts_rot_to_6d_exactly():
    rng = np.random.default_rng(7)
    for _ in range(200):
        R = _rand_rot(rng)
        np.testing.assert_allclose(es.rot_from_6d(es.rot_to_6d(R)), R, atol=1e-12)


def test_rot_6d_round_trip_on_real_takes(takes):
    worst = 0.0
    for _name, Q in takes:
        for q in Q[:: max(1, len(Q) // 50)]:
            R = es.tcp_pose(q)[:3, :3]
            worst = max(worst, float(np.max(np.abs(es.rot_from_6d(es.rot_to_6d(R)) - R))))
    assert worst < 1e-12, worst


def test_carrot_tcp_is_near_180_deg_so_the_6d_encoding_earns_its_keep(takes):
    """DESIGN §2 justifies 6-D over a rotvec by "~175 deg from identity"; pin it."""
    angs = []
    for _name, Q in takes:
        for q in Q[:: max(1, len(Q) // 20)]:
            angs.append(float(np.linalg.norm(ur_kin.so3_log(es.tcp_pose(q)[:3, :3]))))
    assert min(angs) > math.radians(150.0), min(angs)


def test_rot_from_6d_reorthonormalises_a_perturbed_input():
    rng = np.random.default_rng(11)
    R = _rand_rot(rng)
    v = es.rot_to_6d(R) + rng.normal(size=6) * 1e-3
    M = es.rot_from_6d(v)
    np.testing.assert_allclose(M.T @ M, np.eye(3), atol=1e-12)
    assert np.linalg.det(M) == pytest.approx(1.0, abs=1e-12)


def test_rot_from_6d_degenerate_input_is_not_silently_a_rotation():
    """a1 parallel to a2 -> Gram-Schmidt divides by zero.

    PINNED BEHAVIOUR, not an endorsement: the result is non-finite rather than a
    plausible-looking wrong rotation. Only matters if a network is ever made to
    PREDICT 6-D rotations (today the action carries a rotvec and the state's
    6-D always comes from a real R), in which case the caller must normalise.
    """
    a = np.array([1.0, 0.0, 0.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        M = es.rot_from_6d(np.concatenate([a, 2.0 * a]))
        M0 = es.rot_from_6d(np.zeros(6))
    assert not np.all(np.isfinite(M))
    assert not np.all(np.isfinite(M0))


# --------------------------------------------------------------------------- #
# 3. state vector                                                             #
# --------------------------------------------------------------------------- #
def test_state_vector_layout_and_dtype():
    q = [-3.193, -1.490, 1.693, -1.798, -1.590, -3.269]
    s = es.state_vector(q, 0.25)
    assert s.shape == (16,)
    # NOTE the module header says "state : (16,) float32" but the function
    # returns float64. Pinned so that whoever changes it notices the converter
    # is the one that must cast (lerobot wants float32).
    assert s.dtype == np.float64
    T = es.tcp_pose(q)
    np.testing.assert_allclose(s[:6], q, atol=0.0)
    np.testing.assert_allclose(s[6:9], T[:3, 3], atol=0.0)
    np.testing.assert_allclose(s[9:15], es.rot_to_6d(T[:3, :3]), atol=0.0)
    assert s[15] == pytest.approx(0.25)


def test_state_vector_accepts_lists_tuples_and_arrays():
    q = [-3.193, -1.490, 1.693, -1.798, -1.590, -3.269]
    a = es.state_vector(q, 0.0)
    b = es.state_vector(tuple(q), 0.0)
    c = es.state_vector(np.asarray(q, dtype=np.float32), 0.0)
    np.testing.assert_allclose(a, b, atol=0.0)
    np.testing.assert_allclose(a, c, atol=1e-6)


def test_tcp_matches_the_recorder_tcp_pose_table(takes):
    """TOOL_Z_M cross-check against the robot's own TCP broadcaster.

    The recorder's ``tcp_pose`` table is ~0.50 s STALER than ``ur_joint_states``
    (both are starved subscriptions with different QoS depths -- see
    docs/ros2/CARROT_EEF_LAG_NOTE_KO.md), so the two tables are compared on the
    joint clock shifted by that offset, not row-for-row.
    """
    h5py = pytest.importorskip("h5py")
    path = _find_take(TAKE_GLOBS[0])
    if path is None:
        pytest.skip("raw take missing")
    with h5py.File(path, "r") as f:
        tq = np.asarray(f["ur_joint_states"]["t_rel_s"])
        Q = np.stack([np.asarray(f["ur_joint_states"][f"q{i}"]) for i in range(1, 7)], 1)
        tp = np.asarray(f["tcp_pose"]["t_rel_s"])
        P = np.stack([np.asarray(f["tcp_pose"][k]) for k in ("x", "y", "z")], 1)
    shift = 0.50
    sel = (tq - shift >= tp[0]) & (tq - shift <= tp[-1])
    tq, Q = tq[sel], Q[sel]
    ours = np.array([es.tcp_pose(q)[:3, 3] for q in Q[::5]])
    theirs = np.stack([np.interp(tq[::5] - shift, tp, P[:, j]) for j in range(3)], 1)
    err = np.linalg.norm(ours - theirs, axis=1)
    assert float(np.median(err)) < 0.003, float(np.median(err))


# --------------------------------------------------------------------------- #
# 4. action: delta_action / apply_delta are exact inverses                     #
# --------------------------------------------------------------------------- #
def test_apply_delta_inverts_delta_action_on_real_takes(takes):
    worst_p = worst_r = 0.0
    for _name, Q in takes:
        for qa, qb in _sample_pairs(Q):
            a = es.delta_action(qa, qb, 0.5)
            T_got = es.apply_delta(qa, a)
            T_want = es.tcp_pose(qb)
            worst_p = max(worst_p, float(np.max(np.abs(T_got[:3, 3] - T_want[:3, 3]))))
            worst_r = max(worst_r, float(np.max(np.abs(T_got[:3, :3] - T_want[:3, :3]))))
    assert worst_p < 1e-12, worst_p
    assert worst_r < 1e-12, worst_r


def test_action_rotation_is_left_multiplied_in_the_base_frame(takes):
    """dR = R_b @ R_a.T (base frame), NOT R_a.T @ R_b (tool frame).

    Distinguishable only when R_a is not identity -- which is always true here
    (the carrot TCP sits ~175 deg from identity), so a convention flip would be
    caught by this test and by nothing else in the round trip.
    """
    _name, Q = takes[0]
    qa, qb = Q[100], Q[130]
    Ra = es.tcp_pose(qa)[:3, :3]
    Rb = es.tcp_pose(qb)[:3, :3]
    drot = es.delta_action(qa, qb, 0.0)[3:6]
    np.testing.assert_allclose(ur_kin.so3_exp(drot), Rb @ Ra.T, atol=1e-12)
    # the tool-frame alternative really is a different vector here
    tool_frame = ur_kin.so3_log(Ra.T @ Rb)
    assert np.linalg.norm(drot - tool_frame) > 1e-6


def test_delta_action_of_identical_poses_is_zero_except_the_gripper():
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    a = es.delta_action(q, q, 0.77)
    np.testing.assert_allclose(a[:6], np.zeros(6), atol=1e-15)
    assert a[6] == pytest.approx(0.77)
    assert a.shape == (7,)


def test_delta_action_gripper_is_absolute_passthrough():
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    for g in (0.0, 0.3, 1.0):
        assert es.delta_action(q, q, g)[6] == pytest.approx(g)


def test_apply_delta_ignores_the_gripper_channel():
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    a = np.array([0.01, -0.02, 0.003, 0.01, 0.0, -0.02, 0.0])
    b = a.copy()
    b[6] = 1.0
    np.testing.assert_allclose(es.apply_delta(q, a), es.apply_delta(q, b), atol=0.0)


def test_apply_delta_accepts_a_6_vector_too():
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    a7 = np.array([0.01, -0.02, 0.003, 0.01, 0.0, -0.02, 0.5])
    np.testing.assert_allclose(es.apply_delta(q, a7), es.apply_delta(q, a7[:6]), atol=0.0)


def test_apply_delta_returns_a_valid_homogeneous_transform():
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    T = es.apply_delta(q, np.array([0.01, -0.02, 0.003, 0.05, -0.01, 0.02, 0.0]))
    np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=0.0)
    np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# 5. ik_tcp_nearest                                                           #
# --------------------------------------------------------------------------- #
def test_ik_round_trip_recovers_the_next_frame_on_real_takes(takes):
    """dataset action -> apply_delta -> IK must return q_{t+1} (DESIGN §1)."""
    worst = 0.0
    fails = 0
    branches = set()
    n = 0
    for _name, Q in takes:
        branch = ur_kin.branch_id(Q[0])
        branches.add(branch)
        for qa, qb in _sample_pairs(Q, 40):
            T = es.apply_delta(qa, es.delta_action(qa, qb, 0.0))
            q, info = es.ik_tcp_nearest(T, qa, branch=branch)
            n += 1
            if q is None:
                fails += 1
                continue
            assert info["on_branch"] is True
            assert info["branch"] == branch
            assert ur_kin.within_joint_limits(q), q
            worst = max(worst, float(np.max(np.abs(q - qb))))
    assert fails == 0, fails
    assert worst < 1e-9, worst
    assert branches == {2}, branches  # DESIGN §1: every frame is branch 2
    assert n >= 120


def test_ik_seeded_at_the_target_is_the_identity(takes):
    _name, Q = takes[0]
    for q in Q[:: max(1, len(Q) // 25)]:
        got, info = es.ik_tcp_nearest(es.tcp_pose(q), q, branch=ur_kin.branch_id(q))
        assert got is not None
        np.testing.assert_allclose(got, q, atol=1e-9)
        assert info["max_dev"] < 1e-9


def test_ik_unreachable_target_returns_none_not_an_exception():
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    T = es.tcp_pose(q).copy()
    T[0, 3] += 5.0  # far outside the workspace
    got, info = es.ik_tcp_nearest(T, q, branch=2)
    assert got is None
    assert info["n_solutions"] == 0
    assert info["branch"] == -1 and info["on_branch"] is False
    assert math.isnan(info["max_dev"])


def test_ik_nan_target_returns_none_not_an_exception():
    """A non-finite action must degrade to "hold", never to a wild joint target."""
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    T = es.tcp_pose(q).copy()
    T[0, 3] = np.nan
    got, _info = es.ik_tcp_nearest(T, q, branch=2)
    assert got is None


def test_ik_n_solutions_is_branch_filtered_when_a_branch_is_requested(takes):
    """``n_solutions`` counts what the solver RETURNED, not the reachable set.

    With ``branch=`` given, ur_kin._ik_analytic_tagged only verifies that branch,
    so n_solutions is 1 -- it is NOT a conditioning/reachability metric. Pinned
    so nobody builds a singularity check on it.
    """
    _name, Q = takes[0]
    q = Q[len(Q) // 2]
    T = es.tcp_pose(q)
    _, info_b = es.ik_tcp_nearest(T, q, branch=ur_kin.branch_id(q))
    _, info_all = es.ik_tcp_nearest(T, q, branch=None)
    assert info_b["n_solutions"] == 1
    assert info_all["n_solutions"] > 1
    assert info_all["on_branch"] is False  # no branch requested -> flag stays False


def test_ik_branch_none_and_negative_behave_the_same():
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    T = es.tcp_pose(q)
    a, ia = es.ik_tcp_nearest(T, q, branch=None)
    b, ib = es.ik_tcp_nearest(T, q, branch=-1)
    np.testing.assert_allclose(a, b, atol=0.0)
    assert ia["n_solutions"] == ib["n_solutions"]


def test_merge_point_test_agrees_with_eef_delta_branchlock():
    """eef_space uses np.remainder(x+pi, 2pi)-pi; eef_delta uses math.remainder.

    They must select the same joints as "at a merge point", or the two stacks
    would disagree about when branch locking is suspended.

    The two wrapped values agree to ~1e-15 everywhere; the only place the
    BOOLEANS can differ is within 1 ULP of the 0.05 threshold itself (measured:
    at x == 0.05 exactly, np.remainder gives 0.049999999999999822 while
    math.remainder gives 0.05). That is a tie at the threshold, not a
    behavioural difference, so it is excluded explicitly rather than hidden.
    """
    boundary_ties = 0
    for x in np.concatenate([
        np.linspace(-7.0, 7.0, 2801),
        np.array([0.0, 0.05, -0.05, math.pi, -math.pi, 2 * math.pi, -2 * math.pi]),
    ]):
        mine_val = float(np.remainder(x + np.pi, 2 * np.pi) - np.pi)
        theirs_val = math.remainder(float(x), 2 * math.pi)
        # the wrapped values themselves must agree (up to the +/-pi endpoint,
        # where the two conventions pick opposite signs of the same angle)
        assert abs(abs(mine_val) - abs(theirs_val)) < 1e-9, (x, mine_val, theirs_val)
        mine, theirs = abs(mine_val) < 0.05, abs(theirs_val) < 0.05
        if mine != theirs:
            boundary_ties += 1
            assert abs(abs(theirs_val) - 0.05) < 1e-12, (x, mine_val, theirs_val)
            continue
    assert boundary_ties <= 2, boundary_ties


def test_branch_filter_is_suspended_at_a_merge_point():
    """At elbow ~ 0 the branch labels collapse; ik_tcp_nearest must not lock."""
    q = np.array([-3.19, -1.49, 0.01, -1.80, -1.59, -3.27])  # elbow at a merge point
    T = es.tcp_pose(q)
    got, info = es.ik_tcp_nearest(T, q, branch=2)
    assert got is not None
    assert info["on_branch"] is False  # suspended -> full enumeration
    np.testing.assert_allclose(got, q, atol=1e-6)


def test_wrapped_nearest_never_unwraps_the_elbow_but_does_unwrap_carrot_pan():
    """carrot's shoulder_pan ~ -3.19 and wrist_3 down to -4.41 are OUTSIDE
    (-pi, pi], so IK's wrap must be undone on those joints; the elbow must not
    be touched (its physical range is +/-pi)."""
    seed = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -4.410])
    wrapped = ur_kin._wrap_pi(seed.copy())
    out = ur_kin.wrapped_nearest(wrapped, seed)
    np.testing.assert_allclose(out[[0, 1, 3, 4, 5]], seed[[0, 1, 3, 4, 5]], atol=1e-12)
    assert out[2] == wrapped[2]  # elbow left alone (already inside +/-pi here)
    assert ur_kin.within_joint_limits(out)


def test_ik_recovers_the_carrot_extremes_of_pan_and_wrist3(takes):
    """The unwrap rule has to survive the real corpus extremes, not just the mean."""
    qs = []
    for _name, Q in takes:
        qs.append(Q[int(np.argmin(Q[:, 0]))])
        qs.append(Q[int(np.argmin(Q[:, 5]))])
        qs.append(Q[int(np.argmax(Q[:, 5]))])
    for q in qs:
        got, info = es.ik_tcp_nearest(es.tcp_pose(q), q, branch=ur_kin.branch_id(q))
        assert got is not None
        np.testing.assert_allclose(got, q, atol=1e-9)
        assert ur_kin.within_joint_limits(got), got
        assert info["max_dev"] < 1e-9


def test_ik_limit_filter_keeps_the_true_solution_near_a_joint_limit():
    """A seed whose unwrapped neighbours straddle +/-2pi must still resolve."""
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -6.0])
    got, _ = es.ik_tcp_nearest(es.tcp_pose(q), q, branch=ur_kin.branch_id(q))
    assert got is not None
    np.testing.assert_allclose(got, q, atol=1e-9)
    assert ur_kin.within_joint_limits(got)


def test_ik_flags_a_q_OUTSIDE_the_joint_limits():
    """FINDING (A7, medium): the limit filter is "keep the survivors IF ANY".

    When every candidate is out of limits the filter falls back to the
    unfiltered pool, so ``ik_tcp_nearest`` returns an ILLEGAL joint vector with
    ``on_branch=True`` and a small ``max_dev`` -- nothing in ``info`` says the
    result is outside ``ur_kin.JOINT_LIMITS``. Here wrist_3 = -6.30 rad is
    0.017 rad past -2*pi.

    Downstream this is caught by DESIGN §3 step 6 (the ``joint_limits_lo/hi``
    envelope from A2), so it is not a live hazard for the carrot deploy -- but
    any caller that trusts a non-None return is unguarded. Suggested fix:
    add ``info["in_limits"] = within_joint_limits(best)``.
    """
    seed = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -6.25])
    tgt = seed.copy()
    tgt[5] = -6.30
    assert ur_kin.within_joint_limits(seed)
    assert not ur_kin.within_joint_limits(tgt)
    q, info = es.ik_tcp_nearest(es.tcp_pose(tgt), seed, branch=ur_kin.branch_id(seed))
    assert q is not None
    assert q[5] == pytest.approx(-6.30, abs=1e-6)
    assert not ur_kin.within_joint_limits(q)      # <-- illegal, and returned anyway
    assert info["on_branch"] is True
    assert info["max_dev"] == pytest.approx(0.05, abs=1e-6)
    assert info["in_limits"] is False             # fixed by the integrator: caller can tell
    # and a normal in-limit solve reports True
    q2, info2 = es.ik_tcp_nearest(es.tcp_pose(seed), seed, branch=ur_kin.branch_id(seed))
    assert q2 is not None and info2["in_limits"] is True


def test_ik_limit_margin_can_reject_everything_but_never_raises():
    q = np.array([-3.193, -1.490, 1.693, -1.798, -1.590, -3.269])
    got, _ = es.ik_tcp_nearest(es.tcp_pose(q), q, branch=2, limit_margin=10.0)
    # every candidate fails the margin -> the filter falls back to the unfiltered
    # pool rather than returning None (pinned: the CALLER owns the envelope check)
    assert got is not None


# --------------------------------------------------------------------------- #
# 6. clamps                                                                   #
# --------------------------------------------------------------------------- #
def test_clamp_delta_is_norm_proportional_and_preserves_direction():
    rng = np.random.default_rng(3)
    for _ in range(300):
        a = np.concatenate([rng.normal(size=3) * 0.1,
                            rng.normal(size=3) * 0.5,
                            [rng.random()]])
        out, pc, rc = es.clamp_delta(a, 0.02, 0.10)
        for sl, cap, flag in ((slice(0, 3), 0.02, pc), (slice(3, 6), 0.10, rc)):
            n_in = np.linalg.norm(a[sl])
            n_out = np.linalg.norm(out[sl])
            assert flag == bool(n_in > cap)
            if flag:
                assert n_out == pytest.approx(cap, rel=1e-12)
                # direction bit-preserved: unit vectors identical
                np.testing.assert_allclose(out[sl] / n_out, a[sl] / n_in, atol=1e-12)
            else:
                np.testing.assert_allclose(out[sl], a[sl], atol=0.0)
        assert out[6] == a[6]  # gripper channel untouched


def test_clamp_delta_never_grows_a_small_delta():
    a = np.array([1e-4, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0])
    out, pc, rc = es.clamp_delta(a, 0.02, 0.10)
    np.testing.assert_allclose(out, a, atol=0.0)
    assert pc is False and rc is False


def test_clamp_delta_does_not_mutate_its_input():
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    before = a.copy()
    es.clamp_delta(a, 0.02, 0.10)
    np.testing.assert_allclose(a, before, atol=0.0)
    # and it works from a plain list too
    out, _, _ = es.clamp_delta([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.02, 0.10)
    assert out[0] == pytest.approx(0.02)


def test_clamp_delta_translation_and_rotation_are_independent():
    a = np.array([1.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0])
    out, pc, rc = es.clamp_delta(a, 0.02, 0.10)
    assert pc is True and rc is False
    assert out[0] == pytest.approx(0.02)
    assert out[3] == pytest.approx(1e-6)


def test_clamp_delta_zero_cap_is_a_NO_OP_not_a_freeze():
    """PINNED FAIL-OPEN SEMANTICS (``n > max > 0.0``): a cap of 0 -- or a
    negative cap -- DISABLES the clamp instead of forbidding motion.

    That matches this repo's "0.0 = off" idiom (deadband_rad, soft_start_s) but
    it is NOT documented in eef_space, and a yaml typo ``eef_max_step_m: 0.0``
    would silently remove the per-tick Cartesian limit rather than hold the arm.
    """
    a = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    out, pc, rc = es.clamp_delta(a, 0.0, 0.0)
    np.testing.assert_allclose(out, a, atol=0.0)
    assert pc is False and rc is False
    out_neg, pc_neg, _ = es.clamp_delta(a, -1.0, -1.0)
    np.testing.assert_allclose(out_neg, a, atol=0.0)
    assert pc_neg is False


def test_clamp_delta_nan_passes_through_unclamped():
    """PINNED: NaN fails every ``>`` comparison, so a non-finite action is NOT
    caught here. The executor (A4) must validate finiteness before clamping."""
    a = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out, pc, rc = es.clamp_delta(a, 0.02, 0.10)
    assert math.isnan(out[0])
    assert pc is False and rc is False


def test_clamp_position_box_clips_axiswise_and_reports():
    lo = np.array([0.35, -0.30, 0.00])
    hi = np.array([0.78, 0.45, 0.40])
    inside = np.array([0.5, 0.1, 0.2])
    out, hit = es.clamp_position_box(inside, lo, hi)
    np.testing.assert_allclose(out, inside, atol=0.0)
    assert hit is False
    out, hit = es.clamp_position_box([0.9, -0.9, 0.2], lo, hi)
    np.testing.assert_allclose(out, [0.78, -0.30, 0.2], atol=0.0)
    assert hit is True
    # exactly on the boundary is inside
    out, hit = es.clamp_position_box(hi, lo, hi)
    assert hit is False


def test_clamp_position_box_is_axiswise_by_design_unlike_clamp_delta():
    """A POSITION box is a box; only DELTAS must be scaled norm-proportionally.
    Pinned so the two are never "harmonised" into the same rule."""
    out, _ = es.clamp_position_box([10.0, 10.0, 10.0], [0, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(out, [1, 2, 3], atol=0.0)


def test_clamp_position_box_nan_reports_clamped_but_stays_nan():
    """PINNED: ``c != p`` is True for NaN, so the flag says "clamped" while the
    value is still NaN. Another reason finiteness must be checked upstream."""
    out, hit = es.clamp_position_box([np.nan, 0.1, 0.2], [0, 0, 0], [1, 1, 1])
    assert hit is True
    assert math.isnan(out[0])


def test_clamp_position_box_with_inverted_bounds_returns_hi():
    """lo > hi is a config error; pinned behaviour is "collapse to hi", silently."""
    out, hit = es.clamp_position_box([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=0.0)
    assert hit is True


# --------------------------------------------------------------------------- #
# 7. end-to-end: dataset action -> deploy -> q_{t+1}, with the clamps in       #
# --------------------------------------------------------------------------- #
def test_clamped_round_trip_stays_on_branch_and_inside_limits(takes):
    """Full DESIGN §3 chain on real frames: clamp, box, IK. The carrot deltas are
    all well under the caps, so clamping must be a no-op and the IK must still
    land on q_{t+1}."""
    lo = np.array([0.35, -0.30, 0.00])
    hi = np.array([0.78, 0.45, 0.40])
    clamped_any = False
    for _name, Q in takes:
        branch = ur_kin.branch_id(Q[0])
        for qa, qb in _sample_pairs(Q, 25):
            a = es.delta_action(qa, qb, 0.0)
            a, pc, rc = es.clamp_delta(a, 0.02, 0.10)
            clamped_any = clamped_any or pc or rc
            T = es.apply_delta(qa, a)
            p, hit = es.clamp_position_box(T[:3, 3], lo, hi)
            assert hit is False, (p, "workspace box must contain the corpus")
            T[:3, 3] = p
            q, info = es.ik_tcp_nearest(T, qa, branch=branch)
            assert q is not None and info["on_branch"]
            np.testing.assert_allclose(q, qb, atol=1e-9)
    assert clamped_any is False  # per-frame corpus motion is far under the caps
