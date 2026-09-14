#!/usr/bin/env python3
"""Unit tests for ``scripts/dataset/make_carrot_eef_limits.py`` (design row A2).

Every number this tool emits ends up as a safety clamp in ``policy_leader_node``,
so the formulas are pinned here on a *synthetic* take with hand-computable joints
rather than on the 54 real takes (which no test may depend on).

Run (system python3, no ROS overlay needed -- both packages are ROS-free)::

    cd /home/laptop3/gello_software_humble/ros2_ur_ws/src/gello_policy
    PYTHONPATH=../ur_gello_bringup:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      python3 -m pytest -q -p no:cacheprovider test/test_carrot_eef_limits.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pytest

_WT = Path(__file__).resolve().parents[4]          # /home/laptop3/gello_software_humble
for _p in (
    _WT / "scripts" / "dataset",
    _WT / "ros2_ur_ws" / "src" / "ur_gello_bringup",
    _WT / "ros2_ur_ws" / "src" / "gello_policy",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_carrot_eef_limits as mk  # noqa: E402
from gello_policy import eef_space  # noqa: E402
from ur_gello_bringup import ur_kin  # noqa: E402

# A pose known to be on closed-form branch 2 (the carrot dataset's branch) and
# comfortably away from the elbow / wrist_2 merge points.
Q_BASE = np.array([-3.1931, -1.4897, 1.6932, -1.7983, -1.5902, -3.2691])


# --------------------------------------------------------------------------- #
# synthetic take builder
# --------------------------------------------------------------------------- #
def write_take(
    root: Path,
    name: str,
    q_traj: np.ndarray,
    grip_pos: Sequence[float] | None = None,
    grip_cmd: Sequence[float] | None = None,
    fps: float = 30.0,
    joint_offset_s: float = 0.0,
    cmd_traj: np.ndarray | None = None,
) -> Path:
    """Write a minimal ``take_*/vectors.h5`` holding only what A2 reads.

    ``joint_offset_s`` shifts the ur_joint_states clock relative to the cam1
    master clock so the nearest-timestamp resampling can be exercised; with 0.0
    the two clocks coincide and resampling is the identity.
    ``cmd_traj`` (n,6), if given, is written as the ``command`` group on the cam1
    clock -- that is what the joint-cache shift estimator compares against.
    """
    q_traj = np.asarray(q_traj, dtype=float)
    n = len(q_traj)
    tk = root / name
    tk.mkdir(parents=True, exist_ok=True)
    t = np.arange(n, dtype=float) / fps
    gp = np.asarray(grip_pos if grip_pos is not None else np.zeros(n), dtype=float)
    gc = np.asarray(grip_cmd if grip_cmd is not None else np.zeros(n), dtype=float)

    with h5py.File(tk / "vectors.h5", "w") as f:
        g = f.create_group("cam1_frames")
        g.create_dataset("t_rel_s", data=t)
        g.create_dataset("frame_idx", data=np.arange(n, dtype=float))
        g = f.create_group("cam2_frames")
        g.create_dataset("t_rel_s", data=t)
        g = f.create_group("ur_joint_states")
        g.create_dataset("t_rel_s", data=t + joint_offset_s)
        for k in range(6):
            g.create_dataset(f"q{k + 1}", data=q_traj[:, k])
        if cmd_traj is not None:
            g = f.create_group("command")
            g.create_dataset("t_rel_s", data=t)
            for k in range(6):
                g.create_dataset(f"cmd{k + 1}", data=np.asarray(cmd_traj, dtype=float)[:, k])
        g = f.create_group("gripper")
        g.create_dataset("t_rel_s", data=t)
        g.create_dataset("grip_pos", data=gp)
        g.create_dataset("grip_cmd", data=gc)
    return tk


def line_traj(q0: np.ndarray, dq: np.ndarray, n: int) -> np.ndarray:
    """n frames of constant-velocity joint motion starting at q0."""
    return q0[None, :] + np.arange(n)[:, None] * dq[None, :]


# --------------------------------------------------------------------------- #
# pure formula tests
# --------------------------------------------------------------------------- #
def test_ceil_to_never_rounds_down():
    assert mk.ceil_to(0.0144, 0.001) == pytest.approx(0.015)
    assert mk.ceil_to(0.0451, 0.01) == pytest.approx(0.05)
    assert mk.ceil_to(0.064095, 0.01) == pytest.approx(0.07)
    # an exact multiple must stay put, not jump a whole step
    assert mk.ceil_to(0.015, 0.001) == pytest.approx(0.015)
    assert mk.ceil_to(0.05, 0.01) == pytest.approx(0.05)
    # and binary-representation noise must not push it up either
    assert mk.ceil_to(0.1 + 0.2, 0.01) == pytest.approx(0.30)


def test_expand_symmetric_reproduces_the_banana_file():
    """The 1.2x formula is shared with config/joint_safety_limits.json.

    Only to 1e-4: the banana file printed its raw_min/raw_max rounded to 4 decimals
    but derived its limits from the UNROUNDED data, so it is not reproducible from
    its own contents. That is exactly the wart ``outward()``/``GRID`` removes here
    (see test_shipped_json_is_self_consistent, which demands an exact match).
    """
    raw_min = np.array([2.5064, -2.3911, 1.1810, -3.1413, -2.1031, -4.9145])
    raw_max = np.array([3.5620, -1.0215, 2.4876, -1.2021, -1.3287, -1.8314])
    lo, hi = mk.expand_symmetric(raw_min, raw_max)
    np.testing.assert_allclose(
        lo, [2.4008, -2.5281, 1.0504, -3.3353, -2.1805, -5.2228], atol=1.01e-4)
    np.testing.assert_allclose(
        hi, [3.6676, -0.8845, 2.6183, -1.0082, -1.2512, -1.5231], atol=1.01e-4)
    # the expansion is symmetric: the centre is untouched and the span grows 1.2x
    np.testing.assert_allclose(0.5 * (lo + hi), 0.5 * (raw_min + raw_max), atol=1e-12)
    np.testing.assert_allclose(hi - lo, 1.2 * (raw_max - raw_min), atol=1e-12)


def test_floor_to_and_outward_only_ever_widen():
    assert mk.floor_to(-3.70274, mk.GRID) == pytest.approx(-3.7028)
    assert mk.floor_to(-3.7027, mk.GRID) == pytest.approx(-3.7027)   # exact stays put
    assert mk.ceil_to(2.07071, mk.GRID) == pytest.approx(2.0708)
    lo, hi = mk.outward([0.41472, -0.21443], [0.70906, 0.38921])
    np.testing.assert_allclose(lo, [0.4147, -0.2145])
    np.testing.assert_allclose(hi, [0.7091, 0.3893])
    # quantised interval always contains the true one
    assert lo[0] <= 0.41472 and hi[0] >= 0.70906
    assert lo[1] <= -0.21443 and hi[1] >= 0.38921


def test_expand_symmetric_contains_the_raw_envelope():
    raw_min = np.array([-3.7027, -1.7284, 1.3382, -2.5867, -1.8700, -4.4087])
    raw_max = np.array([-2.6810, -0.9593, 2.0708, -1.5425, -1.2622, -2.0574])
    lo, hi = mk.expand_symmetric(raw_min, raw_max)
    assert np.all(lo < raw_min) and np.all(hi > raw_max)


# --------------------------------------------------------------------------- #
# h5 reading / resampling
# --------------------------------------------------------------------------- #
def test_load_take_arrays_identity_clock(tmp_path):
    q = line_traj(Q_BASE, np.full(6, 0.001), 10)
    tk = write_take(tmp_path, "take_01", q, grip_pos=np.linspace(0.0, 0.9, 10),
                    grip_cmd=np.linspace(0.0, 1.0, 10))
    got_q, got_pos, got_cmd = mk.load_take_arrays(str(tk / "vectors.h5"))
    np.testing.assert_allclose(got_q, q)
    np.testing.assert_allclose(got_pos, np.linspace(0.0, 0.9, 10))
    np.testing.assert_allclose(got_cmd, np.linspace(0.0, 1.0, 10))


def test_load_take_arrays_nearest_timestamp_shift(tmp_path):
    """A joint clock shifted by more than half a frame must resample by one row."""
    q = line_traj(Q_BASE, np.full(6, 0.01), 10)
    # joints are stamped 0.7 frames LATE, so master frame k is nearest joint row k-1
    tk = write_take(tmp_path, "take_01", q, joint_offset_s=0.7 / 30.0)
    got_q, _, _ = mk.load_take_arrays(str(tk / "vectors.h5"))
    np.testing.assert_allclose(got_q[1:], q[:-1])
    np.testing.assert_allclose(got_q[0], q[0])  # clipped at the start


def test_load_take_arrays_ffill_bfill_on_grip_cmd(tmp_path):
    q = line_traj(Q_BASE, np.zeros(6), 6)
    cmd = np.array([np.nan, 0.2, np.nan, np.nan, 0.8, np.nan])
    tk = write_take(tmp_path, "take_01", q, grip_cmd=cmd)
    _, _, got_cmd = mk.load_take_arrays(str(tk / "vectors.h5"))
    np.testing.assert_allclose(got_cmd, [0.2, 0.2, 0.2, 0.2, 0.8, 0.8])


# --------------------------------------------------------------------------- #
# start pose: the +/-2pi split is the whole reason wrapped_nearest is used
# --------------------------------------------------------------------------- #
def test_start_pose_unwraps_a_2pi_split_before_averaging(tmp_path):
    """Two takes at the SAME physical pose, one recorded 2pi away on wrist_3."""
    qa = line_traj(Q_BASE, np.zeros(6), 5)
    qb = qa.copy()
    qb[:, 5] += 2.0 * math.pi                      # same wrist_3 angle, other revolution
    write_take(tmp_path, "take_01", qa)
    write_take(tmp_path, "take_02", qb)

    raw = mk.analyse(str(tmp_path), verbose=False)
    # naive mean would be Q_BASE[5] + pi = -0.128 rad, a pose the arm never held
    assert raw["start_pose"][5] == pytest.approx(Q_BASE[5], abs=1e-9)
    np.testing.assert_allclose(raw["start_pose"], Q_BASE, atol=1e-9)
    np.testing.assert_allclose(raw["start_pose_std"], np.zeros(6), atol=1e-9)
    np.testing.assert_allclose(raw["start_pose_max_dev"], np.zeros(6), atol=1e-9)


def test_start_pose_reports_spread(tmp_path):
    qa = line_traj(Q_BASE, np.zeros(6), 4)
    qb = qa + 0.10
    write_take(tmp_path, "take_01", qa)
    write_take(tmp_path, "take_02", qb)
    raw = mk.analyse(str(tmp_path), verbose=False)
    np.testing.assert_allclose(raw["start_pose"], Q_BASE + 0.05, atol=1e-9)
    np.testing.assert_allclose(raw["start_pose_std"], np.full(6, 0.05), atol=1e-9)
    np.testing.assert_allclose(raw["start_pose_max_dev"], np.full(6, 0.05), atol=1e-9)


def test_start_gripper_is_the_median_of_frame_zero(tmp_path):
    q = line_traj(Q_BASE, np.zeros(6), 4)
    for i, g0 in enumerate([0.01, 0.02, 0.90]):    # median 0.02, mean would be 0.31
        write_take(tmp_path, f"take_{i + 1:02d}", q, grip_pos=np.full(4, g0))
    raw = mk.analyse(str(tmp_path), verbose=False)
    assert raw["start_gripper"] == pytest.approx(0.02)


def test_start_branch_matches_ur_kin_and_all_frames_are_on_it(tmp_path):
    q = line_traj(Q_BASE, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01]), 8)
    write_take(tmp_path, "take_01", q)
    raw = mk.analyse(str(tmp_path), verbose=False)
    assert raw["start_branch"] == ur_kin.branch_id(raw["start_pose"])
    assert raw["branch_checked"] == 8
    assert raw["branch_n_off"] == 0


# --------------------------------------------------------------------------- #
# stale joint cache: tau_q
# --------------------------------------------------------------------------- #
def test_estimate_joint_shift_recovers_an_injected_lag(tmp_path):
    """command(t) == q(t + 0.9 s) by construction -> the estimator must find 0.9."""
    dq = np.array([0.003, -0.002, 0.004, 0.001, -0.001, 0.002])
    n, lead = 120, 27                       # 27 frames @ 30 fps == 0.90 s
    q = line_traj(Q_BASE, dq, n)
    cmd = line_traj(Q_BASE + lead * dq, dq, n)     # the pose the arm reaches 0.9 s later
    tk = write_take(tmp_path, "take_01", q, cmd_traj=cmd)

    tau, res, used = mk.estimate_joint_shift(mk.Take(str(tk / "vectors.h5")))
    # nearest-timestamp snapping makes every tau in one frame-width equivalent
    assert tau == pytest.approx(0.90, abs=0.02), tau
    assert res < 1e-9                       # exact by construction
    assert 0 < used <= n
    assert mk.TAU_MIN_OK <= tau <= mk.TAU_MAX_OK


def test_estimate_joint_shift_is_none_without_a_command_stream(tmp_path):
    tk = write_take(tmp_path, "take_01", line_traj(Q_BASE, np.zeros(6), 5))
    tau, res, used = mk.estimate_joint_shift(mk.Take(str(tk / "vectors.h5")))
    assert tau is None and math.isnan(res) and used == 0


def test_q_at_masks_queries_past_the_end_of_the_record(tmp_path):
    """The shifted tail has no joints; nearest_idx would clamp and fake a frozen arm."""
    n = 60
    q = line_traj(Q_BASE, np.full(6, 0.001), n)
    tk = write_take(tmp_path, "take_01", q)
    take = mk.Take(str(tk / "vectors.h5"))
    _, ok0 = take.q_at(0.0)
    qs, ok = take.q_at(0.9)                 # 27 frames of tail fall off the end
    assert ok0.all()
    # 0.9 s == 27 frames; +/-1 because t_k + 0.9 lands on the record's last stamp
    # only up to float64 rounding
    assert abs(int(ok.sum()) - (n - 27)) <= 1
    assert not ok[-26:].any()            # the tail is masked, not silently clamped
    assert ok[:int(ok.sum())].all()      # and the valid frames are a contiguous prefix
    np.testing.assert_allclose(qs[0], q[27])
    # the clamped tail must not contribute a (zero-motion) delta sample
    dp, _ = mk.delta_magnitudes(qs, 1, ok)
    assert len(dp) == int(ok.sum()) - 1


def test_analyse_records_the_shift_per_take(tmp_path):
    dq = np.array([0.003, -0.002, 0.004, 0.001, -0.001, 0.002])
    q = line_traj(Q_BASE, dq, 120)
    cmd = line_traj(Q_BASE + 27 * dq, dq, 120)
    write_take(tmp_path, "take_01", q, cmd_traj=cmd)
    write_take(tmp_path, "take_02", q, cmd_traj=cmd)
    raw = mk.analyse(str(tmp_path), verbose=False)
    assert len(raw["tau_q"]) == 2
    assert all(t == pytest.approx(0.90, abs=0.02) for t in raw["tau_q"])
    assert mk.check_shifts(raw) == []

    doc = mk.build_document(raw, str(tmp_path))
    assert doc["joint_state_shift_median_s"] == pytest.approx(0.90, abs=0.02)
    assert len(doc["joint_state_shift_s"]) == 2
    assert len(doc["joint_state_shift_residual_m"]) == 2
    # frame_1 is derived from the shifted joints, so it has fewer samples than raw
    assert doc["delta_stats"]["frame_1"]["n"] < doc["delta_stats"]["frame_1_unshifted"]["n"]


def test_check_shifts_rejects_an_out_of_window_take():
    raw = {"takes": ["take_01", "take_02", "take_03"],
           "tau_q": [0.90, 0.20, 0.95],
           "tau_residual_m": [0.0005, 0.0005, 0.010]}
    problems = mk.check_shifts(raw)
    assert len(problems) == 2
    assert "take_02" in problems[0] and "outside" in problems[0]
    assert "take_03" in problems[1] and "residual" in problems[1]


def test_start_pose_is_the_destaled_frame_zero(tmp_path):
    """Frame 0 is the one quantity a constant shift does NOT leave alone.

    start_pose must be the arm's real pose when the first image was taken, i.e.
    q interpolated at cam1_t[0] + tau_q -- not the naive q(cam1_t[0]).
    """
    dq = np.array([0.003, -0.002, 0.004, 0.001, -0.001, 0.002])
    q = line_traj(Q_BASE, dq, 120)
    cmd = line_traj(Q_BASE + 27 * dq, dq, 120)       # command leads by 0.9 s
    write_take(tmp_path, "take_01", q, cmd_traj=cmd)
    raw = mk.analyse(str(tmp_path), verbose=False)
    # nearest-timestamp scoring makes every tau within one frame-width equivalent,
    # so the estimate lands anywhere in [0.883, 0.917); interpolate at whatever it
    # found rather than at the nominal 0.90.
    tau = raw["tau_q"][0]
    assert tau == pytest.approx(0.90, abs=0.02)
    expected = Q_BASE + tau * 30.0 * dq
    np.testing.assert_allclose(raw["start_pose"], expected, atol=1e-9)
    np.testing.assert_allclose(raw["start_pose_unshifted"], Q_BASE, atol=1e-9)
    # the de-staled pose is ~27 frames of motion away from the naive one
    assert np.max(np.abs(raw["start_pose"] - raw["start_pose_unshifted"])) > 0.05

    doc = mk.build_document(raw, str(tmp_path))
    assert doc["start_pose"] != doc["start_pose_unshifted"]
    np.testing.assert_allclose(
        doc["start_tcp"], np.round(eef_space.tcp_pose(expected)[:3, 3], 4), atol=1e-4)
    # the pasteable snippet carries the de-staled value; the naive one only as a comment
    body = [ln for ln in doc["yaml_snippet"].splitlines() if not ln.lstrip().startswith("#")]
    assert sum(1 for ln in body if ln.lstrip().startswith("start_pose:")) == 1
    assert not any(ln.lstrip().startswith("start_pose_unshifted:") for ln in body)
    assert any("# start_pose_unshifted:" in ln for ln in doc["yaml_snippet"].splitlines())


def test_q_interp_at_is_linear_not_snapped(tmp_path):
    """The primary de-staling path interpolates; nearest would snap to a sample."""
    n = 60
    q = line_traj(Q_BASE, np.full(6, 0.001), n)
    tk = write_take(tmp_path, "take_01", q)
    take = mk.Take(str(tk / "vectors.h5"))
    half = 0.5 / 30.0                                  # half a frame: exactly between samples
    qi, ok = take.q_interp_at(half)
    qn, _ = take.q_at(half)
    np.testing.assert_allclose(qi[0], 0.5 * (q[0] + q[1]), atol=1e-12)
    assert not np.allclose(qn[0], qi[0])               # nearest lands on a sample, interp does not
    assert ok[:-1].all()


def test_frame_1_is_the_interpolated_series_and_drives_the_step_clamp(tmp_path):
    """eef_max_step_* must come from the interpolated de-staled deltas (A1's recipe)."""
    dq = np.array([0.003, -0.002, 0.004, 0.001, -0.001, 0.002])
    q = line_traj(Q_BASE, dq, 120)
    cmd = line_traj(Q_BASE + 27 * dq, dq, 120)
    write_take(tmp_path, "take_01", q, cmd_traj=cmd)
    doc = mk.build_document(mk.analyse(str(tmp_path), verbose=False), str(tmp_path))
    st = doc["delta_stats"]
    assert "LINEAR" in st["frame_1"]["joints"]
    # the two audit blocks are still there and are NOT what the clamp is built on
    assert "nearest" in st["frame_1_destaled_nearest"]["joints"]
    assert "audit only" in st["frame_1_destaled_nearest"]["joints"]
    assert "audit only" in st["frame_1_unshifted"]["joints"]
    assert doc["eef_max_step_m"] == pytest.approx(
        mk.ceil_to(mk.STEP_SAFETY_FACTOR * st["frame_1"]["dp"]["max"], 0.001))


# --------------------------------------------------------------------------- #
# workspace box
# --------------------------------------------------------------------------- #
def test_workspace_margins_and_z_floor(tmp_path):
    """x/y get +/-0.03; z_lo only 0.01 and never below 0."""
    q = line_traj(Q_BASE, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3)
    write_take(tmp_path, "take_01", q)
    raw = mk.analyse(str(tmp_path), verbose=False)
    doc = mk.build_document(raw, str(tmp_path))

    p = eef_space.tcp_pose(Q_BASE)[:3, 3]
    np.testing.assert_allclose(doc["workspace_raw"]["min"], np.round(p, 4), atol=1e-4)
    np.testing.assert_allclose(doc["workspace_raw"]["max"], np.round(p, 4), atol=1e-4)
    np.testing.assert_allclose(doc["workspace_lo"][:2], np.round(p[:2] - 0.03, 4), atol=1e-4)
    np.testing.assert_allclose(doc["workspace_hi"], np.round(p + 0.03, 4), atol=1e-4)
    assert doc["workspace_lo"][2] == pytest.approx(round(p[2] - 0.01, 4), abs=1e-4)


def test_workspace_z_floor_clamps_at_zero():
    raw = {
        "joint_raw_min": np.zeros(6), "joint_raw_max": np.ones(6),
        "ws_min": np.array([0.4, -0.2, 0.004]), "ws_max": np.array([0.7, 0.4, 0.3]),
        "dp1": np.array([0.001]), "dr1": np.array([0.001]),
        "dp30": np.array([0.05]), "dr30": np.array([0.09]),
        "start_pose": Q_BASE, "start_pose_std": np.zeros(6),
        "start_pose_max_dev": np.zeros(6), "start_branch": 2,
        "start_pose_unshifted": Q_BASE, "start_pose_unshifted_std": np.zeros(6),
        "start_pose_unshifted_max_dev": np.zeros(6),
        "tau_q": [0.89], "tau_residual_m": [0.0006],
        "branch_checked": 1, "branch_stride": 1, "branch_off": [], "branch_n_off": 0,
        "start_gripper": 0.0, "n_takes": 1, "n_frames": 1, "takes": ["take_01"],
        "gripper_pos_raw_min": 0.0, "gripper_pos_raw_max": 1.0,
        "gripper_cmd_raw_min": 0.0, "gripper_cmd_raw_max": 1.0,
    }
    doc = mk.build_document(raw, "/nowhere")
    # 0.004 - 0.01 would be negative -> floored at exactly 0.0, never below the table
    assert doc["workspace_lo"][2] == 0.0
    assert doc["workspace_lo"][0] == pytest.approx(0.37)


# --------------------------------------------------------------------------- #
# delta statistics and the clamp rounding built on top of them
# --------------------------------------------------------------------------- #
def test_delta_magnitudes_match_eef_space(tmp_path):
    q = line_traj(Q_BASE, np.array([0.002, -0.001, 0.003, 0.0, 0.001, -0.002]), 40)
    dp, drot = mk.delta_magnitudes(q, 1)
    assert len(dp) == 39
    for t in (0, 7, 38):
        a = eef_space.delta_action(q[t], q[t + 1], 0.0)
        assert dp[t] == pytest.approx(float(np.linalg.norm(a[:3])))
        assert drot[t] == pytest.approx(float(np.linalg.norm(a[3:6])))
    # a horizon longer than the take yields no samples instead of raising
    assert mk.delta_magnitudes(q, 100)[0].size == 0


def test_step_and_lead_clamps_follow_the_documented_rounding(tmp_path):
    q = line_traj(Q_BASE, np.array([0.002, -0.001, 0.003, 0.0, 0.001, -0.002]), 60)
    write_take(tmp_path, "take_01", q)
    raw = mk.analyse(str(tmp_path), verbose=False)
    doc = mk.build_document(raw, str(tmp_path))

    st = doc["delta_stats"]
    assert doc["eef_max_step_m"] == pytest.approx(
        mk.ceil_to(1.5 * st["frame_1"]["dp"]["max"], 0.001))
    assert doc["eef_max_step_rad"] == pytest.approx(
        mk.ceil_to(1.5 * st["frame_1"]["drot"]["max"], 0.01))
    assert doc["eef_max_lead_m"] == pytest.approx(mk.ceil_to(st["frame_30"]["dp"]["p50"], 0.01))
    assert doc["eef_max_lead_rad"] == pytest.approx(mk.ceil_to(st["frame_30"]["drot"]["p50"], 0.01))

    # the clamps must never sit under what the demonstrations actually did
    assert doc["eef_max_step_m"] > st["frame_1"]["dp"]["max"]
    assert doc["eef_max_step_rad"] > st["frame_1"]["drot"]["max"]
    assert doc["eef_max_lead_m"] >= st["frame_30"]["dp"]["p50"]
    # every emitted number is a multiple of its rounding step
    assert round(doc["eef_max_step_m"] * 1000) == pytest.approx(doc["eef_max_step_m"] * 1000)
    assert round(doc["eef_max_step_rad"] * 100) == pytest.approx(doc["eef_max_step_rad"] * 100)


# --------------------------------------------------------------------------- #
# the yaml snippet A6 pastes
# --------------------------------------------------------------------------- #
def test_yaml_snippet_is_parseable_and_four_decimals(tmp_path):
    yaml = pytest.importorskip("yaml")
    q = line_traj(Q_BASE, np.array([0.002, -0.001, 0.003, 0.0, 0.001, -0.002]), 45)
    write_take(tmp_path, "take_01", q)
    doc = mk.build_document(mk.analyse(str(tmp_path), verbose=False), str(tmp_path))

    snippet = doc["yaml_snippet"]
    # pastes under "policy_leader_node: / ros__parameters:" -> must be indented, and
    # must still parse when that two-line header is put back in front of it.
    assert all(ln.startswith("    ") for ln in snippet.splitlines())
    parsed = yaml.safe_load("policy_leader_node:\n  ros__parameters:\n" + snippet)
    params = parsed["policy_leader_node"]["ros__parameters"]

    for key in ("start_pose", "start_gripper", "joint_limits_lo", "joint_limits_hi",
                "workspace_lo", "workspace_hi", "eef_max_step_m", "eef_max_step_rad",
                "eef_max_lead_m", "eef_max_lead_rad"):
        assert key in params, key
    assert len(params["start_pose"]) == 6
    assert len(params["joint_limits_lo"]) == len(params["joint_limits_hi"]) == 6
    assert len(params["workspace_lo"]) == len(params["workspace_hi"]) == 3

    # values agree with the JSON to the 4 decimals the snippet promises
    np.testing.assert_allclose(params["start_pose"], np.round(doc["start_pose"], 4), atol=1e-9)
    np.testing.assert_allclose(params["workspace_lo"], np.round(doc["workspace_lo"], 4), atol=1e-9)
    assert params["eef_max_step_m"] == pytest.approx(doc["eef_max_step_m"], abs=5e-5)
    # every scalar/list entry is printed with exactly 4 decimals
    for ln in snippet.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        for tok in ln.split(":", 1)[1].replace("[", "").replace("]", "").split(","):
            tok = tok.strip()
            assert len(tok.split(".")[1]) == 4, ln


# --------------------------------------------------------------------------- #
# the shipped artifact
# --------------------------------------------------------------------------- #
def test_shipped_json_is_self_consistent():
    """Guards the committed config/carrot_eef_limits.json against hand edits."""
    import json
    path = _WT / "ros2_ur_ws" / "src" / "gello_policy" / "config" / "carrot_eef_limits.json"
    if not path.exists():
        pytest.skip("carrot_eef_limits.json not generated yet")
    doc = json.loads(path.read_text())

    lo, hi = mk.outward(*mk.expand_symmetric(
        np.array(doc["joint_raw_min"]), np.array(doc["joint_raw_max"])))
    np.testing.assert_allclose(doc["joint_limits_lo"], lo, atol=1e-9)
    np.testing.assert_allclose(doc["joint_limits_hi"], hi, atol=1e-9)

    raw_lo = np.array(doc["workspace_raw"]["min"])
    raw_hi = np.array(doc["workspace_raw"]["max"])
    np.testing.assert_allclose(doc["workspace_hi"], np.round(raw_hi + 0.03, 4), atol=1e-9)
    np.testing.assert_allclose(doc["workspace_lo"][:2], np.round(raw_lo[:2] - 0.03, 4), atol=1e-9)
    assert doc["workspace_lo"][2] == pytest.approx(max(round(raw_lo[2] - 0.01, 4), 0.0), abs=1e-9)

    assert doc["start_branch"] == ur_kin.branch_id(np.array(doc["start_pose"]))
    assert doc["branch_frames_off"] == 0

    # the integrator's acceptance window on the stale joint cache
    taus = [t for t in doc["joint_state_shift_s"] if t is not None]
    assert len(taus) == doc["n_takes"]
    assert all(mk.TAU_MIN_OK <= t <= mk.TAU_MAX_OK for t in taus)
    assert all(r < mk.TAU_RESIDUAL_MAX_M for r in doc["joint_state_shift_residual_m"])
    assert doc["joint_state_shift_median_s"] == pytest.approx(0.89, abs=0.05)

    # start_pose is the DE-STALED frame-0 mean; the naive one is kept but differs
    assert doc["start_pose"] != doc["start_pose_unshifted"]
    assert "LINEAR" in doc["delta_stats"]["frame_1"]["joints"]
    assert doc["delta_stats"]["frame_1"]["dp"]["max"] < \
        doc["delta_stats"]["frame_1_destaled_nearest"]["dp"]["max"]

    # the clamps are exactly what the printed statistics imply
    st = doc["delta_stats"]
    assert doc["eef_max_step_m"] == pytest.approx(
        mk.ceil_to(mk.STEP_SAFETY_FACTOR * st["frame_1"]["dp"]["max"], 0.001))
    assert doc["eef_max_step_rad"] == pytest.approx(
        mk.ceil_to(mk.STEP_SAFETY_FACTOR * st["frame_1"]["drot"]["max"], 0.01))
    assert doc["eef_max_lead_m"] == pytest.approx(mk.ceil_to(st["frame_30"]["dp"]["p50"], 0.01))
    assert doc["eef_max_lead_rad"] == pytest.approx(mk.ceil_to(st["frame_30"]["drot"]["p50"], 0.01))
    # a single tick may never be allowed to cover a whole second of demonstrated motion
    assert doc["eef_max_step_m"] < doc["eef_max_lead_m"]
    # the start pose must survive its own clamps
    assert np.all(np.array(doc["start_pose"]) >= np.array(doc["joint_limits_lo"]))
    assert np.all(np.array(doc["start_pose"]) <= np.array(doc["joint_limits_hi"]))
    p = eef_space.tcp_pose(np.array(doc["start_pose"]))[:3, 3]
    assert np.all(p >= np.array(doc["workspace_lo"]))
    assert np.all(p <= np.array(doc["workspace_hi"]))
