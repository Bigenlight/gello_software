"""controller.py: eef/joint teleop pipeline driven with an explicit clock + FakeLeader.

Plant model: perfect tracking (q_actual == q_cmd of the previous tick) — the
controller only decides commands; the MuJoCo position actuators follow them.
"""
import os

import numpy as np
import pytest

from sim_collect.controller import TeleopController, load_bridge_params
from sim_collect.leader import FakeLeader
from ur_gello_bringup import ur_kin

HOME = np.array([-3.302, -1.563, 1.607, -1.523, -1.615, -3.118])  # real robot branch (J1 ~ -pi)
BASE = "ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml"
EEF = "ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello_eef.yaml"
HZ = 250.0
LEADER_HZ = 30.0


@pytest.fixture(scope="module")
def params():
    p = load_bridge_params([BASE, EEF])
    assert p["control_mode"] == "eef" and p["max_step_rad"] == 0.0025 and p["v_max"] == 0.16
    return p


class Rig:
    """Runs leader sampling (30 Hz) + control ticks (250 Hz) on a simulated clock."""

    def __init__(self, ctrl: TeleopController, leader: FakeLeader, q0=HOME):
        self.c = ctrl
        self.L = leader
        self.q = np.asarray(q0, dtype=float).copy()
        self.t = 0.0
        self.next_leader = 0.0
        self.max_step = 0.0
        self.last_cmd = None

    def run(self, seconds: float, feed_leader: bool = True):
        n = int(round(seconds * HZ))
        for _ in range(n):
            if feed_leader and self.t >= self.next_leader - 1e-9:
                self.c.ingest(self.L.sample_now(self.t))
                self.next_leader += 1.0 / LEADER_HZ
            q_cmd = self.c.tick(self.q, self.t)
            if q_cmd is not None:
                if self.last_cmd is not None:
                    self.max_step = max(self.max_step, float(np.max(np.abs(q_cmd - self.last_cmd))))
                self.last_cmd = q_cmd.copy()
                self.q = q_cmd.copy()
            self.t += 1.0 / HZ

    def engage(self):
        return self.c.engage(self.q, self.t)


def _leader_pose_shifted(q0, dp, T_tool_L):
    """Leader joints whose TCP (fk @ T_tool_L) is the home TCP translated by dp."""
    T = ur_kin.fk(q0) @ T_tool_L
    T[:3, 3] += dp
    T_fl = T @ np.linalg.inv(T_tool_L)
    sols = [ur_kin.wrapped_nearest(s, q0) for s in ur_kin.ik_analytic(T_fl)]
    assert sols, "no IK solution for the shifted leader pose"
    return min(sols, key=lambda s: float(np.max(np.abs(s - q0))))


def test_eef_engage_gates_and_5cm_translation(params):
    c = TeleopController(params, "eef", HZ)
    L = FakeLeader(HOME)
    rig = Rig(c, L)
    # before any tick: no baseline
    ok, key, _ = c.engage(HOME, 0.0)
    assert not ok and key in ("leader_stale", "no_command_baseline")
    rig.run(1.0)
    assert c.eef_state == "HOLD" and c.q_cmd is not None
    assert np.allclose(c.q_cmd, HOME)  # holding: never mirrored the (identical) leader
    ok, key, msg = rig.engage()
    assert ok, (key, msg)
    assert c.engaged and c.eef_state == "ENGAGED"
    p0, _ = c.cmd_tcp()
    T_tool_L = ur_kin.xyz_rpy_to_T(params["tool_l_xyz_rpy"])
    L.set_pose(_leader_pose_shifted(HOME, np.array([-0.05, 0.0, 0.0]), T_tool_L))
    rig.max_step = 0.0
    rig.run(2.0)
    p1, _ = c.cmd_tcp()
    dp = p1 - p0
    assert np.linalg.norm(dp - [-0.05, 0.0, 0.0]) < 2e-3, dp
    assert rig.max_step <= params["max_step_rad"] * (1 + 1e-6), rig.max_step
    assert rig.max_step > 0.0
    info = c.info(rig.t)
    assert info["state"] == "ENGAGED" and info["ctrl_state"] == "ENGAGED"
    assert info["sigma_min"] > params["sigma_warn"]
    assert info["branch_id"] >= 0
    # actual TCP of the plant (perfect tracking) equals the commanded TCP
    T_act = ur_kin.fk(rig.q) @ c.T_tool_R
    assert np.linalg.norm(T_act[:3, 3] - p1) < 1e-9


def test_eef_disengaged_freezes_command(params):
    c = TeleopController(params, "eef", HZ)
    L = FakeLeader(HOME)
    rig = Rig(c, L)
    rig.run(1.0)
    assert rig.engage()[0]
    T_tool_L = ur_kin.xyz_rpy_to_T(params["tool_l_xyz_rpy"])
    L.set_pose(_leader_pose_shifted(HOME, np.array([0.0, 0.03, 0.0]), T_tool_L))
    rig.run(1.0)
    assert c.disengage()[0]
    assert c.eef_state == "DISENGAGED" and not c.engaged
    frozen = c.q_cmd.copy()
    L.set_pose(_leader_pose_shifted(HOME, np.array([0.0, -0.05, 0.05]), T_tool_L))
    rig.max_step = 0.0
    rig.run(1.0)
    assert np.array_equal(c.q_cmd, frozen)
    assert rig.max_step == 0.0
    # re-engage from DISENGAGED works once the leader is still again (zero jump)
    ok, key, msg = rig.engage()
    assert ok, (key, msg)
    assert np.array_equal(c.q_cmd, frozen)


def test_eef_engage_refused_while_leader_moving_and_pos_scale(params):
    c = TeleopController(params, "eef", HZ)
    L = FakeLeader(HOME, wiggle_amp=0.2, wiggle_hz=0.5, wiggle_joints=(0,))
    rig = Rig(c, L)
    rig.run(1.0)
    ok, key, _ = rig.engage()
    assert not ok and key in ("leader_moving", "filter_not_settled")
    assert np.allclose(c.q_cmd, HOME)  # the arm never followed the wiggling leader
    L.wiggle_amp = 0.0
    L.set_pose(HOME)
    rig.run(1.0)
    assert c.set_pos_scale(0.5) == (True, "ok", "pos_scale=0.50")
    assert not c.set_pos_scale(5.0)[0]
    assert rig.engage()[0]
    assert c.set_pos_scale(0.25)[1] == "deferred"
    assert c._eef.pos_scale == 0.5
    T_tool_L = ur_kin.xyz_rpy_to_T(params["tool_l_xyz_rpy"])
    p0, _ = c.cmd_tcp()
    L.set_pose(_leader_pose_shifted(HOME, np.array([-0.04, 0.0, 0.0]), T_tool_L))
    rig.run(2.0)
    p1, _ = c.cmd_tcp()
    assert np.linalg.norm((p1 - p0) - [-0.02, 0.0, 0.0]) < 2e-3  # scaled by 0.5


def test_eef_leader_stale_autodisengages(params):
    c = TeleopController(params, "eef", HZ)
    L = FakeLeader(HOME)
    rig = Rig(c, L)
    rig.run(1.0)
    assert rig.engage()[0]
    rig.run(0.8, feed_leader=False)
    assert c.eef_state == "DISENGAGED" and c.info(rig.t)["auto_reason"] == "leader_stale"
    rig.run(1.0)  # stream back: re-seeded, holding
    assert c.info(rig.t)["seeded"]


def test_reclutch_zero_delta(params):
    c = TeleopController(params, "eef", HZ)
    L = FakeLeader(HOME)
    rig = Rig(c, L)
    rig.run(1.0)
    assert not c.reclutch(rig.t)[0]  # not engaged
    assert rig.engage()[0]
    T_tool_L = ur_kin.xyz_rpy_to_T(params["tool_l_xyz_rpy"])
    L.set_pose(_leader_pose_shifted(HOME, np.array([0.0, 0.0, 0.04]), T_tool_L))
    rig.run(1.5)
    q_before = c.q_cmd.copy()
    ok, key, msg = c.reclutch(rig.t)
    assert ok, (key, msg)
    rig.run(0.5)
    assert np.allclose(c.q_cmd, q_before, atol=1e-6)  # zero delta after reclutch


def test_joint_mode_follows_when_engaged_and_holds_otherwise(params):
    c = TeleopController(params, "joint", HZ)
    L = FakeLeader(HOME)
    rig = Rig(c, L)
    rig.run(0.5)
    target = HOME + np.array([0.2, 0.0, -0.1, 0.0, 0.0, 0.3])
    L.set_pose(target)
    rig.run(1.0)
    assert np.allclose(c.q_cmd, HOME)  # HOLD while not engaged, even though the leader moved
    ok, key, msg = rig.engage()
    assert ok, (key, msg)
    rig.max_step = 0.0
    rig.run(2.0)
    assert np.max(np.abs(c.q_cmd - target)) < 0.01, c.q_cmd - target
    assert rig.max_step <= params["max_step_rad"] * (1 + 1e-6)
    assert c.disengage()[0]
    frozen = c.q_cmd.copy()
    L.set_pose(HOME)
    rig.run(1.0)
    assert np.array_equal(c.q_cmd, frozen)


def test_joint_mode_engage_gap_gate(params):
    c = TeleopController(params, "joint", HZ, joint_engage_max_gap_rad=0.5)
    L = FakeLeader(HOME + np.array([1.0, 0, 0, 0, 0, 0]))
    rig = Rig(c, L)
    rig.run(0.5)
    ok, key, _ = rig.engage()
    assert not ok and key == "gap_too_large"


def test_set_control_mode_only_when_disengaged(params):
    c = TeleopController(params, "eef", HZ)
    L = FakeLeader(HOME)
    rig = Rig(c, L)
    rig.run(1.0)
    assert rig.engage()[0]
    assert not c.set_control_mode("joint")[0]
    c.disengage()
    assert c.set_control_mode("joint")[0]
    assert c.mode == "joint" and c.eef_state == "HOLD"
    rig.run(0.5)
    assert np.allclose(c.q_cmd, HOME)
    assert not c.set_control_mode("bogus")[0]


def test_single_unwrap_chain_with_reanchor(params):
    """The controller consumes the leader's unwrapped chain; at seed the chain is
    re-anchored onto the arm's branch (callback), so consumed == published."""
    L = FakeLeader(HOME)
    c = TeleopController(params, "eef", HZ, reanchor=L.reanchor)
    q_actual = HOME.copy()
    # the leader's raw wrist_3 sits on the other 2*pi branch (+3.165 vs arm -3.118)
    q_lead = HOME.copy(); q_lead[5] = HOME[5] + 2 * np.pi
    L.set_pose(q_lead)
    t = 0.0
    s = L.sample_now(t); c.ingest(s)
    assert s.q_unwrapped[5] == pytest.approx(q_lead[5])
    c.tick(q_actual, t)  # seed -> reanchor(shift = -2*pi on joint 5)
    assert c._raw_target[5] == pytest.approx(HOME[5])
    for k in range(1, 60):
        t = k / LEADER_HZ
        s = L.sample_now(t); c.ingest(s)
        for _ in range(int(HZ / LEADER_HZ)):
            c.tick(q_actual, t)
        assert np.allclose(s.q_unwrapped, c._raw_target), (s.q_unwrapped, c._raw_target)
    info = c.info(t)
    assert np.allclose(info["lead_branch_shift"], 0.0)
    assert s.q_unwrapped[5] == pytest.approx(HOME[5])
