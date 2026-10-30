"""leader.py: FakeLeader thread cadence, unwrap continuity, finite-difference qd."""
import math
import time

import numpy as np

import os
import tempfile

import pytest

from sim_collect.leader import FakeLeader, load_leader_calibration, port_holders, resolve_leader_config

HOME = [-3.302, -1.563, 1.607, -1.523, -1.615, -3.118]


def test_fake_leader_thread_streams_at_rate():
    L = FakeLeader(HOME, hz=30.0, trigger=0.25)
    L.start()
    try:
        time.sleep(1.0)
        s = L.latest()
        assert s is not None
        assert 20 <= s.seq <= 40, s.seq  # ~30 Hz over 1 s
        assert np.allclose(s.q_raw, HOME) and s.trigger == 0.25
        assert time.monotonic() - s.t < 0.2
    finally:
        L.stop()
        L.close()


def test_unwrap_is_continuous_across_pi():
    L = FakeLeader(HOME)
    q = list(HOME); q[5] = 3.10
    L.set_pose(q)
    s1 = L.sample_now(t=0.0)
    q[5] = -3.10  # physically +0.083 rad further, numerically a -6.2 rad jump
    L.set_pose(q)
    s2 = L.sample_now(t=1 / 30)
    assert s2.q_raw[5] == -3.10
    assert abs(s2.q_unwrapped[5] - (3.10 + (2 * math.pi - 6.20))) < 1e-9
    assert abs(s2.qd[5] - (2 * math.pi - 6.20) * 30) < 1e-6
    assert s2.seq == s1.seq + 1


def test_wiggle_and_offset():
    L = FakeLeader(HOME, wiggle_amp=0.1, wiggle_hz=1.0, wiggle_joints=(1,))
    s0 = L.sample_now(t=0.0)
    s1 = L.sample_now(t=0.25)
    assert abs(s1.q_raw[1] - (HOME[1] + 0.1)) < 1e-9
    assert s0.q_raw[1] == HOME[1]
    L.set_offset([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    s2 = L.sample_now(t=0.5)
    assert abs(s2.q_raw[5] - (HOME[5] + 0.5)) < 1e-9
    L.set_trigger(0.9)
    assert L.sample_now(t=0.6).trigger == 0.9


def test_reanchor_shifts_chain_and_latest():
    L = FakeLeader(HOME)
    s0 = L.sample_now(t=0.0)
    L.reanchor([0.0, 0.0, 0.0, 0.0, 0.0, -2 * math.pi])
    assert L.latest().q_unwrapped[5] == pytest.approx(s0.q_unwrapped[5] - 2 * math.pi)
    s1 = L.sample_now(t=1 / 30)
    assert s1.q_unwrapped[5] == pytest.approx(HOME[5] - 2 * math.pi)   # chain continues on the new branch
    assert s1.q_raw[5] == HOME[5]
    assert abs(s1.qd[5]) < 1e-9


def test_port_holders_finds_our_own_open_file():
    with tempfile.NamedTemporaryFile() as f:
        # our own pid is excluded, so spawn a child that holds the file
        import subprocess, sys, time as _t
        child = subprocess.Popen([sys.executable, "-c",
                                  f"import time; fh=open({f.name!r}); time.sleep(30)"])
        try:
            deadline = _t.monotonic() + 5
            holders = []
            while _t.monotonic() < deadline and not holders:
                holders = port_holders(f.name); _t.sleep(0.05)
            assert [h["pid"] for h in holders] == [child.pid], holders
            assert "time.sleep" in holders[0]["cmdline"]
        finally:
            child.kill(); child.wait()
    assert port_holders("/dev/null/does-not-exist") == []


def test_calibration_from_ros_yaml_and_start_joints():
    cal = load_leader_calibration("ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml")
    assert cal["joint_offsets"] == [0.0, 1.571, 4.712, 4.712, 4.712, 3.142]
    assert cal["joint_signs"] == [1, 1, -1, 1, 1, 1]
    assert cal["gripper_config"] == [7, 210.649609375, 168.849609375]
    assert cal["port"].endswith("FTBEO6QK-if00-port0")
    lc = resolve_leader_config({"calibration_source": "ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml",
                                "overrides": {"port": "/dev/ttyUSB9"}}, HOME)
    assert lc["port"] == "/dev/ttyUSB9" and lc["joint_offsets"][0] == 0.0
    assert lc["start_joints"] == HOME + [0.0]
    # the driver's +-2pi wrap (dynamixel.py:84-104) must map today's reading (~+2.98) to the -3.30 branch
    import numpy as np
    c_now, s_start = 2.98, lc["start_joints"][0]
    new_off = np.pi * 2 * np.round((-s_start + c_now) / (2 * np.pi)) * 1 + 0.0
    assert c_now - new_off == pytest.approx(-3.303, abs=1e-3)
    with pytest.raises(KeyError):
        resolve_leader_config({"overrides": {}}, HOME)
