"""ZmqPolicy against the stub REP server (real wire protocol), ReplayPolicy on one take,
ZeroPolicy (DESIGN.md §2.2)."""
import os
import time

import numpy as np
import pytest

from sim_collect.eval.policies import PolicyRefused, ReplayPolicy, ZeroPolicy, ZmqPolicy, make_policy
from sim_collect.tests.stub_policy_server import StubPolicyServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TAKES = os.path.join(ROOT, "ros2_ur_ws", "gello_logs", "sim")
FAKE_OBS = {"state": [-3.3, -1.5, 1.6, -1.5, -1.6, -3.1, 0.0], "cam1_jpeg": b"\xff\xd8\xff\xe0cam1", "cam2_jpeg": b"\xff\xd8\xff\xe0cam2", "t": 0.0}


def _first_take():
    if not os.path.isdir(TAKES):
        pytest.skip("no recorded sim takes")
    takes = sorted(d for d in os.listdir(TAKES) if os.path.isfile(os.path.join(TAKES, d, "vectors.h5")))
    if not takes:
        pytest.skip("no recorded sim takes")
    return os.path.join(TAKES, takes[0])


def test_obs_assembler_is_ros_free():
    import sys
    assert "rclpy" not in sys.modules


def test_zmq_policy_reset_act_against_stub():
    with StubPolicyServer(port=0) as s:
        p = ZmqPolicy(s.endpoint, timeout_s=0.5)
        assert p.reset({}) is None
        assert p.meta["server"]["state_dim"] == 7 and p.meta["server"]["policy_type"] == "stub"
        a = p.act(FAKE_OBS)
        assert a == FAKE_OBS["state"][:6] + [0.0]             # "hold" mode echoes q
        assert s.n_reset == 1 and s.n_act == 1
        assert s.calls[-1]["cam1_bytes"] == len(FAKE_OBS["cam1_jpeg"])
        s.action = [1, 2, 3, 4, 5, 6, 0.7]
        assert p.act(FAKE_OBS) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7]
        st = p.stats()
        assert st["n_acts"] == 2 and st["n_faults"] == 0 and st["rtt_ms_median"] is not None
        p.close()


def test_zmq_policy_timeout_is_a_fault_and_recovers():
    with StubPolicyServer(port=0, delay_s=0.8) as s:
        p = ZmqPolicy(s.endpoint, timeout_s=0.3)
        t0 = time.time()
        assert p.act(FAKE_OBS) is None
        assert 0.25 < time.time() - t0 < 0.75 and "timeout" in p.last_error
        assert p.n_faults == 1
        time.sleep(0.9)                                        # let the stub finish its late reply
        s.delay_s = 0.0
        assert p.act(FAKE_OBS) is not None                     # socket was rebuilt
        p.close()


def test_zmq_policy_refuses_eef_checkpoint():
    with StubPolicyServer(port=0, reset_extra={"policy_type": "act", "state_dim": 16, "action_dim": 7}) as s:
        p = ZmqPolicy(s.endpoint, timeout_s=0.5)
        with pytest.raises(PolicyRefused):
            p.reset({})
        p.close()


def test_zmq_policy_server_error_and_bad_action_are_faults():
    with StubPolicyServer(port=0, action=[float("nan")] * 7) as s:
        p = ZmqPolicy(s.endpoint, timeout_s=0.5)
        assert p.act(FAKE_OBS) is None and "non-finite" in p.last_error
        assert p.act({"state": FAKE_OBS["state"], "cam1_jpeg": None, "cam2_jpeg": None, "t": 0.0}) is None
        p.close()
    with pytest.raises(ValueError):
        ZmqPolicy("nonsense")


def test_replay_policy_on_one_take():
    take = _first_take()
    p = ReplayPolicy(take)
    assert p.needs_images is False
    info = p.reset({"dwell_s": 1.0})
    lay = info["layout_override"]
    assert set(lay) == {"carrot", "pot"}
    for o in lay.values():
        assert len(o["pos"]) == 3 and len(o["quat_wxyz"]) == 4
        assert abs(np.linalg.norm(o["quat_wxyz"]) - 1.0) < 1e-3
    assert len(info["q0"]) == 6 and info["q0"][0] < -2.5
    assert info["max_steps_hint"] == int(np.ceil((p.duration_s + 2.0) * 30))
    a0 = p.act({"t": 0.0})
    assert len(a0) == 7 and a0[:6] == [float(v) for v in p.cmd_q[np.argmin(np.abs(p.cmd_t - p.t_base))]]
    assert 0.0 <= a0[6] <= 1.0
    a_end = p.act({"t": p.duration_s + 100.0})
    assert a_end[:6] == [float(v) for v in p.cmd_q[-1]]
    assert p.meta["recorder_task_success_at_stop"] is True


def test_zero_policy_holds_home():
    z = ZeroPolicy()
    assert z.reset({"home_joints": [1, 2, 3, 4, 5, 6]}) is None
    assert z.act({}) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]
    assert make_policy("zero").meta["policy"] == "zero"
    with pytest.raises(ValueError):
        make_policy("replay")
    with pytest.raises(ValueError):
        make_policy("bogus")
