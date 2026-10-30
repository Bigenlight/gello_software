"""sim_main.py headless smoke test: FakeLeader, --no-viewer, IPC round trips.

No rendering and no DISPLAY are needed (physics only). Uses SIM_COLLECT_IPC=tcp
(set by conftest) so the sockets are plain localhost ports.
"""
import os
import subprocess
import sys
import time

import pytest

from sim_collect import ipc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG = os.path.join(ROOT, "sim_collect", "configs", "carrot_in_pot_sim.yaml")

STATE_KEYS = {
    "t", "sim_t", "tick", "control_mode", "eef_state", "eef_info", "engaged", "pos_scale",
    "q_lead_raw", "q_lead_unwrapped", "q_lead_f", "qd_lead", "trigger", "leader_t",
    "q_cmd", "q", "qd", "eff", "tcp_pos", "tcp_quat_xyzw", "cmd_tcp_pos", "cmd_tcp_quat_xyzw",
    "wrench", "grip_cmd", "grip_pos", "qpos_full", "qvel_full", "objects", "task",
}


def _env():
    env = dict(os.environ)
    env["SIM_COLLECT_IPC"] = "tcp"
    env["PYTHONPATH"] = os.pathsep.join(
        [ROOT, os.path.join(ROOT, "ros2_ur_ws", "src", "ur_gello_bringup"), env.get("PYTHONPATH", "")])
    env.setdefault("MUJOCO_GL", "glfw")
    return env


def test_no_viewer_import_is_lazy():
    code = ("import sys, sim_collect.sim_main; "
            "assert 'mujoco.viewer' not in sys.modules, 'mujoco.viewer imported eagerly'; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def _ports_free(*ports) -> bool:
    import socket
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                return False
    return True


def _state_after(sub, min_tick: int, timeout_s: float = 5.0):
    """First state message published AFTER control tick `min_tick`. NOTE
    `Subscriber.latest()` only drains the ZMQ queue (HWM 4); at 250 Hz the kernel
    socket buffers hold hundreds more, so 'latest' can be far behind — and a
    message can be in flight from the tick that serviced a REP command. Key on
    the tick counter reported by get_status, never on wall-clock age."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = sub.recv(500)
        if m is not None and m["tick"] > min_tick:
            return m
    raise AssertionError(f"no state message after tick {min_tick}")


def test_headless_smoke_with_fake_leader():
    # The endpoints are fixed per user (ipc.py); a stub server left by another
    # test file in this pytest process would answer instead of sim_main.
    if not _ports_free(6701, 6711):
        pytest.skip("tcp 6701/6711 busy (another test's stub server?) — run this file alone")
    proc = subprocess.Popen(
        [sys.executable, "-m", "sim_collect.sim_main", "--config", CFG, "--fake-leader", "--no-viewer",
         "--duration", "60"],
        cwd=ROOT, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cli = ipc.Client("sim_rep", timeout_ms=3000)
    sub = ipc.Subscriber("state_pub", "state")
    try:
        assert ipc.wait_for(cli, 30.0), "sim_main did not answer get_status"
        st = cli.call("get_status")
        assert st["ok"] and st["viewer"] is False and st["leader_fake"] is True
        assert st["control_mode"] == "eef" and st["engaged"] is False and st["eef_state"] == "HOLD"

        msgs = [m for m in (sub.recv(500) for _ in range(30)) if m is not None]
        assert len(msgs) >= 10, len(msgs)
        m = msgs[-1]
        missing = STATE_KEYS - set(m)
        assert not missing, missing
        assert len(m["q"]) == 6 and len(m["qpos_full"]) > 14 and len(m["wrench"]) == 6
        assert set(m["objects"]) >= {"carrot", "pot"}
        assert m["task"]["success"] is False
        assert m["grip_cmd"] == 0.0 and 0.0 <= m["grip_pos"] <= 0.1
        ticks = sorted(x["tick"] for x in msgs)
        assert ticks[-1] > ticks[0]

        # engage: the fake leader is still, so the gates pass after ~0.5 s of history
        rep = None
        for _ in range(20):
            rep = cli.call("engage")
            if rep["ok"]:
                break
            time.sleep(0.25)
        assert rep and rep["ok"], rep
        st = cli.call("get_status")
        assert st["engaged"] is True and st["eef_state"] == "ENGAGED", st
        m = _state_after(sub, st["tick"])
        assert m["engaged"] is True and m["eef_state"] == "ENGAGED"
        assert m["eef_info"]["ctrl_state"] == "ENGAGED" and m["eef_info"]["sigma_min"] > 0.1

        rep = cli.call("reset_scene", seed=3)
        assert rep["ok"] and rep["seed"] == 3 and "carrot" in rep["layout"]
        st = cli.call("get_status")
        assert st["engaged"] is False and st["eef_state"] == "HOLD" and st["layout_seed"] == 3

        assert cli.call("engage")["ok"] or True  # may be refused briefly (filter re-seed); not asserted
        rep = cli.call("disengage")
        assert rep["ok"] and rep["eef_state"] == "DISENGAGED"

        meta = cli.call("get_scene_meta")
        assert meta["ok"]
        md = meta["meta"]
        for k in ("git_commit", "mujoco_version", "config", "objects", "cameras", "layout", "layout_seed",
                  "chosen_food", "container", "control_mode", "gripper_mode", "pos_scale"):
            assert k in md, k
        assert md["layout_seed"] == 3 and set(md["cameras"]) >= {"cam1", "cam2"}

        assert cli.call("gripper_pause")["ok"] and cli.call("gripper_resume")["ok"]
        assert cli.call("home")["ok"]
        assert cli.call("set_control_mode", mode="joint")["ok"]
        assert cli.call("get_status")["control_mode"] == "joint"
        assert cli.call("set_control_mode", mode="eef")["ok"]
        assert not cli.call("bogus")["ok"]

        assert cli.call("shutdown")["ok"]
        out, _ = proc.communicate(timeout=15)
        assert proc.returncode == 0, out
        assert "leader port closed" in out
    finally:
        cli.close()
        sub.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)


# --------------------------------------------------------------------------- #
# In-process SimMain (no IPC, no viewer): startup / reset_scene collision handling
# --------------------------------------------------------------------------- #
import numpy as np  # noqa: E402

from sim_collect import sim_main as sm  # noqa: E402
from sim_collect.scene import SceneConfig, sample_layout  # noqa: E402
from ur_gello_bringup import ur_kin  # noqa: E402

HOME = np.array([-3.302, -1.563, 1.607, -1.523, -1.615, -3.118])
T_TOOL = ur_kin.xyz_rpy_to_T([0, 0, 0.174, 0, 0, 0])


def _q_with_tcp_at(p_xyz):
    """Joints with the home tool orientation and the TCP at p_xyz (nearest branch)."""
    T = ur_kin.fk(HOME) @ T_TOOL
    T[:3, 3] = p_xyz
    sols = [ur_kin.wrapped_nearest(s, HOME) for s in ur_kin.ik_analytic(T @ np.linalg.inv(T_TOOL))]
    assert sols
    return min(sols, key=lambda s: float(np.max(np.abs(s - HOME))))


@pytest.fixture
def app_factory():
    apps = []

    def make(leader_q):
        cfg = SceneConfig.load(CFG)
        a = sm.SimMain(cfg, fake_leader=True, no_viewer=True, publish=False)
        a.leader.set_pose(leader_q)
        apps.append(a)
        return a

    yield make
    for a in apps:
        a.leader.close()


def test_startup_falls_back_to_home_when_leader_pose_collides(app_factory):
    cfg = SceneConfig.load(CFG)
    pot_xy = cfg.layout["items"]["pot"]["nominal_pos"][:2]
    q_low = _q_with_tcp_at([pot_xy[0], pot_xy[1], 0.03])   # fingertips inside the pot
    app = app_factory(q_low)
    app.startup()
    assert "collides" in app.startup_note and "home_joints" in app.startup_note
    assert np.allclose(app.data.qpos[:6], HOME, atol=0.05)
    assert app._robot_contacts() == []


def test_reset_scene_resamples_then_falls_back_to_home(app_factory, monkeypatch):
    cfg = SceneConfig.load(CFG)
    pot_xy = cfg.layout["items"]["pot"]["nominal_pos"][:2]
    q_mid = _q_with_tcp_at([pot_xy[0], pot_xy[1], 0.20])   # low TCP right above the pot spot
    app = app_factory(HOME)
    app.startup()
    assert app.startup_note == ""
    # (a) attempt 0 collides (pot under the TCP), attempt 1 is clean -> re-sampled
    app.leader.set_pose(q_mid)
    import time; time.sleep(0.15)  # let the fake leader publish the new pose
    real = sample_layout

    def fake_layout(c, seed=None, attempt=0):
        lay = real(c, seed, attempt)
        if attempt == 0:
            lay["pot"] = {"pos": [pot_xy[0], pot_xy[1], 0.02], "yaw": 0.0, "fallback": False}
        else:
            lay["pot"] = {"pos": [pot_xy[0] + 0.30, pot_xy[1] - 0.30, 0.02], "yaw": 0.0, "fallback": False}
        return lay

    monkeypatch.setattr(sm, "sample_layout", fake_layout)
    layout = app.reset_scene(seed=11)
    assert layout["_attempt"] == 1 and "attempt 1" in app.last_reset_note
    assert np.allclose(app.data.qpos[:6], q_mid, atol=0.05)   # arm stayed at the leader pose
    assert app._robot_contacts() == []
    # (b) every attempt collides -> arm goes home and the note says so
    monkeypatch.setattr(sm, "sample_layout", lambda c, seed=None, attempt=0: fake_layout(c, seed, 0))
    app.cfg.layout["max_tries"] = 20
    layout = app.reset_scene(seed=12)
    assert "teleported to home_joints" in app.last_reset_note and "20 layouts" in app.last_reset_note
    assert np.allclose(app.data.qpos[:6], HOME, atol=0.05)
    rep = app.handle({"cmd": "reset_scene", "seed": 13})
    assert rep["ok"] and rep["arm_moved_home"] and "home_joints" in rep["msg"]
    # meta names the calibration source (fake leader here) and the startup note
    meta = app.scene_meta()
    assert "leader_calibration_source" in meta and "startup_note" in meta
