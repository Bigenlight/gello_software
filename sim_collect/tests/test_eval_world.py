"""EvalWorld: reset determinism, deploy clamps + 250 Hz slew, success latch dwell, failure
detection, observation format, episode state log round trip (DESIGN.md §2.1)."""
import os

import numpy as np
import pytest

from sim_collect.eval.world import EvalWorld, load_episode_state
from sim_collect.tests.conftest import needs_display

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG = os.path.join(ROOT, "sim_collect", "configs", "carrot_in_pot_sim.yaml")


@pytest.fixture(scope="module")
def world():
    w = EvalWorld(CFG, video=False)
    yield w
    w.close()


def test_eval_block_loaded(world):
    assert world.max_step_rad == 0.0025 and world.soft_start_s == 0.7 and world.max_dev_rad == 0.5
    assert "NOT carrot_eef_limits.json" in world.envelope_source
    from sim_collect.eval.world import EVAL_DEFAULTS
    assert EVAL_DEFAULTS["joint_limits_lo"] == world.ev["joint_limits_lo"]   # yaml == code defaults (sim-derived)
    assert world.dwell_s == 1.0 and world.policy_hz == 30.0 and world.steps_per_tick == 2
    assert np.all(world.home_joints > world.joint_limits_lo) and np.all(world.home_joints < world.joint_limits_hi)
    assert np.all(world.start_pose > world.joint_limits_lo) and np.all(world.start_pose < world.joint_limits_hi)
    # the default envelope is the one derive_limits.py produces from the sim takes (real rule)
    from sim_collect.eval.derive_limits import expand, scan_takes
    takes_dir = os.path.join(ROOT, "ros2_ur_ws", "gello_logs", "sim")
    if os.path.isdir(takes_dir):
        sc = scan_takes(sorted(os.path.join(takes_dir, d) for d in os.listdir(takes_dir)))
        lo, hi = expand(sc["joint_raw_min"], sc["joint_raw_max"], 1.2)
        assert np.allclose(lo, world.joint_limits_lo, atol=1e-9) and np.allclose(hi, world.joint_limits_hi, atol=1e-9)


def test_derive_limits_reproduces_the_real_rule():
    from sim_collect.eval.derive_limits import expand
    lo, hi = expand(np.array([-3.7027, -1.7285, 1.3382, -2.5868, -1.8701, -4.4088]),
                    np.array([-2.6809, -0.9593, 2.0709, -1.5424, -1.2622, -2.0574]))
    assert lo.tolist() == [-3.8049, -1.8055, 1.2649, -2.6913, -1.9309, -4.6440]   # carrot_eef_limits.json
    assert hi.tolist() == [-2.5787, -0.8823, 2.1442, -1.4379, -1.2014, -1.8222]


def test_set_envelope_and_clamp_hits(world):
    world.reset(seed=0)
    q = world.q()
    a = list(q) + [1.5]
    a[2] = 3.0                                        # elbow beyond any envelope
    world.apply(a)
    ch = world.clamp_hits
    assert ch["steps"] == 1 and ch["envelope"] == 1 and ch["envelope_per_joint"][2] == 1 and ch["grip"] == 1
    assert ch["max_dev"] == 1 and ch["max_dev_per_joint"][2] == 1
    lo, hi = world.joint_limits_lo.copy(), world.joint_limits_hi.copy()
    world.set_envelope(None, None)                    # model joint range only
    assert world.joint_limits_hi[2] == pytest.approx(np.pi, abs=1e-3)
    r = world.apply(list(q) + [0.0])
    assert not r["clamped_limit"] and world.clamp_hits["steps"] == 2 and world.clamp_hits["envelope"] == 1
    world.set_envelope(lo, hi)
    assert np.array_equal(world.joint_limits_lo, lo)


def test_reset_is_deterministic(world):
    o1 = world.reset(seed=3)
    lay1 = {k: v for k, v in world.layout.items()}
    q1 = world.data.qpos.copy()
    o2 = world.reset(seed=3)
    assert world.layout == lay1
    assert np.array_equal(world.data.qpos, q1)
    assert o1["state"] == o2["state"]
    world.reset(seed=4)
    assert world.layout["carrot"]["pos"] != lay1["carrot"]["pos"]     # a different seed moves things
    assert world.t == pytest.approx(0.0) and world.frame == 0


def test_reset_starts_at_home_on_minus_pi_branch(world):
    obs = world.reset(seed=0)
    st = np.asarray(obs["state"])
    assert st.shape == (7,)
    assert np.allclose(st[:6], world.home_joints, atol=0.01)
    assert st[0] < -2.5 and st[6] < 0.05           # J1 ~ -3.30 (not +2.98), gripper open


def test_deploy_clamps_in_order(world):
    world.reset(seed=0)
    q = world.q()
    a = list(q) + [0.0]
    a[2] = world.joint_limits_hi[2] + 1.0             # elbow: beyond the envelope AND > 0.5 rad away
    r = world.apply(a)
    assert r["clamped_limit"] and r["clamped_dev"]
    assert all(abs(r["target"][i] - q[i]) <= world.max_dev_rad + 1e-9 for i in range(6))
    assert r["target"][2] <= world.joint_limits_hi[2] + 1e-12
    # a target inside the envelope but 1 rad away binds only the max-deviation clamp
    a = list(q) + [0.0]
    a[3] -= 1.0                                       # wrist_1: -1.523 - 1 = -2.523 > lo -3.0359
    assert a[3] > world.joint_limits_lo[3]
    r = world.apply(a)
    assert r["clamped_dev"] and not r["clamped_limit"]
    assert r["target"][3] == pytest.approx(q[3] - 0.5)
    # grip: clip 0..1, identity, ctrl = grip * 255
    assert world.apply(list(q) + [2.0])["grip"] == 1.0 and world.data.ctrl[6] == 255.0
    assert world.apply(list(q) + [-1.0])["grip"] == 0.0 and world.data.ctrl[6] == 0.0
    assert world.apply(list(q) + [0.4])["grip"] == pytest.approx(0.4) and world.data.ctrl[6] == pytest.approx(102.0)
    with pytest.raises(ValueError):
        world.apply([0.0] * 6)
    with pytest.raises(ValueError):
        world.apply(list(q) + [float("nan")])


def test_slew_is_bounded_per_tick_and_soft_start_already_elapsed(world):
    """Real deploy parity: the bridge's 0.7 s soft start ran at move_to_start/resume, BEFORE
    start_execution, so the first tick of an episode already slews at the full max_step_rad."""
    assert world.soft_start_at_episode_start is False
    world.reset(seed=0)
    assert world.step_eff == world.max_step_rad                    # full step from the first tick
    q0 = world.q()
    a = list(q0) + [0.0]
    a[3] -= 1.0
    world.apply(a)
    world.step()
    assert world.step_eff == world.max_step_rad
    assert 0.0 < world.last_tick_max_delta <= world.max_step_rad + 1e-12
    moved = abs(world.data.ctrl[3] - q0[3])
    assert 7 * world.max_step_rad < moved <= 9 * world.max_step_rad + 1e-9   # 8-9 ticks in 1/30 s, full step
    max_delta = 0.0
    for _ in range(29):
        world.step()
        max_delta = max(max_delta, world.last_tick_max_delta)
    assert max_delta <= world.max_step_rad + 1e-12
    moved = abs(world.data.ctrl[3] - q0[3])
    # 1 s of 250 Hz ticks at the full step -> 0.625 rad, but the max-dev clamp bounds the target at 0.5
    assert 0.48 < moved <= 0.5 + 1e-9
    assert world.t == pytest.approx(1.0, abs=1e-6) and world.frame == 30


def test_soft_start_at_episode_start_opt_in(world):
    world.soft_start_at_episode_start = True
    try:
        world.reset(seed=0)
        assert world.step_eff == pytest.approx(0.15 * world.max_step_rad)
        q0 = world.q()
        a = list(q0) + [0.0]
        a[3] -= 1.0
        world.apply(a)
        world.step()
        assert world.step_eff < world.max_step_rad
        assert abs(world.data.ctrl[3] - q0[3]) < 4 * world.max_step_rad     # 9 ticks at ~15-20 % of the step
        for _ in range(29):
            world.step()
        assert world.step_eff == world.max_step_rad                # ramp over after 0.7 s
        moved = abs(world.data.ctrl[3] - q0[3])
        assert 0.40 < moved < 0.46                                 # (0.7*0.575 + 0.3) * 250 * 0.0025 ~ 0.44
    finally:
        world.soft_start_at_episode_start = False


def test_trajectory_is_deterministic_under_a_wiggling_policy(world):
    def run():
        world.reset(seed=3)
        home = world.q()
        traj = []
        for k in range(90):
            a = list(home) + [0.5 * (1 + np.sin(k / 7.0))]
            a[0] += 0.05 * np.sin(k / 5.0)
            a[2] -= 0.04 * np.cos(k / 9.0)
            world.apply(a)
            world.step()
            traj.append(np.concatenate([world.data.qpos, world.data.ctrl]))
        return np.asarray(traj)
    t1, t2 = run(), run()
    assert t1.shape == (90, world.model.nq + world.model.nu)
    assert float(np.max(np.abs(t1 - t2))) == 0.0


def test_state_uses_the_rebranched_joints(world):
    world.reset(seed=0)
    st = world.observe(images=False)["state"]
    assert st[:6] == world._branched_q()
    assert all(abs(st[i] - world.start_pose[i]) <= np.pi for i in range(6))


def test_hold_without_apply(world):
    world.reset(seed=1)
    q0 = world.q()
    ctrl0 = world.data.ctrl.copy()
    for _ in range(10):
        world.step()
    assert np.array_equal(world.data.ctrl, ctrl0)
    assert np.allclose(world.q(), q0, atol=0.01)


def test_success_latch_needs_dwell(world):
    world.reset(seed=0)
    pot_pos, pot_quat = world.object_pose("pot")
    world.set_object_pose("carrot", pot_pos + np.array([0.0, 0.0, 0.03]))
    world.apply(list(world.q()) + [0.0])            # gripper open
    for _ in range(12):                               # 0.4 s < dwell 1.0 s
        r = world.step()
    assert not r["success"] and not r["done"]
    assert world.t_first_ok is not None
    for _ in range(40):
        r = world.step()
        if r["done"]:
            break
    assert r["success"] and r["outcome"] == "success"
    assert r["t_success_s"] >= world.t_first_ok + world.dwell_s - 1e-9
    assert "IN" in r["detail"] and "grip=open" in r["detail"]
    # a closed gripper defeats the predicate even with the carrot inside
    world.reset(seed=0)
    world.set_object_pose("carrot", pot_pos + np.array([0.0, 0.0, 0.03]))
    world.apply(list(world.q()) + [1.0])
    for _ in range(45):
        r = world.step()
    assert not r["success"] and "grip=closed" in r["detail"]


def test_failure_detection(world):
    world.reset(seed=0)
    world.set_object_pose("carrot", [3.0, 3.0, 0.02])
    r = world.step()
    assert r["failure"] == "object_out_of_bounds:carrot" and r["outcome"] == "failure" and r["done"]


def test_object_and_tcp_pose_accessors(world):
    world.reset(seed=2)
    lay = world.layout
    p, q = world.object_pose("carrot")
    assert np.allclose(p[:2], lay["carrot"]["pos"][:2], atol=0.01) and abs(np.linalg.norm(q) - 1.0) < 1e-6
    T = world.tcp_pose()
    assert T.shape == (4, 4) and 0.3 < T[0, 3] < 0.7 and 0.1 < T[2, 3] < 0.5     # in front of the base
    assert world.object_site_pos("pot", "opening")[2] > world.object_site_pos("pot", "floor")[2]
    info = world.info()
    assert info["seed"] == 2 and set(info["objects"]) == {"carrot", "pot"} and info["task"]["food"] == "carrot"


def test_episode_state_log_round_trip(world, tmp_path):
    world.reset(seed=0)
    world.apply(list(world.q()) + [0.5])
    for _ in range(10):
        world.step()
    path = world.save_episode(str(tmp_path / "ep_0.h5"), {"note": "test"})
    st = load_episode_state(path)
    assert st.qpos.shape[1] == world.model.nq and st.qvel.shape[1] == world.model.nv and st.ctrl.shape[1] == world.model.nu
    assert 40 <= len(st.t_rel_s) <= 44                # 10/30 s at 125 Hz (+ the reset row)
    assert np.all(np.diff(st.t_rel_s) > 0) and st.ctrl[-1, 6] == pytest.approx(127.5)
    import h5py, json
    with h5py.File(path, "r") as f:
        assert json.loads(f["sim_mj_state"].attrs["columns"])[3] == "qpos0"
        assert f["sim_scene"].attrs["xml_sha256"] == world.xml_sha256
        assert json.loads(f["eval"].attrs["meta"])["note"] == "test"


@needs_display
def test_observe_renders_full_size_jpegs(world):
    import cv2
    obs = world.reset(seed=0)
    for cam in ("cam1", "cam2"):
        jpg = obs[f"{cam}_jpeg"]
        assert isinstance(jpg, bytes) and jpg[:2] == b"\xff\xd8"
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (720, 1280, 3)
    assert len(obs["state"]) == 7 and obs["state"][0] < -2.5
    assert world.observe(images=False)["cam1_jpeg"] is None
    assert "rgb" not in obs                           # only with video=True


@needs_display
def test_video_writes_mp4(tmp_path):
    w = EvalWorld(CFG, video=True)
    try:
        obs = w.reset(seed=0, video_dir=str(tmp_path), video_tag="ep_x")
        assert obs["rgb"]["cam1"].shape == (720, 1280, 3)
        for _ in range(3):
            w.step()
            w.observe()
        w._close_video()
        for cam in ("cam1", "cam2"):
            p = tmp_path / f"ep_x_{cam}.mp4"
            assert p.is_file() and p.stat().st_size > 1000
    finally:
        w.close()
