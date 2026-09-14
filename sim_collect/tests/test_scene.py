"""scene.py: deterministic build, cameras, FK parity with ur_kin, object sites, layout."""
import os

import mujoco
import numpy as np
import pytest

from sim_collect import scene
from ur_gello_bringup import ur_kin

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "carrot_in_pot_sim.yaml")


@pytest.fixture(scope="module")
def cfg():
    return scene.SceneConfig.load(CFG)


@pytest.fixture(scope="module")
def built(cfg):
    return scene.build_scene(cfg, scene.sample_layout(cfg, 1))


@pytest.fixture(scope="module")
def model(built):
    return built.load_model()


def _id(model, kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    assert i >= 0, f"{name} not in model"
    return i


def test_build_is_deterministic(cfg):
    lay = scene.sample_layout(cfg, 5)
    a = scene.build_scene(cfg, lay)
    b = scene.build_scene(cfg, lay)
    assert a.xml == b.xml
    assert a.assets.keys() == b.assets.keys()
    assert a.meta["layout"] == b.meta["layout"]


def test_lookat_xyaxes_is_orthonormal_and_points_at_target():
    pos = np.array([0.7, 0.0, 0.571]); tgt = np.array([0.45, 0.0, 0.0])
    xy = scene.lookat_xyaxes(pos, tgt, [0, 0, 1])
    x, y = xy[:3], xy[3:]
    z = np.cross(x, y)
    assert abs(np.linalg.norm(x) - 1) < 1e-9 and abs(np.linalg.norm(y) - 1) < 1e-9
    assert abs(x @ y) < 1e-9
    # camera looks along -z toward the target
    assert np.allclose(-z, (tgt - pos) / np.linalg.norm(tgt - pos), atol=1e-9)
    assert y[2] > 0  # up-ish


def test_cameras_exist_with_expected_poses(model, built):
    for name in ("cam1", "cam1_depth", "cam2", "cam2_depth"):
        _id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    c1 = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam1")
    assert np.linalg.norm(model.cam_pos[c1] - [0.70, 0.0, 0.571]) < 1e-3  # +x: the real robot's side
    assert abs(model.cam_fovy[c1] - 42.0) < 1e-9
    c1d = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam1_depth")
    assert abs(model.cam_fovy[c1d] - 58.7) < 1e-9
    assert abs(np.linalg.norm(model.cam_pos[c1d] - model.cam_pos[c1]) - 0.015) < 1e-6
    # cam2 hangs off wrist_3_link and looks roughly along tool +Z at the home pose (down)
    c2 = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam2")
    w3 = _id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    assert model.cam_bodyid[c2] == w3
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    R = d.cam_xmat[c2].reshape(3, 3)
    look = -R[:, 2]
    assert look[2] < -0.9, f"wrist cam should look down at home, got {look}"
    # mounted on the +x world side of the flange (farthest from the base) and
    # pushed axial_m down the tool (z below the flange)
    att = _id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    assert d.cam_xpos[c2][0] > d.site_xpos[att][0] + 0.03
    assert d.cam_xpos[c2][2] < d.site_xpos[att][2] - 0.05
    # the home pose faces +x with J1 on the real robot's branch
    assert d.qpos[0] == pytest.approx(-3.302) and d.site_xpos[att][0] > 0.4
    assert set(built.meta["cameras"]) >= {"cam1", "cam1_depth", "cam2", "cam2_depth"}


def test_visual_offscreen_buffer(model, cfg):
    assert model.vis.global_.offwidth == 1280 and model.vis.global_.offheight == 720
    assert model.vis.quality.shadowsize == cfg.render.get("viewer_shadowsize", 2048)
    assert model.vis.quality.offsamples == cfg.render.get("offsamples", 0)


def test_fk_matches_ur_kin(model):
    d = mujoco.MjData(model)
    sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        q = rng.uniform(-3.0, 3.0, 6)
        d.qpos[:6] = q
        mujoco.mj_forward(model, d)
        T = ur_kin.fk(q)
        worst = max(worst, float(np.linalg.norm(d.site_xpos[sid] - T[:3, 3])))
        assert np.allclose(d.site_xmat[sid].reshape(3, 3), T[:3, :3], atol=1e-6)
    assert worst < 0.5e-3, f"FK mismatch {worst * 1e3:.3f} mm"


def test_qpos_layout_and_keyframe(model, cfg):
    assert model.nu == 7
    for i, n in enumerate(scene.UR_JOINT_NAMES):
        assert model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] == i
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    assert np.allclose(d.qpos[:6], cfg.home_joints)
    assert np.allclose(d.ctrl[:6], cfg.home_joints)
    assert d.ctrl[6] == 0.0  # gripper open
    # one free joint per object, after the arm + gripper joints
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(free) == len(cfg.objects)
    assert min(model.jnt_qposadr[j] for j in free) >= 14


def test_objects_have_sites_and_masses(model, built):
    for name, o in built.meta["objects"].items():
        _id(model, mujoco.mjtObj.mjOBJ_BODY, o["body"])
        _id(model, mujoco.mjtObj.mjOBJ_SITE, o["center_site"])
        bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, o["frame_body"])
        mass = float(model.body_subtreemass[bid])
        assert 0.02 <= mass <= 1.0, (name, mass)   # strawberry (YCB) is 25 g
        if o["kind"] == "container":
            op = _id(model, mujoco.mjtObj.mjOBJ_SITE, o["opening_site"])
            _id(model, mujoco.mjtObj.mjOBJ_SITE, o["floor_site"])
            assert model.site_size[op][0] == pytest.approx(o["size"]["inner_radius"])
    assert built.meta["objects"]["carrot"]["size"]["source"] == "placeholder" or \
        os.path.isfile(built.meta["objects"]["carrot"]["size"]["source"])


def test_sensors_and_names(model, built):
    n = built.meta["names"]
    _id(model, mujoco.mjtObj.mjOBJ_SITE, n["ft_site"])
    _id(model, mujoco.mjtObj.mjOBJ_SENSOR, n["force_sensor"])
    _id(model, mujoco.mjtObj.mjOBJ_SENSOR, n["torque_sensor"])
    j = _id(model, mujoco.mjtObj.mjOBJ_JOINT, n["gripper_driver_joint"])
    # driver joint range is 0..0.8 in 2f85.xml; grip_pos maps 0..0.871 so an empty-hand
    # close (0.7822 rad measured) reads 0.90 like the real Robotiq driver
    assert np.allclose(model.jnt_range[j], [0.0, 0.8])
    assert n["gripper_driver_range"] == pytest.approx([0.0, 0.871])
    a = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n["gripper_actuator"])
    assert model.actuator_ctrlrange[a][1] == 255.0
    assert model.opt.timestep == pytest.approx(0.002)


def test_floor_present_and_texture_reported(model, built):
    _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert built.meta["floor"]["texture_used"]  # real png or "builtin:checker"


def test_layout_seeded_and_non_overlapping(cfg):
    a = scene.sample_layout(cfg, 42)
    b = scene.sample_layout(cfg, 42)
    c = scene.sample_layout(cfg, 43)
    assert a == b and a != c
    names = cfg.object_names()
    # Objects whose nominal spot cannot satisfy the footprint constraint fall back
    # to their nominal position (flagged): e.g. pear (0.62,0.14,r0.06) vs carrot
    # (0.45,0.18,r0.10) are 0.175 m apart but need 0.18 in the shipped yaml.
    fell_back = [n for n in names if a[n]["fallback"]]
    assert set(fell_back) <= {"pear"}, fell_back
    for i, n1 in enumerate(names):
        p1 = np.asarray(a[n1]["pos"][:2]); r1 = cfg.object_spec(n1)["radius_m"]
        assert np.linalg.norm(p1) >= cfg.layout["base_keepout_radius_m"]
        assert a[n1]["pos"][2] == pytest.approx(cfg.layout["drop_height_m"])
        for n2 in names[i + 1:]:
            if a[n1]["fallback"] or a[n2]["fallback"]:
                continue
            p2 = np.asarray(a[n2]["pos"][:2]); r2 = cfg.object_spec(n2)["radius_m"]
            assert np.linalg.norm(p1 - p2) >= r1 + r2 + cfg.layout["min_gap_m"] - 1e-9
    # looking along +x from the base: left = food (+y), right = containers (-y)
    assert a["carrot"]["pos"][1] > 0 > a["pot"]["pos"][1]
    assert a["carrot"]["pos"][0] > 0 and a["pot"]["pos"][0] > 0
    # attempt > 0 gives a different deterministic draw for the same seed
    d0, d1, d1b = scene.sample_layout(cfg, 42, 0), scene.sample_layout(cfg, 42, 1), scene.sample_layout(cfg, 42, 1)
    assert d0 == a and d1 == d1b and d1 != d0 and d1["_attempt"] == 1


def test_objects_settle_on_floor(cfg):
    built = scene.build_scene(cfg, scene.sample_layout(cfg, 2))
    m = built.load_model(); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    for name, o in built.meta["objects"].items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["frame_body"])
        assert d.xpos[bid][2] > -2e-3, (name, d.xpos[bid])
        assert d.xpos[bid][2] < 0.03, (name, d.xpos[bid])
        j = m.body_jntadr[bid]
        v = d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6]
        # linear speed must be ~0; a settled convex hull keeps a decaying rocking
        # chatter (|w| up to ~0.25 rad/s, position stable) — see assets/objects/README.md
        assert np.linalg.norm(v[:3]) < 0.02, (name, v)
        assert np.linalg.norm(v[3:]) < 0.5, (name, v)
