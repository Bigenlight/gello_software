"""task.py: geometric success test using the container's opening/floor sites."""
import os

import mujoco
import numpy as np
import pytest

from sim_collect import scene, task

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "carrot_in_pot_sim.yaml")


@pytest.fixture(scope="module")
def world():
    cfg = scene.SceneConfig.load(CFG)
    built = scene.build_scene(cfg, scene.sample_layout(cfg, 0))
    m = built.load_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(250):  # settle the objects
        mujoco.mj_step(m, d)
    tc = task.TaskConfig.from_scene(cfg.task, built.meta)
    return cfg, built, m, d, tc


def _pot_pos(m, d, built):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, built.meta["objects"]["pot"]["frame_body"])
    return d.xpos[bid].copy()


def test_success_when_food_inside_container(world):
    cfg, built, m, d, tc = world
    ok0, detail0 = task.evaluate(m, d, tc, grip_cmd=0.0)
    assert ok0 is False and detail0.startswith("OUT")
    p = _pot_pos(m, d, built)
    task.set_free_body_pose(m, d, built.meta["objects"]["carrot"]["frame_body"], p + [0.0, 0.0, 0.02])
    mujoco.mj_forward(m, d)
    ok, detail = task.evaluate(m, d, tc, grip_cmd=0.0)
    assert ok is True, detail
    assert detail.startswith("IN")


def test_false_when_gripper_closed_or_moving_or_outside(world):
    cfg, built, m, d, tc = world
    p = _pot_pos(m, d, built)
    carrot = built.meta["objects"]["carrot"]["frame_body"]
    task.set_free_body_pose(m, d, carrot, p + [0.0, 0.0, 0.02])
    mujoco.mj_forward(m, d)
    assert task.evaluate(m, d, tc, grip_cmd=0.9)[0] is False      # gripper closed
    ev = task.TaskEvaluator(m, tc)
    d.qvel[ev.food_dofadr + 2] = 0.3                                # falling fast
    assert ev.evaluate(d, 0.0)[0] is False
    d.qvel[ev.food_dofadr + 2] = 0.0
    task.set_free_body_pose(m, d, carrot, p + [0.0, 0.0, 0.25])     # above the rim
    mujoco.mj_forward(m, d)
    assert ev.evaluate(d, 0.0)[0] is False
    task.set_free_body_pose(m, d, carrot, p + [0.15, 0.0, 0.02])    # outside the radius
    mujoco.mj_forward(m, d)
    assert ev.evaluate(d, 0.0)[0] is False
    task.set_free_body_pose(m, d, carrot, p + [0.0, 0.0, -0.05])    # below the floor
    mujoco.mj_forward(m, d)
    assert ev.evaluate(d, 0.0)[0] is False


def test_from_scene_validates_names(world):
    cfg, built, m, d, tc = world
    with pytest.raises(KeyError):
        task.TaskConfig.from_scene({"food": "durian", "container": "pot"}, built.meta)
    with pytest.raises(ValueError):
        task.TaskConfig.from_scene({"food": "carrot", "container": "carrot"}, built.meta)


def test_physically_dropped_food_counts(world):
    """Drop the carrot into the pot, settle 3 s: it RESTS on the pot floor (a few
    tenths of a mm below the `_floor` site) and must still count as success."""
    cfg, built, m, d, tc = world
    p = _pot_pos(m, d, built)
    carrot = built.meta["objects"]["carrot"]["frame_body"]
    task.set_free_body_pose(m, d, carrot, p + [0.0, 0.0, 0.15])
    mujoco.mj_forward(m, d)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    ev = task.TaskEvaluator(m, tc)
    ok, detail = ev.evaluate(d, 0.0)
    assert ok, detail
    rel_z = float(detail.split("z=")[1].split("/")[0])
    assert -0.005 < rel_z < 0.01, detail       # resting on the floor, within the slack
    assert ev.evaluate(d, 0.9)[0] is False      # gripper closed -> not yet a success
