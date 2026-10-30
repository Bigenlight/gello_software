"""Bowl-stack scene: orange-left onto purple-right."""
from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from sim_collect import scene, task


BOWL_CFG = "sim_collect/configs/bowl_stack_sim.yaml"
CARROT_CFG = "sim_collect/configs/carrot_in_pot_sim.yaml"


def _world():
    cfg = scene.SceneConfig.load(BOWL_CFG)
    built = scene.build_scene(cfg, scene.sample_layout(cfg, 0))
    model = built.load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return cfg, built, model, data


def test_bowl_scene_inherits_camera_and_has_coloured_aliases():
    cfg, built, model, _ = _world()
    carrot_cfg = scene.SceneConfig.load(CARROT_CFG)
    assert cfg.cameras == carrot_cfg.cameras
    assert cfg.object_names() == ["bowl_purple", "bowl_orange"]
    assert built.meta["objects"]["bowl_orange"]["body"] == "bowl_orange/bowl"
    assert built.meta["objects"]["bowl_purple"]["body"] == "bowl_purple/bowl"
    assert model.geom("bowl_orange/bowl_vis0").rgba[:3] == pytest.approx([0.95, 0.30, 0.06])
    assert model.geom("bowl_purple/bowl_vis0").rgba[:3] == pytest.approx([0.48, 0.16, 0.72])


def test_collection_seeds_are_small_variations_with_fixed_sides():
    cfg = scene.SceneConfig.load(BOWL_CFG)
    layouts = [scene.sample_layout(cfg, seed) for seed in cfg.layout["collection_seeds"]]
    assert len(layouts) == 20
    assert len({tuple(layout["bowl_orange"]["pos"][:2]) for layout in layouts}) == 20
    for layout in layouts:
        orange = np.asarray(layout["bowl_orange"]["pos"][:2])
        purple = np.asarray(layout["bowl_purple"]["pos"][:2])
        assert orange[1] > 0 > purple[1]
        assert np.linalg.norm(orange - [0.45, 0.18]) <= 0.030001
        assert np.linalg.norm(purple - [0.45, -0.20]) <= 0.030001


def test_orange_on_purple_is_success_only_when_released_and_still():
    cfg, built, model, data = _world()
    task_cfg = task.TaskConfig.from_scene(cfg.task, built.meta)
    evaluator = task.TaskEvaluator(model, task_cfg)
    lower_position = data.xpos[evaluator.lower_body].copy()
    orange = built.meta["objects"]["bowl_orange"]["frame_body"]

    task.set_free_body_pose(model, data, orange, lower_position + [0.0, 0.0, 0.04])
    mujoco.mj_forward(model, data)
    ok, detail = evaluator.evaluate(data, grip_cmd=0.0)
    assert ok, detail
    assert detail.startswith("STACKED")
    assert not evaluator.evaluate(data, grip_cmd=0.9)[0]

    task.set_free_body_pose(model, data, orange, lower_position + [0.08, 0.0, 0.04])
    mujoco.mj_forward(model, data)
    assert not evaluator.evaluate(data, grip_cmd=0.0)[0]


def test_dropped_orange_bowl_physically_nests_and_counts_as_success():
    cfg, built, model, data = _world()
    task_cfg = task.TaskConfig.from_scene(cfg.task, built.meta)
    evaluator = task.TaskEvaluator(model, task_cfg)
    lower_position = data.xpos[evaluator.lower_body].copy()
    orange = built.meta["objects"]["bowl_orange"]["frame_body"]
    task.set_free_body_pose(model, data, orange, lower_position + [0.0, 0.0, 0.08])
    mujoco.mj_forward(model, data)
    for _ in range(750):
        mujoco.mj_step(model, data)
    ok, detail = evaluator.evaluate(data, grip_cmd=0.0)
    assert ok, detail
    assert data.ncon < 100

    tilted = [math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0]
    task.set_free_body_pose(model, data, orange, lower_position + [0.0, 0.0, 0.04], tilted)
    mujoco.mj_forward(model, data)
    assert not evaluator.evaluate(data, grip_cmd=0.0)[0]
