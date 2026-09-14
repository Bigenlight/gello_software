"""Geometric task-success test (DESIGN.md §3.4).

`carrot_in_pot`-style: success when the food's origin lies inside the container's
opening cylinder (above the container floor site, below the rim/opening site,
within the opening radius — all measured in the CONTAINER's body frame so a
tilted pot still tests correctly), AND |v_z| of the food < `max_vz`, AND the
gripper command is open (`grip_cmd < gripper_open_below`). Informational only —
the converters take `--outcome` from the human.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np


@dataclasses.dataclass
class TaskConfig:
    name: str
    food: str
    container: str
    max_vz: float = 0.05
    gripper_open_below: float = 0.3
    # resolved from scene meta (see `from_scene`)
    food_frame_body: str = ""
    container_body: str = ""
    opening_site: str = ""
    floor_site: str = ""

    @classmethod
    def from_scene(cls, task_cfg: Dict[str, Any], scene_meta: Dict[str, Any]) -> "TaskConfig":
        food = str(task_cfg["food"])
        cont = str(task_cfg["container"])
        objs = scene_meta["objects"]
        if food not in objs:
            raise KeyError(f"task food {food!r} is not a scene object ({sorted(objs)})")
        if cont not in objs:
            raise KeyError(f"task container {cont!r} is not a scene object ({sorted(objs)})")
        c = objs[cont]
        if not c.get("opening_site") or not c.get("floor_site"):
            raise ValueError(f"container {cont!r} lacks the *_opening / *_floor sites needed by task.py")
        return cls(
            name=str(task_cfg.get("name", f"{food}_in_{cont}")), food=food, container=cont,
            max_vz=float(task_cfg.get("max_vz_m_s", 0.05)),
            gripper_open_below=float(task_cfg.get("gripper_open_below", 0.3)),
            food_frame_body=objs[food]["frame_body"], container_body=c["body"],
            opening_site=c["opening_site"], floor_site=c["floor_site"],
        )


class TaskEvaluator:
    """Caches the MuJoCo ids for `evaluate`; construct once per model."""

    FLOOR_SLACK_M = -0.005  # food origin may sit up to 5 mm below the floor site

    def __init__(self, model: mujoco.MjModel, cfg: TaskConfig) -> None:
        self.cfg = cfg
        self.food_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.food_frame_body)
        self.cont_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.container_body)
        self.opening = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.opening_site)
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.floor_site)
        if min(self.food_body, self.cont_body, self.opening, self.floor) < 0:
            raise KeyError(f"task names not found in model: {cfg}")
        self.radius = float(model.site_size[self.opening][0])
        jnt = model.body_jntadr[self.food_body]
        if jnt < 0 or model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"food body {cfg.food_frame_body!r} has no free joint")
        self.food_dofadr = int(model.jnt_dofadr[jnt])

    def evaluate(self, data: mujoco.MjData, grip_cmd: float) -> Tuple[bool, str]:
        p_food = data.xpos[self.food_body]
        R = data.xmat[self.cont_body].reshape(3, 3)
        p_floor = data.site_xpos[self.floor]
        p_rim = data.site_xpos[self.opening]
        rel = R.T @ (p_food - p_floor)
        depth = float((R.T @ (p_rim - p_floor))[2])
        r_xy = float(np.hypot(rel[0], rel[1]))
        # A food resting ON the container floor sits a fraction of a mm below the
        # `_floor` site (contact penetration), so the lower bound has slack.
        inside = (self.FLOOR_SLACK_M < rel[2] < depth) and (r_xy < self.radius)
        vz = float(data.qvel[self.food_dofadr + 2])
        still = abs(vz) < self.cfg.max_vz
        open_ = grip_cmd < self.cfg.gripper_open_below
        ok = bool(inside and still and open_)
        detail = (f"{'IN' if inside else 'OUT'} r={r_xy:.3f}/{self.radius:.3f} z={rel[2]:.3f}/{depth:.3f} "
                  f"vz={vz:+.3f} grip={'open' if open_ else 'closed'}")
        return ok, detail


def evaluate(model: mujoco.MjModel, data: mujoco.MjData, cfg: TaskConfig, grip_cmd: float = 0.0,
             _cache: Dict[int, TaskEvaluator] = {}) -> Tuple[bool, str]:
    """Convenience wrapper (id lookup cached per model object)."""
    key = id(model)
    ev = _cache.get(key)
    if ev is None or ev.cfg is not cfg:
        ev = _cache[key] = TaskEvaluator(model, cfg)
    return ev.evaluate(data, grip_cmd)


def set_free_body_pose(model: mujoco.MjModel, data: mujoco.MjData, frame_body: str,
                       pos, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> None:
    """Teleport a free-jointed body (used by reset_scene and the task tests)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_body)
    jnt = model.body_jntadr[bid]
    qa, da = int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])
    data.qpos[qa:qa + 3] = np.asarray(pos, dtype=float)
    data.qpos[qa + 3:qa + 7] = np.asarray(quat_wxyz, dtype=float)
    data.qvel[da:da + 6] = 0.0
