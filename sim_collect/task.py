"""Geometric task-success tests (DESIGN.md §3.4).

`carrot_in_pot`-style: success when the food's origin lies inside the container's
opening cylinder (above the container floor site, below the rim/opening site,
within the opening radius — all measured in the CONTAINER's body frame so a
tilted pot still tests correctly), AND |v_z| of the food < `max_vz`, AND the
gripper command is open (`grip_cmd < gripper_open_below`). ``stack`` tasks use
relative XY/Z alignment, uprightness and upper-object linear/angular speed.
Informational only — the converters take `--outcome` from the human.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np


@dataclasses.dataclass
class TaskConfig:
    name: str
    task_type: str = "containment"
    food: str = ""
    container: str = ""
    upper: str = ""
    lower: str = ""
    max_vz: float = 0.05
    max_linear_speed: float = 0.05
    max_angular_speed: float = 0.5
    max_xy_offset: float = 0.035
    min_relative_z: float = 0.01
    max_relative_z: float = 0.075
    max_tilt_deg: float = 20.0
    gripper_open_below: float = 0.3
    # resolved from scene meta (see `from_scene`)
    food_frame_body: str = ""
    container_body: str = ""
    opening_site: str = ""
    floor_site: str = ""
    upper_frame_body: str = ""
    lower_frame_body: str = ""

    @classmethod
    def from_scene(cls, task_cfg: Dict[str, Any], scene_meta: Dict[str, Any]) -> "TaskConfig":
        task_type = str(task_cfg.get("type", "containment"))
        objs = scene_meta["objects"]
        if task_type == "stack":
            upper = str(task_cfg["upper"])
            lower = str(task_cfg["lower"])
            for role, name in (("upper", upper), ("lower", lower)):
                if name not in objs:
                    raise KeyError(f"task {role} {name!r} is not a scene object ({sorted(objs)})")
            return cls(
                name=str(task_cfg.get("name", f"{upper}_on_{lower}")), task_type=task_type,
                upper=upper, lower=lower,
                max_linear_speed=float(task_cfg.get("max_linear_speed_m_s", 0.05)),
                max_angular_speed=float(task_cfg.get("max_angular_speed_rad_s", 0.5)),
                max_xy_offset=float(task_cfg.get("max_xy_offset_m", 0.035)),
                min_relative_z=float(task_cfg.get("min_relative_z_m", 0.01)),
                max_relative_z=float(task_cfg.get("max_relative_z_m", 0.075)),
                max_tilt_deg=float(task_cfg.get("max_tilt_deg", 20.0)),
                gripper_open_below=float(task_cfg.get("gripper_open_below", 0.3)),
                upper_frame_body=objs[upper]["frame_body"],
                lower_frame_body=objs[lower]["frame_body"],
            )
        if task_type != "containment":
            raise ValueError(f"unsupported task type {task_type!r}")
        food = str(task_cfg["food"])
        cont = str(task_cfg["container"])
        if food not in objs:
            raise KeyError(f"task food {food!r} is not a scene object ({sorted(objs)})")
        if cont not in objs:
            raise KeyError(f"task container {cont!r} is not a scene object ({sorted(objs)})")
        c = objs[cont]
        if not c.get("opening_site") or not c.get("floor_site"):
            raise ValueError(f"container {cont!r} lacks the *_opening / *_floor sites needed by task.py")
        return cls(
            name=str(task_cfg.get("name", f"{food}_in_{cont}")), task_type=task_type,
            food=food, container=cont,
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
        if cfg.task_type == "stack":
            self.upper_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, cfg.upper_frame_body)
            self.lower_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, cfg.lower_frame_body)
            if min(self.upper_body, self.lower_body) < 0:
                raise KeyError(f"stack task names not found in model: {cfg}")
            upper_joint = model.body_jntadr[self.upper_body]
            if upper_joint < 0 or model.jnt_type[upper_joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"upper body {cfg.upper_frame_body!r} has no free joint")
            self.upper_dofadr = int(model.jnt_dofadr[upper_joint])
            return
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
        if self.cfg.task_type == "stack":
            return self._evaluate_stack(data, grip_cmd)
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

    def _evaluate_stack(self, data: mujoco.MjData, grip_cmd: float) -> Tuple[bool, str]:
        upper_rotation = data.xmat[self.upper_body].reshape(3, 3)
        lower_rotation = data.xmat[self.lower_body].reshape(3, 3)
        relative = lower_rotation.T @ (data.xpos[self.upper_body] - data.xpos[self.lower_body])
        xy_offset = float(np.hypot(relative[0], relative[1]))
        upright_cos = float(np.clip(np.dot(upper_rotation[:, 2], lower_rotation[:, 2]), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(upright_cos)))
        velocity = data.qvel[self.upper_dofadr:self.upper_dofadr + 6]
        linear_speed = float(np.linalg.norm(velocity[:3]))
        angular_speed = float(np.linalg.norm(velocity[3:]))
        aligned = (xy_offset < self.cfg.max_xy_offset
                   and self.cfg.min_relative_z < relative[2] < self.cfg.max_relative_z
                   and tilt_deg < self.cfg.max_tilt_deg)
        still = (linear_speed < self.cfg.max_linear_speed
                 and angular_speed < self.cfg.max_angular_speed)
        open_ = grip_cmd < self.cfg.gripper_open_below
        ok = bool(aligned and still and open_)
        detail = (
            f"{'STACKED' if aligned else 'UNSTACKED'} xy={xy_offset:.3f}/{self.cfg.max_xy_offset:.3f} "
            f"z={relative[2]:.3f}[{self.cfg.min_relative_z:.3f},{self.cfg.max_relative_z:.3f}] "
            f"tilt={tilt_deg:.1f}/{self.cfg.max_tilt_deg:.1f}deg "
            f"v={linear_speed:.3f} w={angular_speed:.3f} grip={'open' if open_ else 'closed'}")
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
