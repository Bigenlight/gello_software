"""Scene assembly for sim_collect (DESIGN.md §3).

`SceneConfig.load(yaml)` -> `build_scene(cfg, layout)` -> (xml string, assets dict,
meta). The build is DETERMINISTIC given the config file and the layout dict, so
process [A] (sim_main) and process [B] (capture) compile byte-identical models
from the same inputs; `meta` carries every resolved name/number a consumer needs
(object body/site names, camera poses, joint/actuator ids, which floor texture
was actually used).

Layout of the compiled model (verified 2026-09-14 with mujoco 3.10 / dm_control):
    qpos[0:6]    UR7e arm joints (UR order)            ctrl[0:6]  arm position actuators
    qpos[6:14]   2F-85 joints (driver first)           ctrl[6]    fingers_actuator (0..255)
    qpos[14:]    one 7-vector (pos, quat wxyz) per object free joint, config order
The UR7e MJCF is the ROOT of the tree (not attached into an empty arena), so the
arm's element names stay unprefixed (`attachment_site`, `wrist_3_link`,
`shoulder_pan`); the gripper is `robotiq_2f85/...`, and each object is attached
under its own namespace: body `<name>/<name>`, sites `<name>/<name>_center` etc.
The object's free joint lives on the attachment frame body `<name>/`.

`attach_hand_to_arm` is re-implemented here (same 12 lines as the menagerie FAQ /
`gello.robots.sim_robot`) because importing `gello.robots.sim_robot` pulls in
`mujoco.viewer`, which the headless `--no-viewer` path must never import.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from dm_control import mjcf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

UR_JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
UR_ACTUATOR_NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def resolve_path(p: str) -> str:
    """Repo-relative -> absolute (absolute paths pass through)."""
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SceneConfig:
    """Typed view of `configs/*.yaml`. `raw` keeps the full dict for `sim_meta`."""

    name: str
    robot: Dict[str, Any]
    physics: Dict[str, Any]
    floor: Dict[str, Any]
    cameras: Dict[str, Any]
    objects: List[Dict[str, Any]]
    layout: Dict[str, Any]
    task: Dict[str, Any]
    control: Dict[str, Any]
    leader: Dict[str, Any]
    gripper: Dict[str, Any]
    init_from_leader: bool
    raw: Dict[str, Any]
    path: Optional[str] = None
    render: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], path: Optional[str] = None) -> "SceneConfig":
        return cls(
            name=str(d.get("name", "sim_collect")),
            robot=dict(d["robot"]),
            physics=dict(d.get("physics", {})),
            floor=dict(d.get("floor", {})),
            cameras=dict(d.get("cameras", {})),
            objects=[dict(o) for o in d.get("objects", [])],
            layout=dict(d.get("layout", {})),
            task=dict(d.get("task", {})),
            control=dict(d.get("control", {})),
            leader=dict(d.get("leader", {})),
            gripper=dict(d.get("gripper", {})),
            init_from_leader=bool(d.get("init_from_leader", True)),
            raw=d,
            path=path,
            render=dict(d.get("render", {})),
        )

    @classmethod
    def load(cls, path: str) -> "SceneConfig":
        path = resolve_path(path)
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d, path=os.path.abspath(path))

    @property
    def home_joints(self) -> np.ndarray:
        return np.asarray(self.robot["home_joints"], dtype=float).reshape(6)

    @property
    def timestep(self) -> float:
        return float(self.physics.get("timestep", 0.002))

    def object_names(self) -> List[str]:
        return [o["name"] for o in self.objects]

    def object_spec(self, name: str) -> Dict[str, Any]:
        for o in self.objects:
            if o["name"] == name:
                return o
        raise KeyError(name)


# --------------------------------------------------------------------------- #
# Geometry helpers                                                              #
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def lookat_xyaxes(pos: Sequence[float], target: Sequence[float],
                  up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo `xyaxes` (6-vector) for a camera at `pos` looking at `target`.

    MuJoCo cameras look along their -z axis with +y up and +x to the image right,
    so z_cam = unit(pos - target), x_cam = unit(up x z_cam), y_cam = z_cam x x_cam.
    """
    pos = np.asarray(pos, dtype=float)
    z = _unit(pos - np.asarray(target, dtype=float))
    x = np.cross(_unit(up), z)
    if np.linalg.norm(x) < 1e-9:  # up parallel to the view axis: pick any perpendicular
        x = np.cross([1.0, 0.0, 0.0], z)
        if np.linalg.norm(x) < 1e-9:
            x = np.cross([0.0, 1.0, 0.0], z)
    x = _unit(x)
    y = np.cross(z, x)
    return np.concatenate([x, y])


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float)


def quat_wxyz_to_mat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = _unit(np.asarray(q, dtype=float))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def wrist_camera_pose(cam2: Dict[str, Any], site_pos: Sequence[float],
                      site_quat_wxyz: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """(pos, xyaxes) of the wrist camera in its parent-link frame.

    `cam2` gives the mount in the TOOL (attachment_site) frame: `radial_dir_tool`
    (unit radial direction), `radial_m`, `axial_m` (along tool +Z) and
    `pitch_deg` (rotate the tool-+Z view direction toward the tool axis). The
    camera "up" is the outward radial direction, so the fingertips (on the axis
    side) land at the bottom of the frame. `site_pos/quat` place the tool frame
    in the parent link.
    """
    r_hat = _unit(cam2.get("radial_dir_tool", [0.0, 1.0, 0.0]))
    radial = float(cam2.get("radial_m", 0.05))
    axial = float(cam2.get("axial_m", 0.0))
    pitch = math.radians(float(cam2.get("pitch_deg", 15.0)))
    z_tool = np.array([0.0, 0.0, 1.0])
    p_tool = r_hat * radial + z_tool * axial
    d_tool = math.cos(pitch) * z_tool - math.sin(pitch) * r_hat
    xy_tool = lookat_xyaxes(p_tool, p_tool + d_tool, up=r_hat)
    R = quat_wxyz_to_mat(site_quat_wxyz)
    p_link = np.asarray(site_pos, dtype=float) + R @ p_tool
    xy_link = np.concatenate([R @ xy_tool[:3], R @ xy_tool[3:]])
    return p_link, xy_link


# --------------------------------------------------------------------------- #
# Layout sampling                                                               #
# --------------------------------------------------------------------------- #
def sample_layout(cfg: SceneConfig, seed: Optional[int] = None, attempt: int = 0) -> Dict[str, Dict[str, Any]]:
    """Seeded object layout: {name: {"pos": [x, y, z], "yaw": rad}}.

    `attempt` > 0 draws a different (still deterministic) layout for the same
    seed — reset_scene uses it to re-sample when a layout collides with the arm.

    Rejection-samples each object's xy inside `random_xy_radius` of its nominal
    position so footprints (config `radius_m`) do not overlap each other (plus
    `min_gap_m`) or the robot base keep-out disk. z = drop_height_m. Falls back
    to the nominal position after `max_tries` (never fails; logged in the dict).
    """
    lay = cfg.layout
    seed = int(lay.get("seed", 0)) if seed is None else int(seed)
    rng = np.random.default_rng(seed if attempt == 0 else [seed, int(attempt)])
    drop = float(lay.get("drop_height_m", 0.02))
    gap = float(lay.get("min_gap_m", 0.02))
    base_r = float(lay.get("base_keepout_radius_m", 0.22))
    max_tries = int(lay.get("max_tries", 200))
    items = lay.get("items", {})
    placed: List[Tuple[np.ndarray, float]] = []
    out: Dict[str, Dict[str, Any]] = {"_seed": seed, "_attempt": int(attempt)}  # type: ignore[dict-item]
    for spec in cfg.objects:
        name = spec["name"]
        it = items.get(name, {})
        nominal = np.asarray(it.get("nominal_pos", [-0.5, 0.0, 0.0]), dtype=float)[:2]
        rad = float(it.get("random_xy_radius", 0.0))
        r_obj = float(spec.get("radius_m", 0.05))
        yaw = float(rng.uniform(-math.pi, math.pi)) if it.get("random_yaw", False) else 0.0
        chosen = None
        for _ in range(max_tries):
            if rad > 0:
                rho = rad * math.sqrt(rng.uniform())
                ang = rng.uniform(0, 2 * math.pi)
                xy = nominal + rho * np.array([math.cos(ang), math.sin(ang)])
            else:
                xy = nominal.copy()
            if np.linalg.norm(xy) < base_r + r_obj:
                continue
            if all(np.linalg.norm(xy - p) >= r_obj + r_other + gap for p, r_other in placed):
                chosen = xy
                break
        fallback = chosen is None
        if fallback:
            chosen = nominal.copy()
        placed.append((chosen, r_obj))
        out[name] = {"pos": [float(chosen[0]), float(chosen[1]), drop], "yaw": yaw,
                     "fallback": bool(fallback)}
    return out


def nominal_layout(cfg: SceneConfig) -> Dict[str, Dict[str, Any]]:
    """Layout with every object at its nominal position, yaw 0 (used at build)."""
    drop = float(cfg.layout.get("drop_height_m", 0.02))
    items = cfg.layout.get("items", {})
    out: Dict[str, Dict[str, Any]] = {"_seed": -1}  # type: ignore[dict-item]
    for spec in cfg.objects:
        nom = list(items.get(spec["name"], {}).get("nominal_pos", [-0.5, 0.0, 0.0]))
        out[spec["name"]] = {"pos": [float(nom[0]), float(nom[1]), drop], "yaw": 0.0, "fallback": False}
    return out


# --------------------------------------------------------------------------- #
# MJCF pieces                                                                   #
# --------------------------------------------------------------------------- #
def attach_hand_to_arm(arm: mjcf.RootElement, hand: mjcf.RootElement) -> None:
    """Attach `hand` at the arm's `attachment_site`, extending the `home` keyframe
    (mirror of gello.robots.sim_robot.attach_hand_to_arm / the menagerie FAQ)."""
    physics = mjcf.Physics.from_mjcf_model(hand)
    site = arm.find("site", "attachment_site")
    if site is None:
        raise ValueError("arm model has no site named 'attachment_site'")
    arm_key = arm.find("key", "home")
    if arm_key is not None:
        hand_key = hand.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])
    site.attach(hand)


def placeholder_object_model(name: str, spec: Dict[str, Any]) -> Tuple[mjcf.RootElement, Dict[str, Any]]:
    """Procedural stand-in for a missing `assets/objects/<name>/<name>.xml`.

    Follows the §5.2 interface: one top-level body `<name>` with its origin at the
    object's BOTTOM, a `<name>_center` site, and for open containers the
    `<name>_opening` (rim, size = inner radius) and `<name>_floor` sites.
    Returns (model, size_dict).
    """
    shape = spec.get("shape", "sphere")
    rgba = list(spec.get("rgba", [0.6, 0.6, 0.6, 1.0]))
    mass = float(spec.get("mass", 0.1))
    m = mjcf.RootElement(model=name)
    body = m.worldbody.add("body", name=name)
    body.add("site", name=f"{name}_center", pos=[0, 0, 0], size=[0.004], group=4, rgba=[0, 1, 0, 0.5])
    size: Dict[str, Any] = {"shape": shape, "mass": mass}
    if shape == "capsule":
        r = float(spec["radius"]); hl = float(spec["half_length"])
        # lying on the floor along the body x axis; bottom of the capsule at z=0
        body.add("geom", name=f"{name}_geom", type="capsule", size=[r, hl], pos=[0, 0, r],
                 quat=[math.sqrt(0.5), 0, math.sqrt(0.5), 0], mass=mass, rgba=rgba,
                 condim=4, friction=[1.0, 0.02, 0.0005])
        size.update(radius=r, half_length=hl, height=2 * r, length=2 * (hl + r))
    elif shape == "sphere":
        r = float(spec["radius"])
        body.add("geom", name=f"{name}_geom", type="sphere", size=[r], pos=[0, 0, r], mass=mass,
                 rgba=rgba, condim=4, friction=[1.0, 0.02, 0.0005])
        size.update(radius=r, height=2 * r)
    elif shape == "open_cylinder":
        ri = float(spec["inner_radius"]); h = float(spec["height"])
        wall = float(spec.get("wall", 0.006)); bottom = float(spec.get("bottom", 0.006))
        n = int(spec.get("n_wall", 12))
        ro = ri + wall
        m_bottom = 0.4 * mass
        m_wall = 0.6 * mass / n
        body.add("geom", name=f"{name}_bottom", type="cylinder", size=[ro, bottom / 2], pos=[0, 0, bottom / 2],
                 mass=m_bottom, rgba=rgba, friction=[1.0, 0.005, 0.0001])
        half_h = (h - bottom) / 2
        half_chord = ro * math.tan(math.pi / n) * 1.02
        rc = ri + wall / 2
        for k in range(n):
            th = 2 * math.pi * k / n
            body.add("geom", name=f"{name}_wall{k}", type="box", size=[wall / 2, half_chord, half_h],
                     pos=[rc * math.cos(th), rc * math.sin(th), bottom + half_h],
                     quat=yaw_quat_wxyz(th).tolist(), mass=m_wall, rgba=rgba,
                     friction=[1.0, 0.005, 0.0001])
        body.add("site", name=f"{name}_opening", pos=[0, 0, h], size=[ri], group=4, rgba=[0, 0, 1, 0.15])
        body.add("site", name=f"{name}_floor", pos=[0, 0, bottom], size=[0.004], group=4, rgba=[0, 0, 1, 0.5])
        size.update(inner_radius=ri, outer_radius=ro, height=h, rim_z=h, floor_z=bottom, wall=wall)
    else:
        raise ValueError(f"unknown placeholder shape {shape!r} for object {name!r}")
    return m, size


def _object_asset_path(cfg: SceneConfig, spec: Dict[str, Any]) -> Optional[str]:
    p = spec.get("xml")
    if p is None:
        p = os.path.join("sim_collect", "assets", "objects", spec["name"], f"{spec['name']}.xml")
    p = resolve_path(p)
    return p if os.path.isfile(p) else None


def _validate_object_model(name: str, model: mjcf.RootElement, path: str) -> None:
    bodies = model.worldbody.get_children("body")
    if len(bodies) != 1 or bodies[0].name != name:
        raise ValueError(f"{path}: worldbody must hold exactly one top-level <body name={name!r}> "
                         f"(found {[b.name for b in bodies]})")
    if model.find("site", f"{name}_center") is None:
        raise ValueError(f"{path}: missing <site name={name + '_center'!r}>")
    if bodies[0].find_all("joint"):
        raise ValueError(f"{path}: object body must not contain joints (scene.py adds the freejoint)")


def _site_size_info(model: mjcf.RootElement, name: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    op = model.find("site", f"{name}_opening")
    fl = model.find("site", f"{name}_floor")
    if op is not None:
        info["inner_radius"] = float(np.atleast_1d(op.size)[0]) if op.size is not None else None
        info["rim_z"] = float(op.pos[2]) if op.pos is not None else None
    if fl is not None:
        info["floor_z"] = float(fl.pos[2]) if fl.pos is not None else 0.0
    return info


# --------------------------------------------------------------------------- #
# Build                                                                         #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class BuiltScene:
    xml: str
    assets: Dict[str, bytes]
    meta: Dict[str, Any]

    def load_model(self):
        import mujoco
        return mujoco.MjModel.from_xml_string(self.xml, self.assets)


def build_scene(cfg: SceneConfig, layout: Optional[Dict[str, Dict[str, Any]]] = None) -> BuiltScene:
    """Assemble the full MJCF. Deterministic given (cfg, layout).

    `layout` is the dict produced by `sample_layout` / `nominal_layout` (also the
    one `reset_scene` reports); it only sets the objects' initial free-joint
    pose. Returns xml + assets (compile with `BuiltScene.load_model()`) + meta.
    """
    if layout is None:
        layout = nominal_layout(cfg)
    meta: Dict[str, Any] = {"config_name": cfg.name, "layout": layout, "objects": {}, "cameras": {},
                            "floor": {}, "names": {}}

    root = mjcf.from_path(resolve_path(cfg.robot["xml"]))
    root.model = f"sim_collect_{cfg.name}"

    # ---- physics options (root wins over the attached models') -------------
    ph = cfg.physics
    root.option.timestep = float(ph.get("timestep", 0.002))
    root.option.integrator = str(ph.get("integrator", "implicitfast"))
    root.option.cone = str(ph.get("cone", "elliptic"))
    root.option.impratio = float(ph.get("impratio", 10))
    root.option.noslip_iterations = int(ph.get("noslip_iterations", 0))
    # Gravity compensation on the ARM LINKS ONLY (default 1.0): the menagerie
    # position actuators are plain PD (no integral action), so an uncompensated
    # arm sags ~0.011 rad per (re)seed at the home pose (0.022 measured after
    # teleport + seed). A real UR servo holds its setpoint exactly; gravcomp
    # makes the sim arm do the same. The gripper bodies stay uncompensated so the
    # flange F/T sensor still reads the tool weight like the real one.
    gravcomp = float(ph.get("arm_gravcomp", 1.0))
    if gravcomp > 0.0:
        for bn in ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"):
            body = root.find("body", bn)
            if body is not None:
                body.gravcomp = gravcomp

    # ---- offscreen buffer for the capture process --------------------------
    g = root.visual.__getattr__("global")
    g.offwidth = 1280
    g.offheight = 720
    rd = cfg.render
    # Software GL: the viewer's default 4096 shadow map starved the capture
    # workers (140 ms/frame measured); 2048 / no offscreen multisampling.
    root.visual.quality.shadowsize = int(rd.get("viewer_shadowsize", 2048))
    root.visual.quality.offsamples = int(rd.get("offsamples", 0))
    hd = float(rd.get("headlight_diffuse", 0.20))
    ha = float(rd.get("headlight_ambient", 0.12))
    root.visual.headlight.diffuse = [hd, hd, hd]
    root.visual.headlight.ambient = [ha, ha, ha]
    root.visual.headlight.specular = [0.05, 0.05, 0.05]

    # ---- gripper + flange F/T site ----------------------------------------
    hand = mjcf.from_path(resolve_path(cfg.robot["gripper_xml"]))
    ft_name = str(cfg.robot.get("ft_site", "ft_site"))
    base_mount = hand.find("body", "base_mount")
    # base_mount sits 7 mm above the attachment frame; put the site AT the flange.
    bm_pos = np.asarray(base_mount.pos if base_mount.pos is not None else [0, 0, 0], dtype=float)
    base_mount.add("site", name=ft_name, pos=(-bm_pos).tolist(), size=[0.003], group=4, rgba=[1, 0, 0, 0.4])
    attach_hand_to_arm(root, hand)
    ft_full = f"{hand.model}/{ft_name}"
    root.sensor.add("force", name="ft_force", site=root.find("site", ft_full))
    root.sensor.add("torque", name="ft_torque", site=root.find("site", ft_full))
    meta["names"].update(
        arm_joints=list(UR_JOINT_NAMES), arm_actuators=list(UR_ACTUATOR_NAMES),
        gripper_driver_joint=str(cfg.robot.get("gripper_driver_joint", "robotiq_2f85/right_driver_joint")),
        gripper_driver_range=[float(v) for v in cfg.robot.get("gripper_driver_range", [0.0, 0.8])],
        gripper_actuator=str(cfg.robot.get("gripper_actuator", "robotiq_2f85/fingers_actuator")),
        ft_site=ft_full, force_sensor="ft_force", torque_sensor="ft_torque",
        attachment_site="attachment_site",
    )

    # ---- floor ----------------------------------------------------------------
    fl = cfg.floor
    tex_path = fl.get("texture")
    tex_abs = resolve_path(tex_path) if tex_path else None
    if tex_abs and os.path.isfile(tex_abs):
        root.asset.add("texture", name="floor_tex", type="2d", file=tex_abs)
        meta["floor"]["texture_used"] = tex_abs
    else:
        root.asset.add("texture", name="floor_tex", type="2d", builtin="checker", mark="edge",
                       rgb1=[0.62, 0.50, 0.36], rgb2=[0.52, 0.41, 0.29], markrgb=[0.7, 0.6, 0.5],
                       width=256, height=256)
        meta["floor"]["texture_used"] = "builtin:checker"
    root.asset.add("material", name="floor_mat", texture="floor_tex", texuniform=True,
                   texrepeat=[float(v) for v in fl.get("texrepeat", [20, 20])],
                   reflectance=float(cfg.render.get("floor_reflectance", 0.0)),
                   rgba=[float(v) for v in cfg.render.get("floor_tint", [0.8, 0.8, 0.8, 1.0])])
    half = float(fl.get("size_m", 5.0))
    root.worldbody.add("geom", name="floor", type="plane", size=[half, half, 0.1], material="floor_mat",
                       friction=[float(v) for v in fl.get("friction", [1.0, 0.005, 0.0001])],
                       condim=3)
    root.asset.add("texture", type="skybox", builtin="gradient", rgb1=[0.55, 0.65, 0.8], rgb2=[0.9, 0.9, 0.95],
                   width=256, height=1536)
    sd = float(cfg.render.get("sun_diffuse", 0.22))
    root.worldbody.add("light", name="sun", pos=[0.5, -0.5, 2.5], dir=[-0.3, 0.3, -1.0], directional=True,
                       diffuse=[sd, sd, sd], specular=[0.05, 0.05, 0.05], castshadow=True)
    fd = float(cfg.render.get("fill_diffuse", 0.12))
    root.worldbody.add("light", name="fill", pos=[-1.5, 0.5, 1.5], dir=[0.7, -0.3, -0.8], directional=True,
                       diffuse=[fd, fd, fd], castshadow=False)

    # ---- cameras --------------------------------------------------------------
    cams = cfg.cameras
    fovy_c = float(cams.get("color_fovy_deg", 42.0))
    fovy_d = float(cams.get("depth_fovy_deg", 58.7))
    d_off = np.asarray(cams.get("depth_offset_m", [0.015, 0.0, 0.0]), dtype=float)

    c1 = cams.get("cam1", {})
    p1 = np.asarray(c1.get("pos", [-0.70, 0.0, 0.571]), dtype=float)
    xy1 = lookat_xyaxes(p1, c1.get("lookat", [-0.45, 0.0, 0.0]), c1.get("up", [0, 0, 1]))
    R1 = np.stack([xy1[:3], xy1[3:], np.cross(xy1[:3], xy1[3:])], axis=1)
    p1d = p1 + R1 @ d_off
    root.worldbody.add("camera", name="cam1", pos=p1.tolist(), xyaxes=xy1.tolist(), fovy=fovy_c)
    root.worldbody.add("camera", name="cam1_depth", pos=p1d.tolist(), xyaxes=xy1.tolist(), fovy=fovy_d)
    meta["cameras"]["cam1"] = {"parent": "world", "pos": p1.tolist(), "xyaxes": xy1.tolist(), "fovy": fovy_c}
    meta["cameras"]["cam1_depth"] = {"parent": "world", "pos": p1d.tolist(), "xyaxes": xy1.tolist(), "fovy": fovy_d}

    c2 = cams.get("cam2", {})
    parent_name = str(c2.get("parent_body", "wrist_3_link"))
    parent = root.find("body", parent_name)
    site = root.find("site", str(c2.get("mount_frame", "attachment_site")))
    s_pos = site.pos if site.pos is not None else [0, 0, 0]
    s_quat = site.quat if site.quat is not None else [1, 0, 0, 0]
    p2, xy2 = wrist_camera_pose(c2, s_pos, s_quat)
    R2 = np.stack([xy2[:3], xy2[3:], np.cross(xy2[:3], xy2[3:])], axis=1)
    p2d = p2 + R2 @ d_off
    parent.add("camera", name="cam2", pos=p2.tolist(), xyaxes=xy2.tolist(), fovy=fovy_c)
    parent.add("camera", name="cam2_depth", pos=p2d.tolist(), xyaxes=xy2.tolist(), fovy=fovy_d)
    meta["cameras"]["cam2"] = {"parent": parent_name, "pos": p2.tolist(), "xyaxes": xy2.tolist(), "fovy": fovy_c}
    meta["cameras"]["cam2_depth"] = {"parent": parent_name, "pos": p2d.tolist(), "xyaxes": xy2.tolist(), "fovy": fovy_d}
    meta["cameras"]["depth_offset_m"] = d_off.tolist()

    # ---- objects (after the arm+hand so their qpos land after the arm's) ------
    key = root.find("key", "home")
    key_qpos = list(np.asarray(key.qpos, dtype=float))
    key_qpos[:6] = list(cfg.home_joints)
    key_ctrl = list(np.asarray(key.ctrl, dtype=float))
    key_ctrl[:6] = list(cfg.home_joints)
    for spec in cfg.objects:
        name = spec["name"]
        path = _object_asset_path(cfg, spec)
        if path is not None:
            om = mjcf.from_path(path)
            om.model = name
            _validate_object_model(name, om, path)
            size = {"source": path}
            size.update(_site_size_info(om, name))
        else:
            if "placeholder" not in spec:
                raise FileNotFoundError(f"object {name!r}: no asset xml and no placeholder spec")
            om, size = placeholder_object_model(name, spec["placeholder"])
            size["source"] = "placeholder"
        frame = root.worldbody.attach(om)
        frame.add("freejoint", name=f"{name}_free")
        lay = layout.get(name, {"pos": [-0.5, 0.0, 0.02], "yaw": 0.0})
        pos = [float(v) for v in lay["pos"]]
        quat = yaw_quat_wxyz(float(lay.get("yaw", 0.0)))
        frame.pos = pos
        frame.quat = quat.tolist()
        key_qpos += pos + quat.tolist()
        meta["objects"][name] = {
            "kind": spec.get("kind", "object"),
            "frame_body": f"{name}/",
            "body": f"{name}/{name}",
            "center_site": f"{name}/{name}_center",
            "opening_site": f"{name}/{name}_opening" if om.find("site", f"{name}_opening") is not None else None,
            "floor_site": f"{name}/{name}_floor" if om.find("site", f"{name}_floor") is not None else None,
            "radius_m": float(spec.get("radius_m", 0.05)),
            "size": size,
        }
    key.qpos = key_qpos
    key.ctrl = key_ctrl
    meta["home_joints"] = list(map(float, cfg.home_joints))
    meta["timestep"] = float(root.option.timestep)

    xml = root.to_xml_string()
    assets = dict(root.get_assets())
    return BuiltScene(xml=xml, assets=assets, meta=meta)
