#!/usr/bin/env python3
"""Vendor upstream meshes into ``sim_collect/assets/objects/<name>/<name>.xml``.

**The vendored files under ``sim_collect/assets/`` are canonical.**  This module
exists only so that tree can be *regenerated* from upstream (see
``tools/fetch_assets.sh``); nothing in ``sim_collect/`` imports it at runtime and
the test suite never runs it.

What it produces, per DESIGN.md §5.2 — one directory per object holding a
complete, standalone ``<mujoco model="<name>">`` file plus its meshes/textures:

* ``<compiler meshdir="." texturedir="."/>`` so the file is relocatable,
* exactly **one** top-level ``<body name="<name>">`` in ``<worldbody>`` with **no**
  freejoint and no anonymous wrapper body (``scene.py`` attaches it with
  ``mjcf`` and puts the freejoint on the attachment frame),
* visual geoms ``group="2" contype="0" conaffinity="0"``, collision geoms
  ``group="3"``, all with ``condim="6" friction="1 0.02 0.004"`` (matches the
  floor; the rolling-friction term is what stops round fruit rolling forever),
* an explicit ``<inertial>`` — mass is a hand-set realistic value, while the COM,
  the principal axes and the inertia *shape* are measured from the compiled
  collision geometry (upstream's ``density="50..100"`` gives 3 g lemons),
* the object shifted so its origin sits at the **bottom centre** of its bounding
  box: ``pos="0 0 0"`` rests on the floor and the XY origin is the centroid
  (raw YCB meshes are up to 3.3 cm off-centre in XY — that is corrected here),
* ``<site name="<name>_center">`` at the origin, and for containers
  ``<site name="<name>_opening">`` / ``<site name="<name>_floor">`` whose
  positions and radius are **ray-cast out of the compiled cavity**, not guessed.

Upstream-specific fixes applied on the way through:

* LIBERO / robosuite object XMLs wrap the real body in an anonymous ``<body>``
  (which blocks ``<freejoint/>``); it is removed and the inner body renamed.
* Every asset name is namespaced ``<name>_*`` so objects can coexist.
* ``solref="0.001 1"`` (upstream) is dropped: a 1 ms contact time constant is
  unstable at the 2 ms timestep this project uses.  MuJoCo defaults are used.
* LIBERO HOPE-object meshes are in centimetres (``scale 0.01``); the scanned
  objects and robosuite meshes are in metres.  YCB OBJs are in metres.
"""

from __future__ import annotations

import math
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Contact model shared by the floor and every object.  condim 6 = sliding +
# torsional + rolling friction; without the rolling term spheres never stop.
CONDIM = "6"
FRICTION = "1 0.02 0.004"


@dataclass
class Geom:
    """One geom, in the *upstream* body frame.  ``kind`` picks group 2 vs 3."""

    kind: str  # "visual" | "collision"
    attrs: dict = field(default_factory=dict)

    def shifted(self, delta: np.ndarray) -> dict:
        a = dict(self.attrs)
        if "fromto" in a:  # capsules: shift both endpoints, never add a pos
            ft = np.fromstring(a["fromto"], sep=" ")
            a["fromto"] = _v(np.concatenate([ft[:3] + delta, ft[3:] + delta]))
        else:
            pos = np.fromstring(a.get("pos", "0 0 0"), sep=" ")
            a["pos"] = _v(pos + delta)
        if self.kind == "visual":
            a.update(group="2", contype="0", conaffinity="0")
        else:
            a.update(group="3", condim=CONDIM, friction=FRICTION)
        return a


def _v(x: Sequence[float]) -> str:
    return " ".join(f"{v:.6g}" for v in x)


def _geom_xml(attrs: dict) -> str:
    order = ["name", "type", "mesh", "size", "fromto", "pos", "quat", "euler",
             "material", "rgba", "group", "contype", "conaffinity", "condim",
             "friction", "mass", "density"]
    keys = [k for k in order if k in attrs] + [k for k in attrs if k not in order]
    return "<geom " + " ".join(f'{k}="{attrs[k]}"' for k in keys) + "/>"


def _write_model(path: str, name: str, assets: Iterable[str], geoms: Iterable[dict],
                 inertial: str = "", sites: Iterable[str] = ()) -> None:
    body = "\n".join(f"      {s}" for s in sites)
    parts = [
        f'<mujoco model="{name}">',
        '  <compiler angle="radian" meshdir="." texturedir="."/>',
        "  <asset>",
        *(f"    {a}" for a in assets),
        "  </asset>",
        "  <worldbody>",
        f'    <body name="{name}" pos="0 0 0">',
    ]
    if inertial:
        parts.append(f"      {inertial}")
    if body:
        parts.append(body)
    parts += [f"      {_geom_xml(g)}" for g in geoms]
    parts += ["    </body>", "  </worldbody>", "</mujoco>", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(parts))


# --------------------------------------------------------------------------- #
#  measurement helpers (everything is measured, nothing is guessed)
# --------------------------------------------------------------------------- #
def _compile(path: str):
    import mujoco

    return mujoco.MjModel.from_xml_path(path)


def _aabb(model, groups=(2, 3)) -> tuple[np.ndarray, np.ndarray]:
    """Tight world AABB of the geoms in ``groups`` (body at the origin).

    Mesh geoms are measured from their vertices, not from ``geom_aabb``: the
    compiler re-orients every mesh onto its principal axes of inertia, so the
    geom frame is rotated and the *rotated* local box over-estimates the extent
    by up to 50 % (measured on the YCB plum).  Getting this wrong makes the
    object float above the floor.
    """
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        if model.geom_group[gid] not in groups:
            continue
        R = data.geom_xmat[gid].reshape(3, 3)
        p = data.geom_xpos[gid]
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            w = np.asarray(model.mesh_vert[a:a + n], dtype=float) @ R.T + p
            lo = np.minimum(lo, w.min(axis=0))
            hi = np.maximum(hi, w.max(axis=0))
            continue
        c, h = model.geom_aabb[gid][:3], model.geom_aabb[gid][3:]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    w = p + R @ (c + h * np.array([sx, sy, sz]))
                    lo = np.minimum(lo, w)
                    hi = np.maximum(hi, w)
    return lo, hi


def _inertia_from_collision(path: str, name: str, assets, geoms, mass: float) -> str:
    """Compile a density-only probe and scale its inertia to the target mass."""
    import mujoco

    probe = os.path.join(os.path.dirname(path), "_probe_inertia.xml")
    coll = [dict(g, density="1000") for g in geoms if g.get("group") == "3"]
    _write_model(probe, name, assets, coll)
    m = _compile(probe)
    os.remove(probe)
    k = mass / float(m.body_mass[1])
    diag = np.asarray(m.body_inertia[1]) * k
    ipos, iquat = np.asarray(m.body_ipos[1]), np.asarray(m.body_iquat[1])
    return (f'<inertial pos="{_v(ipos)}" quat="{_v(iquat)}" mass="{mass:g}" '
            f'diaginertia="{_v(diag)}"/>')


def measure_cavity(path: str, rim_z: float, n_ang: int = 32) -> tuple[float, float]:
    """Ray-cast the inside of an open container.

    Returns ``(inner_floor_z, opening_radius)``: the height of the inner floor on
    the axis, and the inner radius at the highest level below ``rim_z`` where a
    wall completely surrounds the axis.  Raises if the model turns out to be
    solid (a convex hull masquerading as a bowl) — that is the whole point of
    measuring instead of trusting the upstream mesh.
    """
    import mujoco

    m = _compile(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # collision geoms only (mj_ray also skips fully transparent geoms, which is
    # why every collision geom is emitted at alpha 0.3 rather than 0)
    grp = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    gid = np.zeros(1, dtype=np.int32)

    def ray(pnt, vec) -> float:
        return mujoco.mj_ray(m, d, np.asarray(pnt, float), np.asarray(vec, float),
                             grp, 1, -1, gid)

    top = rim_z + 0.05
    down = ray([0.0, 0.0, top], [0.0, 0.0, -1.0])
    if down < 0:
        raise RuntimeError("no floor under the container axis")
    inner_floor = top - down
    if inner_floor > rim_z - 0.005:
        raise RuntimeError(f"container is solid: inner floor {inner_floor:.4f} "
                           f"reaches the rim {rim_z:.4f}")

    dirs = [(math.cos(2 * math.pi * i / n_ang), math.sin(2 * math.pi * i / n_ang))
            for i in range(n_ang)]
    z = rim_z - 0.002
    while z > inner_floor + 0.005:
        hits = [ray([0.0, 0.0, z], [cx, cy, 0.0]) for cx, cy in dirs]
        if all(h >= 0 for h in hits):
            return inner_floor, min(hits)
        z -= 0.002
    raise RuntimeError("no closed wall found between the inner floor and the rim")


# --------------------------------------------------------------------------- #
#  the one entry point every converter funnels into
# --------------------------------------------------------------------------- #
def emit(name: str, out_dir: str, assets: Sequence[str], geoms: Sequence[Geom],
         mass: float, container: bool = False) -> dict:
    """Write ``<out_dir>/<name>.xml`` (meshes must already be in ``out_dir``)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.xml")

    # pass 1: as authored, to find the bounding box
    raw = [g.shifted(np.zeros(3)) for g in geoms]
    for i, g in enumerate(raw):
        g.setdefault("name", f"{name}_{'vis' if g['group'] == '2' else 'col'}{i}")
    _write_model(path, name, assets, raw)
    m0 = _compile(path)
    lo, hi = _aabb(m0)                    # silhouette -> XY centre
    lo_c, _ = _aabb(m0, groups=(3,))      # collision shell -> resting height

    # pass 2: origin -> XY centre of the silhouette, z of the collision shell.
    # z must come from the collision shell, not the visual mesh: the shell is
    # what actually touches the floor, and LIBERO's box decompositions stop a
    # few mm short of their visual mesh (bowl 1.6 mm, basket 4.2 mm).  Using the
    # visual bottom instead makes the object settle *below* z=0.
    delta = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo_c[2]])
    shifted = [g.shifted(delta) for g in geoms]
    for i, g in enumerate(shifted):
        g.setdefault("name", f"{name}_{'vis' if g['group'] == '2' else 'col'}{i}")
    _write_model(path, name, assets, shifted)
    lo2, hi2 = _aabb(_compile(path))
    bbox = hi2 - lo2

    sites = [f'<site name="{name}_center" pos="0 0 0" size="0.005" rgba="0 0 0 0"/>']
    info = {"name": name, "bbox_cm": [round(float(v) * 100, 2) for v in bbox],
            "mass": mass,
            "visual_below_floor_mm": round(-float(lo2[2]) * 1000, 2)}
    if container:
        # the rim an object must clear is the top of the *collision* shell, not
        # of the visual mesh (upstream visuals overhang their box decomposition)
        rim_z = float(_aabb(_compile(path), groups=(3,))[1][2])
        floor_z, radius = measure_cavity(path, rim_z)
        sites += [
            f'<site name="{name}_opening" pos="0 0 {rim_z:.4f}" '
            f'size="{radius:.4f}" rgba="0 0 0 0"/>',
            f'<site name="{name}_floor" pos="0 0 {floor_z:.4f}" '
            f'size="0.005" rgba="0 0 0 0"/>',
        ]
        info.update(rim_z=round(rim_z, 4), inner_floor_z=round(floor_z, 4),
                    opening_radius=round(radius, 4))

    inertial = _inertia_from_collision(path, name, assets, shifted, mass)
    _write_model(path, name, assets, shifted, inertial, sites)
    _compile(path)  # final proof that what we shipped loads
    return info


# --------------------------------------------------------------------------- #
#  upstream readers
# --------------------------------------------------------------------------- #
def _copy_asset(src: str, dst: str, tex_px: int = 1024) -> None:
    """Copy a mesh/texture, downscaling textures (LIBERO ships 4096^2 PNGs;
    1024^2 is indistinguishable at the 128x128 the policy ever sees)."""
    if src.lower().endswith((".png", ".jpg", ".jpeg")):
        from PIL import Image

        im = Image.open(src).convert("RGB")
        if max(im.size) > tex_px:
            k = tex_px / max(im.size)
            im = im.resize((int(im.size[0] * k), int(im.size[1] * k)), Image.LANCZOS)
        im.save(dst, optimize=True)
    else:
        shutil.copy2(src, dst)


def from_mjcf(src_xml: str, out_root: str, name: str, mass: float,
              container: bool = False, scale: str | None = None) -> dict:
    """LIBERO / robosuite object MJCF -> our layout.

    Upstream nests the real body inside an anonymous ``<body>`` and sprinkles
    placement sites; both are dropped.  Assets are copied flat into the object
    directory and renamed ``<name>_<basename>``.
    """
    src_dir = os.path.dirname(os.path.abspath(src_xml))
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    root = ET.parse(src_xml).getroot()
    asset_el, wb = root.find("asset"), root.find("worldbody")

    ren: dict[str, str] = {}
    assets: list[str] = []
    for el in list(asset_el):
        f = el.get("file")
        if f:
            base = f"{name}_{os.path.basename(f)}"
            _copy_asset(os.path.join(src_dir, f), os.path.join(out_dir, base))
            el.set("file", base)
        if el.get("name"):
            ren[el.get("name")] = f"{name}_{el.get('name')}"
            el.set("name", ren[el.get("name")])
        if el.tag == "mesh" and scale:
            el.set("scale", scale)
        for a in ("texture", "material", "mesh"):
            if el.get(a) in ren:
                el.set(a, ren[el.get(a)])
        assets.append(ET.tostring(el, encoding="unicode").strip())

    obj = next(b for b in wb.iter("body") if b.get("name") == "object")
    geoms: list[Geom] = []
    for g in obj.iter("geom"):
        a = {k: v for k, v in g.attrib.items()
             if k not in ("density", "solimp", "solref", "friction", "condim",
                          "group", "contype", "conaffinity", "name", "mass")}
        for k in ("mesh", "material"):
            if a.get(k) in ren:
                a[k] = ren[a[k]]
        visual = g.get("contype") == "0" and g.get("conaffinity") == "0"
        if not visual and g.get("type") == "box":
            a.pop("rgba", None)  # upstream paints collision boxes translucent grey
            a["rgba"] = "0.5 0.5 0.5 0.3"
        geoms.append(Geom("visual" if visual else "collision", a))
    if not any(g.kind == "visual" for g in geoms):
        # single-geom upstream objects (bread, lemon): the mesh is both roles
        base = geoms[0]
        geoms = [Geom("visual", dict(base.attrs)),
                 Geom("collision", {**base.attrs, "rgba": "0.5 0.5 0.5 0.3"})]
        geoms[1].attrs.pop("material", None)
    return emit(name, out_dir, assets, geoms, mass, container)


def from_ycb(model_dir: str, out_root: str, name: str, mass: float,
             tex_px: int = 1024) -> dict:
    """A YCB ``google_16k`` directory -> our layout (metres, z_min ~ 0)."""
    from PIL import Image

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    src_obj = os.path.join(model_dir, "textured.obj")
    with open(src_obj) as fh:  # drop mtllib/usemtl: our <material> owns the look
        body = [ln for ln in fh if not ln.startswith(("mtllib", "usemtl"))]
    with open(os.path.join(out_dir, f"{name}.obj"), "w") as fh:
        fh.writelines(body)
    im = Image.open(os.path.join(model_dir, "texture_map.png")).convert("RGB")
    s = tex_px / max(im.size)
    if s < 1:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
    im.save(os.path.join(out_dir, f"{name}.png"), optimize=True)

    assets = [
        f'<mesh name="{name}_mesh" file="{name}.obj" maxhullvert="64"/>',
        f'<texture name="{name}_tex" type="2d" file="{name}.png"/>',
        f'<material name="{name}_mat" texture="{name}_tex" specular="0.25" shininess="0.3"/>',
    ]
    geoms = [
        Geom("visual", {"type": "mesh", "mesh": f"{name}_mesh", "material": f"{name}_mat"}),
        Geom("collision", {"type": "mesh", "mesh": f"{name}_mesh", "rgba": "0.5 0.5 0.5 0.3"}),
    ]
    return emit(name, out_dir, assets, geoms, mass)


# --------------------------------------------------------------------------- #
#  procedural objects (no upstream mesh is good enough)
# --------------------------------------------------------------------------- #
def make_pot(out_root: str, name: str = "pot", mass: float = 0.55,
             inner_r: float = 0.090, height: float = 0.110, wall: float = 0.006,
             n_staves: int = 24, handles: bool = True) -> dict:
    """A genuinely hollow cooking pot: ``n_staves`` box staves + a base disc.

    A scanned pot mesh is useless here — MuJoCo collides meshes as convex hulls,
    so any single-mesh pot is a solid lump.  The stave half-width
    ``tan(pi/N)*(R+t/2)*1.06`` closes the ring with a 6 % overlap (no seams for a
    fruit to squeeze through) while staying flat enough that the inside wall is
    the intended cylinder to ~0.3 mm.
    """
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    metal, dark = "0.30 0.31 0.34 1", "0.22 0.23 0.26 1"
    base_h = 0.003
    rc = inner_r + wall / 2
    half_w = math.tan(math.pi / n_staves) * rc * 1.06
    wall_h = (height - 2 * base_h) / 2

    geoms: list[Geom] = []

    def pair(attrs: dict, rgba: str) -> None:
        geoms.append(Geom("visual", {**attrs, "rgba": rgba}))
        geoms.append(Geom("collision", {**attrs, "rgba": "0.5 0.5 0.5 0.3"}))

    pair({"type": "cylinder", "size": f"{inner_r + wall:.4f} {base_h:.4f}",
          "pos": f"0 0 {base_h:.4f}"}, dark)
    for i in range(n_staves):
        a = 2 * math.pi * i / n_staves
        pair({"type": "box",
              "size": f"{wall / 2:.4f} {half_w:.4f} {wall_h:.4f}",
              "pos": f"{rc * math.cos(a):.4f} {rc * math.sin(a):.4f} "
                     f"{2 * base_h + wall_h:.4f}",
              "euler": f"0 0 {a:.4f}"}, metal)
    if handles:
        for sx in (1, -1):
            pair({"type": "box", "size": "0.012 0.022 0.005",
                  "pos": f"{sx * (inner_r + wall + 0.012):.4f} 0 "
                         f"{height - 0.018:.4f}"}, dark)
    return emit(name, out_dir, [], geoms, mass, container=True)


def make_carrot(out_root: str, name: str = "carrot", mass: float = 0.08,
                length: float = 0.140, r_top: float = 0.0195,
                r_tip: float = 0.004, n_seg: int = 4) -> dict:
    """A carrot lying on its side: tapered capsule stack + a green leaf tuft.

    Each segment's axis sits at its own radius so every segment touches z=0 —
    a cone resting on a table, not a cylinder floating on its fat end.  The
    leaves are visual-only: they must not add collision clutter around the
    grasp point, and the 2F-85 closes on the body, not the greens.
    """
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    orange, green = "0.92 0.45 0.09 1", "0.20 0.55 0.18 1"
    x0, x1 = -length / 2, length / 2
    geoms: list[Geom] = []

    def rad(x: float) -> float:
        return r_tip + (x - x0) / (x1 - x0) * (r_top - r_tip)

    for i in range(n_seg):
        xa = x0 + (x1 - x0) * i / n_seg
        xb = x0 + (x1 - x0) * (i + 1) / n_seg
        r = (rad(xa) + rad(xb)) / 2
        ft = f"{xa:.4f} 0 {r:.4f} {xb:.4f} 0 {r:.4f}"
        geoms.append(Geom("visual", {"type": "capsule", "fromto": ft,
                                     "size": f"{r:.4f}", "rgba": orange}))
        geoms.append(Geom("collision", {"type": "capsule", "fromto": ft,
                                        "size": f"{r:.4f}", "rgba": "0.5 0.5 0.5 0.3"}))
    for dy, dz in ((0.0, 0.012), (0.013, 0.004), (-0.013, 0.004)):
        geoms.append(Geom("visual", {
            "type": "capsule",
            "fromto": f"{x1:.4f} 0 {r_top:.4f} {x1 + 0.034:.4f} {dy:.4f} "
                      f"{r_top + dz:.4f}",
            "size": "0.0035", "rgba": green}))
    return emit(name, out_dir, [], geoms, mass)
