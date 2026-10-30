"""Asset contract tests (O1).

Every object under ``sim_collect/assets/objects/`` must satisfy DESIGN.md §5.2
*standalone* -- one top-level body, no freejoint, explicit mass, the sites
``scene.py``/``task.py`` look up -- and must still behave when several of them
are attached into one arena the way ``scene.py`` does it (dm_control
``mjcf.from_path`` -> ``worldbody.attach`` -> ``freejoint`` on the attachment
frame; the idiom is ``gello/robots/sim_robot.py:attach_hand_to_arm``).

The last test is the one that matters most: MuJoCo collides meshes as convex
hulls, so a container whose collision geometry is a single mesh is a solid lump
that nothing can be placed *in*.  ``test_containers_are_hollow`` drops a
strawberry into each container and checks it comes to rest inside the advertised
opening cylinder.  If someone swaps in a prettier bowl mesh, this fails.
"""

from __future__ import annotations

import glob
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
OBJECT_DIRS = sorted(
    d for d in glob.glob(os.path.join(ASSETS, "objects", "*")) if os.path.isdir(d)
)
OBJECT_XMLS = [(os.path.basename(d), os.path.join(d, os.path.basename(d) + ".xml"))
               for d in OBJECT_DIRS]
CONTAINERS = ("pot", "bowl", "basket")
FOODS = ("banana", "bread", "carrot", "lemon", "peach", "pear", "plum", "strawberry")

MASS_MIN, MASS_MAX = 0.02, 1.0
GRASP_MAX_MIN_DIM = 0.07  # a 2F-85 opens to 85 mm; 70 mm leaves finger clearance

mujoco = pytest.importorskip("mujoco")


def _names() -> list[str]:
    return [n for n, _ in OBJECT_XMLS]


def test_the_expected_objects_are_vendored():
    assert set(_names()) == set(CONTAINERS) | set(FOODS)


def test_asset_tree_is_small_enough():
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(os.path.join(ASSETS, "objects")) for f in fs)
    total += sum(os.path.getsize(p) for p in glob.glob(os.path.join(ASSETS, "textures", "*")))
    assert total < 30 * 1024 ** 2, f"vendored assets grew to {total / 1024 ** 2:.1f} MB"


@pytest.mark.parametrize("name", ["martin_novak_wood_table", "seamless_wood_planks_floor"])
def test_floor_texture(name):
    """DESIGN 5.2: power-of-two, <= 2048^2, referenced by the scene yaml."""
    path = os.path.join(ASSETS, "textures", name + ".png")
    assert os.path.exists(path)
    from PIL import Image

    w, h = Image.open(path).size
    assert w == h <= 2048 and (w & (w - 1)) == 0, (w, h)


@pytest.mark.parametrize("name,path", OBJECT_XMLS, ids=_names())
def test_object_loads_standalone(name, path):
    model = mujoco.MjModel.from_xml_path(path)
    # exactly one top-level body, named after the directory, with no joint
    roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0]
    assert len(roots) == 1, f"{name}: {len(roots)} top-level bodies"
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, roots[0]) == name
    assert model.njnt == 0, f"{name} must not carry its own freejoint (DESIGN 5.2)"


@pytest.mark.parametrize("name,path", OBJECT_XMLS, ids=_names())
def test_object_body_shape_contract(name, path):
    """Visual geoms group 2 + non-colliding; collision geoms group 3."""
    root = ET.parse(path).getroot()
    bodies = root.find("worldbody").findall("body")
    assert len(bodies) == 1 and bodies[0].get("name") == name
    assert bodies[0].find("freejoint") is None
    groups = set()
    for g in bodies[0].iter("geom"):
        grp = g.get("group")
        groups.add(grp)
        if grp == "2":
            assert g.get("contype") == "0" and g.get("conaffinity") == "0", name
        else:
            assert grp == "3", f"{name}: geom in group {grp}"
            assert g.get("condim") == "6" and g.get("friction") == "1 0.02 0.004"
    assert groups == {"2", "3"}, f"{name}: geom groups {groups}"


@pytest.mark.parametrize("name,path", OBJECT_XMLS, ids=_names())
def test_object_mass_and_sites(name, path):
    model = mujoco.MjModel.from_xml_path(path)
    mass = float(model.body_mass[1])
    assert MASS_MIN <= mass <= MASS_MAX, f"{name} mass {mass}"
    assert np.all(model.body_inertia[1] > 0), f"{name} has degenerate inertia"

    sites = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
             for i in range(model.nsite)}
    assert f"{name}_center" in sites
    if name in CONTAINERS:
        assert {f"{name}_opening", f"{name}_floor"} <= sites
        opening = model.site(f"{name}_opening")
        floor = model.site(f"{name}_floor")
        assert opening.pos[2] > floor.pos[2] > 0.0
        assert opening.size[0] > 0.03, f"{name} opening radius {opening.size[0]}"
    else:
        assert not {f"{name}_opening", f"{name}_floor"} & sites


@pytest.mark.parametrize("name,path", OBJECT_XMLS, ids=_names())
def test_object_origin_is_bottom_centre(name, path):
    """``pos="0 0 0"`` must rest on the floor, and XY must be centred.

    Raw YCB scans are up to 3.3 cm off-centre in XY; ``convert_object.py``
    corrects that, and ``scene.py``'s layout assumes it.  "Bottom" is the bottom
    of the **collision** shell -- that is what touches the plane.  LIBERO's box
    decompositions stop a few mm short of their visual mesh, so the visual is
    allowed to dip slightly below z=0 (invisible: it is under the object).
    """
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(ASSETS), "tools"))
    import convert_object  # noqa: E402  (regeneration tool, reused as a measurer)

    model = mujoco.MjModel.from_xml_path(path)
    lo, hi = convert_object._aabb(model)
    lo_c, _ = convert_object._aabb(model, groups=(3,))
    assert abs(lo_c[2]) < 1e-3, f"{name} collision bottom at z={lo_c[2]:.4f}"
    assert -5e-3 < lo[2] <= 1e-3, f"{name} visual bottom at z={lo[2]:.4f}"
    assert abs(lo[0] + hi[0]) < 2e-3 and abs(lo[1] + hi[1]) < 2e-3, f"{name} off-centre"


@pytest.mark.parametrize("name", FOODS)
def test_food_is_graspable(name):
    """Smallest bbox dimension must fit between the 2F-85 fingers."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(ASSETS), "tools"))
    import convert_object  # noqa: E402

    path = os.path.join(ASSETS, "objects", name, name + ".xml")
    lo, hi = convert_object._aabb(mujoco.MjModel.from_xml_path(path))
    assert min(hi - lo) <= GRASP_MAX_MIN_DIM, f"{name} min dim {min(hi - lo):.3f} m"


# --------------------------------------------------------------------------- #
#  combined scene -- exactly how scene.py will build it
# --------------------------------------------------------------------------- #
def _arena():
    from dm_control import mjcf

    arena = mjcf.RootElement(model="arena")
    arena.option.timestep = 0.002
    arena.worldbody.add("light", pos=[0, 0, 2], dir=[0, 0, -1])
    arena.worldbody.add("geom", name="floor", type="plane", size=[2, 2, 0.05],
                        condim=6, friction=[1, 0.02, 0.004], rgba=[0.5, 0.45, 0.4, 1])
    return arena


def _attach(arena, name, pos, free=True):
    """`scene.py` does exactly this: load the object file, attach it to the
    arena, and put the freejoint on the *attachment frame* (the object file
    itself must not carry one)."""
    from dm_control import mjcf

    obj = mjcf.from_path(os.path.join(ASSETS, "objects", name, name + ".xml"))
    frame = arena.worldbody.attach(obj)
    frame.pos = pos
    joint = frame.add("freejoint") if free else None
    return obj, joint


def test_every_object_settles_in_one_arena():
    """All 11 objects dropped from 5 cm onto separate spots settle within 1 s.

    "Settled" is asserted as *translational* speed < 1 mm/s plus a pose that
    stops changing, not as ``|qvel| < 1e-3`` over all six DoF.  A convex hull
    resting on a 2-3 point contact keeps a slowly decaying **rocking** mode
    (measured on the pear: omega oscillates 0.2 -> 0.05 rad/s over 3 s while the
    quaternion stays put to 5e-4 rad and the position to 0.2 mm).  That is
    MuJoCo soft-contact chatter, not an asset defect -- it survives
    ``integrator=implicitfast``, ``cone=elliptic``, ``impratio=10``,
    ``noslip_iterations=5``, a 1 ms timestep and a 4x stiffer ``solref``.  The
    angular bound below still catches a fruit that genuinely rolls away.
    """
    from dm_control import mjcf

    arena = _arena()
    handles = {}
    for i, (name, _) in enumerate(OBJECT_XMLS):
        x, y = -0.45 + 0.30 * (i % 4), -0.45 + 0.30 * (i // 4)
        handles[name] = _attach(arena, name, [x, y, 0.05])

    physics = mjcf.Physics.from_mjcf_model(arena)
    for _ in range(500):  # 1.0 s
        physics.step()
    pose0 = {n: np.array(physics.bind(j).qpos) for n, (_, j) in handles.items()}
    for _ in range(100):  # + 0.2 s, to prove it has stopped moving
        physics.step()

    bad = []
    for name, (obj, joint) in handles.items():
        qvel = np.array(physics.bind(joint).qvel)
        qpos = np.array(physics.bind(joint).qpos)
        z = float(physics.bind(obj.find("body", name)).xpos[2])
        lin = float(np.linalg.norm(qvel[:3]))
        ang = float(np.linalg.norm(qvel[3:]))
        drift = float(np.linalg.norm(qpos[:3] - pose0[name][:3]))
        if lin >= 1e-3 or ang >= 0.5 or drift >= 1e-4 or z < -1e-3:
            bad.append((name, round(lin, 6), round(ang, 4), round(drift, 6), round(z, 5)))
    assert not bad, f"did not settle (name, |v|, |w|, drift, z): {bad}"


@pytest.mark.parametrize("container", CONTAINERS)
def test_containers_are_hollow(container):
    """A strawberry dropped on the axis must end up *inside* the opening cylinder.

    This is the test that rejects a convex-hull "container" (the single-mesh YCB
    024_bowl fails it), and it is the same geometric predicate ``task.py`` uses
    for success.
    """
    from dm_control import mjcf

    arena = _arena()
    pot, _ = _attach(arena, container, [0, 0, 0], free=False)
    rim_z = float(pot.find("site", f"{container}_opening").pos[2])
    berry, _ = _attach(arena, "strawberry", [0, 0, rim_z + 0.05])

    physics = mjcf.Physics.from_mjcf_model(arena)
    for _ in range(750):  # 1.5 s: fall + settle inside
        physics.step()

    opening = physics.bind(pot.find("site", f"{container}_opening"))
    floor = physics.bind(pot.find("site", f"{container}_floor"))
    centre = physics.bind(berry.find("site", "strawberry_center")).xpos
    radius = float(opening.size[0])

    # the berry's origin is at its *bottom*, so resting on the inner floor puts
    # it at floor_z minus the contact penetration -- task.py needs the same slack
    assert centre[2] > floor.xpos[2] - 3e-3, f"{container}: berry below the inner floor"
    assert centre[2] < opening.xpos[2], f"{container}: berry sitting on/above the rim"
    assert math.hypot(centre[0] - opening.xpos[0],
                      centre[1] - opening.xpos[1]) < radius, f"{container}: berry outside"
