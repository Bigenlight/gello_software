# sim_collect objects

Vendored, ready-to-attach MuJoCo objects for the `carrot_in_pot` collection scene.
**These files are canonical** — `../../tools/fetch_assets.sh` only regenerates them.
Licences and citations: [`../ATTRIBUTION.md`](../ATTRIBUTION.md).

Every `objects/<name>/<name>.xml` is a complete `<mujoco model="<name>">` per
DESIGN.md §5.2: one top-level `<body name="<name>">`, **no freejoint** (`scene.py`
puts it on the attachment frame), meshes/textures next to the xml
(`meshdir="." texturedir="."`), visual geoms `group="2" contype="0" conaffinity="0"`,
collision geoms `group="3" condim="6" friction="1 0.02 0.004"`, explicit `<inertial>`.

## Table

| name | source | bbox cm (x×y×z) | mass kg | min dim cm | notes |
| --- | --- | --- | --- | --- | --- |
| `banana` | YCB `011_banana` | 10.9 × 17.8 × 3.7 | 0.12 | 3.7 | long; check the layout spacing |
| `strawberry` | YCB `012_strawberry` | 4.5 × 4.5 × 4.6 | 0.025 | 4.5 | lightest food; the hollowness probe |
| `lemon` | YCB `014_lemon` | 6.1 × 5.9 × 5.3 | 0.10 | 5.3 | |
| `peach` | YCB `015_peach` | 6.2 × 6.3 × 5.9 | 0.13 | 5.9 | |
| `pear` | YCB `016_pear` | 6.7 × 10.1 × 6.6 | 0.18 | 6.6 | fattest food still inside the 7 cm gate |
| `plum` | YCB `018_plum` | 5.7 × 5.5 × 5.3 | 0.07 | 5.3 | |
| `bread` | robosuite `bread` | 4.8 × 4.0 × 4.8 | 0.08 | 4.0 | STL, no UV detail; scaled 0.8 upstream |
| `carrot` | **procedural** | 18.5 × 3.7 × 3.6 | 0.08 | 3.6 | **the task object.** Lies on its side |
| `pot` | **procedural** | 24.0 × 19.2 × 11.0 | 0.55 | — | rim 0.1100, inner floor 0.0060, opening r 0.0900 |
| `bowl` | LIBERO `akita_black_bowl` | 11.2 × 11.1 × 5.3 | 0.15 | — | rim 0.0505, inner floor 0.0052, opening r 0.0501 |
| `basket` | LIBERO `basket` | 17.3 × 16.2 × 14.8 | 0.40 | — | rim 0.1415, inner floor 0.0174, opening r 0.0647 |

All eight foods are ≤ 7 cm in their smallest dimension, i.e. graspable by the
2F-85 (85 mm stroke) with finger clearance.  `test_assets.py::test_food_is_graspable`
holds the line.

## Sites (the interface `scene.py` / `task.py` consume)

| site | on | meaning |
| --- | --- | --- |
| `<name>_center` | every object | the body origin — **bottom centre** of the object |
| `<name>_opening` | pot, bowl, basket | `pos z` = rim height, `size[0]` = inner radius just under the rim |
| `<name>_floor` | pot, bowl, basket | `pos z` = the inside bottom surface |

Containment predicate (same one `test_containers_are_hollow` asserts):

```
floor_z - 3mm  <  food_center_z  <  opening_z      and
hypot(food_xy - opening_xy) < opening_size[0]
```

The 3 mm slack is not sloppiness: a food's origin is at its **bottom**, so a food
resting on the container's inner floor sits at `floor_z` minus the contact
penetration.

## Things that will bite you

* **Origin = bottom centre, not centroid.** `pos="0 0 0"` rests on the floor.
  Raw YCB scans are up to 3.3 cm off-centre in XY; that is corrected here, so
  `nominal_pos` in the layout yaml is the object's actual footprint centre.
* **"Bottom" is the bottom of the collision shell.** LIBERO's box decompositions
  stop 1.6 mm (bowl) / 4.2 mm (basket) short of their visual mesh, so those two
  visuals dip that far below z=0 — invisible, it is underneath the object.
* **The pot is procedural on purpose.** MuJoCo collides meshes as convex hulls,
  so *any* single-mesh pot/bowl is a solid lump.  The pot is 24 box staves + a
  base disc; the bowl and basket are used because LIBERO ships a box
  decomposition (40 resp. 5 boxes) that is genuinely hollow.  The single-mesh
  YCB `024_bowl` was rejected for exactly this reason.
* **The carrot's leaf tuft is visual-only** (3 green capsules, no collision), so
  the gripper closes on the body.  Its bbox therefore includes 3.4 cm of greens
  that nothing can hit; the carrot body alone is 14 cm long, Ø 3.5 cm → Ø 1.2 cm.
* **`condim="6"`** everywhere, matching the floor: without the rolling-friction
  term round fruit never stops rolling.
* A convex hull resting on 2–3 contacts keeps a slowly decaying *rocking* mode
  (≤ 0.25 rad/s, position stable to 0.2 mm).  It is MuJoCo soft-contact chatter,
  not an asset defect — see `test_every_object_settles_in_one_arena`.
