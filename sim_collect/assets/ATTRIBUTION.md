# Asset attribution

Everything under `sim_collect/assets/` is vendored third-party art, re-packaged by
`sim_collect/tools/convert_object.py` (see `tools/fetch_assets.sh` for the exact
provenance of each file).  **The vendored files are canonical**; the tools exist
only to regenerate them.  None of the upstream projects is a runtime dependency —
only their mesh/texture files are used.

| what | files | upstream | licence |
| --- | --- | --- | --- |
| floor textures | `textures/martin_novak_wood_table.png`, `textures/seamless_wood_planks_floor.png` | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) `libero/libero/assets/textures/` | MIT — [`LICENSE.LIBERO`](LICENSE.LIBERO) |
| `objects/bowl/` | `bowl_akita_black_bowl_vis.msh`, `bowl_texture.png` | LIBERO `assets/stable_scanned_objects/akita_black_bowl/` | MIT — [`LICENSE.LIBERO`](LICENSE.LIBERO) |
| `objects/basket/` | `basket_basket_vis.msh`, `basket_texture.png` | LIBERO `assets/stable_scanned_objects/basket/` | MIT — [`LICENSE.LIBERO`](LICENSE.LIBERO) |
| `objects/bread/` | `bread_bread.stl`, `bread_bread.png` | [robosuite](https://github.com/ARISE-Initiative/robosuite) `robosuite/models/assets/objects/bread.xml` | MIT — [`LICENSE.robosuite`](LICENSE.robosuite) |
| `objects/{banana,strawberry,lemon,peach,pear,plum}/` | `<name>.obj`, `<name>.png` | [YCB Object and Model Set](https://www.ycbbenchmarks.com/) `google_16k` scans `011`, `012`, `014`, `015`, `016`, `018` | **CC BY 4.0** — [`LICENSE.YCB`](LICENSE.YCB) |
| `objects/{carrot,pot}/` | `<name>.xml` only | written here (`tools/convert_object.py`) | same as this repository |
| `preview.jpg` | — | rendered from the above | — |

## Required YCB attribution

The YCB object and model data is licensed **CC BY 4.0**
(<https://creativecommons.org/licenses/by/4.0/>).  Cite:

> B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel and A. M. Dollar,
> "Benchmarking in Manipulation Research: Using the Yale-CMU-Berkeley Object and
> Model Set", *IEEE Robotics and Automation Magazine*, 22(3):36–52, Sept. 2015.

> B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel and A. M. Dollar,
> "Benchmarking in Manipulation Research: The YCB Object and Model Set and
> Benchmarking Protocols", *IEEE Robotics and Automation Magazine*, 2015.

**Changes made** (CC BY requires stating them): `textured.obj` copied verbatim
(metres, geometry unmodified); `mtllib`/`usemtl` lines stripped so MuJoCo uses the
material declared in our XML; `texture_map.png` downscaled to 1024×1024; the mesh
repositioned so the object origin is the bottom centre of its bounding box; an
explicit mass/inertia added.

## Changes to the LIBERO / robosuite assets

Meshes and textures are byte-identical to upstream except that textures are
downscaled to ≤ 1024² (LIBERO ships 4096²; the policy never sees more than
128×128).  The XMLs are **not** upstream's: the anonymous wrapper `<body>` is
removed (it blocks `<freejoint/>`), asset names are namespaced with the object
name, placement sites are dropped, `density="50..100"` is replaced by a real
mass, and `solref="0.001 1"` is dropped — a 1 ms contact time constant is
unstable at this project's 2 ms timestep.

## Robot

`assets/robots/ur7e/` is produced by the integrator from
[`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0); see its own `README.md`.
