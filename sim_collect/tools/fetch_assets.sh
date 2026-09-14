#!/usr/bin/env bash
# Regenerate sim_collect/assets/{textures,objects} from upstream.
#
#   >>> THE VENDORED TREE UNDER sim_collect/assets/ IS CANONICAL. <<<
#
# Nothing at runtime runs this script and the test suite never calls it; it is
# here so the vendored files can be *reproduced* (or a new object added) without
# guesswork about where they came from and what was done to them.  Every
# transformation lives in tools/convert_object.py -- read its docstring first.
#
#   ./sim_collect/tools/fetch_assets.sh [OUTDIR]      # default: sim_collect/assets
#
# Offline / already-cloned sources (skips the network entirely):
#   LIBERO_DIR=<...>/LIBERO ROBOSUITE_DIR=<...>/robosuite YCB_DIR=<dir with 0NN_name/> \
#     ./sim_collect/tools/fetch_assets.sh
#
# Licences: LIBERO MIT, robosuite MIT, YCB data CC BY 4.0 (see assets/ATTRIBUTION.md).
# Neither LIBERO nor robosuite is a runtime dependency -- only their mesh files
# are vendored.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$(cd "$HERE/.." && pwd)/assets}"
PY="${PY:-/home/laptop3/gello_software/.venv/bin/python}"   # mujoco 3.10, PIL
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT/textures" "$OUT/objects"

# ---------------------------------------------------------------- sources ----
if [[ -n "${LIBERO_DIR:-}" ]]; then
  LIB="$LIBERO_DIR"
else
  LIB="$TMP/libero"
  git clone --filter=blob:none --no-checkout --depth 1 \
      https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIB"
  git -C "$LIB" sparse-checkout init --cone
  git -C "$LIB" sparse-checkout set libero/libero/assets LICENSE
  git -C "$LIB" checkout
fi
if [[ -n "${ROBOSUITE_DIR:-}" ]]; then
  RSU="$ROBOSUITE_DIR"
else
  RSU="$TMP/robosuite"
  git clone --filter=blob:none --no-checkout --depth 1 \
      https://github.com/ARISE-Initiative/robosuite.git "$RSU"
  git -C "$RSU" sparse-checkout init --cone
  git -C "$RSU" sparse-checkout set robosuite/models/assets/objects robosuite/models/assets/textures LICENSE
  git -C "$RSU" checkout
fi
L="$LIB/libero/libero/assets"
R="$RSU/robosuite/models/assets"
[[ -f "$LIB/LICENSE" ]] && cp "$LIB/LICENSE" "$OUT/LICENSE.LIBERO"
[[ -f "$RSU/LICENSE" ]] && cp "$RSU/LICENSE" "$OUT/LICENSE.robosuite"

# YCB fruit (CC BY 4.0).  The bucket is us-east-1; the us-east-2 host 301-redirects.
YCB_BASE="https://ycb-benchmarks.s3.us-east-1.amazonaws.com/data/google"
YCB_IDS="011_banana 012_strawberry 014_lemon 015_peach 016_pear 018_plum"
if [[ -n "${YCB_DIR:-}" ]]; then
  YCB="$YCB_DIR"
else
  YCB="$TMP/ycb"; mkdir -p "$YCB"
  for id in $YCB_IDS; do
    curl -fsSL "$YCB_BASE/${id}_google_16k.tgz" -o "$TMP/$id.tgz"
    tar xzf "$TMP/$id.tgz" -C "$YCB"
  done
fi

# --------------------------------------------------------------- textures ----
# LIBERO ships these at 1270^2 / 2048^2; DESIGN 5.2 wants power-of-two <= 2048.
"$PY" - "$L/textures" "$OUT/textures" <<'PYEOF'
import sys, os
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
for n in ("martin_novak_wood_table.png", "seamless_wood_planks_floor.png"):
    im = Image.open(os.path.join(src, n)).convert("RGB").resize((1024, 1024), Image.LANCZOS)
    im.save(os.path.join(dst, n), optimize=True)
    print("texture", n, im.size)
PYEOF

# ---------------------------------------------------------------- objects ----
"$PY" - "$HERE" "$OUT" "$L" "$R" "$YCB" <<'PYEOF'
import json, os, sys
sys.path.insert(0, sys.argv[1])
import convert_object as C

out, L, R, YCB = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
objects = os.path.join(out, "objects")
rows = []

# --- foods -------------------------------------------------------------------
# YCB google_16k scans: metres, z_min ~ 0, but NOT centred in XY (the converter
# re-centres them).  Masses are real produce weights, not the scan's.
for ycb_id, name, mass in [
    ("011_banana",     "banana",     0.12),
    ("012_strawberry", "strawberry", 0.025),
    ("014_lemon",      "lemon",      0.10),
    ("015_peach",      "peach",      0.13),
    ("016_pear",       "pear",       0.18),
    ("018_plum",       "plum",       0.07),
]:
    rows.append(dict(C.from_ycb(os.path.join(YCB, ycb_id, "google_16k"), objects,
                                name, mass), source="YCB " + ycb_id))

rows.append(dict(C.from_mjcf(os.path.join(R, "objects", "bread.xml"), objects,
                             "bread", 0.08), source="robosuite bread"))
rows.append(dict(C.make_carrot(objects), source="procedural"))

# --- containers --------------------------------------------------------------
# A scanned pot is useless: MuJoCo collides meshes as convex hulls, so any
# single-mesh pot is a solid lump.  The pot is built from box staves instead.
rows.append(dict(C.make_pot(objects), source="procedural"))
# LIBERO's akita bowl / basket ship a 40-box (resp. 5-box) collision
# decomposition, which is genuinely hollow -- that is why they are used and the
# single-mesh YCB 024_bowl is not.
rows.append(dict(C.from_mjcf(
    os.path.join(L, "stable_scanned_objects", "akita_black_bowl", "akita_black_bowl.xml"),
    objects, "bowl", 0.15, container=True), source="LIBERO akita_black_bowl"))
rows.append(dict(C.from_mjcf(
    os.path.join(L, "stable_scanned_objects", "basket", "basket.xml"),
    objects, "basket", 0.40, container=True), source="LIBERO basket"))

with open(os.path.join(objects, "index.json"), "w") as fh:
    json.dump(rows, fh, indent=1)
for r in rows:
    print(json.dumps(r))
PYEOF

cat > "$OUT/LICENSE.YCB" <<'YEOF'
Yale-CMU-Berkeley (YCB) Object and Model Set
The YCB object/model data is released under the Creative Commons Attribution 4.0
International licence (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

Required attribution / citation:

  B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel and A. M. Dollar,
  "Benchmarking in Manipulation Research: Using the Yale-CMU-Berkeley Object and
  Model Set", IEEE Robotics and Automation Magazine, 22(3):36-52, Sept. 2015.

  B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel and A. M. Dollar,
  "Benchmarking in Manipulation Research: The YCB Object and Model Set and
  Benchmarking Protocols", IEEE Robotics and Automation Magazine, 2015.

Changes made here: the google_16k textured.obj was copied verbatim (metres),
mtllib/usemtl lines were stripped, texture_map.png was downscaled to 1024x1024,
and the mesh was re-positioned so the object's origin is at the bottom centre of
its bounding box.  No geometry was modified.
YEOF

# ---------------------------------------------------------------- preview ----
# One contact sheet of every candidate on the textured floor, for the operator.
# MUST run in its own process (a mujoco.Renderer and a passive viewer deadlock in
# one process -- DESIGN 1) and needs a real X display: MUJOCO_GL=glfw only.
if [[ -n "${DISPLAY:-}" ]]; then
  MUJOCO_GL=glfw "$PY" - "$OUT" <<'PYEOF'
import os, sys, math
import numpy as np, cv2
from dm_control import mjcf

out = sys.argv[1]
rows = [
    (0.20, ["carrot", "banana", "pear", "peach"]),
    (0.00, ["strawberry", "plum", "lemon", "bread"]),
    (-0.26, ["pot", "bowl", "basket"]),
]
arena = mjcf.RootElement(model="preview")
arena.compiler.texturedir = os.path.join(out, "textures")
g = arena.visual.get_children("global")
g.offwidth, g.offheight = 1280, 720
arena.visual.headlight.diffuse = [.45, .45, .45]
arena.visual.headlight.ambient = [.35, .35, .35]
arena.visual.headlight.specular = [.05, .05, .05]
arena.asset.add("texture", name="floor_tex", type="2d",
                file="martin_novak_wood_table.png")
arena.asset.add("material", name="floor_mat", texture="floor_tex",
                texrepeat=[3, 3], texuniform=True, reflectance=0.03, specular=0.15)
arena.worldbody.add("geom", name="floor", type="plane", size=[3, 3, .05],
                    material="floor_mat", condim=6, friction=[1, .02, .004])
arena.worldbody.add("light", pos=[0.3, -0.5, 1.5], dir=[-0.2, 0.35, -1],
                    directional=False, diffuse=[.45, .45, .45], specular=[.1, .1, .1])
arena.worldbody.add("light", pos=[-0.5, 0.3, 1.2], dir=[0.4, -0.25, -1],
                    directional=False, diffuse=[.35, .35, .35], specular=[.1, .1, .1])

spots = {}
for y, names in rows:
    span = 0.175 if y > -0.2 else 0.32
    for i, n in enumerate(names):
        x = (i - (len(names) - 1) / 2) * span
        path = os.path.join(out, "objects", n, n + ".xml")
        obj = mjcf.from_path(path)
        frame = arena.worldbody.attach(obj)
        frame.pos = [x, y, 0.0]
        sys.path.insert(0, os.path.join(os.path.dirname(out), "tools"))
        import convert_object, mujoco as _mj
        lo, hi = convert_object._aabb(_mj.MjModel.from_xml_path(path))
        spots[n] = (x, y, float(hi[2]) + 0.035)   # label just above the object

cam = np.array([0.0, -1.02, 0.62]); tgt = np.array([0.0, -0.03, 0.06])
z = (cam - tgt) / np.linalg.norm(cam - tgt)
x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
y = np.cross(z, x)
fovy = 45.0
arena.worldbody.add("camera", name="preview", pos=cam.tolist(), fovy=fovy,
                    xyaxes=[*x, *y])

physics = mjcf.Physics.from_mjcf_model(arena)
m, d = physics.model.ptr, physics.data.ptr
import mujoco
r = mujoco.Renderer(m, 720, 1280)
r.update_scene(d, camera="preview")
img = r.render()
del r

# label each object by projecting its centre through the camera
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "preview")
cp, cm_ = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
f = (720 / 2) / math.tan(math.radians(fovy) / 2)
bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for n, (px, py, pz) in spots.items():
    pc = cm_.T @ (np.array([px, py, pz]) - cp)
    if pc[2] >= 0:
        continue
    u, v = int(1280 / 2 + f * pc[0] / -pc[2]), int(720 / 2 - f * pc[1] / -pc[2])
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(bgr, n, (u - 8 * len(n) // 2, v), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, th, cv2.LINE_AA)
dst = os.path.join(out, "preview.jpg")
cv2.imwrite(dst, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
print("preview", dst, os.path.getsize(dst) // 1024, "KiB")
PYEOF
else
  echo "no DISPLAY: skipping assets/preview.jpg"
fi

du -sh "$OUT"
echo "regenerated $OUT -- now re-run sim_collect/tests/test_assets.py"
