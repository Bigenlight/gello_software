#!/usr/bin/env python
"""Re-render a sim_collect take's cameras on an EXACT 30 Hz grid from the recorded
MuJoCo state, writing a new take directory that the LeRobot converter can consume
without time warp.

Why: the live capture renders in software GL and, under CPU load, delivers 24–27 fps
while the mp4 is stamped 30 fps (the recorder flags this in ``sim_meta.problems``:
"cam1 captured at 25.2 fps but cam1.mp4 is stamped 30 (plays 1.19x fast)"). A video
model trained on that sees dynamics ~15–25 % too fast. Because every take also stores
``/sim_scene`` (the exact MJCF) and ``/sim_mj_state`` (qpos/qvel at 125 Hz), the frames
can be re-rendered offline at any rate: for each grid time ``t_k = t_start + k/fps`` we
take the nearest recorded state row (≤ 4 ms away), ``mj_forward`` it and render both
cameras with the same CameraRig/quality flags the live capture uses. The state tables
are copied verbatim (they were never time-warped — 125 Hz, timestamped), only
``cam{1,2}_frames`` are rewritten to the new frames, and ``sim_meta`` records the
retiming so provenance is explicit.

Output: ``<out_root>/<take name>/`` with ``vectors.h5`` (copy + rewritten frame tables),
``cam1.mp4``, ``cam2.mp4`` (mp4v, exact fps, one frame per grid tick, frame count ==
``camN_frames`` rows). The original take is never modified.

Usage (repo root, .venv python; rendering needs MUJOCO_GL=glfw DISPLAY=:0):
  python -m sim_collect.tools.retime_take <take_dir> --out <out_root> [--fps 30]
  python -m sim_collect.tools.retime_take --all ros2_ur_ws/gello_logs/sim --out ros2_ur_ws/gello_logs/sim_retimed
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "ros2_ur_ws", "src", "ur_gello_bringup"),
           os.path.join(_ROOT, "ros2_ur_ws", "src", "gello_recorder")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CAMS = ("cam1", "cam2")


def _write_frame_table(h5: h5py.File, name: str, t_rel: np.ndarray, frame_idx: np.ndarray,
                       stamp_s: np.ndarray, columns_ref: Optional[List[str]]) -> None:
    """Replace group `name` with the recorder's layout (float64 resizable datasets,
    ``columns`` JSON attr), keeping the original column set (with or without stamp_s)."""
    cols = columns_ref or ["t_rel_s", "frame_idx", "stamp_s"]
    if name in h5:
        del h5[name]
    g = h5.create_group(name)
    g.attrs["columns"] = json.dumps(cols)
    data = {"t_rel_s": np.round(t_rel, 4), "frame_idx": frame_idx.astype(np.float64), "stamp_s": stamp_s}
    for c in cols:
        v = data.get(c, np.full(len(t_rel), np.nan))
        g.create_dataset(c, data=np.asarray(v, dtype=np.float64), maxshape=(None,), chunks=True)


def retime_take(take_dir: str, out_root: str, fps: float = 30.0, cams: Sequence[str] = CAMS,
                overwrite: bool = False, quiet: bool = False) -> Dict[str, Any]:
    import cv2
    import mujoco
    from sim_collect import cameras as _cams
    from sim_collect.tools import replay_take as rt

    take_dir = os.path.abspath(take_dir)
    name = os.path.basename(take_dir.rstrip("/"))
    out_dir = os.path.join(os.path.abspath(out_root), name)
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    scene = rt.load_scene(take_dir)
    st = rt.load_state(take_dir)
    if len(st.t_rel_s) == 0:
        raise ValueError(f"{name}: empty sim_mj_state")
    with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
        meta = json.loads(f.attrs["sim_meta"]) if "sim_meta" in f.attrs else {}
        cfg = json.loads(str(f["sim_scene"].attrs.get("config", "{}"))) if "sim_scene" in f else {}
        frame_cols = {c: (json.loads(f[f"{c}_frames"].attrs["columns"]) if f"{c}_frames" in f else None) for c in cams}
        orig_first = {c: (float(f[f"{c}_frames"]["t_rel_s"][0]) if f"{c}_frames" in f and f[f"{c}_frames"]["t_rel_s"].shape[0] else None) for c in cams}
        t_wall0 = None
        if "synchronized" in f and "t_wall" in f["synchronized"] and f["synchronized"]["t_wall"].shape[0]:
            t_wall0 = float(f["synchronized"]["t_wall"][0]) - float(f["synchronized"]["t_rel_s"][0])

    # Grid: start at the first frame the live capture managed (so the clip covers the
    # same interval), never before the first state row; end at the last state row.
    t_first = max(min(v for v in orig_first.values() if v is not None) if any(orig_first.values()) else st.t_rel_s[0],
                  float(st.t_rel_s[0]))
    t_last = float(st.t_rel_s[-1])
    n = int(np.floor((t_last - t_first) * fps)) + 1
    grid = t_first + np.arange(n) / fps
    rows = np.searchsorted(st.t_rel_s, grid)
    rows = np.clip(rows, 0, len(st.t_rel_s) - 1)
    prev = np.clip(rows - 1, 0, len(st.t_rel_s) - 1)
    rows = np.where(np.abs(st.t_rel_s[prev] - grid) < np.abs(st.t_rel_s[rows] - grid), prev, rows)
    max_dt = float(np.max(np.abs(st.t_rel_s[rows] - grid)))

    model = rt.load_model(scene)
    rig = _cams.CameraRig(scene.xml, scene.assets, cams=cams, render=(cfg.get("render") or {}))
    model = rig.model  # rig compiles the same xml with offscreen buffer sizing + quality flags
    data = rig.data
    if st.qpos.shape[1] != model.nq:
        raise ValueError(f"{name}: sim_mj_state nq {st.qpos.shape[1]} != model nq {model.nq}")
    w, h = rig.color_size
    writers = {c: cv2.VideoWriter(os.path.join(out_dir, f"{c}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) for c in cams}
    t0 = time.perf_counter()
    for k, i in enumerate(rows):
        data.qpos[:] = st.qpos[i]
        data.qvel[:] = st.qvel[i]
        mujoco.mj_forward(model, data)
        for c in cams:
            rgb = rig.render_color(c)
            writers[c].write(np.ascontiguousarray(rgb[:, :, ::-1]))
        if not quiet and (k % 300 == 0 or k == n - 1):
            el = time.perf_counter() - t0
            print(f"  [{name}] frame {k + 1}/{n}  {1e3 * el / (k + 1):.1f} ms/frame", flush=True)
    for wv in writers.values():
        wv.release()
    rig.close() if hasattr(rig, "close") else None

    # vectors.h5: copy, rewrite frame tables, annotate sim_meta
    shutil.copy2(os.path.join(take_dir, "vectors.h5"), os.path.join(out_dir, "vectors.h5"))
    stamp = (grid + t_wall0) if t_wall0 is not None else np.full(n, np.nan)
    with h5py.File(os.path.join(out_dir, "vectors.h5"), "r+") as f:
        for c in cams:
            _write_frame_table(f, f"{c}_frames", grid, np.arange(n), stamp, frame_cols[c])
        meta = json.loads(f.attrs["sim_meta"]) if "sim_meta" in f.attrs else {}
        meta["retimed"] = {
            "from": take_dir, "fps": fps, "frames": n, "grid_start_t_rel_s": round(float(t_first), 4),
            "max_state_lookup_dt_s": round(max_dt, 5),
            "note": ("cam1/cam2 re-rendered from /sim_mj_state on an exact fps grid; state tables verbatim; "
                     "synchronized.cam*_frame_idx still refer to the ORIGINAL live frames"),
            "original_problems": meta.get("problems") or [],
            "original_achieved_fps": meta.get("achieved_fps_take"),
        }
        meta["problems"] = [p for p in (meta.get("problems") or []) if "captured at" not in p]
        meta["achieved_fps_take"] = {c: fps for c in cams}
        f.attrs["sim_meta"] = json.dumps(meta)
    return {"take": name, "out_dir": out_dir, "frames": n, "fps": fps, "max_state_lookup_dt_s": max_dt,
            "render_s": round(time.perf_counter() - t0, 1)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take_dir", nargs="?", help="one take directory")
    ap.add_argument("--all", metavar="ROOT", help="retime every take_* under ROOT")
    ap.add_argument("--out", required=True, help="output root (one sub-dir per take)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest", default=None, help="write a JSON summary here (default <out>/retime_manifest.json)")
    args = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "glfw")
    takes = sorted(os.path.join(args.all, d) for d in os.listdir(args.all) if d.startswith("take_")) if args.all else [args.take_dir]
    if not takes or takes == [None]:
        ap.error("give a take_dir or --all ROOT")
    os.makedirs(args.out, exist_ok=True)
    results = []
    for t in takes:
        try:
            r = retime_take(t, args.out, fps=args.fps, overwrite=args.overwrite)
        except FileExistsError as e:
            print(f"skip (exists): {e}")
            continue
        print(json.dumps(r))
        results.append(r)
    mpath = args.manifest or os.path.join(args.out, "retime_manifest.json")
    with open(mpath, "w") as fh:
        json.dump({"fps": args.fps, "takes": results, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
    print(f"manifest: {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
