#!/usr/bin/env python
"""Rebuild the MuJoCo scene of a sim_collect take and replay / re-render it.

A take records (vectors.h5):
  /sim_scene            xml (the exact MJCF the sim compiled), attrs xml_sha256,
                        assets_manifest (sha256 per asset), layout, config, git_commit
  /sim_mj_state         t_rel_s, sim_t, tick, qpos[nq], qvel[nv], ctrl[nu] at 125 Hz

Asset bytes (meshes/textures, ~35 MB) are NOT copied into the take; they are resolved
by rebuilding the scene with ``scene.build_scene(config, layout)`` from the repo (the
manifest lets you verify nothing changed) and the stored XML is compiled against those
assets. Every recorded row is then ``d.qpos[:] = row.qpos; d.qvel[:] = row.qvel;
mj_forward`` — kinematic replay, no physics re-simulation, so it reproduces exactly what
was recorded (objects, arm, gripper), from any camera.

Usage (from the repo root, .venv python, MUJOCO_GL=glfw DISPLAY=:0 for viewer/render):
  python -m sim_collect.tools.replay_take <take_dir> --check
  python -m sim_collect.tools.replay_take <take_dir> --viewer            # real-time playback
  python -m sim_collect.tools.replay_take <take_dir> --render cam1 cam2 --out /tmp/frames --every 15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "ros2_ur_ws", "src", "ur_gello_bringup")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass
class TakeScene:
    xml: str
    xml_sha256: str
    assets: Dict[str, bytes]
    layout: Dict[str, Any]
    config: Dict[str, Any]
    manifest: Dict[str, Dict[str, Any]]
    rebuilt_matches: Optional[bool]     # None = could not rebuild (no config), else sha equality
    notes: List[str]


@dataclass
class TakeState:
    t_rel_s: np.ndarray
    sim_t: np.ndarray
    tick: np.ndarray
    qpos: np.ndarray    # (n, nq)
    qvel: np.ndarray    # (n, nv)
    ctrl: np.ndarray    # (n, nu)


def load_scene(take_dir: str) -> TakeScene:
    """Read /sim_scene and resolve assets by rebuilding the scene from the repo."""
    notes: List[str] = []
    with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
        if "sim_scene" not in f:
            raise KeyError("take has no /sim_scene group (recorded before scene snapshots existed)")
        g = f["sim_scene"]
        xml = g["xml"][()] if "xml" in g else ""
        xml = xml.decode() if isinstance(xml, bytes) else str(xml)
        sha = str(g.attrs.get("xml_sha256", ""))
        manifest = json.loads(str(g.attrs.get("assets_manifest", "{}")))
        layout = json.loads(str(g.attrs.get("layout", "{}")))
        config = json.loads(str(g.attrs.get("config", "{}")))
    assets: Dict[str, bytes] = {}
    rebuilt_matches: Optional[bool] = None
    if config:
        try:
            from sim_collect.scene import SceneConfig, build_scene
            cfg = SceneConfig.from_dict(dict(config))
            built = build_scene(cfg, layout or None)
            assets = dict(built.assets)
            rebuilt_sha = hashlib.sha256(built.xml.encode()).hexdigest()
            rebuilt_matches = (rebuilt_sha == sha) if sha else None
            if rebuilt_matches is False:
                notes.append("rebuilt scene XML differs from the recorded one (assets/code changed since the "
                             "take?) — compiling the RECORDED xml against the rebuilt assets")
            changed = [k for k, v in manifest.items()
                       if k in assets and hashlib.sha256(assets[k]).hexdigest() != v.get("sha256")]
            missing = [k for k in manifest if k not in assets]
            if changed:
                notes.append(f"{len(changed)} asset(s) changed since the take: {changed[:5]}")
            if missing:
                notes.append(f"{len(missing)} asset(s) missing now: {missing[:5]}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"scene rebuild failed ({type(e).__name__}: {e}); compiling recorded xml without assets")
    else:
        notes.append("no config stored; compiling recorded xml without assets")
    return TakeScene(xml, sha, assets, layout, config, manifest, rebuilt_matches, notes)


def load_model(scene: TakeScene):
    import mujoco
    return mujoco.MjModel.from_xml_string(scene.xml, scene.assets or None)


def load_state(take_dir: str) -> TakeState:
    with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
        if "sim_mj_state" not in f:
            raise KeyError("take has no /sim_mj_state table")
        g = f["sim_mj_state"]
        cols = json.loads(g.attrs["columns"])
        nq, nv, nu = int(g.attrs["nq"]), int(g.attrs["nv"]), int(g.attrs["nu"])
        col = {c: g[c][:] for c in cols}
        n = len(col["t_rel_s"])
        qpos = np.stack([col[f"qpos{i}"] for i in range(nq)], 1) if n else np.zeros((0, nq))
        qvel = np.stack([col[f"qvel{i}"] for i in range(nv)], 1) if n else np.zeros((0, nv))
        ctrl = np.stack([col[f"ctrl{i}"] for i in range(nu)], 1) if n else np.zeros((0, nu))
    return TakeState(col["t_rel_s"], col["sim_t"], col["tick"], qpos, qvel, ctrl)


def set_row(model, data, st: TakeState, i: int) -> None:
    """Kinematic replay of row i: overwrite the generalized state and run mj_forward."""
    import mujoco
    if st.qpos.shape[1] != model.nq or st.qvel.shape[1] != model.nv:
        raise ValueError(f"state dims (nq {st.qpos.shape[1]}, nv {st.qvel.shape[1]}) != model (nq {model.nq}, nv {model.nv})")
    data.qpos[:] = st.qpos[i]
    data.qvel[:] = st.qvel[i]
    if np.all(np.isfinite(st.ctrl[i])) and st.ctrl.shape[1] == model.nu:
        data.ctrl[:] = st.ctrl[i]
    data.time = float(st.sim_t[i]) if np.isfinite(st.sim_t[i]) else data.time
    mujoco.mj_forward(model, data)


def check(take_dir: str) -> int:
    """Rebuild, replay the first/last rows and compare object poses with sim_object_poses."""
    import mujoco
    scene = load_scene(take_dir)
    for n in scene.notes:
        print("  note:", n)
    model = load_model(scene)
    st = load_state(take_dir)
    print(f"model: nq {model.nq} nv {model.nv} nu {model.nu} | rows {len(st.t_rel_s)} | "
          f"rebuilt xml matches: {scene.rebuilt_matches} | assets {len(scene.assets)}")
    data = mujoco.MjData(model)
    worst = 0.0
    with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
        if "sim_object_poses" in f and len(st.t_rel_s):
            g = f["sim_object_poses"]
            cols = json.loads(g.attrs["columns"])
            names = sorted({c[:-2] for c in cols if c.endswith("_x")})
            t_obj = g["t_rel_s"][:]
            for i in (0, len(st.t_rel_s) - 1):
                set_row(model, data, st, i)
                j = int(np.argmin(np.abs(t_obj - st.t_rel_s[i])))
                for name in names:
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}/{name}")
                    if bid < 0:
                        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    if bid < 0:
                        continue
                    rec = np.array([g[f"{name}_{a}"][j] for a in ("x", "y", "z")])
                    err = float(np.linalg.norm(data.xpos[bid] - rec))
                    worst = max(worst, err)
            print(f"object pose reconstruction error (replayed vs recorded sim_object_poses): max {1e3 * worst:.2f} mm")
    return 0 if worst < 0.01 else 1


def replay_viewer(take_dir: str, speed: float = 1.0) -> None:
    import mujoco
    import mujoco.viewer
    scene = load_scene(take_dir)
    model = load_model(scene)
    st = load_state(take_dir)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            t0 = time.monotonic()
            for i in range(len(st.t_rel_s)):
                if not v.is_running():
                    break
                set_row(model, data, st, i)
                v.sync()
                target = t0 + (st.t_rel_s[i] - st.t_rel_s[0]) / max(speed, 1e-6)
                dt = target - time.monotonic()
                if dt > 0:
                    time.sleep(dt)


def render_frames(take_dir: str, cams: Sequence[str], out_dir: str, every: int = 30,
                  size=(1280, 720)) -> int:
    import cv2
    import mujoco
    scene = load_scene(take_dir)
    model = load_model(scene)
    st = load_state(take_dir)
    data = mujoco.MjData(model)
    os.makedirs(out_dir, exist_ok=True)
    r = mujoco.Renderer(model, height=size[1], width=size[0])
    n = 0
    for i in range(0, len(st.t_rel_s), max(1, every)):
        set_row(model, data, st, i)
        for cam in cams:
            r.update_scene(data, camera=cam)
            img = r.render()
            cv2.imwrite(os.path.join(out_dir, f"{cam}_{i:06d}.jpg"), img[:, :, ::-1])
            n += 1
    print(f"wrote {n} frames to {out_dir}")
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take_dir")
    ap.add_argument("--check", action="store_true", help="rebuild + verify object poses against the recorded ones")
    ap.add_argument("--viewer", action="store_true", help="play the take back in a MuJoCo viewer (kinematic)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--render", nargs="*", metavar="CAM", help="render these cameras to --out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=30, help="render every N-th state row (125 Hz rows)")
    args = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "glfw")
    rc = 0
    if args.check or not (args.viewer or args.render):
        rc = check(args.take_dir)
    if args.render:
        render_frames(args.take_dir, args.render or ["cam1", "cam2"], args.out or os.path.join(args.take_dir, "replay_frames"), args.every)
    if args.viewer:
        replay_viewer(args.take_dir, args.speed)
    return rc


if __name__ == "__main__":
    sys.exit(main())
