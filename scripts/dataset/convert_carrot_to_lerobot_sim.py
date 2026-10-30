#!/usr/bin/env python3
"""
Convert the SIMULATED "Put carrot in pot" takes (sim_collect: MuJoCo UR7e + Robotiq 2F-85
teleoperated with a physical GELLO leader) into a LeRobot v3.0 dataset.

This is `convert_carrot_to_lerobot.py` (the real-robot converter) MINUS depth, PLUS sim
provenance. Everything the two converters share is kept bit-for-bit: the same feature
names and shapes for `action` / `observation.state` / `observation.images.cam{1,2}`, the
same fps / robot_type / task string, the same master clock, the same nearest-timestamp
resampling and the same decode order -- so the real validator's independent re-derivation
must still see max|delta| = 0 on the joint features.

Recipe
------
  - fps = 30 (camera rate). Master clock = cam1_frames/t_rel_s. Every other stream is
    resampled onto it by nearest-timestamp lookup on its own t_rel_s.
  - observation.state (7): ur_joint_states q1..q6 + gripper.grip_pos
  - action           (7): command cmd1..cmd6 + gripper.grip_cmd (ffill/bfill)
  - observation.images.cam1 / cam2: 720p RGB video (HWC uint8) -- cam1 = fixed scene
    camera, cam2 = wrist camera, exactly as in the real dataset.
  - NO depth features. The default sim take has no depth.h5 (sim_collect records depth
    only with --depth), and this converter never reads one even if present.
  - observation.sim.object_poses (14, float32, --with-sim-extras, default ON): the
    ground-truth MuJoCo poses of the two task objects, `carrot_{x,y,z,qx,qy,qz,qw}` +
    `pot_{x,y,z,qx,qy,qz,qw}` (metres, world frame, quaternion xyzw), sampled from the
    30 Hz `sim_object_poses` table by the SAME nearest-timestamp rule. It is a plain
    parquet column: policies that do not list it as an input feature ignore it, and
    evaluation / analysis code gets ground truth for free.
  - gello_* / sim_control / sim_leader_filtered / sim_mj_state / sim_scene are not
    carried over (they are in the raw take; sim_scene + sim_mj_state reconstruct any
    instant kinematically -- see sim_collect/tools/replay_take.py).

Timestamp correction
--------------------
The real converter shifts `ur_joint_states` back by a measured recorder lag (tau ~ 0.9 s).
Sim rows are tick-synchronous -- the physics thread stamps every table at the tick it was
produced -- so the default here is **--ur-lag-s 0** (no shift, no stale tail). The same
`--ur-lag-s` / `--lag-json` / `--drop-stale-tail` machinery is kept, unchanged, so the
real validator's checks 14 / 16 read the same `timestamp_correction` block.

Retimed takes (the intended input)
----------------------------------
The live sim capture renders in software GL and delivered 24-27 fps under CPU load while
the mp4 was stamped 30 fps ("plays 1.19x fast"). `sim_collect/tools/retime_take.py`
re-renders cam1/cam2 from the recorded MuJoCo state (`/sim_mj_state`, 125 Hz) on an exact
30 Hz grid and writes `sim_meta.retimed`; the state tables are copied verbatim. By default
this converter REFUSES a take that is not retimed or whose `sim_meta.problems` is not
empty (`--allow-unretimed` to override, e.g. for a take captured at a clean 30 fps).

Side file written after ds.finalize() (so lerobot's own writer cannot clobber meta/):
meta/source_takes.json -- episode_index -> take, the timestamp_correction block (schema
identical to the real converter's), and per-episode `sim` provenance from `sim_meta`:
git_commit, mujoco_version, sim_collect_version, layout_seed as recorded, the sampled
layout, the first-row object xy from `sim_object_poses` (the ground truth even where the
recorded seed is wrong -- see LAYOUT_SEED_NOTE), the `retimed` block,
original_achieved_fps, task_success_at_stop, duration_s, control/gripper mode, scene and
MJCF hashes.

Python 3.12, interpreter /home/laptop3/youngwoong_ws/lr_env/bin/python (lerobot 0.6.1).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Iterable, Sequence

import cv2
import h5py
import numpy as np

TASK = "Put carrot in pot"
FPS = 30
ROBOT_TYPE = "ur7e_gello"
DEFAULT_REPO_ID = "Bigenlight/carrot_in_pot_sim_lerobot_v3"
DEFAULT_DATA_ROOT = "/home/laptop3/gello_software/ros2_ur_ws/gello_logs/sim_retimed"

STATE_NAMES = ["ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6", "grip_pos"]
ACTION_NAMES = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd6", "grip_cmd"]

CAMS = ("cam1", "cam2")
RGB_SHAPE = (720, 1280, 3)

SIM_POSE_KEY = "observation.sim.object_poses"
SIM_OBJECTS = ("carrot", "pot")
SIM_POSE_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
SIM_POSE_NAMES = [f"{o}_{f}" for o in SIM_OBJECTS for f in SIM_POSE_FIELDS]  # 14 columns

LAYOUT_SEED_NOTE = (
    "The first sim_collect session (git 4bac865, 2026-09-15) recorded layout_seed = 0 for "
    "every take because of a since-fixed bug (f6697c3 'record the current layout seed per "
    "take'); the object placement WAS randomised per take. The ground truth is the "
    "sampled layout in sim_meta.scene_meta.layout and the first row of sim_object_poses, "
    "both recorded here as `layout` / `initial_object_xy_m`."
)


# --------------------------------------------------------------------------- #
# resampling helpers (identical to convert_carrot_to_lerobot.py)
# --------------------------------------------------------------------------- #
def nearest_idx(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """For each query timestamp, index of the nearest src sample (src_t sorted asc)."""
    j = np.searchsorted(src_t, query_t)
    j = np.clip(j, 1, len(src_t) - 1)
    left = src_t[j - 1]
    right = src_t[j]
    pick_left = (query_t - left) <= (right - query_t)
    out = np.where(pick_left, j - 1, j)
    return np.clip(out, 0, len(src_t) - 1)


def ffill_bfill(v: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaNs."""
    v = v.copy()
    n = len(v)
    last = np.nan
    for i in range(n):
        if np.isnan(v[i]):
            v[i] = last
        else:
            last = v[i]
    nxt = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(v[i]):
            v[i] = nxt
        else:
            nxt = v[i]
    return v


def load_lag_json(path: str) -> dict[str, float]:
    """take dir name -> tau seconds, from a per-take lag JSON (same shapes as the real converter)."""
    with open(path) as fh:
        obj = json.load(fh)
    if isinstance(obj, dict) and isinstance(obj.get("takes"), dict):
        obj = obj["takes"]
    if not isinstance(obj, dict):
        raise SystemExit(f"[!] {path}: expected a JSON object of take -> tau")
    out: dict[str, float] = {}
    for k, v in obj.items():
        if isinstance(v, dict) and "tau_q_s" in v:
            out[str(k)] = float(v["tau_q_s"])
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = float(v)
    if not out:
        raise SystemExit(f"[!] {path}: no take entries with a tau_q_s field")
    return out


def stale_tail_keep(cam1_t: np.ndarray, ur_t_raw: np.ndarray, ur_lag_s: float) -> int:
    """How many leading master frames still have a real ur sample after shifting (tau <= 0: all)."""
    if ur_lag_s <= 0:
        return len(cam1_t)
    return int(np.count_nonzero(np.asarray(cam1_t) + float(ur_lag_s) <= float(ur_t_raw[-1])))


def load_take_arrays(
    h5_path: str, ur_lag_s: float = 0.0, drop_stale_tail: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (cam1_t, cam2_t, state[N,7], action[N,7], n_dropped_stale_tail).

    Bit-identical to the real converter (same lookup, same dtype casts, same fill rule).
    """
    with h5py.File(h5_path, "r") as f:
        cam1_t = f["cam1_frames"]["t_rel_s"][:]
        cam2_t = f["cam2_frames"]["t_rel_s"][:]

        ur_t_raw = f["ur_joint_states"]["t_rel_s"][:]
        n_drop = 0
        if drop_stale_tail and ur_lag_s > 0:
            keep = stale_tail_keep(cam1_t, ur_t_raw, ur_lag_s)
            n_drop = len(cam1_t) - keep
            cam1_t = cam1_t[:keep]
        n = len(cam1_t)

        ur_t = ur_t_raw - float(ur_lag_s)
        ur_j = nearest_idx(ur_t, cam1_t)
        state = np.zeros((n, 7), dtype=np.float32)
        for k in range(6):
            state[:, k] = f["ur_joint_states"][f"q{k + 1}"][:][ur_j]
        grip_t = f["gripper"]["t_rel_s"][:]
        grip_j = nearest_idx(grip_t, cam1_t)
        state[:, 6] = f["gripper"]["grip_pos"][:][grip_j]

        cmd_t = f["command"]["t_rel_s"][:]
        cmd_j = nearest_idx(cmd_t, cam1_t)
        action = np.zeros((n, 7), dtype=np.float32)
        for k in range(6):
            action[:, k] = f["command"][f"cmd{k + 1}"][:][cmd_j]
        action[:, 6] = ffill_bfill(f["gripper"]["grip_cmd"][:])[grip_j]

    return cam1_t, cam2_t, state, action, n_drop


# --------------------------------------------------------------------------- #
# sim extras
# --------------------------------------------------------------------------- #
def read_sim_meta(h5_path: str) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        if "sim_meta" not in f.attrs:
            raise RuntimeError(f"{h5_path}: no `sim_meta` file attribute -- not a sim_collect take")
        return json.loads(str(f.attrs["sim_meta"]))


def load_sim_object_poses(h5_path: str, master_t: np.ndarray) -> np.ndarray:
    """(N, 14) float32: sim_object_poses columns SIM_POSE_NAMES at nearest t_rel_s to the master clock."""
    with h5py.File(h5_path, "r") as f:
        grp = f["sim_object_poses"]
        cols = json.loads(str(grp.attrs["columns"]))
        want = ["t_rel_s"] + SIM_POSE_NAMES
        if cols != want:
            raise RuntimeError(
                f"{h5_path}: sim_object_poses columns {cols} != expected {want} "
                "(different object set? this converter is for the carrot+pot scene)"
            )
        t = np.asarray(grp["t_rel_s"][:], dtype=np.float64)
        if len(t) == 0:
            raise RuntimeError(f"{h5_path}: sim_object_poses is empty")
        j = nearest_idx(t, np.asarray(master_t, dtype=np.float64))
        out = np.zeros((len(master_t), len(SIM_POSE_NAMES)), dtype=np.float32)
        for k, name in enumerate(SIM_POSE_NAMES):
            out[:, k] = grp[name][:][j]
        if not np.all(np.isfinite(out)):
            raise RuntimeError(f"{h5_path}: non-finite value in sim_object_poses")
    return out


def sim_provenance(meta: dict[str, Any], h5_path: str) -> dict[str, Any]:
    """The per-episode `sim` block of meta/source_takes.json, straight from sim_meta."""
    scene = meta.get("scene_meta") or {}
    with h5py.File(h5_path, "r") as f:
        grp = f["sim_object_poses"]
        first_xy = {o: [float(grp[f"{o}_x"][0]), float(grp[f"{o}_y"][0])] for o in SIM_OBJECTS}
        xml_sha = str(f["sim_scene"].attrs.get("xml_sha256", "")) if "sim_scene" in f else None
    return {
        "simulated": True,
        "take_name": meta.get("take_name"),
        "git_commit": meta.get("git_commit"),
        "sim_collect_version": meta.get("sim_collect_version"),
        "mujoco_version": meta.get("mujoco_version"),
        "robot": meta.get("robot"),
        "control_mode": meta.get("control_mode"),
        "gripper_mode": meta.get("gripper_mode"),
        "pos_scale": meta.get("pos_scale"),
        "layout_seed_recorded": meta.get("layout_seed"),
        "layout": scene.get("layout"),
        "initial_object_xy_m": first_xy,
        "task_success_at_stop": meta.get("task_success_at_stop"),
        "duration_s": meta.get("duration_s"),
        "started_at": meta.get("started_at"),
        "retimed": meta.get("retimed"),
        "original_achieved_fps": (meta.get("retimed") or {}).get("original_achieved_fps"),
        "problems": meta.get("problems") or [],
        "scene_sha": meta.get("scene_sha"),
        "xml_sha256": xml_sha,
        "record_depth": meta.get("record_depth"),
    }


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def open_reader(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    return cap


def build_features(with_sim_extras: bool) -> dict[str, dict[str, Any]]:
    feats: dict[str, dict[str, Any]] = {
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    }
    for cam in CAMS:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": RGB_SHAPE, "names": ["height", "width", "channels"],
        }
    if with_sim_extras:
        feats[SIM_POSE_KEY] = {
            "dtype": "float32", "shape": (len(SIM_POSE_NAMES),), "names": SIM_POSE_NAMES,
        }
    return feats


def dir_size_bytes(root: str) -> int:
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(root)
        for f in fs
        if not os.path.islink(os.path.join(dp, f))
    )


def convert(
    data_root: str,
    out_root: str,
    repo_id: str,
    limit: int | None = None,
    max_frames: int | None = None,
    image_writer_processes: int = 4,
    image_writer_threads: int = 2,
    exclude: Sequence[str] = (),
    ur_lag_s: float = 0.0,
    lag_json: str | None = None,
    drop_stale_tail: bool = True,
    with_sim_extras: bool = True,
    allow_unretimed: bool = False,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if os.path.exists(out_root):
        print(f"[!] output dir already exists: {out_root}", file=sys.stderr)
        print("    remove it first or pass a fresh --out", file=sys.stderr)
        sys.exit(1)

    all_takes = sorted(glob.glob(os.path.join(data_root, "take_*")))
    all_takes = [t for t in all_takes if os.path.isdir(t)]
    skipped = [t for t in all_takes if os.path.basename(t) in exclude]
    takes = [t for t in all_takes if os.path.basename(t) not in exclude]
    for t in skipped:
        print(f"[skip] excluded: {os.path.basename(t)}", flush=True)
    missing = sorted(set(exclude) - {os.path.basename(t) for t in skipped})
    if missing:
        print(f"[!] --exclude names not found in {data_root}: {missing}", file=sys.stderr)
        sys.exit(1)
    if limit:
        takes = takes[:limit]
    if not takes:
        print(f"[!] no takes found under {data_root}", file=sys.stderr)
        sys.exit(1)

    # --- sim gate: every take must be a retimed, problem-free sim_collect take ---
    metas: dict[str, dict[str, Any]] = {}
    refused: list[str] = []
    for tk in takes:
        name = os.path.basename(tk)
        m = read_sim_meta(os.path.join(tk, "vectors.h5"))
        metas[name] = m
        if not m.get("simulated", False):
            refused.append(f"{name}: sim_meta.simulated is not true")
        if not allow_unretimed:
            if not isinstance(m.get("retimed"), dict):
                refused.append(f"{name}: not retimed (no sim_meta.retimed; run "
                               "sim_collect/tools/retime_take.py or pass --allow-unretimed)")
            if m.get("problems"):
                refused.append(f"{name}: sim_meta.problems = {m['problems']}")
    if refused:
        print("[!] refusing to convert:", file=sys.stderr)
        for r in refused:
            print(f"    {r}", file=sys.stderr)
        sys.exit(1)
    commits = sorted({str(m.get("git_commit")) for m in metas.values()})
    mj_versions = sorted({str(m.get("mujoco_version")) for m in metas.values()})
    print(f"[cfg] sim takes: {len(takes)}  git_commit={commits}  mujoco={mj_versions}  "
          f"retimed={'required' if not allow_unretimed else 'not required'}", flush=True)
    if len(commits) > 1 or len(mj_versions) > 1:
        print("[!] WARNING: takes come from more than one code/mujoco version (recorded per episode)",
              file=sys.stderr)

    lag_map: dict[str, float] = {}
    if lag_json:
        lag_map = load_lag_json(lag_json)
        known = {os.path.basename(t) for t in all_takes}
        unknown = sorted(set(lag_map) - known)
        if unknown:
            print(f"[!] --lag-json has {len(unknown)} take names not in {data_root}: "
                  f"{unknown[:5]}", file=sys.stderr)
        missing_lag = [os.path.basename(t) for t in takes if os.path.basename(t) not in lag_map]
        if missing_lag:
            print(f"[!] --lag-json is missing {len(missing_lag)} converted takes "
                  f"(they fall back to --ur-lag-s={ur_lag_s}): {missing_lag[:5]}", file=sys.stderr)
    taus = {os.path.basename(t): float(lag_map.get(os.path.basename(t), ur_lag_s)) for t in takes}
    uniq_tau = sorted(set(taus.values()))
    print(f"[cfg] ur_joint_states lag correction: default {ur_lag_s} s"
          + (f", per-take from {lag_json} -> {uniq_tau} s" if lag_json else "")
          + "  (applied to ur_joint_states ONLY; sim rows are tick-synchronous, 0 is the "
            "intended value)", flush=True)
    any_lag = any(t > 0 for t in taus.values())
    stale_tail_mode = "dropped" if (drop_stale_tail and any_lag) else "kept"
    print(f"[cfg] stale tail (master frames with no ur sample at t+tau): {stale_tail_mode}",
          flush=True)
    print(f"[cfg] sim extras ({SIM_POSE_KEY}, 14 = carrot/pot xyz+quat_xyzw): "
          f"{'ON' if with_sim_extras else 'off'}", flush=True)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=build_features(with_sim_extras),
        root=out_root,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    source_takes: list[dict[str, Any]] = []
    total_frames = 0
    t_start = time.time()

    for ti, tk in enumerate(takes):
        t0 = time.time()
        name = os.path.basename(tk)
        tau = taus[name]
        h5_path = os.path.join(tk, "vectors.h5")
        cam1_t, cam2_t, state, action, n_drop = load_take_arrays(
            h5_path, ur_lag_s=tau, drop_stale_tail=drop_stale_tail)
        n = len(cam1_t)
        if max_frames:
            n = min(n, max_frames)
        cam2_map = nearest_idx(cam2_t, cam1_t)
        poses = load_sim_object_poses(h5_path, cam1_t[:n]) if with_sim_extras else None

        # cam1 is the master: mp4 frame k == master frame k, decoded sequentially.
        cap1 = open_reader(os.path.join(tk, "cam1.mp4"))
        cam1_frames: list[np.ndarray] = []
        n_short1 = 0
        for _ in range(n):
            ok, fr = cap1.read()
            if not ok:  # cam1 ran short: pad with the last frame
                n_short1 += 1
                fr = cam1_frames[-1][:, :, ::-1].copy() if cam1_frames else np.zeros(RGB_SHAPE, np.uint8)
                cam1_frames.append(fr[:, :, ::-1].copy())
                continue
            cam1_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap1.release()
        if n_short1:
            # A retimed take has frame count == rows by construction; a short cam1 means
            # the mp4 and the h5 do not belong together. Loud, not fatal (same as real).
            print(f"[!] {name}: cam1.mp4 is {n_short1} frame(s) short of cam1_frames "
                  f"({n} rows); padded with the last frame", file=sys.stderr)

        # cam2 is decoded whole, then indexed by nearest-timestamp map.
        cap2 = open_reader(os.path.join(tk, "cam2.mp4"))
        cam2_all: list[np.ndarray] = []
        while True:
            ok, fr = cap2.read()
            if not ok:
                break
            cam2_all.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap2.release()
        if not cam2_all:
            raise RuntimeError(f"no frames decoded from {tk}/cam2.mp4")
        cam2_map = np.clip(cam2_map, 0, len(cam2_all) - 1)

        for k in range(n):
            frame = {
                "action": action[k],
                "observation.state": state[k],
                "observation.images.cam1": cam1_frames[k],
                "observation.images.cam2": cam2_all[cam2_map[k]],
                "task": TASK,
            }
            if poses is not None:
                frame[SIM_POSE_KEY] = poses[k]
            ds.add_frame(frame)
        ds.save_episode()

        total_frames += n
        entry = {"episode_index": ti, "take_dir_name": name, "n_frames": n,
                 "excluded": False, "ur_joint_states_lag_s": tau,
                 "n_frames_dropped_stale_tail": int(n_drop),
                 "sim": sim_provenance(metas[name], h5_path)}
        source_takes.append(entry)
        rt = entry["sim"]["retimed"] or {}
        fps0 = entry["sim"]["original_achieved_fps"] or {}
        print(
            f"[{ti + 1:2d}/{len(takes)}] {name:32s} frames={n:4d}  cam2_decoded={len(cam2_all):4d}  "
            f"live_fps={fps0.get('cam1', '?')}/{fps0.get('cam2', '?')}  "
            f"state_dt_max={rt.get('max_state_lookup_dt_s', '?')}s  "
            f"success={entry['sim']['task_success_at_stop']}  {time.time() - t0:6.1f}s",
            flush=True,
        )

    ds.finalize()

    # --- sidecar AFTER finalize(), so lerobot's writer cannot clobber meta/ ---
    meta_dir = os.path.join(out_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    manifest_path = os.path.join(data_root, "retime_manifest.json")
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    with open(os.path.join(meta_dir, "source_takes.json"), "w") as fh:
        json.dump(
            {
                "task": TASK,
                "data_root": os.path.abspath(data_root),
                "simulation": {
                    "simulated": True,
                    "source": "sim_collect (MuJoCo UR7e + Robotiq 2F-85, physical GELLO leader, "
                              "same EEF bridge math and parameter files as the real ROS bridge)",
                    "git_commit": commits,
                    "mujoco_version": mj_versions,
                    "converter": os.path.basename(__file__),
                    "depth": "none (the sim takes were recorded without depth; no depth feature)",
                    "sim_extras_feature": (SIM_POSE_KEY if with_sim_extras else None),
                    "sim_extras_names": (SIM_POSE_NAMES if with_sim_extras else None),
                    "sim_extras_rule": (
                        "sim_object_poses row at nearest t_rel_s to the cam1 master clock "
                        "(same nearest-timestamp rule as every other stream); metres, world "
                        "frame, quaternion xyzw"
                    ),
                    "retimed_required": not allow_unretimed,
                    "retime_manifest": (os.path.abspath(manifest_path) if manifest else None),
                    "retime_note": (
                        "cam1/cam2 were re-rendered offline from the recorded MuJoCo state "
                        "(/sim_mj_state, 125 Hz) on an exact 1/30 s grid, so mp4 frame count == "
                        "cam*_frames rows and the master clock is uniformly spaced; the live "
                        "capture ran at 24-27 fps under CPU load (per-episode "
                        "sim.original_achieved_fps). State tables are verbatim."
                    ),
                    "layout_seed_note": LAYOUT_SEED_NOTE,
                },
                "timestamp_correction": {
                    "ur_joint_states_lag_s": (uniq_tau[0] if len(uniq_tau) == 1 else uniq_tau),
                    "why": ("none needed: sim_collect stamps every table from the physics tick "
                            "that produced it (tick-synchronous rows), so tau = 0 is the intended "
                            "value; the block is kept for schema parity with the real converter"),
                    "tcp_pose/wrench": "not in this dataset",
                    "applied_to": ["observation.state[0:6] (ur_joint_states q1..q6)"],
                    "not_applied_to": ["observation.state[6] (grip_pos)", "action (command + grip_cmd)",
                                       "observation.images.cam1", "observation.images.cam2",
                                       SIM_POSE_KEY],
                    "convention": ("ur_joint_states row clock = t_rel_s - tau before the "
                                   "nearest-timestamp lookup (master frame at t reads the ur row "
                                   "stamped t + tau)"),
                    "stale_tail": stale_tail_mode,
                    "stale_tail_rule": (
                        "a master frame k is kept iff cam1_t[k] + tau <= ur_joint_states "
                        "t_rel_s[-1]; at tau = 0 nothing is dropped. Per-episode count: "
                        "n_frames_dropped_stale_tail."
                    ),
                    "n_frames_dropped_stale_tail_total": int(
                        sum(e["n_frames_dropped_stale_tail"] for e in source_takes)),
                    "source": (os.path.abspath(lag_json) if lag_json else "--ur-lag-s"),
                    "per_episode": {e["take_dir_name"]: e["ur_joint_states_lag_s"]
                                    for e in source_takes},
                },
                "episodes": source_takes,
                "excluded": [{"take_dir_name": os.path.basename(t), "excluded": True} for t in skipped],
            },
            fh,
            indent=2,
        )

    wall = time.time() - t_start
    size_gb = dir_size_bytes(out_root) / 1e9
    print(
        f"\nDONE: {len(takes)} episodes, {total_frames} frames "
        f"(stale tail {stale_tail_mode}: "
        f"{sum(e['n_frames_dropped_stale_tail'] for e in source_takes)} frames), "
        f"{wall / 60:.1f} min wall, "
        f"{size_gb:.2f} GB -> {out_root}"
    )
    if max_frames:
        print(f"[!] --max-frames {max_frames} was set: this is a TEST dataset, not the release.")


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", default=DEFAULT_DATA_ROOT,
                    help="directory of RETIMED sim takes (take_*/{vectors.h5,cam1.mp4,cam2.mp4})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N takes")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="TEST ONLY: truncate every episode to N frames")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--exclude", action="append", default=[],
                    help="take folder name to skip (repeatable); errors if not found")
    ap.add_argument("--ur-lag-s", type=float, default=0.0,
                    help="seconds by which ur_joint_states rows are stamped late (real-recorder "
                         "artefact). Sim rows are tick-synchronous: the default 0 applies no shift")
    ap.add_argument("--lag-json", default=None,
                    help="per-take override for --ur-lag-s (same format as the real converter)")
    ap.add_argument("--drop-stale-tail", dest="drop_stale_tail", action="store_true", default=True,
                    help="drop trailing master frames without a ur sample after the shift "
                         "(a no-op at tau = 0)")
    ap.add_argument("--no-drop-stale-tail", dest="drop_stale_tail", action="store_false")
    ap.add_argument("--with-sim-extras", dest="with_sim_extras", action="store_true", default=True,
                    help=f"add {SIM_POSE_KEY} (float32 (14,)) from sim_object_poses (default ON)")
    ap.add_argument("--no-sim-extras", dest="with_sim_extras", action="store_false",
                    help="joint features + RGB only (schema identical to the real dataset minus depth)")
    ap.add_argument("--allow-unretimed", action="store_true",
                    help="accept takes without sim_meta.retimed / with sim_meta.problems (NOT for "
                         "takes whose live capture fell below 30 fps -- their mp4 plays fast)")
    args = ap.parse_args(list(argv) if argv is not None else None)
    convert(
        args.data, args.out, args.repo_id,
        limit=args.limit, max_frames=args.max_frames,
        image_writer_processes=args.procs, image_writer_threads=args.threads,
        exclude=tuple(args.exclude),
        ur_lag_s=args.ur_lag_s, lag_json=args.lag_json,
        drop_stale_tail=args.drop_stale_tail,
        with_sim_extras=args.with_sim_extras,
        allow_unretimed=args.allow_unretimed,
    )


if __name__ == "__main__":
    main()
