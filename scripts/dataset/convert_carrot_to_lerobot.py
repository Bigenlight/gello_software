#!/usr/bin/env python3
"""
Convert the "Put carrot in pot" gello/UR teleop dataset (vectors.h5 + cam*.mp4 +
depth.h5 per take) into a LeRobot v3.0 dataset WITH native depth video features.

Spec: scratchpad/FACTS_carrot_in_pot.md, section "LeRobot conversion spec".
Base recipe: convert_cube_to_lerobot.py -- state/action/RGB behaviour is kept
bit-for-bit (same master clock, same nearest-timestamp resampling, same decode
order), so an independent re-derivation must see max|delta| = 0.

Recipe
------
  - fps = 30 (camera rate). Master clock = cam1_frames/t_rel_s. Every other
    stream is resampled onto it by nearest-timestamp lookup on its own t_rel_s.
  - observation.state (7): ur_joint_states q1..q6 + gripper.grip_pos
  - action           (7): command cmd1..cmd6 + gripper.grip_cmd (ffill/bfill)
  - observation.images.cam1 / cam2:              720p RGB video (HWC uint8)
  - observation.images.cam1_depth / cam2_depth:  848x480 uint16 mm depth video,
    feature info {"is_depth_map": true}, encoded by DepthEncoderConfig(
    depth_min=0.0, depth_max=10.0, shift=0.0, use_log=False): HEVC gray12le,
    x265 lossless, 12-bit LINEAR quantization over 0-10 m (step 2.442 mm, so
    decoded values are within +-1.25 mm of the raw mm value, raw 0 decodes to
    exactly 0, nothing clips inside the D435 range; max raw seen 9,899 mm).
  - Depth is UNALIGNED to colour (native depth sensor frame). Each cam's depth
    frame is picked by nearest depth t_rel_s to the cam1 master timestamp, the
    same `nearest_idx` semantics used for cam2 RGB.
  - gello_* streams are intentionally ignored (not observable at inference).

Timestamp correction (--ur-lag-s / --lag-json)
----------------------------------------------
The GUI recorder stamps every row at callback-execution time on a single rclpy
spin thread that depth recording starved, so tables published faster than the
service rate sat in their subscription queue and were stamped LATE by
(QoS depth / publish rate). Measured on all 54 takes: `ur_joint_states` is
tau = 0.895-0.900 s late; `command`, `gripper`, the cameras and depth are fresh
(<= 0.022 s). ONLY `ur_joint_states` is corrected here: its row clock becomes
`t_rel_s - tau` before the nearest-timestamp lookup, i.e. the master frame at
time t reads the ur row stamped t + tau. grip_pos, action (command + grip_cmd),
cam2 and both depth streams are UNCHANGED -- shifting them would invent a
misalignment that is not there. The applied tau is recorded per episode in
meta/source_takes.json ("ur_joint_states_lag_s") together with a top-level
"timestamp_correction" block. tau = 0 reproduces the uncorrected v1 dataset.

Side files written after ds.finalize() (so lerobot's own writer cannot clobber
meta/): meta/depth_cameras.json (intrinsics/extrinsics/quantization) and
meta/source_takes.json (episode_index -> take dir).

Python 3.12, interpreter /home/laptop3/youngwoong_ws/lr_env/bin/python.
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
DEFAULT_REPO_ID = "Bigenlight/carrot_in_pot_lerobot_v3"

STATE_NAMES = ["ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6", "grip_pos"]
ACTION_NAMES = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd6", "grip_cmd"]

CAMS = ("cam1", "cam2")
RGB_SHAPE = (720, 1280, 3)
DEPTH_W, DEPTH_H = 848, 480
DEPTH_SHAPE = (DEPTH_H, DEPTH_W, 1)

INVALID_NOTE = (
    "Raw depth 0 = no return / invalid (RealSense D435, 848x480, unaligned to colour). "
    "Quantization is LINEAR 12-bit over 0-10.0 m (step 10000/4095 = 2.442 mm), so raw 0 "
    "decodes to exactly 0.0 mm (mask with decoded == 0) and every valid pixel decodes to "
    "within +-1.25 mm of the recorded millimetre value. Values above 10,000 mm would clip "
    "(none observed; max recorded 9,899 mm)."
)
# Chosen over lerobot's default log quantizer (depth_min 0.01 m) because that one maps
# raw 0 to 10 mm (loses the invalid mask) and is non-uniform.
DEPTH_ENCODER_KW = dict(depth_min=0.0, depth_max=10.0, shift=0.0, use_log=False)


# --------------------------------------------------------------------------- #
# resampling helpers (identical semantics to convert_cube_to_lerobot.py)
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
    """take dir name -> tau seconds, from a per-take lag JSON.

    Accepted shapes: {"takes": {name: {"tau_q_s": ...}}} or a bare
    {name: {"tau_q_s": ...}} / {name: <float>} map (non-take metadata keys whose
    value is neither a dict with tau_q_s nor a number are ignored).
    """
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


def load_take_arrays(
    h5_path: str, ur_lag_s: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (cam1_t, cam2_t, state[N,7], action[N,7]) on the cam1 timeline.

    `ur_lag_s` (tau) shifts ONLY the ur_joint_states row clock to t_rel_s - tau
    before the nearest-timestamp lookup. With tau > 0 the first master frames can
    query before the first (corrected) ur row; nearest_idx clamps to index 0, so
    they take the earliest available row -- no exception, no extrapolation.
    """
    with h5py.File(h5_path, "r") as f:
        cam1_t = f["cam1_frames"]["t_rel_s"][:]
        cam2_t = f["cam2_frames"]["t_rel_s"][:]
        n = len(cam1_t)

        ur_t = f["ur_joint_states"]["t_rel_s"][:] - float(ur_lag_s)
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

    return cam1_t, cam2_t, state, action


# --------------------------------------------------------------------------- #
# depth
# --------------------------------------------------------------------------- #
class DepthTake:
    """Lazy per-take depth reader: master timestamps -> (480,848,1) uint16 mm frames.

    Decoding is on demand with a 1-entry cache; the master->depth index map is
    monotone non-decreasing, so consecutive master frames that land on the same
    depth frame decode once.
    """

    def __init__(self, depth_h5: str, master_t: np.ndarray):
        self._f = h5py.File(depth_h5, "r")
        self.path = depth_h5
        self.index: dict[str, np.ndarray] = {}
        self.n_depth: dict[str, int] = {}
        for cam in CAMS:
            t = self._f[cam]["t_rel_s"][:]
            if len(t) == 0:
                raise RuntimeError(f"{depth_h5}: {cam} has no depth frames")
            self.index[cam] = nearest_idx(t, master_t)
            self.n_depth[cam] = int(self._f[cam]["png"].shape[0])
        self._cache: dict[str, tuple[int, np.ndarray]] = {}

    def frame(self, cam: str, k: int) -> np.ndarray:
        idx = int(self.index[cam][k])
        hit = self._cache.get(cam)
        if hit is not None and hit[0] == idx:
            return hit[1]
        png = np.asarray(self._f[cam]["png"][idx], dtype=np.uint8)
        img = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"{self.path}: {cam}[{idx}] PNG failed to decode")
        if img.shape != (DEPTH_H, DEPTH_W) or img.dtype != np.uint16:
            raise RuntimeError(
                f"{self.path}: {cam}[{idx}] depth PNG is {img.shape}/{img.dtype}, "
                f"expected ({DEPTH_H}, {DEPTH_W})/uint16"
            )
        out = img[..., None]
        self._cache[cam] = (idx, out)
        return out

    def meta(self, cam: str) -> dict[str, Any]:
        grp = self._f[cam]
        info = grp["camera_info"].attrs
        ext = grp["extrinsics_depth_to_color"].attrs
        return {
            "width": int(grp.attrs["width"]),
            "height": int(grp.attrs["height"]),
            "unit": str(grp.attrs["unit"]),
            "aligned_to_color": bool(grp.attrs["aligned_to_color"]),
            "source_topic": str(grp.attrs["source_topic"]),
            "camera_info": {
                "K": np.asarray(info["K"], dtype=float).reshape(9).tolist(),
                "D": np.asarray(info["D"], dtype=float).reshape(-1).tolist(),
                "distortion_model": str(info["distortion_model"]),
                "frame_id": str(info["frame_id"]),
            },
            "extrinsics_depth_to_color": {
                "rotation": np.asarray(ext["rotation"], dtype=float).reshape(9).tolist(),
                "rotation_layout": str(ext["layout"]),
                "translation_m": np.asarray(ext["translation"], dtype=float).reshape(3).tolist(),
            },
        }

    def close(self) -> None:
        self._f.close()


def _cam_meta_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """np.allclose on every numeric field, exact match on the string fields."""
    if (a["width"], a["height"], a["unit"], a["aligned_to_color"], a["source_topic"]) != (
        b["width"], b["height"], b["unit"], b["aligned_to_color"], b["source_topic"]
    ):
        return False
    ai, bi = a["camera_info"], b["camera_info"]
    if (ai["distortion_model"], ai["frame_id"]) != (bi["distortion_model"], bi["frame_id"]):
        return False
    if len(ai["D"]) != len(bi["D"]):
        return False
    ae, be = a["extrinsics_depth_to_color"], b["extrinsics_depth_to_color"]
    if ae["rotation_layout"] != be["rotation_layout"]:
        return False
    for x, y in ((ai["K"], bi["K"]), (ai["D"], bi["D"]),
                 (ae["rotation"], be["rotation"]), (ae["translation_m"], be["translation_m"])):
        if not np.allclose(np.asarray(x), np.asarray(y), atol=1e-9, rtol=0.0):
            return False
    return True


def build_depth_sidecar(
    per_take_meta: list[tuple[str, dict[str, dict[str, Any]]]],
    depth_encoder: Any,
) -> dict[str, Any]:
    """Collapse per-take camera metadata into meta/depth_cameras.json.

    Identical across takes (np.allclose atol 1e-9) -> one block per camera.
    Otherwise the first take's block is kept AND a "per_episode" list is added
    (never a silent pick), with a loud warning printed by the caller.
    """
    quant = {
        "depth_min_m": float(depth_encoder.depth_min),
        "depth_max_m": float(depth_encoder.depth_max),
        "shift_m": float(depth_encoder.shift),
        "use_log": bool(depth_encoder.use_log),
        "quant_bits": 12,
        "vcodec": str(depth_encoder.vcodec),
        "pix_fmt": str(depth_encoder.pix_fmt),
        "extra_options": dict(depth_encoder.extra_options),
    }
    sidecar: dict[str, Any] = {"invalid_value_note": INVALID_NOTE, "quantization": quant}
    mismatched: list[str] = []
    for cam in CAMS:
        key = f"{cam}_depth"
        first = per_take_meta[0][1][cam]
        block = {"color_feature": f"observation.images.{cam}", **first}
        bad = [i for i, (_, m) in enumerate(per_take_meta) if not _cam_meta_equal(first, m[cam])]
        if bad:
            mismatched.append(cam)
            block["per_episode"] = [
                {"episode_index": i, "take": name, **per_take_meta[i][1][cam]}
                for i, (name, _) in enumerate(per_take_meta)
            ]
            block["per_episode_note"] = (
                "camera_info/extrinsics are NOT identical across takes; the top-level block "
                f"is take {per_take_meta[0][0]} and every episode is listed under per_episode. "
                f"Differing episodes: {bad}"
            )
        sidecar[key] = block
    sidecar["_mismatched_cameras"] = mismatched
    return sidecar


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def open_reader(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    return cap


def build_features() -> dict[str, dict[str, Any]]:
    feats: dict[str, dict[str, Any]] = {
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    }
    for cam in CAMS:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": RGB_SHAPE, "names": ["height", "width", "channels"],
        }
    for cam in CAMS:
        feats[f"observation.images.{cam}_depth"] = {
            "dtype": "video", "shape": DEPTH_SHAPE, "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
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
) -> None:
    from lerobot.configs import DepthEncoderConfig
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
          + "  (applied to ur_joint_states ONLY)", flush=True)

    depth_encoder = DepthEncoderConfig(**DEPTH_ENCODER_KW)
    print(f"[cfg] depth encoder: {depth_encoder}", flush=True)
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=build_features(),
        root=out_root,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        depth_encoder=depth_encoder,
    )

    per_take_meta: list[tuple[str, dict[str, dict[str, Any]]]] = []
    source_takes: list[dict[str, Any]] = []
    total_frames = 0
    t_start = time.time()

    for ti, tk in enumerate(takes):
        t0 = time.time()
        name = os.path.basename(tk)
        tau = taus[name]
        cam1_t, cam2_t, state, action = load_take_arrays(
            os.path.join(tk, "vectors.h5"), ur_lag_s=tau)
        n = len(cam1_t)
        if max_frames:
            n = min(n, max_frames)
        cam2_map = nearest_idx(cam2_t, cam1_t)

        depth = DepthTake(os.path.join(tk, "depth.h5"), cam1_t[:n])
        per_take_meta.append((name, {cam: depth.meta(cam) for cam in CAMS}))

        # cam1 is the master: mp4 frame k == master frame k, decoded sequentially.
        cap1 = open_reader(os.path.join(tk, "cam1.mp4"))
        cam1_frames: list[np.ndarray] = []
        for _ in range(n):
            ok, fr = cap1.read()
            if not ok:  # cam1 ran short: pad with the last frame
                fr = cam1_frames[-1][:, :, ::-1].copy() if cam1_frames else np.zeros(RGB_SHAPE, np.uint8)
                cam1_frames.append(fr[:, :, ::-1].copy())
                continue
            cam1_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap1.release()

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
            ds.add_frame({
                "action": action[k],
                "observation.state": state[k],
                "observation.images.cam1": cam1_frames[k],
                "observation.images.cam2": cam2_all[cam2_map[k]],
                "observation.images.cam1_depth": depth.frame("cam1", k),
                "observation.images.cam2_depth": depth.frame("cam2", k),
                "task": TASK,
            })
        ds.save_episode()
        matched = {cam: len(np.unique(depth.index[cam][:n])) for cam in CAMS}
        n_depth = dict(depth.n_depth)
        depth.close()

        total_frames += n
        source_takes.append({"episode_index": ti, "take_dir_name": name, "n_frames": n,
                             "excluded": False, "ur_joint_states_lag_s": tau})
        print(
            f"[{ti + 1:2d}/{len(takes)}] {name:32s} frames={n:4d}  cam2_decoded={len(cam2_all):4d}  "
            f"depth_in_h5={n_depth['cam1']}/{n_depth['cam2']}  depth_matched={matched['cam1']}/{matched['cam2']}  "
            f"{time.time() - t0:6.1f}s",
            flush=True,
        )

    ds.finalize()

    # --- sidecars AFTER finalize(), so lerobot's writer cannot clobber meta/ ---
    meta_dir = os.path.join(out_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    sidecar = build_depth_sidecar(per_take_meta, depth_encoder)
    mismatched = sidecar.pop("_mismatched_cameras")
    if mismatched:
        print(
            f"\n[!] WARNING: camera_info/extrinsics differ across takes for {mismatched}; "
            "meta/depth_cameras.json carries a per_episode mapping.",
            file=sys.stderr,
        )
    with open(os.path.join(meta_dir, "depth_cameras.json"), "w") as fh:
        json.dump(sidecar, fh, indent=2)
    with open(os.path.join(meta_dir, "source_takes.json"), "w") as fh:
        json.dump(
            {
                "task": TASK,
                "data_root": os.path.abspath(data_root),
                "timestamp_correction": {
                    "ur_joint_states_lag_s": (uniq_tau[0] if len(uniq_tau) == 1 else uniq_tau),
                    "why": ("recorder spin-thread starvation; rows stamped at callback time "
                            "were depth/rate late"),
                    "tcp_pose/wrench": "not in this dataset",
                    "applied_to": ["observation.state[0:6] (ur_joint_states q1..q6)"],
                    "not_applied_to": ["observation.state[6] (grip_pos)", "action (command + grip_cmd)",
                                       "observation.images.cam1", "observation.images.cam2",
                                       "observation.images.cam1_depth", "observation.images.cam2_depth"],
                    "convention": ("ur_joint_states row clock = t_rel_s - tau before the "
                                   "nearest-timestamp lookup (master frame at t reads the ur row "
                                   "stamped t + tau)"),
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
        f"\nDONE: {len(takes)} episodes, {total_frames} frames, {wall / 60:.1f} min wall, "
        f"{size_gb:.2f} GB -> {out_root}"
    )
    if max_frames:
        print(f"[!] --max-frames {max_frames} was set: this is a TEST dataset, not the release.")


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", default="Put_carrot_in_pot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N takes")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="TEST ONLY: truncate every episode to N frames")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--exclude", action="append", default=[],
                    help="take folder name to skip (repeatable); errors if not found")
    ap.add_argument("--ur-lag-s", type=float, default=0.9,
                    help="seconds by which ur_joint_states rows are stamped late; its row clock "
                         "becomes t_rel_s - tau. Applied to ur_joint_states ONLY. 0 = no correction")
    ap.add_argument("--lag-json", default=None,
                    help="per-take override for --ur-lag-s: JSON of take dir name -> "
                         "{\"tau_q_s\": float} (or a bare float); wins over --ur-lag-s")
    args = ap.parse_args(list(argv) if argv is not None else None)
    convert(
        args.data, args.out, args.repo_id,
        limit=args.limit, max_frames=args.max_frames,
        image_writer_processes=args.procs, image_writer_threads=args.threads,
        exclude=tuple(args.exclude),
        ur_lag_s=args.ur_lag_s, lag_json=args.lag_json,
    )


if __name__ == "__main__":
    main()
