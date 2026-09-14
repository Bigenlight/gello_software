#!/usr/bin/env python3
"""
Offline validator: "Put carrot in pot" LeRobot v3.0 dataset  vs  raw HDF5/mp4/depth takes.

Verifies that `carrot_in_pot_lerobot` is a faithful rendering of the raw takes in
`Put_carrot_in_pot/take_*/{vectors.h5,cam1.mp4,cam2.mp4,depth.h5}`.

INDEPENDENCE
------------
This script deliberately does NOT import (or read) the converter. Every expected
value is re-implemented here from the written spec, so a disagreement between the
converter and this file surfaces as a FAIL instead of cancelling out.

The one exception is *decoding*: LeRobot's own reader is used to get the depth
videos back (HEVC gray12le 12-bit log-quantized codes -> float32 mm), because the
dequantization contract belongs to lerobot. lerobot is used ONLY to read pixels;
never to re-derive an expected value.

INTERPRETER
-----------
Run this under the lerobot venv:

    /home/laptop3/youngwoong_ws/lr_env/bin/python

(The cube validator ran under act_venv, which has no lerobot. THIS validator needs
lerobot for checks 9/10 -- the pyav depth decode. Every other check is
lerobot-free: run with --skip-depth-decode under any interpreter that has
numpy + h5py + pyarrow/pandas + cv2 and has ffmpeg on PATH.)

The Hugging Face Hub is never contacted: HF_HUB_OFFLINE / HF_DATASETS_OFFLINE are
forced on before lerobot is imported.

SPEC (re-implemented here)
--------------------------
  master clock  = cam1_frames/t_rel_s          (fps = 30)
  observation.state[7] = ur_joint_states q1..q6 @ nearest (ur t_rel_s - tau)
                       + gripper/grip_pos      @ nearest gripper t_rel_s
  action[7]            = command cmd1..cmd6    @ nearest command t_rel_s
                       + ffill_bfill(gripper/grip_cmd) @ nearest gripper t_rel_s
  cam1 frame k         = k-th decoded frame of cam1.mp4 (last frame padded if short)
  cam2 frame k         = decoded frame nearest_idx(cam2_t, cam1_t)[k] of cam2.mp4
  cam<N>_depth frame k = PNG nearest_idx(depth.h5/cam<N>/t_rel_s, cam1_t)[k],
                         uint16 mm, 848x480, unaligned, 0 = no return
  gello_* streams      = dropped

  tau = the recorder timestamp correction (--ur-lag-s, default 0.9 s; per-take
  --lag-json wins). The GUI recorder stamped `ur_joint_states` rows tau seconds
  LATE (spin-thread starvation: queue age = QoS depth / publish rate), so the
  converter shifts THAT TABLE ONLY to t_rel_s - tau before the nearest-timestamp
  lookup. grip_pos, action (command + grip_cmd), cam2 and both depth streams are
  untouched; this validator re-derives with the same rule from the same CLI value
  (never from the converter) so check 3 must still see max|delta| = 0.

Checks (each prints PASS / FAIL / WARN / SKIP):
  0  self-test of this file's nearest-timestamp + ffill implementations
  1  meta/info.json    codebase_version, fps, features, totals
  2  per-episode frame counts vs source cam1_frames row counts (in take order)
  3  numeric fidelity of observation.state / action  (max|diff| < --tol)
  4  no NaN / Inf in stored state & action
  5  task string: exactly one, == --task, every task_index resolves to it
  6  RGB video integrity: sampled frames, LeRobot AV1 vs raw MPEG-4 (correlation)
  7  sanity: episode boundaries monotonic/contiguous, timestamp == frame_index/fps
  8  info.json depth feature dicts (is_depth_map, unit, codec, quantization params)
     and RGB feature dicts unchanged
  9  per-episode depth frame counts == episode length, per depth video file too
 10  depth numeric fidelity: sampled decoded frames vs depth.h5 ground truth
 11  meta/depth_cameras.json vs depth.h5 camera_info/extrinsics of EVERY take
 12  meta/source_takes.json: take order and per-take n_frames
 13  depth timestamp sanity: |t_depth - t_cam1|, two tiers -- WARN at the agreed
     half-frame + 5 ms, FAIL at one full frame period. See the DEPTH_DT_* comment:
     the agreed number is not reachable on this recording (cam1's depth topic runs a
     median 8.6 ms ahead of its colour topic), and this check reads raw clocks only,
     so it reports a recording property and can never catch a converter error.
 14  meta/source_takes.json carries ur_joint_states_lag_s == the CLI/JSON tau for
     every episode, and a top-level timestamp_correction block
 15  physics sanity: mean |observation.state[0:6] - action[0:6]| per episode
     (first 1.6 s excluded) is < 0.02 rad WITH the correction, and larger when
     re-derived at tau = 0 -- the follower tracks its own command, so a correct
     time base is the only way this gets small

Usage
-----
  PY=/home/laptop3/youngwoong_ws/lr_env/bin/python
  $PY /home/laptop3/youngwoong_ws/validate_carrot_conversion.py \
      --raw     /home/laptop3/youngwoong_ws/Put_carrot_in_pot \
      --lerobot /home/laptop3/youngwoong_ws/carrot_in_pot_lerobot \
      --json    /tmp/carrot_validation.json

TEST-ONLY KNOBS (`--max-frames`, `--max-takes`)
----------------------------------------------
A small smoke dataset built from the first few takes truncated to a handful of
frames each would legitimately FAIL check 2 (episode length != full take length)
and the episode-count check. `--max-frames N` makes the validator mirror that
truncation -- it clips each take's master clock to its first N samples -- and
`--max-takes N` clips the take list to the first N takes. Both exist ONLY so a
truncated test dataset can be validated end to end. NEVER pass either when
validating the real release: they weaken checks 2, 3, 6, 9, 10, 12 and 13 to the
truncated prefix, and the printed header/JSON record that they were used.

Exit code 0 = all hard checks passed, 1 = at least one FAIL, 2 = could not run.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import traceback

import numpy as np

# Never contact the Hub. Set before lerobot / huggingface_hub is imported anywhere.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# --------------------------------------------------------------------------
# expected schema
# --------------------------------------------------------------------------
STATE_NAMES = ["ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6", "grip_pos"]
ACTION_NAMES = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd6", "grip_cmd"]

HWC = ["height", "width", "channels"]

EXPECTED_FEATURES = {
    "action": ("float32", [7], ACTION_NAMES),
    "observation.state": ("float32", [7], STATE_NAMES),
    "observation.images.cam1": ("video", [720, 1280, 3], HWC),
    "observation.images.cam2": ("video", [720, 1280, 3], HWC),
    "observation.images.cam1_depth": ("video", [480, 848, 1], HWC),
    "observation.images.cam2_depth": ("video", [480, 848, 1], HWC),
    "timestamp": ("float32", [1], None),
    "frame_index": ("int64", [1], None),
    "episode_index": ("int64", [1], None),
    "index": ("int64", [1], None),
    "task_index": ("int64", [1], None),
}

RGB_KEYS = ["observation.images.cam1", "observation.images.cam2"]
DEPTH_KEYS = ["observation.images.cam1_depth", "observation.images.cam2_depth"]
# LeRobot depth feature key  ->  depth.h5 group name
DEPTH_KEY_TO_CAM = {
    "observation.images.cam1_depth": "cam1",
    "observation.images.cam2_depth": "cam2",
}
# meta/depth_cameras.json key -> depth.h5 group name
SIDECAR_KEY_TO_CAM = {"cam1_depth": "cam1", "cam2_depth": "cam2"}

EXPECTED_DEPTH_INFO = {
    "is_depth_map": True,
    "depth_unit": "mm",
    "video.codec": "hevc",
    "video.pix_fmt": "gray12le",
    "video.depth_min": 0.0,
    "video.depth_max": 10.0,
    "video.shift": 0.0,
    "video.use_log": False,
    "video.height": 480,
    "video.width": 848,
    "video.channels": 1,
}

FPS = 30
# Check 13 is a two-tier check, and the tiers are NOT arbitrary:
#
#   soft = half a frame + 5 ms (the agreed spec's number)  -> WARN when violated
#   hard = one full frame period                           -> FAIL when violated
#
# Why two tiers. Check 13 is computed from RAW clocks only (depth.h5 t_rel_s vs
# cam1_frames t_rel_s), so it can never detect a converter error -- it measures a
# property of the recording. Measured over all 54 raw takes (2026-09-14):
#   cam1 depth runs ~8.6 ms EARLIER than colour (median signed offset), cam2 ~7.2 ms
#   later; both streams are independently sampled 30 Hz topics and the depth timeline
#   has occasional 50-70 ms gaps (dropped frames).
#   fraction within the soft tier : cam1 as low as 77.4 %, cam2 as low as 97.3 %
#   fraction within the hard tier : cam1 >= 99.62 %, cam2 >= 99.31 %   (worst take)
#   worst single |dt| : cam1 44.5 ms, cam2 51.5 ms (both next to a dropped frame)
# So the agreed soft number is not reachable by ANY converter on this recording, and
# failing on it would only report "the two camera topics are not phase-locked".
# The hard tier ("the chosen depth frame is at most one frame away") is the statement
# that actually carries meaning, and it holds with margin. Override with
# --depth-dt-tol-ms / --depth-dt-hard-tol-ms / --depth-dt-frac.
DEPTH_DT_TOL_S = (1.0 / FPS) / 2.0 + 0.005   # soft tier: half a frame + 5 ms
DEPTH_DT_HARD_TOL_S = 1.0 / FPS              # hard tier: one frame period
DEPTH_DT_FRAC = 0.99                         # ... for >= 99 % of frames
# Linear 12-bit quantization over 0-10 m: step 10000/4095 = 2.442 mm -> |err| <= 1.221 mm.
DEPTH_ZERO_DECODE_MAX_MM = 0.0               # raw 0 must decode to exactly 0
DEPTH_ZERO_FRAC = 1.0                        # ... for 100 % of them
DEPTH_ERR_MED_MM = 1.0                       # median |err| of a frame (decoded mm are integer-valued, so 0 or 1)
DEPTH_ERR_P99_MM = 1.25
DEPTH_ERR_MAX_MM = 1.25


# CHECK 15 gates: with a correct time base the UR follows its own command to well
# under 0.02 rad per joint; at tau = 0 the 0.9 s stamp error alone puts ~0.05 rad
# of pure delay between them (measured median 0.063 rad over the 54 takes).
STATE_ACTION_MAX_RAD = 0.02
STATE_ACTION_SKIP_HEAD_S = 1.6


# --------------------------------------------------------------------------
# report plumbing
# --------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.rows = []  # (status, name, detail)

    def _add(self, status, name, detail):
        self.rows.append((status, name, detail))
        tag = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP"}[status]
        print(f"  [{tag}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)

    def check(self, cond, name, detail=""):
        self._add("PASS" if cond else "FAIL", name, detail)
        return bool(cond)

    def pass_(self, name, detail=""):
        self._add("PASS", name, detail)

    def fail(self, name, detail=""):
        self._add("FAIL", name, detail)

    def warn(self, name, detail=""):
        self._add("WARN", name, detail)

    def skip(self, name, detail=""):
        self._add("SKIP", name, detail)

    @property
    def n_fail(self):
        return sum(1 for s, _, _ in self.rows if s == "FAIL")

    @property
    def n_warn(self):
        return sum(1 for s, _, _ in self.rows if s == "WARN")

    @property
    def n_pass(self):
        return sum(1 for s, _, _ in self.rows if s == "PASS")

    @property
    def n_skip(self):
        return sum(1 for s, _, _ in self.rows if s == "SKIP")


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78, flush=True)


# --------------------------------------------------------------------------
# resampling -- independent re-implementation of the converter's spec
# --------------------------------------------------------------------------
def nearest_idx(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """
    For each query timestamp return the index of the nearest sample in `src_t`
    (assumed sorted ascending). Ties resolve to the earlier (left) sample.
    Queries outside [src_t[0], src_t[-1]] clamp to the first/last sample.

    Written from the spec, not copied from the converter.
    """
    src_t = np.asarray(src_t, dtype=np.float64)
    query_t = np.asarray(query_t, dtype=np.float64)
    n = len(src_t)
    if n == 0:
        raise ValueError("empty source timeline")
    if n == 1:
        return np.zeros(len(query_t), dtype=np.int64)

    hi = np.searchsorted(src_t, query_t, side="left")
    hi = np.clip(hi, 1, n - 1)
    lo = hi - 1
    d_lo = query_t - src_t[lo]
    d_hi = src_t[hi] - query_t
    out = np.where(d_lo <= d_hi, lo, hi)
    return out.astype(np.int64)


def nearest_idx_bruteforce(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """O(N*M) reference used to self-verify `nearest_idx` on a small sample."""
    src_t = np.asarray(src_t, dtype=np.float64)
    out = np.empty(len(query_t), dtype=np.int64)
    for i, q in enumerate(query_t):
        d = np.abs(src_t - q)
        out[i] = int(np.argmin(d))  # argmin already prefers the earliest on ties
    return out


def ffill_bfill(v: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaNs. Vectorised re-implementation."""
    v = np.asarray(v, dtype=np.float64).copy()
    n = len(v)
    if n == 0:
        return v
    good = ~np.isnan(v)
    if not good.any():
        return v  # all-NaN: nothing to do (will be caught by the NaN check)
    idx = np.where(good, np.arange(n), -1)
    fwd = np.maximum.accumulate(idx)
    idx_b = np.where(good, np.arange(n), n)
    bwd = np.minimum.accumulate(idx_b[::-1])[::-1]
    src = np.where(fwd >= 0, fwd, bwd)
    src = np.clip(src, 0, n - 1)
    return v[src]


def load_lag_json(path: str) -> dict:
    """take dir name -> tau seconds. Written from the spec, not from the converter.

    Accepts {"takes": {name: {"tau_q_s": x}}}, {name: {"tau_q_s": x}} or
    {name: x}; metadata keys whose value is neither are ignored.
    """
    with open(path) as fh:
        obj = json.load(fh)
    if isinstance(obj, dict) and isinstance(obj.get("takes"), dict):
        obj = obj["takes"]
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object of take -> tau")
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict) and "tau_q_s" in v:
            out[str(k)] = float(v["tau_q_s"])
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = float(v)
    if not out:
        raise ValueError(f"{path}: no take entries with a tau_q_s field")
    return out


def resolve_taus(takes, ur_lag_s: float, lag_json: str | None):
    """take basename -> tau actually expected, CLI default overridden by the JSON."""
    lag_map = load_lag_json(lag_json) if lag_json else {}
    return {os.path.basename(t): float(lag_map.get(os.path.basename(t), ur_lag_s))
            for t in takes}


def derive_take(h5_path: str, max_frames: int = 0, ur_lag_s: float = 0.0):
    """Return (cam1_t, cam2_t, state[N,7] float32, action[N,7] float32).

    `ur_lag_s` is subtracted from the ur_joint_states row clock before the
    nearest-timestamp lookup (and from nothing else).
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        cam1_t = f["cam1_frames"]["t_rel_s"][:]
        cam2_t = f["cam2_frames"]["t_rel_s"][:]
        if max_frames and max_frames > 0:
            cam1_t = cam1_t[:max_frames]
        n = len(cam1_t)

        ur_t = np.asarray(f["ur_joint_states"]["t_rel_s"][:], dtype=np.float64) - float(ur_lag_s)
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


def take_cam1_t(h5_path: str, max_frames: int = 0) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as f:
        t = f["cam1_frames"]["t_rel_s"][:]
    return t[:max_frames] if (max_frames and max_frames > 0) else t


# --------------------------------------------------------------------------
# raw depth reading (h5py + cv2, no ROS, no gello_recorder import)
# --------------------------------------------------------------------------
def depth_timeline(depth_h5: str, cam: str) -> np.ndarray:
    import h5py

    with h5py.File(depth_h5, "r") as f:
        return f[cam]["t_rel_s"][:]


def depth_group_meta(depth_h5: str, cam: str) -> dict:
    """Group attrs + camera_info + extrinsics as plain python, read directly."""
    import h5py

    def _plain(v):
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        return v

    with h5py.File(depth_h5, "r") as f:
        grp = f[cam]
        out = {k: _plain(v) for k, v in grp.attrs.items()}
        out["n_frames"] = int(grp["png"].shape[0])
        for sub in ("camera_info", "extrinsics_depth_to_color"):
            if sub in grp:
                out[sub] = {k: _plain(v) for k, v in grp[sub].attrs.items()}
    return out


def read_raw_depth_frames(depth_h5: str, cam: str, indices):
    """{idx: uint16 (H, W) mm} decoded straight from the stored PNG bytes."""
    import cv2
    import h5py

    want = sorted(set(int(i) for i in indices))
    out = {}
    with h5py.File(depth_h5, "r") as f:
        ds = f[cam]["png"]
        n = ds.shape[0]
        for i in want:
            if i < 0 or i >= n:
                raise IndexError(f"{depth_h5}:{cam} has {n} frames, asked for {i}")
            arr = cv2.imdecode(np.asarray(ds[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"cv2.imdecode failed on {depth_h5}:{cam}[{i}]")
            out[i] = arr
    return out


# --------------------------------------------------------------------------
# parquet reading (pyarrow preferred, pandas fallback)
# --------------------------------------------------------------------------
def _read_parquet(path: str):
    """Return a dict {column_name: numpy array or list}."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None

    if pq is not None:
        table = pq.read_table(path)
        return {name: table.column(name) for name in table.column_names}, "pyarrow"

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Neither pyarrow nor pandas is available in this interpreter; "
            "cannot read LeRobot parquet files."
        ) from exc
    df = pd.read_parquet(path)
    if df.index.name is not None and df.index.name not in df.columns:
        df = df.reset_index()
    return {c: df[c] for c in df.columns}, "pandas"


def _col_to_numpy(col, dtype=None):
    try:
        import pyarrow as pa

        if isinstance(col, (pa.ChunkedArray, pa.Array)):
            arr = np.asarray(col.to_pylist())
            return arr.astype(dtype) if dtype else arr
    except ImportError:
        pass
    arr = np.asarray(list(col))
    return arr.astype(dtype) if dtype else arr


def _col_to_2d(col, width, dtype=np.float32):
    try:
        import pyarrow as pa

        if isinstance(col, (pa.ChunkedArray, pa.Array)):
            vals = col.to_pylist()
        else:
            vals = list(col)
    except ImportError:
        vals = list(col)
    out = np.empty((len(vals), width), dtype=dtype)
    for i, v in enumerate(vals):
        out[i, :] = np.asarray(v, dtype=dtype)
    return out


def _col_to_pylist(col):
    try:
        import pyarrow as pa

        if isinstance(col, (pa.ChunkedArray, pa.Array)):
            return col.to_pylist()
    except ImportError:
        pass
    return list(col)


def _sorted_chunk_files(root: str, subdir: str, ext: str):
    pat = os.path.join(root, subdir, "chunk-*", f"file-*{ext}")
    return sorted(glob.glob(pat))


def load_data_frames(root: str, info: dict):
    """Concatenate all data/**/*.parquet in (chunk, file) order."""
    files = _sorted_chunk_files(root, "data", ".parquet")
    if not files:
        raise RuntimeError(f"no data parquet files under {os.path.join(root, 'data')}")
    cols = {}
    backend = None
    for fp in files:
        d, backend = _read_parquet(fp)
        for k, v in d.items():
            cols.setdefault(k, []).append(v)
    merged = {}
    for k, parts in cols.items():
        if k in ("action", "observation.state"):
            merged[k] = np.concatenate([_col_to_2d(p, 7) for p in parts], axis=0)
        else:
            merged[k] = np.concatenate([_col_to_numpy(p) for p in parts], axis=0)
    return merged, files, backend


def load_episodes_meta(root: str):
    files = _sorted_chunk_files(root, "meta/episodes", ".parquet")
    if not files:
        raise RuntimeError(f"no episode parquet files under {os.path.join(root, 'meta/episodes')}")
    rows = []
    for fp in files:
        d, _ = _read_parquet(fp)
        n = None
        pylists = {}
        for k, v in d.items():
            pylists[k] = _col_to_pylist(v)
            n = len(pylists[k]) if n is None else n
        for i in range(n):
            row = {}
            for k in pylists:
                v = pylists[k][i]
                if k != "tasks" and isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
                    v = v[0]
                row[k] = v
            rows.append(row)
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def load_tasks(root: str):
    """Return {task_index: task_string}."""
    path = os.path.join(root, "meta", "tasks.parquet")
    d, _ = _read_parquet(path)
    names = list(d.keys())
    task_col = None
    for cand in ("task", "__index_level_0__"):
        if cand in names:
            task_col = cand
            break
    if task_col is None:
        for k in names:
            if k != "task_index":
                task_col = k
                break
    if task_col is None or "task_index" not in names:
        raise RuntimeError(f"unexpected tasks.parquet schema: {names}")
    tasks = _col_to_pylist(d[task_col])
    idxs = _col_to_pylist(d["task_index"])
    return {int(i): str(t) for i, t in zip(idxs, tasks)}


# --------------------------------------------------------------------------
# video helpers
# --------------------------------------------------------------------------
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def ffprobe_nb_frames(video_path: str) -> int:
    """Frame count of a video's first video stream; -1 if unavailable."""
    if FFPROBE is None:
        return -1
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1", video_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout.strip()
        if out and out != "N/A":
            return int(out)
    except Exception:
        pass
    # fall back to counting packets (slower but always works)
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries", "stream=nb_read_packets", "-of", "default=nw=1:nk=1", video_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900).stdout.strip()
        if out and out != "N/A":
            return int(out)
    except Exception:
        pass
    return -1


def decode_lerobot_frames(video_path: str, wanted, height: int, width: int):
    """
    Decode the requested *decode-order* frame indices from a LeRobot (AV1) video.

    Deliberately PTS-independent: stream the whole file through ffmpeg as rawvideo
    and count frames. The converter appended frames in strict decode order, so
    counting is the most faithful mapping.
    """
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot decode LeRobot AV1 video")
    wanted = sorted(set(int(i) for i in wanted))
    if not wanted:
        return {}, 0
    stop = wanted[-1]
    need = height * width * 3
    cmd = [
        FFMPEG, "-nostdin", "-v", "error",
        "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-frames:v", str(stop + 1),
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=need)
    want = set(wanted)
    out = {}
    i = 0
    try:
        while i <= stop:
            buf = proc.stdout.read(need)
            if not buf or len(buf) < need:
                break
            if i in want:
                out[i] = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy()
            i += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        proc.wait()
    if len(out) != len(wanted):
        missing = [k for k in wanted if k not in out]
        raise RuntimeError(
            f"decoded only {i} frames from {video_path}; missing indices {missing[:5]}"
            + (f"; ffmpeg: {err[:200]}" if err else "")
        )
    return out, i


def decode_raw_frames(mp4_path: str, wanted_indices):
    """Sequentially decode `mp4_path` -> {idx: RGB frame}, plus frames decoded."""
    import cv2

    wanted = sorted(set(int(i) for i in wanted_indices))
    if not wanted:
        return {}, 0
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    out = {}
    i = 0
    stop = wanted[-1]
    while i <= stop:
        ok, fr = cap.read()
        if not ok:
            break
        last = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        if i in wanted:
            out[i] = last
        i += 1
    n_decoded = i
    cap.release()
    return out, n_decoded


def raw_frame_count(mp4_path: str) -> int:
    import cv2

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float((x * y).sum() / denom)


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def sample_episode_indices(n_ep: int, n_sample: int):
    """Episode indices to sample; first and last always included."""
    if n_sample is None or n_sample <= 0 or n_sample >= n_ep:
        return list(range(n_ep))
    sel = sorted(set([0, n_ep - 1]
                     + [int(round(k * (n_ep - 1) / max(1, n_sample - 1)))
                        for k in range(n_sample)]))
    return sel[:n_sample] if len(sel) > n_sample else sel


def sample_frame_indices(n: int, m: int):
    """m frames spread over [0, n-1]; m == 3 gives first / middle / last."""
    m = max(1, min(int(m), n))
    return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, m)))


# --------------------------------------------------------------------------
# checks 0-7  (same contract as validate_cube_conversion.py)
# --------------------------------------------------------------------------
def self_test_nearest(rep: Report, takes):
    section("CHECK 0  --  self-test of the validator's nearest-timestamp implementation")
    import h5py

    tk = takes[0]
    with h5py.File(os.path.join(tk, "vectors.h5"), "r") as f:
        cam1_t = f["cam1_frames"]["t_rel_s"][:]
        ur_t = f["ur_joint_states"]["t_rel_s"][:]
        cam2_t = f["cam2_frames"]["t_rel_s"][:]
    q = cam1_t[:: max(1, len(cam1_t) // 60)]
    ok1 = np.array_equal(nearest_idx(ur_t, q), nearest_idx_bruteforce(ur_t, q))
    ok2 = np.array_equal(nearest_idx(cam2_t, q), nearest_idx_bruteforce(cam2_t, q))

    ok3 = True
    dpath = os.path.join(tk, "depth.h5")
    if os.path.exists(dpath):
        for cam in ("cam1", "cam2"):
            dt = depth_timeline(dpath, cam)
            ok3 = ok3 and np.array_equal(nearest_idx(dt, q), nearest_idx_bruteforce(dt, q))

    rep.check(ok1 and ok2 and ok3, "vectorised nearest_idx == brute-force argmin reference",
              f"{len(q)} queries against ur/cam2/depth timelines")

    v = np.array([np.nan, np.nan, 1.0, np.nan, np.nan, 2.0, np.nan])
    want = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    rep.check(np.array_equal(ffill_bfill(v), want), "ffill_bfill reference case",
              f"{ffill_bfill(v).tolist()}")


def check_info(rep: Report, info: dict, n_takes: int, expected_total_frames: int, task: str):
    section("CHECK 1  --  meta/info.json")

    rep.check(info.get("codebase_version") == "v3.0",
              "codebase_version == 'v3.0'", f"got {info.get('codebase_version')!r}")
    rep.check(int(info.get("fps", -1)) == FPS, f"fps == {FPS}", f"got {info.get('fps')!r}")

    feats = info.get("features", {})
    missing = [k for k in EXPECTED_FEATURES if k not in feats]
    extra = [k for k in feats if k not in EXPECTED_FEATURES]
    rep.check(not missing, "all expected features present",
              f"missing: {missing}" if missing else f"{len(feats)} features")
    if extra:
        rep.warn("unexpected extra features present", f"{extra}")

    for name, (dtype, shape, names) in EXPECTED_FEATURES.items():
        if name not in feats:
            continue
        f = feats[name]
        got_dtype = f.get("dtype")
        got_shape = list(f.get("shape", []))
        got_names = f.get("names")
        ok = got_dtype == dtype and got_shape == list(shape)
        detail = f"dtype={got_dtype} shape={got_shape}"
        if names is not None:
            ok = ok and list(got_names or []) == list(names)
            detail += f" names={got_names}"
        rep.check(ok, f"feature {name}", detail)

    rep.check(int(info.get("total_episodes", -1)) == n_takes,
              f"total_episodes == {n_takes} (included takes)",
              f"got {info.get('total_episodes')}")
    rep.check(int(info.get("total_frames", -1)) == expected_total_frames,
              f"total_frames == {expected_total_frames} (sum of cam1_frames rows)",
              f"got {info.get('total_frames')}")
    rep.check(int(info.get("total_tasks", -1)) == 1, "total_tasks == 1",
              f"got {info.get('total_tasks')}")

    splits = info.get("splits", {})
    rep.check(splits.get("train") == f"0:{n_takes}", "splits.train covers all episodes",
              f"got {splits!r}")
    if info.get("robot_type"):
        print(f"  [info] robot_type = {info['robot_type']!r}")
    return feats


def check_frame_counts(rep: Report, eps, takes, take_lens):
    section("CHECK 2  --  per-episode frame counts vs source cam1_frames row counts")
    ok_all = True
    mismatches = []
    for i, (ep, tk, n_src) in enumerate(zip(eps, takes, take_lens)):
        n_ep = int(ep["length"])
        span = int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
        if n_ep != n_src or span != n_src:
            ok_all = False
            mismatches.append(f"ep{i} ({os.path.basename(tk)}): length={n_ep} span={span} src={n_src}")
    if ok_all:
        rep.pass_(f"all {len(eps)} episode lengths match their source take",
                  f"total {sum(take_lens)} frames")
    else:
        rep.fail("episode length mismatch", "; ".join(mismatches[:10]))
    return ok_all


def check_numeric(rep: Report, eps, takes, data, tol: float, verbose: bool, max_frames: int,
                  taus: dict):
    section("CHECK 3  --  numeric fidelity of observation.state / action (independent re-derivation)")

    stored_state = data["observation.state"]
    stored_action = data["action"]

    max_dev_state = 0.0
    max_dev_action = 0.0
    worst = ("", 0.0)
    n_bad = 0

    for i, (ep, tk) in enumerate(zip(eps, takes)):
        h5 = os.path.join(tk, "vectors.h5")
        _, _, state, action = derive_take(h5, max_frames, taus[os.path.basename(tk)])
        a = int(ep["dataset_from_index"])
        b = int(ep["dataset_to_index"])
        got_s = stored_state[a:b]
        got_a = stored_action[a:b]

        if got_s.shape != state.shape or got_a.shape != action.shape:
            rep.fail(f"ep{i} shape mismatch",
                     f"stored {got_s.shape}/{got_a.shape} vs derived {state.shape}/{action.shape}")
            n_bad += 1
            continue

        ds = float(np.max(np.abs(got_s.astype(np.float64) - state.astype(np.float64))))
        da = float(np.max(np.abs(got_a.astype(np.float64) - action.astype(np.float64))))
        max_dev_state = max(max_dev_state, ds)
        max_dev_action = max(max_dev_action, da)
        if max(ds, da) > worst[1]:
            worst = (f"ep{i} ({os.path.basename(tk)})", max(ds, da))
        if ds >= tol or da >= tol:
            n_bad += 1
            rep.fail(f"ep{i} ({os.path.basename(tk)}) numeric deviation",
                     f"max|dstate|={ds:.3e} max|daction|={da:.3e}")
        elif verbose:
            print(f"    ep{i:3d} {os.path.basename(tk):32s} "
                  f"dstate={ds:.3e} daction={da:.3e}", flush=True)

    if n_bad == 0:
        rep.pass_(f"all {len(eps)} episodes re-derive exactly (tol={tol:g})",
                  f"max|dstate|={max_dev_state:.3e}  max|daction|={max_dev_action:.3e}")
    print(f"  [info] worst episode: {worst[0]}  max|d|={worst[1]:.3e}")
    return max_dev_state, max_dev_action


def check_nan(rep: Report, data):
    section("CHECK 4  --  no NaN / Inf in stored observation.state and action")
    for key in ("observation.state", "action"):
        arr = data[key]
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        rep.check(n_nan == 0 and n_inf == 0, f"{key} finite",
                  f"NaN={n_nan} Inf={n_inf}  range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")


def check_task(rep: Report, root, eps, data, task: str):
    section("CHECK 5  --  task string")
    tasks = load_tasks(root)
    rep.check(len(tasks) == 1, "exactly one task in meta/tasks.parquet", f"got {tasks}")
    ok_str = len(tasks) == 1 and list(tasks.values())[0] == task
    rep.check(ok_str, f"task string == {task!r}", f"got {list(tasks.values())!r}")

    ti = data["task_index"].astype(np.int64)
    uniq = np.unique(ti)
    rep.check(len(uniq) == 1, "single task_index across all frames", f"unique={uniq.tolist()}")
    resolves = all(int(u) in tasks and tasks[int(u)] == task for u in uniq)
    rep.check(resolves, "every frame's task_index resolves to the task string",
              f"unique={uniq.tolist()} -> {[tasks.get(int(u)) for u in uniq]}")

    bad_eps = []
    for i, ep in enumerate(eps):
        t = ep.get("tasks")
        tl = list(t) if isinstance(t, (list, tuple, np.ndarray)) else [t]
        if [str(x) for x in tl] != [task]:
            bad_eps.append((i, tl))
    rep.check(not bad_eps, "every episode's meta 'tasks' == [task]",
              f"bad: {bad_eps[:5]}" if bad_eps else f"{len(eps)} episodes")


def check_videos(rep: Report, root, info, eps, takes, n_sample, warn_corr, fail_corr,
                 n_frames_per_ep, max_frames):
    section("CHECK 6  --  RGB video integrity (LeRobot AV1 vs raw MPEG-4)")
    if FFMPEG is None:
        rep.skip("video check", "ffmpeg not found on PATH")
        return None

    height = int(info["features"][RGB_KEYS[0]]["shape"][0])
    width = int(info["features"][RGB_KEYS[0]]["shape"][1])
    fps = int(info["fps"])
    video_tmpl = info.get("video_path",
                          "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    sel = sample_episode_indices(len(eps), n_sample)

    all_corr = []
    worst = (None, 1.0)
    n_fail = 0
    plan = {}

    for ei in sel:
        ep = eps[ei]
        tk = takes[ei]
        n = int(ep["length"])
        cam1_t, cam2_t, _, _ = derive_take(os.path.join(tk, "vectors.h5"), max_frames)
        cam2_map = nearest_idx(cam2_t, cam1_t)

        ks = sample_frame_indices(n, n_frames_per_ep)

        raw1, n1 = decode_raw_frames(os.path.join(tk, "cam1.mp4"), ks)
        if n1 < n and raw1:
            last = raw1[max(raw1.keys())]
            for k in ks:
                raw1.setdefault(k, last)
        n2_total = raw_frame_count(os.path.join(tk, "cam2.mp4"))
        hi2 = (n2_total - 1) if n2_total > 0 else int(cam2_map.max())
        want2 = [int(np.clip(cam2_map[k], 0, hi2)) for k in ks]
        raw2, n2 = decode_raw_frames(os.path.join(tk, "cam2.mp4"), want2)
        want2 = [int(np.clip(v, 0, max(0, n2 - 1))) for v in want2]

        for key, raw_map, ref_idx_of in (
            (RGB_KEYS[0], raw1, lambda i, k: k),
            (RGB_KEYS[1], raw2, lambda i, k: want2[i]),
        ):
            ck = int(ep[f"videos/{key}/chunk_index"])
            fk = int(ep[f"videos/{key}/file_index"])
            t0 = float(ep[f"videos/{key}/from_timestamp"])
            vpath = os.path.join(root, video_tmpl.format(video_key=key, chunk_index=ck,
                                                         file_index=fk))
            if not os.path.exists(vpath):
                rep.fail(f"ep{ei} {key}: video file missing", vpath)
                n_fail += 1
                continue
            base = int(round(t0 * fps))
            for i, k in enumerate(ks):
                ridx = ref_idx_of(i, k)
                if ridx not in raw_map:
                    rep.warn(f"ep{ei} {key} frame {k}: raw frame {ridx} not decodable", "")
                    continue
                plan.setdefault((key, vpath), []).append((ei, k, base + k, raw_map[ridx]))

    for (key, vpath), items in sorted(plan.items()):
        wanted = [g for _, _, g, _ in items]
        print(f"    decoding {len(wanted)} frame(s) from {os.path.relpath(vpath, root)} "
              f"(up to frame {max(wanted)}) ...", flush=True)
        try:
            got_map, _ = decode_lerobot_frames(vpath, wanted, height, width)
        except Exception as exc:
            rep.fail(f"{key}: decode failed for {os.path.relpath(vpath, root)}", str(exc)[:200])
            n_fail += 1
            continue
        for ei, k, g, ref in items:
            got = got_map[g]
            if got.shape != ref.shape:
                rep.fail(f"ep{ei} {key} frame {k}: shape mismatch", f"{got.shape} vs {ref.shape}")
                n_fail += 1
                continue
            c = correlation(got, ref)
            mad = mean_abs_diff(got, ref)
            all_corr.append(c)
            if c < worst[1]:
                worst = (f"ep{ei} {key} frame {k}", c)
            status = "ok" if c >= warn_corr else ("low" if c >= fail_corr else "BAD")
            print(f"    ep{ei:3d} {key.split('.')[-1]} frame {k:4d} (video frame {g:5d}) "
                  f"corr={c:.5f} meanabs={mad:6.2f}  [{status}]", flush=True)
            if c < fail_corr:
                n_fail += 1

    if not all_corr:
        rep.skip("video correlation", "no frames compared")
        return None

    arr = np.array(all_corr)
    detail = (f"n={len(arr)} min={arr.min():.5f} mean={arr.mean():.5f} "
              f"worst={worst[0]} (threshold {fail_corr})")
    if n_fail == 0 and arr.min() >= warn_corr:
        rep.pass_("sampled RGB video frames match raw source", detail)
    elif arr.min() >= fail_corr and n_fail == 0:
        rep.warn(f"some frames below {warn_corr} but above {fail_corr} (AV1 is lossy)", detail)
    else:
        rep.fail(f"RGB video frames below correlation threshold {fail_corr}", detail)
    return arr


def check_sanity(rep: Report, info, eps, data):
    section("CHECK 7  --  structural sanity")
    fps = int(info["fps"])

    prev_to = 0
    ok = True
    detail = ""
    for i, ep in enumerate(eps):
        if int(ep["episode_index"]) != i:
            ok, detail = False, f"episode_index out of order at row {i}: {ep['episode_index']}"
            break
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        if a != prev_to or b <= a:
            ok, detail = False, f"ep{i}: from={a} to={b}, expected from={prev_to}"
            break
        prev_to = b
    rep.check(ok, "episode boundaries monotonic & contiguous", detail or f"0 .. {prev_to}")
    rep.check(prev_to == len(data["index"]),
              "episode spans cover all data rows",
              f"last dataset_to_index={prev_to} nrows={len(data['index'])}")

    gidx = data["index"].astype(np.int64)
    rep.check(np.array_equal(gidx, np.arange(len(gidx))),
              "global 'index' == 0..N-1", f"N={len(gidx)}")

    epi = data["episode_index"].astype(np.int64)
    rep.check(np.all(np.diff(epi) >= 0), "episode_index non-decreasing across data", "")

    fi = data["frame_index"].astype(np.int64)
    ok_fi = True
    for i, ep in enumerate(eps):
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        if not np.array_equal(fi[a:b], np.arange(b - a)):
            ok_fi = False
            detail = f"ep{i} frame_index not 0..{b - a - 1}"
            break
        if not np.all(epi[a:b] == i):
            ok_fi = False
            detail = f"ep{i} episode_index column wrong"
            break
    rep.check(ok_fi, "per-episode frame_index restarts at 0 and increments by 1",
              "" if ok_fi else detail)

    ts = data["timestamp"].astype(np.float64)
    expected_ts = fi.astype(np.float64) / fps
    dev = float(np.max(np.abs(ts - expected_ts))) if len(ts) else 0.0
    rep.check(dev < 1e-5, f"timestamp == frame_index / {fps}", f"max dev {dev:.3e}")


# --------------------------------------------------------------------------
# CHECK 8  --  depth feature dicts in info.json
# --------------------------------------------------------------------------
def _info_get(d: dict, key: str):
    """Depth params appear as 'video.depth_min' (lerobot) -- accept the bare name too."""
    if key in d:
        return d[key]
    bare = key.split(".", 1)[1] if key.startswith("video.") else key
    if bare in d:
        return d[bare]
    if ("video." + bare) in d:
        return d["video." + bare]
    return None


def check_depth_features(rep: Report, info: dict):
    section("CHECK 8  --  info.json depth feature dicts")
    feats = info.get("features", {})
    out = {}
    for key in DEPTH_KEYS:
        f = feats.get(key)
        if f is None:
            rep.fail(f"feature {key} present", "missing from info.json/features")
            continue
        rep.check(f.get("dtype") == "video", f"{key} dtype == 'video'", f"got {f.get('dtype')!r}")
        rep.check(list(f.get("shape", [])) == [480, 848, 1], f"{key} shape == [480, 848, 1]",
                  f"got {list(f.get('shape', []))}")
        rep.check(list(f.get("names") or []) == HWC, f"{key} names == {HWC}",
                  f"got {f.get('names')!r}")
        finfo = f.get("info") or {}
        got = {}
        bad = []
        for ik, want in EXPECTED_DEPTH_INFO.items():
            v = _info_get(finfo, ik)
            got[ik] = v
            if isinstance(want, bool):
                ok = bool(v) is want and v is not None
            elif isinstance(want, float):
                ok = v is not None and abs(float(v) - want) < 1e-12
            else:
                ok = v == want
            if not ok:
                bad.append(f"{ik}={v!r} (want {want!r})")
        rep.check(not bad, f"{key} info block (is_depth_map/unit/codec/pix_fmt/quantization)",
                  "; ".join(bad) if bad else
                  f"is_depth_map={got['is_depth_map']} unit={got['depth_unit']!r} "
                  f"codec={got['video.codec']!r} pix_fmt={got['video.pix_fmt']!r} "
                  f"min={got['video.depth_min']} max={got['video.depth_max']} "
                  f"shift={got['video.shift']} use_log={got['video.use_log']}")
        lossless = (finfo.get("video.extra_options") or {}).get("x265-params")
        if lossless != "lossless=1":
            rep.warn(f"{key} x265-params != 'lossless=1'", f"got {lossless!r}")
        out[key] = got

    # RGB features unchanged
    for key in RGB_KEYS:
        f = feats.get(key)
        if f is None:
            rep.fail(f"feature {key} present", "missing")
            continue
        finfo = f.get("info") or {}
        ok = (f.get("dtype") == "video"
              and list(f.get("shape", [])) == [720, 1280, 3]
              and list(f.get("names") or []) == HWC
              and not bool(finfo.get("is_depth_map", False)))
        rep.check(ok, f"{key} unchanged RGB feature",
                  f"shape={list(f.get('shape', []))} codec={finfo.get('video.codec')!r} "
                  f"is_depth_map={finfo.get('is_depth_map')!r}")
    return out


# --------------------------------------------------------------------------
# CHECK 9  --  depth frame counts
# --------------------------------------------------------------------------
def check_depth_counts(rep: Report, root, info, eps):
    section("CHECK 9  --  per-episode depth frame counts")
    fps = int(info["fps"])
    video_tmpl = info.get("video_path",
                          "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    results = {}
    for key in DEPTH_KEYS:
        bad = []
        per_file = {}
        missing_cols = False
        for i, ep in enumerate(eps):
            try:
                ck = int(ep[f"videos/{key}/chunk_index"])
                fk = int(ep[f"videos/{key}/file_index"])
                t0 = float(ep[f"videos/{key}/from_timestamp"])
                t1 = float(ep[f"videos/{key}/to_timestamp"])
            except KeyError:
                missing_cols = True
                break
            n_ep = int(ep["length"])
            n_vid = int(round((t1 - t0) * fps))
            if n_vid != n_ep:
                bad.append(f"ep{i}: video span {n_vid} != length {n_ep}")
            per_file.setdefault((ck, fk), []).append(n_ep)
        if missing_cols:
            rep.fail(f"{key}: episode meta has no videos/{key}/* columns", "")
            continue
        rep.check(not bad, f"{key}: every episode's video span == its frame count",
                  "; ".join(bad[:6]) if bad else f"{len(eps)} episodes")

        # cross-check the actual container frame counts
        file_bad = []
        file_info = []
        for (ck, fk), lens in sorted(per_file.items()):
            vpath = os.path.join(root, video_tmpl.format(video_key=key, chunk_index=ck,
                                                         file_index=fk))
            if not os.path.exists(vpath):
                file_bad.append(f"chunk{ck}/file{fk}: MISSING")
                continue
            nb = ffprobe_nb_frames(vpath)
            want = int(sum(lens))
            file_info.append(f"chunk{ck}/file{fk}: {nb} frames ({len(lens)} eps, want {want})")
            if nb < 0:
                file_bad.append(f"chunk{ck}/file{fk}: nb_frames unavailable")
            elif nb != want:
                file_bad.append(f"chunk{ck}/file{fk}: {nb} != {want}")
        if FFPROBE is None:
            rep.skip(f"{key}: container frame count", "ffprobe not on PATH")
        else:
            rep.check(not file_bad, f"{key}: container nb_frames == sum of episode lengths",
                      "; ".join(file_bad[:6]) if file_bad else "; ".join(file_info[:4]))
        results[key] = {"episodes_ok": not bad, "files": file_info}
    return results


# --------------------------------------------------------------------------
# CHECK 10 -- depth numeric fidelity
# --------------------------------------------------------------------------
def check_depth_numeric(rep: Report, lr_root, repo_id, eps, takes, n_eps, n_frames, max_frames):
    section("CHECK 10  --  depth numeric fidelity (LeRobot decode vs depth.h5 ground truth)")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        rep.skip("depth decode", f"lerobot not importable in this interpreter: {exc}")
        return None

    try:
        ds = LeRobotDataset(repo_id, root=lr_root)
    except Exception as exc:
        rep.fail("LeRobotDataset could not open the dataset", str(exc)[:300])
        return None

    sel = sample_episode_indices(len(eps), n_eps)
    print(f"    sampling {len(sel)} episode(s) x {n_frames} frame(s) x 2 cams", flush=True)

    agg = {k: {"n": 0, "med_max": 0.0, "p99_max": 0.0, "abs_max": 0.0,
               "zero_frac_min": 1.0, "n_valid": 0, "n_zero": 0,
               "bad_med": 0, "bad_p99": 0, "bad_max": 0, "bad_zero": 0}
           for k in DEPTH_KEYS}
    errors = []

    for ei in sel:
        ep = eps[ei]
        tk = takes[ei]
        a = int(ep["dataset_from_index"])
        n = int(ep["length"])
        cam1_t = take_cam1_t(os.path.join(tk, "vectors.h5"), max_frames)
        ks = sample_frame_indices(min(n, len(cam1_t)), n_frames)
        dpath = os.path.join(tk, "depth.h5")
        if not os.path.exists(dpath):
            rep.fail(f"ep{ei}: depth.h5 missing", dpath)
            continue

        for key in DEPTH_KEYS:
            cam = DEPTH_KEY_TO_CAM[key]
            dt = depth_timeline(dpath, cam)
            dmap = nearest_idx(dt, cam1_t)
            want = [int(dmap[k]) for k in ks]
            try:
                raws = read_raw_depth_frames(dpath, cam, want)
            except Exception as exc:
                rep.fail(f"ep{ei} {key}: raw depth decode failed", str(exc)[:200])
                continue
            A = agg[key]
            for k, ri in zip(ks, want):
                try:
                    item = ds[a + k]
                except Exception as exc:
                    rep.fail(f"ep{ei} {key}: LeRobotDataset[{a + k}] raised", str(exc)[:200])
                    continue
                x = item.get(key)
                if x is None:
                    rep.fail(f"ep{ei} {key}: key absent from decoded item", f"keys={list(item)[:8]}")
                    continue
                arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
                dec = np.squeeze(np.asarray(arr, dtype=np.float64))
                raw = raws[ri].astype(np.float64)
                if dec.shape != raw.shape:
                    rep.fail(f"ep{ei} {key} frame {k}: shape mismatch",
                             f"decoded {dec.shape} vs raw {raw.shape}")
                    continue
                valid = raw > 0
                nv = int(valid.sum())
                nz = int((~valid).sum())
                med = p99 = mx = 0.0
                if nv:
                    e = np.abs(dec[valid] - raw[valid])
                    med = float(np.median(e))
                    p99 = float(np.percentile(e, 99))
                    mx = float(e.max())
                zfrac = 1.0
                if nz:
                    zfrac = float((dec[~valid] <= DEPTH_ZERO_DECODE_MAX_MM).mean())
                A["n"] += 1
                A["n_valid"] += nv
                A["n_zero"] += nz
                A["med_max"] = max(A["med_max"], med)
                A["p99_max"] = max(A["p99_max"], p99)
                A["abs_max"] = max(A["abs_max"], mx)
                A["zero_frac_min"] = min(A["zero_frac_min"], zfrac)
                if med > DEPTH_ERR_MED_MM:
                    A["bad_med"] += 1
                    errors.append(f"ep{ei} {cam} f{k}: median err {med:g} mm")
                if p99 > DEPTH_ERR_P99_MM:
                    A["bad_p99"] += 1
                    errors.append(f"ep{ei} {cam} f{k}: p99 err {p99:g} mm")
                if mx > DEPTH_ERR_MAX_MM:
                    A["bad_max"] += 1
                    errors.append(f"ep{ei} {cam} f{k}: max err {mx:g} mm")
                if zfrac < DEPTH_ZERO_FRAC:
                    A["bad_zero"] += 1
                    errors.append(f"ep{ei} {cam} f{k}: only {zfrac * 100:.4f}% of raw-zero "
                                  f"px decode <= {DEPTH_ZERO_DECODE_MAX_MM} mm")

    for key in DEPTH_KEYS:
        A = agg[key]
        if A["n"] == 0:
            rep.skip(f"{key}: depth fidelity", "no frames compared")
            continue
        detail = (f"n={A['n']} frames  valid px={A['n_valid']}  zero px={A['n_zero']}  "
                  f"max-of-medians={A['med_max']:g} mm  max-of-p99={A['p99_max']:g} mm  "
                  f"max abs err={A['abs_max']:g} mm  "
                  f"min raw-zero<= {DEPTH_ZERO_DECODE_MAX_MM}mm frac={A['zero_frac_min'] * 100:.4f}%")
        ok = (A["bad_med"] == 0 and A["bad_p99"] == 0
              and A["bad_max"] == 0 and A["bad_zero"] == 0)
        rep.check(ok, f"{key}: valid-px err (median<={DEPTH_ERR_MED_MM}, p99<={DEPTH_ERR_P99_MM}, "
                      f"max<={DEPTH_ERR_MAX_MM} mm) and raw-zero decode", detail)
    if errors:
        print("  [info] first offending frames: " + "; ".join(errors[:6]))
    return {k: {kk: vv for kk, vv in v.items()} for k, v in agg.items()}


# --------------------------------------------------------------------------
# CHECK 11 -- meta/depth_cameras.json
# --------------------------------------------------------------------------
def _allclose(a, b, atol=1e-9):
    try:
        A = np.asarray(a, dtype=np.float64).reshape(-1)
        B = np.asarray(b, dtype=np.float64).reshape(-1)
    except Exception:
        return False
    if A.shape != B.shape:
        return False
    return bool(np.allclose(A, B, rtol=0.0, atol=atol))


def check_depth_cameras_json(rep: Report, lr_root, takes, atol=1e-9):
    section("CHECK 11  --  meta/depth_cameras.json vs depth.h5 of EVERY take")
    path = os.path.join(lr_root, "meta", "depth_cameras.json")
    if not os.path.exists(path):
        rep.fail("meta/depth_cameras.json exists", path)
        return None
    try:
        with open(path) as fh:
            sc = json.load(fh)
    except Exception as exc:
        rep.fail("meta/depth_cameras.json parses as JSON", str(exc)[:200])
        return None
    rep.pass_("meta/depth_cameras.json exists and parses", f"keys={sorted(sc)}")

    out = {}
    for sk, cam in SIDECAR_KEY_TO_CAM.items():
        entry = sc.get(sk)
        if entry is None:
            rep.fail(f"depth_cameras.json['{sk}'] present", f"keys={sorted(sc)}")
            continue
        ci = entry.get("camera_info") or {}
        ext = entry.get("extrinsics_depth_to_color") or {}
        per_ep = entry.get("per_episode")
        if per_ep is not None:
            rep.warn(f"{sk}: sidecar carries a 'per_episode' list",
                     "intrinsics are NOT identical across takes -- comparing per take")

        rep.check(entry.get("aligned_to_color") is False, f"{sk}.aligned_to_color is false",
                  f"got {entry.get('aligned_to_color')!r}")
        rep.check(str(entry.get("unit")) == "mm", f"{sk}.unit == 'mm'",
                  f"got {entry.get('unit')!r}")
        rep.check(int(entry.get("width", -1)) == 848 and int(entry.get("height", -1)) == 480,
                  f"{sk} width/height == 848x480",
                  f"got {entry.get('width')}x{entry.get('height')}")

        bad = []
        n_checked = 0
        first = None
        for ti, tk in enumerate(takes):
            dpath = os.path.join(tk, "depth.h5")
            if not os.path.exists(dpath):
                bad.append(f"{os.path.basename(tk)}: depth.h5 missing")
                continue
            m = depth_group_meta(dpath, cam)
            if first is None:
                first = m
            ref_ci = m.get("camera_info", {})
            ref_ext = m.get("extrinsics_depth_to_color", {})
            want_ci, want_ext = ci, ext
            if per_ep is not None and ti < len(per_ep):
                pe = per_ep[ti] or {}
                want_ci = pe.get("camera_info", ci)
                want_ext = pe.get("extrinsics_depth_to_color", ext)
            probs = []
            if not _allclose(want_ci.get("K"), ref_ci.get("K"), atol):
                probs.append("K")
            if not _allclose(want_ci.get("D"), ref_ci.get("D"), atol):
                probs.append("D")
            if str(want_ci.get("distortion_model")) != str(ref_ci.get("distortion_model")):
                probs.append("distortion_model")
            if str(want_ci.get("frame_id")) != str(ref_ci.get("frame_id")):
                probs.append("frame_id")
            if int(want_ci.get("width", entry.get("width", -1))) != int(ref_ci.get("width", -2)):
                probs.append("camera_info.width")
            if int(want_ci.get("height", entry.get("height", -1))) != int(ref_ci.get("height", -2)):
                probs.append("camera_info.height")
            if not _allclose(want_ext.get("rotation"), ref_ext.get("rotation"), atol):
                probs.append("extrinsics.rotation")
            t_want = want_ext.get("translation_m", want_ext.get("translation"))
            if not _allclose(t_want, ref_ext.get("translation"), atol):
                probs.append("extrinsics.translation")
            if str(want_ext.get("rotation_layout", want_ext.get("layout"))) != \
                    str(ref_ext.get("layout")):
                probs.append("extrinsics.layout")
            if int(entry.get("width", -1)) != int(m.get("width", -2)) or \
                    int(entry.get("height", -1)) != int(m.get("height", -2)):
                probs.append("group width/height")
            if bool(m.get("aligned_to_color")) is not False:
                probs.append("raw aligned_to_color is True")
            st = entry.get("source_topic")
            if st is not None and str(st) != str(m.get("source_topic")):
                probs.append("source_topic")
            if probs:
                bad.append(f"{os.path.basename(tk)}: {','.join(probs)}")
            n_checked += 1

        rep.check(not bad, f"{sk}: K/D/model/frame_id/extrinsics match all {n_checked} takes "
                           f"(atol {atol:g})",
                  "; ".join(bad[:5]) if bad else
                  (f"K[0]={ref_ci.get('K', [None])[0]!r} "
                   f"|t|={np.linalg.norm(np.asarray(ref_ext.get('translation', [0, 0, 0]), float)):.5f} m"
                   if first else ""))

        cf = entry.get("color_feature")
        if cf is not None:
            want_cf = "observation.images." + cam
            rep.check(str(cf) == want_cf, f"{sk}.color_feature == {want_cf!r}", f"got {cf!r}")
        else:
            rep.warn(f"{sk}.color_feature absent", "spec asks for it")
        out[sk] = {"n_takes_checked": n_checked, "mismatches": bad}

    note = sc.get("invalid_value_note")
    if note is None:
        note = next((v.get("invalid_value_note") for v in sc.values()
                     if isinstance(v, dict) and v.get("invalid_value_note")), None)
    rep.check(bool(note), "invalid_value_note present",
              (str(note)[:100] + "...") if note else "missing")

    quant = sc.get("quantization")
    if quant is None:
        quant = next((v.get("quantization") for v in sc.values()
                      if isinstance(v, dict) and v.get("quantization")), None)
    if quant:
        rep.pass_("quantization block present", f"{json.dumps(quant)[:120]}")
    else:
        # NOTE: FACTS_carrot_in_pot.md's sidecar spec does not list a `quantization`
        # key (only the validator brief does), so a converter that followed FACTS
        # literally will not have written one. WARN, not FAIL.
        rep.warn("quantization block absent from meta/depth_cameras.json",
                 "the FACTS sidecar spec does not list it; the validator brief does")
    return out


# --------------------------------------------------------------------------
# CHECK 12 -- meta/source_takes.json
# --------------------------------------------------------------------------
def _parse_source_takes(obj):
    """
    Tolerant parser: the spec fixes the CONTENT (take order + n_frames) but not the
    exact JSON shape. Accepts:
      [ "take_01_...", ... ]
      [ {"take"/"name"/"take_name"/"dir"/"path": ..., "n_frames"/"length"/"frames": ...}, ... ]
      { "takes": <either of the above>, ... }
    Returns (names, n_frames_or_None_list).
    """
    if isinstance(obj, dict):
        for k in ("takes", "source_takes", "episodes"):
            if k in obj:
                obj = obj[k]
                break
        else:
            # dict keyed by take name
            items = sorted(obj.items())
            names = [k for k, _ in items]
            counts = []
            for _, v in items:
                counts.append(_take_count(v))
            return names, counts
    if not isinstance(obj, list):
        return None, None
    names, counts = [], []
    for e in obj:
        if isinstance(e, str):
            names.append(os.path.basename(e.rstrip("/")))
            counts.append(None)
        elif isinstance(e, dict):
            nm = None
            for k in ("take", "name", "take_name", "take_dir_name", "dir", "dirname",
                      "path", "source", "take_dir", "source_take"):
                if k in e:
                    nm = os.path.basename(str(e[k]).rstrip("/"))
                    break
            names.append(nm)
            counts.append(_take_count(e))
        else:
            names.append(None)
            counts.append(None)
    return names, counts


def _take_count(v):
    if isinstance(v, dict):
        for k in ("n_frames", "num_frames", "length", "frames", "n"):
            if k in v:
                try:
                    return int(v[k])
                except Exception:
                    return None
    elif isinstance(v, (int, np.integer)):
        return int(v)
    return None


def check_source_takes_json(rep: Report, lr_root, takes, eps):
    section("CHECK 12  --  meta/source_takes.json")
    path = os.path.join(lr_root, "meta", "source_takes.json")
    if not os.path.exists(path):
        rep.fail("meta/source_takes.json exists", path)
        return None
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except Exception as exc:
        rep.fail("meta/source_takes.json parses as JSON", str(exc)[:200])
        return None

    names, counts = _parse_source_takes(obj)
    if names is None:
        rep.fail("meta/source_takes.json has a recognisable take list",
                 f"top-level type {type(obj).__name__}")
        return None

    want_names = [os.path.basename(t) for t in takes]
    if names == want_names:
        detail = f"{len(names)} entries, order matches"
    else:
        i = next((i for i, (a, b) in enumerate(zip(names, want_names)) if a != b), None)
        detail = (f"{len(names)} entries (want {len(want_names)})"
                  + (f"; first diff at {i}: got {names[i]!r} want {want_names[i]!r}"
                     if i is not None else ""))
    rep.check(names == want_names,
              f"take list == sorted raw dir minus --exclude ({len(want_names)} takes)", detail)

    if any(c is None for c in counts):
        rep.warn("source_takes.json carries no per-take n_frames", "cannot cross-check lengths")
    else:
        want_counts = [int(e["length"]) for e in eps]
        n = min(len(counts), len(want_counts))
        bad = [f"{want_names[i] if i < len(want_names) else i}: {counts[i]} != {want_counts[i]}"
               for i in range(n) if counts[i] != want_counts[i]]
        rep.check(not bad and len(counts) == len(want_counts),
                  "per-take n_frames == episode length",
                  "; ".join(bad[:6]) if bad else f"sum={sum(counts)} over {len(counts)} takes")
    return {"names": names, "n_frames": counts}


# --------------------------------------------------------------------------
# CHECK 13 -- depth timestamp sanity (raw only)
# --------------------------------------------------------------------------
def check_depth_timestamps(rep: Report, takes, max_frames,
                           soft_tol_s=DEPTH_DT_TOL_S, hard_tol_s=DEPTH_DT_HARD_TOL_S,
                           frac=DEPTH_DT_FRAC):
    """
    |t_depth - t_cam1| for the depth frame the converter is required to pick.

    Computed from RAW timestamps only (no lerobot, no decode): this characterises the
    RECORDING, not the conversion -- a converter bug cannot show up here. Two tiers,
    see the DEPTH_DT_* comment at the top of this file: the agreed half-frame+5 ms
    number is reported as a WARN (cam1's median offset alone eats half of it), and the
    hard FAIL is "the chosen depth frame is more than a whole frame away".
    """
    section("CHECK 13  --  depth timestamp sanity (|t_depth - t_cam1|, raw clocks only)")
    print(f"    soft tier (WARN) {soft_tol_s * 1000:.2f} ms | "
          f"hard tier (FAIL) {hard_tol_s * 1000:.2f} ms | "
          f"both for >= {frac * 100:.0f}% of frames per episode", flush=True)
    out = {}
    for key in DEPTH_KEYS:
        cam = DEPTH_KEY_TO_CAM[key]
        worst_max = 0.0
        worst_where = ""
        worst_soft = 1.0
        worst_hard = 1.0
        med_offsets = []
        soft_bad = []
        hard_bad = []
        missing = []
        for ei, tk in enumerate(takes):
            dpath = os.path.join(tk, "depth.h5")
            if not os.path.exists(dpath):
                missing.append(f"{os.path.basename(tk)}: depth.h5 missing")
                continue
            cam1_t = take_cam1_t(os.path.join(tk, "vectors.h5"), max_frames)
            dt = depth_timeline(dpath, cam)
            j = nearest_idx(dt, cam1_t)
            signed = np.asarray(dt, dtype=np.float64)[j] - cam1_t.astype(np.float64)
            d = np.abs(signed)
            if not len(d):
                continue
            med_offsets.append(float(np.median(signed)))
            f_soft = float((d <= soft_tol_s).mean())
            f_hard = float((d <= hard_tol_s).mean())
            mx = float(d.max())
            if mx > worst_max:
                worst_max, worst_where = mx, f"ep{ei} ({os.path.basename(tk)})"
            worst_soft = min(worst_soft, f_soft)
            worst_hard = min(worst_hard, f_hard)
            # Small-sample guard: with a --max-frames-truncated episode, `frac`=0.99
            # over 45 frames allows 0.45 outliers, i.e. none at all. Allow one outlier
            # unconditionally. For a full take (~330 frames) the fraction gate is the
            # binding one anyway (1 % of 330 = 3.3 frames), so this never loosens the
            # real check.
            allow = max(1, int(round((1.0 - frac) * len(d))))
            if int((d > soft_tol_s).sum()) > allow:
                soft_bad.append(f"ep{ei} ({os.path.basename(tk)}): {f_soft * 100:.2f}%")
            if int((d > hard_tol_s).sum()) > allow:
                hard_bad.append(f"ep{ei} ({os.path.basename(tk)}): {f_hard * 100:.2f}% "
                                f"(max {mx * 1000:.1f} ms)")
        if missing:
            rep.fail(f"{cam}: depth.h5 present for every take", "; ".join(missing[:5]))

        med = float(np.median(med_offsets)) * 1000.0 if med_offsets else float("nan")
        summary = (f"worst-episode in-tol: soft {worst_soft * 100:.2f}% / "
                   f"hard {worst_hard * 100:.2f}%; max |dt| {worst_max * 1000:.2f} ms "
                   f"at {worst_where}; median signed offset {med:+.2f} ms")
        rep.check(not hard_bad,
                  f"{cam}: >= {frac * 100:.0f}% of frames within {hard_tol_s * 1000:.2f} ms "
                  f"(one frame) of the cam1 master clock",
                  "; ".join(hard_bad[:5]) if hard_bad else summary)
        if soft_bad:
            rep.warn(f"{cam}: {len(soft_bad)} episode(s) below {frac * 100:.0f}% within the "
                     f"agreed {soft_tol_s * 1000:.2f} ms (raw-clock phase, not a conversion "
                     f"error)", "; ".join(soft_bad[:5]) + f"  [{summary}]")
        else:
            rep.pass_(f"{cam}: >= {frac * 100:.0f}% of frames within the agreed "
                      f"{soft_tol_s * 1000:.2f} ms", summary)
        out[cam] = {"max_dt_ms": worst_max * 1000.0,
                    "min_in_tol_frac_soft": worst_soft,
                    "min_in_tol_frac_hard": worst_hard,
                    "median_signed_offset_ms": med,
                    "soft_tol_ms": soft_tol_s * 1000.0,
                    "hard_tol_ms": hard_tol_s * 1000.0,
                    "worst_episode": worst_where}
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# CHECK 14 -- the timestamp correction is declared in meta/source_takes.json
# --------------------------------------------------------------------------
def check_lag_metadata(rep: Report, lr_root, takes, taus):
    section("CHECK 14  --  meta/source_takes.json declares the ur_joint_states timestamp correction")
    path = os.path.join(lr_root, "meta", "source_takes.json")
    if not os.path.exists(path):
        rep.fail("meta/source_takes.json exists (check 14)", path)
        return None
    with open(path) as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        rep.fail("source_takes.json is an object with a timestamp_correction block",
                 f"top-level type {type(obj).__name__}")
        return None

    # 14a -- per-episode tau
    eps_list = obj.get("episodes")
    got = {}
    if isinstance(eps_list, list):
        for e in eps_list:
            if isinstance(e, dict) and "ur_joint_states_lag_s" in e:
                nm = e.get("take_dir_name") or e.get("take") or e.get("name")
                if nm is not None:
                    got[os.path.basename(str(nm))] = float(e["ur_joint_states_lag_s"])
    want = {os.path.basename(t): taus[os.path.basename(t)] for t in takes}
    missing = sorted(set(want) - set(got))
    bad = sorted(k for k in want if k in got and abs(got[k] - want[k]) > 1e-9)
    rep.check(
        not missing and not bad and len(got) == len(want),
        f"every episode carries ur_joint_states_lag_s == the expected tau ({len(want)} takes)",
        (f"missing={missing[:5]} mismatched="
         + "; ".join(f"{k}: {got[k]} != {want[k]}" for k in bad[:5])) if (missing or bad)
        else f"distinct tau values: {sorted(set(want.values()))}",
    )

    # 14b -- the top-level block
    tc = obj.get("timestamp_correction")
    if not isinstance(tc, dict):
        rep.fail("top-level timestamp_correction block present",
                 f"got {type(tc).__name__}")
        return {"per_episode_lag_s": got, "timestamp_correction": None}
    rep.pass_("top-level timestamp_correction block present", f"keys={sorted(tc)}")

    val = tc.get("ur_joint_states_lag_s")
    vals = [float(x) for x in (val if isinstance(val, (list, tuple)) else [val])] \
        if val is not None else []
    want_vals = sorted(set(want.values()))
    rep.check(
        bool(vals) and sorted(set(vals)) == want_vals,
        "timestamp_correction.ur_joint_states_lag_s == the applied tau value(s)",
        f"declared {val!r}, applied {want_vals}",
    )
    why = str(tc.get("why") or "")
    rep.check(bool(why.strip()), "timestamp_correction.why is non-empty", why[:120])
    rep.check("tcp_pose/wrench" in tc,
              "timestamp_correction mentions tcp_pose/wrench",
              f"{tc.get('tcp_pose/wrench')!r}")
    return {"per_episode_lag_s": got, "timestamp_correction": tc}


# --------------------------------------------------------------------------
# CHECK 15 -- physics sanity: does the follower actually sit on its command?
# --------------------------------------------------------------------------
def check_state_action_physics(rep: Report, eps, takes, data, taus, max_frames, fps):
    section("CHECK 15  --  physics sanity: mean |state[0:6] - action[0:6]| with vs without "
            "the correction")
    skip_n = int(round(STATE_ACTION_SKIP_HEAD_S * float(fps)))
    stored_state = data["observation.state"]
    stored_action = data["action"]
    corr_means, zero_means, rows = [], [], []
    n_short = 0
    for i, (ep, tk) in enumerate(zip(eps, takes)):
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        if b - a <= skip_n + 5:
            n_short += 1
            continue
        st = stored_state[a:b, :6].astype(np.float64)
        ac = stored_action[a:b, :6].astype(np.float64)
        m_corr = float(np.mean(np.abs(st[skip_n:] - ac[skip_n:])))
        # independent re-derivation at tau = 0 (the uncorrected v1 behaviour)
        _, _, s0, a0 = derive_take(os.path.join(tk, "vectors.h5"), max_frames, 0.0)
        m_zero = float(np.mean(np.abs(s0[skip_n:, :6].astype(np.float64)
                                      - a0[skip_n:, :6].astype(np.float64))))
        corr_means.append(m_corr)
        zero_means.append(m_zero)
        rows.append({"episode_index": i, "take": os.path.basename(tk),
                     "mean_abs_rad_corrected": m_corr, "mean_abs_rad_tau0": m_zero})
    if not rows:
        rep.skip("state/action agreement", f"no episode longer than {skip_n} frames")
        return None

    c = np.array(corr_means)
    z = np.array(zero_means)
    worst = rows[int(np.argmax(c))]
    rep.check(
        float(c.max()) < STATE_ACTION_MAX_RAD,
        f"corrected mean |state-action| < {STATE_ACTION_MAX_RAD} rad in every episode",
        f"median {np.median(c):.5f} rad, max {c.max():.5f} rad "
        f"(worst {worst['take']}), n={len(c)} episodes",
    )
    n_worse = int(np.sum(z > c))
    rep.check(
        n_worse == len(c),
        "uncorrected (tau = 0) re-derivation is worse in every episode",
        f"tau=0 median {np.median(z):.5f} rad, max {z.max():.5f} rad; "
        f"worse in {n_worse}/{len(c)} episodes; median improvement factor "
        f"{np.median(z / np.maximum(c, 1e-12)):.1f}x",
    )
    print(f"  [info] first {STATE_ACTION_SKIP_HEAD_S} s excluded ({skip_n} frames)"
          + (f"; {n_short} episode(s) too short and skipped" if n_short else ""))
    return {
        "skip_head_s": STATE_ACTION_SKIP_HEAD_S,
        "skip_frames": skip_n,
        "gate_rad": STATE_ACTION_MAX_RAD,
        "n_episodes": len(rows),
        "corrected": {"median": float(np.median(c)), "mean": float(c.mean()),
                      "min": float(c.min()), "max": float(c.max())},
        "tau0": {"median": float(np.median(z)), "mean": float(z.mean()),
                 "min": float(z.min()), "max": float(z.max())},
        "n_episodes_improved": n_worse,
        "per_episode": rows,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Validate the carrot_in_pot LeRobot v3.0 dataset (incl. depth) "
                    "against its raw HDF5/mp4/depth takes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw", default="/home/laptop3/youngwoong_ws/Put_carrot_in_pot",
                    help="directory containing take_*/ subdirectories")
    ap.add_argument("--lerobot", default="/home/laptop3/youngwoong_ws/carrot_in_pot_lerobot",
                    help="root of the LeRobot v3.0 dataset")
    ap.add_argument("--repo-id", default="Bigenlight/carrot_in_pot_lerobot_v3",
                    help="repo id label passed to LeRobotDataset (local read only)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="basename of a take to exclude (repeatable)")
    ap.add_argument("--task", default="Put carrot in pot", help="expected task string")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max allowed abs deviation for state/action")
    ap.add_argument("--ur-lag-s", type=float, default=0.9,
                    help="recorder timestamp correction the dataset was built with: the "
                         "ur_joint_states row clock is t_rel_s - tau (applied to that table "
                         "only). 0 = validate an uncorrected dataset")
    ap.add_argument("--lag-json", default=None,
                    help="per-take override for --ur-lag-s: JSON of take dir name -> "
                         "{\"tau_q_s\": float} (or a bare float); wins over --ur-lag-s")
    ap.add_argument("--video-episodes", type=int, default=3,
                    help="episodes sampled for the RGB video check (first and last always included)")
    ap.add_argument("--video-frames", type=int, default=4,
                    help="RGB frames sampled per episode per camera")
    ap.add_argument("--video-warn-corr", type=float, default=0.99,
                    help="RGB correlation below this is a warning")
    ap.add_argument("--video-fail-corr", type=float, default=0.98,
                    help="RGB correlation below this is a hard FAIL")
    ap.add_argument("--depth-episodes", type=int, default=0,
                    help="episodes sampled for the depth fidelity check (0 = every episode)")
    ap.add_argument("--depth-frames-per-episode", type=int, default=3,
                    help="depth frames sampled per episode per camera (3 = first/middle/last)")
    ap.add_argument("--depth-dt-tol-ms", type=float, default=DEPTH_DT_TOL_S * 1000.0,
                    help="check 13 SOFT tier (WARN): the agreed half-frame + 5 ms")
    ap.add_argument("--depth-dt-hard-tol-ms", type=float, default=DEPTH_DT_HARD_TOL_S * 1000.0,
                    help="check 13 HARD tier (FAIL): one frame period")
    ap.add_argument("--depth-dt-frac", type=float, default=DEPTH_DT_FRAC,
                    help="fraction of frames per episode that must be within each tier")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="TEST ONLY: mirror a truncated conversion by clipping each take's "
                         "master clock to its first N frames (0 = full take)")
    ap.add_argument("--max-takes", type=int, default=0,
                    help="TEST ONLY: only consider the first N included takes (0 = all)")
    ap.add_argument("--skip-video", action="store_true", help="skip check 6 (RGB video)")
    ap.add_argument("--skip-depth-decode", action="store_true",
                    help="skip check 10 (the only check that needs lerobot)")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write a machine-readable results dict to this path")
    ap.add_argument("--verbose", action="store_true", help="print per-episode numeric deviations")
    args = ap.parse_args()

    raw_root = os.path.abspath(os.path.expanduser(args.raw))
    lr_root = os.path.abspath(os.path.expanduser(args.lerobot))

    print("LeRobot v3.0 conversion validator  --  carrot_in_pot (RGB + depth)")
    print(f"  interpreter    : {sys.executable}")
    print(f"  raw takes      : {raw_root}")
    print(f"  lerobot dataset: {lr_root}")
    print(f"  repo id        : {args.repo_id}")
    print(f"  excluded takes : {args.exclude or '(none)'}")
    print(f"  expected task  : {args.task!r}")
    print(f"  numeric tol    : {args.tol:g}")
    print(f"  ur lag (tau)   : {args.ur_lag_s} s"
          + (f"  [per-take overrides from {args.lag_json}]" if args.lag_json else ""))
    if args.max_frames or args.max_takes:
        print(f"  !! TEST MODE   : --max-frames={args.max_frames} --max-takes={args.max_takes} "
              f"(checks 2/3/6/9/10/12/13 are evaluated on the truncated prefix only)")

    rep = Report()

    # ---- resolve takes -------------------------------------------------
    all_takes = sorted(glob.glob(os.path.join(raw_root, "take_*")))
    all_takes = [t for t in all_takes if os.path.isdir(t)]
    if not all_takes:
        print(f"\nFATAL: no take_* directories under {raw_root}", file=sys.stderr)
        return 2
    excl = set(args.exclude)
    unknown = sorted(excl - {os.path.basename(t) for t in all_takes})
    takes = [t for t in all_takes if os.path.basename(t) not in excl]
    if args.max_takes and args.max_takes > 0:
        takes = takes[:args.max_takes]
    print(f"  raw takes found: {len(all_takes)}  ->  included: {len(takes)}")
    if unknown:
        rep.warn("--exclude names not found in raw dir", f"{unknown}")

    try:
        taus = resolve_taus(takes, args.ur_lag_s, args.lag_json)
    except Exception as exc:
        print(f"\nFATAL: --lag-json unreadable: {exc}", file=sys.stderr)
        return 2
    tau_values = sorted(set(taus.values()))
    print(f"  expected tau   : {tau_values} s over {len(taus)} takes")
    if args.lag_json:
        lag_missing = [n for n in taus if n not in load_lag_json(args.lag_json)]
        if lag_missing:
            rep_pre_warn = lag_missing
        else:
            rep_pre_warn = []
    else:
        rep_pre_warn = []

    missing_files = []
    for t in takes:
        for fn in ("vectors.h5", "cam1.mp4", "cam2.mp4", "depth.h5"):
            if not os.path.exists(os.path.join(t, fn)):
                missing_files.append(os.path.join(os.path.basename(t), fn))
    if missing_files:
        rep.fail("raw take files missing", f"{missing_files[:8]}")

    if not os.path.isdir(lr_root):
        print(f"\nFATAL: LeRobot dataset not found at {lr_root}", file=sys.stderr)
        return 2

    if rep_pre_warn:
        rep.warn("--lag-json covers every included take",
                 f"{len(rep_pre_warn)} take(s) fall back to --ur-lag-s={args.ur_lag_s}: "
                 f"{rep_pre_warn[:5]}")

    # ---- check 0 -------------------------------------------------------
    self_test_nearest(rep, takes)

    info_path = os.path.join(lr_root, "meta", "info.json")
    if not os.path.exists(info_path):
        print(f"\nFATAL: {info_path} not found", file=sys.stderr)
        return 2
    with open(info_path) as fh:
        info = json.load(fh)

    print("\nreading source cam1_frames row counts ...", flush=True)
    take_lens = [len(take_cam1_t(os.path.join(t, "vectors.h5"), args.max_frames)) for t in takes]
    expected_total = int(sum(take_lens))
    print(f"  {len(takes)} takes, {expected_total} frames expected", flush=True)

    check_info(rep, info, len(takes), expected_total, args.task)

    try:
        data, data_files, backend = load_data_frames(lr_root, info)
        eps = load_episodes_meta(lr_root)
    except Exception:
        print("\nFATAL: could not read the LeRobot parquet files:", file=sys.stderr)
        traceback.print_exc()
        return 2
    print(f"\n  parquet backend: {backend}, {len(data_files)} data file(s), "
          f"{len(eps)} episode rows, {len(data['index'])} frames", flush=True)

    if len(eps) != len(takes):
        rep.fail("episode count vs included take count",
                 f"{len(eps)} episodes vs {len(takes)} takes -- "
                 f"check --exclude; remaining per-episode checks may be misaligned")
        n = min(len(eps), len(takes))
        eps_c, takes_c, lens_c = eps[:n], takes[:n], take_lens[:n]
    else:
        rep.pass_("episode count == included take count", f"{len(eps)}")
        eps_c, takes_c, lens_c = eps, takes, take_lens

    check_frame_counts(rep, eps_c, takes_c, lens_c)
    dev_s, dev_a = check_numeric(rep, eps_c, takes_c, data, args.tol, args.verbose,
                                 args.max_frames, taus)
    check_nan(rep, data)
    check_task(rep, lr_root, eps, data, args.task)

    corr = None
    if args.skip_video:
        section("CHECK 6  --  RGB video integrity")
        rep.skip("RGB video check", "--skip-video")
    else:
        try:
            corr = check_videos(rep, lr_root, info, eps_c, takes_c,
                                args.video_episodes, args.video_warn_corr,
                                args.video_fail_corr, args.video_frames, args.max_frames)
        except Exception:
            rep.fail("RGB video check raised an exception",
                     traceback.format_exc().splitlines()[-1])

    check_sanity(rep, info, eps, data)

    depth_feat = check_depth_features(rep, info)

    try:
        depth_counts = check_depth_counts(rep, lr_root, info, eps_c)
    except Exception:
        depth_counts = None
        rep.fail("depth frame count check raised an exception",
                 traceback.format_exc().splitlines()[-1])

    depth_num = None
    if args.skip_depth_decode:
        section("CHECK 10  --  depth numeric fidelity")
        rep.skip("depth decode", "--skip-depth-decode")
    else:
        try:
            depth_num = check_depth_numeric(rep, lr_root, args.repo_id, eps_c, takes_c,
                                            args.depth_episodes,
                                            args.depth_frames_per_episode, args.max_frames)
        except Exception:
            rep.fail("depth fidelity check raised an exception",
                     traceback.format_exc().splitlines()[-1])

    try:
        cams_json = check_depth_cameras_json(rep, lr_root, takes)
    except Exception:
        cams_json = None
        rep.fail("depth_cameras.json check raised an exception",
                 traceback.format_exc().splitlines()[-1])

    try:
        src_json = check_source_takes_json(rep, lr_root, takes, eps_c)
    except Exception:
        src_json = None
        rep.fail("source_takes.json check raised an exception",
                 traceback.format_exc().splitlines()[-1])

    try:
        depth_ts = check_depth_timestamps(rep, takes_c, args.max_frames,
                                          soft_tol_s=args.depth_dt_tol_ms / 1000.0,
                                          hard_tol_s=args.depth_dt_hard_tol_ms / 1000.0,
                                          frac=args.depth_dt_frac)
    except Exception:
        depth_ts = None
        rep.fail("depth timestamp check raised an exception",
                 traceback.format_exc().splitlines()[-1])

    try:
        lag_meta = check_lag_metadata(rep, lr_root, takes_c, taus)
    except Exception:
        lag_meta = None
        rep.fail("timestamp-correction metadata check raised an exception",
                 traceback.format_exc().splitlines()[-1])

    try:
        phys = check_state_action_physics(rep, eps_c, takes_c, data, taus,
                                          args.max_frames, info.get("fps", FPS))
    except Exception:
        phys = None
        rep.fail("state/action physics check raised an exception",
                 traceback.format_exc().splitlines()[-1])

    # ---- summary --------------------------------------------------------
    section("SUMMARY")
    print(f"  raw takes                 : {len(all_takes)} found, "
          f"{len(all_takes) - len(takes)} excluded/clipped, {len(takes)} included")
    print(f"  episodes in dataset       : {info.get('total_episodes')}")
    print(f"  frames in dataset         : {info.get('total_frames')}  (expected {expected_total})")
    print(f"  max |d observation.state| : {dev_s:.3e}   (tol {args.tol:g})")
    print(f"  max |d action|            : {dev_a:.3e}   (tol {args.tol:g})")
    if corr is not None and len(corr):
        print(f"  RGB video corr (n={len(corr):3d})   : min {corr.min():.5f}  "
              f"mean {corr.mean():.5f}  (fail below {args.video_fail_corr})")
    else:
        print("  RGB video corr            : not evaluated")
    if depth_num:
        for key in DEPTH_KEYS:
            A = depth_num.get(key) or {}
            if A.get("n"):
                print(f"  {key.split('.')[-1]:11s} fidelity  : n={A['n']} frames  "
                      f"median<= {A['med_max']:g} mm  p99<= {A['p99_max']:g} mm  "
                      f"max {A['abs_max']:g} mm  raw-zero ok {A['zero_frac_min'] * 100:.4f}%")
    else:
        print("  depth fidelity            : not evaluated")
    if depth_ts:
        for cam, v in depth_ts.items():
            print(f"  {cam} depth |dt|           : max {v['max_dt_ms']:.2f} ms, median offset "
                  f"{v['median_signed_offset_ms']:+.2f} ms; worst-episode in-tol "
                  f"{v['min_in_tol_frac_soft'] * 100:.2f}% @{v['soft_tol_ms']:.2f} ms / "
                  f"{v['min_in_tol_frac_hard'] * 100:.2f}% @{v['hard_tol_ms']:.2f} ms")
    if phys:
        print(f"  |state-action| corrected  : median {phys['corrected']['median']:.5f} rad, "
              f"max {phys['corrected']['max']:.5f} rad  (gate < {phys['gate_rad']} rad)")
        print(f"  |state-action| at tau = 0 : median {phys['tau0']['median']:.5f} rad, "
              f"max {phys['tau0']['max']:.5f} rad  (comparison only)")
    print(f"  ur_joint_states lag (tau) : {tau_values} s "
          f"(source: {args.lag_json or '--ur-lag-s'})")
    print(f"  task                      : {args.task!r}")
    print()
    print("  RESULT TABLE")
    print("  " + "-" * 74)
    print(f"  {'status':6s}  check")
    print("  " + "-" * 74)
    for status, name, detail in rep.rows:
        print(f"  {status:6s}  {name}")
    print("  " + "-" * 74)
    print()
    print(f"  checks: {rep.n_pass} PASS, {rep.n_fail} FAIL, {rep.n_warn} WARN, {rep.n_skip} SKIP")
    print(f"  {len(rep.rows)} checks, {rep.n_fail} failures")
    verdict = "PASS" if rep.n_fail == 0 else "FAIL"
    print()
    print(f"  OVERALL: {verdict}")
    print("=" * 78)

    if args.json_path:
        results = {
            "verdict": verdict,
            "interpreter": sys.executable,
            "raw_root": raw_root,
            "lerobot_root": lr_root,
            "repo_id": args.repo_id,
            "task": args.task,
            "excluded": args.exclude,
            "test_mode": {"max_frames": args.max_frames, "max_takes": args.max_takes},
            "n_takes_found": len(all_takes),
            "n_takes_included": len(takes),
            "takes": [os.path.basename(t) for t in takes],
            "take_frame_counts": [int(x) for x in take_lens],
            "expected_total_frames": expected_total,
            "info": {k: info.get(k) for k in
                     ("codebase_version", "fps", "total_episodes", "total_frames",
                      "total_tasks", "robot_type", "splits")},
            "max_abs_dev_state": dev_s,
            "max_abs_dev_action": dev_a,
            "rgb_video_corr": ({"n": int(len(corr)), "min": float(corr.min()),
                                "mean": float(corr.mean())} if corr is not None and len(corr)
                               else None),
            "depth_feature_info": depth_feat,
            "depth_counts": depth_counts,
            "depth_fidelity": depth_num,
            "depth_timestamps": depth_ts,
            "depth_cameras_json": cams_json,
            "source_takes_json": src_json,
            "ur_lag_s": args.ur_lag_s,
            "lag_json": (os.path.abspath(args.lag_json) if args.lag_json else None),
            "tau_per_take": taus,
            "lag_metadata": lag_meta,
            "state_action_physics": phys,
            "thresholds": {
                "tol": args.tol,
                "depth_err_p99_mm": DEPTH_ERR_P99_MM,
                "depth_err_max_mm": DEPTH_ERR_MAX_MM,
                "depth_zero_decode_max_mm": DEPTH_ZERO_DECODE_MAX_MM,
                "depth_zero_frac": DEPTH_ZERO_FRAC,
                "depth_dt_soft_tol_ms": args.depth_dt_tol_ms,
                "depth_dt_hard_tol_ms": args.depth_dt_hard_tol_ms,
                "depth_dt_frac": args.depth_dt_frac,
                "video_fail_corr": args.video_fail_corr,
                "video_warn_corr": args.video_warn_corr,
            },
            "counts": {"pass": rep.n_pass, "fail": rep.n_fail,
                       "warn": rep.n_warn, "skip": rep.n_skip,
                       "total": len(rep.rows)},
            "rows": [{"status": s, "check": n, "detail": d} for s, n, d in rep.rows],
        }
        with open(args.json_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"  wrote {args.json_path}")

    return 0 if rep.n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
