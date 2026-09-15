#!/usr/bin/env python3
"""
Offline validator: "orange bowl in purple bowl" LeRobot v3.0 dataset vs raw
HDF5/mp4 takes.

Verifies that `orange_bowl_in_purple_bowl_lerobot_v3` is a faithful rendering of
the raw takes in `Orange_bowl_in_purple_bowl/take_*/{vectors.h5,cam1.mp4,cam2.mp4}`
(NO depth.h5 in this corpus).

INDEPENDENCE
------------
This script deliberately does NOT import (or read) the converter
(scripts/dataset/convert_takes_to_lerobot.py). Every expected value is
re-implemented here from LEROBOT_RECIPE.md, so a disagreement between the
converter and this file surfaces as a FAIL instead of cancelling out. The one
exception is *decoding*: lerobot's own `LeRobotDataset` is used to read the AV1
video pixels back out, because the encode/decode round trip is a property of the
lerobot library, not of the converter's logic.

Modeled on validate_carrot_conversion.py (same Report/JSON plumbing, same parquet
readers, same nearest-timestamp re-implementation approach), with the depth
checks and the ur_joint_states lag / stale-tail-drop machinery removed entirely:
this corpus has real ROS header stamps (`stamp_s`) on every relevant stream, so
alignment uses the recipe's `t0_off = median(ur.stamp_s - ur.t_rel_s)` shift
instead of a hand-tuned recorder-lag correction, and no frames are ever dropped.

INTERPRETER
-----------
Run this under the lerobot venv (has lerobot, h5py, cv2, numpy, pyarrow):

    /home/laptop3/youngwoong_ws/lr_env/bin/python

The Hugging Face Hub is never contacted: HF_HUB_OFFLINE / HF_DATASETS_OFFLINE /
TRANSFORMERS_OFFLINE are forced on before lerobot is imported.

SPEC (re-implemented here from LEROBOT_RECIPE.md, 2026-09-16)
---------------------------------------------------------------
  t0_off              = median(ur_joint_states.stamp_s - ur_joint_states.t_rel_s)
  cam1_cap[i]         = cam1_frames.stamp_s[i] - t0_off        (master clock, N = len(cam1_frames))
  ur_t                = ur_joint_states.stamp_s - t0_off
  cam2_t              = cam2_frames.stamp_s - t0_off
  cmd_t, grip_t       = command.t_rel_s, gripper.t_rel_s        (no header stamp -> callback time)
  nearest(src_t, q):  j = searchsorted(src_t, q); j = clip(j, 1, len-1);
                       pick j-1 if (q - src_t[j-1]) <= (src_t[j] - q) else j; clip to [0, len-1]
  observation.state[i] (float32, 7) = [ur q1..q6 @ nearest(ur_t, cam1_cap[i]),
                                        gripper.grip_pos @ nearest(grip_t, cam1_cap[i])]
  action[i]             (float32, 7) = [command cmd1..6 @ nearest(cmd_t, cam1_cap[i]),
                                        ffill_bfill(gripper.grip_cmd) @ nearest(grip_t, cam1_cap[i])]
  observation.images.cam1[i] = decoded cam1.mp4 frame i (BGR->RGB, HWC uint8, 720x1280)
  observation.images.cam2[i] = decoded cam2.mp4 frame nearest(cam2_t, cam1_cap[i])
  LeRobot timestamps = i / 30 (regular grid); episode = one take, in take-name (sorted) order.
  NO frame dropping, NO lag subtraction (the 0.14 s joint lag behind command is real
  servo tracking and must stay in the data).

Checks (each prints PASS / FAIL / WARN / SKIP, numbers recorded in --json):
  0  self-test of this file's nearest-timestamp + ffill implementations (not one of
     the 10 checks the caller asked for; kept because it is what makes checks 4/7/9/10
     trustworthy re-derivations rather than "the validator's own bug, twice")
  1  meta/info.json: codebase_version v3.0, fps 30, totals, feature set EXACTLY
     {action, observation.state, observation.images.cam1/cam2 video, lerobot
     bookkeeping keys} -- no more, no less (so a stray depth feature is a FAIL, not a
     WARN), robot_type present
  2  tasks: exactly one task string == --task; every frame's task_index resolves to it
  3  episode <-> take mapping from meta/source_takes.json
     (schema "gello_recorder/lerobot_source_takes/2"): entry count, take order,
     take existence under --raw, n_frames agreement (declared == raw cam1 rows ==
     parquet episode length)
  4  numeric fidelity: re-derive observation.state/action for EVERY episode per the
     recipe and diff against the parquet (float32 cast both sides); PASS iff
     max|delta| <= --tol (default 0.0, i.e. exact)
  5  timestamps: parquet timestamp == frame_index/30 within 1e-4; frame_index
     contiguous (0..len-1) per episode
  6  no NaN/Inf in stored state/action/timestamp
  7  video <-> raw: --sample-frames frames per episode (first/last/evenly spaced),
     LeRobotDataset[idx] decode vs raw cam1.mp4/cam2.mp4 (cv2, BGR->RGB) -- cam2 uses
     the recipe's nearest(cam2_t, cam1_cap) index, not the same frame number as cam1.
     PASS iff every sampled frame has corr >= --min-corr (0.97) and MAD <= --max-mad
     (12; AV1 re-encode tolerance)
  8  no absolute local path strings ("/home/") in any meta/**/*.{json,jsonl,md} file
  9  physics sanity (report only, PASS unless absurd): median over all frames of
     |state[:6] - action[:6]| -- expect ~0.01 rad (0.14 s servo lag), NOT ~0 and NOT
     ~0.06 like the uncorrected carrot corpus; per-episode median cam2 reuse count
     (independently re-derived, cross-referenced against source_takes.json's
     declared value when present)
 10  alignment evidence (report only): per episode, median signed (cam1 stamp -
     nearest ur stamp) in ms; fraction of master frames whose nearest ur sample is
     farther than 15 ms (should be ~0 at ~100 Hz)

Usage
-----
  PY=/home/laptop3/youngwoong_ws/lr_env/bin/python
  $PY scripts/dataset/validate_takes_conversion.py \
      --raw     /home/laptop3/youngwoong_ws/Orange_bowl_in_purple_bowl \
      --lerobot /home/laptop3/youngwoong_ws/orange_bowl_in_purple_bowl_lerobot_v3 \
      --task    "put the orange bowl into the purple bowl" \
      --json    scripts/dataset/orange_bowl_in_purple_bowl_validation_2026-09-16.json

TEST-ONLY KNOB (`--limit`)
---------------------------
A partial dry-run dataset (e.g. the first 2 takes converted for a smoke test)
legitimately has fewer episodes/frames than the full corpus. `--limit N` makes
this validator mirror that: only the first N raw takes (sorted) are considered,
so checks 1/3/4/7/9/10 evaluate the truncated prefix instead of the full corpus.
NEVER pass it when validating the real release.

Exit code 0 = all hard checks passed, 1 = at least one FAIL, 2 = could not run.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import traceback

import numpy as np

# Never contact the Hub. Set before lerobot / huggingface_hub is imported anywhere.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# --------------------------------------------------------------------------
# expected schema (LEROBOT_RECIPE.md)
# --------------------------------------------------------------------------
STATE_NAMES = ["ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6", "grip_pos"]
ACTION_NAMES = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd6", "grip_cmd"]
HWC = ["height", "width", "channels"]

EXPECTED_FEATURES = {
    "action": ("float32", [7], ACTION_NAMES),
    "observation.state": ("float32", [7], STATE_NAMES),
    "observation.images.cam1": ("video", [720, 1280, 3], HWC),
    "observation.images.cam2": ("video", [720, 1280, 3], HWC),
    "timestamp": ("float32", [1], None),
    "frame_index": ("int64", [1], None),
    "episode_index": ("int64", [1], None),
    "index": ("int64", [1], None),
    "task_index": ("int64", [1], None),
}
RGB_KEYS = ["observation.images.cam1", "observation.images.cam2"]
FPS = 30
RAW_FILES = ("vectors.h5", "cam1.mp4", "cam2.mp4")
SOURCE_TAKES_SCHEMA = "gello_recorder/lerobot_source_takes/2"

# CHECK 7 gates: AV1 (lerobot) vs raw MPEG-4, re-encode is lossy.
VIDEO_MIN_CORR = 0.97
VIDEO_MAX_MAD = 12.0

# CHECK 9/10 "absurd" gates. These two checks are explicitly "report only" in the
# brief -- they exist to catch a badly broken alignment/physics story, not to
# enforce a tight scientific bound. See each check's docstring for the reasoning
# behind the specific numbers.
STATE_ACTION_ABSURD_LOW = 1e-4    # median < this looks like a zero-order-hold bug
STATE_ACTION_ABSURD_HIGH = 0.3    # median > this looks like a time-base error
ALIGN_FAR_TOL_S = 0.015           # "farther than 15 ms" per the brief
ALIGN_MEDIAN_ABSURD_MS = 20.0     # per-episode median offset this large is not plausible at ~100 Hz
ALIGN_FAR_FRAC_ABSURD = 0.10      # >10% of frames farther than 15ms is not plausible at ~100 Hz


# --------------------------------------------------------------------------
# report plumbing (same shape as validate_carrot_conversion.py)
# --------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.rows = []  # (status, name, detail)

    def _add(self, status, name, detail):
        self.rows.append((status, name, detail))
        print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)

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
# resampling -- independent re-implementation of LEROBOT_RECIPE.md's spec
# --------------------------------------------------------------------------
def nearest_idx(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """
    For each query timestamp return the index of the nearest sample in `src_t`
    (assumed sorted ascending). Ties resolve to the earlier (left) sample.
    Queries outside [src_t[0], src_t[-1]] clamp to the first/last sample.

    Written from LEROBOT_RECIPE.md's "nearest(src_t, q)" definition, not copied
    from the converter.
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


# --------------------------------------------------------------------------
# raw take reading (h5py only, no ROS, no converter import)
# --------------------------------------------------------------------------
def cam1_row_count(h5_path: str) -> int:
    import h5py

    with h5py.File(h5_path, "r") as f:
        return int(f["cam1_frames"]["stamp_s"].shape[0])


def load_take_vectors(h5_path: str) -> dict:
    """Read every raw column the recipe needs, as plain float64 numpy arrays."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        return dict(
            cam1_stamp=f["cam1_frames"]["stamp_s"][:].astype(np.float64),
            cam2_stamp=f["cam2_frames"]["stamp_s"][:].astype(np.float64),
            ur_stamp=f["ur_joint_states"]["stamp_s"][:].astype(np.float64),
            ur_trel=f["ur_joint_states"]["t_rel_s"][:].astype(np.float64),
            ur_q=np.stack(
                [f["ur_joint_states"][f"q{k + 1}"][:] for k in range(6)], axis=1
            ).astype(np.float64),
            cmd_t=f["command"]["t_rel_s"][:].astype(np.float64),
            cmd=np.stack(
                [f["command"][f"cmd{k + 1}"][:] for k in range(6)], axis=1
            ).astype(np.float64),
            grip_t=f["gripper"]["t_rel_s"][:].astype(np.float64),
            grip_pos=f["gripper"]["grip_pos"][:].astype(np.float64),
            grip_cmd=f["gripper"]["grip_cmd"][:].astype(np.float64),
        )


def derive_take(h5_path: str) -> dict:
    """
    Independent re-derivation of observation.state / action / cam2 index, per
    LEROBOT_RECIPE.md's Alignment section.

    Returns:
      cam1_cap  (N,) float64  -- capture instant of each master frame, t_rel frame
      t0_off    float         -- median(ur.stamp_s - ur.t_rel_s) for this take
      state     (N,7) float32
      action    (N,7) float32
      cam2_idx  (N,) int64    -- nearest cam2 frame per master frame
      ur_idx    (N,) int64    -- nearest ur sample per master frame (check 10)
      ur_t      (N_ur,) float64
    """
    raw = load_take_vectors(h5_path)
    t0_off = float(np.median(raw["ur_stamp"] - raw["ur_trel"]))
    cam1_cap = raw["cam1_stamp"] - t0_off
    ur_t = raw["ur_stamp"] - t0_off
    cam2_t = raw["cam2_stamp"] - t0_off

    ur_idx = nearest_idx(ur_t, cam1_cap)
    grip_idx = nearest_idx(raw["grip_t"], cam1_cap)
    cmd_idx = nearest_idx(raw["cmd_t"], cam1_cap)
    cam2_idx = nearest_idx(cam2_t, cam1_cap)

    n = len(cam1_cap)
    state = np.zeros((n, 7), dtype=np.float32)
    state[:, :6] = raw["ur_q"][ur_idx]
    state[:, 6] = raw["grip_pos"][grip_idx]

    action = np.zeros((n, 7), dtype=np.float32)
    action[:, :6] = raw["cmd"][cmd_idx]
    action[:, 6] = ffill_bfill(raw["grip_cmd"])[grip_idx]

    return {
        "cam1_cap": cam1_cap,
        "t0_off": t0_off,
        "state": state,
        "action": action,
        "cam2_idx": cam2_idx,
        "ur_idx": ur_idx,
        "ur_t": ur_t,
    }


# --------------------------------------------------------------------------
# parquet reading (pyarrow preferred, pandas fallback) -- generic lerobot v3.0
# layout readers, not converter-specific
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
def decode_raw_frames(mp4_path: str, wanted_indices):
    """Sequentially decode `mp4_path` -> {idx: RGB uint8 HWC frame}."""
    import cv2

    wanted = sorted(set(int(i) for i in wanted_indices))
    if not wanted:
        return {}
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    out = {}
    i = 0
    stop = wanted[-1]
    want = set(wanted)
    while i <= stop:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def _chw_to_hwc_uint8(x) -> np.ndarray:
    """LeRobotDataset image items are torch tensors, CHW float32 in [0, 1]
    (see lerobot/datasets/video_utils.py: `closest_frames.type(torch.float32) / 255`).
    Convert back to HWC uint8 [0, 255] to compare against a raw cv2 frame."""
    arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.size and arr.max() <= 1.0 + 1e-6:
        arr = arr * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def to_gray_downsampled(img: np.ndarray, factor: int = 4) -> np.ndarray:
    """Box-downsample to grayscale: speeds up correlation and damps per-pixel AV1
    ringing noise that would otherwise dominate a 1:1 pixel comparison."""
    gray = img[..., :3].astype(np.float64).mean(axis=-1) if img.ndim == 3 else img.astype(np.float64)
    h, w = gray.shape
    h2, w2 = h - (h % factor), w - (w % factor)
    gray = gray[:h2, :w2]
    return gray.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


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


def sample_frame_indices(n: int, m: int):
    """m frames spread over [0, n-1] via linspace; endpoints (first/last) always
    included, the rest evenly spaced."""
    m = max(1, min(int(m), n))
    return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, m)))


def open_lerobot_dataset(repo_id: str, lr_root: str, rep: Report):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        rep.fail("lerobot importable in this interpreter", str(exc)[:200])
        return None
    try:
        return LeRobotDataset(repo_id, root=lr_root)
    except Exception as exc:
        rep.fail("LeRobotDataset could not open the dataset", str(exc)[:300])
        return None


# --------------------------------------------------------------------------
# CHECK 0 -- self-test (not one of the 10 required checks; strengthens trust
# in checks 4/7/9/10 which all depend on nearest_idx/ffill_bfill)
# --------------------------------------------------------------------------
def self_test(rep: Report, takes):
    section("CHECK 0  --  self-test of nearest_idx / ffill_bfill (independent re-implementation)")
    import h5py

    tk = takes[0]
    with h5py.File(os.path.join(tk, "vectors.h5"), "r") as f:
        cam1_t = f["cam1_frames"]["t_rel_s"][:]
        ur_t = f["ur_joint_states"]["t_rel_s"][:]
        cam2_t = f["cam2_frames"]["t_rel_s"][:]
    q = cam1_t[:: max(1, len(cam1_t) // 60)]
    ok1 = np.array_equal(nearest_idx(ur_t, q), nearest_idx_bruteforce(ur_t, q))
    ok2 = np.array_equal(nearest_idx(cam2_t, q), nearest_idx_bruteforce(cam2_t, q))
    rep.check(ok1 and ok2, "vectorised nearest_idx == brute-force argmin reference",
              f"{len(q)} queries against ur/cam2 timelines of {os.path.basename(tk)}")

    v = np.array([np.nan, np.nan, 1.0, np.nan, np.nan, 2.0, np.nan])
    want = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    rep.check(np.array_equal(ffill_bfill(v), want), "ffill_bfill reference case",
              f"{ffill_bfill(v).tolist()}")


# --------------------------------------------------------------------------
# CHECK 1 -- meta/info.json
# --------------------------------------------------------------------------
def check_info(rep: Report, info: dict, n_takes: int, expected_total_frames: int, task: str):
    section("CHECK 1  --  meta/info.json")

    rep.check(info.get("codebase_version") == "v3.0",
              "codebase_version == 'v3.0'", f"got {info.get('codebase_version')!r}")
    rep.check(int(info.get("fps", -1)) == FPS, f"fps == {FPS}", f"got {info.get('fps')!r}")
    rep.check(int(info.get("total_episodes", -1)) == n_takes,
              f"total_episodes == {n_takes} (included takes)",
              f"got {info.get('total_episodes')}")
    rep.check(int(info.get("total_frames", -1)) == expected_total_frames,
              f"total_frames == {expected_total_frames} (sum of cam1_frames rows)",
              f"got {info.get('total_frames')}")
    rep.check(int(info.get("total_tasks", -1)) == 1, "total_tasks == 1",
              f"got {info.get('total_tasks')}")

    feats = info.get("features", {})
    got_keys = set(feats.keys())
    want_keys = set(EXPECTED_FEATURES.keys())
    missing = sorted(want_keys - got_keys)
    extra = sorted(got_keys - want_keys)
    rep.check(not missing and not extra,
              "feature set EXACTLY {action, observation.state, cam1/cam2 video, "
              "lerobot bookkeeping} -- no more, no less",
              (f"missing={missing} extra={extra}") if (missing or extra)
              else f"{len(feats)} features, exact match")

    depth_like = sorted(k for k in got_keys if "depth" in k.lower())
    rep.check(not depth_like, "no depth feature present (this corpus has no depth.h5)",
              f"found: {depth_like}" if depth_like else "none")

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
        rep.check(ok, f"feature {name} dtype/shape/names", detail)

    rt = info.get("robot_type")
    rep.check(bool(rt), "robot_type present (non-empty)", f"got {rt!r}")
    return feats


# --------------------------------------------------------------------------
# CHECK 2 -- task string
# --------------------------------------------------------------------------
def check_task(rep: Report, lr_root, eps, data, task: str):
    section("CHECK 2  --  task string")
    tasks = load_tasks(lr_root)
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


# --------------------------------------------------------------------------
# CHECK 3 -- meta/source_takes.json (episode <-> take mapping)
# --------------------------------------------------------------------------
def check_source_takes(rep: Report, lr_root, raw_root, takes, take_cam1_rows, eps, task):
    section("CHECK 3  --  episode <-> take mapping (meta/source_takes.json)")
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

    if not rep.check(isinstance(obj, dict), "source_takes.json top level is an object",
                      f"got {type(obj).__name__}"):
        return None

    rep.check(obj.get("schema") == SOURCE_TAKES_SCHEMA,
              f"schema == {SOURCE_TAKES_SCHEMA!r}", f"got {obj.get('schema')!r}")
    if "task" in obj:
        rep.check(obj.get("task") == task, "declared 'task' matches --task",
                  f"got {obj.get('task')!r}")

    episodes = obj.get("episodes")
    want_names = [os.path.basename(t) for t in takes]
    ok_count = isinstance(episodes, list) and len(episodes) == len(takes)
    rep.check(ok_count, f"episodes list has {len(takes)} entries",
              f"got {len(episodes) if isinstance(episodes, list) else type(episodes).__name__}")
    if not isinstance(episodes, list):
        return {"schema": obj.get("schema"), "n_episodes": 0, "rows": []}

    got_names = [os.path.basename(str(e.get("take"))) if isinstance(e, dict) else None
                 for e in episodes]
    order_ok = got_names == want_names
    if order_ok:
        order_detail = f"{len(got_names)} entries, order matches"
    else:
        i = next((i for i, (a, b) in enumerate(zip(got_names, want_names)) if a != b), None)
        order_detail = (f"{len(got_names)} entries (want {len(want_names)})"
                        + (f"; first diff at {i}: got {got_names[i]!r} want {want_names[i]!r}"
                           if i is not None else ""))
    rep.check(order_ok, "take order == sorted raw take_* dirs", order_detail)

    missing_dirs = [nm for nm in got_names if nm and not os.path.isdir(os.path.join(raw_root, nm))]
    rep.check(not missing_dirs, "every declared take basename exists under --raw",
              f"{missing_dirs[:8]}" if missing_dirs else f"{len(got_names)} takes")

    bad_n = []
    rows = []
    for i, e in enumerate(episodes):
        if not isinstance(e, dict):
            bad_n.append(f"ep{i}: entry is not an object")
            continue
        nm = os.path.basename(str(e.get("take", "")))
        n_decl = e.get("n_frames")
        n_raw = take_cam1_rows[i] if i < len(take_cam1_rows) else None
        n_par = int(eps[i]["length"]) if i < len(eps) else None
        if n_decl != n_raw or n_decl != n_par:
            bad_n.append(f"ep{i} ({nm}): declared={n_decl} raw_cam1_rows={n_raw} parquet_len={n_par}")
        ei_decl = e.get("episode_index")
        if ei_decl is not None and int(ei_decl) != i:
            bad_n.append(f"ep{i} ({nm}): episode_index field == {ei_decl}, expected {i}")
        rows.append({
            "episode_index": i, "take": nm,
            "n_frames_declared": n_decl, "n_frames_raw": n_raw, "n_frames_parquet": n_par,
            "t0_off_s": e.get("t0_off_s"),
            "cam1_stamp_minus_ur_stamp_median_s": e.get("cam1_stamp_minus_ur_stamp_median_s"),
            "cam2_frame_reuse_count": e.get("cam2_frame_reuse_count"),
        })
    rep.check(not bad_n,
              "n_frames per episode == declared == raw cam1 rows == parquet episode length "
              "(episode_index field consistent)",
              "; ".join(bad_n[:8]) if bad_n else f"{len(episodes)} episodes, all agree")

    return {"schema": obj.get("schema"), "n_episodes": len(episodes), "rows": rows}


# --------------------------------------------------------------------------
# CHECK 4 -- numeric fidelity of observation.state / action
# --------------------------------------------------------------------------
def check_numeric(rep: Report, eps, takes, data, tol: float, verbose: bool):
    section("CHECK 4  --  numeric fidelity of observation.state / action (independent re-derivation)")

    stored_state = data["observation.state"]
    stored_action = data["action"]

    max_dev_state = 0.0
    max_dev_action = 0.0
    worst = ("", 0.0)
    n_bad = 0

    for i, (ep, tk) in enumerate(zip(eps, takes)):
        d = derive_take(os.path.join(tk, "vectors.h5"))
        exp_s = d["state"]
        exp_a = d["action"]
        a = int(ep["dataset_from_index"])
        b = int(ep["dataset_to_index"])
        got_s = np.asarray(stored_state[a:b], dtype=np.float32)
        got_a = np.asarray(stored_action[a:b], dtype=np.float32)

        if got_s.shape != exp_s.shape or got_a.shape != exp_a.shape:
            rep.fail(f"ep{i} shape mismatch",
                     f"stored {got_s.shape}/{got_a.shape} vs derived {exp_s.shape}/{exp_a.shape}")
            n_bad += 1
            continue

        ds = float(np.max(np.abs(got_s.astype(np.float64) - exp_s.astype(np.float64))))
        da = float(np.max(np.abs(got_a.astype(np.float64) - exp_a.astype(np.float64))))
        max_dev_state = max(max_dev_state, ds)
        max_dev_action = max(max_dev_action, da)
        if max(ds, da) > worst[1]:
            worst = (f"ep{i} ({os.path.basename(tk)})", max(ds, da))
        if ds > tol or da > tol:
            n_bad += 1
            rep.fail(f"ep{i} ({os.path.basename(tk)}) numeric deviation",
                     f"max|dstate|={ds:.3e} max|daction|={da:.3e} (tol={tol:g})")
        elif verbose:
            print(f"    ep{i:3d} {os.path.basename(tk):32s} "
                  f"dstate={ds:.3e} daction={da:.3e}", flush=True)

    if n_bad == 0:
        rep.pass_(f"all {len(eps)} episodes re-derive with max|delta| <= {tol:g}",
                  f"max|dstate|={max_dev_state:.3e}  max|daction|={max_dev_action:.3e}")
    print(f"  [info] worst episode: {worst[0]}  max|d|={worst[1]:.3e}")
    return max_dev_state, max_dev_action


# --------------------------------------------------------------------------
# CHECK 5 -- timestamps
# --------------------------------------------------------------------------
def check_timestamps(rep: Report, info, eps, data):
    section("CHECK 5  --  timestamps")
    fps = int(info.get("fps", FPS))
    fi = data["frame_index"].astype(np.int64)
    ts = data["timestamp"].astype(np.float64)
    expected_ts = fi.astype(np.float64) / fps
    dev = float(np.max(np.abs(ts - expected_ts))) if len(ts) else 0.0
    rep.check(dev < 1e-4, f"timestamp == frame_index / {fps} (within 1e-4)", f"max dev {dev:.3e}")

    ok_fi = True
    detail = ""
    for i, ep in enumerate(eps):
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        if not np.array_equal(fi[a:b], np.arange(b - a)):
            ok_fi = False
            detail = f"ep{i} frame_index not 0..{b - a - 1}"
            break
    rep.check(ok_fi, "per-episode frame_index contiguous (restarts at 0, +1 each frame)",
              detail if not ok_fi else f"{len(eps)} episodes")


# --------------------------------------------------------------------------
# CHECK 6 -- NaN / Inf
# --------------------------------------------------------------------------
def check_nan(rep: Report, data):
    section("CHECK 6  --  no NaN / Inf in stored state / action / timestamps")
    for key in ("observation.state", "action", "timestamp"):
        arr = np.asarray(data[key], dtype=np.float64)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        rng = (f"[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]" if arr.size else "[]")
        rep.check(n_nan == 0 and n_inf == 0, f"{key} finite", f"NaN={n_nan} Inf={n_inf}  range={rng}")


# --------------------------------------------------------------------------
# CHECK 7 -- video <-> raw
# --------------------------------------------------------------------------
def check_videos(rep: Report, ds, eps, takes, n_sample_frames: int,
                 min_corr: float = VIDEO_MIN_CORR, max_mad: float = VIDEO_MAX_MAD):
    section("CHECK 7  --  RGB video integrity (LeRobot decode vs raw cam1.mp4/cam2.mp4)")
    if ds is None:
        rep.skip("cam1 video vs raw", "LeRobotDataset not available")
        rep.skip("cam2 video vs raw", "LeRobotDataset not available")
        return None

    stats = {"observation.images.cam1": [], "observation.images.cam2": []}
    bad = {"observation.images.cam1": [], "observation.images.cam2": []}

    for i, (ep, tk) in enumerate(zip(eps, takes)):
        a = int(ep["dataset_from_index"])
        n = int(ep["length"])
        name = os.path.basename(tk)
        d = derive_take(os.path.join(tk, "vectors.h5"))
        cam2_idx = d["cam2_idx"]
        ks = sample_frame_indices(n, n_sample_frames)

        raw1 = decode_raw_frames(os.path.join(tk, "cam1.mp4"), ks)
        want2 = sorted(set(int(cam2_idx[k]) for k in ks))
        raw2 = decode_raw_frames(os.path.join(tk, "cam2.mp4"), want2)

        for k in ks:
            try:
                item = ds[a + k]
            except Exception as exc:
                bad["observation.images.cam1"].append(f"ep{i} {name} f{k}: ds[{a + k}] raised {exc}")
                continue

            if k in raw1:
                got1 = _chw_to_hwc_uint8(item["observation.images.cam1"])
                ref1 = raw1[k]
                if got1.shape != ref1.shape:
                    bad["observation.images.cam1"].append(
                        f"ep{i} {name} f{k}: shape {got1.shape} vs {ref1.shape}")
                else:
                    c1 = correlation(to_gray_downsampled(got1), to_gray_downsampled(ref1))
                    m1 = mean_abs_diff(got1, ref1)
                    stats["observation.images.cam1"].append((i, k, c1, m1))
                    if c1 < min_corr or m1 > max_mad:
                        bad["observation.images.cam1"].append(
                            f"ep{i} {name} f{k}: corr={c1:.4f} mad={m1:.2f}")
            else:
                bad["observation.images.cam1"].append(f"ep{i} {name} f{k}: raw frame not decodable")

            ci = int(cam2_idx[k])
            if ci in raw2:
                got2 = _chw_to_hwc_uint8(item["observation.images.cam2"])
                ref2 = raw2[ci]
                if got2.shape != ref2.shape:
                    bad["observation.images.cam2"].append(
                        f"ep{i} {name} f{k}(->raw{ci}): shape {got2.shape} vs {ref2.shape}")
                else:
                    c2 = correlation(to_gray_downsampled(got2), to_gray_downsampled(ref2))
                    m2 = mean_abs_diff(got2, ref2)
                    stats["observation.images.cam2"].append((i, k, c2, m2))
                    if c2 < min_corr or m2 > max_mad:
                        bad["observation.images.cam2"].append(
                            f"ep{i} {name} f{k}(->raw{ci}): corr={c2:.4f} mad={m2:.2f}")
            else:
                bad["observation.images.cam2"].append(
                    f"ep{i} {name} f{k}(->raw{ci}): raw frame not decodable")

        print(f"    ep{i:3d} {name:32s} sampled {len(ks)} frame(s)", flush=True)

    out = {}
    for key in ("observation.images.cam1", "observation.images.cam2"):
        label = key.split(".")[-1]
        vals = stats[key]
        if not vals:
            rep.skip(f"{label}: sampled frames vs raw", "no frames compared")
            out[key] = {"n": 0}
            continue
        cs = np.array([v[2] for v in vals])
        ms = np.array([v[3] for v in vals])
        detail = (f"n={len(vals)}  corr min={cs.min():.5f} mean={cs.mean():.5f}  "
                  f"MAD max={ms.max():.2f} mean={ms.mean():.2f}  (gate corr>={min_corr}, MAD<={max_mad})")
        rep.check(not bad[key], f"{label}: every sampled frame has corr>={min_corr} and MAD<={max_mad}",
                  "; ".join(bad[key][:6]) if bad[key] else detail)
        out[key] = {"n": len(vals), "corr_min": float(cs.min()), "corr_mean": float(cs.mean()),
                    "mad_max": float(ms.max()), "mad_mean": float(ms.mean()),
                    "n_bad": len(bad[key])}
    return out


# --------------------------------------------------------------------------
# CHECK 8 -- no absolute local paths under meta/
# --------------------------------------------------------------------------
def check_no_abs_paths(rep: Report, lr_root):
    section("CHECK 8  --  no absolute local paths ('/home/') under meta/")
    meta_dir = os.path.join(lr_root, "meta")
    bad = []
    n_files = 0
    for root, _, files in os.walk(meta_dir):
        for fn in files:
            if not fn.lower().endswith((".json", ".jsonl", ".md")):
                continue
            fp = os.path.join(root, fn)
            n_files += 1
            try:
                with open(fp, "r", errors="replace") as fh:
                    text = fh.read()
            except Exception as exc:
                bad.append(f"{os.path.relpath(fp, lr_root)}: unreadable ({exc})")
                continue
            if "/home/" in text:
                idx = text.find("/home/")
                snippet = text[max(0, idx - 20):idx + 60].replace("\n", " ")
                bad.append(f"{os.path.relpath(fp, lr_root)}: ...{snippet}...")
    rep.check(not bad, f"no '/home/' substring in any meta/**/*.{{json,jsonl,md}} file "
                       f"({n_files} scanned)",
              "; ".join(bad[:8]) if bad else f"{n_files} files")
    return {"n_files_scanned": n_files, "offenders": bad}


# --------------------------------------------------------------------------
# CHECK 9 -- physics sanity (report only, PASS unless absurd)
# --------------------------------------------------------------------------
def check_physics_sanity(rep: Report, eps, takes, data, src_rows):
    section("CHECK 9  --  physics sanity (report only, PASS unless absurd)")
    stored_state = data["observation.state"]
    stored_action = data["action"]

    all_abs = []
    reuse_counts = []
    for i, (ep, tk) in enumerate(zip(eps, takes)):
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        st = np.asarray(stored_state[a:b, :6], dtype=np.float64)
        ac = np.asarray(stored_action[a:b, :6], dtype=np.float64)
        all_abs.append(np.abs(st - ac))

        d = derive_take(os.path.join(tk, "vectors.h5"))
        cam2_idx = d["cam2_idx"]
        reuse = int(len(cam2_idx) - len(np.unique(cam2_idx)))
        reuse_counts.append(reuse)

    diffs = np.concatenate(all_abs, axis=0) if all_abs else np.zeros((0, 6))
    med = float(np.median(diffs)) if diffs.size else float("nan")
    mean = float(np.mean(diffs)) if diffs.size else float("nan")
    reuse_med = float(np.median(reuse_counts)) if reuse_counts else float("nan")

    declared_reuse = None
    if src_rows:
        vals = [r.get("cam2_frame_reuse_count") for r in src_rows
                if r.get("cam2_frame_reuse_count") is not None]
        if vals:
            declared_reuse = float(np.median(vals))

    absurd = not (STATE_ACTION_ABSURD_LOW <= med <= STATE_ACTION_ABSURD_HIGH)
    detail = (f"median|state[:6]-action[:6]|={med:.5f} rad (mean {mean:.5f}); "
              f"expect ~0.01 rad (0.14s servo lag) -- NOT ~0, NOT ~0.06 like the "
              f"uncorrected carrot corpus; per-episode cam2 reuse count (re-derived) "
              f"median={reuse_med:g}"
              + (f", declared in source_takes.json median={declared_reuse:g}"
                 if declared_reuse is not None
                 else " (source_takes.json has no cam2_frame_reuse_count field to cross-check)"))
    rep.check(not absurd,
              f"median|state[:6]-action[:6]| in plausible range "
              f"[{STATE_ACTION_ABSURD_LOW:g}, {STATE_ACTION_ABSURD_HIGH:g}] rad",
              detail)
    return {
        "median_abs_state_action_rad": med,
        "mean_abs_state_action_rad": mean,
        "cam2_reuse_count_median_rederived": reuse_med,
        "cam2_reuse_count_median_declared": declared_reuse,
        "per_episode_reuse_count_rederived": reuse_counts,
    }


# --------------------------------------------------------------------------
# CHECK 10 -- alignment evidence (report only)
# --------------------------------------------------------------------------
def check_alignment_evidence(rep: Report, eps, takes):
    section("CHECK 10  --  alignment evidence (report only)")
    med_offsets_ms = []
    far_fracs = []
    rows = []
    for i, tk in enumerate(takes):
        d = derive_take(os.path.join(tk, "vectors.h5"))
        cam1_cap, ur_t, ur_idx = d["cam1_cap"], d["ur_t"], d["ur_idx"]
        signed_ms = (cam1_cap - ur_t[ur_idx]) * 1000.0
        med = float(np.median(signed_ms))
        far = float(np.mean(np.abs(signed_ms) > ALIGN_FAR_TOL_S * 1000.0))
        med_offsets_ms.append(med)
        far_fracs.append(far)
        rows.append({"episode_index": i, "take": os.path.basename(tk),
                     "median_offset_ms": med, "far_frac_gt_15ms": far})

    m = np.array(med_offsets_ms)
    fr = np.array(far_fracs)
    absurd = bool(np.max(np.abs(m)) > ALIGN_MEDIAN_ABSURD_MS or np.max(fr) > ALIGN_FAR_FRAC_ABSURD)
    detail = (f"per-episode median offset: min={m.min():.2f}ms max={m.max():.2f}ms "
              f"(absurd gate |{ALIGN_MEDIAN_ABSURD_MS:g}|ms); "
              f"far(>{ALIGN_FAR_TOL_S * 1000:g}ms) fraction: min={fr.min() * 100:.2f}% "
              f"max={fr.max() * 100:.2f}% (should be ~0 at ~100Hz, absurd gate "
              f"{ALIGN_FAR_FRAC_ABSURD * 100:.0f}%)")
    rep.check(not absurd, "cam1<->ur alignment plausible (not absurd)", detail)
    return {
        "per_episode": rows,
        "median_offset_ms_range": [float(m.min()), float(m.max())] if len(m) else None,
        "far_frac_range": [float(fr.min()), float(fr.max())] if len(fr) else None,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Validate the orange_bowl_in_purple_bowl LeRobot v3.0 dataset "
                    "against its raw HDF5/mp4 takes (independent of the converter).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw", required=True, help="directory containing take_*/ subdirectories")
    ap.add_argument("--lerobot", required=True, help="root of the LeRobot v3.0 dataset")
    ap.add_argument("--task", required=True, help="expected task string")
    ap.add_argument("--json", dest="json_path", required=True,
                    help="write a machine-readable results dict to this path")
    ap.add_argument("--sample-frames", type=int, default=6,
                    help="frames sampled per episode per camera for check 7 "
                         "(first/last/evenly spaced)")
    ap.add_argument("--repo-id", default="Bigenlight/orange_bowl_in_purple_bowl_lerobot_v3",
                    help="repo id label passed to LeRobotDataset (local read only)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max allowed abs deviation for state/action (check 4); "
                         "spec wants exact float32 equality")
    ap.add_argument("--min-corr", type=float, default=VIDEO_MIN_CORR,
                    help="check 7: minimum Pearson correlation (grayscale, downsampled)")
    ap.add_argument("--max-mad", type=float, default=VIDEO_MAX_MAD,
                    help="check 7: maximum mean absolute pixel difference")
    ap.add_argument("--limit", type=int, default=0,
                    help="TEST ONLY: only consider the first N raw takes (sorted); "
                         "mirrors a partial dry-run dataset. NEVER use for the real release")
    ap.add_argument("--skip-video", action="store_true", help="skip check 7 (slow)")
    ap.add_argument("--verbose", action="store_true", help="print per-episode numeric deviations")
    args = ap.parse_args()

    raw_root = os.path.abspath(os.path.expanduser(args.raw))
    lr_root = os.path.abspath(os.path.expanduser(args.lerobot))

    print("LeRobot v3.0 conversion validator  --  orange_bowl_in_purple_bowl")
    print(f"  interpreter    : {sys.executable}")
    print(f"  raw takes      : {raw_root}")
    print(f"  lerobot dataset: {lr_root}")
    print(f"  repo id        : {args.repo_id}")
    print(f"  expected task  : {args.task!r}")
    print(f"  numeric tol    : {args.tol:g}")
    print(f"  sample frames  : {args.sample_frames} per episode per camera (check 7)")
    if args.limit and args.limit > 0:
        print(f"  !! TEST MODE   : --limit={args.limit} "
              f"(checks 1/3/4/7/9/10 evaluated on the truncated prefix only)")

    rep = Report()

    # ---- resolve takes -------------------------------------------------
    all_takes = sorted(glob.glob(os.path.join(raw_root, "take_*")))
    all_takes = [t for t in all_takes if os.path.isdir(t)]
    if not all_takes:
        print(f"\nFATAL: no take_* directories under {raw_root}", file=sys.stderr)
        return 2
    takes = all_takes[:args.limit] if (args.limit and args.limit > 0) else all_takes
    print(f"  raw takes found: {len(all_takes)}  ->  included: {len(takes)}")

    missing_files = []
    for t in takes:
        for fn in RAW_FILES:
            if not os.path.exists(os.path.join(t, fn)):
                missing_files.append(os.path.join(os.path.basename(t), fn))
    if missing_files:
        rep.fail("raw take files present (vectors.h5/cam1.mp4/cam2.mp4)", f"{missing_files[:8]}")

    if not os.path.isdir(lr_root):
        print(f"\nFATAL: LeRobot dataset not found at {lr_root}", file=sys.stderr)
        return 2

    info_path = os.path.join(lr_root, "meta", "info.json")
    if not os.path.exists(info_path):
        print(f"\nFATAL: {info_path} not found", file=sys.stderr)
        return 2
    with open(info_path) as fh:
        info = json.load(fh)

    # ---- check 0 ---------------------------------------------------------
    self_test(rep, takes)

    print("\nreading raw cam1_frames row counts ...", flush=True)
    take_cam1_rows = [cam1_row_count(os.path.join(t, "vectors.h5")) for t in takes]
    expected_total_frames = int(sum(take_cam1_rows))
    print(f"  {len(takes)} takes, {expected_total_frames} frames expected", flush=True)

    check_info(rep, info, len(takes), expected_total_frames, args.task)

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
                 f"remaining per-episode checks may be misaligned")
        n = min(len(eps), len(takes))
        eps_c, takes_c, rows_c = eps[:n], takes[:n], take_cam1_rows[:n]
    else:
        rep.pass_("episode count == included take count", f"{len(eps)}")
        eps_c, takes_c, rows_c = eps, takes, take_cam1_rows

    check_task(rep, lr_root, eps, data, args.task)

    try:
        src = check_source_takes(rep, lr_root, raw_root, takes_c, rows_c, eps_c, args.task)
    except Exception:
        src = None
        rep.fail("source_takes.json check raised an exception",
                 traceback.format_exc().splitlines()[-1])
    src_rows = src["rows"] if src else None

    dev_s, dev_a = check_numeric(rep, eps_c, takes_c, data, args.tol, args.verbose)

    check_timestamps(rep, info, eps_c, data)

    check_nan(rep, data)

    video_summary = None
    if args.skip_video:
        section("CHECK 7  --  RGB video integrity")
        rep.skip("cam1: sampled frames vs raw", "--skip-video")
        rep.skip("cam2: sampled frames vs raw", "--skip-video")
    else:
        ds = open_lerobot_dataset(args.repo_id, lr_root, rep)
        try:
            video_summary = check_videos(rep, ds, eps_c, takes_c, args.sample_frames,
                                         args.min_corr, args.max_mad)
        except Exception:
            rep.fail("video check raised an exception",
                     traceback.format_exc().splitlines()[-1])

    abs_paths = check_no_abs_paths(rep, lr_root)

    try:
        phys = check_physics_sanity(rep, eps_c, takes_c, data, src_rows)
    except Exception:
        phys = None
        rep.fail("physics sanity check raised an exception",
                 traceback.format_exc().splitlines()[-1])

    try:
        align = check_alignment_evidence(rep, eps_c, takes_c)
    except Exception:
        align = None
        rep.fail("alignment evidence check raised an exception",
                 traceback.format_exc().splitlines()[-1])

    # ---- summary ---------------------------------------------------------
    section("SUMMARY")
    print(f"  raw takes                 : {len(all_takes)} found, "
          f"{len(all_takes) - len(takes)} excluded/clipped, {len(takes)} included")
    print(f"  episodes in dataset       : {info.get('total_episodes')}")
    print(f"  frames in dataset         : {info.get('total_frames')}  (expected {expected_total_frames})")
    print(f"  max |d observation.state| : {dev_s:.3e}   (tol {args.tol:g})")
    print(f"  max |d action|            : {dev_a:.3e}   (tol {args.tol:g})")
    if video_summary:
        for key in ("observation.images.cam1", "observation.images.cam2"):
            v = video_summary.get(key) or {}
            if v.get("n"):
                print(f"  {key.split('.')[-1]:5s} video corr (n={v['n']:4d}) : "
                      f"min {v['corr_min']:.5f} mean {v['corr_mean']:.5f}  "
                      f"MAD max {v['mad_max']:.2f} mean {v['mad_mean']:.2f}")
    else:
        print("  RGB video corr            : not evaluated")
    if phys:
        print(f"  |state-action| median     : {phys['median_abs_state_action_rad']:.5f} rad "
              f"(mean {phys['mean_abs_state_action_rad']:.5f})")
        print(f"  cam2 reuse count median   : {phys['cam2_reuse_count_median_rederived']:g} "
              f"(re-derived)")
    if align:
        lo, hi = align["median_offset_ms_range"] or (float("nan"), float("nan"))
        print(f"  cam1<->ur offset (median) : [{lo:.2f}, {hi:.2f}] ms across episodes")
    print(f"  task                       : {args.task!r}")
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
    print(f"  {rep.n_pass} passed / {rep.n_fail} failed")
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
            "test_mode": {"limit": args.limit},
            "n_takes_found": len(all_takes),
            "n_takes_included": len(takes),
            "takes": [os.path.basename(t) for t in takes],
            "take_frame_counts": [int(x) for x in take_cam1_rows],
            "expected_total_frames": expected_total_frames,
            "info": {k: info.get(k) for k in
                     ("codebase_version", "fps", "total_episodes", "total_frames",
                      "total_tasks", "robot_type", "splits")},
            "max_abs_dev_state": dev_s,
            "max_abs_dev_action": dev_a,
            "source_takes_json": src,
            "video": video_summary,
            "no_abs_paths": abs_paths,
            "physics_sanity": phys,
            "alignment_evidence": align,
            "thresholds": {
                "tol": args.tol,
                "video_min_corr": args.min_corr,
                "video_max_mad": args.max_mad,
                "state_action_absurd_low": STATE_ACTION_ABSURD_LOW,
                "state_action_absurd_high": STATE_ACTION_ABSURD_HIGH,
                "align_far_tol_s": ALIGN_FAR_TOL_S,
                "align_median_absurd_ms": ALIGN_MEDIAN_ABSURD_MS,
                "align_far_frac_absurd": ALIGN_FAR_FRAC_ABSURD,
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
