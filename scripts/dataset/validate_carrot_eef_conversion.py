#!/usr/bin/env python3
"""
Offline validator: "Put carrot in pot" **EEF-delta** LeRobot v3.0 dataset  vs  the
raw HDF5/mp4 takes.

Verifies that `carrot_in_pot_eef_lerobot` is a faithful rendering of
`Put_carrot_in_pot/take_*/{vectors.h5,cam1.mp4,cam2.mp4}` under the policy space
of `ros2_ur_ws/src/gello_policy/CARROT_EEF_DESIGN.md` section 2.  There is no
depth in this dataset and nothing here looks for any.

INDEPENDENCE  (this is the whole point of the file)
---------------------------------------------------
This script deliberately does NOT import the converter **and does NOT import
`gello_policy.eef_space`**.  Every expected number is re-derived here straight
from the DESIGN text:

  * the master clock / nearest-timestamp / ffill-bfill resampling is
    re-implemented (and self-tested against a brute-force reference, check 0);
  * the TCP is rebuilt as `ur_kin.fk(q)` followed by a +0.174 m translation
    along the flange +Z, from the constant written in the DESIGN, not imported;
  * the 6-D rotation is re-extracted as "the first two COLUMNS of R_tcp";
  * the rotation delta uses **this file's own SO(3) log map** (classic
    trace/antisymmetric formula with an atan2 angle and a near-pi eigenvector
    fallback), which is a different algorithm from `ur_kin.so3_log` (that one
    routes through a Shepperd quaternion).  A sign/convention mistake inside
    eef_space would therefore surface as a FAIL here instead of cancelling out.

`ur_gello_bringup.ur_kin` IS imported -- forward kinematics of a UR7e is the
robot's definition, not the policy space's, and re-deriving DH tables here would
test nothing.  `ur_kin.so3_log` / `so3_exp` are never called.

INTERPRETER
-----------
    /home/laptop3/youngwoong_ws/lr_env/bin/python
with PYTHONPATH=<worktree>/ros2_ur_ws/src/ur_gello_bringup (ur_kin is ROS-free).
Needs numpy + h5py + pyarrow/pandas + cv2 and ffmpeg/ffprobe on PATH.  lerobot
itself is NOT required: the RGB check streams both videos through ffmpeg as
rawvideo, which is one dependency less than trusting a reader.

The Hugging Face Hub is never contacted.

SPEC (re-implemented here)
--------------------------
  master clock = cam1_frames/t_rel_s                     (fps = 30)
  tau_q        = the per-take staleness of the recorder's joint cache, estimated
                 HERE by a different method than the converter's: a parabola fit
                 through the coarse grid minimum of
                 median_t |p_tcp(q @ nearest(ur_t, t+tau)) - p_tcp(cmd_t)|,
                 then snapped to the 0.01 s grid.  Checked against the tau_q the
                 converter recorded in meta/source_takes.json (|d| <= 0.02 s),
                 and the state/action re-derivation uses THIS file's own tau.
  q_t          = ur_joint_states q1..q6 LINEARLY INTERPOLATED at master + tau_q
                 (the joint record ticks at a jittery ~63-67 Hz; nearest-neighbour
                  would repeat/skip samples and inject fake zero and double steps
                  into the finite-difference action).  Re-implemented here by hand
                  rather than via np.interp.
  grip_pos_t   = gripper/grip_pos      @ nearest gripper t_rel_s  (no shift)
  grip_cmd_t   = ffill_bfill(gripper/grip_cmd) @ nearest gripper t_rel_s
  T_tcp(q)     = ur_kin.fk(q) @ translate(0, 0, +0.174)
  state[16]    = [q_t(6), p_tcp(3), R_tcp[:,0](3), R_tcp[:,1](3), grip_pos_t]
  action[7]    = [p_{t+1} - p_t (3), so3_log(R_{t+1} R_t^T) (3), grip_cmd_t]
                 last frame of an episode: first six components are exactly 0
  cam1 frame k = k-th decoded frame of cam1.mp4 (last frame padded if short)
  cam2 frame k = decoded frame nearest_idx(cam2_t, cam1_t)[k] of cam2.mp4

Checks (each prints PASS / FAIL / WARN / SKIP):
  0  self-test: nearest_idx vs brute force, ffill_bfill, and the log map vs a
     Rodrigues round trip written here
  1  meta/info.json: codebase_version, fps, feature dicts (16/7 + the two RGB
     videos and NOTHING else), totals, splits
  2  per-episode frame counts == source cam1_frames row counts, in take order
  3  state[:6]   == the interpolated q at master + tau_q          (|d| <= --tol)
     3b the tau_q this file estimates == the tau_q in meta/source_takes.json
        (|d| <= --tau-tol, default 0.02 s)
  4  state[6:9]  == fk(q) + 0.174 m tool offset                (|d| <= --tol)
  5  state[9:15] == first two columns of R_tcp, and orthonormal
  6  state[15]   == resampled grip_pos
  7  action[:6]  == TCP(q_{t+1}) (-) TCP(q_t); last frame exactly zero
  8  action[6]   == resampled ffill/bfill grip_cmd
  9  no NaN / Inf anywhere in state / action
 10  task string: exactly one, == --task, every task_index resolves to it
 11  RGB video integrity: sampled frames, LeRobot AV1 vs raw MPEG-4 (correlation)
 12  structural sanity: episode boundaries, index columns, timestamp == k/fps
 13  meta/eef_space.json present, action_space_id == "eef_delta_v1", tool 0.174,
     names match the stored feature names
 14  meta/source_takes.json: take order and per-take n_frames

TEST-ONLY KNOBS (`--max-frames`, `--max-takes`) mirror a truncated smoke
conversion; NEVER pass them when validating the real release.

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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ur_gello_bringup import ur_kin  # noqa: E402  (forward kinematics only)

# --------------------------------------------------------------------------
# expected schema -- transcribed from CARROT_EEF_DESIGN.md section 2
# --------------------------------------------------------------------------
TOOL_Z_M = 0.174  # DESIGN section 1: TCP = flange (+Z) 0.174 m.  NOT imported.

STATE_NAMES = [
    "ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6",
    "tcp_x", "tcp_y", "tcp_z",
    "tcp_r11", "tcp_r21", "tcp_r31", "tcp_r12", "tcp_r22", "tcp_r32",
    "grip_pos",
]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "grip_cmd"]
STATE_DIM = len(STATE_NAMES)    # 16
ACTION_DIM = len(ACTION_NAMES)  # 7
ACTION_SPACE_ID = "eef_delta_v1"

HWC = ["height", "width", "channels"]

EXPECTED_FEATURES = {
    "action": ("float32", [ACTION_DIM], ACTION_NAMES),
    "observation.state": ("float32", [STATE_DIM], STATE_NAMES),
    "observation.images.cam1": ("video", [720, 1280, 3], HWC),
    "observation.images.cam2": ("video", [720, 1280, 3], HWC),
    "timestamp": ("float32", [1], None),
    "frame_index": ("int64", [1], None),
    "episode_index": ("int64", [1], None),
    "index": ("int64", [1], None),
    "task_index": ("int64", [1], None),
}
RGB_KEYS = ["observation.images.cam1", "observation.images.cam2"]
FORBIDDEN_FEATURE_SUBSTR = ("depth",)  # this dataset must carry no depth at all

FPS = 30
# Orthonormality of the stored 6-D rotation: the columns are exact in float64 and
# stored as float32, so each component carries <= ~6e-8 absolute error; 1e-5 is
# three orders of margin while still catching a genuinely non-rotation vector.
ORTHO_TOL = 1e-5


# --------------------------------------------------------------------------
# report plumbing
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
# resampling -- independent re-implementation of the converter's spec
# --------------------------------------------------------------------------
def nearest_idx(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Index of the nearest sample in `src_t` (sorted asc) for each query time.

    Ties resolve to the earlier (left) sample; queries outside the source span
    clamp to the first/last sample.  Written from the spec, not copied.
    """
    src_t = np.asarray(src_t, dtype=np.float64)
    query_t = np.asarray(query_t, dtype=np.float64)
    n = len(src_t)
    if n == 0:
        raise ValueError("empty source timeline")
    if n == 1:
        return np.zeros(len(query_t), dtype=np.int64)
    hi = np.clip(np.searchsorted(src_t, query_t, side="left"), 1, n - 1)
    lo = hi - 1
    return np.where(query_t - src_t[lo] <= src_t[hi] - query_t, lo, hi).astype(np.int64)


def nearest_idx_bruteforce(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """O(N*M) reference used to self-verify `nearest_idx` on a small sample."""
    src_t = np.asarray(src_t, dtype=np.float64)
    out = np.empty(len(query_t), dtype=np.int64)
    for i, q in enumerate(query_t):
        out[i] = int(np.argmin(np.abs(src_t - q)))  # argmin prefers the earliest tie
    return out


def ffill_bfill(v: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaNs.  Vectorised re-implementation."""
    v = np.asarray(v, dtype=np.float64).copy()
    n = len(v)
    if n == 0:
        return v
    good = ~np.isnan(v)
    if not good.any():
        return v
    idx = np.where(good, np.arange(n), -1)
    fwd = np.maximum.accumulate(idx)
    idx_b = np.where(good, np.arange(n), n)
    bwd = np.minimum.accumulate(idx_b[::-1])[::-1]
    src = np.clip(np.where(fwd >= 0, fwd, bwd), 0, n - 1)
    return v[src]


# --------------------------------------------------------------------------
# EEF math -- re-derived here from the DESIGN text
# --------------------------------------------------------------------------
def tcp_pose(q) -> np.ndarray:
    """base_link -> TCP (4x4): the flange from ur_kin.fk, pushed +0.174 m along +Z.

    Built by translating the flange ORIGIN along the flange's own z axis, i.e.
    p_tcp = p_flange + R_flange @ [0, 0, TOOL_Z_M], R_tcp = R_flange.  This is
    the DESIGN sentence spelled out; it is deliberately NOT the matrix product
    the converter uses, though it is the same transform.
    """
    T = ur_kin.fk(np.asarray(q, dtype=float).reshape(6))
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = T[:3, 3] + R @ np.array([0.0, 0.0, TOOL_Z_M])
    return out


def so3_log_own(R: np.ndarray) -> np.ndarray:
    """SO(3) -> rotation vector.  Independent of ur_kin.so3_log.

    Classic trace / antisymmetric formulation:  the vee of (R - R^T) has norm
    2*sin(theta) and direction = the rotation axis, and (trace(R) - 1)/2 =
    cos(theta), so theta = atan2(s, c) is well conditioned everywhere except
    theta -> pi, where the antisymmetric part vanishes and we fall back to the
    +1 eigenvector extracted from (R + I)/2 with the sign taken from the
    antisymmetric part.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    s = float(np.linalg.norm(v))                      # sin(theta)
    c = float((np.trace(R) - 1.0) * 0.5)              # cos(theta)
    theta = math.atan2(s, c)
    if theta < 1e-12:
        return np.zeros(3)
    if theta < math.pi - 1e-6:
        return (theta / s) * v
    # theta ~ pi: (R + I)/2 = a a^T, so its largest-norm column is a scaled axis.
    M = (R + np.eye(3)) * 0.5
    k = int(np.argmax(np.linalg.norm(M, axis=0)))
    axis = M[:, k]
    axis = axis / np.linalg.norm(axis)
    if s > 0.0 and float(np.dot(axis, v)) < 0.0:
        axis = -axis
    return theta * axis


def so3_exp_own(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential, used ONLY by the check-0 self-test of the log map."""
    w = np.asarray(w, dtype=float).reshape(3)
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


TAU_GRID = np.round(np.arange(0.0, 1.5 + 1e-9, 0.01), 6)


def _read_streams(h5_path: str):
    """Raw streams needed by both the tau estimate and the state/action rebuild."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        out = {
            "cam1_t": f["cam1_frames"]["t_rel_s"][:],
            "cam2_t": f["cam2_frames"]["t_rel_s"][:],
            "ur_t": f["ur_joint_states"]["t_rel_s"][:],
            "ur_q": np.stack([f["ur_joint_states"][f"q{k + 1}"][:] for k in range(6)], axis=1),
            "cmd_t": f["command"]["t_rel_s"][:],
            "cmd_q": np.stack([f["command"][f"cmd{k + 1}"][:] for k in range(6)], axis=1),
            "grip_t": f["gripper"]["t_rel_s"][:],
            "grip_pos": f["gripper"]["grip_pos"][:],
            "grip_cmd": f["gripper"]["grip_cmd"][:],
        }
    return out


def estimate_tau_own(st) -> dict:
    """Independently estimate the staleness of this take's joint cache, in seconds.

    The converter scores the shift on the COMMAND sample clock; this scores it on
    the CAM1 MASTER clock (for every master timestamp t, compare the TCP of the
    joint sample nearest t+tau with the TCP of the command sample nearest t) --
    a different sample set and a different resampling of the command stream, so
    the two estimates are not the same computation.  A 3-point parabola through
    the grid minimum is reported as well, but only the grid value is used to
    rebuild state/action, because the converter also sampled on the grid.
    """
    ur_t, ur_q = st["ur_t"], st["ur_q"]
    cam1_t = st["cam1_t"]
    p_ur = np.stack([tcp_pose(ur_q[i])[:3, 3] for i in range(len(ur_q))])
    cmd_j = nearest_idx(st["cmd_t"], cam1_t)
    p_cmd = np.stack([tcp_pose(st["cmd_q"][j])[:3, 3] for j in cmd_j])

    keep = (cam1_t + TAU_GRID[-1] <= ur_t[-1]) & (cam1_t + TAU_GRID[0] >= ur_t[0])
    t_eval, p_eval = cam1_t[keep], p_cmd[keep]
    med = np.array([
        float(np.median(np.linalg.norm(p_ur[nearest_idx(ur_t, t_eval + tau)] - p_eval, axis=1)))
        for tau in TAU_GRID
    ])
    k = int(np.argmin(med))
    tau_grid, residual = float(TAU_GRID[k]), float(med[k])

    tau_refined = tau_grid
    if 0 < k < len(med) - 1:
        y0, y1, y2 = med[k - 1], med[k], med[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom > 0:
            tau_refined = tau_grid + 0.01 * 0.5 * (y0 - y2) / denom

    return {"tau": tau_grid, "tau_refined": float(tau_refined), "residual_m": residual,
            "residual_at_zero_m": float(med[0]), "n_scored": int(keep.sum())}


JOINT_STEP_MAX_RAD = 0.2  # continuity precondition for interpolating joint angles


def interp_joints_own(ur_t: np.ndarray, ur_q: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Linear interpolation of the joint record at arbitrary times.

    Written out by hand (bracket via searchsorted, explicit lerp, explicit
    end-clamping) instead of calling ``np.interp``, so that an off-by-one or an
    inverted weight in the converter cannot be reproduced here by using the very
    same library routine.
    """
    ur_t = np.asarray(ur_t, dtype=np.float64)
    ur_q = np.asarray(ur_q, dtype=np.float64)
    t = np.asarray(query_t, dtype=np.float64)
    n = len(ur_t)
    if n == 1:
        return np.repeat(ur_q[:1], len(t), axis=0)
    hi = np.clip(np.searchsorted(ur_t, t, side="left"), 1, n - 1)
    lo = hi - 1
    span = ur_t[hi] - ur_t[lo]
    w = np.where(span > 0.0, (t - ur_t[lo]) / np.where(span > 0.0, span, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)  # clamp outside the record to its first/last sample
    return ur_q[lo] + (ur_q[hi] - ur_q[lo]) * w[:, None]


def build_state_action(st, tau: float, max_frames: int = 0, drop_stale_tail: bool = False):
    """Rebuild (cam1_t, state[N,16], action[N,7], q[N,6], n_tail_clamped) at this tau."""
    cam1_t, ur_t, ur_q = st["cam1_t"], st["ur_t"], st["ur_q"]
    n_tail = int(np.sum(cam1_t + tau > ur_t[-1]))
    if drop_stale_tail and n_tail:
        cam1_t = cam1_t[: max(0, len(cam1_t) - n_tail)]
    if max_frames and max_frames > 0:
        cam1_t = cam1_t[:max_frames]
    n = len(cam1_t)

    q = interp_joints_own(ur_t, ur_q, cam1_t + tau)
    grip_j = nearest_idx(st["grip_t"], cam1_t)
    grip_pos = np.asarray(st["grip_pos"][grip_j], dtype=np.float64)
    grip_cmd = ffill_bfill(st["grip_cmd"])[grip_j]

    poses = np.stack([tcp_pose(q[k]) for k in range(n)]) if n else np.zeros((0, 4, 4))
    state = np.zeros((n, STATE_DIM), dtype=np.float64)
    action = np.zeros((n, ACTION_DIM), dtype=np.float64)
    for k in range(n):
        R = poses[k][:3, :3]
        state[k, 0:6] = q[k]
        state[k, 6:9] = poses[k][:3, 3]
        state[k, 9:12] = R[:, 0]
        state[k, 12:15] = R[:, 1]
        state[k, 15] = grip_pos[k]
        if k + 1 < n:
            action[k, 0:3] = poses[k + 1][:3, 3] - poses[k][:3, 3]
            action[k, 3:6] = so3_log_own(poses[k + 1][:3, :3] @ R.T)
        action[k, 6] = grip_cmd[k]

    return cam1_t, state, action, q, int(np.sum(cam1_t + tau > ur_t[-1]))


def take_cam1_t(h5_path: str, max_frames: int = 0) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as f:
        t = f["cam1_frames"]["t_rel_s"][:]
    return t[:max_frames] if (max_frames and max_frames > 0) else t


# --------------------------------------------------------------------------
# parquet reading (pyarrow preferred, pandas fallback)
# --------------------------------------------------------------------------
def _read_parquet(path: str):
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
        raise RuntimeError("neither pyarrow nor pandas available") from exc
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

        vals = col.to_pylist() if isinstance(col, (pa.ChunkedArray, pa.Array)) else list(col)
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
    return sorted(glob.glob(os.path.join(root, subdir, "chunk-*", f"file-*{ext}")))


def load_data_frames(root: str, info: dict):
    """Concatenate all data/**/*.parquet in (chunk, file) order."""
    files = _sorted_chunk_files(root, "data", ".parquet")
    if not files:
        raise RuntimeError(f"no data parquet files under {os.path.join(root, 'data')}")
    cols, backend = {}, None
    for fp in files:
        d, backend = _read_parquet(fp)
        for k, v in d.items():
            cols.setdefault(k, []).append(v)
    widths = {"action": ACTION_DIM, "observation.state": STATE_DIM}
    merged = {}
    for k, parts in cols.items():
        if k in widths:
            merged[k] = np.concatenate([_col_to_2d(p, widths[k]) for p in parts], axis=0)
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
        n, pylists = None, {}
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
    d, _ = _read_parquet(os.path.join(root, "meta", "tasks.parquet"))
    names = list(d.keys())
    task_col = None
    for cand in ("task", "__index_level_0__"):
        if cand in names:
            task_col = cand
            break
    if task_col is None:
        task_col = next((k for k in names if k != "task_index"), None)
    if task_col is None or "task_index" not in names:
        raise RuntimeError(f"unexpected tasks.parquet schema: {names}")
    return {int(i): str(t) for i, t in zip(_col_to_pylist(d["task_index"]),
                                           _col_to_pylist(d[task_col]))}


# --------------------------------------------------------------------------
# video helpers
# --------------------------------------------------------------------------
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def decode_lerobot_frames(video_path: str, wanted, height: int, width: int):
    """Decode requested *decode-order* frame indices from a LeRobot (AV1) video."""
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH")
    wanted = sorted(set(int(i) for i in wanted))
    if not wanted:
        return {}, 0
    stop = wanted[-1]
    need = height * width * 3
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-i", video_path,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-frames:v", str(stop + 1), "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=need)
    want, out, i = set(wanted), {}, 0
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
        raise RuntimeError(f"decoded only {i} frames from {video_path}; missing {missing[:5]}"
                           + (f"; ffmpeg: {err[:200]}" if err else ""))
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
    out, i, stop = {}, 0, wanted[-1]
    while i <= stop:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out, i


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
    if n_sample is None or n_sample <= 0 or n_sample >= n_ep:
        return list(range(n_ep))
    sel = sorted(set([0, n_ep - 1]
                     + [int(round(k * (n_ep - 1) / max(1, n_sample - 1)))
                        for k in range(n_sample)]))
    return sel[:n_sample] if len(sel) > n_sample else sel


def sample_frame_indices(n: int, m: int):
    m = max(1, min(int(m), n))
    return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, m)))


# --------------------------------------------------------------------------
# CHECK 0 -- self-test of this file's own re-implementations
# --------------------------------------------------------------------------
def self_test(rep: Report, takes):
    section("CHECK 0  --  self-test of the validator's own re-implementations")
    import h5py

    with h5py.File(os.path.join(takes[0], "vectors.h5"), "r") as f:
        cam1_t = f["cam1_frames"]["t_rel_s"][:]
        ur_t = f["ur_joint_states"]["t_rel_s"][:]
        cam2_t = f["cam2_frames"]["t_rel_s"][:]
        grip_t = f["gripper"]["t_rel_s"][:]
    q = cam1_t[:: max(1, len(cam1_t) // 60)]
    ok = all(np.array_equal(nearest_idx(src, q), nearest_idx_bruteforce(src, q))
             for src in (ur_t, cam2_t, grip_t))
    rep.check(ok, "vectorised nearest_idx == brute-force argmin reference",
              f"{len(q)} queries against ur/cam2/gripper timelines")

    v = np.array([np.nan, np.nan, 1.0, np.nan, np.nan, 2.0, np.nan])
    want = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    rep.check(np.array_equal(ffill_bfill(v), want), "ffill_bfill reference case",
              f"{ffill_bfill(v).tolist()}")

    # log(exp(w)) == w over the magnitudes this dataset actually contains, plus
    # the degenerate and near-pi corners.
    # Two bands: the magnitudes this dataset actually contains (|drot| <= 0.03 rad)
    # must round trip to machine precision.  theta -> pi is intrinsically
    # ill-conditioned for ANY log map (the axis is recovered from a matrix whose
    # antisymmetric part has vanished), so it gets its own, looser bar; nothing in
    # this dataset comes anywhere near it.
    rng = np.random.default_rng(20260914)
    for mags, tol, label in (
        ([0.0, 1e-9, 1e-4, 0.03, 0.5, 2.0], 1e-9, "|w| <= 2 rad"),
        ([math.pi - 1e-7, math.pi - 1e-4], 1e-6, "|w| -> pi (ill-conditioned corner)"),
    ):
        worst, worst_w = 0.0, np.zeros(3)
        for mag in mags:
            for _ in range(50):
                ax = rng.normal(size=3)
                ax /= np.linalg.norm(ax)
                w = ax * mag
                e = float(np.max(np.abs(so3_log_own(so3_exp_own(w)) - w)))
                if e > worst:
                    worst, worst_w = e, w
        rep.check(worst < tol, f"so3_log_own(so3_exp_own(w)) == w for {label}",
                  f"max|d|={worst:.3e} (tol {tol:g}) over |w| in {mags}, "
                  f"worst w={np.round(worst_w, 6).tolist()}")

    # the hand-written interpolator: exact on a linear ramp, exact ON knots,
    # correct halfway, and clamped (not extrapolated) outside the record
    src_t = np.array([0.0, 0.1, 0.35, 0.4, 1.0])
    src_q = np.stack([src_t * (k + 1) + k for k in range(6)], axis=1)
    qry = np.array([-5.0, 0.0, 0.05, 0.1, 0.375, 1.0, 7.0])
    got = interp_joints_own(src_t, src_q, qry)
    want = np.stack([np.clip(qry, 0.0, 1.0) * (k + 1) + k for k in range(6)], axis=1)
    rep.check(float(np.max(np.abs(got - want))) < 1e-12,
              "interp_joints_own is exact on a ramp, on knots, and clamps at both ends",
              f"max|d|={float(np.max(np.abs(got - want))):.3e}")
    rng2 = np.random.default_rng(7)
    tt = np.sort(rng2.uniform(0, 3, 40))
    qq = rng2.normal(size=(40, 6))
    probe = np.sort(rng2.uniform(-1, 4, 25))
    ref = np.stack([np.interp(probe, tt, qq[:, k]) for k in range(6)], axis=1)
    rep.check(float(np.max(np.abs(interp_joints_own(tt, qq, probe) - ref))) < 1e-12,
              "interp_joints_own agrees with an independent reference interpolation",
              f"max|d|={float(np.max(np.abs(interp_joints_own(tt, qq, probe) - ref))):.3e}")

    # a known rotation: 90 deg about +z
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    got = so3_log_own(Rz)
    rep.check(np.allclose(got, [0.0, 0.0, math.pi / 2], atol=1e-12),
              "so3_log_own of a +90 deg z-rotation == [0,0,pi/2]",
              f"got {np.round(got, 9).tolist()}")

    # this file's TCP construction agrees with the 'fk @ translate' matrix form
    qt = np.array([-3.19, -1.49, 1.69, -1.80, -1.59, -3.27])
    T_tool = np.eye(4)
    T_tool[2, 3] = TOOL_Z_M
    ref = ur_kin.fk(qt) @ T_tool
    dev = float(np.max(np.abs(tcp_pose(qt) - ref)))
    rep.check(dev < 1e-12, "tcp_pose == fk(q) @ translate(0,0,0.174)", f"max|d|={dev:.3e}")


# --------------------------------------------------------------------------
# CHECK 1 -- meta/info.json
# --------------------------------------------------------------------------
def check_info(rep: Report, info: dict, n_takes: int, expected_total_frames: int, task: str):
    section("CHECK 1  --  meta/info.json")
    rep.check(info.get("codebase_version") == "v3.0", "codebase_version == 'v3.0'",
              f"got {info.get('codebase_version')!r}")
    rep.check(int(info.get("fps", -1)) == FPS, f"fps == {FPS}", f"got {info.get('fps')!r}")

    feats = info.get("features", {})
    missing = [k for k in EXPECTED_FEATURES if k not in feats]
    extra = [k for k in feats if k not in EXPECTED_FEATURES]
    rep.check(not missing, "all expected features present",
              f"missing: {missing}" if missing else f"{len(feats)} features")
    rep.check(not extra, "no unexpected extra features", f"extra: {extra}" if extra else "")

    bad_depth = [k for k in feats if any(s in k.lower() for s in FORBIDDEN_FEATURE_SUBSTR)]
    rep.check(not bad_depth, "dataset carries NO depth feature",
              f"found {bad_depth}" if bad_depth else "0 depth keys")

    for name, (dtype, shape, names) in EXPECTED_FEATURES.items():
        if name not in feats:
            continue
        f = feats[name]
        got_dtype, got_shape, got_names = f.get("dtype"), list(f.get("shape", [])), f.get("names")
        ok = got_dtype == dtype and got_shape == list(shape)
        detail = f"dtype={got_dtype} shape={got_shape}"
        if names is not None:
            ok = ok and list(got_names or []) == list(names)
            detail += f" names={got_names}"
        rep.check(ok, f"feature {name}", detail)

    rep.check(int(info.get("total_episodes", -1)) == n_takes,
              f"total_episodes == {n_takes} (included takes)", f"got {info.get('total_episodes')}")
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
    mismatches = []
    for i, (ep, tk, n_src) in enumerate(zip(eps, takes, take_lens)):
        n_ep = int(ep["length"])
        span = int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
        if n_ep != n_src or span != n_src:
            mismatches.append(f"ep{i} ({os.path.basename(tk)}): length={n_ep} span={span} src={n_src}")
    if not mismatches:
        rep.pass_(f"all {len(eps)} episode lengths match their source take",
                  f"total {sum(take_lens)} frames")
    else:
        rep.fail("episode length mismatch", "; ".join(mismatches[:10]))
    return not mismatches


# --------------------------------------------------------------------------
# CHECKS 3-9 -- state / action re-derivation
# --------------------------------------------------------------------------
def check_numeric(rep: Report, eps, takes, data, tol: float, verbose: bool, max_frames: int,
                  meta_taus=None, tau_tol: float = 0.02, drop_stale_tail: bool = False):
    section("CHECK 3-8  --  state / action independent re-derivation")

    stored_state = data["observation.state"]
    stored_action = data["action"]
    meta_taus = meta_taus or {}

    # worst deviation per contract slice
    dev = {k: 0.0 for k in ("q", "tcp_p", "rot6d", "grip_pos", "dp", "drot", "grip_cmd")}
    worst_ep = {k: "" for k in dev}
    ortho_worst = 0.0
    last_zero_worst = 0.0
    n_bad = 0
    bad_names = []
    taus_own, taus_refined, residuals, tau_mismatch, tail_total = [], [], [], [], 0
    tau_disagree_eps = []
    joint_step_max, joint_step_bad = 0.0, []

    for i, (ep, tk) in enumerate(zip(eps, takes)):
        st = _read_streams(os.path.join(tk, "vectors.h5"))
        step = float(np.max(np.abs(np.diff(st["ur_q"], axis=0))))
        joint_step_max = max(joint_step_max, step)
        if step >= JOINT_STEP_MAX_RAD:
            joint_step_bad.append(f"{os.path.basename(tk)}: {step:.4f} rad")
        tinfo = estimate_tau_own(st)
        tau_own = tinfo["tau"]
        taus_own.append(tau_own)
        taus_refined.append(tinfo["tau_refined"])
        residuals.append(tinfo["residual_m"])

        tau_meta = meta_taus.get(i)
        if tau_meta is None:
            tau_mismatch.append(float("nan"))
        else:
            d_tau = abs(tau_own - float(tau_meta))
            tau_mismatch.append(d_tau)
            if d_tau > tau_tol + 1e-12:
                tau_disagree_eps.append(
                    f"ep{i} ({os.path.basename(tk)}): own {tau_own:.2f} vs meta {tau_meta:.2f}")

        # Compare on the tau the converter DECLARED when the two differ (a
        # different tau legitimately produces different numbers), and report both.
        tau_cmp = tau_own if (tau_meta is None or tau_own == float(tau_meta)) else float(tau_meta)
        _, state, action, _, n_tail = build_state_action(st, tau_cmp, max_frames, drop_stale_tail)
        tail_total += n_tail
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        got_s = stored_state[a:b].astype(np.float64)
        got_a = stored_action[a:b].astype(np.float64)

        if got_s.shape != state.shape or got_a.shape != action.shape:
            rep.fail(f"ep{i} shape mismatch",
                     f"stored {got_s.shape}/{got_a.shape} vs derived {state.shape}/{action.shape}")
            n_bad += 1
            bad_names.append(os.path.basename(tk))
            continue

        d = {
            "q": float(np.max(np.abs(got_s[:, 0:6] - state[:, 0:6]))),
            "tcp_p": float(np.max(np.abs(got_s[:, 6:9] - state[:, 6:9]))),
            "rot6d": float(np.max(np.abs(got_s[:, 9:15] - state[:, 9:15]))),
            "grip_pos": float(np.max(np.abs(got_s[:, 15] - state[:, 15]))),
            "dp": float(np.max(np.abs(got_a[:, 0:3] - action[:, 0:3]))),
            "drot": float(np.max(np.abs(got_a[:, 3:6] - action[:, 3:6]))),
            "grip_cmd": float(np.max(np.abs(got_a[:, 6] - action[:, 6]))),
        }
        for k, v in d.items():
            if v > dev[k]:
                dev[k], worst_ep[k] = v, f"ep{i} ({os.path.basename(tk)})"

        # orthonormality of the STORED 6-D rotation (not the derived one)
        c1, c2 = got_s[:, 9:12], got_s[:, 12:15]
        e = max(
            float(np.max(np.abs(np.linalg.norm(c1, axis=1) - 1.0))),
            float(np.max(np.abs(np.linalg.norm(c2, axis=1) - 1.0))),
            float(np.max(np.abs(np.sum(c1 * c2, axis=1)))),
        )
        ortho_worst = max(ortho_worst, e)

        lz = float(np.max(np.abs(got_a[-1, 0:6])))
        last_zero_worst = max(last_zero_worst, lz)

        if max(d.values()) >= tol or e >= ORTHO_TOL or lz != 0.0:
            n_bad += 1
            bad_names.append(os.path.basename(tk))
            rep.fail(f"ep{i} ({os.path.basename(tk)}) deviation",
                     "  ".join(f"{k}={v:.3e}" for k, v in d.items())
                     + f"  ortho={e:.3e} last_frame_motion={lz:.3e}")
        elif verbose:
            print(f"    ep{i:3d} {os.path.basename(tk):32s} "
                  + " ".join(f"{k}={v:.2e}" for k, v in d.items()), flush=True)

    # --- 3b: does the converter's declared tau_q survive an independent estimate?
    ta = np.array(taus_own, dtype=float)
    tr = np.array(residuals, dtype=float)
    dt = np.array(tau_mismatch, dtype=float)
    if not meta_taus:
        rep.warn("CHECK 3b tau_q vs meta/source_takes.json",
                 "source_takes.json carries no joint_state_shift_s -- cannot cross-check")
    else:
        rep.check(not tau_disagree_eps,
                  f"CHECK 3b |tau_validator - tau_meta| <= {tau_tol:g} s for every episode",
                  "; ".join(tau_disagree_eps[:6]) if tau_disagree_eps
                  else (f"max|d tau|={np.nanmax(dt):.4f} s over {len(dt)} episodes"))
    rep.check(not joint_step_bad,
              f"joint series continuous enough to interpolate (max step < "
              f"{JOINT_STEP_MAX_RAD} rad)",
              "; ".join(joint_step_bad[:6]) if joint_step_bad
              else f"worst consecutive |dq| over all episodes = {joint_step_max:.4f} rad")
    rep.pass_("CHECK 3c independent tau_q estimate",
              f"grid min/med/max = {ta.min():.2f}/{np.median(ta):.2f}/{ta.max():.2f} s; "
              f"parabola-refined med {np.median(taus_refined):.4f} s; residual "
              f"min/med/max = {tr.min() * 1000:.2f}/{np.median(tr) * 1000:.2f}/"
              f"{tr.max() * 1000:.2f} mm")
    if tail_total:
        rep.warn("frames clamped past the end of the joint stream after the tau_q shift",
                 f"{tail_total} frame(s) across {len(eps)} episodes carry a frozen state and "
                 f"zero deltas (the recorder's joint cache ends ~tau_q before the cameras do)")

    if n_bad == 0:
        rep.pass_(f"CHECK 3  state[:6] == interpolated q at master + tau_q (tol={tol:g})",
                  f"max|d|={dev['q']:.3e}  worst {worst_ep['q']}")
        rep.pass_(f"CHECK 4  state[6:9] == fk(q) + {TOOL_Z_M} m tool offset (tol={tol:g})",
                  f"max|d|={dev['tcp_p']:.3e}  worst {worst_ep['tcp_p']}")
        rep.pass_(f"CHECK 5a state[9:15] == first two columns of R_tcp (tol={tol:g})",
                  f"max|d|={dev['rot6d']:.3e}  worst {worst_ep['rot6d']}")
        rep.pass_(f"CHECK 5b state[9:15] columns orthonormal (tol={ORTHO_TOL:g})",
                  f"max|d|={ortho_worst:.3e}")
        rep.pass_(f"CHECK 6  state[15] == resampled grip_pos (tol={tol:g})",
                  f"max|d|={dev['grip_pos']:.3e}  worst {worst_ep['grip_pos']}")
        rep.pass_(f"CHECK 7a action[:3] == p(q_t+1) - p(q_t) (tol={tol:g})",
                  f"max|d|={dev['dp']:.3e}  worst {worst_ep['dp']}")
        rep.pass_(f"CHECK 7b action[3:6] == so3_log(R_t+1 R_t^T), own log map (tol={tol:g})",
                  f"max|d|={dev['drot']:.3e}  worst {worst_ep['drot']}")
        rep.pass_("CHECK 7c last frame of every episode has exactly zero motion",
                  f"max|action[-1,:6]|={last_zero_worst:.3e}")
        rep.pass_(f"CHECK 8  action[6] == resampled ffill/bfill grip_cmd (tol={tol:g})",
                  f"max|d|={dev['grip_cmd']:.3e}  worst {worst_ep['grip_cmd']}")
    else:
        rep.fail(f"{n_bad} episode(s) deviate", f"{bad_names[:6]}")

    return {"per_slice_max_abs_dev": dev, "worst_episode": worst_ep,
            "ortho_max_dev": ortho_worst, "last_frame_max_motion": last_zero_worst,
            "n_bad_episodes": n_bad,
            "tau_own": [float(x) for x in taus_own],
            "tau_own_refined": [float(x) for x in taus_refined],
            "tau_residual_m": [float(x) for x in residuals],
            "tau_abs_diff_vs_meta": [None if np.isnan(x) else float(x) for x in dt],
            "tau_stats": {"min": float(ta.min()), "median": float(np.median(ta)),
                          "max": float(ta.max()),
                          "residual_min_m": float(tr.min()),
                          "residual_median_m": float(np.median(tr)),
                          "residual_max_m": float(tr.max())},
            "tau_disagreements": tau_disagree_eps,
            "joint_step_max_rad": float(joint_step_max),
            "n_frames_clamped_past_joint_stream": int(tail_total)}


def check_nan(rep: Report, data):
    section("CHECK 9  --  no NaN / Inf in stored observation.state and action")
    for key in ("observation.state", "action"):
        arr = data[key]
        n_nan, n_inf = int(np.isnan(arr).sum()), int(np.isinf(arr).sum())
        rep.check(n_nan == 0 and n_inf == 0, f"{key} finite",
                  f"NaN={n_nan} Inf={n_inf}  range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")


def check_task(rep: Report, root, eps, data, task: str):
    section("CHECK 10  --  task string")
    tasks = load_tasks(root)
    rep.check(len(tasks) == 1, "exactly one task in meta/tasks.parquet", f"got {tasks}")
    rep.check(len(tasks) == 1 and list(tasks.values())[0] == task,
              f"task string == {task!r}", f"got {list(tasks.values())!r}")
    ti = data["task_index"].astype(np.int64)
    uniq = np.unique(ti)
    rep.check(len(uniq) == 1, "single task_index across all frames", f"unique={uniq.tolist()}")
    rep.check(all(int(u) in tasks and tasks[int(u)] == task for u in uniq),
              "every frame's task_index resolves to the task string",
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
    section("CHECK 11  --  RGB video integrity (LeRobot AV1 vs raw MPEG-4)")
    if FFMPEG is None:
        rep.skip("video check", "ffmpeg not found on PATH")
        return None

    height = int(info["features"][RGB_KEYS[0]]["shape"][0])
    width = int(info["features"][RGB_KEYS[0]]["shape"][1])
    fps = int(info["fps"])
    video_tmpl = info.get("video_path",
                          "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    all_corr, worst, n_fail, plan = [], (None, 1.0), 0, {}

    for ei in sample_episode_indices(len(eps), n_sample):
        ep, tk = eps[ei], takes[ei]
        n = int(ep["length"])
        # Only the two camera timelines matter here; the tau_q shift never touches
        # the image path, so this deliberately skips the state/action rebuild.
        st = _read_streams(os.path.join(tk, "vectors.h5"))
        cam1_t, cam2_t = st["cam1_t"], st["cam2_t"]
        if max_frames and max_frames > 0:
            cam1_t = cam1_t[:max_frames]
        cam1_t = cam1_t[:n]
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
            fk_ = int(ep[f"videos/{key}/file_index"])
            t0 = float(ep[f"videos/{key}/from_timestamp"])
            vpath = os.path.join(root, video_tmpl.format(video_key=key, chunk_index=ck,
                                                         file_index=fk_))
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
    section("CHECK 12  --  structural sanity")
    fps = int(info["fps"])
    prev_to, ok, detail = 0, True, ""
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
    rep.check(prev_to == len(data["index"]), "episode spans cover all data rows",
              f"last dataset_to_index={prev_to} nrows={len(data['index'])}")

    gidx = data["index"].astype(np.int64)
    rep.check(np.array_equal(gidx, np.arange(len(gidx))), "global 'index' == 0..N-1",
              f"N={len(gidx)}")
    epi = data["episode_index"].astype(np.int64)
    rep.check(np.all(np.diff(epi) >= 0), "episode_index non-decreasing across data", "")

    fi = data["frame_index"].astype(np.int64)
    ok_fi, detail = True, ""
    for i, ep in enumerate(eps):
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        if not np.array_equal(fi[a:b], np.arange(b - a)):
            ok_fi, detail = False, f"ep{i} frame_index not 0..{b - a - 1}"
            break
        if not np.all(epi[a:b] == i):
            ok_fi, detail = False, f"ep{i} episode_index column wrong"
            break
    rep.check(ok_fi, "per-episode frame_index restarts at 0 and increments by 1", detail)

    ts = data["timestamp"].astype(np.float64)
    dev = float(np.max(np.abs(ts - fi.astype(np.float64) / fps))) if len(ts) else 0.0
    rep.check(dev < 1e-5, f"timestamp == frame_index / {fps}", f"max dev {dev:.3e}")


# --------------------------------------------------------------------------
# CHECK 13 -- meta/eef_space.json
# --------------------------------------------------------------------------
def check_eef_space_json(rep: Report, lr_root, info):
    section("CHECK 13  --  meta/eef_space.json")
    path = os.path.join(lr_root, "meta", "eef_space.json")
    if not os.path.exists(path):
        rep.fail("meta/eef_space.json exists", path)
        return None
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except Exception as exc:
        rep.fail("meta/eef_space.json parses as JSON", str(exc)[:200])
        return None

    rep.check(obj.get("action_space_id") == ACTION_SPACE_ID,
              f"action_space_id == {ACTION_SPACE_ID!r}", f"got {obj.get('action_space_id')!r}")
    tool = obj.get("tool_z_m")
    rep.check(isinstance(tool, (int, float)) and abs(float(tool) - TOOL_Z_M) < 1e-12,
              f"tool_z_m == {TOOL_Z_M}", f"got {tool!r}")
    rep.check(list(obj.get("state_names") or []) == STATE_NAMES,
              "state_names == the 16 DESIGN names", f"got {obj.get('state_names')}")
    rep.check(list(obj.get("action_names") or []) == ACTION_NAMES,
              "action_names == the 7 DESIGN names", f"got {obj.get('action_names')}")
    definition = obj.get("definition")
    rep.check(isinstance(definition, str) and len(definition) >= 80,
              "definition is a prose paragraph",
              f"{len(definition) if isinstance(definition, str) else 'missing'} chars")
    align = obj.get("joint_state_alignment")
    rep.check(isinstance(align, str) and "tau_q" in align and "stale" in align.lower(),
              "joint_state_alignment documents the stale joint cache",
              f"{align!r}")
    rep.check(obj.get("joint_state_resampling") == "linear_interp",
              "joint_state_resampling == 'linear_interp'",
              f"got {obj.get('joint_state_resampling')!r}")
    sha = obj.get("eef_space_sha256")
    rep.check(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
              "eef_space_sha256 is a sha256 hex digest", f"got {sha!r}")

    feats = info.get("features", {})
    rep.check(list(feats.get("observation.state", {}).get("names") or []) == list(obj.get("state_names") or []),
              "eef_space.json state_names == info.json observation.state names", "")
    rep.check(list(feats.get("action", {}).get("names") or []) == list(obj.get("action_names") or []),
              "eef_space.json action_names == info.json action names", "")
    return obj


# --------------------------------------------------------------------------
# CHECK 14 -- meta/source_takes.json
# --------------------------------------------------------------------------
def read_meta_taus(lr_root: str) -> dict:
    """{episode_index: joint_state_shift_s} from meta/source_takes.json, or {}.

    Read early (before check 3) so the numeric re-derivation can be compared on
    the tau the converter DECLARED as well as on the validator's own estimate.
    Reading a declared conversion parameter is not a loss of independence: the
    tau itself is still re-estimated here from the raw streams.
    """
    path = os.path.join(lr_root, "meta", "source_takes.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except Exception:
        return {}
    entries = obj.get("episodes") if isinstance(obj, dict) else obj
    if not isinstance(entries, list):
        return {}
    out = {}
    for i, e in enumerate(entries):
        if isinstance(e, dict) and "joint_state_shift_s" in e:
            out[int(e.get("episode_index", i))] = float(e["joint_state_shift_s"])
    return out


def check_source_takes_json(rep: Report, lr_root, takes, eps, raw_lens=None,
                            expect_dropped_tail=False):
    section("CHECK 14  --  meta/source_takes.json")
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

    entries = obj.get("episodes") if isinstance(obj, dict) else obj
    if not isinstance(entries, list):
        rep.fail("source_takes.json has a recognisable episode list",
                 f"top-level type {type(obj).__name__}")
        return None
    names, counts, taus, tau_res, dropped = [], [], {}, {}, []
    for i, e in enumerate(entries):
        if isinstance(e, dict):
            nm = None
            for k in ("take_dir_name", "take", "name", "dir", "path"):
                if k in e:
                    nm = os.path.basename(str(e[k]).rstrip("/"))
                    break
            names.append(nm)
            counts.append(int(e["n_frames"]) if "n_frames" in e else None)
            dropped.append(int(e.get("n_frames_dropped_stale_tail", 0)))
            ei = int(e.get("episode_index", i))
            if "joint_state_shift_s" in e:
                taus[ei] = float(e["joint_state_shift_s"])
            if "joint_state_shift_residual_m" in e:
                tau_res[ei] = float(e["joint_state_shift_residual_m"])
        else:
            names.append(os.path.basename(str(e).rstrip("/")))
            counts.append(None)
            dropped.append(0)

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

    # The dropped tail must be exactly what is missing: stored length + declared
    # drop == the take's raw cam1 frame count, re-counted here from the HDF5.
    if raw_lens is not None and all(c is not None for c in counts):
        n = min(len(counts), len(raw_lens), len(dropped))
        bad = [f"{os.path.basename(takes[i])}: {counts[i]} + {dropped[i]} != {raw_lens[i]}"
               for i in range(n) if counts[i] + dropped[i] != raw_lens[i]]
        rep.check(not bad and n == len(raw_lens),
                  "n_frames + n_frames_dropped_stale_tail == raw cam1 frame count",
                  "; ".join(bad[:6]) if bad
                  else f"{sum(dropped[:n])} frame(s) dropped over {n} takes "
                       f"(raw total {sum(raw_lens[:n])} -> stored {sum(counts[:n])})")
        if expect_dropped_tail:
            rep.check(all(d > 0 for d in dropped[:n]),
                      "--drop-stale-tail: every episode actually dropped frames",
                      f"min {min(dropped[:n])} / max {max(dropped[:n])} per episode")
        elif any(d for d in dropped[:n]):
            rep.fail("episodes declare dropped tail frames but --drop-stale-tail was not passed",
                     f"total {sum(dropped[:n])}")

    if taus:
        tv = np.array(sorted(taus.items()))[:, 1]
        rv = np.array([tau_res[k] for k in sorted(tau_res)]) if tau_res else np.array([np.nan])
        rep.check(len(taus) == len(names), "every episode carries joint_state_shift_s",
                  f"{len(taus)}/{len(names)}; min/med/max = "
                  f"{tv.min():.2f}/{np.median(tv):.2f}/{tv.max():.2f} s")
        rep.check(bool(np.all((tv >= 0.6) & (tv <= 1.2))),
                  "declared joint_state_shift_s within the accepted [0.6, 1.2] s band",
                  f"min {tv.min():.2f} max {tv.max():.2f}")
        rep.check(bool(np.all(rv < 3e-3)),
                  "declared joint_state_shift_residual_m < 3 mm for every episode",
                  f"max {np.nanmax(rv) * 1000:.2f} mm")
    else:
        rep.warn("source_takes.json carries no joint_state_shift_s",
                 "the joint-cache alignment is undocumented in this dataset")
    return {"names": names, "n_frames": counts, "joint_state_shift_s": taus,
            "joint_state_shift_residual_m": tau_res,
            "n_frames_dropped_stale_tail": dropped}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Validate the carrot_in_pot EEF-delta LeRobot v3.0 dataset "
                    "against its raw HDF5/mp4 takes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw", default="/home/laptop3/youngwoong_ws/Put_carrot_in_pot")
    ap.add_argument("--lerobot", default="/home/laptop3/youngwoong_ws/carrot_in_pot_eef_lerobot")
    ap.add_argument("--repo-id", default="Bigenlight/carrot_in_pot_eef_lerobot_v3")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--task", default="Put carrot in pot")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max allowed abs deviation for state/action")
    ap.add_argument("--tau-tol", type=float, default=0.02,
                    help="max allowed |tau_validator - tau_meta| per episode, seconds")
    ap.add_argument("--drop-stale-tail", action="store_true",
                    help="mirror the converter's --drop-stale-tail (episodes are shorter "
                         "than their cam1 frame count by the clamped tail)")
    ap.add_argument("--video-episodes", type=int, default=3)
    ap.add_argument("--video-frames", type=int, default=4)
    ap.add_argument("--video-warn-corr", type=float, default=0.99)
    ap.add_argument("--video-fail-corr", type=float, default=0.98)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="TEST ONLY: mirror a truncated conversion")
    ap.add_argument("--max-takes", type=int, default=0, help="TEST ONLY: first N takes only")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    raw_root = os.path.abspath(os.path.expanduser(args.raw))
    lr_root = os.path.abspath(os.path.expanduser(args.lerobot))

    print("LeRobot v3.0 EEF-delta conversion validator  --  carrot_in_pot (RGB only, no depth)")
    print(f"  interpreter    : {sys.executable}")
    print(f"  ur_kin         : {os.path.abspath(ur_kin.__file__)}")
    print(f"  raw takes      : {raw_root}")
    print(f"  lerobot dataset: {lr_root}")
    print(f"  repo id        : {args.repo_id}")
    print(f"  excluded takes : {args.exclude or '(none)'}")
    print(f"  expected task  : {args.task!r}")
    print(f"  action space   : {ACTION_SPACE_ID}  (state {STATE_DIM} / action {ACTION_DIM})")
    print(f"  numeric tol    : {args.tol:g}")
    if args.max_frames or args.max_takes:
        print(f"  !! TEST MODE   : --max-frames={args.max_frames} --max-takes={args.max_takes}")

    rep = Report()

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

    missing_files = []
    for t in takes:
        for fn in ("vectors.h5", "cam1.mp4", "cam2.mp4"):
            if not os.path.exists(os.path.join(t, fn)):
                missing_files.append(os.path.join(os.path.basename(t), fn))
    if missing_files:
        rep.fail("raw take files missing", f"{missing_files[:8]}")

    if not os.path.isdir(lr_root):
        print(f"\nFATAL: LeRobot dataset not found at {lr_root}", file=sys.stderr)
        return 2

    self_test(rep, takes)

    info_path = os.path.join(lr_root, "meta", "info.json")
    if not os.path.exists(info_path):
        print(f"\nFATAL: {info_path} not found", file=sys.stderr)
        return 2
    with open(info_path) as fh:
        info = json.load(fh)

    print("\nreading source cam1_frames row counts ...", flush=True)
    meta_taus = read_meta_taus(lr_root)
    raw_lens = [len(take_cam1_t(os.path.join(t, "vectors.h5"), args.max_frames)) for t in takes]
    if args.drop_stale_tail:
        # How many trailing frames fall past the joint stream depends on tau, and
        # this file's tau can legitimately sit one 0.01 s grid step away from the
        # converter's (check 3b bounds that difference).  One grid step can move
        # the boundary across a single 33 ms master frame, so the LENGTH
        # expectation is taken from the tau the converter DECLARED -- check 3b is
        # what guards that declaration, and check 3 then re-derives every value
        # at that same tau.  Falls back to this file's own tau if undeclared.
        take_lens = []
        for i, t in enumerate(takes):
            st = _read_streams(os.path.join(t, "vectors.h5"))
            tau = float(meta_taus.get(i, estimate_tau_own(st)["tau"]))
            c1 = st["cam1_t"]
            c1 = c1[: max(0, len(c1) - int(np.sum(c1 + tau > st["ur_t"][-1])))]
            take_lens.append(len(c1[:args.max_frames] if args.max_frames else c1))
    else:
        take_lens = raw_lens
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
                 f"{len(eps)} episodes vs {len(takes)} takes -- check --exclude")
        n = min(len(eps), len(takes))
        eps_c, takes_c, lens_c = eps[:n], takes[:n], take_lens[:n]
    else:
        rep.pass_("episode count == included take count", f"{len(eps)}")
        eps_c, takes_c, lens_c = eps, takes, take_lens

    check_frame_counts(rep, eps_c, takes_c, lens_c)
    numeric = check_numeric(rep, eps_c, takes_c, data, args.tol, args.verbose, args.max_frames,
                            meta_taus=meta_taus, tau_tol=args.tau_tol,
                            drop_stale_tail=args.drop_stale_tail)
    check_nan(rep, data)
    check_task(rep, lr_root, eps, data, args.task)

    corr = None
    if args.skip_video:
        section("CHECK 11  --  RGB video integrity")
        rep.skip("RGB video check", "--skip-video")
    else:
        try:
            corr = check_videos(rep, lr_root, info, eps_c, takes_c, args.video_episodes,
                                args.video_warn_corr, args.video_fail_corr,
                                args.video_frames, args.max_frames)
        except Exception:
            rep.fail("RGB video check raised an exception",
                     traceback.format_exc().splitlines()[-1])

    check_sanity(rep, info, eps, data)

    try:
        eef_json = check_eef_space_json(rep, lr_root, info)
    except Exception:
        eef_json = None
        rep.fail("eef_space.json check raised an exception",
                 traceback.format_exc().splitlines()[-1])
    try:
        src_json = check_source_takes_json(rep, lr_root, takes, eps_c, raw_lens=raw_lens,
                                           expect_dropped_tail=args.drop_stale_tail)
    except Exception:
        src_json = None
        rep.fail("source_takes.json check raised an exception",
                 traceback.format_exc().splitlines()[-1])

    # ---- summary --------------------------------------------------------
    section("SUMMARY")
    dev = numeric["per_slice_max_abs_dev"]
    print(f"  raw takes                 : {len(all_takes)} found, {len(takes)} included")
    print(f"  episodes in dataset       : {info.get('total_episodes')}")
    print(f"  frames in dataset         : {info.get('total_frames')}  (expected {expected_total})")
    print(f"  max |d state q|           : {dev['q']:.3e}   (tol {args.tol:g})")
    print(f"  max |d state tcp_p|       : {dev['tcp_p']:.3e}")
    print(f"  max |d state rot6d|       : {dev['rot6d']:.3e}")
    print(f"  max |d state grip_pos|    : {dev['grip_pos']:.3e}")
    print(f"  max |d action dp|         : {dev['dp']:.3e}")
    print(f"  max |d action drot|       : {dev['drot']:.3e}")
    print(f"  max |d action grip_cmd|   : {dev['grip_cmd']:.3e}")
    print(f"  6-D orthonormality dev    : {numeric['ortho_max_dev']:.3e} (tol {ORTHO_TOL:g})")
    print(f"  last-frame residual motion: {numeric['last_frame_max_motion']:.3e}")
    ts = numeric["tau_stats"]
    print(f"  tau_q (own estimate)      : min {ts['min']:.2f} / median {ts['median']:.2f} / "
          f"max {ts['max']:.2f} s")
    print(f"  tau_q residual            : min {ts['residual_min_m'] * 1000:.2f} / median "
          f"{ts['residual_median_m'] * 1000:.2f} / max {ts['residual_max_m'] * 1000:.2f} mm")
    print(f"  frames clamped past joints: {numeric['n_frames_clamped_past_joint_stream']}")
    if corr is not None and len(corr):
        print(f"  RGB video corr (n={len(corr):3d})   : min {corr.min():.5f}  mean {corr.mean():.5f}")
    else:
        print("  RGB video corr            : not evaluated")
    print(f"  action space              : {(eef_json or {}).get('action_space_id')!r}")
    print(f"  task                      : {args.task!r}")
    print()
    print("  RESULT TABLE")
    print("  " + "-" * 74)
    for status, name, _ in rep.rows:
        print(f"  {status:6s}  {name}")
    print("  " + "-" * 74)
    print()
    print(f"  checks: {rep.n_pass} PASS, {rep.n_fail} FAIL, {rep.n_warn} WARN, {rep.n_skip} SKIP")
    verdict = "PASS" if rep.n_fail == 0 else "FAIL"
    print()
    print(f"  OVERALL: {verdict}")
    print("=" * 78)

    if args.json_path:
        results = {
            "verdict": verdict,
            "interpreter": sys.executable,
            "validator": os.path.abspath(__file__),
            "ur_kin": os.path.abspath(ur_kin.__file__),
            "raw_root": raw_root,
            "lerobot_root": lr_root,
            "repo_id": args.repo_id,
            "task": args.task,
            "action_space_id": ACTION_SPACE_ID,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "tool_z_m": TOOL_Z_M,
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
            "features": {k: {"dtype": v.get("dtype"), "shape": v.get("shape"),
                             "names": v.get("names")}
                         for k, v in (info.get("features") or {}).items()},
            "numeric": numeric,
            "rgb_video_corr": ({"n": int(len(corr)), "min": float(corr.min()),
                                "mean": float(corr.mean())} if corr is not None and len(corr)
                               else None),
            "eef_space_json": eef_json,
            "source_takes_json": src_json,
            "thresholds": {"tol": args.tol, "ortho_tol": ORTHO_TOL,
                           "video_fail_corr": args.video_fail_corr,
                           "video_warn_corr": args.video_warn_corr},
            "counts": {"pass": rep.n_pass, "fail": rep.n_fail,
                       "warn": rep.n_warn, "skip": rep.n_skip, "total": len(rep.rows)},
            "rows": [{"status": s, "check": n, "detail": d} for s, n, d in rep.rows],
        }
        with open(args.json_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"  wrote {args.json_path}")

    return 0 if rep.n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
