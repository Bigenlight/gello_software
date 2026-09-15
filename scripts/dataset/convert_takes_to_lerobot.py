#!/usr/bin/env python3
"""
Convert gello_recorder takes (vectors.h5 + cam1.mp4 + cam2.mp4 per take, recorded
by the FIXED recorder that stores ROS header stamps) into a task-agnostic,
RGB-only LeRobot v3.0 dataset.

Recipe: scratchpad/bowl/LEROBOT_RECIPE.md (2026-09-16). An independent validator
is written against the same text and must re-derive every row bit-for-bit, so
nothing here may deviate from it.

Recipe
------
  - fps = 30 (camera rate). One episode per take, in take-name order.
  - Master clock = CAPTURE instant of every cam1 frame, expressed in the take's
    `t_rel_s` frame:
        t0_off      = median(ur_joint_states.stamp_s - ur_joint_states.t_rel_s)
        cam1_cap[i] = cam1_frames.stamp_s[i] - t0_off
    `stamp_s` is the ROS header stamp (absolute epoch seconds); `t_rel_s` is the
    recorder's callback time relative to take start. The median over all
    ur_joint_states rows is robust to the per-row callback jitter (up to ~50 ms
    on this machine) and pins the take's t_rel origin in ROS time.
  - Streams WITH a header stamp are placed on their own header stamps:
        ur_t   = ur_joint_states.stamp_s - t0_off
        cam2_t = cam2_frames.stamp_s     - t0_off
    Streams WITHOUT one (command, gripper) are placed on callback time, which is
    fresh (~ms) with the fixed recorder:
        cmd_t = command.t_rel_s ; grip_t = gripper.t_rel_s
  - observation.state (7) = [ur q1..q6 at nearest(ur_t, cam1_cap),
                             gripper.grip_pos at nearest(grip_t, cam1_cap)]
  - action            (7) = [command cmd1..6 at nearest(cmd_t, cam1_cap),
                             ffill_bfill(gripper.grip_cmd) at nearest(grip_t, cam1_cap)]
  - observation.images.cam1[i] = decoded cam1.mp4 frame i (BGR->RGB, HWC uint8)
  - observation.images.cam2[i] = decoded cam2.mp4 frame nearest(cam2_t, cam1_cap[i])
  - `nearest` = nearest_idx below (searchsorted, ties -> earlier index), the same
    function the carrot converter uses.
  - LeRobot timestamps are i / 30 (lerobot assigns frame_index / fps itself).
  - NO frames dropped, NO lag subtracted, NO depth. gello_* streams, tcp_pose,
    wrench and the (empty) synchronized table are ignored.

Why this differs from convert_carrot_to_lerobot.py
--------------------------------------------------
The carrot corpus was recorded by the OLD GUI recorder, which stamped every row
at callback-execution time on a single rclpy spin thread that depth recording
starved: `ur_joint_states` rows were stamped ~0.9 s late (queue depth / rate),
so that converter had to subtract a per-take tau from the ur clock, drop the
stale tail it left behind, and carry a "timestamp_correction" block.

The recorder has since been fixed (header `stamp_s` stored per row, smaller
subscription queues, a separate writer thread, starvation warnings) and depth is
default-OFF. Corpora recorded by it -- like the one this script is for -- carry
the true capture instant of every camera frame and joint sample, so alignment
is done on those stamps directly and there is nothing to correct. The remaining
~0.14 s by which the measured joint state trails the command is REAL servo
tracking of the UR7e behind the streamed target and is deliberately kept in the
data: subtracting it would fabricate a robot that reaches its target instantly,
which a policy trained on the data would then never see at inference time.

Because there is no correction, this script exposes no lag/tail options at all,
and it REFUSES corpora that need the old treatment (any `depth.h5`, or a
`cam1_frames`/`cam2_frames`/`ur_joint_states` table without `stamp_s`) with a
pointer to the carrot converter. Point-wise checks are fail-closed too: cam1 and
cam2 `frame_idx` must be exactly 0..N-1 and each mp4 must decode exactly its
table's row count -- a short video is an error, never a padded or truncated
episode.

Side file written AFTER ds.finalize() (so lerobot's writer cannot clobber
meta/): meta/source_takes.json, schema "gello_recorder/lerobot_source_takes/2",
with the alignment block and one entry per episode (take name, n_frames,
t0_off_s, median(cam1 stamp - nearest ur stamp), cam2 frame reuse count). No
absolute local path is written anywhere in the output; the script scans the
whole output tree for "/home/" at the end and fails if it finds one.

Python 3.12, interpreter /home/laptop3/youngwoong_ws/lr_env/bin/python
(lerobot 0.6.1, codebase v3.0).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Iterable, Iterator, Sequence

import cv2
import h5py
import numpy as np

FPS = 30
ROBOT_TYPE = "ur7e_gello"

STATE_NAMES = ["ur_q1", "ur_q2", "ur_q3", "ur_q4", "ur_q5", "ur_q6", "grip_pos"]
ACTION_NAMES = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd6", "grip_cmd"]

CAMS = ("cam1", "cam2")
RGB_SHAPE = (720, 1280, 3)

SOURCE_TAKES_SCHEMA = "gello_recorder/lerobot_source_takes/2"
# Tables that MUST carry a ROS header stamp column for this recipe to apply.
STAMPED_GROUPS = ("cam1_frames", "cam2_frames", "ur_joint_states")
# Every lookup source clock must be sorted ascending for nearest_idx to be valid.
SOURCE_CLOCKS = (
    ("ur_joint_states", "stamp_s"),
    ("cam2_frames", "stamp_s"),
    ("command", "t_rel_s"),
    ("gripper", "t_rel_s"),
)

ALIGNMENT_NOTE = (
    "No timestamp correction is applied. This corpus was recorded by the fixed "
    "gello_recorder (per-row ROS header stamp_s, small subscription queues, separate "
    "writer thread, depth off), so cam1/cam2/ur_joint_states rows carry their true "
    "capture instants and are aligned on those directly; command and gripper have no "
    "header stamp and use callback time, which is fresh (~ms) with the fixed recorder. "
    "The ~0.14 s by which the measured joint state trails the command is real UR7e "
    "servo tracking of the streamed target and is kept in the data on purpose. "
    "Contrast: the carrot corpus (old recorder) needed ur_joint_states shifted by "
    "~0.9 s and its stale tail dropped; that treatment does NOT apply here."
)


# --------------------------------------------------------------------------- #
# resampling helpers (identical semantics to convert_carrot_to_lerobot.py)
# --------------------------------------------------------------------------- #
def nearest_idx(src_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """For each query timestamp, index of the nearest src sample (src_t sorted asc).

    Ties go to the earlier index. Queries outside [src_t[0], src_t[-1]] clamp to
    the end samples (no extrapolation). The caller guarantees len(src_t) >= 2 and
    ascending order (preflight_take checks both).
    """
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


# --------------------------------------------------------------------------- #
# preflight: refuse the whole corpus BEFORE anything is written
# --------------------------------------------------------------------------- #
class TakeRefused(RuntimeError):
    """A take does not satisfy the recipe's preconditions."""


def preflight_take(take_dir: str) -> dict[str, Any]:
    """Cheap HDF5-only checks (no video decode). Raises TakeRefused on the first violation.

    Returns {"n_cam1": N, "n_cam2": N2, "cam1_stamp_monotonic": bool}.

    `cam1_stamp_monotonic` is informational: cam1 is the QUERY side of every
    nearest-timestamp lookup, so an out-of-order pair of cam1 stamps (seen once in
    the bowl corpus, frames 17/18 of take_32, -33 ms) still maps each frame to a
    well-defined state/action per the recipe. It is reported, not refused, but it
    does mean the cam2 index map is not guaranteed monotone -- which is why cam2
    is decoded whole and indexed, not streamed.
    """
    name = os.path.basename(take_dir)
    if os.path.exists(os.path.join(take_dir, "depth.h5")):
        raise TakeRefused(
            f"{name}: has depth.h5 -> this is a depth-era (old recorder) take. "
            "Use scripts/dataset/convert_carrot_to_lerobot.py for that corpus."
        )
    for fn in ("vectors.h5", "cam1.mp4", "cam2.mp4"):
        if not os.path.isfile(os.path.join(take_dir, fn)):
            raise TakeRefused(f"{name}: missing {fn}")

    with h5py.File(os.path.join(take_dir, "vectors.h5"), "r") as f:
        for g in STAMPED_GROUPS + ("command", "gripper"):
            if g not in f:
                raise TakeRefused(f"{name}: vectors.h5 has no '{g}' table")
        for g in STAMPED_GROUPS:
            if "stamp_s" not in f[g]:
                raise TakeRefused(
                    f"{name}: '{g}' has no stamp_s column -> recorded by the PRE-FIX "
                    "recorder (callback-time rows only). Use "
                    "scripts/dataset/convert_carrot_to_lerobot.py, which knows how to "
                    "correct those clocks."
                )
        for k in range(1, 7):
            for g, col in (("ur_joint_states", f"q{k}"), ("command", f"cmd{k}")):
                if col not in f[g]:
                    raise TakeRefused(f"{name}: '{g}' lacks column {col}")
        for col in ("grip_pos", "grip_cmd"):
            if col not in f["gripper"]:
                raise TakeRefused(f"{name}: 'gripper' lacks column {col}")

        counts: dict[str, int] = {}
        for cam in CAMS:
            grp = f[f"{cam}_frames"]
            if "frame_idx" not in grp:
                raise TakeRefused(f"{name}: '{cam}_frames' has no frame_idx column")
            fi = np.asarray(grp["frame_idx"][:])
            n = len(fi)
            if n < 1:
                raise TakeRefused(f"{name}: '{cam}_frames' is empty")
            if not np.array_equal(fi, np.arange(n)):
                raise TakeRefused(
                    f"{name}: '{cam}_frames'.frame_idx is not contiguous 0..{n - 1} "
                    "(row i == mp4 frame i does not hold; refusing rather than guessing)"
                )
            if len(grp["stamp_s"]) != n:
                raise TakeRefused(f"{name}: '{cam}_frames' stamp_s/frame_idx length mismatch")
            counts[cam] = n

        for g, col in SOURCE_CLOCKS:
            v = np.asarray(f[g][col][:])
            if len(v) < 2:
                raise TakeRefused(f"{name}: '{g}' has {len(v)} rows; need >= 2 for nearest lookup")
            if not np.all(np.isfinite(v)) or np.any(np.diff(v) < 0):
                raise TakeRefused(
                    f"{name}: '{g}'.{col} is not finite/ascending; nearest_idx would be invalid"
                )
        ur = f["ur_joint_states"]
        if len(ur["t_rel_s"]) != len(ur["stamp_s"]):
            raise TakeRefused(f"{name}: 'ur_joint_states' t_rel_s/stamp_s length mismatch")

        cam1_stamp = np.asarray(f["cam1_frames"]["stamp_s"][:])
        cam1_mono = bool(np.all(np.diff(cam1_stamp) > 0))

    return {"n_cam1": counts["cam1"], "n_cam2": counts["cam2"], "cam1_stamp_monotonic": cam1_mono}


# --------------------------------------------------------------------------- #
# per-take arrays
# --------------------------------------------------------------------------- #
def load_take_arrays(h5_path: str) -> dict[str, Any]:
    """Alignment exactly per LEROBOT_RECIPE.md.

    Returns a dict with
      cam1_cap [N]   capture instant of master frame i (t_rel frame, via stamp_s - t0_off)
      cam2_map [N]   cam2.mp4 frame index for master frame i
      state    [N,7] float32   action [N,7] float32
      t0_off         median(ur.stamp_s - ur.t_rel_s)
      cam1_minus_ur_median_s   median over i of (cam1 stamp[i] - stamp of the ur row chosen for i)
      cam2_reuse     number of master frames whose cam2 index equals the previous frame's
    """
    with h5py.File(h5_path, "r") as f:
        ur = f["ur_joint_states"]
        ur_stamp = np.asarray(ur["stamp_s"][:], dtype=np.float64)
        ur_trel = np.asarray(ur["t_rel_s"][:], dtype=np.float64)
        t0_off = float(np.median(ur_stamp - ur_trel))

        cam1_cap = np.asarray(f["cam1_frames"]["stamp_s"][:], dtype=np.float64) - t0_off
        cam2_t = np.asarray(f["cam2_frames"]["stamp_s"][:], dtype=np.float64) - t0_off
        ur_t = ur_stamp - t0_off
        cmd_t = np.asarray(f["command"]["t_rel_s"][:], dtype=np.float64)
        grip_t = np.asarray(f["gripper"]["t_rel_s"][:], dtype=np.float64)
        n = len(cam1_cap)

        ur_j = nearest_idx(ur_t, cam1_cap)
        grip_j = nearest_idx(grip_t, cam1_cap)
        cmd_j = nearest_idx(cmd_t, cam1_cap)
        cam2_map = nearest_idx(cam2_t, cam1_cap)

        state = np.zeros((n, 7), dtype=np.float32)
        for k in range(6):
            state[:, k] = ur[f"q{k + 1}"][:][ur_j]
        state[:, 6] = f["gripper"]["grip_pos"][:][grip_j]

        action = np.zeros((n, 7), dtype=np.float32)
        for k in range(6):
            action[:, k] = f["command"][f"cmd{k + 1}"][:][cmd_j]
        action[:, 6] = ffill_bfill(np.asarray(f["gripper"]["grip_cmd"][:], dtype=np.float64))[grip_j]

    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(action))):
        raise RuntimeError(f"{h5_path}: non-finite state/action after resampling")

    cam1_minus_ur = float(np.median(cam1_cap - ur_t[ur_j]))
    cam2_reuse = int(np.count_nonzero(cam2_map[1:] == cam2_map[:-1])) if n > 1 else 0
    return {
        "cam1_cap": cam1_cap, "cam2_map": cam2_map, "state": state, "action": action,
        "t0_off": t0_off, "cam1_minus_ur_median_s": cam1_minus_ur, "cam2_reuse": cam2_reuse,
    }


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #
def open_reader(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    return cap


def _check_rgb(fr: np.ndarray, path: str, k: int) -> None:
    if fr.shape != RGB_SHAPE or fr.dtype != np.uint8:
        raise RuntimeError(f"{path}: frame {k} is {fr.shape}/{fr.dtype}, expected {RGB_SHAPE}/uint8")


def iter_frames_exact(path: str, expect_n: int, strict_tail: bool = True) -> Iterator[np.ndarray]:
    """Yield exactly `expect_n` RGB frames of an mp4, in decode order.

    Fails loudly if the file decodes FEWER frames than `expect_n` (the caller would
    otherwise silently pad/truncate the episode) and, when `strict_tail`, if it
    decodes MORE (row i == frame i is the whole basis of the recipe and an extra
    frame means that identity is unverified). strict_tail is only relaxed for the
    TEST-ONLY --max-frames truncation.
    """
    cap = open_reader(path)
    try:
        for k in range(expect_n):
            ok, fr = cap.read()
            if not ok or fr is None:
                raise RuntimeError(
                    f"{path}: decoded only {k} frames but the HDF5 table has {expect_n} rows "
                    "-> refusing (never pad or truncate)"
                )
            _check_rgb(fr, path, k)
            yield cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        if strict_tail:
            ok, _ = cap.read()
            if ok:
                raise RuntimeError(
                    f"{path}: decodes MORE than the {expect_n} HDF5 rows -> row i == frame i "
                    "is unverified; refusing"
                )
    finally:
        cap.release()


def decode_all_exact(path: str, expect_n: int) -> list[np.ndarray]:
    """Decode the whole mp4 into memory; must yield exactly `expect_n` frames."""
    return list(iter_frames_exact(path, expect_n, strict_tail=True))


def build_features() -> dict[str, dict[str, Any]]:
    feats: dict[str, dict[str, Any]] = {
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    }
    for cam in CAMS:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": RGB_SHAPE, "names": ["height", "width", "channels"],
        }
    return feats


def dir_size_bytes(root: str) -> int:
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(root)
        for f in fs
        if not os.path.islink(os.path.join(dp, f))
    )


def scan_for_local_paths(root: str, needle: bytes = b"/home/") -> list[str]:
    """Every file under `root` (text AND binary) that contains `needle`, as relative paths."""
    hits: list[str] = []
    for dp, _, fs in os.walk(root):
        for fn in fs:
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                continue
            with open(p, "rb") as fh:
                # Chunked with overlap so a needle straddling a chunk boundary is still found.
                prev = b""
                while True:
                    chunk = fh.read(8 << 20)
                    if not chunk:
                        break
                    if needle in prev[-(len(needle) - 1):] + chunk:
                        hits.append(os.path.relpath(p, root))
                        break
                    prev = chunk
    return hits


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def convert(
    data_root: str,
    out_root: str,
    repo_id: str,
    task: str,
    limit: int | None = None,
    max_frames: int | None = None,
    image_writer_processes: int = 2,
    image_writer_threads: int = 2,
    exclude: Sequence[str] = (),
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if os.path.exists(out_root):
        print(f"[!] output dir already exists: {out_root}", file=sys.stderr)
        print("    remove it first or pass a fresh --out", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(data_root):
        print(f"[!] --data is not a directory: {data_root}", file=sys.stderr)
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

    # --- preflight EVERY selected take before creating the dataset ----------
    # A corpus that needs the old-recorder treatment (or is internally
    # inconsistent) is refused up front, with nothing written to --out.
    pre: dict[str, dict[str, Any]] = {}
    for tk in takes:
        try:
            pre[tk] = preflight_take(tk)
        except TakeRefused as e:
            print(f"[!] REFUSED: {e}", file=sys.stderr)
            sys.exit(2)
    nonmono = [os.path.basename(t) for t in takes if not pre[t]["cam1_stamp_monotonic"]]
    if nonmono:
        print(f"[warn] cam1 stamp_s is not strictly increasing in {len(nonmono)} take(s): "
              f"{nonmono} -- frames keep their own capture stamp per the recipe (cam1 is the "
              "query side of every lookup); nothing is reordered or dropped.", flush=True)
    expected_total = sum(pre[t]["n_cam1"] for t in takes)
    print(f"[cfg] {len(takes)} takes, {expected_total} master frames, task={task!r}, "
          f"fps={FPS}, alignment=stamp_s (no lag correction, no frames dropped, RGB only)",
          flush=True)
    print(f"[cfg] image writer: {image_writer_processes} procs x {image_writer_threads} threads",
          flush=True)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=build_features(),
        root=out_root,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    episodes: list[dict[str, Any]] = []
    total_frames = 0
    t_start = time.time()

    for ti, tk in enumerate(takes):
        t0 = time.time()
        name = os.path.basename(tk)
        arr = load_take_arrays(os.path.join(tk, "vectors.h5"))
        n_master = len(arr["cam1_cap"])
        n = n_master
        if max_frames:
            n = min(n, max_frames)
        cam2_map = arr["cam2_map"]
        state, action = arr["state"], arr["action"]
        n_cam2_rows = pre[tk]["n_cam2"]

        # cam2 is decoded whole (must be exactly its table's row count), then
        # indexed by the nearest-timestamp map. Whole, not streamed: the map is
        # monotone only if cam1 stamps are, and one take in the wild is not.
        cam2_all = decode_all_exact(os.path.join(tk, "cam2.mp4"), n_cam2_rows)
        if int(cam2_map.max()) >= len(cam2_all):
            raise RuntimeError(f"{name}: cam2 index map exceeds decoded cam2 frames")

        # cam1 is the master: mp4 frame k == master frame k, streamed sequentially.
        cam1_iter = iter_frames_exact(os.path.join(tk, "cam1.mp4"), n, strict_tail=(n == n_master))
        for k, cam1_frame in enumerate(cam1_iter):
            ds.add_frame({
                "action": action[k],
                "observation.state": state[k],
                "observation.images.cam1": cam1_frame,
                "observation.images.cam2": cam2_all[int(cam2_map[k])],
                "task": task,
            })
        del cam2_all
        ds.save_episode()

        cam2_reuse = arr["cam2_reuse"] if n == n_master else int(
            np.count_nonzero(cam2_map[1:n] == cam2_map[:n - 1]))
        total_frames += n
        episodes.append({
            "episode_index": ti,
            "take": name,
            "n_frames": int(n),
            "t0_off_s": float(arr["t0_off"]),
            "cam1_stamp_minus_ur_stamp_median_s": float(arr["cam1_minus_ur_median_s"]),
            "cam2_frame_reuse_count": int(cam2_reuse),
        })
        print(
            f"[{ti + 1:2d}/{len(takes)}] {name:26s} frames={n:4d}  cam2_rows={n_cam2_rows:4d}  "
            f"t0_off={arr['t0_off']:.3f}  cam1-ur={arr['cam1_minus_ur_median_s'] * 1e3:+6.1f} ms  "
            f"cam2_reuse={cam2_reuse:2d}  {time.time() - t0:6.1f}s",
            flush=True,
        )

    ds.finalize()

    # --- sidecar AFTER finalize(), so lerobot's writer cannot clobber meta/ ---
    # Basenames only: this file must not leak the machine's directory layout.
    meta_dir = os.path.join(out_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    sidecar = {
        "schema": SOURCE_TAKES_SCHEMA,
        "task": task,
        "alignment": {
            "method": "stamp_s",
            "master_clock": "cam1_frames.stamp_s - t0_off",
            "t0_off_definition": "median(ur.stamp_s - ur.t_rel_s)",
            "joint_lag_subtracted_s": 0.0,
            "frames_dropped": 0,
            "note": ALIGNMENT_NOTE,
        },
        "episodes": episodes,
    }
    with open(os.path.join(meta_dir, "source_takes.json"), "w") as fh:
        json.dump(sidecar, fh, indent=2)

    leaks = scan_for_local_paths(out_root)
    if leaks:
        print(f"[!] output contains an absolute local path ('/home/') in: {leaks}", file=sys.stderr)
        sys.exit(3)

    wall = time.time() - t_start
    size_b = dir_size_bytes(out_root)
    print(
        f"\nDONE: {len(takes)} episodes, {total_frames} frames, "
        f"{size_b:,} bytes ({size_b / 1e9:.2f} GB), {wall:.0f} s ({wall / 60:.1f} min) wall "
        f"-> {os.path.basename(os.path.normpath(out_root))}  (no '/home/' in output: verified)",
        flush=True,
    )
    if max_frames:
        print(f"[!] --max-frames {max_frames} was set: this is a TEST dataset, not the release.")


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", required=True, help="staging dir containing take_* folders")
    ap.add_argument("--out", required=True, help="output dataset root (must not exist)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--task", required=True, help="natural-language task string stored per frame")
    ap.add_argument("--procs", type=int, default=2, help="lerobot image_writer_processes")
    ap.add_argument("--threads", type=int, default=2, help="lerobot image_writer_threads")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N takes")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="TEST ONLY: truncate every episode to N frames")
    ap.add_argument("--exclude", action="append", default=[],
                    help="take folder name to skip (repeatable); errors if not found")
    args = ap.parse_args(list(argv) if argv is not None else None)
    convert(
        args.data, args.out, args.repo_id, args.task,
        limit=args.limit, max_frames=args.max_frames,
        image_writer_processes=args.procs, image_writer_threads=args.threads,
        exclude=tuple(args.exclude),
    )


if __name__ == "__main__":
    main()
