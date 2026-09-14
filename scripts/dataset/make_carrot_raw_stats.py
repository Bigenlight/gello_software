#!/usr/bin/env python3
"""Generate `dataset_stats.json` for the RAW release `Bigenlight/carrot_in_pot_raw`.

Reads every `take_*/vectors.h5`, `take_*/depth.h5`, `take_*/cam1.mp4` and
`take_*/cam2.mp4` in the staging directory and writes one machine-readable JSON with
every number quoted in `README.md` / `DATA_DICTIONARY.md`.

Read-only with respect to the take folders: the script opens the HDF5 files with
h5py mode "r" and only ever *reads* the MP4s (ffprobe / OpenCV). The single file it
writes is `<data-dir>/dataset_stats.json`.

Requires system python3 with h5py + numpy + cv2. Video frame counts come from
`ffprobe -count_frames` when ffprobe is on PATH (exact, decodes every frame); the
fallback is OpenCV's `CAP_PROP_FRAME_COUNT` (container metadata, can be wrong). The
method actually used is recorded in the output as `video_frame_count_method`.

Usage:
    python3 make_carrot_raw_stats.py \
        --data /home/laptop3/youngwoong_ws/Put_carrot_in_pot \
        [--out <path>] [--dataset Bigenlight/carrot_in_pot_raw]

Definitions (kept identical to the sibling cube_in_cup_raw release so the two JSONs
are comparable):

  * mean rate  = (N-1) / (t_last - t_first) per take, then the MEDIAN across takes.
    Reported next to the per-take min/max.
  * dt percentiles are POOLED over all takes (every inter-sample interval of every
    take goes into one array), because the robot streams are bursty and a per-take
    median would hide the bimodality.
  * duration_s = max over all groups of that group's last t_rel_s.
  * value ranges are pooled over all takes; mean/std are computed from streamed
    sum / sum-of-squares so no channel is ever fully materialised twice.
"""

import argparse
import datetime as _dt
import json
import math
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

VECTOR_GROUPS = (
    "cam1_frames",
    "cam2_frames",
    "command",
    "ur_joint_states",
    "tcp_pose",
    "wrench",
    "gripper",
    "gello_joint_states",
    "synchronized",
)
# Groups whose channels go into `value_ranges` (synchronized is empty; the two
# cam*_frames groups contribute only frame_idx, t_rel_s is a clock not a value).
RANGE_GROUPS = (
    "command",
    "ur_joint_states",
    "tcp_pose",
    "wrench",
    "gripper",
    "gello_joint_states",
)
DEPTH_CAMS = ("cam1", "cam2")

# grip_cmd is 0 = open, 1 = closed. A "closure" is a rising edge through this level.
GRIP_CLOSED_LEVEL = 0.7
# A closure whose measured grip_pos peaks above this is a closure on NOTHING (the
# fingers met each other, not the carrot). Measured empty-close plateau is 0.898.
GRIP_EMPTY_LEVEL = 0.80


# --------------------------------------------------------------------------- utils
def _r(x, n=4):
    """Round, mapping numpy scalars to plain floats and non-finite values to None."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, n)


class Accum:
    """Streaming min/max/mean/std accumulator for a pooled channel."""

    def __init__(self):
        self.n = 0
        self.lo = math.inf
        self.hi = -math.inf
        self.s = 0.0
        self.ss = 0.0

    def add(self, a):
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            return
        self.n += int(a.size)
        self.lo = min(self.lo, float(a.min()))
        self.hi = max(self.hi, float(a.max()))
        self.s += float(a.sum())
        self.ss += float(np.square(a).sum())

    def out(self):
        if self.n == 0:
            return {"min": None, "max": None, "mean": None, "std": None, "n": 0}
        mean = self.s / self.n
        var = max(self.ss / self.n - mean * mean, 0.0)
        return {
            "min": _r(self.lo, 6),
            "max": _r(self.hi, 6),
            "mean": _r(mean, 6),
            "std": _r(math.sqrt(var), 6),
            "n": self.n,
        }


def mean_rate(t):
    """(N-1)/(t_last-t_first); None when the stream has < 2 samples or no span."""
    if t is None or len(t) < 2:
        return None
    span = float(t[-1]) - float(t[0])
    if span <= 0:
        return None
    return (len(t) - 1) / span


def nearest_idx(src_t, query_t):
    """Index into src_t of the sample nearest each query_t (both sorted ascending)."""
    j = np.clip(np.searchsorted(src_t, query_t), 1, len(src_t) - 1)
    left, right = src_t[j - 1], src_t[j]
    return np.where(query_t - left <= right - query_t, j - 1, j)


# ------------------------------------------------------------------ video probing
def ffprobe_frames(path):
    """Exact frame count via `ffprobe -count_frames` (decodes), or None on failure."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "default=nokey=1:noprint_wrappers=1", path,
            ],
            capture_output=True, text=True, timeout=600, check=True,
        ).stdout.strip()
        return int(out)
    except Exception:
        return None


def ffprobe_format(path):
    """(codec_name, width, height, pix_fmt, avg_frame_rate) or None."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate",
                "-of", "default=nokey=1:noprint_wrappers=1", path,
            ],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout.split()
        return tuple(out)
    except Exception:
        return None


def cv2_frames(path):
    """Container-metadata frame count (fallback; not guaranteed exact)."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


# ------------------------------------------------------------------ depth helpers
def decode_depth_png(png_cell):
    """uint16 (H, W) millimetre array from one `depth.h5` `png` cell."""
    import cv2

    buf = np.frombuffer(np.asarray(png_cell, dtype=np.uint8).tobytes(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("cv2.imdecode returned None (not a PNG?)")
    return img


def attrs_plain(attrs):
    out = {}
    for k, v in attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        elif isinstance(v, np.ndarray):
            v = [_r(x, 8) for x in v.reshape(-1).tolist()]
        elif isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float):
            v = _r(v, 8)
        out[k] = v
    return out


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/home/laptop3/youngwoong_ws/Put_carrot_in_pot",
                    help="staging directory holding the take_* folders")
    ap.add_argument("--out", default=None, help="output JSON (default <data>/dataset_stats.json)")
    ap.add_argument("--dataset", default="Bigenlight/carrot_in_pot_raw")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data)
    out_path = args.out or os.path.join(data_dir, "dataset_stats.json")

    takes = sorted(d for d in os.listdir(data_dir)
                   if d.startswith("take_") and os.path.isdir(os.path.join(data_dir, d)))
    if not takes:
        sys.exit(f"no take_* folders under {data_dir}")

    have_ffprobe = shutil.which("ffprobe") is not None
    frame_method = "ffprobe -count_frames" if have_ffprobe else "cv2 CAP_PROP_FRAME_COUNT"
    print(f"{len(takes)} takes in {data_dir}; video frame counts via {frame_method}",
          file=sys.stderr)

    # pooled accumulators ---------------------------------------------------
    ranges = {}            # "group/chan" -> Accum
    dt_pool = {g: [] for g in VECTOR_GROUPS if g != "synchronized"}
    dt_pool.update({f"depth_{c}": [] for c in DEPTH_CAMS})
    rate_pool = {k: [] for k in dt_pool}
    gello_minus_ur = {f"q{i}": [] for i in range(1, 7)}
    gello_take_median = {f"q{i}": [] for i in range(1, 7)}
    corr_gello_ur = {f"q{i}": [] for i in range(1, 7)}
    corr_gello_cmd = {f"q{i}": [] for i in range(1, 7)}
    corr_dgello_dcmd = {f"q{i}": [] for i in range(1, 7)}
    cmd_track_err = {f"q{i}": Accum() for i in range(1, 7)}
    cmd_track_err_take_median = {f"q{i}": [] for i in range(1, 7)}
    nan_or_inf_hits = []
    abs_path_hits = []
    stray_files = []
    per_take = []
    cam_diff = {}
    depth_cam_diff = {"cam1": {}, "cam2": {}}
    video_formats = set()
    cam_meta = {c: None for c in DEPTH_CAMS}      # canonical camera_info/extrinsics
    cam_meta_mismatch = {c: [] for c in DEPTH_CAMS}
    regrasp = []
    columns_attr_types = set()
    group_sets = set()
    depth_max_mm = 0
    depth_over_10m_px = 0
    depth_frames_sampled = 0

    for ti, take in enumerate(takes, 1):
        tdir = os.path.join(data_dir, take)
        print(f"[{ti:2d}/{len(takes)}] {take}", file=sys.stderr)

        # cleanliness -------------------------------------------------------
        expected = {"vectors.h5", "depth.h5", "cam1.mp4", "cam2.mp4"}
        for root, dirs, files in os.walk(tdir):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), tdir)
                if rel not in expected:
                    stray_files.append(f"{take}/{rel}")
            for dn in dirs:
                stray_files.append(f"{take}/{dn}/")

        rec = {"take": take}
        vec_path = os.path.join(tdir, "vectors.h5")
        with h5py.File(vec_path, "r") as f:
            group_sets.add(tuple(sorted(f.keys())))
            last_t = []
            for g in VECTOR_GROUPS:
                grp = f[g]
                columns_attr_types.add(type(grp.attrs.get("columns")).__name__)
                t = grp["t_rel_s"][:]
                rec[f"rows_{g}"] = int(t.shape[0])
                r = mean_rate(t)
                rec[f"rate_{g}"] = _r(r, 2)
                if g != "synchronized":
                    if r is not None:
                        rate_pool[g].append(r)
                    if t.size > 1:
                        dt_pool[g].append(np.diff(t))
                if t.size:
                    last_t.append(float(t[-1]))
                # channel ranges + finiteness
                for ch in grp:
                    a = grp[ch][:]
                    if a.size and not np.isfinite(a).all():
                        nan_or_inf_hits.append(f"{take}:{g}/{ch}")
                    if g in RANGE_GROUPS and ch != "t_rel_s":
                        ranges.setdefault(f"{g}/{ch}", Accum()).add(a)
                    elif g in ("cam1_frames", "cam2_frames") and ch == "frame_idx":
                        ranges.setdefault(f"{g}/{ch}", Accum()).add(a)

            rec["duration_s"] = _r(max(last_t) if last_t else 0.0, 2)

            # leader vs follower joint offset, nearest-timestamp aligned ------
            gt = f["gello_joint_states"]["t_rel_s"][:]
            ut = f["ur_joint_states"]["t_rel_s"][:]
            ct = f["command"]["t_rel_s"][:]
            if gt.size > 1 and ut.size > 1 and ct.size > 1:
                ju = nearest_idx(ut, gt)       # follower sample nearest each leader sample
                jc = nearest_idx(ct, gt)       # command sample nearest each leader sample
                juc = nearest_idx(ut, ct)      # follower sample nearest each command sample
                for i in range(1, 7):
                    g = f["gello_joint_states"][f"q{i}"][:]
                    u = f["ur_joint_states"][f"q{i}"][:]
                    c = f["command"][f"cmd{i}"][:]
                    d = g - u[ju]
                    gello_minus_ur[f"q{i}"].append(d)
                    gello_take_median[f"q{i}"].append(float(np.median(d)))
                    corr_gello_ur[f"q{i}"].append(float(np.corrcoef(g, u[ju])[0, 1]))
                    corr_gello_cmd[f"q{i}"].append(float(np.corrcoef(g, c[jc])[0, 1]))
                    corr_dgello_dcmd[f"q{i}"].append(
                        float(np.corrcoef(np.diff(g), np.diff(c[jc]))[0, 1]))
                    # how well the follower tracks its own command
                    err = np.abs(c - u[juc])
                    cmd_track_err[f"q{i}"].add(err)
                    cmd_track_err_take_median[f"q{i}"].append(float(np.median(err)))

            # gripper closures -------------------------------------------------
            gr = f["gripper"]
            gtt, gcmd, gpos = gr["t_rel_s"][:], gr["grip_cmd"][:], gr["grip_pos"][:]
            closed = gcmd >= GRIP_CLOSED_LEVEL
            rises = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
            falls = np.flatnonzero(closed[:-1] & (~closed[1:])) + 1
            closures = []
            for e in rises:
                after = falls[falls > e]
                end = int(after[0]) if after.size else len(gtt) - 1
                closures.append({
                    "t_close_s": _r(gtt[e], 2),
                    "t_open_s": _r(gtt[end], 2) if after.size else None,
                    "grip_pos_peak": _r(gpos[e:end + 1].max(), 4),
                })
            rec["n_grip_closures"] = len(closures)
            rec["grip_closures"] = closures
            rec["grip_pos_max"] = _r(gpos.max(), 4)
            if len(closures) > 1:
                regrasp.append({"take": take, "closures": closures})

        # depth -------------------------------------------------------------
        depth_path = os.path.join(tdir, "depth.h5")
        with h5py.File(depth_path, "r") as f:
            for cam in DEPTH_CAMS:
                grp = f[cam]
                t = grp["t_rel_s"][:]
                n = int(grp["png"].shape[0])
                rec[f"rows_depth_{cam}"] = n
                r = mean_rate(t)
                rec[f"rate_depth_{cam}"] = _r(r, 2)
                if r is not None:
                    rate_pool[f"depth_{cam}"].append(r)
                if t.size > 1:
                    dt_pool[f"depth_{cam}"].append(np.diff(t))
                rec[f"depth_{cam}_minus_{cam}_frames"] = n - rec[f"rows_{cam}_frames"]
                if n != rec[f"rows_{cam}_frames"]:
                    depth_cam_diff[cam][take] = n - rec[f"rows_{cam}_frames"]

                # quality on one sampled MID frame
                if n:
                    mid = n // 2
                    d = decode_depth_png(grp["png"][mid])
                    depth_max_mm = max(depth_max_mm, int(d.max()))
                    depth_over_10m_px += int((d > 10000).sum())
                    depth_frames_sampled += 1
                    valid = d > 0
                    nv = int(valid.sum())
                    rec[f"depth_sample_idx_{cam}"] = mid
                    rec[f"depth_shape_{cam}"] = list(d.shape)
                    rec[f"depth_valid_pct_{cam}"] = _r(100.0 * nv / d.size, 2)
                    rec[f"depth_median_range_m_{cam}"] = (
                        _r(float(np.median(d[valid])) / 1000.0, 4) if nv else None
                    )
                    rec[f"depth_p05_range_m_{cam}"] = (
                        _r(float(np.percentile(d[valid], 5)) / 1000.0, 4) if nv else None
                    )
                    rec[f"depth_p95_range_m_{cam}"] = (
                        _r(float(np.percentile(d[valid], 95)) / 1000.0, 4) if nv else None
                    )

                    # second sample: the frame nearest the FINAL gripper closure, i.e.
                    # the instant the fingers are on the carrot. For the wrist camera
                    # that is the worst case for the D435 minimum range.
                    if rec["grip_closures"]:
                        t_grasp = rec["grip_closures"][-1]["t_close_s"]
                        gi = int(nearest_idx(t, np.array([t_grasp]))[0]) if t.size > 1 else 0
                        dg = decode_depth_png(grp["png"][gi])
                        depth_max_mm = max(depth_max_mm, int(dg.max()))
                        depth_over_10m_px += int((dg > 10000).sum())
                        depth_frames_sampled += 1
                        vg = dg > 0
                        ng = int(vg.sum())
                        rec[f"depth_grasp_idx_{cam}"] = gi
                        rec[f"depth_grasp_t_s_{cam}"] = _r(t[gi], 2)
                        rec[f"depth_valid_pct_at_grasp_{cam}"] = _r(100.0 * ng / dg.size, 2)
                        rec[f"depth_median_range_m_at_grasp_{cam}"] = (
                            _r(float(np.median(dg[vg])) / 1000.0, 4) if ng else None)

                # camera_info / extrinsics: must be identical across all takes
                meta = {
                    "group_attrs": attrs_plain(grp.attrs),
                    "camera_info": attrs_plain(grp["camera_info"].attrs),
                    "extrinsics_depth_to_color": attrs_plain(
                        grp["extrinsics_depth_to_color"].attrs),
                }
                if cam_meta[cam] is None:
                    cam_meta[cam] = meta
                elif meta != cam_meta[cam]:
                    cam_meta_mismatch[cam].append(take)

        rec["bytes_vectors_h5"] = os.path.getsize(vec_path)
        rec["bytes_depth_h5"] = os.path.getsize(depth_path)

        # videos ------------------------------------------------------------
        for cam in ("cam1", "cam2"):
            mp4 = os.path.join(tdir, f"{cam}.mp4")
            rec[f"bytes_{cam}_mp4"] = os.path.getsize(mp4)
            n = ffprobe_frames(mp4) if have_ffprobe else None
            if n is None:
                n = cv2_frames(mp4)
            rec[f"{cam}_video_frames"] = int(n)
            rec[f"{cam}_h5_video_match"] = bool(n == rec[f"rows_{cam}_frames"])
            fmt = ffprobe_format(mp4) if have_ffprobe else None
            if fmt:
                video_formats.add(fmt)

        rec["cam1_minus_cam2"] = rec["rows_cam1_frames"] - rec["rows_cam2_frames"]
        if rec["cam1_minus_cam2"]:
            cam_diff[take] = rec["cam1_minus_cam2"]

        # absolute-path leakage: raw bytes of the small H5, attrs of the big one
        with open(vec_path, "rb") as fh:
            if b"/home/" in fh.read():
                abs_path_hits.append(f"{take}/vectors.h5")
        with h5py.File(depth_path, "r") as f:
            blob = json.dumps({c: attrs_plain(f[c].attrs) for c in DEPTH_CAMS})
            if "/home/" in blob:
                abs_path_hits.append(f"{take}/depth.h5")

        per_take.append(rec)

    # ------------------------------------------------------------------ rates
    rates = {}
    for key in dt_pool:
        dts = np.concatenate(dt_pool[key]) * 1000.0 if dt_pool[key] else np.array([])
        rr = np.array(rate_pool[key], dtype=np.float64)
        med_dt = float(np.median(dts)) if dts.size else None
        rates[key] = {
            "mean_rate_hz_median_over_takes": _r(np.median(rr), 2) if rr.size else None,
            "mean_rate_hz_min": _r(rr.min(), 2) if rr.size else None,
            "mean_rate_hz_max": _r(rr.max(), 2) if rr.size else None,
            "median_dt_ms": _r(med_dt, 2),
            "p05_dt_ms": _r(np.percentile(dts, 5), 2) if dts.size else None,
            "p95_dt_ms": _r(np.percentile(dts, 95), 2) if dts.size else None,
            "max_dt_ms": _r(dts.max(), 2) if dts.size else None,
            "rate_from_median_dt_hz": _r(1000.0 / med_dt, 2) if med_dt else None,
        }

    # -------------------------------------------------------- leader/follower
    leader_vs_follower = {
        "note": (
            "gello_* is the GELLO LEADER. Measured here: the leader joints in this session "
            "are NOT a fixed-offset mirror of the follower (contrast cube_in_cup_raw, where "
            "joints 1-5 tracked within +/-0.011 rad). Per-joint difference statistics are "
            "given for completeness; the per-take median of the difference itself scatters "
            "(see *_take_median_std), so there is no single offset to subtract. "
            "WHY: this session was recorded in END-EFFECTOR (EEF) delta teleop mode "
            "(ur7e_gello_real.launch.py robot_ip:=<ROBOT_IP> headless_mode:=true "
            "control_mode:=eef), not joint mode. The bridge computes the leader end-effector "
            "pose by forward kinematics of a virtual leader chain, takes the pose delta, and "
            "solves IK on the UR7e to produce the joint targets in `command`. The leader's own "
            "joint angles are a different kinematic solution, so joint-1 anti-correlation is "
            "expected and joint 6 is not a 2*pi wrap. cube_in_cup_raw / banana_in_pot_raw were "
            "joint-mode recordings, hence their ~0 offsets. `command` remains the true absolute "
            "joint target in both modes."
        ),
        "per_joint": {},
    }
    for k, chunks in gello_minus_ur.items():
        a = np.concatenate(chunks)
        tm = np.array(gello_take_median[k])
        leader_vs_follower["per_joint"][f"gello_{k}_minus_ur_{k}_rad"] = {
            "median": _r(np.median(a), 4),
            "mean": _r(a.mean(), 4),
            "p05": _r(np.percentile(a, 5), 4),
            "p95": _r(np.percentile(a, 95), 4),
            "min": _r(a.min(), 4),
            "max": _r(a.max(), 4),
            "take_median_min": _r(tm.min(), 4),
            "take_median_max": _r(tm.max(), 4),
            "take_median_std": _r(tm.std(), 4),
            "corr_gello_vs_ur_mean_over_takes": _r(np.mean(corr_gello_ur[k]), 3),
            "corr_gello_vs_cmd_mean_over_takes": _r(np.mean(corr_gello_cmd[k]), 3),
            "corr_dgello_vs_dcmd_mean_over_takes": _r(np.mean(corr_dgello_dcmd[k]), 3),
            "n": int(a.size),
        }

    command_tracking = {
        "note": ("|command cmd_i - ur_joint_states q_i| after nearest-timestamp alignment "
                 "of the follower onto the command clock. This is the servo following "
                 "error, pooled over all takes."),
        "per_joint": {
            f"j{i}": {
                "median_of_take_medians_rad": _r(np.median(cmd_track_err_take_median[f"q{i}"]), 4),
                "mean_rad": cmd_track_err[f"q{i}"].out()["mean"],
                "max_rad": cmd_track_err[f"q{i}"].out()["max"],
            }
            for i in range(1, 7)
        },
    }

    durs = np.array([t["duration_s"] for t in per_take], dtype=np.float64)
    total_bytes = sum(
        t["bytes_cam1_mp4"] + t["bytes_cam2_mp4"] + t["bytes_vectors_h5"] + t["bytes_depth_h5"]
        for t in per_take
    )

    def col(key):
        return np.array([t[key] for t in per_take if t.get(key) is not None], dtype=np.float64)

    aggregate = {
        "n_takes": len(per_take),
        "total_duration_s": _r(durs.sum(), 2),
        "total_cam1_frames": int(sum(t["rows_cam1_frames"] for t in per_take)),
        "total_cam2_frames": int(sum(t["rows_cam2_frames"] for t in per_take)),
        "total_depth_cam1_frames": int(sum(t["rows_depth_cam1"] for t in per_take)),
        "total_depth_cam2_frames": int(sum(t["rows_depth_cam2"] for t in per_take)),
        "total_bytes": int(total_bytes),
        "total_bytes_cam1_mp4": int(sum(t["bytes_cam1_mp4"] for t in per_take)),
        "total_bytes_cam2_mp4": int(sum(t["bytes_cam2_mp4"] for t in per_take)),
        "total_bytes_vectors_h5": int(sum(t["bytes_vectors_h5"] for t in per_take)),
        "total_bytes_depth_h5": int(sum(t["bytes_depth_h5"] for t in per_take)),
        "duration_median_s": _r(np.median(durs), 3),
        "duration_min_s": _r(durs.min(), 2),
        "duration_max_s": _r(durs.max(), 2),
        "duration_min_take": per_take[int(durs.argmin())]["take"],
        "duration_max_take": per_take[int(durs.argmax())]["take"],
        "nan_or_inf_hits": nan_or_inf_hits,
        "absolute_path_hits": abs_path_hits,
        "stray_files": stray_files,
        "video_frame_count_method": frame_method,
        "all_video_frame_counts_match_h5": all(
            t["cam1_h5_video_match"] and t["cam2_h5_video_match"] for t in per_take),
        "video_formats": sorted("|".join(v) for v in video_formats),
        "takes_with_cam1_cam2_diff": cam_diff,
        "takes_with_depth_cam1_colour_diff": depth_cam_diff["cam1"],
        "takes_with_depth_cam2_colour_diff": depth_cam_diff["cam2"],
        "columns_attr_python_types": sorted(columns_attr_types),
        "identical_group_set_across_takes": len(group_sets) == 1,
        "groups": sorted(group_sets)[0] if len(group_sets) == 1 else None,
        "takes_with_multiple_grip_closures": [r["take"] for r in regrasp],
        "depth_valid_pct_cam1": {
            "min": _r(col("depth_valid_pct_cam1").min(), 2),
            "median": _r(np.median(col("depth_valid_pct_cam1")), 2),
            "max": _r(col("depth_valid_pct_cam1").max(), 2),
        },
        "depth_valid_pct_cam2": {
            "min": _r(col("depth_valid_pct_cam2").min(), 2),
            "median": _r(np.median(col("depth_valid_pct_cam2")), 2),
            "max": _r(col("depth_valid_pct_cam2").max(), 2),
        },
        "depth_median_range_m_cam1": {
            "min": _r(col("depth_median_range_m_cam1").min(), 4),
            "median": _r(np.median(col("depth_median_range_m_cam1")), 4),
            "max": _r(col("depth_median_range_m_cam1").max(), 4),
        },
        "depth_median_range_m_cam2": {
            "min": _r(col("depth_median_range_m_cam2").min(), 4),
            "median": _r(np.median(col("depth_median_range_m_cam2")), 4),
            "max": _r(col("depth_median_range_m_cam2").max(), 4),
        },
        "depth_camera_info_identical_across_takes": {
            c: not cam_meta_mismatch[c] for c in DEPTH_CAMS},
        "depth_camera_info_mismatch_takes": cam_meta_mismatch,
        "depth_frames_sampled": depth_frames_sampled,
        "depth_max_mm_sampled": depth_max_mm,
        "depth_pixels_over_10000mm_sampled": depth_over_10m_px,
        "depth_valid_pct_at_grasp_cam1": {
            "min": _r(col("depth_valid_pct_at_grasp_cam1").min(), 2),
            "median": _r(np.median(col("depth_valid_pct_at_grasp_cam1")), 2),
            "max": _r(col("depth_valid_pct_at_grasp_cam1").max(), 2),
        },
        "depth_valid_pct_at_grasp_cam2": {
            "min": _r(col("depth_valid_pct_at_grasp_cam2").min(), 2),
            "median": _r(np.median(col("depth_valid_pct_at_grasp_cam2")), 2),
            "max": _r(col("depth_valid_pct_at_grasp_cam2").max(), 2),
        },
    }

    doc = {
        "dataset": args.dataset,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": (
            f"make_carrot_raw_stats.py — h5py + cv2 + ffprobe audit over all {len(per_take)} "
            "take folders (vectors.h5, depth.h5, cam1.mp4, cam2.mp4). mean rate = "
            "(N-1)/(t_last-t_first) per take, then median across takes; dt percentiles pooled "
            f"over all takes. Video frame counts from `{frame_method}`. Value ranges pooled "
            "over all takes. Depth quality measured on ONE sampled mid frame per camera per "
            "take (decoded with cv2.imdecode(..., IMREAD_UNCHANGED))."
        ),
        "aggregate": aggregate,
        "rates": rates,
        "leader_vs_follower": leader_vs_follower,
        "command_tracking": command_tracking,
        "depth_cameras": cam_meta,
        "regrasp_takes": regrasp,
        "value_ranges": {k: ranges[k].out() for k in sorted(ranges)},
        "per_take": per_take,
    }

    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
