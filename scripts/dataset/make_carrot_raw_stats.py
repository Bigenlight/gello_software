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
  * the `timestamp_lag` block measures the recorder timestamp artefact: how many
    seconds late `ur_joint_states` / `tcp_pose` / `wrench` rows are stamped,
    per take, by joint-space residual and by speed cross-correlation. It adds no
    other output and changes no other number.
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

# ---- recorder timestamp artefact (see the `timestamp_lag` block in the output) ----
# The GUI recorder stamps every row with the time its callback RAN, not the message
# header stamp, and it subscribes /joint_states with KEEP_LAST depth 100 and
# tcp_pose / wrench / commands with depth 50. In this session the single rclpy spin
# thread was slowed to ~60-69 Hz by the new depth recording, so every publisher
# faster than that keeps its queue permanently full and each row is stale by
# depth / publish-rate. The lag is therefore a PURE DELAY: the waveform is intact
# and a constant shift of `t_rel_s` recovers it. These constants only bound the
# search; the measured values are written to the output.
LAG_TAU_MAX_S = 1.5        # widest lag considered
LAG_TAU_STEP_S = 0.005     # 5 ms search grid
LAG_MIN_SAMPLES = 50       # skip a candidate lag with less overlap than this


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


# --------------------------------------------------- recorder timestamp artefact
def lag_grid():
    """The 5 ms candidate-lag grid, 0 .. LAG_TAU_MAX_S inclusive."""
    n = int(round(LAG_TAU_MAX_S / LAG_TAU_STEP_S))
    return np.arange(n + 1) * LAG_TAU_STEP_S


def joint_residual_at(tau, ur_t, ur_q, cmd_t, cmd_q):
    """|ur_q(t - tau) - cmd(t)| statistics, on the command clock.

    `ur_joint_states` is stamped late, so its clock is shifted EARLIER by `tau`
    and linearly interpolated onto the `command` timestamps. Returns
    (median-over-time of the 6-joint mean |err|, mean over everything, n), or
    None when the shift leaves too little overlap.

    The objective minimised is the MEDIAN over time, not the mean: both agree on
    0.900 s for most takes, but the mean is dominated by the few high-velocity
    transients where the real servo following error lives, and that pulls the
    argmin one grid step (5 ms) low on some takes. The median is the robust
    estimator of a constant delay.
    """
    shifted = ur_t - tau
    m = (cmd_t >= shifted[0]) & (cmd_t <= shifted[-1])
    if int(m.sum()) < LAG_MIN_SAMPLES:
        return None
    q = cmd_t[m]
    err = np.abs(
        np.stack([np.interp(q, shifted, ur_q[:, k]) for k in range(ur_q.shape[1])], axis=1)
        - cmd_q[m]
    )
    per_t = err.mean(axis=1)
    return float(np.median(per_t)), float(err.mean()), int(m.sum())


def joint_lag(ur_t, ur_q, cmd_t, cmd_q):
    """Best constant lag of `ur_joint_states` behind `command`, in seconds.

    Returns (tau_median_objective, residuals_there, tau_mean_objective). Both
    objectives are reported because both get published: the median one is stable
    at one value across the whole session, the mean one wanders by a single 5 ms
    grid step on some takes. They are the same measurement.
    """
    best = None
    best_mean = None
    for tau in lag_grid():
        got = joint_residual_at(tau, ur_t, ur_q, cmd_t, cmd_q)
        if got is None:
            continue
        if best is None or got[0] < best[1][0]:
            best = (float(tau), got)
        if best_mean is None or got[1] < best_mean[1]:
            best_mean = (float(tau), got[1])
    if best is None:
        return None
    return best[0], best[1], (best_mean[0] if best_mean else None)


def speed_series(t, x, dt=LAG_TAU_STEP_S):
    """Scalar speed ‖dx/dt‖ of a multi-channel signal on a uniform `dt` grid."""
    if t.size < 4 or float(t[-1] - t[0]) <= 4 * dt:
        return None, None
    grid = np.arange(float(t[0]), float(t[-1]), dt)
    y = np.stack([np.interp(grid, t, x[:, k]) for k in range(x.shape[1])], axis=1)
    return grid, np.linalg.norm(np.gradient(y, dt, axis=0), axis=1)


def xcorr_lag(g_lead, v_lead, g_late, v_late):
    """How many seconds `v_late` trails `v_lead`, by speed-profile correlation.

    Speed (not position) is correlated so that a constant spatial offset — e.g.
    the 0.174 m tool offset between `fk(q)` and `tcp_pose` — cannot bias the
    answer. Returns (lag_s, pearson_r) with lag_s > 0 meaning `v_late` is stamped
    late, or (None, None).
    """
    if g_lead is None or g_late is None:
        return None, None
    lo, hi = max(g_lead[0], g_late[0]), min(g_lead[-1], g_late[-1])
    if hi - lo < 1.0:
        return None, None
    grid = np.arange(lo, hi, LAG_TAU_STEP_S)
    a = np.interp(grid, g_lead, v_lead)
    b = np.interp(grid, g_late, v_late)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = int(round(LAG_TAU_MAX_S / LAG_TAU_STEP_S))
    best = (None, -np.inf)
    for k in range(n + 1):
        aa = a[: a.size - k] if k else a
        bb = b[k:] if k else b
        if aa.size < LAG_MIN_SAMPLES:
            continue
        r = float(np.dot(aa, bb) / aa.size)
        if r > best[1]:
            best = (k * LAG_TAU_STEP_S, r)
    if best[0] is None:
        return None, None
    return float(best[0]), float(best[1])


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
    ap.add_argument("--no-depth", dest="depth", action="store_false", default=None,
                    help="depth-free takes (3 files: vectors.h5 + cam1.mp4 + cam2.mp4). "
                         "Skips every depth section; auto-detected when no take has depth.h5")
    ap.add_argument("--depth", dest="depth", action="store_true",
                    help="require depth.h5 in every take (the default for the real release)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data)
    out_path = args.out or os.path.join(data_dir, "dataset_stats.json")

    takes = sorted(d for d in os.listdir(data_dir)
                   if d.startswith("take_") and os.path.isdir(os.path.join(data_dir, d)))
    if not takes:
        sys.exit(f"no take_* folders under {data_dir}")

    # Depth mode. Explicit flag wins; otherwise auto-detect from the takes on disk so
    # the real 4-file release keeps behaving exactly as before.
    if args.depth is None:
        have_depth = any(os.path.exists(os.path.join(data_dir, t, "depth.h5")) for t in takes)
    else:
        have_depth = bool(args.depth)
    if have_depth:
        missing = [t for t in takes if not os.path.exists(os.path.join(data_dir, t, "depth.h5"))]
        if missing:
            sys.exit(f"depth mode but {len(missing)} take(s) have no depth.h5, e.g. {missing[0]}")
    else:
        print("depth-free mode: expecting 3 files per take, skipping all depth sections",
              file=sys.stderr)

    have_ffprobe = shutil.which("ffprobe") is not None
    frame_method = "ffprobe -count_frames" if have_ffprobe else "cv2 CAP_PROP_FRAME_COUNT"
    print(f"{len(takes)} takes in {data_dir}; video frame counts via {frame_method}",
          file=sys.stderr)

    # pooled accumulators ---------------------------------------------------
    ranges = {}            # "group/chan" -> Accum
    dt_pool = {g: [] for g in VECTOR_GROUPS if g != "synchronized"}
    if have_depth:
        dt_pool.update({f"depth_{c}": [] for c in DEPTH_CAMS})
    rate_pool = {k: [] for k in dt_pool}
    gello_minus_ur = {f"q{i}": [] for i in range(1, 7)}
    gello_take_median = {f"q{i}": [] for i in range(1, 7)}
    corr_gello_ur = {f"q{i}": [] for i in range(1, 7)}
    corr_gello_cmd = {f"q{i}": [] for i in range(1, 7)}
    corr_dgello_dcmd = {f"q{i}": [] for i in range(1, 7)}
    cmd_track_err = {f"q{i}": Accum() for i in range(1, 7)}
    cmd_track_err_take_median = {f"q{i}": [] for i in range(1, 7)}
    lag_rows = []
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
    sim_takes = set()
    columns_attr_types = set()
    group_sets = set()
    depth_max_mm = 0
    depth_over_10m_px = 0
    depth_frames_sampled = 0

    for ti, take in enumerate(takes, 1):
        tdir = os.path.join(data_dir, take)
        print(f"[{ti:2d}/{len(takes)}] {take}", file=sys.stderr)

        # cleanliness -------------------------------------------------------
        expected = {"vectors.h5", "cam1.mp4", "cam2.mp4"}
        if have_depth:
            expected.add("depth.h5")
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
            # simulated takes carry a `sim_meta` file attr; the real recorder writes none
            if "sim_meta" in f.attrs:
                try:
                    sim_takes.add(take) if json.loads(
                        f.attrs["sim_meta"]).get("simulated") else None
                except Exception:
                    pass
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

            # recorder timestamp artefact -------------------------------------
            # How late each starved robot table is stamped, measured two ways.
            lag_rec = {"take": take}
            urq = np.stack([f["ur_joint_states"][f"q{i}"][:] for i in range(1, 7)], axis=1)
            cmq = np.stack([f["command"][f"cmd{i}"][:] for i in range(1, 7)], axis=1)
            if ut.size > 2 and ct.size > 2:
                found = joint_lag(ut, urq, ct, cmq)
                zero = joint_residual_at(0.0, ut, urq, ct, cmq)
                if found is not None:
                    tau, (med_res, mean_res, n_res), tau_mean = found
                    lag_rec["ur_joint_states_lag_s"] = _r(tau, 3)
                    lag_rec["ur_joint_states_lag_s_mean_objective"] = _r(tau_mean, 3)
                    lag_rec["residual_rad_at_lag"] = _r(med_res, 4)
                    lag_rec["residual_rad_at_lag_mean"] = _r(mean_res, 4)
                    lag_rec["n_compared"] = n_res
                if zero is not None:
                    lag_rec["residual_rad_at_zero"] = _r(zero[0], 4)
                    lag_rec["residual_rad_at_zero_mean"] = _r(zero[1], 4)
            # tcp_pose has its own (smaller) lag: same starvation, half the queue
            # depth. Measured by speed cross-correlation against `command`, which
            # is the freshest robot table.
            tcp_xyz = np.stack([f["tcp_pose"][k][:] for k in ("x", "y", "z")], axis=1)
            gc_, vc_ = speed_series(ct, cmq)
            gu_, vu_ = speed_series(ut, urq)
            gp_, vp_ = speed_series(f["tcp_pose"]["t_rel_s"][:], tcp_xyz)
            for key, (ga, va, gb, vb) in {
                "xcorr_command_to_ur_joint_states": (gc_, vc_, gu_, vu_),
                "xcorr_command_to_tcp_pose": (gc_, vc_, gp_, vp_),
                "xcorr_tcp_pose_to_ur_joint_states": (gp_, vp_, gu_, vu_),
            }.items():
                lag_s, r = xcorr_lag(ga, va, gb, vb)
                lag_rec[key + "_s"] = _r(lag_s, 3)
                lag_rec[key + "_r"] = _r(r, 3)
            lag_rows.append(lag_rec)

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
                seg = gpos[e:end + 1]
                closures.append({
                    "t_close_s": _r(gtt[e], 2),
                    "t_open_s": _r(gtt[end], 2) if after.size else None,
                    # seg can be empty if the rising edge is the last sample
                    "grip_pos_peak": _r(seg.max(), 4) if seg.size else None,
                })
            rec["n_grip_closures"] = len(closures)
            rec["grip_closures"] = closures
            rec["grip_pos_max"] = _r(gpos.max(), 4) if gpos.size else None
            if len(closures) > 1:
                regrasp.append({"take": take, "closures": closures})

        # depth -------------------------------------------------------------
        if have_depth:
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
        if have_depth:
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
        if have_depth:
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

    # ------------------------------------------- recorder timestamp artefact
    def lag_col(key):
        return np.array([r[key] for r in lag_rows if r.get(key) is not None],
                        dtype=np.float64)

    def lag_stats(key, nd=3):
        v = lag_col(key)
        if not v.size:
            return None
        return {"min": _r(v.min(), nd), "median": _r(np.median(v), nd),
                "max": _r(v.max(), nd), "n_takes": int(v.size)}

    tau_col = lag_col("ur_joint_states_lag_s")
    timestamp_lag = {
        "note": (
            "RECORDER ARTEFACT, not robot behaviour: `ur_joint_states`, `tcp_pose` and "
            "`wrench` rows are stamped LATE in this release. The GUI recorder writes "
            "every row inside its subscription callback and sets t_rel_s to the time "
            "that callback RAN; ROS header stamps are not stored. All callbacks share "
            "one rclpy spin thread, which a SingleThreadedExecutor services one message "
            "per subscription per round. Adding depth recording to this session dropped "
            "the round rate to ~60-69 Hz, so every publisher faster than that keeps its "
            "KEEP_LAST queue permanently full and each row leaves the queue already "
            "`queue_depth / publish_rate` seconds old: /joint_states depth 100 at ~100 Hz "
            "-> ~1.0 s, tcp_pose and wrench depth 50 at ~100 Hz -> ~0.5 s. The starvation "
            "signature is visible in `rates`: command / ur_joint_states / tcp_pose / "
            "wrench, four topics with three different publish rates, all converge on the "
            "same recorded rate. Streams at or below the round rate never queue and are "
            "fresh: cameras (depth header-stamp age measured at 0.022 s), depth, "
            "gello_joint_states, gripper, and `command`. The lag is a PURE DELAY - the "
            "waveform is intact - so subtracting the constant below from t_rel_s restores "
            "the timebase. NOTHING IS REWRITTEN HERE: the raw files are as recorded. "
            "FIX RULE for consumers: subtract ur_joint_states_lag_s from "
            "ur_joint_states/t_rel_s and ~0.45 s from tcp_pose/t_rel_s and "
            "wrench/t_rel_s; do NOT shift anything else - the gripper and camera tables "
            "are already on the camera timebase and shifting them would create a "
            "misalignment that is not there. The recorder has since been fixed (header "
            "stamps stored, queue depths shrunk, camera/depth writes moved off the spin "
            "thread, starvation warning) - this release predates that fix."
        ),
        "method": (
            "ur_joint_states_lag_s: joint-space residual. For each candidate lag on a "
            f"{int(LAG_TAU_STEP_S * 1000)} ms grid over 0..{LAG_TAU_MAX_S} s the "
            "ur_joint_states clock is shifted EARLIER by that lag, the six joints are "
            "linearly interpolated onto the `command` timestamps, and the lag minimising "
            "the median over time of the 6-joint mean |ur_q - cmd| is reported. "
            "residual_rad_at_lag / residual_rad_at_zero give that residual with and "
            "without the correction; the *_mean variants are the plain mean over all "
            "samples and joints, which is larger because it is dominated by the "
            "high-velocity transients where the genuine servo following error lives. "
            "ur_joint_states_lag_s_mean_objective is the argmin of that plain mean "
            "instead of the median: it is the SAME measurement and agrees to within one "
            "5 ms grid step, and it is reported because independent re-measurements of "
            "this dataset use it and land on 0.895 s for a minority of takes. Quote the "
            "lag as 0.89-0.91 s depending on method and grid; the derived LeRobot "
            "release applies the single constant 0.900 s. "
            "xcorr_*: independent check by speed-profile cross-correlation on the same "
            "grid (speed, not position, so a constant spatial offset such as the 0.174 m "
            "tool offset cannot bias it); a positive value means the second stream is "
            "stamped that many seconds later than the first."
        ),
        "ur_joint_states_lag_s": lag_stats("ur_joint_states_lag_s"),
        "ur_joint_states_lag_s_mean_objective": lag_stats(
            "ur_joint_states_lag_s_mean_objective"),
        "takes_at_modal_lag": (
            int((tau_col == np.median(tau_col)).sum()) if tau_col.size else None),
        "modal_lag_s": _r(np.median(tau_col), 3) if tau_col.size else None,
        "residual_rad_at_lag": lag_stats("residual_rad_at_lag", 4),
        "residual_rad_at_lag_mean": lag_stats("residual_rad_at_lag_mean", 4),
        "residual_rad_at_zero": lag_stats("residual_rad_at_zero", 4),
        "residual_rad_at_zero_mean": lag_stats("residual_rad_at_zero_mean", 4),
        "xcorr_command_to_ur_joint_states_s": lag_stats("xcorr_command_to_ur_joint_states_s"),
        "xcorr_command_to_tcp_pose_s": lag_stats("xcorr_command_to_tcp_pose_s"),
        "xcorr_tcp_pose_to_ur_joint_states_s": lag_stats("xcorr_tcp_pose_to_ur_joint_states_s"),
        "fresh_streams": [
            "cam1_frames", "cam2_frames", "depth.h5 cam1", "depth.h5 cam2",
            "gello_joint_states", "gripper", "command",
        ],
        "late_streams": ["ur_joint_states", "tcp_pose", "wrench"],
        "wrench_note": (
            "`wrench` shares tcp_pose's QoS depth (50) and publisher, so it carries the "
            "same ~0.5 s lag. It is not cross-correlated here because the force signal's "
            "derivative is too noisy to time reliably; the sibling velocity-correlation "
            "analysis measured tcp_pose -> wrench at 0.000 s over 7 of 8 takes, which is "
            "the sharpest confirmation of the queue-depth model (equal depth -> equal "
            "lag), and tcp_pose -> ur_joint_states at +0.495 s (unequal depth)."
        ),
        "per_take": lag_rows,
    }

    if sim_takes:
        # The prose above describes the REAL rig's rclpy spin-thread starvation. A
        # simulated take is stamped by the physics thread and `command` and
        # `ur_joint_states` come out of the SAME tick, so nothing is stamped late:
        # whatever lag the estimator finds here is the simulated arm's own tracking
        # lag behind its commanded target, and must NOT be subtracted.
        timestamp_lag["simulated_takes"] = sorted(sim_takes)
        timestamp_lag["applies_to_this_release"] = False
        timestamp_lag["simulation_note"] = (
            "SIMULATED take family (vectors.h5 carries sim_meta.simulated = true): the "
            "`note`/`method` prose above is inherited from the real-robot release and its "
            "MECHANISM does not apply here. The sim recorder stamps every robot row with "
            "the physics tick that produced it, and `command` / `ur_joint_states` / "
            "`tcp_pose` / `wrench` all come from that same tick, so no table is stamped "
            "late and NOTHING should be shifted. The non-zero ur_joint_states_lag_s "
            "reported above is the simulated arm's mechanical tracking lag behind its "
            "commanded joint target - real robot behaviour that is present in the data, "
            "not a clock error."
        )

    durs = np.array([t["duration_s"] for t in per_take], dtype=np.float64)
    total_bytes = sum(
        t["bytes_cam1_mp4"] + t["bytes_cam2_mp4"] + t["bytes_vectors_h5"]
        + t.get("bytes_depth_h5", 0)
        for t in per_take
    )

    def col(key):
        return np.array([t[key] for t in per_take if t.get(key) is not None], dtype=np.float64)

    def col_stats(key, nd):
        """min/median/max of a per-take column; None-valued when nothing was measured
        (e.g. a take family with no grasp, or a column that this mode never fills)."""
        a = col(key)
        if not a.size:
            return {"min": None, "median": None, "max": None}
        return {"min": _r(a.min(), nd), "median": _r(np.median(a), nd), "max": _r(a.max(), nd)}

    aggregate = {
        "n_takes": len(per_take),
        "total_duration_s": _r(durs.sum(), 2),
        "total_cam1_frames": int(sum(t["rows_cam1_frames"] for t in per_take)),
        "total_cam2_frames": int(sum(t["rows_cam2_frames"] for t in per_take)),
        "total_bytes": int(total_bytes),
        "total_bytes_cam1_mp4": int(sum(t["bytes_cam1_mp4"] for t in per_take)),
        "total_bytes_cam2_mp4": int(sum(t["bytes_cam2_mp4"] for t in per_take)),
        "total_bytes_vectors_h5": int(sum(t["bytes_vectors_h5"] for t in per_take)),
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
        "columns_attr_python_types": sorted(columns_attr_types),
        "identical_group_set_across_takes": len(group_sets) == 1,
        "groups": sorted(group_sets)[0] if len(group_sets) == 1 else None,
        "takes_with_multiple_grip_closures": [r["take"] for r in regrasp],
    }

    if have_depth:
        aggregate.update({
            "total_depth_cam1_frames": int(sum(t["rows_depth_cam1"] for t in per_take)),
            "total_depth_cam2_frames": int(sum(t["rows_depth_cam2"] for t in per_take)),
            "total_bytes_depth_h5": int(sum(t["bytes_depth_h5"] for t in per_take)),
            "takes_with_depth_cam1_colour_diff": depth_cam_diff["cam1"],
            "takes_with_depth_cam2_colour_diff": depth_cam_diff["cam2"],
            "depth_valid_pct_cam1": col_stats("depth_valid_pct_cam1", 2),
            "depth_valid_pct_cam2": col_stats("depth_valid_pct_cam2", 2),
            "depth_median_range_m_cam1": col_stats("depth_median_range_m_cam1", 4),
            "depth_median_range_m_cam2": col_stats("depth_median_range_m_cam2", 4),
            "depth_camera_info_identical_across_takes": {
                c: not cam_meta_mismatch[c] for c in DEPTH_CAMS},
            "depth_camera_info_mismatch_takes": cam_meta_mismatch,
            "depth_frames_sampled": depth_frames_sampled,
            "depth_max_mm_sampled": depth_max_mm,
            "depth_pixels_over_10000mm_sampled": depth_over_10m_px,
            "depth_valid_pct_at_grasp_cam1": col_stats("depth_valid_pct_at_grasp_cam1", 2),
            "depth_valid_pct_at_grasp_cam2": col_stats("depth_valid_pct_at_grasp_cam2", 2),
        })
    else:
        aggregate["has_depth"] = False
    aggregate["takes_with_no_grip_closure"] = [
        t["take"] for t in per_take if not t.get("n_grip_closures")]

    doc = {
        "dataset": args.dataset,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": (
            f"make_carrot_raw_stats.py — h5py + cv2 + ffprobe audit over all {len(per_take)} "
            + ("take folders (vectors.h5, depth.h5, cam1.mp4, cam2.mp4). mean rate = "
               if have_depth else
               "take folders (vectors.h5, cam1.mp4, cam2.mp4 — depth-free mode, every "
               "depth section skipped). mean rate = ") +
            "(N-1)/(t_last-t_first) per take, then median across takes; dt percentiles pooled "
            f"over all takes. Video frame counts from `{frame_method}`. Value ranges pooled "
            "over all takes." + (" Depth quality measured on ONE sampled mid frame per "
            "camera per take (decoded with cv2.imdecode(..., IMREAD_UNCHANGED))."
            if have_depth else "")
        ),
        "aggregate": aggregate,
        "rates": rates,
        "leader_vs_follower": leader_vs_follower,
        "command_tracking": command_tracking,
        "timestamp_lag": timestamp_lag,
        "regrasp_takes": regrasp,
        "value_ranges": {k: ranges[k].out() for k in sorted(ranges)},
        "per_take": per_take,
    }

    if have_depth:
        doc["depth_cameras"] = cam_meta

    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
