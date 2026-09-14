#!/usr/bin/env python3
"""
Convert the "Put carrot in pot" gello/UR teleop takes (vectors.h5 + cam1.mp4 +
cam2.mp4) into a LeRobot v3.0 dataset with a **Cartesian (EEF) delta action
space**.  NO DEPTH.

Contract: `ros2_ur_ws/src/gello_policy/CARROT_EEF_DESIGN.md` (row A1, section 2).
The policy space itself lives in `gello_policy/eef_space.py` and is IMPORTED --
never re-implemented here, so "what the policy was trained on" and "what the
robot executes at deploy time" are literally the same arithmetic.

Recipe
------
  - fps = 30 (camera rate). Master clock = cam1_frames/t_rel_s.  Every other
    stream is resampled onto it by nearest-timestamp lookup on its own t_rel_s.
    This is bit-for-bit the RGB/master-clock/resampling recipe of the joint-space
    converter `convert_carrot_to_lerobot.py`; only the state/action vectors and
    the absence of depth differ.
  - **Joint-stream time alignment (integrator spec change, 2026-09-14).**  The
    recorder's `ur_joint_states` cache is STALE by a constant ~0.90 s in every
    carrot take: the sample timestamped t carries the arm state of t - 0.90.
    `command`, `cam1`, `cam2`, `gello_*` and `grip_*` are fresh.  So the joints
    are read at `t_master + tau_q`, with `tau_q` estimated PER TAKE as
        argmin over tau in [0, 1.5] step 0.01 s  of
            median_t |p_tcp(q @ nearest(ur_t, t + tau)) - p_tcp(command_t)|
    scored on a tau-independent window of command samples.  The take is refused
    unless 0.6 <= tau_q <= 1.2 and the residual median is < 3 mm.  Measured on
    all 54 takes: tau_q = 0.90 s everywhere, residual 0.40-0.87 mm (the same
    quantity at tau = 0 is 27-70 mm).  tau_q and its residual are written per
    episode into meta/source_takes.json.
    The joints are then read at those shifted times by LINEAR INTERPOLATION, not
    nearest-neighbour: the joint cache ticks at a jittery ~63-67 Hz, so snapping
    a 30 Hz clock onto it repeats a sample on ~8.8 % of frames (fake zero
    motion) and skips three on ~24 % (fake double step) -- artefacts that land
    directly in the action, which is a finite difference of this very series.
    CONSEQUENCE, deliberately left visible: the shifted query runs off the end
    of the joint stream for the last ~27 frames of every take (7.9 % of all
    frames), which clamp to the final joint sample -> a frozen tail with zero
    deltas.  `--drop-stale-tail` removes those frames instead; it is OFF by
    default because the frame-count-per-episode contract says "== cam1 frames".
  - observation.state (16) = eef_space.state_vector(q_t, grip_pos_t)
        [q1..q6, tcp_x, tcp_y, tcp_z, r11, r21, r31, r12, r22, r32, grip_pos]
        q = ur_joint_states linearly interpolated at (master + tau_q) [see above]
        grip_pos @ nearest gripper t_rel_s (unshifted -- that stream is fresh).
        TCP = ur_kin.fk(q) translated +0.174 m along the flange +Z.
        Rotation is the 6-D representation (first two COLUMNS of R_tcp) because
        the carrot TCP sits ~175 deg from identity, where a rotation vector
        would wrap.
  - action (7) = eef_space.delta_action(q_t, q_{t+1}, grip_cmd_t)
        [dx, dy, dz, drx, dry, drz, grip_cmd]
        dp   = p_tcp(q_{t+1}) - p_tcp(q_t)                (base frame)
        drot = so3_log(R_{t+1} R_t^T)                     (base frame, left-mult)
        grip_cmd = ABSOLUTE recorded gripper command (0=open..1=closed), the same
        ffill/bfill + nearest resampling as the joint converter.
        **The LAST frame of every episode has dp = drot = 0** (there is no
        q_{t+1}); its grip_cmd is kept.
    WHY the achieved motion and not `command`: in this EEF-teleop session the
    recorded `command` stream LEADS the measured joints by ~0.9 s (61 mm median
    at tau=0, 1 mm at tau=+0.9 s, all 54 takes), so `command_t` is NOT "where the
    arm goes next" -- it is where it will be almost a second later.  DESIGN sec.1
    therefore defines the action from the ACHIEVED motion.  (Note: the docstring
    of `eef_space.delta_action` still describes its arguments as (q, command);
    the function is a pure two-pose difference, and DESIGN sec.2 pins the dataset
    call to `delta_action(q_t, q_{t+1}, grip_cmd_t)`.)
  - observation.images.cam1 / cam2: 720p RGB video (HWC uint8).  cam1 is the
    master (mp4 frame k == master frame k, decoded sequentially, last frame
    padded if the mp4 runs short); cam2 is decoded whole and indexed by
    nearest_idx(cam2_t, cam1_t).
  - gello_* / wrench / tcp_pose / command streams are intentionally ignored
    (not observable at inference, or -- for `command` -- lagged, see above).

Side files written after ds.finalize() (so lerobot's own writer cannot clobber
meta/): meta/source_takes.json (episode_index -> take dir, same format as the
joint converter) and meta/eef_space.json (the policy-space fingerprint,
including the sha256 of eef_space.py so a silently edited policy space is
detectable from the dataset alone).

Python 3.12, interpreter /home/laptop3/youngwoong_ws/lr_env/bin/python, with
PYTHONPATH=<worktree>/ros2_ur_ws/src/ur_gello_bringup:<worktree>/ros2_ur_ws/src/gello_policy
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import os
import sys
import time
from typing import Any, Iterable, Sequence

import cv2
import h5py
import numpy as np

from gello_policy import eef_space

TASK = "Put carrot in pot"
FPS = 30
ROBOT_TYPE = "ur7e_gello"
DEFAULT_REPO_ID = "Bigenlight/carrot_in_pot_eef_lerobot_v3"

STATE_NAMES = list(eef_space.STATE_NAMES)    # 16
ACTION_NAMES = list(eef_space.ACTION_NAMES)  # 7
STATE_DIM = eef_space.STATE_DIM
ACTION_DIM = eef_space.ACTION_DIM

CAMS = ("cam1", "cam2")
RGB_SHAPE = (720, 1280, 3)

# --- stale-joint-cache alignment (integrator spec change 2026-09-14) --------
TAU_MIN_S = 0.0
TAU_MAX_S = 1.5
TAU_STEP_S = 0.01
TAU_ACCEPT_LO_S = 0.6      # refuse the take outside this band ...
TAU_ACCEPT_HI_S = 1.2
TAU_RESIDUAL_MAX_M = 3e-3  # ... or if the aligned residual is worse than this
TAU_MIN_SAMPLES = 30       # refuse if the scoring window is this thin
JOINT_STEP_MAX_RAD = 0.2   # continuity guard before interpolating (measured max 0.0205)

JOINT_ALIGNMENT_NOTE = (
    "ur_joint_states shifted by per-take tau_q (see source_takes.json); "
    "recorder cache was stale ~0.9 s"
)
JOINT_RESAMPLING = "linear_interp"

EEF_SPACE_DEFINITION = (
    "observation.state (16) = [q1..q6 (rad, UR order, as observed -- see the "
    "joint_state_alignment note: the recorder's joint cache is stale by a "
    "per-take tau_q of ~0.9 s, so the joints are read at master_time + tau_q, "
    "by LINEAR INTERPOLATION of the joint record because that record ticks at a "
    "jittery ~63-67 Hz and nearest-neighbour would inject fake zero and fake "
    "double steps into the finite-difference action), "
    "tcp_x, tcp_y, tcp_z (m, base_link frame), "
    "r11, r21, r31, r12, r22, r32 (the first two COLUMNS of the TCP rotation "
    "matrix -- the continuous 6-D rotation representation, used because the "
    "carrot TCP sits ~175 deg from identity where a rotation vector would wrap), "
    "grip_pos (0=open..1=closed)]. TCP = ur_kin.fk(q) (base_link -> tool0 flange) "
    f"translated by +{eef_space.TOOL_Z_M} m along the flange +Z. "
    "action (7) = [dx, dy, dz, drx, dry, drz, grip_cmd]: for dataset frame t on "
    "the 30 fps master clock, dp = p_tcp(q_{t+1}) - p_tcp(q_t) and "
    "drot = so3_log(R_tcp(q_{t+1}) @ R_tcp(q_t)^T), both expressed in the base "
    "frame with the rotation LEFT-multiplied; the last frame of every episode has "
    "dp = drot = 0. grip_cmd is the ABSOLUTE recorded gripper command "
    "(0=open..1=closed), ffill/bfill'd then resampled onto the master clock. "
    "The action is the ACHIEVED motion between consecutive frames, never the "
    "recorded `command` stream (which leads the measured joints by ~0.9 s in this "
    "session). Deploy inverts it exactly: p_target = p_live + dp, "
    "R_target = so3_exp(drot) @ R_live, then analytic IK with branch locking "
    "(eef_space.apply_delta / ik_tcp_nearest). Nothing here is scaled, clipped or "
    "filtered -- clamps belong to the deploy node only."
)


# --------------------------------------------------------------------------- #
# resampling helpers (identical semantics to convert_carrot_to_lerobot.py)
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


def interp_joints(
    ur_t: np.ndarray, ur_q: np.ndarray, query_t: np.ndarray, take_name: str
) -> np.ndarray:
    """Sample the joint record at arbitrary times by LINEAR INTERPOLATION.

    Nearest-neighbour is wrong here and measurably so: the joint cache ticks at
    a jittery ~63-67 Hz (dt 3.6-43.3 ms measured), so snapping a 30 Hz master
    clock to it makes ~8.8 % of frames repeat the previous sample (a fake zero
    motion) and ~24 % jump three samples (a fake double-speed step, 1-frame
    |dp| up to 11.5 mm vs 8.9 mm interpolated).  Those artefacts land straight
    in the action, which is a finite difference of exactly this series.

    Queries outside the record clamp to its first/last sample (numpy's default),
    which is what produces the frozen tail documented in the module docstring.

    Interpolating joint angles is only legitimate while the series is
    continuous, so this refuses a take whose consecutive joint samples ever jump
    more than JOINT_STEP_MAX_RAD (a +-2*pi wrap would look exactly like that).
    """
    ur_q = np.asarray(ur_q, dtype=float)
    step = float(np.max(np.abs(np.diff(ur_q, axis=0)))) if len(ur_q) > 1 else 0.0
    if step >= JOINT_STEP_MAX_RAD:
        j = int(np.unravel_index(int(np.argmax(np.abs(np.diff(ur_q, axis=0)))),
                                 (len(ur_q) - 1, 6))[0])
        raise RuntimeError(
            f"{take_name}: consecutive ur_joint_states samples jump {step:.4f} rad "
            f"(limit {JOINT_STEP_MAX_RAD}) around index {j} (t={ur_t[j]:.3f} s). "
            "Refusing to interpolate -- the joint series is not continuous there "
            "(a 2*pi wrap or a dropped block would look like this)."
        )
    out = np.empty((len(query_t), 6), dtype=float)
    for k in range(6):
        out[:, k] = np.interp(query_t, ur_t, ur_q[:, k])
    return out


def estimate_joint_shift(
    ur_t: np.ndarray, ur_q: np.ndarray, cmd_t: np.ndarray, cmd_q: np.ndarray, take_name: str
) -> tuple[float, float, int]:
    """Estimate how stale this take's ``ur_joint_states`` cache is, in seconds.

    tau_q = argmin over tau in [0, 1.5] step 0.01 of
              median_t |p_tcp(q @ nearest(ur_t, t + tau)) - p_tcp(command_t)|

    The command stream is fresh and the arm tracks it closely in joint mode, so
    ``p_tcp(command_t)`` is a proxy for where the arm ACTUALLY was at time t;
    the shift that best explains the recorded joints is the cache's staleness.

    The scoring window is the SAME set of command samples for every tau (those
    whose shifted query stays inside the joint timeline for the whole tau
    range), so the curve is not biased by end-clamping at large tau.

    Returns (tau_q, residual_median_m, n_samples_scored) and raises
    RuntimeError -- naming the take -- if the estimate falls outside the
    accepted band or the residual is worse than 3 mm.
    """
    taus = np.round(np.arange(TAU_MIN_S, TAU_MAX_S + 0.5 * TAU_STEP_S, TAU_STEP_S), 6)
    p_ur = np.stack([eef_space.tcp_pose(ur_q[i])[:3, 3] for i in range(len(ur_q))])
    p_cmd = np.stack([eef_space.tcp_pose(cmd_q[i])[:3, 3] for i in range(len(cmd_q))])

    keep = (cmd_t + TAU_MAX_S <= ur_t[-1]) & (cmd_t + TAU_MIN_S >= ur_t[0])
    if int(keep.sum()) < TAU_MIN_SAMPLES:
        raise RuntimeError(
            f"{take_name}: only {int(keep.sum())} command samples can score the whole "
            f"tau range [{TAU_MIN_S}, {TAU_MAX_S}] s -- take too short to align"
        )
    t_eval, p_eval = cmd_t[keep], p_cmd[keep]

    med = np.empty(len(taus))
    for i, tau in enumerate(taus):
        j = nearest_idx(ur_t, t_eval + tau)
        med[i] = float(np.median(np.linalg.norm(p_ur[j] - p_eval, axis=1)))
    k = int(np.argmin(med))
    tau_q, residual = float(taus[k]), float(med[k])

    if not (TAU_ACCEPT_LO_S <= tau_q <= TAU_ACCEPT_HI_S):
        raise RuntimeError(
            f"{take_name}: estimated joint-cache shift tau_q={tau_q:.2f} s is outside the "
            f"accepted band [{TAU_ACCEPT_LO_S}, {TAU_ACCEPT_HI_S}] s "
            f"(residual {residual * 1000:.2f} mm, {int(keep.sum())} samples). "
            "Refusing to convert -- the joint stream of this take is not the expected "
            "constant-lag cache."
        )
    if residual >= TAU_RESIDUAL_MAX_M:
        raise RuntimeError(
            f"{take_name}: best joint-cache alignment tau_q={tau_q:.2f} s still leaves a "
            f"median residual of {residual * 1000:.2f} mm (limit "
            f"{TAU_RESIDUAL_MAX_M * 1000:.1f} mm) over {int(keep.sum())} samples. "
            "Refusing to convert -- joints and command do not describe the same motion."
        )
    return tau_q, residual, int(keep.sum())


def load_take_arrays(
    h5_path: str, take_name: str, max_frames: int | None = None, drop_stale_tail: bool = False
) -> dict[str, Any]:
    """Re-derive one take onto the cam1 master clock.

    Everything is resampled onto the master clock first (float64), and only then
    pushed through :mod:`gello_policy.eef_space`; the truncation implied by
    ``--max-frames`` is applied to the master clock, so the last kept frame is
    the episode's last frame and gets the zero action.

    Returns a dict with cam1_t / cam2_t / state / action plus the alignment
    bookkeeping (tau_q, its residual, and how many trailing frames had to clamp
    to the final joint sample).
    """
    with h5py.File(h5_path, "r") as f:
        cam1_t = f["cam1_frames"]["t_rel_s"][:]
        cam2_t = f["cam2_frames"]["t_rel_s"][:]

        ur_t = f["ur_joint_states"]["t_rel_s"][:]
        ur_q_all = np.stack([f["ur_joint_states"][f"q{k + 1}"][:] for k in range(6)], axis=1)
        cmd_t = f["command"]["t_rel_s"][:]
        cmd_q_all = np.stack([f["command"][f"cmd{k + 1}"][:] for k in range(6)], axis=1)

        tau_q, residual, n_scored = estimate_joint_shift(
            ur_t, ur_q_all, cmd_t, cmd_q_all, take_name
        )

        # The joints are read 0.9 s LATER than the frame they describe, so the
        # last ~27 frames of every take have no joint record at all and clamp to
        # the final sample (frozen tail, zero deltas). Counted, and optionally
        # dropped with --drop-stale-tail.
        n_full = len(cam1_t)
        n_tail = int(np.sum(cam1_t + tau_q > ur_t[-1]))
        if drop_stale_tail and n_tail:
            cam1_t = cam1_t[: max(0, n_full - n_tail)]
        if max_frames:
            cam1_t = cam1_t[:max_frames]
        n = len(cam1_t)
        if n < 2:
            raise RuntimeError(f"{take_name}: only {n} frame(s) left after alignment/truncation")
        n_tail_kept = int(np.sum(cam1_t + tau_q > ur_t[-1]))

        q = interp_joints(ur_t, ur_q_all, cam1_t + tau_q, take_name)

        grip_t = f["gripper"]["t_rel_s"][:]
        grip_j = nearest_idx(grip_t, cam1_t)
        grip_pos = f["gripper"]["grip_pos"][:][grip_j]
        grip_cmd = ffill_bfill(f["gripper"]["grip_cmd"][:])[grip_j]

    state = np.zeros((n, STATE_DIM), dtype=np.float32)
    action = np.zeros((n, ACTION_DIM), dtype=np.float32)
    for k in range(n):
        state[k] = eef_space.state_vector(q[k], float(grip_pos[k]))
        if k + 1 < n:
            action[k] = eef_space.delta_action(q[k], q[k + 1], float(grip_cmd[k]))
    # last frame: no q_{t+1} exists -> zero motion, gripper command kept.
    if n:
        action[n - 1, :6] = 0.0
        action[n - 1, 6] = np.float32(grip_cmd[n - 1])

    return {
        "cam1_t": cam1_t,
        "cam2_t": cam2_t,
        "state": state,
        "action": action,
        "tau_q": tau_q,
        "residual_m": residual,
        "n_scored": n_scored,
        "n_tail_clamped": n_tail_kept,
        "n_tail_dropped": (n_tail if drop_stale_tail else 0),
    }


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
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ACTION_NAMES},
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": STATE_NAMES},
    }
    for cam in CAMS:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": RGB_SHAPE, "names": ["height", "width", "channels"],
        }
    return feats


def eef_space_fingerprint() -> dict[str, Any]:
    """meta/eef_space.json content: what policy space this dataset was built in."""
    src = inspect.getsourcefile(eef_space)
    if src is None:  # pragma: no cover -- only if eef_space were frozen/compiled
        raise RuntimeError("cannot locate the source file of gello_policy.eef_space")
    with open(src, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {
        "action_space_id": eef_space.ACTION_SPACE_ID,
        "tool_z_m": float(eef_space.TOOL_Z_M),
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "definition": EEF_SPACE_DEFINITION,
        "joint_state_alignment": JOINT_ALIGNMENT_NOTE,
        "joint_state_resampling": JOINT_RESAMPLING,
        "eef_space_sha256": sha,
        "eef_space_path": os.path.abspath(src),
        "fps": FPS,
        "task": TASK,
    }


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
    drop_stale_tail: bool = False,
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

    fingerprint = eef_space_fingerprint()
    print(f"[cfg] action space : {fingerprint['action_space_id']}  "
          f"state {STATE_DIM} / action {ACTION_DIM}", flush=True)
    print(f"[cfg] eef_space.py : {fingerprint['eef_space_path']}", flush=True)
    print(f"[cfg] sha256       : {fingerprint['eef_space_sha256']}", flush=True)

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

    source_takes: list[dict[str, Any]] = []
    total_frames = 0
    max_step_m = 0.0
    max_step_rad = 0.0
    taus: list[float] = []
    residuals: list[float] = []
    total_tail = 0
    t_start = time.time()

    for ti, tk in enumerate(takes):
        t0 = time.time()
        name = os.path.basename(tk)
        got = load_take_arrays(os.path.join(tk, "vectors.h5"), name, max_frames, drop_stale_tail)
        cam1_t, cam2_t = got["cam1_t"], got["cam2_t"]
        state, action = got["state"], got["action"]
        n = len(cam1_t)
        cam2_map = nearest_idx(cam2_t, cam1_t)

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
                "task": TASK,
            })
        ds.save_episode()

        dp = float(np.max(np.linalg.norm(action[:, :3], axis=1))) if n else 0.0
        dr = float(np.max(np.linalg.norm(action[:, 3:6], axis=1))) if n else 0.0
        max_step_m = max(max_step_m, dp)
        max_step_rad = max(max_step_rad, dr)

        total_frames += n
        taus.append(got["tau_q"])
        residuals.append(got["residual_m"])
        total_tail += int(got["n_tail_clamped"])
        source_takes.append({
            "episode_index": ti,
            "take_dir_name": name,
            "n_frames": n,
            "excluded": False,
            "joint_state_shift_s": got["tau_q"],
            "joint_state_shift_residual_m": got["residual_m"],
            "joint_state_shift_samples": got["n_scored"],
            "n_frames_clamped_to_last_joint_sample": int(got["n_tail_clamped"]),
            "n_frames_dropped_stale_tail": int(got["n_tail_dropped"]),
        })
        print(
            f"[{ti + 1:2d}/{len(takes)}] {name:32s} frames={n:4d}  cam2_decoded={len(cam2_all):4d}  "
            f"tau_q={got['tau_q']:.2f}s res={got['residual_m'] * 1000:.2f}mm "
            f"tail={got['n_tail_clamped']:3d}  "
            f"max|dp|={dp * 1000:5.1f}mm  max|drot|={dr:.4f}rad  {time.time() - t0:6.1f}s",
            flush=True,
        )

    ds.finalize()

    # --- sidecars AFTER finalize(), so lerobot's writer cannot clobber meta/ ---
    meta_dir = os.path.join(out_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "source_takes.json"), "w") as fh:
        json.dump(
            {
                "task": TASK,
                "data_root": os.path.abspath(data_root),
                "episodes": source_takes,
                "excluded": [{"take_dir_name": os.path.basename(t), "excluded": True}
                             for t in skipped],
            },
            fh,
            indent=2,
        )
    with open(os.path.join(meta_dir, "eef_space.json"), "w") as fh:
        json.dump(fingerprint, fh, indent=2)

    wall = time.time() - t_start
    size_gb = dir_size_bytes(out_root) / 1e9
    print(
        f"\nDONE: {len(takes)} episodes, {total_frames} frames, {wall / 60:.1f} min wall, "
        f"{size_gb:.2f} GB -> {out_root}"
    )
    print(f"  worst single-frame step: |dp| {max_step_m * 1000:.1f} mm, "
          f"|drot| {max_step_rad:.4f} rad")
    ta, rs = np.array(taus), np.array(residuals)
    print(f"  joint-cache shift tau_q: min {ta.min():.2f} / median {np.median(ta):.2f} / "
          f"max {ta.max():.2f} s   residual min {rs.min() * 1000:.2f} / "
          f"median {np.median(rs) * 1000:.2f} / max {rs.max() * 1000:.2f} mm")
    if total_tail:
        print(
            f"\n[!] WARNING: {total_tail} frame(s) ({100.0 * total_tail / max(1, total_frames):.2f} %"
            f" of the dataset) sit past the end of the joint stream once shifted by tau_q and "
            f"clamp to the final joint sample.\n"
            f"    Those trailing frames of each episode carry a FROZEN state and ZERO deltas "
            f"while the images keep moving.\n"
            f"    They are kept because the per-episode frame count is contracted to equal the "
            f"cam1 frame count; pass --drop-stale-tail to remove them instead.",
            file=sys.stderr,
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
    ap.add_argument("--drop-stale-tail", action="store_true",
                    help="drop the trailing frames whose tau_q-shifted joint query falls past "
                         "the end of ur_joint_states (~27 per take) instead of letting them "
                         "clamp to the final sample; CHANGES the per-episode frame count")
    args = ap.parse_args(list(argv) if argv is not None else None)
    convert(
        args.data, args.out, args.repo_id,
        limit=args.limit, max_frames=args.max_frames,
        image_writer_processes=args.procs, image_writer_threads=args.threads,
        exclude=tuple(args.exclude), drop_stale_tail=args.drop_stale_tail,
    )


if __name__ == "__main__":
    main()
