#!/usr/bin/env python3
"""Offline analysis library for ensemble HDF5 files written by ensemble_logger.py.

This is the *measurement instrument* for the research question:

    "Does the across-sample variance of a diffusion policy carry a usable
     uncertainty signal?"

Dependency-light on purpose: numpy + h5py + scipy only. NO torch, NO ROS, NO
lerobot. It never touches the real-time control path -- it only reads files that
diffusion_server.py's default-off ensemble side-channel (DIFFUSION_ENSEMBLE_K=0)
produced earlier.

DESIGN STANCE: the analysis must not be able to silently do the wrong thing.
Three things are therefore structural rather than optional:

  (1) Variance is returned as a PROFILE v(h) over the horizon index, never
      collapsed to a scalar by default. A diffusion policy's samples fan out
      monotonically with horizon distance simply because the far future is
      under-determined by the current observation -- that is *aleatoric horizon
      spread*, a property of the task, not of the current situation. A single
      scalar averaged over all `horizon` steps mixes that structural fan-out with
      the situational uncertainty we actually want to detect, and the structural
      term dominates. Keeping v(h) lets a caller normalize per-h against a
      baseline profile, which is the only way the situational signal survives.

  (2) The horizon is split into the EXECUTED slice and the DISCARDED tail, and
      they are reported separately. Only the executed slice can affect the robot.

  (3) Arm dims (0..5) and gripper dim (6) are ALWAYS returned separately and are
      never summed by default. The gripper trajectory is a near-binary open/close
      step surrounded by flat regions; a few-timestep timing disagreement between
      samples produces an enormous apparent variance that encodes *task phase*
      (am I near a grasp?) rather than model uncertainty. Summing dim 6 into an
      L2 over all 7 dims lets that one dim swamp the arm signal. Both are
      available, and `combined=True` exists for anyone who wants to check that
      claim empirically -- but it is never the default.

Second-moment caveat, stated once and repeated in the relevant docstrings:
per-timestep std, mean pairwise L2 and endpoint spread are ALL functions of the
second moment of the sample cloud. They are monotone in "how spread out is this
cloud" and are mathematically incapable of distinguishing

    two tight, well-separated modes   (a genuine decision point -- act on it)
from
    one wide unimodal blob            (a vague but unambiguous motion)

Only the PCA / cluster-structure probes in `multimodality_probe` address that.

--------------------------------------------------------------------------------
FILE LAYOUT (from ensemble_logger.py, mirrored here so this file is standalone)

    /ensemble_trajectories                         attrs: k, horizon, action_dim,
                                                   n_action_steps, n_obs_steps,
                                                   state_dim, t0_wall, [+ any
                                                   later provenance attrs]
    /ensemble_trajectories/meta/{t_rel_s,t_wall,refill_idx,ensemble_ms,dropped_flag}
        one row per refill EVENT, including dropped ones
    /ensemble_trajectories/trajectories    (N, K, horizon, action_dim) float32
    /ensemble_trajectories/committed_chunk (N, n_action_steps, action_dim) float32
    /ensemble_trajectories/obs_state       (N, n_obs_steps, state_dim) float32
        these three have rows ONLY for non-dropped refills, so their row index
        runs over SUCCESSFUL refills and is NOT the same index as meta's rows.
        `EnsembleFile.meta_rows_for_nd()` resolves the alignment.

NORMALIZATION: ensemble_sampler.run_batch documents that it returns the
trajectories in the NORMALIZED action space (what conditional_sample emits before
unnormalize). So variances computed here are in normalized units unless
`norm_stats` provenance attrs are present and the caller asks for physical units.
Older files predate any provenance attr; `norm_stats` is then None, not a crash.

--------------------------------------------------------------------------------
USAGE

    python3 ensemble_analysis.py /path/to/ensemble_20260720_120000.h5
    python3 ensemble_analysis.py --self-check
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:  # h5py is only needed for load(); the metric functions are pure numpy.
    import h5py
except ImportError:  # pragma: no cover - keeps `--self-check` usable without h5py
    h5py = None  # type: ignore[assignment]

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover
    _scipy_stats = None  # type: ignore[assignment]

# Hartigan & Hartigan's dip test is NOT part of scipy (there is no
# scipy.stats.diptest). The `diptest` PyPI package is used when importable;
# otherwise `dip_test` falls back to the bimodality coefficient (see its
# docstring). Strictly optional either way.
try:
    import diptest as _diptest_pkg
except ImportError:  # pragma: no cover
    _diptest_pkg = None  # type: ignore[assignment]


GROUP_NAME = "ensemble_trajectories"
META_COLUMNS_DEFAULT = ["t_rel_s", "t_wall", "refill_idx", "ensemble_ms", "dropped_flag"]

#: Default dimension split for the UR7e + 2F-85 action vector: 6 joint dims + 1
#: gripper dim. Overridable everywhere via the `arm_dims` / `grip_dims` kwargs so
#: this library is not hard-wired to a 7-DoF action space.
ARM_DIMS = (0, 1, 2, 3, 4, 5)
GRIP_DIMS = (6,)


# =============================================================================
# 1. Loading
# =============================================================================


@dataclass
class EnsembleFile:
    """Everything in one ensemble HDF5 file, eagerly loaded into numpy.

    Attributes
    ----------
    trajectories : (N, K, horizon, action_dim) float32
        The ensemble samples, one row per SUCCESSFUL refill. Normalized action
        space (see module docstring).
    committed_chunk : (N, n_action_steps, action_dim) float32
        The chunk the server actually sent to the robot for that refill.
    obs_state : (N, n_obs_steps, state_dim) float32
    meta : dict[str, (M,) float64]
        One entry per meta column, M >= N (dropped refills have meta rows but no
        trajectory rows).
    attrs : dict[str, Any]
        Every attribute on the group, JSON-decoded where possible.
    norm_stats : dict | None
        Provenance written by a later version of the logger. None on older files
        -- callers must handle None, and every function here already does.

    All the scalar attrs (k, horizon, ...) are None when a file predates them,
    but `__post_init__` back-fills them from the array shapes where possible so
    downstream code can rely on them.
    """

    path: str
    trajectories: np.ndarray
    committed_chunk: np.ndarray
    obs_state: np.ndarray
    meta: dict[str, np.ndarray]
    attrs: dict[str, Any] = field(default_factory=dict)
    norm_stats: dict[str, Any] | None = None

    k: int | None = None
    horizon: int | None = None
    action_dim: int | None = None
    n_action_steps: int | None = None
    n_obs_steps: int | None = None
    state_dim: int | None = None
    t0_wall: float | None = None

    def __post_init__(self) -> None:
        # Prefer the shapes on disk over the attrs: shapes cannot be stale.
        if self.trajectories.ndim == 4 and self.trajectories.size:
            n, k, h, d = self.trajectories.shape
            self.k = int(k)
            self.horizon = int(h)
            self.action_dim = int(d)
        if self.committed_chunk.ndim == 3 and self.committed_chunk.shape[0]:
            self.n_action_steps = int(self.committed_chunk.shape[1])
        if self.obs_state.ndim == 3 and self.obs_state.shape[0]:
            self.n_obs_steps = int(self.obs_state.shape[1])
            self.state_dim = int(self.obs_state.shape[2])
        # Fill anything the shapes could not supply (e.g. an empty file).
        for name in ("k", "horizon", "action_dim", "n_action_steps",
                     "n_obs_steps", "state_dim"):
            if getattr(self, name) is None and name in self.attrs:
                try:
                    setattr(self, name, int(self.attrs[name]))
                except (TypeError, ValueError):
                    pass
        if self.t0_wall is None and "t0_wall" in self.attrs:
            try:
                self.t0_wall = float(self.attrs["t0_wall"])
            except (TypeError, ValueError):
                pass

    # ---------------------------------------------------------------- helpers
    @property
    def n_refills(self) -> int:
        """Number of SUCCESSFUL refills (rows in `trajectories`)."""
        return int(self.trajectories.shape[0]) if self.trajectories.ndim == 4 else 0

    @property
    def n_events(self) -> int:
        """Number of refill EVENTS in meta, including dropped ones."""
        for v in self.meta.values():
            return int(v.shape[0])
        return 0

    @property
    def n_dropped(self) -> int:
        flag = self.meta.get("dropped_flag")
        return 0 if flag is None else int(np.count_nonzero(flag > 0.5))

    def meta_rows_for_nd(self) -> np.ndarray:
        """Indices into the meta columns corresponding to `trajectories` rows.

        The ND datasets skip dropped refills, so row i of `trajectories` is NOT
        meta row i. This returns an (N,) int index array such that
        ``meta['t_wall'][meta_rows_for_nd()]`` lines up with `trajectories`.
        Falls back to ``arange(N)`` if the dropped_flag column is missing.
        """
        flag = self.meta.get("dropped_flag")
        if flag is None:
            return np.arange(self.n_refills, dtype=int)
        keep = np.flatnonzero(flag <= 0.5)
        return keep[: self.n_refills]

    def meta_column(self, name: str, aligned_to_nd: bool = False) -> np.ndarray | None:
        """One meta column, optionally re-indexed to line up with `trajectories`."""
        col = self.meta.get(name)
        if col is None:
            return None
        return col[self.meta_rows_for_nd()] if aligned_to_nd else col

    def exec_slice(self) -> slice:
        """The horizon slice that is actually executed. See `executed_tail_split`."""
        return executed_slice(self.n_obs_steps, self.n_action_steps, self.horizon)

    def describe(self) -> str:
        ns = "present" if self.norm_stats else "None (pre-provenance file)"
        return (
            f"EnsembleFile({os.path.basename(self.path)})\n"
            f"  successful refills : {self.n_refills}\n"
            f"  refill events      : {self.n_events} ({self.n_dropped} dropped)\n"
            f"  K / horizon / dim  : {self.k} / {self.horizon} / {self.action_dim}\n"
            f"  n_obs / n_action   : {self.n_obs_steps} / {self.n_action_steps}\n"
            f"  executed slice     : {self.exec_slice()}\n"
            f"  norm_stats         : {ns}\n"
            f"  action space       : NORMALIZED (per ensemble_sampler.run_batch)"
        )


def _decode_attr(value: Any) -> Any:
    """h5py attrs -> plain python. Bytes are decoded, JSON strings are parsed."""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        s = value.strip()
        if s.startswith(("{", "[")):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return value
    return value


def _extract_norm_stats(grp: Any) -> dict[str, Any] | None:
    """Best-effort recovery of normalization provenance.

    Files written before the provenance change simply do not have it. Every
    lookup below is guarded; the function returns None rather than raising, and
    every consumer in this module treats None as "unknown, stay in normalized
    units". Accepts three plausible shapes so a later logger version can pick
    any of them without breaking this reader:
      * a JSON-string group attr named norm_stats / normalization_stats / dataset_stats
      * a subgroup named norm_stats whose datasets are the stats
      * flat attrs named norm_stats.<key>
    """
    if grp is None:
        return None
    def _read_stats_node(node) -> dict[str, Any]:
        """Recursively read an h5 stats subtree into nested dicts of arrays.

        ensemble_logger.py writes norm_stats as NESTED subgroups, one per feature
        key (e.g. norm_stats/observation.state/mean). A flat one-level read would
        hand np.asarray() an h5py.Group, which does NOT raise -- it silently
        returns the array of member NAMES. That defeats the whole point of
        embedding the stats, so recurse on groups and only convert leaf datasets.
        """
        out: dict[str, Any] = {}
        for name in node.keys():
            child = node[name]
            if hasattr(child, "keys"):  # Group -> recurse
                nested = _read_stats_node(child)
                if nested:
                    out[name] = nested
            else:  # Dataset -> leaf value
                try:
                    out[name] = np.asarray(child)
                except Exception:  # noqa: BLE001 - provenance is never load-critical
                    continue
        for name in getattr(node, "attrs", {}):
            out.setdefault(name, _decode_attr(node.attrs[name]))
        return out

    for key in ("norm_stats", "normalization_stats", "dataset_stats", "normalization"):
        if key in grp.attrs:
            decoded = _decode_attr(grp.attrs[key])
            if isinstance(decoded, dict):
                return decoded
            return {key: decoded}
    for key in ("norm_stats", "normalization_stats"):
        sub = grp.get(key) if hasattr(grp, "get") else None
        if sub is not None and hasattr(sub, "keys"):
            out = _read_stats_node(sub)
            return out or None
    flat = {
        str(a).split(".", 1)[1]: _decode_attr(grp.attrs[a])
        for a in grp.attrs
        if str(a).startswith("norm_stats.")
    }
    return flat or None


def load(path: str, group: str = GROUP_NAME) -> EnsembleFile:
    """Load an ensemble HDF5 file into memory.

    Tolerant by construction: missing datasets become correctly-shaped empty
    arrays and missing attrs become None. In particular a file written before
    the norm_stats provenance change loads fine with ``norm_stats is None``.
    """
    if h5py is None:
        raise RuntimeError("h5py is required for load(); metric functions are pure numpy")
    with h5py.File(path, "r") as h5:
        grp = h5.get(group)
        if grp is None:
            raise KeyError(f"{path}: no group {group!r} (groups: {list(h5.keys())})")

        attrs = {str(a): _decode_attr(grp.attrs[a]) for a in grp.attrs}

        def _nd(name: str, ndim: int) -> np.ndarray:
            dset = grp.get(name)
            if dset is None:
                return np.zeros((0,) * ndim, dtype=np.float32)
            return np.asarray(dset, dtype=np.float32)

        trajectories = _nd("trajectories", 4)
        committed_chunk = _nd("committed_chunk", 3)
        obs_state = _nd("obs_state", 3)

        meta: dict[str, np.ndarray] = {}
        meta_grp = grp.get("meta")
        if meta_grp is not None:
            cols = META_COLUMNS_DEFAULT
            raw_cols = _decode_attr(meta_grp.attrs.get("columns", ""))
            if isinstance(raw_cols, list) and raw_cols:
                cols = [str(c) for c in raw_cols]
            for col in cols:
                dset = meta_grp.get(col)
                if dset is not None:
                    meta[col] = np.asarray(dset, dtype=np.float64)
            # Pick up any column present on disk but absent from the attr.
            for col in meta_grp.keys():
                if col not in meta:
                    meta[str(col)] = np.asarray(meta_grp[col], dtype=np.float64)

        return EnsembleFile(
            path=path,
            trajectories=trajectories,
            committed_chunk=committed_chunk,
            obs_state=obs_state,
            meta=meta,
            attrs=attrs,
            norm_stats=_extract_norm_stats(grp),
        )


# =============================================================================
# 3. Executed vs discarded split
# =============================================================================
#
# VERIFIED AGAINST THE INSTALLED SOURCE, not taken on trust:
#   /home/theo_lab/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py
#   (checkout at commit 24017e96), DiffusionModel.generate_actions, lines 315-318:
#
#       315:  # Extract `n_action_steps` steps worth of actions (from the current observation).
#       316:  start = n_obs_steps - 1
#       317:  end = start + self.config.n_action_steps
#       318:  actions = actions[:, start:end]
#
# generate_actions itself is defined at line 295 and the assert on n_obs_steps is
# at line 307. The docstring of DiffusionPolicy.select_action (lines 117-131 of
# the same file) states the same layout in words and notes the requirement
# `n_action_steps <= horizon - n_obs_steps + 1`.
#
# Consequence for this analysis: the ensemble logger stores the FULL horizon
# (ensemble_sampler.run_batch returns (K, horizon, action_dim)), so indices
# [0, n_obs_steps-1) are pre-observation context that generate_actions throws
# away, [start, end) is what the robot actually executes, and [end, horizon) is a
# never-executed tail. Variance in the tail is real model spread but it cannot
# affect the robot, so it must not be pooled with the executed part.


def executed_slice(
    n_obs_steps: int | None,
    n_action_steps: int | None,
    horizon: int | None = None,
) -> slice:
    """The `[start:end)` horizon slice lerobot actually executes.

    Mirrors modeling_diffusion.py:316-318 exactly (see the block comment above).
    Returns ``slice(None)`` if the metadata needed to compute it is missing,
    which is the honest answer for a file that does not record it -- callers can
    detect that with ``sl == slice(None)``.
    """
    if n_obs_steps is None or n_action_steps is None:
        return slice(None)
    start = int(n_obs_steps) - 1
    end = start + int(n_action_steps)
    if horizon is not None:
        end = min(end, int(horizon))
        start = min(start, end)
    return slice(start, end)


def executed_tail_split(
    traj: np.ndarray,
    n_obs_steps: int | None,
    n_action_steps: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a trajectory array on its horizon axis into (executed, tail).

    Parameters
    ----------
    traj : (..., horizon, action_dim)
        Works on a single refill (K, horizon, D) or a stack (N, K, horizon, D).

    Returns
    -------
    executed : (..., n_action_steps, action_dim)
        The slice generate_actions returns, i.e. the only part that ever reaches
        the robot.
    tail : (..., horizon - end, action_dim)
        The never-executed remainder [end:horizon). The discarded *prefix*
        [0:start) is dropped entirely -- it is pre-observation context, not a
        prediction about the future.
    """
    horizon = traj.shape[-2]
    sl = executed_slice(n_obs_steps, n_action_steps, horizon)
    if sl == slice(None):
        return traj, traj[..., horizon:, :]
    return traj[..., sl, :], traj[..., sl.stop:, :]


# =============================================================================
# 2 + 4. Variance profile v(h), with the arm/gripper split
# =============================================================================


def variance_profile(
    traj: np.ndarray,
    dims: Sequence[int] | None = None,
    sample_axis: int = -3,
    ddof: int = 1,
    reduce: str = "mean",
) -> np.ndarray:
    """Variance across the ENSEMBLE SAMPLES as a function of horizon index.

    THE PRIMARY OUTPUT OF THIS MODULE. It is deliberately a vector v(h) of length
    `horizon`, not a scalar. Collapsing it destroys the analysis: the samples of
    a diffusion policy fan out with h no matter what the situation is (the far
    future is simply less determined by the current observation), so a scalar
    mean over h is dominated by that structural fan-out and the situational
    uncertainty we are hunting for is buried inside it. With v(h) in hand a
    caller can divide by a reference profile, look only at the executed prefix,
    or compare the same h across refills -- none of which a scalar permits.

    Parameters
    ----------
    traj : (..., K, horizon, action_dim)
        Ensemble samples. K is at `sample_axis` (default -3), so both a single
        refill (K, H, D) and a stack (N, K, H, D) work.
    dims : sequence of int, optional
        Which action dims to include. None means all. Use ARM_DIMS / GRIP_DIMS.
    ddof : int
        Passed to np.var. ddof=1 (unbiased) is the default because K is small
        (K=16 typically) and the ddof=0 estimator is biased low by a factor
        (K-1)/K = 0.94 at K=16, which matters when comparing across K.
    reduce : {"mean", "sum", "none"}
        How to combine the per-dim variances. "mean" (default) keeps the value
        comparable between subsets of different width -- so v_arm (6 dims) and
        v_grip (1 dim) are on the same scale. "sum" gives the trace of the
        covariance, i.e. total squared spread. "none" returns per-dim v(h, d).

    Returns
    -------
    (..., horizon) array, or (..., horizon, len(dims)) when reduce="none".
    """
    a = np.asarray(traj, dtype=np.float64)
    if dims is not None:
        a = np.take(a, np.asarray(dims, dtype=int), axis=-1)
    if a.shape[sample_axis] < 2:
        raise ValueError(f"need K>=2 samples to compute a variance, got {a.shape[sample_axis]}")
    per_dim = np.var(a, axis=sample_axis, ddof=ddof)  # (..., horizon, n_dims)
    if reduce == "none":
        return per_dim
    if reduce == "sum":
        return per_dim.sum(axis=-1)
    if reduce == "mean":
        return per_dim.mean(axis=-1)
    raise ValueError(f"reduce must be mean|sum|none, got {reduce!r}")


@dataclass
class VarianceProfiles:
    """Arm/gripper x executed/tail variance profiles for one refill (or a stack).

    Never provides a single fused scalar. `combined_*` fields are populated only
    when explicitly requested, so the default path cannot accidentally let the
    gripper dim dominate an all-dims L2.
    """

    v_arm: np.ndarray          # (..., horizon)  mean variance over arm dims
    v_grip: np.ndarray         # (..., horizon)  mean variance over gripper dims
    v_arm_exec: np.ndarray     # (..., n_action_steps)
    v_grip_exec: np.ndarray
    v_arm_tail: np.ndarray     # (..., horizon - end)
    v_grip_tail: np.ndarray
    exec_slice: slice
    arm_dims: tuple[int, ...]
    grip_dims: tuple[int, ...]
    v_combined: np.ndarray | None = None
    v_combined_exec: np.ndarray | None = None
    v_combined_tail: np.ndarray | None = None

    def summary(self) -> dict[str, float]:
        """Scalar reductions FOR REPORTING ONLY.

        Provided so `__main__` can print a table. Do not use these as the
        analysis: they are exactly the collapse that `variance_profile` exists to
        avoid. They are reported separately per limb and per exec/tail so that at
        least the two mixing problems (horizon fan-out and gripper phase) stay
        visible in the printout.
        """
        def m(x: np.ndarray) -> float:
            return float(np.mean(x)) if np.size(x) else float("nan")
        return {
            "arm_exec_mean": m(self.v_arm_exec),
            "arm_tail_mean": m(self.v_arm_tail),
            "grip_exec_mean": m(self.v_grip_exec),
            "grip_tail_mean": m(self.v_grip_tail),
            "arm_exec_max": float(np.max(self.v_arm_exec)) if np.size(self.v_arm_exec) else float("nan"),
            "grip_exec_max": float(np.max(self.v_grip_exec)) if np.size(self.v_grip_exec) else float("nan"),
        }


def variance_profiles(
    traj: np.ndarray,
    n_obs_steps: int | None,
    n_action_steps: int | None,
    arm_dims: Sequence[int] = ARM_DIMS,
    grip_dims: Sequence[int] = GRIP_DIMS,
    ddof: int = 1,
    combined: bool = False,
) -> VarianceProfiles:
    """Arm and gripper variance profiles, split into executed and tail parts.

    ARM/GRIPPER SEPARATION IS NOT OPTIONAL HERE, and that is the point. On this
    robot dim 6 is the 2F-85 gripper command: a step function that sits flat
    open, transitions over a handful of timesteps, then sits flat closed. Two
    samples that agree perfectly about *what* to do but disagree by three
    timesteps about *when* to close produce a variance spike of order
    (open-closed)^2/4 at the transition -- far larger than anything the six joint
    dims ever produce, and driven entirely by task phase (proximity to a grasp)
    rather than by model uncertainty. Pooling dim 6 with dims 0..5 therefore
    yields a "variance signal" that is mostly a grasp-phase detector.

    Whether that story actually holds for this dataset is an empirical question
    (a concurrent task is measuring it). This function takes no side: it returns
    both channels always, and `combined=True` additionally returns the pooled
    profile so the pooled and split versions can be compared directly.

    `arm_dims`/`grip_dims` are silently clipped to the available action_dim, so
    this works unchanged on a 6-dim (no gripper) action space -- v_grip is then
    an empty array rather than an IndexError.
    """
    a = np.asarray(traj, dtype=np.float64)
    d = a.shape[-1]
    arm = tuple(int(i) for i in arm_dims if 0 <= int(i) < d)
    grip = tuple(int(i) for i in grip_dims if 0 <= int(i) < d)

    def prof(dims: tuple[int, ...]) -> np.ndarray:
        if not dims:
            return np.zeros(a.shape[:-1][:-1] + (0,), dtype=np.float64)
        return variance_profile(a, dims=dims, ddof=ddof, reduce="mean")

    v_arm, v_grip = prof(arm), prof(grip)
    sl = executed_slice(n_obs_steps, n_action_steps, a.shape[-2])

    def split(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if v.size == 0:
            return v, v
        if sl == slice(None):
            return v, v[..., v.shape[-1]:]
        return v[..., sl], v[..., sl.stop:]

    v_arm_exec, v_arm_tail = split(v_arm)
    v_grip_exec, v_grip_tail = split(v_grip)

    out = VarianceProfiles(
        v_arm=v_arm, v_grip=v_grip,
        v_arm_exec=v_arm_exec, v_grip_exec=v_grip_exec,
        v_arm_tail=v_arm_tail, v_grip_tail=v_grip_tail,
        exec_slice=sl, arm_dims=arm, grip_dims=grip,
    )
    if combined:
        v_all = variance_profile(a, dims=None, ddof=ddof, reduce="mean")
        out.v_combined = v_all
        out.v_combined_exec, out.v_combined_tail = split(v_all)
    return out


# =============================================================================
# 5. Divergence metrics over the sample axis
# =============================================================================
#
# Four metrics, offered side by side so a caller compares them instead of
# assuming one. The first three are all second-moment functionals of the sample
# cloud and are therefore mutually redundant in a precise sense (see below);
# only pca_spectrum says anything about the SHAPE of the cloud.


def per_timestep_std(
    traj: np.ndarray,
    dims: Sequence[int] | None = None,
    ddof: int = 1,
) -> np.ndarray:
    """sqrt of `variance_profile` -- spread in the units of the action itself.

    SECOND-MOMENT METRIC. Identical information content to variance_profile;
    provided because a std in action units is easier to threshold by eye. Cannot
    distinguish two narrow modes from one wide unimodal spread -- both give the
    same number whenever their second moments match. Use `multimodality_probe`
    for that question.
    """
    return np.sqrt(variance_profile(traj, dims=dims, ddof=ddof, reduce="mean"))


def mean_pairwise_l2(
    traj: np.ndarray,
    dims: Sequence[int] | None = None,
    per_timestep: bool = False,
) -> np.ndarray | float:
    """Mean Euclidean distance between distinct pairs of samples.

    SECOND-MOMENT METRIC. For an isotropic cloud the mean SQUARED pairwise
    distance is exactly ``2 * trace(Cov)`` with the ddof=1 covariance, so this is
    a monotone transform of `variance_profile` up to the sqrt being taken before
    rather than after averaging (that difference is a Jensen gap, not new
    information). It cannot separate two narrow modes from one wide blob.

    Parameters
    ----------
    per_timestep : bool
        False (default): flatten (horizon x dims) and return one scalar per
        refill -- the whole-trajectory divergence.
        True: return an (..., horizon) profile of pairwise distance at each h.

    Returns
    -------
    float (or (...,) array) when per_timestep=False; (..., horizon) otherwise.
    """
    a = np.asarray(traj, dtype=np.float64)
    if dims is not None:
        a = np.take(a, np.asarray(dims, dtype=int), axis=-1)
    if per_timestep:
        # (..., K, H, D) -> distances within each h
        diff = a[..., :, None, :, :] - a[..., None, :, :, :]     # (..., K, K, H, D)
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))               # (..., K, K, H)
        k = a.shape[-3]
        iu = np.triu_indices(k, k=1)
        return dist[..., iu[0], iu[1], :].mean(axis=-2)          # (..., H)
    flat = a.reshape(*a.shape[:-2], -1)                          # (..., K, H*D)
    diff = flat[..., :, None, :] - flat[..., None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))                   # (..., K, K)
    k = flat.shape[-2]
    iu = np.triu_indices(k, k=1)
    vals = dist[..., iu[0], iu[1]].mean(axis=-1)
    return float(vals) if np.ndim(vals) == 0 else vals


def endpoint_spread(
    traj: np.ndarray,
    dims: Sequence[int] | None = None,
    index: int = -1,
) -> float | np.ndarray:
    """RMS distance of the samples' endpoints from their centroid.

    SECOND-MOMENT METRIC. This is just sqrt(trace(Cov)) at a single timestep, so
    it is `variance_profile` evaluated at one h -- the most aggressive collapse
    of all, and included mainly because it is the metric people reach for first.
    It answers "where does this rollout end up" and nothing about cloud shape:
    two narrow modes and one wide blob with the same second moment are identical
    under it.

    `index` selects the horizon step; -1 (the far end of the horizon) is the
    conventional choice but note that step is in the never-executed tail whenever
    n_action_steps < horizon - n_obs_steps + 1, so pass the executed slice's last
    index if you care about what the robot did.
    """
    a = np.asarray(traj, dtype=np.float64)
    if dims is not None:
        a = np.take(a, np.asarray(dims, dtype=int), axis=-1)
    pts = a[..., :, index, :]                     # (..., K, D)
    centered = pts - pts.mean(axis=-2, keepdims=True)
    val = np.sqrt(np.sum(centered ** 2, axis=-1).mean(axis=-1))
    return float(val) if np.ndim(val) == 0 else val


@dataclass
class PCASpectrum:
    """Result of `pca_spectrum`."""
    eigenvalues: np.ndarray        # (min(K-1, F),) descending, = s^2/(K-1)
    explained_ratio: np.ndarray
    components: np.ndarray         # (n_comp, F) right singular vectors
    scores: np.ndarray             # (K, n_comp) sample coordinates
    participation_ratio: float     # (sum l)^2 / sum(l^2): effective dimension
    total_variance: float

    def top1_ratio(self) -> float:
        return float(self.explained_ratio[0]) if self.explained_ratio.size else float("nan")


def pca_spectrum(
    traj: np.ndarray,
    dims: Sequence[int] | None = None,
    n_components: int | None = None,
) -> PCASpectrum:
    """PCA over the flattened trajectories, treating the K samples as datapoints.

    THE ONLY METRIC HERE THAT SEES CLOUD SHAPE. per_timestep_std,
    mean_pairwise_l2 and endpoint_spread are all second-moment functionals: they
    measure how big the cloud is and are provably blind to whether it is one wide
    unimodal blob or two tight, well-separated modes. That distinction is the
    interesting one -- two modes at a fork in the task is an actionable decision
    point, a wide blob is just a sloppy but unambiguous motion -- and it only
    shows up in the structure of the cloud.

    PCA is the first step: a genuinely 2-modal cloud concentrates its variance in
    one direction (high `top1_ratio`, low `participation_ratio`) and its
    projection onto that direction is bimodal. That projection is what
    `multimodality_probe` then tests. Note PCA alone is NOT sufficient -- a
    cigar-shaped unimodal cloud looks the same in the spectrum; the spectrum
    tells you WHERE to look, the dip/BIC tests tell you what is there.

    MEASURED CAVEAT ON `top1_ratio` (from `_self_check`): do NOT use a high
    explained-variance ratio as a multimodality indicator. In a flattened
    trajectory space of a few hundred dimensions, isotropic per-timestep sample
    noise contributes ``n_features * sigma_noise^2`` of total variance, which
    easily exceeds the variance carried by the mode separation itself. The
    self-check builds a cloud with two unmistakable modes (dip p=0.000,
    dBIC=+80) whose PC1 still explains only 0.53 of the variance and whose
    participation ratio is 3.4. The spectrum is a pointer, not a detector: PC1
    reliably *finds the direction* the modes live along (the dip test on
    ``scores[:, 0]`` fires cleanly), but its eigenvalue share is diluted by the
    noise dimensions. Compare `top1_ratio` against the uniform share ``1/rank``,
    never against an absolute cutoff like 0.9.

    Eigenvalues use the ddof=1 convention (s^2/(K-1)) so that
    ``eigenvalues.sum() == variance_profile(..., reduce='sum').sum()``.
    """
    a = np.asarray(traj, dtype=np.float64)
    if dims is not None:
        a = np.take(a, np.asarray(dims, dtype=int), axis=-1)
    if a.ndim != 3:
        raise ValueError(f"pca_spectrum expects a single refill (K, H, D), got {a.shape}")
    k = a.shape[0]
    x = a.reshape(k, -1)
    x = x - x.mean(axis=0, keepdims=True)
    # full_matrices=False -> at most min(K, F) components; rank is <= K-1 after centering.
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    eig = (s ** 2) / max(k - 1, 1)
    rank = min(k - 1, x.shape[1])
    eig, vt = eig[:rank], vt[:rank]
    if n_components is not None:
        eig, vt = eig[:n_components], vt[:n_components]
    total = float(eig.sum())
    ratio = eig / total if total > 0 else np.zeros_like(eig)
    pr = float((eig.sum() ** 2) / np.sum(eig ** 2)) if np.sum(eig ** 2) > 0 else float("nan")
    return PCASpectrum(
        eigenvalues=eig,
        explained_ratio=ratio,
        components=vt,
        scores=x @ vt.T,
        participation_ratio=pr,
        total_variance=total,
    )


# =============================================================================
# 6. Soft multimodality check
# =============================================================================


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sample bimodality coefficient BC = (g^2 + 1) / (k + correction).

    The DOCUMENTED FALLBACK used when the optional `diptest` package is absent
    (scipy has no dip test -- there is no scipy.stats.diptest). BC is the
    SAS-style statistic: g is the sample skewness, k the sample excess kurtosis,
    and the denominator carries the small-sample correction
    ``3(n-1)^2 / ((n-2)(n-3))``. BC > 5/9 ~= 0.555 is the conventional
    (heuristic, not a test) threshold: the uniform distribution sits exactly at
    5/9, a normal at 1/3.

    Weaker than the dip test and easily fooled -- a heavy-tailed unimodal sample
    can push BC up, and BC needs n > 3. Treat it as a hint only, and prefer
    installing `diptest`. Returns NaN for n <= 3 or zero variance.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n <= 3:
        return float("nan")
    sd = v.std(ddof=1)
    if sd <= 0:
        return float("nan")
    z = (v - v.mean()) / sd
    g = float(np.mean(z ** 3))          # skewness (biased, matching the SAS form)
    kurt = float(np.mean(z ** 4)) - 3.0  # excess kurtosis
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denom = kurt + corr
    if denom <= 0:
        return float("nan")
    return float((g ** 2 + 1.0) / denom)


def dip_test(x: np.ndarray) -> tuple[float, float | None, str]:
    """Hartigan & Hartigan dip test for unimodality, with a documented fallback.

    Returns ``(statistic, p_value_or_None, method)`` where method is
    ``"diptest"`` (the real dip test, optional package) or
    ``"bimodality_coefficient"`` (the fallback, p is None because BC is a
    descriptive statistic and not a test).

    *** KNOWN POWER LIMITATION -- READ BEFORE INTERPRETING ANY RESULT ***
    The ensemble is K=16 samples and the dip test is badly underpowered there.
    These are not quoted figures, they were MEASURED with this code (see
    `_self_check`, `diptest` 0.11.0, 400 trials per point, equal-weight
    two-Gaussian mixture, 8+8 samples, alpha=0.05, separation in units of the
    component sigma):

        separation   1.0s   1.5s   2.0s   2.5s   3.0s   4.0s
        power        0.01   0.10   0.25   0.56   0.83   0.99

    with a false-positive rate of 0.006 under a true N(0,1) (conservative, as
    the dip is known to be). So:

      * a FLAG (small p) at K=16 is a SOFT SIGNAL worth investigating, never a
        verdict that the policy is multimodal. It is at least a *specific*
        signal: the false-positive rate is well below nominal.
      * a NON-flag is nearly uninformative. At 2.5 sigma separation -- already
        visually obvious as two modes -- you miss 44% of them, and below 2 sigma
        you miss essentially all of them. The absence of flags must NEVER be
        written up as "the policy is unimodal".
      * the fix is free here. This is OFFLINE analysis with no real-time
        constraint, so re-run the sampler with a much larger K (K=100-500) on the
        stored observations and re-test. Any quantitative claim about
        multimodality should come from that larger-K run, not from the K=16
        online ensemble, which was sized for latency and not for statistics.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 4:
        return float("nan"), None, "insufficient_samples"
    if _diptest_pkg is not None:
        try:
            stat, pval = _diptest_pkg.diptest(v)
            return float(stat), float(pval), "diptest"
        except Exception:  # noqa: BLE001 - fall through to the fallback
            pass
    return bimodality_coefficient(v), None, "bimodality_coefficient"


def _gmm1d_em(
    x: np.ndarray,
    n_components: int,
    n_restarts: int = 8,
    n_iter: int = 300,
    tol: float = 1e-7,
    seed: int = 0,
) -> tuple[float, int]:
    """Tiny 1-D diagonal GMM by EM. Returns (best log-likelihood, n_free_params).

    Deliberately hand-rolled rather than sklearn: this module's dependency
    contract is numpy + h5py + scipy only. Variances are floored at a small
    fraction of the data variance to stop a component from collapsing onto a
    single point and sending the likelihood to +inf, which is the standard
    degeneracy of unconstrained Gaussian mixture likelihoods.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    n = v.size
    data_var = float(np.var(v)) if n > 1 else 1.0
    floor = max(data_var * 1e-4, 1e-12)
    n_params = 3 * n_components - 1  # means + variances + (weights - 1)
    if n_components == 1:
        var = max(data_var, floor)
        ll = float(np.sum(-0.5 * ((v - v.mean()) ** 2 / var + np.log(2 * np.pi * var))))
        return ll, n_params

    rng = np.random.default_rng(seed)
    best_ll = -np.inf
    for _ in range(n_restarts):
        idx = rng.choice(n, size=n_components, replace=False)
        mu = v[idx].astype(np.float64).copy()
        var = np.full(n_components, max(data_var, floor))
        w = np.full(n_components, 1.0 / n_components)
        prev_ll = -np.inf
        ll = -np.inf
        for _ in range(n_iter):
            # E-step in log space for stability.
            logp = (
                np.log(np.maximum(w, 1e-300))[None, :]
                - 0.5 * np.log(2 * np.pi * var)[None, :]
                - 0.5 * (v[:, None] - mu[None, :]) ** 2 / var[None, :]
            )
            m = logp.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))
            ll = float(lse.sum())
            resp = np.exp(logp - lse[:, None])
            # M-step
            nk = resp.sum(axis=0) + 1e-300
            w = nk / n
            mu = (resp * v[:, None]).sum(axis=0) / nk
            var = np.maximum((resp * (v[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk, floor)
            if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll
        if ll > best_ll:
            best_ll = ll
    return float(best_ll), n_params


def gmm_bic_compare(x: np.ndarray, seed: int = 0) -> dict[str, float]:
    """BIC of a 1-component vs a 2-component Gaussian mixture on a 1-D sample.

    Returns ``{bic_k1, bic_k2, delta_bic, loglik_k1, loglik_k2, n}`` with
    ``delta_bic = bic_k1 - bic_k2``: POSITIVE means k=2 is preferred. The usual
    reading of a BIC difference (Kass & Raftery) is >2 positive, >6 strong, >10
    very strong -- but with n=16 samples and 3 extra free parameters the penalty
    ``3 * ln(16) = 8.3`` is large, so only a genuinely well-separated pair of
    modes will clear it. That conservatism is intentional: at K=16 we would
    rather miss modes than manufacture them.

    Same power caveat as `dip_test` applies: at K=16 this is a soft signal.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    out = {"n": float(n)}
    if n < 6:
        out.update(bic_k1=float("nan"), bic_k2=float("nan"), delta_bic=float("nan"),
                   loglik_k1=float("nan"), loglik_k2=float("nan"))
        return out
    ll1, p1 = _gmm1d_em(v, 1, seed=seed)
    ll2, p2 = _gmm1d_em(v, 2, seed=seed)
    bic1 = -2 * ll1 + p1 * math.log(n)
    bic2 = -2 * ll2 + p2 * math.log(n)
    out.update(bic_k1=float(bic1), bic_k2=float(bic2), delta_bic=float(bic1 - bic2),
               loglik_k1=float(ll1), loglik_k2=float(ll2))
    return out


@dataclass
class MultimodalityResult:
    """Per-component multimodality evidence for one refill. SOFT SIGNAL ONLY."""
    pca: PCASpectrum
    components_tested: int
    dip_stats: list[float]
    dip_pvals: list[float | None]
    dip_method: str
    bic_deltas: list[float]
    flagged: bool
    flag_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "components_tested": self.components_tested,
            "top1_explained": self.pca.top1_ratio(),
            "participation_ratio": self.pca.participation_ratio,
            "dip_method": self.dip_method,
            "dip_stats": self.dip_stats,
            "dip_pvals": self.dip_pvals,
            "bic_deltas": self.bic_deltas,
            "flagged": self.flagged,
            "flag_reason": self.flag_reason,
        }


def multimodality_probe(
    traj: np.ndarray,
    dims: Sequence[int] | None = ARM_DIMS,
    n_components: int = 2,
    dip_alpha: float = 0.05,
    bic_threshold: float = 6.0,
) -> MultimodalityResult:
    """PCA -> top components -> dip test + GMM k=1 vs k=2 BIC, per component.

    Pipeline: flatten the K trajectories, PCA (the only shape-aware view we
    have), project onto the top `n_components` directions, and test each 1-D
    projection for unimodality two independent ways. Two tests rather than one
    because they fail differently -- the dip is distribution-free but weak, the
    BIC comparison is parametric and assumes Gaussian modes.

    *** POWER LIMITATION, RESTATED BECAUSE IT GOVERNS EVERY CONCLUSION ***
    At K=16 the dip test has roughly 50% power even against well-separated
    modes. `flagged=True` is therefore a soft signal to investigate, NOT a
    verdict; `flagged=False` is close to uninformative and must never be written
    up as evidence of unimodality. Because this analysis is offline and has no
    real-time deadline, the correct response to any interesting flag is to re-run
    the sampler on the same stored observation with K in the hundreds and repeat
    this probe -- only then is a quantitative multimodality claim defensible.

    `dims` defaults to ARM_DIMS: the gripper dim is excluded by default because
    its step-like shape makes the flattened cloud trivially "bimodal" whenever
    samples disagree on grasp timing, which would flag task phase rather than
    model uncertainty. Pass ``dims=None`` to include it deliberately.
    """
    pca = pca_spectrum(traj, dims=dims)
    n_comp = int(min(n_components, pca.scores.shape[1])) if pca.scores.size else 0
    dip_stats: list[float] = []
    dip_pvals: list[float | None] = []
    bic_deltas: list[float] = []
    method = "n/a"
    reasons: list[str] = []
    for c in range(n_comp):
        proj = pca.scores[:, c]
        stat, pval, method = dip_test(proj)
        bic = gmm_bic_compare(proj, seed=c)
        dip_stats.append(float(stat))
        dip_pvals.append(None if pval is None else float(pval))
        bic_deltas.append(float(bic["delta_bic"]))
        if pval is not None and pval < dip_alpha:
            reasons.append(f"dip p={pval:.3f}<{dip_alpha} on PC{c + 1}")
        if method == "bimodality_coefficient" and np.isfinite(stat) and stat > 5.0 / 9.0:
            reasons.append(f"BC={stat:.3f}>0.556 on PC{c + 1}")
        if np.isfinite(bic["delta_bic"]) and bic["delta_bic"] > bic_threshold:
            reasons.append(f"dBIC={bic['delta_bic']:.1f}>{bic_threshold} on PC{c + 1}")
    return MultimodalityResult(
        pca=pca,
        components_tested=n_comp,
        dip_stats=dip_stats,
        dip_pvals=dip_pvals,
        dip_method=method,
        bic_deltas=bic_deltas,
        flagged=bool(reasons),
        flag_reason="; ".join(reasons) if reasons else "no soft flag (NOT evidence of unimodality)",
    )


# =============================================================================
# 7. STAC-style temporal divergence
# =============================================================================


def _median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    """RBF gamma = 1/(2*median_pairwise_dist^2) on the pooled sample."""
    z = np.vstack([x, y])
    diff = z[:, None, :] - z[None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    iu = np.triu_indices(z.shape[0], k=1)
    med = float(np.median(d2[iu]))
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return 1.0 / med


def mmd2_unbiased(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float | None = None,
    n_permutations: int = 0,
    seed: int = 0,
) -> dict[str, float]:
    """Unbiased MMD^2 between two sample sets under an RBF kernel.

    Parameters
    ----------
    x, y : (n, F), (m, F)
    gamma : RBF bandwidth 1/(2 sigma^2). None -> median heuristic on the pooled
        sample, the standard default; note this makes the statistic's scale
        data-dependent, so only compare MMD values computed with the SAME gamma
        (pass an explicit gamma when comparing across refills).
    n_permutations : if > 0, a permutation test of H0: same distribution.

    Returns ``{mmd2, gamma, p_value(optional), n, m}``. The unbiased estimator
    excludes the diagonal terms and CAN GO SLIGHTLY NEGATIVE under H0 -- that is
    expected, not a bug.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    n, m = a.shape[0], b.shape[0]
    if n < 2 or m < 2:
        return {"mmd2": float("nan"), "gamma": float("nan"), "n": float(n), "m": float(m)}
    if gamma is None:
        gamma = _median_heuristic_gamma(a, b)

    def _k(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        d2 = np.sum(p ** 2, axis=1)[:, None] + np.sum(q ** 2, axis=1)[None, :] - 2 * p @ q.T
        return np.exp(-gamma * np.maximum(d2, 0.0))

    def _stat(p: np.ndarray, q: np.ndarray) -> float:
        kxx, kyy, kxy = _k(p, p), _k(q, q), _k(p, q)
        np_, nq = p.shape[0], q.shape[0]
        sxx = (kxx.sum() - np.trace(kxx)) / (np_ * (np_ - 1))
        syy = (kyy.sum() - np.trace(kyy)) / (nq * (nq - 1))
        return float(sxx + syy - 2.0 * kxy.mean())

    stat = _stat(a, b)
    out = {"mmd2": stat, "gamma": float(gamma), "n": float(n), "m": float(m)}
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        pooled = np.vstack([a, b])
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(pooled.shape[0])
            if _stat(pooled[perm[:n]], pooled[perm[n:]]) >= stat:
                count += 1
        out["p_value"] = (count + 1.0) / (n_permutations + 1.0)
    return out


def _shared_gamma_for(
    ef: "EnsembleFile",
    dims: Sequence[int] | None,
    step: int,
) -> float | None:
    """Median-heuristic RBF gamma estimated ONCE, on the first valid overlap.

    See `stac_series` for why a per-pair bandwidth would be wrong. Returns None
    if no usable overlap exists, in which case the caller lets `mmd2_unbiased`
    fall back to its own per-call heuristic (a degenerate file, so comparability
    is moot).
    """
    horizon = ef.horizon or 0
    if horizon <= step:
        return None
    for i in range(ef.n_refills - 1):
        p, q = ef.trajectories[i], ef.trajectories[i + 1]
        if dims is not None:
            idx = np.asarray([d for d in dims if 0 <= d < p.shape[-1]], dtype=int)
            if idx.size == 0:
                return None
            p, q = np.take(p, idx, axis=-1), np.take(q, idx, axis=-1)
        overlap = horizon - step
        a = p[:, step:step + overlap, :].reshape(p.shape[0], -1)
        b = q[:, :overlap, :].reshape(q.shape[0], -1)
        if a.shape[0] >= 2 and b.shape[0] >= 2 and a.shape[1] > 0:
            return _median_heuristic_gamma(a.astype(np.float64), b.astype(np.float64))
    return None


@dataclass
class STACResult:
    """One consecutive-refill temporal divergence measurement."""
    row_prev: int
    row_next: int
    lag: int
    overlap_len: int
    mmd2: float
    p_value: float | None
    gamma: float
    t_wall: float | None


def stac_temporal_divergence(
    traj_prev: np.ndarray,
    traj_next: np.ndarray,
    lag: int,
    dims: Sequence[int] | None = ARM_DIMS,
    gamma: float | None = None,
    n_permutations: int = 0,
    seed: int = 0,
) -> dict[str, float]:
    """MMD between the temporally OVERLAPPING regions of two consecutive refills.

    This is the post-hoc version of the temporal-consistency signal from STAC
    (Agia et al., "Unpacking Failure Modes of Generative Policies: Runtime
    Monitoring of Consistency and Progress", CoRL 2024). The idea it borrows: a
    policy that is confident should predict roughly the SAME future when it
    re-plans a moment later; when the two consecutive sample batches disagree
    about the same wall-clock interval, something changed that the policy did not
    expect. Compared to across-sample variance this is a *different* axis
    (temporal consistency vs instantaneous spread), which is exactly why it is
    worth having as a comparison signal.

    Two honest caveats:
      * This is NOT a reimplementation of STAC. STAC as published compares the
        current plan against the previously *committed* actions with its own
        statistical machinery; here we compare the two full sample BATCHES with a
        generic MMD. It is "STAC-style", inspired by, not equivalent to.
      * It costs nothing extra: both batches are already on disk, so no
        additional sampling is required.

    Alignment: refill n+1 is taken `lag` timesteps after refill n (lag =
    n_action_steps for back-to-back refills), and both arrays share the same
    internal horizon indexing, so global time index j in `traj_prev` is index
    ``j - lag`` in `traj_next`. The overlap is therefore
    ``traj_prev[:, lag:]`` against ``traj_next[:, :horizon-lag]``.

    Returns the `mmd2_unbiased` dict plus ``overlap_len`` and ``lag``.
    """
    p = np.asarray(traj_prev, dtype=np.float64)
    q = np.asarray(traj_next, dtype=np.float64)
    if p.ndim != 3 or q.ndim != 3:
        raise ValueError(f"expected (K, H, D) arrays, got {p.shape} and {q.shape}")
    if dims is not None:
        idx = np.asarray(dims, dtype=int)
        p, q = np.take(p, idx, axis=-1), np.take(q, idx, axis=-1)
    horizon = min(p.shape[1], q.shape[1])
    lag = int(lag)
    overlap = horizon - lag
    if overlap <= 0:
        return {"mmd2": float("nan"), "gamma": float("nan"), "overlap_len": 0.0,
                "lag": float(lag), "n": float(p.shape[0]), "m": float(q.shape[0])}
    a = p[:, lag:lag + overlap, :]
    b = q[:, :overlap, :]
    out = mmd2_unbiased(a, b, gamma=gamma, n_permutations=n_permutations, seed=seed)
    out["overlap_len"] = float(overlap)
    out["lag"] = float(lag)
    return out


def stac_series(
    ef: EnsembleFile,
    dims: Sequence[int] | None = ARM_DIMS,
    gamma: float | None = None,
    n_permutations: int = 0,
    max_pairs: int | None = None,
) -> list[STACResult]:
    """Run `stac_temporal_divergence` over every consecutive refill pair in a file.

    The lag between two stored rows is derived from the `refill_idx` meta column
    (``(idx_next - idx_prev) * n_action_steps``) so that dropped refills, which
    leave a gap in refill_idx, get the correct larger lag instead of being
    silently treated as adjacent. Falls back to `n_action_steps` when refill_idx
    is unavailable.

    GAMMA IS SHARED ACROSS ALL PAIRS. When `gamma is None` it is estimated ONCE
    by the median heuristic on the first valid overlap and then reused for every
    pair. This matters: `mmd2_unbiased`'s per-call median heuristic would give
    each pair its own bandwidth, and MMD values computed under different kernels
    are not on a common scale -- yet the whole point of this function is to
    return a time SERIES that gets compared across refills (and thresholded by
    `conformal_calibrate`). Estimating gamma per pair would silently make that
    comparison meaningless. Pass an explicit `gamma` to compare across files.
    """
    n = ef.n_refills
    if n < 2:
        return []
    step = ef.n_action_steps or 1
    if gamma is None:
        gamma = _shared_gamma_for(ef, dims=dims, step=step)
    ridx = ef.meta_column("refill_idx", aligned_to_nd=True)
    twall = ef.meta_column("t_wall", aligned_to_nd=True)
    results: list[STACResult] = []
    pairs = range(n - 1) if max_pairs is None else range(min(n - 1, max_pairs))
    for i in pairs:
        if ridx is not None and ridx.size > i + 1:
            lag = int(round(float(ridx[i + 1] - ridx[i]))) * step
        else:
            lag = step
        r = stac_temporal_divergence(
            ef.trajectories[i], ef.trajectories[i + 1], lag=lag, dims=dims,
            gamma=gamma, n_permutations=n_permutations, seed=i,
        )
        results.append(STACResult(
            row_prev=i, row_next=i + 1, lag=lag,
            overlap_len=int(r.get("overlap_len", 0)),
            mmd2=float(r.get("mmd2", float("nan"))),
            p_value=(float(r["p_value"]) if "p_value" in r else None),
            gamma=float(r.get("gamma", float("nan"))),
            t_wall=(float(twall[i + 1]) if twall is not None and twall.size > i + 1 else None),
        ))
    return results


# =============================================================================
# 8. Conformal-style calibration from SUCCESS episodes only
# =============================================================================


@dataclass
class ConformalBand:
    """A time-varying threshold band fit on success episodes."""
    phase: np.ndarray            # (n_bins,) normalized phase centre in [0, 1]
    threshold: np.ndarray        # (n_bins,) the (1-alpha) conformal quantile
    n_per_bin: np.ndarray        # (n_bins,) episodes contributing to each bin
    alpha: float
    n_episodes: int
    global_threshold: float
    time_varying: bool

    def score_alarm(self, scores: np.ndarray) -> np.ndarray:
        """Boolean alarm for one episode's score series, compared bin-wise."""
        s = np.asarray(scores, dtype=np.float64).ravel()
        thr = self._threshold_for(s.size)
        return s > thr

    def _threshold_for(self, length: int) -> np.ndarray:
        """Resample the band onto an episode of `length` refills."""
        if length <= 0:
            return np.zeros(0)
        if not self.time_varying:
            return np.full(length, self.global_threshold)
        query = (np.arange(length) + 0.5) / length
        valid = np.isfinite(self.threshold)
        if not np.any(valid):
            return np.full(length, self.global_threshold)
        return np.interp(query, self.phase[valid], self.threshold[valid])


def conformal_calibrate(
    success_scores: Sequence[np.ndarray],
    alpha: float = 0.1,
    n_bins: int = 10,
    time_varying: bool = True,
) -> ConformalBand:
    """Split-conformal style threshold band from SUCCESS episodes only.

    This is the whole reason the approach is attractive here: it needs NO failure
    data. Take episodes that are known to have succeeded, treat each refill's
    divergence score as a nonconformity score under the "nominal" distribution,
    and take the (1-alpha) quantile with the finite-sample conformal correction
    ``ceil((n+1)(1-alpha)) / n``. Under exchangeability of scores between the
    calibration episodes and a new nominal episode, a new nominal score exceeds
    the threshold with probability at most alpha -- so exceedances are a
    calibrated false-alarm-rate anomaly signal rather than an arbitrary cutoff.

    The band is TIME-VARYING because a single global threshold would be wrong for
    the same structural reason a scalar v(h) is wrong: divergence is not
    stationary over an episode (approach, grasp and retreat have systematically
    different spread). Episodes are resampled onto `n_bins` bins of NORMALIZED
    phase so that episodes of different lengths can be pooled.

    WHAT THE TIME-VARYING BAND DOES AND DOES NOT BUY YOU -- this was measured,
    and the naive expectation is wrong. A single global threshold achieves
    *marginal* coverage just fine (it is the pooled quantile, so by construction
    it exceeds on ~alpha of all held-out nominal timesteps -- the self-check
    measures 0.11 against a target of 0.10, slightly BETTER than the band's
    0.14). Marginal coverage is not the property we want. What the global
    threshold fails at is *conditional* coverage: its exceedances are all
    concentrated in the naturally-high-divergence phase and essentially absent
    elsewhere, so it is a phase detector wearing a threshold's clothing -- it
    cries wolf during every grasp and is blind during the whole approach. The
    self-check measures per-phase-bin exceedance for both and asserts on the
    SPREAD across bins, not the aggregate: the global threshold's rate swings
    across bins by an order of magnitude while the band's stays near alpha
    everywhere. If you only ever report an aggregate false-alarm rate you will
    conclude, incorrectly, that the global threshold is fine.

    Assumption that must be stated when reporting: split-conformal guarantees
    hold under exchangeability. Scores WITHIN one episode are strongly
    autocorrelated, so the per-timestep guarantee is approximate; the honest
    reading is a per-episode guarantee across episodes, and any per-refill claim
    should be validated empirically with held-out success episodes (the
    self-check at the bottom of this file does exactly that).

    Parameters
    ----------
    success_scores : sequence of 1-D arrays, one per SUCCESS episode. Ragged
        lengths are fine.
    alpha : target exceedance rate (0.1 -> a 90% band).
    """
    eps = [np.asarray(s, dtype=np.float64).ravel() for s in success_scores]
    eps = [e[np.isfinite(e)] for e in eps if np.size(e)]
    n_ep = len(eps)
    if n_ep == 0:
        return ConformalBand(np.zeros(0), np.zeros(0), np.zeros(0), alpha, 0,
                             float("nan"), time_varying)

    pooled = np.concatenate(eps)
    global_thr = _conformal_quantile(pooled, alpha)

    centres = (np.arange(n_bins) + 0.5) / n_bins
    if not time_varying:
        return ConformalBand(centres, np.full(n_bins, global_thr),
                             np.full(n_bins, n_ep), alpha, n_ep, global_thr, False)

    # Resample every episode onto the shared phase grid, then take the conformal
    # quantile ACROSS EPISODES at each phase -- so the exchangeable unit is the
    # episode, which is what the guarantee actually rests on.
    grid = np.empty((n_ep, n_bins), dtype=np.float64)
    for i, e in enumerate(eps):
        if e.size == 1:
            grid[i, :] = e[0]
        else:
            src = (np.arange(e.size) + 0.5) / e.size
            grid[i, :] = np.interp(centres, src, e)
    thr = np.array([_conformal_quantile(grid[:, b], alpha) for b in range(n_bins)])
    counts = np.full(n_bins, n_ep, dtype=np.float64)
    return ConformalBand(centres, thr, counts, alpha, n_ep, global_thr, True)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile of `scores`.

    Returns +inf when n is too small for the requested alpha (i.e. when
    ceil((n+1)(1-alpha)) > n) -- the honest answer, since no finite threshold can
    give that coverage guarantee from that few points. Callers must handle inf.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = s.size
    if n == 0:
        return float("nan")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    return float(s[rank - 1])


def refill_scores(
    ef: EnsembleFile,
    dims: Sequence[int] | None = ARM_DIMS,
    executed_only: bool = True,
    statistic: str = "mean_var",
) -> np.ndarray:
    """One scalar divergence score per successful refill, for calibration input.

    This is the ONE place a scalar collapse is legitimate: conformal calibration
    needs a single nonconformity score per event. Even so the defaults keep the
    two confounds out -- `dims=ARM_DIMS` excludes the gripper and
    `executed_only=True` excludes the never-executed tail, so the score is not
    dominated by horizon fan-out or by grasp-phase timing jitter.

    statistic: "mean_var" | "max_var" | "pairwise_l2" | "endpoint".
    """
    if ef.n_refills == 0:
        return np.zeros(0)
    traj = ef.trajectories
    if executed_only:
        traj, _ = executed_tail_split(traj, ef.n_obs_steps, ef.n_action_steps)
    if statistic == "mean_var":
        return variance_profile(traj, dims=dims, reduce="mean").mean(axis=-1)
    if statistic == "max_var":
        return variance_profile(traj, dims=dims, reduce="mean").max(axis=-1)
    if statistic == "pairwise_l2":
        return np.asarray(mean_pairwise_l2(traj, dims=dims), dtype=np.float64)
    if statistic == "endpoint":
        return np.asarray(endpoint_spread(traj, dims=dims), dtype=np.float64)
    raise ValueError(f"unknown statistic {statistic!r}")


# =============================================================================
# Report
# =============================================================================


def report(path: str, max_refills_detailed: int = 3) -> str:
    """Human-readable summary of one ensemble file."""
    ef = load(path)
    lines: list[str] = ["=" * 72, ef.describe(), "=" * 72]
    if ef.n_refills == 0:
        lines.append("\nNo successful refills stored -- nothing to analyse.")
        return "\n".join(lines)

    if ef.norm_stats is None:
        lines.append(
            "\nNOTE: no norm_stats provenance in this file (written before the\n"
            "      provenance change). All numbers below are in NORMALIZED action\n"
            "      units and cannot be converted to physical units."
        )

    vp = variance_profiles(ef.trajectories, ef.n_obs_steps, ef.n_action_steps, combined=True)
    lines.append("\n--- variance profile v(h), averaged over refills ---")
    lines.append("  (NEVER collapse this to one number; per-h is the signal)")
    v_arm_m = vp.v_arm.mean(axis=0)
    v_grip_m = vp.v_grip.mean(axis=0) if vp.v_grip.size else None
    sl = vp.exec_slice
    lines.append(f"  executed slice = [{sl.start}:{sl.stop}) of horizon {ef.horizon}")
    lines.append("    h  |    v_arm    |   v_grip    | part")
    for h in range(v_arm_m.shape[-1]):
        part = "exec" if (sl == slice(None) or sl.start <= h < sl.stop) else (
            "pre" if sl.start > h else "tail")
        g = f"{v_grip_m[h]:11.6f}" if v_grip_m is not None else "        n/a"
        lines.append(f"  {h:3d}  | {v_arm_m[h]:11.6f} | {g} | {part}")

    s = vp.summary()
    lines.append("\n--- scalar reductions (REPORTING ONLY, not the analysis) ---")
    for kk, vv in s.items():
        lines.append(f"  {kk:16s} = {vv:.6f}")
    if vp.v_combined_exec is not None and vp.v_arm_exec.size:
        ratio = float(np.mean(vp.v_combined_exec) / max(np.mean(vp.v_arm_exec), 1e-12))
        lines.append(f"  combined/arm     = {ratio:.3f}x   "
                     "(>1 means the gripper dim inflates the pooled metric)")

    lines.append("\n--- divergence metrics, first refills ---")
    for i in range(min(max_refills_detailed, ef.n_refills)):
        t = ef.trajectories[i]
        te, _ = executed_tail_split(t, ef.n_obs_steps, ef.n_action_steps)
        lines.append(
            f"  refill[{i}] arm: pairwiseL2={mean_pairwise_l2(te, ARM_DIMS):.5f} "
            f"endpoint={endpoint_spread(te, ARM_DIMS):.5f}"
        )
        pc = pca_spectrum(te, dims=ARM_DIMS)
        lines.append(f"             PCA top1={pc.top1_ratio():.3f} "
                     f"participation_ratio={pc.participation_ratio:.2f} "
                     f"(only this metric sees cloud SHAPE)")
        mm = multimodality_probe(te)
        lines.append(f"             multimodality[{mm.dip_method}]: {mm.flag_reason}")

    lines.append("\n  (per_timestep_std / mean_pairwise_l2 / endpoint_spread are all\n"
                 "   second-moment metrics: they CANNOT tell two narrow modes from one\n"
                 "   wide unimodal spread. Only PCA + dip/BIC address that, and at\n"
                 "   K=16 the dip has measured power 0.25 at 2-sigma and 0.57 at\n"
                 "   2.5-sigma separation -- a flag is a soft signal, and a NON-flag\n"
                 "   is NOT evidence of unimodality. Re-sample at K=100-500 offline.)")

    st = stac_series(ef, max_pairs=min(5, ef.n_refills - 1))
    if st:
        lines.append("\n--- STAC-style temporal divergence (consecutive refills) ---")
        lines.append(f"  shared RBF gamma={st[0].gamma:.6g} (one bandwidth for all "
                     "pairs, so these are comparable)")
        for r in st:
            lines.append(f"  {r.row_prev}->{r.row_next} lag={r.lag} "
                         f"overlap={r.overlap_len} MMD^2={r.mmd2:.6f}")

    sc = refill_scores(ef)
    if sc.size:
        band = conformal_calibrate([sc], alpha=0.1)
        lines.append("\n--- conformal band (from THIS file only -- pass a list of\n"
                     "    SUCCESS episodes for a real calibration) ---")
        lines.append(f"  n_episodes={band.n_episodes} alpha={band.alpha} "
                     f"global_threshold={band.global_threshold:.6f}")
        lines.append("  phase : " + " ".join(f"{p:.2f}" for p in band.phase))
        lines.append("  thresh: " + " ".join(f"{t:.4f}" for t in band.threshold))
        if not np.all(np.isfinite(band.threshold)):
            need = math.ceil(1.0 / band.alpha) - 1
            lines.append(
                f"  ^ 'inf' is CORRECT, not a bug: with n_episodes={band.n_episodes} "
                f"the conformal rank ceil((n+1)(1-alpha)) exceeds n, so no finite\n"
                f"    threshold can guarantee {1 - band.alpha:.0%} coverage. You need at "
                f"least {need} success episodes for alpha={band.alpha}; this single-file\n"
                f"    call is a smoke test, not a calibration.")
    return "\n".join(lines)


# =============================================================================
# Self-checks (run: python3 ensemble_analysis.py --self-check)
# =============================================================================


def _self_check() -> int:  # noqa: C901 - a flat list of independent checks
    """Synthetic-data checks with analytically known answers."""
    rng = np.random.default_rng(1234)
    fails: list[str] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    K, H, D = 16, 64, 7
    N_OBS, N_ACT = 2, 8

    # -- 1. variance_profile recovers a known, h-dependent variance -----------
    sigma = np.linspace(0.1, 1.0, H)                      # true std per h
    base = np.cumsum(rng.normal(0, 0.01, size=(H, D)), axis=0)
    big = 4096                                            # large K -> tight estimate
    traj_known = base[None] + sigma[None, :, None] * rng.normal(size=(big, H, D))
    v = variance_profile(traj_known)
    err = np.max(np.abs(v - sigma ** 2) / sigma ** 2)
    ok("variance_profile recovers known sigma(h)^2 (K=4096)", err < 0.12,
       f"max rel err={err:.4f}; v[0]={v[0]:.4f} (true {sigma[0]**2:.4f}), "
       f"v[-1]={v[-1]:.4f} (true {sigma[-1]**2:.4f})")
    ok("variance_profile returns a PROFILE not a scalar", v.shape == (H,), f"shape={v.shape}")
    ok("variance_profile raises for K<2",
       _raises(lambda: variance_profile(np.zeros((1, H, D)))))

    # ddof matters at small K: ddof=0 is biased low by (K-1)/K.
    small = base[None] + sigma[None, :, None] * rng.normal(size=(K, H, D))
    v1 = variance_profile(small, ddof=1).mean()
    v0 = variance_profile(small, ddof=0).mean()
    ok("ddof=0 is biased low by exactly (K-1)/K vs ddof=1",
       abs(v0 / v1 - (K - 1) / K) < 1e-9, f"ratio={v0 / v1:.6f} expected={(K - 1) / K:.6f}")

    # -- 2. executed/tail split matches lerobot's start/end -------------------
    sl = executed_slice(N_OBS, N_ACT, H)
    ok("executed_slice == slice(n_obs-1, n_obs-1+n_act) [modeling_diffusion.py:316-318]",
       (sl.start, sl.stop) == (N_OBS - 1, N_OBS - 1 + N_ACT), f"got {sl}")
    ex, tl = executed_tail_split(small, N_OBS, N_ACT)
    ok("executed/tail shapes partition the post-start horizon",
       ex.shape == (K, N_ACT, D) and tl.shape == (K, H - (N_OBS - 1 + N_ACT), D),
       f"exec={ex.shape} tail={tl.shape}")
    ok("executed slice is the same data lerobot would return",
       np.array_equal(ex, small[:, N_OBS - 1:N_OBS - 1 + N_ACT, :]))
    ok("executed_slice degrades to slice(None) on a file missing the attrs",
       executed_slice(None, None) == slice(None))

    # -- 3. tail variance exceeds exec variance (the horizon fan-out confound) --
    vp = variance_profiles(small, N_OBS, N_ACT)
    s = vp.summary()
    ok("tail variance >> exec variance, so a scalar over all h is dominated by the tail",
       s["arm_tail_mean"] > 3 * s["arm_exec_mean"],
       f"exec={s['arm_exec_mean']:.4f} tail={s['arm_tail_mean']:.4f} "
       f"ratio={s['arm_tail_mean'] / s['arm_exec_mean']:.1f}x")

    # -- 4. arm/gripper: pure timing jitter on a step function ----------------
    # Arm dims: identical across samples (ZERO model uncertainty by construction).
    # Gripper dim: the SAME step, shifted by +-3 timesteps between samples.
    jitter = np.zeros((K, H, D))
    jitter[:, :, :6] = base[None, :, :6]                 # no arm spread at all
    for k in range(K):
        shift = 26 + (k % 7) - 3
        jitter[k, :, 6] = (np.arange(H) >= shift).astype(float)
    vpj = variance_profiles(jitter, N_OBS, N_ACT)
    ok("arm variance is exactly zero when samples agree on the arm",
       float(np.max(vpj.v_arm)) < 1e-24, f"max v_arm={float(np.max(vpj.v_arm)):.3e}")
    ok("gripper timing jitter alone produces large variance (phase, not uncertainty)",
       float(np.max(vpj.v_grip)) > 0.2, f"max v_grip={float(np.max(vpj.v_grip)):.4f}")
    vpj_c = variance_profiles(jitter, N_OBS, N_ACT, combined=True)
    pooled_max = float(np.max(vpj_c.v_combined))
    ok("pooling all 7 dims lets the gripper dim manufacture 'uncertainty' from nothing",
       pooled_max > 0.02 and float(np.max(vpj_c.v_arm)) < 1e-24,
       f"pooled max={pooled_max:.4f} while true arm spread=0")
    ok("v_grip is empty (not an IndexError) on a 6-dim action space",
       variance_profiles(small[:, :, :6], N_OBS, N_ACT).v_grip.size == 0)

    # -- 5. divergence metrics: check the analytic identity -------------------
    # For any cloud, mean squared pairwise distance == 2 * trace(Cov_ddof1).
    iso = rng.normal(size=(K, H, D))
    flat = iso.reshape(K, -1)
    d2 = np.sum((flat[:, None, :] - flat[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(K, k=1)
    mean_d2 = d2[iu].mean()
    trace_cov = variance_profile(iso, reduce="sum").sum()
    ok("mean squared pairwise L2 == 2*trace(Cov): the metrics are second-moment twins",
       abs(mean_d2 / (2 * trace_cov) - 1) < 1e-10,
       f"mean_d2={mean_d2:.3f} 2*trace={2 * trace_cov:.3f}")
    ok("PCA eigenvalues sum to trace(Cov)",
       abs(pca_spectrum(iso).eigenvalues.sum() / trace_cov - 1) < 1e-10,
       f"sum_eig={pca_spectrum(iso).eigenvalues.sum():.3f} trace={trace_cov:.3f}")
    ok("endpoint_spread == sqrt(trace(Cov)) at that timestep",
       abs(endpoint_spread(iso) /
           math.sqrt(np.var(iso[:, -1, :], axis=0, ddof=0).sum()) - 1) < 1e-10)

    # -- 5b. THE BLIND SPOT: matched second moments, different shape ----------
    # Cloud A: two tight modes at +-1. Cloud B: one wide unimodal blob.
    # Both are built to have (nearly) the same variance, so every second-moment
    # metric must agree while the shape probes must not.
    direction = rng.normal(size=(H * D,))
    direction /= np.linalg.norm(direction)
    signs = np.where(np.arange(K) < K // 2, -1.0, 1.0)
    two_mode = (signs[:, None] * direction[None, :] * 1.0
                + rng.normal(scale=0.05, size=(K, H * D)))
    wide = rng.normal(size=(K, H * D)) * 0.0
    wide = np.outer(rng.normal(size=K), direction) + rng.normal(scale=0.05, size=(K, H * D))
    # rescale `wide` to match `two_mode`'s total variance exactly
    tv_a = np.var(two_mode, axis=0, ddof=1).sum()
    tv_b = np.var(wide, axis=0, ddof=1).sum()
    wide *= math.sqrt(tv_a / tv_b)
    A = two_mode.reshape(K, H, D)
    B = wide.reshape(K, H, D)
    va = variance_profile(A, reduce="sum").sum()
    vb = variance_profile(B, reduce="sum").sum()
    ok("constructed 2-mode and 1-blob clouds with MATCHED second moments",
       abs(va / vb - 1) < 1e-9, f"trace A={va:.4f} trace B={vb:.4f}")
    la = float(mean_pairwise_l2(A))
    lb = float(mean_pairwise_l2(B))
    ok("second-moment metrics CANNOT tell them apart (within ~15%)",
       abs(la / lb - 1) < 0.15, f"pairwiseL2 2mode={la:.4f} blob={lb:.4f} "
                                f"ratio={la / lb:.3f}")
    mm_a = multimodality_probe(A, dims=None, n_components=1)
    mm_b = multimodality_probe(B, dims=None, n_components=1)
    print(f"       2-mode probe: method={mm_a.dip_method} dip={mm_a.dip_stats} "
          f"p={mm_a.dip_pvals} dBIC={[round(x, 1) for x in mm_a.bic_deltas]}")
    print(f"       1-blob probe: method={mm_b.dip_method} dip={mm_b.dip_stats} "
          f"p={mm_b.dip_pvals} dBIC={[round(x, 1) for x in mm_b.bic_deltas]}")
    ok("shape probe DOES tell them apart: 2-mode flagged, blob not",
       mm_a.flagged and not mm_b.flagged,
       f"2mode flagged={mm_a.flagged} ({mm_a.flag_reason}) | blob flagged={mm_b.flagged}")
    ok("2-mode BIC prefers k=2 and blob BIC does not",
       mm_a.bic_deltas[0] > 6 > mm_b.bic_deltas[0],
       f"dBIC 2mode={mm_a.bic_deltas[0]:.1f} blob={mm_b.bic_deltas[0]:.1f}")
    # NOTE: the naive assertion "top1_ratio > 0.9 for a 2-mode cloud" is FALSE and
    # was removed after it failed. With H*D=448 feature dims the isotropic noise
    # contributes 448*0.05^2 = 1.12 of total variance vs 1.0 from the mode
    # separation, so PC1 tops out near 0.5 even though the modes are blatant.
    # The correct, and much weaker, claim is relative to the uniform share.
    rank = mm_a.pca.eigenvalues.size
    uniform_share = 1.0 / rank
    print(f"       2-mode PCA: top1={mm_a.pca.top1_ratio():.3f} "
          f"(uniform share 1/{rank}={uniform_share:.3f}, "
          f"ratio {mm_a.pca.top1_ratio() / uniform_share:.1f}x) "
          f"PR={mm_a.pca.participation_ratio:.2f}")
    ok("PC1 carries far more than the uniform share (spectrum points, does not detect)",
       mm_a.pca.top1_ratio() > 3 * uniform_share and mm_a.pca.top1_ratio() < 0.9,
       "top1_ratio is NOT a multimodality detector -- noise dims dilute it")
    ok("but the dip test ON PC1 fires cleanly, so PC1 did find the mode direction",
       mm_a.dip_pvals[0] is not None and mm_a.dip_pvals[0] < 0.01,
       f"dip p on PC1 = {mm_a.dip_pvals[0]}")

    # -- 6a. the fallback path works when `diptest` is unavailable ------------
    proj_2 = mm_a.pca.scores[:, 0]
    proj_1 = mm_b.pca.scores[:, 0]
    bc2, bc1 = bimodality_coefficient(proj_2), bimodality_coefficient(proj_1)
    ok("fallback bimodality_coefficient separates 2-mode from unimodal",
       bc2 > 5 / 9 > bc1, f"BC 2mode={bc2:.3f} unimodal={bc1:.3f} (threshold 0.556)")
    ok("bimodality_coefficient returns NaN for n<=3 instead of crashing",
       math.isnan(bimodality_coefficient(np.array([1.0, 2.0]))))
    ok("dip_test reports which method it used", dip_test(proj_2)[2] in
       ("diptest", "bimodality_coefficient"), f"method={dip_test(proj_2)[2]}")

    # -- 6b. the K=16 power limitation, MEASURED as a curve -------------------
    # An earlier version of this check used a single 4.6-sigma separation, got
    # 100% power, and "failed" the docstring's low-power warning. The separation
    # was simply too easy. Power is a function of separation, so measure the
    # curve -- these numbers are the ones quoted in dip_test's docstring.
    n_trials = 400
    powers: dict[float, float] = {}
    meth = "n/a"
    for sep_sigma in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        hits = 0
        for t in range(n_trials):
            r2 = np.random.default_rng(int(sep_sigma * 1000) + t)
            x = np.concatenate([r2.normal(-sep_sigma / 2, 0.5, 8),
                                r2.normal(+sep_sigma / 2, 0.5, 8)])
            _st, pv, meth = dip_test(x)
            if pv is not None and pv < 0.05:
                hits += 1
        powers[sep_sigma] = hits / n_trials
    print(f"       measured dip power at K=16 [{meth}, {n_trials} trials/point]: "
          + "  ".join(f"{s:.1f}s={p:.2f}" for s, p in powers.items()))
    fp_hits = sum(
        1 for t in range(500)
        if (lambda pv: pv is not None and pv < 0.05)(
            dip_test(np.random.default_rng(50000 + t).normal(size=16))[1])
    )
    print(f"       measured false-positive rate under a true N(0,1), K=16: "
          f"{fp_hits / 500:.3f} (nominal 0.05 -- the dip is conservative)")
    ok("dip power at K=16 collapses for moderately separated modes, as documented",
       powers[2.0] < 0.5 and powers[1.5] < 0.25,
       f"2.0s={powers[2.0]:.2f}, 1.5s={powers[1.5]:.2f} -- a non-flag is NOT "
       "evidence of unimodality")
    ok("dip power is monotone in separation and only reliable at >=3 sigma",
       all(powers[a] <= powers[b] + 0.02 for a, b in
           zip([1.0, 1.5, 2.0, 2.5, 3.0], [1.5, 2.0, 2.5, 3.0, 4.0]))
       and powers[3.0] > 0.7,
       f"3.0s={powers[3.0]:.2f}, 4.0s={powers[4.0]:.2f}")
    ok("dip false-positive rate is at or below nominal (flags are specific)",
       fp_hits / 500 < 0.06, f"fp={fp_hits / 500:.3f}")

    # -- 7. MMD / STAC -------------------------------------------------------
    same_a = rng.normal(size=(K, 20, 6))
    same_b = rng.normal(size=(K, 20, 6))
    shifted = rng.normal(size=(K, 20, 6)) + 2.0
    g = 1.0 / (20 * 6)  # fixed gamma so the two numbers are comparable
    m_same = mmd2_unbiased(same_a, same_b, gamma=g, n_permutations=200)
    m_diff = mmd2_unbiased(same_a, shifted, gamma=g, n_permutations=200)
    ok("MMD^2 ~ 0 for two samples from the same distribution",
       abs(m_same["mmd2"]) < 0.05, f"mmd2={m_same['mmd2']:.5f} p={m_same['p_value']:.3f}")
    ok("MMD^2 large for a shifted distribution",
       m_diff["mmd2"] > 10 * abs(m_same["mmd2"]),
       f"mmd2={m_diff['mmd2']:.5f} p={m_diff['p_value']:.3f}")
    ok("MMD permutation test: high p under H0, low p under H1",
       m_same["p_value"] > 0.1 and m_diff["p_value"] < 0.05,
       f"p_same={m_same['p_value']:.3f} p_diff={m_diff['p_value']:.3f}")

    # STAC: a consistent policy re-plans the same future -> low MMD.
    ramp = np.cumsum(rng.normal(0, 0.05, size=(H + 16, D)), axis=0)
    prev = ramp[None, 0:H, :] + rng.normal(scale=0.02, size=(K, H, D))
    nxt_ok = ramp[None, N_ACT:N_ACT + H, :] + rng.normal(scale=0.02, size=(K, H, D))
    nxt_bad = nxt_ok + 1.5
    r_ok = stac_temporal_divergence(prev, nxt_ok, lag=N_ACT, gamma=1e-2)
    r_bad = stac_temporal_divergence(prev, nxt_bad, lag=N_ACT, gamma=1e-2)
    ok("STAC overlap length == horizon - lag",
       r_ok["overlap_len"] == H - N_ACT, f"overlap={r_ok['overlap_len']}")
    ok("STAC MMD low when consecutive refills agree, high when they disagree",
       r_bad["mmd2"] > 5 * abs(r_ok["mmd2"]),
       f"consistent={r_ok['mmd2']:.5f} inconsistent={r_bad['mmd2']:.5f}")
    ok("STAC returns NaN/0-overlap rather than crashing when lag >= horizon",
       stac_temporal_divergence(prev, nxt_ok, lag=H + 5)["overlap_len"] == 0.0)

    # -- 8. conformal calibration: measured coverage on held-out successes ----
    # 40 calibration + 400 test SUCCESS episodes from the same nominal process,
    # with a deliberately NON-STATIONARY score profile (high in the middle).
    def make_ep(r: np.random.Generator, length: int) -> np.ndarray:
        ph = (np.arange(length) + 0.5) / length
        shape = 0.2 + 1.5 * np.exp(-0.5 * ((ph - 0.5) / 0.12) ** 2)
        return shape * r.lognormal(0.0, 0.35, size=length)

    rc = np.random.default_rng(7)
    cal = [make_ep(rc, int(rc.integers(28, 45))) for _ in range(40)]
    band = conformal_calibrate(cal, alpha=0.1, n_bins=10)
    ok("conformal band is time-varying and tracks the non-stationary profile",
       band.threshold[5] > 3 * band.threshold[0],
       f"thr[mid]={band.threshold[5]:.3f} thr[start]={band.threshold[0]:.3f}")
    test = [make_ep(rc, int(rc.integers(28, 45))) for _ in range(400)]
    flat_band = conformal_calibrate(cal, alpha=0.1, n_bins=10, time_varying=False)
    pt_alarm = np.mean(np.concatenate([band.score_alarm(e) for e in test]))
    pt_flat = np.mean(np.concatenate([flat_band.score_alarm(e) for e in test]))

    # MARGINAL coverage: both are fine, and the global threshold is actually
    # slightly better. An earlier version of this check asserted the opposite and
    # failed -- correctly. The pooled quantile trivially achieves the pooled rate.
    print(f"       MARGINAL per-timestep exceedance on held-out successes: "
          f"band={pt_alarm:.3f}  global={pt_flat:.3f}  (target alpha=0.10)")
    ok("time-varying band keeps marginal false alarms in the right ballpark",
       0.02 < pt_alarm < 0.25, f"exceedance={pt_alarm:.3f} target alpha=0.10")

    # CONDITIONAL coverage: bin held-out timesteps by normalized phase and get
    # the exceedance rate WITHIN each phase bin. This is where the global
    # threshold falls apart.
    def per_bin_rate(b: ConformalBand, nb: int = 10) -> np.ndarray:
        num = np.zeros(nb)
        den = np.zeros(nb)
        for e in test:
            al = b.score_alarm(e)
            ph = (np.arange(e.size) + 0.5) / e.size
            idx = np.minimum((ph * nb).astype(int), nb - 1)
            np.add.at(num, idx, al.astype(float))
            np.add.at(den, idx, 1.0)
        return num / np.maximum(den, 1)

    r_band, r_flat = per_bin_rate(band), per_bin_rate(flat_band)
    print("       CONDITIONAL exceedance per phase bin (this is what matters):")
    print("         band  : " + " ".join(f"{x:.2f}" for x in r_band))
    print("         global: " + " ".join(f"{x:.2f}" for x in r_flat))
    print(f"         spread (max-min): band={np.ptp(r_band):.3f}  "
          f"global={np.ptp(r_flat):.3f}")
    # Assert on the STRUCTURE of the failure, not on a ptp ratio: the global
    # threshold is completely BLIND (rate ~0) over most of the episode and
    # saturates in the high-divergence phase. ptp alone was too noisy a statistic
    # (it measured 2.85x against a 3x threshold and failed while the qualitative
    # conclusion was plainly right).
    blind_flat = int(np.sum(r_flat < 0.02))
    blind_band = int(np.sum(r_band < 0.02))
    ok("global threshold has BAD conditional coverage: blind over most of the "
       "episode, saturated in the high-divergence phase",
       blind_flat >= 4 and r_flat.max() > 0.35 and blind_band == 0,
       f"global: {blind_flat}/10 bins with ~0 alarm rate, peak "
       f"{r_flat.max():.2f}; band: {blind_band}/10 blind bins, range "
       f"{r_band.min():.2f}..{r_band.max():.2f}")
    ok("time-varying band's conditional coverage stays near alpha in every phase bin",
       r_band.max() < 0.35, f"worst bin={r_band.max():.3f}")
    ok("_conformal_quantile returns +inf when n is too small for the guarantee",
       math.isinf(_conformal_quantile(np.arange(5.0), alpha=0.1)),
       "n=5, alpha=0.1 -> rank 6 > 5")
    ok("conformal_calibrate tolerates zero episodes",
       conformal_calibrate([]).n_episodes == 0)

    # -- 1b. loader tolerance for a pre-provenance file -----------------------
    if h5py is not None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "old.h5")
            with h5py.File(p, "w") as f:
                grp = f.create_group(GROUP_NAME)
                # deliberately NO norm_stats, and no k/horizon/... attrs either
                grp.attrs["t0_wall"] = 1.0
                # N=1 successful refill: (N, K, horizon, action_dim)
                grp.create_dataset("trajectories",
                                   data=small[None].astype(np.float32))
                grp.create_dataset("committed_chunk",
                                   data=np.zeros((1, N_ACT, D), np.float32))
                grp.create_dataset("obs_state", data=np.zeros((1, N_OBS, 7), np.float32))
                mg = grp.create_group("meta")
                mg.attrs["columns"] = json.dumps(META_COLUMNS_DEFAULT)
                for c in META_COLUMNS_DEFAULT:
                    mg.create_dataset(c, data=np.zeros(3, np.float64))
                mg["dropped_flag"][1] = 1.0
                mg["refill_idx"][:] = [0.0, 1.0, 2.0]
            ef = load(p)
            ok("load() on a pre-provenance file gives norm_stats=None, not a crash",
               ef.norm_stats is None)
            ok("load() back-fills k/horizon/action_dim from array shapes",
               (ef.k, ef.horizon, ef.action_dim) == (K, H, D),
               f"got {(ef.k, ef.horizon, ef.action_dim)}")
            ok("meta_rows_for_nd skips the dropped refill",
               ef.meta_rows_for_nd().tolist() == [0], f"{ef.meta_rows_for_nd().tolist()}")
            ok("n_events/n_dropped read from meta", (ef.n_events, ef.n_dropped) == (3, 1),
               f"{(ef.n_events, ef.n_dropped)}")

            ok("report() runs end-to-end on a real file",
               "variance profile" in report(p))

            # stac_series must use ONE shared gamma for all pairs, else the
            # returned series is not internally comparable.
            p4 = os.path.join(td, "series.h5")
            rs = np.random.default_rng(3)
            multi = np.stack([
                base[None] + 0.05 * (1 + j) * rs.normal(size=(K, H, D))
                for j in range(4)
            ]).astype(np.float32)
            with h5py.File(p4, "w") as f:
                g4 = f.create_group(GROUP_NAME)
                for kk, vv in dict(k=K, horizon=H, action_dim=D,
                                   n_action_steps=N_ACT, n_obs_steps=N_OBS).items():
                    g4.attrs[kk] = vv
                g4.create_dataset("trajectories", data=multi)
                g4.create_dataset("committed_chunk",
                                  data=np.zeros((4, N_ACT, D), np.float32))
                g4.create_dataset("obs_state", data=np.zeros((4, N_OBS, 7), np.float32))
                m4 = g4.create_group("meta")
                m4.attrs["columns"] = json.dumps(META_COLUMNS_DEFAULT)
                for c in META_COLUMNS_DEFAULT:
                    m4.create_dataset(
                        c, data=(np.arange(4, dtype=np.float64)
                                 if c == "refill_idx" else np.zeros(4)))
            ef4 = load(p4)
            series = stac_series(ef4)
            gammas = {round(r.gamma, 12) for r in series}
            ok("stac_series uses ONE shared gamma across all pairs (comparability)",
               len(series) == 3 and len(gammas) == 1,
               f"{len(series)} pairs, {len(gammas)} distinct gamma(s)={gammas}")
            ok("stac_series derives lag from refill_idx * n_action_steps",
               all(r.lag == N_ACT for r in series), f"lags={[r.lag for r in series]}")

            # A malformed file with a 3-D `trajectories` (missing the N axis) must
            # not crash the loader -- the earlier fixture had this bug and it took
            # two check failures to notice, so it is now covered on purpose.
            p3 = os.path.join(td, "malformed.h5")
            with h5py.File(p3, "w") as f:
                g3 = f.create_group(GROUP_NAME)
                g3.attrs["k"] = K
                g3.attrs["horizon"] = H
                g3.attrs["action_dim"] = D
                g3.create_dataset("trajectories", data=small.astype(np.float32))
            ef3 = load(p3)
            ok("load() survives a malformed 3-D trajectories dataset",
               ef3.n_refills == 0 and ef3.k == K and ef3.horizon == H,
               f"n_refills={ef3.n_refills} (shape-based back-fill declined, "
               f"attrs used instead: k={ef3.k})")

            p2 = os.path.join(td, "empty.h5")
            with h5py.File(p2, "w") as f:
                f.create_group(GROUP_NAME)
            ef2 = load(p2)
            ok("load() on an empty group returns 0 refills instead of raising",
               ef2.n_refills == 0 and ef2.norm_stats is None)
            ok("refill_scores on an empty file returns an empty array",
               refill_scores(ef2).size == 0)
    else:
        print("[SKIP] h5py not importable; loader checks skipped")

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED: {fails}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


def main(argv: Sequence[str]) -> int:
    args = list(argv[1:])
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "--self-check":
        return _self_check()
    path = args[0]
    if not os.path.exists(path):
        print(f"no such file: {path}", file=sys.stderr)
        return 2
    print(report(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
