#!/usr/bin/env python3
"""HDF5 logger for per-refill ensemble samples (offline uncertainty research).

py3.12 side, deliberately dependency-free of gello_recorder (that package lives on
the py3.10/ROS side and is not installed in act_venv): only h5py + numpy + stdlib.
The growable-column pattern below mirrors gello_recorder/hdf5_writer.py
(shape=(0,), maxshape=(None,), float64, chunks=True, columns listed in a JSON group
attribute) without importing it.

One file per diffusion_server.py process lifetime:
    {ensemble_dir}/ensemble_<YYYYmmdd_HHMMSS>.h5
The wall-clock t_wall column is the join key to the gello_recorder take files
(vectors.h5 / MP4s), joined at the take/episode level by the analysis GUI.

Layout (all growable along axis 0):
    /ensemble_trajectories/meta/{t_rel_s,t_wall,refill_idx,ensemble_ms,dropped_flag}
        one float64 row per refill event (dropped or not)
    /ensemble_trajectories/trajectories   (N, k, horizon, action_dim)   float32
    /ensemble_trajectories/committed_chunk(N, n_action_steps, action_dim) float32
    /ensemble_trajectories/obs_state      (N, n_obs_steps, state_dim)   float32
        rows appended ONLY for non-dropped refills, so their common row index runs
        over successful refills; align to meta via the refill_idx column (gaps at
        dropped ticks are expected).

SELF-DESCRIBING PROVENANCE (added; all optional, absent -> attrs simply omitted)

Every number in trajectories/committed_chunk/obs_state lives in lerobot's
NORMALIZED space. Without the normalization stats those arrays cannot be mapped
back to radians, so the file used to be uninterpretable the moment the checkpoint
moved or changed. Both of the following make it self-contained:

  /ensemble_trajectories attrs (provenance, written only when supplied):
      checkpoint_path, checkpoint_hash, lerobot_version, torch_version,
      num_inference_steps, noise_scheduler_type
    checkpoint_hash is computed by `hash_checkpoint()`: sha256 over the raw bytes
    of config.json plus, for each *.safetensors sorted by name, the tuple
    (name, size_bytes, mtime_ns). Deterministic and O(config size) -- deliberately
    NOT a content hash of the weights, which would cost seconds of disk I/O at
    server start. It detects "different checkpoint" and "weights rewritten", not
    bit-identical duplicates copied with preserved mtimes.

  /ensemble_trajectories/norm_stats  (group, a few hundred bytes)
      one subgroup per normalized feature, each holding the raw stat arrays as
      float64 datasets, e.g.
          norm_stats/action/{mean,std}          (or {min,max}, {q01,q99}, ...)
          norm_stats/observation.state/{mean,std}
      Feature keys containing '/' are sanitized to '|' in the group name; the
      original key is kept in the JSON below.
      norm_stats.attrs["json"] = {
          "features": {orig_key: {"type": ..., "shape": [...]}},
          "norm_map": {FEATURE_TYPE: NORMALIZATION_MODE},   # MEAN_STD / MIN_MAX /
                                                            # QUANTILES / ...
          "mode_by_key": {orig_key: NORMALIZATION_MODE},    # resolved per feature
          "group_names": {orig_key: sanitized_group_name},
      }
    so an analysis script can unnormalize with no checkpoint and no lerobot import.

Flush policy: the file is flushed every FLUSH_EVERY successful log_refill calls
(and on close), so a crash loses at most a few refills.

Thread-safety: log_refill is called from the sampler worker thread on success and
from the server (submit) thread on drops; a lock serializes all h5py access.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import h5py
import numpy as np

GROUP_NAME = "ensemble_trajectories"
META_COLUMNS = ["t_rel_s", "t_wall", "refill_idx", "ensemble_ms", "dropped_flag"]
FLUSH_EVERY = 8  # flush to disk every N log_refill calls (plus on close)

# Provenance attrs written (when present) on the top-level group. Order fixed so
# files are diffable; int-valued keys are stored as int64, the rest as strings.
PROVENANCE_KEYS = (
    "checkpoint_path",
    "checkpoint_hash",
    "lerobot_version",
    "torch_version",
    "num_inference_steps",
    "noise_scheduler_type",
)
_PROVENANCE_INT_KEYS = frozenset({"num_inference_steps"})


# ---------------------------------------------------------------- provenance helpers
def hash_checkpoint(checkpoint_path: str) -> str:
    """Cheap, deterministic identity hash of a lerobot checkpoint directory.

    sha256( config.json bytes || for each *.safetensors sorted by name:
            name + size_bytes + mtime_ns ).
    Chosen over hashing the weights themselves because the weights are hundreds of
    MB and this runs during server startup; size+mtime is enough to notice that the
    checkpoint was retrained/overwritten, while config.json bytes capture every
    architectural/scheduler knob. Returns "" if the path is unreadable -- provenance
    must never be able to stop the server from starting.
    """
    try:
        h = hashlib.sha256()
        cfg = os.path.join(checkpoint_path, "config.json")
        if os.path.isfile(cfg):
            with open(cfg, "rb") as f:
                h.update(f.read())
        for root, _dirs, files in sorted(os.walk(checkpoint_path)):
            for name in sorted(files):
                if not name.endswith(".safetensors"):
                    continue
                p = os.path.join(root, name)
                st = os.stat(p)
                rel = os.path.relpath(p, checkpoint_path)
                h.update(f"{rel}:{st.st_size}:{st.st_mtime_ns}".encode())
        return h.hexdigest()
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return ""


def _enum_value(x) -> str:
    """FeatureType.STATE / NormalizationMode.MEAN_STD (str-Enums) -> 'STATE'."""
    return str(getattr(x, "name", None) or getattr(x, "value", None) or x)


def collect_normalization(processor) -> dict | None:
    """Extract {'stats','features','norm_map'} from a lerobot processor pipeline.

    Walks `processor.steps` for the first step exposing both `stats` and `norm_map`
    (NormalizerProcessorStep). Returns None if nothing matches or anything raises --
    the logger then simply omits the norm_stats group.
    """
    try:
        for step in getattr(processor, "steps", []) or []:
            stats = getattr(step, "stats", None)
            norm_map = getattr(step, "norm_map", None)
            if not stats or norm_map is None:
                continue
            features = {}
            for key, ft in (getattr(step, "features", None) or {}).items():
                features[str(key)] = {
                    "type": _enum_value(getattr(ft, "type", "")),
                    "shape": [int(s) for s in getattr(ft, "shape", ()) or ()],
                }
            return {
                "stats": stats,
                "features": features,
                "norm_map": {
                    _enum_value(ft_type): _enum_value(mode)
                    for ft_type, mode in norm_map.items()
                },
            }
    except Exception:  # noqa: BLE001
        return None
    return None


class EnsembleLogger:
    """Append-only HDF5 writer for ensemble refill events. One file per process."""

    def __init__(
        self,
        ensemble_dir: str,
        k: int,
        horizon: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        state_dim: int,
        provenance: dict | None = None,
        normalization: dict | None = None,
    ) -> None:
        """provenance: optional subset of PROVENANCE_KEYS; unknown keys ignored.
        normalization: optional dict as returned by collect_normalization(); the
        stats arrays are embedded so the file needs no checkpoint to be read.
        Both default to None -> the file is byte-for-byte the old layout."""
        self._h5 = None  # set last; close() must tolerate partial construction
        self._lock = threading.Lock()
        self._calls = 0

        os.makedirs(ensemble_dir, exist_ok=True)
        self.t0 = time.time()  # server computes t_rel_s = t_wall - logger.t0
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.t0))
        self.path = os.path.join(ensemble_dir, f"ensemble_{stamp}.h5")

        h5 = h5py.File(self.path, "w")
        grp = h5.create_group(GROUP_NAME)
        grp.attrs["k"] = int(k)
        grp.attrs["horizon"] = int(horizon)
        grp.attrs["action_dim"] = int(action_dim)
        grp.attrs["n_action_steps"] = int(n_action_steps)
        grp.attrs["n_obs_steps"] = int(n_obs_steps)
        grp.attrs["state_dim"] = int(state_dim)
        grp.attrs["t0_wall"] = float(self.t0)

        # Provenance + embedded stats are best-effort: a malformed dict must never
        # prevent the ensemble file (or the server) from coming up.
        self._write_provenance(grp, provenance)
        self._write_norm_stats(grp, normalization)

        # meta: one growable float64 1-D dataset per column (hdf5_writer.py pattern).
        meta = grp.create_group("meta")
        meta.attrs["columns"] = json.dumps(META_COLUMNS)
        self._meta_dsets = [
            meta.create_dataset(
                col, shape=(0,), maxshape=(None,), dtype="float64", chunks=True
            )
            for col in META_COLUMNS
        ]
        self._meta_rows = 0

        def _nd(name, row_shape):
            return grp.create_dataset(
                name,
                shape=(0, *row_shape),
                maxshape=(None, *row_shape),
                dtype="float32",
                chunks=(1, *row_shape),
            )

        self._traj = _nd("trajectories", (int(k), int(horizon), int(action_dim)))
        self._chunk = _nd("committed_chunk", (int(n_action_steps), int(action_dim)))
        self._obs = _nd("obs_state", (int(n_obs_steps), int(state_dim)))
        self._nd_rows = 0

        self._h5 = h5
        print(f"[ensemble_logger] logging ensembles to {self.path}", flush=True)

    # -------------------------------------------------------- provenance (init only)
    @staticmethod
    def _write_provenance(grp, provenance: dict | None) -> None:
        if not provenance:
            return
        for key in PROVENANCE_KEYS:
            if key not in provenance:
                continue
            value = provenance[key]
            if value is None:
                continue
            try:
                if key in _PROVENANCE_INT_KEYS:
                    grp.attrs[key] = int(value)
                else:
                    grp.attrs[key] = str(value)
            except Exception:  # noqa: BLE001 - skip the bad key, keep the rest
                continue

    @staticmethod
    def _write_norm_stats(grp, normalization: dict | None) -> None:
        """Embed the raw normalization stat arrays + a JSON description of which
        normalization mode applies to which feature."""
        if not normalization:
            return
        try:
            stats = normalization.get("stats") or {}
            features = normalization.get("features") or {}
            norm_map = normalization.get("norm_map") or {}
            if not stats:
                return

            ns = grp.create_group("norm_stats")
            group_names: dict[str, str] = {}
            mode_by_key: dict[str, str] = {}
            for key, stat_dict in stats.items():
                key = str(key)
                gname = key.replace("/", "|")
                sub = ns.create_group(gname)
                wrote = False
                for stat_name, arr in (stat_dict or {}).items():
                    try:
                        # torch tensors, numpy arrays and lists all land here.
                        value = np.asarray(
                            arr.detach().cpu().numpy() if hasattr(arr, "detach") else arr,
                            dtype=np.float64,
                        )
                        sub.create_dataset(str(stat_name), data=value)
                        wrote = True
                    except Exception:  # noqa: BLE001 - skip one stat, keep the rest
                        continue
                if not wrote:
                    continue
                group_names[key] = gname
                ft_type = (features.get(key) or {}).get("type")
                if ft_type in norm_map:
                    mode_by_key[key] = norm_map[ft_type]

            ns.attrs["json"] = json.dumps(
                {
                    "features": features,
                    "norm_map": norm_map,
                    "mode_by_key": mode_by_key,
                    "group_names": group_names,
                },
                sort_keys=True,
            )
        except Exception:  # noqa: BLE001 - never block the logger on provenance
            pass

    # ------------------------------------------------------------------ write
    def log_refill(
        self,
        t_rel_s: float,
        t_wall: float,
        refill_idx: int,
        ensemble_ms: float,
        dropped: bool,
        trajectories_or_none=None,
        committed_chunk_or_none=None,
        obs_state_or_none=None,
    ) -> None:
        """Append one meta row (always); append the three ND rows only when the job
        was not dropped and its arrays were provided. No-op after close()."""
        with self._lock:
            if self._h5 is None:
                return

            row = [
                float(t_rel_s),
                float(t_wall),
                float(refill_idx),
                float(ensemble_ms),
                1.0 if dropped else 0.0,
            ]
            new_len = self._meta_rows + 1
            for dset, value in zip(self._meta_dsets, row):
                dset.resize((new_len,))
                dset[self._meta_rows] = value
            self._meta_rows = new_len

            if not dropped and trajectories_or_none is not None:
                n = self._nd_rows + 1
                for dset, arr in (
                    (self._traj, trajectories_or_none),
                    (self._chunk, committed_chunk_or_none),
                    (self._obs, obs_state_or_none),
                ):
                    if arr is None:
                        continue  # defensive; expected all-or-none per spec
                    dset.resize((n, *dset.shape[1:]))
                    dset[n - 1] = np.asarray(arr, dtype=np.float32)
                self._nd_rows = n

            self._calls += 1
            if self._calls % FLUSH_EVERY == 0:
                try:
                    self._h5.flush()
                except Exception:  # noqa: BLE001 - never crash the caller
                    pass

    # ------------------------------------------------------------------ close
    def close(self) -> None:
        """Flush + close. Safe to call multiple times and on a partially
        constructed instance (mirrors gello_recorder RecordingSession.close)."""
        lock = getattr(self, "_lock", None)
        if lock is None:  # constructor failed before the lock existed
            return
        with lock:
            h5 = getattr(self, "_h5", None)
            if h5 is None:
                return
            self._h5 = None
            try:
                h5.flush()
            except Exception:  # noqa: BLE001
                pass
            try:
                h5.close()
            except Exception:  # noqa: BLE001
                pass
            print(
                f"[ensemble_logger] closed {getattr(self, 'path', '?')} "
                f"({self._meta_rows} refill row(s), {self._nd_rows} ensemble(s)).",
                flush=True,
            )
