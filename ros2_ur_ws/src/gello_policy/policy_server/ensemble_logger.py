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

Flush policy: the file is flushed every FLUSH_EVERY successful log_refill calls
(and on close), so a crash loses at most a few refills.

Thread-safety: log_refill is called from the sampler worker thread on success and
from the server (submit) thread on drops; a lock serializes all h5py access.
"""

from __future__ import annotations

import json
import os
import threading
import time

import h5py
import numpy as np

GROUP_NAME = "ensemble_trajectories"
META_COLUMNS = ["t_rel_s", "t_wall", "refill_idx", "ensemble_ms", "dropped_flag"]
FLUSH_EVERY = 8  # flush to disk every N log_refill calls (plus on close)


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
    ) -> None:
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
