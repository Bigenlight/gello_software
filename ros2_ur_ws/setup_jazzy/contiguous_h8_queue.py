"""Opt-in contiguous-H8 serving queue for the real-policy adapters.

The native vision server historically expands eight predicted knots into a
24-frame stride-3 queue.  The common lighted campaign is trained on contiguous
``action[i:i+8]`` rows, so its deployment queue must emit those eight rows
directly and replan after eight ACT requests.  This module installs that
contract on an already imported server module; it never edits the native
``action_queue.py`` on disk.  Legacy processes which do not call
``install_contiguous_h8`` retain the historical queue.
"""
from __future__ import annotations

from typing import Any

import numpy as np

H = 8
STRIDE = 1
N_JOINTS = 6
ACT_DIM = 7
CHUNK_DIM = H * ACT_DIM
N_INTERP = H
N_HOLD = 0
QUEUE_LEN = H


def queue_metadata() -> dict[str, int]:
    """Return truthful wire metadata for the contiguous-H8 queue."""
    return {"H": H, "stride": STRIDE, "interp": N_INTERP, "hold": N_HOLD}


def _direct_knots_to_frames(knots: np.ndarray) -> np.ndarray:
    k = np.asarray(knots, dtype=np.float64)
    if k.shape != (H, ACT_DIM):
        raise ValueError(f"knots must be ({H}, {ACT_DIM}), got {k.shape}")
    # Explicit copy prevents callers from mutating the queue's stored rows.
    return k.copy()


def install_contiguous_h8(server: Any) -> type:
    """Install the queue contract onto an imported native server module.

    ``server`` may be the native server module (with ``server.aq``) or the
    native ``server.aq`` module/namespace itself.  The module's queue
    constants/functions are process-local attributes, while the source
    file and any other legacy process remain untouched.  Return the installed
    queue class so adapters can use it in diagnostics/tests.
    """
    aq = getattr(server, "aq", server)
    if not hasattr(aq, "ActionQueue"):
        raise TypeError("expected native server module or its server.aq action queue module")

    base = aq.ActionQueue

    # Idempotence matters when an adapter installs the helper before and after
    # its common stats hook.  Keep the first class as the base in either case.
    if getattr(base, "_contiguous_h8", False):
        return base

    class ContiguousH8ActionQueue(base):
        _contiguous_h8 = True

        def refill(self, chunk56_n, joint_reference=None):
            knots = self.norm.denormalize(aq.chunk_to_knots(chunk56_n))
            if joint_reference is not None:
                ref = np.asarray(joint_reference, dtype=np.float64).reshape(N_JOINTS)
                # Preserve the native nearest-2*pi branch handling.
                knots[:, :N_JOINTS] += 2.0 * np.pi * np.round(
                    (ref[None, :] - knots[:, :N_JOINTS]) / (2.0 * np.pi)
                )
            frames = _direct_knots_to_frames(knots)
            if not np.all(np.isfinite(frames)):
                raise ValueError("non-finite contiguous-H8 frames")
            self._q.clear()
            self._q.extend(frames)
            self.chunk_id = self.n_refill
            self.n_refill += 1
            self.chunk_step = -1
            return knots

    # PolicyServer constructs and reports through server.aq, so install all
    # process-local constants/functions it consults.  No file/global source is
    # changed, and callers which omit this function continue using stride 3.
    aq.H = H
    aq.STRIDE = STRIDE
    aq.N_JOINTS = N_JOINTS
    aq.ACT_DIM = ACT_DIM
    aq.CHUNK_DIM = CHUNK_DIM
    aq.N_INTERP = N_INTERP
    aq.N_HOLD = N_HOLD
    aq.QUEUE_LEN = QUEUE_LEN
    # Keep the native module's public interpolation tables truthful too; a
    # caller reaching for them directly must not observe the old 24x8 table.
    aq._W = np.eye(H, dtype=np.float64)
    aq._GRIP_IDX = np.arange(H, dtype=np.int64)
    aq.knots_to_frames = _direct_knots_to_frames

    def chunk_to_frames(chunk56_n, norm):
        return _direct_knots_to_frames(norm.denormalize(aq.chunk_to_knots(chunk56_n)))

    aq.chunk_to_frames = chunk_to_frames
    aq.ActionQueue = ContiguousH8ActionQueue
    return ContiguousH8ActionQueue


__all__ = [
    "H", "STRIDE", "N_JOINTS", "ACT_DIM", "CHUNK_DIM", "N_INTERP",
    "N_HOLD", "QUEUE_LEN", "queue_metadata", "install_contiguous_h8",
]
