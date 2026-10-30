"""Opt-in contiguous-H serving queue (generalization of contiguous_h8_queue.py).

Horizon ablation models (E: H=5, F: H=10) are trained on contiguous
``action[i:i+H]`` rows with next observation ``i+H``.  Their deployment queue
must emit those H rows directly (stride 1, no interpolation, no hold) and
replan after H ACT requests.  ``install_contiguous_h(server, H)`` installs that
contract on an already imported native server module, exactly like
``contiguous_h8_queue.install_contiguous_h8`` does for H=8, but parameterized.
The native ``action_queue.py`` on disk and the H8 helper are never edited.
"""
from __future__ import annotations

from typing import Any

import numpy as np

STRIDE = 1
N_JOINTS = 6
ACT_DIM = 7
N_HOLD = 0
SUPPORTED_H = (5, 8, 10)


def queue_metadata(H: int) -> dict[str, int]:
    """Truthful wire metadata for a contiguous-H queue."""
    H = int(H)
    return {"H": H, "stride": STRIDE, "interp": H, "hold": N_HOLD}


def install_contiguous_h(server: Any, H: int) -> type:
    """Install a contiguous-H queue contract onto an imported native server module.

    ``server`` may be the native server module (with ``server.aq``) or the native
    ``server.aq`` action queue module itself.  All process-local constants the
    PolicyServer consults (H, CHUNK_DIM, QUEUE_LEN, ...) are replaced so the
    RESET reply and the chunk validation are truthful for this horizon.
    """
    H = int(H)
    if H not in SUPPORTED_H:
        raise ValueError(f"unsupported contiguous horizon {H}; expected one of {SUPPORTED_H}")
    aq = getattr(server, "aq", server)
    if not hasattr(aq, "ActionQueue"):
        raise TypeError("expected native server module or its server.aq action queue module")

    base = aq.ActionQueue
    installed = getattr(base, "_contiguous_h", None)
    if installed is not None:
        if installed != H:
            raise RuntimeError(f"contiguous queue already installed with H={installed}, refusing H={H}")
        return base
    if getattr(base, "_contiguous_h8", False):
        if H != 8:
            raise RuntimeError("contiguous_h8_queue already installed; refusing a different horizon")
        return base

    chunk_dim = H * ACT_DIM

    def chunk_to_knots(chunk_n):
        a = np.asarray(chunk_n, dtype=np.float64).reshape(-1)
        if a.shape[0] != chunk_dim:
            raise ValueError(f"chunk must have {chunk_dim} entries, got {a.shape[0]}")
        return a.reshape(H, ACT_DIM)

    def knots_to_chunk(knots):
        k = np.asarray(knots, dtype=np.float64)
        if k.shape != (H, ACT_DIM):
            raise ValueError(f"knots must be ({H}, {ACT_DIM}), got {k.shape}")
        return k.reshape(chunk_dim)

    def knots_to_frames(knots):
        k = np.asarray(knots, dtype=np.float64)
        if k.shape != (H, ACT_DIM):
            raise ValueError(f"knots must be ({H}, {ACT_DIM}), got {k.shape}")
        return k.copy()  # rows are executed directly; copy protects the queue's stored rows

    def denorm_chunk(chunk_n, norm):
        return norm.denormalize(chunk_to_knots(chunk_n))

    def chunk_to_frames(chunk_n, norm):
        return knots_to_frames(denorm_chunk(chunk_n, norm))

    def demo_knots(actions, i):
        a = np.asarray(actions, dtype=np.float64)
        idx = np.minimum(i + STRIDE * np.arange(H), a.shape[0] - 1)
        return a[idx]

    class ContiguousHActionQueue(base):
        _contiguous_h = H

        def refill(self, chunk_n, joint_reference=None):
            knots = self.norm.denormalize(chunk_to_knots(chunk_n))
            if joint_reference is not None:
                ref = np.asarray(joint_reference, dtype=np.float64).reshape(N_JOINTS)
                # Preserve the native nearest-2*pi branch handling.
                knots[:, :N_JOINTS] += 2.0 * np.pi * np.round(
                    (ref[None, :] - knots[:, :N_JOINTS]) / (2.0 * np.pi)
                )
            frames = knots_to_frames(knots)
            if not np.all(np.isfinite(frames)):
                raise ValueError(f"non-finite contiguous-H{H} frames")
            self._q.clear()
            self._q.extend(frames)
            self.chunk_id = self.n_refill
            self.n_refill += 1
            self.chunk_step = -1
            return knots

    aq.H = H
    aq.STRIDE = STRIDE
    aq.N_JOINTS = N_JOINTS
    aq.ACT_DIM = ACT_DIM
    aq.CHUNK_DIM = chunk_dim
    aq.N_INTERP = H
    aq.N_HOLD = N_HOLD
    aq.QUEUE_LEN = H
    aq._W = np.eye(H, dtype=np.float64)
    aq._GRIP_IDX = np.arange(H, dtype=np.int64)
    aq.chunk_to_knots = chunk_to_knots
    aq.knots_to_chunk = knots_to_chunk
    aq.knots_to_frames = knots_to_frames
    aq.denorm_chunk = denorm_chunk
    aq.chunk_to_frames = chunk_to_frames
    aq.demo_knots = demo_knots
    aq.ActionQueue = ContiguousHActionQueue
    return ContiguousHActionQueue


def install_episode_logger_h(server: Any, H: int) -> None:
    """Make the native EpisodeLogger accept 7*H chunks / (H, 7) knots.

    ``episode_logger.py`` on disk hard-codes 56/(8, 7) and is hash-pinned by every
    H8 deployment, so it is patched process-locally (like the queue) instead of
    edited.  Only ``log_refill`` is replaced; the written NPZ keys are unchanged.
    """
    H = int(H)
    if H not in SUPPORTED_H:
        raise ValueError(f"unsupported contiguous horizon {H}; expected one of {SUPPORTED_H}")
    logger_cls = getattr(server, "EpisodeLogger", None)
    if logger_cls is None:
        raise TypeError("expected native server module exposing EpisodeLogger")
    installed = getattr(logger_cls, "_contiguous_h", None)
    if installed is not None:
        if installed != H:
            raise RuntimeError(f"EpisodeLogger already patched with H={installed}, refusing H={H}")
        return
    chunk_dim = H * ACT_DIM

    def log_refill(self, chunk_n, knots, t_index, refill_ms):
        ep = self._ep
        if ep is None:
            return
        ep.chunks.append(np.asarray(chunk_n, dtype=np.float32).reshape(chunk_dim))
        ep.chunks_denorm.append(np.asarray(knots, dtype=np.float32).reshape(H, ACT_DIM))
        ep.chunk_t.append(int(t_index))
        ep.refill_ms.append(float(refill_ms))

    logger_cls.log_refill = log_refill
    logger_cls._contiguous_h = H


__all__ = ["STRIDE", "N_JOINTS", "ACT_DIM", "N_HOLD", "SUPPORTED_H", "queue_metadata",
           "install_contiguous_h", "install_episode_logger_h"]
