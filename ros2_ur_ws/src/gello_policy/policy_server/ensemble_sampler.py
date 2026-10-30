#!/usr/bin/env python3
"""Background batch-K ensemble sampler for offline uncertainty research.

Runs inside diffusion_server.py's py3.12 process. At each diffusion net-refill the
server hands this module the ALREADY-COMPUTED global conditioning tensor for the
real observation (captured from the real policy's own _prepare_global_conditioning
-- vision features are NOT recomputed) and this module draws K independent
noise-seeded action trajectories from the SAME observation, entirely off the real
robot-control path:

  * SEPARATE noise scheduler: DDIMScheduler.set_timesteps()/step() mutate instance
    state, so sharing policy.diffusion.noise_scheduler between the real
    select_action() loop and this background loop would silently corrupt BOTH.
    This module builds its OWN scheduler instance via the same lerobot factory
    (_make_noise_scheduler) with the same config kwargs modeling_diffusion.py's
    DiffusionModel.__init__ uses, and never touches the policy's instance.
  * DEDICATED CUDA stream: all GPU work runs under `torch.cuda.stream(self._stream)`
    so its kernels never share the default stream used by the real select_action().
  * SINGLE background worker + DROP (never queue): a one-worker ThreadPoolExecutor;
    submit() drops the job (counts + logs once) if the previous one is still
    running, so ensemble work can never pile up behind the real control loop.
  * NO file I/O here: on completion the (K, horizon, action_dim) float32 CPU tensor
    plus its meta dict is handed to a caller-supplied callback (ensemble_logger.py
    on the server); GPU/sampling concerns stay separate from HDF5 concerns.

The unet weights are shared with the real policy (read-only in eval mode -- safe
for concurrent forwards); only the mutable scheduler state is duplicated.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import torch

# Same factory + param-introspection helpers modeling_diffusion.py itself uses, so
# the private scheduler is constructed EXACTLY like policy.diffusion.noise_scheduler.
from lerobot.policies.diffusion.modeling_diffusion import _make_noise_scheduler
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
)


class EnsembleSampler:
    """Draw K independent conditional samples per refill on a private CUDA stream."""

    def __init__(self, policy, k: int, callback) -> None:
        """
        Args:
            policy: the loaded DiffusionPolicy (read for config + diffusion.unet).
            k: ensemble size (the deploy spec fixes this at 16).
            callback: callable(trajectories_or_none, meta: dict) invoked with the
                (k, horizon, action_dim) float32 CPU tensor on success, or None when
                the job was dropped. meta gains "dropped" (bool) and "elapsed_ms".
                Called from the worker thread on success, from the submitting thread
                on a drop -- the callback must be thread-safe (ensemble_logger is).
        """
        self.k = int(k)
        self._callback = callback
        self._policy = policy
        self._unet = policy.diffusion.unet

        cfg = policy.config
        self._horizon = int(cfg.horizon)
        self._action_dim = int(cfg.action_feature.shape[0])
        # Mirror DiffusionModel.__init__: honor the (possibly DDIM-overridden)
        # inference step count actually resolved at load time.
        self._num_inference_steps = int(policy.diffusion.num_inference_steps)

        # PRIVATE scheduler instance -- same class/config as the real policy's, but
        # never the same object (set_timesteps/step mutate shared state).
        self._scheduler = _make_noise_scheduler(
            cfg.noise_scheduler_type,
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range,
            prediction_type=cfg.prediction_type,
        )
        assert self._scheduler is not policy.diffusion.noise_scheduler

        self._device = get_device_from_parameters(policy.diffusion)
        self._dtype = get_dtype_from_parameters(policy.diffusion)
        self._stream = (
            torch.cuda.Stream() if self._device.type == "cuda" else None
        )

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ensemble_sampler"
        )
        self._inflight = None  # Future of the currently-running job, if any
        self.dropped = 0
        self._warned_drop = False
        self._closed = False

    # ------------------------------------------------------------------ submit
    def submit(self, global_cond: torch.Tensor, meta: dict) -> None:
        """Schedule one batch-K ensemble job for the given (1, global_cond_dim)
        conditioning tensor. If the previous job is still running, DROP this one
        (increment the counter, log once, and report the drop to the callback so
        the logger records a dropped meta row). Never queues a second job."""
        if self._closed:
            return
        if self._inflight is not None and not self._inflight.done():
            self.dropped += 1
            if not self._warned_drop:
                print(
                    "[ensemble_sampler] WARNING: previous ensemble job still "
                    "running at a new refill tick -- dropping this tick's job "
                    "(further drops counted silently).",
                    flush=True,
                )
                self._warned_drop = True
            meta = dict(meta)
            meta["dropped"] = True
            meta["elapsed_ms"] = 0.0
            self._safe_callback(None, meta)
            return
        self._inflight = self._executor.submit(self._job, global_cond, dict(meta))

    # ---------------------------------------------------------------- worker
    def _job(self, global_cond: torch.Tensor, meta: dict) -> None:
        """Worker-thread body: sample, time, deliver. Never raises (a crash here
        must not take out the executor thread or perturb the server)."""
        try:
            t0 = time.perf_counter()
            trajectories = self.run_batch(global_cond)
            meta["dropped"] = False
            meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
            self._safe_callback(trajectories, meta)
        except Exception:  # noqa: BLE001 - research side-channel must never crash the server
            print(
                "[ensemble_sampler] ERROR in background ensemble job "
                "(ignored, robot path unaffected):\n" + traceback.format_exc(),
                flush=True,
            )

    @torch.no_grad()
    def run_batch(self, global_cond: torch.Tensor) -> torch.Tensor:
        """Synchronously draw K conditional samples on the private stream.

        Mirrors modeling_diffusion.py DiffusionModel.conditional_sample exactly,
        with two swaps: the PRIVATE scheduler instance instead of
        policy.diffusion.noise_scheduler, and all GPU work on the private stream.
        Returns a (k, horizon, action_dim) float32 CPU tensor in the NORMALIZED
        action space (same space conditional_sample emits before unnormalize).

        Public so the pre-arm benchmark (benchmark_ensemble_batch16.py) can run the
        exact production sampling loop on its own thread.
        """
        if self._stream is not None:
            # The conditioning tensor was produced by kernels enqueued on the
            # default stream (the real select_action call). Make the private
            # stream wait for that work before consuming it.
            self._stream.wait_stream(torch.cuda.current_stream())
            stream_ctx = torch.cuda.stream(self._stream)
        else:
            import contextlib

            stream_ctx = contextlib.nullcontext()

        with stream_ctx:
            global_cond_k = global_cond.expand(self.k, -1)

            # Sample prior (conditional_sample's noise=None branch).
            sample = torch.randn(
                size=(self.k, self._horizon, self._action_dim),
                dtype=self._dtype,
                device=self._device,
            )

            self._scheduler.set_timesteps(self._num_inference_steps)
            for t in self._scheduler.timesteps:
                model_output = self._unet(
                    sample,
                    torch.full(
                        sample.shape[:1], t, dtype=torch.long, device=sample.device
                    ),
                    global_cond=global_cond_k,
                )
                sample = self._scheduler.step(model_output, t, sample).prev_sample

            out = sample.to("cpu", torch.float32)

        if self._stream is not None:
            self._stream.synchronize()
        return out

    # ----------------------------------------------------------------- misc
    def _safe_callback(self, trajectories, meta: dict) -> None:
        try:
            self._callback(trajectories, meta)
        except Exception:  # noqa: BLE001
            print(
                "[ensemble_sampler] ERROR in result callback (ignored):\n"
                + traceback.format_exc(),
                flush=True,
            )

    def close(self) -> None:
        """Shut the worker down. Only runs at process exit (server shutdown path);
        an in-flight job finishes (~1 s) before shutdown returns. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=True)
        except Exception:  # noqa: BLE001 - best-effort on shutdown
            pass
        if self.dropped:
            print(
                f"[ensemble_sampler] closed; {self.dropped} refill tick(s) were "
                "dropped because the previous ensemble job was still running.",
                flush=True,
            )
