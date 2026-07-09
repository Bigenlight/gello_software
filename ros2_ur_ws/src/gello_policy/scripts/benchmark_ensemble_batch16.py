#!/usr/bin/env python3
"""Contention probe for the batch-16 ensemble side-channel — RUN THIS ON THE ROBOT
PC BEFORE EVER ENABLING DIFFUSION_ENSEMBLE_K ON THE REAL ARM.

WHY: diffusion_server.py can optionally sample a 16-trajectory ensemble per net
refill on a background thread + private CUDA stream (ensemble_sampler.py), purely
for offline uncertainty research. The safety argument is that this background
batch-16 sampling CANNOT delay the real single-sample select_action() calls the
robot depends on. This script measures exactly that on THIS machine's GPU:

  1. BASELINE: N isolated real-style single-sample DDIM-N full-refill
     select_action() calls, p50/p99 (should roughly match
     benchmark_diffusion_latency.py's finding, ~194 ms p99 on the deploy GPU).
  2. CONTENDED: the SAME N timed calls while an EnsembleSampler batch-16 job
     (the exact production run_batch code path: private scheduler, private CUDA
     stream, shared unet) runs continuously on a background thread. The contended
     p99 is the load-bearing number.
  3. Peak GPU memory over the whole run (torch.cuda.max_memory_allocated).
  4. PASS/FAIL: contended p99 must stay under 0.5 s — margin under the real
     act_timeout_s=0.6 s in config/diffusion_deploy.yaml.

IT NEVER MOVES THE ROBOT. No ZMQ, no ROS, no arm — pure GPU/software, safe to run
standalone any time, same as benchmark_diffusion_latency.py.

Usage:
    python benchmark_ensemble_batch16.py --checkpoint <ckpt_dir>
    python benchmark_ensemble_batch16.py --checkpoint <ckpt_dir> \
        --device cuda --num-inference-steps 10 --scheduler DDIM \
        --n-action-steps 32 --iters 50
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time

import numpy as np
import torch

# lerobot 0.6.1 API (mirrors diffusion_server.py's load recipe):
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

# The REAL production sampler (policy_server/ensemble_sampler.py) — benchmark the
# exact code path the server will run, not a re-implementation.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PKG_DIR, "policy_server"))
from ensemble_sampler import EnsembleSampler  # noqa: E402

# Obs keys / shapes: identical to the deployed server (BUILD SPEC §4).
OBS_STATE_KEY = "observation.state"
OBS_CAM1_KEY = "observation.images.cam1"
OBS_CAM2_KEY = "observation.images.cam2"
OBS_TASK_KEY = "task"
STATE_DIM = 7
IMG_SHAPE = (3, 360, 640)  # CHW, float32 [0,1], matches decode_jpeg_to_rgb_float_chw

# PASS/FAIL threshold (s): contended p99 must leave margin under the real
# act_timeout_s=0.6 configured in config/diffusion_deploy.yaml.
CONTENDED_P99_BUDGET_S = 0.5


def _percentile(values, q):
    """q in [0,100]. numpy percentile (linear interpolation)."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def resolve_device(device: str) -> str:
    # Match diffusion_server.py: do NOT silently fall back to CPU. On CPU the numbers
    # here would be meaningless for the real (GPU) deploy.
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "[benchmark] ERROR: --device cuda but CUDA is unavailable. Refusing to "
            "benchmark on CPU (the numbers would not reflect the real GPU deploy). "
            "Fix CUDA, or pass --device cpu deliberately for an offline check.",
            file=sys.stderr, flush=True,
        )
        raise SystemExit(3)
    return device


def load_policy(checkpoint, device, num_inference_steps, scheduler, n_action_steps):
    """Load the diffusion policy + processors EXACTLY as diffusion_server.py does."""
    print(f"[benchmark] loading checkpoint: {checkpoint} on {device}", flush=True)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    if scheduler != "asis":
        cfg.noise_scheduler_type = scheduler       # "DDIM"
        cfg.num_inference_steps = num_inference_steps  # e.g. 10
    cfg.pretrained_backbone_weights = None

    policy = DiffusionPolicy.from_pretrained(checkpoint, config=cfg)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    policy.config.n_action_steps = n_action_steps
    policy.reset()

    print(
        f"[benchmark] n_obs_steps={getattr(policy.config, 'n_obs_steps', '?')} "
        f"horizon={getattr(policy.config, 'horizon', '?')} "
        f"n_action_steps={policy.config.n_action_steps} "
        f"num_inference_steps={num_inference_steps} scheduler={scheduler}",
        flush=True,
    )
    return policy, preprocessor, postprocessor


def make_dummy_obs():
    """Zero observation with the EXACT deployed keys/shapes (never touches a camera)."""
    return {
        OBS_STATE_KEY: torch.zeros(STATE_DIM, dtype=torch.float32),
        OBS_CAM1_KEY: torch.zeros(*IMG_SHAPE, dtype=torch.float32),
        OBS_CAM2_KEY: torch.zeros(*IMG_SHAPE, dtype=torch.float32),
        OBS_TASK_KEY: "",
    }


@torch.no_grad()
def timed_refill(policy, preprocessor, postprocessor, obs, device):
    """One FULL-refill inference: reset() empties the queue so select_action runs the
    entire DDIM sampling loop. Times only pre -> select_action -> post."""
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    proc = preprocessor(obs)
    action = policy.select_action(proc)
    postprocessor(action)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0  # ms


@torch.no_grad()
def capture_global_cond(policy, preprocessor, obs):
    """Produce one real (1, global_cond_dim) conditioning tensor the same way the
    server does: run a select_action so the obs queues are populated, then feed the
    stacked queue batch through _prepare_global_conditioning (exactly what
    generate_actions does internally)."""
    policy.reset()
    preprocessor.reset()
    proc = preprocessor(obs)
    policy.select_action(proc)
    batch = {
        OBS_STATE: torch.stack(list(policy._queues[OBS_STATE]), dim=1),
        OBS_IMAGES: torch.stack(list(policy._queues[OBS_IMAGES]), dim=1),
    }
    global_cond = policy.diffusion._prepare_global_conditioning(batch)
    policy.reset()
    return global_cond


def run_phase(name, policy, preprocessor, postprocessor, obs, device, iters):
    print(f"[benchmark] {name}: timing {iters} full refills...", flush=True)
    samples_ms = []
    for i in range(iters):
        dt = timed_refill(policy, preprocessor, postprocessor, obs, device)
        samples_ms.append(dt)
        print(f"[benchmark]   {name} refill {i + 1:>3}/{iters}: {dt:8.1f} ms",
              flush=True)
    return samples_ms


def summarize(name, samples_ms):
    p50 = _percentile(samples_ms, 50)
    p99 = _percentile(samples_ms, 99)
    print(f"    {name:<10}: min {min(samples_ms):7.1f}  "
          f"mean {statistics.mean(samples_ms):7.1f}  "
          f"p50 {p50:7.1f}  p95 {_percentile(samples_ms, 95):7.1f}  "
          f"p99 {p99:7.1f}  (ms)")
    return p50, p99


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to the pretrained_model dir (config.json + "
                        "model.safetensors + policy_pre/postprocessor jsons at root).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num-inference-steps", type=int, default=10,
                   help="DDIM denoising steps (same knob as the deployed server).")
    p.add_argument("--scheduler", default="DDIM", choices=["DDIM", "DDPM", "asis"])
    p.add_argument("--n-action-steps", type=int, default=32,
                   help="Receding horizon, kept faithful to the deployed server.")
    p.add_argument("--iters", type=int, default=50,
                   help="Timed full-refill inferences per phase (N).")
    p.add_argument("--ensemble-k", type=int, default=16,
                   help="Ensemble batch size (deploy spec fixes this at 16).")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, device, args.num_inference_steps,
        args.scheduler, args.n_action_steps,
    )
    obs = make_dummy_obs()

    # Warmup: compile CUDA kernels / lazy init for BOTH the batch-1 real path and
    # the batch-16 ensemble path, so autotuning doesn't pollute either phase.
    print("[benchmark] warming up (3 single refills + 2 batch-16 ensembles)...",
          flush=True)
    for _ in range(3):
        timed_refill(policy, preprocessor, postprocessor, obs, device)
    global_cond = capture_global_cond(policy, preprocessor, obs)
    sampler = EnsembleSampler(policy, k=args.ensemble_k,
                              callback=lambda traj, meta: None)
    for _ in range(2):
        sampler.run_batch(global_cond)

    iters = max(1, args.iters)

    # --- Phase 1: BASELINE (no background load) ------------------------------
    baseline_ms = run_phase("baseline", policy, preprocessor, postprocessor,
                            obs, device, iters)

    # --- Phase 2: CONTENDED (continuous batch-16 sampling in the background) --
    # The background thread runs the sampler's REAL run_batch (private scheduler +
    # private CUDA stream, shared unet) back-to-back while the main thread times
    # the same N real-style single-sample refills.
    stop = threading.Event()
    bg_stats = {"batches": 0, "total_ms": 0.0}

    def _background_ensemble():
        while not stop.is_set():
            t0 = time.perf_counter()
            sampler.run_batch(global_cond)
            bg_stats["batches"] += 1
            bg_stats["total_ms"] += (time.perf_counter() - t0) * 1000.0

    bg_thread = threading.Thread(target=_background_ensemble,
                                 name="ensemble_bg", daemon=True)
    bg_thread.start()
    try:
        contended_ms = run_phase("contended", policy, preprocessor, postprocessor,
                                 obs, device, iters)
    finally:
        stop.set()
        bg_thread.join(timeout=30.0)
    sampler.close()

    # --- Report ---------------------------------------------------------------
    print("")
    print("=" * 72)
    print(f"  Ensemble batch-{args.ensemble_k} contention probe  (device={device}, "
          f"scheduler={args.scheduler}, DDIM steps={args.num_inference_steps})")
    print(f"    iters/phase : {iters}")
    _, base_p99 = summarize("baseline", baseline_ms)
    _, cont_p99 = summarize("contended", contended_ms)
    if bg_stats["batches"]:
        print(f"    background  : {bg_stats['batches']} batch-{args.ensemble_k} "
              f"ensembles, mean {bg_stats['total_ms'] / bg_stats['batches']:.1f} ms "
              f"each (ran concurrently with the contended phase)")
    else:
        print("    background  : WARNING — 0 ensembles completed; contended phase "
              "did not actually contend. Treat the verdict as INVALID.")
    if device == "cuda":
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"    peak GPU mem: {peak_gib:.2f} GiB "
              f"(torch.cuda.max_memory_allocated over the whole run)")
    print("-" * 72)

    budget_ms = CONTENDED_P99_BUDGET_S * 1000.0
    slowdown = cont_p99 / base_p99 if base_p99 > 0 else float("inf")
    if bg_stats["batches"] and cont_p99 <= budget_ms:
        print(f"  VERDICT: PASS. Contended p99 {cont_p99:.0f} ms <= {budget_ms:.0f} ms "
              f"budget (baseline p99 {base_p99:.0f} ms, x{slowdown:.2f} slowdown).")
        print(f"           Safe margin under act_timeout_s=0.6 s "
              f"(config/diffusion_deploy.yaml). OK to enable "
              f"DIFFUSION_ENSEMBLE_K={args.ensemble_k} on the arm.")
        rc = 0
    else:
        print(f"  VERDICT: FAIL. Contended p99 {cont_p99:.0f} ms EXCEEDS the "
              f"{budget_ms:.0f} ms budget (baseline p99 {base_p99:.0f} ms, "
              f"x{slowdown:.2f} slowdown)." if bg_stats["batches"] else
              "  VERDICT: FAIL. Background ensemble never ran — result invalid.")
        print(f"           Do NOT enable DIFFUSION_ENSEMBLE_K on the real arm on "
              f"this machine. Background batch-{args.ensemble_k} sampling delays "
              f"the real inference too much.")
        rc = 1
    print("=" * 72)
    print("  (This probe never moved the robot.)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
