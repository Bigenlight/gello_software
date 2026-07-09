#!/usr/bin/env python3
"""Standalone latency probe for the Diffusion deploy — RUN THIS ON THE ROBOT PC
BEFORE THE FIRST ARM RUN.

WHY: a diffusion net-refill runs a full DDIM-N denoising loop, which is slower than
ACT's single transformer forward. The ROS safety stack assumes a refill fits inside
the bridge's staleness budget (act_timeout_s=0.6 < staleness_timeout_s=0.8, target
~0.5s). This script measures the REAL full-refill latency on THIS machine's GPU so
you can confirm it fits — or lower DIFFUSION_NUM_INFERENCE_STEPS until it does —
WITHOUT touching the robot.

IT NEVER MOVES THE ROBOT. It talks to no ZMQ socket, no ROS, no arm — it only loads
the policy the SAME way diffusion_server.py does (DDIM-N override) and times
select_action on dummy zero observations. Safe to run any time.

It loads the policy EXACTLY like diffusion_server.py (PreTrainedConfig.from_pretrained
-> noise_scheduler_type=DDIM + num_inference_steps -> DiffusionPolicy.from_pretrained
-> make_pre_post_processors), does M=3 warmup refills, then K timed FULL refills
(policy.reset() before each so the action queue is empty and the net actually re-runs
the whole DDIM sampling loop — this is the worst-case tick the leader must tolerate).

Usage:
    python benchmark_diffusion_latency.py --checkpoint <ckpt_dir>
    python benchmark_diffusion_latency.py --checkpoint <ckpt_dir> \
        --device cuda --num-inference-steps 10 --scheduler DDIM \
        --n-action-steps 32 --iters 30
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
import torch

# lerobot 0.6.1 API (mirrors diffusion_server.py's load recipe):
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors

# Obs keys / shapes: identical to the deployed server (BUILD SPEC §4).
OBS_STATE_KEY = "observation.state"
OBS_CAM1_KEY = "observation.images.cam1"
OBS_CAM2_KEY = "observation.images.cam2"
OBS_TASK_KEY = "task"
STATE_DIM = 7
IMG_SHAPE = (3, 360, 640)  # CHW, float32 [0,1], matches decode_jpeg_to_rgb_float_chw

# Bridge budget (s): a full refill should complete well inside the staleness window.
BRIDGE_BUDGET_S = 0.5
# If p99 exceeds this (s) we recommend cutting DDIM steps (keeps margin under budget).
P99_WARN_S = 0.4


def _percentile(values, q):
    """q in [0,100]. numpy percentile (linear interpolation)."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def resolve_device(device: str) -> str:
    # Match diffusion_server.py: do NOT silently fall back to CPU. On CPU the numbers
    # here would be meaningless for the real (GPU) deploy. Allow --device cpu only if
    # the operator asks for it explicitly (e.g. an offline sanity check).
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
    cfg.num_inference_steps = num_inference_steps   # e.g. 10

    policy = DiffusionPolicy.from_pretrained(checkpoint, config=cfg)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # Set n_action_steps BEFORE reset(): reset() builds the action deque with
    # maxlen=n_action_steps (modeling_diffusion.py), so the net re-runs every
    # n_action_steps calls. We reset() before every timed tick anyway to force a
    # full refill, but keep the config faithful to the deployed server.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to the pretrained_model dir (config.json + "
                        "model.safetensors + policy_pre/postprocessor jsons at root).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num-inference-steps", type=int, default=10,
                   help="DDIM denoising steps (the main latency knob).")
    p.add_argument("--scheduler", default="DDIM", choices=["DDIM", "DDPM", "asis"])
    p.add_argument("--n-action-steps", type=int, default=32,
                   help="Receding horizon (trained default 32). Does not change refill "
                        "latency, kept faithful to the deployed server.")
    p.add_argument("--iters", type=int, default=30,
                   help="Number of timed full-refill inferences (K).")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    policy, preprocessor, postprocessor = load_policy(
        args.checkpoint, device, args.num_inference_steps,
        args.scheduler, args.n_action_steps,
    )
    obs = make_dummy_obs()

    # M=3 warmup refills: compile CUDA kernels / lazy init so they don't pollute timing.
    print("[benchmark] warming up (3 refills, compiling kernels)...", flush=True)
    for _ in range(3):
        timed_refill(policy, preprocessor, postprocessor, obs, device)

    # K timed FULL refills.
    K = max(1, args.iters)
    print(f"[benchmark] timing {K} full DDIM-{args.num_inference_steps} refills "
          f"(scheduler={args.scheduler})...", flush=True)
    samples_ms = []
    for i in range(K):
        dt = timed_refill(policy, preprocessor, postprocessor, obs, device)
        samples_ms.append(dt)
        print(f"[benchmark]   refill {i + 1:>3}/{K}: {dt:8.1f} ms", flush=True)

    mn = min(samples_ms)
    mean = statistics.mean(samples_ms)
    p50 = _percentile(samples_ms, 50)
    p95 = _percentile(samples_ms, 95)
    p99 = _percentile(samples_ms, 99)

    print("")
    print("=" * 68)
    print(f"  Diffusion full-refill latency  (device={device}, "
          f"scheduler={args.scheduler}, DDIM steps={args.num_inference_steps})")
    print(f"    iters   : {K}")
    print(f"    min     : {mn:8.1f} ms")
    print(f"    mean    : {mean:8.1f} ms")
    print(f"    p50     : {p50:8.1f} ms")
    print(f"    p95     : {p95:8.1f} ms")
    print(f"    p99     : {p99:8.1f} ms")
    print("-" * 68)

    budget_ms = BRIDGE_BUDGET_S * 1000.0
    warn_ms = P99_WARN_S * 1000.0
    if p99 <= warn_ms:
        print(f"  VERDICT: OK. p99 {p99:.0f} ms <= {warn_ms:.0f} ms — comfortably "
              f"inside the {budget_ms:.0f} ms bridge budget.")
        print(f"           Keep DIFFUSION_NUM_INFERENCE_STEPS={args.num_inference_steps}.")
        rc = 0
    elif p99 <= budget_ms:
        print(f"  VERDICT: MARGINAL. p99 {p99:.0f} ms is under the {budget_ms:.0f} ms "
              f"budget but over the {warn_ms:.0f} ms safety margin.")
        rec = max(1, int(args.num_inference_steps * (warn_ms / p99)))
        print(f"           Consider reducing DIFFUSION_NUM_INFERENCE_STEPS "
              f"{args.num_inference_steps} -> ~{rec} for headroom.")
        rc = 0
    else:
        # Latency scales roughly linearly with DDIM steps; target the warn threshold.
        rec = max(1, int(args.num_inference_steps * (warn_ms / p99)))
        print(f"  VERDICT: TOO SLOW. p99 {p99:.0f} ms EXCEEDS the {budget_ms:.0f} ms "
              f"bridge budget — refills would risk tripping the staleness watchdog.")
        print(f"           REDUCE DIFFUSION_NUM_INFERENCE_STEPS "
              f"{args.num_inference_steps} -> ~{rec} (latency ~scales with steps) and "
              f"re-run this benchmark. Do NOT widen the ROS timeouts instead.")
        rc = 1
    print("=" * 68)
    print("  (This probe never moved the robot.)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
