#!/usr/bin/env python3
"""Offline K-sample ensemble runner over a LeRobotDataset (no robot, no ROS, no ZMQ).

WHY THIS EXISTS
---------------
diffusion_server.py can log ensembles online (--ensemble-k), but that path runs on
the live robot-control process: it is capped at K=16, it DROPS jobs whenever the
previous one is still running, and every millisecond it spends is a millisecond the
arm's control loop could have wanted. For *research* on the predictive distribution
none of that is acceptable and none of it is necessary.

This script reproduces the same measurement offline against recorded data:
  * no robot, no ROS node, no ZMQ socket, no real-time deadline;
  * K is bounded only by VRAM (K=64 default, K=256 is fine on a 12 GB card) --
    see MEMORY below for why;
  * nothing is ever dropped: every sampled observation produces exactly one row;
  * runs are reproducible (--seed), which the online path deliberately is not.

It writes the SAME HDF5 schema as the online path (it imports and drives
ensemble_logger.EnsembleLogger directly), so ensemble_analysis.py loads an offline
file and an online file with the identical call.

FIDELITY: WHY THE CONDITIONING IS PROVABLY THE SAME AS THE ROBOT'S
------------------------------------------------------------------
The whole point of an ensemble is that all K samples share one conditioning vector.
If this script built that vector even slightly differently from the deploy path, the
numbers would describe a policy that never flies. So it does not build it at all --
it lets the *real* policy build it and steals the result, using the exact technique
diffusion_server.py uses:

  * The policy is loaded with the same recipe as diffusion_server.py:118-146
    (mutate PreTrainedConfig BEFORE from_pretrained so the noise scheduler is
    rebuilt from the overridden config; then make_pre_post_processors with the
    device override). Same scheduler/step-count defaults as the deploy scripts, so
    what we characterize is what the arm would actually run.
  * n_action_steps is set BEFORE the first reset(), and the
    n_action_steps <= horizon - n_obs_steps + 1 guard is re-asserted
    (diffusion_server.py:148-171).
  * `_prepare_global_conditioning` is monkey-patched with the SAME
    `_capture_global_cond` wrapper as diffusion_server.py:240-247: the real
    `policy.select_action()` computes the vision features, and the wrapper keeps a
    reference to the tensor `generate_actions` would otherwise discard. The
    ensemble therefore conditions on a tensor that was produced by the production
    code path, not by a re-implementation of it.
  * The obs dict uses the same keys as diffusion_server.py:352-357
    (observation.state / observation.images.cam1 / observation.images.cam2 / task)
    and the same image contract as image_preprocess.py: RGB, CHW, float32 in [0,1],
    resized to 360x640 with a plain torchvision v2.Resize -- see IMAGES below.
  * Sampling is NOT re-implemented. `EnsembleSampler.run_batch()` from
    ensemble_sampler.py:143-192 is called directly; that method is public exactly so
    external callers can run the production sampling loop. ensemble_sampler.py is
    not modified and not even subclassed here.
  * The committed-chunk / obs-state snapshots are taken exactly as
    diffusion_server.maybe_submit_ensemble does (diffusion_server.py:290-298).

IMAGES
------
LeRobotDataset decodes the mp4s to float32 CHW RGB in [0,1] at the stored 720x1280.
Training used a single deterministic `v2.Resize(size=[360,640])` on exactly such a
float tensor, and the saved policy preprocessor does NOT resize -- see the parity
chain in image_preprocess.py:15-22. We therefore apply the same Resize here, built
the same way (image_preprocess.py:39), and reuse RESIZE_HW from that module so the
two can never drift apart. No BGR->RGB is needed: that step exists only because the
robot path receives JPEGs from cv2, which decodes BGR; the dataset is already RGB.

OBSERVATION WINDOWS
-------------------
A refill tick on the robot has (a) a full n_obs_steps observation queue and (b) an
empty action deque. To recreate that at an arbitrary dataset frame i we:
  1. policy.reset() (clears both queues);
  2. push frames [i-n_obs_steps+1 .. i-1] into the obs queues using lerobot's own
     `populate_queues`, replaying modeling_diffusion.select_action:149-153 verbatim
     but WITHOUT the denoise -- calling select_action for these would burn a full
     sampling loop per warm-up frame for a result we throw away. Frames before the
     episode start are simply omitted; populate_queues then pads by repeating the
     first observation, which is precisely what happens on the robot's first ticks
     (populate_queues, lerobot/policies/utils.py:40-43);
  3. call the real `policy.select_action()` on frame i. The action deque is empty,
     so this IS a refill: the hook fires and global_cond is captured.

DETERMINISM
-----------
The online path deliberately does not seed anything: each refill must draw fresh
randomness, and a seeded robot would be a robot whose "uncertainty" is a fixed
lookup table. Offline that argument does not apply and reproducibility is worth
having, so --seed drives a torch.Generator that derives one per-observation seed.
`run_batch` takes no generator argument (it calls torch.randn on the global RNG),
and we do not modify it, so each call is wrapped in torch.random.fork_rng() with
that per-observation seed installed. The global RNG is left untouched outside the
fork, and the derived-seed stream depends only on --seed -- so --limit / resuming /
reordering episodes does not perturb the seeds of the samples that do run.

MEMORY
------
K expansion happens ONLY on global_cond: `global_cond.expand(k, -1)`
(ensemble_sampler.py:168) is a view, and the sampled tensor is (K, 64, 7) floats --
114 KB at K=64. The expensive tensor is the vision encoder's, and that is computed
once per observation at batch size 1 by the real select_action, never at batch K.
So VRAM grows with K only through the U-Net activations, which are 1-D convs over a
64-step horizon. Observations are processed strictly one at a time and the peak
allocation is printed at the end so a larger K can be planned against a real number.

Usage:
    python scripts/offline_ensemble.py \
        --checkpoint checkpoints/diffusion_banana_in_pot_joint \
        --dataset /path/to/banana_in_pot_lerobot_v3 \
        --k 64 --episodes 0,1 --stride 32 --out /path/to/out_dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ACTION / OBS_STATE / OBS_IMAGES: keys of the policy's internal queues
# (modeling_diffusion.DiffusionPolicy.reset:91-100). Read non-mutatingly to snapshot
# what the refill was conditioned on and what it committed to.
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# The SAME queue-filling helper select_action uses (modeling_diffusion.py:153), so
# warm-up frames enter the queues through production code rather than a copy of it.
from lerobot.policies.utils import populate_queues

from torchvision.transforms import v2

# policy_server/ is a sibling of scripts/; import the deploy modules from there so
# this script and the robot path can never drift apart.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
from policy_server.image_preprocess import RESIZE_HW  # noqa: E402
from policy_server.ensemble_sampler import EnsembleSampler  # noqa: E402
from policy_server.ensemble_logger import (  # noqa: E402
    GROUP_NAME,
    EnsembleLogger,
    collect_normalization,
    hash_checkpoint,
)

# Obs keys, mirrored from zmq_protocol via diffusion_server.py:352-357. Hard-coded
# here rather than imported because zmq_protocol pulls in pyzmq, which this offline
# script has no reason to require.
OBS_STATE_KEY = "observation.state"
OBS_CAM1_KEY = "observation.images.cam1"
OBS_CAM2_KEY = "observation.images.cam2"
OBS_TASK_KEY = "task"

# Same construction as image_preprocess.py:39 (v2 defaults: bilinear + antialias).
_RESIZE = v2.Resize(size=RESIZE_HW)


# =============================================================================
# Policy loading (mirrors diffusion_server.DiffusionInferenceEngine.__init__)
# =============================================================================
def load_policy(checkpoint: str, device: str, n_action_steps: int,
                num_inference_steps: int, scheduler: str):
    """Load policy + processors exactly like diffusion_server.py:112-182."""
    print(f"[offline_ensemble] loading checkpoint: {checkpoint} on {device}", flush=True)

    # diffusion_server.py:118-133 -- mutate the config BEFORE from_pretrained so the
    # noise scheduler is rebuilt from it. Without this the trained DDPM/100 scheduler
    # survives silently, which is a different (10x slower, differently-distributed)
    # sampler than the one the arm runs.
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    if scheduler != "asis":
        cfg.noise_scheduler_type = scheduler
        cfg.num_inference_steps = num_inference_steps
    # from_pretrained overwrites the backbone anyway; skipping the ImageNet download
    # removes a network dependency at startup (diffusion_server.py:129-133).
    cfg.pretrained_backbone_weights = None

    policy = DiffusionPolicy.from_pretrained(checkpoint, config=cfg)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # CRITICAL ordering (diffusion_server.py:148-151): reset() sizes the action deque
    # from n_action_steps, so it must be set before the first reset().
    policy.config.n_action_steps = n_action_steps

    horizon = policy.config.horizon
    n_obs_steps = policy.config.n_obs_steps
    # diffusion_server.py:161-171 -- lerobot does not enforce this itself.
    assert n_action_steps <= horizon - n_obs_steps + 1, (
        f"n_action_steps ({n_action_steps}) must be <= horizon - n_obs_steps + 1 "
        f"({horizon} - {n_obs_steps} + 1 = {horizon - n_obs_steps + 1})"
    )
    # diffusion_server.py:173-182 -- the SpatialSoftmax grid is sized from
    # resize_shape at init, so a mismatch with our pre-resize is a silent
    # degradation, not a crash. Fail at load instead.
    rs = getattr(policy.config, "resize_shape", None)
    assert rs is not None and list(rs) == list(RESIZE_HW), (
        f"checkpoint resize_shape={rs} but this script pre-resizes to {RESIZE_HW}"
    )

    print(
        f"[offline_ensemble] n_obs_steps={policy.config.n_obs_steps} "
        f"horizon={policy.config.horizon} "
        f"n_action_steps={policy.config.n_action_steps} "
        f"num_inference_steps={policy.config.num_inference_steps} "
        f"noise_scheduler_type={policy.config.noise_scheduler_type}",
        flush=True,
    )
    return policy, preprocessor, postprocessor


def install_global_cond_hook(policy, sink: dict):
    """Capture the conditioning tensor the REAL refill computes.

    Byte-for-byte the technique at diffusion_server.py:240-247: wrap
    `policy.diffusion._prepare_global_conditioning`, keep a reference to its return
    value (which `generate_actions` discards), and pass it through untouched. The
    ensemble then reuses those vision features instead of recomputing them, which is
    both faster and the only way to *guarantee* the K samples are conditioned on
    exactly what the policy itself was conditioned on.
    """
    _orig_prep = policy.diffusion._prepare_global_conditioning

    def _capture_global_cond(batch):
        gc = _orig_prep(batch)
        sink["global_cond"] = gc
        return gc

    policy.diffusion._prepare_global_conditioning = _capture_global_cond


# =============================================================================
# Dataset plumbing
# =============================================================================
def resolve_dataset_args(dataset: str, repo_id: str | None) -> tuple[str, str | None]:
    """--dataset accepts either a local LeRobotDataset root or a hub repo id."""
    if os.path.isdir(dataset):
        root = os.path.abspath(dataset)
        rid = repo_id or f"local/{os.path.basename(root.rstrip('/'))}"
        return rid, root
    return (repo_id or dataset), None


def make_obs(item: dict, task: str) -> dict:
    """Dataset item -> the obs dict diffusion_server.act() builds (line 352-357).

    Images arrive from LeRobotDataset as float32 CHW RGB in [0,1] at 720x1280; the
    Resize to 360x640 is what image_preprocess.decode_jpeg_to_rgb_float_chw does for
    the robot path. No colour swap: that one is a cv2-decodes-BGR artifact only.
    """
    return {
        OBS_STATE_KEY: item[OBS_STATE_KEY].to(torch.float32),          # (7,)
        OBS_CAM1_KEY: _RESIZE(item[OBS_CAM1_KEY]),                     # (3,360,640)
        OBS_CAM2_KEY: _RESIZE(item[OBS_CAM2_KEY]),                     # (3,360,640)
        OBS_TASK_KEY: task,
    }


@torch.no_grad()
def push_obs_only(policy, proc_batch: dict) -> None:
    """Push one observation into the policy's queues WITHOUT running the denoiser.

    Replays modeling_diffusion.DiffusionPolicy.select_action:145-153 exactly (drop
    ACTION, stack the per-camera images into OBS_IMAGES, then populate_queues) and
    stops right before the `if len(self._queues[ACTION]) == 0` refill branch. Used
    only for the n_obs_steps-1 warm-up frames that precede a sampled frame; the
    sampled frame itself goes through the real select_action().

    The ACTION pop at select_action:146-147 is LOAD-BEARING, not defensive: the
    saved lerobot processor pipeline puts an `action` key (None at inference) into
    every transition it emits. populate_queues iterates over whatever keys the batch
    has, so without the pop it fills the ACTION deque with n_action_steps copies of
    that None -- the subsequent select_action then sees a non-empty deque, never
    refills (so no global_cond is ever computed) and returns None. Verified: with
    the pop removed, the deque is 32 long before the sampled frame's select_action.
    """
    batch = proc_batch
    if ACTION in batch:
        batch = dict(batch)
        batch.pop(ACTION)
    if policy.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack(
            [batch[key] for key in policy.config.image_features], dim=-4
        )
    policy._queues = populate_queues(policy._queues, batch)


# =============================================================================
# Main loop
# =============================================================================
@torch.no_grad()
def run(args) -> int:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[offline_ensemble] ERROR: --device cuda but CUDA is unavailable.",
              file=sys.stderr, flush=True)
        return 3

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy, pre, post = load_policy(
        args.checkpoint, device, args.n_action_steps,
        args.num_inference_steps, args.scheduler,
    )
    cfg = policy.config
    n_obs_steps = int(cfg.n_obs_steps)

    sink: dict = {}
    install_global_cond_hook(policy, sink)

    # Callback is a no-op: we drive run_batch() synchronously and log ourselves.
    # Constructing the sampler is still the right move -- it is what builds the
    # PRIVATE noise scheduler with the same _make_noise_scheduler kwargs the policy
    # used (ensemble_sampler.py:75-85), which is the piece that must not be
    # re-implemented.
    sampler = EnsembleSampler(policy, k=args.k, callback=lambda *_a, **_kw: None)

    try:
        import lerobot
        lerobot_version = getattr(lerobot, "__version__", "")
    except Exception:  # noqa: BLE001
        lerobot_version = ""

    repo_id, root = resolve_dataset_args(args.dataset, args.repo_id)

    os.makedirs(args.out, exist_ok=True)
    logger = EnsembleLogger(
        args.out,
        k=args.k,
        horizon=int(cfg.horizon),
        action_dim=int(cfg.action_feature.shape[0]),
        n_action_steps=int(cfg.n_action_steps),
        n_obs_steps=n_obs_steps,
        state_dim=int(cfg.robot_state_feature.shape[0]),
        provenance={
            "checkpoint_path": os.path.abspath(args.checkpoint),
            "checkpoint_hash": hash_checkpoint(args.checkpoint),
            "lerobot_version": lerobot_version,
            "torch_version": torch.__version__,
            "num_inference_steps": cfg.num_inference_steps,
            "noise_scheduler_type": cfg.noise_scheduler_type,
        },
        normalization=collect_normalization(pre),
    )

    # Which episodes. Default: every episode in the dataset.
    probe = LeRobotDataset(repo_id, root=root, episodes=[0],
                           video_backend=args.video_backend)
    total_episodes = int(probe.meta.total_episodes)
    del probe
    if args.episodes:
        episodes = [int(x) for x in args.episodes.replace(" ", "").split(",") if x != ""]
    else:
        episodes = list(range(total_episodes))
    print(f"[offline_ensemble] dataset repo_id={repo_id} root={root} "
          f"total_episodes={total_episodes}; running {len(episodes)} episode(s): "
          f"{episodes}", flush=True)

    # Per-observation seeds come from one Generator so they depend only on --seed,
    # not on how many samples actually ran (see DETERMINISM in the module docstring).
    seed_gen = torch.Generator(device="cpu")
    seed_gen.manual_seed(int(args.seed))

    index_rows: list[dict] = []   # (episode, frame) provenance for each logged row
    t_obs_s: list[float] = []     # dataset read + resize + preprocess + warm-up
    t_refill_s: list[float] = []  # the real select_action refill
    t_ens_s: list[float] = []     # EnsembleSampler.run_batch (K samples)
    t_start = time.perf_counter()
    n_done = 0
    stop = False

    for ep in episodes:
        if stop:
            break
        ds = LeRobotDataset(repo_id, root=root, episodes=[ep],
                            video_backend=args.video_backend)
        ep_len = len(ds)
        frames = list(range(0, ep_len, max(1, args.stride)))
        print(f"[offline_ensemble] episode {ep}: {ep_len} frames -> "
              f"{len(frames)} sample(s) at stride {args.stride}", flush=True)

        for i in frames:
            if args.limit and n_done >= args.limit:
                stop = True
                break

            t0 = time.perf_counter()
            # Fresh queues, then replay the n_obs_steps-1 preceding frames so the obs
            # window at frame i is what the robot's would have been.
            policy.reset()
            pre.reset()
            post.reset()
            for j in range(max(0, i - n_obs_steps + 1), i):
                push_obs_only(policy, pre(make_obs(ds[j], ds[j].get("task", ""))))

            item = ds[i]
            obs = make_obs(item, item.get("task", ""))
            proc = pre(obs)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Action deque is empty after reset() -> this call IS a refill, so
            # _prepare_global_conditioning runs and the hook fires.
            sink.pop("global_cond", None)
            action_norm = policy.select_action(proc)      # (1, 7), NORMALIZED
            if device == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            global_cond = sink.get("global_cond")
            if global_cond is None:
                raise RuntimeError(
                    "global_cond was not captured -- select_action did not refill. "
                    "The queue/reset contract in this script is broken; refusing to "
                    "log conditioning we cannot vouch for."
                )

            # Snapshots, exactly as diffusion_server.py:288-298.
            committed = torch.cat(
                [action_norm] + list(policy._queues[ACTION]), dim=0
            ).cpu().numpy().astype(np.float32)            # (n_action_steps, 7)
            obs_state = torch.stack(
                list(policy._queues[OBS_STATE]), dim=1
            ).squeeze(0).cpu().numpy().astype(np.float32)  # (n_obs_steps, state_dim)

            # K samples through the PRODUCTION sampling loop (ensemble_sampler.py:143).
            # run_batch() has no generator argument and we do not modify it, so the
            # seed is installed on a forked global RNG state.
            sample_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=seed_gen).item())
            fork_devices = [torch.cuda.current_device()] if device == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(sample_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(sample_seed)
                trajectories = sampler.run_batch(global_cond)   # (k, horizon, 7) cpu
            if device == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            t_wall = time.time()
            logger.log_refill(
                t_rel_s=t_wall - logger.t0,
                t_wall=t_wall,
                refill_idx=n_done,
                ensemble_ms=(t3 - t2) * 1000.0,
                dropped=False,
                trajectories_or_none=trajectories.numpy(),
                committed_chunk_or_none=committed,
                obs_state_or_none=obs_state,
            )
            index_rows.append({
                "row": n_done, "episode": int(ep), "frame": int(i),
                "seed": sample_seed,
            })
            t_obs_s.append(t1 - t0)
            t_refill_s.append(t2 - t1)
            t_ens_s.append(t3 - t2)
            n_done += 1

            if n_done % max(1, args.progress_every) == 0 or n_done == 1:
                print(f"[offline_ensemble] {n_done} sample(s) | ep {ep} frame {i} | "
                      f"obs {(t1 - t0) * 1000:.0f}ms refill {(t2 - t1) * 1000:.0f}ms "
                      f"ensemble(k={args.k}) {(t3 - t2) * 1000:.0f}ms", flush=True)

        del ds

    # --- Additive provenance: which (episode, frame) each row came from ----------
    # EnsembleLogger has no public hook for extra attrs and PROVENANCE_KEYS is a
    # fixed tuple, so this reaches into ._h5 to write ADDITIONAL group attrs. The
    # datasets and the documented attrs are untouched, and ensemble_analysis.load()
    # slurps every attr into a plain dict, so an offline file still loads exactly
    # like an online one -- it just carries more information.
    try:
        grp = logger._h5[GROUP_NAME]  # noqa: SLF001 - see comment above
        grp.attrs["offline"] = 1
        grp.attrs["offline_dataset_repo_id"] = str(repo_id)
        grp.attrs["offline_dataset_root"] = str(root or "")
        grp.attrs["offline_stride"] = int(args.stride)
        grp.attrs["offline_seed"] = int(args.seed)
        grp.attrs["offline_episodes"] = json.dumps(episodes)
        grp.attrs["offline_index_json"] = json.dumps(index_rows)
    except Exception as exc:  # noqa: BLE001 - never lose the data over an attr
        print(f"[offline_ensemble] WARNING: could not write offline attrs: "
              f"{type(exc).__name__}: {exc}", flush=True)

    out_path = logger.path
    logger.close()
    sampler.close()

    total = time.perf_counter() - t_start
    print("\n[offline_ensemble] ===== TIMING SUMMARY =====", flush=True)
    print(f"  samples logged      : {n_done}")
    print(f"  k                   : {args.k}")
    print(f"  scheduler           : {cfg.noise_scheduler_type} "
          f"({cfg.num_inference_steps} inference steps)")
    if n_done:
        def _stat(name, xs):
            xs = np.asarray(xs)
            print(f"  {name:<20}: mean {xs.mean() * 1000:8.1f} ms | "
                  f"median {np.median(xs) * 1000:8.1f} ms | "
                  f"max {xs.max() * 1000:8.1f} ms | total {xs.sum():6.1f} s")
        _stat("obs build + warmup", t_obs_s)
        _stat("refill select_action", t_refill_s)
        _stat(f"ensemble k={args.k}", t_ens_s)
    print(f"  wall clock          : {total:.1f} s "
          f"({total / max(1, n_done):.2f} s/sample)")
    if device == "cuda":
        print(f"  peak GPU allocated  : "
              f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
        print(f"  peak GPU reserved   : "
              f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")
    print(f"  output              : {out_path}", flush=True)
    return 0


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Pretrained model dir (config.json + model.safetensors + "
                        "policy_pre/postprocessor.json).")
    p.add_argument("--dataset", required=True,
                   help="Local LeRobotDataset root dir, or a hub repo id.")
    p.add_argument("--repo-id", default=None,
                   help="Override the repo id when --dataset is a local dir.")
    p.add_argument("--k", type=int, default=64,
                   help="Ensemble size. Offline there is no 16 cap and no drop "
                        "policy; bounded only by VRAM (default: 64).")
    p.add_argument("--episodes", default="",
                   help="Comma-separated episode indices (default: all).")
    p.add_argument("--stride", type=int, default=32,
                   help="Sample every Nth frame within an episode (default: 32, "
                        "= the deploy n_action_steps, i.e. the natural refill "
                        "cadence on the robot).")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N sampled observations (0 = no limit).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out", required=True,
                   help="Output DIRECTORY. EnsembleLogger names the file "
                        "ensemble_<YYYYmmdd_HHMMSS>.h5 inside it, same as online.")
    p.add_argument("--seed", type=int, default=0,
                   help="Reproducibility seed (the online path is deliberately "
                        "unseeded; see the module docstring).")
    p.add_argument("--n-action-steps", type=int, default=32,
                   help="Must match the deploy value for committed_chunk to mean "
                        "the same thing as an online file.")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--scheduler", default="DDIM", choices=["DDIM", "DDPM", "asis"],
                   help="Defaults mirror diffusion_server.py so the offline "
                        "measurement characterizes the deployed sampler.")
    p.add_argument("--video-backend", default="pyav",
                   help="LeRobotDataset video decode backend (default: pyav).")
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
