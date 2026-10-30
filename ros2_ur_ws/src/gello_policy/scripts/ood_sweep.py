#!/usr/bin/env python3
"""OOD contrast sweep: paired K-sample ensemble variance under controlled perturbations.

The offline experiment that produced §5 of docs/ros2/GELLO_DIFFUSION_ENSEMBLE_OFFLINE.md.
Drives offline_ensemble + ood_perturbations read-only; touches no real-time control path.

Design notes
------------
* PAIRED: one fixed list of (episode, frame) observations is decoded ONCE, cached in
  RAM as post-resize obs dicts, and reused for every condition. So condition-to-
  condition differences cannot be confounded by frame selection or by video decode.
* The per-observation ensemble sampling seed stream is reset to the same base seed at
  the start of every condition, so sample i in condition A and sample i in condition B
  start from the same noise. Differences are then attributable to the conditioning,
  not to the prior draw.
* The perturbation is applied to EVERY frame in the observation window (the n_obs_steps
  warm-up frames as well as the target frame), because on a real robot an OOD condition
  is not a single-frame event.
* Injection point is exactly diffusion_server.py L352-357: after make_obs (resize,
  [0,1] RGB CHW) and before the checkpoint preprocessor.

Nothing in the real-time control path is imported for mutation; offline_ensemble and
ood_perturbations are imported read-only and their functions reused verbatim.

Usage (see the doc's "How to run" section for the full recipe):

    act_venv/bin/python src/gello_policy/scripts/ood_sweep.py \
        --checkpoint src/gello_policy/checkpoints/diffusion_banana_in_pot_joint \
        --dataset Bigenlight/banana_in_pot_lerobot_v3 \
        --out /path/to/offline_runs/ood --k 64

`--dataset` accepts a hub repo id (re-downloads to the HF cache) or a local
LeRobotDataset root. Add `--probe` for a tiny timing-only run first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Resolve the package + scripts dirs from this file's location (no absolute paths).
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_SCRIPTS)
for p in (_PKG, _SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

import offline_ensemble as OE  # noqa: E402
import ood_perturbations as OP  # noqa: E402
from policy_server.ensemble_sampler import EnsembleSampler  # noqa: E402
from policy_server.ensemble_logger import (  # noqa: E402
    GROUP_NAME, EnsembleLogger, collect_normalization, hash_checkpoint,
)

# Defaults: checkpoint ships (gitignored) under the package; dataset defaults to the
# hub repo id so a fresh checkout re-downloads it rather than pointing at stale scratch.
DEFAULT_CKPT = os.path.join(_PKG, "checkpoints", "diffusion_banana_in_pot_joint")
DEFAULT_DATASET = "Bigenlight/banana_in_pot_lerobot_v3"
DEFAULT_REPO_ID = "Bigenlight/banana_in_pot_lerobot_v3"


def build_obs_cache(repo_id, root, episodes, fracs, n_obs_steps, video_backend="pyav"):
    """Decode once. Returns list of dicts: {ep, frame, window: [obs_dict, ...]}.

    Frames are chosen as FRACTIONS of each episode's own length, so every episode
    contributes the same number of windows spanning the same task phases regardless
    of its length (231..817 frames in this dataset). window is ordered oldest ->
    newest; the last element is the target frame.

    root=None loads from the hub cache by repo_id; otherwise root is a local
    LeRobotDataset dir.
    """
    cache = []
    for ep in episodes:
        kw = dict(episodes=[ep], video_backend=video_backend)
        if root:
            kw["root"] = root
        ds = LeRobotDataset(repo_id, **kw)
        ep_len = len(ds)
        frames = [min(ep_len - 1, max(0, int(round(f * (ep_len - 1))))) for f in fracs]
        for fr in frames:
            window = []
            for j in range(max(0, fr - n_obs_steps + 1), fr + 1):
                item = ds[j]
                window.append(OE.make_obs(item, item.get("task", "")))
            cache.append({"ep": int(ep), "frame": int(fr), "window": window})
        del ds
        print(f"  [cache] episode {ep}: ep_len={ep_len} frames={frames}", flush=True)
    return cache


def perturb_window(window, pert, severity, seed):
    if pert is None or severity == 0.0:
        return window
    return [pert(o, severity, seed) for o in window]


@torch.no_grad()
def run_condition(policy, pre, post, sampler, sink, cache, cond_name, pert, severity,
                  out_dir, k, base_seed, device, cfg, provenance_extra):
    os.makedirs(out_dir, exist_ok=True)
    logger = EnsembleLogger(
        out_dir, k=k, horizon=int(cfg.horizon),
        action_dim=int(cfg.action_feature.shape[0]),
        n_action_steps=int(cfg.n_action_steps),
        n_obs_steps=int(cfg.n_obs_steps),
        state_dim=int(cfg.robot_state_feature.shape[0]),
        provenance=provenance_extra,
        normalization=collect_normalization(pre),
    )
    seed_gen = torch.Generator(device="cpu")
    seed_gen.manual_seed(int(base_seed))
    index_rows = []
    t_start = time.perf_counter()

    for n, entry in enumerate(cache):
        window = perturb_window(entry["window"], pert, severity, seed=0)
        policy.reset(); pre.reset(); post.reset()
        for o in window[:-1]:
            OE.push_obs_only(policy, pre(o))
        proc = pre(window[-1])
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        sink.pop("global_cond", None)
        action_norm = policy.select_action(proc)
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        gc = sink.get("global_cond")
        if gc is None:
            raise RuntimeError(f"{cond_name}: global_cond not captured (no refill)")

        committed = torch.cat([action_norm] + list(policy._queues[ACTION]), dim=0
                              ).cpu().numpy().astype(np.float32)
        obs_state = torch.stack(list(policy._queues[OBS_STATE]), dim=1
                                ).squeeze(0).cpu().numpy().astype(np.float32)

        sample_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=seed_gen).item())
        fork_devices = [torch.cuda.current_device()] if device == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(sample_seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(sample_seed)
            traj = sampler.run_batch(gc)
        if device == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        t_wall = time.time()
        logger.log_refill(
            t_rel_s=t_wall - logger.t0, t_wall=t_wall, refill_idx=n,
            ensemble_ms=(t3 - t2) * 1000.0, dropped=False,
            trajectories_or_none=traj.numpy(),
            committed_chunk_or_none=committed,
            obs_state_or_none=obs_state,
        )
        index_rows.append({"row": n, "episode": entry["ep"], "frame": entry["frame"],
                           "seed": sample_seed})

    try:
        grp = logger._h5[GROUP_NAME]
        grp.attrs["offline"] = 1
        grp.attrs["ood_condition"] = cond_name
        grp.attrs["ood_perturbation"] = "control" if pert is None else pert.name
        grp.attrs["ood_severity"] = float(severity)
        grp.attrs["offline_seed"] = int(base_seed)
        grp.attrs["offline_index_json"] = json.dumps(index_rows)
    except Exception as exc:
        print(f"  WARNING: attrs failed: {exc}", flush=True)

    path = logger.path
    logger.close()
    el = time.perf_counter() - t_start
    print(f"  [{cond_name}] {len(index_rows)} refills in {el:.1f}s "
          f"({el / max(1, len(index_rows)):.2f} s/obs) -> {os.path.basename(path)}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="Pretrained diffusion checkpoint dir (default: package checkpoints/).")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="Hub repo id OR a local LeRobotDataset root dir.")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help="Repo id to use when --dataset is a local dir.")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--episodes", default="0,7,14,21,28,35")
    ap.add_argument("--fracs", default="0.1,0.3,0.5,0.7,0.9",
                    help="target frames as fractions of each episode's length")
    ap.add_argument("--out", required=True,
                    help="Output dir; one subdir per condition + manifest.json.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-b", type=int, default=12345)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--scheduler", default="DDIM")
    ap.add_argument("--n-action-steps", type=int, default=32)
    ap.add_argument("--perturbations", default=(
        "gaussian_noise,brightness_down,occlusion_center_black,shift_x,blur,"
        "color_shift_region,state_offset_random"))
    ap.add_argument("--severities", default="0.25,0.5,1.0")
    ap.add_argument("--probe", action="store_true", help="tiny run for timing only")
    args = ap.parse_args()

    device = args.device
    eps = [int(x) for x in args.episodes.split(",") if x]
    fracs = [float(x) for x in args.fracs.split(",") if x]
    sevs = [float(x) for x in args.severities.split(",") if x]
    pnames = [x for x in args.perturbations.split(",") if x]
    if args.probe:
        eps = eps[:1]
        fracs = fracs[:2]
        pnames = pnames[:1]
        sevs = sevs[:1]

    # --dataset may be a hub repo id (root=None -> hub cache) or a local dir.
    ckpt = args.checkpoint
    if os.path.isdir(args.dataset):
        ds_root, repo_id = args.dataset, args.repo_id
    else:
        ds_root, repo_id = None, args.dataset

    policy, pre, post = OE.load_policy(ckpt, device, args.n_action_steps,
                                       args.num_inference_steps, args.scheduler)
    cfg = policy.config
    sink = {}
    OE.install_global_cond_hook(policy, sink)
    sampler = EnsembleSampler(policy, k=args.k, callback=lambda *a, **kw: None)

    print(f"[ood_sweep] caching {len(eps) * len(fracs)} observation windows "
          f"(n_obs_steps={cfg.n_obs_steps}) ...", flush=True)
    tc = time.perf_counter()
    cache = build_obs_cache(repo_id, ds_root, eps, fracs, int(cfg.n_obs_steps))
    print(f"[ood_sweep] cached {len(cache)} windows in {time.perf_counter() - tc:.1f}s", flush=True)

    prov = {
        "checkpoint_path": os.path.abspath(ckpt),
        "checkpoint_hash": hash_checkpoint(ckpt),
        "torch_version": torch.__version__,
        "num_inference_steps": cfg.num_inference_steps,
        "noise_scheduler_type": cfg.noise_scheduler_type,
    }

    conditions = [("control", None, 0.0)]
    for pn in pnames:
        p = OP.PERTURBATIONS[pn]
        for s in sevs:
            conditions.append((f"{pn}__sev{s:g}", p, s))
    # second control with a DIFFERENT sampling seed = the noise floor arm
    conditions.append(("control_b", None, 0.0))

    os.makedirs(args.out, exist_ok=True)
    print(f"[ood_sweep] {len(conditions)} conditions x {len(cache)} obs, K={args.k}", flush=True)
    manifest = {}
    t0 = time.perf_counter()
    for name, pert, sev in conditions:
        seed = args.seed_b if name == "control_b" else args.seed
        d = os.path.join(args.out, name)
        manifest[name] = run_condition(policy, pre, post, sampler, sink, cache, name,
                                       pert, sev, d, args.k, seed, device, cfg, prov)
    total = time.perf_counter() - t0
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"conditions": manifest, "k": args.k, "n_obs": len(cache),
                   "episodes": eps, "fracs": fracs,
                   "obs_index": [{"ep": c["ep"], "frame": c["frame"]} for c in cache],
                   "seed": args.seed, "seed_b": args.seed_b,
                   "scheduler": args.scheduler,
                   "num_inference_steps": args.num_inference_steps}, fh, indent=2)
    print(f"\n[ood_sweep] TOTAL {total / 60:.1f} min for {len(conditions)} conditions")
    if device == "cuda":
        print(f"[ood_sweep] peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
