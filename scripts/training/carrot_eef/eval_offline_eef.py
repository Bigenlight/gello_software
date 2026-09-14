#!/usr/bin/env python3
"""Open-loop offline evaluation of a carrot_in_pot EEF-delta checkpoint (ACT or FM).

WHAT IT MEASURES
    For every frame of every held-out episode the real logged observation
    (16-D state + cam1 + cam2 + task string) is fed to `policy.select_action`,
    and the predicted 7-D action ``[dx, dy, dz, drx, dry, drz, grip_cmd]``
    (CARROT_EEF_DESIGN.md §2) is compared against the dataset's ground-truth
    action at that timestep. No robot, no simulator, no closed loop: the
    observation stream is always the recorded one, so errors do not compound.

    Because the action is a PER-FRAME DELTA at 30 fps, the numbers are small by
    construction -- the measured |Δp| median is 2.2 mm (DESIGN §1). Read them
    against that scale, not against a joint-space MAE.

    metrics per episode and aggregated over all frames:
        position_mae_m        mean |err| over the 3 translation axes        [m]
        position_err_norm_m   mean ||dp_pred - dp_gt||  (per-frame 3-vector) [m]
        rotation_mae_rad      mean ||drot_pred - drot_gt|| (rotvec difference,
                              i.e. the norm asked for in the A3 brief)       [rad]
        rotation_axis_mae_rad mean |err| over the 3 rotvec axes              [rad]
        gripper_mae           mean |err| on the absolute gripper command     [0..1]
        gripper_accuracy      agreement of (value > 0.5) between pred and gt
        overall_l1            mean |err| over all 7 dims
        per_dim_mae / per_dim_rmse

WHY IT IS POLICY-AGNOSTIC
    The policy class is resolved from the checkpoint's own config via lerobot's
    factory, exactly like policy_server/fm_server.py does:
        cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.device/...
        PolicyCls = get_policy_class(cfg.type); PolicyCls.from_pretrained(ckpt, cfg)
    so `act` -> ACTPolicy and `multi_task_dit` -> MultiTaskDiTPolicy without
    importing either by name, and a future policy type needs no change here.

IMAGE PATH -- must match how the checkpoint was TRAINED
    ACT was trained with a dataset-level Resize to 360x640 (train_act_carrot_eef.sh),
    so this harness applies the identical ImageTransforms.
    The FM (multi_task_dit) policy resizes to config.image_resize_shape (224)
    INTERNALLY and was trained on native frames, so no external transform is applied.
    `--image-resize auto` (default) picks this from cfg.type; override only to debug.

Usage:
    python eval_offline_eef.py \
        --checkpoint ~/workspace/youngwoong/carrot_eef/outputs/fm_carrot_eef/checkpoints/010000/pretrained_model \
        --dataset-root ~/workspace/youngwoong/carrot_eef/dataset/carrot_in_pot_eef_lerobot \
        --out evaluations/fm/010000 --device cuda --num-integration-steps 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

# Caches must land in a writable place before torch/lerobot are imported
# (banana TROUBLESHOOTING §5: kanu's $HOME is shared and nearly full).
_SCRATCH = os.environ.get(
    "SCRATCH_DIR", os.path.expanduser("~/workspace/youngwoong/carrot_eef/.cache")
)
os.environ.setdefault("HF_LEROBOT_HOME", os.path.join(_SCRATCH, "hf_lerobot_home"))
os.environ.setdefault("TORCH_HOME", os.path.join(_SCRATCH, "torch_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.transforms.transforms import (  # noqa: E402
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

DEFAULT_REPO_ID = "Bigenlight/carrot_in_pot_eef_lerobot_v3"
DEFAULT_TASK = "Put carrot in pot"
# Last 6 of 54 episodes == the --dataset.eval_split=0.111 held-out block. Keep in
# sync with the training scripts or the eval leaks training data.
DEFAULT_EPISODES = (48, 49, 50, 51, 52, 53)
ACT_RESIZE_HW = [360, 640]
GRIPPER_THRESHOLD = 0.5
# 7-D EEF-delta layout (DESIGN §2). Used only as a fallback if meta has no names.
FALLBACK_DIM_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "grip_cmd"]
POS_SLICE = slice(0, 3)
ROT_SLICE = slice(3, 6)
GRIP_INDEX = 6
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="a saved pretrained_model dir (config.json + model.safetensors + processor jsons)")
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--episodes", default=",".join(map(str, DEFAULT_EPISODES)),
                   help="comma-separated held-out episode indices (default: the eval_split block 48..53)")
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="fallback task string when a frame carries none (CLIP text conditioning for FM)")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--num-integration-steps", type=int, default=10,
                   help="flow-matching Euler steps at eval; matches the deploy default. Ignored for ACT.")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="receding-horizon chunk length; default = the checkpoint's trained value")
    p.add_argument("--image-resize", default="auto", choices=("auto", "act", "none"),
                   help="auto: 360x640 for ACT, none for multi_task_dit (policy resizes internally)")
    p.add_argument("--video-backend", default="torchcodec", choices=("torchcodec", "pyav"))
    p.add_argument("--max-frames", type=int, default=None, help="cap frames per episode (smoke/debug)")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", required=True, type=Path, help="output dir for metrics.json + episodes.csv")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    """Seed every RNG used by preprocessing and flow-matching ODE sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_image_transforms(resize_hw: list[int]) -> ImageTransforms:
    """Reproduce the training-time transform of train_act_carrot_eef.sh.

    enable=true, max_num_transforms=1 and a single Resize make this deterministic
    (n_subset=1 always picks the one transform).
    """
    cfg = ImageTransformsConfig(
        enable=True,
        max_num_transforms=1,
        random_order=False,
        tfs={"resize": ImageTransformConfig(weight=1.0, type="Resize", kwargs={"size": resize_hw})},
    )
    return ImageTransforms(cfg)


def load_policy(checkpoint: Path, device: str, dataset_stats: dict,
                num_integration_steps: int | None, n_action_steps: int | None):
    """Generic checkpoint load (mirrors policy_server/fm_server.py).

    The config is mutated BEFORE from_pretrained because the ODE integrator and
    the action queue are built at model init from those fields.
    """
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.pretrained_path = str(checkpoint)
    cfg.device = device
    # hasattr-guarded: ACT has no ODE, so the flag is a no-op there rather than a crash.
    if num_integration_steps is not None and hasattr(cfg, "num_integration_steps"):
        cfg.num_integration_steps = num_integration_steps

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(str(checkpoint), config=cfg)
    policy.to(device)
    policy.eval()

    if n_action_steps is not None:
        # Must be set BEFORE reset(): reset() sizes the action deque from it.
        horizon = getattr(policy.config, "horizon", None)
        n_obs = getattr(policy.config, "n_obs_steps", 1)
        if horizon is not None and n_action_steps > horizon - n_obs + 1:
            raise ValueError(
                f"--n-action-steps {n_action_steps} > horizon - n_obs_steps + 1 "
                f"({horizon} - {n_obs} + 1); the queue would starve"
            )
        chunk = getattr(policy.config, "chunk_size", None)
        if chunk is not None and n_action_steps > chunk:
            raise ValueError(f"--n-action-steps {n_action_steps} > chunk_size {chunk}")
        policy.config.n_action_steps = n_action_steps

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, cfg


def metric_dict(prediction: np.ndarray, target: np.ndarray) -> dict:
    """EEF-delta metrics. Layout is fixed by DESIGN §2: [0:3] m, [3:6] rad, [6] grip."""
    error = prediction - target
    absolute = np.abs(error)
    per_dim_mae = absolute.mean(axis=0)
    per_dim_rmse = np.sqrt(np.square(error).mean(axis=0))

    pos_err = error[:, POS_SLICE]
    rot_err = error[:, ROT_SLICE]
    grip_pred = prediction[:, GRIP_INDEX] > GRIPPER_THRESHOLD
    grip_target = target[:, GRIP_INDEX] > GRIPPER_THRESHOLD
    return {
        "frames": int(prediction.shape[0]),
        "position_mae_m": float(np.abs(pos_err).mean()),
        "position_err_norm_m": float(np.linalg.norm(pos_err, axis=1).mean()),
        "position_err_norm_p99_m": float(np.percentile(np.linalg.norm(pos_err, axis=1), 99)),
        # The brief's "rotation MAE (rad, norm of rotvec difference)".
        "rotation_mae_rad": float(np.linalg.norm(rot_err, axis=1).mean()),
        "rotation_axis_mae_rad": float(np.abs(rot_err).mean()),
        "rotation_err_norm_p99_rad": float(np.percentile(np.linalg.norm(rot_err, axis=1), 99)),
        "gripper_mae": float(per_dim_mae[GRIP_INDEX]),
        "gripper_accuracy": float((grip_pred == grip_target).mean()),
        "overall_l1": float(absolute.mean()),
        "per_dim_mae": per_dim_mae.tolist(),
        "per_dim_rmse": per_dim_rmse.tolist(),
    }


def rollout_episode(*, episode: int, dataset_root: Path, repo_id: str, video_backend: str,
                    task: str, policy, preprocessor, postprocessor, camera_keys: list[str],
                    transforms: ImageTransforms | None,
                    max_frames: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Feed each recorded frame of one episode to select_action; return (pred, gt)."""
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=[episode],
        image_transforms=transforms,
        video_backend=video_backend,
    )
    frame_count = len(dataset) if max_frames is None else min(len(dataset), max_frames)

    # Both chunked policies keep an internal action queue; reset it per episode so
    # episode N does not start by replaying the tail of episode N-1.
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for frame_index in range(frame_count):
        frame = dataset[frame_index]
        observation = {OBS_STATE: frame[OBS_STATE], "task": frame.get("task") or task}
        for camera_key in camera_keys:
            observation[camera_key] = frame[camera_key]
        with torch.inference_mode():
            processed = preprocessor(observation)
            action = policy.select_action(processed)
            action = postprocessor(action)
        predictions.append(action.squeeze(0).float().cpu().numpy())
        targets.append(frame[ACTION].float().cpu().numpy())

    return np.asarray(predictions), np.asarray(targets)


def print_table(rows: list[dict], aggregate: dict, dim_names: list[str]) -> None:
    header = (f"{'episode':>8} {'T':>5} {'posMAE_m':>10} {'posNorm_m':>10} "
              f"{'rotMAE_rad':>11} {'gripMAE':>9} {'gripAcc':>8} {'L1':>10}")
    print("\n" + "=" * len(header))
    print("OPEN-LOOP OFFLINE EVAL  (predicted EEF delta vs dataset ground truth)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['episode']:>8} {row['frames']:>5} {row['position_mae_m']:>10.6f} "
              f"{row['position_err_norm_m']:>10.6f} {row['rotation_mae_rad']:>11.6f} "
              f"{row['gripper_mae']:>9.5f} {row['gripper_accuracy']:>8.3f} {row['overall_l1']:>10.6f}")
    print("-" * len(header))
    print(f"{'ALL':>8} {aggregate['frames']:>5} {aggregate['position_mae_m']:>10.6f} "
          f"{aggregate['position_err_norm_m']:>10.6f} {aggregate['rotation_mae_rad']:>11.6f} "
          f"{aggregate['gripper_mae']:>9.5f} {aggregate['gripper_accuracy']:>8.3f} "
          f"{aggregate['overall_l1']:>10.6f}")
    print("=" * len(header))
    print("\nPer-dimension MAE / RMSE (all evaluated frames):")
    units = ["m", "m", "m", "rad", "rad", "rad", "grip"]
    print(f"{'dim':>10} {'MAE':>12} {'RMSE':>12}   units")
    for j, name in enumerate(dim_names):
        unit = units[j] if j < len(units) else "?"
        print(f"{name:>10} {aggregate['per_dim_mae'][j]:>12.6f} "
              f"{aggregate['per_dim_rmse'][j]:>12.6f}   {unit}")
    print()


def main() -> None:
    args = parse_args()
    if args.num_integration_steps is not None and args.num_integration_steps <= 0:
        raise SystemExit("--num-integration-steps must be positive")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    seed_everything(args.seed)

    missing = sorted(n for n in REQUIRED_CHECKPOINT_FILES if not (args.checkpoint / n).is_file())
    if missing:
        raise SystemExit(f"incomplete checkpoint {args.checkpoint}: missing {missing}")

    episodes = [int(v) for v in args.episodes.split(",") if v.strip()]
    if not episodes:
        raise SystemExit("--episodes is empty")
    args.out.mkdir(parents=True, exist_ok=True)

    meta_dataset = LeRobotDataset(args.repo_id, root=args.dataset_root,
                                  episodes=[episodes[0]], video_backend=args.video_backend)
    meta = meta_dataset.meta
    camera_keys = list(meta.camera_keys)
    dim_names = list(meta.features[ACTION].get("names") or []) or list(FALLBACK_DIM_NAMES)
    stats = meta.stats
    del meta_dataset

    policy, preprocessor, postprocessor, cfg = load_policy(
        args.checkpoint, device, stats, args.num_integration_steps, args.n_action_steps
    )

    if args.image_resize == "auto":
        use_act_resize = cfg.type == "act"
    else:
        use_act_resize = args.image_resize == "act"
    transforms = build_image_transforms(ACT_RESIZE_HW) if use_act_resize else None

    action_dim = len(dim_names)
    if action_dim != 7:
        print(f"[warn] action dim is {action_dim}, not the 7 of the EEF contract; "
              f"position/rotation slices assume 7", file=sys.stderr)

    print(f"checkpoint = {args.checkpoint}")
    print(f"policy     = {cfg.type} objective={getattr(cfg, 'objective', 'n/a')} "
          f"integration_steps={getattr(cfg, 'num_integration_steps', 'n/a')} "
          f"n_action_steps={getattr(policy.config, 'n_action_steps', 'n/a')}")
    print(f"images     = {'Resize' + str(ACT_RESIZE_HW) if use_act_resize else 'native (policy resizes internally)'}")
    print(f"episodes   = {episodes}  cameras = {camera_keys}  device = {device}  seed = {args.seed}")
    print(f"action dims ({action_dim}): {dim_names}")

    rows: list[dict] = []
    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    for episode in episodes:
        pred, gt = rollout_episode(
            episode=episode, dataset_root=args.dataset_root, repo_id=args.repo_id,
            video_backend=args.video_backend, task=args.task, policy=policy,
            preprocessor=preprocessor, postprocessor=postprocessor,
            camera_keys=camera_keys, transforms=transforms, max_frames=args.max_frames,
        )
        metrics = metric_dict(pred, gt)
        rows.append({"episode": episode, **metrics})
        all_pred.append(pred)
        all_gt.append(gt)
        print(f"[info] episode {episode}: T={metrics['frames']} "
              f"posMAE={metrics['position_mae_m']:.6f} m "
              f"rotMAE={metrics['rotation_mae_rad']:.6f} rad "
              f"gripAcc={metrics['gripper_accuracy']:.3f}")

    aggregate = metric_dict(np.concatenate(all_pred), np.concatenate(all_gt))
    print_table(rows, aggregate, dim_names)

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "repo_id": args.repo_id,
        "episodes": episodes,
        "task": args.task,
        "policy_type": cfg.type,
        "objective": getattr(cfg, "objective", None),
        "integration_method": getattr(cfg, "integration_method", None),
        "integration_steps": getattr(cfg, "num_integration_steps", None),
        "n_action_steps": getattr(policy.config, "n_action_steps", None),
        "image_resize": ACT_RESIZE_HW if use_act_resize else None,
        "device": device,
        "seed": args.seed,
        "action_names": dim_names,
        "action_space_id": "eef_delta_v1",
        "per_episode": rows,
        "aggregate": aggregate,
    }
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    fields = ["episode", "frames", "position_mae_m", "position_err_norm_m", "rotation_mae_rad",
              "rotation_axis_mae_rad", "gripper_mae", "gripper_accuracy", "overall_l1"]
    with (args.out / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row[f] for f in fields})
        writer.writerow({f: aggregate[f] for f in fields if f != "episode"} | {"episode": "ALL"})
    print(f"wrote {args.out / 'metrics.json'}")
    print(f"wrote {args.out / 'episodes.csv'}")


if __name__ == "__main__":
    main()
