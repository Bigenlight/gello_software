#!/usr/bin/env python3
"""Train a JAX rectified-flow policy on canonical HIL-SERL demonstrations.

The model predicts a chunk of the deployed seven-dimensional action contract:
six normalized EEF deltas plus the gripper channel.  Gripper is part of the
same flow field and is thresholded to ``-1/+1`` only after Euler integration.

The output is a standalone FM-init artifact, never a hybrid SAC resume
checkpoint and never written below a production learner run root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
sys.path.insert(
    0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher")
)

from ur_env.compat import (  # noqa: E402
    configure_flax_local_io,
    configure_pure_python_protobuf,
)

configure_pure_python_protobuf()

from ur_env.learner import (  # noqa: E402
    FrozenResNet10TrunkExtractor,
    LearnerConfig,
    convert_loaded_demos_to_feature_pool,
    create_frozen_trunk_feature_agent,
    default_resnet_source,
    load_demo_pickles,
)
from ur_env.learner.flow_matching import (  # noqa: E402
    ACTION_DIM,
    EEF_DIM,
    GRIPPER_INDEX,
    FlowMatchingConfig,
    FlowMatchingPolicy,
    make_flow_training_batch,
    masked_velocity_losses,
    sample_action_chunks,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train JAX flow matching on canonical EEF-delta demos."
    )
    parser.add_argument("--demo-path", action="append", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--resnet-source")
    parser.add_argument("--resnet-cache")
    parser.add_argument("--extraction-batch-size", type=int, default=64)
    parser.add_argument("--report-path")
    parser.add_argument(
        "--checkpoint-path",
        help="new directory for best/final JAX FM parameter artifacts",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "batch_size",
        "horizon",
        "integration_steps",
        "eval_samples",
        "eval_every",
        "hidden_dim",
        "residual_blocks",
        "extraction_batch_size",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= args.holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0,1)")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")


def _materialize_feature_pool(pool) -> dict[str, object]:
    batch = pool.warmup_batch(len(pool))
    observations = {
        key: np.asarray(value) for key, value in batch["observations"].items()
    }
    actions = np.asarray(batch["actions"], dtype=np.float32)
    groups: list[str] = []
    steps: list[int] = []
    for sidecar in pool.sidecars:
        metadata = sidecar.metadata
        group = metadata.get("source_take") or metadata.get("session_id")
        if group is None:
            group = f"{sidecar.source_path}:episode-{metadata.get('episode_id', 0)}"
        groups.append(str(group))
        steps.append(int(metadata.get("step_id", sidecar.item_index)))
    return {
        "observations": observations,
        "actions": actions,
        "groups": np.asarray(groups),
        "steps": np.asarray(steps, dtype=np.int64),
    }


def _build_action_chunks(
    data: dict[str, object], horizon: int
) -> dict[str, object]:
    actions = data["actions"]
    groups = data["groups"]
    steps = data["steps"]
    chunks = np.zeros((len(actions), horizon, ACTION_DIM), dtype=np.float32)
    valid = np.zeros((len(actions), horizon), dtype=np.float32)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        indices = indices[np.argsort(steps[indices], kind="stable")]
        for offset, row_index in enumerate(indices):
            following = indices[offset : offset + horizon]
            chunks[row_index, : len(following)] = actions[following]
            valid[row_index, : len(following)] = 1.0
    return {
        "observations": data["observations"],
        "actions": chunks,
        "valid": valid,
        "groups": groups,
    }


def _split_indices(
    groups: np.ndarray, holdout_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    order = rng.permutation(unique_groups)
    count = int(round(len(order) * holdout_fraction))
    if holdout_fraction > 0.0 and len(order) > 1:
        count = max(1, min(len(order) - 1, count))
    is_holdout = np.isin(groups, order[:count])
    return np.flatnonzero(~is_holdout), np.flatnonzero(is_holdout)


def _take(data: dict[str, object], indices: np.ndarray) -> dict[str, object]:
    return {
        "observations": {
            key: value[indices] for key, value in data["observations"].items()
        },
        "actions": data["actions"][indices],
        "valid": data["valid"][indices],
        "groups": data["groups"][indices],
    }


def _sample_metrics(
    predicted: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    mask = valid.astype(bool)
    continuous_error = np.square(
        predicted[..., :EEF_DIM] - target[..., :EEF_DIM]
    )
    gripper_correct = (
        predicted[..., GRIPPER_INDEX] == target[..., GRIPPER_INDEX]
    )
    return {
        "continuous_mse": float(np.mean(continuous_error[mask])),
        "first_continuous_mse": float(
            np.mean(continuous_error[:, 0, :])
        ),
        "gripper_accuracy": float(np.mean(gripper_correct[mask])),
        "first_gripper_accuracy": float(np.mean(gripper_correct[:, 0])),
    }


def _average_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([value[key] for value in values]))
        for key in values[0]
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _save_checkpoint(
    *,
    destination: Path,
    model: FlowMatchingPolicy,
    best_params,
    final_params,
    best_epoch: int,
    best_metrics: dict[str, float],
    final_metrics: dict[str, float],
    args: argparse.Namespace,
    resnet_sha256: str,
) -> dict[str, object]:
    from flax import serialization

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    files = {}
    for label, params in (("best", best_params), ("final", final_params)):
        name = f"{label}_flow_params.msgpack"
        payload = serialization.to_bytes(params)
        serialization.from_bytes(params, payload)
        _write_new(destination / name, payload)
        files[label] = {
            "path": name,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }

    demos = []
    for value in args.demo_path:
        path = Path(value).expanduser().resolve()
        demos.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "format": "hil-serl-jax-flow-matching",
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_checkpoint_compatible": False,
        "optimizer_state_included": False,
        "action_contract": (
            "chunk[horizon,7]: normalized EEF delta[6] + gripper[-1,+1]; "
            "all channels trained by rectified flow"
        ),
        "observation_contract": (
            "cam1/cam2 frozen ResNet10 maps (1,4,4,512) + state (1,19)"
        ),
        "model_config": model.config.document(),
        "resnet_sha256": resnet_sha256,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "parameter_files": files,
        "demo_files": demos,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "holdout_fraction": args.holdout_fraction,
            "eval_samples": args.eval_samples,
            "seed": args.seed,
        },
    }
    manifest_bytes = json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    _write_new(destination / "manifest.json", manifest_bytes)

    report = (
        str(Path(args.report_path).expanduser().resolve())
        if args.report_path
        else "(not written)"
    )
    readme = (
        "# JAX flow-matching artifact\n\n"
        f"- best weights: `{destination / files['best']['path']}`\n"
        f"- final weights: `{destination / files['final']['path']}`\n"
        f"- manifest: `{destination / 'manifest.json'}`\n"
        f"- report: `{report}`\n"
        f"- model code: `{Path(__file__).resolve().parents[1] / 'ur_env/learner/flow_matching.py'}`\n\n"
        "Load `best_flow_params.msgpack` with `FlowMatchingPolicy` and sample "
        "with `sample_action_chunks`. This is not a SAC resume checkpoint.\n"
    )
    _write_new(destination / "README.md", readme.encode("utf-8"))

    completion = {
        "complete": True,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "best_parameter_sha256": files["best"]["sha256"],
        "final_parameter_sha256": files["final"]["sha256"],
    }
    _write_new(
        destination / "completion.json",
        json.dumps(completion, indent=2, sort_keys=True).encode("utf-8"),
    )
    return {
        "path": str(destination),
        "best_parameter_sha256": files["best"]["sha256"],
        "final_parameter_sha256": files["final"]["sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    configure_flax_local_io()

    import jax
    import jax.numpy as jnp
    import optax

    print(f"[fm] jax backend: {jax.default_backend()}", flush=True)
    resnet_source = Path(
        args.resnet_source or default_resnet_source()
    ).expanduser().resolve()
    agent = create_frozen_trunk_feature_agent(
        config=LearnerConfig(),
        resnet_source_path=resnet_source,
        resnet_cache_path=args.resnet_cache,
    )
    extractor = FrozenResNet10TrunkExtractor(
        agent, resnet_asset_path=resnet_source
    )
    extractor.validate_agent_invariant(agent)

    loaded = load_demo_pickles(args.demo_path)
    pool = convert_loaded_demos_to_feature_pool(
        loaded,
        feature_extractor=extractor,
        seed=args.seed,
        extraction_batch_size=args.extraction_batch_size,
    )
    flat = _materialize_feature_pool(pool)
    raw_actions = flat["actions"]
    values, counts = np.unique(raw_actions[:, GRIPPER_INDEX], return_counts=True)
    distribution = {
        str(float(value)): int(count) for value, count in zip(values, counts)
    }
    if raw_actions.shape[1] != ACTION_DIM:
        raise SystemExit(f"expected action width {ACTION_DIM}, got {raw_actions.shape}")
    if not np.all(np.isin(values, (-1.0, 1.0))):
        raise SystemExit(
            "this FM artifact discretizes the sampled gripper at zero and "
            f"requires binary training values; got {distribution}"
        )
    if not np.isfinite(raw_actions).all() or np.max(np.abs(raw_actions)) > 1.0001:
        raise SystemExit("actions must be finite and normalized to [-1,1]")

    chunked = _build_action_chunks(flat, args.horizon)
    train_index, holdout_index = _split_indices(
        chunked["groups"], args.holdout_fraction, args.seed
    )
    train = _take(chunked, train_index)
    holdout = _take(chunked, holdout_index) if holdout_index.size else train
    print(
        f"[fm] transitions={len(raw_actions)} train={len(train_index)} "
        f"holdout={len(holdout_index)} groups="
        f"{np.unique(train['groups']).size}/{np.unique(holdout['groups']).size} "
        f"horizon={args.horizon} gripper={distribution}",
        flush=True,
    )

    model_config = FlowMatchingConfig(
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        residual_blocks=args.residual_blocks,
        integration_steps=args.integration_steps,
    )
    model = FlowMatchingPolicy(model_config)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    example_observations = {
        key: value[:1] for key, value in train["observations"].items()
    }
    example_actions = train["actions"][:1]
    params = model.init(
        init_rng,
        example_observations,
        example_actions,
        jnp.zeros((1,), dtype=jnp.float32),
    )["params"]
    parameter_count = sum(
        int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params)
    )
    print(f"[fm] parameters={parameter_count:,}", flush=True)

    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params_, opt_state_, observations, actions, valid, step_rng):
        def loss_fn(candidate):
            noisy, timestep, target_velocity = make_flow_training_batch(
                actions, step_rng
            )
            predicted_velocity = model.apply(
                {"params": candidate}, observations, noisy, timestep
            )
            losses = masked_velocity_losses(
                predicted_velocity, target_velocity, valid
            )
            return losses["loss"], losses

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_)
        updates, new_opt_state = optimizer.update(grads, opt_state_, params_)
        return optax.apply_updates(params_, updates), new_opt_state, metrics

    @jax.jit
    def sample_once(params_, observations, sample_rng):
        return sample_action_chunks(
            model,
            params_,
            observations,
            sample_rng,
            integration_steps=args.integration_steps,
            discretize_gripper=True,
        )

    def evaluate_sampled(params_, epoch: int) -> dict[str, float]:
        sample_results = []
        for sample_index in range(args.eval_samples):
            sample_rng = jax.random.fold_in(
                jax.random.PRNGKey(args.seed + 10_000), sample_index
            )
            predicted = np.asarray(
                jax.device_get(
                    sample_once(params_, holdout["observations"], sample_rng)
                )
            )
            sample_results.append(
                _sample_metrics(predicted, holdout["actions"], holdout["valid"])
            )
        result = _average_metrics(sample_results)
        result["epoch"] = epoch
        return result

    shuffle_rng = np.random.default_rng(args.seed + 1)
    history: list[dict[str, float]] = []
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_params = None
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        order = shuffle_rng.permutation(len(train_index))
        epoch_metrics = {
            "loss": [],
            "continuous_velocity_mse": [],
            "gripper_velocity_mse": [],
        }
        for start in range(0, len(order), args.batch_size):
            selection = order[start : start + args.batch_size]
            observations = {
                key: value[selection]
                for key, value in train["observations"].items()
            }
            rng, step_rng = jax.random.split(rng)
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                observations,
                train["actions"][selection],
                train["valid"][selection],
                step_rng,
            )
            for key in epoch_metrics:
                epoch_metrics[key].append(float(metrics[key]))

        record = {"epoch": epoch}
        record.update(
            {
                f"train_{key}": float(np.mean(values))
                for key, values in epoch_metrics.items()
            }
        )
        should_evaluate = (
            epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        )
        if should_evaluate:
            sampled = evaluate_sampled(params, epoch)
            record.update(
                {f"holdout_{key}": value for key, value in sampled.items() if key != "epoch"}
            )
            if (
                best_metrics is None
                or sampled["first_continuous_mse"]
                < best_metrics["first_continuous_mse"]
            ):
                best_epoch = epoch
                best_metrics = sampled
                best_params = jax.tree_util.tree_map(
                    lambda value: np.asarray(jax.device_get(value)).copy(), params
                )
        history.append(record)
        print(f"[fm] {json.dumps(record)}", flush=True)

    elapsed_s = time.perf_counter() - started
    final_metrics = evaluate_sampled(params, args.epochs)
    if best_params is None or best_metrics is None:
        raise RuntimeError("no held-out evaluation was performed")

    report = {
        "demo_paths": list(args.demo_path),
        "transitions": len(raw_actions),
        "train_transitions": len(train_index),
        "holdout_transitions": len(holdout_index),
        "train_groups": sorted(np.unique(train["groups"]).tolist()),
        "holdout_groups": sorted(np.unique(holdout["groups"]).tolist()),
        "model_config": model_config.document(),
        "parameter_count": parameter_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "eval_samples": args.eval_samples,
        "seed": args.seed,
        "jax_backend": jax.default_backend(),
        "elapsed_s": elapsed_s,
        "gripper_distribution": distribution,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "history": history,
    }
    if args.checkpoint_path:
        report["checkpoint"] = _save_checkpoint(
            destination=Path(args.checkpoint_path).expanduser().resolve(),
            model=model,
            best_params=best_params,
            final_params=params,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            final_metrics=final_metrics,
            args=args,
            resnet_sha256=extractor.resnet_sha256,
        )
        print(f"[fm] checkpoint written: {report['checkpoint']['path']}")

    if args.report_path:
        destination = Path(args.report_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        print(f"[fm] report written: {destination}")

    print(
        f"[fm] best epoch={best_epoch} first-action continuous MSE="
        f"{best_metrics['first_continuous_mse']:.6f} gripper accuracy="
        f"{best_metrics['first_gripper_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
