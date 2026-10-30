#!/usr/bin/env python3
"""Fit BC on the canonical demos to test whether the action contract holds.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
This is a diagnostic, not a training stage.  It asks one question: given the
observation the learner actually sees, is the recorded action *predictable*?

If BC cannot fit the demos, then observation and action do not correspond --
a frame convention is inverted, an axis is permuted, the crop moved, or the
recorded action does not describe the motion that occurred.  No amount of RL
repairs that, because every algorithm downstream assumes the pair is coherent.
If BC does fit, the pair is coherent and a failure to learn lives elsewhere
(reward density, exploration, the critic).

It never touches a production run root.  With ``--checkpoint-path`` it writes
an explicitly labelled BC-initialisation artifact containing only the actor
and grasp parameter subtrees; it is not a resumable learner checkpoint.

WHY IT TRAINS THE REAL AGENT HEADS INSTEAD OF FRESH NETWORKS
------------------------------------------------------------
``create_frozen_trunk_feature_agent`` is called and its ``actor`` and
``grasp_critic`` heads are optimized.  Building standalone networks with "the
same" arguments would be a second source of truth for the architecture.  The
continuous actor loss therefore exercises the exact policy tree serialized by
``CheckpointManager``.  The gripper BCE treats the deployed three-value
grasp-Q head's close/open outputs as binary logits.

THE FROZEN TRUNK MUST NOT MOVE
------------------------------
The ResNet-10 trunk is pinned by SHA and shared with the reward classifier and
the replay features.  Training it here would invalidate every feature already
extracted, so its leaves are masked out of the update and the invariant is
re-checked after every epoch through the extractor that owns the contract.

ACTION DIMENSIONALITY
---------------------
The policy is ``action_dim=6``: the EEF delta only.  Element 6 of a stored
action is the gripper, which the hybrid agent routes to a three-choice
``grasp_critic`` (``-1=close, 0=hold, +1=open``), not to the Gaussian policy.
This diagnostic uses BCE only when the supplied corpus is genuinely binary
(``-1/+1``).  It fails loudly on a ``0`` label rather than silently changing
the deployed three-way action contract.

USAGE (do not run this on a host whose GPU is busy training)

    python serl_ur_infra/scripts/diagnose_bc_on_demos.py \
        --demo-path ~/hil-serl-data/datasets/<canonical>.pkl \
        --epochs 10
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


EEF_DIM = 6
GRIPPER_INDEX = 6

#: ``ModuleDict`` stores each submodule under ``modules_<key>`` while
#: ``apply_fn`` still addresses it by the bare key, so the two names differ and
#: both are needed: one to select the subtree, one to call it.
ACTOR_MODULE_NAME = "actor"
ACTOR_PARAM_SUBTREE = "modules_actor"
GRASP_MODULE_NAME = "grasp_critic"
GRASP_PARAM_SUBTREE = "modules_grasp_critic"
GRIPPER_CLOSE_VALUE = -1.0
GRIPPER_OPEN_VALUE = 1.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit behaviour cloning on canonical demos and report whether the "
            "recorded action is predictable from the learner's observation."
        )
    )
    parser.add_argument(
        "--demo-path",
        action="append",
        required=True,
        help="canonical demo pickle; repeat for multiple files",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--gripper-loss-weight",
        type=float,
        default=1.0,
        help="weight of binary close/open BCE beside the continuous actor NLL",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.1,
        help=(
            "fraction of transitions withheld from training; without it a "
            "falling loss cannot be told apart from memorisation"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resnet-source")
    parser.add_argument("--resnet-cache")
    parser.add_argument(
        "--extraction-batch-size",
        type=int,
        default=64,
        help="frozen-trunk forward batch used while converting demos",
    )
    parser.add_argument(
        "--report-path",
        help="optional JSON report destination; stdout always carries a summary",
    )
    parser.add_argument(
        "--checkpoint-path",
        help=(
            "optional new directory for a BC-init artifact; this is deliberately "
            "not a production learner resume checkpoint"
        ),
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "extraction_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= args.holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if (
        not np.isfinite(args.gripper_loss_weight)
        or args.gripper_loss_weight < 0.0
    ):
        raise ValueError("gripper_loss_weight must be finite and non-negative")


def _stacked_pool(pool) -> dict[str, object]:
    """Materialize the whole feature pool once as dense arrays.

    The pool exists to be *sampled*, which is what the learner wants and what
    a fixed-epoch sweep does not: an epoch has to visit each transition once,
    and a held-out split has to stay held out.  The corpus is ~2k transitions
    of (1,4,4,512) features, so materializing is cheap and removes any doubt
    about which rows were seen.
    """

    # ``sample(len(pool))`` still samples WITH replacement.  warmup_batch uses
    # deterministic cyclic indices, and with exactly len(pool) entries that is
    # the sole public path which returns every stored row exactly once.
    batch = pool.warmup_batch(len(pool))
    observations = {
        key: np.asarray(value) for key, value in batch["observations"].items()
    }
    actions = np.asarray(batch["actions"], dtype=np.float32)
    groups = []
    for sidecar in pool.sidecars:
        metadata = sidecar.metadata
        group = metadata.get("source_take") or metadata.get("session_id")
        if group is None and "episode_id" in metadata:
            group = f"{sidecar.source_path}:episode-{metadata['episode_id']}"
        if group is None:
            group = f"{sidecar.source_path}:row-{sidecar.item_index}"
        groups.append(str(group))
    return {
        "observations": observations,
        "actions": actions,
        "groups": np.asarray(groups),
    }


def _split_indices(
    groups: np.ndarray, *, holdout_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    order = rng.permutation(unique_groups)
    holdout_group_count = int(round(len(order) * holdout_fraction))
    if holdout_fraction > 0.0 and len(order) > 1:
        holdout_group_count = max(1, min(len(order) - 1, holdout_group_count))
    holdout_groups = order[:holdout_group_count]
    is_holdout = np.isin(groups, holdout_groups)
    return np.flatnonzero(~is_holdout), np.flatnonzero(is_holdout)


def _take(data: dict[str, object], indices: np.ndarray) -> dict[str, object]:
    observations = {
        key: value[indices] for key, value in data["observations"].items()
    }
    return {
        "observations": observations,
        "actions": data["actions"][indices],
        "groups": data["groups"][indices],
    }


def _direction_report(
    predicted: np.ndarray, recorded: np.ndarray
) -> dict[str, object]:
    """Per-axis agreement between the fitted mean action and the demo action.

    Correlation answers "does it move the right way", scale answers "does it
    move the right amount", and sign agreement is the blunt instrument that
    catches an inverted axis even when correlation is diluted by noise.
    """

    axes: list[dict[str, float]] = []
    for axis in range(EEF_DIM):
        p = predicted[:, axis].astype(np.float64)
        r = recorded[:, axis].astype(np.float64)
        p_std = float(p.std())
        r_std = float(r.std())
        if p_std > 0.0 and r_std > 0.0:
            correlation = float(np.corrcoef(p, r)[0, 1])
        else:
            # A constant prediction has no direction to agree with; reporting
            # 0.0 would read as "uncorrelated" when the real finding is
            # "collapsed".
            correlation = float("nan")
        moving = np.abs(r) > 1e-6
        sign_agreement = (
            float(np.mean(np.sign(p[moving]) == np.sign(r[moving])))
            if moving.any()
            else float("nan")
        )
        axes.append(
            {
                "axis": axis,
                "correlation": correlation,
                "sign_agreement": sign_agreement,
                "mae": float(np.mean(np.abs(p - r))),
                "mse": float(np.mean(np.square(p - r))),
                "predicted_std": p_std,
                "recorded_std": r_std,
                "scale_ratio": (
                    float(p_std / r_std) if r_std > 0.0 else float("nan")
                ),
            }
        )
    return {
        "axes": axes,
        "mean_correlation": float(
            np.nanmean([axis["correlation"] for axis in axes])
        ),
        "mean_sign_agreement": float(
            np.nanmean([axis["sign_agreement"] for axis in axes])
        ),
        "mean_mse": float(np.mean(np.square(predicted - recorded))),
    }


def _gripper_report(
    actions: np.ndarray, close_logits: np.ndarray
) -> dict[str, object]:
    values, counts = np.unique(actions[:, GRIPPER_INDEX], return_counts=True)
    recorded_close = actions[:, GRIPPER_INDEX] == GRIPPER_CLOSE_VALUE
    predicted_close = close_logits >= 0.0
    return {
        "note": (
            "binary diagnostic over grasp_critic close/open outputs; the live "
            "policy still exposes the upstream {-1,0,+1} action set"
        ),
        "distribution": {
            str(float(value)): int(count)
            for value, count in zip(values, counts)
        },
        "accuracy": float(np.mean(predicted_close == recorded_close)),
        "confusion": {
            "recorded_close_predicted_close": int(
                np.sum(recorded_close & predicted_close)
            ),
            "recorded_close_predicted_open": int(
                np.sum(recorded_close & ~predicted_close)
            ),
            "recorded_open_predicted_close": int(
                np.sum(~recorded_close & predicted_close)
            ),
            "recorded_open_predicted_open": int(
                np.sum(~recorded_close & ~predicted_close)
            ),
        },
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_file(path: Path, payload: bytes) -> None:
    """Publish one new file without overwriting and fsync its contents."""

    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _save_bc_checkpoint(
    *,
    destination: Path,
    params,
    args: argparse.Namespace,
    final_metrics: dict[str, float],
    resnet_sha256: str,
) -> dict[str, object]:
    """Write a self-verifying BC-init artifact, not a learner checkpoint.

    ``completion.json`` is the last file published.  A directory without that
    marker is incomplete and must not be consumed.
    """

    import jax
    from flax import serialization

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)

    selected_params = {
        ACTOR_PARAM_SUBTREE: params[ACTOR_PARAM_SUBTREE],
        GRASP_PARAM_SUBTREE: params[GRASP_PARAM_SUBTREE],
    }
    parameter_bytes = serialization.to_bytes(selected_params)
    restored = serialization.from_bytes(selected_params, parameter_bytes)
    if jax.tree_util.tree_structure(restored) != jax.tree_util.tree_structure(
        selected_params
    ):
        raise RuntimeError("BC checkpoint serialization changed parameter tree")

    parameter_name = "actor_grasp_params.msgpack"
    parameter_sha256 = _sha256_bytes(parameter_bytes)
    _write_new_file(destination / parameter_name, parameter_bytes)

    demo_files = []
    for value in args.demo_path:
        path = Path(value).expanduser().resolve()
        demo_files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )

    manifest = {
        "format": "hil-serl-bc-init",
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "BC-initialised actor and grasp parameter subtrees; not a "
            "resumable production learner checkpoint"
        ),
        "production_checkpoint_compatible": False,
        "optimizer_state_included": False,
        "parameter_subtrees": [ACTOR_PARAM_SUBTREE, GRASP_PARAM_SUBTREE],
        "frozen_pretrained_encoder_updated": False,
        "parameter_file": parameter_name,
        "parameter_bytes": len(parameter_bytes),
        "parameter_sha256": parameter_sha256,
        "resnet_sha256": resnet_sha256,
        "demo_files": demo_files,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gripper_loss_weight": args.gripper_loss_weight,
            "holdout_fraction": args.holdout_fraction,
            "seed": args.seed,
        },
        "final_evaluation_metrics": final_metrics,
    }
    manifest_bytes = json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    _write_new_file(destination / "manifest.json", manifest_bytes)

    report_path = (
        str(Path(args.report_path).expanduser().resolve())
        if args.report_path
        else "(not written)"
    )
    readme = (
        "# BC init artifact\n\n"
        f"- weights: `{destination / parameter_name}`\n"
        f"- manifest: `{destination / 'manifest.json'}`\n"
        f"- report: `{report_path}`\n\n"
        "`actor_grasp_params.msgpack` is a local BC-init weight artifact, "
        "not a production learner resume checkpoint.\n"
    )
    _write_new_file(destination / "README.md", readme.encode("utf-8"))

    completion = {
        "complete": True,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "parameter_sha256": parameter_sha256,
    }
    _write_new_file(
        destination / "completion.json",
        json.dumps(completion, indent=2, sort_keys=True).encode("utf-8"),
    )
    return {
        "path": str(destination),
        "format": manifest["format"],
        "parameter_file": parameter_name,
        "parameter_sha256": parameter_sha256,
        "manifest_sha256": completion["manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    configure_flax_local_io()

    import jax
    import jax.numpy as jnp
    import optax

    config = LearnerConfig()
    resnet_source = Path(
        args.resnet_source or default_resnet_source()
    ).expanduser().resolve()

    print(f"[bc] jax backend: {jax.default_backend()}", flush=True)
    print("[bc] building the production agent (actor + grasp heads are fitted)")
    agent = create_frozen_trunk_feature_agent(
        config=config,
        resnet_source_path=resnet_source,
        resnet_cache_path=args.resnet_cache,
    )
    extractor = FrozenResNet10TrunkExtractor(
        agent, resnet_asset_path=resnet_source
    )
    extractor.validate_agent_invariant(agent)

    print(f"[bc] loading demos: {args.demo_path}")
    loaded = load_demo_pickles(args.demo_path)
    pool = convert_loaded_demos_to_feature_pool(
        loaded,
        feature_extractor=extractor,
        seed=args.seed,
        extraction_batch_size=args.extraction_batch_size,
    )
    data = _stacked_pool(pool)
    total = int(data["actions"].shape[0])
    print(f"[bc] transitions: {total}")
    gripper_values, gripper_counts = np.unique(
        data["actions"][:, GRIPPER_INDEX], return_counts=True
    )
    gripper_distribution = {
        str(float(value)): int(count)
        for value, count in zip(gripper_values, gripper_counts)
    }
    if not np.all(
        np.isin(
            gripper_values,
            (GRIPPER_CLOSE_VALUE, GRIPPER_OPEN_VALUE),
        )
    ):
        raise SystemExit(
            "gripper BCE requires an exactly binary {-1,+1} corpus; got "
            f"{gripper_distribution}. Use three-class categorical CE when "
            "the upstream hold action (0) is present."
        )
    print(f"[bc] binary gripper distribution: {gripper_distribution}")

    train_index, holdout_index = _split_indices(
        data["groups"],
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    train = _take(data, train_index)
    holdout = _take(data, holdout_index) if holdout_index.size else None
    holdout_group_count = (
        np.unique(holdout["groups"]).size if holdout is not None else 0
    )
    print(
        f"[bc] train={train_index.size} holdout={holdout_index.size} "
        f"train_groups={np.unique(train['groups']).size} "
        f"holdout_groups={holdout_group_count} "
        f"batch={args.batch_size} epochs={args.epochs}"
    )

    params = agent.state.params
    top_level = sorted(params.keys())
    print(f"[bc] parameter subtrees: {top_level}")
    missing_subtrees = {
        ACTOR_PARAM_SUBTREE,
        GRASP_PARAM_SUBTREE,
    } - set(params)
    if missing_subtrees:
        raise SystemExit(
            f"expected parameter subtrees are missing: {sorted(missing_subtrees)}; "
            f"got {top_level}. The architecture moved -- fix this script "
            "rather than training whatever happens to be first."
        )

    # Only the continuous actor and grasp head are fitted, and inside both the
    # pinned trunk stays put.  The mask is built from paths so a renamed module
    # fails loudly rather than quietly training weights the replay depends on.
    def _trainable(path, _leaf) -> bool:
        keys = [getattr(entry, "key", str(entry)) for entry in path]
        if keys[0] not in {ACTOR_PARAM_SUBTREE, GRASP_PARAM_SUBTREE}:
            return False
        return not any("pretrained_encoder" in key for key in keys)

    trainable_mask = jax.tree_util.tree_map_with_path(_trainable, params)
    trainable_count = sum(
        int(flag) for flag in jax.tree_util.tree_leaves(trainable_mask)
    )
    total_leaves = len(jax.tree_util.tree_leaves(params))
    print(
        f"[bc] trainable leaves: {trainable_count} / {total_leaves} "
        "(actor + grasp head, frozen trunks excluded)"
    )
    if trainable_count == 0:
        raise SystemExit("no trainable leaves were selected; refusing to run")

    optimizer = optax.masked(
        optax.adam(args.learning_rate), trainable_mask
    )
    opt_state = optimizer.init(params)

    def loss_fn(trained_params, observations, actions, rng, *, train_mode):
        actor_rng, grasp_rng = jax.random.split(rng)
        distribution = agent.state.apply_fn(
            {"params": trained_params},
            observations,
            name=ACTOR_MODULE_NAME,
            rngs={"dropout": actor_rng},
            train=train_mode,
        )
        actor_nll = -jnp.mean(
            distribution.log_prob(actions[:, :EEF_DIM])
        )
        grasp_values = agent.state.apply_fn(
            {"params": trained_params},
            observations,
            name=GRASP_MODULE_NAME,
            rngs={"dropout": grasp_rng},
            train=train_mode,
        )
        # Upstream indexes grasp values as 0=close, 1=hold, 2=open.  The new
        # corpus has only close/open commands, so their difference is the
        # binary close logit.  A positive value predicts close (-1).
        close_logit = grasp_values[:, 0] - grasp_values[:, 2]
        close_label = (
            actions[:, GRIPPER_INDEX] == GRIPPER_CLOSE_VALUE
        ).astype(jnp.float32)
        gripper_bce = jnp.mean(
            optax.sigmoid_binary_cross_entropy(close_logit, close_label)
        )
        total_loss = actor_nll + args.gripper_loss_weight * gripper_bce
        predicted_action = distribution.mode()
        metrics = {
            "loss": total_loss,
            "actor_nll": actor_nll,
            "continuous_mse": jnp.mean(
                jnp.square(predicted_action - actions[:, :EEF_DIM])
            ),
            "gripper_bce": gripper_bce,
            "gripper_accuracy": jnp.mean(
                (close_logit >= 0.0) == (close_label > 0.5)
            ),
        }
        return total_loss, (metrics, predicted_action, close_logit)

    @jax.jit
    def train_step(trained_params, opt_state_, observations, actions, rng):
        (_, (metrics, _, _)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(
            trained_params,
            observations,
            actions,
            rng,
            train_mode=True,
        )
        updates, opt_state_ = optimizer.update(grads, opt_state_, trained_params)
        return optax.apply_updates(trained_params, updates), opt_state_, metrics

    @jax.jit
    def evaluate(trained_params, observations, actions, rng):
        _, result = loss_fn(
            trained_params,
            observations,
            actions,
            rng,
            train_mode=False,
        )
        return result

    rng = jax.random.PRNGKey(args.seed)
    shuffle_rng = np.random.default_rng(args.seed + 1)
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        order = shuffle_rng.permutation(train_index.size)
        train_metrics: dict[str, list[float]] = {
            "loss": [],
            "actor_nll": [],
            "continuous_mse": [],
            "gripper_bce": [],
            "gripper_accuracy": [],
        }
        for start in range(0, order.size, args.batch_size):
            batch_index = order[start : start + args.batch_size]
            if batch_index.size == 0:
                continue
            batch = _take(train, batch_index)
            rng, step_rng = jax.random.split(rng)
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                batch["observations"],
                batch["actions"],
                step_rng,
            )
            for key in train_metrics:
                train_metrics[key].append(float(metrics[key]))

        record = {"epoch": epoch}
        record.update(
            {
                f"train_{key}": float(np.mean(values))
                for key, values in train_metrics.items()
            }
        )
        if holdout is not None:
            rng, eval_rng = jax.random.split(rng)
            holdout_metrics, _, _ = evaluate(
                params,
                holdout["observations"],
                holdout["actions"],
                eval_rng,
            )
            record.update(
                {
                    f"holdout_{key}": float(value)
                    for key, value in holdout_metrics.items()
                }
            )
        history.append(record)
        # The trunk is re-checked every epoch, not once at the end: a mask
        # mistake would otherwise be discovered only after the whole run.
        extractor.validate_parameter_invariant(params)
        print(f"[bc] {json.dumps(record)}", flush=True)

    elapsed_s = time.perf_counter() - started

    evaluation = holdout if holdout is not None else train
    rng, final_rng = jax.random.split(rng)
    final_metrics, predicted, close_logits = evaluate(
        params,
        evaluation["observations"],
        evaluation["actions"],
        final_rng,
    )
    direction = _direction_report(
        np.asarray(jax.device_get(predicted)),
        evaluation["actions"][:, :EEF_DIM],
    )
    final_metrics_serialized = {
        key: float(value) for key, value in final_metrics.items()
    }

    report = {
        "demo_paths": list(args.demo_path),
        "transitions": total,
        "train_transitions": int(train_index.size),
        "holdout_transitions": int(holdout_index.size),
        "train_groups": sorted(np.unique(train["groups"]).tolist()),
        "holdout_groups": (
            sorted(np.unique(holdout["groups"]).tolist())
            if holdout is not None
            else []
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gripper_loss_weight": args.gripper_loss_weight,
        "seed": args.seed,
        "elapsed_s": elapsed_s,
        "jax_backend": jax.default_backend(),
        "history": history,
        "evaluated_on": "holdout" if holdout is not None else "train",
        "final_metrics": final_metrics_serialized,
        "direction": direction,
        "gripper": _gripper_report(
            evaluation["actions"],
            np.asarray(jax.device_get(close_logits)),
        ),
    }

    if args.checkpoint_path:
        checkpoint_destination = Path(args.checkpoint_path).expanduser().resolve()
        report["checkpoint"] = _save_bc_checkpoint(
            destination=checkpoint_destination,
            params=params,
            args=args,
            final_metrics=final_metrics_serialized,
            resnet_sha256=extractor.resnet_sha256,
        )
        print(f"[bc] BC-init checkpoint written: {checkpoint_destination}")

    print("\n[bc] per-axis direction agreement (0-5 = EEF delta):")
    for axis in direction["axes"]:
        print(
            f"  axis {axis['axis']}: corr={axis['correlation']:+.3f} "
            f"sign={axis['sign_agreement']:.3f} "
            f"scale={axis['scale_ratio']:.3f} mae={axis['mae']:.4f} "
            f"mse={axis['mse']:.5f}"
        )
    print(
        f"[bc] mean correlation {direction['mean_correlation']:+.3f}, "
        f"mean sign agreement {direction['mean_sign_agreement']:.3f}, "
        f"mean MSE {direction['mean_mse']:.5f}"
    )
    print(
        f"[bc] gripper BCE {float(final_metrics['gripper_bce']):.4f}, "
        f"accuracy {float(final_metrics['gripper_accuracy']):.3f}"
    )
    print(
        "[bc] READ THIS AS: near-zero correlation or ~0.5 sign agreement means "
        "the action is not predictable from the observation -- fix that before "
        "interpreting any RL result. A negative correlation on one axis is an "
        "inverted convention on that axis specifically."
    )

    if args.report_path:
        destination = Path(args.report_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[bc] report written: {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
