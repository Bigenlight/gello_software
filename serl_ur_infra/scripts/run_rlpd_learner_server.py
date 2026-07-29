#!/usr/bin/env python3
"""Run the single-process local gRPC ingress and HIL-SERL learner."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time


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

from ur_env.grpc_actor_transport import create_grpc_server  # noqa: E402
from ur_env.learner import (  # noqa: E402
    CheckpointManager,
    CheckpointRunLock,
    FROZEN_TRUNK_CONTRACT,
    FROZEN_TRUNK_MODEL_REVISION,
    FROZEN_TRUNK_SYNTHETIC_E2E_MODEL_REVISION,
    FaultGatedReplayIngress,
    FeatureReplayIngress,
    FeatureReplayMemoryError,
    FrozenResNet10TrunkExtractor,
    JsonlWandbLogger,
    LearnerConfig,
    LearnerFingerprint,
    LearnerWorker,
    build_actor_service,
    compose_learner,
    convert_loaded_demos_to_feature_pool,
    create_frozen_trunk_feature_agent,
    default_resnet_source,
    estimate_feature_demo_memory,
    estimate_feature_replay_memory,
    file_sha256,
    load_demo_pickles,
    preflight_checkpoint_run,
    prepare_learner_state,
    system_available_memory_bytes,
    validate_learner_dependencies,
)
from ur_env.classifier_sidecar import CLASSIFIER_INPUT_ID  # noqa: E402
from ur_env.learner.demo import SYNTHETIC_ACCEPTANCE_ONLY_KEY  # noqa: E402
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)
from ur_env.rlpd_receive_server import (  # noqa: E402
    DEFAULT_CLASSIFIER_CONFIRMATIONS,
    DEFAULT_INTERVENTION_CAPACITY,
    DEFAULT_REPLAY_CAPACITY,
    DEFAULT_REWARD_THRESHOLD,
    RewardClassifierRuntime,
)


# Directory sha256 (ur_env.classifier_sidecar.directory_sha256) of the canonical
# cube-in-cup orbax checkpoint tree:
#
#   classifier_ckpt/cube_in_cup_all3/checkpoint_150
#
# Recompute after ANY change to that tree with:
#
#   /home/laptop3/venvs/gello-hil-actor/bin/python -c "
#   import sys; sys.path.insert(0, '<repo>/serl_ur_infra')
#   from ur_env.classifier_sidecar import directory_sha256
#   print(directory_sha256('<repo>/classifier_ckpt/cube_in_cup_all3/checkpoint_150'))"
#
# WHY THE OLD VALUE HAD TO GO.  This used to be
# e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997, the digest
# of a RETIRED checkpoint scoring 0% recall on the current data domain.  Before
# directory hashing existed the mismatch was masked (the old file-only
# checkpoint_sha256() could not read an orbax *directory* at all, so it raised).
# With directory hashing in place, leaving the stale pin here would let the
# learner start cleanly against the retired weights and then emit reward==0 for
# every transition, forever, with no error anywhere -- RLPD would keep training
# and learn nothing.  Silent-permanent-zero is the worst failure mode this
# system has, so the pin is a code default, not an operator flag.
DEFAULT_CLASSIFIER_CHECKPOINT_SHA256 = (
    "512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d"
)

# Advertised over GetServerInfo; the actor pins it via
# ros2_ur_ws/run_hil_actor.sh::EXPECTED_REWARD_MODEL_ID.  The id names the
# checkpoint AND the input contract on purpose: a pre-sidecar actor talking to a
# post-sidecar server (or the reverse) computes reward from different pixels
# than the other side believes, so the pair must be rejected at the handshake
# instead of running a whole session on wrong rewards.  Keep this string, the
# wrapper's default, and CLASSIFIER_INPUT_ID moving together.
DEFAULT_REWARD_MODEL_ID = "cube-in-cup-all3-ckpt150+sidecar-v1"

# ONE-TIME, INTENTIONAL LEARNER-FINGERPRINT BREAK
# -----------------------------------------------
# The classifier SHA, the reward_model_id and the new run_contract fields below
# all feed LearnerFingerprint, so every checkpoint written before this change
# will be refused fail-closed on resume.  That is correct: the old lineage was
# trained against the retired 0%-recall checkpoint, so its critic learned from
# rewards that were structurally zero.  There is nothing of value to resume.
# Start a fresh --checkpoint-root once; after that the fingerprint is stable
# again.  ur_env/learner/composition.py::prepare_learner_state explains this in
# the refusal message so it does not read like a bug.


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run gRPC actor ingress and exactly one local CTA learner worker "
            "against the same RAM replay buffers."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument(
        "--expected-classifier-sha256",
        default=DEFAULT_CLASSIFIER_CHECKPOINT_SHA256,
    )
    parser.add_argument("--reward-model-id", default=DEFAULT_REWARD_MODEL_ID)
    parser.add_argument(
        "--reward-threshold", type=float, default=DEFAULT_REWARD_THRESHOLD
    )
    # WHY THE DEFAULT IS 1 (smoothing OFF).  Operator decision, not an accident:
    # the current checkpoint behaves well at DEFAULT_REWARD_THRESHOLD=0.2, and a
    # server that silently smoothed its decision would disagree with the live
    # classifier viewer (REWARD_CLASSIFIER_LIVE_KO.md), which reports raw
    # per-frame probability.  Anyone comparing the two would be debugging the
    # filter instead of the robot.  This value feeds the learner fingerprint, so
    # changing it also breaks resume -- deliberately.
    parser.add_argument(
        "--success-confirmations",
        type=int,
        # Single source of truth with the finalizer's own default so the CLI
        # and RewardTransitionFinalizer cannot drift apart.
        default=DEFAULT_CLASSIFIER_CONFIRMATIONS,
        help=(
            "consecutive over-threshold classifications required before an "
            "episode is called successful; 1 (default) means no smoothing"
        ),
    )
    parser.add_argument(
        "--demo-path",
        action="append",
        required=True,
        help="trusted canonical demo pickle; repeat for multiple files",
    )
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument(
        "--checkpoint-reserve-gib",
        type=float,
        default=2.0,
        help=(
            "minimum free filesystem space which must remain after every "
            "checkpoint; checkpoints are never pruned"
        ),
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-path")
    resume.add_argument(
        "--resume-latest",
        action="store_true",
        help=(
            "resume the newest structurally complete checkpoint in the output "
            "root; if a higher damaged entry exists, use --resume-path with "
            "a new empty --checkpoint-root"
        ),
    )
    parser.add_argument("--jsonl-path")
    parser.add_argument(
        "--memory-preflight-path",
        help=(
            "append-only JSONL audit for startup RAM gates; defaults to "
            "memory-preflight.jsonl beside --jsonl-path"
        ),
    )
    parser.add_argument("--wandb-dir")
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default="offline",
    )
    parser.add_argument("--wandb-project", default="hil-serl")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--hil-serl-root",
        default=str(_REPO_ROOT / "third_party" / "hil-serl"),
    )
    parser.add_argument("--resnet-source")
    parser.add_argument("--resnet-cache")
    parser.add_argument(
        "--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY
    )
    parser.add_argument(
        "--intervention-capacity",
        type=int,
        default=DEFAULT_INTERVENTION_CAPACITY,
    )
    parser.add_argument(
        "--feature-memory-reserve-gib",
        type=float,
        default=2.0,
        help=(
            "minimum RAM left beyond persistent feature replay/demo tensors; "
            "raw images are never stored in learner buffers"
        ),
    )
    parser.add_argument(
        "--demo-extraction-batch-size",
        type=int,
        default=64,
        help="one-time frozen-trunk demo conversion batch size",
    )
    parser.add_argument(
        "--grasp-penalty",
        type=float,
        default=-0.02,
        help=(
            "configured redundant gripper command penalty; learner data may "
            "contain only 0 or this non-positive value"
        ),
    )
    parser.add_argument(
        "--utd-ratio",
        type=int,
        default=1,
        help=(
            "outer learner steps permitted per newly accepted online "
            "transition after replay warm-up; CTA remains independently 2:1"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-message-bytes", type=int, default=16 * 1024 * 1024
    )
    parser.add_argument(
        "--require-jax-backend",
        choices=("cpu", "gpu"),
        required=True,
        help="fail closed if JAX selected a different backend",
    )
    parser.add_argument(
        "--target-learner-step",
        type=int,
        help=(
            "absolute, checkpoint-aligned learner step at which to stop; "
            "omit for continuous production training; synthetic E2E mode "
            "requires a bounded target"
        ),
    )
    parser.add_argument("--poll-interval", type=float, default=0.1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="build and smoke every component, log preflight, but do not bind",
    )
    mode.add_argument(
        "--synthetic-e2e",
        action="store_true",
        help=(
            "accept only synthetic acceptance demos, bind the real gRPC "
            "learner, and use one-step publish/checkpoint periods for a "
            "bounded laptop-to-server learning acceptance run"
        ),
    )
    parser.add_argument(
        "--synthetic-actor-id",
        default="fake-e2e-actor",
        help="only actor identity accepted by --synthetic-e2e",
    )
    parser.add_argument(
        "--synthetic-run-id",
        help="only run identity accepted by --synthetic-e2e",
    )
    parser.add_argument(
        "--synthetic-transition-count",
        type=int,
        default=100,
        help="exact accepted transition count required by --synthetic-e2e",
    )
    parser.add_argument(
        "--synthetic-timeout-s",
        type=float,
        default=300.0,
        help="wall-clock deadline for a bounded --synthetic-e2e run",
    )
    return parser.parse_args(argv)


def _emit(event: str, **fields: object) -> None:
    try:
        print(
            json.dumps(
                {"event": event, **fields},
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    except (BrokenPipeError, OSError):
        pass


def _grpc_bind_address(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _validate_args(args: argparse.Namespace) -> None:
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("learner server is loopback-only; use an SSH tunnel")
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    for name in (
        "replay_capacity",
        "intervention_capacity",
        "demo_extraction_batch_size",
        "max_workers",
        "max_message_bytes",
        "utd_ratio",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.success_confirmations < 1:
        raise ValueError("success_confirmations must be at least 1")
    if args.target_learner_step is not None and args.target_learner_step < 0:
        raise ValueError("target_learner_step must be non-negative")
    if args.synthetic_e2e:
        if (
            args.target_learner_step is None
            or args.target_learner_step <= 0
            or args.target_learner_step > 10
        ):
            raise ValueError(
                "synthetic_e2e requires target_learner_step in [1, 10]"
            )
        if args.replay_capacity < LearnerConfig().training_starts:
            raise ValueError(
                "synthetic_e2e replay_capacity must be at least the "
                "production training_starts threshold"
            )
        if not isinstance(args.synthetic_actor_id, str) or not args.synthetic_actor_id:
            raise ValueError("synthetic_e2e requires synthetic_actor_id")
        if not isinstance(args.synthetic_run_id, str) or not args.synthetic_run_id:
            raise ValueError("synthetic_e2e requires synthetic_run_id")
        if args.synthetic_transition_count != LearnerConfig().training_starts:
            raise ValueError(
                "synthetic_transition_count must equal the production "
                "training_starts threshold"
            )
        if args.synthetic_transition_count > args.replay_capacity:
            raise ValueError(
                "synthetic_transition_count cannot exceed replay_capacity"
            )
        if (
            not math.isfinite(args.synthetic_timeout_s)
            or not 1.0 <= args.synthetic_timeout_s <= 1_800.0
        ):
            raise ValueError("synthetic_timeout_s must be in [1, 1800]")
    elif args.synthetic_run_id is not None:
        raise ValueError("synthetic_run_id requires --synthetic-e2e")
    if (
        not math.isfinite(args.checkpoint_reserve_gib)
        or args.checkpoint_reserve_gib < 0.0
    ):
        raise ValueError("checkpoint_reserve_gib must be finite and non-negative")
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0.0:
        raise ValueError("poll_interval must be positive and finite")
    if not math.isfinite(args.grasp_penalty) or args.grasp_penalty > 0.0:
        raise ValueError("grasp_penalty must be finite and non-positive")
    if (
        not math.isfinite(args.feature_memory_reserve_gib)
        or args.feature_memory_reserve_gib < 0.0
    ):
        raise ValueError(
            "feature_memory_reserve_gib must be finite and non-negative"
        )


def _learner_config(args: argparse.Namespace) -> LearnerConfig:
    """Build the fingerprinted algorithm config for production or acceptance."""

    config_options: dict[str, object] = {
        "wandb_mode": args.wandb_mode,
        "utd_ratio": args.utd_ratio,
    }
    if args.synthetic_e2e:
        # Keep the real batch=256, replay threshold=100, CTA ratio, optimizer,
        # and model.  Only lifecycle periods are shortened so one bounded fake
        # run proves publish and checkpoint without pretending to be a
        # 5,000-step robot experiment.
        config_options.update(publish_period=1, checkpoint_period=1)
    return LearnerConfig(**config_options)


def _validate_synthetic_progress_target(
    args: argparse.Namespace, prepared_state: object
) -> None:
    if not args.synthetic_e2e:
        return
    start = getattr(prepared_state, "learner_step", None)
    if args.target_learner_step != start + 1:
        raise ValueError(
            "synthetic_e2e target_learner_step must be exactly one step "
            "beyond the fresh/restored learner step"
        )


def _preflight_combined_feature_memory(
    *,
    replay_bytes: int,
    demo_bytes: int,
    reserve_bytes: int,
    available_bytes: int | None = None,
) -> int:
    """Fail before conversion/allocation unless persistent feature RAM fits."""

    values = {
        "replay_bytes": replay_bytes,
        "demo_bytes": demo_bytes,
        "reserve_bytes": reserve_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    available = (
        system_available_memory_bytes()
        if available_bytes is None
        else available_bytes
    )
    if (
        isinstance(available, bool)
        or not isinstance(available, int)
        or available < 0
    ):
        raise ValueError("available_bytes must be a non-negative integer")
    required = replay_bytes + demo_bytes + reserve_bytes
    if available < required:
        raise FeatureReplayMemoryError(
            "feature learner needs "
            f"{(replay_bytes + demo_bytes) / 1024**3:.3f} GiB persistent "
            f"tensors plus {reserve_bytes / 1024**3:.3f} GiB reserve, but "
            f"only {available / 1024**3:.3f} GiB is available"
        )
    return available


def _memory_preflight_path(
    args: argparse.Namespace, checkpoint_manager: CheckpointManager
) -> Path:
    if args.memory_preflight_path:
        return Path(args.memory_preflight_path).expanduser().resolve()
    learner_log = Path(
        args.jsonl_path
        or checkpoint_manager.root / "logs" / "learner.jsonl"
    ).expanduser().resolve()
    return learner_log.with_name("memory-preflight.jsonl")


def _append_memory_preflight_record(
    path: os.PathLike[str] | str, record: dict[str, object]
) -> Path:
    """Durably append one startup decision before learner service starts."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def _record_combined_feature_memory_preflight(
    *,
    report_path: os.PathLike[str] | str,
    phase: str,
    replay_bytes: int,
    demo_bytes: int,
    reserve_bytes: int,
    replay_capacity: int,
    intervention_capacity: int,
    demo_transition_count: int,
    demo_sha256: list[str],
) -> int:
    """Use the existing RAM gate and persist both acceptance and refusal."""

    available = system_available_memory_bytes()
    refusal: FeatureReplayMemoryError | None = None
    try:
        checked_available = _preflight_combined_feature_memory(
            replay_bytes=replay_bytes,
            demo_bytes=demo_bytes,
            reserve_bytes=reserve_bytes,
            available_bytes=available,
        )
    except FeatureReplayMemoryError as exc:
        checked_available = available
        refusal = exc

    required = replay_bytes + demo_bytes + reserve_bytes
    record = {
        "event": "rlpd_learner_memory_preflight",
        "schema_version": 1,
        "time_ns": time.time_ns(),
        "phase": phase,
        "decision": "rejected" if refusal is not None else "accepted",
        "available_memory_bytes": checked_available,
        "replay_fixed_tensor_bytes": replay_bytes,
        "demo_tensor_bytes_in_this_gate": demo_bytes,
        "reserve_bytes": reserve_bytes,
        "required_available_bytes": required,
        "margin_bytes": checked_available - required,
        "replay_capacity": replay_capacity,
        "intervention_capacity": intervention_capacity,
        "offline_demo_transition_count": demo_transition_count,
        "offline_demo_sha256": demo_sha256,
    }
    if refusal is not None:
        record["refusal_detail"] = str(refusal)
    destination = _append_memory_preflight_record(report_path, record)
    _emit(
        "rlpd_learner_memory_preflight",
        phase=phase,
        decision=record["decision"],
        available_memory_bytes=checked_available,
        required_available_bytes=required,
        margin_bytes=checked_available - required,
        report_path=str(destination),
    )
    if refusal is not None:
        raise refusal
    return checked_available


def _validate_jax_backend(actual: str, required: str) -> str:
    actual = str(actual).strip().lower()
    if actual != required:
        raise RuntimeError(
            f"JAX backend is {actual!r}, but {required!r} is required"
        )
    return actual


def _validate_demo_serving_scope(
    demos: object,
    *,
    dry_run: bool,
    synthetic_e2e: bool = False,
) -> int:
    """Reject synthetic acceptance artifacts before allocating live services."""

    sidecars = getattr(demos, "sidecars", None)
    if sidecars is None:
        raise TypeError("loaded demos must expose provenance sidecars")
    synthetic = tuple(
        sidecar
        for sidecar in sidecars
        if sidecar.metadata.get(SYNTHETIC_ACCEPTANCE_ONLY_KEY) is True
    )
    if synthetic_e2e:
        if not synthetic or len(synthetic) != len(sidecars):
            raise ValueError(
                "--synthetic-e2e requires every offline demo item to carry "
                "the synthetic acceptance-only marker"
            )
    elif synthetic and not dry_run:
        first = synthetic[0]
        raise ValueError(
            "synthetic acceptance-only demo data is permitted only with "
            "--dry-run or bounded --synthetic-e2e; real learner serving "
            "requires robot demo data "
            f"(first synthetic item: {first.source_path}[{first.item_index}])"
        )
    return len(synthetic)


def _validate_demo_grasp_penalty(demos: object, *, expected: float) -> None:
    """Enforce the same configured penalty values for offline and online data."""

    transitions = getattr(demos, "transitions", None)
    if transitions is None:
        raise TypeError("loaded demos must expose transitions")
    for index, transition in enumerate(transitions):
        value = float(transition["grasp_penalty"])
        if not (
            math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-7)
            or math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-7)
        ):
            raise ValueError(
                "offline demo grasp_penalty must be either 0 or the "
                f"configured penalty {expected}; item {index} has {value}"
            )


def _log_best_effort(logger: JsonlWandbLogger, event: str, **fields: object) -> None:
    try:
        logger.log(event, **fields)
    except Exception as exc:
        _emit(
            "learner_aux_log_failed",
            event_name=event,
            error_type=type(exc).__name__,
            detail=str(exc)[:2_000],
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    reserve_bytes = int(args.checkpoint_reserve_gib * 1024**3)
    checkpoint_manager = CheckpointManager(
        args.checkpoint_root,
        minimum_free_bytes_after_save=reserve_bytes,
    )
    with CheckpointRunLock(checkpoint_manager.root):
        return _run_locked(args, checkpoint_manager)


def _run_locked(
    args: argparse.Namespace,
    checkpoint_manager: CheckpointManager,
) -> int:
    config = _learner_config(args)
    policy_model_id = (
        FROZEN_TRUNK_SYNTHETIC_E2E_MODEL_REVISION
        if args.synthetic_e2e
        else FROZEN_TRUNK_MODEL_REVISION
    )
    feature_memory_reserve_bytes = int(
        args.feature_memory_reserve_gib * 1024**3
    )
    if (
        args.target_learner_step is not None
        and args.target_learner_step % config.checkpoint_period
    ):
        raise ValueError(
            "target_learner_step must fall on a checkpoint boundary"
        )
    resume_path = preflight_checkpoint_run(
        checkpoint_manager,
        resume_path=args.resume_path,
        resume_latest=args.resume_latest,
    )
    versions = validate_learner_dependencies(
        include_logging=args.wandb_mode != "disabled"
    )
    configure_flax_local_io()

    import jax

    jax_backend = _validate_jax_backend(
        jax.default_backend(), args.require_jax_backend
    )
    demo_sha256 = [file_sha256(path) for path in args.demo_path]
    demos = load_demo_pickles(args.demo_path)
    if [file_sha256(path) for path in args.demo_path] != demo_sha256:
        raise RuntimeError("a demo artifact changed while it was being loaded")
    if not demos.transitions:
        raise ValueError("at least one canonical offline demo is required")
    synthetic_demo_count = _validate_demo_serving_scope(
        demos,
        dry_run=args.dry_run,
        synthetic_e2e=args.synthetic_e2e,
    )
    _validate_demo_grasp_penalty(demos, expected=args.grasp_penalty)
    demo_count = len(demos.transitions)
    replay_memory_estimate = estimate_feature_replay_memory(
        replay_capacity=args.replay_capacity,
        intervention_capacity=args.intervention_capacity,
    )
    demo_memory_estimate = estimate_feature_demo_memory(demo_count)
    memory_preflight_path = _memory_preflight_path(args, checkpoint_manager)
    available_memory_bytes = _record_combined_feature_memory_preflight(
        report_path=memory_preflight_path,
        phase="forecast_before_model_setup",
        replay_bytes=replay_memory_estimate.fixed_tensor_bytes,
        demo_bytes=demo_memory_estimate.total_bytes,
        reserve_bytes=feature_memory_reserve_bytes,
        replay_capacity=args.replay_capacity,
        intervention_capacity=args.intervention_capacity,
        demo_transition_count=demo_count,
        demo_sha256=demo_sha256,
    )
    resnet_source = Path(
        args.resnet_source or default_resnet_source()
    ).expanduser().resolve()
    classifier = RewardClassifierRuntime(
        checkpoint_path=args.classifier_checkpoint,
        expected_sha256=args.expected_classifier_sha256,
        threshold=args.reward_threshold,
        reward_model_id=args.reward_model_id,
        hil_serl_root=args.hil_serl_root,
        resnet_source_path=resnet_source,
        resnet_cache_path=args.resnet_cache,
    )
    run_contract = {
        "contract_revision": "frozen_trunk_feature_hybrid_sac_v1",
        "execution_scope": (
            "synthetic_laptop_server_e2e_v1"
            if args.synthetic_e2e
            else "production_robot_data_v1"
        ),
        "policy_model_id": policy_model_id,
        "augmentation": config.augmentation,
        "policy_observations": {
            "representation": "canonical_raw_uint8_v1",
            "schema_hash": CANONICAL_OBSERVATION_SCHEMA_HASH,
        },
        "learner_observations": FROZEN_TRUNK_CONTRACT.document(),
        "action": {
            "dtype": "float32",
            "shape": [7],
            "eef_range": [-1.0, 1.0],
            "gripper_values": [-1.0, 0.0, 1.0],
        },
        "reward_classifier": {
            "sha256": classifier.checkpoint_sha256,
            "threshold": classifier.threshold,
            "reward_model_id": classifier.reward_model_id,
            # WHICH PIXELS the classifier scored, independent of which weights
            # scored them.  Without this a replay buffer filled from cropped
            # policy observations and one filled from uncropped sidecar frames
            # produce the same fingerprint, and a run could be resumed across
            # the change with two incompatible reward distributions mixed in
            # one critic.  The recorded value is the identity of the transport
            # contract, not a flag: it stays CLASSIFIER_INPUT_ID as long as the
            # server decodes full-frame JPEG passthrough.
            "input_contract": CLASSIFIER_INPUT_ID,
            # Part of the reward definition: at n>1 the same probabilities
            # produce a different terminal step, so it must not be silently
            # changeable mid-lineage.
            "success_confirmations": int(args.success_confirmations),
        },
        "offline_demo_sha256": demo_sha256,
        "offline_demo_transition_count": demo_count,
        "grasp_penalty": {
            "allowed_values": [0.0, args.grasp_penalty],
            "contract_revision": "configured_redundant_command_v1",
        },
        "algorithm_dependencies": {
            name: versions[name]
            for name in (
                "jax",
                "jaxlib",
                "flax",
                "distrax",
                "tensorflow_probability",
            )
        },
    }
    fingerprint = LearnerFingerprint.create(
        config=config,
        resnet_asset_path=resnet_source,
        run_contract=run_contract,
    )
    agent_template = create_frozen_trunk_feature_agent(
        config=config,
        hil_serl_root=args.hil_serl_root,
        resnet_source_path=resnet_source,
        resnet_cache_path=args.resnet_cache,
        validate_versions=False,
    )
    prepared_state = prepare_learner_state(
        agent_template=agent_template,
        checkpoint_manager=checkpoint_manager,
        fingerprint=fingerprint,
        config=config,
        resume_path=resume_path,
    )
    _validate_synthetic_progress_target(args, prepared_state)
    feature_extractor = FrozenResNet10TrunkExtractor(
        agent_template,
        resnet_asset_path=resnet_source,
        image_keys=config.image_keys,
    )
    # A checkpoint may never redefine the trunk named by the verified asset
    # SHA: every persistent feature is meaningful only under these exact
    # weights.  This also rejects legacy/bad checkpoints before conversion.
    feature_extractor.validate_agent_invariant(prepared_state.agent)
    feature_demos = convert_loaded_demos_to_feature_pool(
        demos,
        feature_extractor=feature_extractor,
        seed=config.seed,
        extraction_batch_size=args.demo_extraction_batch_size,
    )
    if feature_demos.storage_nbytes != demo_memory_estimate.total_bytes:
        raise RuntimeError(
            "converted demo allocation differs from its preflight estimate"
        )
    # The long-lived pool owns only float32 features and copied provenance.
    # Drop all canonical raw demo references before allocating live rings.
    del demos
    gc.collect()

    available_before_replay_bytes = _record_combined_feature_memory_preflight(
        report_path=memory_preflight_path,
        phase="gate_before_replay_allocation",
        replay_bytes=replay_memory_estimate.fixed_tensor_bytes,
        # The converted demo and model are resident now, so their RAM is
        # already reflected in MemAvailable and must not be counted twice.
        demo_bytes=0,
        reserve_bytes=feature_memory_reserve_bytes,
        replay_capacity=args.replay_capacity,
        intervention_capacity=args.intervention_capacity,
        demo_transition_count=demo_count,
        demo_sha256=demo_sha256,
    )
    raw_ingress = FeatureReplayIngress(
        feature_extractor=feature_extractor,
        replay_capacity=args.replay_capacity,
        intervention_capacity=args.intervention_capacity,
        seed=config.seed,
        expected_grasp_penalty=args.grasp_penalty,
        available_memory_bytes=available_before_replay_bytes,
        memory_reserve_bytes=feature_memory_reserve_bytes,
    )
    ingress = FaultGatedReplayIngress(raw_ingress)

    jsonl_path = Path(
        args.jsonl_path
        or checkpoint_manager.root / "logs" / "learner.jsonl"
    )
    logger = JsonlWandbLogger(
        jsonl_path,
        wandb_mode=args.wandb_mode,
        wandb_dir=args.wandb_dir,
        project=args.wandb_project,
        run_name=args.run_name,
        config={
            "learner": config.fingerprint_values(),
            "fingerprint_sha256": fingerprint.sha256,
            "run_contract": run_contract,
            "demo_count": demo_count,
            "synthetic_acceptance_demo_count": synthetic_demo_count,
            "observation_schema_hash": CANONICAL_OBSERVATION_SCHEMA_HASH,
            "dependencies": versions,
            "jax_backend": jax_backend,
            "checkpoint_reserve_bytes": (
                checkpoint_manager.minimum_free_bytes_after_save
            ),
            "feature_memory": {
                "available_at_preflight_bytes": available_memory_bytes,
                "available_before_replay_allocation_bytes": (
                    available_before_replay_bytes
                ),
                "preflight_report_path": str(memory_preflight_path),
                "replay_fixed_tensor_bytes": (
                    replay_memory_estimate.fixed_tensor_bytes
                ),
                "replay_camera_bytes": replay_memory_estimate.camera_bytes,
                "offline_demo_tensor_bytes": demo_memory_estimate.total_bytes,
                "reserve_bytes": feature_memory_reserve_bytes,
            },
        },
        enable_wandb=args.wandb_mode != "disabled",
    )
    worker: LearnerWorker | None = None
    server = None
    service = None
    shutdown_event = threading.Event()
    old_handlers: dict[int, object] = {}
    exit_code = 0
    logger_closed = False
    synthetic_candidate = False
    synthetic_event_fields: dict[str, object] | None = None
    try:
        assembly = compose_learner(
            agent_template=agent_template,
            ingress=ingress,
            offline_demos=feature_demos,
            checkpoint_manager=checkpoint_manager,
            fingerprint=fingerprint,
            config=config,
            logger=logger,
            prepared_state=prepared_state,
            parameter_validator=(
                feature_extractor.validate_parameter_invariant
            ),
            candidate_postprocessor=feature_extractor.repin_target_trunk,
            policy_model_id=policy_model_id,
        )
        service = build_actor_service(
            assembly=assembly,
            classifier=classifier,
            success_confirmations=args.success_confirmations,
            allowed_actor_ids=(
                (args.synthetic_actor_id,) if args.synthetic_e2e else None
            ),
            allowed_run_ids=(
                (args.synthetic_run_id,) if args.synthetic_e2e else None
            ),
        )
        restored_path = (
            str(assembly.restored_checkpoint.path)
            if assembly.restored_checkpoint is not None
            else None
        )
        logger.log(
            "learner_process_ready",
            learner_step=assembly.learner.learner_step,
            gradient_step=assembly.learner.gradient_step,
            policy_version=assembly.policy_runtime.policy_version,
            restored_checkpoint=restored_path,
            demo_count=demo_count,
            synthetic_acceptance_demo_count=synthetic_demo_count,
            jax_backend=jax_backend,
        )
        if args.dry_run:
            _emit(
                "rlpd_learner_dry_run_passed",
                learner_step=assembly.learner.learner_step,
                gradient_step=assembly.learner.gradient_step,
                policy_version=assembly.policy_runtime.policy_version,
                restored_checkpoint=restored_path,
                demo_count=demo_count,
                synthetic_acceptance_demo_count=synthetic_demo_count,
                fingerprint_sha256=fingerprint.sha256,
                jsonl_path=str(logger.path),
                jax_backend=jax_backend,
            )
            try:
                logger.close()
            except Exception as exc:
                _emit(
                    "rlpd_learner_logger_close_failed",
                    error_type=type(exc).__name__,
                    detail=str(exc)[:2_000],
                )
                return 5
            finally:
                # Avoid a second close from the outer finally.  In particular,
                # Python evaluates a return value before executing finally, so
                # merely mutating exit_code there cannot repair a false-zero
                # dry-run result.
                logger_closed = True
            return 0

        server, bound_port = create_grpc_server(
            service,
            bind_address=_grpc_bind_address(args.host, args.port),
            max_workers=args.max_workers,
            max_message_bytes=args.max_message_bytes,
        )
        worker = LearnerWorker(
            assembly.learner,
            replay_insert_count=lambda: (
                raw_ingress.status().replay_insert_count
            ),
            target_learner_step=args.target_learner_step,
            poll_interval=args.poll_interval,
        )

        def request_shutdown(_signum, _frame) -> None:
            shutdown_event.set()

        for signal_number in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, request_shutdown)

        server.start()
        worker.start()
        synthetic_deadline = (
            time.monotonic() + args.synthetic_timeout_s
            if args.synthetic_e2e
            else None
        )
        _emit(
            "rlpd_learner_server_ready",
            host=args.host,
            port=bound_port,
            policy_version=assembly.policy_runtime.policy_version,
            learner_step=assembly.learner.learner_step,
            reward_model_id=classifier.reward_model_id,
            # The full reward contract on one line: which weights, which pixels,
            # how much smoothing.  Cross-check against the actor's
            # EXPECTED_REWARD_MODEL_ID before trusting a session's rewards.
            classifier_sha256=classifier.checkpoint_sha256,
            classifier_input_contract=CLASSIFIER_INPUT_ID,
            success_confirmations=args.success_confirmations,
            policy_model_id=assembly.policy_runtime.model_id,
            replay_capacity=raw_ingress.replay_capacity,
            intervention_capacity=raw_ingress.intervention_capacity,
            demo_count=demo_count,
            fingerprint_sha256=fingerprint.sha256,
            jax_backend=jax_backend,
            jax_device_count=len(jax.devices()),
            feature_encoding=raw_ingress.feature_encoding_id,
            utd_ratio=config.utd_ratio,
            critic_to_actor_ratio=config.cta_ratio,
            feature_replay_fixed_tensor_bytes=(
                replay_memory_estimate.fixed_tensor_bytes
            ),
            offline_demo_tensor_bytes=demo_memory_estimate.total_bytes,
            persistence="learner_checkpoints_only_replay_ram",
            synthetic_actor_id=(
                args.synthetic_actor_id if args.synthetic_e2e else None
            ),
            synthetic_run_id=(
                args.synthetic_run_id if args.synthetic_e2e else None
            ),
            synthetic_transition_count=(
                args.synthetic_transition_count if args.synthetic_e2e else None
            ),
        )

        learner_fault_reported = False
        while not shutdown_event.wait(args.poll_interval):
            if (
                synthetic_deadline is not None
                and time.monotonic() >= synthetic_deadline
            ):
                exit_code = 6
                worker.request_stop()
                _emit(
                    "rlpd_learner_synthetic_e2e_timeout",
                    timeout_s=args.synthetic_timeout_s,
                    replay_insert_count=(
                        raw_ingress.status().replay_insert_count
                    ),
                    expected_transition_count=args.synthetic_transition_count,
                )
                break
            alive, ready, detail = service.health()
            if not alive or not ready:
                exit_code = 3
                worker.request_stop()
                _log_best_effort(
                    logger,
                    "actor_service_fault",
                    learner_step=assembly.learner.learner_step,
                    gradient_step=assembly.learner.gradient_step,
                    policy_version=assembly.policy_runtime.policy_version,
                    detail=detail,
                )
                _emit("rlpd_learner_actor_service_fault", detail=detail)
                break
            status = worker.status
            if status.state == "faulted" and not learner_fault_reported:
                learner_fault_reported = True
                _emit(
                    "rlpd_learner_worker_fault",
                    learner_step=status.learner_step,
                    gradient_step=status.gradient_step,
                    policy_version=status.policy_version,
                    detail=status.detail,
                    serving_last_known_good=True,
                )
                if args.target_learner_step is not None:
                    exit_code = 2
                    break
            if (
                args.target_learner_step is not None
                and status.state == "completed"
            ):
                break
        if args.synthetic_e2e:
            status = worker.status
            target = int(args.target_learner_step)
            ingress_status = raw_ingress.status()
            checkpoint_path = None
            checkpoint_selection_error = None
            if status.state == "completed":
                try:
                    checkpoint_path = checkpoint_manager.latest_path()
                except Exception as exc:
                    checkpoint_selection_error = (
                        f"{type(exc).__name__}: {str(exc)[:2_000]}"
                    )
            synthetic_candidate = (
                exit_code == 0
                and status.state == "completed"
                and status.learner_step == target
                and status.learner_step - prepared_state.learner_step == 1
                and status.gradient_step - prepared_state.gradient_step
                == config.cta_ratio
                and status.policy_version - prepared_state.policy_version == 1
                and ingress_status.replay_insert_count
                == args.synthetic_transition_count
                and checkpoint_path is not None
                and checkpoint_path.name == f"checkpoint_{target:012d}"
            )
            synthetic_event_fields = {
                "target_learner_step": target,
                "start_learner_step": prepared_state.learner_step,
                "learner_step": status.learner_step,
                "start_gradient_step": prepared_state.gradient_step,
                "gradient_step": status.gradient_step,
                "start_policy_version": prepared_state.policy_version,
                "policy_version": status.policy_version,
                "worker_state": status.state,
                "checkpoint_path": (
                    None if checkpoint_path is None else str(checkpoint_path)
                ),
                "replay_size": ingress_status.replay_size,
                "replay_insert_count": ingress_status.replay_insert_count,
                "expected_transition_count": args.synthetic_transition_count,
                "fingerprint_sha256": fingerprint.sha256,
            }
            if checkpoint_selection_error is not None:
                synthetic_event_fields["checkpoint_selection_error"] = (
                    checkpoint_selection_error
                )
    finally:
        shutdown_event.set()
        if worker is not None:
            worker.request_stop()
        if server is not None:
            try:
                stopped = server.stop(grace=2.0).wait(timeout=10.0)
                if stopped is False:
                    exit_code = max(exit_code, 5)
                    _emit("rlpd_learner_grpc_shutdown_timeout")
            except Exception as exc:
                exit_code = max(exit_code, 5)
                _emit(
                    "rlpd_learner_grpc_shutdown_failed",
                    error_type=type(exc).__name__,
                    detail=str(exc)[:2_000],
                )
        if worker is not None:
            while not worker.join(timeout=5.0):
                exit_code = max(exit_code, 4)
                _emit(
                    "rlpd_learner_waiting_for_worker_shutdown",
                    detail=(
                        "the non-daemon JAX worker is still completing its "
                        "current update; logger remains open"
                    ),
                )
        for signal_number, handler in old_handlers.items():
            try:
                signal.signal(signal_number, handler)
            except Exception as exc:
                exit_code = max(exit_code, 5)
                _emit(
                    "rlpd_learner_signal_restore_failed",
                    signal_number=signal_number,
                    error_type=type(exc).__name__,
                    detail=str(exc)[:2_000],
                )
        if service is not None and worker is not None:
            status = worker.status
            _log_best_effort(
                logger,
                "learner_process_stopped",
                learner_step=status.learner_step,
                gradient_step=status.gradient_step,
                policy_version=status.policy_version,
                worker_state=status.state,
                exit_code=exit_code,
            )
        if not logger_closed:
            try:
                logger.close()
            except Exception as exc:
                exit_code = max(exit_code, 5)
                _emit(
                    "rlpd_learner_logger_close_failed",
                    error_type=type(exc).__name__,
                    detail=str(exc)[:2_000],
                )
    if args.synthetic_e2e:
        if synthetic_event_fields is None:
            synthetic_event_fields = {
                "target_learner_step": args.target_learner_step,
                "start_learner_step": prepared_state.learner_step,
                "start_gradient_step": prepared_state.gradient_step,
                "start_policy_version": prepared_state.policy_version,
                "fingerprint_sha256": fingerprint.sha256,
            }
        checkpoint_verified = False
        if synthetic_candidate and exit_code == 0:
            try:
                restored = checkpoint_manager.load(
                    agent_template=agent_template,
                    fingerprint=fingerprint,
                    path=synthetic_event_fields["checkpoint_path"],
                )
                if (
                    restored.learner_step != args.target_learner_step
                    or restored.gradient_step
                    != prepared_state.gradient_step + config.cta_ratio
                    or restored.policy_version
                    != prepared_state.policy_version + 1
                ):
                    raise RuntimeError(
                        "round-trip checkpoint counters do not match the "
                        "completed synthetic E2E step"
                    )
                feature_extractor.validate_agent_invariant(restored.agent)
                checkpoint_verified = True
            except Exception as exc:
                synthetic_event_fields["checkpoint_roundtrip_error"] = (
                    f"{type(exc).__name__}: {str(exc)[:2_000]}"
                )
                exit_code = max(exit_code, 6)
        synthetic_event_fields["checkpoint_roundtrip_verified"] = (
            checkpoint_verified
        )
        if synthetic_candidate and checkpoint_verified and exit_code == 0:
            _emit(
                "rlpd_learner_synthetic_e2e_passed",
                **synthetic_event_fields,
            )
        else:
            exit_code = max(exit_code, 6)
            synthetic_event_fields["exit_code"] = exit_code
            _emit(
                "rlpd_learner_synthetic_e2e_failed",
                **synthetic_event_fields,
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
