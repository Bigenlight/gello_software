#!/usr/bin/env python3
"""Run the single-process local gRPC ingress and HIL-SERL learner."""

from __future__ import annotations

import argparse
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
    CanonicalTransitionPool,
    CheckpointManager,
    CheckpointRunLock,
    FaultGatedReplayIngress,
    JsonlWandbLogger,
    LearnerConfig,
    LearnerFingerprint,
    LearnerWorker,
    build_actor_service,
    compose_learner,
    create_hybrid_sac_agent,
    default_resnet_source,
    file_sha256,
    load_demo_pickles,
    preflight_checkpoint_run,
    prepare_learner_state,
    validate_learner_dependencies,
)
from ur_env.learner.demo import SYNTHETIC_ACCEPTANCE_ONLY_KEY  # noqa: E402
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)
from ur_env.rlpd_receive_server import (  # noqa: E402
    DEFAULT_INTERVENTION_CAPACITY,
    DEFAULT_REPLAY_CAPACITY,
    DEFAULT_REWARD_THRESHOLD,
    ReplayIngress,
    RewardClassifierRuntime,
)


DEFAULT_CLASSIFIER_CHECKPOINT_SHA256 = (
    "e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997"
)


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
    parser.add_argument("--reward-model-id")
    parser.add_argument(
        "--reward-threshold", type=float, default=DEFAULT_REWARD_THRESHOLD
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
        "--grasp-penalty",
        type=float,
        default=-0.02,
        help=(
            "configured redundant gripper command penalty; learner data may "
            "contain only 0 or this non-positive value"
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
            "omit for continuous training"
        ),
    )
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and smoke every component, log preflight, but do not bind",
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
        "max_workers",
        "max_message_bytes",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.target_learner_step is not None and args.target_learner_step < 0:
        raise ValueError("target_learner_step must be non-negative")
    if (
        not math.isfinite(args.checkpoint_reserve_gib)
        or args.checkpoint_reserve_gib < 0.0
    ):
        raise ValueError("checkpoint_reserve_gib must be finite and non-negative")
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0.0:
        raise ValueError("poll_interval must be positive and finite")
    if not math.isfinite(args.grasp_penalty) or args.grasp_penalty > 0.0:
        raise ValueError("grasp_penalty must be finite and non-positive")


def _validate_jax_backend(actual: str, required: str) -> str:
    actual = str(actual).strip().lower()
    if actual != required:
        raise RuntimeError(
            f"JAX backend is {actual!r}, but {required!r} is required"
        )
    return actual


def _validate_demo_serving_scope(demos: object, *, dry_run: bool) -> int:
    """Reject synthetic acceptance artifacts before allocating live services."""

    sidecars = getattr(demos, "sidecars", None)
    if sidecars is None:
        raise TypeError("loaded demos must expose provenance sidecars")
    synthetic = tuple(
        sidecar
        for sidecar in sidecars
        if sidecar.metadata.get(SYNTHETIC_ACCEPTANCE_ONLY_KEY) is True
    )
    if synthetic and not dry_run:
        first = synthetic[0]
        raise ValueError(
            "synthetic acceptance-only demo data is permitted only with "
            "--dry-run; real learner serving requires robot demo data "
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
    config = LearnerConfig(wandb_mode=args.wandb_mode)
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
    )
    _validate_demo_grasp_penalty(demos, expected=args.grasp_penalty)
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
        "contract_revision": "raw_pixels_hybrid_sac_v1",
        "augmentation": "random_crop_pad4",
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
        },
        "offline_demo_sha256": demo_sha256,
        "offline_demo_transition_count": len(demos.transitions),
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
    agent_template = create_hybrid_sac_agent(
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
    raw_ingress = ReplayIngress(
        replay_capacity=args.replay_capacity,
        intervention_capacity=args.intervention_capacity,
        hil_serl_root=args.hil_serl_root,
        learner_mode=True,
        expected_grasp_penalty=args.grasp_penalty,
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
            "demo_count": len(demos.transitions),
            "synthetic_acceptance_demo_count": synthetic_demo_count,
            "observation_schema_hash": CANONICAL_OBSERVATION_SCHEMA_HASH,
            "dependencies": versions,
            "jax_backend": jax_backend,
            "checkpoint_reserve_bytes": (
                checkpoint_manager.minimum_free_bytes_after_save
            ),
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
    try:
        offline_pool = CanonicalTransitionPool(
            demos.transitions,
            seed=config.seed,
        )
        assembly = compose_learner(
            agent_template=agent_template,
            ingress=ingress,
            offline_demos=offline_pool,
            checkpoint_manager=checkpoint_manager,
            fingerprint=fingerprint,
            config=config,
            logger=logger,
            prepared_state=prepared_state,
        )
        service = build_actor_service(
            assembly=assembly,
            classifier=classifier,
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
            demo_count=len(demos.transitions),
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
                demo_count=len(demos.transitions),
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
        _emit(
            "rlpd_learner_server_ready",
            host=args.host,
            port=bound_port,
            policy_version=assembly.policy_runtime.policy_version,
            learner_step=assembly.learner.learner_step,
            reward_model_id=classifier.reward_model_id,
            replay_capacity=raw_ingress.replay_capacity,
            intervention_capacity=raw_ingress.intervention_capacity,
            demo_count=len(demos.transitions),
            fingerprint_sha256=fingerprint.sha256,
            jax_backend=jax_backend,
            jax_device_count=len(jax.devices()),
            persistence="learner_checkpoints_only_replay_ram",
        )

        learner_fault_reported = False
        while not shutdown_event.wait(args.poll_interval):
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
