#!/usr/bin/env python3
"""Run the receive-only HIL-SERL gRPC server with real reward inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
sys.path.insert(
    0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher")
)

from ur_env.actor_network import ActorSessionService  # noqa: E402
from ur_env.grpc_actor_transport import create_grpc_server  # noqa: E402
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
)
from ur_env.rlpd_receive_server import (  # noqa: E402
    DEFAULT_INTERVENTION_CAPACITY,
    DEFAULT_REPLAY_CAPACITY,
    DEFAULT_REWARD_THRESHOLD,
    FakeActionRuntime,
    ReplayIngress,
    RewardClassifierRuntime,
    RewardTransitionFinalizer,
)


DEFAULT_CHECKPOINT_SHA256 = (
    "e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256
    )
    parser.add_argument("--reward-model-id")
    parser.add_argument("--threshold", type=float, default=DEFAULT_REWARD_THRESHOLD)
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument(
        "--intervention-capacity",
        type=int,
        default=DEFAULT_INTERVENTION_CAPACITY,
    )
    parser.add_argument(
        "--hil-serl-root",
        default=str(_REPO_ROOT / "third_party" / "hil-serl"),
    )
    parser.add_argument("--policy-version", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-message-bytes", type=int, default=16 * 1024 * 1024
    )
    parser.add_argument(
        "--sample-probe-after",
        type=int,
        default=100,
        help="sample real replay batches after this many inserts; 0 disables",
    )
    parser.add_argument("--sample-probe-batch-size", type=int, default=8)
    parser.add_argument(
        "--require-jax-backend",
        choices=("gpu", "cpu", "any"),
        default="gpu",
        help="fail before serving if JAX selected a different backend",
    )
    return parser.parse_args()


def _emit(event: str, **fields) -> None:
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


def _validate_sample_batch(
    batch: Mapping[str, Any], *, batch_size: int
) -> dict[str, Any]:
    observations = batch["observations"]
    expected_shapes = {
        "cam1": (batch_size, 2, 128, 128, 3),
        "cam2": (batch_size, 2, 128, 128, 3),
        "state": (batch_size, 1, 19),
    }
    actual_shapes = {
        key: tuple(int(dim) for dim in observations[key].shape)
        for key in expected_shapes
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"sample probe observation shapes mismatch: {actual_shapes}"
        )
    action_shape = tuple(int(dim) for dim in batch["actions"].shape)
    policy_action_shape = tuple(
        int(dim) for dim in batch["policy_actions"].shape
    )
    if action_shape != (batch_size, 7) or policy_action_shape != (
        batch_size,
        7,
    ):
        raise RuntimeError("sample probe action shapes mismatch")
    return {
        "batch_size": batch_size,
        "cam1_shape": list(actual_shapes["cam1"]),
        "cam2_shape": list(actual_shapes["cam2"]),
        "state_shape": list(actual_shapes["state"]),
        "action_shape": list(action_shape),
    }


def _grpc_bind_address(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _validate_jax_backend(actual: str, required: str) -> str:
    actual = str(actual).strip().lower()
    required = str(required).strip().lower()
    if not actual:
        raise RuntimeError("JAX reported an empty backend name")
    if required not in {"gpu", "cpu", "any"}:
        raise ValueError(f"unsupported required JAX backend: {required!r}")
    if required != "any" and actual != required:
        raise RuntimeError(
            f"JAX backend is {actual!r}, but {required!r} is required"
        )
    return actual


def main() -> int:
    args = _parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("receive server is loopback-only; use an SSH tunnel")
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    if args.max_workers <= 0 or args.max_message_bytes <= 0:
        raise ValueError("server limits must be positive")
    if args.sample_probe_after < 0 or args.sample_probe_batch_size <= 0:
        raise ValueError("sample probe settings are invalid")

    classifier = RewardClassifierRuntime(
        checkpoint_path=args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        threshold=args.threshold,
        reward_model_id=args.reward_model_id,
        hil_serl_root=args.hil_serl_root,
    )
    if not classifier.ready:
        raise RuntimeError(
            f"reward classifier is not ready: {classifier.fault_detail}"
        )
    # Import lazily so dependency-light CLI unit tests do not initialize JAX.
    # The classifier warm-up above has already selected and exercised the
    # backend; this check prevents an unnoticed CUDA-to-CPU fallback on Kanu.
    import jax

    jax_backend = _validate_jax_backend(
        jax.default_backend(), args.require_jax_backend
    )
    jax_device_count = len(jax.devices())
    ingress = ReplayIngress(
        replay_capacity=args.replay_capacity,
        intervention_capacity=args.intervention_capacity,
        hil_serl_root=args.hil_serl_root,
    )
    fake_action = FakeActionRuntime(policy_version=args.policy_version)
    probe_state = {"completed": args.sample_probe_after == 0}

    def accept_with_sample_probe(data, intervened):
        ingress(data, intervened)
        status = ingress.status()
        if (
            probe_state["completed"]
            or status.replay_insert_count < args.sample_probe_after
        ):
            return
        replay_batch_size = min(
            args.sample_probe_batch_size, status.replay_size
        )
        replay_summary = _validate_sample_batch(
            ingress.sample_replay(batch_size=replay_batch_size),
            batch_size=replay_batch_size,
        )
        intervention_summary = None
        if status.intervention_size:
            intervention_batch_size = min(
                args.sample_probe_batch_size, status.intervention_size
            )
            intervention_summary = _validate_sample_batch(
                ingress.sample_intervention(
                    batch_size=intervention_batch_size
                ),
                batch_size=intervention_batch_size,
            )
        probe_state["completed"] = True
        _emit(
            "rlpd_receive_sample_probe_passed",
            replay=replay_summary,
            intervention=intervention_summary,
            replay_insert_count=status.replay_insert_count,
            intervention_insert_count=status.intervention_insert_count,
        )

    service = ActorSessionService(
        sample_action=fake_action,
        model_id=fake_action.model_id,
        reward_authority="server_classifier",
        reward_model_id=classifier.reward_model_id,
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        finalize_transition=RewardTransitionFinalizer(classifier),
        accept_data=accept_with_sample_probe,
        buffer_status_provider=ingress.status,
    )
    server, bound_port = create_grpc_server(
        service,
        bind_address=_grpc_bind_address(args.host, args.port),
        max_workers=args.max_workers,
        max_message_bytes=args.max_message_bytes,
    )
    server.start()
    _emit(
        "rlpd_receive_server_ready",
        host=args.host,
        port=bound_port,
        model_id=fake_action.model_id,
        policy_version=fake_action.policy_version,
        reward_model_id=classifier.reward_model_id,
        checkpoint_sha256=classifier.checkpoint_sha256,
        threshold=classifier.threshold,
        classifier_warmup_ms=round(classifier.warmup_ms, 3),
        jax_backend=jax_backend,
        jax_device_count=jax_device_count,
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        replay_capacity=ingress.replay_capacity,
        intervention_capacity=ingress.intervention_capacity,
        persistence="ram_only",
        sample_probe_after=args.sample_probe_after,
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2.0).wait()
    status = ingress.status()
    _emit(
        "rlpd_receive_server_stopped",
        replay_size=status.replay_size,
        replay_insert_count=status.replay_insert_count,
        replay_overwrite_count=status.replay_overwrite_count,
        intervention_size=status.intervention_size,
        intervention_insert_count=status.intervention_insert_count,
        intervention_overwrite_count=status.intervention_overwrite_count,
        last_transition_id=status.last_transition_id,
        last_env_step=status.last_env_step,
        sample_probe_completed=probe_state["completed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
