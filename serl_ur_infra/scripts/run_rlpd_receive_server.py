#!/usr/bin/env python3
"""Run the receive-only HIL-SERL gRPC server with real reward inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    return parser.parse_args()


def _emit(event: str, **fields) -> None:
    print(
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def main() -> int:
    args = _parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("receive server is loopback-only; use an SSH tunnel")
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    if args.max_workers <= 0 or args.max_message_bytes <= 0:
        raise ValueError("server limits must be positive")

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
    ingress = ReplayIngress(
        replay_capacity=args.replay_capacity,
        intervention_capacity=args.intervention_capacity,
        hil_serl_root=args.hil_serl_root,
    )
    fake_action = FakeActionRuntime(policy_version=args.policy_version)
    service = ActorSessionService(
        sample_action=fake_action,
        model_id=fake_action.model_id,
        reward_authority="server_classifier",
        reward_model_id=classifier.reward_model_id,
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        finalize_transition=RewardTransitionFinalizer(classifier),
        accept_data=ingress,
        buffer_status_provider=ingress.status,
    )
    server, bound_port = create_grpc_server(
        service,
        bind_address=f"{args.host}:{args.port}",
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
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        replay_capacity=ingress.replay_capacity,
        intervention_capacity=ingress.intervention_capacity,
        persistence="ram_only",
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
