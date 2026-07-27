#!/usr/bin/env python3
"""Robot-laptop actor with separate inference and transition connections."""

from __future__ import annotations

import argparse
import importlib
import socket
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "hil-serl" / "examples"))
sys.path.insert(
    0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher")
)

from ur_env.remote_policy import (  # noqa: E402
    GET_ACTION_REQUEST,
    RemotePolicyClient,
)
from ur_env.rlpd_actor import run_actor  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--ur-config-module",
        help="Optional module exporting CONFIG_MAPPING for the UR task.",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--client-id", default=socket.gethostname())
    parser.add_argument("--learner-ip", default="127.0.0.1")
    parser.add_argument("--learner-port", type=int, default=5588)
    parser.add_argument("--learner-broadcast-port", type=int, default=5589)
    parser.add_argument("--inference-ip", default="127.0.0.1")
    parser.add_argument("--inference-port", type=int, default=5590)
    parser.add_argument("--inference-broadcast-port", type=int, default=5591)
    parser.add_argument(
        "--transition-flush-s",
        type=int,
        default=1,
        help="Agentlace transition upload interval in seconds.",
    )
    parser.add_argument("--timeout-ms", type=int, default=3000)
    return parser.parse_args()


def _trainer_config(*, port: int, broadcast_port: int, request_types):
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(
        port_number=port,
        broadcast_port=broadcast_port,
        request_types=list(request_types),
    )


def main() -> int:
    args = _parse_args()
    if args.transition_flush_s <= 0:
        raise ValueError("transition-flush-s must be positive")

    from agentlace.data.data_store import QueuedDataStore
    from agentlace.trainer import TrainerClient
    from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
    from experiments.mappings import CONFIG_MAPPING

    if args.ur_config_module:
        module = importlib.import_module(args.ur_config_module)
        CONFIG_MAPPING.update(module.CONFIG_MAPPING)
    if args.exp_name not in CONFIG_MAPPING:
        raise KeyError(f"unknown experiment {args.exp_name!r}")

    config = CONFIG_MAPPING[args.exp_name]()
    env = RecordEpisodeStatistics(
        config.get_environment(
            fake_env=False,
            save_video=args.save_video,
            classifier=True,
        )
    )

    replay_store = QueuedDataStore(50000)
    intervention_store = QueuedDataStore(50000)
    learner_config = _trainer_config(
        port=args.learner_port,
        broadcast_port=args.learner_broadcast_port,
        request_types=("send-stats",),
    )
    transition_client = TrainerClient(
        "actor_env",
        args.learner_ip,
        learner_config,
        data_stores={
            "actor_env": replay_store,
            "actor_env_intvn": intervention_store,
        },
        wait_for_server=True,
        timeout_ms=args.timeout_ms,
    )

    inference_config = _trainer_config(
        port=args.inference_port,
        broadcast_port=args.inference_broadcast_port,
        request_types=(GET_ACTION_REQUEST,),
    )
    inference_transport = TrainerClient(
        "policy_inference",
        args.inference_ip,
        inference_config,
        data_stores={},
        wait_for_server=True,
        timeout_ms=args.timeout_ms,
    )
    policy_client = RemotePolicyClient(
        inference_transport.request,
        client_id=args.client_id,
        action_shape=env.action_space.shape,
    )

    transition_client.start_async_update(interval=args.transition_flush_s)
    try:
        run_actor(
            policy_client,
            replay_store,
            intervention_store,
            transition_client,
            env,
            config=config,
            checkpoint_path=args.checkpoint_path,
        )
    finally:
        transition_client.update()
        transition_client.stop()
        inference_transport.stop()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
