#!/usr/bin/env python3
"""Run the robot-laptop actor against a configurable remote transport."""

from __future__ import annotations

import argparse
import importlib
import socket
import sys
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "hil-serl" / "examples"))
sys.path.insert(
    0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher")
)

from ur_env.actor_network import create_actor_network  # noqa: E402
from ur_env.remote_actor import EnvTimestampAdapter, run_remote_actor  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--ur-config-module")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fake-env", action="store_true")
    parser.add_argument("--actor-id", default=socket.gethostname())
    parser.add_argument("--network-type", choices=("grpc", "agentlace"))
    parser.add_argument("--server-host")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--max-response-age-s", type=float)
    return parser.parse_args()


def _network_config(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    configured = getattr(config, "NETWORK", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise TypeError("experiment NETWORK must be a mapping")
    result = dict(configured)
    overrides = {
        "type": args.network_type,
        "host": args.server_host,
        "port": args.server_port,
        "timeout_s": args.timeout_s,
        "max_response_age_s": args.max_response_age_s,
    }
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    result.setdefault("type", "grpc")
    result.setdefault("host", "127.0.0.1")
    result.setdefault("port", 50052)
    result.setdefault("timeout_s", 0.6)
    result.setdefault("max_response_age_s", 0.8)
    result.setdefault("retry_count", 1)
    return result


def main() -> int:
    args = _parse_args()
    from experiments.mappings import CONFIG_MAPPING
    from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

    if args.ur_config_module:
        module = importlib.import_module(args.ur_config_module)
        CONFIG_MAPPING.update(module.CONFIG_MAPPING)
    if args.exp_name not in CONFIG_MAPPING:
        raise KeyError(f"unknown experiment {args.exp_name!r}")

    config = CONFIG_MAPPING[args.exp_name]()
    env = EnvTimestampAdapter(
        RecordEpisodeStatistics(
            config.get_environment(
                fake_env=args.fake_env,
                save_video=args.save_video,
                classifier=True,
            )
        )
    )
    network_config = _network_config(config, args)
    network = create_actor_network(
        network_config,
        actor_id=args.actor_id,
        action_shape=tuple(env.action_space.shape),
    )
    try:
        alive, ready, detail = network.health()
        if not alive or not ready:
            raise RuntimeError(
                f"remote actor server is not ready: alive={alive}, "
                f"ready={ready}, detail={detail}"
            )
        info = network.get_server_info()
        print(
            f"[remote-actor] server={network_config['type']}://"
            f"{network_config.get('host')}:{network_config.get('port')} "
            f"model={info.model_id} protocol={info.protocol_version}",
            flush=True,
        )
        summary = run_remote_actor(
            network,
            env,
            config=config,
            actor_id=args.actor_id,
            checkpoint_path=args.checkpoint_path,
        )
        print(
            f"[remote-actor] run={summary.run_id} steps={summary.env_steps} "
            f"episodes={summary.episodes_started} "
            f"intervention_steps={summary.intervention_steps}",
            flush=True,
        )
    finally:
        network.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
