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
from ur_env.envs.wrappers import (  # noqa: E402
    wrap_gripper_penalty_from_task_config,
)
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
    assert_actor_environment_state_layout,
)
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
    parser.add_argument("--observation-schema-hash")
    parser.add_argument("--expected-model-id")
    parser.add_argument("--expected-reward-authority")
    parser.add_argument("--expected-reward-model-id")
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
        "observation_schema_hash": args.observation_schema_hash,
        "expected_model_id": args.expected_model_id,
        "expected_reward_authority": args.expected_reward_authority,
        "expected_reward_model_id": args.expected_reward_model_id,
    }
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    result.setdefault("type", "grpc")
    result.setdefault("host", "127.0.0.1")
    result.setdefault("port", 50053)
    result.setdefault("timeout_s", 0.6)
    result.setdefault("max_response_age_s", 0.8)
    result.setdefault("retry_count", 1)
    result.setdefault(
        "observation_schema_hash", CANONICAL_OBSERVATION_SCHEMA_HASH
    )
    return result


def _build_actor_environment(config: Any, args: argparse.Namespace) -> Any:
    """Build the robot-local wrapper chain with an explicit task penalty."""

    from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

    task_env = config.get_environment(
        fake_env=args.fake_env,
        save_video=args.save_video,
        # Reward/termination is authoritative on the remote server.
        classifier=False,
    )
    try:
        assert_actor_environment_state_layout(task_env)
    except Exception:
        task_env.close()
        raise
    task_env = wrap_gripper_penalty_from_task_config(
        task_env,
        experiment_config=config,
    )
    return EnvTimestampAdapter(RecordEpisodeStatistics(task_env))


def main() -> int:
    args = _parse_args()
    from experiments.mappings import CONFIG_MAPPING

    if args.ur_config_module:
        module = importlib.import_module(args.ur_config_module)
        CONFIG_MAPPING.update(module.CONFIG_MAPPING)
    if args.exp_name not in CONFIG_MAPPING:
        raise KeyError(f"unknown experiment {args.exp_name!r}")

    config = CONFIG_MAPPING[args.exp_name]()
    env = _build_actor_environment(config, args)
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
