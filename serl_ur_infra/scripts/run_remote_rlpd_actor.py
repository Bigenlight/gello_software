#!/usr/bin/env python3
"""Run the robot-laptop actor against a configurable remote transport."""

from __future__ import annotations

import argparse
import importlib
import math
import socket
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

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
    parser.add_argument(
        "--deadman",
        choices=("topic", "spacebar"),
        default="topic",
        help=(
            "Intervention deadman source. 'topic' (default) follows "
            "/hil/deadman from the HIL GUI: 20 Hz heartbeat with a staleness "
            "watchdog that fails safe to the policy. 'spacebar' is a GLOBAL "
            "pynput listener with no watchdog -- SPACE in any window engages "
            "and it can stick ON. Only 'topic' is safe on the real rig."
        ),
    )
    parser.add_argument(
        "--arm",
        action="store_true",
        help=(
            "Actually publish to the robot (clears the task config's DRY_RUN). "
            "Default is off, in which case neither the arm nor the gripper "
            "moves. WARNING: the UR7e moves physically."
        ),
    )
    parser.add_argument(
        "--mock-policy-noise",
        type=float,
        default=0.0,
        help=(
            "Replace the server's action with zero-mean Gaussian noise of this "
            "sigma, in normalised action units. For bringing the loop up "
            "against a zero-action server: it exercises the full "
            "policy->robot->intervention path before a real policy exists. "
            "The perturbed action is what gets executed AND what gets stored, "
            "so the buffer stays self-consistent. 0 (default) disables it."
        ),
    )
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


def _mock_policy_transform(sigma: float):
    """Gaussian jitter on the pose channels, for bring-up without a policy.

    The server we bring up against returns zero actions, so the robot never
    moves and the policy -> robot -> intervention path cannot be exercised.
    This substitutes a stand-in "policy" of the requested magnitude.

    The gripper channel (index 6) is deliberately left untouched: the demo
    contract requires it to be exactly one of {-1, 0, 1}, and jittering it
    would produce transitions the learner's strict loader rejects.  Pose
    channels are clipped to the [-1, 1] action space.
    """

    if sigma <= 0.0:
        return None
    if not math.isfinite(sigma):
        raise SystemExit("--mock-policy-noise must be finite")

    rng = np.random.default_rng()

    def transform(action):
        jittered = np.asarray(action, dtype=np.float32).copy()
        noise = rng.normal(0.0, sigma, size=jittered[:6].shape)
        jittered[:6] = np.clip(jittered[:6] + noise, -1.0, 1.0)
        return jittered

    print(
        f"[remote-actor] MOCK POLICY: N(0, {sigma}) on the 6 pose channels, "
        "gripper untouched. This is not a trained policy.",
        flush=True,
    )
    return transform


def _build_actor_environment(config: Any, args: argparse.Namespace) -> Any:
    """Build the robot-local wrapper chain with an explicit task penalty."""

    # Import from the package root, not gymnasium.wrappers.record_episode_statistics:
    # gymnasium 1.0 moved the class into gymnasium.wrappers.common and deleted the
    # old per-wrapper module, so the deep path raises ModuleNotFoundError on the
    # gymnasium 1.2.0 we pin.  The package-root name is re-exported by both.
    from gymnasium.wrappers import RecordEpisodeStatistics

    task_env = config.get_environment(
        fake_env=args.fake_env,
        save_video=args.save_video,
        # Reward/termination is authoritative on the remote server.
        classifier=False,
        # A spec, not an object: RosTopicDeadman subscribes on the backend's
        # rclpy node, which does not exist until UR7eEnv has been built.
        deadman=args.deadman,
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


def _load_config_mapping(ur_config_module: str | None) -> dict:
    """Upstream's task registry if it imports, ours either way.

    ``experiments.mappings`` eagerly imports all four Franka task configs, and
    those pull in jax, pyspacemouse, hidapi and pyrealsense2.  The robot laptop
    has none of them on purpose -- inference and learning live on the remote
    GPU, and this process must stay a plain rclpy/numpy actor.  Requiring the
    Franka registry just to look up a UR task name would make jax a hard
    dependency of the robot side.

    So a failed upstream import is downgraded to a warning, but only when a UR
    registry was supplied: without one there is nothing left to look up and the
    original error is the useful message.
    """

    mapping: dict = {}
    try:
        from experiments.mappings import CONFIG_MAPPING as upstream_mapping
    except Exception as exc:  # noqa: BLE001 - any import error is equivalent here
        if not ur_config_module:
            raise
        print(
            f"[remote-actor] upstream experiments.mappings unavailable "
            f"({type(exc).__name__}: {exc}); continuing with "
            f"{ur_config_module} only",
            flush=True,
        )
    else:
        mapping.update(upstream_mapping)

    if ur_config_module:
        module = importlib.import_module(ur_config_module)
        mapping.update(module.CONFIG_MAPPING)
    return mapping


def main() -> int:
    args = _parse_args()
    CONFIG_MAPPING = _load_config_mapping(args.ur_config_module)

    if args.exp_name not in CONFIG_MAPPING:
        raise KeyError(f"unknown experiment {args.exp_name!r}")

    config = CONFIG_MAPPING[args.exp_name]()

    # Must precede _build_actor_environment: DRY_RUN is read once, when
    # UR7eEnv constructs URRosBackend.  Setting it afterwards is a no-op.
    robot_config = getattr(config, "robot_config", None)
    if robot_config is None or not hasattr(robot_config, "DRY_RUN"):
        if args.arm:
            raise SystemExit(
                f"--arm: {args.exp_name} has no robot_config.DRY_RUN to clear"
            )
    else:
        robot_config.DRY_RUN = not args.arm
        if args.arm:
            print(
                "[remote-actor] ARMED: commands go to the robot, the UR7e "
                "moves physically. E-STOP within reach?",
                flush=True,
            )
        else:
            print(
                "[remote-actor] DRY RUN: no command is published (pass --arm "
                "to move the robot).",
                flush=True,
            )

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
            policy_action_transform=_mock_policy_transform(
                args.mock_policy_noise
            ),
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
