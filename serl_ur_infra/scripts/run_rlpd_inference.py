#!/usr/bin/env python3
"""Server-side RLPD inference process fed by learner parameter broadcasts."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import threading
from pathlib import Path

# Avoid a second JAX process reserving all GPU memory next to the learner.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "hil-serl" / "examples"))
sys.path.insert(
    0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher")
)

from ur_env.remote_policy import (  # noqa: E402
    GET_ACTION_REQUEST,
    RemotePolicyService,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--ur-config-module")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--learner-ip", default="127.0.0.1")
    parser.add_argument("--learner-port", type=int, default=5588)
    parser.add_argument("--learner-broadcast-port", type=int, default=5589)
    parser.add_argument("--inference-port", type=int, default=5590)
    parser.add_argument("--inference-broadcast-port", type=int, default=5591)
    return parser.parse_args()


def _trainer_config(*, port: int, broadcast_port: int, request_types):
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(
        port_number=port,
        broadcast_port=broadcast_port,
        request_types=list(request_types),
    )


def _build_agent(config, env, seed):
    from serl_launcher.utils.launcher import (
        make_sac_pixel_agent,
        make_sac_pixel_agent_hybrid_dual_arm,
        make_sac_pixel_agent_hybrid_single_arm,
    )

    kwargs = dict(
        seed=seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
    )
    if config.setup_mode in (
        "single-arm-fixed-gripper",
        "dual-arm-fixed-gripper",
    ):
        return make_sac_pixel_agent(**kwargs)
    if config.setup_mode == "single-arm-learned-gripper":
        return make_sac_pixel_agent_hybrid_single_arm(**kwargs)
    if config.setup_mode == "dual-arm-learned-gripper":
        return make_sac_pixel_agent_hybrid_dual_arm(**kwargs)
    raise NotImplementedError(f"unknown setup mode: {config.setup_mode}")


class _VersionedJaxPolicy:
    def __init__(self, agent, seed):
        import jax

        self._jax = jax
        self._agent = agent
        self._rng = jax.random.PRNGKey(seed)
        self._version = 0
        self._lock = threading.Lock()

    def sample_action(self, observation, deterministic):
        import numpy as np

        with self._lock:
            self._rng, key = self._jax.random.split(self._rng)
            action = self._agent.sample_actions(
                observations=self._jax.device_put(observation),
                seed=key,
                argmax=bool(deterministic),
            )
            return np.asarray(self._jax.device_get(action)), self._version

    def update_params(self, params):
        with self._lock:
            self._agent = self._agent.replace(
                state=self._agent.state.replace(params=params)
            )
            self._version += 1

def main() -> int:
    args = _parse_args()

    from agentlace.trainer import TrainerClient, TrainerServer
    from experiments.mappings import CONFIG_MAPPING
    from flax.training import checkpoints

    if args.ur_config_module:
        module = importlib.import_module(args.ur_config_module)
        CONFIG_MAPPING.update(module.CONFIG_MAPPING)
    if args.exp_name not in CONFIG_MAPPING:
        raise KeyError(f"unknown experiment {args.exp_name!r}")

    config = CONFIG_MAPPING[args.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=True)
    agent = _build_agent(config, env, args.seed)
    if args.checkpoint_path:
        restored = checkpoints.restore_checkpoint(
            os.path.abspath(args.checkpoint_path), agent.state
        )
        agent = agent.replace(state=restored)
    runtime = _VersionedJaxPolicy(agent, args.seed)

    learner_config = _trainer_config(
        port=args.learner_port,
        broadcast_port=args.learner_broadcast_port,
        request_types=("send-stats",),
    )
    learner_client = TrainerClient(
        "policy_inference",
        args.learner_ip,
        learner_config,
        data_stores={},
        wait_for_server=True,
        timeout_ms=3000,
    )
    learner_client.recv_network_callback(runtime.update_params)

    inference_config = _trainer_config(
        port=args.inference_port,
        broadcast_port=args.inference_broadcast_port,
        request_types=(GET_ACTION_REQUEST,),
    )
    service = RemotePolicyService(
        runtime.sample_action,
        action_shape=env.action_space.shape,
    )
    server = TrainerServer(
        inference_config,
        request_callback=service.handle,
    )
    print(
        f"[rlpd-inference] ready on :{args.inference_port}; "
        f"learner broadcast={args.learner_ip}:{args.learner_broadcast_port}",
        flush=True,
    )
    try:
        server.start(threaded=False)
    finally:
        learner_client.stop()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
