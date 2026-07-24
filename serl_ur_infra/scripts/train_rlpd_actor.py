#!/usr/bin/env python3
"""Run the UR7e actor while reusing the upstream HIL-SERL bootstrap.

This entry point is intentionally actor-only.  Upstream constructs the task
environment and SAC agent; the local runner owns transition creation, routing,
and persistence so ``third_party/hil-serl`` remains untouched.

Example (after the UR task config is registered in CONFIG_MAPPING):

    python serl_ur_infra/scripts/train_rlpd_actor.py \
        --actor --exp_name <task> --checkpoint_path <path> --ip 127.0.0.1 \
        --ur_config_module <python.module.with.CONFIG_MAPPING>
"""

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "hil-serl" / "examples"))
sys.path.insert(
    0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher")
)

import train_rlpd as upstream  # noqa: E402

from ur_env.rlpd_actor import run_actor  # noqa: E402

upstream.flags.DEFINE_string(
    "ur_config_module",
    None,
    "Optional local module exporting CONFIG_MAPPING; keeps UR task configs "
    "outside third_party/hil-serl.",
)

_upstream_actor = upstream.actor


def _local_actor(agent, replay_store, intervention_store, env, sampling_rng):
    if upstream.FLAGS.eval_checkpoint_step:
        return _upstream_actor(
            agent, replay_store, intervention_store, env, sampling_rng
        )
    return run_actor(
        agent,
        replay_store,
        intervention_store,
        env,
        sampling_rng,
        config=upstream.config,
        flags=upstream.FLAGS,
    )


def _main(argv):
    if not upstream.FLAGS.actor or upstream.FLAGS.learner:
        raise ValueError("this local entry point requires --actor only")
    if upstream.FLAGS.ur_config_module:
        config_module = importlib.import_module(upstream.FLAGS.ur_config_module)
        upstream.CONFIG_MAPPING.update(config_module.CONFIG_MAPPING)
    return upstream.main(argv)


upstream.actor = _local_actor

if __name__ == "__main__":
    upstream.app.run(_main)
