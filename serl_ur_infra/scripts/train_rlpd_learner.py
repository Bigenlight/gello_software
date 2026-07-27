#!/usr/bin/env python3
"""Run the upstream RLPD learner with metadata-preserving replay buffers.

This entry point leaves ``third_party/hil-serl`` untouched.  It replaces only
the replay-buffer constructor used by upstream ``train_rlpd.py`` so
``policy_actions``, ``intervened`` and ``timestamp_ns`` survive Agentlace
upload and are stored in both online and intervention buffers.
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

from ur_env.rlpd_replay_metadata import (  # noqa: E402
    install_metadata_schema,
    normalize_transition_metadata,
)

_UpstreamReplayBuffer = upstream.MemoryEfficientReplayBufferDataStore

upstream.flags.DEFINE_string(
    "ur_config_module",
    None,
    "Optional local module exporting CONFIG_MAPPING; keeps UR task configs "
    "outside third_party/hil-serl.",
)


def _metadata_replay_buffer(*args, **kwargs):
    replay_buffer = install_metadata_schema(_UpstreamReplayBuffer(*args, **kwargs))
    upstream_insert = replay_buffer.insert

    def insert(transition):
        return upstream_insert(normalize_transition_metadata(transition))

    # DataStoreBase.batch_insert() dispatches through self.insert(), so this
    # one instance-level hook covers Agentlace batches and checkpoint/demo
    # restoration without modifying the vendored implementation.
    replay_buffer.insert = insert
    return replay_buffer


def _main(argv):
    if not upstream.FLAGS.learner or upstream.FLAGS.actor:
        raise ValueError("this entry point requires --learner only")
    if upstream.FLAGS.ur_config_module:
        config_module = importlib.import_module(upstream.FLAGS.ur_config_module)
        upstream.CONFIG_MAPPING.update(config_module.CONFIG_MAPPING)
    return upstream.main(argv)


upstream.MemoryEfficientReplayBufferDataStore = _metadata_replay_buffer

if __name__ == "__main__":
    upstream.app.run(_main)
