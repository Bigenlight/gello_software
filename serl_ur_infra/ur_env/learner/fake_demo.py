"""Deterministic canonical demo artifact for local acceptance runs.

This helper is intentionally separate from production data collection.  It
creates a tiny, trusted pickle for acceptance checks without a real robot demo.
Every item carries a synthetic provenance marker, and the production learner
CLI accepts the artifact only with ``--dry-run`` or the bounded
``--synthetic-e2e`` acceptance mode.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle

import numpy as np

from ur_env.learner.demo import (
    SYNTHETIC_ACCEPTANCE_ONLY_KEY,
    load_demo_object,
)


FAKE_DEMO_PICKLE_PROTOCOL = 4


def _observation(value: int) -> dict[str, np.ndarray]:
    return {
        "state": np.full((1, 19), value / 255.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _transition(
    value: int,
    *,
    reward: float,
    penalty: float,
) -> dict[str, object]:
    action = np.zeros((7,), dtype=np.float32)
    action[:6] = np.float32((value % 5) / 10.0)
    action[-1] = np.float32((-1.0, 0.0, 1.0)[value % 3])
    terminal = bool(reward)
    return {
        "observations": _observation(value),
        "next_observations": _observation(value + 1),
        "actions": action,
        "rewards": np.float32(reward),
        "masks": np.float32(0.0 if terminal else 1.0),
        "dones": terminal,
        "grasp_penalty": np.float32(penalty),
        "has_grasp_penalty": np.uint8(1),
        SYNTHETIC_ACCEPTANCE_ONLY_KEY: True,
    }


def build_fake_demo_payload() -> list[dict[str, object]]:
    """Return fixed flat and actor-backup examples accepted by the loader."""

    successful_flat = _transition(2, reward=1.0, penalty=0.0)
    successful_flat.update(
        success=True,
        episode_id=17,
        step_id=3,
    )

    actor_transition = _transition(3, reward=0.0, penalty=-0.02)
    actor_transition.update(
        classifier_success=np.uint8(0),
        episode_id=18,
        step_id=0,
    )
    actor_backup = {
        "meta": {
            "run_id": "fake-acceptance-run",
            "actor_id": "fake-actor",
            "timestamp_ns": 1_234_567_890,
            "intervened": 1,
            "policy_action": np.zeros((7,), dtype=np.float32),
            SYNTHETIC_ACCEPTANCE_ONLY_KEY: True,
        },
        "transition": actor_transition,
    }

    payload = [successful_flat, actor_backup]
    # Fail before touching the requested output if this helper ever drifts from
    # the same strict contract used by the learner CLI.
    load_demo_object(payload, source_path="<generated-fake-demo>")
    return payload


def write_fake_demo_pickle(path: os.PathLike[str] | str) -> Path:
    """Write the canonical fake demo once, without replacing an existing file."""

    output = Path(path).expanduser().resolve()
    payload = build_fake_demo_payload()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        pickle.dump(payload, stream, protocol=FAKE_DEMO_PICKLE_PROTOCOL)
    return output


__all__ = [
    "FAKE_DEMO_PICKLE_PROTOCOL",
    "build_fake_demo_payload",
    "write_fake_demo_pickle",
]
