"""Tests for the shared laptop/server observation contract."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import ActorProtocolError  # noqa: E402
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
    observation_schema_document,
    validate_canonical_observation,
)


def _observation():
    return {
        "state": np.zeros((1, 19), dtype=np.float32),
        "cam1": np.zeros((1, 128, 128, 3), dtype=np.uint8),
        "cam2": np.zeros((1, 128, 128, 3), dtype=np.uint8),
    }


def test_canonical_observation_validates_and_copies():
    source = _observation()

    result = validate_canonical_observation(source)

    assert set(result) == {"state", "cam1", "cam2"}
    assert all(result[key].flags.c_contiguous for key in result)
    assert all(result[key] is not source[key] for key in result)
    assert len(CANONICAL_OBSERVATION_SCHEMA_HASH) == 64
    assert observation_schema_document()["schema_id"].endswith("-v1")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda obs: obs.pop("cam2"), "keys mismatch"),
        (
            lambda obs: obs.__setitem__(
                "state", np.zeros((1, 18), dtype=np.float32)
            ),
            "shape",
        ),
        (
            lambda obs: obs.__setitem__(
                "cam1", np.zeros((1, 128, 128, 3), dtype=np.float32)
            ),
            "dtype",
        ),
        (
            lambda obs: obs["state"].__setitem__((0, 0), np.nan),
            "non-finite",
        ),
    ],
)
def test_canonical_observation_rejects_contract_drift(mutate, message):
    observation = _observation()
    mutate(observation)

    with pytest.raises(ActorProtocolError, match=message):
        validate_canonical_observation(observation)
