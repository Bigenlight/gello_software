"""Dependency-light checks for receive-server deployment probes."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest


_SCRIPT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "scripts", "run_rlpd_receive_server.py"
    )
)
_SPEC = importlib.util.spec_from_file_location("run_rlpd_receive_server", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _batch(batch_size=2):
    return {
        "observations": {
            "cam1": np.zeros((batch_size, 2, 128, 128, 3), np.uint8),
            "cam2": np.zeros((batch_size, 2, 128, 128, 3), np.uint8),
            "state": np.zeros((batch_size, 1, 19), np.float32),
        },
        "actions": np.zeros((batch_size, 7), np.float32),
        "policy_actions": np.zeros((batch_size, 7), np.float32),
    }


def test_sample_probe_validates_only_shapes_and_returns_summary():
    summary = _MODULE._validate_sample_batch(_batch(), batch_size=2)

    assert summary == {
        "batch_size": 2,
        "cam1_shape": [2, 2, 128, 128, 3],
        "cam2_shape": [2, 2, 128, 128, 3],
        "state_shape": [2, 1, 19],
        "action_shape": [2, 7],
    }


def test_sample_probe_rejects_noncanonical_packed_batch():
    batch = _batch()
    batch["observations"]["state"] = np.zeros((2, 19), np.float32)

    with pytest.raises(RuntimeError, match="observation shapes mismatch"):
        _MODULE._validate_sample_batch(batch, batch_size=2)


def test_ipv6_loopback_bind_is_bracketed():
    assert _MODULE._grpc_bind_address("127.0.0.1", 50053) == "127.0.0.1:50053"
    assert _MODULE._grpc_bind_address("::1", 50053) == "[::1]:50053"


def test_required_jax_backend_fails_closed_on_cpu_fallback():
    assert _MODULE._validate_jax_backend("GPU", "gpu") == "gpu"
    assert _MODULE._validate_jax_backend("cpu", "any") == "cpu"

    with pytest.raises(RuntimeError, match="'gpu' is required"):
        _MODULE._validate_jax_backend("cpu", "gpu")
