"""Unit tests for the DSRL real-recording compatibility filter."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[3]
MODULE = WORKSPACE / "setup_jazzy" / "dsrl_recording_compat.py"


def _module():
    spec = importlib.util.spec_from_file_location("dsrl_recording_compat_under_test", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hdf5_filter_keeps_numeric_and_drops_json_only_fields() -> None:
    module = _module()
    full = {
        "sampler": "dsrl_det",
        "noise_scale": 0.75,
        "q_evaluated": False,
        "latent_z_f32": [0.1, -0.2],
        "prng_key": np.asarray([1, 2], np.uint32),
        "q_reason": "critic_observations_unavailable",
        "trace_schema": "dsrl-refill-trace/v1",
        "prng": {"before": [1, 2], "after": [3, 4]},
        "missing": None,
    }

    safe, dropped = module.hdf5_safe_diagnostics(full)

    assert set(safe) == {"noise_scale", "q_evaluated", "latent_z_f32", "prng_key"}
    assert dropped == ("missing", "prng", "q_reason", "sampler", "trace_schema")
    assert full["sampler"] == "dsrl_det"
    assert full["prng"] == {"before": [1, 2], "after": [3, 4]}
    np.testing.assert_array_equal(safe["prng_key"], [1, 2])


def test_dsrl_launcher_routes_upstream_through_recording_compat() -> None:
    launcher = (WORKSPACE / "run_ur7e_dsrl_real.sh").read_text()
    assert "DSRL_RECORDING_COMPAT=" in launcher
    assert 'IFQL_SERVER_PY="$DSRL_RECORDING_COMPAT"' in launcher
    assert 'DSRL_UPSTREAM_SERVER_PY="$DSRL_SERVER_PY"' in launcher
