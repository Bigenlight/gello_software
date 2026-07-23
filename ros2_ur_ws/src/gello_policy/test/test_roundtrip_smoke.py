import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "remote_diffusion_roundtrip_smoke.py"
SPEC = importlib.util.spec_from_file_location("remote_diffusion_roundtrip_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(smoke)


def test_positive_int_env(monkeypatch):
    monkeypatch.delenv("ROUNDTRIP_REQUESTS", raising=False)
    assert smoke.positive_int_env("ROUNDTRIP_REQUESTS", 1) == 1
    monkeypatch.setenv("ROUNDTRIP_REQUESTS", "49")
    assert smoke.positive_int_env("ROUNDTRIP_REQUESTS", 1) == 49


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "not-a-number"])
def test_positive_int_env_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("ROUNDTRIP_REQUESTS", value)
    with pytest.raises(ValueError, match="must be a positive integer"):
        smoke.positive_int_env("ROUNDTRIP_REQUESTS", 1)


def test_latency_summary_splits_refill_requests_and_computes_percentiles():
    records = []
    for value, refill in ((10.0, True), (20.0, False), (30.0, False)):
        records.append(
            {
                "chunk_refill": refill,
                "round_trip_ms": value,
                "preprocess_ms": value,
                "inference_ms": value,
                "total_server_ms": value,
                "observation_age_ms": value,
            }
        )

    summary = smoke.latency_summary(records)

    assert summary["all"]["count"] == 3
    assert summary["all"]["round_trip_ms"]["p50"] == 20.0
    assert summary["all"]["round_trip_ms"]["p95"] == pytest.approx(29.0)
    assert summary["all"]["round_trip_ms"]["p99"] == pytest.approx(29.8)
    assert summary["refill"]["count"] == 1
    assert summary["refill"]["inference_ms"]["p99"] == 10.0
    assert summary["non_refill"]["count"] == 2
    assert summary["non_refill"]["round_trip_ms"]["mean"] == 25.0


def test_latency_summary_handles_empty_refill_group():
    record = {
        "chunk_refill": False,
        "round_trip_ms": 10.0,
        "preprocess_ms": 2.0,
        "inference_ms": 3.0,
        "total_server_ms": 5.0,
        "observation_age_ms": 11.0,
    }

    summary = smoke.latency_summary([record])

    assert summary["refill"]["count"] == 0
    assert summary["refill"]["round_trip_ms"]["p99"] is None
