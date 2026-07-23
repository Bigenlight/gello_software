import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_nonnegative_int_env(monkeypatch):
    monkeypatch.delenv("ROUNDTRIP_WARMUP_REQUESTS", raising=False)
    assert smoke.nonnegative_int_env("ROUNDTRIP_WARMUP_REQUESTS", 0) == 0
    monkeypatch.setenv("ROUNDTRIP_WARMUP_REQUESTS", "24")
    assert smoke.nonnegative_int_env("ROUNDTRIP_WARMUP_REQUESTS", 0) == 24


@pytest.mark.parametrize("value", ["-1", "1.5", "not-a-number"])
def test_nonnegative_int_env_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("ROUNDTRIP_WARMUP_REQUESTS", value)
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        smoke.nonnegative_int_env("ROUNDTRIP_WARMUP_REQUESTS", 0)


def test_boolean_env(monkeypatch):
    monkeypatch.delenv("ROUNDTRIP_INCLUDE_REQUESTS", raising=False)
    assert smoke.boolean_env("ROUNDTRIP_INCLUDE_REQUESTS", True) is True
    monkeypatch.setenv("ROUNDTRIP_INCLUDE_REQUESTS", "0")
    assert smoke.boolean_env("ROUNDTRIP_INCLUDE_REQUESTS", True) is False
    monkeypatch.setenv("ROUNDTRIP_INCLUDE_REQUESTS", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        smoke.boolean_env("ROUNDTRIP_INCLUDE_REQUESTS", True)


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


def test_main_excludes_warmups_and_omits_records_in_compact_mode(
    monkeypatch, capsys
):
    class FakeWorker:
        def __init__(self):
            self.next_request_id = 1

        def get_server_info(self):
            return SimpleNamespace(
                resize_height=720,
                resize_width=1280,
                checkpoint_revision="sha256:test",
                model_id="multi_task_dit",
                scheduler="multi_task_dit:flow_matching",
                num_inference_steps=10,
                n_action_steps=24,
            )

        def reset_episode(self):
            return "test-session"

        def start(self):
            pass

        def submit(self, observation):
            self.observation = observation

        def error(self):
            return None

        def take_result(self, max_age_s):
            request_id = self.next_request_id
            self.next_request_id += 1
            return SimpleNamespace(
                action=(0.0,) * 7,
                chunk_refill=request_id in (1, 4),
                inference_ms=float(request_id),
                observation_age_s=0.01,
                preprocess_ms=2.0,
                remaining_chunk_actions=23,
                request_id=request_id,
                total_server_ms=3.0,
            )

        def close(self):
            pass

    monkeypatch.setenv("ROUNDTRIP_WARMUP_REQUESTS", "2")
    monkeypatch.setenv("ROUNDTRIP_REQUESTS", "3")
    monkeypatch.setenv("ROUNDTRIP_INCLUDE_REQUESTS", "0")
    monkeypatch.setattr(smoke, "create_worker", lambda *args, **kwargs: FakeWorker())
    monkeypatch.setattr(smoke, "make_black_jpeg", lambda height, width: b"jpeg")

    assert smoke.main() == 0
    output = json.loads(capsys.readouterr().out)

    assert output["warmup_request_count"] == 2
    assert output["request_count"] == 3
    assert "requests" not in output
    assert output["summary"]["all"]["count"] == 3
    # Warm-up IDs 1 and 2 are absent. Measured IDs 3, 4, 5 include one refill.
    assert output["summary"]["refill"]["count"] == 1
    assert output["summary"]["refill"]["inference_ms"]["mean"] == 4.0
