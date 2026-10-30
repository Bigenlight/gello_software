"""A broken metrics sink degrades the logger; it does not stop the learner.

Kept out of ``test_learner_policy_checkpoint.py`` on purpose: that module is
gated behind ``importorskip("jax")``, and none of the behaviour asserted here
needs jax.  The interpreter that actually runs the actor/gRPC stack has no jax,
so these tests would have been silently skipped exactly where the regression
would hurt.  The one case that genuinely needs a live learner --
``train_once`` not faulting on a dead sink -- stays in that module.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _INFRA)

from ur_env.learner.logging import (  # noqa: E402
    LOGGING_WARN_STREAK,
    JsonlWandbLogger,
    LearnerLoggingError,
    _flatten_for_wandb,
)


class _FakeRun:
    def __init__(self):
        self.records = []
        self.finished = False

    def log(self, record, step):
        self.records.append((record, step))

    def finish(self):
        self.finished = True


class _DeadRun(_FakeRun):
    """A W&B run whose service socket is gone, as after a service crash.

    ``ConnectionResetError("Connection lost")`` is not invented for the test:
    it is what asyncio's ``StreamWriter.drain`` raises through
    ``InterfaceSock._publish``, and it is verbatim what the 2026-07-31 learner
    recorded when the W&B service died.
    """

    def __init__(self, *, fail_finish: bool = True):
        super().__init__()
        self.fail_finish = fail_finish
        self.log_calls = 0

    def log(self, record, step):
        self.log_calls += 1
        raise ConnectionResetError("Connection lost")

    def finish(self):
        if self.fail_finish:
            raise ConnectionResetError("Connection lost")
        super().finish()


class _FakeWandb:
    def __init__(self, run):
        self.run = run
        self.kwargs = None

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


def _logger(path, run, warnings):
    return JsonlWandbLogger(
        path,
        wandb_dir=path.parent,
        wandb_module=_FakeWandb(run),
        warn=warnings.append,
    )


def test_dead_wandb_mirror_is_muted_counted_and_leaves_the_jsonl_intact(tmp_path):
    warnings: list[str] = []
    run = _DeadRun()
    path = tmp_path / "learner.jsonl"
    logger = _logger(path, run, warnings)

    for step in range(3):
        logger.log("learner_update", learner_step=step, gradient_step=2 * step)
    logger.close()

    # The record the mirror could not take is still on disk, in full.
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["learner_step"] for record in lines] == [0, 1, 2]
    # Muted after the first failure: the W&B client latches its broken
    # connection and re-raises it, so retrying every step would only cost an
    # exception per learner step.
    assert run.log_calls == 1
    assert logger.wandb_muted
    # finish() also failed, and also did not raise out of close().
    assert logger.wandb_fault_count == 2
    assert "ConnectionResetError" in logger.last_wandb_fault
    assert logger.jsonl_fault_count == 0
    assert logger.degraded
    assert "W&B MUTED" in logger.degraded_detail
    assert len(warnings) == 2
    assert "MUTED" in warnings[0]


def test_a_healthy_close_still_finishes_the_run_and_reports_no_degradation(tmp_path):
    warnings: list[str] = []
    run = _FakeRun()
    logger = _logger(tmp_path / "learner.jsonl", run, warnings)

    logger.log("learner_update", learner_step=7, gradient_step=14)
    logger.close()

    assert run.finished
    assert run.records[0][1] == 7
    assert not logger.degraded
    assert logger.degraded_detail == ""
    assert warnings == []


def test_dead_jsonl_sink_keeps_the_mirror_and_warns_on_the_streak_cadence(tmp_path):
    warnings: list[str] = []
    run = _FakeRun()
    logger = _logger(tmp_path / "learner.jsonl", run, warnings)

    class _DeadStream:
        def write(self, payload):
            raise OSError(28, "No space left on device")

        def flush(self):
            raise AssertionError("flush must not be reached after a failed write")

        def close(self):
            pass

    logger._stream.close()
    logger._stream = _DeadStream()
    for step in range(LOGGING_WARN_STREAK + 1):
        logger.log("learner_update", learner_step=step)
    logger.close()

    # A full disk costs the events, not the run -- and not the dashboard.
    assert logger.jsonl_fault_count == LOGGING_WARN_STREAK + 1
    assert len(run.records) == LOGGING_WARN_STREAK + 1
    assert "No space left" in logger.last_jsonl_fault
    assert "JSONL DEGRADED" in logger.degraded_detail
    assert logger.wandb_fault_count == 0
    # First failure of the streak, then one line per LOGGING_WARN_STREAK.
    assert len(warnings) == 2


def test_a_malformed_record_still_raises_because_it_is_not_a_sink_failure(tmp_path):
    warnings: list[str] = []
    run = _FakeRun()
    logger = _logger(tmp_path / "learner.jsonl", run, warnings)

    with pytest.raises(LearnerLoggingError, match="finite"):
        logger.log("learner_update", learner_step=1, metrics={"loss": float("nan")})
    with pytest.raises(LearnerLoggingError, match="large tensor"):
        logger.log("learner_update", learner_step=1, blob=np.zeros(64, np.float32))
    with pytest.raises(ValueError, match="learner_step"):
        logger.log("learner_update", learner_step="3")
    logger.close()

    # "the record is wrong" is a statement about the learner, not about the
    # sink, so it neither degrades the logger nor gets counted as a sink fault,
    # and nothing partial reaches either sink.
    assert not logger.degraded
    assert warnings == []
    assert run.records == []
    assert (tmp_path / "learner.jsonl").read_text() == ""


def test_a_closed_logger_still_raises_because_that_is_a_caller_bug(tmp_path):
    warnings: list[str] = []
    logger = _logger(tmp_path / "learner.jsonl", _FakeRun(), warnings)
    logger.close()

    with pytest.raises(LearnerLoggingError, match="closed"):
        logger.log("learner_update", learner_step=1)
    assert not logger.degraded


# --------------------------------------------------------------------------
# The W&B mirror is FLATTENED; the JSONL audit trail is NOT.
#
# W&B charts one series per top-level key and stores a nested dict as a single
# opaque value.  The live 2026-07-31 run proved the cost: its whole summary was
# 18 keys, with every training scalar buried in one `metrics` blob, so the
# operator asking for critic/actor losses saw nothing.
# --------------------------------------------------------------------------


def _learner_update_metrics():
    """The nested `metrics` dict `HILSERLLearner.train_once` actually builds.

    Keys copied from `runtime.py`: `_flatten_scalars(critic_info)` under
    `critic/`, `_flatten_scalars(update_info)` under `update/`, then the
    `timing/` and `buffer/` scalars.
    """

    return {
        "critic/critic/critic_loss": 0.42,
        "critic/critic/predicted_qs": -1.25,
        "critic/critic/target_qs": -1.20,
        "critic/grasp_critic/grasp_critic_loss": 0.11,
        "critic/actor_lr": 0.0003,
        "update/actor/actor_loss": -3.5,
        "update/actor/entropy": 1.75,
        "update/temperature/temperature_loss": 0.02,
        "update/actor_lr": 0.0003,
        "timing/sample_ms": 4.0,
        "timing/critic_update_ms": 12.0,
        "timing/full_update_ms": 25.0,
        "timing/learner_step_ms": 30.0,
        "buffer/replay_size": 316.0,
        "buffer/offline_demo_size": 2037.0,
        "buffer/online_intervention_size": 210.0,
        "buffer/intervention_ratio": 0.5,
    }


def test_nested_metrics_reach_wandb_as_one_series_per_leaf(tmp_path):
    warnings: list[str] = []
    run = _FakeRun()
    logger = _logger(tmp_path / "learner.jsonl", run, warnings)

    metrics = _learner_update_metrics()
    logger.log(
        "learner_update",
        learner_step=11,
        gradient_step=22,
        policy_version=3,
        metrics=metrics,
    )
    logger.close()

    mirrored, step = run.records[0]
    assert step == 11
    # No nested value survives: everything W&B is handed is a plottable leaf.
    assert not any(isinstance(value, dict) for value in mirrored.values())
    assert "metrics" not in mirrored
    assert set(mirrored) == {
        "event",
        "time_ns",
        "learner_step",
        "gradient_step",
        "policy_version",
        *(f"metrics/{key}" for key in metrics),
    }
    # Values survive the move unchanged, at the flattened key.
    for key, value in metrics.items():
        assert mirrored[f"metrics/{key}"] == value
    assert mirrored["event"] == "learner_update"
    assert mirrored["policy_version"] == 3


def test_the_jsonl_record_is_byte_for_byte_what_it_was_before_flattening(tmp_path):
    """The audit trail must not move because the dashboard needed a fix.

    `_json_value` shapes this line and other tooling reads it, so the nested
    `metrics` object stays nested and the file never grows a `metrics/...` key.
    """

    warnings: list[str] = []
    run = _FakeRun()
    path = tmp_path / "learner.jsonl"
    logger = _logger(path, run, warnings)

    metrics = _learner_update_metrics()
    logger.log(
        "learner_update",
        learner_step=11,
        gradient_step=22,
        policy_version=3,
        metrics=metrics,
    )
    logger.close()

    raw = path.read_text()
    assert "metrics/" not in raw
    line = raw.splitlines()[0]
    record = json.loads(line)
    assert record["metrics"] == metrics  # still one nested object
    # ...and the line is exactly `json.dumps` of that nested record, i.e. the
    # pre-flattening serialization, `time_ns` included.
    assert line == json.dumps(
        {
            "event": "learner_update",
            "time_ns": record["time_ns"],
            "learner_step": 11,
            "gradient_step": 22,
            "policy_version": 3,
            "metrics": metrics,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def test_flattening_keeps_the_leaf_types_wandb_already_took(tmp_path):
    warnings: list[str] = []
    run = _FakeRun()
    logger = _logger(tmp_path / "learner.jsonl", run, warnings)

    logger.log(
        "learner_update",
        learner_step=1,
        detail="ConnectionResetError",
        count=7,
        ratio=0.5,
        flag=True,
        missing=None,
        series=[1.0, 2.0, 3.0],
        nested={"empty": {}, "deep": {"deeper": {"leaf": 9.0}}},
    )
    logger.close()

    mirrored, _ = run.records[0]
    assert mirrored["detail"] == "ConnectionResetError"
    assert mirrored["count"] == 7
    assert mirrored["ratio"] == 0.5
    assert mirrored["flag"] is True
    assert mirrored["missing"] is None
    # A list is a leaf: W&B takes it as-is, and indexing it would invent series
    # whose shape changes between steps.
    assert mirrored["series"] == [1.0, 2.0, 3.0]
    assert mirrored["nested/deep/deeper/leaf"] == 9.0
    # An empty nested mapping contributes nothing at all -- not a stray key.
    assert not any(key.startswith("nested/empty") for key in mirrored)
    assert "nested" not in mirrored


def test_a_flat_record_is_handed_to_wandb_unchanged(tmp_path):
    """Records with no nesting (`policy_published`, `learner_fault`, ...) are
    untouched, so the flatten cannot regress what already charted."""

    warnings: list[str] = []
    run = _FakeRun()
    logger = _logger(tmp_path / "learner.jsonl", run, warnings)

    logger.log(
        "policy_published", learner_step=50, gradient_step=100, policy_version=1
    )
    logger.close()

    mirrored, _ = run.records[0]
    line = json.loads((tmp_path / "learner.jsonl").read_text().splitlines()[0])
    assert mirrored == line


def test_a_nested_key_colliding_with_a_literal_slash_key_is_last_write_wins():
    """DOCUMENTED CHOICE, not an accident.

    The parent key is consumed by the prefix, so nesting on its own can never
    collide.  A collision needs a caller to pass BOTH `{"a": {"b": ...}}` and a
    literal `"a/b"` in the same record; then the later one in the record's own
    iteration order wins and the other value is absent from the mirror.  We do
    not raise: this is the dashboard path, and the JSONL line still carries
    both values in full.
    """

    assert _flatten_for_wandb({"a": {"b": 1}, "a/b": 2}) == {"a/b": 2}
    assert _flatten_for_wandb({"a/b": 2, "a": {"b": 1}}) == {"a/b": 1}
    # No collision in the shape the learner actually produces: the metric keys
    # already contain "/" and still land under their own parent.
    assert _flatten_for_wandb({"metrics": {"critic/critic_loss": 0.5}}) == {
        "metrics/critic/critic_loss": 0.5
    }


def test_a_record_that_cannot_be_flattened_degrades_only_the_mirror(tmp_path):
    """Flattening runs inside the mirror's `try`, after the JSONL write.

    In production this is unreachable: `_json_value` rebuilds every mapping as
    a plain dict before either sink sees it, so the mirror is only ever handed
    a plain dict tree.  The point of the test is the boundary -- if flattening
    ever did fail it must mute the dashboard, exactly like a dead W&B socket,
    and must not raise into `train_once` or be counted against the JSONL sink.
    """

    warnings: list[str] = []
    run = _FakeRun()
    path = tmp_path / "learner.jsonl"
    logger = _logger(path, run, warnings)

    class _HostileMapping(dict):
        def items(self):
            raise RuntimeError("mapping exploded during flatten")

    logger._mirror_to_wandb(_HostileMapping({"event": "learner_update"}), 1)

    assert run.records == []
    assert logger.wandb_muted
    assert logger.wandb_fault_count == 1
    assert "mapping exploded" in logger.last_wandb_fault
    assert logger.jsonl_fault_count == 0
    assert path.read_text() == ""
    assert len(warnings) == 1 and "MUTED" in warnings[0]
    logger.close()
