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
