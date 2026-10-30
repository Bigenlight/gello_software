"""Tests for the opt-in per-step latency sink.

Three of these are load-bearing, in the sense that the module is worthless --
or worse, harmful -- without them:

* ``test_disabled_profiler_never_touches_the_filesystem`` -- the opt-out path
  must be inert.  ``open`` and ``Path.mkdir`` are booby-trapped for the duration
  of the call, so "no file appeared" is proven by the absence of a call, not by
  an ``iterdir`` that a later flush could still contradict.
* ``test_write_failure_degrades_once_and_never_raises`` -- this sink sits in the
  step path of a live UR7e session; an exception here would end an episode with
  the arm mid-reach.
* ``test_concurrent_commits_produce_one_line_each_with_unique_seq`` -- the gRPC
  servicer serves Step from four handler threads, so interleaved writes must not
  be able to produce a torn line or a duplicated sequence number.

No ROS, no hardware, no network: the module is pure stdlib by design.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _INFRA)

from ur_env import latency_profile  # noqa: E402
from ur_env.latency_profile import (  # noqa: E402
    DIR_ENV_VAR,
    ENABLE_ENV_VAR,
    FLUSH_EVERY_N_COMMITS,
    SCHEMA_VERSION,
    LatencyProfiler,
    LatencyRecord,
    is_profiling_enabled,
)

_LOGGER_NAME = "ur_env.latency_profile"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _read_lines(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _warning_count(caplog) -> int:
    return sum(
        1
        for rec in caplog.records
        if rec.name == _LOGGER_NAME and rec.levelno >= logging.WARNING
    )


class _ExplodingStream:
    """Stands in for a file whose writes fail (full disk, read-only mount)."""

    def __init__(self) -> None:
        self.write_calls = 0
        self.closed = False

    def write(self, _text: str) -> int:
        self.write_calls += 1
        raise OSError(28, "No space left on device")

    def flush(self) -> None:
        raise OSError(28, "No space left on device")

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def enabled_env(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    monkeypatch.delenv(DIR_ENV_VAR, raising=False)


# --------------------------------------------------------------------------- #
# Opt-out path                                                                 #
# --------------------------------------------------------------------------- #
def test_disabled_profiler_never_touches_the_filesystem(tmp_path, monkeypatch):
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("disabled profiler touched the filesystem")

    monkeypatch.setattr(builtins, "open", _boom)
    monkeypatch.setattr(Path, "mkdir", _boom)

    profiler = LatencyProfiler.from_env("actor", out_path=tmp_path)
    assert profiler.enabled is False
    assert profiler.path is None

    record = profiler.record()
    with record.phase("env_step"):
        pass
    record.set("transition_id", "abc")
    record.mark("post_step")
    record.commit()
    record.commit()
    profiler.close()
    profiler.close()

    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_disabled_profiler_hands_out_one_shared_stateless_record(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    profiler = LatencyProfiler.from_env("server")

    first = profiler.record()
    second = profiler.record()
    assert first is second, "disabled profiler must not allocate per step"
    assert first.fields == {}
    first.set("k", 1)
    assert first.fields == {}, "the shared no-op record must stay stateless"
    assert profiler.seq == 0


def test_out_path_is_not_resolved_while_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    monkeypatch.setenv(DIR_ENV_VAR, os.fspath(tmp_path / "should_not_appear"))
    profiler = LatencyProfiler.from_env("actor", out_path=tmp_path / "x.jsonl")
    assert profiler.path is None
    assert not (tmp_path / "should_not_appear").exists()


# --------------------------------------------------------------------------- #
# Enabled path: schema                                                         #
# --------------------------------------------------------------------------- #
def test_enabled_profiler_emits_one_valid_json_line_per_commit(tmp_path, enabled_env):
    out = tmp_path / "runs" / "latency_actor.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)
    assert profiler.enabled is True
    assert profiler.path == out
    assert not out.exists(), "the file must be created lazily, at first commit"

    before = time.time()
    for index in range(3):
        record = profiler.record()
        assert isinstance(record, LatencyRecord)
        with record.phase("env_step"):
            pass
        record.set("env_step_index", index)
        record.commit()
    profiler.close()
    after = time.time()

    lines = _read_lines(out)
    assert len(lines) == 3
    for index, line in enumerate(lines):
        assert line["schema"] == SCHEMA_VERSION
        assert line["role"] == "actor"
        assert line["seq"] == index
        assert before <= line["t_epoch"] <= after
        assert line["env_step_index"] == index
        assert isinstance(line["env_step_ms"], float)


def test_phase_duration_is_positive_and_rounded_to_three_decimals(tmp_path, enabled_env):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)

    record = profiler.record()
    with record.phase("step_rpc"):
        time.sleep(0.02)
    record.commit()
    profiler.close()

    (line,) = _read_lines(out)
    value = line["step_rpc_ms"]
    # A 20 ms sleep can overshoot on a loaded machine but can never come back
    # early, and can never be reported as zero.
    assert 15.0 <= value < 5000.0
    assert round(value, 3) == value


def test_duplicate_phase_names_accumulate(tmp_path, enabled_env):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)

    record = profiler.record()
    for _ in range(3):
        with record.phase("sidecar_encode"):
            time.sleep(0.005)
    record.commit()
    profiler.close()

    (line,) = _read_lines(out)
    assert [key for key in line if key.startswith("sidecar_encode")] == [
        "sidecar_encode_ms"
    ]
    # Three ~5 ms sleeps: the sum, not the last one.
    assert line["sidecar_encode_ms"] >= 12.0


def test_phase_records_partial_time_and_re_raises(tmp_path, enabled_env):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)

    record = profiler.record()
    with pytest.raises(RuntimeError):
        with record.phase("step_rpc"):
            time.sleep(0.005)
            raise RuntimeError("rpc died")
    record.set("failed", True)
    record.commit()
    profiler.close()

    (line,) = _read_lines(out)
    assert line["failed"] is True
    assert line["step_rpc_ms"] >= 3.0


def test_set_values_pass_through_and_mark_is_an_offset(tmp_path, enabled_env):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("server", out_path=out)

    record = profiler.record()
    record.set("transition_id", "t-42")
    record.set("concurrent_rpcs", 3)
    record.set("classifier_degraded", True)
    record.set("server_inference_ms", 12.5)
    record.set("missing", None)
    record.set("shape", [1, 2, 3])
    time.sleep(0.005)
    record.mark("after_decode")
    record.commit()
    profiler.close()

    (line,) = _read_lines(out)
    assert line["transition_id"] == "t-42"
    assert line["concurrent_rpcs"] == 3
    assert line["classifier_degraded"] is True
    assert line["server_inference_ms"] == 12.5
    assert line["missing"] is None
    assert line["shape"] == [1, 2, 3]
    assert line["after_decode_at_ms"] >= 3.0


def test_repeated_set_and_mark_keep_the_last_value(tmp_path, enabled_env):
    profiler = LatencyProfiler.from_env("actor", out_path=tmp_path / "a.jsonl")
    record = profiler.record()
    record.set("env_step", 1)
    record.set("env_step", 2)
    record.mark("t")
    first = record.fields["t_at_ms"]
    time.sleep(0.005)
    record.mark("t")
    assert record.fields["env_step"] == 2
    assert record.fields["t_at_ms"] > first
    profiler.close()


def test_caller_cannot_shadow_the_reserved_keys(tmp_path, enabled_env):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)
    record = profiler.record()
    record.set("role", "impostor")
    record.set("seq", 999)
    record.set("schema", 42)
    record.set("t_epoch", 0.0)
    record.commit()
    profiler.close()

    (line,) = _read_lines(out)
    assert line["role"] == "actor"
    assert line["seq"] == 0
    assert line["schema"] == SCHEMA_VERSION
    assert line["t_epoch"] > 0.0


def test_non_serializable_value_degrades_to_a_string_not_a_dead_sink(
    tmp_path, enabled_env
):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)
    record = profiler.record()
    record.set("weird", object())
    record.commit()
    record2 = profiler.record()
    record2.set("ok", 1)
    record2.commit()
    profiler.close()

    lines = _read_lines(out)
    assert len(lines) == 2
    assert isinstance(lines[0]["weird"], str)
    assert lines[1]["ok"] == 1
    assert profiler.degraded is False


def test_commit_is_single_shot(tmp_path, enabled_env):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)
    record = profiler.record()
    record.commit()
    record.commit()
    # Post-commit mutation must not resurrect the record either.
    record.set("late", True)
    with record.phase("late_phase"):
        pass
    record.commit()
    profiler.close()

    lines = _read_lines(out)
    assert len(lines) == 1
    assert "late" not in lines[0]
    assert "late_phase_ms" not in lines[0]


# --------------------------------------------------------------------------- #
# Failure handling                                                             #
# --------------------------------------------------------------------------- #
def test_write_failure_degrades_once_and_never_raises(tmp_path, enabled_env, caplog):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)
    stream = _ExplodingStream()
    profiler._stream = stream  # simulate a sink that dies mid-session

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        for _ in range(5):
            record = profiler.record()
            with record.phase("env_step"):
                pass
            record.commit()
        profiler.close()

    assert stream.write_calls == 1, "a dead sink must stop being written to"
    assert stream.closed is True
    assert profiler.degraded is True
    assert profiler.enabled is False
    assert _warning_count(caplog) == 1


def test_unwritable_destination_degrades_at_first_commit_not_at_construction(
    tmp_path, enabled_env, caplog
):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory\n", encoding="utf-8")
    out = blocker / "nested" / "a.jsonl"

    # Construction must succeed: a bad path can never fail a session start.
    profiler = LatencyProfiler.from_env("actor", out_path=out)
    assert profiler.enabled is True

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        record = profiler.record()
        record.set("env_step", 0)
        record.commit()
        for _ in range(3):
            profiler.record().commit()
        profiler.close()

    assert profiler.enabled is False
    assert profiler.degraded is True
    assert _warning_count(caplog) == 1
    assert profiler.seq == 0


def test_records_taken_before_a_degrade_still_commit_harmlessly(tmp_path, enabled_env):
    profiler = LatencyProfiler.from_env("actor", out_path=tmp_path / "a.jsonl")
    record = profiler.record()
    profiler._stream = _ExplodingStream()
    profiler.record().commit()  # kills the sink
    assert profiler.enabled is False
    record.set("late", True)
    record.commit()  # must not raise


# --------------------------------------------------------------------------- #
# Buffering / lifecycle                                                        #
# --------------------------------------------------------------------------- #
def test_buffer_flushes_every_n_commits(tmp_path, enabled_env):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)

    for _ in range(FLUSH_EVERY_N_COMMITS - 1):
        profiler.record().commit()
    # These commits take microseconds, so the 5 s timer cannot have fired.
    assert len(_read_lines(out)) == 0

    profiler.record().commit()
    assert len(_read_lines(out)) == FLUSH_EVERY_N_COMMITS

    profiler.record().commit()
    profiler.close()
    assert len(_read_lines(out)) == FLUSH_EVERY_N_COMMITS + 1


def test_elapsed_time_flushes_even_below_the_commit_threshold(
    tmp_path, enabled_env, monkeypatch
):
    monkeypatch.setattr(latency_profile, "FLUSH_INTERVAL_S", 0.0)
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)
    profiler.record().commit()
    assert len(_read_lines(out)) == 1
    profiler.close()


def test_close_is_idempotent_and_flushes(tmp_path, enabled_env):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("actor", out_path=out)
    profiler.record().commit()
    profiler.record().commit()

    profiler.close()
    assert len(_read_lines(out)) == 2
    assert profiler.enabled is False

    profiler.close()
    profiler.close()
    assert len(_read_lines(out)) == 2

    # A record handed out before close must not write after it.
    assert profiler.record() is latency_profile._NULL_RECORD


def test_reopening_the_same_path_appends(tmp_path, enabled_env):
    out = tmp_path / "a.jsonl"
    for _ in range(2):
        profiler = LatencyProfiler.from_env("actor", out_path=out)
        profiler.record().commit()
        profiler.close()
    assert len(_read_lines(out)) == 2


def test_concurrent_commits_produce_one_line_each_with_unique_seq(
    tmp_path, enabled_env
):
    out = tmp_path / "a.jsonl"
    profiler = LatencyProfiler.from_env("server", out_path=out)

    threads_n = 8
    per_thread = 25
    start = threading.Barrier(threads_n)
    errors = []

    def worker(worker_id: int) -> None:
        try:
            start.wait(timeout=10.0)
            for index in range(per_thread):
                record = profiler.record()
                with record.phase("total"):
                    record.set("worker", worker_id)
                    record.set("index", index)
                record.commit()
        except Exception as exc:  # pragma: no cover - surfaced by the assert
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    profiler.close()

    assert errors == []
    lines = _read_lines(out)
    assert len(lines) == threads_n * per_thread
    assert sorted(line["seq"] for line in lines) == list(range(len(lines)))
    assert sorted((line["worker"], line["index"]) for line in lines) == sorted(
        (worker, index)
        for worker in range(threads_n)
        for index in range(per_thread)
    )


# --------------------------------------------------------------------------- #
# Environment resolution                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("  True  ", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("", False),
        ("false", False),
        ("no", False),
        ("off", False),
        # Deliberately strict: anything outside the allowlist is off, so a
        # typo'd value fails closed instead of silently profiling.
        ("2", False),
        ("please", False),
    ],
)
def test_enable_flag_truthiness(monkeypatch, tmp_path, value, expected):
    monkeypatch.setenv(ENABLE_ENV_VAR, value)
    assert is_profiling_enabled() is expected
    profiler = LatencyProfiler.from_env("actor", out_path=tmp_path / "a.jsonl")
    assert profiler.enabled is expected
    profiler.close()


def test_enable_flag_unset_is_off(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    assert is_profiling_enabled() is False


def test_explicit_env_mapping_is_honoured(tmp_path, monkeypatch):
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    profiler = LatencyProfiler.from_env(
        "actor", out_path=tmp_path / "a.jsonl", env={ENABLE_ENV_VAR: "1"}
    )
    assert profiler.enabled is True
    profiler.close()


def test_jsonl_out_path_is_used_verbatim(tmp_path, enabled_env):
    out = tmp_path / "logs" / "latency_server.jsonl"
    profiler = LatencyProfiler.from_env("server", out_path=out)
    assert profiler.path == out


def test_non_jsonl_out_path_is_treated_as_a_directory(tmp_path, enabled_env):
    out_dir = tmp_path / "hil_latency"
    profiler = LatencyProfiler.from_env("actor", out_path=out_dir)
    assert profiler.path.parent == out_dir
    assert re.fullmatch(
        r"\d{8}_\d{6}_actor_\d+\.jsonl", profiler.path.name
    ), profiler.path.name
    assert profiler.path.name.endswith(f"_actor_{os.getpid()}.jsonl")


def test_profile_dir_env_var_overrides_the_call_site_default(tmp_path, monkeypatch):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    override = tmp_path / "operator_choice"
    monkeypatch.setenv(DIR_ENV_VAR, os.fspath(override))

    profiler = LatencyProfiler.from_env("server", out_path=tmp_path / "ignored.jsonl")
    assert profiler.path.parent == override
    assert re.fullmatch(r"\d{8}_\d{6}_server_\d+\.jsonl", profiler.path.name)
    assert not override.exists(), "resolution must not create the directory"

    profiler.record().commit()
    profiler.close()
    assert override.is_dir()
    assert len(_read_lines(profiler.path)) == 1


def test_profile_dir_env_var_expands_user(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    monkeypatch.setenv(DIR_ENV_VAR, "~/hil_latency_test")
    profiler = LatencyProfiler.from_env("actor")
    assert "~" not in os.fspath(profiler.path)
    assert profiler.path.parent == Path(os.path.expanduser("~")) / "hil_latency_test"
    assert not profiler.path.exists()


def test_blank_profile_dir_falls_back_to_the_call_site_default(tmp_path, monkeypatch):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    monkeypatch.setenv(DIR_ENV_VAR, "   ")
    out = tmp_path / "logs" / "latency_server.jsonl"
    profiler = LatencyProfiler.from_env("server", out_path=out)
    assert profiler.path == out


def test_role_appears_in_every_line_and_in_the_generated_name(tmp_path, enabled_env):
    profiler = LatencyProfiler.from_env("server", out_path=tmp_path)
    assert "_server_" in profiler.path.name
    profiler.record().commit()
    profiler.close()
    (line,) = _read_lines(profiler.path)
    assert line["role"] == "server"
