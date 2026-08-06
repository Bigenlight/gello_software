"""The actor loop's opt-in per-step latency records.

WHAT THESE PIN DOWN
-------------------
``ur_env/latency_profile.py`` is tested on its own in
``test_latency_profile.py``.  What is tested HERE is the wiring: that the
production loop in ``remote_actor._run_remote_actor_impl`` emits exactly one
record per *executed* env step, that the record's ``transition_id`` is the same
string the Step RPC carried (the key the offline analyzer joins the two hosts'
files on -- a record whose id is rebuilt from parts could drift and nobody would
notice), and that the whole thing evaporates when ``HIL_LATENCY_PROFILE`` is
unset.

Three properties get most of the attention because they are the ones that could
hurt a live UR7e session:

* **Disabled means disabled.**  No file, no directory, and -- proven by
  differential comparison against the same scenario run with profiling on --
  no change to a single value the actor ships or returns.
* **The sink is closed on the way out, including the exception path.**  The
  writer buffers, so an unclosed profiler loses its tail; and the sessions worth
  profiling are exactly the ones that die by exception, where the tail is the
  evidence.  These tests read the file *after* the run and would fail on an
  unflushed buffer, which is why they inject a profiler the test still holds a
  reference to (a garbage-collected file object would flush on its own and
  certify nothing).
* **A dead sink degrades; it does not end the episode.**  Same rule as
  ``c86dc54`` for the learner's metrics sink.

``step_rpc_ms`` is asserted to be the round trip the loop already measures for
``_RoundTripStats`` rather than a second measurement of the same call: the two
would disagree under load and the file would be reporting time nothing spent.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    ActionResult,
    ActorSessionService,
    BeginEpisodeCommand,
    ObservationPacket,
    StepCommand,
    StepResult,
    TransitionAck,
    TransitionOutcome,
)
from ur_env.latency_profile import (  # noqa: E402
    DIR_ENV_VAR,
    ENABLE_ENV_VAR,
    SCHEMA_VERSION,
    LatencyProfiler,
)
from ur_env.remote_actor import (  # noqa: E402
    DEFAULT_LATENCY_PROFILE_DIR,
    run_remote_actor,
)


# --------------------------------------------------------------------- #
# Fakes -- shaped after tests/test_actor_abort_lifecycle.py
# --------------------------------------------------------------------- #


class _ActionSpace:
    shape = (7,)


def _observation(marker):
    value = np.uint8(marker % 255)
    return {
        "state": np.full((1, 19), float(marker), dtype=np.float32),
        "cam1": np.full((1, 4, 4, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 4, 4, 3), value, dtype=np.uint8),
    }


def _full_frame(fill: int) -> np.ndarray:
    """A decoded 720p BGR camera frame, the shape ``build_sidecar`` expects."""

    return np.full((720, 1280, 3), fill, dtype=np.uint8)


class _Env:
    """Deterministic env with scriptable per-step ``info`` and terminals."""

    action_space = _ActionSpace()

    def __init__(self, *, infos=None, done_at=(), raise_at=None, cameras=False):
        self.reset_count = 0
        self.step_count = 0
        self.infos = list(infos or [])
        self.done_at = set(done_at)
        self.raise_at = raise_at
        self.cameras = bool(cameras)
        self.frame_calls = 0

    def reset(self, **kwargs):
        self.reset_count += 1
        return _observation(self.reset_count), {
            "timestamp_ns": np.int64(1_000 + self.reset_count)
        }

    def step(self, action):
        index = self.step_count
        if self.raise_at is not None and index == self.raise_at:
            raise RuntimeError("scripted env failure")
        self.step_count += 1
        info = {
            "timestamp_ns": np.int64(2_000 + self.step_count),
            "intervened": 0,
            "held": False,
        }
        if index < len(self.infos):
            info.update(self.infos[index])
        if info.get("intervened") and "intervene_action" not in info:
            info["intervene_action"] = np.asarray(action, dtype=np.float32).copy()
        return (
            _observation(100 + self.step_count),
            0.0,
            index in self.done_at,
            False,
            info,
        )

    def last_camera_frames(self):
        if not self.cameras:
            raise AttributeError("last_camera_frames")
        self.frame_calls += 1
        return {"cam1": _full_frame(11), "cam2": _full_frame(200)}


class _Network:
    """Minimal server double built from the REAL transport dataclasses.

    ``StepResult``/``TransitionAck``/``ActionResult`` rather than
    ``SimpleNamespace`` on purpose: ``request_id`` and ``server_inference_ms``
    are read off these objects by the wiring under test, and a hand-rolled
    namespace would let a renamed production field pass silently here and fail
    on the robot.
    """

    def __init__(self, *, inference_ms=3.5):
        self.begin_calls = []
        self.step_calls = []
        self.inference_ms = float(inference_ms)
        self._request_id = 1

    def _action(self, *, version, session_id, observation_id):
        return ActionResult(
            action=np.zeros(7, dtype=np.float32),
            policy_version=version,
            session_id=session_id,
            request_id=self._request_id,
            request_created_monotonic_ns=0,
            observation_id=observation_id,
            server_inference_ms=self.inference_ms,
            round_trip_ms=0.0,
        )

    def begin_episode(self, observation, **kwargs):
        self._request_id = 1
        self.begin_calls.append((observation, dict(kwargs)))
        action = self._action(
            version=len(self.begin_calls) - 1,
            session_id=kwargs.get("session_id", ""),
            observation_id=kwargs.get("observation_id", ""),
        )
        self._request_id = 2
        return action

    def step(self, next_observation, **kwargs):
        self.step_calls.append((next_observation, dict(kwargs)))
        data = kwargs["data"]
        meta = data["meta"]
        transition = data["transition"]
        done = bool(transition["dones"])
        truncated = bool(transition["truncated"])
        outcome = TransitionOutcome(
            transition_id=meta["transition_id"],
            reward=0.0,
            mask=float(transition["masks"]),
            done=done,
            truncated=truncated,
            success=False,
            classifier_evaluated=False,
        )
        ack = TransitionAck(
            accepted=True,
            transition_id=meta["transition_id"],
            session_id=meta["session_id"],
            request_id=self._request_id,
        )
        action = (
            None
            if (done or truncated)
            else self._action(
                version=1,
                session_id=meta["session_id"],
                observation_id=kwargs.get("next_observation_id", ""),
            )
        )
        self._request_id += 1
        return StepResult(ack=ack, outcome=outcome, action=action)


class _Scheduler:
    """Attaches a sidecar on a fixed set of ``should_attach`` ordinals."""

    def __init__(self, attach_on=()):
        self._attach_on = set(attach_on)
        self.query_count = 0
        self.outcomes = []

    def should_attach(self, state, provisional_terminal):
        index = self.query_count
        self.query_count += 1
        return index in self._attach_on

    def note_outcome(self, outcome):
        self.outcomes.append(outcome)

    def reset(self):
        return None


def _config(max_steps):
    return SimpleNamespace(max_steps=max_steps, random_steps=0, buffer_period=0)


def _run(env, network, *, max_steps=3, run_id="run", **kwargs):
    return run_remote_actor(
        network,
        env,
        config=_config(max_steps),
        actor_id="actor",
        run_id=run_id,
        session_id_factory=iter(("s0", "s1", "s2", "s3", "s4")).__next__,
        **kwargs,
    )


def _enable(monkeypatch, directory):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    monkeypatch.setenv(DIR_ENV_VAR, str(directory))


def _disable(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    monkeypatch.delenv(DIR_ENV_VAR, raising=False)


def _records(directory):
    """Every JSONL line the run wrote, in file order."""

    files = sorted(directory.glob("*.jsonl"))
    assert len(files) == 1, f"expected exactly one profile file, got {files}"
    return [
        json.loads(line)
        for line in files[0].read_text().splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------- #
# One record per executed step
# --------------------------------------------------------------------- #


def test_enabled_run_emits_one_record_per_executed_env_step(
    monkeypatch, tmp_path
):
    _enable(monkeypatch, tmp_path)
    env = _Env()
    network = _Network()

    _run(env, network, max_steps=3)

    records = _records(tmp_path)
    assert len(records) == env.step_count == 3
    assert [record["env_step"] for record in records] == [0, 1, 2]
    assert [record["seq"] for record in records] == [0, 1, 2]
    assert {record["role"] for record in records} == {"actor"}
    assert {record["schema"] for record in records} == {SCHEMA_VERSION}
    assert {record["run_id"] for record in records} == {"run"}


def test_transition_id_is_the_string_the_step_rpc_carried(
    monkeypatch, tmp_path
):
    _enable(monkeypatch, tmp_path)
    env = _Env()
    network = _Network()

    _run(env, network, max_steps=3)

    shipped = [
        call[1]["data"]["meta"]["transition_id"] for call in network.step_calls
    ]
    assert [record["transition_id"] for record in _records(tmp_path)] == shipped


def test_every_required_phase_and_id_is_present(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)

    _run(_Env(), _Network(), max_steps=2)

    for record in _records(tmp_path):
        for key in (
            "run_id",
            "episode_id",
            "env_step",
            "transition_id",
            "request_id",
            "intervened",
            "sidecar_attached",
            "env_step_ms",
            "transition_build_ms",
            "step_rpc_ms",
            "server_inference_ms",
            "t_epoch",
        ):
            assert key in record, f"{key} missing from {record}"
        assert record["step_rpc_ms"] >= 0.0
        assert record["env_step_ms"] >= 0.0
        assert record["intervened"] is False
        assert record["sidecar_attached"] is False


def test_iter_interval_is_absent_on_the_first_record_and_present_after(
    monkeypatch, tmp_path
):
    """It is a delta between iteration STARTS, so record 0 has no predecessor.

    Emitting a zero there would be a lie an analyzer cannot distinguish from a
    genuinely instantaneous loop, so the field is simply absent.
    """

    _enable(monkeypatch, tmp_path)

    _run(_Env(), _Network(), max_steps=4)

    records = _records(tmp_path)
    assert "iter_interval_ms" not in records[0]
    for record in records[1:]:
        assert record["iter_interval_ms"] >= 0.0
    # Start-to-start, so it must cover at least the step it brackets.
    assert records[1]["iter_interval_ms"] >= records[0]["env_step_ms"]


def test_terminal_steps_are_profiled_too(monkeypatch, tmp_path):
    """The episode boundary leaves the iteration by ``continue``/``break``.

    A record committed at the bottom of the loop would drop exactly the steps
    that ended an episode -- the ones an operator investigating a stall looks
    for first.
    """

    _enable(monkeypatch, tmp_path)
    env = _Env(done_at=(0, 2))
    network = _Network()

    _run(env, network, max_steps=3)

    records = _records(tmp_path)
    assert [record["env_step"] for record in records] == [0, 1, 2]
    # Episode 0 is one step long; the terminal Step asked for no action, so the
    # ack is the only place its request_id could have come from.
    assert [record["episode_id"] for record in records] == [0, 1, 1]
    assert records[0]["request_id"] is not None
    assert "server_inference_ms" not in records[0]


# --------------------------------------------------------------------- #
# Disabled means disabled
# --------------------------------------------------------------------- #


def test_disabled_run_creates_no_file_and_no_directory(monkeypatch, tmp_path):
    _disable(monkeypatch)
    target = tmp_path / "never"

    _run(_Env(), _Network(), max_steps=3)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_disabled_run_ships_exactly_what_the_enabled_run_ships(
    monkeypatch, tmp_path
):
    """Differential proof that the instrumentation is observationally inert.

    The two runs are the same scenario with the same deterministic fakes, so
    every transition, every summary field and every reset must match value for
    value.  Anything the profiling path mutated -- an observation it copied, a
    counter it advanced, an ``info`` key it added -- shows up here.
    """

    def once():
        env = _Env(
            infos=[{"intervened": 1}, {}, {"intervention_saturated": True}],
            done_at=(1,),
        )
        network = _Network()
        summary = _run(env, network, max_steps=3)
        return (
            summary,
            [copy.deepcopy(call[1]["data"]) for call in network.step_calls],
            env.reset_count,
            env.step_count,
        )

    _disable(monkeypatch)
    off_summary, off_data, off_resets, off_steps = once()

    _enable(monkeypatch, tmp_path)
    on_summary, on_data, on_resets, on_steps = once()

    assert (off_resets, off_steps) == (on_resets, on_steps)
    assert off_summary.env_steps == on_summary.env_steps
    assert off_summary.episodes_started == on_summary.episodes_started
    assert off_summary.intervention_steps == on_summary.intervention_steps
    assert off_summary.sidecar_attached_steps == on_summary.sidecar_attached_steps
    assert len(off_data) == len(on_data)
    for before, after in zip(off_data, on_data):
        for section in ("meta", "transition"):
            assert set(before[section]) == set(after[section])
            for key, value in before[section].items():
                # Array-aware: ``meta.policy_action`` and the transition's
                # observations are numpy, and ``==`` on those is not a bool.
                np.testing.assert_array_equal(
                    value, after[section][key], err_msg=f"{section}.{key}"
                )
    # ...and the enabled run really did write, so the comparison above was not
    # two identical no-ops.
    assert len(_records(tmp_path)) == 3


def test_default_directory_is_derived_from_this_checkout(monkeypatch, tmp_path):
    """No absolute path baked in: a second checkout must not write into the first.

    Also asserts that merely importing/constructing resolves the path without
    creating it -- a disabled session touches no filesystem.
    """

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    assert DEFAULT_LATENCY_PROFILE_DIR == os.path.join(
        repo, "ros2_ur_ws", "gello_logs", "hil_latency"
    )

    _disable(monkeypatch)
    existed = os.path.isdir(DEFAULT_LATENCY_PROFILE_DIR)

    _run(_Env(), _Network(), max_steps=2)

    assert os.path.isdir(DEFAULT_LATENCY_PROFILE_DIR) is existed


def test_dir_env_var_overrides_the_call_site_default(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)

    _run(_Env(), _Network(), max_steps=1)

    written = sorted(tmp_path.glob("*.jsonl"))
    assert len(written) == 1
    assert written[0].name.endswith(f"_actor_{os.getpid()}.jsonl")


# --------------------------------------------------------------------- #
# Lifecycle: the sink is closed on the way out
# --------------------------------------------------------------------- #


def test_profiler_is_closed_on_normal_exit(tmp_path):
    """Closure is proven by CONTENT, not by a spy.

    The writer flushes every 50 commits or 5 s; three records in a fast test
    reach neither bound, so a line readable from disk afterwards can only be
    there because ``close()`` flushed it.  The profiler is injected so this test
    keeps a reference to it -- letting it be garbage-collected would flush the
    file for free and certify nothing.
    """

    path = tmp_path / "actor.jsonl"
    profiler = LatencyProfiler("actor", path, enabled=True)

    _run(_Env(), _Network(), max_steps=3, latency_profiler=profiler)

    assert profiler.enabled is False
    assert len(path.read_text().splitlines()) == 3


def test_profiler_is_closed_when_the_loop_raises(tmp_path):
    """The exception path is the one that matters: it is where sessions end."""

    path = tmp_path / "actor.jsonl"
    profiler = LatencyProfiler("actor", path, enabled=True)
    env = _Env(raise_at=2)

    with pytest.raises(RuntimeError, match="scripted env failure"):
        _run(env, _Network(), max_steps=5, latency_profiler=profiler)

    assert profiler.enabled is False
    # The two steps that completed are on disk; the third never produced a
    # transition, so it correctly has no record.
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["env_step"] for record in records] == [0, 1]


def test_close_is_idempotent_and_safe_for_a_disabled_profiler(monkeypatch):
    """A disabled run still calls ``close`` once; it must be a no-op."""

    _disable(monkeypatch)
    profiler = LatencyProfiler("actor", None, enabled=False)

    _run(_Env(), _Network(), max_steps=2, latency_profiler=profiler)
    profiler.close()

    assert profiler.enabled is False
    assert profiler.path is None


def test_a_dead_sink_degrades_and_the_run_still_finishes(
    monkeypatch, tmp_path
):
    """Same rule as ``c86dc54``: a dead sink degrades the logger, not the run.

    The output directory is pre-empted by a regular FILE, so the lazy ``mkdir``
    at the first commit raises.  The actor must complete every step anyway.
    """

    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory\n")
    _enable(monkeypatch, blocker)
    env = _Env()

    summary = _run(env, _Network(), max_steps=3)

    assert env.step_count == 3
    assert summary.env_steps == 3
    assert blocker.is_file()


# --------------------------------------------------------------------- #
# Passthrough values
# --------------------------------------------------------------------- #


def test_intervention_info_fields_pass_through_when_present(
    monkeypatch, tmp_path
):
    """Copied from ``info``, never recomputed.

    ``UR7eEnv`` measures saturation because the follower can out-run
    ``ACTION_SCALE`` now that the budget is gone (``08_OPEN_GAPS.md`` G33).
    Recomputing it here would let the profile and the stored transition
    disagree about the same step.
    """

    _enable(monkeypatch, tmp_path)
    env = _Env(
        infos=[
            {
                "intervened": 1,
                "intervention_saturation": 1.75,
                "intervention_saturated": True,
                "intervention_follow_ticks": 12,
            },
            {},
        ]
    )

    _run(env, _Network(), max_steps=2)

    first, second = _records(tmp_path)
    assert first["intervention_saturation"] == pytest.approx(1.75)
    assert first["intervention_saturated"] is True
    assert first["intervention_follow_ticks"] == 12
    assert first["intervened"] is True
    # Absent, not zero: the env said nothing about this step, and a zero would
    # read as "measured, and it was fine".
    for key in (
        "intervention_saturation",
        "intervention_saturated",
        "intervention_follow_ticks",
    ):
        assert key not in second
    assert second["intervened"] is False


def test_server_inference_ms_is_copied_from_the_step_response(
    monkeypatch, tmp_path
):
    _enable(monkeypatch, tmp_path)

    _run(_Env(), _Network(inference_ms=7.25), max_steps=2)

    for record in _records(tmp_path):
        assert record["server_inference_ms"] == pytest.approx(7.25)


def test_step_rpc_ms_is_the_round_trip_the_loop_already_measured(
    monkeypatch, tmp_path
):
    """Not a second stopwatch around the same call.

    ``_RoundTripStats`` sees the identical samples, so the summary's mean/max
    must bracket every ``step_rpc_ms`` in the file.  A duplicate measurement
    would drift out of that envelope under load, which is exactly when the
    number matters.
    """

    _enable(monkeypatch, tmp_path)

    summary = _run(_Env(), _Network(), max_steps=4)

    samples = [record["step_rpc_ms"] for record in _records(tmp_path)]
    assert len(samples) == 4
    assert max(samples) <= summary.plain_round_trip_ms_max + 1e-3
    assert min(samples) >= 0.0
    # Rounding to 3 decimals is the only transformation allowed between the two.
    assert summary.plain_round_trip_ms_mean == pytest.approx(
        sum(samples) / len(samples), abs=1e-3
    )


def test_sidecar_attachment_is_flagged_and_timed(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    env = _Env(cameras=True)
    scheduler = _Scheduler(attach_on=(1,))

    _run(env, _Network(), max_steps=3, sidecar_scheduler=scheduler)

    records = _records(tmp_path)
    assert [record["sidecar_attached"] for record in records] == [
        False,
        True,
        False,
    ]
    assert records[1]["sidecar_encode_ms"] >= 0.0
    # Absent on the steps that shipped no pixels: a phase named after work must
    # not report time for work that did not happen.
    assert "sidecar_encode_ms" not in records[0]
    assert "sidecar_encode_ms" not in records[2]


# --------------------------------------------------------------------- #
# Join keys against the real server-side session service
# --------------------------------------------------------------------- #


class _InProcessNetwork:
    """Drives the REAL ``ActorSessionService`` in-process.

    The join between the actor's file and the server's is by
    ``transition_id`` + ``request_id``, so those two fields have to be the
    values the real server validated, not values a fake invented.
    ``ActorSessionService`` enforces the request_id sequence itself
    (``actor_network.py:694``), which is what makes this a proof.
    """

    def __init__(self, service):
        self._service = service
        self._run_id = ""
        self._session_id = ""
        self._request_id = 1
        self._clock = 10_000
        self.step_request_ids = []

    def begin_episode(
        self,
        observation,
        *,
        run_id,
        session_id,
        episode_id,
        observation_id,
        timestamp_ns,
        deterministic=False,
    ):
        self._run_id = run_id
        self._session_id = session_id
        self._request_id = 2
        self._clock += 1
        return self._service.begin_episode(
            BeginEpisodeCommand(
                PROTOCOL_VERSION,
                "actor",
                run_id,
                session_id,
                episode_id,
                1,
                self._clock,
                ObservationPacket(observation_id, timestamp_ns, observation),
                deterministic,
            )
        )

    def step(
        self,
        next_observation,
        *,
        next_observation_id,
        next_timestamp_ns,
        data,
        request_action,
        deterministic=False,
    ):
        self._clock += 1
        self.step_request_ids.append(self._request_id)
        result = self._service.step(
            StepCommand(
                PROTOCOL_VERSION,
                "actor",
                self._run_id,
                self._session_id,
                self._request_id,
                self._clock,
                copy.deepcopy(data),
                ObservationPacket(
                    next_observation_id, next_timestamp_ns, next_observation
                ),
                request_action,
                deterministic,
            )
        )
        self._request_id += 1
        return result


def test_join_keys_match_the_real_session_service(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 3)
    )
    network = _InProcessNetwork(service)

    _run(_Env(), network, max_steps=3)

    records = _records(tmp_path)
    assert [record["request_id"] for record in records] == (
        network.step_request_ids
    )
    assert [record["transition_id"] for record in records] == [
        f"run:{index}" for index in range(3)
    ]
    for record in records:
        # The real service times its own policy call, so this is a genuine
        # measurement rather than a constant a fake handed back.
        assert record["server_inference_ms"] >= 0.0
