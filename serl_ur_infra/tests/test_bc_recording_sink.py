"""Contract tests for ``ur_env.bc_recording_sink.EpisodeRecordingSink``.

The sink is the ``accept_data`` callback of the BC evaluation server: it is the
only thing standing between an evaluation run and a lost episode, so what it
writes has to be loadable by the *same* strict loader the learner uses
(``ur_env.learner.demo.load_demo_object``).  These tests therefore build
fixture transitions that satisfy the real wire contract
(``ur_env.actor_network.ActorSessionService._validate_data``: a
``{"meta": ..., "transition": ...}`` wrapper) AND the demo contract, so the
round-trip assertion is honest rather than self-referential.

Run (from ``/home/laptop3/gello_software``)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
      -p no:cacheprovider serl_ur_infra/tests/test_bc_recording_sink.py

Assertions here are about types and behaviour, never about exact message text.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import pickle
import sys
from typing import Any, Mapping

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import SCHEMA_VERSION  # noqa: E402
from ur_env.bc_recording_sink import EpisodeRecordingSink  # noqa: E402
from ur_env.learner import load_demo_object  # noqa: E402


ARTIFACT_SHA = "8ffcfac5" + "0" * 56
MODEL_ID = "bc-cube-in-cup-raw0731-bcinit-v1"
RUN_ID = "bc_eval_20260731_000000"

# The one action shape/domain the whole stack agrees on: float32 (7,) in
# [-1, 1] with the gripper channel in {-1, 0, 1}.
ACTION = np.array([0.1, -0.1, 0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Fixtures: minimal-but-valid canonical transitions.                            #
# --------------------------------------------------------------------------- #
def _observation(value: int = 0) -> dict[str, np.ndarray]:
    """A canonical observation: cam1/cam2 uint8 (1,128,128,3), state f32 (1,19)."""
    return {
        "cam1": np.full((1, 128, 128, 3), value % 256, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), (255 - value) % 256, dtype=np.uint8),
        "state": np.full((1, 19), value / 100.0, dtype=np.float32),
    }


def _data(
    *,
    run_id: str = RUN_ID,
    episode_id: int = 0,
    step_id: int = 0,
    env_step: int | None = None,
    intervened: bool = False,
    dones: bool = False,
    truncated: bool = False,
    action: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one ``{"meta", "transition"}`` item exactly as the actor sends it.

    Field placement mirrors ``ActorSessionService._validate_data``:
    ``run_id``/``env_step``/``intervened``/``transition_id`` live in ``meta``;
    ``episode_id``/``step_id``/``actions``/``dones``/``truncated``/``rewards``/
    ``masks``/``success`` live in ``transition``.
    """
    assert not (dones and truncated), "the wire contract forbids done AND truncated"
    executed = ACTION if action is None else np.asarray(action, dtype=np.float32)
    mask = 0.0 if dones else 1.0
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "actor_id": "laptop3-bc-eval",
        "session_id": f"{run_id}:{episode_id}",
        "transition_id": f"{run_id}:{episode_id}:{step_id}",
        "env_step": int(step_id if env_step is None else env_step),
        "timestamp_ns": 1_700_000_000_000_000_000 + int(step_id),
        "policy_version": 1,
        # Non-intervened steps must carry policy_action == executed action.
        "policy_action": executed.copy(),
        "intervened": int(bool(intervened)),
        "auto_success": 0,
        "operator_success": 0,
    }
    transition = {
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "observation_id": f"{run_id}:{episode_id}:obs{step_id}",
        "next_observation_id": f"{run_id}:{episode_id}:obs{step_id + 1}",
        "observations": _observation(step_id),
        "next_observations": _observation(step_id + 1),
        "actions": executed.copy(),
        "rewards": 0.0,
        "masks": mask,
        "dones": bool(dones),
        "truncated": bool(truncated),
        "grasp_penalty": 0.0,
        # Written by the identity finalizer on the server side.
        "success": np.uint8(0),
        "classifier_evaluated": np.uint8(0),
        "classifier_probability": 0.0,
        "classifier_threshold": 0.0,
        "classifier_success": np.uint8(0),
        "reward_model_id": "",
    }
    return {"meta": meta, "transition": transition}


def _make_sink(root, **overrides) -> EpisodeRecordingSink:
    kwargs = {"artifact_sha256": ARTIFACT_SHA, "model_id": MODEL_ID}
    kwargs.update(overrides)
    return EpisodeRecordingSink(root, **kwargs)


def _episode_path(root, run_id: str = RUN_ID, episode_id: int = 0):
    """The pinned layout: ``<root>/<run_id>/episode_<id 0-padded 4>.pkl``."""
    return root / run_id / f"episode_{episode_id:04d}.pkl"


def _jsonl_records(root) -> list[dict[str, Any]]:
    path = root / "actions.jsonl"
    assert path.is_file(), "the sink must append to <root>/actions.jsonl"
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)  # a torn/interleaved write fails right here
        assert isinstance(record, dict)
        records.append(record)
    return records


def _field(record: Mapping[str, Any], names: tuple[str, ...], label: str) -> Any:
    """Read one logical field that may be spelled singular or plural.

    Deliberately tolerant about the SPELLING only.  The presence of the field
    and the type of its value are what this suite pins; a jsonl line that
    cannot say which run/episode/action it belongs to is useless for audit.
    """
    present = [name for name in names if name in record]
    assert present, f"jsonl record must carry {label} (one of {names}): {sorted(record)}"
    return record[present[0]]


def _feed_episode(sink, *, run_id=RUN_ID, episode_id=0, steps=3,
                  terminal="dones", intervened=()):
    """Feed ``steps`` transitions; the last one is terminal unless ``terminal`` is None."""
    for step in range(steps):
        last = step == steps - 1
        sink(
            _data(
                run_id=run_id,
                episode_id=episode_id,
                step_id=step,
                intervened=step in intervened,
                dones=bool(last and terminal == "dones"),
                truncated=bool(last and terminal == "truncated"),
            ),
            step in intervened,
        )


# --------------------------------------------------------------------------- #
# 1. Construction                                                               #
# --------------------------------------------------------------------------- #
def test_construction_requires_existing_root_and_writes_metadata(tmp_path):
    missing = tmp_path / "does_not_exist"
    # The sink opens episode pickles with "xb" and never creates directories,
    # so a missing root has to fail at construction, not mid-episode.
    with pytest.raises((ValueError, OSError)):
        _make_sink(missing)
    assert not missing.exists(), "a rejected root must not be created as a side effect"

    root = tmp_path / "bc_eval"
    root.mkdir()
    sink = _make_sink(root)

    metadata_path = root / "metadata.json"
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert isinstance(metadata, dict)
    assert ARTIFACT_SHA in metadata.values() or ARTIFACT_SHA in json.dumps(metadata)
    assert MODEL_ID in metadata.values() or MODEL_ID in json.dumps(metadata)
    assert sink.replay_count == 0
    assert sink.intervention_count == 0


def test_construction_accepts_a_string_root(tmp_path):
    root = tmp_path / "as_str"
    root.mkdir()
    sink = _make_sink(str(root))
    assert (root / "metadata.json").is_file()
    assert sink.replay_count == 0


# --------------------------------------------------------------------------- #
# 2. Re-construction on an existing root                                        #
# --------------------------------------------------------------------------- #
def test_reconstruction_matches_or_refuses(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    _make_sink(root)
    before = (root / "metadata.json").read_bytes()

    # Same identity: the sink re-attaches to the run instead of failing on "x".
    again = _make_sink(root)
    assert again.replay_count == 0
    assert (root / "metadata.json").read_bytes() == before, (
        "re-construction must verify the existing metadata, not rewrite it"
    )

    # Different weights under the same root would silently mix two policies'
    # episodes into one evaluation directory.
    with pytest.raises(Exception) as excinfo:
        _make_sink(root, artifact_sha256="f" * 64)
    assert not isinstance(excinfo.value, (AssertionError, ImportError, NameError))
    assert (root / "metadata.json").read_bytes() == before


def test_reconstruction_refuses_a_different_model_id(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    _make_sink(root)
    with pytest.raises(Exception) as excinfo:
        _make_sink(root, model_id="some-other-policy-v9")
    assert not isinstance(excinfo.value, (AssertionError, ImportError, NameError))


# --------------------------------------------------------------------------- #
# 3. Round trip: a finished episode must load through the strict demo loader.   #
# --------------------------------------------------------------------------- #
def test_terminal_episode_round_trips_through_load_demo_object(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    sink = _make_sink(root)

    _feed_episode(sink, steps=3, terminal="dones", intervened={1})

    path = _episode_path(root)
    assert path.is_file(), f"expected the episode pickle at {path}"

    with open(path, "rb") as stream:
        items = pickle.load(stream)
    assert isinstance(items, list), "an episode file holds a list of transitions"
    assert len(items) == 3
    for index, item in enumerate(items):
        assert isinstance(item, Mapping)
        assert set(item) == {"meta", "transition"}, (
            "items must stay actor-shaped wrappers so provenance survives"
        )
        assert int(item["transition"]["step_id"]) == index, "order must be step order"
        assert item["meta"]["run_id"] == RUN_ID
    assert bool(items[-1]["transition"]["dones"]) is True
    assert all(not bool(item["transition"]["dones"]) for item in items[:-1])

    # The honest part: the learner's own strict loader accepts the file.
    loaded = load_demo_object(items, source_path=str(path))
    assert len(loaded) == 3
    for transition in loaded.transitions:
        assert transition["actions"].dtype == np.dtype(np.float32)
        assert transition["actions"].shape == (7,)
        assert transition["observations"]["cam1"].shape == (1, 128, 128, 3)
        assert transition["observations"]["state"].dtype == np.dtype(np.float32)

    assert sink.replay_count == 3
    assert sink.intervention_count == 1


def test_actions_jsonl_has_one_auditable_line_per_call(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    sink = _make_sink(root)

    _feed_episode(sink, steps=3, terminal="dones", intervened={1})

    records = _jsonl_records(root)
    assert len(records) == 3, "one jsonl line per accept_data call"
    for index, record in enumerate(records):
        assert _field(record, ("run_id",), "run_id") == RUN_ID
        assert int(_field(record, ("episode_id",), "episode_id")) == 0
        action = _field(record, ("actions", "action"), "the executed action")
        assert isinstance(action, list)
        assert len(action) == 7
        assert all(isinstance(value, float) for value in action)
        assert all(-1.0 <= value <= 1.0 for value in action)
        intervened = _field(record, ("intervened",), "the intervention flag")
        assert intervened in (True, False, 0, 1)
        assert bool(intervened) is (index == 1)
        done = _field(record, ("dones", "done"), "the terminal flag")
        assert done in (True, False, 0, 1)
        assert bool(done) is (index == 2)


# --------------------------------------------------------------------------- #
# 4. Truncation is a terminal too.                                              #
# --------------------------------------------------------------------------- #
def test_truncated_episode_is_finalized(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    sink = _make_sink(root)

    # END EPISODE / step-limit truncation: dones stays False, masks stays 1.0.
    _feed_episode(sink, steps=2, terminal="truncated")

    path = _episode_path(root)
    assert path.is_file(), "truncated episodes must be written, not dropped"
    with open(path, "rb") as stream:
        items = pickle.load(stream)
    assert len(items) == 2
    assert bool(items[-1]["transition"]["truncated"]) is True
    assert bool(items[-1]["transition"]["dones"]) is False
    assert float(items[-1]["transition"]["masks"]) == 1.0
    assert len(load_demo_object(items, source_path=str(path))) == 2
    assert sink.replay_count == 2


# --------------------------------------------------------------------------- #
# 5. Never overwrite a recorded episode.                                        #
# --------------------------------------------------------------------------- #
def test_refuses_to_overwrite_an_existing_episode_file(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    sink = _make_sink(root)

    path = _episode_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = b"an earlier episode that must survive"
    path.write_bytes(sentinel)

    with pytest.raises(Exception) as excinfo:
        _feed_episode(sink, steps=2, terminal="dones")
    # FileExistsError, or something wrapping it -- but never a silent success
    # and never an accident of the test harness.
    assert not isinstance(excinfo.value, (AssertionError, ImportError, NameError))

    assert path.read_bytes() == sentinel, "the pre-existing episode was clobbered"


# --------------------------------------------------------------------------- #
# 6. Independent buffering per run.                                             #
# --------------------------------------------------------------------------- #
def test_two_interleaved_runs_buffer_independently(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    sink = _make_sink(root)

    run_a, run_b = "bc_eval_run_a", "bc_eval_run_b"
    sink(_data(run_id=run_a, episode_id=0, step_id=0), False)
    sink(_data(run_id=run_b, episode_id=3, step_id=0), True)
    sink(_data(run_id=run_a, episode_id=0, step_id=1, dones=True), False)
    sink(_data(run_id=run_b, episode_id=3, step_id=1), False)
    sink(_data(run_id=run_b, episode_id=3, step_id=2, truncated=True), False)

    path_a = _episode_path(root, run_a, 0)
    path_b = _episode_path(root, run_b, 3)
    assert path_a.is_file() and path_b.is_file()

    with open(path_a, "rb") as stream:
        items_a = pickle.load(stream)
    with open(path_b, "rb") as stream:
        items_b = pickle.load(stream)

    assert len(items_a) == 2, "run A must not absorb run B's transitions"
    assert len(items_b) == 3
    assert {item["meta"]["run_id"] for item in items_a} == {run_a}
    assert {item["meta"]["run_id"] for item in items_b} == {run_b}
    assert [int(i["transition"]["step_id"]) for i in items_b] == [0, 1, 2]
    assert len(load_demo_object(items_a, source_path=str(path_a))) == 2
    assert len(load_demo_object(items_b, source_path=str(path_b))) == 3

    assert sink.replay_count == 5
    assert sink.intervention_count == 1


# --------------------------------------------------------------------------- #
# 7. The absence of prime_observation is load-bearing.                          #
# --------------------------------------------------------------------------- #
def test_sink_must_not_expose_prime_observation(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    sink = _make_sink(root)

    # WHY THIS IS AN ASSERTION AND NOT A DETAIL:
    # ActorSessionService._prime_replay_observation() does
    # ``getattr(self._accept_data, "prime_observation", None)`` and, when it
    # finds a callable, feeds the returned frozen-trunk FEATURES to the policy
    # instead of the actor's pixels.  The BC agent's encoder is dual-input, so
    # it must receive raw pixels and run the trunk itself.  Growing a
    # prime_observation attribute here would silently change what the policy
    # sees on the real robot while every gate still reports healthy.
    assert not hasattr(sink, "prime_observation")


# --------------------------------------------------------------------------- #
# 8. An unfinished episode is still auditable.                                  #
# --------------------------------------------------------------------------- #
def test_unfinished_episode_writes_no_pickle_but_keeps_the_jsonl(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    sink = _make_sink(root)

    _feed_episode(sink, steps=3, terminal=None, intervened={0, 2})

    assert not _episode_path(root).exists(), (
        "an episode that never reached a terminal must not be finalized"
    )
    records = _jsonl_records(root)
    assert len(records) == 3, "the actions log is the only trace of a lost episode"
    assert all(not bool(_field(r, ("dones", "done"), "terminal flag")) for r in records)
    assert sink.replay_count == 3
    assert sink.intervention_count == 2


# --------------------------------------------------------------------------- #
# Thread safety (the sink is pinned as a thread-safe callable).                 #
# --------------------------------------------------------------------------- #
def test_concurrent_calls_keep_counters_and_jsonl_intact(tmp_path):
    root = tmp_path / "bc_eval"
    root.mkdir()
    sink = _make_sink(root)

    workers, steps = 4, 12

    def feed(index: int) -> None:
        for step in range(steps):
            sink(
                _data(run_id=f"bc_eval_t{index}", episode_id=0, step_id=step),
                step % 3 == 0,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(feed, index) for index in range(workers)]:
            future.result()

    total = workers * steps
    assert sink.replay_count == total, "a lost increment means the counter is unlocked"
    assert sink.intervention_count == workers * len(
        [step for step in range(steps) if step % 3 == 0]
    )
    # json.loads on every line catches interleaved (torn) appends.
    assert len(_jsonl_records(root)) == total
