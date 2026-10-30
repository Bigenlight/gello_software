"""Contract tests for ``ur_env.bc_inference_log.InferenceLoggingPolicy``.

WHAT THIS WRAPPER IS
--------------------
``scripts/run_bc_policy_server.py`` wraps the ``VersionedPolicyRuntime`` in
``InferenceLoggingPolicy`` and hands the wrapper to
``ActorSessionService(sample_action=...)``.  The service calls it positionally,
once per Step RPC, as ``sample_action(observation, deterministic) ->
(action, version)`` (``ur_env/actor_network.py``
``_sample_policy_action``).  So the wrapper sits directly in the robot's
control path: everything it does costs latency on a real arm, and anything it
raises kills the RPC that was about to move that arm.

That is why the three properties pinned hardest here are NOT about the log
file:

1. **The result comes back unchanged.**  The action object and the version
   integer the policy produced are what the service must see; a wrapper that
   copies, casts or re-dtypes on the way out would change what the robot
   executes and what ``build_data`` records as ``policy_action``.
2. **A logging failure never breaks inference.**  A full disk, a
   read-only mount or a path that is a directory must degrade to "no log
   line", never to a dead episode.  The log is an audit convenience; the
   rollout is the experiment.
3. **The wrapper is transparent.**  ``__getattr__`` delegation keeps
   ``model_id`` and friends reachable through it, and the wrapper must not
   invent attributes the wrapped policy does not have -- ``prime_observation``
   is the name this stack probes with ``getattr(..., "prime_observation",
   None)`` to decide whether to swap pixels for frozen-trunk features, so an
   accidental attribute of that name is exactly the kind of thing that changes
   inference input while every gate still reports healthy.

The log schema itself is pinned as a set of keys with typed values --
``ts``, ``latency_ms``, ``deterministic``, ``policy_version``, ``action``
(7 floats), ``state`` (19 floats or null) -- because
``scripts/analyze_bc_rollout.py`` reads these lines back.  Extra keys are
allowed; a missing key is not.

Assertions here are about types and behaviour, never about exact message text.

Run (from ``/home/laptop3/gello_software``)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \
      -p no:cacheprovider serl_ur_infra/tests/test_bc_inference_log.py
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.bc_inference_log import InferenceLoggingPolicy  # noqa: E402


#: The 7-D EEF-delta action contract (6 pose deltas + gripper).
ACTION_DIM = 7

#: The canonical proprioceptive state width (``observation_schema``).
STATE_DIM = 19

#: Arbitrary but non-1 so a hard-coded default cannot pass by accident.
POLICY_VERSION = 3

#: Every key the analyzer is allowed to rely on.  Subset check: a logger may
#: add fields, but it may not drop one of these.
PINNED_KEYS = frozenset(
    {"ts", "latency_ms", "deterministic", "policy_version", "action", "state"}
)


# --------------------------------------------------------------------------- #
# Stand-in policy                                                              #
# --------------------------------------------------------------------------- #
class _StubPolicy:
    """Minimal stand-in for ``VersionedPolicyRuntime``.

    Same call signature the service uses -- ``(observation, deterministic)``
    positional, returning ``(action, version)`` -- and it hands back the SAME
    array object every call so the identity assertions below are meaningful.
    """

    #: A real attribute for the delegation test; the runtime carries this too.
    model_id = "bc-cube-in-cup-raw0731-bcinit-v1"

    def __init__(self, *, version: int = POLICY_VERSION, delay_s: float = 0.0) -> None:
        self.version = int(version)
        self.delay_s = float(delay_s)
        self.action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.calls: list[tuple[Any, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, observation, deterministic):
        if self.delay_s:
            time.sleep(self.delay_s)
        with self._lock:
            self.calls.append((observation, deterministic))
        return self.action, self.version


def _state_row() -> np.ndarray:
    """Distinct values, so "the state reached the log" is provable."""

    return (np.arange(STATE_DIM, dtype=np.float32) + 1.0) / 100.0


def _observation(*, with_state: bool = True) -> dict[str, np.ndarray]:
    """A canonical actor observation: cam uint8 (1,128,128,3), state f32 (1,19)."""

    observation: dict[str, np.ndarray] = {
        "cam1": np.zeros((1, 128, 128, 3), dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 7, dtype=np.uint8),
    }
    if with_state:
        observation["state"] = _state_row().reshape(1, STATE_DIM)
    return observation


# --------------------------------------------------------------------------- #
# Log readers                                                                  #
# --------------------------------------------------------------------------- #
def _records(log_path) -> list[dict[str, Any]]:
    path = Path(log_path)
    assert path.is_file(), f"expected an inference log at {path}"
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # A torn or interleaved append fails right here.
        record = json.loads(line)
        assert isinstance(record, dict), "each line must be a JSON object"
        records.append(record)
    return records


def _one_record(log_path) -> dict[str, Any]:
    records = _records(log_path)
    assert len(records) == 1, f"expected exactly one logged call, got {len(records)}"
    return records[0]


def _flat_floats(value: Any, dim: int, label: str) -> list[float]:
    """Require ``dim`` JSON numbers, tolerating one level of row nesting.

    The canonical wire observation stores state as ``(1, 19)``; a logger may
    write the flattened row or the single-row nesting it came from.  What is
    pinned is that ``dim`` numbers -- and the right ones -- land in the log,
    not which of those two spellings the writer chose.
    """

    assert isinstance(value, list), f"{label} must be a JSON list, got {type(value)}"
    flat = value
    while len(flat) == 1 and isinstance(flat[0], list):
        flat = flat[0]
    assert len(flat) == dim, f"{label} must hold {dim} numbers, got {len(flat)}"
    for item in flat:
        assert isinstance(item, (int, float)) and not isinstance(item, bool), (
            f"{label} must hold JSON numbers, got {type(item)}"
        )
    return [float(item) for item in flat]


def _assert_pinned_shape(record: dict[str, Any]) -> None:
    """Every property the analyzer may depend on, checked by type."""

    missing = PINNED_KEYS - set(record)
    assert not missing, f"inference log line is missing {sorted(missing)}"

    timestamp = record["ts"]
    assert isinstance(timestamp, (str, int, float)), "ts must be text or a number"
    assert str(timestamp).strip(), "ts must not be empty"

    latency = record["latency_ms"]
    assert isinstance(latency, (int, float)) and not isinstance(latency, bool)
    assert latency >= 0.0, "latency_ms must never be negative"

    assert record["deterministic"] in (True, False), "deterministic must be boolean"

    version = record["policy_version"]
    assert isinstance(version, int) and not isinstance(version, bool)

    _flat_floats(record["action"], ACTION_DIM, "action")


# --------------------------------------------------------------------------- #
# 1. The wrapper is transparent to the control path.                            #
# --------------------------------------------------------------------------- #
def test_returns_the_wrapped_result_unchanged(tmp_path):
    policy = _StubPolicy()
    wrapper = InferenceLoggingPolicy(policy, tmp_path / "inference.jsonl")

    observation = _observation()
    action, version = wrapper(observation, False)

    # Identity, not equality: a copy/cast on the way out would change what the
    # robot executes and what build_data records as the policy action.
    assert action is policy.action
    assert isinstance(action, np.ndarray)
    assert action.dtype == np.dtype(np.float32)
    assert action.shape == (ACTION_DIM,)
    assert version == POLICY_VERSION
    assert isinstance(version, int) and not isinstance(version, bool)


def test_observation_and_deterministic_flag_reach_the_wrapped_policy(tmp_path):
    policy = _StubPolicy()
    wrapper = InferenceLoggingPolicy(policy, tmp_path / "inference.jsonl")

    observation = _observation()
    wrapper(observation, True)

    assert len(policy.calls) == 1
    seen_observation, seen_deterministic = policy.calls[0]
    assert bool(seen_deterministic) is True
    # Equality, not identity: a defensive copy on the way IN is allowed.
    assert set(seen_observation) == set(observation)
    np.testing.assert_array_equal(seen_observation["state"], observation["state"])


# --------------------------------------------------------------------------- #
# 2. One line per call, with the pinned schema.                                 #
# --------------------------------------------------------------------------- #
def test_one_jsonl_line_per_call_carries_the_pinned_keys(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    policy = _StubPolicy()
    wrapper = InferenceLoggingPolicy(policy, log_path)

    observation = _observation()
    for _ in range(3):
        wrapper(observation, False)

    records = _records(log_path)
    assert len(records) == 3, "the log is append-only: one line per inference"
    for record in records:
        _assert_pinned_shape(record)
        assert record["policy_version"] == POLICY_VERSION
        assert _flat_floats(record["action"], ACTION_DIM, "action") == [0.0] * ACTION_DIM
        assert _flat_floats(record["state"], STATE_DIM, "state") == pytest.approx(
            [float(value) for value in _state_row()]
        )


@pytest.mark.parametrize("deterministic", [True, False])
def test_deterministic_flag_is_logged_as_passed(tmp_path, deterministic):
    log_path = tmp_path / "inference.jsonl"
    wrapper = InferenceLoggingPolicy(_StubPolicy(), log_path)

    wrapper(_observation(), deterministic)

    record = _one_record(log_path)
    _assert_pinned_shape(record)
    assert bool(record["deterministic"]) is deterministic


def test_policy_version_logged_is_the_version_the_policy_returned(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    policy = _StubPolicy(version=11)
    wrapper = InferenceLoggingPolicy(policy, log_path)

    wrapper(_observation(), False)

    assert _one_record(log_path)["policy_version"] == 11


# --------------------------------------------------------------------------- #
# 3. state is opt-out-able and absence-tolerant.                                #
# --------------------------------------------------------------------------- #
def test_state_is_null_when_logging_is_disabled(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    wrapper = InferenceLoggingPolicy(_StubPolicy(), log_path, log_state=False)

    wrapper(_observation(), False)

    record = _one_record(log_path)
    _assert_pinned_shape(record)
    # The key must still be present: the analyzer distinguishes "not logged"
    # from "field absent from this build's schema".
    assert record["state"] is None


def test_state_is_null_when_the_observation_carries_none(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    wrapper = InferenceLoggingPolicy(_StubPolicy(), log_path)

    action, version = wrapper(_observation(with_state=False), False)

    assert action is not None and version == POLICY_VERSION
    record = _one_record(log_path)
    _assert_pinned_shape(record)
    assert record["state"] is None, "a missing state must log null, not raise"


def test_log_state_defaults_to_true(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    wrapper = InferenceLoggingPolicy(_StubPolicy(), log_path)

    wrapper(_observation(), False)

    assert _one_record(log_path)["state"] is not None


# --------------------------------------------------------------------------- #
# 4. latency_ms is milliseconds of wall clock around the policy call.           #
# --------------------------------------------------------------------------- #
def test_latency_ms_is_milliseconds_and_never_negative(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    # 20 ms is far above scheduler noise and far below any plausible unit
    # confusion: seconds would log ~0.02 and nanoseconds ~2e7.
    wrapper = InferenceLoggingPolicy(_StubPolicy(delay_s=0.02), log_path)

    wrapper(_observation(), False)

    latency = _one_record(log_path)["latency_ms"]
    assert latency >= 5.0, f"latency_ms={latency} looks like seconds, not ms"
    assert latency < 60_000.0, f"latency_ms={latency} looks like ns/us, not ms"


def test_latency_ms_of_an_instant_policy_is_non_negative(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    wrapper = InferenceLoggingPolicy(_StubPolicy(), log_path)

    wrapper(_observation(), False)

    assert _one_record(log_path)["latency_ms"] >= 0.0


# --------------------------------------------------------------------------- #
# 5. call_count                                                                 #
# --------------------------------------------------------------------------- #
def test_call_count_starts_at_zero_and_increments(tmp_path):
    wrapper = InferenceLoggingPolicy(_StubPolicy(), tmp_path / "inference.jsonl")

    # The server's shutdown report reads this attribute; it has to exist before
    # the first inference, not only after one.
    assert wrapper.call_count == 0

    observation = _observation()
    for expected in (1, 2, 3):
        wrapper(observation, False)
        assert wrapper.call_count == expected


# --------------------------------------------------------------------------- #
# 6. Attribute delegation.                                                      #
# --------------------------------------------------------------------------- #
def test_attributes_delegate_to_the_wrapped_policy(tmp_path):
    policy = _StubPolicy()
    wrapper = InferenceLoggingPolicy(policy, tmp_path / "inference.jsonl")

    # The server passes the same object to code that reads runtime identity.
    assert wrapper.model_id == policy.model_id
    assert wrapper.version == policy.version


def test_absent_attribute_raises_attributeerror(tmp_path):
    wrapper = InferenceLoggingPolicy(_StubPolicy(), tmp_path / "inference.jsonl")

    # __getattr__ must re-raise, not return None: callers use hasattr() to
    # probe for optional capabilities.
    with pytest.raises(AttributeError):
        getattr(wrapper, "definitely_not_an_attribute_of_this_policy")


def test_wrapper_does_not_advertise_prime_observation(tmp_path):
    policy = _StubPolicy()
    wrapper = InferenceLoggingPolicy(policy, tmp_path / "inference.jsonl")

    # WHY THIS IS AN ASSERTION AND NOT A DETAIL: this stack decides whether to
    # feed frozen-trunk FEATURES instead of pixels by probing
    # ``getattr(obj, "prime_observation", None)``.  The BC encoder is
    # dual-input and must receive raw pixels.  A wrapper that grew that
    # attribute -- or a __getattr__ that answered every name instead of
    # raising -- would silently change inference input.
    assert not hasattr(policy, "prime_observation"), "stub sanity check"
    assert not hasattr(wrapper, "prime_observation")


# --------------------------------------------------------------------------- #
# 7. Logging is best-effort; inference is not.                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("broken", ["directory", "missing_parent"])
def test_unwritable_log_path_never_breaks_inference(tmp_path, broken):
    if broken == "directory":
        # open(<dir>, "a") raises IsADirectoryError on every call.
        log_path = tmp_path / "inference.jsonl"
        log_path.mkdir()
    else:
        # A parent that does not exist.  Creating it is a legitimate
        # implementation choice; what is pinned is only that inference works
        # either way.
        log_path = tmp_path / "no" / "such" / "dir" / "inference.jsonl"

    policy = _StubPolicy()
    # Construction must survive too: the server builds this wrapper before the
    # first episode, and a constructor that opens the file eagerly would turn
    # a bad log path into a server that never starts.
    wrapper = InferenceLoggingPolicy(policy, log_path)

    observation = _observation()
    for index in range(3):
        action, version = wrapper(observation, False)
        assert action is policy.action, "a log failure must not swallow the action"
        assert version == POLICY_VERSION
        assert wrapper.call_count == index + 1

    assert len(policy.calls) == 3, "every call must still reach the policy"


def test_inference_survives_a_log_path_that_breaks_midway(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    policy = _StubPolicy()
    wrapper = InferenceLoggingPolicy(policy, log_path)
    observation = _observation()

    wrapper(observation, False)
    assert len(_records(log_path)) == 1

    # Replace the file with a directory: appends fail from here on.
    log_path.unlink()
    log_path.mkdir()

    for _ in range(2):
        action, version = wrapper(observation, False)
        assert action is policy.action
        assert version == POLICY_VERSION
    assert wrapper.call_count == 3


# --------------------------------------------------------------------------- #
# 8. Thread safety (gRPC serves Step from several handler threads).             #
# --------------------------------------------------------------------------- #
def test_concurrent_calls_produce_one_parseable_line_each(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    policy = _StubPolicy()
    wrapper = InferenceLoggingPolicy(policy, log_path)

    workers, calls = 4, 25

    def drive(_index: int) -> None:
        observation = _observation()
        for _ in range(calls):
            action, version = wrapper(observation, False)
            assert action is policy.action
            assert version == POLICY_VERSION

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(drive, index) for index in range(workers)]:
            future.result()

    total = workers * calls
    # json.loads on every line catches interleaved (torn) appends; a short
    # count means a dropped line or an unlocked counter.
    records = _records(log_path)
    assert len(records) == total
    for record in records:
        _assert_pinned_shape(record)
    assert wrapper.call_count == total
