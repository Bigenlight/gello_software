"""Receive-server reward and replay tests without robot or gradient updates."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import sys
from typing import Any

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_INFRA_ROOT, ".."))
sys.path.insert(0, _INFRA_ROOT)

from ur_env.actor_network import ActorProtocolError  # noqa: E402
from ur_env.classifier_sidecar import (  # noqa: E402
    build_sidecar,
    decode_classifier_frames,
)
from ur_env.rlpd_receive_server import (  # noqa: E402
    CLASSIFIER_FAULT_CAUSE_LIMIT,
    UNCLASSIFIED_WARN_STREAK,
    ClassificationResult,
    FakeActionRuntime,
    ReplayIngress,
    RewardClassifierError,
    RewardClassifierRuntime,
    RewardTransitionFinalizer,
    ScriptedRewardClassifierRuntime,
    checkpoint_sha256,
    sigmoid_probability,
    validate_classifier_frames,
)


def _observation(value: int) -> dict[str, np.ndarray]:
    return {
        "state": np.full((1, 19), value / 100.0, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _bgr(seed: int) -> np.ndarray:
    """A decoded, uncropped camera frame at a NON-classifier resolution.

    Deliberately not 128x128, so anything that forgets the classifier-side
    resize shows up as a shape error rather than as a silent pass.
    """
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, seed % 3] = (seed * 7) % 256
    return frame


def _jpeg(seed: int) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".jpg", _bgr(seed))
    assert ok
    return encoded.tobytes()


def _sidecar(seed: int = 0) -> dict[str, np.ndarray]:
    """The payload ActorSessionService hands the finalizer for a classified step."""
    return build_sidecar({"cam1": _bgr(seed), "cam2": _bgr(seed + 1)})


class _WarningSink:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)


class _FlakyClassifier:
    """A classifier whose next ``classify`` can be made to raise anything.

    ``ScriptedRewardClassifierRuntime`` latches itself not-ready on its first
    failure (deliberately, because the real runtime does), which is exactly
    what the rate-limiting tests want and exactly what a recovery test cannot
    use.  It also converts every scripted failure into RewardClassifierError,
    so it cannot express "the input validation rejected these tensors" either.
    """

    def __init__(
        self,
        probabilities: list[float],
        *,
        threshold: float = 0.5,
        reward_model_id: str = "flaky-reward-v0",
    ) -> None:
        self._probabilities = list(probabilities)
        self.threshold = threshold
        self.reward_model_id = reward_model_id
        self.fail_with: BaseException | None = None
        self.evaluation_count = 0

    @property
    def ready(self) -> bool:
        return self.fail_with is None

    def classify(self, frames: Any) -> ClassificationResult:
        # Same input contract the real runtime enforces before the model runs.
        validate_classifier_frames(frames)
        if self.fail_with is not None:
            raise self.fail_with
        probability = self._probabilities[
            self.evaluation_count % len(self._probabilities)
        ]
        self.evaluation_count += 1
        return ClassificationResult(
            probability=probability,
            threshold=self.threshold,
            success=probability > self.threshold,
            reward_model_id=self.reward_model_id,
            inference_ms=0.0,
        )


def _data(
    *,
    step: int = 0,
    intervened: bool = False,
    reward: float = 0.0,
    done: bool = False,
    truncated: bool = False,
    source_value: int | None = None,
    next_value: int | None = None,
    transition_id: str | None = None,
    session_id: str = "session-0",
    auto_success: bool = True,
    operator_success: bool = False,
) -> dict[str, Any]:
    source_value = step if source_value is None else source_value
    next_value = step + 1 if next_value is None else next_value
    policy_action = np.full(7, step / 100.0, dtype=np.float32)
    executed_action = (
        np.full(7, -step / 100.0, dtype=np.float32)
        if intervened
        else policy_action.copy()
    )
    return {
        "meta": {
            "schema_version": 3,
            "run_id": "run-0",
            "actor_id": "actor-0",
            "session_id": session_id,
            "transition_id": transition_id or f"transition-{step}",
            "env_step": step,
            "timestamp_ns": 1_700_000_000_000_000_000 + step,
            "policy_version": 0,
            "policy_action": policy_action,
            "intervened": int(intervened),
            "auto_success": auto_success,
            "operator_success": operator_success,
        },
        "transition": {
            "episode_id": 0,
            "step_id": step,
            "observation_id": f"observation-{step}",
            "actions": executed_action,
            "next_observation_id": f"observation-{step + 1}",
            "rewards": reward,
            "masks": 0.0 if done else 1.0,
            "dones": done,
            "truncated": truncated,
            "observations": _observation(source_value),
            "next_observations": _observation(next_value),
        },
    }


def _finalized_data(**kwargs: Any) -> dict[str, Any]:
    probability = kwargs.pop("probability", 0.1)
    classified = kwargs.pop("classified", True)
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([probability]), warn=_WarningSink()
    )
    data, _ = finalizer(_data(**kwargs), _sidecar() if classified else None)
    return data


class _FakeMemoryStore:
    """Small boundary-aware stand-in for dependency-free unit tests."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._first = True
        self.dataset_dict: dict[str, Any] = {}
        self.items: list[dict[str, Any]] = []
        self.started_sequences: list[bool] = []
        self.fail_next_insert = False

    def __len__(self) -> int:
        return min(len(self.items), self._capacity)

    def insert(self, transition: dict[str, Any]) -> None:
        if self.fail_next_insert:
            self.fail_next_insert = False
            raise BufferError("scripted insert failure")
        self.started_sequences.append(bool(self._first))
        self.items.append(copy.deepcopy(transition))
        self._first = bool(transition["dones"])

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        return {"args": args, "kwargs": kwargs, "items": self.items}


class _StoreFactory:
    def __init__(self) -> None:
        self.stores: list[_FakeMemoryStore] = []

    def __call__(self, *, capacity: int, **kwargs: Any) -> _FakeMemoryStore:
        del kwargs
        store = _FakeMemoryStore(capacity)
        self.stores.append(store)
        return store


def test_fake_action_runtime_defaults_to_safe_zero_and_validates_schema():
    runtime = FakeActionRuntime()

    action, version = runtime(_observation(1), deterministic=True)

    np.testing.assert_array_equal(action, np.zeros(7, dtype=np.float32))
    assert version == 0
    assert runtime.sample_count == 1
    malformed = _observation(1)
    malformed["state"] = malformed["state"].astype(np.float64)
    with pytest.raises(ActorProtocolError, match="dtype float32"):
        runtime(malformed, deterministic=True)


def test_fake_action_runtime_uses_script_once_and_fails_closed():
    expected = np.linspace(-0.5, 0.5, 7, dtype=np.float32)
    runtime = FakeActionRuntime([expected], policy_version=7)

    action, version = runtime(_observation(0), deterministic=False)

    np.testing.assert_array_equal(action, expected)
    assert version == 7
    with pytest.raises(RuntimeError, match="sequence exhausted"):
        runtime(_observation(1), deterministic=False)


@pytest.mark.parametrize(
    ("logit", "expected"),
    [(-1_000.0, 0.0), (0.0, 0.5), (1_000.0, 1.0)],
)
def test_sigmoid_probability_is_stable(logit: float, expected: float):
    assert sigmoid_probability(logit) == pytest.approx(expected)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_sigmoid_probability_rejects_non_finite_values(value: float):
    with pytest.raises(RewardClassifierError, match="finite"):
        sigmoid_probability(value)


def test_reward_classifier_validates_checksum_exact_inputs_and_warms_up(tmp_path):
    checkpoint = tmp_path / "checkpoint_150"
    checkpoint.write_bytes(b"test-flax-checkpoint")
    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls: list[dict[str, np.ndarray]] = []

    def loader(sample):
        assert sample["state"].shape == (1, 1)
        assert sample["state"].dtype == np.float32
        assert sample["cam1"].shape == (1, 128, 128, 3)

        def classifier(observation):
            calls.append({key: np.asarray(value) for key, value in observation.items()})
            return np.array(2.0, dtype=np.float32)

        return classifier

    runtime = RewardClassifierRuntime(
        checkpoint_path=str(checkpoint),
        expected_sha256=expected_sha,
        classifier_loader=loader,
        reward_model_id="cube-in-cup-test",
    )
    frames = decode_classifier_frames(_sidecar(5))
    result = runtime.classify(frames)

    assert runtime.ready
    assert runtime.evaluation_count == 1
    assert checkpoint_sha256(str(checkpoint)) == expected_sha
    assert len(calls) == 2  # one JIT/warmup-equivalent call plus inference
    assert calls[1]["state"].shape == (1, 1)
    assert np.count_nonzero(calls[1]["state"]) == 0
    np.testing.assert_array_equal(calls[1]["cam1"], frames["cam1"])
    assert result.probability == pytest.approx(sigmoid_probability(2.0))
    assert result.success is True
    assert result.reward_model_id == "cube-in-cup-test"


def test_classifier_input_is_the_sidecar_never_the_policy_observation():
    """The cropped policy observation must not be classifiable by accident."""
    classifier = ScriptedRewardClassifierRuntime([0.9])

    with pytest.raises(ActorProtocolError, match="extra=\\['state'\\]"):
        classifier.classify(_observation(23))
    assert classifier.evaluation_count == 0
    # The decoded sidecar is the only accepted shape, and it matches the
    # canonical camera geometry without ever going through IMAGE_CROP.
    frames = validate_classifier_frames(decode_classifier_frames(_sidecar(3)))
    assert set(frames) == {"cam1", "cam2"}
    assert frames["cam1"].shape == (1, 128, 128, 3)
    assert frames["cam1"].dtype == np.uint8


def test_checkpoint_sha256_pins_orbax_directories_deterministically(tmp_path):
    """G19: the canonical checkpoint is an orbax OCDBT directory, not a file."""
    checkpoint = tmp_path / "checkpoint_150"
    (checkpoint / "ocdbt.process_0").mkdir(parents=True)
    (checkpoint / "_METADATA").write_bytes(b"metadata")
    (checkpoint / "ocdbt.process_0" / "d").write_bytes(b"shard-payload")
    (checkpoint / "manifest.ocdbt").write_bytes(b"manifest")

    first = checkpoint_sha256(str(checkpoint))
    second = checkpoint_sha256(str(checkpoint))

    assert first == second
    assert len(first) == 64
    # A single-file checkpoint still hashes to the plain content digest, so
    # every SHA already pinned in a runbook or CLI default stays valid.
    single = tmp_path / "checkpoint_file"
    single.write_bytes(b"test-flax-checkpoint")
    assert checkpoint_sha256(str(single)) == hashlib.sha256(
        b"test-flax-checkpoint"
    ).hexdigest()
    (checkpoint / "ocdbt.process_0" / "d").write_bytes(b"shard-payloae")
    assert checkpoint_sha256(str(checkpoint)) != first
    with pytest.raises(FileNotFoundError):
        checkpoint_sha256(str(tmp_path / "absent"))


def test_reward_classifier_rejects_wrong_checkpoint_hash(tmp_path):
    checkpoint = tmp_path / "checkpoint_150"
    checkpoint.write_bytes(b"wrong")

    with pytest.raises(RewardClassifierError, match="SHA256 mismatch"):
        RewardClassifierRuntime(
            checkpoint_path=str(checkpoint),
            expected_sha256="0" * 64,
            classifier_loader=lambda sample: lambda observation: 0.0,
        )


@pytest.mark.parametrize(
    (
        "probability",
        "local_done",
        "local_truncated",
        "expected_reward",
        "expected_done",
        "expected_truncated",
        "expected_mask",
        "expected_success",
    ),
    [
        (0.10, False, False, 0.0, False, False, 1.0, False),
        (0.90, False, False, 1.0, True, False, 0.0, True),
        (0.10, False, True, 0.0, False, True, 1.0, False),
        # A classifier-positive O(t+1) wins over a simultaneous time limit.
        (0.90, False, True, 1.0, True, False, 0.0, True),
        (0.10, True, False, 0.0, True, False, 0.0, False),
    ],
)
def test_reward_finalizer_server_authority(
    probability,
    local_done,
    local_truncated,
    expected_reward,
    expected_done,
    expected_truncated,
    expected_mask,
    expected_success,
):
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime(
            [probability], threshold=0.85, reward_model_id="cube-in-cup-v1"
        ),
        warn=_WarningSink(),
    )
    data, outcome = finalizer(
        _data(done=local_done, truncated=local_truncated), _sidecar()
    )

    transition = data["transition"]
    assert transition["rewards"] == expected_reward
    assert transition["dones"] is expected_done
    assert transition["truncated"] is expected_truncated
    assert transition["masks"] == expected_mask
    assert bool(transition["classifier_evaluated"])
    assert float(transition["classifier_probability"]) == pytest.approx(probability)
    assert bool(transition["classifier_success"]) is expected_success
    assert outcome.reward == expected_reward
    assert outcome.done is expected_done
    assert outcome.truncated is expected_truncated
    assert outcome.mask == expected_mask
    assert outcome.success is expected_success
    assert outcome.classifier_evaluated is True
    assert outcome.reward_model_id == "cube-in-cup-v1"


def test_manual_mode_keeps_classifier_telemetry_but_not_success_authority():
    classifier = ScriptedRewardClassifierRuntime(
        [0.9], threshold=0.5, reward_model_id="cube-in-cup-v1"
    )
    finalizer = RewardTransitionFinalizer(classifier, warn=_WarningSink())

    data, outcome = finalizer(
        _data(auto_success=False, operator_success=False), _sidecar()
    )

    transition = data["transition"]
    assert classifier.evaluation_count == 1
    assert outcome.classifier_evaluated is True
    assert outcome.classifier_probability == pytest.approx(0.9)
    assert int(transition["classifier_success"]) == 1
    assert transition["reward_model_id"] == "cube-in-cup-v1"
    assert outcome.success is False
    assert int(transition["success"]) == 0
    assert transition["rewards"] == 0.0
    assert transition["dones"] is False

    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=4,
        store_factory=_StoreFactory(),
    )
    ingress(data, False)
    stored = ingress.replay_store.items[0]
    assert int(stored["auto_success"]) == 0
    assert int(stored["operator_success"]) == 0
    assert int(stored["success"]) == 0
    assert int(stored["classifier_success"]) == 1
    record = ingress.replay_sidecar()[0]
    assert record.auto_success is False
    assert record.operator_success is False
    assert record.success is False


def test_manual_operator_success_is_effective_despite_negative_classifier():
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime(
            [0.1], threshold=0.5, reward_model_id="cube-in-cup-v1"
        ),
        warn=_WarningSink(),
    )

    data, outcome = finalizer(
        _data(auto_success=False, operator_success=True), _sidecar()
    )

    assert outcome.classifier_evaluated is True
    assert int(data["transition"]["classifier_success"]) == 0
    assert int(data["transition"]["success"]) == 1
    assert outcome.success is True
    assert outcome.reward == 1.0
    assert outcome.done is True
    assert outcome.truncated is False
    assert outcome.mask == 0.0


def test_auto_mode_rejects_operator_success_before_classification():
    classifier = ScriptedRewardClassifierRuntime([0.9], threshold=0.5)
    finalizer = RewardTransitionFinalizer(classifier, warn=_WarningSink())

    with pytest.raises(
        ActorProtocolError, match="operator_success is forbidden"
    ):
        finalizer(
            _data(auto_success=True, operator_success=True), _sidecar()
        )

    assert classifier.evaluation_count == 0


def test_reward_finalizer_uses_strict_threshold():
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([0.85], threshold=0.85)
    )

    data, outcome = finalizer(_data(), _sidecar())

    assert outcome.success is False
    assert data["transition"]["dones"] is False


def test_reward_finalizer_rejects_provisional_positive_reward_when_negative():
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([0.1], threshold=0.85)
    )

    data, outcome = finalizer(_data(reward=1.0), _sidecar())

    assert data["transition"]["rewards"] == 0.0
    assert outcome.reward == 0.0
    assert outcome.success is False


def test_classifier_fault_degrades_to_unclassified_instead_of_ending_the_run():
    """The behaviour change: a classifier fault is survivable, not terminal.

    Previously this raised RewardClassifierError, which reached
    ActorSessionService._set_fault and latched _ready = False forever, so one
    bad scored step ended the session and the learner had to be restarted.
    """
    sink = _WarningSink()
    classifier = ScriptedRewardClassifierRuntime([RuntimeError("GPU fault")])
    finalizer = RewardTransitionFinalizer(classifier, warn=sink)

    data, outcome = finalizer(_data(reward=1.0), _sidecar())

    transition = data["transition"]
    # Exactly the existing unclassified state -- no third reward meaning.
    assert transition["rewards"] == 0.0
    assert outcome.reward == 0.0
    assert int(transition["classifier_evaluated"]) == 0
    assert float(transition["classifier_probability"]) == 0.0
    assert float(transition["classifier_threshold"]) == 0.0
    assert int(transition["classifier_success"]) == 0
    assert int(transition["success"]) == 0
    assert transition["reward_model_id"] == ""
    assert outcome.classifier_evaluated is False
    assert outcome.success is False
    assert outcome.reward_model_id == ""
    # dones/truncated/masks pass through untouched, as when a sidecar is absent.
    assert transition["dones"] is False
    assert transition["truncated"] is False
    assert transition["masks"] == 1.0

    assert finalizer.classification_count == 0
    assert finalizer.classifier_fault_count == 1
    assert finalizer.classifier_degraded is True
    # Loud, and it names the actual exception.
    assert len(sink.messages) == 1
    assert "FAULTED" in sink.messages[0]
    assert "GPU fault" in sink.messages[0]
    assert "RewardClassifierError" in finalizer.last_classifier_fault


def test_classifier_fault_is_never_conflated_with_an_absent_sidecar():
    sink = _WarningSink()
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([RuntimeError("GPU fault")]), warn=sink
    )

    finalizer(_data(step=0), None)
    assert sink.messages == []
    assert finalizer.classifier_fault_count == 0
    assert finalizer.classifier_degraded is False

    finalizer(_data(step=1), _sidecar())

    # Both steps are unclassified, but only one of them is a fault.
    assert finalizer.transition_count == 2
    assert finalizer.classification_count == 0
    assert finalizer.classifier_fault_count == 1
    assert len(sink.messages) == 1
    assert "no classifier sidecar" not in sink.messages[0]


def test_a_faulted_classifier_warns_exactly_once_not_on_a_cadence():
    """One line per learner process, then silence; the GUI carries the state.

    The operator's words: "로그를 막 계속 띄운다는 말은 아니지? 그냥 한번 뜨고
    gui에서만 보이면 돼."  A cadence -- any cadence -- is a slower way of
    repeating one fact, so this asserts the count over MANY streak periods.
    """
    sink = _WarningSink()
    # A single scripted failure latches the scripted runtime not-ready, exactly
    # like RewardClassifierRuntime.classify does, so every later call faults.
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([RuntimeError("GPU fault")]), warn=sink
    )

    for step in range(UNCLASSIFIED_WARN_STREAK * 5):
        # The helpers fill uint8 pixels from their seed, so keep it in range;
        # content is irrelevant to a classifier that always faults, and the
        # session is unchanged so every one of these is one long streak.
        finalizer(_data(step=step % 200), _sidecar(step % 200))

    # Counters keep accumulating: only the printing is one-shot.
    assert finalizer.classifier_fault_count == UNCLASSIFIED_WARN_STREAK * 5
    assert finalizer.classifier_degraded is True
    # Two, and only because this test double says something genuinely
    # different once it latches ("scripted classifier is not ready", which
    # shares no text with the original failure).  The REAL runtime wraps its
    # original detail instead, and that collapses to a single line -- see
    # test_a_latched_classifier_repeating_its_own_cause_is_not_a_new_line.
    assert len(sink.messages) == 2
    assert "GPU fault" in sink.messages[0]
    assert "FAULTED" in sink.messages[0]
    # It has to say where the live state now lives, or "printed once" reads as
    # "stopped mattering".
    assert "GUI" in sink.messages[0]
    assert "--check" in sink.messages[0]
    # The absent-sidecar cadence is silenced while faults drive the streak --
    # it would otherwise reinstate the exact per-100-steps line just removed.
    assert "no classifier sidecar" not in sink.messages[0]


def test_a_latched_classifier_repeating_its_own_cause_is_not_a_new_line():
    """The production shape: one fault, one line, for the whole process.

    ``RewardClassifierRuntime.classify`` latches ``_ready = False`` and then
    answers every later call with its ORIGINAL detail wrapped in a prefix
    ("reward classifier is not ready: inference failed: ..."), which is the
    same cause, not a new one.
    """
    sink = _WarningSink()
    classifier = _FlakyClassifier(probabilities=[0.1])
    finalizer = RewardTransitionFinalizer(classifier, warn=sink)

    original = "inference failed: ValueError: Incompatible shapes"
    classifier.fail_with = RewardClassifierError(original)
    finalizer(_data(step=0), _sidecar(0))
    classifier.fail_with = RewardClassifierError(
        f"reward classifier is not ready: {original}"
    )
    for step in range(1, 20):
        finalizer(_data(step=step), _sidecar(step))

    assert finalizer.classifier_fault_count == 20
    assert len(sink.messages) == 1
    assert original in sink.messages[0]


def test_a_new_fault_cause_is_worth_one_more_line_up_to_the_cap():
    """Same exception repeating is not news; a different one is."""
    sink = _WarningSink()
    classifier = _FlakyClassifier(probabilities=[0.1])
    finalizer = RewardTransitionFinalizer(classifier, warn=sink)

    step = 0
    for cause in range(CLASSIFIER_FAULT_CAUSE_LIMIT + 3):
        classifier.fail_with = RewardClassifierError(f"cause {cause}")
        for _ in range(3):  # the same cause, repeatedly
            finalizer(_data(step=step), _sidecar(step))
            step += 1

    assert finalizer.classifier_fault_count == (
        CLASSIFIER_FAULT_CAUSE_LIMIT + 3
    ) * 3
    # One line per distinct cause, and then the cap holds: an exception whose
    # text embeds varying numbers cannot become a cadence by the back door.
    assert len(sink.messages) == CLASSIFIER_FAULT_CAUSE_LIMIT
    assert "cause 0" in sink.messages[0]
    assert f"cause {CLASSIFIER_FAULT_CAUSE_LIMIT - 1}" in sink.messages[-1]
    assert "no further fault causes will be printed" in sink.messages[-1]


def test_the_absent_sidecar_streak_warning_is_unchanged_without_faults():
    """The OTHER user of the streak: a sidecar that never arrives at all.

    This signal predates the degrade-instead-of-die change and is about a dead
    sidecar scheduler or an arm that never stands still, so it keeps both its
    cadence and its original wording.
    """
    sink = _WarningSink()
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([0.1] * 4), warn=sink
    )

    for step in range(UNCLASSIFIED_WARN_STREAK * 2):
        finalizer(_data(step=step), None)

    assert len(sink.messages) == 2
    for message in sink.messages:
        assert "no classifier sidecar" in message
        assert "sidecar scheduler running" in message


def test_the_absent_sidecar_cadence_resumes_once_scoring_recovers():
    sink = _WarningSink()
    classifier = _FlakyClassifier(probabilities=[0.1])
    finalizer = RewardTransitionFinalizer(classifier, warn=sink)

    classifier.fail_with = RewardClassifierError("transient CUDA fault")
    step = 0
    for _ in range(UNCLASSIFIED_WARN_STREAK):
        finalizer(_data(step=step), _sidecar(step))
        step += 1
    assert len(sink.messages) == 1  # the one-shot fault line, nothing else

    # One successful classification clears the fault streak, so the older
    # signal is armed again.
    classifier.fail_with = None
    finalizer(_data(step=step), _sidecar(step))
    step += 1
    sink.messages.clear()
    for _ in range(UNCLASSIFIED_WARN_STREAK):
        finalizer(_data(step=step), None)
        step += 1

    assert len(sink.messages) == 1
    assert "no classifier sidecar" in sink.messages[0]


def test_manual_operator_success_still_ends_the_episode_when_classifier_faults():
    """MANUAL is the production default; MARK SUCCESS must survive a fault."""
    sink = _WarningSink()
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([RuntimeError("GPU fault")]), warn=sink
    )

    data, outcome = finalizer(
        _data(auto_success=False, operator_success=True), _sidecar()
    )

    assert outcome.success is True
    assert outcome.reward == 1.0
    assert outcome.done is True
    assert outcome.truncated is False
    assert outcome.mask == 0.0
    assert int(data["transition"]["success"]) == 1
    # ... while the classifier telemetry stays honestly unevaluated.
    assert outcome.classifier_evaluated is False
    assert outcome.reward_model_id == ""
    assert int(data["transition"]["classifier_success"]) == 0
    assert finalizer.classifier_fault_count == 1


def test_auto_mode_can_never_succeed_while_the_classifier_is_faulted():
    """Fail-closed, and the episode-boundary report says so out loud."""
    sink = _WarningSink()
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([RuntimeError("GPU fault")]), warn=sink
    )

    for step in range(3):
        _, outcome = finalizer(
            _data(step=step, auto_success=True), _sidecar(step)
        )
        assert outcome.success is False
        assert outcome.reward == 0.0
    # The step limit is the only remaining way out of an AUTO episode.
    sink.messages.clear()
    _, outcome = finalizer(
        _data(step=3, auto_success=True, truncated=True), _sidecar(3)
    )

    assert outcome.success is False
    assert outcome.truncated is True
    # NOTHING at the episode boundary.  The per-session report would fire once
    # per episode for as long as the fault lasts, which is a cadence measured
    # in episodes instead of steps, and it would say what the GUI is already
    # showing continuously.  "classified NONE" is suppressed for the same
    # reason: while faulted it is a restatement of the fault, not the missing-
    # sidecar signal it exists to raise.
    assert [message for message in sink.messages if "session" in message] == []
    # ... and the fault itself was already announced once, before this episode.
    assert finalizer.classifier_fault_count == 4


def test_a_recovered_classification_clears_the_fault_streak_not_the_total():
    sink = _WarningSink()
    # Faults, then works: the scripted runtime cannot recover, so drive the
    # two outcomes from a classifier whose readiness the test controls.
    classifier = _FlakyClassifier(probabilities=[0.1])
    finalizer = RewardTransitionFinalizer(classifier, warn=sink)

    classifier.fail_with = RewardClassifierError("transient CUDA fault")
    finalizer(_data(step=0), _sidecar(0))
    classifier.fail_with = None
    _, outcome = finalizer(_data(step=1), _sidecar(1))

    assert outcome.classifier_evaluated is True
    assert finalizer.classification_count == 1
    assert finalizer.classifier_fault_count == 1
    # The SAME cause returning is not new information, so it is not reprinted:
    # the operator watches the GUI headline go red again.  The total still
    # counts it, which is what --check and any post-hoc log reading need.
    classifier.fail_with = RewardClassifierError("transient CUDA fault")
    finalizer(_data(step=2), _sidecar(2))
    assert finalizer.classifier_fault_count == 2
    assert len(sink.messages) == 1


def test_health_detail_shows_a_degraded_classifier_to_run_hil_server_check():
    """--check prints Health's `detail`; a broken classifier must appear there."""
    from ur_env.actor_network import ActorSessionService

    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([RuntimeError("GPU fault")]),
        warn=_WarningSink(),
    )
    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        reward_model_id="scripted-reward-v0",
        finalize_transition=finalizer,
    )
    assert service.health() == (True, True, "ready")

    finalizer(_data(), _sidecar())

    alive, ready, detail = service.health()
    # Still serving: a degraded classifier is not a dead server.
    assert alive is True and ready is True
    assert "DEGRADED" in detail
    assert "GPU fault" in detail


@pytest.mark.parametrize(
    "error",
    [
        ValueError("cam1 is not a frozen trunk feature"),
        ActorProtocolError("cam1 must be uint8"),
    ],
)
def test_non_classifier_errors_are_still_raised_rather_than_swallowed(error):
    """The try/except boundary is narrow on purpose.

    ``classify`` validates its input through ``_classifier_model_input``
    BEFORE it touches the model, and that path raises the ValueError family
    (``FrozenTrunkFeatureSchemaError``, ``validate_classifier_frames``).  A
    malformed tensor means the transition itself is wrong, so it must still
    stop the run instead of being recorded as an ordinary zero-reward sample.
    """
    finalizer = RewardTransitionFinalizer(
        _FlakyClassifier(probabilities=[0.9]), warn=_WarningSink()
    )
    finalizer.classifier.fail_with = error

    with pytest.raises(type(error)):
        finalizer(_data(), _sidecar())

    assert finalizer.classifier_fault_count == 0


def test_reward_finalizer_reports_the_instantaneous_viewer_probability():
    """confirmations == 1 must report exactly what the live viewer shows."""
    classifier = ScriptedRewardClassifierRuntime(
        [0.1234, 0.4321], threshold=0.2, reward_model_id="cube-in-cup-v1"
    )
    finalizer = RewardTransitionFinalizer(classifier)
    assert finalizer.confirmations == 1

    _, negative = finalizer(_data(step=0), _sidecar(1))
    _, positive = finalizer(_data(step=1), _sidecar(2))

    assert negative.classifier_probability == pytest.approx(0.1234)
    assert negative.success is False
    assert positive.classifier_probability == pytest.approx(0.4321)
    assert positive.success is True
    for outcome in (negative, positive):
        # The invariant three independent layers re-derive.
        assert outcome.success == (
            outcome.classifier_probability > outcome.classifier_threshold
        )
    assert finalizer.classification_count == 2
    assert finalizer.transition_count == 2


@pytest.mark.parametrize(
    ("probabilities", "expected_reported", "expected_success"),
    [
        # Two strong frames are not enough for three confirmations, and the
        # reported probability is the window floor (the 0.0 pre-fill), so the
        # success/probability invariant still holds.
        ([0.9, 0.9], [0.0, 0.0], [False, False]),
        # Three in a row confirm; the reported value is the weakest of them.
        ([0.9, 0.7, 0.8], [0.0, 0.0, 0.7], [False, False, True]),
        # One dip inside the window blocks confirmation, and the dip itself is
        # what gets reported -- never an average that could read as a success.
        ([0.9, 0.1, 0.8, 0.9], [0.0, 0.0, 0.1, 0.1], [False, False, False, False]),
    ],
)
def test_reward_finalizer_n_of_n_smoothing_keeps_the_success_invariant(
    probabilities, expected_reported, expected_success
):
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime(
            probabilities, threshold=0.2, reward_model_id="cube-in-cup-v1"
        ),
        confirmations=3,
    )

    for index, (reported, success) in enumerate(
        zip(expected_reported, expected_success)
    ):
        data, outcome = finalizer(_data(step=index), _sidecar(index))
        assert outcome.classifier_probability == pytest.approx(reported)
        assert outcome.success is success
        assert outcome.classifier_evaluated is True
        assert outcome.success == (
            outcome.classifier_probability > outcome.classifier_threshold
        )
        assert bool(data["transition"]["classifier_success"]) is success
        if success:
            assert outcome.reward == 1.0 and outcome.done is True
        else:
            assert outcome.reward == 0.0


def test_reward_finalizer_confirmation_window_does_not_cross_episodes():
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime(
            [0.9] * 4, threshold=0.2, reward_model_id="cube-in-cup-v1"
        ),
        confirmations=2,
        warn=_WarningSink(),
    )

    finalizer(_data(step=0), _sidecar(0))
    # A local time limit ends the episode; the half-filled window must not be
    # completed by the first frame of the next one.
    _, terminal = finalizer(_data(step=1, truncated=True), _sidecar(1))
    _, first_of_next = finalizer(
        _data(step=2, session_id="session-1"), _sidecar(2)
    )
    _, second_of_next = finalizer(
        _data(step=3, session_id="session-1"), _sidecar(3)
    )

    assert terminal.success is True
    assert first_of_next.success is False
    assert first_of_next.classifier_probability == 0.0
    assert second_of_next.success is True


@pytest.mark.parametrize(
    ("local_done", "local_truncated", "local_reward"),
    [(False, False, 0.0), (False, False, 1.0), (True, False, 0.0), (False, True, 0.0)],
)
def test_unclassified_transition_field_table(
    local_done, local_truncated, local_reward
):
    """The exact contract for a step the actor did not ask us to classify."""
    classifier = ScriptedRewardClassifierRuntime([0.9], threshold=0.2)
    finalizer = RewardTransitionFinalizer(classifier, warn=_WarningSink())

    data, outcome = finalizer(
        _data(
            done=local_done, truncated=local_truncated, reward=local_reward
        ),
        None,
    )

    transition = data["transition"]
    # rewards: forced to 0, discarding any locally proposed positive reward.
    assert transition["rewards"] == 0.0
    assert outcome.reward == 0.0
    # masks / dones / truncated: the local proposal survives untouched, which
    # is what keeps ReplayIngress._convert's mask-vs-done check satisfied.
    assert transition["dones"] is local_done
    assert transition["truncated"] is local_truncated
    assert transition["masks"] == (0.0 if local_done else 1.0)
    # classifier_*: all zero / empty, which _convert REQUIRES when unevaluated.
    assert int(transition["classifier_evaluated"]) == 0
    assert float(transition["classifier_probability"]) == 0.0
    assert float(transition["classifier_threshold"]) == 0.0
    assert int(transition["classifier_success"]) == 0
    assert transition["reward_model_id"] == ""
    assert outcome.classifier_evaluated is False
    assert outcome.success is False
    assert outcome.classifier_probability == 0.0
    assert outcome.classifier_threshold == 0.0
    assert outcome.reward_model_id == ""
    # Nothing was consumed from the classifier: no sidecar, no inference.
    assert classifier.evaluation_count == 0
    assert finalizer.classification_count == 0


def test_unclassified_transitions_route_into_replay_unchanged():
    """_convert must already accept classifier_evaluated=0; no relaxation added."""
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=4, intervention_capacity=4, store_factory=factory
    )

    ingress(_finalized_data(step=0, classified=False), False)
    ingress(_finalized_data(step=1, classified=True), False)

    stored = ingress.replay_store.items
    assert [int(item["classifier_evaluated"]) for item in stored] == [0, 1]
    assert float(stored[0]["classifier_probability"]) == 0.0
    assert float(stored[0]["classifier_threshold"]) == 0.0
    assert int(stored[0]["classifier_success"]) == 0
    assert float(stored[0]["rewards"]) == 0.0
    assert ingress.status().replay_size == 2
    assert [record.reward_model_id for record in ingress.replay_sidecar()] == [
        "",
        "scripted-reward-v0",
    ]


def test_unevaluated_transition_may_not_smuggle_classifier_values():
    """The existing guard, re-pinned: it is what makes a 0-reward honest."""
    ingress = ReplayIngress(
        replay_capacity=4, intervention_capacity=4, store_factory=_StoreFactory()
    )
    smuggled = _finalized_data(step=0, classified=False)
    smuggled["transition"]["classifier_probability"] = 0.9

    with pytest.raises(ActorProtocolError, match="unevaluated transition"):
        ingress(smuggled, False)


def test_zero_classification_session_warns_loudly_at_the_episode_boundary():
    sink = _WarningSink()
    # 0.1 stays under the 0.2 threshold, so the classified step below is an
    # ordinary non-terminal one and the episode ends where the test says it does.
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([0.1]), warn=sink
    )

    finalizer(_data(step=0), None)
    assert sink.messages == []
    finalizer(_data(step=1, done=True), None)

    assert len(sink.messages) == 1
    assert "session-0" in sink.messages[0]
    assert "classified NONE" in sink.messages[0]

    # A session that DID classify stays quiet.
    sink.messages.clear()
    finalizer(_data(step=2, session_id="session-1"), _sidecar())
    finalizer(_data(step=3, session_id="session-1", done=True), None)
    assert sink.messages == []


def test_long_unclassified_streak_warns_even_inside_one_episode():
    sink = _WarningSink()
    finalizer = RewardTransitionFinalizer(
        ScriptedRewardClassifierRuntime([0.9]), warn=sink
    )

    for step in range(UNCLASSIFIED_WARN_STREAK - 1):
        finalizer(_data(step=step), None)
    assert sink.messages == []
    finalizer(_data(step=UNCLASSIFIED_WARN_STREAK - 1), None)

    assert len(sink.messages) == 1
    assert "no classifier sidecar" in sink.messages[0]
    # One classification resets the streak counter.
    finalizer(_data(step=UNCLASSIFIED_WARN_STREAK), _sidecar())
    finalizer(_data(step=UNCLASSIFIED_WARN_STREAK + 1), None)
    assert len(sink.messages) == 1


def test_corrupt_sidecar_bytes_fail_the_transition_instead_of_going_silent():
    """Re-pinned as the OTHER side of the degrade boundary.

    ``_classifier_frames`` sits outside the try that degrades classifier
    faults: undecodable bytes mean the actor's payload is corrupt, not that
    the model has an opinion it cannot express, so this must still raise.
    """
    finalizer = RewardTransitionFinalizer(ScriptedRewardClassifierRuntime([0.9]))
    corrupt = {
        "cam1_jpeg": np.frombuffer(b"not-a-jpeg", dtype=np.uint8),
        "cam2_jpeg": np.frombuffer(_jpeg(0), dtype=np.uint8),
    }

    with pytest.raises(RewardClassifierError, match="decode failed"):
        finalizer(_data(), corrupt)


def test_replay_ingress_routes_all_and_interventions_with_gap_boundaries():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=8,
        intervention_capacity=4,
        store_factory=factory,
    )
    assert factory.stores == [ingress.replay_store, ingress.intervention_store]

    ingress(_finalized_data(step=0, intervened=True), True)
    ingress(_finalized_data(step=1, intervened=False), False)
    ingress(_finalized_data(step=2, intervened=True), True)

    replay, intervention = factory.stores
    assert replay.started_sequences == [True, False, False]
    assert intervention.started_sequences == [True, True]
    assert len(replay.items) == 3
    assert len(intervention.items) == 2
    assert [int(item["intervened"]) for item in replay.items] == [1, 0, 1]
    assert all(item["policy_actions"].shape == (7,) for item in replay.items)
    assert set(ingress._NUMERIC_METADATA).issubset(replay.dataset_dict)

    status = ingress.status()
    assert status.replay_size == 3
    assert status.intervention_size == 2
    assert status.replay_insert_count == 3
    assert status.intervention_insert_count == 2
    assert status.last_transition_id == "transition-2"
    assert status.last_env_step == 2
    assert [record.transition_id for record in ingress.replay_sidecar()] == [
        "transition-0",
        "transition-1",
        "transition-2",
    ]


def test_replay_ingress_uses_truncation_only_as_stack_boundary():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=2,
        store_factory=factory,
    )
    ingress(_finalized_data(step=0, truncated=True), False)

    stored = ingress.replay_store.items[0]
    assert bool(stored["dones"]) is True
    assert bool(stored["terminated"]) is False
    assert bool(stored["truncated"]) is True
    assert float(stored["masks"]) == 1.0


def test_replay_ingress_duplicate_is_idempotent_and_collision_rejected():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=4,
        store_factory=factory,
    )
    original = _finalized_data(step=0, intervened=True)

    ingress(original, True)
    ingress(copy.deepcopy(original), True)

    assert ingress.status().replay_insert_count == 1
    assert ingress.status().intervention_insert_count == 1
    assert len(ingress.replay_store.items) == 1
    assert len(ingress.intervention_store.items) == 1
    conflicting = copy.deepcopy(original)
    conflicting["transition"]["actions"][0] = -0.5
    with pytest.raises(ActorProtocolError, match="collision"):
        ingress(conflicting, True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["meta"].__setitem__("timestamp_ns", 42),
        lambda data: data["transition"]["next_observations"]["cam1"].__setitem__(
            (0, 0, 0, 0), 99
        ),
        lambda data: data["transition"].__setitem__(
            "classifier_probability", 0.2
        ),
    ],
)
def test_replay_ingress_id_collision_covers_metadata_and_observation_bytes(
    mutation,
):
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=4,
        store_factory=_StoreFactory(),
    )
    original = _finalized_data(step=0)
    ingress(original, False)
    conflicting = copy.deepcopy(original)
    mutation(conflicting)

    with pytest.raises(ActorProtocolError, match="collision"):
        ingress(conflicting, False)


def test_replay_ingress_retries_only_missing_route_after_partial_failure():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=4,
        store_factory=factory,
    )
    data = _finalized_data(step=0, intervened=True)
    ingress.intervention_store.fail_next_insert = True

    with pytest.raises(BufferError, match="scripted insert failure"):
        ingress(data, True)
    assert len(ingress.replay_store.items) == 1
    assert len(ingress.intervention_store.items) == 0

    ingress(data, True)

    assert len(ingress.replay_store.items) == 1
    assert len(ingress.intervention_store.items) == 1
    assert ingress.status().replay_insert_count == 1
    assert ingress.status().intervention_insert_count == 1


def test_replay_ingress_status_reports_logical_circular_overwrites():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=2,
        intervention_capacity=1,
        store_factory=factory,
        ledger_capacity=2,
    )
    for step in range(3):
        ingress(_finalized_data(step=step, intervened=True), True)

    status = ingress.status()
    assert status.replay_size == 2
    assert status.intervention_size == 1
    assert status.replay_insert_count == 3
    assert status.intervention_insert_count == 3
    assert status.replay_overwrite_count == 1
    assert status.intervention_overwrite_count == 2
    assert len(ingress.replay_sidecar()) == 2
    assert len(ingress.intervention_sidecar()) == 1


def test_replay_ingress_learner_mode_requires_explicit_grasp_penalty():
    factory = _StoreFactory()
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=4,
        store_factory=factory,
        learner_mode=True,
    )
    missing = _finalized_data(step=0)
    with pytest.raises(ActorProtocolError, match="required in learner mode"):
        ingress(missing, False)

    present = _finalized_data(step=1)
    present["transition"]["grasp_penalty"] = -0.02
    ingress(present, False)
    assert ingress.status().replay_size == 1


def test_replay_ingress_learner_mode_rejects_wrong_grasp_penalty_value():
    ingress = ReplayIngress(
        replay_capacity=4,
        intervention_capacity=4,
        store_factory=_StoreFactory(),
        learner_mode=True,
        expected_grasp_penalty=-0.07,
    )
    wrong = _finalized_data(step=0)
    wrong["transition"]["grasp_penalty"] = -0.02
    with pytest.raises(ActorProtocolError, match="configured penalty -0.07"):
        ingress(wrong, False)

    no_penalty_event = _finalized_data(step=1)
    no_penalty_event["transition"]["grasp_penalty"] = 0.0
    ingress(no_penalty_event, False)
    matching = _finalized_data(step=2)
    matching["transition"]["grasp_penalty"] = -0.07
    ingress(matching, False)
    assert ingress.status().replay_size == 2


@pytest.mark.parametrize("value", [True, np.nan, 0.01, [0.0]])
def test_replay_ingress_rejects_invalid_expected_grasp_penalty(value):
    with pytest.raises(ValueError, match="expected_grasp_penalty"):
        ReplayIngress(
            replay_capacity=4,
            intervention_capacity=4,
            store_factory=_StoreFactory(),
            learner_mode=True,
            expected_grasp_penalty=value,
        )


def _actual_hil_serl_root() -> str:
    root = os.environ.get(
        "HIL_SERL_ROOT", os.path.join(_REPO_ROOT, "third_party", "hil-serl")
    )
    launcher = os.path.join(
        root, "serl_launcher", "serl_launcher", "data", "data_store.py"
    )
    if not os.path.isfile(launcher):
        pytest.skip("pinned HIL-SERL submodule is not initialized")
    for dependency in ("jax", "flax", "agentlace"):
        pytest.importorskip(dependency)
    return root


def test_actual_upstream_buffers_sample_packed_batch_and_keep_gap_boundaries():
    hil_serl_root = _actual_hil_serl_root()
    ingress = ReplayIngress(
        replay_capacity=32,
        intervention_capacity=16,
        hil_serl_root=hil_serl_root,
    )
    ingress(
        _finalized_data(
            step=0, intervened=True, source_value=10, next_value=11
        ),
        True,
    )
    ingress(
        _finalized_data(
            step=1, intervened=False, source_value=11, next_value=12
        ),
        False,
    )
    ingress(
        _finalized_data(
            step=2, intervened=True, source_value=20, next_value=21
        ),
        True,
    )

    replay_batch = ingress.sample_replay(batch_size=8)
    assert replay_batch["observations"]["cam1"].shape == (
        8,
        2,
        128,
        128,
        3,
    )
    assert replay_batch["observations"]["state"].shape == (8, 1, 19)
    assert replay_batch["policy_actions"].shape == (8, 7)
    assert replay_batch["classifier_probability"].shape == (8,)

    # Both intervention samples are isolated sequences.  Packed frames must be
    # their actual O(t),O(t+1) pairs, never the prior policy-only frame.
    intervention_batch = ingress.sample_intervention(batch_size=256)
    pixels = np.asarray(intervention_batch["observations"]["cam1"])
    pairs = set(
        zip(
            pixels[:, 0, 0, 0, 0].tolist(),
            pixels[:, 1, 0, 0, 0].tolist(),
        )
    )
    assert pairs == {(10, 11), (20, 21)}


def test_actual_upstream_capacity_counts_logical_transitions_not_bootstrap_frames():
    hil_serl_root = _actual_hil_serl_root()
    ingress = ReplayIngress(
        replay_capacity=2,
        intervention_capacity=2,
        hil_serl_root=hil_serl_root,
    )
    for step, value in enumerate((10, 20, 30)):
        ingress(
            _finalized_data(
                step=step,
                intervened=True,
                source_value=value,
                next_value=value + 1,
                session_id=f"isolated-session-{step}",
            ),
            True,
        )

    status = ingress.status()
    assert status.replay_size == 2
    assert status.intervention_size == 2
    assert status.replay_insert_count == 3
    assert status.intervention_insert_count == 3
    assert status.replay_overwrite_count == 1
    assert status.intervention_overwrite_count == 1
    assert np.count_nonzero(ingress.replay_store._is_correct_index) == 2
    assert np.count_nonzero(ingress.intervention_store._is_correct_index) == 2

    for sample in (
        ingress.sample_replay(batch_size=256),
        ingress.sample_intervention(batch_size=256),
    ):
        pixels = np.asarray(sample["observations"]["cam1"])
        pairs = set(
            zip(
                pixels[:, 0, 0, 0, 0].tolist(),
                pixels[:, 1, 0, 0, 0].tolist(),
            )
        )
        assert pairs == {(20, 21), (30, 31)}


@pytest.mark.parametrize(
    "sessions",
    [
        ["continuous"] * 9,
        ["a", "a", "b", "b", "b", "c", "c", "d", "d"],
    ],
)
def test_actual_upstream_logical_capacity_survives_wraps_and_mixed_boundaries(
    sessions,
):
    hil_serl_root = _actual_hil_serl_root()
    ingress = ReplayIngress(
        replay_capacity=2,
        intervention_capacity=2,
        hil_serl_root=hil_serl_root,
    )
    for step, session_id in enumerate(sessions):
        ingress(
            _finalized_data(
                step=step,
                intervened=True,
                source_value=10 + step,
                next_value=11 + step,
                session_id=session_id,
            ),
            True,
        )

    status = ingress.status()
    expected_overwrites = len(sessions) - 2
    assert status.replay_size == status.replay_capacity == 2
    assert status.intervention_size == status.intervention_capacity == 2
    assert status.replay_overwrite_count == expected_overwrites
    assert status.intervention_overwrite_count == expected_overwrites
    expected_pairs = {
        (10 + len(sessions) - 2, 11 + len(sessions) - 2),
        (10 + len(sessions) - 1, 11 + len(sessions) - 1),
    }
    for store, sample in (
        (ingress.replay_store, ingress.sample_replay(batch_size=512)),
        (
            ingress.intervention_store,
            ingress.sample_intervention(batch_size=512),
        ),
    ):
        assert np.count_nonzero(store._is_correct_index) == 2
        pixels = np.asarray(sample["observations"]["cam1"])
        pairs = set(
            zip(
                pixels[:, 0, 0, 0, 0].tolist(),
                pixels[:, 1, 0, 0, 0].tolist(),
            )
        )
        assert pairs == expected_pairs
