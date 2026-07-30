"""Server-side reward finalization and HIL-SERL replay ingress.

This module deliberately contains no network transport.  It supplies the
callbacks consumed by :class:`ur_env.actor_network.ActorSessionService`:

* :class:`FakeActionRuntime` is a safe placeholder policy for receive-only
  bring-up;
* :class:`RewardClassifierRuntime` owns the cube-in-cup Flax checkpoint and
  classifies the UNCROPPED sidecar frames the actor attached to ``O(t+1)`` --
  never the cropped policy observation, see :func:`validate_classifier_frames`;
* :class:`RewardTransitionFinalizer` makes reward/termination authoritative on
  the server, including for the majority of transitions that arrive with no
  sidecar and are therefore left unclassified and unrewarded; and
* :class:`ReplayIngress` routes the finalized transition into the actual
  upstream HIL-SERL memory-efficient replay datastores.

The heavy JAX/Flax/Agentlace/upstream imports are lazy.  Importing this module
on the robot laptop therefore remains dependency-light; only the server needs
the pinned learner environment.
"""

from __future__ import annotations

from collections import OrderedDict, deque
import copy
from dataclasses import dataclass
import hashlib
import math
import os
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Protocol

import numpy as np

from ur_env.actor_network import (
    ActorProtocolError,
    BufferStatus,
    TransitionOutcome,
)
from ur_env.compat import configure_flax_local_io
from ur_env.observation_schema import (
    CANONICAL_OBSERVATION_SPEC,
    validate_canonical_observation,
)


IMAGE_KEYS = ("cam1", "cam2")
ACTION_SHAPE = (7,)
DEFAULT_REPLAY_CAPACITY = 50_000
DEFAULT_INTERVENTION_CAPACITY = 10_000
# Re-measured 2026-07-29 with code-enforced frame-content-hash leak detection,
# which invalidated the 07-28 numbers that used to live here.  On the deployed
# cube_in_cup_all3 checkpoint the genuinely held-out set is 266 success / 470
# failure frames, and the false-positive rate is exactly 0% at 0.85, 0.5 AND
# 0.2 alike -- FPR cannot distinguish the three.  Recall can: the worst
# cross-validation fold on the newer 0724 domain goes 44.3% -> 65.7% between
# 0.85 and 0.5, an order of magnitude above the 2.8pp retrain seed noise.
#
# 0.5 is the current conservative deployment point.  It preserves the measured
# 0% held-out false-positive rate while restoring more margin above the hardest
# known negative than the previous 0.2 setting.  The full measurements and the
# decision history remain in REWARD_CLASSIFIER_THRESHOLD_KO.md.
#
# This value feeds the learner fingerprint, so changing it breaks resume of
# checkpoints trained under the old value.
# See REWARD_CLASSIFIER_THRESHOLD_KO.md.
DEFAULT_REWARD_THRESHOLD = 0.5


class ReceiveRuntimeError(RuntimeError):
    """A server runtime dependency or inference operation failed."""


class RewardClassifierError(ReceiveRuntimeError):
    """The reward classifier cannot safely produce a result."""


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ActorProtocolError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result > np.iinfo(np.int64).max:
        raise ActorProtocolError(f"{name} must be a non-negative signed int64")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ActorProtocolError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ActorProtocolError(f"{name} must be finite")
    return result


def _binary_flag(value: Any, *, name: str) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if value not in (False, True, 0, 1):
        raise ActorProtocolError(f"{name} must be 0 or 1")
    return bool(value)


def _required_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActorProtocolError(f"{name} is required")
    return value


def validate_reward_threshold(value: Any) -> float:
    """Return a finite probability threshold in ``[0, 1]``."""
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("reward threshold must be numeric") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("reward threshold must be finite and in [0, 1]")
    return threshold


def sigmoid_probability(logit: Any) -> float:
    """Convert one binary-classifier logit without overflowing ``exp``."""
    try:
        value = float(logit)
    except (TypeError, ValueError) as exc:
        raise RewardClassifierError("classifier logit must be scalar") from exc
    if not math.isfinite(value):
        raise RewardClassifierError("classifier logit must be finite")
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def checkpoint_sha256(path: str) -> str:
    """Pin one reward-classifier checkpoint, file or orbax directory.

    G19: this used to require ``os.path.isfile()``, which made the canonical
    checkpoint impossible to pin at all — ``classifier_ckpt/cube_in_cup_all3/``
    ``checkpoint_150`` is an orbax OCDBT *directory*, so
    :class:`RewardClassifierRuntime` died with ``FileNotFoundError`` before it
    ever opened the classifier.  ``directory_sha256`` covers both shapes and
    keeps the single-file digest byte-identical to the old implementation, so
    every SHA already written into a runbook or a CLI default stays valid.
    """
    from ur_env.classifier_sidecar import directory_sha256

    return directory_sha256(path)


def _validated_sha256(value: str) -> str:
    expected = str(value).lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
    return expected


def validate_classifier_frames(
    frames: Mapping[str, Any], *, copy: bool = False
) -> dict[str, np.ndarray]:
    """Validate the decoded sidecar frames the classifier is allowed to see.

    These are NOT the policy observation.  They are produced by
    ``ur_env.classifier_sidecar.decode_classifier_frames`` from the camera's
    original JPEG bytes with the same preprocessing the live viewer uses (the
    viewer was validated on real hardware 2026-07-29), which is exactly what
    makes the server verdict agree with the viewer.  The policy's ``IMAGE_CROP``
    is deliberately absent here and must stay absent: the crop is a measured
    dataset property the policy is the first consumer of, while this checkpoint
    was trained on uncropped full frames.

    Shapes are read from the canonical spec on purpose — the classifier's image
    tensors and the policy's have the same geometry even though their CONTENT
    differs, and a schema change must not leave the two silently out of step.
    """
    if not isinstance(frames, Mapping):
        raise ActorProtocolError("classifier frames must be a mapping")
    actual = set(frames)
    expected = set(IMAGE_KEYS)
    if actual != expected:
        raise ActorProtocolError(
            "classifier frames keys mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    result: dict[str, np.ndarray] = {}
    for key in IMAGE_KEYS:
        expected_dtype, expected_shape = CANONICAL_OBSERVATION_SPEC[key]
        array = np.asarray(frames[key])
        if array.dtype != expected_dtype:
            raise ActorProtocolError(
                f"classifier frame {key!r} must have dtype "
                f"{expected_dtype.name}, got {array.dtype}"
            )
        if array.shape != expected_shape:
            raise ActorProtocolError(
                f"classifier frame {key!r} must have shape {expected_shape}, "
                f"got {array.shape}"
            )
        contiguous = np.ascontiguousarray(array)
        result[key] = contiguous.copy() if copy else contiguous
    return result


def _classifier_model_input(
    frames: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    validated = validate_classifier_frames(frames)
    # The classifier checkpoint was trained with use_proprio=False and a
    # one-element dummy state.  Policy/replay state remains the canonical 19-D
    # tensor and is intentionally not fed to this checkpoint.
    return {
        "state": np.zeros((1, 1), dtype=np.float32),
        "cam1": validated["cam1"],
        "cam2": validated["cam2"],
    }


class FakeActionRuntime:
    """Receive-only policy callback, defaulting to version-0 zero actions.

    Supplying ``action_sequence`` is useful for deterministic integration
    tests.  A finite scripted sequence fails closed when exhausted rather than
    silently reusing an action.
    """

    model_id = "fake-zero-action-v0"

    def __init__(
        self,
        action_sequence: Optional[Iterable[Any]] = None,
        *,
        policy_version: int = 0,
    ) -> None:
        self._policy_version = _nonnegative_int(
            policy_version, name="policy_version"
        )
        self._actions: Optional[tuple[np.ndarray, ...]]
        if action_sequence is None:
            self._actions = None
        else:
            actions = tuple(
                self._validate_action(action, name=f"action_sequence[{index}]")
                for index, action in enumerate(action_sequence)
            )
            if not actions:
                raise ValueError("action_sequence must not be empty")
            self._actions = actions
        self._index = 0
        self._lock = threading.Lock()

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._index

    def __call__(
        self, observation: Mapping[str, Any], deterministic: bool
    ) -> tuple[np.ndarray, int]:
        del deterministic
        validate_canonical_observation(observation, copy=False)
        with self._lock:
            if self._actions is None:
                action = np.zeros(ACTION_SHAPE, dtype=np.float32)
            else:
                if self._index >= len(self._actions):
                    raise ReceiveRuntimeError("scripted fake action sequence exhausted")
                action = self._actions[self._index]
            self._index += 1
            return action.copy(), self._policy_version

    @staticmethod
    def _validate_action(value: Any, *, name: str) -> np.ndarray:
        action = np.asarray(value, dtype=np.float32)
        if action.shape != ACTION_SHAPE:
            raise ValueError(f"{name} must have shape {ACTION_SHAPE}")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{name} must contain only finite values")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError(f"{name} must stay within [-1, 1]")
        return np.ascontiguousarray(action).copy()


@dataclass(frozen=True)
class ClassificationResult:
    probability: float
    threshold: float
    success: bool
    reward_model_id: str
    inference_ms: float


class RewardClassifier(Protocol):
    threshold: float
    reward_model_id: str

    def classify(self, frames: Mapping[str, Any]) -> ClassificationResult:
        """Classify ONE decoded sidecar frame pair; see validate_classifier_frames.

        ``ClassificationResult.probability`` is always the INSTANTANEOUS sigmoid
        of this frame pair — identical to what the live viewer prints for the
        same frames.  Any temporal smoothing belongs to the caller
        (:class:`RewardTransitionFinalizer`), never here.
        """
        ...


class RewardClassifierRuntime:
    """Load, verify, warm up, and run one Flax reward classifier."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        expected_sha256: str,
        threshold: float = DEFAULT_REWARD_THRESHOLD,
        reward_model_id: Optional[str] = None,
        hil_serl_root: Optional[str] = None,
        resnet_source_path: Optional[os.PathLike[str] | str] = None,
        resnet_cache_path: Optional[os.PathLike[str] | str] = None,
        classifier_loader: Optional[
            Callable[[Mapping[str, np.ndarray]], Callable[[Mapping[str, Any]], Any]]
        ] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        expected = _validated_sha256(expected_sha256)
        actual = checkpoint_sha256(self.checkpoint_path)
        if actual != expected:
            raise RewardClassifierError(
                "reward classifier checkpoint SHA256 mismatch: "
                f"expected={expected}, actual={actual}"
            )
        self.checkpoint_sha256 = actual
        self.threshold = validate_reward_threshold(threshold)
        self.reward_model_id = reward_model_id or (
            f"reward-classifier:{os.path.basename(self.checkpoint_path)}@{actual[:12]}"
        )
        if not self.reward_model_id:
            raise ValueError("reward_model_id is required")
        self._clock = clock
        self._lock = threading.Lock()
        self._ready = False
        self._fault_detail = "loading"
        self._evaluation_count = 0
        self._warmup_ms = 0.0

        sample = {
            "state": np.zeros((1, 1), dtype=np.float32),
            "cam1": np.zeros(CANONICAL_OBSERVATION_SPEC["cam1"][1], dtype=np.uint8),
            "cam2": np.zeros(CANONICAL_OBSERVATION_SPEC["cam2"][1], dtype=np.uint8),
        }
        try:
            loader = classifier_loader or self._upstream_loader(
                hil_serl_root,
                resnet_source_path=resnet_source_path,
                resnet_cache_path=resnet_cache_path,
            )
            self._classifier = loader(sample)
            started = self._clock()
            warmup = self._classifier(sample)
            if hasattr(warmup, "block_until_ready"):
                warmup.block_until_ready()
            self._scalar_logit(warmup)
            self._warmup_ms = (self._clock() - started) * 1000.0
            if not math.isfinite(self._warmup_ms) or self._warmup_ms < 0.0:
                raise RewardClassifierError("classifier warmup timing is invalid")
        except Exception as exc:
            self._fault_detail = f"load/warmup failed: {type(exc).__name__}: {exc}"
            if isinstance(exc, RewardClassifierError):
                raise
            raise RewardClassifierError(self._fault_detail) from exc
        self._ready = True
        self._fault_detail = ""

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def fault_detail(self) -> str:
        with self._lock:
            return self._fault_detail

    @property
    def evaluation_count(self) -> int:
        with self._lock:
            return self._evaluation_count

    @property
    def warmup_ms(self) -> float:
        return self._warmup_ms

    def classify(self, frames: Mapping[str, Any]) -> ClassificationResult:
        classifier_observation = _classifier_model_input(frames)
        with self._lock:
            if not self._ready:
                raise RewardClassifierError(
                    f"reward classifier is not ready: {self._fault_detail}"
                )
            started = self._clock()
            try:
                output = self._classifier(classifier_observation)
                if hasattr(output, "block_until_ready"):
                    output.block_until_ready()
                logit = self._scalar_logit(output)
                probability = sigmoid_probability(logit)
                inference_ms = (self._clock() - started) * 1000.0
                if not math.isfinite(inference_ms) or inference_ms < 0.0:
                    raise RewardClassifierError("classifier timing is invalid")
            except Exception as exc:
                self._ready = False
                self._fault_detail = (
                    f"inference failed: {type(exc).__name__}: {exc}"
                )
                if isinstance(exc, RewardClassifierError):
                    raise
                raise RewardClassifierError(self._fault_detail) from exc
            self._evaluation_count += 1
            return ClassificationResult(
                probability=probability,
                threshold=self.threshold,
                success=probability > self.threshold,
                reward_model_id=self.reward_model_id,
                inference_ms=inference_ms,
            )

    def _upstream_loader(
        self,
        hil_serl_root: Optional[str],
        *,
        resnet_source_path: Optional[os.PathLike[str] | str] = None,
        resnet_cache_path: Optional[os.PathLike[str] | str] = None,
    ) -> Callable[[Mapping[str, np.ndarray]], Callable[[Mapping[str, Any]], Any]]:
        if hil_serl_root:
            upstream_root = os.path.abspath(os.path.expanduser(hil_serl_root))
            launcher_root = os.path.join(upstream_root, "serl_launcher")
            if launcher_root not in sys.path:
                sys.path.insert(0, launcher_root)
        else:
            upstream_root = None
        try:
            import jax
            from ur_env.learner.agent import (
                default_hil_serl_root,
                ensure_resnet10_cache,
            )
            from serl_launcher.networks.reward_classifier import load_classifier_func
        except ImportError as exc:
            raise ReceiveRuntimeError(
                "JAX/Flax and the pinned HIL-SERL serl_launcher package are "
                "required on the receive server"
            ) from exc

        if upstream_root is None:
            upstream_root = os.fspath(default_hil_serl_root())
        resnet_source = os.path.abspath(
            os.path.expanduser(
                resnet_source_path
                or os.path.join(
                    upstream_root,
                    "examples",
                    "experiments",
                    "resnet10_params.pkl",
                )
            )
        )
        resnet_cache = (
            os.path.abspath(os.path.expanduser(resnet_cache_path))
            if resnet_cache_path
            else None
        )
        upstream_cache = os.path.abspath(
            os.path.expanduser("~/.serl/resnet10_params.pkl")
        )

        def load(
            sample: Mapping[str, np.ndarray]
        ) -> Callable[[Mapping[str, Any]], Any]:
            try:
                # Upstream reads ~/.serl/resnet10_params.pkl directly while
                # constructing the classifier.  Stage it only from the
                # repository asset after both source and any existing cache
                # satisfy the immutable SHA-256 contract.
                verified_cache = ensure_resnet10_cache(
                    source_path=resnet_source,
                    cache_path=resnet_cache,
                )
                # The unmodified upstream classifier opens this fixed path.
                # When the learner selected a custom cache, mirror only its
                # already-verified bytes into the fixed cache.  The helper's
                # exclusive create/no-overwrite contract still applies.
                if os.path.abspath(os.fspath(verified_cache)) != upstream_cache:
                    ensure_resnet10_cache(
                        source_path=verified_cache,
                        cache_path=upstream_cache,
                    )
            except Exception as exc:
                raise RewardClassifierError(
                    "verified ResNet-10 setup failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            configure_flax_local_io()
            return load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=dict(sample),
                image_keys=list(IMAGE_KEYS),
                checkpoint_path=self.checkpoint_path,
            )

        return load

    @staticmethod
    def _scalar_logit(value: Any) -> float:
        array = np.asarray(value)
        if array.size != 1:
            raise RewardClassifierError(
                f"classifier must return one logit, got shape {array.shape}"
            )
        logit = float(array.reshape(()))
        if not math.isfinite(logit):
            raise RewardClassifierError("classifier logit must be finite")
        return logit


class ScriptedRewardClassifierRuntime:
    """Dependency-free classifier used by protocol and ingress tests."""

    def __init__(
        self,
        probabilities: Iterable[Any],
        *,
        threshold: float = DEFAULT_REWARD_THRESHOLD,
        reward_model_id: str = "scripted-reward-v0",
    ) -> None:
        values = tuple(probabilities)
        if not values:
            raise ValueError("probabilities must not be empty")
        self._values = values
        self.threshold = validate_reward_threshold(threshold)
        self.reward_model_id = _required_text(
            reward_model_id, name="reward_model_id"
        )
        self._index = 0
        self._ready = True
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def evaluation_count(self) -> int:
        with self._lock:
            return self._index

    def classify(self, frames: Mapping[str, Any]) -> ClassificationResult:
        # Validate the same input contract the real runtime enforces, so a test
        # that hands the classifier a policy observation by mistake fails here
        # instead of quietly "passing" against a checkpoint it never saw.
        validate_classifier_frames(frames)
        with self._lock:
            if not self._ready:
                raise RewardClassifierError("scripted classifier is not ready")
            if self._index >= len(self._values):
                self._ready = False
                raise RewardClassifierError("scripted classifier sequence exhausted")
            value = self._values[self._index]
            self._index += 1
            if isinstance(value, BaseException):
                self._ready = False
                raise RewardClassifierError(
                    f"scripted classifier failed: {type(value).__name__}: {value}"
                ) from value
            try:
                probability = float(value)
            except (TypeError, ValueError) as exc:
                self._ready = False
                raise RewardClassifierError(
                    "scripted probability must be numeric"
                ) from exc
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                self._ready = False
                raise RewardClassifierError(
                    "scripted probability must be finite and in [0, 1]"
                )
            return ClassificationResult(
                probability=probability,
                threshold=self.threshold,
                success=probability > self.threshold,
                reward_model_id=self.reward_model_id,
                inference_ms=0.0,
            )


#: Number of classifications that must agree before success is declared.
#: 1 == no smoothing, and that is the operator's decision for the first run.
DEFAULT_CLASSIFIER_CONFIRMATIONS = 1

#: Consecutive unclassified transitions that trigger a loud operator warning.
#: At the 10 Hz control loop this is ten seconds during which nothing on this
#: server could ever have produced a reward.
UNCLASSIFIED_WARN_STREAK = 100


def _emit_operator_warning(message: str) -> None:
    """Write one operator-visible line to stderr.

    Deliberately not the ``logging`` module: nothing else in ``ur_env`` config-
    ures logging, so a logger call would land in a null handler and the silent-
    failure mode this warning exists to prevent would itself be silent.
    """
    print(f"[reward-classifier] WARNING: {message}", file=sys.stderr, flush=True)


class RewardTransitionFinalizer:
    """Apply one server-authoritative reward result to one transition.

    WHAT IT CLASSIFIES
    ------------------
    The uncropped sidecar frames the actor attached to O(t+1), NOT
    ``transition["next_observations"]``.  The policy observation is cropped by
    ``IMAGE_CROP`` while this checkpoint was trained on full frames; classifying
    the policy observation cost recall@0.85 100% -> 33.3%.

    WHY MOST TRANSITIONS ARE UNCLASSIFIED, AND WHY THAT IS A FEATURE
    ---------------------------------------------------------------
    The actor ships a sidecar at roughly 2 Hz and only while the arm is
    stationary, so at a 10 Hz control loop the large majority of transitions
    arrive with ``classifier_sidecar=None``.  That sparsity is intentional:

    * it lets the scene settle after the cube is released, instead of scoring
      motion-blurred frames mid-throw;
    * it removes flicker from the success verdict — a verdict that is only
      recomputed a few times per second cannot oscillate at 10 Hz; and
    * it happens to cost far less bandwidth, which is a side effect, not the
      reason.

    THE UNCLASSIFIED TRANSITION, FIELD BY FIELD
    -------------------------------------------
    ``rewards``      forced to 0.0 unless MANUAL carries an explicit one-shot
                     operator success.  No classifier result means AUTO has no
                     evidence of success; local proposed reward is discarded.
    ``masks``        the local proposal is preserved, which is what keeps the
                     mask/done consistency check in ``ReplayIngress._convert``
                     (masks == 0.0 iff dones) satisfied without special-casing.
    ``dones`` /      the local proposal passes through untouched.  This is not
    ``truncated``    new behaviour: it is what already happens on every
                     classifier-negative transition today.
    ``classifier_evaluated``  0
    ``classifier_probability`` / ``classifier_threshold``  0.0 / 0.0 — required
                     to be zero when unevaluated by ``_convert`` and by
                     ``grpc_actor_transport``; a "real but unused" probability
                     here would be rejected.
    ``classifier_success``    0
    ``reward_model_id``       "" — likewise required to be empty when
                     unevaluated (``actor_network._validate_finalized_``
                     ``transition``, ``grpc_actor_transport.outcome_from_proto``).

    Net effect on learning: an unclassified AUTO transition is an ordinary
    zero-reward sample.  In MANUAL, an operator success may still end it with a
    positive reward; classifier telemetry remains explicitly unevaluated.
    """

    def __init__(
        self,
        classifier: RewardClassifier,
        *,
        confirmations: int = DEFAULT_CLASSIFIER_CONFIRMATIONS,
        warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.classifier = classifier
        self.confirmations = _positive_int(confirmations, name="confirmations")
        self._warn = warn or _emit_operator_warning
        # Pre-filled with 0.0 so the window is ALWAYS full: while it is still
        # filling, min() == 0.0, which cannot exceed a threshold in [0, 1] and
        # therefore cannot declare success on fewer than `confirmations`
        # classifications.  With confirmations == 1 the deque holds exactly the
        # frame just classified, so the reported probability is the
        # instantaneous sigmoid, bit for bit what the live viewer shows.
        self._window: deque[float] = deque(
            [0.0] * self.confirmations, maxlen=self.confirmations
        )
        self._lock = threading.Lock()
        self._session_id = ""
        self._session_transitions = 0
        self._session_classifications = 0
        self._unclassified_streak = 0
        self._transition_count = 0
        self._classification_count = 0

    @property
    def transition_count(self) -> int:
        with self._lock:
            return self._transition_count

    @property
    def classification_count(self) -> int:
        with self._lock:
            return self._classification_count

    def __call__(
        self,
        data: dict[str, Any],
        classifier_sidecar: Optional[Mapping[str, Any]] = None,
    ) -> tuple[dict[str, Any], TransitionOutcome]:
        finalized = copy.deepcopy(data)
        if set(finalized) != {"meta", "transition"}:
            raise ActorProtocolError("data must contain exactly meta and transition")
        meta = finalized["meta"]
        transition = finalized["transition"]
        if not isinstance(meta, MutableMapping) or not isinstance(
            transition, MutableMapping
        ):
            raise ActorProtocolError("data.meta and data.transition must be mappings")
        transition_id = _required_text(
            meta.get("transition_id"), name="meta.transition_id"
        )
        auto_success = _binary_flag(
            meta.get("auto_success", False), name="meta.auto_success"
        )
        operator_success = _binary_flag(
            meta.get("operator_success", False),
            name="meta.operator_success",
        )
        if auto_success and operator_success:
            raise ActorProtocolError(
                "meta.operator_success is forbidden while auto_success is enabled"
            )
        if "next_observations" not in transition:
            # Still required, even though it is no longer what gets classified:
            # ActorSessionService attaches it and ReplayIngress needs it, so its
            # absence here means the pipeline is already broken upstream.
            raise ActorProtocolError(
                "transition.next_observations is required before reward finalization"
            )

        session_id = meta.get("session_id")
        session_id = session_id if isinstance(session_id, str) else ""
        self._begin_transition(session_id)

        if classifier_sidecar is None:
            probability = 0.0
            threshold = 0.0
            classifier_success = False
            evaluated = False
            reward_model_id = ""
        else:
            from ur_env.classifier_sidecar import decode_classifier_frames

            # Decoding here rather than in ActorSessionService keeps the JPEG
            # codec out of the transport contract and off the path of servers
            # that do not classify at all.
            try:
                frames = decode_classifier_frames(classifier_sidecar)
            except ValueError as exc:
                # The structure was already validated on ingress, so reaching
                # here means genuinely corrupt JPEG bytes.  Fail the transition
                # rather than silently downgrading it to "unclassified": a
                # camera producing undecodable frames must stop the run.
                raise RewardClassifierError(
                    f"classifier sidecar decode failed: {exc}"
                ) from exc
            result = self.classifier.classify(frames)
            probability, classifier_success = self._confirm(result)
            threshold = float(result.threshold)
            evaluated = True
            reward_model_id = _required_text(
                result.reward_model_id, name="classifier.reward_model_id"
            )

        # Reward remains server-authoritative.  The server accepts only the
        # protected one-shot operator assertion in MANUAL or its own classifier
        # verdict in AUTO; arbitrary locally proposed reward is discarded.
        effective_success = operator_success or (
            auto_success and classifier_success
        )
        transition["rewards"] = 1.0 if effective_success else 0.0
        if effective_success:
            # An effective success wins if the local time limit happened on
            # this same observation, matching one-positive HIL-SERL behavior.
            transition["masks"] = 0.0
            transition["dones"] = True
            transition["truncated"] = False

        reward = _finite_float(transition.get("rewards"), name="transition.rewards")
        mask = _finite_float(transition.get("masks"), name="transition.masks")
        done = bool(transition.get("dones"))
        truncated = bool(transition.get("truncated"))

        # These server-only fields are not serialized back as transition input;
        # they are retained by ReplayIngress and mirrored in TransitionOutcome.
        transition["classifier_evaluated"] = np.uint8(evaluated)
        transition["classifier_probability"] = float(probability)
        transition["classifier_threshold"] = float(threshold)
        transition["classifier_success"] = np.uint8(classifier_success)
        transition["success"] = np.uint8(effective_success)
        transition["reward_model_id"] = reward_model_id

        self._end_transition(
            session_id, evaluated=evaluated, terminal=done or truncated
        )

        return finalized, TransitionOutcome(
            transition_id=transition_id,
            reward=reward,
            mask=mask,
            done=done,
            truncated=truncated,
            success=effective_success,
            classifier_evaluated=evaluated,
            classifier_probability=probability,
            classifier_threshold=threshold,
            reward_model_id=reward_model_id,
        )

    def _confirm(self, result: ClassificationResult) -> tuple[float, bool]:
        """Fold one instantaneous verdict into the confirmation window.

        Returns ``(reported_probability, success)``.

        THE INVARIANT THAT MUST SURVIVE ANY SMOOTHING SCHEME
        ----------------------------------------------------
        ``classifier_success == (classifier_probability > classifier_threshold)``
        is re-derived and enforced independently in THREE places —
        ``actor_network._validate_finalized_transition``,
        ``grpc_actor_transport.outcome_from_proto`` and
        ``ReplayIngress._convert``.  They exist so the number a human reads in a
        log can always be checked against the verdict that was actually trained
        on; a smoothing scheme that reports one number and decides on another
        would be rejected by all three, and rightly so.

        So smoothing is expressed as a CHOICE OF WHICH REAL PROBABILITY TO
        REPORT, never as a separate decision rule: the reported value is the
        weakest classification in the window of the last ``confirmations``
        samples.  ``min(window) > threshold`` holds exactly when every one of
        those samples cleared the threshold, i.e. N-of-N consecutive
        confirmations, and the invariant is true by construction.

        With ``confirmations == 1`` (the default, and what the first real run
        uses) the window is that single sample: the reported probability IS the
        instantaneous sigmoid, so the server and the live viewer print the same
        number for the same frames.  ONLY when ``confirmations > 1`` does
        ``classifier_probability`` stop being instantaneous — it becomes the
        window floor.  Note the window is NOT reset by the gaps between sparse
        classifications, so at ~2 Hz sampling three confirmations span roughly
        1.5 seconds of real time; it IS reset at every episode boundary.
        """
        instantaneous = float(result.probability)
        threshold = float(result.threshold)
        with self._lock:
            self._window.append(instantaneous)
            reported = min(self._window)
        if self.confirmations == 1 and reported != instantaneous:
            raise RewardClassifierError(
                "unsmoothed confirmation window did not report the "
                "instantaneous probability"
            )
        return reported, reported > threshold

    def _begin_transition(self, session_id: str) -> None:
        warnings: list[str] = []
        with self._lock:
            if session_id != self._session_id:
                # Covers the session that ended without a terminal transition
                # (actor crash, operator abort) — it still gets its verdict.
                warnings.extend(self._flush_session_locked())
                self._session_id = session_id
                # A new session is a new episode: confirmations must never
                # carry a verdict across an episode boundary.
                self._reset_window_locked()
            self._transition_count += 1
            self._session_transitions += 1
        for message in warnings:
            self._warn(message)

    def _end_transition(
        self, session_id: str, *, evaluated: bool, terminal: bool
    ) -> None:
        warnings: list[str] = []
        with self._lock:
            if evaluated:
                self._classification_count += 1
                self._session_classifications += 1
                self._unclassified_streak = 0
            else:
                self._unclassified_streak += 1
                if self._unclassified_streak % UNCLASSIFIED_WARN_STREAK == 0:
                    warnings.append(
                        f"{self._unclassified_streak} consecutive transitions "
                        "carried no classifier sidecar; no reward can be "
                        "produced while this continues (is the actor's sidecar "
                        "scheduler running, and is the arm ever stationary?)"
                    )
            if terminal:
                # Report at the episode boundary rather than only when the next
                # session shows up, so an operator sees it while the episode is
                # still on screen.
                warnings.extend(self._flush_session_locked())
                self._reset_window_locked()
        for message in warnings:
            self._warn(message)

    def _flush_session_locked(self) -> list[str]:
        """Close the accounting for the current session; caller emits, unlocked."""
        messages: list[str] = []
        if self._session_transitions > 0 and self._session_classifications == 0:
            messages.append(
                f"session {self._session_id!r} finalized "
                f"{self._session_transitions} transitions and classified NONE "
                "of them; every reward in it is 0 by default, not by verdict"
            )
        self._session_transitions = 0
        self._session_classifications = 0
        return messages

    def _reset_window_locked(self) -> None:
        self._window.clear()
        self._window.extend([0.0] * self.confirmations)


class _ReplayStore(Protocol):
    dataset_dict: MutableMapping[str, Any]
    _capacity: int
    _first: bool
    _insert_index: int
    _is_correct_index: np.ndarray

    def __len__(self) -> int:
        ...

    def insert(self, transition: MutableMapping[str, Any]) -> None:
        ...

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        ...


StoreFactory = Callable[..., _ReplayStore]


@dataclass(frozen=True)
class IngressRecord:
    """Bounded summary sidecar; it intentionally contains no tensor values."""

    transition_id: str
    run_id: str
    actor_id: str
    session_id: str
    observation_id: str
    next_observation_id: str
    reward_model_id: str
    auto_success: bool
    operator_success: bool
    success: bool
    env_step: int
    episode_id: int
    step_id: int
    intervened: bool
    terminal_for_stack: bool


@dataclass
class _RouteState:
    signature: bytes
    replay_inserted: bool = False
    intervention_inserted: bool = False
    completed: bool = False


def _make_spaces() -> tuple[Any, Any]:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ReceiveRuntimeError("gymnasium is required by replay ingress") from exc

    observation_spaces = {}
    for key, (dtype, shape) in CANONICAL_OBSERVATION_SPEC.items():
        if dtype == np.dtype(np.uint8):
            observation_spaces[key] = gym.spaces.Box(
                low=0, high=255, shape=shape, dtype=dtype
            )
        else:
            observation_spaces[key] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=dtype
            )
    observation_space = gym.spaces.Dict(observation_spaces)
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=ACTION_SHAPE, dtype=np.float32
    )
    return observation_space, action_space


def _upstream_store_factory(hil_serl_root: Optional[str]) -> StoreFactory:
    if hil_serl_root:
        launcher_root = os.path.join(
            os.path.abspath(os.path.expanduser(hil_serl_root)), "serl_launcher"
        )
        if launcher_root not in sys.path:
            sys.path.insert(0, launcher_root)
    try:
        from serl_launcher.data.data_store import (
            MemoryEfficientReplayBufferDataStore,
        )
    except ImportError as exc:
        raise ReceiveRuntimeError(
            "the pinned HIL-SERL serl_launcher package and its JAX/Flax/"
            "Agentlace dependencies are required by replay ingress"
        ) from exc

    def create(
        *,
        observation_space: Any,
        action_space: Any,
        capacity: int,
        image_keys: tuple[str, ...],
    ) -> _ReplayStore:
        return MemoryEfficientReplayBufferDataStore(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            image_keys=image_keys,
        )

    return create


class ReplayIngress:
    """Thread-safe, idempotent routing into two upstream RAM buffers.

    ``dones`` passed to the memory-efficient store is the image-stack boundary
    (``done or truncated``); ``terminated`` retains the true environment/RL
    terminal flag while ``masks`` keeps Bellman semantics.  The intervention
    store additionally forces ``_first`` whenever selected intervention steps
    are not consecutive, preventing images across a policy-only gap from being
    stacked together.
    """

    _NUMERIC_METADATA = {
        "policy_actions": (ACTION_SHAPE, np.float32),
        "intervened": ((), np.uint8),
        "timestamp_ns": ((), np.int64),
        "policy_version": ((), np.int64),
        "env_step": ((), np.int64),
        "episode_id": ((), np.int64),
        "step_id": ((), np.int64),
        "terminated": ((), np.bool_),
        "truncated": ((), np.bool_),
        "classifier_evaluated": ((), np.uint8),
        "classifier_probability": ((), np.float32),
        "classifier_threshold": ((), np.float32),
        "classifier_success": ((), np.uint8),
        "auto_success": ((), np.uint8),
        "operator_success": ((), np.uint8),
        "success": ((), np.uint8),
        "has_grasp_penalty": ((), np.uint8),
        "grasp_penalty": ((), np.float32),
    }

    def __init__(
        self,
        *,
        replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
        intervention_capacity: int = DEFAULT_INTERVENTION_CAPACITY,
        hil_serl_root: Optional[str] = None,
        store_factory: Optional[StoreFactory] = None,
        ledger_capacity: Optional[int] = None,
        learner_mode: bool = False,
        require_grasp_penalty: Optional[bool] = None,
        expected_grasp_penalty: float = -0.02,
    ) -> None:
        if not isinstance(learner_mode, bool):
            raise ValueError("learner_mode must be bool")
        if require_grasp_penalty is not None and not isinstance(
            require_grasp_penalty, bool
        ):
            raise ValueError("require_grasp_penalty must be bool when provided")
        if learner_mode and require_grasp_penalty is False:
            raise ValueError(
                "learner_mode cannot disable the grasp_penalty contract"
            )
        self.require_grasp_penalty = (
            learner_mode
            if require_grasp_penalty is None
            else require_grasp_penalty
        )
        if isinstance(expected_grasp_penalty, (bool, np.bool_)):
            raise ValueError("expected_grasp_penalty must be numeric")
        expected_penalty_array = np.asarray(expected_grasp_penalty)
        if (
            expected_penalty_array.shape != ()
            or expected_penalty_array.dtype.kind not in "iuf"
        ):
            raise ValueError("expected_grasp_penalty must be numeric")
        self.expected_grasp_penalty = float(expected_penalty_array)
        if (
            not math.isfinite(self.expected_grasp_penalty)
            or self.expected_grasp_penalty > 0.0
        ):
            raise ValueError(
                "expected_grasp_penalty must be finite and non-positive"
            )
        self.replay_capacity = _positive_int(
            replay_capacity, name="replay_capacity"
        )
        self.intervention_capacity = _positive_int(
            intervention_capacity, name="intervention_capacity"
        )
        self._ledger_capacity = _positive_int(
            ledger_capacity or max(self.replay_capacity, self.intervention_capacity),
            name="ledger_capacity",
        )
        image_stack_sizes = {
            CANONICAL_OBSERVATION_SPEC[key][1][0] for key in IMAGE_KEYS
        }
        if len(image_stack_sizes) != 1:
            raise ReceiveRuntimeError(
                "canonical camera tensors must use one common stack size"
            )
        self._num_stack = image_stack_sizes.pop()
        # Upstream stores one bootstrap frame for every new image sequence in
        # addition to the logical transition.  Allocate for the worst case
        # (every transition starts a sequence), then explicitly retain only
        # the latest logical-capacity valid indices below.
        self._replay_physical_capacity = self.replay_capacity * (
            self._num_stack + 1
        )
        self._intervention_physical_capacity = self.intervention_capacity * (
            self._num_stack + 1
        )
        observation_space, action_space = _make_spaces()
        factory = store_factory or _upstream_store_factory(hil_serl_root)
        self.replay_store = factory(
            observation_space=observation_space,
            action_space=action_space,
            capacity=self._replay_physical_capacity,
            image_keys=IMAGE_KEYS,
        )
        self.intervention_store = factory(
            observation_space=observation_space,
            action_space=action_space,
            capacity=self._intervention_physical_capacity,
            image_keys=IMAGE_KEYS,
        )
        self._validate_store_layout(
            self.replay_store, self._replay_physical_capacity
        )
        self._validate_store_layout(
            self.intervention_store, self._intervention_physical_capacity
        )
        self._extend_numeric_schema(
            self.replay_store, self._replay_physical_capacity
        )
        self._extend_numeric_schema(
            self.intervention_store, self._intervention_physical_capacity
        )

        self._lock = threading.RLock()
        self._ledger: OrderedDict[str, _RouteState] = OrderedDict()
        self._replay_sidecar: deque[IngressRecord] = deque(
            maxlen=self.replay_capacity
        )
        self._intervention_sidecar: deque[IngressRecord] = deque(
            maxlen=self.intervention_capacity
        )
        self._replay_insert_count = 0
        self._intervention_insert_count = 0
        self._replay_overwrite_count = 0
        self._intervention_overwrite_count = 0
        self._replay_valid_indices: deque[Optional[int]] = deque()
        self._intervention_valid_indices: deque[Optional[int]] = deque()
        self._last_replay_sequence: Optional[IngressRecord] = None
        self._last_intervention_sequence: Optional[IngressRecord] = None
        self._last_transition_id = ""
        self._last_env_step: Optional[int] = None

    def __call__(self, data: dict[str, Any], intervened: bool) -> None:
        record, transition = self.validate_data(
            data,
            intervened=intervened,
            require_grasp_penalty=self.require_grasp_penalty,
            expected_grasp_penalty=self.expected_grasp_penalty,
        )
        signature = self.fingerprint(record, transition)
        with self._lock:
            route = self._ledger.get(record.transition_id)
            if route is None:
                route = _RouteState(signature=signature)
                self._ledger[record.transition_id] = route
            elif route.signature != signature:
                raise ActorProtocolError(
                    f"transition_id collision for {record.transition_id!r}"
                )
            elif route.completed:
                self._ledger.move_to_end(record.transition_id)
                return

            if not route.replay_inserted:
                replay_boundary = self._starts_new_sequence(
                    self._last_replay_sequence, record
                )
                replay_overwrote = self._insert_route(
                    self.replay_store,
                    transition,
                    force_boundary=replay_boundary,
                    valid_indices=self._replay_valid_indices,
                    logical_capacity=self.replay_capacity,
                )
                route.replay_inserted = True
                self._replay_insert_count += 1
                if replay_overwrote:
                    self._replay_overwrite_count += 1
                self._replay_sidecar.append(record)
                self._last_replay_sequence = record

            if intervened and not route.intervention_inserted:
                intervention_boundary = self._starts_new_sequence(
                    self._last_intervention_sequence, record
                )
                intervention_overwrote = self._insert_route(
                    self.intervention_store,
                    transition,
                    force_boundary=intervention_boundary,
                    valid_indices=self._intervention_valid_indices,
                    logical_capacity=self.intervention_capacity,
                )
                route.intervention_inserted = True
                self._intervention_insert_count += 1
                if intervention_overwrote:
                    self._intervention_overwrite_count += 1
                self._intervention_sidecar.append(record)
                self._last_intervention_sequence = record

            route.completed = route.replay_inserted and (
                route.intervention_inserted or not intervened
            )
            if not route.completed:
                raise ReceiveRuntimeError("transition routing did not complete")
            self._last_transition_id = record.transition_id
            self._last_env_step = record.env_step
            self._ledger.move_to_end(record.transition_id)
            self._trim_ledger()

    def status(self) -> BufferStatus:
        """Return summary-only counters suitable for ``GetBufferStatus``."""
        with self._lock:
            return BufferStatus(
                replay_size=len(self._replay_valid_indices),
                replay_capacity=self.replay_capacity,
                intervention_size=len(self._intervention_valid_indices),
                intervention_capacity=self.intervention_capacity,
                replay_insert_count=self._replay_insert_count,
                intervention_insert_count=self._intervention_insert_count,
                replay_overwrite_count=self._replay_overwrite_count,
                intervention_overwrite_count=self._intervention_overwrite_count,
                last_transition_id=self._last_transition_id,
                last_env_step=self._last_env_step,
            )

    def sample_replay(self, batch_size: int, **kwargs: Any) -> Any:
        """Sample a learner-ready replay batch with packed image pairs."""
        return self._sample(
            self.replay_store, batch_size=batch_size, **kwargs
        )

    def sample_intervention(self, batch_size: int, **kwargs: Any) -> Any:
        """Sample a learner-ready intervention batch with packed image pairs."""
        return self._sample(
            self.intervention_store, batch_size=batch_size, **kwargs
        )

    def replay_sidecar(self) -> tuple[IngressRecord, ...]:
        with self._lock:
            return tuple(self._replay_sidecar)

    def intervention_sidecar(self) -> tuple[IngressRecord, ...]:
        with self._lock:
            return tuple(self._intervention_sidecar)

    @classmethod
    def validate_data(
        cls,
        data: Mapping[str, Any],
        *,
        intervened: bool,
        require_grasp_penalty: bool = False,
        expected_grasp_penalty: float = -0.02,
    ) -> tuple[IngressRecord, dict[str, Any]]:
        """Validate canonical ingress data without retaining the raw input.

        This public validation boundary is shared by alternate infra-owned
        replay backends.  In particular, the feature-native replay backend
        must enforce exactly the same actor, classifier, and gripper contracts
        before replacing camera images with frozen-trunk feature maps.
        """

        return cls._convert(
            data,
            intervened=intervened,
            require_grasp_penalty=require_grasp_penalty,
            expected_grasp_penalty=expected_grasp_penalty,
        )

    @classmethod
    def fingerprint(
        cls, record: IngressRecord, transition: Mapping[str, Any]
    ) -> bytes:
        """Return the collision fingerprint used by the idempotency ledger."""

        return cls._signature(record, transition)

    def _sample(self, store: _ReplayStore, *, batch_size: int, **kwargs: Any) -> Any:
        size = _positive_int(batch_size, name="batch_size")
        if "pack_obs_and_next_obs" in kwargs:
            if kwargs["pack_obs_and_next_obs"] is not True:
                raise ValueError("pack_obs_and_next_obs must remain true")
        else:
            kwargs["pack_obs_and_next_obs"] = True
        with self._lock:
            return store.sample(batch_size=size, **kwargs)

    @classmethod
    def _extend_numeric_schema(cls, store: _ReplayStore, capacity: int) -> None:
        if not hasattr(store, "dataset_dict") or not isinstance(
            store.dataset_dict, MutableMapping
        ):
            raise ReceiveRuntimeError("replay store must expose mutable dataset_dict")
        for key, (shape, dtype) in cls._NUMERIC_METADATA.items():
            if key in store.dataset_dict:
                raise ReceiveRuntimeError(
                    f"upstream replay schema unexpectedly already contains {key!r}"
                )
            store.dataset_dict[key] = np.empty((capacity, *shape), dtype=dtype)

    def _validate_store_layout(
        self, store: _ReplayStore, expected_physical_capacity: int
    ) -> None:
        if hasattr(store, "_capacity") and int(store._capacity) != int(
            expected_physical_capacity
        ):
            raise ReceiveRuntimeError(
                "replay store physical capacity does not match ingress layout"
            )
        if hasattr(store, "_num_stack") and int(store._num_stack) != int(
            self._num_stack
        ):
            raise ReceiveRuntimeError(
                "replay store image stack size does not match canonical schema"
            )

    @classmethod
    def _insert_route(
        cls,
        store: _ReplayStore,
        transition: MutableMapping[str, Any],
        *,
        force_boundary: bool,
        valid_indices: deque[Optional[int]],
        logical_capacity: int,
    ) -> bool:
        tracks_indices = cls._tracks_upstream_valid_indices(store)
        evicted: Optional[int] = None
        overwrote = len(valid_indices) >= logical_capacity
        if overwrote:
            evicted = valid_indices.popleft()
            if tracks_indices:
                if evicted is None:
                    raise ReceiveRuntimeError(
                        "upstream replay index tracker lost a physical index"
                    )
                store._is_correct_index[evicted] = False
        if force_boundary:
            if not hasattr(store, "_first"):
                raise ReceiveRuntimeError(
                    "memory-efficient replay store does not expose _first boundary"
                )
            store._first = True
        try:
            store.insert(copy.deepcopy(transition))
        except Exception:
            if overwrote:
                valid_indices.appendleft(evicted)
                if tracks_indices and evicted is not None:
                    store._is_correct_index[evicted] = True
            raise

        if not tracks_indices:
            valid_indices.append(None)
            return overwrote

        physical_capacity = int(store._capacity)
        new_index = (int(store._insert_index) - 1) % physical_capacity
        if not bool(store._is_correct_index[new_index]):
            raise ReceiveRuntimeError(
                "upstream replay did not mark the inserted transition sampleable"
            )
        if new_index in valid_indices:
            raise ReceiveRuntimeError(
                "upstream replay overwrote a transition inside logical capacity"
            )
        invalid_retained = [
            index
            for index in valid_indices
            if index is None or not bool(store._is_correct_index[index])
        ]
        if invalid_retained:
            raise ReceiveRuntimeError(
                "upstream replay invalidated a transition inside logical capacity"
            )
        valid_indices.append(new_index)
        return overwrote

    @staticmethod
    def _tracks_upstream_valid_indices(store: _ReplayStore) -> bool:
        return (
            hasattr(store, "_insert_index")
            and hasattr(store, "_is_correct_index")
            and isinstance(store._is_correct_index, np.ndarray)
            and hasattr(store, "_capacity")
        )

    @staticmethod
    def _starts_new_sequence(
        previous: Optional[IngressRecord], current: IngressRecord
    ) -> bool:
        if previous is None or previous.terminal_for_stack:
            return True
        return not (
            previous.actor_id == current.actor_id
            and previous.session_id == current.session_id
            and previous.episode_id == current.episode_id
            and current.step_id == previous.step_id + 1
        )

    @staticmethod
    def _signature(
        record: IngressRecord, transition: Mapping[str, Any]
    ) -> bytes:
        # ActorSessionService already fingerprints requests.  ReplayIngress is
        # also safe as a standalone sink: the same ID may only repeat with the
        # exact same strings, numeric metadata, actions, and observation bytes.
        digest = hashlib.sha256()
        ReplayIngress._hash_signature_value(
            digest,
            {
                "record": {
                    "transition_id": record.transition_id,
                    "run_id": record.run_id,
                    "actor_id": record.actor_id,
                    "session_id": record.session_id,
                    "observation_id": record.observation_id,
                    "next_observation_id": record.next_observation_id,
                    "reward_model_id": record.reward_model_id,
                    "auto_success": record.auto_success,
                    "operator_success": record.operator_success,
                    "success": record.success,
                    "env_step": record.env_step,
                    "episode_id": record.episode_id,
                    "step_id": record.step_id,
                    "intervened": record.intervened,
                    "terminal_for_stack": record.terminal_for_stack,
                },
                "transition": transition,
            },
        )
        return digest.digest()

    @staticmethod
    def _hash_signature_value(digest: Any, value: Any) -> None:
        if isinstance(value, Mapping):
            digest.update(b"{")
            for key in sorted(value):
                if not isinstance(key, str):
                    raise ActorProtocolError(
                        "signature mapping keys must be strings"
                    )
                digest.update(key.encode("utf-8"))
                digest.update(b"\0")
                ReplayIngress._hash_signature_value(digest, value[key])
            digest.update(b"}")
            return
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b"array\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
            return
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool, type(None))):
            digest.update(type(value).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(value).encode("utf-8"))
            return
        raise ActorProtocolError(
            f"cannot fingerprint replay value of type {type(value).__name__}"
        )

    def _trim_ledger(self) -> None:
        while len(self._ledger) > self._ledger_capacity:
            removable = next(
                (
                    key
                    for key, state in self._ledger.items()
                    if state.completed
                    and key != self._last_transition_id
                ),
                None,
            )
            if removable is None:
                raise ReceiveRuntimeError(
                    "idempotency ledger is full of incomplete routes"
                )
            self._ledger.pop(removable)

    @staticmethod
    def _convert(
        data: Mapping[str, Any],
        *,
        intervened: bool,
        require_grasp_penalty: bool = False,
        expected_grasp_penalty: float = -0.02,
    ) -> tuple[IngressRecord, dict[str, Any]]:
        if not isinstance(data, Mapping) or set(data) != {"meta", "transition"}:
            raise ActorProtocolError("data must contain exactly meta and transition")
        meta = data["meta"]
        source = data["transition"]
        if not isinstance(meta, Mapping) or not isinstance(source, Mapping):
            raise ActorProtocolError("data.meta and data.transition must be mappings")
        labelled_intervention = meta.get("intervened")
        if isinstance(labelled_intervention, np.generic):
            labelled_intervention = labelled_intervention.item()
        if labelled_intervention not in (False, True, 0, 1):
            raise ActorProtocolError("meta.intervened must be 0 or 1")
        if bool(labelled_intervention) != bool(intervened):
            raise ActorProtocolError(
                "intervention route does not match meta.intervened"
            )
        auto_success = _binary_flag(
            meta.get("auto_success", False), name="meta.auto_success"
        )
        operator_success = _binary_flag(
            meta.get("operator_success", False),
            name="meta.operator_success",
        )
        if auto_success and operator_success:
            raise ActorProtocolError(
                "meta.operator_success is forbidden while auto_success is enabled"
            )

        observation = validate_canonical_observation(
            source.get("observations"), copy=True
        )
        next_observation = validate_canonical_observation(
            source.get("next_observations"), copy=True
        )
        actions = FakeActionRuntime._validate_action(
            source.get("actions"), name="transition.actions"
        )
        policy_actions = FakeActionRuntime._validate_action(
            meta.get("policy_action"), name="meta.policy_action"
        )
        reward = _finite_float(source.get("rewards"), name="transition.rewards")
        mask = _finite_float(source.get("masks"), name="transition.masks")
        done_value = source.get("dones")
        truncated_value = source.get("truncated")
        if not isinstance(done_value, (bool, np.bool_)) or not isinstance(
            truncated_value, (bool, np.bool_)
        ):
            raise ActorProtocolError("transition dones/truncated must be bool")
        done = bool(done_value)
        truncated = bool(truncated_value)
        if done and truncated:
            raise ActorProtocolError("transition cannot be done and truncated")
        expected_mask = 0.0 if done else 1.0
        if mask != expected_mask:
            raise ActorProtocolError(
                f"transition.masks must be {expected_mask} for dones={done}"
            )

        classifier_evaluated = bool(source.get("classifier_evaluated", False))
        probability = _finite_float(
            source.get("classifier_probability", 0.0),
            name="transition.classifier_probability",
        )
        threshold = _finite_float(
            source.get("classifier_threshold", 0.0),
            name="transition.classifier_threshold",
        )
        classifier_success = bool(source.get("classifier_success", False))
        success = _binary_flag(
            source.get("success"), name="transition.success"
        )
        reward_model_id = source.get("reward_model_id", "")
        if not isinstance(reward_model_id, str):
            raise ActorProtocolError("transition.reward_model_id must be a string")
        # Sparse classification needed NO change here, and that is deliberate.
        # The strict `success == (probability > threshold)` cross-check already
        # sits INSIDE the evaluated branch, and the `elif` already rejects any
        # non-zero classifier field on an unevaluated transition.  So an
        # unclassified transition is representable without weakening either
        # guard: it must be exactly (0, 0.0, 0.0, false, "").  Relaxing the
        # cross-check to admit unevaluated rows would have opened a hole for
        # evaluated ones too.
        if classifier_evaluated:
            if not 0.0 <= probability <= 1.0 or not 0.0 <= threshold <= 1.0:
                raise ActorProtocolError(
                    "classifier probability/threshold must be within [0, 1]"
                )
            if not reward_model_id:
                raise ActorProtocolError(
                    "reward_model_id is required for evaluated transitions"
                )
            if classifier_success != (probability > threshold):
                raise ActorProtocolError(
                    "classifier_success must use strict probability > threshold"
                )
        elif classifier_success or probability != 0.0 or threshold != 0.0:
            raise ActorProtocolError(
                "unevaluated transition cannot carry classifier results"
            )

        effective_success = operator_success or (
            auto_success and classifier_success
        )
        if success != effective_success:
            raise ActorProtocolError(
                "transition.success does not match operator_success OR "
                "(auto_success AND classifier_success)"
            )
        expected_reward = 1.0 if success else 0.0
        if reward != expected_reward:
            raise ActorProtocolError(
                f"transition.rewards must be {expected_reward} for success={success}"
            )
        if success and (not done or truncated or mask != 0.0):
            raise ActorProtocolError(
                "successful transition must be done, not truncated, with mask 0"
            )

        has_grasp_penalty = "grasp_penalty" in source
        if require_grasp_penalty and not has_grasp_penalty:
            raise ActorProtocolError(
                "transition.grasp_penalty is required in learner mode"
            )
        grasp_penalty = _finite_float(
            source.get("grasp_penalty", 0.0), name="transition.grasp_penalty"
        )
        if require_grasp_penalty and not (
            math.isclose(grasp_penalty, 0.0, rel_tol=0.0, abs_tol=1e-7)
            or math.isclose(
                grasp_penalty,
                expected_grasp_penalty,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
        ):
            raise ActorProtocolError(
                "transition.grasp_penalty must be either 0 or the configured "
                f"penalty {expected_grasp_penalty}"
            )
        terminal_for_stack = done or truncated
        episode_id = _nonnegative_int(
            source.get("episode_id"), name="transition.episode_id"
        )
        step_id = _nonnegative_int(
            source.get("step_id"), name="transition.step_id"
        )
        env_step = _nonnegative_int(meta.get("env_step"), name="meta.env_step")

        record = IngressRecord(
            transition_id=_required_text(
                meta.get("transition_id"), name="meta.transition_id"
            ),
            run_id=_required_text(meta.get("run_id"), name="meta.run_id"),
            actor_id=_required_text(meta.get("actor_id"), name="meta.actor_id"),
            session_id=_required_text(
                meta.get("session_id"), name="meta.session_id"
            ),
            observation_id=_required_text(
                source.get("observation_id"), name="transition.observation_id"
            ),
            next_observation_id=_required_text(
                source.get("next_observation_id"),
                name="transition.next_observation_id",
            ),
            reward_model_id=reward_model_id,
            auto_success=auto_success,
            operator_success=operator_success,
            success=success,
            env_step=env_step,
            episode_id=episode_id,
            step_id=step_id,
            intervened=bool(intervened),
            terminal_for_stack=terminal_for_stack,
        )
        transition = {
            "observations": observation,
            "actions": actions,
            "policy_actions": policy_actions,
            "intervened": np.uint8(intervened),
            "next_observations": next_observation,
            "rewards": np.float32(reward),
            "masks": np.float32(mask),
            # Upstream MemoryEfficientReplayBuffer uses `dones` solely to set
            # its next image-stack boundary.  Include time limits here.
            "dones": np.bool_(terminal_for_stack),
            "terminated": np.bool_(done),
            "truncated": np.bool_(truncated),
            "timestamp_ns": np.int64(
                _nonnegative_int(meta.get("timestamp_ns"), name="meta.timestamp_ns")
            ),
            "policy_version": np.int64(
                _nonnegative_int(
                    meta.get("policy_version"), name="meta.policy_version"
                )
            ),
            "env_step": np.int64(env_step),
            "episode_id": np.int64(episode_id),
            "step_id": np.int64(step_id),
            "classifier_evaluated": np.uint8(classifier_evaluated),
            "classifier_probability": np.float32(probability),
            "classifier_threshold": np.float32(threshold),
            "classifier_success": np.uint8(classifier_success),
            "auto_success": np.uint8(auto_success),
            "operator_success": np.uint8(operator_success),
            "success": np.uint8(success),
            "has_grasp_penalty": np.uint8(has_grasp_penalty),
            "grasp_penalty": np.float32(grasp_penalty),
        }
        return record, transition
