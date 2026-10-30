"""The degraded-reward-classifier signal, from the actor to the operator GUI.

The learner now says "the classifier is broken" ONCE (see
``test_rlpd_receive_server``); this is the other half of that decision -- the
CONTINUOUS indication that replaces the terminal cadence.  It travels on the
existing ``/hil/actor_status`` JSON as two optional fields, never on the
actor<->learner gRPC contract, so nothing here can put a half-upgraded pair
into silent disagreement.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import pytest  # noqa: E402


def _gui_status_module():
    """Load the GUI-side parser from SOURCE, under its own module name.

    This suite's canonical command puts the BUILT ``ros2_ur_ws/install``
    overlay on ``PYTHONPATH``, and that copy (it is a copy, not a symlink) lags
    this tree until the next ``colcon build``; another test importing
    ``ur_gello_bringup`` first would also pin the stale one in ``sys.modules``.
    Neither is what this test is asking about -- it is asking whether the
    payload the actor publishes is the payload the GUI code in this commit
    reads -- so load the file directly and leave ``sys.modules`` alone.
    """

    path = os.path.join(
        _HERE,
        "..",
        "..",
        "ros2_ur_ws",
        "src",
        "ur_gello_bringup",
        "ur_gello_bringup",
        "hil_actor_status.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_hil_actor_status_source", os.path.abspath(path)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

from ur_env.actor_network import CLASSIFIER_DEGRADED_MARKER  # noqa: E402
from ur_env.operator_session import ActorStatus, ActorStatusTracker  # noqa: E402
from ur_env.remote_actor import (  # noqa: E402
    CLASSIFIER_DEGRADED_DETAIL_LIMIT,
    _ClassifierDegradedProbe,
    _OperatorReporter,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class _Network:
    """A network whose Health answer and call count the test controls."""

    def __init__(self, detail: str = "ready", *, raises: bool = False) -> None:
        self.detail = detail
        self.raises = raises
        self.health_calls = 0

    def health(self) -> tuple[bool, bool, str]:
        self.health_calls += 1
        if self.raises:
            raise RuntimeError("channel is down")
        return True, True, self.detail


_DEGRADED_HEALTH = (
    f"ready; {CLASSIFIER_DEGRADED_MARKER}: 12 faulted classification(s), "
    "3 succeeded; transitions are being recorded unclassified with reward 0 "
    "(last fault: RewardClassifierError: inference failed)"
)


def test_unattached_steps_are_never_evidence_of_anything():
    """Most steps carry no sidecar by design; they must stay silent."""
    network = _Network()
    probe = _ClassifierDegradedProbe(network, clock=_Clock())

    for _ in range(50):
        assert probe.observe(attached=False, evaluated=False) is False

    assert probe.degraded is False
    assert probe.detail == ""
    # A healthy run pays exactly zero extra RPCs.
    assert network.health_calls == 0


def test_a_sidecar_that_comes_back_unscored_is_the_degraded_evidence():
    network = _Network(_DEGRADED_HEALTH)
    probe = _ClassifierDegradedProbe(network, clock=_Clock())

    assert probe.observe(attached=True, evaluated=False) is True
    assert probe.degraded is True
    # The server already phrased it, counters and exception included.
    assert probe.detail == _DEGRADED_HEALTH
    assert network.health_calls == 1


def test_health_is_enrichment_only_never_the_verdict():
    """A dead Health RPC must not hide a degraded reward path."""
    network = _Network(raises=True)
    probe = _ClassifierDegradedProbe(network, clock=_Clock())

    assert probe.observe(attached=True, evaluated=False) is True
    assert probe.degraded is True
    assert "no verdict" in probe.detail
    assert "health unavailable" in probe.detail
    assert "RuntimeError" in probe.detail


def test_a_marker_less_health_still_reports_the_local_evidence():
    """A server with no classifier at all is a degraded reward path too."""
    probe = _ClassifierDegradedProbe(_Network("ready"), clock=_Clock())

    probe.observe(attached=True, evaluated=False)

    assert probe.degraded is True
    assert "no verdict" in probe.detail
    assert "server health: ready" in probe.detail


def test_health_is_polled_on_an_interval_not_per_step():
    clock = _Clock()
    network = _Network(_DEGRADED_HEALTH)
    probe = _ClassifierDegradedProbe(network, interval_s=10.0, clock=clock)

    probe.observe(attached=True, evaluated=False)
    assert network.health_calls == 1
    for _ in range(20):
        clock.now += 0.4
        assert probe.observe(attached=True, evaluated=False) is False
    assert network.health_calls == 1

    clock.now += 10.0
    probe.observe(attached=True, evaluated=False)
    assert network.health_calls == 2


def test_a_scored_step_clears_the_degraded_state():
    clock = _Clock()
    network = _Network(_DEGRADED_HEALTH)
    probe = _ClassifierDegradedProbe(network, clock=clock)

    probe.observe(attached=True, evaluated=False)
    assert probe.observe(attached=True, evaluated=True) is True
    assert probe.degraded is False
    assert probe.detail == ""

    # And it re-arms: a later fault polls immediately rather than waiting out
    # the interval left over from the previous episode.
    calls = network.health_calls
    assert probe.observe(attached=True, evaluated=False) is True
    assert network.health_calls == calls + 1


def test_a_very_long_server_detail_cannot_push_the_gui_panel_around():
    probe = _ClassifierDegradedProbe(_Network("x" * 4000), clock=_Clock())

    probe.observe(attached=True, evaluated=False)

    assert len(probe.detail) <= CLASSIFIER_DEGRADED_DETAIL_LIMIT
    assert probe.detail.endswith("...")


def test_the_reporter_latches_the_condition_across_every_state():
    """HOMING and the operator gates publish it too, or the GUI would blink."""
    reporter = _OperatorReporter(session=None, run_id="run-0")
    reporter.set_classifier_degraded(True, "inference failed")

    for state in ("POLICY_RUNNING", "HOMING", "WAIT_HOME_APPROVAL"):
        status = reporter.status(state, "NONE", message="")
        assert status.classifier_degraded is True
        assert status.classifier_degraded_detail == "inference failed"

    reporter.set_classifier_degraded(False)
    status = reporter.status("POLICY_RUNNING", "POLICY", message="")
    assert status.classifier_degraded is False
    assert status.classifier_degraded_detail == ""


def test_status_defaults_keep_the_field_optional_and_typed():
    tracker = ActorStatusTracker("run-0")

    status = tracker.status(
        state="POLICY_RUNNING",
        control_owner="POLICY",
        episode_id=0,
        episode_step=0,
        env_step=0,
    )
    assert status.classifier_degraded is False
    assert status.classifier_degraded_detail == ""

    with pytest.raises(ValueError):
        ActorStatus(
            schema_version=status.schema_version,
            state="POLICY_RUNNING",
            control_owner="POLICY",
            run_id="run-0",
            episode_id=0,
            episode_step=0,
            env_step=0,
            classifier_evaluated=False,
            classifier_probability=0.0,
            classifier_threshold=0.0,
            classifier_env_step=-1,
            success=False,
            terminal_reason="",
            message="",
            classifier_degraded="yes",  # type: ignore[arg-type]
        )


def test_a_detail_without_a_degraded_flag_is_dropped():
    """The two fields cannot disagree: no flag means no explanation either."""
    tracker = ActorStatusTracker("run-0")

    status = tracker.status(
        state="POLICY_RUNNING",
        control_owner="POLICY",
        episode_id=0,
        episode_step=0,
        env_step=0,
        classifier_degraded=False,
        classifier_degraded_detail="stale text from a recovered fault",
    )

    assert status.classifier_degraded_detail == ""


def _publish(status: ActorStatus) -> dict:
    """Exactly what RosOperatorSession puts on the wire, and back again."""
    return _gui_status_module().parse_actor_status(
        json.dumps(status.as_payload())
    )


def test_the_gui_reads_the_degraded_state_off_the_published_status():
    gui = _gui_status_module()
    classifier_verdict_summary = gui.classifier_verdict_summary
    format_actor_status = gui.format_actor_status

    reporter = _OperatorReporter(session=None, run_id="run-0")
    reporter.set_classifier_degraded(True, _DEGRADED_HEALTH)
    parsed = _publish(reporter.status("POLICY_RUNNING", "POLICY", message=""))

    assert parsed["classifier_degraded"] is True
    assert parsed["classifier_degraded_detail"] == _DEGRADED_HEALTH

    headline, mode_text, verdict = classifier_verdict_summary(
        None, auto_success=True, degraded=parsed["classifier_degraded"]
    )
    assert "DEGRADED" in headline
    assert verdict is None  # -> red in the GUI, not the "no result yet" grey
    assert "NEVER declare success" in mode_text

    formatted = format_actor_status(
        parsed, classifier_latch=None, terminal_latch="", age_s=0.1
    )
    assert "FAULTED" in formatted["classifier_current"]
    assert "sparse/unscored" not in formatted["classifier_current"]
    assert _DEGRADED_HEALTH in formatted["classifier_current"]


def test_a_gui_that_predates_the_field_still_parses_the_status():
    """The reason this is not a SCHEMA_VERSION bump.

    The GUI runs from the built install/ overlay, which lags this tree; a bump
    would make a stale overlay reject every status and lose the whole actor
    panel, deadman-adjacent state banner included, in exchange for a classifier
    warning.  An unknown extra field is ignored instead.
    """
    gui = _gui_status_module()
    SCHEMA_VERSION = gui.SCHEMA_VERSION
    parse_actor_status = gui.parse_actor_status

    reporter = _OperatorReporter(session=None, run_id="run-0")
    reporter.set_classifier_degraded(True, "inference failed")
    payload = reporter.status("POLICY_RUNNING", "POLICY", message="").as_payload()

    assert payload["schema_version"] == SCHEMA_VERSION

    # Old actor, new GUI: the fields simply are not there.
    payload.pop("classifier_degraded")
    payload.pop("classifier_degraded_detail")
    parsed = parse_actor_status(json.dumps(payload))
    assert parsed["classifier_degraded"] is False
    assert parsed["classifier_degraded_detail"] == ""


def test_malformed_degraded_telemetry_is_still_rejected():
    parse_actor_status = _gui_status_module().parse_actor_status

    reporter = _OperatorReporter(session=None, run_id="run-0")
    payload = reporter.status("POLICY_RUNNING", "POLICY", message="").as_payload()
    payload["classifier_degraded"] = 1

    with pytest.raises(ValueError):
        parse_actor_status(json.dumps(payload))

    payload["classifier_degraded"] = True
    payload["classifier_degraded_detail"] = ["not", "a", "string"]
    with pytest.raises(ValueError):
        parse_actor_status(json.dumps(payload))


def test_the_probe_survives_a_network_without_health():
    """Test doubles and the local loop need not implement Health at all."""
    probe = _ClassifierDegradedProbe(SimpleNamespace(), clock=_Clock())

    assert probe.observe(attached=True, evaluated=False) is True
    assert probe.degraded is True
    assert "no verdict" in probe.detail


# ---------------------------------------------------------------------------
# The wiring, through the real actor loop.
# ---------------------------------------------------------------------------


class _ActionSpace:
    shape = (7,)


class _LoopEnv:
    """Minimal env with cameras, so the loop actually attaches a sidecar."""

    action_space = _ActionSpace()

    def __init__(self) -> None:
        self.reset_count = 0
        self.step_count = 0

    @staticmethod
    def _observation(marker: int) -> dict:
        import numpy as np

        value = np.uint8(marker % 255)
        return {
            "state": np.full((1, 19), float(marker), dtype=np.float32),
            "cam1": np.full((1, 4, 4, 3), value, dtype=np.uint8),
            "cam2": np.full((1, 4, 4, 3), value, dtype=np.uint8),
        }

    def last_camera_frames(self) -> dict:
        import numpy as np

        return {
            "cam1": np.zeros((48, 64, 3), dtype=np.uint8),
            "cam2": np.zeros((48, 64, 3), dtype=np.uint8),
        }

    def reset(self, **kwargs):
        import numpy as np

        self.reset_count += 1
        return self._observation(self.reset_count), {
            "timestamp_ns": np.int64(1_000 + self.reset_count)
        }

    def step(self, action):
        import numpy as np

        self.step_count += 1
        return (
            self._observation(100 + self.step_count),
            0.0,
            False,
            False,
            {
                "timestamp_ns": np.int64(2_000 + self.step_count),
                "intervened": 0,
                "held": False,
            },
        )


class _AlwaysAttachScheduler:
    def reset(self) -> None:
        pass

    def should_attach(self, state, terminal) -> bool:
        return True

    def note_outcome(self, outcome) -> None:
        pass


class _UnscoredNetwork(_Network):
    """Acks every step but never returns a classifier verdict."""

    def begin_episode(self, observation, **kwargs):
        import numpy as np

        return SimpleNamespace(
            action=np.zeros(7, dtype=np.float32), policy_version=0
        )

    def step(self, next_observation, **kwargs):
        import numpy as np

        from ur_env.actor_network import TransitionOutcome

        outcome = TransitionOutcome(
            transition_id=kwargs["data"]["meta"]["transition_id"],
            reward=0.0,
            mask=1.0,
            done=False,
            truncated=False,
            success=False,
            classifier_evaluated=False,
            classifier_probability=0.0,
            classifier_threshold=0.0,
            reward_model_id="",
        )
        return SimpleNamespace(
            outcome=outcome,
            action=SimpleNamespace(
                action=np.zeros(7, dtype=np.float32), policy_version=1
            ),
        )


class _RecordingSession:
    def __init__(self) -> None:
        self.statuses = []

    def publish(self, status) -> None:
        self.statuses.append(status)

    def wait_for_scene_ready(self, status) -> None:
        self.statuses.append(status)

    def wait_for_home_approval(self, status) -> None:
        self.statuses.append(status)


def test_the_actor_loop_publishes_the_degraded_state_it_observes(capsys):
    """End to end in-process: sidecar attached, no verdict, GUI told."""
    from ur_env.remote_actor import run_remote_actor

    network = _UnscoredNetwork(_DEGRADED_HEALTH)
    session = _RecordingSession()

    run_remote_actor(
        network,
        _LoopEnv(),
        config=SimpleNamespace(max_steps=3, random_steps=0, buffer_period=0),
        actor_id="actor-0",
        run_id="run-0",
        sidecar_scheduler=_AlwaysAttachScheduler(),
        operator_session=session,
    )

    degraded = [s for s in session.statuses if s.classifier_degraded]
    assert degraded, "no published status carried the degraded flag"
    assert _DEGRADED_HEALTH in degraded[0].classifier_degraded_detail
    # It keeps being published: this is a condition, not one notification.
    assert len(degraded) >= 2
    # ... while the terminal says it once, no matter how many steps ran.
    printed = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "reward classifier DEGRADED" in line
    ]
    assert len(printed) == 1
