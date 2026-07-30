"""The 30 Hz in-window leader resampling path, driven for real.

WHY THIS FILE EXISTS
--------------------
Every other intervention test drives ``GelloIntervention`` against a *stub*
env whose ``step`` only records the action it was handed.  Those tests pin the
wrapper's bookkeeping, but they can never enter ``UR7eEnv.step``'s substep loop:
no controller, no backend, no window.  Without this file the whole substep
mechanism — the thing that turned 16-40% fully-stopped joint time into 0% —
would ship with zero executed coverage, which is exactly the trap G25 records
for ``test_env_fake_backend.py`` (a ``main()`` script pytest collects 0 tests
from, so its 10 Hz pacing and upsampler-budget assertions never actually run).

WHAT MAKES IT DETERMINISTIC
---------------------------
The substep loop is paced off the wall clock, so "how many substeps did this
window contain" is a wall-clock question and would be flaky on a loaded machine.
The ``time`` module of BOTH participants — ``ur_env.envs.ur7e_env`` (which paces
the window) and ``ur_env.envs.wrappers`` (which dates leader samples) — is
therefore replaced by a VIRTUAL clock that only advances when the code under
test sleeps.  Nothing about the loop changes; time simply stops being a race.
(``test_deadman_wiring`` already uses this technique on
``ur_experiments.cube_in_cup``.)

The leader is likewise deterministic: the fake backend reproduces
``URRosBackend._aged`` semantics exactly — it returns ``(sample, monotonic() -
receive_instant)`` — because the wrapper identifies a NEW leader sample by
reconstructing that receive instant, and a fake that returned a constant ``age``
would make "is this the same sample?" depend on real elapsed time.  Whether a
new sample has landed is a separate knob (``sample_every_n_polls``), because
polling a 30 Hz cache at 30 Hz genuinely does re-read samples.

ROS-free and robot-free: no rclpy node, no camera, no serial port, and pynput is
blocked so running the suite never installs a global keyboard listener.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA = os.path.join(_HERE, "..")
_REPO = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _INFRA)
# ur_gello_bringup.ur_kin (fk / ik_numeric / jacobian) is pure numpy.
sys.path.insert(0, os.path.join(_REPO, "ros2_ur_ws", "src", "ur_gello_bringup"))

from ur_env.envs import ur7e_env as ur7e_env_module  # noqa: E402
from ur_env.envs import wrappers as wrappers_module  # noqa: E402
from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.ur7e_env import UR7eEnv  # noqa: E402
from ur_env.envs.wrappers import DeadmanSource, GelloIntervention  # noqa: E402

Q6 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])


# --------------------------------------------------------------------------- #
# fixtures / doubles                                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _block_pynput(monkeypatch):
    """``UR7eEnv.__init__`` builds an ESC listener inside a bare except."""

    monkeypatch.setitem(sys.modules, "pynput", None)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", None)


class _VirtualClock:
    """A ``time`` stand-in that advances ONLY when something sleeps."""

    def __init__(self, start: float = 0.0):
        self.now = float(start)
        self.sleeps = []

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        seconds = float(seconds)
        self.sleeps.append(seconds)
        if seconds > 0.0:
            self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    """Virtual time for BOTH modules that measure it in the loop under test.

    ``ur7e_env`` paces the window; ``wrappers`` identifies a new leader sample by
    ``time.monotonic() - age``.  Patching only the first would leave sample
    identity depending on how many real microseconds a substep happened to take —
    the flakiness this fixture exists to remove.
    """

    virtual = _VirtualClock()
    monkeypatch.setattr(ur7e_env_module, "time", virtual)
    monkeypatch.setattr(wrappers_module, "time", virtual)
    return virtual


class _FakeDeadman(DeadmanSource):
    """Operator signal under direct test control (no keyboard, no topic)."""

    def __init__(self, engaged=True, gain=1.0):
        self.engaged = bool(engaged)
        self._gain = float(gain)

    def is_engaged(self):
        return self.engaged

    def gain(self):
        return self._gain


class _SubstepBackend:
    """URRosBackend stand-in: an ideal robot plus a scriptable GELLO leader.

    ``dry_run`` is True only so ``go_to_reset`` skips its arrival loop; commands
    are still recorded and still teleport the fake robot, because the substep
    loop's whole observable effect is "how many targets did you publish, and
    what were they".
    """

    dry_run = True

    def __init__(self, clock, q_robot=Q6, q_leader=Q6):
        self._clock = clock
        self.q = np.asarray(q_robot, dtype=float).copy()
        self.leader_q = np.asarray(q_leader, dtype=float).copy()
        self._leader_rx = clock.monotonic()
        #: per-sample leader advance, i.e. simulated human motion
        self.leader_step = np.zeros(6)
        #: a NEW sample lands on every n-th poll; >1 means the substep loop
        #: re-reads a sample the leader publisher has not replaced yet.
        self.sample_every_n_polls = 1
        #: polls strictly after this index report an unusable (stale) leader
        self.leader_stale_after = None
        #: called as on_poll(poll_index) before each leader reading is built
        self.on_poll = None

        self.polls = 0
        self.commands = []
        self.holds = 0

    # ---- leader ---- #
    def get_gello_state(self):
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self.polls)
        if self.leader_stale_after is not None and self.polls > self.leader_stale_after:
            # Fresh joints, dead clock: exactly the LEADER_STALE_S failure.
            return np.concatenate([self.leader_q, [float("nan")]]), 999.0
        if (self.polls - 1) % self.sample_every_n_polls == 0:
            self.leader_q = self.leader_q + self.leader_step
            self._leader_rx = self._clock.monotonic()
        # Reproduce URRosBackend._aged (ros_backend.py:567-571) exactly: the age
        # is measured against the sample's receive instant, which is how the
        # wrapper recovers that instant and tells samples apart.
        return (
            np.concatenate([self.leader_q, [float("nan")]]),
            self._clock.monotonic() - self._leader_rx,
        )

    # ---- robot state ---- #
    def get_joint_state(self):
        return self.q.copy(), np.zeros(6), 0.0

    def get_gripper_percent(self):
        return 0.0, 0.0

    def get_wrench(self):
        return None, float("inf")

    def get_tcp_pose(self):
        return None, float("inf")

    # ---- commands ---- #
    def send_joint_command(self, q_cmd):
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(6).copy()
        self.commands.append(q_cmd)
        self.q = q_cmd

    def send_gripper_percent(self, fraction):
        pass

    def request_hold(self):
        self.holds += 1
        return self.q.copy()

    def reset_command_stream(self):
        pass

    def close(self):
        pass


class _Config(DefaultUR7eEnvConfig):
    """No cameras, fk poses, no gripper debounce — the control path only."""

    CAMERAS: dict = {}
    IMAGE_CROP: dict = {}
    DISPLAY_IMAGE = False
    TCP_POSE_SOURCE = "fk"
    RESET_JOINTS = Q6
    RESET_POSE = np.zeros(6)
    MAX_EPISODE_LENGTH = 1000
    GRIPPER_SLEEP = 0.0


def _build(clock, *, hz=10.0, substep_hz=30.0, leader_step=0.0, deadman=None):
    """A reset, engaged GelloIntervention over a real UR7eEnv + fake backend.

    ``substep_hz=None`` omits the whole ``INTERVENTION`` block, i.e. the exact
    pre-substep code path.
    """

    config = _Config()
    config.HZ = float(hz)
    if substep_hz is None:
        config.INTERVENTION = {}
    else:
        config.INTERVENTION = dict(
            DefaultUR7eEnvConfig.INTERVENTION, substep_hz=float(substep_hz)
        )

    backend = _SubstepBackend(clock)
    backend.leader_step = np.full(6, float(leader_step))
    env = UR7eEnv(fake_env=False, config=config, backend=backend)
    wrapper = GelloIntervention(
        env, deadman=deadman if deadman is not None else _FakeDeadman(engaged=True)
    )
    wrapper.reset()
    backend.commands.clear()
    backend.polls = 0
    return wrapper, env, backend


def _zeros_action():
    return np.zeros(7, dtype=np.float32)


# =========================================================================== #
# 1. the loop actually runs, and every substep publishes one target            #
# =========================================================================== #
def test_substeps_run_and_each_one_refreshes_the_joint_target(clock):
    wrapper, env, backend = _build(clock, hz=10.0, substep_hz=30.0, leader_step=0.002)

    started = clock.now
    _, _, _, _, info = wrapper.step(_zeros_action())

    # 30 Hz inside a 10 Hz window: the target set by _apply_action plus two
    # refreshes.  Publishing once was the whole defect.
    assert info["intervention_substeps"] == 2
    assert len(backend.commands) == 3
    # ...and each one is a DIFFERENT target: a loop that re-sent the same goal
    # would leave the upsampler exactly as starved as before.
    assert not np.array_equal(backend.commands[0], backend.commands[1])
    assert not np.array_equal(backend.commands[1], backend.commands[2])
    # The env step period is unchanged — substepping spends the sleep, not more.
    assert clock.now - started == pytest.approx(1.0 / env.hz, abs=1e-6)


def test_substep_count_follows_the_configured_rate(clock):
    wrapper, _, backend = _build(clock, hz=10.0, substep_hz=60.0, leader_step=0.001)

    _, _, _, _, info = wrapper.step(_zeros_action())

    assert info["intervention_substeps"] == 5  # 60 Hz over a 100 ms window
    assert len(backend.commands) == 6


# =========================================================================== #
# 2. the window can never spend more than one ACTION_SCALE step               #
# =========================================================================== #
def test_a_leader_far_faster_than_the_budget_cannot_overspend_the_window(clock):
    # 0.03 rad of leader travel per poll is many times the budget: without the
    # shared InterventionBudget the three targets would sum to ~3x ACTION_SCALE
    # while the stored action saturated at 1.0 — storage silently UNDER-reporting
    # the motion, the exact invariant breach the budget exists to prevent.
    wrapper, env, backend = _build(clock, hz=10.0, substep_hz=30.0, leader_step=0.03)

    wrapper.step(_zeros_action())  # engage / anchor
    p_before = env.controller.tcp_cmd()[:3, 3].copy()
    _, _, _, _, info = wrapper.step(_zeros_action())
    travelled = float(np.linalg.norm(env.controller.tcp_cmd()[:3, 3] - p_before))

    assert travelled <= env.action_scale[0] * (1.0 + 1e-9)
    # The window still got its full budget AND all of its target refreshes: a
    # first target allowed to drain the budget at t=0 would silently reduce this
    # to one refresh and 0.0050 m of travel (the governor's per-substep cap),
    # i.e. reinstate most of the dead time this change exists to remove.
    assert info["intervention_substeps"] == 2
    assert travelled == pytest.approx(env.action_scale[0], rel=1e-3)
    assert np.linalg.norm(np.asarray(info["intervene_action"])[:3]) <= 1.0
    assert np.all(np.abs(np.asarray(info["intervene_action"])) <= 1.0)


def test_the_recorded_action_is_the_displacement_the_window_commanded(clock):
    """저장 액션 불변식, measured on the real control path.

    ``info["intervene_action"]`` must BE the motion the window commanded, not a
    request that something downstream then shrank.  The saturating case is the
    one that catches an over-charge: the budget is charged with the request, so
    a request bigger than the governor's per-substep allowance would be billed
    in full and executed in part (measured 2.5x over-report before the request
    was paced to the budget's own per-substep share).
    """

    wrapper, env, backend = _build(clock, hz=10.0, substep_hz=30.0, leader_step=0.03)
    wrapper.step(_zeros_action())  # engage / anchor

    for _ in range(4):
        p_before = env.controller.tcp_cmd()[:3, 3].copy()
        _, _, _, _, info = wrapper.step(_zeros_action())
        commanded = env.controller.tcp_cmd()[:3, 3] - p_before
        recorded = np.asarray(info["intervene_action"], dtype=float)[:3]

        # PolicyDeltaController integrates position as a plain vector sum, so
        # the two are equal by construction whenever nothing was cut.
        np.testing.assert_allclose(
            recorded * env.action_scale[0], commanded, rtol=2e-3, atol=1e-7
        )


def test_the_recorded_action_stays_a_legal_stored_action_over_many_windows(clock):
    wrapper, env, backend = _build(clock, hz=10.0, substep_hz=30.0, leader_step=0.02)

    for _ in range(6):
        _, _, _, _, info = wrapper.step(_zeros_action())
        recorded = np.asarray(info["intervene_action"])
        assert recorded.shape == (7,)
        assert np.all(np.isfinite(recorded))
        # The actor's validate_action (ur_env/actor_network.py:266-275) rejects
        # anything outside this, and rejection means a dropped transition.
        assert np.all(recorded >= -1.0) and np.all(recorded <= 1.0)
        assert recorded[6] in (-1.0, 0.0, 1.0)


# =========================================================================== #
# 3. consumed_action() is what reaches info["intervene_action"]                #
# =========================================================================== #
def test_the_budgets_consumed_action_is_the_reported_action(clock):
    wrapper, env, backend = _build(clock, hz=10.0, substep_hz=30.0, leader_step=0.004)

    wrapper.step(_zeros_action())
    _, _, _, _, info = wrapper.step(_zeros_action())

    expected = np.zeros(7, dtype=np.float32)
    expected[:6] = np.clip(
        wrapper._budget.consumed_action().astype(np.float32), -1.0, 1.0
    )
    expected[6] = wrapper._window_gripper
    np.testing.assert_array_equal(info["intervene_action"], expected)
    assert np.asarray(info["intervene_action"]).dtype == np.float32
    # It really is the SUM of the window, not just the first target.
    assert np.any(np.abs(expected[:6]) > 0.0)
    # The transport key is consumed, not left behind for RelativeFrame to find
    # in the base frame after intervene_action has been rotated out of it.
    assert "intervention_window_action" not in info


# =========================================================================== #
# 4. backwards compatibility: no substep -> byte-for-byte the old behaviour    #
# =========================================================================== #
def test_a_window_with_no_room_for_a_substep_reproduces_the_old_path(clock):
    """substep_hz == HZ leaves no room, and must then change nothing at all.

    Compared against an env with no ``INTERVENTION`` block whatsoever — the
    literal pre-change code path (no driver, no budget, ``controller.step(xi)``
    with no ``dt``) — driven through an identical leader script.
    """

    plain_wrapper, plain_env, plain_backend = _build(
        clock, hz=10.0, substep_hz=None, leader_step=0.003
    )
    same_wrapper, same_env, same_backend = _build(
        clock, hz=10.0, substep_hz=10.0, leader_step=0.003
    )

    for _ in range(4):
        _, _, _, _, plain_info = plain_wrapper.step(_zeros_action())
        _, _, _, _, same_info = same_wrapper.step(_zeros_action())

        assert same_info.get("intervention_substeps", 0) == 0
        np.testing.assert_array_equal(
            same_info["intervene_action"], plain_info["intervene_action"]
        )

    assert len(same_backend.commands) == len(plain_backend.commands) == 4
    for expected, actual in zip(plain_backend.commands, same_backend.commands):
        np.testing.assert_array_equal(actual, expected)


def test_the_policy_path_never_installs_a_driver_and_keeps_dt_none(clock):
    """The policy path must not change by one bit, so assert the call itself."""

    wrapper, env, backend = _build(
        clock, hz=10.0, substep_hz=30.0, deadman=_FakeDeadman(engaged=False)
    )
    seen = []
    real_step = env.controller.step

    def spy(xi, dt=None):
        seen.append(dt)
        return real_step(xi, dt)

    env.controller.step = spy

    policy_action = np.zeros(7, dtype=np.float32)
    policy_action[0] = 0.5
    _, _, _, _, info = wrapper.step(policy_action)

    assert seen == [None]                       # nominal 1/HZ governor budget
    assert len(backend.commands) == 1           # one target, one window
    assert "intervene_action" not in info
    assert "intervention_substeps" not in info
    assert env._intervention_driver is None


# =========================================================================== #
# 5. the leader dying mid-window ends it and records an exact HOLD             #
# =========================================================================== #
def test_a_leader_lost_mid_window_brakes_and_records_exactly_zeros(clock):
    wrapper, env, backend = _build(clock, hz=10.0, substep_hz=30.0, leader_step=0.004)
    # Poll 1 is GelloIntervention.action() (fresh: it anchors); every substep
    # poll after it reports an unusable leader.
    backend.leader_stale_after = 1

    started = clock.now
    _, _, _, _, info = wrapper.step(_zeros_action())

    np.testing.assert_array_equal(
        info["intervene_action"], np.zeros(7, dtype=np.float32)
    )
    assert info["intervened"] == 1               # never an implicit handback
    assert info["held"] is True
    assert info["reject_reason"] == "GELLO_STALE"
    assert info["intervention_substeps"] == 0    # the stale substep issued nothing
    assert len(backend.commands) == 1            # only _apply_action's target
    assert backend.holds == 1                    # braked to the stop point
    # The controller was re-anchored on the braking endpoint, so the wrapper's
    # anchor is dropped and the next engaged step re-anchors with a zero delta.
    assert wrapper._anchored is False
    # ...and the step period is still the step period.
    assert clock.now - started == pytest.approx(1.0 / env.hz, abs=1e-6)


def test_a_leader_lost_mid_window_recovers_on_the_next_window(clock):
    wrapper, env, backend = _build(clock, hz=10.0, substep_hz=30.0, leader_step=0.004)
    backend.leader_stale_after = 1
    wrapper.step(_zeros_action())
    assert wrapper._anchored is False

    backend.leader_stale_after = None
    _, _, _, _, info = wrapper.step(_zeros_action())

    assert wrapper._anchored is True
    assert info["intervened"] == 1
    assert info.get("reject_reason") in (None, "BUDGET_EXHAUSTED")
    assert info["intervention_substeps"] >= 1


# =========================================================================== #
# 6. the sensitivity gain is latched at the engage edge, not read per substep  #
# =========================================================================== #
def test_a_slider_moved_mid_window_does_not_rescale_that_window(clock):
    """Same guarantee as test_gain_is_latched_at_the_engage_edge, per SUBSTEP.

    Reading ``expert.gain()`` inside the loop would be the natural way to write
    it and would reintroduce a discontinuity 30 times a second instead of 10.
    """

    def _takeover(gain, slider_moves_to=None):
        deadman = _FakeDeadman(engaged=True, gain=gain)
        wrapper, _, backend = _build(
            clock, hz=10.0, substep_hz=30.0, leader_step=0.001, deadman=deadman
        )
        if slider_moves_to is not None:
            def _move(poll_index, deadman=deadman, moves_to=slider_moves_to):
                if poll_index >= 2:  # poll 1 is action(); 2+ are substeps
                    deadman._gain = moves_to

            backend.on_poll = _move
        _, _, _, _, info = wrapper.step(_zeros_action())
        return np.asarray(info["intervene_action"], dtype=float)

    latched = _takeover(0.5, slider_moves_to=1.0)
    quiet = _takeover(0.5)
    full = _takeover(1.0)

    # The mid-window slider move changed nothing...
    np.testing.assert_array_equal(latched, quiet)
    # ...and the gain does otherwise scale translation, so this is not vacuous.
    assert np.linalg.norm(full[:3]) > np.linalg.norm(quiet[:3]) > 0.0
    np.testing.assert_allclose(quiet[:3], 0.5 * full[:3], rtol=1e-5, atol=1e-12)


# =========================================================================== #
# 7. the one-euro filter sees each leader sample exactly once                  #
# =========================================================================== #
def _count_note_sample(wrapper):
    calls = []
    real = wrapper._filter.note_sample

    def counting(q_lead, dt):
        calls.append(dt)
        real(q_lead, dt)

    wrapper._filter.note_sample = counting
    return calls


def test_re_reading_the_same_cached_leader_sample_notes_it_only_once(clock):
    """The backend caches at 30 Hz; polling it 30 times/s mostly re-reads.

    Feeding every poll to ``note_sample`` would advance the filter's speed
    estimate with dt values far below the true sample spacing, so its cutoff
    would open in a pulse per sample instead of low-passing (LeaderFilter class
    docstring).  A frozen leader is the extreme case: many polls, one sample.
    """

    wrapper, _, backend = _build(clock, hz=10.0, substep_hz=30.0, leader_step=0.002)
    # The publisher replaces the cached sample far more rarely than we poll it,
    # so every poll after the first re-reads a sample already consumed.
    backend.sample_every_n_polls = 1000
    calls = _count_note_sample(wrapper)

    wrapper.step(_zeros_action())   # 1 action() poll + 2 substep polls
    wrapper.step(_zeros_action())   # 3 more polls of the same cached sample
    assert backend.polls == 6

    assert len(calls) == 1, "a re-read of one cached sample must count once"
    assert calls[0] is None, "the first sample has no interval to divide by"


def test_every_genuinely_new_leader_sample_is_noted_with_its_own_interval(clock):
    wrapper, _, backend = _build(clock, hz=10.0, substep_hz=30.0, leader_step=0.001)
    calls = _count_note_sample(wrapper)

    wrapper.step(_zeros_action())

    assert backend.polls == 3
    assert len(calls) == 3
    assert calls[0] is None
    # Real, positive sample intervals — never the same instant twice.
    assert all(dt is not None and dt > 0.0 for dt in calls[1:])


def test_the_filter_is_dropped_at_every_engage_edge(clock):
    """A takeover must not inherit the leader speed of the previous one."""

    deadman = _FakeDeadman(engaged=True)
    wrapper, _, backend = _build(
        clock, hz=10.0, substep_hz=30.0, leader_step=0.002, deadman=deadman
    )
    wrapper.step(_zeros_action())
    assert wrapper._leader_rx is not None

    deadman.engaged = False
    wrapper.step(_zeros_action())        # policy resumes; anchor dropped
    assert wrapper._anchored is False

    deadman.engaged = True
    calls = _count_note_sample(wrapper)
    wrapper.step(_zeros_action())        # fresh engage edge

    assert wrapper._anchored is True
    assert calls[0] is None, "the engage edge must make the next sample the first"


# =========================================================================== #
# 8. substepping is off unless a config asks for it                            #
# =========================================================================== #
def test_a_config_without_an_intervention_block_disables_substepping(clock):
    wrapper, env, backend = _build(clock, hz=10.0, substep_hz=None, leader_step=0.002)

    _, _, _, _, info = wrapper.step(_zeros_action())

    assert env.intervention_substep_hz == 0.0
    assert wrapper.substep_hz == 0.0
    assert wrapper._filter is None and wrapper._budget is None
    assert "intervention_substeps" not in info
    assert len(backend.commands) == 1


def test_the_shipped_default_config_enables_substepping_at_the_leader_rate():
    """30.0 is pinned to gello_publisher.publish_rate_hz — not a free knob."""

    assert DefaultUR7eEnvConfig.INTERVENTION["substep_hz"] == 30.0
    yaml_path = os.path.join(
        _REPO, "ros2_ur_ws", "src", "ur_gello_bringup", "config", "ur7e_gello.yaml"
    )
    with open(yaml_path, encoding="utf-8") as handle:
        yaml_text = handle.read()
    assert "publish_rate_hz: 30.0" in yaml_text
    for name, value in (
        ("one_euro_min_cutoff", 1.0),
        ("one_euro_beta", 2.0),
        ("one_euro_d_cutoff", 1.0),
    ):
        assert DefaultUR7eEnvConfig.INTERVENTION[name] == value
        assert f"{name}: {value}" in yaml_text
