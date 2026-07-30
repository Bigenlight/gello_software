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


def _build(
    clock,
    *,
    hz=10.0,
    substep_hz=30.0,
    leader_step=0.0,
    deadman=None,
    governor=None,
):
    """A reset, engaged GelloIntervention over a real UR7eEnv + fake backend.

    ``substep_hz=None`` omits the whole ``INTERVENTION`` block, i.e. the exact
    pre-substep code path.

    ``governor`` overrides individual ``GOVERNOR`` keys.  Tightening ``v_max``
    is the ONLY way to make the task-space rate cap bind on the intervention
    path at all: with the shipped caps a paced request
    (``ACTION_SCALE[0]/3`` = 0.00417 m) always fits inside the per-substep
    allowance (``v_max/substep_hz`` = 0.0050 m), which is exactly why the
    2026-07-30 ARMED hardware run reported ``governed=0`` in every window.
    """

    config = _Config()
    config.HZ = float(hz)
    if substep_hz is None:
        config.INTERVENTION = {}
    else:
        config.INTERVENTION = dict(
            DefaultUR7eEnvConfig.INTERVENTION, substep_hz=float(substep_hz)
        )
    if governor is not None:
        config.GOVERNOR = dict(DefaultUR7eEnvConfig.GOVERNOR, **governor)

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


@pytest.mark.parametrize(
    "substep_hz, expected_substeps", [(12.0, 0), (15.1, 1), (20.0, 1), (30.0, 2)]
)
def test_a_substep_only_fits_above_one_and_a_half_times_hz(
    clock, substep_hz, expected_substeps
):
    """The half-period guard puts the real threshold at 1.5*HZ, not HZ.

    ``_drive_intervention_substeps`` refuses a tick with less than half a period
    of window left, so a substep only fits when ``1.5/substep_hz < 1/HZ``.  Two
    consequences worth pinning, both reported rather than silently adjusted
    (which of the two rates is wrong is a task decision, exactly as the
    off-nominal-HZ warning in ``UR7eEnv.__init__`` says, and config.py pins
    substep_hz to 30.0 with no CLI exposing it):

    * The startup NOTE beside that loop is worded against ``substep_hz <= HZ``,
      so a ``substep_hz`` in ``(HZ, 1.5*HZ]`` — 12.0 here — runs ZERO substeps
      while announcing nothing at all.
    * ``substep_hz == 1.5*HZ`` exactly (15.0 at HZ=10) is float-DEGENERATE and
      is therefore deliberately absent from the cases below: ``next_tick`` and
      ``window_end - 0.5*period`` land on the same value up to rounding, so the
      identical config yields 0 or 1 substeps depending on the window's absolute
      start instant (measured: 0 on the second window, 1 on the first).  Only
      the logged ``intervention_substeps`` column is affected — the displacement
      budget bounds the window either way — but a config sitting on that point
      would log two different things for the same setup.
    """

    wrapper, _, _ = _build(
        clock, hz=10.0, substep_hz=substep_hz, leader_step=0.002
    )

    _, _, _, _, info = wrapper.step(_zeros_action())

    assert info["intervention_substeps"] == expected_substeps


# =========================================================================== #
# 9. governed/governed_scale describe the WHOLE window, not its first tick     #
# =========================================================================== #
class _ScriptedDriver:
    """The substep-driver protocol with scripted substep results, nothing else.

    WHY A DOUBLE HERE, when the rest of this file deliberately drives the real
    ``GelloIntervention``.  With the shipped GOVERNOR the governor can never bind
    on the intervention path: ``_paced_request`` shrinks every request to
    ``ACTION_SCALE/3`` (0.00417 m), which is below the per-substep allowance
    ``v_max/substep_hz`` (0.0050 m).  So on the real path all three ticks of a
    window carry the same ``governed_scale`` and the aggregation cannot be told
    apart from any of its broken variants.  Scripting the substeps is the only
    way to ask "WHICH tick's cut ends up in info".

    The first target is NOT scripted: ``charge()`` returns a real ``xi`` and the
    real ``PolicyDeltaController`` produces its ``governed_scale``, so that half
    of the merge is measured rather than asserted into existence.  The
    end-to-end counterpart — the real driver, a tightened ``v_max`` — is the
    last test in this section.
    """

    def __init__(self, results, *, first_xi=None, window_action=None):
        self._results = [dict(r) for r in results]
        self._first_xi = (
            np.zeros(6) if first_xi is None else np.asarray(first_xi, dtype=float)
        )
        self._window_action = (
            np.zeros(7, dtype=np.float32)
            if window_action is None
            else np.asarray(window_action, dtype=np.float32)
        )
        self.charged = []
        self.substep_dts = []

    def charge(self, xi):
        self.charged.append(np.asarray(xi, dtype=float).copy())
        return self._first_xi.copy()

    def substep(self, dt):
        self.substep_dts.append(float(dt))
        if self._results:
            return dict(self._results.pop(0))
        return {"issued": True}

    def consumed_window_action(self):
        return self._window_action.copy()


def _cut(scale):
    """One substep result the governor shrank to ``scale``."""
    return {"issued": True, "governed": True, "governed_scale": float(scale)}


def _uncut():
    """One substep result nothing cut.

    ``governed_scale`` is PRESENT and 1.0 because that is what
    ``PolicyDeltaController.step`` always returns — the key is seeded on every
    tick.  An uncut substep that omitted the key would hide the ``dict.update``
    overwrite this section exists to pin.
    """
    return {"issued": True, "governed": False, "governed_scale": 1.0}


def _scripted_window(clock, *, substeps, first_scale=None, governor=None):
    """Run ONE intervened window: real env + real controller + scripted substeps.

    ``first_scale`` is the cut the governor must apply to the FIRST target, or
    None for a request small enough that nothing is cut.  The magnitude is
    derived from the live env (``v_max * 1/substep_hz``), so a config change
    cannot silently turn these tests into no-ops.
    """

    _, env, backend = _build(clock, hz=10.0, substep_hz=30.0, governor=governor)
    cap = env.controller.v_max / env.intervention_substep_hz
    xi = np.zeros(6)
    xi[0] = cap * 0.5 if first_scale is None else cap / float(first_scale)
    driver = _ScriptedDriver(substeps, first_xi=xi)
    env.begin_intervention_window(driver)
    _, _, _, _, info = env.step(_zeros_action())
    return info, driver, env, backend


def test_a_cut_that_happens_only_in_a_substep_still_reaches_the_window(clock):
    """First half of the defect: ``governed`` was missing from the OR list.

    ``held``/``clipped`` were OR-ed across substeps and ``governed`` was not, so
    a log line saying ``governed=0`` meant "the FIRST target was not cut" — and
    the first target is precisely the tick that, with the shipped config, is
    structurally never cut.  The claim an operator reads it as ("nothing in this
    window was truncated") was therefore unsupported by the data.
    """

    info, _, _, _ = _scripted_window(clock, substeps=[_cut(0.7), _uncut()])

    assert info["governed"] is True
    assert info["governed_scale"] == pytest.approx(0.7, rel=1e-6)


def test_a_cut_that_happens_only_at_the_first_target_survives_the_merge(clock):
    """Second half: ``dict.update`` let a later 1.0 erase the first target's cut.

    Not a corner case — the real substeps always report a ``governed_scale``
    (``PolicyDeltaController`` seeds the key on every tick), so an uncut substep
    after a cut first target overwrote it every single time.
    """

    info, _, _, _ = _scripted_window(
        clock, first_scale=0.5, substeps=[_uncut(), _uncut()]
    )

    assert info["governed"] is True
    assert info["governed_scale"] == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize(
    "first_scale, substep_scales, expected",
    [
        (0.9, [0.7, 1.0], 0.7),    # the tightest cut is a substep
        (0.5, [1.0, 1.0], 0.5),    # the tightest cut is the first target
        (None, [0.4, 0.8], 0.4),   # and it is not simply the last one seen
        (0.6, [0.6, 0.6], 0.6),    # a tie is still that same number
    ],
)
def test_governed_scale_is_the_tightest_cut_anywhere_in_the_window(
    clock, first_scale, substep_scales, expected
):
    """``governed_scale`` is a MINIMUM over the window, not a last-write-wins.

    It answers "how hard was this window cut at worst".  A mean or a last value
    would let a single 1.0 tick hide a 0.4 one, which is the same
    undiagnosable-truncation failure ``governed`` exists to remove.
    """

    substeps = [_uncut() if s >= 1.0 else _cut(s) for s in substep_scales]

    info, _, _, _ = _scripted_window(
        clock, first_scale=first_scale, substeps=substeps
    )

    assert info["governed_scale"] == pytest.approx(expected, rel=1e-6)
    # The flag and the number must never disagree: a consumer that reads only
    # one of them must not get a different answer from a consumer reading the
    # other (run_real_hil.py writes both to the same CSV row).
    assert info["governed"] is (expected < 1.0)


def test_an_uncut_window_reports_no_governing_at_all(clock):
    info, _, _, _ = _scripted_window(clock, substeps=[_uncut(), _uncut()])

    assert info["governed"] is False
    assert info["governed_scale"] == 1.0
    assert info["intervention_substeps"] == 2


def test_a_window_with_no_substeps_keeps_the_first_targets_governing(clock):
    """``substep_hz == HZ`` leaves no room, so nothing may be folded in.

    The fold-in must not invent a ``governed_scale`` for a window that never ran
    a substep; the first target's own number has to arrive unchanged.
    """

    wrapper, env, _ = _build(
        clock, hz=10.0, substep_hz=10.0, leader_step=0.01, governor={"v_max": 0.02}
    )
    wrapper.step(_zeros_action())          # engage / anchor

    seen = []
    real_step = env.controller.step

    def spy(xi, dt=None):
        q_cmd, ctrl_info = real_step(xi, dt)
        seen.append(float(ctrl_info["governed_scale"]))
        return q_cmd, ctrl_info

    env.controller.step = spy
    _, _, _, _, info = wrapper.step(_zeros_action())

    assert len(seen) == 1                  # one target, no substep
    assert info["intervention_substeps"] == 0
    assert info["governed"] is True        # not vacuous: the cap really did bind
    assert info["governed_scale"] == seen[0]


def test_the_policy_path_reports_only_its_own_governing(clock):
    """A policy step must not acquire substep aggregate keys, ever."""

    wrapper, env, _ = _build(
        clock,
        hz=10.0,
        substep_hz=30.0,
        governor={"v_max": 0.02},
        deadman=_FakeDeadman(engaged=False),
    )
    action = np.zeros(7, dtype=np.float32)
    action[0] = 1.0

    _, _, _, _, info = wrapper.step(action)

    assert env._intervention_driver is None
    assert "intervention_substeps" not in info
    assert info["governed"] is True
    # dt=None means the NOMINAL 1/HZ cap, NOT the 1/substep_hz one: 0.02*0.1 =
    # 0.0020 m allowed against a full-scale 0.0125 m request.
    expected = 0.02 * (1.0 / env.hz) / env.action_scale[0]
    assert info["governed_scale"] == pytest.approx(expected, rel=1e-6)


def test_a_held_first_target_never_gains_governing_from_a_substep(clock):
    """A held window takes the pre-substep branch and reports no governor keys.

    Reporting a substep's ``governed_scale`` on a held window would claim the
    rate cap truncated a command that was never issued at all.
    """

    wrapper, env, backend = _build(
        clock, hz=10.0, substep_hz=30.0, leader_step=0.01, governor={"v_max": 0.02}
    )
    wrapper.step(_zeros_action())          # engage / anchor
    issued_before = len(backend.commands)

    env.request_hold("EXTERNAL_HOLD")
    _, _, _, _, info = wrapper.step(_zeros_action())

    assert info["held"] is True
    assert info["reject_reason"] == "EXTERNAL_HOLD"
    assert "intervention_substeps" not in info
    # _apply_action's hold branch returns no governor keys of its own, so a held
    # window reports step()'s DEFAULTS — not values synthesized from a substep
    # aggregate that does not exist. The defaults exist so the info schema does
    # not depend on which branch ran (same rule as held/clipped); a held window
    # commanded nothing, so "not governed, scale 1.0" is the honest reading.
    assert info["governed"] is False and info["governed_scale"] == 1.0
    assert len(backend.commands) == issued_before


def test_the_real_intervention_path_reports_a_substep_only_cut(clock):
    """The same defect end to end: real driver, real controller, no scripting.

    The FIRST engaged window is the one real-path case where the two sources can
    be told apart.  ``_engage`` anchors on the very leader sample the first
    target is built from, so that target is the exact ZERO delta and cannot be
    governed, while every substep afterwards sees leader motion.  ``v_max`` is
    tightened because with the shipped value a paced request never reaches the
    cap — the same reason the 2026-07-30 ARMED run logged ``governed=0`` in
    every window.
    """

    wrapper, env, _ = _build(
        clock, hz=10.0, substep_hz=30.0, leader_step=0.01, governor={"v_max": 0.02}
    )
    seen = []
    real_step = env.controller.step

    def spy(xi, dt=None):
        q_cmd, ctrl_info = real_step(xi, dt)
        seen.append(float(ctrl_info["governed_scale"]))
        return q_cmd, ctrl_info

    env.controller.step = spy
    _, _, _, _, info = wrapper.step(_zeros_action())

    assert info["intervention_substeps"] == 2
    assert len(seen) == 3
    assert seen[0] == 1.0, "the engage-edge first target is a zero delta"
    assert all(s < 1.0 for s in seen[1:]), "only the substeps are cut here"
    assert info["governed"] is True
    assert info["governed_scale"] == pytest.approx(min(seen), rel=1e-6)


# =========================================================================== #
# 10. recorded action vs executed motion — the two places they can disagree    #
# =========================================================================== #
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN GAP (adversarial review 2026-07-30): InterventionBudget.take "
        "charges the REQUEST, so any gate BELOW the budget that shrinks the "
        "commanded step leaves info['intervene_action'] overstating the motion. "
        "GelloIntervention._paced_request argues this away for the GOVERNOR "
        "only; the IK line search is not covered by that argument and reports "
        "nothing (governed stays False). Fixing it needs a refund/commit path "
        "in leader_stream.InterventionBudget, i.e. outside ur7e_env.py."
    ),
)
def test_a_line_search_shrink_must_not_be_left_out_of_the_recorded_action(clock):
    """저장 액션 == 실행 액션, when the JOINT gate (not the governor) is binding.

    ``PolicyDeltaController``'s line search shrinks the task step until
    ``max|dq| <= dq_step_max``.  It fires whenever the requested step needs more
    joint travel than the (dt-scaled) gate allows — reachable by shrinking the
    gate, as done deterministically here, or at the shipped gate by a
    poorly-conditioned Jacobian, which is the same branch (commit 4197f5b
    measured the pre-line-search STEP_LIMIT storm at sigma=0.56, i.e. nowhere
    near a singularity).

    The budget has already been charged with the full request by then, and
    ``consumed_action()`` reports the charge, so the transition claims motion
    the arm never made.  ``governed`` is False here — the governor is not the
    gate that fired — so there is no observable at all.  Measured over-report
    at ``dq_step_max=0.002``: 28x.
    """

    wrapper, env, _ = _build(
        clock,
        hz=10.0,
        substep_hz=30.0,
        leader_step=0.05,
        governor={"dq_step_max": 0.002},
    )
    wrapper.step(_zeros_action())          # engage / anchor

    p_before = env.controller.tcp_cmd()[:3, 3].copy()
    _, _, _, _, info = wrapper.step(_zeros_action())
    commanded = float(
        np.linalg.norm(env.controller.tcp_cmd()[:3, 3] - p_before)
    )
    recorded = float(
        np.linalg.norm(np.asarray(info["intervene_action"], dtype=float)[:3])
    ) * env.action_scale[0]

    assert not info["held"], info["reject_reason"]
    assert info["governed"] is False, "the governor is not the gate under test"
    assert recorded == pytest.approx(commanded, rel=0.05)


def test_a_braked_window_records_zeros_for_a_target_it_already_commanded(clock):
    """Pins the SIZE of the deliberate under-report in a mid-window dropout.

    ``_drive_intervention_substeps`` records exactly ``zeros(7)`` when the
    leader dies mid-window, and argues the brake "has already cut whatever
    motion was in flight".  It cuts FUTURE motion; the first target of the
    window was issued from a fresh leader sample and its displacement is not
    unwound by ``request_hold()``/``controller.reset()`` — those re-anchor the
    integrator ON the stop point.  So the record under-reports a genuinely
    human-commanded displacement of one paced substep share, i.e. 1/3 of a
    full-scale action.  Deliberate, bounded, and asserted here so it stays
    measured instead of assumed.

    (The FIRST engaged window cannot show this — the engage edge makes its first
    target a zero delta — so this drops out on the second window.)
    """

    wrapper, env, backend = _build(
        clock, hz=10.0, substep_hz=30.0, leader_step=0.01
    )
    wrapper.step(_zeros_action())          # engage / anchor

    p_before = env.controller.tcp_cmd()[:3, 3].copy()
    issued_before = len(backend.commands)
    # Fresh for this window's action() poll, unusable for every substep poll.
    backend.leader_stale_after = backend.polls + 1

    _, _, _, _, info = wrapper.step(_zeros_action())

    np.testing.assert_array_equal(
        info["intervene_action"], np.zeros(7, dtype=np.float32)
    )
    assert info["held"] is True
    assert info["reject_reason"] == "GELLO_STALE"
    assert info["intervention_substeps"] == 0
    assert backend.holds == 1
    # The first target DID reach the backend before the brake...
    assert len(backend.commands) == issued_before + 1
    # ...and it moved the commanded pose by exactly one paced substep share
    # (InterventionBudget.nominal_substep_share = ACTION_SCALE/3), which the
    # zero record omits.
    commanded = float(
        np.linalg.norm(env.controller.tcp_cmd()[:3, 3] - p_before)
    )
    assert commanded == pytest.approx(env.action_scale[0] / 3.0, rel=1e-5)
