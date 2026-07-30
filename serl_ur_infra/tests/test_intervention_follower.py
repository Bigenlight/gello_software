"""The BACKGROUND intervention follower: mux, safety gates, and the record.

WHY THIS FILE EXISTS
--------------------
``tests/test_intervention_substeps.py`` covers the *in-window* path, where the
leader is resampled inside ``UR7eEnv.step``'s nominal 1/HZ window.  On the
production actor that window is 100 ms out of a 512 ms real step period — the
other ~412 ms is a blocking gRPC ``Step`` RPC plus the camera decode, both
outside ``env.step`` — so in-window following left the joint target unrefreshed
for 87% of every period.  Measured consequences: 2.4 cm/s top intervention speed
(12.4 cm/s on the validation rig), a target gap that crosses the upsampler's
``target_stale_s`` and brakes to HOLD once per window, and a ``soft_start_s``
re-arm after each of those brakes that pins the slew ceiling at 46%.

``follow_mode="background"`` moves the following onto a daemon thread that runs
for the whole real period, and turns the RL loop into an observer.  That buys the
fix and buys a second, larger safety problem: two threads now reach one arm.
This file is the coverage for both halves.

WHAT MAKES IT DETERMINISTIC
---------------------------
``_follow_tick(now)`` takes its clock as an ARGUMENT and reads none of its own —
the same rule ``AccelerationLimitedJointStream.advance`` and
``PolicyDeltaController.step`` already follow.  So almost everything here is
tested by CALLING IT, with a virtual clock and no thread at all: the loop's
pacing shell is three lines and has nothing to assert about.

The handful of properties that only exist because there IS a thread — the
ownership mux, the teardown order, the lock that makes ``step()`` atomic — are
driven with real threads and forced into their worst interleaving with
``Event``/``Barrier``.  Nothing here waits on a ``sleep`` for a state to become
true; the two ``sleep`` calls that do appear are DISPROOF windows ("assert this
did NOT happen during 5 tick periods") and are commented as such.

ROS-free and robot-free: no rclpy node, no camera, no serial port, and pynput is
blocked so running the suite never installs a global keyboard listener.
"""

import os
import sys
import threading

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_INFRA = os.path.join(_HERE, "..")
_REPO = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _INFRA)
# ur_gello_bringup.ur_kin (fk / ik_numeric / jacobian) is pure numpy.
sys.path.insert(0, os.path.join(_REPO, "ros2_ur_ws", "src", "ur_gello_bringup"))

from ur_gello_bringup.ur_kin import fk  # noqa: E402

from ur_env.envs import ur7e_env as ur7e_env_module  # noqa: E402
from ur_env.envs import wrappers as wrappers_module  # noqa: E402
from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.policy_delta_controller import PolicyDeltaController  # noqa: E402
from ur_env.envs.ur7e_env import UR7eEnv  # noqa: E402
from ur_env.envs.wrappers import (  # noqa: E402
    DeadmanHeartbeatStaleError,
    DeadmanSource,
    GelloIntervention,
)

Q6 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])

#: Real step period measured on the production actor loop (1.95 Hz).  The whole
#: reason this path exists, so it is the number the window tests use.
PRODUCTION_STEP_S = 0.512


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
    """Virtual time for the RL thread's clock, in both modules that read it.

    The FOLLOWER's pacing shell deliberately does not go through these globals
    (``ur7e_env._wall_monotonic`` is bound at import), so patching here cannot
    make a background thread drive the virtual clock — which is exactly why the
    tests that use this fixture also stop that thread.
    """

    virtual = _VirtualClock()
    monkeypatch.setattr(ur7e_env_module, "time", virtual)
    monkeypatch.setattr(wrappers_module, "time", virtual)
    return virtual


class _FakeDeadman(DeadmanSource):
    """Operator signal under direct test control (no keyboard, no topic).

    ``stale=True`` reproduces ``RosTopicDeadman``'s "the heartbeat stopped
    arriving" behaviour, which is an EXCEPTION and never ``False`` — the
    distinction the whole fail-closed path turns on.
    """

    def __init__(self, engaged=True, gain=1.0, stale=False):
        self.engaged = bool(engaged)
        self._gain = float(gain)
        self.stale = bool(stale)
        self.engaged_reads = 0
        self.gain_reads = 0

    def is_engaged(self):
        self.engaged_reads += 1
        if self.stale:
            raise DeadmanHeartbeatStaleError(age_s=0.6, stale_s=0.5)
        return self.engaged

    def gain(self):
        self.gain_reads += 1
        return self._gain


class _FollowBackend:
    """URRosBackend stand-in: an ideal robot plus a scriptable GELLO leader.

    ``dry_run`` is True only so ``go_to_reset`` skips its arrival loop; commands
    are still recorded and still teleport the fake robot.
    """

    dry_run = True

    def __init__(self, clock_source, q_robot=Q6, q_leader=Q6):
        self._clock = clock_source
        self.q = np.asarray(q_robot, dtype=float).copy()
        self.leader_q = np.asarray(q_leader, dtype=float).copy()
        self._leader_rx = clock_source.monotonic()
        self.leader_step = np.zeros(6)
        self.leader_stale = False

        self.lock = threading.Lock()
        self.polls = 0
        self.commands = []
        self.holds = 0
        self.closed_at = None
        #: called as ``on_command(index)`` from inside send_joint_command
        self.on_command = None

    # ---- leader ---- #
    def get_gello_state(self):
        with self.lock:
            self.polls += 1
            if self.leader_stale:
                # Fresh joints, dead clock: exactly the LEADER_STALE_S failure.
                return np.concatenate([self.leader_q, [float("nan")]]), 999.0
            self.leader_q = self.leader_q + self.leader_step
            self._leader_rx = self._clock.monotonic()
            return (
                np.concatenate([self.leader_q, [float("nan")]]),
                self._clock.monotonic() - self._leader_rx,
            )

    # ---- robot state ---- #
    def get_joint_state(self):
        with self.lock:
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
        with self.lock:
            self.commands.append(q_cmd)
            self.q = q_cmd
            index = len(self.commands)
        if self.on_command is not None:
            self.on_command(index)

    def send_gripper_percent(self, fraction):
        pass

    def request_hold(self):
        with self.lock:
            self.holds += 1
            return self.q.copy()

    def reset_command_stream(self):
        pass

    def close(self):
        self.closed_at = {
            "commands": len(self.commands),
        }


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


class _RealClock:
    """``time`` façade for the doubles when the test uses the real clock."""

    @staticmethod
    def monotonic():
        import time as _t

        return _t.monotonic()


def _build(
    clock_source,
    *,
    hz=10.0,
    substep_hz=30.0,
    follow_mode="background",
    leader_step=0.0,
    deadman=None,
    governor=None,
    run_thread=False,
):
    """A reset, engaged GelloIntervention over a real UR7eEnv + fake backend.

    ``run_thread=False`` (the default) STOPS the follower's pacing thread right
    after construction, so the test drives ``_follow_tick(now)`` itself and the
    only thing time can do is what the test tells it to.  The pacing shell is
    covered separately, by the few tests that need a real thread.
    """

    config = _Config()
    config.HZ = float(hz)
    config.INTERVENTION = dict(
        DefaultUR7eEnvConfig.INTERVENTION,
        substep_hz=float(substep_hz),
        follow_mode=follow_mode,
    )
    if governor is not None:
        config.GOVERNOR = dict(DefaultUR7eEnvConfig.GOVERNOR, **governor)

    backend = _FollowBackend(clock_source)
    backend.leader_step = np.full(6, float(leader_step))
    env = UR7eEnv(fake_env=False, config=config, backend=backend)
    if not run_thread and env._follow_thread is not None:
        env._follow_stop.set()
        env._follow_thread.join(2.0)
        assert not env._follow_thread.is_alive()
    wrapper = GelloIntervention(
        env, deadman=deadman if deadman is not None else _FakeDeadman(engaged=True)
    )
    wrapper.reset()
    backend.commands.clear()
    backend.polls = 0
    return wrapper, env, backend


def _zeros_action():
    return np.zeros(7, dtype=np.float32)


def _run_ticks(env, clock_source, count, period=1.0 / 30.0):
    """Drive ``count`` follower ticks off the virtual clock, one per period."""

    results = []
    for _ in range(count):
        results.append(env._follow_tick(clock_source.now))
        clock_source.sleep(period)
    return results


def _overrun_the_step(env, clock_source, seconds):
    """Make every ``env.step`` take ``seconds`` longer than its nominal window.

    Injected in ``_get_obs`` because that is where the in-``step`` half of the
    real cost lives (the camera read), and it runs after the window closed and
    before ``step`` returns — the same side of the loop as the gRPC round trip.
    """

    real_get_obs = env._get_obs

    def slow_get_obs():
        clock_source.sleep(seconds)
        return real_get_obs()

    env._get_obs = slow_get_obs


# =========================================================================== #
# 1. THE MUX: while the human follows, the RL step commands nothing            #
# =========================================================================== #
def test_an_armed_follower_makes_apply_action_issue_no_joint_command(clock):
    wrapper, env, backend = _build(clock, leader_step=0.002)

    wrapper.step(_zeros_action())            # engage -> arm
    assert env.intervention_follow_armed()
    assert env.command_owner() == env.OWNER_HUMAN

    issued_before = len(backend.commands)
    policy_action = np.zeros(7, dtype=np.float32)
    policy_action[0] = 1.0                   # a full-scale policy demand
    info = env._apply_action(policy_action)

    assert len(backend.commands) == issued_before, (
        "the RL thread commanded the arm while the human owned it"
    )
    assert info["arm_command_owner"] == env.OWNER_HUMAN
    # ...and nothing moved in the controller either: a blocked emit must not
    # advance T_cmd, or the human's anchor would drift under them.
    assert info["held"] is False


def test_the_gripper_channel_still_runs_while_the_follower_owns_the_arm(clock):
    """The follower drives the ARM only; the gripper stays a per-step decision.

    The Robotiq needs ~0.5 s per actuation and is debounced accordingly, so a
    per-tick gripper decision would be meaningless.  Blocking the arm command
    must therefore not block the gripper command that travels with it.
    """

    wrapper, env, backend = _build(clock, leader_step=0.002)
    wrapper.step(_zeros_action())

    sent = []
    env.backend.send_gripper_percent = lambda value: sent.append(float(value))
    action = np.zeros(7, dtype=np.float32)
    action[6] = -1.0                          # CLOSE
    env._apply_action(action)

    assert sent == [1.0], "0.0=OPEN..1.0=CLOSED; a close must still be sent"


def test_a_policy_step_owns_the_arm_again_the_moment_the_human_disengages(clock):
    deadman = _FakeDeadman(engaged=True)
    wrapper, env, backend = _build(clock, leader_step=0.002, deadman=deadman)
    wrapper.step(_zeros_action())
    assert env.command_owner() == env.OWNER_HUMAN

    deadman.engaged = False
    issued_before = len(backend.commands)
    _, _, _, _, info = wrapper.step(_zeros_action())

    assert env.command_owner() == env.OWNER_POLICY
    assert not env.intervention_follow_armed()
    assert len(backend.commands) == issued_before + 1  # the policy target
    assert info["intervened"] == 0
    assert "intervene_action" not in info


def test_send_joint_command_only_ever_receives_a_controller_step_result(clock):
    """No path may compute a joint vector some other way.

    The workspace box, the governor, the branch-continuous IK seed and the
    joint-step line search all live inside ``controller.step``.  A command that
    did not come out of it has been through none of them.
    """

    wrapper, env, backend = _build(clock, leader_step=0.004)
    returned = []
    real_step = env.controller.step

    def spy(xi, dt=None):
        q_cmd, info = real_step(xi, dt)
        returned.append(np.asarray(q_cmd, dtype=float).copy())
        return q_cmd, info

    env.controller.step = spy

    wrapper.step(_zeros_action())             # engage / arm
    _run_ticks(env, clock, 6)
    wrapper.step(_zeros_action())

    assert len(backend.commands) >= 6
    assert len(backend.commands) == len(returned)
    for published, produced in zip(backend.commands, returned):
        np.testing.assert_array_equal(published, produced)


# =========================================================================== #
# 2. THE REGRESSION: following does not stop when env.step's window closes      #
# =========================================================================== #
def test_the_follower_keeps_refreshing_the_target_outside_the_step_window(clock):
    """THE point of the change, at the measured production step period.

    A 512 ms step contains 15 ticks of a 30 Hz follower and only 3 of them fit
    inside the nominal 100 ms window.  The in-window path issues those 3 and then
    leaves the target untouched for 412 ms — long past the upsampler's 0.30 s
    ``target_stale_s``, which is what brakes the stream to HOLD once per window
    and re-arms ``soft_start_s`` on top of it.
    """

    wrapper, env, backend = _build(clock, leader_step=0.002)
    _overrun_the_step(env, clock, PRODUCTION_STEP_S - 1.0 / 10.0)

    wrapper.step(_zeros_action())             # engage / arm
    issued_after_step = len(backend.commands)

    # The 12 ticks that land AFTER the nominal window would have closed.
    results = _run_ticks(env, clock, 15)[3:]

    assert all(r["issued"] for r in results), [r["reason"] for r in results]
    assert len(backend.commands) - issued_after_step == 15
    # ...and the gap between two consecutive target refreshes never reaches the
    # upsampler's stale deadline, which is the property the operator feels.
    assert 1.0 / env.intervention_substep_hz < env.config.UPSAMPLER["target_stale_s"]


def test_a_long_window_keeps_following_past_one_action_scale_step(clock):
    """NO per-window displacement budget.  Operator decision, 2026-07-30.

    The budget capped a window at one ``ACTION_SCALE`` step so the stored action
    could never understate the motion — a RECORDING constraint enforced by
    throttling the ARM, which at the 512 ms production step set the operator's
    ceiling at 2.4 cm/s against 12.4 cm/s on the validation rig.  An intervention
    path nobody can drive produces no demonstrations, so the cap is gone and the
    breach is REPORTED instead (see the saturation tests below).
    """

    wrapper, env, backend = _build(clock, leader_step=0.01)
    wrapper.step(_zeros_action())             # engage / arm

    p_before = env.controller.tcp_cmd()[:3, 3].copy()
    moved = []
    for _ in range(15):                       # one 512 ms window at 30 Hz
        before = env.controller.tcp_cmd()[:3, 3].copy()
        env._follow_tick(clock.now)
        clock.sleep(1.0 / 30.0)
        moved.append(
            float(np.linalg.norm(env.controller.tcp_cmd()[:3, 3] - before))
        )

    # Every tick moves, including the last third — a budgeted window would be
    # commanding zeros by tick 3.
    assert min(moved) > 0.0, moved
    travelled = float(
        np.linalg.norm(env.controller.tcp_cmd()[:3, 3] - p_before)
    )
    assert travelled > env.action_scale[0], (
        "the window is still capped at one ACTION_SCALE step"
    )


def test_the_governor_is_the_speed_limit_once_the_budget_is_gone(clock):
    """With no budget, ``v_max`` is the ONLY thing bounding the follower.

    Per tick: ``v_max * dt`` = 0.15 / 30 = 5 mm.  That has to hold for a leader
    demanding arbitrarily more, on every tick, no matter how long the window —
    it is what turns "a 5 s gRPC stall" into a bounded 75 cm rather than an
    unbounded lunge, and the workspace box (next test) is what bounds it further.
    """

    wrapper, env, backend = _build(clock, leader_step=0.2)    # absurdly fast
    wrapper.step(_zeros_action())             # engage / anchor

    cap = env.controller.v_max / env.intervention_substep_hz
    steps = []
    for _ in range(30):
        before = env.controller.tcp_cmd()[:3, 3].copy()
        env._follow_tick(clock.now)
        clock.sleep(1.0 / 30.0)
        steps.append(
            float(np.linalg.norm(env.controller.tcp_cmd()[:3, 3] - before))
        )

    assert max(steps) <= cap * (1.0 + 1e-6), max(steps)
    # Not vacuous: the cap is actually BINDING here, so this is measuring the
    # governor and not merely a slow leader.
    assert max(steps) == pytest.approx(cap, rel=1e-3)


def test_the_workspace_box_still_binds_on_every_follower_tick(clock):
    """The box lives inside ``controller.step``; the follower must go through it.

    With the displacement budget gone the box is the only POSITIONAL bound on an
    intervention, so a follower tick that computed a joint command any other way
    would have no wall at all.  It is also written back into the controller's
    integrator, so it is a hard stop rather than a windup the next reversal has
    to unwind.
    """

    config = _Config()
    config.INTERVENTION = dict(
        DefaultUR7eEnvConfig.INTERVENTION, substep_hz=30.0, follow_mode="background"
    )
    home = fk(Q6)[:3, 3]
    # A 1 cm wall in +x, everything else wide open.  Index 3 is |rx|.
    config.ABS_POSE_LIMIT_LOW = np.array([home[0] - 1.0, -2.0, -2.0, 0.0, -4.0, -4.0])
    config.ABS_POSE_LIMIT_HIGH = np.array(
        [home[0] + 0.01, 2.0, 2.0, 4.0, 4.0, 4.0]
    )

    backend = _FollowBackend(clock)
    backend.leader_step = np.zeros(6)
    env = UR7eEnv(fake_env=False, config=config, backend=backend)
    env._follow_stop.set()
    env._follow_thread.join(2.0)
    assert env._safety_box_active, "the box must be armed or this proves nothing"

    wrapper = GelloIntervention(env, deadman=_FakeDeadman(engaged=True))
    wrapper.reset()
    wrapper.step(_zeros_action())             # engage / arm

    # A source that drives straight at the wall, forever, with no budget.
    env._follow_source = _ScriptedSource(np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]))
    results = _run_ticks(env, clock, 20)

    x = float(env.controller.tcp_cmd()[0, 3])
    assert x <= config.ABS_POSE_LIMIT_HIGH[0] + 1e-9, x
    assert any(r.get("info", {}).get("clipped") for r in results), (
        "the box never reported a clamp — the follower is bypassing clip_pose"
    )
    _, _, _, _, info = env.step(_zeros_action())
    assert info["clipped"] is True, "a clamped follower tick must reach the log"
    env.close()


# =========================================================================== #
# 3. SAFETY: the deadman is re-read on EVERY tick                              #
# =========================================================================== #
def test_a_deadman_release_stops_the_arm_on_the_very_next_tick(clock):
    """33 ms, not 512 ms.  This is the change's central safety claim.

    Under in-window following a release only took effect at the next window
    boundary, so the arm kept going for up to one REAL step period.
    """

    deadman = _FakeDeadman(engaged=True)
    wrapper, env, backend = _build(clock, leader_step=0.002, deadman=deadman)
    wrapper.step(_zeros_action())
    _run_ticks(env, clock, 2)
    issued = len(backend.commands)

    deadman.engaged = False
    result = env._follow_tick(clock.now)

    assert result["reason"] == "DISENGAGED"
    assert len(backend.commands) == issued, "a released deadman still commanded"
    assert not env.intervention_follow_armed()
    assert env.command_owner() == env.OWNER_POLICY
    # A release is an AUTHORIZED hand-back, so it disarms without braking: a
    # brake here would put a HOLD + soft-start re-arm on every normal release.
    assert backend.holds == 0
    # Every later tick is inert, not merely the first.
    assert _run_ticks(env, clock, 3)[-1]["reason"] == "DISARMED"
    assert len(backend.commands) == issued


def test_the_deadman_is_read_once_per_tick_and_the_gain_is_never_re_read(clock):
    """Per-tick deadman, engage-edge gain.  Both halves matter.

    Re-reading ``gain()`` per tick would let a slider nudged mid-motion rescale
    an in-progress anchored delta 30 times a second — the discontinuity the
    engage-edge latch exists to remove, reintroduced faster.
    """

    deadman = _FakeDeadman(engaged=True)
    wrapper, env, _ = _build(clock, leader_step=0.002, deadman=deadman)
    wrapper.step(_zeros_action())

    reads_before, gain_before = deadman.engaged_reads, deadman.gain_reads
    _run_ticks(env, clock, 5)

    assert deadman.engaged_reads - reads_before == 5
    assert deadman.gain_reads == gain_before


def test_a_stale_heartbeat_brakes_once_and_is_re_raised_on_the_rl_thread(clock):
    """``is_engaged`` RAISING is not ``is_engaged`` returning False.

    A lost publisher authorizes nothing, so it may not be read as a release.
    The follower stops the arm on its own thread and leaves the exception where
    the RL thread will find it — a daemon thread's traceback reaches nobody.
    """

    deadman = _FakeDeadman(engaged=True)
    wrapper, env, backend = _build(clock, leader_step=0.002, deadman=deadman)
    wrapper.step(_zeros_action())
    issued = len(backend.commands)

    deadman.stale = True
    result = env._follow_tick(clock.now)

    assert result["reason"] == "DEADMAN_FAULT"
    assert len(backend.commands) == issued
    assert backend.holds == 1                 # braked to the stop point
    assert not env.intervention_follow_armed()
    assert env.command_owner() == env.OWNER_POLICY

    # A second tick must not brake again — it is disarmed, not retrying.
    env._follow_tick(clock.now)
    assert backend.holds == 1

    with pytest.raises(DeadmanHeartbeatStaleError):
        env.step(_zeros_action())
    # LATCHED: it is a stop condition, so it does not evaporate on the next call.
    with pytest.raises(DeadmanHeartbeatStaleError):
        env.step(_zeros_action())
    with pytest.raises(DeadmanHeartbeatStaleError):
        env.reset()
    assert isinstance(env.clear_follow_fault(), DeadmanHeartbeatStaleError)
    assert env.clear_follow_fault() is None


def test_a_latched_fault_refuses_to_hand_the_arm_back_to_the_human(clock):
    deadman = _FakeDeadman(engaged=True, stale=True)
    wrapper, env, _ = _build(clock, leader_step=0.002, deadman=deadman)
    env._latch_follow_fault(DeadmanHeartbeatStaleError(age_s=0.6, stale_s=0.5))

    with pytest.raises(DeadmanHeartbeatStaleError):
        env.arm_intervention_follow(wrapper)
    assert not env.intervention_follow_armed()


def test_a_leader_that_goes_stale_mid_window_brakes_and_asks_to_re_anchor(clock):
    wrapper, env, backend = _build(clock, leader_step=0.004)
    wrapper.step(_zeros_action())
    _run_ticks(env, clock, 2)
    issued = len(backend.commands)

    backend.leader_stale = True
    result = env._follow_tick(clock.now)

    assert result["reason"] == "GELLO_STALE"
    assert len(backend.commands) == issued
    assert backend.holds == 1
    assert not env.intervention_follow_armed()
    # A dropout is an ordinary event, NOT a fault: no exception is latched.
    assert env._follow_fault is None

    # The RL thread learns about it BEFORE it decides anything: the anchor is
    # dropped at the top of step(), action() re-engages on the braking endpoint,
    # and the first delta of the new anchor is exactly zero.  Learning about it
    # only from the returned info would arm the follower for a whole window on
    # an anchor whose first command is the entire braking distance.
    assert env._follow_needs_reanchor is True
    backend.leader_stale = False
    engages_before = wrapper.expert.deadman.gain_reads
    _, _, _, _, info = wrapper.step(_zeros_action())

    assert info["reject_reason"] == "GELLO_STALE"
    assert info["held"] is True
    assert wrapper._anchored is True
    assert "intervention_window_braked" not in info
    # THE discriminator: the gain is latched exactly once per engage edge, so a
    # fresh read is proof this step re-anchored rather than reusing the anchor
    # the brake invalidated.
    assert wrapper.expert.deadman.gain_reads == engages_before + 1
    # ...and the request was consumed, so the NEXT step does not re-anchor again.
    assert env.consume_follow_reanchor() is False


def test_an_unanchored_wrapper_disarms_the_follower_quietly(clock):
    """A policy hand-back must not look like a leader failure.

    ``_disengage`` on the RL thread leaves the follower with nothing to follow.
    Reporting that as a stale leader would put a HOLD — and the soft-start re-arm
    that follows it — on every single normal release.
    """

    wrapper, env, backend = _build(clock, leader_step=0.002)
    wrapper.step(_zeros_action())
    wrapper._disengage()

    result = env._follow_tick(clock.now)

    assert result["reason"] == "DISENGAGED"
    assert backend.holds == 0
    assert not env.intervention_follow_armed()


def test_a_disengage_cannot_interleave_into_the_followers_anchor_maths():
    """``_disengage`` must never be observable half-done.

    It clears ``_anchored`` and then both anchors, and ``follow_xi`` checks the
    anchor and then USES it (``_expert_delta_xi`` re-reads it).  Unlocked, a
    disengage landing between those two produces ``None[:3, :3]`` inside
    ``_expert_delta_xi``.  On a daemon thread that TypeError goes nowhere the
    operator can see: the arm just stops following, and the only trace is a
    latched fault the next ``env.step`` re-raises as something unrelated-looking.

    Forced rather than hoped for: the follower is held inside its own anchor
    maths and the disengage is issued from another thread, which must BLOCK.
    """

    import time as real_time

    wrapper, env, _ = _build(_RealClock(), leader_step=0.002)
    wrapper.step(_zeros_action())             # engage / arm (thread stopped)
    assert env.intervention_follow_armed()

    entered = threading.Event()
    release = threading.Event()
    real_leader_T = wrapper._leader_T

    def blocking_leader_T(q_lead):
        entered.set()
        release.wait(5.0)
        return real_leader_T(q_lead)

    wrapper._leader_T = blocking_leader_T
    ticker = threading.Thread(
        target=lambda: env._follow_tick(real_time.monotonic()), daemon=True
    )
    ticker.start()
    assert entered.wait(5.0), "the follower never reached the anchor maths"

    done = threading.Event()
    disengager = threading.Thread(
        target=lambda: (wrapper._disengage(), done.set()), daemon=True
    )
    disengager.start()
    # Disproof window: the disengage must not be able to land in the middle.
    assert not done.wait(0.2), (
        "_disengage ran while the follower was inside its anchor maths — the "
        "follow lock does not span follow_xi"
    )

    release.set()
    ticker.join(5.0)
    disengager.join(5.0)
    assert done.is_set()
    assert env._follow_fault is None, env._follow_fault


# =========================================================================== #
# 4. THE RECORD: what the window committed, and when the clip binds            #
# =========================================================================== #
class _ScriptedSource:
    """The follower protocol with a scripted xi, so the record can be forced.

    Deliberately NOT budgeted: the recording clip is a canary for the budget
    having failed, and the only way to test a canary is to make the thing it
    watches for happen.
    """

    def __init__(self, xi, engaged=True):
        self.xi = np.asarray(xi, dtype=float)
        self.engaged = bool(engaged)

    def follow_is_engaged(self):
        return self.engaged

    def follow_xi(self, now):
        return self.xi.copy()


def test_the_recorded_action_is_the_displacement_the_window_committed(clock):
    wrapper, env, _ = _build(clock, leader_step=0.004)
    wrapper.step(_zeros_action())             # engage / arm

    p_before = env.controller.tcp_cmd()[:3, 3].copy()
    _run_ticks(env, clock, 3)
    _, _, _, _, info = wrapper.step(_zeros_action())
    commanded = env.controller.tcp_cmd()[:3, 3] - p_before

    recorded = np.asarray(info["intervene_action"], dtype=float)
    assert recorded.shape == (7,)
    # PolicyDeltaController integrates position as a plain vector sum, so the
    # two are equal by construction whenever nothing was cut.
    np.testing.assert_allclose(
        recorded[:3] * env.action_scale[0], commanded, rtol=1e-6, atol=1e-9
    )
    assert np.all(np.abs(recorded) <= 1.0)
    assert info["intervention_saturated"] is False
    assert info["intervention_saturation"] <= 1.0
    assert info["intervention_follow_ticks"] == 3
    assert info["intervened"] == 1


def test_a_window_inside_action_scale_reports_no_saturation(clock):
    wrapper, env, _ = _build(clock)
    # Half an ACTION_SCALE step, split over two ticks.
    env._follow_source = _ScriptedSource(
        np.array([env.action_scale[0] * 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    env._command_owner = env.OWNER_HUMAN
    env._follow_armed.set()

    _run_ticks(env, clock, 2)
    _, _, _, _, info = env.step(_zeros_action())

    assert info["intervention_saturated"] is False
    assert info["intervention_saturation"] == pytest.approx(0.5, rel=1e-6)
    committed = np.asarray(info["intervention_committed_action"], dtype=float)
    assert committed[0] == pytest.approx(0.5, rel=1e-6)


def test_a_window_past_action_scale_is_clipped_and_says_so(clock):
    """The clip is a report of a broken invariant, never a silent correction.

    A stored action that saturates while the arm kept moving is the buffer
    UNDER-reporting the motion, which is the one thing the stored-action
    invariant forbids.  It has to be visible in the transition's info.
    """

    wrapper, env, _ = _build(clock)
    env._follow_source = _ScriptedSource(
        np.array([env.action_scale[0] * 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    env._command_owner = env.OWNER_HUMAN
    env._follow_armed.set()

    _run_ticks(env, clock, 3)                 # 2.7x ACTION_SCALE requested
    _, _, _, _, info = env.step(_zeros_action())

    committed = np.asarray(info["intervention_committed_action"], dtype=float)
    assert info["intervention_saturated"] is True
    assert info["intervention_saturation"] > 1.0
    assert committed[0] == pytest.approx(1.0, rel=1e-9)
    assert np.all(np.abs(committed) <= 1.0), "a stored action must stay legal"


def test_the_record_covers_the_gap_between_two_env_steps_not_just_the_window(clock):
    """The window is boundary-to-boundary: obs_k to obs_{k+1}.

    The follower is still driving during the actor's gRPC round trip, and that
    motion happened between the two observations of THIS transition.  Attributing
    it anywhere else would break (obs_k, action_k, obs_{k+1}) alignment.
    """

    wrapper, env, _ = _build(clock, leader_step=0.004)
    wrapper.step(_zeros_action())             # engage / arm

    p_before = env.controller.tcp_cmd()[:3, 3].copy()
    _run_ticks(env, clock, 4)                 # "outside env.step" following
    _, _, _, _, info = wrapper.step(_zeros_action())
    commanded = env.controller.tcp_cmd()[:3, 3] - p_before

    recorded = np.asarray(info["intervene_action"], dtype=float)[:3]
    np.testing.assert_allclose(
        recorded * env.action_scale[0], commanded, rtol=1e-6, atol=1e-9
    )
    assert info["intervention_follow_ticks"] == 4


def test_a_policy_transition_never_acquires_the_follower_record(clock):
    deadman = _FakeDeadman(engaged=False)
    wrapper, env, _ = _build(clock, leader_step=0.002, deadman=deadman)

    _, _, _, _, info = wrapper.step(_zeros_action())

    assert "intervene_action" not in info
    assert "intervention_committed_action" not in info
    assert "intervention_saturated" not in info
    assert env._follow_thread is None or not env.intervention_follow_armed()


def test_the_gripper_axis_of_the_record_is_the_windows_single_decision(clock):
    wrapper, env, _ = _build(clock, leader_step=0.002)
    wrapper._window_gripper = 0.0
    wrapper.step(_zeros_action())
    wrapper._grip_cmd = -1.0                  # operator squeezed the trigger
    wrapper._expert_gripper = lambda grip: -1.0

    _run_ticks(env, clock, 2)
    _, _, _, _, info = wrapper.step(_zeros_action())

    assert float(np.asarray(info["intervene_action"])[6]) == -1.0


# =========================================================================== #
# 5. RESET / go_to_reset: the follower must be PROVEN stopped                   #
# =========================================================================== #
def test_go_to_reset_refuses_while_the_follower_owns_the_arm(clock):
    wrapper, env, _ = _build(clock, leader_step=0.002)
    wrapper.step(_zeros_action())
    assert env.command_owner() == env.OWNER_HUMAN

    with pytest.raises(RuntimeError, match="POLICY command ownership"):
        env.go_to_reset()


def test_reset_disarms_the_follower_before_it_streams_the_home_target(clock):
    """go_to_reset republishes at 20 Hz; the follower publishes at 30 Hz.

    The backend is last-write-wins, so a live follower simply WINS: the arm never
    converges to ``RESET_TOLERANCE_RAD`` and the operator gets ten seconds of
    blind motion followed by "reset did not arrive".
    """

    wrapper, env, _ = _build(clock, leader_step=0.002)
    wrapper.step(_zeros_action())
    assert env.intervention_follow_armed()

    owner_seen = []
    real_go_to_reset = env.go_to_reset

    def spy(**kwargs):
        owner_seen.append((env.command_owner(), env.intervention_follow_armed()))
        return real_go_to_reset(**kwargs)

    env.go_to_reset = spy
    wrapper.reset()

    assert owner_seen == [(env.OWNER_POLICY, False)]
    assert env._follow_source is None


def test_reset_refuses_when_the_follower_will_not_confirm_it_stopped(clock):
    """A disarm REQUEST is not proof.  reset() waits for the acknowledgement.

    Simulated with a live thread that never parks, which is what a wedged
    follower looks like from here.
    """

    wrapper, env, _ = _build(clock, leader_step=0.002)
    wrapper.step(_zeros_action())

    wedged = threading.Event()
    thread = threading.Thread(target=lambda: wedged.wait(10.0), daemon=True)
    thread.start()
    env._follow_thread = thread
    env._follow_idle.clear()
    env.FOLLOW_QUIESCE_TIMEOUT_S = 0.05       # keep the test quick

    try:
        with pytest.raises(RuntimeError, match="did not confirm it stopped"):
            env.reset()
    finally:
        wedged.set()
        thread.join(2.0)


def test_suspend_follower_is_available_for_the_operator_gates(clock):
    """Exposed, not wired: ``remote_actor``'s gates are another session's file.

    ``WAIT_SCENE_READY`` / ``WAIT_HOME_APPROVAL`` block the RL thread
    indefinitely and the latter does not consult the deadman at all, so a
    follower left armed across one keeps driving for as long as the operator
    takes to answer.
    """

    wrapper, env, _ = _build(clock, leader_step=0.002)
    wrapper.step(_zeros_action())
    assert env.intervention_follow_armed()

    with env.suspend_follower():
        assert not env.intervention_follow_armed()
        assert env.command_owner() == env.OWNER_POLICY


# =========================================================================== #
# 6. follow_mode="in_window" preserves the pre-change path exactly              #
# =========================================================================== #
def test_in_window_mode_runs_no_thread_and_keeps_the_substep_driver(clock):
    wrapper, env, backend = _build(
        clock, follow_mode="in_window", leader_step=0.002
    )

    assert env._follow_thread is None
    assert env.intervention_follow_mode == "in_window"
    assert wrapper._follow_mode == "in_window"

    _, _, _, _, info = wrapper.step(_zeros_action())

    assert info["intervention_substeps"] == 2   # 30 Hz inside a 10 Hz window
    assert len(backend.commands) == 3
    assert "intervention_committed_action" not in info
    assert "intervention_saturated" not in info
    assert env.command_owner() == env.OWNER_POLICY


def test_a_config_without_an_intervention_block_forces_in_window(clock):
    config = _Config()
    config.INTERVENTION = {}
    backend = _FollowBackend(clock)
    env = UR7eEnv(fake_env=False, config=config, backend=backend)
    try:
        assert env.intervention_follow_mode == "in_window"
        assert env._follow_thread is None
        assert env._follow_dt is None
    finally:
        env.close()


def test_an_unknown_follow_mode_is_refused_at_construction(clock):
    config = _Config()
    config.INTERVENTION = dict(
        DefaultUR7eEnvConfig.INTERVENTION, follow_mode="whatever"
    )
    with pytest.raises(ValueError, match="follow_mode"):
        UR7eEnv(fake_env=False, config=config, backend=_FollowBackend(clock))


def test_the_shipped_default_is_background():
    assert DefaultUR7eEnvConfig.INTERVENTION["follow_mode"] == "background"


# =========================================================================== #
# 7. REAL THREADS: the properties that only exist because there is one         #
# =========================================================================== #
def test_the_follower_thread_is_a_daemon():
    """A non-daemon follower makes ``wait "$ACTOR_PID"`` block forever.

    ``run_hil_actor.sh`` traps on the actor exiting to restore the trajectory
    controller; if the process never exits, the arm is left on
    forward_position_controller permanently.
    """

    wrapper, env, _ = _build(_RealClock(), run_thread=True)
    try:
        assert env._follow_thread is not None
        assert env._follow_thread.daemon is True
        assert env._follow_thread.is_alive()
    finally:
        env.close()


def test_close_stops_the_follower_before_it_closes_the_backend():
    """Ordering, and a bounded one.

    ``URRosBackend.close`` SKIPS ``destroy_node()`` when its joins time out, and
    ``hil_restore_controller_after_actor`` then refuses the controller switch
    with "still has N publisher(s)" — the arm stays on FPC.  So the follower has
    to be gone BEFORE the backend teardown starts, and quickly.
    """

    import time as real_time

    wrapper, env, backend = _build(_RealClock(), run_thread=True)
    thread = env._follow_thread
    observed = {}

    real_close = backend.close
    observed["alive"] = []

    def close_spy():
        # EVERY call, not the last: a teardown that closed the backend first and
        # then closed it again after the join would otherwise look correct.
        observed["alive"].append(thread.is_alive())
        return real_close()

    backend.close = close_spy

    started = real_time.monotonic()
    env.close()
    elapsed = real_time.monotonic() - started

    assert observed["alive"], "the backend was never closed"
    assert not any(observed["alive"]), (
        "the backend was torn down while the follower was still alive"
    )
    assert not thread.is_alive()
    assert elapsed < env.FOLLOW_JOIN_TIMEOUT_S + 1.0
    assert env.FOLLOW_JOIN_TIMEOUT_S + env.FOLLOW_QUIESCE_TIMEOUT_S <= 3.0


def test_a_disarm_that_lands_mid_command_still_ends_the_following():
    """WORST INTERLEAVING, forced: disarm while a tick is inside the emit.

    The follower is held inside ``send_joint_command`` — past the ownership test,
    with the command lock held — and the disarm is issued from another thread.
    It must BLOCK there (proving the lock really spans the emit) and, once it
    completes, no further command may appear.
    """

    import time as real_time

    wrapper, env, backend = _build(
        _RealClock(), leader_step=0.002, run_thread=True
    )
    inside = threading.Event()
    release = threading.Event()
    try:
        # Arm FIRST, with no block installed: the RL thread must never be the one
        # that stalls, or env.step's own harvest would be waiting on the same
        # lock and the interleaving under test would never happen.
        wrapper.step(_zeros_action())         # engage / arm; the thread follows
        assert env.intervention_follow_armed()

        blocked_once = threading.Event()

        def block_the_next_command(index):
            if blocked_once.is_set():
                return
            blocked_once.set()
            inside.set()
            release.wait(5.0)

        backend.on_command = block_the_next_command
        assert inside.wait(5.0), "the follower never reached the emit"
        disarmed = threading.Event()
        disarmer = threading.Thread(
            target=lambda: (env.disarm_intervention_follow(), disarmed.set()),
            daemon=True,
        )
        disarmer.start()
        # It must NOT be able to complete while the emit holds the lock.  This
        # sleep is a DISPROOF window, not a wait-for-a-state.
        assert not disarmed.wait(0.2), (
            "disarm returned while a command was still in flight — the mux lock "
            "does not span the emit"
        )

        release.set()
        assert disarmed.wait(5.0)
        assert env.await_follower_quiescent()
        settled = len(backend.commands)
        real_time.sleep(5.0 / env.intervention_substep_hz)   # disproof window
        assert len(backend.commands) == settled
        assert env.command_owner() == env.OWNER_POLICY
    finally:
        release.set()
        backend.on_command = None
        env.close()


def test_the_running_follower_drives_the_arm_while_the_rl_thread_is_blocked():
    """END TO END, with the real thread: the arm does not wait for env.step.

    ``_get_obs`` blocks until the follower has published well past the nominal
    window, so the assertion cannot pass unless the follower is genuinely running
    outside it.  Event-driven: an in-window follower makes this TIME OUT rather
    than merely produce a smaller number.
    """

    wrapper, env, backend = _build(
        _RealClock(), leader_step=0.002, run_thread=True
    )
    try:
        target = 12
        reached = threading.Event()

        def note(index):
            if index >= target:
                reached.set()

        backend.on_command = note

        real_get_obs = env._get_obs

        def blocking_get_obs():
            reached.wait(5.0)
            return real_get_obs()

        env._get_obs = blocking_get_obs
        wrapper.step(_zeros_action())         # engage / arm

        assert reached.is_set(), (
            "the follower did not publish beyond the nominal window while the "
            "RL thread was blocked"
        )
        assert len(backend.commands) >= target
    finally:
        backend.on_command = None
        env.close()


# =========================================================================== #
# 8. PolicyDeltaController is itself thread safe (the reproduced defects)      #
# =========================================================================== #
def _controller():
    ctrl = PolicyDeltaController(dict(DefaultUR7eEnvConfig.GOVERNOR), 10.0)
    ctrl.reset(Q6)
    return ctrl


def test_a_hold_re_issues_what_this_controller_last_published(clock):
    """Reproduced: 25 of 130,028 HOLD ticks PUBLISHED MOTION, up to 0.2047 rad.

    ``_hold`` returned ``self.q_cmd``, a field the other caller had meanwhile
    overwritten.  The GUI shows HOLD while the arm moves — the worst failure this
    class can have.
    """

    ctrl = _controller()
    xi = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_issued, info = ctrl.step(xi)
    assert info["held"] is False

    ctrl.q_cmd = q_issued + 0.2               # the other thread, mid-flight
    q_hold, hold_info = ctrl.step(np.full(6, np.nan))

    assert hold_info["held"] is True
    assert hold_info["reject_reason"] == "BAD_INPUT"
    np.testing.assert_array_equal(q_hold, q_issued)
    np.testing.assert_array_equal(ctrl.joint_cmd(), q_issued)


def test_the_command_pair_is_replaced_atomically(clock):
    """Reproduced: (T_cmd, q_cmd) DISAGREED on 11.1% of 108,949 samples.

    They were two separate stores, so IK could be seeded with one tick's q
    against another tick's T.  Branch continuity is exactly what that seed
    provides; losing it puts the wrist a full turn away, and the 250 Hz upsampler
    rate-limits without vetoing — it executes it as a ~10 s blind sweep.
    """

    entered = threading.Event()
    release = threading.Event()

    def blocking_clip(T):
        entered.set()
        release.wait(5.0)
        return T, False

    ctrl = PolicyDeltaController(
        dict(DefaultUR7eEnvConfig.GOVERNOR), 10.0, clip_pose=blocking_clip
    )
    ctrl.reset(Q6)

    stepper = threading.Thread(
        target=lambda: ctrl.step(np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])),
        daemon=True,
    )
    stepper.start()
    assert entered.wait(5.0)

    read_done = threading.Event()
    reader = threading.Thread(
        target=lambda: (ctrl.tcp_cmd(), read_done.set()), daemon=True
    )
    reader.start()
    # Disproof window: a reader must not be able to observe the command state
    # while a step is halfway through replacing it.
    assert not read_done.wait(0.2), "tcp_cmd() read a half-updated command state"

    release.set()
    stepper.join(5.0)
    reader.join(5.0)
    assert read_done.is_set()
    # fk(q_cmd) and T_cmd describe the same tick, up to the IK residual.
    np.testing.assert_allclose(fk(ctrl.q_cmd)[:3, 3], ctrl.T_cmd[:3, 3], atol=1e-4)


def test_two_threads_stepping_the_controller_keep_the_joint_gate(clock):
    """Reproduced: the ``dq_step_max`` gate leaked on 6.08% of ticks.

    The gate is measured against ``self.q_cmd``; with two callers that field was
    replaced between the measurement and the commit.  Under the lock every
    published command is within one (dt-scaled) joint step of the previous one.
    """

    ctrl = _controller()
    xi = np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0])
    published = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait(5.0)
        for _ in range(60):
            q_cmd, _ = ctrl.step(xi, dt=1.0 / 30.0)
            with lock:
                published.append(np.asarray(q_cmd, dtype=float).copy())

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20.0)
        assert not thread.is_alive()

    budget = ctrl.dq_step_max * (1.0 / 30.0) / ctrl.dt
    steps = [
        float(np.max(np.abs(published[i + 1] - published[i])))
        for i in range(len(published) - 1)
    ]
    assert max(steps) <= budget * (1.0 + 1e-6), max(steps)
