"""The gripper must be OPEN when an episode starts.

WHY THIS FILE EXISTS
--------------------
Neither ``UR7eEnv.reset()`` nor ``go_to_reset()`` used to touch the gripper, so
an episode that ended with the policy holding something started the NEXT episode
still closed.  Every offline demo starts from an open gripper, and
``gripper_position`` is ``state[0]`` — the first element the encoder sees — so a
closed start is out of distribution before the policy has acted once.

The reset open is deliberately NOT the policy's ``_send_gripper_command``
channel (3-state, ``GRIPPER_SLEEP`` debounced, skips when already there).  A
reset is not a policy action: it commands unconditionally and then CONFIRMS,
because the first observation of the episode is read immediately afterwards.

ROS-free and robot-free: a fake backend stands in for URRosBackend, and pynput
is blocked so running the suite never installs a global keyboard listener.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..", "..")
sys.path.insert(0, os.path.join(_HERE, ".."))
# ur_gello_bringup.ur_kin (fk / ik_numeric) is pure numpy — no ROS needed.
sys.path.insert(0, os.path.join(_REPO, "ros2_ur_ws", "src", "ur_gello_bringup"))

from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.ur7e_env import UR7eEnv  # noqa: E402

Q6 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])


@pytest.fixture(autouse=True)
def _block_pynput(monkeypatch):
    """``UR7eEnv.__init__`` builds an ESC listener inside a bare except."""

    monkeypatch.setitem(sys.modules, "pynput", None)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", None)


class _GripperBackend:
    """Fake backend whose gripper answers whatever the test scripts.

    ``dry_run`` is True only so ``go_to_reset`` skips its arrival loop; the
    gripper path is unaffected by it.
    """

    dry_run = True

    def __init__(self, start_percent=1.0, opens=True, drop_first=0):
        self.q = Q6.copy()
        #: 0.0 = OPEN .. 1.0 = CLOSED (robotiq_gripper_modbus_node contract)
        self.gripper = float(start_percent)
        #: False = a stuck gripper that never reports open
        self.opens = bool(opens)
        #: How many leading commands vanish before one takes effect. This is
        #: the measured defect, not a hypothetical: DDS drops a datagram sent
        #: before the subscription matched, and the Robotiq driver's rate
        #: limiter discards a setpoint that lands inside command_min_period
        #: *without* recording that it did — so the command is gone and every
        #: downstream view stays self-consistent about the stale position.
        self.drop_first = int(drop_first)
        self.gripper_commands = []
        self.commands = []

    # ---- gripper ---- #
    def send_gripper_percent(self, fraction):
        self.gripper_commands.append(float(fraction))
        if self.drop_first > 0:
            self.drop_first -= 1
            return
        if self.opens:
            self.gripper = float(fraction)

    def get_gripper_percent(self):
        return self.gripper, 0.0

    # ---- robot ---- #
    def get_joint_state(self):
        return self.q.copy(), np.zeros(6), 0.0

    def send_joint_command(self, q_cmd):
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(6).copy()
        self.commands.append(q_cmd)
        self.q = q_cmd

    def get_gello_state(self):
        return None, float("inf")

    def get_wrench(self):
        return None, float("inf")

    def get_tcp_pose(self):
        return None, float("inf")

    def request_hold(self):
        return self.q.copy()

    def reset_command_stream(self):
        pass

    def close(self):
        pass


class _Config(DefaultUR7eEnvConfig):
    DISPLAY_IMAGE = False
    CAMERAS: dict = {}
    TCP_POSE_SOURCE = "fk"
    RESET_JOINTS = Q6
    # go_to_reset refuses long moves; the fake starts already at RESET_JOINTS.
    RESET_MAX_DIST_RAD = 0.5


class _ArmOnlyConfig(_Config):
    """tests/run_real_hil.py without --gripper: gripper channel disabled."""

    ACTION_SCALE = np.array([0.0125, 0.0625, 0.0])


def _env(config=None, backend=None):
    backend = backend or _GripperBackend()
    env = UR7eEnv(fake_env=False, config=config or _Config(), backend=backend)
    return env, backend


# --------------------------------------------------------------------------- #
# the behaviour itself                                                         #
# --------------------------------------------------------------------------- #
def test_a_closed_gripper_is_opened_and_confirmed():
    env, backend = _env(backend=_GripperBackend(start_percent=1.0))
    assert env.open_gripper_for_reset() is True
    # 0.0 = OPEN. Sent unconditionally, exactly once.
    assert backend.gripper_commands == [0.0]
    assert backend.gripper <= env.GRIPPER_OPEN_CONFIRM
    env.close()


def test_an_already_open_gripper_is_still_commanded_not_assumed():
    """_send_gripper_command skips when already open; a reset must not.

    The policy channel may skip because a redundant Modbus write costs an
    actuation slot. A reset has no such budget and every skipped command is a
    silent assumption about hardware state we did not verify.
    """
    env, backend = _env(backend=_GripperBackend(start_percent=0.0))
    assert env.open_gripper_for_reset() is True
    assert backend.gripper_commands == [0.0]
    env.close()


def test_reset_leaves_the_episode_starting_open():
    """The integration that matters: reset() -> first observation is open."""
    env, backend = _env(backend=_GripperBackend(start_percent=1.0))
    obs, _ = env.reset()
    assert backend.gripper_commands == [0.0]
    # state[0] is gripper_position — the first element the encoder sees.
    assert float(obs["state"]["gripper_pose"][0]) <= env.GRIPPER_OPEN_CONFIRM
    env.close()


# --------------------------------------------------------------------------- #
# the cases where it must NOT act                                              #
# --------------------------------------------------------------------------- #
def test_a_disabled_gripper_channel_is_left_alone():
    """ACTION_SCALE[2] == 0 isolates the arm path on purpose (run_real_hil).

    Opening here would move hardware the operator explicitly asked us not to
    touch, so the reset reports success without commanding anything.
    """
    env, backend = _env(config=_ArmOnlyConfig(), backend=_GripperBackend(1.0))
    assert env.open_gripper_for_reset() is True
    assert backend.gripper_commands == []
    assert backend.gripper == 1.0  # untouched
    env.close()


def test_a_fake_env_commands_no_hardware():
    env = UR7eEnv(fake_env=True, config=_Config())
    assert env.open_gripper_for_reset() is True
    env.close()


# --------------------------------------------------------------------------- #
# failure is reported, not raised                                              #
# --------------------------------------------------------------------------- #
def test_a_stuck_gripper_warns_and_reports_failure(capsys):
    """The arm is already parked, so the episode can still run — but the
    operator has to be told the start is off-distribution, because nothing
    downstream will say so."""
    env, backend = _env(backend=_GripperBackend(start_percent=1.0, opens=False))
    assert env.open_gripper_for_reset(timeout_s=0.15) is False
    assert backend.gripper_commands  # it did try
    assert set(backend.gripper_commands) == {0.0}  # and only ever asked OPEN
    out = capsys.readouterr().out
    assert "WARNING" in out and "gripper" in out
    env.close()


def test_a_stuck_gripper_does_not_break_reset(capsys):
    """A failed open must not raise: raising here would abort an episode over
    a condition the operator can see and decide about."""
    env, backend = _env(backend=_GripperBackend(start_percent=1.0, opens=False))
    env.GRIPPER_OPEN_CONFIRM = 0.15
    obs, info = env.reset()
    assert info == {"succeed": False}
    assert obs is not None
    env.close()


# --------------------------------------------------------------------------- #
# a lost command is re-asserted (FIX D)                                        #
#                                                                              #
# Real-hardware measurement 2026-08-06                                         #
# (ros2_ur_ws/gello_logs/diag_gripper_halfopen_20260806_173307): 7 of 14 open  #
# cycles parked at 0.043 .. 0.42 instead of 0.012 because the single publish   #
# carrying the final setpoint was thrown away. Re-asserting for ~1 s: 0 of 6   #
# failed. This boundary is the worst place to lose one — every offline demo    #
# starts OPEN and gripper_position is state[0].                                #
# --------------------------------------------------------------------------- #
def test_a_dropped_open_is_re_asserted_until_it_takes():
    """The defect itself: the first command vanishes and nobody can tell.

    Without the re-assert this times out and the episode starts half-closed
    with only a printed warning to show for it.
    """
    backend = _GripperBackend(start_percent=1.0, drop_first=1)
    env, backend = _env(backend=backend)
    assert env.open_gripper_for_reset(timeout_s=1.0) is True
    assert len(backend.gripper_commands) >= 2, "a lost open was never repeated"
    assert backend.gripper <= env.GRIPPER_OPEN_CONFIRM
    env.close()


def test_the_re_assert_only_ever_commands_open():
    """It must never re-command a CLOSING setpoint.

    Fingers that stop short of closed are a successful grasp (grip_pos maxes at
    ~0.506 holding an object), so a retry there would be squeezing, not fixing.
    """
    backend = _GripperBackend(start_percent=1.0, opens=False)
    env, backend = _env(backend=backend)
    env.open_gripper_for_reset(timeout_s=0.3)
    assert backend.gripper_commands
    assert set(backend.gripper_commands) == {0.0}
    env.close()


def test_the_re_assert_stops_as_soon_as_feedback_confirms():
    """Bounded by the confirmation, not by the window: the RL loop pays nothing
    extra at the boundary when the very first open lands."""
    env, backend = _env(backend=_GripperBackend(start_percent=1.0))
    assert env.open_gripper_for_reset(timeout_s=1.0) is True
    assert backend.gripper_commands == [0.0]
    env.close()


def test_a_zero_reassert_rate_restores_the_single_publish():
    """The knob must be able to give back exactly the old behaviour."""
    backend = _GripperBackend(start_percent=1.0, opens=False)
    env, backend = _env(backend=backend)
    env.GRIPPER_REASSERT_HZ = 0.0
    assert env.open_gripper_for_reset(timeout_s=0.3) is False
    assert backend.gripper_commands == [0.0]
    env.close()


def test_the_reset_never_re_asserts_a_disabled_gripper_channel():
    """ACTION_SCALE[2] == 0 still means "do not touch the hardware"."""
    env, backend = _env(config=_ArmOnlyConfig(), backend=_GripperBackend(1.0, opens=False))
    assert env.open_gripper_for_reset(timeout_s=0.3) is True
    assert backend.gripper_commands == []
    env.close()


def test_the_re_assert_rate_is_honoured():
    """~10 Hz by default: fast enough to beat the driver's 50 ms rate limiter,
    slow enough that the ROS traffic stays negligible."""
    backend = _GripperBackend(start_percent=1.0, opens=False)
    env, backend = _env(backend=backend)
    env.GRIPPER_REASSERT_HZ = 10.0
    env.open_gripper_for_reset(timeout_s=0.55)
    # first publish + ~5 re-asserts; allow scheduler slop in both directions.
    assert 3 <= len(backend.gripper_commands) <= 9, backend.gripper_commands
    env.close()


def test_the_open_charges_the_policy_debounce():
    """The policy's first step must not re-close on top of a travelling open.

    _send_gripper_command is debounced by GRIPPER_SLEEP against
    last_gripper_act; the reset open sets it so the debounce covers the
    actuation it just started.
    """
    env, backend = _env(backend=_GripperBackend(start_percent=1.0))
    env.last_gripper_act = 0.0
    env.open_gripper_for_reset()
    before = len(backend.gripper_commands)
    env._send_gripper_command(-1.0)  # policy asks to CLOSE immediately
    assert len(backend.gripper_commands) == before, "debounce did not hold"
    env.close()
