"""``UR7eEnv.go_to_reset`` must travel the SHORT way round every joint.

forward_position_controller interpolates linearly in raw joint space with no
2*pi awareness, so a reset target is a literal number rather than an angle.
``cube_in_cup``'s RESET_JOINTS sits right on the +-pi seam (wrist_3 at -3.1331,
shoulder_pan ~0.003 rad past +pi), so a parked arm on the other branch is
numerically ~2*pi away while being physically ~0 away.

The fixture below is a real measurement taken off this rig.  Sending the raw
target from it would spin wrist_3 a full revolution -- ~10 s of blind slew and
the Robotiq 2F-85 tool-comm cable wound once around the wrist.

Everything here is a pure unit test against a fake backend; nothing is
published to a robot.
"""

import os
import re
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_OVERLAY = os.path.join(_HERE, "..", "..", "ros2_ur_ws", "src", "ur_gello_bringup")
if os.path.isdir(_OVERLAY) and _OVERLAY not in sys.path:
    sys.path.insert(0, _OVERLAY)

pytest.importorskip(
    "ur_gello_bringup.ur_kin", reason="needs the ros2_ur_ws overlay on sys.path"
)

from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.ur7e_env import UR7eEnv  # noqa: E402

# --- measured on the real rig ---------------------------------------------- #
PARKED_Q = np.array([-1.6313, 2.0202, -1.8113, -1.4961, 3.0013, 3.1795])
RESET_JOINTS = np.array([3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331])

#: wrist_3: +3.1795 vs -3.1331 is 0.0294 rad physically, 6.3126 rad numerically.
NAIVE_GAP = 6.3126
#: worst joint once every target is on the arm's own turn.  It is the ELBOW,
#: and it is 3.5281 rather than the 2.755 a naive circular distance reports:
#: the elbow's range is +-pi, so the "shorter" wrapped elbow target (-4.5664)
#: is outside the feasible set and is not a path the arm can take.
BRANCH_GAP = 3.5281
#: wrist_3 must be commanded here -- next to where it already is.
WRIST3_SHORT_WAY = 3.1501


class _FakeBackend:
    """Ideal robot; optionally refuses to move (arrival-timeout coverage)."""

    dry_run = False

    def __init__(self, q0, frozen=False):
        self.q = np.asarray(q0, dtype=float).copy()
        self.frozen = frozen
        self.commands = []

    def get_joint_state(self):
        return self.q.copy(), np.zeros(6), 0.0

    def get_gripper_percent(self):
        return 0.0, 0.0

    def get_wrench(self):
        return None, float("inf")

    def get_tcp_pose(self):
        return None, float("inf")

    def send_joint_command(self, q_cmd):
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(6).copy()
        self.commands.append(q_cmd)
        if not self.frozen:
            self.q = q_cmd

    def send_gripper_percent(self, fraction):
        pass

    def reset_command_stream(self):
        pass

    def close(self):
        pass


class _Config(DefaultUR7eEnvConfig):
    CAMERAS = {}
    DISPLAY_IMAGE = False
    TCP_POSE_SOURCE = "fk"
    RESET_JOINTS = RESET_JOINTS
    RESET_MAX_DIST_RAD = 4.0   # generous, so the branch logic is what is tested
    RESET_TIMEOUT_S = 0.3
    RESET_TOLERANCE_RAD = 0.02


def _env(q0=PARKED_Q, config=None, frozen=False):
    config = config or _Config()
    backend = _FakeBackend(q0, frozen=frozen)
    return UR7eEnv(fake_env=False, config=config, backend=backend), backend


def test_commanded_target_takes_the_short_way_round():
    """The cable-winding bug, directly.

    Raw RESET_JOINTS would command wrist_3 to -3.1331 while it sits at +3.1795:
    one full revolution the wrong way.
    """
    env, backend = _env()
    env.go_to_reset()

    commanded = backend.commands[0]
    assert commanded[5] == pytest.approx(WRIST3_SHORT_WAY, abs=1e-4)
    assert commanded[5] != pytest.approx(-3.1331, abs=1e-2), "raw target re-sent"
    assert abs(commanded[5] - PARKED_Q[5]) < 0.05, "wrist_3 still travels ~2*pi"
    env.close()


def test_no_joint_is_commanded_more_than_half_a_turn():
    """A reset should never contain a >pi joint move; only the elbow, whose
    range is +-pi and which is therefore never unwrapped, may exceed it."""
    env, backend = _env()
    env.go_to_reset()

    travel = np.abs(backend.commands[0] - PARKED_Q)
    off_elbow = np.delete(travel, 2)
    assert np.all(off_elbow <= np.pi + 1e-9), travel


def test_commanded_target_is_the_same_physical_pose():
    """Branch mapping may only add whole turns -- never a different pose."""
    env, backend = _env()
    env.go_to_reset()

    turns = (backend.commands[0] - RESET_JOINTS) / (2.0 * np.pi)
    assert np.allclose(turns, np.round(turns), atol=1e-9), turns
    env.close()


def test_commanded_target_stays_inside_the_joint_limits():
    import ur_gello_bringup.ur_kin as ur_kin

    env, backend = _env()
    env.go_to_reset()
    assert ur_kin.within_joint_limits(backend.commands[0], margin=0.0)
    env.close()


def test_refusal_reports_the_branch_safe_distance_not_the_naive_one():
    """The misleading-error half of the bug.

    The guard used to announce 6.31 rad for a move whose worst joint is 3.53,
    which reads as a wiring fault rather than "the arm is parked elsewhere".
    """

    class Strict(_Config):
        RESET_MAX_DIST_RAD = 0.5   # the cube_in_cup value

    env, _ = _env(config=Strict())
    with pytest.raises(RuntimeError, match="reset distance") as excinfo:
        env.go_to_reset()

    reported = float(re.search(r"reset distance ([\d.]+) rad", str(excinfo.value))[1])
    assert reported == pytest.approx(BRANCH_GAP, abs=0.01)
    assert reported != pytest.approx(NAIVE_GAP, abs=0.1)
    env.close()


def test_arrival_is_judged_against_the_commanded_target():
    """The wait loop compares against the branch-mapped target, so a fake robot
    that lands exactly on the command counts as arrived.

    Judging against the raw config literal instead would time out on a robot
    that had in fact arrived -- wrist_3 would read 6.31 rad "remaining".
    """
    env, backend = _env()
    env.go_to_reset()  # must not raise
    assert np.allclose(backend.q, backend.commands[0])
    env.close()


def test_arrival_timeout_reports_the_branch_safe_remainder():
    env, _ = _env(frozen=True)
    with pytest.raises(RuntimeError, match="did not arrive") as excinfo:
        env.go_to_reset()

    remaining = float(re.search(r"remaining ([\d.]+) rad", str(excinfo.value))[1])
    assert remaining == pytest.approx(BRANCH_GAP, abs=0.01)
    env.close()


def test_a_pose_already_on_the_right_branch_is_untouched():
    """Regression guard for the ordinary case: no gratuitous re-mapping."""
    q0 = RESET_JOINTS + 0.05
    env, backend = _env(q0=q0)
    env.go_to_reset()
    assert np.allclose(backend.commands[0], RESET_JOINTS, atol=1e-12)
    env.close()


def test_elbow_is_never_unwrapped_out_of_its_range():
    """The elbow's range is +-pi.  From an elbow near -pi the circularly
    "shorter" target is -4.57, which the arm cannot reach; the literal 1.7168
    is the only legal command."""
    class Loose(_Config):
        RESET_MAX_DIST_RAD = 5.0  # the 4.72 rad elbow travel is the point here

    q0 = PARKED_Q.copy()
    q0[2] = -3.0
    env, backend = _env(q0=q0, config=Loose())
    env.go_to_reset()
    assert backend.commands[0][2] == pytest.approx(RESET_JOINTS[2], abs=1e-12)
    assert abs(backend.commands[0][2]) <= np.pi
    env.close()


def test_a_target_that_cannot_be_reached_on_this_turn_is_refused():
    """Defensive guard: if the nearest branch of a joint would leave +-2pi,
    refuse rather than command an out-of-limits joint."""
    q0 = PARKED_Q.copy()
    q0[0] = 6.2820   # just under +2pi, and >pi past the target -> maps to +9.42
    env, _ = _env(q0=q0)
    with pytest.raises(RuntimeError, match="leaves the"):
        env.go_to_reset()
    env.close()
