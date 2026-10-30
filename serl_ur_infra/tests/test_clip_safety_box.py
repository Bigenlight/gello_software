"""The workspace safety box: does it actually clip, and does it clip sanely?

Two layers are covered here.

* ``UR7eEnv.clip_safety_box`` — the port of
  ``franka_env.envs.franka_env.FrankaEnv.clip_safety_box``.  Pure geometry,
  runs with ``fake_env=True`` so it needs neither ROS nor kinematics.
* ``PolicyDeltaController`` with the box wired in — where the clamp meets an
  integrating controller, i.e. where windup would live.  Needs ``ur_kin``
  (pure numpy, but it lives in the ros2_ur_ws overlay).

The numbers are the measured ``cube_in_cup`` ones (23 takes, 19,802 samples),
in the FLANGE frame, so the poses in the assertions are poses this rig really
visits.
"""

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
# ur_kin / eef_delta are numpy-only, so the overlay is importable without ROS.
_OVERLAY = os.path.join(_HERE, "..", "..", "ros2_ur_ws", "src", "ur_gello_bringup")
if os.path.isdir(_OVERLAY) and _OVERLAY not in sys.path:
    sys.path.insert(0, _OVERLAY)

from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.ur7e_env import UR7eEnv  # noqa: E402

# The measured cube_in_cup workspace box.  Index 3 is |rx|, not rx.
LOW = np.array([0.375, -0.229, 0.1785, 2.60, -0.30, 1.10])
HIGH = np.array([0.642, 0.272, 0.550, np.pi, 0.35, 2.20])

RESET_JOINTS = np.array([3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331])
# fk(RESET_JOINTS): position, and euler "xyz" — note rx is on the NEGATIVE
# branch, which is the case a naive np.clip destroys.
RESET_XYZ = np.array([0.50364064, 0.13649009, 0.41378238])
RESET_RPY = np.array([-3.14037725, -0.04918776, 1.55886153])


class BoxConfig(DefaultUR7eEnvConfig):
    DISPLAY_IMAGE = False
    TCP_POSE_SOURCE = "fk"
    RESET_JOINTS = RESET_JOINTS
    ABS_POSE_LIMIT_LOW = LOW
    ABS_POSE_LIMIT_HIGH = HIGH


def _env(config=None):
    """A box-only env: fake_env short-circuits before any ROS/kinematics."""
    return UR7eEnv(fake_env=True, config=config or BoxConfig())


def _pose(xyz, rpy):
    return np.concatenate(
        [np.asarray(xyz, float), Rotation.from_euler("xyz", rpy).as_quat()]
    )


def _rpy(pose7):
    return Rotation.from_quat(pose7[3:]).as_euler("xyz")


# --------------------------------------------------------------------------- #
# clip_safety_box: geometry                                                    #
# --------------------------------------------------------------------------- #
def test_pose_inside_the_box_is_returned_unchanged():
    """The measured episode-start pose must survive the safety gate untouched.

    If it did not, every episode would begin by fighting its own box.
    """
    env = _env()
    pose = _pose(RESET_XYZ, RESET_RPY)
    out = env.clip_safety_box(pose)
    assert np.allclose(out[:3], RESET_XYZ, atol=1e-12)
    assert np.allclose(_rpy(out), RESET_RPY, atol=1e-9)


def test_position_is_clamped_per_axis():
    env = _env()
    out = env.clip_safety_box(_pose([1.0, 0.0, 0.05], RESET_RPY))
    assert out[0] == pytest.approx(HIGH[0])   # x above -> x_high
    assert out[1] == pytest.approx(0.0)       # y in range -> untouched
    assert out[2] == pytest.approx(LOW[2])    # z below the table -> z_low


def test_z_floor_is_the_table():
    """z_low is a physical hard stop; below it the fingertips are in the table."""
    env = _env()
    out = env.clip_safety_box(_pose([0.5, 0.0, -0.30], RESET_RPY))
    assert out[2] == pytest.approx(0.1785)


def test_negative_rx_branch_keeps_its_sign():
    """The upstream magnitude-clip, and the exact bug it exists to prevent.

    The tool points straight down, so rx lives near +-pi and 12% of the
    cube_in_cup samples read as negative.  ``np.clip(-2.0, 2.60, pi)`` returns
    +2.60 — a 4.6 rad wrist flip issued as a safety correction.  Upstream
    clips |rx| and restores the sign, giving -2.60.
    """
    env = _env()
    out_rx = _rpy(env.clip_safety_box(_pose(RESET_XYZ, [-2.0, 0.0, 1.5])))[0]
    assert out_rx < 0.0, "sign of the rx branch was not preserved"
    assert out_rx == pytest.approx(-2.60, abs=1e-9)
    assert out_rx != pytest.approx(+2.60, abs=1e-3), "naive np.clip semantics"


def test_positive_rx_branch_is_clamped_symmetrically():
    env = _env()
    out_rx = _rpy(env.clip_safety_box(_pose(RESET_XYZ, [+2.0, 0.0, 1.5])))[0]
    assert out_rx == pytest.approx(+2.60, abs=1e-9)


@pytest.mark.parametrize("rx", [-3.14037725, +3.14037725, -2.7, +2.7])
def test_rx_inside_the_magnitude_band_is_untouched_on_both_branches(rx):
    """The +-pi seam must not be a discontinuity: both branches are in-box."""
    env = _env()
    out_rx = _rpy(env.clip_safety_box(_pose(RESET_XYZ, [rx, 0.0, 1.5])))[0]
    assert out_rx == pytest.approx(rx, abs=1e-9)


def test_ry_and_rz_use_plain_clip():
    """Only index 3 is a magnitude; ry/rz are ordinary signed ranges."""
    env = _env()
    out = _rpy(env.clip_safety_box(_pose(RESET_XYZ, [-3.10, 1.0, 0.5])))
    assert out[1] == pytest.approx(HIGH[4], abs=1e-9)   # ry 1.0 -> 0.35
    assert out[2] == pytest.approx(LOW[5], abs=1e-9)    # rz 0.5 -> 1.10


def test_clip_does_not_mutate_the_caller_s_array():
    """Upstream mutates in place; we are also handed observations, not just
    scratch buffers, so a mutating clip would silently rewrite an obs."""
    env = _env()
    pose = _pose([1.0, 0.0, 0.05], RESET_RPY)
    before = pose.copy()
    env.clip_safety_box(pose)
    assert np.array_equal(pose, before)


# --------------------------------------------------------------------------- #
# refusing to arm a box that is not a box                                      #
# --------------------------------------------------------------------------- #
def _far_outside():
    return _pose([1.5, 1.5, 1.5], [0.0, 1.4, -2.0])


def test_zero_volume_box_is_refused_not_applied(capsys):
    """DefaultUR7eEnvConfig ships zeros = "not measured yet".

    Clamping to it would command the TCP to the base origin with rx=ry=rz=0 —
    the arm driven down through its own base, as a safety feature.  The box
    must disable itself and say so.
    """
    env = _env(DefaultUR7eEnvConfig())
    warning = capsys.readouterr().out
    assert not env._safety_box_active
    assert "DISABLED" in warning

    pose = _far_outside()
    assert np.array_equal(env.clip_safety_box(pose), pose)


def test_none_limits_are_refused(capsys):
    class NoBox(BoxConfig):
        ABS_POSE_LIMIT_LOW = None

    env = _env(NoBox())
    assert not env._safety_box_active
    assert "DISABLED" in capsys.readouterr().out


def test_inverted_box_is_refused(capsys):
    """np.clip with low>high silently returns `high` for every input: an
    inverted box teleports every pose to one corner."""

    class Inverted(BoxConfig):
        ABS_POSE_LIMIT_LOW = HIGH.copy()
        ABS_POSE_LIMIT_HIGH = LOW.copy()

    env = _env(Inverted())
    assert not env._safety_box_active
    assert "DISABLED" in capsys.readouterr().out
    pose = _far_outside()
    assert np.array_equal(env.clip_safety_box(pose), pose)


def test_negative_rx_magnitude_bound_is_refused(capsys):
    """Index 3 bounds |rx|; a signed-looking range there means the author
    misread the convention and the magnitude clip would surprise them."""

    class SignedRx(BoxConfig):
        ABS_POSE_LIMIT_LOW = np.array([0.375, -0.229, 0.1785, -np.pi, -0.30, 1.10])

    env = _env(SignedRx())
    assert not env._safety_box_active
    assert "|rx|" in capsys.readouterr().out


def test_wrong_shape_is_refused(capsys):
    class Short(BoxConfig):
        ABS_POSE_LIMIT_LOW = np.zeros(3)

    env = _env(Short())
    assert not env._safety_box_active
    assert "DISABLED" in capsys.readouterr().out


def test_measured_box_is_armed():
    env = _env()
    assert env._safety_box_active
    assert np.allclose(env.xyz_bounding_box.high, HIGH[:3])
    assert np.allclose(env.rpy_bounding_box.low, LOW[3:])


def test_nonzero_tool_offset_warns_about_the_frame_coupling(capsys):
    """The box is enforced on the flange pose the controller integrates."""

    class ToolOffset(BoxConfig):
        TCP_OFFSET_XYZ_RPY = [0.0, 0.0, 0.174, 0.0, 0.0, 0.0]

    _env(ToolOffset())
    assert "FLANGE" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# controller integration: the clamp reaches the commanded joints              #
# --------------------------------------------------------------------------- #
ur_kin = pytest.importorskip(
    "ur_gello_bringup.ur_kin", reason="needs the ros2_ur_ws overlay on sys.path"
)

from ur_env.envs.policy_delta_controller import PolicyDeltaController  # noqa: E402

GOVERNOR = DefaultUR7eEnvConfig.GOVERNOR
HZ = 10.0
# 0.0125 = ACTION_SCALE[0], i.e. exactly one full-scale +x step. Keep this
# tied to the config value: it was 0.01 before the 1.25x speed retune, and a
# stale literal here silently stops testing the full-scale case.
DX = np.array([0.0125, 0.0, 0.0, 0.0, 0.0, 0.0])  # one full-scale +x step


def _controller(env=None):
    ctrl = PolicyDeltaController(
        GOVERNOR, HZ, clip_pose=None if env is None else env._clip_command_pose
    )
    ctrl.reset(RESET_JOINTS)
    return ctrl


def test_fixture_reset_pose_matches_forward_kinematics():
    """Guards the hard-coded RESET_XYZ/RESET_RPY above against a kinematics
    change; every other assertion in this file is anchored to them."""
    T = ur_kin.fk(RESET_JOINTS)
    assert np.allclose(T[:3, 3], RESET_XYZ, atol=1e-6)
    assert np.allclose(Rotation.from_matrix(T[:3, :3]).as_euler("xyz"), RESET_RPY,
                       atol=1e-6)


def test_command_stops_at_the_wall_and_says_so():
    env = _env()
    ctrl = _controller(env)

    saw_clip = False
    for _ in range(40):
        _, info = ctrl.step(DX)
        saw_clip = saw_clip or info["clipped"]
        assert not info["held"], info["reject_reason"]
        assert ctrl.tcp_cmd()[0, 3] <= HIGH[0] + 1e-9

    assert saw_clip, "info['clipped'] never reported the clamp"
    assert ctrl.tcp_cmd()[0, 3] == pytest.approx(HIGH[0], abs=1e-6)


def test_inside_the_box_nothing_is_reported_as_clipped():
    env = _env()
    ctrl = _controller(env)
    for _ in range(5):  # 5 cm of headroom before x_high
        _, info = ctrl.step(DX)
        assert not info["clipped"]


def test_no_integrator_windup_at_the_wall():
    """The reason the clamped pose is written back into the controller state.

    The controller integrates its OWN output, so if the clamp were applied only
    to the outgoing command the internal target would keep running past the
    wall at one full ACTION_SCALE step per tick.  After 60 pinned steps it
    would sit more than half a metre outside the box, and the first reversing
    action would produce NO motion for dozens more steps before the arm
    suddenly moved.  Here the reversal must bite on the very next tick.
    """
    env = _env()
    ctrl = _controller(env)

    for _ in range(60):
        ctrl.step(DX)
    x_wall = ctrl.tcp_cmd()[0, 3]
    assert x_wall == pytest.approx(HIGH[0], abs=1e-6), "command state ran past the box"

    _, info = ctrl.step(-DX)
    moved = x_wall - ctrl.tcp_cmd()[0, 3]
    # Derived from DX, not hardcoded: the expected travel IS one full-scale
    # step, so a change to ACTION_SCALE must not require editing this number.
    assert moved == pytest.approx(DX[0], abs=1e-3), (
        f"reversal moved {moved:.4f} m instead of a full step — windup"
    )
    assert not info["clipped"]


def test_the_clamp_is_a_wall_not_a_trap():
    """Pinned at the wall in x, motion along the free axes must still work —
    otherwise the box turns into a HOLD storm the moment the policy leans on
    it (the STEP_LIMIT failure mode the line search exists to avoid)."""
    env = _env()
    ctrl = _controller(env)
    for _ in range(40):
        ctrl.step(DX)

    # 0.008 on each axis: norm 0.0113 < v_max*dt (0.015), so the governor does
    # not scale it and the expected y travel is exactly 5 * 0.008.
    y0 = ctrl.tcp_cmd()[1, 3]
    for _ in range(5):
        _, info = ctrl.step(np.array([0.008, -0.008, 0.0, 0.0, 0.0, 0.0]))
        assert not info["held"], info["reject_reason"]
    assert y0 - ctrl.tcp_cmd()[1, 3] == pytest.approx(0.04, abs=2e-3)
    assert ctrl.tcp_cmd()[0, 3] == pytest.approx(HIGH[0], abs=1e-6)


def test_an_unclipped_tick_is_bit_identical_with_and_without_the_box():
    """The box hook must not perturb ordinary commands.

    ``_clip_command_pose`` returns the ORIGINAL matrix when nothing is out of
    bounds precisely so the matrix->euler->matrix round-trip never touches the
    99.9% of ticks that are inside the box.
    """
    env = _env()
    with_box, without_box = _controller(env), _controller(None)
    for _ in range(5):
        q_a, _ = with_box.step(DX)
        q_b, _ = without_box.step(DX)
        assert np.array_equal(q_a, q_b)


class _FakeBackend:
    """Ideal robot: teleports to whatever joint command it is given."""

    dry_run = False

    def __init__(self, q0):
        self.q = np.asarray(q0, dtype=float).copy()

    def get_joint_state(self):
        return self.q.copy(), np.zeros(6), 0.0

    def get_gripper_percent(self):
        return 0.0, 0.0

    def get_wrench(self):
        return None, float("inf")

    def get_tcp_pose(self):
        return None, float("inf")

    def send_joint_command(self, q_cmd):
        self.q = np.asarray(q_cmd, dtype=float).reshape(6).copy()

    def send_gripper_percent(self, fraction):
        pass

    def reset_command_stream(self):
        pass

    def close(self):
        pass


def test_env_step_surfaces_clipped_in_info():
    """The operator-facing half of the feature.

    A policy pinned against the wall looks exactly like a policy that has
    stopped learning: big actions, no motion, nothing in the reward to explain
    it.  ``info['clipped']`` is what makes that diagnosable after the fact, so
    it has to survive the trip out through ``UR7eEnv.step``.
    """

    class AtTheWall(BoxConfig):
        CAMERAS = {}
        # x_high 4 mm ahead of the start pose: the first step already clamps.
        ABS_POSE_LIMIT_HIGH = np.array(
            [RESET_XYZ[0] + 0.004, 0.272, 0.550, np.pi, 0.35, 2.20]
        )

    config = AtTheWall()
    env = UR7eEnv(
        fake_env=False, config=config, backend=_FakeBackend(config.RESET_JOINTS)
    )
    env.reset()

    _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert info["clipped"] is False and info["held"] is False

    action = np.zeros(7, dtype=np.float32)
    action[0] = 1.0  # full-scale +x, straight into the wall
    _, _, _, _, info = env.step(action)
    assert info["clipped"] is True
    assert info["succeed"] is False
    env.close()


def test_a_command_state_outside_the_box_returns_at_a_governed_rate():
    """Recovery case: reset latches a pose outside the box (box narrowed
    between runs, or a hand-positioned start).  The clamp is then a jump, not
    a delta — it must be re-governed into a rate-limited approach rather than
    one unbounded lunge, and it must not HOLD forever at the boundary.
    """

    class Narrow(BoxConfig):
        # x_high 3 cm BEHIND where the arm starts
        ABS_POSE_LIMIT_HIGH = np.array(
            [RESET_XYZ[0] - 0.03, 0.272, 0.550, np.pi, 0.35, 2.20]
        )

    env = _env(Narrow())
    ctrl = _controller(env)
    x_high = float(env.xyz_bounding_box.high[0])

    xs = [ctrl.tcp_cmd()[0, 3]]
    for _ in range(20):
        _, info = ctrl.step(np.zeros(6))
        xs.append(ctrl.tcp_cmd()[0, 3])

    steps = -np.diff(xs)
    assert np.all(steps >= -1e-9), "recovery must be monotonic toward the box"
    assert steps.max() <= GOVERNOR["v_max"] / HZ + 1e-9, "recovery outran the governor"
    assert xs[-1] == pytest.approx(x_high, abs=1e-6), "never reached the box"
