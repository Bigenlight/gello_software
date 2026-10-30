"""The governor's tick length: is the rate cap actually a RATE cap?

``PolicyDeltaController._govern`` caps a step at ``v_max * dt`` / ``w_max * dt``.
Before ``step(xi, dt=...)`` existed, ``dt`` was frozen at construction
(``1.0 / hz``) and every call was treated as exactly one 1/hz tick.  That is
fine while the only caller is ``UR7eEnv.step`` at 10 Hz, and it is a safety hole
the moment anything sub-steps: the measured fix for stiff HIL intervention is to
refresh joint targets at 30 Hz instead of 10 Hz (dwell 16% -> 0%, ripple
2.27 -> 0.85), and three calls inside one 100 ms window against a 1/10 s cap
would have let through

    position   0.015 -> 0.045 m       (v_max 0.15 m/s)
    rotation   0.075 -> 0.225 rad     (w_max 0.75 rad/s)
    joint step 0.0625 -> 0.1875 rad   (dq_step_max)

i.e. 3x the hardware budget, silently.  The governor is the rate safety net for
this arm (``ur_env/envs/config.py`` — the ``GOVERNOR`` block and the comment
above it, currently :87-98; find it by name, the line numbers move), so that
must not be reachable.

Two things are pinned here.

* **Nothing changed for the policy path.**  ``dt=None`` must be bit-identical to
  the pre-``dt`` controller, so the frozen ``q_cmd`` literals below are values
  captured from the code *before* the parameter was added.
* **Sub-stepping cannot outrun the budget.**  N calls at ``dt=1/(N*hz)`` may
  never travel further than one call at ``dt=None``.

Also covered: the new ``governed`` / ``governed_scale`` telemetry.  The governor
used to shrink actions with no trace at all (``scale`` was computed and dropped),
which makes a violated ACTION_SCALE*HZ headroom invariant (``config.py:55-63``)
invisible — and that invariant is what makes stored transitions honest about how
far the arm actually moved.

numpy only: ``ur_kin`` is pure numpy even though it lives in the ros2_ur_ws
overlay, and the env is built with ``fake_env=True`` / a fake backend, so no ROS.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
# ur_kin is numpy-only, so the overlay is importable without ROS (same trick as
# tests/test_clip_safety_box.py).
_OVERLAY = os.path.join(_HERE, "..", "..", "ros2_ur_ws", "src", "ur_gello_bringup")
if os.path.isdir(_OVERLAY) and _OVERLAY not in sys.path:
    sys.path.insert(0, _OVERLAY)

ur_kin = pytest.importorskip(
    "ur_gello_bringup.ur_kin", reason="needs the ros2_ur_ws overlay on sys.path"
)

from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.policy_delta_controller import PolicyDeltaController  # noqa: E402
from ur_env.envs.ur7e_env import UR7eEnv  # noqa: E402

GOVERNOR = DefaultUR7eEnvConfig.GOVERNOR
HZ = 10.0
DT = 1.0 / HZ

# Same measured cube_in_cup pose and box as tests/test_clip_safety_box.py, so
# the poses exercised here are poses this rig really visits.
RESET_JOINTS = np.array([3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331])
RESET_XYZ = np.array([0.50364064, 0.13649009, 0.41378238])
LOW = np.array([0.375, -0.229, 0.1785, 2.60, -0.30, 1.10])
HIGH = np.array([0.642, 0.272, 0.550, np.pi, 0.35, 2.20])

# One full-scale +x step: 0.0125 = ACTION_SCALE[0]. Norm 0.0125 < v_max*DT
# (0.015), so this is the IN-BUDGET reference request.
DX = np.array([0.0125, 0.0, 0.0, 0.0, 0.0, 0.0])
# Deliberately over budget on BOTH channels: 0.05 m vs the 0.015 m allowance
# (scale 0.30) and 0.30 rad vs the 0.075 rad allowance (scale 0.25).
XI_SAT = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.30])


class BoxConfig(DefaultUR7eEnvConfig):
    DISPLAY_IMAGE = False
    CAMERAS: dict = {}
    TCP_POSE_SOURCE = "fk"
    RESET_JOINTS = RESET_JOINTS
    ABS_POSE_LIMIT_LOW = LOW
    ABS_POSE_LIMIT_HIGH = HIGH


def _env(config=None):
    return UR7eEnv(fake_env=True, config=config or BoxConfig())


def _controller(env=None, governor=None, hz=HZ):
    ctrl = PolicyDeltaController(
        governor or GOVERNOR, hz, clip_pose=None if env is None else env._clip_command_pose
    )
    ctrl.reset(RESET_JOINTS)
    return ctrl


# --------------------------------------------------------------------------- #
# 1. dt=None is the OLD controller, bit for bit                                #
# --------------------------------------------------------------------------- #
# Captured by running the controller BEFORE step() grew a dt parameter, at
# GOVERNOR = {v_max 0.15, w_max 0.75, dq_step_max 0.0625}, hz 10, from
# RESET_JOINTS.  Full repr precision on purpose: these are exact doubles, and
# the assertion below is atol=0.  If a kinematics or governor change moves them,
# that is a real behaviour change on the live policy path and wants a human.
FROZEN_DX_STEPS = [
    [3.138282224572998, -1.49843361828612, 1.685776759877686,
     -1.7573471900066286, -1.5215999346888027, -3.133017675849208],
    [3.1383605562823713, -1.4693612124262518, 1.6539232881102066,
     -1.7545699808497182, -1.5215998727793423, -3.1329392492759918],
    [3.1384352652963123, -1.4403503116963317, 1.621215230571654,
     -1.750876502428792, -1.5215998140145504, -3.1328644497850386],
]
# XI_SAT is governed (scale 0.25) AND then line-searched, so it pins both gates.
FROZEN_XI_SAT_STEP = [
    3.1382858376578575, -1.5051781750477058, 1.6923180091349386,
    -1.7543761419090325, -1.5217224168713401, -3.189354614294231,
]


def test_dt_none_reproduces_the_pre_dt_controller_bit_for_bit():
    ctrl = _controller()
    for i, expected in enumerate(FROZEN_DX_STEPS):
        q, info = ctrl.step(DX)
        assert np.array_equal(q, np.array(expected)), f"step {i} drifted"
        assert not info["held"], info["reject_reason"]


def test_dt_none_reproduces_the_pre_dt_controller_when_both_gates_bite():
    ctrl = _controller()
    q, info = ctrl.step(XI_SAT)
    assert np.array_equal(q, np.array(FROZEN_XI_SAT_STEP))
    assert not info["held"], info["reject_reason"]


@pytest.mark.parametrize("xi", [DX, XI_SAT, -XI_SAT, np.zeros(6)])
def test_passing_the_nominal_dt_explicitly_is_the_same_as_none(xi):
    """``dt=1/hz`` must not merely round to the default — it must BE it.

    ``_resolve_dt`` returns ``self.dq_step_max`` unmultiplied for ``None``; this
    pins that the multiplied path agrees exactly at the nominal tick, so callers
    can be explicit without perturbing anything.
    """
    a, b = _controller(), _controller()
    for _ in range(4):
        q_none, info_none = a.step(xi)
        q_dt, info_dt = b.step(xi, dt=DT)
        assert np.array_equal(q_none, q_dt)
        assert info_none == info_dt


def test_bad_input_still_holds_and_reports_the_reason():
    """The pre-existing BAD_INPUT gate must fire before anything else."""
    ctrl = _controller()
    q, info = ctrl.step(np.array([np.nan, 0, 0, 0, 0, 0]))
    assert info["held"] and info["reject_reason"] == "BAD_INPUT"
    assert np.array_equal(q, RESET_JOINTS)
    assert info["governed"] is False and info["governed_scale"] == 1.0


# --------------------------------------------------------------------------- #
# 2. THE regression: sub-stepping must not multiply the budget                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 3, 5])
def test_n_substeps_never_travel_further_than_one_nominal_tick(n):
    """The whole point of the dt parameter.

    With the cap frozen at ``1/hz`` this asserted 1.0 and measured n.0.
    """
    one = _controller()
    one.step(XI_SAT)
    d_one = float(np.linalg.norm(one.tcp_cmd()[:3, 3] - RESET_XYZ))

    many = _controller()
    for _ in range(n):
        _, info = many.step(XI_SAT, dt=DT / n)
        assert not info["held"], info["reject_reason"]
    d_many = float(np.linalg.norm(many.tcp_cmd()[:3, 3] - RESET_XYZ))

    assert d_many <= d_one + 1e-12, (
        f"{n} sub-steps travelled {d_many:.6f} m vs {d_one:.6f} m for one tick "
        f"({d_many / d_one:.2f}x) — the rate cap is not a rate cap"
    )
    # And it must not have over-corrected into a freeze: sub-stepping exists to
    # deliver the SAME motion more smoothly, not less motion.
    assert d_many == pytest.approx(d_one, rel=0.05)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_n_substeps_never_out_rotate_one_nominal_tick(n):
    xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.30])  # 4x the 0.075 rad allowance
    R0 = ur_kin.fk(RESET_JOINTS)[:3, :3]

    def total_rotation(ctrl):
        return float(np.linalg.norm(ur_kin.so3_log(ctrl.tcp_cmd()[:3, :3] @ R0.T)))

    one = _controller()
    one.step(xi)
    many = _controller()
    for _ in range(n):
        many.step(xi, dt=DT / n)

    assert total_rotation(many) <= total_rotation(one) + 1e-12


@pytest.mark.parametrize("n", [2, 3, 5])
def test_per_substep_joint_travel_stays_inside_the_scaled_gate(n):
    """The third number in the docstring table: 0.0625 -> 0.1875 rad.

    Uses a governor with the task-space caps lifted so the JOINT gate is the
    only thing that can bind — otherwise the rate cap masks it and the test
    cannot tell a scaled dq_step_max from a frozen one.
    """
    loose = dict(GOVERNOR, v_max=10.0, w_max=10.0)
    budget = GOVERNOR["dq_step_max"]

    ctrl = _controller(governor=loose)
    q_prev = ctrl.q_cmd.copy()
    ctrl.step(XI_SAT)
    dq_nominal = float(np.max(np.abs(ctrl.q_cmd - q_prev)))
    # Sanity: the joint gate really is the binding limit at dt=None here.
    assert dq_nominal == pytest.approx(budget, rel=0.15)

    ctrl = _controller(governor=loose)
    for _ in range(n):
        q_prev = ctrl.q_cmd.copy()
        _, info = ctrl.step(XI_SAT, dt=DT / n)
        assert not info["held"], info["reject_reason"]
        dq = float(np.max(np.abs(ctrl.q_cmd - q_prev)))
        assert dq <= budget / n + 1e-9, (
            f"joint step {dq:.5f} rad exceeds the 1/{n}-tick budget "
            f"{budget / n:.5f} rad — dq_step_max did not scale with dt"
        )


def test_dq_step_max_scales_proportionally_with_dt():
    """dq_step_max is a per-tick expression of a RATE, not an absolute step.

    config.py's ``UPSAMPLER`` block (its ``max_step_rad`` comment, currently
    :106-109) ties 0.0625 rad @ 10 Hz to the upsampler's 0.0025 rad @
    250 Hz — both 0.625 rad/s — and the backend cannot emit faster than that, so
    a 30 Hz tick keeping the full 0.0625 rad would only bank command-vs-measured
    lag.  Direct check on the resolver so the intent is pinned, not inferred.
    """
    ctrl = _controller()
    assert ctrl._resolve_dt(None) == (DT, GOVERNOR["dq_step_max"])
    for n in (2, 3, 5, 25):
        dt, dq = ctrl._resolve_dt(DT / n)
        assert dt == pytest.approx(DT / n)
        assert dq == pytest.approx(GOVERNOR["dq_step_max"] / n, rel=1e-12)
        # rate invariance is the property that matters
        assert dq / dt == pytest.approx(GOVERNOR["dq_step_max"] / DT, rel=1e-12)


def test_a_larger_dt_widens_the_budget_proportionally():
    """The parameter is not a one-way clamp: dt is the tick length, so a longer
    tick legitimately gets a longer step.  This is exactly why dt must be
    declared by the caller and never sampled off the wall clock — a stalled
    step measured at 1.12 s would quietly buy an 11x cap."""
    two_ticks = _controller()
    two_ticks.step(XI_SAT, dt=2 * DT)
    d_two = float(np.linalg.norm(two_ticks.tcp_cmd()[:3, 3] - RESET_XYZ))

    one_tick = _controller()
    one_tick.step(XI_SAT)
    d_one = float(np.linalg.norm(one_tick.tcp_cmd()[:3, 3] - RESET_XYZ))

    assert d_two > d_one * 1.5


# --------------------------------------------------------------------------- #
# 3. an invalid dt is refused, not repaired                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad", [0.0, -0.1, -1.0 / 30.0, float("nan"), float("inf"), float("-inf")]
)
def test_non_positive_or_non_finite_dt_raises(bad):
    ctrl = _controller()
    with pytest.raises(ValueError, match="finite and positive"):
        ctrl.step(DX, dt=bad)


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
def test_an_invalid_dt_leaves_the_command_state_untouched(bad):
    """It must raise BEFORE integrating, or a caught exception would leave the
    controller half-stepped and the arm target desynchronised from T_cmd."""
    ctrl = _controller()
    ctrl.step(DX)
    q_before, T_before = ctrl.q_cmd.copy(), ctrl.tcp_cmd()
    with pytest.raises(ValueError):
        ctrl.step(DX, dt=bad)
    assert np.array_equal(ctrl.q_cmd, q_before)
    assert np.array_equal(ctrl.tcp_cmd(), T_before)


# --------------------------------------------------------------------------- #
# 4. governed / governed_scale telemetry                                       #
# --------------------------------------------------------------------------- #
def test_a_saturating_request_reports_the_scale_it_was_cut_by():
    ctrl = _controller()
    _, info = ctrl.step(XI_SAT)
    # translation wants 0.05 m of a 0.015 m allowance -> 0.30
    # rotation    wants 0.30 rad of a 0.075 rad allowance -> 0.25  (the binding one)
    assert info["governed"] is True
    assert info["governed_scale"] == pytest.approx(0.25, rel=1e-9)


def test_an_in_budget_request_reports_no_governing():
    ctrl = _controller()
    for _ in range(5):
        _, info = ctrl.step(DX)
        assert info["governed"] is False
        assert info["governed_scale"] == 1.0


def test_a_zero_request_reports_no_governing():
    """Both norms below the 1e-12 guard: scale must stay exactly 1.0, not
    become 0 or NaN."""
    ctrl = _controller()
    _, info = ctrl.step(np.zeros(6))
    assert info["governed"] is False and info["governed_scale"] == 1.0


def test_governed_scale_is_the_cut_that_was_actually_applied():
    """A request cut by the reported scale must land where the scale says."""
    ctrl = _controller()
    xi = np.array([0.06, 0.0, 0.0, 0.0, 0.0, 0.0])  # 4x the 0.015 m allowance
    _, info = ctrl.step(xi)
    travelled = ctrl.tcp_cmd()[0, 3] - RESET_XYZ[0]
    assert info["governed_scale"] == pytest.approx(0.25, rel=1e-9)
    assert travelled == pytest.approx(xi[0] * info["governed_scale"], abs=1e-4)


def test_governing_is_reported_at_the_substep_dt_too():
    """At dt=1/30 the allowance is 1/3, so a request that was in budget at
    10 Hz is now governed — and must say so."""
    ctrl = _controller()
    _, info = ctrl.step(DX)                 # 0.0125 m vs 0.015 m: fits
    assert info["governed"] is False
    ctrl = _controller()
    _, info = ctrl.step(DX, dt=DT / 3)      # 0.0125 m vs 0.005 m: does not
    assert info["governed"] is True
    assert info["governed_scale"] == pytest.approx(0.005 / 0.0125, rel=1e-9)


def test_the_existing_info_keys_keep_their_meaning():
    """Additive only: run_real_hil.py and the GUI read these three by name."""
    ctrl = _controller()
    _, info = ctrl.step(DX)
    assert {"held", "reject_reason", "clipped"} <= set(info)
    assert info["held"] is False
    assert info["reject_reason"] is None
    assert info["clipped"] is False


# --------------------------------------------------------------------------- #
# 5. governed is independent of clipped (they are different gates)             #
# --------------------------------------------------------------------------- #
def test_governed_without_clipped_inside_the_box():
    """Rate cap bites, workspace box does not: the request heads INTO the box
    (-x, away from x_high) and is 4x the per-tick allowance."""
    env = _env()
    ctrl = _controller(env)
    _, info = ctrl.step(np.array([-0.06, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert info["governed"] is True and info["governed_scale"] < 1.0
    assert info["clipped"] is False
    assert LOW[0] < ctrl.tcp_cmd()[0, 3] < HIGH[0]


def test_clipped_without_governed_at_the_wall():
    """Workspace box bites, rate cap does not: pinned at x_high, pushing with an
    in-budget full-scale step.  The clamp shortens the step, so the re-govern of
    the net motion is identity and governed must stay False."""
    env = _env()
    ctrl = _controller(env)
    for _ in range(40):  # walk into x_high
        ctrl.step(DX)
    _, info = ctrl.step(DX)
    assert info["clipped"] is True
    assert info["governed"] is False and info["governed_scale"] == 1.0
    assert ctrl.tcp_cmd()[0, 3] == pytest.approx(HIGH[0], abs=1e-6)


def test_both_gates_can_fire_on_the_same_tick():
    env = _env()
    ctrl = _controller(env)
    for _ in range(40):
        ctrl.step(DX)
    # 0.06 m straight at the wall: governed down to 0.015 m, then clamped to 0.
    _, info = ctrl.step(np.array([0.06, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert info["governed"] is True and info["clipped"] is True


def test_a_command_state_outside_the_box_reports_the_recovery_as_governed():
    """The second governor pass (the post-clamp net step) is the only one that
    can bind here, which is what makes reporting min(both) rather than just the
    request pass necessary."""

    class Narrow(BoxConfig):
        # x_high 3 cm BEHIND where the arm starts -> reset latches outside
        ABS_POSE_LIMIT_HIGH = np.array(
            [RESET_XYZ[0] - 0.03, 0.272, 0.550, np.pi, 0.35, 2.20]
        )

    env = _env(Narrow())
    ctrl = _controller(env)
    _, info = ctrl.step(np.zeros(6))  # zero request: pass 1 cannot govern
    assert info["governed"] is True
    assert info["governed_scale"] < 1.0
    step = RESET_XYZ[0] - ctrl.tcp_cmd()[0, 3]
    assert step == pytest.approx(GOVERNOR["v_max"] * DT, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. the keys survive the trip out through UR7eEnv.step                        #
# --------------------------------------------------------------------------- #
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


def test_env_step_surfaces_governed_in_info():
    """Operator-facing half.  A full-scale DIAGONAL action is over budget with
    the shipped config — 0.0125*sqrt(2) = 0.0177 m vs the 0.015 m allowance —
    so the governor cuts it and the stored action overstates the motion.  That
    is precisely the case ``governed`` exists to make visible in logs.
    """
    config = BoxConfig()
    env = UR7eEnv(
        fake_env=False, config=config, backend=_FakeBackend(config.RESET_JOINTS)
    )
    env.reset()

    action = np.zeros(7, dtype=np.float32)
    action[0] = 1.0
    _, _, _, _, info = env.step(action)  # single axis: inside the headroom
    assert info["governed"] is False and info["governed_scale"] == 1.0

    action[1] = -1.0
    _, _, _, _, info = env.step(action)  # diagonal: 18% over the allowance
    assert info["governed"] is True
    expected = GOVERNOR["v_max"] * DT / (config.ACTION_SCALE[0] * np.sqrt(2.0))
    # float32 tolerance, NOT the rel=1e-9 the direct-controller cases above use.
    # This is the only assertion here that reaches the governor through
    # ``env.step``, and the action dtype is float32 by contract (run_contract
    # "action": {"dtype": "float32"} in scripts/run_rlpd_learner_server.py), so
    # ``xi = action * ACTION_SCALE`` carries ~1e-7 relative float32 error into
    # the scale. rel=1e-9 is tighter than the arithmetic can be and only passed
    # by luck of NEP 50 promotion: measured 2026-07-30, numpy 2.2.6 gave
    # 0.8485281374 (passed) and numpy 1.26.4 gave 0.8485281248 (failed, 1.5e-8
    # off) for the same code. The regressions this guards against — the cap not
    # binding at all (1.0) or binding 3x wrong — are orders of magnitude away
    # from 1e-6, so nothing is lost by loosening it.
    assert info["governed_scale"] == pytest.approx(expected, rel=1e-6)
    assert info["clipped"] is False and info["held"] is False
    env.close()
