"""Leader-trigger wiring: two ROS topics -> one 7-element leader reading.

Regression cover for a defect confirmed on the real UR7e stack: gello_publisher
publishes only 6 joints on /gello/joint_states and sends the trigger on the
separate std_msgs/Float32 topic
/gripper/gripper_client/target_gripper_width_percent. The backend used to cache
only the JointState, so ``len(arr) > 6`` was never true, GelloExpert.get_leader
always reported grip=None, and the 0.7/0.3 hysteresis in GelloIntervention was
dead code — a human could never command the gripper while intervening.

Pure unit tests: no rclpy node is constructed, no serial port is opened, and
nothing is published to the robot.
"""

import math
import os
import sys

import gymnasium as gym
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(
    0, os.path.join(_HERE, "..", "..", "ros2_ur_ws", "src", "ur_gello_bringup")
)

from ur_env.envs.ros_backend import (  # noqa: E402
    GELLO_TRIGGER_STALE_S,
    GELLO_TRIGGER_TOPIC,
    merge_gello_state,
)
from ur_env.envs.wrappers import GelloExpert, GelloIntervention  # noqa: E402


Q6 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])


# --------------------------------------------------------------------------- #
# doubles                                                                      #
# --------------------------------------------------------------------------- #
class _Deadman:
    """Deadman under test control (default: engaged, so action() reaches the
    gripper path)."""

    def __init__(self, engaged=True):
        self.engaged = engaged

    def is_engaged(self):
        return self.engaged

    def gain(self):
        return 1.0


class _Backend:
    """Backend double that emulates the REAL two-topic split.

    ``joints`` is what /gello/joint_states carries (6 elements on real
    hardware); ``trigger``/``trigger_age`` are the separate Float32 topic. The
    merge is the production one, so these tests exercise the shipped policy.
    """

    def __init__(self, joints=Q6, trigger=None, trigger_age=0.0, joint_age=0.0):
        self.joints = joints
        self.trigger = trigger
        self.trigger_age = trigger_age
        self.joint_age = joint_age

    def get_gello_state(self):
        return merge_gello_state(
            self.joints, self.joint_age, self.trigger, self.trigger_age
        )

    def get_gello_trigger(self):
        if self.trigger is None:
            return None, float("inf")
        return self.trigger, self.trigger_age


class _Controller:
    def tcp_cmd(self):
        return np.eye(4)


class _StubEnv(gym.Env):
    def __init__(self, backend):
        self.backend = backend
        self.controller = _Controller()
        self.action_scale = np.array([0.01, 0.05, 1.0])
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.last_action = None
        self.hold_reasons = []

    def request_hold(self, reason):
        self.hold_reasons.append(reason)

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        return {}, 0.0, False, False, {}


def _wrapper(backend, engaged=True):
    return GelloIntervention(_StubEnv(backend), deadman=_Deadman(engaged))


# --------------------------------------------------------------------------- #
# the topic contract itself                                                    #
# --------------------------------------------------------------------------- #
def test_trigger_topic_name_matches_gello_publisher():
    """The backend must subscribe to the topic gello_publisher actually uses."""
    assert GELLO_TRIGGER_TOPIC == (
        "/gripper/gripper_client/target_gripper_width_percent"
    )
    node = os.path.join(
        _HERE,
        "..",
        "..",
        "ros2_ur_ws",
        "src",
        "ur_gello_bringup",
        "ur_gello_bringup",
        "gello_publisher_node.py",
    )
    with open(node, encoding="utf-8") as fh:
        source = fh.read()
    assert GELLO_TRIGGER_TOPIC in source
    # ...and the JointState really does carry only the 6 arm joints, which is
    # why the trigger has to come from somewhere else.
    assert "js_msg.position = js[:6].tolist()" in source


# --------------------------------------------------------------------------- #
# merge_gello_state: absent / stale / normal                                   #
# --------------------------------------------------------------------------- #
def test_merge_returns_none_when_no_leader_joints():
    arr, age = merge_gello_state(None, float("inf"), 0.5, 0.0)
    assert arr is None
    assert age == float("inf")


def test_merge_rejects_a_short_leader_message():
    arr, age = merge_gello_state(np.zeros(3), 0.0, 0.5, 0.0)
    assert arr is None
    assert age == float("inf")


def test_merge_normal_case_produces_seven_elements():
    arr, age = merge_gello_state(Q6, 0.02, 0.42, 0.01)
    assert arr.shape == (7,)
    np.testing.assert_allclose(arr[:6], Q6)
    assert arr[6] == 0.42
    # age is the JOINT age; the trigger has its own freshness handling.
    assert age == 0.02


def test_merge_full_trigger_span_survives_the_join():
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        arr, _ = merge_gello_state(Q6, 0.0, value, 0.0)
        assert arr[6] == value


def test_merge_absent_trigger_is_nan_not_zero():
    """0.0 is a legal trigger value (fully open), so it cannot be the sentinel."""
    arr, age = merge_gello_state(Q6, 0.05, None, float("inf"))
    assert arr.shape == (7,)
    assert math.isnan(arr[6])
    assert age == 0.05


def test_merge_stale_trigger_is_nan():
    arr, _ = merge_gello_state(Q6, 0.0, 0.9, GELLO_TRIGGER_STALE_S + 0.01)
    assert math.isnan(arr[6])


def test_merge_trigger_at_the_staleness_boundary_is_still_used():
    arr, _ = merge_gello_state(Q6, 0.0, 0.9, GELLO_TRIGGER_STALE_S)
    assert arr[6] == 0.9


def test_merge_stale_trigger_does_not_disable_arm_teleop():
    """A dead trigger must not inflate the joint age past LEADER_STALE_S."""
    arr, age = merge_gello_state(Q6, 0.01, None, float("inf"))
    assert age < GelloIntervention.LEADER_STALE_S
    np.testing.assert_allclose(arr[:6], Q6)


def test_merge_legacy_seven_element_jointstate_still_works():
    """No publisher packs 7 today, but the old code path assumed it."""
    q7 = np.concatenate([Q6, [0.8]])
    arr, _ = merge_gello_state(q7, 0.0, None, float("inf"))
    assert arr.shape == (7,)
    assert arr[6] == 0.8


def test_merge_stale_topic_beats_legacy_fallback():
    """Once the topic has spoken, going stale is a failure — not a fallback."""
    q7 = np.concatenate([Q6, [0.8]])
    arr, _ = merge_gello_state(q7, 0.0, 0.9, GELLO_TRIGGER_STALE_S + 1.0)
    assert math.isnan(arr[6])


# --------------------------------------------------------------------------- #
# GelloExpert.get_leader                                                       #
# --------------------------------------------------------------------------- #
def test_get_leader_reports_trigger_from_the_separate_topic():
    """THE regression: 6-element joints + Float32 trigger must yield grip != None."""
    expert = GelloExpert(_Backend(joints=Q6, trigger=0.83), deadman=_Deadman())
    q, grip, age = expert.get_leader()
    np.testing.assert_allclose(q, Q6)
    assert grip == 0.83
    assert age == 0.0


def test_get_leader_maps_absent_and_stale_trigger_to_none():
    for backend in (
        _Backend(trigger=None, trigger_age=float("inf")),
        _Backend(trigger=0.9, trigger_age=GELLO_TRIGGER_STALE_S + 0.5),
    ):
        _, grip, _ = GelloExpert(backend, deadman=_Deadman()).get_leader()
        assert grip is None


def test_get_leader_keeps_zero_trigger_distinct_from_absent():
    _, grip, _ = GelloExpert(
        _Backend(trigger=0.0), deadman=_Deadman()
    ).get_leader()
    assert grip == 0.0  # fully open, NOT "no signal"


def test_get_leader_without_any_leader_message():
    backend = _Backend(joints=None, joint_age=float("inf"))
    q, grip, age = GelloExpert(backend, deadman=_Deadman()).get_leader()
    assert q is None and grip is None and age == float("inf")


# --------------------------------------------------------------------------- #
# hysteresis + 3-state invariant                                               #
# --------------------------------------------------------------------------- #
def _states(values, wrapper=None):
    wrapper = wrapper or _wrapper(_Backend())
    return [wrapper._expert_gripper(v) for v in values]


def test_hysteresis_close_open_and_latch():
    w = _wrapper(_Backend())
    assert w._expert_gripper(0.5) == 0.0     # mid-band before any crossing
    assert w._expert_gripper(0.71) == -1.0   # above close threshold -> CLOSE
    assert w._expert_gripper(0.5) == -1.0    # mid-band latches CLOSE
    assert w._expert_gripper(0.29) == 1.0    # below open threshold -> OPEN
    assert w._expert_gripper(0.5) == 1.0     # mid-band latches OPEN
    assert w._expert_gripper(0.69) == 1.0    # still inside the band
    assert w._expert_gripper(0.7) == -1.0    # boundary is inclusive -> CLOSE


def test_hysteresis_boundaries_are_inclusive():
    w = _wrapper(_Backend())
    assert w._expert_gripper(GelloIntervention.GRIP_CLOSE_THR) == -1.0
    assert w._expert_gripper(GelloIntervention.GRIP_OPEN_THR) == 1.0


def test_absent_trigger_holds_and_preserves_the_latch():
    w = _wrapper(_Backend())
    assert w._expert_gripper(0.95) == -1.0
    # dropout -> HOLD (0.0), which UR7eEnv._send_gripper_command ignores
    assert w._expert_gripper(None) == 0.0
    assert w._expert_gripper(None) == 0.0
    # the human's last intent is resumed, not forced to re-cross a threshold
    assert w._expert_gripper(0.5) == -1.0


def test_gripper_command_is_always_three_state():
    w = _wrapper(_Backend())
    values = list(np.linspace(0.0, 1.0, 41)) + [None] * 3 + list(
        np.linspace(1.0, 0.0, 41)
    )
    for out in _states(values, w):
        assert out in (-1.0, 0.0, 1.0)


def test_reset_clears_the_latch():
    w = _wrapper(_Backend())
    w._expert_gripper(0.99)
    assert w._grip_cmd == -1.0
    w.reset()
    assert w._grip_cmd == 0.0
    assert w._expert_gripper(0.5) == 0.0


# --------------------------------------------------------------------------- #
# end to end through the wrapper's action()                                    #
# --------------------------------------------------------------------------- #
def test_intervention_action_carries_the_trigger_command():
    backend = _Backend(joints=Q6, trigger=0.9)
    w = _wrapper(backend)
    expert_a, replaced = w.action(np.zeros(7, dtype=np.float32))
    assert replaced
    assert expert_a.shape == (7,)
    assert expert_a[6] == -1.0  # trigger squeezed -> CLOSE

    backend.trigger = 0.05
    expert_a, _ = w.action(np.zeros(7, dtype=np.float32))
    assert expert_a[6] == 1.0   # trigger released -> OPEN


def test_intervention_survives_a_trigger_dropout_without_losing_the_arm():
    backend = _Backend(joints=Q6, trigger=0.9)
    w = _wrapper(backend)
    w.action(np.zeros(7, dtype=np.float32))

    backend.trigger, backend.trigger_age = None, float("inf")
    expert_a, replaced = w.action(np.zeros(7, dtype=np.float32))
    assert replaced                       # arm teleop continues
    assert expert_a[6] == 0.0             # gripper holds
    assert np.all(np.isfinite(expert_a))  # no NaN leaks into the action


def test_stale_leader_joints_hold_arm_and_gripper_instead_of_using_policy():
    backend = _Backend(
        joints=Q6, trigger=0.9,
        joint_age=GelloIntervention.LEADER_STALE_S + 0.1,
    )
    w = _wrapper(backend)
    policy_action = np.linspace(-0.3, 0.3, 7, dtype=np.float32)
    out, replaced = w.action(policy_action)
    assert replaced
    np.testing.assert_array_equal(out, np.zeros(7, dtype=np.float32))


def test_missing_leader_joints_hold_arm_and_gripper_instead_of_using_policy():
    backend = _Backend(joints=None, trigger=0.9)
    w = _wrapper(backend)
    policy_action = np.linspace(-0.3, 0.3, 7, dtype=np.float32)
    out, replaced = w.action(policy_action)
    assert replaced
    np.testing.assert_array_equal(out, np.zeros(7, dtype=np.float32))
