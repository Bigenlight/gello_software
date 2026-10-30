"""Contract tests for per-step policy/intervention metadata."""

import os
import sys

import gymnasium as gym
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(
    0, os.path.join(_HERE, "..", "..", "ros2_ur_ws", "src", "ur_gello_bringup")
)

from ur_env.envs.wrappers import GelloIntervention  # noqa: E402


class _Deadman:
    def is_engaged(self):
        return False

    def gain(self):
        return 1.0


class _Backend:
    def get_gello_state(self):
        return None, float("inf")


class _Controller:
    def tcp_cmd(self):
        return np.eye(4)


class _StubEnv(gym.Env):
    def __init__(self):
        self.backend = _Backend()
        self.controller = _Controller()
        self.action_scale = np.array([0.01, 0.05, 1.0])
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.last_action = None

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        return {}, 0, False, False, {}


def _make_wrapper():
    return GelloIntervention(_StubEnv(), deadman=_Deadman())


def test_policy_step_reports_policy_action_and_zero_label():
    env = _make_wrapper()
    policy_action = np.linspace(-0.3, 0.3, 7, dtype=np.float32)

    _, _, _, _, info = env.step(policy_action)

    np.testing.assert_array_equal(env.unwrapped.last_action, policy_action)
    np.testing.assert_array_equal(info["policy_action"], policy_action)
    assert info["intervened"] == 0
    assert "intervene_action" not in info


def test_intervention_preserves_policy_action_and_reports_executed_action():
    env = _make_wrapper()
    policy_action = np.linspace(-0.3, 0.3, 7, dtype=np.float32)
    human_action = np.linspace(0.7, -0.7, 7, dtype=np.float32)
    env.action = lambda _: (human_action, True)

    _, _, _, _, info = env.step(policy_action)

    np.testing.assert_array_equal(env.unwrapped.last_action, human_action)
    np.testing.assert_array_equal(info["intervene_action"], human_action)
    np.testing.assert_array_equal(info["policy_action"], policy_action)
    assert info["intervened"] == 1

    policy_action[:] = 1.0
    human_action[:] = 1.0
    np.testing.assert_allclose(info["policy_action"], np.linspace(-0.3, 0.3, 7))
    np.testing.assert_allclose(
        info["intervene_action"], np.linspace(0.7, -0.7, 7)
    )
