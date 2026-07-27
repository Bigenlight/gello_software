"""Robot-local redundant gripper command penalty tests."""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.wrappers import GripperPenaltyWrapper  # noqa: E402


def _observation(gripper: float) -> dict[str, np.ndarray]:
    state = np.zeros((1, 19), dtype=np.float32)
    state[0, -1] = gripper
    return {
        "state": state,
        "cam1": np.zeros((1, 128, 128, 3), np.uint8),
        "cam2": np.zeros((1, 128, 128, 3), np.uint8),
    }


class _Env(gym.Env):
    def __init__(self, positions):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.positions = iter(positions)
        self.info = {}

    def reset(self, **kwargs):
        return _observation(next(self.positions)), {}

    def step(self, action):
        return _observation(next(self.positions)), 0.0, False, False, dict(self.info)


def test_default_penalty_matches_upstream_task_configuration():
    assert DefaultUR7eEnvConfig.GRASP_PENALTY == -0.02


def test_penalises_only_repeated_close_and_open_commands():
    wrapper = GripperPenaltyWrapper(
        _Env([1.0, 1.0, 0.5, 0.0, 0.0]), penalty=-0.07
    )
    wrapper.reset()

    _, _, _, _, closed = wrapper.step(
        np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
    )
    _, _, _, _, moving = wrapper.step(
        np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
    )
    _, _, _, _, opening = wrapper.step(
        np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
    )
    _, _, _, _, opened = wrapper.step(
        np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
    )

    assert closed["grasp_penalty"] == -0.07
    assert moving["grasp_penalty"] == 0.0
    assert opening["grasp_penalty"] == 0.0
    assert opened["grasp_penalty"] == -0.07


def test_intervention_action_is_the_penalty_authority():
    env = _Env([1.0, 1.0])
    env.info = {
        "intervene_action": np.array(
            [0, 0, 0, 0, 0, 0, -1], dtype=np.float32
        )
    }
    wrapper = GripperPenaltyWrapper(env)
    wrapper.reset()

    _, _, _, _, info = wrapper.step(
        np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    )

    assert info["grasp_penalty"] == -0.02
