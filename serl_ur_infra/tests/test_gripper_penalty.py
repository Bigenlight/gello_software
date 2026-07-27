"""Robot-local redundant gripper command penalty tests."""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.envs.config import DefaultUR7eEnvConfig  # noqa: E402
from ur_env.envs.wrappers import (  # noqa: E402
    GripperPenaltyWrapper,
    wrap_gripper_penalty_from_task_config,
)
from ur_env.observation_schema import GRIPPER_POSITION_INDEX  # noqa: E402


def _observation(gripper: float) -> dict[str, np.ndarray]:
    state = np.zeros((1, 19), dtype=np.float32)
    state[0, GRIPPER_POSITION_INDEX] = gripper
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


def test_task_config_penalty_is_explicit_and_authoritative():
    class TaskConfig:
        GRASP_PENALTY = np.float32(-0.07)

    class ExperimentConfig:
        GRASP_PENALTY = -0.07

    env = _Env([1.0, 1.0])
    env.config = TaskConfig()

    wrapper = wrap_gripper_penalty_from_task_config(
        env,
        experiment_config=ExperimentConfig(),
    )
    wrapper.reset()
    _, _, _, _, info = wrapper.step(
        np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
    )

    assert isinstance(wrapper, GripperPenaltyWrapper)
    assert wrapper.penalty == pytest.approx(-0.07)
    assert info["grasp_penalty"] == pytest.approx(-0.07)


def test_task_and_experiment_penalty_disagreement_fails_fast():
    env = _Env([1.0])
    env.config = type("TaskConfig", (), {"GRASP_PENALTY": -0.02})()
    experiment = type(
        "ExperimentConfig", (), {"GRASP_PENALTY": -0.07}
    )()

    with pytest.raises(ValueError, match="disagrees"):
        wrap_gripper_penalty_from_task_config(
            env,
            experiment_config=experiment,
        )


@pytest.mark.parametrize("value", [True, np.nan, 0.01, [0.0]])
def test_invalid_task_penalty_fails_before_actor_start(value):
    env = _Env([1.0])
    env.config = type("TaskConfig", (), {"GRASP_PENALTY": value})()

    with pytest.raises(ValueError, match="GRASP_PENALTY"):
        wrap_gripper_penalty_from_task_config(env)


def test_missing_task_penalty_has_no_silent_default():
    env = _Env([1.0])
    env.config = object()

    with pytest.raises(
        ValueError, match=r"env\.unwrapped\.config\.GRASP_PENALTY"
    ):
        wrap_gripper_penalty_from_task_config(env)
