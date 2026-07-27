"""Dependency-light checks for gripper penalty wiring in both actor CLIs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest


_HERE = Path(__file__).resolve().parent
_INFRA = _HERE.parent
sys.path.insert(0, str(_INFRA))
sys.path.insert(
    0,
    str(
        _INFRA.parent
        / "ros2_ur_ws"
        / "src"
        / "ur_gello_bringup"
    ),
)

from ur_env.envs.wrappers import GripperPenaltyWrapper  # noqa: E402


def _load_script(filename: str, module_name: str):
    path = _INFRA / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(gripper: float) -> dict[str, np.ndarray]:
    state = np.zeros((1, 19), np.float32)
    state[0, -1] = gripper
    return {
        "state": state,
        "cam1": np.zeros((1, 128, 128, 3), np.uint8),
        "cam2": np.zeros((1, 128, 128, 3), np.uint8),
    }


class _TaskEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, (1, 19), np.float32),
            "cam1": gym.spaces.Box(0, 255, (1, 128, 128, 3), np.uint8),
            "cam2": gym.spaces.Box(0, 255, (1, 128, 128, 3), np.uint8),
        }
    )

    def __init__(self, penalty: float = -0.07):
        self.config = SimpleNamespace(GRASP_PENALTY=penalty)
        self.closed = True

    def reset(self, **kwargs):
        del kwargs
        return _observation(float(self.closed)), {}

    def step(self, action):
        del action
        return _observation(float(self.closed)), 0.0, False, False, {}


def test_remote_actor_builds_penalty_inside_stats_and_timestamp_wrappers():
    module = _load_script(
        "run_remote_rlpd_actor.py", "test_run_remote_rlpd_actor_penalty"
    )
    task_env = _TaskEnv(-0.07)

    class ExperimentConfig:
        GRASP_PENALTY = -0.07

        def get_environment(self, **kwargs):
            assert kwargs == {
                "fake_env": True,
                "save_video": False,
                "classifier": False,
            }
            return task_env

    env = module._build_actor_environment(
        ExperimentConfig(),
        SimpleNamespace(fake_env=True, save_video=False),
    )

    assert isinstance(env.env.env, GripperPenaltyWrapper)
    assert env.env.env.env is task_env
    assert env.env.env.penalty == pytest.approx(-0.07)
    env.reset()
    _, _, _, _, info = env.step(
        np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
    )
    assert info["grasp_penalty"] == pytest.approx(-0.07)
    assert int(info["timestamp_ns"]) > 0


def _fake_upstream(*, eval_checkpoint_step: bool = False) -> ModuleType:
    module = ModuleType("train_rlpd")
    module.flags = SimpleNamespace(DEFINE_string=lambda *args, **kwargs: None)
    module.FLAGS = SimpleNamespace(
        eval_checkpoint_step=eval_checkpoint_step,
        actor=True,
        learner=False,
        ur_config_module=None,
    )
    module.config = SimpleNamespace(GRASP_PENALTY=-0.07)
    module.CONFIG_MAPPING = {}
    module.actor = lambda *args, **kwargs: ("upstream", args, kwargs)
    module.main = lambda argv: argv
    module.app = SimpleNamespace(run=lambda callback: callback([]))
    return module


def test_legacy_training_actor_receives_configured_penalty_wrapper(monkeypatch):
    upstream = _fake_upstream()
    monkeypatch.setitem(sys.modules, "train_rlpd", upstream)
    module = _load_script(
        "train_rlpd_actor.py", "test_train_rlpd_actor_penalty"
    )
    captured = {}

    def run_actor(agent, replay, intervention, env, rng, **kwargs):
        del agent, replay, intervention, rng, kwargs
        captured["env"] = env
        return "local"

    monkeypatch.setattr(module, "run_actor", run_actor)
    task_env = _TaskEnv(-0.07)

    result = module._local_actor(
        object(), object(), object(), task_env, object()
    )

    assert result == "local"
    assert isinstance(captured["env"], GripperPenaltyWrapper)
    assert captured["env"].env is task_env
    assert captured["env"].penalty == pytest.approx(-0.07)


def test_legacy_evaluation_actor_is_wrapped_too(monkeypatch):
    upstream = _fake_upstream(eval_checkpoint_step=True)
    monkeypatch.setitem(sys.modules, "train_rlpd", upstream)
    module = _load_script(
        "train_rlpd_actor.py", "test_train_rlpd_actor_eval_penalty"
    )
    task_env = _TaskEnv(-0.07)

    result = module._local_actor(
        object(), object(), object(), task_env, object()
    )

    assert result[0] == "upstream"
    assert isinstance(result[1][3], GripperPenaltyWrapper)
    assert result[1][3].env is task_env
