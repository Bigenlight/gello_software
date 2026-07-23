"""Unit tests for the generic wrapper without a LeRobot/GPU installation."""

from collections import deque
import importlib
import sys
import types

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="wrapper runtime dependency is installed in the policy image")


class Feature:
    def __init__(self, kind, shape):
        self.type = kind
        self.shape = shape


class FakeConfig:
    type = "diffusion"
    device = "cpu"
    pretrained_path = None
    pretrained_backbone_weights = "ResNet18_Weights.IMAGENET1K_V1"
    n_action_steps = 2
    horizon = 4
    n_obs_steps = 2
    resize_shape = None
    input_features = {
        "observation.state": Feature("state", (7,)),
        "observation.images.cam1": Feature("visual", (3, 8, 8)),
        "observation.images.cam2": Feature("visual", (3, 8, 8)),
    }
    output_features = {"action": Feature("action", (7,))}

    @classmethod
    def from_pretrained(cls, source):
        config = cls()
        config.loaded_from = source
        return config


class FakeProcessor:
    steps = ()

    def __init__(self, post=False):
        self.post = post
        self.reset_calls = 0
        self.inputs = []

    def __call__(self, value):
        self.inputs.append(value)
        return {"action": value} if self.post else value

    def reset(self):
        self.reset_calls += 1


class FakePolicy:
    loaded_config = None

    def __init__(self, config):
        self.config = config
        self._action_queue = deque()
        self.reset_calls = 0
        self.select_calls = 0

    @classmethod
    def from_pretrained(cls, source, config):
        cls.loaded_config = config
        return cls(config)

    def to(self, device):
        return self

    def eval(self):
        return self

    def reset(self):
        self.reset_calls += 1
        self._action_queue.clear()

    def select_action(self, observation):
        self.select_calls += 1
        if not self._action_queue:
            self._action_queue.extend([torch.zeros(7), torch.ones(7)])
        return self._action_queue.popleft().unsqueeze(0)


@pytest.fixture
def wrapper_module(monkeypatch):
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor(post=True)
    configs = types.ModuleType("lerobot.configs")
    configs.PreTrainedConfig = FakeConfig
    factory = types.ModuleType("lerobot.policies.factory")
    factory.get_policy_class = lambda policy_type: FakePolicy
    factory.make_pre_post_processors = lambda **kwargs: (preprocessor, postprocessor)
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.configs", configs)
    monkeypatch.setitem(sys.modules, "lerobot.policies", types.ModuleType("lerobot.policies"))
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory)
    sys.modules.pop("policy_server.lerobot_policy_wrapper", None)
    module = importlib.import_module("policy_server.lerobot_policy_wrapper")
    module._test_preprocessor = preprocessor
    module._test_postprocessor = postprocessor
    yield module
    sys.modules.pop("policy_server.lerobot_policy_wrapper", None)


def _jpeg():
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_diffusion_offline_default_and_explicit_override(wrapper_module):
    wrapper = wrapper_module.LeRobotPolicyWrapper("checkpoint", device="cpu")
    assert wrapper.config.pretrained_backbone_weights is None
    wrapper = wrapper_module.LeRobotPolicyWrapper(
        "checkpoint", device="cpu",
        config_overrides={"pretrained_backbone_weights": "explicit"},
    )
    assert wrapper.config.pretrained_backbone_weights == "explicit"


def test_decode_preprocess_select_postprocess_reset_and_refill(wrapper_module):
    wrapper = wrapper_module.LeRobotPolicyWrapper("checkpoint", device="cpu")
    jpeg = _jpeg()
    action = wrapper.act(np.zeros(7), jpeg, jpeg)
    assert action.shape == (7,)
    assert wrapper.policy.select_calls == 1
    assert wrapper.last_act_metadata["chunk_refill"] is True
    assert wrapper.last_act_metadata["remaining_chunk_actions"] == 1
    wrapper.act(np.zeros(7), jpeg, jpeg)
    assert wrapper.last_act_metadata["chunk_refill"] is False
    wrapper.reset()
    assert wrapper.policy.reset_calls == 2
    assert wrapper_module._test_preprocessor.reset_calls == 2
    assert wrapper_module._test_postprocessor.reset_calls == 2


def test_invalid_decode_and_shape_are_pre_policy_input_errors(wrapper_module):
    wrapper = wrapper_module.LeRobotPolicyWrapper("checkpoint", device="cpu")
    with pytest.raises(wrapper_module.PolicyInputError, match="imdecode"):
        wrapper.act(np.zeros(7), b"not jpeg", _jpeg())
    with pytest.raises(wrapper_module.PolicyInputError, match="shape"):
        wrapper.act(np.zeros(6), _jpeg(), _jpeg())
    assert wrapper.policy.select_calls == 0


def test_task_detection_and_horizon_validation(wrapper_module, monkeypatch):
    monkeypatch.setattr(FakeConfig, "type", "multi_task_dit")
    wrapper = wrapper_module.LeRobotPolicyWrapper("checkpoint", device="cpu")
    assert wrapper.task_required
    with pytest.raises(wrapper_module.PolicyInputError, match="non-empty task"):
        wrapper.act(np.zeros(7), _jpeg(), _jpeg())

    monkeypatch.setattr(FakeConfig, "n_action_steps", 4)
    with pytest.raises(ValueError, match="horizon - n_obs_steps"):
        wrapper_module.LeRobotPolicyWrapper("checkpoint", device="cpu")


def test_n_action_steps_must_be_positive(wrapper_module, monkeypatch):
    monkeypatch.setattr(FakeConfig, "n_action_steps", 0)
    with pytest.raises(ValueError, match="must be positive"):
        wrapper_module.LeRobotPolicyWrapper("checkpoint", device="cpu")


def test_explicit_task_modes(wrapper_module, monkeypatch):
    wrapper = wrapper_module.LeRobotPolicyWrapper(
        "checkpoint", device="cpu", task_mode="required"
    )
    assert wrapper.task_required
    with pytest.raises(wrapper_module.PolicyInputError, match="non-empty task"):
        wrapper.act(np.zeros(7), _jpeg(), _jpeg())
    monkeypatch.setattr(FakeConfig, "type", "multi_task_dit")
    with pytest.raises(ValueError, match="conflicts"):
        wrapper_module.LeRobotPolicyWrapper(
            "checkpoint", device="cpu", task_mode="disabled"
        )
    with pytest.raises(ValueError, match="auto, required, or disabled"):
        wrapper_module.LeRobotPolicyWrapper(
            "checkpoint", device="cpu", task_mode="sometimes"
        )


def test_contract_feature_kinds_camera_shapes_and_resize(wrapper_module, monkeypatch):
    monkeypatch.setitem(
        FakeConfig.input_features,
        "observation.state",
        Feature("action", (7,)),
    )
    with pytest.raises(ValueError, match="state feature"):
        wrapper_module.LeRobotPolicyWrapper("checkpoint", device="cpu")
    monkeypatch.setitem(
        FakeConfig.input_features,
        "observation.state",
        Feature("state", (7,)),
    )
    monkeypatch.setitem(
        FakeConfig.input_features,
        "observation.images.cam2",
        Feature("visual", (1, 8, 8)),
    )
    with pytest.raises(ValueError, match="CHW"):
        wrapper_module.LeRobotPolicyWrapper("checkpoint", device="cpu")
    monkeypatch.setitem(
        FakeConfig.input_features,
        "observation.images.cam2",
        Feature("visual", (3, 8, 8)),
    )
    with pytest.raises(ValueError, match="positive"):
        wrapper_module.LeRobotPolicyWrapper(
            "checkpoint", device="cpu", external_image_size=(0, 224)
        )
    wrapper = wrapper_module.LeRobotPolicyWrapper(
        "checkpoint", device="cpu", external_image_size=None
    )
    assert wrapper.external_image_size is None
    assert wrapper._resize is None
