#!/usr/bin/env python3
"""ZMQ wire-protocol v2 tests for the three inference servers (CARROT_EEF_DESIGN §4).

Runs WITHOUT a checkpoint: the heavy lerobot entry points (`ACTPolicy.from_pretrained`,
`PreTrainedConfig.from_pretrained` + `get_policy_class`, `make_pre_post_processors`)
and the JPEG decoder are monkeypatched with stubs, so the REAL engine classes and the
REAL `_handle`/`handle_request` run against a fake policy. What that buys over a pure
duck-typed stub: the dimension read (`config.input_features/output_features`), the
exit-4 refusal, the warm-up width and the chunk_start bookkeeping are all covered by
the code that actually ships.

Interpreter: ros2_ur_ws/act_venv/bin/python (py3.12, torch + lerobot 0.6.1).
    cd ros2_ur_ws/src/gello_policy
    PYTHONPATH=<wt>/ros2_ur_ws/src/ur_gello_bringup:<wt>/ros2_ur_ws/src/gello_policy \
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../act_venv/bin/python -m pytest -q \
      -p no:cacheprovider test/test_zmq_protocol_v2.py
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import pytest

# The server modules live in the py3.12 act_venv (torch + lerobot + cv2 + zmq). The
# package's OWN suite runs under the py3.10 ROS overlay, which has none of those, so
# skip cleanly there instead of erroring the whole collection. zmq_protocol itself is
# import-light and is covered by the py3.10-friendly tests in the client's test file.
for _mod in ("torch", "lerobot", "cv2", "zmq"):
    pytest.importorskip(
        _mod, reason="policy_server tests need the py3.12 act_venv (torch/lerobot)")

import torch  # noqa: E402  (must follow the importorskip guard above)

from gello_policy import eef_space  # noqa: E402
from policy_server import act_server, diffusion_server, fm_server  # noqa: E402
from policy_server import zmq_protocol as proto  # noqa: E402

JPEG = b"\xff\xd8not-a-real-jpeg"  # never decoded: the decoder is stubbed out


# =============================================================================
# stubs
# =============================================================================
class _Feature:
    """Stand-in for lerobot's PolicyFeature (only .shape is read by the servers)."""

    def __init__(self, *shape: int) -> None:
        self.shape = tuple(shape)


class _Config:
    """Stand-in for a lerobot PreTrainedConfig. Attributes are set freely because
    the engines mutate n_action_steps / pretrained_path / device on load."""

    def __init__(self, state_dim: int = 7, action_dim: int = 7, type_: str = "act") -> None:
        self.input_features = {proto.OBS_STATE_KEY: _Feature(state_dim),
                               proto.OBS_CAM1_KEY: _Feature(3, 720, 1280),
                               proto.OBS_CAM2_KEY: _Feature(3, 720, 1280)}
        self.output_features = {proto.POLICY_ACTION_FEATURE: _Feature(action_dim)}
        self.type = type_
        # ACT guards
        self.chunk_size = 30
        self.temporal_ensemble_coeff = None
        self.n_action_steps = 2
        # FM guards / prints
        self.horizon = 16
        self.n_obs_steps = 2
        self.num_integration_steps = 10
        self.integration_method = "euler"
        self.image_resize_shape = [224, 224]


class _Policy:
    """Fake policy with a REAL receding-horizon deque, so chunk_start is exercised
    the way the servers detect it (queue empty before select_action == net re-ran)."""

    def __init__(self, cfg: _Config) -> None:
        self.config = cfg
        self._action_queue: deque = deque()          # ACT reads this attribute
        self._queues = {"action": self._action_queue}  # FM / diffusion read this
        self.select_calls = 0

    # -- lerobot surface the engines touch --------------------------------------
    def to(self, device):  # noqa: D102 - trivial
        return self

    def eval(self):  # noqa: D102 - trivial
        return self

    def reset(self) -> None:
        self._action_queue.clear()

    def select_action(self, _batch) -> torch.Tensor:
        self.select_calls += 1
        if not self._action_queue:  # "net re-ran": refill the horizon
            for _ in range(self.config.n_action_steps):
                self._action_queue.append(0.0)
        self._action_queue.popleft()
        n = self.config.output_features[proto.POLICY_ACTION_FEATURE].shape[0]
        return torch.arange(n, dtype=torch.float32).unsqueeze(0)  # (1, action_dim)


class _Processor:
    """Identity pre/postprocessor (the engines only call it and .reset())."""

    def __call__(self, x):
        return x

    def reset(self) -> None:
        pass


def _install_act_stubs(monkeypatch, state_dim: int = 7, action_dim: int = 7,
                       type_: str = "act") -> _Policy:
    policy = _Policy(_Config(state_dim, action_dim, type_))

    class _ACTPolicy:
        @staticmethod
        def from_pretrained(_ckpt):
            return policy

    monkeypatch.setattr(act_server, "ACTPolicy", _ACTPolicy)
    monkeypatch.setattr(act_server, "make_pre_post_processors",
                        lambda **_kw: (_Processor(), _Processor()))
    monkeypatch.setattr(act_server, "decode_jpeg_to_rgb_float_chw",
                        lambda _b: torch.zeros(3, 360, 640))
    return policy


def _install_fm_stubs(monkeypatch, state_dim: int = 7, action_dim: int = 7,
                      type_: str = "multi_task_dit") -> _Policy:
    cfg = _Config(state_dim, action_dim, type_)
    policy = _Policy(cfg)

    class _PreTrainedConfig:
        @staticmethod
        def from_pretrained(_ckpt):
            return cfg

    class _PolicyCls:
        __name__ = "FakeMultiTaskDiTPolicy"

        @staticmethod
        def from_pretrained(_ckpt, config=None):
            return policy

    monkeypatch.setattr(fm_server, "PreTrainedConfig", _PreTrainedConfig)
    monkeypatch.setattr(fm_server, "get_policy_class", lambda _t: _PolicyCls)
    monkeypatch.setattr(fm_server, "make_pre_post_processors",
                        lambda **_kw: (_Processor(), _Processor()))
    monkeypatch.setattr(fm_server, "decode_jpeg_to_rgb_float_chw",
                        lambda _b: torch.zeros(3, 8, 8))
    return policy


def _act_engine(monkeypatch, *, state_dim=7, action_space=proto.ACTION_SPACE_JOINT,
                checkpoint="/ckpt/fake") -> act_server.ACTInferenceEngine:
    _install_act_stubs(monkeypatch, state_dim=state_dim)
    return act_server.ACTInferenceEngine(checkpoint, "cpu", 2, action_space=action_space)


class _StubEngine:
    """Minimal duck-typed engine: exactly the surface `_handle` touches. Used for the
    diffusion server, whose real load recipe (scheduler + resize_shape + ensemble) is
    out of scope here -- the protocol surface is what this file is about."""

    def __init__(self, state_dim=7, action_space=proto.ACTION_SPACE_JOINT):
        self.state_dim = state_dim
        self.action_dim = proto.ACTION_DIM
        self.action_space = action_space
        self.checkpoint = "/ckpt/stub"
        self.policy_type = "diffusion"
        self.last_chunk_start = False
        self.resets = 0
        self._left = 0

    def reset(self):
        self.resets += 1
        self._left = 0

    def handshake(self):
        return proto.handshake(self.action_space, self.state_dim, self.action_dim,
                               self.policy_type, self.checkpoint)

    def act(self, state, _c1, _c2):
        self.last_chunk_start = self._left == 0
        if self._left == 0:
            self._left = 2
        self._left -= 1
        return np.arange(proto.ACTION_DIM, dtype=np.float64)


def _reset_frames():
    return [json.dumps({proto.KEY_CMD: proto.CMD_RESET}).encode("utf-8")]


def _act_frames(state):
    ctrl = {proto.KEY_CMD: proto.CMD_ACT, proto.KEY_STATE: [float(x) for x in state]}
    return [json.dumps(ctrl).encode("utf-8"), JPEG, JPEG]


# =============================================================================
# zmq_protocol constants
# =============================================================================
def test_v1_literals_unchanged():
    """Every v1 literal keeps its exact spelling: a v1 client must still work."""
    assert (proto.CMD_RESET, proto.CMD_ACT) == ("reset", "act")
    assert (proto.KEY_CMD, proto.KEY_STATE, proto.KEY_OK, proto.KEY_ACTION,
            proto.KEY_ERR) == ("cmd", "state", "ok", "action", "err")
    assert proto.ACT_REQUEST_NFRAMES == 3
    assert (proto.OBS_STATE_KEY, proto.OBS_CAM1_KEY, proto.OBS_CAM2_KEY,
            proto.OBS_TASK_KEY) == ("observation.state", "observation.images.cam1",
                                    "observation.images.cam2", "task")
    assert (proto.STATE_DIM, proto.ACTION_DIM) == (7, 7)
    assert proto.default_endpoint("1.2.3.4", 9) == "tcp://1.2.3.4:9"
    assert (proto.DEFAULT_HOST, proto.DEFAULT_PORT) == ("127.0.0.1", 5591)


def test_v2_key_literals():
    """These must match gello_policy/obs_assembler.py (the py3.10 client) exactly."""
    assert proto.KEY_ACTION_SPACE == "action_space"
    assert proto.KEY_STATE_DIM == "state_dim"
    assert proto.KEY_ACTION_DIM == "action_dim"
    assert proto.KEY_CHUNK_START == "chunk_start"
    assert proto.KEY_POLICY_TYPE == "policy_type"
    assert proto.KEY_CHECKPOINT == "checkpoint"


def test_action_space_table_comes_from_eef_space():
    """The EEF identity/width are NOT duplicated here -- eef_space owns them."""
    assert proto.ACTION_SPACE_JOINT == "joint"
    assert proto.ACTION_SPACE_EEF == eef_space.ACTION_SPACE_ID == "eef_delta_v1"
    assert proto.STATE_DIM_BY_SPACE == {"joint": 7, "eef_delta_v1": 16}
    assert proto.STATE_DIM_BY_SPACE[proto.ACTION_SPACE_EEF] == eef_space.STATE_DIM
    assert proto.ACTION_DIM == eef_space.ACTION_DIM == 7
    assert proto.state_dim_for_space("joint") == 7
    assert proto.state_dim_for_space("eef_delta_v1") == 16
    with pytest.raises(ValueError):
        proto.state_dim_for_space("nope")


def test_handshake_body_shape():
    hs = proto.handshake("joint", 7, 7, "act", "/ckpt")
    assert hs == {"ok": True, "action_space": "joint", "state_dim": 7,
                  "action_dim": 7, "policy_type": "act", "checkpoint": "/ckpt"}


@pytest.mark.parametrize("space,state_dim,action_dim", [
    ("joint", 7, 7),
    ("eef_delta_v1", 16, 7),
])
def test_no_conflict_when_dims_agree(space, state_dim, action_dim):
    assert proto.action_space_conflict(space, state_dim, action_dim) is None
    proto.enforce_action_space(space, state_dim, action_dim, "/ckpt", "[t]")


@pytest.mark.parametrize("space,state_dim,action_dim", [
    ("eef_delta_v1", 7, 7),    # joint checkpoint served as EEF
    ("joint", 16, 7),          # EEF checkpoint served as joint
    ("joint", 7, 6),           # action width wrong for both spaces
    ("bogus_space", 7, 7),     # unknown space
])
def test_conflicting_dims_exit_4(space, state_dim, action_dim):
    assert proto.action_space_conflict(space, state_dim, action_dim) is not None
    with pytest.raises(SystemExit) as exc:
        proto.enforce_action_space(space, state_dim, action_dim, "/ckpt", "[t]")
    assert exc.value.code == proto.EXIT_ACTION_SPACE_MISMATCH == 4


# =============================================================================
# act_server: real engine + real _handle against the stub policy
# =============================================================================
def test_act_engine_reads_dims_from_checkpoint(monkeypatch):
    engine = _act_engine(monkeypatch)
    assert (engine.state_dim, engine.action_dim) == (7, 7)
    assert engine.action_space == "joint"
    assert engine.policy_type == "act"


def test_act_engine_refuses_eef_on_a_joint_checkpoint(monkeypatch):
    """The smoke-test case: a 7-D checkpoint asked to serve eef_delta_v1 exits 4."""
    _install_act_stubs(monkeypatch, state_dim=7)
    with pytest.raises(SystemExit) as exc:
        act_server.ACTInferenceEngine("/ckpt/fake", "cpu", 2,
                                      action_space=proto.ACTION_SPACE_EEF)
    assert exc.value.code == 4


def test_act_engine_refuses_joint_on_an_eef_checkpoint(monkeypatch):
    _install_act_stubs(monkeypatch, state_dim=16)
    with pytest.raises(SystemExit) as exc:
        act_server.ACTInferenceEngine("/ckpt/fake", "cpu", 2,
                                      action_space=proto.ACTION_SPACE_JOINT)
    assert exc.value.code == 4


def test_act_reset_reply_is_the_handshake(monkeypatch):
    engine = _act_engine(monkeypatch, checkpoint="/ckpt/act_banana_in_pot")
    reply = act_server._handle(engine, _reset_frames())
    assert reply == {"ok": True, "action_space": "joint", "state_dim": 7,
                     "action_dim": 7, "policy_type": "act",
                     "checkpoint": "/ckpt/act_banana_in_pot"}


def test_act_eef_reset_reply(monkeypatch):
    engine = _act_engine(monkeypatch, state_dim=16,
                         action_space=proto.ACTION_SPACE_EEF)
    reply = act_server._handle(engine, _reset_frames())
    assert reply["action_space"] == "eef_delta_v1"
    assert reply["state_dim"] == 16 and reply["action_dim"] == 7


def test_act_reply_has_action_and_chunk_start(monkeypatch):
    engine = _act_engine(monkeypatch)
    act_server._handle(engine, _reset_frames())
    reply = act_server._handle(engine, _act_frames(range(7)))
    assert set(reply) == {"ok", "action", "chunk_start"}
    assert reply["ok"] is True
    assert len(reply["action"]) == 7
    assert all(isinstance(x, float) for x in reply["action"])


def test_chunk_start_true_on_first_call_then_false(monkeypatch):
    """n_action_steps=2 -> the queue empties every other tick: T, F, T, F."""
    engine = _act_engine(monkeypatch)
    act_server._handle(engine, _reset_frames())
    flags = [act_server._handle(engine, _act_frames(range(7)))["chunk_start"]
             for _ in range(4)]
    assert flags == [True, False, True, False]


def test_chunk_start_true_again_after_reset(monkeypatch):
    engine = _act_engine(monkeypatch)
    act_server._handle(engine, _reset_frames())
    assert act_server._handle(engine, _act_frames(range(7)))["chunk_start"] is True
    assert act_server._handle(engine, _act_frames(range(7)))["chunk_start"] is False
    act_server._handle(engine, _reset_frames())   # episode boundary drops the queue
    assert act_server._handle(engine, _act_frames(range(7)))["chunk_start"] is True


def test_act_rejects_wrong_state_length(monkeypatch):
    engine = _act_engine(monkeypatch)
    with pytest.raises(ValueError, match="state must be length 7"):
        act_server._handle(engine, _act_frames(range(16)))


def test_act_eef_engine_wants_16_and_rejects_7(monkeypatch):
    engine = _act_engine(monkeypatch, state_dim=16,
                         action_space=proto.ACTION_SPACE_EEF)
    reply = act_server._handle(engine, _act_frames(range(16)))
    assert reply["ok"] is True and len(reply["action"]) == 7
    with pytest.raises(ValueError, match="state must be length 16"):
        act_server._handle(engine, _act_frames(range(7)))


def test_act_warmup_uses_the_real_state_dim(monkeypatch):
    """A 16-D checkpoint must be warmed with a 16-vector, else the normalizer throws.

    The stub processors cannot notice a width error, so assert the width directly.
    """
    seen = {}
    policy = _install_act_stubs(monkeypatch, state_dim=16)

    class _WidthCheckingProcessor(_Processor):
        def __call__(self, obs):
            if isinstance(obs, dict) and proto.OBS_STATE_KEY in obs:
                seen["state_len"] = int(obs[proto.OBS_STATE_KEY].shape[0])
            return obs

    monkeypatch.setattr(act_server, "make_pre_post_processors",
                        lambda **_kw: (_WidthCheckingProcessor(), _Processor()))
    act_server.ACTInferenceEngine("/ckpt/fake", "cpu", 2,
                                  action_space=proto.ACTION_SPACE_EEF)
    assert seen["state_len"] == 16
    assert policy.select_calls == 2  # the warm-up ran


@pytest.mark.parametrize("frames,needle", [
    ([], "empty request"),
    ([b'{"cmd":"nope"}'], "unknown cmd"),
    ([b"not json at all"], "JSONDecodeError"),
    ([b'{"cmd":"act","state":[0,0,0,0,0,0,0]}'], "needs 3 frames"),
    ([b'{"cmd":"act","state":[1,2]}', JPEG, JPEG], "state must be length 7"),
    ([b'{"cmd":"act"}', JPEG, JPEG], "state"),
])
def test_bad_frames_become_ok_false(monkeypatch, frames, needle):
    """handle_request must ANSWER every request: a REQ socket blocks forever otherwise."""
    engine = _act_engine(monkeypatch)
    reply = act_server.handle_request(engine, frames)
    assert reply[proto.KEY_OK] is False
    assert needle in reply[proto.KEY_ERR]
    assert proto.KEY_ACTION not in reply


def test_act_cli_action_space_default_and_choices():
    assert act_server.parse_args(["--checkpoint", "/c"]).action_space == "joint"
    assert act_server.parse_args(
        ["--checkpoint", "/c", "--action-space", "eef_delta_v1"]).action_space \
        == "eef_delta_v1"
    with pytest.raises(SystemExit):  # argparse rejects an unknown space (exit 2)
        act_server.parse_args(["--checkpoint", "/c", "--action-space", "bogus"])


def test_act_cli_action_space_from_env(monkeypatch):
    monkeypatch.setenv("ACT_ACTION_SPACE", "eef_delta_v1")
    assert act_server.parse_args(["--checkpoint", "/c"]).action_space == "eef_delta_v1"


# =============================================================================
# fm_server
# =============================================================================
def test_fm_engine_handshake_and_chunk_start(monkeypatch):
    _install_fm_stubs(monkeypatch, state_dim=16)
    engine = fm_server.FlowMatchingInferenceEngine(
        "/ckpt/fm", "cpu", 2, None, "Put carrot in pot",
        action_space=proto.ACTION_SPACE_EEF)
    reply = fm_server._handle(engine, _reset_frames())
    assert reply == {"ok": True, "action_space": "eef_delta_v1", "state_dim": 16,
                     "action_dim": 7, "policy_type": "multi_task_dit",
                     "checkpoint": "/ckpt/fm"}
    flags = [fm_server._handle(engine, _act_frames(range(16)))["chunk_start"]
             for _ in range(4)]
    assert flags == [True, False, True, False]
    with pytest.raises(ValueError, match="state must be length 16"):
        fm_server._handle(engine, _act_frames(range(7)))


def test_fm_engine_refuses_contradicting_space(monkeypatch):
    _install_fm_stubs(monkeypatch, state_dim=7)
    with pytest.raises(SystemExit) as exc:
        fm_server.FlowMatchingInferenceEngine("/ckpt/fm", "cpu", 2, None, "t",
                                              action_space=proto.ACTION_SPACE_EEF)
    assert exc.value.code == 4


def test_fm_task_default_depends_on_action_space():
    assert fm_server.default_task_for_space("joint") == "put the right banana in the pot"
    assert fm_server.default_task_for_space("eef_delta_v1") == "Put carrot in pot"
    # argparse leaves it None so main() can resolve it against --action-space.
    assert fm_server.parse_args(["--checkpoint", "/c"]).task is None
    assert fm_server.parse_args(["--checkpoint", "/c", "--task", "x"]).task == "x"


def test_fm_task_env_still_wins(monkeypatch):
    monkeypatch.setenv("FM_TASK", "operator override")
    assert fm_server.parse_args(["--checkpoint", "/c"]).task == "operator override"


def test_fm_cli_action_space(monkeypatch):
    assert fm_server.parse_args(["--checkpoint", "/c"]).action_space == "joint"
    monkeypatch.setenv("FM_ACTION_SPACE", "eef_delta_v1")
    assert fm_server.parse_args(["--checkpoint", "/c"]).action_space == "eef_delta_v1"


def test_fm_bad_frames_become_ok_false(monkeypatch):
    _install_fm_stubs(monkeypatch)
    engine = fm_server.FlowMatchingInferenceEngine("/ckpt/fm", "cpu", 2, None, "t")
    reply = fm_server.handle_request(engine, [b'{"cmd":"walk"}'])
    assert reply[proto.KEY_OK] is False and "unknown cmd" in reply[proto.KEY_ERR]


# =============================================================================
# diffusion_server: protocol surface only (duck-typed engine)
# =============================================================================
@pytest.mark.parametrize("module", [act_server, fm_server, diffusion_server],
                         ids=["act", "fm", "diffusion"])
def test_all_three_servers_share_one_protocol(module):
    """The same stub engine must produce byte-identical replies from all three
    servers -- that is what lets one client speak to any of them."""
    engine = _StubEngine()
    assert module._handle(engine, _reset_frames()) == {
        "ok": True, "action_space": "joint", "state_dim": 7, "action_dim": 7,
        "policy_type": "diffusion", "checkpoint": "/ckpt/stub"}
    assert engine.resets == 1
    flags = [module._handle(engine, _act_frames(range(7)))["chunk_start"]
             for _ in range(4)]
    assert flags == [True, False, True, False]
    bad = module.handle_request(engine, [b'{"cmd":"act","state":[1]}', JPEG, JPEG])
    assert bad[proto.KEY_OK] is False and "state must be length 7" in bad[proto.KEY_ERR]


@pytest.mark.parametrize("module", [act_server, fm_server, diffusion_server],
                         ids=["act", "fm", "diffusion"])
def test_all_three_servers_serve_eef_state_width(module):
    engine = _StubEngine(state_dim=16, action_space=proto.ACTION_SPACE_EEF)
    hs = module._handle(engine, _reset_frames())
    assert hs["action_space"] == "eef_delta_v1" and hs["state_dim"] == 16
    assert module._handle(engine, _act_frames(range(16)))["ok"] is True


def test_diffusion_cli_action_space(monkeypatch):
    assert diffusion_server.parse_args(["--checkpoint", "/c"]).action_space == "joint"
    monkeypatch.setenv("DIFFUSION_ACTION_SPACE", "eef_delta_v1")
    assert diffusion_server.parse_args(["--checkpoint", "/c"]).action_space == "eef_delta_v1"


# =============================================================================
# client parity (the py3.10 side keeps its own literals; they must not drift)
# =============================================================================
def test_wire_literals_match_the_client_module():
    """obs_assembler.py duplicates these literals on purpose (different venv). This
    test is the thing that catches a drift between the two copies."""
    from gello_policy import obs_assembler as client

    assert client.CMD_RESET == proto.CMD_RESET
    assert client.CMD_ACT == proto.CMD_ACT
    for name in ("KEY_CMD", "KEY_STATE", "KEY_OK", "KEY_ACTION", "KEY_ERR",
                 "KEY_CHUNK_START", "KEY_ACTION_SPACE", "KEY_STATE_DIM",
                 "KEY_ACTION_DIM", "KEY_POLICY_TYPE", "KEY_CHECKPOINT"):
        assert getattr(client, name) == getattr(proto, name), name
    assert client.JOINT_ACTION_SPACE_ID == proto.ACTION_SPACE_JOINT
    assert client.EEF_ACTION_SPACE_ID == proto.ACTION_SPACE_EEF
    assert client.STATE_LEN_BY_ACTION_SPACE == proto.STATE_DIM_BY_SPACE
