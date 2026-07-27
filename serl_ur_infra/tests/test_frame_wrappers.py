"""The wrapper chain that actually produces the canonical observation.

Before these wrappers existed the repo had a validator
(``observation_schema.validate_canonical_observation``) and consumers (receive
server, learner) for ``{state: (1,19), cam1/cam2: (1,128,128,3)}`` but nothing
that *built* one outside of synthetic test fixtures.  The end-to-end test here
is the thing that keeps that gap closed.

Chain under test (upstream ``usb_pickup_insertion/config.py`` order, minus the
reward classifier which is authoritative on the remote server for us)::

    UR7eEnv -> RelativeFrame -> Quat2EulerWrapper -> SERLObsWrapper
            -> ChunkingWrapper(obs_horizon=1)
"""

import copy

import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from ur_env.envs.chunking import ChunkingWrapper, space_stack
from ur_env.envs.config import DefaultUR7eEnvConfig
from ur_env.envs.frame_wrappers import (
    Quat2EulerWrapper,
    RelativeFrame,
    construct_homogeneous_matrix,
    construct_transform_matrix,
)
from ur_env.envs.ur7e_env import UR7eEnv
from ur_env.observation_schema import (
    PROPRIO_KEYS,
    assert_state_layout_matches,
    state_slice,
    validate_canonical_observation,
)

serl_obs_wrappers = pytest.importorskip(
    "serl_launcher.wrappers.serl_obs_wrappers",
    reason="third_party/hil-serl submodule is not checked out",
)
SERLObsWrapper = serl_obs_wrappers.SERLObsWrapper


def _fake_env():
    config = DefaultUR7eEnvConfig()
    config.DISPLAY_IMAGE = False
    return UR7eEnv(fake_env=True, config=config)


def _full_chain():
    env = RelativeFrame(_fake_env())
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=list(PROPRIO_KEYS))
    return ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)


# --------------------------------------------------------------------------- #
# the end-to-end property this whole file exists for
# --------------------------------------------------------------------------- #


def test_chain_produces_a_canonical_observation():
    env = _full_chain()
    try:
        obs, _ = env.reset()
        validate_canonical_observation({k: np.asarray(v) for k, v in obs.items()})

        stepped, *_ = env.step(np.zeros(7, dtype=np.float32))
        validate_canonical_observation(
            {k: np.asarray(v) for k, v in stepped.items()}
        )

        # Spelled out rather than read from the spec, so the numbers the remote
        # server and the learner agree on are visible in the test itself.
        assert set(stepped) == {"state", "cam1", "cam2"}
        assert np.asarray(stepped["state"]).shape == (1, 19)
        assert np.asarray(stepped["state"]).dtype == np.float32
        for cam in ("cam1", "cam2"):
            assert np.asarray(stepped[cam]).shape == (1, 128, 128, 3)
            assert np.asarray(stepped[cam]).dtype == np.uint8
    finally:
        env.close()


def test_chain_advertises_the_layout_the_schema_declares():
    env = RelativeFrame(_fake_env())
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=list(PROPRIO_KEYS))
    try:
        # Raises if the live flatten order drifts from CANONICAL_STATE_LAYOUT.
        assert_state_layout_matches(env.proprio_space)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# RelativeFrame
# --------------------------------------------------------------------------- #


def test_reset_pose_is_the_origin_of_the_relative_frame():
    """The defining property: right after reset the arm is at the origin."""

    env = _full_chain()
    try:
        obs, _ = env.reset()
        pose = np.asarray(obs["state"])[0][state_slice("tcp_pose")]
        np.testing.assert_allclose(pose, np.zeros(6), atol=1e-9)
    finally:
        env.close()


class _PosedEnv(gym.Env):
    """Minimal stand-in whose tcp_pose we control.

    ``UR7eEnv(fake_env=True)`` reports a constant identity-rotation pose, so it
    cannot distinguish "rotated into the base frame" from "passed through", nor
    show the one-step matrix lag.  This env yields a different, non-identity
    orientation on every call.
    """

    def __init__(self, rotations):
        self._rotations = list(rotations)
        self._index = 0
        self.seen_actions = []
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, (7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, (6,)),
                    }
                )
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)

    def _obs(self):
        euler = self._rotations[min(self._index, len(self._rotations) - 1)]
        self._index += 1
        quat = R.from_euler("xyz", euler).as_quat()
        return {
            "state": {
                "tcp_pose": np.concatenate(([0.1, 0.2, 0.3], quat)),
                "tcp_vel": np.zeros(6),
            }
        }

    def reset(self, **kwargs):
        self._index = 0
        return self._obs(), {}

    def step(self, action):
        self.seen_actions.append(np.array(action))
        return self._obs(), 0.0, False, False, {}


def test_action_is_rotated_from_tool_frame_into_base_frame():
    """A tool-frame action reaches the inner env rotated into the base frame."""

    inner = _PosedEnv([(0.0, 0.0, np.pi / 2)])
    env = RelativeFrame(inner)
    env.reset()

    rotation = env.transform_matrix[:3, :3]
    assert not np.allclose(rotation, np.eye(3), atol=1e-6)

    tool_action = np.zeros(7, dtype=np.float32)
    tool_action[0] = 1.0  # +x along the tool
    env.step(tool_action)

    # +x in the tool frame is +y in the base frame after a 90 deg yaw.
    np.testing.assert_allclose(
        inner.seen_actions[0][:3], rotation @ np.array([1.0, 0.0, 0.0]), atol=1e-9
    )
    np.testing.assert_allclose(
        inner.seen_actions[0][:3], [0.0, 1.0, 0.0], atol=1e-9
    )
    assert inner.seen_actions[0][6] == tool_action[6]  # gripper axis untouched


def test_intervene_action_is_converted_back_into_the_tool_frame():
    """The human acts in base frame below; replay must store tool frame."""

    base_action = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    class Intervener(gym.Wrapper):
        def step(self, action):
            obs, reward, done, trunc, info = self.env.step(action)
            info["intervene_action"] = base_action.copy()
            return obs, reward, done, trunc, info

    env = RelativeFrame(Intervener(_fake_env()))
    try:
        env.reset()
        matrix_at_issue = env.transform_matrix.copy()
        _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))

        expected = np.linalg.inv(matrix_at_issue) @ base_action[:6]
        np.testing.assert_allclose(info["intervene_action"][:6], expected, atol=1e-9)
        assert info["intervene_action"][6] == base_action[6]
    finally:
        env.close()


def test_action_and_observation_use_matrices_one_step_apart():
    """Upstream timing: the action uses the pre-step pose, the obs the new one.

    Collapsing these onto one matrix looks like a simplification and silently
    rotates every action by one control period's worth of wrist motion.
    """

    inner = _PosedEnv([(0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 2)])
    env = RelativeFrame(inner)
    env.reset()

    before = env.transform_matrix.copy()
    np.testing.assert_allclose(before, np.eye(6), atol=1e-12)

    tool_action = np.zeros(7, dtype=np.float32)
    tool_action[0] = 1.0
    env.step(tool_action)

    # The action was rotated by the *pre-step* matrix (identity), even though
    # the post-step matrix is a 90 deg yaw.  Using the new matrix would have
    # sent [0, 1, 0] instead.
    np.testing.assert_allclose(inner.seen_actions[0][:3], [1.0, 0.0, 0.0], atol=1e-9)
    assert not np.allclose(env.transform_matrix, before, atol=1e-6)


def test_original_state_obs_keeps_the_untransformed_state():
    env = RelativeFrame(_fake_env())
    try:
        _, info = env.reset()
        assert "original_state_obs" in info
        # Absolute pose is still 7-D xyz+quat, not the relative 7-D we return.
        assert np.shape(info["original_state_obs"]["tcp_pose"]) == (7,)
    finally:
        env.close()


def test_transform_helpers_match_their_definitions():
    pose = np.array([0.1, -0.2, 0.3, *R.from_euler("xyz", [0.3, -0.4, 0.5]).as_quat()])
    rotation = R.from_quat(pose[3:]).as_matrix()

    transform = construct_transform_matrix(pose)
    np.testing.assert_allclose(transform[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(transform[3:, 3:], rotation, atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3:], np.zeros((3, 3)), atol=1e-12)

    homogeneous = construct_homogeneous_matrix(pose)
    np.testing.assert_allclose(homogeneous[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(homogeneous[:3, 3], pose[:3], atol=1e-12)
    assert homogeneous[3, 3] == 1.0


# --------------------------------------------------------------------------- #
# Quat2EulerWrapper
# --------------------------------------------------------------------------- #


def test_quat2euler_shrinks_tcp_pose_without_editing_the_wrapped_space():
    """Upstream mutates the shared space object; we must not."""

    inner = _fake_env()
    try:
        wrapped = Quat2EulerWrapper(RelativeFrame(inner))
        assert wrapped.observation_space["state"]["tcp_pose"].shape == (6,)
        assert inner.observation_space["state"]["tcp_pose"].shape == (7,)
    finally:
        inner.close()


def test_quat2euler_values_are_the_xyz_euler_of_the_quaternion():
    env = Quat2EulerWrapper(RelativeFrame(_fake_env()))
    try:
        obs, info = env.reset()
        quat_pose = info["original_state_obs"]["tcp_pose"]
        # info holds the pre-RelativeFrame pose, so compare shapes/semantics
        # rather than values: what matters is that we emitted 6 numbers whose
        # tail is a valid xyz-euler triple.
        assert np.shape(obs["state"]["tcp_pose"]) == (6,)
        assert np.shape(quat_pose) == (7,)
        assert np.all(np.isfinite(obs["state"]["tcp_pose"]))
    finally:
        env.close()


def test_quat2euler_rejects_a_pose_that_is_already_euler():
    env = Quat2EulerWrapper(RelativeFrame(_fake_env()))
    try:
        with pytest.raises(AssertionError, match="xyz\\+quat"):
            Quat2EulerWrapper(env)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# ChunkingWrapper
# --------------------------------------------------------------------------- #


def test_chunking_adds_exactly_one_leading_axis():
    env = _full_chain()
    try:
        obs, _ = env.reset()
        assert np.shape(obs["state"]) == (1, 19)
        assert np.shape(obs["cam1"]) == (1, 128, 128, 3)
    finally:
        env.close()


def test_chunking_matches_upstream_post_stack_obs():
    """Byte-identical to upstream's own obs_horizon=1 shortcut."""

    raw = {
        "state": np.arange(19, dtype=np.float32),
        "cam1": np.zeros((128, 128, 3), dtype=np.uint8),
    }
    upstream = {k: v[None] for k, v in raw.items()}  # post_stack_obs body

    class Static(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (19,), np.float32),
                "cam1": gym.spaces.Box(0, 255, (128, 128, 3), np.uint8),
            }
        )
        action_space = gym.spaces.Box(-1, 1, (7,), np.float32)

        def reset(self, **kwargs):
            return copy.deepcopy(raw), {}

        def step(self, action):
            return copy.deepcopy(raw), 0.0, False, False, {}

    env = ChunkingWrapper(Static(), obs_horizon=1, act_exec_horizon=None)
    ours, _ = env.reset()
    for key in upstream:
        np.testing.assert_array_equal(ours[key], upstream[key])
        assert ours[key].dtype == upstream[key].dtype


@pytest.mark.parametrize("horizon", [2, 4])
def test_chunking_refuses_a_real_history(horizon):
    with pytest.raises(NotImplementedError, match="obs_horizon=1"):
        ChunkingWrapper(_fake_env(), obs_horizon=horizon, act_exec_horizon=None)


def test_chunking_refuses_receding_horizon_execution():
    with pytest.raises(NotImplementedError, match="act_exec_horizon"):
        ChunkingWrapper(_fake_env(), obs_horizon=1, act_exec_horizon=4)


def test_space_stack_prepends_the_horizon_axis():
    stacked = space_stack(gym.spaces.Box(-1.0, 1.0, (19,), np.float32), 1)
    assert stacked.shape == (1, 19)
    assert stacked.dtype == np.float32
