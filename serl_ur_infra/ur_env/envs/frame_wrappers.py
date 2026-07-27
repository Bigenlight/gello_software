"""End-effector frame wrappers — faithful ports of the two upstream wrappers
that sit between the task env and ``SERLObsWrapper``.

Ported, not imported.  The upstream classes live in modules that drag in
hardware SDKs this robot does not have:

* ``franka_env.envs.relative_env`` (``RelativeFrame``) does ``from gym import
  Env`` — the *legacy* ``gym`` package, which we do not install.
* ``franka_env.envs.wrappers`` (``Quat2EulerWrapper``) imports
  ``SpaceMouseExpert`` (``pyspacemouse`` + ``hidapi``) and ``FrankaEnv``
  (``pyrealsense2``) at module scope, so importing one name pulls in all of it.

The behaviour below is a line-by-line port of
``third_party/hil-serl/serl_robot_infra/franka_env/``:
``utils/transformations.py`` (``construct_transform_matrix``,
``construct_homogeneous_matrix``), ``envs/relative_env.py`` (``RelativeFrame``)
and ``envs/wrappers.py`` (``Quat2EulerWrapper``).  Keep it that way: the whole
point of this file is that a UR7e run and an upstream Franka run see the same
observation and action semantics.

Chain position (from ``examples/experiments/usb_pickup_insertion/config.py``)::

    env = UR7eTaskEnv(...)
    env = GelloIntervention(env)      # upstream: SpacemouseIntervention
    env = RelativeFrame(env)          # <- here
    env = Quat2EulerWrapper(env)      # <- here
    env = SERLObsWrapper(env, proprio_keys=PROPRIO_KEYS)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = GripperPenaltyWrapper(env, penalty=config.GRASP_PENALTY)

``RelativeFrame`` must sit *outside* the intervention wrapper (applied after
it) and *inside* ``Quat2EulerWrapper`` (applied before it): it reads
``tcp_pose`` as xyz+quat (7,), and it converts the base-frame expert action the
intervention wrapper produced back into the policy's end-effector frame.
"""

import copy
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

__all__ = [
    "construct_transform_matrix",
    "construct_homogeneous_matrix",
    "RelativeFrame",
    "Quat2EulerWrapper",
]


def construct_transform_matrix(tcp_pose: np.ndarray) -> np.ndarray:
    """blockdiag(R, R) for a 6-vector twist / pose delta.

    :param tcp_pose: (7,) as ``x, y, z, qx, qy, qz, qw`` — scipy's quaternion
        order, which is also what ``/tcp_pose_broadcaster/pose`` gives us.
    """

    rotation = R.from_quat(np.asarray(tcp_pose)[3:]).as_matrix()
    transform_matrix = np.zeros((6, 6))
    transform_matrix[:3, :3] = rotation
    transform_matrix[3:, 3:] = rotation
    return transform_matrix


def construct_homogeneous_matrix(tcp_pose: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform for a pose. Same argument convention."""

    tcp_pose = np.asarray(tcp_pose)
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = tcp_pose[:3]
    T[3, 3] = 1.0
    return T


class RelativeFrame(gym.Wrapper):
    """Express observations and actions in the end-effector frame.

    Three separate transforms, easy to conflate:

    1. **Action** (policy -> env): the policy emits a delta in the *tool*
       frame; ``transform_action`` rotates it into the base frame that
       ``UR7eEnv``/``PolicyDeltaController`` expect.  So "+x" means "along the
       tool's own x axis" no matter how the wrist is turned.
    2. **Observation ``tcp_vel``**: base-frame twist -> tool frame.
    3. **Observation ``tcp_pose``** (when ``include_relative_pose``): absolute
       base pose -> pose *relative to this episode's reset pose*, so the state
       right after ``reset()`` is always identity.  A workspace that shifts by
       a few centimetres therefore does not move the state distribution.

    ``info["intervene_action"]`` arrives from below in the base frame and is
    converted back to the tool frame, so the action stored in the replay buffer
    lives in the same frame the policy acts in.

    Note ``tcp_force`` / ``tcp_torque`` are deliberately left alone — upstream
    does not touch them either.  Our ``/force_torque_sensor_broadcaster/wrench``
    publishes in ``tool0``, and libfranka's ``K_F_ext_hat_K`` is likewise in the
    stiffness (end-effector) frame, so both stacks end up with wrench in the
    tool frame alongside a tool-frame ``tcp_vel``.

    Timing subtlety, preserved from upstream: ``step()`` transforms the action
    and ``info["intervene_action"]`` with the transform matrix from *before*
    the step (the pose the action was issued from), and only then refreshes the
    matrix for the returned observation.  Do not "fix" this into a single
    matrix — the two uses are one control period apart on purpose.
    """

    def __init__(self, env: gym.Env, include_relative_pose: bool = True):
        super().__init__(env)
        self.transform_matrix = np.zeros((6, 6))
        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transform from the reset pose's frame to the base frame.
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        info["original_state_obs"] = copy.deepcopy(obs["state"])

        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])
        return self.transform_observation(obs), reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["original_state_obs"] = copy.deepcopy(obs["state"])

        self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )
        return self.transform_observation(obs), info

    def transform_observation(self, obs: Mapping[str, Any]):
        """Base(spatial) frame -> end-effector(body) frame."""

        transform_inv = np.linalg.inv(self.transform_matrix)
        obs["state"]["tcp_vel"] = transform_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        """End-effector(body) frame -> base(spatial) frame."""

        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.transform_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray) -> np.ndarray:
        """Base(spatial) frame -> end-effector(body) frame."""

        action = np.array(action)
        action[:6] = np.linalg.inv(self.transform_matrix) @ action[:6]
        return action


class Quat2EulerWrapper(gym.ObservationWrapper):
    """``tcp_pose`` xyz+quat (7,) -> xyz+euler (6,).

    This is what takes the flat state from 20-D to the canonical 19-D, so it is
    not optional: without it ``validate_canonical_observation`` rejects every
    observation.

    Unlike upstream we deep-copy the observation space before editing it.
    ``gym.ObservationWrapper.__init__`` assigns ``self.observation_space =
    env.observation_space`` — the *same object* — so upstream's in-place
    assignment also rewrites the wrapped env's advertised space, leaving the
    inner env claiming a (6,) ``tcp_pose`` it never produces.  That matters
    here because ``assert_actor_environment_state_layout`` walks the wrapper
    stack and compares spaces.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,), (
            "Quat2EulerWrapper expects xyz+quat tcp_pose (7,); got "
            f"{env.observation_space['state']['tcp_pose'].shape}. It must be "
            "applied directly outside RelativeFrame, before SERLObsWrapper."
        )
        self.observation_space = copy.deepcopy(env.observation_space)
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation: Mapping[str, Any]):
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
