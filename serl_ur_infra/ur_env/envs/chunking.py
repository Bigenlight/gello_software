"""jax-free port of upstream's ``ChunkingWrapper`` for ``obs_horizon=1``.

Why port instead of import: upstream
``serl_launcher/wrappers/chunking.py`` does ``import jax`` at module scope,
purely so ``stack_obs`` can call ``jax.tree_map(np.stack, ...)``.  The robot
laptop deliberately has no jax — inference and learning both live on the remote
GPU, and the actor process must stay a plain rclpy/numpy process.  Importing
the upstream module here would make jax a hard dependency of the robot side.

For ``obs_horizon=1`` — the only setting every upstream experiment config uses
(``ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)``) — that
``jax.tree_map`` reduces to prepending one axis, which upstream itself spells
out in the same file as::

    def post_stack_obs(obs, obs_horizon=1):
        obs = {k: v[None] for k, v in obs.items()}

``np.stack`` over a one-element list and ``v[None]`` produce byte-identical
arrays, so a trajectory recorded through this wrapper is interchangeable with
one recorded through upstream's.  Any other horizon raises rather than silently
diverging: stacking a real history needs the deque semantics, and getting that
subtly wrong would corrupt every transition without failing a shape check.

This axis is what turns ``state (19,)`` into the canonical ``(1, 19)`` and
``cam (128,128,3)`` into ``(1,128,128,3)``.
"""

from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np

__all__ = ["space_stack", "ChunkingWrapper"]


def space_stack(space: gym.Space, repeat: int) -> gym.Space:
    """Port of upstream ``space_stack`` (pure gymnasium/numpy, unchanged)."""

    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    raise TypeError(f"space_stack does not handle {type(space).__name__}")


def _stack_one(obs: Mapping[str, Any]) -> dict:
    """``obs_horizon=1`` case of upstream ``stack_obs``: prepend a time axis."""

    return {k: np.asarray(v)[None] for k, v in obs.items()}


class ChunkingWrapper(gym.Wrapper):
    """Adds the leading time axis the canonical observation schema requires.

    Restricted to ``obs_horizon=1`` / ``act_exec_horizon=None``.  See the module
    docstring for why the general case is refused instead of approximated.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_horizon: int,
        act_exec_horizon: Optional[int] = None,
    ):
        super().__init__(env)
        if obs_horizon != 1:
            raise NotImplementedError(
                "this jax-free port only implements obs_horizon=1 (what every "
                f"upstream experiment config uses); got {obs_horizon!r}. Use "
                "serl_launcher.wrappers.chunking.ChunkingWrapper on a host "
                "that has jax if you need a real observation history."
            )
        if act_exec_horizon is not None:
            raise NotImplementedError(
                "this jax-free port only implements act_exec_horizon=None; got "
                f"{act_exec_horizon!r}."
            )
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        self.observation_space = space_stack(env.observation_space, obs_horizon)
        self.action_space = env.action_space

    def step(self, action, *args):
        obs, reward, done, trunc, info = self.env.step(action, *args)
        return _stack_one(obs), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return _stack_one(obs), info
