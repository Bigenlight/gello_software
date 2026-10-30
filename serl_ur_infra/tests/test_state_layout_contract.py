"""Regression tests for the flat 19-D ``state`` INDEX LAYOUT.

``test_observation_schema.py`` checks the schema document against itself: it
asserts ``STATE_FEATURES[-1] == "gripper_position"`` by reading
``STATE_FEATURES``.  That tautology passes no matter what the pipeline does, and
``validate_canonical_observation`` only inspects dtype/shape — which a permuted
19-vector satisfies perfectly.  So a silently reordered state is invisible to
every existing check, on both the laptop and Kanu.

These tests close that hole by deriving the layout from the things that actually
produce it — the live ``UR7eEnv.observation_space`` and the live
``gymnasium.spaces.flatten`` used by upstream ``SERLObsWrapper`` — and comparing
against the declared contract.  Nothing below hardcodes "alphabetical": if a
gymnasium upgrade changes ``spaces.Dict`` ordering, or the env renames/resizes a
proprio group, the probe returns the new truth and the comparison fails.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import os
import sys

import gymnasium as gym
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.actor_network import ActorProtocolError  # noqa: E402
from ur_env.envs.ur7e_env import UR7eEnv  # noqa: E402
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
    CANONICAL_STATE_LAYOUT,
    GRIPPER_POSITION_INDEX,
    OBSERVATION_SCHEMA_ID,
    PROPRIO_KEYS,
    STATE_DIM,
    STATE_FEATURES,
    STATE_GROUPS,
    assert_actor_environment_state_layout,
    assert_state_layout_matches,
    flatten_state_layout,
    gripper_position_from_state,
    observation_schema_document,
    state_slice,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_UPSTREAM_LAUNCHER = os.path.join(
    _REPO_ROOT, "third_party", "hil-serl", "serl_launcher"
)


def _euler_proprio_space() -> gym.spaces.Dict:
    """The proprio space the wrapper chain hands to ``SERLObsWrapper``.

    Built from the REAL ``UR7eEnv.observation_space`` (fake backend, no ROS, no
    robot motion), then mutated exactly the way upstream ``Quat2EulerWrapper``
    mutates it: ``observation_space["state"]["tcp_pose"] = Box(shape=(6,))``.
    In-place assignment on an existing key is deliberate — it is what upstream
    does, and it must not reorder the Dict.
    """
    env = UR7eEnv(fake_env=True)
    state_space = env.observation_space["state"]
    assert state_space["tcp_pose"].shape == (7,), (
        "Quat2EulerWrapper asserts tcp_pose is (7,) upstream; if the env "
        "changed, the whole wrapper chain assertion changed with it"
    )
    state_space["tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,))
    return state_space


# --------------------------------------------------------------------------- #
# 1. The mechanism itself                                                      #
# --------------------------------------------------------------------------- #
def test_gymnasium_dict_reorders_plain_mappings():
    """Pin WHY the layout is what it is, so a behaviour change is legible.

    ``gym.spaces.Dict`` sorts a plain mapping ("for legacy reasons ... this
    could matter for projects flatten the dictionary" — gymnasium
    spaces/dict.py).  ``SERLObsWrapper`` builds its ``proprio_space`` from a
    dict comprehension, so ``proprio_keys`` chooses WHICH groups appear, never
    in WHAT ORDER they are laid out.
    """
    insertion_order = ["tcp_pose", "tcp_vel", "gripper_pose", "tcp_force"]
    space = gym.spaces.Dict(
        {key: gym.spaces.Box(-1, 1, shape=(2,)) for key in insertion_order}
    )

    assert list(space.spaces.keys()) != insertion_order, (
        f"gymnasium {gym.__version__} no longer reorders spaces.Dict — the flat "
        "state layout has changed; re-derive STATE_GROUPS before training"
    )
    assert list(space.spaces.keys()) == sorted(insertion_order)


def test_ordered_dict_input_preserves_order():
    """The one input form gymnasium does NOT sort — used by the negative tests."""
    keys = ["tcp_vel", "gripper_pose", "tcp_pose"]
    space = gym.spaces.Dict(
        collections.OrderedDict(
            (key, gym.spaces.Box(-1, 1, shape=(1,))) for key in keys
        )
    )
    assert list(space.spaces.keys()) == keys


# --------------------------------------------------------------------------- #
# 2. The real pipeline vs. the declared contract                               #
# --------------------------------------------------------------------------- #
def test_real_env_flattens_to_the_declared_layout():
    """THE regression test: live env + live gymnasium == observation_schema.py."""
    layout = flatten_state_layout(_euler_proprio_space())

    assert layout == CANONICAL_STATE_LAYOUT
    assert layout[-1][2] == STATE_DIM
    assert_state_layout_matches(_euler_proprio_space(), source="UR7eEnv proprio")


def test_actor_environment_walks_wrappers_and_checks_live_proprio_space():
    class Inner:
        proprio_space = _euler_proprio_space()

    class Outer:
        env = Inner()

    assert (
        assert_actor_environment_state_layout(Outer())
        == CANONICAL_STATE_LAYOUT
    )


def test_actor_environment_rejects_missing_live_layout_probe():
    class Wrapper:
        env = object()

    with pytest.raises(ActorProtocolError, match="no wrapper exposing"):
        assert_actor_environment_state_layout(Wrapper())


def test_declared_groups_cover_the_env_groups_exactly():
    space = _euler_proprio_space()

    assert set(space.spaces.keys()) == set(PROPRIO_KEYS)
    assert set(key for key, _ in STATE_GROUPS) == set(PROPRIO_KEYS)
    for key, features in STATE_GROUPS:
        assert space[key].shape == (len(features),), (
            f"proprio group {key!r} is {space[key].shape} in the env but "
            f"{len(features)} in STATE_GROUPS"
        )


def test_every_flat_index_maps_to_the_declared_feature_name():
    """Value-level check: a sentinel per scalar, decoded back to a name."""
    space = _euler_proprio_space()
    sample = {}
    expected: list[str] = []
    for key, _start, _stop in flatten_state_layout(space):
        features = dict(STATE_GROUPS)[key]
        base = len(expected)
        sample[key] = np.arange(base, base + len(features), dtype=np.float32)
        expected.extend(features)

    flat = np.asarray(gym.spaces.flatten(space, sample))

    assert list(expected) == list(STATE_FEATURES)
    assert np.array_equal(flat, np.arange(STATE_DIM, dtype=np.float32))


def test_gripper_is_at_index_zero_and_the_last_index_is_angular_velocity():
    """The concrete bug: ``state[0, -1]`` is NOT the gripper.

    ``GripperPenaltyWrapper._gripper_position`` reads ``state[0, -1]`` and
    range-checks it against [0, 1].  Index -1 is ``tcp_angular_velocity_z``, so
    the penalty is computed from a velocity and the range check rejects any
    genuine rotation.  Callers must use ``GRIPPER_POSITION_INDEX``.
    """
    space = _euler_proprio_space()
    sentinel = {
        "gripper_pose": np.array([0.42], dtype=np.float32),
        "tcp_force": np.zeros(3, dtype=np.float32),
        "tcp_pose": np.zeros(6, dtype=np.float32),
        "tcp_torque": np.zeros(3, dtype=np.float32),
        "tcp_vel": np.array([0, 0, 0, 0, 0, -7.5], dtype=np.float32),
    }

    flat = np.asarray(gym.spaces.flatten(space, sentinel))

    assert flat[GRIPPER_POSITION_INDEX] == pytest.approx(0.42)
    assert GRIPPER_POSITION_INDEX == 0
    assert flat[-1] == pytest.approx(-7.5)
    assert STATE_FEATURES[-1] == "tcp_angular_velocity_z"
    assert gripper_position_from_state(flat.reshape(1, STATE_DIM)) == pytest.approx(
        0.42
    )


@pytest.mark.skipif(
    not os.path.isdir(_UPSTREAM_LAUNCHER),
    reason="third_party/hil-serl submodule not checked out in this worktree",
)
def test_upstream_serl_obs_wrapper_produces_the_declared_layout():
    """Run the ACTUAL upstream assembler, not a local re-implementation.

    Skips when the pinned submodule is absent (worktrees often lack it); it must
    pass wherever training actually runs.  Note the proprio_keys handed in are
    deliberately in the "intuitive" order used by the upstream example configs —
    the point is that it changes nothing.
    """
    if _UPSTREAM_LAUNCHER not in sys.path:
        sys.path.insert(0, _UPSTREAM_LAUNCHER)
    if importlib.util.find_spec("serl_launcher.wrappers.serl_obs_wrappers") is None:
        pytest.skip("serl_launcher not importable")
    from serl_launcher.wrappers.serl_obs_wrappers import (  # noqa: E402
        SERLObsWrapper,
    )

    class _Stub(gym.Env):
        def __init__(self, state_space):
            self.observation_space = gym.spaces.Dict(
                {
                    "state": state_space,
                    "images": gym.spaces.Dict(
                        {
                            "cam1": gym.spaces.Box(0, 255, (128, 128, 3), np.uint8),
                            "cam2": gym.spaces.Box(0, 255, (128, 128, 3), np.uint8),
                        }
                    ),
                }
            )
            self.action_space = gym.spaces.Box(-1, 1, (7,), np.float32)

    wrapped = SERLObsWrapper(
        _Stub(_euler_proprio_space()),
        proprio_keys=["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"],
    )

    assert wrapped.observation_space["state"].shape == (STATE_DIM,)
    assert flatten_state_layout(wrapped.proprio_space) == CANONICAL_STATE_LAYOUT

    obs = wrapped.observation(
        {
            "state": {
                "gripper_pose": np.array([0.9], dtype=np.float32),
                "tcp_force": np.full(3, 3.0, dtype=np.float32),
                "tcp_pose": np.full(6, 1.0, dtype=np.float32),
                "tcp_torque": np.full(3, 4.0, dtype=np.float32),
                "tcp_vel": np.full(6, 5.0, dtype=np.float32),
            },
            "images": {},
        }
    )
    flat = np.asarray(obs["state"])
    assert flat.shape == (STATE_DIM,)
    assert flat[GRIPPER_POSITION_INDEX] == pytest.approx(0.9)
    for key, start, stop in CANONICAL_STATE_LAYOUT:
        expected = {
            "gripper_pose": 0.9,
            "tcp_force": 3.0,
            "tcp_pose": 1.0,
            "tcp_torque": 4.0,
            "tcp_vel": 5.0,
        }[key]
        assert np.allclose(flat[start:stop], expected), f"group {key} misplaced"


# --------------------------------------------------------------------------- #
# 3. The checker fails on the failures it exists to catch                      #
# --------------------------------------------------------------------------- #
def test_layout_check_rejects_a_reordered_space():
    """Simulates a gymnasium that stops sorting: same dtype, same shape, wrong order."""
    reordered = gym.spaces.Dict(
        collections.OrderedDict(
            [
                ("tcp_pose", gym.spaces.Box(-np.inf, np.inf, shape=(6,))),
                ("tcp_vel", gym.spaces.Box(-np.inf, np.inf, shape=(6,))),
                ("tcp_force", gym.spaces.Box(-np.inf, np.inf, shape=(3,))),
                ("tcp_torque", gym.spaces.Box(-np.inf, np.inf, shape=(3,))),
                ("gripper_pose", gym.spaces.Box(-1, 1, shape=(1,))),
            ]
        )
    )
    # Shape/dtype validation cannot see this: it is still a 19-D float32 vector.
    assert gym.spaces.flatten_space(reordered).shape == (STATE_DIM,)

    with pytest.raises(ActorProtocolError, match="declares"):
        assert_state_layout_matches(reordered)


def test_layout_check_rejects_a_resized_group():
    """e.g. Quat2EulerWrapper dropped from the chain: tcp_pose stays (7,)."""
    space = UR7eEnv(fake_env=True).observation_space["state"]

    with pytest.raises(ActorProtocolError, match="declares"):
        assert_state_layout_matches(space)


def test_layout_check_rejects_a_renamed_group():
    space = _euler_proprio_space()
    renamed = gym.spaces.Dict(
        {
            ("gripper_position" if key == "gripper_pose" else key): sub
            for key, sub in space.spaces.items()
        }
    )

    with pytest.raises(ActorProtocolError, match="declares"):
        assert_state_layout_matches(renamed)


@pytest.mark.parametrize("bad", [None, {}, gym.spaces.Box(-1, 1, shape=(19,))])
def test_layout_check_rejects_non_dict_spaces(bad):
    with pytest.raises(ActorProtocolError, match="non-empty gymnasium Dict"):
        flatten_state_layout(bad)


# --------------------------------------------------------------------------- #
# 4. The schema hash must move when the layout moves                           #
# --------------------------------------------------------------------------- #
def test_schema_hash_changes_when_the_feature_order_changes():
    """``json.dumps(sort_keys=True)`` sorts dict KEYS, never list ELEMENTS.

    So the ordered ``state_features``/``state_layout`` lists really do make the
    hash order-sensitive — a permuted layout cannot share a hash with this one.
    """

    def digest(document):
        return hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    document = observation_schema_document()
    assert digest(document) == CANONICAL_OBSERVATION_SCHEMA_HASH

    permuted_features = dict(document)
    permuted_features["state_features"] = list(reversed(document["state_features"]))
    assert digest(permuted_features) != CANONICAL_OBSERVATION_SCHEMA_HASH

    permuted_layout = dict(document)
    permuted_layout["state_layout"] = list(reversed(document["state_layout"]))
    assert digest(permuted_layout) != CANONICAL_OBSERVATION_SCHEMA_HASH


def test_schema_hash_and_id_are_pinned():
    """Force any layout change to be a deliberate, reviewed edit.

    The hash is the laptop<->Kanu handshake value; if this literal changes,
    every peer must be redeployed, and any replay buffer or checkpoint recorded
    under the previous id is no longer index-compatible.
    """
    assert OBSERVATION_SCHEMA_ID == "hil-serl-ur-canonical-observation-v2"
    assert CANONICAL_OBSERVATION_SCHEMA_HASH == (
        "3459098d8050886f4cb0e1f10dbf47c994a30bf5ec90994503be2c61c0352903"
    )


def test_schema_document_reports_the_layout():
    document = observation_schema_document()

    assert document["state_layout"] == [
        {"key": key, "start": start, "stop": stop}
        for key, start, stop in CANONICAL_STATE_LAYOUT
    ]
    assert document["state_features"] == list(STATE_FEATURES)


# --------------------------------------------------------------------------- #
# 5. Accessor helpers                                                          #
# --------------------------------------------------------------------------- #
def test_state_slice_round_trips_against_the_layout():
    for key, start, stop in CANONICAL_STATE_LAYOUT:
        assert state_slice(key) == slice(start, stop)

    with pytest.raises(ActorProtocolError, match="unknown proprio group"):
        state_slice("tcp_position")


@pytest.mark.parametrize(
    "state",
    [
        np.zeros((19,), dtype=np.float32),
        np.zeros((1, 19), dtype=np.float64),
        np.zeros((1, 18), dtype=np.float32),
    ],
)
def test_gripper_position_from_state_rejects_non_canonical_states(state):
    with pytest.raises(ActorProtocolError, match="canonical state"):
        gripper_position_from_state(state)
