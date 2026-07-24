"""Tests for local RLPD transition construction and routing."""

import os
import pickle
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.rlpd_actor import (  # noqa: E402
    _dump_transitions,
    _start_step,
    build_transition,
    route_transition,
)


class _Store:
    def __init__(self):
        self.items = []

    def insert(self, transition):
        self.items.append(transition)


def _build(info):
    return build_transition(
        observation={"state": np.array([1.0])},
        policy_action=np.array([0.1, -0.2], dtype=np.float32),
        next_observation={"state": np.array([2.0])},
        reward=0.0,
        done=False,
        info=info,
    )


def test_policy_transition_uses_policy_action_and_replay_only():
    transition = _build(
        {
            "policy_action": np.array([0.1, -0.2], dtype=np.float32),
            "intervened": 0,
        }
    )
    replay_store = _Store()
    intervention_store = _Store()

    route_transition(transition, replay_store, intervention_store)

    np.testing.assert_allclose(transition["actions"], [0.1, -0.2])
    np.testing.assert_allclose(transition["policy_actions"], [0.1, -0.2])
    assert transition["intervened"] == np.uint8(0)
    assert replay_store.items == [transition]
    assert intervention_store.items == []


def test_intervention_transition_preserves_both_actions_and_routes_twice():
    policy_action = np.array([0.1, -0.2], dtype=np.float32)
    human_action = np.array([-0.8, 0.7], dtype=np.float32)
    transition = _build(
        {
            "policy_action": policy_action,
            "intervene_action": human_action,
            "intervened": 1,
        }
    )
    replay_store = _Store()
    intervention_store = _Store()

    route_transition(transition, replay_store, intervention_store)

    np.testing.assert_array_equal(transition["actions"], human_action)
    np.testing.assert_array_equal(transition["policy_actions"], policy_action)
    assert transition["intervened"] == np.uint8(1)
    assert replay_store.items == [transition]
    assert intervention_store.items == [transition]

    policy_action[:] = 1.0
    human_action[:] = 1.0
    np.testing.assert_allclose(transition["actions"], [-0.8, 0.7])
    np.testing.assert_allclose(transition["policy_actions"], [0.1, -0.2])


@pytest.mark.parametrize(
    "info",
    [
        {"intervened": 1},
        {"intervened": 0, "intervene_action": np.zeros(2)},
        {"intervened": 2, "intervene_action": np.zeros(2)},
    ],
)
def test_inconsistent_intervention_metadata_is_rejected(info):
    with pytest.raises(ValueError):
        _build(info)


def test_original_spacemouse_contract_is_inferred_without_explicit_label():
    transition = _build(
        {"intervene_action": np.array([-0.8, 0.7], dtype=np.float32)}
    )

    assert transition["intervened"] == np.uint8(1)
    np.testing.assert_allclose(transition["actions"], [-0.8, 0.7])
    np.testing.assert_allclose(transition["policy_actions"], [0.1, -0.2])


def test_top_level_policy_action_wins_over_inner_wrapper_metadata():
    transition = _build(
        {
            # This may be the same policy output transformed to base frame by
            # an inner intervention wrapper.
            "policy_action": np.array([0.9, 0.8], dtype=np.float32),
            "intervened": 0,
        }
    )

    np.testing.assert_allclose(transition["policy_actions"], [0.1, -0.2])
    np.testing.assert_allclose(transition["actions"], [0.1, -0.2])


def test_pickle_dump_retains_metadata_and_resume_step(tmp_path):
    policy_transition = _build(
        {
            "policy_action": np.array([0.1, -0.2], dtype=np.float32),
            "intervened": 0,
        }
    )
    intervention_transition = _build(
        {
            "policy_action": np.array([0.1, -0.2], dtype=np.float32),
            "intervene_action": np.array([-0.8, 0.7], dtype=np.float32),
            "intervened": 1,
        }
    )

    _dump_transitions(
        str(tmp_path),
        50,
        [policy_transition, intervention_transition],
        [intervention_transition],
    )

    with open(tmp_path / "buffer" / "transitions_50.pkl", "rb") as f:
        replay = pickle.load(f)
    with open(tmp_path / "demo_buffer" / "transitions_50.pkl", "rb") as f:
        interventions = pickle.load(f)

    assert [int(item["intervened"]) for item in replay] == [0, 1]
    assert [int(item["intervened"]) for item in interventions] == [1]
    np.testing.assert_allclose(
        interventions[0]["policy_actions"], [0.1, -0.2]
    )
    assert _start_step(str(tmp_path)) == 51
