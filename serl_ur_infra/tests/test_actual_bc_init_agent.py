"""Opt-in real JAX tests for grafting a bc-init artifact into the SAC agent.

These build the production frozen-trunk agent, serialise its own two BC
subtrees back out as a synthetic ``hil-serl-bc-init`` artifact, and load them
again.  That round trip is the only way to prove the properties that matter
before this graft drives a real arm:

* the graft lands on exactly ``modules_actor`` + ``modules_grasp_critic`` and
  leaves everything else -- critic, temperature, and the frozen ResNet-10 trunk
  inside the actor -- as the template's;
* the loaded weights actually reach the served action (a hook that ignored its
  ``params`` argument would serve the untrained template while every startup
  gate stayed green -- the review finding this file exists for);
* a trunk that BC touched is still rejected, downstream of a graft that
  structurally succeeds;
* and the three fail-open shapes ``flax.serialization.from_bytes`` swallows --
  an extra key, a reshaped leaf, a missing leaf -- are refusals here.

Run explicitly with::

    cd /home/laptop3/gello_software
    env RUN_HIL_SERL_ACTUAL_BC=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
      PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/hilserl/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_actual_bc_init_agent.py

Without ``RUN_HIL_SERL_ACTUAL_BC=1`` the whole file skips, so the default
suite pays only for collection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HIL_SERL_ACTUAL_BC") != "1",
    reason="set RUN_HIL_SERL_ACTUAL_BC=1 to run the real bc-init graft test",
)

_INFRA = Path(__file__).resolve().parents[1]
_REPO = _INFRA.parent
if str(_INFRA) not in sys.path:
    sys.path.insert(0, str(_INFRA))

# The env-var gate above is the primary one; this keeps the file honest on an
# interpreter that has no jax at all (the actor venv), where it reports as a
# skip rather than a collection error.
pytest.importorskip("jax")

from ur_env.learner.bc_init import (  # noqa: E402
    BC_INIT_FORMAT,
    BC_INIT_FORMAT_VERSION,
    BC_INIT_SUBTREES,
    BcInitError,
    deterministic_sample_action,
    load_bc_init_manifest,
    load_bc_init_params,
    verify_resnet_asset,
)


_RESNET_ASSET = (
    _REPO
    / "third_party"
    / "hil-serl"
    / "examples"
    / "experiments"
    / "resnet10_params.pkl"
)

#: Path of the shared frozen trunk *inside* ``modules_actor``.  BC may not move
#: it, so the head perturbation below stops here and the trunk gets its own
#: test.
_TRUNK_PATH = ("modules_actor", "encoder", "encoder_cam1", "pretrained_encoder")

PARAMETER_FILENAME = "actor_grasp_params.msgpack"
MANIFEST_FILENAME = "manifest.json"
COMPLETION_FILENAME = "completion.json"

_PERTURBATION = 0.01


# --------------------------------------------------------------------------
# The real agent, built once.
# --------------------------------------------------------------------------


def _config():
    from ur_env.learner import LearnerConfig

    return LearnerConfig(
        batch_size=2,
        training_starts=1,
        publish_period=1,
        checkpoint_period=1,
    )


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    """The production template agent -- expensive, so exactly one per module."""

    from ur_env.learner import create_frozen_trunk_feature_agent

    return create_frozen_trunk_feature_agent(
        config=_config(),
        resnet_source_path=_RESNET_ASSET,
        resnet_cache_path=(
            tmp_path_factory.mktemp("resnet-cache") / "verified-resnet10.pkl"
        ),
        validate_versions=True,
    )


# --------------------------------------------------------------------------
# State-dict plumbing: materialise, perturb, compare.
# --------------------------------------------------------------------------


def _numpy(value: Any) -> np.ndarray:
    import jax

    return np.asarray(jax.device_get(value))


def _materialize(node: Any) -> Any:
    """Copy a state dict into plain dicts of numpy arrays.

    Working in numpy from here on keeps mutation, deepcopy and comparison
    trivial, and ``msgpack_serialize`` writes the same bytes either way.
    """

    if isinstance(node, dict):
        return {key: _materialize(value) for key, value in node.items()}
    return _numpy(node)


def _subtree_state(agent) -> dict[str, Any]:
    from flax import serialization

    params = agent.state.params
    return _materialize(
        serialization.to_state_dict({key: params[key] for key in BC_INIT_SUBTREES})
    )


def _in_trunk(path: tuple[str, ...]) -> bool:
    return path[: len(_TRUNK_PATH)] == _TRUNK_PATH


def _bump(array: np.ndarray) -> np.ndarray:
    """Shift a float leaf without changing its shape or dtype."""

    if array.dtype.kind != "f":
        return array
    return (array + np.asarray(_PERTURBATION, dtype=array.dtype)).astype(
        array.dtype, copy=False
    )


def _perturb(state: dict[str, Any], *, head: bool, trunk: bool) -> dict[str, Any]:
    def walk(node: Any, path: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            return {key: walk(value, path + (key,)) for key, value in node.items()}
        wanted = trunk if _in_trunk(path) else head
        return _bump(node) if wanted else np.array(node, copy=True)

    return walk(state, ())


def _leaf_paths(state: Any) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + (key,))
            return
        paths.append(path)

    walk(state, ())
    return paths


def _get(state: Any, path: tuple[str, ...]) -> Any:
    node = state
    for key in path:
        node = node[key]
    return node


def _set(state: Any, path: tuple[str, ...], value: Any) -> None:
    _get(state, path[:-1])[path[-1]] = value


def _assert_state_equal(left: Any, right: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert set(left) == set(right), path
        for key in left:
            _assert_state_equal(left[key], right[key], path + (key,))
        return
    np.testing.assert_array_equal(
        _numpy(left), _numpy(right), err_msg="/".join(path)
    )


def _state_differs(left: Any, right: Any) -> bool:
    return any(
        not np.array_equal(_numpy(_get(left, path)), _numpy(_get(right, path)))
        for path in _leaf_paths(left)
    )


# --------------------------------------------------------------------------
# Writing a synthetic artifact the loader must accept (or refuse).
# --------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_artifact(directory: Path, state: dict[str, Any]) -> Path:
    """Serialise ``state`` as a complete, self-consistent bc-init artifact."""

    from flax import serialization

    directory.mkdir(parents=True, exist_ok=True)
    payload = serialization.msgpack_serialize(serialization.to_state_dict(state))
    parameter_path = directory / PARAMETER_FILENAME
    parameter_path.write_bytes(payload)

    manifest_path = directory / MANIFEST_FILENAME
    _write_json(
        manifest_path,
        {
            "format": BC_INIT_FORMAT,
            "format_version": BC_INIT_FORMAT_VERSION,
            "parameter_subtrees": list(BC_INIT_SUBTREES),
            "parameter_file": PARAMETER_FILENAME,
            "parameter_bytes": parameter_path.stat().st_size,
            "parameter_sha256": hashlib.sha256(payload).hexdigest(),
            "resnet_sha256": _sha256_file(_RESNET_ASSET),
        },
    )
    # completion.json is written last and signs the manifest bytes, exactly as
    # a real writer must order it.
    _write_json(
        directory / COMPLETION_FILENAME,
        {
            "complete": True,
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "parameter_sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    return directory


@pytest.fixture(scope="module")
def template_state(agent) -> dict[str, Any]:
    return _subtree_state(agent)


@pytest.fixture(scope="module")
def head_state(template_state) -> dict[str, Any]:
    """Every BC-owned leaf shifted, except the frozen trunk."""

    return _perturb(template_state, head=True, trunk=False)


@pytest.fixture(scope="module")
def head_artifact(tmp_path_factory, head_state) -> Path:
    return _write_artifact(
        tmp_path_factory.mktemp("bc-init-head") / "artifact", head_state
    )


# --------------------------------------------------------------------------
# 1. Round trip and graft.
# --------------------------------------------------------------------------


def test_graft_replaces_the_bc_subtrees_and_nothing_else(
    agent, template_state, head_state, head_artifact
):
    from flax import serialization

    from ur_env.learner import FrozenResNet10TrunkExtractor

    # The perturbation has to have reached real leaves, or every assertion
    # below would hold vacuously.
    assert _state_differs(template_state, head_state)

    grafted = load_bc_init_params(head_artifact, agent)

    grafted_state = _materialize(serialization.to_state_dict(grafted))
    full_template_state = _materialize(
        serialization.to_state_dict(agent.state.params)
    )
    assert set(grafted_state) == set(full_template_state)
    assert set(BC_INIT_SUBTREES) <= set(grafted_state)

    for key in BC_INIT_SUBTREES:
        _assert_state_equal(grafted_state[key], head_state[key], (key,))
    for key in set(full_template_state) - set(BC_INIT_SUBTREES):
        _assert_state_equal(grafted_state[key], full_template_state[key], (key,))

    # The frozen trunk lives inside a grafted subtree, so its survival is the
    # load-bearing part of "nothing else moved".
    _assert_state_equal(
        _get(grafted_state, _TRUNK_PATH),
        _get(full_template_state, _TRUNK_PATH),
        _TRUNK_PATH,
    )
    FrozenResNet10TrunkExtractor(agent).validate_parameter_invariant(grafted)

    # And the artifact's declared ResNet digest is the asset this process
    # actually built the trunk from.
    verify_resnet_asset(load_bc_init_manifest(head_artifact), _RESNET_ASSET)


# --------------------------------------------------------------------------
# 2. The grafted weights actually reach the served action.
# --------------------------------------------------------------------------


def _action(value: Any) -> np.ndarray:
    return _numpy(value)


def _assert_action_contract(action: np.ndarray) -> None:
    assert action.shape == (7,)
    assert action.dtype == np.dtype(np.float32)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert float(action[-1]) in (-1.0, 0.0, 1.0)


def test_grafted_weights_change_the_served_action(agent, head_artifact):
    import jax

    from ur_env.learner import canonical_policy_observation

    observation = canonical_policy_observation()
    sample = deterministic_sample_action(agent)
    grafted = load_bc_init_params(head_artifact, agent)

    template_action = _action(
        sample(agent.state.params, observation, jax.random.PRNGKey(0), False)
    )
    grafted_action = _action(
        sample(grafted, observation, jax.random.PRNGKey(0), False)
    )

    # Same agent, same observation, same seed: the only difference is the
    # parameter tree.  If the hook dropped its ``params`` argument these two
    # would be bit-identical and an untrained policy would be serving.
    assert np.max(np.abs(grafted_action - template_action)) > 1e-6

    _assert_action_contract(template_action)
    _assert_action_contract(grafted_action)

    # BC evaluation serves the mode action on every step, and the production
    # actor sends deterministic=False on every step.  The wire flag must
    # therefore not reintroduce sampling noise.
    repeated = _action(
        sample(grafted, observation, jax.random.PRNGKey(0), True)
    )
    np.testing.assert_array_equal(repeated, grafted_action)


# --------------------------------------------------------------------------
# 3. A trunk BC touched loads, and is then rejected.
# --------------------------------------------------------------------------


def test_trunk_perturbation_loads_but_fails_the_frozen_invariant(
    agent, template_state, tmp_path
):
    from ur_env.learner import (
        FrozenResNet10TrunkExtractor,
        FrozenTrunkFeatureSchemaError,
    )

    artifact = _write_artifact(
        tmp_path / "trunk-artifact",
        _perturb(template_state, head=False, trunk=True),
    )

    # Structure is untouched, so the loader has nothing to object to: this
    # failure is only catchable downstream, which is why the trunk invariant
    # runs after the graft rather than instead of it.
    grafted = load_bc_init_params(artifact, agent)

    with pytest.raises(FrozenTrunkFeatureSchemaError):
        FrozenResNet10TrunkExtractor(agent).validate_parameter_invariant(grafted)


# --------------------------------------------------------------------------
# 4. The three fail-open shapes ``from_bytes`` would swallow.
# --------------------------------------------------------------------------


def _first_matching_path(state, predicate) -> tuple[str, ...]:
    for path in _leaf_paths(state):
        if predicate(path, _get(state, path)):
            return path
    raise AssertionError("no parameter leaf matched the mutation precondition")


def _with_extra_leaf(state: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(state)
    mutated[BC_INIT_SUBTREES[1]]["bc_init_bogus_extra"] = np.zeros(
        (3,), dtype=np.float32
    )
    return mutated


def _with_reshaped_leaf(state: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(state)
    path = _first_matching_path(
        mutated,
        lambda path, leaf: (
            path[0] == BC_INIT_SUBTREES[0]
            and not _in_trunk(path)
            and getattr(leaf, "ndim", 0) >= 2
        ),
    )
    _set(mutated, path, np.reshape(_get(mutated, path), (-1,)))
    return mutated


def _without_one_leaf(state: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(state)
    path = _first_matching_path(
        mutated,
        lambda path, leaf: len(path) >= 2
        and len(_get(mutated, path[:-1])) >= 2
        and not _in_trunk(path),
    )
    del _get(mutated, path[:-1])[path[-1]]
    return mutated


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_with_extra_leaf, id="extra-key"),
        pytest.param(_with_reshaped_leaf, id="reshaped-leaf"),
        pytest.param(_without_one_leaf, id="missing-leaf"),
    ],
)
def test_structurally_divergent_artifacts_are_refused(
    agent, template_state, tmp_path, mutate
):
    artifact = _write_artifact(
        tmp_path / "divergent-artifact", mutate(template_state)
    )

    with pytest.raises(BcInitError):
        load_bc_init_params(artifact, agent)


# --------------------------------------------------------------------------
# 5. Bytes that changed after the manifest was signed.
# --------------------------------------------------------------------------


def test_corrupted_parameter_bytes_are_refused(agent, head_state, tmp_path):
    artifact = _write_artifact(tmp_path / "corrupt-artifact", head_state)
    parameter_path = artifact / PARAMETER_FILENAME
    payload = bytearray(parameter_path.read_bytes())
    payload[-1] ^= 0xFF
    parameter_path.write_bytes(bytes(payload))

    with pytest.raises(BcInitError):
        load_bc_init_params(artifact, agent)
