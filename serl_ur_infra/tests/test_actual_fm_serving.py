"""Opt-in real-jax tests for serving actions out of a flow-matching policy.

These integrate the three pieces that only meet each other at runtime: the
feature extractor the actor's observation goes through, the Euler integrator in
``ur_env.learner.flow_matching``, and the ``(7,)`` action contract the wire
demands.  Nothing here builds the production ResNet-10 agent -- a tiny
``FlowMatchingConfig`` exercises the same code paths in a fraction of a second,
and the real trunk is proven elsewhere.  What this file is for instead:

* the served action really is the first action of a freshly integrated chunk,
  in range, with a gripper the discretiser left at a pole;
* the policy is *stochastic and seeded* -- two instances built with the same
  ``rng_seed`` agree exactly, so a run is reproducible, while the wire's
  ``deterministic`` flag is ignored (BC and SAC both honour it; FM cannot, and
  a flag that silently did nothing without a test would be a lie either way);
* a synthetic on-disk artifact round-trips through ``load_flow_artifact``,
  including the two corruptions its digests exist to catch.

The velocity head is zero-initialised, so a freshly-initialised model integrates
noise to (clipped) noise.  That is exactly what makes the seeding assertions
sharp here: with the parameters contributing nothing, any difference between
two actions came from the RNG, and any agreement came from the seed.

Run explicitly with::

    cd /home/laptop3/gello_software
    env RUN_HIL_SERL_ACTUAL_FM=1 JAX_PLATFORMS=cpu \\
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/hilserl/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_actual_fm_serving.py

Without ``RUN_HIL_SERL_ACTUAL_FM=1`` the whole file skips, so the default suite
pays only for collection.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HIL_SERL_ACTUAL_FM") != "1",
    reason="set RUN_HIL_SERL_ACTUAL_FM=1 to run the real flow-matching serving test",
)

_INFRA = Path(__file__).resolve().parents[1]
if str(_INFRA) not in sys.path:
    sys.path.insert(0, str(_INFRA))

# The env-var gate above is the primary one; this keeps the file honest on an
# interpreter that has no jax at all (the actor venv), where it reports as a
# skip rather than a collection error.
pytest.importorskip("jax")

from ur_env.fm_serving import (  # noqa: E402
    FM_ARTIFACT_FORMAT,
    FM_MODEL_ID,
    FmServedPolicy,
    first_action_from_chunk,
)
from ur_env.learner.flow_matching import (  # noqa: E402
    FlowMatchingConfig,
    FlowMatchingPolicy,
    sample_action_chunks,
    load_flow_artifact,
)


MANIFEST_FILENAME = "manifest.json"
COMPLETION_FILENAME = "completion.json"
PARAMETER_FILENAME = "flow_params_best.msgpack"

#: What the frozen ResNet-10 trunk emits for ONE canonical observation:
#: ``(T=1, 4, 4, 512)``, no batch axis.  The fake extractor below returns the
#: same shape so that whatever ``FmServedPolicy`` does to reach the model's
#: ``(B, 1, 4, 4, 512)`` input is exercised here rather than assumed.
FEATURE_SHAPE = (1, 4, 4, 512)
STATE_DIM = 19


def _small_config() -> FlowMatchingConfig:
    """Same architecture, ~4 orders of magnitude fewer parameters."""

    return FlowMatchingConfig(
        horizon=4,
        spatial_features=2,
        camera_bottleneck_dim=8,
        proprio_bottleneck_dim=8,
        hidden_dim=16,
        residual_blocks=1,
        time_embedding_dim=8,
        integration_steps=2,
    )


# --------------------------------------------------------------------------
# Fixtures: a small model, its parameters, and the observation the actor sends.
# --------------------------------------------------------------------------


def _model_observations(batch_size: int = 1) -> dict[str, Any]:
    import jax.numpy as jnp

    return {
        "cam1": jnp.zeros((batch_size, *FEATURE_SHAPE), dtype=jnp.float32),
        "cam2": jnp.zeros((batch_size, *FEATURE_SHAPE), dtype=jnp.float32),
        "state": jnp.zeros((batch_size, STATE_DIM), dtype=jnp.float32),
    }


@pytest.fixture(scope="module")
def config() -> FlowMatchingConfig:
    return _small_config()


@pytest.fixture(scope="module")
def model(config) -> FlowMatchingPolicy:
    return FlowMatchingPolicy(config)


@pytest.fixture(scope="module")
def params(model, config):
    import jax
    import jax.numpy as jnp

    return model.init(
        jax.random.PRNGKey(0),
        _model_observations(),
        jnp.zeros((1, config.horizon, config.action_dim), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )["params"]


class FakeFeatureExtractor:
    """Stand-in for ``FrozenResNet10TrunkExtractor``.

    Returns exactly what the real extractor returns for one canonical
    observation -- ``(1, 4, 4, 512)`` per camera plus the untouched ``(1, 19)``
    state -- and records what it was handed, so the tests can prove the raw
    observation reached it unmodified.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.last_observation: Any = None

    def __call__(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        self.calls += 1
        self.last_observation = observation
        state = np.asarray(observation["state"], dtype=np.float32)
        return {
            "cam1": np.zeros(FEATURE_SHAPE, dtype=np.float32),
            "cam2": np.zeros(FEATURE_SHAPE, dtype=np.float32),
            "state": state,
        }


def canonical_observation(value: int = 7) -> dict[str, np.ndarray]:
    """The raw observation shape the actor puts on the wire."""

    return {
        "state": np.zeros((1, STATE_DIM), dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), value, dtype=np.uint8),
    }


def make_policy(
    model,
    params,
    *,
    policy_version: int = 11,
    rng_seed: int = 0,
    integration_steps: int | None = 2,
    extractor: FakeFeatureExtractor | None = None,
) -> tuple[FmServedPolicy, FakeFeatureExtractor]:
    extractor = extractor or FakeFeatureExtractor()
    policy = FmServedPolicy(
        model,
        params,
        extractor,
        policy_version=policy_version,
        rng_seed=rng_seed,
        integration_steps=integration_steps,
    )
    return policy, extractor


def assert_action_contract(action: Any) -> np.ndarray:
    array = np.asarray(action)
    assert array.shape == (7,)
    assert array.dtype == np.dtype(np.float32)
    assert np.isfinite(array).all()
    assert np.all(array >= -1.0) and np.all(array <= 1.0)
    # The sampler discretises on the sign, so 0 is unreachable by construction.
    assert float(array[6]) in {-1.0, 1.0}
    return array


# --------------------------------------------------------------------------
# 1. The small model integrates, and its chunk feeds the numpy validator.
# --------------------------------------------------------------------------


def test_a_sampled_chunk_passes_the_serving_validator(model, params, config):
    """The two halves of this stack agree on what a chunk looks like.

    ``sample_action_chunks`` is the only producer of the tensor
    ``first_action_from_chunk`` consumes, so this is the seam between the jax
    module and the pure-numpy one.
    """

    import jax

    chunk = sample_action_chunks(
        model, params, _model_observations(), jax.random.PRNGKey(3)
    )
    assert chunk.shape == (1, config.horizon, config.action_dim)

    action = first_action_from_chunk(np.asarray(chunk))

    assert_action_contract(action)
    np.testing.assert_allclose(action, np.asarray(chunk)[0, 0], rtol=0, atol=0)


# --------------------------------------------------------------------------
# 2. Serving one action from a raw observation.
# --------------------------------------------------------------------------


def test_serving_returns_a_valid_action_and_the_pinned_version(model, params):
    policy, extractor = make_policy(model, params, policy_version=11)

    assert policy.call_count == 0
    observation = canonical_observation()

    action, version = policy(observation, False)

    assert_action_contract(action)
    assert version == 11
    assert isinstance(version, int) and not isinstance(version, bool)
    assert policy.call_count == 1
    # The raw observation went to the extractor untouched -- pixels, not
    # features, are what this policy's front end expects.
    assert extractor.calls == 1
    assert set(extractor.last_observation) == {"state", "cam1", "cam2"}
    assert np.asarray(extractor.last_observation["cam1"]).dtype == np.dtype(np.uint8)


def test_call_count_tracks_every_inference(model, params):
    policy, extractor = make_policy(model, params)
    observation = canonical_observation()

    for expected in (1, 2, 3):
        policy(observation, False)
        assert policy.call_count == expected
        assert extractor.calls == expected


def test_the_model_id_is_the_pinned_default(model, params):
    policy, _ = make_policy(model, params)

    assert policy.model_id == FM_MODEL_ID


def test_an_explicit_model_id_overrides_the_default(model, params):
    policy = FmServedPolicy(
        model,
        params,
        FakeFeatureExtractor(),
        policy_version=1,
        model_id="fm-some-other-run-v9",
    )

    assert policy.model_id == "fm-some-other-run-v9"


def test_the_served_policy_never_advertises_prime_observation(model, params):
    """``actor_network`` probes for this attribute and changes its input if it finds one.

    ``getattr(sink, "prime_observation", None)`` deciding to hand over
    pre-encoded frozen-trunk features would bypass this policy's own extractor
    with no error raised anywhere -- so its absence is a contract, not an
    implementation detail.
    """

    policy, _ = make_policy(model, params)

    assert not hasattr(policy, "prime_observation")


# --------------------------------------------------------------------------
# 3. Stochastic, but seeded; and the wire flag changes nothing.
# --------------------------------------------------------------------------


def _sequence(model, params, *, rng_seed: int, deterministic: bool, count: int = 3):
    policy, _ = make_policy(model, params, rng_seed=rng_seed)
    observation = canonical_observation()
    return [np.asarray(policy(observation, deterministic)[0]) for _ in range(count)]


def test_the_same_seed_reproduces_the_same_action_sequence(model, params):
    left = _sequence(model, params, rng_seed=0, deterministic=False)
    right = _sequence(model, params, rng_seed=0, deterministic=False)

    for index, (one, other) in enumerate(zip(left, right)):
        np.testing.assert_array_equal(one, other, err_msg=f"call {index}")


def test_serving_is_stochastic_across_calls_and_across_seeds(model, params):
    """A flow policy resamples its noise every call; nothing here is a mode."""

    same_seed = _sequence(model, params, rng_seed=0, deterministic=False)
    other_seed = _sequence(model, params, rng_seed=1, deterministic=False)

    assert any(
        not np.array_equal(same_seed[0], later) for later in same_seed[1:]
    ), "successive calls of one policy were identical"
    assert not np.array_equal(same_seed[0], other_seed[0])

    for action in (*same_seed, *other_seed):
        assert_action_contract(action)


def test_the_wire_deterministic_flag_is_ignored(model, params):
    """There is no mode to serve, so the flag must not fork behaviour.

    The production actor sends ``deterministic=False`` on every step; a flag
    that quietly changed the integrator would make an FM session depend on a
    caller detail no runbook mentions.
    """

    stochastic = _sequence(model, params, rng_seed=5, deterministic=False)
    flagged = _sequence(model, params, rng_seed=5, deterministic=True)

    for index, (one, other) in enumerate(zip(stochastic, flagged)):
        np.testing.assert_array_equal(one, other, err_msg=f"call {index}")


# --------------------------------------------------------------------------
# 4. The on-disk artifact round trip.
# --------------------------------------------------------------------------


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_artifact(
    directory: Path,
    config: FlowMatchingConfig,
    params: Any,
    *,
    artifact_format: str = FM_ARTIFACT_FORMAT,
) -> Path:
    """Write a complete, self-consistent flow artifact.

    ``completion.json`` is written last and signs the manifest bytes, exactly
    as a real writer must order it, so a mutation of the manifest that forgets
    to re-sign is a different defect from the one under test.
    """

    from flax import serialization

    directory.mkdir(parents=True, exist_ok=True)
    payload = serialization.to_bytes(params)
    (directory / PARAMETER_FILENAME).write_bytes(payload)

    manifest_path = directory / MANIFEST_FILENAME
    _write_json(
        manifest_path,
        {
            # Written from the fm_serving constant on purpose: the loader
            # accepting it is the proof that the two pins agree.
            "format": artifact_format,
            "model_id": FM_MODEL_ID,
            "model_config": config.document(),
            "parameter_files": {
                "best": {
                    "path": PARAMETER_FILENAME,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            },
        },
    )
    _write_json(
        directory / COMPLETION_FILENAME,
        {
            "complete": True,
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        },
    )
    return directory


def _leaves(tree: Any) -> list[np.ndarray]:
    import jax

    return [np.asarray(jax.device_get(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]


def test_an_artifact_round_trips_and_then_serves(tmp_path, config, params):
    artifact = write_artifact(tmp_path / "fm-artifact", config, params)

    loaded_model, loaded_params, manifest = load_flow_artifact(artifact)

    assert manifest["format"] == FM_ARTIFACT_FORMAT
    assert loaded_model.config == config

    restored = _leaves(loaded_params)
    original = _leaves(params)
    assert len(restored) == len(original) and restored
    for index, (one, other) in enumerate(zip(restored, original)):
        np.testing.assert_array_equal(one, other, err_msg=f"leaf {index}")

    # The point of the round trip: parameters that came off disk drive a real
    # served action, not just a structural comparison.
    from_disk, _ = make_policy(
        loaded_model, loaded_params, policy_version=3, rng_seed=0
    )
    action, version = from_disk(canonical_observation(), False)

    assert_action_contract(action)
    assert version == 3

    # Same seed, same observation, in-memory parameters: loading changed
    # nothing observable about what gets served.
    in_memory, _ = make_policy(loaded_model, params, policy_version=3, rng_seed=0)
    np.testing.assert_array_equal(
        np.asarray(
            make_policy(loaded_model, loaded_params, rng_seed=0)[0](
                canonical_observation(), False
            )[0]
        ),
        np.asarray(in_memory(canonical_observation(), False)[0]),
    )


def test_a_str_path_loads_like_a_path_object(tmp_path, config, params):
    artifact = write_artifact(tmp_path / "fm-artifact-str", config, params)

    assert load_flow_artifact(str(artifact))[2] == load_flow_artifact(artifact)[2]


def test_corrupted_parameter_bytes_are_refused(tmp_path, config, params):
    artifact = write_artifact(tmp_path / "fm-corrupt", config, params)
    parameter_path = artifact / PARAMETER_FILENAME
    payload = bytearray(parameter_path.read_bytes())
    payload[-1] ^= 0xFF
    parameter_path.write_bytes(bytes(payload))

    with pytest.raises(ValueError):
        load_flow_artifact(artifact)


def test_a_foreign_artifact_format_is_refused(tmp_path, config, params):
    """A BC or SAC artifact in an FM directory must not load as a flow model.

    The completion digest is re-signed over the edited manifest, so the format
    string is the ONLY defect here -- otherwise this would pass on the SHA-256
    check and prove nothing about the format gate.
    """

    artifact = write_artifact(
        tmp_path / "fm-foreign",
        config,
        params,
        artifact_format="hil-serl-bc-init",
    )

    with pytest.raises(ValueError):
        load_flow_artifact(artifact)


def test_an_incomplete_artifact_is_refused(tmp_path, config, params):
    artifact = write_artifact(tmp_path / "fm-incomplete", config, params)
    completion_path = artifact / COMPLETION_FILENAME
    document = json.loads(completion_path.read_text(encoding="utf-8"))
    document["complete"] = False
    _write_json(completion_path, document)

    with pytest.raises(ValueError):
        load_flow_artifact(artifact)
