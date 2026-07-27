"""One-time raw-demo to frozen-trunk feature-pool tests."""

from __future__ import annotations

import gc
import os
import sys
import weakref

import numpy as np
import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from ur_env.learner.config import (  # noqa: E402
    FROZEN_TRUNK_FEATURE_SHAPE,
    FROZEN_TRUNK_REPRESENTATION,
    LEARNER_AUGMENTATION,
)
from ur_env.learner.demo import DemoSidecar, LoadedDemos  # noqa: E402
from ur_env.learner.feature_demo import (  # noqa: E402
    FeatureDemoContractError,
    FrozenTrunkFeatureDemoPool,
    convert_loaded_demos_to_feature_pool,
    estimate_feature_demo_memory,
)


def _raw_observation(value: int) -> dict[str, np.ndarray]:
    return {
        "state": np.full((1, 19), value, dtype=np.float32),
        "cam1": np.full((1, 128, 128, 3), value, dtype=np.uint8),
        "cam2": np.full((1, 128, 128, 3), 255 - value, dtype=np.uint8),
    }


def _raw_transition(value: int) -> dict[str, object]:
    return {
        "observations": _raw_observation(value),
        "next_observations": _raw_observation(value + 1),
        "actions": np.array(
            [value / 100.0, -0.1, 0.2, 0.0, 0.0, 0.0, -1.0],
            dtype=np.float32,
        ),
        "rewards": np.float32(value % 2),
        "masks": np.float32(1 - value % 2),
        "grasp_penalty": np.float32(-0.02),
    }


def _loaded(*values: int) -> LoadedDemos:
    return LoadedDemos(
        transitions=tuple(_raw_transition(value) for value in values),
        sidecars=tuple(
            DemoSidecar(
                source_path=f"/demo/{value}.pkl",
                item_index=index,
                metadata={"episode_id": value, "success": bool(value % 2)},
            )
            for index, value in enumerate(values)
        ),
    )


class _FakeBatchedExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, observation):
        self.calls.append(tuple(observation["state"].shape))
        batch_size = int(observation["state"].shape[0])
        result = {"state": observation["state"].copy()}
        for key in ("cam1", "cam2"):
            values = observation[key][:, 0, 0, 0, 0].astype(np.float32)
            result[key] = np.broadcast_to(
                values[:, None, None, None, None],
                (batch_size, *FROZEN_TRUNK_FEATURE_SHAPE),
            ).copy()
        return result


def _assert_batches_equal(left, right) -> None:
    for tree_name in ("observations", "next_observations"):
        assert set(left[tree_name]) == {"state", "cam1", "cam2"}
        for key in ("state", "cam1", "cam2"):
            np.testing.assert_array_equal(
                left[tree_name][key], right[tree_name][key]
            )
    for key in ("actions", "rewards", "masks", "grasp_penalty"):
        np.testing.assert_array_equal(left[key], right[key])


def test_converts_in_chunks_preserves_provenance_and_samples_explicit_features():
    loaded = _loaded(2, 4, 6)
    first_extractor = _FakeBatchedExtractor()
    second_extractor = _FakeBatchedExtractor()

    first = convert_loaded_demos_to_feature_pool(
        loaded,
        feature_extractor=first_extractor,
        seed=17,
        extraction_batch_size=2,
    )
    second = convert_loaded_demos_to_feature_pool(
        loaded,
        feature_extractor=second_extractor,
        seed=17,
        extraction_batch_size=2,
    )

    assert len(first) == 3
    assert first.observation_representation == FROZEN_TRUNK_REPRESENTATION
    assert first.augmentation == LEARNER_AUGMENTATION == "none"
    assert first.feature_shape == FROZEN_TRUNK_FEATURE_SHAPE
    # Two chunks, with one call for current and one for next observations.
    assert first_extractor.calls == [(2, 1, 19), (2, 1, 19), (1, 1, 19), (1, 1, 19)]
    assert first.sidecars == loaded.sidecars
    assert first.sidecars is not loaded.sidecars
    with pytest.raises(TypeError):
        first.sidecars[0].metadata["episode_id"] = 999

    first_batch = first.sample(8)
    second_batch = second.sample(8)
    _assert_batches_equal(first_batch, second_batch)
    assert first_batch["observations"]["cam1"].shape == (
        8,
        *FROZEN_TRUNK_FEATURE_SHAPE,
    )
    assert first_batch["observations"]["cam1"].dtype == np.float32
    assert first_batch["next_observations"]["cam2"].shape == (
        8,
        *FROZEN_TRUNK_FEATURE_SHAPE,
    )
    current_values = first_batch["observations"]["state"][:, 0, 0]
    next_values = first_batch["next_observations"]["state"][:, 0, 0]
    np.testing.assert_array_equal(next_values, current_values + 1)
    np.testing.assert_array_equal(
        first_batch["observations"]["cam1"][:, 0, 0, 0, 0],
        current_values,
    )
    np.testing.assert_array_equal(
        first_batch["next_observations"]["cam1"][:, 0, 0, 0, 0],
        next_values,
    )
    assert set(first_batch) == {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "masks",
        "grasp_penalty",
    }
    with pytest.raises(ValueError, match="packed-image"):
        first.sample(1, packed=True)


def test_pool_retains_no_raw_images_or_loaded_demo_references():
    def build_pool_and_raw_refs():
        loaded = _loaded(7)
        raw_refs = tuple(
            weakref.ref(loaded.transitions[0][tree_name][image_key])
            for tree_name in ("observations", "next_observations")
            for image_key in ("cam1", "cam2")
        )
        pool = FrozenTrunkFeatureDemoPool(
            loaded,
            feature_extractor=_FakeBatchedExtractor(),
            seed=3,
        )
        return pool, raw_refs

    pool, raw_refs = build_pool_and_raw_refs()
    gc.collect()

    assert all(reference() is None for reference in raw_refs)
    batch = pool.sample(1)
    assert all(
        array.dtype == np.float32
        for tree_name in ("observations", "next_observations")
        for array in batch[tree_name].values()
    )
    assert all(
        tuple(array.shape[-4:]) != (1, 128, 128, 3)
        for tree_name in ("observations", "next_observations")
        for array in batch[tree_name].values()
    )


def test_memory_estimate_matches_the_exact_pool_allocation():
    estimate = estimate_feature_demo_memory(2)
    pool = FrozenTrunkFeatureDemoPool(
        _loaded(1, 2),
        feature_extractor=_FakeBatchedExtractor(),
    )

    expected_camera_bytes = 2 * 2 * 2 * np.prod(FROZEN_TRUNK_FEATURE_SHAPE) * 4
    assert estimate.camera_bytes == expected_camera_bytes
    assert estimate.state_bytes == 2 * 2 * 19 * 4
    assert estimate.action_bytes == 2 * 7 * 4
    assert estimate.scalar_bytes == 2 * 3 * 4
    assert estimate.total_bytes == pool.storage_nbytes
    assert pool.memory_estimate == estimate


def test_empty_pool_never_calls_extractor_and_cannot_sample():
    class _MustNotRun:
        def __call__(self, observation):
            raise AssertionError("extractor must not run")

    pool = FrozenTrunkFeatureDemoPool(
        LoadedDemos(transitions=(), sidecars=()),
        feature_extractor=_MustNotRun(),
    )

    assert len(pool) == 0
    assert pool.storage_nbytes == 0
    with pytest.raises(ValueError, match="empty"):
        pool.sample(1)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda output: output.pop("cam2"), "exactly state, cam1, cam2"),
        (
            lambda output: output.update(
                state=output["state"].astype(np.float64)
            ),
            "state must have dtype float32",
        ),
        (
            lambda output: output["state"].__setitem__((0, 0, 0), 99.0),
            "exactly equal",
        ),
        (
            lambda output: output.update(
                cam1=np.zeros((1, 4, 4, 512), np.float32)
            ),
            "cam1 must have shape",
        ),
        (
            lambda output: output.update(
                cam1=output["cam1"].astype(np.float64)
            ),
            "cam1 must have dtype float32",
        ),
        (
            lambda output: output["cam1"].__setitem__((0, 0, 0, 0, 0), np.nan),
            "non-finite",
        ),
    ],
)
def test_extractor_output_schema_fails_closed(mutate, match):
    class _MalformedExtractor(_FakeBatchedExtractor):
        def __call__(self, observation):
            output = super().__call__(observation)
            mutate(output)
            return output

    with pytest.raises(FeatureDemoContractError, match=match):
        FrozenTrunkFeatureDemoPool(
            _loaded(1), feature_extractor=_MalformedExtractor()
        )


def test_loaded_demo_type_length_sidecar_and_transition_alignment_fail_closed():
    extractor = _FakeBatchedExtractor()
    with pytest.raises(TypeError, match="LoadedDemos"):
        FrozenTrunkFeatureDemoPool(object(), feature_extractor=extractor)
    with pytest.raises(FeatureDemoContractError, match="equal length"):
        FrozenTrunkFeatureDemoPool(
            LoadedDemos(transitions=(_raw_transition(1),), sidecars=()),
            feature_extractor=extractor,
        )
    with pytest.raises(FeatureDemoContractError, match="DemoSidecar"):
        FrozenTrunkFeatureDemoPool(
            LoadedDemos(
                transitions=(_raw_transition(1),),
                sidecars=(object(),),
            ),
            feature_extractor=extractor,
        )
    malformed = _raw_transition(1)
    malformed["actions"] = np.zeros(7, np.float64)
    with pytest.raises(FeatureDemoContractError, match="dtype float32"):
        FrozenTrunkFeatureDemoPool(
            LoadedDemos(
                transitions=(malformed,),
                sidecars=(DemoSidecar("demo", 0, {}),),
            ),
            feature_extractor=extractor,
        )


@pytest.mark.parametrize(
    ("argument", "value", "match"),
    [
        ("seed", True, "integer"),
        ("seed", -1, "non-negative"),
        ("extraction_batch_size", 0, "positive"),
    ],
)
def test_constructor_integer_contracts(argument, value, match):
    kwargs = {argument: value}
    with pytest.raises(ValueError, match=match):
        FrozenTrunkFeatureDemoPool(
            _loaded(1),
            feature_extractor=_FakeBatchedExtractor(),
            **kwargs,
        )

    with pytest.raises(ValueError, match="integer"):
        estimate_feature_demo_memory(True)
