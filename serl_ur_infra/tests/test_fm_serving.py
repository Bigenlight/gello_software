"""The pure-numpy half of FM serving, on the interpreter that actually runs it.

``ur_env.fm_serving`` is imported by the actor-side tooling, where no jax and
no flax exist.  Everything asserted here is therefore numpy and constants:
what ``first_action_from_chunk`` accepts out of a sampled chunk, what it
refuses, and the three wire identities the actor and the server have to agree
on byte-for-byte.  The jax half -- building the policy, integrating the flow,
reading an artifact off disk -- lives in ``test_actual_fm_serving.py`` and only
runs on the hilserl interpreter.

Two conventions this file keeps deliberately:

* **Rejections are asserted by exception TYPE only.**  ``FmServingError`` is the
  contract; the message inside it is a diagnostic for a human reading a failed
  startup and may be reworded at any time.
* **Each malformed chunk carries exactly one defect.**  A chunk that was both
  out of range *and* the wrong shape could pass its test for the wrong reason,
  so the fixtures below start from a valid chunk and break one thing.

The constants are asserted against their literal strings rather than against
themselves.  They are a handshake -- the FM artifact writer, this loader, and
the server's ``GetServerInfo`` reply all have to spell them identically -- so a
test that read the value back out of the module would agree with any rename and
prove nothing.

Run with::

    cd /home/laptop3/gello_software
    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_fm_serving.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pytest


_INFRA = Path(__file__).resolve().parents[1]
if str(_INFRA) not in sys.path:
    sys.path.insert(0, str(_INFRA))

from ur_env.fm_serving import (  # noqa: E402
    FM_ARTIFACT_FORMAT,
    FM_MODEL_ID,
    FM_REWARD_MODEL_ID,
    FmServedPolicy,
    FmServingError,
    first_action_from_chunk,
)


_FM_SERVING_SOURCE = _INFRA / "ur_env" / "fm_serving.py"

#: Horizon of the synthetic chunks below.  Small, and >1 so that "the FIRST
#: action" is a distinguishable claim.
HORIZON = 4


# --------------------------------------------------------------------------
# Chunk fixtures: one valid shape, then one defect at a time.
# --------------------------------------------------------------------------


def valid_chunk(horizon: int = HORIZON) -> np.ndarray:
    """A ``(horizon, 7)`` chunk whose rows all differ from one another."""

    rows = []
    for index in range(horizon):
        scale = (index + 1) / (horizon + 1)
        continuous = np.linspace(-0.9, 0.9, 6, dtype=np.float32) * np.float32(scale)
        gripper = np.float32(1.0 if index % 2 == 0 else -1.0)
        rows.append(np.concatenate([continuous, [gripper]]).astype(np.float32))
    chunk = np.stack(rows).astype(np.float32)
    assert chunk.shape == (horizon, 7)
    return chunk


def _broken(mutate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return mutate(valid_chunk())


def _nan_continuous(chunk: np.ndarray) -> np.ndarray:
    chunk[0, 2] = np.nan
    return chunk


def _nan_gripper(chunk: np.ndarray) -> np.ndarray:
    chunk[0, 6] = np.nan
    return chunk


def _out_of_range_high(chunk: np.ndarray) -> np.ndarray:
    chunk[0, 1] = np.float32(1.5)
    return chunk


def _out_of_range_low(chunk: np.ndarray) -> np.ndarray:
    chunk[0, 4] = np.float32(-1.0001)
    return chunk


def _fractional_gripper(chunk: np.ndarray) -> np.ndarray:
    chunk[0, 6] = np.float32(0.5)
    return chunk


def _short_last_dim(chunk: np.ndarray) -> np.ndarray:
    return chunk[:, :6]


def _long_last_dim(chunk: np.ndarray) -> np.ndarray:
    return np.concatenate([chunk, chunk[:, :1]], axis=1)


def _rank_one(chunk: np.ndarray) -> np.ndarray:
    return chunk[0]


def _rank_four(chunk: np.ndarray) -> np.ndarray:
    return chunk[None, None]


def _two_batch_rows(chunk: np.ndarray) -> np.ndarray:
    """Right rank, wrong batch: only one action can come back per call."""

    return np.stack([chunk, chunk])


# --------------------------------------------------------------------------
# 1. What a valid chunk yields.
# --------------------------------------------------------------------------


def test_the_served_action_is_the_first_row_of_the_chunk():
    chunk = valid_chunk()
    action = first_action_from_chunk(chunk)

    assert isinstance(action, np.ndarray)
    assert action.shape == (7,)
    assert action.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(action, chunk[0])
    # The rows differ, so "first" is load-bearing rather than incidentally true.
    assert not np.array_equal(action, chunk[1])


def test_a_batch_of_one_chunk_is_accepted_and_yields_the_same_action():
    chunk = valid_chunk()

    np.testing.assert_array_equal(
        first_action_from_chunk(chunk[None]), first_action_from_chunk(chunk)
    )


def test_a_float64_chunk_is_served_as_float32():
    """The sampler hands back float32; anything wider is narrowed, not refused.

    The wire packs float32 and the action validator downstream re-asserts that
    dtype, so a float64 chunk arriving from numpy arithmetic must not become a
    float64 action that only fails later.
    """

    chunk = valid_chunk().astype(np.float64)
    action = first_action_from_chunk(chunk)

    assert action.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(action, chunk[0].astype(np.float32))


@pytest.mark.parametrize("gripper", [-1.0, 0.0, 1.0])
def test_the_three_legal_gripper_values_pass(gripper):
    """``-1``/``0``/``+1`` are the discrete gripper alphabet of the contract.

    The FM sampler only ever emits the two poles (it discretises on the sign),
    but ``0`` is legal on this wire and is what an unchanged gripper looks like
    coming from the rest of the stack, so the validator must not narrow it away.
    """

    chunk = valid_chunk()
    chunk[0, 6] = np.float32(gripper)

    action = first_action_from_chunk(chunk)

    assert float(action[6]) == pytest.approx(gripper)


def test_the_boundary_values_are_inside_the_range():
    chunk = valid_chunk()
    chunk[0, :6] = np.array([1.0, -1.0, 1.0, -1.0, 0.0, 0.0], dtype=np.float32)

    action = first_action_from_chunk(chunk)

    np.testing.assert_array_equal(action[:6], chunk[0, :6])


# --------------------------------------------------------------------------
# 2. What it refuses.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_nan_continuous, id="nan-continuous"),
        pytest.param(_nan_gripper, id="nan-gripper"),
        pytest.param(_out_of_range_high, id="above-one"),
        pytest.param(_out_of_range_low, id="below-minus-one"),
        pytest.param(_fractional_gripper, id="fractional-gripper"),
        pytest.param(_short_last_dim, id="six-channels"),
        pytest.param(_long_last_dim, id="eight-channels"),
        pytest.param(_rank_one, id="rank-1"),
        pytest.param(_rank_four, id="rank-4"),
        pytest.param(_two_batch_rows, id="two-batch-rows"),
    ],
)
def test_malformed_chunks_are_refused(mutate):
    with pytest.raises(FmServingError):
        first_action_from_chunk(_broken(mutate))


def test_an_infinite_value_is_refused_like_a_nan():
    chunk = valid_chunk()
    chunk[0, 0] = np.float32(np.inf)

    with pytest.raises(FmServingError):
        first_action_from_chunk(chunk)


# --------------------------------------------------------------------------
# 3. The wire identities.
# --------------------------------------------------------------------------


def test_the_wire_identities_are_the_pinned_literals():
    """Spelled out, because these three strings ARE the handshake.

    ``FM_ARTIFACT_FORMAT`` must match what the flow artifact writer stamps into
    ``manifest.json`` and what ``load_flow_artifact`` refuses to load without;
    the two model ids are what the actor pins against ``GetServerInfo``.
    """

    assert FM_ARTIFACT_FORMAT == "hil-serl-jax-flow-matching"
    assert FM_MODEL_ID == "fm-cube-in-cup-raw0731-h16-euler8-v1"
    assert FM_REWARD_MODEL_ID == "operator-manual-success-v1"


def test_the_identities_are_distinct_non_empty_strings():
    identities = (FM_ARTIFACT_FORMAT, FM_MODEL_ID, FM_REWARD_MODEL_ID)

    for value in identities:
        assert isinstance(value, str) and value
        assert value.strip() == value
    assert len(set(identities)) == len(identities)


def test_the_error_type_is_a_runtime_error():
    """Callers already catch ``RuntimeError`` around serving; stay inside it."""

    assert issubclass(FmServingError, RuntimeError)


def test_the_served_policy_class_never_advertises_prime_observation():
    """A ``prime_observation`` attribute would silently change the input.

    ``actor_network`` probes its sink with
    ``getattr(..., "prime_observation", None)`` and, when it finds one, hands
    the policy pre-encoded frozen-trunk features instead of raw pixels.  The FM
    policy runs its own feature extractor over the raw observation, so growing
    that attribute would feed it somebody else's tensors with no error anywhere.
    The instance-level check is in ``test_actual_fm_serving.py``; the class is
    checkable here, without jax.
    """

    assert not hasattr(FmServedPolicy, "prime_observation")


# --------------------------------------------------------------------------
# 4. Import weight.
# --------------------------------------------------------------------------


def test_module_scope_stays_free_of_jax():
    """Importing ``fm_serving`` must not drag jax or flax in.

    On the actor interpreter neither package is installed, so the module-level
    import at the top of this file already proves it by succeeding.  Executing a
    private second copy and watching ``sys.modules`` keeps the guarantee
    meaningful under the hilserl interpreter, where both exist and a module-scope
    import would otherwise go unnoticed.
    """

    spec = importlib.util.spec_from_file_location(
        "_fm_serving_import_weight_probe", _FM_SERVING_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    before = set(sys.modules)
    spec.loader.exec_module(module)
    imported_roots = {
        name.split(".", 1)[0] for name in set(sys.modules) - before
    }

    assert "jax" not in imported_roots
    assert "flax" not in imported_roots
    # The probe really did execute the module under test.
    assert module.FM_MODEL_ID == FM_MODEL_ID
