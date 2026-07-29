"""Dependency-light checks for receive-server deployment probes."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest


_SCRIPT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "scripts", "run_rlpd_receive_server.py"
    )
)
_SPEC = importlib.util.spec_from_file_location("run_rlpd_receive_server", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _batch(batch_size=2):
    return {
        "observations": {
            "cam1": np.zeros((batch_size, 2, 128, 128, 3), np.uint8),
            "cam2": np.zeros((batch_size, 2, 128, 128, 3), np.uint8),
            "state": np.zeros((batch_size, 1, 19), np.float32),
        },
        "actions": np.zeros((batch_size, 7), np.float32),
        "policy_actions": np.zeros((batch_size, 7), np.float32),
    }


def test_sample_probe_validates_only_shapes_and_returns_summary():
    summary = _MODULE._validate_sample_batch(_batch(), batch_size=2)

    assert summary == {
        "batch_size": 2,
        "cam1_shape": [2, 2, 128, 128, 3],
        "cam2_shape": [2, 2, 128, 128, 3],
        "state_shape": [2, 1, 19],
        "action_shape": [2, 7],
    }


def test_sample_probe_rejects_noncanonical_packed_batch():
    batch = _batch()
    batch["observations"]["state"] = np.zeros((2, 19), np.float32)

    with pytest.raises(RuntimeError, match="observation shapes mismatch"):
        _MODULE._validate_sample_batch(batch, batch_size=2)


def test_ipv6_loopback_bind_is_bracketed():
    assert _MODULE._grpc_bind_address("127.0.0.1", 50053) == "127.0.0.1:50053"
    assert _MODULE._grpc_bind_address("::1", 50053) == "[::1]:50053"


def test_required_jax_backend_fails_closed_on_cpu_fallback():
    assert _MODULE._validate_jax_backend("GPU", "gpu") == "gpu"
    assert _MODULE._validate_jax_backend("cpu", "any") == "cpu"

    with pytest.raises(RuntimeError, match="'gpu' is required"):
        _MODULE._validate_jax_backend("cpu", "gpu")


# --------------------------------------------------------------------------- #
# Reward contract pins                                                         #
# --------------------------------------------------------------------------- #
#: The digest of the RETIRED classifier checkpoint that scores 0% recall on the
#: current data domain.  Serving it does not fail -- it makes every reward 0
#: forever, silently -- so the pin must never regress to this value.
_RETIRED_ZERO_RECALL_SHA256 = (
    "e329986b0dc2051bdf1baf4437f47e20448ac4ca81f12e4748932fc860d7a997"
)


def test_default_checkpoint_sha_is_the_live_orbax_tree_not_the_retired_pin():
    sha = _MODULE.DEFAULT_CHECKPOINT_SHA256

    assert sha != _RETIRED_ZERO_RECALL_SHA256
    assert len(sha) == 64 and set(sha) <= set("0123456789abcdef")
    # Directory sha256 of classifier_ckpt/cube_in_cup_all3/checkpoint_150.
    # Recompute with ur_env.classifier_sidecar.directory_sha256 if the
    # checkpoint tree is ever restaged; see the constant's comment.
    assert sha == (
        "512b657530af0ad78b746d40fd09e561b33a2ea92dede83d096477599162846d"
    )


def test_default_reward_model_id_encodes_the_classifier_input_contract():
    from ur_env.classifier_sidecar import CLASSIFIER_INPUT_ID

    model_id = _MODULE.DEFAULT_REWARD_MODEL_ID

    # A pre-sidecar actor pinned the bare checkpoint name.  The served id must
    # differ from it so an old actor is rejected at the handshake rather than
    # scoring rewards from the wrong pixels for a whole session.
    assert model_id != "cube-in-cup-checkpoint-150"
    assert "ckpt150" in model_id
    assert "sidecar" in model_id
    assert CLASSIFIER_INPUT_ID.startswith("fullframe-jpeg-passthrough")


def _parse(monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_rlpd_receive_server.py", "--checkpoint", "/tmp/ckpt", *extra],
    )
    return _MODULE._parse_args()


def test_cli_defaults_pin_the_reward_contract_without_smoothing(monkeypatch):
    args = _parse(monkeypatch)

    # 1 == no smoothing.  Deliberate: it keeps the server's verdict identical
    # to the live classifier viewer, which reports raw per-frame probability.
    assert args.success_confirmations == 1
    assert args.expected_checkpoint_sha256 == _MODULE.DEFAULT_CHECKPOINT_SHA256
    assert args.reward_model_id == _MODULE.DEFAULT_REWARD_MODEL_ID
    # The threshold decision (FP/FN cost asymmetry) is independent of this
    # change and must not have moved.
    assert args.threshold == pytest.approx(0.2)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_success_confirmations(monkeypatch, value):
    _parse(monkeypatch, "--success-confirmations", value)

    # main() re-parses argv; it must refuse before it ever loads a checkpoint.
    with pytest.raises(ValueError, match="success_confirmations"):
        _MODULE.main()
