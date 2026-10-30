"""Regression tests for the classifier's verified local restore boundary."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


_HERE = Path(__file__).resolve().parent
_INFRA_ROOT = _HERE.parent
sys.path.insert(0, os.fspath(_INFRA_ROOT))

flax_io = pytest.importorskip("flax.io")
pytest.importorskip("jax")

from ur_env.learner import agent as learner_agent  # noqa: E402
from ur_env.compat import CompatibilityError, configure_flax_local_io  # noqa: E402
from ur_env.rlpd_receive_server import (  # noqa: E402
    RewardClassifierError,
    RewardClassifierRuntime,
)


def _install_fake_upstream(monkeypatch, load_classifier_func) -> None:
    launcher = ModuleType("serl_launcher")
    launcher.__path__ = []
    networks = ModuleType("serl_launcher.networks")
    networks.__path__ = []
    reward_classifier = ModuleType("serl_launcher.networks.reward_classifier")
    reward_classifier.load_classifier_func = load_classifier_func
    monkeypatch.setitem(sys.modules, "serl_launcher", launcher)
    monkeypatch.setitem(sys.modules, "serl_launcher.networks", networks)
    monkeypatch.setitem(
        sys.modules,
        "serl_launcher.networks.reward_classifier",
        reward_classifier,
    )


def _uninitialized_runtime(checkpoint: Path) -> RewardClassifierRuntime:
    runtime = object.__new__(RewardClassifierRuntime)
    runtime.checkpoint_path = os.fspath(checkpoint)
    return runtime


def test_flax_local_io_is_an_explicit_idempotent_process_contract(monkeypatch):
    monkeypatch.setattr(flax_io, "io_mode", flax_io.BackendMode.TF)

    configure_flax_local_io()
    configure_flax_local_io()

    assert flax_io.io_mode is flax_io.BackendMode.DEFAULT


def test_flax_local_io_contract_fails_closed_if_mode_does_not_change(monkeypatch):
    monkeypatch.setattr(flax_io, "io_mode", flax_io.BackendMode.TF)
    monkeypatch.setattr(flax_io, "set_mode", lambda mode: None)

    with pytest.raises(CompatibilityError, match="refused"):
        configure_flax_local_io()


def test_upstream_restore_forces_local_flax_io_after_resnet_verification(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint_150"
    checkpoint.write_bytes(b"local-flax-checkpoint")
    hil_serl_root = tmp_path / "hil-serl"
    expected_resnet_source = tmp_path / "explicit-resnet10.pkl"
    expected_resnet_cache = tmp_path / "explicit-cache.pkl"
    upstream_cache = Path("~/.serl/resnet10_params.pkl").expanduser().resolve()
    events: list[tuple[str, object]] = []

    def ensure_resnet10_cache(*, source_path, cache_path=None):
        destination = Path(
            cache_path or "~/.serl/resnet10_params.pkl"
        ).expanduser().resolve()
        events.append(("resnet", (Path(source_path).resolve(), destination)))
        return destination

    def load_classifier_func(*, key, sample, image_keys, checkpoint_path):
        del key, sample, image_keys
        events.append(("restore_mode", flax_io.io_mode))
        # This resolves to built-in local file I/O only in DEFAULT mode.  In
        # TF mode the annotation shim deliberately rejects gfile access.
        with flax_io.GFile(checkpoint_path, "rb") as stream:
            assert stream.read() == b"local-flax-checkpoint"
        return lambda observation: np.asarray(0.0, dtype=np.float32)

    monkeypatch.setattr(
        learner_agent, "ensure_resnet10_cache", ensure_resnet10_cache
    )
    _install_fake_upstream(monkeypatch, load_classifier_func)
    monkeypatch.setattr(flax_io, "io_mode", flax_io.BackendMode.TF)

    runtime = _uninitialized_runtime(checkpoint)
    classifier = runtime._upstream_loader(
        os.fspath(hil_serl_root),
        resnet_source_path=os.fspath(expected_resnet_source),
        resnet_cache_path=os.fspath(expected_resnet_cache),
    )(
        {"cam1": np.zeros((1, 1, 1, 3), dtype=np.uint8)}
    )

    assert float(classifier({})) == 0.0
    assert events == [
        (
            "resnet",
            (expected_resnet_source.resolve(), expected_resnet_cache.resolve()),
        ),
        (
            "resnet",
            (expected_resnet_cache.resolve(), upstream_cache),
        ),
        ("restore_mode", flax_io.BackendMode.DEFAULT),
    ]
    assert flax_io.io_mode is flax_io.BackendMode.DEFAULT


def test_resnet_verification_failure_prevents_upstream_restore(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint_150"
    checkpoint.write_bytes(b"unused")
    restore_called = False
    verification_request = None

    def reject_resnet(*, source_path, cache_path=None):
        nonlocal verification_request
        verification_request = (Path(source_path), cache_path)
        raise learner_agent.ResNetAssetError("bad immutable digest")

    def load_classifier_func(**kwargs):
        del kwargs
        nonlocal restore_called
        restore_called = True
        raise AssertionError("restore must not run")

    monkeypatch.setattr(learner_agent, "ensure_resnet10_cache", reject_resnet)
    _install_fake_upstream(monkeypatch, load_classifier_func)

    runtime = _uninitialized_runtime(checkpoint)
    loader = runtime._upstream_loader(os.fspath(tmp_path / "hil-serl"))
    with pytest.raises(
        RewardClassifierError,
        match="verified ResNet-10 setup failed: ResNetAssetError",
    ):
        loader({"cam1": np.zeros((1, 1, 1, 3), dtype=np.uint8)})

    assert restore_called is False
    assert verification_request == (
        tmp_path / "hil-serl" / "examples" / "experiments" / "resnet10_params.pkl",
        None,
    )
