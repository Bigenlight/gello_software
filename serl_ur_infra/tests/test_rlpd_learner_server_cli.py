"""Dependency-light learner-server CLI and strict-construction checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_HERE = Path(__file__).resolve().parent
_INFRA = _HERE.parent
_SCRIPT = _INFRA / "scripts" / "run_rlpd_learner_server.py"
sys.path.insert(0, str(_INFRA))

_SPEC = importlib.util.spec_from_file_location(
    "run_rlpd_learner_server_test", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

from ur_env.learner.demo import SYNTHETIC_ACCEPTANCE_ONLY_KEY  # noqa: E402


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--classifier-checkpoint",
        str(tmp_path / "classifier"),
        "--demo-path",
        str(tmp_path / "demo.pkl"),
        "--checkpoint-root",
        str(tmp_path / "checkpoints"),
        "--require-jax-backend",
        "cpu",
    ]


def test_cli_defaults_to_loopback_and_has_no_penalty_escape_hatch(tmp_path):
    args = _MODULE._parse_args(_required_args(tmp_path))

    assert args.host == "127.0.0.1"
    assert args.port == 50053
    assert args.require_jax_backend == "cpu"
    assert args.wandb_mode == "offline"
    assert args.grasp_penalty == pytest.approx(-0.02)
    assert args.feature_memory_reserve_gib == pytest.approx(2.0)
    assert args.demo_extraction_batch_size == 64
    assert args.synthetic_e2e is False
    assert not hasattr(args, "require_grasp_penalty")
    assert not hasattr(args, "learner_mode")
    _MODULE._validate_args(args)


def test_cli_rejects_nonloopback_and_backend_fallback(tmp_path):
    args = _MODULE._parse_args(
        [*_required_args(tmp_path), "--host", "0.0.0.0"]
    )
    with pytest.raises(ValueError, match="loopback-only"):
        _MODULE._validate_args(args)

    assert _MODULE._validate_jax_backend("CPU", "cpu") == "cpu"
    with pytest.raises(RuntimeError, match="'gpu' is required"):
        _MODULE._validate_jax_backend("cpu", "gpu")


def test_cli_requires_an_explicit_jax_backend(tmp_path):
    args = _required_args(tmp_path)
    backend_index = args.index("--require-jax-backend")
    del args[backend_index : backend_index + 2]

    with pytest.raises(SystemExit):
        _MODULE._parse_args(args)


def test_cli_bounded_target_must_be_checkpoint_aligned(tmp_path):
    args = [
        *_required_args(tmp_path),
        "--target-learner-step",
        "1",
    ]

    with pytest.raises(ValueError, match="checkpoint boundary"):
        _MODULE.main(args)


def test_cli_synthetic_e2e_is_bounded_and_keeps_real_learning_scale(tmp_path):
    args = _MODULE._parse_args(
        [
            *_required_args(tmp_path),
            "--synthetic-e2e",
            "--synthetic-run-id",
            "fake-run",
            "--target-learner-step",
            "1",
        ]
    )
    _MODULE._validate_args(args)
    config = _MODULE._learner_config(args)

    assert config.batch_size == 256
    assert config.training_starts == 100
    assert config.cta_ratio == 2
    assert config.publish_period == 1
    assert config.checkpoint_period == 1

    missing_target = _MODULE._parse_args(
        [
            *_required_args(tmp_path),
            "--synthetic-e2e",
            "--synthetic-run-id",
            "fake-run",
        ]
    )
    with pytest.raises(ValueError, match=r"target_learner_step in \[1, 10\]"):
        _MODULE._validate_args(missing_target)

    small_replay = _MODULE._parse_args(
        [
            *_required_args(tmp_path),
            "--synthetic-e2e",
            "--synthetic-run-id",
            "fake-run",
            "--target-learner-step",
            "1",
            "--replay-capacity",
            "99",
        ]
    )
    with pytest.raises(ValueError, match="training_starts"):
        _MODULE._validate_args(small_replay)


def test_cli_synthetic_e2e_and_dry_run_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        _MODULE._parse_args(
            [*_required_args(tmp_path), "--synthetic-e2e", "--dry-run"]
        )


def test_cli_synthetic_e2e_requires_explicit_run_and_deadline(tmp_path):
    missing_run = _MODULE._parse_args(
        [
            *_required_args(tmp_path),
            "--synthetic-e2e",
            "--target-learner-step",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="synthetic_run_id"):
        _MODULE._validate_args(missing_run)

    bad_timeout = _MODULE._parse_args(
        [
            *_required_args(tmp_path),
            "--synthetic-e2e",
            "--synthetic-run-id",
            "fake-run",
            "--target-learner-step",
            "1",
            "--synthetic-timeout-s",
            "0.5",
        ]
    )
    with pytest.raises(ValueError, match="synthetic_timeout_s"):
        _MODULE._validate_args(bad_timeout)


def test_synthetic_resume_target_requires_one_new_learner_step(tmp_path):
    args = _MODULE._parse_args(
        [
            *_required_args(tmp_path),
            "--synthetic-e2e",
            "--synthetic-run-id",
            "resume-run",
            "--target-learner-step",
            "2",
        ]
    )
    _MODULE._validate_args(args)
    _MODULE._validate_synthetic_progress_target(
        args, SimpleNamespace(learner_step=1)
    )
    with pytest.raises(ValueError, match="exactly one step"):
        _MODULE._validate_synthetic_progress_target(
            args, SimpleNamespace(learner_step=2)
        )


def test_cli_main_constructs_feature_ingress_from_restored_agent_contract(
    tmp_path,
    monkeypatch,
):
    captured = {}
    classifier_kwargs = {}
    (tmp_path / "demo.pkl").write_bytes(b"synthetic-demo")

    class StopAfterIngress(RuntimeError):
        pass

    def feature_replay_ingress(**kwargs):
        captured.update(kwargs)
        raise StopAfterIngress

    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(default_backend=lambda: "cpu"),
    )
    monkeypatch.setattr(
        _MODULE,
        "validate_learner_dependencies",
        lambda **kwargs: {
            "jax": "0.5.3",
            "jaxlib": "0.5.3",
            "flax": "0.10.5",
            "distrax": "0.1.5",
            "tensorflow_probability": "0.25.0",
        },
    )
    monkeypatch.setattr(_MODULE, "configure_flax_local_io", lambda: None)
    monkeypatch.setattr(
        _MODULE,
        "load_demo_pickles",
        lambda paths: SimpleNamespace(
            transitions=({"demo": True, "grasp_penalty": -0.02},),
            sidecars=(
                SimpleNamespace(
                    source_path="fake.pkl",
                    item_index=0,
                    metadata={SYNTHETIC_ACCEPTANCE_ONLY_KEY: True},
                ),
            ),
        ),
    )
    agent = SimpleNamespace(state=SimpleNamespace(params=object()))
    monkeypatch.setattr(
        _MODULE, "create_frozen_trunk_feature_agent", lambda **kwargs: agent
    )
    monkeypatch.setattr(
        _MODULE,
        "LearnerFingerprint",
        SimpleNamespace(create=lambda **kwargs: object()),
    )
    monkeypatch.setattr(
        _MODULE,
        "prepare_learner_state",
        lambda **kwargs: SimpleNamespace(agent=agent),
    )
    extractor = SimpleNamespace(
        validate_parameter_invariant=lambda params: None,
        validate_agent_invariant=lambda candidate: None,
        repin_target_trunk=lambda candidate: candidate,
    )
    monkeypatch.setattr(
        _MODULE,
        "FrozenResNet10TrunkExtractor",
        lambda *args, **kwargs: extractor,
    )
    demo_bytes = _MODULE.estimate_feature_demo_memory(1).total_bytes
    monkeypatch.setattr(
        _MODULE,
        "convert_loaded_demos_to_feature_pool",
        lambda *args, **kwargs: SimpleNamespace(storage_nbytes=demo_bytes),
    )
    monkeypatch.setattr(
        _MODULE,
        "_preflight_combined_feature_memory",
        lambda **kwargs: 32 * 1024**3,
    )
    def reward_classifier_runtime(**kwargs):
        classifier_kwargs.update(kwargs)
        return SimpleNamespace(
            checkpoint_sha256="c" * 64,
            threshold=0.85,
            reward_model_id="test-classifier",
        )

    monkeypatch.setattr(
        _MODULE, "RewardClassifierRuntime", reward_classifier_runtime
    )
    monkeypatch.setattr(
        _MODULE, "FeatureReplayIngress", feature_replay_ingress
    )

    args = [
        *_required_args(tmp_path),
        "--resnet-source",
        str(tmp_path / "resnet10.pkl"),
        "--resnet-cache",
        str(tmp_path / "resnet-cache.pkl"),
        "--wandb-mode",
        "disabled",
        "--dry-run",
    ]
    with pytest.raises(StopAfterIngress):
        _MODULE.main(args)

    assert captured["feature_extractor"] is extractor
    assert captured["expected_grasp_penalty"] == pytest.approx(-0.02)
    assert captured["memory_reserve_bytes"] == 2 * 1024**3
    assert "learner_mode" not in captured
    assert "hil_serl_root" not in captured
    assert classifier_kwargs["resnet_source_path"] == (
        tmp_path / "resnet10.pkl"
    ).resolve()
    assert classifier_kwargs["resnet_cache_path"] == str(
        tmp_path / "resnet-cache.pkl"
    )


@pytest.mark.parametrize("value", ["nan", "inf", "0.01"])
def test_cli_rejects_invalid_grasp_penalty(tmp_path, value):
    args = _MODULE._parse_args(
        [*_required_args(tmp_path), "--grasp-penalty", value]
    )
    with pytest.raises(ValueError, match="grasp_penalty"):
        _MODULE._validate_args(args)


def test_cli_rejects_offline_demo_with_different_penalty():
    demos = SimpleNamespace(
        transitions=({"grasp_penalty": -0.07},),
    )
    with pytest.raises(ValueError, match="configured penalty -0.02"):
        _MODULE._validate_demo_grasp_penalty(demos, expected=-0.02)


def test_cli_rejects_synthetic_demo_before_real_serving(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "demo.pkl").write_bytes(b"synthetic-demo")
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(default_backend=lambda: "cpu"),
    )
    monkeypatch.setattr(
        _MODULE, "validate_learner_dependencies", lambda **kwargs: {}
    )
    monkeypatch.setattr(_MODULE, "configure_flax_local_io", lambda: None)
    synthetic = SimpleNamespace(
        source_path="acceptance-fake.pkl",
        item_index=1,
        metadata={SYNTHETIC_ACCEPTANCE_ONLY_KEY: True},
    )
    monkeypatch.setattr(
        _MODULE,
        "load_demo_pickles",
        lambda paths: SimpleNamespace(
            transitions=({"demo": True},),
            sidecars=(synthetic,),
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "create_frozen_trunk_feature_agent",
        lambda **kwargs: pytest.fail("agent must not be allocated"),
    )

    with pytest.raises(ValueError, match=r"only with --dry-run"):
        _MODULE.main(
            [
                *_required_args(tmp_path),
                "--wandb-mode",
                "disabled",
            ]
        )


def test_synthetic_e2e_requires_an_all_synthetic_demo_pool():
    synthetic = SimpleNamespace(
        source_path="fake.pkl",
        item_index=0,
        metadata={SYNTHETIC_ACCEPTANCE_ONLY_KEY: True},
    )
    real = SimpleNamespace(
        source_path="real.pkl",
        item_index=1,
        metadata={},
    )

    assert _MODULE._validate_demo_serving_scope(
        SimpleNamespace(sidecars=(synthetic,)),
        dry_run=False,
        synthetic_e2e=True,
    ) == 1
    with pytest.raises(ValueError, match="every offline demo item"):
        _MODULE._validate_demo_serving_scope(
            SimpleNamespace(sidecars=(synthetic, real)),
            dry_run=False,
            synthetic_e2e=True,
        )
    with pytest.raises(ValueError, match="every offline demo item"):
        _MODULE._validate_demo_serving_scope(
            SimpleNamespace(sidecars=(real,)),
            dry_run=False,
            synthetic_e2e=True,
        )


def test_combined_feature_memory_preflight_accounts_for_demo_and_reserve():
    assert _MODULE._preflight_combined_feature_memory(
        replay_bytes=10,
        demo_bytes=20,
        reserve_bytes=30,
        available_bytes=60,
    ) == 60
    with pytest.raises(
        _MODULE.FeatureReplayMemoryError, match="persistent tensors"
    ):
        _MODULE._preflight_combined_feature_memory(
            replay_bytes=10,
            demo_bytes=20,
            reserve_bytes=30,
            available_bytes=59,
        )
