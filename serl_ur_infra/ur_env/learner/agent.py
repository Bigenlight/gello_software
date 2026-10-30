"""Pinned upstream hybrid SAC construction and ResNet asset verification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping

import numpy as np

from ur_env.compat import configure_pure_python_protobuf
from ur_env.learner.config import LearnerConfig, RESNET10_SHA256
from ur_env.learner.policy import canonical_policy_observation


class LearnerDependencyError(RuntimeError):
    """The local Python environment does not match the learner lock."""


class ResNetAssetError(RuntimeError):
    """The pretrained ResNet asset is absent or has the wrong digest."""


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_hil_serl_root() -> Path:
    return _default_repo_root() / "third_party" / "hil-serl"


def default_resnet_source() -> Path:
    return (
        default_hil_serl_root()
        / "examples"
        / "experiments"
        / "resnet10_params.pkl"
    )


def file_sha256(path: os.PathLike[str] | str) -> str:
    file_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    digest = hashlib.sha256()
    with open(file_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_resnet10_asset(
    path: os.PathLike[str] | str,
    *,
    expected_sha256: str = RESNET10_SHA256,
) -> str:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ResNetAssetError(
            "ResNet-10 SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual}, path={path}"
        )
    return actual


def ensure_resnet10_cache(
    *,
    source_path: os.PathLike[str] | str | None = None,
    cache_path: os.PathLike[str] | str | None = None,
) -> Path:
    """Verify the repository asset and create, but never replace, its cache."""

    source = Path(source_path or default_resnet_source()).expanduser().resolve()
    verify_resnet10_asset(source)
    cache = Path(cache_path or "~/.serl/resnet10_params.pkl").expanduser()
    if cache.exists():
        if not cache.is_file():
            raise ResNetAssetError(f"ResNet cache is not a file: {cache}")
        verify_resnet10_asset(cache)
        return cache

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(source, "rb") as source_stream, open(cache, "xb") as cache_stream:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                cache_stream.write(chunk)
            cache_stream.flush()
            os.fsync(cache_stream.fileno())
    except FileExistsError:
        # A concurrent learner won creation.  It still has to satisfy the same
        # immutable digest contract.
        pass
    verify_resnet10_asset(cache)
    return cache


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", ""))


def validate_learner_dependencies(*, include_logging: bool = True) -> dict[str, str]:
    """Fail closed unless the process uses the versions validated locally."""

    configure_pure_python_protobuf()
    try:
        import distrax
        import flax
        import jax
        import jaxlib
        import tensorflow_probability
    except Exception as exc:
        raise LearnerDependencyError(
            f"learner dependency import failed: {type(exc).__name__}: {exc}"
        ) from exc
    modules = {
        "jax": jax,
        "jaxlib": jaxlib,
        "flax": flax,
        "distrax": distrax,
        "tensorflow_probability": tensorflow_probability,
    }
    expected = {
        "jax": "0.5.3",
        "jaxlib": "0.5.3",
        "flax": "0.10.5",
        "distrax": "0.1.5",
        "tensorflow_probability": "0.25.0",
    }
    if include_logging:
        try:
            import wandb
        except Exception as exc:
            raise LearnerDependencyError(f"W&B import failed: {exc}") from exc
        modules["wandb"] = wandb
        expected["wandb"] = "0.26.0"
    actual = {name: _version(module) for name, module in modules.items()}
    mismatches = {
        name: (expected[name], actual[name])
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatches:
        detail = ", ".join(
            f"{name}: expected {wanted}, got {got or '<unknown>'}"
            for name, (wanted, got) in mismatches.items()
        )
        raise LearnerDependencyError(f"learner dependency version mismatch: {detail}")
    return actual


def _install_hil_serl_path(root: os.PathLike[str] | str) -> Path:
    hil_serl_root = Path(root).expanduser().resolve()
    launcher_root = hil_serl_root / "serl_launcher"
    if not (launcher_root / "serl_launcher").is_dir():
        raise LearnerDependencyError(
            f"HIL-SERL launcher package not found under {hil_serl_root}"
        )
    value = str(launcher_root)
    if value not in sys.path:
        sys.path.insert(0, value)
    return hil_serl_root


def _batch_augmentation(image_keys: tuple[str, ...]) -> Any:
    import jax
    from serl_launcher.vision.data_augmentations import batched_random_crop

    def augment_observations(rng: Any, observations: Any) -> Any:
        result = observations
        for image_key in image_keys:
            result = result.copy(
                add_or_replace={
                    image_key: batched_random_crop(
                        result[image_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return result

    def augment_batch(batch: Any, rng: Any) -> Any:
        _, observation_rng, next_observation_rng = jax.random.split(rng, 3)
        return batch.copy(
            add_or_replace={
                "observations": augment_observations(
                    observation_rng, batch["observations"]
                ),
                "next_observations": augment_observations(
                    next_observation_rng, batch["next_observations"]
                ),
            }
        )

    return augment_batch


def _load_resnet10_params_from_file(
    agent: Any,
    image_keys: tuple[str, ...],
    cache_path: Path,
) -> Any:
    """Apply the verified asset without using upstream's download path."""

    import jax
    from flax.core import FrozenDict, freeze, unfreeze

    with open(cache_path, "rb") as stream:
        encoder_params = pickle.load(stream)
    if not jax.tree_util.tree_leaves(encoder_params):
        raise ResNetAssetError("ResNet-10 asset contains no parameter leaves")

    was_frozen = isinstance(agent.state.params, FrozenDict)
    new_params = unfreeze(agent.state.params) if was_frozen else agent.state.params
    total_replaced = 0
    for image_key in image_keys:
        try:
            target = new_params["modules_actor"]["encoder"][
                f"encoder_{image_key}"
            ]
            if "pretrained_encoder" in target:
                target = target["pretrained_encoder"]
        except Exception as exc:
            raise ResNetAssetError(
                f"agent is missing the pretrained encoder for {image_key!r}"
            ) from exc
        replaced = 0
        for key in tuple(target):
            if key in encoder_params:
                target[key] = encoder_params[key]
                replaced += 1
        total_replaced += replaced
    # Flax parameter sharing stores the common frozen encoder under the first
    # camera module; later camera wrappers may therefore contribute no unique
    # leaves even though they resolve to that same module at apply time.
    if not total_replaced:
        raise ResNetAssetError("ResNet-10 asset did not match the agent encoder")
    if was_frozen:
        new_params = freeze(new_params)
    return agent.replace(state=agent.state.replace(params=new_params))


def _create_legacy_raw_augmented_hybrid_sac_agent(
    *,
    config: LearnerConfig = LearnerConfig(),
    hil_serl_root: os.PathLike[str] | str | None = None,
    resnet_source_path: os.PathLike[str] | str | None = None,
    resnet_cache_path: os.PathLike[str] | str | None = None,
    validate_versions: bool = True,
) -> Any:
    """Construct upstream's raw/random-crop agent for equivalence tests only."""

    if validate_versions:
        validate_learner_dependencies(include_logging=False)
    root = _install_hil_serl_path(hil_serl_root or default_hil_serl_root())
    cache = ensure_resnet10_cache(
        source_path=resnet_source_path
        or root / "examples" / "experiments" / "resnet10_params.pkl",
        cache_path=resnet_cache_path,
    )

    import jax
    from jax import nn
    from serl_launcher.agents.continuous.sac_hybrid_single import (
        SACAgentHybridSingleArm,
    )
    from serl_launcher.utils import train_utils

    observations: Mapping[str, np.ndarray] = canonical_policy_observation()
    actions = np.zeros((7,), dtype=np.float32)
    original_loader = train_utils.load_resnet10_params

    def verified_loader(agent: Any, image_keys: Any = config.image_keys) -> Any:
        return _load_resnet10_params_from_file(
            agent, tuple(image_keys), cache
        )

    train_utils.load_resnet10_params = verified_loader
    try:
        return SACAgentHybridSingleArm.create_pixels(
            jax.random.PRNGKey(config.seed),
            observations,
            actions,
            encoder_type=config.encoder_type,
            use_proprio=True,
            image_keys=config.image_keys,
            policy_kwargs={
                "tanh_squash_distribution": True,
                "std_parameterization": "exp",
                "std_min": 1e-5,
                "std_max": 5,
            },
            critic_network_kwargs={
                "activations": nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            },
            grasp_critic_network_kwargs={
                "activations": nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            },
            policy_network_kwargs={
                "activations": nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            },
            temperature_init=1e-2,
            discount=config.discount,
            backup_entropy=False,
            critic_ensemble_size=2,
            critic_subsample_size=None,
            reward_bias=0.0,
            target_entropy=None,
            augmentation_function=_batch_augmentation(config.image_keys),
        )
    finally:
        train_utils.load_resnet10_params = original_loader


def create_hybrid_sac_agent(
    *,
    config: LearnerConfig = LearnerConfig(),
    hil_serl_root: os.PathLike[str] | str | None = None,
    resnet_source_path: os.PathLike[str] | str | None = None,
    resnet_cache_path: os.PathLike[str] | str | None = None,
    validate_versions: bool = True,
) -> Any:
    """Construct the production no-augmentation frozen-feature SAC agent.

    The old public name remains as a compatibility entry point, but it may no
    longer silently create an augmented raw-pixel learner under the feature
    config.  The legacy factory is private and used only by an architecture
    equivalence test.
    """

    from ur_env.learner.frozen_trunk import create_frozen_trunk_feature_agent

    return create_frozen_trunk_feature_agent(
        config=config,
        hil_serl_root=hil_serl_root,
        resnet_source_path=resnet_source_path,
        resnet_cache_path=resnet_cache_path,
        validate_versions=validate_versions,
    )
