"""Frozen ResNet-10 trunk features for the no-augmentation learner path.

The replay representation is the output of the pretrained ResNet trunk at its
existing ``stop_gradient`` boundary.  It deliberately does *not* include
pooling or the trainable spatial/Dense/LayerNorm visual head.

Heavy JAX/Flax/HIL-SERL imports remain inside factories so importing the actor
or receive-only server does not acquire learner dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache, partial
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ur_env.learner.agent import (
    _install_hil_serl_path,
    _load_resnet10_params_from_file,
    default_hil_serl_root,
    ensure_resnet10_cache,
    validate_learner_dependencies,
    verify_resnet10_asset,
)
from ur_env.learner.config import (
    FROZEN_TRUNK_FEATURE_SHAPE,
    FROZEN_TRUNK_MODEL_REVISION,
    FROZEN_TRUNK_REPRESENTATION,
    LEARNER_AUGMENTATION,
    LearnerConfig,
    RESNET10_SHA256,
)
from ur_env.learner.policy import canonical_policy_observation


FROZEN_TRUNK_FEATURE_CONTRACT_REVISION = FROZEN_TRUNK_REPRESENTATION
FROZEN_TRUNK_FEATURE_DTYPE = np.dtype(np.float32)
FROZEN_TRUNK_PARAMETER_PATH = (
    "modules_actor",
    "encoder",
    "encoder_cam1",
    "pretrained_encoder",
)
RAW_IMAGE_SHAPE = (1, 128, 128, 3)


@lru_cache(maxsize=1)
def _exact_tree_equal_function():
    """Build one compiled equality reduction reused by every invariant check."""

    import jax
    import jax.numpy as jnp

    @jax.jit
    def equal(candidate: Any, reference: Any) -> Any:
        comparisons = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(jnp.array_equal, candidate, reference)
        )
        return jnp.all(jnp.stack(comparisons))

    return equal


class FrozenTrunkFeatureSchemaError(ValueError):
    """A cached feature does not satisfy the immutable replay contract."""


@dataclass(frozen=True)
class FrozenTrunkFeatureContract:
    """Serializable identity of the feature cut point and tensor schema."""

    revision: str = FROZEN_TRUNK_FEATURE_CONTRACT_REVISION
    model_revision: str = FROZEN_TRUNK_MODEL_REVISION
    resnet10_sha256: str = RESNET10_SHA256
    image_keys: tuple[str, str] = ("cam1", "cam2")
    raw_image_shape: tuple[int, int, int, int] = RAW_IMAGE_SHAPE
    feature_shape: tuple[int, int, int, int] = FROZEN_TRUNK_FEATURE_SHAPE
    feature_dtype: str = "float32"
    cut_point: str = "pretrained_resnet10.stop_gradient"
    pixel_augmentation: str = LEARNER_AUGMENTATION
    downstream_head: str = (
        "SpatialLearnedEmbeddings8+Dropout0.1+Dense256+LayerNorm+tanh"
    )

    def document(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("image_keys", "raw_image_shape", "feature_shape"):
            value[key] = list(value[key])
        return value


FROZEN_TRUNK_CONTRACT = FrozenTrunkFeatureContract()


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in value.shape)
    except Exception as exc:
        raise FrozenTrunkFeatureSchemaError("value must be an array") from exc


def validate_frozen_trunk_feature(
    value: Any,
    *,
    name: str = "feature",
    batched: bool = False,
    copy: bool = False,
) -> np.ndarray:
    """Validate one camera feature without silently casting or reshaping it.

    A single canonical observation stores ``(T=1, 4, 4, 512)``.  A sampled
    learner batch stores ``(B, T=1, 4, 4, 512)``.
    """

    array = np.asarray(value)
    expected = FROZEN_TRUNK_FEATURE_SHAPE
    actual = _shape(array)
    if batched:
        if len(actual) != len(expected) + 1 or actual[0] <= 0:
            raise FrozenTrunkFeatureSchemaError(
                f"{name} must have shape (B, {', '.join(map(str, expected))})"
            )
        if actual[1:] != expected:
            raise FrozenTrunkFeatureSchemaError(
                f"{name} must have trailing shape {expected}, got {actual}"
            )
    elif actual != expected:
        raise FrozenTrunkFeatureSchemaError(
            f"{name} must have shape {expected}, got {actual}"
        )
    if array.dtype != FROZEN_TRUNK_FEATURE_DTYPE:
        raise FrozenTrunkFeatureSchemaError(
            f"{name} must have dtype float32, got {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise FrozenTrunkFeatureSchemaError(f"{name} contains a non-finite value")
    return array.copy() if copy else array


def validate_frozen_trunk_observation(
    observation: Mapping[str, Any],
    *,
    batched: bool = False,
    copy: bool = False,
    image_keys: Iterable[str] = ("cam1", "cam2"),
) -> dict[str, np.ndarray]:
    """Validate the complete state-plus-feature observation contract."""

    if not isinstance(observation, Mapping):
        raise FrozenTrunkFeatureSchemaError("observation must be a mapping")
    keys = tuple(image_keys)
    expected_keys = {"state", *keys}
    if set(observation) != expected_keys:
        missing = sorted(expected_keys - set(observation))
        extra = sorted(set(observation) - expected_keys)
        raise FrozenTrunkFeatureSchemaError(
            f"observation keys mismatch: missing={missing}, extra={extra}"
        )

    state = np.asarray(observation["state"])
    expected_state_shape = (1, 19)
    if batched:
        if (
            state.ndim != 3
            or state.shape[0] <= 0
            or tuple(state.shape[1:]) != expected_state_shape
        ):
            raise FrozenTrunkFeatureSchemaError(
                "state must have shape (B, 1, 19)"
            )
        batch_size = int(state.shape[0])
    else:
        if tuple(state.shape) != expected_state_shape:
            raise FrozenTrunkFeatureSchemaError(
                f"state must have shape {expected_state_shape}, got {state.shape}"
            )
        batch_size = None
    if state.dtype != np.dtype(np.float32):
        raise FrozenTrunkFeatureSchemaError(
            f"state must have dtype float32, got {state.dtype}"
        )
    if not np.isfinite(state).all():
        raise FrozenTrunkFeatureSchemaError("state contains a non-finite value")

    result = {"state": state.copy() if copy else state}
    for image_key in keys:
        feature = validate_frozen_trunk_feature(
            observation[image_key],
            name=image_key,
            batched=batched,
            copy=copy,
        )
        if batched and int(feature.shape[0]) != batch_size:
            raise FrozenTrunkFeatureSchemaError(
                f"{image_key} batch size does not match state"
            )
        result[image_key] = feature
    return result


@lru_cache(maxsize=1)
def _dual_input_encoder_type() -> type:
    """Define the Flax module lazily while preserving upstream parameter names."""

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from serl_launcher.vision.resnet_v1 import SpatialLearnedEmbeddings

    class DualInputFrozenTrunkEncoder(nn.Module):
        """Run the frozen trunk for pixels or accept its exact cached output."""

        pretrained_encoder: nn.Module
        num_spatial_blocks: int = 8
        bottleneck_dim: int = 256

        @nn.compact
        def __call__(
            self,
            observations: Any,
            encode: bool = True,
            train: bool = True,
        ) -> Any:
            del encode  # EncodingWrapper always passes True; shape is authoritative.
            trailing_shape = tuple(int(value) for value in observations.shape[-3:])
            dtype = np.dtype(observations.dtype)
            if trailing_shape == (128, 128, 3):
                if dtype != np.dtype(np.uint8):
                    raise FrozenTrunkFeatureSchemaError(
                        "raw camera input must have dtype uint8"
                    )
                x = self.pretrained_encoder(observations, train=train)
            elif trailing_shape == (4, 4, 512):
                if dtype != FROZEN_TRUNK_FEATURE_DTYPE:
                    raise FrozenTrunkFeatureSchemaError(
                        "cached trunk input must have dtype float32"
                    )
                # Mirror the upstream frozen trunk boundary for cached tensors.
                x = jax.lax.stop_gradient(observations)
            else:
                raise FrozenTrunkFeatureSchemaError(
                    "camera input must end in (128, 128, 3) pixels or "
                    "(4, 4, 512) frozen-trunk features"
                )

            height, width, channels = x.shape[-3:]
            x = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channels,
                num_features=self.num_spatial_blocks,
            )(x)
            x = nn.Dropout(0.1, deterministic=not train)(x)
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            return jnp.tanh(x)

    DualInputFrozenTrunkEncoder.__name__ = "DualInputFrozenTrunkEncoder"
    return DualInputFrozenTrunkEncoder


def create_frozen_trunk_feature_agent(
    *,
    config: LearnerConfig = LearnerConfig(),
    hil_serl_root: os.PathLike[str] | str | None = None,
    resnet_source_path: os.PathLike[str] | str | None = None,
    resnet_cache_path: os.PathLike[str] | str | None = None,
    validate_versions: bool = True,
) -> Any:
    """Create hybrid SAC with equivalent raw and cached-trunk input paths.

    The architecture after the feature cut remains the upstream trainable
    per-camera spatial/Dense/LayerNorm head.  Pixel augmentation is hard-off;
    passing cached features through a crop would violate their semantics.
    """

    if validate_versions:
        validate_learner_dependencies(include_logging=False)
    root = _install_hil_serl_path(hil_serl_root or default_hil_serl_root())
    source = Path(
        resnet_source_path
        or root / "examples" / "experiments" / "resnet10_params.pkl"
    ).expanduser().resolve()
    verify_resnet10_asset(source)
    cache = ensure_resnet10_cache(source_path=source, cache_path=resnet_cache_path)

    import jax
    from jax import nn as jax_nn
    from serl_launcher.agents.continuous.sac_hybrid_single import (
        SACAgentHybridSingleArm,
    )
    from serl_launcher.common.encoding import EncodingWrapper
    from serl_launcher.networks.actor_critic_nets import (
        Critic,
        GraspCritic,
        Policy,
        ensemblize,
    )
    from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
    from serl_launcher.networks.mlp import MLP
    from serl_launcher.vision.resnet_v1 import resnetv1_configs

    image_keys = tuple(config.image_keys)
    pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
        pre_pooling=True,
        name="pretrained_encoder",
    )
    encoder_type = _dual_input_encoder_type()
    camera_encoders = {
        image_key: encoder_type(
            pretrained_encoder=pretrained_encoder,
            num_spatial_blocks=8,
            bottleneck_dim=256,
            name=f"encoder_{image_key}",
        )
        for image_key in image_keys
    }
    encoder_def = EncodingWrapper(
        encoder=camera_encoders,
        use_proprio=True,
        enable_stacking=True,
        image_keys=image_keys,
    )

    critic_backbone = partial(
        MLP,
        hidden_dims=[256, 256],
        activations=jax_nn.tanh,
        use_layer_norm=True,
        activate_final=True,
    )
    critic_backbone = ensemblize(critic_backbone, 2)(name="critic_ensemble")
    critic_def = partial(
        Critic, encoder=encoder_def, network=critic_backbone
    )(name="critic")
    grasp_critic_def = partial(
        GraspCritic,
        encoder=encoder_def,
        network=MLP(
            hidden_dims=[256, 256],
            activations=jax_nn.tanh,
            use_layer_norm=True,
        ),
    )(name="grasp_critic")
    policy_def = Policy(
        encoder=encoder_def,
        network=MLP(
            hidden_dims=[256, 256],
            activations=jax_nn.tanh,
            use_layer_norm=True,
            activate_final=True,
        ),
        action_dim=6,
        tanh_squash_distribution=True,
        std_parameterization="exp",
        std_min=1e-5,
        std_max=5,
        name="actor",
    )
    temperature_def = GeqLagrangeMultiplier(
        init_value=1e-2,
        constraint_shape=(),
        constraint_type="geq",
        name="temperature",
    )

    observations: Mapping[str, np.ndarray] = canonical_policy_observation()
    actions = np.zeros((7,), dtype=np.float32)
    agent = SACAgentHybridSingleArm.create(
        jax.random.PRNGKey(config.seed),
        observations,
        actions,
        actor_def=policy_def,
        critic_def=critic_def,
        grasp_critic_def=grasp_critic_def,
        temperature_def=temperature_def,
        discount=config.discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        image_keys=image_keys,
        reward_bias=0.0,
        augmentation_function=None,
        feature_contract_revision=FROZEN_TRUNK_FEATURE_CONTRACT_REVISION,
        model_revision=FROZEN_TRUNK_MODEL_REVISION,
        feature_contract=FROZEN_TRUNK_CONTRACT.document(),
    )
    return _load_resnet10_params_from_file(agent, image_keys, cache)


class FrozenResNet10TrunkExtractor:
    """Apply the exact verified trunk weights used by a dual-input agent."""

    parameter_path = FROZEN_TRUNK_PARAMETER_PATH

    def __init__(
        self,
        agent: Any,
        *,
        resnet_asset_path: os.PathLike[str] | str | None = None,
        image_keys: Iterable[str] = ("cam1", "cam2"),
    ) -> None:
        root = default_hil_serl_root()
        _install_hil_serl_path(root)
        asset = Path(
            resnet_asset_path
            or root / "examples" / "experiments" / "resnet10_params.pkl"
        ).expanduser().resolve()
        self.resnet_sha256 = verify_resnet10_asset(asset)
        self.image_keys = tuple(image_keys)
        if self.image_keys != ("cam1", "cam2"):
            raise FrozenTrunkFeatureSchemaError(
                "image_keys must be exactly ('cam1', 'cam2')"
            )
        try:
            params = agent.state.params
            for key in self.parameter_path:
                params = params[key]
            self._params = params
        except Exception as exc:
            raise FrozenTrunkFeatureSchemaError(
                "agent does not contain the shared pretrained ResNet-10 trunk"
            ) from exc

        from serl_launcher.vision.resnet_v1 import resnetv1_configs

        self._trunk = resnetv1_configs["resnetv1-10-frozen"](
            pre_pooling=True
        )

    @property
    def parameter_reference(self) -> Any:
        """Return the immutable agent trunk reference without copying it."""

        return self._params

    @classmethod
    def _parameter_subtree(cls, params: Any) -> Any:
        target = params
        try:
            for key in cls.parameter_path:
                target = target[key]
        except Exception as exc:
            raise FrozenTrunkFeatureSchemaError(
                "parameters do not contain the frozen ResNet-10 trunk at "
                f"{cls.parameter_path}"
            ) from exc
        return target

    def validate_parameter_invariant(self, params: Any) -> None:
        """Reject any candidate whose cached-feature trunk changed exactly."""

        candidate = self._parameter_subtree(params)
        self._validate_subtree(candidate, name="online")

    def _validate_subtree(self, candidate: Any, *, name: str) -> None:
        """Compare one online/target trunk subtree with the trusted asset."""

        import jax

        candidate_leaves, candidate_tree = jax.tree_util.tree_flatten(candidate)
        reference_leaves, reference_tree = jax.tree_util.tree_flatten(self._params)
        if candidate_tree != reference_tree or len(candidate_leaves) != len(
            reference_leaves
        ):
            raise FrozenTrunkFeatureSchemaError(
                f"frozen ResNet-10 {name} trunk parameter structure changed"
            )
        for index, (actual, expected) in enumerate(
            zip(candidate_leaves, reference_leaves)
        ):
            if tuple(actual.shape) != tuple(expected.shape) or np.dtype(
                actual.dtype
            ) != np.dtype(expected.dtype):
                raise FrozenTrunkFeatureSchemaError(
                    f"frozen ResNet-10 {name} trunk parameter shape/dtype "
                    f"changed at leaf {index}"
                )
        if not candidate_leaves or not bool(
            np.asarray(
                jax.device_get(
                    _exact_tree_equal_function()(candidate, self._params)
                )
            )
        ):
            raise FrozenTrunkFeatureSchemaError(
                f"frozen ResNet-10 {name} trunk weights changed; cached "
                "replay features are no longer compatible"
            )

    def validate_agent_invariant(self, agent: Any) -> None:
        """Validate both checkpointed online and target trunk parameters."""

        try:
            state = agent.state
        except Exception as exc:
            raise FrozenTrunkFeatureSchemaError(
                "agent must expose checkpointable state"
            ) from exc
        self.validate_parameter_invariant(state.params)
        target = self._parameter_subtree(state.target_params)
        self._validate_subtree(target, name="target")

    @classmethod
    def _replace_parameter_subtree(cls, tree: Any, replacement: Any) -> Any:
        """Copy only mappings along the trunk path; share all array leaves."""

        from flax.core import FrozenDict

        def replace(node: Any, remaining: tuple[str, ...]) -> Any:
            if not remaining:
                return replacement
            key = remaining[0]
            if key not in node:
                raise FrozenTrunkFeatureSchemaError(
                    "target parameters do not contain the frozen trunk path"
                )
            child = replace(node[key], remaining[1:])
            if isinstance(node, FrozenDict):
                return node.copy(add_or_replace={key: child})
            if isinstance(node, Mapping):
                copied = node.copy()
                copied[key] = child
                return copied
            raise FrozenTrunkFeatureSchemaError(
                "frozen trunk parameter path crosses a non-mapping node"
            )

        return replace(tree, cls.parameter_path)

    def repin_target_trunk(self, agent: Any) -> Any:
        """Undo upstream Polyak arithmetic on the unused frozen target trunk."""

        target_params = self._replace_parameter_subtree(
            agent.state.target_params, self._params
        )
        return agent.replace(
            state=agent.state.replace(target_params=target_params)
        )

    @staticmethod
    def _validate_raw_observation(
        observation: Mapping[str, Any], image_keys: tuple[str, ...]
    ) -> tuple[dict[str, np.ndarray], bool]:
        if not isinstance(observation, Mapping):
            raise FrozenTrunkFeatureSchemaError("observation must be a mapping")
        expected_keys = {"state", *image_keys}
        if set(observation) != expected_keys:
            raise FrozenTrunkFeatureSchemaError(
                "raw observation keys must be exactly state, cam1, cam2"
            )
        state = np.asarray(observation["state"])
        if state.dtype != np.dtype(np.float32) or not np.isfinite(state).all():
            raise FrozenTrunkFeatureSchemaError(
                "raw state must be finite float32"
            )
        if tuple(state.shape) == (1, 19):
            batched = False
            expected_image_shape = RAW_IMAGE_SHAPE
        elif (
            state.ndim == 3
            and state.shape[0] > 0
            and tuple(state.shape[1:]) == (1, 19)
        ):
            batched = True
            expected_image_shape = (int(state.shape[0]), *RAW_IMAGE_SHAPE)
        else:
            raise FrozenTrunkFeatureSchemaError(
                "raw state must have shape (1, 19) or (B, 1, 19)"
            )
        result = {"state": state}
        for image_key in image_keys:
            image = np.asarray(observation[image_key])
            if tuple(image.shape) != expected_image_shape:
                raise FrozenTrunkFeatureSchemaError(
                    f"{image_key} must have shape {expected_image_shape}, "
                    f"got {image.shape}"
                )
            if image.dtype != np.dtype(np.uint8):
                raise FrozenTrunkFeatureSchemaError(
                    f"{image_key} must have dtype uint8, got {image.dtype}"
                )
            result[image_key] = image
        return result, batched

    def __call__(
        self,
        observation: Mapping[str, Any],
        *,
        copy_state: bool = False,
    ) -> dict[str, np.ndarray]:
        import jax

        canonical, batched = self._validate_raw_observation(
            observation, self.image_keys
        )
        state = canonical["state"]
        result = {"state": state.copy() if copy_state else state}
        for image_key in self.image_keys:
            pixels = canonical[image_key]
            if batched:
                batch_size = int(pixels.shape[0])
                flat_pixels = pixels.reshape(batch_size, 128, 128, 3)
            else:
                # EncodingWrapper removes the canonical T=1 stacking axis for
                # a single observation.  Match that exact convolution path so
                # the extracted trunk tensor is bit-for-bit identical.
                flat_pixels = pixels[0]
            value = self._trunk.apply(
                {"params": self._params}, flat_pixels, train=False
            )
            feature = np.asarray(jax.device_get(value), dtype=np.float32)
            if batched:
                feature = feature.reshape(
                    batch_size, *FROZEN_TRUNK_FEATURE_SHAPE
                )
            else:
                feature = feature[None]
            result[image_key] = validate_frozen_trunk_feature(
                feature,
                name=image_key,
                batched=batched,
                copy=False,
            )
        return validate_frozen_trunk_observation(
            result,
            batched=batched,
            copy=False,
            image_keys=self.image_keys,
        )


__all__ = [
    "FROZEN_TRUNK_CONTRACT",
    "FROZEN_TRUNK_FEATURE_CONTRACT_REVISION",
    "FROZEN_TRUNK_FEATURE_DTYPE",
    "FROZEN_TRUNK_FEATURE_SHAPE",
    "FROZEN_TRUNK_MODEL_REVISION",
    "FROZEN_TRUNK_PARAMETER_PATH",
    "FrozenResNet10TrunkExtractor",
    "FrozenTrunkFeatureContract",
    "FrozenTrunkFeatureSchemaError",
    "create_frozen_trunk_feature_agent",
    "validate_frozen_trunk_feature",
    "validate_frozen_trunk_observation",
]
