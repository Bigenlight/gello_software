"""Graft a ``hil-serl-bc-init`` artifact into the production frozen-trunk agent.

WHAT A BC-INIT ARTIFACT IS
--------------------------
Another party trains behaviour cloning on the canonical demo corpus and saves a
directory holding *only the two top-level parameter subtrees* of the production
hybrid SAC agent that BC can supervise:

    <artifact>/actor_grasp_params.msgpack   flax state-dict of
                                            ``modules_actor`` + ``modules_grasp_critic``
    <artifact>/manifest.json                format id, subtree list, sizes, digests
    <artifact>/completion.json              write-completion + digest cross-check
    <artifact>/README.md                    human notes, never parsed

It is therefore *not* an orbax checkpoint and cannot be restored as one
(``production_checkpoint_compatible: false`` in the real manifest).  The serving
path is: build the template agent with the ordinary production factory
(``frozen_trunk.create_frozen_trunk_feature_agent``), then replace exactly those
two subtrees with the trained ones.  Everything else -- critic, temperature, and
above all the frozen ResNet-10 trunk -- stays the template's, which is what lets
``FrozenResNet10TrunkExtractor.validate_parameter_invariant`` still pass after the
graft.

WHY THE VALIDATION IS BIDIRECTIONAL AND HAND-WRITTEN
----------------------------------------------------
The obvious loader is ``flax.serialization.from_bytes(template, raw)``.  It is a
fail-open path and must not be used here: measured on flax 0.10.5 it silently
DROPS keys the artifact has and the template does not, and it silently ACCEPTS a
leaf whose shape changed.  A BC agent built from a different config would then
load without an error, pass every startup gate -- parameter-tree signature, trunk
invariant, action-contract smoke -- and serve *a function nobody trained* to a
real robot.

So :func:`load_bc_init_params` restores the raw tree with ``msgpack_restore`` and
compares it against the template state-dict in both directions, key path by key
path, shape and dtype included, before ``from_state_dict`` is allowed to run.
Any difference is a refusal that names the offending paths.  A structural
mismatch means the trainer's config diverged from production; that is a question
for the trainer, not something to coerce.

Digest checks are the other half: the artifact declares its own parameter size
and SHA256, ``completion.json`` re-declares that digest plus a digest of the
manifest bytes, and the manifest declares the ResNet asset SHA the training run
saw.  A half-written or hand-edited artifact fails before any array is touched.

THIS MODULE HAS NO TRAINING CODE, DELIBERATELY.  It only loads, verifies and
grafts.  Retraining is owned elsewhere; adding a trainer here would give this
file a reason to import jax at module scope.

IMPORT WEIGHT
-------------
Module scope is stdlib only.  The manifest half runs in the actor venv, which has
no jax and no flax, so :func:`load_bc_init_manifest` and
:func:`verify_resnet_asset` are pure file/digest work.  jax and flax are imported
inside :func:`load_bc_init_params`.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


#: Artifact format identity.  Both are checked exactly; a future format bumps
#: the version rather than silently reusing this loader.
BC_INIT_FORMAT = "hil-serl-bc-init"
BC_INIT_FORMAT_VERSION = 1

#: The only parameter subtrees a bc-init artifact may carry, in this order.  BC
#: supervises the actor and the grasp critic; the continuous critic, the
#: temperature and the frozen trunk are not learnable from demonstrations alone.
BC_INIT_SUBTREES = ("modules_actor", "modules_grasp_critic")

#: Advertised over the wire so an actor pinning ``EXPECTED_MODEL_ID`` cannot be
#: pointed at the online learner by accident, and vice versa.
BC_MODEL_ID = "bc-cube-in-cup-raw0731-bcinit-v1"

#: BC evaluation runs MANUAL: the operator's ``MARK SUCCESS`` is the only reward
#: authority.  No classifier checkpoint scores these episodes.
BC_REWARD_MODEL_ID = "operator-manual-success-v1"

MANIFEST_FILENAME = "manifest.json"
COMPLETION_FILENAME = "completion.json"

#: Cap on how many mismatching parameter paths one refusal message lists.  The
#: count of the remainder is always reported, so nothing is hidden.
_MAX_REPORTED_PATHS = 10

_CHUNK_BYTES = 1024 * 1024


class BcInitError(RuntimeError):
    """A bc-init artifact is absent, incomplete, or not what it claims to be."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BcInitError(f"bc-init {label} is missing: {path}")
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise BcInitError(
            f"bc-init {label} is not readable JSON: {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BcInitError(f"bc-init {label} must be a JSON object: {path}")
    return value


def _hex_digest_field(
    document: Mapping[str, Any], key: str, *, path: Path, label: str
) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BcInitError(
            f"bc-init {label} {key} must be a hex SHA256 string, got {value!r}: {path}"
        )
    return value.strip().lower()


def _parameter_file_path(directory: Path, manifest: Mapping[str, Any]) -> Path:
    name = manifest.get("parameter_file")
    if not isinstance(name, str) or not name:
        raise BcInitError(
            "bc-init manifest parameter_file must be a non-empty string, got "
            f"{name!r}: {directory / MANIFEST_FILENAME}"
        )
    relative = Path(name)
    # The manifest names a file *inside* its own artifact directory; anything
    # else would let a manifest point the loader at an unverified tree.
    if relative.is_absolute() or ".." in relative.parts:
        raise BcInitError(
            "bc-init manifest parameter_file must be relative to the artifact "
            f"directory, got {name!r}: {directory}"
        )
    return directory / relative


def load_bc_init_manifest(artifact_dir: "os.PathLike[str] | str") -> dict[str, Any]:
    """Verify the artifact on disk and return its parsed manifest.

    Stdlib only -- no jax, no flax, no numpy.  Every declared invariant is
    checked before the caller is allowed to believe any field: format identity,
    subtree list, parameter file size and digest, and the ``completion.json``
    cross-checks that prove the writer finished and that this manifest is the one
    it finished against.
    """

    directory = Path(artifact_dir).expanduser()
    if not directory.is_dir():
        raise BcInitError(f"bc-init artifact directory is missing: {directory}")

    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise BcInitError(f"bc-init manifest is missing: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise BcInitError(
            f"bc-init manifest is not readable JSON: {manifest_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise BcInitError(f"bc-init manifest must be a JSON object: {manifest_path}")

    actual_format = manifest.get("format")
    if actual_format != BC_INIT_FORMAT:
        raise BcInitError(
            f"bc-init manifest format is {actual_format!r}, expected "
            f"{BC_INIT_FORMAT!r}: {manifest_path}"
        )
    version = manifest.get("format_version")
    if isinstance(version, bool) or version != BC_INIT_FORMAT_VERSION:
        raise BcInitError(
            f"bc-init manifest format_version is {version!r}, expected "
            f"{BC_INIT_FORMAT_VERSION}: {manifest_path}"
        )
    subtrees = manifest.get("parameter_subtrees")
    if subtrees != list(BC_INIT_SUBTREES):
        raise BcInitError(
            f"bc-init manifest parameter_subtrees is {subtrees!r}, expected "
            f"{list(BC_INIT_SUBTREES)!r}: {manifest_path}"
        )

    parameter_path = _parameter_file_path(directory, manifest)
    if not parameter_path.is_file():
        raise BcInitError(f"bc-init parameter file is missing: {parameter_path}")
    declared_bytes = manifest.get("parameter_bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
        raise BcInitError(
            "bc-init manifest parameter_bytes must be an integer, got "
            f"{declared_bytes!r}: {manifest_path}"
        )
    actual_bytes = parameter_path.stat().st_size
    if actual_bytes != declared_bytes:
        raise BcInitError(
            "bc-init parameter file size mismatch: "
            f"expected={declared_bytes}, actual={actual_bytes}, "
            f"path={parameter_path}"
        )
    declared_sha = _hex_digest_field(
        manifest, "parameter_sha256", path=manifest_path, label="manifest"
    )
    actual_sha = _sha256_file(parameter_path)
    if actual_sha != declared_sha:
        raise BcInitError(
            "bc-init parameter SHA256 mismatch: "
            f"expected={declared_sha}, actual={actual_sha}, path={parameter_path}"
        )

    completion_path = directory / COMPLETION_FILENAME
    completion = _read_json_object(completion_path, label="completion.json")
    # ``is not True`` on purpose: a truthy string or 1 is a malformed writer, not
    # a completed one.
    if completion.get("complete") is not True:
        raise BcInitError(
            "bc-init artifact is not marked complete "
            f"(complete={completion.get('complete')!r}): {completion_path}"
        )
    completion_parameter_sha = _hex_digest_field(
        completion, "parameter_sha256", path=completion_path, label="completion.json"
    )
    if completion_parameter_sha != declared_sha:
        raise BcInitError(
            "bc-init completion.json parameter_sha256 disagrees with the "
            f"manifest: completion={completion_parameter_sha}, "
            f"manifest={declared_sha}, path={completion_path}"
        )
    completion_manifest_sha = _hex_digest_field(
        completion, "manifest_sha256", path=completion_path, label="completion.json"
    )
    actual_manifest_sha = _sha256_bytes(manifest_bytes)
    if completion_manifest_sha != actual_manifest_sha:
        raise BcInitError(
            "bc-init manifest SHA256 mismatch: "
            f"expected={completion_manifest_sha}, actual={actual_manifest_sha}, "
            f"path={manifest_path}"
        )
    return manifest


def verify_resnet_asset(
    manifest: Mapping[str, Any], resnet_path: "os.PathLike[str] | str"
) -> None:
    """Require the local ResNet-10 asset to be the one BC trained against.

    The graft keeps the *template's* trunk, so a trainer that saw different trunk
    weights produced a head that is fed features this process cannot reproduce.
    That is undetectable downstream -- the trunk invariant only proves the
    template's own trunk is unchanged -- so it is caught here instead.
    """

    path = Path(resnet_path).expanduser()
    expected = _hex_digest_field(
        manifest, "resnet_sha256", path=path, label="manifest"
    )
    if not path.is_file():
        raise BcInitError(f"ResNet asset for bc-init verification is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise BcInitError(
            "bc-init ResNet asset SHA256 mismatch: "
            f"expected={expected}, actual={actual}, path={path}"
        )


def _dtype_name(dtype: Any) -> str:
    import numpy as np

    try:
        return np.dtype(dtype).str
    except Exception:
        return str(dtype)


def _leaf_signature(leaf: Any) -> tuple[tuple[int, ...], str]:
    """Describe one leaf identically whether it is numpy or a jax array."""

    shape = getattr(leaf, "shape", None)
    dtype = getattr(leaf, "dtype", None)
    if shape is None or dtype is None:
        import numpy as np

        array = np.asarray(leaf)
        shape, dtype = array.shape, array.dtype
    return tuple(int(dimension) for dimension in shape), _dtype_name(dtype)


def _flatten_tree(tree: Any) -> dict[str, tuple[tuple[int, ...], str]]:
    """Flatten a parameter tree to ``"a/b/c" -> (shape, dtype)``.

    ``FrozenDict`` is a ``Mapping``, so restored plain dicts and template frozen
    dicts flatten to the same key paths.  An empty mapping is recorded as its own
    entry: it has no leaves, and without this a structural difference between the
    two sides would flatten away to nothing.
    """

    flat: dict[str, tuple[tuple[int, ...], str]] = {}

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, Mapping):
            if not node:
                flat[prefix] = ((), "<empty-mapping>")
                return
            for key in node:
                child = f"{prefix}/{key}" if prefix else str(key)
                walk(node[key], child)
            return
        flat[prefix] = _leaf_signature(node)

    walk(tree, "")
    return flat


def _audit_against_template(restored: Any, template_state: Any) -> None:
    """Refuse unless the artifact and the template agree in BOTH directions.

    This is the check ``flax.serialization.from_bytes`` does not do.  Extra keys
    and changed leaf shapes are exactly what it swallows, so both are named here.
    """

    restored_flat = _flatten_tree(restored)
    template_flat = _flatten_tree(template_state)
    problems: list[str] = []
    for path in sorted(set(template_flat) - set(restored_flat)):
        problems.append(f"{path}: absent from the artifact")
    for path in sorted(set(restored_flat) - set(template_flat)):
        problems.append(f"{path}: absent from the agent template")
    for path in sorted(set(restored_flat) & set(template_flat)):
        artifact_signature = restored_flat[path]
        template_signature = template_flat[path]
        if artifact_signature != template_signature:
            problems.append(
                f"{path}: artifact {artifact_signature[0]}/{artifact_signature[1]} "
                f"!= template {template_signature[0]}/{template_signature[1]}"
            )
    if not problems:
        return
    shown = problems[:_MAX_REPORTED_PATHS]
    remainder = len(problems) - len(shown)
    detail = "; ".join(shown)
    if remainder > 0:
        detail = f"{detail}; ... and {remainder} more"
    raise BcInitError(
        f"bc-init parameters do not match the agent template "
        f"({len(problems)} mismatching path(s)): {detail}"
    )


def load_bc_init_params(
    artifact_dir: "os.PathLike[str] | str", agent_template: Any
) -> Any:
    """Return the template's full params with the two BC subtrees grafted in.

    The return value has the same container type as
    ``agent_template.state.params`` so it can go straight into
    ``VersionedPolicyRuntime(params=...)``, which then re-validates the leaf
    signature, the trunk invariant and the action contract.

    jax/flax are imported here, not at module scope: the manifest half of this
    module runs in the actor venv where neither exists.
    """

    manifest = load_bc_init_manifest(artifact_dir)
    directory = Path(artifact_dir).expanduser()
    parameter_path = _parameter_file_path(directory, manifest)

    from flax import serialization as flax_serialization
    from flax.core import FrozenDict

    raw_bytes = parameter_path.read_bytes()
    try:
        restored = flax_serialization.msgpack_restore(raw_bytes)
    except Exception as exc:
        raise BcInitError(
            f"bc-init parameter file could not be restored: {parameter_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(restored, Mapping):
        raise BcInitError(
            "bc-init parameter file did not restore to a parameter mapping: "
            f"{parameter_path}"
        )

    try:
        params = agent_template.state.params
    except AttributeError as exc:
        raise BcInitError(
            "agent_template must expose state.params to receive a bc-init graft"
        ) from exc
    if not isinstance(params, Mapping):
        raise BcInitError(
            "agent_template.state.params must be a mapping of parameter subtrees, "
            f"got {type(params).__name__}"
        )
    template_subtrees = {}
    for key in BC_INIT_SUBTREES:
        if key not in params:
            raise BcInitError(
                f"agent template parameters do not contain the subtree {key!r}"
            )
        template_subtrees[key] = params[key]

    template_state = flax_serialization.to_state_dict(template_subtrees)
    _audit_against_template(restored, template_state)

    # Only now: the audit has proven this is a same-shape, same-dtype, same-key
    # tree, so from_state_dict has nothing left to swallow.
    grafted_subtrees = flax_serialization.from_state_dict(template_subtrees, restored)

    if isinstance(params, FrozenDict):
        return params.copy(add_or_replace=grafted_subtrees)
    merged = dict(params)
    merged.update(grafted_subtrees)
    return merged


def deterministic_sample_action(
    agent: Any,
) -> Callable[[Any, Mapping[str, Any], Any, bool], Any]:
    """Build the ``sample_action`` hook that serves BC's mode action.

    Mirror of ``VersionedPolicyRuntime._sample_with_agent`` (``policy.py:208-222``)
    with two deliberate pins, both of which are review findings rather than
    stylistic choices:

    * ``params`` MUST be injected via ``agent.replace(state=...)``.  The runtime
      passes the grafted tree as the first argument and holds it as the published
      snapshot; a hook that closed over ``agent`` alone and ignored ``params``
      would serve the UNGRAFTED template while the parameter-tree signature, the
      trunk invariant and the action-contract smoke all still pass -- an
      untrained policy driving a real arm with every gate green.
    * The wire ``deterministic`` flag is deliberately ignored.  BC evaluation
      serves the mode action on every step, and the production actor sends
      ``deterministic=False`` on every step, so honouring the flag would sample
      from the BC policy's noise instead of evaluating it.  ``argmax=True`` is
      hard-coded; ``VersionedPolicyRuntime._smoke`` therefore exercises this one
      trace twice rather than two traces once.
    """

    if not hasattr(agent, "replace") or not hasattr(agent, "state"):
        raise BcInitError(
            "agent must expose replace() and state to serve bc-init parameters"
        )

    def sample(
        params: Any,
        observation: Mapping[str, Any],
        seed: Any,
        deterministic: bool,
    ) -> Any:
        del deterministic  # BC evaluation always serves the mode action.
        candidate = agent.replace(state=agent.state.replace(params=params))
        return candidate.sample_actions(
            observations=observation,
            seed=seed,
            argmax=True,
        )

    return sample


__all__ = [
    "BC_INIT_FORMAT",
    "BC_INIT_FORMAT_VERSION",
    "BC_INIT_SUBTREES",
    "BC_MODEL_ID",
    "BC_REWARD_MODEL_ID",
    "BcInitError",
    "COMPLETION_FILENAME",
    "MANIFEST_FILENAME",
    "deterministic_sample_action",
    "load_bc_init_manifest",
    "load_bc_init_params",
    "verify_resnet_asset",
]
