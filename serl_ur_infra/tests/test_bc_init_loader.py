"""File-level gates of a ``hil-serl-bc-init`` artifact, without jax or flax.

This half of the loader is the one that runs in the actor venv, where neither
jax nor flax exists, so every test here is stdlib byte work: does the manifest
say what it must say, does the parameter file weigh and hash what the manifest
claims, and did the writer actually finish.  The parameter payload is therefore
deliberately *not* a real msgpack tree -- at this level it is opaque bytes, and
a test that fed it real parameters would be quietly asserting something the
manifest reader never looks at.

Every rejection is asserted by exception TYPE only.  Message text is a
diagnostic for a human reading a failed startup, not a contract.

Each corruption below is *isolated*: a mutation that edits ``manifest.json``
rewrites ``completion.json`` too, so the artifact carries exactly one defect
and cannot pass for the wrong reason.  The single exception is the stale-digest
case, where the un-refreshed ``completion.json`` IS the defect.

Run with::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_bc_init_loader.py

(from ``/home/laptop3/gello_software``)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import pytest


_INFRA = Path(__file__).resolve().parents[1]
if str(_INFRA) not in sys.path:
    sys.path.insert(0, str(_INFRA))

from ur_env.learner.bc_init import (  # noqa: E402
    BC_INIT_FORMAT,
    BC_INIT_FORMAT_VERSION,
    BC_INIT_SUBTREES,
    BC_MODEL_ID,
    BC_REWARD_MODEL_ID,
    BcInitError,
    load_bc_init_manifest,
    verify_resnet_asset,
)


_BC_INIT_SOURCE = _INFRA / "ur_env" / "learner" / "bc_init.py"

PARAMETER_FILENAME = "actor_grasp_params.msgpack"
MANIFEST_FILENAME = "manifest.json"
COMPLETION_FILENAME = "completion.json"

#: Opaque stand-in for the real 30 MB msgpack tree.  The manifest reader only
#: measures and hashes it, so its content is irrelevant here by construction.
PARAMETER_PAYLOAD = b"\x81\xa6params" + bytes(range(256)) * 3

#: A syntactically valid digest that no file in these fixtures hashes to.  The
#: manifest half never reads the ResNet asset, so the valid fixture can carry a
#: placeholder; the ``verify_resnet_asset`` tests below supply a real one.
PLACEHOLDER_RESNET_SHA256 = "0" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _flip_hex(digest: str) -> str:
    """Return ``digest`` with its final hex character changed."""

    return digest[:-1] + ("0" if digest[-1] != "0" else "1")


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest_then_completion(
    directory: Path, manifest: dict[str, Any]
) -> None:
    """Write both control files in the order a real writer must use.

    ``completion.json`` hashes the manifest bytes, so it can only be written
    once ``manifest.json`` is on disk in its final form.
    """

    manifest_path = directory / MANIFEST_FILENAME
    _write_json(manifest_path, manifest)
    _write_json(
        directory / COMPLETION_FILENAME,
        {
            "complete": True,
            "manifest_sha256": _sha256(manifest_path.read_bytes()),
            "parameter_sha256": manifest["parameter_sha256"],
        },
    )


def make_artifact(
    tmp_path: Path,
    *,
    mutate: Callable[[Path], None] | None = None,
    name: str = "bc-init",
    payload: bytes = PARAMETER_PAYLOAD,
    resnet_sha256: str = PLACEHOLDER_RESNET_SHA256,
) -> Path:
    """Build a fully valid artifact directory, then let ``mutate`` corrupt it."""

    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / PARAMETER_FILENAME).write_bytes(payload)
    _write_manifest_then_completion(
        directory,
        {
            "format": BC_INIT_FORMAT,
            "format_version": BC_INIT_FORMAT_VERSION,
            "parameter_subtrees": list(BC_INIT_SUBTREES),
            "parameter_file": PARAMETER_FILENAME,
            "parameter_bytes": len(payload),
            "parameter_sha256": _sha256(payload),
            "resnet_sha256": resnet_sha256,
        },
    )
    if mutate is not None:
        mutate(directory)
    return directory


def _rewrite_manifest(directory: Path, **changes: Any) -> None:
    """Edit ``manifest.json`` and refresh ``completion.json`` around it."""

    manifest = _read_json(directory / MANIFEST_FILENAME)
    manifest.update(changes)
    _write_manifest_then_completion(directory, manifest)


def _rewrite_completion(directory: Path, **changes: Any) -> None:
    completion = _read_json(directory / COMPLETION_FILENAME)
    completion.update(changes)
    _write_json(directory / COMPLETION_FILENAME, completion)


# --------------------------------------------------------------------------
# The valid artifact, and the module's own import weight.
# --------------------------------------------------------------------------


def test_valid_artifact_parses_and_reports_its_parameter_digest(tmp_path):
    directory = make_artifact(tmp_path)

    manifest = load_bc_init_manifest(directory)

    assert manifest["parameter_sha256"] == _sha256(PARAMETER_PAYLOAD)
    assert manifest["format"] == BC_INIT_FORMAT
    assert manifest["format_version"] == BC_INIT_FORMAT_VERSION
    assert manifest["parameter_subtrees"] == list(BC_INIT_SUBTREES)
    assert manifest["parameter_bytes"] == len(PARAMETER_PAYLOAD)


def test_a_str_path_is_accepted_like_a_path_object(tmp_path):
    directory = make_artifact(tmp_path)

    assert load_bc_init_manifest(str(directory)) == load_bc_init_manifest(
        directory
    )


def test_module_scope_stays_stdlib_only():
    """Importing the loader must not drag jax or flax in.

    The actor venv has neither installed, so on the canonical interpreter this
    is proven by the module-level import above merely succeeding.  Executing a
    private second copy and watching ``sys.modules`` keeps the same guarantee
    meaningful when the file is run under the hilserl interpreter, where both
    packages exist and a module-scope import would otherwise go unnoticed.
    """

    spec = importlib.util.spec_from_file_location(
        "_bc_init_import_weight_probe", _BC_INIT_SOURCE
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
    assert module.BC_INIT_FORMAT == BC_INIT_FORMAT


def test_wire_identities_are_non_empty_and_distinct():
    assert isinstance(BC_MODEL_ID, str) and BC_MODEL_ID
    assert isinstance(BC_REWARD_MODEL_ID, str) and BC_REWARD_MODEL_ID
    assert BC_MODEL_ID != BC_REWARD_MODEL_ID
    assert issubclass(BcInitError, RuntimeError)


# --------------------------------------------------------------------------
# Every way an artifact can lie about itself.
# --------------------------------------------------------------------------


def _drop_manifest(directory: Path) -> None:
    (directory / MANIFEST_FILENAME).unlink()


def _wrong_format(directory: Path) -> None:
    _rewrite_manifest(directory, format="hil-serl-bc-init-something-else")


def _wrong_format_version(directory: Path) -> None:
    _rewrite_manifest(directory, format_version=BC_INIT_FORMAT_VERSION + 1)


def _reordered_subtrees(directory: Path) -> None:
    _rewrite_manifest(
        directory, parameter_subtrees=list(reversed(BC_INIT_SUBTREES))
    )


def _short_subtree_list(directory: Path) -> None:
    _rewrite_manifest(directory, parameter_subtrees=[BC_INIT_SUBTREES[0]])


def _drop_parameter_file(directory: Path) -> None:
    (directory / PARAMETER_FILENAME).unlink()


def _parameter_bytes_off_by_one(directory: Path) -> None:
    manifest = _read_json(directory / MANIFEST_FILENAME)
    _rewrite_manifest(directory, parameter_bytes=manifest["parameter_bytes"] + 1)


def _parameter_sha_off_by_one_character(directory: Path) -> None:
    # Flipped in the manifest, and therefore in the completion marker that
    # mirrors it, so the single remaining defect is "declared digest is not the
    # file's digest".
    manifest = _read_json(directory / MANIFEST_FILENAME)
    _rewrite_manifest(
        directory, parameter_sha256=_flip_hex(manifest["parameter_sha256"])
    )


def _drop_completion(directory: Path) -> None:
    (directory / COMPLETION_FILENAME).unlink()


def _not_complete(directory: Path) -> None:
    _rewrite_completion(directory, complete=False)


def _completion_parameter_sha_disagrees(directory: Path) -> None:
    completion = _read_json(directory / COMPLETION_FILENAME)
    _rewrite_completion(
        directory, parameter_sha256=_flip_hex(completion["parameter_sha256"])
    )


def _manifest_rewritten_after_completion(directory: Path) -> None:
    # A whitespace-only edit: still valid JSON with identical fields, but no
    # longer the bytes the writer signed.  Only the completion digest sees it.
    manifest_path = directory / MANIFEST_FILENAME
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_drop_manifest, id="manifest-missing"),
        pytest.param(_wrong_format, id="format-wrong"),
        pytest.param(_wrong_format_version, id="format-version-wrong"),
        pytest.param(_reordered_subtrees, id="subtrees-reordered"),
        pytest.param(_short_subtree_list, id="subtrees-incomplete"),
        pytest.param(_drop_parameter_file, id="parameter-file-missing"),
        pytest.param(_parameter_bytes_off_by_one, id="parameter-bytes-off-by-one"),
        pytest.param(_parameter_sha_off_by_one_character, id="parameter-sha-wrong"),
        pytest.param(_drop_completion, id="completion-missing"),
        pytest.param(_not_complete, id="completion-not-complete"),
        pytest.param(
            _completion_parameter_sha_disagrees, id="completion-parameter-sha"
        ),
        pytest.param(
            _manifest_rewritten_after_completion, id="completion-manifest-sha-stale"
        ),
    ],
)
def test_corrupt_artifacts_are_refused(tmp_path, mutate):
    directory = make_artifact(tmp_path, mutate=mutate)

    with pytest.raises(BcInitError):
        load_bc_init_manifest(directory)


def test_missing_artifact_directory_is_refused(tmp_path):
    with pytest.raises(BcInitError):
        load_bc_init_manifest(tmp_path / "no-such-artifact")


def test_an_empty_directory_is_refused(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(BcInitError):
        load_bc_init_manifest(empty)


# --------------------------------------------------------------------------
# The ResNet asset the trainer saw.
# --------------------------------------------------------------------------


def _resnet_fixture(tmp_path: Path) -> tuple[Path, str]:
    asset = tmp_path / "resnet10_params.pkl"
    asset.write_bytes(b"pretend-resnet10-weights")
    return asset, _sha256(asset.read_bytes())


def test_verify_resnet_asset_accepts_the_declared_asset(tmp_path):
    asset, digest = _resnet_fixture(tmp_path)
    manifest = load_bc_init_manifest(
        make_artifact(tmp_path, resnet_sha256=digest)
    )

    verify_resnet_asset(manifest, asset)


def test_verify_resnet_asset_rejects_a_different_asset(tmp_path):
    asset, digest = _resnet_fixture(tmp_path)
    manifest = load_bc_init_manifest(
        make_artifact(tmp_path, resnet_sha256=_flip_hex(digest))
    )

    with pytest.raises(BcInitError):
        verify_resnet_asset(manifest, asset)


def test_verify_resnet_asset_rejects_a_missing_asset(tmp_path):
    asset, digest = _resnet_fixture(tmp_path)
    manifest = load_bc_init_manifest(
        make_artifact(tmp_path, resnet_sha256=digest)
    )
    asset.unlink()

    with pytest.raises(BcInitError):
        verify_resnet_asset(manifest, asset)
