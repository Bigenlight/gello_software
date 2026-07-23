import json

import pytest

from policy_server.checkpoint_identity import (
    compute_checkpoint_identity,
    verify_expected_identity,
)


def _checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"type": "multi_task_dit"}), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (path / "policy_preprocessor_step_0_processor.safetensors").write_bytes(b"stats")
    (path / "policy_postprocessor.json").write_text("{}", encoding="utf-8")


def test_identity_hashes_config_weights_and_processors(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint)
    first = compute_checkpoint_identity(checkpoint)
    assert first.policy_type == "multi_task_dit"
    assert first.revision.startswith("sha256:")
    assert {name for name, _ in first.files} == {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_0_processor.safetensors",
        "policy_postprocessor.json",
    }
    (checkpoint / "policy_preprocessor.json").write_text('{"changed":true}', encoding="utf-8")
    assert compute_checkpoint_identity(checkpoint).revision != first.revision
    first = compute_checkpoint_identity(checkpoint)
    (checkpoint / "policy_preprocessor_step_0_processor.safetensors").write_bytes(
        b"changed stats"
    )
    assert compute_checkpoint_identity(checkpoint).revision != first.revision


def test_expected_identity_is_enforced(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint)
    identity = compute_checkpoint_identity(checkpoint)
    verify_expected_identity(
        identity,
        expected_policy_type="multi_task_dit",
        expected_revision=identity.revision,
    )
    with pytest.raises(ValueError, match="revision"):
        verify_expected_identity(
            identity,
            expected_policy_type="multi_task_dit",
            expected_revision="sha256:" + "0" * 64,
        )
    with pytest.raises(ValueError, match="policy type"):
        verify_expected_identity(
            identity,
            expected_policy_type="act",
            expected_revision=identity.revision,
        )
    with pytest.raises(ValueError, match="policy type"):
        verify_expected_identity(identity, expected_revision=identity.revision)
    with pytest.raises(ValueError, match="pinned"):
        verify_expected_identity(
            identity,
            expected_policy_type="multi_task_dit",
            expected_revision="auto",
        )


def test_identity_requires_local_complete_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="config.json"):
        compute_checkpoint_identity(tmp_path)
    (tmp_path / "config.json").write_text('{"type":"act"}', encoding="utf-8")
    with pytest.raises(ValueError, match="weights"):
        compute_checkpoint_identity(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="preprocessor"):
        compute_checkpoint_identity(tmp_path)
    (tmp_path / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="postprocessor"):
        compute_checkpoint_identity(tmp_path)


def test_identity_hashes_weight_index_and_rejects_symlink(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    index.write_text('{"weight_map":{}}', encoding="utf-8")
    first = compute_checkpoint_identity(checkpoint)
    assert "model.safetensors.index.json" in {name for name, _ in first.files}
    index.write_text('{"weight_map":{"changed":"model.safetensors"}}', encoding="utf-8")
    assert compute_checkpoint_identity(checkpoint).revision != first.revision

    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    (checkpoint / "extra_preprocessor.json").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        compute_checkpoint_identity(checkpoint)


def test_identity_rejects_processor_directory_symlink(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint)
    external = tmp_path / "external_preprocessor"
    external.mkdir()
    (external / "state.safetensors").write_bytes(b"external stats")
    (checkpoint / "other_preprocessor").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        compute_checkpoint_identity(checkpoint)
