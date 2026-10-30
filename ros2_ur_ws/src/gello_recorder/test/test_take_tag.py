"""Unit tests for completed-take folder tagging (no ROS or Qt required)."""

import os

import pytest

from gello_recorder.take_tag import TakeTagError, normalize_tag, tag_take_dir


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("recover", "recover"),
        ("Other Side", "other_side"),
        ("recover-other_side", "recover_other_side"),
    ],
)
def test_normalize_tag(raw, expected):
    assert normalize_tag(raw) == expected


@pytest.mark.parametrize("raw", ["", "../escape", "tag/child", "한글"])
def test_unsafe_tag_is_rejected(raw):
    with pytest.raises(TakeTagError):
        normalize_tag(raw)


def test_tag_take_dir_renames_without_touching_contents(tmp_path):
    take = tmp_path / "take_07_20260918_221221"
    take.mkdir()
    payload = take / "vectors.h5"
    payload.write_bytes(b"data")

    result = tag_take_dir(str(take), str(tmp_path), "Other Side")

    renamed = tmp_path / "take_07_20260918_221221_other_side"
    assert result == {
        "old_path": str(take),
        "path": str(renamed),
        "tag": "other_side",
        "changed": True,
    }
    assert not take.exists()
    assert (renamed / "vectors.h5").read_bytes() == b"data"


def test_tags_accumulate_and_duplicate_last_tag_is_idempotent(tmp_path):
    take = tmp_path / "take_12_20260918_221221"
    take.mkdir()
    first = tag_take_dir(str(take), str(tmp_path), "recover")
    second = tag_take_dir(first["path"], str(tmp_path), "other_side")
    duplicate = tag_take_dir(second["path"], str(tmp_path), "other_side")

    assert second["path"].endswith("_recover_other_side")
    assert duplicate["changed"] is False
    assert duplicate["path"] == second["path"]


def test_take_must_be_direct_child_with_expected_name(tmp_path):
    outside = tmp_path / "nested"
    outside.mkdir()
    take = outside / "take_01_20260918_221221"
    take.mkdir()

    with pytest.raises(TakeTagError):
        tag_take_dir(str(take), str(tmp_path), "recover")


def test_existing_destination_is_never_overwritten(tmp_path):
    take = tmp_path / "take_01_20260918_221221"
    destination = tmp_path / "take_01_20260918_221221_recover"
    take.mkdir()
    destination.mkdir()

    with pytest.raises(TakeTagError):
        tag_take_dir(str(take), str(tmp_path), "recover")

    assert os.path.isdir(take)
    assert os.path.isdir(destination)


def test_preset_tags_default_when_env_unset(monkeypatch):
    from gello_recorder import take_tag

    monkeypatch.delenv(take_tag.TAG_PRESETS_ENV_VAR, raising=False)
    assert take_tag.preset_tags() == list(take_tag.DEFAULT_TAG_PRESETS)


def test_preset_tags_from_env_normalized_deduped_and_invalid_skipped(monkeypatch):
    from gello_recorder import take_tag

    monkeypatch.setenv(
        take_tag.TAG_PRESETS_ENV_VAR, " Recover , other side,recover,,bad/tag"
    )
    assert take_tag.preset_tags() == ["recover", "other_side"]


def test_preset_tags_explicit_empty_means_none():
    from gello_recorder import take_tag

    assert take_tag.preset_tags("") == []
