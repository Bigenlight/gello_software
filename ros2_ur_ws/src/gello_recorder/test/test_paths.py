"""Recording defaults must stay with the active checkout."""

from pathlib import Path

from gello_recorder import paths


def test_explicit_repository_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GELLO_REPO_ROOT", str(tmp_path))
    assert paths.default_repo_root() == str(tmp_path)


def test_resolves_checkout_without_override(monkeypatch):
    monkeypatch.delenv("GELLO_REPO_ROOT", raising=False)
    root = Path(paths.default_repo_root())
    assert (root / "ros2_ur_ws/src/gello_recorder").is_dir()
    assert Path(paths.__file__).resolve().is_relative_to(root)
