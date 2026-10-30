"""Tests for the shared camera-serial defaults file (no ROS, no cameras)."""

import os

from gello_recorder import camera_defaults


def test_parse_env_file_handles_comments_quotes_and_export():
    values = camera_defaults.parse_env_file(
        "# header\n"
        "CAM1_SERIAL=111   # scene\n"
        "export CAM2_SERIAL='222'\n"
        "not a pair\n"
    )
    assert values == {"CAM1_SERIAL": "111", "CAM2_SERIAL": "222"}


def test_load_default_serials_from_explicit_path(tmp_path):
    path = tmp_path / "serials.env"
    path.write_text("CAM1_SERIAL=aaa\nCAM2_SERIAL=bbb\n")
    assert camera_defaults.load_default_serials(str(path)) == ("aaa", "bbb")


def test_missing_file_yields_empty_serials(tmp_path):
    missing = tmp_path / "nope.env"
    assert camera_defaults.load_default_serials(str(missing)) == ("", "")


def test_env_var_selects_the_file(tmp_path, monkeypatch):
    path = tmp_path / "rig.env"
    path.write_text("CAM1_SERIAL=x1\nCAM2_SERIAL=x2\n")
    monkeypatch.setenv(camera_defaults.SERIALS_FILE_ENV_VAR, str(path))
    assert camera_defaults.default_serials_file() == str(path)
    assert camera_defaults.load_default_serials() == ("x1", "x2")


def test_repo_file_exists_and_defines_two_distinct_serials(monkeypatch):
    monkeypatch.delenv(camera_defaults.SERIALS_FILE_ENV_VAR, raising=False)
    path = camera_defaults.default_serials_file()
    assert os.path.isfile(path), path
    cam1, cam2 = camera_defaults.load_default_serials(path)
    assert cam1 and cam2 and cam1 != cam2
