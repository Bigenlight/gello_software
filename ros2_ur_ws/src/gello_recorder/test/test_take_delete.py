"""Unit tests for ``gello_recorder.take_delete`` -- NO ROS, NO Qt, tmp_path
only.

Covers every guard ``validate_take_dir`` documents (pattern, symlink,
outside-root, nested-dir, root-itself), the successful delete path, the
``take_NN...`` index parser, and the pure take-index bookkeeping helper the
node's ``delete_last_take()`` uses (``next_take_index_after_delete``) so that
decrement rule is exercised without touching rclpy at all.

Run:
    cd ros2_ur_ws/src/gello_recorder
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q test/test_take_delete.py
"""

import os

import pytest

from gello_recorder.take_delete import (
    TAKE_DIR_RE,
    TakeDeleteError,
    delete_take_dir,
    next_take_index_after_delete,
    take_dir_summary,
    take_index_from_dir,
    validate_take_dir,
)


def _make_take(root, name="take_01_20260916_120000", files=None):
    """Create a well-formed take directory under ``root`` and return its path."""
    take_dir = os.path.join(root, name)
    os.makedirs(take_dir)
    for fname, contents in (files or {"cam1.mp4": b"a", "cam2.mp4": b"bb",
                                       "vectors.h5": b"ccc"}).items():
        with open(os.path.join(take_dir, fname), "wb") as f:
            f.write(contents)
    return take_dir


# --------------------------------------------------------------------------- #
# TAKE_DIR_RE / take_index_from_dir
# --------------------------------------------------------------------------- #
def test_pattern_accepts_well_formed_take_names():
    assert TAKE_DIR_RE.match("take_01_20260916_120000")
    assert TAKE_DIR_RE.match("take_23_20260101_000000")
    # Not capped at two digits -- start_recording()'s :02d is a MINIMUM width.
    assert TAKE_DIR_RE.match("take_100_20260916_120000")


@pytest.mark.parametrize("bad", [
    "take_1_20260916_120000",       # index not zero-padded to >=2 digits
    "take_01_2026916_120000",       # date too short
    "take_01_20260916_12000",       # time too short
    "take_01_20260916_120000_extra",
    "TAKE_01_20260916_120000",      # case-sensitive
    "take_ab_20260916_120000",      # non-numeric index
    "not_a_take_dir",
    "",
])
def test_pattern_rejects_malformed_names(bad):
    assert not TAKE_DIR_RE.match(bad)


def test_take_index_from_dir_parses_index():
    assert take_index_from_dir("take_01_20260916_120000") == 1
    assert take_index_from_dir("take_23_20260101_000000") == 23
    assert take_index_from_dir("/some/root/take_07_20260916_120000") == 7


def test_take_index_from_dir_rejects_malformed_name():
    with pytest.raises(TakeDeleteError):
        take_index_from_dir("not_a_take_dir")


# --------------------------------------------------------------------------- #
# validate_take_dir guards
# --------------------------------------------------------------------------- #
def test_validate_accepts_well_formed_take(tmp_path):
    root = str(tmp_path)
    take_dir = _make_take(root)
    resolved = validate_take_dir(take_dir, root)
    assert resolved == os.path.realpath(take_dir)


def test_validate_rejects_missing_dir(tmp_path):
    root = str(tmp_path)
    missing = os.path.join(root, "take_01_20260916_120000")
    with pytest.raises(TakeDeleteError):
        validate_take_dir(missing, root)


def test_validate_rejects_non_directory(tmp_path):
    root = str(tmp_path)
    path = os.path.join(root, "take_01_20260916_120000")
    with open(path, "wb") as f:
        f.write(b"not a directory")
    with pytest.raises(TakeDeleteError):
        validate_take_dir(path, root)


def test_validate_rejects_symlinked_take_dir(tmp_path):
    root = str(tmp_path)
    real_dir = _make_take(root, name="take_01_20260916_120000")
    link = os.path.join(root, "take_02_20260916_130000")
    os.symlink(real_dir, link, target_is_directory=True)
    with pytest.raises(TakeDeleteError):
        validate_take_dir(link, root)
    # And the thing it pointed at must survive -- the whole point of the
    # symlink guard is that a symlink is never followed into a delete.
    assert os.path.isdir(real_dir)


def test_validate_rejects_bad_basename(tmp_path):
    root = str(tmp_path)
    bad = os.path.join(root, "not_a_take_dir")
    os.makedirs(bad)
    with pytest.raises(TakeDeleteError):
        validate_take_dir(bad, root)


def test_validate_rejects_outside_root(tmp_path):
    root = os.path.join(str(tmp_path), "recordings")
    os.makedirs(root)
    outside_parent = os.path.join(str(tmp_path), "elsewhere")
    os.makedirs(outside_parent)
    outside_take = _make_take(outside_parent)
    with pytest.raises(TakeDeleteError):
        validate_take_dir(outside_take, root)


def test_validate_rejects_dotdot_traversal_that_resolves_outside_root(tmp_path):
    """A path whose BASENAME is a well-formed take name but whose ``..``
    components walk it out of output_root: the pattern guard passes, so the
    realpath-based direct-child guard is the only thing standing between this
    and an rmtree outside the recordings tree."""
    root = os.path.join(str(tmp_path), "recordings")
    os.makedirs(root)
    elsewhere = os.path.join(str(tmp_path), "elsewhere")
    os.makedirs(elsewhere)
    victim = _make_take(elsewhere, name="take_09_20260916_120000")
    crafted = os.path.join(root, "..", "elsewhere", "take_09_20260916_120000")
    assert os.path.isdir(crafted)                     # the path IS reachable
    with pytest.raises(TakeDeleteError):
        validate_take_dir(crafted, root)
    with pytest.raises(TakeDeleteError):
        delete_take_dir(crafted, root)
    assert os.path.isdir(victim)


def test_validate_accepts_a_take_when_output_root_itself_is_a_symlink(tmp_path):
    """``~/gello_recordings -> /mnt/disk/recordings`` is a normal setup. The
    node stores take paths built from the SYMLINKED root, so both sides must
    be realpath'd or every legitimate delete would be refused as "not a
    direct child" -- and the returned path must be the REAL one, since that
    is what gets rmtree'd."""
    real_root = os.path.join(str(tmp_path), "real_recordings")
    os.makedirs(real_root)
    link_root = os.path.join(str(tmp_path), "gello_recordings")
    os.symlink(real_root, link_root, target_is_directory=True)
    take_dir = _make_take(link_root)                  # created THROUGH the link
    assert not os.path.islink(take_dir)               # the take itself is real
    resolved = validate_take_dir(take_dir, link_root)
    assert resolved == os.path.join(real_root, os.path.basename(take_dir))
    result = delete_take_dir(take_dir, link_root)
    assert result["path"] == resolved
    assert not os.path.exists(take_dir)
    assert os.path.isdir(real_root)                   # only the take went


def test_validate_rejects_a_broken_symlink_as_a_symlink_not_as_missing(tmp_path):
    """lexists (not exists) is what routes a dangling link to the symlink
    guard; with exists() it would read as "does not exist" -- still refused,
    but for the wrong reason, and a later "if missing, forget it" shortcut
    would then silently drop a link nobody inspected."""
    root = str(tmp_path)
    link = os.path.join(root, "take_03_20260916_120000")
    os.symlink(os.path.join(root, "gone"), link)
    with pytest.raises(TakeDeleteError, match="symlink"):
        validate_take_dir(link, root)
    assert os.path.islink(link)                       # untouched


def test_validate_rejects_nested_take_dir(tmp_path):
    # A take dir that is two levels below output_root (e.g. inside another
    # take-shaped directory) is not a DIRECT child and must be refused.
    root = str(tmp_path)
    outer = _make_take(root, name="take_01_20260916_120000", files={})
    nested = os.path.join(outer, "take_02_20260916_130000")
    os.makedirs(nested)
    with open(os.path.join(nested, "cam1.mp4"), "wb") as f:
        f.write(b"a")
    with pytest.raises(TakeDeleteError):
        validate_take_dir(nested, root)


def test_validate_rejects_root_itself(tmp_path):
    # Name the root itself like a take dir so the basename-pattern guard
    # cannot be what trips this -- this exercises the direct-child/root-
    # itself guard specifically ("parent(resolved) != root_real").
    root = os.path.join(str(tmp_path), "take_01_20260916_120000")
    os.makedirs(root)
    with open(os.path.join(root, "cam1.mp4"), "wb") as f:
        f.write(b"a")
    with pytest.raises(TakeDeleteError):
        validate_take_dir(root, root)


def test_validate_rejects_subdirectory_inside_take(tmp_path):
    root = str(tmp_path)
    take_dir = _make_take(root, files={"cam1.mp4": b"a"})
    os.makedirs(os.path.join(take_dir, "stray_subdir"))
    with pytest.raises(TakeDeleteError):
        validate_take_dir(take_dir, root)


def test_validate_rejects_symlink_inside_take(tmp_path):
    root = str(tmp_path)
    take_dir = _make_take(root, files={"cam1.mp4": b"a"})
    target = os.path.join(root, "outside_file.bin")
    with open(target, "wb") as f:
        f.write(b"x")
    os.symlink(target, os.path.join(take_dir, "vectors.h5"))
    with pytest.raises(TakeDeleteError):
        validate_take_dir(take_dir, root)


# --------------------------------------------------------------------------- #
# take_dir_summary / delete_take_dir
# --------------------------------------------------------------------------- #
def test_take_dir_summary_counts_files_and_bytes(tmp_path):
    root = str(tmp_path)
    take_dir = _make_take(root, files={"cam1.mp4": b"a" * 10, "cam2.mp4": b"b" * 20,
                                        "vectors.h5": b"c" * 3})
    summary = take_dir_summary(take_dir)
    assert summary == {"n_files": 3, "bytes": 33}


def test_delete_take_dir_removes_folder_and_returns_counts(tmp_path):
    root = str(tmp_path)
    take_dir = _make_take(root, files={"cam1.mp4": b"a" * 10, "cam2.mp4": b"b" * 20,
                                        "vectors.h5": b"c" * 3, "depth.h5": b"d" * 4})
    result = delete_take_dir(take_dir, root)
    assert result["path"] == os.path.realpath(take_dir)
    assert result["n_files"] == 4
    assert result["bytes"] == 37
    assert not os.path.exists(take_dir)


def test_delete_take_dir_refuses_invalid_and_leaves_it_alone(tmp_path):
    root = str(tmp_path)
    bad = os.path.join(root, "not_a_take_dir")
    os.makedirs(bad)
    with pytest.raises(TakeDeleteError):
        delete_take_dir(bad, root)
    assert os.path.isdir(bad)


# --------------------------------------------------------------------------- #
# next_take_index_after_delete -- the node's decrement rule, tested pure
# --------------------------------------------------------------------------- #
def test_next_index_decrements_when_deleting_the_most_recent_take():
    assert next_take_index_after_delete(5, "take_05_20260916_120000") == 4


def test_next_index_unchanged_when_deleting_an_older_take():
    # Not reachable through the GUI today (it only ever offers the most
    # recently stopped take), but the helper must still be correct/pure.
    assert next_take_index_after_delete(5, "take_03_20260916_120000") == 5


def test_next_index_unchanged_when_current_index_is_zero_and_deleted_is_one():
    assert next_take_index_after_delete(0, "take_01_20260916_120000") == 0
