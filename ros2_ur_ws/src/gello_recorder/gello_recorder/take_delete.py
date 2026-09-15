#!/usr/bin/env python3
"""Safe deletion of a single recorder "take" directory -- stdlib only, no ROS,
no Qt.

This module exists because ``delete_last_take()`` on the GUI node removes
user-recorded data (a few MB of cam1.mp4 / cam2.mp4 / vectors.h5 / depth.h5)
with a single click, and a single click that deletes the wrong directory is
not recoverable. ``GelloRecorderGuiNode`` only ever offers the take it most
recently STOPPED in this process (see ``deletable_take_dir()``), so in
practice the path handed to this module is always trustworthy -- but the
helper is written to be safe on its own, independent of that guarantee,
because:

* the node's guarantee lives in a different file and can rot silently if
  someone refactors it later without noticing this module depends on it;
* a future caller (a CLI cleanup tool, a test, another node) may call this
  module directly without going through the GUI at all;
* the operation is ``shutil.rmtree`` -- there is no "are you sure" once it
  runs, so the cost of one extra guard here is nothing next to the cost of
  a wrong one being missing.

Guards implemented by ``validate_take_dir`` and why each one exists:

* **must exist / must be a directory** -- refuse to silently no-op or to
  ``rmtree`` a plain file (rmtree raises on a non-directory anyway, but we
  want a precise reason before doing anything, not an incidental OSError).
* **must not be a symlink** (checked on the *un-resolved* path, i.e.
  ``os.path.islink`` before any ``realpath``) -- a symlinked "take dir" could
  point anywhere on disk; deleting through it would delete whatever it
  points to, not a take. ``shutil.rmtree`` itself refuses a symlinked top
  path, but we want the same precise reason.
* **basename must match ``take_NN_YYYYMMDD_HHMMSS``** -- this is the one
  invariant the recorder itself guarantees about every directory it creates
  (see ``GelloRecorderGuiNode.start_recording``). Anything else living under
  ``output_root`` is not a take this module has any business touching.
* **must resolve to a DIRECT child of ``output_root``** -- computed via
  ``realpath`` on both sides so a crafted path with ``..`` components, or a
  symlink hiding in an *ancestor* directory, cannot walk this out of the
  recordings tree. "Direct child" also explicitly excludes the root itself
  (deleting ``output_root`` would delete every other take, not just one) and
  excludes anything nested more than one level deep (the recorder never
  creates take directories inside take directories).
* **must contain only regular files, no subdirectories** -- a normal take is
  flat (cam1.mp4, cam2.mp4, vectors.h5, optionally depth.h5). A subdirectory
  inside a "take" is not a shape this recorder ever produces, so it is
  refused rather than guessed about. Entries are checked with
  ``follow_symlinks=False`` so a symlink planted *inside* a take directory
  (pointing anywhere) is refused too, not silently followed and deleted.

Nothing in this module ever follows a symlink into a deletion: the top-level
symlink check plus the flat/regular-files-only check together mean
``shutil.rmtree`` only ever walks real directories/files that are actually
inside ``output_root``.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Union

StrPath = Union[str, "os.PathLike[str]"]

# The one shape ``GelloRecorderGuiNode.start_recording`` ever creates:
# ``take_{idx:02d}_{YYYYmmdd_HHMMSS}``. ``idx`` is zero-padded to at least two
# digits by ``f"{take_idx:02d}"`` but is not capped, so three-plus-digit take
# counts (take_100_...) must still match -- hence ``\d{2,}``, not ``\d{2}``.
TAKE_DIR_RE = re.compile(r"^take_\d{2,}_\d{8}_\d{6}$")

# Same shape as TAKE_DIR_RE but with the index captured, used only to pull
# the NN back out once TAKE_DIR_RE has already confirmed the full name is
# well-formed. Kept as a second, private pattern rather than adding a group
# to TAKE_DIR_RE so the public constant's text stays exactly the one thing
# the recorder's directory-naming convention is defined by.
_INDEX_CAPTURE_RE = re.compile(r"^take_(\d+)_\d{8}_\d{6}$")


class TakeDeleteError(ValueError):
    """A take directory failed validation and will not be touched."""


def take_index_from_dir(take_dir: StrPath) -> int:
    """Parse the ``NN`` take index out of a ``take_NN_YYYYMMDD_HHMMSS`` dir.

    Raises ``TakeDeleteError`` if the basename does not match that shape --
    this is used both to report/rename indices and (via
    ``next_take_index_after_delete``) to decide whether a deleted take was
    the most-recently-allocated one, so a silent wrong parse would corrupt
    the take counter rather than just fail loudly.
    """
    basename = os.path.basename(os.path.normpath(os.fspath(take_dir)))
    match = _INDEX_CAPTURE_RE.match(basename)
    if not match:
        raise TakeDeleteError(
            f"not a take directory name (expected take_NN_YYYYMMDD_HHMMSS): "
            f"{basename!r}"
        )
    return int(match.group(1))


def validate_take_dir(take_dir: StrPath, output_root: StrPath) -> str:
    """Validate that ``take_dir`` is safe to ``rmtree``; return its realpath.

    See the module docstring for why each guard below exists. Every failure
    raises ``TakeDeleteError`` with a reason specific to which guard tripped
    -- callers (the GUI, in particular) surface that message directly to the
    operator, so it needs to say what was wrong, not just that something was.
    """
    take_dir = os.fspath(take_dir)
    output_root = os.fspath(output_root)

    # lexists (not exists): a broken symlink must reach the islink check
    # below and be refused AS a symlink, not reported as "does not exist".
    if not os.path.lexists(take_dir):
        raise TakeDeleteError(f"take directory does not exist: {take_dir!r}")

    if os.path.islink(take_dir):
        raise TakeDeleteError(
            f"refusing to delete a symlink (not a real take directory): "
            f"{take_dir!r}"
        )

    if not os.path.isdir(take_dir):
        raise TakeDeleteError(f"not a directory: {take_dir!r}")

    basename = os.path.basename(os.path.normpath(take_dir))
    if not TAKE_DIR_RE.match(basename):
        raise TakeDeleteError(
            f"directory name does not look like a take (expected "
            f"take_NN_YYYYMMDD_HHMMSS): {basename!r}"
        )

    root_real = os.path.realpath(output_root)
    resolved = os.path.realpath(take_dir)

    if resolved == root_real:
        raise TakeDeleteError(
            f"refusing to delete output_root itself: {take_dir!r}"
        )
    if os.path.dirname(resolved) != root_real:
        raise TakeDeleteError(
            f"not a direct child of output_root {output_root!r}: "
            f"{take_dir!r} (resolved to {resolved!r})"
        )

    with os.scandir(resolved) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                if entry.is_symlink():
                    kind = "symlink"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "subdirectory"
                else:
                    kind = "special file"
                raise TakeDeleteError(
                    f"take directory contains an unexpected {kind} "
                    f"({entry.name!r}) -- refusing to delete: {take_dir!r}"
                )

    return resolved


def take_dir_summary(take_dir: StrPath) -> dict:
    """Return ``{"n_files": int, "bytes": int}`` for a (validated) take dir.

    Only counts regular files (``follow_symlinks=False``), matching the
    "flat, regular files only" invariant ``validate_take_dir`` enforces --
    called on an already-validated directory this is exactly every entry,
    but it does not assume that and simply ignores anything else rather than
    raising, so it stays safe to call for diagnostics on its own.
    """
    n_files = 0
    total_bytes = 0
    with os.scandir(os.fspath(take_dir)) as it:
        for entry in it:
            if entry.is_file(follow_symlinks=False):
                n_files += 1
                total_bytes += entry.stat(follow_symlinks=False).st_size
    return {"n_files": n_files, "bytes": total_bytes}


def delete_take_dir(take_dir: StrPath, output_root: StrPath) -> dict:
    """Validate, summarise, then permanently delete ``take_dir``.

    Returns ``{"path": <realpath>, "n_files": int, "bytes": int}`` -- the
    counts are taken BEFORE the delete so they describe what was actually
    removed. Never follows a symlink: ``validate_take_dir`` has already
    guaranteed ``take_dir`` itself is not a symlink and contains no symlinks,
    so ``shutil.rmtree`` cannot be tricked into deleting anything outside
    the validated directory tree.
    """
    resolved = validate_take_dir(take_dir, output_root)
    summary = take_dir_summary(resolved)
    shutil.rmtree(resolved)
    return {"path": resolved, "n_files": summary["n_files"], "bytes": summary["bytes"]}


def next_take_index_after_delete(current_index: int, deleted_dir: StrPath) -> int:
    """Return the take-index counter to use after deleting ``deleted_dir``.

    If the deleted take was the most-recently-allocated index
    (``current_index``), the counter is decremented by one so the NEXT
    recording REUSES that number -- safe because the take directory name
    always carries a fresh timestamp, so the reused ``NN`` cannot collide
    with the just-deleted folder (different timestamp) or any earlier one
    (different NN).

    ``GelloRecorderGuiNode`` only ever offers the most-recently-STOPPED take
    for deletion, so in practice ``deleted_dir`` is always the current-index
    take and this always decrements. The "leave it alone otherwise" branch
    exists so this pure function is correct (and testable) independent of
    that caller guarantee: renumbering an EARLIER, already-final take would
    be surprising and is not needed to avoid any collision.
    """
    deleted_index = take_index_from_dir(deleted_dir)
    if deleted_index == current_index:
        return current_index - 1
    return current_index
