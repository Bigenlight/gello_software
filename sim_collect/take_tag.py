"""Safe folder-name tags for completed sim_collect takes."""

from __future__ import annotations

import os
import re


class TakeTagError(RuntimeError):
    """The requested take tag or source directory is unsafe."""


_TAKE_NAME_RE = re.compile(
    r"^take_[0-9]+_[0-9]{8}_[0-9]{6}(?:_[a-z0-9]+(?:_[a-z0-9]+)*)?$"
)
_TAG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def normalize_tag(raw_tag: str) -> str:
    tag = re.sub(r"[\s-]+", "_", str(raw_tag).strip().lower())
    tag = re.sub(r"_+", "_", tag).strip("_")
    if not tag:
        raise TakeTagError("tag is empty")
    if len(tag) > 64:
        raise TakeTagError("tag is too long (maximum 64 characters)")
    if not _TAG_RE.fullmatch(tag):
        raise TakeTagError("tag may contain only letters, numbers, spaces, '-' and '_'")
    return tag


def tag_take_dir(take_dir: str, output_root: str, raw_tag: str) -> dict:
    """Atomically append a normalized tag to one completed take directory."""
    tag = normalize_tag(raw_tag)
    root = os.path.realpath(os.path.expanduser(output_root))
    source_arg = os.path.expanduser(take_dir)
    source = os.path.realpath(source_arg)
    basename = os.path.basename(source)

    if os.path.dirname(source) != root:
        raise TakeTagError("take is not an immediate child of output_root")
    if not _TAKE_NAME_RE.fullmatch(basename):
        raise TakeTagError(f"take folder has an unexpected name: {basename}")
    if not os.path.isdir(source) or os.path.islink(source_arg):
        raise TakeTagError("take folder does not exist or is a symlink")

    suffix = "_" + tag
    if basename.endswith(suffix):
        return {"old_path": source, "path": source, "tag": tag, "changed": False}
    destination = os.path.join(root, basename + suffix)
    if os.path.lexists(destination):
        raise TakeTagError(f"tagged destination already exists: {destination}")
    os.rename(source, destination)
    return {"old_path": source, "path": destination, "tag": tag, "changed": True}
