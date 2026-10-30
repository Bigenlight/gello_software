"""Safe, atomic folder-name tags for completed recorder takes.

A tag is appended to a finished take's folder name
(``take_03_20260101_120000`` -> ``take_03_20260101_120000_failed``), so takes
can be filtered later without opening them. The GUI's "Tag last take" combo
offers the preset list from :func:`preset_tags` and also accepts free text.
"""

import os
import re
from typing import List, Optional

#: Comma-separated preset tags for the GUI combo, e.g. ``failed,review``.
TAG_PRESETS_ENV_VAR = "RECORDER_TAG_PRESETS"
#: Neutral presets used when ``RECORDER_TAG_PRESETS`` is unset.
DEFAULT_TAG_PRESETS = ("failed", "review")


class TakeTagError(RuntimeError):
    """The requested take tag or source directory is unsafe."""


_TAKE_NAME_RE = re.compile(
    r"^take_[0-9]+_[0-9]{8}_[0-9]{6}(?:_[a-z0-9]+(?:_[a-z0-9]+)*)?$"
)
_TAG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def normalize_tag(raw_tag: str) -> str:
    """Return the canonical folder suffix for an operator-entered tag."""
    tag = re.sub(r"[\s-]+", "_", str(raw_tag).strip().lower())
    tag = re.sub(r"_+", "_", tag).strip("_")
    if not tag:
        raise TakeTagError("tag is empty")
    if len(tag) > 64:
        raise TakeTagError("tag is too long (maximum 64 characters)")
    if not _TAG_RE.fullmatch(tag):
        raise TakeTagError(
            "tag may contain only letters, numbers, spaces, '-' and '_'"
        )
    return tag


def preset_tags(raw: Optional[str] = None) -> List[str]:
    """Return the normalized, de-duplicated preset tags for the GUI combo.

    ``raw`` defaults to ``$RECORDER_TAG_PRESETS``; when that is unset the
    :data:`DEFAULT_TAG_PRESETS` are used. Invalid entries are skipped (the
    operator can still type any valid tag by hand), and an explicitly empty
    value yields no presets.
    """
    if raw is None:
        raw = os.environ.get(TAG_PRESETS_ENV_VAR)
    if raw is None:
        candidates = list(DEFAULT_TAG_PRESETS)
    else:
        candidates = str(raw).split(",")
    tags = []
    for candidate in candidates:
        try:
            tag = normalize_tag(candidate)
        except TakeTagError:
            continue
        if tag not in tags:
            tags.append(tag)
    return tags


def tag_take_dir(take_dir: str, output_root: str, raw_tag: str) -> dict:
    """Atomically append a normalized tag to one completed take directory."""
    tag = normalize_tag(raw_tag)
    root = os.path.realpath(os.path.expanduser(output_root))
    source = os.path.realpath(os.path.expanduser(take_dir))
    basename = os.path.basename(source)

    if os.path.dirname(source) != root:
        raise TakeTagError("take is not an immediate child of output_root")
    if not _TAKE_NAME_RE.fullmatch(basename):
        raise TakeTagError("take folder has an unexpected name: {}".format(basename))
    if not os.path.isdir(source) or os.path.islink(os.path.expanduser(take_dir)):
        raise TakeTagError("take folder does not exist or is a symlink")

    suffix = "_" + tag
    if basename.endswith(suffix):
        return {"old_path": source, "path": source, "tag": tag, "changed": False}

    destination = os.path.join(root, basename + suffix)
    if os.path.lexists(destination):
        raise TakeTagError(
            "tagged destination already exists: {}".format(destination)
        )

    os.rename(source, destination)
    return {
        "old_path": source,
        "path": destination,
        "tag": tag,
        "changed": True,
    }
