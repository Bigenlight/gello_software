"""Filesystem defaults that follow the checkout used by the recorder."""

import os
from pathlib import Path


def default_repo_root() -> str:
    """Return ``GELLO_REPO_ROOT`` or the repository containing this package."""
    configured = os.environ.get("GELLO_REPO_ROOT", "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))

    package_path = Path(__file__).resolve()
    for parent in package_path.parents:
        if (parent / "ros2_ur_ws" / "src" / "gello_recorder").is_dir():
            return str(parent)
    return str(Path.home() / "gello_software")
