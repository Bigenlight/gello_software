"""Default RealSense serials, read from ``ros2_ur_ws/camera_serials.env``.

The lab's camera serials used to be hard-coded in several scripts and in
``gello_recorder_gui``. They now live in one KEY=VALUE file that the shell
launchers (via ``_resolve_camera_serials.sh``) and the Python GUIs both read.

Precedence (highest first):

1. ``CAM1_SERIAL`` / ``CAM2_SERIAL`` environment variables (applied by callers).
2. The file named by ``GELLO_CAMERA_SERIALS_FILE``.
3. ``<repo>/ros2_ur_ws/camera_serials.env``.
4. ``""`` -- no default; the serial resolver then reports what is connected.
"""

import os
from typing import Dict, Optional, Tuple

from gello_recorder.paths import default_repo_root

SERIALS_FILE_ENV_VAR = "GELLO_CAMERA_SERIALS_FILE"
SERIALS_FILE_NAME = "camera_serials.env"


def default_serials_file() -> str:
    """Path of the serials file (env override, else the repo default)."""
    configured = os.environ.get(SERIALS_FILE_ENV_VAR, "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(default_repo_root(), "ros2_ur_ws", SERIALS_FILE_NAME)


def parse_env_file(text: str) -> Dict[str, str]:
    """Parse ``KEY=VALUE`` lines; ``#`` starts a comment, quotes are stripped."""
    values = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        values[key] = value.strip().strip("'\"")
    return values


def load_default_serials(path: Optional[str] = None) -> Tuple[str, str]:
    """Return ``(cam1_serial, cam2_serial)`` from the serials file.

    A missing or unreadable file yields ``("", "")`` rather than raising, so
    importing the GUI module never depends on the lab file being present.
    """
    path = path or default_serials_file()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            values = parse_env_file(fh.read())
    except OSError:
        return "", ""
    return values.get("CAM1_SERIAL", ""), values.get("CAM2_SERIAL", "")
