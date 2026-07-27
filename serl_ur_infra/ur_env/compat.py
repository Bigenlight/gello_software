"""Runtime compatibility checks shared by learner and transport entrypoints."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version


class CompatibilityError(RuntimeError):
    """The process imported an incompatible dependency runtime."""


def configure_pure_python_protobuf() -> None:
    """Select protobuf's pure-Python implementation, or fail if it is too late.

    The environment variable is read when protobuf first imports.  Calling
    this function after an upb-backed protobuf import cannot safely repair the
    process, so that case is reported explicitly.
    """

    try:
        protobuf_major = int(version("protobuf").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        protobuf_major = 4
    if protobuf_major < 4:
        return
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    if "google.protobuf" not in sys.modules:
        return
    from google.protobuf.internal import api_implementation

    implementation = api_implementation.Type()
    if implementation != "python":
        raise CompatibilityError(
            "protobuf was imported with implementation "
            f"{implementation!r} before HIL-SERL compatibility setup; start "
            "Python with serl_ur_infra on PYTHONPATH so sitecustomize can set "
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
        )
