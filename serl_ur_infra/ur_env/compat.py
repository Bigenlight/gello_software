"""Runtime compatibility checks shared by learner and transport entrypoints."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version


class CompatibilityError(RuntimeError):
    """The process imported an incompatible dependency runtime."""


def configure_flax_local_io() -> None:
    """Force Flax checkpoint operations onto its local filesystem backend.

    Flax 0.10.5 selects TensorFlow I/O merely when ``tensorflow`` is
    importable.  This repository intentionally provides an annotation-only
    TensorFlow shim, so that auto-detection would select an unusable backend.
    Learner/server process entrypoints must call this idempotent compatibility
    contract before any Flax checkpoint restore or save.
    """

    try:
        import flax.io as flax_io
    except ImportError as exc:
        raise CompatibilityError(
            "Flax is required before configuring local checkpoint I/O"
        ) from exc
    try:
        flax_io.set_mode(flax_io.BackendMode.DEFAULT)
    except Exception as exc:
        raise CompatibilityError(
            f"failed to select Flax local checkpoint I/O: {exc}"
        ) from exc
    if flax_io.io_mode is not flax_io.BackendMode.DEFAULT:
        raise CompatibilityError("Flax refused the local checkpoint I/O backend")


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
