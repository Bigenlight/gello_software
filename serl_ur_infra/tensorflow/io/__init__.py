"""Import-compatible, runtime-disabled ``tensorflow.io`` surface."""

from __future__ import annotations

from typing import NoReturn


class _UnavailableGFile:
    def __getattr__(self, name: str) -> NoReturn:
        raise RuntimeError(
            "TensorFlow I/O is unavailable in the HIL-SERL annotation-only "
            f"compatibility shim (requested tf.io.gfile.{name})"
        )


gfile = _UnavailableGFile()
