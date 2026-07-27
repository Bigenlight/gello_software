"""Annotation-only TensorFlow compatibility shim for upstream HIL-SERL.

The learner only needs ``tf.Tensor`` in ``serl_launcher.common.typing``.
Shipping the full TensorFlow runtime solely for that annotation is both large
and unnecessary.  TensorFlow I/O is deliberately unavailable: code paths
which need it must install and run in a real TensorFlow environment instead of
silently receiving a partial implementation.
"""

from __future__ import annotations

from typing import NoReturn

from tensorflow import errors, io


class Tensor:
    """Marker class used only while evaluating upstream type aliases."""


class Variable:
    """Marker queried by einops while selecting a non-TensorFlow backend."""


__version__ = "0+hil-serl-annotation-shim"


def __getattr__(name: str) -> NoReturn:
    raise RuntimeError(
        "the HIL-SERL TensorFlow compatibility shim only provides tf.Tensor; "
        f"requested tensorflow.{name}"
    )
