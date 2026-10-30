"""Minimal exception names imported by Flax's optional TensorFlow backend."""


class NotFoundError(FileNotFoundError):
    """Placeholder; TensorFlow I/O calls fail before this can be raised."""
