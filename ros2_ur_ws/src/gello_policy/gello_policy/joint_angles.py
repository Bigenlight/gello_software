"""Helpers for keeping periodic UR joint angles in the policy convention."""

import math


def nearest_equivalent(angle, reference):
    """Return the 2*pi-equivalent of ``angle`` nearest to ``reference``."""
    return float(reference) + math.remainder(float(angle) - float(reference), math.tau)


def positions_near_reference(positions, reference):
    """Express each periodic joint position in the corresponding reference branch."""
    if len(positions) != len(reference):
        raise ValueError("positions and reference must have the same length")
    return [nearest_equivalent(value, ref) for value, ref in zip(positions, reference)]


def angular_deviations(positions, reference):
    """Return absolute shortest angular deviations for paired joint positions."""
    normalized = positions_near_reference(positions, reference)
    return [abs(value - ref) for value, ref in zip(normalized, reference)]
