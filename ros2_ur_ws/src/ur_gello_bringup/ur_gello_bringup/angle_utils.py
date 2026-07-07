"""Branch-cut-safe angle helpers shared by the GELLO->UR nodes.

The UR scaled_joint_trajectory_controller and forward_position_controller both
interpolate LINEARLY in raw joint space with no 2*pi awareness. With a wrist
joint_offset near +/-pi (e.g. wrist_3 rotated ~180 deg so a leader-mounted
gripper stays ergonomic while a follower-mounted camera faces up), normal
operation sits right next to the +/-pi branch cut, so a raw GELLO value and the
arm's actual value can be ~2*pi apart numerically while physically ~0 apart.
Every angular COMPARISON must therefore be circular, and every commanded
TARGET must be the shortest-path equivalent to the arm's current pose. Both
derive from the single primitive wrap_to_pi().
"""

import math


def wrap_to_pi(x: float) -> float:
    """Wrap an angle (or angle difference) into [-pi, pi]."""
    return math.remainder(x, 2.0 * math.pi)


def circular_dist(a: float, b: float) -> float:
    """Smallest absolute angular distance (rad) between a and b, branch-cut aware.

    Two angles physically close but on opposite sides of the +/-pi cut (e.g.
    +3.09 vs -3.09) report ~0.05 rad, not ~6.2 rad.
    """
    return abs(wrap_to_pi(a - b))


def wrapped_nearest(target: list[float], reference: list[float]) -> list[float]:
    """Per-joint, shift ``target`` by an integer multiple of 2*pi so it becomes
    the angular equivalent NEAREST to ``reference`` (|result - reference| <= pi).

    Send this (never the raw target) to a controller that interpolates linearly
    in joint space, so it always travels the short way, never ~2*pi the long way.
    """
    return [
        reference[i] + wrap_to_pi(target[i] - reference[i])
        for i in range(len(target))
    ]
