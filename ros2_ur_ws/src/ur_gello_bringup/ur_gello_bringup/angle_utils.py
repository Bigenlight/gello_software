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


def leader_quasi_still(history, window_s: float, max_speed: float) -> bool:
    """SAFETY-CRITICAL gate: True only if the leader is demonstrably quasi-still.

    ``history`` is an ordered sequence (OLDEST-FIRST) of
    ``(monotonic_ts: float, pose: list[float])`` tuples — the recent samples of
    the GELLO leader. The gate returns True only when the leader's per-joint
    angular speed, measured over roughly the last ``window_s`` seconds, is
    <= ``max_speed`` (rad/s) on EVERY joint.

    CONSERVATIVE-DEFAULT CONTRACT (safety-critical — do not weaken)
    --------------------------------------------------------------
    This gate authorizes physical robot motion (a resume/catch-up), so it MUST
    fail closed: whenever the evidence is insufficient to POSITIVELY demonstrate
    stillness it returns ``False`` ("not known to be still"), never True. In
    particular it returns False when:

      * there are fewer than 2 samples; or
      * the samples inside the window span less than half of ``window_s``
        (a sparse or just-started stream — too little temporal coverage to
        trust a speed estimate).

    Stillness must be positively demonstrated by real, time-spanning data; a
    frozen or dead stream can therefore never falsely gate as "still".

    Branch-cut aware: per-joint displacement uses ``circular_dist`` so a joint
    dithering across the +/-pi cut is not read as a spurious ~2*pi excursion.

    Does NOT assume 6 joints — it iterates over ``len(newest_pose)``.
    """
    # (1) Need at least two samples to measure any displacement over time.
    if len(history) < 2:
        return False
    # (2) Newest sample is the last (oldest-first ordering).
    newest_t, newest_pose = history[-1]
    # (3) Scan oldest-first for the FIRST entry still inside the window.
    oldest_t, oldest_pose = newest_t, newest_pose
    cutoff = newest_t - window_s
    for ts, pose in history:
        if ts >= cutoff:
            oldest_t, oldest_pose = ts, pose
            break
    # (4) Reject insufficient temporal coverage (span guard).
    span = newest_t - oldest_t
    if span < window_s * 0.5:
        return False
    # (5) Per-joint circular speed; take the worst joint.
    speed = max(
        circular_dist(newest_pose[i], oldest_pose[i]) / span
        for i in range(len(newest_pose))
    )
    # (6) Still only if the worst joint is at or below the speed threshold.
    return speed <= max_speed
