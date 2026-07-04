#!/usr/bin/env python3
"""Offline command smoothing helpers for GELLO -> UR logs."""

from __future__ import annotations

import numpy as np


def accel_limited_command(
    target_q: np.ndarray,
    rate_hz: float,
    max_step_rad: float,
    max_accel_rad_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Track target_q with velocity and acceleration limits.

    max_step_rad keeps the existing per-publish slew cap. max_accel_rad_s2 makes
    the command velocity ramp between steps and adds distance-based braking.
    """
    target_q = np.asarray(target_q, dtype=float)
    if target_q.ndim != 2:
        raise ValueError("target_q must be a 2D array")
    if len(target_q) == 0:
        return target_q.copy(), target_q.copy()
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    if max_step_rad <= 0.0:
        raise ValueError("max_step_rad must be positive")
    if max_accel_rad_s2 <= 0.0:
        raise ValueError("max_accel_rad_s2 must be positive")

    dt = 1.0 / float(rate_hz)
    vmax = float(max_step_rad) / dt
    amax = float(max_accel_rad_s2)

    out = np.empty_like(target_q)
    vel = np.zeros(target_q.shape[1], dtype=float)
    vel_out = np.zeros_like(target_q)
    out[0] = target_q[0]

    for idx in range(1, len(target_q)):
        error = target_q[idx] - out[idx - 1]
        braking_v = np.sqrt(2.0 * amax * np.abs(error))
        desired_v = np.sign(error) * np.minimum(vmax, braking_v)
        vel += np.clip(desired_v - vel, -amax * dt, amax * dt)

        step = vel * dt
        next_q = out[idx - 1] + step
        overshoot = np.abs(step) > np.abs(error)
        if np.any(overshoot):
            next_q[overshoot] = target_q[idx, overshoot]
            vel[overshoot] = 0.0

        out[idx] = next_q
        vel_out[idx] = vel

    return out, vel_out
