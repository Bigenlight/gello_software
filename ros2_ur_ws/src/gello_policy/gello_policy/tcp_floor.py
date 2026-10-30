"""TCP Z-floor clamp: pure numpy, no rclpy, testable offline.

Motivation (2026-09-21): the multi-task GR00T/VLA policies are executed under
the carrot joint-box envelope only; a joint box does not bound the tool centre
point, and several VLA trials drove the Robotiq 2F-85 into the table.  This
module evaluates the TCP height of a commanded joint target with the package
forward kinematics (``ur_gello_bringup.ur_kin.fk``, base_link -> tool0) plus a
tool offset along flange +Z, and shortens the commanded step along the straight
joint-space segment ``live_q -> target`` until the TCP stays at or above the
floor.

Floor value provenance: measured ``tcp_pose/z`` minima over the four Bigenlight
GELLO task datasets (see ``FM_RL/vla_eval/models/n17/tcp_z_floor_report.json``).

Joint order: the six arm joints in UR order
(shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3), exactly what
``policy_leader_node`` publishes (``obs_assembler.UR_JOINT_ORDER``).  FK is
2*pi-periodic per joint, so the unwrapped start_pose branch used by the leader
is irrelevant for the height.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ur_gello_bringup import ur_kin

_N = 6


def tool_transform(tool_z: float) -> np.ndarray:
    """flange -> TCP transform for a tool that extends ``tool_z`` m along flange +Z."""
    T = np.eye(4)
    T[2, 3] = float(tool_z)
    return T


def tcp_z(q: Sequence[float], tool_z: float) -> float:
    """TCP height (m, base frame) for joint vector ``q`` (UR order, radians)."""
    q = np.asarray(q, dtype=float).reshape(_N)
    T = ur_kin.fk(q) @ tool_transform(tool_z)
    return float(T[2, 3])


@dataclass(frozen=True)
class FloorClamp:
    target: list        # joint target actually to publish (UR order)
    z_raw: float        # TCP z of the requested target
    z: float            # TCP z of the returned target
    alpha: float        # fraction of the live_q -> target step kept (1.0 = untouched)
    clamped: bool


def clamp_to_floor(live_q: Sequence[float], target: Sequence[float], z_min: float,
                   tool_z: float, iterations: int = 12) -> FloorClamp:
    """Largest alpha in [0, 1] with TCP z(live_q + alpha*(target-live_q)) >= z_min.

    * target already above the floor -> returned unchanged (alpha = 1).
    * otherwise bisection on alpha; the interpolated joint vector is returned.
    * if even alpha = 0 (the live pose itself) is below the floor, the live pose
      is returned: the arm never commands further downward motion, and any
      upward request that clears the floor is still honoured by the first test.
    """
    live = np.asarray(live_q, dtype=float).reshape(_N)
    tgt = np.asarray(target, dtype=float).reshape(_N)
    z_raw = tcp_z(tgt, tool_z)
    if z_raw >= z_min:
        return FloorClamp([float(v) for v in tgt], z_raw, z_raw, 1.0, False)
    z_live = tcp_z(live, tool_z)
    if z_live < z_min:
        return FloorClamp([float(v) for v in live], z_raw, z_live, 0.0, True)
    lo, hi = 0.0, 1.0        # z(lo) >= z_min, z(hi) < z_min
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        if tcp_z(live + mid * (tgt - live), tool_z) >= z_min:
            lo = mid
        else:
            hi = mid
    out = live + lo * (tgt - live)
    return FloorClamp([float(v) for v in out], z_raw, tcp_z(out, tool_z), lo, True)
