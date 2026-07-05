#!/usr/bin/env python3
"""Measure the real UR7e's steady-state arrival error and validate the
GELLO->UR7e handshake tolerance chain, so `chase_tol` etc. can be set from data
instead of by eye. Run it AFTER a handshake has converged (or at any moment the
arm is holding and you are holding GELLO dead still).

The convergence gate needs, on real hardware:

    e  <=  arrival_tolerance  <  chase_tol  <=  resume_align_tol

where `e` is the per-joint steady-state disagreement between the arm's ACTUAL
pose (`/joint_states`) and the LIVE leader (`/gello/joint_states`). The fake mock
arrives exactly (e~=0) and hides any real residual; this tool measures the real
`e` and tells you whether the deployed tolerances are safe or need retuning.

Usage (needs ROS, system python):
    /usr/bin/python3 scripts/calibrate_handshake.py                 # measure 3s, deployed tolerances
    /usr/bin/python3 scripts/calibrate_handshake.py --secs 5
    /usr/bin/python3 scripts/calibrate_handshake.py --arrival 0.05 --chase-tol 0.06 --resume-align 0.08

Hold the leader STILL while it samples — the tool warns if it detects the leader
(or arm) moving during the window, because `e` is only meaningful when both are
static.
"""
import argparse
import statistics as stats
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

UR_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
# Deployed defaults (config/ur7e_gello.yaml). Override with flags if you retune.
DEF_ARRIVAL, DEF_CHASE, DEF_RESUME = 0.05, 0.06, 0.08
MOVE_WARN = 0.01  # rad: per-joint spread above this over the window => "was moving"


class Collector(Node):
    def __init__(self):
        super().__init__("handshake_calibrator")
        self.actual = {}   # joint -> list of positions
        self.gello = {}    # joint -> list of positions
        self.create_subscription(JointState, "/joint_states",
                                 lambda m: self._grab(m, self.actual), 10)
        self.create_subscription(JointState, "/gello/joint_states",
                                 lambda m: self._grab(m, self.gello), 10)

    @staticmethod
    def _grab(m, store):
        for name, pos in zip(m.name, m.position):
            if name in UR_JOINTS:
                store.setdefault(name, []).append(float(pos))


def _spread(xs):
    """Max-min over samples — a cheap 'was it moving' proxy."""
    return (max(xs) - min(xs)) if len(xs) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=float, default=3.0, help="sampling window (s)")
    ap.add_argument("--arrival", type=float, default=DEF_ARRIVAL,
                    help="arrival_tolerance (FollowJointTrajectory goal tol)")
    ap.add_argument("--chase-tol", type=float, default=DEF_CHASE,
                    help="chase_tol (convergence gate)")
    ap.add_argument("--resume-align", type=float, default=DEF_RESUME,
                    help="resume_align_tol (bridge ~/resume gate)")
    a = ap.parse_args()

    rclpy.init()
    node = Collector()
    t_end = time.monotonic() + a.secs
    while rclpy.ok() and time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)

    missing_a = [j for j in UR_JOINTS if not node.actual.get(j)]
    missing_g = [j for j in UR_JOINTS if not node.gello.get(j)]
    if missing_a or missing_g:
        print("ERROR: no samples for some joints in the window.")
        if missing_a:
            print(f"  /joint_states missing: {missing_a} (is the robot/driver up?)")
        if missing_g:
            print(f"  /gello/joint_states missing: {missing_g} (is GELLO publishing?)")
        rclpy.shutdown()
        return 2

    # Per-joint steady-state error e_j = |median(actual) - median(gello)|.
    print(f"\nSampled {a.secs:.1f}s.  Per-joint steady-state error "
          f"e_j = |actual - GELLO|:\n")
    print(f"  {'joint':<22}{'e_j (rad)':>11}{'e_j (deg)':>11}   moved?")
    moving = []
    e_by_joint = {}
    for j in UR_JOINTS:
        med_a = stats.median(node.actual[j])
        med_g = stats.median(node.gello[j])
        e = abs(med_a - med_g)
        e_by_joint[j] = e
        spread = max(_spread(node.actual[j]), _spread(node.gello[j]))
        flag = "  <-- MOVING" if spread > MOVE_WARN else ""
        if spread > MOVE_WARN:
            moving.append(j)
        import math
        print(f"  {j:<22}{e:>11.4f}{math.degrees(e):>11.2f}{flag}")

    e = max(e_by_joint.values())
    worst = max(e_by_joint, key=e_by_joint.get)
    print(f"\n  e = max_j e_j = {e:.4f} rad ({e*57.2958:.2f} deg)  at {worst}")

    if moving:
        print(f"\n  WARNING: {moving} moved >{MOVE_WARN} rad during the window — "
              f"`e` is unreliable. Hold GELLO and the arm DEAD STILL and re-run.")

    # Validate the chain e <= arrival < chase_tol <= resume_align.
    print(f"\nTolerance chain check  (deployed: arrival={a.arrival} "
          f"chase_tol={a.chase_tol} resume_align_tol={a.resume_align}):\n")
    ok = True

    def check(cond, ok_msg, bad_msg):
        nonlocal ok
        print(("  [PASS] " + ok_msg) if cond else ("  [FAIL] " + bad_msg))
        ok = ok and cond

    check(e <= a.arrival,
          f"e ({e:.4f}) <= arrival_tolerance ({a.arrival}) — catch-up trajectories will SUCCEED",
          f"e ({e:.4f}) > arrival_tolerance ({a.arrival}) — catch-up trajectories will ABORT; "
          f"raise arrival_tolerance above {e:.4f}")
    check(a.arrival < a.chase_tol,
          f"arrival_tolerance ({a.arrival}) < chase_tol ({a.chase_tol}) — no dead-band livelock",
          f"arrival_tolerance ({a.arrival}) >= chase_tol ({a.chase_tol}) — DEAD-BAND LIVELOCK "
          f"(gate never passes; node auto-raises chase_tol with a warning)")
    check(a.chase_tol <= a.resume_align,
          f"chase_tol ({a.chase_tol}) <= resume_align_tol ({a.resume_align}) — resume won't refuse a converged arm",
          f"chase_tol ({a.chase_tol}) > resume_align_tol ({a.resume_align}) — a converged arm can be "
          f"REFUSED at ~/resume; raise resume_align_tol to >= {a.chase_tol}")

    # Recommendation: tightest safe chain given measured e (margins ~1.5x / +0.01).
    import math
    rec_arrival = max(round(1.5 * e + 0.005, 3), 0.01)
    rec_chase = round(rec_arrival + 0.01, 3)
    rec_resume = round(rec_chase + 0.02, 3)
    print(f"\nRecommended tightest-safe chain for measured e={e:.4f}:")
    print(f"  arrival_tolerance: {rec_arrival}   chase_tol: {rec_chase}   "
          f"resume_align_tol: {rec_resume}")
    print("  (The deployed defaults 0.05/0.06/0.08 are conservative and also safe "
          "whenever e < 0.05 — tighten only if you want a smaller handover residual.)")

    verdict = "READY" if ok and not moving else ("RECHECK (leader moved)" if moving else "FIX TOLERANCES")
    print(f"\nVERDICT: {verdict}\n")
    rclpy.shutdown()
    return 0 if (ok and not moving) else 1


if __name__ == "__main__":
    sys.exit(main())
