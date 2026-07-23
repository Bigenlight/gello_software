"""ROS-independent verdict logic for remote-policy fake-UR7e validation."""

from __future__ import annotations

import math


UR_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def angular_error(a: float, b: float) -> float:
    """Smallest absolute distance between two wrapped joint angles."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def reorder_joint_state(names, positions):
    by_name = dict(zip(names, positions))
    if any(name not in by_name for name in UR_JOINTS):
        return None
    values = tuple(float(by_name[name]) for name in UR_JOINTS)
    return values if all(math.isfinite(value) for value in values) else None


class FakeUr7eVerdict:
    """Accumulate public command/joint observations into deterministic gates."""

    def __init__(self, *, tolerance: float, hold_stable_s: float = 0.75,
                 max_command_gap_s: float = 0.25):
        if tolerance <= 0 or hold_stable_s <= 0 or max_command_gap_s <= 0:
            raise ValueError("validation thresholds must be positive")
        self.tolerance = tolerance
        self.hold_stable_s = hold_stable_s
        self.max_command_gap_s = max_command_gap_s
        self.phase = "initial_hold"
        self.failure = ""
        self.initial_command = None
        self.initial_since = None
        self.last_command = None
        self.last_command_time = None
        self.live_joints = None
        self.post_arm_finite = False
        self.tracking_seen = False
        self.max_displacement = 0.0
        self.hold_reference = None
        self.hold_since = None
        self.policy_state = ""

    def observe_policy_state(self, state: str):
        self.policy_state = str(state)

    def observe_joint_state(self, names, positions):
        values = reorder_joint_state(names, positions)
        if values is not None:
            self.live_joints = values
        return values

    def observe_command(self, now: float, values):
        command = tuple(float(value) for value in values)
        if len(command) != 6 or not all(math.isfinite(value) for value in command):
            self.failure = "controller command must contain 6 finite values"
            return
        if self.last_command_time is not None and now - self.last_command_time > self.max_command_gap_s:
            if self.phase == "hold_after_execution":
                self.failure = "controller command publication stopped during HOLD"
        self.last_command, self.last_command_time = command, now

        if self.phase == "initial_hold":
            if self.initial_command is None:
                self.initial_command, self.initial_since = command, now
            elif max(angular_error(a, b) for a, b in zip(command, self.initial_command)) > self.tolerance:
                self.initial_command, self.initial_since = command, now
        elif self.phase == "executing":
            # start_execution returns when asynchronous gRPC ARMING begins.
            # HOLD commands emitted during ARMING are not policy actions.
            if self.policy_state != "EXECUTE":
                return
            self.post_arm_finite = True
            if self.initial_command is not None:
                self.max_displacement = max(
                    self.max_displacement,
                    max(angular_error(a, b) for a, b in zip(command, self.initial_command)),
                )
            if self.live_joints is not None:
                error = max(angular_error(a, b) for a, b in zip(command, self.live_joints))
                self.tracking_seen |= error <= self.tolerance
        elif self.phase == "hold_after_execution" and self.hold_reference is not None:
            error = max(angular_error(a, b) for a, b in zip(command, self.hold_reference))
            if error <= self.tolerance:
                self.hold_since = now if self.hold_since is None else self.hold_since
            else:
                self.hold_since = None

    def initial_hold_ready(self, now: float) -> bool:
        return (
            not self.failure and self.initial_since is not None
            and now - self.initial_since >= self.hold_stable_s
        )

    def mark_started(self):
        if not self.initial_hold_ready(self.last_command_time or 0.0):
            raise RuntimeError("initial HOLD stability was not established")
        self.phase = "executing"

    def execution_ready_for_hold(self) -> bool:
        return not self.failure and self.post_arm_finite and self.tracking_seen

    def mark_hold(self, now: float):
        if self.live_joints is None:
            raise RuntimeError("cannot validate HOLD without mock joint state")
        self.phase = "hold_after_execution"
        self.hold_reference = self.live_joints
        self.hold_since = None
        self.last_command_time = now

    def passed(self, now: float) -> bool:
        return (
            not self.failure and self.phase == "hold_after_execution"
            and self.hold_since is not None
            and now - self.hold_since >= self.hold_stable_s
            and self.last_command_time is not None
            and now - self.last_command_time <= self.max_command_gap_s
        )
