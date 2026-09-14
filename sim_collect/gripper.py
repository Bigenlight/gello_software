"""GELLO trigger -> 2F-85 command mapping (DESIGN.md §2.1 "Gripper").

Value conventions (CLAUDE.md "그리퍼 값 규약"): trigger, grip_cmd and grip_pos are
all 0.0 = OPEN .. 1.0 = CLOSED (the topic/recording convention). The 2F-85
`fingers_actuator` takes ctrl 0..255 (0 = open), so `ctrl = grip_cmd * 255`
exactly like `gello.robots.sim_robot.MujocoRobotServer._to_ctrl`.

`discrete_latch` / `validate_discrete_thresholds` re-implement the two pure
functions of `ur_gello_bringup.gello_gripper_bridge_node` (an rclpy module, so
it cannot be imported in the .venv) with identical semantics — see
tests/test_gripper.py for the truth table.
"""
from __future__ import annotations

from typing import Optional

DISCRETE_OPEN = 0.0
DISCRETE_CLOSED = 1.0


def validate_discrete_thresholds(open_at: float, close_at: float) -> Optional[str]:
    """None if usable, else a reason. Contract: 0.0 < open_at < close_at < 1.0."""
    for name, value in (("discrete_open_at", open_at), ("discrete_close_at", close_at)):
        if value != value:  # NaN
            return f"{name}={value} is not a number"
    if not 0.0 < open_at:
        return f"discrete_open_at={open_at} must be > 0.0"
    if not close_at < 1.0:
        return f"discrete_close_at={close_at} must be < 1.0"
    if not open_at < close_at:
        return (f"discrete_open_at={open_at} must be < discrete_close_at={close_at} "
                "(they bracket the hysteresis band)")
    return None


def discrete_latch(value: float, previous: Optional[float], open_at: float, close_at: float) -> Optional[float]:
    """Fold a continuous trigger into the latched OPEN/CLOSED endpoint.

    `>= close_at` -> CLOSED (checked FIRST), `<= open_at` -> OPEN, otherwise the
    previous latch (None = never crossed a threshold -> caller keeps its last
    command rather than guessing).
    """
    if value >= close_at:
        return DISCRETE_CLOSED
    if value <= open_at:
        return DISCRETE_OPEN
    return previous


def grip_cmd_to_ctrl(grip_cmd: float) -> float:
    """grip_cmd (0 open .. 1 closed) -> fingers_actuator ctrl (0..255)."""
    return float(min(max(grip_cmd, 0.0), 1.0)) * 255.0


def grip_pos_from_driver(q_driver: float, lo: float = 0.0, hi: float = 0.871) -> float:
    """2F-85 `right_driver_joint` angle -> 0 open .. 1 closed.

    The joint range in 2f85.xml is 0..0.8 rad, but an empty-hand full close
    settles at 0.7822 rad (measured, ctrl 255) while the real Robotiq driver
    reports ~0.898 for the same state, so the default `hi` = 0.7822 / 0.898 =
    0.871 makes a closed empty hand read 0.90 like the real recordings (config
    `robot.gripper_driver_range`)."""
    if hi <= lo:
        return 0.0
    return float(min(max((q_driver - lo) / (hi - lo), 0.0), 1.0))


class GripperMapper:
    """Stateful trigger -> grip_cmd mapper with pause/resume, mirroring the ROS
    `gello_gripper_bridge` publish decision.

    mode "continuous": the command follows the trigger; a change with
    `|trigger - last| < deadband` is skipped (the ROS deadband gate — suppresses
    at-rest Dynamixel jitter). mode "discrete": 0.3/0.7 hysteresis latch; inside
    the band the last command is kept.
    `pause()` freezes the command. `resume(now)` starts the ROS resume ramp: for
    `resume_ramp_s` the command SLEWS at `resume_slew_per_s` from its current
    value toward the (latched) leader value, bypassing the deadband, so a leader
    far from the held command never produces a jump (a latched endpoint is the
    worst case: the whole 0..1 stroke, ~1.67 s at 0.6/s).
    """

    def __init__(self, mode: str = "continuous", deadband: float = 0.02,
                 open_at: float = 0.3, close_at: float = 0.7, initial: float = 0.0,
                 resume_ramp_s: float = 2.0, resume_slew_per_s: float = 0.6,
                 staleness_timeout_s: float = 0.5) -> None:
        if mode not in ("continuous", "discrete"):
            raise ValueError(f"gripper mode must be continuous|discrete, got {mode!r}")
        err = validate_discrete_thresholds(open_at, close_at)
        if err is not None:
            raise ValueError(err)
        self.mode = mode
        self.deadband = float(deadband)
        self.open_at = float(open_at)
        self.close_at = float(close_at)
        self.resume_ramp_s = float(resume_ramp_s)
        self.resume_slew_per_s = float(resume_slew_per_s)
        self.staleness_timeout_s = float(staleness_timeout_s)
        self.grip_cmd = float(min(max(initial, 0.0), 1.0))
        self._latch: Optional[float] = None
        self._last_trigger: Optional[float] = None
        self._ramp_until: Optional[float] = None
        self._last_t: Optional[float] = None
        self.paused = False

    @property
    def ramping(self) -> bool:
        return self._ramp_until is not None

    def pause(self) -> None:
        self.paused = True

    def resume(self, now: Optional[float] = None) -> None:
        """Un-pause and start the slew ramp (ROS `~/resume`)."""
        import time as _t
        now = _t.monotonic() if now is None else float(now)
        self.paused = False
        self._last_trigger = None  # the ramp publishes unconditionally; deadband re-bases after it
        self._ramp_until = now + self.resume_ramp_s if self.resume_ramp_s > 0 else None
        self._last_t = now

    def reset(self, grip_cmd: float = 0.0) -> None:
        """Force the command (e.g. `open` after a scene reset); latch/deadband/ramp restart."""
        self.grip_cmd = float(min(max(grip_cmd, 0.0), 1.0))
        self._latch = None
        self._last_trigger = None
        self._ramp_until = None

    def _target(self, t: float) -> Optional[float]:
        """Leader value after the (optional) discrete latch; None = unknown (keep)."""
        if self.mode == "discrete":
            latched = discrete_latch(t, self._latch, self.open_at, self.close_at)
            if latched is None:
                return None
            self._latch = latched
            return latched
        return t

    def update(self, trigger: float, now: Optional[float] = None) -> float:
        """Feed one trigger sample (0 open .. 1 closed); returns the current grip_cmd."""
        import time as _t
        now = _t.monotonic() if now is None else float(now)
        if self.paused:
            return self.grip_cmd
        t = float(min(max(trigger, 0.0), 1.0))
        target = self._target(t)
        if target is None:
            return self.grip_cmd
        if self._ramp_until is not None:
            if now < self._ramp_until:
                # RESUME RAMP: slew from the current command toward the leader,
                # bypassing the deadband; the slew budget is only the time the
                # output was free to move, capped like the ROS node.
                elapsed = min(max(now - (self._last_t if self._last_t is not None else now), 0.0),
                              self.staleness_timeout_s)
                max_delta = self.resume_slew_per_s * elapsed
                delta = target - self.grip_cmd
                if delta > max_delta:
                    self.grip_cmd += max_delta
                elif delta < -max_delta:
                    self.grip_cmd -= max_delta
                else:
                    self.grip_cmd = target
                self._last_t = now
                self._last_trigger = self.grip_cmd
                return self.grip_cmd
            self._ramp_until = None
        # DEADBAND gate (continuous): skip tiny at-rest changes.
        if self.mode == "continuous" and self._last_trigger is not None and abs(target - self._last_trigger) < self.deadband:
            return self.grip_cmd
        self._last_trigger = target
        self.grip_cmd = target
        self._last_t = now
        return self.grip_cmd

    @property
    def ctrl(self) -> float:
        return grip_cmd_to_ctrl(self.grip_cmd)
