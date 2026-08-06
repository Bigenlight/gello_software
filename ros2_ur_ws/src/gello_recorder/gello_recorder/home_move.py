#!/usr/bin/env python3
"""GO-HOME orchestration for the task recorder GUI -- pure Python, no ROS, no Qt.

WHAT THIS IS
------------
The recorder's GO-HOME button has to run a five-step sequence against three
different ROS interfaces (two Trigger services, controller_manager, a
FollowJointTrajectory action, and a Float32 topic), every one of which is
asynchronous and can hang.  Doing that inline in an rclpy node mixes three
hard things -- ROS plumbing, Qt event-loop etiquette, and the actual safety
sequencing -- into code that can only be exercised on a real UR7e.

So the sequencing lives here as a plain state machine with ALL ROS I/O
injected through a duck-typed ``ops`` object, which makes the interesting part
(ordering, deadlines, the fail-closed controller restore, the branch-cut-safe
target) testable on a laptop with no robot, no ROS and no display.  This module
imports nothing but the stdlib on purpose -- adding ``rclpy`` here would undo
that.

THE SEQUENCE (and why it is this order)
---------------------------------------
1. PAUSING            pause the GELLO->UR arm bridge AND the gripper bridge.
                      The leader arm is a passive, human-held device: if the
                      bridges keep streaming while we drive the follower to
                      HOME, the trajectory controller and the bridge fight over
                      the same joints.  Pausing is unconditional and idempotent
                      per the bridge contract, so it is safe to fire always.
                      Then ask controller_manager who is actually active.
2. SWITCHING_TO_JTC   forward_position_controller (what teleop streams into)
                      cannot execute a timed trajectory, so hand the joints to
                      scaled_joint_trajectory_controller.  Skipped when STJC is
                      already active.
3. MOVING             one FollowJointTrajectory goal to the branch-cut-safe
                      equivalent of HOME_JOINTS, with a distance-proportional
                      duration.
4. OPENING_GRIPPER    publish 0.0 (= OPEN) repeatedly and confirm via the
                      position feedback, then KEEP re-publishing for a short
                      settle window (see GRIPPER_OPEN_REASSERT_S).  WARN-ONLY
                      (see below).
5. RESTORING_FPC      give the joints back to forward_position_controller so
                      the operator can re-engage teleop.  This runs on the
                      failure path too -- see FAIL-CLOSED below.

Bridges are NEVER auto-resumed, on success or on failure.  Resuming teleop
means the follower starts chasing wherever the human happens to be holding the
leader right now, which after a HOME move is by definition somewhere else.  The
operator re-engages deliberately from the EEF GUI, with the leader in hand.

FAIL-CLOSED CONTROLLER RESTORE
------------------------------
Once step 2 has succeeded, forward_position_controller is deactivated, and a
bare failure at that point would leave the arm with no controller the teleop
stack can use -- the operator would have to notice and repair that by hand.  So
every failure from that point onward is recorded as *pending* and the machine
still walks through RESTORING_FPC before landing in FAILED; the FAILED message
then reports BOTH the original failure and the restore outcome.  Failures
before that point (pause, controller query, an explicitly-refused strict
switch) changed nothing, so they land in FAILED directly.

THREADING CONTRACT
------------------
``ops`` callbacks (``done_cb``) may fire on ANY thread -- in the node they run
on the rclpy executor thread.  They are therefore allowed to do exactly one
thing: record their result into an instance field (a single guarded attribute
store).  EVERY state transition happens inside tick(), which the node calls
from one 10 Hz timer.  That makes this class single-threaded-advance by
construction, which is why it carries no lock of its own; the node's own lock
around tick()/status() is all the mutual exclusion that exists.

abort() is the one other method that may be called from a foreign thread (the
Qt GUI thread, while tick() runs on the ROS timer thread).  It obeys the same
rule: it only RECORDS a request, and tick() consumes that flag -- before any
per-state handling -- and performs the transition.

Late callbacks (one that arrives after its step already moved on) are dropped
by a monotonic step id, enforced on BOTH sides: _record() refuses to STORE a
payload whose step id is no longer current, and _take_result() refuses to READ
one.  The write-side half is not redundant.  When the move deadline fires we
cancel the action and immediately start the FPC restore; the cancelled action's
late result and the restore's service reply travel different transports with no
ordering guarantee, so a write-unguarded _record() lets the stale action result
land after the restore's and overwrite it -- the restore then looks like a
timeout and the operator is told teleop is broken when FPC is in fact fine.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

# --------------------------------------------------------------------------- #
# Robot constants
# --------------------------------------------------------------------------- #

# Manually-synced copy of CubeInCupConfig.RESET_JOINTS
# (serl_ur_infra/ur_experiments/cube_in_cup.py) and of RESET_JOINTS_CSV in
# ros2_ur_ws/run_hil_preposition.sh.  ALL THREE MUST STAY IDENTICAL: recorded
# demonstrations, the HIL preposition step and this button all have to put the
# arm in the same pose, or the data and the policy disagree about where an
# episode begins.  gello_recorder deliberately has no import dependency on
# serl_ur_infra or ur_gello_bringup, so this is a copy and not an import --
# test_home_move.py pins the literal so a drift shows up as a test failure.
HOME_JOINTS = (3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331)

# The order the UR driver's controllers expect.  /joint_states does NOT publish
# in this order (it is roughly alphabetical), which is why reorder_joint_positions()
# maps by name and never by index.
UR_JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

FPC = "forward_position_controller"
STJC = "scaled_joint_trajectory_controller"

# Scale contract of robotiq_gripper_modbus_node on
# /robotiq_gripper/command_percent: 0.0 = OPEN .. 1.0 = CLOSED.
GRIPPER_OPEN_VALUE = 0.0
# /robotiq_gripper/position_percent at or below this counts as open.  Same
# 0.15 the HIL preposition script uses -- the Robotiq never settles at exactly
# 0.0 and the fingers are unambiguously open well before that.
GRIPPER_OPEN_CONFIRM = 0.15

# --------------------------------------------------------------------------- #
# Timing defaults (all overridable through the constructor)
# --------------------------------------------------------------------------- #

# Worst-joint angular speed the generated trajectory is allowed to ask for.
# 0.8 rad/s is deliberately gentle: this move happens with a human standing at
# the workspace, and nothing about it is time-critical.
SPEED_BUDGET_RAD_S = 0.8
MIN_DURATION_S = 2.0
MAX_DURATION_S = 10.0
GRIPPER_OPEN_TIMEOUT_S = 3.0
# How long OPENING_GRIPPER keeps re-publishing the OPEN command AFTER the
# position feedback first says the gripper is open.  0.0 restores the older
# behaviour exactly (stop the instant it confirms).
#
# WHY A FLOOR AT ALL, when the step already republishes every tick: the confirm
# can be true on the FIRST tick -- the gripper was already open, or it happens
# to be sitting just inside GRIPPER_OPEN_CONFIRM -- and then exactly ONE message
# ever reaches the wire.  One message is the failure mode this window exists to
# rule out (2026-08-06, diag_gripper_halfopen_20260806_173307): a single publish
# can be dropped by DDS discovery, and robotiq_gripper_modbus_node additionally
# rate-limits command_percent, so the sample that says "fully open" is lost
# roughly half the time.  Re-asserting the final setpoint for ~1 s was measured
# 0-of-6 failures against 7-of-14 without it, so 1.0 s is the validated number
# and not a guess.
#
# BOUNDED BY GRIPPER_OPEN_TIMEOUT_S: the deadline always wins, so raising this
# above the timeout cannot make GO HOME sit here longer (see
# _tick_opening_gripper).  Re-asserting OPEN is safe to repeat by construction
# -- it can never fight an object grasp, which is a CLOSING setpoint.
GRIPPER_OPEN_REASSERT_S = 1.0
# Deadline for any single service-backed step (pause / query / switch).
#
# COUPLED TO task_gui_node._SWITCH_TIMEOUT_S (the `timeout` field the node puts
# in the SwitchController request).  REQUIRED INEQUALITY:
#
#     STEP_TIMEOUT_S > task_gui_node._SWITCH_TIMEOUT_S
#
# i.e. this state machine's deadline must fire STRICTLY LATER than the server's
# own, never at the same instant.  When they were equal (both 5.0) a switch that
# legitimately used its full server budget -- realistic whenever the UR External
# Control program is not PLAYING, so controller_manager's RT loop is not
# advancing -- tripped _tick_switching's timeout at the very moment the server
# was answering.  That set _fpc_restore_owed and fired the reverse switch while
# the first was still pending; the two serialise server-side and the restore's
# STRICT deactivate of a still-inactive STJC is REFUSED, so the operator was
# told the restore "ALSO failed" for what was only a slow switch.
# If either constant moves, re-check the inequality.
STEP_TIMEOUT_S = 7.0
# The move's own deadline is the SPEED-SCALED duration plus this, i.e. how long
# past the expected end of the trajectory we wait for the action RESULT before
# giving up.  See _enter_moving() for the scaling.
MOVE_RESULT_MARGIN_S = 5.0
# Speed scale assumed when ops cannot tell us the real one.
#
# scaled_joint_trajectory_controller stretches execution by the pendant's speed
# slider: at 25% a 3.9 s trajectory really takes ~15.7 s.  The two errors are
# NOT symmetric -- a too-short deadline CANCELS a move that is proceeding
# correctly and strands the arm partway between the demo pose and HOME (and
# then STRICT-switches back to FPC there), while a too-long one merely delays
# noticing a genuinely hung action.  So when the scale is unknown we assume the
# operator may be running the slider as low as 25% and wait accordingly.  This
# deadline exists to catch a hang, NOT to enforce timing.
MIN_ASSUMED_SPEED_SCALE = 0.25


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
# wrap_to_pi / circular_dist / wrapped_nearest are VENDORED from
# ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/angle_utils.py (bit-identical
# behaviour).  They are copied rather than imported because gello_recorder has
# no build- or run-time dependency on ur_gello_bringup and we are not adding one
# for three lines of arithmetic; keep them in sync if angle_utils ever changes.


def wrap_to_pi(x: float) -> float:
    """Wrap an angle (or angle difference) into [-pi, pi]."""
    return math.remainder(x, 2.0 * math.pi)


def circular_dist(a: float, b: float) -> float:
    """Smallest absolute angular distance (rad) between a and b, branch-cut aware.

    Two angles physically close but on opposite sides of the +/-pi cut (e.g.
    +3.09 vs -3.09) report ~0.05 rad, not ~6.2 rad.
    """
    return abs(wrap_to_pi(a - b))


def wrapped_nearest(
    target: Sequence[float], reference: Sequence[float]
) -> list[float]:
    """Per-joint, shift ``target`` by a multiple of 2*pi so it is the angular
    equivalent NEAREST to ``reference`` (|result - reference| <= pi).

    LOAD-BEARING, not a nicety.  scaled_joint_trajectory_controller interpolates
    LINEARLY in raw joint space with no 2*pi awareness, and HOME_JOINTS sits
    right on the branch cut: shoulder_pan is +3.1382 (~0.003 rad inside +pi) and
    wrist_3 is -3.1331 (~0.008 rad inside -pi).  If the arm arrives wound the
    other way -- shoulder_pan reading -3.140, physically ~0.005 rad from HOME --
    then commanding the raw literal +3.1382 asks the controller to sweep ~6.28
    rad the LONG way around, at full speed, through whatever is in the workspace.
    Command this function's output; never the raw literal.
    """
    if len(target) != len(reference):
        raise ValueError("wrapped_nearest: target and reference length mismatch")
    return [
        reference[i] + wrap_to_pi(target[i] - reference[i])
        for i in range(len(target))
    ]


def compute_home_duration(
    current: Sequence[float],
    target: Sequence[float],
    speed_budget: float = SPEED_BUDGET_RAD_S,
    min_s: float = MIN_DURATION_S,
    max_s: float = MAX_DURATION_S,
) -> float:
    """Trajectory duration (s) = worst-joint |wrapped gap| / speed_budget, clamped.

    Distance-proportional ON PURPOSE.  A fixed duration (what the HIL
    preposition path uses) means the implied joint velocity is whatever the
    starting distance happens to divide out to -- start far away and the arm is
    commanded to move fast, with no upper bound anywhere in the pipeline.  Here
    the speed budget is the invariant and the clock stretches instead.

    The gap is measured with circular_dist for the same branch-cut reason as
    wrapped_nearest, so a joint reading -3.14 against a +3.1382 target costs
    0.005 rad of budget and not 6.28.

    min_s exists so a tiny correction still gets a smooth, non-jerky ramp;
    max_s caps the pathological case of a very small speed budget.
    """
    if len(current) != len(target):
        raise ValueError("compute_home_duration: current and target length mismatch")
    if not target:
        raise ValueError("compute_home_duration: empty joint vectors")
    if speed_budget <= 0.0:
        raise ValueError("compute_home_duration: speed_budget must be > 0")
    if min_s > max_s:
        raise ValueError("compute_home_duration: min_s must be <= max_s")
    worst = max(circular_dist(target[i], current[i]) for i in range(len(target)))
    return min(max(worst / speed_budget, min_s), max_s)


def reorder_joint_positions(
    names: Sequence[str], positions: Sequence[float]
) -> Optional[list[float]]:
    """Map a sensor_msgs/JointState-style (names, positions) pair into UR_JOINT_ORDER.

    Returns six floats, or None if ANY UR joint is missing (or has no matching
    position entry).  /joint_states publishes in roughly alphabetical order, NOT
    UR order, so mapping by name is load-bearing: an index-based read silently
    permutes the arm's pose, and a permuted pose fed to wrapped_nearest produces
    a perfectly plausible-looking command to the wrong place.

    Returning None rather than a partial vector is deliberate -- the caller must
    treat "I do not know where the arm is" as a refusal to move, not as a zero.
    """
    index = {str(name): i for i, name in enumerate(names)}
    out: list[float] = []
    for joint in UR_JOINT_ORDER:
        i = index.get(joint)
        if i is None or i >= len(positions):
            return None
        out.append(float(positions[i]))
    return out


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #


class HomeMoveState:
    """Public state names (plain strings so the GUI can display them directly)."""

    IDLE = "IDLE"
    PAUSING = "PAUSING"
    SWITCHING_TO_JTC = "SWITCHING_TO_JTC"
    MOVING = "MOVING"
    OPENING_GRIPPER = "OPENING_GRIPPER"
    RESTORING_FPC = "RESTORING_FPC"
    DONE = "DONE"
    FAILED = "FAILED"


#: States in which a GO-HOME is in flight (button disabled, arm may be moving).
ACTIVE_STATES = (
    HomeMoveState.PAUSING,
    HomeMoveState.SWITCHING_TO_JTC,
    HomeMoveState.MOVING,
    HomeMoveState.OPENING_GRIPPER,
    HomeMoveState.RESTORING_FPC,
)

# Human-readable labels for the in-flight step, reused in timeout messages.
_STEP_PAUSE = "bridge pause"
_STEP_QUERY = "controller states"
_STEP_SWITCH = "controller switch to " + STJC
_STEP_MOVE = "home trajectory"
_STEP_GRIPPER = "gripper open"
_STEP_RESTORE = "controller switch back to " + FPC

# The post-DONE operator instructions.  The gripper sentence covers BOTH bridge
# modes because they end differently: in CONTINUOUS mode a resume ramps to
# whatever the trigger currently reads, so a squeezed trigger re-closes the
# gripper GO HOME just opened; in DISCRETE mode the latch is reset by the pause,
# so a resume with the trigger inside the dead band publishes nothing and the
# Robotiq holds the open position.  The advice is the same either way -- it is
# exactly right for continuous and costs nothing in discrete.
_DONE_TAIL = (
    "teleop bridges remain PAUSED -- re-engage with the EEF GUI ENGAGE button. "
    "Hold the GELLO trigger OPEN before resuming the gripper: CONTINUOUS mode "
    "ramps to the live trigger over ~2 s; DISCRETE mode resets its latch on "
    "pause, so a resume in the dead band leaves the gripper open."
)


class HomeMoveController:
    """Drives the GO-HOME sequence. See module docstring for the threading contract.

    ``ops`` is a duck-typed object supplied by the node.  Required surface:

      ops.now() -> float
          Monotonic seconds.  Every deadline in here is measured with it, so a
          test can drive the machine with a fake clock.
      ops.current_joints() -> list[float] | None
          Latest arm joints, already reordered into UR_JOINT_ORDER by the node.
          None means "unknown" and blocks the move.
      ops.pause_bridges(done_cb)
          Fire the arm AND gripper bridge pause Triggers (unconditional and
          idempotent per the bridge contract).  done_cb(ok: bool, msg: str)
          once both resolve.  MUST report ok=False fast if a service is absent:
          that means the teleop stack is not running.
      ops.get_controller_states(done_cb)
          done_cb(ok: bool, fpc_active: bool, stjc_active: bool, msg: str).
      ops.switch_controllers(activate: str, deactivate: str | None, done_cb)
          STRICT switch; done_cb(ok: bool, msg: str).
      ops.send_home_trajectory(positions: list[float], duration_s: float, done_cb)
          done_cb(ok: bool, msg: str) when the action RESULT arrives
          (ok iff STATUS_SUCCEEDED).
      ops.cancel_home_trajectory()
          Best-effort cancel, used when the move deadline expires and when the
          operator hits STOP (see abort()).  Best-effort AND not instantaneous.
      ops.command_gripper_open()
          Publish Float32 GRIPPER_OPEN_VALUE once.  Called on EVERY tick of
          OPENING_GRIPPER, so the adapter must keep a long-lived publisher --
          creating and destroying one per call reintroduces the DDS discovery
          loss this step is defending against.
      ops.gripper_position() -> float | None
          Latest /robotiq_gripper/position_percent.

    OPTIONAL surface (accessed defensively -- an ops object without it works):

      ops.speed_scale() -> float | None
          The UR speed-scaling factor currently in force (1.0 = 100%, i.e. the
          pendant slider at full), or None when it is not known.  In the node
          this comes from the driver's speed-scaling broadcaster.  Read ONCE,
          when entering MOVING, to size the move deadline.  Anything that is not
          a finite number greater than zero -- a raise, None, NaN, 0.0 -- counts
          as "unknown" and falls back to MIN_ASSUMED_SPEED_SCALE.
    """

    def __init__(
        self,
        ops,
        *,
        speed_budget_rad_s: float = SPEED_BUDGET_RAD_S,
        min_duration_s: float = MIN_DURATION_S,
        max_duration_s: float = MAX_DURATION_S,
        gripper_open_timeout_s: float = GRIPPER_OPEN_TIMEOUT_S,
        gripper_open_reassert_s: float = GRIPPER_OPEN_REASSERT_S,
        step_timeout_s: float = STEP_TIMEOUT_S,
    ) -> None:
        self._ops = ops
        self._speed_budget_rad_s = float(speed_budget_rad_s)
        self._min_duration_s = float(min_duration_s)
        self._max_duration_s = float(max_duration_s)
        self._gripper_open_timeout_s = float(gripper_open_timeout_s)
        # Negative is clamped rather than rejected: it means the same thing as
        # 0.0 ("do not hold"), and a GO HOME must never refuse to run over a
        # tuning knob.
        self._gripper_open_reassert_s = max(0.0, float(gripper_open_reassert_s))
        self._step_timeout_s = float(step_timeout_s)

        self._state: str = HomeMoveState.IDLE
        self._message: str = "idle"
        self._duration_s: Optional[float] = None
        # Total wall-clock budget granted to the move (speed-scaled duration +
        # MOVE_RESULT_MARGIN_S). Kept so the timeout message quotes the number
        # the deadline was actually built from.
        self._move_budget_s: Optional[float] = None

        # --- in-flight step bookkeeping ---------------------------------- #
        self._step: str = ""
        # Monotonic id of the in-flight step. Every done_cb closes over the id
        # it was created with, so a late reply to an already-timed-out step is
        # discarded instead of being read as the answer to the current one.
        self._step_id: int = 0
        self._deadline: Optional[float] = None
        # Written by done_cbs (ANY thread) as one atomic attribute store;
        # read and cleared only by tick().
        self._result: Optional[tuple] = None

        # --- failure / restore bookkeeping ------------------------------- #
        # True once STJC owns the joints, i.e. once FPC may be deactivated:
        # from then on a failure must still walk through RESTORING_FPC.
        self._fpc_restore_owed: bool = False
        self._pending_failure: Optional[str] = None
        self._gripper_warning: str = ""
        # OPENING_GRIPPER bookkeeping (tick() only). _gripper_confirmed latches:
        # once the feedback has said "open" the step stops needing it again, it
        # only owes the rest of its settle window.
        self._gripper_confirmed: bool = False
        self._gripper_reassert_until: Optional[float] = None

        # --- operator STOP ------------------------------------------------ #
        # Set by abort() from ANY thread; consumed (and acted on) by tick().
        self._abort_requested: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def request(self) -> bool:
        """Start a GO-HOME. Returns False (and does not start) if refused.

        Refused while a move is already in flight -- in that case state and
        message are left UNTOUCHED, because clobbering the live progress line
        with "already in progress" would hide what the arm is actually doing.
        The caller (a button handler) already knows it asked.

        Refused, with an explanatory message, when the arm's joints are unknown:
        we cannot compute a branch-cut-safe target without knowing the current
        winding, and commanding HOME blind is exactly the ~2*pi hazard
        wrapped_nearest exists to prevent.
        """
        if self._state in ACTIVE_STATES:
            return False
        if self._ops.current_joints() is None:
            self._state = HomeMoveState.FAILED
            self._message = (
                "cannot GO HOME: no robot joint states received yet "
                "(is /joint_states publishing?)"
            )
            return False

        self._duration_s = None
        self._move_budget_s = None
        self._pending_failure = None
        self._gripper_warning = ""
        self._fpc_restore_owed = False
        # A STOP that raced the end of the PREVIOUS run must not abort this
        # fresh one: clear the flag on the way in.
        self._abort_requested = False
        self._state = HomeMoveState.PAUSING
        self._message = "pausing GELLO teleop bridges"
        self._start_pause()
        return True

    def abort(self) -> bool:
        """Operator STOP for an in-flight GO HOME. Returns False if there is none.

        Returns False and changes NOTHING when no move is in flight (the state
        is not in ACTIVE_STATES) -- there is nothing to stop, and a terminal
        DONE/FAILED message must not be overwritten by a stop that arrived too
        late.

        Otherwise returns True and only RECORDS the request.  It deliberately
        performs no transition: this is called from the Qt GUI thread while
        tick() runs on the ROS timer thread, and tick() being the single mutator
        is what lets this class carry no lock (see module docstring).  The stop
        actually happens on the next tick(), i.e. within ~100 ms at the node's
        10 Hz.

        BEST-EFFORT AND NOT INSTANTANEOUS.  All this can do is ask the action
        server to cancel; scaled_joint_trajectory_controller then decelerates on
        its own schedule, so the arm KEEPS MOVING briefly after the button
        click, and it stops wherever that leaves it -- partway between the demo
        pose and HOME.  It is not, and must not be presented as, an E-stop.
        Fail-closed still applies: if FPC may already have been taken away the
        machine routes through RESTORING_FPC before landing in FAILED, so teleop
        is usable again afterwards.  Bridges are NOT resumed (as ever).
        """
        if self._state not in ACTIVE_STATES:
            return False
        self._abort_requested = True
        return True

    def tick(self) -> None:
        """Advance the machine by one step. Call at ~10 Hz; safe to over-call.

        This is the ONLY place a transition ever happens (see module docstring):
        done_cbs and abort() merely record, so there is exactly one thread
        mutating the state machine and no lock is needed inside this class.
        """
        # The operator STOP is consumed FIRST, ahead of every per-state handler:
        # once the human has asked for the arm to stop, nothing else this tick
        # may advance the sequence.
        if self._abort_requested:
            self._abort_requested = False
            # Re-check under tick()'s single-mutator rule: abort() may have set
            # the flag on another thread just as the machine landed in
            # DONE/FAILED, and a terminal result must not be rewritten.
            if self._state in ACTIVE_STATES:
                self._perform_abort()
                return

        state = self._state
        if state == HomeMoveState.PAUSING:
            self._tick_pausing()
        elif state == HomeMoveState.SWITCHING_TO_JTC:
            self._tick_switching()
        elif state == HomeMoveState.MOVING:
            self._tick_moving()
        elif state == HomeMoveState.OPENING_GRIPPER:
            self._tick_opening_gripper()
        elif state == HomeMoveState.RESTORING_FPC:
            self._tick_restoring()
        # IDLE / DONE / FAILED are terminal: nothing to advance.

    def status(self) -> dict:
        """Fresh snapshot dict for the GUI: state / active / message / duration_s.

        A NEW dict every call -- the GUI is free to keep, mutate or stash it
        without reaching back into the machine's internals.
        """
        return {
            "state": self._state,
            "active": self._state in ACTIVE_STATES,
            "message": self._message,
            "duration_s": self._duration_s,
        }

    # ------------------------------------------------------------------ #
    # Step plumbing
    # ------------------------------------------------------------------ #
    def _begin_step(self, label: str, timeout_s: Optional[float]) -> int:
        """Arm a new in-flight step: bump the id, drop any stale result, set deadline."""
        self._step = label
        self._step_id += 1
        self._result = None
        self._deadline = (
            None if timeout_s is None else self._ops.now() + float(timeout_s)
        )
        return self._step_id

    def _record(self, step_id: int, payload: tuple) -> None:
        """done_cb sink -- may run on ANY thread. One guarded store, nothing else.

        The guard is the write half of the step-id drop (module docstring): a
        callback for a step we have already moved on from must not be allowed to
        sit in _result at all, because the NEXT step's genuine reply may have
        landed there first.  The concrete case is the move deadline -- we cancel
        the trajectory and start the FPC restore in the same tick, and the
        cancelled action's late result would otherwise overwrite the restore's
        reply and make a healthy restore report as a timeout.

        Reading self._step_id here is a plain attribute read racing tick()'s
        bump; the worst outcome is a store that _take_result() then rejects,
        i.e. exactly the pre-existing read-side behaviour.  Still one store,
        still no lock.
        """
        if step_id != self._step_id:
            return
        self._result = (step_id, payload)

    def _take_result(self) -> Optional[tuple]:
        """Result of the CURRENT step if it has arrived, else None. tick() only."""
        result = self._result
        if result is None or result[0] != self._step_id:
            return None
        self._result = None
        return result[1]

    def _timed_out(self) -> bool:
        return self._deadline is not None and self._ops.now() >= self._deadline

    def _timeout_message(self) -> str:
        return "timed out after {:.1f}s waiting for {}".format(
            self._step_timeout_s, self._step
        )

    # ------------------------------------------------------------------ #
    # Terminal transitions
    # ------------------------------------------------------------------ #
    def _fail(self, reason: str) -> None:
        """Fail -- via the FPC restore if we may have taken FPC away (fail-closed)."""
        if self._fpc_restore_owed:
            self._pending_failure = reason
            self._start_restore()
            return
        self._state = HomeMoveState.FAILED
        self._message = reason
        self._deadline = None

    def _finish_failed(self, restore_note: str) -> None:
        self._state = HomeMoveState.FAILED
        # Both halves, always: the operator needs to know what went wrong AND
        # whether the arm is usable for teleop again.
        self._message = "{}; {}".format(self._pending_failure, restore_note)
        self._pending_failure = None
        self._deadline = None

    def _perform_abort(self) -> None:
        """Carry out a consumed operator STOP. tick() only, state is ACTIVE."""
        stop = "STOPPED by operator"
        if self._state == HomeMoveState.MOVING:
            # Same best-effort shape as the move-deadline path: try to stop the
            # arm before we go anywhere near a controller switch, and if even
            # the cancel raised, say so -- the operator must not read "STOPPED"
            # as "the arm is stopping" when the request never left.
            try:
                self._ops.cancel_home_trajectory()
            except Exception as exc:  # noqa: BLE001 - never mask the stop
                self._fail(
                    "{}: home trajectory cancel ALSO failed: {} -- the arm may "
                    "still be moving".format(stop, exc)
                )
                return
            stop = (
                "{}: home trajectory cancel requested (the controller "
                "decelerates on its own schedule, so the arm keeps moving "
                "briefly)".format(stop)
            )
        # Through _fail() on purpose: fail-closed still holds, so if FPC may be
        # gone we walk RESTORING_FPC first and the FAILED message ends up
        # carrying both the stop and the restore outcome.  A stop that lands
        # while the restore is ALREADY in flight (_fpc_restore_owed cleared by
        # _start_restore) goes straight to FAILED instead -- restarting a switch
        # that is mid-flight is exactly the collision Fix 1 is about, and by
        # then nothing is moving anyway.
        self._fail(stop)

    def _finish_done(self) -> None:
        self._state = HomeMoveState.DONE
        gripper_note = self._gripper_warning or "gripper open"
        self._message = "HOME reached; {}; {}".format(gripper_note, _DONE_TAIL)
        self._deadline = None

    # ------------------------------------------------------------------ #
    # PAUSING (pause bridges, then query controller_manager)
    # ------------------------------------------------------------------ #
    def _start_pause(self) -> None:
        step_id = self._begin_step(_STEP_PAUSE, self._step_timeout_s)
        self._ops.pause_bridges(
            lambda ok, msg: self._record(step_id, (bool(ok), str(msg)))
        )

    def _start_query(self) -> None:
        step_id = self._begin_step(_STEP_QUERY, self._step_timeout_s)
        self._ops.get_controller_states(
            lambda ok, fpc, stjc, msg: self._record(
                step_id, (bool(ok), bool(fpc), bool(stjc), str(msg))
            )
        )

    def _tick_pausing(self) -> None:
        result = self._take_result()
        if result is None:
            if self._timed_out():
                self._fail(self._timeout_message())
            return

        if self._step == _STEP_PAUSE:
            ok, msg = result
            if not ok:
                # A missing pause service means the teleop stack is not running
                # at all. Refuse loudly rather than driving the arm with an
                # unknown bridge state.
                self._fail("teleop pause failed: {}".format(msg))
                return
            self._message = "bridges paused; reading controller states"
            self._start_query()
            return

        # _STEP_QUERY
        ok, fpc_active, stjc_active, msg = result
        if not ok:
            self._fail("controller_manager query failed: {}".format(msg))
            return
        if stjc_active:
            # Nothing to switch -- somebody (a previous GO HOME, the HIL
            # preposition script) already left the joints with STJC.
            self._enter_moving()
            return
        if fpc_active:
            self._start_switch()
            return
        self._fail("neither controller active -- robot stack not ready")

    # ------------------------------------------------------------------ #
    # SWITCHING_TO_JTC
    # ------------------------------------------------------------------ #
    def _start_switch(self) -> None:
        self._state = HomeMoveState.SWITCHING_TO_JTC
        self._message = "handing joints to {}".format(STJC)
        step_id = self._begin_step(_STEP_SWITCH, self._step_timeout_s)
        self._ops.switch_controllers(
            STJC, FPC, lambda ok, msg: self._record(step_id, (bool(ok), str(msg)))
        )

    def _tick_switching(self) -> None:
        result = self._take_result()
        if result is None:
            if self._timed_out():
                # A timed-out switch is UNRESOLVED: controller_manager may well
                # have performed it and only the reply went missing, so we must
                # assume FPC is gone and go through the restore.
                self._fpc_restore_owed = True
                self._fail(self._timeout_message())
            return
        ok, msg = result
        if not ok:
            # A STRICT switch that answers "failed" is atomic: nothing changed,
            # FPC still owns the joints, so there is nothing to restore.
            self._state = HomeMoveState.FAILED
            self._message = "controller switch to {} failed: {}".format(STJC, msg)
            self._deadline = None
            return
        self._enter_moving()

    # ------------------------------------------------------------------ #
    # MOVING
    # ------------------------------------------------------------------ #
    def _enter_moving(self) -> None:
        self._state = HomeMoveState.MOVING
        # From here on the session owes an FPC restore even on failure: either
        # we just deactivated FPC, or STJC was already active and the operator
        # still needs FPC back to teleop.
        self._fpc_restore_owed = True

        current = self._ops.current_joints()
        if current is None:
            self._fail("robot joint states went stale before the move")
            return
        if len(current) != len(HOME_JOINTS):
            self._fail(
                "robot reported {} joints, expected {}".format(
                    len(current), len(HOME_JOINTS)
                )
            )
            return

        # NEVER send the raw HOME_JOINTS literal -- see wrapped_nearest().
        target = wrapped_nearest(HOME_JOINTS, current)
        duration = compute_home_duration(
            current,
            target,
            self._speed_budget_rad_s,
            self._min_duration_s,
            self._max_duration_s,
        )
        self._duration_s = duration

        # The deadline is the SPEED-SCALED duration plus a margin for the action
        # result to come back -- not a fixed timeout, or a legitimately long
        # move would be cancelled mid-flight.  duration is what the trajectory
        # PLANS; STJC executes it stretched by the pendant's speed slider, so
        # the wall-clock wait is duration / scale.
        #
        # Read ONCE, here: the operator turning the slider mid-move must not
        # retroactively move a deadline the machine is already counting down.
        scale = self._read_speed_scale()
        divisor = min(
            max(MIN_ASSUMED_SPEED_SCALE if scale is None else scale,
                MIN_ASSUMED_SPEED_SCALE),
            1.0,  # a scale above 1.0 must never SHORTEN the deadline
        )
        budget = duration / divisor + MOVE_RESULT_MARGIN_S
        self._move_budget_s = budget
        # Say why we are willing to wait this long, so a slow move does not look
        # like a stuck GUI.
        self._message = (
            "moving to HOME over {:.1f}s planned ({}; waiting up to "
            "{:.1f}s)".format(
                duration,
                "speed scale {:.0f}% measured".format(divisor * 100.0)
                if scale is not None
                else "speed scale unknown, assuming {:.0f}%".format(
                    divisor * 100.0
                ),
                budget,
            )
        )
        step_id = self._begin_step(_STEP_MOVE, budget)
        self._ops.send_home_trajectory(
            target, duration, lambda ok, msg: self._record(step_id, (bool(ok), str(msg)))
        )

    def _read_speed_scale(self) -> Optional[float]:
        """UR speed scaling if ops can report it, else None.

        Defensive on every axis: the hook is OPTIONAL (an ops object predating
        it, or a test double, simply has no attribute), it may raise, and it may
        hand back None or a nonsense value.  All of those mean "unknown", and
        unknown is handled by MIN_ASSUMED_SPEED_SCALE rather than by refusing to
        move -- a missing broadcaster is not a reason to block a GO HOME.
        """
        getter = getattr(self._ops, "speed_scale", None)
        if getter is None:
            return None
        try:
            value = getter()
            if value is None:
                return None
            scale = float(value)
        except Exception:  # noqa: BLE001 - any failure here just means "unknown"
            return None
        if not math.isfinite(scale) or scale <= 0.0:
            return None
        return scale

    def _tick_moving(self) -> None:
        result = self._take_result()
        if result is None:
            if self._timed_out():
                # Best-effort: stop the arm before reporting. The cancel may
                # itself fail (that is why it is best-effort) but leaving a
                # trajectory running while we switch controllers is worse.
                budget = self._move_budget_s or 0.0
                try:
                    self._ops.cancel_home_trajectory()
                except Exception as exc:  # noqa: BLE001 - never mask the timeout
                    self._fail(
                        "home trajectory did not finish within {:.1f}s; cancel "
                        "also failed: {}".format(budget, exc)
                    )
                    return
                self._fail(
                    "home trajectory did not finish within {:.1f}s; cancel "
                    "requested".format(budget)
                )
            return
        ok, msg = result
        if not ok:
            self._fail("home trajectory failed: {}".format(msg))
            return
        self._enter_opening_gripper()

    # ------------------------------------------------------------------ #
    # OPENING_GRIPPER
    # ------------------------------------------------------------------ #
    def _enter_opening_gripper(self) -> None:
        self._state = HomeMoveState.OPENING_GRIPPER
        self._message = "opening gripper"
        self._gripper_confirmed = False
        self._gripper_reassert_until = (
            self._ops.now() + self._gripper_open_reassert_s
        )
        self._begin_step(_STEP_GRIPPER, self._gripper_open_timeout_s)

    def _tick_opening_gripper(self) -> None:
        # Republish EVERY tick. The command is a plain Float32 topic with no
        # ack: a single publish can be lost to a subscriber that has not
        # matched yet, and re-sending an idempotent "open" costs nothing.
        self._ops.command_gripper_open()

        position = self._ops.gripper_position()
        if position is not None and position <= GRIPPER_OPEN_CONFIRM:
            # Latch the confirmation, but do NOT leave yet: leaving on the tick
            # that confirms can mean exactly one message ever went out, which is
            # the losable-single-publish case GRIPPER_OPEN_REASSERT_S exists to
            # rule out. With the window at 0.0 this is the old behaviour, byte
            # for byte -- now() >= now() is true on this very tick.
            self._gripper_confirmed = True
            if (
                self._gripper_reassert_until is None
                or self._ops.now() >= self._gripper_reassert_until
            ):
                self._start_restore()
                return

        if self._timed_out():
            # The deadline outranks the settle window: a re-assert window longer
            # than the timeout must not extend how long GO HOME sits here. A
            # step that already confirmed leaves silently; only an unconfirmed
            # one carries the warning.
            if self._gripper_confirmed:
                self._start_restore()
                return
            # WARN-ONLY, deliberately -- mirrors run_hil_preposition.sh step
            # [5b/6]. The gripper may be absent entirely (arm-only bring-up) or
            # merely slow to report, and neither is a reason to strand the arm
            # on STJC with teleop unusable. Carry the warning into the DONE
            # message so the operator checks it visually.
            observed = "no feedback" if position is None else "{:.2f}".format(position)
            self._gripper_warning = (
                "GRIPPER OPEN NOT CONFIRMED within {:.1f}s ({}) -- check it "
                "visually".format(self._gripper_open_timeout_s, observed)
            )
            self._start_restore()

    # ------------------------------------------------------------------ #
    # RESTORING_FPC (runs on BOTH the success and the failure path)
    # ------------------------------------------------------------------ #
    def _start_restore(self) -> None:
        self._state = HomeMoveState.RESTORING_FPC
        self._message = "restoring {}".format(FPC)
        # Cleared here so that a failure INSIDE the restore lands in FAILED
        # instead of re-entering the restore forever.
        self._fpc_restore_owed = False
        step_id = self._begin_step(_STEP_RESTORE, self._step_timeout_s)
        self._ops.switch_controllers(
            FPC, STJC, lambda ok, msg: self._record(step_id, (bool(ok), str(msg)))
        )

    def _tick_restoring(self) -> None:
        result = self._take_result()
        if result is None:
            if self._timed_out():
                note = (
                    "{} restore timed out after {:.1f}s -- teleop may be unusable "
                    "until a controller is activated by hand".format(
                        FPC, self._step_timeout_s
                    )
                )
                if self._pending_failure is not None:
                    self._finish_failed(note)
                else:
                    self._state = HomeMoveState.FAILED
                    self._message = "HOME reached but {}".format(note)
                    self._deadline = None
            return

        ok, msg = result
        if self._pending_failure is not None:
            self._finish_failed(
                "{} restored".format(FPC)
                if ok
                else "{} restore ALSO failed: {}".format(FPC, msg)
            )
            return
        if not ok:
            self._state = HomeMoveState.FAILED
            self._message = (
                "HOME reached but restoring {} failed: {} -- teleop is unusable "
                "until a controller is activated by hand".format(FPC, msg)
            )
            self._deadline = None
            return
        self._finish_done()
