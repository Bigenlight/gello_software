"""Unit tests for the GO-HOME orchestration state machine.

Everything here runs with NO ROS, NO robot and NO clock: ``FakeOps`` records
the calls the controller makes and hands back the ``done_cb`` so the test fires
it by hand, and ``FakeClock`` makes every deadline exactly reproducible.

Deliberately NARROW.  The EEF teleop stack underneath is already verified in
the field; this suite covers only the things whose failure would make the real
arm do the wrong thing, or leave the operator with a robot they cannot use:
the branch-cut-safe target, the speed-bounding duration, the fail-closed
controller restore, the request() gate, the operator STOP, and the two ways the
move deadline can hurt (cancelling a healthy move, or a late callback making a
healthy restore look dead).  Message wording, step-id bookkeeping and
near-duplicate permutations are intentionally NOT pinned.

Run:
    cd ros2_ur_ws/src/gello_recorder
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q test/test_home_move.py
"""

import os
import re

import pytest

from gello_recorder.home_move import (
    FPC,
    GRIPPER_OPEN_REASSERT_S,
    GRIPPER_OPEN_VALUE,
    HOME_JOINTS,
    MIN_ASSUMED_SPEED_SCALE,
    MOVE_RESULT_MARGIN_S,
    STJC,
    UR_JOINT_ORDER,
    HomeMoveController,
    HomeMoveState,
    compute_home_duration,
    reorder_joint_positions,
)

# A pose that is already at HOME (so duration clamps to the minimum and the
# wrapped target equals the literal) -- the boring baseline for sequencing tests.
AT_HOME = list(HOME_JOINTS)


class FakeClock:
    def __init__(self, t0: float = 1000.0) -> None:
        self.t = float(t0)

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class FakeOps:
    """Duck-typed stand-in for the node's ROS surface. Records; never calls back
    on its own -- the test fires the stored callbacks explicitly.

    Deliberately has NO ``speed_scale``: that hook is OPTIONAL, so the default
    double doubles as the "ops object predating the hook" case.  Tests that want
    a scale attach one to the instance."""

    def __init__(self, clock: FakeClock, joints=None) -> None:
        self.clock = clock
        self.joints = None if joints is None else list(joints)
        self.calls: list[str] = []          # ordered log of every op invoked
        self.pause_cb = None
        self.states_cb = None
        self.switch_cb = None
        self.switch_args: list[tuple] = []  # (activate, deactivate)
        self.traj_cb = None
        self.traj_args: list[tuple] = []    # (positions, duration_s)
        self.cancel_count = 0
        self.gripper_cmds: list[float] = []
        self.grip_pos = None

    # --- clock / state -------------------------------------------------- #
    def now(self) -> float:
        return self.clock.now()

    def current_joints(self):
        return None if self.joints is None else list(self.joints)

    # --- bridges --------------------------------------------------------- #
    def pause_bridges(self, done_cb) -> None:
        self.calls.append("pause")
        self.pause_cb = done_cb

    def resume_bridges(self, done_cb) -> None:
        # Deliberately present and callable: if the controller ever resumed the
        # bridges, duck typing means it would succeed here and the "never
        # resumed" assertions would catch it.
        self.calls.append("resume")

    # --- controller_manager ---------------------------------------------- #
    def get_controller_states(self, done_cb) -> None:
        self.calls.append("states")
        self.states_cb = done_cb

    def switch_controllers(self, activate, deactivate, done_cb) -> None:
        self.calls.append("switch")
        self.switch_args.append((activate, deactivate))
        self.switch_cb = done_cb

    # --- trajectory ------------------------------------------------------- #
    def send_home_trajectory(self, positions, duration_s, done_cb) -> None:
        self.calls.append("traj")
        self.traj_args.append((list(positions), duration_s))
        self.traj_cb = done_cb

    def cancel_home_trajectory(self) -> None:
        self.calls.append("cancel")
        self.cancel_count += 1

    # --- gripper ---------------------------------------------------------- #
    def command_gripper_open(self) -> None:
        self.calls.append("grip")
        self.gripper_cmds.append(GRIPPER_OPEN_VALUE)

    def gripper_position(self):
        return self.grip_pos


def make(joints=AT_HOME, **kwargs):
    clock = FakeClock()
    ops = FakeOps(clock, joints)
    return clock, ops, HomeMoveController(ops, **kwargs)


def drive_to_moving(ctrl, ops):
    """request() -> pause ok -> FPC active -> switch ok -> MOVING."""
    assert ctrl.request() is True
    ops.pause_cb(True, "paused")
    ctrl.tick()
    ops.states_cb(True, True, False, "listed")
    ctrl.tick()
    ops.switch_cb(True, "switched")
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.MOVING


def drive_to_done(ctrl, ops):
    drive_to_moving(ctrl, ops)
    ops.traj_cb(True, "succeeded")
    ctrl.tick()                      # -> OPENING_GRIPPER
    ops.grip_pos = 0.01
    # Past the OPEN re-assert settle window, so this single tick both confirms
    # and leaves (see the settle-window tests for what the window itself does).
    ops.clock.advance(GRIPPER_OPEN_REASSERT_S)
    ctrl.tick()                      # publish + confirm -> RESTORING_FPC
    ops.switch_cb(True, "switched")
    ctrl.tick()                      # -> DONE


# ===================================================================== #
# Drift guards -- the same pose lives in three files and cannot be imported
# ===================================================================== #
def _repo_file(*parts):
    here = os.path.dirname(os.path.abspath(__file__))          # .../test
    pkg = os.path.dirname(here)                                # .../gello_recorder
    ws = os.path.dirname(os.path.dirname(pkg))                 # .../ros2_ur_ws
    root = os.path.dirname(ws)                                 # .../gello_software
    path = os.path.join(root, *parts)
    if not os.path.exists(path):
        pytest.skip("source tree not available at {}".format(path))
    return path


def test_home_joints_match_hil_preposition_script():
    """The three copies of this pose must stay identical (see HOME_JOINTS comment)."""
    text = open(
        _repo_file("ros2_ur_ws", "run_hil_preposition.sh"), encoding="utf-8"
    ).read()
    match = re.search(r'RESET_JOINTS_CSV="([^"]+)"', text)
    assert match, "RESET_JOINTS_CSV not found in run_hil_preposition.sh"
    csv = tuple(float(v) for v in match.group(1).split(","))
    assert csv == HOME_JOINTS


def test_the_node_ticks_fast_enough_for_the_settle_window_to_be_a_stream():
    """The re-assert RATE lives in the node, the WINDOW lives here.

    home_move counts wall-clock seconds; how many OPEN commands actually reach
    the driver inside that window is the window divided by the node's tick
    period.  A slower _HOME_TICK_PERIOD_S silently thins the stream back towards
    the single publish this whole mechanism exists to avoid, and nothing else
    couples the two numbers -- task_gui_node cannot be imported here (it needs
    rclpy), so the constant is read out of the source.
    """
    text = open(
        _repo_file(
            "ros2_ur_ws", "src", "gello_recorder", "gello_recorder",
            "task_gui_node.py",
        ),
        encoding="utf-8",
    ).read()
    match = re.search(r"^_HOME_TICK_PERIOD_S\s*=\s*([0-9.]+)", text, re.MULTILINE)
    assert match, "_HOME_TICK_PERIOD_S not found in task_gui_node.py"
    period = float(match.group(1))
    assert period > 0
    assert GRIPPER_OPEN_REASSERT_S / period >= 5.0


def test_the_hil_preposition_script_streams_its_gripper_open_too():
    """Same defect, same fix, a different process (it is a shell script).

    ``ros2 topic pub --once`` builds a publisher, sends one message and destroys
    it, which is exactly the measured DDS-discovery loss.  The default path must
    stream instead, and whatever it starts must be stopped on the way out --
    an orphaned publisher would keep asserting 0.0 at the gripper for as long as
    it lived.
    """
    text = open(
        _repo_file("ros2_ur_ws", "run_hil_preposition.sh"), encoding="utf-8"
    ).read()

    # Streams by default (and the old one-shot survives only as the 0 opt-out).
    assert re.search(
        r'GRIPPER_OPEN_HOLD_S="\$\{GRIPPER_OPEN_HOLD_S:-([0-9.]+)\}"', text
    ), "GRIPPER_OPEN_HOLD_S default not found"
    default_hold = float(
        re.search(
            r'GRIPPER_OPEN_HOLD_S="\$\{GRIPPER_OPEN_HOLD_S:-([0-9.]+)\}"', text
        ).group(1)
    )
    assert default_hold > 0.0
    assert 'ros2 topic pub -r "$GRIPPER_OPEN_RATE_HZ" "$GRIPPER_CMD_TOPIC"' in text

    # ...and the stream is always stopped, including on the Ctrl-C path.
    cleanup = re.search(r"^cleanup\(\) \{\n(.*?)^\}", text, re.MULTILINE | re.DOTALL)
    assert cleanup, "cleanup() not found in run_hil_preposition.sh"
    assert "gripper_open_publish_stop" in cleanup.group(1)


def test_home_joints_match_cube_in_cup_config():
    text = open(
        _repo_file("serl_ur_infra", "ur_experiments", "cube_in_cup.py"),
        encoding="utf-8",
    ).read()
    match = re.search(r"RESET_JOINTS:[^=]*=\s*np\.array\(\s*\[([^\]]+)\]", text)
    assert match, "RESET_JOINTS not found in cube_in_cup.py"
    values = tuple(float(v) for v in match.group(1).replace("\n", "").split(","))
    assert values == HOME_JOINTS


# ===================================================================== #
# Pure helpers
# ===================================================================== #
@pytest.mark.parametrize(
    ("target", "speed_budget", "expected"),
    [
        ([2.4, 0.0, 0.0, 0.0, 0.0, 0.0], 0.8, 3.0),    # 2.4 rad / 0.8 rad/s
        ([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.2, 10.0),   # 15 s -> clamped at max
        ([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 0.8, 2.0),    # 0.125 s -> clamped at min
    ],
)
def test_duration_is_distance_proportional_and_clamped_at_both_ends(
    target, speed_budget, expected
):
    """This is the ONLY thing bounding the joint speed of the move.

    A fixed duration would make the implied velocity whatever the starting
    distance happens to divide out to; here the speed budget is the invariant
    and the clock stretches instead.  min_s keeps a tiny correction smooth,
    max_s caps a pathologically small budget.
    """
    assert compute_home_duration(
        [0.0] * 6, target, speed_budget, 2.0, 10.0
    ) == pytest.approx(expected)


def test_reorder_maps_by_name_and_refuses_a_partial_pose():
    """/joint_states is roughly alphabetical, NOT UR order.

    An index-based read silently permutes the arm's pose, and a permuted pose
    produces a perfectly plausible-looking command to the wrong place.  A
    missing joint must read as "I do not know where the arm is" (None), never
    as a partial vector the caller might treat as a pose.
    """
    names = [
        "elbow_joint",
        "shoulder_lift_joint",
        "shoulder_pan_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    positions = [3.0, 2.0, 1.0, 4.0, 5.0, 6.0]
    assert reorder_joint_positions(names, positions) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    missing = [n for n in UR_JOINT_ORDER if n != "wrist_3_joint"]
    assert reorder_joint_positions(missing, [0.0] * 5) is None
    assert reorder_joint_positions(list(UR_JOINT_ORDER), [0.0] * 5) is None
    assert reorder_joint_positions([], []) is None


# ===================================================================== #
# Happy path
# ===================================================================== #
def test_happy_path_full_sequence_and_state_trace():
    clock, ops, ctrl = make()
    trace = []

    assert ctrl.request() is True
    trace.append(ctrl.status()["state"])

    ops.pause_cb(True, "paused")
    ctrl.tick()
    trace.append(ctrl.status()["state"])          # still PAUSING (querying)

    ops.states_cb(True, True, False, "fpc active")
    ctrl.tick()
    trace.append(ctrl.status()["state"])          # SWITCHING_TO_JTC

    ops.switch_cb(True, "ok")
    ctrl.tick()
    trace.append(ctrl.status()["state"])          # MOVING

    ops.traj_cb(True, "succeeded")
    ctrl.tick()
    trace.append(ctrl.status()["state"])          # OPENING_GRIPPER

    ops.grip_pos = 0.02
    clock.advance(GRIPPER_OPEN_REASSERT_S)        # past the OPEN settle window
    ctrl.tick()
    trace.append(ctrl.status()["state"])          # RESTORING_FPC

    ops.switch_cb(True, "ok")
    ctrl.tick()
    trace.append(ctrl.status()["state"])          # DONE

    assert trace == [
        HomeMoveState.PAUSING,
        HomeMoveState.PAUSING,
        HomeMoveState.SWITCHING_TO_JTC,
        HomeMoveState.MOVING,
        HomeMoveState.OPENING_GRIPPER,
        HomeMoveState.RESTORING_FPC,
        HomeMoveState.DONE,
    ]
    assert ops.calls == ["pause", "states", "switch", "traj", "grip", "switch"]
    assert ops.switch_args == [(STJC, FPC), (FPC, STJC)]
    assert ops.cancel_count == 0
    assert ctrl.status()["active"] is False
    # NEVER auto-resumed: the follower would chase wherever the human happens to
    # be holding the leader, which after a HOME move is somewhere else entirely.
    assert "resume" not in ops.calls


def test_move_command_takes_the_short_way_when_the_arm_is_wound_the_other_way():
    """The branch-cut hazard, end to end: what actually gets commanded.

    HOME sits ON the +/-pi cut (shoulder_pan +3.1382, wrist_3 -3.1331) and STJC
    interpolates linearly in raw joint space with no 2*pi awareness.  With the
    arm wound the other way the raw literal is ~6.28 rad away numerically but
    ~0.005 rad away physically; commanding it sweeps the arm the LONG way round
    at full speed through whatever is in the workspace.
    """
    current = [-3.1400, -1.5276, 1.7168, -1.7592, -1.5216, 3.1400]
    _, ops, ctrl = make(joints=current)
    drive_to_moving(ctrl, ops)
    positions, duration = ops.traj_args[0]

    # Each joint stays on the branch it is already on.
    assert positions[0] == pytest.approx(-3.1449853, abs=1e-6)
    assert positions[5] == pytest.approx(3.1500853, abs=1e-6)
    assert positions[0] != pytest.approx(HOME_JOINTS[0])   # NOT the raw literal
    # No joint is asked to travel anywhere near 2*pi.
    assert max(abs(p - c) for p, c in zip(positions, current)) < 0.02
    # ...so the whole move is ~0.01 rad and the duration clamps to the minimum.
    assert duration == 2.0
    assert ctrl.status()["duration_s"] == pytest.approx(duration)


@pytest.mark.parametrize(
    ("fpc_active", "stjc_active", "expected_state", "expect_trajectory"),
    [
        # STJC already owns the joints (a previous GO HOME, or the HIL
        # preposition script): nothing to switch, move straight away.
        (False, True, HomeMoveState.MOVING, True),
        # Nothing active at all: the robot stack is not ready. Refuse rather
        # than command an arm no controller is holding.
        (False, False, HomeMoveState.FAILED, False),
    ],
)
def test_controller_state_branching(
    fpc_active, stjc_active, expected_state, expect_trajectory
):
    _, ops, ctrl = make()
    assert ctrl.request() is True
    ops.pause_cb(True, "paused")
    ctrl.tick()
    ops.states_cb(True, fpc_active, stjc_active, "listed")
    ctrl.tick()

    assert ctrl.status()["state"] == expected_state
    assert ops.switch_args == []                      # neither branch switches
    assert bool(ops.traj_args) is expect_trajectory


# ===================================================================== #
# Failure paths
# ===================================================================== #
def test_a_pause_failure_fails_before_anything_is_switched_or_moved():
    """A missing pause service means the teleop stack is not running: refuse
    loudly rather than drive the arm with an unknown bridge state."""
    _, ops, ctrl = make()
    assert ctrl.request() is True
    ops.pause_cb(False, "/gello_ur_bridge/pause unavailable")
    ctrl.tick()

    status = ctrl.status()
    assert status["state"] == HomeMoveState.FAILED
    assert status["active"] is False
    assert ops.switch_args == []
    assert ops.traj_args == []
    assert "resume" not in ops.calls      # the failure path never resumes either


@pytest.mark.parametrize(
    ("trigger", "restore_ok", "original", "restore_note"),
    [
        # FPC was definitely deactivated and then the move itself failed.
        ("move_failed", True, "path tolerance violated", "{} restored".format(FPC)),
        ("move_failed", False, "path tolerance violated", "ALSO failed"),
        # The switch reply never came. That is UNRESOLVED, not refused:
        # controller_manager may well have performed the switch and only the
        # reply went missing, so FPC must be assumed gone and restored anyway.
        ("switch_timeout", True, "timed out", "{} restored".format(FPC)),
    ],
)
def test_a_failure_once_fpc_may_be_gone_fails_closed_through_the_restore(
    trigger, restore_ok, original, restore_note
):
    """A bare failure here would leave the arm with no controller the teleop
    stack can use -- the operator would have to notice and repair that by hand.
    So the machine still walks through RESTORING_FPC, and the FAILED message
    reports BOTH the original failure and the restore outcome."""
    clock, ops, ctrl = make(step_timeout_s=5.0)

    if trigger == "move_failed":
        drive_to_moving(ctrl, ops)
        ops.traj_cb(False, "ABORTED: path tolerance violated")
        ctrl.tick()
    else:
        assert ctrl.request() is True
        ops.pause_cb(True, "paused")
        ctrl.tick()
        ops.states_cb(True, True, False, "listed")
        ctrl.tick()
        assert ctrl.status()["state"] == HomeMoveState.SWITCHING_TO_JTC
        clock.advance(5.1)                       # ...and no reply ever arrives
        ctrl.tick()

    assert ctrl.status()["state"] == HomeMoveState.RESTORING_FPC
    assert ops.switch_args[-1] == (FPC, STJC)

    ops.switch_cb(restore_ok, "no controller to activate")
    ctrl.tick()
    status = ctrl.status()
    assert status["state"] == HomeMoveState.FAILED
    assert original in status["message"]                    # original failure
    assert restore_note in status["message"]                # restore outcome


def test_the_move_deadline_cancels_the_trajectory_and_still_restores_fpc():
    clock, ops, ctrl = make()
    drive_to_moving(ctrl, ops)
    duration = ops.traj_args[0][1]
    # FakeOps reports no speed scale, so the deadline assumes the slowest slider
    # setting we are willing to plan for (see the deadline test below).
    budget = duration / MIN_ASSUMED_SPEED_SCALE + MOVE_RESULT_MARGIN_S

    # A legitimately long move must not be cancelled mid-flight.
    clock.advance(budget - 0.1)
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.MOVING
    assert ops.cancel_count == 0

    clock.advance(0.2)
    ctrl.tick()
    assert ops.cancel_count == 1                 # stop the arm first...
    assert ctrl.status()["state"] == HomeMoveState.RESTORING_FPC
    assert ops.switch_args[-1] == (FPC, STJC)    # ...then hand FPC back

    ops.switch_cb(True, "ok")
    ctrl.tick()
    status = ctrl.status()
    assert status["state"] == HomeMoveState.FAILED
    assert "cancel requested" in status["message"]
    assert "{} restored".format(FPC) in status["message"]
    assert "resume" not in ops.calls


@pytest.mark.parametrize(
    ("reported", "expected_divisor"),
    [
        (0.5, 0.5),                            # measured: wait twice as long
        (2.0, 1.0),                            # >1 must never SHORTEN it
        (None, MIN_ASSUMED_SPEED_SCALE),       # hook present, value unknown
        ("absent", MIN_ASSUMED_SPEED_SCALE),   # ops predating the hook entirely
    ],
)
def test_the_move_deadline_is_speed_scaling_aware(reported, expected_divisor):
    """STJC stretches execution by the pendant's speed slider.

    At 25% a 3.9 s trajectory really takes ~15.7 s, so a deadline built from the
    PLANNED duration alone cancels a move that is proceeding correctly and
    strands the arm partway to HOME.  When the scale is unknown we assume the
    slider may be as low as MIN_ASSUMED_SPEED_SCALE: over-waiting only delays
    noticing a hang, under-waiting aborts a healthy move.
    """
    clock, ops, ctrl = make()
    if reported != "absent":
        ops.speed_scale = lambda: reported
    drive_to_moving(ctrl, ops)

    duration = ops.traj_args[0][1]
    budget = duration / expected_divisor + MOVE_RESULT_MARGIN_S
    # The operator can see how long it intends to wait, and on what basis.
    message = ctrl.status()["message"]
    assert "{:.1f}s".format(budget) in message
    assert ("measured" if reported not in (None, "absent") else "assuming") in message

    clock.advance(budget - 0.1)
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.MOVING     # still waiting
    assert ops.cancel_count == 0

    clock.advance(0.2)
    ctrl.tick()
    assert ops.cancel_count == 1                              # ...and only now


def test_a_late_action_result_cannot_clobber_the_restores_reply():
    """The move deadline cancels the action and starts the restore in one tick.

    The cancelled action's late result and the restore's service reply travel
    different transports with no ordering guarantee, so the stale result can
    land LAST.  If it were stored it would evict the restore's genuine reply and
    the restore would report as a timeout -- telling the operator teleop is
    unusable, and inviting manual controller surgery, when FPC is in fact fine.
    """
    clock, ops, ctrl = make()
    drive_to_moving(ctrl, ops)
    stale_traj_cb = ops.traj_cb

    clock.advance(ops.traj_args[0][1] / MIN_ASSUMED_SPEED_SCALE + MOVE_RESULT_MARGIN_S)
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.RESTORING_FPC

    ops.switch_cb(True, "ok")                 # the restore genuinely succeeded
    stale_traj_cb(True, "succeeded")          # ...and then the stale one lands
    ctrl.tick()

    status = ctrl.status()
    assert status["state"] == HomeMoveState.FAILED
    assert "{} restored".format(FPC) in status["message"]     # not "timed out"


def test_operator_stop_cancels_the_move_and_still_hands_fpc_back():
    """The only software stop for an in-flight GO HOME (otherwise: pendant E-stop).

    abort() must NOT transition -- it runs on the Qt thread while tick() runs on
    the ROS timer thread, and tick() being the single mutator is what lets this
    class carry no lock.  Fail-closed still applies: FPC comes back before the
    machine lands in FAILED, and the bridges are never resumed.
    """
    _, ops, ctrl = make()
    drive_to_moving(ctrl, ops)

    assert ctrl.abort() is True
    assert ops.cancel_count == 0                  # recorded only, not acted on
    assert ctrl.status()["state"] == HomeMoveState.MOVING

    ctrl.tick()
    assert ops.cancel_count == 1                  # stop the arm first...
    assert ctrl.status()["state"] == HomeMoveState.RESTORING_FPC
    assert ops.switch_args[-1] == (FPC, STJC)     # ...then hand FPC back

    ops.switch_cb(True, "ok")
    ctrl.tick()
    status = ctrl.status()
    assert status["state"] == HomeMoveState.FAILED
    assert "STOPPED by operator" in status["message"]
    assert "{} restored".format(FPC) in status["message"]
    assert "resume" not in ops.calls


def test_stop_is_refused_when_no_move_is_in_flight():
    """Nothing to stop, and a terminal message must not be overwritten by a
    stop that arrived too late."""
    _, ops, ctrl = make()
    assert ctrl.abort() is False                  # IDLE
    assert ctrl.status()["state"] == HomeMoveState.IDLE
    assert ops.calls == []

    drive_to_done(ctrl, ops)
    done = ctrl.status()
    assert done["state"] == HomeMoveState.DONE
    assert ctrl.abort() is False
    ctrl.tick()
    assert ctrl.status() == done                  # message untouched

    # A stop that raced the END of that run -- abort() read ACTIVE on the Qt
    # thread while tick() was landing in DONE -- must not abort the NEXT one.
    # The race is unreachable single-threaded, so plant the flag directly.
    ctrl._abort_requested = True
    assert ctrl.request() is True
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.PAUSING


def drive_to_opening_gripper(ctrl, ops):
    """...through a successful move, so the machine sits in OPENING_GRIPPER."""
    drive_to_moving(ctrl, ops)
    ops.traj_cb(True, "succeeded")
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.OPENING_GRIPPER


def test_the_open_command_is_reasserted_across_the_whole_settle_window():
    """ONE publish is not a command -- it is a lottery ticket.

    Measured 2026-08-06: a single publish is lost to DDS discovery (3 of 7
    sends), and robotiq_gripper_modbus_node's rate limiter drops the FRESHEST
    sample inside its window, so the message that says "fully open" disappears
    roughly half the time; the gripper then parks short (0.043 / 0.055 / 0.11 /
    0.42 instead of 0.0118).  Re-asserting for ~1 s at ~10 Hz was 0-of-6
    failures on hardware.  So confirming on the FIRST tick must NOT end the
    step: that is precisely the case where exactly one message went out.
    """
    clock, ops, ctrl = make(gripper_open_timeout_s=3.0, gripper_open_reassert_s=1.0)
    drive_to_opening_gripper(ctrl, ops)
    ops.grip_pos = 0.01                       # confirms immediately

    # The node's 10 Hz tick, driven by hand for the length of the window.
    for _ in range(10):
        ctrl.tick()
        assert ctrl.status()["state"] == HomeMoveState.OPENING_GRIPPER
        clock.advance(0.1)

    assert ops.gripper_cmds == [GRIPPER_OPEN_VALUE] * 10   # ~10 Hz for ~1 s
    ctrl.tick()                                            # window is over
    assert ctrl.status()["state"] == HomeMoveState.RESTORING_FPC

    ops.switch_cb(True, "ok")
    ctrl.tick()
    status = ctrl.status()
    assert status["state"] == HomeMoveState.DONE
    assert "NOT CONFIRMED" not in status["message"]        # it did confirm


def test_a_zero_settle_window_restores_the_leave_on_confirm_behaviour():
    """The knob's 0 must mean EXACTLY what the code did before it existed."""
    _, ops, ctrl = make(gripper_open_reassert_s=0.0)
    drive_to_opening_gripper(ctrl, ops)
    ops.grip_pos = 0.01

    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.RESTORING_FPC
    assert ops.gripper_cmds == [GRIPPER_OPEN_VALUE]        # one publish, as before


def test_the_settle_window_never_outlives_the_confirm_timeout():
    """The deadline outranks the window, so this step's cost stays bounded.

    A re-assert window mis-set longer than the timeout must not hold the arm on
    STJC for longer than GO HOME has always been willing to wait.  And a step
    that DID confirm leaves quietly -- the warning is only for a gripper nobody
    could confirm.
    """
    clock, ops, ctrl = make(gripper_open_timeout_s=3.0, gripper_open_reassert_s=30.0)
    drive_to_opening_gripper(ctrl, ops)
    ops.grip_pos = 0.01

    clock.advance(3.1)
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.RESTORING_FPC

    ops.switch_cb(True, "ok")
    ctrl.tick()
    status = ctrl.status()
    assert status["state"] == HomeMoveState.DONE
    assert "NOT CONFIRMED" not in status["message"]


def test_a_gripper_that_never_confirms_is_a_warning_not_a_failure():
    """Mirrors run_hil_preposition.sh [5b/6]: the gripper may be absent entirely
    (arm-only bring-up) or merely slow to report, and neither is a reason to
    strand the arm on STJC with teleop unusable."""
    clock, ops, ctrl = make(gripper_open_timeout_s=3.0)
    drive_to_moving(ctrl, ops)
    ops.traj_cb(True, "succeeded")
    ctrl.tick()
    ops.grip_pos = None                          # gripper node absent entirely

    clock.advance(3.1)
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.RESTORING_FPC
    assert ops.gripper_cmds == [GRIPPER_OPEN_VALUE]

    ops.switch_cb(True, "ok")
    ctrl.tick()
    status = ctrl.status()
    assert status["state"] == HomeMoveState.DONE          # NOT FAILED
    assert "NOT CONFIRMED" in status["message"]           # ...but carried across


# ===================================================================== #
# request() gating
# ===================================================================== #
def test_request_gating_over_a_whole_lifecycle():
    """Refused with no pose, refused mid-flight, allowed again after DONE and
    after FAILED."""
    _, ops, ctrl = make(joints=None)

    # No pose: we cannot compute a branch-cut-safe target, and commanding HOME
    # blind is exactly the ~2*pi hazard. Refuse, and say why.
    assert ctrl.request() is False
    assert ctrl.status()["state"] == HomeMoveState.FAILED
    assert "joint states" in ctrl.status()["message"]
    assert ops.calls == []                       # nothing was fired

    # Mid-flight: refused WITHOUT clobbering the live progress line.
    ops.joints = list(AT_HOME)
    drive_to_moving(ctrl, ops)
    before = ctrl.status()
    calls_before = list(ops.calls)
    assert ctrl.request() is False
    assert ctrl.status() == before               # state AND message untouched
    assert ops.calls == calls_before             # nothing re-fired

    # After DONE: allowed, with the per-run fields reset.
    ops.traj_cb(True, "succeeded")
    ctrl.tick()
    ops.grip_pos = 0.0
    ops.clock.advance(GRIPPER_OPEN_REASSERT_S)   # past the OPEN settle window
    ctrl.tick()
    ops.switch_cb(True, "ok")
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.DONE
    assert ctrl.request() is True
    assert ctrl.status()["state"] == HomeMoveState.PAUSING
    assert ctrl.status()["duration_s"] is None

    # After FAILED: allowed too -- a failed attempt must not wedge the button.
    ops.pause_cb(False, "unavailable")
    ctrl.tick()
    assert ctrl.status()["state"] == HomeMoveState.FAILED
    assert ctrl.request() is True
