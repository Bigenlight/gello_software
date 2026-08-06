#!/usr/bin/env python3
"""PyQt5 TASK recorder GUI: the teleop recorder, plus a one-button GO HOME.

This is a sibling of ``gello_recorder_gui.py``, not a replacement for it. It
records exactly the same signals into exactly the same take layout (all vector
logs + both RealSense MP4s) and launches the same two camera subprocesses. The
main addition is a GO-HOME bar:

    GO HOME  ->  force-pause both teleop bridges
             ->  hand the joints to scaled_joint_trajectory_controller
             ->  drive to the fixed HOME pose (fast, but speed-budgeted)
             ->  open the Robotiq 2F-85
             ->  hand the joints back to forward_position_controller

which exists so an operator recording repeated task demonstrations can reset the
arm between takes without dropping to a terminal.

Next to it sits STOP HOME: one click, no confirm, live only while a sequence is
in flight. It cancels the trajectory and routes the machine through the same
fail-closed FPC restore before landing in FAILED. It is NOT an E-stop -- the
controller decelerates on its own schedule -- and the label says so.

Closing the window (or Ctrl-C) mid-move is REFUSED once and turned into that
same stop, because neither one does anything to a trajectory the controller has
already accepted: the arm would finish its sweep with the joints stranded on
STJC and teleop silently dead. Both routes funnel through
:meth:`TaskRecorderWindow.closeEvent`, and the refusal is bounded so a wedged
sequencer can never trap the operator -- see ``_HOME_CLOSE_GRACE_S`` and
:meth:`TaskRecorderWindow._veto_close_during_home_move`.

The inherited Teleop bar comes along with two changes. "Pause Teleop" stays
fully functional, but "Resume Teleop" is permanently disabled here and
relabelled, because it calls the JOINT-mode ``resume_chase`` and this is an EEF
session -- see :meth:`TaskRecorderWindow._neutralize_teleop_resume`. And the bar
gains a compact DISCRETE-LATCH READOUT fed by
``/gello_gripper_bridge/discrete_state``: this window is the one actually open
during a take, and whether ``grip_cmd`` is being recorded as a continuous value
or as a binary endpoint is the single fact about the gripper that changes what
the take MEANS -- see :meth:`TaskRecorderWindow._build_teleop_bar`.

Teleop is deliberately LEFT PAUSED when it finishes -- resuming would have the
follower chase wherever the human happens to be holding the leader, which after
a HOME move is by definition somewhere else. The operator re-engages from the
separate EEF GUI's ENGAGE button, which re-anchors at the current pose (so
homing cannot break it) -- and should hold the GELLO trigger OPEN while doing
so. In CONTINUOUS mode that matters because the gripper bridge ramps toward the
live trigger value over ~2 s on resume, so a squeezed trigger closes the
freshly-opened gripper; in DISCRETE mode the latch resets to UNKNOWN on pause,
so a resume with the trigger inside the dead band publishes nothing at all and
the gripper simply stays where GO HOME left it. Both facts are surfaced in the
status label, not just in this docstring.

Everything else -- the window, the preview, the state panel, the camera
subprocess lifecycle, the spin thread, the SIGINT pump -- is inherited from
``gello_recorder_gui.MainWindow`` / reused from its module-level helpers.

console_script entry point: ``task_recorder_gui = gello_recorder.task_recorder_gui:main``
"""

import os
import signal
import sys
import threading
import time

import rclpy

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

# In-package import of the base window AND of its private helpers. This is the
# established precedent in this package (policy_run_gui.py already imports
# MainWindow and _spin_node from here); duplicating the camera-serial resolution
# or the process-group kill would be strictly worse, since a divergence between
# two copies is exactly the failure mode _resolve_camera_serials() exists to
# catch in the first place.
from gello_recorder.gello_recorder_gui import (
    DEFAULT_CAM1_NAME,
    DEFAULT_CAM1_SERIAL,
    DEFAULT_CAM2_NAME,
    DEFAULT_CAM2_SERIAL,
    DEFAULT_COLOR_PROFILE,
    MainWindow,
    _BIG_BUTTON_STYLE,
    _launch_realsense,
    _resolve_camera_serials,
    _spin_node,
)

# home_move is pure Python (no rclpy, no Qt), so importing it at module scope
# costs nothing and keeps the state names the GUI paints identical to the ones
# the sequencer publishes. FPC/STJC come from the same place so the recovery
# command this GUI prints can never name a controller the sequencer does not
# actually switch.
from gello_recorder.home_move import ACTIVE_STATES, FPC, STJC, HomeMoveState

# Resting / armed labels for the two-click confirm, exactly like the base
# window's Resume button: the resting label says "click twice" up front so a
# first-time operator is never surprised that one click does nothing.
_HOME_BUTTON_TEXT = "GO HOME (click twice)"
_HOME_BUTTON_ARMED_TEXT = "Click AGAIN to GO HOME (robot moves FAST!)"
# Milliseconds the armed state survives before it disarms itself. Same 3 s the
# base Resume button uses.
_HOME_CONFIRM_MS = 3000

# STOP HOME is a SINGLE-click control on purpose: it is the exact inverse of GO
# HOME's two-click arm. Two clicks exist to stop an accidental START; making a
# STOP cost two clicks would only add latency to the one action an operator
# reaches for when something is going wrong. The label carries the one fact
# that is easy to get wrong -- it is not an E-stop.
_STOP_HOME_BUTTON_TEXT = "STOP HOME (arm coasts to a stop)"
_STOP_HOME_TOOLTIP = (
    "Single click, no confirm: cancels the running GO HOME.\n\n"
    "NOT an E-stop and NOT instantaneous. All this does is ask the trajectory "
    "action server to cancel; scaled_joint_trajectory_controller then "
    "decelerates on its own schedule, so the arm keeps moving briefly after "
    "the click and stops partway between here and HOME.\n\n"
    "The sequence still routes through the fail-closed "
    + FPC
    + " restore before it lands in FAILED, so teleop is usable again "
    "afterwards. Bridges are NOT resumed -- as with a normal GO HOME, "
    "re-engage from the EEF GUI.\n\n"
    "For a real emergency stop use the pendant."
)

# Shown permanently under the bar: the end state of GO HOME is not obvious, and
# in CONTINUOUS mode getting it wrong (grabbing the leader and pressing ENGAGE
# with the trigger squeezed) makes the gripper snap shut on resume. In DISCRETE
# mode it cannot: the latch resets on pause, so a resume inside the dead band
# publishes nothing. The advice is unchanged either way -- it is exactly right
# for continuous mode and harmless in discrete.
_HOME_HINT_TEXT = (
    "GO HOME force-pauses GELLO teleop, drives the arm to HOME and opens the "
    "gripper. Teleop stays PAUSED afterwards -- re-engage from the EEF GUI, and "
    "hold the GELLO trigger OPEN before pressing Gripper Resume (continuous "
    "mode ramps to the live trigger over ~2 s; discrete mode resets its latch "
    "on pause, so a resume in the dead band leaves the gripper open). STOP HOME "
    "cancels a running sequence with one click, but the arm coasts to a stop -- "
    "it is not an E-stop."
)

# --------------------------------------------------------------------------- #
# The discrete gripper-latch readout in the inherited Teleop bar
# --------------------------------------------------------------------------- #
# A READOUT, not a control: it says which of the two meanings the take's
# `grip_cmd` column currently has (continuous 0..1, or a latched 0.0/1.0
# endpoint), which is the one gripper fact that changes what a recording means.
#
# NOTHING HERE IS RED. None of these states is a fault: DISABLED is the ordinary
# continuous mode nearly every session runs in, and UNKNOWN merely means no
# threshold has been crossed since the bridge started (or since a pause reset
# the latch). Painting either as an alarm would train the operator to ignore the
# bar. RAMPING is orange to match the base window's own RAMPING colour, so the
# two labels next to each other read as one convention.
_GRIP_LATCH_STALE_S = 2.0
_GRIP_LATCH_STYLE = {
    "DISABLED": ("latch: continuous", "#888888"),
    "UNKNOWN": ("latch: UNKNOWN", "#b8860b"),
    "OPEN": ("latch: OPEN", "#22aa22"),
    "CLOSED": ("latch: CLOSED", "#1565c0"),
    "RAMPING": ("latch: RAMPING", "#dd8800"),
}
# Never received / stale / a token this GUI does not know. Greyed rather than
# rendered raw, exactly like the EEF GUI's copy of this indicator: a vocabulary
# change on the bridge side should show up as a dead lamp, not a confident lie.
_GRIP_LATCH_NOSIGNAL = ("latch: --", "#888888")
_GRIP_LATCH_TOOLTIP = (
    "Discrete gripper latch (/gello_gripper_bridge/discrete_state).\n\n"
    "'continuous' is the normal mode: grip_cmd is recorded as the trigger's "
    "0..1 value. OPEN / CLOSED / UNKNOWN mean the bridge is folding the trigger "
    "to its endpoints, so grip_cmd in this take is BINARY (0.0 / 1.0).\n\n"
    "Which one was in force is written into every take as gripper_mode.json.\n\n"
    "Read-only: this window never commands the gripper bridge."
)

# --------------------------------------------------------------------------- #
# Closing the window while the arm is being driven
# --------------------------------------------------------------------------- #
# Seconds a refused close holds the operator off before a further close attempt
# is honoured regardless. Bounded on purpose: a wedged sequencer must never be
# able to trap an operator inside a GUI. Measured from the FIRST refused close,
# and reset by _refresh_home_controls once the sequence is no longer active, so
# a later GO HOME always gets a full window of its own.
_HOME_CLOSE_GRACE_S = 15.0

# What a refused close says in the window (status bar -- never a modal dialog:
# a modal here would block the very Qt thread the operator needs to press STOP
# HOME, and this codebase has none anywhere).
_HOME_CLOSE_REFUSED_TEXT = (
    "Window did NOT close -- a GO HOME is still running. STOP HOME was "
    "requested for you: the trajectory is being cancelled and "
    + FPC
    + " restored. The arm coasts briefly. Close again once the Go home line "
    "reads DONE or FAILED (a further close is honoured anyway after "
    "{:.0f} s).".format(_HOME_CLOSE_GRACE_S)
)

# One line to the terminal as well, because the Ctrl-C route lands here too and
# that operator is looking at the shell, not at the window.
_HOME_CLOSE_REFUSED_STDERR = (
    "### task recorder: close/quit refused -- GO HOME is still active. "
    "STOP HOME requested; the arm is being stopped and " + FPC + " restored."
)

# Printed when the bounded escape hatch fires, i.e. we are letting go of a
# sequence that never finished. Loud, and it ends with the two ways out. The
# controller names are substituted rather than spelled out so this text cannot
# drift from what home_move actually switches; lines are pre-wrapped around
# them.
_HOME_STRANDED_WARNING = (
    "\n"
    "##################################################################\n"
    "### WARNING: closed the task recorder while GO HOME was ACTIVE.\n"
    "###\n"
    "### THE ARM MAY STILL BE MOVING. Closing this window does not\n"
    "### cancel a trajectory the controller has already accepted, and\n"
    "### the GUI can no longer restore the controller for you.\n"
    "###\n"
    "### THE CONTROLLER MAY BE LEFT ON\n"
    "###   {stjc}\n"
    "### with this one INACTIVE:\n"
    "###   {fpc}\n"
    "### If it is, EEF teleop will look fine and do NOTHING -- the\n"
    "### bridge publishes to /{fpc}/commands\n"
    "### and no controller is listening.\n"
    "###\n"
    "### EASIEST FIX -- relaunch this GUI and press GO HOME once. The\n"
    "### sequence sees STJC already active, skips the inbound switch,\n"
    "### and restores FPC at the end: it self-heals.\n"
    "###\n"
    "### MANUAL FIX -- the single line below, unprefixed so it can be\n"
    "### copied straight into a terminal:\n"
    "##################################################################\n"
    "ros2 control switch_controllers --activate {fpc} --deactivate {stjc}\n"
).format(fpc=FPC, stjc=STJC)

# The inherited "Resume Teleop" button is neutralized in this window -- see
# TaskRecorderWindow._neutralize_teleop_resume for the full reasoning. The label
# has to name the replacement, because a disabled button with no explanation is
# indistinguishable from a broken one.
_TELEOP_RESUME_REDIRECT_TEXT = "Resume: use EEF GUI ENGAGE"
_TELEOP_RESUME_TOOLTIP = (
    "Disabled on purpose in the task recorder.\n\n"
    "This button calls /gello_ur_bridge/resume_chase, the JOINT-mode chase "
    "resume: the bridge enters JOINT_BOOTSTRAP (joint passthrough) and its gate "
    "admits a leader-vs-arm gap of up to 1.5 rad PER JOINT. After a GO HOME the "
    "arm is at HOME while the GELLO leader is wherever you left it, so pressing "
    "it would glide the arm toward the leader's absolute joint configuration -- "
    "not EEF delta motion.\n\n"
    "Re-arm EEF teleop with ENGAGE in the EEF GUI instead: it has no alignment "
    "gate and re-anchors at the arm's current pose."
)

_HOME_COLOR_IDLE = "#888888"     # gray
_HOME_COLOR_ACTIVE = "#dd8800"   # orange
_HOME_COLOR_DONE = "#22aa22"     # green
_HOME_COLOR_FAILED = "#cc3333"   # red


# These two are module-level FUNCTIONS, not methods, on purpose. The window's
# refresh/click logic has to keep working when it is invoked unbound against a
# duck-typed stand-in (the headless-GUI test style this repo uses, e.g.
# ur_gello_bringup/test/test_hil_actor_status.py): on such a stand-in every
# ``self._helper(...)`` resolves to the stand-in's own no-op placeholder rather
# than to our code, which would silently paint the label with None.

def _home_status_text(state, message):
    """``STATE`` or ``STATE: message``.

    The message carries all the detail, including the post-DONE operator
    instructions HomeMoveController appends -- this never summarizes or
    truncates it, because that text is how the operator learns that teleop is
    still paused and what a gripper resume will (or will not) do in each mode.
    """
    return "{}: {}".format(state, message) if message else str(state)


def _grip_latch_display(state, age_s):
    """``(text, colour)`` for the discrete-latch readout.

    A module-level FUNCTION for the same reason the two above are (see the
    comment there), and pure so the mapping can be tested without Qt.

    Staleness is treated exactly as the base window treats its own teleop state
    labels: nothing for _GRIP_LATCH_STALE_S (2 s, i.e. 10 missed publishes at
    the bridge's 5 Hz) means unknown, never a stale word left standing. An
    unrecognised token lands in the same place. Neither is a fault colour.
    """
    fresh = state is not None and age_s is not None and age_s < _GRIP_LATCH_STALE_S
    if not fresh:
        return _GRIP_LATCH_NOSIGNAL
    return _GRIP_LATCH_STYLE.get(str(state).strip().upper(), _GRIP_LATCH_NOSIGNAL)


def _home_status_color(state, active):
    """Colour for the status label: red FAILED, green DONE, orange in flight."""
    if state == HomeMoveState.FAILED:
        return _HOME_COLOR_FAILED
    if state == HomeMoveState.DONE:
        return _HOME_COLOR_DONE
    # `active` comes straight from the sequencer; the ACTIVE_STATES fallback
    # keeps the paint honest if a caller hands us a status dict without it.
    if active or state in ACTIVE_STATES:
        return _HOME_COLOR_ACTIVE
    return _HOME_COLOR_IDLE


class TaskRecorderWindow(MainWindow):
    """Recorder window (inherited whole) + a GO-HOME bar above the record bar."""

    def __init__(self, node, cam1_proc, cam2_proc, cam_warnings=None):
        super().__init__(node, cam1_proc, cam2_proc, cam_warnings=cam_warnings)
        self.setWindowTitle("GELLO -> UR7e Task Recorder")

    # ------------------------------------------------------------------ UI --
    def _build_control_bar(self):
        """Return the GO-HOME bar stacked ON TOP of the base record bar.

        Called from the base ``_build_ui()``, which is itself called from the
        base ``__init__`` -- so this is also where GO-HOME window state has to be
        initialized (attributes cannot be set before QMainWindow's own __init__,
        the same constraint PolicyRunWindow documents).

        The base bar is COMPOSED, never replaced: ``super()._build_control_bar()``
        still builds Start/Stop/take/elapsed/status exactly as it does in the
        plain recorder, so take behaviour is bit-identical between the two GUIs.
        """
        self._home_armed = False
        # Last state we announced in the status bar, so a terminal state is
        # announced once on the transition instead of five times a second.
        self._home_last_state = None
        # Monotonic deadline after which a close is honoured even though the
        # sequence is still active; None means "no close has been refused yet".
        # See _veto_close_during_home_move.
        self._home_close_deadline = None
        # One-shot latch for the static half of the neutralized Resume button
        # (see _neutralize_teleop_resume): its text/style/tooltip never change,
        # so re-applying them at 5 Hz forever is pure churn.
        self._teleop_resume_neutralized = False

        # A permanent hint under the bar: the end state of GO HOME is not
        # obvious, and in continuous mode getting it wrong (grabbing the leader
        # and pressing ENGAGE with the trigger squeezed) makes the gripper snap
        # shut on resume.
        hint = QLabel(_HOME_HINT_TEXT)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #777777; font-size: 9pt;")

        outer = QVBoxLayout()
        outer.addLayout(self._build_home_bar())
        outer.addWidget(hint)
        outer.addLayout(super()._build_control_bar())
        return outer

    def _build_teleop_bar(self):
        """Base teleop bar, COMPOSED with a compact discrete-latch readout.

        WHY IT BELONGS HERE. The indicator already exists in the EEF GUI, but
        this is the window that is actually open while a take is being recorded,
        and the latch decides what the take's ``grip_cmd`` column MEANS
        (continuous 0..1 vs a binary endpoint). An operator should not have to
        look at another window to know which kind of data they are producing.

        Deliberately a LABEL and nothing else: no click target, no service call.
        This window never commands the gripper bridge.

        The base bar is composed rather than rebuilt, and the insertion point is
        found by ``indexOf`` rather than by a hard-coded index, so re-ordering
        the base bar cannot silently drop this label somewhere absurd (past the
        stretch, on top of the buttons). Called from the base ``_build_ui()``
        during ``__init__``, so this is also where its state is initialized --
        the same constraint ``_build_control_bar`` documents.
        """
        bar = super()._build_teleop_bar()
        text, color = _GRIP_LATCH_NOSIGNAL
        self._grip_latch_label = QLabel(text)
        self._grip_latch_label.setMinimumWidth(150)
        self._grip_latch_label.setToolTip(_GRIP_LATCH_TOOLTIP)
        self._grip_latch_label.setStyleSheet(
            "color: {}; font-weight: bold;".format(color)
        )
        index = bar.indexOf(self._teleop_grip_label)
        bar.insertWidget(
            index + 1 if index >= 0 else bar.count(), self._grip_latch_label
        )
        return bar

    def _build_home_bar(self):
        """One row: title | live status | STOP HOME | GO HOME.

        STOP HOME sits to the LEFT of GO HOME and is red rather than default:
        it is the emergency-ish control of the pair and must not be reachable
        by muscle memory aimed at GO HOME. Its style is set exactly once here
        (Qt greys a disabled button by itself), so the ~5 Hz refresh only ever
        drives its enabled state.
        """
        bar = QHBoxLayout()

        title = QLabel("Go home:")
        title.setStyleSheet("font-weight: bold;")

        self._home_status_label = QLabel(HomeMoveState.IDLE)
        self._home_status_label.setWordWrap(True)
        self._home_status_label.setStyleSheet(
            "color: {}; font-weight: bold;".format(_HOME_COLOR_IDLE)
        )

        self._stop_home_button = QPushButton(_STOP_HOME_BUTTON_TEXT)
        self._stop_home_button.setStyleSheet(
            _BIG_BUTTON_STYLE
            + "background-color: {}; color: white; font-weight: bold;".format(
                _HOME_COLOR_FAILED
            )
        )
        self._stop_home_button.setToolTip(_STOP_HOME_TOOLTIP)
        # Enabled only while a sequence is in flight; _refresh_home_controls
        # owns it from here on.
        self._stop_home_button.setEnabled(False)
        self._stop_home_button.clicked.connect(self._on_stop_home_clicked)

        self._home_button = QPushButton(_HOME_BUTTON_TEXT)
        self._home_button.setStyleSheet(_BIG_BUTTON_STYLE)
        self._home_button.clicked.connect(self._on_go_home_clicked)

        bar.addWidget(title)
        bar.addSpacing(8)
        bar.addWidget(self._home_status_label, stretch=1)
        bar.addWidget(self._stop_home_button)
        bar.addWidget(self._home_button)
        return bar

    # -------------------------------------------------------------- timers --
    def _refresh_controls(self):
        """~5 Hz (base class's _control_timer): base controls, then our own.

        Order matters. ``super()._refresh_controls()`` ends by calling
        ``_refresh_teleop()``, which RE-DRIVES the Teleop bar's button
        enablement every single tick -- so both of our overrides have to run
        AFTER it, or the base simply undoes them 5 times a second.
        """
        super()._refresh_controls()
        self._refresh_home_controls()
        self._neutralize_teleop_resume()
        self._refresh_grip_latch()

    def _refresh_grip_latch(self):
        """Paint the discrete-latch readout from the node's cached token.

        Polls the node exactly like every other label in this window: the node
        caches the topic under its own lock on the spin thread, this reads a
        snapshot on the Qt thread. No ROS is touched from here, and nothing on
        the spin thread touches this widget.

        Unlike the two overrides above, this one is not fighting the base for a
        widget -- the base's ``_refresh_teleop`` does not know this label exists
        -- so its position in the tick is a matter of taste, not correctness.
        """
        status = self._node.get_gripper_discrete_status()
        text, color = _grip_latch_display(status.get("state"), status.get("age_s"))
        self._grip_latch_label.setText(text)
        self._grip_latch_label.setStyleSheet(
            "color: {}; font-weight: bold;".format(color)
        )

    def _neutralize_teleop_resume(self):
        """Permanently disable the inherited "Resume Teleop" button.

        WHY, and why permanently rather than only while homing:

        That button calls ``/gello_ur_bridge/resume_chase`` (the base node's
        ``_svc_teleop["arm_resume_chase"]``), which is the JOINT-mode chase
        resume: the bridge lands in JOINT_BOOTSTRAP -- joint passthrough -- and
        its gate admits a per-joint leader-vs-arm gap of up to
        ``resume_chase_max_gap`` = 1.5 rad. THIS GUI drives a
        ``control_mode:=eef`` session, and GO HOME deliberately parks the arm at
        HOME while the passive GELLO leader stays wherever the operator left it,
        which is precisely that divergence. Inside the gate the arm would glide
        up to 1.5 rad PER JOINT toward the leader's absolute joint
        configuration -- not the Cartesian-delta semantics an operator expects
        from a button labelled "Resume"; outside the gate it is merely refused,
        which is confusing rather than harmful. There is no EEF-session use for
        it either way.

        The correct EEF re-arm is the EEF GUI's ENGAGE button, which chains
        eef_resume -> pos_scale -> eef_engage: no alignment gate, and it
        re-anchors at the CURRENT pose, so a GO HOME cannot break it.

        "Pause Teleop" is deliberately left fully functional -- pausing is
        unconditional, always safe, and genuinely useful for scene resets.

        The widget stays VISIBLE (operators recognise the bar; a vanished
        button reads as a broken GUI) but can never be pressed: the base's
        ``_refresh_teleop`` force-enables it whenever ``_teleop_resume_armed``
        is set, so we clear that flag as well as the enabled state. Both
        assignments happen inside the same slot invocation as the base's, with
        no event processing in between, so the button is never momentarily
        clickable.

        WHAT IS REASSERTED EVERY TICK AND WHAT IS NOT. Only the two lines the
        base actually fights us over are re-driven at 5 Hz -- ``setEnabled``
        and the armed flag, both of which ``_refresh_teleop`` re-drives on
        every one of its own ticks. The label, stylesheet and tooltip are
        static and nothing else writes them (the base only rewrites the label
        from ``_disarm_teleop_resume``, which is reachable only from the click
        handler of a button that can never be clicked), so they are applied
        exactly once. That is not just tidiness: ``setStyleSheet`` forces a
        full unpolish/polish of the widget on EVERY call even when the sheet is
        byte-identical, which is 5 restyles a second for the lifetime of the
        session, forever.
        """
        self._teleop_resume_armed = False
        self._teleop_resume_button.setEnabled(False)
        if not self._teleop_resume_neutralized:
            self._teleop_resume_neutralized = True
            self._teleop_resume_button.setText(_TELEOP_RESUME_REDIRECT_TEXT)
            self._teleop_resume_button.setStyleSheet(_BIG_BUTTON_STYLE)
            self._teleop_resume_button.setToolTip(_TELEOP_RESUME_TOOLTIP)

    def _refresh_home_controls(self):
        """Drive BOTH home buttons + the status label from the node status.

        Single enablement path on purpose: GO HOME, STOP HOME and the close
        guard's deadline all move on this one ~5 Hz tick, so there is no second
        refresh that could disagree with this one about whether a move is in
        flight.
        """
        status = self._node.get_home_status()
        state = status.get("state", HomeMoveState.IDLE)
        active = bool(status.get("active", False))
        message = status.get("message") or ""
        recording = self._node.is_recording()

        # Never home while a take is recording (the homing motion is not part of
        # the demonstration) and never re-enter a running sequence. The one
        # exception is the armed window: keep the button live so the CONFIRMING
        # click can land, exactly the trick the base uses for Resume.
        if self._home_armed:
            self._home_button.setEnabled(True)
        else:
            self._home_button.setEnabled(not recording and not active)

        # STOP HOME is the exact complement: live only while something is in
        # flight to stop, and driven from THIS tick so there is a single
        # enablement path for the whole bar. `recording` deliberately does not
        # appear -- a stop must never be gated on anything.
        self._stop_home_button.setEnabled(active)

        # A finished sequence releases the close guard, so the next GO HOME
        # gets a full grace window instead of inheriting an expired deadline
        # from the previous one (see _veto_close_during_home_move).
        if not active:
            self._home_close_deadline = None

        if active:
            # Do not offer to record the homing motion; the base refresh already
            # ran, so this override sticks until the sequence ends and the base
            # restores the normal label on the next tick.
            self._start_button.setEnabled(False)
            self._start_button.setText("Going home...")
            # (Resuming teleop mid-sequence is covered by
            # _neutralize_teleop_resume, which disables that button
            # unconditionally -- not just while a GO HOME is in flight.)

        self._home_status_label.setText(_home_status_text(state, message))
        self._home_status_label.setStyleSheet(
            "color: {}; font-weight: bold;".format(_home_status_color(state, active))
        )

        # Announce the terminal outcome once, on the transition into it.
        if state != self._home_last_state:
            self._home_last_state = state
            if state in (HomeMoveState.DONE, HomeMoveState.FAILED):
                self.statusBar().showMessage(
                    "GO HOME {}: {}".format(state, message), 15000
                )

    # ------------------------------------------------------------- actions --
    def _on_go_home_clicked(self):
        """Two-click confirm, identical in shape to the base Resume button.

        The first click only ARMS: it turns the button orange, spells out that
        the robot will move fast, and schedules a 3 s auto-disarm. The second
        click within that window is what actually requests the motion.
        """
        if not self._home_armed:
            self._home_armed = True
            self._home_button.setText(_HOME_BUTTON_ARMED_TEXT)
            self._home_button.setStyleSheet(
                _BIG_BUTTON_STYLE
                + "background-color: #dd8800; color: white; font-weight: bold;"
            )
            QTimer.singleShot(_HOME_CONFIRM_MS, self._disarm_go_home)
            self.statusBar().showMessage(
                "Click again within 3 s to GO HOME -- teleop will be FORCE-PAUSED "
                "and the robot WILL move fast. Keep clear.",
                _HOME_CONFIRM_MS,
            )
            return

        self._disarm_go_home()

        # Re-check RECORDING here, not only in the ~5 Hz refresh. The armed
        # window deliberately force-enables this button (see
        # _refresh_home_controls) so the confirming click is never swallowed by
        # a timer tick -- which means the refresh's "never home while a take is
        # recording" rule is BYPASSED for exactly those 3 s, and a take started
        # between the two clicks would otherwise let GO HOME drive the arm
        # through the middle of a recording. The other half of the mutual
        # exclusion ("already running") is enforced downstream by
        # HomeMoveController.request(); `recording` has no equivalent there
        # because home_move.py knows nothing about takes. Both click handlers
        # run on the Qt thread, so this check cannot race _on_start_clicked.
        if self._node.is_recording():
            self.statusBar().showMessage(
                "GO HOME refused -- a take is RECORDING. Stop the take first.",
                6000,
            )
            return

        fired = self._node.request_go_home()
        if fired:
            self.statusBar().showMessage(
                "GO HOME started: pausing teleop, driving to HOME, opening the "
                "gripper. Teleop stays PAUSED afterwards.",
                8000,
            )
        else:
            # request() refuses when a sequence is already running, or when the
            # arm's pose is unknown; either way the reason is in the status line
            # the ~5 Hz refresh is already painting.
            self.statusBar().showMessage(
                "GO HOME not started -- see the Go home status line.", 6000
            )

    def _disarm_go_home(self):
        """Revert the confirm button to its resting look.

        The ~5 Hz refresh re-drives its ENABLED state on the next tick, so this
        only has to undo the text and the colour.
        """
        self._home_armed = False
        self._home_button.setText(_HOME_BUTTON_TEXT)
        self._home_button.setStyleSheet(_BIG_BUTTON_STYLE)

    def _on_stop_home_clicked(self):
        """SINGLE click, no confirm: the operator STOP for an in-flight GO HOME.

        Deliberately the opposite of :meth:`_on_go_home_clicked`. The two-click
        arm there exists to stop an ACCIDENTAL START; a stop has no equivalent
        failure mode, and every extra click is time the arm spends still
        driving. Nothing gates this button except "is there a move to stop"
        (:meth:`_refresh_home_controls`) -- not `recording`, not anything else.

        This is also the one abort path in this window: :meth:`closeEvent`
        calls this method rather than the node directly, so a refused close and
        a button press cannot drift apart.

        Returns the node's verdict (True = the machine took the request), which
        is surfaced either way -- a stop button that silently does nothing is
        worse than no stop button.
        """
        fired = self._node.request_stop_home()
        if fired:
            self.statusBar().showMessage(
                "STOP HOME requested -- cancelling the home trajectory and "
                "restoring {}. NOT instantaneous and NOT an E-stop: the "
                "controller decelerates on its own schedule, so the arm keeps "
                "moving briefly and stops partway to HOME. Teleop stays "
                "PAUSED.".format(FPC),
                10000,
            )
        else:
            # abort() returns False when nothing is in flight; the button is
            # normally disabled then, so this is the tail of a race with the
            # sequence finishing on its own.
            self.statusBar().showMessage(
                "STOP HOME ignored -- no GO HOME is in flight. See the Go home "
                "status line for how the last one ended.",
                6000,
            )
        return fired

    # --------------------------------------------------------------- close --
    def closeEvent(self, event):
        """Refuse to close out from under a moving arm; otherwise defer to base.

        THE FAILURE THIS EXISTS TO PREVENT. Closing the window is the
        instinctive panic response, and without this override it does nothing
        whatsoever to the robot: a FollowJointTrajectory goal STJC has already
        accepted is unaffected by the client going away (ROS 2 actions are not
        cancelled when their client dies), so the arm finishes its sweep to
        HOME while the state machine that was going to hand the joints back to
        forward_position_controller is gone. The arm ends up parked with the
        bridges PAUSED and scaled_joint_trajectory_controller ACTIVE -- and the
        next ENGAGE on the EEF GUI then streams to
        /forward_position_controller/commands with nothing listening: teleop
        that looks alive and is silently dead, recoverable only from a
        terminal.

        So: while a move is active the first close is REFUSED and turned into
        the STOP HOME the operator actually wanted. Once the machine is
        terminal, closing runs the base ``closeEvent`` untouched -- take flush,
        camera teardown, rclpy shutdown, exactly as in the plain recorder.

        The refusal is bounded (see ``_HOME_CLOSE_GRACE_S``): a wedged
        sequencer must not be able to hold an operator hostage in a GUI.
        """
        if self._veto_close_during_home_move(event):
            return
        super().closeEvent(event)

    def _veto_close_during_home_move(self, event):
        """True if this close was refused (and turned into a STOP HOME).

        Split out of :meth:`closeEvent` so the decision is a plain function of
        the status dict plus the deadline -- callable, and testable, without a
        QApplication.

        Three outcomes:

        * not active -> False, and the guard resets. Close proceeds to the base.
        * active, within the grace window -> the event is IGNORED, STOP HOME is
          requested through the same path as the button, the operator is told
          why in the window (never a modal -- one would block the very Qt
          thread they need in order to press STOP HOME), and True is returned.
        * active, but the grace window has expired -> False, after a loud
          stderr warning naming the recovery command. Letting go of a moving
          arm is bad; trapping the operator in a window is worse, and the
          warning is what makes the first survivable.
        """
        status = self._node.get_home_status()
        if not bool(status.get("active", False)):
            self._home_close_deadline = None
            return False

        now = time.monotonic()
        if self._home_close_deadline is None:
            # First refusal starts the clock. It is NOT extended by later
            # attempts, or repeated closing would postpone the escape hatch
            # forever.
            self._home_close_deadline = now + _HOME_CLOSE_GRACE_S
        elif now >= self._home_close_deadline:
            print(_HOME_STRANDED_WARNING, file=sys.stderr)
            try:
                sys.stderr.flush()
            except Exception:  # noqa: BLE001 -- never block a close on stderr
                pass
            return False

        event.ignore()
        self._on_stop_home_clicked()
        # Shown last so it is the message left standing in the status bar.
        self.statusBar().showMessage(_HOME_CLOSE_REFUSED_TEXT, 10000)
        print(_HOME_CLOSE_REFUSED_STDERR, file=sys.stderr)
        return True


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main(args=None):
    """Same startup as gello_recorder_gui.main(), with the task node/window.

    Kept deliberately parallel to the base main(): identical env-var surface,
    identical camera-serial resolution, identical subprocess/spin/SIGINT
    plumbing. The only additions are the class names and an optional
    TASK_RECORDER_OUTPUT_ROOT override; with no env vars set at all, takes land
    in the same ros2_ur_ws/gello_logs the plain recorder writes to.
    """
    # Config from env vars, matching run_recorder.sh's names/defaults.
    cam1_serial = os.environ.get("CAM1_SERIAL", DEFAULT_CAM1_SERIAL)
    cam2_serial = os.environ.get("CAM2_SERIAL", DEFAULT_CAM2_SERIAL)
    cam1_name = os.environ.get("CAM1_NAME", DEFAULT_CAM1_NAME)
    cam2_name = os.environ.get("CAM2_NAME", DEFAULT_CAM2_NAME)
    color_profile = os.environ.get("COLOR_PROFILE", DEFAULT_COLOR_PROFILE)

    # --- Resolve serials against what is actually on the USB bus ---------- #
    # Same policy as the base GUI: a hard mismatch (< 2 devices) aborts BEFORE
    # any camera subprocess is spawned, because a realsense2_camera node bound
    # to a serial we already know is wrong comes up, publishes nothing, and
    # produces a silently-broken GUI instead of a clear error.
    cam_warnings = []
    try:
        cam1_serial, cam2_serial, cam_warnings = _resolve_camera_serials(
            cam1_serial, cam2_serial
        )
    except ImportError:
        cam_warnings = [
            "pyrealsense2 not importable; using configured serials as-is"
        ]
    except RuntimeError as exc:
        print("### ERROR resolving camera serials: {}".format(exc), file=sys.stderr)
        sys.exit(1)

    for warning in cam_warnings:
        print("### WARN {}".format(warning), file=sys.stderr)

    # Parse "WxHxFPS" -> fps for the node's camera_fps arg (best-effort).
    camera_fps = 30.0
    try:
        camera_fps = float(color_profile.split("x")[-1])
    except (ValueError, IndexError):
        pass
    camera_warmup_s = float(os.environ.get("CAMERA_WARMUP_S", "4.0"))
    output_root = os.environ.get(
        "TASK_RECORDER_OUTPUT_ROOT",
        os.environ.get(
            "RECORDER_OUTPUT_ROOT",
            os.path.join(
                os.environ.get(
                    "GELLO_REPO_ROOT",
                    os.path.expanduser("~/gello_software"),
                ),
                "ros2_ur_ws", "gello_logs",
            ),
        ),
    )

    # The two color topics the realsense nodes publish under their namespace.
    cam1_topic = "/{0}/{0}/color/image_raw/compressed".format(cam1_name)
    cam2_topic = "/{0}/{0}/color/image_raw/compressed".format(cam2_name)

    # --- 1. Launch the two RealSense camera nodes as subprocesses --------- #
    cam1_proc = _launch_realsense(cam1_name, cam1_serial, color_profile)
    cam2_proc = _launch_realsense(cam2_name, cam2_serial, color_profile)

    # --- 2. Bring up ROS + the node --------------------------------------- #
    rclpy.init(args=args)

    # Imported here (not at module top) to mirror the base GUI's lazy-import
    # pattern: this module stays importable/py_compilable without a sourced ROS
    # overlay, which the console_script wrapper needs anyway.
    from gello_recorder.task_gui_node import TaskRecorderGuiNode

    # opencv-python's wheel bundles its own Qt5 and points
    # QT_QPA_PLATFORM_PLUGIN_PATH at it as a side effect of `import cv2` (pulled
    # in transitively by the node). That is a different Qt5 build than the
    # system PyQt5 we want; loading both in one process breaks Qt's xcb plugin.
    # Clear it before any QApplication is constructed.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    node = TaskRecorderGuiNode(
        cam1_topic=cam1_topic,
        cam2_topic=cam2_topic,
        camera_fps=camera_fps,
        camera_warmup_s=camera_warmup_s,
        output_root=output_root,
    )

    # --- 3. Spin the node on a background daemon thread ------------------- #
    # Plain SingleThreadedExecutor via rclpy.spin -- the GO-HOME path is written
    # against exactly that (no blocking waits anywhere), so do NOT swap in a
    # MultiThreadedExecutor without re-reading task_gui_node's threading notes.
    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()

    # --- 4. Qt event loop on the main thread ------------------------------ #
    app = QApplication(sys.argv if args is None else args)
    window = TaskRecorderWindow(node, cam1_proc, cam2_proc, cam_warnings=cam_warnings)
    window.resize(1280, 800)
    window.show()

    # Ctrl-C in the launching terminal is otherwise SILENTLY SWALLOWED: Qt's C++
    # event loop never yields back to the interpreter, so Python's SIGINT
    # handler never gets a chance to run. Make SIGINT ask the WINDOW to close,
    # and use a short-interval QTimer purely to tick the interpreter often
    # enough for that handler to fire promptly.
    #
    # WHY window.close() AND NOT THE BASE'S app.quit(). Ctrl-C mid-GO-HOME has
    # exactly the same end state as clicking the X -- a moving arm, bridges
    # paused, the joints stranded on scaled_joint_trajectory_controller -- but
    # it reaches it by a different route: app.quit() leaves the event loop
    # WITHOUT ever delivering closeEvent, so TaskRecorderWindow.closeEvent (and
    # with it the STOP HOME + fail-closed controller restore) never runs.
    # window.close() delivers a real close event, so both routes funnel through
    # the one guard: refused while a move is active, honoured once the machine
    # is terminal or the bounded grace has expired, and then Qt's default
    # quitOnLastWindowClosed ends exec_() as before. A second Ctrl-C is simply a
    # second close attempt, which is what makes the escape hatch reachable from
    # the terminal too.
    #
    # QApplication.aboutToQuit was the alternative and is strictly weaker here:
    # it fires once quitting is already under way and CANNOT veto it, so it
    # could warn about a stranded controller but never prevent one. Nothing else
    # in this main() calls app.quit(), so there is no quit path left uncovered.
    signal.signal(signal.SIGINT, lambda *_: window.close())
    sigint_pump = QTimer()
    sigint_pump.start(200)
    sigint_pump.timeout.connect(lambda: None)

    exit_code = app.exec_()

    # Belt-and-suspenders cleanup in case closeEvent didn't run (it normally
    # does for a window close). These are all idempotent / guarded.
    window._shutdown_cameras()
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    spin_thread.join(timeout=2.0)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
