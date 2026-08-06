#!/usr/bin/env python3
"""GELLO gripper -> Robotiq 2F-85 command_percent streaming bridge.

This node reads the GELLO leader's normalized gripper width and republishes it
as a Robotiq command percent for the Modbus gripper node. GELLO stays PASSIVE:
this node only reads a Float32 topic and publishes a Float32 topic — it never
touches any Dynamixel or robot hardware directly.

Direction invariant (SAFETY-CRITICAL)
-------------------------------------
The GELLO width topic ``/gripper/gripper_client/target_gripper_width_percent``
publishes ``js[6]`` in [0.0, 1.0] where **0.0 = OPEN, 1.0 = CLOSED** (see the
gello DynamixelRobot normalization). The Modbus node's ``~/command_percent`` is
defined with the SAME convention: **0.0 = OPEN, 1.0 = CLOSED** (raw Robotiq
POS = round(percent*255), 0=open/255=closed). Therefore the mapping is a DIRECT
IDENTITY: ``command_percent = clamp(width, 0, 1)`` — NO inversion. This preserves
direction end-to-end so closing your hand on GELLO CLOSES the robot gripper.

The ``invert`` parameter (default **False**) exists only for a future mechanical
recalibration flip; it MUST stay False for this hardware. Inverting it would
command robot-CLOSE when the GELLO hand OPENS — a crush hazard. The startup log
prints the resolved direction so it is verifiable before touching the robot.

Pipeline
--------
On each received width: apply the (identity) map, optional invert, clamp to
[clamp_min, clamp_max], optional EMA smoothing (ema_alpha=1.0 => off), the
OPTIONAL discrete latch (see below, off by default), then a deadband gate that
suppresses tiny at-rest changes before publishing. The authoritative rate-limit
toward the single-client Modbus bus is performed by the gripper node itself, so
this bridge republishes on receive at the ~30 Hz GELLO rate. The only timers it
owns are the ~/state publisher and the SETTLE RE-ASSERT below, and both are
silent at rest.

Delivery robustness (settle re-assert / reconcile) — measured 2026-08-06
------------------------------------------------------------------------
The deadband gate above means every setpoint is published EXACTLY ONCE, at the
moment it changes. That is one UDP-ish DDS message per intent, and the Robotiq
node's command rate-limiter used to DISCARD any sample arriving inside its
1/command_rate_hz window — so the LAST sample of a fast trigger release, the one
that says "fully open", was lost roughly half the time. Measured on hardware:
**7 of 14 open cycles parked short** (0.043 / 0.055 / 0.11 / 0.42 instead of
0.0118), and re-asserting the final setpoint for 1 s made it **0 of 6**.

The real fix is in the driver (its rate limiter must coalesce, not drop). These
two are DEFENCE IN DEPTH in the teleop path, and they are what was actually
validated 0/6 on the robot:

* **Settle re-assert** (``settle_reassert_s``, ``settle_reassert_hz``). For a
  short window after the OUTPUT LAST CHANGED, re-publish the same value at a low
  rate, then go silent. Position-blind on purpose: it re-asserts the operator's
  own live intent and is never triggered by where the fingers ended up, so it
  cannot be provoked by a grasp. ``settle_reassert_s: 0.0`` disables the timer
  entirely and restores the publish-once behaviour exactly.
* **Reconcile** (``reconcile_tol``, ``reconcile_after_s``). Watches the measured
  ``actual_topic`` position and, when the gripper sits MORE CLOSED than the last
  thing this bridge commanded for longer than ``reconcile_after_s``, re-asserts
  that command once. See ``_maybe_reconcile`` for why the direction is
  one-sided — that asymmetry is what makes it impossible to squeeze a grasped
  object. ``reconcile_after_s: 0.0`` (or ``reconcile_tol: 0.0``) disables it.
* **Resume force-publish** (``_force_next_publish``). This one has NO parameter
  and CANNOT be turned off, so "set the new params to 0 and you have the old
  node" is true of the two above and FALSE of this one. It is a bug fix, not a
  feature: ``_on_resume`` writes ``_last_pub = seed`` without publishing it, and
  a leader already resting within ``deadband`` of that seed was then suppressed
  by the deadband gate forever, so the gripper received nothing at all while
  ``~/state`` reported FOLLOWING. The extra message is one publish of the value
  the operator's own live trigger is commanding at that instant, so there is no
  configuration in which restoring the hang would be correct.

Neither of the two re-assert paths ever introduces a NEW setpoint: both
republish ``_last_pub`` verbatim, so the sequence of distinct values on the wire
is bit-identical to what it was — only the number of copies changes, and only
inside bounded windows. At rest, with a converged output and no discrepancy, the bridge still
publishes NOTHING; the Modbus bus is single-client and must stay quiet.

Discrete mode (OPT-IN, ``discrete_mode`` default False)
-------------------------------------------------------
The GELLO trigger is spring-loaded and read by a Dynamixel encoder, and across
125 recorded takes / 52,607 samples it is already essentially binary: 90.95% of
samples sit below 0.02 or above 0.98 and resting plateaus land ONLY at the
extremes (0 of them in 0.3..0.7). BUT roughly once or twice per session the
spring does not fully return and the trigger comes to rest PARTWAY OPEN — the
worst observed resting values are 0.229 (2026-07-07) and 0.243 (2026-07-20).
This bridge faithfully forwards that value, so the Robotiq only opens ~76% and
the operator sees a gripper that "didn't fully open". EVERY observed outlier is
below 0.3, so snapping anything <= 0.3 to exactly 0.0 fixes all of them. That is
the entire purpose of this feature.

The thresholds 0.3 / 0.7 and the hysteresis rule are NOT new: they are the
canonical constants already used by the offline RL demo converter
(``serl_ur_infra/ur_env/learner/recorded_demo.py``) and are specified in
``docs/ros2/GELLO_UR7E_UNITS_REFERENCE.md`` §5.1 together with the evidence
table above. This implementation mirrors that latch bit-for-bit::

    value >= discrete_close_at (0.7)  ->  latch = CLOSED (1.0)
    value <= discrete_open_at  (0.3)  ->  latch = OPEN   (0.0)
    otherwise                         ->  keep the previous latch  (hysteresis)

Hysteresis (rather than one threshold) is what stops the output chattering while
the trigger sweeps through the middle. Note the RL convention is the OPPOSITE
sign (+1=OPEN, -1=CLOSED); on THIS topic 0.0=OPEN and 1.0=CLOSED (§5), so the
latch endpoints here are 0.0/1.0, not -1/+1.

``~/discrete_state`` (String, same ``state_publish_rate_hz`` as ``~/state``)
reports DISABLED / UNKNOWN / RAMPING / OPEN / CLOSED. It is a SEPARATE topic on
purpose: ``~/state``'s PAUSED/WAITING/RAMPING/FOLLOWING vocabulary is parsed by
other code and must not grow new tokens. RAMPING covers the whole post-resume
slew window: there the wire carries intermediate values that are NOT the latch,
and reporting the committed endpoint while the gripper is still travelling would
turn the operator's indicator into a lie exactly when it matters most.

``~/discrete_trigger`` (Float32, same rate) carries the RAW CLAMPED trigger value
that the latch thresholds against, so a GUI can show the live reading NEXT TO the
latch. That pairing is what makes a mis-set threshold visible: with
``discrete_open_at`` set below a stuck resting value (0.243 was observed) the
trigger never crosses anything, the latch freezes, and the gripper silently stops
responding — indistinguishable, from the latch alone, from an operator holding a
grasp. The same condition also raises a THROTTLED warning after ~1.5 s of
continuous dwell strictly inside the band.

Pause / resume (drop-hazard mitigation)
---------------------------------------
A passive GELLO leader, when the operator lets go, COLLAPSES and its flopping
gripper axis streams straight through this bridge — a crush-or-drop hazard on
the real robot. Two NEW Trigger services gate that:

* ``~/pause``  — UNCONDITIONAL, always succeeds. Sets ``_paused`` and the width
  callback returns BEFORE the pipeline, so NOTHING is republished. The Robotiq
  holds its last commanded position onboard. Not-streaming is the safe state.
* ``~/resume`` — FAIL-CLOSED. Refuses (staying paused + silent) unless a FRESH
  leader sample exists AND an actual gripper position is known. On acceptance it
  seeds the output at the gripper's ACTUAL position (zero jump) and, for
  ``resume_ramp_s`` seconds, slew-limits every published sample toward the live
  leader value (bypassing the deadband so the ramp advances monotonically). Only
  after the ramp window does it revert to the plain deadbanded pass-through. An
  accepted resume ALSO resets the discrete latch to UNKNOWN — the world may have
  moved while we were paused (the task recorder's GO HOME force-opens the Robotiq
  behind this bridge's back), so the pre-pause intent must be re-earned from the
  operator's live trigger rather than replayed. See ``_on_resume``.

``~/state`` (String, published at ``state_publish_rate_hz``) reports one of
PAUSED / WAITING / RAMPING / FOLLOWING for the operator UI.
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

# Discrete latch endpoints, in THIS topic's convention (0=OPEN, 1=CLOSED — see
# module docstring / UNITS_REFERENCE §5). NOT the RL ±1 convention.
DISCRETE_OPEN = 0.0
DISCRETE_CLOSED = 1.0

# ~/discrete_state vocabulary. DISABLED covers both "discrete_mode:=false" and
# "asked for discrete but the thresholds were invalid so we fell back", because
# from a consumer's point of view they are the same thing: the continuous value
# is being forwarded. The loud startup error is where the difference lives.
DISCRETE_STATE_DISABLED = "DISABLED"
DISCRETE_STATE_UNKNOWN = "UNKNOWN"
DISCRETE_STATE_OPEN = "OPEN"
DISCRETE_STATE_CLOSED = "CLOSED"
# Post-resume slew window. Deliberately the SAME WORD as ~/state's RAMPING (they
# describe the same window) but on a different topic, so no existing ~/state
# parser sees a new token. It outranks OPEN/CLOSED because during the window the
# published value is an intermediate slew sample, not the latch, and it outranks
# UNKNOWN because after a resume the latch is ALWAYS UNKNOWN for at least one
# sample — reporting that would hide the ramp entirely.
DISCRETE_STATE_RAMPING = "RAMPING"

# In-band dwell alarm (see _track_band_dwell). 1.5 s is far longer than any
# human sweep THROUGH the band (the §5.1 corpus records 0 resting plateaus inside
# 0.3..0.7) and far shorter than the operator's patience with a gripper that has
# stopped responding.
DISCRETE_BAND_DWELL_WARN_S = 1.5
# Re-warn period, so a trigger parked in the band logs once every few seconds
# instead of at the 30 Hz sample rate.
DISCRETE_BAND_WARN_PERIOD_S = 5.0

# Re-warn period for the reconcile watchdog (see _maybe_reconcile). Same value
# and same reasoning as DISCRETE_BAND_WARN_PERIOD_S, kept as its own name
# because the two throttles guard unrelated conditions and either may be
# retuned alone. A discrepancy the gripper can never close (jammed fingers, or
# a command that was in fact delivered so the driver's own deadband swallows
# every duplicate) is PERMANENT: the re-assert must keep contesting it, but the
# log must not run at the re-assert rate for the rest of the session.
RECONCILE_WARN_PERIOD_S = 5.0


def validate_discrete_thresholds(open_at: float, close_at: float) -> str | None:
    """Return None if the thresholds are usable, else a human-readable reason.

    The contract is ``0.0 < open_at < close_at < 1.0``. The strict outer bounds
    matter as much as the ordering: ``open_at == 0.0`` would only ever latch OPEN
    on an EXACT 0.0 sample (which defeats the whole point — the 0.229 outlier
    would never snap), and ``close_at == 1.0`` is the mirror image. Equal
    thresholds collapse the hysteresis band to zero and re-introduce the chatter
    the band exists to prevent.
    """
    for name, value in (("discrete_open_at", open_at), ("discrete_close_at", close_at)):
        if value != value:  # NaN
            return f"{name}={value} is not a number"
    if not 0.0 < open_at:
        return f"discrete_open_at={open_at} must be > 0.0"
    if not close_at < 1.0:
        return f"discrete_close_at={close_at} must be < 1.0"
    if not open_at < close_at:
        return (
            f"discrete_open_at={open_at} must be < discrete_close_at={close_at} "
            "(they bracket the hysteresis band)"
        )
    return None


def discrete_latch(
    value: float,
    previous: float | None,
    open_at: float,
    close_at: float,
) -> float | None:
    """Fold a continuous trigger value into the latched OPEN/CLOSED endpoint.

    Mirrors ``recorded_demo.py``'s latch bit-for-bit (including the ``>=`` /
    ``<=`` inclusivity and the close-first ordering), except that the endpoints
    are this topic's 0.0=OPEN / 1.0=CLOSED rather than the RL ±1 convention.

    ``previous`` is the latch carried over from the last sample, or None when no
    threshold has ever been crossed. Returning None means "still unknown" — the
    caller MUST publish nothing in that case rather than guessing an endpoint.
    """
    if value >= close_at:
        return DISCRETE_CLOSED
    if value <= open_at:
        return DISCRETE_OPEN
    return previous  # hysteresis: inside the band, remember


class GelloGripperBridge(Node):
    """Map the GELLO gripper width to a Robotiq command_percent (0=open..1=closed)."""

    def __init__(self) -> None:
        super().__init__("gello_gripper_bridge")

        # --- Parameters --------------------------------------------------
        # invert MUST stay False for this robot (see module docstring / SAFETY).
        self.invert = bool(self.declare_parameter("invert", False).value)
        # Deadband (in percent units) suppressing at-rest Dynamixel jitter on the
        # gripper axis; a change smaller than this is not republished.
        self.deadband = float(self.declare_parameter("deadband", 0.02).value)
        # EMA smoothing factor: 1.0 == OFF (no lag; safest so an emergency hand
        # open reaches the gripper immediately). <1 smooths but adds lag.
        self.ema_alpha = float(self.declare_parameter("ema_alpha", 1.0).value)
        self.clamp_min = float(self.declare_parameter("clamp_min", 0.0).value)
        self.clamp_max = float(self.declare_parameter("clamp_max", 1.0).value)
        self.publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 30.0).value
        )
        self._in_topic = str(
            self.declare_parameter(
                "input_topic",
                "/gripper/gripper_client/target_gripper_width_percent",
            ).value
        )
        # Absolute topic so it lands on /robotiq_gripper/command_percent
        # regardless of this node's namespace.
        self._out_topic = str(
            self.declare_parameter(
                "output_topic", "/robotiq_gripper/command_percent"
            ).value
        )
        # Actual gripper position feedback (Robotiq modbus node publishes this as
        # Float32 on /robotiq_gripper/position_percent, 0=open..1=closed). Used to
        # SEED resume so the first output equals where the gripper actually is.
        self._actual_topic = str(
            self.declare_parameter(
                "actual_topic", "/robotiq_gripper/position_percent"
            ).value
        )
        # --- Pause/resume params ---
        # A leader sample older than this (s) is STALE — resume refuses on it so a
        # dead/frozen stream can never re-open the drop hazard.
        self.staleness_timeout_s = float(
            self.declare_parameter("staleness_timeout_s", 0.5).value
        )
        # Duration (s) of the slew-limited ramp after a resume.
        self.resume_ramp_s = float(
            self.declare_parameter("resume_ramp_s", 2.0).value
        )
        # Max output change per second (fraction of stroke) during the resume ramp.
        # 0.6 => full 0..1 stroke covered in ~1.67 s, so resume_ramp_s=2.0 always
        # lets a static leader converge inside the window.
        self.resume_slew_per_s = float(
            self.declare_parameter("resume_slew_per_s", 0.6).value
        )
        self.state_publish_rate_hz = float(
            self.declare_parameter("state_publish_rate_hz", 5.0).value
        )
        # --- Delivery robustness params (see module docstring) ---
        # Window (s) after the OUTPUT LAST CHANGED during which the same value is
        # re-published at settle_reassert_hz. 1.0 s is the value measured 0/6 on
        # hardware (2026-08-06) against 7/14 without it. 0.0 => no timer is
        # created at all and the node publishes exactly once per change, which is
        # byte-for-byte the pre-2026-08-06 behaviour.
        self.settle_reassert_s = float(
            self.declare_parameter("settle_reassert_s", 1.0).value
        )
        # Re-assert rate inside that window. 10 Hz => at most 10 duplicate
        # messages per operator gesture, well under the driver's 20 Hz limiter,
        # and ZERO at rest.
        self.settle_reassert_hz = float(
            self.declare_parameter("settle_reassert_hz", 10.0).value
        )
        # Reconcile: how far the MEASURED position may sit on the CLOSED side of
        # the last commanded value before we consider the command lost. 0.05 is
        # ~13 raw Robotiq counts — comfortably above the driver's own 0.01
        # command deadband and above encoder noise, and far below the smallest
        # observed failure (0.043 vs a 0.0118 target). 0.0 disables reconcile.
        self.reconcile_tol = float(
            self.declare_parameter("reconcile_tol", 0.05).value
        )
        # How long that discrepancy must persist CONTINUOUSLY before one
        # re-assert, and the minimum period between re-asserts. 0.5 s is long
        # enough for a real move to finish (full stroke is ~0.5 s at speed 150)
        # so a gripper still travelling toward the setpoint is never re-commanded
        # mid-flight. 0.0 disables reconcile.
        self.reconcile_after_s = float(
            self.declare_parameter("reconcile_after_s", 0.5).value
        )
        # --- Discrete (binary open/closed) mode params ---
        # OFF by default: with discrete_mode=False every line below is inert and
        # the pipeline is byte-identical to what it was before this feature.
        self.discrete_mode = bool(
            self.declare_parameter("discrete_mode", False).value
        )
        # Canonical 0.3/0.7 from UNITS_REFERENCE §5.1 — do NOT invent new values.
        # Exposed as parameters (not constants) because the 2026-07-20 outlier
        # left only 0.057 of margin to 0.3; a re-calibrated gripper may need them
        # re-measured, and the operator must be able to do that without a rebuild.
        self.discrete_close_at = float(
            self.declare_parameter("discrete_close_at", 0.7).value
        )
        self.discrete_open_at = float(
            self.declare_parameter("discrete_open_at", 0.3).value
        )
        # Start paused (drop-hazard-safe default is False for standalone runs; the
        # integrated launch may override to true to pre-spawn held).
        self._paused = bool(
            self.declare_parameter("start_paused", False).value
        )

        # --- State -------------------------------------------------------
        self._f: float | None = None  # EMA-filtered value
        self._last_pub: float | None = None  # last published value (deadband ref)
        self._last_pub_time: float = 0.0  # time.monotonic() of last publish
        # Latest leader sample — updated in _on_width ALWAYS (even while paused) so
        # resume can check freshness.
        self._last_rx_time: float | None = None
        self._last_rx_value: float | None = None
        # Actual gripper position (from _actual_topic); None until first feedback.
        self._actual_pos: float | None = None
        # time.monotonic() of that sample. Separate from the value because a
        # position that stopped arriving is UNUSABLE for reconcile: it would
        # describe where the fingers were, not where they are, and acting on it
        # could re-command a gripper that has since moved. None = never seen.
        self._actual_time: float | None = None
        # time.monotonic() at which the PUBLISHED VALUE last actually CHANGED.
        # Deliberately NOT _last_pub_time: that one is re-based by every publish
        # (it is the resume ramp's slew budget base), whereas the settle window
        # must measure "time since the operator's intent last moved". A re-assert
        # republishes the same number and therefore advances neither. None until
        # the first publish.
        self._output_changed_time: float | None = None
        # Set by an ACCEPTED ~/resume, consumed by the next publish decision in
        # _on_width. Without it, an accepted resume writes _last_pub = seed and
        # publishes nothing, so a leader already sitting within `deadband` of the
        # seed is suppressed by the deadband gate FOREVER and the gripper never
        # receives a command at all (the "phantom seed" bug).
        self._force_next_publish: bool = False
        # time.monotonic() at which the CURRENT reconcile discrepancy started (or
        # the last re-assert fired). None = no discrepancy in progress. Reset on
        # every check that does not qualify, so the dwell must be CONTINUOUS.
        self._reconcile_since: float | None = None
        # time.monotonic() of the last reconcile WARN. Separate from
        # _reconcile_since because the re-assert and its log line are throttled
        # differently on purpose: the re-assert repeats forever (a discrepancy
        # that persists must keep being contested), while the log must not.
        # Cleared with _reconcile_since so the FIRST re-assert of every episode
        # is always logged. See _maybe_reconcile.
        self._reconcile_warn_time: float | None = None
        # End time (time.monotonic()) of the post-resume slew ramp. 0.0 => no ramp.
        self._ramp_until: float = 0.0
        # Discrete latch: None = UNKNOWN (no threshold crossed yet), else
        # DISCRETE_OPEN / DISCRETE_CLOSED. Deliberately NOT seeded from anything.
        # Guessing "probably open" would command a real motion on the strength of
        # a guess; guessing "probably closed" could crush. Holding output until
        # the operator's own trigger crosses a threshold is the only honest start,
        # and it costs nothing: the trigger reaches an extreme within the first
        # grasp cycle (0 takes out of 125 ever failed to reach one).
        # An accepted ~/resume returns it to this same UNKNOWN start, for the
        # same reason — see _on_resume.
        self._discrete_latch: float | None = None
        # Latest RAW CLAMPED trigger value (the exact number the latch compares
        # against), for ~/discrete_trigger. TELEMETRY ONLY — nothing downstream
        # reads it. None until the first leader sample.
        self._trigger_clamped: float | None = None
        # Wall-clock (time.monotonic) at which the trigger most recently ENTERED
        # the hysteresis band, and of the last dwell warning. Both None = idle.
        self._band_since: float | None = None
        self._band_warn_time: float | None = None

        # --- Validate the discrete thresholds (FAIL-SAFE, never fatal) --------
        # A mis-set threshold must never produce a gripper that behaves
        # unpredictably, so on violation we fall back to plain continuous
        # forwarding — the known-good behaviour this node has always had. We do
        # NOT raise: the arm teleop stack depends on this process staying up, and
        # killing it over a typo'd gripper parameter would take teleop down with
        # it. The error is logged loudly instead, and ~/discrete_state reports
        # DISABLED so the operator sees that discrete mode did not take effect.
        self._discrete_fallback_reason: str | None = None
        if self.discrete_mode:
            reason = validate_discrete_thresholds(
                self.discrete_open_at, self.discrete_close_at
            )
            if reason is not None:
                self._discrete_fallback_reason = reason
                self.discrete_mode = False

        # --- ROS interfaces ----------------------------------------------
        self._pub = self.create_publisher(Float32, self._out_topic, 10)
        self._sub = self.create_subscription(
            Float32, self._in_topic, self._on_width, 10
        )
        self._actual_sub = self.create_subscription(
            Float32, self._actual_topic, self._on_actual, 10
        )
        self._pause_srv = self.create_service(
            Trigger, "~/pause", self._on_pause
        )
        self._resume_srv = self.create_service(
            Trigger, "~/resume", self._on_resume
        )
        self._state_pub = self.create_publisher(String, "~/state", 10)
        # SEPARATE topic for the discrete latch. The ~/state vocabulary
        # (PAUSED/WAITING/RAMPING/FOLLOWING) is parsed by other code, so it is
        # left structurally untouched; a consumer that only understands the old
        # tokens keeps working and simply never subscribes here.
        self._discrete_state_pub = self.create_publisher(
            String, "~/discrete_state", 10
        )
        # Created UNCONDITIONALLY, exactly like ~/discrete_state: a consumer must
        # be able to subscribe and discover "DISABLED / no samples" rather than
        # find no topic at all and be unable to tell a continuous bridge from a
        # dead one.
        self._discrete_trigger_pub = self.create_publisher(
            Float32, "~/discrete_trigger", 10
        )
        if self.state_publish_rate_hz > 0.0:
            self.create_timer(
                1.0 / self.state_publish_rate_hz, self._publish_state
            )
        # SETTLE RE-ASSERT timer. Created ONLY when the feature is on, so with
        # settle_reassert_s:=0.0 (or settle_reassert_hz:=0.0) this node owns no
        # extra timer and its wakeup profile is exactly what it was before. The
        # callback itself is a no-op outside the window, so even when enabled the
        # bus sees nothing at rest — load-bearing, the Modbus link is
        # single-client.
        #
        # No explicit callback group: like every other entity here it lands in
        # the node's DEFAULT group, which is MutuallyExclusive. So _settle_tick,
        # _on_width, _on_actual and the two services are serialised against each
        # other by the executor, and _last_pub / _output_changed_time /
        # _reconcile_since / _force_next_publish need no lock. Do NOT move any of
        # these onto a Reentrant group without adding one.
        self._settle_enabled = (
            self.settle_reassert_s > 0.0 and self.settle_reassert_hz > 0.0
        )
        if self._settle_enabled:
            self.create_timer(1.0 / self.settle_reassert_hz, self._settle_tick)
        # Reconcile needs BOTH knobs positive: a zero tolerance would fire on any
        # difference at all (including quantisation of the 0..255 raw position),
        # and a zero dwell would fire on the first sample of a move still in
        # flight. Either at 0.0 means "off", i.e. today's behaviour.
        self._reconcile_enabled = (
            self.reconcile_after_s > 0.0 and self.reconcile_tol > 0.0
        )

        # --- Startup log (state the direction invariant explicitly) ------
        direction = (
            "width 1=OPEN..0=CLOSED (INVERTED!)"
            if self.invert
            else "width 0=OPEN..1=CLOSED -> command_percent 0=open..1=closed"
        )
        self.get_logger().info(
            "gello_gripper_bridge started | "
            f"{direction} (invert={self.invert}) | "
            f"deadband={self.deadband} ema_alpha={self.ema_alpha} "
            f"clamp=[{self.clamp_min},{self.clamp_max}] | "
            f"start_paused={self._paused}"
        )
        if self.invert:
            self.get_logger().warn(
                "invert=True: GELLO-open will command robot-CLOSE — CRUSH HAZARD. "
                "This MUST be False for this hardware."
            )
        # Say ONCE, at startup, which mode is live and with what thresholds.
        if self._discrete_fallback_reason is not None:
            self.get_logger().error(
                "discrete_mode=True REJECTED: "
                f"{self._discrete_fallback_reason}. Required: "
                "0.0 < discrete_open_at < discrete_close_at < 1.0. "
                "FALLING BACK to continuous mode (the value is forwarded "
                "unchanged, exactly as before); ~/discrete_state=DISABLED. "
                "Fix the parameters and relaunch to enable discrete mode."
            )
        if self.discrete_mode:
            self.get_logger().info(
                "gripper mode: DISCRETE (binary) | "
                f"trigger >= {self.discrete_close_at} -> CLOSED(1.0), "
                f"trigger <= {self.discrete_open_at} -> OPEN(0.0), "
                "in between -> hold previous (hysteresis) | "
                "initial latch UNKNOWN: nothing is published until the first "
                "threshold crossing"
            )
            if self.ema_alpha < 1.0:
                self.get_logger().warn(
                    f"ema_alpha={self.ema_alpha} < 1.0 with discrete_mode=True: "
                    "the latch reads the RAW clamped value, so EMA smoothing has "
                    "NO effect on the discrete output (a smoothed 0<->1 step "
                    "would otherwise reach the endpoint only asymptotically)."
                )
            # SILENT-NET WARNING. ~/discrete_state and ~/discrete_trigger are fed
            # ONLY by the state timer, and state_publish_rate_hz:=0.0 is a
            # legitimate way to silence ~/state. Doing so in discrete mode also
            # removes the operator's stuck-latch indicator — the stated safety
            # net for this whole feature — and, before this line, removed it with
            # no log line at all. The mode still runs (the latch does not depend
            # on the timer), so this is a warning, not a fallback.
            if self.state_publish_rate_hz <= 0.0:
                self.get_logger().warn(
                    f"state_publish_rate_hz={self.state_publish_rate_hz} with "
                    "discrete_mode=True: the state timer is DISABLED, so "
                    "~/discrete_state and ~/discrete_trigger will never publish "
                    "and the GUI latch indicator (the safety net for a stuck "
                    "latch) is blind. The in-band dwell warning still fires on "
                    "the log. Set state_publish_rate_hz > 0.0 (default 5.0) to "
                    "restore the indicator."
                )
        else:
            self.get_logger().info(
                "gripper mode: CONTINUOUS (discrete_mode=False) | "
                "leader value forwarded through clamp/EMA/deadband unchanged"
            )
        # Delivery robustness, stated at startup so a session log records whether
        # the 0/6-on-hardware configuration was actually live.
        self.get_logger().info(
            "delivery robustness | settle re-assert: "
            + (
                f"{self.settle_reassert_s:.2f}s @ {self.settle_reassert_hz:.1f}Hz "
                "after each output change (silent at rest)"
                if self._settle_enabled
                else "OFF (publish-once per change, pre-2026-08-06 behaviour)"
            )
            + " | reconcile: "
            + (
                f"re-assert once per {self.reconcile_after_s:.2f}s when the "
                f"measured position sits > {self.reconcile_tol:.3f} CLOSED of "
                "the commanded value (opening direction only — never fights a grasp)"
                if self._reconcile_enabled
                else "OFF"
            )
        )
        self.get_logger().info(
            f"subscribing {self._in_topic} -> publishing {self._out_topic} | "
            f"actual feedback {self._actual_topic} | "
            f"pause/resume: ~/pause ~/resume | "
            f"state: ~/state ~/discrete_state ~/discrete_trigger"
        )

    # ---------------------------------------------------------------------
    def _on_actual(self, msg: Float32) -> None:
        """Record the gripper's actual position (0=open..1=closed) and reconcile.

        Two consumers: ~/resume's zero-jump seed (as before) and
        _maybe_reconcile. The timestamp is new — see ``_actual_time``.

        Reconcile is driven from HERE rather than from a timer of its own on
        purpose: this callback is the only moment new position information
        exists, the driver already publishes it at status_rate_hz (5 Hz, ~10
        samples per the 0.5 s default dwell), and hanging it off the existing
        stream means the feature adds no wakeups and cannot outlive the feedback
        it reasons about. If the driver stops publishing, reconcile simply stops
        — which is correct: with no measurement there is no discrepancy to
        assert, and guessing would be exactly the failure mode this guards.
        """
        self._actual_pos = float(msg.data)
        self._actual_time = time.monotonic()
        self._maybe_reconcile(self._actual_time)

    # ---------------------------------------------------------------------
    def _maybe_reconcile(self, now: float) -> None:
        """Re-assert the last commanded value when the gripper sits short of it.

        WHY THIS EXISTS. The bridge publishes each setpoint exactly once. One
        lost message (the driver's old rate limiter dropped the freshest sample;
        DDS discovery can eat a first publish) leaves the operator's intent
        stranded in this process forever, because the deadband gate compares
        against ``_last_pub`` — what we MEANT to send — and so never notices.
        Measured 2026-08-06: 7 of 14 trigger releases parked short of full open.
        Third-party publishers on the same topic (the task recorder's GO HOME,
        the RL reset, run_hil_preposition.sh) can strand it the same way.

        THE ONE-SIDED DIRECTION IS THE SAFETY ARGUMENT — read this before
        relaxing it. On this axis 0.0 = OPEN and 1.0 = CLOSED, and we act ONLY
        when::

            actual - _last_pub > reconcile_tol      (fingers MORE CLOSED than commanded)

        so the re-assert always commands the gripper to OPEN toward a position it
        is not yet at. The mirrored case — fingers more OPEN than commanded — is
        deliberately IGNORED, because it is indistinguishable from a successful
        grasp: a 2F-85 holding an object stops where the object is, and
        ``grip_pos`` maxes out around 0.506 while gripping (UNITS_REFERENCE
        §5.1). Position alone cannot separate "the close command was lost" from
        "the close command worked and there is a cube in there", so this function
        refuses to guess and never re-commands a closing setpoint.

        HOW IT CANNOT SQUEEZE, AND CANNOT OSCILLATE, in four steps:
          1. It publishes ``_last_pub`` VERBATIM. It never computes a new
             setpoint, so it can never command a value more closed than one the
             operator's own trigger already produced.
          2. It only fires when the fingers are MORE CLOSED than that value, so
             every message it sends is an OPENING command relative to the
             measured state. Grip force can only fall, never rise.
          3. A grasp therefore lands in the ignored direction (actual < command)
             and the guard in (2) returns before anything is published — the one
             hardware case the spec forbids fighting.
          4. The dwell must be CONTINUOUS (any non-qualifying sample clears
             ``_reconcile_since``) and firing RESETS the clock, so re-asserts are
             spaced at least ``reconcile_after_s`` apart. Since each one is
             idempotent and monotone toward the same target, repeating it cannot
             produce a cycle — the output value is unchanged, only repeated.

        Gated to FOLLOWING and unpaused. PAUSED means "not streaming is the safe
        state" (a collapsed leader must not drive the gripper) and reconcile is
        streaming, so it is silent there. During the resume ramp ``_last_pub`` is
        an intermediate slew sample the gripper is still chasing, so a mismatch
        is EXPECTED rather than evidence of a lost message. A stale leader means
        the operator's intent is itself unknown, and re-asserting an unknown
        intent is not defence in depth, it is a guess.
        """
        if not self._reconcile_enabled:
            return
        # Any condition that is not "a live, settled, following bridge with a
        # measured discrepancy" clears the dwell: the timer must measure ONE
        # continuous episode, not the union of several unrelated ones.
        if (
            self._paused
            or self._last_pub is None
            or self._actual_pos is None
            or self._actual_time is None
            # Inside the resume ramp the output is deliberately mid-travel.
            or now < self._ramp_until
            # Stale leader => intent unknown (same gate ~/state calls WAITING).
            or self._last_rx_time is None
            or (now - self._last_rx_time) > self.staleness_timeout_s
            # Stale measurement => cannot be trusted to describe the present.
            # Always false when called from _on_actual; kept so the function is
            # correct for any future caller and so the rule is stated in code.
            or (now - self._actual_time) > self.staleness_timeout_s
            # THE DIRECTIONAL GATE. See the docstring — one-sided ON PURPOSE.
            or (self._actual_pos - self._last_pub) <= self.reconcile_tol
        ):
            self._reconcile_since = None
            self._reconcile_warn_time = None
            return
        if self._reconcile_since is None:
            self._reconcile_since = now
            return
        if (now - self._reconcile_since) < self.reconcile_after_s:
            return
        # FIRE. Restarting the clock (rather than clearing it) is the rate limit:
        # a discrepancy that persists is re-asserted at most once every
        # reconcile_after_s, forever, without ever escalating.
        self._reconcile_since = now
        # One message is enough even against the driver's rate limiter: it can
        # only drop a sample that arrives within 1/command_rate_hz (50 ms) of the
        # previous accepted one, and by construction this one is at least
        # reconcile_after_s (0.5 s) after any other traffic from us.
        out = Float32()
        out.data = float(self._last_pub)
        self._pub.publish(out)
        # Deliberately does NOT touch _last_pub / _last_pub_time /
        # _output_changed_time: the commanded value has not changed, the ramp
        # budget base has not moved, and re-opening the settle window here would
        # turn a single bounded probe into a repeating burst.
        #
        # THROTTLE THE LOG, NOT THE RE-ASSERT. When the gripper physically
        # CANNOT reach the command — jammed fingers, or a driver whose own
        # command deadband swallows the duplicate because the message was never
        # actually lost — the discrepancy is permanent and this fires once per
        # reconcile_after_s forever (measured: 99 fires/minute at the defaults).
        # The re-assert itself is cheap and must keep going (it costs no bus
        # traffic: the driver deadbands an unchanged setpoint before its rate
        # limiter). An unthrottled WARN at ~1.7 Hz, by contrast, buries every
        # other line in the operator's console for the rest of the session.
        # Same period and same reasoning as _track_band_dwell; the first fire of
        # each episode is always logged because _reconcile_warn_time is cleared
        # together with _reconcile_since.
        if (
            self._reconcile_warn_time is None
            or (now - self._reconcile_warn_time) >= RECONCILE_WARN_PERIOD_S
        ):
            self._reconcile_warn_time = now
            self.get_logger().warn(
                f"reconcile: gripper measured at {self._actual_pos:.3f} but last "
                f"commanded {self._last_pub:.3f} (0=open..1=closed) for "
                f">{self.reconcile_after_s:.2f}s — re-asserting the command. This "
                "means a command_percent message was lost or a third party moved "
                "the gripper; the re-assert only ever commands OPENING, so it "
                "cannot tighten a grasp. (Re-asserts continue every "
                f"{self.reconcile_after_s:.2f}s while this persists; this WARN is "
                f"throttled to one per {RECONCILE_WARN_PERIOD_S:.0f}s.)"
            )

    # ---------------------------------------------------------------------
    def _settle_tick(self) -> None:
        """Re-publish the current output for settle_reassert_s after it changed.

        The whole feature is this: a setpoint published once can be lost, and the
        operator only notices at the end of a gesture (a release that parks the
        gripper half open). Sending the same number a few more times while the
        gesture is still settling costs a handful of messages per grasp and made
        the measured failure rate 7/14 -> 0/6 on hardware.

        Three properties matter more than the mechanism:

        * **It stops.** The window is anchored to ``_output_changed_time``, which
          only a genuine value CHANGE advances, and re-asserts do not advance it.
          So a converged output falls out of the window and the node returns to
          publishing nothing at all. The Modbus bus is single-client; steady-state
          traffic is not acceptable and there is none.
        * **It is position-blind.** It re-asserts what the LEADER asked for and
          never looks at ``_actual_pos``, so it cannot be provoked by fingers
          stopping short on an object — a grasp simply holds, and the same
          already-commanded value is repeated for at most one window.
        * **It is silent while paused and while the leader is stale.** Paused
          means the output is intentionally silenced (drop hazard); a stale
          leader means we would be re-asserting an intent nobody is expressing.

        Runs in discrete mode too, on purpose: discrete publishes even LESS often
        (one message per latch transition), so a lost message there strands the
        gripper for an entire grasp cycle rather than a fraction of a stroke.
        """
        if not self._settle_enabled or self._paused:
            return
        if self._last_pub is None or self._output_changed_time is None:
            return
        now = time.monotonic()
        if (now - self._output_changed_time) >= self.settle_reassert_s:
            return  # window closed — at rest this node publishes NOTHING
        if (
            self._last_rx_time is None
            or (now - self._last_rx_time) > self.staleness_timeout_s
        ):
            return  # stale leader: no live intent to re-assert
        out = Float32()
        out.data = float(self._last_pub)
        self._pub.publish(out)

    # ---------------------------------------------------------------------
    def _publish_output(self, value: float, now: float) -> None:
        """Publish ``value`` and update the deadband / ramp / settle references.

        The single place the command topic is written with a NEW value (the two
        re-assert paths republish ``_last_pub`` and update nothing). Splitting it
        out keeps ``_output_changed_time`` impossible to forget: the settle
        window opens exactly when the number on the wire actually changes, and an
        unchanged republish — the converged tail of a resume ramp — does not
        extend it.
        """
        if self._last_pub is None or value != self._last_pub:
            self._output_changed_time = now
        out = Float32()
        out.data = float(value)
        self._pub.publish(out)
        self._last_pub = value
        self._last_pub_time = now

    # ---------------------------------------------------------------------
    def _on_width(self, msg: Float32) -> None:
        """Map a GELLO width sample to command_percent and publish (deadbanded)."""
        p_raw = float(msg.data)
        now = time.monotonic()
        # ALWAYS record the freshest leader sample — even while paused — so a later
        # ~/resume can gate on its age. This MUST precede the pause early-return.
        self._last_rx_time = now
        self._last_rx_value = p_raw

        # invert + clamp are PURE functions of the raw sample (no state), so they
        # can be evaluated before the pause gate. The result is recorded for
        # ~/discrete_trigger as TELEMETRY ONLY — it is not fed to the latch, the
        # EMA or the output while paused, so the "flopping leader must not
        # re-decide intent" property below is untouched; it only lets a GUI show
        # a LIVE trigger reading next to a deliberately frozen latch.
        p = p_raw
        if self.invert:
            p = 1.0 - p
        p = min(max(p, self.clamp_min), self.clamp_max)
        self._trigger_clamped = p

        # PAUSED: silence output entirely. The Robotiq holds its last commanded
        # position onboard; not-streaming is the safe state (drop hazard mitigated).
        # Return BEFORE the EMA/discrete/ramp/deadband pipeline — so the discrete
        # latch also freezes here, which is right: while paused a flopping leader
        # must not be able to re-decide the operator's intent. (It is re-decided
        # from the first sample AFTER an accepted resume instead — _on_resume.)
        if self._paused:
            return

        # EMA (ema_alpha=1.0 => identity, no lag).
        if self._f is None:
            self._f = p
        else:
            self._f = (1.0 - self.ema_alpha) * self._f + self.ema_alpha * p

        # DISCRETE LATCH (opt-in). Placed AFTER the EMA and BEFORE the resume
        # ramp, so it is downstream of invert/clamp and upstream of BOTH the ramp
        # and the deadband gate. With discrete_mode=False this block is inert.
        #
        # EMA INTERACTION — the latch reads the RAW clamped `p`, not `self._f`.
        # ema_alpha defaults to 1.0 (identity) so the two are normally the same
        # value, but the config yaml invites lowering it if the gripper chatters,
        # and an alpha < 1 exponentially smooths a hard 0<->1 step: the filtered
        # value approaches the endpoint asymptotically and can dwell inside the
        # 0.3..0.7 hysteresis band for many samples, so a deliberate hand-open
        # would land LATE (or, for a small alpha and a brief squeeze, never cross
        # at all). Thresholding the raw value keeps the mode honest: the latch is
        # a decision about operator INTENT, and the evidence (§5.1) says that
        # intent is already binary in the raw signal — smoothing before
        # thresholding can only add lag. We still overwrite self._f with the
        # endpoint so that everything downstream (ramp target, deadband
        # reference, published value) agrees on the latched value, which also
        # makes the EMA a genuine no-op in discrete mode rather than leaving a
        # stale filter state behind. The alternative — refusing to run unless
        # ema_alpha == 1.0 — was rejected: it would let one gripper parameter
        # veto the mode, when simply not consulting the filter is both simpler
        # and strictly more correct. A warning is logged at startup instead.
        if self.discrete_mode:
            self._discrete_latch = discrete_latch(
                p,
                self._discrete_latch,
                self.discrete_open_at,
                self.discrete_close_at,
            )
            self._track_band_dwell(p, now)
            if self._discrete_latch is None:
                # UNKNOWN: no threshold has been crossed yet. Publish NOTHING —
                # hold whatever the gripper already has rather than guessing an
                # endpoint (see _discrete_latch init). _last_pub is deliberately
                # left untouched so the deadband reference stays correct.
                #
                # _last_pub_time IS re-based, and that is load-bearing: it is the
                # base of the ramp's `elapsed`, and slew budget must accrue only
                # over time in which the output was actually free to move. A
                # suppressed sample is time in which it was NOT. Without this
                # line, a latch that stayed UNKNOWN for 1.9 s of a 2.0 s ramp
                # banked max_delta = 0.6 * 1.9 = 1.14 > 1.0 stroke, so the first
                # published sample was the FULL endpoint and the ramp degenerated
                # into precisely the single-step jump it exists to prevent.
                self._last_pub_time = now
                return
            self._f = self._discrete_latch

        # RESUME RAMP: for resume_ramp_s after a resume, slew-limit the published
        # value from the seeded actual position toward the leader target. The
        # deadband gate is DELIBERATELY BYPASSED here: during the ramp we must
        # re-assert an advancing output on every sample to crawl monotonically from
        # the actual position to the leader value; if a deadband skip suppressed a
        # sub-threshold step the output would stall mid-ramp and never converge.
        #
        # DISCRETE INTERACTION — discrete mode HONOURS the ramp; it does not
        # short-circuit it. The ramp exists to stop the output jumping from the
        # gripper's ACTUAL position to a far-away leader value the instant the
        # operator resumes, and that hazard is not merely unchanged under
        # discretisation, it is MAXIMAL: a latched target is by construction a
        # full-stroke extreme, so the potential jump is the whole 0..1 stroke.
        # The cost is bounded and rare — at most resume_ramp_s (2.0 s), only
        # after an explicit operator ~/resume. Because the snap happens ABOVE
        # this branch, the value being crawled toward is already the latched
        # endpoint, so the ramp converges to 0.0 or 1.0 and never rests at a mid
        # value; the intermediate published values are transient slew, and BOTH
        # ~/state and ~/discrete_state report RAMPING for the whole window, so
        # the operator's indicator never shows a settled endpoint while the
        # output is mid-travel. At 0.6/s a full stroke needs ~1.67 s, which fits
        # inside the 2.0 s window, so convergence completes within the ramp.
        if now < self._ramp_until:
            prev = self._last_pub if self._last_pub is not None else self._f
            # `elapsed` is the slew BUDGET, so it must measure only time in which
            # the output was free to move, and it must stay bounded:
            #   * every suppressing path re-bases _last_pub_time (the UNKNOWN
            #     return above), so a suppressed sample banks nothing;
            #   * the cap stops a leader stream that DIES mid-ramp from banking a
            #     full-stroke step in its silence. staleness_timeout_s is the
            #     right cap because resume itself refuses any stream older than
            #     that, so for every stream this node will resume on, the cap is
            #     inert (at 30 Hz, elapsed ~0.033 s vs a 0.5 s cap) and it bites
            #     only where the contract is already violated.
            elapsed = min(max(now - self._last_pub_time, 0.0), self.staleness_timeout_s)
            max_delta = self.resume_slew_per_s * elapsed
            delta = self._f - prev
            if delta > max_delta:
                out_val = prev + max_delta
            elif delta < -max_delta:
                out_val = prev - max_delta
            else:
                out_val = self._f
            # The ramp publishes unconditionally, so an accepted resume's
            # force-publish debt is settled here too.
            self._force_next_publish = False
            self._publish_output(out_val, now)
            return

        # DEADBAND gate: skip tiny at-rest changes (normal following).
        #
        # FORCE, consumed here. _on_resume seeds _last_pub from the gripper's
        # ACTUAL position without publishing it; if the live leader already sits
        # within `deadband` of that seed, this gate suppresses every sample and
        # the gripper receives NOTHING for the rest of the session (silently —
        # ~/state says FOLLOWING throughout). One forced publish after an
        # accepted resume closes that hole. It is consumed at the PUBLISH
        # DECISION, not at the top of the callback, so the discrete-UNKNOWN
        # return above leaves it pending: force exists to defeat the deadband,
        # never the latch's "publish nothing until intent is re-earned" rule.
        force = self._force_next_publish
        self._force_next_publish = False
        if (
            not force
            and self._last_pub is not None
            and abs(self._f - self._last_pub) < self.deadband
        ):
            return
        self._publish_output(self._f, now)

    # ---------------------------------------------------------------------
    def _on_pause(self, request, response):
        """UNCONDITIONAL pause: silence output, always succeed (safe state)."""
        self._paused = True
        response.success = True
        response.message = (
            "paused: gripper output silenced; Robotiq holds last position"
        )
        self.get_logger().info("~/pause: paused (output silenced)")
        return response

    # ---------------------------------------------------------------------
    def _on_resume(self, request, response):
        """FAIL-CLOSED resume: refuse (staying paused + silent) unless a fresh
        leader sample AND an actual gripper position exist. On acceptance, seed at
        the actual position (zero jump), reset the discrete latch to UNKNOWN,
        ARM the force-publish flag and start the slew ramp. Never publishes
        itself — the next _on_width emits the seeded/slew-limited value.

        NOTE the seed below uses ``_actual_pos`` WITHOUT a freshness check, while
        _maybe_reconcile requires a fresh one. That is deliberate and not an
        oversight: a stale position is still the best available zero-jump seed
        (and a missing position stream means the driver is down, so nothing will
        move anyway), whereas reconcile ACTS on the value and must not act on a
        measurement that may predate a move. Tightening this gate would newly
        refuse resumes that work today, for no safety gain."""
        if not self._paused:
            response.success = True
            response.message = "already following (not paused)"
            # No force-publish here. This path writes no seed, so there is no
            # phantom to flush; the flag exists purely to undo the seed's
            # interaction with the deadband gate.
            return response

        now = time.monotonic()
        # Gate (a): fresh leader sample.
        if (
            self._last_rx_time is None
            or (now - self._last_rx_time) > self.staleness_timeout_s
        ):
            age = (
                "never"
                if self._last_rx_time is None
                else f"{now - self._last_rx_time:.2f}s"
            )
            response.success = False
            response.message = (
                f"REFUSED: stale leader (age={age} > "
                f"{self.staleness_timeout_s}s); staying paused & silent"
            )
            self.get_logger().warn(response.message)
            return response

        # Gate (b): actual gripper position known — prefer the measured feedback,
        # fall back to this bridge's own last-published value, else refuse.
        seed = self._actual_pos if self._actual_pos is not None else self._last_pub
        if seed is None:
            response.success = False
            response.message = (
                "REFUSED: no actual gripper position (no position_percent feedback "
                "and nothing published yet); staying paused & silent"
            )
            self.get_logger().warn(response.message)
            return response

        # ACCEPT: seed from the actual position for a zero-jump first output, then
        # ramp. Do NOT publish from here — the next _on_width emits the seeded/
        # slew-limited value.
        #
        # RESET THE DISCRETE LATCH TO UNKNOWN. This is the seed group's third
        # member: _f and _last_pub are re-seeded from where the gripper ACTUALLY
        # is, and the latch — the target they ramp toward — must not be a memory
        # from before the pause, because the world moves during a pause. The task
        # recorder does it once per take: GO HOME pauses BOTH bridges and then
        # force-opens the Robotiq by publishing 0.0 straight to
        # ~/command_percent, bypassing this (paused) bridge. Resuming on a frozen
        # CLOSED latch then ramps a freshly-opened gripper to FULL stroke over
        # ~1.67 s for any trigger position that is not already <= open_at.
        # Continuous mode cannot do this: the same hand position there yields the
        # trigger value itself (0.45 -> 45% closed), whereas discretising it
        # yields 100% closed at full force over a much wider range of hands.
        #
        # Cleared HERE and not in _on_pause, for four reasons:
        #   1. The hazard is not that a stale latch EXISTS, it is that a stale
        #      latch gets USED as a ramp target. That happens only here.
        #   2. While paused, the latch is genuinely informative — it is the intent
        #      that commanded the position the Robotiq is now holding, and
        #      _publish_state documents reporting it. Clearing at pause would make
        #      the GUI say UNKNOWN over a gripper visibly clamped on an object.
        #   3. A resume that is REFUSED (stale leader) must change nothing; if
        #      pause had cleared the latch, that record would already be gone.
        #   4. It keeps "a flopping leader must not re-decide intent while paused"
        #      exactly as it was: intent is still FROZEN for the whole pause (the
        #      width callback returns before the latch), and is re-decided from
        #      the first post-resume sample. Freezing and re-deciding are
        #      compatible — the reset is what makes the re-decision honest instead
        #      of a replay.
        #
        # Each case on the FIRST post-resume sample (verified in
        # test_gripper_discrete.py):
        #   trigger >= close_at  -> re-latches CLOSED immediately, ramps toward
        #      1.0. IDENTICAL to the old behaviour, and honest: the operator is
        #      squeezing the trigger right now.
        #   open_at < trigger < close_at -> latch stays None, so the UNKNOWN
        #      return publishes NOTHING and the Robotiq holds exactly what it has.
        #      Strictly better than the old jump to the frozen endpoint.
        #   trigger <= open_at   -> latches OPEN (0.0); after a GO HOME the seed
        #      is already ~0.0, so delta is 0 and the gripper simply stays open.
        #   => after a GO HOME, every trigger position except a live squeeze past
        #      close_at leaves the gripper OPEN, which is what the operator wants.
        # The ramp is NOT stalled by the reset: an UNKNOWN latch means there is no
        # target to ramp toward, and holding is the correct answer to that.
        self._discrete_latch = None
        # The in-band dwell alarm measures CONTINUOUS dwell while the pipeline is
        # actually forwarding, and _track_band_dwell is not reached while paused —
        # so a pre-pause entry time would report the pause itself as dwell. Restart
        # the measurement (and its throttle) with the pipeline.
        self._band_since = None
        self._band_warn_time = None
        self._f = seed
        self._last_pub = seed
        self._last_pub_time = now
        # FORCE ONE PUBLISH. _last_pub above is a PHANTOM: it claims a value was
        # published when none was. The deadband gate reads it, so a leader
        # already resting within `deadband` of the seed produced NO output ever
        # again — the bridge reported FOLLOWING while the gripper sat unattended.
        # With resume_ramp_s > 0 the ramp branch happens to publish anyway, so
        # this was invisible at the default config and fatal at
        # resume_ramp_s:=0.0; the flag makes it correct in both.
        self._force_next_publish = True
        # CLOSE THE SETTLE WINDOW. _output_changed_time is otherwise untouched by
        # pause and by resume, so a window opened by a PRE-PAUSE output change is
        # still open here — and _last_pub is now the PHANTOM SEED, a value taken
        # from _actual_pos that was never published and that the operator's live
        # trigger did not ask for. _settle_tick would then re-assert it.
        #
        # Reachable and NOT harmless (reproduced offline): discrete mode, the
        # driver's position stream drops so _actual_pos goes stale at 0.9 (the
        # seed deliberately has no freshness check — see this docstring), GO HOME
        # pauses the bridge and force-opens the Robotiq to 0.0, the operator
        # resumes with the trigger resting inside the hysteresis band. The latch
        # is UNKNOWN so _on_width publishes NOTHING — exactly as promised three
        # paragraphs below and in the response message — yet the settle timer
        # published the stale 0.9 twelve times over 0.6 s and CLOSED the
        # freshly-opened gripper to 90%. That is the one path by which a
        # re-assert can issue a CLOSING command the live trigger never
        # requested, which is precisely what _maybe_reconcile's one-sided rule
        # exists to exclude. Continuous mode escapes it only by accident (the
        # force-publish overwrites _last_pub with the live leader value first).
        #
        # A new window must be EARNED by a genuine post-resume output change:
        # _publish_output re-opens it on the first ramp/forced sample whose value
        # actually differs, which is also the first sample worth protecting.
        self._output_changed_time = None
        # Nothing is owed to reconcile from before the pause: the world moved,
        # and the discrepancy (if any) must re-accrue against the NEW seed.
        self._reconcile_since = None
        self._reconcile_warn_time = None
        self._ramp_until = now + self.resume_ramp_s
        self._paused = False
        response.success = True
        response.message = (
            f"resumed: seeded at actual={seed:.3f}, ramping to leader over "
            f"{self.resume_ramp_s:.1f}s (slew {self.resume_slew_per_s:.2f}/s)"
            + (
                "; discrete latch reset to UNKNOWN (nothing is published until "
                "the live trigger crosses a threshold)"
                if self.discrete_mode
                else ""
            )
        )
        self.get_logger().info(response.message)
        return response

    # ---------------------------------------------------------------------
    def _track_band_dwell(self, value: float, now: float) -> None:
        """Warn (THROTTLED) when the trigger comes to REST strictly inside the band.

        A mis-set threshold fails silently and the latch alone cannot reveal it:
        with ``discrete_open_at`` below a stuck resting value, a trigger parked at
        0.243 never crosses anything, the latch stays CLOSED, and that reads as a
        perfectly normal "the operator is squeezing". Dwell time is what separates
        the two. A human crossing the band passes THROUGH it in a fraction of a
        second — the §5.1 corpus has 0 resting plateaus inside 0.3..0.7 across
        52,607 samples — so more than DISCRETE_BAND_DWELL_WARN_S of CONTINUOUS
        in-band dwell means the trigger has settled where no threshold will ever
        fire, and the gripper has stopped responding.

        Strict inequality on both sides: a value exactly ON a threshold has
        latched (the latch is ``<=`` / ``>=``), so it is not in the dead zone.
        Only reached from the discrete branch of _on_width, i.e. never while
        paused and never in continuous mode.
        """
        if not (self.discrete_open_at < value < self.discrete_close_at):
            self._band_since = None
            return
        if self._band_since is None:
            self._band_since = now
            return
        dwell = now - self._band_since
        if dwell < DISCRETE_BAND_DWELL_WARN_S:
            return
        if (
            self._band_warn_time is not None
            and (now - self._band_warn_time) < DISCRETE_BAND_WARN_PERIOD_S
        ):
            return
        self._band_warn_time = now
        latched = (
            "UNKNOWN (nothing has been published yet)"
            if self._discrete_latch is None
            else ("CLOSED (1.0)" if self._discrete_latch >= 0.5 else "OPEN (0.0)")
        )
        self.get_logger().warn(
            f"discrete gripper trigger has RESTED inside the hysteresis band for "
            f"{dwell:.1f}s: value={value:.3f} is strictly between "
            f"discrete_open_at={self.discrete_open_at} and "
            f"discrete_close_at={self.discrete_close_at}, so NO threshold can "
            f"fire and the output is frozen at latch={latched} — the gripper "
            f"will not respond. Either release/squeeze the trigger fully, or "
            f"re-measure the thresholds: raising discrete_open_at above "
            f"{value:.3f} would make this rest position latch OPEN."
        )

    # ---------------------------------------------------------------------
    def discrete_state_token(self, now: float | None = None) -> str:
        """Return the ~/discrete_state token for the current latch.

        DISABLED  — continuous mode (either not requested, or requested with
                    invalid thresholds and fallen back); the latch is not in use.
        RAMPING   — discrete mode, inside the post-resume slew window: the value
                    on the wire is an intermediate slew sample, NOT the latch, so
                    no endpoint may be advertised as settled yet.
        UNKNOWN   — discrete mode, no threshold crossed yet, nothing published.
        OPEN      — latched at 0.0.
        CLOSED    — latched at 1.0.

        Precedence is DISABLED > RAMPING > UNKNOWN > OPEN/CLOSED. RAMPING sits
        above UNKNOWN because an accepted resume clears the latch, so UNKNOWN is
        the state of EVERY ramp until the trigger crosses — reporting it would
        hide the ramp completely.
        """
        if not self.discrete_mode:
            return DISCRETE_STATE_DISABLED
        if (time.monotonic() if now is None else now) < self._ramp_until:
            return DISCRETE_STATE_RAMPING
        if self._discrete_latch is None:
            return DISCRETE_STATE_UNKNOWN
        return (
            DISCRETE_STATE_CLOSED
            if self._discrete_latch >= 0.5
            else DISCRETE_STATE_OPEN
        )

    # ---------------------------------------------------------------------
    def _publish_state(self) -> None:
        """Publish PAUSED / WAITING / RAMPING / FOLLOWING (in precedence order)
        on ~/state, the discrete latch on ~/discrete_state, and the live clamped
        trigger on ~/discrete_trigger.

        The ~/state computation and vocabulary are UNCHANGED — other code parses
        those four tokens. The discrete information rides on its own topics at the
        same rate rather than extending or restructuring this one. Note the latch
        is reported even while PAUSED: it is the last resolved operator intent,
        and the Robotiq is holding the position that intent commanded. The trigger
        keeps updating while paused too — that is the point of it being a separate
        reading from the latch.
        """
        now = time.monotonic()
        if self._paused:
            s = "PAUSED"
        elif (
            self._last_rx_time is None
            or (now - self._last_rx_time) > self.staleness_timeout_s
        ):
            s = "WAITING"
        elif now < self._ramp_until:
            s = "RAMPING"
        else:
            s = "FOLLOWING"
        msg = String()
        msg.data = s
        self._state_pub.publish(msg)
        dmsg = String()
        dmsg.data = self.discrete_state_token(now)
        self._discrete_state_pub.publish(dmsg)
        # Nothing is published on ~/discrete_trigger until the first leader sample
        # arrives: there is no honest value to report, and a placeholder (0.0 =
        # "fully open trigger") would be a lie in the safety-relevant direction.
        if self._trigger_clamped is not None:
            tmsg = Float32()
            tmsg.data = float(self._trigger_clamped)
            self._discrete_trigger_pub.publish(tmsg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GelloGripperBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
