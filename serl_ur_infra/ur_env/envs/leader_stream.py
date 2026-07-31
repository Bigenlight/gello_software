"""Leader resampling (One-Euro) + per-window displacement budget for HIL.

Pure numpy/stdlib on purpose: no rclpy, no gym, no config import, so the code
that is unit-tested IS the code the actor runs (same rule as
``ros2_ur_ws/src/ur_gello_bringup/ur_gello_bringup/bridge_stages.py:31-33``).

WHERE THIS FILE SITS IN THE MECHANISM.  It is the pure-maths piece — a leader
filter and a per-window displacement budget, no clock and no robot.  Its only
caller is ``GelloIntervention`` (``ur_env/envs/wrappers.py``: both objects are
built in ``__init__:353`` and spent in ``charge:607`` / ``substep:618``), which
is in turn driven by ``UR7eEnv._drive_intervention_substeps``
(``ur_env/envs/ur7e_env.py:553``).  The ordered call chain of the whole
intervention-substep path is written out once, in the ``wrappers.py`` module
docstring.

WHY THIS MODULE EXISTS — THE MEASURED STIFFNESS
----------------------------------------------
Operators report that the arm feels "빳빳" (stiff / notchy) while they drive it
by GELLO during an intervention, and the cause is not the filter or the gains:
it is the *target update cadence*.  ``UR7eEnv.step`` applied the action
**once** — a single ``_apply_action`` — and then slept out the rest of the
control period without ever re-reading the leader.  That branch still exists and
is now the POLICY path only (``ur_env/envs/ur7e_env.py:670-678``); intervened
steps take the substep branch instead.  The 250 Hz command stream underneath is
acceleration limited by design (``ur_env/envs/ros_backend.py:140``
``AccelerationLimitedJointStream``, ``ur_env/envs/config.py:117-124``
``max_accel_rad_s2 = 8.0``), and an acceleration-limited stream chasing ONE
fixed goal per window necessarily runs "accelerate -> arrive -> brake -> **sit
still**" once per window.

Measured directly on the upsampler with a constant 0.15 rad/s leader:

    joint-target updates per 100 ms | time fully stopped | ripple
    1 (one target per window)       | 16% nominal, and   | 2.27
                                    | 40% at the real    |
                                    | 0.197 s step period|
                                    | (gRPC p99 included)|
    3 (30 Hz resampling)            | 0%                 | 0.85

Two conclusions from that table, both load-bearing:

* Raising the target cadence to ~30 Hz removes the dead time entirely.  30 Hz
  is not a guess — it is the cadence of the PROVEN EEF teleop stack, which
  publishes the leader at 30 Hz (``config/ur7e_gello.yaml:26``), One-Euro
  filters it (``ur7e_gello.yaml:40``) and upsamples to 250 Hz
  (``ur7e_gello.yaml:120``).  That stack is the reference "good hand feel".
* **Do not touch the acceleration limit.**  Removing it measures *worse*
  (76% stopped): the stream then bangs between the per-tick step ceiling and
  zero.  The limiter is not the disease.  Two neighbouring knobs measure the
  same way; all three are pinned with the numbers in ``config.py`` next to
  ``UPSAMPLER``.

Everything in the table is OFFLINE measurement — it comes from driving
``AccelerationLimitedJointStream`` directly, with no robot.  The resulting
30 Hz path was afterwards confirmed on hardware: ARMED on the real UR7e
2026-07-30 (``tests/run_real_hil.py --arm --scale 1.0``, 120 intervened steps,
every check PASS, operator confirmed the hand feel).  Both sets of numbers, and
the method for reproducing the offline ones, are in
``docs/testing/04_HIL_INTERVENTION.md`` §9.

ONE FILTER, TWO CADENCES (the API split is a correctness requirement)
--------------------------------------------------------------------
:class:`OneEuro` is a behaviour-preserving port of ``bridge_stages.py:42-106``
(that module is stdlib-only, so the port is a copy of the arithmetic, and
``tests/test_leader_stream.py`` asserts bit-identical output against the
original).  Its one sharp edge, quoted from the original docstring
(``bridge_stages.py:56-59``): the speed estimate must be updated at the *leader
sample* cadence and NOT on every output tick, otherwise

    "each 30 Hz sample step would look like a much faster 250 Hz jump and the
     filter would open up in visible pulses"

The reason is a zero-order hold: between two 30 Hz leader samples the input is
constant, so a per-tick speed estimate sees (n-1) zeros and one 8x-too-large
jump instead of one correct velocity.  The adaptive cutoff then pulses, and so
does the output.  The live bridge honours this by calling ``update_input`` only
from the GELLO subscriber callback (``gello_ur_bridge_node.py:924-935``,
comments at ``:1174`` and ``:1867-1870``) and ``__call__`` only from the publish
timer.  Here that discipline is *enforced by the API* rather than left to the
caller's memory: :meth:`LeaderFilter.note_sample` is the only entry point that
advances the speed estimate, :meth:`LeaderFilter.filtered` is the only one that
advances the output.  Mixing them is then a visible mistake at the call site.

THE OUTPUT TICK IS NOT PERIODIC (why ``filtered`` takes a ``dt``)
-----------------------------------------------------------------
The bridge calls ``OneEuro.__call__`` from a **250 Hz fixed timer**, so there
``dt = 1/250`` is a fact and hard-coding it at construction is correct.  On this
path it is not: the "output tick" is an intervention SUBSTEP, and substeps are
paced inside a control window whose length is whatever the actor's RPC and the
learner's GPU contention make it.

The substep loop (``UR7eEnv._drive_intervention_substeps``) is paced against the
**nominal** window end — ``UR7eEnv.step`` hands it ``start_time + 1.0/self.hz``,
not the step's real duration — with two consequences that have to be read
together:

* the substep count does **not** vary with the real step period.  At HZ=10 and
  30 Hz substeps it is exactly **2**, every window, whether the step took 100 ms
  or 1.12 s.
* everything by which the step overruns that nominal window — the actor's gRPC
  round trip, the camera read in ``_get_obs`` — therefore lands OUTSIDE the
  loop, in one lump between the last substep of a window and the first substep
  of the next.

So the ``filtered`` tick spacing is neither the constructed 1/30 s nor uniform.
It ALTERNATES::

    1/30 s ,  T - 1/30 s ,  1/30 s ,  T - 1/30 s , ...

where **T is the REAL step period**.  ``T - 1/30`` and NOT ``1/HZ - 1/30``:
those two coincide only when a step lands exactly on its nominal period, which
is the one case this rig does not run.  Measured on the production actor loop:
mean **1.99 Hz, i.e. T = 502 ms**, worst step 854 ms; at the 1.12 s
learner-contended step the long interval is **1.087 s**, not the 66.7 ms a
nominal window would suggest — a 16x difference in exactly the quantity the
filter divides by.

Carrying the construction period into that call made the filter's time constant
be counted in WINDOWS instead of seconds, so a heavier system smoothed harder
(measured, at rest, ``alpha = 0.1732``):

    window T | MEAN ``filtered`` spacing | dt the filter believed | over-smooth
    0.100 s  | 50.0 ms                   | 33.3 ms                | 1.5x
    0.250 s  | 125.0 ms                  | 33.3 ms                | 3.8x
    1.120 s  | 560.0 ms                  | 33.3 ms                | 16.8x

The middle column is the MEAN of the alternating pair above (``T/2``); no tick is
ever actually spaced that way, and the real pattern is the harsher of the two —
see "WHAT THE dt FIX DOES NOT FIX".  Driven through the real pattern, the last
row is a **2.95 s** effective at-rest time constant on a filter tuned for
0.159 s (18.5x): exactly when the operator most needs the arm to answer (the
machine is already struggling) it answers slowest.  So ``dt`` is now the time
that ACTUALLY elapsed since the previous output tick, measured by the caller
(``GelloIntervention.substep``), and ``alpha = 1/(1 + tau/dt)`` recovers its
wall-clock meaning.  ``dt=None`` keeps the fixed-period behaviour bit for bit,
which is what the bridge-parity tests assert and what any fixed-rate caller
should keep passing.

WHAT THE ``dt`` FIX DOES NOT FIX (it is necessary, not sufficient)
-----------------------------------------------------------------
Passing the honest interval removes the WINDOW-COUNTING error.  It does not make
a 1.12 s window feel like a 0.1 s one, and nothing in this module can.  Do not
read the section above as "the 1.12 s window is fixed":

* **The tick rate is itself below the filter's Nyquist rate.**  Two output ticks
  per window is 1.79 Hz at T = 1.12 s, against a ``min_cutoff`` of 1.0 Hz, which
  needs 2 Hz to be resolved at all.  Driving the real alternating pattern, the
  at-rest effective time constant is 0.185 s at T = 0.100 s, 0.321 s at the
  measured T = 0.502 s and still **0.498 s (3.13x the 0.159 s the gains ask
  for)** at T = 1.12 s — against 0.263 s / 1.32 s / 2.95 s (1.65x / 8.29x /
  18.5x) on those same three windows with the bug.  That residual is a
  DISCRETISATION FLOOR, not a
  defect: it grows like ``T/ln T`` instead of like ``T`` (8.6x at T = 5 s), so
  the fix turns an error proportional to the window into a sublinear one and
  stops there.  ``tests/test_leader_stream.py``
  ``::test_the_time_constant_is_wall_clock_not_a_window_count`` drives that
  alternating pattern and pins both ends.
* **The backend brakes between windows no matter how good the estimate is.**
  A window's last target is issued at ``start + 2/30`` and the next at
  ``start + T``, so the target-update gap is ``T - 0.0667``.  That crosses the
  upsampler's ``target_stale_s = 0.30`` (``config.py`` ``UPSAMPLER``) as soon as
  ``T > 0.367 s``, i.e. on EVERY window at the measured T = 0.502 s (27.7% of
  the time spent in HOLD, and the soft-start re-arm that follows each stale
  brake pins the upsampler's slew ceiling at 46%); at T = 1.12 s roughly 0.75 s
  of every 1.12 s window is a HOLD.  A smoother leader estimate cannot refresh a
  target the loop is not there to send.
* So the honest ``dt`` is a NECESSARY condition for good hand feel at a long
  window, not a sufficient one.  The root fix is to shorten T itself, or to take
  intervention following off the RL window altogether —
  ``docs/testing/04_HIL_INTERVENTION.md`` §9.5 and ``docs/testing/08_OPEN_GAPS.md``
  G21.

NO DEADBAND — ON PURPOSE
------------------------
``ur7e_gello.yaml:71`` carries ``deadband_rad: 0.004`` and it is tempting to
port it as "the other anti-jitter knob".  It is **dead configuration**:
``filter_stage_joint`` branches on ``use_euro = euro is not None``
(``bridge_stages.py:134-143``) and the operational config selects
``filter_type: "one_euro"`` (``ur7e_gello.yaml:40``), so on the proven path the
deadband/EMA arm never executes.  The hand feel we are copying was produced by
One-Euro **alone**.  Adding a deadband here would ship an untested behaviour
under the label of a validated one, and a position deadband is exactly the
wrong shape for this problem anyway: it re-introduces stair-stepping (a held
target = a stopped joint), which is the thing this module exists to remove.

WHY A DISPLACEMENT BUDGET IS REQUIRED (buffer-correctness INVARIANT)
--------------------------------------------------------------------
Resampling the leader at 30 Hz makes the arm follow continuously, and that
alone would break the stored-action invariant.  The recorded action is a
7-vector in ``[-1,1]`` and one step's task-space increment is by definition
``action * ACTION_SCALE`` = at most 0.0125 m / 0.0625 rad
(``ur_env/envs/config.py:55-73``); the "저장 액션 불변식 (buffer correctness)"
section of ``serl_ur_infra/README.md`` requires
the action written to the replay buffer to be *the action that executed*.  But
the real step period is not the nominal 100 ms — it is 158 ms in the good case,
a measured mean of 502 ms on the production actor loop (worst step 854 ms), and
up to 1.12 s when the learner contends for the GPU.  A human tracking
continuously through a 1 s window travels many ACTION_SCALE steps, while the
stored action can only ever say ``1.0``: the transition would then
systematically **understate** the motion that actually happened, which is the
precise failure the INVARIANT comment in ``config.py`` warns about.

:class:`InterventionBudget` resolves the conflict honestly.  Inside a window the
arm follows the leader smoothly at 30 Hz, but the *total* task-space
displacement it is allowed to accumulate is one ACTION_SCALE step; once that is
spent the remaining substeps HOLD.  The human feels a soft ceiling (the arm
stops tracking until the next window opens) instead of the buffer telling a
lie.  Because the window budget IS one ACTION_SCALE step, dividing the
accumulated displacement by the budget (:meth:`InterventionBudget.consumed_action`)
is the same division ``GelloIntervention`` already performs
(``ur_env/envs/wrappers.py:433-440``), so the reported action lands in
``[-1,1]`` by construction rather than by clipping.

PROPORTIONAL, NEVER PER-AXIS
----------------------------
Over-budget requests are shrunk by the **norm** of the position vector and,
separately, the norm of the rotation vector — the same treatment as the
anti-windup clamp at ``ur_env/envs/wrappers.py:417-431``, and for the same
measured reason recorded there: an axis-wise ``clip`` of a saturated diagonal
request bends its direction (``[2.0, 0.5] -> [1.0, 0.5]``), which the operator
feels as the arm going somewhere they did not point.  Direction is preserved;
only magnitude is rationed.

WHAT THIS MODULE DOES NOT DO
----------------------------
* It never filters *policy* actions.  Only the human leader stream is smoothed,
  so action -> effect credit assignment in replay is untouched — the same line
  the upsampler docstring draws (``ros_backend.py:160-163``).
* It owns no clock, no ROS handles and no arm.  The caller decides when a
  leader sample arrived (``note_sample``), when an output tick happens
  (``filtered``) and when a control window opens (``begin_window``).
* The gripper is not budgeted.  ``action_scale[2]`` exists only to keep the
  argument shape identical to ``ACTION_SCALE``; the gripper action is a
  discrete 3-state command, not a displacement (``wrappers.py:442-469``).
* ``ros_backend.py`` was deliberately NOT touched to make any of this work, and
  should not be: ``send_joint_command`` is already a short-lock last-write-wins
  target update and ``_upsample_loop`` does not care how often the target moves,
  so a faster target cadence needs nothing from the backend.  Hand feel is a
  question about this module and the cadence above it, not about the backend.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# One-Euro gains of the PROVEN teleop stack.  Pinned in two places that already
# agree with each other — ``config/ur7e_gello.yaml:43,46,48`` and the node's
# ``declare_parameter`` defaults (``gello_ur_bridge_node.py:401-408``) — so
# these are measured operating values, not tuning suggestions.  A test asserts
# they still match the yaml.
ONE_EURO_DEFAULTS: Dict[str, float] = {
    "min_cutoff": 1.0,   # Hz; at-rest smoothing.  LOWER = smoother, more lag.
    "beta": 2.0,         # how fast the cutoff opens with speed.
    "d_cutoff": 1.0,     # Hz; low-pass on the internal speed estimate.
}

# The nominal RL control rate (``config.py:32`` ``HZ = 10.0``).  Used ONLY to
# describe how many substeps a window nominally contains — never to enforce a
# per-substep limit; see :attr:`InterventionBudget.nominal_substep_share`.
NOMINAL_CONTROL_HZ: float = 10.0

# Relative slack when deciding "the budget is spent".  Accumulating norms and
# then comparing against the budget is not exact in floating point: a request
# scaled to land exactly on the remaining budget can leave a few ulps behind,
# and without slack the window would report itself unexhausted forever and hand
# out zero-length substeps.
_BUDGET_REL_TOL: float = 1e-9


class OneEuro:
    """Scalar 1-Euro filter (Casiez, Roussel & Vogel 2012) at a DEFAULT period.

    Behaviour-preserving port of ``bridge_stages.py:42-106``.  Adaptive
    low-pass: the cutoff RISES with the low-passed signal speed, so a nearly
    still leader is smoothed hard (hand tremor + Dynamixel encoder/motor noise)
    while a fast move opens the cutoff and tracks with little lag.

    ``min_cutoff`` (Hz) sets the at-rest smoothing (LOWER = smoother / more lag
    when slow).  ``beta`` sets how fast the cutoff opens with speed (HIGHER =
    snappier on fast moves).  ``d_cutoff`` low-passes the internal speed
    estimate so jitter does not inflate the cutoff.

    The constructor ``dt`` is the DEFAULT output period (the nominal tick rate
    of :meth:`__call__`), which is generally NOT the leader sample period:
    leader samples arrive at ~30 Hz while the command stream ticks faster, so
    the speed estimate is advanced by :meth:`update_input` at the sample cadence
    and the output by :meth:`__call__` at the tick cadence.  Calling
    ``update_input`` per output tick makes each sample step look like a much
    faster jump and the filter "opens up in visible pulses" (original docstring,
    ``bridge_stages.py:56-59``; quantified in ``tests/test_leader_stream.py``).

    :meth:`__call__` accepts a per-tick ``dt`` because on the intervention path
    the output tick is NOT periodic; ``None`` selects the constructor value and
    is bit-identical to the original.  See the module docstring, "THE OUTPUT
    TICK IS NOT PERIODIC".

    The arithmetic — including operation order — is copied verbatim so the two
    implementations agree bit for bit; a test asserts exactly that against the
    original class.  Do not "clean up" the expressions.
    """

    def __init__(self, dt: float, min_cutoff: float, beta: float,
                 d_cutoff: float) -> None:
        self._dt = dt
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._raw_prev: Optional[float] = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        # Smoothing factor of a 1st-order low-pass at the given cutoff (Hz).
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def seed(self, x: float) -> None:
        """Preload the filter state (e.g. from the robot's actual pose)."""
        self._x_prev = x
        self._raw_prev = x
        self._dx_prev = 0.0

    def update_input(self, x: float, dt: Optional[float]) -> None:
        """Update the low-passed input-speed estimate at the source cadence.

        ``dt`` is the interval since the previous *sample* (None/<=1e-6 on the
        first sample or a degenerate clock: adopt the value, leave the speed
        estimate alone, never divide).
        """
        if self._raw_prev is None or dt is None or dt <= 1e-6:
            self._raw_prev = x
            return
        dx = (x - self._raw_prev) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        self._dx_prev = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._raw_prev = x

    def _output_dt(self, dt: Optional[float]) -> float:
        """Resolve one output tick's interval.  Never returns a poisonous value.

        ``None`` -> the constructor period.  That is the fixed-rate contract of
        ``bridge_stages.OneEuro`` and the ONLY path the bit-identity tests
        exercise, so it is deliberately a plain passthrough with no arithmetic
        of its own.

        REJECTED (falls back to the constructor period, loudly in the sense that
        the fallback is deterministic and testable, never a crash):

        * ``dt <= 0``.  ``alpha = 1/(1 + tau/dt)`` is not merely wrong there, it
          is *unstable*: ``dt == 0`` divides by zero, ``dt == -tau`` divides by
          zero one level up, ``-tau < dt < 0`` gives ``alpha > 1`` (the filter
          EXTRAPOLATES past the leader) and ``dt < -tau`` gives ``alpha < 0``
          (it runs away from the leader).  Both of the latter compound, because
          ``x_hat`` feeds back into itself.
        * ``NaN``/``inf``.  A single NaN reaching ``_x_prev`` freezes that joint
          for the rest of the session — the same argument
          ``_as_joint_vector`` already makes about a bad Dynamixel read.  ``inf``
          is arithmetically harmless (``alpha`` -> 1, i.e. "track the leader")
          but it can only come from a broken clock, and one predictable rule for
          "the clock lied" is worth more than a special case.

        ACCEPTED, and deliberately not clamped:

        * A tiny positive ``dt``.  For any finite ``dt > 0``, ``alpha`` stays in
          ``[0, 1]``, so it cannot poison anything, and "almost no time passed,
          so barely advance" IS the correct wall-clock answer.  ``dt`` has to be
          absurdly small before anything special happens: ``tau/dt`` only
          overflows to ``inf`` (giving ``alpha == 0.0`` = "hold this tick") below
          ``dt = tau/DBL_MAX`` ~ 8.9e-310, i.e. deep in the subnormals.  At
          ``dt = 1e-300`` — already a nonsense clock reading — ``alpha`` is
          6.3e-300: finite, normal, and behaving exactly as the formula says.
        * A huge ``dt``.  There is NO upper cap.  As ``dt`` grows ``alpha`` rises
          monotonically toward 1 and the output becomes ``x`` — the leader
          itself, never an overshoot, since ``alpha <= 1`` always.  A 5 s window
          therefore effectively disables the smoothing, which is the SAFE
          direction: after 5 s of wall clock a 0.16 s-time-constant filter
          genuinely should have converged, and pretending otherwise is precisely
          the window-counting bug this parameter exists to remove.  Nothing
          downstream depends on the filter for bounding motion either — the
          per-window ``InterventionBudget``, ``_paced_request``, the governor and
          the 250 Hz acceleration limiter each bound it independently — so an
          open filter costs smoothness, not safety.
        """
        if dt is None or not math.isfinite(dt) or dt <= 0.0:
            return self._dt
        return float(dt)

    def __call__(self, x: float, dt: Optional[float] = None) -> float:
        """Advance the OUTPUT by one tick.  Stateful: exactly once per tick.

        ``dt`` is the time that actually elapsed since the previous output tick.
        ``None`` (the default, and what a fixed-rate caller such as the 250 Hz
        bridge timer passes) uses the constructor period and reproduces
        ``bridge_stages.OneEuro.__call__`` bit for bit.
        """
        if self._x_prev is None:
            self.seed(x)
            return x
        # Speed-adaptive cutoff: faster motion -> higher cutoff -> less lag.
        cutoff = self._min_cutoff + self._beta * abs(self._dx_prev)
        a = self._alpha(cutoff, self._output_dt(dt))
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


class LeaderFilter:
    """Per-joint One-Euro bundle that separates leader samples from output ticks.

    One :class:`OneEuro` per joint, all constructed at the OUTPUT period
    (``1/output_hz``), mirroring the bridge's per-joint filter list
    (``ur_gello_bringup/gello_ur_bridge_node.py:410-416``).  Joints are
    independent — no cross-joint coupling — so this is a bundle, not a filter.

    The whole point of the class is that the two cadences cannot be confused:

    * :meth:`note_sample` — call ONLY when a new leader sample arrives (~30 Hz).
      Advances the speed estimate; does not produce output.
    * :meth:`filtered` — call on EVERY output tick, exactly once, with the time
      that really elapsed since the previous one.  Advances the output state and
      returns it.

    See the module docstring ("ONE FILTER, TWO CADENCES") for why conflating
    them makes the filter pulse instead of smooth.
    """

    def __init__(self, output_hz: float, n_joints: int = 6,
                 **euro_params: float) -> None:
        output_hz = float(output_hz)
        if not math.isfinite(output_hz) or output_hz <= 0.0:
            raise ValueError(f"output_hz must be positive and finite (got {output_hz})")
        n_joints = int(n_joints)
        if n_joints <= 0:
            raise ValueError(f"n_joints must be positive (got {n_joints})")

        params = dict(ONE_EURO_DEFAULTS)
        unknown = sorted(set(euro_params) - set(params))
        if unknown:
            raise TypeError(
                f"unknown One-Euro parameter(s) {unknown}; "
                f"expected any of {sorted(params)}"
            )
        params.update({k: float(v) for k, v in euro_params.items()})
        for key, value in params.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} must be positive and finite (got {value})")

        self.output_hz = output_hz
        self.output_dt = 1.0 / output_hz
        self.n_joints = n_joints
        # Copy: ONE_EURO_DEFAULTS is module state and must not be reachable
        # for mutation through an instance.
        self.euro_params: Dict[str, float] = dict(params)
        self._euro: List[OneEuro] = []
        self.reset()

    # -- state ------------------------------------------------------------- #
    def reset(self) -> None:
        """Drop all filter state (rebuild unseeded).

        Rebuilding rather than zeroing is deliberate: an unseeded
        :class:`OneEuro` returns its first input verbatim, so the tick after a
        reset re-anchors on the live leader instead of dragging a stale
        low-pass state (and a stale speed estimate) across the discontinuity.
        Call it at every engage/re-anchor edge — the leader can be anywhere
        while the arm was under policy control, and filtering across that gap
        would command a ramp toward a pose the human never asked for.
        """
        self._euro = [
            OneEuro(self.output_dt, **self.euro_params)
            for _ in range(self.n_joints)
        ]

    def seed(self, q_lead: np.ndarray) -> None:
        """Preload output AND raw state with ``q_lead`` (no motion, no speed).

        :meth:`filtered` already self-seeds on its first call, so this is only
        needed when the caller wants the state aligned *before* the first tick
        (e.g. seeding from measured joints at an engage edge).
        """
        q = self._as_joint_vector(q_lead, "q_lead")
        for i in range(self.n_joints):
            self._euro[i].seed(float(q[i]))

    # -- the two cadences -------------------------------------------------- #
    def note_sample(self, q_lead: np.ndarray, dt: Optional[float]) -> None:
        """A NEW leader sample arrived (~30 Hz).  Speed estimate only.

        ``dt`` is the interval since the previous leader sample, or None for the
        first one.  Never call this per output tick — see the class docstring.
        """
        q = self._as_joint_vector(q_lead, "q_lead")
        for i in range(self.n_joints):
            self._euro[i].update_input(float(q[i]), dt)

    def filtered(self, q_lead: np.ndarray,
                 dt: Optional[float] = None) -> np.ndarray:
        """One OUTPUT tick: smoothed joint vector, shape ``(n_joints,)``.

        Stateful — each joint's filter advances exactly once per call, matching
        the bridge's "called exactly ONCE per joint per tick" contract
        (``bridge_stages.py:122-125``).  Feed it the most recent leader sample
        (zero-order hold between samples); the held input is precisely what the
        speed estimate must NOT be differentiated from.

        ``dt`` is the interval since the previous output tick — the WALL CLOCK
        one, not the nominal period, because on this path the tick is a substep
        of a variable-length control window (module docstring, "THE OUTPUT TICK
        IS NOT PERIODIC").  ``None`` means "use ``output_dt``" and is the right
        value for the first tick after construction/:meth:`reset`, which has no
        previous tick to measure from; it is also what a genuinely fixed-rate
        caller should keep passing.  All six joints share one ``dt``: they are
        ticked by the same loop, so their intervals cannot differ.  Degenerate
        values are handled per joint by :meth:`OneEuro._output_dt` — a filter
        state that feeds back into itself must never see a NaN.
        """
        q = self._as_joint_vector(q_lead, "q_lead")
        out = np.empty(self.n_joints, dtype=float)
        for i in range(self.n_joints):
            out[i] = self._euro[i](float(q[i]), dt)
        return out

    # -- helpers ----------------------------------------------------------- #
    def _as_joint_vector(self, q: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.shape != (self.n_joints,):
            raise ValueError(
                f"{name} must have {self.n_joints} joints (got shape "
                f"{np.asarray(q).shape})"
            )
        # A single NaN/inf entering a low-pass is PERMANENT (x_hat feeds back
        # into itself), so a one-off bad Dynamixel read would freeze that joint's
        # command for the rest of the session.  Refuse at the boundary, loudly,
        # while the caller can still fall back to HOLD.
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite (got {arr!r})")
        return arr


class InterventionBudget:
    """Task-space displacement a single control window may spend.

    The budget is ONE ``ACTION_SCALE`` step per window — 0.0125 m of translation
    and 0.0625 rad of rotation for cube_in_cup (``config.py:73``) — rationed
    across the 30 Hz substeps of one window.  That count is fixed by the NOMINAL
    window (exactly 2 at HZ=10), NOT by how long the step really took, so a step
    that overruns to the measured 0.502 s (or to 1.12 s under learner
    contention) still gets one ACTION_SCALE step spread over the same two
    substeps: the operator's ceiling in m/s falls as the window grows.  See "THE
    OUTPUT TICK IS NOT PERIODIC" above and G21.  Position and rotation have
    independent budgets, exactly as they have independent scales.

    Accounting is on **path length** (the sum of substep magnitudes), not on the
    net displacement.  Path length is monotone, so ``exhausted`` latches for the
    rest of the window instead of flickering back to False when the operator
    reverses; and since the net displacement's norm can never exceed the path
    travelled, budgeting the path bounds the reported action too.  The trade is
    deliberate: an out-and-back inside one window is charged for both legs.

    Usage per control window::

        budget.begin_window()
        while <substeps remain in this window>:
            allowed, spent = budget.take(requested_xi)
            <command allowed>            # zeros once the budget is gone
            if spent:
                <HOLD for the rest of the window>
        action[:6] = budget.consumed_action()   # what actually executed

    ``consumed_action()`` is the value that goes in the replay buffer, and it is
    in ``[-1,1]`` by construction because the budget and ``ACTION_SCALE`` are the
    same number (module docstring, "WHY A DISPLACEMENT BUDGET IS REQUIRED").
    """

    def __init__(self, action_scale: np.ndarray, substep_hz: float) -> None:
        scale = np.asarray(action_scale, dtype=float).reshape(-1)
        if scale.shape != (3,):
            raise ValueError(
                "action_scale must be the 3-vector [pos m/step, rot rad/step, "
                f"gripper] (got shape {np.asarray(action_scale).shape})"
            )
        if not np.all(np.isfinite(scale[:2])) or np.any(scale[:2] <= 0.0):
            raise ValueError(
                f"action_scale[:2] must be positive and finite (got {scale[:2]!r})"
            )
        substep_hz = float(substep_hz)
        if not math.isfinite(substep_hz) or substep_hz <= 0.0:
            raise ValueError(f"substep_hz must be positive and finite (got {substep_hz})")

        self.action_scale = scale.copy()
        self.pos_budget = float(scale[0])
        self.rot_budget = float(scale[1])
        self.substep_hz = substep_hz
        self.substep_dt = 1.0 / substep_hz

        self._used_pos = 0.0
        self._used_rot = 0.0
        self._sum_pos = np.zeros(3, dtype=float)
        self._sum_rot = np.zeros(3, dtype=float)
        self._exhausted = False
        self.begin_window()

    # -- window lifecycle -------------------------------------------------- #
    def begin_window(self) -> None:
        """Open a fresh control window: full budget, nothing consumed."""
        self._used_pos = 0.0
        self._used_rot = 0.0
        self._sum_pos = np.zeros(3, dtype=float)
        self._sum_rot = np.zeros(3, dtype=float)
        self._exhausted = False

    def take(self, xi: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Request a task-space increment; get back what the budget allows.

        ``xi`` is ``(6,)`` = ``[dx, dy, dz (m), wx, wy, wz (rad)]``, the
        increment for ONE substep (base-frame, the convention
        ``PolicyDeltaController``/``GelloIntervention`` already use,
        ``wrappers.py:387-392``).

        Returns ``(allowed_xi (6,), exhausted)``.  Requests that fit are
        returned untouched — the budget is a ceiling on the window, not a
        per-substep rate limit, so a fast operator is not throttled while
        credit remains.  Over-budget requests are scaled by the norm of their
        position and rotation parts *separately*, preserving direction (never a
        per-axis clip; see "PROPORTIONAL, NEVER PER-AXIS").

        ``exhausted`` describes the state AFTER this call: True means the
        position or rotation credit for this window is gone and every later
        ``take`` in this window returns zeros — the caller should HOLD.  Either
        channel latches the flag, because translation and rotation are driven
        together by one hand: continuing to rotate while translation is frozen
        would distort the coupled motion the operator commanded, the same
        direction-distortion argument that rules out per-axis clipping.
        """
        v = np.asarray(xi, dtype=float).reshape(-1)
        if v.shape != (6,):
            raise ValueError(
                f"xi must be a 6-vector [pos(3) m, rot(3) rad] (got shape "
                f"{np.asarray(xi).shape})"
            )
        if not np.all(np.isfinite(v)):
            raise ValueError(f"xi must be finite (got {v!r})")

        p = v[:3].copy()
        w = v[3:].copy()

        rem_pos = max(0.0, self.pos_budget - self._used_pos)
        rem_rot = max(0.0, self.rot_budget - self._used_rot)

        n_pos = float(np.linalg.norm(p))
        n_rot = float(np.linalg.norm(w))
        if n_pos > rem_pos:
            # n_pos > rem_pos >= 0 implies n_pos > 0, so this cannot divide by 0.
            p = p * (rem_pos / n_pos)
        if n_rot > rem_rot:
            w = w * (rem_rot / n_rot)

        self._used_pos += float(np.linalg.norm(p))
        self._used_rot += float(np.linalg.norm(w))
        self._sum_pos = self._sum_pos + p
        self._sum_rot = self._sum_rot + w

        if (self._used_pos >= self.pos_budget * (1.0 - _BUDGET_REL_TOL)
                or self._used_rot >= self.rot_budget * (1.0 - _BUDGET_REL_TOL)):
            self._exhausted = True

        return np.concatenate([p, w]), self._exhausted

    # -- reporting --------------------------------------------------------- #
    def consumed_action(self) -> np.ndarray:
        """Net allowed displacement of this window as a ``(6,)`` env action.

        ``÷ budget`` IS ``÷ ACTION_SCALE`` (they are the same number), so this is
        the same normalisation ``GelloIntervention`` performs at
        ``wrappers.py:433-440`` — which is what makes it a *legal stored action*
        rather than a rescaled guess.  Each component is already inside
        ``[-1,1]`` (``|sum_i| <= ||sum|| <= path <= budget``); the clip is a
        belt-and-braces guard against accumulated float error, never a
        correction, because a real clip here would mean the buffer understates
        the motion — the exact INVARIANT breach this class prevents.
        """
        act = np.concatenate([
            self._sum_pos / self.pos_budget,
            self._sum_rot / self.rot_budget,
        ])
        return np.clip(act, -1.0, 1.0)

    def remaining(self) -> np.ndarray:
        """Unspent ``[pos (m), rot (rad)]`` credit; diagnostics/telemetry."""
        return np.array([
            max(0.0, self.pos_budget - self._used_pos),
            max(0.0, self.rot_budget - self._used_rot),
        ], dtype=float)

    @property
    def exhausted(self) -> bool:
        """True once this window's position or rotation credit is spent."""
        return self._exhausted

    @property
    def nominal_substeps(self) -> int:
        """Substeps a NOMINAL (``1/NOMINAL_CONTROL_HZ``) window would contain."""
        return max(1, int(round(self.substep_hz / NOMINAL_CONTROL_HZ)))

    @property
    def nominal_substep_share(self) -> np.ndarray:
        """Advisory ``[pos, rot]`` fair share of one substep.  NOT enforced.

        Enforcing a per-substep cap would throttle a legitimately fast operator
        who is still inside the window budget, and it is not what the invariant
        asks for: the invariant constrains the window TOTAL (``config.py:55-73``,
        ``README.md`` "저장 액션 불변식").  Exposed for telemetry and for pacing decisions
        the caller may want to make with full knowledge that they are policy,
        not correctness.
        """
        return np.array([self.pos_budget, self.rot_budget]) / self.nominal_substeps
