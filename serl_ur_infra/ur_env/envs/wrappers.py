"""GELLO intervention wrapper — the SpacemouseIntervention analog.

Contract kept identical to franka_env.envs.wrappers.SpacemouseIntervention so
train_rlpd.py / record_demos.py work unchanged: when the human is intervening,
step() executes the human action instead of the policy action and reports it
in info["intervene_action"] (in env action units, [-1,1]^7).

UR-specific metadata is reported on every step:
    info["policy_action"]  = policy output before any human override
    info["intervened"]     = 1 if the human action was executed, else 0
This keeps counterfactual policy output and action-source labeling at the
robot-infra boundary without patching the vendored hil-serl actor.

Semantics differ from the spacemouse in one fundamental way: GELLO is an
absolute-pose device with no zero-rest, so intervention is gated by an
explicit DEADMAN (hold-to-engage), not by output magnitude.

    deadman press  -> engage: latch (leader TCP, robot commanded TCP) anchors
    while held     -> expert action = anchored delta, re-expressed as a
                      per-step env delta:  xi = log(T_cmd^-1 @ T_des),
                      clipped to the action space (clipping = natural chase
                      rate limit; correspondence recovers over a few steps)
    leader lost   -> invalidate the anchor and execute a zero arm/gripper HOLD;
                     this remains an intervention, so stale input can never
                     authorize an implicit handback to the policy
    deadman release-> disengage, policy resumes instantly

For the ROS-topic source, an explicit fresh release is different from losing
the publisher: once a heartbeat has been received, 0.5 s of silence raises
``DeadmanHeartbeatStaleError`` before any policy action reaches the robot.

One env step is still one transition, but an INTERVENED step no longer sets the
joint target only once.  ``UR7eEnv.step`` used to spend its 100 ms window
asleep, so the 250 Hz upsampler replayed "accelerate -> arrive -> brake -> sit
still" every window (measured 16-40% fully-stopped joint time at a 0.15 rad/s
leader, 2.27 units of ripple).  This wrapper now drives the env's window at
30 Hz — the leader's own publish rate — re-reading the leader, re-running the
anchored-delta maths and refreshing the target each substep (0% stopped, 0.85
ripple).  Two things make that safe rather than a way to smuggle extra motion
into one transition:

  * one shared per-window displacement BUDGET (``InterventionBudget``), so the
    whole window can still never exceed one ``ACTION_SCALE`` step; and
  * the recorded ``info["intervene_action"]`` is that budget's cumulative
    allowance, harvested when the window closes — the executed==reported
    invariant survives even though the total is unknown when ``step`` starts.

BOTH OF THOSE DESCRIBE ``follow_mode="in_window"``, which is no longer the
default.  Since 2026-07-30 the shipped path is ``follow_mode="background"``: a
daemon thread inside ``UR7eEnv`` follows the leader for the WHOLE real step
period (512 ms on the production actor, of which ``env.step`` is 100 ms), the RL
loop only samples transitions, and there is NO displacement budget at all — the
governor, the workspace box and the 250 Hz upsampler are the bounds, and the
recorded action is the displacement the controller actually committed, clipped
with ``info["intervention_saturation"]`` reporting by how much.  The three
methods under "UR7eEnv BACKGROUND-FOLLOWER protocol" below are that path;
``GelloIntervention.follow_xi`` carries the argument for dropping the budget and
the price of doing so.

THE WHOLE PATH, in call order.  This is the canonical copy of the list; the
other four files of the mechanism carry a short pointer back here instead of
repeating it:

  GelloIntervention.step (:705) -> _open_substep_window (:683) installs THIS
    object as the driver, via UR7eEnv.begin_intervention_window (ur7e_env.py:526)
  UR7eEnv.step (ur7e_env.py:657) consumes the one-shot driver, then
    _apply_action(action, driver) (ur7e_env.py:449) issues the FIRST target:
      charge (:607) -> PolicyDeltaController.step(xi, dt=1/substep_hz)
      (policy_delta_controller.py:183, wired at ur7e_env.py:502-505) ->
      backend.send_joint_command
    _drive_intervention_substeps (ur7e_env.py:553) then paces the rest of the
    window at 30 Hz instead of sleeping it away:
      substep(dt) (:618) -> leader re-read -> LeaderFilter (leader_stream.py:259)
      -> InterventionBudget.take (leader_stream.py:447) -> controller.step(dt=1/30)
      -> backend.send_joint_command
    window closes -> consumed_window_action (:663) is harvested into
      info["intervention_window_action"] (ur7e_env.py:654)
  back in step() that key becomes info["intervene_action"] (:738).

The POLICY path is untouched by all of this, by construction: no driver is
installed, so ``UR7eEnv.step`` takes the same single-target-plus-sleep branch it
always did.  Protocol contract: ``UR7eEnv.begin_intervention_window``; the
substep methods live under "UR7eEnv substep-driver protocol" below.

VERIFIED on the real UR7e 2026-07-30: ARMED ``run_real_hil.py --arm --scale
1.0``, 120 intervened steps, every check PASS — substeps=2 in every window,
frame-map residual 0.130 / alpha 0.983, action-exec dp_ratio median 1.000,
held 0%, operator confirmed the hand feel.  Evidence:
docs/testing/04_HIL_INTERVENTION.md §9.

TODO(together):
- deadman hardware: pynput key hold for now; USB footswitch later.
- success/fail marking keys for record_success_fail.py (spacemouse buttons
  there; we need two more keys).
- T_tool handling: anchor math below uses flange poses; wire the real
  T_tool_L/R from the bridge config for TCP-accurate mapping.
"""

import math
import threading
import time
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from ur_env.envs.leader_stream import (
    ONE_EURO_DEFAULTS,
    InterventionBudget,
    LeaderFilter,
)
from ur_env.observation_schema import gripper_position_from_state

# A poll of the backend's leader cache returns (arr, age); the sample's receive
# instant is therefore ``time.monotonic() - age`` (URRosBackend._aged,
# ur_env/envs/ros_backend.py:567-571).  Re-polling the SAME cached sample
# reproduces that instant to within the few microseconds between the two
# monotonic reads, while a genuinely new sample moves it by ~1/30 s.  1 ms sits
# two orders of magnitude above the former and one below the latter.
_LEADER_RX_EPSILON_S = 1e-3

try:
    from ur_gello_bringup.ur_kin import fk, so3_log

    _KIN_AVAILABLE = True
except ImportError:
    _KIN_AVAILABLE = False


def _ensure_kinematics_available() -> None:
    """Retry the optional workspace import after callers extend ``sys.path``."""

    global _KIN_AVAILABLE, fk, so3_log
    if _KIN_AVAILABLE:
        return
    try:
        from ur_gello_bringup.ur_kin import fk as imported_fk
        from ur_gello_bringup.ur_kin import so3_log as imported_so3_log
    except ImportError as exc:
        raise RuntimeError("ur_gello_bringup not importable") from exc
    fk = imported_fk
    so3_log = imported_so3_log
    _KIN_AVAILABLE = True


# ---------------------------------------------------------------------------- #
# Deadman sources — the "engaged" signal + a sensitivity gain can come from     #
# either the spacebar (default) or a ROS topic published by the HIL GUI.        #
# ---------------------------------------------------------------------------- #
class DeadmanSource:
    """Interface: the human deadman 'engaged' signal + a sensitivity gain.

    gain() == 1.0 means 1:1 leader->robot translation. Implementations must be
    thread-safe: is_engaged()/gain() are read from the env step thread while
    the underlying state is written from a listener / ROS callback thread.
    A health-checked source may raise from is_engaged() when silence makes the
    state unknowable; callers must not reinterpret that as a disengagement.
    """

    def is_engaged(self) -> bool:
        raise NotImplementedError

    def gain(self) -> float:
        return 1.0


class SpacebarDeadman(DeadmanSource):
    """DEFAULT deadman: hold SPACEBAR to engage (pynput). gain fixed at 1.0.

    Behavior is identical to the listener that used to live inline in
    GelloExpert.__init__. Added guard: mirror UR7eEnv's ESC-listener try/except
    so a headless/no-display box (pynput raises with no X server) degrades to
    is_engaged()==False forever instead of crashing env construction.
    """

    def __init__(self):
        self._engaged = False
        self._lock = threading.Lock()
        self._listener = None
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.space:
                    with self._lock:
                        self._engaged = True

            def on_release(key):
                if key == keyboard.Key.space:
                    with self._lock:
                        self._engaged = False

            self._listener = keyboard.Listener(
                on_press=on_press, on_release=on_release
            )
            self._listener.start()
        except Exception as e:
            print(
                f"[SpacebarDeadman] keyboard listener unavailable ({e}) — "
                "deadman will never engage (is_engaged()==False)"
            )

    def is_engaged(self) -> bool:
        with self._lock:
            return self._engaged

    def gain(self) -> float:
        return 1.0


class DeadmanHeartbeatStaleError(RuntimeError):
    """A previously live deadman heartbeat crossed its safety deadline.

    This is deliberately an exception rather than ``is_engaged() == False``.
    ``False`` means the operator sent a fresh, explicit DISENGAGE and policy
    control may resume.  A lost publisher provides no such authorization, so
    the actor must stop before it can forward another policy action.
    """

    def __init__(self, *, age_s: float, stale_s: float):
        self.age_s = float(age_s)
        self.stale_s = float(stale_s)
        super().__init__(
            "/hil/deadman heartbeat stale: "
            f"age={self.age_s:.3f}s exceeds {self.stale_s:.3f}s; "
            "refusing policy fallback and stopping the actor"
        )


class RosTopicDeadman(DeadmanSource):
    """Deadman driven by /hil/deadman, published by the HIL GUI.

    FROZEN CONTRACT (shared with the GUI — do NOT deviate):
      topic  /hil/deadman        type std_msgs/Float32MultiArray
      data   [engaged, gain]      engaged in {0.0, 1.0}; gain in [0.10, 1.00]
      rate   20 Hz heartbeat      (published continuously, not only on change)
      QoS    default reliable, depth 10
      stale  after first rx, newest msg older than STALE_S -> raise and stop actor

    Subscribes on the passed rclpy node (the URRosBackend node, spun in its bg
    thread), so callbacks fire without this class owning an executor.
    """

    STALE_S = 0.5

    def __init__(self, node):
        self._lock = threading.Lock()
        self._engaged_raw = 0.0
        self._gain = 1.0
        self._last_rx: Optional[float] = None  # monotonic; None => never rx
        from std_msgs.msg import Float32MultiArray

        node.create_subscription(
            Float32MultiArray, "/hil/deadman", self._on_msg, 10
        )

    def _on_msg(self, msg):
        # Malformed input must not refresh the heartbeat.  In particular, an
        # empty/NaN/fractional message cannot impersonate the GUI's explicit
        # ``[0.0, gain]`` policy hand-back.  A transient bad packet leaves the
        # last valid state intact; a sustained bad stream reaches STALE_S and
        # fails closed through ``is_engaged``.
        try:
            data = list(msg.data)
            if len(data) != 2:
                return
            engaged = float(data[0])
            gain = float(data[1])
        except (TypeError, ValueError, OverflowError):
            return
        if (
            not math.isfinite(engaged)
            or engaged not in (0.0, 1.0)
            or not math.isfinite(gain)
            or not 0.10 <= gain <= 1.00
        ):
            return
        with self._lock:
            self._engaged_raw = engaged
            self._gain = gain
            self._last_rx = time.monotonic()

    def fresh_engaged(self) -> bool:
        """Return exact state only after a currently fresh valid heartbeat.

        The scene-ready authorization gate uses this stricter read so the
        pre-first-message compatibility behavior of ``is_engaged`` cannot be
        mistaken for an explicit DISENGAGE edge.
        """

        with self._lock:
            if self._last_rx is None:
                raise RuntimeError("/hil/deadman has no valid heartbeat")
            age_s = time.monotonic() - self._last_rx
            if age_s > self.STALE_S:
                raise DeadmanHeartbeatStaleError(
                    age_s=age_s, stale_s=self.STALE_S
                )
            return self._engaged_raw == 1.0

    def is_engaged(self) -> bool:
        with self._lock:
            if self._last_rx is None:
                # The task config waits up to 15 s for the first heartbeat and
                # rejects startup if none arrives.  Keep this method inert in
                # that pre-start state so constructing the source is harmless.
                return False
            age_s = time.monotonic() - self._last_rx
            if age_s > self.STALE_S:
                raise DeadmanHeartbeatStaleError(
                    age_s=age_s, stale_s=self.STALE_S
                )
            return self._engaged_raw == 1.0

    def gain(self) -> float:
        with self._lock:
            return float(np.clip(self._gain, 0.10, 1.00))


class GelloExpert:
    """Deadman state + latest leader reading (SpaceMouseExpert analog).

    Leader joints come from the env's ROS backend (/gello/joint_states); the
    deadman 'engaged'/gain signal is delegated to a DeadmanSource (spacebar by
    default, or the /hil/deadman ROS topic from the GUI).
    """

    def __init__(self, backend, deadman: Optional[DeadmanSource] = None):
        self._backend = backend
        self.deadman = deadman if deadman is not None else SpacebarDeadman()

    def is_engaged(self) -> bool:
        return self.deadman.is_engaged()

    def gain(self) -> float:
        return self.deadman.gain()

    def get_leader(self) -> Tuple[Optional[np.ndarray], Optional[float], float]:
        """Returns (q_lead(6,), trigger in [0,1] or None, joint age_seconds).

        The backend joins /gello/joint_states (6 joints) with the separate
        trigger topic and reports an unusable trigger — never received, or
        stale — as NaN in arr[6]. NaN is mapped to None here so the single
        "no trigger right now" case has one representation for callers.
        """
        arr, age = self._backend.get_gello_state()
        if arr is None:
            return None, None, age
        q = np.asarray(arr[:6], dtype=float)
        grip = None
        if len(arr) > 6:
            g = float(arr[6])
            if math.isfinite(g):
                grip = g
        return q, grip, age


class GelloIntervention(gym.ActionWrapper):
    LEADER_STALE_S = 0.3  # leader older than this -> refuse to intervene
    GRIP_CLOSE_THR = 0.7  # leader trigger hysteresis
    GRIP_OPEN_THR = 0.3

    def __init__(self, env, deadman: Optional[DeadmanSource] = None):
        super().__init__(env)
        _ensure_kinematics_available()
        self.expert = GelloExpert(env.unwrapped.backend, deadman=deadman)
        self.action_scale = env.unwrapped.action_scale

        # ONE re-entrant lock over every piece of state the background follower
        # touches: the anchor triple, the leader filter and the displacement
        # budget.  Not a nicety — ``_disengage`` clears ``_anchored`` and then
        # both anchors, so a follower reading the three fields separately can
        # see ``_anchored is True`` and ``T_g_anchor is None`` one line apart and
        # die on ``None[:3, :3]``.  On a daemon thread that TypeError goes
        # nowhere: the follower disappears, the arm silently stops following,
        # and nothing in the log says why.  So the follower snapshots all three
        # under this lock, in one place.
        #
        # LOCK ORDER, one way only: this one is always RELEASED before
        # ``UR7eEnv._emit_arm_command`` asks for the command lock — ``follow_xi``
        # returns its increment and the env commands it afterwards.  So the
        # follower never holds this while waiting for the command lock, and the
        # two cannot form a cycle.
        self._follow_lock = threading.RLock()

        self._anchored = False
        self.T_g_anchor: Optional[np.ndarray] = None
        self.T_r_anchor: Optional[np.ndarray] = None
        self._grip_cmd = 0.0  # last hysteresis output in {-1, 0, +1}
        self._gain = 1.0      # sensitivity gain, latched at each engage edge
        self._hold_requested = False

        # ---- in-window leader resampling (see the substep protocol below) --- #
        # ``substep_hz`` is read from the ENV, not from the config, so there is
        # exactly one authority on the substep rate: the env owns the pacing loop
        # and hands us the same period back through ``substep(dt)``.  An env that
        # does not expose it (fake/stub envs, and any config without an
        # ``INTERVENTION`` block, which resolves to 0.0) leaves ``_filter`` and
        # ``_budget`` at None and this wrapper behaves exactly as it did before
        # substepping existed.
        self.substep_hz = float(
            getattr(env.unwrapped, "intervention_substep_hz", 0.0)
        )
        #: "background" — the env runs a daemon follower and this wrapper is its
        #: source of leader maths; "in_window" — the pre-2026-07-30 path, where
        #: this wrapper is instead a one-shot driver for ``env.step``'s sleep.
        #: Read from the ENV for the same reason ``substep_hz`` is: one authority
        #: per knob.  Envs that do not declare it (stub/fake envs in the tests)
        #: get the conservative pre-change value.
        self._follow_mode = str(
            getattr(env.unwrapped, "intervention_follow_mode", "in_window")
        )
        task_config = getattr(env.unwrapped, "config", None)
        intervention_cfg = getattr(task_config, "INTERVENTION", None) or {}
        one_euro = {
            name: float(intervention_cfg.get(f"one_euro_{name}", default))
            for name, default in ONE_EURO_DEFAULTS.items()
        }
        self._filter: Optional[LeaderFilter] = None
        self._budget: Optional[InterventionBudget] = None
        if self.substep_hz > 0.0:
            # output_hz is the rate ``filtered()`` is ticked at, i.e. the substep
            # rate — NOT the leader sample rate, which reaches the filter through
            # note_sample().  Conflating the two is what makes a One-Euro filter
            # pulse instead of smooth (LeaderFilter class docstring).
            self._filter = LeaderFilter(self.substep_hz, **one_euro)
            self._budget = InterventionBudget(self.action_scale, self.substep_hz)
        self._leader_rx: Optional[float] = None
        # Monotonic instant of the previous ``LeaderFilter.filtered`` tick.  The
        # filter needs the interval that REALLY elapsed, not the nominal substep
        # period — see substep() and leader_stream's "THE OUTPUT TICK IS NOT
        # PERIODIC".  None => the next tick is the filter's first.
        self._last_filtered_at: Optional[float] = None
        self._window_gripper = 0.0

    # ------------------------------------------------------------------ #
    def _leader_T(self, q_lead: np.ndarray) -> np.ndarray:
        # TODO: @ T_tool_L (flange-only for the skeleton)
        return fk(q_lead)

    def _robot_T_cmd(self) -> np.ndarray:
        return self.env.unwrapped.controller.tcp_cmd()

    def _engage(self, q_lead: np.ndarray):
        with self._follow_lock:
            self.T_g_anchor = self._leader_T(q_lead)
            self.T_r_anchor = self._robot_T_cmd()
            # LATCH the sensitivity gain at the engage edge (not read live
            # per-tick) so a slider nudged mid-motion cannot discontinuously
            # rescale the in-progress anchored delta.  The substep loop and the
            # background follower both reuse this latched value — neither may
            # re-read expert.gain() per 30 Hz tick, which would reintroduce
            # exactly the discontinuity this latch removes.  Latched HERE, on
            # the RL thread, which is why a follower that loses its anchor asks
            # the RL thread to re-anchor instead of doing it itself.
            self._gain = self.expert.gain()
            self._anchored = True
            # Drop filter state at the engage edge: the leader could be anywhere
            # while the policy was driving, and low-passing across that gap would
            # ramp the arm toward a pose the human never asked for.  Dropping the
            # receive bookkeeping with it makes the next sample the filter's
            # first.
            self._reset_leader_filter()

    def _disengage(self):
        # Under the lock as ONE transition: the follower must never observe the
        # intermediate state (anchored, but with a None anchor).
        with self._follow_lock:
            self._anchored = False
            self.T_g_anchor = None
            self.T_r_anchor = None

    def _anchor_snapshot(self):
        """``(T_g_anchor, T_r_anchor)`` or ``(None, None)``, read atomically."""

        with self._follow_lock:
            if not self._anchored:
                return None, None
            if self.T_g_anchor is None or self.T_r_anchor is None:
                return None, None
            return self.T_g_anchor, self.T_r_anchor

    def _expert_delta_xi(self, q_lead: np.ndarray) -> np.ndarray:
        """Anchored leader delta -> task-space increment xi (6,), real units.

        ``[dpos (m), drot rotvec (rad)]`` in the WORLD/base frame — the same
        convention PolicyDeltaController expects (FrankaEnv semantics): position
        error along base axes, rotation error as a base-frame rotvec.

        Split out of ``_expert_delta_action`` so the 30 Hz substep loop can feed
        the controller directly without a normalize/re-scale round trip (which
        is not bit-exact and would make the recorded action drift away from the
        commanded one).  The maths below is byte-for-byte the previous body; only
        the anchor READ changed, from three separate field reads to one locked
        snapshot (see ``__init__`` on ``_follow_lock``).
        """
        T_g_anchor, T_r_anchor = self._anchor_snapshot()
        if T_g_anchor is None:
            raise RuntimeError(
                "_expert_delta_xi called without an anchor — engage first"
            )
        T_g = self._leader_T(q_lead)
        # world-frame delta (same convention as eef_delta.py: R_g @ R_anchor^T)
        R_delta = T_g[:3, :3] @ T_g_anchor[:3, :3].T
        p_delta = T_g[:3, 3] - T_g_anchor[:3, 3]

        T_des = np.eye(4)
        T_des[:3, :3] = R_delta @ T_r_anchor[:3, :3]
        # Apply the LATCHED sensitivity gain to the translation delta only
        # (rotation stays 1:1). This scaling happens BEFORE the /action_scale ->
        # clip below, so the returned `act` is exactly what env.step re-scales
        # and executes: the executed==reported buffer-correctness invariant
        # (README "저장 액션 불변식") is preserved — gain is upstream of the report.
        T_des[:3, 3] = T_r_anchor[:3, 3] + self._gain * p_delta

        T_cmd = self._robot_T_cmd()
        p_err = T_des[:3, 3] - T_cmd[:3, 3]                 # base-frame
        w_err = so3_log(T_des[:3, :3] @ T_cmd[:3, :3].T)    # base-frame rotvec

        # Anti-windup: bound the per-step demand to ONE action_scale step (port
        # of eef_delta's lag clamp). PROPORTIONAL (norm) clamp, NOT the old
        # per-axis np.clip: an axis-wise clip of a saturated diagonal move
        # distorts its DIRECTION (e.g. [2.0, 0.5] -> [1.0, 0.5] bends the path),
        # which the operator feels as a wrong-direction/scale response. Scaling
        # each of the position and rotation error vectors by its own norm keeps
        # the commanded direction exact and caps |act| at 1 by construction, so
        # the reported action stays in [-1,1] and — with the controller's line
        # search now delivering a feasible step — close to what is executed.
        nv, nw = float(np.linalg.norm(p_err)), float(np.linalg.norm(w_err))
        if nv > self.action_scale[0]:
            p_err = p_err * (self.action_scale[0] / nv)
        if nw > self.action_scale[1]:
            w_err = w_err * (self.action_scale[1] / nw)
        return np.concatenate([p_err, w_err])

    def _expert_delta_action(self, q_lead: np.ndarray) -> np.ndarray:
        """Anchored leader delta -> per-step env action (6,), in [-1,1]."""

        xi = self._expert_delta_xi(q_lead)
        act = np.concatenate(
            [xi[:3] / self.action_scale[0], xi[3:] / self.action_scale[1]]
        )
        return np.clip(act, -1.0, 1.0)

    def _expert_gripper(self, grip: Optional[float]) -> float:
        """Leader trigger (0 open .. 1 closed) -> 3-state action in {-1, 0, +1}.

        UR7eEnv._send_gripper_command reads this as: <= -0.5 CLOSE, >= +0.5
        OPEN, anything between is a no-op. So the three states are close / hold
        / open, and the mid-band latch below means "keep doing what the human
        last asked" rather than chattering around a single threshold.

        No trigger (topic silent or stale) -> 0.0 = HOLD, and the latch is left
        untouched. Rationale: 0.0 is the only value that cannot move the
        gripper, so a dead trigger topic can neither drop a held payload nor
        clamp on something. Replaying the latch instead would keep re-issuing a
        grasp command derived from a signal we no longer have — the same
        fail-safe-to-inert rule used for the gripper trigger. (The independent
        /hil/deadman heartbeat instead aborts the actor when stale.) The latch
        is preserved (not zeroed) so a brief dropout resumes the human's
        last intent instead of forcing them to re-cross a threshold; a real
        release is a fresh trigger value, which arrives on the next message.
        Arm teleop is deliberately NOT gated on the trigger: losing the gripper
        signal must not also yank the arm away from the human mid-motion.
        """
        if grip is None:
            return 0.0
        if grip >= self.GRIP_CLOSE_THR:
            self._grip_cmd = -1.0
        elif grip <= self.GRIP_OPEN_THR:
            self._grip_cmd = 1.0
        return self._grip_cmd

    # ------------------------------------------------------------------ #
    def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        self._hold_requested = False
        q_lead, grip, age = self.expert.get_leader()

        # A fresh, explicit deadman release authorizes policy handback.
        if not self.expert.is_engaged():
            if self._anchored:
                self._disengage()
            return action, False

        # Missing/stale leader joints while the deadman remains engaged are a
        # signal failure, not an operator release.  Handing the arm to an
        # unrelated (and potentially saturated) policy action here would turn a
        # GELLO dropout into motion.  Invalidate the old anchor and execute a
        # zero 7-D action instead: arm delta = HOLD, gripper command = HOLD.
        # ``step`` also asks the backend to replace any accumulated lagging
        # target with its acceleration-limited stop point and re-anchors the
        # task-space integrator there.  It is deliberately reported as an
        # intervention so transition.actions is the exact zero action selected;
        # once fresh joints return, the ordinary engage path below re-anchors
        # and its first delta is zero.
        if q_lead is None or age > self.LEADER_STALE_S:
            if self._anchored:
                self._disengage()
            self._hold_requested = True
            return np.zeros(self.action_space.shape, dtype=np.float32), True

        if not self._anchored:
            self._engage(q_lead)

        # The filter is fed here as well as in substep(): this is a real leader
        # sample at a real receive instant, and skipping it would leave the
        # filter blind across the whole inter-window gap (which on this rig is
        # the larger half of the step period).  The action below still uses the
        # RAW leader, exactly as before this change — see substep() for why the
        # filter is a substep-only consumer.
        self._note_leader_sample(q_lead, age)

        expert_a = np.zeros(7, dtype=np.float32)
        expert_a[:6] = self._expert_delta_action(q_lead)
        expert_a[6] = self._expert_gripper(grip)
        return expert_a, True

    # ------------------------------------------------------------------ #
    # UR7eEnv substep-driver protocol: 30 Hz leader resampling in-window  #
    #                                                                     #
    # WHY THE DRIVER LIVES HERE AND THE LOOP LIVES IN THE ENV.  The maths  #
    # (anchor, gain, budget) is intervention semantics and belongs to this  #
    # wrapper; the pacing, the sleep it replaces and the controller/backend #
    # handles belong to UR7eEnv.  A wrapper cannot reach into the middle of  #
    # env.step(), so the env calls back into us — see                       #
    # UR7eEnv.begin_intervention_window for the protocol contract.          #
    # ------------------------------------------------------------------ #
    def _reset_leader_filter(self) -> None:
        """Forget every leader sample seen so far (no-op when substepping is off)."""
        with self._follow_lock:
            if self._filter is not None:
                self._filter.reset()
            self._leader_rx = None
            # The output-tick clock is dropped with the state it timed: a reset
            # filter re-anchors on its next input verbatim, so the interval
            # across the engage/episode gap describes nothing and must not be
            # measured.
            self._last_filtered_at = None

    def _note_leader_sample(
        self, q_lead: np.ndarray, age: float, now: Optional[float] = None
    ) -> None:
        """Feed the one-euro filter, but ONLY on a genuinely new leader sample.

        ``now`` is the monotonic instant this poll happened at.  ``None`` reads
        the module clock, which is what the RL-thread callers do and what the
        virtual-clock tests patch.  The BACKGROUND follower passes its own
        reading instead, because ``_follow_tick`` is a pure function of ``now``:
        one clock read per tick, taken by the pacing shell, used for both the
        sample-identity arithmetic here and the filter's output interval.

        The backend caches the leader at its publisher's 30 Hz
        (``URRosBackend._on_gello``, ur_env/envs/ros_backend.py:528-532) while
        the substep loop polls that cache at up to the same rate, so a poll very
        often returns a sample the filter has already consumed.  Calling
        ``note_sample`` on every tick would advance the filter with dt values
        far below the true sample spacing: its speed-adaptive cutoff would open
        in a PULSE each time a new sample landed and collapse in between, which
        is the opposite of the steady low-pass the proven teleop stack gets from
        the same parameters.  So each sample is admitted once, timed by its own
        receive instant.

        A non-finite leader joint is dropped rather than raised on: a single NaN
        entering a low-pass is permanent (``LeaderFilter._as_joint_vector``), and
        the pre-substep behaviour for a bad Dynamixel read was a controller
        BAD_INPUT hold, not an exception that kills the actor.  Leaving the
        filter untouched preserves that.
        """
        if self._filter is None or not np.all(np.isfinite(q_lead)):
            return
        rx = (time.monotonic() if now is None else float(now)) - age
        previous = self._leader_rx
        if previous is not None and rx <= previous + _LEADER_RX_EPSILON_S:
            return
        self._leader_rx = rx
        # dt=None on the first sample: LeaderFilter adopts it without touching
        # the speed estimate, which is right — there is no interval to divide by.
        self._filter.note_sample(
            q_lead, None if previous is None else rx - previous
        )

    def _paced_request(self, xi: np.ndarray) -> np.ndarray:
        """Shrink ONE substep's request to this window's fair per-substep share.

        IN-WINDOW PATH ONLY.  The background follower deliberately has no budget
        and therefore no pacing — see ``follow_xi``.

        This is pacing policy, not an invariant — ``InterventionBudget`` says so
        itself (``nominal_substep_share``: "advisory, NOT enforced").  It is here
        because without it the accounting and the arm disagree, measurably:

        ``InterventionBudget.take`` charges the REQUEST, while the governor caps
        what the controller commits at ``v_max * dt`` = 0.150/30 = 0.0050 m per
        substep.  The first request of a window is the leader lag accumulated
        across the inter-window gap (~0.097 s of the measured 0.197 s period), so
        at ordinary operating speeds it exceeds 0.0050 m routinely — not as an
        edge case.  Charged in full but commanded at 0.0050 m, it would drain the
        0.0125 m window budget at t=0: every later substep would get zeros (no
        target refresh, i.e. the stiffness this whole change removes), and the
        stored action would claim 2.5x the motion the arm made.

        The nominal share (0.0125/3 = 0.00417 m) is BELOW the governor's
        per-substep allowance, so a paced request is never governed down: charged
        == commanded == recorded.  The window TOTAL is unchanged — three paced
        substeps still spend exactly one ACTION_SCALE step — so lag recovery per
        env step is the same as before substepping existed; only its
        distribution inside the window changes, which is the point.

        Cost, accepted: a burst confined to one substep is spread over the
        window instead of being spent at once.  Proportional norm scaling only,
        never a per-axis clip (see _expert_delta_xi for why).

        A window with one nominal substep has share == budget and is returned
        untouched, bit-for-bit, so the degenerate ``substep_hz <= HZ`` case
        cannot drift away from the pre-substep numbers.
        """
        if self._budget.nominal_substeps <= 1:
            return xi
        share = self._budget.nominal_substep_share
        p, w = xi[:3].copy(), xi[3:].copy()
        n_p, n_w = float(np.linalg.norm(p)), float(np.linalg.norm(w))
        if n_p > share[0]:
            p = p * (share[0] / n_p)
        if n_w > share[1]:
            w = w * (share[1] / n_w)
        return np.concatenate([p, w])

    def charge(self, xi: np.ndarray) -> np.ndarray:
        """Draw this window's FIRST target from its displacement budget.

        Called by ``UR7eEnv._apply_action``; see the INVARIANT notes there for
        why the first target is charged and governed exactly like every substep.
        """
        allowed, _exhausted = self._budget.take(
            self._paced_request(np.asarray(xi, dtype=float).reshape(6))
        )
        return allowed

    def substep(self, dt: float) -> dict:
        """One 30 Hz target refresh inside the current ``env.step`` window.

        The deadman is deliberately NOT re-read here.  One operator read per env
        step keeps ``is_engaged()``/``gain()`` on the same clock as the recorded
        transition (gain is latched at the engage edge for the same reason), and
        a release therefore takes effect at the window boundary — after at most
        one window of further motion, itself bounded by the window budget.
        """
        q_lead, _, age = self.expert.get_leader()
        # Mid-window signal failure: end the window and brake.  Same rule as
        # action() — a GELLO dropout must never turn into motion, and a target
        # already in flight is motion.  A non-finite joint is the same class of
        # failure (an unusable leader), so it takes the same exit.
        if (
            q_lead is None
            or age > self.LEADER_STALE_S
            or not np.all(np.isfinite(q_lead))
        ):
            return {"issued": False, "stop": "GELLO_STALE", "brake": True}

        self._note_leader_sample(q_lead, age)
        # THE FILTER'S CLOCK IS THE WALL CLOCK, not the substep counter.  ``dt``
        # here is the ``1/substep_hz`` the env PLANS to pace at; the interval
        # between two ``filtered()`` calls is not that number and is not even
        # constant.  ``_drive_intervention_substeps`` paces against the NOMINAL
        # window end (``start_time + 1/HZ``), so it always runs the same two
        # ticks 1/30 s apart and everything the step overruns its nominal window
        # by — the actor's gRPC round trip, the camera read — lands in ONE lump
        # before the next window's first tick.  The spacing therefore alternates
        # ``1/30 s, T - 1/30 s`` where T is the REAL step period: 33 ms then
        # 469 ms at the measured mean T = 0.502 s, 33 ms then 1087 ms at the
        # contended T = 1.12 s.  "33, then 67, then 33" holds only for
        # T == 1/HZ exactly, which is what the virtual-clock tests construct and
        # what the rig never does.  Feeding the filter the nominal period instead
        # made its time constant scale WITH the window, so a contended learner
        # turned a 0.159 s filter into a 2.95 s one — sluggish exactly when the
        # operator is most likely to be taking over.  Measure the real interval
        # instead.  (It is necessary, not sufficient: leader_stream's "WHAT THE
        # dt FIX DOES NOT FIX" has what a long window still costs.)
        #
        # ``time`` is read through the module global on purpose: the substep
        # tests replace ``wrappers.time`` with a virtual clock, and this
        # measurement has to move with the same clock that paces the window.
        # ``monotonic`` and not ``time()``: the env paces the window with the
        # WALL clock, but a wall-clock jump (NTP step, DST) between two ticks
        # would be handed to a filter as a real interval — or as a negative one.
        # Only differences of monotonic readings are ever taken here, and
        # ``OneEuro._output_dt`` refuses a non-positive dt, so the two clocks
        # disagreeing costs nothing.
        now = time.monotonic()
        previous_tick = self._last_filtered_at
        self._last_filtered_at = now
        # None on the first tick after an engage/episode edge: there is no
        # previous tick to measure from, so LeaderFilter falls back to
        # ``output_dt``.  Harmless in practice — the filter was just reset, so
        # that tick returns the leader verbatim and ignores dt entirely — but
        # stated explicitly rather than left to that coincidence.
        tick_dt = None if previous_tick is None else now - previous_tick
        xi = self._expert_delta_xi(self._filter.filtered(q_lead, tick_dt))
        allowed, exhausted = self._budget.take(self._paced_request(xi))
        if exhausted and not np.any(allowed):
            # Budget spent: stop refreshing the target so the 250 Hz upsampler
            # decelerates into its own HOLD instead of being handed the same
            # goal again (which would also keep resetting the backend's
            # target_stale_s brake).
            #
            # ``exhausted and`` is load-bearing: a leader held perfectly still
            # produces a legitimately ZERO request, which is not a spent budget.
            # Ending the window there would stop refreshing the target every
            # time the operator paused, and would mislabel it in the log.
            return {"issued": False, "stop": "BUDGET_EXHAUSTED", "brake": False}

        unwrapped = self.env.unwrapped
        # Through the mux, like every other joint command: the in-window path
        # has no concurrency of its own, but "every arm command goes through one
        # door" is only a property if there are no exceptions to point at.
        q_cmd, ctrl_info = unwrapped._emit_arm_command(
            allowed, dt, unwrapped.OWNER_POLICY
        )
        if q_cmd is None:  # somebody else owns the arm; issue nothing
            return {"issued": False, "stop": "NOT_COMMAND_OWNER", "brake": False}
        result = dict(ctrl_info)
        result["issued"] = True
        result["stop"] = "BUDGET_EXHAUSTED" if exhausted else None
        result["brake"] = False
        return result

    def consumed_window_action(self) -> np.ndarray:
        """The 7-D env action this window actually executed, in [-1,1].

        Arm axes: the budget's normalized cumulative allowance, i.e. the sum of
        every increment that was permitted to reach the controller during the
        window.  Gripper: the single 3-state command the window issued, decided
        once by ``_expert_gripper`` (the 0.6 s Robotiq debounce makes a per-
        substep gripper decision meaningless, so the gripper is untouched here).
        """
        consumed = np.asarray(
            self._budget.consumed_action(), dtype=np.float32
        ).reshape(-1)
        executed = np.zeros(7, dtype=np.float32)
        executed[:6] = consumed[:6]
        executed[6] = self._window_gripper
        # The budget bounds the norm, so this clip is a guard against float
        # round-off only; the actor's validate_action rejects anything outside
        # [-1,1] outright (ur_env/actor_network.py:273).
        return np.clip(executed, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    # UR7eEnv BACKGROUND-FOLLOWER protocol (follow_mode="background")      #
    #                                                                     #
    # Same division of labour as the substep driver above — the anchored   #
    # leader maths lives here, the pacing/handles/mux live in UR7eEnv —    #
    # but the caller is now a daemon thread that keeps following while the #
    # RL loop is off doing its gRPC round trip.  Three rules make that     #
    # safe, and all three are visible in the three methods below:          #
    #                                                                     #
    #   * this thread NEVER re-anchors and NEVER re-latches the gain.  It  #
    #     reports "I lost my anchor" by returning None and the RL thread   #
    #     re-engages at its next step, so the gain latch stays in exactly  #
    #     one place (``_engage``).                                         #
    #   * every read of the anchor triple is ONE locked snapshot.          #
    #   * the deadman is re-read on EVERY tick (``UR7eEnv._follow_tick``   #
    #     calls ``follow_is_engaged`` first), which is what takes a release #
    #     from "up to one 512 ms step period" down to one 33 ms tick.      #
    # ------------------------------------------------------------------ #
    def follow_is_engaged(self) -> bool:
        """May the background follower drive the arm right now?

        RAISES rather than returning False when the deadman's state is unknowable
        (``DeadmanHeartbeatStaleError``); ``UR7eEnv._follow_tick`` turns that into
        a braked, latched fault re-raised on the RL thread.  Returning False there
        would be reinterpreting "I cannot tell" as "the operator let go", which is
        the exact confusion that exception class exists to prevent.

        ``gain()`` is deliberately NOT touched: it is latched at the engage edge.

        Unanchored counts as not engaged — with no anchor there is nothing to
        follow, and this is how a ``_disengage`` on the RL thread stops the
        follower QUIETLY (a disarm, no brake) instead of provoking a HOLD on a
        perfectly ordinary policy hand-back.
        """

        engaged = bool(self.expert.is_engaged())
        if not engaged:
            return False
        return self._anchor_snapshot()[0] is not None

    def follow_xi(self, now: float) -> Optional[np.ndarray]:
        """One tick's task-space increment, UNBUDGETED, or None if unusable.

        ``None`` means "the leader cannot be used right now" — absent, stale, or
        non-finite — and the env answers it with a brake plus a re-anchor
        request.  Same rule as ``action()`` and ``substep()``: a GELLO dropout
        must never turn into motion, and a target already in flight is motion.

        NO ``InterventionBudget`` AND NO ``_paced_request``.  This is an operator
        decision, taken 2026-07-30, and it is the single largest difference from
        the in-window path — read it together with the cost note on
        ``UR7eEnv._harvest_follow_window``.

        The budget capped a window's displacement at one ``ACTION_SCALE`` step so
        the stored action could never understate the motion.  That is a RECORDING
        constraint enforced by throttling the ARM, and at the production 512 ms
        step period it set the operator's ceiling at 12.5 mm / 512 ms = 2.4 cm/s
        — against 12.4 cm/s on the validation rig.  Rationing it across the
        measured window length instead (the intermediate design) fixes the dead
        time but keeps that same ``ACTION_SCALE / T`` ceiling, and the operator's
        requirement is explicitly the other thing: "intervention 시는 그냥
        teleop이 서버 통신 시간과 상관없이 쭉 되는 거고, 정보만 그때그때 주는
        것".  An intervention path nobody can drive produces no demonstrations at
        all, so the ceiling had to go.

        WHAT STILL BOUNDS THE ARM, and it is now the whole list:

          * the GOVERNOR, per tick, inside ``controller.step``: ``v_max`` 0.15
            m/s and ``w_max`` 0.75 rad/s scaled by ``dt = 1/substep_hz``, plus
            the ``dq_step_max`` joint gate and its line search;
          * the WORKSPACE BOX, also inside ``controller.step`` (``clip_pose``),
            written back into the integrator so it is a hard wall and not a
            windup;
          * the 250 Hz acceleration-limited upsampler underneath.

        All three live below ``controller.step``, which is exactly why the
        follower must never compute a joint command any other way.  With the
        budget gone the governor is the only speed limit, so a 5 s gRPC stall now
        permits ~75 cm of free travel — 2.8x the measured workspace box's x
        extent.  The box is what stops that, and it only exists on this path
        because every tick goes through ``UR7eEnv._emit_arm_command`` ->
        ``controller.step``.
        """

        q_lead, _grip, age = self.expert.get_leader()
        if (
            q_lead is None
            or age > self.LEADER_STALE_S
            or not np.all(np.isfinite(q_lead))
        ):
            return None

        with self._follow_lock:
            if self._anchor_snapshot()[0] is None:
                return None
            self._note_leader_sample(q_lead, age, now=now)
            previous_tick = self._last_filtered_at
            self._last_filtered_at = now
            # The wall-clock interval since the previous output tick, for the
            # same reason substep() measures it: alpha = 1/(1 + tau/dt) is only
            # a time constant if dt is time.  None on the first tick after an
            # engage/episode edge, where there is no previous tick to measure.
            tick_dt = None if previous_tick is None else now - previous_tick
            return self._expert_delta_xi(self._filter.filtered(q_lead, tick_dt))

    def _open_substep_window(self, replaced: bool, new_action) -> None:
        """Arm one window of substepping, or leave the env on its old path.

        Skipped — leaving ``UR7eEnv.step`` byte-identical to its pre-change
        behaviour — when substepping is off, when the policy is driving, when
        this step is a HOLD (the anchor is gone and ``_apply_action`` will brake
        instead), or when the wrapped env does not implement the protocol at all
        (fake/stub envs).
        """
        begin = getattr(self.env.unwrapped, "begin_intervention_window", None)
        if (
            begin is None
            or self._budget is None
            or not replaced
            or self._hold_requested
            or not self._anchored
        ):
            return
        self._window_gripper = float(np.asarray(new_action).reshape(-1)[6])
        self._budget.begin_window()
        begin(self)

    def _update_follow_arming(self, replaced: bool, new_action) -> None:
        """Promote or demote the background follower.  RL thread, step boundary.

        The ONLY promotion site in the codebase, deliberately: starting to follow
        is a decision that needs the anchor, the gain latch and the deadman read
        that ``action()`` has just done, and all three are RL-thread state.
        Demotion has no such requirement and happens from anywhere.

        ``env.arm_intervention_follow`` returning False (a fake env, an env
        without the follower, ``follow_mode="in_window"``) leaves the wrapper on
        whatever path that env does support — the same graceful-degradation rule
        ``_open_substep_window`` follows for stub envs.
        """

        unwrapped = self.env.unwrapped
        arm = getattr(unwrapped, "arm_intervention_follow", None)
        if arm is None:
            return
        if replaced and not self._hold_requested and self._anchored:
            self._window_gripper = float(np.asarray(new_action).reshape(-1)[6])
            arm(self)
        else:
            unwrapped.disarm_intervention_follow()

    def step(self, action):
        # Copy before the wrapped env sees the array: the policy output is a
        # counterfactual record during intervention and must not alias either
        # the caller's buffer or the executed human action.
        policy_action = np.asarray(action).copy()
        # BEFORE action(): a background brake since the last step re-anchored the
        # controller's integrator on its braking endpoint, so this wrapper's
        # T_r_anchor describes a pose the arm is no longer commanded to.  Dropping
        # it here means action() re-engages with a fresh anchor and its first
        # delta is exactly zero; learning about it only from the returned info
        # would arm the follower for one whole window on the stale anchor, whose
        # first command is the entire braking distance.
        if self._follow_mode == "background":
            consume = getattr(
                self.env.unwrapped, "consume_follow_reanchor", None
            )
            if consume is not None and consume():
                self._disengage()
        new_action, replaced = self.action(action)
        if self._hold_requested:
            self.env.unwrapped.request_hold("GELLO_STALE")
        if self._follow_mode == "background":
            self._update_follow_arming(replaced, new_action)
        else:
            self._open_substep_window(replaced, new_action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        # WHY THE EXECUTED ACTION COMES BACK THROUGH info AND NOT FROM action().
        # gym's ActionWrapper convention fixes the action BEFORE env.step runs.
        # With substepping, the displacement the window executes is only known
        # once the window closes, so that convention cannot express it: the one
        # thing the storage invariant forbids is reporting a number the robot
        # did not run (README.md:197-205).  UR7eEnv therefore harvests
        # `consumed_window_action()` at the end of the window — after the last
        # substep, before it reads obs_{k+1} — and hands it back here.  The
        # (obs_k, action_k, obs_{k+1}) triple stays aligned because the harvest
        # sits between the last command of window k and that observation.
        #
        # No key means no substep refreshed the target (policy step, stub env,
        # HOLD, or a window with no room), and then the single action passed
        # down IS what ran — the pre-change record, bit-for-bit.
        window_action = info.pop("intervention_window_action", None)
        # The background follower's record: the displacement the controller
        # actually COMMITTED between obs_k and obs_{k+1}, which is a strictly
        # better answer than the budget's charge (the charge bills the request,
        # so an IK line-search shrink is over-reported — the standing xfail on
        # the in-window path) and than ``new_action`` (which, on this path, the
        # mux never let reach the arm at all).
        committed = info.pop("intervention_committed_action", None)
        if (
            info.pop("intervention_window_braked", False)
            and self._follow_mode != "background"
        ):
            # The window braked and re-anchored the controller's integrator, so
            # this wrapper's T_r_anchor no longer describes the commanded pose.
            # Drop it; the next engaged step re-anchors with a zero first delta,
            # exactly as after an action()-time GELLO_STALE hold.
            #
            # In BACKGROUND mode the same brake was already consumed at the top
            # of this method, before action() re-anchored — disengaging again
            # here would throw away the anchor that step just built and cost an
            # extra window of policy control.
            self._disengage()
        if replaced:
            if committed is not None:
                executed = np.zeros(7, dtype=np.float32)
                executed[:6] = np.asarray(
                    committed, dtype=np.float32
                ).reshape(-1)[:6]
                # The gripper is decided ONCE per window by _expert_gripper (the
                # 0.6 s Robotiq debounce makes a per-tick decision meaningless),
                # exactly as consumed_window_action does for the other path.
                executed[6] = self._window_gripper
                info["intervene_action"] = executed
            else:
                info["intervene_action"] = (
                    np.asarray(new_action).copy()
                    if window_action is None
                    else np.asarray(window_action, dtype=np.float32)
                )
        info["policy_action"] = policy_action
        info["intervened"] = int(replaced)
        # spacemouse-button fields kept for script compatibility
        info["left"] = False
        info["right"] = False
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self._disengage()
        self._grip_cmd = 0.0
        self._hold_requested = False
        self._window_gripper = 0.0
        # The budget object survives (it is rearmed per window by begin_window);
        # what must not survive an episode boundary is filter state, for the same
        # reason as at an engage edge.
        self._reset_leader_filter()
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.Wrapper):
    """Penalize only redundant discrete gripper commands.

    Place this outside ``GelloIntervention`` and after the canonical
    observation wrappers.  It therefore sees ``state`` as ``(1, 19)`` and can
    use ``info['intervene_action']`` whenever a human action replaced the
    policy action.  The penalty describes the action that physically ran, not
    the counterfactual policy proposal.
    """

    def __init__(
        self,
        env,
        *,
        penalty: float = -0.02,
        open_threshold: float = 0.15,
        closed_threshold: float = 0.85,
    ):
        super().__init__(env)
        if tuple(env.action_space.shape) != (7,):
            raise ValueError("GripperPenaltyWrapper requires a 7D action space")
        self.penalty = float(penalty)
        if not math.isfinite(self.penalty) or self.penalty > 0.0:
            raise ValueError("penalty must be finite and non-positive")
        self.open_threshold = float(open_threshold)
        self.closed_threshold = float(closed_threshold)
        if not (
            0.0
            <= self.open_threshold
            < self.closed_threshold
            <= 1.0
        ):
            raise ValueError(
                "gripper thresholds must satisfy 0 <= open < closed <= 1"
            )
        self._last_gripper_position: Optional[float] = None

    @staticmethod
    def _gripper_position(observation) -> float:
        if not isinstance(observation, dict) or "state" not in observation:
            raise ValueError("canonical observation with state is required")
        position = gripper_position_from_state(observation["state"])
        if not math.isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("gripper position must be finite and in [0, 1]")
        return position

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_gripper_position = self._gripper_position(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._last_gripper_position is None:
            raise RuntimeError("GripperPenaltyWrapper.step called before reset")
        result_info = dict(info)
        executed = np.asarray(
            result_info.get("intervene_action", action)
        )
        if executed.shape != (7,) or executed.dtype.kind not in "fiu":
            raise ValueError("executed gripper action must have shape (7,)")
        if not np.all(np.isfinite(executed)):
            raise ValueError("executed gripper action must be finite")

        gripper_action = float(executed[-1])
        redundant_close = (
            gripper_action < -0.5
            and self._last_gripper_position >= self.closed_threshold
        )
        redundant_open = (
            gripper_action > 0.5
            and self._last_gripper_position <= self.open_threshold
        )
        result_info["grasp_penalty"] = (
            self.penalty if redundant_close or redundant_open else 0.0
        )
        self._last_gripper_position = self._gripper_position(observation)
        return observation, reward, terminated, truncated, result_info


def _validated_grasp_penalty(value, *, source: str) -> float:
    """Return one task penalty without accepting implicit/bool coercions."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{source}.GRASP_PENALTY must be a numeric scalar")
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf":
        raise ValueError(f"{source}.GRASP_PENALTY must be a numeric scalar")
    penalty = float(array)
    if not math.isfinite(penalty) or penalty > 0.0:
        raise ValueError(
            f"{source}.GRASP_PENALTY must be finite and non-positive"
        )
    return penalty


def wrap_gripper_penalty_from_task_config(env, *, experiment_config=None):
    """Apply the robot task's explicit learned-gripper penalty contract.

    The environment's unwrapped robot config is authoritative because it is
    the config that controls the physical task.  An experiment config may
    repeat ``GRASP_PENALTY`` for convenience; when it does, disagreement is a
    startup error instead of silently selecting one value.  There is
    deliberately no ``-0.02`` fallback here: learner-mode ingress requires
    every actor transition to carry a penalty produced by an explicit task
    configuration.
    """

    unwrapped = getattr(env, "unwrapped", None)
    task_config = getattr(unwrapped, "config", None)
    if task_config is None or not hasattr(task_config, "GRASP_PENALTY"):
        raise ValueError(
            "env.unwrapped.config.GRASP_PENALTY is required for the "
            "learned-gripper actor"
        )
    penalty = _validated_grasp_penalty(
        getattr(task_config, "GRASP_PENALTY"),
        source="env.unwrapped.config",
    )

    if experiment_config is not None and hasattr(
        experiment_config, "GRASP_PENALTY"
    ):
        configured = _validated_grasp_penalty(
            getattr(experiment_config, "GRASP_PENALTY"),
            source="experiment config",
        )
        if not math.isclose(configured, penalty, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                "experiment config GRASP_PENALTY disagrees with "
                "env.unwrapped.config.GRASP_PENALTY"
            )

    if isinstance(env, GripperPenaltyWrapper):
        if not math.isclose(
            env.penalty, penalty, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(
                "existing GripperPenaltyWrapper penalty disagrees with "
                "env.unwrapped.config.GRASP_PENALTY"
            )
        return env
    return GripperPenaltyWrapper(env, penalty=penalty)
