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

See the "UR7eEnv substep-driver protocol" section below and
``UR7eEnv.begin_intervention_window``.  The POLICY path is untouched by all of
this, by construction: no driver is installed, so ``UR7eEnv.step`` takes the
same single-target-plus-sleep branch it always did.

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
        data = list(msg.data)
        engaged = data[0] if len(data) > 0 else 0.0
        gain = data[1] if len(data) > 1 else 1.0
        with self._lock:
            self._engaged_raw = float(engaged)
            self._gain = float(gain)
            self._last_rx = time.monotonic()

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
            return self._engaged_raw >= 0.5

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
        self._window_gripper = 0.0

    # ------------------------------------------------------------------ #
    def _leader_T(self, q_lead: np.ndarray) -> np.ndarray:
        # TODO: @ T_tool_L (flange-only for the skeleton)
        return fk(q_lead)

    def _robot_T_cmd(self) -> np.ndarray:
        return self.env.unwrapped.controller.tcp_cmd()

    def _engage(self, q_lead: np.ndarray):
        self.T_g_anchor = self._leader_T(q_lead)
        self.T_r_anchor = self._robot_T_cmd()
        # LATCH the sensitivity gain at the engage edge (not read live per-tick)
        # so a slider nudged mid-motion cannot discontinuously rescale the
        # in-progress anchored delta.  The substep loop reuses this latched
        # value too — it must NOT re-read expert.gain() per 30 Hz tick, which
        # would reintroduce exactly the discontinuity this latch removes.
        self._gain = self.expert.gain()
        self._anchored = True
        # Drop filter state at the engage edge: the leader could be anywhere
        # while the policy was driving, and low-passing across that gap would
        # ramp the arm toward a pose the human never asked for.  Dropping the
        # receive bookkeeping with it makes the next sample the filter's first.
        self._reset_leader_filter()

    def _disengage(self):
        self._anchored = False
        self.T_g_anchor = None
        self.T_r_anchor = None

    def _expert_delta_xi(self, q_lead: np.ndarray) -> np.ndarray:
        """Anchored leader delta -> task-space increment xi (6,), real units.

        ``[dpos (m), drot rotvec (rad)]`` in the WORLD/base frame — the same
        convention PolicyDeltaController expects (FrankaEnv semantics): position
        error along base axes, rotation error as a base-frame rotvec.

        Split out of ``_expert_delta_action`` so the 30 Hz substep loop can feed
        the controller directly without a normalize/re-scale round trip (which
        is not bit-exact and would make the recorded action drift away from the
        commanded one).  The maths below is byte-for-byte the previous body.
        """
        T_g = self._leader_T(q_lead)
        # world-frame delta (same convention as eef_delta.py: R_g @ R_anchor^T)
        R_delta = T_g[:3, :3] @ self.T_g_anchor[:3, :3].T
        p_delta = T_g[:3, 3] - self.T_g_anchor[:3, 3]

        T_des = np.eye(4)
        T_des[:3, :3] = R_delta @ self.T_r_anchor[:3, :3]
        # Apply the LATCHED sensitivity gain to the translation delta only
        # (rotation stays 1:1). This scaling happens BEFORE the /action_scale ->
        # clip below, so the returned `act` is exactly what env.step re-scales
        # and executes: the executed==reported buffer-correctness invariant
        # (README "저장 액션 불변식") is preserved — gain is upstream of the report.
        T_des[:3, 3] = self.T_r_anchor[:3, 3] + self._gain * p_delta

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
        if self._filter is not None:
            self._filter.reset()
        self._leader_rx = None

    def _note_leader_sample(self, q_lead: np.ndarray, age: float) -> None:
        """Feed the one-euro filter, but ONLY on a genuinely new leader sample.

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
        rx = time.monotonic() - age
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
        xi = self._expert_delta_xi(self._filter.filtered(q_lead))
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
        q_cmd, ctrl_info = unwrapped.controller.step(allowed, dt=dt)
        unwrapped.backend.send_joint_command(q_cmd)
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

    def step(self, action):
        # Copy before the wrapped env sees the array: the policy output is a
        # counterfactual record during intervention and must not alias either
        # the caller's buffer or the executed human action.
        policy_action = np.asarray(action).copy()
        new_action, replaced = self.action(action)
        if self._hold_requested:
            self.env.unwrapped.request_hold("GELLO_STALE")
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
        if info.pop("intervention_window_braked", False):
            # The window braked and re-anchored the controller's integrator, so
            # this wrapper's T_r_anchor no longer describes the commanded pose.
            # Drop it; the next engaged step re-anchors with a zero first delta,
            # exactly as after an action()-time GELLO_STALE hold.
            self._disengage()
        if replaced:
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
