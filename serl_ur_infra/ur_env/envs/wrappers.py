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

from ur_env.observation_schema import gripper_position_from_state

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
        # in-progress anchored delta.
        self._gain = self.expert.gain()
        self._anchored = True

    def _disengage(self):
        self._anchored = False
        self.T_g_anchor = None
        self.T_r_anchor = None

    def _expert_delta_action(self, q_lead: np.ndarray) -> np.ndarray:
        """Anchored leader delta -> per-step env action (6,), in [-1,1].

        Output is a WORLD/base-frame delta toward the anchored target — the
        same convention PolicyDeltaController expects (FrankaEnv semantics):
        position error along base axes, rotation error as a base-frame rotvec.
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

        act = np.concatenate(
            [p_err / self.action_scale[0], w_err / self.action_scale[1]]
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

        expert_a = np.zeros(7, dtype=np.float32)
        expert_a[:6] = self._expert_delta_action(q_lead)
        expert_a[6] = self._expert_gripper(grip)
        return expert_a, True

    def step(self, action):
        # Copy before the wrapped env sees the array: the policy output is a
        # counterfactual record during intervention and must not alias either
        # the caller's buffer or the executed human action.
        policy_action = np.asarray(action).copy()
        new_action, replaced = self.action(action)
        if self._hold_requested:
            self.env.unwrapped.request_hold("GELLO_STALE")
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = np.asarray(new_action).copy()
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
