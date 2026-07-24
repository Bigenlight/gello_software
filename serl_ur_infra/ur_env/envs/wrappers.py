"""GELLO intervention wrapper — the SpacemouseIntervention analog.

Contract kept identical to franka_env.envs.wrappers.SpacemouseIntervention so
train_rlpd.py / record_demos.py work unchanged: when the human is intervening,
step() executes the human action instead of the policy action and reports it
in info["intervene_action"] (in env action units, [-1,1]^7).

Semantics differ from the spacemouse in one fundamental way: GELLO is an
absolute-pose device with no zero-rest, so intervention is gated by an
explicit DEADMAN (hold-to-engage), not by output magnitude.

    deadman press  -> engage: latch (leader TCP, robot commanded TCP) anchors
    while held     -> expert action = anchored delta, re-expressed as a
                      per-step env delta:  xi = log(T_cmd^-1 @ T_des),
                      clipped to the action space (clipping = natural chase
                      rate limit; correspondence recovers over a few steps)
    deadman release-> disengage, policy resumes instantly

TODO(together):
- deadman hardware: pynput key hold for now; USB footswitch later.
- success/fail marking keys for record_success_fail.py (spacemouse buttons
  there; we need two more keys).
- T_tool handling: anchor math below uses flange poses; wire the real
  T_tool_L/R from the bridge config for TCP-accurate mapping.
"""

import threading
import time
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

try:
    from ur_gello_bringup.ur_kin import fk, so3_log

    _KIN_AVAILABLE = True
except ImportError:
    _KIN_AVAILABLE = False


# ---------------------------------------------------------------------------- #
# Deadman sources — the "engaged" signal + a sensitivity gain can come from     #
# either the spacebar (default) or a ROS topic published by the HIL GUI.        #
# ---------------------------------------------------------------------------- #
class DeadmanSource:
    """Interface: the human deadman 'engaged' signal + a sensitivity gain.

    gain() == 1.0 means 1:1 leader->robot translation. Implementations must be
    thread-safe: is_engaged()/gain() are read from the env step thread while
    the underlying state is written from a listener / ROS callback thread.
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


class RosTopicDeadman(DeadmanSource):
    """Deadman driven by /hil/deadman, published by the HIL GUI.

    FROZEN CONTRACT (shared with the GUI — do NOT deviate):
      topic  /hil/deadman        type std_msgs/Float32MultiArray
      data   [engaged, gain]      engaged in {0.0, 1.0}; gain in [0.10, 1.00]
      rate   20 Hz heartbeat      (published continuously, not only on change)
      QoS    default reliable, depth 10
      stale  newest msg older than STALE_S -> engaged=False (fail-safe to policy)

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
                return False  # no msg yet -> fail safe to policy
            if time.monotonic() - self._last_rx > self.STALE_S:
                return False  # staleness watchdog -> fail safe to policy
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
        """Returns (q_lead(6,), gripper in [0,1] or None, age_seconds)."""
        arr, age = self._backend.get_gello_state()
        if arr is None:
            return None, None, age
        q = np.asarray(arr[:6], dtype=float)
        grip = float(arr[6]) if len(arr) > 6 else None
        return q, grip, age


class GelloIntervention(gym.ActionWrapper):
    LEADER_STALE_S = 0.3  # leader older than this -> refuse to intervene
    GRIP_CLOSE_THR = 0.7  # leader trigger hysteresis
    GRIP_OPEN_THR = 0.3

    def __init__(self, env, deadman: Optional[DeadmanSource] = None):
        super().__init__(env)
        if not _KIN_AVAILABLE:
            raise RuntimeError("ur_gello_bringup not importable")
        self.expert = GelloExpert(env.unwrapped.backend, deadman=deadman)
        self.action_scale = env.unwrapped.action_scale

        self._anchored = False
        self.T_g_anchor: Optional[np.ndarray] = None
        self.T_r_anchor: Optional[np.ndarray] = None
        self._grip_cmd = 0.0  # last hysteresis output in {-1, 0, +1}
        self._gain = 1.0      # sensitivity gain, latched at each engage edge

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

        act = np.concatenate(
            [p_err / self.action_scale[0], w_err / self.action_scale[1]]
        )
        return np.clip(act, -1.0, 1.0)

    def _expert_gripper(self, grip: Optional[float]) -> float:
        if grip is None:
            return 0.0
        if grip >= self.GRIP_CLOSE_THR:
            self._grip_cmd = -1.0
        elif grip <= self.GRIP_OPEN_THR:
            self._grip_cmd = 1.0
        return self._grip_cmd

    # ------------------------------------------------------------------ #
    def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        q_lead, grip, age = self.expert.get_leader()

        if not self.expert.is_engaged() or q_lead is None or age > self.LEADER_STALE_S:
            if self._anchored:
                self._disengage()
            return action, False

        if not self._anchored:
            self._engage(q_lead)

        expert_a = np.zeros(7, dtype=np.float32)
        expert_a[:6] = self._expert_delta_action(q_lead)
        expert_a[6] = self._expert_gripper(grip)
        return expert_a, True

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        # spacemouse-button fields kept for script compatibility
        info["left"] = False
        info["right"] = False
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self._disengage()
        self._grip_cmd = 0.0
        return self.env.reset(**kwargs)
