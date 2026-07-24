"""UR7e single-arm gym environment for HIL-SERL.

Replicates the observation/action contract of
franka_env.envs.franka_env.FrankaEnv so the entire hil-serl stack — wrapper
chain (RelativeFrame, Quat2EulerWrapper, SERLObsWrapper, ChunkingWrapper,
reward-classifier wrappers), train_rlpd / record_demos actor loops, and
per-task EnvConfig subclassing — runs unchanged.

What is different underneath (and why this file exists):

    FrankaEnv                          UR7eEnv
    ---------                          -------
    Flask HTTP robot server            URRosBackend (rclpy, ur_gello_bringup graph)
    impedance controller absorbs       PolicyDeltaController synthesizes softness
      rough targets in hardware          (governor -> IK -> gates -> HOLD)
    COMPLIANCE/PRECISION params        no compliance; conservative rate/step caps
    franka error recovery              TODO: UR dashboard/fault handling

Validation status (2026-07-24):
- OFFLINE-TESTED: the full control/observation path (scale -> governor -> IK ->
  gates -> commands -> fk obs) passes tests/test_env_fake_backend.py on a
  ROS-free machine (world-frame delta semantics, gripper scale/direction,
  10 Hz pacing, upsampler convergence all asserted).
- NOT yet validated on a real ROS graph — run tests/run_rviz_fake_rl.py on the
  robot laptop against mock hardware first (QoS, topic wiring; see VERIFY(hw)
  comments), and keep config.DRY_RUN=True near a real robot until the
  PolicyDeltaController is upgraded to the full eef_delta gate stack
  (README: "PolicyDeltaController 현황 vs eef_delta 본체").
"""

import queue
import threading
import time
from typing import Dict

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from ur_env.envs.config import DefaultUR7eEnvConfig
from ur_env.envs.policy_delta_controller import PolicyDeltaController


class ImageDisplayer(threading.Thread):
    """Live camera window (ported 1:1 from franka_env.envs.franka_env).

    Shows every camera view side by side at the policy's input resolution —
    this is the screen the human watches to time interventions.
    """

    def __init__(self, img_queue, name):
        threading.Thread.__init__(self)
        self.queue = img_queue
        self.daemon = True
        self.name = name

    def run(self):
        import cv2

        while True:
            img_array = self.queue.get()
            if img_array is None:  # exit signal
                break
            frame = np.concatenate(
                [v for k, v in img_array.items() if "full" not in k], axis=1
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


def _pose6_to_pose7(pose6: np.ndarray) -> np.ndarray:
    """xyz+rpy -> xyz+quat (FrankaEnv's tcp_pose convention)."""
    return np.concatenate(
        [pose6[:3], Rotation.from_euler("xyz", pose6[3:]).as_quat()]
    )


class UR7eEnv(gym.Env):
    def __init__(
        self,
        hz: float = None,
        fake_env: bool = False,
        save_video: bool = False,
        config: DefaultUR7eEnvConfig = None,
        backend=None,
    ):
        """backend: inject a URRosBackend-compatible object (tests use a
        ROS-free fake; ur_kin/eef_delta are pure numpy so the full control
        loop runs without ROS). None -> construct the real URRosBackend."""
        self.config = config or DefaultUR7eEnvConfig()
        self.hz = hz or self.config.HZ
        self.action_scale = np.asarray(self.config.ACTION_SCALE, dtype=float)

        # Buffer-correctness invariant (see config.ACTION_SCALE): a full-scale
        # action must never be cut by the governor, or stored transitions
        # (policy and intervention alike) overstate the executed motion.
        gov = self.config.GOVERNOR
        if self.action_scale[0] * self.hz > gov["v_max"]:
            print(
                f"[UR7eEnv] WARNING: ACTION_SCALE pos*hz "
                f"({self.action_scale[0] * self.hz:.3f} m/s) exceeds governor "
                f"v_max ({gov['v_max']}) — recorded actions will overstate motion"
            )
        if self.action_scale[1] * self.hz > gov["w_max"]:
            print(
                f"[UR7eEnv] WARNING: ACTION_SCALE rot*hz "
                f"({self.action_scale[1] * self.hz:.3f} rad/s) exceeds governor "
                f"w_max ({gov['w_max']}) — recorded actions will overstate motion"
            )
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH
        self._TARGET_POSE = np.asarray(self.config.TARGET_POSE, dtype=float)
        self._REWARD_THRESHOLD = np.asarray(self.config.REWARD_THRESHOLD, dtype=float)
        self.resetpos7 = _pose6_to_pose7(np.asarray(self.config.RESET_POSE, dtype=float))
        self.save_video = save_video
        self.recording_frames = []

        # workspace safety box (same semantics as FrankaEnv.clip_safety_box)
        self.xyz_bounding_box = gym.spaces.Box(
            np.asarray(self.config.ABS_POSE_LIMIT_LOW[:3], dtype=np.float64),
            np.asarray(self.config.ABS_POSE_LIMIT_HIGH[:3], dtype=np.float64),
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.asarray(self.config.ABS_POSE_LIMIT_LOW[3:], dtype=np.float64),
            np.asarray(self.config.ABS_POSE_LIMIT_HIGH[3:], dtype=np.float64),
            dtype=np.float64,
        )

        # ---- spaces: byte-for-byte the FrankaEnv contract ---- #
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        # UR e-series built-in TCP F/T estimate (zeros when the
                        # force_torque broadcaster is absent, e.g. mock hardware)
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0,
                            255,
                            shape=(*self.config.IMAGE_OBS_SIZE, 3),
                            dtype=np.uint8,
                        )
                        for key in self.config.CAMERAS
                    }
                ),
            }
        )

        self.curr_path_length = 0
        self.currpos = self.resetpos7.copy()  # tcp_pose (7,) xyz+quat
        self.currvel = np.zeros(6)            # tcp twist [v, w], base frame
        self.curr_q = np.zeros(6)
        self.curr_gripper_pos = np.zeros(1)
        self.curr_force = np.zeros(3)
        self.curr_torque = np.zeros(3)
        self.terminate = False
        self.fake_env = fake_env

        if fake_env:
            return

        # ---- real mode ---- #
        from ur_gello_bringup import ur_kin

        self._kin = ur_kin
        # flange -> TCP transform (bridge tool_r_xyz_rpy convention)
        tool = np.asarray(self.config.TCP_OFFSET_XYZ_RPY, dtype=float)
        self.T_tool = np.eye(4)
        self.T_tool[:3, :3] = Rotation.from_euler("xyz", tool[3:]).as_matrix()
        self.T_tool[:3, 3] = tool[:3]

        if backend is not None:
            self.backend = backend
        else:
            from ur_env.envs.ros_backend import URRosBackend

            self.backend = URRosBackend(
                self.config.ROS,
                camera_topics=self.config.CAMERAS,
                upsampler_cfg=self.config.UPSAMPLER,
                dry_run=self.config.DRY_RUN,
            )
        self.controller = PolicyDeltaController(self.config.GOVERNOR, self.hz)
        self.last_gripper_act = time.time()

        if self.config.DISPLAY_IMAGE:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, "ur7e_env")
            self.displayer.start()

        # ESC to terminate, same as FrankaEnv (headless/test runs: no X, skip)
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        except Exception as e:
            print(f"[UR7eEnv] keyboard listener unavailable ({e}) — no ESC stop")

    # ------------------------------------------------------------------ #
    # step / reset                                                        #
    # ------------------------------------------------------------------ #
    def _action_to_xi(self, action: np.ndarray) -> np.ndarray:
        """Normalized action [-1,1]^7 -> task-space increment xi (6,) in
        real units ([m, m, m, rad, rad, rad] per step)."""
        return np.concatenate(
            [
                action[:3] * self.action_scale[0],
                action[3:6] * self.action_scale[1],
            ]
        )

    def _apply_action(self, action: np.ndarray) -> dict:
        """Action post-processing + dispatch: scale -> governor/IK gates ->
        publish arm and gripper commands. Returns the controller info dict
        (held / reject_reason)."""
        xi = self._action_to_xi(action)
        q_cmd, ctrl_info = self.controller.step(xi)
        # workspace box: clamp is applied on the *commanded* TCP pose.
        # TODO(together): fold clip_safety_box into the controller gates so
        # a clamped target re-solves IK instead of holding.
        self.backend.send_joint_command(q_cmd)
        self._send_gripper_command(action[6] * self.action_scale[2])
        return ctrl_info

    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        ctrl_info = {}
        if not self.fake_env:
            ctrl_info = self._apply_action(action)

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = (
            self.curr_path_length >= self.max_episode_length
            or bool(reward)
            or self.terminate
        )
        # held/reject_reason surfaced for operator display and logging; extra
        # info keys are ignored by the hil-serl actor loop (it only pops
        # intervene_action/left/right and reads succeed).
        info = {"succeed": bool(reward)}
        info.update(ctrl_info)
        return ob, int(reward), done, False, info

    def reset(self, **kwargs):
        if self.save_video:
            self._save_video_recording()
        self.curr_path_length = 0
        self.terminate = False

        if not self.fake_env:
            self.go_to_reset()
            self.backend.reset_command_stream()  # next target re-seeds from measured q
            q, _, _ = self.backend.get_joint_state()
            self.controller.reset(q)

        self._update_currpos()
        return self._get_obs(), {"succeed": False}

    def go_to_reset(self):
        """Stream to the fixed init pose (RESET_JOINTS) between episodes.

        Mechanism: reuse the 250 Hz upsampler — set it one far target and let
        its slew cap (~0.5 rad/s) pace the whole move. No controller switch;
        the same command path as normal steps.

        Long moves are refused (RESET_MAX_DIST_RAD): a streaming reset sweeps
        blindly through whatever lies on the way, so from a distant/unknown
        pose the operator pre-positions with the proven move-to-start tooling
        first.

        TODO(together): RANDOM_RESET (task-space offset around the init pose),
        UR fault recovery before moving.
        """
        target = np.asarray(self.config.RESET_JOINTS, dtype=float).reshape(6)

        q, _, _ = self.backend.get_joint_state()
        if q is None:
            raise RuntimeError("no /joint_states — cannot reset")
        gap = float(np.max(np.abs(q - target)))
        if gap > self.config.RESET_MAX_DIST_RAD:
            raise RuntimeError(
                f"reset distance {gap:.2f} rad exceeds "
                f"RESET_MAX_DIST_RAD={self.config.RESET_MAX_DIST_RAD} — "
                "pre-position the arm with move-to-start first"
            )

        self.backend.send_joint_command(target)
        if self.backend.dry_run:
            return  # nothing will move; don't wait for arrival

        deadline = time.time() + self.config.RESET_TIMEOUT_S
        while time.time() < deadline:
            q, _, age = self.backend.get_joint_state()
            if age < self.config.JOINT_STATE_STALE_S and np.max(
                np.abs(q - target)
            ) < self.config.RESET_TOLERANCE_RAD:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"reset did not arrive within {self.config.RESET_TIMEOUT_S}s "
            f"(remaining {np.max(np.abs(q - target)):.3f} rad)"
        )

    # ------------------------------------------------------------------ #
    # reward (default: pose proximity; classifier wrapper overrides)      #
    # ------------------------------------------------------------------ #
    def compute_reward(self, obs) -> bool:
        if not self._REWARD_THRESHOLD.any():
            return False
        current_pose = obs["state"]["tcp_pose"]
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_euler = Rotation.from_matrix(current_rot.T @ target_rot).as_euler("xyz")
        delta = np.abs(
            np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler])
        )
        return bool(np.all(delta < self._REWARD_THRESHOLD))

    # ------------------------------------------------------------------ #
    # state / obs                                                         #
    # ------------------------------------------------------------------ #
    def _update_currpos(self):
        """Refresh cached robot state from the backend (POST /getstate analog).

        tcp_pose source is config-selected (TCP_POSE_SOURCE):
          "driver" — /tcp_pose_broadcaster/pose, the path already proven by the
                     recorder on this rig.
          "fk"     — ur_kin.fk(q) @ T_tool, exactly consistent with the command
                     path's kinematics (eef_delta's same-fk rule).
        Note the intervention anchor math is self-consistent either way: it
        compares controller.tcp_cmd() against itself, never against this obs.
        """
        if self.fake_env:
            return

        q, dq, age = self.backend.get_joint_state()
        if q is None:
            raise RuntimeError("no /joint_states received yet")
        if age > self.config.JOINT_STATE_STALE_S:
            # TODO(together): safe-stop policy (freeze + operator prompt)
            # instead of raise, decided together with the camera-stale case.
            raise RuntimeError(f"/joint_states stale ({age:.2f}s)")
        self.curr_q = q

        if self.config.TCP_POSE_SOURCE == "driver":
            pose, pose_age = self.backend.get_tcp_pose()
            if pose is None:
                raise RuntimeError(
                    "no /tcp_pose_broadcaster/pose received — is the "
                    "tcp_pose_broadcaster in the launch? (or set "
                    'TCP_POSE_SOURCE="fk")'
                )
            if pose_age > self.config.JOINT_STATE_STALE_S:
                raise RuntimeError(f"tcp pose stale ({pose_age:.2f}s)")
            self.currpos = pose
        else:  # "fk"
            T = self._kin.fk(q) @ self.T_tool
            self.currpos = np.concatenate(
                [T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()]
            )

        # TCP twist in base frame: flange twist J(q)@dq, linear part shifted
        # to the TCP point (v_tcp = v_flange + w x r).
        twist = self._kin.jacobian(q) @ dq
        r = (self._kin.fk(q)[:3, :3]) @ self.T_tool[:3, 3]
        self.currvel = np.concatenate([twist[:3] + np.cross(twist[3:], r), twist[3:]])

        # Robotiq position_percent is already 0.0 (open) .. 1.0 (closed).
        pct, _ = self.backend.get_gripper_percent()
        self.curr_gripper_pos = np.array([pct if pct is not None else 0.0])

        # UR e-series built-in TCP F/T. Zeros when the broadcaster is absent
        # (mock hardware) — the obs keys stay valid either way.
        wrench, _ = self.backend.get_wrench()
        if wrench is not None:
            self.curr_force = wrench[:3].copy()
            self.curr_torque = wrench[3:].copy()

    def _get_obs(self) -> Dict:
        images = self.get_im() if not self.fake_env else {
            key: np.zeros(self.observation_space["images"][key].shape, dtype=np.uint8)
            for key in self.config.CAMERAS
        }
        state = {
            "tcp_pose": self.currpos.copy(),
            "tcp_vel": self.currvel.copy(),
            "gripper_pose": self.curr_gripper_pos.copy(),
            "tcp_force": self.curr_force.copy(),
            "tcp_torque": self.curr_torque.copy(),
        }
        return {"state": state, "images": images}

    # ------------------------------------------------------------------ #
    # gripper                                                             #
    # ------------------------------------------------------------------ #
    def _send_gripper_command(self, pos: float):
        """3-state gripper (FrankaEnv convention): action[6] <= -0.5 close,
        >= +0.5 open, else hold. Debounced; skips if already there.

        Why discrete, when the GELLO teleop line streams the trigger
        continuously (gello_gripper_bridge identity map): hil-serl's hybrid
        agent learns the gripper with a separate DISCRETE grasp critic
        (train_rlpd's grasp_critic branch), GripperPenaltyWrapper assumes
        open/close events, and the Robotiq physically takes ~0.5 s per
        actuation anyway (hence GRIPPER_SLEEP debounce). Keep this discrete
        unless the upstream agent changes.

        Scale/direction contract with robotiq_gripper_modbus_node:
        command_percent / position_percent are 0.0 = OPEN .. 1.0 = CLOSED.
        VERIFY(hw): on first hardware bring-up confirm direction and scale by
        eye — a silent inversion here is a crush-or-drop hazard."""
        now = time.time()
        if now - self.last_gripper_act < self.config.GRIPPER_SLEEP:
            return
        if pos <= -0.5 and self.curr_gripper_pos[0] < 0.85:  # close
            self.backend.send_gripper_percent(1.0)
            self.last_gripper_act = now
        elif pos >= 0.5 and self.curr_gripper_pos[0] > 0.15:  # open
            self.backend.send_gripper_percent(0.0)
            self.last_gripper_act = now

    # ------------------------------------------------------------------ #
    # cameras                                                             #
    # ------------------------------------------------------------------ #
    # Same downstream flow as FrankaEnv.get_im (crop -> resize -> RGB),
    # but frames come from the launch_cameras.sh realsense2_camera topics
    # via the backend instead of an in-process RSCapture — a RealSense
    # cannot be opened by two processes, and this keeps the existing
    # viewer/recorder workflow usable next to RL.
    def get_im(self) -> Dict[str, np.ndarray]:
        import cv2

        images = {}
        display_images = {}
        full_res = {}
        for key in self.config.CAMERAS:
            jpeg, age = self.backend.get_image(key)
            if jpeg is None or age > self.config.IMAGE_STALE_S:
                # TODO(together): safe-stop policy instead of raise (match
                # FrankaEnv's "frozen camera -> prompt & relaunch" behavior?)
                raise RuntimeError(
                    f"camera '{key}' has no fresh frame (age={age:.2f}s) — "
                    "is launch_cameras.sh running?"
                )
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            cropped = (
                self.config.IMAGE_CROP[key](bgr)
                if key in self.config.IMAGE_CROP
                else bgr
            )
            resized = cv2.resize(
                cropped, self.observation_space["images"][key].shape[:2][::-1]
            )
            images[key] = resized[..., ::-1]  # obs are RGB, FrankaEnv convention
            display_images[key] = resized     # BGR for cv2.imshow
            full_res[key] = cropped.copy()
        if self.save_video:
            self.recording_frames.append(full_res)
        if self.config.DISPLAY_IMAGE:
            self.img_queue.put(display_images)
        return images

    def _save_video_recording(self):
        # TODO: port FrankaEnv.save_video_recording if we want mp4 dumps
        self.recording_frames.clear()

    def close(self):
        if not self.fake_env:
            self.backend.close()
