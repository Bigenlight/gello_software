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

        # Full-resolution UNCROPPED BGR frames, keyed by camera name, refreshed
        # by get_im().  These exist so the reward classifier can be shipped the
        # whole scene while the policy keeps its measured IMAGE_CROP: one image
        # cannot serve both, and that mismatch cost recall@0.85 100% -> 33.3%.
        # get_im() decodes this array anyway, so stashing the reference costs
        # no CPU, no second decode and no copy.
        # See ur_env/classifier_sidecar.py for the consumer.
        # Declared before the fake_env early return so the accessor is always
        # safe to call; a fake env simply never fills it.
        self._last_camera_frames: Dict[str, np.ndarray] = {}

        # workspace safety box (same semantics as FrankaEnv.clip_safety_box)
        self._build_safety_box()

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
        self._hold_next_action_reason = None

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
        self._await_robot_state()
        self.controller = PolicyDeltaController(
            self.config.GOVERNOR, self.hz, clip_pose=self._clip_command_pose
        )
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
    # workspace safety box                                                #
    # ------------------------------------------------------------------ #
    def _build_safety_box(self) -> None:
        """Validate ABS_POSE_LIMIT_* and arm (or refuse to arm) the clip.

        Sets ``self.xyz_bounding_box`` / ``self.rpy_bounding_box`` (the two
        gym Boxes FrankaEnv.clip_safety_box reads) and
        ``self._safety_box_active``.

        REFUSE-DON'T-CLAMP.  ``DefaultUR7eEnvConfig`` ships zeros for both
        limits — "not measured yet", not "the workspace is a point at the base
        origin".  Clamping to a zero-volume box would command the TCP to
        (0,0,0) with rx=ry=rz=0, i.e. drive the arm straight down through its
        own base: a far worse outcome than no box at all.  The same is true of
        an inverted box (``np.clip`` with low>high silently returns ``high``,
        so every pose is teleported to one corner).  In both cases we disable
        the clip and say so loudly.  A task config that actually intends a box
        is expected to validate its own numbers and fail at construction —
        ``ur_experiments/cube_in_cup.py`` does exactly that; this is the
        second line of defence for configs that don't.
        """
        low_cfg = getattr(self.config, "ABS_POSE_LIMIT_LOW", None)
        high_cfg = getattr(self.config, "ABS_POSE_LIMIT_HIGH", None)

        self.xyz_bounding_box = None
        self.rpy_bounding_box = None
        self._safety_box_active = False

        if low_cfg is None or high_cfg is None:
            print(
                "[UR7eEnv] WARNING: ABS_POSE_LIMIT_LOW/HIGH is None — workspace "
                "safety box DISABLED (commands are not clipped)."
            )
            return

        low = np.asarray(low_cfg, dtype=np.float64).reshape(-1)
        high = np.asarray(high_cfg, dtype=np.float64).reshape(-1)
        if low.shape != (6,) or high.shape != (6,):
            print(
                f"[UR7eEnv] WARNING: ABS_POSE_LIMIT_* must be 6-vectors, got "
                f"{low.shape}/{high.shape} — workspace safety box DISABLED."
            )
            return
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            print(
                "[UR7eEnv] WARNING: ABS_POSE_LIMIT_* contains non-finite values "
                "— workspace safety box DISABLED."
            )
            return
        if not np.all(high > low):
            print(
                "[UR7eEnv] WARNING: ABS_POSE_LIMIT_HIGH must exceed "
                f"ABS_POSE_LIMIT_LOW on every axis (low={low}, high={high}) — "
                "this is a zero-volume or inverted workspace, NOT a workspace. "
                "Safety box DISABLED rather than clamping the arm to a point; "
                "measure the box (see ur_experiments/cube_in_cup.py)."
            )
            return
        # Index 3 is |rx|, not rx (see clip_safety_box) — a negative bound there
        # means the config author wrote a signed range and the magnitude clip
        # would do something they did not intend.
        if low[3] < 0.0 or high[3] <= 0.0:
            print(
                "[UR7eEnv] WARNING: ABS_POSE_LIMIT_*[3] is the bound on |rx| "
                f"(magnitude, always >= 0), got low={low[3]} high={high[3]} — "
                "workspace safety box DISABLED."
            )
            return

        self.xyz_bounding_box = gym.spaces.Box(low[:3], high[:3], dtype=np.float64)
        self.rpy_bounding_box = gym.spaces.Box(low[3:], high[3:], dtype=np.float64)
        self._safety_box_active = True

        # FRAME COUPLING.  The box is enforced on the pose PolicyDeltaController
        # integrates, which is fk(q) — the FLANGE.  With a non-zero TCP offset
        # the observation is the tool tip while the box is the flange, and the
        # two differ by the whole tool length with no error anywhere (this is
        # the 17.4 cm trap documented in ur_experiments/cube_in_cup.py).
        tool = np.asarray(self.config.TCP_OFFSET_XYZ_RPY, dtype=float)
        if np.any(tool != 0.0):
            print(
                "[UR7eEnv] WARNING: TCP_OFFSET_XYZ_RPY is non-zero but the "
                "workspace box is enforced on the FLANGE pose the controller "
                "integrates. Either measure the box in the flange frame or "
                "make PolicyDeltaController integrate at fk(q) @ T_tool."
            )

    def _clip_xyz_euler(self, xyz: np.ndarray, euler: np.ndarray):
        """The actual clip, in the representation the box is defined in.

        Ported verbatim from franka_env.envs.franka_env.FrankaEnv.
        clip_safety_box (upstream lines 185-207); both public entry points
        below go through here so they cannot drift apart.

        The rx special case is the whole reason this is not one np.clip: our
        tool points straight DOWN, so rx sits near +-pi and
        ``Rotation.as_euler("xyz")`` reports it on whichever branch is closer.
        12% of the cube_in_cup samples land on the negative branch. A naive
        ``np.clip(rx, 2.60, pi)`` maps rx=-3.14 to +2.60 — a 5.7 rad wrist
        flip, commanded as a "safety" measure. Upstream therefore clips the
        MAGNITUDE and restores the sign, which is continuous across the
        +-pi seam. Index 3 of ABS_POSE_LIMIT_* is consequently a bound on
        |rx|, not on rx.

        Upstream edge case kept as-is: ``np.sign(0.0) == 0`` leaves rx at 0.
        That pose is 180 deg from the operating branch and unreachable by the
        0.05 rad/step increments this controller emits; the alternative
        (forcing a sign) would command exactly the wrist flip the branch
        handling exists to prevent.
        """
        xyz_c = np.clip(xyz, self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        euler_c = np.asarray(euler, dtype=float).copy()
        sign = np.sign(euler_c[0])
        euler_c[0] = sign * np.clip(
            np.abs(euler_c[0]),
            self.rpy_bounding_box.low[0],
            self.rpy_bounding_box.high[0],
        )
        euler_c[1:] = np.clip(
            euler_c[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        return xyz_c, euler_c

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip a COMMANDED tcp_pose (7,) xyz+quat into the workspace box.

        Same signature and semantics as FrankaEnv.clip_safety_box, with one
        deliberate difference: we return a copy instead of mutating the
        caller's array. Upstream can mutate safely because it only ever passes
        its own scratch ``self.nextpos``; our callers pass observations too.

        WHEN it is applied matches upstream exactly: on the pose being
        COMMANDED, not on the observation. Clipping the observation would lie
        to the policy about where the arm is; clipping the command is what
        keeps the arm inside the box. A no-op when the box is disabled.
        """
        pose = np.asarray(pose, dtype=float).copy()
        if not self._safety_box_active:
            return pose
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
        xyz_c, euler_c = self._clip_xyz_euler(pose[:3], euler)
        pose[:3] = xyz_c
        pose[3:] = Rotation.from_euler("xyz", euler_c).as_quat()
        return pose

    def _clip_command_pose(self, T: np.ndarray):
        """4x4 adapter handed to PolicyDeltaController as ``clip_pose``.

        Returns ``(T_clipped, clipped)``. When nothing is out of bounds the
        ORIGINAL matrix is returned untouched — a matrix->euler->matrix
        round-trip would inject ~1e-16 of drift into every single command and
        make ``clipped`` impossible to detect exactly.
        """
        if not self._safety_box_active:
            return T, False
        euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
        xyz_c, euler_c = self._clip_xyz_euler(T[:3, 3], euler)
        # np.clip returns the input value bit-for-bit when it is in range, and
        # sign*abs is exact, so equality here is exactly "nothing was clipped".
        if np.array_equal(xyz_c, T[:3, 3]) and np.array_equal(euler_c, euler):
            return T, False
        T_c = np.eye(4)
        T_c[:3, :3] = Rotation.from_euler("xyz", euler_c).as_matrix()
        T_c[:3, 3] = xyz_c
        return T_c, True

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
        """Action post-processing + dispatch: scale -> governor/box/IK gates ->
        publish arm and gripper commands. Returns the controller info dict
        (held / reject_reason / clipped).

        The workspace box is enforced INSIDE the controller, not here: it is
        applied to the commanded pose BEFORE IK, so a clamped target re-solves
        instead of being held, and the clamped pose becomes the controller's
        integrator state (see PolicyDeltaController.step for the anti-windup
        argument). ``self._clip_command_pose`` is the hook — this env still
        owns the box, the controller just asks it per tick.
        """
        if self._hold_next_action_reason is not None:
            reason = self._hold_next_action_reason
            self._hold_next_action_reason = None
            q_hold = self.backend.request_hold()
            self.controller.reset(q_hold)
            return {"held": True, "reject_reason": reason, "clipped": False}

        xi = self._action_to_xi(action)
        q_cmd, ctrl_info = self.controller.step(xi)
        self.backend.send_joint_command(q_cmd)
        self._send_gripper_command(action[6] * self.action_scale[2])
        return ctrl_info

    def request_hold(self, reason: str = "EXTERNAL_HOLD") -> None:
        """Make the next real step brake the command stream and re-anchor it."""
        self._hold_next_action_reason = str(reason)

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
        # held/reject_reason/clipped surfaced for operator display and logging;
        # extra info keys are ignored by the hil-serl actor loop (it only pops
        # intervene_action/left/right and reads succeed).
        #
        # "clipped" is not decoration: a policy fighting the workspace wall
        # looks exactly like a policy that has stopped learning — the action is
        # large, the arm does not move, and nothing in the reward explains it.
        # Silent clamping is the single hardest failure to diagnose after the
        # fact, so every clamped tick is reported. Present (False) even in fake
        # mode so log schemas do not depend on the run mode.
        info = {"succeed": bool(reward), "held": False, "clipped": False}
        info.update(ctrl_info)
        return ob, int(reward), done, False, info

    def _await_robot_state(self) -> None:
        """Block until the robot state stream is live, or fail with a diagnosis.

        DDS discovery plus the first /joint_states message takes on the order of
        a second, and reset() is the very first thing every caller does.  Without
        this wait the env raises "no /joint_states -- cannot reset" on a rig that
        is in fact perfectly healthy, which reads as a wiring fault and sends
        people looking at QoS.  Waiting here rather than in each entry point
        keeps the contract with the caller simple: once the constructor returns,
        the env is usable.
        """

        deadline = time.time() + float(self.config.ROBOT_STATE_WAIT_S)
        while time.time() < deadline:
            q, _, _ = self.backend.get_joint_state()
            if q is not None:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                f"no /joint_states within {self.config.ROBOT_STATE_WAIT_S}s of "
                f"subscribing to '{self.config.ROS.get('joint_states_topic', '/joint_states')}'. "
                "Is the UR driver running (ros2 topic hz /joint_states), and does "
                "this process share its ROS_DOMAIN_ID?"
            )
        # Its OWN budget, not whatever the joint-state wait left over. DDS
        # discovery can eat most of ROBOT_STATE_WAIT_S, which would leave the
        # cameras near-zero seconds and time them out on a healthy rig -- the
        # exact failure this wait exists to prevent.
        self._await_first_frames(
            time.time() + float(self.config.ROBOT_STATE_WAIT_S)
        )

    def _await_first_frames(self, deadline: float) -> None:
        """Block until every configured camera has delivered a frame.

        Same argument as the joint-state wait above, and the same failure it
        prevents: get_im() rejects any frame older than IMAGE_STALE_S (0.5 s),
        but a RealSense needs seconds to produce its first one.  Without this,
        reset() -- the first thing every caller does -- throws on a healthy rig
        that has simply not finished starting its cameras.

        The hasattr guard is load-bearing, not defensive: fake/stub backends in
        the test suite implement get_joint_state() without get_image().
        """

        cameras = getattr(self.config, "CAMERAS", None)
        if not cameras or not hasattr(self.backend, "get_image"):
            return
        pending = list(cameras)
        while time.time() < deadline:
            pending = [
                key for key in pending if self.backend.get_image(key)[0] is None
            ]
            if not pending:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"no frame from camera(s) {pending} within "
            f"{self.config.ROBOT_STATE_WAIT_S}s. Is launch_cameras.sh running, "
            "and do the configured serials match the connected cameras?"
        )

    def reset(self, **kwargs):
        if self.save_video:
            self._save_video_recording()
        self.curr_path_length = 0
        self.terminate = False
        self._hold_next_action_reason = None

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

        The target is first mapped onto the arm's CURRENT turn (branch cut)
        so no joint is ever sent the long way round — see the block comment
        below; this is a cable-winding hazard, not a cosmetic detail.

        TODO(together): RANDOM_RESET (task-space offset around the init pose),
        UR fault recovery before moving.
        """
        target = np.asarray(self.config.RESET_JOINTS, dtype=float).reshape(6)

        q, _, _ = self.backend.get_joint_state()
        if q is None:
            raise RuntimeError("no /joint_states — cannot reset")

        # ---- BRANCH CUT: map the target onto the arm's current turn ---- #
        # Do NOT "simplify" this back to a raw abs(q - target). A joint angle
        # sent to forward_position_controller is a LITERAL NUMBER, not an angle:
        # the controller interpolates linearly in raw joint space with no 2*pi
        # awareness (ur_gello_bringup/angle_utils.py opens with this exact
        # warning). So target and q can be ~2*pi apart as numbers while being
        # the same physical pose.
        #
        # This task lives right on that seam. cube_in_cup's RESET_JOINTS has
        # wrist_3 at -3.1331 and shoulder_pan ~0.003 rad past +pi; a measured
        # parked pose on this rig had wrist_3 at +3.1795 — physically 0.029 rad
        # from the target, numerically 6.3126 apart. Two consequences, both real:
        #
        #   1. Commanding the raw -3.1331 sends wrist_3 the LONG way: a full
        #      6.31 rad revolution instead of 0.03 rad, ~10 s of blind slew,
        #      and the Robotiq 2F-85 tool-comm cable wound once around the
        #      wrist (hazard H3 in this repo's docs). DRY_RUN is the only
        #      reason the measured case did not do this.
        #   2. The guard below reported "6.31 rad" for a move whose true worst
        #      joint is 3.53 — which reads as a wiring fault and sends the
        #      operator looking in the wrong place.
        #
        # ur_kin.wrapped_nearest picks, per joint, the 2*pi-equivalent of the
        # target nearest the CURRENT q. It deliberately leaves the elbow
        # (index 2) alone: the elbow's range is +-pi, so its "shorter" wrapped
        # target would sit outside the feasible set. The elbow therefore keeps
        # its literal travel, which is its real travel — that is why the
        # branch-safe distance here is 3.53 and not the 2.75 a naive circular
        # distance reports.
        #
        # Everything downstream (distance guard, command, arrival test, error
        # messages) uses THIS target, never self.config.RESET_JOINTS: arrival is
        # convergence to what we actually commanded.
        target = self._kin.wrapped_nearest(target, q)
        if not self._kin.within_joint_limits(target, margin=0.0):
            raise RuntimeError(
                f"branch-mapped reset target {np.round(target, 4)} leaves the "
                f"joint limits (current q={np.round(q, 4)}) — the arm is on a "
                "turn from which RESET_JOINTS is not reachable without "
                "unwinding; pre-position with move-to-start first"
            )

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

        # The raw abs() below is correct BECAUSE `target` is the branch-mapped
        # one: the driver reports joint positions continuously within each
        # joint's +-2pi range and never re-wraps mid-move, so q converges to the
        # literal number we sent. Re-wrapping here would be worse than useless —
        # it would score a full-turn error as arrival.
        deadline = time.time() + self.config.RESET_TIMEOUT_S
        while time.time() < deadline:
            # The backend deliberately treats a target older than ~3 policy
            # periods as stale and brakes to HOLD.  Reset is the one command
            # which legitimately spans seconds, so refresh the same immutable
            # branch-mapped target while we wait; this changes no path or goal.
            self.backend.send_joint_command(target)
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
            # Keep the decoded frame BEFORE the crop on the next line.  The
            # policy observation below is cropped on purpose (config.IMAGE_CROP
            # is measured from the dataset); the reward classifier was trained
            # on the full frame and gets this uncropped one instead.
            #
            # A reference, not a copy, and that is safe: the crop is a numpy
            # slice (a read-only view over this same buffer), cv2.resize
            # allocates its own destination, and both `full_res` and the
            # display path copy before they keep anything.  Nothing below
            # writes back into `bgr`.  Were that ever to change, copy here and
            # accept the ~2.7 MB/frame memcpy.
            self._last_camera_frames[key] = bgr
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

    def last_camera_frames(self) -> Dict[str, np.ndarray]:
        """Full-resolution BGR frames behind the most recent :meth:`get_im`.

        UNCROPPED — the whole scene as the camera published it, only JPEG
        decoding applied.  ``classifier_sidecar.build_sidecar`` resizes these
        to 128x128 and re-encodes before they go on the wire; forwarding the
        camera's original JPEG was measured at ~400 KiB per attachment, which
        does not fit a 13 Mbit/s link inside a 100 ms step.

        Returns a fresh dict so a caller cannot re-key the env's state, but the
        arrays themselves are shared and are the same buffers ``get_im`` crops
        from.  **Treat them as read-only** — writing through one would corrupt
        the policy observation of the step it came from.

        Empty until :meth:`get_im` has run once, and always empty for a fake
        env, which has no camera at all.
        """
        return dict(self._last_camera_frames)

    def _save_video_recording(self):
        # TODO: port FrankaEnv.save_video_recording if we want mp4 dumps
        self.recording_frames.clear()

    def close(self):
        if not self.fake_env:
            self.backend.close()
