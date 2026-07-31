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

Validation status:
- OFFLINE-TESTED: the full control/observation path (scale -> governor -> IK ->
  gates -> commands -> fk obs) passes tests/test_env_fake_backend.py on a
  ROS-free machine (world-frame delta semantics, gripper scale/direction,
  10 Hz pacing, upsampler convergence all asserted).
- REAL HARDWARE (supersedes the 2026-07-24 "not yet validated on a real ROS
  graph" note this file used to carry): the intervention runner passed on the
  real UR7e 2026-07-28 (docs/testing/04_HIL_INTERVENTION.md §4.5), the first
  full policy+intervention E2E ran 2026-07-29 (CLAUDE.md), and the 30 Hz
  in-window substep path passed ARMED 2026-07-30 (§9). config.DRY_RUN stays
  True by default and run_real_hil.py still needs an explicit --arm to publish.
- STILL NOT validated on hardware: the workspace box. The default config leaves
  ABS_POSE_LIMIT_* at zeros, so the runs above detected a zero-volume box and
  disabled it (08_OPEN_GAPS.md G1); the measured box lives only in
  ur_experiments/cube_in_cup.py. PolicyDeltaController is also still the
  simplified gate stack, not the full eef_delta one (G2; README:
  "PolicyDeltaController 현황 vs eef_delta 본체").
"""

import queue
import threading
import time
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from ur_env.envs.config import DefaultUR7eEnvConfig
from ur_env.envs.leader_stream import NOMINAL_CONTROL_HZ
from ur_env.envs.policy_delta_controller import PolicyDeltaController
from ur_env.observation_preprocess import preprocess_frame

# The background follower's own clock, bound at import time and NEVER reached
# through the module global ``time``.  The substep tests replace
# ``ur7e_env.time`` with a virtual clock whose ``sleep`` ADVANCES it; a daemon
# thread pacing itself through that global would drive somebody else's virtual
# clock from a second thread and make every timing assertion in the suite a
# race.  The pacing shell is the only code that uses this name, and
# ``_follow_tick`` — the part that has behaviour worth testing — takes its clock
# reading as an ARGUMENT instead of reading one at all.
_wall_monotonic = time.monotonic


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

        # ---- human-intervention substepping (see step / config.INTERVENTION) --
        # A ONE-SHOT driver object installed by GelloIntervention immediately
        # before each intervened step and consumed by that step.  ``None`` is
        # the POLICY path and must stay byte-for-byte the pre-change behaviour:
        # one target per window plus one sleep.  Declared before the fake_env
        # early return so the attribute always exists.
        self._intervention_driver = None
        intervention_cfg = getattr(self.config, "INTERVENTION", None) or {}
        #: Public because GelloIntervention reads it as the single authority on
        #: the substep rate (its leader filter's output period and its budget are
        #: both defined by it).  0.0 = feature off, i.e. the pre-substep path.
        self.intervention_substep_hz = float(
            intervention_cfg.get("substep_hz", 0.0)
        )
        #: "background" (a daemon thread follows the leader independently of the
        #: RL loop) or "in_window" (the pre-2026-07-30 path, substeps packed into
        #: ``step``'s sleep).  See ``config.INTERVENTION["follow_mode"]`` for the
        #: measurement that motivated the move.  Rejected rather than defaulted
        #: on a typo: silently selecting a control topology is exactly the class
        #: of "it ran, but not the way you think" failure this stack keeps
        #: paying for.
        follow_mode = str(intervention_cfg.get("follow_mode", "background")).lower()
        if follow_mode not in ("background", "in_window"):
            raise ValueError(
                f"INTERVENTION['follow_mode'] must be 'background' or "
                f"'in_window' (got {follow_mode!r})"
            )
        if self.intervention_substep_hz <= 0.0:
            # No tick rate => no follower to run.  Degenerate to the path that
            # needs none, which is also the pre-substep behaviour.
            follow_mode = "in_window"
        self.intervention_follow_mode = follow_mode
        #: Tick period of the background follower AND the ``dt`` its commands are
        #: governed over.  ``None`` when there is no tick rate at all.
        self._follow_dt: Optional[float] = (
            1.0 / self.intervention_substep_hz
            if self.intervention_substep_hz > 0.0
            else None
        )
        self._init_command_mux()
        if self.intervention_substep_hz > 0.0 and follow_mode == "in_window":
            # 1.5x, not 1x: _drive_intervention_substeps only refreshes while at
            # least HALF a substep period is left in the window
            # (`next_tick < window_end - 0.5 * period`), so the first substep
            # needs 1.5 periods of room. substep_hz in (hz, 1.5*hz] therefore
            # runs ZERO substeps while looking enabled — and exactly 1.5*hz is
            # float-degenerate, yielding 0 or 1 depending on the window's
            # absolute start instant. Measured 2026-07-30 at HZ=10:
            # substep_hz 12.0 -> 0 substeps, 15.0 -> 0 or 1, 30.0 -> 2.
            if self.intervention_substep_hz <= 1.5 * self.hz:
                print(
                    f"[UR7eEnv] NOTE: INTERVENTION substep_hz "
                    f"({self.intervention_substep_hz}) is not above 1.5x HZ "
                    f"({1.5 * self.hz}) — no substep reliably fits inside a step "
                    "window, so in-window leader resampling is effectively "
                    "DISABLED"
                )
            elif abs(self.hz - NOMINAL_CONTROL_HZ) > 1e-9:
                # GelloIntervention rations each substep's request with
                # InterventionBudget.nominal_substep_share, which divides the
                # window budget by substep_hz/NOMINAL_CONTROL_HZ.  Off-nominal HZ
                # makes that share describe a window length this env does not
                # have: too small a share under-delivers the window budget (the
                # arm lags the leader), too large a one exhausts it early (the
                # window goes quiet).  Warn rather than silently adjust — which
                # of the two rates is wrong is a task decision.
                print(
                    f"[UR7eEnv] WARNING: HZ ({self.hz}) differs from the "
                    f"nominal control rate ({NOMINAL_CONTROL_HZ}) that the "
                    "intervention budget's per-substep share is defined against "
                    "— in-window leader resampling will mis-ration this task's "
                    "window budget"
                )

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
        # AFTER the controller exists: the follower's very first action is a
        # controller.step(), so a thread started before this line could observe
        # a half-built env.  It comes up DISARMED and stays blocked on
        # ``_follow_armed`` until an RL step promotes it, so an env that is never
        # intervened pays one idle thread and nothing else.
        self._start_follow_thread()

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

    # ================================================================== #
    # COMMAND OWNERSHIP MUX + BACKGROUND INTERVENTION FOLLOWER            #
    #                                                                    #
    # WHAT THIS SOLVES.  ``_drive_intervention_substeps`` (below) follows #
    # the leader only INSIDE ``step``'s nominal 1/HZ window.  On the      #
    # production actor the real step period is 512 ms — 100 ms of         #
    # ``env.step`` plus ~412 ms of blocking gRPC and camera decode that   #
    # happen OUTSIDE it — so the leader was resampled for 13% of each     #
    # period and ignored for the other 87%.  Measured consequences:       #
    # 2.4 cm/s top intervention speed (12.4 cm/s on the validation rig),  #
    # a 445 ms target gap that crosses UPSAMPLER.target_stale_s and       #
    # brakes the stream to HOLD once per window, and a soft_start re-arm  #
    # after every one of those brakes that pins the slew ceiling at 46%.  #
    #                                                                    #
    # THE FIX, in the operator's words: "사람에게 제어권이 넘어올 때는     #
    # 그냥 원래 eef teleop처럼 팔이 움직이고, RL은 10 Hz로 정보만 빼간다". #
    # A daemon thread follows the leader at ``substep_hz`` for as long as #
    # the human is engaged, and ``env.step`` becomes an OBSERVER that     #
    # samples one transition per call.  The arm never waits for the RL    #
    # loop.                                                              #
    #                                                                    #
    # WHY A MUX AND NOT JUST A THREAD.  The contended resource is not the #
    # backend's joint target (last-write-wins, short lock, indifferent).  #
    # It is ``PolicyDeltaController.T_cmd``/``q_cmd``: the controller     #
    # integrates its OWN output and seeds IK from its OWN previous        #
    # solution, so two threads interleaving ``step()`` would seed each    #
    # other's branch continuity — which comes out of the arm as a wrist   #
    # flip, not as a log line.  So exactly one owner may command at a     #
    # time, and ``_emit_arm_command`` is the ONLY door.                   #
    #                                                                    #
    # PROMOTION IS ASYMMETRIC, deliberately.  POLICY -> HUMAN happens     #
    # only on the RL thread at a step boundary (``arm_intervention_       #
    # follow``); HUMAN -> POLICY may happen on any thread at any tick     #
    # (``disarm_intervention_follow`` / ``_brake_follow`` /               #
    # ``_fail_closed``).  Starting is a decision; stopping is a reflex.   #
    # ================================================================== #
    OWNER_POLICY = "POLICY"
    OWNER_HUMAN = "HUMAN"
    #: How long the idle follower blocks per armed-check.  It bounds the delay
    #: between an RL-thread promotion and the first following tick, so it is
    #: deliberately far below one substep period.
    FOLLOW_IDLE_POLL_S = 0.02
    #: ``close()`` waits at most this long for the follower to leave its loop,
    #: and ``reset()`` at most this long for it to acknowledge a disarm.
    #:
    #: THE BUDGET IS NOT ARBITRARY.  ``hil_restore_controller_after_actor``
    #: waits 30 x 0.1 s for our command publisher to disappear before it will
    #: switch the arm back from forward_position_controller to the scaled joint
    #: trajectory controller, and ``URRosBackend.close`` SKIPS ``destroy_node()``
    #: entirely if its own joins time out — so a slow teardown here does not
    #: merely delay the exit, it leaves the arm on FPC with the controller
    #: switch refused ("still has N publisher(s)").  The follower exits within
    #: one tick of ``_follow_stop`` being set (it waits on that event rather
    #: than sleeping), so 0.5 s is already two orders of magnitude of slack, and
    #: the two waits together stay well inside the 3 s the restore hook allows.
    FOLLOW_JOIN_TIMEOUT_S = 0.5
    FOLLOW_QUIESCE_TIMEOUT_S = 0.5

    def _init_command_mux(self) -> None:
        """Declare every mux/follower attribute, fake envs included.

        Before the ``fake_env`` early return on purpose: ``close()``,
        ``reset()`` and the accessors must be safe to call on an env that never
        builds a controller at all, and a ``hasattr`` dance at each of those
        sites is exactly how a safety gate ends up silently skipped.
        """

        self._command_lock = threading.RLock()
        self._command_owner = self.OWNER_POLICY
        self._follow_source = None
        self._follow_armed = threading.Event()
        self._follow_stop = threading.Event()
        #: Set by the follower whenever it is parked on ``_follow_armed`` — i.e.
        #: PROOF that it has observed a disarm and is not between the ownership
        #: test and a ``send_joint_command``.  ``go_to_reset`` needs the proof
        #: and not just the request: it republishes its target at 20 Hz while
        #: the follower publishes at 30 Hz, and the backend is last-write-wins,
        #: so a follower that is still going WINS.  The reset then never reaches
        #: ``RESET_TOLERANCE_RAD`` and ends in 10 s of blind motion plus a
        #: RuntimeError.  Starts set: no thread means nothing to wait for.
        self._follow_idle = threading.Event()
        self._follow_idle.set()
        self._follow_thread: Optional[threading.Thread] = None
        #: Exception raised on the follower thread and re-raised on the RL
        #: thread at its next ``step``/``reset``.  See ``_raise_follow_fault``.
        self._follow_fault: Optional[BaseException] = None
        #: Reason string of a brake that happened between two RL steps, reported
        #: once as ``info["intervention_window_braked"]``.
        self._follow_brake_reason: Optional[str] = None
        #: Set with it, consumed separately by the wrapper BEFORE its next
        #: ``action()``; see ``consume_follow_reanchor``.
        self._follow_needs_reanchor = False
        self._reset_follow_window_locked()

    def _reset_follow_window_locked(self) -> None:
        """Open a fresh recording window.  Caller holds ``_command_lock``."""

        #: Task-space displacement the controller has actually COMMITTED under
        #: HUMAN ownership since the last harvest — see ``_harvest_follow_window``
        #: for why this replaced ``InterventionBudget`` as the record.
        self._follow_sum_p = np.zeros(3, dtype=float)
        self._follow_sum_w = np.zeros(3, dtype=float)
        self._follow_ticks = 0
        self._follow_window_had_human = False
        self._follow_agg: Dict[str, object] = {
            "held": False,
            "clipped": False,
            "governed": False,
            "governed_scale": 1.0,
            "reject_reason": None,
        }

    # ---- the single door to the arm --------------------------------------- #
    def _emit_arm_command(self, xi: np.ndarray, dt, owner: str):
        """THE only path from a task-space increment to a joint command.

        Returns ``(q_cmd, ctrl_info)``, or ``(None, ctrl_info)`` when the caller
        does not own the arm — in which case NOTHING happened: no controller
        state moved, no target was published.

        The owner test is the FIRST thing inside the lock and it is the whole
        mux.  Removing it does not produce a race that shows up as a log line;
        it produces two threads interleaving ``PolicyDeltaController.step`` and
        seeding each other's IK, i.e. a wrist that turns over.

        ``dt=None`` reproduces the policy path bit for bit
        (``PolicyDeltaController._resolve_dt`` returns the nominal tick and the
        unscaled joint gate verbatim for ``None``), so routing the policy path
        through here costs one lock acquisition and one 4x4 copy and changes no
        arithmetic.

        The controller's RETURN VALUE is what reaches the backend — never a
        joint vector computed some other way.  The workspace box, the governor,
        the branch-continuous IK seed and the joint-step line search all live
        inside ``controller.step``; a "quick" direct publish would bypass every
        one of them at once.  ``go_to_reset`` is the single grandfathered
        exception and it guards itself (POLICY ownership + branch mapping +
        distance gate).
        """

        with self._command_lock:
            if owner != self._command_owner:
                return None, {
                    "held": False,
                    "reject_reason": None,
                    "clipped": False,
                    "governed": False,
                    "governed_scale": 1.0,
                    #: Present ONLY on a blocked emit, so a consumer can tell
                    #: "this call issued nothing because somebody else is
                    #: driving" apart from "this call issued a HOLD".
                    "arm_command_owner": self._command_owner,
                }
            T_before = self.controller.tcp_cmd()
            q_cmd, ctrl_info = self.controller.step(xi, dt=dt)
            self.backend.send_joint_command(q_cmd)
            if owner == self.OWNER_HUMAN:
                self._record_human_commit(T_before, ctrl_info)
            return q_cmd, ctrl_info

    def _record_human_commit(self, T_before: np.ndarray, ctrl_info: dict) -> None:
        """Accumulate what the controller COMMITTED.  Caller holds the lock.

        THE COMMANDED DISPLACEMENT, not the measured TCP.  Every piece of anchor
        maths in ``GelloIntervention`` is expressed against ``controller.
        tcp_cmd()``, so measuring the record against the same quantity keeps the
        report self-consistent with the thing that produced it; measuring the
        arm instead would fold in tracking lag that no action caused.

        Committed, not requested.  ``T_before -> T_after`` is what survived the
        governor, the workspace clamp and the joint-step line search, so a tick
        the controller HELD contributes exactly zero and a tick it shrank
        contributes exactly what it shrank to.  That closes, for this path, the
        over-report ``test_a_line_search_shrink_must_not_be_left_out_of_the_
        recorded_action`` xfails on the in-window path (the budget there charges
        the REQUEST).
        """

        T_after = self.controller.tcp_cmd()
        self._follow_sum_p += T_after[:3, 3] - T_before[:3, 3]
        self._follow_sum_w += self._kin.so3_log(
            T_after[:3, :3] @ T_before[:3, :3].T
        )
        self._follow_window_had_human = True
        self._follow_ticks += 1
        for key in ("held", "clipped", "governed"):
            if ctrl_info.get(key):
                self._follow_agg[key] = True
        scale = ctrl_info.get("governed_scale")
        if scale is not None:
            self._follow_agg["governed_scale"] = min(
                float(scale), float(self._follow_agg["governed_scale"])
            )
        if ctrl_info.get("reject_reason"):
            self._follow_agg["reject_reason"] = ctrl_info["reject_reason"]

    # ---- ownership transitions -------------------------------------------- #
    def arm_intervention_follow(self, source) -> bool:
        """POLICY -> HUMAN.  RL thread, at a step boundary, ONLY.

        ``source`` implements the follower protocol:

            source.follow_is_engaged() -> bool
                The deadman, read fresh.  MAY RAISE (a stale heartbeat is not a
                release); anything it raises fails the arm closed.
            source.follow_xi(now) -> (6,) or None
                The anchored leader delta in real units for the tick at ``now``.
                UNBUDGETED — the governor and the workspace box inside
                ``controller.step`` are the only bounds.  ``None`` means the
                leader is unusable => brake and re-anchor.

        Returns False when this env cannot follow in the background (fake env,
        or ``follow_mode="in_window"``), so the caller can fall back.
        """

        if self.fake_env or self._follow_thread is None:
            return False
        # Never promote into a latched fault: the operator's state is unknown,
        # which is the one condition under which the human must NOT be handed
        # the arm.
        self._raise_follow_fault()
        with self._command_lock:
            self._follow_source = source
            self._command_owner = self.OWNER_HUMAN
            self._follow_window_had_human = True
            self._follow_armed.set()
        return True

    def disarm_intervention_follow(self) -> None:
        """HUMAN -> POLICY.  Any thread, any tick, no questions asked.

        Once this returns, no further HUMAN command can reach the backend: a
        follower tick already inside ``_emit_arm_command`` holds the lock, so
        this call waits for it, and the next tick re-tests ownership under the
        same lock and finds POLICY.
        """

        with self._command_lock:
            self._follow_armed.clear()
            self._command_owner = self.OWNER_POLICY

    def await_follower_quiescent(
        self, timeout: Optional[float] = None
    ) -> bool:
        """Disarm AND wait for the follower to acknowledge it. Returns success.

        WHY AN ACK AND NOT JUST A DISARM.  ``disarm_intervention_follow`` is
        already a hard guarantee that no further HUMAN command reaches the
        backend (the ownership test is inside the same lock a running tick
        holds).  This adds the second thing ``go_to_reset`` needs: evidence that
        the follower is PARKED, not merely locked out.  ``go_to_reset``
        republishes one target at 20 Hz for up to ``RESET_TIMEOUT_S``; a
        follower publishing at 30 Hz into a last-write-wins backend simply wins,
        the arm never converges to ``RESET_TOLERANCE_RAD``, and the operator
        gets ten seconds of blind motion followed by "reset did not arrive".

        Bounded, and a failure is reported rather than waited out: this is
        called from ``reset``, which must not become the thing that wedges a
        session.
        """

        self.disarm_intervention_follow()
        thread = self._follow_thread
        if thread is None or not thread.is_alive():
            return True
        return self._follow_idle.wait(
            self.FOLLOW_QUIESCE_TIMEOUT_S if timeout is None else timeout
        )

    def suspend_follower(self):
        """Context manager: no human following inside this block.

        NOT WIRED UP YET — exposed for ``remote_actor``'s operator gates
        (``WAIT_SCENE_READY`` / ``WAIT_HOME_APPROVAL``), which block the RL
        thread indefinitely and, in the ``WAIT_HOME_APPROVAL`` case, do not
        consult the deadman at all.  A follower left armed across one of those
        would keep driving the arm for as long as the operator takes to answer a
        prompt.  The gates live in ``ur_env/remote_actor.py``, which this change
        deliberately does not touch; wiring them is a separate, named follow-up.

        Re-arming is the caller's job on purpose: promotion is only ever allowed
        from the RL thread at a step boundary, and this context manager cannot
        know it is at one.
        """

        env = self

        class _Suspend:
            def __enter__(self_inner):
                env.await_follower_quiescent()
                return env

            def __exit__(self_inner, *exc_info):
                return False

        return _Suspend()

    def intervention_follow_armed(self) -> bool:
        """Is the background follower currently allowed to drive the arm?"""

        return self._follow_armed.is_set()

    def command_owner(self) -> str:
        """``OWNER_POLICY`` or ``OWNER_HUMAN`` — who may command right now."""

        with self._command_lock:
            return self._command_owner

    def _brake_locked(self, reason: str) -> None:
        """Stop the command stream and re-anchor.  Caller holds the lock.

        The same two lines ``_apply_action``'s hold branch runs, and for the
        same reason: ``request_hold()`` replaces the goal with its
        acceleration-limited stop point, and ``controller.reset()`` re-anchors
        the task-space integrator there so ``T_cmd`` cannot keep a phantom
        excursion the arm never made.
        """

        try:
            q_hold = self.backend.request_hold()
            self.controller.reset(q_hold)
        except Exception as exc:  # pragma: no cover - backend-specific
            # A failed HOLD must not take the disarm down with it: the arm is
            # already safest with no further targets (the 250 Hz upsampler
            # brakes on its own at ``target_stale_s``), and swallowing here is
            # what keeps that true.
            print(
                f"[UR7eEnv] WARNING: HOLD for {reason} failed ({exc}) — the "
                "upsampler's target_stale_s brake is now the only stop"
            )

    def _brake_follow(self, reason: str) -> None:
        """Leader/no-anchor failure: stop, hand back, and ask for a re-anchor.

        NOT a fault.  A GELLO dropout is an ordinary event; the RL thread
        re-anchors on its next step (which is also where the sensitivity gain is
        latched, so keeping re-anchoring there keeps that latch in ONE place).
        """

        with self._command_lock:
            self._follow_armed.clear()
            self._command_owner = self.OWNER_POLICY
            self._brake_locked(reason)
            self._follow_brake_reason = str(reason)
            self._follow_needs_reanchor = True

    def consume_follow_reanchor(self) -> bool:
        """Did a brake invalidate the anchor since this was last asked?

        TWO consumers, two lifetimes, which is why this is not the same flag as
        the one the harvest reports.  ``info["intervention_window_braked"]``
        describes the TRANSITION and is read after ``env.step`` returns; this one
        has to be read BEFORE ``action()`` runs, because ``_brake_locked`` moved
        the controller's integrator to the braking endpoint and the wrapper's
        ``T_r_anchor`` no longer describes it.  A wrapper that re-armed on the
        stale anchor would command the whole braking distance as a step.
        """

        with self._command_lock:
            pending = self._follow_needs_reanchor
            self._follow_needs_reanchor = False
        return pending

    def _fail_closed(self, exc: BaseException) -> None:
        """Unknowable operator/leader state: stop, hand back, LATCH the fault.

        The distinction that matters is ``DeadmanHeartbeatStaleError``'s own:
        ``is_engaged() == False`` is a fresh, explicit release and policy control
        may resume, while a heartbeat that simply stopped arriving authorizes
        nothing.  So a raise here is never reinterpreted as a release — it stops
        the arm and is re-raised on the RL thread, where the existing FAULT
        reporting path already lives.

        Anything else the follower can raise is treated identically.  Catching
        only the one class would leave a kinematics error or a backend error
        running a following loop whose own state is unknown, which is precisely
        the situation the class exists to refuse.
        """

        with self._command_lock:
            self._follow_armed.clear()
            self._command_owner = self.OWNER_POLICY
            self._brake_locked("FOLLOW_FAULT")
            if self._follow_fault is None:
                self._follow_fault = exc

    def _latch_follow_fault(self, exc: BaseException) -> None:
        with self._command_lock:
            if self._follow_fault is None:
                self._follow_fault = exc

    def _raise_follow_fault(self) -> None:
        """Re-raise a follower fault on the RL thread.  First line of step/reset.

        Disarms before raising: an exception propagating out of ``step`` must
        never leave a thread still driving the arm, and a caller that catches it
        somewhere far above would otherwise inherit exactly that.

        The latch is deliberately PERMANENT — every later ``step``/``reset``
        raises again — because the fault's own message says what it means
        ("refusing policy fallback and stopping the actor").  ``clear_follow_
        fault()`` is the explicit operator escape hatch; nothing clears it
        implicitly.
        """

        fault = self._follow_fault
        if fault is None:
            return
        self.disarm_intervention_follow()
        raise fault

    def clear_follow_fault(self) -> Optional[BaseException]:
        """Explicitly discharge a latched follower fault. Returns what it was."""

        with self._command_lock:
            fault, self._follow_fault = self._follow_fault, None
        return fault

    # ---- the follower itself ---------------------------------------------- #
    def _start_follow_thread(self) -> None:
        if self.fake_env or self.intervention_follow_mode != "background":
            return
        if self._follow_dt is None:
            return
        self._follow_thread = threading.Thread(
            target=self._follow_loop,
            name="ur7e-intervention-follow",
            daemon=True,
        )
        self._follow_thread.start()

    def _follow_loop(self) -> None:
        """Pacing shell.  ALL the clock reading lives here and nowhere else.

        Split this way so ``_follow_tick`` is a pure function of ``(now, world
        state)`` and can be driven from a virtual clock in tests without a
        thread at all — the same rule ``AccelerationLimitedJointStream.advance``
        and ``PolicyDeltaController.step`` already follow (both take their clock
        as an argument; see the latter's "WHY THE CALLER PASSES dt" note).

        Waits on ``_follow_stop`` rather than sleeping so ``close()`` is prompt
        rather than up to one period late, and blocks on ``_follow_armed`` while
        idle so an env that is never intervened costs one parked thread.
        """

        period = float(self._follow_dt)
        next_tick = None
        try:
            while not self._follow_stop.is_set():
                # PARKED: set the ack BEFORE blocking, so a disarm that lands
                # while we are here is acknowledged immediately rather than one
                # poll later, and clear it the instant we are awake and about to
                # command again.
                self._follow_idle.set()
                if not self._follow_armed.wait(self.FOLLOW_IDLE_POLL_S):
                    next_tick = None  # a fresh arm starts a fresh phase
                    continue
                self._follow_idle.clear()
                if not self._follow_armed.is_set():
                    # Disarmed between the wait returning and here: park again
                    # rather than run a tick nobody authorized.
                    continue
                now = _wall_monotonic()
                if next_tick is None:
                    next_tick = now
                self._follow_tick(now)
                next_tick += period
                delay = next_tick - _wall_monotonic()
                if delay <= 0.0:
                    # Fell behind (a long IK, a scheduling hiccup): re-phase on
                    # the next reading instead of firing a catch-up burst, which
                    # would hand the governor several ticks' worth of motion in
                    # one instant.
                    next_tick = None
                elif self._follow_stop.wait(delay):
                    break
        except BaseException as exc:  # pragma: no cover - defence in depth
            self._latch_follow_fault(exc)
        finally:
            # A dead follower must not leave HUMAN ownership behind, or the RL
            # thread would be locked out of the arm forever.  (Missing ticks are
            # already fail-safe on their own: with no target refresh the 250 Hz
            # upsampler brakes to HOLD at ``target_stale_s``.  The dangerous
            # failure is a follower that stays ALIVE and keeps commanding from
            # stale state, which is what the per-tick gates below address.)
            self.disarm_intervention_follow()
            # A thread that is gone is quiescent by definition; leaving the ack
            # clear would make ``await_follower_quiescent`` block for its whole
            # timeout on every reset after a follower crash.
            self._follow_idle.set()

    def _follow_tick(self, now: float) -> dict:
        """ONE following tick at instant ``now``.  Reads no clock of its own.

        THE SAFETY GATES, in the order they run and why that order:

        1. ARMED.  Re-tested here, under nothing, and again inside
           ``_emit_arm_command`` under the lock.  The second test is the binding
           one; this one only avoids the work.
        2. THE DEADMAN, EVERY TICK.  This is the point of the whole change.
           Under in-window following a release only took effect at the next
           window boundary, i.e. after up to one REAL step period — 512 ms of
           further motion on the production loop.  Read per tick it is 33 ms.
           ``is_engaged() == False`` disarms (a release is an authorization to
           resume policy control, so no brake is needed); anything RAISED fails
           closed instead, because a heartbeat that stopped arriving authorizes
           nothing.
           ``gain()`` is deliberately NOT re-read: it is latched at the engage
           edge so a slider nudged mid-motion cannot discontinuously rescale an
           in-progress delta, and re-reading it 30 times a second would
           reintroduce that discontinuity 30 times faster than the bug it
           replaced.
        3. THE LEADER.  ``follow_xi`` returns None for a stale/absent/non-finite
           leader; a non-finite xi is caught here as well, because a NaN reaching
           ``send_joint_command`` is a raise on the wrong thread. Either brakes
           and asks the RL thread for a re-anchor.
        4. THE MUX.  ``_emit_arm_command`` re-checks ownership under the lock and
           is the only door to the arm.

        Returns a small dict for tests and logging; the caller ignores it.
        """

        if not self._follow_armed.is_set():
            return {"issued": False, "reason": "DISARMED"}
        source = self._follow_source
        if source is None:
            self.disarm_intervention_follow()
            return {"issued": False, "reason": "NO_SOURCE"}

        try:
            engaged = bool(source.follow_is_engaged())
        except Exception as exc:
            self._fail_closed(exc)
            return {"issued": False, "reason": "DEADMAN_FAULT"}
        if not engaged:
            self.disarm_intervention_follow()
            return {"issued": False, "reason": "DISENGAGED"}

        try:
            xi = source.follow_xi(now)
        except Exception as exc:
            self._fail_closed(exc)
            return {"issued": False, "reason": "LEADER_FAULT"}
        if xi is None:
            self._brake_follow("GELLO_STALE")
            return {"issued": False, "reason": "GELLO_STALE"}
        xi = np.asarray(xi, dtype=float).reshape(6)
        if not np.all(np.isfinite(xi)):
            self._brake_follow("BAD_INPUT")
            return {"issued": False, "reason": "BAD_INPUT"}

        try:
            q_cmd, ctrl_info = self._emit_arm_command(
                xi, self._follow_dt, self.OWNER_HUMAN
            )
        except Exception as exc:
            self._fail_closed(exc)
            return {"issued": False, "reason": "COMMAND_FAULT"}
        if q_cmd is None:
            return {"issued": False, "reason": "NOT_OWNER", "info": ctrl_info}
        return {"issued": True, "reason": None, "info": ctrl_info}

    # ---- the record ------------------------------------------------------- #
    def _harvest_follow_window(self, ctrl_info: dict) -> dict:
        """Turn what the follower COMMITTED into this transition's action.

        WHAT THE RECORD IS.  The task-space displacement the CONTROLLER
        COMMITTED under HUMAN ownership during this window, divided by
        ``ACTION_SCALE``.  Not the budget's charge (which bills the REQUEST, so
        anything the IK line search shrinks below it is over-reported — the
        standing xfail on the in-window path), and not the measured TCP (which
        would fold in tracking lag no action caused).

        THE COST OF THE UNBUDGETED FOLLOWER, stated plainly because it is a
        KNOWN, ACCEPTED breach of the stored-action invariant and not an
        oversight.  ``GelloIntervention.follow_xi`` no longer rations the
        window against ``ACTION_SCALE``, so a long window can commit several
        ``ACTION_SCALE`` steps of motion while the recorded action can only ever
        say 1.0.  The transition then tells the learner "action 1.0 moved the arm
        one ACTION_SCALE step" when it moved N of them, and the critic learns an
        OPTIMISTIC dynamics model — the exact direction ``config.ACTION_SCALE``'s
        INVARIANT note and ``README.md``'s 저장 액션 불변식 forbid.

        It was adopted anyway, on 2026-07-30, because the alternative measured
        worse: with the budget in the loop the operator's ceiling is
        ``ACTION_SCALE / T`` = 2.4 cm/s at the production 512 ms step, which is
        not drivable, and an intervention path nobody can drive yields ZERO
        demonstrations.  A biased sample beats an empty one.  The real fixes are
        (a) dropping saturated transitions server-side — which needs a new proto
        field, a ``SCHEMA_VERSION`` bump and both ends upgraded together, because
        protobuf silently DISCARDS unknown fields and a half-upgraded Kanu would
        train on the under-reported actions with no symptom at all — or (b)
        shortening the window itself (08_OPEN_GAPS.md G21).

        ``intervention_saturated`` / ``intervention_saturation`` are how that
        cost is measured instead of assumed: the ratio BEFORE the clip, so 4.94
        means this transition under-reports its motion 4.94-fold.  Local
        instrumentation ONLY — they are deliberately not on the wire (see above),
        and ``tests/run_real_hil.py`` writes them to its CSV so the decision
        between (a) and (b) can be made on numbers.

        THE WINDOW IS BOUNDARY-TO-BOUNDARY, not the duration of ``env.step``.
        The accumulator is reset here, immediately before ``_update_currpos`` /
        ``_get_obs`` sample ``obs_{k+1}``, so a window covers exactly the
        interval between the two observations of one transition — INCLUDING the
        ~412 ms the actor spends in its gRPC round trip, during which the
        follower is still driving.  That motion belongs to this transition and
        nowhere else.
        """

        with self._command_lock:
            had_human = self._follow_window_had_human
            sum_p = self._follow_sum_p.copy()
            sum_w = self._follow_sum_w.copy()
            ticks = int(self._follow_ticks)
            agg = dict(self._follow_agg)
            braked = self._follow_brake_reason
            self._follow_brake_reason = None
            self._reset_follow_window_locked()

        if not had_human and braked is None:
            return ctrl_info

        out = dict(ctrl_info)
        if had_human:
            raw = np.concatenate(
                [sum_p / self.action_scale[0], sum_w / self.action_scale[1]]
            )
            # PROPORTIONAL (norm) shrink, NOT a per-axis np.clip.  Two reasons,
            # and the second one is a crash we actually hit on the real rig:
            #
            #  1. Direction.  An axis-wise clip of a saturated diagonal bends
            #     the recorded path ([2.0, 0.5] -> [1.0, 0.5]), which is the
            #     same argument _expert_delta_xi makes for its own clamp.
            #
            #  2. RelativeFrame ROTATES this vector.  ``transform_action_inv``
            #     applies blockdiag(R, R), which preserves each 3-vector's NORM
            #     but not its per-component maximum.  A per-axis clip leaves a
            #     norm of up to sqrt(3) = 1.73, so [1.0, 1.0, 0] can come out of
            #     the rotation as [1.41, 0, 0] — outside [-1, 1].  That is
            #     exactly the 2026-07-30 real-rig failure:
            #     ``ActorProtocolError: executed_action must be within [-1, 1]``
            #     raised by validate_action in remote_actor.build_data.
            #     Shrinking by norm keeps norm <= 1, and a rotation of a
            #     norm <= 1 vector still has every component in [-1, 1].
            #
            # The old budget path was safe for this reason too: its clamp was on
            # the norm, so the invariant survived the frame change for free.
            committed = raw.copy()
            n_p = float(np.linalg.norm(committed[:3]))
            n_w = float(np.linalg.norm(committed[3:]))
            if n_p > 1.0:
                committed[:3] = committed[:3] / n_p
            if n_w > 1.0:
                committed[3:] = committed[3:] / n_w
            out["intervention_committed_action"] = committed.astype(np.float32)
            # Saturation is reported on the NORMS, matching what was shrunk.
            out["intervention_saturation"] = float(max(n_p, n_w))
            out["intervention_saturated"] = bool(n_p > 1.0 or n_w > 1.0)
            out["intervention_follow_ticks"] = ticks
            # Namespaced rather than OR-ed into ``held``: ``held`` decides the
            # operator status line (remote_actor._control_state maps it to HOLD
            # instead of HUMAN_INTERVENTION), and a single held follower tick in
            # a 512 ms window is not a held transition.
            out["intervention_follow_held"] = bool(agg["held"])
            out["intervention_follow_reject_reason"] = agg["reject_reason"]
            # ``clipped``/``governed`` ARE merged into the standard keys: both
            # docstrings argue at length that a silent clamp or a silent rate cut
            # is the hardest failure to diagnose after the fact, and neither key
            # gates any behaviour anywhere.
            if agg["clipped"]:
                out["clipped"] = True
            scale = float(agg["governed_scale"])
            if scale < float(out.get("governed_scale", 1.0)):
                out["governed_scale"] = scale
            if agg["governed"]:
                out["governed"] = True
        if braked is not None:
            # Same transport the in-window brake uses, so GelloIntervention's
            # existing "drop the anchor, re-anchor next step" handling applies
            # unchanged.
            out["intervention_window_braked"] = True
            out["held"] = True
            out.setdefault("reject_reason", None)
            out["reject_reason"] = braked
        return out

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

    def _apply_action(self, action: np.ndarray, driver=None) -> dict:
        """Action post-processing + dispatch: scale -> governor/box/IK gates ->
        publish arm and gripper commands. Returns the controller info dict
        (held / reject_reason / clipped).

        The workspace box is enforced INSIDE the controller, not here: it is
        applied to the commanded pose BEFORE IK, so a clamped target re-solves
        instead of being held, and the clamped pose becomes the controller's
        integrator state (see PolicyDeltaController.step for the anti-windup
        argument). ``self._clip_command_pose`` is the hook — this env still
        owns the box, the controller just asks it per tick.

        ``driver`` is the human-intervention substep driver, or None on the
        policy path.  Two things change when one is present, and only then:

        * the first target is CHARGED against the window's displacement budget;
        * it is governed over ONE SUBSTEP (``dt = 1/substep_hz``) instead of the
          whole window.

        INVARIANT 1 (buffer correctness — why the charge is not optional).  A
        stored intervention action must stay inside [-1,1]^7 and must describe
        the motion that actually ran (serl_ur_infra/README.md:197-205,
        config.ACTION_SCALE:55-73).  The substeps after this one keep commanding
        motion inside the SAME env step, so without a single shared budget the
        window's total displacement would exceed one ACTION_SCALE step and the
        recorded action would saturate at 1.0 while the arm kept moving — i.e.
        storage would silently UNDER-report the executed motion.  The first
        target is part of that total, so it draws from the same budget.

        INVARIANT 2 (the three-layer speed ratio, config.py:65-72 — why ``dt``).
        The governor caps per-tick motion at ``v_max * dt``.  Passing the full
        ``1/HZ`` here and ``1/substep_hz`` to the substeps would let one window
        through 0.0150 + 2*0.0050 = 0.0250 m of governor headroom instead of
        0.0150 m: a 1.67x widening of the intervention path's rate ceiling.
        Today the budget (0.0125 m) binds first so nothing visibly breaks, but
        that would quietly promote the budget from "buffer correctness" to "the
        only thing holding the three-layer contract together" — and the contract
        has to survive the budget being relaxed.  One consistent rate accounting
        for the whole window is the fix.  ``dt`` also scales ``dq_step_max``
        proportionally inside the controller, so the joint-step gate tightens
        with it rather than being left at a full-window budget.
        """
        if self._hold_next_action_reason is not None:
            reason = self._hold_next_action_reason
            self._hold_next_action_reason = None
            # A HOLD is a STOP, so it revokes HUMAN ownership too — stopping is
            # allowed from any thread at any instant, and doing it under the
            # mux lock is what stops ``controller.reset`` racing a follower tick
            # that is mid-``controller.step``.  No ``_follow_brake_reason`` is
            # latched: this brake was REQUESTED by the caller, which reports it
            # itself in the returned info, so latching would double-report it.
            with self._command_lock:
                self._follow_armed.clear()
                self._command_owner = self.OWNER_POLICY
                self._brake_locked(reason)
            return {"held": True, "reject_reason": reason, "clipped": False}

        xi = self._action_to_xi(action)
        if driver is None:
            # POLICY PATH: dt=None, i.e. the nominal tick and the unscaled joint
            # gate, bit-identical to the bare ``controller.step(xi)`` this used
            # to call.  Blocked (q_cmd is None) exactly when the background
            # follower owns the arm, which is the whole point: during a
            # background intervention the RL loop contributes the GRIPPER
            # channel and nothing else.
            _q_cmd, ctrl_info = self._emit_arm_command(xi, None, self.OWNER_POLICY)
        else:
            _q_cmd, ctrl_info = self._emit_arm_command(
                driver.charge(xi),
                1.0 / self.intervention_substep_hz,
                self.OWNER_POLICY,
            )
        self._send_gripper_command(action[6] * self.action_scale[2])
        return ctrl_info

    def request_hold(self, reason: str = "EXTERNAL_HOLD") -> None:
        """Make the next real step brake the command stream and re-anchor it."""
        self._hold_next_action_reason = str(reason)

    # ------------------------------------------------------------------ #
    # human intervention: 30 Hz leader resampling inside the step window  #
    #                                                                    #
    # THIS FILE'S PIECE: hold the one-shot driver, charge its first      #
    # target (_apply_action, :502-505), pace the window                  #
    # (_drive_intervention_substeps), harvest what executed (:654).      #
    # UPSTREAM: GelloIntervention.step -> _open_substep_window           #
    # (wrappers.py:705, :683).  DOWNSTREAM: driver.substep ->            #
    # LeaderFilter / InterventionBudget (wrappers.py:618,                #
    # leader_stream.py:259, :379).  Ordered call chain: the wrappers.py  #
    # module docstring.                                                  #
    # ------------------------------------------------------------------ #
    def begin_intervention_window(self, driver) -> None:
        """Install a ONE-SHOT substep driver for the NEXT ``step`` call.

        ``GelloIntervention`` calls this just before ``env.step`` whenever it is
        executing a human action (see ur_env/envs/wrappers.py).  The driver is
        consumed and cleared at the top of ``step``, so it can never survive
        into a policy step even if the caller forgets — the policy path has to
        stay bit-identical, and "a leftover driver" would be an invisible way to
        break that.

        The driver protocol (implemented by GelloIntervention, which owns the
        anchored-leader maths this env deliberately knows nothing about):

            driver.charge(xi) -> xi
                Draw this window's FIRST target from the displacement budget.
            driver.substep(dt) -> dict
                Re-read the leader, recompute the anchored delta, draw from the
                budget and issue the joint command.  Returns the substep's
                controller info plus the keys ``issued`` (bool: was a target
                actually refreshed), ``stop`` (None or a reason string ending
                the window) and ``brake`` (bool: the command stream must be
                braked and re-anchored right now).
            driver.consumed_window_action() -> (7,)
                The normalized action the whole window actually executed.
        """
        self._intervention_driver = driver

    def _drive_intervention_substeps(self, driver, window_end: float) -> dict:
        """Replace the step window's single sleep with paced leader resampling.

        WHY INSIDE THE WINDOW, rather than as more env steps.  The 100 ms sleep
        is dead time by construction: the target has already been set and the
        backend's 250 Hz upsampler (ur_env/envs/ros_backend.py:665-706) is the
        only thing still working.  Its slew/acceleration limits mean it reaches
        a single 100 ms-old target early and then publishes a HOLD for the rest
        of the window — measured 16% (nominal) to 40% (real 0.197 s period)
        fully-stopped time at a 0.15 rad/s leader.  Shortening the env step
        instead is not an option: the step period is what ACTION_SCALE, the
        governor caps and the whole stored-transition contract are calibrated
        against, and the actor's RPC sits in the same loop.  So the fix is to
        keep one env step == one transition and simply stop wasting the sleep.
        Nothing in ``ros_backend`` needed changing for this:
        ``send_joint_command`` is a short-lock last-write-wins target update
        (ros_backend.py:617-631) and ``_upsample_loop`` is indifferent to how
        often the target moves.

        Returns the extra ctrl_info this window produced.  Booleans are OR-ed
        across substeps because they are operator/log signals ("did anything
        get clamped/held in this window at all"), not per-tick state.
        ``governed_scale`` is instead reduced by MINIMUM — "how hard was this
        window cut at worst" — and the first target's value is folded in by the
        caller, which is the only scope that sees both.

        VERIFIED on the real UR7e 2026-07-30 (ARMED, ``run_real_hil.py --arm
        --scale 1.0``, 120 intervened steps, every check PASS): this loop ran
        ``substeps=2`` in EVERY intervened window (i.e. 3 target updates per
        window), frame-map residual 0.130 / alpha 0.983, action-exec dp_ratio
        median 1.000, held 0%, and the operator confirmed the hand feel.  The
        mid-window brake branch below is NOT covered by that run — held was 0%,
        so no GELLO dropout was exercised on hardware; it is unit-tested only
        (tests/test_intervention_substeps.py).  Evidence:
        docs/testing/04_HIL_INTERVENTION.md §9.
        """
        period = 1.0 / self.intervention_substep_hz
        extra: Dict[str, object] = {"intervention_substeps": 0}
        # window_end - 1/hz is the step's start instant; the first substep sits
        # one period after the target _apply_action already issued.
        next_tick = window_end - (1.0 / self.hz) + period

        # Only refresh the target while at least half a substep period is left:
        # a refresh microseconds before the window closes cannot change what the
        # upsampler does, and float accumulation of `period` makes an exact
        # `< window_end` test emit exactly such a degenerate tick.
        while next_tick < window_end - 0.5 * period:
            now = time.time()
            if now < next_tick:
                time.sleep(next_tick - now)
            result = driver.substep(period)

            if result.get("issued"):
                extra["intervention_substeps"] = (
                    int(extra["intervention_substeps"]) + 1
                )
            for key in ("held", "clipped", "governed"):
                if result.get(key):
                    extra[key] = True
            # governed_scale is a MINIMUM, not an OR: it answers "how hard was
            # this window cut at worst", so a 0.7 substep must not be hidden by
            # a later 1.0 one.  Seeded from the substeps only; the first
            # target's value is folded in at the merge site in step(), which is
            # the only place that can see both.
            scale = result.get("governed_scale")
            if scale is not None:
                extra["governed_scale"] = min(
                    float(scale), float(extra.get("governed_scale", 1.0))
                )
            if result.get("reject_reason"):
                extra["reject_reason"] = result["reject_reason"]

            stop = result.get("stop")
            if stop:
                extra.setdefault("reject_reason", stop)
                if result.get("brake"):
                    # The leader died mid-window.  Same two lines as the
                    # _hold_next_action_reason branch of _apply_action, and for
                    # the same reason: request_hold() replaces the goal with its
                    # acceleration-limited stop point, and controller.reset()
                    # re-anchors the task-space integrator there so T_cmd cannot
                    # keep a phantom excursion the arm never made.
                    q_hold = self.backend.request_hold()
                    self.controller.reset(q_hold)
                    extra["held"] = True
                    extra["intervention_window_braked"] = True
                break
            next_tick += period

        # Sleep out whatever is left, so the env step period is unchanged.
        time.sleep(max(0.0, window_end - time.time()))

        # HARVEST, and only if a substep really refreshed the target.  With zero
        # substeps the window executed exactly the one action _apply_action was
        # given, so reporting nothing here makes the wrapper fall back to that
        # action and the record stays bit-identical to the pre-change code.
        #
        # This is harvested HERE — after the window closes, before
        # _update_currpos/_get_obs below — because the total displacement of an
        # intervened window is only known once the window ends.  That is exactly
        # why the executed action cannot be produced by the gym ActionWrapper
        # convention (``action()`` returning the final action before step runs);
        # see the note in GelloIntervention.step.  Harvesting at this point
        # keeps (obs_k, action_k, obs_{k+1}) aligned: obs_{k+1} is read next.
        if extra.get("intervention_window_braked"):
            # A window whose leader died is a HOLD window, recorded as the exact
            # zero action — the same rule GelloIntervention.action applies when
            # the leader is already unusable when the window opens.
            #
            # This UNDERSTATES the window, and the cost is not zero. Measured
            # 2026-07-30: the first target had already commanded 0.004167 m
            # (= ACTION_SCALE/3, normalised action 0.333) before the brake, and
            # request_hold() + controller.reset() cut only what was still IN
            # FLIGHT — they re-anchor the integrator ON the stop point, they do
            # not rewind the displacement already commanded. (Two earlier
            # versions of this comment claimed the brake "already cut whatever
            # motion was in flight" and that reporting it would be "a command no
            # human issued"; both were wrong — that first target came from a
            # fresh leader sample, so a human did issue it.) The first
            # intervened window hides this because the engage anchor makes its
            # first delta exactly zero; it shows from the second window on.
            #
            # Recording zeros is still the choice: it keeps one rule for "the
            # leader was unusable in this window" whether the failure landed
            # before or during it, and the understatement is bounded by one
            # window budget. tests/test_intervention_substeps.py pins the size
            # so the trade stays visible instead of being rediscovered.
            extra["intervention_window_action"] = np.zeros(7, dtype=np.float32)
        elif int(extra["intervention_substeps"]) >= 1:
            extra["intervention_window_action"] = driver.consumed_window_action()
        return extra

    def step(self, action: np.ndarray) -> tuple:
        # FIRST LINE, before anything touches the arm: a follower that lost the
        # deadman heartbeat stopped the arm on its own thread and left the
        # exception here.  Re-raising it on this thread is what puts it back on
        # the actor's existing FAULT path (a background thread's traceback
        # reaches nobody).
        self._raise_follow_fault()
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Consume the one-shot intervention driver (None on the policy path).
        driver = self._intervention_driver
        self._intervention_driver = None

        ctrl_info = {}
        if not self.fake_env:
            ctrl_info = self._apply_action(action, driver=driver)

        self.curr_path_length += 1
        # POLICY PATH — unchanged, and required to stay unchanged: one target,
        # one sleep, one ``controller.step(xi)`` without ``dt``.  A held first
        # tick also comes here, so an EXTERNAL_HOLD / NO_IK / STEP_LIMIT window
        # stays held for its whole duration instead of being un-held by a
        # substep (and controller.reset() has just moved T_cmd out from under
        # the wrapper's anchor, which a substep must not chase).
        if driver is None or self.fake_env or ctrl_info.get("held"):
            dt = time.time() - start_time
            time.sleep(max(0, (1.0 / self.hz) - dt))
        else:
            ctrl_info = dict(ctrl_info)
            first_scale = float(ctrl_info.get("governed_scale", 1.0))
            extra = self._drive_intervention_substeps(
                driver, start_time + (1.0 / self.hz)
            )
            # dict.update would let the substeps' governed_scale erase the first
            # target's. The window's answer is the tightest cut anywhere in it,
            # so fold the first target in here — this is the only scope that
            # sees both _apply_action's ctrl_info and the substep aggregate.
            if "governed_scale" in extra:
                extra["governed_scale"] = min(
                    float(extra["governed_scale"]), first_scale
                )
            ctrl_info.update(extra)

        # Close the recording window HERE — after the last command this
        # transition can contain, immediately before ``obs_{k+1}`` is sampled.
        # A no-op unless the background follower committed something (or braked)
        # since the previous step, so the policy path and the in-window path are
        # untouched.
        if not self.fake_env:
            ctrl_info = self._harvest_follow_window(ctrl_info)

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
        # governed/governed_scale get defaults for the same reason held/clipped
        # do. Without them a HELD window is the ONE window with no governor keys
        # at all (the hold branch of _apply_action returns early, before any
        # controller.step), so the info schema would depend on which branch ran
        # — exactly what the sentence above says it must not do.
        info = {
            "succeed": bool(reward),
            "held": False,
            "clipped": False,
            "governed": False,
            "governed_scale": 1.0,
        }
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
        # FIRST, before the latch check and before anything moves: an episode
        # boundary is unconditionally a POLICY-owned move (go_to_reset streams a
        # multi-second joint target), so the human follower cannot be left
        # driving into it.  Disarming BEFORE the fault re-raise is deliberate —
        # if the raise came first, a latched fault would leave a follower armed
        # on every subsequent reset attempt.
        #
        # And a disarm alone is not enough: ``go_to_reset`` republishes its
        # target at 20 Hz against a follower publishing at 30 Hz into a
        # last-write-wins backend, so it needs PROOF the follower is parked, not
        # just a request that it stop.  Refusing here is the correct failure —
        # the alternative is ten seconds of blind motion that ends in "reset did
        # not arrive" and blames the arm.
        quiescent = self.await_follower_quiescent()
        self._force_policy_ownership()
        if not quiescent:
            raise RuntimeError(
                "intervention follower did not confirm it stopped within "
                f"{self.FOLLOW_QUIESCE_TIMEOUT_S}s — refusing to reset, because "
                "go_to_reset() streams its target at 20 Hz and a live follower "
                "at 30 Hz would win every write"
            )
        self._raise_follow_fault()
        if self.save_video:
            self._save_video_recording()
        self.curr_path_length = 0
        self.terminate = False
        self._hold_next_action_reason = None

        options = kwargs.get("options") or {}
        operator_approved_home = bool(options.get("operator_approved_home", False))

        if not self.fake_env:
            self.go_to_reset(operator_approved=operator_approved_home)
            self.backend.reset_command_stream()  # next target re-seeds from measured q
            q, _, _ = self.backend.get_joint_state()
            self.controller.reset(q)
            # After the move, not before: the drop point is then the fixed reset
            # pose rather than wherever the last episode happened to end.
            self.open_gripper_for_reset()

        self._update_currpos()
        return self._get_obs(), {"succeed": False}

    def go_to_reset(self, *, operator_approved: bool = False):
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

        THE ONE PLACE that publishes a joint command without going through
        ``_emit_arm_command``, and therefore the one place that has to check
        command ownership by hand.  It is grandfathered because it is not a
        delta at all — it is an absolute, branch-mapped, distance-gated joint
        target streamed over seconds — but a background follower driving the arm
        at the same time would be two authorities on one stream, so this refuses
        rather than races.  It is a ``RuntimeError`` and not an ``assert``: an
        assert disappears under ``python -O``, and this gate must not.

        TODO(together): RANDOM_RESET (task-space offset around the init pose),
        UR fault recovery before moving.
        """
        with self._command_lock:
            if self._command_owner != self.OWNER_POLICY:
                raise RuntimeError(
                    "go_to_reset() requires POLICY command ownership, but the "
                    f"arm is owned by {self._command_owner} (the background "
                    "intervention follower is driving) — disarm it first "
                    "(env.disarm_intervention_follow(), which reset() does "
                    "automatically)"
                )
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
        if gap > self.config.RESET_MAX_DIST_RAD and not operator_approved:
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

    #: position_percent at or below which the gripper counts as OPEN. Same
    #: 0.15 band _send_gripper_command uses for its "already open" test, so the
    #: reset confirmation and the policy channel agree on what open means.
    GRIPPER_OPEN_CONFIRM: float = 0.15

    def open_gripper_for_reset(self, timeout_s: float = 2.0) -> bool:
        """Open the gripper at the episode boundary. Returns True if confirmed.

        WHY: neither reset() nor go_to_reset() used to touch the gripper, so an
        episode that ended with the policy holding something started the NEXT
        episode still closed.  Every offline demo starts from an open gripper,
        so a closed start is out of distribution before the policy has taken a
        single action — and `gripper_position` is state[0], the first element
        the encoder sees.

        Deliberately NOT routed through _send_gripper_command: that one is the
        POLICY's 3-state channel and carries the GRIPPER_SLEEP debounce plus a
        "skip if already there" test.  A reset is not a policy action; it must
        command the open unconditionally and then confirm it happened, because
        the first observation of the episode is read right after.

        Skipped when the gripper channel is disabled (ACTION_SCALE[2] == 0, e.g.
        tests/run_real_hil.py without --gripper): that config deliberately
        isolates the arm path, and opening here would move hardware the
        operator asked us to leave alone.

        The scale is the robotiq_gripper_modbus_node contract documented on
        _send_gripper_command: command_percent 0.0 = OPEN .. 1.0 = CLOSED.
        """
        if self.fake_env or float(self.action_scale[2]) == 0.0:
            return True

        self.backend.send_gripper_percent(0.0)  # 0.0 = OPEN
        # Charge the debounce so the policy's first step cannot immediately
        # re-close on top of an actuation that is still travelling.
        self.last_gripper_act = time.time()

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            pos, age = self.backend.get_gripper_percent()
            if pos is not None and age < 1.0 and pos <= self.GRIPPER_OPEN_CONFIRM:
                return True
            time.sleep(0.05)

        pos, age = self.backend.get_gripper_percent()
        # A warning, not a raise: the arm is already at the reset pose and the
        # episode can still run — but the operator must know the first
        # observation is off-distribution, because nothing else will say so.
        print(
            f"[UR7eEnv] WARNING: gripper did not confirm OPEN within {timeout_s}s "
            f"(position_percent={pos}, age={age:.2f}s) — this episode starts "
            "out of distribution (every demo starts open)"
        )
        return False

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
            crop = self.config.IMAGE_CROP.get(key)
            cropped = bgr if crop is None else crop.apply(bgr)
            # One recipe, shared with the demo converter and the classifier
            # (ur_env/observation_preprocess): crop -> resize -> RGB -> uint8.
            # Spelling it out here a fourth time is how G15 happened.
            images[key] = preprocess_frame(
                bgr,
                crop=crop,
                size=self.observation_space["images"][key].shape[:2][::-1],
            )
            display_images[key] = cv2.resize(  # BGR for cv2.imshow
                cropped, self.observation_space["images"][key].shape[:2][::-1]
            )
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

    def _force_policy_ownership(self) -> None:
        """Disarm, hand the arm back to POLICY and drop the recording window.

        Used at every episode boundary and on the way out.  Idempotent, and safe
        on a fake env (the mux attributes are declared before the ``fake_env``
        early return precisely so this needs no guard).
        """

        with self._command_lock:
            self._follow_armed.clear()
            self._command_owner = self.OWNER_POLICY
            self._follow_source = None
            self._follow_brake_reason = None
            self._follow_needs_reanchor = False
            self._reset_follow_window_locked()

    def close(self):
        """Teardown, in an order the follower thread makes load-bearing.

        ``armed.clear()`` -> ``owner = POLICY`` -> ``stop.set()`` -> ``join`` ->
        ``backend.close()``.

        The join has to complete BEFORE ``backend.close()``.  The backend's own
        teardown stops the 250 Hz publisher and then destroys the node, and
        ``run_hil_actor.sh``'s cleanup waits on that publisher disappearing to
        decide the arm is quiet; a follower still alive across it would be
        calling ``send_joint_command`` into a closing backend (a no-op after
        ``_closed``, but the wait would have proven nothing).  Clearing ``armed``
        first, rather than relying on the join alone, means a follower stuck
        anywhere cannot issue one more command while we wait for it.
        """

        self._force_policy_ownership()
        self._follow_stop.set()
        thread = self._follow_thread
        if thread is not None and thread.is_alive():
            thread.join(self.FOLLOW_JOIN_TIMEOUT_S)
            if thread.is_alive():
                # Not fatal: it is disarmed and owns nothing, so it can no longer
                # command the arm.  Say so rather than hanging the shutdown.
                print(
                    "[UR7eEnv] WARNING: intervention follower did not exit "
                    f"within {self.FOLLOW_JOIN_TIMEOUT_S}s (it is disarmed and "
                    "cannot command the arm)"
                )
        if not self.fake_env:
            self.backend.close()
