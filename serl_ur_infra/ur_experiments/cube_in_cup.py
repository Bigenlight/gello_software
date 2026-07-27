"""``cube_in_cup`` task configuration — the UR7e analog of an upstream
``examples/experiments/<task>/config.py``.

This module is what ``--ur-config-module`` loads.  It supplies the two things
the upstream entry points look for and this repo never had:

* a robot/task config (poses, limits, cameras) — ``CubeInCupEnvConfig``
* ``get_environment()``, which assembles the wrapper chain — ``CubeInCupConfig``

Chain (upstream ``usb_pickup_insertion/config.py:114-136`` order)::

    UR7eEnv -> GelloIntervention -> RelativeFrame -> Quat2EulerWrapper
            -> SERLObsWrapper -> ChunkingWrapper(1, None)

Two deliberate differences from upstream, both because of how this rig splits
work between the robot laptop and the remote GPU:

* **No reward-classifier wrapper.** Reward and termination are decided by the
  receive server (``rlpd_receive_server.RewardTransitionFinalizer``), which
  owns the classifier checkpoint.  ``get_environment(classifier=True)`` raises
  rather than silently returning an env whose local reward would be ignored.
* **No ``GripperPenaltyWrapper`` here.** ``run_remote_rlpd_actor.py`` adds it
  through ``wrap_gripper_penalty_from_task_config`` so the penalty always comes
  from the task config rather than a wrapper default.  Adding it here too would
  double-apply it.

The package is named ``ur_experiments`` and not ``experiments`` on purpose:
the actor entry point puts ``third_party/hil-serl/examples`` on ``sys.path``
ahead of ``serl_ur_infra``, so an ``experiments`` package here would be
shadowed by upstream's.
"""

from typing import Any, Optional

import numpy as np

from ur_env.envs.chunking import ChunkingWrapper
from ur_env.envs.config import DefaultUR7eEnvConfig
from ur_env.envs.frame_wrappers import Quat2EulerWrapper, RelativeFrame
from ur_env.envs.ur7e_env import UR7eEnv
from ur_env.envs.wrappers import GelloIntervention
from ur_env.observation_schema import PROPRIO_KEYS

# ``UNSET`` marks a field that must be measured on the real rig before this
# task can run.  Leaving them as zeros -- the DefaultUR7eEnvConfig default --
# is how you end up streaming the arm toward a fully-extended horizontal pose
# or advertising a zero-volume workspace box that clamps everything.  A config
# review found exactly that hazard, so unmeasured values fail loudly instead.
UNSET = None


class CubeInCupEnvConfig(DefaultUR7eEnvConfig):
    """Physical task configuration for "put the cube in the cup"."""

    # --- episode start pose -------------------------------------------------
    # Median first-frame ur_joint_states over the 23 usable cube_in_cup takes
    # (take_23 excluded: a 1.64 s recording with the arm parked).  Joint order:
    # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3.
    #
    # NOT the value in the ACT/Diffusion/FM deploy configs
    # (ros2_ur_ws/src/gello_policy/config/act_deploy.yaml:35 and siblings,
    # [3.106, -1.817, 1.653, -1.618, -1.628, -3.195]).  Those policies were
    # trained on "put right banana in pot", and that literal is the *banana*
    # start pose: it differs from cube_in_cup by 0.29 rad at the shoulder,
    # putting the TCP 13 cm higher and 10 cm further back.  Measuring the two
    # datasets separately is what caught this; the yaml comment only says "the
    # dataset's start pose" without naming which dataset.
    #
    # Median rather than mean: they agree to <0.02 rad here, and the median is
    # the robust choice.  Per-joint spread is 0.05-0.10 rad, i.e. the operator
    # really did return the arm to one pose between takes -- unlike the banana
    # session, whose shoulder spread is 0.32 rad (no fixed reset pose).
    # The resulting TCP (0.506, 0.136, 0.423) sits inside ABS_POSE_LIMIT below.
    #
    # WRAP HAZARD: shoulder_pan sits ~0.003 rad past +pi and wrist_3 ~0.008 rad
    # past -pi.  Any comparison against this pose must be branch-cut safe (see
    # ur_gello_bringup.angle_utils.nearest_equivalent); a naive difference
    # reports these joints as ~2*pi away from an identical physical pose.
    RESET_JOINTS: np.ndarray = np.array(
        [3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331]
    )

    # Reset streams to RESET_JOINTS through the same 250 Hz upsampler used for
    # normal steps, sweeping blindly through whatever is on the way.  Keep this
    # tight so a reset from an unexpected pose is refused rather than executed;
    # pre-position with the proven move-to-start tooling instead.
    RESET_MAX_DIST_RAD: float = 0.5
    RESET_TOLERANCE_RAD: float = 0.02
    RESET_TIMEOUT_S: float = 10.0

    # --- tool / reference point ---------------------------------------------
    # EVERYTHING BELOW IS IN THE **FLANGE** (tool0) FRAME, and these three
    # settings are one coupled set -- change one and the workspace floor moves
    # 17.4 cm without any error.
    #
    # The Robotiq 2F-85 really is 174 mm along the flange z, but the pendant
    # TCP was 0 when cube_in_cup was recorded.  Verified rather than assumed:
    # replaying ur_joint_states through UR7e forward kinematics reproduces the
    # recorded /tcp_pose_broadcaster/pose to a median 0.6 mm as the *flange*,
    # versus 174.2 mm under the gripper-tip hypothesis.  Cross-check: the
    # measured flange z floor of 0.1785 minus 0.174 is 0.0045 m, i.e. the
    # fingertips exactly touching the table.
    #
    # We therefore stay in the flange frame end to end.  PolicyDeltaController
    # also integrates at the flange (fk(q_now), no T_tool), so this is the only
    # choice that keeps the controller, the observation and the box in one
    # frame.  Moving to true-TCP semantics means doing all four together:
    #   TCP_POSE_SOURCE -> "fk"
    #   TCP_OFFSET_XYZ_RPY -> [0, 0, 0.174, 0, 0, 0]
    #   z limits -> shifted down by 0.174  (0.0045 .. 0.376)
    #   PolicyDeltaController -> integrate at fk(q) @ T_tool
    TCP_POSE_SOURCE: str = "driver"
    TCP_OFFSET_XYZ_RPY: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # --- workspace box ------------------------------------------------------
    # EEF-space limits rather than joint limits: HIL-SERL explores, and a joint
    # box lets the arm reach the same pose through an unexpected elbow
    # configuration -- exactly the surprise we are trying to avoid.
    #
    # Derived from 23 cube_in_cup takes (19,802 samples; take_23 dropped, it is
    # a 1.64 s recording with the arm parked).  Position: x/y expanded to
    # 1.5x the demonstrated range about its centre for exploration headroom;
    # z_low left at the measured minimum because that is the table, a physical
    # hard stop with no headroom to give.  Rotation follows the same 1.5x rule
    # rather than being opened up: a large tilt slides the fingertips sideways
    # while still respecting z_low, and pushes wrist_3 toward its limit.
    # 100% of the demonstrated samples fall inside this box.
    #
    # Index 3 is |rx|: the tool points straight down, so rx lives near +-pi and
    # clip_safety_box clips the magnitude and restores the sign.  A naive
    # np.clip(rx, low, high) would flip the wrist on the ~12% of samples that
    # land on the negative branch.
    ABS_POSE_LIMIT_LOW: Optional[np.ndarray] = np.array(
        [0.375, -0.229, 0.1785, 2.60, -0.30, 1.10]
    )
    ABS_POSE_LIMIT_HIGH: Optional[np.ndarray] = np.array(
        [0.642, 0.272, 0.550, np.pi, 0.35, 2.20]
    )

    # --- cameras ------------------------------------------------------------
    # Square crops, so the 128x128 resize introduces no aspect distortion.  With
    # IMAGE_CROP={} the 1280x720 frame is squashed to 1:1, compressing the
    # horizontal axis to 0.5625 of the vertical.  Coordinates are in the raw
    # 1280x720 frame, applied as img[y0:y1, x0:x1] before the resize.
    #
    # cam1 is the fixed tripod scene camera.  cam2 is the WRIST camera, rigidly
    # mounted to the gripper: its crop is centred on the measured grasp axis
    # (x=781, midway between the fingertips at x 512-563 and x 1000-1117) rather
    # than on a table region, because the fingers stay at fixed pixels while the
    # background moves with the arm.
    #
    # KNOWN CONFLICT -- classifier preprocessing does NOT crop:
    #   reward_classifier_runtime.decode_classifier_image() resizes the full
    #   frame, and its docstring calls that "the exact observation tree used to
    #   initialize the checkpoint".  But rlpd_receive_server._classifier_
    #   observation() feeds the classifier this env's *cropped* canonical
    #   observation.  Enabling a crop therefore takes classifier input out of
    #   distribution, and the classifier is authoritative for reward/done.
    #   Harmless while reward is stubbed (our loop ignores it); must be resolved
    #   -- retrain on crops, or split policy/classifier preprocessing -- before
    #   any run that trusts the reward signal.
    IMAGE_CROP: Optional[dict] = {
        "cam1": lambda img: img[20:670, 340:990],   # 650x650
        "cam2": lambda img: img[0:720, 420:1140],   # 720x720
    }

    # --- reward -------------------------------------------------------------
    # Intentionally left at the zero defaults: the receive server's classifier
    # is authoritative for reward and termination, so UR7eEnv.compute_reward()
    # is never consulted in this deployment.  Episodes end on MAX_EPISODE_LENGTH
    # locally and on classifier success remotely.
    MAX_EPISODE_LENGTH: int = 100  # 10 s at HZ=10

    GRASP_PENALTY: float = -0.02  # upstream task value

    # Safety: never publish to the arm until an operator has reviewed the whole
    # chain on this rig.  run_real_hil.py / the actor must flip this explicitly.
    DRY_RUN: bool = True

    def __init__(self) -> None:
        missing = [
            name
            for name in ("ABS_POSE_LIMIT_LOW", "ABS_POSE_LIMIT_HIGH", "IMAGE_CROP")
            if getattr(self, name) is UNSET
        ]
        if missing:
            raise ValueError(
                "cube_in_cup is not commissioned yet: "
                + ", ".join(missing)
                + " still need real-rig measurements. Fill them in "
                "ur_experiments/cube_in_cup.py -- do NOT substitute zeros, "
                "which read as 'a zero-volume workspace' and 'no crop'."
            )
        for name in ("ABS_POSE_LIMIT_LOW", "ABS_POSE_LIMIT_HIGH"):
            box = np.asarray(getattr(self, name), dtype=float)
            if box.shape != (6,):
                raise ValueError(f"{name} must be a 6-vector, got {box.shape}")
        low = np.asarray(self.ABS_POSE_LIMIT_LOW, dtype=float)
        high = np.asarray(self.ABS_POSE_LIMIT_HIGH, dtype=float)
        if not np.all(high > low):
            raise ValueError(
                "ABS_POSE_LIMIT_HIGH must exceed ABS_POSE_LIMIT_LOW on every "
                f"axis; got low={low}, high={high}"
            )


class CubeInCupConfig:
    """Training-side config: the fields the actor entry points read."""

    #: Actor protocol v2 forbids random warm-up actions -- the server validates
    #: that a non-intervened transition's action equals the action it issued,
    #: and a sampled action would also violate the {-1, 0, 1} gripper contract.
    random_steps: int = 0
    max_steps: int = 1_000_000
    #: 0 disables the local transition backup; set >0 to also dump demo pickles.
    buffer_period: int = 0

    proprio_keys = list(PROPRIO_KEYS)

    #: Defaults only; --server-host/--server-port override these.
    NETWORK = {
        "type": "grpc",
        "host": "127.0.0.1",
        "port": 50053,
        "timeout_s": 0.6,
        "max_response_age_s": 0.8,
        "retry_count": 1,
    }

    def __init__(self) -> None:
        self.robot_config = CubeInCupEnvConfig()

    def get_environment(
        self,
        fake_env: bool = False,
        save_video: bool = False,
        classifier: bool = False,
        deadman: Any = None,
    ):
        if classifier:
            raise NotImplementedError(
                "reward is server-authoritative on this rig: the classifier "
                "runs in rlpd_receive_server, not in a robot-side wrapper. "
                "Call get_environment(classifier=False)."
            )

        env = UR7eEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=self.robot_config,
        )
        if not fake_env:
            # Needs the live GELLO leader; upstream gates SpacemouseIntervention
            # the same way.  Must sit inside RelativeFrame: it emits base-frame
            # deltas, which RelativeFrame converts back to the policy frame
            # before they are stored.
            env = GelloIntervention(env, deadman=deadman)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)

        from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        return ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
