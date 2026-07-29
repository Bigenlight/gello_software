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

import time
from typing import Any, Optional

import numpy as np

from ur_env.envs.chunking import ChunkingWrapper
from ur_env.envs.config import DefaultUR7eEnvConfig
from ur_env.envs.frame_wrappers import Quat2EulerWrapper, RelativeFrame
from ur_env.envs.ur7e_env import UR7eEnv
from ur_env.envs.wrappers import (
    GelloIntervention,
    RosTopicDeadman,
    SpacebarDeadman,
)
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
    # fk(RESET_JOINTS) = (0.5036, 0.1365, 0.4138), inside ABS_POSE_LIMIT below.
    # (Not (0.506, 0.136, 0.423) -- that is the mean of the per-take FK
    # positions, which is a different quantity and 9 mm off in z.)
    #
    # WRAP HAZARD: shoulder_pan sits ~0.003 rad *inside* +pi and wrist_3 ~0.008
    # rad inside -pi, i.e. both are just within (-pi, pi] -- but the task is
    # not: 96% of demonstrated samples have wrist_3 < -pi and 4% have
    # shoulder_pan > +pi, so the work manifold straddles the branch cut even
    # though the reset pose does not.  Any comparison against this pose must be
    # branch-cut safe (ur_kin.wrapped_nearest); a naive difference reports
    # these joints as ~2*pi away from an identical physical pose, which is
    # exactly what made go_to_reset refuse a pose 0.30 rad away.
    RESET_JOINTS: np.ndarray = np.array(
        [3.1382, -1.5276, 1.7168, -1.7592, -1.5216, -3.1331]
    )

    # Reset streams to RESET_JOINTS through the same 250 Hz upsampler used for
    # normal steps, sweeping blindly through whatever is on the way, so this is
    # a real limit and not a formality.  It still has to clear the poses the
    # task actually ends in: measured over the 23 cube_in_cup takes, the
    # branch-safe distance from the final frame back to RESET_JOINTS has median
    # 0.615 rad and max 0.774 (wrist_1 dominates).  0.5 -- the value this
    # started at -- refuses 16 of 23 takes, i.e. reset() raises after most
    # normal episodes.  0.9 clears every recorded take with headroom while
    # still refusing a reset from an arbitrary parked pose.
    RESET_MAX_DIST_RAD: float = 0.9
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
    # contact-free flange z floor of 0.1808 minus 0.174 is 0.0068 m, i.e. the
    # fingertips just above the table.
    #
    # We therefore stay in the flange frame end to end.  PolicyDeltaController
    # also integrates at the flange (fk(q_now), no T_tool), so this is the only
    # choice that keeps the controller, the observation and the box in one
    # frame.  Moving to true-TCP semantics means doing all four together:
    #   TCP_POSE_SOURCE -> "fk"
    #   TCP_OFFSET_XYZ_RPY -> [0, 0, 0.174, 0, 0, 0]
    #   z limits -> shifted down by 0.174  (0.011 .. 0.376)
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
    # 1.5x the demonstrated range about its centre for exploration headroom.
    # Rotation is NOT opened up beyond roughly the same rule: a large tilt
    # slides the fingertips sideways while still respecting z_low, and pushes
    # wrist_3 toward its limit.
    #
    # z_low is NOT the demonstrated minimum.  That minimum (0.1785) comes from
    # a single take: all 40 samples at it are take_11, where the gripper closed
    # on nothing (grip_pos 0.05-0.18) and pressed the table at -24 to -133 N for
    # 0.4 s.  It is a failed grasp driven into the surface, not the surface.
    # Filtering to contact-free samples (fz > -5 N) puts the actual table at
    # 0.1808, i.e. the demonstrated minimum is 2.3 mm *below* it.  That matters
    # more than it looks: PolicyDeltaController writes the clamped pose back
    # into its integrator, so a policy pushing down would park the commanded
    # flange at exactly the depth that took 133 N, and nothing in this loop
    # limits force.  0.185 clears the free-space surface, drops only take_11's
    # 73 samples (0.37%), and still sits 9 mm below the lowest successful grasp
    # (0.1941).  Use 0.190 if the table or base mount is ever re-seated -- 0.5
    # degrees of tilt across this box is 5 mm of z.
    # Every other demonstrated sample falls inside this box.
    #
    # Index 3 is |rx|: the tool points straight down, so rx lives near +-pi and
    # clip_safety_box clips the magnitude and restores the sign.  A naive
    # np.clip(rx, low, high) would flip the wrist on the ~12% of samples that
    # land on the negative branch.
    ABS_POSE_LIMIT_LOW: Optional[np.ndarray] = np.array(
        [0.375, -0.229, 0.185, 2.60, -0.30, 1.10]
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
    # KNOWN CONFLICT (G15) -- the pinned classifier was trained WITHOUT a crop:
    #   its training pipeline resized the full 1280x720 frame straight to
    #   128x128 (kanu hil-serl examples/cube_classifier_pipeline.py, whose
    #   export passes crop=None), while get_im() crops first.  Measured
    #   2026-07-28: bit-exact pixel match to the no-crop hypothesis, and
    #   recall@0.85 falls 100% -> 33.3% when the crop is applied.
    #   rlpd_receive_server._classifier_observation() performs NO image
    #   transform -- it forwards this env's cropped canonical observation
    #   unchanged -- so the mismatch originates here, not on the server.
    #
    #   DO NOT "fix" this by removing the crop.  These boxes are measured and
    #   serve the policy, which is the primary consumer.  The resolution is to
    #   retrain the classifier against these crops (the run that produced the
    #   current checkpoint took 46 s, and cube_classifier_pipeline.py already
    #   accepts --cam1-crop/--cam2-crop):
    #       cam1 -> --cam1-crop 340,20,990,670      (this file stores y0,y1,x0,x1;
    #       cam2 -> --cam2-crop 420,0,1140,720       the pipeline takes x0,y0,x1,y1)
    #
    #   Until that lands the classifier is out of distribution, and it is
    #   authoritative for reward AND done.  See docs/testing/08_OPEN_GAPS.md
    #   G15 and HIL_SERL_LEARNER_STATUS_AND_NEXT_KO.md §12.
    #
    #   Do not confuse this with
    #   gello_recorder.reward_classifier_runtime.decode_classifier_image().
    #   That function resizes the FULL frame with no crop -- it is correct for
    #   what it serves, the standalone ZMQ viewer (remote_reward_classifier_
    #   server.py), which subscribes to the raw camera topics and never sees a
    #   canonical observation.  It has NO call path to the gRPC receive server,
    #   so "fixing" it does nothing for this defect.  An earlier version of
    #   this comment named it as the cause; that was wrong and cost a session.
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

    @staticmethod
    def _resolve_deadman(deadman: Any, env: Any) -> Any:
        """Turn a ``--deadman`` string into a live DeadmanSource.

        Callers cannot build ``RosTopicDeadman`` themselves: it subscribes on
        the URRosBackend rclpy node, which does not exist until ``UR7eEnv`` has
        been constructed.  So the entry point passes a *spec* and resolution
        happens here, once the node is available.

        A non-string is passed through untouched -- tests supply a stub object
        and assert identity.  ``None`` keeps the historical fallback, which is
        ``SpacebarDeadman``; that listener is global and has no watchdog, so
        prefer ``"topic"`` on the real rig.
        """

        if not isinstance(deadman, str):
            return deadman
        if deadman == "spacebar":
            print(
                "[deadman] spacebar: global pynput listener, no heartbeat and "
                "no watchdog -- SPACE in ANY window engages. Prefer 'topic'.",
                flush=True,
            )
            return SpacebarDeadman()
        if deadman != "topic":
            raise ValueError(
                f"unknown deadman {deadman!r}; expected 'topic' or 'spacebar'"
            )

        source = RosTopicDeadman(env.backend._node)
        # A GUI that never started is otherwise SILENT: RosTopicDeadman fails
        # safe to "not engaged", so the operator sees no error at all -- just an
        # intervention that never triggers.  Wait for one heartbeat and say so.
        deadline = time.monotonic() + 15.0
        while source._last_rx is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if source._last_rx is None:
            raise RuntimeError(
                "no /hil/deadman message after 15 s -- start the HIL GUI "
                "(ros2_ur_ws/run_hil_gui.sh) before the actor"
            )
        print(f"[deadman] /hil/deadman ok (gain={source.gain():.2f})", flush=True)
        return source

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
            env = GelloIntervention(env, deadman=self._resolve_deadman(deadman, env))
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)

        from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        return ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
