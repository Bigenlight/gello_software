"""Configuration for UR7eEnv.

Mirrors franka_env.envs.franka_env.DefaultEnvConfig so per-task configs look
the same as the hil-serl examples (ram_insertion/config.py etc.), with the
Franka-only knobs (COMPLIANCE_PARAM, LOAD_PARAM, ...) replaced by the
governor / safety knobs our stiff forward_position_controller path needs.
"""

from typing import Callable, Dict

import numpy as np


class DefaultUR7eEnvConfig:
    """Fill these in per task (subclass, like RAMEnv's EnvConfig does)."""

    # ---- cameras ---- #
    # Unlike franka_env (RSCapture opens the device in-process), we subscribe to
    # the realsense2_camera driver topics from launch_cameras.sh: a RealSense
    # cannot be opened twice, and this keeps the existing viewer/recorder
    # workflow usable alongside RL. Keys become the image observation keys.
    CAMERAS: Dict[str, str] = {
        "cam1": "/cam1/cam1/color/image_raw/compressed",   # scene (tripod)
        "cam2": "/cam2/cam2/color/image_raw/compressed",   # close-up
    }
    IMAGE_CROP: Dict[str, Callable] = {}
    IMAGE_OBS_SIZE: tuple = (128, 128)   # (H, W) of image observations
    IMAGE_STALE_S: float = 0.5   # newest frame older than this -> treat as failure
    DISPLAY_IMAGE: bool = True

    # ---- episode ---- #
    HZ: float = 10.0
    MAX_EPISODE_LENGTH: int = 100

    # ---- task poses (xyz + rpy, UR base frame, meters/radians) ---- #
    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    RESET_POSE: np.ndarray = np.zeros((6,))
    RANDOM_RESET: bool = False
    RANDOM_XY_RANGE: float = 0.0
    RANDOM_RZ_RANGE: float = 0.0

    # ---- episode reset (init pose) ---- #
    # Joint-space on purpose: task-space + IK would leave the arrival branch
    # ambiguous, and RL wants the exact same configuration every episode.
    # RESET_POSE above stays the *task-space* description of the same pose
    # (fake-mode obs, randomization); measure both from the same physical pose.
    RESET_JOINTS: np.ndarray = np.zeros((6,))
    RESET_TOLERANCE_RAD: float = 0.02   # per-joint arrival tolerance
    RESET_TIMEOUT_S: float = 10.0
    # Streaming reset sweeps whatever lies between here and RESET_JOINTS —
    # refuse long moves; pre-position with the move-to-start tooling instead.
    RESET_MAX_DIST_RAD: float = 1.5

    # ---- action scaling: [pos (m/step), rot (rad/step), gripper] ---- #
    # action in [-1,1]^7; per-step task-space increment = action * ACTION_SCALE.
    #
    # INVARIANT (buffer correctness): ACTION_SCALE * HZ must stay BELOW the
    # governor caps (v_max/w_max) with headroom. Otherwise a full-scale action
    # claims a displacement the governor then cuts, and every stored
    # (obs, action, next_obs) transition — policy AND GELLO intervention —
    # systematically overstates the motion. The governor is a safety net, not
    # a working limit; UR7eEnv.__init__ warns if this invariant is violated.
    #
    # Working speed is pinned to the PROVEN EEF teleop stack, not chosen freely.
    # The three layers below are a uniform 1.25x of the original values, which
    # lands UPSAMPLER.max_step_rad exactly on the teleop value (see below) —
    # that per-joint slew cap is what actually protects the hardware, and it is
    # the limit that binds during fast motion in teleop too
    # (ur7e_gello_eef.yaml:234-238). Keep the three layers in this ratio: raise
    # one alone and the next silently truncates it, which re-breaks the
    # invariant above.
    ACTION_SCALE: np.ndarray = np.array([0.0125, 0.0625, 1.0])
    # DO NOT raise this because "the intervention feels slow". This number IS
    # the meaning of a policy action (1.0 = 12.5 cm/s = the policy's top speed);
    # raise it and the human demonstrates motion the policy cannot execute, and
    # the 2,037 canonical demo actions silently change physical meaning — the
    # learner fingerprint does not carry ACTION_SCALE, so nothing rejects the
    # mismatch (08_OPEN_GAPS.md G18). To try a different hand feel for ONE run,
    # use ``tests/run_real_hil.py --scale``: it scales all three layers together
    # and stores nothing. Smoothness is INTERVENTION.substep_hz, not this.

    # ---- workspace safety box (TCP, UR base frame) ---- #
    ABS_POSE_LIMIT_LOW: np.ndarray = np.zeros((6,))
    ABS_POSE_LIMIT_HIGH: np.ndarray = np.zeros((6,))

    # ---- governor / IK safety (feeds PolicyDeltaController; the analog of ---- #
    # ---- Franka's COMPLIANCE_PARAM — software-synthesized softness).      ---- #
    # Caps sit ~20% above ACTION_SCALE * HZ (0.125 m/s, 0.625 rad/s) — pure
    # safety net; never the binding limit in normal operation (see INVARIANT).
    # Still at or below the proven teleop caps (v_max 0.16, w_max 1.0 in
    # ur7e_gello_eef.yaml:238,241), so this stack is not faster than what has
    # already run on this arm.
    GOVERNOR: Dict[str, float] = {
        "v_max": 0.15,          # m/s   task-space translational rate cap
        "w_max": 0.75,          # rad/s task-space rotational rate cap
        "dq_step_max": 0.0625,  # rad   per-tick joint step acceptance gate
    }

    # ---- 250 Hz command upsampler ---- #
    # forward_position_controller does not interpolate (bridge docstring), so
    # the backend streams bounded steps toward the latest 10 Hz target. This is
    # actuator-side trajectory shaping only: EMA/One-Euro remains deliberately
    # absent, because filtering policy actions would blur action->effect credit
    # assignment in replay.
    UPSAMPLER: Dict[str, float] = {
        "hz": 250.0,
        # per-tick joint step: 0.0025 rad @ 250 Hz = 0.625 rad/s, the same rate
        # ceiling as GOVERNOR.dq_step_max (0.0625 rad @ 10 Hz).
        #
        # 0.0025 is EXACTLY the proven teleop value (ur7e_gello.yaml:64, same
        # 250 Hz upsampler). Do NOT raise it past ~0.003 while at 250 Hz: the
        # driver runs a 500 Hz cycle, so the per-driver-cycle step is half this,
        # and ur7e_gello.yaml:56-63 pins ~0.00314 rad as the ceiling there.
        # Going faster means raising hz first, not this.
        "max_step_rad": 0.0025,
        # Finite acceleration replaces the old instantaneous 0 -> 0.625 rad/s
        # velocity jump. 8 rad/s^2 is the conservative offline replay candidate
        # already used for the recorded UR sessions
        # (scripts/ur_command_smoothing.py and analyze_replay.py): it reaches the
        # existing speed ceiling in ~78 ms and needs ~0.026 rad worst-joint
        # discrete braking distance at that ceiling. It changes neither the
        # 0.0025 rad command-step cap nor policy/replay action values.
        "max_accel_rad_s2": 8.0,
        # Mirror the real GELLO bridge's proven anti-snap seed ramp
        # (ur7e_gello.yaml): begin at 15% of the step ceiling and reach full
        # speed over 0.7 s. The first command itself is still measured q exactly.
        "soft_start_s": 0.7,
        "soft_start_fraction": 0.15,
        # The policy loop is 10 Hz. Permit normal scheduling jitter/two missed
        # updates, but after roughly three target periods stop chasing the old
        # goal: acceleration-limit to zero and publish the current stream HOLD.
        "target_stale_s": 0.30,
        # DO NOT revert max_accel_rad_s2 / target_stale_s / soft_start_s because
        # the INTERVENTION feels stiff. Measured 2026-07-30, all three move the
        # wrong way: dropping the accel limit takes fully-stopped time 16% ->
        # 76%, target_stale_s 0.30 -> 0.50 takes it 17.5% -> 54.5%, and
        # soft_start_s 0.7 -> 0 takes ripple 1.92 -> 4.09. The stiffness came
        # from the target UPDATE RATE, fixed by INTERVENTION.substep_hz below.
        # Numbers and method: docs/testing/04_HIL_INTERVENTION.md §9.3.
    }

    # ---- human intervention: in-window leader resampling ---- #
    # WHY THIS EXISTS.  ``UR7eEnv.step`` used to set the joint target exactly
    # ONCE per 100 ms window and then sleep, never re-reading the leader.  The
    # 250 Hz upsampler therefore replayed "accelerate -> arrive -> brake ->
    # sit still" every single window: measured at a 0.15 rad/s leader, the
    # joints were fully stopped 16% of a nominal window and 40% of a real
    # (0.197 s) one, with 2.27 units of ripple.  Refreshing the target at 30 Hz
    # instead measured 0% stopped time and 0.85 ripple.  The operator feels the
    # difference as "stiff/notchy" vs "continuous".
    #
    # 30.0 is NOT a free parameter.  It is (a) exactly the leader publish rate
    # of the PROVEN teleop stack -- ``gello_publisher.publish_rate_hz: 30.0``
    # in ros2_ur_ws/src/ur_gello_bringup/config/ur7e_gello.yaml:26 -- and
    # (b) exactly the rate at which the backend's leader cache actually turns
    # over (``URRosBackend._on_gello``, ur_env/envs/ros_backend.py:528-532, is
    # driven by that same publisher).  Substepping faster than 30 Hz would
    # re-issue the SAME cached leader sample under a different name, which buys
    # nothing and makes the one-euro filter's dt bookkeeping lie.  Substepping
    # slower throws away leader samples the rig already paid for.
    #
    # ``substep_hz <= HZ`` disables the feature IN "in_window" MODE: no substep
    # fits inside the window, so ``UR7eEnv.step`` degenerates to its pre-change
    # single target + sleep.  Configs without an ``INTERVENTION`` block behave
    # that way too.  In "background" mode (see ``follow_mode`` below) the tick
    # rate is not packed into the window at all, so ``substep_hz`` has no such
    # relationship to ``HZ`` — any positive rate runs.
    #
    # The one_euro_* values are the leader-side anti-tremor filter of the same
    # proven teleop config (ur7e_gello.yaml:43-48).  They are applied to the
    # LEADER joints only -- never to a policy action; filtering policy actions
    # would blur action->effect credit assignment in replay, which is why
    # UPSAMPLER above still refuses to carry any filter.
    #
    # WHO READS THESE.  ``substep_hz`` -> ``UR7eEnv.intervention_substep_hz``
    # (ur7e_env.py:193), which paces the in-window loop AND is the rate
    # ``GelloIntervention`` builds its LeaderFilter + InterventionBudget at
    # (wrappers.py:353).  ``one_euro_*`` reach ``LeaderFilter`` only, never the
    # policy path.  Full call chain: leader_stream.py module docstring.
    #
    # VERIFIED on the real UR7e 2026-07-30: ARMED ``run_real_hil.py --arm
    # --scale 1.0``, 120 intervened steps, every check PASS (substeps=2 in every
    # window = 3 target updates per window, dp_ratio median 1.000, held 0%,
    # operator confirmed the hand feel).  Evidence: 04_HIL_INTERVENTION.md §9.
    #
    # ---- follow_mode: WHERE the 30 Hz following runs ---- #
    # "background" (DEFAULT) — a daemon thread inside ``UR7eEnv`` follows the
    #   leader continuously while the human is engaged.  The RL loop is then a
    #   pure OBSERVER: it samples a transition every ``env.step`` and never gates
    #   the arm.  This is the operator's requirement, stated verbatim: "사람에게
    #   제어권이 넘어올 때는 그냥 원래 eef teleop처럼 팔이 움직이고, RL은 10 Hz로
    #   정보만 빼간다".
    #
    #   WHY IT HAD TO MOVE OUT OF THE WINDOW.  ``_drive_intervention_substeps``
    #   paces against ``start_time + 1/HZ`` — the NOMINAL window — but the
    #   production actor loop measures 512 ms per step (1.95 Hz): ~412 ms of it
    #   is the blocking gRPC Step RPC plus the camera decode, both OUTSIDE
    #   ``env.step``.  In-window following therefore refreshed the target for
    #   66.7 ms out of every 512 ms and then went silent, with three consequences
    #   that compound:
    #     * top intervention speed = ACTION_SCALE / real period = 12.5 mm /
    #       512 ms = 2.4 cm/s, against 12.4 cm/s measured on the validation rig;
    #     * the 445 ms target gap crosses UPSAMPLER.target_stale_s (0.30 s), so
    #       the upsampler BRAKED to HOLD once per window;
    #     * every stale brake re-arms ``soft_start_s`` (0.7 s), and no window is
    #       that long, so the slew ceiling sat pinned at 46%.
    #   A background thread ticks at ``substep_hz`` regardless of what the RL
    #   loop is doing, so none of the three can happen.
    #
    #   It is also the SAFETY improvement: the follower re-reads the deadman on
    #   EVERY tick, so a release stops the arm within 1/substep_hz = 33 ms
    #   instead of at the next window boundary (up to 512 ms of further motion).
    #
    #   NO ``InterventionBudget`` ON THIS PATH (operator decision, 2026-07-30).
    #   The per-window ACTION_SCALE cap was a RECORDING constraint enforced by
    #   throttling the arm, and at a 512 ms step it pinned the operator's ceiling
    #   at ACTION_SCALE/T = 2.4 cm/s — not drivable, hence zero demonstrations.
    #   The remaining bounds are the GOVERNOR above, the workspace box, and the
    #   250 Hz upsampler, all of them inside ``PolicyDeltaController.step``.
    #   The price is that a long window's stored action UNDER-reports its motion;
    #   it is measured and reported per transition as
    #   ``info["intervention_saturation"]`` rather than assumed away.  Full
    #   argument, and the two real fixes, on
    #   ``UR7eEnv._harvest_follow_window`` and ``GelloIntervention.follow_xi``.
    #
    # "in_window" — the pre-2026-07-30 path, preserved verbatim: substeps are
    #   packed into ``UR7eEnv.step``'s sleep and rationed by
    #   ``InterventionBudget``.  Kept because it is what the 2026-07-30 ARMED
    #   hardware run validated, and because the substep tests pin it.
    #
    # ``substep_hz <= 0`` (no INTERVENTION block) forces "in_window", i.e. the
    # feature is off entirely — there is no tick rate to run a follower at.
    INTERVENTION: Dict[str, float] = {
        "substep_hz": 30.0,
        "follow_mode": "background",
        "one_euro_min_cutoff": 1.0,
        "one_euro_beta": 2.0,
        "one_euro_d_cutoff": 1.0,
    }

    # ---- ROS 2 wiring (defaults match ur_gello_bringup / gello_recorder) ---- #
    ROS: Dict[str, str] = {
        "joint_states_topic": "/joint_states",
        "command_topic": "/forward_position_controller/commands",
        "gello_topic": "/gello/joint_states",
        "gripper_command_topic": "/robotiq_gripper/command_percent",
        "gripper_state_topic": "/robotiq_gripper/position_percent",
        # UR e-series built-in TCP F/T estimate (same topic the recorder uses)
        "wrench_topic": "/force_torque_sensor_broadcaster/wrench",
        # driver's TCP pose (factory-calibrated kinematics + pendant TCP offset)
        "tcp_pose_topic": "/tcp_pose_broadcaster/pose",
    }

    # Where tcp_pose observations come from:
    #   "driver" - /tcp_pose_broadcaster/pose, the proven path (recorder uses it).
    #              Requires the broadcaster in the launch and the TCP offset set
    #              on the pendant.
    #   "fk"     - our ur_kin.fk(q) @ TCP_OFFSET_XYZ_RPY. No extra broadcaster,
    #              exactly consistent with the command path's kinematics.
    # tcp_vel always comes from ur_kin.jacobian(q) @ dq — the driver has no
    # twist broadcaster.
    TCP_POSE_SOURCE: str = "driver"

    # TCP offset from the flange, xyz+rpy — same convention as the bridge's
    # tool_r_xyz_rpy param. tcp_pose obs / command targets are both TCP.
    TCP_OFFSET_XYZ_RPY: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # robot /joint_states older than this -> unsafe to act on
    # How long __init__ waits for the first /joint_states before giving up.
    # DDS discovery plus the first message costs about a second; callers
    # reset() immediately, so without this the env fails on a healthy rig.
    ROBOT_STATE_WAIT_S: float = 15.0
    JOINT_STATE_STALE_S: float = 0.2

    # ---- gripper ---- #
    GRIPPER_SLEEP: float = 0.6   # debounce between binary open/close commands
    # Learned-gripper auxiliary reward.  This matches the upstream task
    # configs and is emitted on every transition by GripperPenaltyWrapper.
    GRASP_PENALTY: float = -0.02

    # Safety default for the skeleton phase: compute everything but do NOT
    # publish robot commands unless explicitly armed.
    DRY_RUN: bool = True
