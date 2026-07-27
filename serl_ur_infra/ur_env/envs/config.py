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
    ACTION_SCALE: np.ndarray = np.array([0.01, 0.05, 1.0])

    # ---- workspace safety box (TCP, UR base frame) ---- #
    ABS_POSE_LIMIT_LOW: np.ndarray = np.zeros((6,))
    ABS_POSE_LIMIT_HIGH: np.ndarray = np.zeros((6,))

    # ---- governor / IK safety (feeds PolicyDeltaController; the analog of ---- #
    # ---- Franka's COMPLIANCE_PARAM — software-synthesized softness).      ---- #
    # Caps sit ~20% above ACTION_SCALE * HZ (0.1 m/s, 0.5 rad/s) — pure safety
    # net; never the binding limit in normal operation (see INVARIANT above).
    GOVERNOR: Dict[str, float] = {
        "v_max": 0.12,        # m/s   task-space translational rate cap
        "w_max": 0.60,        # rad/s task-space rotational rate cap
        "dq_step_max": 0.05,  # rad   per-tick joint step acceptance gate
    }

    # ---- 250 Hz command upsampler ---- #
    # forward_position_controller does not interpolate (bridge docstring), so
    # the backend streams slew-limited steps toward the latest 10 Hz target.
    # Same role as the bridge's upsample+slew; EMA deliberately omitted — RL
    # actions are already governed/step-capped and tremor-free, and EMA lag
    # would blur action->effect credit assignment.
    UPSAMPLER: Dict[str, float] = {
        "hz": 250.0,
        # per-tick joint step: 0.002 rad @ 250 Hz = 0.5 rad/s, the same rate
        # ceiling as GOVERNOR.dq_step_max (0.05 rad @ 10 Hz).
        "max_step_rad": 0.002,
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
    JOINT_STATE_STALE_S: float = 0.2

    # ---- gripper ---- #
    GRIPPER_SLEEP: float = 0.6   # debounce between binary open/close commands
    # Learned-gripper auxiliary reward.  This matches the upstream task
    # configs and is emitted on every transition by GripperPenaltyWrapper.
    GRASP_PENALTY: float = -0.02

    # Safety default for the skeleton phase: compute everything but do NOT
    # publish robot commands unless explicitly armed.
    DRY_RUN: bool = True
