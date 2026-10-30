"""Convert gello_recorder HDF5/MP4 takes into canonical learner demos.

The recorder predates the remote actor and stores native-rate signals rather
than actor ``data{meta, transition}`` items.  This module reconstructs the
same 10 Hz boundary offline:

* latest-at-or-before (as-of) synchronization, matching ROS subscriber state;
* the cube-in-cup crop -> 128x128 -> RGB image path;
* the exact canonical 19-D state layout;
* tool-frame normalized actions recovered from recorded joint commands; and
* the three-state GELLO gripper action plus redundant-command penalty.

Raw images exist only in the resulting canonical input artifact.  The learner
encodes that artifact once at startup and its long-lived demo/replay pools own
only frozen ResNet-10 features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from ur_env.learner.demo import load_demo_object
from ur_env.observation_preprocess import (
    CropBox,
    PreprocessRuleError,
    preprocess_frame,
)
from ur_env.observation_schema import STATE_DIM, validate_canonical_observation


CONVERSION_REVISION = "gello-recorder-to-canonical-demo-v1"
ACTION_SOURCE = "recorded-command-fk-delta-v1"
DEFAULT_SAMPLE_RATE_HZ = 10.0
DEFAULT_MAX_SIGNAL_AGE_S = 0.20
DEFAULT_MAX_CAMERA_AGE_S = 0.20
DEFAULT_GRASP_PENALTY = -0.02

_JOINT_COLUMNS = tuple(f"q{index}" for index in range(1, 7))
_VELOCITY_COLUMNS = tuple(f"qd{index}" for index in range(1, 7))
_COMMAND_COLUMNS = tuple(f"cmd{index}" for index in range(1, 7))
_TCP_COLUMNS = ("x", "y", "z", "qx", "qy", "qz", "qw")
_WRENCH_COLUMNS = ("fx", "fy", "fz", "tx", "ty", "tz")


class RecordedDemoConversionError(ValueError):
    """A recorder take cannot be converted without violating the contract."""


@dataclass(frozen=True)
class TakeConversionStats:
    source_take: str
    outcome: str
    episode_id: int
    transition_count: int
    first_time_s: float
    last_time_s: float
    sample_rate_hz: float
    saturated_action_count: int
    saturated_action_fraction: float
    max_raw_action_group_norm: float
    max_signal_age_s: float
    max_camera_age_s: float
    redundant_gripper_penalty_count: int

    def document(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConvertedRecordedDemos:
    transitions: tuple[dict[str, Any], ...]
    takes: tuple[TakeConversionStats, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "conversion_revision": CONVERSION_REVISION,
            "transition_count": len(self.transitions),
            "take_count": len(self.takes),
            "takes": [take.document() for take in self.takes],
        }


@dataclass(frozen=True)
class _TimedValues:
    times: np.ndarray
    values: np.ndarray


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _read_columns(group: Any, columns: Sequence[str], *, name: str) -> _TimedValues:
    missing = [column for column in ("t_rel_s", *columns) if column not in group]
    if missing:
        raise RecordedDemoConversionError(f"{name} is missing columns {missing}")
    times = np.asarray(group["t_rel_s"][:], dtype=np.float64)
    values = np.stack(
        [np.asarray(group[column][:], dtype=np.float64) for column in columns],
        axis=1,
    )
    if times.ndim != 1 or values.shape != (len(times), len(columns)):
        raise RecordedDemoConversionError(f"{name} has inconsistent column lengths")
    finite = np.isfinite(times) & np.all(np.isfinite(values), axis=1)
    times = times[finite]
    values = values[finite]
    if len(times) == 0:
        raise RecordedDemoConversionError(f"{name} has no complete finite rows")
    if np.any(np.diff(times) < 0.0):
        raise RecordedDemoConversionError(f"{name}.t_rel_s must be nondecreasing")
    return _TimedValues(times=times, values=values)


def _asof(
    source: _TimedValues,
    query_times: np.ndarray,
    *,
    name: str,
    max_age_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(source.times, query_times, side="right") - 1
    if np.any(indices < 0):
        first = float(query_times[np.flatnonzero(indices < 0)[0]])
        raise RecordedDemoConversionError(
            f"{name} has no sample at or before query time {first:.6f}s"
        )
    ages = query_times - source.times[indices]
    if np.any(ages < -1e-9):
        raise RecordedDemoConversionError(f"{name} as-of synchronization went forward")
    if np.any(ages > max_age_s + 1e-9):
        worst = float(np.max(ages))
        raise RecordedDemoConversionError(
            f"{name} is stale by {worst:.6f}s; maximum is {max_age_s:.6f}s"
        )
    return source.values[indices], ages


def _pose_matrix(pose7: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose7, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise RecordedDemoConversionError("TCP pose must be one finite xyz+quaternion")
    quaternion_norm = float(np.linalg.norm(pose[3:]))
    if quaternion_norm < 1e-6:
        raise RecordedDemoConversionError("TCP quaternion has near-zero norm")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def policy_action_from_command_poses(
    command_now: np.ndarray,
    command_next: np.ndarray,
    observation_pose: np.ndarray,
    *,
    position_scale: float,
    rotation_scale: float,
    so3_log: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Recover one tool-frame policy action and its unclipped six-vector."""

    position_scale = _positive_finite(position_scale, name="position_scale")
    rotation_scale = _positive_finite(rotation_scale, name="rotation_scale")
    current = np.asarray(command_now, dtype=np.float64)
    following = np.asarray(command_next, dtype=np.float64)
    observed = np.asarray(observation_pose, dtype=np.float64)
    for name, value in (
        ("command_now", current),
        ("command_next", following),
        ("observation_pose", observed),
    ):
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise RecordedDemoConversionError(f"{name} must be a finite 4x4 transform")

    rotation_base_to_tool = observed[:3, :3].T
    translation_base = following[:3, 3] - current[:3, 3]
    rotation_base = np.asarray(
        so3_log(following[:3, :3] @ current[:3, :3].T), dtype=np.float64
    )
    if rotation_base.shape != (3,) or not np.all(np.isfinite(rotation_base)):
        raise RecordedDemoConversionError("so3_log returned an invalid rotation vector")
    raw = np.concatenate(
        [
            rotation_base_to_tool @ translation_base / position_scale,
            rotation_base_to_tool @ rotation_base / rotation_scale,
        ]
    )
    # Match GelloIntervention's anti-windup rule: preserve direction and cap
    # translation/rotation vector norms independently.  Per-axis clipping
    # bends diagonal expert motion and therefore is not buffer-correct.
    bounded = raw.copy()
    for action_slice in (slice(0, 3), slice(3, 6)):
        norm = float(np.linalg.norm(bounded[action_slice]))
        if norm > 1.0:
            bounded[action_slice] /= norm
    clipped = np.clip(bounded, -1.0, 1.0).astype(np.float32)
    return clipped, raw


def _task_contract() -> tuple[np.ndarray, Mapping[str, CropBox]]:
    try:
        from ur_experiments.cube_in_cup import CubeInCupEnvConfig
    except Exception as exc:
        raise RecordedDemoConversionError(
            "cannot import cube_in_cup task preprocessing contract"
        ) from exc
    scale = np.asarray(CubeInCupEnvConfig.ACTION_SCALE, dtype=np.float64)
    crops = CubeInCupEnvConfig.IMAGE_CROP
    if scale.shape != (3,) or not np.all(np.isfinite(scale)):
        raise RecordedDemoConversionError("cube_in_cup ACTION_SCALE is invalid")
    if not isinstance(crops, Mapping) or set(crops) != {"cam1", "cam2"}:
        raise RecordedDemoConversionError("cube_in_cup IMAGE_CROP must define cam1/cam2")
    if not all(isinstance(crop, CropBox) for crop in crops.values()):
        raise RecordedDemoConversionError(
            "cube_in_cup IMAGE_CROP windows must be CropBox values"
        )
    return scale, crops


def _kinematics() -> tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    try:
        from ur_gello_bringup.ur_kin import fk, jacobian, so3_log
    except ImportError as exc:
        raise RecordedDemoConversionError(
            "ur_gello_bringup.ur_kin is required; source ros2_ur_ws/install/"
            "setup.bash or add ros2_ur_ws/src/ur_gello_bringup to PYTHONPATH"
        ) from exc
    return fk, jacobian, so3_log


def _read_video_frames(
    path: Path,
    frame_indices: np.ndarray,
    *,
    crop: CropBox,
) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise RecordedDemoConversionError("opencv-python is required") from exc

    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0):
        raise RecordedDemoConversionError(f"{path.name} frame indices are invalid")
    needed = set(int(value) for value in indices)
    maximum = max(needed)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RecordedDemoConversionError(f"cannot open video {path}")
    selected: dict[int, np.ndarray] = {}
    try:
        for index in range(maximum + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RecordedDemoConversionError(
                    f"{path.name} ended before frame {maximum} (failed at {index})"
                )
            if index not in needed:
                continue
            # The task's own recipe, not a copy of it: this is the same call
            # UR7eEnv.get_im makes, so a converted take and a live observation
            # cannot drift apart (ur_env/observation_preprocess).
            try:
                rgb = preprocess_frame(frame, crop=crop)
            except PreprocessRuleError as exc:
                raise RecordedDemoConversionError(
                    f"{path.name} preprocessing failed: {exc}"
                ) from exc
            if rgb.shape != (128, 128, 3):
                raise RecordedDemoConversionError(
                    f"{path.name} preprocessing returned {rgb.shape}"
                )
            selected[index] = rgb
    finally:
        capture.release()
    return [selected[int(index)] for index in indices]


def _validate_outcome(value: str) -> str:
    if value not in {"success", "truncated"}:
        raise ValueError("outcome must be 'success' or 'truncated'")
    return value


def _episode_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("episode_id must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("episode_id must be non-negative")
    return result


def _canonical_state(
    *,
    gripper_position: float,
    wrench: np.ndarray,
    absolute_pose: np.ndarray,
    reset_pose_inverse: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    jacobian: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    if not math.isfinite(gripper_position) or not 0.0 <= gripper_position <= 1.0:
        raise RecordedDemoConversionError(
            f"gripper position must be in [0,1], got {gripper_position}"
        )
    wrench = np.asarray(wrench, dtype=np.float64)
    if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
        raise RecordedDemoConversionError("wrench must be a finite six-vector")
    relative = reset_pose_inverse @ absolute_pose
    relative_pose = np.concatenate(
        [
            relative[:3, 3],
            Rotation.from_matrix(relative[:3, :3]).as_euler("xyz"),
        ]
    )
    twist_base = np.asarray(jacobian(joint_position), dtype=np.float64) @ np.asarray(
        joint_velocity, dtype=np.float64
    )
    if twist_base.shape != (6,) or not np.all(np.isfinite(twist_base)):
        raise RecordedDemoConversionError("UR Jacobian produced an invalid TCP twist")
    rotation_base_to_tool = absolute_pose[:3, :3].T
    twist_tool = np.concatenate(
        [
            rotation_base_to_tool @ twist_base[:3],
            rotation_base_to_tool @ twist_base[3:],
        ]
    )
    state = np.concatenate(
        [
            [gripper_position],
            wrench[:3],
            relative_pose,
            wrench[3:],
            twist_tool,
        ]
    ).astype(np.float32)
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise RecordedDemoConversionError("canonical state construction failed")
    return np.ascontiguousarray(state[None])


def convert_recorded_take(
    take_dir: os.PathLike[str] | str,
    *,
    outcome: str,
    episode_id: int,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_signal_age_s: float = DEFAULT_MAX_SIGNAL_AGE_S,
    max_camera_age_s: float = DEFAULT_MAX_CAMERA_AGE_S,
    grasp_penalty: float = DEFAULT_GRASP_PENALTY,
) -> tuple[tuple[dict[str, Any], ...], TakeConversionStats]:
    """Convert one recorder directory without guessing its terminal label."""

    outcome = _validate_outcome(outcome)
    episode_id = _episode_id(episode_id)
    rate = _positive_finite(sample_rate_hz, name="sample_rate_hz")
    signal_age_limit = _nonnegative_finite(
        max_signal_age_s, name="max_signal_age_s"
    )
    camera_age_limit = _nonnegative_finite(
        max_camera_age_s, name="max_camera_age_s"
    )
    penalty = float(grasp_penalty)
    if not math.isfinite(penalty) or penalty > 0.0:
        raise ValueError("grasp_penalty must be finite and non-positive")

    directory = Path(take_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    h5_path = directory / "vectors.h5"
    video_paths = {key: directory / f"{key}.mp4" for key in ("cam1", "cam2")}
    for path in (h5_path, *video_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    try:
        import h5py
    except ImportError as exc:
        raise RecordedDemoConversionError("h5py is required for conversion") from exc

    with h5py.File(h5_path, "r") as h5:
        required_groups = {
            "command",
            "ur_joint_states",
            "gripper",
            "wrench",
            "tcp_pose",
            "cam1_frames",
            "cam2_frames",
        }
        missing_groups = sorted(required_groups - set(h5))
        if missing_groups:
            raise RecordedDemoConversionError(
                f"{h5_path} is missing groups {missing_groups}"
            )
        command = _read_columns(h5["command"], _COMMAND_COLUMNS, name="command")
        joints = _read_columns(
            h5["ur_joint_states"],
            (*_JOINT_COLUMNS, *_VELOCITY_COLUMNS),
            name="ur_joint_states",
        )
        tcp = _read_columns(h5["tcp_pose"], _TCP_COLUMNS, name="tcp_pose")
        wrench = _read_columns(h5["wrench"], _WRENCH_COLUMNS, name="wrench")
        gripper_position = _read_columns(
            h5["gripper"], ("grip_pos",), name="gripper.grip_pos"
        )
        gello_gripper = _read_columns(
            h5["gripper"], ("gello_grip",), name="gripper.gello_grip"
        )
        cameras = {
            key: _read_columns(
                h5[f"{key}_frames"], ("frame_idx",), name=f"{key}_frames"
            )
            for key in ("cam1", "cam2")
        }

    streams = (
        command,
        joints,
        tcp,
        wrench,
        gripper_position,
        gello_gripper,
        cameras["cam1"],
        cameras["cam2"],
    )
    first_time = max(float(stream.times[0]) for stream in streams)
    available_last_time = min(float(stream.times[-1]) for stream in streams)
    period = 1.0 / rate
    transition_count = int(
        math.floor((available_last_time - first_time) / period + 1e-9)
    )
    if transition_count < 1:
        raise RecordedDemoConversionError(
            f"{directory.name} has less than one {period:.3f}s transition in its "
            "common valid interval"
        )
    sample_times = first_time + np.arange(transition_count + 1) * period

    command_q, command_ages = _asof(
        command, sample_times, name="command", max_age_s=signal_age_limit
    )
    joint_values, joint_ages = _asof(
        joints, sample_times, name="ur_joint_states", max_age_s=signal_age_limit
    )
    tcp_values, tcp_ages = _asof(
        tcp, sample_times, name="tcp_pose", max_age_s=signal_age_limit
    )
    wrench_values, wrench_ages = _asof(
        wrench, sample_times, name="wrench", max_age_s=signal_age_limit
    )
    gripper_values, gripper_ages = _asof(
        gripper_position,
        sample_times,
        name="gripper.grip_pos",
        max_age_s=signal_age_limit,
    )
    gello_gripper_values, gello_gripper_ages = _asof(
        gello_gripper,
        sample_times[:-1],
        name="gripper.gello_grip",
        max_age_s=signal_age_limit,
    )
    camera_indices: dict[str, np.ndarray] = {}
    camera_ages: list[np.ndarray] = []
    for key in ("cam1", "cam2"):
        values, ages = _asof(
            cameras[key], sample_times, name=f"{key}_frames", max_age_s=camera_age_limit
        )
        rounded = np.rint(values[:, 0]).astype(np.int64)
        if not np.allclose(values[:, 0], rounded, rtol=0.0, atol=1e-9):
            raise RecordedDemoConversionError(f"{key} frame indices are not integers")
        camera_indices[key] = rounded
        camera_ages.append(ages)

    action_scale, crop_functions = _task_contract()
    fk, jacobian, so3_log = _kinematics()
    absolute_poses = [_pose_matrix(value) for value in tcp_values]
    command_poses = [np.asarray(fk(value), dtype=np.float64) for value in command_q]
    reset_pose_inverse = np.linalg.inv(absolute_poses[0])

    images = {
        key: _read_video_frames(
            video_paths[key], camera_indices[key], crop=crop_functions[key]
        )
        for key in ("cam1", "cam2")
    }
    observations: list[dict[str, np.ndarray]] = []
    for index in range(transition_count + 1):
        state = _canonical_state(
            gripper_position=float(gripper_values[index, 0]),
            wrench=wrench_values[index],
            absolute_pose=absolute_poses[index],
            reset_pose_inverse=reset_pose_inverse,
            joint_position=joint_values[index, :6],
            joint_velocity=joint_values[index, 6:],
            jacobian=jacobian,
        )
        observation = {
            "state": state,
            "cam1": np.ascontiguousarray(images["cam1"][index][None]),
            "cam2": np.ascontiguousarray(images["cam2"][index][None]),
        }
        observations.append(validate_canonical_observation(observation, copy=False))

    transitions: list[dict[str, Any]] = []
    gripper_latch = 0.0
    saturated = 0
    max_raw_action_group_norm = 0.0
    penalty_count = 0
    for index in range(transition_count):
        arm_action, raw_action = policy_action_from_command_poses(
            command_poses[index],
            command_poses[index + 1],
            absolute_poses[index],
            position_scale=float(action_scale[0]),
            rotation_scale=float(action_scale[1]),
            so3_log=so3_log,
        )
        raw_group_norm = max(
            float(np.linalg.norm(raw_action[:3])),
            float(np.linalg.norm(raw_action[3:])),
        )
        max_raw_action_group_norm = max(
            max_raw_action_group_norm, raw_group_norm
        )
        if raw_group_norm > 1.0 + 1e-6:
            saturated += 1

        trigger = float(gello_gripper_values[index, 0])
        if not 0.0 <= trigger <= 1.0:
            raise RecordedDemoConversionError(
                f"GELLO gripper trigger must be in [0,1], got {trigger}"
            )
        if trigger >= 0.7:
            gripper_latch = -1.0
        elif trigger <= 0.3:
            gripper_latch = 1.0
        action = np.empty((7,), dtype=np.float32)
        action[:6] = arm_action
        action[6] = np.float32(gripper_latch)

        previous_position = float(gripper_values[index, 0])
        redundant_close = gripper_latch < -0.5 and previous_position >= 0.85
        redundant_open = gripper_latch > 0.5 and previous_position <= 0.15
        step_penalty = penalty if redundant_close or redundant_open else 0.0
        penalty_count += int(step_penalty != 0.0)

        terminal = index == transition_count - 1
        success = terminal and outcome == "success"
        truncated = terminal and outcome == "truncated"
        done = bool(success)
        item = {
            "meta": {
                "schema_version": CONVERSION_REVISION,
                "run_id": directory.name,
                "actor_id": "gello-recorder-offline-converter",
                "session_id": directory.name,
                "transition_id": f"{directory.name}:{index}",
                "env_step": index,
                "policy_version": 0,
                "intervened": 1,
                "source_take": directory.name,
                "recording_time_s": float(sample_times[index]),
                "action_source": ACTION_SOURCE,
                "sample_rate_hz": rate,
                "position_action_scale": float(action_scale[0]),
                "rotation_action_scale": float(action_scale[1]),
                "camera_preprocessing": "cube_in_cup_crop-resize128-rgb-v1",
            },
            "transition": {
                "observations": observations[index],
                "next_observations": observations[index + 1],
                "actions": action,
                "rewards": np.float32(1.0 if success else 0.0),
                "masks": np.float32(0.0 if done else 1.0),
                "dones": done,
                "truncated": bool(truncated),
                "success": bool(success),
                "grasp_penalty": np.float32(step_penalty),
                "episode_id": episode_id,
                "step_id": index,
                "observation_id": f"{directory.name}:o{index}",
                "next_observation_id": f"{directory.name}:o{index + 1}",
            },
        }
        # Exercise the production strict loader one transition at a time.  This
        # avoids materializing a second full copy of a potentially large demo.
        load_demo_object([item], source_path=str(directory))
        transitions.append(item)

    signal_age_arrays = (
        command_ages,
        joint_ages,
        tcp_ages,
        wrench_ages,
        gripper_ages,
        gello_gripper_ages,
    )
    stats = TakeConversionStats(
        source_take=directory.name,
        outcome=outcome,
        episode_id=episode_id,
        transition_count=transition_count,
        first_time_s=float(sample_times[0]),
        last_time_s=float(sample_times[-1]),
        sample_rate_hz=rate,
        saturated_action_count=saturated,
        saturated_action_fraction=saturated / transition_count,
        max_raw_action_group_norm=max_raw_action_group_norm,
        max_signal_age_s=max(float(np.max(value)) for value in signal_age_arrays),
        max_camera_age_s=max(float(np.max(value)) for value in camera_ages),
        redundant_gripper_penalty_count=penalty_count,
    )
    return tuple(transitions), stats


def convert_recorded_takes(
    take_dirs: Iterable[os.PathLike[str] | str],
    *,
    outcome: str,
    episode_id_start: int = 0,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_signal_age_s: float = DEFAULT_MAX_SIGNAL_AGE_S,
    max_camera_age_s: float = DEFAULT_MAX_CAMERA_AGE_S,
    grasp_penalty: float = DEFAULT_GRASP_PENALTY,
) -> ConvertedRecordedDemos:
    directories = tuple(take_dirs)
    if not directories:
        raise ValueError("at least one take directory is required")
    first_episode = _episode_id(episode_id_start)
    all_transitions: list[dict[str, Any]] = []
    all_stats: list[TakeConversionStats] = []
    for offset, directory in enumerate(directories):
        transitions, stats = convert_recorded_take(
            directory,
            outcome=outcome,
            episode_id=first_episode + offset,
            sample_rate_hz=sample_rate_hz,
            max_signal_age_s=max_signal_age_s,
            max_camera_age_s=max_camera_age_s,
            grasp_penalty=grasp_penalty,
        )
        all_transitions.extend(transitions)
        all_stats.append(stats)
    return ConvertedRecordedDemos(tuple(all_transitions), tuple(all_stats))


def write_recorded_demo_pickle(
    path: os.PathLike[str] | str,
    converted: ConvertedRecordedDemos,
) -> Path:
    """Write atomically without replacing an existing demo artifact."""

    if not isinstance(converted, ConvertedRecordedDemos):
        raise TypeError("converted must be ConvertedRecordedDemos")
    if not converted.transitions:
        raise ValueError("cannot write an empty demo artifact")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with open(temporary, "xb") as stream:
            pickle.dump(list(converted.transitions), stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails rather than overwriting.
        os.link(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


__all__ = [
    "ACTION_SOURCE",
    "CONVERSION_REVISION",
    "ConvertedRecordedDemos",
    "DEFAULT_GRASP_PENALTY",
    "DEFAULT_MAX_CAMERA_AGE_S",
    "DEFAULT_MAX_SIGNAL_AGE_S",
    "DEFAULT_SAMPLE_RATE_HZ",
    "RecordedDemoConversionError",
    "TakeConversionStats",
    "convert_recorded_take",
    "convert_recorded_takes",
    "policy_action_from_command_poses",
    "write_recorded_demo_pickle",
]
