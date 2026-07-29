"""Recorder HDF5/MP4 -> canonical learner demo contract tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


_INFRA = Path(__file__).resolve().parents[1]
_REPO = _INFRA.parent
_UR_GELLO = _REPO / "ros2_ur_ws" / "src" / "ur_gello_bringup"
for path in (_INFRA, _UR_GELLO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ur_env.learner.demo import (  # noqa: E402
    _NumpyCompatibilityUnpickler,
    load_demo_pickle,
)
from ur_env.learner.recorded_demo import (  # noqa: E402
    convert_recorded_takes,
    policy_action_from_command_poses,
    write_recorded_demo_pickle,
)


def test_policy_action_is_reexpressed_in_observed_tool_frame():
    command_now = np.eye(4)
    command_next = np.eye(4)
    command_next[:3, 3] = [0.0, 0.00625, 0.0]
    command_next[:3, :3] = Rotation.from_rotvec(
        [0.0, 0.0, 0.03125]
    ).as_matrix()
    observation_pose = np.eye(4)
    observation_pose[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()

    action, raw = policy_action_from_command_poses(
        command_now,
        command_next,
        observation_pose,
        position_scale=0.0125,
        rotation_scale=0.0625,
        so3_log=lambda matrix: Rotation.from_matrix(matrix).as_rotvec(),
    )

    expected_translation = np.array([0.5, 0.0, 0.0])
    assert action.dtype == np.float32
    np.testing.assert_allclose(raw[:3], expected_translation, atol=1e-7)


def test_policy_action_known_translation_and_rotation():
    command_now = np.eye(4)
    command_next = np.eye(4)
    command_next[:3, 3] = [0.00625, 0.0, 0.0]
    command_next[:3, :3] = Rotation.from_rotvec(
        [0.0, 0.0, 0.03125]
    ).as_matrix()

    action, raw = policy_action_from_command_poses(
        command_now,
        command_next,
        np.eye(4),
        position_scale=0.0125,
        rotation_scale=0.0625,
        so3_log=lambda matrix: Rotation.from_matrix(matrix).as_rotvec(),
    )

    np.testing.assert_allclose(raw, [0.5, 0, 0, 0, 0, 0.5], atol=1e-7)
    np.testing.assert_allclose(action, raw, atol=1e-7)


def test_numpy2_private_core_name_maps_to_pinned_numpy1_name():
    unpickler = _NumpyCompatibilityUnpickler(BytesIO(b""))
    resolved = unpickler.find_class("numpy._core.numeric", "_frombuffer")

    assert resolved.__name__ == "_frombuffer"


def _write_group(h5, name: str, values: dict[str, np.ndarray]) -> None:
    group = h5.create_group(name)
    for key, value in values.items():
        group.create_dataset(key, data=np.asarray(value, dtype=np.float64))


def _write_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1280, 720)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer is unavailable")
    try:
        for color in colors:
            writer.write(np.full((720, 1280, 3), color, dtype=np.uint8))
    finally:
        writer.release()


def test_synthetic_take_converts_and_strict_loader_accepts(tmp_path):
    h5py = pytest.importorskip("h5py")
    from ur_gello_bringup.ur_kin import fk

    take = tmp_path / "take_01_20990101_000000"
    take.mkdir()
    times = np.array([0.0, 0.1, 0.2])
    q = np.array([3.0, -1.5, 1.7, -1.7, -1.5, -3.0])
    transform = fk(q)
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    pose = np.concatenate([transform[:3, 3], quaternion])
    with h5py.File(take / "vectors.h5", "w") as h5:
        _write_group(
            h5,
            "command",
            {"t_rel_s": times, **{f"cmd{i + 1}": [q[i]] * 3 for i in range(6)}},
        )
        _write_group(
            h5,
            "ur_joint_states",
            {
                "t_rel_s": times,
                **{f"q{i + 1}": [q[i]] * 3 for i in range(6)},
                **{f"qd{i + 1}": [0.0] * 3 for i in range(6)},
            },
        )
        _write_group(
            h5,
            "tcp_pose",
            {
                "t_rel_s": times,
                **{
                    key: [pose[index]] * 3
                    for index, key in enumerate(("x", "y", "z", "qx", "qy", "qz", "qw"))
                },
            },
        )
        _write_group(
            h5,
            "wrench",
            {"t_rel_s": times, **{key: [0.0] * 3 for key in ("fx", "fy", "fz", "tx", "ty", "tz")}},
        )
        _write_group(
            h5,
            "gripper",
            {
                "t_rel_s": times,
                "gello_grip": [0.0] * 3,
                "grip_cmd": [0.0] * 3,
                "grip_pos": [0.0] * 3,
            },
        )
        for camera in ("cam1", "cam2"):
            _write_group(
                h5,
                f"{camera}_frames",
                {"t_rel_s": times, "frame_idx": [0, 1, 2]},
            )
    _write_video(take / "cam1.mp4", [(10, 20, 30)] * 3)
    _write_video(take / "cam2.mp4", [(40, 50, 60)] * 3)

    converted = convert_recorded_takes([take], outcome="success")

    assert len(converted.transitions) == 2
    first = converted.transitions[0]["transition"]
    final = converted.transitions[-1]["transition"]
    assert first["observations"]["state"].shape == (1, 19)
    assert first["observations"]["cam1"].shape == (1, 128, 128, 3)
    np.testing.assert_allclose(first["actions"][:6], 0.0, atol=1e-6)
    assert first["actions"][6] == 1.0
    assert first["grasp_penalty"] == np.float32(-0.02)
    assert final["rewards"] == np.float32(1.0)
    assert final["masks"] == np.float32(0.0)
    assert final["dones"] is True
    # Video is BGR, canonical observation is RGB.  Allow codec quantization.
    np.testing.assert_allclose(
        first["observations"]["cam1"][0, 64, 64], [30, 20, 10], atol=8
    )

    output = write_recorded_demo_pickle(tmp_path / "demo.pkl", converted)
    loaded = load_demo_pickle(output)
    assert len(loaded) == 2
    assert loaded.sidecars[-1].metadata["success"] is True
    with pytest.raises(FileExistsError):
        write_recorded_demo_pickle(output, converted)
