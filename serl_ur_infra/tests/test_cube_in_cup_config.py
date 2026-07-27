"""The cube_in_cup task registry: guards, frame coupling, and chain assembly."""

import numpy as np
import pytest

from ur_env.observation_schema import (
    assert_actor_environment_state_layout,
    state_slice,
    validate_canonical_observation,
)
from ur_experiments.cube_in_cup import CubeInCupConfig, CubeInCupEnvConfig
from ur_experiments.mappings import CONFIG_MAPPING

pytest.importorskip(
    "serl_launcher.wrappers.serl_obs_wrappers",
    reason="third_party/hil-serl submodule is not checked out",
)

_CROP = {
    "cam1": lambda img: img[40:720, 250:930],
    "cam2": lambda img: img[0:720, 280:1000],
}


class _Commissioned(CubeInCupEnvConfig):
    IMAGE_CROP = _CROP
    DISPLAY_IMAGE = False


def _config():
    config = CubeInCupConfig.__new__(CubeInCupConfig)
    config.robot_config = _Commissioned()
    return config


def test_registry_exposes_the_task():
    assert CONFIG_MAPPING["cube_in_cup"] is CubeInCupConfig


def test_uncommissioned_config_refuses_to_instantiate():
    """Unmeasured values must fail loudly, not default to zeros.

    A zero ABS_POSE_LIMIT reads as a zero-volume workspace and a zero
    RESET_JOINTS parks the arm horizontally across the table.  Every field is
    measured today, so the guard is exercised by blanking one back out.
    """

    class Uncommissioned(CubeInCupEnvConfig):
        IMAGE_CROP = None

    with pytest.raises(ValueError, match="IMAGE_CROP"):
        Uncommissioned()

    class NoBox(CubeInCupEnvConfig):
        ABS_POSE_LIMIT_LOW = None

    with pytest.raises(ValueError, match="ABS_POSE_LIMIT_LOW"):
        NoBox()


def test_commissioned_config_instantiates():
    """The shipped values are complete: no placeholder left behind."""

    CubeInCupEnvConfig()


def test_workspace_box_is_non_degenerate():
    config = _Commissioned()
    low = np.asarray(config.ABS_POSE_LIMIT_LOW, dtype=float)
    high = np.asarray(config.ABS_POSE_LIMIT_HIGH, dtype=float)
    assert low.shape == high.shape == (6,)
    assert np.all(high > low)


def test_inverted_workspace_box_is_rejected():
    class Inverted(_Commissioned):
        ABS_POSE_LIMIT_LOW = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ABS_POSE_LIMIT_HIGH = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="must exceed"):
        Inverted()


def test_reset_pose_lies_inside_the_workspace_box():
    """Otherwise every episode starts by violating its own safety box."""

    ur_kin = pytest.importorskip(
        "ur_gello_bringup.ur_kin",
        reason="needs the ros2_ur_ws overlay on sys.path",
    )
    fk = ur_kin.fk

    config = _Commissioned()
    position = fk(np.asarray(config.RESET_JOINTS, dtype=float))[:3, 3]
    low = np.asarray(config.ABS_POSE_LIMIT_LOW, dtype=float)[:3]
    high = np.asarray(config.ABS_POSE_LIMIT_HIGH, dtype=float)[:3]
    assert np.all(position >= low), (position, low)
    assert np.all(position <= high), (position, high)


def test_pose_reference_point_stays_coupled():
    """The box was measured in the flange frame; three settings must agree.

    Adding the real 0.174 m tool offset, or switching the pose source to fk,
    moves the observed pose 17.4 cm without any error -- putting the z floor
    below the table.  See the frame note in cube_in_cup.py.
    """

    config = _Commissioned()
    assert config.TCP_POSE_SOURCE == "driver"
    assert list(config.TCP_OFFSET_XYZ_RPY) == [0.0] * 6
    assert float(config.ABS_POSE_LIMIT_LOW[2]) == pytest.approx(0.1785)


def test_reset_pose_is_the_cube_dataset_not_the_banana_one():
    """Regression: the ACT/Diffusion/FM configs carry the banana start pose."""

    banana_shoulder = -1.817
    shoulder = float(_Commissioned().RESET_JOINTS[1])
    assert abs(shoulder - banana_shoulder) > 0.2
    assert shoulder == pytest.approx(-1.5276, abs=1e-4)


def test_get_environment_builds_a_canonical_chain():
    env = _config().get_environment(fake_env=True)
    try:
        assert_actor_environment_state_layout(env)

        obs, _ = env.reset()
        validate_canonical_observation({k: np.asarray(v) for k, v in obs.items()})
        # RelativeFrame: the reset pose is the origin of the relative frame.
        pose = np.asarray(obs["state"])[0][state_slice("tcp_pose")]
        np.testing.assert_allclose(pose, np.zeros(6), atol=1e-9)

        stepped, *_ = env.step(np.zeros(7, dtype=np.float32))
        validate_canonical_observation(
            {k: np.asarray(v) for k, v in stepped.items()}
        )
    finally:
        env.close()


def test_local_reward_classifier_is_refused():
    """Reward is decided by the receive server, not a robot-side wrapper."""

    with pytest.raises(NotImplementedError, match="server-authoritative"):
        _config().get_environment(fake_env=True, classifier=True)


def test_actor_protocol_forbids_random_warmup_steps():
    assert CubeInCupConfig.random_steps == 0
    assert CubeInCupConfig.max_steps > 0
    assert CubeInCupConfig.buffer_period >= 0
