"""Local-loop tests for server-authoritative reward and termination."""

from __future__ import annotations

import copy
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
# ur_gello_bringup (ur_kin is numpy-only) — UR7eEnv needs it outside fake mode,
# and the env-level sidecar test below builds a real one on a fake backend.
sys.path.insert(
    0, os.path.join(_HERE, "..", "..", "ros2_ur_ws", "src", "ur_gello_bringup")
)

from ur_env.actor_network import (  # noqa: E402
    PROTOCOL_VERSION,
    ActorSessionService,
    BeginEpisodeCommand,
    ObservationPacket,
    StepCommand,
    TransitionOutcome,
)
from ur_env.actor_smoke import synthetic_observation  # noqa: E402
from ur_env.classifier_sidecar import (  # noqa: E402
    CLASSIFIER_SIDECAR_KEY,
    SIDECAR_TENSOR_KEYS,
    SidecarScheduler,
)
from ur_env.observation_schema import state_slice  # noqa: E402
from ur_env.remote_actor import (  # noqa: E402
    resolve_camera_frame_source,
    run_remote_actor,
)

#: The three keys the canonical policy/replay observation is allowed to have.
#: Anything else in a locally dumped observation makes the pickle unloadable by
#: ur_env/learner/demo.py, which validates strictly.
CANONICAL_OBSERVATION_KEYS = {"state", "cam1", "cam2"}


class _ActionSpace:
    shape = (7,)


class _ServerSuccessThenLocalDoneEnv:
    action_space = _ActionSpace()

    def __init__(self):
        self.reset_count = 0
        self.step_count = 0

    def reset(self):
        self.reset_count += 1
        return synthetic_observation(self.step_count), {
            "timestamp_ns": np.int64(1_000 + self.step_count)
        }

    def step(self, action):
        del action
        self.step_count += 1
        local_done = self.step_count == 2
        return (
            synthetic_observation(self.step_count),
            float(local_done),
            local_done,
            False,
            {
                "timestamp_ns": np.int64(1_000 + self.step_count),
                "intervened": 0,
            },
        )


class _InProcessNetwork:
    def __init__(self, service):
        self._service = service
        self._run_id = ""
        self._session_id = ""
        self._request_id = 1
        self._clock = 10_000

    def begin_episode(
        self,
        observation,
        *,
        run_id,
        session_id,
        episode_id,
        observation_id,
        timestamp_ns,
        deterministic=False,
    ):
        self._run_id = run_id
        self._session_id = session_id
        self._request_id = 2
        self._clock += 1
        return self._service.begin_episode(
            BeginEpisodeCommand(
                PROTOCOL_VERSION,
                "actor",
                run_id,
                session_id,
                episode_id,
                1,
                self._clock,
                ObservationPacket(observation_id, timestamp_ns, observation),
                deterministic,
            )
        )

    def step(
        self,
        next_observation,
        *,
        next_observation_id,
        next_timestamp_ns,
        data,
        request_action,
        deterministic=False,
    ):
        self._clock += 1
        result = self._service.step(
            StepCommand(
                PROTOCOL_VERSION,
                "actor",
                self._run_id,
                self._session_id,
                self._request_id,
                self._clock,
                data,
                ObservationPacket(
                    next_observation_id,
                    next_timestamp_ns,
                    next_observation,
                ),
                request_action,
                deterministic,
            )
        )
        self._request_id += 1
        return result


def test_actor_resets_on_server_classifier_success_and_keeps_final_values(
    tmp_path,
):
    accepted = []

    def finalize(data):
        transition = data["transition"]
        success = int(data["meta"]["env_step"]) == 0
        if success:
            transition.update(
                rewards=1.0,
                masks=0.0,
                dones=True,
                truncated=False,
            )
        return data, TransitionOutcome(
            transition_id=data["meta"]["transition_id"],
            reward=float(transition["rewards"]),
            mask=float(transition["masks"]),
            done=bool(transition["dones"]),
            truncated=bool(transition["truncated"]),
            success=success,
            classifier_evaluated=True,
            classifier_probability=0.9 if success else 0.1,
            classifier_threshold=0.85,
            reward_model_id="scripted-reward",
        )

    service = ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0),
        finalize_transition=finalize,
        accept_data=lambda data, intervened: accepted.append(data),
    )
    sessions = iter(("session-0", "session-1"))
    env = _ServerSuccessThenLocalDoneEnv()

    summary = run_remote_actor(
        _InProcessNetwork(service),
        env,
        config=SimpleNamespace(max_steps=2, random_steps=0, buffer_period=1),
        actor_id="actor",
        checkpoint_path=str(tmp_path),
        run_id="run",
        session_id_factory=lambda: next(sessions),
    )

    assert summary.episodes_started == 2
    assert env.reset_count == 2
    assert service.inference_count == 2
    assert len(accepted) == 2
    assert accepted[0]["transition"]["rewards"] == 1.0
    assert accepted[0]["transition"]["masks"] == 0.0
    assert accepted[0]["transition"]["dones"] is True
    assert accepted[0]["transition"]["truncated"] is False
    with open(
        tmp_path / "actor_data" / "run" / "replay" / "data_0.pkl", "rb"
    ) as stream:
        local_backup = pickle.load(stream)
    assert len(local_backup) == 1
    assert set(local_backup[0]["transition"]["observations"]) == {
        "state",
        "cam1",
        "cam2",
    }
    assert set(local_backup[0]["transition"]["next_observations"]) == {
        "state",
        "cam1",
        "cam2",
    }


# --------------------------------------------------------------------------- #
# Reward-classifier sidecar                                                     #
#                                                                               #
# The classifier was trained on uncropped frames but the policy observation is  #
# cropped (ur_experiments/cube_in_cup.py IMAGE_CROP), which cost recall@0.85    #
# 100% -> 33.3%.  The actor therefore ships the classifier its own UNCROPPED    #
# frame as an extra observation key on SOME steps.  What these tests protect:   #
#                                                                               #
#   * the extra key reaches the server and nothing else,                        #
#   * it never reaches the local backup pickle (demo.py rejects extra keys),    #
#   * BeginEpisode never carries it (O0 is no transition's next_observations),  #
#   * the policy's own observation bytes are untouched.                         #
# --------------------------------------------------------------------------- #


#: Stand-in for a decoded camera frame: full 720p, BGR, uint8 -- the shape
#: ``build_sidecar`` resizes down to 128x128 before anything goes on the wire.
def _full_frame(fill: int) -> np.ndarray:
    return np.full((720, 1280, 3), fill, dtype=np.uint8)


class _ScriptedSidecarScheduler:
    """Attaches on a fixed set of ``should_attach`` call ordinals."""

    def __init__(self, attach_on=()):
        self._attach_on = set(attach_on)
        self.query_count = 0
        self.reset_count = 0
        self.observed_states = []
        self.observed_terminals = []
        self.outcomes = []

    def should_attach(self, state, provisional_terminal):
        index = self.query_count
        self.query_count += 1
        self.observed_states.append(state)
        self.observed_terminals.append(bool(provisional_terminal))
        return index in self._attach_on

    def note_outcome(self, outcome):
        self.outcomes.append(outcome)

    def reset(self):
        self.reset_count += 1


class _CameraEnv:
    """Deterministic env that also exposes uncropped full-resolution frames."""

    action_space = _ActionSpace()

    def __init__(self, *, episode_length=None):
        self.reset_count = 0
        self.step_count = 0
        self._steps_this_episode = 0
        self._episode_length = episode_length
        self.frame_calls = 0

    def reset(self):
        self.reset_count += 1
        self._steps_this_episode = 0
        return synthetic_observation(self.step_count), {
            "timestamp_ns": np.int64(1_000 + self.step_count)
        }

    def step(self, action):
        del action
        self.step_count += 1
        self._steps_this_episode += 1
        done = (
            self._episode_length is not None
            and self._steps_this_episode >= self._episode_length
        )
        return (
            synthetic_observation(self.step_count),
            0.0,
            done,
            False,
            {
                "timestamp_ns": np.int64(1_000 + self.step_count),
                "intervened": 0,
            },
        )

    def last_camera_frames(self):
        self.frame_calls += 1
        return {"cam1": _full_frame(11), "cam2": _full_frame(200)}


class _RecordingNetwork(_InProcessNetwork):
    """``_InProcessNetwork`` that keeps what the actor handed the transport."""

    def __init__(self, service):
        super().__init__(service)
        self.begin_observations = []
        self.step_observations = []

    def begin_episode(self, observation, **kwargs):
        self.begin_observations.append(copy.deepcopy(dict(observation)))
        return super().begin_episode(observation, **kwargs)

    def step(self, next_observation, **kwargs):
        self.step_observations.append(copy.deepcopy(dict(next_observation)))
        return super().step(next_observation, **kwargs)


def _passthrough_service():
    """A server that accepts everything and terminates nothing on its own."""

    return ActorSessionService(
        lambda observation, deterministic: (np.zeros(7, np.float32), 0)
    )


def test_sidecar_is_attached_only_on_scheduled_steps(tmp_path):
    scheduler = _ScriptedSidecarScheduler(attach_on=(1, 3))
    network = _RecordingNetwork(_passthrough_service())
    env = _CameraEnv()

    summary = run_remote_actor(
        network,
        env,
        config=SimpleNamespace(max_steps=4, random_steps=0, buffer_period=4),
        actor_id="actor",
        checkpoint_path=str(tmp_path),
        run_id="run",
        session_id_factory=lambda: "session-0",
        sidecar_scheduler=scheduler,
    )

    assert scheduler.query_count == 4
    assert [set(obs) for obs in network.step_observations] == [
        CANONICAL_OBSERVATION_KEYS,
        CANONICAL_OBSERVATION_KEYS | {CLASSIFIER_SIDECAR_KEY},
        CANONICAL_OBSERVATION_KEYS,
        CANONICAL_OBSERVATION_KEYS | {CLASSIFIER_SIDECAR_KEY},
    ]
    assert summary.sidecar_attached_steps == 2
    assert summary.sidecar_build_failures == 0
    # Built from the env's raw JPEGs, once per attached step -- never eagerly.
    assert env.frame_calls == 2
    sidecar = network.step_observations[1][CLASSIFIER_SIDECAR_KEY]
    assert set(sidecar) == set(SIDECAR_TENSOR_KEYS)
    for tensor in sidecar.values():
        assert tensor.dtype == np.uint8
        assert tensor.ndim == 1
    # Latency is bucketed by attachment so the cost is measurable in the field.
    assert summary.sidecar_round_trip_ms_max >= 0.0
    assert summary.plain_round_trip_ms_max >= 0.0


def test_begin_episode_never_carries_the_sidecar(tmp_path):
    # Attach on every query: even then BeginEpisode must stay canonical, since
    # O0 is not any transition's next_observations and its verdict could not be
    # attributed to a reward.
    scheduler = _ScriptedSidecarScheduler(attach_on=range(8))
    network = _RecordingNetwork(_passthrough_service())
    env = _CameraEnv(episode_length=2)

    run_remote_actor(
        network,
        env,
        config=SimpleNamespace(max_steps=4, random_steps=0, buffer_period=0),
        actor_id="actor",
        checkpoint_path=None,
        run_id="run",
        session_id_factory=iter(("session-0", "session-1")).__next__,
        sidecar_scheduler=scheduler,
    )

    assert len(network.begin_observations) == 2
    for observation in network.begin_observations:
        assert set(observation) == CANONICAL_OBSERVATION_KEYS
    assert all(
        CLASSIFIER_SIDECAR_KEY in observation for observation in network.step_observations
    )


def test_local_backup_never_contains_the_sidecar(tmp_path):
    scheduler = _ScriptedSidecarScheduler(attach_on=range(4))
    network = _RecordingNetwork(_passthrough_service())
    env = _CameraEnv()

    run_remote_actor(
        network,
        env,
        config=SimpleNamespace(max_steps=4, random_steps=0, buffer_period=4),
        actor_id="actor",
        checkpoint_path=str(tmp_path),
        run_id="run",
        session_id_factory=lambda: "session-0",
        sidecar_scheduler=scheduler,
    )

    with open(
        tmp_path / "actor_data" / "run" / "replay" / "data_3.pkl", "rb"
    ) as stream:
        local_backup = pickle.load(stream)
    assert len(local_backup) == 4
    for item in local_backup:
        transition = item["transition"]
        assert set(transition["observations"]) == CANONICAL_OBSERVATION_KEYS
        assert set(transition["next_observations"]) == CANONICAL_OBSERVATION_KEYS


def test_policy_observation_is_byte_identical_with_and_without_sidecar(tmp_path):
    """Attachment must be purely additive on the wire, and invisible locally."""

    def _run(scheduler, directory):
        network = _RecordingNetwork(_passthrough_service())
        run_remote_actor(
            network,
            _CameraEnv(),
            config=SimpleNamespace(
                max_steps=3, random_steps=0, buffer_period=3
            ),
            actor_id="actor",
            checkpoint_path=str(directory),
            run_id="run",
            session_id_factory=lambda: "session-0",
            sidecar_scheduler=scheduler,
        )
        with open(
            directory / "actor_data" / "run" / "replay" / "data_2.pkl", "rb"
        ) as stream:
            return network, pickle.load(stream)

    plain_dir = tmp_path / "plain"
    attached_dir = tmp_path / "attached"
    plain_dir.mkdir()
    attached_dir.mkdir()

    plain_network, plain_backup = _run(None, plain_dir)
    attached_network, attached_backup = _run(
        _ScriptedSidecarScheduler(attach_on=range(3)), attached_dir
    )

    for plain_item, attached_item in zip(plain_backup, attached_backup):
        for field in ("observations", "next_observations"):
            plain_obs = plain_item["transition"][field]
            attached_obs = attached_item["transition"][field]
            assert set(plain_obs) == set(attached_obs)
            for key, value in plain_obs.items():
                assert value.dtype == attached_obs[key].dtype
                assert value.tobytes() == attached_obs[key].tobytes()

    # ... and the canonical entries that DID go over the wire are the same
    # arrays too: the sidecar rides beside them, it does not rewrite them.
    for plain_obs, attached_obs in zip(
        plain_network.step_observations, attached_network.step_observations
    ):
        assert set(attached_obs) - set(plain_obs) == {CLASSIFIER_SIDECAR_KEY}
        for key in CANONICAL_OBSERVATION_KEYS:
            assert plain_obs[key].tobytes() == attached_obs[key].tobytes()


def test_scheduler_is_reset_per_episode_and_sees_every_outcome(tmp_path):
    scheduler = _ScriptedSidecarScheduler(attach_on=(0,))
    network = _RecordingNetwork(_passthrough_service())
    env = _CameraEnv(episode_length=2)

    summary = run_remote_actor(
        network,
        env,
        config=SimpleNamespace(max_steps=4, random_steps=0, buffer_period=0),
        actor_id="actor",
        checkpoint_path=None,
        run_id="run",
        session_id_factory=iter(("session-0", "session-1")).__next__,
        sidecar_scheduler=scheduler,
    )

    # One reset before the first episode plus one per episode boundary; the
    # trailing terminal ends the run instead of starting a third episode.
    assert summary.episodes_started == 2
    assert env.reset_count == 2
    assert scheduler.reset_count == 2
    # Every step's outcome is reported back, not just the attached ones.
    assert len(scheduler.outcomes) == 4
    assert all(
        outcome.classifier_evaluated is False for outcome in scheduler.outcomes
    )
    # The scheduler is told when the step is provisionally terminal, which is
    # exactly when a success verdict matters most.
    assert scheduler.observed_terminals == [False, True, False, True]
    # ... and it is handed the canonical flat state so it can read TCP speed.
    for state in scheduler.observed_states:
        assert state.shape == (1, 19)
        assert state.dtype == np.float32


def test_sidecar_build_failure_does_not_abort_the_run(tmp_path):
    """A bad frame costs that step its reward, never the episode."""

    class _NoCameraEnv(_CameraEnv):
        def last_camera_frames(self):
            self.frame_calls += 1
            raise RuntimeError("camera frame vanished")

    scheduler = _ScriptedSidecarScheduler(attach_on=range(3))
    network = _RecordingNetwork(_passthrough_service())

    summary = run_remote_actor(
        network,
        _NoCameraEnv(),
        config=SimpleNamespace(max_steps=3, random_steps=0, buffer_period=0),
        actor_id="actor",
        checkpoint_path=None,
        run_id="run",
        session_id_factory=lambda: "session-0",
        sidecar_scheduler=scheduler,
    )

    assert summary.sidecar_attached_steps == 0
    assert summary.sidecar_build_failures == 3
    for observation in network.step_observations:
        assert set(observation) == CANONICAL_OBSERVATION_KEYS


def test_real_scheduler_composes_with_the_actor_loop(tmp_path):
    """Integration: the shipped SidecarScheduler drives the shipped loop.

    The tests above script the scheduler so they test only this file's wiring.
    This one uses the real one, because the two halves agreeing on what
    ``should_attach`` is handed -- the canonical flat state, from which it reads
    TCP speed -- is not something either side can verify alone.
    """

    class _StationaryCameraEnv(_CameraEnv):
        """Same as ``_CameraEnv`` but with the arm parked (tcp_vel == 0)."""

        @staticmethod
        def _park(observation):
            state = observation["state"].copy()
            state[0, state_slice("tcp_vel")] = 0.0
            observation["state"] = state
            return observation

        def reset(self):
            observation, info = super().reset()
            return self._park(observation), info

        def step(self, action):
            observation, reward, done, truncated, info = super().step(action)
            return self._park(observation), reward, done, truncated, info

    network = _RecordingNetwork(_passthrough_service())
    scheduler = SidecarScheduler(
        interval_steps=3,
        stationary_speed_max=0.05,
        escalate_probability=1.0,  # never escalate: p is only ever 0.0 here
    )

    summary = run_remote_actor(
        network,
        _StationaryCameraEnv(),
        config=SimpleNamespace(max_steps=7, random_steps=0, buffer_period=0),
        actor_id="actor",
        checkpoint_path=None,
        run_id="run",
        session_id_factory=lambda: "session-0",
        sidecar_scheduler=scheduler,
    )

    attached = [
        CLASSIFIER_SIDECAR_KEY in observation
        for observation in network.step_observations
    ]
    # reset() arms the counter, so the first stationary step attaches and the
    # cadence runs from there.
    assert attached == [True, False, False, True, False, False, True]
    assert summary.sidecar_attached_steps == 3
    assert summary.sidecar_build_failures == 0


def test_camera_frame_source_is_found_through_the_wrapper_chain():
    """The actor is handed a wrapper stack, not the bare UR7eEnv."""

    base = _CameraEnv()

    class _Wrapper:
        def __init__(self, env):
            self.env = env

        @property
        def unwrapped(self):
            return getattr(self.env, "unwrapped", self.env)

    accessor = resolve_camera_frame_source(_Wrapper(_Wrapper(base)))
    assert accessor is not None
    assert set(accessor()) == {"cam1", "cam2"}
    # An env without cameras (fake env, test stubs) resolves to nothing rather
    # than raising: the sidecar is an enrichment, not a precondition.
    assert resolve_camera_frame_source(SimpleNamespace()) is None


def test_env_stashes_the_uncropped_frame_while_the_policy_keeps_its_crop():
    """The whole point of the sidecar, asserted at the source.

    ``UR7eEnv.get_im`` produces the policy's CROPPED 128x128 observation and,
    from the same decode, stashes the UNCROPPED full frame.  If these two ever
    became the same image the classifier would be back out of distribution
    (recall@0.85 100% -> 33.3%) with nothing to indicate it.
    """

    import cv2

    from ur_env.envs.ur7e_env import UR7eEnv
    from ur_experiments.cube_in_cup import CubeInCupEnvConfig

    rng = np.random.default_rng(7)
    # Distinct noise per camera, so a crop is detectable by content and not
    # only by shape.
    published = {
        camera: cv2.imencode(
            ".jpg", rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        )[1].tobytes()
        for camera in ("cam1", "cam2")
    }

    class _Backend:
        dry_run = False

        def get_joint_state(self):
            return np.zeros(6), np.zeros(6), 0.0

        def get_gripper_percent(self):
            return 0.0, 0.0

        def get_wrench(self):
            return None, float("inf")

        def get_tcp_pose(self):
            return None, float("inf")

        def get_image(self, camera):
            return published[camera], 0.0

        def close(self):
            pass

    config = CubeInCupEnvConfig()
    config.DISPLAY_IMAGE = False
    config.TCP_POSE_SOURCE = "fk"  # no tcp_pose_broadcaster on a fake backend
    env = UR7eEnv(fake_env=False, config=config, backend=_Backend())

    images = env.get_im()
    frames = env.last_camera_frames()

    for camera in config.CAMERAS:
        full = cv2.imdecode(
            np.frombuffer(published[camera], np.uint8), cv2.IMREAD_COLOR
        )
        # Stashed: the FULL frame, uncropped and unresized.
        assert frames[camera].shape == (720, 1280, 3)
        assert frames[camera].tobytes() == full.tobytes()
        # Policy: still cropped and resized, byte-identical to the pipeline
        # that existed before the sidecar.
        cropped = config.IMAGE_CROP[camera](full)
        expected = cv2.resize(
            cropped, env.observation_space["images"][camera].shape[:2][::-1]
        )[..., ::-1]
        assert images[camera].tobytes() == expected.tobytes()
        # ... and the two really are different pictures.
        assert cropped.shape != frames[camera].shape

    # get_im() crops from the very buffer it stashed, so the crop path must not
    # write through it.  It slices (a view) and cv2.resize allocates, so it
    # does not -- re-running get_im proves the stash is still pristine.
    env.get_im()
    for camera in config.CAMERAS:
        full = cv2.imdecode(
            np.frombuffer(published[camera], np.uint8), cv2.IMREAD_COLOR
        )
        assert env.last_camera_frames()[camera].tobytes() == full.tobytes()

    # The accessor hands out a fresh dict: re-keying it must not reach the env.
    handed_out = env.last_camera_frames()
    handed_out.pop("cam1")
    assert set(env.last_camera_frames()) == set(config.CAMERAS)
