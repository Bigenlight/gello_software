#!/usr/bin/env python3
"""Run or safely probe the robot-laptop actor against a remote server.

Without ``--arm`` this entrypoint is deliberately *not* a dry execution loop:
it reads one observation, verifies server identity/schema, requests one policy
action with ``BeginEpisode``, validates that action, and exits without an
environment step or transition ``Step`` RPC.  Only ``--arm`` enters the
production transition loop.
"""

from __future__ import annotations

import argparse
import importlib
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "hil-serl" / "examples"))
sys.path.insert(
    0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher")
)

from ur_env.actor_network import create_actor_network  # noqa: E402
from ur_env.classifier_sidecar import SidecarScheduler  # noqa: E402
from ur_env.envs.wrappers import (  # noqa: E402
    wrap_gripper_penalty_from_task_config,
)
from ur_env.observation_schema import (  # noqa: E402
    CANONICAL_OBSERVATION_SCHEMA_HASH,
    assert_actor_environment_state_layout,
)
from ur_env.operator_session import RosOperatorSession  # noqa: E402
from ur_env.remote_actor import (  # noqa: E402
    EnvTimestampAdapter,
    run_remote_actor,
    run_remote_actor_probe,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--ur-config-module")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--fake-env",
        action="store_true",
        help=(
            "Build a synthetic local environment for the no-submit probe. "
            "It is not replay/learner acceptance; use "
            "scripts/run_fake_e2e_actor.py for that. Cannot be combined with "
            "--arm."
        ),
    )
    parser.add_argument("--actor-id", default=socket.gethostname())
    parser.add_argument("--network-type", choices=("grpc", "agentlace"))
    parser.add_argument("--server-host")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--max-response-age-s", type=float)
    parser.add_argument("--observation-schema-hash")
    parser.add_argument("--expected-model-id")
    parser.add_argument("--expected-reward-authority")
    parser.add_argument("--expected-reward-model-id")
    parser.add_argument(
        "--deadman",
        choices=("topic", "spacebar"),
        default="topic",
        help=(
            "Intervention deadman source. 'topic' (default) follows "
            "/hil/deadman from the HIL GUI: 20 Hz heartbeat with a staleness "
            "watchdog; after the first heartbeat, staleness aborts the actor "
            "before any policy action can reach the robot (there is no policy "
            "fallback). 'spacebar' is a GLOBAL pynput listener with no "
            "watchdog -- SPACE in any window engages and it can stick ON. "
            "Only 'topic' is safe on the real rig."
        ),
    )
    parser.add_argument(
        "--arm",
        action="store_true",
        help=(
            "Enter the production transition loop and actually publish to the "
            "robot (clears the task config's DRY_RUN). Default is off: one "
            "sensor/GetServerInfo/BeginEpisode policy probe runs, no action is "
            "executed, no Step RPC is sent, and no replay item is inserted. "
            "WARNING: with --arm the UR7e moves physically."
        ),
    )
    # --- reward classifier sidecar ------------------------------------------
    # Defaults come from the task config's CLASSIFIER_SIDECAR block; these
    # flags exist so an operator can retune the cadence on the rig without
    # editing (and accidentally committing) the task config.
    parser.add_argument(
        "--no-classifier-sidecar",
        action="store_true",
        help=(
            "Do not attach uncropped camera frames for the reward classifier. "
            "The server then has nothing to score, so every transition comes "
            "back with classifier_evaluated=false and reward 0. Use only to "
            "isolate the sidecar's latency cost or to reproduce pre-sidecar "
            "behaviour."
        ),
    )
    parser.add_argument(
        "--classifier-sidecar-interval",
        type=int,
        help=(
            "Attach a classifier frame at most once every N env steps "
            "(default: the task config). At HZ=10, 5 means ~2 Hz. Sparse on "
            "purpose: it damps flicker in the success verdict and lets the "
            "scene settle after a release before it is scored. 1 scores every "
            "step and costs the most latency."
        ),
    )
    parser.add_argument(
        "--classifier-stationary-speed-max",
        type=float,
        help=(
            "Only attach when TCP speed is below this (m/s). Keeps blurred "
            "mid-motion frames out of the classifier. Default: the task "
            "config value, which is a PLACEHOLDER until measured on the rig."
        ),
    )
    parser.add_argument(
        "--classifier-escalate-probability",
        type=float,
        help=(
            "NOT a random chance: the classifier probability at or above which "
            "the scheduler abandons the interval and scores EVERY step, so the "
            "step that actually crosses the reward threshold is not missed by "
            "up to interval-1 steps. Belongs below DEFAULT_REWARD_THRESHOLD "
            "(0.5). Default: the task config."
        ),
    )
    parser.add_argument(
        "--mock-policy-noise",
        type=float,
        default=0.0,
        help=(
            "Replace the server's action with zero-mean Gaussian noise of this "
            "sigma, in normalised action units. For bringing the loop up "
            "against a zero-action server: it exercises the full "
            "policy->robot->intervention path before a real policy exists. "
            "The perturbed action is what gets executed AND what gets stored, "
            "so the buffer stays self-consistent. 0 (default) disables it."
            " Requires --arm; no-submit probe mode always validates the "
            "unmodified server action."
        ),
    )
    return parser.parse_args()


def _network_config(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    configured = getattr(config, "NETWORK", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise TypeError("experiment NETWORK must be a mapping")
    result = dict(configured)
    overrides = {
        "type": args.network_type,
        "host": args.server_host,
        "port": args.server_port,
        "timeout_s": args.timeout_s,
        "max_response_age_s": args.max_response_age_s,
        "observation_schema_hash": args.observation_schema_hash,
        "expected_model_id": args.expected_model_id,
        "expected_reward_authority": args.expected_reward_authority,
        "expected_reward_model_id": args.expected_reward_model_id,
    }
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    result.setdefault("type", "grpc")
    result.setdefault("host", "127.0.0.1")
    result.setdefault("port", 50053)
    result.setdefault("timeout_s", 0.6)
    result.setdefault("max_response_age_s", 0.8)
    result.setdefault("retry_count", 1)
    result.setdefault(
        "observation_schema_hash", CANONICAL_OBSERVATION_SCHEMA_HASH
    )
    return result


def _sidecar_settings(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Resolve ``SidecarScheduler`` kwargs: task config, then CLI overrides.

    Same precedence rule as ``_network_config``: the task config is the record
    of what was measured on this rig, and a flag is a deliberate, per-run
    departure from it.
    """

    configured = getattr(config, "CLASSIFIER_SIDECAR", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise TypeError("experiment CLASSIFIER_SIDECAR must be a mapping")
    result = dict(configured)
    overrides = {
        "interval_steps": args.classifier_sidecar_interval,
        "stationary_speed_max": args.classifier_stationary_speed_max,
        "escalate_probability": args.classifier_escalate_probability,
    }
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    # --no-classifier-sidecar is a kill switch, not an override: it wins over
    # anything the task config says.
    if args.no_classifier_sidecar:
        result["enabled"] = False
    result.setdefault("enabled", True)
    unknown = set(result) - {
        "enabled",
        "interval_steps",
        "stationary_speed_max",
        "escalate_probability",
    }
    if unknown:
        # Typos here are silent otherwise: an unrecognised key would be dropped
        # and the run would use a default cadence nobody chose.
        raise SystemExit(
            "unknown CLASSIFIER_SIDECAR settings: " + ", ".join(sorted(unknown))
        )
    return result


def _build_sidecar_scheduler(settings: Mapping[str, Any]) -> Any:
    """Construct the scheduler, or ``None`` when the sidecar is disabled."""

    if not settings.get("enabled", True):
        print(
            "[remote-actor] classifier sidecar DISABLED: the server receives "
            "no uncropped frame and cannot score reward for this run.",
            flush=True,
        )
        return None

    kwargs = {key: value for key, value in settings.items() if key != "enabled"}
    scheduler = SidecarScheduler(enabled=True, **kwargs)
    print(
        "[remote-actor] classifier sidecar: every "
        f"{settings.get('interval_steps')} steps when TCP speed < "
        f"{settings.get('stationary_speed_max')} m/s "
        f"(every step once p >= {settings.get('escalate_probability')}). "
        "Reward is scored on those steps only -- this is intended, not a "
        "shortcut.",
        flush=True,
    )
    return scheduler


def _mock_policy_transform(sigma: float):
    """Gaussian jitter on the pose channels, for bring-up without a policy.

    The server we bring up against returns zero actions, so the robot never
    moves and the policy -> robot -> intervention path cannot be exercised.
    This substitutes a stand-in "policy" of the requested magnitude.

    The gripper channel (index 6) is deliberately left untouched: the demo
    contract requires it to be exactly one of {-1, 0, 1}, and jittering it
    would produce transitions the learner's strict loader rejects.  Pose
    channels are clipped to the [-1, 1] action space.
    """

    if sigma <= 0.0:
        return None
    if not math.isfinite(sigma):
        raise SystemExit("--mock-policy-noise must be finite")

    rng = np.random.default_rng()

    def transform(action):
        jittered = np.asarray(action, dtype=np.float32).copy()
        noise = rng.normal(0.0, sigma, size=jittered[:6].shape)
        jittered[:6] = np.clip(jittered[:6] + noise, -1.0, 1.0)
        return jittered

    print(
        f"[remote-actor] MOCK POLICY: N(0, {sigma}) on the 6 pose channels, "
        "gripper untouched. This is not a trained policy.",
        flush=True,
    )
    return transform


def _preflight_command_topics(
    env: Any,
    robot_config: Any,
    *,
    discovery_timeout_s: float = 3.0,
    poll_interval_s: float = 0.1,
) -> None:
    """Refuse to arm while another node publishes to the same controller.

    ``forward_position_controller`` is a ForwardCommandController: it applies
    whatever arrives last.  With the teleop bridge still up, two publishers
    drive it toward different targets and the arm jitters between them.  The
    reference runner has checked this since before it was ever armed
    (tests/run_real_hil.py:preflight_command_topics); adding --arm here without
    it would reopen the same hazard.

    Read-only: counting publishers publishes nothing.
    """

    base = getattr(env, "unwrapped", None)
    backend = getattr(base, "backend", None)
    node = getattr(backend, "_node", None)
    if node is None:
        print(
            "[remote-actor] cannot reach the rclpy node -- skipping the "
            "command-topic preflight.",
            flush=True,
        )
        return

    ros_config = getattr(robot_config, "ROS", {}) or {}
    topic = ros_config.get(
        "command_topic", "/forward_position_controller/commands"
    )
    # The backend node is new and can initially see zero remote readers even
    # after controller_manager has completed the switch.  A one-shot check
    # caused a false refusal immediately after a successful handoff.  Wait only
    # for graph discovery; this does not publish a command or relax ownership.
    deadline = time.monotonic() + max(0.0, float(discovery_timeout_s))
    while True:
        n_pub = node.count_publishers(topic)
        n_sub = node.count_subscribers(topic)
        if n_pub > 1 or n_sub >= 1:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(max(0.0, float(poll_interval_s)), remaining))
    print(
        f"[remote-actor] {topic}: publishers={n_pub} (ours included), "
        f"subscribers={n_sub}",
        flush=True,
    )
    if n_pub > 1:
        raise SystemExit(
            f"--arm refused: {n_pub} publishers on {topic}. Another node "
            "(the teleop bridge?) is driving the same controller; shut it "
            "down before arming."
        )
    if n_sub < 1:
        raise SystemExit(
            f"--arm refused: nothing subscribes to {topic}. Is "
            "forward_position_controller active? "
            "(ros2 control list_controllers)"
        )


def _build_actor_environment(config: Any, args: argparse.Namespace) -> Any:
    """Build the robot-local wrapper chain with an explicit task penalty."""

    # Import from the package root, not gymnasium.wrappers.record_episode_statistics:
    # gymnasium 1.0 moved the class into gymnasium.wrappers.common and deleted the
    # old per-wrapper module, so the deep path raises ModuleNotFoundError on the
    # gymnasium 1.2.0 we pin.  The package-root name is re-exported by both.
    from gymnasium.wrappers import RecordEpisodeStatistics

    task_env = config.get_environment(
        fake_env=args.fake_env,
        save_video=args.save_video,
        # Reward/termination is authoritative on the remote server.
        classifier=False,
        # A spec, not an object: RosTopicDeadman subscribes on the backend's
        # rclpy node, which does not exist until UR7eEnv has been built.
        deadman=args.deadman,
    )
    try:
        assert_actor_environment_state_layout(task_env)
        task_env = wrap_gripper_penalty_from_task_config(
            task_env,
            experiment_config=config,
        )
        return EnvTimestampAdapter(RecordEpisodeStatistics(task_env))
    except BaseException:
        task_env.close()
        raise


def _load_config_mapping(ur_config_module: str | None) -> dict:
    """Upstream's task registry if it imports, ours either way.

    ``experiments.mappings`` eagerly imports all four Franka task configs, and
    those pull in jax, pyspacemouse, hidapi and pyrealsense2.  The robot laptop
    has none of them on purpose -- inference and learning live on the remote
    GPU, and this process must stay a plain rclpy/numpy actor.  Requiring the
    Franka registry just to look up a UR task name would make jax a hard
    dependency of the robot side.

    So a failed upstream import is downgraded to a warning, but only when a UR
    registry was supplied: without one there is nothing left to look up and the
    original error is the useful message.
    """

    mapping: dict = {}
    try:
        from experiments.mappings import CONFIG_MAPPING as upstream_mapping
    except Exception as exc:  # noqa: BLE001 - any import error is equivalent here
        if not ur_config_module:
            raise
        print(
            f"[remote-actor] upstream experiments.mappings unavailable "
            f"({type(exc).__name__}: {exc}); continuing with "
            f"{ur_config_module} only",
            flush=True,
        )
    else:
        mapping.update(upstream_mapping)

    if ur_config_module:
        module = importlib.import_module(ur_config_module)
        mapping.update(module.CONFIG_MAPPING)
    return mapping


def main() -> int:
    args = _parse_args()
    if args.arm and args.fake_env:
        raise SystemExit(
            "--arm cannot be combined with --fake-env: production actor data "
            "must come from the real environment. Use "
            "scripts/run_fake_e2e_actor.py for bounded synthetic acceptance."
        )
    if not args.arm and args.mock_policy_noise != 0.0:
        raise SystemExit(
            "--mock-policy-noise requires --arm; no-arm probe mode validates "
            "the unmodified server policy action and executes nothing."
        )
    CONFIG_MAPPING = _load_config_mapping(args.ur_config_module)

    if args.exp_name not in CONFIG_MAPPING:
        raise KeyError(f"unknown experiment {args.exp_name!r}")

    config = CONFIG_MAPPING[args.exp_name]()

    # Must precede _build_actor_environment: DRY_RUN is read once, when
    # UR7eEnv constructs URRosBackend.  Setting it afterwards is a no-op.
    robot_config = getattr(config, "robot_config", None)
    if robot_config is None or not hasattr(robot_config, "DRY_RUN"):
        if args.arm:
            raise SystemExit(
                f"--arm: {args.exp_name} has no robot_config.DRY_RUN to clear"
            )
    else:
        robot_config.DRY_RUN = not args.arm
        if args.arm:
            print(
                "[remote-actor] ARMED: commands go to the robot, the UR7e "
                "moves physically. E-STOP within reach?",
                flush=True,
            )
        else:
            print(
                "[remote-actor] NO-SUBMIT PROBE: command publication is "
                "disabled. One observation and one server policy action will "
                "be validated; env.step and transition Step RPC are forbidden "
                "(pass --arm to enter the production loop).",
                flush=True,
            )

    env = _build_actor_environment(config, args)
    network = None
    try:
        if args.arm:
            _preflight_command_topics(env, robot_config)
        network_config = _network_config(config, args)
        network = create_actor_network(
            network_config,
            actor_id=args.actor_id,
            action_shape=tuple(env.action_space.shape),
        )
        alive, ready, detail = network.health()
        if not alive or not ready:
            raise RuntimeError(
                f"remote actor server is not ready: alive={alive}, "
                f"ready={ready}, detail={detail}"
            )
        if not args.arm:
            probe = run_remote_actor_probe(
                network,
                env,
                actor_id=args.actor_id,
            )
            info = probe.server_info
            source = "synthetic --fake-env" if args.fake_env else "live sensors"
            print(
                f"[remote-actor] server={network_config['type']}://"
                f"{network_config.get('host')}:{network_config.get('port')} "
                f"model={info.model_id} protocol={info.protocol_version} "
                f"schema_version={info.schema_version} "
                f"schema_hash={info.observation_schema_hash} "
                f"reward_authority={info.reward_authority} "
                f"reward_model={info.reward_model_id}",
                flush=True,
            )
            action_text = np.array2string(
                probe.policy_action,
                precision=5,
                separator=", ",
                suppress_small=False,
                max_line_width=200,
            )
            print(
                f"[remote-actor] NO-SUBMIT PROBE PASSED: source={source} "
                f"run={probe.run_id} policy_version={probe.policy_version} "
                f"action={action_text} "
                f"max_abs={float(np.max(np.abs(probe.policy_action))):.5f} "
                f"inference={probe.server_inference_ms:.2f}ms "
                f"round_trip={probe.round_trip_ms:.2f}ms. "
                "Step RPCs=0, executed actions=0, replay inserts from this "
                "actor=0. Classifier/reward/transition routing was not tested.",
                flush=True,
            )
            return 0

        info = network.get_server_info()
        print(
            f"[remote-actor] server={network_config['type']}://"
            f"{network_config.get('host')}:{network_config.get('port')} "
            f"model={info.model_id} protocol={info.protocol_version}",
            flush=True,
        )
        operator_session = None
        if args.deadman == "topic":
            # Required for the armed topic path: the existing backend node owns
            # the Trigger service and its already-running executor processes the
            # GUI edge while the actor main thread waits at HOME.
            operator_session = RosOperatorSession.from_environment(env)
        summary = run_remote_actor(
            network,
            env,
            config=config,
            actor_id=args.actor_id,
            checkpoint_path=args.checkpoint_path,
            policy_action_transform=_mock_policy_transform(
                args.mock_policy_noise
            ),
            sidecar_scheduler=_build_sidecar_scheduler(
                _sidecar_settings(config, args)
            ),
            operator_session=operator_session,
        )
        print(
            f"[remote-actor] run={summary.run_id} steps={summary.env_steps} "
            f"episodes={summary.episodes_started} "
            f"intervention_steps={summary.intervention_steps}",
            flush=True,
        )
        # Attached vs unattached round trips are reported apart so the sidecar's
        # share of the 100 ms step budget is a measurement, not a guess.
        print(
            "[remote-actor] step RTT: "
            f"sidecar n={summary.sidecar_attached_steps} "
            f"mean={summary.sidecar_round_trip_ms_mean:.1f}ms "
            f"max={summary.sidecar_round_trip_ms_max:.1f}ms | "
            f"plain mean={summary.plain_round_trip_ms_mean:.1f}ms "
            f"max={summary.plain_round_trip_ms_max:.1f}ms",
            flush=True,
        )
        if summary.sidecar_build_failures:
            print(
                "[remote-actor] !! "
                f"{summary.sidecar_build_failures} classifier sidecars could "
                "not be built; those steps carry no classifier reward.",
                flush=True,
            )
        if summary.policy_actions_synthetic:
            print(
                "[remote-actor] !! THIS RUN USED A MOCK POLICY "
                f"(--mock-policy-noise {args.mock_policy_noise}). Its actions "
                "are noise, not policy output. Every transition is stamped "
                "meta.policy_actions_synthetic=true; do not use this data as "
                "a demo or as evidence of policy behaviour.",
                flush=True,
            )
    finally:
        # The robot command publisher is the safety-critical resource.  Close
        # it first, and still close the transport if ROS teardown raises.  The
        # old network-first ordering could skip env.close() entirely when a
        # gRPC close failed, leaving the 250 Hz command thread alive.
        try:
            env.close()
        finally:
            if network is not None:
                network.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
