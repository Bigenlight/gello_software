#!/usr/bin/env python3
"""Serve a BC-initialised policy over the production actor gRPC contract.

Same wire contract as ``run_rlpd_learner_server.py`` (protocol 2 / schema 3),
but with no learner, no replay training and no reward classifier: the weights
are frozen at whatever the ``hil-serl-bc-init`` artifact contains and every
accepted transition is written to disk for offline evaluation.

Runs on port 50054 so the production learner keeps 50053 to itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sys
import threading
import traceback

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
# Mirrors the learner server: the agent factory needs serl_launcher importable
# even when the launcher did not export PYTHONPATH (local CPU smokes).
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher"))

# Nothing above this line may import jax, flax, grpc or protobuf: _configure_env
# must win the race for XLA_PYTHON_CLIENT_PREALLOCATE / CUDA_VISIBLE_DEVICES and
# for PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION, all of which are read at first
# import of their libraries.  Every ur_env import therefore lives inside _serve.

_LOG_PREFIX = "[bc-server]"

_DEFAULT_ARTIFACT_DIR = (
    "/home/junhyeong/hil-serl-data/diagnostics/"
    "bc_cube_in_cup_raw_0731_bce_group_holdout_20epoch_20260731_213638.bc-init"
)

# 50053 is the production learner's port.  This server must never take it, not
# even when an operator asks: the learner is a live online-RL lineage and a
# second bind would either fail or, after the learner dies, silently answer for
# it with frozen BC weights.
_PRODUCTION_LEARNER_PORT = 50053

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _log(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}", flush=True)


def _default_record_root() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"~/hil-serl-data/bc_eval/bc_eval_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a hil-serl-bc-init policy on the actor gRPC contract"
    )
    parser.add_argument("--artifact-dir", default=_DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50054)
    parser.add_argument("--record-root", default=_default_record_root())
    # Defaults resolve to ur_env.learner.bc_init.BC_MODEL_ID /
    # BC_REWARD_MODEL_ID inside _serve: that module cannot be imported before
    # the accelerator env vars are set.
    parser.add_argument(
        "--model-id", default="", help="default: bc_init.BC_MODEL_ID"
    )
    parser.add_argument(
        "--reward-model-id",
        default="",
        help="default: bc_init.BC_REWARD_MODEL_ID",
    )
    parser.add_argument(
        "--hil-serl-root",
        default=str(_REPO_ROOT / "third_party" / "hil-serl"),
    )
    parser.add_argument("--resnet-source")
    parser.add_argument("--resnet-cache")
    # Logging is ON by default: a rollout whose actions were not recorded
    # cannot be audited after the fact, and the log lives inside the run
    # directory so it cannot outlive or be orphaned from its recording.
    parser.add_argument(
        "--no-inference-log",
        action="store_true",
        help="do not write <record-root>/inference.jsonl",
    )
    # Off by default, unlike the inference log: it instruments the request path
    # of a live session, so it is opted into per investigation.
    parser.add_argument(
        "--step-timing",
        action="store_true",
        help=(
            "write <record-root>/timing.jsonl (also enabled by "
            "HIL_STEP_TIMING=1)"
        ),
    )
    # Empty string means "do not touch CUDA_VISIBLE_DEVICES" -- CPU-only smokes.
    parser.add_argument("--gpu-index", default="0")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.host not in _LOOPBACK_HOSTS:
        raise ValueError("bc policy server is loopback-only; use an SSH tunnel")
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    if args.port == _PRODUCTION_LEARNER_PORT:
        raise ValueError(
            f"port {_PRODUCTION_LEARNER_PORT} belongs to the production RLPD "
            "learner; the BC policy server must use a different port "
            "(default 50054)"
        )
    if not str(args.artifact_dir).strip():
        raise ValueError("--artifact-dir is required")
    if not str(args.record_root).strip():
        raise ValueError("--record-root is required")


def _configure_env(args: argparse.Namespace) -> None:
    """Set the accelerator env vars before anything can import jax.

    ``setdefault`` throughout so an operator export on the command line or in
    the launcher wins over these defaults.
    """

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    gpu_index = str(args.gpu_index)
    if gpu_index:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_index)


def _grpc_bind_address(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _serve(args: argparse.Namespace) -> int:
    from ur_env.compat import (
        configure_flax_local_io,
        configure_pure_python_protobuf,
    )

    configure_pure_python_protobuf()

    from ur_env.actor_network import ActorSessionService
    from ur_env.bc_inference_log import InferenceLoggingPolicy
    from ur_env.bc_recording_sink import EpisodeRecordingSink
    from ur_env.grpc_actor_transport import create_grpc_server
    from ur_env.learner import (
        FrozenResNet10TrunkExtractor,
        LearnerConfig,
        VersionedPolicyRuntime,
        create_frozen_trunk_feature_agent,
        default_resnet_source,
    )
    from ur_env.learner.bc_init import (
        BC_MODEL_ID,
        BC_REWARD_MODEL_ID,
        BcInitError,
        deterministic_sample_action,
        load_bc_init_manifest,
        load_bc_init_params,
        verify_resnet_asset,
    )
    from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH
    from ur_env.step_timing import (
        SERVER_TIMING_FILENAME,
        STEP_TIMING_ENV,
        FailOpenJsonlWriter,
        ServiceStepTimingProxy,
        StepTimingRecorder,
        TimingSink,
        step_timing_enabled,
    )

    step_timing = args.step_timing or step_timing_enabled(
        os.environ.get(STEP_TIMING_ENV)
    )

    model_id = args.model_id or BC_MODEL_ID
    reward_model_id = args.reward_model_id or BC_REWARD_MODEL_ID

    # No config knobs are exposed: the artifact was trained against the
    # production defaults, and a mismatch must surface as a load-time graft
    # rejection rather than as a server that quietly reshapes the agent.
    config = LearnerConfig()
    _log(f"config ready image_keys={','.join(config.image_keys)}")

    resnet_source = (
        Path(args.resnet_source or default_resnet_source())
        .expanduser()
        .resolve()
    )
    artifact_dir = Path(args.artifact_dir).expanduser()
    manifest = load_bc_init_manifest(artifact_dir)
    verify_resnet_asset(manifest, resnet_source)
    artifact_sha256 = str(manifest.get("parameter_sha256", "") or "")
    if not artifact_sha256:
        raise BcInitError("manifest is missing parameter_sha256")
    _log(
        f"artifact verified dir={artifact_dir} "
        f"parameter_sha256={artifact_sha256}"
    )

    configure_flax_local_io()
    agent_template = create_frozen_trunk_feature_agent(
        config=config,
        hil_serl_root=args.hil_serl_root,
        resnet_source_path=resnet_source,
        resnet_cache_path=args.resnet_cache,
        validate_versions=False,
    )
    _log("agent template created")

    grafted = load_bc_init_params(artifact_dir, agent_template)
    _log("bc-init parameters grafted")

    extractor = FrozenResNet10TrunkExtractor(
        agent_template,
        resnet_asset_path=resnet_source,
        image_keys=config.image_keys,
    )
    _log(f"trunk extractor ready resnet_sha256={extractor.resnet_sha256}")

    # The constructor validates the grafted tree, re-checks the frozen trunk
    # and smokes both the deterministic and stochastic action contracts.  Any
    # failure here must reach the operator, not be downgraded.
    runtime = VersionedPolicyRuntime(
        agent_template,
        params=grafted,
        model_id=model_id,
        sample_action=deterministic_sample_action(agent_template),
        parameter_validator=extractor.validate_parameter_invariant,
    )
    _log(f"policy runtime ready model_id={runtime.model_id}")

    # Created last of the on-disk side effects so an assembly failure leaves no
    # stray run directory behind and a retry on the same path still works.
    record_root = Path(args.record_root).expanduser()
    record_root.mkdir(parents=True, exist_ok=False)
    sink = EpisodeRecordingSink(
        record_root,
        artifact_sha256=artifact_sha256,
        model_id=model_id,
    )
    _log(f"recording sink ready root={record_root}")

    # ``sink`` keeps naming the recorder itself: the shutdown summary reads its
    # counters, and TimingSink deliberately has no __getattr__ to forward them.
    accept_data = sink
    timing_writer = None
    timing_recorder = None
    if step_timing:
        timing_writer = FailOpenJsonlWriter(record_root / SERVER_TIMING_FILENAME)
        timing_recorder = StepTimingRecorder(timing_writer)
        accept_data = TimingSink(sink, timing_recorder)
        _log(f"step timing path={record_root / SERVER_TIMING_FILENAME}")

    # Wrapped here and not earlier: the log path lives under record_root, which
    # does not exist until the line above, and VersionedPolicyRuntime's
    # constructor smoke inferences must stay out of the rollout log.
    if args.no_inference_log:
        policy = runtime
        inference_logger = None
        _log("inference log path=disabled")
    else:
        inference_logger = InferenceLoggingPolicy(
            runtime, record_root / "inference.jsonl"
        )
        policy = inference_logger
        _log(f"inference log path={record_root / 'inference.jsonl'}")

    # reward_authority="local": there is no server classifier in this process,
    # so the operator's MARK SUCCESS is the only success authority.  The
    # default identity finalizer keeps that path intact.
    service = ActorSessionService(
        sample_action=policy,
        model_id=model_id,
        reward_authority="local",
        reward_model_id=reward_model_id,
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        accept_data=accept_data,
    )
    if timing_recorder is not None:
        # After the smoke inferences above, which call the policy directly:
        # timing.jsonl stays a record of actor requests only.
        service = ServiceStepTimingProxy(service, timing_recorder)
    server, bound_port = create_grpc_server(
        service, bind_address=_grpc_bind_address(args.host, args.port)
    )

    shutdown = threading.Event()

    def request_shutdown(_signum, _frame) -> None:
        shutdown.set()

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, request_shutdown)

    server.start()
    # EXACT line the launcher greps for; absence of it means startup failed.
    ready_line = (
        f"{_LOG_PREFIX} ready host={args.host} port={bound_port} "
        f"model_id={model_id} parameter_sha256={artifact_sha256} "
        f"record_root={record_root}"
    )
    if timing_writer is not None:
        # Appended, never inserted: the launcher matches the fixed prefix
        # '[bc-server] ready ', so a trailing token cannot break the grep.
        ready_line += " step_timing=1"
    print(ready_line, flush=True)
    try:
        while not shutdown.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(grace=5.0).wait()
    stopped = (
        f"stopped replay_count={sink.replay_count} "
        f"intervention_count={sink.intervention_count} "
        f"record_root={record_root}"
    )
    if inference_logger is not None:
        # getattr fallback: a shutdown report must not raise on a wrapper that
        # never got to count anything.
        stopped += f" inference_calls={getattr(inference_logger, 'call_count', 0)}"
    if timing_writer is not None:
        stopped += f" timing_records={getattr(timing_writer, 'write_count', 0)}"
        timing_writer.close()
    _log(stopped)
    return 0


def main() -> int:
    args = _parse_args()
    try:
        _validate_args(args)
        _configure_env(args)
        return _serve(args)
    except Exception as exc:  # noqa: BLE001 - one fail-closed operator report
        print(
            f"{_LOG_PREFIX} FATAL {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
