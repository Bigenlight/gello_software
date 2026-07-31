#!/usr/bin/env python3
"""Serve a flow-matching policy over the production actor gRPC contract.

Sibling of ``run_bc_policy_server.py``: same wire contract as
``run_rlpd_learner_server.py`` (protocol 2 / schema 3), no learner, no replay
training and no reward classifier.  The weights are frozen at whatever the
``hil-serl-jax-flow-matching`` artifact contains and every accepted transition
is written to disk for offline evaluation.

Runs on port 50055 so the production learner keeps 50053 and the BC policy
server keeps 50054.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sys
import threading
import time
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

_LOG_PREFIX = "[fm-server]"

_DEFAULT_ARTIFACT_DIR = (
    "/home/junhyeong/hil-serl-data/diagnostics/"
    "jax_fm_cube_in_cup_raw_0731_h16_euler8_200epoch_20260731_215835.fm-init"
)

# The only artifact family this entrypoint knows how to integrate.  load_flow_
# artifact already rejects anything else; re-asserted here so a future loader
# that relaxes the check cannot silently hand this server a foreign model.
_FLOW_ARTIFACT_FORMAT = "hil-serl-jax-flow-matching"

# 50053 is the production learner's port.  This server must never take it, not
# even when an operator asks: the learner is a live online-RL lineage and a
# second bind would either fail or, after the learner dies, silently answer for
# it with frozen FM weights.
_PRODUCTION_LEARNER_PORT = 50053

# 50054 belongs to the BC policy server.  Allowed but loud: the two eval
# servers are interchangeable at the wire level, so a mixed-up port produces a
# perfectly healthy session that evaluated the wrong policy family.
_BC_POLICY_SERVER_PORT = 50054

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _log(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}", flush=True)


def _default_record_root() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"~/hil-serl-data/fm_eval/fm_eval_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a hil-serl-jax-flow-matching policy on the actor gRPC "
            "contract"
        )
    )
    parser.add_argument("--artifact-dir", default=_DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--which",
        choices=("best", "final"),
        default="best",
        help="which parameter file inside the artifact to serve",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50055)
    parser.add_argument("--record-root", default=_default_record_root())
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="seed for the flow sampler's noise draw",
    )
    # None means "use the integration_steps the model was trained with" (8).
    # Overriding it changes the sampler, not the weights: a rollout served at a
    # different step count is a different policy and must be recorded as such.
    parser.add_argument(
        "--integration-steps",
        type=int,
        default=None,
        help="Euler steps for ODE integration (default: the model's own)",
    )
    # Defaults resolve to ur_env.fm_serving.FM_MODEL_ID / FM_REWARD_MODEL_ID
    # inside _serve: that module cannot be imported before the accelerator env
    # vars are set.
    parser.add_argument(
        "--model-id", default="", help="default: fm_serving.FM_MODEL_ID"
    )
    parser.add_argument(
        "--reward-model-id",
        default="",
        help="default: fm_serving.FM_REWARD_MODEL_ID",
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
    # Empty string means "do not touch CUDA_VISIBLE_DEVICES" -- CPU-only smokes.
    parser.add_argument("--gpu-index", default="0")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.host not in _LOOPBACK_HOSTS:
        raise ValueError("fm policy server is loopback-only; use an SSH tunnel")
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    if args.port == _PRODUCTION_LEARNER_PORT:
        raise ValueError(
            f"port {_PRODUCTION_LEARNER_PORT} belongs to the production RLPD "
            "learner; the FM policy server must use a different port "
            "(default 50055)"
        )
    if not str(args.artifact_dir).strip():
        raise ValueError("--artifact-dir is required")
    if not str(args.record_root).strip():
        raise ValueError("--record-root is required")
    # Both are re-checked by FmServedPolicy's constructor.  Repeated here only
    # so a typo is refused now instead of after the minute it takes to build
    # the agent template and load the artifact.
    if args.integration_steps is not None and args.integration_steps <= 0:
        raise ValueError("--integration-steps must be a positive integer")
    if args.rng_seed < 0:
        raise ValueError("--rng-seed must be non-negative")
    if args.port == _BC_POLICY_SERVER_PORT:
        # Not fatal: nothing is unsafe about it, and an operator running only
        # this server may legitimately reuse an already-tunnelled port.
        _log(
            f"WARNING port {_BC_POLICY_SERVER_PORT} is the BC policy server's "
            "port; make sure the tunnel and the eval you think you are running "
            "both mean the FM policy"
        )


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
    import numpy as np

    from ur_env.compat import (
        configure_flax_local_io,
        configure_pure_python_protobuf,
    )

    configure_pure_python_protobuf()

    from ur_env.actor_network import (
        ActorSessionService,
        validate_action,
        validate_counter,
    )
    from ur_env.bc_inference_log import InferenceLoggingPolicy
    from ur_env.bc_recording_sink import EpisodeRecordingSink
    from ur_env.fm_serving import (
        FM_MODEL_ID,
        FM_REWARD_MODEL_ID,
        FmServedPolicy,
        FmServingError,
    )
    from ur_env.grpc_actor_transport import create_grpc_server
    from ur_env.learner import (
        FrozenResNet10TrunkExtractor,
        LearnerConfig,
        canonical_policy_observation,
        create_frozen_trunk_feature_agent,
        default_resnet_source,
    )
    # bc_init is imported for one helper only: verify_resnet_asset reads
    # manifest["resnet_sha256"], which the FM manifest carries under the same
    # key, so the two artifact families share the check verbatim.
    from ur_env.learner.bc_init import verify_resnet_asset
    from ur_env.learner.flow_matching import load_flow_artifact
    from ur_env.observation_schema import CANONICAL_OBSERVATION_SCHEMA_HASH

    model_id = args.model_id or FM_MODEL_ID
    reward_model_id = args.reward_model_id or FM_REWARD_MODEL_ID

    # No config knobs are exposed: the artifact was trained against the
    # production defaults, and a mismatch must surface as a load-time rejection
    # rather than as a server that quietly reshapes the feature extractor.
    config = LearnerConfig()
    _log(f"config ready image_keys={','.join(config.image_keys)}")

    resnet_source = (
        Path(args.resnet_source or default_resnet_source())
        .expanduser()
        .resolve()
    )
    artifact_dir = Path(args.artifact_dir).expanduser()

    configure_flax_local_io()
    # Verifies completion.json, the manifest SHA-256 and the chosen parameter
    # file's SHA-256 internally; anything it accepts is byte-identical to what
    # the trainer wrote.
    model, params, manifest = load_flow_artifact(artifact_dir, which=args.which)

    manifest_format = str(manifest.get("format", ""))
    if manifest_format != _FLOW_ARTIFACT_FORMAT:
        raise FmServingError(
            f"unsupported flow artifact format: {manifest_format!r} "
            f"(expected {_FLOW_ARTIFACT_FORMAT!r})"
        )
    # The FM head consumes cached trunk features, so a trainer that saw
    # different trunk weights produced a head fed features this process cannot
    # reproduce.  Undetectable downstream, so it is caught here.
    verify_resnet_asset(manifest, resnet_source)

    parameter_files = manifest.get("parameter_files") or {}
    parameter_record = parameter_files.get(args.which) or {}
    artifact_sha256 = str(parameter_record.get("sha256", "") or "")
    if not artifact_sha256:
        raise FmServingError(
            f"manifest is missing parameter_files[{args.which!r}].sha256"
        )
    _log(
        f"artifact verified dir={artifact_dir} which={args.which} "
        f"parameter_sha256={artifact_sha256}"
    )

    # The agent template exists only to source the verified frozen trunk for
    # the extractor: the FM policy has its own parameters and never touches the
    # SAC heads.
    agent_template = create_frozen_trunk_feature_agent(
        config=config,
        hil_serl_root=args.hil_serl_root,
        resnet_source_path=resnet_source,
        resnet_cache_path=args.resnet_cache,
        validate_versions=False,
    )
    _log("agent template created")

    extractor = FrozenResNet10TrunkExtractor(
        agent_template,
        resnet_asset_path=resnet_source,
        image_keys=config.image_keys,
    )
    _log(f"trunk extractor ready resnet_sha256={extractor.resnet_sha256}")

    policy_version = int(manifest.get("best_epoch", 0))
    policy = FmServedPolicy(
        model,
        params,
        extractor,
        policy_version=policy_version,
        rng_seed=args.rng_seed,
        integration_steps=args.integration_steps,
        model_id=model_id,
    )
    _log(
        f"fm policy ready model_id={model_id} policy_version={policy_version} "
        f"rng_seed={args.rng_seed} "
        f"integration_steps={args.integration_steps or 'model-default'}"
    )

    def validated_smoke_action(result: object, *, name: str) -> "np.ndarray":
        """Mirror VersionedPolicyRuntime._validated_policy_action exactly.

        FmServedPolicy is a plain callable with no runtime wrapper, so nothing
        else in this process ever checks the wire contract before the first
        real actor request does -- at which point a violation is an actor-side
        ActorProtocolError mid-episode instead of a startup refusal.
        """

        if not isinstance(result, tuple) or len(result) != 2:
            raise FmServingError(
                f"{name} must be an (action, policy_version) tuple"
            )
        value, version = result
        try:
            # The transport's own counter rule, applied by its own validator.
            validate_counter(version, name=f"{name} policy_version")
        except Exception as exc:
            raise FmServingError(str(exc)) from exc
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise FmServingError(f"{name} could not be materialized") from exc
        if array.dtype != np.dtype(np.float32):
            raise FmServingError(
                f"{name} must have dtype float32, got {array.dtype}"
            )
        try:
            # Shape (7,), finite, and within [-1, 1] -- the transport's own
            # rules, applied by the transport's own validator.
            action = validate_action(array, action_shape=(7,), name=name)
        except Exception as exc:
            raise FmServingError(str(exc)) from exc
        if float(action[-1]) not in (-1.0, 0.0, 1.0):
            raise FmServingError(
                f"{name} gripper component must be in {{-1, 0, 1}}"
            )
        return action

    # Both sampling modes are smoked before ready is printed: deterministic and
    # stochastic take distinct JAX traces, and warming only one would let the
    # server advertise ready while the first real request of the other mode
    # spent its RPC budget compiling.
    observation = canonical_policy_observation()
    for deterministic, label in (
        (True, "deterministic fm smoke action"),
        (False, "stochastic fm smoke action"),
    ):
        started = time.perf_counter()
        result = policy(observation, deterministic)
        latency_ms = (time.perf_counter() - started) * 1000.0
        validated_smoke_action(result, name=label)
        _log(
            f"policy smoke ok deterministic={int(deterministic)} "
            f"latency_ms={latency_ms:.1f}"
        )

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

    # Wrapped here and not earlier: the log path lives under record_root, which
    # does not exist until the line above, and the startup smoke inferences
    # must stay out of the rollout log.
    if args.no_inference_log:
        served_policy = policy
        inference_logger = None
        _log("inference log path=disabled")
    else:
        inference_logger = InferenceLoggingPolicy(
            policy, record_root / "inference.jsonl"
        )
        served_policy = inference_logger
        _log(f"inference log path={record_root / 'inference.jsonl'}")

    # reward_authority="local": there is no server classifier in this process,
    # so the operator's MARK SUCCESS is the only success authority.  The
    # default identity finalizer keeps that path intact.
    service = ActorSessionService(
        sample_action=served_policy,
        model_id=model_id,
        reward_authority="local",
        reward_model_id=reward_model_id,
        observation_schema_hash=CANONICAL_OBSERVATION_SCHEMA_HASH,
        accept_data=sink,
    )
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
    print(
        f"{_LOG_PREFIX} ready host={args.host} port={bound_port} "
        f"model_id={model_id} parameter_sha256={artifact_sha256} "
        f"record_root={record_root}",
        flush=True,
    )
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
