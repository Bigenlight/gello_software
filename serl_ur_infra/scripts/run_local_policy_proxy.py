#!/usr/bin/env python3
"""Serve the HIL-SERL policy from laptop3 and forward transitions to the server.

WHAT IT RUNS
------------
One process that listens on 127.0.0.1:50253 speaking the actor's own
ActorTransport protocol.  ``remote_actor.py`` dials it instead of the ssh
tunnel and is otherwise unchanged; the proxy answers every action locally and
replays the raw request bytes to the real learner server in the background.
See ``ur_env/local_policy/proxy.py`` for the design and
``serl_ur_infra/HIL_LOCAL_INFERENCE_KO.md`` for the operator story.

BOOT ORDER, AND WHY IT IS THIS ORDER
------------------------------------
1. **device prologue** -- ``JAX_PLATFORMS`` is read at the FIRST import of jax
   and never again, so ``--device`` must be applied before any import that
   could pull jax in.  Every heavy import in this file therefore lives inside
   :func:`main`, below the prologue, exactly like ``bench_local_policy.py``.
2. **agent** -- ``create_frozen_trunk_feature_agent``, the same network the
   learner serves, so the exported parameter tree fits leaf for leaf.
3. **initial parameters** -- poll the export directory until a blob exists.
   The proxy cannot serve without one and must not pretend it can, so this is
   a bounded, loudly-logged wait rather than a fallback to random weights.
4. **validate + smoke** -- building ``LocalPolicyRuntime`` tree-checks the
   blob and compiles BOTH policy traces (0.7-1.6 s each, once).  Paying that
   here is the point: an actor whose first Step compiles blows its deadline.
5. **remote handshake** -- Health + GetServerInfo against the real server;
   protocol/schema/observation-hash compatibility is checked by the actor's
   own client class, and the reward fields are mirrored into what this proxy
   reports.  Refusing to start on a mismatch is deliberate: the transitions
   this proxy forwards must be ones that server would have accepted anyway.
6. **serve** -- uploader, parameter poller, then the gRPC listener.  Health
   only reports SERVING once 3, 4 and 5 have all happened.

SHUTDOWN
--------
SIGINT/SIGTERM stop the listener first, then DRAIN the upload queue (bounded,
with progress logging).  Exit 0 means every captured transition reached the
server; exit 4 means some are still queued and the run's replay is incomplete.
That distinction is the whole reason draining is not fire-and-forget.

VENV
----
Needs jax AND grpc in one interpreter: ``/home/laptop3/venvs/gello-local-policy``
(jax 0.5.3 + CUDA plugin + grpcio 1.74).  ``~/venvs/hilserl`` works for a CPU
run.  Never system python3.

Examples::

    # against the live learner, params over ssh from its run root.  The run
    # root is a path ON THE SERVER, so it is written out absolute: a leading
    # ~ would be expanded by THIS shell, to this laptop's home.
    /home/laptop3/venvs/gello-local-policy/bin/python \\
        serl_ur_infra/scripts/run_local_policy_proxy.py \\
        --params-run-root \\
        /home/junhyeong/hil-serl-data/runs/cube_in_cup_real_20260806_120000

    # fully local development: a fake export directory, no ssh
    .../python serl_ur_infra/scripts/run_local_policy_proxy.py \\
        --params-fetch local --params-remote-dir /tmp/params_live --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
sys.path.insert(
    0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher")
)

# Imports here are jax-free BY CONTRACT (tests/test_local_policy_runtime.py
# asserts it for the runtime module): the device prologue below has to win the
# race against the first `import jax`, and it cannot if importing this file
# already lost it.
from ur_env.local_policy.runtime import (  # noqa: E402
    LOCAL_POLICY_DEVICE_ENV,
    configure_jax_platform,
)


DEFAULT_PARAMS_WAIT_S = 300.0
PARAMS_REMOTE_DIR_ENV = "HIL_PARAMS_REMOTE_DIR"
PARAMS_RUN_ROOT_ENV = "HIL_PARAMS_RUN_ROOT"

#: Exit codes.  Distinguished so a supervising shell can tell "never started"
#: from "started, served, and lost nothing" from "served but did not drain".
EXIT_OK = 0
EXIT_STARTUP_FAILED = 3
EXIT_DRAIN_INCOMPLETE = 4


def _emit(event: str, **fields: Any) -> None:
    """One machine-readable line per lifecycle event, like the learner's."""

    try:
        print(
            json.dumps(
                {"event": event, **fields}, sort_keys=True, separators=(",", ":")
            ),
            flush=True,
        )
    except (BrokenPipeError, OSError):
        pass


def _log(message: str) -> None:
    print(f"[local-policy-proxy] {message}", flush=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the HIL-SERL policy locally and forward transitions."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host; loopback only (the actor is on this machine).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Proxy listen port. Default: $HIL_LOCAL_POLICY_PORT, else 50253. "
            "0 lets the kernel choose (tests)."
        ),
    )
    parser.add_argument(
        "--remote-target",
        default=None,
        help=(
            "host:port of the REAL server, normally the tunnel-local end "
            "(default 127.0.0.1:50153)."
        ),
    )
    parser.add_argument(
        "--params-remote-dir",
        default=None,
        help=(
            "Directory the learner exports parameters into "
            "(<run_root>/params_live). Default: $HIL_PARAMS_REMOTE_DIR."
        ),
    )
    parser.add_argument(
        "--params-run-root",
        default=None,
        help=(
            "Learner run root; params_live is derived from it. Mutually "
            "exclusive with --params-remote-dir. Default: $HIL_PARAMS_RUN_ROOT."
        ),
    )
    parser.add_argument(
        "--params-fetch",
        choices=("ssh", "local"),
        default="ssh",
        help=(
            "How to read the export directory: ssh/scp with ControlMaster "
            "(production), or a local path (tests, dev, a shared mount)."
        ),
    )
    parser.add_argument(
        "--params-ssh-host",
        default=None,
        help="ssh alias for --params-fetch ssh. Default: $HIL_SSH_HOST or junhyeong_ai.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=None,
        help="Parameter poll cadence. Default: $HIL_PARAMS_POLL_S, else 2.0 s.",
    )
    parser.add_argument(
        "--params-wait-s",
        type=float,
        default=DEFAULT_PARAMS_WAIT_S,
        help=(
            "How long to wait for the learner's first parameter blob before "
            "giving up. The learner exports version 0 when it becomes ready."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default=None,
        help=(
            f"Backend for local inference. Default: ${LOCAL_POLICY_DEVICE_ENV}, "
            "else auto (jax picks the GPU when its CUDA plugin sees one)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="gRPC handler threads.",
    )
    parser.add_argument(
        "--upload-timeout-s",
        type=float,
        default=10.0,
        help="Per-RPC deadline for forwarded transitions (not latency critical).",
    )
    parser.add_argument(
        "--drain-timeout-s",
        type=float,
        default=60.0,
        help="How long shutdown waits for the upload queue to empty.",
    )
    parser.add_argument(
        "--status-interval-s",
        type=float,
        default=60.0,
        help="Cadence of the periodic status line. 0 disables it.",
    )
    args = parser.parse_args(argv)
    if args.params_remote_dir and args.params_run_root:
        parser.error("--params-remote-dir and --params-run-root are mutually exclusive")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("the local policy proxy is loopback-only; the actor is local")
    if args.params_wait_s <= 0 or args.drain_timeout_s <= 0:
        parser.error("--params-wait-s and --drain-timeout-s must be positive")
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    return args


def resolve_params_dir(args: argparse.Namespace, env: Any = None) -> str:
    """``--params-remote-dir`` > ``--params-run-root`` > the two env vars."""

    from ur_env.local_policy.params_sync import remote_params_dir

    source = os.environ if env is None else env
    explicit = args.params_remote_dir or source.get(PARAMS_REMOTE_DIR_ENV, "")
    if explicit:
        return str(explicit).rstrip("/")
    run_root = args.params_run_root or source.get(PARAMS_RUN_ROOT_ENV, "")
    if run_root:
        return remote_params_dir(str(run_root))
    raise SystemExit(
        "the parameter export directory is required: pass "
        "--params-remote-dir (or --params-run-root), or set "
        f"${PARAMS_REMOTE_DIR_ENV} / ${PARAMS_RUN_ROOT_ENV}"
    )


def apply_device_prologue(args: argparse.Namespace) -> Optional[str]:
    """Set ``JAX_PLATFORMS`` from ``--device`` BEFORE anything imports jax."""

    if args.device is not None:
        os.environ[LOCAL_POLICY_DEVICE_ENV] = args.device
    return configure_jax_platform()


def wait_for_initial_params(
    sync: Any,
    *,
    timeout_s: float,
    poll_interval_s: float,
    sleep: Any = time.sleep,
    clock: Any = time.monotonic,
) -> int:
    """Poll until a parameter blob has been applied.  Returns its version.

    ``poll_once`` never raises -- it converts every failure into a rate-limited
    warning -- so the only two outcomes here are "applied" and "the deadline
    passed", and the second one has to be loud: the alternative is a proxy that
    serves an untrained network to a real robot.
    """

    deadline = clock() + float(timeout_s)
    announced = False
    while True:
        if sync.poll_once():
            return int(sync.applied_version)
        version = int(sync.applied_version)
        if version >= 0:
            # Another writer (or a previous poll) already published one.
            return version
        if clock() >= deadline:
            raise SystemExit(
                f"no parameter blob appeared in {sync.remote_dir} within "
                f"{timeout_s:.0f} s.  The learner exports version 0 when it "
                "becomes ready -- check that it was started with "
                "HIL_PARAMS_EXPORT=1 and that this is its run root."
            )
        if not announced:
            announced = True
            _log(
                f"waiting for the learner's first parameter blob in "
                f"{sync.manifest_path} (up to {timeout_s:.0f} s)"
            )
        sleep(poll_interval_s)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    platform = apply_device_prologue(args)

    # Everything below imports jax (directly or transitively).  Keep it here.
    from ur_env.local_policy.params_sync import (
        FrozenTrunkParamsCodec,
        LocalDirectoryFetcher,
        ParamsHolder,
        ParamsSyncClient,
        SshParamsFetcher,
        resolve_poll_interval,
    )
    from ur_env.local_policy.proxy import (
        LocalPolicyProxy,
        ProxyLatencyProbe,
        ProxyReadiness,
        RemoteBufferStatusMirror,
        RemoteLink,
        ValidatingParamsSwap,
        DEFAULT_REMOTE_TARGET,
        GATE_PARAMS_LOADED,
        GATE_REMOTE_VERIFIED,
        GATE_SMOKE_PASSED,
        resolve_proxy_port,
    )
    from ur_env.local_policy.runtime import (
        LOCAL_POLICY_MODEL_ID,
        LocalPolicyRuntime,
        build_local_policy_agent,
    )
    from ur_env.local_policy.uploader import TransitionUploader

    params_dir = resolve_params_dir(args)
    port = resolve_proxy_port(args.port)
    remote_target = args.remote_target or DEFAULT_REMOTE_TARGET
    poll_interval_s = resolve_poll_interval(args.poll_interval_s)
    readiness = ProxyReadiness()

    _emit(
        "local_policy_proxy_starting",
        host=args.host,
        port=port,
        remote_target=remote_target,
        params_dir=params_dir,
        params_fetch=args.params_fetch,
        jax_platforms=platform or "auto",
        poll_interval_s=poll_interval_s,
    )

    proxy: Optional[Any] = None
    link: Optional[Any] = None
    exit_code = EXIT_OK
    try:
        # -- 2. the agent ------------------------------------------------- #
        started = time.monotonic()
        agent = build_local_policy_agent()
        # Read the backend BACK rather than trusting --device: a run that asked
        # for gpu and silently got cpu must be visible in the log, not in the
        # operator's assumptions.  Reporting it must never be able to fail the
        # boot, though, which is why it is wrapped.
        try:
            import jax

            devices = jax.devices()
            backend = devices[0].platform if devices else "none"
            device_count = len(devices)
        except Exception as exc:  # noqa: BLE001
            backend = f"unknown ({type(exc).__name__})"
            device_count = 0
        _emit(
            "local_policy_proxy_agent_built",
            seconds=round(time.monotonic() - started, 3),
            jax_backend=backend,
            jax_device_count=device_count,
        )
        codec = FrozenTrunkParamsCodec.from_agent(agent)

        # -- 3. the first parameter blob ---------------------------------- #
        holder = ParamsHolder()
        swap = ValidatingParamsSwap(holder)
        fetcher: Any
        if args.params_fetch == "local":
            fetcher = LocalDirectoryFetcher(params_dir)
            remote_dir = params_dir
        else:
            fetcher = SshParamsFetcher(
                **({} if args.params_ssh_host is None else {"host": args.params_ssh_host})
            )
            remote_dir = params_dir
        sync = ParamsSyncClient(
            fetcher,
            remote_dir,
            codec.load,
            holder=holder,
            swap_callback=swap,
            poll_interval_s=poll_interval_s,
        )
        version = wait_for_initial_params(
            sync,
            timeout_s=args.params_wait_s,
            poll_interval_s=poll_interval_s,
        )
        readiness.satisfy(GATE_PARAMS_LOADED, f"v{version}")
        _emit(
            "local_policy_proxy_params_loaded",
            version=version,
            learner_step=sync.applied_learner_step,
            remote_dir=sync.remote_dir,
        )

        # -- 4. validate + smoke ------------------------------------------ #
        latency_probe = ProxyLatencyProbe.from_env()
        smoke_started = time.monotonic()
        runtime = LocalPolicyRuntime(agent, holder, latency_probe=latency_probe)
        # From here every later blob is smoked on the puller thread before it
        # can reach the control loop.
        swap.set_validator(runtime.validate_candidate)
        readiness.satisfy(GATE_SMOKE_PASSED)
        _emit(
            "local_policy_proxy_smoke_passed",
            seconds=round(time.monotonic() - smoke_started, 3),
            policy_version=runtime.policy_version,
            model_id=LOCAL_POLICY_MODEL_ID,
            latency_profile=str(latency_probe.path or ""),
        )

        # -- 5. the remote handshake -------------------------------------- #
        link = RemoteLink(remote_target)
        contract = link.verify()
        mirror = RemoteBufferStatusMirror(link)
        try:
            # Prime it so the first operator GetBufferStatus has an answer
            # without the provider ever doing I/O under the service lock.
            mirror.refresh()
        except Exception as exc:  # noqa: BLE001 - a diagnostic, not a gate
            _log(f"WARNING: could not prime the remote buffer status: {exc}")
        readiness.satisfy(GATE_REMOTE_VERIFIED, contract.target)
        _emit(
            "local_policy_proxy_remote_verified",
            target=contract.target,
            remote_model_id=contract.model_id,
            reward_authority=contract.reward_authority,
            reward_model_id=contract.reward_model_id,
            observation_schema_hash=contract.observation_schema_hash,
        )

        # -- 6. serve ------------------------------------------------------ #
        uploader = TransitionUploader(
            remote_target, timeout_s=args.upload_timeout_s
        )
        proxy = LocalPolicyProxy(
            runtime,
            remote=contract,
            uploader=uploader,
            latency_probe=latency_probe,
            buffer_status_provider=mirror,
            params_sync=sync,
            params_holder=holder,
            readiness=readiness,
            host=args.host,
            port=port,
            max_workers=args.max_workers,
        )
        shutdown = threading.Event()

        def request_shutdown(_signum: int, _frame: Any) -> None:
            shutdown.set()

        old_handlers = {}
        for number in (signal.SIGINT, signal.SIGTERM):
            old_handlers[number] = signal.getsignal(number)
            signal.signal(number, request_shutdown)
        try:
            bound = proxy.start()
            alive, ready, detail = proxy.health()
            _emit(
                "local_policy_proxy_ready",
                host=args.host,
                port=bound,
                alive=alive,
                ready=ready,
                detail=detail,
                model_id=LOCAL_POLICY_MODEL_ID,
                params_version=version,
                remote_target=contract.target,
            )
            _log(
                f"serving on {args.host}:{bound} "
                f"(model_id={LOCAL_POLICY_MODEL_ID}, params v{version}, "
                f"forwarding to {contract.target}).  Ctrl-C to stop."
            )
            interval = float(args.status_interval_s)
            while not shutdown.wait(interval if interval > 0 else 3600.0):
                if interval > 0:
                    _emit("local_policy_proxy_status", **proxy.metrics())
        finally:
            for number, handler in old_handlers.items():
                try:
                    signal.signal(number, handler)
                except Exception:  # noqa: BLE001
                    pass

        _log("stopping; draining the transition upload queue...")
        drained = proxy.stop(drain=True, drain_timeout_s=args.drain_timeout_s)
        metrics = proxy.metrics()
        _emit("local_policy_proxy_stopped", drained=drained, **metrics)
        if not drained:
            _log(
                "WARNING: the upload queue did NOT drain; "
                f"{uploader.backlog_depth} transition(s) never reached "
                f"{contract.target} and are lost with this process."
            )
            exit_code = EXIT_DRAIN_INCOMPLETE
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else EXIT_STARTUP_FAILED
        if exc.code and not isinstance(exc.code, int):
            _log(f"ERROR: {exc.code}")
        _emit("local_policy_proxy_startup_failed", detail=str(exc.code)[:2000])
        return code or EXIT_STARTUP_FAILED
    except KeyboardInterrupt:
        _log("interrupted before the proxy was serving")
        return EXIT_STARTUP_FAILED
    except Exception as exc:  # noqa: BLE001 - one operator-legible failure
        _emit(
            "local_policy_proxy_startup_failed",
            error_type=type(exc).__name__,
            detail=str(exc)[:2000],
        )
        _log(f"ERROR: {type(exc).__name__}: {exc}")
        return EXIT_STARTUP_FAILED
    finally:
        if proxy is not None:
            # Idempotent: a normal shutdown already stopped it.
            proxy.stop(drain=False)
        if link is not None:
            link.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
