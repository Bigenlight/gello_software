#!/usr/bin/env python3
"""Measure PURE policy-inference latency locally: no gRPC, no robot, no actor.

WHAT THIS MEASURES, AND WHAT IT IS COMPARABLE TO
------------------------------------------------
``run_bc_policy_server.py`` / ``run_fm_policy_server.py`` serve one action per
``Step`` RPC and, with ``--step-timing``, record an ``infer_ms`` span around the
policy call.  This script builds the SAME policy callable, from the SAME
artifact, through the SAME loaders, and times exactly that call.  The number it
prints is therefore comparable to the servers' ``infer_ms`` and is a floor under
any measured wire latency: most of what the server adds (protobuf decode,
feature ingress, recording sink, gRPC) sits outside its ``infer_ms`` span too.

THE COMPARISON IS ONE-DIRECTIONAL: bench <= server ``infer_ms``.  Three things
sit INSIDE the server's timed span and OUTSIDE this bench's:

* ``copy_observation(observation)`` -- the defensive per-call observation copy
  (``ur_env/actor_network.py:1505``), which walks and duplicates the image
  tensors before the policy ever sees them;
* ``validate_action`` / ``validate_counter`` on the returned action and policy
  version (``ur_env/actor_network.py:1507-1511``);
* the ``InferenceLoggingPolicy`` wrapper, which json-encodes one record and
  ``write`` + ``flush``-es it to disk on EVERY call
  (``ur_env/bc_inference_log.py:134-181``).  It is ON by default in both local
  launchers -- ``run_bc_policy_server.py`` only drops it for
  ``--no-inference-log`` -- and it wraps the policy, so its own cost lands
  inside the server's ``infer_ms``.

So a server ``infer_ms`` slightly above a bench number is expected and is not
evidence of a slower GPU; only the reverse would be surprising.

It is NOT comparable to an end-to-end actor step: no camera decode, no
``env.step`` pacing, no network.

NO-LOAD CAVEAT: laptop3 has one 6 GB GPU shared with the camera viewers, the
GUI and (when serving locally) the policy server itself.  A bench taken on an
idle machine is a lower bound on what a live session sees, not a prediction of
it.  The ``gpu_memory_used_mib`` snapshot in every result record exists so a
reader can tell an idle bench from a contended one after the fact.

WHY THE ENV PROLOGUE IS SPLIT FROM THE IMPORTS
-----------------------------------------------
``XLA_PYTHON_CLIENT_PREALLOCATE`` and ``JAX_PLATFORMS`` are read at the first
import of jax and never again.  Every jax/flax/ur_env-learner import in this
file therefore lives INSIDE a function, below ``configure_env``, exactly like
the two server scripts.  The side effect is deliberately useful: the module
itself imports in the actor venv, which has no jax at all, so the arg parsing,
the statistics and the writers are unit-testable there
(``tests/test_bench_local_policy.py`` asserts ``jax`` stays out of
``sys.modules`` after importing this module).

DEVICE SELECTION
----------------
``--device cpu``  -> ``JAX_PLATFORMS=cpu``   (works in ~/venvs/hilserl)
``--device gpu``  -> ``JAX_PLATFORMS=cuda``  (needs a CUDA-enabled jax venv)
``--device auto`` -> leave ``JAX_PLATFORMS`` alone; jax picks.

The request is only a request.  Every result record carries
``device_platform``, read back from the live ``jax.devices()[0].platform``, so a
run that asked for gpu and silently got cpu is visible in the output rather
than in the operator's assumptions.

OBSERVATIONS
------------
Default is synthetic: ``canonical_policy_observation()`` from
``ur_env/learner/policy.py`` -- the production contract builder, never a
re-invented shape.  Inference latency does not depend on pixel content, so
synthetic observations measure the same thing as real ones; ``--obs-pkl`` exists
to prove that rather than to assume it.  Example corpus::

    --obs-pkl /home/laptop3/hil-serl-artifacts/demos/cube_in_cup_20260720_success_23takes.pkl

Run (from /home/laptop3/gello_software)::

    /home/laptop3/venvs/hilserl/bin/python serl_ur_infra/scripts/bench_local_policy.py \\
        --policy both --device cpu --iters 200

    /home/laptop3/venvs/gello-local-policy/bin/python \\
        serl_ur_infra/scripts/bench_local_policy.py --policy both --device gpu
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "serl_ur_infra"))
# Mirrors the two policy servers: the agent factory needs serl_launcher
# importable even when the caller did not export PYTHONPATH.
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher"))

# Nothing above this line -- and nothing at module scope below it -- may import
# jax, flax or any ur_env.learner module that pulls them: configure_env must win
# the race for XLA_PYTHON_CLIENT_PREALLOCATE / JAX_PLATFORMS, which are read at
# the first import of jax and never re-read.

_LOG_PREFIX = "[bench]"

#: laptop3-local mirror of the server's diagnostics layout (fetched by
#: ``ros2_ur_ws/fetch_policy_artifacts.sh``).  The server scripts' own defaults
#: point at ``/home/junhyeong/...`` and are wrong here by construction.
_DEFAULT_BC_ARTIFACT_DIR = (
    "/home/laptop3/hil-serl-data/diagnostics/"
    "bc_cube_in_cup_raw_0731_bce_group_holdout_20epoch_20260731_213638.bc-init"
)
_DEFAULT_FM_ARTIFACT_DIR = (
    "/home/laptop3/hil-serl-data/diagnostics/"
    "jax_fm_cube_in_cup_raw_0731_h16_euler8_200epoch_20260731_215835.fm-init"
)

_DEFAULT_BENCH_ROOT = "/home/laptop3/hil-serl-data/bench"

#: Documentation only -- printed in ``--help`` and in the report so the operator
#: knows which corpus ``--obs-pkl`` was meant for (ADDENDUM 6).
EXAMPLE_OBS_PKL = (
    "/home/laptop3/hil-serl-artifacts/demos/"
    "cube_in_cup_20260720_success_23takes.pkl"
)

RESULTS_FILENAME = "bench_results.jsonl"
REPORT_FILENAME = "bench_report.md"

#: ``--device`` -> ``JAX_PLATFORMS``.  ``None`` means "do not set it".
DEVICE_PLATFORMS: Mapping[str, str | None] = {
    "cpu": "cpu",
    "gpu": "cuda",
    "auto": None,
}

#: The keys every result record carries, in report order.  Extra keys are
#: allowed (provenance); these are the ones a consumer may rely on.
RESULT_FIELDS = (
    "policy",
    "mode",
    "device_platform",
    "jax_version",
    "iters",
    "warmup",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "max_ms",
    "min_ms",
    "obs_source",
    "artifact_sha",
)

#: FM has no deterministic mode: ``FmServedPolicy.__call__`` does ``del
#: deterministic`` and always integrates the ODE from a fresh Gaussian draw.
#: Benchmarking "both modes" for FM would report the same trace twice under two
#: names, so only one is measured and this sentence goes into the report.
FM_DETERMINISM_NOTE = (
    "FM ignores the deterministic flag (fm_serving.FmServedPolicy.__call__ "
    "drops it and always samples from noise), so only one mode is measured."
)

#: BC has no stochastic mode either, for the mirror-image reason: the served
#: sampler does ``del deterministic`` and hard-codes ``argmax=True``
#: (``ur_env/learner/bc_init.py:451-456``, ``deterministic_sample_action``), so
#: the flag never reaches the network.  Measuring "deterministic" and
#: "stochastic" would time ONE trace twice and publish the pure timing noise
#: between the two runs as if it were a mode difference -- which is exactly what
#: an earlier version of this bench did.  One measurement, one honest label.
BC_DETERMINISM_NOTE = (
    "BC ignores the deterministic flag (learner.bc_init.deterministic_sample_"
    "action drops it and hard-codes argmax=True, bc_init.py:451-456), so only "
    "the served argmax mode is measured; a second 'stochastic' row would be the "
    "same trace timed twice."
)

#: What ``mode`` says for each policy.  Both are single-trace labels on purpose:
#: neither policy honours the wire ``deterministic`` flag.
BC_MODE = "served-argmax"
FM_MODE = "stochastic"

#: The bench-vs-server caveat, in one line, for the report.  Long form lives in
#: the module docstring.
SERVER_SPAN_NOTE = (
    "Bench <= server infer_ms by construction: copy_observation "
    "(actor_network.py:1505), validate_action/validate_counter (1507-1511) and "
    "the InferenceLoggingPolicy json write+flush per call "
    "(bc_inference_log.py:134-181, on by default in both local launchers) are "
    "inside the server's timed span and outside this one. The bias is "
    "one-directional."
)


def _log(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}", flush=True)


# --------------------------------------------------------------------------- #
# Pure-python helpers: arg parsing, statistics, writers.  None of these may     #
# import jax, so the actor venv can test them.                                  #
# --------------------------------------------------------------------------- #


def default_out_dir(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    return f"{_DEFAULT_BENCH_ROOT}/bench_{stamp}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure pure BC/FM policy inference latency locally (no gRPC, "
            "no robot)"
        )
    )
    parser.add_argument(
        "--policy", choices=("bc", "fm", "both"), default="both"
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="untimed calls before measurement, to absorb jit compilation",
    )
    parser.add_argument(
        "--device",
        choices=tuple(DEVICE_PLATFORMS),
        default="auto",
        help="cpu -> JAX_PLATFORMS=cpu, gpu -> cuda, auto -> leave unset",
    )
    parser.add_argument("--bc-artifact-dir", default=_DEFAULT_BC_ARTIFACT_DIR)
    parser.add_argument("--fm-artifact-dir", default=_DEFAULT_FM_ARTIFACT_DIR)
    parser.add_argument(
        "--fm-which",
        choices=("best", "final"),
        default="best",
        help="which parameter file inside the FM artifact to benchmark",
    )
    parser.add_argument(
        "--obs-pkl",
        default="",
        help=(
            "canonical demo pickle to draw real observations from; default is "
            f"synthetic. Example: {EXAMPLE_OBS_PKL}"
        ),
    )
    parser.add_argument(
        "--obs-count",
        type=int,
        default=32,
        help="how many observations to cycle through during the bench",
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--hil-serl-root",
        default=str(_REPO_ROOT / "third_party" / "hil-serl"),
    )
    parser.add_argument("--resnet-source")
    parser.add_argument("--resnet-cache")
    # Empty string means "do not touch CUDA_VISIBLE_DEVICES".  laptop3 has one
    # GPU, so unlike the servers (which default to "0" on an 8-GPU host) the
    # default here is to leave the operator's environment alone.
    parser.add_argument("--gpu-index", default="")
    args = parser.parse_args(argv)
    if not str(args.out).strip():
        args.out = default_out_dir()
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.obs_count <= 0:
        raise ValueError("--obs-count must be positive")
    if args.policy not in ("bc", "fm", "both"):
        raise ValueError("--policy must be bc, fm or both")
    if args.device not in DEVICE_PLATFORMS:
        raise ValueError("--device must be cpu, gpu or auto")
    if args.fm_which not in ("best", "final"):
        raise ValueError("--fm-which must be best or final")
    if not str(args.out).strip():
        raise ValueError("--out is required")


def jax_platforms_for_device(device: str) -> str | None:
    """Return the ``JAX_PLATFORMS`` value for ``--device``, or None."""

    if device not in DEVICE_PLATFORMS:
        raise ValueError(f"unknown device: {device!r}")
    return DEVICE_PLATFORMS[device]


def configure_env(
    args: argparse.Namespace,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Set the accelerator env vars before anything can import jax.

    ``XLA_PYTHON_CLIENT_PREALLOCATE`` uses ``setdefault`` like the servers do:
    an operator export is a deliberate choice and must win.  ``JAX_PLATFORMS``
    does NOT: ``--device`` is a per-invocation request, and a stale export that
    silently redirected the run would be invisible in the command line.  The
    override is logged, and the result record reports the platform actually
    obtained, so neither path can mislabel a measurement.
    """

    target = os.environ if environ is None else environ
    target.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    gpu_index = str(args.gpu_index or "")
    if gpu_index:
        target.setdefault("CUDA_VISIBLE_DEVICES", gpu_index)
    platform = jax_platforms_for_device(args.device)
    if platform is None:
        return
    previous = target.get("JAX_PLATFORMS")
    if previous is not None and previous != platform:
        _log(
            f"WARNING overriding JAX_PLATFORMS={previous!r} with {platform!r} "
            f"for --device {args.device}"
        )
    target["JAX_PLATFORMS"] = platform


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method."""

    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (q / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(samples_ms: Sequence[float]) -> dict[str, float]:
    """Reduce per-call latencies to the five numbers the report quotes."""

    if not samples_ms:
        raise ValueError("cannot summarize an empty sample")
    values = [float(value) for value in samples_ms]
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 50.0),
        "p95_ms": percentile(values, 95.0),
        "max_ms": max(values),
        "min_ms": min(values),
    }


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Deterministic spread of ``count`` indices over ``range(total)``.

    Deterministic on purpose: two benches of the same corpus must draw the same
    observations, otherwise a latency difference cannot be attributed.
    """

    if total <= 0:
        raise ValueError("total must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= total:
        return list(range(total))
    return [(index * total) // count for index in range(count)]


def nvidia_smi_memory_used_mib(timeout_s: float = 5.0) -> int | None:
    """First GPU's used memory in MiB, or None.  Fail-open by design.

    A bench must never die because a diagnostic snapshot was unavailable.
    """

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            return int(token)
        except ValueError:
            return None
    return None


def write_results_jsonl(path: Path | str, results: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as stream:
        for record in results:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return destination


def read_results_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _format_ms(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def render_report(
    results: Sequence[Mapping[str, Any]],
    *,
    failures: Sequence[Mapping[str, str]] = (),
    notes: Sequence[str] = (),
) -> str:
    """Render the small operator-facing table.  Prose is not a contract."""

    lines: list[str] = ["# Local policy inference bench", ""]
    if not results:
        lines.append("No measurement completed.")
        lines.append("")
    else:
        lines.append(
            "| policy | mode | device | iters | mean_ms | p50_ms | p95_ms | "
            "max_ms | min_ms | obs | artifact_sha |"
        )
        lines.append(
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "--- | --- |"
        )
        for record in results:
            sha = str(record.get("artifact_sha", "") or "")
            lines.append(
                "| {policy} | {mode} | {device} | {iters} | {mean} | {p50} | "
                "{p95} | {max} | {min} | {obs} | {sha} |".format(
                    policy=record.get("policy", "?"),
                    mode=record.get("mode", "?"),
                    device=record.get("device_platform", "?"),
                    iters=record.get("iters", "?"),
                    mean=_format_ms(record.get("mean_ms")),
                    p50=_format_ms(record.get("p50_ms")),
                    p95=_format_ms(record.get("p95_ms")),
                    max=_format_ms(record.get("max_ms")),
                    min=_format_ms(record.get("min_ms")),
                    obs=record.get("obs_source", "?"),
                    sha=sha[:12] if sha else "n/a",
                )
            )
        lines.append("")
    if failures:
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(
                f"- **{failure.get('policy', '?')}**: "
                f"{failure.get('error', 'unknown error')}"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append(
        "- Timed span is the policy call only: no gRPC, no recording sink, no "
        "camera decode. Comparable to the servers' `infer_ms`, not to an "
        "actor step."
    )
    lines.append(f"- {SERVER_SPAN_NOTE}")
    lines.append(
        "- Warmup calls are untimed and exist to absorb jit compilation; the "
        "first real call of a cold process is not in these numbers."
    )
    lines.append(
        "- Measured on an otherwise idle machine unless noted. laptop3 shares "
        "one 6 GB GPU with the viewers and the local policy server, so a live "
        "session can be slower; `gpu_memory_used_mib` in the jsonl records "
        "what was resident at measurement time."
    )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    out_dir: Path | str,
    results: Sequence[Mapping[str, Any]],
    *,
    failures: Sequence[Mapping[str, str]] = (),
    notes: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Write ``bench_results.jsonl`` + ``bench_report.md`` into ``out_dir``."""

    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    results_path = directory / RESULTS_FILENAME
    report_path = directory / REPORT_FILENAME
    write_results_jsonl(results_path, results)
    report_path.write_text(
        render_report(results, failures=failures, notes=notes), encoding="utf-8"
    )
    return results_path, report_path


def refuse_existing_results(out_dir: Path | str) -> None:
    """Refuse to overwrite a previous bench's results in the same directory."""

    existing = Path(out_dir).expanduser() / RESULTS_FILENAME
    if existing.exists():
        raise FileExistsError(
            f"{existing} already exists; pass a different --out rather than "
            "overwriting a previous measurement"
        )


# --------------------------------------------------------------------------- #
# Everything below imports jax (lazily, inside the functions).                  #
# --------------------------------------------------------------------------- #


def build_synthetic_observations(count: int) -> list[dict[str, Any]]:
    """Canonical contract observations, straight from the production builder.

    Shapes/dtypes are NEVER restated here: ``canonical_policy_observation`` is
    the same function the servers smoke with, so a contract change cannot
    silently leave this bench measuring the wrong tensor.  The pixel value is
    varied per observation only so a hypothetical constant-folding cache cannot
    flatter the numbers; inference cost does not depend on it.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    from ur_env.learner import canonical_policy_observation

    return [canonical_policy_observation(index % 256) for index in range(count)]


def load_pickle_observations(path: Path | str, count: int) -> list[dict[str, Any]]:
    """Draw ``count`` real observations from a canonical demo pickle.

    Uses the production strict loader (``load_demo_pickle``) rather than a
    private read: anything it rejects is not an observation the servers would
    ever see.  It validates the whole corpus, which costs seconds and roughly a
    gigabyte transiently for the 2,037-transition canonical demo -- acceptable
    for an opt-in flag, and the alternative (a bespoke reader) is exactly the
    kind of drift this repo has been bitten by.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    from ur_env.learner import load_demo_pickle

    loaded = load_demo_pickle(path)
    total = len(loaded.transitions)
    if total == 0:
        raise ValueError(f"{path} contains no transitions")
    indices = evenly_spaced_indices(total, count)
    return [dict(loaded.transitions[index]["observations"]) for index in indices]


def _jax_runtime_info() -> dict[str, Any]:
    import jax

    devices = jax.devices()
    platform = str(devices[0].platform) if devices else "unknown"
    return {
        "jax_version": str(jax.__version__),
        "device_platform": platform,
        "jax_devices": repr(devices),
    }


def _block_on(value: Any) -> None:
    """Force the action to be host-resident inside the timed span.

    Both policy callables already ``device_get`` internally, so this is a
    re-assertion rather than the only barrier -- but a future policy that
    returned a jax array would otherwise let async dispatch report a dispatch
    time as an inference time.
    """

    import numpy as np

    ready = getattr(value, "block_until_ready", None)
    if callable(ready):
        ready()
        return
    array = np.asarray(value)
    if array.size:
        float(array.reshape(-1)[0])


def measure_policy(
    policy: Any,
    observations: Sequence[Mapping[str, Any]],
    *,
    deterministic: bool,
    iters: int,
    warmup: int,
) -> list[float]:
    """Time ``iters`` policy calls in milliseconds, after ``warmup`` untimed."""

    if not observations:
        raise ValueError("no observations to bench")
    if iters <= 0:
        raise ValueError("iters must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    for index in range(warmup):
        action, _version = policy(
            observations[index % len(observations)], deterministic
        )
        _block_on(action)
    samples: list[float] = []
    for index in range(iters):
        observation = observations[index % len(observations)]
        started = time.perf_counter()
        action, _version = policy(observation, deterministic)
        _block_on(action)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def load_bc_policy(args: argparse.Namespace) -> tuple[Any, str]:
    """Build the BC policy exactly as ``run_bc_policy_server.py`` does.

    Same loaders, same graft, same ``VersionedPolicyRuntime`` construction
    (server lines 190-238) minus the recording/inference-log wrappers, which are
    not part of the ``infer_ms`` span this bench reproduces.
    """

    from ur_env.compat import configure_flax_local_io
    from ur_env.learner import (
        FrozenResNet10TrunkExtractor,
        LearnerConfig,
        VersionedPolicyRuntime,
        create_frozen_trunk_feature_agent,
        default_resnet_source,
    )
    from ur_env.learner.bc_init import (
        BC_MODEL_ID,
        BcInitError,
        deterministic_sample_action,
        load_bc_init_manifest,
        load_bc_init_params,
        verify_resnet_asset,
    )

    config = LearnerConfig()
    resnet_source = (
        Path(args.resnet_source or default_resnet_source())
        .expanduser()
        .resolve()
    )
    artifact_dir = Path(args.bc_artifact_dir).expanduser()
    manifest = load_bc_init_manifest(artifact_dir)
    verify_resnet_asset(manifest, resnet_source)
    artifact_sha256 = str(manifest.get("parameter_sha256", "") or "")
    if not artifact_sha256:
        raise BcInitError("manifest is missing parameter_sha256")
    _log(f"bc artifact verified dir={artifact_dir} sha={artifact_sha256}")

    configure_flax_local_io()
    agent_template = create_frozen_trunk_feature_agent(
        config=config,
        hil_serl_root=args.hil_serl_root,
        resnet_source_path=resnet_source,
        resnet_cache_path=args.resnet_cache,
        validate_versions=False,
    )
    grafted = load_bc_init_params(artifact_dir, agent_template)
    extractor = FrozenResNet10TrunkExtractor(
        agent_template,
        resnet_asset_path=resnet_source,
        image_keys=config.image_keys,
    )
    runtime = VersionedPolicyRuntime(
        agent_template,
        params=grafted,
        model_id=BC_MODEL_ID,
        sample_action=deterministic_sample_action(agent_template),
        parameter_validator=extractor.validate_parameter_invariant,
    )
    _log(f"bc policy ready model_id={runtime.model_id}")
    return runtime, artifact_sha256


def load_fm_policy(args: argparse.Namespace) -> tuple[Any, str]:
    """Build the FM policy exactly as ``run_fm_policy_server.py`` does.

    Same ``load_flow_artifact`` + ``FmServedPolicy`` assembly (server lines
    260-317).  ``--rng-seed``/``--integration-steps`` are not exposed: this
    bench measures the SERVED sampler, and a different Euler count is a
    different policy whose number would not compare to the server's.
    """

    from ur_env.compat import configure_flax_local_io
    from ur_env.fm_serving import FM_MODEL_ID, FmServedPolicy, FmServingError
    from ur_env.learner import (
        FrozenResNet10TrunkExtractor,
        LearnerConfig,
        create_frozen_trunk_feature_agent,
        default_resnet_source,
    )
    from ur_env.learner.bc_init import verify_resnet_asset
    from ur_env.learner.flow_matching import load_flow_artifact

    config = LearnerConfig()
    resnet_source = (
        Path(args.resnet_source or default_resnet_source())
        .expanduser()
        .resolve()
    )
    artifact_dir = Path(args.fm_artifact_dir).expanduser()

    configure_flax_local_io()
    model, params, manifest = load_flow_artifact(artifact_dir, which=args.fm_which)
    verify_resnet_asset(manifest, resnet_source)
    parameter_record = (manifest.get("parameter_files") or {}).get(
        args.fm_which
    ) or {}
    artifact_sha256 = str(parameter_record.get("sha256", "") or "")
    if not artifact_sha256:
        raise FmServingError(
            f"manifest is missing parameter_files[{args.fm_which!r}].sha256"
        )
    _log(
        f"fm artifact verified dir={artifact_dir} which={args.fm_which} "
        f"sha={artifact_sha256}"
    )

    agent_template = create_frozen_trunk_feature_agent(
        config=config,
        hil_serl_root=args.hil_serl_root,
        resnet_source_path=resnet_source,
        resnet_cache_path=args.resnet_cache,
        validate_versions=False,
    )
    extractor = FrozenResNet10TrunkExtractor(
        agent_template,
        resnet_asset_path=resnet_source,
        image_keys=config.image_keys,
    )
    policy = FmServedPolicy(
        model,
        params,
        extractor,
        policy_version=int(manifest.get("best_epoch", 0)),
        model_id=FM_MODEL_ID,
    )
    _log(f"fm policy ready model_id={FM_MODEL_ID}")
    return policy, artifact_sha256


def _result_record(
    *,
    policy: str,
    mode: str,
    samples_ms: Sequence[float],
    args: argparse.Namespace,
    obs_source: str,
    obs_count: int,
    artifact_sha: str,
    runtime_info: Mapping[str, Any],
    note: str = "",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "policy": policy,
        "mode": mode,
        "device_platform": runtime_info.get("device_platform", "unknown"),
        "jax_version": runtime_info.get("jax_version", "unknown"),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "obs_source": obs_source,
        "artifact_sha": artifact_sha,
    }
    record.update(summarize(samples_ms))
    record.update(
        {
            "device_request": args.device,
            "jax_devices": runtime_info.get("jax_devices", ""),
            "obs_count": int(obs_count),
            # Gated on "not cpu", never on a positive platform name: a jax
            # CudaDevice reports ``platform == "gpu"``, not "cuda" (measured on
            # laptop3's RTX 3060), so an == "cuda" test left this field None on
            # every single GPU run -- the one run it exists for.  "cpu" is the
            # only platform we can be sure has no GPU snapshot worth taking;
            # everything else (gpu/cuda/rocm/unknown) gets asked, and
            # ``nvidia_smi_memory_used_mib`` is already fail-open.
            "gpu_memory_used_mib": (
                nvidia_smi_memory_used_mib()
                if str(runtime_info.get("device_platform", "")) != "cpu"
                else None
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": note,
        }
    )
    return record


def _observations(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    obs_pkl = str(args.obs_pkl or "").strip()
    if obs_pkl:
        observations = load_pickle_observations(obs_pkl, args.obs_count)
        return observations, f"pkl:{obs_pkl}"
    return build_synthetic_observations(args.obs_count), "synthetic"


def run_bench(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]], list[dict[str, str]], list[str]
]:
    """Load, warm and time every requested (policy, mode) pair."""

    observations, obs_source = _observations(args)
    _log(f"observations ready count={len(observations)} source={obs_source}")
    runtime_info = _jax_runtime_info()
    _log(
        f"jax {runtime_info['jax_version']} platform="
        f"{runtime_info['device_platform']} devices={runtime_info['jax_devices']}"
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    notes: list[str] = []

    if args.policy in ("bc", "both"):
        try:
            policy, artifact_sha = load_bc_policy(args)
            # ONE measurement, with deterministic=False -- the flag the
            # production actor actually sends.  See BC_DETERMINISM_NOTE: the
            # served sampler discards it, so a second "deterministic" row would
            # be this same trace timed twice.
            samples = measure_policy(
                policy,
                observations,
                deterministic=False,
                iters=args.iters,
                warmup=args.warmup,
            )
            record = _result_record(
                policy="bc",
                mode=BC_MODE,
                samples_ms=samples,
                args=args,
                obs_source=obs_source,
                obs_count=len(observations),
                artifact_sha=artifact_sha,
                runtime_info=runtime_info,
                note=BC_DETERMINISM_NOTE,
            )
            results.append(record)
            notes.append(BC_DETERMINISM_NOTE)
            _log(
                f"bc {BC_MODE} mean_ms={record['mean_ms']:.3f} "
                f"p50_ms={record['p50_ms']:.3f} p95_ms={record['p95_ms']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            failures.append({"policy": "bc", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{_LOG_PREFIX} FATAL bc {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()

    if args.policy in ("fm", "both"):
        try:
            policy, artifact_sha = load_fm_policy(args)
            samples = measure_policy(
                policy,
                observations,
                deterministic=False,
                iters=args.iters,
                warmup=args.warmup,
            )
            record = _result_record(
                policy="fm",
                mode=FM_MODE,
                samples_ms=samples,
                args=args,
                obs_source=obs_source,
                obs_count=len(observations),
                artifact_sha=artifact_sha,
                runtime_info=runtime_info,
                note=FM_DETERMINISM_NOTE,
            )
            results.append(record)
            notes.append(FM_DETERMINISM_NOTE)
            _log(
                f"fm {FM_MODE} mean_ms={record['mean_ms']:.3f} "
                f"p50_ms={record['p50_ms']:.3f} p95_ms={record['p95_ms']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            failures.append({"policy": "fm", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{_LOG_PREFIX} FATAL fm {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()

    notes.append(f"observations: {obs_source} (count={len(observations)})")
    notes.append(
        f"jax {runtime_info['jax_version']} on "
        f"{runtime_info['device_platform']}; devices="
        f"{runtime_info['jax_devices']}"
    )
    return results, failures, notes


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        refuse_existing_results(args.out)
        configure_env(args)
        results, failures, notes = run_bench(args)
        # Written even on partial failure: a bench that measured BC and lost FM
        # must not also lose BC.
        results_path, report_path = write_outputs(
            args.out, results, failures=failures, notes=notes
        )
        _log(f"wrote results={results_path}")
        _log(f"wrote report={report_path}")
        if failures:
            _log(
                "FAILED policies: "
                + ", ".join(sorted(item["policy"] for item in failures))
            )
            return 1
        if not results:
            _log("no results measured")
            return 1
        return 0
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
