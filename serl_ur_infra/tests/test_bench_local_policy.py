"""Contract tests for ``scripts/bench_local_policy.py``.

WHY MOST OF THIS RUNS IN THE ACTOR VENV
---------------------------------------
The bench exists to time jax, but the part that can silently go wrong is the
part around jax: which ``JAX_PLATFORMS`` a ``--device`` request produces, what
the percentiles mean, and whether the jsonl a later comparison table is built
from actually round-trips.  All of that is pure python, and it is pinned here in
``/home/laptop3/venvs/gello-hil-actor`` -- the interpreter with NO jax -- so a
regression cannot hide behind an import error or a skip.

That is also a load-bearing property of the bench module itself: its accelerator
env prologue must win the race against the first ``import jax``, which is only
possible if nothing at module scope imports jax.  ``test_module_import_does_not_
import_jax`` asserts exactly that, in-process and again in a fresh interpreter.

THE ONE REAL-INFERENCE TEST IS OPT-IN
-------------------------------------
``RUN_HIL_SERL_ACTUAL_LOCAL_BENCH=1`` plus a jax-capable interpreter plus the
fetched BC artifact.  It runs the real loader and three real inferences through
``main()``; without all three preconditions it skips with a reason.

Run (from /home/laptop3/gello_software)::

    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \\
      PYTHONPATH="$PWD/serl_ur_infra:$PWD/third_party/hil-serl/serl_launcher" \\
      /home/laptop3/venvs/gello-hil-actor/bin/python -m pytest -q \\
      -p no:cacheprovider serl_ur_infra/tests/test_bench_local_policy.py
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


_HERE = Path(os.path.abspath(__file__)).parent
_SERL_UR_INFRA = _HERE.parent
_REPO_ROOT = _SERL_UR_INFRA.parent
_SCRIPT = _SERL_UR_INFRA / "scripts" / "bench_local_policy.py"

_REAL_BENCH_ENV = "RUN_HIL_SERL_ACTUAL_LOCAL_BENCH"


def _load_module():
    """Import the CLI from its path: ``scripts/`` is not a package."""

    assert _SCRIPT.is_file(), f"pinned CLI is missing: {_SCRIPT}"
    spec = importlib.util.spec_from_file_location("_bench_local_policy", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_module()

#: Sampled at import time, before any test can pull jax in for its own reasons.
_JAX_PRESENT_AFTER_IMPORT = "jax" in sys.modules


# --------------------------------------------------------------------------- #
# Argument parsing                                                             #
# --------------------------------------------------------------------------- #


def test_defaults_match_the_documented_contract():
    args = bench.parse_args([])
    assert args.policy == "both"
    assert args.iters == 200
    assert args.warmup == 10
    assert args.device == "auto"
    assert args.fm_which == "best"
    assert args.obs_pkl == ""
    assert args.obs_count == 32
    assert args.bc_artifact_dir == (
        "/home/laptop3/hil-serl-data/diagnostics/"
        "bc_cube_in_cup_raw_0731_bce_group_holdout_20epoch_20260731_213638.bc-init"
    )
    assert args.fm_artifact_dir == (
        "/home/laptop3/hil-serl-data/diagnostics/"
        "jax_fm_cube_in_cup_raw_0731_h16_euler8_200epoch_20260731_215835.fm-init"
    )
    # Default output lands under the laptop-local bench root, timestamped so two
    # benches cannot silently share a directory.
    assert args.out.startswith("/home/laptop3/hil-serl-data/bench/bench_")
    # Artifact defaults are laptop-local, NOT the servers' /home/junhyeong paths.
    assert "/home/junhyeong" not in args.bc_artifact_dir
    assert "/home/junhyeong" not in args.fm_artifact_dir


def test_explicit_arguments_win():
    args = bench.parse_args(
        [
            "--policy",
            "bc",
            "--iters",
            "7",
            "--warmup",
            "2",
            "--device",
            "gpu",
            "--fm-which",
            "final",
            "--obs-pkl",
            "/tmp/demo.pkl",
            "--obs-count",
            "4",
            "--out",
            "/tmp/bench-out",
        ]
    )
    assert (args.policy, args.iters, args.warmup, args.device) == ("bc", 7, 2, "gpu")
    assert args.fm_which == "final"
    assert args.obs_pkl == "/tmp/demo.pkl"
    assert args.obs_count == 4
    assert args.out == "/tmp/bench-out"


@pytest.mark.parametrize(
    "argv",
    [
        ["--iters", "0"],
        ["--warmup", "-1"],
        ["--obs-count", "0"],
    ],
)
def test_validate_args_refuses_degenerate_counts(argv):
    with pytest.raises(ValueError):
        bench.validate_args(bench.parse_args(argv))


def test_validate_args_accepts_the_defaults():
    bench.validate_args(bench.parse_args([]))


def test_unknown_choices_are_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        bench.parse_args(["--policy", "sac"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--device", "tpu"])


# --------------------------------------------------------------------------- #
# device -> JAX_PLATFORMS, without importing jax                               #
# --------------------------------------------------------------------------- #


def test_device_maps_to_jax_platforms():
    assert bench.jax_platforms_for_device("cpu") == "cpu"
    # "gpu" is the operator's word; jax's platform name is "cuda".
    assert bench.jax_platforms_for_device("gpu") == "cuda"
    assert bench.jax_platforms_for_device("auto") is None
    with pytest.raises(ValueError):
        bench.jax_platforms_for_device("rocm")


def test_configure_env_sets_platform_and_preallocate():
    environ: dict[str, str] = {}
    bench.configure_env(bench.parse_args(["--device", "gpu"]), environ)
    assert environ["JAX_PLATFORMS"] == "cuda"
    assert environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    # No --gpu-index by default: laptop3 has one GPU and the operator's
    # CUDA_VISIBLE_DEVICES is left alone.
    assert "CUDA_VISIBLE_DEVICES" not in environ


def test_configure_env_cpu_and_auto():
    cpu_env: dict[str, str] = {}
    bench.configure_env(bench.parse_args(["--device", "cpu"]), cpu_env)
    assert cpu_env["JAX_PLATFORMS"] == "cpu"

    auto_env: dict[str, str] = {}
    bench.configure_env(bench.parse_args(["--device", "auto"]), auto_env)
    assert "JAX_PLATFORMS" not in auto_env
    assert auto_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_operator_preallocate_export_wins_but_device_request_does_not_lose():
    environ = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "JAX_PLATFORMS": "cpu",
    }
    bench.configure_env(bench.parse_args(["--device", "gpu"]), environ)
    # setdefault: a deliberate memory choice is respected.
    assert environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    # NOT setdefault: a stale export must never silently redirect a --device run.
    assert environ["JAX_PLATFORMS"] == "cuda"


def test_gpu_index_is_opt_in():
    environ: dict[str, str] = {}
    bench.configure_env(
        bench.parse_args(["--device", "gpu", "--gpu-index", "0"]), environ
    )
    assert environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_configure_env_does_not_import_jax():
    environ: dict[str, str] = {}
    bench.configure_env(bench.parse_args(["--device", "cpu"]), environ)
    assert "jax" not in sys.modules


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #


def test_percentiles_on_a_hand_checkable_sample():
    # 101 samples 1..101: the percentile index is q/100*(n-1) = q exactly, so
    # every quantile below is an integer no reader has to trust a library for.
    samples = [float(value) for value in range(1, 102)]
    assert bench.percentile(samples, 50.0) == 51.0
    assert bench.percentile(samples, 95.0) == 96.0
    assert bench.percentile(samples, 0.0) == 1.0
    assert bench.percentile(samples, 100.0) == 101.0


def test_percentile_interpolates_between_neighbours():
    # Two samples: p50 is the midpoint, p95 is 95% of the way up.
    assert bench.percentile([10.0, 20.0], 50.0) == 15.0
    assert bench.percentile([10.0, 20.0], 95.0) == pytest.approx(19.5)


def test_percentile_is_order_independent():
    assert bench.percentile([5.0, 1.0, 3.0], 50.0) == 3.0


def test_percentile_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError):
        bench.percentile([], 50.0)
    with pytest.raises(ValueError):
        bench.percentile([1.0], 101.0)


def test_summarize_known_values():
    samples = [float(value) for value in range(1, 102)]
    stats = bench.summarize(samples)
    assert stats["mean_ms"] == pytest.approx(51.0)
    assert stats["p50_ms"] == 51.0
    assert stats["p95_ms"] == 96.0
    assert stats["max_ms"] == 101.0
    assert stats["min_ms"] == 1.0


def test_summarize_single_sample():
    stats = bench.summarize([4.5])
    assert stats == {
        "mean_ms": 4.5,
        "p50_ms": 4.5,
        "p95_ms": 4.5,
        "max_ms": 4.5,
        "min_ms": 4.5,
    }


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        bench.summarize([])


def test_evenly_spaced_indices_is_deterministic_and_bounded():
    assert bench.evenly_spaced_indices(10, 5) == [0, 2, 4, 6, 8]
    assert bench.evenly_spaced_indices(10, 1) == [0]
    # Asking for more than exist yields every index once, never a duplicate.
    assert bench.evenly_spaced_indices(3, 8) == [0, 1, 2]
    with pytest.raises(ValueError):
        bench.evenly_spaced_indices(0, 3)
    with pytest.raises(ValueError):
        bench.evenly_spaced_indices(10, 0)


# --------------------------------------------------------------------------- #
# Writers                                                                      #
# --------------------------------------------------------------------------- #


def _record(policy: str = "bc", mode: str = "served-argmax") -> dict:
    return {
        "policy": policy,
        "mode": mode,
        "device_platform": "cpu",
        "jax_version": "0.5.3",
        "iters": 200,
        "warmup": 10,
        "mean_ms": 6.25,
        "p50_ms": 6.0,
        "p95_ms": 8.5,
        "max_ms": 12.0,
        "min_ms": 5.5,
        "obs_source": "synthetic",
        "artifact_sha": "8ffcfac5" + "0" * 56,
        "gpu_memory_used_mib": None,
    }


def test_jsonl_round_trip(tmp_path):
    records = [_record(), _record(policy="fm", mode="stochastic")]
    path = bench.write_results_jsonl(tmp_path / "bench_results.jsonl", records)
    assert path.is_file()
    reloaded = bench.read_results_jsonl(path)
    assert reloaded == records
    # One JSON object per line, no trailing blank record.
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["policy"] == "bc"


def test_write_outputs_creates_both_files(tmp_path):
    out_dir = tmp_path / "bench_20260806"
    records = [_record(), _record(policy="fm", mode="stochastic")]
    results_path, report_path = bench.write_outputs(
        out_dir, records, notes=["observations: synthetic (count=32)"]
    )
    assert results_path.name == bench.RESULTS_FILENAME
    assert report_path.name == bench.REPORT_FILENAME
    assert bench.read_results_jsonl(results_path) == records

    report = report_path.read_text(encoding="utf-8")
    # Every measured config appears as a row, with its numbers.
    assert "| bc | served-argmax | cpu |" in report
    assert "| fm | stochastic | cpu |" in report
    assert "6.250" in report and "8.500" in report
    # Short sha, not the whole digest, but enough to identify the artifact.
    assert "8ffcfac5" in report
    assert "observations: synthetic (count=32)" in report


def test_report_records_failures_and_the_fm_determinism_caveat(tmp_path):
    _results_path, report_path = bench.write_outputs(
        tmp_path,
        [_record()],
        failures=[{"policy": "fm", "error": "FileNotFoundError: missing artifact"}],
        notes=[bench.FM_DETERMINISM_NOTE],
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Failures" in report
    assert "missing artifact" in report
    assert "deterministic flag" in report


def test_report_of_an_empty_bench_says_so():
    report = bench.render_report([])
    assert "No measurement completed." in report


def test_report_always_states_the_one_directional_server_bias():
    """The bench is a FLOOR, and the report has to say why.

    ``copy_observation``, the action/counter validators and the per-call
    inference-log write+flush are inside the server's ``infer_ms`` and outside
    this bench's span, so a server number slightly above a bench number is
    expected rather than evidence of a slower device.  A reader who does not see
    this sentence will read the gap as measurement disagreement.
    """

    report = bench.render_report([_record()])
    assert bench.SERVER_SPAN_NOTE in report
    for citation in (
        "copy_observation",
        "validate_action",
        "InferenceLoggingPolicy",
        "one-directional",
    ):
        assert citation in report


def test_refuse_existing_results_protects_a_previous_bench(tmp_path):
    bench.write_outputs(tmp_path, [_record()])
    with pytest.raises(FileExistsError):
        bench.refuse_existing_results(tmp_path)
    # A fresh directory is fine.
    bench.refuse_existing_results(tmp_path / "another")


# --------------------------------------------------------------------------- #
# Measurement core: measure_policy / _result_record / run_bench.               #
#                                                                              #
# NOT opt-in.  These are the functions the published numbers come out of, and   #
# until this section existed the only coverage of them was behind              #
# RUN_HIL_SERL_ACTUAL_LOCAL_BENCH -- i.e. never run.  A fake policy with the    #
# served call signature needs no jax, so the loop contract, the record schema   #
# and the per-policy row count are all pinnable in the actor venv.             #
# --------------------------------------------------------------------------- #


class _FakePolicy:
    """Served-policy call signature, zero jax: ``(observation, deterministic)``."""

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def __call__(self, observation, deterministic):
        self.calls.append(bool(deterministic))
        return np.zeros(7, np.float32), 0


def _runtime_info(platform: str = "gpu") -> dict:
    return {
        "jax_version": "0.5.3",
        "device_platform": platform,
        "jax_devices": "[CudaDevice(id=0)]",
    }


def _bench_args(tmp_path, *extra):
    return bench.parse_args(
        ["--iters", "5", "--warmup", "2", "--out", str(tmp_path), *extra]
    )


def test_measure_policy_excludes_warmup_from_the_statistics():
    """warmup calls happen, and are not sampled.

    The whole point of ``--warmup`` is that a jit compilation lands in an
    untimed call; if the warmup loop were folded into the samples the p95 of
    every cold run would be the compile time.
    """

    policy = _FakePolicy()
    samples = bench.measure_policy(
        policy,
        [{"state": np.zeros(19, np.float32)}],
        deterministic=False,
        iters=5,
        warmup=2,
    )
    assert len(policy.calls) == 7  # 2 warmup + 5 timed
    assert len(samples) == 5  # ...but only the timed ones are sampled
    assert all(value >= 0.0 for value in samples)
    # The flag is passed through verbatim on every call, warmup included.
    assert policy.calls == [False] * 7


def test_measure_policy_cycles_observations_and_refuses_degenerate_counts():
    observations = [{"i": np.float32(index)} for index in range(3)]
    policy = _FakePolicy()
    bench.measure_policy(
        policy, observations, deterministic=True, iters=4, warmup=0
    )
    assert policy.calls == [True] * 4
    with pytest.raises(ValueError):
        bench.measure_policy(policy, [], deterministic=False, iters=1, warmup=0)
    with pytest.raises(ValueError):
        bench.measure_policy(
            policy, observations, deterministic=False, iters=0, warmup=0
        )
    with pytest.raises(ValueError):
        bench.measure_policy(
            policy, observations, deterministic=False, iters=1, warmup=-1
        )


def test_result_record_carries_exactly_the_documented_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "nvidia_smi_memory_used_mib", lambda *a, **k: 4321)
    record = bench._result_record(
        policy="bc",
        mode=bench.BC_MODE,
        samples_ms=[1.0, 2.0, 3.0],
        args=_bench_args(tmp_path),
        obs_source="synthetic",
        obs_count=32,
        artifact_sha="ab" * 32,
        runtime_info=_runtime_info("cpu"),
        note="a note",
    )
    assert set(record) == {
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
        "device_request",
        "jax_devices",
        "obs_count",
        "gpu_memory_used_mib",
        "timestamp",
        "note",
    }
    # Every field the report and any downstream comparison rely on is present.
    for field in bench.RESULT_FIELDS:
        assert field in record
    assert record["iters"] == 5 and record["warmup"] == 2
    assert record["mean_ms"] == pytest.approx(2.0)
    assert record["note"] == "a note"


@pytest.mark.parametrize(
    "platform, expected",
    [
        # jax's CudaDevice reports platform "gpu", not "cuda" (measured on
        # laptop3's RTX 3060).  Gating the snapshot on == "cuda" left this field
        # None on every GPU run -- the only run it exists for.
        ("gpu", 4321),
        ("cuda", 4321),
        ("cpu", None),
    ],
)
def test_gpu_memory_snapshot_is_taken_on_every_non_cpu_platform(
    tmp_path, monkeypatch, platform, expected
):
    monkeypatch.setattr(bench, "nvidia_smi_memory_used_mib", lambda *a, **k: 4321)
    record = bench._result_record(
        policy="bc",
        mode=bench.BC_MODE,
        samples_ms=[1.0],
        args=_bench_args(tmp_path),
        obs_source="synthetic",
        obs_count=1,
        artifact_sha="ab" * 32,
        runtime_info=_runtime_info(platform),
    )
    assert record["gpu_memory_used_mib"] == expected


def test_bc_is_measured_once_because_the_flag_never_reaches_the_network(
    tmp_path, monkeypatch
):
    """Regression pin: ONE bc row, not a deterministic/stochastic pair.

    ``deterministic_sample_action`` does ``del deterministic`` and hard-codes
    ``argmax=True`` (bc_init.py:451-456).  Timing "both modes" measured one
    trace twice and published the noise between the two runs as a comparable
    difference.
    """

    policy = _FakePolicy()
    monkeypatch.setattr(bench, "load_bc_policy", lambda args: (policy, "cd" * 32))
    monkeypatch.setattr(bench, "_jax_runtime_info", lambda: _runtime_info("gpu"))
    monkeypatch.setattr(
        bench,
        "_observations",
        lambda args: ([{"state": np.zeros(19, np.float32)}], "synthetic"),
    )
    monkeypatch.setattr(bench, "nvidia_smi_memory_used_mib", lambda *a, **k: 4321)

    results, failures, notes = bench.run_bench(
        _bench_args(tmp_path, "--policy", "bc")
    )
    assert failures == []
    assert len(results) == 1
    record = results[0]
    assert (record["policy"], record["mode"]) == ("bc", "served-argmax")
    assert record["mode"] != "deterministic"
    # deterministic=False is what the production actor sends, so that is what is
    # measured; no call asks for the flag the sampler would ignore anyway.
    assert policy.calls == [False] * 7  # 2 warmup + 5 timed, one mode only
    # The caveat reaches BOTH sinks: the jsonl record and the report.
    assert record["note"] == bench.BC_DETERMINISM_NOTE
    assert bench.BC_DETERMINISM_NOTE in notes
    assert record["gpu_memory_used_mib"] == 4321

    report = bench.render_report(results, notes=notes)
    assert len([line for line in report.splitlines() if line.startswith("| bc |")]) == 1
    assert "argmax=True" in report


# --------------------------------------------------------------------------- #
# Synthetic observations, and the no-jax-at-import invariant                   #
# --------------------------------------------------------------------------- #


def test_synthetic_observations_come_from_the_production_builder():
    from ur_env.learner import canonical_policy_observation

    observations = bench.build_synthetic_observations(3)
    assert len(observations) == 3
    expected = canonical_policy_observation(0)
    for observation in observations:
        assert set(observation) == set(expected)
        for key, reference in expected.items():
            assert observation[key].shape == reference.shape
            assert observation[key].dtype == reference.dtype
    # Pixel values vary across the cycle so no constant-folding cache can
    # flatter the measurement.
    assert observations[0]["cam1"].flat[0] != observations[1]["cam1"].flat[0]
    with pytest.raises(ValueError):
        bench.build_synthetic_observations(0)


def test_module_import_does_not_import_jax():
    # Sampled at this module's import time, so a later test that pulls jax in
    # cannot mask the regression.
    assert not _JAX_PRESENT_AFTER_IMPORT
    assert "jax" not in sys.modules


def test_fresh_interpreter_import_does_not_import_jax():
    """Strongest form: a cold process must not touch jax on import.

    The accelerator env prologue is only effective if the first ``import jax``
    happens after ``configure_env``; an import pulled in at module scope by any
    transitive dependency would defeat it silently.
    """

    program = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('b', r'{_SCRIPT}')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.build_synthetic_observations(2)\n"
        "print('JAX_IMPORTED' if 'jax' in sys.modules else 'NO_JAX')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(_SERL_UR_INFRA),
            str(_REPO_ROOT / "third_party" / "hil-serl" / "serl_launcher"),
            env.get("PYTHONPATH", ""),
        ]
    ).strip(os.pathsep)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "NO_JAX"


# --------------------------------------------------------------------------- #
# Opt-in: the real BC loader and three real inferences                         #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.environ.get(_REAL_BENCH_ENV) != "1",
    reason=f"set {_REAL_BENCH_ENV}=1 to run a real 3-iteration BC bench",
)
def test_actual_bc_bench_three_iterations(tmp_path):
    pytest.importorskip("jax", reason="real bench needs a jax-capable venv")
    artifact_dir = Path(bench._DEFAULT_BC_ARTIFACT_DIR)
    if not artifact_dir.is_dir():
        pytest.skip(f"BC artifact not fetched: {artifact_dir}")

    out_dir = tmp_path / "bench"
    # --device cpu is honoured only if jax was not already imported by this
    # process; the hilserl venv's jax is CPU-only either way, so the platform
    # recorded in the result is the one that actually ran.
    exit_code = bench.main(
        [
            "--policy",
            "bc",
            "--iters",
            "3",
            "--warmup",
            "1",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    records = bench.read_results_jsonl(out_dir / bench.RESULTS_FILENAME)
    # ONE row: BC's served sampler ignores the deterministic flag, so a second
    # mode would be this same trace timed twice.
    assert [record["mode"] for record in records] == [bench.BC_MODE]
    for record in records:
        for field in bench.RESULT_FIELDS:
            assert field in record
        assert record["policy"] == "bc"
        assert record["iters"] == 3
        assert record["warmup"] == 1
        assert record["mean_ms"] > 0.0
        assert record["min_ms"] <= record["p50_ms"] <= record["max_ms"]
        assert len(record["artifact_sha"]) == 64
    assert (out_dir / bench.REPORT_FILENAME).is_file()
