"""Fail-closed tests for the staged-only DSRL real launcher.

These tests manufacture bytes-only artifacts.  They never source ROS, bind ZMQ,
or start a robot process; the launcher is exercised only with DSRL_DRY_RUN=1.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


WORKSPACE = Path(__file__).resolve().parents[3]
PREFLIGHT = WORKSPACE / "setup_jazzy" / "dsrl_real_preflight.py"
LAUNCHER = WORKSPACE / "run_ur7e_dsrl_real.sh"
GENERIC_LAUNCHER = WORKSPACE / "run_ur7e_ifql_real.sh"
SERVER = WORKSPACE.parents[1] / "model_code" / "vision_carrot" / "dsrl" / "dsrl_server.py"


def _module():
    spec = importlib.util.spec_from_file_location("dsrl_real_preflight_under_test", PREFLIGHT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, step: int | None = None, **extra: object) -> dict[str, object]:
    out: dict[str, object] = {"path": str(path), "sha256": _sha(path), **extra}
    if step is not None:
        out["step"] = step
    return out


def _staged_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "dsrl_real"
    base = tmp_path / "ifql_base"
    run.mkdir()
    base.mkdir()
    checkpoint = run / "params_11.pkl"
    checkpoint.write_bytes(b"dsrl-real-weights")
    base_checkpoint = base / "params_22.pkl"
    base_checkpoint.write_bytes(b"base-ifql-weights")
    norm = run / "norm_stats_real.json"
    norm.write_text(json.dumps({"action_dim": 56, "H": 8, "domain": "real", "task": "carrot"}))
    deploy = run / "ifql_deploy.yaml"
    deploy.write_text(
        "policy_leader_node:\n  ros__parameters:\n    act_port: 5596\n"
        "    auto_start_on_stream: false\n    start_pose: [0, 0, 0, 0, 0, 0]\n"
    )
    flags = {
        "policy_family": "dsrl", "domain": "real", "nonprivileged_actor": True,
        "dry_run": False, "use_critic_obs": False, "actor_use_critic_obs": False,
        "agent_module": "dsrl_na", "base_run_dir": str(base), "base_epoch": 22,
        "norm_stats": str(norm), "real_serve_meta": "real_serve_meta.json",
        "agent": {"agent_name": "dsrl_na", "nonprivileged_actor": True,
                  "use_critic_obs": False, "actor_use_critic_obs": False, "action_dim": 56},
        "dims": {"z_dim": 56, "obs_dim": 2055, "cobs_dim": 2055},
    }
    flags_path = run / "flags.json"
    flags_path.write_text(json.dumps(flags))
    artifacts = {
        "flags": _artifact(flags_path), "checkpoint": _artifact(checkpoint, 11),
        "base_checkpoint": _artifact(base_checkpoint, 22), "norm_stats": _artifact(norm),
        "deploy_yaml": _artifact(deploy, task="carrot"),
    }
    manifest = {
        "schema": "dsrl-real-serve/v1", "status": "staged", "placeholder": False,
        "policy_family": "dsrl", "domain": "real", "nonprivileged_actor": True,
        "dry_run": False, "use_critic_obs": False, "actor_use_critic_obs": False,
        "task": "carrot", "sampler": "dsrl_det",
        "runtime": {"action_dim": 7, "chunk_dim": 56, "chunk_horizon": 8, "n_action_steps": 24, "port": 5596},
        "artifacts": artifacts, "start_pose": [0, 0, 0, 0, 0, 0],
        "bindings": {"checkpoint_sha256": artifacts["checkpoint"]["sha256"],
                     "base_checkpoint_sha256": artifacts["base_checkpoint"]["sha256"],
                     "norm_stats_sha256": artifacts["norm_stats"]["sha256"],
                     "flags_sha256": artifacts["flags"]["sha256"]},
    }
    manifest_path = run / "real_serve_meta.json"
    manifest_path.write_text(json.dumps(manifest))
    return run, manifest_path


def test_preflight_accepts_only_fully_pinned_nonprivileged_real_run(tmp_path: Path) -> None:
    module = _module()
    run, manifest = _staged_run(tmp_path)
    values = module.validate(run, manifest)
    assert values["DSRL_STEP"] == "11"
    assert values["DSRL_CHECKPOINT_SHA256"] == _sha(run / "params_11.pkl")


@pytest.mark.parametrize(
    ("path", "value", "needle"),
    [
        (("domain",), "sim", "manifest.domain"),
        (("nonprivileged_actor",), False, "nonprivileged_actor"),
        (("bindings", "norm_stats_sha256"), "0" * 64, "bindings.norm_stats_sha256"),
    ],
)
def test_preflight_refuses_domain_privilege_or_binding_tamper(
    tmp_path: Path, path: tuple[str, ...], value: object, needle: str
) -> None:
    module = _module()
    run, manifest_path = _staged_run(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(module.Refused, match=needle):
        module.validate(run, manifest_path)


def test_dry_run_stops_before_ros_or_server_bind(tmp_path: Path) -> None:
    run, _ = _staged_run(tmp_path)
    completed = subprocess.run(
        ["bash", str(LAUNCHER)], text=True, capture_output=True, check=False,
        env={"PATH": "/usr/bin:/bin", "DSRL_DRY_RUN": "1", "DSRL_RUN_DIR": str(run),
             "DSRL_SERVER_PY": str(SERVER), "DSRL_PY": sys.executable},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "no ROS, server bind, or robot process will start" in completed.stdout
    assert "staged only (not a released real DSRL)" in completed.stdout


def test_dsrl_health_check_does_not_skip_sleep_or_port_probe() -> None:
    """Regression: an unconditional continue made the advertised 120 s wait busy-spin in ~2 s."""
    text = GENERIC_LAUNCHER.read_text()
    start = text.index('if [ -z "${PX_OK}" ]')
    end = text.index('if command -v ss', start)
    dsrl_px_block = text[start:end]
    assert "continue" not in dsrl_px_block
    loop_start = text.index('for i in $(seq 1 "${IFQL_WARMUP_TIMEOUT_S}")')
    loop_end = text.index('if [ -z "${LISTENING}" ]', loop_start)
    assert "sleep 1" in text[loop_start:loop_end]
