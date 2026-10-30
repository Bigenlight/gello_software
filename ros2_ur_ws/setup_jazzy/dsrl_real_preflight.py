#!/usr/bin/env python3
"""Fail-closed provenance gate for a staged real-domain DSRL deployment.

This program only validates files and prints a command environment.  It never
imports the policy, binds a socket, sources ROS, or starts a process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
from typing import Any

import yaml


SCHEMA = "dsrl-real-serve/v1"
KNOWN_SIM_ONLY_RUN_NAMES = ("dsrl_v0_s0", "v2ns075_s0", "v3aobs_s0")
EXPECTED_RUNTIME = {
    "action_dim": 7,
    "chunk_dim": 56,
    "chunk_horizon": 8,
    "n_action_steps": 24,
    "port": 5596,
}


class Refused(ValueError):
    """A deployment attestation failed closed."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refused(reason)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Refused(f"{label} is unreadable JSON: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def _resolve_artifact(run_dir: Path, artifacts: dict[str, Any], name: str) -> tuple[Path, str, dict[str, Any]]:
    entry = artifacts.get(name)
    _require(isinstance(entry, dict), f"manifest.artifacts.{name} is required")
    raw_path = entry.get("path")
    expected_hash = entry.get("sha256")
    _require(isinstance(raw_path, str) and raw_path and "\n" not in raw_path,
             f"manifest.artifacts.{name}.path must be a non-empty single-line string")
    _require(isinstance(expected_hash, str) and len(expected_hash) == 64,
             f"manifest.artifacts.{name}.sha256 must be a 64-character SHA256")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    path = path.resolve()
    _require(path.is_file(), f"missing {name}: {path}")
    actual_hash = sha256(path)
    _require(actual_hash == expected_hash.lower(),
             f"{name} SHA256 mismatch: expected {expected_hash.lower()}, got {actual_hash}: {path}")
    return path, actual_hash, entry


def _exact_false(mapping: dict[str, Any], key: str, where: str) -> None:
    _require(mapping.get(key) is False, f"{where}.{key} must be explicitly false")


def _exact_true(mapping: dict[str, Any], key: str, where: str) -> None:
    _require(mapping.get(key) is True, f"{where}.{key} must be explicitly true")


def _finite_pose(value: Any, where: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 6, f"{where} must contain exactly 6 joint values")
    try:
        pose = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise Refused(f"{where} must contain numeric joint values") from exc
    _require(all(math.isfinite(item) for item in pose), f"{where} must contain only finite joint values")
    return pose


def _checkpoint_step(path: Path, entry: dict[str, Any]) -> int:
    step = entry.get("step")
    _require(type(step) is int and step > 0, "manifest.artifacts.checkpoint.step must be a positive integer")
    _require(path.name == f"params_{step}.pkl",
             f"checkpoint filename does not match step {step}: {path.name}")
    return step


def validate(run_dir: Path, manifest_path: Path | None = None) -> dict[str, str]:
    run_dir = run_dir.expanduser().resolve()
    _require(run_dir.is_dir(), f"DSRL_RUN_DIR is not a directory: {run_dir}")
    manifest_path = (manifest_path or run_dir / "real_serve_meta.json").expanduser().resolve()
    _require(manifest_path == run_dir / "real_serve_meta.json" and manifest_path.is_file(),
             f"required manifest must be DSRL_RUN_DIR/real_serve_meta.json: {manifest_path}")
    manifest = _read_json(manifest_path, "real_serve_meta.json")

    _require(manifest.get("schema") == SCHEMA, f"manifest.schema must be {SCHEMA!r}")
    _require(manifest.get("placeholder") is False, "placeholder manifest is not deployable")
    _require(manifest.get("status") == "staged", "manifest.status must be 'staged'")
    _require(manifest.get("policy_family") == "dsrl", "manifest.policy_family must be 'dsrl'")
    _require(manifest.get("domain") == "real", "manifest.domain must be 'real'; sim artifacts are refused")
    _exact_true(manifest, "nonprivileged_actor", "manifest")
    _exact_false(manifest, "dry_run", "manifest")
    _exact_false(manifest, "use_critic_obs", "manifest")
    _exact_false(manifest, "actor_use_critic_obs", "manifest")
    task = manifest.get("task")
    _require(isinstance(task, str) and task.strip() == task and task,
             "manifest.task must be a non-empty normalized task id")
    sampler = manifest.get("sampler")
    _require(sampler in ("base", "dsrl_det", "dsrl_sample"),
             "manifest.sampler must be base|dsrl_det|dsrl_sample")

    run_name = run_dir.name.lower()
    for known in KNOWN_SIM_ONLY_RUN_NAMES:
        _require(known not in run_name,
                 f"known public sim-only negative-result run is never real-compatible: {known}")

    runtime = manifest.get("runtime")
    _require(isinstance(runtime, dict), "manifest.runtime is required")
    for key, expected in EXPECTED_RUNTIME.items():
        _require(runtime.get(key) == expected,
                 f"manifest.runtime.{key} must be {expected}, got {runtime.get(key)!r}")

    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, dict), "manifest.artifacts is required")
    flags_path, flags_hash, _ = _resolve_artifact(run_dir, artifacts, "flags")
    checkpoint, checkpoint_hash, checkpoint_entry = _resolve_artifact(run_dir, artifacts, "checkpoint")
    base_checkpoint, base_hash, base_entry = _resolve_artifact(run_dir, artifacts, "base_checkpoint")
    norm_stats, norm_hash, _ = _resolve_artifact(run_dir, artifacts, "norm_stats")
    deploy_yaml, deploy_hash, deploy_entry = _resolve_artifact(run_dir, artifacts, "deploy_yaml")
    step = _checkpoint_step(checkpoint, checkpoint_entry)
    base_step = _checkpoint_step(base_checkpoint, base_entry)
    _require(checkpoint.parent == run_dir, "DSRL checkpoint must be directly inside DSRL_RUN_DIR")

    bindings = manifest.get("bindings")
    _require(isinstance(bindings, dict), "manifest.bindings is required")
    for key, actual in (
        ("checkpoint_sha256", checkpoint_hash),
        ("base_checkpoint_sha256", base_hash),
        ("norm_stats_sha256", norm_hash),
        ("flags_sha256", flags_hash),
    ):
        _require(bindings.get(key) == actual,
                 f"manifest.bindings.{key} must exactly match the resolved artifact hash")

    flags = _read_json(flags_path, "flags.json")
    _require(flags_path == run_dir / "flags.json", "manifest flags path must resolve to DSRL_RUN_DIR/flags.json")
    _require(flags.get("policy_family") == "dsrl", "flags.policy_family must be explicitly 'dsrl'")
    _require(flags.get("domain") == "real", "flags.domain must be explicitly 'real'; sim/unknown is refused")
    _exact_true(flags, "nonprivileged_actor", "flags")
    _exact_false(flags, "dry_run", "flags")
    _exact_false(flags, "use_critic_obs", "flags")
    _exact_false(flags, "actor_use_critic_obs", "flags")
    _require(flags.get("agent_module") == "dsrl_na", "flags.agent_module must be 'dsrl_na'")
    agent = flags.get("agent")
    _require(isinstance(agent, dict), "flags.agent is required")
    _require(agent.get("agent_name") == "dsrl_na", "flags.agent.agent_name must be 'dsrl_na'")
    _exact_true(agent, "nonprivileged_actor", "flags.agent")
    _exact_false(agent, "use_critic_obs", "flags.agent")
    _exact_false(agent, "actor_use_critic_obs", "flags.agent")
    _require(agent.get("action_dim") == EXPECTED_RUNTIME["chunk_dim"],
             "flags.agent.action_dim must be 56")
    dims = flags.get("dims")
    _require(isinstance(dims, dict), "flags.dims is required")
    _require(dims.get("z_dim") == EXPECTED_RUNTIME["chunk_dim"], "flags.dims.z_dim must be 56")
    _require(type(dims.get("obs_dim")) is int and dims["obs_dim"] > 7, "flags.dims.obs_dim must be an integer > 7")
    _require(type(dims.get("cobs_dim")) is int and dims["cobs_dim"] >= dims["obs_dim"],
             "flags.dims.cobs_dim must be an integer >= obs_dim")

    flagged_base = flags.get("base_run_dir")
    _require(isinstance(flagged_base, str) and Path(flagged_base).expanduser().resolve() == base_checkpoint.parent,
             "flags.base_run_dir must resolve to the manifest base checkpoint directory")
    _require(flags.get("base_epoch") == base_step,
             "flags.base_epoch must match manifest.artifacts.base_checkpoint.step")
    flagged_norm = flags.get("norm_stats")
    _require(isinstance(flagged_norm, str) and Path(flagged_norm).expanduser().resolve() == norm_stats,
             "flags.norm_stats must resolve to the manifest norm_stats artifact")
    _require(flags.get("real_serve_meta") in ("real_serve_meta.json", str(manifest_path)),
             "flags.real_serve_meta must explicitly point to this manifest")

    norm = _read_json(norm_stats, "norm_stats")
    _require(norm.get("action_dim") == EXPECTED_RUNTIME["chunk_dim"], "norm_stats.action_dim must be 56")
    _require(norm.get("H") == EXPECTED_RUNTIME["chunk_horizon"], "norm_stats.H must be 8")
    _require(norm.get("domain") == "real", "norm_stats.domain must be explicitly 'real'")
    _require(norm.get("task") == task, "norm_stats.task must match manifest.task")

    _require(deploy_entry.get("task") == task,
             "manifest.artifacts.deploy_yaml.task must match manifest.task")
    try:
        deploy = yaml.safe_load(deploy_yaml.read_text())
        params = deploy["policy_leader_node"]["ros__parameters"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise Refused(f"deploy YAML lacks policy_leader_node.ros__parameters: {deploy_yaml}") from exc
    _require(params.get("act_port") == EXPECTED_RUNTIME["port"], "deploy YAML act_port must be 5596")
    _require(params.get("auto_start_on_stream") is False,
             "deploy YAML auto_start_on_stream must be false")
    yaml_pose = _finite_pose(params.get("start_pose"), "deploy YAML start_pose")
    manifest_pose = _finite_pose(manifest.get("start_pose"), "manifest.start_pose")
    _require(yaml_pose == manifest_pose, "manifest.start_pose must exactly match deploy YAML start_pose")

    return {
        "DSRL_MANIFEST_PATH": str(manifest_path),
        "DSRL_RUN_DIR": str(run_dir),
        "DSRL_STEP": str(step),
        "DSRL_CHECKPOINT": str(checkpoint),
        "DSRL_BASE_CHECKPOINT": str(base_checkpoint),
        "DSRL_NORM_STATS": str(norm_stats),
        "DSRL_PARAMS_FILE": str(deploy_yaml),
        "DSRL_TASK": task,
        "DSRL_SAMPLER": sampler,
        "DSRL_CHECKPOINT_SHA256": checkpoint_hash,
        "DSRL_BASE_CHECKPOINT_SHA256": base_hash,
        "DSRL_NORM_STATS_SHA256": norm_hash,
        "DSRL_FLAGS_SHA256": flags_hash,
        "DSRL_DEPLOY_YAML_SHA256": deploy_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: <run-dir>/real_serve_meta.json")
    parser.add_argument("--emit-shell", action="store_true",
                        help="emit shell-quoted assignments after successful validation")
    args = parser.parse_args()
    try:
        values = validate(args.run_dir, args.manifest)
    except Refused as exc:
        print(f"DSRL real preflight REFUSED: {exc}")
        return 2
    if args.emit_shell:
        for key, value in values.items():
            print(f"{key}={shlex.quote(value)}")
    else:
        print(
            "DSRL real preflight OK (artifact staging only; performance unvalidated): "
            f"task={values['DSRL_TASK']} checkpoint={values['DSRL_CHECKPOINT']} "
            f"sha256={values['DSRL_CHECKPOINT_SHA256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
