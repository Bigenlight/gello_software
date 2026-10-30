#!/usr/bin/env python3
"""Fail-closed provenance check for a staged real-domain SVF deployment."""
import argparse
import hashlib
import json
from pathlib import Path

REAL_DATA_SHA256 = "0a2b2872c1f5fac1f7e5fa02e7e3e7360ecfd0e423402574f3c05e852eadfd4e"
REAL_NORM_SHA256 = "08492698000e1bd42475153003a2f3a721c5411cb302b06bea114c40edeafedf"
TRAINING_RUNTIME = {
    "main.py": "3d1b54d5c980be10709627fa3566d9707fe500358a8aadd108b2f89cc43af14e",
    "qflow_rc.py": "7c94912b23d86bc435d6885fa01ec4b40172d5c842f7e6b690dffd80d6497006",
    "ifql_server.py": "489767f24cbea9f304c870937d6cff30c62cd95cc135900162157565aa8b79b8",
}
DEPLOY_RUNTIME = {
    "main.py": TRAINING_RUNTIME["main.py"],
    "qflow_rc.py": TRAINING_RUNTIME["qflow_rc.py"],
    "ifql_server.py": "1e2e3d826fcc5bb29b7bfe147dc3f7ee92b51d7e1ce36451a55c40db85df7222",
    "episode_logger.py": "6fb5372d988e9754e92b9839e5b371d5af050bc8b52eefa451371de8ee9f8aa8",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--norm-stats", type=Path, required=True)
    p.add_argument("--real-data", type=Path, required=True)
    p.add_argument("--qflow-dir", type=Path, required=True)
    p.add_argument("--server-py", type=Path, required=True)
    args = p.parse_args()

    run = args.run_dir.resolve()
    flags_path = run / "flags.json"
    manifest_path = run / "training_manifest.json"
    ckpt = run / f"params_{args.step}.pkl"
    for path in (flags_path, manifest_path, ckpt, args.norm_stats, args.real_data):
        if not path.is_file():
            raise SystemExit(f"SVF preflight refused: missing required file: {path}")
    if args.step <= 0:
        raise SystemExit("SVF preflight refused: --step must be positive")

    flags = json.loads(flags_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "svf-real-training/v1"
        or manifest.get("domain") != "real"
        or manifest.get("status") != "completed"
    ):
        raise SystemExit("SVF preflight refused: manifest must be completed svf-real-training/v1 for domain=real")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SystemExit("SVF preflight refused: training_manifest.files is absent")
    for name, path in (("flags.json", flags_path), (f"params_{args.step}.pkl", ckpt)):
        entry = files.get(name)
        if not isinstance(entry, dict) or entry.get("sha256") != sha256(path):
            raise SystemExit(f"SVF preflight refused: {name} does not match its training-manifest sha256")
    for basename, expected in TRAINING_RUNTIME.items():
        matches = [entry for entry in files.values() if isinstance(entry, dict) and entry.get("path", "").endswith(basename)]
        if not any(entry.get("sha256") == expected for entry in matches):
            raise SystemExit(f"SVF preflight refused: manifest lacks pinned {basename} runtime hash")
    actual_runtime = {
        "main.py": args.qflow_dir / "main.py",
        "qflow_rc.py": args.qflow_dir / "agents" / "qflow_rc.py",
        "ifql_server.py": args.server_py,
        "episode_logger.py": args.server_py.parent / "episode_logger.py",
    }
    for basename, path in actual_runtime.items():
        if not path.is_file() or sha256(path) != DEPLOY_RUNTIME[basename]:
            raise SystemExit(f"SVF preflight refused: selected {basename} is missing or hash-mismatched: {path}")
    agent = flags.get("agent")
    if not isinstance(agent, dict):
        raise SystemExit("SVF preflight refused: flags.agent is absent or not an object")
    required = {
        "agent_name": "qflow_rc",
        "use_critic_obs": False,
        "deploy_mode": "actor",
        "inner_target": "soft_value",
    }
    for key, expected in required.items():
        if agent.get(key) != expected:
            raise SystemExit(
                f"SVF preflight refused: flags.agent.{key}={agent.get(key)!r}, expected {expected!r}"
            )

    flagged_data = flags.get("env_name")
    if not isinstance(flagged_data, str) or not flagged_data:
        raise SystemExit("SVF preflight refused: flags.env_name is absent")
    if Path(flagged_data).resolve() != args.real_data.resolve():
        raise SystemExit(
            "SVF preflight refused: SVF_REAL_DATA does not equal flags.env_name "
            f"({args.real_data} != {flagged_data})"
        )
    if sha256(args.real_data) != REAL_DATA_SHA256:
        raise SystemExit("SVF preflight refused: real dataset sha256 does not match the pinned artifact")
    if sha256(args.norm_stats) != REAL_NORM_SHA256:
        raise SystemExit("SVF preflight refused: norm_stats sha256 does not match the pinned artifact")

    print(
        "SVF real preflight OK: "
        f"run={run} step={args.step} checkpoint_sha256={sha256(ckpt)} "
        f"data_sha256={REAL_DATA_SHA256} norm_sha256={REAL_NORM_SHA256}"
    )


if __name__ == "__main__":
    main()
