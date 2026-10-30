#!/usr/bin/env python3
"""Fail closed before serving the lighted-carrot common Base Flow / IFQL."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import jax
import numpy as np

EXPECTED = {
    "checkpoint": "ab0903ef802b75a9d610b210d50e01be6f9633aca9c55b8e335e373ee6778119",
    "actor": "7c4dfbe5c2a070e5ca0f232512b11e197a6b01601c83a9bd72e65be208290c10",
    "actor_init": "4be0c61f642b72290d90cc019cd7b21a588bc2307a47788417e5e1bc6d4543fe",
    "export_manifest": "4253c4bcd77d1b151a342a03c28acd0e27ed74eb8f47d85d82270cddba63660e",
    "ifql_source": "58b77159be7a520ab00366078f3cb5cadffc2437efc98f0fc32d1ade975910c4",
    "encoder": "f0ed030d5b12e7a61d5f4050aa34bdacc7411b96f8ff0ad1234c844c18f76920",
    "head": "14018ab1b1124f5d9816906e2cda3c4f016e40a989a360471bd871467d4201ee",
    "package_norm": "dcb56c7e75a5a65ec3c2709517b0debe07735c9d84235217952485cb119a4b67",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_digest(actor) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(actor):
        value = np.asarray(leaf)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def require_sha(label: str, path: Path, expected: str) -> None:
    if not path.is_file():
        raise SystemExit(f"common IFQL preflight refused missing {label}: {path}")
    actual = sha(path)
    if actual != expected:
        raise SystemExit(f"common IFQL preflight refused {label} SHA: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--ifql-source", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base-runtime", type=Path, required=True)
    args = parser.parse_args()
    if args.step != 50000:
        raise SystemExit("common IFQL preflight refused non-candidate step (expected 50000)")

    checkpoint = args.run_dir / f"params_{args.step}.pkl"
    require_sha("checkpoint", checkpoint, EXPECTED["checkpoint"])
    require_sha("export manifest", args.export_manifest, EXPECTED["export_manifest"])
    require_sha("IFQL source", args.ifql_source, EXPECTED["ifql_source"])
    require_sha("encoder", args.package / "encoder.pt", EXPECTED["encoder"])
    require_sha("Flow head", args.package / "heads/carrot.pt", EXPECTED["head"])
    require_sha("package norm", args.package / "norm_stats.npz", EXPECTED["package_norm"])
    for path in (args.source_dir / "ifql_server.py", args.source_dir / "common_encoder.py",
                 args.source_dir / "real_pipeline/common_encoder_svf.py",
                 args.base_runtime / "utils/flax_utils.py", args.base_runtime / "utils/networks.py"):
        if not path.is_file():
            raise SystemExit(f"common IFQL preflight refused missing runtime file: {path}")

    flags = json.loads((args.run_dir / "flags.json").read_text())
    identity = flags.get("identity") or {}
    agent = identity.get("agent") or {}
    expected_agent = {
        "agent_name": "ifql", "action_dim": 56, "ob_dims": [2055],
        "freeze_actor": True, "clip_flow_actions": False,
        "use_critic_obs": False, "q_agg": "mean", "num_qs": 2,
        "flow_steps": 10, "num_samples": 32,
    }
    if flags.get("schema") != "common-gello-ifql-run/v1":
        raise SystemExit("common IFQL preflight refused flags schema")
    for key, expected in expected_agent.items():
        if agent.get(key) != expected:
            raise SystemExit(f"common IFQL preflight refused agent.{key}={agent.get(key)!r}")

    export = json.loads(args.export_manifest.read_text())
    if (export.get("schema"), export.get("task"), export.get("condition_dim"), export.get("action_dim")) != \
            ("common-gello-ifql-export/v1", "carrot", 2055, 56):
        raise SystemExit("common IFQL preflight refused export schema/task/dimensions")
    if export.get("package", {}).get("path") != str(args.package.resolve()):
        raise SystemExit("common IFQL preflight refused encoder package path")
    if export.get("action_normalization") != \
            "(raw_chunk - package action_mean[56]) / package action_std[56]; no clipping; no refit":
        raise SystemExit("common IFQL preflight refused action normalization contract")
    norm = export.get("normalization") or {}
    for key, shape in (("proprio_mean", (7,)), ("proprio_std", (7,)),
                       ("action_mean", (56,)), ("action_std", (56,))):
        values = np.asarray(norm.get(key), dtype=np.float32)
        if values.shape != shape or not np.isfinite(values).all() or (key.endswith("std") and np.any(values <= 0)):
            raise SystemExit(f"common IFQL preflight refused normalization {key}")
    actor_init = Path(identity.get("actor_init", ""))
    require_sha("actor_init", actor_init, EXPECTED["actor_init"])

    with checkpoint.open("rb") as stream:
        payload = pickle.load(stream)
    trainer = payload.get("trainer") or {}
    if trainer.get("schema") != "common-gello-ifql-checkpoint/v1" or trainer.get("step") != args.step:
        raise SystemExit("common IFQL preflight refused checkpoint trainer identity")
    actor = (((payload.get("agent") or {}).get("network") or {}).get("params") or {}).get("modules_actor_flow")
    if actor is None or actor_digest(actor) != EXPECTED["actor"]:
        raise SystemExit("common IFQL preflight refused frozen actor digest")
    compatibility = trainer.get("compatibility") or {}
    if compatibility.get("actor_sha256") != EXPECTED["actor"]:
        raise SystemExit("common IFQL preflight refused checkpoint actor compatibility")
    print(
        f"common IFQL preflight OK: step={args.step} D_a=2055 action=56 "
        f"actor={EXPECTED['actor'][:12]} package={args.package}"
    )


if __name__ == "__main__":
    main()
