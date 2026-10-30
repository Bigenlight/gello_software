#!/usr/bin/env python3
"""Fail-closed validation for a staged SVF real-robot artifact."""
import argparse, hashlib, json
from pathlib import Path

import numpy as np

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--norm-stats", required=True, type=Path)
    p.add_argument("--server-py", required=True, type=Path)
    p.add_argument("--qflow-dir", required=True, type=Path)
    a = p.parse_args()
    manifest_path = a.run_dir / "deployment_manifest.json"
    source_dir = a.server_py.resolve().parent
    native_server = source_dir / "svf_server.py"
    action_queue = source_dir / "action_queue.py"
    episode_logger = source_dir / "episode_logger.py"
    encoder_bridge = source_dir / "real_pipeline/common_encoder_svf.py"
    export_meta = source_dir / "export_serve_meta.py"
    extract_features = source_dir / "extract_features.py"
    preprocess = source_dir / "preprocess.py"
    required = [a.run_dir / "flags.json", a.run_dir / f"params_{a.step}.pkl",
                a.norm_stats, a.server_py, a.qflow_dir / "agents/qflow_rc.py", manifest_path,
                native_server, action_queue, episode_logger, encoder_bridge, export_meta,
                extract_features, preprocess]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"SVF preflight refused missing file: {path}")
    m = json.loads(manifest_path.read_text())
    if m.get("schema") != "svf-real-deployment/v2" or m.get("status") != "ready":
        raise SystemExit("SVF preflight refused invalid deployment manifest")
    for label, path in (("flags", required[0]), ("checkpoint", required[1]),
                        ("norm_stats", a.norm_stats), ("server", a.server_py),
                        ("qflow_rc", required[4]), ("native_server", native_server),
                        ("action_queue", action_queue), ("episode_logger", episode_logger),
                        ("encoder_bridge", encoder_bridge), ("export_serve_meta", export_meta),
                        ("extract_features", extract_features), ("preprocess", preprocess)):
        want = (m.get("sha256") or {}).get(label)
        got = sha(path)
        if not want or got != want:
            raise SystemExit(f"SVF preflight refused {label} sha mismatch: {got}")
    flags = json.loads(required[0].read_text())
    agent = flags.get("agent") or {}
    if agent.get("agent_name") != "qflow_rc" or agent.get("deploy_mode") != "actor" or agent.get("use_critic_obs") is not False:
        raise SystemExit("SVF preflight refused non-actor/asymmetric/unknown agent")
    norm = json.loads(a.norm_stats.read_text())
    expected_scalars = {
        "D_a": 2055, "action_dim": 56, "D_f": 1024,
        "D_f_cam1": 1024, "D_f_cam2": 1024,
        "H": 8, "stride": 3, "chunk_frames": 24,
    }
    for key, expected in expected_scalars.items():
        if int(norm.get(key, -1)) != expected:
            raise SystemExit(
                f"SVF preflight refused norm {key}={norm.get(key)!r}; expected {expected}"
            )
    if norm.get("schema") != "norm_stats/v1" or norm.get("variant") not in {"r18_common_carrot", "r18_ss"}:
        raise SystemExit("SVF preflight refused unexpected norm schema/variant")
    if norm.get("obs_layout") != "[cam1(D_f) | cam2(D_f) | proprio(7)]":
        raise SystemExit("SVF preflight refused unexpected actor observation layout")
    for label, values, expected in (
        ("proprio.mean", (norm.get("proprio") or {}).get("mean"), 7),
        ("proprio.std", (norm.get("proprio") or {}).get("std"), 7),
        ("action.lo", (norm.get("action") or {}).get("lo"), 7),
        ("action.hi", (norm.get("action") or {}).get("hi"), 7),
        ("feature.cam1.mean", ((norm.get("feature") or {}).get("cam1") or {}).get("mean"), 1024),
        ("feature.cam1.std", ((norm.get("feature") or {}).get("cam1") or {}).get("std"), 1024),
        ("feature.cam2.mean", ((norm.get("feature") or {}).get("cam2") or {}).get("mean"), 1024),
        ("feature.cam2.std", ((norm.get("feature") or {}).get("cam2") or {}).get("std"), 1024),
    ):
        if not isinstance(values, list) or len(values) != expected:
            raise SystemExit(f"SVF preflight refused {label} shape; expected ({expected},)")
    identity = norm.get("encoder_identity")
    if norm.get("variant") == "r18_common_carrot":
        if not isinstance(identity, dict):
            raise SystemExit("SVF preflight refused missing common encoder identity")
        root = Path(identity.get("root", ""))
        package_manifest_path = root / "artifact_manifest.json"
        if not root.is_absolute() or not package_manifest_path.is_file():
            raise SystemExit("SVF preflight refused missing common encoder package/manifest")
        package_manifest = json.loads(package_manifest_path.read_text())
        if package_manifest.get("schema") != "common-gello-artifacts/v1" or identity.get("manifest") != package_manifest:
            raise SystemExit("SVF preflight refused common encoder package manifest mismatch")
        declared = {row.get("path"): row.get("sha256") for row in package_manifest.get("artifacts", [])}
        identity_sha = identity.get("sha256") or {}
        for rel in ("encoder.pt", "heads/carrot.pt", "norm_stats.npz"):
            path = root / rel
            got = sha(path) if path.is_file() else None
            if not identity_sha.get(rel) or got != identity_sha.get(rel) or declared.get(rel) != got:
                raise SystemExit(f"SVF preflight refused common encoder identity mismatch: {rel}")
    elif identity is not None:
        raise SystemExit("SVF preflight refused unexpected encoder identity on legacy r18_ss norm")
    print(f"SVF preflight v2 OK: run={a.run_dir} step={a.step} variant={norm.get('variant')}")

if __name__ == "__main__": main()
