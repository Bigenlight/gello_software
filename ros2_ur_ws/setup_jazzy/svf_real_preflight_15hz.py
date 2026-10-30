#!/usr/bin/env python3
"""Fail-closed validation for a staged 15 Hz (frame-stride-2, H8) SVF/BC real-robot artifact.

Copy of svf_real_preflight_h.py plus the 15 Hz contract: norm_stats variant r18_common_carrot_h8_15hz,
policy_hz/frame_stride in norm_stats and manifest, ``execution.command_rate_hz`` == 15, and the ROS
params YAML handed to the leader (--params-yaml) must set policy_leader_node.publish_rate_hz == 15."""
import argparse, hashlib, json
import yaml
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
    p.add_argument("--params-yaml", required=True, type=Path)
    p.add_argument("--sampler", required=True, choices=("actor", "bc"))
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
    execution = m.get("execution") or {}
    H = int(execution.get("horizon", 0))
    if H != 8 or execution.get("stride") != 1 or execution.get("hold") != 0 \
            or execution.get("interpolation") is not False or execution.get("replan_after_requests") != H \
            or execution.get("command_rate_hz") != 15 or execution.get("frame_stride") != 2:
        raise SystemExit(f"SVF preflight refused execution contract {execution!r}")
    if m.get("sampler") != a.sampler:
        raise SystemExit(f"SVF preflight refused sampler {a.sampler!r}; deployment is staged for {m.get('sampler')!r}")
    if int(m.get("step", -1)) != a.step:
        raise SystemExit(f"SVF preflight refused step {a.step}; deployment is staged for {m.get('step')!r}")
    if norm.get("policy_hz") != 15 or norm.get("frame_stride") != 2:
        raise SystemExit(f"SVF preflight refused norm policy_hz/frame_stride {norm.get('policy_hz')!r}/{norm.get('frame_stride')!r}")
    params = yaml.safe_load(a.params_yaml.read_text())
    rate = ((params.get("policy_leader_node") or {}).get("ros__parameters") or {}).get("publish_rate_hz")
    if rate != 15.0:
        raise SystemExit(f"SVF preflight refused params yaml {a.params_yaml}: publish_rate_hz={rate!r}, expected 15.0")
    want_yaml = (m.get("extra_sha256") or {}).get(str(a.params_yaml.resolve()))
    if not want_yaml or sha(a.params_yaml) != want_yaml:
        raise SystemExit(f"SVF preflight refused params yaml sha mismatch: {a.params_yaml}")
    for path, want in (m.get("extra_sha256") or {}).items():
        if not Path(path).is_file() or sha(path) != want:
            raise SystemExit(f"SVF preflight refused extra dependency sha mismatch: {path}")
    if "/contiguous_h_queue.py" not in " ".join(m.get("extra_sha256") or {}):
        raise SystemExit("SVF preflight refused: deployment does not pin contiguous_h_queue.py")
    expected_scalars = {
        "D_a": 2055, "action_dim": 7 * H, "D_f": 1024,
        "D_f_cam1": 1024, "D_f_cam2": 1024,
        "H": H, "stride": 1, "chunk_frames": H,
    }
    for key, expected in expected_scalars.items():
        if int(norm.get(key, -1)) != expected:
            raise SystemExit(
                f"SVF preflight refused norm {key}={norm.get(key)!r}; expected {expected}"
            )
    # Horizon exports name their variant r18_common_carrot_h<H>; the encoder package is the H8 common carrot one.
    common_variant = norm.get("variant") == "r18_common_carrot_h8_15hz"
    if norm.get("schema") != "norm_stats/v1" or not common_variant:
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
    if common_variant:
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
    print(f"SVF preflight 15Hz OK: run={a.run_dir} step={a.step} sampler={a.sampler} H={H} 15 Hz yaml={a.params_yaml}")

if __name__ == "__main__": main()
