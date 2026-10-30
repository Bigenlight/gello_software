#!/usr/bin/env python3
"""Fail-closed artifact gate for the common20 H8 IFQL/DSRL pair."""
from __future__ import annotations
import argparse, hashlib, json, pickle
from pathlib import Path
import jax
import numpy as np

DEPLOYMENT = Path("/home/junhyeong/real_common_baselines20_20260919/deployment_manifest.json")
PIN = json.loads(DEPLOYMENT.read_text())
SHA = PIN["sha"]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def digest(tree):
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        a = np.asarray(leaf); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape, dtype=np.int64).tobytes()); h.update(a.tobytes(order="C"))
    return h.hexdigest()

def require(label, path, expected=None):
    path = Path(path)
    if not path.is_file(): raise SystemExit(f"common20 H8 preflight refused missing {label}: {path}")
    pinned = PIN["files"].get(str(path.resolve()))
    if pinned is None: raise SystemExit(f"common20 preflight refused unpinned file: {path}")
    expected = pinned
    actual = sha(path)
    if expected and actual != expected: raise SystemExit(f"common20 H8 preflight refused {label} SHA: {actual}")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--method", choices=("ifql", "dsrl"), required=True)
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--step", type=int, required=True)
    p.add_argument("--export-manifest", type=Path, required=True); p.add_argument("--package", type=Path, required=True)
    p.add_argument("--ifql-source", type=Path, required=True); p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--base-runtime", type=Path, required=True); a = p.parse_args()
    for file_path, file_sha in PIN["files"].items():
        require("deployment dependency", file_path, file_sha)
    expected_step = PIN["steps"][a.method]
    if a.step != expected_step: raise SystemExit(f"common20 H8 preflight refused step {a.step}; expected {expected_step}")
    require("checkpoint", a.run_dir / f"params_{a.step}.pkl", SHA[a.method + "_checkpoint"])
    require("export manifest", a.export_manifest, SHA["export"]); require("IFQL source", a.ifql_source, SHA["ifql_source"])
    require("encoder", a.package / "encoder.pt", SHA["encoder"]); require("Flow head", a.package / "heads/carrot.pt", SHA["head"]); require("package norm", a.package / "norm_stats.npz", SHA["norm"])
    runtime_hashes = {
        a.source_dir / 'ifql_server.py': 'c40896000751ad18f7c7733c9d8cb7df982bdbea91fda39e0a45db730527eceb',
        a.source_dir / 'common_encoder.py': '00256b9e87e640f94bb2fa41eacaf8aded1f07f45c3acf186a738b81edc799f3',
        a.source_dir / 'real_pipeline/common_encoder_svf.py': 'd3389df2a5d626ca4d5033f3785387340e1c15e423c6db6baedf6f2890cf2727',
        a.source_dir / 'action_queue.py': 'b9a3ccdd083d6e4d48ee0ab65c304d506e596a6e25060176e3418e84a869969c',
        a.source_dir / 'episode_logger.py': '4304150122678028b658685e1df11b8cdd3d31b6eba2239283414e7c7a2d80ea',
        a.base_runtime / 'utils/flax_utils.py': '1173f9437c736cff9fbe028df60a4f2ab3ca36554236f27514f98d74878d9a43',
        a.base_runtime / 'utils/networks.py': '63cca589ea79b7fb424eedbc3b4c5ec9e2cb7defcb713aae034700a056ebe573',
    }
    for path, expected_hash in runtime_hashes.items():
        require('runtime file', path, expected_hash)
    flags = json.loads((a.run_dir / "flags.json").read_text()); identity = flags.get("identity") or {}; agent = identity.get("agent") or {}
    require("flags", a.run_dir / "flags.json", SHA[a.method + "_flags"])
    if a.method == "ifql":
        if flags.get("schema") != "common-gello-ifql-run/v1" or agent.get("agent_name") != "ifql": raise SystemExit("common20 H8 preflight refused IFQL flags identity")
        for k,v in {"action_dim":56,"ob_dims":[2055],"freeze_actor":True,"clip_flow_actions":False,"q_agg":"mean","num_qs":2,"flow_steps":10,"num_samples":32,"use_critic_obs":False}.items():
            if agent.get(k) != v: raise SystemExit(f"common20 H8 preflight refused agent.{k}={agent.get(k)!r}")
    else:
        if flags.get("schema") != "common-gello-dsrl-run/v1" or agent.get("agent_name") != "dsrl_na": raise SystemExit("common20 H8 preflight refused DSRL flags identity")
        for k,v in {"action_dim":56,"ob_dims":[2055],"q_agg":"mean","num_qs":2,"flow_steps":10,"use_critic_obs":False,"base_clip_flow_actions":False,"noise_scale":0.75}.items():
            if agent.get(k) != v: raise SystemExit(f"common20 H8 preflight refused DSRL agent.{k}={agent.get(k)!r}")
    export = json.loads(a.export_manifest.read_text())
    if (export.get("schema"), export.get("task"), export.get("condition_dim"), export.get("action_dim")) != ("common-gello-ifql-export/v1", "carrot", 2055, 56): raise SystemExit("common20 H8 preflight refused export identity")
    if export.get("package", {}).get("path") != str(a.package.resolve()): raise SystemExit("common20 H8 preflight refused package path")
    actor_init = identity.get("actor_init") or "/home/junhyeong/ifql_common_lighted_20260919/carrot_data/actor_init.npz"
    require("actor_init", actor_init, SHA["actor_init"])
    if a.method == "dsrl":
        if agent.get('actor_use_critic_obs') is not False:
            raise SystemExit('DSRL actor privileged input refused')
        require('base flags', a.run_dir.parent / 'ifql_seed0/flags.json', SHA['ifql_flags'])
        require('base trainer', a.run_dir.parent / 'source/vision_carrot/train_common_ifql.py', '202f35be8be8bd70416f727b16eb8a1d926a9fec68521939cfb5539fe6ff1007')
        require("DSRL source", a.run_dir.parent / "source/qflow_svf/agents/dsrl_na.py", "15f35cb9812075a181d050b4724241b22774de40c99502bb21d45fa38e91c294")
        require("DSRL trainer", a.run_dir.parent / "source/vision_carrot/train_common_dsrl.py", "98fb9187a527b3c114638cd8cc9e0ddc5c458c4168b337aa9b608dbd112749c3")
        require("base IFQL checkpoint", a.run_dir.parent / "ifql_seed0/params_100000.pkl", "df862a14e93fa68c9dc7883bbcf4604aa7392e7d61c087634cfb873d86b049c6")
    payload = pickle.loads((a.run_dir / f"params_{a.step}.pkl").read_bytes()); trainer = payload.get("trainer") or {}
    expected_schema = "common-gello-ifql-checkpoint/v1" if a.method == "ifql" else "common-gello-dsrl-checkpoint/v1"
    if trainer.get("schema") != expected_schema or trainer.get("step") != a.step: raise SystemExit("common20 H8 preflight refused checkpoint identity")
    if a.method == "ifql":
        actor = payload["agent"]["network"]["params"]["modules_actor_flow"]
        if digest(actor) != SHA["actor"]: raise SystemExit("common20 H8 preflight refused frozen IFQL actor")
        if (trainer.get("compatibility") or {}).get("actor_sha256") != SHA["actor"]: raise SystemExit("common20 H8 preflight refused IFQL actor compatibility")
    else:
        base_actor = payload["agent"]["base"]["params"]["modules_actor_flow"]
        if digest(base_actor) != SHA["actor"]: raise SystemExit("common20 H8 preflight refused frozen DSRL base actor")
    print(f"common20 H8 {a.method} preflight OK: step={a.step} D_a=2055 action=56 actor={SHA['actor'][:12]}")

if __name__ == "__main__": main()

