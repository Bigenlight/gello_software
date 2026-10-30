#!/usr/bin/env python3
"""Fail-closed artifact gate for the retrained common IFQL/DSRL pair."""
from __future__ import annotations
import argparse, hashlib, json, pickle
from pathlib import Path
import jax
import numpy as np

SHA = {
    "ifql_flags": "34867b317111e4d54da0ceee5cbdbf692c4ce76c72d17da21f1bc36fe6734ddd",
    "dsrl_flags": "8a1c8f4dcca77e022d69207173428a41da4bd7a8ea67a074449ccb97dbc57e23",
    "ifql_checkpoint": "ed261eb508df94570c630b6bbbd1a082030ba2f77281e8cb1a4b977150da1fa5",
    "dsrl_checkpoint": "fedf3bf6bce822cc9c9c86e0410ecae7afddc7b760dcbac1fedf938767134df1",
    "actor": "7c4dfbe5c2a070e5ca0f232512b11e197a6b01601c83a9bd72e65be208290c10",
    "actor_init": "4be0c61f642b72290d90cc019cd7b21a588bc2307a47788417e5e1bc6d4543fe",
    "export": "4253c4bcd77d1b151a342a03c28acd0e27ed74eb8f47d85d82270cddba63660e",
    "ifql_source": "58b77159be7a520ab00366078f3cb5cadffc2437efc98f0fc32d1ade975910c4",
    "encoder": "f0ed030d5b12e7a61d5f4050aa34bdacc7411b96f8ff0ad1234c844c18f76920",
    "head": "14018ab1b1124f5d9816906e2cda3c4f016e40a989a360471bd871467d4201ee",
    "norm": "dcb56c7e75a5a65ec3c2709517b0debe07735c9d84235217952485cb119a4b67",
}

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
    if not path.is_file(): raise SystemExit(f"retrained common preflight refused missing {label}: {path}")
    actual = sha(path)
    if expected and actual != expected: raise SystemExit(f"retrained common preflight refused {label} SHA: {actual}")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--method", choices=("ifql", "dsrl"), required=True)
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--step", type=int, required=True)
    p.add_argument("--export-manifest", type=Path, required=True); p.add_argument("--package", type=Path, required=True)
    p.add_argument("--ifql-source", type=Path, required=True); p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--base-runtime", type=Path, required=True); a = p.parse_args()
    expected_step = 40000 if a.method == "ifql" else 100000
    if a.step != expected_step: raise SystemExit(f"retrained common preflight refused step {a.step}; expected {expected_step}")
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
        if flags.get("schema") != "common-gello-ifql-run/v1" or agent.get("agent_name") != "ifql": raise SystemExit("retrained common preflight refused IFQL flags identity")
        for k,v in {"action_dim":56,"ob_dims":[2055],"freeze_actor":True,"clip_flow_actions":False,"q_agg":"mean","num_qs":2,"flow_steps":10,"num_samples":32,"use_critic_obs":False}.items():
            if agent.get(k) != v: raise SystemExit(f"retrained common preflight refused agent.{k}={agent.get(k)!r}")
    else:
        if flags.get("schema") != "common-gello-dsrl-run/v1" or agent.get("agent_name") != "dsrl_na": raise SystemExit("retrained common preflight refused DSRL flags identity")
        for k,v in {"action_dim":56,"ob_dims":[2055],"q_agg":"mean","num_qs":2,"flow_steps":10,"use_critic_obs":False,"base_clip_flow_actions":False,"noise_scale":0.75}.items():
            if agent.get(k) != v: raise SystemExit(f"retrained common preflight refused DSRL agent.{k}={agent.get(k)!r}")
    export = json.loads(a.export_manifest.read_text())
    if (export.get("schema"), export.get("task"), export.get("condition_dim"), export.get("action_dim")) != ("common-gello-ifql-export/v1", "carrot", 2055, 56): raise SystemExit("retrained common preflight refused export identity")
    if export.get("package", {}).get("path") != str(a.package.resolve()): raise SystemExit("retrained common preflight refused package path")
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
    if trainer.get("schema") != expected_schema or trainer.get("step") != a.step: raise SystemExit("retrained common preflight refused checkpoint identity")
    if a.method == "ifql":
        actor = payload["agent"]["network"]["params"]["modules_actor_flow"]
        if digest(actor) != SHA["actor"]: raise SystemExit("retrained common preflight refused frozen IFQL actor")
        if (trainer.get("compatibility") or {}).get("actor_sha256") != SHA["actor"]: raise SystemExit("retrained common preflight refused IFQL actor compatibility")
    else:
        base_actor = payload["agent"]["base"]["params"]["modules_actor_flow"]
        if digest(base_actor) != SHA["actor"]: raise SystemExit("retrained common preflight refused frozen DSRL base actor")
    print(f"retrained common {a.method} preflight OK: step={a.step} D_a=2055 action=56 actor={SHA['actor'][:12]}")

if __name__ == "__main__": main()
