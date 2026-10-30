#!/usr/bin/env python3
"""Fail-closed gate for the paper-protocol common IFQL profile (final100k, q_agg min|mean).

It first runs the unchanged 40k gate (``common_retrained_preflight.py``) so every
shared artifact (export manifest, package, IFQL source, runtime files, flags,
actor_init) is checked exactly as before, then pins the prespecified final
checkpoint ``params_100000.pkl`` and the serving-time ``q_agg`` choice.  The
40k launcher, adapter and gate are not modified.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL_STEP = 100000
CANDIDATE_STEP_FOR_SHARED_GATE = 40000
SHA = {
    # prespecified final IFQL checkpoint (ifql_seed0/params_100000.pkl)
    "ifql_final_checkpoint": "df862a14e93fa68c9dc7883bbcf4604aa7392e7d61c087634cfb873d86b049c6",
    # unchanged 40k gate / adapter and the opt-in H8 queue this profile installs
    "base_preflight": "ba4a4ad70566357415e0174c39e5765bac20232706f3c6080a6b883c88ebac45",
    "base_adapter": "b129ad06470e78e4be9ebd02140375c666b5e984bc4268012b7147cc9889f9a8",
    "contiguous_h8_queue": "bc345a48b9a84f15d5a0158575034766b4a01445cd6463e3b36a43db78f6e821",
}


def _load_base():
    spec = importlib.util.spec_from_file_location("common_retrained_preflight_base", HERE / "common_retrained_preflight.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("ifql",), required=True)
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--step", type=int, required=True)
    p.add_argument("--q-agg", choices=("min", "mean"), required=True)
    p.add_argument("--export-manifest", type=Path, required=True); p.add_argument("--package", type=Path, required=True)
    p.add_argument("--ifql-source", type=Path, required=True); p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--base-runtime", type=Path, required=True); a = p.parse_args()
    if a.step != FINAL_STEP:
        raise SystemExit(f"final common preflight refused step {a.step}; expected prespecified {FINAL_STEP}")
    base = _load_base()
    base.require("base preflight", HERE / "common_retrained_preflight.py", SHA["base_preflight"])
    base.require("base adapter", HERE / "common_retrained_ifql_server.py", SHA["base_adapter"])
    base.require("contiguous H8 queue", HERE / "contiguous_h8_queue.py", SHA["contiguous_h8_queue"])
    # 1. shared artifacts, exactly as the 40k candidate gate checks them
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], "--method", "ifql", "--run-dir", str(a.run_dir), "--step", str(CANDIDATE_STEP_FOR_SHARED_GATE),
                "--export-manifest", str(a.export_manifest), "--package", str(a.package), "--ifql-source", str(a.ifql_source),
                "--source-dir", str(a.source_dir), "--base-runtime", str(a.base_runtime)]
    try:
        base.main()
    finally:
        sys.argv = saved_argv
    # 2. prespecified final checkpoint
    ckpt = a.run_dir / f"params_{FINAL_STEP}.pkl"
    base.require("final checkpoint", ckpt, SHA["ifql_final_checkpoint"])
    payload = pickle.loads(ckpt.read_bytes()); trainer = payload.get("trainer") or {}
    if trainer.get("schema") != "common-gello-ifql-checkpoint/v1" or trainer.get("step") != FINAL_STEP:
        raise SystemExit("final common preflight refused checkpoint identity")
    actor = payload["agent"]["network"]["params"]["modules_actor_flow"]
    if base.digest(actor) != base.SHA["actor"]:
        raise SystemExit("final common preflight refused frozen IFQL actor at 100k")
    if (trainer.get("compatibility") or {}).get("actor_sha256") != base.SHA["actor"]:
        raise SystemExit("final common preflight refused IFQL actor compatibility at 100k")
    flags = json.loads((a.run_dir / "flags.json").read_text())
    trained_q_agg = flags["identity"]["agent"].get("q_agg")
    if trained_q_agg != "mean":
        raise SystemExit(f"final common preflight expected trained q_agg=mean, got {trained_q_agg!r}")
    print(f"retrained common ifql FINAL preflight OK: step={FINAL_STEP} selection=prespecified_final "
          f"trained_q_agg={trained_q_agg} inference_q_agg={a.q_agg} execution=H8/stride1/hold0")


if __name__ == "__main__":
    main()
