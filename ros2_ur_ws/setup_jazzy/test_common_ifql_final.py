#!/usr/bin/env python3
"""CPU cached-feature inference smoke for the final100k common IFQL profile (min vs mean).

Restores ifql_seed0/params_100000.pkl through the campaign's own bridge, once with the
staged serving config q_agg=min and once with q_agg=mean, and checks on 16 held-out
common-encoder observations that: q_agg is a serving-time choice (identical actor
candidates for the same PRNG key, aggregate Q equals min/mean of the raw twin heads),
outputs are finite, and the BoN32 selections differ between min and mean.
No robot, camera, ROS, socket or GPU.  Not a task-success or safety claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import flax
import jax
import numpy as np

CAMPAIGN = Path("/home/junhyeong/real_common_baselines_20260919")
DATA = Path("/home/junhyeong/ifql_common_lighted_20260919/carrot_data")
QFLOW = Path("/home/junhyeong/svf_real_carrot_commonenc_20260919/runtime/qflow_svf_merged")
FINAL_STEP = 100000
FINAL_SHA = "df862a14e93fa68c9dc7883bbcf4604aa7392e7d61c087634cfb873d86b049c6"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _restore(smoke, bridge, ifql_source, observations, actions, q_agg):
    flags = json.loads((CAMPAIGN / "ifql_seed0/flags.json").read_text())
    flags["identity"]["agent"]["q_agg"] = q_agg          # staged serving config, checkpoint untouched
    agent = smoke._build_ifql(bridge, ifql_source, QFLOW, flags, observations, actions)
    with (CAMPAIGN / f"ifql_seed0/params_{FINAL_STEP}.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    if int(payload["trainer"]["step"]) != FINAL_STEP:
        raise ValueError("wrong checkpoint step")
    agent = flax.serialization.from_state_dict(agent, payload["agent"])
    if agent.config["q_agg"] != q_agg or agent.config["rho"] != 0.0 or agent.config["num_qs"] != 2:
        raise ValueError(f"unexpected served config: {dict(agent.config)}")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    import hashlib
    h = hashlib.sha256()
    with (CAMPAIGN / f"ifql_seed0/params_{FINAL_STEP}.pkl").open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != FINAL_SHA:
        raise SystemExit("final checkpoint SHA mismatch")
    smoke = _load("campaign_smoke_common_policies", CAMPAIGN / "tests/smoke_common_policies.py")
    bridge, ifql_source, _ = smoke._load_ifql_bridge(CAMPAIGN, QFLOW)
    train, val = smoke._load_rows(DATA)
    heldout = np.linspace(0, len(val["observations"]) - 1, 16, dtype=np.int64)
    obs = val["observations"][heldout].astype(np.float32)
    act = train["actions"][:1].astype(np.float32)
    norm = json.loads((DATA / "ifql_export_manifest.json").read_text())["normalization"]
    a_mean = np.asarray(norm["action_mean"], np.float32); a_std = np.asarray(norm["action_std"], np.float32)
    agents = {q: _restore(smoke, bridge, ifql_source, obs, act, q) for q in ("min", "mean")}
    per_seed = []
    total_diff = 0
    for s in range(args.seeds):
        key = jax.random.PRNGKey(20260922 + s)
        out = {}
        for q, agent in agents.items():
            cands, qs, heads = agent.sample_candidates(obs, seed=key, num_samples=32, return_q_heads=True)
            chosen = np.asarray(agent.sample_actions(obs, seed=key, num_samples=32))
            out[q] = dict(cands=np.asarray(cands), qs=np.asarray(qs), heads=np.asarray(heads), chosen=chosen)
        c_min, c_mean = out["min"], out["mean"]
        if not (np.array_equal(c_min["cands"], c_mean["cands"]) and np.array_equal(c_min["heads"], c_mean["heads"])):
            raise ValueError("candidates/raw Q heads differ between min and mean agents (they must not)")
        if c_min["heads"].shape != (2, 16, 32):
            raise ValueError(f"bad q_heads shape {c_min['heads'].shape}")
        np.testing.assert_allclose(c_min["qs"], c_min["heads"].min(axis=0), rtol=0, atol=1e-6)
        np.testing.assert_allclose(c_mean["qs"], c_mean["heads"].mean(axis=0), rtol=0, atol=1e-5)
        for q in ("min", "mean"):
            if not np.isfinite(out[q]["chosen"]).all() or out[q]["chosen"].shape != (16, 56):
                raise ValueError(f"non-finite/bad chosen actions for {q}")
            best = np.argmax(out[q]["qs"], axis=-1)
            np.testing.assert_array_equal(out[q]["chosen"], out[q]["cands"][np.arange(16), best])
            raw = out[q]["chosen"] * a_std + a_mean
            if not np.isfinite(raw).all():
                raise ValueError("non-finite denormalized action")
        arg_min = np.argmax(c_min["qs"], axis=-1); arg_mean = np.argmax(c_mean["qs"], axis=-1)
        n_diff = int((arg_min != arg_mean).sum())
        total_diff += n_diff
        per_seed.append({"seed": 20260922 + s, "argmax_differs": n_diff, "of": 16,
                         "max_abs_action_diff": float(np.abs(c_min["chosen"] - c_mean["chosen"]).max()),
                         "q_head_gap_mean": float(np.abs(c_min["heads"][0] - c_min["heads"][1]).mean()),
                         "q_agg_min_chosen_mean": float(c_min["qs"][np.arange(16), arg_min].mean()),
                         "q_agg_mean_chosen_mean": float(c_mean["qs"][np.arange(16), arg_mean].mean())})
    if total_diff == 0:
        raise ValueError("min and mean selected identical candidates for every observation and seed")
    report = {"schema": "common-ifql-final100k-smoke/v1", "checkpoint": str(CAMPAIGN / f"ifql_seed0/params_{FINAL_STEP}.pkl"),
              "checkpoint_sha256": FINAL_SHA, "trained_q_agg": "mean", "device": jax.devices()[0].platform,
              "heldout_indices": heldout.tolist(), "per_seed": per_seed, "total_argmax_differs": total_diff,
              "finite": True, "q_agg_is_serving_time_choice": True,
              "note": "cached-feature inference only; no robot rollout, success or safety claim", "robot_validated": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "out": str(args.out), "total_argmax_differs": total_diff, "device": report["device"]}))


if __name__ == "__main__":
    main()
