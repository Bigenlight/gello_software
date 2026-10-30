#!/usr/bin/env python3
"""Bounded, robot-free serving-contract checks for the retrained common CLIs.

This file is intentionally standalone and does not alter either checkout.  It
does not start ROS, a socket, cameras, or the native server loop.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import flax
import jax
import numpy as np


ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path("/home/junhyeong/real_common_baselines_20260919")
DATA = Path("/home/junhyeong/ifql_common_lighted_20260919/carrot_data")
NATIVE = Path("/home/junhyeong/ifql_common_lighted_20260919/source/vision_carrot")
RUNTIME = Path("/home/junhyeong/svf_real_carrot_commonenc_20260919/runtime/qflow_svf_merged")
PACKAGE = Path("/home/junhyeong/common_encoder_lighted_20260919/carrot_seed0")


def tree_digest(tree) -> str:
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        a = np.asarray(leaf)
        h.update(str(a.dtype).encode())
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_action_queue():
    return load_module("common_cli_native_action_queue", NATIVE / "action_queue.py")


def test_stats_canonicalization():
    adapter = load_module("common_cli_ifql_adapter", ROOT / "common_retrained_ifql_server.py")
    stats = adapter.CommonStats.load(DATA / "ifql_export_manifest.json")
    state = stats.p_mean.astype(np.float64)
    shifted = state.copy()
    shifted[:6] += 2.0 * np.pi * np.array([1, -1, 2, -2, 1, -1], dtype=np.float64)
    f1 = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    f2 = np.linspace(1.0, -1.0, 1024, dtype=np.float32)
    a = stats.actor_obs(state, f1, f2)
    b = stats.actor_obs(shifted, f1, f2)
    if a.shape != (1, 2055) or not np.isfinite(a).all():
        raise AssertionError(f"bad canonical observation: {a.shape}")
    np.testing.assert_array_equal(a, b)
    return {"shape": list(a.shape), "max_abs_shift_invariant_error": float(np.max(np.abs(a - b)))}


def test_queue_contract():
    adapter = load_module("common_cli_ifql_queue_adapter", ROOT / "common_retrained_ifql_server.py")
    aq = native_action_queue()
    fake = types.SimpleNamespace(aq=types.SimpleNamespace(ActionQueue=aq.ActionQueue,
                                                           chunk_to_knots=aq.chunk_to_knots,
                                                           knots_to_frames=aq.knots_to_frames),
                                 PolicyServer=type("PolicyServer", (), {}))
    adapter.install_common(fake, PACKAGE)
    stats = adapter.CommonStats.load(DATA / "ifql_export_manifest.json")
    q = fake.aq.ActionQueue(stats.action)
    normalized = np.linspace(-0.8, 0.8, 56, dtype=np.float64)
    reference = stats.p_mean[:6].astype(np.float64) + np.array([2 * np.pi, -2 * np.pi, 0, 2 * np.pi, 0, -2 * np.pi])
    knots = q.refill(normalized, joint_reference=reference)
    if knots.shape != (8, 7) or len(q) != 24 or not np.isfinite(knots).all():
        raise AssertionError(f"queue contract failed: knots={knots.shape}, queue={len(q)}")
    expected = aq.knots_to_frames(knots)
    actual = np.stack([q.pop() for _ in range(24)])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    if not q.empty():
        raise AssertionError("queue did not consume exactly 24 native frames")
    return {"knots_shape": list(knots.shape), "frame_count": int(len(actual)),
            "max_native_interpolation_error": float(np.max(np.abs(actual - expected)))}


def test_dsrl_determinism_and_frozen_base():
    adapter = load_module("common_cli_dsrl_adapter", ROOT / "common_retrained_dsrl_server.py")
    agent, flags = adapter.restore_agent()
    with np.load(DATA / "carrot_ifql_val.npz", allow_pickle=False) as archive:
        all_obs = np.asarray(archive["observations"], dtype=np.float32)
    heldout = np.linspace(0, len(all_obs) - 1, 16, dtype=np.int64)
    obs = all_obs[heldout]
    key = jax.random.PRNGKey(20260919)
    z1 = np.asarray(agent.sample_z(obs, seed=key, deterministic=True))
    z2 = np.asarray(agent.sample_z(obs, seed=key, deterministic=True))
    a1 = np.asarray(agent.decode(obs, z1))
    a2 = np.asarray(agent.decode(obs, z2))
    np.testing.assert_array_equal(z1, z2)
    np.testing.assert_array_equal(a1, a2)
    if z1.shape != (16, 56) or a1.shape != (16, 56) or not np.isfinite(a1).all():
        raise AssertionError(f"bad DSRL deterministic output: z={z1.shape}, action={a1.shape}")
    sampler = adapter.DeterministicSampler(agent)
    sampler_chunk, sampler_meta = sampler(obs[:1], jax.random.PRNGKey(20260919))
    direct_chunk = np.asarray(agent.sample_actions(obs[:1], seed=jax.random.PRNGKey(20260919), deterministic=True))[0]
    np.testing.assert_array_equal(sampler_chunk, direct_chunk)
    with (CAMPAIGN / "dsrl_seed0/params_100000.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    expected = payload["trainer"]["frozen_base_actor_sha256"]
    actual = tree_digest(agent.base.params["modules_actor_flow"])
    if actual != expected:
        raise AssertionError(f"frozen base actor digest mismatch: {actual} != {expected}")
    literal = "7c4dfbe5c2a070e5ca0f232512b11e197a6b01601c83a9bd72e65be208290c10"
    if actual != literal:
        raise AssertionError(f"frozen base actor digest differs from pinned contract: {actual} != {literal}")
    if flags["identity"]["agent"].get("use_critic_obs") or flags["identity"]["agent"].get("actor_use_critic_obs"):
        raise AssertionError("DSRL serving contract enables privileged critic observations")
    return {"heldout_indices": heldout.tolist(), "z_shape": list(z1.shape),
            "action_shape": list(a1.shape), "deterministic_equal": True,
            "sampler_backend": sampler_meta["sampler"], "frozen_base_actor_sha256": actual}


def main():
    results = {
        "stats_canonicalization": test_stats_canonicalization(),
        "queue_contract": test_queue_contract(),
        "dsrl_determinism_and_frozen_base": test_dsrl_determinism_and_frozen_base(),
    }
    print(json.dumps({"status": "PASS", "tests": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
