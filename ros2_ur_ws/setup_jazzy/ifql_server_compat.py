#!/usr/bin/env python3
"""Explicit compatibility adapter for the pinned ROS-free IFQL server.

Use this only when the server source needs the r18-only encoder workaround or
the export is inference-only (``params_<step>.infer.pkl``). The original
server script remains read-only and all other variants fail closed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import shutil
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--server-script", required=True)
    parser.add_argument("--inference-checkpoint", default=None)
    parser.add_argument("--q-agg-override", choices=("mean", "min"), default=None,
                        help="override flags.agent.q_agg before constructing the agent")
    known, server_args = parser.parse_known_args()
    return known, server_args


def r18_only(self, device="cuda"):
    import torch
    import torchvision
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    if not weights_path.is_file():
        raise RuntimeError(f"cached ResNet18 weights missing; refusing download: {weights_path}")

    self.torch = torch
    self.device = device
    self.dino = None
    self.r18 = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    ).to(device).eval()
    self.xs = torch.linspace(-1, 1, 7, device=device)
    for parameter in self.r18.parameters():
        parameter.requires_grad_(False)


def main() -> int:
    adapter_args, server_args = parse_args()
    server_path = Path(adapter_args.server_script).resolve()
    if not server_path.is_file():
        raise SystemExit(f"server script not found: {server_path}")

    run_index = server_args.index("--run-dir")
    run_dir = Path(server_args[run_index + 1]).resolve()
    step_index = server_args.index("--step") if "--step" in server_args else None
    step = int(server_args[step_index + 1]) if step_index is not None else None
    if step is None:
        raise SystemExit("compat adapter requires explicit --step")

    stats_index = server_args.index("--norm-stats")
    stats_path = Path(server_args[stats_index + 1])
    stats = json.loads(stats_path.read_text())
    variant = stats.get("variant")
    if variant != "r18_ss":
        raise SystemExit(f"compat adapter supports only variant r18_ss, got {variant!r}")
    if int(stats.get("D_a", -1)) != 2055:
        raise SystemExit(f"r18_ss compatibility expects norm_stats D_a=2055, got {stats.get('D_a')}")

    full_checkpoint = run_dir / f"params_{step}.pkl"
    infer_checkpoint = Path(adapter_args.inference_checkpoint).resolve() if adapter_args.inference_checkpoint else None
    if full_checkpoint.is_file() and infer_checkpoint is None:
        selected_checkpoint = full_checkpoint
    else:
        if infer_checkpoint is None:
            infer_checkpoint = run_dir / f"params_{step}.infer.pkl"
        if not infer_checkpoint.is_file():
            raise SystemExit(f"no full or inference checkpoint for step {step}")
        selected_checkpoint = infer_checkpoint

    if selected_checkpoint.suffix == ".pkl" and selected_checkpoint.name.endswith(".infer.pkl"):
        with selected_checkpoint.open("rb") as checkpoint_file:
            state = pickle.load(checkpoint_file)["agent"]
        network = state.get("network", {})
        if "params" not in network or "step" not in network:
            raise SystemExit("inference checkpoint lacks network params/step")

    server_module_name = "ifql_server_compat_target"
    sys.path.insert(0, str(server_path.parent))
    spec = importlib.util.spec_from_file_location(server_module_name, server_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load server script: {server_path}")
    server = importlib.util.module_from_spec(spec)
    sys.modules[server_module_name] = server
    spec.loader.exec_module(server)
    sys.path.insert(0, os.environ["QFLOW_DIR"])
    import extract_features as feature_module

    feature_module.FeatureHeads.__init__ = r18_only
    original_run_dir = run_dir
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    stage_dir = Path(tempfile.mkdtemp(prefix="ifql_compat_"))
    try:
        flags_path = run_dir / "flags.json"
        if adapter_args.q_agg_override:
            flags = json.loads(flags_path.read_text())
            agent_config = flags.get("agent")
            if not isinstance(agent_config, dict):
                raise SystemExit(f"{flags_path} has no agent config")
            trained_q_agg = agent_config.get("q_agg")
            agent_config["q_agg"] = adapter_args.q_agg_override
            (stage_dir / "flags.json").write_text(json.dumps(flags, indent=2, sort_keys=True) + "\n")
            print(f"IFQL compatibility q_agg: trained={trained_q_agg} "
                  f"inference={adapter_args.q_agg_override}", flush=True)
        else:
            os.symlink(flags_path, stage_dir / "flags.json")
        staged_name = f"params_{step}.pkl"
        os.symlink(selected_checkpoint, stage_dir / staged_name)
        server_args[run_index + 1] = str(stage_dir)

        if selected_checkpoint.name.endswith(".infer.pkl"):
            import flax
            import jax.numpy as jnp
            from utils import flax_utils

            def restore_infer(agent, _run_dir, _step):
                with selected_checkpoint.open("rb") as checkpoint_file:
                    state = pickle.load(checkpoint_file)["agent"]
                saved_params = state["network"]["params"]
                template_leaves = flax.traverse_util.flatten_dict(
                    flax.core.unfreeze(agent.network.params), sep="/")
                saved_leaves = flax.traverse_util.flatten_dict(saved_params, sep="/")
                if set(template_leaves) != set(saved_leaves):
                    raise ValueError("inference checkpoint parameter keys differ from model template")
                for key, template in template_leaves.items():
                    actual = np.asarray(saved_leaves[key])
                    expected = np.asarray(template)
                    if actual.shape != expected.shape or actual.dtype != expected.dtype:
                        raise ValueError(
                            f"inference checkpoint shape/dtype mismatch at {key}: "
                            f"{actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
                params = flax.serialization.from_state_dict(agent.network.params, saved_params)
                restored = agent.replace(network=agent.network.replace(
                    params=params, step=int(np.asarray(state["network"]["step"]))))
                if state.get("rng") is not None:
                    restored = restored.replace(
                        rng=jnp.asarray(np.asarray(state["rng"], dtype=np.uint32)))
                return restored

            flax_utils.restore_agent = restore_infer

        original_build = server.build

        def build_with_truthful_metadata(args):
            policy_server = original_build(args)
            actual = str(selected_checkpoint)
            policy_server.meta["ckpt"] = actual
            policy_server.meta["run_dir"] = str(original_run_dir)
            policy_server.logger.meta["ckpt"] = actual
            policy_server.logger.meta["run_dir"] = str(original_run_dir)
            if adapter_args.q_agg_override:
                policy_server.meta["q_agg"] = adapter_args.q_agg_override
                policy_server.logger.meta["q_agg"] = adapter_args.q_agg_override
            if selected_checkpoint.name.endswith(".infer.pkl"):
                policy_server.meta["network_step"] = int(np.asarray(network["step"]))
                policy_server.logger.meta["network_step"] = policy_server.meta["network_step"]
            return policy_server

        server.build = build_with_truthful_metadata

        sys.argv = [str(server_path)] + server_args
        return server.main(server_args)
    finally:
        shutil.rmtree(stage_dir)
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
