#!/usr/bin/env python3
"""Serving adapter for the 2026-09-19 retrained common IFQL checkpoint.

This file deliberately stays outside the ROS checkout.  ``install_common`` is
also used by the DSRL adapter: it installs the common encoder, statistics and
joint-reference action queue onto an already imported native server module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


def _arg(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"retrained common adapter requires {name}") from exc


def _replace(argv, name, value):
    argv[argv.index(name) + 1] = value


class ZScoreActionNorm:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float64).reshape(8, 7)
        self.std = np.asarray(std, np.float64).reshape(8, 7)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("invalid common action z-score statistics")

    def denormalize(self, value):
        knots = np.asarray(value, np.float64).reshape(8, 7) * self.std + self.mean
        knots[:, 6] = np.clip(knots[:, 6], 0.0, 1.0)
        return knots


class CommonStats:
    def __init__(self, document):
        if document.get("schema") != "common-gello-ifql-export/v1":
            raise ValueError("wrong common export manifest")
        norm = document["normalization"]
        self.raw = document
        self.variant = "r18_common_carrot"
        self.D_f = self.D_f1 = self.D_f2 = 1024
        self.D_a = self.D_c = 2055
        self.p_mean = np.asarray(norm["proprio_mean"], np.float32).reshape(7)
        self.p_std = np.asarray(norm["proprio_std"], np.float32).reshape(7)
        self.action = ZScoreActionNorm(norm["action_mean"], norm["action_std"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def actor_obs(self, state, f_cam1, f_cam2):
        f1 = np.asarray(f_cam1, np.float32).reshape(1024)
        f2 = np.asarray(f_cam2, np.float32).reshape(1024)
        state = np.asarray(state, np.float64).reshape(7).copy()
        state[:6] += 2 * np.pi * np.round((self.p_mean[:6] - state[:6]) / (2 * np.pi))
        proprio = (state.astype(np.float32) - self.p_mean) / self.p_std
        return np.concatenate((f1, f2, proprio)).astype(np.float32)[None]


def install_common(server, package):
    """Install common stats, encoder and queue hooks on *server*.

    The function intentionally does not build a policy or alter checkpoint
    loading, allowing callers to use either native IFQL or DSRL policy code.
    """
    package = Path(package).resolve()

    class CommonActionQueue(server.aq.ActionQueue):
        def refill(self, chunk56_z, joint_reference=None):
            if joint_reference is None:
                raise ValueError("common action queue requires a joint reference")
            knots = self.norm.denormalize(server.aq.chunk_to_knots(chunk56_z))
            reference = np.asarray(joint_reference, np.float64).reshape(6)
            knots[:, :6] += 2 * np.pi * np.round((reference[None, :] - knots[:, :6]) / (2 * np.pi))
            frames = server.aq.knots_to_frames(knots)
            if not np.isfinite(frames).all():
                raise ValueError("non-finite common action frames")
            self._q.clear(); self._q.extend(frames)
            self.chunk_id = self.n_refill; self.n_refill += 1; self.chunk_step = -1
            return knots

    def common_refill(policy_server, state, jpeg1, jpeg2):
        started = server.time.perf_counter()
        out = policy_server.policy.plan(state, jpeg1, jpeg2)
        knots = policy_server.queue.refill(out["chunk"], joint_reference=state[:6])
        out.update(refill_ms=(server.time.perf_counter() - started) * 1e3, knots=knots)
        return out

    class CommonEncoder:
        def __init__(self, variant, D_f, allow_stub, device="cuda", D_f_cam1=None, D_f_cam2=None):
            if variant != "r18_common_carrot" or int(D_f) != 1024 or allow_stub:
                raise ValueError("common encoder contract refused")
            from real_pipeline.common_encoder_svf import CommonEncoderBridge
            self.bridge = CommonEncoderBridge(package, device=device)
            self.kind = self.bridge.kind

        def encode_pair(self, jpeg1, jpeg2):
            rgb1, rgb2 = server.jpeg_bytes_to_rgb(jpeg1), server.jpeg_bytes_to_rgb(jpeg2)
            return list(self.bridge.encode_rgb_pair(rgb1, rgb2))

    server.NormStats = CommonStats
    server.aq.ActionQueue = CommonActionQueue
    server.PolicyServer._refill = common_refill
    server.EncoderBridge = CommonEncoder
    return CommonStats


def main():
    argv = sys.argv[1:]
    run_dir = Path(_arg(argv, "--run-dir")).resolve()
    step = int(_arg(argv, "--step"))
    sampler = _arg(argv, "--sampler")
    if step != 40000 or sampler not in ("bc", "bon"):
        raise SystemExit("retrained common IFQL adapter only accepts step 40000 and sampler bc|bon")
    package = Path(os.environ["COMMON_ENCODER_PACKAGE"]).resolve()
    source_file = Path(os.environ["COMMON_IFQL_SOURCE"]).resolve()
    native_server = Path(os.environ["COMMON_IFQL_NATIVE_SERVER"]).resolve()
    source_dir = native_server.parent
    sys.path.insert(0, str(source_dir))
    base_runtime = Path(os.environ["COMMON_IFQL_BASE_RUNTIME"]).resolve()
    with tempfile.TemporaryDirectory(prefix="common-retrained-ifql-") as tmp:
        root = Path(tmp); runtime = root / "runtime"; runtime.mkdir()
        for src in base_runtime.iterdir():
            if src.name != "agents": (runtime / src.name).symlink_to(src, target_is_directory=src.is_dir())
        agents = runtime / "agents"; agents.mkdir()
        for src in (base_runtime / "agents").iterdir():
            if src.name not in ("ifql.py", "__pycache__"): (agents / src.name).symlink_to(src, target_is_directory=src.is_dir())
        (agents / "ifql.py").symlink_to(source_file)
        staged = root / "run"; staged.mkdir()
        flags = json.loads((run_dir / "flags.json").read_text()); ident = flags["identity"]
        (staged / "flags.json").write_text(json.dumps({"agent": ident["agent"], "seed": ident["seed"], "env_name": ident["env_name"], "exp_name": "common_retrained_carrot_step40000"}, indent=2) + "\n")
        (staged / f"params_{step}.pkl").symlink_to(run_dir / f"params_{step}.pkl")
        os.environ["QFLOW_DIR"] = str(runtime); sys.path.insert(0, str(runtime))
        import ifql_server as server
        from common_retrained_preflight import main as preflight
        saved_argv = sys.argv
        sys.argv = [saved_argv[0], '--method', 'ifql', '--run-dir', str(run_dir),
                    '--step', str(step), '--export-manifest', _arg(argv, '--norm-stats'),
                    '--package', str(package), '--ifql-source', str(source_file),
                    '--source-dir', str(source_dir), '--base-runtime', str(base_runtime)]
        try:
            preflight()
        finally:
            sys.argv = saved_argv
        server.Q = str(runtime); install_common(server, package)
        original_build = server.build
        def build(args):
            if (args.random_init or args.stub_encoder or args.px or args.serve_meta
                    or args.agent_module not in (None, 'ifql')
                    or (sampler == 'bon' and args.num_samples != 32)):
                raise ValueError('Only pinned common40k BC K1 / IFQL BoN32 mean is supported')
            policy_server = original_build(args)
            policy_server.meta.update(ckpt=str(run_dir / f"params_{step}.pkl"), run_dir=str(run_dir), encoder="common-gello-carrot/task-adapted-r18ss", action_normalization="package-zscore-56/no-clip", joint_topology="canonicalize obs to train branch; unwrap commands near current state")
            policy_server.logger.meta.update(policy_server.meta)
            return policy_server
        server.build = build
        _replace(argv, "--run-dir", str(staged))
        if "--policy-type" not in argv: argv += ["--policy-type", "common_flow_bc" if sampler == "bc" else "common_ifql"]
        return server.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
