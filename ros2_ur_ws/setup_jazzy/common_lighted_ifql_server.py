#!/usr/bin/env python3
"""Adapter for the exact lighted-carrot common encoder + Base Flow/IFQL contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


def argument(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"common IFQL adapter requires {name}") from exc


def replace_argument(argv: list[str], name: str, value: str) -> None:
    argv[argv.index(name) + 1] = value


def stage_runtime(base: Path, overlay_ifql: Path, destination: Path) -> None:
    for source in base.iterdir():
        if source.name != "agents":
            (destination / source.name).symlink_to(source, target_is_directory=source.is_dir())
    agents = destination / "agents"
    agents.mkdir()
    for source in (base / "agents").iterdir():
        if source.name not in ("ifql.py", "__pycache__"):
            (agents / source.name).symlink_to(source, target_is_directory=source.is_dir())
    (agents / "ifql.py").symlink_to(overlay_ifql)


class ZScoreActionNorm:
    def __init__(self, mean, std) -> None:
        self.mean = np.asarray(mean, np.float64).reshape(8, 7)
        self.std = np.asarray(std, np.float64).reshape(8, 7)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("invalid common action z-score statistics")

    def denormalize(self, knots_z):
        value = np.asarray(knots_z, np.float64).reshape(8, 7) * self.std + self.mean
        value[:, 6] = np.clip(value[:, 6], 0.0, 1.0)
        return value


class CommonStats:
    def __init__(self, document) -> None:
        if document.get("schema") != "common-gello-ifql-export/v1":
            raise ValueError("common IFQL adapter received the wrong export manifest")
        norm = document["normalization"]
        self.raw = document
        self.variant = "r18_common_carrot"
        self.D_f = self.D_f1 = self.D_f2 = 1024
        self.D_a = self.D_c = 2055
        self.p_mean = np.asarray(norm["proprio_mean"], np.float32).reshape(7)
        self.p_std = np.asarray(norm["proprio_std"], np.float32).reshape(7)
        self.action = ZScoreActionNorm(norm["action_mean"], norm["action_std"])

    @classmethod
    def load(cls, path: str):
        return cls(json.loads(Path(path).read_text()))

    def actor_obs(self, state, f_cam1, f_cam2):
        f1 = np.asarray(f_cam1, np.float32).reshape(1024)
        f2 = np.asarray(f_cam2, np.float32).reshape(1024)
        # UR reports continuous joints modulo 2pi.  The lighted demos stored q1/q6
        # near +pi while the real leader commonly reports the equivalent -pi branch.
        # Canonicalize to the training branch before applying package z-scores.
        canonical = np.asarray(state, np.float64).reshape(7).copy()
        canonical[:6] += 2.0 * np.pi * np.round((self.p_mean[:6] - canonical[:6]) / (2.0 * np.pi))
        proprio = (canonical.astype(np.float32) - self.p_mean) / self.p_std
        return np.concatenate((f1, f2, proprio)).astype(np.float32)[None]


def main() -> int:
    argv = sys.argv[1:]
    original_run = Path(argument(argv, "--run-dir")).resolve()
    step = int(argument(argv, "--step"))
    sampler = argument(argv, "--sampler")
    if step != 50000 or sampler not in ("bc", "bon"):
        raise SystemExit("common IFQL adapter only accepts step 50000 with sampler bc|bon")
    package = Path(os.environ["COMMON_ENCODER_PACKAGE"]).resolve()
    source_file = Path(os.environ["COMMON_IFQL_SOURCE"]).resolve()
    base_runtime = Path(os.environ["COMMON_IFQL_BASE_RUNTIME"]).resolve()
    native_server = Path(os.environ["COMMON_IFQL_NATIVE_SERVER"]).resolve()
    source_dir = native_server.parent
    sys.path.insert(0, str(source_dir))

    with tempfile.TemporaryDirectory(prefix="common-ifql-server-") as temporary:
        root = Path(temporary)
        runtime = root / "runtime"
        runtime.mkdir()
        stage_runtime(base_runtime, source_file, runtime)
        staged_run = root / "run"
        staged_run.mkdir()
        flags = json.loads((original_run / "flags.json").read_text())
        identity = flags["identity"]
        legacy = {
            "agent": identity["agent"], "seed": identity["seed"],
            "env_name": identity["env_name"], "exp_name": "common_lighted_carrot_step50000",
        }
        (staged_run / "flags.json").write_text(json.dumps(legacy, indent=2) + "\n")
        (staged_run / f"params_{step}.pkl").symlink_to(original_run / f"params_{step}.pkl")

        os.environ["QFLOW_DIR"] = str(runtime)
        sys.path.insert(0, str(runtime))
        import ifql_server as server
        server.Q = str(runtime)
        server.NormStats = CommonStats

        class CommonActionQueue(server.aq.ActionQueue):
            def refill(self, chunk56_z, joint_reference=None):
                knots = self.norm.denormalize(server.aq.chunk_to_knots(chunk56_z))
                if joint_reference is None:
                    raise ValueError("common action queue requires a joint reference")
                reference = np.asarray(joint_reference, np.float64).reshape(6)
                knots[:, :6] += 2.0 * np.pi * np.round(
                    (reference[None, :] - knots[:, :6]) / (2.0 * np.pi)
                )
                frames = server.aq.knots_to_frames(knots)
                if not np.isfinite(frames).all():
                    raise ValueError("non-finite common action frames")
                self._q.clear()
                self._q.extend(frames)
                self.chunk_id = self.n_refill
                self.n_refill += 1
                self.chunk_step = -1
                return knots

        server.aq.ActionQueue = CommonActionQueue

        def common_refill(policy_server, state, jpeg1, jpeg2):
            started = server.time.perf_counter()
            out = policy_server.policy.plan(state, jpeg1, jpeg2)
            knots = policy_server.queue.refill(out["chunk"], joint_reference=state[:6])
            out.update(refill_ms=(server.time.perf_counter() - started) * 1e3, knots=knots)
            return out

        server.PolicyServer._refill = common_refill

        class CommonEncoder:
            def __init__(self, variant, D_f, allow_stub, device="cuda", D_f_cam1=None, D_f_cam2=None):
                if variant != "r18_common_carrot" or int(D_f) != 1024 or allow_stub:
                    raise ValueError("common IFQL adapter refused encoder contract")
                from real_pipeline.common_encoder_svf import CommonEncoderBridge
                self.bridge = CommonEncoderBridge(package, device=device)
                self.kind = self.bridge.kind

            def encode_pair(self, jpeg1, jpeg2):
                rgb1, rgb2 = server.jpeg_bytes_to_rgb(jpeg1), server.jpeg_bytes_to_rgb(jpeg2)
                f1, f2 = self.bridge.encode_rgb_pair(rgb1, rgb2)
                return [f1, f2]

        server.EncoderBridge = CommonEncoder
        original_build = server.build

        def build(args):
            policy_server = original_build(args)
            policy_server.meta["ckpt"] = str(original_run / f"params_{step}.pkl")
            policy_server.meta["run_dir"] = str(original_run)
            policy_server.meta["encoder"] = "common-gello-carrot/task-adapted-r18ss"
            policy_server.meta["action_normalization"] = "package-zscore-56/no-clip"
            policy_server.meta["joint_topology"] = "canonicalize obs to train branch; unwrap commands near current state"
            return policy_server

        server.build = build
        replace_argument(argv, "--run-dir", str(staged_run))
        if "--policy-type" not in argv:
            argv += ["--policy-type", "common_flow_bc" if sampler == "bc" else "common_ifql"]
        return server.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
