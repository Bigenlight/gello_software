#!/usr/bin/env python3
"""Serving adapter for the paper-protocol common IFQL profile (final100k, q_agg min|mean).

Sibling of ``common_retrained_ifql_server.py`` (40k, mean, legacy 24-frame queue),
which is imported unchanged for its common encoder/z-score/joint-reference hooks.
Differences, all explicit:

* checkpoint: prespecified ``params_100000.pkl`` (no validation-based selection);
* ``q_agg``: read from ``IFQL_Q_AGG`` (min|mean) and written into the staged
  ``flags.json`` agent config.  q_agg is a serving-time ensemble reduction in
  ``agents/ifql.py`` (``_agg`` -> ``sample_candidates`` -> ``sample_actions``);
  the checkpoint weights (trained with q_agg=mean targets) are untouched;
* execution: opt-in contiguous H8 queue (queue8/stride1/hold0), the fixed
  real-robot time-axis contract, instead of the legacy 24-frame stride-3 queue.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
FINAL_STEP = 100000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    base = _load("common_retrained_ifql_base_adapter", HERE / "common_retrained_ifql_server.py")
    argv = sys.argv[1:]
    run_dir = Path(base._arg(argv, "--run-dir")).resolve()
    step = int(base._arg(argv, "--step"))
    sampler = base._arg(argv, "--sampler")
    q_agg = os.environ.get("IFQL_Q_AGG", "")
    if step != FINAL_STEP or sampler != "bon":
        raise SystemExit("final common IFQL adapter only accepts step 100000 and sampler bon")
    if q_agg not in ("min", "mean"):
        raise SystemExit("final common IFQL adapter requires IFQL_Q_AGG=min|mean (serving-time ensemble reduction)")
    package = Path(os.environ["COMMON_ENCODER_PACKAGE"]).resolve()
    source_file = Path(os.environ["COMMON_IFQL_SOURCE"]).resolve()
    native_server = Path(os.environ["COMMON_IFQL_NATIVE_SERVER"]).resolve()
    source_dir = native_server.parent
    sys.path.insert(0, str(source_dir))
    base_runtime = Path(os.environ["COMMON_IFQL_BASE_RUNTIME"]).resolve()
    with tempfile.TemporaryDirectory(prefix="common-retrained-ifql-final-") as tmp:
        root = Path(tmp); runtime = root / "runtime"; runtime.mkdir()
        for src in base_runtime.iterdir():
            if src.name != "agents": (runtime / src.name).symlink_to(src, target_is_directory=src.is_dir())
        agents = runtime / "agents"; agents.mkdir()
        for src in (base_runtime / "agents").iterdir():
            if src.name not in ("ifql.py", "__pycache__"): (agents / src.name).symlink_to(src, target_is_directory=src.is_dir())
        (agents / "ifql.py").symlink_to(source_file)
        staged = root / "run"; staged.mkdir()
        flags = json.loads((run_dir / "flags.json").read_text()); ident = flags["identity"]
        agent_cfg = dict(ident["agent"]); trained_q_agg = agent_cfg.get("q_agg")
        agent_cfg["q_agg"] = q_agg
        (staged / "flags.json").write_text(json.dumps({"agent": agent_cfg, "seed": ident["seed"], "env_name": ident["env_name"],
                                                       "exp_name": f"common_retrained_carrot_final_step{FINAL_STEP}_q{q_agg}"}, indent=2) + "\n")
        (staged / f"params_{step}.pkl").symlink_to(run_dir / f"params_{step}.pkl")
        os.environ["QFLOW_DIR"] = str(runtime); sys.path.insert(0, str(runtime))
        sys.path.insert(0, str(HERE))
        import ifql_server as server
        preflight = _load("common_retrained_preflight_final_gate", HERE / "common_retrained_preflight_final.py")
        saved_argv = sys.argv
        sys.argv = [saved_argv[0], '--method', 'ifql', '--run-dir', str(run_dir), '--step', str(step), '--q-agg', q_agg,
                    '--export-manifest', base._arg(argv, '--norm-stats'), '--package', str(package),
                    '--ifql-source', str(source_file), '--source-dir', str(source_dir), '--base-runtime', str(base_runtime)]
        try:
            preflight.main()
        finally:
            sys.argv = saved_argv
        server.Q = str(runtime); base.install_common(server, package)
        from contiguous_h8_queue import install_contiguous_h8, queue_metadata
        install_contiguous_h8(server.aq)
        original_build = server.build

        def build(args):
            if (args.random_init or args.stub_encoder or args.px or args.serve_meta
                    or args.agent_module not in (None, 'ifql') or args.num_samples != 32):
                raise ValueError('Only pinned common final100k IFQL BoN32 (q_agg min|mean) is supported')
            policy_server = original_build(args)
            served = policy_server.policy.agent.config.get('q_agg')
            if served != q_agg:
                raise ValueError(f'served agent q_agg={served!r} does not match requested {q_agg!r}')
            if server.aq.QUEUE_LEN != 8 or server.aq.N_HOLD != 0:
                raise ValueError('contiguous H8 queue was not installed')
            policy_server.meta.update(ckpt=str(run_dir / f"params_{step}.pkl"), run_dir=str(run_dir),
                                      q_agg=q_agg, q_agg_trained=trained_q_agg,
                                      checkpoint_selection=f"prespecified_final_{FINAL_STEP}",
                                      execution=queue_metadata(),
                                      encoder="common-gello-carrot/task-adapted-r18ss",
                                      action_normalization="package-zscore-56/no-clip",
                                      joint_topology="canonicalize obs to train branch; unwrap commands near current state")
            policy_server.logger.meta.update(policy_server.meta)
            server.log(f"q_agg: trained={trained_q_agg} inference={q_agg} (serving-time ensemble reduction; weights unchanged)")
            server.log(f"checkpoint selection: prespecified final step {FINAL_STEP}; execution queue8/stride1/hold0")
            return policy_server
        server.build = build
        base._replace(argv, "--run-dir", str(staged))
        if "--policy-type" not in argv: argv += ["--policy-type", f"common_ifql_final{FINAL_STEP}_q{q_agg}"]
        return server.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
