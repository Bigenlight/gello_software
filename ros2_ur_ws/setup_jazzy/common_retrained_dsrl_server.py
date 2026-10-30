#!/usr/bin/env python3
"""Common lighted-carrot DSRL100k adapter; no privileged observations."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

CAMPAIGN = Path('/home/junhyeong/real_common_baselines_20260919')
SOURCE = Path('/home/junhyeong/ifql_common_lighted_20260919/source/vision_carrot')
RUNTIME = Path('/home/junhyeong/svf_real_carrot_commonenc_20260919/runtime/qflow_svf_merged')
PACKAGE = Path('/home/junhyeong/common_encoder_lighted_20260919/carrot_seed0')
EXPORT = Path('/home/junhyeong/ifql_common_lighted_20260919/carrot_data/ifql_export_manifest.json')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def restore_agent():
    import flax
    trainer = load_module('common_dsrl_serving_trainer', CAMPAIGN / 'source/vision_carrot/train_common_ifql.py')
    cls, get_config = trainer.load_ifql(CAMPAIGN / 'source/qflow_svf/agents/ifql.py', RUNTIME)
    base_run = CAMPAIGN / 'ifql_seed0'
    flags = json.loads((base_run / 'flags.json').read_text())
    cfg = get_config()
    for key, value in flags['identity']['agent'].items():
        if key in cfg:
            cfg[key] = tuple(value) if isinstance(cfg[key], tuple) and isinstance(value, list) else value
    obs, act = np.zeros((1, 2055), np.float32), np.zeros((1, 56), np.float32)
    base = cls.create(0, obs, act, cfg)
    with (base_run / 'params_100000.pkl').open('rb') as stream:
        base = flax.serialization.from_state_dict(base, pickle.load(stream)['agent'])
    dsrl = load_module('common_dsrl_serving_agent', CAMPAIGN / 'source/qflow_svf/agents/dsrl_na.py')
    run = CAMPAIGN / 'dsrl_seed0'
    flags = json.loads((run / 'flags.json').read_text())
    cfg = dsrl.get_config()
    for key, value in flags['identity']['agent'].items():
        if key in cfg:
            cfg[key] = tuple(value) if isinstance(cfg[key], tuple) and isinstance(value, list) else value
    agent = dsrl.DSRLNAAgent.create(0, obs, obs, act, base, cfg)
    with (run / 'params_100000.pkl').open('rb') as stream:
        agent = flax.serialization.from_state_dict(agent, pickle.load(stream)['agent'])
    if agent.config['use_critic_obs'] or agent.config['actor_use_critic_obs'] or agent.config['base_clip_flow_actions']:
        raise ValueError('DSRL serving contract mismatch')
    return agent, flags


class DeterministicSampler:
    backend = 'dsrl_deterministic_latent'

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, obs, key):
        z = self.agent.sample_z(obs, seed=key, deterministic=True)
        chunk = np.asarray(self.agent.decode(obs, z), np.float32)[0]
        return chunk, {'sampler': self.backend, 'K': 1,
                       'latent_abs_max': float(np.abs(np.asarray(z)).max())}


def main():
    os.environ['QFLOW_DIR'] = str(RUNTIME)
    sys.path.insert(0, str(SOURCE))
    import ifql_server as server  # Native module imports torch before JAX.
    from common_retrained_ifql_server import install_common
    from common_retrained_preflight import main as preflight
    # Invoke strict artifact validation even when called without the shell wrapper.
    old_argv = sys.argv
    sys.argv = [old_argv[0], '--method', 'dsrl', '--run-dir', str(CAMPAIGN / 'dsrl_seed0'),
                '--step', '100000', '--export-manifest', str(EXPORT), '--package', str(PACKAGE),
                '--ifql-source', str(CAMPAIGN / 'source/qflow_svf/agents/ifql.py'),
                '--source-dir', str(SOURCE), '--base-runtime', str(RUNTIME)]
    try:
        preflight()
    finally:
        sys.argv = old_argv
    install_common(server, PACKAGE)

    def build(args):
        if (Path(args.run_dir).resolve() != CAMPAIGN / 'dsrl_seed0' or args.step != 100000
                or args.sampler != 'actor' or Path(args.norm_stats).resolve() != EXPORT
                or args.random_init or args.stub_encoder or args.px):
            raise ValueError('DSRL adapter only serves pinned common100k deterministic actor')
        agent, flags = restore_agent()
        stats = server.NormStats.load(str(EXPORT))
        encoder = server.EncoderBridge(stats.variant, stats.D_f, False, device=args.device,
                                       D_f_cam1=stats.D_f1, D_f_cam2=stats.D_f2)
        sampler = DeterministicSampler(agent)
        policy = server.IFQLPolicy(agent, sampler, stats, encoder)
        meta = {'ckpt': str(CAMPAIGN / 'dsrl_seed0/params_100000.pkl'), 'ckpt_epoch': 100000,
                'run_dir': str(CAMPAIGN / 'dsrl_seed0'), 'norm_stats': str(EXPORT),
                'variant': stats.variant, 'encoder': encoder.kind, 'D_a': 2055, 'D_c': 2055,
                'agent_module': 'dsrl_na', 'agent_config': flags['identity']['agent'],
                'policy_type': 'common_dsrl', 'sampler': 'actor', 'num_samples': 1,
                'sampler_backend': sampler.backend, 'px': False, 'random_init': False,
                'action_normalization': 'package-zscore-56/no-clip',
                'joint_topology': 'canonicalize obs to train branch; unwrap commands near current state'}
        server.log(f'norm_stats: {EXPORT}')
        server.log('agent ready: dsrl_na px=false sampler=deterministic D_a=2055 D_c=2055 action_dim=56')
        return server.PolicyServer(policy, stats, meta, args.host, args.port, args.budget_s, args.log_dir)

    server.build = build
    return server.main(sys.argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
