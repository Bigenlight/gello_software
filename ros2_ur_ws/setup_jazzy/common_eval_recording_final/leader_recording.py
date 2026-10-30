"""Paper-protocol final100k common IFQL metadata; reuse the common shim and the unchanged ROS recorder."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SETUP = HERE.parent
COMMON = SETUP / 'common_eval_recording' / 'leader_recording.py'
spec = importlib.util.spec_from_file_location('common_final_base_leader_recording', COMMON)
common = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = common
spec.loader.exec_module(common)
instrumented_leader = common.instrumented_leader

FINAL_STEP = 100000
PROFILES = {'retrained-ifql-final100k-min': 'min', 'retrained-ifql-final100k-mean': 'mean'}
EXECUTION = {'H': 8, 'stride': 1, 'interp': 8, 'hold': 0}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def launch_metadata():
    profile = os.environ['COMMON_EVAL_PROFILE']
    if profile not in PROFILES:
        raise ValueError(f'unsupported final common recording profile: {profile}')
    q_agg = PROFILES[profile]
    if os.environ.get('IFQL_Q_AGG') != q_agg:
        raise ValueError(f'IFQL_Q_AGG={os.environ.get("IFQL_Q_AGG")!r} does not match profile {profile}')
    if int(os.environ['IFQL_STEP']) != FINAL_STEP:
        raise ValueError(f'final profile requires IFQL_STEP={FINAL_STEP}')
    if os.environ['IFQL_SAMPLER'] != 'bon' or int(os.environ['IFQL_NUM_SAMPLES']) != 32:
        raise ValueError('final profile requires sampler bon K=32')
    if Path(os.environ['IFQL_SERVER_PY']).resolve() != (SETUP / 'common_retrained_ifql_final_server.py'):
        raise ValueError('final profile requires the final serving adapter')
    saved = os.environ['COMMON_EVAL_PROFILE']
    os.environ['COMMON_EVAL_PROFILE'] = 'retrained-ifql'
    try:
        meta = common.launch_metadata()
    finally:
        os.environ['COMMON_EVAL_PROFILE'] = saved
    trained = meta['flags']['identity']['agent'].get('q_agg')
    if trained != 'mean':
        raise ValueError(f'expected trained q_agg=mean, got {trained!r}')
    for path in (Path(__file__).resolve(), SETUP / 'common_retrained_preflight_final.py', SETUP / 'contiguous_h8_queue.py'):
        meta['files'][str(path)] = _sha(path)
    meta.update(profile=profile, q_agg=q_agg, q_agg_trained=trained,
                checkpoint_selection=f'prespecified_final_{FINAL_STEP}', execution=EXECUTION,
                protocol='paper: final checkpoint preselected, q_agg=min official / mean ablation')
    return meta
