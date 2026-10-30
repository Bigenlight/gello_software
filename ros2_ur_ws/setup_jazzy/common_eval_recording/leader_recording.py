"""New common-policy metadata; reuse the existing, unchanged ROS recorder."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

RECORDER = Path(__file__).resolve().parent.parent / 'real_eval_recording_dev'
sys.path.insert(0, str(RECORDER))
spec = importlib.util.spec_from_file_location('common_original_leader_recording', RECORDER / 'leader_recording.py')
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
instrumented_leader = original.instrumented_leader


def launch_metadata():
    profile = os.environ['COMMON_EVAL_PROFILE']
    if profile not in ('retrained-bc', 'retrained-ifql', 'retrained-dsrl'):
        raise ValueError(f'unsupported common recording profile: {profile}')
    run = Path(os.environ['IFQL_RUN_DIR']).resolve()
    step = int(os.environ['IFQL_STEP'])
    norm_path = Path(os.environ['IFQL_NORM_STATS']).resolve()
    package = Path(os.environ['COMMON_ENCODER_PACKAGE']).resolve()
    flags = json.loads((run / 'flags.json').read_text())
    norm = json.loads(norm_path.read_text())
    files = [run / 'flags.json', run / f'params_{step}.pkl', norm_path,
             Path(os.environ['IFQL_PARAMS_FILE']), Path(os.environ['IFQL_SERVER_PY']),
             package / 'encoder.pt', package / 'heads/carrot.pt', package / 'norm_stats.npz',
             Path(__file__).resolve(), *RECORDER.glob('*.py')]
    digests = {}
    for path in files:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        digests[str(path)] = h.hexdigest()
    return {'render': True, 'method': {'retrained-bc': 'flow_bc', 'retrained-ifql': 'ifql',
                                      'retrained-dsrl': 'dsrl'}[profile],
            'profile': profile, 'run_dir': str(run), 'step': step,
            'sampler': os.environ['IFQL_SAMPLER'], 'num_samples': int(os.environ['IFQL_NUM_SAMPLES']),
            'q_agg': 'mean' if profile == 'retrained-ifql' else None,
            'files': digests, 'flags': flags, 'encoder_identity': norm.get('package'),
            'action_normalization': norm['action_normalization']}
