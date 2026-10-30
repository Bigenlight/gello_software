"""Non-ROS START/HOLD test using installed common metadata and existing recorder."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

WS = Path('/home/junhyeong/gello_software_jazzy/ros2_ur_ws')
SHIM = WS / 'setup_jazzy/common_eval_recording/leader_recording.py'
spec = importlib.util.spec_from_file_location('common_recording_shim_test', SHIM)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
from test_leader_recording import Base
import h5py


def main():
    os.environ.update(IFQL_PARAMS_FILE=str(WS / 'src/gello_policy/config/ifql_deploy.yaml'),
        IFQL_NORM_STATS='/home/junhyeong/ifql_common_lighted_20260919/carrot_data/ifql_export_manifest.json',
        COMMON_ENCODER_PACKAGE='/home/junhyeong/common_encoder_lighted_20260919/carrot_seed0')
    results = []
    for profile, method, sampler, count, step, family in (
        ('bc', 'flow_bc', 'bc', 1, 40000, 'ifql'),
        ('ifql', 'ifql', 'bon', 32, 40000, 'ifql'),
        ('dsrl', 'dsrl', 'actor', 1, 100000, 'dsrl')):
        os.environ.update(COMMON_EVAL_PROFILE='retrained-' + profile,
            IFQL_RUN_DIR=f'/home/junhyeong/real_common_baselines_20260919/{family}_seed0',
            IFQL_STEP=str(step), IFQL_SAMPLER=sampler, IFQL_NUM_SAMPLES=str(count),
            IFQL_SERVER_PY=str(WS / f'setup_jazzy/common_retrained_{family}_server.py'))
        metadata = shim.launch_metadata()
        assert metadata['method'] == method and metadata['step'] == step
        assert metadata['profile'] == 'retrained-' + profile
        assert len(metadata['files']) >= 8 and metadata['encoder_identity']
        with tempfile.TemporaryDirectory(prefix='common-recording-test-') as root:
            node = shim.instrumented_leader(Base, 'EXECUTE', root, {**metadata, 'render': False})()
            assert node._try_start_execution()[0]
            node._zmq_roundtrip([b'{"cmd":"act","state":[0,0,0,0,0,0,0]}', b'jpeg1', b'jpeg2'])
            node._publish_arm([1, 2, 3, 4, 5, 5.5])
            node._srv_hold(None, SimpleNamespace())
            episode = node._eval_finished[-1].wait(5)
            saved = json.loads((episode / 'episode.json').read_text())
            assert saved['metadata']['method'] == method and saved['termination_reason'] == 'operator_hold'
            assert saved['accepted_start'] and saved['status'] == 'ready'
            with h5py.File(episode / 'recording.h5') as f:
                assert len(f['request/payload']) == 1
                assert len(f['leader_command/payload']) == 1
            assert json.loads((episode / 'outcome.json').read_text())['outcome'] == 'unlabeled', 'HOLD must not assign outcome'
            node.destroy_node()
        results.append(profile)
    print(json.dumps({'status': 'PASS', 'metadata_and_start_hold': results, 'robot': False}))


if __name__ == '__main__':
    main()
