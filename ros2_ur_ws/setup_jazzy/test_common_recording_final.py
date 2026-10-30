"""Non-ROS metadata + synthetic START/HOLD check for the final100k common IFQL profiles (env from launcher)."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

WS = Path('/home/junhyeong/gello_software_jazzy/ros2_ur_ws')
SHIM = WS / 'setup_jazzy/common_eval_recording_final/leader_recording.py'
spec = importlib.util.spec_from_file_location('common_final_recording_shim_test', SHIM)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
from test_leader_recording import Base
import h5py


def main():
    profile = os.environ['COMMON_EVAL_PROFILE']
    metadata = shim.launch_metadata()
    assert metadata['method'] == 'ifql' and metadata['step'] == 100000
    assert metadata['profile'] == profile and metadata['q_agg'] == shim.PROFILES[profile]
    assert metadata['q_agg_trained'] == 'mean' and metadata['execution'] == shim.EXECUTION
    assert str(WS / 'setup_jazzy/common_retrained_ifql_final_server.py') in metadata['files']
    assert str(SHIM) in metadata['files'] and len(metadata['files']) >= 11
    with tempfile.TemporaryDirectory(prefix='common-final-recording-test-') as root:
        node = shim.instrumented_leader(Base, 'EXECUTE', root, {**metadata, 'render': False})()
        assert node._try_start_execution()[0]
        node._zmq_roundtrip([b'{"cmd":"act","state":[0,0,0,0,0,0,0]}', b'jpeg1', b'jpeg2'])
        node._publish_arm([1, 2, 3, 4, 5, 5.5])
        node._srv_hold(None, SimpleNamespace())
        episode = node._eval_finished[-1].wait(5)
        saved = json.loads((episode / 'episode.json').read_text())
        assert saved['metadata']['method'] == 'ifql' and saved['metadata']['q_agg'] == metadata['q_agg']
        assert saved['metadata']['step'] == 100000 and saved['termination_reason'] == 'operator_hold'
        assert saved['accepted_start'] and saved['status'] == 'ready'
        with h5py.File(episode / 'recording.h5') as f:
            assert len(f['request/payload']) == 1 and len(f['leader_command/payload']) == 1
        assert json.loads((episode / 'outcome.json').read_text())['outcome'] == 'unlabeled'
        node.destroy_node()
    print(json.dumps({'recording_check': 'PASS', 'profile': profile, 'q_agg': metadata['q_agg'],
                      'step': metadata['step'], 'robot_started': False}))


if __name__ == '__main__':
    main()
