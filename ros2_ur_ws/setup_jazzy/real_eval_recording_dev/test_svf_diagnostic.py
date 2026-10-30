import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from svf_diagnostic import validate_alignment, verified, sha, graph


class CriticTests(unittest.TestCase):
    def fixture(self):
        f = {'request/payload': [json.dumps({'request_id':7, 'control':{'cmd':'act', 'state':[0]*7}}).encode()],
             'cam1/timestamp_ns': np.array([100]), 'cam2/timestamp_ns': np.array([100]),
             'cam1/payload': [b'a'], 'cam2/payload': [b'b']}
        z = dict(state=np.zeros((1,7)), reset_counter=3, cam1_jpeg=[b'a'], cam2_jpeg=[b'b'],
                 chunk_t=np.array([0]), chunks=np.zeros((1,56)))
        return f, z, {'reset_reply':{'reset_counter':3}}

    def test_alignment_rejects_wrong_episode_or_input(self):
        f,z,m = self.fixture()
        self.assertEqual(validate_alignment(f,z,m)[2].tolist(), [0])
        for key, value in [('state',np.ones((1,7))), ('reset_counter',4),
                           ('cam1_jpeg',[b'wrong']), ('chunk_t',np.array([1]))]:
            changed = dict(z, **{key:value})
            with self.assertRaises(ValueError, msg=key):
                validate_alignment(f,changed,m)

    def test_hash_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'artifact'; p.write_bytes(b'original')
            files = {str(p):sha(p)}
            self.assertEqual(verified(p,files),p)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verified(p,files)

    def test_graph_constant_values_and_before_first_refill(self):
        rows=[dict(elapsed_s=.1,q1=-1,q2=-1,q_mean=-1)]
        for t in (0,.1,1):
            self.assertEqual(graph(rows,t,1).shape,(300,1280,3))

    def test_optional_diagnostic_failure_keeps_base_video_and_unlocks(self):
        import cv2
        from recording import EpisodeWriter
        from render import render
        with tempfile.TemporaryDirectory() as root:
            writer=EpisodeWriter(root, {'method':'svf','svf_diagnostic':True,
                                       'diagnostic_python':'/nonexistent/python'}, start_ns=1)
            jpeg=cv2.imencode('.jpg',np.zeros((20,20,3),np.uint8))[1].tobytes()
            writer.append_image('cam1',jpeg,2); writer.append_image('cam2',jpeg,2)
            writer.stop(timestamp_ns=100000001)
            path=writer.wait(5)
            render(path)
            self.assertTrue((path/'diagnostic.mp4').exists())
            self.assertFalse((path/'.tag.lock').exists())
            self.assertEqual(json.loads((path/'svf_diagnostic_status.json').read_text())['status'],'failed')


if __name__ == '__main__':
    unittest.main()
