"""Scene tag in the episode folder name (2026-09-22): placement_metadata + recording.tag_episode."""
import json, os, sys, tempfile, unittest, uuid
from pathlib import Path
sys.path.insert(0, '/home/junhyeong/gello_software_jazzy/ros2_ur_ws')
sys.path.insert(0, '/home/junhyeong/gello_software_jazzy/ros2_ur_ws/setup_jazzy/real_eval_recording_dev')
from placement_metadata import scene_tag, scene_name, PlacementMetadata
from recording import tag_episode

CUBE = {'task': 'cube', 'enabled': True, 'mode': 'single_episode', 'episode_indices': [59],
        'selected_take': 'take_59_20260919_215546', 'split': 'all'}
TRIPLE = dict(CUBE, task='bowl_triple', selected_take='take_27_20260919_195422_recover')
MEAN = {'task': 'carrot', 'enabled': True, 'mode': 'split_mean', 'split': 'train',
        'episode_indices': list(range(56)), 'selected_take': None}


def make_episode(root, name, identity, status='ready', start_ns=1000):
    ep = Path(root) / name
    ep.mkdir(parents=True)
    (ep / 'episode.json').write_text(json.dumps({'episode_id': identity, 'status': status,
                                                 'accepted_start': True, 'start_ns': start_ns,
                                                 'stop_ns': start_ns + 10, 'counts': {'cam1': 3},
                                                 'metadata': {}}))
    (ep / 'outcome.json').write_text(json.dumps({'episode_id': identity, 'outcome': 'unlabeled', 'history': []}))
    return ep


class SceneTagTest(unittest.TestCase):
    def test_tag_text(self):
        self.assertEqual(scene_tag(CUBE), 'cube-ep59-take59')
        self.assertEqual(scene_tag(TRIPLE), 'bowl_triple-ep59-take27')   # index 59, take 27
        self.assertEqual(scene_tag(MEAN), 'carrot-trainmean56')
        self.assertEqual(scene_tag(dict(CUBE, enabled=False)), 'cube-off')
        self.assertIsNone(scene_tag({'enabled': True}))
        self.assertIsNone(scene_tag(None))
        self.assertEqual(scene_tag({'task': '표/기 x', 'enabled': False}), 'x-off')  # path-safe only

    def test_name_insert_and_replace(self):
        base = 'ep_20260922_145958.203283_KST_' + str(uuid.uuid4()) + '__unlabeled'
        tagged = scene_name(base, 'cube-ep59-take59')
        self.assertTrue(tagged.endswith('__cube-ep59-take59__unlabeled'))
        self.assertEqual(scene_name(tagged, 'cube-ep59-take59'), tagged)      # idempotent
        self.assertTrue(scene_name(tagged, 'bowl-off').endswith('__bowl-off__unlabeled'))
        self.assertEqual(scene_name(base, None), base)

    def test_tag_episode_preserves_scene(self):
        with tempfile.TemporaryDirectory() as root:
            ident = str(uuid.uuid4())
            ep = make_episode(root, f'ep_20260922_120000.000000_KST_{ident}__cube-ep59-take59__unlabeled', ident)
            out = tag_episode(ep, 'success', labeler='test')
            self.assertEqual(Path(out).name, f'ep_20260922_120000.000000_KST_{ident}__cube-ep59-take59__success')
            self.assertEqual(json.loads((Path(out) / 'outcome.json').read_text())['success_score'], 1.0)
            out2 = tag_episode(out, 'partial_success', labeler='test')
            self.assertTrue(Path(out2).name.endswith('__cube-ep59-take59__partial_success'))

    def test_tag_episode_without_scene_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            ident = str(uuid.uuid4())
            ep = make_episode(root, f'ep_20260922_120000.000000_KST_{ident}__unlabeled', ident)
            out = tag_episode(ep, 'failure', labeler='test')
            self.assertEqual(Path(out).name, f'ep_20260922_120000.000000_KST_{ident}__failure')

    def test_tag_episode_rejects_foreign_identity(self):
        with tempfile.TemporaryDirectory() as root:
            ident = str(uuid.uuid4())
            ep = make_episode(root, f'ep_20260922_120000.000000_KST_{uuid.uuid4()}__cube-ep59__unlabeled', ident)
            with self.assertRaises(ValueError):
                tag_episode(ep, 'success')

    def test_placement_process_renames_and_writes(self):
        with tempfile.TemporaryDirectory() as hub:
            run = Path(hub) / 'run'
            (run).mkdir()
            (run / 'launch_manifest.json').write_text(json.dumps({'policy': {}, 'recording': {}}))
            ident = str(uuid.uuid4())
            ep = make_episode(run / 'episodes', f'ep_20260922_120000.000000_KST_{ident}__unlabeled', ident, start_ns=1500)
            pm = PlacementMetadata(hub)
            pm.enqueue(run, 1000, 2000, CUBE)
            pm.process()
            renamed = ep.with_name(f'ep_20260922_120000.000000_KST_{ident}__cube-ep59-take59__unlabeled')
            self.assertTrue(renamed.is_dir(), pm.message)
            self.assertFalse(ep.exists())
            self.assertEqual(json.loads((renamed / 'placement_reference.json').read_text())['task'], 'cube')
            self.assertFalse((renamed / '.tag.lock').exists())
            self.assertEqual(list(pm.directory.glob('*.json')), [])
            # relabel keeps the scene segment
            out = tag_episode(renamed, 'failure', labeler='test')
            self.assertTrue(Path(out).name.endswith('__cube-ep59-take59__failure'))

    def test_placement_process_keeps_name_on_collision(self):
        with tempfile.TemporaryDirectory() as hub:
            run = Path(hub) / 'run'
            run.mkdir()
            (run / 'launch_manifest.json').write_text(json.dumps({'policy': {}, 'recording': {}}))
            ident = str(uuid.uuid4())
            ep = make_episode(run / 'episodes', f'ep_20260922_120000.000000_KST_{ident}__unlabeled', ident, start_ns=1500)
            (run / 'episodes' / f'ep_20260922_120000.000000_KST_{ident}__cube-ep59-take59__unlabeled').mkdir()
            pm = PlacementMetadata(hub)
            pm.enqueue(run, 1000, 2000, CUBE)
            pm.process()
            self.assertTrue(ep.is_dir())                       # original untouched
            self.assertTrue((ep / 'placement_reference.json').exists())
            self.assertFalse((ep / '.tag.lock').exists())


if __name__ == '__main__':
    unittest.main()
