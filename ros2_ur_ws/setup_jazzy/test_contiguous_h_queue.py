"""Unit tests for contiguous_h_queue against the native action_queue module (numpy only)."""
import importlib, sys, types, unittest
from pathlib import Path
import numpy as np

SRC = Path('/home/junhyeong/svf_real_carrot_commonenc_20260919/source/vision_carrot')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contiguous_h_queue import install_contiguous_h, install_episode_logger_h, queue_metadata


def fresh_aq():
    sys.path.insert(0, str(SRC))
    sys.modules.pop('action_queue', None)
    return importlib.import_module('action_queue')


class Test(unittest.TestCase):
    def _norm(self, aq):
        return aq.ActionNorm(lo=[-4, -2, 1, -3, -2, -4, 0.0], hi=[-2, 0, 3, -1, 0, -2, 1.0])

    def test_h5_and_h10_direct_rows(self):
        for H in (5, 10):
            aq = fresh_aq()
            server = types.SimpleNamespace(aq=aq)
            cls = install_contiguous_h(server, H)
            self.assertEqual(aq.H, H); self.assertEqual(aq.CHUNK_DIM, 7 * H); self.assertEqual(aq.QUEUE_LEN, H)
            self.assertEqual(aq.STRIDE, 1); self.assertEqual(aq.N_HOLD, 0); self.assertEqual(aq.N_INTERP, H)
            self.assertEqual(queue_metadata(H), {'H': H, 'stride': 1, 'interp': H, 'hold': 0})
            norm = self._norm(aq)
            q = aq.ActionQueue(norm)
            self.assertIsInstance(q, cls)
            rng = np.random.default_rng(H)
            chunk = rng.uniform(-1, 1, 7 * H)
            knots = q.refill(chunk)
            self.assertEqual(knots.shape, (H, 7)); self.assertEqual(len(q), H)
            expect = norm.denormalize(chunk.reshape(H, 7))
            for i in range(H):
                np.testing.assert_allclose(q.pop(), expect[i])
            self.assertTrue(q.empty())
            with self.assertRaises(ValueError):
                q.refill(np.zeros(56))

    def test_branch_shift(self):
        aq = fresh_aq(); install_contiguous_h(types.SimpleNamespace(aq=aq), 5)
        norm = self._norm(aq); q = aq.ActionQueue(norm)
        knots = q.refill(np.zeros(35), joint_reference=np.array([-3 + 2 * np.pi, -1, 2, -2, -1, -3 + 2 * np.pi]))
        self.assertAlmostEqual(knots[0, 0], -3 + 2 * np.pi); self.assertAlmostEqual(knots[0, 5], -3 + 2 * np.pi)
        self.assertAlmostEqual(knots[0, 1], -1)

    def test_refuses_conflicting_install(self):
        aq = fresh_aq(); ns = types.SimpleNamespace(aq=aq)
        install_contiguous_h(ns, 10); install_contiguous_h(ns, 10)
        with self.assertRaises(RuntimeError):
            install_contiguous_h(ns, 5)
        with self.assertRaises(ValueError):
            install_contiguous_h(types.SimpleNamespace(aq=fresh_aq()), 7)

    def test_h8_matches_existing_helper(self):
        from contiguous_h8_queue import install_contiguous_h8
        a = fresh_aq(); install_contiguous_h8(types.SimpleNamespace(aq=a))
        b = fresh_aq(); install_contiguous_h(types.SimpleNamespace(aq=b), 8)
        chunk = np.random.default_rng(0).uniform(-1, 1, 56)
        na, nb = self._norm(a), self._norm(b)
        qa, qb = a.ActionQueue(na), b.ActionQueue(nb)
        np.testing.assert_array_equal(qa.refill(chunk), qb.refill(chunk))
        for _ in range(8):
            np.testing.assert_array_equal(qa.pop(), qb.pop())
        self.assertEqual((a.H, a.STRIDE, a.N_INTERP, a.N_HOLD, a.QUEUE_LEN, a.CHUNK_DIM),
                         (b.H, b.STRIDE, b.N_INTERP, b.N_HOLD, b.QUEUE_LEN, b.CHUNK_DIM))


class TestLogger(unittest.TestCase):
    def test_episode_logger_accepts_h_chunks(self):
        sys.path.insert(0, str(SRC)); sys.modules.pop('episode_logger', None)
        el = importlib.import_module('episode_logger')
        ns = types.SimpleNamespace(EpisodeLogger=el.EpisodeLogger)
        import tempfile
        tmp = tempfile.mkdtemp()
        native = el.EpisodeLogger(tmp, async_write=False); native.start_episode(1, 0)
        with self.assertRaises(ValueError):
            native.log_refill(np.zeros(35), np.zeros((5, 7)), 0, 1.0)  # unpatched: 56-only
        install_episode_logger_h(ns, 5)
        lg = el.EpisodeLogger(tmp, async_write=False)
        lg.start_episode(1, 0)
        lg.log_refill(np.zeros(35), np.zeros((5, 7)), 0, 1.0)
        self.assertEqual(lg._ep.chunks[0].shape, (35,)); self.assertEqual(lg._ep.chunks_denorm[0].shape, (5, 7))
        with self.assertRaises(ValueError):
            lg.log_refill(np.zeros(56), np.zeros((8, 7)), 5, 1.0)
        with self.assertRaises(RuntimeError):
            install_episode_logger_h(ns, 10)


if __name__ == '__main__':
    unittest.main()
