"""CPU contract tests for the opt-in contiguous-H8 serving queue."""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
NATIVE = Path("/home/junhyeong/svf_real_carrot_commonenc_20260919/source/vision_carrot/action_queue.py")


def _load_native(name: str):
    spec = importlib.util.spec_from_file_location(name, NATIVE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_helper():
    spec = importlib.util.spec_from_file_location("contiguous_h8_queue_test_helper", ROOT / "contiguous_h8_queue.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _norm(native):
    return native.ActionNorm(np.zeros(7), np.ones(7))


def test_native_stride3_contract_is_unchanged_in_separate_module():
    native = _load_native("native_stride3_queue")
    assert native.STRIDE == 3
    assert native.QUEUE_LEN == 24
    knots = np.arange(56, dtype=float).reshape(8, 7)
    frames = native.knots_to_frames(knots)
    assert frames.shape == (24, 7)
    np.testing.assert_allclose(frames[1, :6], (2 * knots[0, :6] + knots[1, :6]) / 3)
    assert frames[22, 0] == knots[7, 0]


def test_install_reports_contiguous_metadata_and_exact_eight_rows():
    native = _load_native("native_contiguous_queue")
    helper = _load_helper()
    server = types.SimpleNamespace(aq=native)
    installed = helper.install_contiguous_h8(server)
    assert installed is native.ActionQueue
    assert helper.queue_metadata() == {"H": 8, "stride": 1, "interp": 8, "hold": 0}
    assert (native.H, native.STRIDE, native.N_INTERP, native.N_HOLD, native.QUEUE_LEN) == (8, 1, 8, 0, 8)

    q = native.ActionQueue(_norm(native))
    chunk = np.arange(56, dtype=float) / 56.0
    knots = q.refill(chunk)
    assert knots.shape == (8, 7)
    assert len(q) == 8
    np.testing.assert_allclose(np.stack([q.pop() for _ in range(8)]), knots)
    assert q.empty()


def test_joint_reference_unwrap_is_preserved_and_refill_boundary_is_eight_acts():
    native = _load_native("native_unwrap_queue")
    helper = _load_helper()
    helper.install_contiguous_h8(native)  # accepts server.aq directly
    norm = native.ActionNorm(np.full(7, -np.pi), np.full(7, np.pi))
    q = native.ActionQueue(norm)

    # Normalized -pi-ish targets should be unwrapped to the live +pi branch.
    chunk = np.zeros(56, dtype=float)
    chunk.reshape(8, 7)[:, 0] = -0.99
    first = q.refill(chunk, joint_reference=np.array([np.pi - 0.01, 0, 0, 0, 0, 0]))
    assert np.all(first[:, 0] > 3.0)
    first_rows = np.stack([q.pop() for _ in range(8)])
    assert q.empty()

    # A fresh refill starts exactly at the ninth ACT, with no interpolation or
    # stale hold rows crossing the boundary.
    second_chunk = np.ones(56, dtype=float) * 0.5
    second = q.refill(second_chunk, joint_reference=np.zeros(6))
    second_rows = np.stack([q.pop() for _ in range(8)])
    np.testing.assert_allclose(second_rows, second)
    assert not np.allclose(first_rows[-1], second_rows[0])
    assert q.n_refill == 2
