"""Focused contracts for the offline eval diagnostic renderer."""
import json

import cv2
import numpy as np
import pytest

from sim_collect.eval.render_eval_video import (
    OUTPUT_SIZE,
    Diagnostic,
    EpisodeMeta,
    JoinError,
    _chunk_status,
    _join_diagnostics,
    _nearest_rows,
    _q_history_values,
    _synthetic_diagnostics,
    draw_panel,
    load_policy_episode,
)


def _meta(n_steps=48, reset_counter=7):
    return EpisodeMeta("101", 101, "success", n_steps, "IN r=0.01", None, None, {},
                       {"reset_counter": reset_counter})


def _indices(n=48):
    request = np.arange(n, dtype=np.int64)
    chunk_id = np.repeat(np.arange((n + 23) // 24), 24)[:n]
    chunk_step = np.tile(np.arange(24), (n + 23) // 24)[:n]
    chunk_t = np.arange(0, n, 24, dtype=np.int64)
    return request, chunk_t, chunk_id, chunk_step


def _q_row(request_idx, decision_idx, q, **extra):
    return {
        "request_idx": request_idx,
        "decision_idx": decision_idx,
        "K": 32,
        "q_chosen": q,
        "q_mean": q - 0.1,
        "q_std": 0.2,
        "q_min": q - 0.4,
        "q_max": q,
        "q_spread": 0.4,
        "encode_ms": 10.0,
        "sample_ms": 20.0,
        "refill_ms": 31.0,
        **extra,
    }


def test_v2_join_is_discrete_and_holds_q_until_next_boundary():
    request, chunk_t, chunk_id, chunk_step = _indices()
    rows = [_q_row(0, 0, -1.0), _q_row(24, 1, -0.2)]
    joined, notes = _join_diagnostics(request, chunk_t, chunk_id, chunk_step, rows)
    assert notes == []
    assert joined[0].boundary and joined[0].sidecar_updated
    assert joined[0].q["q_chosen"] == -1.0
    assert not joined[23].boundary and joined[23].q["q_chosen"] == -1.0
    assert joined[24].boundary and joined[24].q["q_chosen"] == -0.2
    assert joined[-1].decision_idx == 1 and joined[-1].request_idx == 47


def test_missing_boundary_row_never_claims_critic_update():
    request, chunk_t, chunk_id, chunk_step = _indices()
    joined, _ = _join_diagnostics(
        request, chunk_t, chunk_id, chunk_step, [_q_row(0, 0, -0.4, refill=91)]
    )
    assert joined[24].boundary
    assert joined[24].decision_idx == 1
    assert joined[24].q["q_chosen"] == -0.4
    assert joined[24].sidecar_refill == 91
    assert not joined[24].sidecar_updated
    assert _chunk_status(joined[24]) == "NEW CHUNK / Q NOT LOGGED - PREVIOUS Q HELD"
    assert _chunk_status(joined[25]) == "PREVIOUS CHUNK Q HELD"


def test_boundary_row_without_q_is_not_a_critic_update():
    request, chunk_t, chunk_id, chunk_step = _indices()
    row = {"request_idx": 0, "decision_idx": 0, "refill": 12, "refill_ms": 3.0}
    joined, _ = _join_diagnostics(request, chunk_t, chunk_id, chunk_step, [row])
    assert joined[0].sidecar_updated and joined[0].q is None
    assert joined[0].sidecar_refill == 12
    assert _chunk_status(joined[0]) == "NEW CHUNK / Q NOT LOGGED"


def test_legacy_join_uses_t_index_but_never_offsets_global_refill():
    request, chunk_t, chunk_id, chunk_step = _indices()
    rows = []
    for req, refill, q in ((0, 91, -0.9), (24, 92, -0.1)):
        row = _q_row(req, req // 24, q, refill=refill)
        row["t_index"] = row.pop("request_idx")
        row.pop("decision_idx")
        rows.append(row)
    joined, notes = _join_diagnostics(request, chunk_t, chunk_id, chunk_step, rows)
    assert "legacy" in notes[0]
    assert joined[0].decision_idx == 0 and joined[0].sidecar_refill == 91
    assert joined[24].decision_idx == 1 and joined[24].sidecar_refill == 92
    assert joined[0].sidecar_updated and joined[24].sidecar_updated


@pytest.mark.parametrize("row", [
    _q_row(1, 0, -0.5),
    _q_row(24, 0, -0.5),
    {**_q_row(24, 1, -0.5), "t_index": 23},
])
def test_join_rejects_shifted_or_inconsistent_indices(row):
    with pytest.raises(JoinError):
        _join_diagnostics(*_indices(), [row])


def test_policy_npz_selection_uses_exact_reset_counter_and_object_jpegs(tmp_path):
    n = 24
    image = np.full((12, 16, 3), 127, np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpegs = np.empty(n, dtype=object)
    jpegs[:] = [encoded.tobytes()] * n
    np.savez(
        tmp_path / "ep_0007.npz",
        cam1_jpeg=jpegs,
        cam2_jpeg=jpegs,
        state=np.zeros((n, 7), np.float32),
        action=np.zeros((n, 7), np.float32),
        chunk_id=np.zeros(n, np.int32),
        chunk_step=np.arange(n, dtype=np.int32),
        chunk_t=np.array([0], np.int32),
        reset_counter=np.int64(7),
    )
    row = {"reset_counter": 7, "refill": 55, "t_index": 0, **_q_row(0, 0, -0.3)}
    row.pop("request_idx")
    row.pop("decision_idx")
    (tmp_path / "refill_stats.jsonl").write_text(json.dumps(row) + "\n")
    loaded = load_policy_episode(tmp_path, _meta(n_steps=n))
    assert loaded.path.name == "ep_0007.npz"
    assert len(loaded.cam1_jpeg) == n
    assert loaded.diagnostics[-1].q["q_chosen"] == -0.3
    assert any("request_idx absent" in note for note in loaded.notes)


def test_policy_log_without_reset_metadata_requires_explicit_map(tmp_path):
    meta = EpisodeMeta("x", None, "timeout", 1, "", None, None, {}, {})
    with pytest.raises(JoinError, match="--reset-map"):
        load_policy_episode(tmp_path, meta)


def test_nearest_h5_rows_and_panel_contract():
    rows = _nearest_rows(np.arange(0.0, 1.001, 0.008), 30, 30)
    assert len(rows) == 30 and rows[0] == 0 and rows[-1] > rows[0]
    panel = draw_panel(_meta(), Diagnostic(3, -1, -1, -1, False), 0.25, "H5 kinematic replay")
    assert panel.shape == (OUTPUT_SIZE[1], 480, 3)
    assert panel.dtype == np.uint8 and np.unique(panel.reshape(-1, 3), axis=0).shape[0] > 10


def test_q_graph_values_and_image_are_deterministic():
    history = [
        Diagnostic(0, 0, 0, 0, True, {"q_chosen": -1.0}, sidecar_updated=True),
        Diagnostic(1, 0, 0, 1, False, {"q_chosen": -1.0}),
        Diagnostic(2, 0, 0, 2, False, {"q_chosen": -0.5}),
    ]
    np.testing.assert_array_equal(_q_history_values(history), [-1.0, -1.0, -0.5])
    panel_a = draw_panel(_meta(), history[-1], 0.25, "policy JPEG", history, 30)
    panel_b = draw_panel(_meta(), history[-1], 0.25, "policy JPEG", history, 30)
    assert np.array_equal(panel_a, panel_b)
    graph = panel_a[72:167, 238:469]
    assert np.unique(graph.reshape(-1, 3), axis=0).shape[0] > 20


def test_q_graph_missing_values_are_not_fabricated():
    history = [Diagnostic(i, -1, -1, -1, False) for i in range(3)]
    assert np.isnan(_q_history_values(history)).all()
    panel = draw_panel(_meta(), history[-1], None, "H5 kinematic replay", history, 30)
    graph = panel[72:167, 238:469]
    # The unavailable box still contains text, but no yellow temporal trace.
    yellow_trace = (graph[:, :, 0] == 80) & (graph[:, :, 1] == 215) & (graph[:, :, 2] == 255)
    assert not yellow_trace.any()


def test_synthetic_diagnostics_mark_only_real_synthetic_boundaries_updated():
    diagnostics = _synthetic_diagnostics(25)
    assert diagnostics[0].boundary and diagnostics[0].sidecar_updated
    assert _chunk_status(diagnostics[0]) == "NEW CHUNK / CRITIC UPDATED"
    assert not diagnostics[1].sidecar_updated
    assert diagnostics[24].boundary and diagnostics[24].sidecar_updated
    assert _chunk_status(diagnostics[24]) == "NEW CHUNK / CRITIC UPDATED"
