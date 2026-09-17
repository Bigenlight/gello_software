"""Focused contracts for the offline eval diagnostic renderer."""
import json
import sys
import types

import cv2
import h5py
import numpy as np
import pytest

from sim_collect.eval.render_eval_video import (
    OUTPUT_SIZE,
    Diagnostic,
    EpisodeMeta,
    JoinError,
    _chunk_status,
    _join_diagnostics,
    _load_refill_rows,
    _nearest_rows,
    _q_history_values,
    _step_trace_points,
    _q_history_series,
    _real_diagnostics,
    _real_elapsed_s,
    _synthetic_diagnostics,
    _chosen_q_heads,
    _resolve_ffmpeg,
    draw_dashboard,
    draw_real_dashboard,
    load_real_h5_episode,
    load_policy_episode,
    render_real_h5_episode,
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


def test_q_heads_select_exact_candidate_for_each_critic_head():
    request, chunk_t, chunk_id, chunk_step = _indices(n=24)
    row = _q_row(0, 0, -0.2, K=3, q_argmax=2,
                 q_heads=[[-0.7, -0.5, -0.1], [-0.8, -0.4, -0.3]])
    joined, _ = _join_diagnostics(request, chunk_t, chunk_id, chunk_step, [row])
    assert _chosen_q_heads(joined[0].q) == (-0.1, -0.3)
    np.testing.assert_array_equal(_q_history_series(joined[:3], "q1"), [-0.1, -0.1, -0.1])
    np.testing.assert_array_equal(_q_history_series(joined[:3], "q2"), [-0.3, -0.3, -0.3])


@pytest.mark.parametrize("row", [
    _q_row(0.5, 0, -0.2),
    _q_row("0", 0, -0.2),
    _q_row(0, False, -0.2),
    _q_row(0, 0, -0.2, refill=1.0),
    _q_row(0, 0, -0.2, K=2, q_argmax=1.9,
           q_heads=[[-0.2, -0.1], [-0.3, -0.2]]),
])
def test_discrete_sidecar_fields_reject_float_string_and_bool(row):
    with pytest.raises(JoinError, match="integer token"):
        _join_diagnostics(*_indices(n=24), [row])


def test_reset_counter_requires_json_integer_token(tmp_path):
    sidecar = tmp_path / "refill_stats.jsonl"
    sidecar.write_text(json.dumps({"reset_counter": 7.0, "request_idx": 0}) + "\n")
    with pytest.raises(JoinError, match="reset_counter.*integer token"):
        _load_refill_rows(sidecar, 7)


@pytest.mark.parametrize("extra", [
    {"q_heads": [[-0.1, -0.2]], "q_argmax": 3},
    {"q_heads": [[-0.1, -0.2]], "q_argmax": -1},
    {"q_heads": [[-0.1], [float("nan")]], "q_argmax": 0},
    {"q_heads": [[-0.1, -0.2]], "K": 32, "q_argmax": 0},
])
def test_q_heads_reject_malformed_or_misaligned_candidate_axis(extra):
    request, chunk_t, chunk_id, chunk_step = _indices(n=24)
    with pytest.raises(JoinError, match="q_heads|q_argmax|K="):
        _join_diagnostics(request, chunk_t, chunk_id, chunk_step, [_q_row(0, 0, -0.2, **extra)])


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


def test_nearest_h5_rows_and_dashboard_contract():
    rows = _nearest_rows(np.arange(0.0, 1.001, 0.008), 30, 30)
    assert len(rows) == 30 and rows[0] == 0 and rows[-1] > rows[0]
    dashboard = draw_dashboard(_meta(), Diagnostic(3, -1, -1, -1, False), 0.25, "H5 kinematic replay")
    assert dashboard.shape == (360, OUTPUT_SIZE[0], 3)
    assert dashboard.dtype == np.uint8 and np.unique(dashboard.reshape(-1, 3), axis=0).shape[0] > 10


def test_q_graph_values_and_image_are_deterministic():
    history = [
        Diagnostic(0, 0, 0, 0, True, {"q_chosen": -1.0}, sidecar_updated=True),
        Diagnostic(1, 0, 0, 1, False, {"q_chosen": -1.0}),
        Diagnostic(2, 0, 0, 2, False, {"q_chosen": -0.5}),
    ]
    np.testing.assert_array_equal(_q_history_values(history), [-1.0, -1.0, -0.5])
    dashboard_a = draw_dashboard(_meta(), history[-1], 0.25, "policy JPEG", history, 30)
    dashboard_b = draw_dashboard(_meta(), history[-1], 0.25, "policy JPEG", history, 30)
    assert np.array_equal(dashboard_a, dashboard_b)
    graph = dashboard_a[52:341, 14:791]
    assert np.unique(graph.reshape(-1, 3), axis=0).shape[0] > 20


def test_q_graph_trace_is_zero_order_hold_at_value_changes():
    values = np.asarray([-1.0, -1.0, -0.5], dtype=np.float64)
    paths = _step_trace_points(values, lambda i, value: (i * 10, int(round(value * 10))))
    assert paths == [[(0, -10), (10, -10), (10, -10), (20, -10), (20, -5)]]
    # In particular, the old value is extended to x=20 before the vertical
    # jump; there is no direct diagonal (10, -10) -> (20, -5).


def test_q_graph_missing_values_are_not_fabricated():
    history = [Diagnostic(i, -1, -1, -1, False) for i in range(3)]
    assert np.isnan(_q_history_values(history)).all()
    dashboard = draw_dashboard(_meta(), history[-1], None, "H5 kinematic replay", history, 30)
    graph = dashboard[109:312, 62:777]
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
    q1, q2 = _chosen_q_heads(diagnostics[24].q)
    assert q1 is not None and q2 is not None and q1 != q2


def test_synthetic_watermark_uses_reserved_dashboard_banner_only():
    history = _synthetic_diagnostics(3)
    plain = draw_dashboard(_meta(), history[-1], 0.5, "H5 kinematic replay", history, 30)
    marked = draw_dashboard(
        _meta(), history[-1], 0.5, "H5 kinematic replay", history, 30, synthetic=True
    )
    assert not np.array_equal(plain[:36], marked[:36])
    # Watermark must not touch the graph or right-side information region.
    np.testing.assert_array_equal(plain[36:], marked[36:])


def _real_h5(tmp_path, policy, *, include_optional=True, include_q=False):
    """Create a compact, self-contained common real_eval/v1 fixture."""
    path = tmp_path / f"ep_{policy.lower()}.h5"
    image = np.full((16, 24, 3), 80, np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = np.frombuffer(encoded.tobytes(), dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "real_eval/v1"
        f.attrs["complete"] = True
        f.attrs["policy"] = policy
        f.attrs["checkpoint"] = "ckpt-123"
        f.attrs["sampler"] = "ddim"
        f.attrs["finalize_reason"] = "operator_stop"
        requests = f.create_group("requests")
        requests.create_dataset("request_idx", data=np.asarray([10, 13, 20], dtype=np.int64))
        vlen = h5py.vlen_dtype(np.dtype("uint8"))
        cams1 = requests.create_dataset("cam1_jpeg", (3,), dtype=vlen)
        cams2 = requests.create_dataset("cam2_jpeg", (3,), dtype=vlen)
        for i in range(3):
            cams1[i] = jpeg
            cams2[i] = jpeg
        if include_optional:
            requests.create_dataset("state", data=np.zeros((3, 7), dtype=np.float32))
            requests.create_dataset("output", data=np.ones((3, 7), dtype=np.float32))
            requests.create_dataset("gripper", data=np.asarray([0.0, 0.5, 1.0], dtype=np.float32))
            requests.create_dataset("chunk_step", data=np.asarray([0, 1, 0], dtype=np.int64))
            requests.create_dataset("t_rel_s", data=np.asarray([0.0, 0.1, 0.2], dtype=np.float64))
        decisions = f.create_group("decisions")
        decisions.create_dataset("request_idx", data=np.asarray([10, 20], dtype=np.int64))
        decisions.create_dataset("decision_idx", data=np.asarray([3, 4], dtype=np.int64))
        if policy == "IFQL":
            decisions.create_dataset("K", data=np.asarray([8, 8], dtype=np.int64))
            decisions.create_dataset("chosen_idx", data=np.asarray([2, 4], dtype=np.int64))
            decisions.create_dataset("q_heads", data=np.asarray([
                [[-2.3, -1.9, -1.7, -1.4, -1.1, -0.8, -0.6, -0.4],
                 [-2.2, -2.0, -1.8, -1.5, -1.2, -0.9, -0.7, -0.5]],
                [[-1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1],
                 [-1.2, -1.0, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1]],
            ], dtype=np.float32))
            decisions.create_dataset("q_chosen", data=np.asarray([-1.65, -0.2], dtype=np.float32))
            decisions.create_dataset("q_mean", data=np.asarray([-1.25, -0.6], dtype=np.float32))
            decisions.create_dataset("q_std", data=np.asarray([0.5, 0.3], dtype=np.float32))
            decisions.create_dataset("q_min", data=np.asarray([-2.0, -1.0], dtype=np.float32))
            decisions.create_dataset("q_max", data=np.asarray([-0.5, -0.1], dtype=np.float32))
            decisions.create_dataset("q_spread", data=np.asarray([1.5, 0.9], dtype=np.float32))
            decisions.create_dataset("candidate_norm", data=np.asarray([3.0, 4.0], dtype=np.float32))
            decisions.create_dataset("noise_norm", data=np.asarray([0.2, 0.3], dtype=np.float32))
            decisions.create_dataset("sample_ms", data=np.asarray([4.0, 5.0], dtype=np.float32))
        elif policy == "SVF":
            decisions.create_dataset("prng_key", data=np.asarray([[11, 12], [13, 14]], dtype=np.int64))
            decisions.create_dataset("noise_norm", data=np.asarray([0.2, 0.3], dtype=np.float32))
            decisions.create_dataset("chunk_norm", data=np.asarray([1.2, 1.4], dtype=np.float32))
        elif policy == "DSRL":
            decisions.create_dataset("z_norm", data=np.asarray([2.0, 3.0], dtype=np.float32))
            decisions.create_dataset("bound_fraction", data=np.asarray([0.1, 0.2], dtype=np.float32))
            decisions.create_dataset("z_abs_max", data=np.asarray([0.7, 0.9], dtype=np.float32))
            decisions.create_dataset("noise_scale", data=np.asarray([0.5, 0.4], dtype=np.float32))
            decisions.create_dataset("latency_ms", data=np.asarray([8.0, 9.0], dtype=np.float32))
            if include_q:
                decisions.create_dataset("q1", data=np.asarray([-0.4, -0.2], dtype=np.float32))
                decisions.create_dataset("q2", data=np.asarray([-0.5, -0.3], dtype=np.float32))
                decisions.create_dataset("q_chosen", data=np.asarray([-0.5, -0.3], dtype=np.float32))
    return path


def test_real_ifql_h5_joins_decisions_exactly_and_holds_values(tmp_path):
    real = load_real_h5_episode(_real_h5(tmp_path, "IFQL"))
    diagnostics = _real_diagnostics(real)
    assert [item.request_idx for item in diagnostics] == [10, 13, 20]
    assert [item.decision_idx for item in diagnostics] == [3, 3, 4]
    assert diagnostics[0].boundary and not diagnostics[1].boundary and diagnostics[2].boundary
    assert diagnostics[1].values["chosen_idx"] == 2
    assert diagnostics[2].values["chosen_idx"] == 4
    assert diagnostics[0].q["q_mean"] == pytest.approx(-1.25)
    assert diagnostics[0].q["q_std"] == pytest.approx(0.5)
    assert _chosen_q_heads(diagnostics[0].q) == pytest.approx((-1.7, -1.8))
    assert _chosen_q_heads(diagnostics[2].q) == pytest.approx((-0.5, -0.4))
    np.testing.assert_allclose(_q_history_series(diagnostics, "q1"), [-1.7, -1.7, -0.5])
    np.testing.assert_allclose(_q_history_series(diagnostics, "q2"), [-1.8, -1.8, -0.4])
    dashboard = draw_real_dashboard(real, 1, diagnostics[1], diagnostics[:2])
    assert dashboard.shape == (360, 1280, 3)


@pytest.mark.parametrize("policy", ["IFQL", "SVF", "DSRL"])
def test_real_h5_policy_fixtures_preserve_only_logged_diagnostics(tmp_path, policy):
    real = load_real_h5_episode(_real_h5(tmp_path, policy, include_q=(policy == "DSRL")))
    diagnostics = _real_diagnostics(real)
    if policy == "IFQL":
        assert diagnostics[0].values["candidate_norm"] == pytest.approx(3.0)
        assert diagnostics[0].q is not None and diagnostics[0].q["K"] == 8
    elif policy == "SVF":
        assert diagnostics[0].q is None
        assert np.isnan(_q_history_values(diagnostics)).all()
    else:
        assert diagnostics[0].q["q1"] == pytest.approx(-0.4)
        assert diagnostics[0].values["z_norm"] == pytest.approx(2.0)


def test_real_h5_missing_fields_remain_missing_and_video_is_verified(tmp_path):
    path = _real_h5(tmp_path, "SVF", include_optional=False)
    real = load_real_h5_episode(path)
    diagnostics = _real_diagnostics(real)
    assert real.state is None and real.output is None and real.gripper is None and real.chunk_step is None
    assert diagnostics[1].q is None and diagnostics[1].chunk_step == -1
    assert _real_elapsed_s(real, 1, 30) is None
    out = tmp_path / "real.mp4"
    result = render_real_h5_episode(path, out)
    assert result == out.resolve() and result.is_file()
    cap = cv2.VideoCapture(str(result))
    try:
        assert cap.isOpened()
        assert int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) == 3
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
    finally:
        cap.release()


def test_real_h5_preserves_explicit_request_input_and_output_trace(tmp_path):
    path = _real_h5(tmp_path, "SVF")
    with h5py.File(path, "r+") as f:
        f["requests"].create_dataset("model_input", data=np.full((3, 4), 2.0, np.float32))
        f["requests"].create_dataset("model_output", data=np.full((3, 7), 3.0, np.float32))
        f["requests"].create_dataset("noise", data=np.full((3, 2), 4.0, np.float32))
    real = load_real_h5_episode(path)
    np.testing.assert_array_equal(real.request_values["model_input"][1], [2.0] * 4)
    np.testing.assert_array_equal(real.request_values["model_output"][2], [3.0] * 7)
    np.testing.assert_array_equal(real.request_values["noise"][0], [4.0] * 2)
    assert _real_elapsed_s(real, 2, 30) == pytest.approx(0.2)


def test_resolve_ffmpeg_falls_back_to_imageio_ffmpeg(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled-ffmpeg"
    bundled.write_text("#!/bin/sh\n")
    bundled.chmod(0o755)
    import sim_collect.eval.render_eval_video as renderer

    monkeypatch.setattr(renderer.shutil, "which", lambda _: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", types.SimpleNamespace(
        get_ffmpeg_exe=lambda: str(bundled)
    ))
    assert _resolve_ffmpeg() == str(bundled)


def test_real_h5_rejects_incomplete_and_nonmonotonic_decisions(tmp_path):
    path = _real_h5(tmp_path, "IFQL")
    with h5py.File(path, "r+") as f:
        f.attrs["complete"] = False
    with pytest.raises(JoinError, match="complete=true"):
        load_real_h5_episode(path)
    with h5py.File(path, "r+") as f:
        f.attrs["complete"] = True
        f["decisions/decision_idx"][:] = [3, 3]
    with pytest.raises(JoinError, match="strictly increasing"):
        load_real_h5_episode(path)
