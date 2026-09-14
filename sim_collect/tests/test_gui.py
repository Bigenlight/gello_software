"""Tests for [C] sim_collect/gui.py against the §2 stub servers.

Everything that builds a Tk window carries the conftest `needs_display` marker (DISPLAY=:0
exists on this machine, so they run here). The GUI is driven the way an operator drives
it — `app.on_primary()`, `app.on_start_take()` — and then the STUB is asked what it saw,
so these are contract tests of the request stream, not of the widget tree.
"""
from __future__ import annotations

import socket
import time

import pytest

from conftest import needs_display          # pytest puts this dir on sys.path
from sim_collect import gui as gui_mod
from sim_collect import ipc
from sim_collect.tests.stub_servers import StubSim, start_stubs, solid_jpeg


def _free_ports(n: int) -> list:
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


@pytest.fixture(autouse=True)
def private_endpoints(monkeypatch):
    """Give every test its own TCP ports.

    The contract pins 6701/6702/6711/6712, but those are also what a real sim_main /
    capture (or another implementer's concurrent `pytest sim_collect/tests`) binds — and a
    foreign server answering `get_status` would make "shows disconnected" pass or fail for
    reasons that have nothing to do with the GUI. `ipc.endpoint()` reads this table on every
    call, so patching it moves the stubs AND the app together onto ports nobody else holds.
    """
    for name, port in zip(("sim_rep", "capture_rep", "state_pub", "preview_pub"),
                          _free_ports(4)):
        monkeypatch.setitem(ipc._TCP_PORTS, name, port)
    monkeypatch.setenv("SIM_COLLECT_IPC", "tcp")


def _make_app(**kw):
    return gui_mod.SimCollectGui(
        sim_timeout_ms=kw.pop("sim_timeout_ms", 800),
        status_timeout_ms=kw.pop("status_timeout_ms", 200),
        autostart=kw.pop("autostart", False),
        title="test",
        **kw)


@pytest.fixture
def stubs():
    stack = start_stubs()
    try:
        yield stack
    finally:
        stack.stop()


@pytest.fixture
def app(stubs):
    a = _make_app()
    a.poll_sim()
    a.poll_capture()
    a.refresh()
    try:
        yield a
    finally:
        a.close()


def _wait(app, pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        app.pump(0.05)
        if pred():
            return True
    return pred()


# --------------------------------------------------------------------------- #
# pure helper (no display needed)                                              #
# --------------------------------------------------------------------------- #
def test_jpeg_to_ppm_roundtrip():
    ppm = gui_mod.jpeg_to_ppm(solid_jpeg((0, 0, 255)))
    assert ppm is not None
    assert ppm.startswith(b"P6 320 180 255\n")
    assert len(ppm) == len(b"P6 320 180 255\n") + 320 * 180 * 3
    # first pixel is RGB, i.e. the BGR (0,0,255) came back as red
    px = ppm[len(b"P6 320 180 255\n"):][:3]
    assert px[0] > 200 and px[1] < 60 and px[2] < 60
    assert gui_mod.jpeg_to_ppm(b"") is None
    assert gui_mod.jpeg_to_ppm(b"not a jpeg at all") is None


# --------------------------------------------------------------------------- #
# primary toggle                                                               #
# --------------------------------------------------------------------------- #
@needs_display
def test_primary_needs_two_clicks_to_engage_then_one_to_disengage(app, stubs):
    assert not app.engaged

    app.on_primary()                       # first click only ARMS
    assert app.is_armed("primary")
    assert not stubs.sim.saw("engage")
    assert "CONFIRM ENGAGE" in app.primary.cget("text")

    app.on_primary()                       # second click fires
    assert stubs.sim.saw("engage")
    assert stubs.sim.engaged
    assert app.engaged                     # call_sim re-polled, so the GUI knows
    assert not app.is_armed("primary")
    assert "DISENGAGE" in app.primary.cget("text")

    app.on_primary()                       # DISENGAGE is single click
    assert stubs.sim.saw("disengage")
    assert not stubs.sim.engaged
    assert not app.engaged
    assert "ENGAGE" in app.primary.cget("text")


@needs_display
def test_engage_confirm_window_expires(app, stubs, monkeypatch):
    monkeypatch.setattr(gui_mod, "CONFIRM_MS", 60)
    app.on_primary()
    assert app.is_armed("primary")
    app.pump(0.3)
    assert not app.is_armed("primary")
    assert not stubs.sim.saw("engage")
    assert "CONFIRM" not in app.primary.cget("text")


@needs_display
def test_space_key_path_runs_the_primary(app, stubs):
    app._key(app.on_primary)
    app._key(app.on_primary)
    assert stubs.sim.saw("engage")


# --------------------------------------------------------------------------- #
# pos_scale + control mode                                                     #
# --------------------------------------------------------------------------- #
@needs_display
def test_pos_scale_slider_commits_to_the_sim(app, stubs):
    app.scale_var.set(50)
    app.on_pos_scale_commit()
    calls = stubs.sim.calls_of("set_pos_scale")
    assert calls and calls[-1]["value"] == pytest.approx(0.5)
    assert stubs.sim.pos_scale == pytest.approx(0.5)
    assert "pending 0.50" in app.scale_lbl_var.get()


@needs_display
def test_engage_commits_the_pending_scale_first(app, stubs):
    app.scale_var.set(30)
    app.on_primary()
    app.on_primary()
    cmds = stubs.sim.cmds()
    assert cmds.index("set_pos_scale") < cmds.index("engage")
    assert stubs.sim.pos_scale == pytest.approx(0.3)


@needs_display
def test_control_mode_radio_is_locked_while_engaged(app, stubs):
    app.on_set_mode("joint")
    assert stubs.sim.control_mode == "joint"
    app.on_set_mode("eef")

    app.on_primary(); app.on_primary()             # engage
    assert app.engaged
    assert app.mode_buttons["joint"].cget("state") == "disabled"
    stubs.sim.clear()
    assert app.on_set_mode("joint") is None
    assert not stubs.sim.saw("set_control_mode")
    assert stubs.sim.control_mode == "eef"


# --------------------------------------------------------------------------- #
# gripper / scene                                                              #
# --------------------------------------------------------------------------- #
@needs_display
def test_gripper_pause_is_one_click_resume_is_two(app, stubs):
    app.on_gripper_pause()
    assert stubs.sim.saw("gripper_pause") and stubs.sim.gripper_paused

    app.on_gripper_resume()
    assert app.is_armed("grip_resume")
    assert not stubs.sim.saw("gripper_resume")
    app.on_gripper_resume()
    assert stubs.sim.saw("gripper_resume") and not stubs.sim.gripper_paused


@needs_display
def test_home_and_reset_scene_with_seed(app, stubs):
    app.on_home()
    assert stubs.sim.home_count == 1

    app.seed_var.set("7")
    app.on_reset_scene()
    assert stubs.sim.calls_of("reset_scene")[-1].get("seed") == 7

    app.seed_var.set("not-an-int")          # bad seed -> ignored, still resets
    app.on_reset_scene()
    assert "seed" not in stubs.sim.calls_of("reset_scene")[-1]
    assert stubs.sim.reset_count == 2
    assert any("정수가 아니다" in l for l in app.log_lines())


# --------------------------------------------------------------------------- #
# recording                                                                    #
# --------------------------------------------------------------------------- #
@needs_display
def test_start_and_stop_take(app, stubs):
    assert app.start_btn.cget("state") == "normal"
    app.on_start_take()
    assert stubs.capture.saw("start_take")
    assert app.recording and stubs.capture.recording
    assert app.take_var.get() == "Take: 1 (recording)"
    assert "RECORDING take 1" in app.rec_var.get()
    assert app.start_btn.cget("state") == "disabled"
    assert app.stop_btn.cget("state") == "normal"

    app.on_stop_take()
    assert stubs.capture.saw("stop_take")
    assert not app.recording
    assert app.rec_var.get() == "PREVIEW"
    assert any("saved take_01" in l for l in app.log_lines())


@needs_display
def test_toggle_take_key(app, stubs):
    app.on_toggle_take()
    assert stubs.capture.recording
    app.on_toggle_take()
    assert not stubs.capture.recording


@needs_display
def test_reset_scene_is_refused_while_recording(app, stubs):
    app.on_start_take()
    assert app.recording
    assert app.reset_btn.cget("state") == "disabled"

    assert app.on_reset_scene() is None
    assert not stubs.sim.saw("reset_scene")
    assert stubs.sim.reset_count == 0
    assert any("RESET SCENE 거부" in l for l in app.log_lines())

    app.on_stop_take()
    app.on_reset_scene()
    assert stubs.sim.reset_count == 1


@needs_display
def test_discard_last_needs_a_confirm_and_is_blocked_while_recording(app, stubs):
    app.on_start_take()
    app.on_discard_last()
    assert not stubs.capture.saw("discard_last_take")
    app.on_stop_take()

    app.on_discard_last()                       # arms
    assert app.is_armed("discard")
    assert not stubs.capture.saw("discard_last_take")
    app.on_discard_last()                       # fires
    assert stubs.capture.saw("discard_last_take")
    assert stubs.capture.discarded


# --------------------------------------------------------------------------- #
# preview                                                                      #
# --------------------------------------------------------------------------- #
@needs_display
def test_preview_frames_become_photoimages(app, stubs):
    assert _wait(app, lambda: app.pump_preview() or app._photo["cam1"] is not None, 5.0)
    for cam in ("cam1", "cam2"):
        photo = app._photo[cam]
        assert photo is not None, cam
        assert (photo.width(), photo.height()) == (320, 180)
        assert photo.get(0, 0) != (0, 0, 0)        # a solid colour, not an empty image
        assert app.cam_labels[cam].cget("image")   # the label actually shows it
        assert "320×180" in app.cam_status[cam].get()


# --------------------------------------------------------------------------- #
# connection handling                                                          #
# --------------------------------------------------------------------------- #
@needs_display
def test_polling_autostart_connects_both_servers(stubs):
    a = _make_app(autostart=True)
    try:
        assert _wait(a, lambda: a.sim_connected and a.cap_connected, 5.0)
        assert a.sim_conn_var.get() == "sim: CONNECTED"
        assert a.cap_conn_var.get() == "capture: CONNECTED"
        assert _wait(a, lambda: a._photo["cam1"] is not None, 5.0)
    finally:
        a.close()


@needs_display
def test_no_servers_shows_disconnected_and_never_raises():
    a = _make_app(status_timeout_ms=80, sim_timeout_ms=120)
    try:
        a.poll_sim()
        a.poll_capture()
        a.refresh()
        assert "DISCONNECTED" in a.sim_conn_var.get()
        assert "DISCONNECTED" in a.cap_conn_var.get()
        assert not a.sim_connected and not a.cap_connected
        assert a.primary.cget("state") == "disabled"
        assert "no signal" in a.state_var.get()

        # Commands against a dead server must log a timeout, not raise.
        a.on_primary()
        a.on_primary()
        a.on_start_take()
        a.on_home()
        a.pump(0.1)
        assert any("timeout" in l for l in a.log_lines())
    finally:
        a.close()


@needs_display
def test_sim_restart_is_picked_up(stubs):
    a = _make_app(autostart=False)
    try:
        a.poll_sim(); a.refresh()
        assert a.sim_connected

        stubs.sim.stop()
        stubs.sim = None
        a.poll_sim(); a.refresh()
        assert not a.sim_connected
        assert "DISCONNECTED" in a.sim_conn_var.get()

        fresh = StubSim().start_and_wait()
        stubs.sim = fresh
        fresh.pos_scale = 0.4                      # the new process has its own value
        assert _wait(a, lambda: bool(a.poll_sim().get("ok")), 5.0)
        a.refresh()
        assert a.sim_connected
        assert a.scale_var.get() == 40             # GUI re-synced on the reconnect edge
        assert any("connected" in l for l in a.log_lines())
    finally:
        a.close()


@needs_display
def test_log_pane_is_a_200_line_ring(app):
    for i in range(260):
        app.log(f"line {i}")
    lines = app.log_lines()
    assert len(lines) <= gui_mod.LOG_LINES
    assert "line 259" in lines[-1]


# --------------------------------------------------------------------------- #
# the REAL status contract (keys copied from sim_main/capture/recorder)        #
# --------------------------------------------------------------------------- #
@needs_display
def test_gui_reads_the_real_capture_status_keys(app, stubs):
    """take_index / duration_s / frames / depth_frames / rows-DICT / render / drops."""
    app.on_start_take()
    time.sleep(0.25)
    app.poll_capture(); app.refresh()

    st = stubs.capture.status()
    assert "take_index" in st and "duration_s" in st and isinstance(st["rows"], dict)
    assert "take_count" not in st and "elapsed_s" not in st       # the keys we invented
    assert isinstance(st["counts"], dict) and isinstance(st["problems"], list)
    assert isinstance(st["capture"], dict) and "fps_ok" in st["capture"]

    assert app.take_var.get() == "Take: 1 (recording)"
    assert app.elapsed_var.get().startswith("Elapsed: 0.")
    counts = app.counts_var.get()
    assert "rows Σ" in counts and "depth" in counts and "state msgs" in counts
    total = int(counts.split("rows Σ")[1].split(" ")[0])
    assert total > 0
    # the rows DICT is shown per stream; frame/bookkeeping counters are not streams
    streams = app.rows_var.get()
    assert "ur_joint_states" in streams and "tcp_pose" in streams and "gello_grip" in streams
    assert "state_dropped" not in streams and "cam1_frames" not in streams

    health = app.cap_health_var.get()
    assert "scene ready" in health and "workers ok" in health and "sim alive" in health
    assert "cam1 30.0 Hz" in health and "12.6 ms" in health          # render stats per cam
    assert "fps 30.0/30.0 of 30" in health                            # capture_stats block
    assert app.drops_var.get() == "drops 0"
    assert app.drops_lbl.cget("fg") != "#cc3333"
    assert stubs.capture.root in app.take_dir_var.get()
    app.on_stop_take()


@needs_display
def test_drop_counters_go_red(app, stubs):
    stubs.capture.state_dropped = 3
    stubs.capture.queue_drops = 2
    stubs.capture.bad_msgs = 1
    app.poll_capture(); app.refresh()
    txt = app.drops_var.get()
    assert txt.startswith("DROPS")
    assert "state_dropped 3" in txt and "cam1_queue_drops 2" in txt and "bad_msgs 1" in txt
    assert app.drops_lbl.cget("fg") == "#cc3333"


@needs_display
def test_recorder_problems_outrank_the_counters_and_are_logged(app, stubs):
    """`problems` is the recorder's own verdict (empty = clean); it must be visible."""
    app.on_start_take()
    stubs.capture.write_errors = 2
    stubs.capture.missed_ticks = 5
    app.poll_capture(); app.refresh()
    txt = app.drops_var.get()
    assert txt.startswith("PROBLEMS (2)")
    assert "write errors" in txt and "never reached the recorder" in txt
    assert app.drops_lbl.cget("fg") == "#cc3333"
    assert any("PROBLEM: 2 write errors" in l for l in app.log_lines())
    app.on_stop_take()


@needs_display
def test_slow_render_is_flagged_amber(app, stubs):
    stubs.capture.achieved_fps = 21.0
    app.poll_capture(); app.refresh()
    assert app.drops_var.get() == "render below nominal fps"
    assert app.drops_lbl.cget("fg") == "#dd8800"
    assert "SLOW" in app.cap_health_var.get()


@needs_display
def test_idle_capture_has_no_rows_and_a_null_take_dir(app, stubs):
    """The real recorder reports rows={} and take_dir=None while idle — not 0 and ""."""
    st = stubs.capture.status()
    assert st["rows"] == {} and st["take_dir"] is None
    app.poll_capture(); app.refresh()
    assert "rows Σ0" in app.counts_var.get()
    assert app.rows_var.get() == "streams: —"
    assert app.take_dir_var.get().startswith("take dir: —")


@needs_display
def test_capture_health_flags_a_dead_worker(app, stubs):
    stubs.capture.workers_alive = False
    stubs.capture.sim_alive = False
    app.poll_capture(); app.refresh()
    health = app.cap_health_var.get()
    assert "WORKERS DEAD" in health and "NO SIM STATE" in health and "SCENE NOT READY" in health
    assert app.start_btn.cget("state") == "normal"      # the server refuses, not the GUI
    rep = app.on_start_take()
    assert rep is not None and rep["ok"] is False
    assert any("not ready" in l for l in app.log_lines())


@needs_display
def test_gui_reads_the_real_sim_status_keys(app, stubs):
    st = stubs.sim.status()
    assert st["msg"] == "alive" and isinstance(st["task_success"], bool)
    assert "task" not in st                       # sim get_status has NO task dict

    stubs.sim.task_success = True
    stubs.sim.tick = 12345
    stubs.sim.rt_ratio = 0.87
    app.poll_sim(); app.refresh()
    assert app.task_var.get().startswith("TASK SUCCESS")
    assert app.task_lbl.cget("bg") == "#22aa22"
    health = app.sim_health_var.get()
    assert "tick 12345" in health and "rt 0.87×" in health
    assert "LEADER=FAKE" in health and "seed 1" in health
    assert "sigma_min 0.124" in app.info_var.get()

    stubs.sim.gripper_paused = True
    stubs.sim.leader_ok = False
    app.poll_sim(); app.refresh()
    assert "GRIPPER PAUSED" in app.sim_health_var.get()


@needs_display
def test_engage_rejection_reason_is_shown(app, stubs, monkeypatch):
    monkeypatch.setattr(type(stubs.sim), "eef_info",
                        lambda self: {"sigma_min": 0.02, "reject_reason": None,
                                      "auto_reason": None, "last_gate": "singular_anchor",
                                      "last_gate_detail": "sigma_min=0.0200 <= 0.1000"})
    app.poll_sim(); app.refresh()
    assert "singular_anchor: sigma_min=0.0200" in app.info_var.get()


# --------------------------------------------------------------------------- #
# keyboard                                                                     #
# --------------------------------------------------------------------------- #
@needs_display
def test_held_space_does_not_engage_and_disengage(app, stubs):
    """X11 auto-repeat = a stream of presses; only the first one may act."""
    for _ in range(6):
        app._on_key_press("space", app.on_primary)       # held down, no release
    assert len(stubs.sim.calls_of("engage")) == 0        # still only ARMED
    assert app.is_armed("primary")

    app._on_key_release("space")
    app.pump(0.2)                                        # release settles
    app._on_key_press("space", app.on_primary)
    assert len(stubs.sim.calls_of("engage")) == 1

    # an auto-repeat release immediately followed by a press must NOT re-fire
    app._on_key_release("space")
    app._on_key_press("space", app.on_primary)
    assert not stubs.sim.saw("disengage")


@needs_display
def test_seed_entry_does_not_swallow_the_shortcuts_forever(app, stubs):
    app.seed_entry.focus_set()
    app.pump(0.05)
    app._key(app.on_home)                       # typing: shortcut ignored
    assert stubs.sim.home_count == 0

    app.seed_var.set("42")
    app._blur_entry()                           # <Return> hands focus back
    app.pump(0.05)
    app._key(app.on_home)
    assert stubs.sim.home_count == 1
    assert app.seed_var.get() == "42"           # Return keeps the seed

    app.seed_entry.focus_set(); app.pump(0.05)
    app._blur_entry(clear=True)                 # <Escape> blurs AND clears
    assert app.seed_var.get() == ""
    app.pump(0.05)
    app._key(app.on_home)
    assert stubs.sim.home_count == 2
