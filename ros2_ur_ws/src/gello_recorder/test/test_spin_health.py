"""``gello_recorder.spin_health`` -- the three defences against a starved spin
thread, tested WITHOUT ROS, Qt, a robot or a camera.

The defect these pin down is the 2026-09-14 recorder timestamp artifact: every
callback ran on one rclpy spin thread, per-frame decode/encode/HDF5 work
saturated it, and every topic publishing faster than the resulting round rate
was then read out of a permanently full KEEP_LAST queue -- so each recorded row
was ``QoS depth / publish rate`` seconds old (0.900 s for /joint_states) with
nothing in the file saying so.
"""

import threading
import time

import pytest

from gello_recorder.spin_health import (
    DEPTH_ON_BANNER,
    FRAME_QUEUE_MAXSIZE,
    QOS_DEPTH_CAMERA,
    QOS_DEPTH_GELLO,
    QOS_DEPTH_GRIPPER,
    QOS_DEPTH_ROBOT_STATE,
    ROS_LAG_WARN_S,
    STARVATION_RATE_FLOOR_HZ,
    STARVATION_RATE_TOL,
    FrameWriteQueue,
    PreviewDecoder,
    detect_spin_starvation,
    native_rate_table,
    stop_health_suffix,
)


# --------------------------------------------------------------------------- #
# Queue-depth budget
# --------------------------------------------------------------------------- #
def test_robot_state_depth_bounds_staleness_at_50ms_at_100hz():
    # THE rule: worst-case staleness = depth / publish rate. The defective
    # corpus had depth 100 at ~100 Hz = 1.00 s.
    assert QOS_DEPTH_ROBOT_STATE / 100.0 <= 0.05
    assert QOS_DEPTH_ROBOT_STATE / 500.0 <= 0.01
    # ...and it must stay well under the live warning threshold, or the alarm
    # would fire on a perfectly healthy recorder.
    assert QOS_DEPTH_ROBOT_STATE / 100.0 < ROS_LAG_WARN_S


def test_slow_topics_keep_their_shallow_but_not_tiny_queues():
    # 30-40 Hz streams never reach the round-rate ceiling, so there is nothing
    # to fix there -- but nothing may be DEEPER than the robot topics either.
    for depth in (QOS_DEPTH_CAMERA, QOS_DEPTH_GELLO, QOS_DEPTH_GRIPPER):
        assert depth >= QOS_DEPTH_ROBOT_STATE
        assert depth <= 20


# --------------------------------------------------------------------------- #
# FrameWriteQueue: ordering, exact drain, counted drops
# --------------------------------------------------------------------------- #
def test_writer_preserves_submission_order():
    seen = []
    q = FrameWriteQueue(lambda item: seen.append(item[0:1] + item[2:3]), maxsize=256)
    for i in range(200):
        assert q.submit(("cam1", b"x", float(i), float("nan"))) is True
    assert q.close() is True
    assert [t for _, t in seen] == [float(i) for i in range(200)]


def test_writer_interleaves_two_streams_but_keeps_each_in_order():
    seen = []
    q = FrameWriteQueue(lambda item: seen.append((item[0], item[2])), maxsize=256)
    for i in range(50):
        q.submit(("cam1", b"", float(i), float("nan")))
        q.submit(("cam2", b"", float(i), float("nan")))
    assert q.close()
    cam1 = [t for kind, t in seen if kind == "cam1"]
    cam2 = [t for kind, t in seen if kind == "cam2"]
    assert cam1 == sorted(cam1) and cam2 == sorted(cam2)
    assert len(cam1) == len(cam2) == 50


def test_close_drains_everything_n_enqueued_equals_n_written():
    # A deliberately SLOW handler, so close() is genuinely waiting on a backlog
    # rather than finding the queue already empty.
    seen = []

    def slow(item):
        time.sleep(0.002)
        seen.append(item)

    q = FrameWriteQueue(slow, maxsize=FRAME_QUEUE_MAXSIZE)
    n = 40
    for i in range(n):
        assert q.submit(("cam1", b"", float(i), float("nan")))
    assert q.pending > 0, "the handler is too fast for this test to mean anything"
    assert q.close() is True
    assert len(seen) == n == q.submitted
    assert q.pending == 0
    assert q.dropped == 0
    assert not q.is_alive()


def test_full_queue_drops_and_counts_instead_of_blocking():
    gate = threading.Event()

    def blocked(item):
        gate.wait(5.0)

    q = FrameWriteQueue(blocked, maxsize=2)
    accepted = 0
    t0 = time.monotonic()
    for i in range(50):
        if q.submit(("cam1", b"", float(i), float("nan"))):
            accepted += 1
    elapsed = time.monotonic() - t0
    # The producer is the rclpy spin thread: it must NEVER wait on the writer.
    assert elapsed < 0.5, "submit() blocked ({:.3f}s)".format(elapsed)
    assert q.dropped == 50 - accepted > 0
    assert accepted <= 2 + 1  # queue slots + the one item in the handler
    gate.set()
    q.close()


def test_submit_after_close_is_refused_and_counted():
    q = FrameWriteQueue(lambda item: None, maxsize=4)
    assert q.close()
    before = q.dropped
    assert q.submit(("cam1", b"", 0.0, float("nan"))) is False
    assert q.dropped == before + 1


def test_handler_exception_does_not_kill_the_writer():
    seen = []

    def flaky(item):
        if item[2] == 1.0:
            raise RuntimeError("boom")
        seen.append(item[2])

    q = FrameWriteQueue(flaky, maxsize=16)
    for i in range(4):
        q.submit(("cam1", b"", float(i), float("nan")))
    assert q.close()
    assert seen == [0.0, 2.0, 3.0], seen
    assert q.errors == 1


def test_close_is_idempotent():
    q = FrameWriteQueue(lambda item: None, maxsize=4)
    assert q.close()
    assert q.close()


# --------------------------------------------------------------------------- #
# PreviewDecoder: latest-wins, never a backlog
# --------------------------------------------------------------------------- #
def test_preview_decoder_is_latest_wins_per_key():
    gate = threading.Event()
    got = []

    def decode(payload):
        if payload == b"first":
            gate.wait(5.0)
        return payload

    dec = PreviewDecoder(lambda key, frame: got.append((key, frame)),
                         decode=decode)
    try:
        dec.submit(1, b"first")          # occupies the decoder thread
        time.sleep(0.05)
        for i in range(20):              # all superseded except the last
            dec.submit(1, "v{}".format(i).encode())
        gate.set()
        assert dec.drain(2.0)
        time.sleep(0.05)
    finally:
        dec.stop()
    assert [k for k, _ in got] == [1, 1], got
    assert got[0][1] == b"first"
    assert got[1][1] == b"v19", "latest-wins must deliver the NEWEST payload"
    assert dec.stats["superseded"] == 19


def test_preview_decoder_keeps_separate_slots_per_camera():
    got = []
    dec = PreviewDecoder(lambda key, frame: got.append((key, frame)),
                         decode=lambda p: p)
    try:
        dec.submit(1, b"a")
        dec.submit(2, b"b")
        assert dec.drain(2.0)
        time.sleep(0.05)
    finally:
        dec.stop()
    assert sorted(got) == [(1, b"a"), (2, b"b")]


def test_preview_decoder_drops_undecodable_payloads_silently():
    got = []
    dec = PreviewDecoder(lambda key, frame: got.append(frame),
                         decode=lambda p: None)
    try:
        dec.submit(1, b"garbage")
        assert dec.drain(2.0)
        time.sleep(0.05)
    finally:
        dec.stop()
    assert got == []
    assert dec.stats["failed"] == 1


# --------------------------------------------------------------------------- #
# The starvation heuristic
# --------------------------------------------------------------------------- #
def test_the_actual_defective_take_is_flagged():
    # take_01 of carrot_in_pot, measured: four topics with different publish
    # rates recording at the same rate to five decimal places.
    report = detect_spin_starvation({
        "command": 59.79218,
        "ur_joint_states": 59.79089,
        "tcp_pose": 59.79282,
        "wrench": 59.79153,
    })
    assert report["suspected"] is True
    assert report["converged_hz"] == pytest.approx(59.7918, abs=1e-3)
    assert "converged at 59.79 Hz" in report["message"]
    assert "spin thread starved" in report["message"]


def test_take_02_of_the_defective_corpus_is_flagged_too():
    report = detect_spin_starvation({
        "command": 69.084, "ur_joint_states": 69.086,
        "tcp_pose": 69.172, "wrench": 69.166,
    })
    assert report["suspected"] is True


def test_a_healthy_july_take_is_not_flagged():
    # Nothing starved: all four sat at the controller_manager rate.
    report = detect_spin_starvation({
        "command": 98.91, "ur_joint_states": 98.90,
        "tcp_pose": 98.92, "wrench": 98.89,
    })
    assert report["suspected"] is False
    assert "not starvation" in report["reason"]
    assert report["message"] is None


def test_slow_but_disagreeing_rates_are_not_flagged():
    # A genuinely slow publisher is not the same thing as a shared ceiling.
    report = detect_spin_starvation({
        "command": 60.0, "ur_joint_states": 40.0,
        "tcp_pose": 55.0, "wrench": 55.0,
    })
    assert report["suspected"] is False
    assert "kept their own rates" in report["reason"]


def test_one_table_alone_proves_nothing():
    report = detect_spin_starvation({"ur_joint_states": 60.0})
    assert report["suspected"] is False
    assert "fewer than two" in report["reason"]
    assert detect_spin_starvation({})["suspected"] is False


def test_tolerance_boundary_is_where_it_says_it_is():
    base = 60.0
    # Spread just inside 0.5 % -> flagged; just outside -> not.
    inside = base * (1 + STARVATION_RATE_TOL * 0.4)
    outside = base * (1 + STARVATION_RATE_TOL * 4.0)
    assert detect_spin_starvation({"a": base, "b": inside})["suspected"] is True
    assert detect_spin_starvation({"a": base, "b": outside})["suspected"] is False


def test_floor_boundary_uses_the_published_rate_not_a_guess():
    hz = STARVATION_RATE_FLOOR_HZ + 1.0
    assert detect_spin_starvation({"a": hz, "b": hz})["suspected"] is False
    hz = STARVATION_RATE_FLOOR_HZ - 1.0
    assert detect_spin_starvation({"a": hz, "b": hz})["suspected"] is True


def test_non_numeric_and_zero_rates_are_ignored():
    report = detect_spin_starvation({
        "command": 60.0, "ur_joint_states": 60.0,
        "tcp_pose": 0.0, "wrench": None,
    })
    assert sorted(report["rates_hz"]) == ["command", "ur_joint_states"]
    assert report["suspected"] is True


# --------------------------------------------------------------------------- #
# native_rate_table
# --------------------------------------------------------------------------- #
def test_native_rate_table_is_rows_over_duration():
    rates = native_rate_table(
        {"command": 600, "ur_joint_states": 597, "tcp_pose": 599,
         "wrench": 598, "cam1_frames": 300, "gello_joint_states": 300},
        10.0,
    )
    # Only the four >=100 Hz robot tables; cameras/GELLO are not evidence.
    assert sorted(rates) == ["command", "tcp_pose", "ur_joint_states", "wrench"]
    assert rates["command"] == pytest.approx(60.0)


def test_native_rate_table_is_empty_without_a_usable_duration():
    assert native_rate_table({"command": 10}, 0.0) == {}
    assert native_rate_table({"command": 10}, None) == {}
    assert native_rate_table({}, 10.0) == {}
    # A table with no rows is silence, not a 0 Hz measurement.
    assert native_rate_table({"command": 0}, 10.0) == {}


# --------------------------------------------------------------------------- #
# Operator-facing wording
# --------------------------------------------------------------------------- #
def test_a_clean_take_says_nothing_extra():
    assert stop_health_suffix({
        "dropped_frames": {"total": 0},
        "spin_starvation_suspected": False,
        "ros_lag_s_max": 0.01,
    }) == ""
    assert stop_health_suffix({}) == ""
    assert stop_health_suffix(None) == ""


def test_a_bad_take_says_all_three_things():
    txt = stop_health_suffix({
        "dropped_frames": {"cam1": 3, "total": 3},
        "spin_starvation_suspected": True,
        "native_rates_hz": {"command": 60.0, "wrench": 60.0},
        "ros_lag_s_max": 0.9,
    })
    assert "DROPPED 3 frame(s)" in txt
    assert "SPIN STARVED" in txt and "60.0 Hz" in txt
    assert "peak ros lag 0.900s" in txt


# --------------------------------------------------------------------------- #
# The depth opt-in banner
# --------------------------------------------------------------------------- #
def test_the_depth_banner_names_all_three_costs():
    """Depth is opt-in because it is expensive; the banner is where the operator
    is told the price, at the moment they choose to pay it. All three costs must
    be in it -- disk, extra subscriptions, CPU -- plus the one number to watch."""
    txt = DEPTH_ON_BANNER
    assert "depth ON" in txt
    assert "MB/s" in txt          # disk
    assert "subscriptions" in txt  # spin-thread load
    assert "CPU" in txt
    assert "ros_lag_s" in txt      # what to watch while it runs
    assert len(txt) < 200, "it has to fit on one log line"
