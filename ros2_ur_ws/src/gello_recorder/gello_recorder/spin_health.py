#!/usr/bin/env python3
"""Keeping the single rclpy spin thread free -- and noticing when it wasn't.

Everything in this module exists because of ONE defect, diagnosed on the
carrot_in_pot corpus recorded 2026-09-14 (54 takes) and written up in
``docs/ros2/GELLO_UR7E_RECORDING.md``:

    Every recorder subscription callback runs on the SAME background rclpy spin
    thread, and every row's ``t_rel_s`` is the time that callback RAN. A
    ``SingleThreadedExecutor`` services at most ONE message per subscription per
    round, so a subscription's service rate equals the round rate. Once depth
    recording (commit ``c694d2c``) put ~1.6 s of decode/encode/HDF5 work per
    wall-clock second onto that thread, the round rate fell from ~100 Hz to
    60-69 Hz. Every topic publishing FASTER than that then kept its KEEP_LAST
    history permanently full, and since KEEP_LAST hands the reader the OLDEST
    sample it still holds, each row was stale by exactly

        staleness = QoS depth / publish rate

    -> /joint_states (depth 100 @ ~100 Hz) = 0.900 s late,
       tcp_pose + wrench (depth 50 @ ~100 Hz) = ~0.45 s late.

    Nothing in the data said so. The signal shape was intact, the file was
    valid, and the only visible fingerprint was that the four starved tables'
    recorded rates converged onto the round rate to five decimal places.

Three defences live here, and they are independent on purpose:

* :class:`FrameWriteQueue` -- move the heavy per-frame I/O (JPEG decode, MP4
  encode, depth-PNG HDF5 append) OFF the spin thread onto one background
  daemon writer, so the round rate stops collapsing in the first place.
* :class:`PreviewDecoder` -- the same idea for the GUI's live preview decode,
  which is pure display work and must never cost the recorder a round. It is
  LATEST-WINS: a preview frame that has been superseded is simply dropped.
* :func:`detect_spin_starvation` -- the regression alarm. If it ever happens
  again, the take says so out loud instead of looking healthy.

Deliberately ROS-free / Qt-free: it imports only the standard library plus
``cv2``/``numpy`` (for the preview decode), so every behaviour below is
unit-testable without an rclpy environment.
"""

import queue
import threading
import time
import traceback

# --------------------------------------------------------------------------- #
# Queue-depth budget (imported by the recorder nodes; the ONE place the rule is
# written down). Worst-case staleness of a subscription is  depth / rate .
# --------------------------------------------------------------------------- #
#: Robot state / command topics published by controller_manager at ~100 Hz
#: (and the bridge's ``commands`` at up to 250 Hz). depth 5 bounds the
#: worst case at 5/100 = 50 ms, and 5/500 = 10 ms at 500 Hz. A deeper queue
#: buys this recorder NOTHING -- it writes whatever arrives, so an extra slot
#: only converts "a row was skipped" into "a row is old", and a skipped row is
#: strictly better than a silently mis-stamped one.
QOS_DEPTH_ROBOT_STATE = 5
#: 30 Hz camera streams (colour + compressedDepth). 10/30 = 0.33 s worst case,
#: but these never actually queue: they publish slower than the round rate.
QOS_DEPTH_CAMERA = 10
#: GELLO leader joints, 30 Hz. Was 50; 10 is the same 0.33 s bound as the
#: cameras and matches the "shallow unless proven otherwise" default.
QOS_DEPTH_GELLO = 10
#: Robotiq gripper trigger/command/position, ~40 Hz. 20/40 = 0.5 s worst case;
#: it has never been observed to queue, so it is left as it was.
QOS_DEPTH_GRIPPER = 20
#: Low-rate status/state strings (5 Hz teleop state, speed scaling, ...).
QOS_DEPTH_STATUS = 10

#: Live warning threshold for ``ros_lag_s`` (now - latest /joint_states header
#: stamp). 0.15 s is ~3x the depth-5 worst case at 100 Hz, so it cannot fire on
#: the healthy path, and it is 6x below the 0.900 s that went unnoticed.
ROS_LAG_WARN_S = 0.15
#: How often the live lag warning may be logged.
ROS_LAG_WARN_PERIOD_S = 5.0

#: One line printed/logged at startup whenever depth recording is ON, so the
#: operator sees the bill BEFORE the take rather than in the analysis six
#: months later. Depth is opt-in (``ENABLE_DEPTH=1``) exactly because of this
#: cost: on 2026-09-14 it was on by default for one day and silently back-dated
#: every robot row of a 54-take corpus (module docstring above). The defect is
#: fixed; the cost is not.
DEPTH_ON_BANNER = (
    "depth ON: ~3 MB/s disk and +2 subscriptions per camera, recorder CPU "
    "+3-7 pp, robot-row stamp age median ~10 -> ~25 ms (measured 2026-09-14); "
    "watch ros_lag_s"
)

#: Frames the writer thread may be behind before it starts dropping. 64 frames
#: is ~1 s of both colour streams plus both depth streams at 30 Hz -- long
#: enough to ride out an HDF5 chunk flush, short enough that a genuinely
#: overloaded machine drops frames (counted, reported) instead of growing an
#: unbounded backlog and eating RAM.
FRAME_QUEUE_MAXSIZE = 64

#: Starvation heuristic: the four native robot tables agreeing this closely is
#: not physics, it is a shared service-rate ceiling. Measured on the defective
#: corpus: take_01 gave 59.79218 / 59.79089 / 59.79282 / 59.79153 Hz (relative
#: spread 3.2e-5); healthy July takes sat at their own native rates.
STARVATION_RATE_TOL = 0.005
#: ...but only call it starvation if the shared rate is BELOW the slowest
#: publisher we expect (controller_manager runs at 100 Hz).
STARVATION_RATE_FLOOR_HZ = 90.0
#: The four native-rate tables fed by topics that publish at >= 100 Hz.
STARVATION_TABLES = ("command", "ur_joint_states", "tcp_pose", "wrench")

_SENTINEL = object()


class FrameWriteQueue:
    """One background daemon thread draining submitted work items in FIFO order.

    Callers hand over ``(kind, payload, t_rel_s, stamp_s)`` tuples with
    :meth:`submit`, which NEVER blocks: on a full queue the item is dropped and
    counted. The thread calls ``handler(item)`` for each; a handler exception is
    printed and swallowed, because losing the writer thread would silently
    freeze every camera stream for the rest of the session.

    The ``t_rel_s`` inside the item is the caller's (arrival-time) stamp -- the
    whole point of this class is that the writer's own clock must never end up
    in the data. That is the defect this module exists to prevent.
    """

    def __init__(self, handler, maxsize: int = FRAME_QUEUE_MAXSIZE,
                 name: str = "recording-frame-writer"):
        self._handler = handler
        self._q = queue.Queue(maxsize=max(1, int(maxsize)))
        self._cv = threading.Condition()
        self._pending = 0          # submitted but not yet handled
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._errors = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # ---- producer side (the rclpy spin thread) ---------------------------
    def submit(self, item) -> bool:
        """Enqueue one item. Returns False (and counts a drop) if it could not
        be taken -- either the queue is full or the writer is already closing."""
        with self._cv:
            if self._closed:
                self._dropped += 1
                return False
            self._pending += 1
            self._submitted += 1
        try:
            self._q.put_nowait(item)
        except queue.Full:
            with self._cv:
                self._pending -= 1
                self._submitted -= 1
                self._dropped += 1
                self._cv.notify_all()
            return False
        return True

    # ---- consumer side ----------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            try:
                self._handler(item)
            except Exception:  # noqa: BLE001 - never let the writer thread die
                with self._cv:
                    self._errors += 1
                traceback.print_exc()
            finally:
                with self._cv:
                    self._pending -= 1
                    self._written += 1
                    self._cv.notify_all()

    # ---- lifecycle --------------------------------------------------------
    def drain(self, timeout=None) -> bool:
        """Block until every submitted item has been handled. Returns True if
        the queue actually emptied, False on timeout."""
        with self._cv:
            return bool(self._cv.wait_for(lambda: self._pending <= 0, timeout))

    def close(self, timeout: float = 30.0) -> bool:
        """Stop accepting work, drain what is queued, and join the thread.

        Returns True if everything drained. Callers finalise their files only
        after this returns, which is what keeps "frames enqueued" ==
        "frames written" exact.
        """
        with self._cv:
            already = self._closed
            self._closed = True
        drained = self.drain(timeout)
        if not already:
            try:
                self._q.put(_SENTINEL, timeout=max(0.1, timeout))
            except queue.Full:  # pragma: no cover - drained above, cannot be full
                pass
        self._thread.join(timeout)
        return drained

    # ---- introspection ----------------------------------------------------
    @property
    def pending(self) -> int:
        with self._cv:
            return self._pending

    @property
    def submitted(self) -> int:
        with self._cv:
            return self._submitted

    @property
    def written(self) -> int:
        with self._cv:
            return self._written

    @property
    def dropped(self) -> int:
        with self._cv:
            return self._dropped

    @property
    def errors(self) -> int:
        with self._cv:
            return self._errors

    @property
    def closed(self) -> bool:
        with self._cv:
            return self._closed

    def is_alive(self) -> bool:
        return self._thread.is_alive()


class PreviewDecoder:
    """LATEST-WINS background JPEG decoder for a GUI live preview.

    One slot per key (camera index): submitting a new payload for a key that
    has not been decoded yet simply REPLACES it. The preview can therefore never
    build a backlog and can never slow the producer down -- and the producer here
    is the rclpy spin thread, which must stay free.

    ``sink(key, frame)`` is called from the decoder thread with the decoded
    BGR ``numpy`` array; a payload that fails to decode calls nothing (the GUI
    keeps showing the last good frame, exactly as before).
    """

    def __init__(self, sink, name: str = "recorder-preview-decoder",
                 decode=None):
        self._sink = sink
        self._decode = decode if decode is not None else _decode_jpeg
        self._cv = threading.Condition()
        self._slots = {}
        self._order = []
        self._superseded = 0
        self._decoded = 0
        self._failed = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, key, payload) -> None:
        """Offer one payload for ``key``. Never blocks, never raises."""
        with self._cv:
            if self._stop:
                return
            if key in self._slots:
                self._superseded += 1
            else:
                self._order.append(key)
            self._slots[key] = payload
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._slots and not self._stop:
                    self._cv.wait()
                if self._stop and not self._slots:
                    return
                key = self._order.pop(0)
                payload = self._slots.pop(key)
            try:
                frame = self._decode(payload)
            except Exception:  # noqa: BLE001 - a bad frame must not kill preview
                frame = None
                traceback.print_exc()
            if frame is None:
                with self._cv:
                    self._failed += 1
                continue
            with self._cv:
                self._decoded += 1
            try:
                self._sink(key, frame)
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def stop(self, timeout: float = 2.0) -> None:
        with self._cv:
            self._stop = True
            self._slots.clear()
            self._order = []
            self._cv.notify_all()
        self._thread.join(timeout)

    def drain(self, timeout: float = 2.0) -> bool:
        """Wait until no payload is queued (test helper)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cv:
                if not self._slots:
                    return True
            time.sleep(0.005)
        return False

    @property
    def stats(self) -> dict:
        with self._cv:
            return {
                "decoded": self._decoded,
                "superseded": self._superseded,
                "failed": self._failed,
            }

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def _decode_jpeg(payload):
    """Default :class:`PreviewDecoder` codec -- imported lazily so this module
    stays importable (and the queue/heuristic testable) without OpenCV."""
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
# The regression alarm
# --------------------------------------------------------------------------- #
def native_rate_table(message_counts: dict, duration_s: float,
                      tables=STARVATION_TABLES) -> dict:
    """``{table: recorded mean Hz}`` for the native-rate robot tables.

    Rate is rows / duration -- the same number the offline analysis computes,
    so a WARN raised here and a WARN raised six months later from the file agree
    by construction. Tables with no rows are omitted (a topic that never
    published is not evidence of anything)."""
    out = {}
    try:
        duration_s = float(duration_s)
    except (TypeError, ValueError):
        return out
    if not duration_s or duration_s <= 0.0:
        return out
    counts = message_counts or {}
    for name in tables:
        n = counts.get(name)
        if not n:
            continue
        out[name] = float(n) / duration_s
    return out


def detect_spin_starvation(rates: dict, tol: float = STARVATION_RATE_TOL,
                           floor_hz: float = STARVATION_RATE_FLOOR_HZ) -> dict:
    """Decide whether the four native robot tables converged onto one rate.

    The fingerprint (see the module docstring): four topics with DIFFERENT
    publish rates recording at the SAME rate means all four hit the executor's
    round-robin ceiling, i.e. all four were reading out of a full queue and
    every row is stale. Two conditions, both required:

      * at least two of the tables present, and their relative spread
        ``(max - min) / mean <= tol`` (default 0.5 %); and
      * that shared rate is below ``floor_hz`` (default 90 Hz), because
        controller_manager publishes at 100 Hz -- agreeing AT 100 Hz just means
        nothing was starved.

    Returns a JSON-safe dict; ``suspected`` is the answer and ``message`` is the
    exact WARN text to log (``None`` when not suspected).
    """
    clean = {}
    for name, hz in (rates or {}).items():
        try:
            hz = float(hz)
        except (TypeError, ValueError):
            continue
        if hz > 0.0:
            clean[name] = hz

    report = {
        "suspected": False,
        "rates_hz": {k: round(v, 5) for k, v in sorted(clean.items())},
        "converged_hz": None,
        "spread_rel": None,
        "reason": "",
        "message": None,
    }
    if len(clean) < 2:
        report["reason"] = (
            "fewer than two native tables had rows -- nothing to compare")
        return report

    values = list(clean.values())
    mean = sum(values) / len(values)
    spread = (max(values) - min(values)) / mean if mean > 0 else float("inf")
    report["converged_hz"] = round(mean, 5)
    report["spread_rel"] = round(spread, 8)

    if spread > tol:
        report["reason"] = (
            "native tables kept their own rates (spread {:.3%} > {:.3%})".format(
                spread, tol))
        return report
    if min(values) >= floor_hz:
        report["reason"] = (
            "native tables agree at {:.2f} Hz, at or above the {:.0f} Hz "
            "publish rate -- not starvation".format(mean, floor_hz))
        return report

    report["suspected"] = True
    report["reason"] = (
        "native tables agree within {:.3%} at {:.2f} Hz, below {:.0f} Hz".format(
            spread, mean, floor_hz))
    report["message"] = (
        "native tables converged at {:.2f} Hz -- spin thread starved; "
        "robot rows may be stale ({})".format(
            mean,
            ", ".join("{} {:.2f} Hz".format(k, v)
                      for k, v in sorted(clean.items()))))
    return report


def stop_health_suffix(meta: dict) -> str:
    """The spin-health tail of a take's one-line stop summary (GUI status bar).

    Empty string when the take was clean -- an operator should not have to read
    a health report after every good take; they should only ever see this line
    when something is actually wrong. Pure and Qt-free so the wording is
    testable without a display."""
    if not isinstance(meta, dict):
        return ""
    parts = []
    dropped = meta.get("dropped_frames") or {}
    try:
        total = int(dropped.get("total", 0))
    except (TypeError, ValueError, AttributeError):
        total = 0
    if total:
        parts.append("DROPPED {} frame(s)".format(total))
    if meta.get("spin_starvation_suspected"):
        hz = meta.get("native_rates_hz") or {}
        try:
            avg = sum(float(v) for v in hz.values()) / max(1, len(hz))
            parts.append(
                "SPIN STARVED (native tables converged at {:.1f} Hz -- robot "
                "rows may be stale)".format(avg))
        except (TypeError, ValueError):
            parts.append("SPIN STARVED (robot rows may be stale)")
    lag_max = meta.get("ros_lag_s_max")
    try:
        lag_max = None if lag_max is None else float(lag_max)
    except (TypeError, ValueError):
        lag_max = None
    if lag_max is not None and lag_max > ROS_LAG_WARN_S:
        parts.append("peak ros lag {:.3f}s".format(lag_max))
    if not parts:
        return ""
    return " | !! " + " | ".join(parts)


if __name__ == "__main__":  # pragma: no cover - manual smoke
    seen = []
    q = FrameWriteQueue(lambda item: seen.append(item), maxsize=4)
    for i in range(4):
        assert q.submit(("cam1", b"", float(i), float("nan")))
    assert q.close()
    assert [s[2] for s in seen] == [0.0, 1.0, 2.0, 3.0], seen

    starved = detect_spin_starvation({
        "command": 59.79218, "ur_joint_states": 59.79089,
        "tcp_pose": 59.79282, "wrench": 59.79153,
    })
    assert starved["suspected"], starved
    healthy = detect_spin_starvation({
        "command": 98.9, "ur_joint_states": 98.9,
        "tcp_pose": 98.9, "wrench": 98.9,
    })
    assert not healthy["suspected"], healthy
    print("SELF-TEST OK")
