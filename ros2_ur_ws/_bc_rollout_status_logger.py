#!/usr/bin/env python3
"""Log the HIL session state timeline during a BC real-robot rollout.

The server writes one record per transition; the robot side writes raw camera
and joint streams.  Neither of them knows WHEN the session was homing, waiting
for the operator, running the policy, or being driven by a human -- that lives
only on ``/hil/actor_status`` (schema-v2 JSON) and ``/hil/deadman``.  This
script is the third leg: an append-only, receipt-timestamped JSONL of both
topics, so the other two recordings can be aligned against episode and control
ownership boundaries after the fact.

It is a PASSIVE OBSERVER.  It publishes nothing, calls no service, and owns no
safety contract -- releasing or holding the deadman is entirely the GUI's job
(``gello_hil_gui_node``), and this logger must never be read as a substitute
for it.  Consequently it also must never take the session down: a malformed
payload, a missing ``ur_gello_bringup`` overlay, or a schema the parser rejects
all degrade to ``parsed: null`` with the raw bytes retained, never to an
exception.  Raw is the archive; parsed is the convenience.

Two honest limits worth knowing before aligning anything with these lines:

* ``ts`` is the RECEIPT time in this process, not a publisher stamp.  Neither
  topic carries a stamp (``/hil/deadman`` is a bare ``Float32MultiArray`` of
  ``[engaged, gain]``, ``/hil/actor_status`` is a bare ``String``), so there is
  no better clock available on the wire.  Treat it as "the GUI/actor had said
  this by now", not "this happened exactly then".
* QoS is the frozen contract of both publishers -- default reliable, depth 10
  (``create_publisher(..., 10)`` in ``gello_hil_gui_node`` and
  ``ur_env/operator_session``).  A late-joining logger therefore gets no
  history: start it BEFORE the actor, or the first states are simply absent.

Run with SYSTEM ``python3`` under the ROS overlay.  NOT the actor's gRPC venv
(no rclpy there), and do not overwrite ``PYTHONPATH`` -- see
``docs/testing/00_SETUP_AND_SAFETY.md`` §3.4.

    set +u; source /opt/ros/humble/setup.bash
    source ros2_ur_ws/install/setup.bash; set -u
    ./ros2_ur_ws/_bc_rollout_status_logger.py --output <run_dir>

Exit 0 on a clean SIGINT/SIGTERM stop, 2 when ROS or the output directory is
unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from datetime import datetime, timezone


ACTOR_STATUS_TOPIC = "/hil/actor_status"
DEADMAN_TOPIC = "/hil/deadman"
DEFAULT_TOPICS = f"{ACTOR_STATUS_TOPIC},{DEADMAN_TOPIC}"
OUTPUT_NAME = "status.jsonl"
LOG_PREFIX = "[bc-status-logger]"

# Frozen publisher contract on both topics: default reliable QoS, depth 10.
QOS_DEPTH = 10

# Topic kinds this logger knows how to decode.  An unknown topic is refused
# rather than guessed: subscribing with the wrong message type would silently
# record nothing, which is the one failure mode a recording tool must not have.
KIND_ACTOR_STATUS = "actor_status"
KIND_DEADMAN = "deadman"
KNOWN_TOPICS = {
    ACTOR_STATUS_TOPIC: KIND_ACTOR_STATUS,
    DEADMAN_TOPIC: KIND_DEADMAN,
}


def _utc_now() -> str:
    """Receipt timestamp: UTC ISO-8601 with microseconds and an explicit zone."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def encode_deadman(data) -> tuple[str, dict | None]:
    """Compact ``(raw, parsed)`` for one ``[engaged, gain]`` heartbeat.

    The type carries no stamp, so ``parsed`` is only the two payload fields.
    A short or non-numeric payload yields ``parsed=None`` while ``raw`` keeps
    whatever arrived -- the same fail-soft rule the status parser gets.
    """

    try:
        values = [float(value) for value in data]
    except (TypeError, ValueError):
        return repr(data), None
    raw = json.dumps(values, separators=(",", ":"))
    if len(values) < 2:
        return raw, None
    engaged, gain = values[0], values[1]
    return raw, {"engaged": engaged == 1.0, "gain": gain}


def _load_status_parser():
    """Return ``parse_actor_status`` or ``None`` if the overlay is missing.

    A stale or absent ``ur_gello_bringup`` must not cost the recording: without
    the parser every status line still lands with its full raw JSON, which is
    the part that matters for later alignment.
    """

    try:
        from ur_gello_bringup.hil_actor_status import parse_actor_status
    except Exception as exc:  # pragma: no cover - depends on the ROS overlay
        print(
            f"{LOG_PREFIX} WARNING: ur_gello_bringup unavailable ({exc}); "
            f"{ACTOR_STATUS_TOPIC} lines will carry raw only",
            file=sys.stderr,
        )
        return None
    return parse_actor_status


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record /hil/actor_status and /hil/deadman to "
            f"<output>/{OUTPUT_NAME} for BC rollout alignment."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help=f"existing directory; the log is written to <dir>/{OUTPUT_NAME}",
    )
    parser.add_argument(
        "--topics",
        default=DEFAULT_TOPICS,
        help=f"comma-separated topics to record (default: {DEFAULT_TOPICS})",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.output):
        parser.error(f"--output directory does not exist: {args.output}")

    topics = [name.strip() for name in args.topics.split(",") if name.strip()]
    if not topics:
        parser.error("--topics must name at least one topic")
    unknown = [name for name in topics if name not in KNOWN_TOPICS]
    if unknown:
        parser.error(
            "unsupported topic(s): "
            + ", ".join(unknown)
            + "; known topics are "
            + ", ".join(sorted(KNOWN_TOPICS))
        )
    args.topic_list = topics
    return args


def _record(topics: list[str], path: str) -> int:
    """Spin until a signal arrives; return the process exit code."""

    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray, String
    except ImportError as exc:
        print(f"ERROR: ROS Python overlay unavailable: {exc}", file=sys.stderr)
        return 2

    parse_actor_status = _load_status_parser()

    try:
        sink = open(path, "a", encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot open {path}: {exc}", file=sys.stderr)
        return 2

    class Logger(Node):
        def __init__(self):
            super().__init__("bc_rollout_status_logger")
            self.lines = 0
            self.write_errors = 0
            for topic in topics:
                kind = KNOWN_TOPICS[topic]
                if kind == KIND_ACTOR_STATUS:
                    self.create_subscription(
                        String,
                        topic,
                        self._make_callback(topic, kind),
                        QOS_DEPTH,
                    )
                else:
                    self.create_subscription(
                        Float32MultiArray,
                        topic,
                        self._make_callback(topic, kind),
                        QOS_DEPTH,
                    )

        def _make_callback(self, topic: str, kind: str):
            def callback(message) -> None:
                self._on_message(topic, kind, message)

            return callback

        def _on_message(self, topic: str, kind: str, message) -> None:
            # Nothing below may raise: a rollout recording that dies on one bad
            # payload is worse than one with a null `parsed` field in it.
            timestamp = _utc_now()
            try:
                if kind == KIND_ACTOR_STATUS:
                    raw = message.data
                    parsed = None
                    if parse_actor_status is not None:
                        try:
                            parsed = parse_actor_status(raw)
                        except Exception:
                            parsed = None
                    if not isinstance(raw, str):
                        raw = repr(raw)
                else:
                    raw, parsed = encode_deadman(message.data)
            except Exception:
                raw, parsed = repr(message), None
            self._write(
                {"ts": timestamp, "topic": topic, "raw": raw, "parsed": parsed}
            )

        def _write(self, record: dict) -> None:
            try:
                sink.write(json.dumps(record, separators=(",", ":")) + "\n")
                # Flush every line on purpose.  /hil/deadman is 20 Hz, but the
                # lines that matter (episode boundaries, FAULT) are rare and
                # often immediately precede whatever killed the session, so
                # durability beats buffering here.
                sink.flush()
                self.lines += 1
            except Exception as exc:
                self.write_errors += 1
                if self.write_errors == 1:
                    print(
                        f"{LOG_PREFIX} WARNING: write failed ({exc}); "
                        "continuing",
                        file=sys.stderr,
                    )

    stop = {"requested": False}

    def _request_stop(signum, _frame) -> None:
        stop["requested"] = True

    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    # Installed AFTER rclpy.init so these win over its own handlers: shutdown
    # has to close the file and print the summary, not unwind through rclpy.
    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, _request_stop),
        signal.SIGTERM: signal.signal(signal.SIGTERM, _request_stop),
    }
    node = Logger()
    print(f"{LOG_PREFIX} logging to {path}", flush=True)
    print(
        f"{LOG_PREFIX} topics: {', '.join(topics)} (reliable, depth "
        f"{QOS_DEPTH})",
        flush=True,
    )
    try:
        while rclpy.ok() and not stop["requested"]:
            rclpy.spin_once(node, timeout_sec=0.1)
        return 0
    finally:
        lines = node.lines
        node.destroy_node()
        if owns_context:
            rclpy.shutdown()
        for signum, handler in previous.items():
            if handler is not None:
                signal.signal(signum, handler)
        sink.close()
        print(f"{LOG_PREFIX} stopped lines={lines} path={path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    path = os.path.join(args.output, OUTPUT_NAME)
    return _record(args.topic_list, path)


if __name__ == "__main__":
    raise SystemExit(main())
