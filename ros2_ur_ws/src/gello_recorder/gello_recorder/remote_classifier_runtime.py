"""ROS-independent ZMQ transport helpers for remote reward classification."""

import json
import queue
import threading
import time
import uuid


PROTOCOL_VERSION = "reward-classifier-v1"
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5594"
# Mirror of reward_classifier_runtime.DEFAULT_THRESHOLD.  It is duplicated
# rather than imported so this transport module stays free of cv2/numpy, which
# the GPU-side server would otherwise need before it decides to load anything.
DEFAULT_THRESHOLD = 0.2


def oldest_pair_receipt_monotonic(cam1_received, cam2_received):
    """Receipt time used for worst-case age of a two-camera observation."""
    return min(float(cam1_received), float(cam2_received))


def encode_request(request_id, cam1_stamp_ns, cam2_stamp_ns, cam1_jpeg, cam2_jpeg,
                   threshold=DEFAULT_THRESHOLD):
    header = {
        "protocol": PROTOCOL_VERSION,
        "request_id": str(request_id),
        "cam1_stamp_ns": int(cam1_stamp_ns),
        "cam2_stamp_ns": int(cam2_stamp_ns),
        "threshold": float(threshold),
    }
    return [
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode(),
        bytes(cam1_jpeg),
        bytes(cam2_jpeg),
    ]


def decode_request(frames, max_jpeg_bytes=20_000_000):
    if len(frames) != 3:
        raise ValueError("request must contain header, cam1 JPEG, cam2 JPEG")
    header = json.loads(frames[0].decode())
    if header.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol")
    if not header.get("request_id"):
        raise ValueError("missing request_id")
    if any(not data or len(data) > max_jpeg_bytes for data in frames[1:]):
        raise ValueError("empty or oversized JPEG")
    return header, bytes(frames[1]), bytes(frames[2])


def encode_reply(**values):
    values.setdefault("protocol", PROTOCOL_VERSION)
    return json.dumps(values, separators=(",", ":"), sort_keys=True).encode()


def decode_reply(frame, expected_request_id):
    result = json.loads(frame.decode())
    if result.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("unsupported reply protocol")
    if result.get("request_id") != expected_request_id:
        raise ValueError("reply request_id mismatch")
    return result


class RemoteInferenceWorker:
    """One-in-flight REQ worker with latest-only backpressure."""

    def __init__(self, endpoint=DEFAULT_ENDPOINT, timeout_s=2.0,
                 threshold=DEFAULT_THRESHOLD):
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self.threshold = float(threshold)
        self._pending = queue.Queue(maxsize=1)
        self._results = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._socket = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def submit(self, pair):
        """Replace an unstarted request; never queue behind an in-flight request."""
        try:
            self._pending.put_nowait(pair)
            return True
        except queue.Full:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                return False
            self._pending.put_nowait(pair)
            return False

    def poll(self):
        latest = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                return latest

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.timeout_s + 0.5))

    def _connect(self):
        import zmq

        if self._socket is not None:
            self._socket.close(0)
        sock = zmq.Context.instance().socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, max(1, int(self.timeout_s * 1000)))
        sock.setsockopt(zmq.SNDTIMEO, max(1, int(self.timeout_s * 1000)))
        sock.connect(self.endpoint)
        self._socket = sock

    def _run(self):
        import zmq

        try:
            while not self._stop.is_set():
                try:
                    pair = self._pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                request_id = uuid.uuid4().hex
                started = time.monotonic()
                capture_monotonic = pair[4] if len(pair) > 4 else started
                try:
                    if self._socket is None:
                        self._connect()
                    frames = encode_request(
                        request_id, pair[1], pair[3], pair[0], pair[2],
                        self.threshold)
                    self._socket.send_multipart(frames)
                    result = decode_reply(self._socket.recv(), request_id)
                    result["roundtrip_ms"] = (
                        time.monotonic() - started) * 1000.0
                    result["capture_age_ms"] = (
                        time.monotonic() - capture_monotonic) * 1000.0
                except (zmq.ZMQError, TimeoutError, ValueError,
                        json.JSONDecodeError, OSError) as exc:
                    result = {
                        "ok": False, "request_id": request_id,
                        "message": "remote inference error: %s" % exc,
                    }
                    try:
                        self._connect()
                    except Exception:
                        self._socket = None
                self._results.put(result)
        finally:
            if self._socket is not None:
                self._socket.close(0)
